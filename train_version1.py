import sys
import os

from models.mobilenetv3_transformer import CustomMobileNetV3
# from models.model_student_cnntrans import vgg19_trans_student
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
# from utils import cal_para
from torch.nn.modules import Module

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
from torchvision import datasets
from datasets.crowd import Crowd
from math import ceil
import logging
from datetime import datetime
import numpy as np
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='CNN-Transformer-SKT distillation')
    parser.add_argument('--model-name', default='vgg19_trans', help='the name of the model')
    parser.add_argument('--data-dir', default='../UCF_Train_Val_Test/',
                        help='training data directory')
    parser.add_argument('--save-dir', default='model',
                        help='directory to save models.')
    parser.add_argument('--save-all', type=bool, default=False,
                        help='whether to save all best model')
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=1200,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=25,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=0,
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')
    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=16,
                        help='downsample ratio')
    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.15,
                        help='background ratio')
    
    args = parser.parse_args()
    return args

def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s",
                                     "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

class Save_Handle(object):
    """handle the number of """
    def __init__(self, max_num):
        self.save_list = []
        self.max_num = max_num

    def append(self, save_path):
        if len(self.save_list) < self.max_num:
            self.save_list.append(save_path)
        else:
            remove_path = self.save_list[0]
            del self.save_list[0]
            self.save_list.append(save_path)
            if os.path.exists(remove_path):
                os.remove(remove_path)

class Bay_Loss(Module):
    def __init__(self, use_background):
        super(Bay_Loss, self).__init__()
        self.use_bg = use_background

    def forward(self, prob_list, target_list, pre_density):
        loss = 0
        for idx, prob in enumerate(prob_list):  # iterative through each sample
            if prob is None:  # image contains no annotation points
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32).cuda()
            else:
                N = len(prob)
                if self.use_bg:
                    target = torch.zeros((N,), dtype=torch.float32).cuda()
                    target[:-1] = target_list[idx]
                else:
                    target = target_list[idx]
                pre_count = torch.sum(pre_density[idx].view((1, -1)) * prob, dim=1)  # flatten into vector

            res = torch.abs(target - pre_count)
            num = ceil(0.9 * (len(res) - 1))
            loss += torch.sum(torch.topk(res[:-1], num, largest=False)[0])
            loss += res[-1]
        loss = loss / len(prob_list)
        return loss

class Post_Prob(Module):
    def __init__(self, sigma, c_size, stride, background_ratio, use_background):
        super(Post_Prob, self).__init__()
        assert c_size % stride == 0

        self.sigma = sigma
        self.bg_ratio = background_ratio
        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32).cuda() + stride / 2
        self.cood.unsqueeze_(0)
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background

    def forward(self, points, st_sizes):
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0)

        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1)
            y = all_points[:, 1].unsqueeze_(1)
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
            y_dis.unsqueeze_(2)
            x_dis.unsqueeze_(1)
            dis = y_dis + x_dis
            dis = dis.view((dis.size(0), -1))

            dis_list = torch.split(dis, num_points_per_image)
            prob_list = []
            for dis, st_size in zip(dis_list, st_sizes):
                if len(dis) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)
                        d = st_size * self.bg_ratio
                        bg_dis = (d - torch.sqrt(min_dis))**2
                        dis = torch.cat([dis, bg_dis], 0)  # concatenate background distance to the last
                    dis = -dis / (2.0 * self.sigma ** 2)
                    prob = self.softmax(dis)
                else:
                    prob = None
                prob_list.append(prob)
        else:
            prob_list = []
            for _ in range(len(points)):
                prob_list.append(None)
        return prob_list

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count

def train_epoch(train_loader, model, criterion, optimizer, post_prob, save_list, save_dir, epoch):
    losses_h = AverageMeter()
    losses_trans_cos = AverageMeter()
    epoch_loss = AverageMeter()
    epoch_mae = AverageMeter()
    epoch_mse = AverageMeter()

    model.train()
    epoch_start = time.time()

    for i, (img, points, targets, st_sizes) in enumerate(tqdm(train_loader)):
        # data_time.update(time.time() - end)
        img = img.cuda()
        img = Variable(img)
        st_sizes = st_sizes.cuda()
        gd_count = np.array([len(p) for p in points], dtype=np.float32)
        points = [p.cuda() for p in points]
        targets = [t.cuda() for t in targets]

        prob_list = post_prob(points, st_sizes)
        output, features = model(img)
        # model.distilled_features.append(output)
        loss_ct = 0
        for feature in features:
            mean_feature = torch.mean(feature, dim=0)
            mean_sum = torch.sum(mean_feature**2)**0.5
            cosine = 1 - torch.sum(feature*mean_feature, dim=1) / (mean_sum * torch.sum(feature**2, dim=1)**0.5 + 1e-5)
            loss_ct += torch.sum(cosine)
        
        loss_h = criterion(prob_list, targets, output)


        loss = loss_h + loss_ct

        losses_h.update(loss_h.item(), img.size(0))
        losses_trans_cos.update(loss_ct.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        N = img.size(0)
        pre_count = torch.sum(output.view(N, -1), dim=1).detach().cpu().numpy()
        res = pre_count - gd_count
        epoch_loss.update(loss_h.item(), N)
        epoch_mse.update(np.mean(res*res), N)
        epoch_mae.update(np.mean(abs(res)), N)
        
    logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                 .format(epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                         time.time()-epoch_start))
    model_state_dic = model.state_dict()
    save_path = os.path.join(save_dir, '{}_stu_ckpt.tar'.format(epoch))
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model_state_dic
    }, save_path)
    save_list.append(save_path)

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes

def val_epoch(loader, model, epoch):
    epoch_start = time.time()
    model.eval()  # Set model to evaluate mode
    epoch_res = []
    # Iterate over data.
    for inputs, count, name in tqdm(loader):
        inputs = inputs.cuda()
        # inputs are images with different sizes
        b, c, h, w = inputs.shape
        h, w = int(h), int(w)
        assert b == 1, 'the batch size should equal to 1 in validation mode'
        input_list = []
        if h >= 3584 or w >= 3584:
            h_stride = int(ceil(1.0 * h / 3584))
            w_stride = int(ceil(1.0 * w / 3584))
            h_step = h // h_stride
            w_step = w // w_stride
            for i in range(h_stride):
                for j in range(w_stride):
                    h_start = i * h_step
                    if i != h_stride - 1:
                        h_end = (i + 1) * h_step
                    else:
                        h_end = h
                    w_start = j * w_step
                    if j != w_stride - 1:
                        w_end = (j + 1) * w_step
                    else:
                        w_end = w
                    input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
            with torch.set_grad_enabled(False):
                pre_count = 0.0
                for idx, input in enumerate(input_list):
                    output = model(input)[0]
                    pre_count += torch.sum(output)
            res = count[0].item() - pre_count.item()
            epoch_res.append(res)
        else:
            with torch.set_grad_enabled(False):
                outputs = model(inputs)[0]
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

    epoch_res = np.array(epoch_res)
    mse = np.sqrt(np.mean(np.square(epoch_res)))
    mae = np.mean(np.abs(epoch_res))
    logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                    .format(epoch, mse, mae, time.time()-epoch_start))

    model_state_dic = model.state_dict()
    return mse, mae, model_state_dic

if __name__ == '__main__':
    args = parse_args()
    sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.save_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    setlogger(os.path.join(save_dir, 'train.log'))
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    downsample_ratio = args.downsample_ratio
    datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x) for x in ['train', 'test']}
    dataloaders = {x: DataLoader(datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=10,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'test']}
    
    model = CustomMobileNetV3()
    # model = vgg19_trans_student(trans_layer=1, direct=True)
    # cal_para(model)
    model = model.to(device)
    criterion = Bay_Loss(use_background=True)
    # optimizer = torch.optim.Adam(student.parameters(), args.lr, weight_decay=5 * 1e-4)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=5 * 1e-4)
    if os.path.exists(args.save_dir) is False:
        os.makedirs(args.out.decode('utf-8'))

    args.start_epoch = 0
    if args.resume:
        suf = args.teacher_ckpt.rsplit('.', 1)[-1]
        print("=> loading checkpoint '{}'".format(args.resume))
        if suf == 'tar':
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
        elif suf == "pth":
            model.load_state_dict(torch.load(args.resume))
    
    post_prob = Post_Prob(args.sigma, args.crop_size, args.downsample_ratio, args.background_ratio, args.use_background)
    best_mae = np.inf
    best_mse = np.inf
    save_list = Save_Handle(max_num=args.max_model_num)
    for epoch in range(args.start_epoch, args.max_epoch):
        logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
        train_epoch(dataloaders['train'], model, criterion, optimizer, post_prob, save_list, save_dir, epoch)
        if epoch % args.val_epoch == 0 and epoch >= args.val_start:
            mse, mae, model_state_dic = val_epoch(dataloaders['test'], model, epoch)
            if (2.0 * mse + mae) < (2.0 * best_mse + best_mae):
                best_mse = mse
                best_mae = mae
                logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(best_mse, best_mae, epoch))                                                                    
                torch.save(model_state_dic, os.path.join(save_dir, 'best_model.pth'))



logging.basicConfig(
    level=logging.INFO,  # 设置记录级别为INFO
    format='%(asctime)s %(levelname)s %(message)s',  # 设置日志格式
    handlers=[
        logging.FileHandler('myapp.log'),  # 添加一个FileHandler处理程序
        logging.StreamHandler()  # 添加一个StreamHandler处理程序
    ]
)



