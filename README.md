# aiaa-5032
crowd counting based no SKT and boosting

for mobienetv3 size test, input code as follows:
```ruby
import torch

input_tensor = torch.randn(1, 3, 224, 224)
net_small = CustomMobileNetV3()
output = net_small(input_tensor)

# print
for i, feature_map in enumerate(net_small.extractions):
    print(f"Feature map {i+1}: Shape {feature_map.shape}")

```

