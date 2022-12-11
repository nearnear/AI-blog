---
title: "[pytorch] ResNet 처음부터 구현하기"
categories: ML
tags:
    - pytorch
---

Residual 구조를 포함하는 합성곱 네트워크인 ResNet의 네트워크의 차원 변화를 계산하고,
PyTorch로 처음부터 구현해본다.

> ResNet 논문 : [Deep Residual Learning for Image Recognition, 2015](https://arxiv.org/abs/1512.03385)

이번 글의 목적은 아래와 같은 차원을 가지는 **ResNet**을 구현하는 것이다. 

<figure>
  <img src="/imgs/post-imgs/ml-resnet-dim.png" alt="">
  <figcaption>ResNet Layer dimensions.</figcaption>
</figure> 

표와 같은 차원을 구현하기 위해 입력 차원은 $(N, 3, 224, 224)$ 라고 가정한다. 
이때 $N$은 미니 배치의 크기다.

<figure style="width: 600px" class="align-center">
  <img src="/imgs/post-imgs/ml-bottleneck-block.png" alt="">
  <figcaption>ResNet blocks.</figcaption>
</figure> 

왼쪽은 ResNet 18, 34에서 사용되는 Residual 블럭이고, 오른쪽은 ResNet 50, 101, 152 에서 사용되는
BottleNect 블럭이다. ResNet의 특징이 되는 residual function이 구현되어 있는 것을 확인할 수 있다.


## 텐서 차원 계산하기

텐서 차원 변화에 직접 영향을 끼치는 레이어는 `nn.Conv2d`, `nn.MaxPool2d`, `nn.AvgPool2d`,
`nn.tensor.view`, `nn.Linear` 이고 `nn.BatchNorm2d`, `nn.ReLU` 는 영향을 끼치지 않는다는 
점에 유의하자. 
{: .notice}

ResNet 34와 ResNet 50의 텐서 차원 변화를 계산하면 다음과 같다.

<figure style="width: 500px" class="align-center">
  <img src="/imgs/post-imgs/ml-resnet34-50-dim.png" alt="">
  <figcaption>Left: Tensor size of ResNet 34, Right: ResNet 50.</figcaption>
</figure> 

두 네트워크의 차이점은 building block이 상이하므로 전체 레이어 수에 차이가 난다는 점과 (각각 34개, 50개이다)
expansion의 차이로 인해 (각각 1과 4의 값을 가진다) 두번째 텐서 차원에 차이가 난다는 점이다.

<figure style="width: 300px" class="align-center">
  <img src="/imgs/post-imgs/ml-resnet-conv3_x.png" alt="">
  <figcaption>conv3_x block from ResNet 50.</figcaption>
</figure> 

ResNet 50의 conv3_x 블럭을 들여다보면, 첫번째 BottleNeck 레이어를 거치면 stride=2로 인해
(세번째와 네번째) 결과 차원은 두배로 줄어들고, expansion이 실행되어 두번째 차원은 늘어나는 것을 확인할 수 있다. 


## 구현하기

### Basic block

우선 편의를 위해 Conv 와 Batch normalization, activation을 합친 기본 블럭을 작성한다.

```python
# Basic Convolutional building block.
# Performs : Convolution -> Batch Normalization -> ReLU
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=True, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        if not self.activation:
            return self.batch_norm(self.conv(x))
        return self.relu(self.batch_norm(self.conv(x)))
```

이때 activation을 제한할 수 있도록 했다.

### Building block

ResNet의 building block을 위에서 본 그림에 따라 두 종류로 구현한다.
앞서 정의한 기본 블럭을 활용했다.

```python
# Building block for ResNet 18, 34
# kwargs=(kernel_size=3, stride=1, padding=1, ..)
class Residual(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual = nn.Sequential(
            BasicBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            BasicBlock(out_channels, out_channels * Residual.expansion, activation=False, 
                      kernel_size=3, stride=1, padding=1)
        )
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        # projection mapping using 1x1 convolution
        if stride != 1 or in_channels != Residual.expansion * out_channels:
            self.shortcut = BasicBlock(in_channels, out_channels * Residual.expansion,
                                      kernel_size=1, stride=stride)


    def forward(self, x):
        x = self.residual(x) + self.shortcut(x)
        x = self.relu(x)
        return x
      

# Building block for ResNet 50, 101, 105
class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, **kwargs): 
        super().__init__()

        self.residual = nn.Sequential(
            BasicBlock(in_channels, out_channels, kernel_size=1, stride=1, **kwargs),
            BasicBlock(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, **kwargs),
            BasicBlock(out_channels, out_channels * self.expansion, activation=False, 
                       kernel_size=1, stride=1, **kwargs)
        )
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = BasicBlock(in_channels, out_channels * BottleNeck.expansion,
                                      kernel_size=1, stride=stride)


    def forward(self, x):
        x = self.residual(x) + self.shortcut(x)
        x = self.relu(x)
        return x
```

ResNet에서는 Residual 또는 BottleNeck 블럭의 첫번째 레이어에서 shortcut 연결을 사용한다.
때문에 building block으로 네트워크를 구축할 때 stride를 다르게 입력하는 것이 필요하다 (그래서 stride를 인수로 
지정하였다).

### Network

```python
# Whole ResNet Network.
# Performs : ResNet Network 
#            given a building block and a list containing numbers of blocks.
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = BasicBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self.conv_seq(block, 64, num_blocks[0], 1)
        self.conv3_x = self.conv_seq(block, 128, num_blocks[1], 2)
        self.conv4_x = self.conv_seq(block, 256, num_blocks[2], 2)
        self.conv5_x = self.conv_seq(block, 512, num_blocks[3], 2)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def conv_seq(self, block, out_channels, num_layers, stride):
        strides = [stride] + [1] * (num_layers - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
```

네트워크를 정의할 때 `conv_seq`에 의해 out_channels는 

이때 ResNet50의 forward propagation은 다음과 같이 정의한다. 

```python
    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
```

처음 등장한 표에 나온 것과 정확히 같은 과정을 거친다.


## 구조 확인하기

이제, 설정한 모델 구조가 처음의 표와 같이 의도한 차원을 가지는지 확인해볼 수 있다.

```python
def resnet_50():
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=10)

# input_shape  : (4, 3, 224, 224)
# output_shape : (4, 10)
# Shows        : ResNet50 structure with random input
def show_resnet50(**kwargs):
    x = torch.randn(4, 3, 224, 224).to(device)
    model = resnet_50().to(device)
    assert model(x).shape == torch.Size([4, 10])
    print(summary(model, (3, 224, 224), batch_size=4))
    return model
```

모델 구조에 대한 요약은 다음과 같았다.

```python
show_resnet50()
```
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [4, 64, 112, 112]           9,472
       BatchNorm2d-2          [4, 64, 112, 112]             128
              ReLU-3          [4, 64, 112, 112]               0
        BasicBlock-4          [4, 64, 112, 112]               0
         MaxPool2d-5            [4, 64, 56, 56]               0
            Conv2d-6            [4, 64, 56, 56]           4,160
       BatchNorm2d-7            [4, 64, 56, 56]             128
              ReLU-8            [4, 64, 56, 56]               0
        BasicBlock-9            [4, 64, 56, 56]               0
           Conv2d-10            [4, 64, 56, 56]          36,928
      BatchNorm2d-11            [4, 64, 56, 56]             128
             ReLU-12            [4, 64, 56, 56]               0
       BasicBlock-13            [4, 64, 56, 56]               0
           Conv2d-14           [4, 256, 56, 56]          16,640
      BatchNorm2d-15           [4, 256, 56, 56]             512
       BasicBlock-16           [4, 256, 56, 56]               0
           Conv2d-17           [4, 256, 56, 56]          16,640
      BatchNorm2d-18           [4, 256, 56, 56]             512
             ReLU-19           [4, 256, 56, 56]               0
       BasicBlock-20           [4, 256, 56, 56]               0
             ReLU-21           [4, 256, 56, 56]               0
       BottleNeck-22           [4, 256, 56, 56]               0
           Conv2d-23            [4, 64, 56, 56]          16,448
      BatchNorm2d-24            [4, 64, 56, 56]             128
             ReLU-25            [4, 64, 56, 56]               0
       BasicBlock-26            [4, 64, 56, 56]               0
           Conv2d-27            [4, 64, 56, 56]          36,928
      BatchNorm2d-28            [4, 64, 56, 56]             128
             ReLU-29            [4, 64, 56, 56]               0
       BasicBlock-30            [4, 64, 56, 56]               0
           Conv2d-31           [4, 256, 56, 56]          16,640
      BatchNorm2d-32           [4, 256, 56, 56]             512
       BasicBlock-33           [4, 256, 56, 56]               0
             ReLU-34           [4, 256, 56, 56]               0
       BottleNeck-35           [4, 256, 56, 56]               0
           Conv2d-36            [4, 64, 56, 56]          16,448
      BatchNorm2d-37            [4, 64, 56, 56]             128
             ReLU-38            [4, 64, 56, 56]               0
       BasicBlock-39            [4, 64, 56, 56]               0
           Conv2d-40            [4, 64, 56, 56]          36,928
      BatchNorm2d-41            [4, 64, 56, 56]             128
             ReLU-42            [4, 64, 56, 56]               0
       BasicBlock-43            [4, 64, 56, 56]               0
           Conv2d-44           [4, 256, 56, 56]          16,640
      BatchNorm2d-45           [4, 256, 56, 56]             512
       BasicBlock-46           [4, 256, 56, 56]               0
             ReLU-47           [4, 256, 56, 56]               0
       BottleNeck-48           [4, 256, 56, 56]               0
           Conv2d-49           [4, 128, 56, 56]          32,896
      BatchNorm2d-50           [4, 128, 56, 56]             256
             ReLU-51           [4, 128, 56, 56]               0
       BasicBlock-52           [4, 128, 56, 56]               0
           Conv2d-53           [4, 128, 28, 28]         147,584
      BatchNorm2d-54           [4, 128, 28, 28]             256
             ReLU-55           [4, 128, 28, 28]               0
       BasicBlock-56           [4, 128, 28, 28]               0
           Conv2d-57           [4, 512, 28, 28]          66,048
      BatchNorm2d-58           [4, 512, 28, 28]           1,024
       BasicBlock-59           [4, 512, 28, 28]               0
           Conv2d-60           [4, 512, 28, 28]         131,584
      BatchNorm2d-61           [4, 512, 28, 28]           1,024
             ReLU-62           [4, 512, 28, 28]               0
       BasicBlock-63           [4, 512, 28, 28]               0
             ReLU-64           [4, 512, 28, 28]               0
       BottleNeck-65           [4, 512, 28, 28]               0
           Conv2d-66           [4, 128, 28, 28]          65,664
      BatchNorm2d-67           [4, 128, 28, 28]             256
             ReLU-68           [4, 128, 28, 28]               0
       BasicBlock-69           [4, 128, 28, 28]               0
           Conv2d-70           [4, 128, 28, 28]         147,584
      BatchNorm2d-71           [4, 128, 28, 28]             256
             ReLU-72           [4, 128, 28, 28]               0
       BasicBlock-73           [4, 128, 28, 28]               0
           Conv2d-74           [4, 512, 28, 28]          66,048
      BatchNorm2d-75           [4, 512, 28, 28]           1,024
       BasicBlock-76           [4, 512, 28, 28]               0
             ReLU-77           [4, 512, 28, 28]               0
       BottleNeck-78           [4, 512, 28, 28]               0
           Conv2d-79           [4, 128, 28, 28]          65,664
      BatchNorm2d-80           [4, 128, 28, 28]             256
             ReLU-81           [4, 128, 28, 28]               0
       BasicBlock-82           [4, 128, 28, 28]               0
           Conv2d-83           [4, 128, 28, 28]         147,584
      BatchNorm2d-84           [4, 128, 28, 28]             256
             ReLU-85           [4, 128, 28, 28]               0
       BasicBlock-86           [4, 128, 28, 28]               0
           Conv2d-87           [4, 512, 28, 28]          66,048
      BatchNorm2d-88           [4, 512, 28, 28]           1,024
       BasicBlock-89           [4, 512, 28, 28]               0
             ReLU-90           [4, 512, 28, 28]               0
       BottleNeck-91           [4, 512, 28, 28]               0
           Conv2d-92           [4, 128, 28, 28]          65,664
      BatchNorm2d-93           [4, 128, 28, 28]             256
             ReLU-94           [4, 128, 28, 28]               0
       BasicBlock-95           [4, 128, 28, 28]               0
           Conv2d-96           [4, 128, 28, 28]         147,584
      BatchNorm2d-97           [4, 128, 28, 28]             256
             ReLU-98           [4, 128, 28, 28]               0
       BasicBlock-99           [4, 128, 28, 28]               0
          Conv2d-100           [4, 512, 28, 28]          66,048
     BatchNorm2d-101           [4, 512, 28, 28]           1,024
      BasicBlock-102           [4, 512, 28, 28]               0
            ReLU-103           [4, 512, 28, 28]               0
      BottleNeck-104           [4, 512, 28, 28]               0
          Conv2d-105           [4, 256, 28, 28]         131,328
     BatchNorm2d-106           [4, 256, 28, 28]             512
            ReLU-107           [4, 256, 28, 28]               0
      BasicBlock-108           [4, 256, 28, 28]               0
          Conv2d-109           [4, 256, 14, 14]         590,080
     BatchNorm2d-110           [4, 256, 14, 14]             512
            ReLU-111           [4, 256, 14, 14]               0
      BasicBlock-112           [4, 256, 14, 14]               0
          Conv2d-113          [4, 1024, 14, 14]         263,168
     BatchNorm2d-114          [4, 1024, 14, 14]           2,048
      BasicBlock-115          [4, 1024, 14, 14]               0
          Conv2d-116          [4, 1024, 14, 14]         525,312
     BatchNorm2d-117          [4, 1024, 14, 14]           2,048
            ReLU-118          [4, 1024, 14, 14]               0
      BasicBlock-119          [4, 1024, 14, 14]               0
            ReLU-120          [4, 1024, 14, 14]               0
      BottleNeck-121          [4, 1024, 14, 14]               0
          Conv2d-122           [4, 256, 14, 14]         262,400
     BatchNorm2d-123           [4, 256, 14, 14]             512
            ReLU-124           [4, 256, 14, 14]               0
      BasicBlock-125           [4, 256, 14, 14]               0
          Conv2d-126           [4, 256, 14, 14]         590,080
     BatchNorm2d-127           [4, 256, 14, 14]             512
            ReLU-128           [4, 256, 14, 14]               0
      BasicBlock-129           [4, 256, 14, 14]               0
          Conv2d-130          [4, 1024, 14, 14]         263,168
     BatchNorm2d-131          [4, 1024, 14, 14]           2,048
      BasicBlock-132          [4, 1024, 14, 14]               0
            ReLU-133          [4, 1024, 14, 14]               0
      BottleNeck-134          [4, 1024, 14, 14]               0
          Conv2d-135           [4, 256, 14, 14]         262,400
     BatchNorm2d-136           [4, 256, 14, 14]             512
            ReLU-137           [4, 256, 14, 14]               0
      BasicBlock-138           [4, 256, 14, 14]               0
          Conv2d-139           [4, 256, 14, 14]         590,080
     BatchNorm2d-140           [4, 256, 14, 14]             512
            ReLU-141           [4, 256, 14, 14]               0
      BasicBlock-142           [4, 256, 14, 14]               0
          Conv2d-143          [4, 1024, 14, 14]         263,168
     BatchNorm2d-144          [4, 1024, 14, 14]           2,048
      BasicBlock-145          [4, 1024, 14, 14]               0
            ReLU-146          [4, 1024, 14, 14]               0
      BottleNeck-147          [4, 1024, 14, 14]               0
          Conv2d-148           [4, 256, 14, 14]         262,400
     BatchNorm2d-149           [4, 256, 14, 14]             512
            ReLU-150           [4, 256, 14, 14]               0
      BasicBlock-151           [4, 256, 14, 14]               0
          Conv2d-152           [4, 256, 14, 14]         590,080
     BatchNorm2d-153           [4, 256, 14, 14]             512
            ReLU-154           [4, 256, 14, 14]               0
      BasicBlock-155           [4, 256, 14, 14]               0
          Conv2d-156          [4, 1024, 14, 14]         263,168
     BatchNorm2d-157          [4, 1024, 14, 14]           2,048
      BasicBlock-158          [4, 1024, 14, 14]               0
            ReLU-159          [4, 1024, 14, 14]               0
      BottleNeck-160          [4, 1024, 14, 14]               0
          Conv2d-161           [4, 256, 14, 14]         262,400
     BatchNorm2d-162           [4, 256, 14, 14]             512
            ReLU-163           [4, 256, 14, 14]               0
      BasicBlock-164           [4, 256, 14, 14]               0
          Conv2d-165           [4, 256, 14, 14]         590,080
     BatchNorm2d-166           [4, 256, 14, 14]             512
            ReLU-167           [4, 256, 14, 14]               0
      BasicBlock-168           [4, 256, 14, 14]               0
          Conv2d-169          [4, 1024, 14, 14]         263,168
     BatchNorm2d-170          [4, 1024, 14, 14]           2,048
      BasicBlock-171          [4, 1024, 14, 14]               0
            ReLU-172          [4, 1024, 14, 14]               0
      BottleNeck-173          [4, 1024, 14, 14]               0
          Conv2d-174           [4, 256, 14, 14]         262,400
     BatchNorm2d-175           [4, 256, 14, 14]             512
            ReLU-176           [4, 256, 14, 14]               0
      BasicBlock-177           [4, 256, 14, 14]               0
          Conv2d-178           [4, 256, 14, 14]         590,080
     BatchNorm2d-179           [4, 256, 14, 14]             512
            ReLU-180           [4, 256, 14, 14]               0
      BasicBlock-181           [4, 256, 14, 14]               0
          Conv2d-182          [4, 1024, 14, 14]         263,168
     BatchNorm2d-183          [4, 1024, 14, 14]           2,048
      BasicBlock-184          [4, 1024, 14, 14]               0
            ReLU-185          [4, 1024, 14, 14]               0
      BottleNeck-186          [4, 1024, 14, 14]               0
          Conv2d-187           [4, 512, 14, 14]         524,800
     BatchNorm2d-188           [4, 512, 14, 14]           1,024
            ReLU-189           [4, 512, 14, 14]               0
      BasicBlock-190           [4, 512, 14, 14]               0
          Conv2d-191             [4, 512, 7, 7]       2,359,808
     BatchNorm2d-192             [4, 512, 7, 7]           1,024
            ReLU-193             [4, 512, 7, 7]               0
      BasicBlock-194             [4, 512, 7, 7]               0
          Conv2d-195            [4, 2048, 7, 7]       1,050,624
     BatchNorm2d-196            [4, 2048, 7, 7]           4,096
      BasicBlock-197            [4, 2048, 7, 7]               0
          Conv2d-198            [4, 2048, 7, 7]       2,099,200
     BatchNorm2d-199            [4, 2048, 7, 7]           4,096
            ReLU-200            [4, 2048, 7, 7]               0
      BasicBlock-201            [4, 2048, 7, 7]               0
            ReLU-202            [4, 2048, 7, 7]               0
      BottleNeck-203            [4, 2048, 7, 7]               0
          Conv2d-204             [4, 512, 7, 7]       1,049,088
     BatchNorm2d-205             [4, 512, 7, 7]           1,024
            ReLU-206             [4, 512, 7, 7]               0
      BasicBlock-207             [4, 512, 7, 7]               0
          Conv2d-208             [4, 512, 7, 7]       2,359,808
     BatchNorm2d-209             [4, 512, 7, 7]           1,024
            ReLU-210             [4, 512, 7, 7]               0
      BasicBlock-211             [4, 512, 7, 7]               0
          Conv2d-212            [4, 2048, 7, 7]       1,050,624
     BatchNorm2d-213            [4, 2048, 7, 7]           4,096
      BasicBlock-214            [4, 2048, 7, 7]               0
            ReLU-215            [4, 2048, 7, 7]               0
      BottleNeck-216            [4, 2048, 7, 7]               0
          Conv2d-217             [4, 512, 7, 7]       1,049,088
     BatchNorm2d-218             [4, 512, 7, 7]           1,024
            ReLU-219             [4, 512, 7, 7]               0
      BasicBlock-220             [4, 512, 7, 7]               0
          Conv2d-221             [4, 512, 7, 7]       2,359,808
     BatchNorm2d-222             [4, 512, 7, 7]           1,024
            ReLU-223             [4, 512, 7, 7]               0
      BasicBlock-224             [4, 512, 7, 7]               0
          Conv2d-225            [4, 2048, 7, 7]       1,050,624
     BatchNorm2d-226            [4, 2048, 7, 7]           4,096
      BasicBlock-227            [4, 2048, 7, 7]               0
            ReLU-228            [4, 2048, 7, 7]               0
      BottleNeck-229            [4, 2048, 7, 7]               0
       AvgPool2d-230            [4, 2048, 1, 1]               0
          Linear-231                    [4, 10]          20,490
================================================================
Total params: 23,555,082
Trainable params: 23,555,082
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 2.30
Forward/backward pass size (MB): 1531.31
Params size (MB): 89.86
Estimated Total Size (MB): 1623.47
----------------------------------------------------------------
```

조금 길지만, 위에서 계산한 것과 같은 텐서 차원 변화를 거치는 것을 확인할 수 있다.


이렇게 ResNet을 처음부터 구현해 보았다. Residual 연결은 이후의 네트워크에서도 
regularization을 수행하는 구조로 일반적으로 활용되는 것같다. 이미지 데이터에 대해 유독 차원이 
헷갈리는 경우가 있어 시작했는데, 모델을 자세히 들여다 보고 공부하기 좋은 방법이었다.


### 참고자료
1. [Deep Residual Learning for Image Recognition, 2015](https://arxiv.org/abs/1512.03385)
2. https://pytorch.org/hub/pytorch_vision_resnet/
3. https://velog.io/@gibonki77/ResNetwithPyTorch
4. https://deep-learning-study.tistory.com/534