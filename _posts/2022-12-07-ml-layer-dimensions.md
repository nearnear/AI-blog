---
title: "[pytorch] 텐서 차원 구하기"
categories: ML
tags:
    - multi-modal
    - pytorch
---


PyTorch의 레이어가 텐서 차원을 어떻게 변화시키는지 구해본다.

## 공식

다음의 레이어들이 텐서 차원을 변화시키는 공식을 알아본다.
- torch.nn.Linear
- torch.nn.Conv2d
- torch.nn.MaxPool2d

이후 글에서 $N$은 모두 미니 배치의 크기를 나타낸다.

### [torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=nn+linear#torch.nn.Linear)

> torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)

$$
(N, D_\text{in})
\longrightarrow 
(N, D_\text{out})
$$

Linear 레이어는 Fully connected 레이어로 불리기도 하며, $y = x \cdot A^{T} + b$의 선형 변환을 
수행한다. 이때 $A$는 가중치 학습 행렬이고 $b$는 bias이다. 

### [torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d)

> torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

합성곱 레이어는 커널(kennel) 또는 필터(filter) 행렬과 대상 행렬을 비교해 특성 맵(feature map)을 
얻는 연산이다. 커널과 대상 행렬의 커널 크기 부분 행렬의 원소별 값을 곱해 모두 더한다. 
이때 스트라이드(stride)는 커널이 이동하는 크기를, 패딩(padding)은 차원을 맞추기 위해 대상 행렬의 외부에 null 값으로 
추가하는 차원의 수를 의미한다.

$$
(N, C_\text{in}, H, W)
\longrightarrow 
(N, C_\text{out}, H_\text{out}, W_\text{out})
$$

즉 입력값과 출력값을 비교했을 때 채널 수 $C$와 $H$, $W$ 값만 변화 가능성이 있고, 
배치 수 $N$은 변화시키지 않음을 알 수 있다. 

$$
H_{out} = \lfloor \frac{H_{in} + 2*\text{padding} - \text{dilation}*
(\text{kernel size}-1) - 1}{\text{stride}} + 1  \rfloor
$$

예를 들어 `nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)` 레이어가 있다면, 
공식에 의해 $H_{out} = H_{in}$ 이다.

**Note**: 픽셀의 넓이 $W$에 대해서도 같은 공식이 적용된다. 대체로 정사각형 이미지에 대해 적용하므로 $H=W$ 이지만,
각각 다른 값을 적용할 수도 있다.
{: .notice}

### [torch.nn.MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=nn+maxpool2d#torch.nn.MaxPool2d)

> torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

$$
(N, C, H, W)
\longrightarrow
(N, C, H_\text{out}, W_\text{out})
$$

즉 MaxPool2D 레이어는 2D 픽셀의 높이 $H$와 넓이 $W$만 변화시킨다. 

$$
H_{out} = \lfloor \frac{H_{in} + 2*\text{padding} - \text{dilation}*
(\text{kernel size}-1) - 1}{\text{stride}} + 1  \rfloor
$$

예를 들어 `nn.MaxPool2d(kernel_size=2, stride=2)`일 때 차원 변화는 다음과 같다. 

$$
H_\text{out} = \frac{H_\text{in} + 2*0 - 1*1 - 1}{2} + 1 = H_\text{in} / 2
$$

즉 MaxPool2D 레이어에 의해 입력 픽셀 높이와 너비가 두배 줄어든다.

**Note**: torch.nn.AvgPool2d도 같은 Pooling이기 때문에 연산이 다를 뿐, 차원에 대해서는 같은 작업을 수행한다.
{: .notice--info}

## 예시

아래와 같이 정의된 모델의 차원 변화를 구해보자.

아래의 코드는 단일 이미지와 자연어 문단이 주어졌을 때 이를 분류하는 멀티 모달 네트워크이다. 
이미지 크기는 `128` 정방형으로 변환되어 있으며, 자연어는 `4096` 차원으로 벡터화되어 있다고 가정한다.

```python
class CustomModel(nn.Module):
    def __init__(self, num_classes=len(le.classes_)):
        super(CustomModel, self).__init__()
        # Image
        self.cnn_extract = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Text
        self.nlp_extract = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(4160, num_classes)
        )
            

    def forward(self, img, text):
        img_feature = self.cnn_extract(img)
        img_feature = torch.flatten(img_feature, start_dim=1)
        text_feature = self.nlp_extract(text)
        feature = torch.cat([img_feature, text_feature], axis=1)
        output = self.classifier(feature)
        return output
```

**Code from** https://dacon.io/competitions/official/235978/codeshare/6565?
page=1&dtype=recent
{: .notice--info}

결과부터 표현하면 각 레이어를 거치는 텐서의 차원 변화는 다음과 같다.

<figure style="width: 300px" class="align-center">
  <img src="/imgs/post-imgs/ml-dimensions.png" alt="">
  <figcaption>Tensor dimensions of the network.</figcaption>
</figure> 

가장 먼저 activation 함수인 ReLU는 각 원소에 대한 연산일 뿐, 텐서 차원을 변화시키지 않는다는 점에 유의하자.

Conv2d 레이어로 인한 변화를 살펴보면, `nn.Conv2d(3, 8, kernel_size=3, stride=1, 
padding=1)`에서 커널 크기가 3이고, 패딩이 1이며 스트라이드도 1이므로 픽셀 수인 H와 W는 그대로 
유지된다. 그에 반해 채널 수는 두번째 인수인 $C_{out}$으로 변화하므로, 입력 텐서 차원이 $(N, 3, 128, 128)$일 때,
결과 텐서 차원은 $(N, 8, 128, 128)$이다.

반면 MaxPool2d 레이어는 오직 픽셀 크기 H와 W만 변화시키는데, 위에서 살펴본 것과 같은 설정값을 가지므로
입력 텐서 차원이 $(N, 8, 128, 128)$일 때, 결과 텐서 차원은 $(N, 8, 64, 64)$이다.

자연어를 벡터로 변환한 후, Linear 레이어는 미니 배치 $N$을 제외하고 인수로 받은 차원을 가진다.

이미지를 나타내는 텐서와 자연어를 나타내는 텐서를 하나의 텐서로 축적(concatenate)하기 위해 
이미지 텐서를 `torch.flatten`을 통해 N개의 1차원 벡터로 변화시킨다. 이때 $(N, 64, 7, 7)$ 이었던 
텐서는 $N$을 제외한 값을 모두 곱해 $(N, 3136)$ 차원이 된다. 그 후 `torch.cat`을 통해 이미지 텐서와
자연어 텐서를 합치면 두 텐서의 차원을 더해 $(N,4160)$ 차원이 된다.

분류 네트워크인 `self.classifier`는 이 벡터를 입력값으로 받아 `num_classes`로 분류한다. 
전체 네트워크의 최종 차원은 $(N, \text{num_classes})$이다.


## 참고 자료
1. PyTorch Documentation : https://pytorch.org/docs
2. Dacon Baseline code : https://dacon.io/competitions/official/235978/codeshare/6565?page=1&dtype=recent