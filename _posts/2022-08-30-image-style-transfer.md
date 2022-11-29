---
title: "Image Style Transfer, 2016"
categories: Papers
tags:
  - vision
---

> Paper : [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

깊은 합성곱 신경망을 이용해 이미지의 내용과 스타일을 분리하고 결합하는 방법을 소개한다. 


## 1. 들어가며

스타일을 이미지에 렌더링하는 것은 이미지의 질감을 바꾸는 것으로 이해할 수 있다. 

**Notation**: texture transformation은 질감 변화로, deep image representation은 깊은 
이미지 표현식으로, content representation은 내용 표현식으로, style representation은 스타일 표현식으로,
style transfer는 스타일 전이로 번역하였다.
{: .notice}

### 질감 변화에 대한 사전 연구

- 질감 변화의 목표는 이미지의 의미 정보를 보존하기 위해 제한적으로 질감을 합성하는 것이다. 
- 질감 합성은 그동안 파라미터를 사용하지 않고 이미지의 픽셀을 재구성하는 방식으로 적용되었다. 즉 변형 후에도 이미지의 구조는 유지되었다.   
- 질감 변화의 사전 연구들은 이미지의 저수준(개체보다 픽셀에 가까운) 성질 만을 활용해왔다. 

### 깊은 합성곱 신경망으로 생긴 가능성

- 깊은 합성곱 신경망은 고수준(광범위한, 개체 인식에 가까운) 성질을 추출할 수 있음이 증명되었다. 즉 이미지의 고수준 성질을 의미론적 정보(semantics)로 활용할 수 있다. 
- 특히 개체 인식 작업은 다양한 조명하에서의 얼굴 사진(DeepFace)이나 다양한 사람이 쓴 손글씨 이미지, 또 보다 일반적인 개체에 대해 효과성을 입증해왔다.

이 논문에서 보이는 방법은 합성곱을 이용한 질감 합성이라고 할 수 있다. 깊은 합성곱 신경망을 통한 파라미터 질감 모델과 표현식을 도치(invert)(Mahendran, Vedaldi, 2014)해서 이미지를 생성하는 방식을 활용한다.


## 2. 깊은 이미지 표현식

<figure>
	<a href="/imgs/post-imgs/style-transfer-image-representation.png"><img src="/imgs/post-imgs/style-transfer-image-representation.png"></a>
	<figcaption>Image representation.</figcaption>
</figure>

- 16개의 합성곱과 5개의 풀링 층으로 구성된 VGG 네트워크를 활용했다.
- Content Reconstruction에서는 사전 학습된 VGG 네트워크의 레이어를 활용해 이미지를 재구축했다. 초기 레이어는 원본을 그대로 반영하지만 네트워크가 진행될 수록 픽셀 정보는 손실되고 전체 구성은 유지되는 모습을 보였다. 반면 Style Reconstruction은 질감 특성을 분류하는 피쳐 맵을 통해 이미지를 구축해, 전체 구성은 잃고 픽셀 정보를 남기도록 하였다. 
- 풀링을 평균으로 진행하는(average pooling) 것이 맥시멈 풀링보다 조금 더 나은 결과를 도출했다.


### 2.1. 내용 표현식

초기의 레이어는 주로 픽셀 값을 반영하며 재구성(reconstruction)에 영향을 받는데 반해, 네트워크의 후반부에 위치한 레이어일 수록 고수준 *내용* 즉 개체와 그 위치를 반영한다. 따라서 보다 높은 레벨의 레이어들이 표현하는 특성를 *내용 표현식(content representation)* 이라고 부르기로 했다. 

입력 이미지 $x$ 는 각 레이어의 필터 반응으로 인코딩된다. 

레이어 $l$ 에 대해:
- $N_l$ : 서로 다른 필터 개수
- $M_l$ : 각 피쳐맵의 크기(피쳐맵의 높이와 넓이를 곱한 값)
- $F_{ij}^{l}$ : 위치 $j$ 에서의 $i$ 번째 필터의 활성화 함수

레이어 $l$ 에서의 표현식은:

$$
F^l \in \mathcal{R}^{N_l \times M_l}
$$

#### 레이어 시각화하기

각 레이어에서 이미지 정보를 시각화하는 방법은, 화이트 노이즈 이미지에 대해 그래디언트 디센트를 실행 해 원본 이미지의 특성 반응과 일치하는 다른 이미지를 찾는 것이다. 

- $p$, $P^l$ : 원본 이미지와 레이어 $l$ 에서의 그 표현식
- $x$, $F^l$ : 생성된 이미지와 레이어 $l$ 에서의 **내용** 표현식

일 때, 다음과 같이 정의한다.

- 두 특성 표현식에 대한 손실은 **평균 제곱 오차**이다:

$$ 
\mathcal{L}_{content}^{p, x, l} = \frac{1}{2} \sum_{i, j} (F_{ij}^{l} - P_{ij}^{l})^2
$$

역전파(back-propagation)를 위한 이 손실의 미분값은:

$$
\frac{\partial \mathcal{L}_{content}}{\partial \mathcal{F}_{ij}^{l}} = 
\left\lbrace
\begin{aligned} 
& (F^l - P^l)_{ij} & if \ \ \mathcal{F}_{ij}^{l} > 0 \\ 
& 0 & otherwise 
\end{aligned}\right.
$$

이다. 즉 이미지 $x$ 에 대한 그래디언트는 표준 오차를 통해 역전파로 계산된다. 즉 처음의 랜덤 (노이즈) 이미지 $x$ 를 합성곱 네트워크의 특정 레이어에서 원본 이미지 $p$ 와 같은 반응을 보일 때까지 변형한다. (즉 목표는 $F^l \approx P^l$ 이다.)


### 2.2. 스타일 표현식

이미지에서 스타일을 추출하는 데에는 질감 정보를 추출하는 특성 공간을 활용한다. 이 특성 공간은 서로 다른 필터 반응의 상관성(correlation), 즉 그람 행렬(Gram Matrix) $G^l \in \mathcal{R}^{N_l \times N_l}$ 로 정의한다. 이때 $G_{ij}^{l}$ 은 레이어 $l$에서 특성 맵 $i$ 와 $j$ 의 내적이다.

$$
G_{ij}^{l} = \sum_{k} F_{ik}^{l} F_{kj}^{l}
$$

#### 레이어 시각화하기

특성 상관성을 통해 질감 정보를 포함하고 전체적인(global) 배열은 포함하지 않는 입력 이미지의 고정된 멀티-스케일 표현식을 얻을 수 있다. 이 또한 화이트 노이즈 이미지에서 시작해, 그람 행렬들 간의 평균제곱 거리를 최소화하는 방법으로 그래디언트 디센트를 진행하는 것으로 시각화 할 수 있다.

- $a$, $A^l$ : 원본 이미지와 레이어 $l$ 에서의 표현식
- $x$, $G^l$ : 생성한 이미지와 레이어 $l$ 에서의 **스타일** 표현식

에 대해 다음과 같이 정의할 수 있다.

- 레이어 $l$ 이 전체 손실에 미치는 영향:

$$
E_l = \frac{1}{4N_l^2 M_l^2} \sum_{i, j} (G_{ij}^{l} - A_{ij}^{l})^2
$$

- 전체 스타일 손실 :

$$
\mathcal{L}_{style}^{(a, x)} = \sum_l w_l E_l
$$

즉 전체 스타일 손실은 가중치 $w_l$ 을 가지는 각 레이어 손실의 선형 결합으로 정의한다. 따라서 활성화 함수에 대한 $E_l$ 의 미분값은 아래와 같이 정의된다.

$$
\frac{\partial E_l}{\partial F_{ij}^l} = 
\left\lbrace
\begin{aligned} 
& \frac{1}{N_l^2 M_l^2} ((F^l)^T (G^l - P^l))_{ji} & if \ \ \mathcal{F}_{ij}^{l} > 0 \\
& 0 & otherwise
\end{aligned}\right.
$$


### 2.3 스타일 전이하기

<figure>
	<a href="/imgs/post-imgs/style-transfer-algorithm.png"><img src="/imgs/post-imgs/style-transfer-algorithm.png"></a>
	<figcaption>Style Transfer Algorithm.</figcaption>
</figure>

아이디어는 학습한 내용 표현 벡터 $p$ 와 스타일 표현 벡터 $a$ 에 동시에 부합하는 새로운 이미지를 합성하는 것이다. 즉 내용 표현식과 스타일 표현식의 거리를 최소화하는 것을 목표로 학습한다. 전체 손실 함수는 **각 표현식 오차의 선형 결합**이다:

$$
\mathcal{L}_{total} {(p, a, x)} = \alpha \mathcal{L}_{content} (p, x) + \beta \mathcal{L}_{style}(a, x)
$$

$\alpha$ 와 $\beta$ 는 각각 내용과 스타일 재구성의 가중치이다. 스타일 전이를 위해 형성된 이미지 $x$ 에 대한 손실의 미분값을 최적화에 사용할 수 있다 :

$$
\frac{\partial \mathcal{L}_{total}^{(p, a, x)}}{\partial F_{ij}^l} = 
\left\lbrace
\begin{aligned} 
& \alpha (F^l - P^l)_{ij} + \frac{\beta}{N_l^2 M_l^2} ((F^l)^T (G^l - P^l))_{ji} & if \ \ \mathcal{F}_{ij}^{l} > 0 \\
& 0 & ohterwise
\end{aligned}\right.
$$

- L-BFGS(쿼시-뉴턴) 최적화 알고리즘을 사용했다. 
- 모든 이미지를 같은 사이즈로 스케일해서 입력하므로 정규화(reguralization)할 필요가 없다. 따라서 이미지 prior를 사용하지 않지만, 저수준 이미지 특성들이 prior로 작용한다고 볼 수도 있다. 
- 합성 결과물은 네트워크 구조와 최적화 방법에 따라 약간의 차이가 있다.



## 3. 결과

### 3.1. 내용과 스타일의 trade-off

- 이미지 합성에서 내용과 스타일을 "최적으로" 반영하는 고정된 이미지는 존재하지 않는다.
- 다만 손실 함수를 내용 손실과 스타일 손실의 선형 결합으로 정의했으므로, 가중치를 통해 각각의 반영 비율을 조절할 수 있다. 
- 내용의 가중치를 높이면 이미지의 내용은 식별하기 쉽지만, 스타일은 잘 반영되지 않는다. 스타일의 가중치를 높이면 스타일과 유사한 이미지를 얻을 수 있지만, 이미지의 내용을 식별하기는 어렵다.

<figure style="width: 500px" class="align-center">
	<a href="/imgs/post-imgs/style-transfer-weight-variations.png"><img src="/imgs/post-imgs/style-transfer-weight-variations.png"></a>
	<figcaption>Style transfer weight variations.</figcaption>
</figure>

내용 가중치와 스타일 가중치 비율에 따른 결과값 예로, 그림 위 숫자는 가중치 비율 $\alpha$ / $\beta$ 를 의미한다.


### 3.2. 각 레이어가 미치는 영향

<figure style="width: 400px" class="align-center">
	<a href="/imgs/post-imgs/style-transfer-layer-effects.png"><img src="/imgs/post-imgs/style-transfer-layer-effects.png"></a>
	<figcaption>Layer effects: Varying image as network gets deeper.</figcaption>
</figure>

- 이미지 합성의 중요한 과정은 레이어를 선택하는데에 있다. 
- 스타일 표현식은 신경망의 여러 레이어를 포함하는 멀티-스케일 표현식이다.
- 레이어의 크기와 순서는 스타일이 매치되는 지역적 스케일을 결정한다.


### 3.3. 그래디언트 디센트 초기화 방법

- 그래디언트 디센트를 내용 이미지나 스타일 이미지로 초기화해도 이미지 합성에 큰 영향을 끼치지 않았다.
    - 초기화한 이미지의 특성 공간을 조금 더 반영한다. 
- 단, 노이즈 이미지로 초기화 해야 임의 개수의 서로 다른 이미지를 생성할 수 있다.
    - 고정된 이미지로 초기화하면 그래디언트 디센트의 확률적 과정에 의해 언제나 같은 결과를 도출한다.


### 3.4. 사진에서 추출한 스타일 전이하기

- 같은 과정을 통해 사진에서도 스타일을 추출할 수 있다. 
- 실험에서 사진의 특성이 완전히 보전되지는 않았지만, 합성된 이미지는 스타일 사진의 색과 조명 등을 유사하게 반영했다.


## 4. 논의점

### 한계
- 합성된 이미지의 해상도.
    - 최적화 과제의 차원과 합성곱 신경망의 유닛 개수는 픽셀 수에 선형적으로 증가한다. 따라서 합성 속도는 픽셀 수에 크게 의존한다.
- 합성된 이미지에서 저수준 노이즈가 발견되는 문제.
    - 스타일 이미지가 사진인 경우에 특히 문제가 된다. 그러나 네트워크의 특성 필터와 유사한 경향을 보이므로, 노이즈를 제거하는 방법을 만들 수 있다.
- 사실상 스타일과 내용을 구분하는 뚜렷한 기준이 없다.
    - 이 논문에서는 이미지가 스타일과 '합성된 듯'한 기준으로 결과를 판단하였지만, 정확하거나 보편적이지는 않다.

비록 한계점들이 있지만, 새로운 접근 방법으로 보다 그럴듯한 이미지 합성을 가능하게 했으며 스타일과 내용 및 합성곱 신경망에서 학습하는 사항들에 대한 이해를 증가시켰다.


## Comment

- 스타일 전이는 (개체 인식, 질감 분류)의 선형 결합으로 생성한다. 이 방법은 픽셀 단위의 변환을 넘어서지만 그림의 전체적인 구성을 유지한다. 즉 **스타일 이미지의 저수준과 내용 이미지의 고수준을 합성한 결과를 도출한다.** 이 결과는 개체 인식이 고수준을 학습하고, 질감 분류가 저수준을 학습하는데서 기인한다.
    - 의도한 결과를 위해 저수준의 가중치를 고수준 가중치 보다 훨씬($10$ ~ $10^4$ 배) 적게 설정했다.

한가지 눈에 띄었던 점은, 의도한 결과를 얻기 위해서 입력하는 스타일 이미지에 제약이 있는 것 같다. 
- 이 논문에 소개된 예는 개체가 거의 없는 추상화나 색이나 선 등의 저수준 특징이 내용 사진과 두드러지게 다른 인상주의 그림을 스타일 이미지로 사용했다. J.M.W Turner의 그림(Figure 3.B)처럼 묘사가 많은(descriptive) 그림을 입력했을 때는 의도한 결과를 얻기 힘들었다. 
- 또한 사진을 스타일 이미지로 입력한 예(Figure 7)에서도 내용 이미지와 구성이 닮은 입력값을 선택하였으므로, 이 외의 경우에 대한 탐색이 필요할 것 같다.
- **Q.** 반대로 내용 이미지를 그림으로, 스타일 이미지를 사진으로 입력하면 사진처럼 묘사하는 이미지를 얻을 수 있을지 궁금하다. (painting -> photograph) 사진의 묘사적인 특성은 그림자나 연속적인 경계인데 이는 픽셀보다 넓은 영역에 대해 판단되므로 이미지의 저수준에서 중간 수준을 학습해야 한다. 스타일 표현식을 얻는 과정을 조금 변형해 연구해 볼 수 있다. 







