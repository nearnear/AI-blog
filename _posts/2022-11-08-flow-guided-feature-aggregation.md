---
title: "FGFA for Video Object Detection, 2017"
categories: Papers
tags:
    - video
gallery:
  - url: /imgs/post-imgs/fgfa-figure3.png
    image_path: /imgs/post-imgs/fgfa-figure3.png
    alt: "움직임 속도에 따른 정확도 실험"
  - url: /imgs/post-imgs/fgfa-figure4.png
    image_path: /imgs/post-imgs/fgfa-figure4.png
    alt: "움직임 속도에 따른 정확도 실험"
---

**원 논문 :** [Flow-Guided Feature Aggregation for Video Object Detection, 2017](https://arxiv.org/pdf/1703.10025.pdf)
{: .notice--info}


## 들어가며

시퀀스 데이터에 관심을 가지다 보니, 최근에는 시퀀셜한 이미지인 비디오 데이터(sequence of related frames)에 관심을 
가지게 되었다. 물론 프레임 별로 이미지 과제를 수행하면 수행 시간이나 정확도 측면에서 비효율적일 것이라는 생각이 들었다. 
그러므로 현재는 어떤 식으로 비디오 처리 과제를 수행하고 있는지 관련 발표 자료를 찾아보게 되었다.

실제로 현실에서 대부분의 사물은 움직이는 상태인데, 예를 들어 제조 산업의 컨베이어 벨트 위의 사물을 
인식하는 경우도 그렇다. 이때 사진을 찍기 위해 멈추지 않고 움직이는 상태에서 얻은 이미지를 통해서 일정 정도의 정확성을 얻을 수 있다면
사용 환경에서 있어서 시간 효율적이다. 또는 자율주행처럼 특정 시간 내에서 지속적으로 인식과 판단이 필요한 경우, 
이미지 처리에서 얻을 수 없는 움직임 또는 속도 벡터를 얻는 과제를 목표로 할 수 있다.

[여기 블로그](https://towardsdatascience.com/ug-vod-the-ultimate-guide-to-video-object-detection-816a76073aef)
에 따르면 비디오 개체 인식 방법에는 크게 세가지가 있다. *사후 처리 방식(Post-processing method)*, 
*다중 프레임 방식(Multi-frame Method)*, 그리고 *광학 흐름 방식(Optical Flow Method)* 이다.
광학 흐름 방식은 특성 뒤틀림(feature warping)을 감지해 움직이는 개체를 배경과 구분하여 인식하는 방법이다. 

<figure style="width: 400px" class="align-center">
	<a href="/imgs/post-imgs/fgfa-figure1.png"><img src="/imgs/post-imgs/fgfa-figure1.png"></a>
    <figcaption>FGFA: aggregating feature warping</figcaption>
</figure>

이 논문은 이미지 처리와 다른 비디오 처리의 특징에 초점을 두고 사후 처리 방식이나 다중 프레임 방식보다 더 적은 연산으로 더
효율적인 결과를 얻는데에 초점을 맞추었다; 즉 움직임 경로(motion path)에 있는 프레임 상관성을 종합해 프레임당 특징
(per-frame features)을 예측한다. 결과적으로 단일 프레임에 대한 개체 인식에 대해 ImageNet VID 2016, 
2017에서 성과를 거두었다고 한다. 


## 모티브

**Notation :** 본 논문에서는 snap-shot을 통해 사후 처리 방식으로 temporal information(시간적인 정보, 
더 나은 번역이 생각나지 않는다.)을 얻는 방법을 *box level* 방식이라고 불렀다. 반면 원칙에 따라(principled way)
끝에서 끝까지(end-to-end) 학습한, 즉 비디오 자체에서 특성(feature)을 추출하여 광학 흐름 네트워크로 학습한 모델을 
*FGFA(Flow-Guided Feature Aggregation)*이라고 불렀다.
{: .notice--info}

<figure style="width: 500px" class="align-center">
	<a href="/imgs/post-imgs/fgfa-figure2.png"><img src="/imgs/post-imgs/fgfa-figure2.png"></a>
    <figcaption>Typical deteriorated appearances in video.</figcaption>
</figure>

움직임에 의한 흐려짐, 초점 안 맞음, 일부 가려짐, 특이한 포즈(동물의 배나 발..) 등 비디오에서는 급격한 형태 변화를 관찰할 수 
있다. 이는 흔한 경우이고, (예시의 경우) 사람이 비디오를 보고 있다면 사물을 인식할 수 있을 것이다. 
이는 고정 이미지에서는 보기 힘든 비디오의 특성이다.

모티브가 된 선행 연구는 다음과 같다.
- 이미지 개체 인식
- 비디오 개체 인식
- 흐름(flow)을 통한 움직임 예측
- 특성 통합(Feature aggregation)
- 시각적 추적(Visual tracking)


## 모델

### 디자인

**WANT :** 입력 비디오 프레임 {$I_i$}에 대해 모든 프레임에 대한 개체 인식 박스 {$y_i$}를 완성된 형태로 반환하는 것
{: .notice}

일반적인 CNN 기반 개체 인식은 깊은 합성곱 네트워크 $N_{\text(feat)}$을 통해 특성 맵 $f = N_{\text(feat)}(I)$를 
얻고, 얕은 감지 네트워크 $N_{\text(det)}$를 통해 결과 $y = N_{\text(det)}(f)$를 얻는다.

$$
I_i 
\xrightarrow{\qquad N_{\text{feat}} \qquad} f 
\xrightarrow{\qquad N_{\text{det}} \qquad} y_i
$$

FGFA도 유사한 방식으로 디자인된다. 
보다 구체적으로 특성 맵 추출은 두가지 절차로 나뉜다. 첫째로는 **흐름-지도 뒤틀림 인식(Flow-guided warping)** 으로, 
흐름 장 $M_{i \rightarrow j}$를 흐름 네트워크 $F$로 예측한다. 그 다음 뒤틀림 함수 $f$는 각각의 특성 맵에 있는
모든 위치에 대해 적용되는 선형 함수 $W$에 의해 결정된다(i와 j의 위치 변화에 주목):

$$
\begin{aligned} 
    M_{i \rightarrow j} &= F(I_i, I_j) \\
    f_{j \rightarrow i} &= W(f_j, M_{i \rightarrow j}) \\
                        &= W(f_j, F(I_i, I_j)) 
\end{aligned}
$$

두번째는 **특성 종합(Feature aggregation)** 으로 배경과 분리된 움직임을 학습한 후 앞-뒤 시간 단위 $K$개의
특성 맵을 모두 종합한다(현재 프레임도 포함하므로 총 2K + 1개 프레임). 이 특성 맵들은 제각각 빛, 관점, 포즈 등 고정되지 않은
특성들을 가지고 있다. 종합을 위해 **모든 채널이 같은 2D 공간 가중치** $w_{j \rightarrow i}$를 가지도록 한다:

$$
\bar{f_i} = \sum_{j=i-K}^{i + K} w_{j \rightarrow i} f_{j \rightarrow i}
$$

(이 식은 특성 맵에 위치에 따라 다른 가중치를 부여한다는 점에서 어텐션 방식과 유사하다!)

이제 감지 네트워크를 통과해 결과값을 반환하는데, 이때 시간에 따른 특성 맵이 결과를 얻은 후가 아닌 **결과를 얻기 전에** 
변수로 작용한다는 점을 주목하자:

$$
y_i = N_{\text{det}} (\bar{f_i})
$$

공간 $p$에 따른 가중치 $w_i$ 결정 방식은 레퍼런스 특성 $f_i$과 뒤틀림 특성 $f_{j \rightarrow i}$ 
사이의 유사도를 cosine similarity로 측정한다. 각 특성에는 작은 합성곱 네트워크 $\varepsilon$를 적용한다. 
이를 임베딩 특성 $f^e = \varepsilon (f)$로 표시한다. 최종적으로 가중치를 결정하는 방법은 다음과 같다:

$$
w_{j \rightarrow i}(p) = 
exp( \frac{f_{}^e(p) \cdot f_i^e (p)}{|f_{}^e(p)| \cdot |f_i^e (p)|} )
$$

이후 가중치 $w_{j \rightarrow i}(p)$를 주변 $K$ 개의 프레임에 대해 모든 위치 $p$에 대해 normalize한다:

$$
\sum_{j=i-K}^{i + K} w_{j \rightarrow i}(p) = 1
$$

### 학습법

<figure style="width: 400px" class="align-center">
	<a href="/imgs/post-imgs/fgfa-sudo-algorithm.png"><img src="/imgs/post-imgs/fgfa-sudo-algorithm.png"></a>
    <figcaption>Pseudo-algorithm for FGFA.</figcaption>
</figure>

주어진 시간 $i$에 대해,
(1)앞 뒤 K개의 프레임에 대해 특성 맵을 계산한 뒤, (2)레퍼런스 특성과의 유사도를 계산, 그리고 
(3)두 특성의 코사인 유사도를 계산한다. 이를 통해 얻은 시점 $i$의 특성 맵을 모두 종합해서 결과값을 얻어
$(i+K+1)$ 입력값에서 특성 맵을 추출해 해당 시점의 특성 버퍼에 업데이트하는 방식으로 학습한다.

**시간 복잡도 비율 :** 단일 프레임 베이스라인에 대한 FGFA의 시간 복잡도 비율은 다음과 같다. 
이때 $O(N_{\text{feat}})$에 비해 $O(\varepsilon) + O(W)$는 작은 값이다.

$$
\begin{aligned}
    r &= 1 + \frac{(2K + 1) \cdot (O(F) + O(\varepsilon) + O(W))}{O(N_{\text {feat}}) + ON_{\text{det}}} \\
      &= 1 + \frac{(2K + 1) \cdot O(F)}{O(N_{\text{feat}})}
\end{aligned}
$$

**Temporal dropout :** 학습에는 $K = 2$, 인퍼런스에서는 $K=10$로 설정하였다. 이는 학습에서 앞 뒤 프레임을
무작위로 선택하여 일종의 dropout을 실행한 것이다. 앞서 공간에 따른 가중치를 학습과 인퍼런스에서 모두 정규화했으므로 
각각 다른 K 값으로 설정할 수 있다. 
{: .notice}

구체적으로 활용한 네트워크는 다음과 같다.
- Flow network : FlowNet
- Feature network : Pretrained ResNet(50, 101) and Aligned-Inception-Resnet
- Embedding network : 1 x 1 x 512, 3 x 3 x 512, 1 x 1 x 2048 convolution
- Detection network : R-FCN


## 실험

### 비교 기준

**ImageNet VID 데이터셋**에 대해 **느림, 중간, 빠른** 움직임에 대해 검증을 실행했다.
학습과 검증 데이터는 (3862, 555)개 비디오 조각이며, 30개의 분류 클래스가 있다. 
학습은 two-phased로 진행되었다: 처음에는 VID가 속한 ImageNet DET 데이터셋의 30개 클래스에 대해 SGD를 
학습했고(120K 반복, 4 GPUs), 두번째로 IamgeNet VID 데이터셋의 같은 30개 클래스에 대해 FGFA를 학습했다
(60K 반복, 4 GPUs).

**움직임 속도에 대한 실험.** 속도에 대한 평가는 근처(양쪽 10개) 
프레임에 대한 IoU를 기준으로 느린 움직임은 IoU > 0.9, 중간은 [0.7, 0.9], 빠른 움직임은 IoU < 0.7로 정의했다.
평가 척도는 mAP(mean Average Precision)으로 움직임 속도에 따라 나누어 평가했다.

{% include gallery caption="Histogram, and example video snippets." %}

첫번째 그림은 IoU 점수를 통한 속도 분류 기준을 나타내고, 두번째 그림은 속도에 따른 학습 예시이다.

**모델 구조 탈락 실험.** 또한 모델 구조의 일부분을 탈락(ablation)시켜 성능을 평가하는 방법으로 구조적인 평가를 진행했다. 
평가되는 모델 구조는 각각 (a), (b), (c), (d), (e)로 표기했다.

**box-level 모델로 후처리를 했을 때 성능 비교.** motion guided propagation(MGP), Tubelet rescoring, 
그리고 Seq-NMS 방법으로 후처리를 한 결과를 비교했다.

**Sota 모델과 결과 비교.** ImageNet VID challenge 2016에서 검증 81.2% mAP로 이겼던 
NUIST Team을 baseline으로 결과를 비교했다.


### 결과

<figure style="width: 700px" class="align-center">
	<a href="/imgs/post-imgs/fgfa-table1.png"><img src="/imgs/post-imgs/fgfa-table1.png"></a>
    <figcaption>Difference in model architecture.</figcaption>
</figure>

평가 대상이 되는 [다중 프레임 특성 통합, 적응 가중치, 흐름 지도, end-to-end 학습] 구조를 모두 포함한
(d) 모델이 가장 높은 mAP 점수를 보였다. End-to-end 방식을 제외한 (e)도 같은 성능을 보였지만, 그 외의
구조를 제외한 경우 현저하게 낮은 성능을 보였다. 또 움직임이 빨라질수록 mAP가 감소했다.

<figure style="width: 700px" class="align-center">
	<a href="/imgs/post-imgs/fgfa-table3.png"><img src="/imgs/post-imgs/fgfa-table3.png"></a>
    <figcaption>Difference in number of frames.</figcaption>
</figure>

ResNet-50을 통해 학습 K와 검증 K의 변화에 따른 mAP를 비교했다. *로 표기된 디폴트값이 한계 성능 증가량이
가장 큰 지점임을 알 수 있다.

<figure style="width: 400px" class="align-center">
	<a href="/imgs/post-imgs/fgfa-table4.png"><img src="/imgs/post-imgs/fgfa-table4.png"></a>
    <figcaption>Combination with box-level entries.</figcaption>
</figure>

다양한 모델로 후처리 했을 때의 성능을 비교했다. FGFA와 Seq-NMS의 조합이 가장 높은 성능을 보였다.

<figure style="width: 700px" class="align-center">
	<a href="/imgs/post-imgs/fgfa-figure6.png"><img src="/imgs/post-imgs/fgfa-figure6.png"></a>
    <figcaption>Comparison of results with FGFA(ResNet-101) vs Baseline.</figcaption>
</figure>

ResNet-101을 사용하고 Seq-NMS로 후처리한 FGFA를 Baseline 모델과 비교한 결과. 녹색 선은 correct 
detection, 노란 선은 incorrect detection을 나타낸다. 


{% include video id="R2h3DbTPvVg" provider="youtube" %}

0:33 부터 Baseline과 비교한 FGFA 결과 예시를 볼 수 있다. 모티브에서 염두에 둔 {움직임에 의한 흐려짐, 초점 안 맞음,
일부 가려짐, 특이한 포즈} 등에서 baseline에 비해 더 나은 결과를 보인다.


## 나가며

비디오 개체 인식에서 단일 프레임 예측은 전후 프레임이 메타 데이터로서 감독(supervision)하여 풍성한 정보에 기반하여
예측을 수행하는 것과 같다(이전에 CLIP을 통해 이미지 과제를 간단한 자연어 감독을 통해서 성능을 향상시키는 경우와 
유사하다). 보다 보편적으로 모든 시간에 따른(temporal) 데이터는 일종의 메타 데이터로 이해할 수 있을 것이다. 

비디오는 물리 세계에서 얻을 수 있는 자연스러운 데이터이며 웹에서도 무궁무진하게 얻을 수 있다는 점에서 활용처가 많다. 
다만 비디오 데이터에서 정확도와 학습 효율성을 모두 얻는 점은 연구 과제로 남을 것이다. 
높은 정확도를 위해서는 느린 움직임을 대상으로 해야하는데, 이는 시간 단위를 줄이는 것과 같고, 즉 프레임 수 K가 늘어나 학습 속도는 감소한다. 
또 사람이 라벨을 매 프레임마다 표기하기에는 시간 단위에 관계없이 프레임 수가 늘어날 수록 부담스럽기 때문에, 
semi-supervised 또는 unsupervised 방식의 학습 방법이 요구될 것이다.


### 참고 자료
- [The Ultimate Guide to Video Object Detection](https://towardsdatascience.com/ug-vod-the-ultimate-guide-to-video-object-detection-816a76073aef)