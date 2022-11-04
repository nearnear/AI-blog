---
title: "Knowledge Distillation, 2015"
categories:
    - Papers
tags:
    - distillation
    - mnist
    - speech recognition
---

> 원 논문: [Knowledge Distillation in Neural Network](https://arxiv.org/abs/1503.02531v1)

연산 크기가 큰 모델의 Soft target을 이용하여 크기가 작은 모델로 정보를 이전하는 Distillation 방법을 소개한다. 

## 연구 계기

머신러닝 모델은 학습할 때는 큰 데이터셋을 처리하기위해 많은 연산 능력을 필요로하지만 배포할 때에는 호출 시간(latency)과 연산 자원에 있어서 보다 큰 제약을 가진다. 여러 모델의 결과를 평균하는 앙상블은 일반적으로 더 연산 효율적인 결과를 도출하지만 여러 모델을 학습하기 위해 연산 비용이 증가한다는 단점이 있다. 이에 대해 선행 논문(Caruana et al., 2006)에서 knowledge transfer를 통해 앙상블 모델의 효과를 단일 모델로 구현할 수 있음을 증명하였다. 그러나 knowledge의 가중치에 집중해 생성하는 단일 모델의 구조에 제약이 있었다. 이에 본 논문에서는 보다 일반적인 knowledge transfer 방법인 "Distillation"을 제안하고 선행 논문을 distillation의 특수 사례로 포괄했다.


## 1. 정의와 이론

### 메인 아이디어

**Notation:** 혼란을 방지하고 설명을 단순화하기 위해 배포 이전의 학습 단계의 모델을 *무거운 모델*로(논문에서는 "cumbersome model"로 표기), 이로부터 생성한 배포를 위한 작은 모델을 *가벼운 모델*("small model") 또는 *distilled 모델*로 표기했다.
{: .notice}

우선 크기가 큰 학습 모델의 knowledge를 단지 모델의 가중치로 보는 관점에서 벗어나 입력 벡터에서 결과 벡터로 향하는 mapping으로 보았다. 일반적으로 학습은 일반화를 위해 "정답인" 클래스(**hard target**)에 대한 평균 로그 확률을 최대화하는 방식으로 진행된다. 즉 정답이 아닌 클래스에 대해서도 확률을 측정하며, 이 "답이 아닌" 클래스(**soft target**)에 대한 확률의 비율을 비교하여 클래스 간의 유사도 정보를 얻을 수 있다. 만약 가능한한 많은 데이터 셋에 대해 학습한 무거운 모델의 일반화(generalization) 성능을 보다 작은 모델에 이전(transfer)할 수 있다면 처음부터 작은 데이터 셋에 대해서만 학습한 모델보다 더 좋은 성능을 낼 수 있을 것이다.

즉 일반화 성능을 이전하는 방법은: **무거운 모델의 클래스 확률을 가벼운 모델의 soft target으로 사용**하는 것이다.

- 가벼운 모델의 학습 데이터는 (라벨이 있거나 없는) transfer set을 사용하거나 무거운 모델의 데이터셋을 그대로 활용할 수 있다.
- 무거운 모델이 앙상블 모델인 경우, 각 모델의 클래스 확률을 산술 또는 기하 평균한 값을 활용한다.

**모델 학습시 참고:** soft target이 충분히 엔트로피가 높으면, hard target에 비해 더 많은 정보를 담고 있으며, 학습시 그래디언트의 variance 또한 작다는 것을 의미한다. 따라서 가벼운 모델은 더 작은 데이터셋에 대해 더 큰 learning rate를 사용해 학습할 수 있다.
{: .notice--info}

#### 문제: soft target 확률이 작아 cross-entopy에 반영되지 않는다.
예를 들어 MNIST 학습처럼 충분히 큰 모델이 높은 정확도를 내는 과제에서 학습된 정보는 대부분 soft target에 저장된다. 예를 들어 "2"를 나타내는 데이터는 "3"으로 분류될 확률이 $10^{-6}$인데 반해 "7"로 분류될 확률은 $10^{-9}$일 수 있다. 이런 데이터에 대한 확률 비율은 클래스 유사도를 나타내는 귀중한 정보이지만 확률 자체의 값이 너무 작아 cross-entropy 비용 함수에서는 거의 반영되지 않는다.

#### 해결: softmax 함수의 temperature를 높인다.
우선 무거운 모델이 충분히 "soft"한 target set을 도출할 때까지 temperature를 높여, 이 높은 temperature를 가벼운 모델을 학습할 때 적용하여 계산한다.

temperature $T$를 포함한 softmax는 로짓 $z_i$를 다른 로짓들 ${z_i}$와 비교하여 확률 $q_i$를 도출하는 함수이다.

$$
q_i = \frac{exp(z_i / T)}{\sum_j exp(z_j/ T)}
$$

일반적으로 $T=1$로 설정하며, $T$를 높이면 더 "soft"한 클래스 확률 분포를 얻을 수 있다.

#### transfer set 활용방법
transfer set의 데이터에 대해 무거운 모델의 soft target 분포를 활용하면 distilation을 할 수 있다. 만약 transfer set이 라벨된 경우, soft target과 hard target에 대한 cross-entropy를 각각 계산하여 가중합 할 수 있다. 
1. soft target의 cross-entropy를 오리지널 모델의 temperature를 사용하여 계산한다.
2. hard target의 cross-entorpy를 $T=1$로 계산한다.

이때 soft target으로 생성한 그래디언트의 크기는 $1/T^2$배 스케일되어 있으므로 그래디언트에 최종적으로 $T^2$를 곱해야 $T$를 변화해도 hard와 soft target의 비율을 유지할 수 있다.

### distillation의 특수 사례로서 logit 활용하기

앞서 선행 논문을 distillation의 특수 사례로 포괄했다고 했다. transfer set의 (softmax의 결과값인 확률 대신 그 입력값인) 로짓 $z_i$를 통해 계산한 cross-entropy 그래디언트는 오리지널 모델의 로짓 $v_i$와 soft target 확률 $p_i$에 대해 쓰여진다.

$$
\begin{aligned}
    \frac{\partial C}{\partial z_i} &= \frac{1}{T} (q_i - p_i) \\
    &= \frac{1}{T}(\frac{e^{z_i/T}}{\sum_j e^{z_j/T}} - \frac{e^{v_i/T}}{\sum_j e^{v_j/T}})
\end{aligned}
$$

이때 $T$가 충분히 커지면 $e^{z_i/T} \simeq 1 + z_i/T$이다. 또한 로짓이 각 데이터에 대해 평균이 0이라고 가정하면 $\sum_j z_j = \sum_j v_j = 0$이다. 따라서 위 식을 다음과 같이 근사할 수 있다.

$$
\begin{aligned}
    \frac{\partial C}{\partial z_i} & \simeq \frac{1}{T}(\frac{1 + z_i/T}{N + \sum_j z_i/T} - \frac{1 + v_i/T}{N + \sum_j v_i/T}) \\
    &\simeq \frac{1}{NT^2}(z_i - v_i)
\end{aligned}
$$

따라서 $T$가 충분히 크고 각 데이터의 로짓이 0을 기준으로 분포할 때, 로짓을 통한 knowledge transfer(그래디언트 디센트)는 그래디언트 근사값 $1/2(z_i - v_i)^2$를 최소화하는 작업이 된다. 로짓 오차 제곱은 음수 로짓값을 무시하는 한편 오리지널 모델의 로짓을 통해 노이즈가 제거되는 효과를 얻을 수 있다. 종합적으로 어떤 결과를 얻을지는 경험적인(empirical) 문제이며, 이후의 실험에서 distilled 모델의 크기가 오리지널 모델의 정보를 모두 담기에 너무 작은 경우에 적당한 크기의 temperature에 대해 로짓으로 계산한 distillation이 유용함을 확인했다.

## 2. 성능 비교실험

### 기본 실험: MNIST 

distillation으로 큰 모델과 비슷한 성능을 낼 수 있는지 실험했다.

| size of NN | generalization | test errors |
| :---: | :---: | :---: |
| big | Dropout | 67 |
| small | None | 146 |
| small | Distillation | 74 |

모델은 모두 2개의 hidden layer를 가지며 큰 모델의 경우 1200개, 작은 모델의 경우 800개의 hidden unit을 가진다. 결과적으로 MNIST의 6만개 데이터셋에 대해 학습한 큰 모델과 그 soft target을 추가적으로 학습한 작은 모델은 성능이 비슷했다. 충분한 bias(3.5)를 고려하면 작은 모델의 학습 데이터에서 하나의 클래스가 누락되더라도 distillation을 통해 누락된 클래스에 대해 98.6%의 예측 정확도를 얻을 수 있었다. 두개의 클래스가 누락된 경우 bias를 더 높여(7~8) 정확도를 높일 수 있었다.

### 앙상블 효과 실험: speech recognition

앙상블 모델을 distilling하는 것이 같은 크기의 단순 모델에 비해 성능이 뛰어나다는 것을 증명하고자 했다.

논문이 쓰일 당시의 자동 음성 인식은 DNN을 통해 매 시점에 waveform을 독립 Hidden Markov 모델(HMM)의 특정 상태에 대한 확률 분포를 예측했으며, 라벨은 시퀀스 순서로 강제되었다. 즉 일반적으로 DNN을 통해 예측값과 라벨의 frame당 클래스를 cross entropy 최소화 방식으로 학습했으므로 distillation을 수행하기 적합했다. 

시간 t에서 acoustic 데이터 $s_t$를 HMM의 "옳은"(hard target) 상태 확률 $h_t$로 나타내는 확률로 모델의 파라미터 $\theta$를 다음과 같이 결정할 수 있다:

$$
\theta = \text{argmax}_{\theta'}P(h_t|s_t;\theta')
$$

| System | Test Frame Accuracy | WER |
| :---: | :---: | :---: |
| Baseline | 58.9% | 10.9% |
| 10xEnsemble | 61.1% | 10.7% |
| Distilled Single Model | 60.8% | 10.7% |

Baseline은 Android voice search이며 앙상블은 10개의 모델에 대해 진행하였다. 의도한대로 앙상블 모델의 성능이 전이되었음을 Accuracy와 WER의 유사함을 통해 확인할 수 있다.

### Regularizer 효과 실험

Soft target을 hard target과 같이 활용하는 것이 regularizer 역할을 할 수 있다는 것을 보인다. 

| System & training set | Train Frame Accuracy | Test Frame Accuracy |
| :---: | :---: | :---: |
| Baseline(100% of training set) | 63.4% | 58.9% |
| Baseline(3% of training set) | 67.3% | 44.5% |
| Soft Targets(3% of training set) | 65.4% | 57.0% |

앞서 본 speech recognition과제의 baseline은 85M개의 파라미터를 가졌다. 학습 데이터의 3% 만으로 새로운 모델을 학습하면 심한 overfitting이 나타났지만, soft target을 활용하여 학습하면 전체 데이터셋의 대부분의 정보를 학습할 뿐만 아니라 early stopping 없이도 수렴하는 결과를 보였다.



