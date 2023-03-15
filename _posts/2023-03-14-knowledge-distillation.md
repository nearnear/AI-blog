---
title: "Knowledge Distillation, 2015"
categories: Papers
tags:
    - transfer learning
---


지식 증류(Knowledge Distillation)는 학습된 크기가 큰 모델(들)의 정보를 최대한 보존하면서 크기는 작은 모델로 옮기는 방법을 다루고 있다. 따라서 모델 학습을 마친 후 배포하는 과정 등 연산 제약이 있는 경우 고려해 볼 수 있는 기술이다. 
{: .notice}

> Paper : [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)


## 1. Distillation

### 1.1 개요

지식 증류는 분류를 수행하는 큰(cumbersome) 뉴럴 네트워크가 학습한 soft target을 활용해 규모가 작은(small) 모델에 효과적으로 transfer learning을 하는 방법이다. 이때 hard target이란  또는 1로 표현되는 데이터 샘플의 true label을, soft target이란 softmax 함수의 결과 확률 또는 로짓을 의미한다. Soft target은 그 데이터셋에 대한 많은 정보를 포함하고 있으므로, 모든 클래스에 대한 값이 의미를 가진다. 이런 정보를 효과적으로 더 작은 모델로 이전하는 것이 지식 증류의 목적이다. 

실제로는 soft target에 대한 여러 클래스의 softmax 확률 값이 매우 작기 때문에, cross-entropy 비용 함수에 효과적으로 이 정보를 입력하기 어렵다. 지식 증류는 이 문제를 softmax 함수의 온도(temperature)를 높여서 해결한다. 여기서 온도란 softmax 함수의 변수로, 온도를 높일수록 함수는 더 랜덤한 결과를 도출해 클래스 확률 분포를 줄인다. 높은 온도를 통해 더 "부드러운(soft)" 클래스 확률 분포를 얻을 수 있으므로, 결론적으로 soft target들의 확률을 전반적으로 높일 수 있다. 

지식증류는 큰 모델이 soft target을 충분히 도출할 때까지 온도를 올리고, 이 온도를 작은 모델에 적용해 soft target을 작은 모델에 매치시킨다.  이때 증류된 모델이 학습하는 1️⃣ transfer set에 라벨이 없는 경우, 큰 모델의 soft target과 높은 온도를 이용하여 학습한다. 2️⃣ transfer set의 라벨을 얻을 수 있는 경우, 앞선 학습과 correct target을 1의 온도로 학습한 cross-entropy 함수의 가중치 평균을 활용하여 학습하는 것이 효과적이다. 

온도 $T$를 포함한 softmax 함수는 다음과 같다:

$$
q_i = \frac{exp(z_i/T)}{\sum_j exp(z_j)/T}
$$

2️⃣ 의 경우에 대한 최종 손실 함수는 가중지 $\alpha$에 대해 다음과 같이 쓸 수 있다 ($T^2$를 스케일링 해주는 그래디언트 계산 때문인데, 자세히는 아래에서 살펴본다):

$$
L = \alpha T^2 \text{CE} (P_{small}^{T=t}, P_{big}^{T=t}) + (1- \alpha) \text{CE}(P_{small}^{T=1}, \text{true label})
$$


#### 🍉 의사 코드

위에서 정의한 softmax 함수와 손실 함수를 바탕으로 간략한 의사 코드를 쓸 수 있다.

```python
# Define distilled model which has softmax layer with temperature as final layer.
class DistilledModel(Module):
    def __init__(self, input_dim, output_dim, temp=1, name=None):
        super().__init__(name=name)
        self.small_model = ... # define some small model.
        self.temp = temp
        
    def softmax_temp(self, x):
        exp_x = np.exp(x / self.temp)
        exp_x = exp_x - np.max(exp_x)
        sum_exp_x = np.sum(exp_x)
        return exp_x / sum_exp_x

    def __call__(self, x):
        logit = self.small_model(x) 
        output = self.softmax_temp(logit)
        return output
        
# Define final weighted cross-entropy function for distilled model.
# By the paper, it is recommended alpha > 0.5.
def final_cross_entropy(y_true, y_pred, y_soft, big_soft, alpha, high_temp):
    entropy_1 = CrossEntropy(y_soft, big_soft)
    entropy_2 = CrossEntropy(y_true, y_pred)
    return alpha * (high_temp ** 2) * entropy_1 + (1. - alpha) * entropy_2

# Train small model with soft target and high temperature from the cumbersome model.
distilled_model_1 = DistilledModel(input_dim, output_dim, temp=high_temp)
distilled_model_1.compile(...)
y_soft = distilled_model_1.fit((transfer_set, soft_target))

# Train true target of transfer set with temperature 1.
distilled_model_2 = DistilledModel(input_dim, output_dim)
distilled_model_2.compile(...)
y_pred = distilled_model_2.fit((transfer_set, true_target))

# Compute final cross-entropy function.
final_cross_entropy = final_cross_entropy(y_true, y_pred, 
                                          y_soft, big_soft,
                                          alpha, high_temp)
```


### 1.2 Logit과 Distillation

실제로는 Logit을 활용하여 cross-entropy 함수를 변환하는 것 또한 지식 증류의 한 사례다. 이를 확인하기 위해 로짓의 그래디언트 디센트를 해보자.

- 큰 모델의 로짓 $v_i$, 확률 $p_i$
- 증류된 모델의 로짓 $z_i$, 확률 $q_i$
- 증류된 모델의 cross-entopy $C$

이때 증류된 모델의 cross-entropy 그래디언트는 다음과 같다(여기서는 하나의 로짓에 대한 편미분으로 나타냈다):

$$
\begin{aligned}
\frac{\partial C}{\partial z_i} 
&= \frac{1}{T} ( q_i - p_i) \\
&= \frac{1}{T} (\frac{e^{z_i/T}}{\sum_j e^{z_j/T}} - \frac{e^{v_i/T}}{\sum_j e^{v_j/T}})
\end{aligned}
$$

여기서 1️⃣ 온도가 로짓의 값보다 상대적으로 크면 (즉 $1/T$가 전체값을 충분히 작게 만들면) Smoothing을 적용할 수 있다:

$$
\frac{\partial C}{\partial z_i} 
\approx 
\frac{1}{NT} (\frac{1 + e^{z_i/T}}{N + \sum_j e^{z_j/T}} - \frac{1 + e^{v_i/T}}{N + \sum_j e^{v_j/T}})
$$

또한 2️⃣ 가정을 통해 두 로짓의 평균이 $0$이 되도록 만들 수 있다 (즉 $\sum_j z_j = \sum_j v_j = 0 $). 위의 두가지 조건이 만족되면 그래디언트를 다음과 같이 **두 로짓의 오차에 대한 상수배**값으로 근사할 수 있다:

$$
\frac{\partial C}{\partial z_i} 
\approx 
\frac{1}{NT^2} (z_i - v_i)
$$

위의 편미분을 모든 로짓 $z_i$에 대한 그래디언트를 구하고 $T^2$로 스케일링을 하면, 지식 증류는 두 로짓의 오차 제곱 $\frac{1}{2} (z_i - q_i)^2$를 최소화하는 작업이 된다.


