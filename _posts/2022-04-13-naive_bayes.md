---
title: NLP, 나이브 베이즈 분류
categories: ML
tags:
    - NLP
    - Sentiment Analysis
---

**Note:** 본 글은 이전에 작성한 글을 백업한 것입니다.
{: .notice--info}

이 글은 deeplearning.ai의 NLP Specialization를 참고하여 나이브 베이즈 모델을 텍스트 정서 분석에 초점을 맞춰 정리한 글입니다.

> [Github에서 Naive Bayes 코드 보기](https://github.com/snowith/nlp_model_practices/blob/main/naive_bayes/naive_bayes_sentiment.ipynb)


## 0. 모델 개략
나이브 베이즈 모델은 분류 과제를 위한 확률 모델이다. 훈련 데이터에 등장하는 모든 단어의 빈도를 세어서 각 데이터에 대한 조건부 확률의 비율을 계산하므로 분류 과제를 수행하는데 적합하다.

### 나이브 베이즈 모델은
- 훈련과 예측을 빠르게 수행할 수 있으므로 baseline 모델로 적합하다.
- 문장에 있는 각 단어들이 독립적이라고 가정하기 때문에, 문장 내 단어들의 관계를 측정하거나 문장 내의 빈칸을 채우는 등의 과제에는 적합하지 않다.
- 훈련 데이터 내 단어들이 등장하는 빈도에 기반하기 때문에, 훈련 데이터에 포함되지 않은 새로운 단어에 대한 예측이나 단어의 순서를 판단하는 과제에는 적합하지 않다.
- 감정 분석, 저자 분류, 스팸 필터링, 문서 요약, 동음이의어 구분 등의 과제에 활용할 수 있다.

이 글에서는 나이브 베이즈 모델로 이진 분류를 수행하는 상황을 가정하겠다. 특히 어떤 문장을 입력 받아서 문장이 긍정적인 정서를 내포하고 있으면 `1`을, 부정적인 정서를 내포하고 있으면 `0`을 반환하는 감정 분석(sentiment analysis) 과제를 수행한다.

```python
input_s = 'This is my best day ever.'
model(input_s) # 1, 긍정

input_s = 'the class was in a terrible mood...'
model(input_s) # 0, 부정
```
_위와 같은 작업을 수행하는 모델 `model`을 얻는 것이 목적이다._



## 1. 조건부 확률과 베이즈 룰
모델을 살펴보기에 앞서 조건부 확률과 베이즈 룰에 대해 알아보자.

### 🎲 조건부 확률(Conditional Probability) 이란?
모수에서 조건 A가 만족될 확률을 $P(A)$, 조건 B가 만족될 확률을 $P(B)$라고 하자. 이때 B에 대한 A의 조건부 확률 $P(A\mid B)$ 는 조건 B를 만족하는 표본에서 조건 A를 만족하는 표본을 선택할 확률을 의미한다. 즉 $P(A\mid B)$는 조건 A와 B를 모두 만족하는 표본을 선택할 확률인 $P(A \cap B)$에 모수에서 조건 B를 만족하는 표본을 선택할 확률 $P(B)$를 나눈 값으로 정의된다.

$$
P(A\mid B) =\frac{P(A \cap B)}{P(B)}
$$

조건부 확률은 뽑을 샘플의 범위를 표본 대신 조건으로 제한하는 효과가 있다. 

### 🎲 베이즈 정리란
위의 정의로 부터 두개의 조건부 확률을 표현 할 수 있다.

$$
P(A\mid B) = \frac{P(A \cap B)}{P(B)} \\
P(B\mid A) = \frac{P(A \cap B)}{P(A)} 
$$

예를 들어 조건 `A`가 `20대`이고 조건 `B`가 `심장병`이라고 하자. 몇 개의 병원에서 표본 집단을 모아서 **심장병에 걸린 사람이 20대일 확률**을 조사하고자 한다. 우리는 표본 집단 데이터베이스로 부터 **심장병이 걸린 사람의 비율**과 **20대의 비율**을 알고있으며, 나아가 **20대 중에서 심장병에 걸린 사람의 비율**을 알 수 있다. 베이즈 정리는 세가지 정보로부터 심장병에 걸린 사람이 20대일 확률을 도출한다. 대수 연산을 통해 $P(A \mid B)$를 $P(B \mid A)$에 대해 다음과 같이 표현할 수 있다.

$$
P(A\mid B) = \frac{P(A)}{P(B)} \times P(B\mid A) ...... (*)
$$

두 조건의 조건부 확률의 관계를 나타내는 수식 (*)을 **베이즈 정리**라고 한다. 이때 $P(A)$를 **사전확률(prior)**, $P(A \mid B)$를 **사후확률(posterior)**, 그리고 $P(B \mid A)$를 **우도(likelihood)**라고 부른다.


## 3. 조건부 빈도 세기
텍스트가 내포하는 감정을 이진 분류하기 위해 분류 클래스를 $class \in \{positive, negative\}$로 정의하자. $m$개의 단어를 포함하는 corpus에 속하는 단어 $w_i \in corpus$에 대해 우리가 구하고자 하는 값은 $P(class \mid w_i)$, 즉 단어가 주어졌을 때 단어가 특정 class에 속할 확률을 구하는 것이다. 베이즈 정리를 떠올려 보면:

$$
P(class\mid w_i) = \frac{P(class) \cdot P(w_i\mid class)}{P(w_i)}
$$

이며 $P(w_i)$는 $w_i$에 대한 상수값이므로 확률을 계산할 때 무시할 수 있다. 우리는 문장을 입력받아 각 단어의 조건부 확률을 계산해서 나이브 가정에 따라 한 문장의 조건부 확률을 반환하고자 한다. 따라서 **나이브 베이즈** 모델의 아이디어를 수식으로 나타내면 예측하고자 하는 문장 $sentence$에 속한 모든 단어 $w_i \in sentence$ ($i=1, .., n$)에 대해 다음과 같이 쓸 수 있다: 

$$
\hat{y} = argmax_{class} P(class) \prod_{i=1}^{n}P(w_i\mid class)
$$

위 식은 최대 우도 추정(Maximum Likelihood Estimation, MLE)의 아이디어이기도 하다. 우선은 corpus에 대한 조건부 빈도인 $P(w_i\mid class)$를 계산해야 한다.

다시 베이즈 정리에 의해, 클래스에 대한 조건부 확률을 다음과 같이 계산할 수 있다.

$$
\begin{aligned}
P(w \mid  class) &= \frac{P(w \cap class)}{P(class)} \\
&= \frac{freq(w, class)}{N_{class}}
\end{aligned}
$$

위 식에서 $freq(w, class)$는 $class$에서 $w$가 나타나는 횟수로, $P(w \cap class)$와 같다. $N_{class}$는 클래스에 포함되는 모든 단어의 빈도이다.

### 🎲 Laplacian Smoothing
Laplacian Smoothing은 조건부 확률이 0이 되는 것을 방지하기 위해 사용하는 기법이다. 위에서 본 최대 우도 추정에 따르면 모든 특성에 대해 likelihood를 곱하게 되는데, 만약 corpus에 없는 단어가 들어오면 다른 특성들에 관계없이 예측값이 0이 될 것이다. 분자값에 bias를 1 더하면 우리가 원하는 분류를 수행할 수 있다. 

$$
P(w\mid class) = \frac{freq(w, class) + 1}{N_{class} + V_{class}}
$$

$V_{class}$는 클래스에 등장하는 **유일한** 단어의 개수이다. 분모에는 $V_{class}$를 더함으로서 모든 단어에 대한 likelihood가 1이 넘지 않도록 설정할 수 있다:

$$
\sum_{w}P(w\mid class) = \frac{\sum_w freq(w,class) + V_{class}}{N_{class} + V_{class}}
$$



## 3. Likelihood 계산하기
앞에서 표현한 최대 우도 추정 방식을 조금 변형해, 이 글에서는 **Likelihood-ratio** 방법을 통해 분류 작업을 수행하고자 한다. 

우선 ratio란 분류 $class$에 대한 조건부 확률의 비율이다. 임의의 단어 $w_i$에 대해 $ratio(w_i)$는 다음과 같이 정의할 수 있다.

$$
ratio(w_i) = \frac{P(w_i\mid Pos)}{P(w_i\mid Neg)}
$$

Likelihood란 표본을 결합 확률로 나타낸 함수이며, 여기서는 입력 문장$s$가 임의의 $class$일 확률을 의미한다. 여기서는 Likelihood를 ratio에 대해 정의하자. 즉 모든 입력값 $w_i \in s$에 대해 ratio를 곱한 값으로 표현한다.

$$
likelihood(s) = \prod^{m}_{i=1}\frac{P(w_i\mid Pos)}{P(w_i\mid Neg)}
$$

만약 입력값의 모든 단어 $w_i$가 corpus의 긍정적인 라벨과 부정적인 라벨에서 같은 빈도로 나타났다면 likelihood 값은 `1`로 나타날 것이다. 이 결과를 긍정적이지도 부정적이지도 않은 **중립 값**이라고 볼 수 있다. 반면 분모 분자는 빈도 수이므로 likelihood는 음의 값을 가질 수 없고, 분모 $P(w_i \mid  Neg )$가 분자 $P(w_i \mid  Pos)$ 보다 커질 수록 0에 가까워지고 반대의 경우 양의 무한대 값에 가까워질 수 있다.

### 🎲 Naive 란?
베이즈 모델이 **naive**(순진하다)는 말은 모수의 모든 표본이 상호 독립적이고 완전하다고 가정하는 것을 뜻한다. 즉 머신 러닝 모델에서는 데이터의 모든 특성들(features)을 알 수 있고, 나아가 특성들이 서로 독립적이라고 가정하는 것을 뜻한다. 예를 들어 한 문장을 데이터 한개라고 하면, 문장에 속한 단어를 데이터의 특성들로 볼 수 있고 나이브 베이즈 모델은 이 단어들이 상호 연관(covariate) 되어있지 않다고 가정한다. 

단어 $w_i (i= 1, ..., n)$을 포함하는 문장이 class에 속할 확률은 결합 확률 $P(class, w_1, ..., w_n)$인데, 연쇄 법칙에 따르면:

$$
\begin{aligned}
P(class, w_1, ..., w_n) 
&= P(class) \cdot P(w_1, ..., w_n) \\
&= P(class) \cdot P(w_1\mid class) \cdot P(w_2, ..., w_n) \\
&= P(class) \cdot P(w_1\mid class) \cdot P(w_2\mid class, w_1) \cdot P(w_3, ..., w_n) \\
&= ...
\end{aligned} 
$$

이렇게 문장의 조건부 확률을 앞에 등장한 단어들과 class에 대한 조건부 확률 곱으로 나타낼 수 있다. 여기서 나이브 가정은 특성들 간의 관계를 독립적이라고 가정하므로 임의의 쌍 $i \neq j$에 대해 $P(w_i \mid  class) = P(w_i \mid  class, w_j)$를 만족한다. 따라서 문장의 조건부 확률을 보다 간단하게 표현할 수 있다.

$$
\begin{aligned}
P(class, w_1, ..., w_n) 
&= P(class) \cdot P(w_1\mid class) \cdot ... \cdot P(w_n\mid class) \\
&= P(class) \cdot \prod_{i=1}^{n} P(w_i\mid class)
\end{aligned} 
$$

현실 세계의 많은 현상이 상호 의존적임에도 불구하고, 나이브 베이즈 모델은 오랫동안 효율적이고 효과적인 모델로 활용되어왔다.

> [나이브 베이즈 모델의 효과 분석 자료 보기](https://web.archive.org/web/20171210133853/http://www.research.ibm.com/people/r/rish/papers/RC22230.pdf)

### 🎲 로그값으로 계산하기
확률은 0과 1 사이의 값이므로, 확률을 여러번 곱하면 전산적으로 언더플로우의 위험이 커진다. 너무 큰 값이나 너무 작은 값을 다루는 전형적인 방법은 로그를 사용하는 것이다. 로그를 취한 log likelihood는 log ratio의 합으로 쓸 수 있다.

$$
\begin{aligned}
log\_ ratio(w_i) &= log \frac{P(w_i\mid Pos)}{P(w_i\mid Neg)} \\
log\_ likelihood &= \sum^{m}_{i=1} log \frac{P(w_i\mid Pos)}{P(w_i\mid Neg)}
\end{aligned}
$$

한가지 개념을 추가하자면, log ratio의 값을 lambda 함수로 표현하기도 한다. 즉 $\lambda(w)$를 log ratio로 표현하면 likelihood를 더 간단하게 쓸수 있다.

$$
\begin{aligned}
\lambda(w_i) &= log\frac{P(w_i\mid Pos)}{P(w_i\mid Neg)} \\
log\_ likelihood(s) &= \sum^{n}_{i=1} \lambda(w_i)
\end{aligned}
$$

로그를 취하게 되면 likelihood의 중립 값은 $1$에서 $log 1 = 0$으로 변하게 된다. $0$에서 양의 무한대에 대한 로그값은 음의 무한대에서 양의 무한대이므로 log likelihood의 값의 범위도 $0$을 중립 값으로하는 음의 무한대에서 양의 무한대 값을 반환할 것이다.

### 🎲 사전확률
예를 들어 코로나 팬데믹에 대한 트윗을 모아서 정서 분석을 수행한다면, 부정적인 트윗이 긍정적인 트윗보다 많을 것이다. 현실 데이터 corpus에서 분류 클래스가 균등하게 나눠지는 경우는 드물기 때문에, 데이터의 불균형을 보정하기 위한 절차가 필요하다. 

나이브 베이즈 모델에서는 사전확률이 이 역할을 수행한다. Likelihood와 마찬가지로 사전확률을 class의 비율로 정의하고 로그값을 취할 수 있다. 

$$
log\_prior = log \frac{P(Pos)}{P(Neg)}
$$

어떤 입력값에 대해 log likelihood가 0이라고 하면, 예측값은 log prior와 거의 같을 것이다. 

## 4. 나이브 베이즈 모델
다시 나이브 베이즈 연산을 보면, 입력 문장 $s$에 대해:

$$
NB = log\frac{P(Pos)}{P(Neg)} + \sum^{n}_{i=1} \lambda(w_i)
$$

위의 식으로 계산되며, 

$$
NB = log\_prior + log\_likelihood
$$

로 정리할 수 있다. 나이브 베이즈 모델의 연산의 순서는 다음과 같이 정리할 수 있다. 

0. 훈련 데이터를 전처리 한다.
1. 토큰화 된 단어의 빈도 $freq(w, class)$를 계산한다.
2. 모든 훈련 데이터의 단어에 대해 훈련해 log prior와 log likelihood 값을 구한다.
    - 모든 훈련 데이터의 단어에 대해 $P(w\mid Pos)$와 $P(w\mid Neg)$ 값을 구한다.
    - 모든 훈련 데이터의 단어에 대해 $P(Pos)$와 $P(Neg)$ 값을 구한다.
3. 훈련한 모델의 가중치로 정서를 분류한다.


### 📂 구현하기

#### 1. 토큰화 된 단어의 빈도 $freq(w, class)$를 계산한다.
```python
def get_freq(dd, train_x, train_y):
    '''
    Get frequency dictionary from the training data.
    input:
        dd : a defaultdict of integer.
        train_x : list of tokened sentences of training data.
        train_y : list of 0 or 1 corresponding to the train_x. 
    return:
        result : dictionary of (key, value) = (word label pair, frequency).
    '''
    for label, sentence in zip(train_y, train_x):
        for word in process(sentence):
            dd[(word, label)] += 1

    return dd

# count frequency dictionary from train_x and train_y.
freqs = get_freq(defaultdict(int), train_x, train_y)
```

#### 2. 모든 훈련 데이터의 단어에 대해 훈련해 log prior와 log likelihood 값을 구한다.
```python
def train_naive_bayes(freqs, train_x, train_y):
    '''
    Train Naive Bayes model, that is, get prior and likelihood from the training data.
    return:
        log_prior : an integer. P(Pos) / P(Neg) value.
        log_likelihood : a dictionary of (key, value) = (word, log likelihood)
    '''
    # log_likelihood relies on words
    log_likelihood = {}
    # log prior value relies on the corpus
    log_prior = 0

    # get unique words from the frequency dict
    vocab = list(set(freq.keys()))
    V = len(vocab)

    # get N_pos and N_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if label is 1(> 0), the word is positive.
        if pair[1] > 0:
            N_pos += freqs[pair]
        # if label is 0, the word is negative.
        else:
            N_neg += freqs[pair]

    # get log likelihood
    for w in vocab:
        # get positive and negative frequency of word w.
        freq_pos = freqs.get((w, 1), 0)
        freq_neg = freqs.get((w, 0), 0)

        # get P(w\mid Pos) and P(w\mid Neg).
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        log_likelihood[w] = np.log(p_w_pos) - np.log(p_w_neg)

    # to compute log_prior,
    # get the number of positive and negative labels
    num_label = len(train_y)
    num_pos = len(train_y[train_y == 1])
    num_neg = len(train_y[train_y == 0])

    # log prior = log(P(Pos)) - log(P(Neg))
    log_prior = np.log(num_pos / num_label) - np.log(num_neg / num_label)

    return log_prior, log_likelihood

# get log prior and log likelihood from the training data
# so that we can train on test data.
log_prior, log_likelihood = train_naive_bayes(freqs, train_x, train_y)
```

#### 3. 훈련한 모델의 가중치로 정서를 분류한다.
```python
def predict_naive_bayes(s, log_prior, log_likelihood):
    '''
    input:
        s : a list. Input sentence.
        log_prior : log prior from trained naive bayes.
        log_likelihood : log likelihood from trained naive bayes.
    return:
        log_prob : float between 0 and 1. probability that s is positive.
    '''    
    
    words = proprocess(s)

    log_prob = 0

    for w in words:
        if w in log_likelihood:
            log_prob += log_likelihood[w]

    log_prob += log_prior

    return log_prob

# print probability of test data.
test_data = 'hope you get well soon. it hurts to see you ill 😢'
print('prediction:', predict_naive_bayes(test_data, log_prior, log_likelihood))

# output: 3.5905424260671044 -- 긍정 정서로 예측했다.
```


## 참고 자료
1. Coursera, deeplearning.ai, Natural Language Processing with Classification and Vector Spaces, week 2 
2. Wikipedia, Naive Bayes Classification, https://en.wikipedia.org/wiki/Naive_Bayes_classifier
3. Wikipedia, Additive Smooothing, https://en.wikipedia.org/wiki/Additive_smoothing
4. Wikipedia, Likelihood-ratio test, https://en.wikipedia.org/wiki/Likelihood-ratio_test