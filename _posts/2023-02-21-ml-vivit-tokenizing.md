---
title: "[Vision] ViViT의 토큰화 방법 비교하기"
categories: ML
tags:
    - transformer
    - vivit
---

비디오 트랜스포머인 ViViT의 토큰화 방법 두가지를 구현하고 비교해보자.

비디오 같은 3D 비전 데이터셋은 2D 이미지 데이터셋과 유사하지만 시간 차원을 하나 더 가지고 있다. 이미지 데이터를 트랜스포머로 다루는 ViT에서 토큰화 방법이 중요했던 만큼, ViViT에서도 토큰화 방법이 중요한 영향을 끼치지 않을까하는 질문에서 시작해 논문 [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)에 소개된 두가지 토큰화 방법을 비교해보기로 했다. 실험은 각각의 토큰화 방법을 활용한 모델을 같은 데이터셋과 하이퍼파라미터로 학습을 했을 때 수렴하는 여부, 학습과 검증의 accuracy와 loss, 그리고 학습 시간의 차이를 비교했다. [MedMNIST3D](https://medmnist.com/)의 6개 데이터셋에 대해 실험했으며, 사전 학습은 진행하지 않았고, 결과의 시각화를 위해 TensorBoard를 사용했다. 전체 코드는 [깃허브](https://github.com/nearnear/vision-studies/blob/main/ViViT/Tokenization_comparison_in_ViViT.ipynb)에서 볼 수 있다.
{: .notice--info}


## 1. Uniform Frame Sampling

<figure style="width: 500px" class="align-center">
	<a href="/imgs/post-imgs/vivit-ufs.png"><img src="/imgs/post-imgs/vivit-ufs.png"></a>
	<figcaption>Uniform Frame Sampling, from the paper.
</figcaption>
</figure>

Uniform Frame Sampling은 비디오에서 시간에 따른 $n_t$개의 2D 프레임을 샘플링한 뒤 2D 합성곱을 실행해서 시간 순으로 쌓는 방법이다 (3D 데이터의 경우 임의의 축을 시간축으로 생각할 수 있다). $n_t = \lfloor \frac{T}{\text{patch size}} \rfloor < T$ 이므로 샘플링 되지 않거나 중복되는 정보가 있을 수 있다. 코드는 다음과 같다.

```python
class UniformFrameSampling(layers.Layer):
    def __init__(self, embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection2d = layers.Conv2D(
            filters=embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="valid"
        )
        self.concat = layers.Concatenate(axis=1)
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        t = videos.shape[1]
        projected_patches = []
        for frame in range(self.patch_size, t, self.patch_size): # Sample n_t
            patch = self.projection2d(videos[:, frame])  # (B, n_h, n_w, embed_dim)
            _, n_h, n_w, embed_dim = patch.shape
            patch = layers.Reshape((-1, 1, n_h, n_w, embed_dim))(patch)
            projected_patches.append(patch)
        projected_patches = self.concat(projected_patches)  # (B, n_t, n_h, n_w, embed_dim)
        flattened_patches = self.flatten(projected_patches)  # (B, num_patches, embed_dim)
        return flattened_patches
```

코드 주석에 차원 변화를 기록했다. 


## 2. Tubelet Embedding

<figure style="width: 500px" class="align-center">
	<a href="/imgs/post-imgs/vivit-tubelet-embedding.png"><img src="/imgs/post-imgs/vivit-tubelet-embedding.png"></a>
	<figcaption>Tubelet Embedding, from the paper.
</figcaption>
</figure>

앞선 방법과 달리 Tubelet Embedding은 3D 데이터를 그대로 3D 합성곱으로 토큰화하므로 모든 프레임을 임베딩한다. 또한 그림에 묘사된 것처럼 시간에 대해 선형적으로 프레임의 토큰을 묶기 때문에 시공간 정보를 포함할 수 있다. 이때 패치 크기가 클 수록 하나의 Tubelet(관)은 더 많은 정보를 포함할 것이다. 데이터의 차원을 고려하면 3D 합성곱을 활용하는 방법이 더 직관적으로 여겨지며, 코드도 더 간결하다.

```python
class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.projection3d = layers.Conv3D(
            filters=embed_dim,
            kernel_size=(patch_size, patch_size, patch_size),
            strides=(patch_size, patch_size, patch_size),
            padding="valid"
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):  # videos.shape = (B, T, H, W, C)
        projected_patches = self.projection3d(videos)  # (B, n_t, n_h, n_w, embed_dim)
        flattened_patches = self.flatten(projected_patches)  # (B, num_patches, embed_dim)
        return flattened_patches
```

두 토큰화 방법의 결과 차원이 같은 것을 확인할 수 있다.


## 3. 비교

임베딩의 차이만 있을 뿐, Positional Encoding은 두 임베딩 이후 동일하게 적용했다. 그 후 ViViT Classifier를 정의해 토큰화 방법을 실험하였다.

<figure style="width: 600px" class="align-center">
	<a href="/imgs/post-imgs/vivit-result-nodule.png"><img src="/imgs/post-imgs/vivit-result-nodule.png"></a>
	<figcaption>Training and validation metrics.
</figcaption>
</figure>

<figure style="width: 600px" class="align-center">
	<a href="/imgs/post-imgs/vivit-result-organ.png"><img src="/imgs/post-imgs/vivit-result-organ.png"></a>
	<figcaption>Training and validation metrics.
</figcaption>
</figure>

Nodule MNIST 3D와 Organ MNIST 3D 데이터셋에 대해, Tubelet Embedding은 검증 정확도와 손실에서 더 나은 결과를 보여줬다. 결과적으로 다양한 데이터셋에서 Tubelet Embedding 방법이 Unifrom Frame Sampling 보다 같은 시간(epoch) 대비 더 나은 학습 성과를 보였다.

<figure style="width: 600px" class="align-center">
	<a href="/imgs/post-imgs/vivit-result-adrenal.png"><img src="/imgs/post-imgs/vivit-result-adrenal.png"></a>
	<figcaption>Same hyperparameter, different results.
</figcaption>
</figure>

반면 Adrenal MNIST 3D 데이터셋에서는 같은 learning rate에서도 Uniform Frame Sampling은 검증이 수렴하였으나 Tubelet Embedding은 발산하는 결과를 보였다. 따라서 두 토큰화 방법에 일괄적으로 학습이나 모델의 hyperparameter를 설정하여 결과를 비교하기 어렵다는 것을 알 수 있었다. 

이때 Uniform Frame Sampling은 일부 정보를 누락하거나 중복하기 때문에 학습 속도가 비교적 빠르지 않을까 생각할 수 있다. 즉 더 적은 정보를 학습한다면 학습 속도가 더 빠를 것이라는 가설을 세웠다. 

<figure style="width: 700px" class="align-center">
	<a href="/imgs/post-imgs/vivit-train-time.png"><img src="/imgs/post-imgs/vivit-train-time.png"></a>
	<figcaption>Training time comparison.
</figcaption>
</figure>

그러나 데이터셋과 learning rate에 따라 임베딩 간의 학습 속도 차이는 크지 않았으며 여러 데이터셋과 learning rate에 대해 유의하지 않았다. 즉 두 임베딩에 따른 학습량의 차이는 크지 않은 것으로 보인다. 

전체 결과 그래프는 [Tensorboard.dev](https://tensorboard.dev/experiment/PKs2SEeNQLO68B4tB7U8tg)에서 확인할 수 있다.
{:.notice}


## 4. 결론

결론적으로 작은 규모의 비디오 데이터에 대한 학습과 검증 메트릭에서 일반적으로 Tubelet Embedding이 Uniform Frame Sampling보다 더 나은 결과를 보여주었다. 두 토큰화 방법 중 한쪽이 더 빠른 학습 속도를 보이지는 않았다. 그러나 AdrenalMNIST3D 데이터에 대한 결과가 나타내듯이, 두 임베딩 방법을 엄밀하게 같은 조건에서 비교할 수는 없었다. 또한 작은 규모의 3D 이미지에 대해 사전 학습 없이 작은 Transformer로 분석한 결과라는 점도 염두에 두어야 할 것이다. 
{: .notice--info}

이 실험을 통해 실제 학습 결과를 분석하는 것이 흥미로운 한편, 엄밀한 의미에서 두개의 모델을 비교하는 것이 얼마나 어려운 일인지 알 수 있었다.