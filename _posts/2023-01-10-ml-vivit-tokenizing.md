---
title: "ViViT의 토큰화 방법 비교하기"
categories: ML
tags:
    - transformer
    - vivit
---

비디오 트랜스포머인 ViViT의 토큰화 방법 두가지를 구현하고 비교해보자.

3D 비전 데이터셋(대표적으로 비디오)은 2D 이미지 데이터셋과 유사하지만 시간 차원을 하나 더 가지고 있다. 이미지 데이터를 트랜스포머로 다루기 위해 토큰화 방법이 중요했던 만큼, 3D 비전 데이터셋에서도 토큰화 방법이 성능에 중요한 영향을 끼치지 않을까하는 가설을 세우고 논문 [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)에 소개된 두가지 토큰화 방법을 비교해보기로 했다. 실험은 각각의 토큰화 방법을 활용한 모델을 같은 데이터셋과 하이퍼파라미터로 학습을 했을 때 수렴하는 정도와 시간의 차이를 비교했다. [MedMNIST3D](https://medmnist.com/)의 6개 데이터셋에 대해 실험했으며, 사전 학습은 진행하지 않았고, 결과의 시각화를 위해 TensorBoard를 사용했다. 결과적으로 . 사용된 데이터셋은 모두 신체 조직에 대한 3D 이미지로 데이터의 크기, 길이, 종류에 있어서 유사하므로 결과를 일반화 할 수 없다는 점을 염두에 두자. 전체 코드는 [깃허브]()에서 볼 수 있다.
{: .notice--info}


## 1. Uniform Frame Sampling

<figure>
	<a href="/imgs/post-imgs/vivit-ufs.png"><img src="/imgs/post-imgs/vivit-ufs.png"></a>
	<figcaption>Uniform Frame Sampling, from the paper.
</figcaption>
</figure>

Uniform Frame Sampling은 3D 데이터에서 시간에 따른 (또는 3D 이미지라면 랜덤한 축에 대해) $n_t$개의 2D 데이터를 샘플링한 뒤 합성곱을 실행해서 시간 순으로 (또는 앞서 고른 축의 임의의 방향에 대해) 쌓는 방법이다. $n_t = \lfloor \frac{T}{\text{patch size}} \rfloor < T$ 이므로 샘플링 되지 않은 정보는 유실된다. 코드는 다음과 같다.

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

<figure>
	<a href="/imgs/post-imgs/vivit-tubelet-embedding.png"><img src="/imgs/post-imgs/vivit-tubelet-embedding.png"></a>
	<figcaption>Tubelet Embedding, from the paper.
</figcaption>
</figure>

앞선 방법과 달리 Tubelet Embedding은 3D 데이터를 그대로 3D 합성곱으로 토큰화하므로 (모든 프레임을 임베딩한다) 시간 데이터를 토큰 안에 포함할 수 있다. 이때 패치 크기가 클 수록 Tubelet(관)에 더 많은 정보를 포함할 것이다. 데이터의 차원을 고려하면 3D 합성곱을 활용하는 이 방법이 더 직관적이라고 생각되며, 코드도 더 간결하다.

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

<figure>
	<a href="/imgs/post-imgs/vivit-result-nodule.png"><img src="/imgs/post-imgs/vivit-result-nodule.png"></a>
	<figcaption>Training and validation metrics.
</figcaption>
</figure>

<figure>
	<a href="/imgs/post-imgs/vivit-result-organ.png"><img src="/imgs/post-imgs/vivit-result-organ.png"></a>
	<figcaption>Training and validation metrics.
</figcaption>
</figure>

결과적으로 다양한 데이터셋에서 Tubelet Embedding 방법이 Unifrom Frame Sampling 보다 학습시에 더 빨리 수렴하고 validation epoch 또한 더 빨리 감소하였다. 

이때 Uniform Frame Sampling은 일부 정보를 누락하기 때문에 당연히 정보의 손실이 적은 Tubelet Embedding이 학습에 유리하지 않을까 생각할 수 있다. 이에 대한 검증을 위해 Sampling 개수를 늘릴 수 있지만, 이렇게 되면 전체 차원에 영향을 미쳐 Tubelet Embedding의 차원도 변화하게 되므로 최적의 학습이 불가능하다고 생각했다. 한편 더 적은 정보를 학습한다면 그만큼 학습이 빠를 것이라는 가설을 세워 결과를 비교하였다.

<figure>
	<a href="/imgs/post-imgs/vivit-train-time.png"><img src="/imgs/post-imgs/vivit-train-time.png"></a>
	<figcaption>Training time comparison.
</figcaption>
</figure>

그러나 데이터셋과 learning rate에 따라 임베딩 간의 학습 속도 차이는 유의미하지 않았다. 즉 두 임베딩에 따른 학습량의 차이는 크지 않았다. 

<figure>
	<a href="/imgs/post-imgs/vivit-result-adrenal.png"><img src="/imgs/post-imgs/vivit-result-adrenal.png"></a>
	<figcaption>Same hyperparameter, different results.
</figcaption>
</figure>

한편 두 임베딩 방법에 같은 하이퍼파라미터를 적용했을 때에도 한쪽은 수렴하고 반대쪽은 발산하는 결과가 나타났다. 임베딩 방법에 따른 하이퍼파라미터 조정이 필요함을 알 수 있으며, 따라서 엄밀히 말해 `같은 조건` 아래에서 두 임베딩 방법을 비교할 수 없었다.

## 4. 결론

결론적으로 Unifrom Frame Sampling이 Tubelet Embedding 보다 일반적으로 더 빠른 학습 속도를 보이지는 않으며, 작은 규모의 비디오 데이터에 대한 학습과 검증 메트릭에서 일반적으로 Tubelet Embedding이 더 나은 결과를 보여주었다. 그러나 AdrenalMNIST3D 데이터에 대한 결과가 나타내듯이, 두 임베딩 방법을 엄밀하게 같은 조건에서 비교할 수는 없었다. 또한 작은 규모의 3D 이미지에 대해 사전 학습 없이 분석한 결과라는 점도 염두에 두어야 할 것이다. 

이 실험을 통해 실제 학습 결과를 분석하는 것이 흥미로운 한편, 엄밀한 의미에서 두개의 모델을 비교하는 것이 얼마나 어려운 일인지 알 수 있었다.