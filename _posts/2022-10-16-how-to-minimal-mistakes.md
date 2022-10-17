---
title: "Minimal Mistakes 사용법"
categories: updates
tags:
    - jekyll
    - minimal-mistakes
---

Minimal Mistakes 테마를 이용하는 방법을 간단히 정리한다. 로컬 서버 실행, 글 작성 포맷, 미디어 첨부 방법 등을 알아본다.

## 로컬서버 실행하기

터미널에서 `bundle exec jekyll serve`로 로컬서버에 연결해, `localhost:4000`에서 preview를 볼 수 있다. (Gatsby의 경우 로컬서버가 8000이었다.)


## 포스트 작성법

포스트는 마크다운으로 작성한다. **필수 요소**는 세가지다.
1. `_posts` 디렉터리에 있어야한다.
2. 제목이 `%Y-%M-%D-filename.md` 형식을 따라야한다.
3. 다음과 같은 front matter를 작성해야 Jekyll이 마크다운을 포스트로 인식하여 `_site`에 html 파일을 생성한다.
    ```md
    ---
    title: "동적 블로그에서 정적 블로그로"
    categories: 업데이트
    tags:
        - Jekyll
        - minimal-mistakes
    ---
    ```

- front matter의 요소는 빠지거나 더할 수 있다.
- `categories:`의 요소는 taxonomy이다. 작성일 기준 가능한 taxonomy는 다음과 같다: 
  `updates`, `recents`, `ml-dl`, `papers`, `mlops`, `cs`, `python`, `git`, `dbms`, `aws`, `crawl`, `math`, `stats`, `information`, `physics`, `planning`
  - `/_data/navigation.uml` 이나 `/_pages`의 각 페이지 마크다운에서 확인할 수 있다.

포스트는 디폴트로 코멘트와 관련 포스트, 사이드바를 표시하도록 설정했다.


## 컬렉션(프로젝트) 작성법

컬렉션은 포스트와 별도로 작성하는 외부 페이지이다. `_config`와 `_pages`에 `projects`라는 컬렉션을 작성해두었다.

프로젝트에 글을 올리려면 `_projects` 디렉터리에 front matter를 포함한 마크다운을 작성하면 된다. 포스트와 같은 제목 형식은 없다. 디폴트로 코멘트, 관련 포스트, 저자 프로필은 없으며, 사이드바는 front matter에서 설정한다.


## front matter 작성법

- 마크다운의 front matter를 통해 디폴트 설정을 오버라이드할 수 있다. 
- 또한 `header:`로 헤더(Notion의 커버)와 티저(미리보기 이미지)를 설정하거나, 
- `gallery:`를 통해 배열된 미디어를 첨부하고, 
- `sidebar:`로 사이드바를 설정할 수 있다.

예시는 다음과 같다.
```md
---
title: "Make some noise"
excerpt: "Waiting for Rock Festival"
search: false
classes: wide
layout: single          
toc: false              
author_profile: false   
comments: false         
share: false            
related: false 
last_modified_at: 2022-10-17T08:06:00-05:00
header:
  image: /assets/images/unsplash-gallery-image-1.jpg
  teaser: assets/images/unsplash-gallery-image-1-th.jpg
sidebar:
  - title: "Role"
    image: http://placehold.it/350x250
    image_alt: "logo"
    text: "Designer, Front-End Developer"
  - title: "Responsibilities"
    text: "Reuters try PR stupid commenters should isn't a business model"
gallery:
  - url: /assets/images/unsplash-gallery-image-1.jpg
    image_path: assets/images/unsplash-gallery-image-1-th.jpg
    alt: "placeholder image 1"
  - url: /assets/images/unsplash-gallery-image-2.jpg
    image_path: assets/images/unsplash-gallery-image-2-th.jpg
    alt: "placeholder image 2"
  - url: /assets/images/unsplash-gallery-image-3.jpg
    image_path: assets/images/unsplash-gallery-image-3-th.jpg
    alt: "placeholder image 3"
---
```
이때 사진 경로는 `/imgs/bio-cup.jpg` 처럼 Jekyll 폴더에서 시작하는 full 경로여야한다.


## 마크다운 설정들

### 버튼 만들기
`.btn` class로 작성한다.
- 옵션: `sucess`, `warning`, `info`, `danger`

```
<a href="#" class="btn btn--success">Success Button</a>
```

<a href="#" class="btn btn--success">Success Button</a>

### 노티스 박스 만들기
노티스 박스는 문단 다음 줄에 `{: .notice}`를 붙여 작성한다.
- 옵션: `info`, `danger`

**Note:** Like this.
{: .notice--info}

**Please Note:** Like this.
{: .notice--danger}

### 이미지 첨부하기
이미지를 첨부하는 방식은 크게 두가지가 있다. `![](url)` 형식을 이용하거나, html의 `<figure>`를 이용하는 것이다.

우선 마크다운 형식으로는 다음과 같이 정렬할 수 있다.
```md
![image-center](/imgs/bio-cup.jpg){: .align-center}
```

`<figure>`를 이용해 다음과 같이 더 깔끔하게(pretty) 업로드가 가능하며, 이미지 정렬과 크기 설정도 가능하다.

```md
<figure style="width: 300px" class="align-left">
  <img src="/imgs/bio-cup.jpg" alt="">
  <figcaption>Image caption.</figcaption>
</figure> 
```

첫번째 방식으로 오른쪽 정렬을, 두번째 방식으로 왼쪽 정렬을 하면 다음과 같다. 

![image-right](/imgs/logo-trans.png){: .align-right}

<figure style="width: 100px" class="align-left">
  <img src="/imgs/logo-trans.png" alt="">
  <figcaption>My logo.</figcaption>
</figure> 

샘플 텍스트: Megalograptus is a genus of eurypterid, an extinct group of aquatic arthropods. Fossils of Megalograptus have been recovered in deposits of Katian (Late Ordovician) age in North America. The genus contains five species: M. alveolatus, M. ohioensis, M. shideleri, M. welchi and M. williamsae, all based on fossil material found in the United States. Fossils unassigned to any particular species have also been found in Canada. 

### 갤러리 배치 및 캡션 달기

갤러리를 추가하기 위해서는 앞서 본 front matter에 `gallery:` 항목을 추가한 후 원하는 위치에 캡션과 함께 다음과 같이 작성하면 된다. 

```liquid
{% raw %}{% include gallery caption="gallery caption" %}{% endraw %}
```

### 유튜브 동영상 첨부하기

`liquid` code block으로 다음과 같이 작성한다:

```liquid
{% raw %}{% include video id="FHUHKIrUH6Y" provider="youtube" %}{% endraw %}
```

{% include video id="FHUHKIrUH6Y" provider="youtube" %}

World is fancy!