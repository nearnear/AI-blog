---
title: "Gatsby 블로그에서 Jekyll 블로그로"
categories: Recents
tags:
    - jekyll
    - minimal-mistakes
---

## 블로그 이사의 이유

그동안 Gatsby 기반의 깃허브 블로그를 사용하였다. 포스트 작성에 중점을 둔 동적 웹사이트였다. 그렇지만 velog와 유사한 API가 기존 글보다도 업데이트 내용을 강조해 글이 고정되어 있는(fixed) 느낌이 들지 않았다.

마침 2021년 5월부터 Github의 마크다운에 latex 렌더링이 지원된다는 소식을 접했다. 또한 기존에는 최대한 완성도 있게 쓰고자 몇일간 글을 작성했지만, 'TIL'이란 프로젝트를 접하고 짧은 단위의 글을 자주 푸쉬하는 방식으로 학습을 부스팅하고 싶었다. 따라서 한동안 깃허브의 `WIL`(What I Learned)라는 르포에 배운 것들을 기록했다. 그렇지만 (예상가능한 대로) 가독성이 떨어지고 페이지 탐색이 매끄럽지 않았다.

그럼에도 불구하고 기록 자체는 학습의 흔적이자 재구성이기 때문에 의미가 있다. 구글 검색에서 접하는 멋진 블로그들을 둘러보던 끝에, 학습을 기록하는데에 **정적 웹사이트**로 체계를 만드는 것이 적합하다고 판단했다. 


## 만들어본 결과..

그리고 시험적으로 Jekyll에 기반한 Minimal-Mistakes 테마로 사이트를 만들어본 결과... Gatsby에 비해 커스터마이징이 더 간단했다. 당장 떠오르는 이유는:
- Gatsby는 오픈소스 플러그인을 자유롭게 사용할 수 있지만 Jekyll로 깃허브 페이지를 생성하는데는 플러그인이 제한되어 있다.
- 깃허브 페이지는 Jekyll로 작동하고, 내가 사용하고자하는 목적으로 오래동안 사용되어와서 문서화와 에러 대응이 잘 되어있다.

이전에도 오픈소스 테마를 사용했지만 GraphQL등 새로 익힐 것들이 있었다. 제대로 공부할 시간이 있으면 좋겠지만, 머신러닝이 목적이므로 깊게 익히지는 않았다. 그에 비해 Jekyll은 이하의 문서들을 참고해서 초보자도 쉽게 만들 수 있었다. 특히 첫번째 소스에서 커스터마이징을 위한 대부분의 자료를 얻을 수 있었다.

- [minimal-mistakes 문서](https://mmistakes.github.io/minimal-mistakes/docs/quick-start-guide/)
- [minimal-mistakes 깃허브](https://github.com/mmistakes/minimal-mistakes)
- [jekyll 문서](https://jekyllrb.com/docs/)


## 앞으로의 업데이트 방향

이 블로그를 시작한 목적은 배운 것을 체계적으로 모아서 정리하는데에 있으므로, 우선 기존의 포스트를 백업해야 한다. 이전의 블로그와 WIL 르포외에도 갈 곳이 없어 떠도는 마크다운 파일들을 모을 예정이다.

또한 latex 렌더링을 고려하여 포스트를 업데이트 할 것이다. 