---
title: "FastAPI 문서 번역하기"
categories: news
tags:
    - pr
---

토이 프로젝트를 웹에 배포하기 위해 FastAPI를 사용하고자 한다. 마침 공식 문서에
번역이 덜 된 부분이 있어서 [도커로 배포하기](https://fastapi.tiangolo.com/ko/deployment/docker/) 
부분을 번역해 [PR](https://github.com/tiangolo/fastapi/pull/5657)을 진행중이다.
이 글은 다음 번역을 위해 작업 과정을 기록한 것이다.


## 작업 방법

### 1. 번역할 문서 찾기
- [한국어 번역 이슈](https://github.com/tiangolo/fastapi/issues/2017) ([FEATURE] 
  Korean translations #2017)에서 번역 관습을 볼 수 있다.
- [한국어 번역 진행 내역 디스커션](https://github.com/tiangolo/fastapi/discussions/3167)
  에서 PR 진행이 되지 않은 번역 문서를 찾는다.

### 2. 가상 환경 설정하기

#### 가상 환경 설정
- 가상 환경을 설정하여 FastAPI 의존 라이브러리를 설치한다.

#### 번역하기
- [번역을 위한 공식 문서](https://fastapi.tiangolo.com/contributing/#translations) 참고

- `python ./scripts/docs.py live ko` 명령으로 로컬 환경 시작한다.
- `docs/en/docs/번역할 파일.md` 파일을 `docs/ko/docs/번역할 파일.md` 로 복사하여 번역한다.
- `docs/en/mkdocs.yml`을 참고하여 `docs/ko/mkdocs.yml`을 편집한다.

#### 미리보기
- `python ./scripts/docs.py live` 명령으로 미리보기. 오직 한국어로 번역된 문서만 나타난다. 
- `python ./scripts/docs.py build-all` 명령을 통해 번역된 문서는 한국어로, 나머지 문서는 영어로 볼 수 
  있다.


### 3. PR 생성
- `tiangolo/fastapi`에 PR을 생성한다. 제목 형식은 Add Korean translation for 
  `docs/ko/docs/development/docker.md`와 같이 작성한다.
- PR 후 [여기](https://github.com/tiangolo/fastapi/discussions/3167#discussioncomment-3335089)에 커멘트를 남겨 
  한국어 중복 번역을 방지할 수 있다.

  
