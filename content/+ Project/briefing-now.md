---
title: "[진행중] briefing-now"
date: 2023-05-22 13:34:13
subtitle: "💡 AI 기사 브리핑 웹 서비스"
category: "+ Project"
draft: false
---

---
## 실시간 뉴스 브리핑 한국어 서비스
### 기간 : 2023.05 ~ 진행중
### 주요 기능 : 
- 1분 마다 네이버 뉴스의 `분야별 주요 뉴스` 스크레이핑
- HuggingFace API를 통해 뉴스 기사 요약
### 사용 도구 :
- Help of ChatGPT on writing `Javascript` on `Node.js`
- `HuggingFace Model API` from Hub `ainaize/kobart-news`
- `Netlify` on deployment
<!--- `Docker` for container-->
 
<!-- [프로젝트 페이지 가기 🔈](URL)-->

---
<br>


## 프로젝트에 대해

<br>

### 계기
#### Why AI?
ChatGPT의 이슈화로 AI 서비스에 대한 수요와 공급이 증가하고 있는데, 당장 한국어로 활용할 수 있는 어플리케이션은 크게 체감되지 않았으며 또한 AI 기술로 무엇을 할 수 있을지에 대한 고민이 생겼다. 따라서 "Start Cheap"의 모토로 API를 활용해 간단한 AI 서비스를 제공하는 웹 페이지를 구현하고, 다양한 모델을 실험해 보고자 한다. 

#### Why JavaScript?
익숙한 Python으로 FastAPI를 활용해 구현할 것인지 익숙하지 않은 JavaScript를 활용할 것인지 고민이 있었다. 그러던 중 `AI서비스와 함께 누구나 만드는 웹 프로젝트! feat. ChatGPT` 라는 [프로그래머스](programmers.co.kr)에서 제공하는 짧은 강좌를 통해 JavaScript로 API를 활용한 간단한 웹서비스를 만들었다. ChatGPT를 활용하면 JavaScript로 개발이 가능하지 않을까 하는 생각에 이 프로젝트를 JavaScript로 구현하기로 했다.

<br>

### 단계
#### ChatGPT 답을 기반으로 Node.js 환경을 형성하고 JavaScript 코드를 작성했다.
- `axios`와 `cheerio`로 크롤링과 HuggingFace API를 구현했다.
- `iconv-lite`로 `charset=EUC-KR`로 작성된 페이지를 디코딩했다.
- 이때 크롤링과 API를 과하게 요청하는 것을 방지하기 위해 1분 단위로 크롤링을 진행해 전역변수를 업데이트하고, `window.onload` 핸들러로 요청이 있을시에 내용을 전달하는 방식으로 구현했다. 
    - 1분 단위이면 서비스 페이지 접속량에 관계없이 하루에 1440회만 크롤링 및 API를 요청하는데, 네이버 검색 API가 하루 25000회로 제한되어 있는 것을 고려하면 적은 횟수이다.
#### 테스트와 로컬 서버를 구현했다.
- `mocha`로 테스트를 구현했다.
- `express`로 local server를 구현했다.(`local host:3000`) 
        

### 개선 방향
- 음성 생성 모델을 통한 읽어주기 기능 추가
- 이때 시각에 의존하지 않고 기능을 실행할 수 있도록 단축키나 알림음을 추가한다.
- 전통적인 뉴스 미디어 외에 다른 
- Docker를 활용해 컨테이너화
- 해외 뉴스 크롤링
