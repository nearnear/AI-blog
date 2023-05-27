---
title: "[iconv] decode()-ing 및 Buffer() deprecation"
date: 2023-05-21 04:49:13
subtitle: "Iconv-lite warning: decode()-ing strings is deprecated."
category: "JS"
draft: false
---

## 요약 : javascript string을 두번 디코딩하지 말자.

<br>

다음과 같은 warning이 발생하며 디코딩이 작동하지 않는다.

```bash
Iconv-lite warning: decode()-ing strings is deprecated. Refer to https://github.com/ashtuchkin/iconv-lite/wiki/Use-Buffers-when-decoding
```

상황은 axios와 iconv로 스크레이핑을 하는 다음 코드에서 발생했다. `response` 버퍼에 저장된 값을 불러와 `EUC-KR`로 디코딩을 시도했다.
- Node.js v18.16.0

```Js
// 웹 페이지에 GET 요청 보내기
const response = await axios.get(newsUrl);

// Cheerio를 사용하여 HTML 파싱
const html = iconv.decode(response.data, 'EUC-KR');
const $ = cheerio.load(html);
```

<br>

### 원인은 이중 디코딩

로그에서 주어진 링크로 들어가면 deprecation 원인이 설명되어 있다. 이미 저장된 string을 불러와 디코딩하는 경우, 처음 string 저장을 할 때 이미 JavaScript에서 `utf-8`으로 디코딩을 **자동**으로 진행하기 때문에 `iconv.decode()`가 호출되면 이중으로 디코딩을 요청하게 된다. 이때 이중 디코딩은 잘못된 디코딩 결과를 내기도 하며, `utf-8` 디코딩은 정보를 손실하기 때문에 정확한 디코딩이 이루어질 수 없다. 즉 이 warning을 무시할 수 있지만, 대부분의 경우 의도한 디코딩 효과를 얻을 수 없으므로 해결하는 것이 좋다.

<br>

### 해결방법은 `Buffer.from()`, `Buffer.concat()`, `Buffer.allocUnsafe()` 를 활용하는 것이다.
### 즉, raw data를 가져와 디코딩을 진행하자.

`Buffer.from()`은 변수가 저장된 버퍼에 접근하여 디코딩 되지 않은 Node.js의 binary 데이터에 접근할 수 있게 한다.

* 이때 `Buffer()`는 보안 이슈로 인해 역시 deprecated 되었으므로 위의 세 함수를 활용하자.

수정한 코드는 다음과 같으며 warning을 제거할 수 있었다.

```Js
// 웹 페이지에 GET 요청 보내기
const response = await axios.get(newsUrl);

// Cheerio를 사용하여 HTML 파싱
const html = iconv.decode(Buffer.from(response.data), 'EUC-KR');
const $ = cheerio.load(html);
```
