---
title: "[플러그인] Gatsby Katex 설치"
date: 2023-05-21 13:11:13
subtitle: "gatsby-remark-katex"
category: "JS"
draft: false
---

### [gatsby-remark-katex](https://www.gatsbyjs.org/packages/gatsby-remark-katex)를 설치했다.

$$
\text{Blog} \xrightarrow{\text{PlugIn } \in \text{ Easy}} \text{Blog with PlugIn}
$$

<br>

#### 먼저, `gatsby-remark-katex`와 `katex`를 npm으로 다운받는다.

```bash
$ npm install --save gatsby-remark-katex katex
```

#### 이제 `gatsby-config.js` 파일에 플러그인을 추가한다.

```JS
module.exports = {
  plugins: [
    {
      resolve: `gatsby-transformer-remark`,
      options: {
        plugins: [
          
          {
            resolve: `gatsby-remark-katex`,
            options: {
              // Add any KaTeX options from https://github.com/KaTeX/KaTeX/blob/master/docs/options.md here
              strict: `ignore`,
            },
          },
        ],
      },
    },
  ],
}
```

### 마지막으로 `/src/templates/blog-post.js`에 다음 코드를 추가한다.
이 경로는 Gatsby 폴더 구조에 따라 조금 다를 것이다.

```
// reference katex.min.css
import "katex/dist/katex.min.css";
```

<br>

Gatsby의 플러그인 설치는 간단한 것이 장점이라고 느꼈다. 

