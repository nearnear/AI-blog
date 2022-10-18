---
title: "commit 되돌리기"
categories: git
tags:
    - git
---

## 발단

궁금증은 블로그를 개편하면서(2022-10-16) 발생했다. 

```
	  A---B---C master, dev (o)
	 /
    D---E---F---G origin (x)
```
디폴트 세팅으로 origin은 remote를, master는 local을 가리킨다.

위와 같은 상황에서 C 커밋을 그대로 remote로 push하고 싶었다. 즉:
- origin을 clone해서 `git reset --soft ...`를 통해 master가 이전 커밋 D 를 가리키게 했다.
- 그 후 master에서 dev 브랜치를 만들어 작업하고, master를 C 커밋으로 이동시켰다.

pull request를 하는 일반적인 상황에서 remote와 local이 branching된 경우 `git merge`를 통해 두 브랜치를 합치려하겠지만(이에 따라 conflict가 발생할 수 있다), 이번 상황은 remote에 잘못 push한 commmit을 되돌려 놓는 것이 목표였다. 강제로 푸싱하는 `git push -f origin master` 명령어로 상황을 해결할 수 있었지만, `soft` 대신 `hard`나 `merged` 모드를 사용하면 뭐가 달라졌을까 하는 의문이 생겼다. 또는 `git revert`를 사용했다면? 이외에는 어떤 방법이 있을까? 


## 1. HEAD 이동하기: git reset과 mode
[문서](https://git-scm.com/docs/git-reset)에 따르면 `git reset [mode] [commit]` 커맨드의 [mode]에는 6개의 옵션이 있는데, 이중 `--soft`, `--mixed`, `--hard`, `--merge` 네가지를 살펴보자. 이하 옵션은 모두 입력한 [commit]으로 head를 이동시킨다. 이들의 차이점은 커밋 index와 working tree의 변경 유무에 있다. 
- 또는 `git reset [mode] HEAD~[reset times]`로 커밋을 횟수 단위로 되돌릴 수도 있다.

| reset | --soft | --mixed | --hard | --merge |
|:---:|:---:|:---:|:---:|:---:|
| index | X | R | R | R |
| working tree | X | X | R | M |

1. `--soft` 옵션은 index를 그대로 두고 working tree를 유지한다. 따라서 head가 이동해도 커밋되지 않고 add된 변경사항들을 그대로 유지하게 된다.
2. `--mixed` 옵션은 index를 리셋하고 working tree를 유지한다. 디폴트 옵션이다. 
3. `--hard` 옵션은 index를 리셋하고 working tree 또한 비운다. 즉 commit이나 add가 되지 않은 작업은 모두 소실된다. 즉 취소한 커밋의 history가 남지 않고, git에서 track하지 않는 파일을 제외하면 [commit]을 막 실행한 상태와 동일해진다.
4. `--merge` 옵션은 index를 리셋하고 working tree는 merge한다. 즉 add 되어있지 않은 작업은 보존하고, index에 있으면서 커밋되지 않은 작업은 버린다. merge이기 때문에, 만약에 현재 커밋과 이동하려는 커밋 사이의 변화된 파일이 add(또는 staging) 되지 않은 작업이 있는 경우 aborted된다. 

### 함께쓰기 1: git restore --source [commit]
git restore는 문자 그대로 커밋의 내용을 working tree에 저장하는 명령어이다. 커밋을 되돌리고자 할때 특정 커밋의 변경 사항을 저장할 때 사용하며, branch를 변경시키지 않는다.  

다음과 같은 상황에서 dev의 분기점을 E로 변경하되, B와 C 커밋은 변경할 필요가 없다고 하자.

```
	  A---B---C dev
	 /
    D---E---F---G master
```

그러면 restore로 B와 C의 커밋을 working tree에 저장한 후 dev를 D로 리셋한 다음, E로 이동해 working tree의 내용을 커밋하면 된다.

코드로 풀어보면 다음과 같다.

```
$ git restore --source B  
$ git reset --soft E
$ git commit ...
```

dev 브랜치에서 작업한다. 우선 restore로 working tree에 B 부터의 변화 저장한다. 둘째로 reset으로 dev가 E를 가리키게 하며, --soft 옵션으로 working tree는 유지한다. 그 후 C에서 가지고 있던 working tree의 내용으로 커밋을 진행한다.

### 함께쓰기 2: git clean -f
앞서 `git reset --hard ...` 명령어가 untracked 파일은 취소하지 않는다고 했다. working tree에 있지만 track되지 않은 파일을 확인하고 삭제하는 방법은 다음과 같다. 

```
$ git reset --hard ...
$ git clean -n
$ git clean -f
```

위의 코드를 실행하면 잘못된 커밋을 하기 이전의 상태와 정확히 같은 상태가 된다.


## 2. 커밋 되돌리기: git revert
`git revert [commit]`는 생성된 커밋에 반대되는 새로운 커밋을 생성한다. 결론부터 말하면 이 글의 발단처럼 remote의 커밋보다 이전 커밋으로 돌아가는 경우 revert를 활용하면 force push를 하지 않아도 커밋을 되돌릴 수 있다. 즉 log tree는 다음과 같은 형태를 띈다.

<figure>
	<a href="/imgs/post-imgs/revert-commit.png"><img src="/imgs/post-imgs/revert-commit.png"></a>
	<figcaption>Log after reverting commit.</figcaption>
</figure>


## 결론

결론적으로 발단에 가장 적합한 커맨드는 `git revert ...` 이다. 처음 git을 배우고난 후 활용하면서 명령어의 mode나 명령어 사이의 차이점을 잊어서 필요할때마다 찾아보면서 구멍을 메꾸는 것이 필요하다는 것을 알았다.