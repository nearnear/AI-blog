---
title: "git 치트 시트"
categories: Git
tags:
  - git
  - github
---

**Note:** 본 글은 이전에 작성한 글의 백업입니다.
{: .notice--info}


## 1. 배시 쉘 명령어 (Bash shell commands)
* 디렉토리 .git 내 모든 파일 출력
    ```bash
    # look inside a git directory
    ls -l .git/
    ```
* 숨겨진 파일을 포함해 현재 디렉토리에 위치한 모든 파일 출력
    ```bash
    # list files with a dot (hidden files)
    ls -la
    ```
* 파일 file_name.py 생성하기
    ```bash
    # create a file
    touch file_name.py
    ```
* atom 에디터로 파일 file_name.py 열기
    ```bash
    # open a file with atom editor
    atom file_name.py
    ```
* 파일 file_name.py 내용 쉘에서 보기
    ```bash
    # lookup a file
    cat file_name.py
    ```

## 2. 설정 명령어 (Configuration commands)
* 깃 현재 설정 출력하기 
    ```bash
    # check current configuration
    git config -l
    ```
* 깃의 이름, 이메일 설정 변경하기
    ```bash
    # change configuration
    git config --global user.name "user_name"
    git config --global user.email "user_email"
    ```
* 깃허브 키를 15분 동안 캐시하기
    ```bash
    # cache github key for 15 minutes
    git config --global credential.helper cache
    ```


## 3. 스테이징 & 커밋 명령어 (Staging & Commit commands)
* 현재 디렉토리에서 새로운 깃 르포 생성하기
    ```bash
    # initialize an empty git repository in current directory
    git init
    ```
* 현재 워킹 트리의 정보 출력하기
    ```bash
    # get information of current working tree
    git status
    ```

### 스테이징 에리어 조작하기
* 파일 file_name.py를 스테이징 에리어에 올리기 (커밋 준비)
    ```bash
    # command git to track follwing file
    git add file_name.py
    ```
* 모든 변경 사항 스테이징 에리어에 올리기
    ```bash
    # stage every change in the staging area
    git add .
    ```
* 스테이징 되었지만 커밋되지 않은 파일 출력하기
    ```bash
    # alias to --cached, show all staged but not commited files
    git diff --staged
    ```
* 스테이징 에리어 리셋하기
    ```bash
    # reset(clear) staging area
    git reset HEAD --
    ```
* 파일 file1.py의 이름을 file2.py로 변경하기
    ```bash
    # rename file1 with file2
    # similar to Linux mv command
    git mv [file1.py] [file2.py]
    ```
* 파일 file_name.py 삭제하기
    ```bash
    # remove file_name.py from working space
    # similar to Linux rm command
    git rm [file_name.py]
    ```


### 커밋하기
* 커밋 메시지 창을 열며 현재 스테이징 에리어에 있는 것을 전부 커밋하기
    ```bash
    # commit everything in current staging area
    # opens a text editor to enter a commit message
    git commit
    ```
* 커밋 메시지를 입력하지 않고 커밋하기
    ```bash
    # stage changes to tracked file & commit in one step
    git commit -a
    ```
* 간략한 커밋 메시지 commit message와 함께 커밋하기
    ```bash
    # stage & commit & enter message
    git commit -a -m "commit message"
    ```

### 커밋 진행상황 보기
* 모든 커밋 히스토리 출력하기
    ```bash
    # check history of all commits
    git log
    ```
* 각 커밋에서 수정된 사항을 줄 별로 출력하기
    ```bash
    # show actual lines that changed in each commit
    git log -p
    ```
* 커밋의 통계 정보 출력하기
    ```bash
    # show statistics about the changes in the commit
    git log --stat
    ```
* 커밋의 브랜치 트리 출력하기
    ```bash
    # show commit branch tree
    git log --graph --oneline --all
    ```
* 커밋 아이디 commit_id에 해당하는 커밋 정보 출력하기
    ```bash
    # show the information about the commit and associated petches
    git show [commit_id]
    ```
* commit_ id1과 commit_ id2에 해당하는 두 커밋 비교하기
    ```bash
    # similar to Linux diff command
    git diff [commit_id1] [commit_id2]
    ```

### 커밋 되돌리기
* HEAD가 가리키는 브랜치가 commit_id를 가리키게 하기
    ```bash
    # resets the repo in the Index, the next snapshot to commit
    git reset --soft [commit_id]
    ```
* HEAD 브랜치를 이동하고 스테이징 에리어를 리셋하기
    ```bash
    # update Index to the snapshot that HEAD is pointing 
    git reset --mixed [commit_id]
    ```
* HEAD 브랜치를 이동하고 스테이징 에리어와 워킹 디렉토리를 리셋하기
    ```bash
    # update Index to the snapshot that HEAD is pointing 
    git reset --hard [commit_id]
    ```
* 현재 스테이징 에리어에 있는 내용을 커밋 내용으로 덮어쓰기
    ```bash
    # make changes to commits after-the-fact on local commits
    git commit --amend
    ```
* 히스토리를 유지한채 새로운 커밋으로 커밋 commit_id로 되돌리기
    ```bash
    # make a new commit which rolls back a previous commit
    git revert HEAD/[commit_id]
    ```
* 커밋을 num_commit_to_reverse 개수만큼 되돌리기
    ```bash
    # roll back [num_commit_to_reverse]-many commit
    git reset --soft HEAD~[num_commit_to_reverse]
    ```


## 4. 브랜치 명령어 (Branch commands)
* 모든 브랜치 출력하기
    ```bash
    # list all branches
    git branch
    ```
* 읽기 전용 원격 브랜치 출력하기
    ```bash
    # shows read-only remote branches
    git branch -r
    ```
* 브랜치 branch_name 생성하기
    ```bash
    # creates the branch_name branch
    git branch [branch_name]
    ```
* 브랜치 branch_name으로 이동하기
    ```bash
    # switch to branch_name
    git checkout [branch_name]
    ```
* 브랜치 branch_name을 생성하고 그 브랜치로 위치 이동하기
    ```bash
    # creates a new branch and switches to it
    git checkout -b [branch_name]
    ```
* 브랜치 branch_name 삭제하기
    ```bash
    # deletes the branch branch_name
    git branch -d [branch_name]
    ```
* 브랜치 branch_name 삭제 강제하기
    ```bash
    # forcibly deletes the branch
    git branch -D [branch_name]
    ```
* 브랜치 branch_name을 마스터 브랜치로 합치기
    ```bash
    # joins branches together to the master branch
    git merge [branch_name]
    ```
* 브랜치 충돌(merge conflicts)이 발생했을 때, 머지 취소하기
    ```bash
    # when merge conflicts, abort merge action
    git merge --abort
    ```

## 5. 깃허브 관련 명령어 (Github commands)
* 깃허브는 personal access token을 사용해 계정의 권한을 조절하고 있다. https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

* 깃허브에 push할 때는 파일당 **120MB의 용량 제한**이 있다. 이보다 큰 용량의 파일을 push하면 remote rejected 에러를 마주하게 된다.

* 로컬 환경에 URL로 원격 르포 복제하기
    ```bash
    # clone a remote repository into a local workspace
    git clone [URL] 
    ```
* 로컬 파일을 원격 르포에 푸시하기
    ```bash
    # push commits from local repo to a remote repo
    git push
    ```
* 원격 르포의 커밋을 로컬 환경에 별도의 브랜치로 복사해오기
    ```bash
    # copy the commits done in the remote repository
    git fetch
    ```
* 원격 르포의 커밋을 로컬 환경의 브랜치와 머지하기
    ```bash
    # fetch from remote & merge
    git pull
    ```
* 원격 르포 출력하기
    ```bash
    # List remote repos
    git remote
    ```
* 원격 르포 URL 출력하기
    ```bash
    # show URL of remote repo
    git remote -v
    ```
* 원격 르포 remote_name 정보 출력하기
    ```bash
    # Describes a single remote repo
    git remote show [remote_URL]
    ```
* 원격 르포의 업데이트를 로컬 환경에 불러오기
    ```bash
    # Fetches the most up-to-date objects
    git remote update
    ```
* 로컬 르포가 연결된 원격 르포 new-url로 이전하기
    ```bash
    # transfer a repository from origin to [new-url]
    git remote set-url origin [new-url]
    ```
* 브랜치 branch_name의 베이스 커밋을 바꾸기
    ```bash
    # change the base commit used for the branch [branch_name]
    git rebase [branch_name]
    ```


## 참고 자료
1. Git Docs ([link](https://git-scm.com/doc))
2. GitHub Docs ([link](https://docs.github.com/en))
3. Coursera, Google, Introduction to Git and Github ([link](https://www.coursera.org/learn/introduction-git-github))
