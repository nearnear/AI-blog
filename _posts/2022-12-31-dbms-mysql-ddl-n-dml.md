---
title: "[MySQL] DDL과 DML"
categories: DBMS
tags:
---

**Note:** 본 글은 이전에 작성한 글의 백업입니다.
{: .notice}

## DDL
*DDL(Data Definition Language)*

테이블 자체를 추가/삭제/변형한다.

### CREATE TABLE

- `CREATE TABLE {테이블 이름} ({칼럼1}{칼럼1 자료형}{칼럼1 옵션}, ..., {칼럼N}{칼럼N 자료형}{칼럼N 옵션})` : 테이블 생성

```sql
CREATE TABLE table1 (
    col1 INT AUTO_INCREMENT PRIMARY KEY,
    col2 VARCHAR(10) NOT NULL,
    col3 VARCHAR(10) UNIQUE NOT NULL,
    col5 INT UNSIGNED,
    col6 INT DEFAULT 0
)
```
- 옵션은 생략할 수 있다.
- `AUTO_INCREMENT`는 행을 생성 할 때마다 자동으로 1씩 증가하는 칼럼 옵션이다.
- `UNIQUE`는 중복 입력만 불가한 칼럼 옵션이다.
- `NOT NULL`은 NULL 값만 입력 불가한 칼럼 옵션이다.
- `PRIMARY KEY`는 중복 입력과 NULL 값 입력이 불가한 칼럼 옵션이다. `UNIQUE`와 `NOT NULL`을 합한 기능을 제공한다.
- `UNSIGNED`는 숫자 자료형에만 적용 가능하며, 양수만 입력받는 칼럼 옵션이다.

### DROP TABLE

- `DROP TABLE {테이블 이름}` : 테이블 삭제

```sql
DROP TABLE table1
```

### ALTER TABLE

- `ALTER TABLE {테이블 이름} RENAME TO {바꿀 이름}` : 테이블 이름 변경

```sql
ALTER TABLE table1 RENAME TO better_table
```

칼럼 이름을 통해 칼럼에 접근해 자료형을 수정할 수 있다.
- `ALTER COLUMN {칼럼 이름} {바꿀 이름} {바꿀 자료형}` : 칼럼 이름과 자료형 변경
- `ADD COLUMN {추가할 칼럼 이름} {자료형} {옵션}` : 칼럼 추가
- `DROP COLUMN {칼럼 이름}` : 칼럼 삭제

```sql
ALTER COLUMN col1 col1 INT,
ALTER COLUMN col2 cool_col VARCHAR(10), 
DROP COLUMN col3,
ADD COLUMN col1_2 INT AFTER col1
```
- 복수의 명령어들을 `,`로 구분해 한번에 실행할 수 있다.


## DML

*DML : Data Manipulation Language*

테이블의 행 및 칼럼을 추가/삭제/변형한다.

### INSERT

- `INSERT INTO {테이블 이름} ({칼럼1, 칼럼2, 칼럼3}) VALUES ({칼럼1 값, 칼럼2 값, 칼럼3 값})` : 행 추가

```sql
INSERT INTO table1 ({col1, col2, col3})
     VALUES ({val1, val2, val3})
```

### DELETE

특정 조건에 해당하는 데이터를 삭제하기 위해서는 
- `DELETE FROM {테이블 이름} WHERE {조건문}` : 조건에 해당하는 행 삭제
- `TRUNCATE {테이블 이름}` : 테이블 초기화

```sql
DELETE FROM table1
      WHERE ISNULL(col1)
```

### UPDATE

테이블 내의 데이터들을 특정 조건으로 필터링한 뒤에 필터링 된 데이터만 바꾸고 싶은 경우가 있다.
- `UPDATE {테이블 이름} SET {변경 사항들} WHERE {조건문}` : 조건에 해당하는 칼럼을 수정

```sql
UPDATE table1
SET 
    col1 = 'changed',
    col2 = 2
WHERE col3 IN (
        SELECT col1
        FROM table2
    )
```

단, `WHERE`로 시작하는 조건문 없이, 모든 행을 변경할 수는 없다. 



