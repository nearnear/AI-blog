---
title: "[MySQL] Basics"
categories: DBMS
tags:
---

**Note:** 본 글은 이전에 작성한 글의 백업입니다.
{: .notice}

- [여기](https://www.w3schools.com/mysql/trymysql.asp?filename=trysql_select_all)서 웹으로 SQL 쿼리를 실행해 볼 수 있다.
- SQL 쿼리는 대문자와 소문자를 구분하지 않는다. 즉, 쿼리를 대문자로 쓰던 소문자로 쓰던 섞어서 쓰던 같은 결과를 도출한다. 아래에서 다루겠지만, 함수뿐만이 아닌 문자열 데이터도 대문자와 소문자를 구분하지 않는다.
- SQL 쿼리의 한 줄 주석은 `-- {주석}` 형태로 쓴다.


## 기본 사용법
데이터베이스에서 테이블의 행을 검색하기 위한 가장 기본적인 함수는 `SELECT {칼럼(column)의 이름, 또는 불러올 값}`이며 칼럼을 검색하는 경우 뒤에 `FROM {테이블}`로 칼럼이 속한 테이블 이름을 명시한다.

```sql
SELECT 
    col1 AS First, 
    col2, 3, 'sql', NULL
FROM table1
```

- 전체 행을 선택하기 위해 `SELECT *` 문을 쓸 수 있다.
- `SELECT` 뒤에는 상수, 문자열, NULL 값을 쓸 수 있다.
- 검색 결과에 가져올 칼럼의 이름을 바꾸려면 `SELECT {본래 이름} AS {바꿀 이름}`을 활용한다.


나아가 `WHERE {행 조건}`으로 검색할 행의 조건을 지정할 수 있으며 `ORDER BY {정렬 기준}`으로 행을 정렬할 수 있다. 또한 `LIMIT {가져올 갯수}`를 통해 가져올 행의 개수를 설정할 수 있다.

```sql
SELECT col1, col2, col3
FROM table1
WHERE condition1
ORDER BY sort_standard1
LIMIT integer1
```

- `ORDER BY`는 디폴트로 오름차순으로 정리하며, `ORDER BY {정렬 기준} ASC`으로 오름차순을 명시할 수 있다. 내림차순은 `ORDER BY {정렬 기준} DESC`를 활용한다.
- `LIMIT {건너뛸 갯수}, {가져올 갯수}`로 인자를 두개 입력할 수 있다. 이 경우, `건너뛸 갯수 : 건너뛸 갯수 + 가져올 갯수` 범위만큼 검색한다.

## 기본 함수

사칙연산  `+, - , *, /`과 나머지 연산 `%, MOD`을 사용할 수 있다.

```sql
SELECT (col1 + col2) / 2, col1 MOD 3
FROM table1
```

논리 연산자로 `{값} IS (NOT) {True 또는 False}`을 사용할 수 있다.
```sql
SELECT col1 IS TRUE, col2 IS NOT TRUE
FROM table1
```

논리 연산자로 `AND 또는 &&`와 `OR 또는 ||`를 사용할 수 있다.
```sql
SELECT col1, col2
FROM table1
WHERE col1 IS TRUE AND col2 IS NOT TRUE
```

등호 `=`와 부등호 `!= 또는 <>`를 사용할 수 있다. 등호는 파이썬과 다르게 `=`로 나타냄에 주의하자.
```sql
SELECT col1, col1 = col2 AS Equality, col3
FROM table1
WHERE col3 <> 1
```

`(NOT) BETWEEN {최소값} AND {최대값}`로 데이터의 범위를 검증할 수 있다. 
```sql
SELECT col1, col2
FROM table1
WHERE col2 BETWEEN 0 AND 5
```

데이터가 어떤 값 사이에 있는지 `(NOT) IN ({값들})`로 검증할 수 있다.

```sql
SELECT col1 IN ('good', 'fair', 'excellent'), col2
FROM table1
```

최소값 `MIN({값들})`, 최대값 `MAX({값들})`, 개수 `COUNT({값들})`, 합 `SUM({값들})`, 평균 `AVG({값들})`을 구할 수 있다.
```sql
SELECT
    MAX(col1),
    MIN(col1),
    COUNT(col1),
    SUM(col1),
    AVG(col1)
FROM table1
```
- `COUNT({값들})`는 NULL 값을 제외한 행의 개수를 센다.



## 참고자료
1. 얄코의 MySQL, https://www.yalco.kr/lectures/sql/
3. 프로그래머스 SQL 고득점 Kit, https://school.programmers.co.kr/learn/challenges?tab=sql_practice_kit
1. 책 <데이터베이스 첫걸음>, 미크, 기무라 메이지
4. MySQL 8.0 Reference Manual, https://docs.oracle.com/cd/E17952_01/mysql-8.0-en/index.html