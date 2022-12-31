---
title: "[MySQL] 함수와 집계함수"
categories: DBMS
tags:
---

**Note:** 본 글은 이전에 작성한 글의 백업입니다.
{: .notice}

## 함수

### 문자열 함수

파이썬에는 정규표현식 개념이 있어서 패턴에 맞는 문자열을 검색할 수 있다. SQL에도 비슷한 기능이 있다. 
- `LIKE {...%패턴...}` : `%`에 해당되는 문자가 없거나 여러개의 문자가 해당될 수 있다.
- `LIKE {..._패턴...}` : `_`에는 한 개의 문자만 대응 될 수 있다.

```sql
SELECT
    'SQL' LIKE '%sql%', -- True
    'SQL' LIKE 'sql%',  -- True
    'SQL' LIKE 'S%',    -- True
    'SQL' LIKE '%L',    -- True
    'SQL' LIKE 'S%L',   -- True
    'SQL' LIKE '_ql',   -- True
    'SQL' LIKE '_q_',   -- True
    'SQL' LIKE '__ql_', -- False
```

- `REPLACE({문자열}, {대체될 문자열}, {대체할 문자열})` : 문자열에서 부분 문자열을 찾아 다른 문자열로 대체할 수 있다.
```sql
SELECT
    REPLACE('It is 12 AM', 'AM', 'PM') -- It is 12 PM
```

입력한 문자열을 순서대로 이어주는 파이썬의 join과 비슷한 기능을 하는 함수도 있다. 
- `CONCAT ({문자열들})`은 입력 문자열을 그대로 이어주고, 
- `CONCAT_WS({연결 문자열}, {문자열들})`을 통해 첫번쨰 인자로 전달한 문자열을 이용해 다음 인자로 주어진 문자열들을 잇는다.

```sql
SELECT CONCAT('Hi', ' ', 'there ', '!'), -- Hi there !
SELECT CONCAT_WS('-', 2022, 7, 8, 'AM') -- 2022-7-8-AM
```
- 문자열이 아닌 정수값을 입력해도 문자열로 이은 결과를 반환한다.


문자열을 왼쪽 또는 오른쪽에서 자를 수도 있다. 
- `LEFT({대상 문자열}, {자를 길이})` : 문자열을 왼쪽에서 부터 인자로 받은 길이만큼 자른다.  
- `RIGHT({대상 문자열}, {자를 길이})` : 문자열을 오른쪽에서 부터 인자로 받은 길이만큼 자른다.  

```sql
SELECT
    LEFT('yyyy-mm-dd hh:mm', 10) AS DATE, -- yyyy-mm-dd
    RIGHT('yyyy-mm-dd hh:mm', 5) AS TIME -- hh:mm
```

문자열의 왼쪽 또는 오른쪽에 패딩을 할 수 있다. 
- `LPAD ({대상 문자열}, {패딩 횟수}, {패딩 문자})` : 왼쪽 패딩 
- `RPAD ({대상 문자열}, {패딩 횟수}, {패딩 문자})` : 오른쪽 패딩

```sql
SELECT
  LPAD('This is padding', 3, '.'), -- ...This is padding
  RPAD('This is padding', 3, '.')  -- This is padding...
```

공백을 줄여주는 함수도 있다. 
- `TRIM {문자열}` : 입력 문자열의 양쪽 공백을 제거한다.
```sql
SELECT
    TRIM(' trim me ') -- 'trim me'
```


### 시간/날짜 함수

MySQL에는 날짜와 시간을 나타내는 변수 자료형 DATETIME이 있다. 

DATETIME 자료형을 생성하기 위해
- `DATE({날짜 문자열})` 또는 
- `TIME({시간 문자열})`을 이용한다.

```sql
SELECT
  '2022-7-1' = '2022-07-01', -- FALSE
  DATE('2022-7-1') = DATE('2022-07-01'), -- TRUE
  '1:18:4' = '01:18:04', -- FALSE
  TIME('1:18:4') = TIME('01:18:04'); -- TRUE
```

- `DATE`와 `TIME`에 전달하는 문자열은 0을 제외해도 무방하다.

DATETIME에서 연도, 월, 요일명, 날짜, 시간, 분, 초를 추출하는 함수는 각각 다음과 같다.
- `YEAR({DATETIME})` : DATETIME 변수의 연도 반환
- `MONTH({DATETIME})` : DATETIME 변수의 월 반환
- `DAYNAME({DATETIME})` : DATETIME 변수의 요일명 반환
- `DAY({DATETIME})` : DATETIME 변수의 날짜 반환
- `HOUR({DATETIME})` : DATETIME 변수의 시간 반환
- `MINUTE({DATETIME})` : DATETIME 변수의 분 반환
- `SECOND({DATETIME})` : DATETIME 변수의 초 반환

```sql
SELECT
  YEAR(NOW()) AS YEAR,
  MONTH(NOW()) AS MONTH,
  DAYNAME(NOW()) AS DAYNAME,
  DAY(NOW()) AS DAY,
  HOUR(NOW()) AS HOUR,
  MINUTE(NOW()) AS MINUTE,
  SECOND(NOW()) AS SECOND
```
- `NOW()` : 현재 날짜와 시간을 DATETIME 자료형으로 반환


### 조건문

조건에 따라 다른 값을 반환하도록 지정할 수 있다. 
- `IF ({조건}, {조건이 참일 때 반환값}, {조건이 거짓일 때 반환값})` 형식으로 사용한다.

```sql
SELECT 
    IF (col1 > 3, 'Bigger than 3', 'Equal or Smaller than 3')
FROM table1
```

반환하고자 하는 경우의 수를 더 세세하게 나눌 수도 있다. 
- `CASE 
    WHEN {경우1} THEN {결과값1} 
    ... 
    ELSE {이외의 경우 반환값} 
   END` 
- 어떤 경우가 앞서 나온 조건에 해당되면 이미 반환값이 결정되므로, 뒤에 나온 조건은 앞서 나온 조건들에 해당되지 않는 경우만 포함한다.

```sql
SELECT
    CASE
        WHEN -1 > 0 THEN '-1은 양수다.'
        WHEN -1 = 0 THEN '-1은 0이다.'
        ELSE '-1은 음수다.'
    END
```

검색한 행에 NULL값이 있고 이 값만 변경하고자 하면 
- `IF ({비교 대상} = NULL, {바꿀 값}, {비교 대상})`으로 표현할 수 있다. 
- `IFNULL({비교 대상}, {값})` : 비교 대상이 NULL값인 경우 지정된 값을 반환한다.

```sql
SELECT
    IFNULL(col1, '0')
FROM table1
```


## 집계함수

행을 그룹으로 묶어서 검색할 수 있다. 
- `GROUP BY {그룹들}`은 그룹별 값을 도출하며, 
- `HAVING {그룹 내 조건}`을 통해 그룹으로 묶은 뒤의 조건을 설정할 수 있다.

```sql
SELECT 
    col1, 
    COUNT (DISTINCT col2)
FROM table1
WHERE condition1
GROUP BY col1
HAVING in_group_condition1
```

- `GROUP BY {그룹들}`은 복수의 기준으로 그룹을 묶을 수 있다.
- `WHERE`가 그룹하기 전에 조건을 기준으로 하는 필터라면, `HAVING`은 그룹한 이후의 필터이다. 
- `DISTINCT`는 중복 값을 제거하는 연산이며, 종합하여 `COUNT (DISTINCT {칼럼})`은 칼럼 내의 고유한 값들의 개수를 세어 반환한다.

