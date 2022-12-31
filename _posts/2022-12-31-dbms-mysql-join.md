---
title: "[MySQL] JOIN"
categories: DBMS
tags:
---

**Note:** 본 글은 이전에 작성한 글의 백업입니다.
{: .notice}

원하는 정보가 여러 테이블에 흩어져 있는 경우, 몇 개의 외부 테이블을 합치는 명령어가 필요하다. Pandas 패키지에서 데이터프레임을 병합하는 경우를 생각해보자(사실 pandas 공식 문서에는 merge를 SQL의 JOIN을 들어 설명하고 있지만, 이미 pandas를 사용해본 경험이 있다면 이해가 빠를 것이다). `df1.merge(df2, how='inner', on='col1')`명령어는 df1에 df2를 양쪽 데이터프레임에 공통적으로 존재하는 col1의 값들이 일치되도록 병합한다. 반면 `df1.merge(df2, how='left', on='col1')`은 왼쪽, 즉 df1의 키만 보존하며 두 데이터프레임을 병합한다.

이처럼 테이블을 합치는 방법은 여러가지가 있으며, 여기서는 INNER, OUTER, CROSS JOIN을 소개한다. 


## INNER JOIN

INNER JOIN을 위해서는 합치고자 하는 테이블에 공통된 키가 있어야 한다. 쿼리는 `JOIN {외부 테이블} ON {공통된 칼럼}` 형식이다. 공통된 키에 대해 병합되기 때문에 공통된 칼럼에 대한 두 테이블의 교집합을 떠올릴 수 있다(단, 엄밀히 테이블은 집합이 아니다!).

```sql
SELECT A.col1, A.col2, B.col2, C.col2
FROM table1 A
JOIN table2 B
  ON A.col1 = B.col1
JOIN table3 C
  ON A.col2 = C.col3
WHERE condition1
ORDER BY sort_standard1
```
- `FROM table1 A`는 table1을 A로 명명하고, `JOIN table2 B`는 table2를 B로 명명한다. 이후 어느 테이블에서 행을 가져오는지 명시할 때 새로 명시한 이름을 사용할 수 있다.
- 위의 예에서 처럼 여러개의 테이블을 병합할 수 있다. 쿼리 형식 `JOIN {외부 테이블} ON {공통된 칼럼}`만 지키면 된다.
- 또한, `WHERE`나 `ORDER BY`등 `SELECT`에서 익힌 조건문들도 그대로 사용할 수 있다는 걸 잊지 말자.


그렇다면 하나의 테이블을 두번 JOIN할 일은 없을까? 의외로 하나의 테이블을 다시 외부 테이블처럼 JOIN하는 활용처도 있으며, 이를 SELF JOIN이라고 한다.

```sql
SELECT
    A.col1, A.col2 AS val,
    B.col1, B.col2 AS next_val
FROM table1 A 
JOIN table2 B
  ON A.col2 + 1 = B.col2
```
이처럼 하나의 칼럼에 lag를 두는 경우, 서로 다른 칼럼을 가져오는 것처럼 JOIN으로 불러와 활용할 수 있다. 

## OUTER JOIN

OUTER JOIN은 공통된 키가 아닌 하나의 테이블이 가지는 키를 기준으로 합치는 방법이며, 합치고자 하는 방향에 따라 LEFT JOIN과 RIGHT JOIN으로 나뉜다. 쿼리는 `LEFT JOIN {외부 테이블}`과 `RIGHT JOIN {외부 테이블}`이다.

```sql
SELECT A.col1, B.col1, A.col2, A.col3
FROM table1 A
LEFT JOIN table2 B
       ON A.col2 = B.col2 AND A.col3 = B.col3
```

- `LEFT JOIN`을 `RIGHT JOIN`으로 바꿔 쓸 수 있다. 
- LEFT JOIN은 table1(또는 A)의 키를 기준으로 합치며, RIGHT JOIN은 table2(또는 B)의 키를 기준으로 합친다. 기준이 되지 않는 테이블에 있는 칼럼 값이 NULL이더라도 기준이 되는 칼럼 값이 존재하면 반환하므로, 결과값으로 반환되는 테이블의 행의 갯수는 기준이 되는 칼럼의 합치기 전 행의 갯수와 같다.

### CROSS JOIN
두 칼럼 값의 모든 조합 쌍을 나타내는 카르테지안 프로덕트(Cartesian Product)를 실행하는 방법도 있다. `CROSS JOIN {외부 테이블}` 명령으로 구현하며, 이를 CROSS JOIN이라고 한다.

```sql
SELECT A.col1, B.col2
FROM table1 A
CROSS JOIN table2 B
```
결과값은 `A.col1`와 `B.col2`의 모든 조합 썅을 담은 테이블이다.