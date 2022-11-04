---
title: "python 컬렉션 자료구조 - set"
categories: Python
tags:
  - python
  - data structure
  - set
---

Python의 set(셋)은 수학의 집합 개념에 기반한 컬렉션 데이터 타입으로, 중복 요소가 없고 정렬되지 않은 
컨테이너(container)이다. 셋에는 순서의 개념이 없으므로 인덱스 연산은 불가능하며 주로 멤버십 테스트 및 중복 항목 제거를 위해 
사용된다. 셋에서 항목을 삽입하는 시간 복잡도는 $O(1)$이고, 두개 집합의 합집합(union)의 경우 시간 복잡도는 각 집합의 원소 
개수 m과 n에 대해 $O(m+n)$이다. 교집합(intersection)의 경우 $m > n$일때 $O(n)$이다.


## set 메서드

### add()
`A.add(x)`는 셋 A에 항목 x를 추가한다. 이미 항목 x가 있는 경우 아무 작업도 수행하지 않는다.

```
>>> drinks = {'카페 라떼', '자몽허니 블랙티'}
>>> drinks.add('초코 프라푸치노')
>>> drinks
{'카페 라떼', '자몽허니 블랙티', '초코 프라푸치노'}
```

### update()과 연산자 |=
`A.update(B)` 또는 `A |= B`는 셋 B를 셋 A에 합집합의 개념으로 추가한다.

```
>>> drinks = {'카페 라떼', '자몽허니 블랙티'}
>>> drinks.update({'카페 라뗴', '카라멜 마끼아또'})
>>> drinks
{'카페 라떼', '자몽허니 블랙티', '카라멜 마끼아또'}
>>> drinks |= {'카페 라떼', '자몽허니 블랙티'}
>>> drinks
{'카페 라떼', '자몽허니 블랙티', '카라멜 마끼아또', '자몽허니 블랙티'}
```

### union()과 연산자 |
`A.union(B)` 또는 `A | B`는 update와 같은 작업을 수행하지만 연산 결과를 사본으로 반환하는 점에서 다르다.

```
>>> drinks = {'카페 라떼', '자몽허니 블랙티'}
>>> drinks.union({'카페 라뗴', '카라멜 마끼아또'})
{'카페 라떼', '자몽허니 블랙티', '카라멜 마끼아또'}
>>> drinks | {'카페 라떼', '자몽허니 블랙티'}
{'카페 라떼', '자몽허니 블랙티', '카라멜 마끼아또', '자몽허니 블랙티'}
```

### intersection() 또는 연산자 &
`A.intersection(B)` 또는 `A & B`는 셋 A와 셋 B의 교집합의 복사본을 반환한다.

```
>>> path1 = {'point_a', 'point_b'}
>>> path2 = {'point_a', 'point_c', 'point_d'}
>>> path1.intersection(path2)
{'point_a'}
>>> path2.intersection(path1)
{'point_a'}
>>> path1 & pathB
{'point_a'}
```

### difference() 또는 연산자 -
`A.difference(B)` 또는 `A - B`는 셋 A에서 셋 B를 뺀 차집합의 복사본을 반환한다.

```
>>> path1 = {'point_a', 'point_b'}
>>> path2 = {'point_a', 'point_c', 'point_d'}
>>> path1.difference(path2)
{'point_b'}
>>> path1 - path2
{'point_b'}
>>> path2 - path1
{'point_c', 'point_d'}
```

### clear()
`A.clear()`는 셋 A의 모든 항목을 제거한다.

```
>>> numbers = set(k for k in range(5, 100) if k % 7 == 1)
>>> numbers.clear()
>>> numbers()
set()
```

### discard(), remove(), pop()
`A.discard(x)`는 셋 A에서 항목 x를 찾아 제거하며 반환값은 없다. `A.remove(x)`는 discard()와 같지만 
항목 x가 없는 경우 KeyError 예외를 발생시킨다. `A.pop()`은 셋 A의 항목 하나를 무작위로 제거하고 항목을 반환한다. 
셋이 비어 제거할 항목이 없는 경우 KeyError 예외를 발생시킨다.

```
>>> numbers = set(range(10))
>>> numbers.discard(1)
>>> numbers
{0, 2, 3, 4, 5, 6, 7, 8, 9}
>>> numbers.discard(10)

>>> numbers.remove(2)
>>> numbers
{0, 3, 4, 5, 6, 7, 8, 9}
>>> numbers.remove(10)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 10
>>> numbers.pop()
0
>>> set().pop()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'pop from an empty set'
```

셋과 유사한 데이터 타입 frozen set은 불변 객체로, `frozenset()`으로 생성한다. 가변 객체인 셋과 달리 프로즌 셋의 
요소를 변경하는 메서드를 사용할 수 없다.
```
>>> fs = frozenset(range(10))
>>> 3 in fs
True
>>> len(fs)
10
>>> fs.add(5)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'frozenset' object has no attribute 'add'
>>> fs.union(range(12, 14)
frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13})
>>> fs.intersection(range(5))
frozenset({0, 1, 2, 3, 4})
>>> fs.difference(range(7))
frozenset({7, 8, 9})
>>> fs.clear()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'frozenset' object has no attribute 'clear'
>>> fs.discard()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'frozenset' object has no attribute 'discard'
```
{: .notice--info}


## 셋과 리스트

`set(A)`는 리스트 A를 셋 타입으로 변환(casting)하며, `list(set(A))`를 통해 연산 결과를 리스트 타입으로 반환할 
수 있다.

```python
def remove_dup(list1):
    return list(set(list1))

def intersection(list1, list2):
    return list(set(list1) & set(list2))

def union(list1, list2):
    return list(set(list1) | set(list2))

def test():
    list1 = [1, 2, 3, 4, 5, 5, 5, 0, -1]
    list2 = [2, 5, 6, 6, 0]
    assert(remove_dup(list1) == [0, 1, 2, 3, 4, 5, -1])
    assert(intersection(list1, list2) != [2, 5, 0])
    assert(intersection(list1, list2) == [0, 2, 5]) # 순서에 유의하자.
    assert(union(list1, list2) != [1, 2, 3, 4, 5, 6, 0, -1])
    assert(union(list1, list2) == [0, 1, 2, 3, 4, 5, 6, -1]) # 순서에 유의하자.
    print('테스트 통과!')

if __name__ == '__main__':
    test()
```

```
테스트 통과!
```

주목할 것은 `list(set(A))`의 반환값이 리스트 A의 순서를 그대로 보존하지 않는다는 점이다. 

리스트 뿐만 아니라 딕셔너리도 셋 속성을 사용할 수 있다.

```python
def dict_set_operation():
    d1 = {'k1': 4, 'k2': 5, 'k3': 2}
    d2 = {'k7': 5, 'k2': 5, 'k8': 2}
    print(f'Dict1\t\t\t: {d1}\nDict2\t\t\t: {d2}')

    its = d1.keys() & d2.keys()
    its_items = d1.items() & d2.items()
    print(f'Itersection\t\t: {its}\n\t\t\t\t: {its_items}')

    sbt1 = d1.keys() - d2.keys()
    sbt1_items = d1.items() - d2.items()
    print(f'Subtraction 1\t: {sbt1}\n\t\t\t\t: {sbt1_items}')

    sbt2 = d2.keys() - d1.keys()
    sbt2_items = d2.items() - d1.items()
    print(f'Subtraction 2\t: {sbt2}\n\t\t\t\t: {sbt2_items}')
if __name__ == '__main__':
    dict_set_operation()
```

```
Dict1			: {'k1': 4, 'k2': 5, 'k3': 2}
Dict2			: {'k7': 5, 'k2': 5, 'k8': 2}
Itersection		: {'k2'}
				: {('k2', 5)}
Subtraction 1	: {'k3', 'k1'}
				: {('k1', 4), ('k3', 2)}
Subtraction 2	: {'k8', 'k7'}
				: {('k8', 2), ('k7', 5)}
```

### 참고자료
- An Introduction to Python & Algorithms, Mia Stein