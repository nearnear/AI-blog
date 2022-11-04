---
title: "python 시퀀스 자료구조 - list"
categories: Python
tags:
  - python
  - data structure
  - list
---

Python의 내장 시퀀스 데이터 타입인 리스트에 대해 알아본다.

## 리스트
파이썬 리스트는 동적 배열(array)로, 연결 리스트(linked list)와는 관련이 없다. 리스트는 항목을 쉼표 `,`로 
구분하고, 대괄호 `[]`로 감싼다. 리스트의 항목은 서로 다른 데이터 타입일 수 있으며, 인덱스로 특정 요소에 접근하여 
항목 값을 변경할 수 있다(가변 타입).


## 리스트 메서드

### append()
`A.append(x)`는 리스트 A의 오른쪽에 항목 x를 추가한다. 시간복잡도는 $O(1)$이다. `A[len(A):] = [x]`로 
같은 기능을 구현할 수 있다.

```
>>> books = ['금각사', '수레 바퀴 아래서']
>>> books.append('배를 엮다')
>>> books
['금각사', '수레 바퀴 아래서', '배를 엮다']
>>> books[len(books):] = ['아내를 모자로 착각한 남자']
>>> books
['금각사', '수레 바퀴 아래서', '배를 엮다', '아내를 모자로 착각한 남자']
```

### extend()
`A.extend(B)`는 iterable 타입인 B를 리스트 A에 추가한다. `A[len(A):] = B` 또는 `A += B`로 
같은 기능을 구현할 수 있다. 

```
>>> books = ['금각사']
>>> books.extend(['봄눈'])
>>> books
['금각사', '봄눈']
>>> books.extend("아무거나")
>>> books
['금각사', '봄눈', '아', '무', '거', '나']
```

### insert()
`A.insert(i, x)`는 리스트 A의 인덱스 위치 i에 항목 x를 삽입한다. 시간복잡도는 리스트 A의 길이 n에 대해 $O(n)$이다. 

```
>>> books = ['금각사', '수레 바퀴 아래서']
>>> books.insert(1, '데미안')
>>> books
['금각사', '데미안', 수레 바퀴 아래서']
```

### remove()
`A.remove(x)`는 리스트 A에서 항목 x를 찾은 뒤 제거한다. 항목 x가 존재하지 않으면 ValueError를 발생시킨다.
항목 탐색 과정에 의해 $O(n)$ 시간이 소요된다.

```
>>> int_range = list(range(5, 92))
>>> int_range.remove(88)
>>> int_range
[5, 6, ..., 87, 89, .., 91]
>>> int_range.remove(99)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: list.remove(x): x not in list
```

### pop()
`A.pop(i)`는 리스트 A에서 인덱스 i에 있는 항목을 제거하고 그 항목을 반환한다. i를 지정하지 않으면 리스트의 맨 오른쪽 
항목을 제거하고 반환한다. i가 지정되지 않은 경우는 $O(1)$ 시간이, i가 지정된 경우 최대 $O(n)$ 시간이 소요된다.

```
>>> int_range = list(range(5, 92))
>>> int_range.pop()
91
>>> int_range.pop(33)
38
```

### del문
del문은 리스트 인덱스를 지정하거나 인덱스 슬라이싱을 통해 주어진 범위의 항목들을 삭제한다. 

```
>>> numbers = list(range(10))
>>> del numbers[5]
>>> numbers
[0, 1, 2, 3, 4, 6, 7, 8, 9]
>>> del numbers[4:6]
>>> numbers
[0, 1, 2, 3, 7, 8, 9]
>>> del numbers
>>> numbers
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'numbers' is not defined
```

마지막 사례처럼 객체 참조가 삭제 되었을 때 그 데이터 항목은 파이썬의 garbage collector에 수집된다.

### index()
`A.index(x)`는 리스트 A에서 항목 x를 찾아 그 항목의 인덱스를 반환한다. 리스트 탐색 과정으로 $O(n)$ 시간이 소요된다.
remove()와 마찬가지로 항목 x가 리스트에 없으면 ValueError를 일으킨다.

```
>>> numbers = list(range(22, 33)) + list(range(67, 89))
>>> numbers.index(77)
21
>>> numbers.index(55)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: 55 is not in list
```

### count()
`A.count(x)`는 리스트 A에 항목 x가 들어있는 개수를 반환한다. 리스트 전체를 탐색하므로 시간복잡도는 $O(n)$이다.

```
>>> numbers = [0] * 55 + [1] * 22
>>> numbers.count(1)
22
```

### sort()
`A.sort(key, reverse)`는 리스트 A를 정령하여 변수에 in place로 적용한다. 인수 key는 함수를 전달해야하며, 
리스트 항목을 내림차순으로 정렬할 때는 `sort(reverse=True)` 형식으로 지정한다. 시간복잡도는 내부에서 Timsort로 
구현되어 $O(nlogn)$이다.

```
>>> import random
>>> numbers = list(range(10))
>>> random.shuffle(numbers)
>>> numbers
[7, 2, 4, 5, 3, 0, 8, 1, 6, 9]
>>> numbers.sort()
>>> numbers
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### reverse()
`A.reverse()` 메서드는 리스트 A의 항목 순서를 반전해 변수에 in place로 적용한다. 시간복잡도는 전체 탐색으로 인해 
$O(n)$이다. `A[::-1]`과 같은 구현을 한다.

```
>>> numbers = list(range(10))
>>> numbers.reverse()
[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

## 리스트 언패킹
리스트 언패킹은 튜플 언패킹과 유사하다.

```
>>> first, *rest = [1, 2, 3, 4, 5]
>>> first
1
>>> rest
[2, 3, 4, 5]
```

리스트에 starred argument `*`를 붙여 함수의 인수로 전달할 수도 있다.

```
>>> fn = lambda x, y, z: x * y * z
>>> numbers = [11, 22, 33]
>>> fn(*numbers)
7986
```

## 리스트 컴프리헨션
List comprehension은 반복문의 표현식으로 다음과 같은 형식이 있다.
- `[x for x in <iterable>]`
- `[<repr> for x in <iterable>]`
- `[<repr> for x in <iterable> if <condition>]`

다음은  리스트 컴프리헨션의 사용 예시다.

```
>>> pis = [str(rount(math.pi, i)) for i in range(1, 6)]
>>> pis
['3.1', '3.14', '3.142', '3.1416', '3.14159']
>>> years = [year for year in range(1995, 2022) if year % 5 == 0]
[1995, 2000, 2005, 2010, 2015, 2020]
```

리스트 컴프리헨션은 연결(concatenate)보다 일반적으로 빠르다.

```python
import timeit

# list concatenation
def test_concat():
    l = []
    for i in range(10000):
        l += [i]

# list comprehension
def test_compre():
    l = list(range(10000))


if __name__ == '__main__':
    t1 = timeit.Timer("test_concat()", "from __main__ import test_concat")
    print("concat ", t1.timeit(number=10000), 'ms')
    t2 = timeit.Timer("test_compre()", "from __main__ import test_compre")
    print("comprehension ", t2.timeit(number=10000), 'ms')
```

```
concat  3.919952416 ms
comprehension  0.8694067079999996 ms
```

만 번 수행한 경우 리스트 컴프리헨션이 for문을 통한 연결보다 4~5배 정도 빨랐다.

### 참고자료
- An Introduction to Python & Algorithms, Mia Stein