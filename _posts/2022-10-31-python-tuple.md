---
title: "python 시퀀스 자료구조 - tuple"
categories: Python
tags:
  - python
  - data structure
  - tuple
---

Python의 내장 시퀀스 데이터 타입인 튜플에 대해 알아본다.

### 튜플
튜플tuple은 쉼표 `,`로 구분된 값으로 이루어진 불변 시퀀스 타입이다. 문자열과 같이 튜플은 인덱스를 통한 객체 참조가 가능하다.

```
>>> tup1 = 'pi', 3.14259
>>> tup1[0]
'pi'
>>> tup1
('pi', 3.14259)
>>> tup2 = 'pi',
>>> tup2
('pi',)
>>> tup3 = ('pi')
'pi'
```

쉼표 `,`는 튜플을 만들지만 괄호 `()`만으로는 튜플을 만들 수 없다.

```
>>> tup = 'pi', [3,1,4,1,5,9]
>>> tup
('pi', [3, 1, 4, 1, 5, 9])
>>> type(tup)
<class 'tuple'>
```

튜플은 리스트 같은 가변 객체를 값으로 포함할 수 있다.

### 튜플 메서드
`A.count(x)`는 튜플 A에 담긴 값 x의 개수를 반환하며, `A.index(x)`는 x의 인덱스를 반환한다.

```
>>> tup = 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5
>>> t.count(5)
3
>>> t.index(5)
4
```

### 튜플 언패킹
파이썬의 모든 iterable 객체는 시퀀스 언패킹 연산자(sequence unpacking operator) `*`를 통해 언패킹할 
수 있다. 반환값은 list임에 주의하자.

```
>>> dec_int, *dec_real = 3, 1, 4, 1, 5, 9
>>> dec_int
3 
>>> dec_real
[1, 4, 1, 5, 9]
>>> *x, y = 3, 1, 4, 1, 5, 9
>>> x
[3, 1, 4, 1, 5]
>>> y
9
```

### 네임드 튜플
파이썬 표준 모듈 collections에 시퀀스 데이터 타입인 네임드 튜플이 정의되어 있다. 네임드 튜플은 
튜플과 같이 불변 객체이지만, 이름으로 값을 참조할 수 있다는 점에서 다르다.

`collections.namedtuple()` 메서드에 첫째로 사용자 정의 튜플 데이터 타입의 이름을, 두번째로 튜플의 각 항목을 
지정하는 공백으로 구분된 문자열(또는 리스트나 튜플)을 인자로 전달한다. 

```
>>> import collections
>>> Person = collections.namedtuple()
>>> p = Person('현정', 28, '여자')
>>> p
Person(name='현정', age=28, gender='여자')
>>> p.age
28
>>> p.age = 20
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: can't set attribute
```

즉 네임드 튜플을 정의하는 일은 클래스와 불변 클래스 어트리뷰트를 지정하는 것과 같다.

### 참고자료
- An Introduction to Python & Algorithms, Mia Stein