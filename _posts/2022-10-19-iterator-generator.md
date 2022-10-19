---
title: "python iterator와 generator"
categories:
    - Python
tags:
    - python
    - coroutine
---

## 1. iterator 객체

### next 메서드 (PEP 234, python 2.1)

파이썬의 iterator 객체는 메모리를 할당하지 않고, 이전 원소로 다음 원소를 계산해 리턴한다. 즉 iterator는 메모리를 절약하는 것이 장점이며, `range()` 함수가 대표적인 예이다. iterator 사용과 메모리 할당 방법의 메모리 차이를 다음과 같이 확인할 수 있다. 

```python
import sys

iter_large = range(10 ** 9)
large_list = [i for i in iter_large]
print(sys.getsizeof(iter_large)) # 48
print(sys.getsizeof(large_list)) # 8058558872
```

iterator 객체는 자신을 반환하는 `__iter__()`와 다음 값을 반환하는 `__next__()` 메서드를 가진다. 빌트인 `iter()` 함수를 활용하여 리스트 값을 iterator 객체로 변환할 수도 있다.

```python
class UnicodeIterator(object):
    def __init__(self, word):
        self.word = word
        self.unicode = bytearray(self.word.encode('utf-8')) # str

    def __iter__(self):
        return self

    def __next__(self):
        if self.unicode[-1] > 0x10FFFF:
            raise StopIteration
        self.unicode[-1] += 1
        self.word = bytes(self.unicode).decode('utf-8')
        return self.unicode, self.word
```

이 예는 입력한 유니코드 문자열에서 마지막 문자의 다음 유니코드 문자를 반환한다. `StopIteration` 예외를 발생시키면 iteration이 종료된다. 여기서는 마지막 문자의 유니코드가 유니코드 범위를 넘었을 때 종료하도록 했다. 이제 객체를 생성해서 동작을 확인할 수 있다.

```python
ui = UnicodeIterator('하이') 
for _ in range(5):
    print(next(ui)[-1]) # 하익, 하읶, 하읷, 하인, 하읹
```

예제는 입력된 유니코드 문자열의 마지막 문자를 다음 유니코드 문자로 변형시켜 출력한다. 



## 2. generator 객체

### yield 메서드 (PEP 255, python 2.2)

generator 객체를 생성하는 방법은 PEP 255(Simple Generators)에서 추가되었다. `yield()`를 통해 함수를 정의하면 함수를 호출했을 때 generator 객체가 리턴되며, generator 객체는 iterator와 같이 `__next__()` 어트리뷰트를 가지고 있다.

```python
def generate_number(low, up):
    for i in range(low, up + 1):
        if i ** 2 != 4:
            yield(i)
```

위의 함수는 주어진 정수 범위에서 제곱근이 2가 아닌 숫자를 하나씩 반환한다. 

```python
>>> fn = generate_number(-5, 5)
>>> type(fn)
<class 'generator'>
>>> '__next__' in dir(fn)
True
```

하지만 정의한 함수의 type은 function이 아니라 함수를 감싸는 generator임에 유의하자. 또한 함수의 어트리뷰트에 '__next__'가 있으므로 `next(fn)`을 통해 값을 가져 올 수 있다.

yield를 활용해 함수로 generator를 만들면 iterator를 보다 간편하게 만들 수 있다. class 내에 iterator로 구현했던 기능을 함수 내의 yield로 작성해보자. 

```python
def next_unicode(word):
    encoding = bytearray(word.encode('utf-8'))
    while True:
        if encoding[-1] > 0x10FFFF:
            raise StopIteration
        encoding[-1] += 1
        word = bytes(encoding).decode('utf-8')
        yield encoding, word
```

코드가 보다 간결해진 것을 확인할 수 있다. 이 함수는 StopIteration을 만날 때까지 next로 값을 반환할 수 있다.


#### generator 언패킹
이처럼 iterator 혹은 generator가 반환할 수의 범위를 알고 있는 경우, 언패킹(unpacking)도 가능하다.

```python
>>> a, b, c, d, e = generate_number(-3, 3) 
-3, -1, 0, 1, 3
```

#### generator 내부 에러
만약 generator 내부에서 예외가 발생하면 어떻게 될까?

```python
>>> def fn_zerodivision():
        yield 1
        yield 1 / 0

>>> fn = fn_zerodivision()
>>> next(fn)
1
>>> next(fn)
ZeroDivisionError: division by zero
>>> next(fn)
StopIteration
```

처음 next를 호출하면 1을 반환하며, 그 다음에는 ZeroDivisionError를 반환한다. 한번 더 호출하면 다시 1을 반환하는 것이 아니라 StopIteration이 지정되어 더 이상 next를 호출할 수 없게 된다.