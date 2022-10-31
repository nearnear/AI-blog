---
title: "python 문자열"
categories: Python
tags:
    - python
    - string
---

Python의 내장 시퀀스 데이터 타입인 문자열에 대해 알아본다.

---

문자열(string)은 문자(character)의 시퀀스이다. 파이썬은 불변의 str 타입을 통해 문자열을 표현한다. 
파이썬의 객체는 두가지 출력 형식이 있는데, 문자열 형식은 형식은 사람이 읽기 위한 형태이고, representation 형식은 파이썬 
인터프리터가 읽기 위해 사용되는 형식으로 디버깅에 사용된다. 


## 유니코드 문자열

유니코드(Unicode)란 아스키 문자를 기반으로 지구의 모든 문자를 표현하는 목적으로 정의된 국제 표준 코드이다. 유니코드에는 공백, 
특수문자, 수학 기호 등을 포함하고 있다. Python은 3부터 모든 문자열은 유니코드이며, 따라서 파이썬 str 변수에 한글 문자열을 저장할 수 
있다. 유니코드 표현에는 16비트가 필요하며, 7비트로 표현하는 ASCII 코드를 모두 포함하고 있다. 유니코드는 버전을 통해 문자들을 
추가하고 있는데, emoji의 경우 유니코드 6.0에서 도입되어 이전 버전에서는 인코딩이 안되는 경우가 있다.

파이썬에서 문자열 앞에 `u`를 붙이는 것으로 유니코드 문자열을 만들 수 있다. `u0020`은 whitespace를 나타내는 유니코드 
문자이다.

```
>>> u'안녕\u0020디지몬!'
'안녕 디지몬!'
```

유니코드 인코딩 방식은 다양하지만, 그중 빈번히 쓰이는 UTF-8 인코딩 방법에 대해 알아보자.

### UTF-8 인코딩
UTF-8 인코딩은 시작 비트가 전체 바이트 수를 결정한다.
  - 0으로 시작할 경우 1 바이트,
  - 110으로 시작할 경우 2 바이트,
  - 1110으로 시작할 경우 3 바이트,
  - 11110으로 시작할 경우 4 바이트이다.

이외에 시작 비트에 따라오는 수는 10으로 시작한다. 0으로 시작하는 바이트는 ASCII 인코딩을 그대로 따르며, 
대부분의 중국어, 일본어, 한국어 문자는 3 바이트로 표현된다. 인코딩된 바이트가 정수 리스트로 주어졌을 때, 
그 값이 UTF-8으로 인코딩 되어있는지 다음과 같이 판별할 수 있다.

```python
def valid_utf8(data: list[int]) -> bool:
    
    # (1) 시작하는 비트에 따라오는 수의 조건을 확인한다.
    def check(size):
        for i in range(start + 1, start + size + 1):
            if i > len(data) or (data[i] >> 6) != 0b10:
                return False
        return True

    # (2) 맨 앞의 바이트 부터 시작하는 비트를 확인한다.
    start = 0
    while start < len(data):
        first = data[start]
        if (first >> 3) == 0b11110 and check(3):
            start += 4
        elif (first >> 4) == 0b1110 and check(2):
            start += 3
        elif (first  >> 5) == 0b110 and check(1):
            start += 2
        elif (first >> 7) == 0:
            start += 1
        else:
            return False

    return True
```

- (1) : 해당 바이트에 따라오는 수가 전체 데이터 길이를 넘지 않는지, 또는 0b10으로 시작하는지를 확인한다.
- (2) : 대상 바이트의 앞자리가 특정 수이고 따라오는 수가 주어진 조건을 만족하면 그 다음 바이트를 확인한다. 
    - 예를 들어 대상 바이트가 0b1110으로 시작하고 따라오는 수도 조건을 만족하면, 세 바이트 뒤의 값을 다시 조사한다.
    - 이때 남은 바이트 값 만큼 오른쪽으로 shift해서 동치를 확인할 수 있다.


## 문자열 메서드

### join()
`A.join(B)`는 리스트 B의 모든 문자열을 단일 문자열 A로 결합한다. for문으로 + 연산을 통해 문자열을 결합하는 것보다 
효율적이다.

```
>>> tools = ["포크", "나이프", "숟가락"]
>>> " 그리고 ".join(lights)
'포크 그리고 나이프 그리고 숟가락'
```

### ljust(), rjust()
`A.ljust(width, fillchar)`는 문자열 A를 시작으로, fillchar문자를 width만큼 이어붙인 문자열을 반환한다.
`A.rjust(width, fillchar)`는 문자열 A를 끝으로, fillchar 문자를 width만큼 이어붙인 문자열을 반환한다.
이들은 정렬의 개념으로 구현되므로 fillchar는 하나의 문자(character 또는 길이가 1인 string)이며 디폴트 값은 
`' '`값이다. 또한 ljust()와 rjust()를 중복하여 사용할 수 없다. 

```
>>> zero = "0"
>>> zero.ljust(15, '-')
'0______________'
>>> b = zero.rjust(10, '-')
'---------0'
>>> zero.ljust(15, '_').rjust(10, '-')
'0______________'
```

### format()
`A.format()`은 문자열 A에 변수를 추가하거나 형식을 만든다.

```
>>> "{} {}".format("Gute", "Nacht")
"Gute Nacht"
>>> "이름: {who}, 나이: {age}".format(who="단델리온", age=18)
"이름: 단델리온, 나이: 18"
>>> "이름: {who}, 나이: {0}".format(12, who="다다")
"이름: 다다, 나이: 12
```

format() 메서드는 3개의 지정자가 있다. 지정자 s는 문자열 형식을. r은 표현 형식을, a는 아스키 코드 형식을 의미한다.

```
>>> import decimal
>>> "{0} {0!s} {0!r} {0!a}".format(format(decimal.Decimal("99.99"))
"99.99 99.99 Decimal('99.99') Decimal('99.99')"
```

### 문자열 언패킹
`**` 연산자는 문자열 매핑 언패킹(mapping unpacking) 연산자이다. 

```
>>> name = "모스크"
>>> order = "3"
>>> "{name}: {order}".format(**locals())
'모스크: 3'
```

`locals()` 메서드는 현재 scope에 있는 지역 변수들을 딕셔너리로 변환한다. 문자열 매핑 언패킹 연산자는 키-값 
딕셔너리를 생성한다.

### splitlines()
`A.splitlines()`는 줄바꿈 문자 `\n`를 기준으로 분리한 문자열을 리스트로 반환한다.

```
>>> sentence = "그리고\n아무도\n없었다."
>>> sentence.splitlines()
['그리고', '아무도', '없었다.']
```

### split() 메서드
`A.split(t, n)`은 문자열 A에서 문자열 t를 기준으로 정수 n번만큼 분리한 문자열 리스트를 반환한다. n의 디폴트 값은 
A를 t로 최대한 분리한다. t를 지정하지 않으면 whitespace로 구분한 문자열 리스트를 반환한다.

```
>>> one_row = "Bach-y-Rita*Scientist*Professor"
>>> attributes = one_row.split("*")
['Bach-y-Rita', 'Scientist', 'Professor']
>>> attributes[0].split('-')
['Bach', 'y', 'Rita']
>>> attributes[0].rsplit('-', 1)
['Bach-y', 'Rita']
```

`A.rsplit(t, n)`은 문자열의 오른쪽에서 부터 t를 기준으 n번 분리한 문자열 리스트를 반환한다.

### strip()
`A.stip(B)`는 문자열 A의 좌우에 있는 문자열 B를 제거한다. B가 주어지지 않으면 whitespace를 제거한다.

```
>>> tools = "**스푼과 나이프**  "
>>> tools.strip()
'**스푼과 나이프**'
>>> tools.strip().strip('*')
'스푼과 나이프'
```

### swapcase() 메서드
`A.swapcase()`는 이름 그대로 문자열 A에서 알파벳 대소문자를 반전한 문자열의 복사본을 반환한다. 

```
>>> encoding = "impfY-zVJM2lwAON9ZmAAQ"
>>> encoding.swapcase()
'IMPFy-Zvjm2LWaon9zMaaq'
```

### index(), find() 메서드
`A.index(sub, start, end)`는 문자열 A에서 부분 문자열 sub의 인덱스 위치를 [start, end) 인덱스 
범위에서 찾는다. 값을 찾지 못하면 ValueError 예외를 발생시킨다. `A.find(sub, start, end)`는 같은 기능을 
수행하지만 값을 찾지 못할 경우 `-1`을 반환하는 점만 다르다. 인덱스 start와 end는 생략가능하며, 디폴트는 각각 0(문자열 
시작 인덱스)과 len(A) (문자열 끝 인덱스를 포함)이다.

```
>>> encoding = "impfY-zVJM2lwAON9ZmAAQ"
>>> encoding.index("mA")
18
>>> encoding.find("mA")
18
>>> encoding.swapcase().index("mA")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: substring not found
>>> encoding.swapcase().find("mA")
-1
```

### count() 메서드
`A.count(sub, start, end)`는 문자열 A에서 인덱스 [start, end) 내의 부분 문자열 sub가 등장한 
정수 횟수를 반환한다.

```
>>> encoding = "impfY-zVJM2lwAON9ZmAAQ"
>>> encoding.count("m")
2
>>> encoding.count("m", 0, 1)
0
```

### replace() 메서드
`A.replace(old, new, maxreplace)`는 문자열 A의 왼쪽에서부터 문자열 old를 찾아 문자열 new로 
maxreplace 횟수만큼 대체한 문자열의 복사본을 반환한다. maxreplace를 지정하지 않으면 모든 old를 new로 대체한다.

```
>>> encoding = "impfY-zVJM2lwAON9ZmAAQ"
>>> encoding.replcae("m", "vb", 1)
ivbpfY-zVJM2lwAON9ZmAAQ
```

### f-strings
f-스트링(formatted string literal)은 파이썬 3.6부터 사용 가능한 방식으로, %나 format에 비해 간결한 
코드를 작성할 수 있으며 속도도 빨라졌다. 다음과 같이 format specifier들을 활용해 작성할 수 있다.

```
>>> name = "샤샤"
>>> f"선생님의 이름은 {name!r}이다."
'선생님의 이름은 '샤샤'이다.'
>>> number = 1024583
>>> f"{number: 02b}"
'11111010001001000111'
>>> import decimal
>>> width = 10
>>> precision = 4
>>> pi = decimal.Decimal("3.14159")
>>> f"파이 값: {pi:{width}.{precision}}"
'파이 값:      3.142'
>>> from datetime import datetime
>>> now = datetime(year=2022, month=10, day=31, hour=16, minute=24)
>>> f"{now:%b %d, %Y %I:%M %p}"
'Oct 31, 2022 04:24 PM'
```

정수 포맷 지정자(specifier)는 [여기](https://docs.python.org/ko/3/library/string.html#format-string-syntax)를, 날짜 포맷 지정자는 [여기](https://docs.python.org/ko/3/library/datetime.html#strftime-and-strptime-format-codes)를 참고하자.

### 참고자료
- An Introduction to Python & Algorithms, Mia Stein
- [PEP 498](https://peps.python.org/pep-0498/)
- [Formatted string literals](https://docs.python.org/3/reference/lexical_analysis.html#f-strings)