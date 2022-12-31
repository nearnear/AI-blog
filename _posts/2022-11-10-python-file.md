---
title: "python 파일 처리"
categories: Python
tags:
    - file
    - shutil
    - pickle
    - struct
---

Python으로 파일을 다루는 방법을 알아본다.


## 파일 처리

예시를 통해 알아보자. 다음 코드는 파일을 읽어들이고 빈줄을 제거해 저장한다.

```python
import sys

# filename 파일에서 strip이 존재하는 줄만 리스트에 저장하여 반환한다.
def read_file_only_strip(filename):
    lines = []
    with open(filename) as f: # 파일 읽기
        for line in f:
            if line.strip():
                lines.append(line)
    return lines

# lines 리스트의 항목을 하나씩 filename 파일에 쓴다.
def write_file(lines, filename):
    f = None
    with open(filename, "w") as f: # 파일 쓰기
        for line in lines:
            f.write(line)

# 앞서 정의한 두 함수로 파일을 읽어들여 strip을 삭제해 저장한다. 
def remove_blank_lines():
    # 파일이 없는 경우, 아무것도 출력하지 않는다.
    if len(sys.argv) < 2:
        print("")

    for filename in sys.argv[1:]:
        lines = read_file_only_strip(filename)
        if lines:
            write_file(lines, filename)

if __name__ == "__main__":
    remove_blank_lines()
```


## 파일 처리 메서드

### open(), close()

`open(filename, mode, encoding)` 메서드는 파일 객체를 반환한다. 
`close()`는 파일 객체를 닫으며 try .. except .. finally 구문으로 다음과 같이 사용할 수 있다.

```python
def read_file_lines(filename):
    lines = []
    f = None
    
    try:
        f = open(filename)
        for line in f:
            if line.strip():
                lines.append(line)
    except (IOError, OSError) as err:
        print(err)
    finally:
        if f:
            f.close()
            
    return lines
```

또는 처음의 예시처럼 `with open(...) as f_object`로 close()를 명시하지 않고 같은 기능을 구현할 수 있다.
close()는 열린 파일이 차지하는 시스템 자원을 해제(free up)한다. 파일을 닫으면 True를 반환한다.

open()의 인자인 mode는 문자열로 지정하며 디폴트는 읽기 모드(`r`)이다.
- `r` : 파일 읽기 모드.
- `w` : 파일 쓰기 모드. 동명 파일이 있는 경우, 대체한다.
- `a` : 파일 추가 모드. 동명 파일이 있는 경우, 그 파일 끝에 내용을 추가한다.
- `r+` : 파일 읽기와 쓰기 모드.
- `t` : 파일의 종류인 텍스트 모드.
- `b` : 파일의 종류인 바이너리 모드.
예를 들어 `rt`는 텍스트 모드로 읽기를, `rb`는 바이너리로 읽기이다. `wt`, `wb`, `at` 등도 마찬가지이다.

encoding도 문자열로 지정하며 디폴트는 시스템 기본 인코딩이다. 맥 OS의 디폴트는 `'utf-8'`이며 윈도우는 `'cp949'`이다.

### read(), readline(), readlines()
`read(size)`는 size 인자 만큼의 바이트 수로 내용을 읽어들여 open()에서 지정한 mode로 반환한다. 인수가 생략된 
경우 전체 파일의 내용을 반환한다. 파일의 끝에서는 빈 문자열을 반환한다. 
`readline()`은 한 줄만 읽어들인다. 파일의 마지막 행에서만 개행 문자가 생략된다. 
`readlines(size)`는 size 만큼 읽어들이고, 행을 완성하는 데 필요한 만큼 더 읽어서 반환한다. 
인수를 전달하지 않으면 모든 데이터 행을 포함한 리스트를 반환한다. 즉 언제나 완전한 행을 반환한다. 

```python
>>> f = open('open_me.py')
>>> f.readlines()
['def function1():\n', "    print('Did you call me?')\n", '\n', '\n', "if __name__ == '__main__':\n", '    function1()\n']
```

### write()
`write(text)`는 text를 모드에서 지정한 객체로 파일에 쓴 다음, None을 반환한다. 

```python
>>> f = open('new_file.txt', 'w')
>>> for i in range(5):
...     f.write(f"{i} 번째 행입니다.\n")
>>> f = open('new_file.py', 'r')
>>> f.readlines()
['0 번째 행입니다.', '1 번째 행입니다.', '2 번째 행입니다.', '3 번째 행입니다.', '4 번째 행입니다.']
```

### tell(), seek()
`tell()`은 파일의 현재 위치를 나타내는 바이트단위 정수를 반환한다. `seek(offset, from-what)`은
파일내 탐색 위치를 from-what 포인트에서 offset을 더한 값으로 변경하여 반환한다. 

```python
>>> f.tell()
110
>>> f.seek(10, 0)
10
>>> f.tell()
10
```

### input()
`input()`은 사용자의 입력을 받으며, return 키를 누를 때까지 대기한다.

### fileno()
`fileno()`는 파일에 접근하는 OS의 추상 키인 파일 서술자(file descriptor)를 반환한다.

```python
>>> f.fileno()
8
```


## shutil 모듈

shutil 모듈을 통해 터미널에서 파일을 조작할 수 있다. 다음과 같은 `change_ext_file.py`가 있을 때:

```python
import os
import sys
import shutil

# 특정 파일의 확장자를 바꾼다.
def change_file_ext():
    # 만약 한개 이하의 sys만 있다면 상태를 출력하고 exit한다.
    if len(sys.argv) < 2:
        print(f"Usage : python {sys.argv[0]} filename.old_ext 'new_ext'")
        sys.exit()

    # sys.argv에서 이름을 가져와 출력한다.
    name = os.path.splitext(sys.argv[1] + '.' + sys.argv[2])
    print(name)

    # 새 확장자의 이름으로 복사본을 만든다.
    try:
        shutil.copyfile(sys.argv[1], name)
    except OSError as err:
        print(err)

if __name__ == '__main__':
    change_file_ext()
```

이제 이 파일을 활용해 터미널에서 `some_file.py`를 `txt`확장자로 바꿀 수 있다.

```python
$ python change_ext_file.py some_file.py txt
```


## pickle 모듈

$$
\text{python object} 
\xrightleftharpoons[\text{unpickling (deserialization)}]{\text{pickling (serialization)}}
\text{str(binary)}
$$

pickle 모듈을 통해 파이썬 객체를 바이너리 형태로 저장했다가 다시 불러낼 수 있다. 정확히는 바이너리 모드로
파일에 접근하여 파이썬 객체를 문자열 표현으로 변환하여 피클링(직렬화)을 한 뒤, 저장된 피클을 꺼내서 언피클링(역직렬화) 한다.

```python
>>> import pickle
>>> x = {'circum1': 'action1', 'circum2': 'action2'}
>>> with open("tactics.pkl", "wb") as f: # pickle!
...     pickle.dump(x, f)
...
>>> with open("tactics.pkl", "rb") as f: # unpickle!
...     tactics = pickle.load(f)
...
>>> tactics
{'circum1': 'action1', 'circum2': 'action2'}
```


## struct 모듈

struct 모듈을 통해 파이썬 객체를 이진 표현으로 변환하거나 역변환할 수 있다. 객체는 특정 길이의 문자열만 처리한다.

```python
>>> import struct
>>> abc = struct.pack('>hhl', 1, 2, 3)
>>> abc
b'\x00\x01\x00\x02\x00\x00\x00\x03'
>>> struct.unpack('>hhl', abc)
(1, 2, 3)
>>> struct.calcsize('>hhl')
8
```

struct에는 포맷을 전달하기 위한 Format character가 있다. 
[공식문서](https://docs.python.org/3/library/struct.html#format-characters)를 참고하자. 
여기서 `>`은 big-endian, `h`는 C 타입 short, `l`은 C 타입 long을 의미한다.
- `pack(format, v1, v2, ...)`은 v1, v2, ... 를 format 형식을 따라 바이트 객체를 반환한다.
- `unpack(format, buffer)`는 format 형식의 byte 또는 bytearray 객체를 buffer로 받아 값을 
  반환한다.
- `calcsize(format)`는 format이 차지할 바이트 수를 반환한다.

format과 buffer가 일치하지 않는 경우, struct.error 예외가 발생한다.
```
>>> struct.unpack('>hll', abc)
Traceback (most recent call last):
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/code.py", line 90, in runcode
    exec(code, self.locals)
  File "<input>", line 1, in <module>
struct.error: unpack requires a buffer of 10 bytes
```

### 참고자료
- An Introduction to Python & Algorithms, Mia Stein