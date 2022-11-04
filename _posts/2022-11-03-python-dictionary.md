---
title: "python 컬렉션 자료구조 - dict"
categories: Python
tags:
  - python
  - data structure
  - dictionary
---

Python의 dict(딕셔너리)는 해시 테이블로 구현되어 있다. 따라서 딕셔너리의 고유한 키에 해당하는 값을 상수 시간 내에 얻을 수 
있다. 딕셔너리는 가변 객체이므로 항목의 추가 및 제거가 가능하지만, 인덱스 위치를 사용하여 접근하는 것은 불가능하며
삽입 순서를 기억하지 않는다.

컬렉션 매핑 타입(mapping type)인 딕셔너리는 iterable 객체로, 멤버십 연산자 `in`과 `len()` 함수를 지원한다. 
매핑은 key-value 항목의 컬렉션으로 각 항목에 대해 메서드를 제공한다. 

```
>>> mia_info = {'name': '미아', 'age': 45, 'job': 'teacher'}
>>> mia_info
{'name': '미아', 'age': 45, 'job': 'teacher'}
>>> mia_info = dict(name='미아', 'age'=45, 'job'='teacher')
>>> mia_info
{'name': '미아', 'age': 45, 'job': 'teacher'}
>>> mia_info = dict([('name', '미아'), ('age', 45), ('job', 'teacher')])
{'name': '미아', 'age': 45, 'job': 'teacher'}
```


## 딕셔너리 메서드

### setdefault()
`A.setdefault(key, default)`는 딕셔너리 A에 key가 존재하는 경우 key에 해당하는 값을 반환하고, 
key가 존재하지 않으면 key를 키로 default를 값으로하는 항목을 딕셔너리에 추가한다.

```python
def pairs_to_dict(data_pairs):
    new_dict = {}
    for k, v in data_pairs:
        new_dict.setdefault(k, []).append(v)
    return new_dict

def test():
    data = ((1, 5), (2, 1), (2, 3), (2, 5), (3, 5), (4, 5))
    d1 = pairs_to_dict(data)
    print(f'Result\t: {d1}')

if __name__ == '__main__':
    test()
```

```python
Result	: {1: [5], 2: [1, 3, 5], 3: [5], 4: [5]}
```

### update()
`A.update(B)`는 딕셔너리 A에 딕셔너리 B의 key-value 쌍을 갱신한다.
A에 B의 key k가 존재할 때, 해당 A의 k의 값을 B의 k에 해당하는 값으로 갱신한다. 
A에 존재하지 않는 B의 key-value 쌍은 A에 추가한다. 

```
>>> menu_to_price = {'말차 라떼': 4000, '카페 라떼': 3800}
>>> menu_to_price.update({'말차 라떼': 5600, '아메리카노': 3000})
>>> menu_to_price
{'말차 라떼': 5600, '카페 라떼': 3800, '아메리카노': 3000}
```

### get()
`A.get(key, default)`는 딕셔너리 A의 key 값을 반환하되, key가 존재하지 않을 경우 default 값을 반환한다. 
default를 지정하지 않으면 None을 반환한다. get 메서드는 setdefault나 update와 달리 딕셔너리 A를 변경하지 않는다.

```
>>> menu_to_price = {'말차 라떼': 5600, '카페 라떼': 3800, '아메리카노': 3000}
>>> menu_to_price.get('카페 라떼', 500)
3800
>>> menu_to_price.get('핫초코', 500)
500
>>> menu_to_price
{'말차 라떼': 5600, '카페 라떼': 3800, '아메리카노': 3000}
```

### items(), values(), keys()

items(), values(), keys()는 딕셔너리 view로 딕셔너리의 항목을 조회하는 읽기 전용 객체다.

```
>>> menu_to_price = {'말차 라떼': 5600, '카페 라떼': 3800, '아메리카노': 3000}
>>> menu_to_price.items()
dict_items([('말차 라떼', 5600), ('카페 라떼', 3800), ('아메리카노', 3000)])
>>> menu_to_price.values()
dict_values([5600, 3800, 3000])
>>> menu_to_price.keys()
dict_keys(['말차 라떼', '카페 라떼', '아메리카노'])
```

### pop(), popitem()
`A.pop(key)`는 딕셔너리 A의 key 항목을 제거한 뒤 그 값을 반환한다. 
`A.popitem(key)`는 딕셔너리 A의 key 항목을 제거한 뒤 key와 값 쌍을 반환한다.

```
>>> menu_to_price = {'알리오올리오': 12000, '시금치 감자 뇨끼': 15600}
>>> menu_to_price.pop('알리오올리오')
12000
>>> menu_to_price
{'시금치 감자 뇨끼': 15600}
>>> menu_to_price.popitem('시금치 감자 뇨끼')
('시금치 감자 뇨끼', 15600)
>>> menu_to_price
{}
```

### clear()
딕셔너리의 항목을 모두 제거한다.

```
>>> menu_to_price = {'알리오올리오': 12000, '시금치 감자 뇨끼': 15600}
>>> menu_to_price.clear()
>>> menu_to_price
{}
```


## 딕셔너리 순회
딕셔너리를 순회하기 위해서 임의의 순서로 나타나는 키를 사용할 수 있다. 또는 키를 sorted()로 정렬하여 탐색할 수도 있다.

```
>>> d1 = {-k: k + 5 for k in range(5)}
>>> d1
{0: 5, -1: 6, -2: 7, -3: 8, -4: 9}
>>> for key in sorted(d1.keys()):
...     print(key, d1[key])
...
-4 9
-3 8
-2 7
-1 6
0 5
```

## 딕셔너리 분기
딕셔너리를 활용해 if ~ elif ~ 문을 구현할 수 있다.

```python
def plan_a():
    print('Plan A')

def plan_b():
    print('Plan B')

if __name__ == '__main__':
    action = 'run'
    functions = dict(walk=plan_a, run=plan_b)
    functions[action]()
```

```
Plan B
```

### 참고자료
- An Introduction to Python & Algorithms, Mia Stein