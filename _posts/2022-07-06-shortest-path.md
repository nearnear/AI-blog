---
title: 최단 경로 알고리즘
categories: Algorithm
tags:
    - algorithm
    - Dijkstra algorithm
    - Floyd-Warshall algorithm
---

**Note:** 본 글은 이전에 작성한 글을 백업한 것입니다.
{: .notice--info}

## 최단 경로 알고리즘

그래프는 간선에 가중치 정보를 추가할 수 있다. 이런 그래프를 가중 그래프(weighted graph) 라고 하며, 자연스럽게 출발 노드에서 특정 노드로 가는 경로의 가중치 합이 최소가 되는 경로를 찾는 문제가 발생한다. 깊이 우선 탐색(DFS)에서 이동하는 `경로의 개수를 최소화`하는 것이 목적이었다면 이 문제에서는 가중 그래프에서 `경로 가중치의 합을 최소화` 하는 것이 목적이며, 이 문제를 `최단 경로 알고리즘`이라고 부른다.

이 글에서는 두개의 최단 경로 알고리즘을 정리하고자 한다.

1. Dijkstra 알고리즘
2. Floyd-Warshall 알고리즘

## 1. Dijkstra 알고리즘
Dijkstra 알고리즘은 최단 경로 알고리즘으로, 시작 노드에서 모든 노드까지의 최단 경로를 계산한다. Dijkstra 알고리즘은 매 시점에서 가장 비용이 적은 노드를 선택하는 **그리디 알고리즘**이며, 이 때문에 가중치가 모두 양수(positive number)인 유향(directed) 그래프에 대해서만 작동한다.

### 📂 우선 순위 큐를 활용한 Dijkstra 알고리즘 구현
_최단 거리를 정렬하기 위해 우선 순위 큐를 사용하지만, 큐를 사용하지 않고서도 알고리즘을 구현할 수 있다._

Dijkstra 알고리즘은 우선 순위 큐를 활용한 재귀 함수로 구현할 수 있다. 
1. 저장된 최단 거리가 입력 받은 거리보다 짧은 경우, 함수를 종료한다.
2. 1번에서 끝나지 않은 경우, 현재 노드에서 모든 연결된 노드로 가는 경로를 고려해, 이미 저장된 경로와 거리를 비교한다. 저장된 최단 거리가 계산한 거리보다 긴 경우, 더 짧은 거리로 `dist` 배열의 해당 값을 변경한다.
3. 우선 순위 큐가 빌 때 까지 위 과정을 반복한다.

#### 파이썬 코드
```python
from heapq import heappush, heappop
from collections import defaultdict

# 노드 개수 n과 간선 개수 m을 입력받는다.
n, m = map(int, input().split())

# 시작 노드를 입력 받는다.
start = int(input())

# 1차원 배열 graph를 초기화한다.
graph = defaultdict(list)

# graph[i]는 (node_number, weight)를 원소로 하는 리스트다.
for _ in range(m):
    i, j, w = map(int, input().split())
    graph[i].append(j, w)

# graph와 노드 개수 n을 입력받아
# `시작 노드`에서 `모든 노드`까지의 최단 거리를 반환한다. 
def shortest_path(graph, n, start):

    # distance[i] : 시작 노드에서 노드 i 까지의 최단 거리
    # 큰 값으로 초기화한다.
    INF = int(1e9)
    distance = [INF] * (n + 1)

    def dijkstra(start):
        # q의 원소: (shortest_distance, node_number)
        # 시작노드에서 시작노드까지의 거리는 0이다.
        hq = []
        heappush(hq, (0, start))
        distance[start] = 0

        # q가 존재하는 한 계속한다.
        while hq:
            dist, now = heappop(q)
            # 저장된 최단 거리가 계산한 거리보다 짧은 경우, 
            # 변경하지 않는다.
            if distance[now] < dist:
                continue
            # 연결된 노드에 대해 새로운 경로의 거리를 비교한다.
            for n, d in graph[now]:
                w_dist = dist + d
                # 저장된 최단 거리가 계산한 거리보다 긴 경우, 
                # 더 짧은 거리로 dist 값을 변경한다.
                if w_dist < distance[n]:
                    distance[n] = w_dist
                    heappush(hq, (w_dist, n))

    # 시작노드에 대해 dijkstra를 구현한다.
    dijkstra(start)
    return distance

shortest = shortest_path(graph, n, start))
for i in range(n):
    print(f"Shortest Path from node {start} to node {i}: {shortest[i+1]}")
```
시간 복잡도는 우선 순위 큐의 정렬에 의해 노드 개수 $N$과 가중치 개수 $M$에 대해 $O(MlogN)$이다.

## 2. Floyd-Warshall 알고리즘

Floyd-Warshall 알고리즘은 모든 노드에서 모든 노드까지의 최단 경로를 계산하는 **다이내믹 프로그래밍** 알고리즘이다. 따라서 점화식을 알기만 하면 구현이 비교적 간단하다는 장점이 있다.

`노드 i`에서 `노드 j`로 가는 임의의 경로가 있다고 하자. 만약 i에서 j로 가는 다른 경로가 있다면 이 경로는 임의의 경로가 지나지 않는 다른 노드를 거쳐갈 것이다. 다른 노드를 임의로 `노드 k`라고 할때, `노드 i`에서 `노드 k`로 다시 `노드 k`에서 `노드 j`로 가는 경로와 그렇지 않은 경로를 비교할 수 있다. $dist[i][j]$를 노드 i에서 j로 가는 최단 경로라고 정의하면 다음과 같이 점화식을 쓸 수 있다.
$$
dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
$$
따라서 노드 개수 N에 대해 $(N, N)$크기의 2차원 배열에 임의의 `노드 i`에서 임의의 `노드 j`로 가는 최단 경로를 저장한다. 

### 📂 Floyd-Warshall 알고리즘 구현
1. 노드 개수 $N$에 대해 $(N, N)$ 크기의 2차원 배열을 초기화한다.
2. 위의 점화식을 이용해 모든 k에 대해 2차원 배열을 순회한다.

#### 파이썬 코드
```python
# 노드 개수 n과 간선 개수 m을 입력받는다.
n, m = map(int, input().split())

# 2차원 dist 배열을 큰 값으로 초기화 한다.
# dist[i][j] : 노드 i에서 노드 j로 가는 최단 경로
INF = 1e9
dist = [INF for _ in range(n)] for _ in range(n)

# 간선 개수만큼 간선 정보를 2차원 배열에 입력받는다.
for _ in range(m):
    i, j, w = map(int, input().split())
    dist[i][j] = w

# dist와 노드 개수 n을 입력받아
# `모든 노드`에서 `모든 노드`까지의 최단 거리를 저장한 2차원 배열을 반환한다. 
for k in range(n+1):
    for i in range(n+1):
        for j in range(n+1):
            # 저장되어 있는 최단 경로 dist[i][j]와
            # 노드 k를 거치는 경로 dist[i][k] + dist[k][j]를 비교한다.
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

for i in range(n):
    for j in range(n):
        print(f"Shortest Path from node {i} to node {j}: {dist[i+1][j+1]}")
```

Floyd-Warshall 알고리즘은 삼중 for문에 의해 시간 복잡도가 $O(N^3)$이므로 노드 개수가 많은 그래프는 수행시간에 유의해야 한다. 


## 참고자료
- (이코테 2021 강의 몰아보기) 7. 최단 경로 알고리즘, https://www.youtube.com/watch?v=acqm9mM1P6o