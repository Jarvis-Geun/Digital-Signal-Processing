# Heap sort (힙 정렬)
### Reference
- [힙(heap) 이란](https://gmlwjd9405.github.io/2018/05/10/data-structure-heap.html)
- [힙 정렬(heap sort)이란](https://gmlwjd9405.github.io/2018/05/10/algorithm-heap-sort.html)
- [[도서] C언어로 쉽게 풀어쓴 자료구조](http://www.yes24.com/Product/Goods/69750539)

<br>

## 간단 요약

### 힙
- 완전 이진트리의 일종, 우선순위 큐를 위하여 만들어진 자료구조
- 프로그램 구현을 쉽게 하기 위하여 배열의 `첫번째 인덱스인 0은 사용되지 않음`
- 중복된 값 허용 ➡️ 이진 탐색 트리는 허용 X

<br>

`-` 최대 히프(max heap) ➡️ key(부모 노드) >= key(자식 노드)
- 부모 노드의 키값이 자식 노드의 키값보다 크거나 같은 완전 이진 트리(내림차순 느낌)

<br>

`-` 최소 히프(min heap) ➡️ key(부모 노드) <= key(자식 노드)
- 부모 노드의 키값이 자식 노드의 키값보다 작거나 같은 완전 이진 트리(오름차순 느낌)

<br>

`-` 히프트리에서의 자식노드와 부모노드의 관계
- 왼쪽 자식의 인덱스 = (부모의 인덱스) * 2
- 오른쪽 자식의 인덱스 = (부모의 인덱스) * 2 + 1
- 부모의 인덱스 = (자식의 인덱스) / 2

<br>

`-` 힙의 삽입 연산
- [링크 참고](https://gmlwjd9405.github.io/2018/05/10/data-structure-heap.html)
- 힙에 새로운 요소가 들어오면, 새로운 노드를 힙의 마지막 노드로 삽입함

```C
// Reference : C언어로 쉽게 풀어쓴 자료구조

// 힙 트리의 삽입 알고리즘(pseudo code)
insert_max_heap(A, key):

1. heap_size <- heap_size + 1;
2. i <- heap_size;
3. A[i] <- key;
4. while i != 1 and A[i] > A[PARENT(i)] do
5.     A[i] <-> A[PARENT];
6.     i <- PARENT(i);
```

```C
// 힙 트리의 삽입 알고리즘(pseudo code) 분석
insert_max_heap(A, key):

// 1. 힙 크기를 하나 증가시킨다.
1. heap_size <- heap_size + 1;

// 2 ~ 3. 증가된 힙 크기 위치에 새로운 노드를 삽입한다.
2. i <- heap_size;
3. A[i] <- key;

// 4. i가 루트 노드가 아니고 i번째 노드가 i의 부모 노드보다 크면
4. while i != 1 and A[i] > A[PARENT(i)] do

// 5. i번째 노드와 부모 노드를 교환
5.     A[i] <-> A[PARENT(i)];

// 6. 한 레벨 위로 올라간다(승진).
6.     i <- PARENT(i)
```

<br>

`-` 힙의 삭제 연산
- [링크 참고](https://gmlwjd9405.github.io/2018/05/10/data-structure-heap.html)
- 최대값을 가진 요소를 삭제하는 것(최대 힙 기준) ➡️ 루트 노드(최대값) 삭제

```C
// Reference : C언어로 쉽게 풀어쓴 자료구조
// 힙 트리의 삭제 알고리즘

delete_max_heap(A):

1. item <- A[1];
2. A[1] <- A[heap_size];
3. heap_size <- heap_size - 1;
4. i <- 2;
5. while i <= heap_size do
6.     if i < heap_size and A[i+1] > A[i]
7.         then largest <- i + 1;
8.         else largest <- i;
9.     if A[PARENT(largest)] > A[largest]
10.        then break;
11.    A[PARENT(largest)] <-> A[largest];
12.    i <- CHILD(largest);

return item;
```
```C
// 힙 트리의 삭제 알고리즘

delete_max_heap(A):

// 1. 루트 노드 값 반환을 위하여 item 변수로 옮긴다.
1. item <- A[1];

// 2. 말단 노드를 루트 노드로 옮긴다.
2. A[1] <- A[heap_size];

// 3. 힙의 크기를 하나 줄인다.
3. heap_size <- heap_size - 1;

// 4. 루트의 왼쪽 자식부터 비교를 시작한다.
4. i <- 2;

// 5. i가 힙트리의 크기보다 작으면 (즉, 힙트리를 벗어나지 않았으면)
5. while i <= heap_size do

// 6. 오른쪽 자식이 더 크면
6.     if i < heap_size and A[i+1] > A[i]

// 7 - 8. 두 개의 자식 노드 중 큰 값의 인덱스를 largest로 옮긴다.
7.         then largest <- i + 1;
8.         else largest <- i;

// 9 - 10. largest의 부모 노드가 largest보다 크면 중지
9.     if A[PARENT(largest)] > A[largest]
10.        then break;

// 11. 그렇지 않으면 largest와 largest 부모 노드를 교환한다.
11.    A[PARENT(largest)] <-> A[largest];

// 12. 한 레벨 밑으로 내려간다.
12.    i <- CHILD(largest);

// 13. 최대값을 반환한다.
13. return item;
```
