#include <cstdio>

// __global__ : CUDA에서 실행될 함수라는 의미
__global__ void hello(void) {
	printf("Hello, CUDA!\n");
}

int main(void) {
	// CUDA 병렬처리를 위해 사용, CUDA가 call될 것이라는 의미
	hello << <1, 1 >> > ();
	return 0;
}