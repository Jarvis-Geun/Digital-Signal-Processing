#include <cstdio>

// __global__ : CUDA���� ����� �Լ���� �ǹ�
__global__ void hello(void) {
	printf("Hello, CUDA!\n");
}

int main(void) {
	// CUDA ����ó���� ���� ���, CUDA�� call�� ���̶�� �ǹ�
	hello << <1, 1 >> > ();
	return 0;
}