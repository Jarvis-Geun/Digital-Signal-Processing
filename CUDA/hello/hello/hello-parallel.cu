#include <stdio.h>

__global__ void hello(void) {
	printf("Hello CUDA %d!\n", threadIdx.x);
}

int main(void) {
	hello << <1, 8 >> > ();
	fflush(stdout);
	return 0;
}