#include <stdio.h>

__global__ void hello(void) {
	printf("Hello CUDA %d!\n", threadIdx.x);
}

int main(void) {
	hello << <8, 2 >> > ();
	fflush(stdout);
	return 0;
}