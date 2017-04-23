#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <xmmintrin.h>

#include <iostream>
#include <chrono>

#define NOW() std::chrono::high_resolution_clock::now()
#define ELAPSED(msg, x, n) std::cout << msg << ": " \
  << (float)(n) * 0.001 / (float)(std::chrono::duration_cast<std::chrono::microseconds>(NOW() - x).count()) \
  << "GFPS" << std::endl


void gemm(float * a, float * b, float * c, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int p = 0; p < n; ++p) {
	c[i * n + j] += a[i * n + p] * b[p * n + j];
      }
    }
  }
}


void gemm1(float * a, float * b, float * c, int n) {
  for (int j = 0; j < n; j += 4) {
    for (int i = 0; i < n; ++i) {
      float * cij = c + i * n + j;
      float * ai = a + i * n;
      __m128 cij4 = {cij[0], cij[1], cij[2], cij[3]};
      float * bpj = b + j;
      for (int p = 0; p < n; p += 4){
	__m128 aip4 = _mm_set_ps1(*ai);
	cij4 = _mm_add_ps(_mm_mul_ps(aip4, *((__m128 *)bpj)), cij4);
	++ai;
	bpj += n;
	
	aip4 = _mm_set_ps1(*ai);
	cij4 = _mm_add_ps(_mm_mul_ps(aip4, *((__m128 *)bpj)), cij4);
	++ai;
	bpj += n;
	
	aip4 = _mm_set_ps1(*ai);
	cij4 = _mm_add_ps(_mm_mul_ps(aip4, *((__m128 *)bpj)), cij4);
	++ai;
	bpj += n;

	aip4 = _mm_set_ps1(*ai);
	cij4 = _mm_add_ps(_mm_mul_ps(aip4, *((__m128 *)bpj)), cij4);
	++ai;
	bpj += n;
      }
      cij[0] = cij4[0];
      cij[1] = cij4[1];
      cij[2] = cij4[2];
      cij[3] = cij4[3];
    }
  }
}

int main(int argc, char ** argv) {
  srand(time(NULL));
  for (int size = 200; size < 1000; size += 100) {
    int sz = size * size;
    float * a = (float*) malloc(sz * sizeof(float));
    float * b = (float*) malloc(sz * sizeof(float));
    float * c = (float*) malloc(sz * sizeof(float));
    for (int i = 0; i < size * size; ++i) {
      a[i] = (float((i * 11) % 10000) / 10000.0) - 0.5;
      b[i] = (float((i * 7) % 10000) / 10000.0) - 0.5;
      c[i] = 0.0;
    }
    
    float total = size;
    total *= (float)size;
    total *= (float)size;
    auto start = NOW();
    gemm1(a, b, c, size);
    ELAPSED(std::to_string(size), start, total);
    
    /*
    if (size == 200) {
      float * c1 = new float[size * size];
      for (int i = 0; i < size * size; ++i)  c1[i] = 0.0;
      gemm(a, b, c1, size);
      for (int i = 0; i < size * size; ++i) {
	if (fabs(c[i] - c1[i]) > 1e-9) {
	  std::cout << fabs(c[i] - c1[i]) << std::endl;
	}
      }
      delete []c1;
    }
    */
    free(a);
    free(b);
    free(c);
  }
}

