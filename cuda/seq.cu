#include <iostream>
using namespace std;

#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int main()
{
  const int N = 50000;
  thrust::device_vector<int> a(N);
  thrust::sequence(a.begin(), a.end(), 0);
  int sumA = thrust::reduce(a.begin(), a.end(), 0);
  
  int sumCheck = 0;
  for (int i = 0; i < N; i++) sumCheck += i;
  
  if (sumA == sumCheck) cout << "Test Succeeded!" << endl;
  else {
    cout << "Test Failed" << endl;
    return 1;
  }
  return 0;
}