#include <cstdio>
#include <cstdlib>
#include <iostream>
using namespace std;

#include <cuda_runtime.h>

#define CUDA_CALL(func, name)						\
  {									\
    cudaError_t e = (func);						\
    if(e != cudaSuccess)						\
      cout << "CUDA: " << cudaGetErrorString(e) << ": " << name << endl; \
    else								\
      cout << "CUDA SUCC: " << (name) << endl;				\
  }

void fill_array(int * data, const int num) {
  for(int i = 0; i < num; i++){
    data[i] = i;
  }
}

void check_array(char * device_prefix,
		 int * data,
		 const int num) {
  bool error_found = false;
  for(int i = 0; i < num; i++) {
    if(data[i] != i * 2){
      cout << "error: " << device_prefix << "\t" << i << "\t" << data[i] << endl;
      error_found = true;
    }
  }
  if (!error_found)
    cout << "passed: " << device_prefix << endl;
}

__global__ void gpu_test_kernel(int * data) {
  const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  for(int i = 0; i < 10000; i++){
    data[tid] *= 2;
    data[tid] /= 2;
  }
  data[tid] *= 2;
}

#define MAX_NUM_DEVICES (4)
#define NUM_ELEM (1024*1024*8)

cudaStream_t stream[MAX_NUM_DEVICES];

char device_prefix[MAX_NUM_DEVICES][300];

int * gpu_data[MAX_NUM_DEVICES];
int * cpu_src_data[MAX_NUM_DEVICES];
int * cpu_dst_data[MAX_NUM_DEVICES];

cudaEvent_t kernel_start_event[MAX_NUM_DEVICES];
cudaEvent_t memcpy_to_start_event[MAX_NUM_DEVICES];
cudaEvent_t memcpy_from_start_event[MAX_NUM_DEVICES];
cudaEvent_t memcpy_from_stop_event[MAX_NUM_DEVICES];

__host__ void gpu_kernel(void) {

  const int shared_memory_usage = 0;
  const size_t single_gpu_chunk_size = sizeof(int) * NUM_ELEM;
  const int num_threads = 256;
  const int num_blocks = ((NUM_ELEM + (num_threads - 1)) / num_threads);
  cout << "begin" << endl;

  int num_devices;
  CUDA_CALL(cudaGetDeviceCount(&num_devices), "cudaGetDeviceCount");
  if(num_devices > MAX_NUM_DEVICES)
    num_devices = MAX_NUM_DEVICES;
  
  cout << "num devices: " << num_devices << endl;

  for(int device_num = 0; device_num < num_devices; device_num++) {
    CUDA_CALL(cudaSetDevice(device_num), "cudaSetDevice");
    
    struct cudaDeviceProp device_prop;
    CUDA_CALL(cudaGetDeviceProperties(&device_prop,
				      device_num), "cudaGetDeviceProperties");
    sprintf(&device_prefix[device_num][0], "\nID: %d %s : ", device_num,
	    device_prop.name);
    
    CUDA_CALL(cudaStreamCreate(&stream[device_num]), "cudaStreamCreate");
    CUDA_CALL(cudaMalloc((void**)&gpu_data[device_num], single_gpu_chunk_size), "cudaMalloc");

    CUDA_CALL(cudaMallocHost((void**)&cpu_src_data[device_num],
			     single_gpu_chunk_size), "cudaMallocHost");

    CUDA_CALL(cudaMallocHost((void**)&cpu_dst_data[device_num],
			     single_gpu_chunk_size), "cudaMallocHost");

    fill_array(cpu_src_data[device_num], NUM_ELEM);

    CUDA_CALL(cudaEventCreate(&memcpy_to_start_event[device_num]), "create memcpy_to_start_event");
    CUDA_CALL(cudaEventCreate(&kernel_start_event[device_num]), "create kernel_start_event");
    CUDA_CALL(cudaEventCreate(&memcpy_from_start_event[device_num]), "create memcpy_from_start_event");
    CUDA_CALL(cudaEventCreate(&memcpy_from_stop_event[device_num]), "create memcpy_from_stop_event");

    CUDA_CALL(cudaEventRecord(memcpy_to_start_event[device_num]), "memcpy_to_start_event");
    CUDA_CALL(cudaMemcpyAsync(gpu_data[device_num],
			      cpu_src_data[device_num],
			      single_gpu_chunk_size,
			      cudaMemcpyHostToDevice,
			      stream[device_num]), "cudaMemcpyAsync");

    CUDA_CALL(cudaEventRecord(kernel_start_event[device_num]), "cudaEventRecord");
    gpu_test_kernel<<<num_blocks, 
      num_threads, 
      shared_memory_usage,
      stream[device_num]>>>(gpu_data[device_num]);

    CUDA_CALL(cudaEventRecord(memcpy_from_start_event[device_num]), "memcpy_from_start_event");
    CUDA_CALL(cudaMemcpyAsync(cpu_dst_data[device_num],
			      gpu_data[device_num],
			      single_gpu_chunk_size,
			      cudaMemcpyDeviceToHost,
			      stream[device_num]), "cudaMemcpyAsync");
    CUDA_CALL(cudaEventRecord(memcpy_from_stop_event[device_num]), "memcpy_from_stop_event");
  }
  
  for(int device_num = 0; device_num < num_devices; device_num++) {
    CUDA_CALL(cudaSetDevice(device_num), "");
    CUDA_CALL(cudaStreamSynchronize(stream[device_num]), "");

    float time_copy_to_ms;
    CUDA_CALL(cudaEventElapsedTime(&time_copy_to_ms,
				   memcpy_to_start_event[device_num],
				   kernel_start_event[device_num]), "");

    float time_kernel_ms;
    CUDA_CALL(cudaEventElapsedTime(&time_kernel_ms,
				   kernel_start_event[device_num],
				   memcpy_from_start_event[device_num]), "");

    float time_copy_from_ms;
    CUDA_CALL(cudaEventElapsedTime(&time_copy_from_ms,
				   memcpy_from_start_event[device_num],
				   memcpy_from_stop_event[device_num]), "");

    float time_exec_ms;
    CUDA_CALL(cudaEventElapsedTime(&time_exec_ms,
				   memcpy_to_start_event[device_num],
				   memcpy_from_stop_event[device_num]), "");    

    CUDA_CALL(cudaStreamDestroy(stream[device_num]), "");
    CUDA_CALL(cudaFree(gpu_data[device_num]), "");

    check_array(device_prefix[device_num],
		cpu_dst_data[device_num],
		NUM_ELEM);

    CUDA_CALL(cudaFreeHost(cpu_src_data[device_num]), "");
    CUDA_CALL(cudaFreeHost(cpu_dst_data[device_num]), "");
    CUDA_CALL(cudaDeviceReset(), "");

    cout << time_copy_to_ms << "\t" << time_kernel_ms << "\t" << time_copy_from_ms << "\n"
	 << time_exec_ms << "\t" << time_copy_to_ms + time_kernel_ms + time_copy_from_ms << endl;
  }
}

int main(){
  gpu_kernel();
}