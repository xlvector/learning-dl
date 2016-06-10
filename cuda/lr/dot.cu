#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <utility>
#include <cmath>
using namespace std;

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CALL(func)							\
  {									\
    cudaError_t e = (func);						\
    if(e != cudaSuccess)						\
      cout << "CUDA: " << cudaGetErrorString(e) << endl;		\
  }

#define CURAND_CALL(func)			\
  {						\
    curandStatus_t e = (func);			\
    if(e != CURAND_STATUS_SUCCESS)		\
      cout << "CURAND: " << e << endl;		\
  }

#define NUM_THREADS 1024

cudaStream_t stream;

float * copyFloatToGPU(const vector<float> & x) {
  float * ret;
  CUDA_CALL(cudaMalloc((void**)&ret, sizeof(float) * x.size()));
  CUDA_CALL(cudaMemcpyAsync(ret, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice, stream));
  return ret;
}

int * copyIntToGPU(const vector<int> & x) {
  int * ret;
  CUDA_CALL(cudaMalloc((void**)&ret, sizeof(int) * x.size()));
  CUDA_CALL(cudaMemcpyAsync(ret, x.data(), sizeof(int) * x.size(), cudaMemcpyHostToDevice, stream));
  return ret;
}

int total_count(const vector< vector< pair<int, float> > > & data) {
  int ret = 0;
  for(int i = 0; i < data.size(); i++) ret += data[i].size();
  return ret;
}

struct CooMatrix {
  float *val, *err, *label, *act;
  int *row_ind, *col_ind;
  int nnz, nrow;
  int max_length;
};

__global__ void dot(float * val, int *row_ind, int *col_ind, int nnz, float * ret, float * w) {
  const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < nnz) {
    int r = row_ind[tid];
    int c = col_ind[tid];
    float v = val[tid];
    atomicAdd(&ret[r], v * w[c]);
  }
}

__global__ void vec_sigmoid(float * d, int num) {
  const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < num) {
    if(d[tid] > 10.0) d[tid] = 1.0;
    else if(d[tid] < -10.0) d[tid] = 0.0;
    else d[tid] = 1.0 / (1.0 + exp(-1.0 * d[tid]));
  }
}

__global__ void grad(float * val, int * row_ind, int *col_ind, float * mat_err,
		     int nnz, float *act, float *label, 
		     float *w, float learning_rate) {
  const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < nnz) {
    int r = row_ind[tid];
    int c = col_ind[tid];
    float v = val[tid];
    mat_err[tid] = abs(label[r] - act[r]);
    float err = v * (label[r] - act[r]);
    atomicAdd(&w[c], learning_rate * err);
  }
}

CooMatrix zeroCooMatrix(int batch_size, int max_length) {
  CooMatrix mat;
  mat.max_length = max_length;
  CUDA_CALL(cudaMalloc((void**)&mat.val, max_length * sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&mat.act, batch_size * sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&mat.label, batch_size * sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&mat.err, max_length * sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&mat.row_ind, max_length * sizeof(int)));
  CUDA_CALL(cudaMalloc((void**)&mat.col_ind, max_length * sizeof(int)));
  return mat;
}

struct CooMatrixHost {
  float * val;
  int *row_ind;
  int *col_ind;
  int max_length;
  int nnz;
};

CooMatrixHost zeroCooMatrixHost(int batch_size, int max_length) {
  CooMatrixHost mat;
  mat.max_length = max_length;
  CUDA_CALL(cudaMallocHost((void**)&mat.val, sizeof(float)*max_length));
  CUDA_CALL(cudaMallocHost((void**)&mat.row_ind, sizeof(int)*max_length));
  CUDA_CALL(cudaMallocHost((void**)&mat.col_ind, sizeof(int)*max_length));
  return mat;
}

void vec2coo(const vector< vector< pair<int, float> > > & data, CooMatrixHost * mat_host, CooMatrix * mat) {
  int nnz = total_count(data);
  if(nnz > mat->max_length) cout << nnz << "\t" << mat->max_length << endl;
  mat->nnz = nnz;
  mat->nrow = data.size();
  CUDA_CALL(cudaMemset(mat->err, 0, mat->max_length * sizeof(float)));

  int n = 0;
  for(int i = 0; i < data.size(); i++){
    for(vector< pair<int, float> >::const_iterator j = data[i].begin(); 
	j != data[i].end(); j++) {
      mat_host->val[n] = j->second;
      mat_host->row_ind[n] = i;
      mat_host->col_ind[n] = j->first;
      ++n;
    }
  }
  
  CUDA_CALL(cudaMemcpyAsync(mat->val, mat_host->val, nnz*sizeof(float),
			    cudaMemcpyHostToDevice, stream));
  CUDA_CALL(cudaMemcpyAsync(mat->row_ind, mat_host->row_ind, nnz*sizeof(int),
			    cudaMemcpyHostToDevice, stream));
  CUDA_CALL(cudaMemcpyAsync(mat->col_ind, mat_host->col_ind, nnz*sizeof(int),
			    cudaMemcpyHostToDevice, stream));
}

void lr(const vector< vector< pair<int, float> > > & data, 
	const vector<float> & label,
	CooMatrixHost * coo_mat_host, 
	CooMatrix * coo_mat,
	float * w, int ncol, int batch) {
  vec2coo(data, coo_mat_host, coo_mat);
  CUDA_CALL(cudaMemcpyAsync(coo_mat->label, label.data(), sizeof(float) * label.size(), cudaMemcpyHostToDevice, stream));
  CUDA_CALL(cudaMemset(coo_mat->act, 0, sizeof(float) * data.size()));

  int shared_memory_usage = 1;
  int num_blocks = ((coo_mat->nnz + (NUM_THREADS - 1)) / NUM_THREADS);
  dot<<<num_blocks, NUM_THREADS, shared_memory_usage, stream>>>(coo_mat->val,
								coo_mat->row_ind,
								coo_mat->col_ind,
								coo_mat->nnz, 
								coo_mat->act, w);
  
  num_blocks = ((data.size() + (NUM_THREADS - 1)) / NUM_THREADS);
  vec_sigmoid<<<num_blocks, NUM_THREADS, shared_memory_usage, stream>>>(coo_mat->act, data.size());
  
  num_blocks = ((coo_mat->nnz + (NUM_THREADS - 1)) / NUM_THREADS);
  grad<<<num_blocks, NUM_THREADS, shared_memory_usage, stream>>>(coo_mat->val,
								 coo_mat->row_ind,
								 coo_mat->col_ind,
								 coo_mat->err,
								 coo_mat->nnz, 
								 coo_mat->act,
								 coo_mat->label, 
								 w, 0.01);
  if (batch % 10000 == 0){
    float * err = (float*) malloc(sizeof(float) * coo_mat->nnz);
    CUDA_CALL(cudaMemcpyAsync(err, coo_mat->err, sizeof(float) * coo_mat->nnz, cudaMemcpyDeviceToHost, stream));
    float total = 0.;
    for(int i = 0; i < coo_mat->nnz; i++) total += err[i];
    cout << total / (float) coo_mat->nnz << endl;
  }
}

void mock_sample(const int max_feature_id, vector< pair<int, float> > & out, int * label) {
  int count = rand() % 100 + 10;
  int ret = 0;
  for(int i = 0; i < count; i++) {
    int fid = rand() % max_feature_id;
    if(fid % 2 == 0) ret += 1;
    else ret -= 1;
    if(abs(ret) > 10) break;
    out.push_back(make_pair<int, float>(fid, 1.0));
  }
  *label = (ret > 0) ? 1 : 0;
}

#define MODEL_SIZE 1000000

__global__ void fill(float * w, float val, int size) {
  const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < size) w[tid] = val;
}


int main(int argc, char ** argv) {
  srand(time(NULL));
  CUDA_CALL(cudaSetDevice(1));
  CUDA_CALL(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));

  float * w;
  CUDA_CALL(cudaMalloc((void**)&w, sizeof(float) * MODEL_SIZE));
  CUDA_CALL(cudaMemset(w, 0, sizeof(float) * MODEL_SIZE));
  const int shared_memory_usage = 0;
  const int num_blocks = ((MODEL_SIZE + (NUM_THREADS - 1)) / NUM_THREADS);
  fill<<<num_blocks, 
    NUM_THREADS, 
    shared_memory_usage,
    stream>>>(w, 1, MODEL_SIZE);
  
  curandGenerator_t rand_gen;
  const curandRngType_t gen_type = CURAND_RNG_PSEUDO_DEFAULT;

  CURAND_CALL(curandCreateGenerator(&rand_gen, gen_type));
  CURAND_CALL(curandSetStream(rand_gen, stream));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rand_gen, time(NULL)));
  CURAND_CALL(curandGenerateNormal(rand_gen, w, MODEL_SIZE, 0, 0.1));
  
  int batch_size = atoi(argv[1]);
  int total_batch = 1024 * 1024 / batch_size;
  CooMatrix mat = zeroCooMatrix(batch_size, batch_size * 256);
  CooMatrixHost mat_host = zeroCooMatrixHost(batch_size, batch_size * 256);
  for(int batch = 0; batch < total_batch; batch++){
    vector< vector< pair<int, float> > > samples;
    vector<float> labels;
    for(int i = 0; i < batch_size; i++){
      vector< pair<int, float> > sample;
      int label;
      mock_sample(MODEL_SIZE, sample, &label);
      samples.push_back(sample);
      labels.push_back((float)label);
    }
    lr(samples, labels, &mat_host, &mat, w, MODEL_SIZE, batch);
  }
  CUDA_CALL(cudaStreamDestroy(stream));
}
