#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <utility>
#include <cmath>
using namespace std;

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <curand_kernel.h>

#define CUDA_CALL(func)							\
  {									\
    cudaError_t e = (func);						\
    if(e != cudaSuccess)						\
      cout << "CUDA: " << cudaGetErrorString(e) << endl;		\
  }

#define CUSP_CALL(func)							\
  {									\
    cusparseStatus_t e = (func);					\
    if(e != CUSPARSE_STATUS_SUCCESS)					\
      cout << "CUSP: " << e << endl;					\
  }

#define CURAND_CALL(func)			\
  {						\
    curandStatus_t e = (func);			\
    if(e != CURAND_STATUS_SUCCESS)		\
      cout << "CURAND: " << e << endl;		\
  }

cudaStream_t stream;
cusparseHandle_t sparse_handle;

float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-1.0 * x));
}

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
  float *val;
  int *row_ind, *col_ind;
  int nnz, nrow;
};

CooMatrix vec2coo(const vector< vector< pair<int, float> > > & data) {
  int nnz = total_count(data);
  CooMatrix mat;
  mat.nnz = nnz;
  mat.nrow = data.size();
  CUDA_CALL(cudaMalloc((void**)&mat.val, nnz * sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&mat.row_ind, nnz * sizeof(int)));
  CUDA_CALL(cudaMalloc((void**)&mat.col_ind, nnz * sizeof(int)));

  vector<float> val(nnz, 0);
  vector<int> row_ind(nnz, 0), col_ind(nnz, 0);
  int n = 0;
  for(int i = 0; i < data.size(); i++){
    for(vector< pair<int, float> >::const_iterator j = data[i].begin(); 
	j != data[i].end(); j++) {
      val[n] = j->second;
      row_ind[n] = i;
      col_ind[n] = j->first;
      ++n;
    }
  }
  
  CUDA_CALL(cudaMemcpyAsync(mat.val, val.data(), nnz*sizeof(float),
			    cudaMemcpyHostToDevice, stream));
  CUDA_CALL(cudaMemcpyAsync(mat.row_ind, row_ind.data(), nnz*sizeof(int),
			    cudaMemcpyHostToDevice, stream));
  CUDA_CALL(cudaMemcpyAsync(mat.col_ind, col_ind.data(), nnz*sizeof(int),
			    cudaMemcpyHostToDevice, stream));

  return mat;
}

struct CsrMatrix {
  float *val;
  int *row_ptr, *col_ind;
  int nnz, nrow;
};

CsrMatrix coo2csr(CooMatrix coo_mat) {
  CsrMatrix csr_mat;
  csr_mat.nnz = coo_mat.nnz;
  csr_mat.val = coo_mat.val;
  csr_mat.col_ind = coo_mat.col_ind;
  csr_mat.nrow = coo_mat.nrow;
  int byte_size = (coo_mat.nrow + 1) * sizeof(int);
  CUDA_CALL(cudaMalloc((void**)&csr_mat.row_ptr, byte_size));
  CUSP_CALL(cusparseXcoo2csr(sparse_handle, coo_mat.row_ind, coo_mat.nnz, 
			     coo_mat.nrow, csr_mat.row_ptr, CUSPARSE_INDEX_BASE_ZERO));
  return csr_mat;
}

float grad(const vector< vector< pair<int, float> > > & data, 
		 const vector<float> & label,
		 float * b, int ncol) {
  CooMatrix coo_mat = vec2coo(data);
  CsrMatrix csr_mat = coo2csr(coo_mat);

  float alpha = 1.0;
  float beta = 0.0;
  float *y;

  cusparseMatDescr_t descr = 0;
  CUSP_CALL(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
  
  CUDA_CALL(cudaMalloc((void**)&y, sizeof(float)*data.size()));
  CUSP_CALL(cusparseScsrmv(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			   csr_mat.nrow, ncol, csr_mat.nnz, &alpha, 
			   descr, csr_mat.val, 
			   csr_mat.row_ptr, csr_mat.col_ind, b, &beta, y));
  float *pred = (float *)malloc(data.size() * sizeof(float));
  CUDA_CALL(cudaMemcpyAsync(pred, y, data.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
  
  float total_err = 0.;
  for(int i = 0; i < data.size(); i++) {
    float err = label[i] - sigmoid(pred[i]);
    total_err += abs(err);
    vector<int> xind;
    vector<float> xval;  
    for(vector< pair<int, float> >::const_iterator j = data[i].begin();
	j != data[i].end(); j++) {
      xind.push_back(j->first);
      xval.push_back(0.01 * err);
    }
    int * gpu_xind = copyIntToGPU(xind);
    float * gpu_xval = copyFloatToGPU(xval);
    float a = 1;
    CUSP_CALL(cusparseSaxpyi(sparse_handle, xind.size(), &a, 
			     gpu_xval, gpu_xind, b, CUSPARSE_INDEX_BASE_ZERO));
    CUDA_CALL(cudaFree(gpu_xind));
    CUDA_CALL(cudaFree(gpu_xval));
  }
  free(pred);
  return total_err / (float)data.size();
}

void mock_sample(const int max_feature_id, vector< pair<int, float> > & out, int * label) {
  int count = rand() % 100 + 100;
  int ret = 0;
  for(int i = 0; i < count; i++) {
    int fid = rand() % max_feature_id;
    if(fid % 2 == 0) ret += 1;
    else ret -= 1;
    out.push_back(make_pair<int, float>(fid, 1.0));
  }
  *label = (ret > 0) ? 1 : 0;
}

#define MODEL_SIZE 1000000

__global__ void fill(float * w, float val, int size) {
  const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < size) w[tid] = val;
}


int main() {
  srand(time(NULL));
  CUDA_CALL(cudaSetDevice(1));
  CUDA_CALL(cudaStreamCreate(&stream));
  CUSP_CALL(cusparseCreate(&sparse_handle));
  CUSP_CALL(cusparseSetStream(sparse_handle, stream));

  float * w;
  CUDA_CALL(cudaMalloc((void**)&w, sizeof(float) * MODEL_SIZE));
  CUDA_CALL(cudaMemset(w, 0, sizeof(float) * MODEL_SIZE));
  const int shared_memory_usage = 0;
  const int num_threads = 256;
  const int num_blocks = ((MODEL_SIZE + (num_threads - 1)) / num_threads);
  fill<<<num_blocks, 
    num_threads, 
    shared_memory_usage,
    stream>>>(w, 1, MODEL_SIZE);
  
  curandGenerator_t rand_gen;
  const curandRngType_t gen_type = CURAND_RNG_PSEUDO_DEFAULT;

  CURAND_CALL(curandCreateGenerator(&rand_gen, gen_type));
  CURAND_CALL(curandSetStream(rand_gen, stream));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rand_gen, time(NULL)));
  CURAND_CALL(curandGenerateNormal(rand_gen, w, MODEL_SIZE, 0, 0.1));
  
  for(int batch = 0; batch < 100000; batch++){
    vector< vector< pair<int, float> > > samples;
    vector<float> labels;
    for(int i = 0; i < 50; i++){
      vector< pair<int, float> > sample;
      int label;
      mock_sample(MODEL_SIZE, sample, &label);
      samples.push_back(sample);
      labels.push_back((float)label);
    }
    
    float err = grad(samples, labels, w, MODEL_SIZE);
    if(batch % 1000 == 0) cout << err << endl;
  }
}