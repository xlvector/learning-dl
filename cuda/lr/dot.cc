#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <utility>
#include <cmath>
using namespace std;

#include <cuda_runtime.h>
#include <cusparse.h>

double cpu_dot(const vector< pair<int,double> > & a, const vector<double> & b) {
  double ret = 0.0;
  for(vector< pair<int,double> >::const_iterator i = a.begin();
      i != a.end(); i++) {
    ret += i->second * b[i->first];
  }
  return ret;
}

vector<double> cpu_batch_dot(const vector< vector< pair<int, double> > > & data, const vector<double> & b) {
  vector<double> rets(data.size(), 0);
  for(int i = 0; i < data.size(); i++){
    rets[i] = cpu_dot(data[i], b);
  }
  return rets;
}

double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-1.0 * x));
}

double cpu_grad(const vector< pair<int, double> > & x, 
	      const double wtx,
	      const int label,
	      vector<double> & w,
	      const double learning_rate,
	      const double lambda) {
  double err = (double)label - sigmoid(wtx);
  for(vector< pair<int, double> >::const_iterator i = x.begin();
      i != x.end(); i++) {
    w[i->first] += learning_rate * (err - lambda * w[i->first]);
  }
  return abs(err);
}

double cpu_batch_grad(const vector< vector< pair<int, double> > > & data,
	      const vector< int > & label,
	      vector<double> & b,
	      const double learning_rate,
	      const double lambda) {
  vector<double> dot = cpu_batch_dot(data, b);
  double err = 0.;
  double total = 0.;
  for(int i = 0; i < data.size(); i++) {
    err += cpu_grad(data[i], dot[i], label[i], b, learning_rate, lambda);
    total += 1.;
  }
  return err / total;
}

void mock_sample(const int max_feature_id, vector< pair<int, double> > & out, int * label) {
  int count = rand() % 100 + 10;
  int ret = 0;
  for(int i = 0; i < count; i++) {
    int fid = rand() % max_feature_id;
    if(fid % 2 == 0) ret += 1;
    else ret -= 1;
    out.push_back(make_pair<int, double>(fid, 1.0));
  }
  *label = (ret > 0) ? 1 : 0;
}

void cpu_lr(const int max_feature_id, const int n_batch, const int batch_size) {
  double learning_rate = 0.01;
  double lambda = 0.01;
  vector<double> model(max_feature_id + 1, 0);
  for(int i = 0; i < model.size(); i++) {
    model[i] = 0.5 - (double)(rand() % 10000) / 10000.0;
  }
  for(int i = 0; i < n_batch; i++) {
    vector< vector< pair<int, double> > > batch_data;
    vector< int > batch_label;
    for(int j = 0; j < batch_size; j++) {
      vector< pair<int, double> > x;
      int l;
      mock_sample(max_feature_id, x, &l);
      batch_data.push_back(x);
      batch_label.push_back(l);
    }
    
    double err = cpu_batch_grad(batch_data, batch_label, model, 
				  learning_rate, lambda);
    
    if(i % 10000 == 0){
      cout << "iter " << i << "\t" << err << endl;
      for(int k = 0; k < 10; k++) {
	cout << model[k] << "\t";
      }
      cout << endl;
    }
  }
}

int main() {
  cpu_lr(1000000, 1000000, 50);
}
