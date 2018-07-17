#include <iostream>
#include <torch/torch.h>
#include <vector>

using namespace at; // using ATen https://github.com/pytorch/pytorch/tree/master/aten
using namespace std;

bool DEBUG = true;

signed long OUT_OF_BOUNDS = -1;

template<typename T, size_t N>
class GeometryTensorAccessor : public TensorAccessor<T,N> {
    public:
        string geometry;
        bool out_of_bounds=false;
        GeometryTensorAccessor(T * data_, const int64_t * sizes_, const int64_t * strides_, string geometry, bool out_of_bounds=false)
            : TensorAccessor<T,N>(data_,sizes_,strides_) {
                this->geometry = geometry;
                this->out_of_bounds = out_of_bounds;
            }

        GeometryTensorAccessor<T,N-1> operator[](int64_t i) {
            if (this->geometry=="flat"){
                if (this->out_of_bounds or ((i < 0) or (i >= this->sizes_[0])))
                    return GeometryTensorAccessor<T,N-1>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1, geometry, true);
                return GeometryTensorAccessor<T,N-1>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1, geometry, false);
            } else if (this->geometry=="toroidal"){
                unsigned int j = (i%this->sizes_[0] + (abs(i/this->sizes_[0]) + 1)*this->sizes_[0])%this->sizes_[0];
                return GeometryTensorAccessor<T,N-1>(this->data_ + this->strides_[0]*j,this->sizes_+1,this->strides_+1, geometry, false);
            }
        }

        ~GeometryTensorAccessor(){}
};

template<typename T>
class GeometryTensorAccessor<T,1> : public TensorAccessor<T,1> {
    public:
        string geometry;
        bool out_of_bounds=false;
        GeometryTensorAccessor(T * data_, const int64_t * sizes_, const   int64_t * strides_, string geometry, bool out_of_bounds)
        : TensorAccessor<T,1>(data_,sizes_,strides_) {
            this->geometry = geometry;
            this->out_of_bounds = out_of_bounds;
        }

        T & operator[](int64_t i) {
            if (this->geometry=="flat"){
                // cout << "|<" << this->data_[this->strides_[0]*i] << ">|" << typeid(this->data_[this->strides_[0]*i]).name() << "??";
                if (this->out_of_bounds or ((i < 0) or (i >= this->sizes_[0]))){
                    return OUT_OF_BOUNDS;
                }
                return this->data_[this->strides_[0]*i];
            } else if (this->geometry=="toroidal"){
                unsigned int j = (i%this->sizes_[0] + (abs(i/this->sizes_[0]) + 1)*this->sizes_[0])%this->sizes_[0];
                // cout << "|<" << this->data_[this->strides_[0]*j] << ">|" << typeid(this->data_[this->strides_[0]*j]).name();
                return this->data_[this->strides_[0]*j];
            }
        }

        ~GeometryTensorAccessor(){}
};

class GridEnv {

    public:
        GridEnv(){}
        ~GridEnv(){}

    protected:
        unsigned short batch_size;
        unsigned short grid_dim_x;
        unsigned short grid_dim_y;

        unsigned char is_gpu;
        unsigned char device_id;

        string grid_geometry;
        Tensor grid;
        GeometryTensorAccessor<long,3>  *grid_a;

        void create_grid(void);

        GeometryTensorAccessor<long,3> *accessor()  {
            return new GeometryTensorAccessor<long,3>(this->grid.data<long>(),this->grid.sizes().data(),this->grid.strides().data(), this->grid_geometry);
        }

};

void GridEnv::create_grid(void) {
    auto deviceFunc = CPU;
    if (this->is_gpu){
        throw runtime_error("CUDA not currently supported.");
        deviceFunc = CUDA;
    }

    auto dataType = kLong;
    if (sizeof(char*) == 4) throw runtime_error("x86 architecture not supported!"); //dataType = kInt;

    this->grid = deviceFunc(dataType).zeros({this->batch_size, this->grid_dim_x, this->grid_dim_y});
    this->grid_a = this->accessor();

    if (DEBUG) {
        //this->grid_a_type = ToroidalTensorAccessor<long, 3>;
        //(*static_cast<ToroidalTensorAccessor<long, 3>*>(this->grid_a))[0][2][2] = 5;
        //long b = (*static_cast<ToroidalTensorAccessor<long, 3>*>(this->grid_a))[0][0][0];
        long b = (*this->grid_a)[4][1][2];
        cout << "b: " << b << endl;
        if (DEBUG) cout << "INIT GRID";
    }
}


class PredatorPreyEnv : public GridEnv {

    public:
        PredatorPreyEnv(unsigned short, unsigned short, unsigned short, unsigned short, unsigned short, unsigned char, unsigned char, string grid_geometry);
        ~PredatorPreyEnv(){}

    private:
        unsigned short n_prey;
        unsigned short n_predators;

        void init_grid(void);
};

PredatorPreyEnv::PredatorPreyEnv(unsigned short batch_size,
                                 unsigned short grid_dim_x,
                                 unsigned short grid_dim_y,
                                 unsigned short n_prey,
                                 unsigned short n_predators,
                                 unsigned char is_gpu,
                                 unsigned char device_id,
                                 string grid_geometry){

    this->batch_size = batch_size;
    this->grid_dim_x = grid_dim_x;
    this->grid_dim_y = grid_dim_y;

    this->n_prey = n_prey;
    this->n_predators = n_predators;

    this->is_gpu = is_gpu;
    this->device_id = device_id;

    this->grid_geometry = grid_geometry;

    this->create_grid();
    this->init_grid();
}

void PredatorPreyEnv::init_grid(){

    //cout << this->grid_a[0][0][0];

}

// void PredatorPreyEnv::Test(void){
//    std::cout << "HELLO!";
//    throw runtime_error("hello edar");
// }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<PredatorPreyEnv> animal(m, "PredatorPreyEnv");
    animal
        .def(py::init<unsigned short,
                      unsigned short,
                      unsigned short,
                      unsigned short,
                      unsigned short,
                      unsigned char,
                      unsigned char,
                      string>());
}

        // .def("Test", &PredatorPreyEnv::Test);

/*// s'(z) = (1 - s(z)) * s(z)
at::Tensor d_sigmoid(at::Tensor z) {
  auto s = at::sigmoid(z);
  return (1 - s) * s;
}

// tanh'(z) = 1 - tanh^2(z)
at::Tensor d_tanh(at::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
at::Tensor d_elu(at::Tensor z, at::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}*/

/*std::vector<at::Tensor> lltm_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell) {
  auto X = at::cat({old_h, input}, 1);

  auto gate_weights = at::addmm(bias, X, weights.transpose(0, 1));
  auto gates = gate_weights.chunk(3, 1);

  auto input_gate = at::sigmoid(gates[0]);
  auto output_gate = at::sigmoid(gates[1]);
  auto candidate_cell = at::elu(gates[2], 1.0);

  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = at::tanh(new_cell) * output_gate;

  return {new_h,
          new_cell,
          input_gate,
          output_gate,
          candidate_cell,
          X,
          gate_weights};
}

std::vector<at::Tensor> lltm_backward(
    at::Tensor grad_h,
    at::Tensor grad_cell,
    at::Tensor new_cell,
    at::Tensor input_gate,
    at::Tensor output_gate,
    at::Tensor candidate_cell,
    at::Tensor X,
    at::Tensor gate_weights,
    at::Tensor weights) {
  auto d_output_gate = at::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, 1);
  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
      at::cat({d_input_gate, d_output_gate, d_candidate_cell}, 1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(0, true);

  auto d_X = d_gates.mm(weights);
  const auto state_size = grad_h.size(1);
  auto d_old_h = d_X.slice(1, 0, state_size);
  auto d_input = d_X.slice(1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}*/
/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}*/