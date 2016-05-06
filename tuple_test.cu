#include <cstdio>
#include <moderngpu/cpp11.hxx>

typedef mgpu::tuple<int, int, int, int> test_t;

BEGIN_MGPU_NAMESPACE
/*
namespace detail {

template<int nt, typename seq_t, typename tpl_t>
struct cached_segment_load_storage_t;

template<int nt, size_t... seq_i, typename... args_t>
struct cached_segment_load_storage_t<
  nt, index_sequence<seq_i...>, tuple<args_t...> > {

  typedef tuple<args_t...> pointers_t;
  typedef tuple_iterator_value_t<pointers_t> values_t;

  template<typename type_t> struct array { type_t x[nt]; };
  
  typedef tuple<array<typename std::tuple_element<seq_i, values_t>::type>...> 
    shared_tuple_t;

  char bytes[sizeof(shared_tuple_t)];

  MGPU_HOST_DEVICE void store_to_shared(values_t values, int shared_index) {
    shared_tuple_t& shared = *(shared_tuple_t*)bytes;
    swallow(get<seq_i>(shared).x[shared_index] = get<seq_i>(values)...);
  }

  template<typename... args2_t>
  MGPU_HOST_DEVICE void swallow(args2_t... args) { }
};


extern "C" __global__ void foo() {
  typedef mgpu::detail::cached_segment_load_storage_t<
    15, 
    mgpu::detail::tuple_make_index_sequence<3>, 
    mgpu::tuple<int*, double*, float*> 
  > outer;


  __shared__ union {
   // typename outer::shared_tuple_t inner;
  } yo;
}
}
*/

END_MGPU_NAMESPACE

typedef mgpu::tuple<int, int, int, int> tuple_t;

#ifdef __CUDACC__
extern "C" __global__ void gpu_copy(const tuple_t* input, tuple_t* output) {
  int tid = threadIdx.x;
  output[tid] = input[tid];
}
#endif

int main(int argc, char** argv) { 

  int i[] = { 0, 1, 2, 3, 4 };
  double d[] = { .1, 1.1, 2.1, 3.1, 4.1 };
  float f[] = { .2f, 1.2f, 2.2f, 3.2f, 4.2f };
  typedef mgpu::tuple<int*, double*, float*> pointers_t;
  pointers_t p(i, d, f);

  auto ref = mgpu::dereference(p, 3);
  mgpu::tuple<int, double, float> foo = ref;
  auto foo2 = mgpu::load(p, 3);
 // auto values = mgpu::load(p, 3);

/*
  typedef mgpu::detail::cached_segment_load_storage_t<
    15, 
    mgpu::detail::tuple_make_index_sequence<3>, 
    mgpu::tuple<int*, double*, float*> 
  > shared_t;

  */
  return 0;
}