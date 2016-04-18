#include <moderngpu/cta_reduce.hxx>
#include <moderngpu/memory.hxx>

BEGIN_MGPU_NAMESPACE

// Pass share to upsweep.
template<int digit_size, typename launch_arg_t = empty_t, 
  typename input_it, typename digit_f>
void radixsort_upsweep(input_it input, int count, digit_f digits_func,
  int* frequencies, context_t& context) {

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 8>,
      arch_35_cta<256, 8>,
      arch_52_cta<256, 12>
    >
  >::type_t launch_t;

  auto upsweep_k = [=]MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { 
      nt = params_t::nt, vt = params_t::vt, nv = nt * vt,
      num_warps = nt / warp_size,
      num_digits = 1<< digit_size, num_rows = div_up(num_digits, 4),
      num_reducers = div_up(num_rows, num_warps)
    };
    typedef warp_reduce_t<nt, int> reduce_t;

    __shared__ union {
      int counters[nt * num_rows];
      unsigned char counters_char[nt * num_digits];
      struct { 
        typename reduce_t::storage_t reduce;
        int frequencies[num_digits];
      };
    } shared;

    int warp = tid / warp_size;
    int lane = tid % warp_size;

    auto inc_counter = [&](int digit) {
      int counter = (~3 & digit) * nt + 4 * tid;
      int byte = 3 & digit;
#if __CUDA_ARCH__ < 500
      ++shared.counters_char[counter + byte];
#else
      atomicAdd((int*)(shared.counters_char + counter), 1<< (8 * byte));
#endif      
    };

    int2 unpacked_counts[num_reducers] = { int2() };
    auto unpack_counter = [&]() {
      iterate<num_reducers>([&](int i) {
        int row = num_warps * i + warp;
        if(row < num_rows) {
          iterate<num_warps>([&](int warp) {
            int index = nt * row + warp_size * warp + lane;
            int packed = shared.counters[index];
            
            // Zero-extend to convert from char to short.
            unpacked_counts[i].x += prmt(packed, 0, 0x4140);
            unpacked_counts[i].y += prmt(packed, 0, 0x4342);
            shared.counters[index] = 0;
          });
        }
      });
      __syncthreads();
    };

    // Loop over the input interval for this CTA.
    int counter_progress = 0;
    iterate<num_rows>([&](int row) {
      shared.counters[nt * row + tid] = 0;
    });
    __syncthreads();
     
    { 
      // Avoid an overflow by widening the counters into register.
      if(counter_progress + vt > 255)
        unpack_counter();
      
      // Stream through a portion of the inputs.
      range_t tile = get_tile(cta, nv, count);
      int digits[vt];

      strided_iterate_if_else<nt, vt>(
        [&](int i, int j) { digits[i] = digits_func(input[tile.begin + j]); },
        [&](int i, int j) { digits[i] = num_digits - 1; },
        tid, tile.count()
      );

      iterate<vt>([&](int i) {
        // Extract the digit and increment the counter.
        inc_counter(digits[i]);
      });

      // Reflect the advance in values.
      counter_progress += vt;
    }
    if(counter_progress > 0) unpack_counter();

    // Unpack the counters from shorts to ints and reduce across the CTA.
    iterate<num_reducers>([&](int i) {
      // Zero-extend to convert from short to int.
      int x[4] = { 
        0xffff & unpacked_counts[i].x,
        (int)((uint)unpacked_counts[i].x>> 16),
        0xffff & unpacked_counts[i].y,
        (int)((uint)unpacked_counts[i].y>> 16)
      };

      // Warp reduce these four counts.
      int row = num_warps * i + warp;
      iterate<4>([&](int j) {
        int freq = reduce_t().reduce(tid, x[j], shared.reduce);

        // Store the results to shared memory.
        int digit = 4 * row + j;
        if(digit < num_digits && !lane) shared.frequencies[digit] = freq;
      });
    });
    __syncthreads();

    // Copy the frequencies to the output.
    if(tid < num_digits) frequencies[tid] = shared.frequencies[tid];
  };

  cta_launch<launch_t>(upsweep_k, 100, context);
}

void radixsort(uint32_t* keys, int count, context_t& context) {
  int bit = 1;
  radixsort_upsweep<5>(keys, count, 
    [=]MGPU_DEVICE(int arg) { return 31 & (arg>> bit); },
    (int*)nullptr, context);
}

END_MGPU_NAMESPACE

int main(int argc, char** argv) {

  return 0;
}