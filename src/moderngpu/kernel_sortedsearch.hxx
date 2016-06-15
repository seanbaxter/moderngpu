#pragma once
#include "cta_merge.hxx"
#include "search.hxx"

BEGIN_MGPU_NAMESPACE

template<bounds_t bounds = bounds_lower, typename launch_arg_t = empty_t,
  typename func_t, typename needles_it, typename haystack_it, typename comp_it,
  typename... args_t>
void transform_search(func_t f, needles_it needles, int num_needles, 
  haystack_it haystack, int num_haystack, comp_it comp, context_t& context,
  args_t... args) {

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 11>,
      arch_35_cta<128, 7>,
      arch_52_cta<128, 11>
    >
  >::type_t launch_t;

  typedef typename std::iterator_traits<needles_it>::value_type type_t;

  // Partition the needles and haystacks into tiles.
  mem_t<int> partitions = merge_path_partitions<bounds>(needles, num_needles,
    haystack, num_haystack, launch_t::nv(context), comp, context);
  const int* mp_data = partitions.data();

  auto k = [=]MGPU_DEVICE(int tid, int cta, args_t... args) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, vt0 = params_t::vt0 };
    enum { nv = nt * vt };
    
    __shared__ union {
      type_t keys[nt * vt + 1];
      int indices[nt * vt];
    } shared;

    // Load the range for this CTA and merge the values into register.
    int mp0 = mp_data[cta + 0];
    int mp1 = mp_data[cta + 1];
    merge_range_t range = compute_merge_range(num_needles, num_haystack, cta,
      nv, mp0, mp1);

    // Merge the values needles and haystack.
    merge_pair_t<type_t, vt> merge = cta_merge_from_mem<bounds, nt, vt>(
      needles, haystack, range, tid, comp, shared.keys);

    // Store the needle indices to shared memory.
    iterate<vt>([&](int i) {
      if(merge.indices[i] < range.a_count()) {
        int needle = merge.indices[i];
        int haystack = range.b_begin + vt * tid + i - needle;
        shared.indices[needle] = haystack;
      }
    });
    __syncthreads();

    // Load the indices in strided order.
    array_t<int, vt> indices = shared_to_reg_strided<nt, vt>(
      shared.indices, tid);

    // Invoke the user-supplied functor f.
    strided_iterate<nt, vt, vt0>([=](int i, int j) {
      f(range.a_begin + j, indices[i], args...);
    }, tid, range.a_count());
  };

  cta_transform<launch_t>(k, num_needles + num_haystack, context, args...);
}


template<bounds_t bounds, typename launch_arg_t = empty_t,
  typename needles_it, typename haystack_it, typename indices_it,
  typename comp_it>
void sorted_search(needles_it needles, int num_needles, haystack_it haystack,
  int num_haystack, indices_it indices, comp_it comp, context_t& context) {

  transform_search<bounds, launch_arg_t>(
    [=]MGPU_DEVICE(int needle, int haystack, int* indices) {
      indices[needle] = haystack;
    }, needles, num_needles, haystack, num_haystack, comp, context, indices
  );
}


END_MGPU_NAMESPACE

