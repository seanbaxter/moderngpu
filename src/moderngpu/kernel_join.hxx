// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "kernel_sortedsearch.hxx"
#include "kernel_scan.hxx"
#include "kernel_load_balance.hxx"

BEGIN_MGPU_NAMESPACE

template<typename launch_arg_t = empty_t, 
  typename a_it, typename b_it, typename comp_t>
mem_t<int2> inner_join(a_it a, int a_count, b_it b, int b_count, 
  comp_t comp, context_t& context) {

  // Compute lower bounds of a into b.
  mem_t<int> lower(a_count, context);
  sorted_search<bounds_lower, launch_arg_t>(a, a_count, b, b_count, 
    lower.data(), comp, context);

  // Compute upper bounds of a into b. Emit the difference upper-lower, which
  // is the number of matches for each element of a into b.
  mem_t<int> segments(a_count, context);
  transform_search<bounds_upper, launch_arg_t>(
    []MGPU_DEVICE(int needle, int haystack, const int* lower, int* matches) {
      matches[needle] = haystack - lower[needle];
    }, a, a_count, b, b_count, comp, context, lower.data(), segments.data()
  );
  
  // Scan the matches into a segments descriptor array.
  mem_t<int> join_count(1, context, memory_space_host);
  scan_event(segments.data(), a_count, segments.data(), plus_t<int>(), 
    join_count.data(), context, context.event());

  // Allocate space for the join.
  cudaEventSynchronize(context.event());
  int count = join_count.data()[0];
  mem_t<int2> output(count, context);

  // Use load-balancing search on the segments. The output is a pair with
  // a_index = seg and b_index = lower[seg] + rank.
  transform_lbs<launch_arg_t>(
    []MGPU_DEVICE(
      int index, int seg, int rank, 
      tuple<int> lower, int2* output
    ) {
      output[index] = make_int2(seg, get<0>(lower) + rank);
    }, count, segments.data(), a_count, 
    make_tuple(lower.data()), context, output.data()
  );

  return output;
}

END_MGPU_NAMESPACE
