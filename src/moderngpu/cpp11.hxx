// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "tuple.hxx"

BEGIN_MGPU_NAMESPACE

///////////////////////
// tuple_iterator_value

template<typename tpl_t>
struct tuple_iterator_value;

template<typename... args_t>
struct tuple_iterator_value<tuple<args_t...> > {
  typedef tuple<typename std::iterator_traits<args_t>::value_type...> type;
};

template<typename tpl_t>
using tuple_iterator_value_t = typename tuple_iterator_value<tpl_t>::type;

////////////////////////////////////
// load and store to pointer tuples.

namespace detail {

template<typename int_t, typename... pointers_t, size_t... seq_i>
MGPU_HOST_DEVICE auto _lvalue_dereference(tuple<pointers_t...> pointers, 
  index_sequence<seq_i...> seq, int_t index) ->
  decltype(forward_as_tuple(get<seq_i>(pointers)[0]...)) {

  return forward_as_tuple(get<seq_i>(pointers)[index]...);
}

}

// Returns lvalues for each of the dereferenced pointers in the tuple.
template<typename int_t, typename... pointers_t>
MGPU_HOST_DEVICE auto dereference(tuple<pointers_t...> pointers, 
  int_t index) -> decltype(detail::_lvalue_dereference(pointers, 
    make_index_sequence<sizeof...(pointers_t)>(), index)) {

  return detail::_lvalue_dereference(pointers, 
    make_index_sequence<sizeof...(pointers_t)>(), index);
}

template<typename int_t, typename... pointers_t>
MGPU_HOST_DEVICE void store(tuple<pointers_t...> pointers, 
  tuple_iterator_value_t<tuple<pointers_t...> > values, 
  int_t index) {

  dereference(pointers, index) = values;
}

template<typename int_t, typename... pointers_t>
tuple_iterator_value_t<tuple<pointers_t...> > 
MGPU_HOST_DEVICE load(tuple<pointers_t...> pointers, int_t index) {
  typedef tuple_iterator_value_t<tuple<pointers_t...> > value_t;
  return value_t(dereference(pointers, index));
}


END_MGPU_NAMESPACE
