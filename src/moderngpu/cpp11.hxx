// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "meta.hxx"

#ifndef __TUPLE_ANNOTATION
#define __TUPLE_ANNOTATION __host__ __device__
#define __TUPLE_NAMESPACE mgpu
#endif

#include "../../libs/tuple/include/tuple.hpp"

BEGIN_MGPU_NAMESPACE

using std::get;
using std::tuple_element;
using std::tuple_size;

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

template<typename... pointers_t, size_t... seq_i>
MGPU_HOST_DEVICE auto _lvalue_dereference(tuple<pointers_t...> pointers, 
  detail::tuple_index_sequence<seq_i...> seq, size_t index) ->
  decltype(forward_as_tuple(get<seq_i>(pointers)[0]...)) {

  return forward_as_tuple(get<seq_i>(pointers)[index]...);
}

}

// Returns lvalues for each of the dereferenced pointers in the tuple.
template<typename... pointers_t>
MGPU_HOST_DEVICE auto dereference(tuple<pointers_t...> pointers, 
  size_t index) -> decltype(detail::_lvalue_dereference(pointers, 
    detail::tuple_make_index_sequence<sizeof...(pointers_t)>(), index)) {

  return detail::_lvalue_dereference(pointers, 
   detail::tuple_make_index_sequence<sizeof...(pointers_t)>(), index);
}

template<typename... pointers_t>
MGPU_HOST_DEVICE void store(tuple<pointers_t...> pointers, 
  tuple_iterator_value_t<tuple<pointers_t...> > values, 
  size_t index) {

  dereference(pointers, index) = values;
}

template<typename... pointers_t>
tuple_iterator_value_t<tuple<pointers_t...> > 
MGPU_HOST_DEVICE load(tuple<pointers_t...> pointers, size_t index) {
  typedef tuple_iterator_value_t<tuple<pointers_t...> > value_t;
  return value_t(dereference(pointers, index));
}

/////////////////////////////////
// Size of the largest component in the tuple.

template<size_t... values>
struct var_max;

template<size_t value_, size_t... values_> 
struct var_max<value_, values_...> {
  constexpr static size_t value = max(value_, var_max<values_...>::value);
};

template<size_t value_>
struct var_max<value_> {
  constexpr static size_t value = value_;
};

template<> struct var_max<> {
  constexpr static size_t value = 0;
};

template<typename tpl_t>
struct tuple_union_size;

template<typename... args_t>
struct tuple_union_size<tuple<args_t...> > {
  constexpr static size_t value = var_max<sizeof(args_t)...>::value;
};

END_MGPU_NAMESPACE

