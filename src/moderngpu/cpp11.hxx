// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "meta.hxx"

BEGIN_MGPU_NAMESPACE

template<typename type_t>
MGPU_HOST_DEVICE type_t&& 
forward(typename std::remove_reference<type_t>::type& a) noexcept {
  return static_cast<type_t&&>(a);
} 

template<typename type_t>
MGPU_HOST_DEVICE typename std::remove_reference<type_t>::type&&
move(type_t&& a) noexcept {
  typedef typename std::remove_reference<type_t>::type&& rval_t;
  return static_cast<rval_t>(a);
}

/////////////////
// index_sequence

// Somewhat improved linear index_sequence from
// http://talesofcpp.fusionfenix.com/post-22/true-story-efficient-packing
template<size_t... int_s>
struct index_sequence { 
  enum { size = sizeof...(int_s) };
};

namespace detail {
template<typename seq_t>
struct _next;

template<size_t... seq_i>
struct _next<index_sequence<seq_i...> > {
  // grow the sequence by one element.
  typedef index_sequence<seq_i..., sizeof...(seq_i)> type;
};

template<size_t count>
struct _make_index_sequence : 
  _next<typename _make_index_sequence<count - 1>::type> { };

template<> struct _make_index_sequence<0> {
  typedef index_sequence<> type;
};
} // namespace detail

template<size_t count>
using make_index_sequence = 
  typename detail::_make_index_sequence<count>::type;

//////////////////////////////
// forward declare the tuples.

namespace detail {

template<size_t i, typename arg_t>
struct _tuple_leaf; 

template<typename seq_t, typename... args_t>
struct _tuple_impl;

} // namespace detail

template<typename... args_t>
struct tuple;

/////////////
// tuple_size


template<typename tpl_t>
struct tuple_size;

template<typename... args_t>
struct tuple_size<tuple<args_t...> > {
  enum { value = sizeof...(args_t) };
};

////////////////
// tuple_element

namespace detail {

template<size_t i, typename type_t>
struct _indexed { typedef type_t type; };

template<typename seq_t, typename... args_t>
struct _indexer;

// Multiple inheritance from similar base classes as tuple.
template<size_t... seq_i, typename... args_t>
struct _indexer<index_sequence<seq_i...>, args_t...> :
  _indexed<seq_i, args_t>... { };

template<size_t i, typename... args_t>
struct _at_index {
  template<typename arg_t>
  static _indexed<i, arg_t> _select(_indexed<i, arg_t>);

  typedef _indexer<make_index_sequence<sizeof...(args_t)>, args_t...> _impl;
  typedef typename decltype(_select(_impl{}))::type type;
};

} // namespace detail

template<size_t i, typename tpl_t>
struct tuple_element;

template<size_t i, typename... args_t>
struct tuple_element<i, tuple<args_t...> > : detail::_at_index<i, args_t...> { };

template<size_t i, typename seq_t, typename... args_t>
struct tuple_element<i, detail::_tuple_impl<seq_t, args_t...> > : detail::_at_index<i, args_t...> { };

template<size_t i, typename tpl_t>
struct tuple_element<i, const tpl_t> {
  typedef typename tuple_element<i, tpl_t>::type value_t;
  typedef typename std::add_const<value_t>::type type;
};

template<size_t i, typename tpl_t>
struct tuple_element<i, volatile tpl_t> {
  typedef typename tuple_element<i, tpl_t>::type value_t;
  typedef typename std::add_volatile<value_t>::type type;
};

template<size_t i, typename tpl_t>
struct tuple_element<i, const volatile tpl_t> {
  typedef typename tuple_element<i, tpl_t>::type value_t;
  typedef typename std::add_cv<value_t>::type type;
};

template<size_t i, typename tpl_t>
using tuple_element_t = typename tuple_element<i, tpl_t>::type;

//////
// get

namespace detail {

template<size_t i, typename arg_t>
MGPU_HOST_DEVICE arg_t& 
_get(_tuple_leaf<i, arg_t>& tpl) {
  return tpl.x;
}

template<size_t i, typename arg_t>
MGPU_HOST_DEVICE const arg_t& 
_get(const _tuple_leaf<i, arg_t>& tpl) {
  return tpl.x;
}

} // detail

template<size_t i, typename... args_t>
MGPU_HOST_DEVICE tuple_element_t<i, tuple<args_t...> >&
get(tuple<args_t...>& tpl) {
  return detail::_get<i>(tpl);
}

template<size_t i, typename... args_t>
MGPU_HOST_DEVICE const tuple_element_t<i, tuple<args_t...> >&
get(const tuple<args_t...>& tpl) {
  return detail::_get<i>(tpl);
}

template<size_t i, typename... args_t>
MGPU_HOST_DEVICE tuple_element_t<i, tuple<args_t...> >&&
get(tuple<args_t...>&& tpl) {
  return forward<tuple_element_t<i, tuple<args_t...> > >(tpl);
}

////////
// tuple

namespace detail {

template<size_t i, typename arg_t>
struct _tuple_leaf { 
  arg_t x; 

  _tuple_leaf() = default;
  _tuple_leaf(const _tuple_leaf&) = default;
  _tuple_leaf(_tuple_leaf&&) = default;
  _tuple_leaf& operator=(const _tuple_leaf& rhs) = default;
  _tuple_leaf& operator=(_tuple_leaf&&) = default;

  template<typename arg2_t>
  MGPU_HOST_DEVICE _tuple_leaf(const _tuple_leaf<i, arg2_t>& rhs) : 
    x(rhs.x) { }

  template<typename arg2_t>
  MGPU_HOST_DEVICE _tuple_leaf(_tuple_leaf<i, arg2_t>&& rhs) : x(rhs.x) { }

  template<typename arg2_t>
  MGPU_HOST_DEVICE _tuple_leaf(const arg2_t& y) : x(y) { }

  template<typename arg2_t>
  MGPU_HOST_DEVICE _tuple_leaf(arg2_t&& y) : x(forward<arg2_t>(y)) { }
};

template<size_t... seq_i, typename... args_t>
struct _tuple_impl<index_sequence<seq_i...>, args_t...> :
  _tuple_leaf<seq_i, args_t>... {

  typedef index_sequence<seq_i...> seq_t;

  _tuple_impl() = default;
  _tuple_impl(const _tuple_impl&) = default;
  _tuple_impl(_tuple_impl&&) = default;
  _tuple_impl& operator=(const _tuple_impl& rhs) = default;
  _tuple_impl& operator=(_tuple_impl&&) = default;

  // Construct or assign from tuples.
  template<typename... args2_t>
  MGPU_HOST_DEVICE _tuple_impl(const _tuple_impl<seq_t, args2_t...>& tpl) :
    _tuple_leaf<seq_i, args_t>(_tuple_leaf<seq_i, args2_t>(tpl))... { }

  template<typename... args2_t>
  MGPU_HOST_DEVICE _tuple_impl(_tuple_impl<seq_t, args2_t...>&& tpl) :
    _tuple_leaf<seq_i, args_t>(
      forward<_tuple_leaf<seq_i, args2_t> >(tpl))... { }

  template<typename... args2_t>
  MGPU_HOST_DEVICE _tuple_impl& operator=(const _tuple_impl<args2_t...>& tpl) {
    int f[] = {
      (_get<seq_i>(*this) = _get<seq_i>(tpl), 0)...
    };
    return *this;
  }

  template<typename... args2_t>
  MGPU_HOST_DEVICE _tuple_impl& operator=(_tuple_impl<args2_t...>&& tpl) {
    int f[] = {
      (_get<seq_i>(*this) = 
        forward<tuple_element_t<seq_i, _tuple_impl<args2_t...> > >(
          _get<seq_i>(tpl)
        ), 
      0)...
    };
    return *this;
  }

  // Construct from arguments.
  MGPU_HOST_DEVICE explicit _tuple_impl(const args_t&... args) :
    _tuple_leaf<seq_i, args_t>(args)... { }

  template<typename... args2_t,
    typename = typename std::enable_if<
      sizeof...(args2_t) == sizeof...(args_t)>::type>
  MGPU_HOST_DEVICE explicit _tuple_impl(args2_t&&... args) :
    _tuple_leaf<seq_i, args2_t>(forward<args2_t>(args))... { }
};

} // namespace detail

template<typename... args_t>
struct tuple : detail::_tuple_impl<
  make_index_sequence<sizeof...(args_t)>,
  args_t...
> {
  typedef make_index_sequence<sizeof...(args_t)> seq_t;
  typedef detail::_tuple_impl<seq_t, args_t...> impl_t;

  tuple() = default;
  tuple(const tuple&) = default;
  tuple(tuple&&) = default;
  tuple& operator=(const tuple& rhs) = default;
  tuple& operator=(tuple&&) = default;

  // Construct or assign from a tuple.
  template<typename... args2_t>
  MGPU_HOST_DEVICE tuple(const tuple<args2_t...>& tpl) : impl_t(tpl) { }

  template<typename... args2_t>
  MGPU_HOST_DEVICE tuple(tuple<args2_t...>&& tpl) : 
    impl_t(forward<tuple<args2_t...> >(tpl)) { }

  template<typename... args2_t>
  MGPU_HOST_DEVICE tuple& operator=(const tuple<args2_t...>& tpl) {
    static_cast<impl_t&>(*this) = tpl;
    return *this;
  }

  template<typename... args2_t>
  MGPU_HOST_DEVICE tuple& operator=(tuple<args2_t...>&& tpl) {
    static_cast<impl_t&>(*this) = move(tpl);
    return *this;
  }

  // Construct from arguments.

  // const& args ctor.
  MGPU_HOST_DEVICE explicit tuple(const args_t&... args) : impl_t(args...) { }

  // && args ctor.
  template<typename... args2_t,
    typename = typename std::enable_if<
      sizeof...(args2_t) == sizeof...(args_t)>::type>
  MGPU_HOST_DEVICE explicit tuple(args_t&&... args) : 
    impl_t(forward<args_t>(args)...) { }
};

template<> struct tuple<> { };

///////
/// tie

template<typename... args_t>
MGPU_HOST_DEVICE tuple<args_t&...> tie(args_t&... args) { 
  return tuple<args_t&...>(args...);
}

///////////////////
// forward_as_tuple

template<typename... args_t>
MGPU_HOST_DEVICE tuple<args_t&&...> forward_as_tuple(args_t&&... args) {
  return tuple<args_t&&...>(forward<args_t>(args)...);
}

/////////////
// make_tuple

template<typename... args_t>
MGPU_HOST_DEVICE tuple<typename std::decay<args_t>::type...>
make_tuple(args_t&&... args) {
  return tuple<typename std::decay<args_t>::type...>(
    forward<args_t>(args)...
  );
}

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
MGPU_HOST_DEVICE auto _dereference(tuple<pointers_t...> pointers, 
  index_sequence<seq_i...> seq, size_t index) ->
  decltype(forward_as_tuple(get<seq_i>(pointers)[0]...)) {

  return forward_as_tuple(get<seq_i>(pointers)[index]...);
}
}

// Returns lvalues for each of the dereferenced pointers in the tuple.
template<typename... pointers_t>
MGPU_HOST_DEVICE auto dereference(tuple<pointers_t...> pointers, 
  size_t index) -> decltype(detail::_dereference(pointers, 
    make_index_sequence<sizeof...(pointers_t)>(), index)) {

  return detail::_dereference(pointers, 
    make_index_sequence<sizeof...(pointers_t)>(), index);
}

template<typename... pointers_t>
MGPU_HOST_DEVICE void store(tuple<pointers_t...> pointers, 
  tuple_iterator_value<tuple<pointers_t...> > values, 
  size_t index) {

  dereference(pointers, index) = values;
}

template<typename... pointers_t>
tuple_iterator_value<tuple<pointers_t...> > 
MGPU_HOST_DEVICE load(tuple<pointers_t...> pointers, size_t index) {
  typedef tuple_iterator_value<tuple<pointers_t...> > value_t;
  return value_t(dereference(pointers, index));
}


/////////////////////////////
// Tuple comparison operators

namespace detail {
template<size_t i, size_t count>
struct _tuple_compare {
  template<typename tpl_t>
  MGPU_HOST_DEVICE static bool eq(const tpl_t a, const tpl_t b) {
    return get<i>(a) == get<i>(b) && _tuple_compare<i + 1, count>::eq(a, b);
  }

  template<typename tpl_t>
  MGPU_HOST_DEVICE static bool less(const tpl_t a, const tpl_t b) {
    return get<i>(a) < get<i>(b) || 
      (!(get<i>(b) < get<i>(a)) && _tuple_compare<i + 1, count>::less(a, b));
  }
};

template<size_t count>
struct _tuple_compare<count, count> {
  template<typename tpl_t>
  MGPU_HOST_DEVICE static bool eq(const tpl_t, const tpl_t) {
    return true;
  }

  template<typename tpl_t>
  MGPU_HOST_DEVICE static bool less(const tpl_t, const tpl_t) {
    return false;
  }
};

} // namespace detail

template<typename... args_t>
MGPU_HOST_DEVICE bool operator<(tuple<args_t...> a, tuple<args_t...> b) {
  return detail::_tuple_compare<0, sizeof...(args_t)>::less(a, b);
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator<=(tuple<args_t...> a, tuple<args_t...> b) {
  return !(b < a);
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator>(tuple<args_t...> a, tuple<args_t...> b) {
  return b < a;
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator>=(tuple<args_t...> a, tuple<args_t...> b) {
  return !(a < b);
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator==(tuple<args_t...> a, tuple<args_t...> b) {
  return detail::_tuple_compare<0, sizeof...(args_t)>::eq(a, b);
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator!=(tuple<args_t...> a, tuple<args_t...> b) {
  return !(a == b);
}

END_MGPU_NAMESPACE

