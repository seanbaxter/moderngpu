#include <cstdio>
#include <type_traits>
#include <utility>


template<typename type_t>
using decay_t = typename std::decay<type_t>::type;

//////////
// var_and

template<bool... args_b>
struct var_and;

template<bool arg_a, bool... args_b> 
struct var_and<arg_a, args_b...> {
  enum { value = arg_a && var_and<args_b...>::value };
};
template<bool arg_a>
struct var_and<arg_a> {
  enum { value = arg_a };
};
template<>
struct var_and<> {
  enum { value = true };
};

//////////
// var_or

template<bool... args_b>
struct var_or;

template<bool arg_a, bool... args_b> 
struct var_or<arg_a, args_b...> {
  enum { value = arg_a || var_or<args_b...>::value };
};
template<bool arg_a>
struct var_or<arg_a> {
  enum { value = arg_a };
};
template<>
struct var_or<> {
  enum { value = false };
};



// Forward declare the tuple.
template<typename... args_t>
struct tuple;

template<size_t i, typename tpl_t> 
struct tuple_element;

template<size_t i, typename arg_t, typename... args_t>
struct tuple_element<i, tuple<arg_t, args_t...> > : 
  tuple_element<i - 1, tuple<args_t...> > { };

template<typename arg_t, typename... args_t>
struct tuple_element<0, tuple<arg_t, args_t...> > {
  typedef arg_t type;
};

template<size_t i, typename tpl_t>
using tuple_element_t = typename tuple_element<i, tpl_t>::type;

namespace detail {

template<size_t i, typename arg_t, bool is_empty = std::is_empty<arg_t>::value>
struct tuple_leaf {
  arg_t x;

   arg_t& get() { return x; }
   const arg_t& get() const { return x; }

  tuple_leaf() = default;
  tuple_leaf(const tuple_leaf&) = default;

  template<typename arg2_t,
    typename = typename std::enable_if<
      std::is_constructible<arg_t, arg2_t&&>::value
    >::type
  >  
  tuple_leaf(arg2_t&& arg) : x(std::forward<arg2_t>(arg)) { }

  template<typename arg2_t,
    typename = typename std::enable_if<
      std::is_constructible<arg_t, const arg2_t&>::value
    >::type
  >  
  tuple_leaf(const arg2_t& arg) : x(arg) { }
};

template<size_t i, typename arg_t>
struct tuple_leaf<i, arg_t, true> : arg_t { 
  arg_t& get() { return *this; }
  const arg_t& get() const { return *this; }

  template<typename arg2_t,
    typename = typename std::enable_if<
      std::is_constructible<arg_t, const arg2_t&>::value
    >::type
  >  
  tuple_leaf(const arg2_t& arg) : arg_t(arg) { }
};

template<size_t i, typename... args_t>
struct tuple_impl;

template<size_t i>
struct tuple_impl<i> { };

template<size_t i, typename arg_t, typename... args_t>
struct tuple_impl<i, arg_t, args_t...> :
  tuple_leaf<i, arg_t>,
  tuple_impl<i + 1, args_t...> {

  typedef tuple_leaf<i, arg_t> head_t;
  typedef tuple_impl<i + 1, args_t...> tail_t;

   arg_t& head() { return head_t::get(); }
   const arg_t& head() const { return head_t::get(); }

   tail_t& tail() { return *this; }
   const tail_t& tail() const { return *this; }

  // Constructors.
  tuple_impl() = default;
  explicit tuple_impl(const tuple_impl&) = default;

  template<typename... args2_t>  
  explicit tuple_impl(const tuple_impl<i, args2_t...>& rhs) :
    head_t(rhs), tail_t(rhs) { }

  template<typename... args2_t>  
  explicit tuple_impl(tuple_impl<i, args2_t...>&& rhs) :
    head_t(std::move(rhs)), 
    tail_t(std::forward<tuple_impl<i, args2_t...> >(rhs)) { }

  template<typename arg2_t, typename... args2_t,
    typename = typename std::enable_if<
      sizeof...(args_t) == sizeof...(args2_t) &&
      std::is_constructible<arg_t, arg2_t&&>::value &&
      var_and<std::is_constructible<args_t, args2_t&&>::value...>::value
    >::type
  >  
  tuple_impl(arg2_t&& arg, args2_t&&... args) :
    head_t(std::forward<arg2_t>(arg)), 
    tail_t(std::forward<args2_t>(args)...) { }

  template<typename arg2_t, typename... args2_t,
    typename = typename std::enable_if<
      std::is_constructible<arg_t, const arg2_t&>::value &&
      var_and<std::is_constructible<args_t, const args2_t&>::value...>::value
    >::type
  >  
  tuple_impl(const arg2_t& arg, const args2_t&... args) :
    head_t(arg), tail_t(args...) { }

  // Assignment
};

template<size_t i, typename arg_t>  
tuple_leaf<i, arg_t>& get_leaf(tuple_leaf<i, arg_t>& leaf) {
  return leaf;
}

template<size_t i, typename arg_t>  
const tuple_leaf<i, arg_t>& get_leaf(const tuple_leaf<i, arg_t>& leaf) {
  return leaf;
}

} // namespace detail



template<typename... args_t>
struct tuple : detail::tuple_impl<0, args_t...> { 
  typedef detail::tuple_impl<0, args_t...> impl_t;

  tuple() = default;
  tuple(const tuple&) = default;

  template<typename... args2_t,
    typename = typename std::enable_if<
      sizeof...(args2_t) == sizeof...(args_t) &&
      var_and<std::is_constructible<args_t, const args2_t&>::value...>::value
    >::type
  > 
  explicit tuple(const tuple<args2_t...>& rhs) : impl_t(rhs) { }
  
//  template<typename... args2_t> 
//  tuple(tuple<args2_t...>&& rhs) : 
//    impl_t(std::forward<detail::tuple_impl<0, args2_t...> >(rhs)) { }

  template<typename... args2_t,
    typename = typename std::enable_if<
      sizeof...(args2_t) == sizeof...(args_t) &&
      var_and<std::is_constructible<args_t, args2_t&&>::value...>::value
    >::type
  >  
  tuple(args2_t&&... args) : impl_t(std::forward<args2_t>(args)...) { }

  template<typename... args2_t,
    typename = typename std::enable_if<
      sizeof...(args2_t) == sizeof...(args_t) &&
      var_and<std::is_constructible<args_t, const args2_t&>::value...>::value
    >::type
  >  
  tuple(const args2_t&... args) : impl_t(args...) { }

} __attribute((aligned));

namespace detail {

template<size_t i, typename arg_t>  
arg_t& _get(tuple_leaf<i, arg_t>& leaf) {
  return leaf.get();
}

template<size_t i, typename arg_t>  
const arg_t& _get(const tuple_leaf<i, arg_t>& leaf) {
  return leaf.const_get();
}

}

template<size_t i, typename... args_t>  
tuple_element_t<i, tuple<args_t...> >&
get(tuple<args_t...>& tpl) {
  return detail::_get<i>(tpl);
}

template<size_t i, typename... args_t>  
const tuple_element_t<i, tuple<args_t...> >&
get(const tuple<args_t...>& tpl) {
  return detail::_get<i>(tpl);
}

template<size_t i, typename... args_t>  
const tuple_element_t<i, tuple<args_t...> >&&
get(tuple<args_t...>&& tpl) {
  return std::forward<tuple_element_t<i, tuple<args_t...> > >(get<i>(tpl));
}

template<typename... args_t>  
tuple<decay_t<args_t>...> make_tuple(args_t&&... args) {
  return tuple<decay_t<args_t>...>(std::forward<args_t>(args)...);
}

template<typename... args_t> 
tuple<args_t&&...> forward_as_tuple(args_t&&... args) {
  return tuple<args_t&&...>(std::forward<args_t>(args)...);
}

template<typename t, typename u>
struct decay_equiv {
  enum { value = std::is_same<typename std::decay<t>::type, u>::value };
};

int main(int argc, char** argv) {
  int i[] = { 0, 1, 2, 3, 4 };
  double d[] = { 0.1, 1.1, 2.1, 3.1, 4.1 };
  float f[] = { 0.2f, 1.2f, 2.2f, 3.2f, 4.2f };

  auto x = make_tuple(i);


  //auto x = make_tuple(i);
// get<0>(x) = i;
  // auto x = make_tuple(i, d, f);
 // tuple_t x = tuple_t(i, d, f);

  typedef int(&foo)[5];
  printf("Decay works %d\n", decay_equiv<foo, int*>::value);


  printf("%d\n", std::is_constructible<int*, int(&)[5]>::value);

 // printf("%x %x %x\n", get<0>(x), get<1>(x), get<2>(x));
  return 0;
}