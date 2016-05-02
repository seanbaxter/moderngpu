// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#pragma once
#include "cpp11.hxx"

// A barebones tuple implementation for CUDA.

BEGIN_MGPU_NAMESPACE

// tuple

////////////////////////////////////////////////////////////////////////////////
// Tuple comparison operators.

template<typename... args_t>
MGPU_HOST_DEVICE bool operator<(tuple<args_t...> a, tuple<args_t...> b) {
  if(get<0>(a) < get<0>(b)) return true;
  if(get<0>(b) < get<0>(a)) return false;
  return a.inner() < b.inner();
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
  return (get<0>(a) == get<0>(b)) && (a.inner() == b.inner());
}
template<typename... args_t>
MGPU_HOST_DEVICE bool operator!=(tuple<args_t...> a, tuple<args_t...> b) {
  return !(a == b);
}

MGPU_HOST_DEVICE bool operator<(tuple<> a, tuple<> b) {
  return false;
}
MGPU_HOST_DEVICE bool operator<=(tuple<> a, tuple<> b) {
  return true;
}
MGPU_HOST_DEVICE bool operator>(tuple<> a, tuple<> b) {
  return false;
}
MGPU_HOST_DEVICE bool operator>=(tuple<> a, tuple<> b) {
  return true;
}
MGPU_HOST_DEVICE bool operator==(tuple<> a, tuple<> b) {
  return true;
}
MGPU_HOST_DEVICE bool operator!=(tuple<> a, tuple<> b) {
  return false;
}

END_MGPU_NAMESPACE
