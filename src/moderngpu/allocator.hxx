#pragma once

#include <map>
#include <list>
#include <mutex>
#include "types.hxx"

BEGIN_MGPU_NAMESPACE

class block_allocator_t {
public:
  struct pool_t {
    int ordinal;
    cudaStream_t stream;
    bool operator<(pool_t rhs) const {
      return ordinal < rhs.ordinal || 
        (!(rhs.ordinal < ordinal) && stream < rhs.stream);
    }
    bool operator==(pool_t rhs) const {
      return ordinal == rhs.ordinal && stream == rhs.stream;
    }
    bool operator!=(pool_t rhs) const {
      return !(*this == rhs);
    }
  };

private:
  enum { block_size = 4<< 20 };   // typical allocation size.

  struct node_t;
  struct chunk_t;

  typedef std::map<char*, node_t>::iterator node_it;
  typedef std::map<char*, node_t>::const_iterator const_it;
  typedef std::list<chunk_t>::iterator chunk_it;
  typedef std::multimap<size_t, node_it>::iterator free_it;


  const pool_t dirty = pool_t { -1, nullptr };
  const pool_t zero = pool_t { -2, nullptr };
  const pool_t used = pool_t { -3, nullptr };

  struct node_t {
    chunk_it chunk;
    pool_t pool;
    std::multimap<size_t, node_it>::iterator free_node;
  };

  struct chunk_t {
    size_t capacity;
    char* p;
  };

  // Mutex for exclusive access.
  mutable std::recursive_mutex mutex;

  // List of all chunks.
  std::list<chunk_t> chunks;

  // Sort free nodes by size.
  std::map<pool_t, std::multimap<size_t, node_it> > free_nodes;

  // Data for all nodes, indexed by address.
  std::map<char*, node_t> nodes;

  size_t node_size(const_it node) const {
    chunk_it chunk = node->second.chunk;
    const_it next = std::next(node);
    const char* end = (nodes.end() != next && chunk == next->second.chunk) ?
      next->first : chunk->p + chunk->capacity;
    return end - node->first;
  }

  // Return the free nodes in the dirty heap only for this chunk.
  size_t available(chunk_t chunk) const {
    size_t size = 0;
    auto end = nodes.lower_bound(chunk.p + chunk.capacity);
    for(auto it = nodes.lower_bound(chunk.p); it != end; ++it) {
      if(dirty == it->second.pool || zero == it->second.pool)
        size += node_size(it);
    }
    return size;
  }

  void remove(node_it node) {
    remove_free(node);
    nodes.erase(node);
  }

  free_it set_free(node_it node, pool_t pool) {
    remove_free(node);

    node->second.free_node = free_nodes[pool].insert(
      std::make_pair(node_size(node), node)
    );
    node->second.pool = pool;
    return node->second.free_node;
  }

  void remove_free(node_it node) {
    if(used != node->second.pool)
      free_nodes[node->second.pool].erase(node->second.free_node);
    node->second.pool = used;
    node->second.free_node = free_nodes[dirty].end();
  }

  free_it get_free_node(size_t size) {
    auto free_node = free_nodes[dirty].lower_bound(size);
    if(free_nodes[dirty].end() == free_node) {
      // TODO: Calculate free nodes waiting on streams and optionally
      // cudaDeviceSynchronize and release all free lists to clear up
      // the most space.
      // Release unused nodes here.
      slim();

      size_t capacity = std::max<size_t>(size, block_size);

      std::list<chunk_t>::iterator chunk = chunks.insert(
        chunks.begin(), 
        chunk_t { capacity, (char*)block_allocate(capacity) }
      );
      
      // Insert and link this node to the next node.
      node_it node = insert(chunk, chunk->p);

      // Add the node to the free list.
      free_node = set_free(node, dirty);
    }
    return free_node;
  }

  node_it insert(chunk_it chunk, char* p) {
    return nodes.insert(std::make_pair(
      p, node_t { chunk, used, free_nodes[dirty].end() }
    )).first;
  }

  static size_t align_offset(size_t size) {
    size_t align = 128;
    while(align > size) align /= 2;
    return align - 1;
  }

  static char* align_ptr(char* p, size_t offset) {
    size_t x = (size_t)p;
    x = ~offset & (x + offset);
    return (char*)x;
  }

  // Split a free node into an allocated node and a smaller free node.
  void split(node_it node, size_t size) {
    pool_t pool = node->second.pool;
    chunk_it chunk = node->second.chunk;
    remove_free(node);

    // Split the allocated node into two nodes.
    size_t excess = node_size(node) - size;
    if(excess >= 7) {
      // Create a new node from the end of the old one.
      node_it new_node = insert(chunk, node->first + size);

      // Add the new node to the free list.
      set_free(new_node, pool);
    }
  }

  void coalesce(node_it node, pool_t pool) {
    // Collapse this node into the left.
    remove_free(node);
    chunk_it chunk = node->second.chunk;
    node_it prev = std::prev(node);
    node_it next = std::next(node);
    bool prev_free = nodes.end() != prev && prev->second.chunk == chunk &&
      pool == prev->second.pool;
    bool next_free = nodes.end() != next && next->second.chunk == chunk &&
      pool == next->second.pool;

    if(prev_free) {
      remove(node);
      node = prev;
    }
    if(next_free) {
      remove(next);
    }

    // Insert the free node into the pool's buffer.
    set_free(node, pool);
  }

protected:
  virtual void* block_allocate(size_t size) = 0;
  virtual void block_free(void* p) = 0;
  virtual void block_zero(void* p, size_t size) = 0;
  virtual ~block_allocator_t() { };

public:
  struct usage_t {
    size_t allocated, available;
    size_t largest_available;
    int blocks;
    int free_blocks;
    int free_nodes;
    int nodes;
  };

  usage_t usage() const {
    std::lock_guard<std::recursive_mutex> guard(mutex);

    usage_t u = usage_t();
    const auto& free_map = free_nodes.find(dirty)->second;
    u.largest_available = free_map.size() ? free_map.rbegin()->first : 0;
    for(auto chunk : chunks) {
      size_t chunk_available = available(chunk);
      u.allocated += chunk.capacity;
      u.available += chunk_available;
      if(chunk.capacity == chunk_available)
        ++u.free_blocks;
    }
    u.blocks = (int)chunks.size();
    u.nodes = (int)nodes.size();
    u.free_nodes = (int)free_map.size();
    return u;
  }

  // de-allocate all free blocks and zero blocks that are mostly free.
  void slim(pool_t pool) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    
    pool_release(zero);
    if(dirty != pool) {
      if(nullptr == pool.stream)
        pool_release_ordinal(pool.ordinal);
      else
        pool_release(pool);
    }

    for(auto chunk = chunks.begin(); chunk != chunks.end(); ) {
      auto next = std::next(chunk);

      // This is a free chunk. Remove it.
      if(available(*chunk) == chunk->capacity) {
        node_it node = nodes.find(chunk->p);

        // Remove it from the free map and the nodes map.
        remove(node);

        // De-allocate the block memory.
        block_free(chunk->p);
        chunks.erase(chunk);
      }
      chunk = next;
    }
  }

  void slim() {
    slim(dirty);
  }

  // Release the free nodes on a pool back to the dirty pool. This should
  // be called from a cudaStreamSynchronize.
  void pool_release(pool_t pool) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    
    while(free_nodes[pool].size())
      coalesce(free_nodes[pool].begin()->second, dirty);
  }

  void pool_release_ordinal(int ordinal) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    
    for(auto& pool : free_nodes) {
      if(ordinal == pool.first.ordinal)
        pool_release(pool.first);
    }
  }

  void pool_release_all() {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    
    for(auto& pool : free_nodes) {
      if(dirty != pool.first)
        pool_release(pool.first);
    }
  }

  // Called by the derived type's destructor.
  void reset() {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    
    // Release all pools.
    pool_release_all();
    slim();
  }

  // Allocate dirty memory. This bypasses any pool cache.
  void* allocate(size_t size) {
    std::lock_guard<std::recursive_mutex> guard(mutex);

    if(!size) size = 1;
    size_t offset = align_offset(size);
    size += offset;
    free_it free_node = get_free_node(size, dirty);

    node_it node = free_node->second;
    split(node, size);

    return align_ptr(node->first, offset);
  }

  // Allocate zero'd memory.
  void* allocate_zero(size_t size, pool_t pool) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    
    // Request an available free node in the zero queue.
    if(!size) size = 1;
    size_t offset = align_offset(size);
    size += offset;
    auto free_node = free_nodes[zero].lower_bound(size);

    // Get memory from this stream's pool first.

    if(free_node == free_nodes[zero].end()) {
      // Request at least 128KB of adjacent memory to quickly zero.
      size_t capacity = std::max<size_t>(size, 128<< 10);
      free_node = get_free_node(capacity);

      // Clear the returned dirty node.
      node_it node = free_node->second;
      block_zero(node->first, node_size(node));

      // Move the node to the zero list.
      free_node = set_free(node, zero);
    }

    node_it node = free_node->second;
    split(node, size);

    return align_ptr(node->first, offset);
  }

  // Allocate dirty memory. Prefer memory available in the specified pool.
  void* allocate(size_t size, pool_t pool) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    
    // Try to allocate from the current pool, then from the dirty pool.
    if(!size) size = 1;
    size_t offset = align_offset(size);
    size += offset;

    auto free_node = free_nodes[pool].lower_bound(size);
    if(free_node != free_nodes[pool].end()) {
      node_it node = free_node->second;
      split(node, size);
      return align_ptr(node->first, offset);
    } else
      return allocate(size - offset);
  }

  // Return for immediate reuse within the same pool. 
  // Using the default dirty pool may be unsafe in a multi-stream
  // environment.
  void free(void* p_, pool_t pool) {
    std::lock_guard<std::recursive_mutex> guard(mutex);
    
    char* p = static_cast<char*>(p_);
    node_it node = std::prev(nodes.upper_bound(p));
    coalesce(node, pool);
  }

  void free(void* p_) {
    return free(p_, dirty);
  }
};

class device_allocator_t : public block_allocator_t {
protected:
  virtual void* block_allocate(size_t size) {
    void* p;
    cudaError_t result = cudaMalloc(&p, size);
    if(cudaSuccess != result) throw cuda_exception_t(result);
    return p;
  }
  
  virtual void block_free(void* p) {
    cudaError_t result = cudaFree(p);
    if(cudaSuccess != result) throw cuda_exception_t(result);
  }

  virtual void block_zero(void* p, size_t size) {
    cudaMemset(p, 0, size);
  }
public:
  virtual ~device_allocator_t() { 
    reset();
  }
};

class host_allocator_t : public block_allocator_t {
protected:
  virtual void* block_allocate(size_t size) {
    void* p;
    cudaError_t result = cudaMallocHost(&p, size);
    if(cudaSuccess != result) throw cuda_exception_t(result);
    return p;
  }
  
  virtual void block_free(void* p) {
    cudaError_t result = cudaFreeHost(p);
    if(cudaSuccess != result) throw cuda_exception_t(result);
  }

  virtual void block_zero(void* p, size_t size) {
    memset(p, 0, size);
  }  
public:
  virtual ~host_allocator_t() { 
    reset();
  }
};

END_MGPU_NAMESPACE
