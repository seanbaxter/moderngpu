// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#include <map>
#include <list>
#include <cassert>
#include <vector>

#include <moderngpu/context.hxx>

BEGIN_MGPU_NAMESPACE

class block_allocator_t {
  enum { block_size = 4<< 20 };   // typical allocation size.

  enum : size_t { dirty = (size_t)-1, zero = (size_t)-2, used = (size_t)-3 };

  struct node_t;
  struct chunk_t;

  typedef std::map<char*, node_t>::iterator node_it;
  typedef std::list<chunk_t>::iterator chunk_it;
  typedef std::multimap<size_t, node_it>::iterator free_it;

  struct node_t {
    chunk_it chunk;
    size_t stream;
    std::multimap<size_t, node_it>::iterator free_node;
  };

  struct chunk_t {
    size_t capacity;
    size_t available;
    char* p;
  };

  // List of all chunks.
  std::list<chunk_t> chunks;

  // Sort free nodes by size.
  std::map<size_t, std::multimap<size_t, node_it> > free_nodes;

  // Data for all nodes, indexed by address.
  std::map<char*, node_t> nodes;

  size_t node_size(node_it node) const {
    chunk_it chunk = node->second.chunk;
    node_it next = std::next(node);
    const char* end = (nodes.end() != next && chunk == next->second.chunk) ?
      next->first : chunk->p + chunk->capacity;
    return end - node->first;
  }

  void remove(node_it node) {
    remove_free(node);
    nodes.erase(node);
  }

  free_it set_free(node_it node, size_t stream = dirty) {
    remove_free(node);

    node->second.free_node = free_nodes[stream].insert(
      std::make_pair(node_size(node), node)
    );
    node->second.stream = stream;
    return node->second.free_node;
  }

  void remove_free(node_it node) {
    if(used != node->second.stream)
      free_nodes[node->second.stream].erase(node->second.free_node);
    node->second.stream = used;
    node->second.free_node = free_nodes[dirty].end();
  }

  free_it get_free_node(size_t size) {
    auto free_node = free_nodes[dirty].lower_bound(size);
    if(free_nodes[dirty].end() == free_node) {
      size_t capacity = std::max<size_t>(size, block_size);

      std::list<chunk_t>::iterator chunk = chunks.insert(
        chunks.begin(), 
        chunk_t {
          capacity, 
          capacity, 
          (char*)block_allocate(capacity)
        }
      );
      
      // Insert and link this node to the next node.
      node_it node = insert(chunk, chunk->p);

      // Add the node to the free list.
      free_node = set_free(node);
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
    size_t stream = node->second.stream;
    chunk_it chunk = node->second.chunk;
    remove_free(node);

    // Subtract the node's size from the available space.
    chunk->available -= node_size(node);
    
    // Split the allocated node into two nodes.
    size_t excess = node_size(node) - size;
    if(excess >= 7) {
      // Update the sizing of the old node and chunk.
      chunk->available += excess;

      // Create a new node from the end of the old one.
      node_it new_node = insert(chunk, node->first + size);

      // Add the new node to the free list.
      set_free(new_node, stream);
    }
  }

  void coalesce(node_it node) {
    // Collapse this node into the left.
    remove_free(node);
    node_it prev = std::prev(node);
    node_it next = std::next(node);
    chunk_it chunk = node->second.chunk;
    bool prev_free = nodes.end() != prev && prev->second.chunk == chunk &&
      dirty == prev->second.stream;
    bool next_free = nodes.end() != next && next->second.chunk == chunk &&
      dirty == next->second.stream;

    if(prev_free) {
      remove(node);
      node = prev;
    }
    if(next_free) {
      remove(next);
    }

    // Insert the free node into the dirty buffer.
    set_free(node);
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
    usage_t u = usage_t();
    const auto& free_map = free_nodes.find(dirty)->second;
    u.largest_available = free_map.size() ? free_map.rbegin()->first : 0;
    for(const chunk_t& chunk : chunks) {
      u.allocated += chunk.capacity;
      u.available += chunk.available;
      if(chunk.capacity == chunk.available)
        ++u.free_blocks;
    }
    u.blocks = (int)chunks.size();
    u.nodes = (int)nodes.size();
    u.free_nodes = (int)free_map.size();
    return u;
  }

  // de-allocate all free blocks.
  void slim() {
    for(auto chunk = chunks.begin(); chunk != chunks.end(); ) {
      auto next = std::next(chunk);

      // This is a free chunk. Remove it.
      if(chunk->available == chunk->capacity) {
        node_it node = nodes.find(chunk->p);

        // De-allocate the block memory.
        block_free(chunk->p);
        chunk->p = nullptr;
        chunks.erase(chunk);

        // Remove it from the free map and the nodes map.
        remove(node);
      }
      chunk = next;
    }
  }

  void stream_sync(cudaStream_t stream) {
    while(free_nodes[(size_t)stream].size())
      coalesce(free_nodes[(size_t)stream].begin()->second);
  }

  // Called by the derived type's destructor.
  void reset() {
    stream_sync((cudaStream_t)zero);
    slim();
  }

  // return pointer,size to next part.
  // add lhs to managed memory.
  void* allocate(size_t size) {
    if(!size) size = 1;
    size_t offset = align_offset(size);
    size += offset;
    free_it free_node = get_free_node(size);

    node_it node = free_node->second;
    split(node, size);

    return align_ptr(node->first, offset);
  }

  void* allocate_zero(size_t size) {
    // Request an available free node in the zero queue.
    if(!size) size = 1;
    size_t offset = align_offset(size);
    size += offset;
    auto free_node = free_nodes[zero].lower_bound(size);

    if(free_node == free_nodes[zero].end()) {
      // Request 128KB of dirty data.
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

  void* allocate(size_t size, stream_t stream) {
    // Try to allocate from the current stream, then from the dirty pool.
    if(!size) size = 1;
    size_t offset = align_offset(size);
    size += offset;

    auto free_node = free_nodes[stream].lower_bound(size);
    // if(free_node == free_nodes[zero])


    return align_ptr(node, offset);
  }

  void free(void* p_) {
    char* p = static_cast<char*>(p_);
    node_it node = std::prev(nodes.upper_bound(p));

    chunk_it chunk = node->second.chunk;
    chunk->available += node_size(node);

    coalesce(node);
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

int main(int argc, char** argv) {
  {
    mgpu::host_allocator_t alloc;

    int count =       1000000;
    int iterations = 10000000;
    std::vector<void*> x(count);

    int alloc_count = 0;
    int free_count = 0;

    for(int i = 0; i < iterations; ++i) {
      int index = rand() % count;

      // Free or allocate.
      if(x[index]) {
        alloc.free(x[index]);
        x[index] = nullptr;
        ++free_count;
      } else {
        x[index] = alloc.allocate(rand() % 5000);
        ++alloc_count;
      }

      if(0 == i % 1000) {
        alloc.slim();
        auto usage = alloc.usage();

        printf("Iterations %5d  %lu %lu (%f%%)\n", i, 
          usage.allocated, usage.available, 100.0 * usage.available / usage.allocated);
      }
    }

    // Free all allocations.
    for(int i = 0; i < count; ++i)
      if(x[i]) {
        alloc.free(x[i]);
      }
    auto usage = alloc.usage();
    printf("DONE  %lu %lu %d %d\n", usage.allocated, usage.available, usage.nodes, usage.free_nodes);
  }

  return 0;
}