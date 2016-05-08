// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#include <map>
#include <list>
#include <cassert>
#include <vector>

class block_allocator_t {
  struct node_t;
  struct chunk_t;

  typedef std::map<char*, node_t>::iterator node_it;
  typedef std::list<chunk_t>::iterator chunk_it;

  struct node_t {
    chunk_it chunk;
    node_it prev, next;
    size_t size;
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
  std::multimap<size_t, node_it> zero_nodes;
  std::multimap<size_t, node_it> free_nodes;

  // Data for all nodes, indexed by address.
  std::map<char*, node_t> nodes;

  void remove(node_it node) {
    // Disconnect this node from the chain.
    if(nodes.end() != node->second.next)
      node->second.next->second.prev = node->second.prev;
    if(nodes.end() != node->second.prev)
      node->second.prev->second.next = node->second.next;

    // Erase the free node.
    if(free_nodes.end() != node->second.free_node)
      free_nodes.erase(node->second.free_node);

    // Erase the node.
    nodes.erase(node);
  }

  void change_key(node_it node) {
    free_nodes.erase(node->second.free_node);
    node->second.free_node = free_nodes.insert(
      std::make_pair(node->second.size, node)
    );
  }



protected:
  virtual void* block_allocate(size_t size) = 0;
  virtual void block_free(void* p) = 0;
  virtual void block_zero(void* p, size_t size) = 0;

public:

  struct usage_t {
    size_t allocated, available;
    size_t largest_available;
    int blocks;
    int free_blocks;
  };

  ~block_allocator_t() {
    assert(free_nodes.size() == chunks.size());
    slim();
  }

  usage_t usage() const {
    usage_t u = usage_t();
    u.largest_available = free_nodes.size() ? free_nodes.rbegin()->first : 0;
    for(const chunk_t& chunk : chunks) {
      u.allocated += chunk.capacity;
      u.available += chunk.available;
      if(chunk.capacity == chunk.available)
        ++u.free_blocks;
    }
    u.blocks = (int)chunks.size();

    return u;
  }

  // de-allocate all free blocks.
  void slim() {
    for(auto chunk = chunks.begin(); chunk != chunks.end(); ) {
      auto next = std::next(chunk);

        // This is a free chunk. Remove it.
      if(chunk->available == chunk->capacity) {
        node_it node = nodes.find(chunk->p);

        // Remove it from the free map and the nodes map.
        remove(node);

        // De-allocate the block memory.
        block_free(chunk->p);
        chunk->p = nullptr;
        chunks.erase(chunk);
      }
      chunk = next;
    }
  }

  void* allocate_zero(size_t size) {
    // Allocate from zero memory.
    auto zero_node = zero_nodes.lower_bound(size);


  }

  // return pointer,size to next part.
  // add lhs to managed memory.
  void* allocate(size_t size) {
    // TODO: align me.
    // TODO: look in the stream free list first.
    if(size < 4) size = 4;
    auto free_node = free_nodes.lower_bound(size);
    if(free_node == free_nodes.end()) {
      size_t alloc_size = 1<< 20;
      std::list<chunk_t>::iterator chunk = chunks.insert(
        chunks.begin(), 
        chunk_t {
          alloc_size, 
          alloc_size, 
          (char*)block_allocate(size)
        }
      );
      printf("ALLOCATED %p\n", chunk->p);

      node_it next = nodes.begin();
      node_it node = nodes.insert(std::make_pair(
        chunk->p, 
        node_t {
          chunk,
          nodes.end(), next, 
          chunk->available,
          free_nodes.end()
         }
      )).first;

      if(nodes.end() != next)
        next->second.prev = node;
      
      free_node = node->second.free_node = free_nodes.insert(
        std::make_pair(node->second.size, node)
      );
    }

    node_it node = free_node->second;
    chunk_it chunk = node->second.chunk;

    // Set this node to allocated.
    free_nodes.erase(free_node);
    node->second.free_node = free_nodes.end();

    // Subtract the node's size from the available space.
    chunk->available -= node->second.size;
    
    // Split the allocated node into two nodes.
    size_t excess = node->second.size - size;
    if(excess >= 16) {
      // Update the sizing of the old node and chunk.
      node->second.size -= excess;
      chunk->available += excess;

      // Create a new node from the end of the old one.
      node_it new_node = nodes.insert(std::make_pair(
        node->first + size,
        node_t {
          chunk,
          node, node->second.next,
          excess,
          free_nodes.end()
        }
      )).first;

      if(nodes.end() != node->second.next)
        node->second.next->second.prev = new_node;
      node->second.next = new_node;
      new_node->second.free_node = 
        free_nodes.insert(std::make_pair(excess, new_node));
    }
    return node->first;
  }

  void free(void* p_) {
    char* p = static_cast<char*>(p_);
    node_it node = nodes.lower_bound(p);

    chunk_it chunk = node->second.chunk;
    chunk->available += node->second.size;

    node_it prev = node->second.prev;
    node_it next = node->second.next;

    bool collapse_left = (nodes.end() != prev) && 
      (prev->second.chunk == chunk) &&
      (free_nodes.end() != prev->second.free_node);
    bool collapse_right = (nodes.end() != next) &&
      (next->second.chunk == chunk) &&
      (free_nodes.end() != next->second.free_node);

    if(collapse_left) {
      prev->second.size += node->second.size;
      remove(node);
      if(collapse_right) {
        prev->second.size += next->second.size;
        remove(next);
      }
      change_key(prev);
    } else {
      if(collapse_right) {
        node->second.size += next->second.size;
        remove(next);
      }
      node->second.free_node = free_nodes.insert(
        std::make_pair(node->second.size, node)
      );
    }
  }
};

class device_allocator_t : block_allocator_t {
protected:
  virtual void* block_allocate(size_t size) {
    void* p;
    cudaResult_t result = cudaMalloc(&p, size);
    if(cudaSuccess != result) throw cuda_exception_t(result);
    return p;
  }
  
  virtual void block_free(void* p) {
    cudaResult_t result = cudaFree(p);
    if(cudaSuccess != result) throw cuda_exception_t(result);
  }

  virtual void block_zero(void* p, size_t size) {
    cudaMemset(p, 0, size);
  }
};

class host_allocator_t : block_allocator_t {
protected:
  virtual void* block_allocate(size_t size) {
    void* p;
    cudaResult_t result = cudaMallocHost(&p, size);
    if(cudaSuccess != result) throw cuda_exception_t(result);
    return p;
  }
  
  virtual void block_free(void* p) {
    cudaResult_t result = cudaFreeHost(p);
    if(cudaSuccess != result) throw cuda_exception_t(result);
  }

  virtual void block_zero(void* p, size_t size) {
    memset(p, 0, size);
  }  
};

#if 0
  // Alloc GPU memory.
  virtual void* alloc(size_t size, memory_space_t space) {
    void* p = nullptr;
    if(size) {
      cudaError_t result = (memory_space_device == space) ? 
        cudaMalloc(&p, size) :
        cudaMallocHost(&p, size);
      if(cudaSuccess != result) throw cuda_exception_t(result);
    }
    return p;    
  }

  virtual void free(void* p, memory_space_t space) {
    if(p) {
      cudaError_t result = (memory_space_device == space) ? 
        cudaFree(p) :
        cudaFreeHost(p);
      if(cudaSuccess != result) throw cuda_exception_t(result);
    }
  }
#endif

int main(int argc, char** argv) {
  {
    block_allocator_t alloc;

    int count =       10000;
    int iterations = 1000000;
    std::vector<void*> x(count);

    for(int i = 0; i < iterations; ++i) {
      int index = rand() % count;

      // Free or allocate.
      if(x[index]) {
        alloc.free(x[index]);
        x[index] = nullptr;
      } else {
        x[index] = alloc.allocate(rand() % 5000);
      }

      if(0 == i % 1000) {
        auto usage = alloc.usage();
     //   printf("Iterations %5d  %lu %lu\n", i, usage.allocated, usage.available);
      }
    }

    // Free all allocations.
    for(int i = 0; i < count; ++i)
      if(x[i]) {
        alloc.free(x[i]);
      }
        auto usage = alloc.usage();
        printf("DONE %lu %lu\n", usage.allocated, usage.available);
  }

  return 0;
}