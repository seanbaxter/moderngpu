// moderngpu copyright (c) 2016, Sean Baxter http://www.moderngpu.com
#include <map>
#include <list>
#include <cassert>
#include <vector>

#include <moderngpu/context.hxx>

BEGIN_MGPU_NAMESPACE

class block_allocator_t {
  struct node_t;
  struct chunk_t;

  typedef std::map<char*, node_t>::iterator node_it;
  typedef std::list<chunk_t>::iterator chunk_it;
  typedef std::multimap<size_t, node_it>::iterator free_it;

  struct node_t {
    chunk_it chunk;
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
    // Erase the free node.
    if(free_nodes.end() != node->second.free_node)
      free_nodes.erase(node->second.free_node);

    // Erase the node.
    nodes.erase(node);
  }

  free_it set_free(node_it node) {
    remove_free(node);

    node->second.free_node = free_nodes.insert(
      std::make_pair(node->second.size, node)
    );
    return node->second.free_node;
  }

  void remove_free(node_it node) {
    if(free_nodes.end() != node->second.free_node)
      free_nodes.erase(node->second.free_node);
    node->second.free_node = free_nodes.end();
  }

  free_it insert_chunk(size_t alloc_size) {
    std::list<chunk_t>::iterator chunk = chunks.insert(
      chunks.begin(), 
      chunk_t {
        alloc_size, 
        alloc_size, 
        (char*)block_allocate(alloc_size)
      }
    );

    // Insert and link this node to the next node.
    node_it node = insert(chunk, chunk->p, chunk->available);

    // Add the node to the free list.
    return set_free(node);
  }

  node_it insert(chunk_it chunk, char* p, size_t size) {
    return nodes.insert(std::make_pair(
      p, 
      node_t {
        chunk,
        size,
        free_nodes.end()
      }
    )).first;
  }

protected:
  virtual void* block_allocate(size_t size) = 0;
  virtual void block_free(void* p) = 0;
  virtual void block_zero(void* p, size_t size) = 0;
  virtual ~block_allocator_t() { 
    assert(free_nodes.size() == chunks.size());
  };

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
    u.largest_available = free_nodes.size() ? free_nodes.rbegin()->first : 0;
    for(const chunk_t& chunk : chunks) {
      u.allocated += chunk.capacity;
      u.available += chunk.available;
      if(chunk.capacity == chunk.available)
        ++u.free_blocks;
    }
    u.blocks = (int)chunks.size();
    u.nodes = (int)nodes.size();
    u.free_nodes = (int)free_nodes.size();
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

  // return pointer,size to next part.
  // add lhs to managed memory.
  void* allocate(size_t size) {
    // TODO: align me.
    // TODO: look in the stream free list first.
    if(size < 4) size = 4;
    auto free_node = free_nodes.lower_bound(size);

    if(free_node == free_nodes.end())
      free_node = insert_chunk(std::max<size_t>(size, 1<< 20));
    
    node_it node = free_node->second;
    chunk_it chunk = node->second.chunk;

    // Set this node to allocated.
    remove_free(node);

    // Subtract the node's size from the available space.
    chunk->available -= node->second.size;
    
    // Split the allocated node into two nodes.
    size_t excess = node->second.size - size;
    if(excess >= 16) {
      // Update the sizing of the old node and chunk.
      node->second.size -= excess;
      chunk->available += excess;

      // Create a new node from the end of the old one.
      node_it new_node = insert(chunk, node->first + size, excess);

      // Add the new node to the free list.
      set_free(new_node);
    }

    return node->first;
  }

  void free(void* p_) {
    char* p = static_cast<char*>(p_);
    node_it node = nodes.lower_bound(p);

    chunk_it chunk = node->second.chunk;
    chunk->available += node->second.size;

    // Collapse this node into the left.
    node_it prev = std::prev(node);
    node_it next = std::next(node);
    bool prev_free = nodes.end() != prev && prev->second.chunk == chunk &&
      free_nodes.end() != prev->second.free_node;
    bool next_free = nodes.end() != next && next->second.chunk == chunk &&
      free_nodes.end() != next->second.free_node;

    if(prev_free) {
      prev->second.size += node->second.size;
      node->second.size = 0;
      remove(node);
      node = prev;
    }
    if(next_free) {
      node->second.size += next->second.size;
      next->second.size = 0;
      remove(next);
    }

    // Insert the free node.
    set_free(node);
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
    slim();
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
    slim();
  }
};

END_MGPU_NAMESPACE

int main(int argc, char** argv) {
  {
    mgpu::host_allocator_t alloc;

    int count =       10000;
    int iterations = 1000000;
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
        auto usage = alloc.usage();
        printf("Iterations %5d  %lu %lu\n", i, usage.allocated, usage.available);
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