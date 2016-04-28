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
    std::map<size_t, node_it>::iterator free_node;
  };

  struct chunk_t {
    size_t capacity;
    size_t available;
    char* p;
  };

  // List of all chunks.
  std::list<chunk_t> chunks;

  // Sort free nodes by size.
  std::map<size_t, node_it> free_nodes;

  // Data for all nodes, indexed by address.
  std::map<char*, node_t> nodes;


public:

  size_t allocated() const {
    size_t a = 0;
    for(const chunk_t& chunk : chunks)
      a += chunk.capacity - chunk.available;
    return a;
  }
  size_t capacity() const {
    size_t a = 0;
    for(const chunk_t& chunk : chunks)
      a += chunk.capacity;
    return a;
  }

  void* allocate(size_t size) {
    // TODO: align me.
    // TODO: look in the stream free list first.
    auto free_node = free_nodes.lower_bound(size);
    if(free_node == free_nodes.end()) {
      size_t alloc_size = 1<< 20;
      std::list<chunk_t>::iterator chunk = chunks.insert(
        chunks.begin(), 
        chunk_t {
          alloc_size, 
          alloc_size, 
          (char*)malloc(alloc_size)
        }
      );

      node_it next = nodes.begin();
      node_it node = nodes.insert(std::make_pair(
        chunk->p, 
        node_t {
          chunk,
          nodes.end(), next, 
          chunk->available,
         }
      )).first;

      assert(nodes.end() == next->second.prev);
      next->second.prev = node;
      free_node = node->second.free_node = free_nodes.insert(
        std::make_pair(node->second.size, node)
      ).first;
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
          node->second.chunk,
          node, node->second.next,
          excess
        }
      )).first;

      if(nodes.end() != node->second.next->second.prev)
        node->second.next->second.prev = new_node;
      node->second.next = new_node;
      new_node->second.free_node = 
        free_nodes.insert(std::make_pair(excess, new_node)).first;
    }

    return node->first;
  }

  void free(void* p_) {
    char* p = static_cast<char*>(p_);
    node_it node = nodes.lower_bound(p);
    assert(nodes.end() != node);

    chunk_it chunk = node->second.chunk;
    assert(p < chunk->p + node->second.size);

    chunk->available += node->second.size;

    // Collapse this node with its neighbor to the left.
    node_it prev = node->second.prev;
    node_it next = node->second.next;
    if(nodes.end() != prev && 
      prev->second.chunk == chunk &&
      free_nodes.end() != prev->second.free_node) {
      
      // The preceding node is in the same chunk and also free. Coalesce 
      // and erase this node.
      prev->second.next = next;
      prev->second.size += node->second.size;
      nodes.erase(node);
      node = prev;
    } else {
      node->second.free_node = free_nodes.insert(
        std::make_pair(node->second.size, node)).first;
    }

    // Collapse this node with the neighbor to the right.
    if(nodes.end() != next &&
      next->second.chunk == chunk &&
      free_nodes.end() != next->second.free_node) {

      // The next node is in the same chunk and also free. Coalesce and 
      // erase that node.
      if(nodes.end() != next->second.next)
        next->second.next->second.prev = node;
      node->second.size += next->second.size;
      free_nodes.erase(next->second.free_node);
      nodes.erase(next);
    }
  }
};

int main(int argc, char** argv) {

  block_allocator_t alloc;

  int count = 10000;
  int iterations = 100000;
  std::vector<void*> x(count);

  for(int i = 0; i < iterations; ++i) {
    int index = rand() % count;

    // Free or allocate.
    if(x[index]) {
      alloc.free(x[index]);
      x[index] = nullptr;
    } else {
      x[index] = alloc.allocate(rand() % 100000);
    }
  }

  // Free all allocations.
  for(int i = 0; i < count; ++i)
    if(x[i]) alloc.free(x[i]);

  printf("%lu %lu\n", alloc.allocated(), alloc.capacity());

  return 0;
}