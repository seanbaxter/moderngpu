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

  void remove(node_it node) {
    // Disconnect this node from the chain.
    if(nodes.end() != node->second.next)
      node->second.next->second.prev = node->second.prev;
    if(nodes.end() != node->second.prev)
      node->second.prev->second.next = node->second.next;

    // Erase the free node.
    if(free_nodes.end() != node->second.free_node) {
      assert(node->second.free_node->first == node->second.size);
      free_nodes.erase(node->second.free_node);
    } else 
      node->second.chunk->available += node->second.size;

    // Erase the node.
    nodes.erase(node);
  }

public:

  ~block_allocator_t() {
    assert(free_nodes.size() == chunks.size());
    while(chunks.size()) {
      assert(chunks.begin()->capacity == chunks.begin()->available);
      chunks.erase(chunks.begin());
    }
  }

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

  void validate_list() const {
    
    for(auto i = nodes.begin(); i != nodes.end(); ++ i) {
      if(nodes.end() != i->second.prev) {
        assert(i->second.prev->second.next == i);
      }
      if(nodes.end() != i->second.next) {
        assert(i->second.next->second.prev == i);
      }
    }

  }

  void print_nodes() const {
    validate_list();
    printf("%d nodes:\n", (int)nodes.size());
    for(const auto& n : nodes) {
      printf("%8lu : %p (%8lu)  (%8lu - %8lu)\n", n.second.size, n.first, 
        free_nodes.end() != n.second.free_node ? n.second.free_node->first : 0,
        n.second.prev != nodes.end() ? n.second.prev->second.size : 0,
        n.second.next != nodes.end() ? n.second.next->second.size : 0
      );

      if(free_nodes.end() != n.second.free_node) {
        assert(n.second.free_node->first == n.second.size);
      }
    }
    printf("%d free nodes:\n", (int)free_nodes.size());
    for(const auto& n : free_nodes)
      printf("%8lu : %p (%d)  %lu\n", n.first, n.second->first, 
        free_nodes.end() != n.second->second.free_node,
        n.second->second.size);
  }

  void* allocate(size_t size) {
    print_nodes();
    printf("ALLOC %lu\n", size);
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
          free_nodes.end()
         }
      )).first;

      if(nodes.end() != next) {
        assert(nodes.end() == next->second.prev);
        next->second.prev = node;
      }
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
        free_nodes.insert(std::make_pair(excess, new_node)).first;
    }

    validate_list();
    printf("ALOC %8lu - %p\n", size, node->first);

    return node->first;
  }

  void free(void* p_) {
    print_nodes();
    char* p = static_cast<char*>(p_);
    node_it node = nodes.lower_bound(p);
    assert(nodes.end() != node);

    printf("FREE %8lu - %p\n", node->second.size, node->first);

    chunk_it chunk = node->second.chunk;
    chunk->available += node->second.size;

    // Collapse this node with its neighbor to the left.
    node_it prev = node->second.prev;
    node_it next = node->second.next;
    if(nodes.end() != prev && 
      prev->second.chunk == chunk &&
      free_nodes.end() != prev->second.free_node) {

      assert(prev->second.size == prev->second.free_node->first);

      // The preceding node is in the same chunk and also free. Coalesce 
      // and erase this node.
      prev->second.size += node->second.size;
      node->second.size = 0;
      free_nodes.erase(prev->second.free_node);
      prev->second.free_node = free_nodes.insert(
        std::make_pair(prev->second.size, prev)
      ).first;

      remove(node);      
      node = prev;
    } else {
      node->second.free_node = free_nodes.insert(
        std::make_pair(node->second.size, node)).first;
    }

    print_nodes();
    printf("FREE(2)\n");
    // Collapse this node with the neighbor to the right.
    if(nodes.end() != next &&
      next->second.chunk == chunk &&
      free_nodes.end() != next->second.free_node) {

      assert(next->second.size == next->second.free_node->first);

      node->second.size += next->second.size;

      free_nodes.erase(node->second.free_node);
      node->second.free_node = free_nodes.insert(
        std::make_pair(node->second.size, node)
      ).first;

      remove(next);
    }
    printf("FREE DONE\n");
  }
};

int main(int argc, char** argv) {

  block_allocator_t alloc;

  int count = 100;
  int iterations = 1000;
  std::vector<void*> x(count);

  for(int i = 0; i < iterations; ++i) {
    int index = rand() % count;

    // Free or allocate.
    if(x[index]) {
      alloc.free(x[index]);
      x[index] = nullptr;
    } else {
      x[index] = alloc.allocate(rand() % 10000);
    }

    printf("Iterations %5d  %lu %lu\n", i, alloc.allocated(), alloc.capacity());

    printf("\n\n");

  }

  // Free all allocations.
  for(int i = 0; i < count; ++i)
    if(x[i]) {
      alloc.free(x[i]);
      alloc.print_nodes();
    }
  printf("%lu %lu\n", alloc.allocated(), alloc.capacity());

  return 0;
}