#include <thread>
#include <iostream>


int main(int argc, char** argv) {
  std::cout << std::thread::hardware_concurrency() << std::endl;
  return 0;
}