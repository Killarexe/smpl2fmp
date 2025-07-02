#include "CLI11/CLI11.hpp"
#include <iostream>

int main(int argc, char** argv) {
  std::cout << "Hello, world!" << std::endl;

  CLI::App app{"A prgram which convert samples into FM patches."};
  argv = app.ensure_utf8(argv);

  CLI11_PARSE(app, argc, argv);
  return 0;
}
