// A harness over Google's snappy, never linked into lz: writes the
// reference's encoding of a file, for google_snappy.py.
//
//   snappy_ref compress RAW OUT

#include <snappy.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

static std::string slurp(const char *path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path); std::exit(2); }
  return std::string(std::istreambuf_iterator<char>(f), {});
}

static void spit(const char *path, const char *p, size_t n) {
  std::ofstream f(path, std::ios::binary);
  f.write(p, (std::streamsize)n);
  if (!f) { std::fprintf(stderr, "cannot write %s\n", path); std::exit(2); }
}

int main(int argc, char **argv) {
  if (argc == 4 && !std::strcmp(argv[1], "compress")) {
    std::string raw = slurp(argv[2]), out;
    snappy::Compress(raw.data(), raw.size(), &out);
    spit(argv[3], out.data(), out.size());
    return 0;
  }
  std::fprintf(stderr, "usage: snappy_ref compress RAW OUT\n");
  return 2;
}
