// A harness over Google's snappy, never linked into lz: writes the
// reference's encoding of a file, decodes one, or times both.
//
//   snappy_ref compress RAW OUT
//   snappy_ref decompress COMPRESSED OUT
//   snappy_ref bench RAW COMPRESSED RUNS   best MB/s of RUNS, of the raw size

#include <snappy.h>

#include <chrono>
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

static double now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  if (argc >= 4 && !std::strcmp(argv[1], "compress")) {
    std::string raw = slurp(argv[2]), out;
    snappy::Compress(raw.data(), raw.size(), &out);
    spit(argv[3], out.data(), out.size());
    return 0;
  }
  if (argc >= 4 && !std::strcmp(argv[1], "decompress")) {
    std::string z = slurp(argv[2]), out;
    if (!snappy::Uncompress(z.data(), z.size(), &out)) {
      std::fprintf(stderr, "invalid snappy input\n");
      return 1;
    }
    spit(argv[3], out.data(), out.size());
    return 0;
  }
  if (argc >= 5 && !std::strcmp(argv[1], "bench")) {
    std::string raw = slurp(argv[2]), z = slurp(argv[3]);
    int runs = std::atoi(argv[4]);
    std::string zbuf(snappy::MaxCompressedLength(raw.size()), '\0');
    std::string rbuf(raw.size(), '\0');
    double best_c = 1e30, best_d = 1e30;
    size_t zlen = 0;
    for (int i = 0; i < runs; i++) {
      double t = now();
      snappy::RawCompress(raw.data(), raw.size(), &zbuf[0], &zlen);
      t = now() - t;
      if (t < best_c) best_c = t;
    }
    for (int i = 0; i < runs; i++) {
      double t = now();
      if (!snappy::RawUncompress(z.data(), z.size(), &rbuf[0])) {
        std::fprintf(stderr, "invalid snappy input\n");
        return 1;
      }
      t = now() - t;
      if (t < best_d) best_d = t;
    }
    if (rbuf != raw) { std::fprintf(stderr, "decode differs\n"); return 1; }
    double mb = raw.size() / 1e6;
    std::printf("%s: encode %.0f MB/s (ratio %.3f), decode %.0f MB/s\n",
                argv[2], mb / best_c, (double)raw.size() / zlen, mb / best_d);
    return 0;
  }
  std::fprintf(stderr, "usage: snappy_ref compress|decompress|bench ...\n");
  return 2;
}
