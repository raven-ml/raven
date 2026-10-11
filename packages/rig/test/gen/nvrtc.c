/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Compiles one CUDA source to a cubin with NVRTC, as a driver compiles its
   kernels at run time, with only --gpu-architecture and --minimal:

     nvrtc SOURCE ARCH OUT

   such as nvrtc simple_add.cu sm_89 simple_add_sm89.cubin. It prints the
   compiler's log when the source does not compile. */

#define _GNU_SOURCE
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>

static char *slurp(const char *path) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  rewind(f);
  char *s = malloc(n + 1);
  if (s == NULL || fread(s, 1, n, f) != (size_t)n) {
    fclose(f);
    free(s);
    return NULL;
  }
  fclose(f);
  s[n] = '\0';
  return s;
}

static int check(nvrtcResult r, const char *what) {
  if (r == NVRTC_SUCCESS) return 0;
  fprintf(stderr, "nvrtc: %s: %s\n", what, nvrtcGetErrorString(r));
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: nvrtc SOURCE ARCH OUT\n");
    return 2;
  }
  char *src = slurp(argv[1]);
  if (src == NULL) {
    perror(argv[1]);
    return 1;
  }
  char arch[64];
  snprintf(arch, sizeof arch, "--gpu-architecture=%s", argv[2]);
  const char *opts[] = {arch, "--minimal"};
  nvrtcProgram p;
  if (check(nvrtcCreateProgram(&p, src, argv[1], 0, NULL, NULL), "create"))
    return 1;
  if (nvrtcCompileProgram(p, 2, opts) != NVRTC_SUCCESS) {
    size_t n;
    nvrtcGetProgramLogSize(p, &n);
    char *log = malloc(n);
    if (log == NULL) {
      fprintf(stderr, "nvrtc: no memory for the log\n");
      return 1;
    }
    nvrtcGetProgramLog(p, log);
    fprintf(stderr, "%s\n", log);
    return 1;
  }
  size_t n;
  if (check(nvrtcGetCUBINSize(p, &n), "cubin size")) return 1;
  char *cubin = malloc(n);
  if (cubin == NULL) {
    fprintf(stderr, "nvrtc: no memory for the cubin\n");
    return 1;
  }
  if (check(nvrtcGetCUBIN(p, cubin), "cubin")) return 1;
  FILE *out = fopen(argv[3], "wb");
  if (out == NULL || fwrite(cubin, 1, n, out) != n || fclose(out) != 0) {
    perror(argv[3]);
    return 1;
  }
  return 0;
}
