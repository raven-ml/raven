/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Firmware: SHA-256, and the system libraries that decompress and download
   images, loaded when first needed. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#ifndef _WIN32
#include <dlfcn.h>
#endif

/* SHA-256 (FIPS 180-4) */

static const uint32_t sha_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha_block(uint32_t h[8], const uint8_t *p) {
  uint32_t w[64];
  for (int i = 0; i < 16; i++)
    w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 |
           (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3);
    uint32_t s1 = ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5],
           g = h[6], k = h[7];
  for (int i = 0; i < 64; i++) {
    uint32_t t1 = k + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) +
                  ((e & f) ^ (~e & g)) + sha_k[i] + w[i];
    uint32_t t2 =
        (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    k = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h[0] += a;
  h[1] += b;
  h[2] += c;
  h[3] += d;
  h[4] += e;
  h[5] += f;
  h[6] += g;
  h[7] += k;
}

value caml_device_pci_sha256(value s) {
  CAMLparam1(s);
  CAMLlocal1(r);
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  size_t n = caml_string_length(s), i = 0;
  const uint8_t *p = (const uint8_t *)String_val(s);
  for (; i + 64 <= n; i += 64) sha_block(h, p + i);
  uint8_t tail[128] = {0};
  size_t rest = n - i;
  memcpy(tail, p + i, rest);
  tail[rest] = 0x80;
  size_t len = rest + 9 <= 64 ? 64 : 128;
  uint64_t bits = (uint64_t)n * 8;
  for (int j = 0; j < 8; j++) tail[len - 1 - j] = (uint8_t)(bits >> (8 * j));
  sha_block(h, tail);
  if (len == 128) sha_block(h, tail + 64);
  r = caml_alloc_string(32);
  for (int j = 0; j < 8; j++)
    for (int b = 0; b < 4; b++)
      Bytes_val(r)[4 * j + b] = (uint8_t)(h[j] >> (24 - 8 * b));
  CAMLreturn(r);
}

/* Libraries the system may have, loaded when first needed. */

#ifndef _WIN32
static void *load_library(const char *const *names) {
  for (; *names; names++) {
    void *h = dlopen(*names, RTLD_NOW | RTLD_LOCAL);
    if (h) return h;
  }
  return NULL;
}
#endif

/* Decompression: zstd frames and xz streams, as distributions ship firmware.
   [None] if the library is missing; [Failure] if the data is corrupt or
   decompresses past MAX_IMAGE. */

/* The largest image accepted; firmware images are under 64 MiB. */
#define MAX_IMAGE (1ULL << 30)

/* lzma_ret values (lzma/base.h). */
enum { LZMA_OK = 0, LZMA_STREAM_END = 1, LZMA_BUF_ERROR = 10 };

typedef unsigned long long (*zstd_size_t)(const void *, size_t);
typedef size_t (*zstd_decompress_t)(void *, size_t, const void *, size_t);
typedef unsigned (*zstd_is_error_t)(size_t);

value caml_device_pci_unzstd(value s) {
  CAMLparam1(s);
  CAMLlocal1(out);
#ifndef _WIN32
  static const char *const names[] = {"libzstd.so.1", "libzstd.so",
                                      "libzstd.1.dylib", "libzstd.dylib",
                                      NULL};
  void *h = load_library(names);
  if (!h) CAMLreturn(Val_none);
  zstd_size_t size = (zstd_size_t)dlsym(h, "ZSTD_getFrameContentSize");
  zstd_decompress_t dec = (zstd_decompress_t)dlsym(h, "ZSTD_decompress");
  zstd_is_error_t is_error = (zstd_is_error_t)dlsym(h, "ZSTD_isError");
  if (!size || !dec || !is_error) CAMLreturn(Val_none);
  unsigned long long n = size(String_val(s), caml_string_length(s));
  if (n > MAX_IMAGE)
    caml_failwith("a zstd frame whose size is unknown or over 1 GiB");
  out = caml_alloc_string(n);
  size_t r = dec(Bytes_val(out), n, String_val(s), caml_string_length(s));
  if (is_error(r) || r != n) caml_failwith("a corrupt zstd frame");
  CAMLreturn(caml_alloc_some(out));
#else
  (void)s;
  CAMLreturn(Val_none);
#endif
}

typedef int (*lzma_decode_t)(uint64_t *, uint32_t, const void *,
                             const uint8_t *, size_t *, size_t, uint8_t *,
                             size_t *, size_t);

value caml_device_pci_unxz(value s) {
  CAMLparam1(s);
  CAMLlocal1(out);
#ifndef _WIN32
  static const char *const names[] = {"liblzma.so.5", "liblzma.so",
                                      "liblzma.5.dylib", "liblzma.dylib",
                                      NULL};
  void *h = load_library(names);
  if (!h) CAMLreturn(Val_none);
  lzma_decode_t dec = (lzma_decode_t)dlsym(h, "lzma_stream_buffer_decode");
  if (!dec) CAMLreturn(Val_none);
  size_t cap = caml_string_length(s) * 4 + 4096;
  for (;;) {
    uint8_t *buf = malloc(cap);
    if (!buf) caml_raise_out_of_memory();
    uint64_t limit = UINT64_MAX;
    size_t in_pos = 0, out_pos = 0;
    int r = dec(&limit, 0, NULL, (const uint8_t *)String_val(s), &in_pos,
                caml_string_length(s), buf, &out_pos, cap);
    if (r == LZMA_OK || r == LZMA_STREAM_END) {
      out = caml_alloc_initialized_string(out_pos, (const char *)buf);
      free(buf);
      CAMLreturn(caml_alloc_some(out));
    }
    free(buf);
    if (r != LZMA_BUF_ERROR) caml_failwith("a corrupt xz stream");
    if (cap >= MAX_IMAGE) caml_failwith("an xz stream over 1 GiB");
    cap = cap * 2 < MAX_IMAGE ? cap * 2 : MAX_IMAGE;
  }
#else
  (void)s;
  CAMLreturn(Val_none);
#endif
}

/* HTTPS downloads through libcurl, with the runtime released. [Error why] if
   the library is missing or the transfer fails. */

/* Option and info codes (curl/curl.h). */
#define CURLOPT_WRITEDATA 10001
#define CURLOPT_URL 10002
#define CURLOPT_LOW_SPEED_LIMIT 19
#define CURLOPT_LOW_SPEED_TIME 20
#define CURLOPT_FAILONERROR 45
#define CURLOPT_FOLLOWLOCATION 52
#define CURLOPT_CONNECTTIMEOUT 78
#define CURLOPT_WRITEFUNCTION 20011
#define CURLINFO_RESPONSE_CODE 0x200002

typedef void *(*curl_init_t)(void);
typedef int (*curl_setopt_t)(void *, int, ...);
typedef int (*curl_perform_t)(void *);
typedef int (*curl_getinfo_t)(void *, int, ...);
typedef void (*curl_cleanup_t)(void *);
typedef const char *(*curl_strerror_t)(int);

struct sink {
  char *data;
  size_t len, cap;
};

static size_t sink_write(char *p, size_t size, size_t n, void *userdata) {
  struct sink *s = userdata;
  size_t k = size * n;
  if (s->len + k > s->cap) {
    size_t cap = s->cap ? s->cap : 1 << 20;
    while (cap < s->len + k) cap *= 2;
    char *d = realloc(s->data, cap);
    if (!d) return 0;
    s->data = d;
    s->cap = cap;
  }
  memcpy(s->data + s->len, p, k);
  s->len += k;
  return k;
}

static value result(int ok, value v) {
  CAMLparam1(v);
  CAMLlocal1(r);
  r = caml_alloc_small(1, ok ? 0 : 1);
  Field(r, 0) = v;
  CAMLreturn(r);
}

value caml_device_pci_download(value url) {
  CAMLparam1(url);
  CAMLlocal1(v);
#ifndef _WIN32
  static const char *const names[] = {"libcurl.so.4", "libcurl.so",
                                      "libcurl-gnutls.so.4", "libcurl.4.dylib",
                                      "libcurl.dylib", NULL};
  void *h = load_library(names);
  if (!h)
    CAMLreturn(result(0, caml_copy_string("libcurl is not installed")));
  curl_init_t init = (curl_init_t)dlsym(h, "curl_easy_init");
  curl_setopt_t setopt = (curl_setopt_t)dlsym(h, "curl_easy_setopt");
  curl_perform_t perform = (curl_perform_t)dlsym(h, "curl_easy_perform");
  curl_getinfo_t getinfo = (curl_getinfo_t)dlsym(h, "curl_easy_getinfo");
  curl_cleanup_t cleanup = (curl_cleanup_t)dlsym(h, "curl_easy_cleanup");
  curl_strerror_t strerr = (curl_strerror_t)dlsym(h, "curl_easy_strerror");
  if (!init || !setopt || !perform || !getinfo || !cleanup || !strerr)
    CAMLreturn(result(0, caml_copy_string("libcurl lacks the easy interface")));
  char *u = caml_stat_strdup(String_val(url));
  struct sink s = {NULL, 0, 0};
  long status = 0;
  int rc;
  caml_release_runtime_system();
  void *c = init();
  if (!c) {
    rc = -1;
  } else {
    setopt(c, CURLOPT_URL, u);
    setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    setopt(c, CURLOPT_WRITEFUNCTION, sink_write);
    setopt(c, CURLOPT_WRITEDATA, &s);
    setopt(c, CURLOPT_FAILONERROR, 1L);
    setopt(c, CURLOPT_CONNECTTIMEOUT, 30L);
    setopt(c, CURLOPT_LOW_SPEED_LIMIT, 1L);
    setopt(c, CURLOPT_LOW_SPEED_TIME, 60L);
    rc = perform(c);
    getinfo(c, CURLINFO_RESPONSE_CODE, &status);
    cleanup(c);
  }
  caml_acquire_runtime_system();
  caml_stat_free(u);
  if (rc != 0) {
    free(s.data);
    char msg[512];
    snprintf(msg, sizeof msg, "%s (HTTP status %ld)",
             rc < 0 ? "libcurl failed to start" : strerr(rc), status);
    CAMLreturn(result(0, caml_copy_string(msg)));
  }
  v = caml_alloc_initialized_string(s.len, s.data ? s.data : "");
  free(s.data);
  CAMLreturn(result(1, v));
#else
  (void)url;
  CAMLreturn(result(0, caml_copy_string("downloads need a POSIX system")));
#endif
}
