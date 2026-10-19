/* The vendor toolkit's probe for run.py: native SCENARIO [ARG...], through
   CUDA's driver API (-DCUDA), HIP (-D__HIP_PLATFORM_AMD__) or Metal
   (Objective-C on macOS). Each scenario prints one line of KEY=VALUE fields;
   compare/README.md says what each one measures. */

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* The floats an addition adds, and the micro rows' samples. */
enum { COUNT = 64, WARMUPS = 3, SAMPLES = 5, IDLE_US = 20000 };

/* The pcie row's copy. */
#define PCIE_BYTES ((size_t)256 << 20)

static void fail(const char *what, long error) {
  fprintf(stderr, "%s: error %ld\n", what, error);
  exit(1);
}

#define CHECK(call) do { long error = (long)(call); if (error) fail(#call, error); } while (0)

/* The device: open, allocate, copy, add, wait. Copies complete before they
   return; additions are queued in order, and [gpu_sync] waits for them. */

#ifdef __APPLE__
#import <Metal/Metal.h>

typedef id<MTLBuffer> buffer;
typedef id<MTLComputePipelineState> kernel;
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLCommandBuffer> last;

static void gpu_open(void) {
  device = MTLCreateSystemDefaultDevice();
  if (!device) fail("MTLCreateSystemDefaultDevice", 1);
  queue = [device newCommandQueue];
}

static buffer gpu_alloc(size_t n) {
  buffer b = [device newBufferWithLength:n options:MTLResourceStorageModeShared];
  if (!b) fail("newBufferWithLength", 1);
  return b;
}

static void gpu_put(buffer b, const void *src, size_t n) { memcpy(b.contents, src, n); }

static void gpu_get(void *dst, buffer b, size_t at, size_t n) {
  memcpy(dst, (char *)b.contents + at, n);
}

static void gpu_free(buffer b) { [b release]; }

static kernel gpu_kernel(const char *path) {
  NSError *error = nil;
  NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
  id<MTLFunction> add = [library newFunctionWithName:@"add"];
  kernel k = add ? [device newComputePipelineStateWithFunction:add error:&error] : nil;
  if (!k) {
    fprintf(stderr, "%s: %s\n", path, error.description.UTF8String);
    exit(1);
  }
  return k;
}

/* The kernel reads the three arrays' GPU addresses, then their length, from
   buffer 0. */
static void gpu_add(kernel k, buffer a, buffer b, buffer out) {
  uint64_t args[] = { a.gpuAddress, b.gpuAddress, out.gpuAddress, COUNT };
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:k];
  [encoder setBytes:args length:sizeof args atIndex:0];
  [encoder useResource:a usage:MTLResourceUsageRead];
  [encoder useResource:b usage:MTLResourceUsageRead];
  [encoder useResource:out usage:MTLResourceUsageWrite];
  [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(COUNT, 1, 1)];
  [encoder endEncoding];
  [command commit];
  last = command;
}

static void gpu_sync(void) {
  if (!last) return;
  [last waitUntilCompleted];
  if (last.status != MTLCommandBufferStatusCompleted) fail("command buffer", (long)last.status);
  last = nil;
}

static uint64_t now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

/* The vendor rows' calls. A release is an empty command buffer committed
   and waited for, as Metal waits; Metal's queues wait on no word of
   memory, and its memory is unified, so it has no in-queue waits and no
   copies across a bus. */
#define HAS_BUS 0

static int gpu_can_wait(void) { return 0; }

static void gpu_release(int n) {
  id<MTLCommandBuffer> command = nil;
  for (int i = 0; i < n; ++i) {
    command = [queue commandBuffer];
    [command commit];
  }
  [command waitUntilCompleted];
}

static void gpu_waits(int n) { (void)n; }

static void gpu_copy(buffer dst, buffer src, size_t n) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
  [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:n];
  [blit endEncoding];
  [command commit];
  [command waitUntilCompleted];
}

static void gpu_d2h(void *dst, buffer src, size_t n) { (void)dst, (void)src, (void)n; }

static void gpu_map_host(void *p, size_t n) {
  buffer b = [device newBufferWithBytesNoCopy:p length:n options:MTLResourceStorageModeShared
                                  deallocator:nil];
  if (!b) fail("newBufferWithBytesNoCopy", 1);
  [b release];
}

#else

#ifdef CUDA
#include <cuda.h>
typedef CUdeviceptr buffer;
typedef CUfunction kernel;
#define ALLOC(p, n) cuMemAlloc(p, n)
#define H2D(p, s, n) cuMemcpyHtoD(p, s, n)
#define D2H(d, p, n) cuMemcpyDtoH(d, p, n)
#define SYNC() cuCtxSynchronize()
#define FREE(p) cuMemFree(p)
#define HOST_ALLOC(p, n) cuMemAllocHost(p, n)
#define HOST_FREE(p) cuMemFreeHost(p)
#define LAUNCH(f, a) cuLaunchKernel(f, 1, 1, 1, COUNT, 1, 1, 0, NULL, a, NULL)
#define AT(b, at) ((b) + (at))
#define D2D(d, s, n) cuMemcpyDtoD(d, s, n)
#define WORD_DEVICE(d, h) cuMemHostGetDevicePointer(d, h, 0)
#define WRITE64(d, v) cuStreamWriteValue64(NULL, d, v, 0)
#define WAIT64(d, v) cuStreamWaitValue64(NULL, d, v, CU_STREAM_WAIT_VALUE_GEQ)
#define REGISTER(p, n) cuMemHostRegister(p, n, 0)
#define UNREGISTER(p) cuMemHostUnregister(p)
typedef CUdeviceptr word_address;

/* Whether the stream waits on a 64-bit word, as the device states. */
static int gpu_can_wait(void) {
  CUdevice d;
  int can;
  CHECK(cuDeviceGet(&d, 0));
  CHECK(cuDeviceGetAttribute(&can, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, d));
  return can;
}

static void gpu_open(void) {
  CUcontext context;
  CHECK(cuInit(0));
  CHECK(cuDevicePrimaryCtxRetain(&context, 0));
  CHECK(cuCtxSetCurrent(context));
}

static kernel gpu_kernel(const char *path) {
  CUmodule module;
  CUfunction f;
  CHECK(cuModuleLoad(&module, path));
  CHECK(cuModuleGetFunction(&f, module, "add"));
  return f;
}
#else
#include <hip/hip_runtime_api.h>
typedef void *buffer;
typedef hipFunction_t kernel;
#define ALLOC(p, n) hipMalloc(p, n)
#define H2D(p, s, n) hipMemcpy(p, s, n, hipMemcpyHostToDevice)
#define D2H(d, p, n) hipMemcpy(d, p, n, hipMemcpyDeviceToHost)
#define SYNC() hipDeviceSynchronize()
#define FREE(p) hipFree(p)
#define HOST_ALLOC(p, n) hipHostMalloc(p, n, 0)
#define HOST_FREE(p) hipHostFree(p)
#define LAUNCH(f, a) hipModuleLaunchKernel(f, 1, 1, 1, COUNT, 1, 1, 0, NULL, a, NULL)
#define AT(b, at) ((void *)((char *)(b) + (at)))
#define D2D(d, s, n) hipMemcpy(d, s, n, hipMemcpyDeviceToDevice)
#define WORD_DEVICE(d, h) hipHostGetDevicePointer(d, h, 0)
#define WRITE64(d, v) hipStreamWriteValue64(NULL, d, v, 0)
#define WAIT64(d, v) hipStreamWaitValue64(NULL, d, v, hipStreamWaitValueGte, ~(uint64_t)0)
#define REGISTER(p, n) hipHostRegister(p, n, 0)
#define UNREGISTER(p) hipHostUnregister(p)
typedef void *word_address;

/* Whether the stream waits on a word, as the device states. */
static int gpu_can_wait(void) {
  int can;
  CHECK(hipDeviceGetAttribute(&can, hipDeviceAttributeCanUseStreamWaitValue, 0));
  return can;
}

static void gpu_open(void) {
  CHECK(hipInit(0));
  CHECK(hipSetDevice(0));
}

static kernel gpu_kernel(const char *path) {
  hipModule_t module;
  hipFunction_t f;
  CHECK(hipModuleLoad(&module, path));
  CHECK(hipModuleGetFunction(&f, module, "add"));
  return f;
}
#endif

static buffer gpu_alloc(size_t n) {
  buffer b;
  CHECK(ALLOC(&b, n));
  return b;
}

static void gpu_put(buffer b, const void *src, size_t n) { CHECK(H2D(b, src, n)); }

static void gpu_get(void *dst, buffer b, size_t at, size_t n) {
  CHECK(D2H(dst, AT(b, at), n));
}

static void gpu_free(buffer b) { CHECK(FREE(b)); }

/* The kernel takes the three addresses, then the arrays' length. */
static void gpu_add(kernel k, buffer a, buffer b, buffer out) {
  uint64_t count = COUNT;
  void *args[] = { &a, &b, &out, &count };
  CHECK(LAUNCH(k, args));
}

static void gpu_sync(void) { CHECK(SYNC()); }

static uint64_t now(void) {
  struct timespec t;
  if (clock_gettime(CLOCK_MONOTONIC, &t)) fail("clock_gettime", 1);
  return (uint64_t)t.tv_sec * 1000000000 + (uint64_t)t.tv_nsec;
}

/* The vendor rows' calls. A release is a 64-bit write of the next value
   into a word of page-locked host memory, made by the default stream after
   its earlier work, and a spin until the host reads it; a wait is a 64-bit
   wait of the stream for a value the word already holds. */
#define HAS_BUS 1

static volatile uint64_t *word;
static word_address word_gpu;
static uint64_t released;

static void words(void) {
  if (word) return;
  void *host;
  CHECK(HOST_ALLOC(&host, sizeof *word));
  word = host;
  *word = 0;
  CHECK(WORD_DEVICE(&word_gpu, host));
}

static void gpu_release(int n) {
  words();
  for (int i = 0; i < n; ++i) CHECK(WRITE64(word_gpu, ++released));
  while (*word < released) {
  }
}

static void gpu_waits(int n) {
  words();
  for (int i = 0; i < n; ++i) CHECK(WAIT64(word_gpu, released));
  gpu_release(1);
}

static void gpu_copy(buffer dst, buffer src, size_t n) {
  CHECK(D2D(dst, src, n));
  gpu_sync();
}

static void gpu_d2h(void *dst, buffer src, size_t n) {
  CHECK(D2H(dst, src, n));
  gpu_sync();
}

static void gpu_map_host(void *p, size_t n) {
  CHECK(REGISTER(p, n));
  CHECK(UNREGISTER(p));
}

#endif

static buffer floats(float value) {
  float values[COUNT];
  for (int i = 0; i < COUNT; ++i) values[i] = value;
  buffer b = gpu_alloc(sizeof values);
  gpu_put(b, values, sizeof values);
  return b;
}

static void check(buffer out, float expected) {
  float values[COUNT];
  gpu_get(values, out, 0, sizeof values);
  for (int i = 0; i < COUNT; ++i)
    if (values[i] != expected) {
      fprintf(stderr, "output %d: %g, expected %g\n", i, values[i], expected);
      exit(1);
    }
}

/* Scenarios */

/* The first kernel: one addition, checked after it completed; the time it
   completed. */
static uint64_t first(const char *image) {
  kernel k = gpu_kernel(image);
  buffer a = floats(41), b = floats(1), out = floats(NAN);
  gpu_add(k, a, b, out);
  gpu_sync();
  uint64_t completed = now();
  check(out, 42);
  return completed;
}

static void start(const char *image) {
  printf("completed_ns=%llu\n", (unsigned long long)first(image));
}

static void fit(const char *image) {
  first(image);
  printf("pid=%ld\n", (long)getpid());
  fflush(stdout);
  if (getchar() != '\n') fail("fit: no newline on standard input", 1);
}

/* The timer covers opening and mapping each file, allocating its device buffer
   and the completed copy; the check reads every byte back after it. */
static void load(int n, char **paths) {
  struct file { size_t size; unsigned char *map; buffer gpu; } *files = calloc((size_t)n, sizeof *files);
  if (!files) fail("calloc", 1);
  size_t bytes = 0;
  uint64_t started = now();
  for (int i = 0; i < n; ++i) {
    struct stat st;
    int fd = open(paths[i], O_RDONLY);
    if (fd < 0 || fstat(fd, &st) || st.st_size <= 0) { perror(paths[i]); exit(1); }
    files[i].size = (size_t)st.st_size;
    files[i].map = mmap(NULL, files[i].size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (files[i].map == MAP_FAILED) { perror("mmap"); exit(1); }
    close(fd);
    files[i].gpu = gpu_alloc(files[i].size);
    gpu_put(files[i].gpu, files[i].map, files[i].size);
    bytes += files[i].size;
  }
  gpu_sync();
  uint64_t elapsed = now() - started;
  const size_t chunk = 4 << 20;
  unsigned char *actual = malloc(chunk);
  if (!actual) fail("malloc", 1);
  for (int i = 0; i < n; ++i) {
    for (size_t at = 0; at < files[i].size; at += chunk) {
      size_t length = files[i].size - at < chunk ? files[i].size - at : chunk;
      gpu_get(actual, files[i].gpu, at, length);
      if (memcmp(actual, files[i].map + at, length)) {
        fprintf(stderr, "%s differs in the chunk at %zu\n", paths[i], at);
        exit(1);
      }
    }
    gpu_free(files[i].gpu);
    munmap(files[i].map, files[i].size);
  }
  free(actual);
  free(files);
  printf("bytes=%zu load_ns=%llu\n", bytes, (unsigned long long)elapsed);
}

/* [calls] dependent additions, alternating between two arrays, and a wait. */
static void add(int calls, const char *image) {
  kernel k = gpu_kernel(image);
  buffer a = floats(0), b = floats(1), out = floats(0);
  float values[COUNT];
  uint64_t times[SAMPLES];
  for (int s = 0; s < WARMUPS + SAMPLES; ++s) {
    for (int i = 0; i < COUNT; ++i) values[i] = 41;
    gpu_put(a, values, sizeof values);
    for (int i = 0; i < COUNT; ++i) values[i] = NAN;
    gpu_put(out, values, sizeof values);
    usleep(IDLE_US);
    uint64_t started = now();
#ifdef __APPLE__
    @autoreleasepool {
#endif
    for (int i = 0; i < calls; ++i) {
      if (i % 2 == 0) gpu_add(k, a, b, out);
      else gpu_add(k, out, b, a);
    }
    gpu_sync();
#ifdef __APPLE__
    }
#endif
    uint64_t elapsed = now() - started;
    check(calls % 2 == 0 ? a : out, (float)(41 + calls));
    if (s >= WARMUPS) times[s - WARMUPS] = elapsed;
  }
  printf("samples_ns=");
  for (int s = 0; s < SAMPLES; ++s) printf(s ? ",%llu" : "%llu", (unsigned long long)times[s]);
  printf("\n");
}

/* One completed pinned host-to-device copy after a warm one; every source byte
   changes in between, so a copy that did not happen cannot pass the check. */
static void pcie(void) {
#ifdef __APPLE__
  fail("pcie: unified memory has no host-to-device transfer", 1);
#else
  buffer gpu = gpu_alloc(PCIE_BYTES);
  void *pinned;
  CHECK(HOST_ALLOC(&pinned, PCIE_BYTES));
  unsigned char *expected = pinned, *actual = malloc(PCIE_BYTES);
  if (!actual) fail("malloc", 1);
  for (size_t i = 0; i < PCIE_BYTES; ++i) expected[i] = (unsigned char)(i % 251);
  gpu_put(gpu, pinned, PCIE_BYTES);
  for (size_t i = 0; i < PCIE_BYTES; ++i) expected[i] ^= 255;
  usleep(IDLE_US);
  uint64_t started = now();
  gpu_put(gpu, pinned, PCIE_BYTES);
  gpu_sync();
  uint64_t elapsed = now() - started;
  gpu_get(actual, gpu, 0, PCIE_BYTES);
  if (memcmp(actual, expected, PCIE_BYTES)) fail("pcie: copy differs", 1);
  printf("bytes=%zu copy_ns=%llu\n", PCIE_BYTES, (unsigned long long)elapsed);
  free(actual);
  CHECK(HOST_FREE(pinned));
  gpu_free(gpu);
#endif
}

/* Vendor rows: what one of rig's driver rows does, made through the
   vendor's interface alone. Each prints the samples of [SAMPLES] runs after
   [WARMUPS] untimed ones, each after an idle of IDLE_US. */

static void samples(void (*run)(void *), void *arg) {
  uint64_t times[SAMPLES];
  for (int s = 0; s < WARMUPS + SAMPLES; ++s) {
    usleep(IDLE_US);
    uint64_t started = now();
    run(arg);
    uint64_t elapsed = now() - started;
    if (s >= WARMUPS) times[s - WARMUPS] = elapsed;
  }
  printf("samples_ns=");
  for (int s = 0; s < SAMPLES; ++s) printf(s ? ",%llu" : "%llu", (unsigned long long)times[s]);
  printf("\n");
}

static void run_release(void *n) { gpu_release(*(int *)n); }

/* [n] empty submissions, each released into a word, and a wait for the
   last. */
static void release(int n) { samples(run_release, &n); }

static void run_waits(void *n) { gpu_waits(*(int *)n); }

/* A submission of [n] waits for values already reached, then its release.
   A device whose queues wait on no word of memory, such as Metal's, prints
   [unsupported=1] instead, and run.py leaves the case out. */
static void waits(int n) {
  if (!gpu_can_wait()) {
    printf("unsupported=1\n");
    return;
  }
  samples(run_waits, &n);
}

struct copy { buffer dst, src; void *host; size_t n; };

static void run_d2d(void *c) {
  struct copy *x = c;
  gpu_copy(x->dst, x->src, x->n);
}

static void run_d2h(void *c) {
  struct copy *x = c;
  gpu_d2h(x->host, x->src, x->n);
}

static void run_h2d(void *c) {
  struct copy *x = c;
  gpu_put(x->dst, x->host, x->n);
  gpu_sync();
}

/* One completed copy of [n] bytes [how]: d2d within the device's memory,
   d2h into and h2d from page-locked host memory. The bytes are checked once
   after the samples. */
static void copy(const char *how, size_t n) {
  int within = !strcmp(how, "d2d"), to_host = !strcmp(how, "d2h");
  if (!within && !to_host && strcmp(how, "h2d")) fail("copy: d2d, d2h or h2d", 1);
  if (!within && !HAS_BUS) fail("copy: unified memory has no transfer across a bus", 1);
  struct copy c = { gpu_alloc(n), gpu_alloc(n), NULL, n };
  unsigned char *expected = malloc(n), *actual = malloc(n);
  if (!expected || !actual) fail("malloc", 1);
  for (size_t i = 0; i < n; ++i) expected[i] = (unsigned char)(i % 251);
  gpu_put(c.src, expected, n);
#ifndef __APPLE__
  CHECK(HOST_ALLOC(&c.host, n));
  if (to_host) memset(c.host, 0, n);
  else memcpy(c.host, expected, n);
#endif
  samples(within ? run_d2d : to_host ? run_d2h : run_h2d, &c);
  if (to_host) memcpy(actual, c.host, n);
  else gpu_get(actual, c.dst, 0, n);
  if (memcmp(actual, expected, n)) fail("copy: the bytes differ", 1);
#ifndef __APPLE__
  CHECK(HOST_FREE(c.host));
#endif
  free(expected);
  free(actual);
  gpu_free(c.dst);
  gpu_free(c.src);
}

static void run_alloc(void *n) { gpu_free(gpu_alloc(*(size_t *)n)); }

/* An allocation of [n] bytes of the device's memory and its free. */
static void alloc(size_t n) { samples(run_alloc, &n); }

struct host { void *p; size_t n; };

static void run_map_host(void *h) {
  struct host *x = h;
  gpu_map_host(x->p, x->n);
}

/* A mapping of [n] bytes of host memory, page-aligned and written once, for
   the device, and its unmapping. */
static void map_host(size_t n) {
  size_t page = (size_t)sysconf(_SC_PAGESIZE);
  struct host h = { NULL, (n + page - 1) / page * page };
  if (posix_memalign(&h.p, page, h.n)) fail("posix_memalign", 1);
  memset(h.p, 1, h.n);
  samples(run_map_host, &h);
  free(h.p);
}

int main(int argc, char **argv) {
  const char *usage = "usage: native start IMAGE | fit IMAGE | load FILE... | add CALLS IMAGE | pcie\n"
                      "       | release N | waits N | copy d2d|d2h|h2d BYTES | alloc BYTES | map_host BYTES\n";
  if (argc < 2) { fputs(usage, stderr); return 2; }
  const char *scenario = argv[1];
#ifdef __APPLE__
  @autoreleasepool {
#endif
  gpu_open();
  if (!strcmp(scenario, "start") && argc == 3) start(argv[2]);
  else if (!strcmp(scenario, "fit") && argc == 3) fit(argv[2]);
  else if (!strcmp(scenario, "load") && argc >= 3) load(argc - 2, argv + 2);
  else if (!strcmp(scenario, "add") && argc == 4) add(atoi(argv[2]), argv[3]);
  else if (!strcmp(scenario, "pcie") && argc == 2) pcie();
  else if (!strcmp(scenario, "release") && argc == 3) release(atoi(argv[2]));
  else if (!strcmp(scenario, "waits") && argc == 3) waits(atoi(argv[2]));
  else if (!strcmp(scenario, "copy") && argc == 4) copy(argv[2], (size_t)atoll(argv[3]));
  else if (!strcmp(scenario, "alloc") && argc == 3) alloc((size_t)atoll(argv[2]));
  else if (!strcmp(scenario, "map_host") && argc == 3) map_host((size_t)atoll(argv[2]));
  else { fputs(usage, stderr); return 2; }
#ifdef __APPLE__
  }
#endif
  return 0;
}
