#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

/* The queues host programs submitted, not yet taken by the device. */

typedef struct stream { char *words; uint64_t size; struct stream *next; } stream;

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static stream *first = NULL, *last = NULL;
static long outstanding = 0; /* submitted, and not yet run to their end */

/* Called by a batch's host program: [addr] holds [size] bytes of commands.
   [head] is the commands' first byte, which orders the call after the host
   program's writes into them. */
void tolk_null_submit(uint64_t addr, uint64_t size, uint64_t head) {
  (void)head;
  stream *s = malloc(sizeof(stream));
  s->words = malloc(size);
  memcpy(s->words, (const void *)(uintptr_t)addr, size);
  s->size = size;
  s->next = NULL;
  pthread_mutex_lock(&lock);
  if (last) last->next = s; else first = s;
  last = s;
  outstanding++;
  pthread_mutex_unlock(&lock);
}

value tolk_null_submit_address(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)(uintptr_t)&tolk_null_submit);
}

/* The submitted command buffers, oldest first, as strings. */
value tolk_null_take(value unit) {
  CAMLparam1(unit);
  CAMLlocal3(list, cell, words);
  pthread_mutex_lock(&lock);
  stream *s = first;
  first = last = NULL;
  pthread_mutex_unlock(&lock);
  stream *rev = NULL;
  while (s) { stream *n = s->next; s->next = rev; rev = s; s = n; }
  list = Val_emptylist;
  while (rev) {
    stream *n = rev->next;
    words = caml_alloc_initialized_string(rev->size, rev->words);
    cell = caml_alloc_small(2, 0);
    Field(cell, 0) = words;
    Field(cell, 1) = list;
    list = cell;
    free(rev->words);
    free(rev);
    rev = n;
  }
  CAMLreturn(list);
}

value tolk_null_outstanding(value unit) {
  (void)unit;
  pthread_mutex_lock(&lock);
  long n = outstanding;
  pthread_mutex_unlock(&lock);
  return Val_long(n);
}

value tolk_null_finished(value n) {
  pthread_mutex_lock(&lock);
  outstanding -= Long_val(n);
  pthread_mutex_unlock(&lock);
  return Val_unit;
}

/* Memory, by the address the device's work uses: the host's. */

value tolk_null_load(value addr) {
  return Val_long(*(volatile int64_t *)(uintptr_t)Nativeint_val(addr));
}

value tolk_null_store(value addr, value v) {
  __atomic_store_n((int64_t *)(uintptr_t)Nativeint_val(addr), (int64_t)Long_val(v), __ATOMIC_SEQ_CST);
  return Val_unit;
}

value tolk_null_copy(value dst, value src, value n) {
  memmove((void *)(uintptr_t)Nativeint_val(dst), (const void *)(uintptr_t)Nativeint_val(src), Long_val(n));
  return Val_unit;
}

/* Calls a host program's entry, void f(void **buffers, const int64_t *values). */
value tolk_null_call(value fn, value buffers, value values) {
  CAMLparam3(fn, buffers, values);
  mlsize_t nb = Wosize_val(buffers), nv = Wosize_val(values);
  void **b = malloc((nb + 1) * sizeof(void *));
  int64_t *v = malloc((nv + 1) * sizeof(int64_t));
  for (mlsize_t i = 0; i < nb; i++) b[i] = (void *)(uintptr_t)Long_val(Field(buffers, i));
  for (mlsize_t i = 0; i < nv; i++) v[i] = Long_val(Field(values, i));
  ((void (*)(void **, const int64_t *))(uintptr_t)Nativeint_val(fn))(b, v);
  free(b);
  free(v);
  CAMLreturn(Val_unit);
}

#include <dlfcn.h>
#include <caml/fail.h>

/* The address of the C function [name] among the process's libraries. */
value tolk_null_dlsym(value name) {
  CAMLparam1(name);
  void *f = dlsym(RTLD_DEFAULT, String_val(name));
  if (f == NULL) caml_invalid_argument(String_val(name));
  CAMLreturn(caml_copy_nativeint((intnat)(uintptr_t)f));
}
