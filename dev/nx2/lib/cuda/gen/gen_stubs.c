/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* kernels.ml's text, from kernels.h as the device compiler reads it: the
   kernels' instances, the tiles, the aligned bits and each parameter
   struct's fields by offset. A struct's fields listed here must tile it
   from byte 0 to its size, so a field added to kernels.h and not listed
   here fails the build, as does a renamed one. */

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "kernels.h"

/* The bound on the kernels nx_cuda.ml's submission table holds. */
#define MAX_KERNELS 1024

static char *text;
static size_t len, cap;

static void out(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (len + n + 1 > cap) {
    while (len + n + 1 > cap) cap = cap ? 2 * cap : 4096;
    text = realloc(text, cap);
    if (text == NULL) caml_raise_out_of_memory();
  }
  va_start(ap, fmt);
  vsnprintf(text + len, n + 1, fmt, ap);
  va_end(ap);
  len += n;
}

/* [s] as an OCaml constructor: its first letter capitalised. */
static void constructor(const char *s) {
  out("%c%s", s[0] >= 'a' && s[0] <= 'z' ? s[0] - 'a' + 'A' : s[0], s + 1);
}

/* Instances */

/* A kernel: its name, its family and its arguments' tokens. */
typedef struct {
  const char *name, *family, *args[4];
  int nargs;
} kernel;

/* A row by its count of arguments: none, 1, 2 or 4. */
#define K0(name, family) {#name, #family, {0}, 0},
#define K1(name, family, a) {#name, #family, {#a}, 1},
#define K2(name, family, a, b) {#name, #family, {#a, #b}, 2},
#define K4(name, family, a, b, c, d) {#name, #family, {#a, #b, #c, #d}, 4},
#define PICK(_1, _2, _3, _4, _5, _6, k, ...) k
#define ROW(...) PICK(__VA_ARGS__, K4, K3, K2, K1, K0, _)(__VA_ARGS__)
static const kernel kernels[] = {NX_CUDA_KERNELS(ROW)};
#define COUNT(xs) (sizeof xs / sizeof xs[0])

/* The distinct tokens of a type, in order of first use. */
typedef struct {
  const char *name, *tokens[32];
  int n;
} type;

static void token(type *t, const char *s) {
  for (int i = 0; i < t->n; i++)
    if (strcmp(t->tokens[i], s) == 0) return;
  if (t->n == 32) caml_failwith("kernels.h: more than 32 tokens of one type");
  t->tokens[t->n++] = s;
}

static void type_decl(const type *t) {
  out("type %s =", t->name);
  for (int i = 0; i < t->n; i++) {
    out(" | ");
    constructor(t->tokens[i]);
  }
  out("\n");
}

/* Tiles */

typedef struct {
  const char *name;
  int bm, bn, bkb, wm, wn, stages;
} tile;

#define TILE(name, bm, bn, bkb, wm, wn, s) {#name, bm, bn, bkb, wm, wn, s},
static const tile tiles[] = {NX_CUDA_TILES(TILE)};

/* Structs */

typedef struct {
  const char *name;
  size_t offset, size;
} field;

#define F(s, f) {#f, offsetof(s, f), sizeof(((s *)0)->f)}

static const field contract_fields[] = {
    F(contract_params, a),          F(contract_params, b),
    F(contract_params, init),       F(contract_params, y),
    F(contract_params, partials),   F(contract_params, tickets),
    F(contract_params, sa),         F(contract_params, sb),
    F(contract_params, si),         F(contract_params, sy),
    F(contract_params, batch),      F(contract_params, m),
    F(contract_params, n),          F(contract_params, k),
    F(contract_params, splits),     F(contract_params, a_dtype),
    F(contract_params, b_dtype),    F(contract_params, init_dtype),
    F(contract_params, y_dtype),    F(contract_params, acc_dtype),
    F(contract_params, aligned),    F(contract_params, unused)};

static const field pack_fields[] = {
    F(pack_params, src),   F(pack_params, dst),  F(pack_params, s),
    F(pack_params, lead),  F(pack_params, batch), F(pack_params, rows),
    F(pack_params, k),     F(pack_params, dtype), F(pack_params, out),
    F(pack_params, bytes)};

static int by_offset(const void *x, const void *y) {
  const field *a = x, *b = y;
  return a->offset < b->offset ? -1 : a->offset > b->offset;
}

/* The module [m] of the struct [s] of [size] bytes: its size and each
   field's offset, once the fields are checked to tile it. */
static void structure(const char *m, const char *s, size_t size,
                      const field *fs, size_t n) {
  field sorted[32];
  char why[128];
  memcpy(sorted, fs, n * sizeof *fs);
  qsort(sorted, n, sizeof *sorted, by_offset);
  size_t at = 0;
  for (size_t i = 0; i <= n; i++) {
    const size_t next = i < n ? sorted[i].offset : size;
    if (next != at) {
      snprintf(why, sizeof why, "%s: bytes %zu-%zu %s", s,
               next > at ? at : next, next > at ? next : at,
               next > at ? "unlisted" : "listed twice");
      caml_failwith(why);
    }
    if (i < n) at += sorted[i].size;
  }
  out("\nmodule %s = struct\n  let size = %zu\n", m, size);
  for (size_t i = 0; i < n; i++)
    out("  let %s = %zu\n", fs[i].name, fs[i].offset);
  out("end\n");
}

CAMLprim value nx_cuda_gen_text(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(r);
  if (COUNT(kernels) > MAX_KERNELS) caml_failwith("kernels.h: past 1024 kernels");

  type kind = {.name = "kind"}, axis = {.name = "axis"},
       acc = {.name = "acc"}, tile_t = {.name = "tile"};
  for (size_t i = 0; i < COUNT(tiles); i++) token(&tile_t, tiles[i].name);
  for (size_t i = 0; i < COUNT(kernels); i++) {
    const kernel *k = &kernels[i];
    if (strcmp(k->family, "MMA") == 0) {
      token(&kind, k->args[0]), token(&axis, k->args[1]);
      token(&axis, k->args[2]), token(&tile_t, k->args[3]);
    } else if (strcmp(k->family, "SIMT") == 0 || strcmp(k->family, "SKINNY") == 0)
      token(&acc, k->args[0]);
  }

  out("(* Generated from kernels.h by gen/gen.exe: do not edit. *)\n\n");
  type_decl(&kind), type_decl(&axis), type_decl(&acc), type_decl(&tile_t);
  out("\ntype instance =\n");
  for (size_t i = 0; i < COUNT(kernels); i++) {
    const kernel *k = &kernels[i];
    if (k->nargs == 0) {
      out("  | ");
      constructor(k->name);
      out("\n");
    }
  }
  out("  | Mma of kind * axis * axis * tile\n  | Simt of acc * int\n"
      "  | Skinny of acc\n\n");

  out("let count = %zu\n\nlet instances =\n  [|\n", COUNT(kernels));
  for (size_t i = 0; i < COUNT(kernels); i++) {
    const kernel *k = &kernels[i];
    out("    ");
    if (k->nargs == 0) constructor(k->name);
    else {
      constructor(strcmp(k->family, "SKINNY") == 0 ? "skinny"
                  : strcmp(k->family, "SIMT") == 0 ? "simt"
                                                   : "mma");
      out(" (");
      for (int j = 0; j < k->nargs; j++) {
        if (j) out(", ");
        if (strcmp(k->family, "SIMT") == 0 && j == 1) out("%s", k->args[j]);
        else constructor(k->args[j]);
      }
      out(")");
    }
    out(";\n");
  }
  out("  |]\n\nlet names =\n  [|\n");
  for (size_t i = 0; i < COUNT(kernels); i++) out("    \"%s\";\n", kernels[i].name);
  out("  |]\n");

  out("\ntype shape = { bm : int; bn : int; bkb : int; wm : int; wn : int; "
      "stages : int }\n\nlet tiles = [|");
  for (size_t i = 0; i < COUNT(tiles); i++) {
    out(" ");
    constructor(tiles[i].name);
    out(";");
  }
  out(" |]\n\nlet shape = function\n");
  for (size_t i = 0; i < COUNT(tiles); i++) {
    const tile *t = &tiles[i];
    out("  | ");
    constructor(t->name);
    out(" -> { bm = %d; bn = %d; bkb = %d; wm = %d; wn = %d; stages = %d }\n",
        t->bm, t->bn, t->bkb, t->wm, t->wn, t->stages);
  }

  out("\nlet a_vectors = %d\nlet b_vectors = %d\nlet b_across = %d\n"
      "let y_whole = %d\nlet skinny_rows = %d\n",
      NX_CONTRACT_A_VECTORS, NX_CONTRACT_B_VECTORS, NX_CONTRACT_B_ACROSS,
      NX_CONTRACT_Y_WHOLE, NX_SKINNY_ROWS);

  structure("Contract_params", "contract_params", sizeof(contract_params),
            contract_fields, COUNT(contract_fields));
  structure("Pack_params", "pack_params", sizeof(pack_params), pack_fields,
            COUNT(pack_fields));

  r = caml_alloc_initialized_string(len, text);
  free(text);
  text = NULL, len = cap = 0;
  CAMLreturn(r);
}
