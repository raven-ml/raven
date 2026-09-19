# RFC 0003: Loading weights

- Status: committed
- Date: 2026-09-18
- Packages: nx (buffer, io, `cast`, effect), kaun (`Checkpoint`, `Kaun_hf`,
  examples), rune (`to_device`, capture binding, unaligned vector types on the
  CPU device)

## Summary

Loading a checkpoint reads its header and maps the file. A model's importer is
an ordinary function that asks for each entry by name, at the dtype it wants,
and builds the parameter record itself, so there is no template and leaves may
have different dtypes. A tensor over a mapped file is an ordinary value whose
pages the process does not own. `Rune.to_device` moves a tensor into a device
buffer and returns it as a resident value, and a compiled function that
captures a resident value binds its buffer without copying, so an importer
that places each leaf as it builds it leaves one committed copy of the model,
on the device. A file that tensors were read from does not change while they
are reachable, so every writer of a file raven maps replaces it atomically. By
the sizes of what the code allocates, a 13.8 GB checkpoint with 4-bit experts
stored as uint8 blocks loads on a 32 GB machine at a peak of 16.5 GB of
weights and staging before placement exists and at most 15.0 GB with it, where
today it cannot load at all. The design adds one buffer primitive, two
checkpoint functions and one rune function.

## Motivation

Measured, loading and importing Llama 3.2 1B peaks at 15.8 GB of host memory
for a 2.47 GB file. The shard is read into one OCaml string, its data section
is copied into a second, each tensor is copied out of that (uint8 element by
element through a closure), and `Checkpoint.to_params` then needs a fully
allocated template of the target dtype whose values it discards
(`nx_safetensors.ml:154-161`, `safetensors.ml:619-622`,
`checkpoint.ml:102-109`). `Kaun_hf.load_checkpoint` keeps every shard's
tensors until extraction ends. Llama 3.1 8B needs 48 GB before extraction
starts and does not load on a 32 GB machine.

gpt-oss-20b is the forcing case. Its checkpoint is 13.76 GB in three shards:
10.15 GB of uint8 blocks and scales that must stay uint8 and are dequantised
inside the compiled program, and 3.61 GB at bfloat16. A float32 template is 84
GB. `to_params` is over `Ptree.Uniform`, one leaf type, so a record with float
leaves beside uint8 leaves has no extraction path except existential leaves
unpacked at every use. On Metal a captured host tensor is copied to a device
buffer and stays alive on the host, 27.5 GB for this model.

Three smaller defects ride along. Entries whose dtype nx lacks (`F8_E8M0`
scales, `F4`) vanish with a warning on stderr. A 16-bit tensor at an odd
offset inside the data section raises. A save truncates its destination in
place, which is a crash the day loading becomes a mapping.

## Guide

### Importing a model

```ocaml
(* llama.ml *)
let of_hf ?device cfg dt ckpt =
  let place x = match device with None -> x | Some device -> Rune.to_device ~device x in
  let weight ~shape name = Checkpoint.to_float ~shape dt (name ^ ".weight") ckpt in
  let norm name = { Rms_norm.gamma = place (weight ~shape:[| cfg.dim |] name) } in
  (* The file stores a projection as [outputs; inputs]. *)
  let linear ~inputs ~outputs name =
    { Linear.w = place (Nx.matrix_transpose (weight ~shape:[| outputs; inputs |] name)); b = None }
  in
  let q_dim = cfg.n_heads * cfg.head_dim and kv_dim = cfg.n_kv_heads * cfg.head_dim in
  let block i =
    let at leaf = Printf.sprintf "model.layers.%d.%s" i leaf in
    {
      attn_norm = norm (at "input_layernorm");
      attn = { q = linear ~inputs:cfg.dim ~outputs:q_dim (at "self_attn.q_proj");
               (* k, v, out likewise *) ... };
      ffn_norm = norm (at "post_attention_layernorm");
      gate = linear ~inputs:cfg.dim ~outputs:cfg.hidden_dim (at "mlp.gate_proj");
      (* up, down likewise *) ...
    }
  in
  {
    tok = { Embedding.table = place (weight ~shape:[| cfg.vocab_size; cfg.dim |] "model.embed_tokens") };
    blocks = List.init cfg.n_layers block;
    norm = norm "model.norm";
    head = (if cfg.tied then None else Some (linear ~inputs:cfg.dim ~outputs:cfg.vocab_size "lm_head"));
  }

let from_pretrained ?device ?(repo_id = default_repo) dt =
  let cfg = config_of_json (Kaun_hf.load_config repo_id) in
  (cfg, of_hf ?device cfg dt (Kaun_hf.load_checkpoint repo_id))
```

The importer has the outline of `make`: the same `~inputs ~outputs`, a name in
the file where `make` has an initializer. Structure comes from the
configuration and the dtype is an argument. Each name is written at the field
it fills. A rename is a different string at the field. A transpose is
`Nx.matrix_transpose` and a fused tensor is `Nx.split`, both views. `~shape`
is checked against the file, so a configuration that disagrees with the
checkpoint fails at import with the entry's name. At the file's own dtype
nothing is copied for an aligned file: every leaf is a view of it. Asking
`to_float` for another float dtype casts that leaf, which allocates it.

`place` is let-bound, so it serves leaves of any dtype. Each leaf is read,
cast if asked, transposed and placed before the next is touched, so at most
one leaf's cast or contiguous copy is alive on the host.

### Float leaves beside uint8 leaves

```ocaml
type mxfp4 = { blocks : Nx.uint8_t; scales : Nx.uint8_t }
type 'a experts = { gate_up : mxfp4; gate_up_bias : 'a; down : mxfp4; down_bias : 'a }

let of_hf ?device cfg dt ckpt =   (* [place] as above *)
  let float ~shape name = place (Checkpoint.to_float ~shape dt name ckpt) in
  let bytes ~shape name = place (Checkpoint.to_tensor ~shape Nx.uint8 name ckpt) in
  let mxfp4 ~inputs ~outputs name =
    { blocks = bytes ~shape:[| cfg.experts; outputs; inputs / 32; 16 |] (name ^ "_blocks");
      scales = bytes ~shape:[| cfg.experts; outputs; inputs / 32 |] (name ^ "_scales") }
  in
  ...
```

A record mixes as many leaf types as it likes, because each field's type is
the dtype passed to the accessor. The uint8 leaves are never cast, and are
copied only into the upload's chunk: `to_tensor` is strict, and `to_float` on
a block entry raises. Inference never traverses this record, since the step
function captures it. When a traversal is wanted, it is a hand-written
`Ptree.S` whose `map` applies its polymorphic function to every field, as the
`Step` record of the Llama example already does.

### Running it

```ocaml
let device = "METAL"
let cfg, params = Gpt_oss.from_pretrained ~device Nx.bfloat16
let step = Rune.jit2 ~device ~donate:true (module Step) (module Step) (step cfg params)
```

This is RFC 0002's generate loop unchanged. Loading three shards reads three
headers, and the import copies each leaf from the file's pages into a device
buffer. Compiling `step` binds those buffers and uploads nothing; prefill,
decode and any other compiled function over `params` share them. The process
owns 13.76 GB of device buffers, and the file's pages are cache that the
kernel drops under pressure without writing anything.

Before stage 3, and after it without `~device`, the program is still correct:
captures are uploaded at the first call, one copy per compiled function, and a
leaf cast on the host stays alive beside its device copy (see Peak memory).
The examples' `--dtype` defaults to the file's dtype: a program that takes its
default from the file loads the configuration and the checkpoint, reads the
stored dtype of one entry, and calls `of_hf`, where `from_pretrained dt`
serves a caller that knows its dtype. Placing after the import, `Params.map
(Rune.to_device ~device) params`, is correct too and holds every cast leaf
until the map ends. Training places its state the same way, before the first
step, which removes the first call's double copy.

### Restarting training

```ocaml
let ckpt = Checkpoint.load path in
let params = Checkpoint.to_params (module Llama.Params) ~prefix:"model" ~like:state.params ckpt
```

Files raven wrote name their entries by the structure's paths, and a restart
holds a value of the structure, so `to_params ~like` stays. It is `to_tensor`
at each leaf's path, shape and dtype.

### The rule about files

A file must not change while a tensor read from it is reachable. Replace a
checkpoint by writing a new file and renaming it over the old one, which is
what `Checkpoint.save` does. `Nx.copy` gives a tensor that is independent of
its file, and the file becomes replaceable on every platform once the tensors
over it have been collected. A mapping lasts until the garbage collector
finalises the last tensor over it, which can be later than the last use.

## Reference

### `Nx_buffer.reinterpret`

```ocaml
val reinterpret : ('a, 'b) kind -> ('c, 'd) t -> ('a, 'b) t
```

`reinterpret kind b` is `b`'s memory read as elements of `kind`, without a
copy. Its length is `b`'s size in bytes divided by the element size of `kind`.
The two buffers alias, as the results of `Bigarray.Array1.sub` do, and over
external memory the owner stays the caller's concern; RFC 0001's contract
starts where a buffer is wrapped as a tensor. It raises `Invalid_argument` if
the size is not a multiple of the element size, if `b`'s address is not
aligned to it, or if either kind is `int4` or `uint4`.

It is required. bfloat16, float8, bool, uint32 and uint64 carry their kind in
bits of the bigarray flags that only allocating stubs set
(`nx_buffer_stubs.h:28-57`), so no existing memory can be viewed at those
kinds today, and bfloat16 is the weight dtype of every target model.

The stub never builds a header. It calls the runtime's exported `caml_ba_sub`
over the whole of `b`, which allocates the header, copies `b`'s custom
operations (a mapped file's finaliser unmaps, the stock one frees) and joins
`b`'s proxy under the runtime's atomic count, then rewrites the result's kind
bits and length. `caml_ba_sub` is exported by the runtime and absent from its
header, so the stub declares it, and the OCaml side passes the element size,
which the runtime keeps internal. The runtime's proxy function and its mapped
operations table are private, so this is the only correct route; nx's own
header builder marks its result managed, and a finaliser would then free an
address inside the mapping. A test maps a file, keeps only a reinterpreted
sub-array, collects, reads it, drops it, collects, and checks that the file
was unmapped once. The JavaScript stubs gain the same function, a typed array
of the new kind over the same `ArrayBuffer`. It is storage-level: no backend
operation is added.

### `Nx_buffer.file_range`

```ocaml
type file = { path : string; size : int; mtime : float; inode : int }
val register_file : file -> (int, uint8_elt) t -> unit
val file_range : ('a, 'b) t -> (file * int) option
```

`file_range buf` is the file `buf`'s memory is a mapping of and the byte
offset in it of `buf`'s first element, for any buffer whose memory lies inside
a recorded mapping. The answer is address arithmetic, so the results of
`Array1.sub` and `reinterpret` answer without carrying anything. It is `None`
for every other buffer, which includes the entries a load copied, and always
on JavaScript. `register_file file buf` records a mapping that `Unix.map_file`
just returned, before any view of it exists, with the file's identity as the
mapping descriptor's `fstat` gave it. The loader records its mappings. nx's
buffer library does not link unix, so the mapping call stays in nx io and the
buffer module only records it.

A record must die exactly when its mapping does, since a stale one would
describe whatever is mapped at that address next. The runtime unmaps a file in
the finaliser of the last bigarray over it, and every bigarray derived from a
mapped one inherits its custom operations. Registration replaces the root's
operations with a copy whose finaliser removes the record when it is about to
run the runtime's finaliser for the last time, that is when the array has no
proxy or the proxy's count is one. Finalisers of different views may run on
different domains, so that decision and the runtime's decrement are taken
under one lock. A test maps, records, drops and collects, then maps another
file, which the system places at the same address, and checks that it is not
taken for the first.

The consumer is rune's upload. A path may name another file by the time it is
read, so the file opened must have the recorded size, modification time and
inode, and an upload that finds another file, or a short read, copies from the
mapping instead.

### `Nx_io.load_safetensors`

```ocaml
val load_safetensors : string -> archive      (* unchanged signature *)
val save_safetensors : ?overwrite:bool -> string -> (string * packed) list -> unit
```

`load_safetensors path` opens one descriptor, requires a regular file, reads 8
bytes, requires the header length to be at most 100 MB (the existing limit)
and within the file, reads the header, runs the existing parser and validator,
requires `8 + header_len + data_len = file size`, maps the file once with
`Unix.map_file` as bytes (a mapping per tensor has the same alignment, since
the offset inside the page is the file's, and costs an `open` and an `mmap`
per entry), requires the mapped length to equal the validated one, and closes
the descriptor. Every failure is a `Failure` naming the path. A killed
download fails here with both lengths in the message; without the check the
first touch of a missing page is a bus error, which OCaml cannot turn into an
exception. A repeated name in a header is a format error.

`Unix.map_file` maps private and writable, so a stray write through `Nx.data`
lands in a private page and never reaches the file, and pages not yet touched
show later writes to the file: a private mapping is no snapshot, which is why
Law 1 exists. RFC 0001 already names the wrap as legal, read-only mapped pages
behind `of_bigarray`. Donation never consumes such a tensor, and on the CPU
device it is never a kernel output.

Each entry of the archive is a tensor.

| Entry | Tensor | Bytes read by the load |
| --- | --- | --- |
| a dtype nx has, at an address that is a multiple of its element size, little-endian host | a view of the mapping: `Array1.sub`, `reinterpret`, `Nx.of_buffer`. A `BOOL` byte other than 0 or 1 is handed out as stored | none |
| misaligned address, or a big-endian host | a fresh host tensor: the bytes are copied out of the mapping, and swapped, by the load | the entry |
| `F8_E8M0`, `F4`, `F6_E2M3`, `F6_E3M2` | a uint8 view of the entry's bytes. A one-byte dtype keeps the entry's shape, since elements and bytes coincide; a sub-byte dtype has no shape in bytes, so the view is `[| nbytes |]` | none |
| zero elements | an empty tensor of the entry's shape | none |

The archive holds what it hands out, as today, so an entry read twice is one
tensor and a compiled function uploads it once. For an aligned file that costs
nothing. For a misaligned file the load copies every entry, and the leaves an
importer keeps at the file's dtype are those copies. An importer that casts
every leaf holds the copies beside the casts until the checkpoint is
unreachable, which a reader value would avoid. Such files are published: the
GPT-2 checkpoint the tree's own example loads has a 14283-byte header, so none
of its 160 float32 tensors starts on a multiple of four, and it costs 0.55 GB
on the host beside the device copy until the leaves are placed and the
checkpoint dropped. `Checkpoint.save` writes an aligned file. The Llama and
gpt-oss shards are aligned.

A leading-axis slice of a mapped entry is one byte range of the file, read
when touched; on a copying device a slice with a non-zero offset is made
contiguous before upload. nx io has no function over byte offsets and no
notion of a sharded set: `Kaun_hf.load_checkpoint` keeps its code, since a
retained table of views is free.

`save_safetensors` writes a temporary file in the destination's directory,
calls `Unix.fsync` on it because a restart trusts the file's length, and
renames it, as the npy, image and gzip writers already do. If the rename is
refused it runs `Gc.full_major ()` and retries once; if it is refused again it
raises `Failure` naming the destination and the temporary file, which it
keeps, so a finished run is not lost. `Kaun_hf.clear_cache` collects and
retries the same way. `Kaun_hf.download_file` downloads to a uniquely named
temporary sibling of its destination and renames it; the loser of a race
renames an identical file over the winner's, or serves the winner's file where
the rename is refused. Saves are created with mode `0o640`, as the other
writers', and downloads with `0o644`. `Checkpoint.save` of a loaded checkpoint
reads every entry and, until a streaming writer exists, holds twice the
checkpoint's size. An entry whose dtype nx lacks is written back as `U8` with
the shape it was handed out at.

### `Checkpoint.to_tensor` and `Checkpoint.to_float`

```ocaml
val to_tensor : shape:int array -> ('a, 'b) Nx.dtype -> string -> t -> ('a, 'b) Nx.t
val to_float : shape:int array -> (float, 'b) Nx.dtype -> string -> t -> (float, 'b) Nx.t
```

`to_tensor ~shape dtype name t` is `name`'s entry, which must have `dtype` and
`shape`. It is returned as stored, so an entry of a loaded file stays a view
of the file. `to_float ~shape dtype name t` is a float16, bfloat16, float32 or
float64 entry at `dtype`: as stored when it already has `dtype`, and otherwise
cast with `Nx.cast`, which allocates the leaf. A float8 entry is refused,
since its scales live in other entries; `to_tensor` reads it. A float8 `dtype`
is refused for the same reason. Both raise `Invalid_argument`, naming the
entry, if `name` has no entry or `shape` differs; `to_tensor` on any dtype
mismatch, `to_float` on an entry that is not floating-point. An importer that
ties weights binds the tensor once and uses it twice.

`?cast` disappears from `to_params` and `to_packed`, and they raise on any
dtype mismatch: a template states the dtype it expects, and a cast is asked
for by name, through `to_float`. `Nx.cast` at the tensor's own dtype returns
the tensor, where today it copies. `Kaun_hf.rename`, `transpose` and `split`
are removed. They existed because `to_params` finds entries by the template's
paths; with extraction by function they have no caller, and they cannot
express a dtype per leaf.

### `Rune.to_device`

```ocaml
val to_device : ?device:string -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
```

`to_device x` is `x` with its bytes held by `device`, resolved as in `jit`.
Where a tensor lives is a run-time attribute, so `to_device` has the
identity's type, and its result equals `x` in shape, dtype and value. The
bytes are copied 64 MiB at a time into one device buffer, allocated outside
the allocator's cache, and the result is resident like an unread output of a
compiled call. Metadata reads are free, a compiled function on `device` that
takes it as an input leaf uses the buffer with no transfer, `~donate:true`
consumes it when it is an input leaf, and the first host read of its data
copies it back and releases the buffer. Every nx operation outside a compiled
function is a host read, views included. `x` is untouched and may be dropped.
A contiguous source, offset or not, is read in place chunk by chunk. A strided
source is cut along its leading axis into pieces of at most a chunk, each made
contiguous on its own, so it never costs a whole host copy. A source over a
mapped file (`Nx_buffer.file_range`) is read from the file: each chunk of a
contiguous one with `pread` into the staging chunk, and a strided one that
permutes a contiguous run of the file, a transposed weight, as that run read
into a host copy which is then cut into pieces. The file opened must be the
one that was mapped, and otherwise the mapping is read. A value already
resident on `device` is returned as it is, and one resident elsewhere goes
through the host. On the CPU device the result is `Nx.contiguous x`, unless
`RUNE_JIT_FORCE_COPY` is set. Inside `jit`, `grad`, `jvp` and `vmap` it is
`x`: it performs nx's existing `to_device` effect, which every rune handler
continues with its argument, and places only when no handler answers.

**Captures bind.** A compiled function that captures an unforced resident
value on its own single device binds that value's buffer as the constant, when
its first trace meets the capture. Compiled outputs bind like placed values:
an unread output that another function captures stays on the device for the
life of that function, where it used to be read to the host. No bytes move,
and every signature and every compiled function that captures it shares the
one buffer. The trace table keys such a capture by its resident id; captures
are forced today because the table hashes keys structurally (`jit.ml`,
`lift_const`). nx's deferred tensors keep their id after a read, which they
drop today (`nx_effect.ml:371-374`). A capture that is a host tensor is
uploaded once per compiled function, as today. A capture resident on another
device is read to the host and uploaded, and under `pmap` a placed capture is
read back and replicated, as today. Under `grad`, `vmap` or `with_debug`
outside a compiled function the function runs eagerly, so its placed captures
are read back like any eager use: differentiate inside `jit`.

**Reads and donation of a bound value.** Binding is permanent: a bound value
keeps its buffer until the value is unreachable, and a compiled function keeps
the values it binds reachable. A host read of a bound value copies out, keeps
the host copy on the handle as any forced tensor does, and leaves the buffer
in place. A bound value passed as an input leaf of a `~donate:true` call seeds
the input with no transfer and is not consumed, and an output that returns it
unchanged is a device copy. A read or a donation before the first compile
treats the value as unbound. `RUNE_JIT_DEBUG=1` reports a bound leaf as
`bound`.

**Accounting.** `RUNE_JIT_RESIDENT_BUDGET` compares against the bytes of
compiled outputs only: with 13.76 GB of weights resident it would otherwise
run a major collection before every output allocation of every decode step
(`jit.ml`, `create_fresh_buffer`). `jit_stats().resident_bytes` still counts
placed values. Every upload and every read-back in rune takes the chunked
path, which replaces the whole-leaf staging `Bytes` kept per distinct size
with one shared chunk and, per compiled function, the shorter lengths it
transfers on replay, and the chunked path synchronises after 256 MiB copied
since the last synchronize, which bounds CUDA's pending pinned buffers with
tolk's runtime left as ported.

### Unaligned vector types on the CPU device

Rune's CPU programs read and write host memory in place, and tolk's C renderer
declares each vector type aligned to its size: 16 bytes for four floats, 32
for four doubles. A mapped tensor is aligned to its element size at best (in
the cached Llama 3.2 1B file all 146 tensors sit at addresses congruent to 8
modulo 16), and on x86-64 an aligned vector load of such an address is a fault
no handler catches. The hazard predates mapping: glibc aligns an allocation to
16 bytes, so four-wide float64 kernels already break their declaration about
half the time.

Rune compiles its CPU programs with unaligned vector types, which tolk already
renders under its `ALIGNED=0` setting and now also takes as an argument of its
CPU device, a small extension recorded among tolk's divergences, so every host
pointer is read in place whatever its address. On arm64 it costs nothing:
Llama 3.2 1B decodes at the same rate either way. A guard that copies a
misaligned pointer was the first design and was measured out. At 64 bytes, the
widest alignment the renderer declares, it copies every weight and every cache
fed back on every call on Linux, where glibc places each allocation of 128 KB
or more at 16 modulo 4096, and no entry of any cached checkpoint passes it. At
16 bytes it leaves the float64 hazard in place.

### Peak memory

Committed memory is what the process owns: the OCaml heap, malloc, device
buffers. File cache is listed apart, because the kernel reclaims it without
I/O. `D` is the model at its run dtypes, `C` the leaves cast on the host, `S`
the sum of distinct leaf sizes (rune stages an upload in a `Bytes` it keeps
per size for as long as the compiled function is reachable, `jit.ml:2828,
3018`), `T` the largest strided leaf. Before placement exists, committed
memory is `C` after the import, `C + D + S + T` during the first compiled call
and `C + D + S` after it. With placement inside the importer it is the leaves
placed so far, the one being placed included, plus that leaf's cast and
strided copy plus one chunk, and `D` from the end of the import. The bound
below is free of the order in which the importer's record fields are
evaluated, which OCaml leaves unspecified: `D` plus the largest cast and
strided copy plus one chunk. A misaligned file adds its size from the load
onward. `T` counts one strided copy: each is garbage once uploaded, so the
bound holds as fast as the collector frees them, and the ceiling is the sum of
the strided leaves, 2.4 GB for gpt-oss and 15.0 GB for Llama 3.1 8B.

On a 32 GB M1 Max, in GB, peak and steady:

| Model | Configuration | Captured host leaves | Placed leaves | Today |
| --- | --- | --- | --- | --- |
| GPT-2 124M, 0.55, misaligned | Metal, float32 | 1.2, 1.2 | 1.1, 0.5 | 1.8 |
| Llama 3.2 1B, 2.47 | Metal, bfloat16 | 3.1, 3.0 | 2.6, 2.5 | 15.8 measured |
| Llama 3.2 1B | Metal, float32 by host cast | 11.1, 11.0 | 6.1, 4.9 | 15.8 measured |
| Llama 3.1 8B, 16.06 | Metal, bfloat16 | 18.3, 17.3 | 17.2, 16.1 | 48: does not load |
| Llama 3.1 8B | Metal, float16 by host cast | 34.4: does not fit | 18.2, 16.1 | does not load |
| gpt-oss-20b, 13.76 | Metal, uint8 and bfloat16 | 16.5, 15.4 | 15.0, 13.8 | does not load |
| gpt-oss-20b | CPU device | 3.6 committed, 11.3 file cache | same | does not load |

Every figure but today's is derived from sizes and code paths. Measured on
Llama 3.2 1B after stage 2: loading and importing at the file's bfloat16
commits 0.01 GB in 0.35 s, and 4.96 GB in 0.73 s at float32, against 15.8 GB
in 4.0 s before. With a first compiled call at bfloat16 the peak is 3.26 GB on
Metal and 2.40 GB on the CPU device. After stage 3, with placed leaves: 2.79
GB at bfloat16 and 6.49 GB at float32 on Metal. A synthetic checkpoint with
Llama 3.1 8B's headers peaks at 16.80 GB at bfloat16 and 18.16 GB at float16
on Metal, and at 15.27 GB on the CPU device, where it peaked at 18.85 GB
before the chunked path. The weights of a synthetic gpt-oss-20b place at 13.76
GB. That model's program is another matter: its twelve-token prefill does not
fit in 32 GB, because the all-experts form dequantises every expert of a layer
and a compiled program gives each intermediate its own buffer for the whole
call. It is outside this RFC. The device reports a working set of 26.8 GB and
a maximum buffer of 20.1 GB; the largest leaf is 1.16 GB.

### Order of work

0. Atomic save, atomic download, a CI test that decides the Windows question
   (load, keep one tensor, save over the path, delete the file), and a load
   benchmark (one process per configuration under `/usr/bin/time -l`, wall
   time to the end of the first compiled call and peak memory footprint, on a
   synthetic checkpoint with Llama 3.2 1B's header, aligned and misaligned,
   CPU device and Metal), run by hand since CI has neither the memory nor
   Metal, and first on today's loader.
1. `reinterpret`, the mapped loader with the old one compared tensor by tensor
   on the cached real files before it is deleted, and unaligned vector types
   for rune's CPU programs. New tests: a truncated file, a hand-written
   misaligned file equal to its aligned twin, `F8_E8M0` as bytes,
   `reinterpret` per kind and across a major collection. Test helpers and
   documentation examples that delete a file after loading it copy what they
   return and run a major collection first.
2. `to_tensor`, `to_float`, `Nx.cast`, the importers of `04-gpt2` and
   `05-llama`, the removals, the docs. `backend_intf.ml:108-110` is corrected
   to RFC 0001's wording.
3. `Rune.to_device`, bound captures, the chunked path with its synchronize
   bound, the budget rule, `rune.mli`'s residency text; importers gain
   `?device`. One test per rule of the `Rune.to_device` section, under
   `RUNE_JIT_FORCE_COPY=1`, in the manner of `test_jit.ml`'s donation group.

**The shortest path to gpt-oss-20b is five commits:** atomic save,
`reinterpret`, the mapped loader, the two accessors, and the gpt-oss example's
importer (on its own branch) moved to them at bfloat16. Loaded means the first
compiled call returns with every capture uploaded. Each is tested on the
cached tiny gpt-oss and Llama 3.2 1B files, then on a synthetic checkpoint
written from the real shards' headers.

## Laws

1. **A file that a reachable tensor views does not change, and raven never
   writes in place a path it maps.** Saves and downloads write a uniquely
   named temporary file and rename it. Prevents a bus error from a truncated
   mapping, which no handler can catch, and tensors whose values change after
   they were read.
2. **A file is validated before it is mapped:** the header length is within
   the limit and the file; header, data and file lengths agree; the entries
   tile the data section. Prevents a fault at first touch of a truncated
   download, and views outside the file. A length check does not detect a
   preallocated partial file; files fetched by other tools fall under Law 1.
3. **Loading an aligned file reads no tensor bytes.** An entry's pages are
   read when first used. Prevents a peak proportional to the file before
   extraction starts.
4. **An entry is viewed in place only when its address is a multiple of its
   element size, rune's CPU programs read host pointers through unaligned
   vector types, and no function reports or depends on either.** Prevents
   unaligned typed loads, which fault in vector code on x86-64, and code that
   breaks on the copy path.
5. **Bytes are never reinterpreted silently.** Only `to_float` casts, and only
   between floating-point dtypes; every other mismatch raises, a template's
   dtype included; a dtype nx lacks arrives as its bytes. Prevents
   block-quantised weights being value-cast to floats, a restart narrowing its
   optimizer state without a word, and tensors vanishing from a load.
6. **An import names what it reads.** A missing entry, a wrong shape and a
   refused dtype raise at import with the entry's name. Prevents a wrong
   configuration surfacing as a shape error inside the first forward pass,
   after the upload.
7. **Placement changes no observable.** `to_device x` equals `x` in type,
   shape, dtype and value. Prevents device-typed tensors and a second set of
   rules for donation and reads.
8. **A resident capture is bound, never read back or uploaded, and an upload
   stages a bounded chunk.** Prevents a device copy per compiled function, the
   read-back of a placed weight, and whole-model staging.
9. **A compiled function keeps the values it binds reachable; a bound buffer
   is never released by a read and never consumed by donation.** Prevents a
   program reading a freed constant, and a donating step consuming the weights
   it captures.
10. **Buffers `to_device` allocates bypass the allocator cache.** Prevents a
    dropped model staying allocated in an unbounded cache.

## Drawbacks

- A truncated or rewritten file, a dropped network mount, a file that another
  host replaces (even by rename) or an unplugged disk is `Bus error` under a
  mapping, with no exception and no backtrace, where a read raises. The docs
  say to keep checkpoints on local disk and that `Nx.copy` detaches.
- Reading through the mapping runs at page-fault speed, 0.5 to 0.9 GB/s at
  full size, cold or warm. Uploads avoid it by reading the file (see
  Unresolved questions, first entry). What still walks the mapping is every
  eager use of a mapped tensor, a host cast among them, the leaves the CPU
  device reads in place at their first touch, and a strided view that is not a
  permutation of a contiguous run. An upload of a transposed leaf reads its run
  into a host copy of the leaf for the length of the upload, 1.16 GB at most
  for gpt-oss-20b, which is the `T` of the peak-memory bound. After the read
  the placement of a transposed leaf is bound by the strided copy that makes
  its pieces contiguous, about 1 GB/s.
- On Windows, and on Linux under strict overcommit, a private writable mapping
  is charged in full against the commit limit: 13.76 GB of page file for this
  checkpoint, 141 GB for a 70B model. A read-only mapping stub removes the
  charge. On Windows a loaded file cannot be deleted or replaced until its
  tensors are unreachable and collected.
- `Linear` stores `[inputs; outputs]`, the transpose of the file's layout, so
  every projection reaches a device through one contiguous copy, and on the
  CPU device every projection is a permanent anonymous copy: 15.0 GB of the
  8B's 16.06 GB.
- Stage 3 is about a hundred lines in rune's constant binding and donation,
  where a wrong decision is a use after free. Each rule refuses an action, and
  stage 3 tests each. Until it lands: a leaf cast on the host stays beside its
  device copy, a misaligned file's copies stay beside the device copy, Llama
  3.1 8B at float16 does not load on 32 GB, a second compiled function uploads
  a second device copy, training keeps its first-call double copy, uploads
  stage a whole leaf per distinct size for the life of the compiled function
  (1.6 GB for gpt-oss), and CUDA parks a pinned buffer per copy until the next
  synchronize.
- A host read of a placed value that no compiled function has bound evicts it,
  as for any resident value: a save in the middle of training evicts the
  state, and the next step uploads it into buffers the program owns for life,
  so a training loop places its state again after a save. A host read of a
  bound value keeps its host copy, so a program that reads every bound weight,
  as a save does, holds the model twice.
- A dropped compiled function's own buffers return to tolk's allocator cache,
  which is unbounded and keyed by size, so a process that drops one model and
  loads another can hold both until an allocation fails.
- While an upload walks the file, cache grows toward the file's size beside
  the device buffers: up to 27.5 GB resident for gpt-oss on a 32 GB machine
  with placement, 30 before it, if the kernel evicts nothing. The pages are
  clean, so eviction is a drop. A process monitor's resident column reads
  about twice the model; the docs point at the memory footprint on macOS and
  `RssAnon` on Linux.
- Importers grow a `?device` argument by convention. nx has no JavaScript test
  target, so the JavaScript stub of `reinterpret` ships unexercised.

## Rationale and alternatives

**Leave placement to its own RFC and ship reading and extraction alone.** The
strongest alternative. Reading and extraction load the forcing case, a mapped
capture costs no committed memory, and RFC 0001 reserves placement. Without
`to_device` the one-copy property rests on two things in the caller's code
that the API does not reveal (keep the file's dtype on the host; capture from
one compiled function), Llama 3.1 8B at float16 does not load, and training
keeps its first-call double copy. The single-device function has little design
room: its result is the resident value RFC 0001 already defines, and the rules
for bound buffers are forced by memory safety. What RFC 0001 reserves stays
reserved (see Non-goals), and `to_device` extends to it by optional arguments.

**`to_device` without bound captures; every placed value surviving a host
read; placement inside kaun.** The first costs a second upload: rune forces a
captured resident value to the host, releases it and uploads it again. The
second is right for saves during training and is a statement about all
resident values, compiled outputs included; one semantics is kept, with the
single exception memory safety forces. The third fails on the importer's
transpose, which comes after the read: an eager operation on a resident value
reads it back to the host.

**A reader value that retains nothing, with `Checkpoint.t` storing the reader
and a name.** It frees the unused copies of a misaligned file, those of cast
leaves during the import and all of them while the checkpoint stays reachable,
0.55 GB in the GPT-2 training example. The leaves an importer keeps at the
file's dtype are copies under either design. It costs a public module beside
`load_safetensors`, a checkpoint that is no longer a collection of tensors,
and a fresh tensor per read, which rune's identity-keyed captures upload once
each. Deferring a misaligned entry's copy to its first access saves nothing
either: every movement operation and every capture forces it.

**Weights as inputs of the step, returned unchanged under donation.** The
first call copies every host input into buffers the program owns for life and
produces a second set, replay walks 459 more leaves per token, and a step that
forgets to return the weights loses them.

**A generic extractor (a free template, a names-aware traversal, a ppx) or
checkpoint combinators (`rename`, `transpose`, `split`).** `to_params` is over
one leaf type, so gpt-oss unpacks every leaf at every use or every model
implements a second traversal, and the importer stays two descriptions of one
structure joined by a string table. The combinators are the first three
operations of a vocabulary nx already has; transformers' conversion table is
past two thousand lines.

**One accessor that casts floats, or none.** An importer can write the cast
itself: `Checkpoint.get` returns an existential and `Nx.cast` accepts any
source dtype. The helper then owns the floating-point guard, the shape check
and the entry's name in the error, every importer repeats it, and one that
omits the guard value-casts uint8 blocks, so Law 5 would rest on user code. A
single `to_tensor` that casts floats hides an allocation the size of the model
behind the accessor every leaf uses. Two functions keep the guard in kaun and
state in a type that only floats are cast.

**A `?cast` flag; no shape argument, or an optional one.** Every importer
would set the flag and every restart would leave it unset. Without `~shape`
the forward pass catches a wrong shape after the upload and without the
entry's name.

**Tensors that own their bytes, read with `pread`, or a knob to choose.** A
read raises where a mapping faults, and it is several times faster than a walk
of the mapping. Owning the bytes turns the model into anonymous memory in
every eager run, and the CPU device loses its in-place leaves, 11.3 GB of
gpt-oss. The mapping stays the reading surface: header-only opens, views at no
committed memory, in-place reads on the CPU device, lazy slices. `pread` is
how an upload fills its chunk, which is where the throughput matters, and it
needs only a way from a buffer to its file range. `Nx.copy` serves the caller
who must overwrite a file it loaded.

**Wrapping the mapping as a Metal buffer with no copy.** It saves one pass
over the model, needs a Metal-only stub and a buffer per file with tensors as
offsets, and cannot serve a cast or transposed leaf.

## Non-goals

- Sharded placement, meshes, a read of an unbound resident value that does not
  evict it, a query for where a value lives, moves between devices without the
  host. RFC 0001 reserves them for one RFC.
- Writing at scale: sharded and asynchronous saves, a save that streams tensor
  by tensor, optimizer-state conventions, the data position. It is the
  roadmap's own item, and loading constrains writers only through Law 1.
- Direct I/O, reads that bypass the page cache, copies overlapped per GPU:
  later forms of the same roadmap item, behind the signatures here.
- A change to `Linear`'s layout: a decision about kaun's layers, recorded
  under Drawbacks as a cost.
- Formats other than safetensors.

## Unresolved questions

Resolved by measurement, 2026-09-19, 32 GB M1 Max under memory pressure:
- The mapped upload is bound by the page-fault path. Copying a 13.76 GB file
  into Metal buffers ran at 0.49 GB/s cold and 0.93 GB/s warm through the
  mapping, since the cache cannot hold the file beside the buffers, against
  2.4 to 4.8 GB/s with `pread`. A read-ahead hint bought 1.5 times and doubled
  the footprint to 27.4 GB; copying straight into the buffer without the
  staging chunk did not help. So an upload reads a file-backed leaf with
  `pread` into its chunk, through `Nx_buffer.file_range`. Placing the synthetic
  gpt-oss-20b on Metal went from 28.6 s to 7.3 s, cold or warm, and its peak
  footprint from 14.28 to 15.50 GB, the difference being the host copy of the
  largest transposed leaf. Llama 3.2 1B cold, to the end of the import: 5.8 to
  3.2 s on Metal and 4.8 to 2.8 s on the CPU device. Warm, where the file fits
  in the cache, the read costs 0.1 to 0.3 s more than the mapping.

Open:
- What unaligned vector types cost on x86-64. It is unmeasured; current CPUs
  execute an unaligned load of an aligned address at the same speed.
- Whether Windows lets a file with a live view be deleted or renamed over (the
  stage 0 test); raven assumes it does not.

During stage 1:
- Whether a 1 GB host transient is collected before the next is allocated,
  which decides whether Llama 3.1 8B's bound holds before stage 3. The load
  benchmark records it.

## Future possibilities

A reader value with a row-range read, for a process that reads its slice of a
misaligned entry; direct reads into pinned staging or into a device's
host-visible buffer through tolk's unimplemented `copy_from_disk` hook, which
`Nx_buffer.file_range` now makes possible and which would remove the staging
copy from the 7.3 s above; a streaming writer; a read-only mapping stub. Nothing listed here is a reason to
accept this or a later RFC.
