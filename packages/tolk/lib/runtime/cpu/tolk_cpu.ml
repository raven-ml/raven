(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

let buffer_kind = Tolk_uop.Storage.Host_allocator.kind

(* FFI Externals *)

external exec_alloc : int -> nativeint = "caml_tolk_cpu_jit_alloc"
external exec_free : nativeint -> int -> unit = "caml_tolk_cpu_jit_free"
external exec_write : nativeint -> bytes -> unit = "caml_tolk_cpu_jit_write"
external monotonic_ns : unit -> int = "caml_tolk_cpu_monotonic_ns" [@@noalloc]

external exec_call : nativeint -> nativeint array -> int64 array -> unit
  = "caml_tolk_cpu_jit_call"

external link_symbol_raw : string array -> string -> nativeint
  = "caml_tolk_cpu_jit_link_symbol"

type loaded_program = {
  base : nativeint;
  entry : nativeint;
  size : int;
  unloaded : bool Atomic.t;
}

let unload_program loaded =
  if Atomic.compare_and_set loaded.unloaded false true then
    exec_free loaded.base loaded.size

(* Compiler builtins. No tinygrad counterpart: its loader fails the same way.

   LLVM lowers what a target cannot do inline to calls into its builtins
   library, and a loaded object has no copy of it. On x86-64 without
   AVX512-BF16, a bfloat16 value merged across a branch, such as a gated load
   whose fallback is a constant, is widened to float32 and rounded back by a
   call to [__truncsfbf2], though the source converts nothing. Such a call
   links to a copy compiled once by the kernels' compiler for this host, so
   it follows their calling convention on every host. It rounds as the
   renderer's manual cast does, so it gives back the bits a widened
   bfloat16 came from, NaN payloads included. *)
let builtin_sources =
  [
    ( "__truncsfbf2",
      {|__bf16 __truncsfbf2(float x) {
  union { float f; unsigned int u; } in = { x };
  unsigned int b = in.u;
  b = (-b & 0x7f800000u) ? b + ((b >> 16) & 1u) + 0x7fffu
      : (b & 0xffffu) ? (b | 0x10000u) : b;
  union { unsigned short u; __bf16 f; } out = { (unsigned short)(b >> 16) };
  return out.f;
}|} );
  ]

let builtins = Hashtbl.create 1
let builtins_lock = Mutex.create ()

let rec link_symbol ?(libs = []) name =
  match List.assoc_opt name builtin_sources with
  | Some src -> builtin name src
  | None -> link_symbol_raw (Array.of_list libs) name

and builtin name src =
  Mutex.protect builtins_lock (fun () ->
      match Hashtbl.find_opt builtins name with
      | Some loaded -> loaded.entry
      | None ->
          let loaded =
            load_program ~name ~lib:(Compiler_cpu.compile_clang src)
          in
          Hashtbl.add builtins name loaded;
          loaded.entry)

and load_program ~name ~lib =
  let prepared = Elf_cpu_loader.load ~link_symbol ~entry:name lib in
  let size = Elf_cpu_loader.alloc_size prepared in
  let base = exec_alloc size in
  try
    let image = Elf_cpu_loader.link ~base prepared in
    exec_write base image;
    let entry = Nativeint.add base
        (Nativeint.of_int (Elf_cpu_loader.entry_offset prepared)) in
    let loaded = { base; entry; size; unloaded = Atomic.make false } in
    Gc.finalise unload_program loaded;
    loaded
  with exn ->
    exec_free base size;
    raise exn

(* Allocator *)

(* Device Registration *)

let create ?aligned name =
  let runtime (obj : Tolk_uop.Tiny_elf.t) =
    let entry_name = obj.name and lib = obj.lib in
    let buffers, scalars = List.partition
        (fun (arg : Tolk_uop.Tiny_elf.argument) -> arg.addrspace <> Tolk_uop.Dtype.Alu)
        obj.signature in
    let slots offset args =
      let slots = Array.of_list (List.map
          (fun (arg : Tolk_uop.Tiny_elf.argument) -> arg.slot - offset) args) in
      if Array.for_all Fun.id (Array.mapi (fun i slot -> i = slot) slots)
      then None else Some slots in
    let buffer_slots = slots 0 buffers and scalar_slots = slots (List.length buffers) scalars in
    let reorder slots args = match slots with
      | None -> args
      | Some slots -> Array.map (Array.get args) slots in
    let loaded = load_program ~name:entry_name ~lib in
    let call bufs ~global:_ ~local:_ ~vals ~wait ~timeout:_ =
      let bufs = Array.map (fun buf ->
          Option.value (Device.Buffer.get ~device:name buffer_kind buf) ~default:0n) bufs in
      if Atomic.get loaded.unloaded then invalid_arg "CPU program has been unloaded";
      let st = if wait then monotonic_ns () else 0 in
      Fun.protect
        ~finally:(fun () -> ignore (Sys.opaque_identity loaded))
        (fun () -> exec_call loaded.entry
            (reorder buffer_slots bufs) (reorder scalar_slots vals));
      if wait then Some (float_of_int (monotonic_ns () - st) *. 1e-9)
      else None
    in
    Device.{ call; free = (fun () -> unload_program loaded); handle = 0n }
  in
  let synchronize () = () in
  let renderer_set = Device.Renderer_set.make ~device:name ~arch:(Compiler_cpu.host_arch ())
      [ "CLANG", (fun target ->
          let arch = match String.split_on_char ',' target.Tolk_uop.Target.arch with
            | machine :: _ -> (match Gpu_target.cpu_of_machine machine with
                | Some arch -> arch
                | None -> invalid_arg ("unsupported CPU architecture: " ^ target.arch))
            | [] -> assert false in
          (* The table name records the compiler, which decides native
             bfloat16, as a file name. *)
          let cc = String.map (fun c -> match c with
              | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '.' | '-' -> c
              | _ -> '_') (Compiler_cpu.cc ()) in
          let compiler = Compiler.make ~name:"CLANG"
              ~cachekey:("compile_clang_jit_" ^ cc ^ "_" ^ target.arch)
              ~compile:(Compiler_cpu.compile_clang ~arch:target.arch) () in
          Renderer.with_compiler compiler
            (Cstyle.clang ~native_bf16:(Compiler_cpu.supports_bf16 ~arch:target.arch ())
               ?aligned arch)) ] in
  let allocator =
    Device.Allocator.Pack
      (Device.Lru_allocator.wrap (Tolk_uop.Storage.Host_allocator.make ~synchronize))
  in
  let bufferize u = match Tolk_uop.Uop.as_param u with
    | Some {param = {allocation = Some ("cfunc", data); _}; _} ->
        let libs, symbol = (Marshal.from_string data 0 : string list * string) in
        let address = link_symbol ~libs symbol in
        let buffer = Device.Buffer.create ~device:name ~size:1 ~dtype:Tolk_uop.Dtype.uint64 allocator in
        Device.Buffer.ensure_allocated buffer;
        let bytes = Bytes.create 8 in
        Bytes.set_int64_le bytes 0 (Int64.of_nativeint address);
        Device.Buffer.copyin buffer bytes;
        Some buffer
    | _ -> None in
  Device.make ~name ~allocator ~renderer_set ~runtime ~synchronize:(fun timeout -> ignore timeout; synchronize ())
    ~shares_host_memory:true ~bufferize ()
