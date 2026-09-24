(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

(* FFI Externals *)

external cpu_alloc : int -> nativeint = "caml_tolk_cpu_alloc"
external cpu_free : nativeint -> unit = "caml_tolk_cpu_free"
external cpu_copyin : nativeint -> bytes -> unit = "caml_tolk_cpu_copyin"
external cpu_copyout : bytes -> nativeint -> unit = "caml_tolk_cpu_copyout"

external cpu_as_buffer : nativeint -> int -> Device.Allocator.host_view
  = "caml_tolk_cpu_as_buffer"
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
  mutable unloaded : bool;
}

let link_symbol ?(libs = []) name =
  let libs = Array.of_list libs in
  link_symbol_raw libs name

let load_program ~name ~lib =
  let prepared =
    Elf_cpu_loader.load ~link_symbol ~entry:name lib
  in
  let size = Elf_cpu_loader.alloc_size prepared in
  let base = exec_alloc size in
  try
    let image = Elf_cpu_loader.link ~base prepared in
    exec_write base image;
    let entry =
      Nativeint.add base
        (Nativeint.of_int (Elf_cpu_loader.entry_offset prepared))
    in
    { base; entry; size; unloaded = false }
  with exn ->
    exec_free base size;
    raise exn

let unload_program loaded =
  if not loaded.unloaded then begin
    loaded.unloaded <- true;
    exec_free loaded.base loaded.size
  end

(* Allocator *)

let raw_allocator () =
  let alloc size spec =
    match spec.Device.Buffer_spec.external_ptr with
    | Some ptr -> ptr
    | None -> cpu_alloc size
  in
  let free buf _size spec =
    match spec.Device.Buffer_spec.external_ptr with
    | Some _ -> ()
    | None -> cpu_free buf
  in
  let offset buf _size byte_offset =
    if byte_offset < 0 then invalid_arg "CPU buffer offset must be non-negative";
    Nativeint.add buf (Nativeint.of_int byte_offset)
  in
  {
    Device.Allocator.alloc;
    free;
    copyin = cpu_copyin;
    copyout = cpu_copyout;
    as_buffer = Some cpu_as_buffer;
    addr = Fun.id;
    offset = Some offset;
    transfer = None;
    supports_transfer = false;
    copy_from_disk = None;
    supports_copy_from_disk = false;
  }

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
      if loaded.unloaded then invalid_arg "CPU program has been unloaded";
      let st = if wait then monotonic_ns () else 0 in
      exec_call loaded.entry (reorder buffer_slots bufs) (reorder scalar_slots vals);
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
          let compiler = Compiler.make ~name:"CLANG"
              ~cachekey:("compile_clang_jit_" ^ target.arch)
              ~compile:(Compiler_cpu.compile_clang ~arch:target.arch) () in
          Renderer.with_compiler compiler
            (Cstyle.clang ~native_bf16:(Compiler_cpu.supports_bf16 ~arch:target.arch ())
               ?aligned arch)) ] in
  let allocator =
    Device.Allocator.Pack
      (Device.Lru_allocator.wrap (raw_allocator ()))
  in
  Device.make ~name ~allocator ~renderer_set ~runtime ~synchronize ()
