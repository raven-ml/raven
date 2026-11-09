(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

external macos_major : unit -> int = "caml_tolk_metal_macos_major"
external macos_build : unit -> string = "caml_tolk_metal_macos_build"

(* MTLCompiler's file. *)
let library () = C.findlib "MTLCompiler" [ "MTLCompiler" ]

(* MTLCompiler reads its options into LLVM's, which are global to a process, at
   every compile, so compiles run in processes of their own: workers, a C
   program embedded in the library ([compiler_metal_worker.c]), each serving
   compiles in turn. *)
let worker = Compiler_worker.v ~name:"metal" Mtlcompiler_worker.program

(* MetalCompiler *)

(* No changes for compute in 2.0 - 2.4 specs, use 2.0 as default for old
   versions. *)
let metal_version () =
  match macos_major () with
  | m when m >= 26 -> "metal4.0"
  | m when m >= 14 -> "metal3.1"
  | m when m >= 13 -> "metal3.0"
  | _ -> "macos-metal2.0"

let options () =
  Printf.sprintf "-fno-fast-math -std=%s --driver-mode=metal -x metal"
    (metal_version ())

(* Metal fuses a product and a sum into one multiply-add unless the source asks
   it not to; its option -ffp-contract=off does not reach the code. *)
let prologue = "#pragma METAL fp contract(off)\n"

let compile library src =
  (* llvm creates modules.timestamp in the cache path and caches the compilation
     of Metal's standard library there (250 ms to 8 ms). *)
  let params =
    Printf.sprintf "%s -fmodules-cache-path=\"%s\" -fno-caret-diagnostics"
      (options ()) Helpers.cache_dir
  in
  let src = prologue ^ src in
  (* The source is padded to a multiple of 4 bytes with at least one NUL; the
     parameters just end with one. *)
  let n = String.length src in
  let src_padded = src ^ String.make (Helpers.round_up (n + 1) 4 - n) '\000'
  and params_padded = params ^ "\000" in
  let sizes = Bytes.create 16 in
  Bytes.set_int64_le sizes 0 (Int64.of_int (String.length src_padded));
  Bytes.set_int64_le sizes 8 (Int64.of_int (String.length params_padded));
  let request = Bytes.to_string sizes ^ src_padded ^ params_padded in
  match Compiler_worker.call worker library request with
  | Error e -> raise (Renderer.Compiler.Compile_error e)
  | Ok reply ->
      (* The library follows a header and the warnings. *)
      let u32 i = Int32.to_int (String.get_int32_le reply i) land 0xFFFF_FFFF in
      let offset = u32 8 + u32 12 in
      let lib = String.sub reply offset (String.length reply - offset) in
      if
        not
          (String.starts_with ~prefix:"MTLB" lib
          && String.ends_with ~suffix:"ENDT" lib)
      then failwith ("Invalid Metal library. " ^ String.escaped lib);
      lib

(* MTLCompiler is part of macOS: its build names the framework, unless
   MTLCOMPILER_PATH names another, whose file then does. The options and the
   prologue are the rest of what a library is a function of: the worker passes
   the request to MTLCompiler as it is. *)
let table library () =
  let identity =
    String.concat "\n"
      [ macos_build (); C.identity (Some library); options (); prologue ]
  in
  "compile_metal_direct_" ^ Digest.to_hex (Digest.string identity)

let compiler () =
  match library () with
  | None ->
      failwith
        "failed to load library MTLCompiler: try setting MTLCOMPILER_PATH?"
  | Some library ->
      Renderer.Compiler.v ~cachekey:(table library) (compile library)
