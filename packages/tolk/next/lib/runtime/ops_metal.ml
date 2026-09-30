(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

external metal_load : string -> (unit, string) result = "caml_tolk_metal_load"
external macos_major : unit -> int = "caml_tolk_metal_macos_major"

external metal_compile : string -> (string, string) result
  = "caml_tolk_metal_compile"

(* The framework is loaded once, by whichever domain first compiles. *)
let mtl_compiler =
  let lock = Mutex.create () and loaded = ref None in
  fun () ->
    Mutex.protect lock @@ fun () ->
    match !loaded with
    | Some l -> l
    | None ->
        let l =
          match C.findlib "MTLCompiler" [ "MTLCompiler" ] with
          | None ->
              Error
                "failed to load library MTLCompiler: try setting \
                 MTLCOMPILER_PATH?"
          | Some path ->
              Result.map_error
                (fun e -> "failed to load library MTLCompiler: " ^ e)
                (metal_load path)
        in
        loaded := Some l;
        l

(* MetalCompiler *)

let compile src =
  (match mtl_compiler () with
  | Ok () -> ()
  | Error e -> raise (Renderer.Compiler.Compile_error e));
  (* No changes for compute in 2.0 - 2.4 specs, use 2.0 as default for old
     versions. *)
  let metal_version =
    match macos_major () with
    | m when m >= 26 -> "metal4.0"
    | m when m >= 14 -> "metal3.1"
    | m when m >= 13 -> "metal3.0"
    | _ -> "macos-metal2.0"
  in
  (* llvm creates modules.timestamp in the cache path and caches the compilation
     of Metal's standard library there (250 ms to 8 ms). *)
  let params =
    Printf.sprintf
      "-fno-fast-math -std=%s --driver-mode=metal -x metal \
       -fmodules-cache-path=\"%s\" -fno-caret-diagnostics"
      metal_version Helpers.cache_dir
  in
  (* The source is padded to a multiple of 4 bytes with at least one NUL; the
     parameters just end with one. *)
  let n = String.length src in
  let src_padded = src ^ String.make (Helpers.round_up (n + 1) 4 - n) '\000'
  and params_padded = params ^ "\000" in
  let sizes = Bytes.create 16 in
  Bytes.set_int64_le sizes 0 (Int64.of_int (String.length src_padded));
  Bytes.set_int64_le sizes 8 (Int64.of_int (String.length params_padded));
  match metal_compile (Bytes.to_string sizes ^ src_padded ^ params_padded) with
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

let compiler () = Renderer.Compiler.v ~cachekey:"compile_metal_direct" compile
