(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

module Ffi = struct
  external version : unit -> int * int = "caml_tolk_nvrtc_version"

  external compile : string -> string array -> bool -> (bytes, string) result
    = "caml_tolk_nvrtc_compile"
end

let compile_options arch =
  let includes =
    match Helpers.getenv_str "CUDA_PATH" "" with
    | "" -> [ "-I/usr/local/cuda/include"; "-I/usr/include"; "-I/opt/cuda/include" ]
    | cuda_path -> [ "-I" ^ cuda_path ^ "/include" ]
  in
  let options = ("--gpu-architecture=" ^ arch) :: includes in
  let major, minor = Ffi.version () in
  if (major, minor) >= (12, 4) then options @ [ "--minimal" ] else options

let create ?(ptx = true) ~cache_key arch =
  let options = Array.of_list (compile_options arch) in
  let compile src =
    match Ffi.compile src options ptx with
    | Ok lib -> lib
    | Error msg -> raise (Compiler.Compile_error msg)
  in
  Compiler.make
    ~name:(String.uppercase_ascii cache_key)
    ~cachekey:("compile_" ^ cache_key ^ "_" ^ arch)
    ~compile ()
