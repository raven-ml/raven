(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

module Ffi = struct
  external version : unit -> int * int = "caml_tolk_amd_comgr_version"

  external compile : string -> string -> (bytes, string) result
    = "caml_tolk_amd_comgr_compile"
end

let version = Ffi.version

let create ~arch =
  let compile src =
    match Ffi.compile src arch with
    | Ok lib -> lib
    | Error msg -> raise (Compiler.Compile_error msg)
  in
  Compiler.make ~name:"HIP" ~cachekey:("compile_hip_" ^ arch) ~compile ()
