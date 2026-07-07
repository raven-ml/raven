(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jit device factory with Metal support. *)

let create name =
  if String.starts_with ~prefix:"CPU" name then Tolk_cpu.create name
  else if String.starts_with ~prefix:"CUDA" name then (
    try Tolk_cuda.create name
    with Failure msg ->
      invalid_arg (Printf.sprintf "Rune.jit: device %s unavailable: %s" name msg))
  else if String.starts_with ~prefix:"METAL" name then Tolk_metal.create name
  else invalid_arg (Printf.sprintf "Rune.jit: unknown device %s" name)
