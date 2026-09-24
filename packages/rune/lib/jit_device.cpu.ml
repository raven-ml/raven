(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jit device factory on platforms without Metal support. *)

let create name =
  if String.starts_with ~prefix:"CPU" name then
    Tolk_cpu.create ~aligned:false name
  else if String.starts_with ~prefix:"AMD" name then
    try Tolk_amd.create name
    with Failure msg ->
      invalid_arg
        (Printf.sprintf "Rune.device: device %s unavailable: %s" name msg)
  else if String.starts_with ~prefix:"NV" name then
    try Tolk_nv.create name
    with Failure msg ->
      invalid_arg
        (Printf.sprintf "Rune.device: device %s unavailable: %s" name msg)
  else if String.starts_with ~prefix:"CUDA" name then
    try Tolk_cuda.create name
    with Failure msg ->
      invalid_arg
        (Printf.sprintf "Rune.device: device %s unavailable: %s" name msg)
  else if String.starts_with ~prefix:"METAL" name then
    invalid_arg "Rune.device: device METAL is only available on macOS"
  else invalid_arg (Printf.sprintf "Rune.device: unknown device %s" name)
