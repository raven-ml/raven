(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type cuda = SM75 | SM80 | SM89
type amd = RDNA3 | RDNA4 | CDNA3 | CDNA4

let parse_cuda_arch arch =
  let buf = Buffer.create (String.length arch) in
  String.iter
    (fun c -> if c >= '0' && c <= '9' then Buffer.add_char buf c else ())
    arch;
  if Buffer.length buf = 0 then None
  else int_of_string_opt (Buffer.contents buf)

let cuda_of_env () =
  let raw =
    match Sys.getenv_opt "CUDA_ARCH" with
    | Some arch when String.trim arch <> "" -> String.trim arch
    | _ -> (
        match Sys.getenv_opt "CUDA_SM" with
        | Some arch when String.trim arch <> "" -> String.trim arch
        | _ -> "")
  in
  match parse_cuda_arch raw with
  | Some ver when ver >= 89 -> Some SM89
  | Some ver when ver >= 80 -> Some SM80
  | Some ver when ver >= 75 -> Some SM75
  | _ -> None

let parse_amd_arch arch =
  let arch = String.trim arch |> String.lowercase_ascii in
  let contains needle =
    let nlen = String.length needle in
    let alen = String.length arch in
    let rec loop i =
      if i + nlen > alen then false
      else if String.sub arch i nlen = needle then true
      else loop (i + 1)
    in
    nlen > 0 && loop 0
  in
  if contains "gfx950" || contains "9.5.0" then Some CDNA4
  else if contains "gfx942" || contains "9.4.2" then Some CDNA3
  else if
    contains "gfx1200" || contains "gfx1201" || contains "12.0.0"
    || contains "12.0.1"
  then Some RDNA4
  else if contains "gfx11" || contains "11." then Some RDNA3
  else None

let amd_of_env () =
  let first_set vars =
    let rec loop = function
      | [] -> ""
      | var :: vars -> (
          match Sys.getenv_opt var with
          | Some value when String.trim value <> "" -> String.trim value
          | _ -> loop vars)
    in
    loop vars
  in
  first_set
    [ "AMD_ARCH"; "HIP_ARCH"; "HCC_AMDGPU_TARGET"; "HSA_OVERRIDE_GFX_VERSION" ]
  |> parse_amd_arch
