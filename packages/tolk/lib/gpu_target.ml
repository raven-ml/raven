(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type cuda = SM75 | SM80 | SM89
type amd = RDNA3 | RDNA4 | CDNA3 | CDNA4
type metal = Apple of int | Mac of int
type opencl = string
type cpu = X86_64 | Arm64 | Riscv64

let uname flag =
  try
    let ic = Unix.open_process_in ("uname " ^ flag) in
    let value = input_line ic in
    let _ = Unix.close_process_in ic in
    String.trim value
  with _ -> ""

let cpu_of_machine machine =
  match String.lowercase_ascii machine with
  | "x86_64" | "amd64" -> Some X86_64
  | "arm64" | "aarch64" -> Some Arm64
  | "riscv64" -> Some Riscv64
  | _ -> None

let host_cpu () =
  let machine =
    if String.equal Sys.os_type "Win32" then
      match Sys.getenv_opt "PROCESSOR_ARCHITECTURE" with
      | Some arch when String.trim arch <> "" -> arch
      | _ -> "amd64"
    else uname "-m"
  in
  match cpu_of_machine machine with
  | Some arch -> arch
  | None -> invalid_arg (Printf.sprintf "unsupported CPU architecture %S" machine)

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

let parse_metal_arch arch =
  let arch = String.trim arch in
  let split prefix =
    let plen = String.length prefix in
    let alen = String.length arch in
    if alen <= plen || not (String.equal (String.sub arch 0 plen) prefix) then
      None
    else
      match int_of_string_opt (String.sub arch plen (alen - plen)) with
      | Some family when family > 0 -> Some family
      | Some _ | None -> None
  in
  match split "Apple" with
  | Some family -> Some (Apple family)
  | None -> Option.map (fun family -> Mac family) (split "Mac")
