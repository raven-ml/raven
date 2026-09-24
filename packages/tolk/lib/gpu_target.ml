(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type cuda = SM75 | SM80 | SM89 | SM90
type amd = RDNA3 | RDNA4 | CDNA3 | CDNA4
type metal = Apple of int | Mac of int
type opencl = string
type cpu = X86_64 | Arm64 | Riscv64

let cpu_of_machine machine =
  match String.lowercase_ascii machine with
  | "x86_64" | "amd64" -> Some X86_64
  | "arm64" | "aarch64" -> Some Arm64
  | "riscv64" | "riscv" -> Some Riscv64
  | _ -> None

let host_cpu () =
  let machine = Host_config.architecture in
  match cpu_of_machine machine with
  | Some arch -> arch
  | None -> invalid_arg (Printf.sprintf "unsupported CPU architecture %S" machine)

let cuda_of_sm sm =
  if sm >= 90 then Some SM90
  else if sm >= 89 then Some SM89
  else if sm >= 80 then Some SM80
  else if sm >= 75 then Some SM75
  else None

let parse_cuda_arch arch =
  if not (String.starts_with ~prefix:"sm_" arch) then None
  else match int_of_string_opt (String.sub arch 3 (String.length arch - 3)) with
    | Some sm -> cuda_of_sm sm
    | None -> None

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
