(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Rig_nv_abi

let strf = Printf.sprintf

(* GPUs and kernels *)

let ampere = 0xc7c0
let ada = 0xc9c0
let blackwell = 0xcec0
let classes = [ ampere; ada; blackwell ]

let class_name c =
  if c = ampere then "ampere"
  else if c = ada then "ada"
  else if c = blackwell then "blackwell"
  else strf "0x%x" c

let gpu ?(compute_class = ada) ?(sass_version = 0x89)
    ?(shared_window = 0x7294_0000_0000) ?(local_window = 0x7293_0000_0000) () =
  {
    Gpu.compute_class;
    sass_version;
    gpcs = 11;
    tpcs_per_gpc = 6;
    sms_per_tpc = 2;
    warps_per_sm = 48;
    shared_window;
    local_window;
    local = (fun _ -> Ok ());
  }

let kernel ?(code_bytes = 0x100) ?(registers = 32) ?(shared_bytes = 0)
    ?(stack_bytes = 0x20) ?(params_offset = 0) ?(banks = []) () =
  {
    Cubin.code = 0x80;
    code_bytes;
    registers;
    shared_bytes;
    stack_bytes;
    params_offset;
    params_bytes = 0;
    banks;
  }

let launch g k = match Launch.make g k with Ok l -> l | Error e -> failwith e

let pp_bank ppf (b : Cubin.bank) =
  Format.fprintf ppf "{ index = %d; offset = 0x%x; bytes = %d }" b.index
    b.offset b.bytes

let bank = Testable.make ~pp:pp_bank ~equal:( = )

let compute_class =
  Gen.of_list
    ~pp:(fun ppf c -> Format.pp_print_string ppf (class_name c))
    classes

let pp_kernel ppf (k : Cubin.kernel) =
  Format.fprintf ppf
    "{ code = 0x%x; code_bytes = %d; registers = %d; shared_bytes = %d; \
     stack_bytes = %d; params_offset = 0x%x; params_bytes = %d; banks = [%a] }"
    k.code k.code_bytes k.registers k.shared_bytes k.stack_bytes k.params_offset
    k.params_bytes
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       pp_bank)
    k.banks

(* Words *)

let words s =
  List.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

let u64 =
  Gen.frequency
    [
      (3, Gen.int64);
      ( 1,
        Gen.of_list
          ~pp:(fun ppf -> Format.fprintf ppf "0x%Lx")
          [
            0L;
            1L;
            -1L;
            0xffff_ffffL;
            0x1_0000_0000L;
            0x8000_0000L;
            Int64.min_int;
            Int64.max_int;
          ] );
    ]

let le64 n =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 n;
  Bytes.to_string b

let round_up n a = (n + a - 1) / a * a

(* Templates *)

let pp_patch ppf (p : Template.patch) =
  Format.fprintf ppf
    "{ at = %d; bits = %d; slot = %d; add = 0x%Lx; shift = %d }" p.at p.bits
    p.slot p.add p.shift
