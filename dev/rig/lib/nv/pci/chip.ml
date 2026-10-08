(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_pci

let strf = Printf.sprintf

type family = Ampere | Ada | Blackwell

type t = {
  fn : Function.t;
  regs : Window.t;
  family : family;
  implementation : int;
}

let field (lo, n) x = (x lsr lo) land ((1 lsl n) - 1)

(* The families by their architecture in NV_PMC_BOOT_42: each one's name prefix
   and the implementations linux-firmware has GSP firmware for. *)
let families =
  [
    (Defs.nv_pmc_boot_42_architecture_ga100, (Ampere, "GA10", [ 2; 3; 4; 6; 7 ]));
    (Defs.nv_pmc_boot_42_architecture_ad100, (Ada, "AD10", [ 2; 3; 4; 6; 7 ]));
    ( Defs.nv_pmc_boot_42_architecture_gb200,
      (Blackwell, "GB20", [ 2; 3; 5; 6; 7 ]) );
  ]

let chip boot42 =
  let arch = field Defs.nv_pmc_boot_42_architecture boot42 in
  let impl = field Defs.nv_pmc_boot_42_implementation boot42 in
  match List.assoc_opt arch families with
  | Some (family, _, impls) when List.mem impl impls -> Ok (family, impl)
  | Some (_, prefix, _) ->
      Error (strf "the chip %s%X is not one this library boots" prefix impl)
  | None ->
      Error
        (strf
           "the chip of architecture 0x%x, implementation 0x%x, is not one \
            this library boots"
           arch impl)

let name c =
  let prefix =
    match c.family with Ampere -> "GA10" | Ada -> "AD10" | Blackwell -> "GB20"
  in
  strf "%s%X" prefix c.implementation

let get c r = Window.get32 c.regs r
let set c r x = Window.set32 c.regs r x

let of_function fn =
  match Function.map ~combine:false fn 0 with
  | Error _ as e -> e
  | Ok regs -> (
      match chip (Window.get32 regs Defs.nv_pmc_boot_42) with
      | Error _ as e ->
          Function.unmap fn regs;
          e
      | Ok (family, implementation) -> Ok { fn; regs; family; implementation })

(* The GPU's memory in MiB, as its firmware wrote it at boot
   (NV_PGC6_AON_SECURE_SCRATCH_GROUP_42). *)
let memory c =
  match get c Defs.nv_pgc6_aon_secure_scratch_group_42 with
  | 0 -> Error (name c ^ "'s firmware wrote no memory size")
  | mib -> Ok (mib lsl 20)

let booted c = get c Defs.nv_pfb_pri_mmu_wpr2_addr_hi <> 0

let wait c what ~ms f =
  if Machine.wait (Function.machine c.fn) ~us:(ms * 1000) f then Ok ()
  else
    match Function.failed c.fn with
    | Some why -> Error why
    | None -> Error (strf "%s did not happen within %d ms" what ms)

let delay c ms =
  ignore
    (Machine.wait (Function.machine c.fn) ~us:(ms * 1000) (fun () -> false))

(* The command register of the configuration space and its bus master bit (PCI
   Express Base Specification, 7.5.1.1.3). *)
let command = 0x04
let bus_master_bit = 0x4

let bus_master fn on =
  let v = Function.config16 fn command in
  let v = if on then v lor bus_master_bit else v land lnot bus_master_bit in
  Function.set_config16 fn command v
