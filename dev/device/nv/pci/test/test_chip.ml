(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Which chips open, from their NV_PMC_BOOT_42 register. *)

open Windtrap
module Chip = Device_nv_pci.Chip

(* NV_PMC_BOOT_42: the architecture in bits 29:24, the implementation in bits
   23:20; GA100, AD100 and GB200 are architectures 0x17, 0x19 and 0x1b
   (nv_ref.h). *)
let boot42 ~arch ~impl = (arch lsl 24) lor (impl lsl 20)

let family = function
  | Chip.Ampere -> "Ampere"
  | Ada -> "Ada"
  | Blackwell -> "Blackwell"

let opened = function Ok (f, impl) -> Ok (family f, impl) | Error _ as e -> e

let test_listed =
  cases "a listed chip opens with its family"
    ~name:(fun (name, _, _, _) -> name)
    [
      ("GA102", 0x17, 2, "Ampere");
      ("GA107", 0x17, 7, "Ampere");
      ("AD102", 0x19, 2, "Ada");
      ("AD104", 0x19, 4, "Ada");
      ("GB202", 0x1b, 2, "Blackwell");
      ("GB207", 0x1b, 7, "Blackwell");
    ]
    (fun (_, arch, impl, f) ->
      equal
        (result (pair string int) string)
        (Ok (f, impl))
        (opened (Chip.chip (boot42 ~arch ~impl))))

let test_refused =
  cases "an unlisted chip is refused, named"
    ~name:(fun (name, _, _, _) -> name)
    [
      ("GA100", 0x17, 0, "GA100");
      ("GA105", 0x17, 5, "GA105");
      ("GB204", 0x1b, 4, "GB204");
      ("GH100", 0x18, 0, "architecture 0x18");
      ("TU102", 0x16, 2, "architecture 0x16");
    ]
    (fun (_, arch, impl, named) ->
      let why = require_error (Chip.chip (boot42 ~arch ~impl)) in
      contains ~sub:named why)

(* Only the architecture and implementation decide: the revision bits and the
   bits above 29 are ignored. *)
let test_other_bits () =
  for arch = 0 to 0x3f do
    for impl = 0 to 0xf do
      let b = boot42 ~arch ~impl in
      let noise = 0xc00f_ff00 in
      equal
        (result (pair string int) string)
        (opened (Chip.chip b))
        (opened (Chip.chip (b lor noise)))
    done
  done

let () =
  exit
  @@ run "device_nv_pci.chip"
       [
         group ~timeout:10. "chip"
           [
             test_listed;
             test_refused;
             test "the revision does not change the chip" test_other_bits;
           ];
       ]
