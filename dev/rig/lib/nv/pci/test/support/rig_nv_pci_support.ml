(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type bytes =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external address : bytes -> int = "caml_rig_nv_pci_test_address"

(* The bigarrays live as long as the process, so a window never outlives its
   bytes. *)
let kept = ref []

let window n =
  let b = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Bigarray.Array1.fill b '\000';
  kept := b :: !kept;
  Rig_pci.Window.v (address b) n
