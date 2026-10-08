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

(* Fake GPUs *)

type gpu = {
  machine : Rig_pci.Machine.t;
  far : int;
  regs : Rig_pci.Window.t;
  released : int ref;
}

let regs_size = 16 lsl 20

let gpu ?(vendor = fun () -> 0x10de) () =
  let far = Rig_pci_support.far 0 regs_size in
  let tr = Rig_pci.Window.unsafe_transport far in
  let released = ref 0 in
  let fn =
    {
      Rig_pci.Machine.addressing = Physical;
      config8 = (fun _ -> 0);
      config16 = (fun r -> if r = 0 then vendor () else 0);
      config32 = (fun _ -> 0);
      set_config8 = (fun _ _ -> ());
      set_config16 = (fun _ _ -> ());
      set_config32 = (fun _ _ -> ());
      bar = (fun i -> if i = 0 then Some (0, regs_size) else None);
      map =
        (fun ~combine:_ i off n ->
          if i = 0 then Ok (Rig_pci.Window.through tr off n)
          else Error "no such BAR");
      unmap = ignore;
      interrupt = (fun _ -> false);
      reset = (fun () -> Ok ());
      alloc_dma = (fun ~contiguous:_ ~va:_ _ -> Error "no memory");
      free_dma = ignore;
      pin = (fun _ _ -> Error "no memory");
      unpin = (fun _ _ -> ());
      release = (fun () -> incr released);
    }
  in
  let id =
    {
      Rig_pci.Machine.bus = "0000:01:00.0";
      vendor = 0x10de;
      device = 0x2684;
      class_ = 0x030000;
    }
  in
  let ops =
    {
      Rig_pci.Machine.transport = tr;
      page = 4096;
      functions = (fun () -> [ id ]);
      take = (fun _ -> Ok fn);
      reserve = (fun ~base:_ _ -> Ok ());
    }
  in
  let machine = Rig_pci.Machine.make ~name:"far:1" ops in
  { machine; far; regs = Rig_pci.Window.through tr 0 regs_size; released }
