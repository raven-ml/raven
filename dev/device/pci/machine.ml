(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

include Ops

external transport_failed : Window.transport -> string option
  = "caml_device_pci_transport_failed"

external now_ns : unit -> (int[@untagged])
  = "caml_device_pci_now_ns_byte" "caml_device_pci_now_ns"
[@@noalloc]

(* This machine has no transport: nothing fails it. *)
type t = { name : string option; ops : ops }

let address = Address.v
let compare_address = Address.compare
let this = { name = None; ops = Local.ops }
let make ~name ops = { name = Some name; ops }
let name m = m.name
let failed m = transport_failed m.ops.transport
let page m = m.ops.page

let functions m =
  List.sort (fun a b -> Address.compare a.bus b.bus) (m.ops.functions ())

let reserve m ~base n = m.ops.reserve ~base n
let take m bus = m.ops.take bus

(* A wait spins for [spin_ns], where devices mostly answer, then naps [nap_s]
   between calls, so that a long wait holds no core. Elapsed time is compared in
   whole milliseconds against [ms], which cannot overflow. *)
let spin_ns = 1_000_000
let nap_s = 0.0001

let wait m ~ms f =
  let start = now_ns () in
  let rec go () =
    Option.iter failwith (failed m);
    if f () then true
    else
      let elapsed = now_ns () - start in
      if elapsed / 1_000_000 >= ms then false
      else begin
        if elapsed < spin_ns then Domain.cpu_relax () else Unix.sleepf nap_s;
        go ()
      end
  in
  go ()
