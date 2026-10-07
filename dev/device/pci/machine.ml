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

let this =
  {
    name = None;
    ops =
      {
        transport = Window.transport 0;
        page = Local.page;
        functions = Local.functions;
        take = Local.take;
        reserve = Local.reserve;
      };
  }

let make ~name ops = { name = Some name; ops }
let name m = m.name
let failed m = transport_failed m.ops.transport
let page m = m.ops.page

let functions m =
  List.sort (fun a b -> Address.compare a.bus b.bus) (m.ops.functions ())

let reserve m ~base n = m.ops.reserve ~base n
let take m ~lock bus = m.ops.take ~lock bus

(* Elapsed time is compared in whole milliseconds, which cannot overflow. *)
let wait m ~ms f =
  let start = now_ns () in
  let rec go () =
    Option.iter failwith (failed m);
    if f () then true
    else if (now_ns () - start) / 1_000_000 >= ms then false
    else begin
      Domain.cpu_relax ();
      go ()
    end
  in
  go ()
