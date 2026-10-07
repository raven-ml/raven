(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type id = Ops.id = { bus : string; vendor : int; device : int; class_ : int }
type addressing = Ops.addressing = Physical | Iommu

type fn = Ops.fn = {
  addressing : addressing;
  config : int -> int -> int;
  set_config : int -> int -> int -> unit;
  bar : int -> (int * int) option;
  map : int -> int -> int -> Window.t;
  unmap : Window.t -> unit;
  interrupt : int -> bool;
  reset : unit -> unit;
  alloc_dma :
    contiguous:bool -> va:int option -> int -> Window.t * (int * int) list;
  free_dma : Window.t -> unit;
  pin : int -> int -> (int * int) list;
  unpin : int -> int -> unit;
  release : unit -> unit;
}

type ops = Ops.ops = {
  transport : Window.transport;
  page : int;
  functions : unit -> id list;
  take : string -> (fn, string) result;
  reserve : base:int -> int -> unit;
}

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
    Option.iter (fun why -> raise (Fail.Failed why)) (failed m);
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
