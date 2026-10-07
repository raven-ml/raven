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
  take : lock:string -> string -> (fn, string) result;
  reserve : base:int -> int -> unit;
}

external transport_failed : Window.transport -> string option
  = "caml_device_pci_transport_failed"

external now_ms : unit -> (int[@untagged])
  = "caml_device_pci_now_ms_byte" "caml_device_pci_now_ms"
[@@noalloc]

(* A machine's operations. This machine has no transport: nothing fails it. *)
type t = {
  name : string option;
  transport : Window.transport option;
  page : int;
  functions : unit -> id list;
  take : lock:string -> string -> (fn, string) result;
  reserve : base:int -> int -> unit;
}

let address = Address.v
let compare_address = Address.compare

let this =
  {
    name = None;
    transport = None;
    page = Local.page;
    functions = Local.functions;
    take = Local.take;
    reserve = Local.reserve;
  }

let make ~name (o : ops) =
  {
    name = Some name;
    transport = Some o.transport;
    page = o.page;
    functions = o.functions;
    take = o.take;
    reserve = o.reserve;
  }

let name m = m.name
let failed m = Option.bind m.transport transport_failed
let page m = m.page
let functions m = m.functions ()
let reserve m ~base n = m.reserve ~base n
let take m ~lock bus = m.take ~lock bus

let wait m ~ms f =
  let until = now_ms () + ms in
  let rec go () =
    Option.iter failwith (failed m);
    if f () then true
    else if now_ms () >= until then false
    else begin
      Domain.cpu_relax ();
      go ()
    end
  in
  go ()
