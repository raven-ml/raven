(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type id = Ops.id = { bus : string; vendor : int; device : int; class_ : int }
type addressing = Ops.addressing = Physical | Iommu

type fn = Ops.fn = {
  addressing : addressing;
  config8 : int -> int;
  config16 : int -> int;
  config32 : int -> int;
  set_config8 : int -> int -> unit;
  set_config16 : int -> int -> unit;
  set_config32 : int -> int -> unit;
  bar : int -> (int * int) option;
  map : int -> int -> int -> (Window.t, string) result;
  unmap : Window.t -> unit;
  interrupt : int -> bool;
  reset : unit -> (unit, string) result;
  alloc_dma :
    contiguous:bool ->
    va:int option ->
    int ->
    (Window.t * (int * int) list, string) result;
  free_dma : Window.t -> unit;
  pin : int -> int -> ((int * int) list, string) result;
  unpin : int -> int -> unit;
  release : unit -> unit;
}

type ops = Ops.ops = {
  transport : Window.transport;
  page : int;
  functions : unit -> id list;
  take : string -> (fn, string) result;
  reserve : base:int -> int -> (unit, string) result;
}

external transport_failed : Window.transport -> string option
  = "caml_device_pci_transport_failed"

external now_ns : unit -> (int[@untagged])
  = "caml_device_pci_now_ns_byte" "caml_device_pci_now_ns"
[@@noalloc]

(* This machine has no transport: nothing fails it. [reserved] is the ranges
   [reserve] gave, which [Function] checks addresses against before a transport
   is asked; [lock] guards it. *)
type t = {
  name : string option;
  ops : ops;
  lock : Mutex.t;
  mutable reserved : (int * int) list;
}

let address = Address.v
let compare_address = Address.compare
let machine name ops = { name; ops; lock = Mutex.create (); reserved = [] }
let at root = machine None (Local.ops (Sysfs.v root))
let this = machine None (Local.ops Local.this)
let make ~name ops = machine (Some name) ops
let name m = m.name
let failed m = transport_failed m.ops.transport
let page m = m.ops.page

let functions m =
  List.sort (fun a b -> Address.compare a.bus b.bus) (m.ops.functions ())

let reserve m ~base n =
  let r = m.ops.reserve ~base n in
  if Result.is_ok r then
    Mutex.protect m.lock (fun () ->
        if not (List.mem (base, n) m.reserved) then
          m.reserved <- (base, n) :: m.reserved);
  r

let reserved m a n =
  Mutex.protect m.lock @@ fun () ->
  List.exists (fun (base, len) -> a >= base && a <= base + len - n) m.reserved

let take m bus = m.ops.take bus

(* A wait spins for [spin_ns], where devices mostly answer, then naps [nap_s]
   between calls, so that a long wait holds no core. Elapsed time is compared in
   whole milliseconds against [ms], which cannot overflow. *)
let spin_ns = 1_000_000
let nap_s = 0.0001

let wait m ~ms f =
  let start = now_ns () in
  let rec go () =
    let holds = f () in
    if Option.is_some (failed m) then false
    else if holds then true
    else
      let elapsed = now_ns () - start in
      if elapsed / 1_000_000 >= ms then false
      else begin
        if elapsed < spin_ns then Domain.cpu_relax () else Unix.sleepf nap_s;
        go ()
      end
  in
  go ()
