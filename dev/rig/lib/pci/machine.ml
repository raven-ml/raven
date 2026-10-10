(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type id = Ops.id = { bus : string; vendor : int; device : int; class_ : int }
type addressing = Ops.addressing = Physical | Iommu

type fn = Ops.fn = {
  addressing : addressing;
  inherited : bool;
  config : int -> int -> int;
  set_config : int -> int -> int -> unit;
  bar : int -> (int * int) option;
  map : combine:bool -> int -> int -> int -> (Window.t, string) result;
  unmap : Window.t -> unit;
  interrupt : int -> bool;
  reset : unit -> (unit, string) result;
  forget : unit -> (unit, string) result;
  alloc_dma :
    contiguous:bool ->
    va:int option ->
    int ->
    ((Window.t * (int * int) list) option, string) result;
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
  = "caml_rig_pci_transport_failed"

external now_ns : unit -> (int[@untagged])
  = "caml_rig_pci_now_ns_byte" "caml_rig_pci_now_ns"
[@@noalloc]

(* This machine has no transport: nothing fails it. [files] is the files of a
   machine this process reaches without a transport. [reserved] is the ranges
   [reserve] gave, which [Function] checks addresses against before a transport
   is asked: every allocation at an address reads it, so it is read without a
   lock, and [reserve], which is rare, replaces it whole. *)
type t = {
  name : string option;
  ops : ops;
  files : Sysfs.t option;
  reserved : (int * int) list Atomic.t;
}

let address = Bus_address.v
let compare_address = Bus_address.compare
let machine name files ops = { name; ops; files; reserved = Atomic.make [] }

let at root =
  let files = Sysfs.v root in
  machine None (Some files) (Local.ops files)

let this = at "/"
let make ~name ops = machine (Some name) None ops
let files m = m.files
let name m = m.name
let failed m = transport_failed m.ops.transport
let page m = m.ops.page

let functions m =
  List.sort (fun a b -> Bus_address.compare a.bus b.bus) (m.ops.functions ())

let rec record m range =
  let ranges = Atomic.get m.reserved in
  if
    (not (List.mem range ranges))
    && not (Atomic.compare_and_set m.reserved ranges (range :: ranges))
  then record m range

let reserve m ~base n =
  let r = m.ops.reserve ~base n in
  if Result.is_ok r then record m (base, n);
  r

let rec within a n = function
  | [] -> false
  | (base, len) :: l -> (a >= base && a <= base + len - n) || within a n l

let reserved m a n = within a n (Atomic.get m.reserved)
let take m bus = m.ops.take bus

let take_locked m l =
  match m.files with
  | Some files when files == Sysfs.files l -> Local.take l
  | _ -> invalid_arg "Machine.take_locked: the lock is of another machine's"

(* A wait spins for [spin_ns], where devices mostly answer, then naps [nap_s]
   between calls, so that a long wait holds no core. Elapsed time is compared in
   whole microseconds against [us], which cannot overflow. *)
let spin_ns = 1_000_000
let nap_s = 0.0001

let wait m ~us f =
  let start = now_ns () in
  let rec go () =
    let holds = f () in
    if Option.is_some (failed m) then false
    else if holds then true
    else
      let elapsed = now_ns () - start in
      if elapsed / 1_000 >= us then false
      else begin
        if elapsed < spin_ns then Domain.cpu_relax () else Unix.sleepf nap_s;
        go ()
      end
  in
  go ()
