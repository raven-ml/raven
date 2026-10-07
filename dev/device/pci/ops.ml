(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The operations of a machine and of a function it took, which Machine exports
   and this machine (Local) and transports implement. They live apart so that
   Local, below Machine, builds them. *)

type id = { bus : string; vendor : int; device : int; class_ : int }
type addressing = Physical | Iommu

type fn = {
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

type ops = {
  transport : Window.transport;
  page : int;
  functions : unit -> id list;
  take : lock:string -> string -> (fn, string) result;
  reserve : base:int -> int -> unit;
}
