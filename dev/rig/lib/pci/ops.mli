(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The operations of a machine and of a function it took.

    {!Machine} exports and documents them; this machine ({!Local}) and
    transports implement them. They live apart so that {!Local}, below
    {!Machine}, builds a function's. *)

type id = { bus : string; vendor : int; device : int; class_ : int }
type addressing = Physical | Iommu

type fn = {
  addressing : addressing;
  config8 : int -> int;
  config16 : int -> int;
  config32 : int -> int;
  set_config8 : int -> int -> unit;
  set_config16 : int -> int -> unit;
  set_config32 : int -> int -> unit;
  bar : int -> (int * int) option;
  map : combine:bool -> int -> int -> int -> (Window.t, string) result;
  unmap : Window.t -> unit;
  interrupt : int -> bool;
  reset : unit -> (unit, string) result;
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

type ops = {
  transport : Window.transport;
  page : int;
  functions : unit -> id list;
  take : string -> (fn, string) result;
  reserve : base:int -> int -> (unit, string) result;
}
