(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Profiles, documented in rig.mli. *)

open Def

type event = Def.event =
  | Span of {
      device : device;
      lane : string;
      name : string;
      start : int;
      stop : int;
    }
  | Allocation of { device : device; time : int; allocated : int }
  | Load of { program : program; binary : string; time : int }
  | Counters of {
      device : device;
      name : string;
      start : int;
      stop : int;
      counters : (string * int array) list;
    }
  | Trace of {
      device : device;
      name : string;
      start : int;
      stop : int;
      part : int;
      data : string;
    }
  | Overwritten of { device : device; time : int; runs : int }
  | Copy of { src : device; dst : device; bytes : int; start : int; stop : int }

val take :
  ?counters:string list -> ?trace:bool -> (unit -> 'a) -> 'a * event list

val enabled : unit -> bool
val counters : unit -> string list
val traced : unit -> bool
val now : unit -> int
val span : string -> (unit -> 'a) -> 'a
val after : int -> (unit -> event list) -> unit
val record : int -> lane:string -> name:string -> buffer -> unit
val timestamp : nativeint
val output_chrome_trace : out_channel -> event list -> unit
