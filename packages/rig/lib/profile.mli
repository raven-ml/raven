(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Profiles: the implementation of {!Rig.Profile}, over the open profiles of
    {!Prof}.

    Events that the host sees as they happen, such as a load, an allocation
    or a copy the host waited for, are recorded immediately. Events of work a
    device runs are recorded once a wait finds their value reached. A submit
    or a copy that a profile times registers a function with {!Dev.after},
    which reads the times its driver wrote. When {!take}'s function returns,
    [take] waits for the last value each device has such a function on
    before it stops its profile, so those events are in. When the function
    raises, [take] stops the profile immediately and the functions stay
    registered. A device lost meanwhile has every such function dropped. *)

open Def

(** {!Rig.Profile.event}. *)
type event = Def.event =
  | Span of {
      device : device;
      lane : string;
      name : string;
      start : int;
      stop : int;
    }
  | Allocation of { device : device; time : int; allocated : int }
  | Load of { image : image; binary : string; time : int }
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

val take : (unit -> 'a) -> 'a * event list
(** [take f] is {!Rig.Profile.take}[ f]. *)

val enabled : unit -> bool
(** [enabled ()] is {!Rig.Profile.enabled}[ ()]. *)

val now : unit -> int
(** [now ()] is {!Rig.Profile.now}[ ()]. *)

val record : event -> unit
(** [record e] is {!Rig.Profile.record}[ e]. *)

val span : string -> (unit -> 'a) -> 'a
(** [span name f] is {!Rig.Profile.span}[ name f]. *)

val output_chrome_trace : out_channel -> event list -> unit
(** [output_chrome_trace oc events] is {!Rig.Profile.output_chrome_trace}[ oc
    events]. *)
