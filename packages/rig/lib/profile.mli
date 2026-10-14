(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Profiles: the implementation of {!Rig.Profile}, over the open profiles of
    {!Prof}.

    Events that the host sees as they happen, such as a load, an allocation or a
    copy the host ran, are recorded immediately. A submit that a profile times
    leaves the profile an entry ({!Prof.pending}) that reads the times its
    driver wrote. When {!take}'s function returns, its profile takes no more
    entries, and [take] reads those it took into it alone, one at a time, so
    those events are in, then stops it. An entry leaves the profile as [take] starts to
    wait for its value, and is dropped however its read ends. When the
    function raises, [take] stops the profile immediately, which drops its
    entries. A device lost meanwhile has its entries dropped. *)

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
