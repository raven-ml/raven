(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The world's failures (private). {!Device_pci} exports {!Failed}. *)

exception Failed of string
(** [Failed why] is {!Device_pci.Failed}. C raises it by the name
    ["Device_pci.Failed"]. *)

val fail : ('a, unit, string, 'b) format4 -> 'a
(** [fail fmt ...] raises {!Failed} with the formatted message. *)

val step : string -> (unit -> 'a) -> 'a
(** [step what f] is [f ()], whose [Unix.Unix_error] raises {!Failed} as
    ["what: cause"], the cause the system's message for the error. *)
