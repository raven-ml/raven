(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the Metal suite and bench share. *)

(** {1:ring A device's ring, by hand}

    The ring of a device's command buffers, driven without Metal: the suite
    plays the submitter, which commits command buffers, and Metal's handler,
    which completes them in any order. Commit [k] (from [0]) takes the value
    after the last one taken when it ends a submission, [0] otherwise, and
    completes with the times [10k + 1] and [10k + 2]. *)

type ring
(** The type for rings driven by hand. *)

val ring : int -> ring
(** [ring n] is a ring of [n] slots, [1 <= n <= 8], whose word reads [0]. *)

val commit : ring -> last:bool -> int
(** [commit r ~last] takes a slot of [r] and commits its command buffer, which
    ends a submission iff [last]. It is the slot's index. The ring has a free
    slot. *)

val complete : ring -> int -> failed:bool -> unit
(** [complete r i ~failed] completes the command buffer of slot [i], which
    failed with the message ["command buffer k failed"] iff [failed]. *)

val defer : ring -> int
(** [defer r] defers a release to the last slot taken and is its number, from
    [0]. *)

val ran : ring -> int array
(** [ran r] is the releases that ran, in the order they ran. *)

val word : ring -> int
(** [word r] is the value in [r]'s word. *)

val times : ring -> int -> int * int
(** [times r k] is the start and end times written for commit [k], [(0, 0)]
    until written. *)

val failure : ring -> string option
(** [failure r] is the first failure the ring recorded. *)

val sleep : ring -> string option
(** [sleep r] is what the ring's sleep answers when the word changes at once. *)

val stop : ring -> bool
(** [stop r] is [true] after writing the last value taken into the word, iff no
    slot is taken. *)
