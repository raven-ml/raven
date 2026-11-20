(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Shapes and checks shared by every module of {!Mynd}.

    Each check takes [fn], the public name of the function that calls it, and
    raises [Invalid_argument] with a message that starts with it. *)

val invalid_argf : ('a, Format.formatter, unit, 'b) format4 -> 'a
(** [invalid_argf fmt …] raises [Invalid_argument] with the message [fmt]
    formats. *)

val broadcast : string -> int array -> int array -> int array
(** [broadcast fn a b] is the broadcast of the batch shapes [a] and [b], of one
    rank. *)

val image : string -> ('v, 'r, 'd) Nx.t -> unit
(** [image fn x] raises unless [x] has at least three axes. *)

val points :
  string ->
  batch:int ->
  ('v, 'r, 'd) Nx.t ->
  int array * int array * ('v, 'r, 'd) Nx.t
(** [points fn ~batch p] splits positions [[b…; q…; 2]] with [batch] batch axes
    into [b], [q] and the flat [[b…; q; 2]]. It raises unless [p]'s last axis is
    [2] and it has [batch] axes before the points. *)

val coordinate : ('v, 'r, 'd) Nx.t -> int -> ('v, 'r, 'd) Nx.t
(** [coordinate p k] is part [k] of [p] along its last axis, which it drops. *)

val cut : axis:int -> int -> int -> ('v, 'r, 'd) Nx.t -> ('v, 'r, 'd) Nx.t
(** [cut ~axis lo n x] is [x]'s [n] elements from [lo] along [axis], which is
    negative, counted from the end. *)

val dims : int array -> string
(** [dims s] is the shape [s] as [a; b; …], for messages. *)

val mask_shape : string -> ('v, 'r, 'd) Nx.t -> unit
(** [mask_shape fn m] raises unless [m]'s axes are [[…; h; w; 1]]. *)

type extension =
  [ `Zero | `Replicate | `Symmetric_whole | `Symmetric_half | `Periodic ]
(** {!Mynd.extension}. *)

val check_extension : string -> edge:Nx.extension -> ('v, 'r, 'd) Nx.t -> unit
(** [check_extension fn ~edge x] raises where [x] has no row or no column and
    [edge] reads past its edge, every rule but [`Zero]. *)

val narrow : ('v, 'r, 'd) Nx.t -> bool
(** [narrow x] is [true] where [x]'s dtype is narrower than 32 bits. *)
