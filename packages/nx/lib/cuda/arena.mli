(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Workspace a call's plan carves into pieces.

    A plan takes the pieces of a buffer its launches share in order, each
    on a 256-byte boundary, and gives back those it no longer needs: a
    call's launches run in order in one submission, so a piece is free
    once the launches that use it are planned. The buffer then needs the
    most bytes the pieces took at once. *)

type t
(** The type for a workspace's pieces as a plan takes them. *)

val make : unit -> t
(** [make ()] is a workspace of no piece. *)

val take : t -> int -> int
(** [take a n] is the byte where a piece of [n] bytes starts: the first on
    256 bytes past the pieces [a] holds. *)

val mark : t -> int
(** [mark a] is the end of the pieces [a] holds. *)

val release : t -> int -> unit
(** [release a m] gives back the pieces taken since [mark a] was [m]. *)

val peak : t -> int
(** [peak a] is the most bytes [a]'s pieces took at once. *)
