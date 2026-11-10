(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Host arrays as flat bigarrays, for the loops that read tiles and pack
    batches. *)

type ('v, 's) t = ('v, 's, Bigarray.c_layout) Bigarray.Array1.t

val read : ('v, 's, Nx.host) Nx.t -> (('v, 's) t -> 'a) -> 'a
(** [read x f] is [f b], [b] [x]'s elements in C order, over [x]'s memory when
    [x] is C-contiguous and over a copy otherwise. [f] must not write through
    [b] or keep it. *)

val create : ('v, 's) Bigarray.kind -> int -> 'v -> ('v, 's) t
(** [create k n v] is a fresh bigarray of [n] elements [v]. *)

val wrap :
  ('v, 's) Nx.dtype -> int array -> ('v, 's) t -> ('v, 's, Nx.host) Nx.t
(** [wrap dt shape b] is the host value of [shape] over [b]'s memory, which it
    owns: the caller must not use [b] after the call. *)
