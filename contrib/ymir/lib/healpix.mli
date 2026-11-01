(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** HEALPix pixels and spherical harmonics: the implementation of
    {!Ymir.Healpix}. *)

type scheme = Ring | Nested
type 'f t

val v : 'f Frame.t -> nside:int -> scheme -> 'f t
val frame : 'f t -> 'f Frame.t
val nside : 'f t -> int
val scheme : 'f t -> scheme
val pixels : 'f t -> int
val area : 'f t -> (float, Nx.float64_elt, Nx.host) Nx.t Ymir_units.Quantity.t
val pixel : 'f t -> ('f, 'd) Direction.t -> 'd Nx.int64_t
val centre : 'f t -> 'd Nx.int64_t -> ('f, 'd) Direction.t
val corners : 'f t -> 'd Nx.int64_t -> ('f, 'd) Direction.t
val neighbours : 'f t -> 'd Nx.int64_t -> 'd Nx.int64_t
val renumber : 'f t -> scheme -> 'd Nx.int64_t -> 'd Nx.int64_t

val analysis :
  'f t ->
  lmax:int ->
  iterations:int ->
  (float, Nx.float64_elt, 'd) Nx.t ->
  (Complex.t, Nx.complex64_elt, 'd) Nx.t

val synthesis :
  'f t ->
  (Complex.t, Nx.complex64_elt, 'd) Nx.t ->
  (float, Nx.float64_elt, 'd) Nx.t
