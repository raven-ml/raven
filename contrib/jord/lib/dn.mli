(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Digital numbers read in their own dtype, for the modules that compare
    them with special values and codes or read their bits. No read casts
    the DNs: a read of a scene's 16-bit band makes no widened copy of it. *)

val mem : string -> float list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
(** [mem fn ns x] is where the DNs [x] equal one of the numbers [ns]: where
    they are NaN for a NaN in [ns], and nowhere for a number that no value of
    [x]'s dtype equals, as [-9999.] in [uint16] or [7.] in [float4_e2m1fn].

    Raises [Invalid_argument] naming [fn] if [x] holds booleans or complex
    numbers. *)

val any : string -> int list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
(** [any fn bits x] is where the DNs [x] set one of the bits at the positions
    [bits], bit 0 the least significant. A bit at or past the width of [x]'s
    dtype is not set; the top bit of a signed dtype is its sign.

    Raises [Invalid_argument] naming [fn] if [x]'s dtype is not an integer
    dtype. *)

val all : string -> int list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
(** [all fn bits x] is where the DNs [x] set every bit of [bits], read as
    {!any} reads them. *)

val field :
  string ->
  offset:int ->
  length:int ->
  int list ->
  ('v, 's, 'd) Nx.t ->
  'd Nx.bool_t
(** [field fn ~offset ~length vs x] is where the field of [length] bits at
    [offset] of the DNs [x] holds one of the numbers [vs], its bits read as
    {!any} reads them: [x >> offset & (2{^length} - 1)]. *)
