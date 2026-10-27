(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Contractions: two-operand patterns and matrix products lowered to one
    {!Value.Contract} between movements, and patterns of more operands to a
    tree of them. Each function raises [Invalid_argument] naming [by] as
    [Nx.contract], [Nx.einsum], [Nx.matmul] and [Nx.dot] state. *)

val contract :
  by:string ->
  ?sizes:(string * int) list ->
  ?acc:Nx_array.Dtype.any ->
  ?init:('v, 's, 'd) Value.t ->
  ('v, 's) Nx_array.Dtype.t ->
  Pattern.t ->
  ('a, 'b, 'd) Value.t ->
  ('c, 'e, 'd) Value.t ->
  ('v, 's, 'd) Value.t

val matmul :
  by:string ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t

val along :
  by:string ->
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t
(** [along ~by ~batch ~contracting a b] is [a] and [b] contracted along the
    axis pairs [batch] and [contracting], each an axis of [a] and one of [b]
    of equal extent, at [a]'s dtype and accumulated as {!matmul} does. The
    result's axes are the batch axes, then [a]'s others, then [b]'s, each in
    its operand's order. *)

val dot :
  by:string ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t

val einsum :
  by:string ->
  ?sizes:(string * int) list ->
  ?acc:Nx_array.Dtype.any ->
  Pattern.t ->
  ('v, 's, 'd) Value.t list ->
  ('v, 's, 'd) Value.t

val chain :
  by:string -> ('v, 's, 'd) Value.t list -> ('v, 's, 'd) Value.t
(** [chain ~by xs] is [Nx.multi_dot xs], raising naming [by] as it states. *)
