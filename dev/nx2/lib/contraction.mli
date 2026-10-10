(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Contractions: two-operand patterns and matrix products lowered to one
    {!Value.Contract} between movements. Each function raises [Invalid_argument]
    naming [by] as [Nx.contract] and [Nx.matmul] state. *)

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
