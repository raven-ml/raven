(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Movement-op cleanup rewrites.

    Local rewrites that simplify chains of movement ops without changing the
    value they compute: adjacent reshapes and permutes merge, identity
    reshapes and permutes drop, and stacks that merely regroup the elements of
    one source collapse back to it. Appended to the symbolic simplifier so
    movement chains left by higher passes are canonicalised. *)

val mop_cleanup : Upat.Pattern_matcher.t
(** [mop_cleanup] is the pattern matcher of movement-op cleanup rules. *)
