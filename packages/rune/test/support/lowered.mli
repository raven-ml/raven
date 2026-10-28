(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Operations lowered to tolk, compiled and run on the host. *)

val run : (unit -> ('v, 's, 'd) Nx.t) -> 'v array
(** [run f] is the elements, in C order, of [f ()] computed by lowering each
    operation [f] applies ({!Rune2.Lower.op}) into one tolk graph, compiled and
    run on the host. A value [f] did not compute is bound as storage that holds
    its elements, read through its layout where it is a view of an array's
    storage. *)
