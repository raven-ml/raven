(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Differentiable interface, in its own module so the engine internals can
   name it. Re-exported (and documented) as [Rune_next.Differentiable]. *)

module type S = sig
  type t

  val map : ('a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) -> t -> t

  val map2 :
    ('a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) -> t -> t -> t

  val iter : ('a 'b. ('a, 'b) Nx.t -> unit) -> t -> unit
end
