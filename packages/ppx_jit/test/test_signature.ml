(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The [@jit] attribute rewrites implementations only: [test_signature.mli]
   declares the plain function type and the derived [Params] signature without
   mentioning the attribute. *)

module Params = struct
  type t = { w : Nx.float32_t; b : Nx.float32_t } [@@deriving ptree]
end

let[@jit] scale (p : Params.t) (x : Nx.float32_t) : Nx.float32_t =
  Nx.add (Nx.mul p.w x) p.b
