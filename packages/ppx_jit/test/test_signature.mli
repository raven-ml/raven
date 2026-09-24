(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Params : sig
  type t = { w : Nx.float32_t; b : Nx.float32_t } [@@deriving ptree]
end

val scale : Params.t -> Nx.float32_t -> Nx.float32_t
