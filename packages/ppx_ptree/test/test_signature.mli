(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t [@@deriving ptree]
type !'tag wrapped [@@deriving ptree]
type 'p uniform_pair = { first : 'p; second : 'p } [@@deriving ptree]
type dense = { dense_w : Nx.float32_t } [@@deriving ptree ~mirror]
