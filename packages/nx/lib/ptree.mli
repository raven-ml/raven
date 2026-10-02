(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

include
  Ptree_intf.Ptree
    with type ('a, 'b) tensor = ('a, 'b) Value.t
     and type placement = Placement.t
     and type packed = Value.packed
