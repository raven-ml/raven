(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'd t = {
  modality : Modality.t;
  tokens : 'd Nx.int64_t;
  present : 'd Nx.bool_t option;
}
[@@deriving ptree]
