(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tables have no columns yet: the column tranche replaces this file. *)
type t = |

let schema : t -> Schema.t = function _ -> .
let rows : t -> int = function _ -> .
let equal : t -> t -> bool = function _ -> .
