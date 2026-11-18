(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type polarisation = HH | HV | VH | VV

let name = function HH -> "HH" | HV -> "HV" | VH -> "VH" | VV -> "VV"

let of_name = function
  | "HH" -> Some HH
  | "HV" -> Some HV
  | "VH" -> Some VH
  | "VV" -> Some VV
  | _ -> None
