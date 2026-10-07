(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ymir_units

let parsec = Unit.(int 648000 / pi * astronomical_unit)
let julian_year = Unit.(int 31557600 * second)
