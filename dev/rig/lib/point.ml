(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let value_bits = 47
let max_value = (1 lsl value_bits) - 1
let max_index = 65_535
let make index v = (index lsl value_bits) lor v
let index p = p lsr value_bits
let value p = p land max_value
