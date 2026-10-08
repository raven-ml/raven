(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external copy : dst:('v, 's) Nx_array.t -> ('v, 's) Nx_array.t -> int
  = "nx_cpu_copy"

external cast : dst:('w, 'r) Nx_array.t -> ('v, 's) Nx_array.t -> int
  = "nx_cpu_cast"
