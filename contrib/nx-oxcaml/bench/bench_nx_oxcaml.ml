(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let () =
  let at = Nx.Placement.device ~backend:Nx_oxcaml.backend Nx.Device.host in
  Thumper.run "nx_oxcaml" (Bench_nx_common.benchmarks ~place:(Nx.place at))
