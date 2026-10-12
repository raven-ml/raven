(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let buffers x = fst (Nx.shards x)

let same b b' =
  Rig.Buffer.overlaps b b' && Rig.Buffer.offset b = Rig.Buffer.offset b'

let storage =
  let pp ppf bs =
    Format.fprintf ppf "[%s]"
      (String.concat "; "
         (List.map
            (fun b ->
              Printf.sprintf "%s at %d" (Rig.name (Rig.Buffer.device b))
                (Rig.Buffer.offset b))
            bs))
  in
  Windtrap.Testable.make ~pp ~equal:(fun bs bs' ->
      List.compare_lengths bs bs' = 0 && List.for_all2 same bs bs')
