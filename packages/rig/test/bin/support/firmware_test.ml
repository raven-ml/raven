(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_bin

(* BLAKE2b-256 of "rig test fixture\n", computed with Python's hashlib. *)
let digest = "189295c8d51796727e56d48b35301d1066c3401003e207870d60d820f98c7b24"

let images =
  [ "test/missing-before.bin"; "test/image.bin"; "test/missing-after.bin" ]
  |> List.map (fun path ->
      Printf.sprintf "%s\t%s\thttps://fixtures.invalid/-/raw/pinned/%s" path
        digest path)
  |> String.concat "\n"

let () =
  Firmware.fetch images
    (if Array.length Sys.argv > 1 then Some Sys.argv.(1) else None)
