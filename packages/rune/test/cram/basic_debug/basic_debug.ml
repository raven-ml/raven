(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx
open Rune

let () =
  let x = Rng.run ~seed:42 (fun () -> randn Float32 [| 2; 3 |]) in
  let y =
    debug
      (fun () ->
        let a = add x x in
        let b = mul a (full Float32 [| 2; 3 |] 2.0) in
        b)
      ()
  in
  Printf.printf "Result shape: [%s]\n"
    (String.concat "," (Array.to_list (Array.map string_of_int (shape y))))
