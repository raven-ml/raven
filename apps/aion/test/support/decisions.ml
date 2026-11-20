(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let digit_edges = [| -2.5; -1.5; -0.5; 0.5; 1.5; 2.5 |]
let levels = [ 7; 5; 5; 5; 5 ]

(* Token [k]'s digits, the last level's first. *)
let token k =
  snd
    (List.fold_left
       (fun (k, ds) l -> (k / l, ((k mod l) - (l / 2) + 3) :: ds))
       (k, []) levels)

let digits k =
  Array.concat
    (List.map (fun k -> Array.of_list (List.rev (token k))) (Array.to_list k))

