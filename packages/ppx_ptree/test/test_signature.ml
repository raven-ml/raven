(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { weight : Nx.float32_t; label : string [@ptree.ignore] }
[@@deriving ptree]

type 'tag wrapped = {
  wrapped_weight : Nx.float32_t;
  wrapped_tag : 'tag; [@ptree.ignore]
}
[@@deriving ptree]

let () =
  let value = { weight = Nx.zeros Nx.float32 [| 1 |]; label = "signature" } in
  let visited = ref false in
  iter
    (fun tensor ->
      Stdlib.ignore tensor;
      visited := true)
    value;
  if not !visited then failwith "abstract signature traversal did not run"
