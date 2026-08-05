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

type 'p uniform_pair = { first : 'p; second : 'p } [@@deriving ptree]
type dense = { dense_w : Nx.float32_t } [@@deriving ptree ~mirror]

let () =
  let doubled =
    map2_uniform_pair ( + ) { first = 1; second = 2 }
      { first = 10; second = 20 }
  in
  if doubled.second <> 22 then failwith "uniform signature traversal is wrong";
  let names = names_uniform_pair { first = (); second = () } in
  if names.first <> "first" then failwith "uniform signature names are wrong";
  let d = { dense_w = Nx.zeros Nx.float32 [| 2 |] } in
  let restored = of_uniform (to_uniform d) in
  if Nx.numel restored.dense_w <> 2 then failwith "mirror round trip is wrong"
