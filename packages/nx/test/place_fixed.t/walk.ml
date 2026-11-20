(* A structure of fixed arrays only, and one with positions of a parameter. *)

type 'd steps = { count : 'd Nx.int32_t }

let walk_steps c x =
  let open Nx.Ptree.Walk in
  { count = field c "count" tensor x.count }

type 'a linear = { w : 'a; b : 'a option }

let walk_linear c { w; b } =
  let open Nx.Ptree.Walk in
  let w = field c "w" leaf w in
  let b = field c "b" (option leaf) b in
  { w; b }
