(* An interpreter that covers what it forgets with a wildcard. *)

let rule : type r. Nx.Prim.interpretation -> by:string -> r Nx.Prim.t -> r =
 fun _ ~by op ->
  match[@warning "@4@8"] op with
  | Map _ | Copy _ | Move _ -> Nx.Prim.eval ~by op
  | _ -> Nx.Prim.eval ~by op
