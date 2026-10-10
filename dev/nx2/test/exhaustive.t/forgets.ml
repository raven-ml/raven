(* An interpreter that forgets [Check]. *)

let rule : type r. Nx.Prim.interpretation -> by:string -> r Nx.Prim.t -> r =
 fun _ ~by op ->
  match[@warning "@4@8"] op with
  | Map _ | Reduce _ | Scan _ | Gather _ | Scatter _ | Sort _ | Assemble _
  | Copy _ | Move _ | Bitcast _
  | Place _ ->
      Nx.Prim.eval ~by op
