(* An interpreter that names every operation and every load. *)

let rule : type r. Nx.Prim.interpretation -> by:string -> r Nx.Prim.t -> r =
 fun _ ~by op ->
  match[@warning "@4@8"] op with
  | Map _ | Reduce _ | Scan _ | Gather _ | Scatter _ | Assemble _ | Contract _
  | Copy _ | Move _ | Bitcast _ | Place _ | Check _ ->
      Nx.Prim.eval ~by op

let load : type d. d Nx.Prim.load -> unit =
 fun l -> match[@warning "@4@8"] l with Plain _ -> ()
