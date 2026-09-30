open Tolk_next

let value = function
  | #Dtype.value as v -> v
  | `Invalid -> invalid_arg "an invalid value has no value"

let node vars params values u =
  let src = List.map (Ops.Tbl.find values) (Ops.src u) in
  match (Ops.op u, Ops.arg u, src) with
  | Const, Const c, [] -> value c
  | Param, Param { name = Some name; _ }, [] when Ops.is_variable u -> (
      match List.assoc_opt name vars with
      | Some v -> v
      | None -> invalid_arg (Printf.sprintf "variable %s has no value" name))
  | Param, Param { slot; size = None; _ }, [] -> (
      match List.assoc_opt slot params with
      | Some v -> v
      | None -> invalid_arg (Printf.sprintf "parameter %d has no value" slot))
  | Cast, _, [ v ] ->
      let dt = Ops.dtype u in
      Dtype.truncate dt (value (Dtype.const dt v))
  | Bitcast, _, [ v ] -> Dtype.bitcast (Ops.dtype (Ops.nth u 0)) (Ops.dtype u) v
  | op, _, src when Op.Set.mem op Op.Set.alu ->
      value (Ops.exec_alu op (Ops.dtype u) (src :> Dtype.const list))
  | op, _, _ -> invalid_arg (Format.asprintf "cannot evaluate %a" Op.pp op)

let eval ?(vars = []) ?(params = []) u =
  let values = Ops.Tbl.create 64 in
  List.iter
    (fun n -> Ops.Tbl.replace values n (node vars params values n))
    (Ops.toposort u);
  Ops.Tbl.find values u
