open Tolk_next

let name u =
  match (Ops.op u, Ops.arg u) with
  | Param, Param { name = Some name; vmin_vmax = Some _; _ } -> Some name
  | Special, String name -> Some name
  | Range, _ -> Some ("r" ^ Ops.range_str u)
  | _ -> None

let is_invalid = function `Invalid -> true | #Dtype.value -> false

let leaf vars u =
  match (Option.bind (name u) (fun n -> List.assoc_opt n vars), Ops.arg u) with
  | Some v, _ -> v
  | None, Param { bound = Some v; _ } -> v
  | None, _ ->
      invalid_arg
        (Printf.sprintf "%s has no value"
           (Option.value (name u) ~default:"a leaf"))

let weak u = List.mem (Ops.dtype u) Dtype.weaks

(* [held ~check dt v] is [v] as [dt] holds it: converted, then wrapped to [dt]'s
   width, after [check dt] has seen the exact value. *)
let held ~check dt (v : Dtype.value) : Dtype.const =
  match Dtype.const dt v with
  | #Dtype.value as v ->
      check dt v;
      (Dtype.truncate dt v :> Dtype.const)
  | `Invalid -> `Invalid

(* The values of [u]'s operands as compiled code reads them: a weak integer
   operand of an operation on a committed integer type is committed to that
   type, as Uop_weak commits it, and wraps. A selection's condition is no
   operand. *)
let operands ~check values u =
  let src = Ops.src u in
  let operands = if Ops.op u = Op.Where then List.tl src else src in
  let peer =
    List.find_map
      (fun s ->
        let dt = Ops.dtype s in
        if Dtype.is_int dt && not (weak s) then Some dt else None)
      operands
  in
  List.map
    (fun s ->
      match (peer, Ops.Tbl.find values s) with
      | Some dt, (`Int _ as v) when weak s && List.memq s operands ->
          held ~check dt v
      | _, v -> v)
    src

(* [node ~check] computes a node from its sources' values, [check] seeing each
   value that an operation, a cast or a commitment gives a type before it is
   wrapped to it. *)
let node ~check vars params values u =
  match (Ops.op u, Ops.arg u, operands ~check values u) with
  | Const, Const c, [] -> c
  | (Param | Range | Special), _, _ when Option.is_some (name u) ->
      (leaf vars u :> Dtype.const)
  | Param, Param { slot; size = None; _ }, [] -> (
      match List.assoc_opt slot params with
      | Some v -> (v :> Dtype.const)
      | None -> invalid_arg (Printf.sprintf "parameter %d has no value" slot))
  | Where, _, `Invalid :: _ -> `Invalid
  | op, _, src when op <> Op.Where && List.exists is_invalid src -> `Invalid
  | Cast, _, [ (#Dtype.value as v) ] -> held ~check (Ops.dtype u) v
  | Bitcast, _, [ (#Dtype.value as v) ] ->
      (Dtype.bitcast (Ops.dtype (Ops.nth u 0)) (Ops.dtype u) v :> Dtype.const)
  | op, _, src when Op.Set.mem op Op.Set.alu -> (
      let dt = Ops.dtype u in
      match Ops.exec_alu ~truncate_output:false op dt src with
      | #Dtype.value as v -> held ~check dt v
      | `Invalid -> `Invalid)
  | op, _, _ -> invalid_arg (Format.asprintf "cannot evaluate %a" Op.pp op)

let fold ~check ~vars ~params u =
  let values = Ops.Tbl.create 64 in
  List.iter
    (fun n -> Ops.Tbl.replace values n (node ~check vars params values n))
    (Ops.toposort u);
  Ops.Tbl.find values u

let eval ?(vars = []) ?(params = []) u =
  fold ~check:(fun _ _ -> ()) ~vars ~params u

exception Overflow

let overflows ?(vars = []) ?(params = []) u =
  let check dt (v : Dtype.value) =
    match v with
    | `Int z when Dtype.is_int dt && dt <> Dtype.Weak_int ->
        let lo, hi = Dtypes.int_bounds dt in
        if Z.lt z lo || Z.gt z hi then raise Overflow
    | _ -> ()
  in
  match fold ~check ~vars ~params u with
  | _ -> false
  | exception Overflow -> true
