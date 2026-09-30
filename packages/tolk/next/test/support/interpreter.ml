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
  | Load, _, v :: _ -> v
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

(* [over eval vars ranges f acc] folds [f] over the bindings of [vars] extended
   by each value of [ranges], the last varying fastest, where [eval] gives the
   value of a range's end under a binding. *)
let rec over eval vars ranges f acc =
  match ranges with
  | [] -> f vars acc
  | r :: rs -> (
      if Ops.op r <> Op.Range then
        invalid_arg "cannot evaluate a loop over a node that is not a range";
      match eval vars (Ops.nth r 0) with
      | `Int n ->
          let name = Option.get (name r) and acc = ref acc in
          for i = 0 to Z.to_int n - 1 do
            acc := over eval ((name, `Int (Z.of_int i)) :: vars) rs f !acc
          done;
          !acc
      | _ -> invalid_arg "a range's end is not an integer")

(* A reduction and an index evaluate their sources themselves: a reduction at
   each value of its ranges, an index without reading its storage. *)
let rec fold ~check ~vars ~params ~buffers u =
  let values = Ops.Tbl.create 64 in
  let rec value n =
    match Ops.Tbl.find_opt values n with
    | Some v -> v
    | None ->
        let v =
          match Ops.op n with
          | Reduce -> reduction ~check ~vars ~params ~buffers n
          | Index -> read ~check ~vars ~params ~buffers n
          | _ ->
              List.iter (fun s -> ignore (value s)) (Ops.src n);
              node ~check vars params values n
        in
        Ops.Tbl.replace values n v;
        v
  in
  value u

and reduction ~check ~vars ~params ~buffers red =
  let op =
    match Ops.arg red with
    | Reduce { op; num_axes = 0 } -> op
    | _ -> invalid_arg "cannot evaluate a reduction of axes"
  in
  let dt = Ops.dtype red in
  let value, ranges =
    match Ops.src red with
    | v :: rs -> (v, rs)
    | [] -> invalid_arg "a reduction has no value"
  in
  let combine vars (acc : Dtype.const) =
    match (acc, fold ~check ~vars ~params ~buffers value) with
    | (#Dtype.value as acc), (#Dtype.value as v) -> (
        match Ops.exec_alu ~truncate_output:false op dt [ acc; v ] with
        | #Dtype.value as v -> held ~check dt v
        | `Invalid -> `Invalid)
    | _ -> `Invalid
  in
  let eval vars u = fold ~check ~vars ~params ~buffers u in
  over eval vars ranges combine (Ops.identity_element op dt)

and read ~check ~vars ~params ~buffers index =
  let eval u = fold ~check ~vars ~params ~buffers u in
  match Ops.src index with
  | [ storage; i ] -> (
      (* A lane of a vector load reads its element past the vector's offset. *)
      let storage, offset, length =
        match (Ops.op storage, Ops.src storage) with
        | Load, [ vector ] when Ops.op vector = Op.Shrink -> (
            match Ops.src vector with
            | [ storage; offset; length ] -> (storage, eval offset, Some length)
            | _ ->
                invalid_arg
                  "a vector load has a storage, an offset and a length")
        | _ -> (storage, `Int Z.zero, None)
      in
      let slot = storage_slot storage in
      let elements =
        match List.assoc_opt slot buffers with
        | Some elements -> elements
        | None -> invalid_arg (Printf.sprintf "buffer %d has no elements" slot)
      in
      match (offset, eval i) with
      | `Invalid, _ | _, `Invalid -> `Invalid
      | `Int o, `Int k -> (
          (match length with
          | Some n when not Z.(geq k zero && lt k (Ops.to_int n)) ->
              invalid_arg
                (Format.asprintf "lane %a is outside a vector of %a" Z.pp_print
                   k Z.pp_print (Ops.to_int n))
          | _ -> ());
          match Z.add o k with
          | e when Z.(geq e zero && lt e (of_int (Array.length elements))) ->
              (elements.(Z.to_int e) :> Dtype.const)
          | e ->
              invalid_arg
                (Format.asprintf "index %a is outside buffer %d" Z.pp_print e
                   slot))
      | _ -> invalid_arg "an index is not an integer")
  | _ -> invalid_arg "cannot evaluate an index by more than one index"

and storage_slot u =
  match Ops.arg u with
  | Param { slot; size = Some _; _ } -> slot
  | _ -> invalid_arg "cannot access a node that is not storage"

let eval ?(vars = []) ?(params = []) ?(buffers = []) u =
  fold ~check:(fun _ _ -> ()) ~vars ~params ~buffers u

exception Overflow

let overflows ?(vars = []) ?(params = []) ?(buffers = []) u =
  let check dt (v : Dtype.value) =
    match v with
    | `Int z when Dtype.is_int dt && dt <> Dtype.Weak_int ->
        let lo, hi = Dtypes.int_bounds dt in
        if Z.lt z lo || Z.gt z hi then raise Overflow
    | _ -> ()
  in
  match fold ~check ~vars ~params ~buffers u with
  | _ -> false
  | exception Overflow -> true

(* Kernels *)

let compare_write (s0, i0, v0) (s1, i1, v1) =
  match Int.compare s0 s1 with
  | 0 -> (
      match Int.compare i0 i1 with 0 -> Dtype.Value.compare v0 v1 | c -> c)
  | c -> c

let writes ?(vars = []) ?(params = []) ?(buffers = []) u =
  let check _ _ = () in
  let eval vars u = fold ~check ~vars ~params ~buffers u in
  let store s =
    let dst, value, gate =
      match Ops.src s with
      | [ dst; value ] -> (dst, value, None)
      | [ dst; value; gate ] -> (dst, value, Some gate)
      | _ -> invalid_arg "a store has a destination, a value and a gate"
    in
    (* A store through a vector writes each lane of a stack past the offset. *)
    let slot, index, lanes =
      match (Ops.op dst, Ops.src dst) with
      | Index, [ storage; index ] -> (storage_slot storage, index, [ value ])
      | Shrink, [ storage; offset; length ]
        when Ops.op value = Op.Stack
             && Z.equal (Ops.to_int length)
                  (Z.of_int (List.length (Ops.src value))) ->
          (storage_slot storage, offset, Ops.src value)
      | Shrink, _ ->
          invalid_arg "a store through a vector stores a stack of its length"
      | _ -> invalid_arg "cannot evaluate a store through more than one index"
    in
    let inside = Ops.ranges s in
    let ranges =
      List.filter
        (fun n -> Ops.op n = Op.Range && Ops.Nodes.mem n inside)
        (Ops.toposort s)
    in
    let write vars acc =
      let opened =
        match gate with
        | None -> true
        | Some g -> Dtype.equal_const (eval vars g) (`Bool true)
      in
      match eval vars index with
      | `Int i when opened ->
          let write (k, acc) lane =
            match eval vars lane with
            | #Dtype.value as v -> (k + 1, (slot, Z.to_int i + k, v) :: acc)
            | `Invalid -> (k + 1, acc)
          in
          snd (List.fold_left write (0, acc) lanes)
      | _ -> acc
    in
    over eval vars ranges write []
  in
  let stores = List.filter (fun n -> Ops.op n = Op.Store) (Ops.toposort u) in
  List.sort_uniq compare_write (List.concat_map store stores)
