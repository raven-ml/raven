(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let is_const_invalid u =
  match Uop.op u, Uop.arg u with
  | Ops.Const, Uop.Arg.Value c -> Const.view c = Const.Invalid
  | _ -> false

let is_false_const u =
  match Uop.as_const u with
  | Some c ->
      (match Const.view c with Const.Bool false -> true | _ -> false)
  | _ -> false

let is_image_shape u =
  match (try Uop.max_shape u with Invalid_argument _ -> []) with
  | [ _; _; 4 ] -> true
  | _ -> false

let max_numel u =
  match Uop.max_shape u with
  | shape ->
      let size = List.fold_left (fun size n -> Z.mul size (Z.of_int n)) Z.one shape in
      if List.for_all (fun n -> n >= 0) shape && Z.fits_int size then
        Some (Z.to_int size)
      else None
  | exception Invalid_argument _ -> None

let check_oob_enabled () =
  match Sys.getenv_opt "CHECK_OOB" with
  | None -> false
  | Some "" | Some "0" | Some "false" | Some "False" | Some "FALSE" ->
      false
  | Some _ -> true

let interval_empty (lo, hi) = Bound.lt hi lo

let tighten_lower = Bound.max
let tighten_upper = Bound.min

let integer_const u =
  let constant =
    if Bound.equal (Uop.vmin u) (Uop.vmax u) then Uop.as_const u else None
  in
  match constant with
  | Some c ->
      (match Const.view c with
       | Const.Int n when Bound.equal (Uop.vmin u) (`Int n) -> Some (`Int n)
       | Const.Bool b when Bound.equal (Uop.vmin u) (`Bool b) -> Some (`Bool b)
       | Const.Int _ | Const.Bool _ -> None
       | Const.Float _ | Const.Invalid -> None)
  | _ -> None

let refine_cmp_bound idx (lo, hi) lhs rhs =
  match integer_const lhs, integer_const rhs with
  | _, Some n when Uop.equal lhs idx -> lo, tighten_upper hi (Bound.pred n)
  | Some n, _ when Uop.equal rhs idx -> tighten_lower lo (Bound.succ n), hi
  | _ -> lo, hi

let refine_eq_bound idx (lo, hi) lhs rhs =
  match integer_const lhs, integer_const rhs with
  | _, Some n when Uop.equal lhs idx ->
      tighten_lower lo n, tighten_upper hi n
  | Some n, _ when Uop.equal rhs idx ->
      tighten_lower lo n, tighten_upper hi n
  | _ -> lo, hi

let bool_and u = Dtype.is_bool (Uop.dtype u) && Uop.op u = Ops.And

let rec refine_index_bounds_under_gate idx bounds gate =
  match Uop.op gate, Uop.src gate with
  | Ops.And, [| lhs; rhs |] when bool_and gate ->
      let bounds = refine_index_bounds_under_gate idx bounds lhs in
      if interval_empty bounds then bounds
      else refine_index_bounds_under_gate idx bounds rhs
  | Ops.Cmplt, [| lhs; rhs |] -> refine_cmp_bound idx bounds lhs rhs
  | Ops.Cmpeq, [| lhs; rhs |] -> refine_eq_bound idx bounds lhs rhs
  | _ -> bounds

let validate_index_with_gate_bounds size idx gate =
  let lo, hi =
    refine_index_bounds_under_gate idx (Uop.vmin idx, Uop.vmax idx) gate
  in
  interval_empty (lo, hi) || (Bound.le Bound.zero lo && Bound.lt hi size)

(* Reuse weak-integer algebra only after proving that each native operation
   has the same mathematical value. In particular, do not narrow the unsigned
   bounds hull: small C integers may be promoted or stored between operations. *)
let lift_index_proof idx gate =
  let exception Unsupported in
  let require condition = if not condition then raise Unsupported in
  let fits dtype u =
    Dtype.equal dtype Dtype.weakint ||
    (Bound.le (Dtype.min dtype) (Uop.vmin u) &&
     Bound.le (Uop.vmax u) (Dtype.max dtype)) in
  let atom u =
    let lifted = Uop.cast ~src:u ~dtype:Dtype.weakint in
    require (Bound.equal (Uop.vmin u) (Uop.vmin lifted) &&
             Bound.equal (Uop.vmax u) (Uop.vmax lifted));
    lifted in
  let cache = Uop.Tbl.create 32 in
  let rec lift u =
    match Uop.Tbl.find_opt cache u with
    | Some lifted -> lifted
    | None ->
        let dtype = Uop.dtype u in
        let binary op a b = Uop.alu_binary ~op ~lhs:(lift a) ~rhs:(lift b) in
        let shift_count a b =
          ignore (lift b);
          match integer_const b with
          | Some (`Int n) when Z.sign n >= 0 && Z.fits_int n ->
              require (Dtype.equal (Uop.dtype a) Dtype.weakint ||
                       Z.lt n (Z.of_int (Dtype.bitsize (Uop.dtype a))));
              Z.to_int n
          | _ -> raise Unsupported in
        let lifted = match Uop.op u, Uop.src u with
          | Ops.Const, _ when Dtype.is_int dtype || Dtype.is_bool dtype -> u
          | (Ops.And | Ops.Or | Ops.Cmplt | Ops.Cmpne | Ops.Cmpeq as op), [|a; b|]
            when Dtype.is_bool dtype -> binary op a b
          | Ops.Cast, [|src|] when Dtype.is_int dtype && Dtype.is_int (Uop.dtype src) ->
              require (fits dtype src);
              lift src
          | (Ops.Add | Ops.Sub | Ops.Mul as op), [|a; b|] when Dtype.is_int dtype ->
              let result = binary op a b in
              require (fits dtype u);
              result
          | Ops.Shl, [|a; b|] when Dtype.is_int dtype ->
              let value = lift a and count = shift_count a b in
              require (Bound.le Bound.zero (Uop.vmin a) && fits dtype u);
              Uop.alu_binary ~op:Ops.Mul ~lhs:value
                ~rhs:(Uop.const (Const.integer Dtype.weakint (Z.shift_left Z.one count)))
          | Ops.Shr, [|a; b|] when Dtype.is_int dtype ->
              ignore (lift a); ignore (shift_count a b);
              require (Bound.le Bound.zero (Uop.vmin a) && fits dtype u);
              atom u
          | Ops.And, [|a; b|] when Dtype.is_int dtype ->
              ignore (lift a); ignore (lift b);
              require (Bound.le Bound.zero (Uop.vmin a) &&
                       Bound.le Bound.zero (Uop.vmin b) && fits dtype u);
              atom u
          | (Ops.Load | Ops.Range | Ops.Special | Ops.Param | Ops.Buffer), _
            when Dtype.is_int dtype &&
                 (Uop.op u <> Ops.Param && Uop.op u <> Ops.Buffer ||
                  Uop.addrspace u = Some Dtype.Alu) ->
              require (fits dtype u);
              atom u
          | _ -> raise Unsupported in
        Uop.Tbl.add cache u lifted;
        lifted in
  try Some (lift idx, lift gate) with Unsupported -> None

let validate_index_with_symbolic_bounds size idx gate =
  let literal_true proof =
    match Uop.op proof, Uop.as_const proof with
    | Ops.Const, Some c -> Const.view c = Const.Bool true
    | _ -> false in
  let in_bounds idx =
    let zero = Uop.const_int 0 in
    let limit = Uop.const (Const.integer Dtype.weakint (Bound.integer size)) in
    let negative = Uop.alu_binary ~op:Ops.Cmplt ~lhs:idx ~rhs:zero in
    let nonnegative = Uop.alu_binary ~op:Ops.Cmpne ~lhs:negative
        ~rhs:(Uop.const_bool true) in
    let below = Uop.alu_binary ~op:Ops.Cmplt ~lhs:idx ~rhs:limit in
    Uop.alu_binary ~op:Ops.And ~lhs:nonnegative ~rhs:below in
  if not (Dtype.is_int (Uop.dtype idx)) then false
  else
    (* Simplify the proof, not the memory expression: a loaded component can
       be constrained by the gate without changing its storage dependencies. *)
    literal_true (Symbolic.uop_given_valid ~try_simplex:false gate (in_bounds idx)) ||
    match lift_index_proof idx gate with
    | None -> false
    | Some (idx, gate) ->
        let gate = Uop.graph_rewrite ~walk:true (fun node ->
            if Dtype.is_int (Uop.dtype node) then Some (Symbolic.simplify node)
            else None) gate in
        let known_true proof =
          let under_gate proof = literal_true
              (Symbolic.uop_given_valid ~try_simplex:false gate proof
               |> Symbolic.simplify) in
          under_gate proof || under_gate (Symbolic.simplify proof) in
        (* For x=d*q+p, prove that the residual stays on one side of c mod d
           before discarding it. Candidates are existing positive coefficients;
           this changes only the checked proof, never emitted comparisons. *)
        let fold_remainder = Uop.graph_rewrite ~walk:true (fun node ->
            match Uop.op node, Uop.src node with
            | Ops.Cmplt, [|lhs; rhs|] when Dtype.equal (Uop.dtype lhs) Dtype.weakint ->
                (match Uop.const_int_value rhs with
                 | Some c when c > 0 ->
                     let terms = Uop.split_uop lhs Ops.Add in
                     let divisors = List.map Uop.const_factor terms
                         |> List.filter (fun d -> d > 1) |> List.sort_uniq Int.compare in
                     List.find_map (fun divisor ->
                         let scaled, residual = List.partition
                             (fun term -> Uop.const_factor term mod divisor = 0) terms in
                         let residual = Symbolic.simplify (Uop.usum residual) in
                         let interval lo hi = Uop.alu_binary ~op:Ops.And
                             ~lhs:Uop.O.(not_ (residual < Uop.const_int lo))
                             ~rhs:Uop.O.(residual < Uop.const_int hi) in
                         let remainder = c mod divisor in
                         let threshold =
                           if known_true (interval 0 remainder) then Some (c / divisor + 1)
                           else if known_true (interval remainder divisor) then Some (c / divisor)
                           else None in
                         match threshold, Uop.divides (Uop.usum scaled) divisor with
                         | Some threshold, Some quotient ->
                             Some Uop.O.(quotient < Uop.const_int threshold)
                         | _ -> None) divisors
                 | _ -> None)
            | _ -> None) in
        let rec prove proof =
          if literal_true proof then true else
          let next = Symbolic.uop_given_valid ~try_simplex:false gate proof
              |> Symbolic.simplify in
          let next = if literal_true next then next else
              fold_remainder next |> Symbolic.simplify
              |> Symbolic.uop_given_valid ~try_simplex:false gate in
          literal_true next ||
          (* Factoring after one component substitution may expose another
             guarded component. Continue only on a strictly smaller DAG; a
             larger equivalent proof remains conservatively unproved. *)
          (List.length (Uop.toposort next) < List.length (Uop.toposort proof) && prove next) in
        prove (Symbolic.simplify (in_bounds (Symbolic.simplify idx)))

let validate_index ?gate uidx =
  let srcs = Uop.src uidx in
  if Array.length srcs < 2 then true
  else
    let buf = srcs.(0) in
    let idxs = Array.sub srcs 1 (Array.length srcs - 1) |> Array.to_list in
    List.exists is_const_invalid idxs
    || (not (check_oob_enabled ()))
    || is_image_shape buf
    ||
    match gate with
    | Some g when is_false_const g || Bound.equal (Uop.vmax g) Bound.zero -> true
    | _ -> (
        let check_axis size idx =
          (Bound.le Bound.zero (Uop.vmin idx) && Bound.lt (Uop.vmax idx) size)
          ||
          match gate with
          | Some gate ->
              validate_index_with_gate_bounds size idx gate
              || validate_index_with_symbolic_bounds size idx gate
          | None -> false
        in
        match idxs with
        | [ idx ] -> (
            match max_numel buf with
            | None -> false
            | Some size -> check_axis (Bound.int size) idx)
        | _ -> (
            try
              let shape = Uop.shape buf in
              List.length shape <> List.length idxs
              ||
              List.for_all2
                (fun dim idx -> check_axis (Uop.vmin dim) idx)
                shape idxs
            with Invalid_argument _ -> true))

let index_source u =
  match Uop.op u, Uop.src u with
  | Ops.Cast, [| x |] ->
      (match Uop.op x with
       | Ops.Index | Ops.Shrink -> Some x
       | _ -> None)
  | (Ops.Index | Ops.Shrink), _ -> Some u
  | _ -> None

let is_index_source u = Option.is_some (index_source u)

let validate_index_source ?gate u =
  match index_source u with
  | None -> false
  | Some uidx -> validate_index ?gate uidx
