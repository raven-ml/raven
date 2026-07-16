(* Tests for Divandmod rewrite rules. *)

open Windtrap
open Tolk
open Tolk_uop

let rewrite u = Upat.Pattern_matcher.rewrite Divandmod.div_and_mod_symbolic u

(* Index expressions are dtype [Dtype.index]; the matcher keys on it. *)
let var ?(dtype = Dtype.index) ~name ~lo ~hi () =
  Uop.variable ~name ~min_val:lo ~max_val:hi ~dtype ()

let ic n = Uop.const (Const.int Dtype.index n)

let floordiv lhs rhs = Uop.alu_binary ~op:Ops.Floordiv ~lhs ~rhs
let floormod lhs rhs = Uop.alu_binary ~op:Ops.Floormod ~lhs ~rhs

let const_product_value u =
  match Uop.const_int_value u with
  | Some n -> Some n
  | None when Uop.op u = Ops.Mul && Array.length (Uop.src u) = 2 ->
      let src = Uop.src u in
      (match Uop.const_int_value src.(0), Uop.const_int_value src.(1) with
       | Some a, Some b -> Some (a * b)
       | _ -> None)
  | None -> None

let positive_floor_div_does_not_rewrite_without_structure () =
  let x = var ~name:"x" ~lo:0 ~hi:100 () in
  let d = ic 5 in
  let e = floordiv x d in
  is_true ~msg:"no rewrite with plain positive divisor" (rewrite e = None)

(* Rule 1: (x // c + a) // d  ->  (x + a*c) // (c*d) *)
let nested_div_fires () =
  let x = var ~name:"x" ~lo:0 ~hi:100 () in
  let c = ic 2 in
  let a = ic 3 in
  let d = ic 4 in
  let inner = floordiv x c in
  let sum = Uop.alu_binary ~op:Ops.Add ~lhs:inner ~rhs:a in
  let e = floordiv sum d in
  match rewrite e with
  | Some r ->
      is_true ~msg:"rewrites to Floordiv" (Uop.op r = Ops.Floordiv)
  | None ->
      is_true ~msg:"rule fired" false

let nested_div_accepts_negative_inner_divisor () =
  let x = var ~name:"x" ~lo:(-100) ~hi:100 () in
  let c = ic (-2) in
  let a = ic 3 in
  let d = ic 4 in
  let e = floordiv (Uop.O.(floordiv x c + a)) d in
  match rewrite e with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"rewrites to Floordiv with negative combined divisor"
        (Uop.op r = Ops.Floordiv
         && Array.length src = 2
         && const_product_value src.(1) = Some (-8))
  | None ->
      is_true ~msg:"negative inner divisor rule fired" false

let add_const_div_fires_for_negative_constant () =
  let x = var ~name:"x" ~lo:(-10) ~hi:10 () in
  let c = ic (-3) in
  let d = ic 4 in
  let n = Uop.alu_binary ~op:Ops.Add ~lhs:x ~rhs:c in
  let e = floordiv n d in
  match rewrite e with
  | Some r -> is_true ~msg:"rewrites to Add" (Uop.op r = Ops.Add)
  | None -> is_true ~msg:"rule fired" false

let remove_nested_floormod_fires () =
  let x = var ~name:"x" ~lo:(-10) ~hi:10 () in
  let y = var ~name:"y" ~lo:(-10) ~hi:10 () in
  let inner = floormod x (ic 4) in
  let sum = Uop.alu_binary ~op:Ops.Add ~lhs:inner ~rhs:y in
  let e = floormod sum (ic 2) in
  match rewrite e with
  | Some r -> is_true ~msg:"rewrites to Floormod" (Uop.op r = Ops.Floormod)
  | None -> is_true ~msg:"rule fired" false

let crossing_denominator_does_not_fold_zero_singleton () =
  let x = var ~name:"x" ~lo:0 ~hi:0 () in
  let y = var ~name:"y" ~lo:(-5_000_000_000) ~hi:5_000_000_000 () in
  let e = floordiv x y in
  is_true ~msg:"zero-crossing denominator does not fold" (rewrite e = None)

let zero_denominator_raises_before_sentinel_bailout () =
  let x = var ~name:"x" ~lo:min_int ~hi:max_int () in
  let e = floordiv x (ic 0) in
  raises ~msg:"zero denominator is checked before sentinel bounds"
    Division_by_zero (fun () -> ignore (rewrite e))

let singleton_quotient_floordiv_folds () =
  let x = var ~name:"x" ~lo:10 ~hi:14 () in
  let d = ic 5 in
  let e = floordiv x d in
  match rewrite e with
  | Some r ->
      is_true ~msg:"singleton quotient folds to const 2"
        (Uop.const_int_value r = Some 2)
  | None -> is_true ~msg:"singleton quotient rule fired" false

let singleton_quotient_floormod_folds () =
  let x = var ~name:"x" ~lo:10 ~hi:14 () in
  let d = ic 5 in
  let e = floormod x d in
  match rewrite e with
  | Some r -> is_true ~msg:"singleton quotient mod folds to Sub" (Uop.op r = Ops.Sub)
  | None -> is_true ~msg:"singleton quotient mod rule fired" false

(* A divisor bounded below but unbounded above still cancels when the
   numerator stays inside a single quotient band. *)
let cancel_one_sided_bounded_divisor_div () =
  let x = var ~name:"cx" ~lo:0 ~hi:2 () in
  let y = var ~name:"cy" ~lo:3 ~hi:max_int () in
  match rewrite (floordiv x y) with
  | Some r ->
      is_true ~msg:"one-sided-bounded divisor folds div to const 0"
        (Uop.const_int_value r = Some 0)
  | None -> is_true ~msg:"cancel fired for one-sided-bounded divisor (div)" false

let cancel_one_sided_bounded_divisor_mod () =
  let x = var ~name:"cx" ~lo:0 ~hi:2 () in
  let y = var ~name:"cy" ~lo:3 ~hi:max_int () in
  match rewrite (floormod x y) with
  | Some r ->
      is_true ~msg:"one-sided-bounded divisor folds mod to a subtraction"
        (Uop.op r = Ops.Sub)
  | None -> is_true ~msg:"cancel fired for one-sided-bounded divisor (mod)" false

(* (a % 12) % 3 -> a % 3 (remove_nested_mod on a single term). *)
let nested_single_term_mod_folds () =
  let a = var ~name:"nm_a" ~lo:0 ~hi:99 () in
  let e = floormod (floormod a (ic 12)) (ic 3) in
  match rewrite e with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"(a % 12) % 3 folds to a % 3"
        (Uop.op r = Ops.Floormod
         && Array.length src = 2
         && Uop.equal src.(0) a
         && Uop.const_int_value src.(1) = Some 3)
  | None -> is_true ~msg:"nested single-term mod fired" false

(* (a % 12) // 3 -> (a // 3) % 4 (nested_div). *)
let nested_single_term_div_folds () =
  let a = var ~name:"nd_a" ~lo:0 ~hi:99 () in
  let e = floordiv (floormod a (ic 12)) (ic 3) in
  match rewrite e with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"(a % 12) // 3 folds to (a // 3) % 4"
        (Uop.op r = Ops.Floormod
         && Array.length src = 2
         && Uop.op src.(0) = Ops.Floordiv
         && Uop.const_int_value src.(1) = Some 4)
  | None -> is_true ~msg:"nested single-term div fired" false

let symbolic_gcd_divides_variable_denominator_div () =
  let a = var ~name:"a" ~lo:1 ~hi:10 () in
  let b = var ~name:"b" ~lo:1 ~hi:10 () in
  let c = var ~name:"c" ~lo:1 ~hi:10 () in
  let d = var ~name:"d" ~lo:1 ~hi:10 () in
  let x = Uop.O.((a * b) + (a * c)) in
  let y = Uop.O.(a * d) in
  match rewrite (floordiv x y) with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"common symbolic factor is cancelled for div"
        (Uop.op r = Ops.Floordiv
         && Array.length src = 2
         && Uop.op src.(0) = Ops.Add
         && Uop.equal src.(1) d)
  | None -> is_true ~msg:"symbolic gcd div rule fired" false

let symbolic_gcd_divides_variable_denominator_mod () =
  let a = var ~name:"a" ~lo:1 ~hi:10 () in
  let b = var ~name:"b" ~lo:1 ~hi:10 () in
  let c = var ~name:"c" ~lo:1 ~hi:10 () in
  let d = var ~name:"d" ~lo:1 ~hi:10 () in
  let x = Uop.O.((a * b) + (a * c)) in
  let y = Uop.O.(a * d) in
  match rewrite (floormod x y) with
  | Some r ->
      is_true ~msg:"common symbolic factor is restored around mod"
        (Uop.op r = Ops.Mul)
  | None -> is_true ~msg:"symbolic gcd mod rule fired" false

let symbolic_gcd_divides_mixed_constant_factor () =
  let a = var ~name:"a" ~lo:1 ~hi:10 () in
  let b = var ~name:"b" ~lo:1 ~hi:10 () in
  let c = var ~name:"c" ~lo:1 ~hi:10 () in
  let d = var ~name:"d" ~lo:1 ~hi:10 () in
  let two = ic 2 and four = ic 4 in
  let x = Uop.O.((two * (a * b)) + (four * (a * c))) in
  let y = Uop.O.(two * (a * d)) in
  match rewrite (floordiv x y) with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"common symbolic and constant factors cancel"
        (Uop.op r = Ops.Floordiv
         && Array.length src = 2
         && Uop.equal src.(1) d)
  | None -> is_true ~msg:"mixed symbolic gcd div rule fired" false

let factor_remainder_expr ~den_lo ~den_hi op =
  let d = var ~name:"d" ~lo:den_lo ~hi:den_hi () in
  let q = var ~name:"q" ~lo:0 ~hi:10 () in
  let x = Uop.O.((d * q) + ic 100) in
  Uop.alu_binary ~op ~lhs:x ~rhs:d

let factor_remainder_rejects_negative_denominator_range_for_div () =
  let e = factor_remainder_expr ~den_lo:(-2) ~den_hi:3 Ops.Floordiv in
  is_true ~msg:"negative denominator range blocks factor_remainder"
    (rewrite e = None)

let factor_remainder_rejects_negative_denominator_range_for_mod () =
  let e = factor_remainder_expr ~den_lo:(-2) ~den_hi:3 Ops.Floormod in
  is_true ~msg:"negative denominator range blocks factor_remainder"
    (rewrite e = None)

let factor_remainder_still_accepts_positive_denominator_range () =
  let e = factor_remainder_expr ~den_lo:2 ~den_hi:5 Ops.Floordiv in
  match rewrite e with
  | Some r ->
      is_true
        ~msg:(Format.asprintf "positive denominator range rewrites, got %a" Uop.pp r)
        (Uop.op r = Ops.Add)
  | None ->
      is_true ~msg:"positive denominator range still rewrites" false

let factor_remainder_floormod_splits_constant_factor_without_exact_quotient () =
  let a = var ~name:"a" ~lo:0 ~hi:100 () in
  let b = var ~name:"b" ~lo:0 ~hi:100 () in
  let x = Uop.O.((ic 3 * a) + b) in
  let e = floormod x (ic 2) in
  match rewrite e with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"mod split keeps a zero quotient marker"
        (Uop.op r = Ops.Floormod
         && Array.length src = 2
         && Uop.op src.(0) = Ops.Add)
  | None ->
      is_true ~msg:"factor_remainder modulo split rewrites" false

let factor_remainder_preserves_remainder_order () =
  let a = var ~name:"a" ~lo:0 ~hi:100 () in
  let b = var ~name:"b" ~lo:0 ~hi:100 () in
  let x = Uop.O.((ic 3 * a) + b) in
  let e = floormod x (ic 2) in
  match rewrite e with
  | Some r ->
      let src = Uop.src r in
      let rem_terms =
        if Uop.op r = Ops.Floormod && Array.length src = 2
        then Uop.split_uop src.(0) Ops.Add
        else []
      in
      (match rem_terms with
       | [ first; second ] ->
           is_true ~msg:"remainder terms stay in numerator order"
             ((not (Uop.equal first b)) && Uop.equal second b)
       | _ ->
           is_true
             ~msg:
               (Format.asprintf "expected two remainder terms, got %a" Uop.pp r)
             false)
  | None ->
      is_true ~msg:"factor_remainder order case rewrites" false

let factor_remainder_floormod_splits_multiple_constant_factors () =
  let a = var ~name:"a" ~lo:0 ~hi:100 () in
  let b = var ~name:"b" ~lo:0 ~hi:100 () in
  let c = var ~name:"c" ~lo:0 ~hi:100 () in
  let x = Uop.O.((ic 3 * a) + (ic 5 * b) + c) in
  let e = floormod x (ic 2) in
  match rewrite e with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"multiple split factors still rewrite to floormod"
        (Uop.op r = Ops.Floormod
         && Array.length src = 2
         && Uop.op src.(0) = Ops.Add)
  | None ->
      is_true ~msg:"factor_remainder multiple modulo split rewrites" false

let large_constant_residue_double_does_not_overflow_rewrite () =
  let a = var ~name:"a" ~lo:0 ~hi:10 () in
  let coeff = (max_int / 2) + 1 in
  let x = Uop.O.(ic coeff * a) in
  let e = floordiv x (ic max_int) in
  is_true ~msg:"overflowing residue proof is rejected" (rewrite e = None)

let nest_by_factor_accepts_stack_numerator () =
  let a = var ~name:"a" ~lo:(-10) ~hi:10 () in
  let b = var ~name:"b" ~lo:(-10) ~hi:10 () in
  let two = ic 2 in
  let x = Uop.stack [ Uop.O.(two * a); Uop.O.(two * b) ] in
  let e = floordiv x (ic 4) in
  match rewrite e with
  | Some r ->
      let src = Uop.src r in
      is_true ~msg:"stack numerator participates in nest_by_factor"
        (Uop.op r = Ops.Floordiv
         && Array.length src = 2
         && Uop.const_int_value src.(1) = Some 2)
  | None ->
      is_true ~msg:"stack numerator nest_by_factor rewrites" false

(* Property test: every fold the matcher makes must agree numerically with
   the unfolded expression over the whole declared range, under floor
   division and modulo. *)

let floor_div a b =
  let q = a / b and r = a mod b in
  if r <> 0 && (r < 0) <> (b < 0) then q - 1 else q

let floor_mod a b = a - (floor_div a b * b)

let rec eval env u =
  match Uop.op u with
  | Ops.Const ->
      (match Uop.const_int_value u with
       | Some n -> n
       | None -> failwith "eval: non-integer const")
  | Ops.Add -> eval env (Uop.src u).(0) + eval env (Uop.src u).(1)
  | Ops.Sub -> eval env (Uop.src u).(0) - eval env (Uop.src u).(1)
  | Ops.Mul -> eval env (Uop.src u).(0) * eval env (Uop.src u).(1)
  | Ops.Floordiv -> floor_div (eval env (Uop.src u).(0)) (eval env (Uop.src u).(1))
  | Ops.Floormod -> floor_mod (eval env (Uop.src u).(0)) (eval env (Uop.src u).(1))
  | _ ->
      (match List.find_opt (fun (v, _) -> Uop.equal v u) env with
       | Some (_, n) -> n
       | None -> failwith "eval: unbound leaf")

let fresh_name =
  let c = ref 0 in
  fun () ->
    incr c;
    Printf.sprintf "p%d" !c

(* One numerator term: coeff * (var or var % m). Returns the term node and
   the underlying (var, lo, hi) so the driver can enumerate it. *)
let gen_term () =
  let lo = Random.int 6 - 3 in
  let hi = lo + Random.int 5 in
  let v = var ~name:(fresh_name ()) ~lo ~hi () in
  let base = if Random.bool () then floormod v (ic (2 + Random.int 5)) else v in
  let coeff = match Random.int 9 - 4 with 0 -> 1 | c -> c in
  let term = if coeff = 1 then base else Uop.O.(base * ic coeff) in
  (term, (v, lo, hi))

let gen_divisor () =
  if Random.int 3 <> 0 then (ic (1 + Random.int 6), None)
  else
    let lo = 1 + Random.int 8 in
    let hi = if Random.int 3 = 0 then max_int else lo + Random.int 5 in
    let v = var ~name:(fresh_name ()) ~lo ~hi () in
    (v, Some (v, lo, hi))

(* Enumerate the cartesian product of the leaves over their (clamped)
   ranges and check [eval result = pyf (eval num) (eval div)] everywhere.
   Clamping to a sub-range is sound: a valid fold holds over the whole
   declared range, so it must hold over any subset, and it keeps the
   enumeration finite when a bound is the unbounded sentinel. *)
let fold_matches_numerically leaves num div r pyf =
  let rec loop env = function
    | [] ->
        let expected = pyf (eval env num) (eval env div) in
        eval env r = expected
    | (v, lo, hi) :: rest ->
        let hi = min hi (lo + 4) in
        let ok = ref true in
        let x = ref lo in
        while !ok && !x <= hi do
          ok := loop ((v, !x) :: env) rest;
          incr x
        done;
        !ok
  in
  loop [] leaves

let one_random_fold_is_correct () =
  let nterms = 1 + Random.int 3 in
  let terms = List.init nterms (fun _ -> gen_term ()) in
  let leaves = List.map snd terms in
  let base_sum =
    match List.map fst terms with
    | [] -> ic 0
    | t :: ts -> List.fold_left (fun acc t -> Uop.O.(acc + t)) t ts
  in
  let const = Random.int 11 - 5 in
  let num = if const = 0 then base_sum else Uop.O.(base_sum + ic const) in
  let div, div_leaf = gen_divisor () in
  let leaves =
    match div_leaf with Some l -> leaves @ [ l ] | None -> leaves
  in
  let check mk pyf =
    match rewrite (mk num div) with
    | None -> true
    | Some r -> fold_matches_numerically leaves num div r pyf
  in
  check floordiv floor_div && check floormod floor_mod

let property_folds_are_numerically_correct () =
  Random.init 0x5eed;
  let cases = 500 in
  let failures = ref 0 in
  for _ = 1 to cases do
    if not (one_random_fold_is_correct ()) then incr failures
  done;
  is_true
    ~msg:(Printf.sprintf "%d/%d random div/mod folds disagreed numerically"
            !failures cases)
    (!failures = 0)

let () =
  run "tolk.uop.divandmod"
    [
      group "fast rules"
        [
          test "plain positive divisor does not rewrite"
            positive_floor_div_does_not_rewrite_without_structure;
          test "nested div fires" nested_div_fires;
          test "nested div accepts negative inner divisor"
            nested_div_accepts_negative_inner_divisor;
          test "add const div fires for negative constant"
            add_const_div_fires_for_negative_constant;
          test "remove nested floormod fires" remove_nested_floormod_fires;
          test "crossing denominator does not fold zero singleton"
            crossing_denominator_does_not_fold_zero_singleton;
          test "zero denominator raises before sentinel bailout"
            zero_denominator_raises_before_sentinel_bailout;
          test "singleton quotient Floordiv folds"
            singleton_quotient_floordiv_folds;
          test "singleton quotient Floormod folds"
            singleton_quotient_floormod_folds;
          test "cancel folds one-sided-bounded divisor for div"
            cancel_one_sided_bounded_divisor_div;
          test "cancel folds one-sided-bounded divisor for mod"
            cancel_one_sided_bounded_divisor_mod;
          test "nested single-term mod folds" nested_single_term_mod_folds;
          test "nested single-term div folds" nested_single_term_div_folds;
        ];
      group "slow rules"
        [
          test "symbolic gcd divides variable denominator for div"
            symbolic_gcd_divides_variable_denominator_div;
          test "symbolic gcd divides variable denominator for mod"
            symbolic_gcd_divides_variable_denominator_mod;
          test "symbolic gcd divides mixed constant factor"
            symbolic_gcd_divides_mixed_constant_factor;
          test "factor_remainder rejects negative denominator range for div"
            factor_remainder_rejects_negative_denominator_range_for_div;
          test "factor_remainder rejects negative denominator range for mod"
            factor_remainder_rejects_negative_denominator_range_for_mod;
          test "factor_remainder accepts positive denominator range"
            factor_remainder_still_accepts_positive_denominator_range;
          test "factor_remainder floormod splits without exact quotient"
            factor_remainder_floormod_splits_constant_factor_without_exact_quotient;
          test "factor_remainder preserves remainder order"
            factor_remainder_preserves_remainder_order;
          test "factor_remainder floormod splits multiple factors"
            factor_remainder_floormod_splits_multiple_constant_factors;
          test "large constant residue proof rejects overflow"
            large_constant_residue_double_does_not_overflow_rewrite;
          test "nest_by_factor accepts stack numerator"
            nest_by_factor_accepts_stack_numerator;
        ];
      group "property"
        [
          test "random folds are numerically correct"
            property_folds_are_numerically_correct;
        ];
    ]
