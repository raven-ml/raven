(* Shape.div_and_mod_symbolic: the rewrites of divisions and remainders that
   shape.mli states, as their goldens record them, and the law that a rewrite
   keeps the value of what it rewrites. *)

open Windtrap
open Tolk

let var = Simplification.var
let i = Simplification.i
let int = Ops.int
let uop = Simplification.uop
let x = var "x" 0 100
let a = var "a" 0 99
let b = var "b" 0 99

(* [rewrite d] is [d] rewritten once at its root, or [None] where no rule
   applies; [once d] is that rewrite, or [d]. *)
let rewrite d = Ops.Pattern_matcher.rewrite Shape.div_and_mod_symbolic () d
let once d = Option.value (rewrite d) ~default:d

(* A golden holds divisions and remainders, each followed by its rewrite. The
   goldens sit in [divandmod/]. *)
let golden name = Filename.concat "divandmod" (name ^ ".golden")
let rewritten ds = Ops.sink (List.concat_map (fun d -> [ d; once d ]) ds)
let rewrites name d = Golden.graph (golden name) (fun () -> rewritten [ d ])

let rec every_other = function
  | d :: _ :: rest -> d :: every_other rest
  | [] | [ _ ] -> []

let recorded name = every_other (Ops.src (Golden.sink (golden name)))

(* (x // c + a) // d *)

let nested_divisions =
  group "rewrites (x // c + a) // d as its goldens state"
    [
      rewrites "merge_nested_divisions" Ops.O.(((x // int 2) + int 3) // int 4);
      rewrites "merge_nested_divisions_by_a_negative_inner_divisor"
        Ops.O.(((var "x" (-100) 100 // int (-2)) + int 3) // int 4);
      rewrites "keep_nested_divisions_of_committed_integers"
        Ops.O.(((var ~dtype:Int32 "x" 0 100 // int 2) + int 3) // int 4);
      rewrites "split_nested_divisions_by_a_negative_divisor"
        Ops.O.(((x // int 2) + int 3) // int (-4));
    ]

(* At w = 2^30, w // 2 + 2^30 is 2^29 + 2^30 and halved 805306368, which int32
   holds; the merged form adds 2^31 to w, which int32 wraps to -2^31, and gives
   (2^30 - 2^31) // 4 = -268435456. *)
let committed_division_stays () =
  let k = 1 lsl 30 in
  let w = var ~dtype:Int32 "w" 0 k in
  let d = Ops.O.(((w // int 2) + int k) // int 2) in
  let ck = 2 * k in
  let merged = Ops.O.((w + int ck) // int 4) in
  let at u = Interpreter.eval ~vars:[ ("w", i k) ] u in
  equal (option uop) None (rewrite d);
  equal ~msg:"the division" Dtypes.const (i 805306368) (at d);
  equal ~msg:"the merged form" Dtypes.const (i (-268435456)) (at merged)

(* (x + c) // d and (x + c) % d *)

let constant_terms =
  group "rewrites (x + c) // d and (x + c) % d as its goldens state"
    [
      rewrites "split_the_constant_out_of_a_division"
        Ops.O.((x + int 7) // int 4);
      rewrites "split_the_constant_out_of_a_remainder"
        Ops.O.((x + int 7) % int 4);
      rewrites "split_a_negative_constant_out_of_a_division"
        Ops.O.((x + int (-3)) // int 4);
      rewrites "split_the_constant_out_of_a_division_by_a_negative_divisor"
        Ops.O.((x + int 7) // int (-4));
      rewrites "keep_a_constant_smaller_than_the_divisor"
        Ops.O.((x + int 3) // int 4);
      rewrites "keep_a_committed_integer_division"
        Ops.O.((var ~dtype:Int32 "x" 0 100 + int 7) // int 4);
    ]

(* x // y with one possible quotient *)

let one_quotient =
  let y = var "y" 3 1_000_000_000_000 in
  group "folds a quotient with one possible value as its goldens state"
    [
      rewrites "fold_a_division_with_one_quotient"
        Ops.O.(var "x" 10 14 // int 5);
      rewrites "fold_a_remainder_with_one_quotient"
        Ops.O.(var "x" 10 14 % int 5);
      rewrites "fold_a_division_by_a_variable_with_one_quotient"
        Ops.O.(var "x" 0 2 // y);
      rewrites "fold_a_remainder_by_a_variable_with_one_quotient"
        Ops.O.(var "x" 0 2 % y);
    ]

(* x // c and x % c for a positive constant c *)

let constant_divisors =
  let huge = Ops.const (`Int (Bigint.shift_left Bigint.one 100)) in
  let huge_divisor = Ops.const (`Int Bigint.(pred (shift_left one 101))) in
  group "rewrites a division by a positive constant as its goldens state"
    [
      rewrites "nest_the_division_of_a_remainder" Ops.O.(a % int 12 // int 3);
      rewrites "drop_a_nested_remainder" Ops.O.(a % int 12 % int 3);
      rewrites "drop_a_nested_remainder_from_a_sum"
        Ops.O.(((a % int 4) + b) % int 2);
      rewrites "fold_a_remainder_by_congruence"
        Ops.O.(((a * int 5) + int 3) % int 4);
      rewrites "fold_a_division_by_congruence"
        Ops.O.(((a * int 5) + int 3) // int 4);
      rewrites "split_a_constant_factor_out_of_a_remainder"
        Ops.O.(((a * int 3) + b) % int 2);
      rewrites "split_constant_factors_out_of_a_remainder"
        Ops.O.(((a * int 3) + (b * int 5) + x) % int 2);
      rewrites "keep_a_division_of_huge_coefficients_exact"
        Ops.O.(a * huge // huge_divisor);
      rewrites "keep_a_remainder_whose_nested_part_reaches_the_factor"
        Ops.O.(((var "a" 0 5 * int 4) + var "t" 0 4) % int 12);
      rewrites "keep_a_plain_division" Ops.O.(x // int 5);
      rewrites "keep_a_plain_remainder" Ops.O.(x % int 5);
      rewrites "divide_a_common_factor_out_of_a_division"
        Ops.O.(((a * int 2) + int 3) // int 4);
      rewrites "divide_a_common_factor_out_of_a_remainder"
        Ops.O.(((a * int 2) + int 3) % int 4);
      rewrites "nest_a_division_by_a_factor_of_a_term"
        Ops.O.(((a * int 6) + (b * int 2) + int 1) // int 12);
      rewrites "nest_a_remainder_by_a_factor_of_a_term"
        Ops.O.(((a * int 6) + (b * int 2) + int 1) % int 12);
    ]

(* Divisors that are not constants *)

let other_divisors =
  let q = var "q" 0 10 in
  let d = var "d" 2 5 in
  let signed = var "d" (-2) 3 in
  group "rewrites a division by another divisor as its goldens state"
    [
      rewrites "divide_a_common_divisor_out_of_a_division_by_a_variable"
        Ops.O.(((a * int 4) + (b * int 6)) // (q * int 2));
      rewrites "divide_a_common_divisor_out_of_a_remainder_by_a_variable"
        Ops.O.(((a * int 4) + (b * int 6)) % (q * int 2));
      rewrites "take_the_multiples_of_a_variable_divisor_out_of_a_division"
        Ops.O.(((d * q) + int 100) // d);
      rewrites "take_the_multiples_of_a_variable_divisor_out_of_a_remainder"
        Ops.O.(((d * q) + int 100) % d);
      rewrites
        "take_the_multiples_of_a_divisor_that_can_be_zero_out_of_a_division"
        (let d = var "d" 0 5 in
         Ops.O.(((d * q) + int 100) // d));
      rewrites "keep_a_division_by_a_variable_that_can_be_negative"
        Ops.O.(((signed * q) + int 100) // signed);
      rewrites "keep_a_remainder_by_a_variable_that_can_be_negative"
        Ops.O.(((signed * q) + int 100) % signed);
      rewrites "divide_zero_by_a_divisor_that_can_be_zero"
        Ops.O.(var "x" 0 0 // var "y" (-5_000_000_000) 5_000_000_000);
    ]

(* Divisors that are always 0 *)

let zero_divisors =
  let any =
    Ops.variable "x"
      (`Int (Bigint.of_int min_int))
      (`Int (Bigint.of_int max_int))
  in
  [
    ("x // 0", Ops.O.(x // int 0));
    ("x % 0", Ops.O.(x % int 0));
    ("(x + 7) // 0", Ops.O.((x + int 7) // int 0));
    ("x // z, z in [0, 0]", Ops.O.(x // var "z" 0 0));
    ("(x * 0 + x) // 3", Ops.O.(((x * int 0) + x) // int 3));
    ("x // 0, x of any int", Ops.O.(any // int 0));
  ]

(* A rule that computes with the value of invalid *)

let with_invalid =
  [
    ("(x + invalid) // 3", Ops.O.((x + Ops.invalid) // int 3));
    ("(x + invalid) % 3", Ops.O.((x + Ops.invalid) % int 3));
  ]

(* The goldens recorded from tinygrad's tests, each named after its test. *)
let recorded_tests =
  [
    "test_add_div";
    "test_big_mod";
    "test_div_1";
    "test_div_4";
    "test_div_binary_expression";
    "test_div_by_factor_tie_break";
    "test_div_cancel";
    "test_div_congruence";
    "test_div_congruence_multiple_vars";
    "test_div_const_div";
    "test_div_const_div_neg_inner_divisor";
    "test_div_const_div_wrong_sign";
    "test_div_const_div_wrong_sign_divisor";
    "test_div_div";
    "test_div_drop_small_terms";
    "test_div_into_mod";
    "test_div_min_max";
    "test_div_mod_recombine";
    "test_div_mod_recombine_3level";
    "test_div_mod_recombine_after_asymmetric_fold";
    "test_div_mod_recombine_after_nesting";
    "test_div_mod_recombine_folded_mod";
    "test_div_mod_recombine_in_additive_sum";
    "test_div_mod_recombine_large_coeff";
    "test_div_mod_recombine_merged_quotient";
    "test_div_mod_recombine_negative_div";
    "test_div_mod_recombine_negative_div_unsound";
    "test_div_mod_recombine_partial";
    "test_div_mod_recombine_shifted_quotient";
    "test_div_mod_recombine_with_gcd";
    "test_div_mod_recompose_low_order_remainder";
    "test_divmod_variable_denom_fold_to_const";
    "test_div_neg_all_range";
    "test_div_neg_cancel";
    "test_div_neg_min_max";
    "test_div_neg_rem";
    "test_div_neg_then_neg";
    "test_div_nest_by_factor_with_const";
    "test_div_numerator_negative";
    "test_div_partial_quotient";
    "test_div_reduction";
    "test_div_remove";
    "test_div_symbolic_const_gcd";
    "test_floordiv_const_evaluation";
    "test_floordiv_factor_nest_negative_numerator";
    "test_floordiv_gcd_with_remainder_negative_numerator";
    "test_floordiv_lt";
    "test_floordiv_lt_negative_c";
    "test_floormod_const_evaluation";
    "test_fuzz_failure1";
    "test_fuzz_failure10";
    "test_fuzz_failure11";
    "test_fuzz_failure2";
    "test_fuzz_failure3";
    "test_fuzz_failure4";
    "test_fuzz_failure5";
    "test_fuzz_failure6";
    "test_fuzz_failure7";
    "test_fuzz_failure8";
    "test_fuzz_failure9";
    "test_gcd_with_remainder";
    "test_gcd_with_remainder_negative";
    "test_invalid_gate_simplifies_index";
    "test_invalid_gate_simplifies_vectorize";
    "test_merge_invalid_conditions";
    "test_mod_1";
    "test_mod_4";
    "test_mod_binary_expression";
    "test_mod_cancel";
    "test_mod_congruence";
    "test_mod_congruence_mul_add";
    "test_mod_congruence_multiple_vars";
    "test_mod_congruence_tied_remainder";
    "test_mod_const_reduction_negative_offset";
    "test_mod_div_reorder";
    "test_mod_factor";
    "test_mod_factor_negative";
    "test_mod_gcd_factor_neg";
    "test_mod_gcd_fold_neg";
    "test_mod_min_max";
    "test_mod_mod";
    "test_mod_mod_wrong_sign";
    "test_mod_mod_wrong_sign2";
    "test_mod_mul_sum";
    "test_mod_nest_by_factor";
    "test_mod_nest_by_factor_with_const";
    "test_mod_recombine_with_outer_mul";
    "test_mod_remove";
    "test_mod_to_sub";
    "test_mod_variable_denom_factor_remainder";
    "test_mul_div";
    "test_mul_div_factor_div";
    "test_mul_div_factor_div_neg";
    "test_mul_div_factor_mul";
    "test_mul_div_factor_mul_neg";
    "test_mul_mod_large";
    "test_mul_mod_small";
    "test_mul_neg_1";
    "test_multiple_of_cancellation";
    "test_neg_mod";
    "test_nest_div_negative_factor";
    "test_nested_div_mod_negative_inner_divisor";
    "test_nested_div_mod_symbolic_inner_divisor";
    "test_nested_div_negative_divisor";
    "test_nested_mod_negative_range";
    "test_plus_1_div_2";
    "test_plus_1_mod_2";
    "test_range_div_its_symbolic_bound";
    "test_range_mod_its_symbolic_bound";
    "test_range_split_on_mod";
    "test_reshape_index_roundtrip";
    "test_resnet_half";
    "test_sum_div_big_const";
    "test_sum_div_complex1";
    "test_sum_div_complex2";
    "test_sum_div_complex3";
    "test_sum_div_complex4";
    "test_sum_div_const";
    "test_sum_div_const_big";
    "test_sum_div_min_max";
    "test_sum_div_mod_factor";
    "test_sum_div_no_factor";
    "test_sum_div_partial_remove";
    "test_sum_div_remove";
    "test_sum_div_some_factor";
    "test_sum_div_some_partial_factor";
    "test_sum_div_trim_const";
    "test_symbolic_div";
    "test_symbolic_factor_remainder_div";
    "test_symbolic_gcd_div";
    "test_sym_infer_floordiv_floormod";
    "test_times_2_plus_3_div_4";
    "test_times_2_plus_3_div_4_mod_4";
    "test_times_2_plus_3_mod_4";
  ]

let as_recorded name =
  Golden.graph (golden name) (fun () -> rewritten (recorded name))

(* Values *)

(* [divides_by_zero env d] is whether a division of [d] divides by 0 where each
   variable is bound as [env] says. *)
let divides_by_zero env d =
  let by_zero u =
    (Ops.op u = Op.Div || Ops.op u = Op.Mod)
    && Dtype.equal_const (Interpreter.eval ~vars:env (Ops.nth u 1)) (i 0)
  in
  List.exists by_zero (Ops.toposort ~calls:Enter d)

let int_bounds v =
  match (Ops.vmin v, Ops.vmax v) with
  | `Int lo, `Int hi -> (lo, hi)
  | _ -> invalid_arg "a variable that is not an integer"

(* A point binds each of a node's variables to one of its values within its
   bounds, often the least or the greatest. *)
let gen_point d =
  let within v =
    let lo, hi = int_bounds v in
    let count = Bigint.(succ (hi - lo)) in
    let wrap k = Bigint.(of_int k - (count * fdiv (of_int k) count)) in
    Gen.map
      (fun k -> (Ops.expr v, `Int Bigint.(lo + wrap k)))
      (Gen.frequency
         [ (1, Gen.constant 0); (1, Gen.constant (-1)); (4, Gen.nat) ])
  in
  List.fold_right
    (fun v env -> Gen.map (fun (b, e) -> b :: e) (Gen.pair (within v) env))
    (Shape.variables d) (Gen.constant [])

let pp_case ppf (d, env) =
  Format.fprintf ppf "@[<v>%a@,at %a@]" (Testable.pp uop) d
    Simplification.pp_binding env

(* [keeps_value d env] is the claim that the rewrite of [d] has [d]'s value
   where each variable is bound as [env] says, unless a division of [d] divides
   by 0 there, where op.mli gives a value that the rules need not keep. *)
let keeps_value d env =
  assume (not (divides_by_zero env d));
  let r = once d in
  cover "rewritten" (not (Ops.equal d r));
  equal Dtypes.const ~msg:"the value of the rewrite"
    (Interpreter.eval ~vars:env d)
    (Interpreter.eval ~vars:env r)

(* The divisions of the goldens that the interpreter evaluates: over integer
   variables. A variable's first source, the empty stack of its size, holds no
   value and is not read. *)
let golden_divisions =
  lazy
    (let evaluable d =
       let nodes = Ops.toposort ~calls:Enter d in
       let size u = Ops.op u = Op.Param && Ops.src u <> [] in
       let sizes = List.map (fun u -> Ops.nth u 0) (List.filter size nodes) in
       let value u =
         List.memq u sizes
         ||
         match Ops.op u with
         | Op.Param -> Shape.is_variable u && Dtype.is_int (Ops.dtype u)
         | o -> o = Op.Const || o = Op.Cast || Op.Set.mem o Op.Set.alu
       in
       List.for_all value nodes
     in
     List.filter evaluable (List.concat_map recorded recorded_tests))

(* The points a recorded division is checked at: its variables all at their
   least, all at their greatest, all at their middle, alternately at their least
   and greatest, and at two values drawn from a state seeded by [seed]. *)
let points seed d =
  let vs = List.map (fun v -> (Ops.expr v, int_bounds v)) (Shape.variables d) in
  let rng = Random.State.make [| seed |] in
  let drawn lo hi =
    let span = Bigint.(hi - lo) and most = 1 lsl 30 in
    let bound =
      if Bigint.(span < of_int most) then Bigint.to_int span + 1 else most
    in
    Bigint.(lo + of_int (Random.State.int rng bound))
  in
  let at pick =
    List.mapi (fun k (name, (lo, hi)) -> (name, `Int (pick k lo hi))) vs
  in
  List.map at
    [
      (fun _ lo _ -> lo);
      (fun _ _ hi -> hi);
      (fun _ lo hi -> Bigint.(fdiv (lo + hi) (of_int 2)));
      (fun k lo hi -> if k mod 2 = 0 then lo else hi);
      (fun k lo hi -> if k mod 2 = 0 then hi else lo);
      (fun _ lo hi -> drawn lo hi);
      (fun _ lo hi -> drawn lo hi);
    ]

(* Every recorded division at each of its points, but where one of its divisions
   divides by 0. *)
let keeps_recorded_values () =
  let divisions = Lazy.force golden_divisions in
  at_least ~msg:"recorded divisions that read a variable" Windtrap.int ~than:1
    (List.length (List.filter (fun d -> Shape.variables d <> []) divisions));
  List.iteri
    (fun k d ->
      let r = once d in
      List.iter
        (fun env ->
          if not (divides_by_zero env d) then
            equal Dtypes.const
              ~msg:(Format.asprintf "%a" pp_case (d, env))
              (Interpreter.eval ~vars:env d)
              (Interpreter.eval ~vars:env r))
        (points k d))
    divisions

(* Random divisions *)

(* An expression of index arithmetic over three variables, drawn as data so that
   a failing case shrinks. It has no literal 0: the rules divide each term by
   its constant factor, and the symbolic rules, which run before them, fold a
   literal 0 away. *)
type expr =
  | Var of int
  | Const of int
  | Add of expr * expr
  | Mul of expr * int
  | Div of expr * int
  | Mod of expr * int

let rec pp_expr ppf = function
  | Var k -> Format.fprintf ppf "v%d" k
  | Const c -> Format.pp_print_int ppf c
  | Add (e0, e1) -> Format.fprintf ppf "(%a + %a)" pp_expr e0 pp_expr e1
  | Mul (e, c) -> Format.fprintf ppf "(%a * %d)" pp_expr e c
  | Div (e, c) -> Format.fprintf ppf "(%a // %d)" pp_expr e c
  | Mod (e, c) -> Format.fprintf ppf "(%a %% %d)" pp_expr e c

(* A division or remainder of [num] by [den] at the type [dtype], over three
   variables of [bounds]. *)
type division = {
  dtype : Dtype.t;
  bounds : (Bigint.t * Bigint.t) list;
  num : expr;
  den : expr;
  remainder : bool;
}

let pp_division ppf d =
  Format.fprintf ppf "%a: %a %s %a; %a" Dtype.pp d.dtype pp_expr d.num
    (if d.remainder then "%" else "//")
    pp_expr d.den
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
       (fun ppf (lo, hi) ->
         Format.fprintf ppf "[%a, %a]" Bigint.pp_print lo Bigint.pp_print hi))
    d.bounds

(* [build var e] is [e] with variable [k] as [var k]. *)
let rec build var = function
  | Var k -> var k
  | Const c -> int c
  | Add (e0, e1) -> Ops.O.(build var e0 + build var e1)
  | Mul (e, c) -> Ops.O.(build var e * int c)
  | Div (e, c) -> Ops.O.(build var e // int c)
  | Mod (e, c) -> Ops.O.(build var e % int c)

let variables d =
  Array.of_list
    (List.mapi
       (fun k (lo, hi) ->
         Ops.variable ~dtype:d.dtype (Printf.sprintf "v%d" k) (`Int lo)
           (`Int hi))
       d.bounds)

let node var d =
  let num = build var d.num and den = build var d.den in
  if d.remainder then Ops.O.(num % den) else Ops.O.(num // den)

(* A scalar division over its variables. *)
let scalar d =
  let vs = variables d in
  node (Array.get vs) d

(* Vectors of index arithmetic: variable [k] is a vector of two variables, each
   constant a scalar. Lane [l] of its value is the value of the graph with each
   vector replaced by its lane [l], and each broadcast scalar by the scalar. A
   variable is a leaf: its source, the empty stack, is its shape. *)
let vectors d =
  let vs = variables d in
  node (fun k -> Shape.stack [ vs.(k); vs.((k + 1) mod 3) ]) d

(* The types index arithmetic computes in: the weak integer, and each committed
   one. *)
let committed =
  Dtype.[ Int8; Uint8; Int16; Uint16; Int32; Uint32; Int64; Uint64 ]

let gen_dtype =
  Gen.frequency [ (1, Gen.constant Dtype.Weak_int); (1, Gen.of_list committed) ]

(* The least and greatest values of [dt], an int's for the weak integer. *)
let extremes dt =
  match dt with
  | Dtype.Weak_int -> (Bigint.of_int min_int, Bigint.of_int max_int)
  | _ -> Dtypes.int_bounds dt

(* Small bounds, mostly, and bounds at [dt]'s extremes: the whole type, its
   greatest 16 values or its least 16. *)
let gen_bounds dt k =
  let lo, hi = extremes dt in
  let small =
    let lo, hi = [| (0, 15); (-8, 8); (1, 20) |].(k) in
    let lo, hi = if Dtype.is_unsigned dt && lo < 0 then (0, 16) else (lo, hi) in
    (Bigint.of_int lo, Bigint.of_int hi)
  in
  Gen.frequency
    [
      (6, Gen.constant small);
      (1, Gen.constant (lo, hi));
      (1, Gen.constant (Bigint.(hi - of_int 15), hi));
      (1, Gen.constant (lo, Bigint.(lo + of_int 15)));
    ]

let gen_division =
  let open Gen in
  let nonzero = map (fun k -> if k >= 0 then k + 1 else k) (int_range (-6) 7) in
  let variable = map (fun k -> Var k) (int_range 0 2) in
  let leaf = frequency [ (3, variable); (1, map (fun c -> Const c) nonzero) ] in
  let rec expr depth =
    if depth = 0 then leaf
    else
      let sub = expr (depth - 1) in
      frequency
        [
          (2, leaf);
          (3, map (fun (e0, e1) -> Add (e0, e1)) (pair sub sub));
          (3, map (fun (e, c) -> Mul (e, c)) (pair sub nonzero));
          (1, map (fun (e, c) -> Div (e, c)) (pair sub nonzero));
          (1, map (fun (e, c) -> Mod (e, c)) (pair sub nonzero));
        ]
  in
  let divisor =
    frequency
      [
        (3, map (fun c -> Const c) nonzero);
        (1, variable);
        (1, map (fun (e, c) -> Mul (e, c)) (pair variable nonzero));
      ]
  in
  with_pp pp_division
    (let* dtype = gen_dtype in
     let+ b0 = gen_bounds dtype 0
     and+ b1 = gen_bounds dtype 1
     and+ b2 = gen_bounds dtype 2
     and+ num = expr 3
     and+ den = divisor
     and+ remainder = bool in
     { dtype; bounds = [ b0; b1; b2 ]; num; den; remainder })

let rec reads_var = function
  | Var _ -> true
  | Const _ -> false
  | Add (e0, e1) -> reads_var e0 || reads_var e1
  | Mul (e, _) | Div (e, _) | Mod (e, _) -> reads_var e

(* The forms the rules tell apart. *)
let cover_forms d =
  let extreme (lo, hi) =
    let tlo, thi = extremes d.dtype in
    Bigint.equal lo tlo || Bigint.equal hi thi
  in
  cover "a remainder" d.remainder;
  cover "a division" (not d.remainder);
  cover "a variable divisor" (reads_var d.den);
  cover "a negative constant divisor"
    (match d.den with Const c -> c < 0 | _ -> false);
  cover "a committed type" (d.dtype <> Dtype.Weak_int);
  cover "a weak integer" (d.dtype = Dtype.Weak_int);
  cover "a bound at its type's extreme" (List.exists extreme d.bounds)

(* [keeps_lanes d env] is the claim that each lane of the rewrite of [d] has
   that lane of [d]'s value. *)
let lane l u =
  let table = Ops.Tbl.create 16 in
  let rec go u =
    match Ops.Tbl.find_opt table u with
    | Some v -> v
    | None ->
        let v =
          match Ops.op u with
          | Op.Param -> u
          | Op.Stack -> go (Ops.nth u l)
          | Op.Expand | Op.Reshape -> go (Ops.nth u 0)
          | _ -> Ops.replace u ~src:(List.map go (Ops.src u))
        in
        Ops.Tbl.add table u v;
        v
  in
  go u

let keeps_lanes d env =
  assume (not (List.exists (fun l -> divides_by_zero env (lane l d)) [ 0; 1 ]));
  let r = once d in
  cover "rewritten" (not (Ops.equal d r));
  List.iter
    (fun l ->
      equal Dtypes.const
        ~msg:(Printf.sprintf "lane %d of the rewrite" l)
        (Interpreter.eval ~vars:env (lane l d))
        (Interpreter.eval ~vars:env (lane l r)))
    [ 0; 1 ]

(* A random division with a point, the division printed as drawn. *)
let gen_random build =
  Gen.with_pp
    (fun ppf (d, (u, env)) ->
      Format.fprintf ppf "@[<v>%a@,%a@]" pp_division d pp_case (u, env))
    (Gen.bind gen_division (fun d ->
         let u = build d in
         Gen.map (fun env -> (d, (u, env))) (gen_point u)))

let div_and_mod_symbolic =
  group "div_and_mod_symbolic"
    [
      nested_divisions;
      test "(x // c + a) // d stays for a committed x where x + a * c wraps"
        committed_division_stays;
      constant_terms;
      one_quotient;
      constant_divisors;
      other_divisors;
      cases "raises Division_by_zero for a divisor that is always 0" ~id:fst
        zero_divisors (fun (_, d) ->
          raises Division_by_zero (fun () -> rewrite d));
      cases "a rule that computes with the value of invalid does not apply"
        ~id:fst with_invalid (fun (_, d) -> equal (option uop) None (rewrite d));
      group "rewrites each recorded division as its golden states"
        (List.map as_recorded recorded_tests);
      test "keeps the value of each recorded division" keeps_recorded_values;
      prop ~count:2000 "keeps the value of a random division"
        (gen_random scalar) (fun (d, (u, env)) ->
          cover_forms d;
          keeps_value u env);
      prop ~count:2000 "keeps each lane's value of a random division of vectors"
        (gen_random vectors) (fun (d, (u, env)) ->
          cover_forms d;
          keeps_lanes u env);
    ]
