(* Tests of Tolk.Divandmod: each rewrite of a division or remainder is the one
   tinygrad makes, and keeps the value of what it rewrites. *)

open Windtrap
open Tolk

let i n = `Int (Bigint.of_int n)

let var ?(dtype = Dtype.Weak_int) ?multiple_of name lo hi =
  Ops.variable ~dtype ?multiple_of name (i lo) (i hi)

let x = var "x" 0 100
let a = var "a" 0 99
let b = var "b" 0 99
let int = Ops.int
let rewrite d = Ops.Pattern_matcher.rewrite Divandmod.div_and_mod_symbolic () d
let once d = Option.value (rewrite d) ~default:d

(* A golden holds divisions and remainders, each followed by its rewrite. *)
let rewritten ds = Ops.sink (List.concat_map (fun d -> [ d; once d ]) ds)

let rec every_other = function
  | d :: _ :: rest -> d :: every_other rest
  | [] | [ _ ] -> []

let recorded file = every_other (Ops.src (Golden.sink file))

let rewrites name d =
  Golden.graph (name ^ ".golden") (fun () -> rewritten [ d ])

(* (x // c + a) // d *)

let nested_divisions =
  group "nested divisions"
    [
      rewrites "merge_nested_divisions" Ops.O.(((x // int 2) + int 3) // int 4);
      rewrites "merge_nested_divisions_by_a_negative_inner_divisor"
        Ops.O.(((var "x" (-100) 100 // int (-2)) + int 3) // int 4);
      rewrites "merge_nested_divisions_of_committed_integers"
        Ops.O.(((var ~dtype:Int32 "x" 0 100 // int 2) + int 3) // int 4);
      rewrites "split_nested_divisions_by_a_negative_divisor"
        Ops.O.(((x // int 2) + int 3) // int (-4));
      test "a committed division stays where x + a * c wraps" (fun () ->
          let k = 1 lsl 30 in
          let w = var ~dtype:Int32 "w" 0 k in
          let d = Ops.O.(((w // int 2) + int k) // int 2) in
          is_none (rewrite d);
          let vars = [ ("w", i k) ] in
          equal Dtypes.const
            (`Int (Bigint.of_int 805306368))
            (Interpreter.eval ~vars (once d)));
    ]

(* (x + c) // d and (x + c) % d *)

let constant_terms =
  group "constant terms"
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
  group "one quotient"
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

(* Variables declared a multiple of a number *)

let declared_multiples =
  let m = var ~multiple_of:4 "m" 0 100 in
  group "declared multiples"
    [
      rewrites "fold_the_remainder_of_a_declared_multiple" Ops.O.(m % int 4);
      rewrites "fold_the_remainder_of_a_declared_multiple_by_a_divisor_of_it"
        Ops.O.(m % int 2);
      rewrites "keep_the_division_of_a_declared_multiple" Ops.O.(m // int 4);
      rewrites "rewrite_the_remainder_of_a_declared_multiple_by_another_divisor"
        Ops.O.(m % int 3);
    ]

(* x // c and x % c for a positive constant c *)

let constant_divisors =
  let huge = Ops.const (`Int (Bigint.shift_left Bigint.one 100)) in
  let huge_divisor = Ops.const (`Int Bigint.(pred (shift_left one 101))) in
  group "constant divisors"
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
  group "other divisors"
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
  let raises_on d = raises Division_by_zero (fun () -> rewrite d) in
  group "zero divisors"
    [
      test "a division by the constant 0 raises Division_by_zero" (fun () ->
          raises_on Ops.O.(x // int 0));
      test "a remainder by the constant 0 raises Division_by_zero" (fun () ->
          raises_on Ops.O.(x % int 0));
      test "a division of a sum with a constant by 0 raises Division_by_zero"
        (fun () -> raises_on Ops.O.((x + int 7) // int 0));
      test "a division by a variable that is always 0 raises Division_by_zero"
        (fun () -> raises_on Ops.O.(x // var "z" 0 0));
      test "a sum with a term that is always 0 raises Division_by_zero"
        (fun () -> raises_on Ops.O.(((x * int 0) + x) // int 3));
      test "a division of any integer by 0 raises Division_by_zero" (fun () ->
          let any =
            Ops.variable "x"
              (`Int (Bigint.of_int min_int))
              (`Int (Bigint.of_int max_int))
          in
          raises_on Ops.O.(any // int 0));
    ]

(* Invalid *)

let invalid =
  group "invalid values"
    [
      test "a rule that computes with the value of invalid does not apply"
        (fun () ->
          is_none (rewrite Ops.O.((x + Ops.invalid) // int 3));
          is_none (rewrite Ops.O.((x + Ops.invalid) % int 3)));
    ]

(* tinygrad's tests *)

let goldens =
  Sys.readdir (Filename.dirname Sys.executable_name)
  |> Array.to_list
  |> List.filter (fun f -> Filename.check_suffix f ".golden")
  |> List.sort String.compare

(* The goldens recorded from tinygrad's tests are named after them. *)
let tinygrad_goldens = List.filter (String.starts_with ~prefix:"test_") goldens
let as_tinygrad file = Golden.graph file (fun () -> rewritten (recorded file))

let tinygrad_tests =
  group "tinygrad's tests" (List.map as_tinygrad tinygrad_goldens)

(* Values *)

let rewritten_alone d = Option.value (rewrite d) ~default:d

(* [keeps_value d env] is that the rewrite of [d] has [d]'s value where each
   variable is bound as [env] says, unless a division of [d] divides by 0
   there. *)
let keeps_value d env =
  let divides_by_zero u =
    (Ops.op u = Op.Floordiv || Ops.op u = Op.Floormod)
    && Dtype.equal_const (Interpreter.eval ~vars:env (Ops.nth u 1)) (i 0)
  in
  assume (not (List.exists divides_by_zero (Ops.toposort ~calls:Enter d)));
  let r = rewritten_alone d in
  cover "rewritten" (not (Ops.equal d r));
  equal Dtypes.const ~msg:"the value of the rewrite"
    (Interpreter.eval ~vars:env d)
    (Interpreter.eval ~vars:env r)

let int_bounds v =
  match (Ops.vmin v, Ops.vmax v) with
  | `Int lo, `Int hi -> (lo, hi)
  | _ -> invalid_arg "a variable that is not an integer"

(* A point binds each of a node's variables to one of its values: a multiple of
   its declared divisor within its bounds, often the least or the greatest. *)
let gen_point d =
  let within v =
    let lo, hi = int_bounds v in
    let m =
      match Ops.arg v with
      | Param { multiple_of = Some m; _ } -> Bigint.of_int m
      | _ -> Bigint.one
    in
    let first = Bigint.(cdiv lo m) in
    let count = Bigint.(succ (fdiv hi m - first)) in
    Gen.map
      (fun k -> (Ops.expr v, `Int Bigint.(m * (first + erem (of_int k) count))))
      (Gen.frequency
         [ (1, Gen.constant 0); (1, Gen.constant (-1)); (4, Gen.nat) ])
  in
  List.fold_right
    (fun v env -> Gen.map (fun (b, e) -> b :: e) (Gen.pair (within v) env))
    (Ops.variables d) (Gen.constant [])

let pp_case ppf (d, env) =
  Format.fprintf ppf "@[<v>%a@,at %a@]" (Testable.pp Uops.uop) d
    (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf (n, v) ->
         Format.fprintf ppf "%s=%a" n Dtype.pp_const v))
    env

let gen_case gen_division =
  Gen.with_pp pp_case
    (Gen.bind gen_division (fun d ->
         Gen.map (fun env -> (d, env)) (gen_point d)))

(* The divisions of the goldens that the interpreter evaluates: over integer
   variables. *)
let golden_divisions =
  let evaluable u =
    match Ops.op u with
    | Op.Param -> Ops.is_variable u && Dtype.is_int (Ops.dtype u)
    | o -> o = Op.Const || o = Op.Cast || Op.Set.mem o Op.Set.alu
  in
  List.concat_map recorded goldens
  |> List.filter (fun d -> List.for_all evaluable (Ops.toposort ~calls:Enter d))

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

let variables = [| var "v0" 0 15; var "v1" (-8) 8; var "v2" 1 20 |]

let rec build = function
  | Var k -> variables.(k)
  | Const c -> int c
  | Add (e0, e1) -> Ops.O.(build e0 + build e1)
  | Mul (e, c) -> Ops.O.(build e * int c)
  | Div (e, c) -> Ops.O.(build e // int c)
  | Mod (e, c) -> Ops.O.(build e % int c)

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
  map
    (fun (num, den, remainder) ->
      if remainder then Ops.O.(build num % build den)
      else Ops.O.(build num // build den))
    (triple (expr 3) divisor bool)

let values =
  group "values"
    [
      prop ~count:2000
        "each rewrite of the goldens' divisions keeps the division's value"
        (gen_case (Gen.of_list golden_divisions))
        (fun (d, env) -> keeps_value d env);
      prop ~count:2000 "each rewrite of a random division keeps its value"
        (gen_case gen_division) (fun (d, env) -> keeps_value d env);
    ]

let () =
  exit
    (run "Tolk.Divandmod"
       [
         nested_divisions;
         constant_terms;
         one_quotient;
         declared_multiples;
         constant_divisors;
         other_divisors;
         zero_divisors;
         invalid;
         tinygrad_tests;
         values;
       ])
