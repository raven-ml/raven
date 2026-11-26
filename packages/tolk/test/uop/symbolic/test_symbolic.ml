open Windtrap
open Tolk
open Simplification

(* Builders *)

let a = var "a" 0 8
let b = var "b" 0 8
let x = var ~dtype:Int32 "x" 0 8
let y = var ~dtype:Int32 "y" 0 8
let cond = Ops.O.(a < int 4)
let other = Ops.O.(b < int 2)
let f = Shape.param 0 Dtype.Float32
let g = Shape.param 1 Dtype.Float32
let buf = Shape.param ~shape:[ Int 16 ] 2 Dtype.Float32
let r0 = Ops.range ~axis_type:Weak (Int 4) [ 0 ]
let r1 = Ops.range ~axis_type:Weak (Int 4) [ 1 ]
let reduce_range = Ops.range ~axis_type:Weak (Int 4) [ 2 ]
let int32 n = Ops.int ~dtype:Int32 n
let zero32 = Ops.float ~dtype:Float32 0.

(* A node built as given, where the constructors would fold it: a group of one
   node, a stack. *)
let raw ?arg op src = Ops.v ?arg ~src op

(* [simplifies_to input expected] is the claim that [sym] rewrites [input] to
   [expected], whose operands are taken in the order [commutative] gives them,
   as the rewrite does. *)
let simplifies_to input expected =
  equal uop (rewrite Shape.commutative expected) (sym input)

(* Bindings

   A scenario's leaves are its variables [a], [b], [c] and its divisors [d] and
   [e]. A binding draws each within its bounds. *)

let rec each g = function
  | [] -> Gen.constant []
  | v :: vs ->
      Gen.(
        let+ v = g v and+ vs = each g vs in
        v :: vs)

(* A scenario's leaves span at most a few values. *)
let within (lo, hi) =
  Gen.map
    (fun d -> Bigint.(lo + of_int d))
    (Gen.int_range 0 (Bigint.to_int Bigint.(hi - lo)))

type binding = {
  s : Expressions.scenario;
  values : Bigint.t list;  (** Of [a], [b] and [c]. *)
  divisors : Bigint.t list;  (** Of [d] and [e]. *)
}

let pp_bound ppf r =
  let ints = Format.(pp_print_list ~pp_sep:pp_print_space Bigint.pp_print) in
  Format.fprintf ppf "%a at a b c = %a, d e = %a" Expressions.pp_scenario r.s
    ints r.values ints r.divisors

let binding =
  let open Gen in
  with_pp pp_bound
    (let* s = Expressions.scenario in
     let+ values = each within s.vars and+ divisors = each within s.divisors in
     { s; values; divisors })

let env r =
  List.map2 (fun n v -> (n, `Int v)) [ "a"; "b"; "c" ] r.values
  @ List.map2 (fun n v -> (n, `Int v)) [ "d"; "e" ] r.divisors

(* [r]'s variables, each with its value, as [Expressions.node] builds them. *)
let variables r =
  List.map2
    (fun (name, (lo, hi)) v ->
      (Ops.variable ~dtype:r.s.dtype name (`Int lo) (`Int hi), v))
    (List.combine [ "a"; "b"; "c" ] r.s.vars)
    r.values

let eval r u = Interpreter.eval ~vars:(env r) u
let holds r u = Testable.equal Dtypes.const (`Bool true) (eval r u)

(* Conditions *)

(* A clause of form [form] on [(x, v)] that holds where [x] is [v], its bound
   [slack] away, or, in the last form, on [x] and another variable [(y, w)]. *)
let clause form slack (x, v) (y, w) =
  let c d = Ops.const (`Int Bigint.(v + of_int d)) in
  let above = c (1 + slack) and below = c (-1 - slack) in
  let least = c (-slack) and greatest = c slack in
  Ops.O.(
    match form with
    | 0 -> x < above
    | 1 -> below < x
    | 2 -> lnot (x < least)
    | 3 -> lnot (greatest < x)
    | _ -> if Bigint.lt v w then x < y else lnot (x < y))

let given_valid_case =
  Gen.(
    triple binding
      (list ~size:(constant 3) (pair (int_range 0 4) (int_range 0 3)))
      (pair bool bool))

(* At [r], [uop_given_valid] of clauses that hold there keeps the value of [r]'s
   expression: one clause per variable, of the form drawn, and [a + b] at least
   1 when [sum]. A clause that a committed type reads otherwise, by a constant
   outside the type, does not hold, and is left out. *)
let keeps_value_where_valid (r, forms, (sum, try_simplex)) =
  let e = Expressions.node r.s in
  let vars = variables r in
  let clauses =
    List.mapi
      (fun k (form, slack) ->
        clause form slack (List.nth vars k) (List.nth vars ((k + 1) mod 3)))
      forms
  in
  let a = fst (List.nth vars 0) and b = fst (List.nth vars 1) in
  let at_least_1 = Ops.O.(lnot (a + b < int 1)) in
  let clauses = if sum then at_least_1 :: clauses else clauses in
  let clauses = List.filter (holds r) clauses in
  let valid = List.fold_left Ops.O.( land ) (Ops.bool true) clauses in
  let simplified = Symbolic.uop_given_valid ~try_simplex valid e in
  cover "a clause bounds a variable by another"
    (List.exists (fun (form, _) -> form = 4) forms);
  cover "a sum is at least 1" (sum && holds r at_least_1);
  cover "the expression simplifies" (not (Ops.equal simplified e));
  cover "a committed type" (not (Dtype.equal r.s.dtype Weak_int));
  equal ~msg:"value" Dtypes.const (eval r e) (eval r simplified)

(* An int8 sum of a and b that wraps to 1, so no term of it is at least 1. *)
let wrapping_sum =
  let z = Bigint.of_int in
  let s : Expressions.scenario =
    {
      dtype = Int8;
      vars =
        [ (z (-128), z (-128)); (z (-128), z (-127)); (z (-128), z (-128)) ];
      divisors = [ (z 1, z 1); (z 1, z 1) ];
      expr = Add (Add (Mod (Add (Var 1, Var 0), 2), Var 0), Var 0);
    }
  in
  ( { s; values = [ z (-128); z (-127); z (-128) ]; divisors = [ z 1; z 1 ] },
    [ (0, 0); (0, 0); (0, 0) ],
    (true, true) )

let given_valid =
  let c = var "c" 0 3 in
  group "uop_given_valid"
    [
      test "assumes e < c and c < e for a constant c" (fun () ->
          equal uop (Ops.int 0)
            (Symbolic.uop_given_valid Ops.O.(a < int 5) Ops.O.(a // int 5));
          equal uop (Ops.int 1)
            (Symbolic.uop_given_valid Ops.O.(int 4 < a) Ops.O.(a // int 5)));
      test "assumes not (e < c), a bound from below" (fun () ->
          equal uop (Ops.bool false)
            (Symbolic.uop_given_valid
               Ops.O.(lnot (a < int 5))
               Ops.O.(a < int 3)));
      test "assumes not (c < e), a bound from above" (fun () ->
          equal uop (Ops.bool true)
            (Symbolic.uop_given_valid
               Ops.O.(lnot (int 2 < a))
               Ops.O.(a < int 3));
          equal uop (Ops.int 0)
            (Symbolic.uop_given_valid
               Ops.O.(lnot (int 1 < a))
               Ops.O.(a // int 2)));
      test "assumes e < x for an integer x, read at its bounds" (fun () ->
          (* b is at most 8, so a < b bounds a by 7. *)
          equal uop (Ops.int 0)
            (Symbolic.uop_given_valid Ops.O.(a < b) Ops.O.(a // int 8)));
      test "assumes two clauses on one expression together" (fun () ->
          equal uop (Ops.int 1)
            (Symbolic.uop_given_valid
               Ops.O.((a < int 5) land (int 2 < a))
               Ops.O.(a // int 3)));
      test "assumes every clause of a conjunction" (fun () ->
          equal uop (Ops.int 0)
            (Symbolic.uop_given_valid
               Ops.O.((a < int 5) land (b < int 5))
               Ops.O.((a + b) // int 9)));
      test "ignores a clause that is not a bound" (fun () ->
          equal uop (Ops.int 0)
            (Symbolic.uop_given_valid
               Ops.O.((a <> int 1) land (a < int 5))
               Ops.O.(a // int 5));
          let e = Ops.O.(c + a) in
          equal uop e (Symbolic.uop_given_valid Ops.O.(cond lor other) e));
      test "tries each term of a sum at least 1, unless try_simplex is false"
        (fun () ->
          let valid = Ops.O.(lnot (a + b < int 1)) in
          let both_zero = Ops.O.((a < int 1) land (b < int 1)) in
          equal uop (Ops.bool false) (Symbolic.uop_given_valid valid both_zero);
          equal uop both_zero
            (Symbolic.uop_given_valid ~try_simplex:false valid both_zero));
      test "tries no term of a sum with a coefficient" (fun () ->
          let valid = Ops.O.(lnot ((a * int 2) + b < int 1)) in
          let e = Ops.O.((a * int 2 < int 1) land (b < int 1)) in
          equal uop e (Symbolic.uop_given_valid valid e));
      test "tries no constant term of a sum" (fun () ->
          (* Each lane [base + k] simplifies [base] alike: trying the 1 at least
             1 would set the lanes apart. *)
          let a = var "a" 0 7 and b = var "b" 0 7 and d = var "d" 0 1 in
          let valid =
            Ops.O.(lnot (a + b + int 1 < int 1) land lnot (c + d < int 1))
          in
          let base =
            Ops.O.(
              ((a * int 8) + b + (c * int 128) + (d * int 64) + int 1000)
              % int 1024 * int 4)
          in
          let lane k =
            Ops.nth (Symbolic.uop_given_valid valid Ops.O.(base + int k)) 0
          in
          List.iter
            (fun k -> equal ~msg:(string_of_int k) uop (lane 1) (lane k))
            [ 2; 3 ]);
      test "keeps what the terms tried disagree on" (fun () ->
          let valid = Ops.O.(lnot (a + b < int 1)) in
          let e = Ops.O.((a < int 1) lor (b < int 1)) in
          equal uop e (Symbolic.uop_given_valid valid e));
      test "keeps each lane's value where valid holds" (fun () ->
          (* The first lane simplifies only by trying each term of a + b at
             least 1, which the second does not need. *)
          let valid = Ops.O.(lnot (a + b < int 1)) in
          let both_zero = Ops.O.((a < int 1) land (b < int 1)) in
          let pair = raw Stack [ both_zero; Ops.O.(a < int 1) ] in
          let simplified = Symbolic.uop_given_valid valid pair in
          List.iter
            (fun (va, vb) ->
              let vars = [ ("a", i va); ("b", i vb) ] in
              List.iter
                (fun k ->
                  equal
                    ~msg:(Printf.sprintf "lane %d at a=%d, b=%d" k va vb)
                    Dtypes.const
                    (Interpreter.eval ~vars (Ops.nth pair k))
                    (Interpreter.eval ~vars (Ops.nth simplified k)))
                [ 0; 1 ])
            [ (0, 1); (1, 0); (3, 5) ]);
      test "bounds a loaded value by a clause on it" (fun () ->
          let x = var "x" 0 100 in
          let inside = Ops.O.((int 29 < x) land (x < int 80)) in
          let loaded =
            Ops.cast
              (Ops.index
                 (Shape.param ~shape:[ Int 100 ] 1 Dtype.Int32)
                 [ Shape.valid Ops.O.(x + int (-30)) inside ])
              Weak_int
          in
          let valid =
            Ops.O.(inside land (int (-1) < loaded) land (loaded < int 100))
          in
          equal uop loaded
            (Symbolic.uop_given_valid valid Ops.O.(loaded % int 100)));
      test "leaves a loop's range as it is, its bound included" (fun () ->
          (* Assumed inside the bound, a < 3 would make the bound 1. *)
          let r =
            Ops.range ~axis_type:Loop (Sym Ops.O.((a // int 4) + int 1)) [ 9 ]
          in
          equal uop r (Symbolic.uop_given_valid Ops.O.(a < int 3) r));
      prop "has u's values wherever valid holds" ~examples:[ wrapping_sum ]
        given_valid_case keeps_value_where_valid;
    ]

let simplify_valid_case =
  Gen.(
    pair binding
      (list ~size:(int_range 2 4)
         (triple (int_range 0 3) (int_range 0 3) (int_range (-5) 35))))

(* At [r], [simplify_valid] of clauses of the forms [uop_given_valid] reads, on
   [r]'s variables and expression, has the clauses' value. A clause is [(form,
   term, c)], [term] [3] for the expression. *)
let keeps_conjunction (r, clauses) =
  let terms = List.map fst (variables r) @ [ Expressions.node r.s ] in
  let clause (form, term, c) =
    let x = List.nth terms term in
    Ops.O.(
      match form with
      | 0 -> x < int c
      | 1 -> int c < x
      | 2 -> lnot (x < int c)
      | _ -> lnot (int c < x))
  in
  let valid =
    match List.map clause clauses with
    | [] -> Ops.bool true
    | c :: cs -> List.fold_left Ops.O.( land ) c cs
  in
  let simplified = Symbolic.simplify_valid valid in
  cover "simplified" (Option.is_some simplified);
  cover "holds" (holds r valid);
  Option.iter
    (fun v -> equal ~msg:"value" Dtypes.const (eval r valid) (eval r v))
    simplified

let simplify_valid =
  let some = option uop in
  group "simplify_valid"
    [
      test "is None for a single clause" (fun () ->
          equal some None (Symbolic.simplify_valid Ops.O.(a < int 5)));
      test "simplifies a clause knowing a tighter one, ordered first, holds"
        (fun () ->
          equal some
            (Some (Ops.uprod Ops.O.(a < int 3) [ Ops.bool true ]))
            (Symbolic.simplify_valid Ops.O.((a < int 5) land (a < int 3))));
      test "removes duplicate clauses" (fun () ->
          let c = Ops.O.(a < int 5) in
          equal some (Some c) (Symbolic.simplify_valid Ops.O.(c land c)));
      test "orders first a clause on an expression others depend on" (fun () ->
          let x = var "x" 0 100 and y = var "y" 0 1 in
          let valid = Ops.O.((x + y < int 3) land (x < int 2)) in
          equal some
            (Some (Ops.uprod Ops.O.(x < int 2) [ Ops.bool true ]))
            (Symbolic.simplify_valid valid));
      test "is None if only the order of the clauses changes" (fun () ->
          equal some None
            (Symbolic.simplify_valid Ops.O.((a + b < int 5) land (a < int 2))));
      test "simplifies the clauses alike in either order" (fun () ->
          let r = Ops.range ~axis_type:Weak (Int 2) [ 0 ] in
          let first = Ops.O.(r < int 1)
          and second = Ops.O.(((r * int 5) + int 1) % int 6 < int 5) in
          let expected = Some (Ops.uprod first [ Ops.bool true ]) in
          equal some expected
            (Symbolic.simplify_valid Ops.O.(second land first));
          equal some expected
            (Symbolic.simplify_valid Ops.O.(first land second)));
      test "is None if valid involves an index" (fun () ->
          let load = Ops.index buf [ a ] in
          equal some None
            (Symbolic.simplify_valid
               Ops.O.((load < float 1.) land (a < int 3) land (a < int 5))));
      prop "has valid's value" simplify_valid_case keeps_conjunction;
    ]

let simplify_valid_pm =
  let rewrite = rewrite Symbolic.pm_simplify_valid in
  group "pm_simplify_valid"
    [
      test "simplifies a conjunction with simplify_valid" (fun () ->
          equal uop
            (Ops.uprod Ops.O.(a < int 3) [ Ops.bool true ])
            (rewrite Ops.O.((a < int 5) land (a < int 3))));
      test "simplifies a gate's weak integer value knowing its condition holds"
        (fun () ->
          equal uop
            (Shape.valid (Ops.int 0) Ops.O.(a < int 5))
            (rewrite (Shape.valid Ops.O.(a // int 5) Ops.O.(a < int 5))));
      test "simplifies a gated index to one of its value where the gate holds"
        (fun () ->
          let r0 = Ops.range ~axis_type:Weak (Int 30) [ 0 ]
          and r1 = Ops.range ~axis_type:Weak (Int 7) [ 1 ]
          and r2 = Ops.range ~axis_type:Weak (Int 2) [ 2 ] in
          let alu11 = Ops.O.(r1 + r2) in
          let idx =
            Ops.O.(
              ((alu11 + int 1) // int 7 * int (-31))
              + ((((alu11 + int 218) // int 224) + r0) % int 30 * int 1568))
          in
          let gate = Ops.O.((r2 < int 1) land (r1 < int 6)) in
          let simplified = Ops.O.(r0 * int 1568) in
          equal uop
            (Shape.valid simplified gate)
            (rewrite (Shape.valid idx gate));
          (* Where the gate holds, r2 is 0 and r1 below 6. *)
          for v0 = 0 to 29 do
            for v1 = 0 to 5 do
              let vars = [ ("r0", i v0); ("r1", i v1); ("r2", i 0) ] in
              equal
                ~msg:(Printf.sprintf "at r0=%d, r1=%d" v0 v1)
                Dtypes.const
                (Interpreter.eval ~vars idx)
                (Interpreter.eval ~vars simplified)
            done
          done);
      test "leaves a gate's committed value" (fun () ->
          let gated = Shape.valid Ops.O.(x // int32 5) Ops.O.(x < int32 5) in
          equal uop gated (rewrite gated));
      test "tries no term of a sum for a gate's value" (fun () ->
          (* With try_simplex, the selection would be 0. *)
          let both_zero = Ops.O.((a < int 1) land (b < int 1)) in
          let gated =
            Shape.valid
              (Ops.where both_zero (Ops.int 1) (Ops.int 0))
              Ops.O.(lnot (a + b < int 1))
          in
          equal uop gated (rewrite gated));
      test "leaves a gate's value that reads a load" (fun () ->
          let ints = Shape.param ~shape:[ Int 16 ] 3 Dtype.Int32 in
          let loaded = Ops.cast (Ops.load (Ops.index ints [ a ]) []) Weak_int in
          let gated = Shape.valid Ops.O.(loaded + (a // int 5)) cond in
          equal uop gated (rewrite gated));
    ]

let drop_and_clauses =
  let rewrite = rewrite Symbolic.pm_drop_and_clauses in
  group "pm_drop_and_clauses"
    [
      test "drops the clauses that run in none of the value's ranges" (fun () ->
          let gate = Ops.O.((r0 < int 3) land (r1 < int 2) land cond) in
          equal uop
            (Shape.valid r0 (Ops.uprod (Ops.bool true) [ Ops.O.(r0 < int 3) ]))
            (rewrite (Shape.valid r0 gate)));
      test "drops every clause when none runs in the value's ranges" (fun () ->
          let gate = Ops.O.((r1 < int 2) land cond) in
          equal uop
            (Shape.valid r0 (Ops.uprod (Ops.bool true) []))
            (rewrite (Shape.valid r0 gate)));
      test "leaves a gate whose clauses all run in the value's ranges"
        (fun () ->
          let gated =
            Shape.valid Ops.O.(r0 + r1) Ops.O.((r0 < int 3) land (r1 < int 2))
          in
          equal uop gated (rewrite gated));
    ]

let move_where_on_load =
  let rewrite = rewrite Symbolic.pm_move_where_on_load in
  let index idx = Ops.index buf [ idx ] in
  group "pm_move_where_on_load"
    [
      test "moves a clause into the index's gate" (fun () ->
          equal uop
            (Ops.where (Ops.bool true)
               (index (Shape.valid a (Ops.uprod (Ops.bool true) [ cond ])))
               zero32)
            (rewrite (Ops.where cond (index a) (Ops.float 0.))));
      test "moves the negated condition of a selection of 0 first" (fun () ->
          equal uop
            (Ops.where (Ops.bool true)
               (index
                  (Shape.valid a
                     (Ops.uprod (Ops.bool true) [ Ops.logical_not cond ])))
               zero32)
            (rewrite (Ops.where cond (Ops.float 0.) (index a))));
      test "drops a clause the index already requires" (fun () ->
          let idx = Shape.valid a cond in
          equal uop
            (Ops.where (Ops.bool true)
               (index (Shape.valid a (Ops.uprod cond [ other ])))
               zero32)
            (rewrite
               (Ops.where Ops.O.(cond land other) (index idx) (Ops.float 0.))));
      test "moves the clauses it can and keeps the others in the selection"
        (fun () ->
          let on_range = Ops.O.(r0 < int 2) in
          equal uop
            (Ops.where
               (Ops.uprod (Ops.bool true) [ on_range ])
               (index (Shape.valid a (Ops.uprod (Ops.bool true) [ cond ])))
               zero32)
            (rewrite
               (Ops.where Ops.O.(cond land on_range) (index a) (Ops.float 0.))));
      test "keeps a clause on a range the index lacks" (fun () ->
          let sel = Ops.where Ops.O.(r0 < int 2) (index a) zero32 in
          equal uop sel (rewrite sel));
      test "keeps a constant clause" (fun () ->
          let sel = Ops.where (Ops.bool true) (index a) zero32 in
          equal uop sel (rewrite sel));
      test "keeps a clause that reads another index" (fun () ->
          let other_load = Ops.O.(Ops.index buf [ b ] < float 1.) in
          let sel = Ops.where other_load (index a) zero32 in
          equal uop sel (rewrite sel));
      test "leaves a selection of -0." (fun () ->
          (* A gated load reads +0. where its gate fails. *)
          let sel = Ops.where cond (index a) (Ops.float ~dtype:Float32 (-0.)) in
          equal uop sel (rewrite sel));
    ]

let clean_up_group_sink =
  let rewrite = rewrite Symbolic.pm_clean_up_group_sink in
  group "pm_clean_up_group_sink"
    [
      cases "splices into a sink the sources of each source of its kind" ~id:fst
        [
          ("sink", fun () -> raw Sink [ a; b ]);
          ("noop", fun () -> raw Noop [ a; b ]);
          ("stack", fun () -> raw Stack [ a; b ]);
          ("after", fun () -> raw After [ a; b ]);
        ]
        (fun (_, source) ->
          equal uop (raw Sink [ a; b; f ]) (rewrite (raw Sink [ source (); f ])));
      test "leaves a group, which is no sink" (fun () ->
          let nested = Ops.group [ Ops.group [ a; b ]; f ] in
          equal uop nested (rewrite nested));
    ]

(* sym *)

(* The power of constants by an exponent of any size, as
   [Transcendental.xpow_int] states it: squares and products at the type's
   width, a negative exponent giving 0, but 1 for a base of 1 and -1 or 1 by
   parity for a signed -1. *)
let pow_reference dt b e : Dtype.value =
  let two = Bigint.of_int 2 in
  if Bigint.sign e >= 0 then begin
    let m = Bigint.shift_left Bigint.one (Dtype.bitsize dt) in
    let rec go acc b e =
      if Bigint.sign e = 0 then acc
      else
        let acc =
          if Bigint.divisible e two then acc else Bigint.(rem (mul acc b) m)
        in
        go acc Bigint.(rem (mul b b) m) (Bigint.shift_right e 1)
    in
    Dtype.truncate dt (`Int (go Bigint.one (Bigint.rem b m) e))
  end
  else if Bigint.equal b Bigint.one then `Int Bigint.one
  else if Dtype.mem dt Dtype.sints && Bigint.equal b Bigint.minus_one then
    `Int (if Bigint.divisible e two then Bigint.one else Bigint.minus_one)
  else `Int Bigint.zero

(* A base at a type's extremes, 0, 1, -1, 2 and 3, and exponents around 0, -1,
   the width and the larger powers of two. *)
let constant_pow =
  let open Gen in
  let* dt = of_list ~pp:Dtype.pp Dtype.ints in
  let in_type v = Dtype.Value.to_z (Dtype.truncate dt v) in
  let bases =
    [ in_type (Dtype.min dt); in_type (Dtype.max dt) ]
    @ List.map (fun n -> in_type (Dtype.Value.of_int n)) [ -1; 0; 1; 2; 3 ]
  in
  let exponents =
    List.concat_map
      (fun c -> Bigint.[ pred c; c; succ c ])
      (Bigint.zero :: Bigint.minus_one
      :: Bigint.of_int (Dtype.bitsize dt)
      :: List.map (fun n -> Bigint.shift_left Bigint.one n) [ 8; 16; 32; 40 ])
  in
  let z = Testable.pp Dtypes.z in
  let+ b = of_list ~pp:z bases and+ e = of_list ~pp:z exponents in
  (dt, b, e)

let ops_of u = List.map Ops.op (Ops.toposort ~calls:Skip u)

let powers =
  group "powers"
    [
      test "computes a float power as xpow does" (fun () ->
          simplifies_to (Ops.pow f g) (sym (Transcendental.xpow f g)));
      cases "computes a float power by an infinity or NaN as xpow does"
        ~id:Float.to_string [ Float.infinity; Float.nan ] (fun e ->
          equal uop
            (sym (Transcendental.xpow f (Ops.float e)))
            (sym (Ops.pow f (Ops.float e))));
      test "computes an integer power by products, without exp2 or log2"
        (fun () ->
          let p = sym (Ops.alu x Op.Pow [ y ]) in
          equal (list string) []
            (List.filter_map
               (fun o ->
                 match (o : Op.t) with
                 | Pow | Exp2 | Log2 -> Some (Op.name o)
                 | _ -> None)
               (ops_of p)));
      prop "folds a power of committed constants as repeated products do"
        ~examples:[ (Dtype.Int8, Bigint.of_int 2, Bigint.minus_one) ]
        constant_pow
        (fun (dt, b, e) ->
          let u =
            Ops.alu (Ops.const ~dtype:dt (`Int b)) Op.Pow [ Ops.const (`Int e) ]
          in
          cover "an exponent past the width"
            (Bigint.gt e (Bigint.of_int (Dtype.bitsize dt)));
          cover "a negative exponent" (Bigint.sign e < 0);
          let folded = sym u in
          satisfies ~claim:"a constant" uop
            (fun u -> match Ops.op u with Const | Cast -> true | _ -> false)
            folded;
          equal Dtypes.const
            (pow_reference dt b e :> Dtype.const)
            (Ops.value folded));
      test "reads a float power's integer exponent as a float of its type"
        (fun () ->
          (* 16777217 rounds to the even 16777216 in a float32. *)
          let u =
            Ops.alu
              (Ops.const ~dtype:Float32 (`Float (-1.)))
              Op.Pow
              [ Ops.const (`Int (Bigint.of_int 16777217)) ]
          in
          equal Dtypes.const (`Float 1.) (Ops.value (sym u)));
    ]

let stores =
  let index = Ops.index buf [ a ] in
  let load = Ops.load index [] in
  group "stores"
    [
      test "storing what a load of the same index reads does nothing" (fun () ->
          simplifies_to (Ops.store index load) (raw Noop []);
          simplifies_to
            (Ops.store index Ops.O.(load + float (-0.)))
            (raw Noop []));
      test "storing a value other than the load's stays" (fun () ->
          let st = Ops.store index Ops.O.(load + float 1.) in
          simplifies_to st st;
          (* At -0., load + 0. is +0. *)
          let st = Ops.store index Ops.O.(load + float 0.) in
          simplifies_to st st);
      test "storing where g alt (load index) stores alt where g holds"
        (fun () ->
          simplifies_to
            (Ops.store index (Ops.where cond f load))
            (Ops.store (Ops.index buf [ Shape.valid a cond ]) f));
      test "storing where g (load index) alt stores alt where g does not hold"
        (fun () ->
          simplifies_to
            (Ops.store index (Ops.where cond load f))
            (Ops.store
               (Ops.index buf [ Shape.valid a (Ops.logical_not cond) ])
               f));
      test "storing nested selections of the load stores where both apply"
        (fun () ->
          let st =
            Ops.store index (Ops.where cond (Ops.where other load f) load)
          in
          let gate = Ops.O.(cond land Ops.logical_not other) in
          simplifies_to st (Ops.store (Ops.index buf [ Shape.valid a gate ]) f));
      test "storing invalid does nothing" (fun () ->
          simplifies_to (Ops.store index Ops.invalid) (raw Noop []));
      test "storing a gated value stores it where its gate holds" (fun () ->
          simplifies_to
            (Ops.store index (Shape.valid f cond))
            (Ops.store (Ops.index buf [ Shape.valid a cond ]) f));
    ]

let reductions =
  let int_range = Ops.cast reduce_range Int32 in
  group "reductions"
    [
      test "moves the factors that do not depend on a sum's ranges out of it"
        (fun () ->
          simplifies_to
            (Ops.reduce Ops.O.(int_range * x) Add [ reduce_range ])
            Ops.O.(Ops.reduce int_range Add [ reduce_range ] * x));
      test "moves a product that does not depend on a sum's ranges out whole"
        (fun () ->
          simplifies_to
            (Ops.reduce Ops.O.(x * y) Add [ reduce_range ])
            Ops.O.(Ops.reduce (int32 1) Add [ reduce_range ] * (x * y)));
      test "moves non-negative factors out of a maximum, and no other"
        (fun () ->
          simplifies_to
            (Ops.reduce Ops.O.(int_range * x) Max [ reduce_range ])
            Ops.O.(Ops.reduce int_range Max [ reduce_range ] * x);
          let v = var ~dtype:Int32 "v" (-3) 3 in
          let red = Ops.reduce Ops.O.(int_range * v) Max [ reduce_range ] in
          simplifies_to red red);
      cases "leaves the factors of a float sum or maximum" ~id:Op.name
        [ Op.Add; Op.Max ] (fun o ->
          let red =
            Ops.reduce
              Ops.O.(Ops.cast reduce_range Float32 * f)
              o [ reduce_range ]
          in
          simplifies_to red red);
      test "leaves the factors of a reduction other than a sum or a maximum"
        (fun () ->
          let red = Ops.reduce Ops.O.(int_range * x) Mul [ reduce_range ] in
          simplifies_to red red);
    ]

let terms =
  group "terms"
    [
      test "(x + y) * c is x * c + y * c for a weak integer" (fun () ->
          simplifies_to
            Ops.O.((a + b) * int 3)
            Ops.O.((a * int 3) + (b * int 3));
          simplifies_to
            Ops.O.(int (-1) * (a + b))
            Ops.O.((a * int (-1)) + (b * int (-1))));
      test "(x + y) * c folds each term's coefficient" (fun () ->
          let x = var "px" (-10) 10 and y = var "py" (-10) 10 in
          simplifies_to
            Ops.O.((x + (y * int 5)) * int (-1))
            Ops.O.((x * int (-1)) + (y * int (-5))));
      cases "(x + y) * c stays for floats" ~id:Float.to_string [ 3.; -1. ]
        (fun c ->
          let e = Ops.O.((f + g) * float c) in
          simplifies_to e e);
    ]

let cleanup =
  group "cleanup"
    [
      test "simplifies a conjunction as pm_simplify_valid does" (fun () ->
          simplifies_to Ops.O.((a < int 5) land (a < int 3)) Ops.O.(a < int 3));
      test "splices the sinks of a sink" (fun () ->
          simplifies_to
            (raw Sink [ raw Sink [ a; b ]; f ])
            (raw Sink [ a; b; f ]));
    ]

(* Values

   Each graph is evaluated before and after [sym] at the given bindings, where a
   rewrite that computes as unbounded integers or as real numbers would change
   the value: integers wrap at a committed width, and floats follow IEEE. *)

let keeps_machine_value (u, points) =
  let after = sym u in
  List.iter
    (fun vars ->
      let msg = Format.asprintf "%a at %a" Ops.pp u pp_binding vars in
      equal ~msg Dtypes.const (Interpreter.eval ~vars u)
        (Interpreter.eval ~vars after))
    points

let at name values = List.map (fun n -> [ (name, i n) ]) values

let pairs (n0, v0) (n1, v1) =
  List.concat_map (fun a -> List.map (fun b -> [ (n0, i a); (n1, i b) ]) v1) v0

(* At p = q = 12 the int8 product p * q wraps to -112. *)
let factor_max_negative () =
  let p = var ~dtype:Int8 "p" 0 50 and q = var ~dtype:Int8 "q" 0 50 in
  let r = Ops.range ~axis_type:Weak (Int 2) [ 3 ] in
  [
    ( Ops.reduce Ops.O.(Ops.cast r Int8 * p * q) Max [ r ],
      [ [ ("p", i 2); ("q", i 3) ]; [ ("p", i 12); ("q", i 12) ] ] );
  ]

(* At u = 255 the products r * u over r < 4 are 0, 255, 254 and 253: their
   maximum is 255, where 3 * u is 253. *)
let factor_max_wraps () =
  let u = var ~dtype:Uint8 "u" 0 255 in
  let max_of v =
    Ops.reduce
      Ops.O.(Ops.cast reduce_range (Ops.dtype v) * v)
      Max [ reduce_range ]
  in
  let top = Int32.to_int Int32.max_int in
  let w = var ~dtype:Int32 "w" 0 top in
  [ (max_of u, at "u" [ 0; 5; 255 ]); (max_of w, at "w" [ 1000; top ]) ]

(* c * c wraps to 0 at each width where c is 2^(bits / 2), or the greatest power
   of two, the least signed value. *)
let squares () =
  List.concat_map
    (fun dtype ->
      let bits = Dtype.bitsize dtype in
      let w = var ~dtype "w" 0 7 in
      let power k = Bigint.shift_left Bigint.one k in
      let top =
        match Dtype.min dtype with
        | `Int z when Bigint.sign z < 0 -> z
        | _ -> power (bits - 1)
      in
      List.map
        (fun c ->
          let c = Ops.const ~dtype (`Int c) in
          (Ops.O.(w * c * (w * c)), at "w" [ 0; 1; 7 ]))
        [ power (bits / 2); top ])
    Dtype.[ Int8; Int16; Int32; Int64; Uint8; Uint16; Uint32; Uint64 ]

(* Like terms whose weak factors sum past the width, as the greatest value and
   3: the combined factor is read at the width, so the sum, and a quotient,
   remainder or comparison of it, keeps its value. *)
let like_terms () =
  List.concat_map
    (fun dtype ->
      let bits = Dtype.bitsize dtype in
      let _, hi = Dtypes.int_bounds dtype in
      let w = var ~dtype "w" 0 7 and z = var ~dtype "z" 0 3 in
      let half = Bigint.shift_left Bigint.one (bits - 1) in
      let k n = Ops.const (`Int n) in
      let points = pairs ("w", [ 0; 1; 7 ]) ("z", [ 0; 3 ]) in
      List.concat_map
        (fun (c0, c1) ->
          List.concat_map
            (fun s ->
              List.map
                (fun u -> (u, points))
                Ops.O.
                  [
                    s;
                    s // int 2;
                    s // int 3;
                    s % int 4;
                    s // int 16;
                    s < int 16;
                    (s + int 3) // int 8;
                    Ops.ne s (int 0);
                  ])
            Ops.O.
              [
                (w * k c0) + (w * k c1);
                z + (w * k c0) + (w * k c1);
                (w * k c0) + w;
                z + w + (w * k c0);
                z + (w * k c0) + w;
              ])
        Bigint.[ (succ half, pred half); (hi, of_int 3); (hi, one) ])
    Dtype.[ Int8; Int16; Int32; Int64; Uint8; Uint16; Uint32; Uint64 ]

let integer_rows =
  let u = var ~dtype:Uint8 "u" 0 255 and y = var ~dtype:Int8 "y" 0 50 in
  let x8 = var ~dtype:Int8 "x" (-128) 127
  and z8 = var ~dtype:Int8 "z" (-128) 127 in
  let crossing = pairs ("x", [ -128; -3; 0; 5 ]) ("z", [ -128; 3 ]) in
  let v ?(dtype = Dtype.Uint8) name lo hi = var ~dtype name lo hi in
  [
    ( "an unsigned quotient by a weak -1 is a quotient by the greatest",
      fun () -> [ (Ops.O.(u // int (-1)), at "u" [ 0; 1; 5; 255 ]) ] );
    ( "a negation crosses a comparison only where it does not wrap",
      fun () ->
        let w = var "w" (-128) 0 in
        [
          (Ops.O.(x8 * int (-1) < z8 * int (-1)), crossing);
          (Ops.O.(x8 * int (-1) < int 3), crossing);
          (Ops.O.(x8 * int (-1) < Ops.int ~dtype:Int8 (-128)), crossing);
          (Ops.O.(x8 * int (-1) < int 128), crossing);
          (Ops.O.(w * int (-1) < Ops.int ~dtype:Int8 3), at "w" [ -128; -3; 0 ]);
        ] );
    ( "an offset crosses a comparison only where neither side wraps",
      fun () ->
        let points = at "u" [ 0; 5; 255 ] in
        [
          (Ops.O.(u - int 1 < int 255), points);
          (Ops.O.(u - Ops.int ~dtype:Uint8 1 < Ops.int ~dtype:Uint8 255), points);
          (Ops.O.(u + int 1 < int 1), points);
          (Ops.O.(y + int 100 < int 0), at "y" [ 10; 40 ]);
        ] );
    ( "a comparison folds from bounds only where they do not wrap",
      fun () ->
        [
          (Ops.O.(y * int 4 < int 0), at "y" [ 10; 40 ]);
          (Ops.O.(y lsl int 2 < int 0), at "y" [ 10; 40 ]);
          (Ops.O.(Ops.cast (var "w" 0 255) Int8 < int 0), at "w" [ 100; 200 ]);
          (Ops.O.(u lxor int (-1) < int 0), at "u" [ 0; 5 ]);
        ] );
    ( "a chain of integer casts is one cast only where the value fits",
      fun () ->
        [
          ( Ops.cast (Ops.cast Ops.O.(y + int 100) Uint8) Int32,
            at "y" [ 10; 40 ] );
        ] );
    ("a factor leaves a maximum only where no product wraps", factor_max_wraps);
    ( "a factor leaves a maximum only where it is not negative",
      factor_max_negative );
    ("a square of a product of constants", squares);
    ("like terms combine their factors at their width", like_terms);
    ( "a weak operand is read at the width of the operation",
      fun () ->
        let b = v "b" 0 1 and d = v "d" 1 2 and a = v "a" 0 1 in
        let b32 = v ~dtype:Uint32 "b" 0 1 in
        [
          (Ops.O.(Ops.maximum (int 1) b32 // int (-2)), at "b" [ 0; 1 ]);
          ( Ops.O.(Ops.where (int (-3) < b) a (int (-3)) // d),
            [ [ ("a", i 0); ("b", i 1); ("d", i 2) ] ] );
          (Ops.maximum (Ops.int (-3)) a, at "a" [ 0; 1 ]);
        ] );
    ( "a folded result is written at its width",
      fun () ->
        let b = v "b" 0 1 and c = v "c" 2 2 in
        [
          ( Ops.O.(Ops.maximum ~-(Ops.maximum (b % int 2) ~-c) b),
            pairs ("b", [ 0; 1 ]) ("c", [ 2 ]) );
        ] );
    ( "nested maxima fold their constants at the width of the operation",
      fun () ->
        let w = v "w" 0 255 in
        [
          ( Ops.O.(Ops.maximum (Ops.maximum w (int 1)) (int (-3))),
            at "w" [ 0; 254 ] );
        ] );
  ]

(* Each graph is evaluated with [f], [g] and [h] bound to the given float32
   values. *)
let keeps_float_bits (u, points) =
  let after = sym u in
  List.iter
    (fun values ->
      let params =
        List.mapi (fun k v -> (List.nth [ 0; 1; 3 ] k, float32 v)) values
      in
      let msg =
        Format.asprintf "%a at %a" Ops.pp u
          (Format.pp_print_list
             ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
             Format.pp_print_float)
          values
      in
      equal ~msg same_float
        (Interpreter.eval ~params u)
        (Interpreter.eval ~params after))
    points

let h = Shape.param 3 Dtype.Float32
let inf = Float.infinity
let below_one = Ops.O.(f < float 1.)
let sigmoid = Ops.reciprocal Ops.O.(float 1. + f)

(* A Dekker split of [x] into a high part and the rest. *)
let split x =
  let c = Ops.O.(x * float 4097.) in
  let hi = Ops.O.(c - (c - x)) in
  (hi, Ops.O.(x - hi))

let float_rows =
  let tiny = Float.ldexp 1. (-24) in
  [
    ( "identities that IEEE does not keep",
      fun () ->
        [
          (Ops.O.(f + float 0.), [ [ -0. ] ]);
          (Ops.O.(f * float 0.), [ [ inf ]; [ -1. ]; [ Float.nan ] ]);
          (Ops.O.(f / f), [ [ 0. ]; [ inf ] ]);
          (Ops.O.(f * g / g), [ [ 1e10; 1e30 ] ]);
          (Ops.O.(f / g / h), [ [ 1e20; 1e20; 1e20 ] ]);
        ] );
    ( "a negated sum and complementary selections of signed zeros",
      fun () ->
        [
          (Ops.O.(float (-1.) * (f + float 3.)), [ [ -3. ] ]);
          (Ops.O.(float (-1.) * (f + g)), [ [ 1.; -1. ] ]);
          ( Ops.O.(
              Ops.where below_one g (float 0.)
              + Ops.where below_one (float 0.) h),
            [ [ 0.; -0.; 5. ] ] );
        ] );
    ( "constants, like terms and selections of a sum regrouped",
      fun () ->
        let r = Ops.range ~axis_type:Weak (Int 3) [ 2 ] in
        let small = Ops.where below_one (Ops.float tiny) (Ops.float 0.) in
        [
          (Ops.O.(f + float 1e8 + float (-1e8)), [ [ 1. ] ]);
          (Ops.O.(f * float 1e30 * float 1e-30), [ [ 1e10 ] ]);
          (Ops.O.(f + float 1e8 + g), [ [ 3.; 3. ] ]);
          (Ops.O.(g + f + f), [ [ tiny; 1. ] ]);
          (Ops.O.((f * float 1.1) + (f * float 2.2)), [ [ 1. ] ]);
          (Ops.O.(f + (f * float (-1.))), [ [ -1. ]; [ inf ] ]);
          (Ops.O.(g + (f * float 4.) + (f * float (-3.))), [ [ 1e38; 1. ] ]);
          ( Ops.O.(f * float (-1.) * g),
            [ [ 3.; 1e30 ]; [ -0.; 2. ]; [ inf; 0. ]; [ Float.nan; 1. ] ] );
          (Ops.O.(g + small + small), [ [ 0.; 1. ] ]);
          ( Ops.reduce Ops.O.(Ops.cast r Float32 * f * g) Add [ r ],
            [ [ 2e38; 0.25 ] ] );
        ] );
    ( "reciprocals of products and sigmoid forms",
      fun () ->
        [
          (Ops.reciprocal Ops.O.(f * f), [ [ 1e20 ] ]);
          (Ops.reciprocal Ops.O.(f * f * f), [ [ 1e13 ] ]);
          (Ops.reciprocal Ops.O.(f * float 2.), [ [ 1e38 ] ]);
          (Ops.reciprocal Ops.O.(f * float 1e10), [ [ 1e30 ] ]);
          (Ops.O.(f * sigmoid), [ [ 1e-8 ]; [ inf ] ]);
          (Ops.O.(f * (sigmoid * g)), [ [ 1e-8; 1. ] ]);
          (Ops.O.(f * (sigmoid + g)), [ [ 1e-8; 1. ] ]);
        ] );
    ( "maxima of a NaN operand and of zeros",
      fun () ->
        let nan = Ops.O.(float inf + float (-.inf)) in
        [
          (Ops.maximum nan (Ops.maximum f (Ops.float 0.)), [ [ 1. ] ]);
          (Ops.where Ops.O.(f < float (-0.)) (Ops.float 0.) f, [ [ -1e-45 ] ]);
          (Ops.maximum f (Ops.float inf), [ [ Float.nan ] ]);
        ] );
    ( "a Dekker split and product",
      fun () ->
        let fh, fl = split f and gh, gl = split g in
        let p = Ops.O.(f * g) in
        let err = Ops.O.((fh * gh) - p + (fh * gl) + (fl * gh) + (fl * gl)) in
        let points = [ [ 1.1; 3.14159 ]; [ 0.1; 7.3 ]; [ 1e10; 1e-10 ] ] in
        List.map (fun u -> (u, points)) [ fh; fl; err ] );
  ]

let values =
  group "values"
    [
      cases "keeps an integer graph's value at its width, wrapping included"
        ~id:fst integer_rows (fun (_, graphs) ->
          List.iter keeps_machine_value (graphs ()));
      cases "keeps a float graph's value bit for bit" ~id:fst float_rows
        (fun (_, graphs) -> List.iter keeps_float_bits (graphs ()));
      test "folds the square of a product to 0 where it wraps to 0 unsigned"
        (fun () ->
          List.iter
            (fun dtype ->
              let bits = Dtype.bitsize dtype in
              let w = var ~dtype "w" 0 7 in
              List.iter
                (fun k ->
                  let c =
                    Ops.const ~dtype (`Int (Bigint.shift_left Bigint.one k))
                  in
                  equal
                    ~msg:(Format.asprintf "%a, 2^%d" Dtype.pp dtype k)
                    uop
                    (Ops.const ~dtype (`Int Bigint.zero))
                    (sym Ops.O.(w * c * (w * c))))
                [ bits / 2; bits - 1 ])
            Dtype.[ Uint8; Uint16; Uint32; Uint64 ]);
      test "keeps the low part of a Dekker split" (fun () ->
          let _, fl = split f in
          not_equal same_float (`Float 0.)
            (Interpreter.eval ~params:[ (0, float32 1.1) ] (sym fl)));
      test "keeps an int64 operation whose int32 form has no value at int64"
        (fun () ->
          (* An int32 shift by 32 or more, and the least int32's remainder by
             -1, have no value in C, where their int64 forms do: -1 >> 32 is -1
             and -2^31 % -1 is 0 at int64. *)
          let at_int32 u =
            List.filter
              (fun n ->
                Op.Set.mem (Ops.op n) Op.Set.binary
                && Dtype.equal (Ops.dtype n) Int32)
              (Ops.toposort ~calls:Skip (sym u))
          in
          let long x = Ops.cast x Int64 in
          let small = var ~dtype:Int8 "s" (-128) 127 in
          equal (list uop) []
            (at_int32 Ops.O.(long small lsr Ops.int ~dtype:Int64 32));
          let least =
            Ops.variable ~dtype:Int32 "a" (Dtype.min Int32) (Dtype.max Int32)
          in
          let divisor = var ~dtype:Int8 "d" (-2) (-1) in
          List.iter
            (fun rem ->
              equal (list uop) [] (at_int32 (rem (long least) (long divisor))))
            [ Ops.O.( % ); (fun x y -> Ops.alu x Truncmod [ y ]) ]);
    ]

(* Laws *)

let laws =
  let open Expressions in
  group "laws"
    [
      prop "keeps the value of an integer expression where nothing wraps"
        scenario (fun s ->
          let e = node s in
          let checked = keeps_value ~name:"sym" e (sym e) in
          (* About nine cases in ten check a binding: a committed variable at
             its type's extreme often wraps at every binding, which this law
             skips. 0.8 keeps the decision clear of that share and far above the
             2 to 9 % a law that checks nothing would reach. *)
          cover ~share:0.8 "a binding was checked" (checked > 0));
      prop ~count:1000
        "keeps the value of an integer expression at a committed width, \
         wrapping included"
        committed_scenario (fun s ->
          let e = node s in
          let checked = keeps_value ~wrapping:true ~name:"sym" e (sym e) in
          cover ~share:0.9 "a binding was checked" (checked > 0);
          List.iter
            (fun dt -> cover (Dtype.name dt) (Dtype.equal dt s.dtype))
            Dtype.[ Int8; Uint8; Int16; Uint16; Int32; Uint32; Int64; Uint64 ];
          let least, greatest = Dtypes.int_bounds s.dtype in
          cover "a variable at the type's extreme"
            (List.exists
               (fun (lo, hi) ->
                 Bigint.equal lo least || Bigint.equal hi greatest)
               s.vars));
      prop "keeps a float expression's value bit for bit at special values"
        float_scenario (fun (dt, e) ->
          List.iter
            (fun t -> cover (Dtype.name t) (Dtype.equal t dt))
            Dtype.[ Float16; Bfloat16; Float32; Float64 ];
          let e = float_node dt e in
          keeps_float_value ~dtype:dt ~name:"sym" e (sym e));
      prop "is idempotent" scenario (fun s -> Law.idempotent uop sym (node s));
    ]

(* Cost *)

(* [selections name n] is [n] selections in sequence over a value read from
   memory. Each condition is in its true branch, so the where-closure rule looks
   at every selection, and keeps it, since the value reaches an INDEX. *)
let selections name n =
  let rec link x k =
    if k = n then x
    else
      let c = Ops.O.(x < Ops.float ~dtype:Float32 (float_of_int k)) in
      let t = Ops.O.(x + Ops.cast c Dtype.Float32) in
      link (Ops.where c t Ops.O.(x * Ops.float ~dtype:Float32 2.)) (k + 1)
  in
  link (Ops.load (Ops.index buf [ var name 0 15 ]) []) 0

(* [far_conditions name n] is [n] selections in sequence over a value read from
   memory, all on one condition built before the chain: asking whether a branch
   holds it walks the chain below the branch. The branches reach an INDEX, which
   rejects each selection without that walk. *)
let far_conditions name n =
  let x = Ops.load (Ops.index buf [ var name 0 15 ]) [] in
  let c = Ops.O.(x < Ops.float ~dtype:Float32 3.) in
  let rec link x acc k =
    if k = n then acc
    else
      let x = Ops.O.(x + Ops.float ~dtype:Float32 1.) in
      link x (Ops.where c x acc) (k + 1)
  in
  link x x 0

(* [loaded_conditions name n] is [far_conditions] over a load of four elements,
   whose address is a shrink of the buffer: the condition reads memory without
   an INDEX. *)
let loaded_conditions name n =
  let x =
    Ops.load (Ops.v Op.Shrink ~src:[ buf; var name 0 12; Ops.int 4 ]) []
  in
  let lanes u = Ops.broadcast u 4 in
  let c = Ops.O.(x < lanes (Ops.float ~dtype:Float32 3.)) in
  let rec link x acc k =
    if k = n then acc
    else
      let x = Ops.O.(x + lanes (Ops.float ~dtype:Float32 1.)) in
      link x (Ops.where c x acc) (k + 1)
  in
  link x x 0

(* [words f] is the words [f ()] allocates. A full collection first, and a minor
   heap that holds every word [f] allocates, keep the collector from freeing,
   during [f], nodes [f] would otherwise find interned: the count is then [f]'s
   own, whenever it runs. *)
let words f =
  let gc = Gc.get () in
  Gc.full_major ();
  Gc.set { gc with minor_heap_size = 8 lsl 20 };
  let before = Gc.minor_words () in
  ignore (Sys.opaque_identity (f ()));
  let words = Gc.minor_words () -. before in
  Gc.set gc;
  words

(* Linear work is [a * n + b] with [b >= 0]: at most twice the work for twice
   the length, with a tenth of slack for tables that double their capacity at
   different lengths. *)
let linear chain =
  let work n =
    let e = chain (string_of_int n) n in
    words (fun () -> sym e)
  in
  let short = work 250 and long = work 500 in
  less float_exact ~than:(2.2 *. short) long

let cost =
  group "work"
    [
      test "is linear in a chain of selections" (fun () ->
          linear (fun name -> selections ("chain" ^ name)));
      test "is linear in a chain of selections under one condition" (fun () ->
          linear (fun name -> far_conditions ("far" ^ name)));
      test
        "is linear in a chain of selections under one condition read from \
         memory without an index" (fun () ->
          linear (fun name -> loaded_conditions ("loaded" ^ name)));
    ]

let sym_group =
  group "sym"
    [
      powers;
      stores;
      reductions;
      terms;
      cleanup;
      values;
      laws;
      cost;
      Recorded.simplifications;
      Recorded.random_expressions;
    ]

let () =
  exit
    (run "Tolk.Symbolic"
       [
         given_valid;
         simplify_valid;
         simplify_valid_pm;
         drop_and_clauses;
         move_where_on_load;
         clean_up_group_sink;
         sym_group;
       ])
