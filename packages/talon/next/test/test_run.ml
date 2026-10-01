(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next
open Windtrap
module G = Talon_gen
module R = Reference

let schema_w = Testable.make ~pp:Schema.pp ~equal:Schema.equal
let run_ok q = require_ok ~pp:Error.pp (Query.run q)
let value ty = Option.get (G.value ty)

let message f =
  match f () with _ -> "no exception" | exception Invalid_argument m -> m

let map2 f a b =
  Gen.(
    let+ x = a and+ y = b in
    f x y)

let rec all = function
  | [] -> Gen.constant []
  | g :: gs -> map2 List.cons g (all gs)

(* Generated plans

   Tables of the types below, and expressions of one type each, as the reference
   writes them. Arithmetic stays within 32 bits, where the reference computes
   exactly. *)

let palette =
  Type.
    [
      Any int8;
      Any int16;
      Any int32;
      Any uint8;
      Any uint32;
      Any int64;
      Any float16;
      Any float32;
      Any float64;
      Any bool;
      Any string;
      Any (categorical [| "a"; "b"; "é" |]);
      Any binary;
      Any date;
      Any (list int64);
      Any (record [ ("x", Any int8); ("s", Any string) ]);
    ]

let table s =
  Gen.bind (Gen.int_range 0 12) (fun n ->
      let column (name, Type.Any ty) =
        Gen.map
          (fun vs -> (name, Column.of_options ty vs))
          (Gen.array ~size:(Gen.constant n) (Gen.option (value ty)))
      in
      Gen.map v (all (List.map column s)))

let schemas =
  Gen.bind (Gen.int_range 1 4) (fun k ->
      Gen.map
        (List.mapi (fun i t -> (Printf.sprintf "c%d" i, t)))
        (all (List.init k (fun _ -> Gen.of_list palette))))

let names ty s =
  List.filter_map
    (fun (n, Type.Any t) -> if Type.equal t ty then Some n else None)
    s

(* [pair a b] draws an expression of [a] and one of [b], in either order. *)
let pair a b =
  map2
    (fun flip (x, y) -> if flip then (y, x) else (x, y))
    Gen.bool (Gen.pair a b)

(* [anchored ty s d] reads a column of type [ty] of [s]; [operand ty s d] is
   such an expression, a literal or a null, which meets one. *)
let rec anchored : type a. a Type.t -> _ -> int -> a R.expr Gen.t =
 fun ty s d ->
  let col = Gen.map (fun n -> R.Col (ty, n)) (Gen.of_list (names ty s)) in
  if d = 0 then col
  else
    let two = pair (anchored ty s (d - 1)) (operand ty s (d - 1)) in
    let is k = Kind.provably_equal (Type.kind ty) k in
    let arith : a R.expr Gen.t list =
      match (is Kind.int, is Kind.float, ty) with
      | Some Equal, _, (Int8 | Int16 | Int32 | Uint8 | Uint32) ->
          let op = Gen.of_list R.[ Add; Sub; Mul; Div; Mod ] in
          [ map2 (fun op (a, b) -> R.Int (op, a, b)) op two ]
      | _, Some Equal, (Float32 | Float64) ->
          let op = Gen.of_list R.[ Fadd; Fsub; Fmul; Fdiv ] in
          [ map2 (fun op (a, b) -> R.Float (op, a, b)) op two ]
      | _ -> []
    in
    Gen.one_of
      (col
      :: map2 (fun c (a, b) -> R.If (c, a, b)) (predicate s (d - 1)) two
      :: Gen.map (fun (a, b) -> R.Coalesce [ a; b ]) two
      :: arith)

and operand : type a. a Type.t -> _ -> int -> a R.expr Gen.t =
 fun ty s d ->
  let literal =
    match (ty, R.literal ty) with
    | Float16, _ | _, None -> []
    | _, Some _ -> [ (2, Gen.map (fun v -> R.Lit (ty, v)) (value ty)) ]
  in
  Gen.frequency
    ((3, anchored ty s d) :: (1, Gen.constant (R.Null ty)) :: literal)

and predicate s d : bool R.expr Gen.t =
  let compared =
    Gen.bind (Gen.of_list s) (fun (_, Type.Any ty) ->
        let a = anchored ty s d in
        let op = Gen.of_list [ `Eq; `Ne; `Lt; `Le; `Gt; `Ge ] in
        let values = Gen.list ~size:(Gen.int_range 0 3) (value ty) in
        Gen.one_of
          [
            map2
              (fun op (a, b) -> R.Cmp (op, a, b))
              op
              (pair a (operand ty s d));
            Gen.map (fun a -> R.Is_null a) a;
            map2 (fun vs a -> R.Is_in (vs, a)) values a;
          ])
  in
  if d = 0 then compared
  else
    let p = predicate s (d - 1) in
    let constants =
      R.[ Lit (Type.bool, true); Lit (Type.bool, false); Null Type.bool ]
    in
    let truth = Gen.frequency [ (3, p); (1, Gen.of_list constants) ] in
    let logic conj (a, b) = if conj then R.And (a, b) else R.Or (a, b) in
    Gen.frequency
      [
        (2, compared);
        (1, map2 logic Gen.bool (pair p truth));
        (1, Gen.map (fun a -> R.Not a) p);
      ]

let output s name =
  Gen.bind (Gen.of_list s) (fun (_, Type.Any ty) ->
      let out e = R.Out (name, e) in
      Gen.frequency
        [
          (3, Gen.map out (anchored ty s 2));
          (1, Gen.map out (predicate s 1));
          ( 1,
            Gen.of_list
              [ out (R.Lit (Type.int64, 7)); out (R.Lit (Type.string, "é")) ] );
        ])

let outputs s names = all (List.map (output s) names)

(* Each step names its new columns after its level, so that they differ from its
   input's. *)
let step level p =
  let s = R.schema p in
  let fresh prefix k = List.init k (Printf.sprintf "%s%d_%d" prefix level) in
  let select =
    Gen.bind (Gen.int_range 1 3) (fun k ->
        map2
          (fun keep os ->
            R.Select ((if keep = [] then os else R.Keep keep :: os), p))
          (Gen.subsequence (List.map fst s))
          (outputs s (fresh "s" k)))
  in
  let derive =
    Gen.bind
      (Gen.pair (Gen.subsequence (List.map fst s)) (Gen.int_range 0 2))
      (fun (replaced, k) ->
        Gen.map
          (fun os -> R.Derive (os, p))
          (outputs s (replaced @ fresh "d" k)))
  in
  let filter =
    let constant = Gen.map (fun b -> R.Lit (Type.bool, b)) Gen.bool in
    Gen.map
      (fun e -> R.Filter (e, p))
      (Gen.frequency [ (4, predicate s 2); (1, constant) ])
  in
  let slice =
    map2
      (fun offset length -> R.Slice { offset; length; plan = p })
      (Gen.int_range (-6) 14) (Gen.int_range 0 14)
  in
  let append =
    map2
      (fun t names -> R.Append (p, R.Select ([ R.Keep names ], R.Table t)))
      (table s)
      (Gen.permutation (List.map fst s))
  in
  Gen.one_of [ select; derive; filter; slice; append ]

let rec plan level =
  if level = 0 then Gen.map (fun t -> R.Table t) (Gen.bind schemas table)
  else Gen.bind (plan (level - 1)) (step level)

(* [split p] is [p] with each table cut into batches. *)
let rec split : R.plan -> R.plan Gen.t = function
  | Table t -> Gen.map (fun t -> R.Table t) (G.split t)
  | Select (os, p) -> Gen.map (fun p -> R.Select (os, p)) (split p)
  | Derive (os, p) -> Gen.map (fun p -> R.Derive (os, p)) (split p)
  | Filter (e, p) -> Gen.map (fun p -> R.Filter (e, p)) (split p)
  | Slice s -> Gen.map (fun plan -> R.Slice { s with plan }) (split s.plan)
  | Append (p, r) -> map2 (fun p r -> R.Append (p, r)) (split p) (split r)

let pp_plan ppf p = Query.pp ppf (R.query p)
let plans = Gen.bind (Gen.int_range 0 3) plan
let split_plans = Gen.with_pp pp_plan (Gen.bind plans split)

(* The run against the reference *)

let holds t (n, R.Column (ty, vs)) =
  equal ~msg:n
    (array (option (G.witness ty)))
    vs
    (Column.options (Type.kind ty) (column t n))

let agrees p =
  let t = run_ok (R.query p) in
  cover "rows in the result" (rows t > 0);
  at_most int ~than:1 (List.length (batches t));
  equal schema_w (Schema.v (R.schema p)) (schema t);
  List.iter (holds t) (R.run p)

(* Law 7: canonical layouts, byte for byte *)

let hex (type a b) (x : (a, b) Nx.t) =
  let bytes =
    match Nx.dtype x with
    | Bool -> Nx.cast Nx.uint8 x
    | _ -> Nx.flatten (Nx.bitcast Nx.uint8 x)
  in
  String.concat ""
    (List.map (Printf.sprintf "%02x") (Array.to_list (Nx.to_array bytes)))

let rec buffers c =
  let bits = function
    | None -> "no validity"
    | Some b ->
        let bytes, offset = Nx_bits.bytes b in
        Printf.sprintf "bit %d of %s" offset (hex bytes)
  in
  match Column.layout c with
  | Fixed { validity; values = P x } -> [ bits validity; hex x ]
  | Varsize { validity; offsets; child } ->
      bits validity :: hex offsets :: buffers child
  | Children { validity; length; fields } ->
      bits validity :: string_of_int length
      :: List.concat_map (fun (n, c) -> n :: buffers c) fields

let two_splits =
  Gen.with_pp
    (fun ppf (p, _) -> pp_plan ppf p)
    (Gen.bind plans (fun p -> Gen.pair (split p) (split p)))

let same_layouts (p0, p1) =
  let t0 = run_ok (R.query p0) and t1 = run_ok (R.query p1) in
  List.iter
    (fun (n, _) ->
      equal ~msg:n (list string) (buffers (column t0 n)) (buffers (column t1 n)))
    (R.schema p0)

(* Values *)

type expr = E : 'a R.expr -> expr

let values_cases =
  Gen.with_pp
    (fun ppf (p, _) -> pp_plan ppf p)
    (Gen.bind split_plans (fun p ->
         let s = R.schema p in
         Gen.map
           (fun e -> (p, e))
           (Gen.bind (Gen.of_list s) (fun (_, Type.Any ty) ->
                Gen.map (fun e -> E e) (anchored ty s 1)))))

let values_agree (p, E e) =
  match (R.values e p, Query.values (R.expr e) (R.query p)) with
  | Ok expected, Ok actual ->
      equal (array (G.witness (R.type_of e))) expected actual
  | Error row, Error err ->
      contains
        ~sub:(Printf.sprintf ": row %d: the value is null" row)
        (Format.asprintf "%a" Error.pp err)
  | Ok _, Error err -> failf "values failed: %a" Error.pp err
  | Error row, Ok _ -> failf "no failure at the null of row %d" row

let laws =
  group "Laws"
    [
      prop "run gives the reference's rows, whatever the batches" split_plans
        agrees;
      prop "run's layouts are the same bytes whatever the batches" two_splits
        same_layouts;
      prop "values gives the reference's values, or fails at its first null"
        values_cases values_agree;
    ]

(* Cases from the specification *)

let t, f = (Some true, Some false)

let truths =
  v
    [
      ("a", Column.of_options Type.bool [| t; t; t; f; f; f; None; None; None |]);
      ("b", Column.of_options Type.bool [| t; f; None; t; f; None; t; f; None |]);
    ]

(* [result e t] is the column of [e] over [t]'s rows. *)
let result e t =
  column (run_ok (Query.select Expr.[ "r" := e ] (Query.of_table t))) "r"

let rows_are k expected c = equal (array (option (G.witness k))) expected c
let bools e t = Column.options Kind.bool (result e t)
let ints e t = Column.options Kind.int (result e t)

let a = Col.bool "a"
and b = Col.bool "b"

let kleene =
  group "Kleene logic"
    [
      test "a && b is false where either is false, else null where one is"
        (fun () ->
          rows_are Type.bool
            [| t; f; None; f; f; f; None; f; None |]
            (bools Expr.(a && b) truths));
      test "a || b is true where either is true, else null where one is"
        (fun () ->
          rows_are Type.bool
            [| t; t; t; t; f; None; t; None; None |]
            (bools Expr.(a || b) truths));
      test "not a is null where a is" (fun () ->
          rows_are Type.bool
            [| f; f; f; t; t; t; None; None; None |]
            (bools (Expr.not a) truths));
    ]

let int64s xs = Column.of_tensor (Nx.create Nx.int64 [| Array.length xs |] xs)

let arithmetic =
  let x = Col.int "x" and y = Col.int "y" in
  let small =
    v
      [
        ("x", Column.v Type.int32 [| 7; -7; 7 |]);
        ("y", Column.v Type.int32 [| 2; 0; -2 |]);
      ]
  in
  let least =
    v [ ("x", int64s [| Int64.min_int |]); ("y", int64s [| -1L |]) ]
  in
  let tensor e = Nx.to_array (Column.to_tensor Nx.int64 (result e least)) in
  group "Integer division"
    [
      test "x / y truncates toward zero and is null where y is zero" (fun () ->
          rows_are Type.int32
            [| Some 3; None; Some (-3) |]
            (ints Expr.(x / y) small));
      test "x mod y has x's sign and is null where y is zero" (fun () ->
          rows_are Type.int32 [| Some 1; None; Some 1 |]
            (ints Expr.(x mod y) small));
      test "the least int64 divided by -1 wraps to itself" (fun () ->
          equal (array int64) [| Int64.min_int |] (tensor Expr.(x / y)));
      test "the least int64 mod -1 is 0" (fun () ->
          equal (array int64) [| 0L |] (tensor Expr.(x mod y)));
      test "a subexpression read twice is computed once, with one value"
        (fun () ->
          rows_are Type.int32
            [| Some 81; Some 49; Some 25 |]
            (ints
               Expr.(
                 let s = x + y in
                 s * s)
               small));
    ]

let widening =
  let mixed =
    v
      [
        ("a", Column.v Type.int8 [| -1; 5 |]);
        ("b", Column.v Type.int16 [| -1; 300 |]);
        ("c", Column.v Type.uint8 [| 255; 5 |]);
        ("k", Column.v (Type.categorical [| "x"; "y" |]) [| "y"; "x" |]);
        ("s", Column.v Type.string [| "y"; "z" |]);
      ]
  in
  let i = Col.int and s = Col.string in
  let compares name e expected =
    test name (fun () -> rows_are Type.bool expected (bools e mixed))
  in
  group "Operands of two types"
    [
      compares "int8 = int16 compares at int16" Expr.(i "a" = i "b") [| t; f |];
      compares "int8 < int16 compares at int16" Expr.(i "a" < i "b") [| f; t |];
      compares "uint8 < int16 compares at int16" Expr.(i "c" < i "b") [| f; t |];
      compares "a categorical = a string compares as text"
        Expr.(s "k" = s "s")
        [| t; f |];
      compares "a categorical < a string compares as text"
        Expr.(s "k" < s "s")
        [| f; t |];
    ]

let failures =
  let in_two_batches cs0 cs1 = of_batches [ v cs0; v cs1 ] in
  let x = Col.int "x" in
  let celsius =
    Ext.v ~name:"celsius" ~ordered:true Type.float64
      ~dec:(fun c -> if Float.is_nan c then raise Exit else c)
      ~enc:Fun.id
  in
  let degrees cs =
    let valid =
      Nx.create Nx.bool [| Array.length cs |] (Array.map Option.is_some cs)
    in
    let c =
      Column.of_tensor ~validity:(Nx_bits.of_bool valid)
        (Nx.create Nx.float64
           [| Array.length cs |]
           (Array.map (Option.value ~default:0.) cs))
    in
    let ty = Type.Any (Type.ext ~name:"celsius" Type.float64) in
    v [ ("t", Result.get_ok (Column.of_layout ty (Column.layout c))) ]
  in
  let error r = Format.asprintf "%a" Error.pp (require_error r) in
  group "Failures"
    [
      test "values fails at a value outside int, counted over the batches"
        (fun () ->
          let big =
            in_two_batches
              [ ("x", int64s [| 1L; 2L |]) ]
              [ ("x", int64s [| 3L; 0x4000_0000_0000_0000L |]) ]
          in
          expect (error (Query.values x (Query.of_table big)))
          @@ __POS_OF__
               {| values x: row 3: 4611686018427387904 is outside int. |});
      test "values fails at a null" (fun () ->
          let nulls =
            v [ ("x", Column.of_options Type.int8 [| Some 1; None |]) ]
          in
          expect (error (Query.values x (Query.of_table nulls)))
          @@ __POS_OF__
               {| values x: row 1: the value is null; read it through Expr.option. |});
      test "values decodes extension values with their declaration" (fun () ->
          equal (array float_exact) [| 1.5; -2. |]
            (require_ok ~pp:Error.pp
               (Query.values (Ext.col celsius "t")
                  (Query.of_table (degrees [| Some 1.5; Some (-2.) |])))));
      test "a null before a raising declaration fails the run first" (fun () ->
          let t = degrees [| None; Some Float.nan |] in
          expect (error (Query.values (Ext.col celsius "t") (Query.of_table t)))
          @@ __POS_OF__
               {| values t: row 0: the value is null; read it through Expr.option. |});
      test "a raising declaration before a null raises" (fun () ->
          let t = degrees [| Some Float.nan; None |] in
          raises Exit (fun () ->
              Query.values (Ext.col celsius "t") (Query.of_table t)));
    ]

let refusals =
  let t = v [ ("a", Column.v Type.int64 [| 2; 1 |]) ] in
  group "Not yet lowered"
    [
      test "a sort is refused, naming the step" (fun () ->
          expect
            (message (fun () ->
                 Query.run (Query.sort [ Order.asc "a" ] (Query.of_table t))))
          @@ __POS_OF__ {| sort [asc "a"] is not implemented yet |});
      test "a rank is refused, naming the expression" (fun () ->
          expect
            (message (fun () ->
                 Query.run
                   (Query.derive
                      Expr.[ "r" := over (rank (Col.int "a")) ]
                      (Query.of_table t))))
          @@ __POS_OF__ {| over (rank a) is not implemented yet |});
    ]

let () =
  exit (run "Run" [ laws; kleene; arithmetic; widening; failures; refusals ])
