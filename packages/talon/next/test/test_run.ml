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

   Tables of the types below, whose operands meet at their common type, and
   expressions as the reference writes them. Arithmetic stays within 32 bits,
   where the reference computes exactly. Each type that changes its values'
   representation when it widens (decimals, clocks) appears once, since those
   conversions are not lowered yet. OCaml functions hash their argument: one in
   about [p] raises [Boom], and the others give values that an [int8] or an
   unsigned type may not hold. *)

exception Boom of (int * int)

let palette =
  Type.
    [
      Any int8;
      Any int16;
      Any int32;
      Any int64;
      Any uint8;
      Any uint16;
      Any uint32;
      Any uint64;
      Any float16;
      Any float32;
      Any float64;
      Any bool;
      Any string;
      Any (categorical [| "b"; "a" |]);
      Any (categorical [| "b"; "a"; "é" |]);
      Any binary;
      Any (decimal ~precision:10 ~scale:2);
      Any date;
      Any (clock Us);
      Any (duration Ns);
      Any (datetime ~zone:"UTC" Ms);
      Any (tensor Nx.float32 [| 2 |]);
      Any (list int64);
      Any (record [ ("x", Any int8); ("s", Any string) ]);
      Any (ext ~name:"celsius" float64);
    ]

(* [expressible s] is the columns of [s] that a handle reads: all but
   extensions. *)
let expressible s =
  List.filter
    (fun (_, Type.Any t) ->
      Option.is_some (Kind.provably_equal (Type.kind t) (Type.kind t)))
    s

let rec drawn : type a. a Type.t -> int -> Column.t Gen.t =
 fun ty n ->
  match ty with
  | Ext { storage; _ } ->
      let ext c = Result.get_ok (Column.of_layout (Any ty) (Column.layout c)) in
      Gen.map ext (drawn storage n)
  | _ ->
      Gen.map (Column.of_options ty)
        (Gen.array ~size:(Gen.constant n) (Gen.option (value ty)))

let table s =
  Gen.bind (Gen.int_range 0 12) (fun n ->
      let named (name, Type.Any ty) =
        Gen.map (fun c -> (name, c)) (drawn ty n)
      in
      Gen.map v (all (List.map named s)))

let schemas =
  let named = List.mapi (fun i t -> (Printf.sprintf "c%d" i, t)) in
  Gen.bind (Gen.int_range 0 3) (fun k ->
      Gen.map named
        (all
           (Gen.of_list (List.map snd (expressible (named palette)))
           :: List.init k (fun _ -> Gen.of_list palette))))

let names ty s =
  List.filter_map
    (fun (n, Type.Any t) -> if Type.equal t ty then Some n else None)
    s

(* [kin ty s] is the types of [s]'s columns of [ty]'s kind; [meeting] those that
   meet [ty], and [within] those that [ty] contains. *)
let kin : type a. a Type.t -> _ -> a Type.t list =
 fun ty s ->
  List.filter_map
    (fun (_, Type.Any t) ->
      match Kind.provably_equal (Type.kind t) (Type.kind ty) with
      | Some Equal -> Some (t : a Type.t)
      | None -> None)
    s

let meeting ty s =
  List.filter (fun t -> Option.is_some (Type.common [ ty; t ])) (kin ty s)

let within ty s =
  List.filter
    (fun t -> Option.equal Type.equal (Type.common [ ty; t ]) (Some ty))
    (kin ty s)

(* [pair a b] draws an expression of [a] and one of [b], in either order. *)
let pair a b =
  map2
    (fun flip (x, y) -> if flip then (y, x) else (x, y))
    Gen.bool (Gen.pair a b)

(* [anchored ty s d] is an expression of type [ty] that reads a column of [s];
   [operand ty s d] is one, a literal or a null, which meets it. *)
let rec anchored : type a. a Type.t -> _ -> int -> a R.expr Gen.t =
 fun ty s d ->
  let col = Gen.map (fun n -> R.Col (ty, n)) (Gen.of_list (names ty s)) in
  if d = 0 then col
  else
    let narrower =
      Gen.bind (Gen.of_list (within ty s)) (fun t -> anchored t s (d - 1))
    in
    let two =
      Gen.one_of
        [
          pair (anchored ty s (d - 1)) (operand ty s (d - 1));
          pair narrower (anchored ty s (d - 1));
        ]
    in
    let is k = Kind.provably_equal (Type.kind ty) k in
    let lifts : a R.expr Gen.t list =
      match is Kind.int with
      | Some Equal -> [ lift (ty : int Type.t) s (d - 1) ]
      | None -> []
    in
    let arith : a R.expr Gen.t list =
      match (is Kind.int, is Kind.float, ty) with
      | Some Equal, _, (Int8 | Int16 | Int32 | Uint8 | Uint16 | Uint32) ->
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
      :: Gen.map (fun a -> R.Store (ty, a)) narrower
      :: (arith @ lifts))

and lift : int Type.t -> _ -> int -> int R.expr Gen.t =
 fun ty s d ->
  Gen.bind
    (Gen.of_list (expressible s))
    (fun (_, Type.Any t) ->
      let ints =
        Gen.(triple (int_range 2 10) (int_range 1 300) (int_range 0 99))
      in
      map2
        (fun (p, k, label) (bind, a) ->
          let value h = (h mod k) - (k / 4) in
          if bind then
            let f = function
              | None -> Some k
              | Some v ->
                  let h = Hashtbl.hash v in
                  if h mod 3 = 0 then None else Some (value h)
            in
            R.Bind (ty, f, a)
          else
            let f v =
              let h = Hashtbl.hash v in
              if h mod p = 0 then raise (Boom (label, h)) else value h
            in
            R.Map (ty, f, a))
        ints
        (Gen.pair Gen.bool (anchored t s d)))

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
    Gen.bind
      (Gen.of_list (expressible s))
      (fun (_, Type.Any ty) ->
        let a = anchored ty s d in
        let other =
          Gen.bind (Gen.of_list (meeting ty s)) (fun t -> anchored t s d)
        in
        let op = Gen.of_list [ `Eq; `Ne; `Lt; `Le; `Gt; `Ge ] in
        let values = Gen.list ~size:(Gen.int_range 0 3) (value ty) in
        Gen.one_of
          [
            map2
              (fun op (a, b) -> R.Cmp (op, a, b))
              op
              (Gen.one_of [ pair a (operand ty s d); pair a other ]);
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
  Gen.bind
    (Gen.of_list (expressible s))
    (fun (_, Type.Any ty) ->
      let out e = R.Out (name, e) in
      Gen.frequency
        [
          (3, Gen.map out (anchored ty s 2));
          (1, Gen.map out (predicate s 1));
          ( 1,
            Gen.of_list
              [ out (R.Lit (Type.int64, 7)); out (R.Lit (Type.string, "é")) ] );
          ( 3,
            Gen.bind
              (Gen.of_list Type.[ int8; uint8; int32; uint32 ])
              (fun ty -> Gen.map out (lift ty s 1)) );
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

(* The run against the reference

   [Query.t] is abstract, so the reference runs the plan as written. Where the
   optimizer keeps it ([Query.equal]), the run is that plan, and agrees with the
   reference failures included. Elsewhere the optimizer may remove failures and
   move the conjuncts that fail, so the run agrees where the reference does not
   fail. *)

(* [attempt f] is [Ok (f ())], or [Error b] where [f] raises [Boom b]. *)
let attempt f = match f () with r -> Ok r | exception Boom b -> Error b
let error e = Format.asprintf "%a" Error.pp e

(* [same_failure expected actual] checks that the run's [actual] is the
   reference's failure [expected]: a step's error at its row, with its reason,
   or the same exception. *)
let same_failure expected actual =
  match (expected, actual) with
  | Ok (Error (row, why)), Ok (Error e) ->
      contains ~sub:(Printf.sprintf ": row %d: %s." row why) (error e)
  | Error b, Error b' -> equal (Windtrap.pair int int) b b'
  | Ok (Error (row, why)), _ -> failf "no failure at row %d: %s" row why
  | Error (label, _), _ -> failf "no Boom %d raised" label
  | Ok (Ok _), _ -> assert false

let holds t (n, R.Column (ty, vs)) =
  let (R.Column (ty', vs')) = R.decode (column t n) in
  match Kind.provably_equal (Type.kind ty') (Type.kind ty) with
  | Some Equal -> equal ~msg:n (array (option (G.witness ty))) vs vs'
  | None -> failf "%s holds %a, not %a" n Type.pp ty' Type.pp ty

let agrees p =
  let q = R.query p in
  let kept = Query.equal (Query.optimize q) q in
  cover "the optimizer keeps the plan" kept;
  match (attempt (fun () -> R.run p), attempt (fun () -> Query.run q)) with
  | Ok (Ok cs), Ok (Ok t) ->
      cover "rows in the result" (rows t > 0);
      at_most int ~than:1 (List.length (batches t));
      equal schema_w (Schema.v (R.schema p)) (schema t);
      List.iter (holds t) cs
  | Ok (Ok _), Ok (Error e) -> failf "the run fails: %s" (error e)
  | Ok (Ok _), Error (label, _) -> failf "the run raises Boom %d" label
  | expected, actual ->
      cover "a kept plan fails at a row" (kept && Result.is_ok expected);
      cover "a kept plan raises" (kept && Result.is_error expected);
      if kept then same_failure expected actual

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
  let run p () = Query.run (R.query p) in
  match (attempt (run p0), attempt (run p1)) with
  | Ok (Ok t0), Ok (Ok t1) ->
      List.iter
        (fun (n, _) ->
          equal ~msg:n (list string)
            (buffers (column t0 n))
            (buffers (column t1 n)))
        (R.schema p0)
  | Ok (Error e0), Ok (Error e1) -> equal string (error e0) (error e1)
  | Error b0, Error b1 -> equal (Windtrap.pair int int) b0 b1
  | _ -> fail "the two splits end differently"

(* Values *)

type expr = E : 'a R.expr -> expr

let values_cases =
  Gen.with_pp
    (fun ppf (p, _) -> pp_plan ppf p)
    (Gen.bind split_plans (fun p ->
         let s = R.schema p in
         Gen.map
           (fun e -> (p, e))
           (Gen.bind
              (Gen.of_list (expressible s))
              (fun (_, Type.Any ty) -> Gen.map (fun e -> E e) (anchored ty s 1)))))

(* [Query.values] reads its rows through the optimizer, so its failures are the
   reference's where the plan's rows are. *)
let values_agree (p, E e) =
  match attempt (fun () -> R.run p) with
  | Ok (Error _) | Error _ -> ()
  | Ok (Ok _) -> (
      let actual = attempt (fun () -> Query.values (R.expr e) (R.query p)) in
      match (attempt (fun () -> R.values e p), actual) with
      | Ok (Ok expected), Ok (Ok actual) ->
          equal (array (G.witness (R.type_of e))) expected actual
      | Ok (Ok _), Ok (Error e) -> failf "values failed: %s" (error e)
      | Ok (Ok _), Error (label, _) -> failf "values raised Boom %d" label
      | expected, actual ->
          cover "values fails" true;
          same_failure expected actual)

(* A streaming step that fails at a row emits the rows before it *)

type kind = Fails | Raises

let failing =
  Gen.with_pp
    (fun ppf (r, step, kind, t) ->
      Format.fprintf ppf "%s %s at row %d of %d batches" step
        (match kind with Fails -> "fails" | Raises -> "raises")
        r
        (List.length (batches t)))
    (Gen.bind (Gen.int_range 1 40) (fun n ->
         let x = Column.v Type.int64 (Array.init n Fun.id) in
         Gen.map
           (fun (r, (step, kind), t) -> (r, step, kind, t))
           (Gen.triple
              (Gen.int_range 0 (n - 1))
              (Gen.pair
                 (Gen.of_list [ "select"; "derive"; "filter" ])
                 (Gen.of_list [ Fails; Raises ]))
              (G.split (v [ ("x", x) ])))))

(* [failing_at (r, step, kind, t)] is [step] over [t], failing at row [r]. *)
let failing_at (r, step, kind, t) =
  let f x =
    if x <> r then x
    else match kind with Fails -> 100_000 | Raises -> raise (Boom (0, x))
  in
  let y = Expr.(store Type.int16 (const f $ Col.int "x")) in
  let q =
    match step with
    | "select" -> Query.select Expr.[ "y" := y ]
    | "derive" -> Query.derive Expr.[ "y" := y ]
    | _ -> Query.filter Expr.(y >= int 0)
  in
  q (Query.of_table t)

(* [ends_at r kind outcome] checks that [outcome] is the failure at row [r]. *)
let ends_at r kind outcome =
  match (kind, outcome) with
  | Fails, Ok (Error e) ->
      contains ~sub:(Printf.sprintf ": row %d: int16 does not hold" r) (error e)
  | Raises, Error (_, x) -> equal int r x
  | _ -> fail "the run ends otherwise"

(* The rows that [fold] sees before a failure at row [r] are the [r] rows before
   it, whatever the batches. *)
let emits_before ((r, _, kind, _) as c) =
  let seen = ref 0 in
  let fold () =
    Query.fold (failing_at c) ~init:() (fun () b -> seen := !seen + rows b)
  in
  ends_at r kind (attempt fold);
  equal ~msg:"rows folded before the failure" int r !seen

(* A failure past the rows a slice from the start keeps is never met, whatever
   the batches, and one at a row it keeps always is. Before [offset], the
   optimizer may move the slice below the failing step, which then never sees
   the row. The error's row counts over the step's input in the optimized plan,
   so a data error is matched by its reason. *)
let sliced =
  Gen.pair failing (Gen.pair (Gen.int_range 0 10) (Gen.int_range 0 15))

let slice_stops (((r, _, kind, t) as c), (offset, length)) =
  let outcome =
    attempt (fun () -> Query.run (Query.slice ~offset ~length (failing_at c)))
  in
  let stop = offset + length in
  cover "the failure is past the slice" (r >= stop);
  cover "the failure is in the slice" (offset <= r && r < stop);
  match outcome with
  | Ok (Ok ran) when r >= stop ->
      let kept = Int.max 0 (Int.min stop (rows t) - offset) in
      equal ~msg:"rows" int kept (rows ran)
  | _ when r >= stop -> fail "a failure past the slice is met"
  | Ok (Error e) when kind = Fails && r >= offset ->
      contains ~sub:"int16 does not hold 100000." (error e)
  | Error (_, x) when kind = Raises && r >= offset -> equal int r x
  | _ when r >= offset -> fail "the failure in the slice is not met"
  | _ -> ()

let laws =
  group "Laws"
    [
      prop "a streaming step that fails at row r emits the rows before r"
        failing emits_before;
      prop "a slice from the start never meets a failure past its rows" sliced
        slice_stops;
      prop ~count:500 "run gives the reference's rows, whatever the batches"
        split_plans agrees;
      prop "run's layouts are the same bytes whatever the batches" two_splits
        same_layouts;
      prop "values gives the reference's values, or fails where it does"
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
        ("a", Column.v Type.int8 [| -1; 5; 0 |]);
        ("b", Column.v Type.int16 [| -1; 300; 0 |]);
        ("c", Column.v Type.uint8 [| 255; 5; 0 |]);
        ( "k",
          Column.of_options
            (Type.categorical [| "x"; "y" |])
            [| Some "y"; Some "x"; None |] );
        ("s", Column.v Type.string [| "y"; "z"; "x" |]);
      ]
  in
  let i = Col.int and s = Col.string in
  let compares name e expected =
    test name (fun () -> rows_are Type.bool expected (bools e mixed))
  in
  group "Operands of two types"
    [
      compares "int8 = int16 compares at int16"
        Expr.(i "a" = i "b")
        [| t; f; t |];
      compares "int8 < int16 compares at int16"
        Expr.(i "a" < i "b")
        [| f; t; f |];
      compares "uint8 < int16 compares at int16"
        Expr.(i "c" < i "b")
        [| f; t; f |];
      compares "a categorical = a string compares as text"
        Expr.(s "k" = s "s")
        [| t; f; None |];
      compares "a categorical < a string compares as text"
        Expr.(s "k" < s "s")
        [| f; t; None |];
    ]

(* NaN equals NaN and orders after every other value, and -0. equals 0. *)
let float_order =
  let floats =
    v
      [
        ("x", Column.v Type.float64 [| Float.nan; 1.; -0.; Float.nan; 1. |]);
        ("y", Column.v Type.float32 [| Float.nan; Float.nan; 0.; 2.; 1. |]);
      ]
  in
  let x = Col.float "x" and y = Col.float "y" in
  let compares name e expected =
    test name (fun () -> rows_are Type.bool expected (bools e floats))
  in
  group "Float order"
    [
      compares "x = y" Expr.(x = y) [| t; f; t; f; t |];
      compares "x <> y" Expr.(x <> y) [| f; t; f; t; f |];
      compares "x < y" Expr.(x < y) [| f; t; f; f; f |];
      compares "x <= y" Expr.(x <= y) [| t; t; t; f; t |];
      compares "x > y" Expr.(x > y) [| f; f; f; t; f |];
      compares "x >= y" Expr.(x >= y) [| t; f; t; t; t |];
    ]

let celsius =
  Ext.v ~name:"celsius" ~ordered:true Type.float64
    ~dec:(fun c -> if Float.is_nan c then raise Exit else c)
    ~enc:Fun.id

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

let failures =
  let in_two_batches cs0 cs1 = of_batches [ v cs0; v cs1 ] in
  let x = Col.int "x" in
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

let ocaml =
  let x = Col.int "x" and y = Col.int "y" in
  let t =
    v
      [
        ("x", Column.of_options Type.int64 [| Some 1; None; Some 3; Some 4 |]);
        ("y", Column.of_options Type.int64 [| Some 10; Some 20; None; Some 40 |]);
      ]
  in
  let stored e = ints (Expr.store Type.int64 e) t in
  let values e = require_ok ~pp:Error.pp (Query.values e (Query.of_table t)) in
  group "OCaml values"
    [
      test "const is its value on every row" (fun () ->
          rows_are Type.int64 (Array.make 4 (Some 5)) (stored (Expr.const 5)));
      test "f $ a is called once per row where no argument is null" (fun () ->
          let calls = ref 0 in
          let add a b =
            incr calls;
            a + b
          in
          rows_are Type.int64
            [| Some 11; None; None; Some 44 |]
            (stored Expr.(const add $ x $ y));
          equal int 2 !calls);
      test "option is never null, and of_option is null at None" (fun () ->
          equal
            (array (option int))
            [| Some 1; None; Some 3; Some 4 |]
            (values Expr.(const Fun.id $ option x));
          rows_are Type.int64
            [| Some 2; None; Some 4; Some 5 |]
            (stored Expr.(of_option (const (Option.map succ) $ option x))));
      test "values reads OCaml values of rows with nulls" (fun () ->
          equal
            (array (Windtrap.pair int (option int)))
            [| (1, Some 10); (3, None); (4, Some 40) |]
            (require_ok ~pp:Error.pp
               (Query.values
                  Expr.(const (fun a b -> (a, b)) $ x $ option y)
                  (Query.filter Expr.(not (is_null x)) (Query.of_table t)))));
      test "a $ result meeting an extension is encoded with its declaration"
        (fun () ->
          let t =
            v
              [
                ("t", column (degrees [| Some 1.5; None |]) "t");
                ("x", Column.v Type.int64 [| 1; 2 |]);
              ]
          in
          let t' = Ext.col celsius "t" in
          let q =
            Query.derive
              Expr.[ "t" := coalesce [ t'; const float_of_int $ x ] ]
              (Query.of_table t)
          in
          equal (array float_exact) [| 1.5; 2. |]
            (require_ok ~pp:Error.pp (Query.values t' q)));
      test "a function is not called again after it raises" (fun () ->
          let calls = ref 0 in
          let f a =
            incr calls;
            if a = 3 then raise Exit else a
          in
          raises Exit (fun () -> stored Expr.(const f $ x));
          equal int 2 !calls);
      test "a value its store type does not hold fails at its row" (fun () ->
          let hundred a = a * 100 in
          expect
            (error
               (require_error
                  (Query.run
                     (Query.select
                        Expr.[ "z" := store Type.int8 (const hundred $ x) ]
                        (Query.of_table t)))))
          @@ __POS_OF__
               {| select ["z" := store int8 (<const> $ x)]: row 2: int8 does not hold 300. |});
    ]

(* At a row, outputs fail in order and an operand before its node; an earlier
   row fails first. [boom] raises at the row where [x] is 3, and [big] gives a
   value [int8] does not hold there and at the next row. *)
let failure_order =
  let x = Col.int "x" in
  let t =
    of_batches
      [
        v [ ("x", Column.v Type.int64 [| 1; 2 |]) ];
        v [ ("x", Column.v Type.int64 [| 3; 4 |]) ];
      ]
  in
  let boom a = if a = 3 then raise Exit else a in
  let big a = if a >= 3 then 1000 else a in
  let big_at_4 a = if a = 4 then 1000 else a in
  let boom_at_4 a = if a = 4 then raise Exit else a in
  let int8 f e = Expr.(store Type.int8 (const f $ e)) in
  let ends q =
    match Query.run q with
    | Ok _ -> "no failure"
    | Error e -> error e
    | exception Exit -> "Exit"
  in
  let select os = ends (Query.select os (Query.of_table t)) in
  group "Failure order"
    [
      test "a raising $ in the first output" (fun () ->
          equal string "Exit"
            (select Expr.[ "a" := int8 boom x; "b" := int8 big x ]));
      test "a raising $ in the second output" (fun () ->
          expect (select Expr.[ "b" := int8 big x; "a" := int8 boom x ])
          @@ __POS_OF__
               {| select ["b" := store int8 (<const> $ x); "a" := store int8 (<const> $ x)]: row 2: int8 does not hold 1000. |});
      test "a raising $ around a failing operand" (fun () ->
          expect (select Expr.[ "a" := int8 boom (int8 big x) ])
          @@ __POS_OF__
               {| select ["a" := store int8 (<const> $ store int8 (<const> $ x))]: row 2: int8 does not hold 1000. |});
      test "a raising $ inside a failing node" (fun () ->
          equal string "Exit" (select Expr.[ "a" := int8 big (int8 boom x) ]));
      test "a later step's failure at an earlier row comes first" (fun () ->
          let q =
            Query.of_table t
            |> Query.derive Expr.[ "b" := int8 big_at_4 x ]
            |> Query.filter Expr.(int8 boom x > int 0)
          in
          equal string "Exit" (ends q));
      test "an earlier step's failure at an earlier row comes first" (fun () ->
          let q =
            Query.of_table t
            |> Query.derive Expr.[ "b" := int8 big x ]
            |> Query.filter Expr.(int8 boom_at_4 x > int 0)
          in
          expect (ends q)
          @@ __POS_OF__
               {| derive ["b" := store int8 (<const> $ x)]: row 2: int8 does not hold 1000. |});
    ]

(* Lifts *)

let numeric =
  Type.
    [
      Any int8;
      Any int16;
      Any int32;
      Any int64;
      Any uint8;
      Any uint32;
      Any float16;
      Any float32;
      Any float64;
    ]

let unary_lifts =
  Expr.
    [
      ("neg", { f = Nx.neg });
      ("abs", { f = Nx.abs });
      ("square", { f = (fun x -> Nx.mul x x) });
      ( "abs by where",
        { f = (fun x -> Nx.where (Nx.less x (Nx.sub x x)) (Nx.neg x) x) } );
      ( "through float64",
        { f = (fun x -> Nx.cast (Nx.dtype x) (Nx.cast Nx.float64 x)) } );
    ]

let lift_cases =
  Gen.with_pp
    (fun ppf (G.Sample (ty, vs), (name, _)) ->
      Format.fprintf ppf "%s over %a, %d rows" name Type.pp ty (Array.length vs))
    (Gen.pair
       (Gen.bind (Gen.of_list numeric) (fun (Type.Any ty) ->
            Gen.map (fun vs -> G.Sample (ty, vs)) (G.options ty)))
       (Gen.of_list unary_lifts))

(* [nx f x] is [f] on [x]'s stored values with nx, null where [x] is, with zeros
   under its nulls. *)
let lifts_agree (G.Sample (ty, vs), (_, (fn : Expr.fn))) =
  let c = Column.of_options ty vs in
  let expected =
    match Column.layout c with
    | Fixed { validity; values = P x } ->
        let y = fn.f x in
        let y =
          match validity with
          | None -> y
          | Some b -> Nx.where (Nx_bits.to_bool b) y (Nx.zeros_like y)
        in
        Column.of_tensor ?validity y
    | _ -> assert false
  in
  let t = v [ ("x", c) ] in
  let actual = result Expr.(nx fn (Col.v (Type.kind ty) "x")) t in
  equal (list string) (buffers expected) (buffers actual)

let lifts =
  let x = Col.int "x" and y = Col.int "y" in
  let t =
    v
      [
        ("x", Column.of_options Type.int16 [| Some 1; None; Some (-3) |]);
        ("y", Column.of_options Type.int16 [| Some 7; Some 8; None |]);
      ]
  in
  let maximum = Expr.{ f2 = Nx.maximum } in
  group "Lifts"
    [
      prop "nx f x is f on x's values, null where x is" lift_cases lifts_agree;
      test "nx2 f x y is null where either is" (fun () ->
          rows_are Type.int16 [| Some 7; None; None |]
            (ints Expr.(nx2 maximum x y) t));
      test "nx2 f x literal broadcasts the literal" (fun () ->
          rows_are Type.int16 [| Some 1; None; Some 0 |]
            (ints Expr.(nx2 maximum x (int 0)) t));
      test "nx of literals alone is on every row" (fun () ->
          rows_are Type.int64
            [| Some 2; Some 2; Some 2 |]
            (ints Expr.(nx { f = Nx.abs } (int (-2))) t));
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
      cases
        ~name:(fun n ->
          Printf.sprintf "a widening not lowered is refused over %d rows" n)
        "Widening" [ 0; 3 ]
        (fun n ->
          let decimal precision scale =
            Column.v
              (Type.decimal ~precision ~scale)
              (Array.init n (fun i ->
                   Decimal.v ~unscaled:(Int64.of_int i) ~scale))
          in
          let t = v [ ("a", decimal 10 2); ("b", decimal 12 4) ] in
          let p = Expr.(Col.decimal "a" < Col.decimal "b") in
          expect
            (message (fun () -> Query.run (Query.filter p (Query.of_table t))))
          @@ __POS_OF__
               {| widening decimal[10, 2] to decimal[12, 4] is not implemented yet |});
    ]

let () =
  exit
    (run "Run"
       [
         laws;
         kleene;
         arithmetic;
         widening;
         float_order;
         failures;
         ocaml;
         failure_order;
         lifts;
         refusals;
       ])
