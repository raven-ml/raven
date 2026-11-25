(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module G = Talon_gen
module R = Reference

let schema_w = Testable.make ~pp:Schema.pp ~equal:Schema.equal
let run_ok q = require_ok ~pp:Error.pp (Query.run q)
let value ty = Option.get (G.value ty)
let s x = Some x

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
   where the reference computes exactly. OCaml functions hash their argument:
   one in about [p] raises [Boom], and the others give values that an [int8] or
   an unsigned type may not hold. *)

exception Boom of (int * int)

(* [hash t v] hashes the value [v] of type [t]: a tensor by its elements, since
   its representation holds the state of its storage. *)
let hash : type a. a Type.t -> a -> int =
 fun t v ->
  match Kind.provably_equal (Type.kind t) (Kind.tensor Nx.float32) with
  | Some Equal -> Hashtbl.hash (Nx.to_array v)
  | None -> Hashtbl.hash v

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
      Any date;
      Any (clock Us);
      Any (clock Ns);
      Any (duration Ns);
      Any (instant Ms);
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

(* [with_validity v l] is [l] with the validity [v]. *)
let with_validity v : Column.layout -> Column.layout = function
  | Fixed { values; _ } -> Fixed { validity = Some v; values }
  | Varsize { offsets; child; _ } ->
      Varsize { validity = Some v; offsets; child }
  | Children { length; fields; _ } ->
      Children { validity = Some v; length; fields }

(* [scattered ty vs] is the column of [vs] with a value of [ty] under each null
   and its validity a view from bit [k] of longer bits, whose bits around it are
   drawn too: the layout a format or a derived column may hand out. *)
let scattered ty vs =
  let n = Array.length vs in
  Gen.(
    let+ under = array ~size:(constant n) (value ty)
    and+ k = int_range 0 70
    and+ around = array ~size:(constant (n + 70)) bool in
    let filled = Array.mapi (fun i o -> Option.value o ~default:under.(i)) vs in
    let bits =
      Array.init
        (k + n + 9)
        (fun i ->
          if i >= k && i < k + n then Option.is_some vs.(i - k)
          else around.(i mod (n + 70)))
    in
    let v =
      Nx.slice
        [ Nx.R (k, k + n) ]
        (Nx.cast Nx.bit (Nx.create Nx.bool [| k + n + 9 |] bits))
    in
    Result.get_ok
      (Column.of_layout (Any ty)
         (with_validity v (Column.layout (Column.v ty filled)))))

let rec drawn : type a. a Type.t -> int -> Column.t Gen.t =
 fun ty n ->
  match ty with
  | Ext { storage; _ } ->
      let ext c = Result.get_ok (Column.of_layout (Any ty) (Column.layout c)) in
      Gen.map ext (drawn storage n)
  | _ ->
      Gen.bind
        (Gen.array ~size:(Gen.constant n) (Gen.option (value ty)))
        (fun vs ->
          Gen.frequency
            [
              (1, Gen.map (fun () -> Column.of_options ty vs) Gen.unit);
              (1, scattered ty vs);
            ])

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

(* [wider ty] is [ty] and the types of the palette that contain it. *)
let wider ty =
  ty
  :: List.filter
       (fun t -> Option.equal Type.equal (Type.common [ ty; t ]) (Some t))
       (kin ty (List.map (fun t -> ("", t)) palette))

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
      | Some Equal -> lift (ty : int Type.t) s (d - 1) :: converted ty s (d - 1)
      | None -> []
    in
    let slices : a R.expr Gen.t list =
      match is Kind.string with
      | Some Equal when Type.equal ty Type.string ->
          let range = Gen.pair (Gen.int_range (-4) 4) (Gen.int_range 0 4) in
          let slice ((o, l), a) = R.Substring (o, l, a) in
          List.map (Gen.map slice) (List.map (Gen.pair range) (texts s (d - 1)))
      | _ -> []
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
      :: (arith @ lifts @ slices))

(* [texts s d] draws text that reads a column of [s], if [s] has text. *)
and texts s d : string R.expr Gen.t list =
  match kin Type.string s with
  | [] -> []
  | ts -> [ Gen.bind (Gen.of_list ts) (fun t -> anchored t s d) ]

(* [converted ty s d] draws integers of [ty] that cast, measure or parse the
   columns of [s], if it has some to convert. *)
and converted : int Type.t -> _ -> int -> int R.expr Gen.t list =
 fun ty s d ->
  let cast a = R.Cast (ty, a) in
  let field = Gen.of_list [ `Year; `Month; `Day; `Yearday ] in
  let dates =
    match kin Type.date s with
    | [] -> []
    | ts ->
        let date = Gen.bind (Gen.of_list ts) (fun t -> anchored t s d) in
        [ map2 (fun f a -> cast (R.Field (f, a))) field date ]
  in
  let ints =
    match kin ty s with
    | [] -> []
    | ts ->
        [ Gen.map cast (Gen.bind (Gen.of_list ts) (fun t -> anchored t s d)) ]
  in
  ints
  @ List.concat_map
      (fun text ->
        [
          Gen.map (fun a -> cast (R.Length a)) text;
          Gen.map (fun a -> R.Parse (ty, a)) text;
        ])
      (texts s d)
  @ dates

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
                  let h = hash t v in
                  if h mod 3 = 0 then None else Some (value h)
            in
            R.Bind (ty, f, a)
          else
            let f v =
              let h = hash t v in
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
        (* Up to eight texts compare with the rows' bytes, more are coded with
           them. *)
        let size =
          Gen.frequency [ (3, Gen.int_range 0 3); (1, Gen.int_range 8 10) ]
        in
        let values = Gen.list ~size (value ty) in
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

(* Reductions and frames

   A reduction's operand may fail, and so may a frame's, but nothing that fails
   reads their values: where a value fails, the rows the evaluator computes past
   it are not the reference's, which only a failure would see. *)

type agg = Agg : ('a, Expr.agg) R.term -> agg

let reduction s =
  Gen.bind
    (Gen.of_list (expressible s))
    (fun (n, Type.Any ty) ->
      let a =
        Gen.frequency
          [
            (4, anchored ty s 1);
            (1, Gen.map (fun a -> R.Shift (1, a)) (anchored ty s 0));
          ]
      in
      let red r = Gen.map (fun a -> Agg (R.Reduce (r, a))) a in
      let quantiles =
        [
          red Median;
          Gen.bind (Gen.of_list [ 0.; 0.25; 1. ]) (fun p -> red (Quantile p));
        ]
      in
      let is k = Kind.provably_equal (Type.kind ty) k in
      let numbers =
        match (is Kind.int, is Kind.float, ty) with
        | Some Equal, _, (Int8 | Int16 | Int32 | Uint8 | Uint16 | Uint32) ->
            red Sum :: red Mean :: quantiles
        | Some Equal, _, _ | _, Some Equal, _ -> quantiles
        | None, None, _ -> []
      in
      Gen.one_of
        ([
           red Count;
           red Min;
           red Max;
           red First;
           red Last;
           red N_unique;
           red Arg_min;
           red Arg_max;
           Gen.constant (Agg (R.Reduce (Only, R.Col (ty, n))));
           Gen.constant (Agg R.Rows);
         ]
        @ numbers))

let keys s =
  Gen.bind
    (Gen.subsequence (List.map fst (expressible s)))
    (fun names ->
      all
        (List.map
           (fun name ->
             map2
               (fun desc nulls_first -> { R.name; desc; nulls_first })
               Gen.bool Gen.bool)
           names))

(* [framed s name] outputs, as [name], an expression that reads other rows than
   its own. *)
let framed s name =
  let over e =
    map2
      (fun by ks -> R.Out (name, R.Over (by, ks, e)))
      (Gen.subsequence (List.map fst s))
      (keys s)
  in
  Gen.bind
    (Gen.of_list (expressible s))
    (fun (_, Type.Any ty) ->
      let a = anchored ty s 1 in
      Gen.one_of
        [
          Gen.bind
            (Gen.pair (Gen.int_range (-2) 2) a)
            (fun (n, a) ->
              Gen.one_of
                [
                  Gen.constant (R.Out (name, R.Shift (n, a)));
                  over (R.Shift (n, a));
                ]);
          Gen.bind a (fun a -> over (R.Rank a));
          Gen.bind (reduction s) (fun (Agg t) -> over t);
        ])

let output s name =
  Gen.bind
    (Gen.of_list (expressible s))
    (fun (_, Type.Any ty) ->
      let out e = R.Out (name, e) in
      Gen.frequency
        [
          (2, framed s name);
          (3, Gen.map out (anchored ty s 2));
          (1, Gen.map out (predicate s 1));
          ( 1,
            Gen.of_list
              [ out (R.Lit (Type.int64, 7)); out (R.Lit (Type.string, "é")) ] );
          ( 3,
            Gen.bind
              (Gen.of_list Type.[ int8; uint8; int32; uint32 ])
              (fun ty ->
                Gen.map out (Gen.one_of (lift ty s 1 :: converted ty s 1))) );
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
    let ranked =
      Gen.map
        (fun (n, Type.Any ty) ->
          R.(Cmp (`Le, Rank (Col (ty, n)), Lit (Type.int64, 2))))
        (Gen.of_list (expressible s))
    in
    Gen.map
      (fun e -> R.Filter (e, p))
      (Gen.frequency [ (4, predicate s 2); (1, constant); (1, ranked) ])
  in
  let aggregate =
    let out i (Agg t) = R.Out (Printf.sprintf "a%d_%d" level i, t) in
    map2
      (fun by os -> R.Aggregate (by, List.mapi out os, p))
      (Gen.subsequence (List.map fst s))
      (Gen.bind (Gen.int_range 1 3) (fun k ->
           all (List.init k (fun _ -> reduction s))))
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
  (* The right is at most four rows of [p] or of a table of its columns, those a
     handle reads renamed, each stored at a type that contains its own, and a
     third of the time those rows twice, so that every key repeats. Keys the
     reference cannot write get no assertion. *)
  let join columns =
    let renamed (n, r, Type.Any ty) =
      Gen.map
        (fun t -> R.Out (r, R.Store (t, R.Col (ty, n))))
        (Gen.of_list (wider ty))
    in
    let right =
      Gen.bind
        (Gen.quad Gen.bool (Gen.int_range (-4) 3) (Gen.int_range 0 4)
           (Gen.int_range 0 2))
        (fun (own, offset, length, twice) ->
          map2
            (fun plan os ->
              let r = R.Select (os, R.Slice { offset; length; plan }) in
              if twice = 0 then R.Append (r, r) else r)
            (if own then Gen.constant p
             else Gen.map (fun t -> R.Table t) (table s))
            (all (List.map renamed columns)))
    in
    Gen.bind (Gen.subsequence columns) (fun ks ->
        let written =
          List.for_all
            (fun (_, _, Type.Any ty) -> Option.is_some (R.key_text ty))
            ks
        in
        let count =
          if not written then Gen.constant Join.Any
          else
            Gen.frequency
              [
                (3, Gen.constant Join.Any);
                (1, Gen.of_list Join.[ At_most_one; One; At_least_one ]);
              ]
        in
        let pairs = List.map (fun (n, r, _) -> (n, r)) in
        let keys = R.Keys (pairs ks) in
        (* The first columns as keys, and each other one compared. *)
        let range =
          Gen.bind
            (Gen.int_range 0 (List.length ks - 1))
            (fun k ->
              let eqs = List.filteri (fun i _ -> i < k) ks
              and rest = List.filteri (fun i _ -> i >= k) ks in
              Gen.map
                (fun ops ->
                  R.Range
                    ( pairs eqs,
                      List.map2
                        (fun op (n, r, _) -> R.Compare (op, n, r))
                        ops rest ))
                (Gen.list
                   ~size:(Gen.constant (List.length rest))
                   (Gen.of_list [ `Lt; `Le; `Gt; `Ge ])))
        in
        let on =
          if ks = [] then Gen.of_list R.[ Position; All ]
          else
            Gen.frequency
              [
                (3, Gen.constant keys);
                (2, range);
                (1, Gen.of_list R.[ Position; All ]);
              ]
        in
        Gen.map
          (fun ((on, kind), (each_left, each_right), right) ->
            R.Join { kind; each_left; each_right; on; left = p; right })
          (Gen.triple
             (Gen.pair on (Gen.of_list Join.[ Inner; Left; Full; Semi; Anti ]))
             (Gen.pair count count) right))
  in
  let columns =
    List.mapi
      (fun i (n, t) -> (n, Printf.sprintf "j%d_%d" level i, t))
      (expressible s)
  in
  let sort = Gen.map (fun ks -> R.Sort (ks, p)) (keys s) in
  let shared = Gen.constant (R.Append (p, p)) in
  Gen.one_of
    ([ select; derive; filter; slice; append; aggregate; sort; shared ]
    @ if columns = [] then [] else [ join columns ])

let rec plan level =
  if level = 0 then Gen.map (fun t -> R.Table t) (Gen.bind schemas table)
  else Gen.bind (plan (level - 1)) (step level)

(* Sources

   A source over a table's batches, cut into parts that state their rows or not.
   It gives one answer to every conjunct and applies each conjunct of its
   request, which [Exact] needs and [Inexact] allows. *)

let cmp op c =
  match op with
  | `Eq -> c = 0
  | `Ne -> c <> 0
  | `Lt -> c < 0
  | `Le -> c <= 0
  | `Gt -> c > 0
  | `Ge -> c >= 0

type 'r cell = { cell : 'a. 'a Type.t -> 'a option -> 'r }

(* [pred_holds t p] is [p] on each row of [t], as Kleene's logic computes it. *)
let rec pred_holds t (p : Source.Pred.t) =
  let each n { cell } =
    let (R.Column (ty, vs)) = R.decode (column t n) in
    Array.map (cell ty) vs
  in
  let same : type a. a Type.t -> a -> Source.Pred.value -> (int -> bool) -> bool
      =
   fun ty x (Value (ty', y)) test ->
    match Kind.provably_equal (Type.kind ty) (Type.kind ty') with
    | Some Equal -> test (Type.compare_value ty' x y)
    | None -> invalid_arg "pred_holds: a value of another kind"
  in
  match p with
  | Null n -> each n { cell = (fun _ v -> Some (Option.is_none v)) }
  | Valid n -> each n { cell = (fun _ v -> Some (Option.is_some v)) }
  | Cmp (n, op, v) ->
      each n { cell = (fun ty -> Option.map (fun x -> same ty x v (cmp op))) }
  | In (n, vs) ->
      let is_in ty x = List.exists (fun v -> same ty x v (cmp `Eq)) vs in
      each n
        {
          cell = (fun ty x -> Some (Option.fold ~none:false ~some:(is_in ty) x));
        }
  | Not p -> Array.map (Option.map not) (pred_holds t p)
  | And ps ->
      let f a b =
        match (a, b) with
        | Some false, _ | _, Some false -> Some false
        | None, _ | _, None -> None
        | _ -> Some true
      in
      List.fold_left (Array.map2 f)
        (Array.make (rows t) (Some true))
        (List.map (pred_holds t) ps)
  | Or ps ->
      let f a b =
        match (a, b) with
        | Some true, _ | _, Some true -> Some true
        | None, _ | _, None -> None
        | _ -> Some false
      in
      List.fold_left (Array.map2 f)
        (Array.make (rows t) (Some false))
        (List.map (pred_holds t) ps)

(* [answer r b] is the batch [b] read with the request [r]. *)
let answer (r : Source.request) b =
  let keep =
    List.fold_left
      (fun keep p ->
        Array.map2 (fun k v -> k && v = Some true) keep (pred_holds b p))
      (Array.make (rows b) true)
      r.filters
  in
  let idx = List.filter (fun i -> keep.(i)) (List.init (rows b) Fun.id) in
  let idx = Array.of_list (List.map Int64.of_int idx) in
  let b = take (Nx.create Nx.int64 [| Array.length idx |] idx) b in
  let keep =
    if r.columns = [] then [] else Expr.[ keep (Sel.names r.columns) ]
  in
  run_ok (Query.select keep (Query.of_table b))

(* [pred s] is a predicate on the columns of [s] that a handle reads, each value
   at a type that contains its column's, as a filter offers it. *)
let pred s =
  let open Gen in
  let columns = expressible s in
  let at (type a) (ty : a Type.t) =
    let* (t : a Type.t) = of_list (wider ty) in
    map (fun v -> Source.Pred.Value (t, v)) (value t)
  in
  let atom =
    let* n, Type.Any ty = of_list columns in
    let op = of_list [ `Eq; `Ne; `Lt; `Le; `Gt; `Ge ] in
    frequency
      [
        (3, map2 (fun op v -> Source.Pred.Cmp (n, op, v)) op (at ty));
        ( 2,
          let* (t : _ Type.t) = of_list (wider ty) in
          map
            (fun vs ->
              Source.Pred.In (n, List.map (fun v -> Source.Pred.Value (t, v)) vs))
            (list ~size:(int_range 0 3) (value t)) );
        (1, constant (Source.Pred.Null n));
        (1, constant (Source.Pred.Valid n));
      ]
  in
  let rec go d =
    if d = 0 then atom
    else
      let sub = go (d - 1) in
      frequency
        [
          (3, atom);
          (1, map (fun p -> Source.Pred.Not p) sub);
          ( 1,
            map (fun ps -> Source.Pred.And ps) (list ~size:(int_range 0 3) sub)
          );
          (1, map (fun ps -> Source.Pred.Or ps) (list ~size:(int_range 0 3) sub));
        ]
  in
  go 2

let rec pp_pred ppf (p : Source.Pred.t) =
  let value ppf (Source.Pred.Value (t, v)) =
    let text = Column.values Kind.string (Column.print (Column.v t [| v |])) in
    Format.fprintf ppf "%s : %a" text.(0) Type.pp t
  in
  let list ppf ps =
    Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ") ppf ps
  in
  let op = function
    | `Eq -> "="
    | `Ne -> "<>"
    | `Lt -> "<"
    | `Le -> "<="
    | `Gt -> ">"
    | `Ge -> ">="
  in
  match p with
  | Cmp (n, o, v) -> Format.fprintf ppf "%s %s %a" n (op o) value v
  | In (n, vs) -> Format.fprintf ppf "%s in [%a]" n (list value) vs
  | Null n -> Format.fprintf ppf "%s is null" n
  | Valid n -> Format.fprintf ppf "%s is valid" n
  | Not p -> Format.fprintf ppf "not (%a)" pp_pred p
  | And ps -> Format.fprintf ppf "and [%a]" (list pp_pred) ps
  | Or ps -> Format.fprintf ppf "or [%a]" (list pp_pred) ps

let pred_filter =
  let case = Gen.bind schemas (fun s -> Gen.pair (table s) (pred s)) in
  let x = Column.of_options Type.int64 [| Some 1; None; Some 3 |] in
  let passes_fails_null =
    (v [ ("x", x) ], Source.Pred.Cmp ("x", `Gt, Value (Type.int64, 1)))
  in
  prop ~examples:[ passes_fails_null ]
    "a predicate made a filter keeps the rows where it holds"
    (Gen.with_pp
       (fun ppf (t, p) -> Format.fprintf ppf "%a@.%a" pp_pred p Talon.pp t)
       case)
    (fun (t, p) ->
      let held = pred_holds t p in
      cover "a row passes" (Array.exists (( = ) (Some true)) held);
      cover "a row is null" (Array.exists Option.is_none held);
      cover "a row fails" (Array.exists (( = ) (Some false)) held);
      let idx =
        List.filter (fun i -> held.(i) = Some true) (List.init (rows t) Fun.id)
      in
      let idx = Array.of_list (List.map Int64.of_int idx) in
      let expected = take (Nx.create Nx.int64 [| Array.length idx |] idx) t in
      match Source.Pred.filter (schema t) p with
      | None -> fail "no filter for a predicate on handled columns"
      | Some f ->
          equal
            (Testable.make ~pp:Talon.pp ~equal:Talon.equal)
            expected
            (run_ok (Query.filter f (Query.of_table t))))

(* How a plan reads its tables. Two runs of one plan read alike, since the
   answers and the stated rows change the optimized plan, and so its
   failures. *)
type read = Tables | Sources of { answer : Source.answer; stated : bool }

let reads =
  Gen.frequency
    [
      (2, Gen.constant Tables);
      ( 1,
        map2
          (fun answer stated -> Sources { answer; stated })
          (Gen.of_list Source.[ Exact; Inexact; Unsupported ])
          Gen.bool );
    ]

let source ~answer:answer_ ~stated t =
  let bs = batches t in
  Gen.map
    (fun cuts ->
      let parts =
        List.fold_left2
          (fun parts b cut ->
            match parts with
            | p :: ps when not cut -> (b :: p) :: ps
            | ps -> [ b ] :: ps)
          [] bs cuts
        |> List.rev_map List.rev
      in
      let part r bs =
        let bs = List.map (answer r) bs in
        let rest = ref bs in
        let next () =
          match !rest with
          | [] -> Ok None
          | b :: bs ->
              rest := bs;
              Ok (Some b)
        in
        {
          Source.rows =
            (if stated then Some (List.fold_left (fun n b -> n + rows b) 0 bs)
             else None);
          open_ = (fun () -> Ok { Source.next; close = ignore });
        }
      in
      Source.v ~name:"source" ~schema:(schema t)
        ?rows:(if stated then Some (rows t) else None)
        ~pushdown:(fun _ -> answer_)
        (fun r -> Ok (List.map (part r) parts)))
    (Gen.list ~size:(Gen.constant (List.length bs)) Gen.bool)

(* [in_batches read p] is [p] with each table cut into batches, read as [read]
   says. *)
let rec in_batches read (p : R.plan) : R.plan Gen.t =
  let in_batches = in_batches read in
  match p with
  | Table t -> (
      Gen.bind (G.split t) @@ fun t ->
      match read with
      | Tables -> Gen.constant (R.Table t)
      | Sources { answer; stated } ->
          Gen.map (fun s -> R.Source (s, t)) (source ~answer ~stated t))
  | Source _ as p -> Gen.constant p
  | Sort (ks, p) -> Gen.map (fun p -> R.Sort (ks, p)) (in_batches p)
  | Append (p, r) when p == r ->
      Gen.map (fun p -> R.Append (p, p)) (in_batches p)
  | Select (os, p) -> Gen.map (fun p -> R.Select (os, p)) (in_batches p)
  | Derive (os, p) -> Gen.map (fun p -> R.Derive (os, p)) (in_batches p)
  | Filter (e, p) -> Gen.map (fun p -> R.Filter (e, p)) (in_batches p)
  | Slice s -> Gen.map (fun plan -> R.Slice { s with plan }) (in_batches s.plan)
  | Append (p, r) ->
      map2 (fun p r -> R.Append (p, r)) (in_batches p) (in_batches r)
  | Aggregate (by, os, p) ->
      Gen.map (fun p -> R.Aggregate (by, os, p)) (in_batches p)
  | Join j ->
      map2
        (fun left right -> R.Join { j with left; right })
        (in_batches j.left) (in_batches j.right)

let pp_plan ppf p = Query.pp ppf (R.query p)

(* [starved p] joins [p] with none of its rows, asserting that each left row has
   a match: it fails at [p]'s first row, if [p] has one. Random joins rarely
   fail an assertion, so the plans draw this one directly. Its kinds keep the
   right's columns, which the optimizer would otherwise prune. *)
let starved p =
  let renamed i (n, Type.Any ty) =
    R.Out (Printf.sprintf "u%d" i, R.Col (ty, n))
  in
  match List.mapi renamed (expressible (R.schema p)) with
  | [] -> Gen.constant p
  | os ->
      let right = R.Select (os, R.Slice { offset = 0; length = 0; plan = p }) in
      Gen.map
        (fun ((on, kind), each_left) ->
          R.Join { kind; each_left; each_right = Any; on; left = p; right })
        (Gen.pair
           (Gen.pair
              (Gen.of_list R.[ Position; All ])
              (Gen.of_list Join.[ Inner; Left; Full ]))
           (Gen.of_list Join.[ One; At_least_one ]))

(* [failing_select] parses texts that no integer writes, casts integers to
   [int8], which holds few of them, or applies a function that raises at a drawn
   row, under up to two random steps. Random plans fail in these ways too rarely
   to meet each in every run, so the plans draw them directly. *)
let failing_select =
  let rows = Gen.int_range 1 12 in
  let column n ty o gen =
    Gen.map (fun vs -> (n, Column.v ty vs, o)) (Gen.array ~size:rows gen)
  in
  let parse =
    column "s" Type.string
      (R.Out ("p", R.Parse (Type.int8, R.Col (Type.string, "s"))))
      (Gen.map (( ^ ) "x") (value Type.string))
  in
  let cast =
    column "i" Type.int64
      (R.Out ("c", R.Cast (Type.int8, R.Col (Type.int64, "i"))))
      (Gen.int_range (-1000) 1000)
  in
  let raising =
    Gen.bind
      (Gen.array ~size:rows (Gen.int_range (-50) 50))
      (fun xs ->
        map2
          (fun row label ->
            let f x = if x = xs.(row) then raise (Boom (label, x)) else 0 in
            ( "r",
              Column.v Type.int64 xs,
              R.Out ("m", R.Map (Type.int8, f, R.Col (Type.int64, "r"))) ))
          (Gen.int_range 0 (Array.length xs - 1))
          (Gen.int_range 0 99))
  in
  let base (n, c, o) = R.Select ([ R.Keep [ n ]; o ], R.Table (v [ (n, c) ])) in
  let rec above level p =
    if level > 2 then Gen.constant p
    else
      Gen.bind Gen.bool (fun more ->
          if more then Gen.bind (step level p) (above (level + 1))
          else Gen.constant p)
  in
  Gen.bind (Gen.one_of [ parse; cast; raising ]) (fun x -> above 1 (base x))

(* A slice from the start of rows filtered by a function that raises on a row
   past it, over a sort: the optimized plan may filter below the sort, which
   reads every row. The raise's value is the row. *)
let past_limit =
  let open Gen in
  let* n = int_range 3 30 in
  let* k = int_range 1 (n - 1) in
  let* r = int_range k (n - 1) in
  let+ label = int_range 0 99 in
  let x = Column.v Type.int64 (Array.init n Fun.id) in
  let f v = if v = r then raise (Boom (label, v)) else 0 in
  let raising = R.Map (Type.int8, f, R.Col (Type.int64, "x")) in
  let sorted = R.Sort ([], R.Table (v [ ("x", x) ])) in
  let filtered = R.Filter (R.Not (R.Is_null raising), sorted) in
  R.Slice { offset = 0; length = k; plan = filtered }

let plans =
  let drawn = Gen.bind (Gen.int_range 0 3) plan in
  Gen.frequency
    [
      (18, drawn);
      (2, Gen.bind drawn starved);
      (2, failing_select);
      (2, past_limit);
    ]

let split_plans =
  Gen.with_pp pp_plan
    (Gen.bind (Gen.pair plans reads) (fun (p, r) -> in_batches r p))

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

let column_holds t (n, R.Column (ty, vs)) =
  let (R.Column (ty', vs')) = R.decode (column t n) in
  match Kind.provably_equal (Type.kind ty') (Type.kind ty) with
  | Some Equal -> equal ~msg:n (array (option (G.witness ty))) vs vs'
  | None -> failf "%s holds %a, not %a" n Type.pp ty' Type.pp ty

(* [occurs w s] is [true] iff [w] occurs in [s]; [mentions w p] iff [p] prints
   it. *)
let occurs w s =
  let n = String.length w in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = w || at (i + 1))
  in
  at 0

let mentions w p = occurs w (Format.asprintf "%a" pp_plan p)

(* [past p actual] is [true] iff the run's [actual] is a failure that [p] meets
   on a row past a limit ({!R.beyond}): the optimized plan reads such rows where
   a step reads all of its input above a step the limit would stop, and a
   failure there ends the run ({!Query.fold}). An exception is told by its
   function and the value it raised on, a data failure by its reason. *)
let past p actual =
  let raised, reasons = R.beyond p in
  match actual with
  | Error b -> List.exists (function Boom b' -> b' = b | _ -> false) raised
  | Ok (Error e) ->
      List.exists (fun why -> occurs (": " ^ why ^ ".") (error e)) reasons
  | Ok (Ok _) -> false

(* [agreeing expected actual] is [true] iff the run's failure [actual] is the
   reference's [expected]. *)
let agreeing expected actual =
  match (expected, actual) with
  | Error b, Error b' -> b = b'
  | Ok (Error (row, why)), Ok (Error e) ->
      occurs (Printf.sprintf ": row %d: %s." row why) (error e)
  | _ -> false

(* [allowed os] is the outcomes talon allows, {!R.outcomes}, a function's [Boom]
   as {!attempt} gives it. *)
let allowed os =
  Seq.map
    (function Ok r -> Ok r | Error (Boom b) -> Error b | Error e -> raise e)
    os

(* [failure_of o] is the failure of the outcome [o], if any. *)
let failure_of = function
  | Ok (Ok _) -> None
  | Ok (Error f) -> Some (Ok f)
  | Error b -> Some (Error b)

(* [one_of expected allowed actual] checks that the run's [actual] is one of the
   outcomes [allowed], reported against the reference's [expected]. *)
let one_of expected allowed actual =
  if not (Seq.exists (fun o -> agreeing o actual) allowed) then
    same_failure expected actual

let success = function Ok (Ok v) -> Some v | _ -> None

(* A join's right rows by the left's keys: the rows and distinct keys of those
   that a left row matches, and of those that none does, by the reference's
   grouping. *)
type right = { matched : int; matched_keys : int; rest : int; rest_keys : int }

(* [right_keys p] is the [right] of each equality join of [p] whose sides
   run. *)
let rec right_keys : R.plan -> right list = function
  | Table _ | Source _ -> []
  | Select (_, p)
  | Derive (_, p)
  | Filter (_, p)
  | Slice { plan = p; _ }
  | Aggregate (_, _, p)
  | Sort (_, p) ->
      right_keys p
  | Append (p, p') -> right_keys p @ right_keys p'
  | Join { on; left; right; _ } ->
      let rows = function
        | (_, R.Column (_, vs)) :: _ -> Array.length vs
        | [] -> 0
      in
      let here =
        match on with
        | Keys ks -> (
            (* The right's rows that the left matches, [Semi], or not. *)
            let side kind =
              R.Join
                {
                  kind;
                  each_left = Any;
                  each_right = Any;
                  on = Keys (List.map (fun (l, r) -> (r, l)) ks);
                  left = right;
                  right = left;
                }
            in
            let keys p = R.Aggregate (List.map snd ks, [], p) in
            let count p = rows (Result.get_ok (R.run p)) in
            try
              [
                {
                  matched = count (side Semi);
                  matched_keys = count (keys (side Semi));
                  rest = count (side Anti);
                  rest_keys = count (keys (side Anti));
                };
              ]
            with _ -> [])
        | Range _ | Position | All -> []
      in
      here @ right_keys left @ right_keys right

let agrees p =
  let q = R.query p in
  let kept = Query.equal (Query.optimize q) q in
  let framed =
    List.exists (fun w -> mentions w p) [ "over"; "shift"; "rank" ]
  in
  cover "the optimizer keeps the plan" kept;
  cover "an aggregate" (mentions "aggregate" p);
  cover "a join" (mentions "join" p);
  cover "a range join"
    (List.exists (fun w -> mentions (w ^ " \"") p) [ "lt"; "le"; "gt"; "ge" ]);
  let joins = right_keys p in
  let some f = List.exists f joins in
  cover "a join to distinct right keys, two rows or more"
    (some (fun r ->
         r.matched + r.rest >= 2
         && r.matched_keys + r.rest_keys = r.matched + r.rest));
  cover "a join to a repeated right key"
    (some (fun r -> r.matched_keys < r.matched || r.rest_keys < r.rest));
  cover "an expression reads other rows" framed;
  cover "a sort" (mentions "sort" p);
  cover "a source" (mentions "source" p);
  cover "is_in" (mentions "is_in" p);
  let expected = attempt (fun () -> R.run p)
  and actual = attempt (fun () -> Query.run q) in
  let escaped =
    Result.fold ~ok:Result.is_error ~error:(fun _ -> true) actual
    && (not (agreeing expected actual))
    && past p actual
  in
  cover "the run fails past a limit" escaped;
  let checked = kept && not escaped in
  let why = match expected with Ok (Error (_, why)) -> why | _ -> "" in
  let fails = checked && why <> "" in
  let raises = checked && Result.is_error expected in
  cover "rows in the result"
    (match (expected, actual) with
    | Ok (Ok _), Ok (Ok t) -> (not escaped) && rows t > 0
    | _ -> false);
  cover "a kept plan fails at a row" fails;
  cover "a kept plan raises" raises;
  (* The outcomes talon allows differ where a join's read order decides the
     failure. Talon reads each join in the reference's order today, so no run
     can agree with another outcome alone; the cover shows the law meets the
     plans where the order matters. *)
  cover "another allowed outcome fails otherwise"
    ((fails || raises)
    && Seq.exists
         (fun o -> failure_of o <> failure_of expected)
         (allowed (R.outcomes p)));
  cover "a kept plan that aggregates fails"
    ((fails || raises) && mentions "aggregate" p);
  cover "a kept plan that reads other rows fails" ((fails || raises) && framed);
  cover "a kept plan fails a join's assertion"
    (fails && occurs " rows, not " why);
  cover "a kept plan fails at a cast"
    (fails && String.starts_with ~prefix:"cannot cast" why);
  cover "a kept plan fails at a text"
    (fails && String.starts_with ~prefix:"\"" why);
  match (expected, actual) with
  | _ when escaped -> ()
  | Ok (Ok cs), Ok (Ok t) ->
      at_most int ~than:1 (List.length (batches t));
      equal schema_w (Schema.v (R.schema p)) (schema t);
      List.iter (column_holds t) cs
  | Ok (Ok _), Ok (Error e) -> failf "the run fails: %s" (error e)
  | Ok (Ok _), Error (label, _) -> failf "the run raises Boom %d" label
  | expected, actual when kept && not (agreeing expected actual) -> (
      let allowed = allowed (R.outcomes p) in
      match actual with
      | Ok (Ok t) -> (
          match Seq.find_map success allowed with
          | Some cs -> List.iter (column_holds t) cs
          | None -> same_failure expected actual)
      | _ -> one_of expected allowed actual)
  | _ -> ()

(* Canonical layouts, byte for byte *)

(* Elements narrower than a byte are read from their storage, the bits past
   their last element included. A NaN's sign and payload are not a result's (nx
   leaves them to the kernel), so every NaN is written as one. *)
let hex (type a b) (x : (a, b, Nx.host) Nx.t) =
  let x : (a, b, Nx.host) Nx.t =
    match Nx.Dtype.kind (Nx.dtype x) with
    | Float -> Nx.where (Nx.isnan x) (Nx.full_like x Float.nan) x
    | _ -> x
  in
  let bits = Nx.Dtype.bits (Nx.dtype x) in
  let bytes =
    match Nx.dtype x with
    | Bool -> Nx.to_array (Nx.cast Nx.uint8 x)
    | _ when bits < 8 ->
        Nx.Repr.read ~by:"test" x @@ fun a ->
        Rig.Buffer.read (Nx_array.buffer a) Bigarray.int8_unsigned (fun raw ->
            Array.init (((Nx.numel x * bits) + 7) / 8) (fun i -> raw.{i}))
    | _ -> Nx.to_array (Nx.flatten (Nx.bitcast Nx.uint8 x))
  in
  String.concat "" (List.map (Printf.sprintf "%02x") (Array.to_list bytes))

let rec buffers c =
  (* A canonical validity's bytes are its bits from bit 0, the bits past them
     clear. *)
  let bits = function
    | None -> "no validity"
    | Some b ->
        String.concat ""
          (List.map
             (fun v -> if v then "1" else "0")
             (Array.to_list (Nx.to_array b)))
  in
  match Column.layout c with
  | Fixed { validity; values = P x } -> [ bits validity; hex x ]
  | Varsize { validity; offsets; child } ->
      bits validity :: hex offsets :: buffers child
  | Children { validity; length; fields } ->
      bits validity :: string_of_int length
      :: List.concat_map (fun (n, c) -> n :: buffers (held validity n c)) fields

(* [held validity n c] is the field [n]'s column [c] at the rows its record
   holds, canonical: under a null record its slots are unspecified, values and
   validity alike, as in Arrow. *)
and held validity n c =
  match validity with
  | None -> c
  | Some v ->
      let rows = Nx.positions (Nx.cast Nx.bool v) in
      let t = Talon.take rows (Talon.v [ (n, c) ]) in
      Talon.column (run_ok (Query.of_table t)) n

let two_splits =
  Gen.with_pp
    (fun ppf (p, _) -> pp_plan ppf p)
    (Gen.bind (Gen.pair plans reads) (fun (p, r) ->
         Gen.pair (in_batches r p) (in_batches r p)))

(* [same_outcome q0 q1] checks that [q0] and [q1] run to the same buffers, or
   end alike. *)
let same_outcome q0 q1 =
  let run q () = Query.run q in
  match (attempt (run q0), attempt (run q1)) with
  | Ok (Ok t0), Ok (Ok t1) ->
      List.iter
        (fun (n, _) ->
          equal ~msg:n (list string)
            (buffers (column t0 n))
            (buffers (column t1 n)))
        (Schema.columns (Query.schema q0))
  | Ok (Error e0), Ok (Error e1) -> equal string (error e0) (error e1)
  | Error b0, Error b1 -> equal (Windtrap.pair int int) b0 b1
  | _ -> fail "the two runs end differently"

let same_layouts (p0, p1) = same_outcome (R.query p0) (R.query p1)

(* A filter or a join may empty a batch of its input; [fold] hands [f] none
   without a row. The example's filter empties its first batch. *)
let i64 xs = Column.v Type.int64 xs

let emptied_batch =
  let x xs = v [ ("x", Column.v Type.int64 xs) ] in
  R.Filter
    ( R.Cmp (`Gt, R.Col (Type.int64, "x"), R.Lit (Type.int64, 5)),
      R.Table (of_batches [ x [| 1; 2 |]; x [| 7 |] ]) )

let batches_have_rows p =
  cover "a filter" (mentions "filter" p);
  let fold () = Query.fold (R.query p) ~init:[] (fun ns b -> rows b :: ns) in
  match attempt fold with
  | Ok (Ok ns) -> List.iter (greater int ~than:0) ns
  | _ -> ()

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
  let runs =
    match attempt (fun () -> R.run p) with Ok (Ok _) -> true | _ -> false
  in
  let expected = attempt (fun () -> R.values e p)
  and actual = attempt (fun () -> Query.values (R.expr e) (R.query p)) in
  let escaped = (not (agreeing expected actual)) && past p actual in
  let fails = match expected with Ok (Ok _) -> false | _ -> true in
  cover "values fails" (runs && (not escaped) && fails);
  match (expected, actual) with
  | _ when (not runs) || escaped -> ()
  | Ok (Ok expected), Ok (Ok actual) ->
      equal (array (G.witness (R.type_of e))) expected actual
  | Ok (Ok _), Ok (Error e) -> failf "values failed: %s" (error e)
  | Ok (Ok _), Error (label, _) -> failf "values raised Boom %d" label
  | expected, actual when not (agreeing expected actual) -> (
      let allowed = allowed (R.value_outcomes e p) in
      match actual with
      | Ok (Ok vs) -> (
          match Seq.find_map success allowed with
          | Some ws -> equal (array (G.witness (R.type_of e))) ws vs
          | None -> same_failure expected actual)
      | _ -> one_of expected allowed actual)
  | _ -> ()

(* A streaming step that fails at a row emits the rows before it *)

type step = Select | Derive | Filter
type kind = Fails | Raises

let failing =
  Gen.with_pp
    (fun ppf (r, step, kind, t) ->
      Format.fprintf ppf "%s %s at row %d of %d batches"
        (match step with
        | Select -> "select"
        | Derive -> "derive"
        | Filter -> "filter")
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
                 (Gen.of_list [ Select; Derive; Filter ])
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
    | Select -> Query.select Expr.[ "y" := y ]
    | Derive -> Query.derive Expr.[ "y" := y ]
    | Filter -> Query.filter Expr.(y >= int 0)
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

(* A text compared with itself is null on a null row, where one batch of two
   rows compares by words and two of one row compare by bytes: the bits under
   the nulls are one way either way. *)
let text_compare_layouts () =
  let row () = Column.of_options Type.binary [| None |] in
  let one = Talon.v [ ("c", Column.of_options Type.binary [| None; None |]) ]
  and two =
    Talon.of_batches [ Talon.v [ ("c", row ()) ]; Talon.v [ ("c", row ()) ] ]
  in
  let c = Col.binary "c" in
  let q t = Query.select Expr.[ "r" := not (c = c) ] (Query.of_table t) in
  same_outcome (q one) (q two)

(* Plans that meet the rarest covers of [agrees]: joins to distinct and to
   repeated right keys, a failing assertion, assertions on both sides whose
   order decides the failure and a range join, and kept plans that fail in an
   aggregate, at a cast, at a text, in a function and in an expression that
   reads other rows, and an if_ whose untaken branch fails. *)
let raises_at_2 x = if x = 2 then raise (Boom (0, x)) else 0

let rare_plans =
  let t cols =
    R.Table (v (List.map (fun (n, xs) -> (n, Column.v Type.int64 xs)) cols))
  in
  let join ?(each_left = Join.Any) ?(each_right = Join.Any) on l r =
    R.Join { kind = Inner; each_left; each_right; on; left = t l; right = t r }
  in
  let keys = R.Keys [ ("k", "rk") ] in
  let only = R.Reduce (Only, R.Col (Type.int64, "x")) in
  let i = R.Col (Type.int64, "i") and hundred = R.Lit (Type.int64, 100) in
  let zero = R.Lit (Type.int8, 0) in
  [
    join keys [ ("k", [| 1; 2 |]) ] [ ("rk", [| 1; 2; 3 |]) ];
    join keys [ ("k", [| 1 |]) ] [ ("rk", [| 1; 1; 2 |]) ];
    join ~each_left:One keys [ ("k", [| 1; 2 |]) ] [ ("rk", [| 1 |]) ];
    join ~each_left:One ~each_right:One keys
      [ ("k", [| 1; 1 |]) ]
      [ ("rk", [| 1; 1 |]) ];
    join
      (R.Range ([], [ R.Compare (`Lt, "a", "c") ]))
      [ ("a", [| 0 |]) ]
      [ ("c", [| 1; 2 |]) ];
    R.Aggregate
      ( [ "g" ],
        [ R.Out ("o", only) ],
        t [ ("g", [| 0; 0 |]); ("x", [| 1; 2 |]) ] );
    R.Select
      ( [ R.Out ("c", R.Cast (Type.int8, R.Col (Type.int64, "i"))) ],
        t [ ("i", [| 1; 1000 |]) ] );
    R.Select
      ( [ R.Out ("p", R.Parse (Type.int8, R.Col (Type.string, "s"))) ],
        R.Table (v [ ("s", Column.v Type.string [| "1"; "x" |]) ]) );
    R.Select
      ( [ R.Out ("m", R.Map (Type.int8, raises_at_2, R.Col (Type.int64, "i"))) ],
        t [ ("i", [| 1; 2 |]) ] );
    R.Select ([ R.Out ("o", R.Over ([], [], only)) ], t [ ("x", [| 1; 2 |]) ]);
    R.Select
      ( [
          R.Out
            ("y", R.If (R.Cmp (`Lt, i, hundred), R.Cast (Type.int8, i), zero));
        ],
        t [ ("i", [| 1; 1000 |]) ] );
  ]

let null_value =
  let x = Column.of_options Type.int64 [| Some 1; None |] in
  (R.Table (v [ ("x", x) ]), E (R.Col (Type.int64, "x")))

let run_laws =
  [
    prop ~count:1000 ~examples:rare_plans
      "gives the reference's rows, whatever the batches" split_plans agrees;
    prop "lays out the same bytes whatever the batches" two_splits same_layouts;
    test "lays out a text compared with itself one way whatever the batches"
      text_compare_layouts;
  ]

let fold_laws =
  [
    prop ~examples:[ emptied_batch ] "hands f batches of a row or more"
      split_plans batches_have_rows;
    prop "emits the rows before the row where a streaming step fails" failing
      emits_before;
    prop
      ~examples:[ ((1, Select, Fails, v [ ("x", i64 [| 0; 1; 2 |]) ]), (0, 3)) ]
      "never meets a failure past the rows a slice from the start keeps" sliced
      slice_stops;
  ]

let values_law =
  prop ~examples:[ null_value ]
    "gives the reference's values, or fails where it does" values_cases
    values_agree

(* Kleene logic *)

let t, f = (Some true, Some false)

(* Two columns that hold every pair of [true], [false] and null, with drawn
   values under the nulls: three for [a]'s and three for [b]'s. *)
let under_nulls =
  let column values valid =
    Column.of_tensor
      ~validity:(Nx.cast Nx.bit (Nx.create Nx.bool [| 9 |] valid))
      (Nx.create Nx.bool [| 9 |] values)
  in
  let table (xs, ys) =
    let flag o d = Option.value o ~default:d in
    let a = [| t; t; t; f; f; f; None; None; None |]
    and b = [| t; f; None; t; f; None; t; f; None |] in
    let values col under =
      Array.mapi (fun i o -> flag o (List.nth under (i mod 3))) col
    in
    let valid col = Array.map Option.is_some col in
    v
      [
        ("a", column (values a xs) (valid a));
        ("b", column (values b ys) (valid b));
      ]
  in
  let bits = Gen.list ~size:(Gen.int_range 3 3) Gen.bool in
  let pp ppf (xs, ys) =
    let row ppf v = Format.pp_print_char ppf (if v then '1' else '0') in
    let rows = Format.pp_print_list ~pp_sep:(fun _ () -> ()) row in
    Format.fprintf ppf "under a's nulls %a, under b's %a" rows xs rows ys
  in
  Gen.map table (Gen.with_pp pp (Gen.pair bits bits))

(* [result e t] is the column of [e] over [t]'s rows. *)
let result e t =
  column (run_ok (Query.select Expr.[ "r" := e ] (Query.of_table t))) "r"

let rows_are k expected c = equal (array (option (G.witness k))) expected c
let bools e t = Column.options Kind.bool (result e t)
let ints e t = Column.options Kind.int (result e t)

let a = Col.bool "a"
and b = Col.bool "b"

(* A bool column of [n] rows drawn over bit views: values and validity start at
   their own bit offsets in larger buffers, and the bits outside the rows are
   drawn too. *)
type bits = { values : bool array; valid : bool array option; at : int * int }

let pp_bits ppf c =
  let row v = if v then '1' else '0' in
  let str a = String.init (Array.length a) (fun i -> row a.(i)) in
  Format.fprintf ppf "values %s at %d, %s" (str c.values) (fst c.at)
    (match c.valid with
    | None -> "no validity"
    | Some v -> Printf.sprintf "validity %s at %d" (str v) (snd c.at))

let bits_column n =
  let open Gen in
  let* values = array ~size:(constant n) bool in
  let* valid = option (array ~size:(constant n) bool) in
  let+ at = pair (int_range 0 63) (int_range 0 63) in
  { values; valid; at }

(* [view k xs] is [xs] as bits at offset [k] of a buffer whose other bits are
   set. *)
let view k xs =
  let n = Array.length xs in
  let all =
    Array.init (k + n + 9) (fun i -> i < k || i >= k + n || xs.(i - k))
  in
  Nx.slice [ Nx.R (k, k + n) ] (Nx.create Nx.bit [| k + n + 9 |] all)

let of_bits c =
  let validity = Option.map (view (snd c.at)) c.valid in
  Column.of_tensor ?validity (view (fst c.at) c.values)

(* Lengths at the edges of bytes and words, and others. *)
let lengths =
  Gen.frequency
    [
      (2, Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 7; 8; 9; 63; 64; 65; 129 ]);
      (1, Gen.int_range 0 200);
    ]

let bits_pair =
  Gen.with_pp
    (fun ppf (x, y) -> Format.fprintf ppf "a: %a@ b: %a" pp_bits x pp_bits y)
    (Gen.bind lengths (fun n -> Gen.pair (bits_column n) (bits_column n)))

(* Kleene's rules over one byte per value: a row of [a && b] is [false] where an
   operand is a valid [false], else null where an operand is null. *)
let kleene_rows op x y =
  let at c i =
    match c.valid with Some v when not v.(i) -> None | _ -> Some c.values.(i)
  in
  Array.init (Array.length x.values) (fun i ->
      match (op, at x i, at y i) with
      | `And, Some false, _ | `And, _, Some false -> Some false
      | `Or, Some true, _ | `Or, _, Some true -> Some true
      | `And, Some p, Some q -> Some (p && q)
      | `Or, Some p, Some q -> Some (p || q)
      | _ -> None)

let kleene_bits (x, y) =
  cover "an operand inside a byte" (fst x.at mod 8 <> 0);
  cover "a validity inside a byte"
    (Option.is_some x.valid && snd x.at mod 8 <> 0);
  cover "past one word" (Array.length x.values > 64);
  cover "both validities" (Option.is_some x.valid && Option.is_some y.valid);
  let t = v [ ("a", of_bits x); ("b", of_bits y) ] in
  rows_are Type.bool (kleene_rows `And x y) (bools Expr.(a && b) t);
  rows_are Type.bool (kleene_rows `Or x y) (bools Expr.(a || b) t);
  rows_are Type.bool
    (Array.map (Option.map not) (kleene_rows `And x x))
    (bools (Expr.not a) t)

let kleene =
  [
    prop "&&, || and not are Kleene's tables, whatever lies under the nulls"
      under_nulls (fun tbl ->
        rows_are Type.bool
          [| t; f; None; f; f; f; None; f; None |]
          (bools Expr.(a && b) tbl);
        rows_are Type.bool
          [| t; t; t; t; f; None; t; None; None |]
          (bools Expr.(a || b) tbl);
        rows_are Type.bool
          [| f; f; f; t; t; t; None; None; None |]
          (bools (Expr.not a) tbl));
    prop
      ~examples:
        [
          ( { values = Array.make 65 true; valid = None; at = (1, 0) },
            { values = Array.make 65 false; valid = None; at = (0, 0) } );
        ]
      "&&, || and not are Kleene's over bits at any offset and length" bits_pair
      kleene_bits;
  ]

(* Every operand is computed at every row: [cast int8 x] fails at row 1, where
   no row of the result takes its value. The operands are columns, so the
   optimizer folds none of them away. *)
let computed_operands =
  let c = Col.bool "c" and x = Col.int "x" and y = Col.int "y" in
  let t =
    v
      [
        ("c", Column.v Type.bool [| true; false |]);
        ("x", Column.v Type.int64 [| 1; 1000 |]);
        ("y", Column.v Type.int8 [| 1; 2 |]);
      ]
  in
  let fails_at_row_1 e =
    let q = Query.select Expr.[ "r" := e ] (Query.of_table t) in
    contains ~sub:": row 1: cannot cast 1000 to int8."
      (error (require_error (Query.run q)))
  in
  let narrow = Expr.cast Type.int8 x in
  [
    test "&& fails where its right fails and its left is false" (fun () ->
        fails_at_row_1 Expr.(c && is_null narrow));
    test "|| fails where its right fails and its left is true" (fun () ->
        fails_at_row_1 Expr.((not c) || is_null narrow));
    test "if_ fails where its untaken branch fails" (fun () ->
        fails_at_row_1 Expr.(if_ c narrow (int 0)));
    test "coalesce fails where an operand after a value fails" (fun () ->
        fails_at_row_1 Expr.(coalesce [ y; narrow ]));
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
  let one (a, b) = v [ ("x", int64s [| a |]); ("y", int64s [| b |]) ] in
  let tensor e t = Nx.to_array (Column.to_tensor Nx.int64 (result e t)) in
  [
    test "x / y truncates toward zero and is null where y is zero" (fun () ->
        rows_are Type.int32
          [| Some 3; None; Some (-3) |]
          (ints Expr.(x / y) small));
    test "x mod y has x's sign and is null where y is zero" (fun () ->
        rows_are Type.int32 [| Some 1; None; Some 1 |]
          (ints Expr.(x mod y) small));
    (* Two's complement modulo 2^64, as nx's int64 computes. *)
    cases "int64 arithmetic wraps as nx's integers do"
      ~id:(fun (name, _, _, _) -> name)
      Int64.
        [
          ("min_int / -1 is min_int", Expr.(x / y), (min_int, -1L), min_int);
          ("min_int mod -1 is 0", Expr.(x mod y), (min_int, -1L), 0L);
          ("max_int + 1 is min_int", Expr.(x + y), (max_int, 1L), min_int);
          ("min_int - 1 is max_int", Expr.(x - y), (min_int, 1L), max_int);
          ("max_int * 2 is -2", Expr.(x * y), (max_int, 2L), -2L);
        ]
      (fun (_, e, operands, expected) ->
        equal (array int64) [| expected |] (tensor e (one operands)));
  ]

(* [a ** b] as Nx.pow computes it: exact at the special values of C99's Annex F
   (F.10.4.4), a zero or an infinity of the exact power's sign where it rounds
   to one, and within 4 ulps at float32 and 2 at float64 elsewhere. *)
let powers =
  let a = Col.float "a" and b = Col.float "b" in
  let power ty x y =
    let t = v [ ("a", Column.v ty [| x |]); ("b", Column.v ty [| y |]) ] in
    (Column.options Kind.float (result Expr.(a ** b) t)).(0)
  in
  let inf = Float.infinity and nan = Float.nan in
  (* The distance between two finite floats of one sign, in floats of [ty]. *)
  let ulps ty x y =
    let bits =
      if Type.equal ty Type.float32 then fun x ->
        Int64.of_int32 (Int32.bits_of_float x)
      else Int64.bits_of_float
    in
    Int64.(to_int (abs (sub (bits x) (bits y))))
  in
  (* The nearest float32 to a float64 that is not near a float32 tie. *)
  let single x = Int32.float_of_bits (Int32.bits_of_float x) in
  [
    cases "a ** b is Annex F's value at its special values"
      ~id:(fun (name, _, _, _, _) -> name)
      Type.
        [
          ("nan ** 0. is 1.", float64, nan, 0., 1.);
          ("1. ** nan is 1.", float64, 1., nan, 1.);
          ("-1. ** inf is 1.", float64, -1., inf, 1.);
          ("0. ** -1. is inf", float64, 0., -1., inf);
          ("-0. ** -1. is -inf", float64, -0., -1., -.inf);
          ("-0. ** -2. is inf", float64, -0., -2., inf);
          ("-0. ** 3. is -0.", float64, -0., 3., -0.);
          ("-0. ** 2. is 0.", float64, -0., 2., 0.);
          ("0.5 ** inf is 0.", float64, 0.5, inf, 0.);
          ("0.5 ** -inf is inf", float64, 0.5, -.inf, inf);
          ("-inf ** -1. is -0.", float64, -.inf, -1., -0.);
          ("-inf ** 3. is -inf", float64, -.inf, 3., -.inf);
          ("-2. ** 0.5 is nan", float64, -2., 0.5, nan);
          ("-2. ** -1101. rounds to -0.", float64, -2., -1101., -0.);
          ("nan ** 0. is 1. at float32", float32, nan, 0., 1.);
          ("2. ** 200. rounds to inf at float32", float32, 2., 200., inf);
        ]
      (fun (_, ty, x, y, expected) ->
        equal (option float_exact) (Some expected) (power ty x y));
    cases "a ** b is within 4 ulps at float32 and 2 at float64"
      ~id:(fun (name, _, _, _, _, _) -> name)
      Type.
        [
          ("2. ** 0.5 is sqrt 2.", float64, 2., 0.5, Float.sqrt 2., 2);
          ("10. ** -1. is 0.1", float64, 10., -1., 0.1, 2);
          ( "2. ** 0.5 is sqrt 2. at float32",
            float32,
            2.,
            0.5,
            single (Float.sqrt 2.),
            4 );
          ("10. ** -1. is 0.1 at float32", float32, 10., -1., single 0.1, 4);
        ]
      (fun (_, ty, x, y, expected, bound) ->
        let actual = Option.get (power ty x y) in
        at_most int ~than:bound (ulps ty expected actual));
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
  cases "operands compare at their common type"
    ~id:(fun (name, _, _) -> name)
    [
      ("int8 = int16 at int16", Expr.(i "a" = i "b"), [| t; f; t |]);
      ("int8 < int16 at int16", Expr.(i "a" < i "b"), [| f; t; f |]);
      ("uint8 < int16 at int16", Expr.(i "c" < i "b"), [| f; t; f |]);
      ( "a categorical = a string as text",
        Expr.(s "k" = s "s"),
        [| t; f; None |] );
      ( "a categorical < a string as text",
        Expr.(s "k" < s "s"),
        [| f; t; None |] );
    ]
    (fun (_, e, expected) -> rows_are Type.bool expected (bools e mixed))

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
  cases "floats compare in the total order"
    ~id:(fun (name, _, _) -> name)
    [
      ("x = y", Expr.(x = y), [| t; f; t; f; t |]);
      ("x <> y", Expr.(x <> y), [| f; t; f; t; f |]);
      ("x < y", Expr.(x < y), [| f; t; f; f; f |]);
      ("x <= y", Expr.(x <= y), [| t; t; t; f; t |]);
      ("x > y", Expr.(x > y), [| f; f; f; t; f |]);
      ("x >= y", Expr.(x >= y), [| t; f; t; t; t |]);
    ]
    (fun (_, e, expected) -> rows_are Type.bool expected (bools e floats))

let celsius =
  Ext.v ~name:"celsius" ~ordered:true Type.float64
    ~dec:(fun c -> if Float.is_nan c then raise Exit else c)
    ~enc:Fun.id

let degrees cs =
  let valid =
    Nx.create Nx.bool [| Array.length cs |] (Array.map Option.is_some cs)
  in
  let c =
    Column.of_tensor ~validity:(Nx.cast Nx.bit valid)
      (Nx.create Nx.float64
         [| Array.length cs |]
         (Array.map (Option.value ~default:0.) cs))
  in
  let ty = Type.Any (Type.ext ~name:"celsius" Type.float64) in
  v [ ("t", Result.get_ok (Column.of_layout ty (Column.layout c))) ]

let values_failures =
  let in_two_batches cs0 cs1 = of_batches [ v cs0; v cs1 ] in
  let x = Col.int "x" in
  let error r = Format.asprintf "%a" Error.pp (require_error r) in
  [
    test "fails at a value outside int, its row counted over the batches"
      (fun () ->
        let big =
          in_two_batches
            [ ("x", int64s [| 1L; 2L |]) ]
            [ ("x", int64s [| 3L; 0x4000_0000_0000_0000L |]) ]
        in
        expect (error (Query.values x (Query.of_table big)))
        @@ __POS_OF__ {| values x: row 3: 4611686018427387904 is outside int. |});
    test "fails at a null" (fun () ->
        let nulls =
          v [ ("x", Column.of_options Type.int8 [| Some 1; None |]) ]
        in
        expect (error (Query.values x (Query.of_table nulls)))
        @@ __POS_OF__
             {| values x: row 1: the value is null; read it through Expr.option. |});
    test "decodes extension values with their declaration" (fun () ->
        equal (array float_exact) [| 1.5; -2. |]
          (require_ok ~pp:Error.pp
             (Query.values (Ext.col celsius "t")
                (Query.of_table (degrees [| Some 1.5; Some (-2.) |])))));
    test "fails at a null before a row whose declaration raises" (fun () ->
        let t = degrees [| None; Some Float.nan |] in
        expect (error (Query.values (Ext.col celsius "t") (Query.of_table t)))
        @@ __POS_OF__
             {| values t: row 0: the value is null; read it through Expr.option. |});
    test "raises a declaration's exception before a later null" (fun () ->
        let t = degrees [| Some Float.nan; None |] in
        raises Exit (fun () ->
            Query.values (Ext.col celsius "t") (Query.of_table t)));
  ]

(* Two int64 columns with a null in different rows. *)
let ocaml_rows =
  v
    [
      ("x", Column.of_options Type.int64 [| Some 1; None; Some 3; Some 4 |]);
      ("y", Column.of_options Type.int64 [| Some 10; Some 20; None; Some 40 |]);
    ]

let ocaml =
  let x = Col.int "x" and y = Col.int "y" and t = ocaml_rows in
  let stored e = ints (Expr.store Type.int64 e) t in
  let values e = require_ok ~pp:Error.pp (Query.values e (Query.of_table t)) in
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

let values_with_nulls =
  let x = Col.int "x" and y = Col.int "y" and t = ocaml_rows in
  test "reads OCaml values of rows with nulls" (fun () ->
      equal
        (array (Windtrap.pair int (option int)))
        [| (1, Some 10); (3, None); (4, Some 40) |]
        (require_ok ~pp:Error.pp
           (Query.values
              Expr.(const (fun a b -> (a, b)) $ x $ option y)
              (Query.filter Expr.(not (is_null x)) (Query.of_table t)))))

(* An earlier row fails first; at a row, outputs fail in order and an operand
   before its node. [boom] raises at the row where [x] is 3, and [big] gives a
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
  [
    test "an earlier output's exception wins at a row" (fun () ->
        equal string "Exit"
          (select Expr.[ "a" := int8 boom x; "b" := int8 big x ]));
    test "an earlier output's failure wins at a row" (fun () ->
        expect (select Expr.[ "b" := int8 big x; "a" := int8 boom x ])
        @@ __POS_OF__
             {| select ["b" := store int8 (<const> $ x); "a" := store int8 (<const> $ x)]: row 2: int8 does not hold 1000. |});
    test "an operand's failure comes before its node's exception" (fun () ->
        expect (select Expr.[ "a" := int8 boom (int8 big x) ])
        @@ __POS_OF__
             {| select ["a" := store int8 (<const> $ store int8 (<const> $ x))]: row 2: int8 does not hold 1000. |});
    test "the left operand's exception comes before the right's failure"
      (fun () ->
        equal string "Exit" (select Expr.[ "a" := int8 boom x + int8 big x ]));
    test "the left operand's failure comes before the right's exception"
      (fun () ->
        expect (select Expr.[ "a" := int8 big x + int8 boom x ])
        @@ __POS_OF__
             {| select ["a" := store int8 (<const> $ x) + store int8 (<const> $ x)]: row 2: int8 does not hold 1000. |});
    test "an operand's exception comes before its node's failure" (fun () ->
        equal string "Exit" (select Expr.[ "a" := int8 big (int8 boom x) ]));
    test "a later step's exception at an earlier row wins" (fun () ->
        let q =
          Query.of_table t
          |> Query.derive Expr.[ "b" := int8 big_at_4 x ]
          |> Query.filter Expr.(int8 boom x > int 0)
        in
        equal string "Exit" (ends q));
    test "an earlier step's failure at an earlier row wins" (fun () ->
        let q =
          Query.of_table t
          |> Query.derive Expr.[ "b" := int8 big x ]
          |> Query.filter Expr.(int8 boom_at_4 x > int 0)
        in
        expect (ends q)
        @@ __POS_OF__
             {| derive ["b" := store int8 (<const> $ x)]: row 2: int8 does not hold 1000. |});
  ]

let slice_reads_none =
  test "from the start, keeping no row, reads none" (fun () ->
      let x = Col.int "x" in
      let t = v [ ("x", Column.v Type.int64 [| 1; 2; 3; 4 |]) ] in
      let fails_at_0 = Expr.(store Type.int8 (const (fun _ -> 1000) $ x)) in
      let q = Query.derive Expr.[ "b" := fails_at_0 ] (Query.of_table t) in
      let rows q = rows (run_ok q) in
      equal int 0 (rows (Query.slice ~offset:1 ~length:0 q));
      equal int 0 (rows (Query.slice ~offset:0 ~length:0 (Query.sort [] q))))

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
      ("fused multiply-add", { f = (fun x -> Nx.fma x x x) });
      ("plus a created one", { f = (fun x -> Nx.add x (Nx.ones_like x)) });
    ]

(* Lifts that take booleans, which nx computes through bytes. *)
let boolean_lifts =
  Expr.
    [
      ("maximum with itself", { f = (fun x -> Nx.maximum x x) });
      ("where", { f = (fun x -> Nx.where (Nx.equal x x) x x) });
      ( "through float64",
        { f = (fun x -> Nx.cast (Nx.dtype x) (Nx.cast Nx.float64 x)) } );
    ]

let lift_cases_over types lifts =
  Gen.with_pp
    (fun ppf (G.Sample (ty, vs), (name, _)) ->
      Format.fprintf ppf "%s over %a, %d rows" name Type.pp ty (Array.length vs))
    (Gen.pair
       (Gen.bind (Gen.of_list types) (fun (Type.Any ty) ->
            Gen.map (fun vs -> G.Sample (ty, vs)) (G.options ty)))
       (Gen.of_list lifts))

let lift_cases = lift_cases_over numeric unary_lifts

(* [nx f x] is [f] on [x]'s stored values with nx, null where [x] is: under its
   nulls, [f] of [x]'s. *)
let lifts_agree (G.Sample (ty, vs), (_, (fn : Expr.fn))) =
  let c = Column.of_options ty vs in
  let expected =
    match Column.layout c with
    | Fixed { validity; values = P x } -> Column.of_tensor ?validity (fn.f x)
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
  [
    prop "nx f x is f on x's values, null where x is" lift_cases lifts_agree;
    prop "nx f x over bool is f on x's bits, null where x is"
      (lift_cases_over [ Any Type.bool ] boolean_lifts)
      lifts_agree;
    test "nx2 f x y is null where either is" (fun () ->
        rows_are Type.int16 [| Some 7; None; None |]
          (ints Expr.(nx2 maximum x y) t));
    test "nx2 f x literal broadcasts the literal" (fun () ->
        rows_are Type.int16 [| Some 1; None; Some 0 |]
          (ints Expr.(nx2 maximum x (int 0)) t));
    test "nx f x is f's values for functions nx composes, acosh and hypot"
      (fun () ->
        let x = Col.float "f" in
        let t =
          v
            [
              ("f", Column.of_options Type.float64 [| Some 1.; None; Some 10. |]);
            ]
        in
        let at f =
          let e =
            Nx.to_array (f (Nx.create Nx.float64 [| 2 |] [| 1.; 10. |]))
          in
          [| Some e.(0); None; Some e.(1) |]
        in
        let lifted e = Column.options Kind.float (result e t) in
        rows_are Type.float64 (at Nx.acosh)
          (lifted Expr.(nx { f = Nx.acosh } x));
        rows_are Type.float64
          (at (fun x -> Nx.hypot x x))
          (lifted Expr.(nx2 { f2 = Nx.hypot } x x)));
    test "nx of literals alone is on every row" (fun () ->
        rows_are Type.int64
          [| Some 2; Some 2; Some 2 |]
          (ints Expr.(nx { f = Nx.abs } (int (-2))) t));
  ]

(* Reductions *)

let aggregate by os t = Query.run (Query.aggregate ~by os (Query.of_table t))
let floats t n = Column.options Kind.float (column t n)
let counts t n = Column.options Kind.int (column t n)

let durations ticks =
  let values = Nx.create Nx.int64 [| Array.length ticks |] ticks in
  Column.of_layout
    (Any (Type.duration Ns))
    (Fixed { validity = None; values = P values })
  |> Result.get_ok

let reductions =
  let x = Col.float "x" and k = Col.int "k" in
  let g = Column.v Type.int64 [| 0; 1; 1; 1; 1; 2 |] in
  [
    test "over no value count, sum and n_unique are 0, the others null"
      (fun () ->
        let none = v [ ("x", Column.v Type.float64 [||]) ] in
        let t =
          require_ok ~pp:Error.pp
            (aggregate []
               Expr.
                 [
                   "n" := count x;
                   "u" := n_unique x;
                   "s" := sum x;
                   "m" := min x;
                   "a" := mean x;
                 ]
               none)
        in
        equal
          (list (option int))
          [ Some 0; Some 0 ]
          (List.concat_map (fun n -> Array.to_list (counts t n)) [ "n"; "u" ]);
        rows_are Type.float64 [| Some 0.; None; None |]
          (Array.concat (List.map (floats t) [ "s"; "m"; "a" ])));
    test "var and std divide by n - 1, and are null under two values" (fun () ->
        let x' =
          Column.of_options Type.float64
            [| Some 7.; Some 1.; None; Some 2.; Some 3.; Some 3. |]
        in
        let t =
          require_ok ~pp:Error.pp
            (aggregate [ "g" ]
               Expr.[ "v" := var x; "s" := std x ]
               (v [ ("g", g); ("x", x') ]))
        in
        equal
          (array (option (float 1e-12)))
          [| None; Some 1.; None |] (floats t "v");
        equal
          (array (option (float 1e-12)))
          [| None; Some 1.; None |] (floats t "s"));
    test
      "var and std are their definition where sums of values or of squares \
       overflow" (fun () ->
        (* Each group's variance from its definition, within 1e-9: the decimal
           values round. *)
        let g = Column.v Type.int64 [| 0; 0; 0; 1; 1; 1; 2; 2; 3; 3; 3 |] in
        let x' =
          Column.v Type.float64
            [|
              1e154; -1e154; 0.; 1e308; 1e308; 1e308; 1e200; -1e200; 1.; 2.; 3.;
            |]
        in
        let t =
          require_ok ~pp:Error.pp
            (aggregate [ "g" ]
               Expr.[ "v" := var x; "s" := std x ]
               (v [ ("g", g); ("x", x') ]))
        in
        let w = array (option (float_rel ~rel:1e-9 ~abs:0.)) in
        equal w [| Some 1e308; Some 0.; Some infinity; Some 1. |] (floats t "v");
        equal w
          [| Some 1e154; Some 0.; Some (sqrt 2. *. 1e200); Some 1. |]
          (floats t "s"));
    test "std of values whose squares underflow is their definition" (fun () ->
        (* From the definition: [+-1e-170] has a sample variance of [2e-340],
           below float64's range, and a standard deviation of [sqrt 2 *.
           1e-170]; a frame of ordinary values is untouched. *)
        let g = Column.v Type.int64 [| 0; 0; 1; 1; 1; 2; 2 |] in
        let x' =
          Column.v Type.float64
            [| 1e-170; -1e-170; 2e-200; -2e-200; 0.; 1.; 2. |]
        in
        let t =
          require_ok ~pp:Error.pp
            (aggregate [ "g" ]
               Expr.[ "v" := var x; "s" := std x ]
               (v [ ("g", g); ("x", x') ]))
        in
        let w = array (option (float_rel ~rel:1e-9 ~abs:0.)) in
        equal ~msg:"var" w [| Some 0.; Some 0.; Some 0.5 |] (floats t "v");
        equal ~msg:"std" w
          [| Some (sqrt 2. *. 1e-170); Some 2e-200; Some (sqrt 0.5) |]
          (floats t "s"));
    test "std of the least subnormals is not zero" (fun () ->
        let g = Column.v Type.int64 [| 0; 0 |] in
        let x' = Column.v Type.float64 [| 5e-324; -5e-324 |] in
        let t =
          require_ok ~pp:Error.pp
            (aggregate [ "g" ]
               Expr.[ "s" := std x ]
               (v [ ("g", g); ("x", x') ]))
        in
        equal
          (array (option (float_rel ~rel:0.5 ~abs:0.)))
          [| Some 5e-324 |] (floats t "s"));
    test "std of tiny values with nulls is their definition" (fun () ->
        let x' =
          Column.of_options Type.float64
            [| Some 1e-170; None; Some (-1e-170); None |]
        in
        let g = Column.v Type.int64 [| 0; 0; 0; 0 |] in
        let t =
          require_ok ~pp:Error.pp
            (aggregate [ "g" ]
               Expr.[ "s" := std x ]
               (v [ ("g", g); ("x", x') ]))
        in
        equal
          (array (option (float_rel ~rel:1e-9 ~abs:0.)))
          [| Some (sqrt 2. *. 1e-170) |]
          (floats t "s"));
    test "std of tiny values over the whole table is their definition"
      (fun () ->
        let x' = Column.v Type.float64 [| 1e-170; -1e-170; 3e-170 |] in
        let t =
          require_ok ~pp:Error.pp
            (aggregate [] Expr.[ "s" := std x ] (v [ ("x", x') ]))
        in
        equal
          (array (option (float_rel ~rel:1e-9 ~abs:0.)))
          [| Some 2e-170 |] (floats t "s"));
    test "a constant frame's variance is zero at both ends" (fun () ->
        (* A rounded mean leaves every deviation the same [delta], whose square
           a scale must not bring into range: the exact variance of equal values
           is 0. *)
        let g = Column.v Type.int64 [| 0; 0; 0; 0; 0; 0; 0; 1; 1; 1 |] in
        let x' =
          Column.v Type.float64
            (Array.append (Array.make 7 1.3e-170) (Array.make 3 1.3e300))
        in
        let t =
          require_ok ~pp:Error.pp
            (aggregate [ "g" ]
               Expr.[ "v" := var x; "s" := std x ]
               (v [ ("g", g); ("x", x') ]))
        in
        let w = array (option float_exact) in
        equal ~msg:"var" w [| Some 0.; Some 0. |] (floats t "v");
        equal ~msg:"std" w [| Some 0.; Some 0. |] (floats t "s"));
    test "var keeps its digits far from zero, through per_list" (fun () ->
        let c = 1e12 in
        let xs =
          [| c +. (1. /. 256.); c; c +. (2. /. 256.); c +. (3. /. 256.) |]
        in
        let layout =
          Column.Varsize
            {
              validity = None;
              offsets = Nx.create Nx.int64 [| 2 |] [| 0L; 4L |];
              child = Column.of_tensor (Nx.create Nx.float64 [| 4 |] xs);
            }
        in
        let l =
          Result.get_ok (Column.of_layout (Any (Type.list Type.float64)) layout)
        in
        let t =
          require_ok ~pp:Error.pp
            (Query.run
               Query.(
                 of_table (Talon.v [ ("l", l) ])
                 |> select
                      Expr.
                        [
                          "v" :=
                            per_list
                              [ "e" := Col.v Kind.(list float) "l" ]
                              (var (Col.float "e"));
                        ]))
        in
        (* The variance of 1, 0, 2, 3 is 5/3, over 2^16. *)
        let want = 5. /. 3. /. 65536. in
        equal
          (array (option (float_rel ~rel:1e-12 ~abs:0.)))
          [| Some want |] (floats t "v"));
    test "an int32 mean is of the exact sum" (fun () ->
        (* 2^23 values of 2^31 - 1: a float64 sum of them passes 2^53, where it
           rounds. *)
        let n = 1 lsl 23 in
        let x = Column.of_tensor (Nx.full Nx.int32 [| n |] Int32.max_int) in
        let t =
          require_ok ~pp:Error.pp
            (aggregate [] Expr.[ "m" := mean (Col.int "x") ] (v [ ("x", x) ]))
        in
        equal (array (option float_exact)) [| Some 2147483647. |] (floats t "m"));
    test "a uint32 mean is exact in each frame" (fun () ->
        (* 2^22 values of 2^32 - 1 in group 0: a float64 sum of them passes
           2^53, where it rounds. *)
        let n = 1 lsl 22 in
        let g =
          Nx.concatenate ~axis:0
            [ Nx.zeros Nx.int64 [| n |]; Nx.ones Nx.int64 [| 3 |] ]
        in
        let x =
          Nx.concatenate ~axis:0
            [
              Nx.full Nx.uint32 [| n |] 0xFFFF_FFFFl; Nx.ones Nx.uint32 [| 3 |];
            ]
        in
        let t = v [ ("g", Column.of_tensor g); ("x", Column.of_tensor x) ] in
        let t =
          require_ok ~pp:Error.pp
            (aggregate [ "g" ] Expr.[ "m" := mean (Col.int "x") ] t)
        in
        equal
          (array (option float_exact))
          [| Some 4294967295.; Some 1. |]
          (floats t "m"));
    prop
      ~examples:
        [
          [ (0, Int64.max_int); (0, Int64.neg Int64.max_int) ];
          [ (0, Int64.max_int); (0, 1L) ];
        ]
      "a duration sum is exact or a data error"
      Gen.(
        let tick =
          frequency
            [
              (2, int64);
              (1, int64_range (-4L) 4L);
              ( 2,
                of_list
                  Int64.
                    [
                      max_int;
                      min_int;
                      pred max_int;
                      succ min_int;
                      shift_left 1L 42;
                      neg (shift_left 1L 42);
                      pred (shift_left 1L 42);
                      shift_left 1L 21;
                      pred (shift_left 1L 21);
                      neg (shift_left 1L 21);
                    ] );
            ]
        in
        list ~size:(int_range 0 12) (pair (int_range 0 2) tick))
      (fun rows ->
        (* A group's sum, exact in OCaml's 63-bit ints as high and low 32 bits
           apart, [None] where it leaves int64. *)
        let sum g =
          let add (hi, lo) (k, x) =
            if k <> g then (hi, lo)
            else
              ( hi + Int64.to_int (Int64.shift_right x 32),
                lo + Int64.to_int (Int64.logand x 0xFFFF_FFFFL) )
          in
          let hi, lo = List.fold_left add (0, 0) rows in
          let hi = hi + (lo asr 32) and lo = lo land 0xFFFF_FFFF in
          if hi < -0x8000_0000 || hi >= 0x8000_0000 then None
          else Some Int64.(logor (shift_left (of_int hi) 32) (of_int lo))
        in
        let groups =
          List.fold_left
            (fun gs (g, _) -> if List.mem g gs then gs else gs @ [ g ])
            [] rows
        in
        let sums = List.map sum groups in
        (* Whether group [g]'s values' magnitudes add past int64. *)
        let wide g =
          let m (k, x) = if k = g then Float.abs (Int64.to_float x) else 0. in
          List.fold_left (fun a r -> a +. m r) 0. rows > 0x1p63
        in
        cover "an overflow" (List.mem None sums);
        cover "a sum within int64 of values past it"
          ((not (List.mem None sums))
          && List.exists2 (fun g s -> wide g && Option.is_some s) groups sums);
        let t =
          v
            [
              ("g", Column.v Type.int64 (Array.of_list (List.map fst rows)));
              ("d", durations (Array.of_list (List.map snd rows)));
            ]
        in
        let got =
          match aggregate [ "g" ] Expr.[ "s" := sum (Col.span "d") ] t with
          | Error _ -> None
          | Ok t ->
              let s = Column.to_tensor Nx.int64 (column t "s") in
              Some (Array.to_list (Nx.to_array s))
        in
        let want =
          if List.mem None sums then None
          else Some (List.filter_map Fun.id sums)
        in
        equal (option (list int64)) want got);
    prop
      ~examples:[ ([ 1L ], 1 lsl 31) ]
      "a duration sum is exact over 2^31 values and more"
      (Gen.(
         let tick =
           frequency
             [
               (2, int64);
               (1, int64_range (-4L) 4L);
               (1, of_list Int64.[ max_int; min_int; shift_left 1L 32 ]);
             ]
         in
         let* xs = list ~size:(int_range 1 6) tick in
         let most = ((1 lsl 42) - 1) / List.length xs in
         let+ copies =
           of_list [ 1; 1 lsl 31; (1 lsl 31) + 1; 3 lsl 30; most ]
         in
         (xs, copies))
      |> Gen.with_pp (fun ppf (xs, copies) ->
          Format.fprintf ppf "%d copies of [%s]" copies
            (String.concat "; " (List.map Int64.to_string xs))))
      (fun (xs, copies) ->
        (* The sum of [xs], exact in OCaml's 63-bit ints as high and low 32 bits
           apart, then times [copies], [None] where either leaves int64. *)
        let add (hi, lo) x =
          ( hi + Int64.to_int (Int64.shift_right x 32),
            lo + Int64.to_int (Int64.logand x 0xFFFF_FFFFL) )
        in
        let hi, lo = List.fold_left add (0, 0) xs in
        let hi = hi + (lo asr 32) and lo = lo land 0xFFFF_FFFF in
        let s =
          if hi < -0x8000_0000 || hi >= 0x8000_0000 then None
          else Some Int64.(logor (shift_left (of_int hi) 32) (of_int lo))
        in
        let c = Int64.of_int copies in
        let times s =
          let fits =
            if Int64.compare s 0L >= 0 then
              Int64.compare s (Int64.div Int64.max_int c) <= 0
            else Int64.compare s (Int64.div Int64.min_int c) >= 0
          in
          if fits then Some (Int64.mul s c) else None
        in
        let want = Option.bind s times in
        cover "a sum of 2^31 values or more within int64"
          (copies >= 1 lsl 31 && Option.is_some want);
        cover "an overflow past 2^31 values"
          (copies >= 1 lsl 31 && Option.is_none want);
        equal (option int64) want (duration_sum (Array.of_list xs) ~copies));
    test "a duration sum that overflows fails at its group's first row"
      (fun () ->
        let t =
          v
            [
              ("g", g); ("d", durations [| 1L; Int64.max_int; 0L; 1L; 0L; 0L |]);
            ]
        in
        let e =
          require_error (aggregate [ "g" ] Expr.[ "s" := sum (Col.span "d") ] t)
        in
        contains ~sub:": row 1: the sum overflows duration[ns]." (error e));
    test "an output of a group fails at the group's first row" (fun () ->
        let t =
          v [ ("g", g); ("k", Column.v Type.int64 [| 1; 2; 3; 4; 5; 6 |]) ]
        in
        let ten s = Stdlib.(s * 10) in
        let y = Expr.(store Type.int8 (const ten $ sum k)) in
        let e = require_error (aggregate [ "g" ] Expr.[ "y" := y ] t) in
        contains ~sub:": row 1: int8 does not hold 140." (error e));
    test "a quantile rounds its product before its sum" (fun () ->
        let lo = -0x1.984a95b4668d2p-136 and hi = 0x1.c3bc32236009dp-965 in
        let t = v [ ("x", Column.v Type.float64 [| hi; lo |]) ] in
        let q =
          require_ok ~pp:Error.pp
            (aggregate [] Expr.[ "q" := quantile 0.75 x ] t)
        in
        rows_are Type.float64 [| Some (-0x1.984a95b4668dp-138) |] (floats q "q"));
    (* The float64 sum in row order: 2^53 + 1 rounds to 2^53, ties to even. *)
    test "a float64 mean sums its values in row order" (fun () ->
        let p = 0x1p53 in
        let mean xs =
          floats
            (require_ok ~pp:Error.pp
               (aggregate []
                  Expr.[ "m" := mean x ]
                  (v [ ("x", Column.v Type.float64 xs) ])))
            "m"
        in
        rows_are Type.float64 [| Some 0. |] (mean [| p; 1.; -.p |]);
        rows_are Type.float64 [| Some (1. /. 3.) |] (mean [| p; -.p; 1. |]));
    (* Two's complement modulo 2^64, as nx's int64 computes. *)
    test "an int64 sum wraps as nx's integers do" (fun () ->
        let sum xs =
          let t = v [ ("k", int64s xs) ] in
          let s =
            require_ok ~pp:Error.pp (aggregate [] Expr.[ "s" := sum k ] t)
          in
          Nx.to_array (Column.to_tensor Nx.int64 (column s "s"))
        in
        equal (array int64) [| Int64.min_int |] (sum [| Int64.max_int; 1L |]);
        equal (array int64) [| Int64.max_int |] (sum [| Int64.min_int; -1L |]));
  ]

let frames =
  let x = Col.float "x" in
  [
    test "arg_min over an order is a position in that order" (fun () ->
        let t = v [ ("x", Column.v Type.float64 [| 3.; 1.; 2. |]) ] in
        rows_are Type.int64
          [| Some 2; Some 2; Some 2 |]
          (Column.options Kind.int
             (result Expr.(over ~order:[ Order.desc "x" ] (arg_min x)) t)));
  ]

let blocking_emits_nothing =
  test "emits nothing of a step that reads other rows when it fails" (fun () ->
      let k = Col.int "k" in
      let t =
        of_batches
          [
            v [ ("k", Column.v Type.int64 [| 1; 2 |]) ];
            v [ ("k", Column.v Type.int64 [| 300 |]) ];
          ]
      in
      let y = Expr.(store Type.int8 (const Fun.id $ k)) in
      let seen = ref 0 in
      let q =
        Query.derive Expr.[ "y" := y; "z" := over (sum k) ] (Query.of_table t)
      in
      let e =
        require_error
          (Query.fold q ~init:() (fun () b -> seen := !seen + rows b))
      in
      contains ~sub:": row 2: int8 does not hold 300." (error e);
      equal ~msg:"rows folded" int 0 !seen)

(* One column's reductions

   An aggregate's reductions of one column share their count, sums and mean, and
   an integer's or a boolean's extreme is its value's. Rows of every integer,
   float and boolean type, at most forty, some null, cut into at most four
   groups by [g]. *)

let reduced =
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
    ]

let grouped types =
  Gen.with_pp
    (fun ppf (G.Sample (ty, vs), g) ->
      Format.fprintf ppf "%a, groups [%s]: %d rows" Type.pp ty
        (String.concat " " (Array.to_list (Array.map string_of_int g)))
        (Array.length vs))
    (Gen.bind (Gen.of_list types) (fun (Type.Any ty) ->
         Gen.bind (G.options ty) (fun vs ->
             let size = Gen.constant (Array.length vs) in
             (* A third of the time group [0]'s values are all null. *)
             let nulled g =
               let vs =
                 Array.mapi (fun i v -> if g.(i) = 0 then None else v) vs
               in
               (G.Sample (ty, vs), g)
             in
             Gen.bind
               (Gen.array ~size (Gen.int_range 0 3))
               (fun g ->
                 Gen.frequency
                   [
                     (2, Gen.constant (G.Sample (ty, vs), g));
                     (1, Gen.constant (nulled g));
                   ]))))

(* The rows of group [k]. *)
let group_rows vs g k =
  List.filter (fun i -> g.(i) = k) (List.init (Array.length vs) Fun.id)

(* Nulls, a group whose every value is null, and a group of one row. *)
let cover_groups vs g =
  let groups = List.sort_uniq compare (Array.to_list g) in
  let rows k = group_rows vs g k in
  cover "a null" (Array.exists Option.is_none vs);
  cover "a group of nulls alone"
    (List.exists
       (fun k -> List.for_all (fun i -> vs.(i) = None) (rows k))
       groups);
  cover "a group of one row"
    (List.exists (fun k -> List.length (rows k) = 1) groups)

let of_groups ty vs g =
  v [ ("g", Column.v Type.int64 g); ("x", Column.of_options ty vs) ]

(* Far from zero: each group's values are [centre + j h], with [h = 2^-20], a
   power of two, and [j] integers of 20 bits: every value and the group's mean
   numerator are exact. The exact sample variance is the rational [h^2 S / (n -
   1)], [S] the integers' [\u{03A3} j^2 - (\u{03A3} j)^2 / n] by the division
   with remainder of [\u{03A3} j] by [n], whose terms fit an [int]. *)
let exact_squares js =
  let n = Array.length js in
  let total = Array.fold_left ( + ) 0 js in
  let q = total / n and r = total mod n in
  let squares = Array.fold_left (fun a j -> a + (j * j)) 0 js in
  let s = squares - (q * q * n) - (2 * q * r) in
  float_of_int s -. (float_of_int (r * r) /. float_of_int n)

let exact_variance js =
  Float.ldexp (exact_squares js) (-40) /. float_of_int (Array.length js - 1)

let far_from_zero =
  Gen.(
    let+ n = int_range 2 100_000
    and+ groups = int_range 1 4
    and+ seed = nat
    and+ centre = int_range 0 1000 in
    (n, groups, seed, 1e9 +. float_of_int centre))

let law_exact_variance (n, groups, seed, centre) =
  let r = Random.State.make [| seed |] in
  let id = Array.init n (fun _ -> Random.State.int r groups) in
  let js = Array.init n (fun _ -> Random.State.int r 0x10_0000 - 0x8_0000) in
  let t =
    v
      [
        ("g", Column.v Type.int64 id);
        ( "x",
          Column.v Type.float64
            (Array.map
               (fun j -> centre +. Float.ldexp (float_of_int j) (-20))
               js) );
      ]
  in
  let t =
    require_ok ~pp:Error.pp
      (aggregate [ "g" ]
         Expr.[ "v" := var (Col.float "x"); "s" := std (Col.float "x") ]
         t)
  in
  let of_group k =
    let idx = List.filter (fun i -> id.(i) = k) (List.init n Fun.id) in
    Array.of_list (List.map (fun i -> js.(i)) idx)
  in
  let keys = Column.options Kind.int (column t "g") in
  let vs = floats t "v" and ss = floats t "s" in
  (* Chan, Golub and LeVeque's bound for the corrected two-pass, N u + N^3 k^2
     u^3, is below a few N u at this condition. *)
  let bound count = 16. *. float_of_int count *. 0x1p-53 in
  Array.iteri
    (fun i key ->
      match key with
      | None -> ()
      | Some k ->
          let js = of_group k in
          let count = Array.length js in
          if count >= 2 then begin
            let exact = exact_variance js in
            let tol = float_rel ~rel:(bound count) ~abs:0. in
            equal ~msg:"var" (option tol) (Some exact) vs.(i);
            equal ~msg:"std" (option tol) (Some (sqrt exact)) ss.(i)
          end)
    keys

(* Each output of an aggregate of [count], [min], [max] and, over numbers,
   [sum], [mean], [var] and [std] of one column is the bytes of its own
   aggregate. *)
let law_shared (G.Sample (ty, vs), g) =
  cover_groups vs g;
  let x = Col.v (Type.kind ty) "x" in
  let out n e = (n, Expr.(n := e)) in
  let numbers =
    match Kind.provably_equal (Type.kind ty) Kind.bool with
    | Some _ -> []
    | None ->
        Expr.
          [
            out "s" (sum x); out "m" (mean x); out "v" (var x); out "d" (std x);
          ]
  in
  let os =
    Expr.[ out "n" (count x); out "lo" (min x); out "hi" (max x) ] @ numbers
  in
  let t = of_groups ty vs g in
  let run os = require_ok ~pp:Error.pp (aggregate [ "g" ] os t) in
  let together = run (List.map snd os) in
  List.iter
    (fun (n, o) ->
      equal ~msg:n (list string)
        (buffers (column (run [ o ]) n))
        (buffers (column together n)))
    os

(* Streamed aggregates

   An aggregate of the rows of several batches gives exactly the blocking
   aggregate of those rows in one batch: the same bits, NaNs compared as NaNs,
   and the same failure. *)

(* [stored_words dt c] is [c]'s values at the stored dtype [dt], as int64, the
   nulls in their places. *)
let stored_words (type a b) (dt : (a, b) Nx.dtype) c =
  let n = Column.length c in
  let held =
    match Column.validity c with
    | None -> Array.make n true
    | Some v -> Nx.to_array (Nx.cast Nx.bool v)
  in
  let kept = List.filter (fun i -> held.(i)) (List.init n Fun.id) in
  let kept = Array.of_list (List.map Int64.of_int kept) in
  let idx = Nx.create Nx.int64 [| Array.length kept |] kept in
  let vs =
    Nx.to_array
      (Nx.cast Nx.int64
         (Column.to_tensor dt (column (take idx (v [ ("c", c) ])) "c")))
  in
  let next = ref 0 in
  List.init n (fun i ->
      if not held.(i) then "null"
      else begin
        incr next;
        Int64.to_string vs.(!next - 1)
      end)

let show c =
  let (Type.Any ty) = Column.type_ c in
  let is t = Type.equal ty t in
  let float v = if Float.is_nan v then "nan" else Printf.sprintf "%h" v in
  let decoded (type a) (ty : a Type.t) (value : a -> string) =
    Array.to_list
      (Array.map
         (function None -> "null" | Some x -> value x)
         (Column.options (Type.kind ty) c))
  in
  if is Type.int8 then stored_words Nx.int8 c
  else if is Type.int16 then stored_words Nx.int16 c
  else if is Type.int32 then stored_words Nx.int32 c
  else if is Type.int64 then stored_words Nx.int64 c
  else if is Type.uint8 then stored_words Nx.uint8 c
  else if is Type.uint16 then stored_words Nx.uint16 c
  else if is Type.uint32 then stored_words Nx.uint32 c
  else if is Type.uint64 then stored_words Nx.uint64 c
  else if Option.is_some (Kind.provably_equal (Type.kind ty) Kind.span) then
    stored_words Nx.int64 c
  else
    match Kind.provably_equal (Type.kind ty) Kind.float with
    | Some Equal -> decoded ty float
    | None -> decoded ty (Testable.to_string (G.witness ty))

let outcome_w =
  Windtrap.result (list (Testable.pair string (list string))) string

let outcome = function
  | Ok t ->
      Ok
        (List.map
           (fun (n, _) -> (n, show (column t n)))
           (Schema.columns (schema t)))
  | Error e -> Error (Format.asprintf "%a" Error.pp e)

(* The keys a draw groups by: integers, none, text with nulls, compound, and
   floats with NaN, nulls and both zeros. *)
let key_choices =
  [
    ([ "g" ], fun _ -> []);
    ([], fun _ -> []);
    ( [ "t" ],
      fun g ->
        let text =
          Array.map
            (fun k -> [| Some "a"; Some ""; Some "\u{00E9}"; None |].(k mod 4))
            g
        in
        [ ("t", Column.of_options Type.string text) ] );
    ( [ "t"; "g" ],
      fun g ->
        let text = Array.map (fun k -> [| Some "x"; None |].(k mod 2)) g in
        [ ("t", Column.of_options Type.string text) ] );
    ( [ "f" ],
      fun g ->
        let keys =
          Array.map
            (fun k -> [| Some 0.; Some (-0.); Some Float.nan; None |].(k mod 4))
            g
        in
        [ ("f", Column.of_options Type.float64 keys) ] );
  ]

let streamable = reduced @ Type.[ Any string; Any date; Any (duration Ms) ]

let batched =
  Gen.(
    bind
      (pair (grouped streamable) (int_range 0 (List.length key_choices - 1)))
      (fun (((G.Sample (ty, vs) as sample), g), k) ->
        let by, extra = List.nth key_choices k in
        let whole =
          v
            (("g", Column.v Type.int64 g)
            :: ("x", Column.of_options ty vs)
            :: extra g)
        in
        map (fun t -> (sample, by, whole, t)) (G.split whole)))

(* [same_as_one_batch ~by ?plan os stream] states that the aggregate of
   [stream]'s rows is what it is over the same rows in one batch. *)
let same_as_one_batch ~by ?(plan = Fun.id) os stream =
  let whole = run_ok (Query.of_table stream) in
  let run t =
    outcome (Query.run (Query.aggregate ~by os (plan (Query.of_table t))))
  in
  equal outcome_w (run whole) (run stream)

let streamed_edges =
  [
    test "over several batches, within reads a record's fields" (fun () ->
        let t i x =
          v
            [
              ("i", Column.v Type.int64 [| i |]);
              ("f", Column.v Type.float64 [| x |]);
            ]
        in
        let plan =
          Query.derive Expr.[ "r" := record [ "f" := Col.float "f" ] ]
        in
        same_as_one_batch ~by:[] ~plan
          Expr.[ within (Col.v Record.kind "r") ("s" := sum (Col.float "f")) ]
          (of_batches [ t 0 1.; t 1 2. ]));
    test "over several batches, localize_ordered sees each group's rows"
      (fun () ->
        let hour = 3_600_000_000_000L in
        let w h = Time.of_ns (Int64.mul (Int64.of_int h) hour) in
        let t ks hs =
          v
            [
              ("k", Column.v Type.string ks);
              ("w", Column.v (Type.wall Ms) (Array.map w hs));
            ]
        in
        let e =
          Expr.Temporal.localize_ordered Tz.utc ~ambiguous:`Fail ~gap:`Fail
            (Col.wall "w")
        in
        same_as_one_batch ~by:[ "k" ]
          Expr.[ "m" := max e ]
          (of_batches [ t [| "a"; "b" |] [| 10; 9 |]; t [| "a" |] [| 11 |] ]));
    test "over batches a filter empties, all of them, is the blocking one"
      (fun () ->
        let t xs = v [ ("x", Column.v Type.int64 xs) ] in
        let x = Col.int "x" in
        let plan = Query.filter Expr.(x < int 3) in
        let stream = of_batches [ t [| 5 |]; t [| 6 |] ] in
        same_as_one_batch ~by:[] ~plan
          Expr.[ "o" := only x; "n" := rows ]
          stream;
        same_as_one_batch ~by:[ "x" ] ~plan Expr.[ "n" := rows ] stream);
    test "lays a group of nulls out in the same bytes whatever the batches"
      (fun () ->
        let t n =
          v
            [
              ("g", Column.v Type.int64 (Array.make n 0));
              ("x", Column.of_options Type.int64 (Array.make n None));
            ]
        in
        let x = Col.int "x" in
        let q by os t = Query.aggregate ~by os (Query.of_table t) in
        let same by os =
          same_outcome (q by os (t 3)) (q by os (of_batches [ t 2; t 1 ]))
        in
        (* A group of no row: a filter empties both batches. *)
        let empties by os =
          let p t =
            Query.aggregate ~by os
              (Query.filter Expr.(Col.int "g" <> int 0) (Query.of_table t))
          in
          same_outcome (p (t 2)) (p (of_batches [ t 1; t 1 ]))
        in
        List.iter
          (fun by ->
            same by Expr.[ "r" := arg_min x ];
            same by Expr.[ "r" := arg_max x ];
            same by Expr.[ "r" := mean x ];
            empties by Expr.[ "r" := arg_min x ];
            empties by Expr.[ "r" := arg_max x ];
            empties by Expr.[ "r" := mean x ])
          [ []; [ "g" ] ]);
    test "gives var and std of values of every scale, in any batching"
      (fun () ->
        (* From the definition. A first tiny value is negligible beside
           [+-1e154]; [+-1e-170] has a variance below float64's range;
           [+-max_float] has a variance and a standard deviation past it. *)
        let x = Col.float "x" in
        let cells =
          [|
            (0, 1e-250);
            (0, 1e154);
            (0, -1e154);
            (1, 1e154);
            (1, -1e154);
            (1, 1e-250);
            (2, 1e-170);
            (2, -1e-170);
            (3, 1.);
            (3, 1.);
            (3, 1.);
            (4, max_float);
            (4, -.max_float);
          |]
        in
        let table a b =
          v
            [
              ("g", Column.v Type.int64 (Array.map fst (Array.sub cells a b)));
              ("x", Column.v Type.float64 (Array.map snd (Array.sub cells a b)));
            ]
        in
        let n = Array.length cells in
        let os = Expr.[ "v" := var x; "s" := std x ] in
        let tol = float_rel ~rel:1e-12 ~abs:0. in
        for cut = 0 to n do
          let t = of_batches [ table 0 cut; table cut (n - cut) ] in
          same_as_one_batch ~by:[ "g" ] os t;
          let q = Query.aggregate ~by:[ "g" ] os (Query.of_table t) in
          let r = run_ok q in
          equal ~msg:"var"
            (array (option tol))
            [| Some 1e308; Some 1e308; Some 0.; Some 0.; Some infinity |]
            (floats r "v");
          equal ~msg:"std"
            (array (option tol))
            [|
              Some 1e154;
              Some 1e154;
              Some (sqrt 2. *. 1e-170);
              Some 0.;
              Some infinity;
            |]
            (floats r "s")
        done);
    test "after a leading batch a filter empties is the blocking one" (fun () ->
        let t xs = v [ ("x", Column.v Type.int64 xs) ] in
        let plan = Query.filter Expr.(Col.int "x" < int 3) in
        same_as_one_batch ~by:[] ~plan
          Expr.[ "o" := only (Col.int "x") ]
          (of_batches [ t [| 5 |]; t [| 1; 2 |] ]));
  ]

(* The reductions of one column that cannot fail. [only] fails over two values,
   and a duration sum where it overflows: {!fallible} runs them alone, so that
   their failure hides no other output's values. *)
let reductions_of (G.Sample (ty, _)) =
  let x = Col.v (Type.kind ty) "x" in
  let any =
    Expr.
      [
        "rows" := rows;
        "count" := count x;
        "n_unique" := n_unique x;
        "fn" := store Type.int64 (const succ $ rows);
      ]
  in
  let ordered =
    Expr.
      [
        "min" := min x;
        "max" := max x;
        "first" := first x;
        "last" := last x;
        "arg_min" := arg_min x;
        "arg_max" := arg_max x;
      ]
  in
  let is k = Option.is_some (Kind.provably_equal (Type.kind ty) k) in
  let numeric =
    if is Kind.int || is Kind.float then
      Expr.
        [
          "sum" := sum x;
          "mean" := mean x;
          "var" := var x;
          "std" := std x;
          "median" := median x;
          "quantile" := quantile 0.5 x;
        ]
    else []
  in
  any @ ordered @ numeric

let fallible (G.Sample (ty, _)) =
  let x = Col.v (Type.kind ty) "x" in
  let is k = Option.is_some (Kind.provably_equal (Type.kind ty) k) in
  Expr.[ "only" := only x ]
  @ if is Kind.span then Expr.[ "sum" := sum x ] else []

(* The outputs of {!reductions_of} that stream together. *)
let streamed_outs (G.Sample (ty, _)) =
  let x = Col.v (Type.kind ty) "x" in
  let is k = Option.is_some (Kind.provably_equal (Type.kind ty) k) in
  Expr.
    [
      "rows" := rows;
      "count" := count x;
      "min" := min x;
      "max" := max x;
      "first" := first x;
      "last" := last x;
      "arg_min" := arg_min x;
      "arg_max" := arg_max x;
    ]
  @
  if is Kind.int || is Kind.float then
    Expr.[ "sum" := sum x; "mean" := mean x; "var" := var x; "std" := std x ]
  else []

let law_streamed (sample, by, whole, stream) =
  let agg os t = Query.aggregate ~by os (Query.of_table t) in
  let run os t = outcome (Query.run (agg os t)) in
  let outs = reductions_of sample and streams = streamed_outs sample in
  let reduced = Query.run (agg outs whole) in
  let without_value =
    match reduced with
    | Error _ -> false
    | Ok t ->
        let ints n = Nx.to_array (Column.to_tensor Nx.int64 (column t n)) in
        Array.exists2 (fun n r -> n = 0L && r > 0L) (ints "count") (ints "rows")
  in
  (* A filter that empties batches, a leading one included, hands the aggregate
     batches of no rows. *)
  let empties =
    List.exists
      (fun b ->
        let g = Column.to_tensor Nx.int64 (column b "g") in
        Nx.dim 0 g > 0 && Nx.item [] (Nx.all (Nx.equal_s g 0L)))
      (batches stream)
    && List.length (batches stream) > 1
  in
  cover "several batches" (List.length (batches stream) > 1);
  cover "no row" (rows whole = 0);
  cover "a group of nulls only" without_value;
  cover "a batch the filter empties" empties;
  equal ~msg:"all" outcome_w (outcome reduced) (run outs stream);
  (* {!Query.run}'s layouts do not depend on batches, the bytes under a null
     included: a group without a value holds nothing to merge. *)
  same_outcome (agg streams whole) (agg streams stream);
  List.iter
    (fun o -> equal outcome_w (run [ o ] whole) (run [ o ] stream))
    (fallible sample);
  let filtered os t =
    outcome
      (Query.run
         (Query.of_table t
         |> Query.filter Expr.(Col.int "g" <> int 0)
         |> Query.aggregate ~by os))
  in
  equal ~msg:"filtered" outcome_w (filtered streams whole)
    (filtered streams stream)

(* Streamed var and std against their definition

   Each group's values are [(centre + j) 2^e] for integers [j] of at most 10
   bits, exact in float64: their sample variance is the rational [S / (n - 1)
   2^2e], [S] the squared deviations of the [j]s ({!exact_squares}), rounded
   once, so that a run within Chan, Golub and LeVeque's bound for the shifted
   update, [n kappa u] with [kappa^2 < 1 + n], agrees with it within that
   tolerance. Past [max_float] the definition is infinite. *)

type moments_group = { exponent : int; centre : int; js : int option array }

let moments_group =
  Gen.(
    let* centre = of_list [ 0; 1 lsl 40 ] in
    let* exponent =
      of_list
        (if centre = 0 then [ 0; -560; -1040; 500; 502; 900; 1000 ]
         else [ 0; -560; -1040; 500; 502; 900 ])
    in
    let size =
      frequency
        [
          (2, constant 0); (2, constant 1); (2, constant 2); (6, int_range 3 40);
        ]
    in
    let value =
      frequency
        [ (7, map Option.some (int_range (-1023) 1023)); (1, constant None) ]
    in
    let+ js = array ~size value in
    { exponent; centre; js })

let moments_draw =
  Gen.(
    let* groups = array ~size:(int_range 1 6) moments_group in
    let* seed = nat in
    let rows =
      Array.concat
        (Array.to_list
           (Array.mapi
              (fun g { exponent; centre; js } ->
                Array.map
                  (fun j ->
                    let x =
                      Option.map
                        (fun j ->
                          Float.ldexp (float_of_int (centre + j)) exponent)
                        j
                    in
                    (g, x))
                  js)
              groups))
    in
    let r = Random.State.make [| seed |] in
    let keys = Array.map (fun _ -> Random.State.bits r) rows in
    let order =
      List.sort
        (fun i j -> Int.compare keys.(i) keys.(j))
        (List.init (Array.length rows) Fun.id)
    in
    let rows = Array.of_list (List.map (fun i -> rows.(i)) order) in
    let whole =
      v
        [
          ("g", Column.v Type.int64 (Array.map fst rows));
          ("x", Column.of_options Type.float64 (Array.map snd rows));
        ]
    in
    let+ stream = G.split whole in
    (groups, whole, stream))

let moments_outs =
  let x = Col.float "x" in
  Expr.[ "v" := var x; "s" := std x ]

let moments_tolerance n =
  let n = float_of_int n in
  float_rel ~rel:(8. *. n *. sqrt (n +. 1.) *. 0x1p-53) ~abs:0x1p-1070

(* [expected_moments g] is group [g]'s variance and standard deviation by their
   definition, [None] under two values. *)
let expected_moments { exponent; js; _ } =
  let js = Array.of_list (List.filter_map Fun.id (Array.to_list js)) in
  let n = Array.length js in
  if n < 2 then (n, None, None)
  else
    let per_df = exact_squares js /. float_of_int (n - 1) in
    ( n,
      Some (Float.ldexp per_df (2 * exponent)),
      Some (Float.ldexp (sqrt per_df) exponent) )

let law_streamed_moments (groups, whole, stream) =
  cover "several batches" (List.length (batches stream) > 1);
  let n_of g =
    let n, _, _ = expected_moments g in
    n
  in
  cover "a group of one value" (Array.exists (fun g -> n_of g = 1) groups);
  cover "a group of nulls alone"
    (Array.exists (fun g -> Array.length g.js > 0 && n_of g = 0) groups);
  cover "a group of two values" (Array.exists (fun g -> n_of g = 2) groups);
  cover "a near-constant group"
    (Array.exists (fun g -> g.centre > 0 && n_of g >= 2) groups);
  cover "a group of finite variance whose squares overflow a plain sum"
    (Array.exists
       (fun g ->
         let _, v, _ = expected_moments g in
         let finite =
           match v with Some v -> Float.is_finite v | None -> false
         in
         let squares =
           Array.fold_left
             (fun a -> function
               | Some j -> a +. (Float.ldexp (float_of_int j) g.exponent ** 2.)
               | None -> a)
             0. g.js
         in
         finite && not (Float.is_finite squares))
       groups);
  let q t = Query.aggregate ~by:[ "g" ] moments_outs (Query.of_table t) in
  let t = run_ok (q stream) in
  let keys = Column.options Kind.int (column t "g") in
  let vs = floats t "v" and ss = floats t "s" in
  Array.iteri
    (fun i key ->
      match key with
      | None -> ()
      | Some g ->
          let n, v, s = expected_moments groups.(g) in
          let tol = moments_tolerance (Int.max n 1) in
          equal ~msg:"var" (option tol) v vs.(i);
          equal ~msg:"std" (option tol) s ss.(i))
    keys;
  equal ~msg:"blocking" outcome_w
    (outcome (Query.run (q whole)))
    (outcome (Query.run (q stream)));
  same_outcome (q whole) (q stream)

(* A cast at a row of one output and a conflict at the row of another, in every
   order: the failure reported is the one that the single batch finds first. [t]
   holds, for each draw, integers and a text that holds an integer or does not
   parse. *)
let failing_ints =
  Gen.(
    let int =
      frequency
        [
          (6, int_range (-20) 20);
          (1, one_of [ constant 1000; constant (-1000) ]);
        ]
    in
    pair (list ~size:(int_range 0 30) (pair int int)) (int_range 0 3))

let law_failures (rows_, k) =
  let n = List.length rows_ in
  (* Ties: a row where a cast fails, and that is the first row of a group whose
     [x] values conflict. *)
  let first_rows =
    List.filter
      (fun i ->
        not
          (List.exists
             (fun j -> j < i && j mod 3 = i mod 3)
             (List.init n Fun.id)))
      (List.init n Fun.id)
  in
  let fails i =
    let y = snd (List.nth rows_ i) in
    y > 127 || y < -128
  in
  let conflict i =
    let group = List.filter (fun j -> j mod 3 = i mod 3) (List.init n Fun.id) in
    let vs =
      List.sort_uniq compare (List.map (fun j -> fst (List.nth rows_ j)) group)
    in
    List.length vs > 1
  in
  cover "a tie" (List.exists (fun i -> fails i && conflict i) first_rows);
  let xs = Array.of_list (List.map fst rows_) in
  let ys = Array.of_list (List.map snd rows_) in
  let text =
    Array.map (fun i -> if i = 13 then "not a number" else string_of_int i) xs
  in
  let keys = Array.init n (fun i -> i mod 3) in
  let t =
    v
      [
        ("g", Column.v Type.int64 keys);
        ("x", Column.v Type.int64 xs);
        ("y", Column.v Type.int64 ys);
        ("s", Column.v Type.string text);
      ]
  in
  let x = Col.int "x" and y = Col.int "y" and s = Col.string "s" in
  let narrowed = Expr.(cast Type.int8 y) in
  let parsed = Expr.(Str.parse Type.int64 s) in
  let outs =
    Expr.
      [
        [ "o" := only x; "c" := sum narrowed ];
        [ "c" := sum narrowed; "o" := only x ];
        [ "a" := only x + sum narrowed ];
        [ "a" := sum narrowed + only x ];
        [ "p" := first parsed; "o" := only y; "c" := sum narrowed ];
        [ "o" := only y; "p" := first parsed ];
      ]
  in
  let by = [ []; [ "g" ] ] in
  (* Batches of at most [k + 1] rows, so that a failure outlives compactions. *)
  let stream =
    let size = k + 1 in
    let cut i =
      let stop = Stdlib.min n ((i + 1) * size) in
      Talon.take (Nx.arange Nx.int64 (i * size) stop 1) t
    in
    if n = 0 then t else Talon.of_batches (List.init (((n - 1) / size) + 1) cut)
  in
  cover "three batches or more" (List.length (batches stream) >= 3);
  let run os by t =
    outcome (Query.run (Query.aggregate ~by os (Query.of_table t)))
  in
  List.iter
    (fun by ->
      List.iter
        (fun os -> equal outcome_w (run os by t) (run os by stream))
        outs)
    by

(* [min] and [max] of integers and booleans are each group's least and greatest
   value, null over none. *)
let law_extremes (G.Sample (ty, vs), g) =
  cover_groups vs g;
  let x = Col.v (Type.kind ty) "x" in
  let t =
    require_ok ~pp:Error.pp
      (aggregate [ "g" ]
         Expr.[ "lo" := min x; "hi" := max x ]
         (of_groups ty vs g))
  in
  let keys = Column.options Kind.int (column t "g") in
  let fold pick =
    Array.map
      (fun k ->
        let values =
          List.filter_map (fun i -> vs.(i)) (group_rows vs g (Option.get k))
        in
        match values with
        | [] -> None
        | v :: rest ->
            Some
              (List.fold_left
                 (fun a b -> if pick (Type.compare_value ty b a) then b else a)
                 v rest))
      keys
  in
  let k = Type.kind ty in
  rows_are ty (fold (fun c -> c < 0)) (Column.options k (column t "lo"));
  rows_are ty (fold (fun c -> c > 0)) (Column.options k (column t "hi"))

(* A bool column at bit offsets: its min and max are each group's least and
   greatest value, [false] before [true], null over none. *)
let law_bit_extremes (c, g) =
  let n = Array.length c.values in
  let held i = match c.valid with Some v -> v.(i) | None -> true in
  cover "a value at a bit offset" (n > 0 && fst c.at mod 8 <> 0);
  let t = v [ ("g", Column.v Type.int64 g); ("x", of_bits c) ] in
  let x = Col.bool "x" in
  let t =
    require_ok ~pp:Error.pp
      (aggregate [ "g" ] Expr.[ "lo" := min x; "hi" := max x ] t)
  in
  let fold pick =
    Array.map
      (fun k ->
        let rows = List.filter held (group_rows c.values g (Option.get k)) in
        match List.map (fun i -> c.values.(i)) rows with
        | [] -> None
        | vs -> Some (List.fold_left pick (List.hd vs) vs))
      (Column.options Kind.int (column t "g"))
  in
  let bools n = Column.options Kind.bool (column t n) in
  rows_are Type.bool (fold ( && )) (bools "lo");
  rows_are Type.bool (fold ( || )) (bools "hi")

let bit_groups =
  Gen.with_pp
    (fun ppf (c, g) ->
      Format.fprintf ppf "%a, groups [%s]" pp_bits c
        (String.concat " " (Array.to_list (Array.map string_of_int g))))
    (Gen.bind lengths (fun n ->
         Gen.pair (bits_column n)
           (Gen.array ~size:(Gen.constant n) (Gen.int_range 0 3))))

(* [int64_words t n] is the 64-bit words of [t]'s column [n]. *)
let int64_words t n =
  match Column.layout (column t n) with
  | Fixed { values = P x; _ } -> Nx.to_array (Nx.bitcast Nx.int64 x)
  | _ -> assert false

(* A group of nulls alone and a group of one row. *)
let null_group_and_one_row =
  (G.Sample (Type.int64, [| None; None; Some 1 |]), [| 0; 0; 1 |])

(* A group of nulls alone beside one of two values, in two batches. *)
let moments_of_nulls =
  let group js = { exponent = 0; centre = 0; js } in
  let rows g xs =
    v [ ("g", i64 g); ("x", Column.of_options Type.float64 xs) ]
  in
  ( [| group [| None; None |]; group [| Some 1; Some 2 |] |],
    rows [| 0; 0; 1; 1 |] [| None; None; Some 1.; Some 2. |],
    of_batches
      [
        rows [| 0; 0 |] [| None; None |]; rows [| 1; 1 |] [| Some 1.; Some 2. |];
      ] )

let one_column =
  let x = Col.float "x" in
  let first vs =
    let t = v [ ("x", Column.v Type.float64 vs) ] in
    require_ok ~pp:Error.pp
      (aggregate [] Expr.[ "lo" := min x; "hi" := max x ] t)
  in
  [
    prop ~count:20 "var and std follow the exact variance far from zero"
      far_from_zero law_exact_variance;
    prop ~examples:[ null_group_and_one_row ]
      "an integer's or a boolean's min and max are its extremes"
      (grouped
         (List.filter
            (fun (Type.Any ty) ->
              Option.is_none (Kind.provably_equal (Type.kind ty) Kind.float))
            reduced))
      law_extremes;
    test "a float's -0. and 0. are one value, its first row's" (fun () ->
        List.iter
          (fun vs ->
            let t = first vs in
            rows_are Type.float64 [| Some vs.(0) |] (floats t "lo");
            rows_are Type.float64 [| Some vs.(0) |] (floats t "hi"))
          [ [| -0.; 0. |]; [| 0.; -0. |] ]);
    prop "a bool's min and max over bit views are its extremes" bit_groups
      law_bit_extremes;
    test "uint64 extremes order unsigned, int64 ones signed" (fun () ->
        let g = Column.v Type.int64 [| 0; 0; 0; 1; 1 |] in
        let extremes x =
          let t = v [ ("g", g); ("x", Column.of_tensor x) ] in
          let x = Col.int "x" in
          let t =
            require_ok ~pp:Error.pp
              (aggregate [ "g" ] Expr.[ "lo" := min x; "hi" := max x ] t)
          in
          (int64_words t "lo", int64_words t "hi")
        in
        let top = 0x8000_0000_0000_0000L in
        let u = [| -1L; 1L; top; top; 0L |] in
        let lo, hi = extremes (Nx.create Nx.uint64 [| 5 |] u) in
        equal ~msg:"uint64 min" (array int64) [| 1L; 0L |] lo;
        equal ~msg:"uint64 max" (array int64) [| -1L; top |] hi;
        let i = [| Int64.min_int; Int64.max_int; 0L; -1L; 0L |] in
        let lo, hi = extremes (Nx.create Nx.int64 [| 5 |] i) in
        equal ~msg:"int64 min" (array int64) [| Int64.min_int; -1L |] lo;
        equal ~msg:"int64 max" (array int64) [| Int64.max_int; 0L |] hi);
  ]

let aggregate_laws =
  [
    prop ~examples:[ null_group_and_one_row ]
      "gives each output its own aggregate's value" (grouped reduced) law_shared;
    prop ~count:400 "over several batches is the blocking one, bit for bit"
      batched law_streamed;
    prop ~count:300 ~examples:[ moments_of_nulls ]
      "over several batches gives var and std's definition" moments_draw
      law_streamed_moments;
    prop "over several batches fails as the blocking one does" failing_ints
      law_failures;
  ]

(* Joins *)

let join ?kind ?each_left ?each_right on l r =
  Query.join ?kind ?each_left ?each_right ~on (Query.of_table r)
    (Query.of_table l)

let joined ?kind ?each_left ?each_right on l r =
  run_ok (join ?kind ?each_left ?each_right on l r)

let column_is t (n, expected) =
  equal ~msg:n
    (list (option int))
    expected
    (Array.to_list (Column.options Kind.int (column t n)))

let column_names t = List.map fst (Schema.columns (schema t))

(* Left rows 0 and 2 match right rows 1 and 2, left row 1 matches right row 0,
   and left row 3 and right row 3 match nothing. *)
let keyed_left = v [ ("k", i64 [| 2; 1; 2; 3 |]); ("x", i64 [| 0; 1; 2; 3 |]) ]

let keyed_right =
  v [ ("k", i64 [| 1; 2; 2; 4 |]); ("y", i64 [| 10; 11; 12; 13 |]) ]

let join_order =
  cases "keeps each kind's rows, the left's order with matches in the right's"
    ~id:(fun (name, _, _) -> name ^ ": left's order, matches in right's")
    [
      ( "Inner",
        Join.Inner,
        [
          ("k", [ s 2; s 2; s 1; s 2; s 2 ]);
          ("x", [ s 0; s 0; s 1; s 2; s 2 ]);
          ("y", [ s 11; s 12; s 10; s 11; s 12 ]);
        ] );
      ( "Left",
        Left,
        [
          ("k", [ s 2; s 2; s 1; s 2; s 2; s 3 ]);
          ("x", [ s 0; s 0; s 1; s 2; s 2; s 3 ]);
          ("y", [ s 11; s 12; s 10; s 11; s 12; None ]);
        ] );
      ( "Full",
        Full,
        [
          ("k", [ s 2; s 2; s 1; s 2; s 2; s 3; s 4 ]);
          ("x", [ s 0; s 0; s 1; s 2; s 2; s 3; None ]);
          ("y", [ s 11; s 12; s 10; s 11; s 12; None; s 13 ]);
        ] );
      ("Semi", Semi, [ ("k", [ s 2; s 1; s 2 ]); ("x", [ s 0; s 1; s 2 ]) ]);
      ("Anti", Anti, [ ("k", [ s 3 ]); ("x", [ s 3 ]) ]);
    ]
    (fun (_, kind, columns) ->
      let t = joined ~kind (Join.keys [ "k" ]) keyed_left keyed_right in
      equal ~msg:"columns" (list string) (List.map fst columns) (column_names t);
      List.iter (column_is t) columns)

(* Right keys that no left key has: a left of key 1 and a right of keys 1, 2 and
   2, whose repeated key 2 matches nothing; and an empty left, whose right rows
   all match nothing. *)
let unmatched_right =
  let one = v [ ("k", i64 [| 1 |]); ("x", i64 [| 0 |]) ]
  and none = v [ ("k", i64 [||]); ("x", i64 [||]) ] in
  let repeated = v [ ("k", i64 [| 1; 2; 2 |]); ("y", i64 [| 10; 11; 12 |]) ]
  and twice = v [ ("k", i64 [| 1; 1 |]); ("y", i64 [| 10; 11 |]) ] in
  let matched = [ ("k", [ s 1 ]); ("x", [ s 0 ]); ("y", [ s 10 ]) ] in
  let empty = [ ("k", []); ("x", []); ("y", []) ] in
  let kept = [ ("k", []); ("x", []) ] in
  cases "keeps right keys that no left key has as each kind states"
    ~id:(fun (name, _, _, _, _) -> name)
    [
      ( "Inner, a repeated key matching nothing",
        one,
        repeated,
        Join.Inner,
        matched );
      ("Left, a repeated key matching nothing", one, repeated, Left, matched);
      ( "Full, a repeated key matching nothing: both its rows follow",
        one,
        repeated,
        Full,
        [
          ("k", [ s 1; s 2; s 2 ]);
          ("x", [ s 0; None; None ]);
          ("y", [ s 10; s 11; s 12 ]);
        ] );
      ( "Semi, a repeated key matching nothing",
        one,
        repeated,
        Semi,
        [ ("k", [ s 1 ]); ("x", [ s 0 ]) ] );
      ("Anti, a repeated key matching nothing", one, repeated, Anti, kept);
      ("Inner, an empty left", none, twice, Inner, empty);
      ("Left, an empty left", none, twice, Left, empty);
      ( "Full, an empty left: every right row follows",
        none,
        twice,
        Full,
        [ ("k", [ s 1; s 1 ]); ("x", [ None; None ]); ("y", [ s 10; s 11 ]) ] );
      ("Semi, an empty left", none, twice, Semi, kept);
      ("Anti, an empty left", none, twice, Anti, kept);
    ]
    (fun (_, l, r, kind, columns) ->
      let t = joined ~kind (Join.keys [ "k" ]) l r in
      equal ~msg:"columns" (list string) (List.map fst columns) (column_names t);
      List.iter (column_is t) columns)

(* An equality join of an int64 key [k] of 0 to 3 on the left, and [rk] of 0 to
   6 on the right, so that right keys repeat, those past 3 matching nothing, and
   of up to 5 left rows, so that a sixth of the lefts are empty. Either side's
   key is null now and then. *)
let keyed_join =
  let open Gen in
  let key hi = option (int_range 0 hi) in
  let pp_keys ppf ks =
    Format.fprintf ppf "[%s]"
      (String.concat "; "
         (Array.to_list
            (Array.map (function None -> "-" | Some k -> string_of_int k) ks)))
  in
  let pp ppf (l, r, kind) =
    let name =
      match (kind : Join.kind) with
      | Inner -> "Inner"
      | Left -> "Left"
      | Full -> "Full"
      | Semi -> "Semi"
      | Anti -> "Anti"
    in
    Format.fprintf ppf "%s join of %a to %a" name pp_keys l pp_keys r
  in
  with_pp pp
    (triple
       (array ~size:(int_range 0 5) (key 3))
       (array ~size:(int_range 0 8) (key 6))
       (of_list Join.[ Inner; Left; Full; Semi; Anti ]))

let keyed_agrees (lk, rk, kind) =
  let table k v ks =
    let n = Array.length ks in
    R.Table
      (Talon.v
         [
           (k, Column.of_options Type.int64 ks); (v, i64 (Array.init n Fun.id));
         ])
  in
  let p =
    R.Join
      {
        kind;
        each_left = Any;
        each_right = Any;
        on = Keys [ ("k", "rk") ];
        left = table "k" "x" lk;
        right = table "rk" "y" rk;
      }
  in
  let r = List.hd (right_keys p) in
  (* Repeated matched keys place the matched right rows in runs, the others left
     out; distinct ones scatter them, where the rows of a repeated key that
     matches nothing write nothing. *)
  cover "a repeated matched key, and right rows matching nothing"
    (r.matched_keys < r.matched && r.rest > 0);
  cover "distinct matched keys, and a repeated key matching nothing"
    (r.matched_keys = r.matched && r.rest_keys < r.rest);
  cover "an empty left and a right row" (lk = [||] && rk <> [||]);
  match R.run p with
  | Error (row, why) -> failf "the reference fails at row %d: %s" row why
  | Ok cs -> List.iter (column_holds (run_ok (R.query p))) cs

(* Range joins

   A left of a key [k] and columns [a] and [b] of one type, a right of a key
   [kr] and columns [c] and [d] of a type that meets it, up to eight rows each.
   Their values come from one pool of a few, nulls included, so that the sides
   share values; a float's pool holds [-0.], [0.], NaN and an infinity too. A
   third of the time a side is in the order of its first column. The condition
   is up to three comparisons and periods between the sides, a third of the time
   with the keys. *)

type range =
  | Range : {
      ty : 'a Type.t;
      left : (int option * 'a option * 'a option) array;
      right : (int option * 'a option * 'a option) array;
      types : (string * 'a Type.t) list;
      plan : R.plan;
    }
      -> range

let specials : type a. a Type.t -> a list =
  let floats = [ -0.; 0.; Float.nan; Float.infinity ] in
  function
  | Type.Float16 -> floats
  | Type.Float32 -> floats
  | Type.Float64 -> floats
  | _ -> []

let pp_range ppf (Range { plan; _ }) =
  match plan with
  | R.Join { left = Table l; right = Table r; _ } ->
      Format.fprintf ppf "%a@.%a@.%a" pp_plan plan Talon.pp l Talon.pp r
  | _ -> pp_plan ppf plan

(* [range_case ty types preds left right] joins a left of rows [(k, a, b)] to a
   right of rows [(kr, c, d)], the columns of the types [types], on [preds], and
   on [k = kr] if [keyed]. *)
let range_case ~kind ~each_left ~each_right ~keyed ty types preds left right =
  let table (k, x, y) rs =
    let column ty f = Column.of_options ty (Array.map f rs) in
    R.Table
      (Talon.v
         [
           (k, column Type.int64 (fun (k, _, _) -> k));
           (x, column (List.assoc x types) (fun (_, x, _) -> x));
           (y, column (List.assoc y types) (fun (_, _, y) -> y));
         ])
  in
  let ks = if keyed then [ ("k", "kr") ] else [] in
  let plan =
    R.Join
      {
        kind;
        each_left;
        each_right;
        on = Range (ks, preds);
        left = table ("k", "a", "b") left;
        right = table ("kr", "c", "d") right;
      }
  in
  Range { ty; left; right; types; plan }

(* Joins that meet the range law's rarest covers: a left row with two matches, a
   band column bounded at two types that do not meet, and a categorical bounded
   by itself and by text. *)
let rare_ranges =
  let inner ty =
    range_case ~kind:Inner ~each_left:Any ~each_right:Any ~keyed:false ty
  in
  let le l r = R.Compare (`Le, l, r) in
  let cat = Type.categorical [| "b"; "a" |] in
  [
    inner Type.int64
      Type.[ ("a", int64); ("b", int64); ("c", int64); ("d", int64) ]
      [ R.Compare (`Lt, "a", "c") ]
      [| (None, s 0, s 0) |]
      [| (None, s 1, s 1); (None, s 2, s 2) |];
    inner Type.uint8
      Type.[ ("a", int16); ("b", uint16); ("c", uint8); ("d", uint8) ]
      [ le "a" "c"; R.Compare (`Gt, "b", "c") ]
      [| (None, s 2, s 9) |]
      [| (None, s 1, s 0); (None, s 5, s 0) |];
    inner cat
      [ ("a", cat); ("b", Type.string); ("c", cat); ("d", cat) ]
      [ le "a" "c"; le "b" "c" ]
      [| (None, s "b", s "a") |]
      [| (None, s "a", s "a") |];
  ]

let range_join =
  Gen.with_pp pp_range
  @@
  let open Gen in
  let orderable =
    List.map snd (expressible (List.map (fun t -> ("", t)) palette))
  in
  let* (Type.Any ty) = of_list orderable in
  (* Each column's type contains [ty]: one per side, or one per column. *)
  let* wide = of_list (wider ty) and+ flip = bool and+ apart = int_range 0 1 in
  let* own = list ~size:(constant 4) (of_list (wider ty)) in
  let types =
    let tl, tr = if flip then (wide, ty) else (ty, wide) in
    let drawn = List.combine [ "a"; "b"; "c"; "d" ] own in
    let meets (l, r) =
      Option.is_some (Type.common [ List.assoc l drawn; List.assoc r drawn ])
    in
    let pairs = [ ("a", "c"); ("a", "d"); ("b", "c"); ("b", "d") ] in
    if apart = 0 && List.exists meets pairs then drawn
    else [ ("a", tl); ("b", tl); ("c", tr); ("d", tr) ]
  in
  let meet ns =
    Option.is_some (Type.common (List.map (fun n -> List.assoc n types) ns))
  in
  let* pool = list ~size:(int_range 1 3) (value ty) in
  let pool = pool @ specials ty in
  let cell =
    frequency [ (1, constant None); (5, map Option.some (of_list pool)) ]
  in
  let rows =
    let* sorted = int_range 0 3 in
    let+ rs =
      array ~size:(int_range 0 8) (triple (option (int_range 0 2)) cell cell)
    in
    let nulls_last cmp x y =
      match (x, y) with
      | Some x, Some y -> cmp x y
      | None, None -> 0
      | None, _ -> 1
      | _, None -> -1
    in
    let first (_, x, _) (_, y, _) = nulls_last (Type.compare_value ty) x y in
    let keyed ((k, _, _) as a) ((k', _, _) as b) =
      match nulls_last Int.compare k k' with 0 -> first a b | c -> c
    in
    if sorted = 0 then Array.stable_sort first rs;
    if sorted = 1 then Array.stable_sort keyed rs;
    rs
  in
  let compare =
    let pairs =
      List.filter
        (fun (l, r) -> meet [ l; r ])
        [ ("a", "c"); ("a", "d"); ("b", "c"); ("b", "d") ]
    in
    let+ op = of_list [ `Lt; `Le; `Gt; `Ge ] and+ l, r = of_list pairs in
    R.Compare (op, l, r)
  in
  let period (s, e) = map (fun flip -> if flip then (e, s) else (s, e)) bool in
  let overlaps =
    let+ l = period ("a", "b") and+ r = period ("c", "d") in
    R.Overlaps (l, r)
  in
  let atom =
    if meet [ "a"; "b"; "c"; "d" ] then
      frequency [ (3, compare); (1, overlaps) ]
    else compare
  in
  let* preds = list ~size:(int_range 1 3) atom in
  let preds =
    List.fold_left
      (fun ps p -> if List.mem p ps then ps else ps @ [ p ])
      [] preds
  in
  let* keyed = int_range 0 2 in
  let* left = rows in
  let* right = rows in
  let count =
    frequency
      [
        (3, constant Join.Any);
        (1, of_list Join.[ At_most_one; One; At_least_one ]);
      ]
  in
  let+ kind = of_list Join.[ Inner; Left; Full; Semi; Anti ]
  and+ each_left = count
  and+ each_right = count in
  range_case ~kind ~each_left ~each_right ~keyed:(keyed = 0) ty types preds left
    right

let range_agrees (Range { ty; left; right; types; plan }) =
  let preds, keyed =
    match plan with
    | R.Join { on = Range (ks, ps); _ } -> (ps, ks <> [])
    | _ -> assert false
  in
  let compares =
    List.filter_map
      (function R.Compare (_, l, r) -> Some (l, r) | _ -> None)
      preds
  in
  let named side n =
    List.length (List.filter (fun lr -> String.equal (side lr) n) compares)
  in
  let most side ns = List.exists (fun n -> named side n >= 2) ns in
  let is_overlaps = function R.Overlaps _ -> true | R.Compare _ -> false in
  cover "one comparison"
    (match preds with [ R.Compare _ ] -> true | _ -> false);
  cover "two comparisons name one right column"
    (most snd [ "c"; "d" ] && not (most fst [ "a"; "b" ]));
  cover "two comparisons name one left column"
    (most fst [ "a"; "b" ] && not (most snd [ "c"; "d" ]));
  cover "comparisons name two columns of each side"
    (List.length (List.sort_uniq compare (List.map fst compares)) = 2
    && List.length (List.sort_uniq compare (List.map snd compares)) = 2);
  cover "an overlaps alone" (List.for_all is_overlaps preds);
  cover "an overlaps and another atom"
    (List.exists is_overlaps preds && List.length preds > 1);
  cover "equality keys" keyed;
  let cells rs =
    Array.to_list rs |> List.concat_map (fun (_, x, y) -> [ x; y ])
  in
  let all_cells = cells left @ cells right in
  let same x y = Type.compare_value ty x y = 0 in
  cover "a null bound" (List.mem None all_cells);
  cover "a float type" (specials ty <> []);
  cover "a NaN bound"
    (match specials ty with
    | [ _; _; nan; _ ] ->
        List.exists
          (function Some x -> Type.compare_value ty x nan = 0 | None -> false)
          all_cells
    | _ -> false);
  cover "a null equality key"
    (keyed
    && Array.exists (fun (k, _, _) -> k = None) left
    && Array.exists (fun (k, _, _) -> k = None) right);
  let row_is f rs =
    Array.exists
      (fun (_, x, y) ->
        match (x, y) with Some x, Some y -> f x y | _ -> false)
      rs
  in
  cover "a reversed period"
    (row_is (fun x y -> Type.compare_value ty x y > 0) left);
  cover "a zero-length period" (row_is same left || row_is same right);
  cover "a value on both sides"
    (List.exists
       (fun x ->
         List.exists
           (fun y ->
             match (x, y) with Some x, Some y -> same x y | _ -> false)
           (cells right))
       (cells left));
  let kind, each_left, each_right =
    match plan with
    | R.Join { kind; each_left; each_right; _ } -> (kind, each_left, each_right)
    | _ -> assert false
  in
  List.iter
    (fun (name, k) -> cover name (kind = k))
    Join.
      [
        ("Inner", Inner);
        ("Left", Left);
        ("Full", Full);
        ("Semi", Semi);
        ("Anti", Anti);
      ];
  cover "a count" (each_left <> Any || each_right <> Any);
  let typed n = List.assoc n types in
  let at_two =
    List.exists
      (fun x ->
        let partners =
          List.filter_map
            (fun (l, r) ->
              if l = x then Some r else if r = x then Some l else None)
            compares
        in
        List.exists
          (fun p ->
            List.exists (fun q -> not (Type.equal (typed p) (typed q))) partners)
          partners)
      [ "a"; "b"; "c"; "d" ]
  in
  cover "a column compared at two types" at_two;
  let text t = Type.equal t Type.string in
  let categorical t =
    String.starts_with ~prefix:"categorical" (Format.asprintf "%a" Type.pp t)
  in
  cover "a categorical compared with text"
    (List.exists
       (fun (l, r) ->
         (text (typed l) && categorical (typed r))
         || (categorical (typed l) && text (typed r)))
       compares);
  let grouped rs =
    let n = Array.length rs in
    let keyed i =
      let k, x, _ = rs.(i) and k', x', _ = rs.(i + 1) in
      match (k, k') with
      | Some k, Some k' when k <> k' -> k < k'
      | Some _, Some _ -> (
          match (x, x') with
          | Some x, Some x' -> Type.compare_value ty x x' <= 0
          | _, None -> true
          | None, Some _ -> false)
      | _, None -> true
      | None, Some _ -> false
    in
    n >= 3 && List.for_all keyed (List.init (n - 1) Fun.id)
  in
  cover "keys and a side in order of its key, then its first column"
    (keyed && (grouped left || grouped right));
  let reference = R.run plan in
  let matched =
    match reference with
    | Ok ((_, R.Column (_, vs)) :: _) -> Array.length vs
    | _ -> 0
  in
  cover "a left row with two matches or more"
    (kind = Inner && matched > Array.length left);
  cover "a count fails" (Result.is_error reference);
  match (reference, Query.run (R.query plan)) with
  | Ok cs, Ok t ->
      equal schema_w (Schema.v (R.schema plan)) (schema t);
      List.iter (column_holds t) cs
  | Error failure, Error e ->
      (* Talon states neither the order of a join's inputs nor of its sides'
         counts, so the failure of either order is allowed. *)
      one_of (Ok (Error failure)) (allowed (R.outcomes plan)) (Ok (Error e))
  | Ok _, Error e -> failf "the run fails: %s" (error e)
  | Error (row, why), Ok _ -> failf "no failure at row %d: %s" row why

(* Spikes in trials, written from either side: trials [[0, 10)], [[20, 30)] and
   the reversed [(40, 30)], spikes at 25, 5, 15, 28, 30 and 35. Each spike is in
   one trial at most, a reversed trial holds none ([inside] matches only rows
   whose stop is after their start), and the rows come in the left's order. *)
let spikes_in_trials =
  let trials =
    v
      [
        ("start", i64 [| 0; 20; 40 |]);
        ("stop", i64 [| 10; 30; 30 |]);
        ("trial", i64 [| 0; 1; 2 |]);
      ]
  in
  let spikes =
    v
      [
        ("t", i64 [| 25; 5; 15; 28; 30; 35 |]);
        ("spike", i64 [| 0; 1; 2; 3; 4; 5 |]);
      ]
  in
  cases "inside and comparisons find spikes in trials, from either side"
    ~id:(fun (name, _, _, _, _) -> name)
    [
      ( "trials on the left",
        trials,
        spikes,
        Join.(le "start" "t" && gt "stop" "t"),
        [ ("trial", [ s 0; s 1; s 1 ]); ("spike", [ s 1; s 0; s 3 ]) ] );
      ( "spikes on the left",
        spikes,
        trials,
        Join.inside "t" ("start", "stop"),
        [ ("spike", [ s 0; s 1; s 3 ]); ("trial", [ s 1; s 0; s 1 ]) ] );
    ]
    (fun (_, l, r, on, columns) ->
      List.iter (column_is (joined on l r)) columns)

(* SQL-92's <overlaps predicate> (ISO/IEC 9075:1992, 8.11), whose general rules
   read: "3) If D1 is the null value or if E1 < D1, then let S1 = E1 and let T1
   = D1. Otherwise, let S1 = D1 and let T1 = E1." (5 likewise for S2 and T2) "6)
   The result of the <overlaps predicate> is the result of the following
   expression: ( S1 > S2 AND NOT ( S1 >= T2 AND T1 >= T2 ) ) OR ( S2 > S1 AND
   NOT ( S2 >= T1 AND T2 >= T1 ) ) OR ( S1 = S2 AND ( T1 <> T2 OR T1 = T2 ) )"
   Each row is two periods and that expression's value, a pair matching where it
   is true. The expression is symmetric, so each row holds both ways. *)
let sql92_overlaps =
  let i (d, e) =
    (Column.of_options Type.int64 [| d |], Column.of_options Type.int64 [| e |])
  in
  let f (d, e) =
    ( Column.of_options Type.float64 [| d |],
      Column.of_options Type.float64 [| e |] )
  in
  let ints =
    [
      ("disjoint", (s 0, s 2), (s 3, s 5), false);
      ( "touching: one's stop is the other's start",
        (s 0, s 2),
        (s 2, s 3),
        false );
      ("one's start inside the other", (s 0, s 5), (s 3, s 8), true);
      ("one inside the other", (s 0, s 10), (s 3, s 5), true);
      ("equal periods", (s 2, s 6), (s 2, s 6), true);
      ("equal starts, other stops", (s 2, s 3), (s 2, s 9), true);
      ("a point inside the other", (s 5, s 5), (s 3, s 8), true);
      ("a point at the other's start", (s 5, s 5), (s 5, s 9), true);
      ("a point at the other's stop", (s 5, s 5), (s 1, s 5), false);
      ("one point twice", (s 4, s 4), (s 4, s 4), true);
      ("two points", (s 4, s 4), (s 5, s 5), false);
      ("a reversed period, overlapping", (s 8, s 3), (s 0, s 5), true);
      ("a reversed period, touching", (s 2, s 0), (s 2, s 3), false);
      ("a reversed period at the other's stop", (s 5, s 3), (s 0, s 3), false);
      ("a null start, its stop inside the other", (None, s 4), (s 3, s 8), true);
      ("a null stop, its start inside the other", (s 4, None), (s 3, s 8), true);
      ( "a null stop, the start before the other's",
        (s 2, None),
        (s 3, s 8),
        false );
      ("a null stop, equal starts", (s 3, None), (s 3, s 8), false);
      ( "a null stop, the start at the other's stop",
        (s 5, None),
        (s 1, s 5),
        false );
      ("null stops on each side, equal starts", (s 3, None), (s 3, None), false);
      ("a null stop on each side", (s 3, None), (s 4, None), false);
      ("both null", (None, None), (s 0, s 9), false);
    ]
  in
  let floats =
    [
      ( "a NaN stop holds every number after its start",
        (s 0., s Float.nan),
        (s 5., s 6.),
        true );
      ("a NaN start is a stop", (s Float.nan, s 0.), (s 5., s 6.), true);
      ("-0. and 0. are one start", (s (-0.), s 1.), (s 0., s 2.), true);
      ( "an infinite stop before NaN",
        (s 0., s Float.infinity),
        (s Float.nan, s Float.nan),
        false );
    ]
  in
  cases "overlaps is SQL-92's OVERLAPS, over a pair of one-row tables"
    ~id:(fun (name, _, _, _) -> name)
    (List.map (fun (n, l, r, e) -> (n, i l, i r, e)) ints
    @ List.map (fun (n, l, r, e) -> ("float64: " ^ n, f l, f r, e)) floats)
    (fun (_, (ld, le), (rd, re), expected) ->
      let l = v [ ("d1", ld); ("e1", le) ]
      and r = v [ ("d2", rd); ("e2", re) ] in
      let matches l r =
        rows (joined (Join.overlaps ("d1", "e1") ("d2", "e2")) l r) = 1
      in
      let swap t = v [ ("d1", column t "d2"); ("e1", column t "e2") ] in
      let back t = v [ ("d2", column t "d1"); ("e2", column t "e1") ] in
      equal ~msg:"left, right" bool expected (matches l r);
      equal ~msg:"right, left" bool expected (matches (swap r) (back l)))

(* A right whose keys are distinct, as a table looked up by its key: left rows 0
   and 2 match right row 1, left row 1 matches right row 0, and left row 3 and
   right row 2 match nothing. *)
let distinct_right = v [ ("k", i64 [| 1; 2; 4 |]); ("y", i64 [| 10; 11; 13 |]) ]

(* The same lookup with a null key on each side, which match each other: left
   row 4 matches right row 3. Its keys are int64 or text. *)
let lookup_order =
  let ks = [| s 2; s 1; s 2; s 3; None |] and rks = [| s 1; s 2; s 4; None |] in
  (* The keys as int64, or as the text of their digits, read back as ints. *)
  let key_column dtype ks =
    match dtype with
    | `Int64 -> Column.of_options Type.int64 ks
    | `Text ->
        Column.of_options Type.string (Array.map (Option.map string_of_int) ks)
  in
  let keys dtype t =
    match dtype with
    | `Int64 -> Array.to_list (Column.options Kind.int (column t "k"))
    | `Text ->
        let ks = Column.options Kind.string (column t "k") in
        Array.to_list (Array.map (Option.map int_of_string) ks)
  in
  let kinds =
    [
      ( "Inner",
        Join.Inner,
        [ s 2; s 1; s 2; None ],
        [ ("x", [ s 0; s 1; s 2; s 4 ]); ("y", [ s 11; s 10; s 11; s 14 ]) ] );
      ( "Left",
        Left,
        [ s 2; s 1; s 2; s 3; None ],
        [
          ("x", [ s 0; s 1; s 2; s 3; s 4 ]);
          ("y", [ s 11; s 10; s 11; None; s 14 ]);
        ] );
      ( "Full",
        Full,
        [ s 2; s 1; s 2; s 3; None; s 4 ],
        [
          ("x", [ s 0; s 1; s 2; s 3; s 4; None ]);
          ("y", [ s 11; s 10; s 11; None; s 14; s 13 ]);
        ] );
      ("Semi", Semi, [ s 2; s 1; s 2; None ], [ ("x", [ s 0; s 1; s 2; s 4 ]) ]);
      ("Anti", Anti, [ s 3 ], [ ("x", [ s 3 ]) ]);
    ]
  in
  cases "keeps each kind's rows over distinct right keys"
    ~id:(fun (dtype, (name, _, _, _)) ->
      Printf.sprintf "%s over %s keys: each left row with its one match" name
        (match dtype with `Int64 -> "int64" | `Text -> "text"))
    (List.concat_map
       (fun d -> List.map (fun k -> (d, k)) kinds)
       [ `Int64; `Text ])
    (fun (dtype, (_, kind, expected_keys, columns)) ->
      let l = v [ ("k", key_column dtype ks); ("x", i64 [| 0; 1; 2; 3; 4 |]) ]
      and r =
        v [ ("k", key_column dtype rks); ("y", i64 [| 10; 11; 13; 14 |]) ]
      in
      let t = joined ~kind (Join.keys [ "k" ]) l r in
      equal ~msg:"columns" (list string)
        ("k" :: List.map fst columns)
        (column_names t);
      equal ~msg:"k" (list (option int)) expected_keys (keys dtype t);
      List.iter (column_is t) columns)

(* Text keys without a null are numbered as their rows of bytes, in one
   numbering of both sides: a full join finds the right rows without a match
   from the left rows matched to each. *)
let text_lookup =
  test
    "Full over distinct text keys without a null appends unmatched right rows"
    (fun () ->
      let text ks = Column.v Type.string ks in
      let l =
        v [ ("k", text [| "2"; "1"; "2"; "3" |]); ("x", i64 [| 0; 1; 2; 3 |]) ]
      and r =
        v [ ("k", text [| "1"; "2"; "4" |]); ("y", i64 [| 10; 11; 13 |]) ]
      in
      let t = joined ~kind:Full (Join.keys [ "k" ]) l r in
      equal ~msg:"k"
        (list (option string))
        [ s "2"; s "1"; s "2"; s "3"; s "4" ]
        (Array.to_list (Column.options Kind.string (column t "k")));
      List.iter (column_is t)
        [
          ("x", [ s 0; s 1; s 2; s 3; None ]);
          ("y", [ s 11; s 10; s 11; None; s 13 ]);
        ])

(* [shapes] are a left of three rows and a right of two, and the other way. *)
let shapes =
  let l3 = v [ ("x", i64 [| 0; 1; 2 |]) ]
  and r2 = v [ ("y", i64 [| 10; 11 |]) ] in
  let l2 = v [ ("x", i64 [| 0; 1 |]) ]
  and r3 = v [ ("y", i64 [| 10; 11; 12 |]) ] in
  let l0 = v [ ("x", i64 [||]) ] and r0 = v [ ("y", i64 [||]) ] in
  cases "position and all keep the rows each kind states"
    ~id:(fun (name, _, _, _, _, _) -> name)
    [
      ( "position Inner keeps min(n, m) rows",
        Join.position,
        Join.Inner,
        l3,
        r2,
        [ ("x", [ s 0; s 1 ]); ("y", [ s 10; s 11 ]) ] );
      ( "position Semi keeps min(n, m) rows",
        Join.position,
        Semi,
        l3,
        r2,
        [ ("x", [ s 0; s 1 ]) ] );
      ( "position Left keeps n rows",
        Join.position,
        Left,
        l3,
        r2,
        [ ("x", [ s 0; s 1; s 2 ]); ("y", [ s 10; s 11; None ]) ] );
      ( "position Full keeps max(n, m) rows",
        Join.position,
        Full,
        l2,
        r3,
        [ ("x", [ s 0; s 1; None ]); ("y", [ s 10; s 11; s 12 ]) ] );
      ( "position Anti keeps the left's rows past m",
        Join.position,
        Anti,
        l3,
        r2,
        [ ("x", [ s 2 ]) ] );
      ( "position Anti keeps nothing when m >= n",
        Join.position,
        Anti,
        l2,
        r3,
        [ ("x", []) ] );
      ( "all pairs each left row with every right row",
        Join.all,
        Inner,
        l2,
        r2,
        [ ("x", [ s 0; s 0; s 1; s 1 ]); ("y", [ s 10; s 11; s 10; s 11 ]) ] );
      ( "all Left keeps the left over no right row",
        Join.all,
        Left,
        l2,
        r0,
        [ ("x", [ s 0; s 1 ]); ("y", [ None; None ]) ] );
      ( "all Full keeps the right under no left row",
        Join.all,
        Full,
        l0,
        r2,
        [ ("x", [ None; None ]); ("y", [ s 10; s 11 ]) ] );
      ( "all Inner over no right row is empty",
        Join.all,
        Inner,
        l2,
        r0,
        [ ("x", []); ("y", []) ] );
      ( "all Semi keeps every left row over a right row",
        Join.all,
        Semi,
        l2,
        r2,
        [ ("x", [ s 0; s 1 ]) ] );
      ( "all Anti keeps every left row over no right row",
        Join.all,
        Anti,
        l2,
        r0,
        [ ("x", [ s 0; s 1 ]) ] );
    ]
    (fun (_, on, kind, l, r, columns) ->
      let t = joined ~kind on l r in
      equal ~msg:"columns" (list string) (List.map fst columns) (column_names t);
      List.iter (column_is t) columns)

let joins =
  [
    join_order;
    unmatched_right;
    (* A repeated matched key beside right rows matching nothing, and an empty
       left. *)
    prop
      ~examples:
        [
          ([| Some 1 |], [| Some 1; Some 1; Some 5 |], Join.Inner);
          ([||], [| Some 1 |], Join.Inner);
        ]
      "on equality gives the reference's rows" keyed_join keyed_agrees;
    prop ~count:1000 ~examples:rare_ranges
      "on comparisons keeps the pairs its condition holds on" range_join
      range_agrees;
    spikes_in_trials;
    test "bounds a categorical by itself and by text, each at its own type"
      (fun () ->
        (* "b" comes before "a" in the dictionary and after it as text: each
           bound compares at the type of its own two columns. *)
        let d = Type.categorical [| "b"; "a" |] in
        let l =
          v
            [
              ("y1", Column.v d [| "b" |]);
              ("y2", Column.v Type.string [| "a" |]);
            ]
        and r = v [ ("x", Column.v d [| "a" |]) ] in
        equal int 1 (rows (joined Join.(le "y1" "x" && le "y2" "x") l r)));
    test "bounds a band column at two types that do not meet" (fun () ->
        (* uint8 meets int16 and uint16, which do not meet each other. *)
        let l =
          v
            [
              ("lo", Column.v Type.int16 [| 2; -1 |]);
              ("hi", Column.v Type.uint16 [| 9; 2 |]);
            ]
        and r =
          v
            [
              ("t", Column.v Type.uint8 [| 1; 5; 9 |]); ("y", i64 [| 0; 1; 2 |]);
            ]
        in
        column_is
          (joined Join.(le "lo" "t" && gt "hi" "t") l r)
          ("y", [ Some 1; Some 0 ]));
    sql92_overlaps;
    lookup_order;
    text_lookup;
    shapes;
    test "on all && c is the join on c" (fun () ->
        let k = Join.keys [ "k" ] in
        let table = Testable.make ~pp:Talon.pp ~equal:Talon.equal in
        equal table
          (joined k keyed_left keyed_right)
          (joined Join.(all && k) keyed_left keyed_right));
    test "null keys, NaNs and zeros each match as one key" (fun () ->
        let payload = Int64.float_of_bits 0xfff8000000000001L in
        let f =
          Column.of_options Type.float64
            [| Some Float.nan; Some (-0.); None; Some 1. |]
        and g =
          Column.of_options Type.float64
            [| Some 0.; None; Some payload; Some (-1.) |]
        in
        let t =
          joined (Join.eq "f" "g")
            (v [ ("f", f); ("x", i64 [| 0; 1; 2; 3 |]) ])
            (v [ ("g", g); ("y", i64 [| 10; 11; 12; 13 |]) ])
        in
        List.iter (column_is t)
          [
            ("x", [ Some 0; Some 1; Some 2 ]);
            ("y", [ Some 12; Some 10; Some 11 ]);
          ]);
    test "text keys match by their bytes, a categorical as its text" (fun () ->
        let dict = Type.categorical [| "é"; "a" |] in
        let t =
          joined (Join.eq "s" "c")
            (v
               [
                 ("s", Column.v Type.string [| "é"; "e"; "a" |]);
                 ("x", i64 [| 0; 1; 2 |]);
               ])
            (v [ ("c", Column.v dict [| "a"; "é" |]); ("y", i64 [| 10; 11 |]) ])
        in
        List.iter (column_is t)
          [ ("x", [ Some 0; Some 2 ]); ("y", [ Some 11; Some 10 ]) ]);
    test "Full gives the key the common type and an unmatched right row's value"
      (fun () ->
        let t =
          joined ~kind:Full (Join.keys [ "k" ])
            (v [ ("k", Column.v Type.int8 [| 1; 2 |]); ("x", i64 [| 0; 1 |]) ])
            (v
               [
                 ("k", Column.v Type.int16 [| 2; 300 |]); ("y", i64 [| 10; 11 |]);
               ])
        in
        equal schema_w
          (Schema.v
             Type.[ ("k", Any int16); ("x", Any int64); ("y", Any int64) ])
          (schema t);
        List.iter (column_is t)
          [
            ("k", [ Some 1; Some 2; Some 300 ]);
            ("x", [ Some 0; Some 1; None ]);
            ("y", [ None; Some 10; Some 11 ]);
          ]);
    test "fails a count, naming the side, the row's keys and its matches"
      (fun () ->
        let ends ?each_left ?each_right on l r =
          match Query.run (join ?each_left ?each_right on l r) with
          | Ok _ -> "no failure"
          | Error e -> error e
        in
        let l = keyed_left and r = keyed_right in
        let k = Join.keys [ "k" ] in
        let texts =
          v
            [
              ("s", Column.of_options Type.string [| Some "a"; None |]);
              ("b", Column.v Type.bool [| true; true |]);
            ]
        in
        expect
          (String.concat "\n"
             [
               ends ~each_left:One k l r;
               ends ~each_left:At_most_one k l r;
               ends ~each_left:At_least_one k l r;
               ends ~each_right:One k l r;
               ends ~each_right:At_least_one k l r;
               ends ~each_left:One ~each_right:One k l r;
               ends ~each_left:One k l distinct_right;
               ends ~each_right:One k l distinct_right;
               ends ~each_right:At_least_one k l distinct_right;
               ends ~each_left:One Join.position l (v [ ("y", i64 [| 1 |]) ]);
               ends ~each_right:At_most_one Join.all l
                 (v [ ("y", i64 [| 1 |]) ]);
               ends ~each_left:At_most_one
                 (Join.keys [ "s"; "b" ])
                 texts
                 (v
                    [
                      ("s", Column.of_options Type.string [| None; None |]);
                      ("b", Column.v Type.bool [| true; true |]);
                    ]);
             ])
        @@ __POS_OF__
             {|
            join ~on:(keys ["k"]) ~each_left:One: row 0: the left row whose "k" is 2 matches 2 rows, not one.
            join ~on:(keys ["k"]) ~each_left:At_most_one: row 0: the left row whose "k" is 2 matches 2 rows, not at most one.
            join ~on:(keys ["k"]) ~each_left:At_least_one: row 3: the left row whose "k" is 3 matches 0 rows, not at least one.
            join ~on:(keys ["k"]) ~each_right:One: row 1: the right row whose "k" is 2 matches 2 rows, not one.
            join ~on:(keys ["k"]) ~each_right:At_least_one: row 3: the right row whose "k" is 4 matches 0 rows, not at least one.
            join ~on:(keys ["k"]) ~each_left:One ~each_right:One: row 0: the left row whose "k" is 2 matches 2 rows, not one.
            join ~on:(keys ["k"]) ~each_left:One: row 3: the left row whose "k" is 3 matches 0 rows, not one.
            join ~on:(keys ["k"]) ~each_right:One: row 1: the right row whose "k" is 2 matches 2 rows, not one.
            join ~on:(keys ["k"]) ~each_right:At_least_one: row 2: the right row whose "k" is 4 matches 0 rows, not at least one.
            join ~on:position ~each_left:One: row 1: the left row matches 0 rows, not one.
            join ~on:all ~each_right:At_most_one: row 0: the right row matches 4 rows, not at most one.
            join ~on:(keys ["s"; "b"]) ~each_left:At_most_one: row 1: the left row whose "s" is ∅ and "b" is true matches 2 rows, not at most one.
            |});
  ]

let slice_after_failing_join =
  test "keeps no row of a failing join under a slice from the end" (fun () ->
      (* The join fails at its left's row 0, but no slice reads a row. *)
      let t = v [ ("x", i64 [| 0; 1 |]) ] in
      let right =
        R.Select
          ( [ R.Out ("y", R.Col (Type.int64, "x")) ],
            R.Slice { offset = 0; length = 0; plan = R.Table t } )
      in
      let failing =
        R.Join
          {
            kind = Inner;
            each_left = One;
            each_right = Any;
            on = Position;
            left = R.Table t;
            right;
          }
      in
      let p =
        R.Slice
          {
            offset = 0;
            length = 0;
            plan = R.Slice { offset = -1; length = 0; plan = failing };
          }
      in
      equal int 0 (rows (run_ok (R.query p))))

let join_streams =
  test "on all streams its left, batch by batch" (fun () ->
      let l =
        of_batches [ v [ ("x", i64 [| 0; 1 |]) ]; v [ ("x", i64 [| 2 |]) ] ]
      and r = v [ ("y", i64 [| 10; 11 |]) ] in
      let seen =
        Query.fold (join Join.all l r) ~init:[] (fun bs b -> rows b :: bs)
      in
      equal (list int) [ 4; 2 ] (List.rev (require_ok ~pp:Error.pp seen)))

let utc_wall_clock =
  test "wall in UTC reads an instant's calendar" (fun () ->
      let t =
        v
          [
            ( "t",
              Column.v (Type.instant Us)
                [| Time.of_ns 1_710_496_800_000_000_000L |] );
          ]
      in
      let field f =
        Query.values
          Expr.(Temporal.field f (Temporal.wall Tz.utc (Col.instant "t")))
          (Query.of_table t)
      in
      equal (array int) [| 10 |] (require_ok ~pp:Error.pp (field `Hour));
      equal (array int) [| 15 |] (require_ok ~pp:Error.pp (field `Day)))

(* Appending *)

(* The pairs of palette types that meet, among those a handle reads, with the
   type where they meet: a column of either type reads its values at that type's
   kind. *)
let meeting_pairs =
  let common (Type.Any t0) (Type.Any t1) =
    match Kind.provably_equal (Type.kind t0) (Type.kind t1) with
    | Some Equal -> Option.map (fun c -> Type.Any c) (Type.common [ t0; t1 ])
    | None -> None
  in
  let readable = expressible (List.map (fun t -> ("", t)) palette) in
  List.concat_map
    (fun (_, a0) ->
      List.filter_map
        (fun (_, a1) -> Option.map (fun c -> (a0, a1, c)) (common a0 a1))
        readable)
    readable

let appended =
  let side a = Gen.bind (table [ ("x", a) ]) G.split in
  Gen.with_pp
    (fun ppf (Type.Any t0, Type.Any t1, _, a, b) ->
      Format.fprintf ppf "@[<v>%a ⊕ %a@,%a@,%a@]" Type.pp t0 Type.pp t1 pp a pp
        b)
    (Gen.bind (Gen.of_list meeting_pairs) (fun (a0, a1, c) ->
         map2 (fun a b -> (a0, a1, c, a, b)) (side a0) (side a1)))

(* A source of a file: [file t] reads [t]'s rows as one part and logs the
   requests talon makes, the parts it asks for and the readers it opens. It
   answers [pushdown] to every conjunct, [Inexact] by default, and applies those
   of its request. *)
type file_log = {
  mutable parts : int;
  mutable opens : int;
  mutable filters : Source.Pred.t list;
}

let file ?(pushdown = fun _ -> Source.Inexact) t =
  let log = { parts = 0; opens = 0; filters = [] } in
  let part (r : Source.request) =
    let open_ () =
      log.opens <- log.opens + 1;
      let rest = ref [ answer r t ] in
      let next () =
        match !rest with
        | [] -> Ok None
        | b :: bs ->
            rest := bs;
            Ok (Some b)
      in
      Ok { Source.next; close = ignore }
    in
    { Source.rows = None; open_ }
  in
  let parts (r : Source.request) =
    log.parts <- log.parts + 1;
    log.filters <- r.filters;
    Ok [ part r ]
  in
  (Source.v ~name:"file" ~schema:(schema t) ~pushdown parts, log)

(* Filters over constant columns

   A branch holds a file's columns [x] and [s] and three constant columns,
   [year], [region] and [score], some of them null, as a fold of files derives
   them. A filter reads the constants, the file's columns or both, and the
   reference runs the plan row by row. A conjunct that reads only constants and
   is not [true] on a branch's constants must exclude the branch: its source is
   never asked for its parts or opened. The filters are those the optimizer
   folds over literals: comparisons with literals, [is_in], [is_null], [not],
   [&&] and [||]. *)

type constant_branch = {
  year : int option;
  region : string option;
  score : float option;
  x : int option array;
  s : string option array;
}

let constant_branch_gen =
  let open Gen in
  let+ year = of_list [ Some 2000; Some 2001; Some 2002; None ]
  and+ region = of_list [ Some "eu"; Some "us"; None ]
  and+ score =
    of_list [ Some 0.; Some (-0.); Some Float.nan; Some infinity; None ]
  and+ n = int_range 0 4
  and+ x =
    array ~size:(int_range 4 4) (of_list [ Some 0; Some 3; Some 7; None ])
  and+ s = array ~size:(int_range 4 4) (of_list [ Some "a"; Some "b"; None ]) in
  { year; region; score; x = Array.sub x 0 n; s = Array.sub s 0 n }

let constant_pred_gen =
  let open Gen in
  let int = R.Col (Type.int64, "year") and x = R.Col (Type.int64, "x") in
  let text n = R.Col (Type.string, n)
  and score = R.Col (Type.float64, "score") in
  let ops = of_list [ `Eq; `Ne; `Lt; `Le; `Gt; `Ge ] in
  let compare a ty vs =
    map2 (fun op v -> R.Cmp (op, a, R.Lit (ty, v))) ops (of_list vs)
  in
  let atom =
    frequency
      [
        (3, compare int Type.int64 [ 2000; 2001; 2002; 2003 ]);
        (2, compare (text "region") Type.string [ "eu"; "us"; "zz" ]);
        ( 1,
          map
            (fun vs -> R.Is_in (vs, int))
            (list ~size:(int_range 0 3) (int_range 1999 2003)) );
        (1, of_list [ R.Is_null int; R.Is_null (text "region"); R.Is_null x ]);
        (1, compare score Type.float64 [ 0.; -0.; Float.nan; infinity; 1. ]);
        (2, compare x Type.int64 [ 0; 3; 7 ]);
        ( 1,
          map
            (fun v -> R.Cmp (`Eq, text "s", R.Lit (Type.string, v)))
            (of_list [ "a"; "z" ]) );
      ]
  in
  let rec pred depth =
    if depth = 0 then atom
    else
      let p = pred (depth - 1) in
      frequency
        [
          (4, atom);
          (1, map (fun a -> R.Not a) p);
          (2, map2 (fun a b -> R.And (a, b)) p p);
          (2, map2 (fun a b -> R.Or (a, b)) p p);
        ]
  in
  pred 3

(* [columns_of p] is the columns the drawn filter [p] reads. *)
let rec columns_of : type a s. (a, s) R.term -> string list = function
  | Col (_, n) -> [ n ]
  | Cmp (_, a, b) -> columns_of a @ columns_of b
  | And (a, b) | Or (a, b) -> columns_of a @ columns_of b
  | Not a -> columns_of a
  | Is_in (_, a) -> columns_of a
  | Is_null a -> columns_of a
  | _ -> []

let rec conjuncts : bool R.expr -> bool R.expr list = function
  | And (a, b) -> conjuncts a @ conjuncts b
  | p -> [ p ]

let constant_only p =
  List.for_all
    (fun c -> List.mem c [ "year"; "region"; "score" ])
    (columns_of p)

let constant_branch_table b =
  v
    [
      ("x", Column.of_options Type.int64 b.x);
      ("s", Column.of_options Type.string b.s);
    ]

let derive_constants b plan =
  let constant ty = function
    | Some v -> R.Store (ty, R.Lit (ty, v))
    | None -> R.Store (ty, R.Null ty)
  in
  R.Derive
    ( [
        R.Out ("year", constant Type.int64 b.year);
        R.Out ("region", constant Type.string b.region);
        R.Out ("score", constant Type.float64 b.score);
      ],
      plan )

(* [truth b p] is the reference's value of the constant-only [p] on [b]'s
   constants. *)
let truth b p =
  let none ty = Column.of_options ty [| None |] in
  let row = R.Table (v [ ("x", none Type.int64); ("s", none Type.string) ]) in
  match R.run (R.Select ([ R.Out ("p", p) ], derive_constants b row)) with
  | Ok [ (_, R.Column (ty, vs)) ] -> (
      match Kind.provably_equal (Type.kind ty) Kind.bool with
      | Some Equal -> (vs.(0) : bool option)
      | None -> None)
  | _ -> None

let filters_over_constants =
  let gen =
    Gen.pair
      (Gen.list ~size:(Gen.int_range 1 6) constant_branch_gen)
      constant_pred_gen
  in
  prop ~count:500 "a filter over constant columns gives the reference's rows"
    (Gen.with_pp
       (fun ppf (bs, p) ->
         let pp_b ppf b =
           Format.fprintf ppf "(%s, %s, %d rows)"
             (match b.year with Some y -> string_of_int y | None -> "null")
             (Option.value ~default:"null" b.region)
             (Array.length b.x)
         in
         Format.fprintf ppf "%a@.%a"
           (Format.pp_print_list pp_b)
           bs Expr.pp (R.expr p))
       gen)
    (fun (branches, p) ->
      let sources =
        List.map
          (fun b ->
            let t = constant_branch_table b in
            let s, log = file t in
            (b, t, s, log))
          branches
      in
      let plan_of (b, t, s, _) = derive_constants b (R.Source (s, t)) in
      let appended =
        match sources with
        | [] -> assert false
        | b :: bs ->
            List.fold_left (fun q b -> R.Append (q, plan_of b)) (plan_of b) bs
      in
      let plan = R.Filter (p, appended) in
      let constants = List.filter constant_only (conjuncts p) in
      let excluded b =
        List.exists (fun c -> truth b c <> Some true) constants
      in
      cover "a branch is excluded by a constant conjunct"
        (List.exists excluded branches);
      cover "a branch passes the constant conjuncts"
        (List.exists (fun b -> not (excluded b)) branches);
      cover "a null constant is excluded by a comparison"
        (List.exists (fun b -> Option.is_none b.year && excluded b) branches);
      cover "a filter mixes constant and file columns"
        (constants <> [] && not (constant_only p));
      (match (R.run plan, Query.run (R.query plan)) with
      | Ok cs, Ok t -> List.iter (column_holds t) cs
      | Error (row, why), Error e ->
          contains ~sub:(Printf.sprintf ": row %d: %s." row why) (error e)
      | Ok _, Error e -> failf "the run fails: %s" (error e)
      | Error (row, why), Ok _ -> failf "no failure at row %d: %s" row why);
      List.iter
        (fun (b, _, _, log) ->
          if excluded b then begin
            equal ~msg:"parts of an excluded branch" int 0 log.parts;
            equal ~msg:"opens of an excluded branch" int 0 log.opens
          end)
        sources)

let extension_cases =
  let t_ = Ext.col celsius "t" and year = Col.int "year" in
  cases "filters extension columns beside constants as their table"
    ~id:(fun (name, _) -> name)
    [
      ("constant or extension", Expr.(year = int 2001 || t_ > const 20.));
      ("extension only", Expr.(t_ > const 20.));
      ("not extension", Expr.(not (t_ > const 20.)));
      ("constant and extension", Expr.(year = int 2002 && t_ > const 20.));
      ("null extension", Expr.(is_null t_ || year = int 2001));
    ]
    (fun (_, p) ->
      let a = degrees [| Some 30.; Some 10.; None |]
      and b = degrees [| Some 5.; Some 40.; None |] in
      (* The source applies the conjuncts that [Pred.filter] builds, so an
         extension's conjunct is never [Exact]. *)
      let pushdown t p =
        if Option.is_some (Source.Pred.filter (schema t) p) then Source.Exact
        else Source.Unsupported
      in
      let branch y t =
        let s, _ = file ~pushdown:(pushdown t) t in
        Query.of_source s |> Query.derive Expr.[ "year" := int y ]
      in
      let q = branch 2001 a |> Query.append (branch 2002 b) in
      let materialized y t =
        run_ok (Query.derive Expr.[ "year" := int y ] (Query.of_table t))
      in
      let reference = of_batches [ materialized 2001 a; materialized 2002 b ] in
      equal
        (Testable.make ~pp:Talon.pp ~equal:Talon.equal)
        (run_ok (Query.filter p (Query.of_table reference)))
        (run_ok (Query.filter p q)))

(* Kit.union_all *)

type drifted = { wide : bool; y : float option array option; x_ : int array }

let drifted_gen =
  Gen.(
    let+ wide = bool
    and+ n = int_range 0 3
    and+ xs = array ~size:(constant 3) (int_range (-4) 9)
    and+ y =
      frequency
        [
          (1, constant None);
          ( 2,
            map Option.some
              (array ~size:(constant 3)
                 (of_list [ Some 1.5; Some (-2.); None ])) );
        ]
    in
    { wide; y = Option.map (fun a -> Array.sub a 0 n) y; x_ = Array.sub xs 0 n })

let drifted_table d =
  let x =
    if d.wide then Column.v Type.int64 d.x_ else Column.v Type.int32 d.x_
  in
  let y =
    match d.y with
    | Some ys -> [ ("y", Column.of_options Type.float64 ys) ]
    | None -> []
  in
  v (("x", x) :: y)

let union_all_rows =
  prop "gives every branch's rows at the common types"
    (Gen.list ~size:(Gen.int_range 1 7) drifted_gen)
    (fun ds ->
      let qs = List.map (fun d -> Query.of_table (drifted_table d)) ds in
      let wide = List.exists (fun d -> d.wide) ds in
      cover "branches of two types"
        (wide && List.exists (fun d -> not d.wide) ds);
      cover "a branch lacks a column" (List.exists (fun d -> d.y = None) ds);
      let xs = Array.concat (List.map (fun d -> d.x_) ds) in
      let ys =
        Array.concat
          (List.map
             (fun d ->
               match d.y with
               | Some ys -> ys
               | None -> Array.make (Array.length d.x_) None)
             ds)
      in
      let has_y = List.exists (fun d -> Option.is_some d.y) ds in
      let x = if wide then Column.v Type.int64 xs else Column.v Type.int32 xs in
      let expected =
        v
          (("x", x)
          ::
          (if has_y then [ ("y", Column.of_options Type.float64 ys) ] else []))
      in
      equal
        (Testable.make ~pp:Talon.pp ~equal:Talon.equal)
        expected
        (run_ok (Kit.union_all qs)))

let conjunct_reach_gen =
  Gen.(
    let+ n = int_range 3 7 and+ at = int_range 0 6 in
    (n, min at (n - 1)))

let conjunct_reaches_every_stored_branch =
  prop "a filter reaches every branch stored at the common type"
    conjunct_reach_gen (fun (n, at) ->
      let table i =
        if i = at then v [ ("x", Column.v Type.int32 [| 1; 9 |]) ]
        else v [ ("x", Column.v Type.int64 [| 1; 9 |]) ]
      in
      let branches = List.init n (fun i -> file (table i)) in
      let q =
        Kit.union_all (List.map (fun (s, _) -> Query.of_source s) branches)
        |> Query.filter Expr.(Col.int "x" > int 5)
      in
      ignore (run_ok q);
      equal (list int)
        (List.init n (fun i -> if i = at then 0 else 1))
        (List.map (fun (_, log) -> List.length log.filters) branches))

let types_do_not_depend_on_order =
  let tables =
    [
      v [ ("x", Column.v Type.int8 [| 1 |]) ];
      v [ ("x", Column.v Type.uint8 [| 2 |]) ];
      v [ ("x", Column.v Type.int16 [| 3 |]) ];
    ]
  in
  prop "gives types that do not depend on the branches' order"
    (Gen.permutation tables) (fun ts ->
      let q = Kit.union_all (List.map Query.of_table ts) in
      equal schema_w (Schema.v [ ("x", Any Type.int16) ]) (Query.schema q))

let union_all_of_nothing () =
  let q = Kit.union_all [] in
  equal schema_w (Schema.v []) (Query.schema q);
  equal int 0 (rows (run_ok q));
  let t = v [ ("a", Column.v Type.int64 [| 1; 2 |]) ] in
  equal
    (Testable.make ~pp:Talon.pp ~equal:Talon.equal)
    t
    (run_ok (Query.of_table t |> Kit.union (Kit.union_all [])))

let union_all =
  [
    union_all_rows;
    conjunct_reaches_every_stored_branch;
    types_do_not_depend_on_order;
    test "of no query is the identity of union" union_all_of_nothing;
  ]

(* Hive paths *)

let hive_row (n, vs) = (n, Column.of_options Type.string vs)

let hive_table paths columns =
  v
    (("path", Column.v Type.string (Array.of_list paths))
    :: List.map hive_row columns)

let hive_cases =
  let some s = Some s in
  cases "reads the keys and values of paths as Hive writes them"
    ~id:(fun (name, _, _) -> name)
    [
      ( "a second = stays in the value",
        [ "a=b=c/f" ],
        `Rows [ ("a", [| some "b=c" |]) ] );
      ( "a slash decodes in a value",
        [ "a=x%2Fy/f" ],
        `Rows [ ("a", [| some "x/y" |]) ] );
      ( "an invalid escape stays literal",
        [ "a=%zz/f"; "a=%2/g"; "a=%/h" ],
        `Rows [ ("a", [| some "%zz"; some "%2"; some "%" |]) ] );
      ("a value invalid once decoded", [ "a=%FF/f" ], `Error "not valid UTF-8");
      ("a key decodes", [ "k%3D1=v/f" ], `Rows [ ("k=1", [| some "v" |]) ]);
      ("an empty value is text", [ "a=/f" ], `Rows [ ("a", [| some "" |]) ]);
      ("an empty key", [ "=v/f" ], `Error "its key is empty");
      ( "the default partition is null",
        [ "a=__HIVE_DEFAULT_PARTITION__/f" ],
        `Rows [ ("a", [| None |]) ] );
      ( "the file name is never a key",
        [ "a=1/b=2.parquet" ],
        `Rows [ ("a", [| some "1" |]) ] );
      ("a key repeated in a path", [ "a=1/a=2/f" ], `Error "repeated");
      ( "a path without keys beside one with",
        [ "a=1/f"; "g" ],
        `Error "different Hive keys" );
      ("different keys", [ "a=1/f"; "b=1/g" ], `Error "different Hive keys");
      ( "keys matched by name",
        [ "a=1/b=2/f"; "b=3/a=4/g" ],
        `Rows
          [ ("a", [| some "1"; some "4" |]); ("b", [| some "2"; some "3" |]) ]
      );
      ( "a key is text, its digits kept",
        [ "a=01234/f" ],
        `Rows [ ("a", [| some "01234" |]) ] );
      ( "the path column is taken",
        [ "path=1/f" ],
        `Error "holds the file's path" );
      ( "directories without = give no key",
        [ "x/a=1/y/f"; "x/a=2/y/g" ],
        `Rows [ ("a", [| some "1"; some "2" |]) ] );
      ("no directory", [ "f" ], `Rows []);
      ("a path not valid UTF-8", [ "\xff/f" ], `Error "not valid UTF-8");
      ("no paths", [], `Rows []);
    ]
    (fun (_, paths, expected) ->
      let table_w = Testable.make ~pp:Talon.pp ~equal:Talon.equal in
      match (expected, Source.hive paths) with
      | `Rows columns, Ok t -> equal table_w (hive_table paths columns) t
      | `Error sub, Error e -> contains ~sub (error e)
      | `Rows _, Error e -> failf "decoding fails: %s" (error e)
      | `Error sub, Ok _ -> failf "no error, wanted one containing %S" sub)

let hive_round_trip =
  let piece =
    Gen.(
      string_of ~size:(int_range 0 8)
        (of_list [ 'a'; 'Z'; '9'; '='; '/'; '%'; ' ' ]))
  in
  (* Percent-encodes every byte of [s] that [encode] marks, as a writer may. *)
  let encode ~all s =
    let b = Buffer.create 16 in
    String.iter
      (fun c ->
        if all || c = '/' || c = '=' || c = '%' then
          Buffer.add_string b (Printf.sprintf "%%%02X" (Char.code c))
        else Buffer.add_char b c)
      s;
    Buffer.contents b
  in
  prop "a percent-encoded key and value decode to themselves"
    Gen.(triple piece piece bool)
    (fun (key, value, all) ->
      assume (key <> "" && key <> "path");
      assume (value <> "__HIVE_DEFAULT_PARTITION__");
      let path = encode ~all key ^ "=" ^ encode ~all value ^ "/f" in
      match Source.hive [ path ] with
      | Error e -> failf "decoding %S fails: %s" path (error e)
      | Ok t ->
          equal
            (array (option string))
            [| Some value |]
            (Column.options Kind.string (column t key)))

let hive = [ hive_cases; hive_round_trip ]

let appending =
  prop "is each side at the common type, one after the other" appended
    (fun (Type.Any t0, Type.Any t1, Type.Any c, a, b) ->
      cover "two types" (not (Type.equal t0 t1));
      let k = Type.kind c in
      let rows t = Column.options k (column t "x") in
      let r = run_ok (Query.append (Query.of_table b) (Query.of_table a)) in
      equal schema_w (Schema.v [ ("x", Type.Any c) ]) (schema r);
      equal schema_w
        (Schema.v [ ("x", Type.Any c) ])
        (Schema.v [ ("x", Column.type_ (column r "x")) ]);
      equal
        (array (option (G.witness c)))
        (Array.append (rows a) (rows b))
        (rows r))

(* Splitting and replacing *)

(* Text of tokens that separators and replaced texts match, partly, in a row and
   across scalar values. *)
let token_text =
  Gen.map (String.concat "")
    (Gen.list ~size:(Gen.int_range 0 8)
       (Gen.of_list ~pp:Format.pp_print_string
          [ "a"; "b"; ","; ",,"; "\xc3\xa9"; "\xe6\x97\xa5"; "ab" ]))

let literal =
  Gen.of_list ~pp:Format.pp_print_string
    [ "a"; ","; ",,"; "aa"; "ab"; "\xc3\xa9"; "\xe6\x97\xa5\xe6\x97\xa5" ]

(* [over e rows] is [e] of the text column [s] of [rows]. *)
let over_text k e rows =
  let t = v [ ("s", Column.of_options Type.string rows) ] in
  Column.options k
    (column (run_ok (Query.select Expr.[ "e" := e ] (Query.of_table t))) "e")

(* [replaced sub ~by s] replaces [sub] in [s] left to right without overlap,
   byte by byte. *)
let replaced sub ~by s =
  let n = String.length sub and b = Buffer.create (String.length s) in
  let rec go i =
    if i > String.length s - n then
      Buffer.add_string b (String.sub s i (String.length s - i))
    else if String.equal (String.sub s i n) sub then begin
      Buffer.add_string b by;
      go (i + n)
    end
    else begin
      Buffer.add_char b s.[i];
      go (i + 1)
    end
  in
  go 0;
  Buffer.contents b

let split sep rows =
  over_text
    Kind.(list string)
    Expr.(Str.split (Str.literal sep) (Col.string "s"))
    rows

let pieces = array (option (array string))

let splitting =
  [
    prop "split's pieces joined by the separator are the text"
      (Gen.pair literal (Gen.array ~size:(Gen.int_range 0 6) token_text))
      (fun (sep, ss) ->
        cover "a row that holds the separator"
          (Array.exists
             (fun s -> not (String.equal (replaced sep ~by:"" s) s))
             ss);
        let got = split sep (Array.map Option.some ss) in
        equal
          (array (option string))
          (Array.map Option.some ss)
          (Array.map
             (Option.map (fun p -> String.concat sep (Array.to_list p)))
             got));
    prop "replace is the text with each match replaced, left to right"
      (Gen.triple literal literal
         (Gen.array ~size:(Gen.int_range 0 6) token_text))
      (fun (sub, by, ss) ->
        equal
          (array (option string))
          (Array.map (fun s -> Some (replaced sub ~by s)) ss)
          (over_text Kind.string
             Expr.(Str.replace (Str.literal sub) ~by (Col.string "s"))
             (Array.map Option.some ss)));
    cases "split gives the pieces between the separators"
      ~id:(fun (what, _, _, _) -> what)
      [
        ( "leading, trailing and adjacent separators",
          ",",
          ",a,,b,",
          [| ""; "a"; ""; "b"; "" |] );
        ("k occurrences give k + 1 pieces", ",", "a,b,c", [| "a"; "b"; "c" |]);
        ("no occurrence gives the text", ",", "ab", [| "ab" |]);
        ("the empty text is one empty piece", ",", "", [| "" |]);
        ("the text alone is two empty pieces", "ab", "ab", [| ""; "" |]);
        ( "a multi-byte separator",
          "\xc3\xa9",
          "a\xc3\xa9b\xc3\xa9",
          [| "a"; "b"; "" |] );
        ("occurrences do not overlap", "aa", "aaa", [| ""; "a" |]);
      ]
      (fun (_, sep, s, expected) ->
        equal pieces [| Some expected |] (split sep [| Some s |]));
    cases "replace replaces each occurrence, left to right"
      ~id:(fun (what, _, _, _, _) -> what)
      [
        ("occurrences do not overlap", "aa", "b", "aaa", "ba");
        ("every occurrence", ",", ";", ",a,,b,", ";a;;b;");
        ("by the empty text", ",", "", "a,b", "ab");
        ( "a multi-byte text by a longer one",
          "\xc3\xa9",
          "e\xcc\x81",
          "\xc3\xa9t\xc3\xa9",
          "e\xcc\x81te\xcc\x81" );
        ("the empty text", "a", "b", "", "");
      ]
      (fun (_, sub, by, s, expected) ->
        equal
          (array (option string))
          [| Some expected |]
          (over_text Kind.string
             Expr.(Str.replace (Str.literal sub) ~by (Col.string "s"))
             [| Some s |]));
    test "split and replace keep a null row null" (fun () ->
        equal pieces
          [| None; Some [| "a"; "" |]; None |]
          (split "," [| None; Some "a,"; None |]);
        equal
          (array (option string))
          [| None; Some "b" |]
          (over_text Kind.string
             Expr.(Str.replace (Str.literal "a") ~by:"b" (Col.string "s"))
             [| None; Some "a" |]));
  ]

(* Comparing text *)

(* Texts as their scalar values: a row drawn from a literal shares a prefix of
   its scalars, so it can be the literal, a prefix of it, an extension, a text
   of its length with one other byte, or a text that differs past a common
   prefix. Literals are often 8 or 16 bytes, or one byte off, where comparisons
   change method; bytes past 0x7f order after ASCII, and U+0000 below every
   other byte. *)
let scalar =
  Gen.frequency
    [
      (6, Gen.constant "a");
      (1, Gen.constant "b");
      (1, Gen.constant "\xc3\xa9");
      (1, Gen.constant "\000");
      (1, Gen.constant "\xf0\x9d\x84\x9e");
    ]

(* [fill n] is scalars of [n] bytes in all, padded with ["a"]. *)
let rec fill n =
  let open Gen in
  if n = 0 then constant []
  else
    let* s = scalar in
    let s = if String.length s <= n then s else "a" in
    let+ rest = fill (n - String.length s) in
    s :: rest

let scalars =
  let open Gen in
  bind
    (frequency
       [
         (2, constant 8);
         (2, constant 16);
         (3, of_list [ 7; 9; 15; 17 ]);
         (3, int_range 0 24);
       ])
    fill

let rec take k = function x :: xs when k > 0 -> x :: take (k - 1) xs | _ -> []

(* [other s] is a scalar of [s]'s bytes, another where [s] is ASCII. *)
let other = function "a" -> "b" | "b" -> "\000" | "\000" -> "a" | s -> s

(* [near lit] is a text that shares a prefix of the scalars [lit]. *)
let near lit =
  let open Gen in
  let n = List.length lit in
  let shared =
    let+ k = int_range 0 n and+ rest = list ~size:(int_range 0 3) scalar in
    String.concat "" (take k lit @ rest)
  in
  let changed =
    let+ k = int_range 0 (Int.max 0 (n - 1)) in
    String.concat "" (List.mapi (fun i s -> if i = k then other s else s) lit)
  in
  frequency [ (1, constant (String.concat "" lit)); (3, shared); (2, changed) ]

(* [row lit] is [near lit] or null. *)
let row lit =
  Gen.frequency [ (8, Gen.map Option.some (near lit)); (1, Gen.constant None) ]

let compare_op op a b =
  match op with
  | `Eq -> Expr.(a = b)
  | `Ne -> Expr.(a <> b)
  | `Lt -> Expr.(a < b)
  | `Le -> Expr.(a <= b)
  | `Gt -> Expr.(a > b)
  | `Ge -> Expr.(a >= b)

let op_name = function
  | `Eq -> "="
  | `Ne -> "<>"
  | `Lt -> "<"
  | `Le -> "<="
  | `Gt -> ">"
  | `Ge -> ">="

let ops = [ `Eq; `Ne; `Lt; `Le; `Gt; `Ge ]
let pp_op ppf op = Format.pp_print_string ppf (op_name op)

(* Mostly more than 8 rows, so that the result's bits pass their first byte. *)
let rows_size =
  Gen.frequency [ (1, Gen.int_range 0 8); (3, Gen.int_range 9 20) ]

let literal_case =
  let open Gen in
  let* lit = scalars in
  let+ rows = array ~size:rows_size (row lit)
  and+ op = of_list ~pp:pp_op ops
  and+ left = bool in
  (String.concat "" lit, rows, op, left)

(* Up to eight texts compare with the rows' bytes, more are coded with them. *)
let in_case =
  let open Gen in
  let* lit = scalars in
  let size = frequency [ (3, int_range 0 8); (1, int_range 9 10) ] in
  let+ vs = list ~size (near lit) and+ rows = array ~size:rows_size (row lit) in
  (vs, rows)

(* [some p rows] is [true] iff a row of [rows], not null, satisfies [p]. *)
let some p rows = Array.exists (function Some s -> p s | None -> false) rows

let is_prefix s ~of_ =
  String.length s <= String.length of_
  && String.equal s (String.sub of_ 0 (String.length s))

(* Rows that share prefixes and lengths, and a letter written precomposed and
   decomposed: a comparison with a literal compares their bytes, unsigned, a
   prefix first. *)
let text_rows =
  [|
    Some "";
    Some "ab";
    Some "abc";
    Some "a";
    Some "abd";
    None;
    Some "\xc3\xa9";
    Some "e\xcc\x81";
    Some "\xc3\xa9t\xc3\xa9";
  |]

let against_literals =
  List.map
    (fun (lit, op, left) -> (lit, text_rows, op, left))
    [
      ("ab", `Eq, false);
      ("ab", `Eq, true);
      ("ab", `Ne, false);
      ("", `Eq, false);
      ("\xc3\xa9", `Eq, false);
      ("\xc3\xa9t\xc3\xa9", `Ne, true);
      ("ab", `Lt, false);
      ("ab", `Le, false);
      ("ab", `Gt, false);
      ("ab", `Ge, false);
      ("ab", `Lt, true);
      ("abc", `Ge, true);
      ("", `Lt, false);
      ("", `Ge, false);
      ("z", `Gt, false);
      ("\xc3\xa9t\xc3\xa9", `Lt, false);
      ("abcdefgh", `Le, false);
      ("abcdefghijk", `Gt, true);
      ("abcdefghijklmnopqrs", `Ge, false);
    ]

let comparing =
  [
    prop ~examples:against_literals
      "with a text literal orders bytes as String.compare" literal_case
      (fun (lit, rows, op, left) ->
        let len = String.length lit in
        cover "a literal of under 8 bytes" (len < 8);
        cover "a literal of 8 bytes" (len = 8);
        cover "a literal of 9 to 16 bytes" (len > 8 && len <= 16);
        cover "a literal past 16 bytes" (len > 16);
        cover "a row equal to the literal" (some (String.equal lit) rows);
        cover "a row of the literal's length and other bytes"
          (some
             (fun s -> String.length s = len && not (String.equal s lit))
             rows);
        cover "a row a proper prefix of the literal"
          (some (fun s -> String.length s < len && is_prefix s ~of_:lit) rows);
        cover "a null row" (Array.exists Option.is_none rows);
        cover "more than 8 rows" (Array.length rows > 8);
        cover "<> over more than 8 rows" (op = `Ne && Array.length rows > 8);
        let s = Col.string "s" and l = Expr.string lit in
        let e = if left then compare_op op l s else compare_op op s l in
        let order s =
          if left then String.compare lit s else String.compare s lit
        in
        equal
          (array (option bool))
          (Array.map (Option.map (fun s -> cmp op (order s))) rows)
          (over_text Kind.bool e rows));
    prop
      ~examples:
        [
          ( [ "a"; "b"; "c"; "d"; "e"; "f"; "g"; "h"; "i" ],
            [| Some "a"; None; Some "z" |] );
        ]
      "is_in of texts is membership by bytes, false at a null" in_case
      (fun (vs, rows) ->
        let length s v = String.length v = String.length s in
        cover "a row among the texts" (some (fun s -> List.mem s vs) rows);
        cover "a row of a text's length but not among them"
          (some
             (fun s -> (not (List.mem s vs)) && List.exists (length s) vs)
             rows);
        cover "more than eight texts" (List.length vs > 8);
        cover "a null row" (Array.exists Option.is_none rows);
        cover "more than 8 rows" (Array.length rows > 8);
        let member = function Some s -> List.mem s vs | None -> false in
        equal
          (array (option bool))
          (Array.map (fun r -> Some (member r)) rows)
          (over_text Kind.bool Expr.(is_in vs (Col.string "s")) rows));
    cases "with null is null"
      ~id:(fun (name, _) -> name)
      Expr.
        [ ("= null", Col.string "s" = null); ("< null", Col.string "s" < null) ]
      (fun (_, e) ->
        equal
          (array (option bool))
          (Array.make 9 None)
          (over_text Kind.bool e text_rows));
    test "of two text columns of one row compares by bytes" (fun () ->
        let one =
          v
            [
              ("a", Column.v Type.string [| "ab" |]);
              ("b", Column.v Type.string [| "abc" |]);
            ]
        in
        let a = Col.string "a" and b = Col.string "b" in
        rows_are Type.bool [| f; t; t; f |]
          (Array.concat
             [
               bools Expr.(a = b) one;
               bools Expr.(a <> b) one;
               bools Expr.(a < b) one;
               bools Expr.(a >= b) one;
             ]));
  ]

(* Sorts and top-k *)

let pp_key ppf (k : R.key) =
  Format.fprintf ppf "%s %S%s"
    (if k.desc then "desc" else "asc")
    k.name
    (if k.nulls_first then " nulls first" else "")

let order (k : R.key) =
  let o = if k.desc then Order.desc k.name else Order.asc k.name in
  if k.nulls_first then Order.nulls_first o else o

(* [sorted ty k vs] is [vs] sorted stably by [k], as [Type.compare_value] orders
   the values. *)
let sorted ty (k : R.key) vs =
  let compare a b =
    match (a, b) with
    | None, None -> 0
    | None, Some _ -> if k.nulls_first then -1 else 1
    | Some _, None -> if k.nulls_first then 1 else -1
    | Some x, Some y ->
        let c = Type.compare_value ty x y in
        if k.desc then -c else c
  in
  Array.of_list (List.stable_sort compare (Array.to_list vs))

let rec has_ext : type a. a Type.t -> bool = function
  | Ext _ -> true
  | List e -> has_ext e
  | Record fs -> List.exists (fun (_, Type.Any t) -> has_ext t) fs
  | _ -> false

let one_key =
  Gen.with_pp
    (fun ppf (G.Sample (ty, vs), k, t) ->
      Format.fprintf ppf "%a by %a in %d batches" G.pp_sample
        (G.Sample (ty, vs))
        pp_key k
        (List.length (batches t)))
    (Gen.bind G.sample (fun (G.Sample (ty, vs) as sample) ->
         let x =
           if has_ext ty then v ~rows:0 []
           else v [ ("x", Column.of_options ty vs) ]
         in
         Gen.map
           (fun ((desc, nulls_first), t) ->
             (sample, { R.name = "x"; desc; nulls_first }, t))
           (Gen.pair (Gen.pair Gen.bool Gen.bool) (G.split x))))

(* An extension type, or one that holds one, has no order: its sample has no
   column. *)
let sorts_as_compare_value (G.Sample (ty, vs), k, t) =
  assume (rows t > 0 || Array.length vs = 0);
  if Schema.columns (schema t) <> [] then begin
    cover "nulls" (Array.exists Option.is_none vs);
    cover "several batches" (List.length (batches t) > 1);
    let ran = run_ok (Query.sort [ order k ] (Query.of_table t)) in
    equal
      (array (option (G.witness ty)))
      (sorted ty k vs)
      (Column.options (Type.kind ty) (column ran "x"))
  end

let top_k_cases =
  Gen.with_pp
    (fun ppf (t, ks, offset, length) ->
      Format.fprintf ppf "slice ~offset:%d ~length:%d of %a over %a" offset
        length
        (Format.pp_print_list pp_key)
        ks pp t)
    (Gen.bind (Gen.bind schemas table) (fun t ->
         let n = rows t in
         (* [k = offset + length] is drawn first, so that each edge has a branch
            of its own. *)
         let k =
           Gen.frequency
             [
               (1, Gen.constant 0);
               (1, Gen.constant 1);
               (1, Gen.constant n);
               (1, Gen.int_range (n + 1) (n + 3));
               (3, Gen.int_range 0 (n + 3));
             ]
         in
         Gen.bind
           (Gen.pair k (keys (Schema.columns (schema t))))
           (fun (k, ks) ->
             Gen.map
               (fun (offset, t) -> (t, ks, offset, k - offset))
               (Gen.pair (Gen.int_range 0 k) (G.split t)))))

(* A slice from the start over a sort is a top-k; over the sort's result read as
   a table, it is a slice. *)
let top_k_is_slice (t, ks, offset, length) =
  let k = offset + length and n = rows t in
  cover "k = 0" (k = 0);
  cover "k = 1" (k = 1);
  cover "k = rows" (k = n);
  cover "k past rows" (k > n);
  cover "a selection" (length > 0 && k < n && ks <> []);
  let sort = Query.sort (List.map order ks) (Query.of_table t) in
  same_outcome
    (Query.slice ~offset ~length sort)
    (Query.slice ~offset ~length (Query.of_table (run_ok sort)))

let sorting =
  prop "orders one key as Type.compare_value" one_key sorts_as_compare_value

(* Slices of five rows by one key that reach every cover of [top_k_is_slice]: [k
   = 0], [k = 1], a selection, [k = rows] and [k] past the rows. *)
let five_slices =
  let five = v [ ("x", Column.v Type.int64 [| 3; 1; 4; 1; 5 |]) ] in
  let key = { R.name = "x"; desc = false; nulls_first = false } in
  List.map
    (fun (offset, length) -> (five, [ key ], offset, length))
    [ (0, 0); (0, 1); (1, 1); (0, 5); (2, 5) ]

let top_k =
  prop ~examples:five_slices "of a sort is the slice of its rows" top_k_cases
    top_k_is_slice

(* Sources *)

(* A source that logs its calls: its requests, and for each part the readers
   opened, closed, and pulled after their close. A part is a list of batches,
   each read when [next] reaches it. *)
type log = {
  mutable requests : Source.request list;
  opened : int array;
  closed : int array;
  mutable late : int;
}

let counting ?sorted t parts =
  let log =
    {
      requests = [];
      opened = Array.make (List.length parts) 0;
      closed = Array.make (List.length parts) 0;
      late = 0;
    }
  in
  let part i bs =
    let open_ () =
      log.opened.(i) <- log.opened.(i) + 1;
      let rest = ref bs in
      let next () =
        if log.closed.(i) > 0 then log.late <- log.late + 1;
        match !rest with
        | [] -> Ok None
        | b :: bs ->
            rest := bs;
            Result.map Option.some (b ())
      in
      Ok
        {
          Source.next;
          close = (fun () -> log.closed.(i) <- log.closed.(i) + 1);
        }
    in
    { Source.rows = None; open_ }
  in
  let read r =
    log.requests <- r :: log.requests;
    Ok (List.mapi part parts)
  in
  (Source.v ~name:"counting" ~schema:(schema t) ?sorted read, log)

let table_x xs = v [ ("x", Column.v Type.int64 xs) ]
let parts xs = List.map (List.map (fun b () -> Ok (table_x b))) xs
let three = parts [ [ [| 0; 1 |]; [| 2 |] ]; [ [| 3; 4; 5 |] ]; [ [| 6 |] ] ]
let xs t = Column.values Kind.int (column t "x")
let every n log = equal (array int) (Array.make (Array.length log) n) log

(* A float column against values of float64, the type that contains it, each the
   float64 nearest a decimal: [0.1] is no float32's or float16's key, since
   neither holds it, while their own [0.1], widened, is one. *)
let pred_widened =
  cases "Pred compares a float column at its values' wider type"
    ~id:(fun (Type.Any t) -> Format.asprintf "%a" Type.pp t)
    Type.[ Any float32; Any float16 ]
    (fun (Type.Any t) ->
      match Kind.provably_equal (Type.kind t) Kind.float with
      | None -> fail "a float type"
      | Some Equal ->
          let c = Column.v t [| 0.1; 1.0 |] in
          let own = (Column.values Kind.float c).(0) in
          let tab = v [ ("x", c) ] in
          let kept p =
            match Source.Pred.filter (schema tab) p with
            | None -> -1
            | Some f -> rows (run_ok (Query.filter f (Query.of_table tab)))
          in
          let at x = Source.Pred.Value (Type.float64, x) in
          equal ~msg:"= 0.1" int 0 (kept (Cmp ("x", `Eq, at 0.1)));
          equal ~msg:"in [0.1; 2.]" int 0 (kept (In ("x", [ at 0.1; at 2. ])));
          equal ~msg:"= its own 0.1" int 1 (kept (Cmp ("x", `Eq, at own)));
          equal ~msg:"in [its own 0.1; 2.]" int 1
            (kept (In ("x", [ at own; at 2. ]))))

let sources =
  [
    pred_filter;
    pred_widened;
    test "a run opens each part once and closes each reader once" (fun () ->
        let s, log = counting (table_x [||]) three in
        let t = run_ok (Query.of_source s) in
        equal (array int) (Array.init 7 Fun.id) (xs t);
        equal int 1 (List.length log.requests);
        every 1 log.opened;
        every 1 log.closed;
        equal ~msg:"pulls after a close" int 0 log.late);
    test "a slice asks for its rows and opens only the parts that hold them"
      (fun () ->
        let s, log = counting (table_x [||]) three in
        let t = run_ok (Query.slice ~offset:1 ~length:3 (Query.of_source s)) in
        equal (array int) [| 1; 2; 3 |] (xs t);
        equal
          (list (option int))
          [ Some 4 ]
          (List.map (fun (r : Source.request) -> r.limit) log.requests);
        equal (array int) [| 1; 1; 0 |] log.opened;
        equal (array int) [| 1; 1; 0 |] log.closed);
    test "a source's error is the run's, after closing its reader" (fun () ->
        let e = Error.v ~row_group:1 "bad page" in
        let s, log =
          counting (table_x [||])
            [
              [ (fun () -> Ok (table_x [| 0 |])) ];
              [ (fun () -> Error e) ];
              [ (fun () -> Ok (table_x [| 1 |])) ];
            ]
        in
        let r = Query.run (Query.of_source s) in
        equal string (error e)
          (match r with Error e -> error e | Ok _ -> "no error");
        equal (array int) [| 1; 1; 0 |] log.opened;
        equal (array int) [| 1; 1; 0 |] log.closed);
    test "a batch of other columns raises and closes its reader" (fun () ->
        let other = v [ ("y", Column.v Type.int64 [| 0 |]) ] in
        let s, log = counting (table_x [||]) [ [ (fun () -> Ok other) ] ] in
        expect (message (fun () -> Query.run (Query.of_source s)))
        @@ __POS_OF__
             {| Query.run: counting yields a batch of the columns y int64, not x int64 |};
        every 1 log.closed);
    test "a raising function closes every reader" (fun () ->
        let s, log = counting (table_x [||]) three in
        let f x = if x = 4 then raise (Boom (0, x)) else x in
        let q =
          Query.filter
            Expr.(store Type.int64 (const f $ Col.int "x") >= int 0)
            (Query.of_source s)
        in
        raises (Boom (0, 4)) (fun () -> Query.run q);
        equal (array int) [| 1; 1; 0 |] log.closed);
    test "rows out of the stated order fail at the first, across batches"
      (fun () ->
        let s, _ =
          counting
            ~sorted:[ Order.asc "x" ]
            (table_x [||])
            (parts [ [ [| 0; 2 |]; [| 2; 3 |] ]; [ [| 1; 4 |] ] ])
        in
        let seen = ref 0 in
        let r =
          Query.fold (Query.of_source s) ~init:() (fun () b ->
              seen := !seen + rows b)
        in
        expect (match r with Error e -> error e | Ok () -> "no error")
        @@ __POS_OF__
             {| counting (1 column): row 4: the row is out of the source's order [asc "x"]. |};
        equal ~msg:"rows folded before the failure" int 4 !seen);
    test "two places with one request read the source once" (fun () ->
        let s, log = counting (table_x [||]) three in
        let q = Query.of_source s in
        let t = run_ok (Query.append q q) in
        equal int 14 (rows t);
        equal int 1 (List.length log.requests);
        every 1 log.opened;
        every 1 log.closed);
  ]

(* A failure closes every reader a run opened. *)
let closing =
  let closes_opened (log : log) =
    equal ~msg:"closed" (array int) log.opened log.closed
  in
  [
    test "closes every reader when f raises" (fun () ->
        let s, log = counting (table_x [||]) three in
        let f () b = if Array.mem 2 (xs b) then raise (Boom (0, 2)) in
        raises
          (Boom (0, 2))
          (fun () -> Query.fold (Query.of_source s) ~init:() f);
        closes_opened log);
    test "closes every reader when a source's next raises" (fun () ->
        let s, log =
          counting (table_x [||])
            [
              [ (fun () -> Ok (table_x [| 0 |])) ];
              [ (fun () -> raise (Boom (0, 1))) ];
              [ (fun () -> Ok (table_x [| 2 |])) ];
            ]
        in
        raises (Boom (0, 1)) (fun () -> Query.run (Query.of_source s));
        closes_opened log);
    test "closes every reader when a source's open_ raises" (fun () ->
        let opened = ref 0 and closed = ref 0 in
        let read = ref false in
        let next () =
          if !read then Ok None
          else begin
            read := true;
            Ok (Some (table_x [| 0 |]))
          end
        in
        let reader = { Source.next; close = (fun () -> incr closed) } in
        let parts =
          [
            {
              Source.rows = None;
              open_ =
                (fun () ->
                  incr opened;
                  Ok reader);
            };
            { Source.rows = None; open_ = (fun () -> raise (Boom (0, 1))) };
          ]
        in
        let s =
          Source.v ~name:"two"
            ~schema:(schema (table_x [||]))
            (fun _ -> Ok parts)
        in
        raises (Boom (0, 1)) (fun () -> Query.run (Query.of_source s));
        equal ~msg:"closed" int !opened !closed);
  ]

let values_binds =
  test "raises a verb's report, named values, where e does not bind" (fun () ->
      let q = Query.of_table (table_x [| 1 |]) in
      let report = message (fun () -> Query.values (Col.string "x") q) in
      starts_with ~prefix:"values: 1 problem" report;
      contains ~sub:"input (1 column): x int64" report)

let values_reads_its_columns =
  test "asks the source for e's columns alone" (fun () ->
      let both =
        v [ ("x", Column.v Type.int64 [||]); ("y", Column.v Type.int64 [||]) ]
      in
      let s, log = counting both (parts [ [ [| 1; 2 |] ] ]) in
      equal (array int) [| 1; 2 |]
        (require_ok ~pp:Error.pp
           (Query.values (Col.int "x") (Query.of_source s)));
      equal
        (list (list string))
        [ [ "x" ] ]
        (List.map (fun (r : Source.request) -> r.columns) log.requests))

let sharing =
  [
    test "computes equal subexpressions of a step once" (fun () ->
        let calls = ref 0 in
        let f x =
          incr calls;
          x
        in
        let s = Expr.(store Type.int64 (const f $ Col.int "x")) in
        let q =
          Query.select
            Expr.[ "y" := s * s ]
            (Query.of_table (table_x [| 1; 2; 3 |]))
        in
        equal (array int) [| 1; 4; 9 |]
          (require_ok ~pp:Error.pp (Query.values (Col.int "y") q));
        equal ~msg:"calls" int 3 !calls);
    test "runs a step its plan reaches twice once" (fun () ->
        let calls = ref 0 in
        let f x =
          incr calls;
          x
        in
        let q =
          Query.filter
            Expr.(store Type.int64 (const f $ Col.int "x") >= int 0)
            (Query.of_table (table_x (Array.init 7 Fun.id)))
        in
        let t = run_ok (Query.append (Query.slice ~offset:0 ~length:2 q) q) in
        equal (array int) [| 0; 1; 2; 3; 4; 5; 6; 0; 1 |] (xs t);
        equal ~msg:"calls" int 7 !calls);
    test "fails a shared step only where a reader needs its rows" (fun () ->
        let f x = if x = 5 then raise (Boom (0, x)) else x in
        let q =
          Query.derive
            Expr.[ "y" := store Type.int64 (const f $ Col.int "x") ]
            (Query.of_table (table_x (Array.init 7 Fun.id)))
        in
        let shared n = Query.slice ~offset:0 ~length:n (Query.append q q) in
        equal int 4 (rows (run_ok (shared 4)));
        raises (Boom (0, 5)) (fun () -> Query.run (shared 6)));
  ]

let select_reads_input =
  test "reads the input's columns, never another output" (fun () ->
      let x = Col.int "x" in
      let q =
        Query.select
          Expr.[ "x" := x + int 1; "y" := x ]
          (Query.of_table (table_x [| 1; 2 |]))
      in
      let t = run_ok q in
      equal ~msg:"x" (array int) [| 2; 3 |] (xs t);
      equal ~msg:"y" (array int) [| 1; 2 |]
        (Column.values Kind.int (column t "y")))

(* Kit's runs *)

let tokens = [| Some "b"; Some "a"; None; Some "\xc3\xa9"; Some "B"; Some "a" |]

let categorized ?(table = v [ ("s", Column.of_options Type.string tokens) ]) cs
    =
  require_ok ~pp:Error.pp (Kit.categorize cs (Query.of_table table))

let kit_runs =
  [
    test "makes the dictionary of the values in byte order" (fun () ->
        let q = categorized [ "s" ] in
        expect (Format.asprintf "%a" Schema.pp (Query.schema q))
        @@ __POS_OF__ {| s categorical["B", "a", "b", "é"] |};
        equal
          (array (option string))
          tokens
          (Column.options Kind.string (column (run_ok q) "s")));
    test "makes a dictionary blind to order and batches" (fun () ->
        let half a b = Column.of_options Type.string (Array.sub tokens a b) in
        let table =
          of_batches [ v [ ("s", half 3 3) ]; v [ ("s", half 0 3) ] ]
        in
        equal schema_w
          (Query.schema (categorized [ "s" ]))
          (Query.schema (categorized ~table [ "s" ])));
    test "of a categorical keeps only the values it holds" (fun () ->
        let table =
          v
            [
              ( "c",
                Column.v
                  (Type.categorical [| "z"; "y"; "x" |])
                  [| "y"; "z"; "y" |] );
            ]
        in
        expect
          (Format.asprintf "%a" Schema.pp
             (Query.schema (categorized ~table [ "c" ])))
        @@ __POS_OF__ {| c categorical["y", "z"] |});
    test "of no column is the query" (fun () ->
        let q = Query.of_table (v [ ("s", Column.v Type.string [| "a" |]) ]) in
        equal
          (Testable.make ~pp:Query.pp ~equal:Query.equal)
          q
          (require_ok ~pp:Error.pp (Kit.categorize [] q)));
    test "refuses a column it cannot categorize" (fun () ->
        let table =
          v
            [
              ("s", Column.v Type.string [| "a" |]);
              ("n", Column.v Type.int64 [| 1 |]);
            ]
        in
        expect
          (String.concat "\n"
             [
               message (fun () -> categorized ~table [ "s"; "s" ]);
               message (fun () -> categorized ~table [ "s"; "n" ]);
               message (fun () -> categorized ~table [ "nope" ]);
             ])
        @@ __POS_OF__
             {|
            Kit.categorize: "s" is named twice
            select: 1 problem
              "n" := cast string n
                Col.string reads string or categorical, but "n" is int64.
              input (2 columns): s string, n int64
            select: 1 problem
              "nope" := cast string nope
                no column "nope". The columns are "s" and "n".
              input (2 columns): s string, n int64
            |});
    test "returns a failing run's error" (fun () ->
        let q =
          Query.of_table (v [ ("s", Column.v Type.string [| "1"; "x" |]) ])
          |> Query.derive
               Expr.[ "s" := Str.slice ~offset:0 ~length:1 (Col.string "s") ]
          |> Query.filter Expr.(Str.parse Type.int64 (Col.string "s") > int 0)
        in
        match Kit.categorize [ "s" ] q with
        | Ok _ -> fail "no error"
        | Error e ->
            expect (Format.asprintf "%a" Error.pp e)
            @@ __POS_OF__
                 {| filter (Str.parse int64 s > 0): row 1: "x": not an integer. |});
  ]

(* Values under nulls

   A null row's value is unspecified: whatever lies under it, a run reads the
   row as null, and lays it out the same whatever the batches. *)

(* [garbled vs under] is the int64 column of [vs] with [under.(i)] under the
   null row [i], its validity a view from bit 3 of longer bits, set around
   it. *)
let garbled vs under =
  let n = Array.length vs in
  let values = Array.mapi (fun i o -> Option.value o ~default:under.(i)) vs in
  let bits =
    Array.init (n + 11) (fun i ->
        i < 3 || i >= n + 3 || Option.is_some vs.(i - 3))
  in
  let validity =
    Nx.slice
      [ Nx.R (3, n + 3) ]
      (Nx.cast Nx.bit (Nx.create Nx.bool [| n + 11 |] bits))
  in
  Column.of_tensor ~validity (Nx.create Nx.int64 [| n |] values)

let validity_bytes c =
  match Column.validity c with
  | None -> None
  | Some b ->
      let n = Nx.numel b in
      Nx.Repr.read ~by:"test" b @@ fun a ->
      Rig.Buffer.read (Nx_array.buffer a) Bigarray.int8_unsigned (fun raw ->
          Some (Array.init ((n + 7) / 8) (fun i -> raw.{i})))

let null_keys = [| s 1L; None; s 2L; None; s 1L |]

let equal_tables =
  test "reads the nulls of a column, not the values under them" (fun () ->
      let t under = v [ ("k", garbled null_keys under) ] in
      let table = Testable.make ~pp:Talon.pp ~equal:Talon.equal in
      equal table (t [| 0L; 7L; 0L; -9L; 0L |]) (t [| 0L; 0L; 0L; 0L; 0L |]);
      not_equal table
        (t [| 0L; 7L; 0L; -9L; 0L |])
        (v [ ("k", Column.v Type.int64 [| 1; 7; 2; -9; 1 |]) ]))

let null_group =
  test "makes one group of null keys over different values" (fun () ->
      let t =
        v
          [
            ("k", garbled null_keys [| 0L; 7L; 0L; -9L; 0L |]);
            ("y", i64 [| 10; 11; 12; 13; 14 |]);
          ]
      in
      let g =
        run_ok
          (Query.aggregate ~by:[ "k" ]
             Expr.[ "n" := count (Col.int "y"); "s" := sum (Col.int "y") ]
             (Query.of_table t))
      in
      let by_key =
        List.sort compare
          (List.combine
             (Array.to_list (Column.options Kind.int (column g "k")))
             (Array.to_list (Column.options Kind.int (column g "s"))))
      in
      equal
        (list (Windtrap.pair (option int) (option int)))
        [ (None, s 24); (s 1, s 24); (s 2, s 12) ]
        by_key)

let null_match =
  test "matches null keys over different values to each other" (fun () ->
      let l =
        v
          [
            ("k", garbled [| None; s 5L |] [| 3L; 0L |]); ("x", i64 [| 0; 1 |]);
          ]
      in
      let r =
        v
          [
            ("k", garbled [| s 3L; None |] [| 0L; -1L |]);
            ("y", i64 [| 10; 11 |]);
          ]
      in
      let t = joined ~kind:Left (Join.keys [ "k" ]) l r in
      column_is t ("x", [ s 0; s 1 ]);
      column_is t ("y", [ s 11; None ]))

(* Three parts of 13 rows, null where [(i + j) mod 4 = 0], the values under
   their nulls [-1]: filtered to nulls and values above 2, sorted descending
   with nulls first. *)
let offset_validities =
  test
    "reads the nulls of offset validities through a filter, a sort and an \
     append" (fun () ->
      let at i j = if (i + j) mod 4 = 0 then None else s j in
      let part i =
        garbled
          (Array.init 13 (fun j -> Option.map Int64.of_int (at i j)))
          (Array.make 13 (-1L))
      in
      let t = of_batches (List.init 3 (fun i -> v [ ("x", part i) ])) in
      let x = Col.int "x" in
      let q =
        Query.(
          of_table t
          |> filter Expr.(is_null x || x > int 2)
          |> sort [ Order.nulls_first (Order.desc "x") ])
      in
      let all = List.concat (List.init 3 (fun i -> List.init 13 (at i))) in
      let nulls = List.filter Option.is_none all in
      let values =
        List.filter (function Some j -> j > 2 | None -> false) all
      in
      let expected = nulls @ List.sort (fun a b -> compare b a) values in
      let c = column (run_ok q) "x" in
      equal ~msg:"null count" int (List.length nulls) (Column.null_count c);
      equal
        (list (option int))
        expected
        (Array.to_list (Column.options Kind.int c)))

let canonical_validity =
  test "lays out a validity of 13 rows from three parts canonically" (fun () ->
      let part vs = v [ ("x", garbled vs (Array.make (Array.length vs) 5L)) ] in
      let t =
        of_batches
          [
            part [| None; s 1L; s 2L; s 3L; None |];
            part [| s 1L; None; s 1L; s 1L |];
            part [| s 9L; s 9L; None; s 9L |];
          ]
      in
      let c = column (run_ok (Query.of_table t)) "x" in
      (* Rows 0, 4, 6 and 11 are null: 0b10101110 then 0b00010111. *)
      equal (option (array int)) (Some [| 0xae; 0x17 |]) (validity_bytes c))

let tensor_layout =
  test
    "lays out a comparison of tensor cells with null in the same bytes in one \
     batch and row by row" (fun () ->
      let cells = Nx.create Nx.float32 [| 2; 2 |] [| 0.; 1.; 0.; 0. |] in
      let row i =
        v [ ("c", Column.of_tensor (Nx.slice [ R (i, i + 1) ] cells)) ]
      in
      let c = Col.v (Kind.tensor Nx.float32) "c" in
      let q t = Query.(select Expr.[ "r" := not (c = null) ] (of_table t)) in
      let one = run_ok (q (v [ ("c", Column.of_tensor cells) ])) in
      let rows = run_ok (q (of_batches [ row 0; row 1 ])) in
      equal (list string) (buffers (column rows "r")) (buffers (column one "r")))

let group_key =
  test "holds a group's key at its first row's value" (fun () ->
      let first ks =
        let t =
          v [ ("k", Column.v Type.float64 ks); ("n", int64s [| 1L; 2L |]) ]
        in
        floats (run_ok (Query.aggregate ~by:[ "k" ] [] (Query.of_table t))) "k"
      in
      rows_are Type.float64 [| s (-0.) |] (first [| -0.; 0. |]);
      rows_are Type.float64 [| s 0. |] (first [| 0.; -0. |]))

let () =
  exit
    (run "Run"
       [
         group "Talon.equal" [ equal_tables ];
         group "Elementwise operations"
           (arithmetic @ powers @ [ widening; float_order ] @ comparing @ kleene
          @ computed_operands @ lifts);
         group "Reductions" (reductions @ one_column);
         group "Frames" frames;
         group "OCaml values" ocaml;
         group "Expr.Str" splitting;
         group "Expr.Temporal" [ utc_wall_clock ];
         group "Source" (sources @ hive);
         group "Join" joins;
         group "Query.select" [ select_reads_input ];
         group "Query.sort" [ sorting ];
         group "Query.slice"
           [ top_k; slice_reads_none; slice_after_failing_join ];
         group "Query.aggregate"
           (aggregate_laws @ streamed_edges @ [ null_group; group_key ]);
         group "Query.join" [ join_streams; null_match ];
         group "Query.append" [ appending ];
         group "Query.fold"
           ((fold_laws @ [ blocking_emits_nothing ]) @ failure_order @ closing);
         group "Query.run"
           (run_laws @ [ offset_validities; canonical_validity; tensor_layout ]);
         group "Query.values"
           ((values_law :: values_failures)
           @ [ values_with_nulls; values_binds; values_reads_its_columns ]);
         group "Query.optimize"
           (filters_over_constants :: extension_cases :: sharing);
         group "Kit.union_all" union_all;
         group "Kit.categorize" kit_runs;
       ])
