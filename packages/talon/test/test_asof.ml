(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

let exec q = require_ok ~pp:Error.pp (Query.run q)
let ints_of t c = Column.options Kind.int (Talon.column t c)
let places n = Column.v Type.int64 (Array.init n Fun.id)

(* Frames

   A drawn frame is a left table of rows [i], a key [g] and a value [t], and a
   right table of rows [j], a key [g], a value [rt] and a payload [v], in no
   order. Values of one kind draw from few values, so that ties, extremes and
   rows without a match come often. Keys are integers, or floats from [-0.],
   [0.] and NaN, which are two keys by identity. *)

type coord = I of int | F of float | S of string

type kind =
  | Int8
  | Uint8
  | Int
  | Float32
  | Float
  | Date
  | Datetime
  | Duration
  | Clock
  | String
  | Bool

type keys = Unkeyed | Ints | Floats

type frame = {
  kind : kind;
  keys : keys;
  direction : Kit.direction;
  strict : bool;
  lg : float option array;
  lt : coord option array;
  rg : float option array;
  rt : coord option array;
  v : int option array;
}

let kinds =
  [
    (Int8, "int8");
    (Uint8, "uint8");
    (Int, "int64");
    (Float32, "float32");
    (Float, "float64");
    (Date, "date");
    (Datetime, "datetime");
    (Duration, "duration");
    (Clock, "clock");
    (String, "string");
    (Bool, "bool");
  ]

let directions =
  Kit.[ (Backward, "backward"); (Forward, "forward"); (Nearest, "nearest") ]

let coords kind =
  let ints = List.map (fun i -> I i) and floats = List.map (fun x -> F x) in
  match kind with
  | Int8 -> ints [ -128; -1; 0; 1; 127 ]
  | Uint8 -> ints [ 0; 1; 2; 254; 255 ]
  | Int -> ints [ min_int; -3; -1; 0; 1; 2; 4; max_int ]
  | Float | Float32 ->
      floats [ neg_infinity; -2.5; -1.; -0.; 0.; 1.; 2.5; infinity; nan ]
  | Date -> ints [ -1; 0; 1; 3 ]
  | Datetime | Duration -> ints [ -2000; 0; 1000; 1500; 4000 ]
  | Clock -> ints [ 0; 1000; 1500; 4000 ]
  | String -> List.map (fun s -> S s) [ ""; "a"; "ab"; "b" ]
  | Bool -> ints [ 0; 1 ]

let frame =
  let open Gen in
  let* kind = of_list (List.map fst kinds) in
  let nullable g = frequency [ (1, constant None); (5, map Option.some g) ] in
  let size = frequency [ (1, int_range 0 1); (5, int_range 2 10) ] in
  let* n, m = pair size size in
  let rows n g = array ~size:(constant n) (nullable g) in
  let* keys =
    frequency
      [ (2, constant Unkeyed); (1, constant Ints); (1, constant Floats) ]
  in
  let key =
    match keys with
    | Floats -> of_list [ -0.; 0.; nan ]
    | Unkeyed | Ints -> of_list [ 0.; 1. ]
  in
  let value = of_list (coords kind) in
  let* lg, lt = pair (rows n key) (rows n value) in
  let* rg, rt = pair (rows m key) (rows m value) in
  let* v = rows m (int_range 0 99) in
  (* Strings and booleans have no distance. Nearest comes twice as often
     otherwise, so that its distance ties come often. *)
  let measured = not (List.mem kind [ String; Bool ]) in
  let weight (d, _) =
    if d <> Kit.Nearest then 1 else if measured then 2 else 0
  in
  let* direction =
    frequency (List.map (fun d -> (weight d, constant (fst d))) directions)
  in
  let+ strict = bool in
  { kind; keys; direction; strict; lg; lt; rg; rt; v }

let column kind cs =
  let v ty f = Column.of_options ty (Array.map (Option.map f) cs) in
  let int = function I i -> i | _ -> assert false in
  let float = function F x -> x | _ -> assert false in
  let span c = Time.Span.of_ns (Int64.of_int (int c)) in
  match kind with
  | Int8 -> v Type.int8 int
  | Uint8 -> v Type.uint8 int
  | Int -> v Type.int64 int
  | Float32 -> v Type.float32 float
  | Float -> v Type.float64 float
  | Date -> v Type.date (fun c -> Option.get (Time.Date.of_days (int c)))
  | Datetime ->
      v (Type.instant Type.Ns) (fun c -> Time.of_ns (Int64.of_int (int c)))
  | Duration -> v (Type.duration Type.Ns) span
  | Clock -> v (Type.clock Type.Ns) span
  | String -> v Type.string (function S s -> s | _ -> assert false)
  | Bool -> v Type.bool (fun c -> int c = 1)

let key_column f gs =
  match f.keys with
  | Floats -> Column.of_options Type.float64 gs
  | Unkeyed | Ints ->
      Column.of_options Type.int64 (Array.map (Option.map Float.to_int) gs)

let tables f =
  let n = Array.length f.lt and m = Array.length f.rt in
  let g = if f.keys = Unkeyed then [] else [ ("g", key_column f f.rg) ] in
  ( Talon.v ~rows:n
      [ ("i", places n); ("g", key_column f f.lg); ("t", column f.kind f.lt) ],
    Talon.v ~rows:m
      ((("j", places m) :: g)
      @ [ ("rt", column f.kind f.rt); ("v", Column.of_options Type.int64 f.v) ]
      ) )

let pp ppf f =
  let left, right = tables f in
  Format.fprintf ppf "%s %s strict=%b@.%a@.%a" (List.assoc f.kind kinds)
    (List.assoc f.direction directions)
    f.strict Talon.pp left Talon.pp right

let join_asof f =
  let left, right = tables f in
  let by = if f.keys = Unkeyed then [] else [ "g" ] in
  Kit.join_asof ~by ~direction:f.direction ~strict:f.strict ~on:("t", "rt")
    (Query.of_table right) (Query.of_table left)

(* The reference *)

(* Talon's total order: NaN after every number, [-0.] equal to [0.]. *)
let compare_coord a b =
  match (a, b) with
  | I a, I b -> Int.compare a b
  | S a, S b -> String.compare a b
  | F a, F b -> (
      match (Float.is_nan a, Float.is_nan b) with
      | false, false -> Float.compare (a +. 0.) (b +. 0.)
      | na, nb -> Bool.compare na nb)
  | _ -> assert false

(* Keys match by identity: [-0.] is [0.], NaN is NaN, null is null. *)
let same_key = Option.equal (fun a b -> compare_coord (F a) (F b) = 0)

(* [dist l c] is the exact distance between [l] and [c], integers subtracting in
   64 bits unsigned: [0] between equal values, infinite where either is infinite
   or NaN. Positive floats order as their bits. *)
let dist l c =
  match (l, c) with
  | I l, I c -> Int64.(sub (of_int (Int.max l c)) (of_int (Int.min l c)))
  | _ when compare_coord l c = 0 -> 0L
  | F l, F c when Float.is_finite l && Float.is_finite c ->
      Int64.bits_of_float (Float.abs (l -. c))
  | _ -> Int64.bits_of_float infinity

(* [pick f ~forward i] is the right row that row [i]'s matches select: the
   greatest [rt] at or before [t], the last of its ties, or forward the least at
   or after it, the first of its ties. *)
let pick f ~forward i =
  match f.lt.(i) with
  | None -> None
  | Some l ->
      let side c =
        let o = compare_coord c l in
        if forward then o > 0 || (o = 0 && not f.strict)
        else o < 0 || (o = 0 && not f.strict)
      in
      let matches j =
        match f.rt.(j) with
        | Some c -> (f.keys = Unkeyed || same_key f.rg.(j) f.lg.(i)) && side c
        | None -> false
      in
      let better j b =
        let o = compare_coord (Option.get f.rt.(j)) (Option.get f.rt.(b)) in
        if forward then o < 0 else o >= 0
      in
      let keep best j =
        if not (matches j) then best
        else
          match best with Some b when not (better j b) -> best | _ -> Some j
      in
      List.fold_left keep None (List.init (Array.length f.rt) Fun.id)

let reference f =
  let nearest i =
    match (pick f ~forward:false i, pick f ~forward:true i) with
    | Some a, Some b ->
        let l = Option.get f.lt.(i) and key j = Option.get f.rt.(j) in
        let da = dist l (key a) and db = dist l (key b) in
        Some (if Int64.unsigned_compare da db <= 0 then a else b)
    | a, None -> a
    | None, b -> b
  in
  Array.init (Array.length f.lt) (fun i ->
      match f.direction with
      | Backward -> pick f ~forward:false i
      | Forward -> pick f ~forward:true i
      | Nearest -> nearest i)

let odd = function
  | F x -> Float.is_nan x || Float.abs x = infinity
  | _ -> false

let law f =
  let got = exec (join_asof f) and want = reference f in
  let n = Array.length f.lt in
  let present = List.filter_map Fun.id (Array.to_list f.rt) in
  (* Another key of the left row's group lies at the picked key's distance. *)
  let tie i =
    match (f.lt.(i), want.(i)) with
    | Some l, Some j ->
        let picked = Option.get f.rt.(j) in
        let rival k c =
          (f.keys = Unkeyed || same_key f.rg.(k) f.lg.(i))
          && compare_coord c picked <> 0
          && dist l c = dist l picked
        in
        Array.exists Fun.id
          (Array.mapi
             (fun k c -> Option.fold ~none:false ~some:(rival k) c)
             f.rt)
    | _ -> false
  in
  let keyed p =
    f.keys = Floats
    && Array.exists (Option.fold ~none:false ~some:p) (Array.append f.lg f.rg)
  in
  let zero neg g = g = 0. && Float.sign_bit g = neg in
  cover (List.assoc f.kind kinds) true;
  cover (List.assoc f.direction directions) true;
  cover "strict" f.strict;
  cover "tied right values"
    (List.length (List.sort_uniq compare present) < List.length present);
  cover "a null value"
    (Array.exists Option.is_none f.lt || Array.exists Option.is_none f.rt);
  cover "a NaN or infinite value" (List.exists odd present);
  cover "an empty side" (n = 0 || Array.length f.rt = 0);
  cover "a left row without a match" (Array.exists Option.is_none want);
  cover "a nearest distance tie"
    (f.direction = Kit.Nearest && List.exists tie (List.init n Fun.id));
  cover "a -0. and a 0. key" (keyed (zero true) && keyed (zero false));
  cover "a NaN key" (keyed Float.is_nan);
  equal (array (option int)) (Array.init n Option.some) (ints_of got "i");
  equal (array (option int)) want (ints_of got "j");
  equal
    (array (option int))
    (Array.map (Option.fold ~none:None ~some:(Array.get f.v)) want)
    (ints_of got "v")

(* A nearest pick between keys 1 and 3 from 2, at equal distances. *)
let nearest_tie =
  {
    kind = Int;
    keys = Unkeyed;
    direction = Nearest;
    strict = false;
    lg = [| None |];
    lt = [| Some (I 2) |];
    rg = [| None; None |];
    rt = [| Some (I 1); Some (I 3) |];
    v = [| Some 0; Some 1 |];
  }

(* Exact distances *)

let nearest l rs =
  let left = Query.of_table (Talon.v [ ("t", l) ]) in
  let right =
    Query.of_table (Talon.v [ ("j", places (Column.length rs)); ("rt", rs) ])
  in
  ints_of
    (exec (Kit.join_asof ~direction:Nearest ~on:("t", "rt") right left))
    "j"

let bits dt xs =
  Column.of_tensor (Nx.bitcast dt (Nx.create Nx.int64 [| Array.length xs |] xs))

let p53 = Float.ldexp 1. 53
let far_ns = Type.instant Type.Ns

let far =
  Time.[| of_ns Int64.(add min_int 10L); of_ns Int64.(sub max_int 10L) |]

let distances =
  cases "Nearest's distance is exact" ~id:fst
    [
      (* [0.5 - -2^53] and [2^53 - 0.5] both round to 2^53. *)
      ( "float64 0.5 between -2^53 and 2^53",
        ( Column.v Type.float64 [| 0.5 |],
          Column.v Type.float64 [| -.p53; p53 |],
          1 ) );
      ( "float64 -0.5 between -2^53 and 2^53",
        ( Column.v Type.float64 [| -0.5 |],
          Column.v Type.float64 [| -.p53; p53 |],
          0 ) );
      (* The distances wrap in their type. *)
      ( "int8 0 between -128 and 127",
        (Column.v Type.int8 [| 0 |], Column.v Type.int8 [| -128; 127 |], 1) );
      ( "int64 0 between min_int and max_int",
        (bits Nx.int64 [| 0L |], bits Nx.int64 Int64.[| min_int; max_int |], 1)
      );
      ( "int64 -1 between min_int and max_int",
        (bits Nx.int64 [| -1L |], bits Nx.int64 Int64.[| min_int; max_int |], 0)
      );
      ( "uint64 2^63 between 0 and 2^64 - 1",
        (bits Nx.uint64 [| Int64.min_int |], bits Nx.uint64 [| 0L; -1L |], 1) );
      ( "uint64 2^63 - 1 between 0 and 2^64 - 1",
        (bits Nx.uint64 [| Int64.max_int |], bits Nx.uint64 [| 0L; -1L |], 0) );
      (* A lone pick needs no distance, however far. *)
      ( "a lone datetime 2^64 ns away",
        (Column.v far_ns [| far.(1) |], Column.v far_ns [| far.(0) |], 0) );
    ]
    (fun (_, (l, rs, want)) ->
      equal (array (option int)) [| Some want |] (nearest l rs))

(* Oracles

   golden/asof.txt holds the right rows that pandas' merge_asof picks over
   sorted integer values, which Polars' join_asof agrees with but on nearest
   ties (gen/asof.py). *)

let golden =
  lazy
    (In_channel.with_open_text "golden/asof.txt" In_channel.input_lines
    |> List.map (String.split_on_char ' '))

let side name key index =
  let rows =
    List.filter_map
      (function s :: r when s = name -> Some r | _ -> None)
      (Lazy.force golden)
  in
  let col i =
    Array.of_list (List.map (fun r -> int_of_string (List.nth r i)) rows)
  in
  Query.of_table
    (Talon.v
       [
         ("g", Column.v Type.int64 (col 0));
         (key, Column.v Type.int64 (col 1));
         (index, places (List.length rows));
       ])

let pandas (direction, exact) =
  let name = List.assoc direction directions in
  let picks =
    List.find_map
      (function
        | "pick" :: d :: e :: js when d = name && e = exact -> Some js
        | _ -> None)
      (Lazy.force golden)
  in
  let want =
    List.map
      (function "null" -> None | j -> Some (int_of_string j))
      (Option.get picks)
  in
  let q =
    Kit.join_asof ~by:[ "g" ] ~direction ~strict:(exact = "0") ~on:("t", "rt")
      (side "right" "rt" "j") (side "left" "t" "i")
  in
  equal (array (option int)) (Array.of_list want) (ints_of (exec q) "j")

(* A source of [t] that counts the reads of its one part. *)
let counted t =
  let opens = ref 0 in
  let reader (r : Source.request) () =
    incr opens;
    let batch =
      Talon.v ~rows:(Talon.rows t)
        (List.map (fun c -> (c, Talon.column t c)) r.columns)
    in
    let left = ref [ batch ] in
    let next () =
      match !left with
      | [] -> Ok None
      | b :: rest ->
          left := rest;
          Ok (Some b)
    in
    Ok { Source.next; close = ignore }
  in
  let source =
    Source.v ~name:"counted" ~schema:(Talon.schema t) (fun r ->
        Ok [ { Source.rows = None; open_ = reader r } ])
  in
  (Query.of_source source, opens)

let small_left =
  Talon.v
    [
      ("g", Column.v Type.int64 [| 0; 1 |]);
      ("t", Column.v Type.int64 [| 2; 5 |]);
    ]

let small_right =
  Talon.v
    [
      ("g", Column.v Type.int64 [| 0; 1; 0 |]);
      ("rt", Column.v Type.int64 [| 1; 4; 3 |]);
      ("v", Column.v Type.int64 [| 10; 11; 12 |]);
    ]

let () =
  let pandas_rows =
    List.concat_map
      (fun (d, name) ->
        [ ((d, "1"), name ^ " exact"); ((d, "0"), name ^ " strict") ])
      directions
  in
  exit
    (run "Asof"
       [
         group "Kit.join_asof"
           [
             prop ~count:500 ~examples:[ nearest_tie ]
               "pairs each left row with its pick" (Gen.with_pp pp frame) law;
             test "its columns are left's, then right's without by" (fun () ->
                 let got =
                   exec
                     (Kit.join_asof ~by:[ "g" ] ~on:("t", "rt")
                        (Query.of_table small_right)
                        (Query.of_table small_left))
                 in
                 equal (list string) [ "g"; "t"; "rt"; "v" ]
                   (List.map fst (Schema.columns (Talon.schema got))));
             distances;
             test
               "a span past duration's range between two picks is a data error"
               (fun () ->
                 let right =
                   Talon.v [ ("j", places 2); ("rt", Column.v far_ns far) ]
                 in
                 let left =
                   Talon.v [ ("t", Column.v far_ns [| Time.of_ns 100L |]) ]
                 in
                 ignore
                   (require_error
                      (Query.run
                         (Kit.join_asof ~direction:Nearest ~on:("t", "rt")
                            (Query.of_table right) (Query.of_table left)))));
             cases "agrees with pandas' merge_asof" ~id:snd pandas_rows
               (fun (row, _) -> pandas row);
             cases "reads each side at most twice" ~id:snd directions
               (fun (direction, _) ->
                 let left, lopens = counted small_left in
                 let right, ropens = counted small_right in
                 ignore
                   (exec
                      (Kit.join_asof ~by:[ "g" ] ~direction ~on:("t", "rt")
                         right left));
                 at_most int ~than:2 !lopens;
                 at_most int ~than:2 !ropens);
             test "a name on both sides is the join's problem" (fun () ->
                 let right =
                   Query.of_table
                     (Talon.v [ ("t", Column.v Type.int64 [| 1 |]) ])
                 in
                 let left =
                   Query.of_table
                     (Talon.v [ ("t", Column.v Type.int64 [| 2 |]) ])
                 in
                 raises_match (Exn.invalid_arg ~sub:"join") (fun () ->
                     Kit.join_asof ~on:("t", "t") right left));
             (let q = Query.of_table small_left in
              let r = Query.of_table small_right in
              let nearest_of l rc =
                Kit.join_asof ~direction:Nearest ~on:("t", "rt")
                  (Query.of_table (Talon.v [ ("rt", rc) ]))
                  (Query.of_table (Talon.v [ ("t", l) ]))
              in
              cases "raises Invalid_argument" ~id:fst
                [
                  ( "if by names a column twice",
                    fun () -> Kit.join_asof ~by:[ "g"; "g" ] ~on:("t", "rt") r q
                  );
                  ( "for Nearest over strings",
                    fun () ->
                      nearest_of
                        (Column.v Type.string [| "b" |])
                        (Column.v Type.string [| "a" |]) );
                  ( "for Nearest over booleans",
                    fun () ->
                      nearest_of
                        (Column.v Type.bool [| true |])
                        (Column.v Type.bool [| false |]) );
                ]
                (fun (_, f) ->
                  raises_match (Exn.invalid_arg ~sub:"Kit.join_asof") f));
           ];
       ])
