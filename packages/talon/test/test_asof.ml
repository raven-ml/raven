(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

(* Frames

   A drawn frame is a left table of rows [i], a group [g] and a key [t], and a
   right table of rows [j], a group [g], a key [rt] and a value [v], in no
   order. Keys of one kind draw from few values, so that ties, extremes and
   rows without a match come often. *)

type coord = I of int | F of float | S of string

type kind =
  | Int8 | Uint8 | Int | Float32 | Float | Date | Datetime | Duration | Clock
  | String | Bool

type frame = {
  kind : kind;
  keyed : bool;
  direction : Kit.direction;
  strict : bool;
  lg : int option array;
  lt : coord option array;
  rg : int option array;
  rt : coord option array;
  v : int option array;
}

let kinds =
  [ (Int8, "int8"); (Uint8, "uint8"); (Int, "int64"); (Float32, "float32");
    (Float, "float64"); (Date, "date"); (Datetime, "datetime");
    (Duration, "duration"); (Clock, "clock"); (String, "string");
    (Bool, "bool") ]

let directions = Kit.[ (Backward, "backward"); (Forward, "forward"); (Nearest, "nearest") ]

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
  let key = of_list (coords kind) and group = int_range 0 1 in
  let* lg, lt = pair (rows n group) (rows n key) in
  let* rg, rt = pair (rows m group) (rows m key) in
  let* v = rows m (int_range 0 99) in
  (* Strings and booleans have no distance. *)
  let measured = not (List.mem kind [ String; Bool ]) in
  (* Nearest comes twice as often, so that its distance ties come often. *)
  let weight (d, _) = if d <> Kit.Nearest then 1 else if measured then 2 else 0 in
  let* direction = frequency (List.map (fun d -> (weight d, constant (fst d))) directions) in
  let+ keyed, strict = pair bool bool in
  { kind; keyed; direction; strict; lg; lt; rg; rt; v }

let column kind cs =
  let v ty f = Column.of_options ty (Array.map (Option.map f) cs) in
  let int = function I i -> i | _ -> assert false in
  let span c = Time.Span.of_ns (Int64.of_int (int c)) in
  match kind with
  | Int8 -> v Type.int8 int
  | Uint8 -> v Type.uint8 int
  | Int -> v Type.int64 int
  | Float32 -> v Type.float32 (function F x -> x | _ -> assert false)
  | Float -> v Type.float64 (function F x -> x | _ -> assert false)
  | Date -> v Type.date (fun c -> Option.get (Time.Date.of_days (int c)))
  | Datetime ->
      v (Type.datetime ~zone:"UTC" Type.Ns) (fun c -> Time.of_ns (Int64.of_int (int c)))
  | Duration -> v (Type.duration Type.Ns) span
  | Clock -> v (Type.clock Type.Ns) span
  | String -> v Type.string (function S s -> s | _ -> assert false)
  | Bool -> v Type.bool (fun c -> int c = 1)

let ints a = Column.of_options Type.int64 a
let places n = Column.v Type.int64 (Array.init n Fun.id)
let exec q = Error.get_ok (Query.run q)
let ints_of t c = Column.options Kind.int (Talon.column t c)

let tables f =
  let n = Array.length f.lt and m = Array.length f.rt in
  let g = if f.keyed then [ ("g", ints f.rg) ] else [] in
  ( Query.of_table
      (Talon.v ~rows:n [ ("i", places n); ("g", ints f.lg); ("t", column f.kind f.lt) ]),
    Query.of_table
      (Talon.v ~rows:m
         ((("j", places m) :: g) @ [ ("rt", column f.kind f.rt); ("v", ints f.v) ])) )

let pp ppf f =
  let left, right = tables f in
  Format.fprintf ppf "%s %s strict=%b by=%b@.%a@.%a" (List.assoc f.kind kinds)
    (List.assoc f.direction directions) f.strict f.keyed Talon.pp (exec left)
    Talon.pp (exec right)

(* The reference: [Join]'s pairs, and a selection per left row *)

(* [matches f cmp] is, for each left row, the right rows that [Join] pairs it
   with on [cmp "t" "rt"] within its group, in right order, each with its
   key's rank. *)
let matches f cmp =
  let left, right = tables f in
  let ranks = Expr.{ column = (fun _ x -> "r" := rank x) } in
  let ranked = Query.derive Expr.[ each (Sel.names [ "rt" ]) ranks ] right in
  let on = if f.keyed then Join.(keys [ "g" ] && cmp "t" "rt") else cmp "t" "rt" in
  let pairs = exec (Query.join ~on ranked left) in
  let i = ints_of pairs "i" and j = ints_of pairs "j" and r = ints_of pairs "r" in
  let of_row row p = if i.(p) = Some row then Some (Option.get j.(p), Option.get r.(p)) else None in
  Array.init (Array.length f.lt) (fun row ->
      List.filter_map (of_row row) (List.init (Array.length i) Fun.id))

(* [pick ~last ms] is the match of greatest rank, the last among ties, or with
   [~last:false] the first of least rank. *)
let pick ~last ms =
  let better (_, r) (_, rb) = if last then r >= rb else r < rb in
  let keep best m = match best with Some b when not (better m b) -> best | _ -> Some m in
  Option.map fst (List.fold_left keep None ms)

(* Talon's total order: NaN after every number, [-0.] equal to [0.]. *)
let same a b =
  match (a, b) with
  | F a, F b when Float.is_nan a || Float.is_nan b -> Float.is_nan a && Float.is_nan b
  | a, b -> a = b

(* [dist l c] is the exact distance between [l] and [c], integers subtracting
   in 64 bits unsigned: [0] between equal values, infinite where either is
   infinite or NaN. *)
let dist l c =
  match (l, c) with
  | I l, I c -> Int64.(sub (of_int (Int.max l c)) (of_int (Int.min l c)))
  | _ when same l c -> 0L
  | F l, F c when Float.is_finite l && Float.is_finite c ->
      Int64.bits_of_float (Float.abs (l -. c))
  | _ -> Int64.bits_of_float infinity

let reference f =
  let back = Array.map (pick ~last:true) (matches f (if f.strict then Join.gt else Join.ge)) in
  let fwd = Array.map (pick ~last:false) (matches f (if f.strict then Join.lt else Join.le)) in
  let nearest i a b =
    match (a, b) with
    | Some a, Some b ->
        let l = Option.get f.lt.(i) and key j = Option.get f.rt.(j) in
        (* Positive floats order as their bits. *)
        Some (if Int64.unsigned_compare (dist l (key a)) (dist l (key b)) <= 0 then a else b)
    | a, None -> a
    | None, b -> b
  in
  match f.direction with
  | Backward -> back
  | Forward -> fwd
  | Nearest -> Array.init (Array.length back) (fun i -> nearest i back.(i) fwd.(i))

let odd = function F x -> Float.is_nan x || Float.abs x = infinity | _ -> false

let asof_law f =
  let left, right = tables f in
  let by = if f.keyed then [ "g" ] else [] in
  let q = Kit.join_asof ~by ~direction:f.direction ~strict:f.strict ~on:("t", "rt") right left in
  let got = exec q and want = reference f in
  let n = Array.length f.lt in
  let keys = List.filter_map Fun.id (Array.to_list f.rt) in
  (* Another key of the left row's group lies at the picked key's distance. *)
  let tie i =
    match (f.lt.(i), want.(i)) with
    | Some l, Some j ->
        let picked = Option.get f.rt.(j) in
        let rival k c =
          ((not f.keyed) || f.rg.(k) = f.lg.(i))
          && (not (same c picked)) && dist l c = dist l picked
        in
        Array.exists Fun.id (Array.mapi (fun k c -> Option.fold ~none:false ~some:(rival k) c) f.rt)
    | _ -> false
  in
  cover (List.assoc f.kind kinds) true;
  cover (List.assoc f.direction directions) true;
  cover "strict" f.strict;
  cover "tied right keys" (List.length (List.sort_uniq compare keys) < List.length keys);
  cover "a null key" (Array.exists Option.is_none f.lt || Array.exists Option.is_none f.rt);
  cover "a NaN or infinite key" (List.exists odd keys);
  cover "an empty side" (n = 0 || Array.length f.rt = 0);
  cover "a left row without a match" (Array.exists Option.is_none want);
  cover "a nearest distance tie" (f.direction = Kit.Nearest && List.exists tie (List.init n Fun.id));
  equal (array (option int)) (Array.init n Option.some) (ints_of got "i");
  equal (array (option int)) want (ints_of got "j");
  let v = Array.map (Option.fold ~none:None ~some:(Array.get f.v)) want in
  equal (array (option int)) v (ints_of got "v")

(* Exact distances *)

let nearest l rs =
  let left = Talon.v [ ("t", l) ] in
  let right = Talon.v [ ("j", places (Column.length rs)); ("rt", rs) ] in
  ints_of (exec Query.(Kit.join_asof ~direction:Nearest ~on:("t", "rt") (of_table right) (of_table left))) "j"

let exact () =
  let one ?msg ty l rs want =
    equal ?msg (array (option int)) [| Some want |] (nearest (Column.v ty [| l |]) (Column.v ty rs))
  in
  let bits dt xs = Column.of_tensor (Nx.bitcast dt (Nx.create Nx.int64 [| Array.length xs |] xs)) in
  let p53 = Float.ldexp 1. 53 in
  (* [0.5 - -2^53] and [2^53 - 0.5] both round to 2^53. *)
  one ~msg:"float64" Type.float64 0.5 [| -.p53; p53 |] 1;
  one ~msg:"float64, mirrored" Type.float64 (-0.5) [| -.p53; p53 |] 0;
  (* The distances wrap in their type. *)
  one ~msg:"int8" Type.int8 0 [| -128; 127 |] 1;
  let int64 l = equal ~msg:"int64" (array (option int)) [| Some (if l = 0L then 1 else 0) |]
      (nearest (bits Nx.int64 [| l |]) (bits Nx.int64 Int64.[| min_int; max_int |])) in
  int64 0L; int64 (-1L);
  (* 2^63 and 2^63 - 1 against 0 and 2^64 - 1. *)
  let uint64 l want = equal ~msg:"uint64" (array (option int)) [| Some want |]
      (nearest (bits Nx.uint64 [| l |]) (bits Nx.uint64 [| 0L; -1L |])) in
  uint64 Int64.min_int 1; uint64 Int64.max_int 0;
  (* A lone pick needs no distance, however far. *)
  let ns = Type.datetime ~zone:"UTC" Type.Ns in
  one ~msg:"a far datetime" ns (Time.of_ns Int64.(sub max_int 10L)) [| Time.of_ns Int64.(add min_int 10L) |] 0;
  (* With both picks, a span past duration[ns]'s range fails the run. *)
  let far = Talon.v [ ("j", places 2); ("rt", Column.v ns Time.[| of_ns Int64.(add min_int 10L); of_ns Int64.(sub max_int 10L) |]) ] in
  let near = Talon.v [ ("t", Column.v ns [| Time.of_ns 100L |]) ] in
  let q = Query.(Kit.join_asof ~direction:Nearest ~on:("t", "rt") (of_table far) (of_table near)) in
  equal ~msg:"a span out of range" bool true (Result.is_error (Query.run q));
  raises_match ~msg:"a string key" (Exn.invalid_arg ~sub:"Kit.join_asof") (fun () ->
      nearest (Column.v Type.string [| "b" |]) (Column.v Type.string [| "a" |]))

(* Oracles

   golden/asof.txt holds the right rows that pandas' merge_asof picks over
   sorted integer keys, which Polars' join_asof agrees with but on nearest
   ties. *)

let golden =
  let lines = In_channel.with_open_text "golden/asof.txt" In_channel.input_lines in
  List.map (String.split_on_char ' ') lines

let side name key index =
  let rows = List.filter_map (function s :: r when s = name -> Some r | _ -> None) golden in
  let col i = Array.of_list (List.map (fun r -> int_of_string (List.nth r i)) rows) in
  Query.of_table
    (Talon.v
       [ ("g", Column.v Type.int64 (col 0)); (key, Column.v Type.int64 (col 1));
         (index, places (List.length rows)) ])

let oracle (direction, exact, picks) =
  let direction = fst (List.find (fun (_, n) -> n = direction) directions) in
  let q =
    Kit.join_asof ~by:[ "g" ] ~direction ~strict:(exact = "0") ~on:("t", "rt")
      (side "right" "rt" "j") (side "left" "t" "i")
  in
  let want = List.map (function "null" -> None | j -> Some (int_of_string j)) picks in
  equal (array (option int)) (Array.of_list want) (ints_of (exec q) "j")

let () =
  let modes = List.filter_map (function "pick" :: d :: e :: js -> Some (d, e, js) | _ -> None) golden in
  exit
    (run "asof"
       [
         prop ~count:500 "the row each left row's matches select" (Gen.with_pp pp frame) asof_law;
         test "distances are exact" exact;
         cases "pandas' merge_asof" ~id:(fun (d, e, _) -> d ^ " exact=" ^ e) modes oracle;
       ])
