(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module G = Talon_gen

let table_w =
  let pp ppf t =
    Format.fprintf ppf "%a: %d rows" Schema.pp (schema t) (rows t)
  in
  Testable.make ~pp ~equal:Talon.equal

let one c = v [ ("a", c) ]
let nulls vs = Array.fold_left (fun n v -> if v = None then n + 1 else n) 0 vs

let message f =
  match f () with _ -> fail "no exception" | exception Invalid_argument m -> m

(* [rows_are ty vs c] asserts that [c] holds [vs]; a column no kind reads, its
   nulls. *)
let rows_are (type a) (ty : a Type.t) (vs : a option array) c =
  let k = Type.kind ty in
  match Kind.provably_equal k k with
  | Some _ -> equal (array (option (G.witness ty))) vs (Column.options k c)
  | None ->
      equal int (Array.length vs) (Column.length c);
      equal int (nulls vs) (Column.null_count c)

(* A sample, and a table of it cut into batches. *)
let split =
  Gen.bind G.sample (fun (G.Sample (ty, vs) as s) ->
      Gen.pair
        (Gen.constant ~pp:G.pp_sample s)
        (G.split (one (Column.of_options ty vs))))

(* Rows and their indices *)

let column_of_split (G.Sample (ty, vs), t) =
  cover "several batches" (List.length (batches t) > 1);
  rows_are ty vs (column t "a")

let indices n =
  if n = 0 then Gen.constant [||]
  else Gen.array ~size:(Gen.int_range 0 50) (Gen.int_range 0 (n - 1))

let take_of_split =
  let gen =
    Gen.bind split (fun ((_, t) as s) ->
        Gen.pair (Gen.constant s) (indices (rows t)))
  in
  prop "take reads the rows at its indices, as one batch" gen
    (fun ((G.Sample (ty, vs), t), idx) ->
      let r =
        take
          (Nx.create Nx.int64
             [| Array.length idx |]
             (Array.map Int64.of_int idx))
          t
      in
      at_most int ~than:1 (List.length (batches r));
      rows_are ty (Array.map (Array.get vs) idx) (column r "a"))

(* Equality *)

(* Two rows of one type, the second's rows drawn from the first's. *)
type pair = Pair : 'a Type.t * 'a option array * 'a option array -> pair

let pp_pair ppf (Pair (ty, vs0, vs1)) =
  Format.fprintf ppf "@[<v>%a@,%a@]" G.pp_sample
    (G.Sample (ty, vs0))
    G.pp_sample
    (G.Sample (ty, vs1))

let pairs =
  let draw (G.Sample (ty, vs)) =
    let n = Array.length vs in
    let pair idx = Pair (ty, vs, Array.map (Array.get vs) idx) in
    let idx = Gen.array ~size:(Gen.constant n) (Gen.int_range 0 (n - 1)) in
    if n = 0 then Gen.constant (pair [||])
    else
      Gen.frequency
        [
          (1, Gen.constant (pair (Array.init n Fun.id))); (3, Gen.map pair idx);
        ]
  in
  Gen.with_pp pp_pair (Gen.bind G.sample draw)

let table ty vs = one (Column.of_options ty vs)

let key_identity (Pair (ty, vs0, vs1)) =
  let same = Option.equal (fun a b -> Type.compare_value ty a b = 0) in
  let expected = Array.for_all2 same vs0 vs1 in
  cover "equal" expected;
  cover "unequal" (not expected);
  equal bool expected (Talon.equal (table ty vs0) (table ty vs1))

let equality =
  group "Equality"
    [
      prop "equal is key identity row by row" pairs key_identity;
      prop "equal is an equivalence" pairs (fun (Pair (ty, vs0, vs1)) ->
          Law.equivalence table_w (table ty vs0, table ty vs1));
      prop "equal ignores batches" split (fun (G.Sample (ty, vs), t) ->
          equal table_w (table ty vs) t);
      test "equal makes -0. the key of 0. and every NaN one key" (fun () ->
          let t xs = one (Column.v Type.float64 xs) in
          equal table_w (t [| 0.; Float.nan |]) (t [| -0.; -.Float.nan |]);
          equal bool false (Talon.equal (t [| -0. |]) (t [| 1. |])));
      test "equal tells a null element or field from a value" (fun () ->
          let list vs =
            let child = Column.of_options Type.uint8 vs in
            let offsets = Nx.create Nx.int64 [| 2 |] [| 0L; 1L |] in
            let l = Column.Varsize { validity = None; offsets; child } in
            one
              (Result.get_ok (Column.of_layout (Any (Type.list Type.uint8)) l))
          in
          equal bool false (Talon.equal (list [| None |]) (list [| Some 0 |]));
          let record v =
            let ty = Type.record [ ("a", Any Type.bool) ] in
            one (Column.v ty [| Record.(add Kind.bool "a" v empty) |])
          in
          equal bool false (Talon.equal (record None) (record (Some false))));
      test "equal tells a null from a value and other schemas apart" (fun () ->
          let t vs = one (Column.of_options Type.bool vs) in
          equal bool false (Talon.equal (t [| None |]) (t [| Some false |]));
          equal bool false
            (Talon.equal (t [| Some false |])
               (v [ ("b", Column.v Type.bool [| false |]) ])));
    ]

(* Canonical columns *)

let hex (type a b) (x : (a, b) Nx.t) =
  let bytes =
    match Nx.dtype x with
    | Bool -> Nx.cast Nx.uint8 x
    | _ -> Nx.flatten (Nx.bitcast Nx.uint8 x)
  in
  String.concat ""
    (List.map (Printf.sprintf "%02x")
       (Array.to_list (Nx.to_array (Nx.flatten bytes))))

(* The bytes of every buffer of [c]'s layout. *)
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
      :: List.concat_map (fun (n, c) -> n :: buffers c) fields

let shifted_bits =
  Option.map (fun b ->
      let pad = Nx.ones Nx.bit [| 3 |] in
      let m = Nx.concatenate ~axis:0 [ pad; b; pad ] in
      Nx.shrink [| (3, 3 + Nx.numel b) |] m)

(* [c] with one more row in front. *)
let prepend c =
  let (Type.Any ty) = Column.type_ c in
  let first : Column.t =
    match ty with
    | Uint8 -> Column.v Type.uint8 [| 7 |]
    | _ -> Column.of_options ty [| None |]
  in
  column (of_batches [ one first; one c ]) "a"

(* [view c] holds [c]'s rows in buffers that are views: a validity at a bit
   offset with bits set past it, values inside larger storage, offsets from
   1. *)
let rec view c =
  let l : Column.layout =
    match Column.layout c with
    | Fixed { validity; values = P x } ->
        let ends i _ = if i = 0 then (1, 1) else (0, 0) in
        let padded =
          Nx.pad (Array.mapi ends (Nx.shape x)) (Nx_dtype.zero (Nx.dtype x)) x
        in
        let inner i d = if i = 0 then (1, d - 1) else (0, d) in
        let values =
          Nx.P (Nx.shrink (Array.mapi inner (Nx.shape padded)) padded)
        in
        Fixed { validity = shifted_bits validity; values }
    | Varsize { validity; offsets; child } ->
        Varsize
          {
            validity = shifted_bits validity;
            offsets = Nx.add_s offsets 1L;
            child = view (prepend child);
          }
    | Children { validity; length; fields } ->
        Children
          {
            validity = shifted_bits validity;
            length;
            fields = List.map (fun (n, c) -> (n, view c)) fields;
          }
  in
  Result.get_ok (Column.of_layout (Column.type_ c) l)

let canonical (G.Sample (ty, vs), t) =
  let c = Column.of_options ty vs in
  cover "a null" (nulls vs > 0);
  equal (list string) (buffers c) (buffers (column (one (view c)) "a"));
  equal (list string) (buffers c) (buffers (column t "a"));
  equal table_w (one c) (one (view c))

let canonical_columns =
  group "Canonical columns"
    [
      prop "column's layout is the same bytes whatever the buffers and batches"
        split canonical;
      test "column of one batch of a canonical column is that column" (fun () ->
          let c = Column.of_options Type.string [| Some "é"; None |] in
          satisfies ~claim:"the same column" pass
            (fun c' -> c' == c)
            (column (one c) "a"));
    ]

(* Tensors *)

let numbers =
  v
    [
      ("i", Column.v Type.int8 [| 1; -2 |]);
      ("f", Column.v Type.float32 [| 0.5; 3. |]);
      ("b", Column.v Type.bool [| true; false |]);
      ("s", Column.v Type.string [| "a"; "b" |]);
      ("n", Column.of_options Type.int8 [| Some 1; None |]);
    ]

let tensors =
  group "Tensors"
    [
      test "to_tensor stacks the named columns, cast, across batches" (fun () ->
          let row i = take (Nx.create Nx.int64 [| 1 |] [| i |]) numbers in
          let t = of_batches [ row 0L; row 1L ] in
          equal (array float_exact)
            [| 0.5; 1.; 1.; 3.; -2.; 0. |]
            (Nx.to_array (to_tensor Nx.float64 [ "f"; "i"; "b" ] t)));
      test "to_tensor of no column has the rows" (fun () ->
          equal (array int) [| 2; 0 |]
            (Nx.shape (to_tensor Nx.int32 [] numbers)));
    ]

(* Refusals *)

let refuse name f expected = (name, (fun () -> ignore (f ())), expected)
let at is = Nx.create Nx.int64 [| Array.length is |] is

let refusals =
  [
    refuse "v of columns of other lengths" (fun () ->
        v [ ("a", Column.v Type.int8 [| 1 |]); ("b", Column.v Type.int8 [||]) ])
    @@ __POS_OF__ {| Talon.v: column "b" has 0 rows, not 1 |};
    refuse "v of no column without rows" (fun () -> v [])
    @@ __POS_OF__ {| Talon.v: no column and no rows |};
    refuse "v of rows other than the columns'" (fun () ->
        v ~rows:2 [ ("a", Column.v Type.int8 [| 1 |]) ])
    @@ __POS_OF__ {| Talon.v: column "a" has 1 rows, not 2 |};
    refuse "v of negative rows" (fun () -> v ~rows:(-1) [])
    @@ __POS_OF__ {| Talon.v: rows is -1, below 0 |};
    refuse "v of a duplicate name" (fun () ->
        v [ ("a", Column.v Type.int8 [||]); ("a", Column.v Type.int8 [||]) ])
    @@ __POS_OF__ {| Talon.v: duplicate column "a" |};
    refuse "v of a name that is not UTF-8" (fun () ->
        v [ ("\xff", Column.v Type.int8 [||]) ])
    @@ __POS_OF__ {| Talon.v: column name "\255" is not UTF-8 |};
    refuse "of_batches of no table" (fun () -> of_batches [])
    @@ __POS_OF__ {| Talon.of_batches: no table |};
    refuse "of_batches of other schemas" (fun () ->
        of_batches [ numbers; one (Column.v Type.int8 [||]) ])
    @@ __POS_OF__
         {| Talon.of_batches: schemas i int8, f float32, b bool, s string, n int8 and a int8 differ |};
    refuse "column of a missing name" (fun () -> column numbers "x")
    @@ __POS_OF__ {| Talon.column: no column "x" |};
    refuse "take past the rows" (fun () -> take (at [| 0L; 2L |]) numbers)
    @@ __POS_OF__ {| Talon.take: index 2 of a table of 2 rows |};
    refuse "take before the rows" (fun () -> take (at [| -1L |]) numbers)
    @@ __POS_OF__ {| Talon.take: index -1 of a table of 2 rows |};
    refuse "take of 2-D indices" (fun () ->
        take (Nx.zeros Nx.int64 [| 1; 1 |]) numbers)
    @@ __POS_OF__ {| Talon.take: indices of shape [1,1], not 1-D |};
    refuse "to_tensor of text" (fun () -> to_tensor Nx.float32 [ "s" ] numbers)
    @@ __POS_OF__
         {| Talon.to_tensor: "s" is string, neither numeric nor boolean |};
    refuse "to_tensor of a null" (fun () ->
        to_tensor Nx.float32 [ "n" ] numbers)
    @@ __POS_OF__ {| Talon.to_tensor: "n" has a null |};
    refuse "to_tensor of a missing name" (fun () ->
        to_tensor Nx.float32 [ "x" ] numbers)
    @@ __POS_OF__ {| Talon.to_tensor: no column "x" |};
  ]

let refusal_cases =
  cases
    ~name:(fun (n, _, _) -> n)
    "Refusals" refusals
    (fun (_, f, expected) -> expect (message f) expected)

let () =
  exit
    (run "Table"
       [
         group "Rows"
           [
             prop "column reads the rows of every split" split column_of_split;
             take_of_split;
             test "a table of rows without columns keeps them through a run"
               (fun () ->
                 let t = v ~rows:3 [] in
                 equal (pair int int) (3, 2)
                   ( rows (Error.get_ok (Query.run (Query.of_table t))),
                     rows
                       (Error.get_ok
                          (Query.run
                             (Query.slice ~offset:1 ~length:5 (Query.of_table t))))
                   ));
             test "a table without rows has no batch" (fun () ->
                 equal int 0
                   (List.length (batches (one (Column.v Type.int8 [||])))));
           ];
         equality;
         canonical_columns;
         tensors;
         refusal_cases;
       ])
