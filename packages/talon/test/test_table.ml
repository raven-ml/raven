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

let schema_w = Testable.make ~pp:Schema.pp ~equal:Schema.equal
let one c = v [ ("a", c) ]
let nulls vs = Array.fold_left (fun n v -> if v = None then n + 1 else n) 0 vs

(* [refusal f] is the message of the [Invalid_argument] that [f] raises. *)
let refusal f =
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
  prop "is the rows at its indices, as one batch" gen
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

(* A filter that keeps almost all rows copies whole runs of them, and must read
   as a gather of each kept row does, for every column type. *)
let filter_runs =
  let gen =
    Gen.bind G.sample (fun (G.Sample (_, vs) as s) ->
        let n = Array.length vs in
        Gen.bind (Gen.int_range 64 400) (fun rows ->
            let drops =
              if n = 0 then Gen.constant []
              else
                Gen.list
                  ~size:(Gen.int_range 0 (rows / 32))
                  (Gen.int_range 0 (rows - 1))
            in
            Gen.map (fun d -> (s, rows, d)) drops))
    |> Gen.with_pp (fun ppf (s, rows, drops) ->
        Format.fprintf ppf "@[<v>%a@,%d rows, dropped %a@]" G.pp_sample s rows
          (Format.pp_print_list
             ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ",")
             Format.pp_print_int)
          drops)
  in
  prop "keeps each row of long runs, as take of their rows does" gen
    (fun (G.Sample (ty, vs), rows, drops) ->
      let n = Array.length vs in
      let rows = if n = 0 then 0 else rows in
      let row i = vs.(i mod n) in
      let dropped = Array.make rows false in
      List.iter (fun d -> dropped.(d) <- true) drops;
      let keep = Array.init rows (fun i -> not dropped.(i)) in
      let kept = List.filter (fun i -> keep.(i)) (List.init rows Fun.id) in
      let breaks =
        List.length
          (List.filter
             (fun i -> keep.(i) <> keep.(i - 1))
             (List.init (max 0 (rows - 1)) succ))
      in
      cover "runs long enough to copy whole"
        (kept <> [] && List.length kept / (1 + (breaks / 2)) >= 16);
      cover "a row dropped" (drops <> []);
      cover "no row dropped" (drops = []);
      let table =
        v
          [
            ("a", Column.of_options ty (Array.init rows row));
            ("k", Column.of_options Type.bool (Array.map Option.some keep));
          ]
      in
      let r =
        Error.get_ok
          (Query.run (Query.filter (Col.bool "k") (Query.of_table table)))
      in
      rows_are ty (Array.of_list (List.map row kept)) (column r "a");
      let positions =
        Nx.create Nx.int64
          [| List.length kept |]
          (Array.of_list (List.map Int64.of_int kept))
      in
      rows_are ty
        (Array.of_list (List.map row kept))
        (column (take positions table) "a"))

(* Indices that read runs of rows out of order take the rows at them. *)
let take_runs =
  let gen =
    Gen.bind G.sample (fun (G.Sample (_, vs) as s) ->
        let n = max 1 (Array.length vs) in
        let run =
          Gen.pair (Gen.int_range 0 (n - 1)) (Gen.int_range 1 (max 1 n))
        in
        Gen.map (fun r -> (s, r)) (Gen.list ~size:(Gen.int_range 1 8) run))
    |> Gen.with_pp (fun ppf (s, _) -> G.pp_sample ppf s)
  in
  prop "reads runs of rows out of order" gen (fun (G.Sample (ty, vs), runs) ->
      let n = Array.length vs in
      let at =
        if n = 0 then []
        else
          List.concat_map
            (fun (first, len) ->
              List.init (min len (n - first)) (fun k -> first + k))
            runs
      in
      cover "long enough to cut into runs" (List.length at >= 16);
      let rec backwards = function
        | a :: (b :: _ as rest) -> b <= a || backwards rest
        | _ -> false
      in
      cover "indices that go backwards" (backwards at);
      let t = one (Column.of_options ty vs) in
      let positions =
        Nx.create Nx.int64
          [| List.length at |]
          (Array.of_list (List.map Int64.of_int at))
      in
      rows_are ty
        (Array.of_list (List.map (Array.get vs) at))
        (column (take positions t) "a"))

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
  let t0 = table ty vs0 and t1 = table ty vs1 in
  if expected then equal table_w t0 t1 else not_equal table_w t0 t1

let equality =
  [
    prop "is key identity row by row" pairs key_identity;
    prop "is an equivalence" pairs (fun (Pair (ty, vs0, vs1)) ->
        Law.equivalence table_w (table ty vs0, table ty vs1));
    prop "ignores batches" split (fun (G.Sample (ty, vs), t) ->
        equal table_w (table ty vs) t);
    test "makes -0. the key of 0. and every NaN one key" (fun () ->
        let t xs = one (Column.v Type.float64 xs) in
        equal table_w (t [| 0.; Float.nan |]) (t [| -0.; -.Float.nan |]);
        not_equal table_w (t [| -0. |]) (t [| 1. |]));
    test "tells a null element or field from a value" (fun () ->
        let list vs =
          let child = Column.of_options Type.uint8 vs in
          let offsets = Nx.create Nx.int64 [| 2 |] [| 0L; 1L |] in
          let l = Column.Varsize { validity = None; offsets; child } in
          one (Result.get_ok (Column.of_layout (Any (Type.list Type.uint8)) l))
        in
        not_equal table_w (list [| None |]) (list [| Some 0 |]);
        let record v =
          let ty = Type.record [ ("a", Any Type.bool) ] in
          one (Column.v ty [| Record.(add Kind.bool "a" v empty) |])
        in
        not_equal table_w (record None) (record (Some false)));
    test "tells a null from a value and other schemas apart" (fun () ->
        let t vs = one (Column.of_options Type.bool vs) in
        not_equal table_w (t [| None |]) (t [| Some false |]);
        not_equal table_w (t [| Some false |])
          (v [ ("b", Column.v Type.bool [| false |]) ]));
  ]

(* Canonical columns *)

(* Elements narrower than a byte are read from their storage, the bits past
   their last element included. *)
let hex (type a b) (x : (a, b, Nx.host) Nx.t) =
  let bits = Nx.Dtype.bits (Nx.dtype x) in
  let bytes =
    match Nx.dtype x with
    | Bool -> Nx.to_array (Nx.flatten (Nx.cast Nx.uint8 x))
    | _ when bits < 8 ->
        Nx.Repr.read ~by:"test" x @@ fun a ->
        Rig.Buffer.read (Nx_array.buffer a) Bigarray.int8_unsigned (fun raw ->
            Array.init (((Nx.numel x * bits) + 7) / 8) (fun i -> raw.{i}))
    | _ -> Nx.to_array (Nx.flatten (Nx.bitcast Nx.uint8 x))
  in
  String.concat "" (List.map (Printf.sprintf "%02x") (Array.to_list bytes))

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
      Nx.slice [ Nx.R (3, 3 + Nx.numel b) ] m)

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
          Nx.pad (Array.mapi ends (Nx.shape x)) (Nx.Dtype.one (Nx.dtype x)) x
        in
        let values = Nx.P (Nx.slice [ Nx.R (1, -1) ] padded) in
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

(* Rows cut from the front of a larger tensor of elements narrower than a byte
   share their last byte with the rows after them. *)
let front_cuts =
  let first n x = Nx.slice [ Nx.R (0, n) ] x in
  (* [x] is copied into storage of its own: a filled tensor is a broadcast
     view. *)
  let cut name x n =
    let rows = first n (Nx.copy x) in
    (name, Column.of_tensor rows, Column.of_tensor (Nx.copy rows))
  in
  [
    cut "bool values" (Nx.ones Nx.bit [| 8 |]) 3;
    cut "bit cells" (Nx.ones Nx.bit [| 4; 3 |]) 1;
    cut "int4 cells" (Nx.full Nx.int4 [| 3; 1 |] 5) 1;
    cut "uint4 cells" (Nx.full Nx.uint4 [| 3; 1 |] 9) 1;
  ]

let canonical_columns =
  [
    cases "is its rows' own bytes when they are cut from packed storage"
      ~id:(fun (n, _, _) -> n)
      front_cuts
      (fun (_, cut, own) ->
        equal (list string)
          (buffers (column (one own) "a"))
          (buffers (column (one cut) "a")));
    prop "is the same bytes whatever the buffers and batches" split canonical;
    test "of one batch of a canonical column is that column" (fun () ->
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
  [
    test "stacks the named columns, cast, across batches" (fun () ->
        let row i = take (Nx.create Nx.int64 [| 1 |] [| i |]) numbers in
        let t = of_batches [ row 0L; row 1L ] in
        equal (array float_exact)
          [| 0.5; 1.; 1.; 3.; -2.; 0. |]
          (Nx.to_array (to_tensor Nx.float64 [ "f"; "i"; "b" ] t)));
    test "of no column has the rows" (fun () ->
        equal (array int) [| 2; 0 |] (Nx.shape (to_tensor Nx.int32 [] numbers)));
  ]

let at is = Nx.create Nx.int64 [| Array.length is |] is

(* Text rows of zero, one and several bytes, read at strided and broadcast
   indices: a view's positions, not its memory's. *)
let texts = v [ ("s", Column.v Type.string [| ""; "a"; "bcd"; "\xc3\xa9" |]) ]
let text_at idx = Column.options Kind.string (column (take idx texts) "s")

let text_views =
  [
    test "reads text at strided indices" (fun () ->
        let idx = Nx.slice [ Nx.Rs (1, 4, 2) ] (at [| 0L; 1L; 2L; 3L |]) in
        equal
          (array (option string))
          [| Some "a"; Some "\xc3\xa9" |]
          (text_at idx));
    test "reads text at broadcast indices" (fun () ->
        let idx = Nx.broadcast_to [| 3 |] (at [| 2L |]) in
        equal
          (array (option string))
          [| Some "bcd"; Some "bcd"; Some "bcd" |]
          (text_at idx));
  ]

(* Batches *)

(* A sample, and two tables of it cut into batches. *)
let two_splits =
  Gen.bind G.sample (fun (G.Sample (ty, vs) as s) ->
      let t = one (Column.of_options ty vs) in
      Gen.map (fun (a, b) -> (s, a, b)) (Gen.pair (G.split t) (G.split t)))
  |> Gen.with_pp (fun ppf (s, a, b) ->
      Format.fprintf ppf "@[<v>%a@,%d and %d batches@]" G.pp_sample s
        (List.length (batches a))
        (List.length (batches b)))

let batches_of_batches (_, a, b) =
  let r = of_batches [ a; b ] in
  cover "several batches each" (List.length (batches a) > 1);
  equal (list table_w) (batches a @ batches b) (batches r)

let () =
  exit
    (run "Table"
       [
         group "v"
           [
             test "gives a table without columns its rows" (fun () ->
                 equal int 3 (rows (v ~rows:3 [])));
             test "is one batch" (fun () ->
                 equal int 1
                   (List.length (batches (one (Column.v Type.int8 [| 1; 2 |])))));
             test
               "raises on columns of other lengths or rows, a duplicate name \
                or a name that is not UTF-8" (fun () ->
                 let int8s n = Column.v Type.int8 (Array.make n 0) in
                 expect
                   (refusal (fun () -> v [ ("a", int8s 1); ("b", int8s 0) ]))
                 @@ __POS_OF__ {| Talon.v: column "b" has 0 rows, not 1 |};
                 expect (refusal (fun () -> v []))
                 @@ __POS_OF__ {| Talon.v: no column and no rows |};
                 expect (refusal (fun () -> v ~rows:2 [ ("a", int8s 1) ]))
                 @@ __POS_OF__ {| Talon.v: column "a" has 1 rows, not 2 |};
                 expect (refusal (fun () -> v ~rows:(-1) []))
                 @@ __POS_OF__ {| Talon.v: rows is -1, below 0 |};
                 expect
                   (refusal (fun () -> v [ ("a", int8s 0); ("a", int8s 0) ]))
                 @@ __POS_OF__ {| Talon.v: duplicate column "a" |};
                 expect (refusal (fun () -> v [ ("\xff", int8s 0) ]))
                 @@ __POS_OF__ {| Talon.v: column name "\255" is not UTF-8 |});
           ];
         group "of_batches"
           [
             prop "has the rows of every split" split column_of_split;
             prop "has each table's batches, in order" two_splits
               batches_of_batches;
             test "raises on no table or tables of other schemas" (fun () ->
                 expect (refusal (fun () -> of_batches []))
                 @@ __POS_OF__ {| Talon.of_batches: no table |};
                 expect
                   (refusal (fun () ->
                        of_batches [ numbers; one (Column.v Type.int8 [||]) ]))
                 @@ __POS_OF__
                      {| Talon.of_batches: schemas i int8, f float32, b bool, s string, n int8 and a int8 differ |});
           ];
         group "batches"
           [
             test "of a table without rows is none" (fun () ->
                 equal int 0
                   (List.length (batches (one (Column.v Type.int8 [||])))));
           ];
         group "schema"
           [
             test "is the columns' names and types, in order" (fun () ->
                 let t =
                   v
                     [
                       ("b", Column.v Type.int8 [| 1 |]);
                       ("a", Column.v Type.string [| "x" |]);
                     ]
                 in
                 equal schema_w
                   (Schema.v [ ("b", Any Type.int8); ("a", Any Type.string) ])
                   (schema t));
           ];
         group "column"
           (canonical_columns
           @ [
               test "raises on a name the table lacks" (fun () ->
                   expect (refusal (fun () -> column numbers "x"))
                   @@ __POS_OF__ {| Talon.column: no column "x" |});
             ]);
         group "take"
           ([ take_of_split; take_runs ]
           @ text_views
           @ [
               test "raises on indices outside the rows or not 1-D" (fun () ->
                   expect (refusal (fun () -> take (at [| 0L; 2L |]) numbers))
                   @@ __POS_OF__ {| Talon.take: index 2 of a table of 2 rows |};
                   expect (refusal (fun () -> take (at [| -1L |]) numbers))
                   @@ __POS_OF__ {| Talon.take: index -1 of a table of 2 rows |};
                   expect
                     (refusal (fun () ->
                          take (Nx.zeros Nx.int64 [| 1; 1 |]) numbers))
                   @@ __POS_OF__
                        {| Talon.take: indices of shape [1; 1], not 1-D |});
             ]);
         group "to_tensor"
           (tensors
           @ [
               test
                 "raises on a column that is text, holds a null or is missing"
                 (fun () ->
                   expect
                     (refusal (fun () -> to_tensor Nx.float32 [ "s" ] numbers))
                   @@ __POS_OF__
                        {| Talon.to_tensor: "s" is string, neither numeric nor boolean |};
                   expect
                     (refusal (fun () -> to_tensor Nx.float32 [ "n" ] numbers))
                   @@ __POS_OF__ {| Talon.to_tensor: "n" has a null |};
                   expect
                     (refusal (fun () -> to_tensor Nx.float32 [ "x" ] numbers))
                   @@ __POS_OF__ {| Talon.to_tensor: no column "x" |});
             ]);
         group "equal" equality;
         group "Query.filter" [ filter_runs ];
         group "Query.run"
           [
             test "keeps the rows of a table without columns, through a slice"
               (fun () ->
                 let t = v ~rows:3 [] in
                 let run q = rows (Error.get_ok (Query.run q)) in
                 equal int 3 (run (Query.of_table t));
                 equal int 2
                   (run (Query.slice ~offset:1 ~length:5 (Query.of_table t))));
           ];
       ])
