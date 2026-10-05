(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module G = Talon_gen

let any_w =
  Testable.make
    ~pp:(fun ppf (Type.Any t) -> Type.pp ppf t)
    ~equal:(fun (Type.Any a) (Type.Any b) -> Type.equal a b)

let readable (type a) (ty : a Type.t) =
  Option.is_some (Kind.provably_equal (Type.kind ty) (Type.kind ty))

let rows ty = array (option (G.witness ty))
let nulls vs = List.length (List.filter Option.is_none (Array.to_list vs))
let rejects f = raises_match Exn.invalid_arg f

let message f =
  match f () with _ -> fail "no exception" | exception Invalid_argument m -> m

(* The codec *)

let has_ext_field fields =
  List.exists (function _, Type.Any (Type.Ext _) -> true | _ -> false) fields

let record_of_ext_record fields =
  let inner = function
    | _, Type.Any (Type.Record fs) -> has_ext_field fs
    | _ -> false
  in
  List.exists inner fields

let round_trip (G.Sample (ty, vs)) =
  cover "a null" (nulls vs > 0);
  cover "no row" (vs = [||]);
  cover "more rows than a builder starts with" (Array.length vs > 16);
  cover "nested" (match ty with List _ | Record _ -> true | _ -> false);
  cover "a record of a record with an extension field"
    (match ty with Record fs -> record_of_ext_record fs | _ -> false);
  cover "no kind reads it" (not (readable ty));
  if readable ty then
    Law.round_trip (rows ty) pass (Column.of_options ty)
      (Column.options (Type.kind ty))
      vs
  else
    rejects (fun () -> Column.options (Type.kind ty) (Column.of_options ty vs))

let observations (G.Sample (ty, vs)) =
  let c = Column.of_options ty vs in
  equal any_w (Any ty) (Column.type_ c);
  equal int (Array.length vs) (Column.length c);
  equal int (nulls vs) (Column.null_count c);
  match Column.validity c with
  | None -> equal int 0 (nulls vs)
  | Some b -> equal (array bool) (Array.map Option.is_some vs) (Nx.to_array b)

let without_nulls (G.Sample (ty, vs)) =
  let xs = Array.of_list (List.filter_map Fun.id (Array.to_list vs)) in
  if readable ty then begin
    let k = Type.kind ty and c = Column.v ty xs in
    equal (array (G.witness ty)) xs (Column.values k c);
    equal (rows ty) (Array.map Option.some xs) (Column.options k c)
  end

let codec =
  group "Codec"
    [
      prop "options reads back what of_options stores" G.sample round_trip;
      prop "a column has its rows' type, length, nulls and validity" G.sample
        observations;
      prop "values and options read back what v stores" G.sample without_nulls;
    ]

(* Values at the edges of their types *)

type edge = Edge : string * 'a Type.t * 'a option array -> edge

let instant ns = Time.of_ns ns
let span ns = Time.Span.of_ns ns
let day n = Option.get (Time.Date.of_days n)
let i32 = Int32.(to_int min_int, to_int max_int)

let edges =
  Type.
    [
      Edge ("an empty column", int8, [||]);
      Edge ("an empty column of a record", record [ ("a", Any string) ], [||]);
      Edge ("an all-null column", string, [| None; None |]);
      Edge ("an all-null extension column", ext ~name:"m" float64, [| None |]);
      Edge ("int8 bounds", int8, [| Some (-128); Some 127 |]);
      Edge ("int16 bounds", int16, [| Some (-32768); Some 32767 |]);
      Edge ("int32 bounds", int32, [| Some (fst i32); Some (snd i32) |]);
      Edge ("int64 bounds", int64, [| Some min_int; Some max_int |]);
      Edge ("uint8 bounds", uint8, [| Some 0; Some 255 |]);
      Edge ("uint16 bounds", uint16, [| Some 0; Some 65535 |]);
      Edge ("uint32 bounds", uint32, [| Some 0; Some 0xffff_ffff |]);
      Edge ("uint64 bounds", uint64, [| Some 0; Some max_int |]);
      Edge
        ( "float64 -0, NaN and infinities",
          float64,
          [|
            Some (-0.);
            Some Float.nan;
            Some Float.infinity;
            Some Float.neg_infinity;
          |] );
      Edge
        ( "float32 -0 and its largest finite value",
          float32,
          [| Some (-0.); Some 0x1.fffffep127 |] );
      Edge
        ( "float16 -0 and its largest finite value",
          float16,
          [| Some (-0.); Some 65504. |] );
      Edge
        ("date bounds", date, [| Some (day (fst i32)); Some (day (snd i32)) |]);
      Edge
        ( "clock bounds",
          clock S,
          [| Some (span 0L); Some (span 86_399_000_000_000L) |] );
      Edge
        ( "instant bounds in ns",
          datetime Ns,
          [| Some (instant Int64.min_int); Some (instant Int64.max_int) |] );
      Edge
        ( "instant bounds in s",
          datetime S,
          [|
            Some (instant (-9_223_372_036_000_000_000L));
            Some (instant 9_223_372_036_000_000_000L);
          |] );
      Edge ("empty text and NUL", string, [| Some ""; Some "\000" |]);
      Edge
        ( "empty lists and an empty element",
          list string,
          [| Some [||]; Some [| "" |] |] );
    ]

let edge_cases =
  cases
    ~name:(fun (Edge (n, _, _)) -> n)
    "Edges" edges
    (fun (Edge (_, ty, vs)) ->
      let c = Column.of_options ty vs in
      equal int (nulls vs) (Column.null_count c);
      if readable ty then equal (rows ty) vs (Column.options (Type.kind ty) c))

let float_bits =
  group "Float bits"
    [
      test "a NaN payload of float64 is kept" (fun () ->
          let nan = Int64.float_of_bits 0x7ff8_0000_dead_beefL in
          let back =
            Column.values Kind.float (Column.v Type.float64 [| nan |])
          in
          equal int64 0x7ff8_0000_dead_beefL (Int64.bits_of_float back.(0)));
      test "float32 and float16 round a value to their nearest, ties to even"
        (fun () ->
          let read ty x =
            (Column.values Kind.float (Column.v ty [| x |])).(0)
          in
          equal float_exact 0x1p-24 (read Type.float32 (0x1p-24 +. 0x1p-80));
          equal float_exact 2048. (read Type.float16 2049.);
          equal float_exact 2052. (read Type.float16 2051.));
      test "float16 rounds once, from the double" (fun () ->
          let read x =
            (Column.values Kind.float (Column.v Type.float16 [| x |])).(0)
          in
          equal float_exact 2050. (read (2049. +. 0x1p-14));
          equal float_exact 0x1p-24 (read (0x1p-25 +. 0x1p-60)));
      test "float32 holds values below its limit and refuses the limit"
        (fun () ->
          let below = Float.pred 0x1.ffffffp127 in
          equal float_exact 0x1.fffffep127
            (Column.values Kind.float (Column.v Type.float32 [| below |])).(0);
          rejects (fun () -> Column.v Type.float32 [| 0x1.ffffffp127 |]));
      prop "float32 stores a double's nearest float32, ties to even"
        (Gen.frequency
           [ (1, Gen.any_float); (3, Gen.float_range (-3.5e38) 3.5e38) ])
        (fun x ->
          assume (Type.holds Type.float32 x);
          let nearest = Int32.float_of_bits (Int32.bits_of_float x) in
          equal (G.witness Type.float32) nearest
            (Column.values Kind.float (Column.v Type.float32 [| x |])).(0));
    ]

(* Storage *)

let storage =
  group "Storage"
    [
      test "to_tensor is the storage of temporal and categorical values"
        (fun () ->
          let tensor dt ty vs =
            Nx.to_array (Column.to_tensor dt (Column.v ty vs))
          in
          equal (array int32) [| -1l; 2l |]
            (tensor Nx.int32 Type.date [| day (-1); day 2 |]);
          equal (array int64) [| -1L; 3L |]
            (tensor Nx.int64 (Type.datetime Type.Ms)
               [| instant (-1_000_000L); instant 3_000_000L |]);
          equal (array int32) [| 1l; 0l |]
            (tensor Nx.int32 (Type.categorical [| "x"; "y" |]) [| "y"; "x" |]);
          equal (array bool) [| true; false |]
            (tensor Nx.bool Type.bool [| true; false |]));
      test "to_tensor of a tensor column has the cells as rows" (fun () ->
          let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
          let c = Column.of_tensor x in
          equal any_w (Any (Type.tensor Nx.float32 [| 2 |])) (Column.type_ c);
          equal (array float_exact) [| 1.; 2.; 3.; 4. |]
            (Nx.to_array (Column.to_tensor Nx.float32 c)));
      test "to_tensor of of_tensor is the tensor itself" (fun () ->
          let x = Nx.create Nx.float64 [| 2 |] [| 1.; 2. |] in
          satisfies ~claim:"the same tensor" pass
            (fun y -> y == x)
            (Column.to_tensor Nx.float64 (Column.of_tensor x)));
      test "of_tensor reads strided, broadcast and transposed views" (fun () ->
          let x = Nx.create Nx.int32 [| 6 |] [| 0l; 1l; 2l; 3l; 4l; 5l |] in
          equal (array int) [| 0; 2; 4 |]
            (Column.values Kind.int
               (Column.of_tensor (Nx.slice [ Rs (0, 6, 2) ] x)));
          let seven =
            Nx.broadcast_to [| 3 |] (Nx.create Nx.int8 [| 1 |] [| 7 |])
          in
          equal (array int) [| 7; 7; 7 |]
            (Column.values Kind.int (Column.of_tensor seven));
          let m =
            Nx.create Nx.float32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |]
          in
          let rows =
            Column.values (Kind.tensor Nx.float32)
              (Column.of_tensor (Nx.transpose m))
          in
          equal
            (array (array float_exact))
            [| [| 0.; 3. |]; [| 1.; 4. |]; [| 2.; 5. |] |]
            (Array.map Nx.to_array rows));
      test "of_tensor reads a 1-D tensor as its dtype's scalar type" (fun () ->
          let c =
            Column.of_tensor (Nx.create Nx.uint32 [| 2 |] [| -1l; 7l |])
          in
          equal any_w (Any Type.uint32) (Column.type_ c);
          equal (array int) [| 0xffff_ffff; 7 |] (Column.values Kind.int c));
      test "of_tensor's validity makes nulls" (fun () ->
          let validity =
            Nx.cast Nx.bit (Nx.create Nx.bool [| 3 |] [| true; false; true |])
          in
          let c =
            Column.of_tensor ~validity (Nx.create Nx.int8 [| 3 |] [| 1; 2; 3 |])
          in
          equal int 1 (Column.null_count c);
          equal
            (array (option int))
            [| Some 1; None; Some 3 |]
            (Column.options Kind.int c));
      test "of_tensor keeps a validity with every row set, which has no null"
        (fun () ->
          let validity =
            Nx.cast Nx.bit (Nx.create Nx.bool [| 1 |] [| true |])
          in
          let c =
            Column.of_tensor ~validity (Nx.create Nx.int8 [| 1 |] [| 1 |])
          in
          is_some (Column.validity c);
          equal int 0 (Column.null_count c));
    ]

(* Offsets and validities *)

let n x = Nx.create Nx.int64 [| Array.length x |] x
let bits b = Nx.cast Nx.bit (Nx.create Nx.bool [| Array.length b |] b)

(* Records *)

let mass = Type.ext ~name:"m" Type.float64
let inner_ty = Type.record [ ("m", Any mass) ]

(* No public constructor builds a record with an extension field, so the record
   {m = null} is read from a column. *)
let m_null =
  let fields = [ ("m", Column.of_options mass [| None |]) ] in
  let l = Column.Children { validity = None; length = 1; fields } in
  let c = Result.get_ok (Column.of_layout (Any inner_ty) l) in
  (Column.values Record.kind c).(0)

let records =
  group "Records"
    [
      test "a record field holding a record with an extension field" (fun () ->
          let outer = Type.record [ ("r", Any inner_ty) ] in
          let r v = Record.(add Record.kind "r" v empty) in
          let back =
            Column.values Record.kind
              (Column.v outer [| r (Some m_null); r None |])
          in
          let inner r =
            Option.map Record.names (Record.field Record.kind "r" r)
          in
          equal
            (array (option (list string)))
            [| Some [ "m" ]; None |] (Array.map inner back));
      test "a null field of another kind is a null" (fun () ->
          let r = Record.(add Kind.float "a" None empty) in
          let ty = Type.record [ ("a", Any Type.int8) ] in
          let back = Column.values Record.kind (Column.v ty [| r |]) in
          equal (option int) None (Record.field Kind.int "a" back.(0)));
    ]

(* Refusals *)

let ab = Record.(empty |> add Kind.int "a" (Some 1))
let a_float = Record.(empty |> add Kind.float "a" (Some 1.))
let a_300 = Record.(empty |> add Kind.int "a" (Some 300))
let int8s = Column.of_options Type.int8 [| Some 1; None |]
let big = Column.of_tensor (Nx.create Nx.int64 [| 2 |] [| 0L; Int64.max_int |])
let refuse name f expected = (name, (fun () -> ignore (f ())), expected)

let refusals =
  let open Type in
  [
    refuse "an int out of range" (fun () -> Column.v int8 [| 1; 300 |])
    @@ __POS_OF__ {| Column.v: row 1: int8 does not hold 300 |};
    refuse "the first of two ints out of range" (fun () ->
        Column.v int8 [| 1; 300; 400 |])
    @@ __POS_OF__ {| Column.v: row 1: int8 does not hold 300 |};
    refuse "the first value outside int that is not null" (fun () ->
        let validity =
          Nx.cast Nx.bit (Nx.create Nx.bool [| 3 |] [| false; true; true |])
        in
        let x = Nx.create Nx.uint64 [| 3 |] [| -1L; 5L; -1L |] in
        Column.options Kind.int (Column.of_tensor ~validity x))
    @@ __POS_OF__
         {| Column.options: row 2: 18446744073709551615 is outside int |};
    refuse "a float that rounds to infinity" (fun () ->
        Column.v float16 [| 65520. |])
    @@ __POS_OF__ {| Column.v: row 0: float16 does not hold 65520. |};
    refuse "text that is not UTF-8" (fun () ->
        Column.of_options string [| None; Some "a\xff" |])
    @@ __POS_OF__ {| Column.of_options: row 1: string does not hold "a\xff" |};
    refuse "a string outside the dictionary" (fun () ->
        Column.v (categorical [| "x" |]) [| "y" |])
    @@ __POS_OF__ {| Column.v: row 0: categorical["x"] does not hold "y" |};
    refuse "a clock past the day" (fun () ->
        Column.v (clock S) [| Time.Span.hours 24 |])
    @@ __POS_OF__ {| Column.v: row 0: clock[s] does not hold 24h |};
    refuse "a span that is not a whole unit" (fun () ->
        Column.v (duration S) [| Time.Span.ms 1500 |])
    @@ __POS_OF__ {| Column.v: row 0: duration[s] does not hold 1s500ms |};
    refuse "a tensor of another shape" (fun () ->
        Column.v (tensor Nx.float32 [| 2 |]) [| Nx.zeros Nx.float32 [| 3 |] |])
    @@ __POS_OF__
         {| Column.v: row 0: tensor[float32, 2] does not hold a tensor of shape [3] |};
    refuse "a list element out of range" (fun () ->
        Column.v (list int8) [| [| 1 |]; [| 2; 300 |] |])
    @@ __POS_OF__ {| Column.v: row 1: element 1: int8 does not hold 300 |};
    refuse "a record of other fields" (fun () ->
        Column.v (record [ ("b", Any int8) ]) [| ab |])
    @@ __POS_OF__
         {| Column.v: row 0: record[b int8] does not hold a record of fields [a] |};
    refuse "a record field of another kind" (fun () ->
        Column.v (record [ ("a", Any int8) ]) [| a_float |])
    @@ __POS_OF__
         {| Column.v: row 0: field a: int8 does not hold a float value |};
    refuse "a record field out of range" (fun () ->
        Column.v (record [ ("a", Any int8) ]) [| a_300 |])
    @@ __POS_OF__ {| Column.v: row 0: field a: int8 does not hold 300 |};
    refuse "values with a null" (fun () -> Column.values Kind.int int8s)
    @@ __POS_OF__ {| Column.values: row 1 is null |};
    refuse "a kind that does not read the type" (fun () ->
        Column.options Kind.float int8s)
    @@ __POS_OF__ {| Column.options: int8 is not read as float |};
    refuse "an extension column read by its kind" (fun () ->
        let ty = ext ~name:"m" float64 in
        Column.options (kind ty) (Column.of_options ty [| None |]))
    @@ __POS_OF__ {| Column.options: no kind reads ext[m, float64] |};
    refuse "an int64 outside OCaml's int" (fun () ->
        Column.options Kind.int big)
    @@ __POS_OF__
         {| Column.options: row 1: 9223372036854775807 is outside int |};
    refuse "a uint64 outside OCaml's int" (fun () ->
        Column.values Kind.int
          (Column.of_tensor (Nx.create Nx.uint64 [| 1 |] [| -1L |])))
    @@ __POS_OF__
         {| Column.values: row 0: 18446744073709551615 is outside int |};
    refuse "of_tensor of a scalar" (fun () ->
        Column.of_tensor (Nx.scalar Nx.int8 1))
    @@ __POS_OF__ {| Column.of_tensor: a scalar has no rows |};
    refuse "of_tensor of 1-D bfloat16" (fun () ->
        Column.of_tensor (Nx.zeros Nx.bfloat16 [| 2 |]))
    @@ __POS_OF__
         {| Column.of_tensor: no scalar type stores bfloat16; make it 2-D |};
    refuse "of_tensor with a validity of another length" (fun () ->
        let validity = Nx.cast Nx.bit (Nx.create Nx.bool [| 1 |] [| true |]) in
        Column.of_tensor ~validity (Nx.zeros Nx.int8 [| 2 |]))
    @@ __POS_OF__ {| Column.of_tensor: a validity of shape [1] for 2 rows |};
    refuse "to_tensor at another dtype" (fun () ->
        Column.to_tensor Nx.int64 (Column.v date [| day 0 |]))
    @@ __POS_OF__ {| Column.to_tensor: date is stored as int32, not int64 |};
    refuse "to_tensor with a null" (fun () -> Column.to_tensor Nx.int8 int8s)
    @@ __POS_OF__ {| Column.to_tensor: row 1 is null |};
    refuse "to_tensor of text" (fun () ->
        Column.to_tensor Nx.uint8 (Column.v string [| "a" |]))
    @@ __POS_OF__
         {| Column.to_tensor: string is not stored one element per row |};
    refuse "ragged of a number column" (fun () ->
        Column.ragged Nx.int8 (Column.v int8 [| 1 |]))
    @@ __POS_OF__ {| Column.ragged: int8 is not a list of int8, text or bytes |};
    refuse "ragged of a list at another dtype" (fun () ->
        Column.ragged Nx.int64 (Column.v (list int32) [| [| 1 |] |]))
    @@ __POS_OF__ {| Column.ragged: list[int32] is stored as int32, not int64 |};
    refuse "ragged of text at another dtype than uint8" (fun () ->
        Column.ragged Nx.int8 (Column.v string [| "a" |]))
    @@ __POS_OF__ {| Column.ragged: string is stored as uint8, not int8 |};
    refuse "ragged of a list of text" (fun () ->
        Column.ragged Nx.uint8 (Column.v (list string) [| [| "a" |] |]))
    @@ __POS_OF__
         {| Column.ragged: list[string] is not a list of uint8, text or bytes |};
    refuse "ragged with a null row" (fun () ->
        Column.ragged Nx.uint8
          (Column.of_options binary [| Some (Binary.of_string "a"); None |]))
    @@ __POS_OF__ {| Column.ragged: row 1 is null |};
    refuse "ragged with a null element" (fun () ->
        let elements = Column.of_options int32 [| Some 1; Some 2; None |] in
        let l =
          Column.Varsize
            { validity = None; offsets = n [| 0L; 2L; 3L |]; child = elements }
        in
        Column.ragged Nx.int32
          (Result.get_ok (Column.of_layout (Any (list int32)) l)))
    @@ __POS_OF__ {| Column.ragged: an element of row 1 is null |};
    refuse "of_ragged with a validity of another length" (fun () ->
        Column.of_ragged ~validity:(bits [| true |])
          (Nx_ragged.v
             ~offsets:(n [| 0L; 1L; 1L |])
             (Nx.create Nx.int32 [| 1 |] [| 7l |])))
    @@ __POS_OF__ {| Column.of_ragged: a validity of shape [1] for 2 rows |};
    refuse "of_ragged of bfloat16 elements" (fun () ->
        Column.of_ragged
          (Nx_ragged.v ~offsets:(n [| 0L; 1L |]) (Nx.zeros Nx.bfloat16 [| 1 |])))
    @@ __POS_OF__
         {| Column.of_ragged: no scalar type stores bfloat16; make it 2-D |};
  ]

let refusal_cases =
  cases
    ~name:(fun (n, _, _) -> n)
    "Refusals" refusals
    (fun (_, f, expected) -> expect (message f) expected)

(* Layouts *)

let pp_ty ppf (Type.Any t) = Type.pp ppf t

let pp_validity ppf = function
  | None -> Format.pp_print_string ppf "no null"
  | Some b -> Nx.pp ppf b

let rec pp_layout ppf c =
  let ty = Column.type_ c and v = Column.validity c in
  match Column.layout c with
  | Fixed { values = P x; _ } ->
      Format.fprintf ppf "@[<hv 2>%a,@ %a,@ %a@]" pp_ty ty pp_validity v Nx.pp x
  | Varsize { offsets; child; _ } ->
      Format.fprintf ppf "@[<hv 2>%a,@ %a,@ %a,@ %a@]" pp_ty ty pp_validity v
        Nx.pp offsets pp_layout child
  | Children { length; fields; _ } ->
      let field ppf (n, c) = Format.fprintf ppf "%s: %a" n pp_layout c in
      Format.fprintf ppf "@[<hv 2>%a,@ %d rows,@ %a,@ %a@]" pp_ty ty length
        pp_validity v
        (Format.pp_print_list field)
        fields

let same_values (Nx.P x) (Nx.P y) =
  match Nx_dtype.equal_witness (Nx.dtype x) (Nx.dtype y) with
  | Some Equal ->
      Nx.shape x = Nx.shape y && compare (Nx.to_array x) (Nx.to_array y) = 0
  | None -> false

(* Columns are the same when their types, validities and buffers' values are, at
   every depth. *)
let rec same a b =
  let bits c = Option.map Nx.to_array c in
  let same_type (Type.Any a) (Type.Any b) = Type.equal a b in
  same_type (Column.type_ a) (Column.type_ b)
  && bits (Column.validity a) = bits (Column.validity b)
  &&
  match (Column.layout a, Column.layout b) with
  | Fixed x, Fixed y -> same_values x.values y.values
  | Varsize x, Varsize y ->
      Nx.to_array x.offsets = Nx.to_array y.offsets && same x.child y.child
  | Children x, Children y ->
      x.length = y.length
      && List.equal
           (fun (n, a) (m, b) -> String.equal n m && same a b)
           x.fields y.fields
  | _ -> false

let column_w = Testable.make ~pp:pp_layout ~equal:same

let layout_round_trip (G.Sample (ty, vs)) =
  cover "a null" (nulls vs > 0);
  cover "text" (match ty with String -> true | _ -> false);
  cover "nested" (match ty with List _ | Record _ -> true | _ -> false);
  Law.round_trip column_w pass Column.layout
    (fun l -> require_ok (Column.of_layout (Any ty) l))
    (Column.of_options ty vs)

(* An extension column holds its storage's values, and a record field of it
   holds them as its storage's. *)
let ext_in_record (G.Sample (ty, vs)) =
  assume (match ty with Ext _ -> false | _ -> true);
  cover "a null" (nulls vs > 0);
  cover "a list or a record"
    (match ty with List _ | Record _ -> true | _ -> false);
  let c = Column.of_options ty vs in
  let ety = Type.ext ~name:"m" ty in
  let e = require_ok (Column.of_layout (Any ety) (Column.layout c)) in
  let rty = Type.record [ ("e", Any ety) ] in
  let fields = [ ("e", e) ] and length = Array.length vs in
  let l = Column.Children { validity = None; length; fields } in
  let r = require_ok (Column.of_layout (Any rty) l) in
  Law.round_trip column_w pass (Column.values Record.kind) (Column.v rty) r

let fixed ?validity dt x =
  Column.Fixed { validity; values = P (Nx.create dt [| Array.length x |] x) }

let bytes ?validity rows =
  let s = String.concat "" rows in
  let next (o, os) r = (o + String.length r, Int64.of_int o :: os) in
  let last, os = List.fold_left next (0, []) rows in
  let offsets = n (Array.of_list (List.rev (Int64.of_int last :: os))) in
  let child =
    Column.of_tensor
      (Nx.create Nx.uint8
         [| String.length s |]
         (Array.init (String.length s) (fun i -> Char.code s.[i])))
  in
  Column.Varsize { validity; offsets; child }

let unheld =
  let open Type in
  let code = categorical [| "x"; "y" |] in
  [
    ( "the first and last scalar values of each length",
      Any string,
      bytes
        [
          "\x00\x7f";
          "\xc2\x80\xdf\xbf";
          "\xe0\xa0\x80\xed\x9f\xbf";
          "\xee\x80\x80\xef\xbf\xbf";
          "\xf0\x90\x80\x80\xf4\x8f\xbf\xbf";
        ],
      None );
    ( "an overlong form",
      Any string,
      bytes [ "a"; "\xc0\xaf" ],
      Some (1, "invalid UTF-8 at byte 0") );
    ( "a surrogate",
      Any string,
      bytes [ "ab\xed\xa0\x80" ],
      Some (0, "invalid UTF-8 at byte 2") );
    ( "a value past U+10FFFF",
      Any string,
      bytes [ "\xf4\x90\x80\x80" ],
      Some (0, "invalid UTF-8 at byte 0") );
    ( "a sequence cut by the row's end",
      Any string,
      bytes [ "\xe2\x82"; "\xac" ],
      Some (0, "invalid UTF-8 at byte 0") );
    ( "the last row",
      Any string,
      bytes [ "é"; "日本"; "\xff" ],
      Some (2, "invalid UTF-8 at byte 0") );
    ( "invalid text under a null",
      Any string,
      bytes ~validity:(bits [| true; false |]) [ "a"; "\xff" ],
      None );
    ("binary holds any byte", Any binary, bytes [ "\xff" ], None);
    ( "an extension's storage",
      Any (ext ~name:"m" string),
      bytes [ "\xff" ],
      Some (0, "invalid UTF-8 at byte 0") );
    ( "a code past the dictionary",
      Any code,
      fixed Nx.int32 [| 1l; 2l |],
      Some (1, {|categorical["x", "y"] has no code 2|}) );
    ( "a negative code",
      Any code,
      fixed Nx.int32 [| -1l |],
      Some (0, {|categorical["x", "y"] has no code -1|}) );
    ( "a code past the dictionary under a null",
      Any code,
      fixed ~validity:(bits [| false |]) Nx.int32 [| 2l |],
      None );
    ( "the last clock tick of the day",
      Any (clock S),
      fixed Nx.int64 [| 0L; 86_399L |],
      None );
    ( "a clock at the end of the day",
      Any (clock Ms),
      fixed Nx.int64 [| 0L; 86_400_000L |],
      Some (1, "clock[ms] tick 86400000 is outside the day") );
    ( "a negative clock",
      Any (clock Ns),
      fixed Nx.int64 [| -1L |],
      Some (0, "clock[ns] tick -1 is outside the day") );
  ]

let unheld_cases =
  cases
    ~name:(fun (n, _, _, _) -> n)
    "Values a type does not hold" unheld
    (fun (_, ty, l, expected) ->
      let got =
        match Column.of_layout ty l with Ok _ -> None | Error e -> Some e
      in
      equal (option (pair int string)) expected got)

(* [first_invalid s] is the first byte of [s] that starts no valid UTF-8
   sequence, as the standard library decodes it. *)
let first_invalid s =
  let rec go i =
    if i >= String.length s then None
    else
      let d = String.get_utf_8_uchar s i in
      if Uchar.utf_decode_is_valid d then go (i + Uchar.utf_decode_length d)
      else Some i
  in
  go 0

(* Rows of ASCII runs long and short, scalar values of each length, and bytes
   that start no valid sequence: lone continuations, overlong leads, cut
   sequences, surrogates, values past U+10FFFF. *)
let text_rows =
  let ascii =
    Gen.string_of ~size:(Gen.int_range 0 20) (Gen.char_range ' ' '~')
  in
  let valid = Gen.of_list [ "\x7f"; "é"; "日"; "𝄞"; "\xf4\x8f\xbf\xbf" ] in
  let bad =
    Gen.of_list
      [
        "\x80";
        "\xbf";
        "\xc0\xaf";
        "\xc1";
        "\xc3";
        "\xe0\x80\x80";
        "\xe2\x82";
        "\xed\xa0\x80";
        "\xf0\x80";
        "\xf4\x90\x80\x80";
        "\xf5";
        "\xff";
      ]
  in
  let piece = Gen.frequency [ (6, ascii); (3, valid); (1, bad) ] in
  let row =
    Gen.map (String.concat "") (Gen.list ~size:(Gen.int_range 0 6) piece)
  in
  Gen.list ~size:(Gen.int_range 0 6) row

let utf_8_law rows =
  let firsts = List.map first_invalid rows in
  cover "valid text" (List.for_all Option.is_none firsts);
  cover "an invalid byte past eight ASCII bytes"
    (List.exists (function Some i -> i >= 8 | None -> false) firsts);
  let why r i = (r, Printf.sprintf "invalid UTF-8 at byte %d" i) in
  let expected = List.find_mapi (fun r -> Option.map (why r)) firsts in
  let got =
    match Column.of_layout (Any Type.string) (bytes rows) with
    | Ok _ -> None
    | Error e -> Some e
  in
  equal (option (pair int string)) expected got

let layouts =
  group "Layouts"
    [
      prop "of_layout reads back a column's layout" G.sample layout_round_trip;
      prop "of_layout refuses the first row of text that is not UTF-8" text_rows
        utf_8_law;
      prop "a record field of an extension type reads and writes its storage"
        G.sample ext_in_record;
      test "of_layout shares its values" (fun () ->
          let x = Nx.create Nx.int32 [| 2 |] [| 1l; 2l |] in
          let c =
            Column.of_layout (Any Type.date)
              (Fixed { validity = None; values = P x })
          in
          satisfies ~claim:"x itself"
            (Testable.make ~pp:Nx.pp ~equal:( == ))
            (fun y -> y == x)
            (Column.to_tensor Nx.int32 (require_ok c)));
      unheld_cases;
    ]

let int8s_layout = fixed Nx.int8 [| 1; 2 |]
let child = Column.v Type.int8 [| 1; 2 |]
let of_layout ty l () = Column.of_layout (Type.Any ty) l

let layout_refusals =
  let open Type in
  let varsize offsets child =
    Column.Varsize { validity = None; offsets = n offsets; child }
  in
  let children ?(length = 2) fields =
    Column.Children { validity = None; length; fields }
  in
  [
    refuse "values of another dtype" (of_layout int16 int8s_layout)
    @@ __POS_OF__
         {| Column.of_layout: int8 values of shape [2] do not lay out int16 |};
    refuse "values of another cell shape"
      (of_layout (tensor Nx.int8 [| 3 |])
         (Fixed { validity = None; values = P (Nx.zeros Nx.int8 [| 2; 2 |]) }))
    @@ __POS_OF__
         {| Column.of_layout: int8 values of shape [2,2] do not lay out tensor[int8, 3] |};
    refuse "a scalar for a scalar type"
      (of_layout int8
         (Fixed { validity = None; values = P (Nx.scalar Nx.int8 1) }))
    @@ __POS_OF__
         {| Column.of_layout: int8 values of shape [] do not lay out int8 |};
    refuse "a validity of another length"
      (of_layout int8 (fixed ~validity:(bits [| true |]) Nx.int8 [| 1; 2 |]))
    @@ __POS_OF__ {| Column.of_layout: a validity of shape [1] for 2 rows |};
    refuse "offsets that are 2-D"
      (of_layout (list int8)
         (Varsize
            { validity = None; offsets = Nx.zeros Nx.int64 [| 1; 1 |]; child }))
    @@ __POS_OF__
         {| Column.of_layout: offsets of shape [1,1], not 1-D with an entry |};
    refuse "no offsets" (of_layout (list int8) (varsize [||] child))
    @@ __POS_OF__
         {| Column.of_layout: offsets of shape [0], not 1-D with an entry |};
    refuse "offsets that start below 0"
      (of_layout (list int8) (varsize [| -1L; 0L |] child))
    @@ __POS_OF__ {| Column.of_layout: offsets start at -1 |};
    refuse "offsets that decrease"
      (of_layout (list int8) (varsize [| 0L; 2L; 1L |] child))
    @@ __POS_OF__ {| Column.of_layout: offsets decrease at row 1 |};
    refuse "offsets past the child"
      (of_layout (list int8) (varsize [| 0L; 3L |] child))
    @@ __POS_OF__
         {| Column.of_layout: offsets end at 3, past the child's 2 rows |};
    refuse "a list child of another type"
      (of_layout (list int16) (varsize [| 0L; 2L |] child))
    @@ __POS_OF__
         {| Column.of_layout: a child of int8 does not lay out list[int16] |};
    refuse "text over a child of another type"
      (of_layout string (varsize [| 0L; 2L |] child))
    @@ __POS_OF__
         {| Column.of_layout: a child of int8 does not lay out string |};
    refuse "text over a child with a null"
      (of_layout string
         (varsize [| 0L; 1L |] (Column.of_options uint8 [| Some 97; None |])))
    @@ __POS_OF__
         {| Column.of_layout: a child with a null does not lay out string |};
    refuse "fields of other names"
      (of_layout (record [ ("a", Any int8) ]) (children [ ("b", child) ]))
    @@ __POS_OF__
         {| Column.of_layout: fields [b] do not lay out record[a int8] |};
    refuse "a field of another type"
      (of_layout (record [ ("a", Any int16) ]) (children [ ("a", child) ]))
    @@ __POS_OF__
         {| Column.of_layout: field a of int8 does not lay out record[a int16] |};
    refuse "a field of another length"
      (of_layout
         (record [ ("a", Any int8) ])
         (children ~length:3 [ ("a", child) ]))
    @@ __POS_OF__ {| Column.of_layout: field a of 2 rows for 3 |};
    refuse "a negative length"
      (of_layout (record []) (children ~length:(-1) []))
    @@ __POS_OF__ {| Column.of_layout: a length of -1 |};
    refuse "offsets for a scalar type"
      (of_layout int8 (varsize [| 0L; 2L |] child))
    @@ __POS_OF__
         {| Column.of_layout: offsets and a child do not lay out int8 |};
    refuse "fields for a list" (of_layout (list int8) (children []))
    @@ __POS_OF__ {| Column.of_layout: fields do not lay out list[int8] |};
  ]

let layout_refusal_cases =
  cases
    ~name:(fun (n, _, _) -> n)
    "Layout refusals" layout_refusals
    (fun (_, f, expected) -> expect (message f) expected)

(* Ragged arrays *)

(* The type for a dtype, its values' equality and a generator of them. *)
type dtype =
  | Dtype : string * ('a, 'b) Nx.dtype * ('a -> 'a -> bool) * 'a Gen.t -> dtype

let bits_equal a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let dtypes =
  let int lo hi = Gen.int_range lo hi in
  [
    Dtype ("bool", Nx.bool, Bool.equal, Gen.bool);
    Dtype ("int8", Nx.int8, Int.equal, int (-128) 127);
    Dtype ("int16", Nx.int16, Int.equal, int (-32768) 32767);
    Dtype ("int32", Nx.int32, Int32.equal, Gen.int32);
    Dtype ("int64", Nx.int64, Int64.equal, Gen.int64);
    Dtype ("uint8", Nx.uint8, Int.equal, int 0 255);
    Dtype ("uint16", Nx.uint16, Int.equal, int 0 65535);
    Dtype ("uint32", Nx.uint32, Int32.equal, Gen.int32);
    Dtype ("uint64", Nx.uint64, Int64.equal, Gen.int64);
    Dtype
      ( "float16",
        Nx.float16,
        bits_equal,
        Gen.map Int.to_float (int (-2048) 2048) );
    Dtype
      ( "float32",
        Nx.float32,
        bits_equal,
        Gen.map
          (fun x -> Int32.float_of_bits (Int32.bits_of_float x))
          Gen.any_float );
    Dtype ("float64", Nx.float64, bits_equal, Gen.any_float);
  ]

(* [ragged_gen dt v] draws up to six rows of up to four values, over values that
   may hold rows before the first offset and after the last. *)
let ragged_gen (type a b) (dt : (a, b) Nx.dtype) (v : a Gen.t) =
  let open Gen in
  let* lengths = list ~size:(int_range 0 6) (int_range 0 4) in
  let* before = int_range 0 2 in
  let* after = int_range 0 2 in
  let total = before + List.fold_left ( + ) 0 lengths + after in
  let+ values = array ~size:(constant total) v in
  let offsets =
    List.fold_left (fun acc l -> (List.hd acc + l) :: acc) [ before ] lengths
    |> List.rev_map Int64.of_int |> Array.of_list
  in
  Nx_ragged.v ~offsets:(n offsets) (Nx.create dt [| total |] values)

let ragged_w (type a b) (eq : a -> a -> bool) : (a, b) Nx_ragged.t Testable.t =
  let rows r =
    let o = Nx.to_array (Nx_ragged.offsets r)
    and v = Nx.to_array (Nx_ragged.values r) in
    Array.init
      (Array.length o - 1)
      (fun i ->
        let first = Int64.to_int o.(i) in
        Array.sub v first (Int64.to_int o.(i + 1) - first))
  in
  Testable.make
    ~pp:(fun ppf r ->
      Format.fprintf ppf "%d rows of lengths %a" (Nx_ragged.length r) Nx.pp
        (Nx_ragged.lengths r))
    ~equal:(fun a b ->
      let ra = rows a and rb = rows b in
      Array.length ra = Array.length rb
      && Array.for_all2
           (fun x y -> Array.length x = Array.length y && Array.for_all2 eq x y)
           ra rb)

let ragged =
  group "Ragged arrays"
    (List.map
       (fun (Dtype (name, dt, eq, v)) ->
         prop
           (Printf.sprintf "ragged %s of of_ragged is the ragged array" name)
           (Gen.with_pp
              (fun ppf r -> Format.fprintf ppf "%d rows" (Nx_ragged.length r))
              (ragged_gen dt v))
           (fun r ->
             cover "no row" (Nx_ragged.length r = 0);
             cover "an empty row"
               (Array.exists (Int64.equal 0L)
                  (Nx.to_array (Nx_ragged.lengths r)));
             Law.round_trip (ragged_w eq) column_w Column.of_ragged
               (Column.ragged dt) r))
       dtypes
    @ [
        test "of_ragged is a list of the dtype's type, sharing the buffers"
          (fun () ->
            let r =
              Nx_ragged.v
                ~offsets:(n [| 1L; 3L; 3L |])
                (Nx.create Nx.int32 [| 4 |] [| 9l; 1l; 2l; 9l |])
            in
            let c = Column.of_ragged r in
            equal any_w (Any Type.(list int32)) (Column.type_ c);
            equal
              (rows Type.(list int32))
              [| Some [| 1; 2 |]; Some [||] |]
              (Column.options Kind.(list int) c);
            let back = Column.ragged Nx.int32 c in
            satisfies ~claim:"the same values" pass
              (fun v -> v == Nx_ragged.values r)
              (Nx_ragged.values back);
            satisfies ~claim:"the same offsets" pass
              (fun o -> o == Nx_ragged.offsets r)
              (Nx_ragged.offsets back));
        test "of_ragged's validity makes rows null" (fun () ->
            let r =
              Nx_ragged.v
                ~offsets:(n [| 0L; 1L; 2L |])
                (Nx.create Nx.int8 [| 2 |] [| 1; 2 |])
            in
            equal
              (rows Type.(list int8))
              [| Some [| 1 |]; None |]
              (Column.options
                 Kind.(list int)
                 (Column.of_ragged ~validity:(bits [| true; false |]) r)));
        test "of_ragged of values of two axes is a list of tensors" (fun () ->
            let r =
              Nx_ragged.v
                ~offsets:(n [| 0L; 2L |])
                (Nx.zeros Nx.float32 [| 2; 3 |])
            in
            equal any_w
              (Any Type.(list (tensor Nx.float32 [| 3 |])))
              (Column.type_ (Column.of_ragged r)));
        prop "ragged uint8 of text is each row's bytes"
          (Gen.array ~size:(Gen.int_range 0 8)
             (Option.get (G.value Type.string)))
          (fun ss ->
            let r = Column.ragged Nx.uint8 (Column.v Type.string ss) in
            equal (array int)
              (Array.map String.length ss)
              (Array.map Int64.to_int (Nx.to_array (Nx_ragged.lengths r)));
            let bytes = String.concat "" (Array.to_list ss) in
            let o = Int64.to_int (Nx.item [ 0 ] (Nx_ragged.offsets r)) in
            equal (array int)
              (Array.init (String.length bytes) (fun i -> Char.code bytes.[i]))
              (Array.sub
                 (Nx.to_array (Nx_ragged.values r))
                 o (String.length bytes)));
        test "a null element outside the rows is not the column's" (fun () ->
            let elements =
              Column.of_options Type.int32 [| None; Some 1; None |]
            in
            let l =
              Column.Varsize
                { validity = None; offsets = n [| 1L; 2L |]; child = elements }
            in
            let c =
              Result.get_ok (Column.of_layout (Any Type.(list int32)) l)
            in
            let r = Column.ragged Nx.int32 c in
            equal (array int64) [| 1L; 2L |] (Nx.to_array (Nx_ragged.offsets r)));
      ])

(* Nulls

   A validity's count is read at the first null_count and kept. The model is a
   column's validity as booleans; columns made from others count their own
   nulls, and two domains that count at once agree with some order of the
   counts. *)

let valid_bits m = Nx.cast Nx.bit (Nx.create Nx.bool [| Array.length m |] m)
let int64s n = Nx.init Nx.int64 [| n |] (fun i -> Int64.of_int i.(0))

let column_of m =
  Column.of_tensor ~validity:(valid_bits m) (int64s (Array.length m))

let count_model m = Array.fold_left (fun n v -> if v then n else n + 1) 0 m
let rows_of c = Column.length c
let column_w = abstract "c"

let masks =
  Gen.with_pp
    (fun ppf m ->
      Array.iter (fun v -> Format.pp_print_char ppf (if v then '1' else '0')) m)
    (Gen.array ~size:(Gen.int_range 0 40) Gen.bool)

(* Indices, outside the rows too: such a row is null. *)
let picks =
  Gen.with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space Format.pp_print_int)
    (Gen.list ~size:(Gen.int_range 0 12) (Gen.int_range (-2) 45))

let take_model is m =
  Array.of_list (List.map (fun i -> i >= 0 && i < Array.length m && m.(i)) is)

let take_column is c =
  let idx =
    Nx.create Nx.int64
      [| List.length is |]
      (Array.of_list (List.map Int64.of_int is))
  in
  let inside = List.for_all (fun i -> i >= 0 && i < rows_of c) is in
  if inside then column (take idx (v [ ("x", c) ])) "x"
  else
    (* Talon.take refuses an index outside the rows: a left join pads. *)
    let l = v [ ("k", Column.of_tensor idx) ] in
    let r = v [ ("k", Column.of_tensor (int64s (rows_of c))); ("x", c) ] in
    let joined =
      Error.get_ok
        Query.(
          run
            (of_table l
            |> join ~kind:Left ~each_left:At_most_one ~on:(Join.keys [ "k" ])
                 (of_table r)))
    in
    column joined "x"

let counts =
  [
    command "of_tensor" (masks @-> makes column_w) Fun.id column_of;
    command "null_count"
      (column_w ^-> returns int)
      count_model Column.null_count;
    command "take"
      (picks @-> column_w ^-> makes column_w)
      take_model take_column;
    command "concat"
      (column_w ^-> column_w ^-> makes column_w)
      (fun a b -> Array.append a b)
      (fun a b ->
        let t c = v [ ("x", c) ] in
        column (of_batches [ t a; t b ]) "x");
    command "validity"
      (column_w ^-> returns (option (array bool)))
      (fun m -> if count_model m = 0 then None else Some m)
      (fun c ->
        match Column.validity c with
        | Some v when Column.null_count c > 0 -> Some (Nx.to_array v)
        | _ -> None);
  ]

let nulls_cases =
  group "nulls"
    [
      stateful "null_count is the number of null rows, read once" counts;
      stateful ~domains:2
        "null_count from two domains at once is the number of null rows" counts;
      test "a validity with every row set has no null" (fun () ->
          let c = column_of [| true; true; true |] in
          is_some (Column.validity c);
          equal int 0 (Column.null_count c));
      test "a validity whose bits past its rows are set counts its rows only"
        (fun () ->
          (* Two bytes of ones, the second's bits past row 13 set too, and row 2
             cleared. *)
          let bytes = Nx.create Nx.uint8 [| 2 |] [| 0xfb; 0xff |] in
          let validity =
            Nx.shrink
              [| (0, 13) |]
              (Nx.reshape [| -1 |] (Nx.bitcast Nx.bit bytes))
          in
          let c = Column.of_tensor ~validity (int64s 13) in
          equal int 1 (Column.null_count c);
          let t = Error.get_ok Query.(run (of_table (v [ ("x", c) ]))) in
          let b = Option.get (Column.validity (column t "x")) in
          let raw =
            Nx_device.Buffer.bigarray Bigarray.int8_unsigned (Nx.to_buffer b)
          in
          equal int 0 (raw.{1} lsr 5));
      test "run drops a validity with no null" (fun () ->
          let c = column_of [| true; true |] in
          let t = Error.get_ok Query.(run (of_table (v [ ("x", c) ]))) in
          is_none (Column.validity (column t "x")));
    ]

let () =
  exit
    (run "Column"
       [
         codec;
         edge_cases;
         float_bits;
         storage;
         ragged;
         records;
         refusal_cases;
         layouts;
         layout_refusal_cases;
         nulls_cases;
       ])
