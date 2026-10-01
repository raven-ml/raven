(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Talon_next
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

let round_trip (G.Sample (ty, vs)) =
  cover "a null" (nulls vs > 0);
  cover "no row" (vs = [||]);
  cover "nested" (match ty with List _ | Record _ -> true | _ -> false);
  cover "no kind reads it" (not (readable ty));
  let c = Column.of_options ty vs in
  if readable ty then
    Law.round_trip (rows ty) pass (Fun.const c)
      (Column.options (Type.kind ty))
      vs
  else rejects (fun () -> Column.options (Type.kind ty) c)

let observations (G.Sample (ty, vs)) =
  let c = Column.of_options ty vs in
  equal any_w (Any ty) (Column.type_ c);
  equal int (Array.length vs) (Column.length c);
  equal int (nulls vs) (Column.null_count c);
  match Column.validity c with
  | None -> equal int 0 (nulls vs)
  | Some b ->
      equal (array bool)
        (Array.map Option.is_some vs)
        (Nx.to_array (Nx_bits.to_bool b))

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
        ( "decimal digits at the precision",
          decimal ~precision:4 ~scale:2,
          [|
            Some (Decimal.v ~unscaled:(-9999L) ~scale:2);
            Some (Decimal.v ~unscaled:1L ~scale:0);
          |] );
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
    ]

(* Storage *)

let storage =
  group "Storage"
    [
      test
        "to_tensor is the storage of temporal, decimal and categorical values"
        (fun () ->
          let tensor dt ty vs =
            Nx.to_array (Column.to_tensor dt (Column.v ty vs))
          in
          equal (array int32) [| -1l; 2l |]
            (tensor Nx.int32 Type.date [| day (-1); day 2 |]);
          equal (array int64) [| -1L; 3L |]
            (tensor Nx.int64 (Type.datetime Type.Ms)
               [| instant (-1_000_000L); instant 3_000_000L |]);
          equal (array int64) [| 1250L |]
            (tensor Nx.int64
               (Type.decimal ~precision:5 ~scale:3)
               [| Decimal.v ~unscaled:125L ~scale:2 |]);
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
      test "of_tensor reads a 1-D tensor as its dtype's scalar type" (fun () ->
          let c =
            Column.of_tensor (Nx.create Nx.uint32 [| 2 |] [| -1l; 7l |])
          in
          equal any_w (Any Type.uint32) (Column.type_ c);
          equal (array int) [| 0xffff_ffff; 7 |] (Column.values Kind.int c));
      test "of_tensor's validity makes nulls" (fun () ->
          let validity =
            Nx_bits.of_bool (Nx.create Nx.bool [| 3 |] [| true; false; true |])
          in
          let c =
            Column.of_tensor ~validity (Nx.create Nx.int8 [| 3 |] [| 1; 2; 3 |])
          in
          equal int 1 (Column.null_count c);
          equal
            (array (option int))
            [| Some 1; None; Some 3 |]
            (Column.options Kind.int c));
      test "of_tensor drops a validity with every row set" (fun () ->
          let validity =
            Nx_bits.of_bool (Nx.create Nx.bool [| 1 |] [| true |])
          in
          let c =
            Column.of_tensor ~validity (Nx.create Nx.int8 [| 1 |] [| 1 |])
          in
          is_none (Column.validity c));
      test "ragged is the bytes of each row" (fun () ->
          let r = Column.ragged (Column.v Type.string [| "é"; ""; "ab" |]) in
          equal (array int64) [| 2L; 0L; 2L |]
            (Nx.to_array (Nx_ragged.lengths r));
          equal (array int)
            [| 0xc3; 0xa9; 0x61; 0x62 |]
            (Nx.to_array (Nx_ragged.values r)));
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
    refuse "a float that rounds to infinity" (fun () ->
        Column.v float16 [| 65520. |])
    @@ __POS_OF__ {| Column.v: row 0: float16 does not hold 65520 |};
    refuse "text that is not UTF-8" (fun () ->
        Column.of_options string [| None; Some "a\xff" |])
    @@ __POS_OF__ {| Column.of_options: row 1: string does not hold "a\xff" |};
    refuse "a string outside the dictionary" (fun () ->
        Column.v (categorical [| "x" |]) [| "y" |])
    @@ __POS_OF__ {| Column.v: row 0: categorical["x"] does not hold "y" |};
    refuse "a decimal of too many digits" (fun () ->
        Column.v
          (decimal ~precision:3 ~scale:1)
          [| Decimal.v ~unscaled:1234L ~scale:2 |])
    @@ __POS_OF__ {| Column.v: row 0: decimal[3, 1] does not hold 12.34 |};
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
        let validity = Nx_bits.of_bool (Nx.create Nx.bool [| 1 |] [| true |]) in
        Column.of_tensor ~validity (Nx.zeros Nx.int8 [| 2 |]))
    @@ __POS_OF__ {| Column.of_tensor: a validity of length 1 for 2 rows |};
    refuse "to_tensor at another dtype" (fun () ->
        Column.to_tensor Nx.int64 (Column.v date [| day 0 |]))
    @@ __POS_OF__ {| Column.to_tensor: date is stored as int32, not int64 |};
    refuse "to_tensor with a null" (fun () -> Column.to_tensor Nx.int8 int8s)
    @@ __POS_OF__ {| Column.to_tensor: row 1 is null |};
    refuse "to_tensor of text" (fun () ->
        Column.to_tensor Nx.uint8 (Column.v string [| "a" |]))
    @@ __POS_OF__
         {| Column.to_tensor: string is not stored one element per row |};
    refuse "ragged of a number column" (fun () -> Column.ragged int8s)
    @@ __POS_OF__ {| Column.ragged: int8 is neither string nor binary |};
    refuse "ragged with a null" (fun () ->
        Column.ragged (Column.of_options binary [| None |]))
    @@ __POS_OF__ {| Column.ragged: row 0 is null |};
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
  | Some b -> Nx.pp ppf (Nx_bits.to_bool b)

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
  let bits c = Option.map (fun b -> Nx.to_array (Nx_bits.to_bool b)) c in
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

let n x = Nx.create Nx.int64 [| Array.length x |] x
let bits b = Nx_bits.of_bool (Nx.create Nx.bool [| Array.length b |] b)

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
  let code = categorical [| "x"; "y" |]
  and dec = decimal ~precision:3 ~scale:1 in
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
    ( "a decimal at its precision",
      Any dec,
      fixed Nx.int64 [| -999L; 999L |],
      None );
    ( "a decimal past its precision",
      Any dec,
      fixed Nx.int64 [| 999L; 1000L |],
      Some (1, "decimal[3, 1] unscaled value 1000 has more than 3 digits") );
    ( "a negative decimal past its precision",
      Any dec,
      fixed Nx.int64 [| -1000L |],
      Some (0, "decimal[3, 1] unscaled value -1000 has more than 3 digits") );
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

let layouts =
  group "Layouts"
    [
      prop "of_layout reads back a column's layout" G.sample layout_round_trip;
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
    @@ __POS_OF__ {| Column.of_layout: a validity of length 1 for 2 rows |};
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

let () =
  exit
    (run "Column"
       [
         codec;
         edge_cases;
         float_bits;
         storage;
         refusal_cases;
         layouts;
         layout_refusal_cases;
       ])
