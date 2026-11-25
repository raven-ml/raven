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
let n x = Nx.create Nx.int64 [| Array.length x |] x
let bits b = Nx.cast Nx.bit (Nx.create Nx.bool [| Array.length b |] b)
let day d = Option.get (Time.Date.of_days d)
let ticks ns = Time.of_ns ns
let span ns = Time.Span.of_ns ns

(* [refusal f] is the message of the [Invalid_argument] that [f] raises. *)
let refusal f =
  match f () with _ -> fail "no exception" | exception Invalid_argument m -> m

(* [shares x y] is [true] iff [x] and [y] read the same memory. A column holds a
   value of its own over a tensor's memory, so that a donation of the tensor
   leaves it. *)
let shares x y =
  let buffer v = Nx_array.buffer (Option.get (Nx.Repr.array ~by:"test" v)) in
  Rig.Buffer.overlaps (buffer x) (buffer y)

let tensor_w () = Testable.make ~pp:Nx.pp ~equal:( == )

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
  match Nx.Dtype.equal_witness (Nx.dtype x) (Nx.dtype y) with
  | Some Equal ->
      Nx.shape x = Nx.shape y && compare (Nx.to_array x) (Nx.to_array y) = 0
  | None -> false

(* Columns are the same when their types, validities and buffers' values are, at
   every depth, those under their nulls included. *)
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

let fixed ?validity dt x =
  Column.Fixed { validity; values = P (Nx.create dt [| Array.length x |] x) }

(* [bytes rows] lays out the text [rows] as offsets over a [uint8] child. *)
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

(* Taking rows *)

let int64s n = Nx.init Nx.int64 [| n |] (fun i -> Int64.of_int i.(0))

(* [taken is c] is [c]'s rows at [is], through {!Talon.take}. *)
let taken is c =
  let idx = n (Array.of_list (List.map Int64.of_int is)) in
  column (take idx (v [ ("x", c) ])) "x"

(* [padded is c] is [c]'s rows at [is], null where [i] names no row, through a
   left join of the positions with the rows' places. *)
let padded is c =
  let l =
    v [ ("k", Column.of_tensor (n (Array.of_list (List.map Int64.of_int is)))) ]
  in
  let r = v [ ("k", Column.of_tensor (int64s (Column.length c))); ("x", c) ] in
  let joined =
    Error.get_ok
      Query.(
        run
          (of_table l
          |> join ~kind:Left ~each_left:At_most_one ~on:(Join.keys [ "k" ])
               (of_table r)))
  in
  column joined "x"

(* [inside n is] maps each position of [is] into [n] rows, and keeps none when
   there is no row. *)
let inside n is =
  if n = 0 then [] else List.map (fun i -> ((i mod n) + n) mod n) is

let run_x q c =
  column (Error.get_ok (Query.run (q (Query.of_table (v [ ("x", c) ]))))) "x"

(* Column *)

(* [consume x] donates [x] to an addition of it to itself, whose consumer may
   write its sum over [x]'s memory. *)
let consume x =
  let h = Nx.donate x in
  ignore (Nx.add h h)

let floats () = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |]
let three_bits () = bits [| true; false; true |]
let expected = [| Some 1.; None; Some 3. |]
let holds c = equal (rows Type.float32) expected (Column.options Kind.float c)

let donation =
  [
    test "donating the tensor and validity a column took leaves the column"
      (fun () ->
        let x = floats () and v = three_bits () in
        let c = Column.of_tensor ~validity:v x in
        consume x;
        ignore (Nx.logical_not (Nx.donate v));
        holds c);
    test "donating the tensors a column gave leaves the column" (fun () ->
        let c = Column.of_tensor ~validity:(three_bits ()) (floats ()) in
        ignore (Nx.logical_not (Nx.donate (Option.get (Column.validity c))));
        (match Column.layout c with
        | Fixed { values = P x; _ } -> consume x
        | _ -> fail "a fixed layout");
        holds c;
        let d = Column.of_tensor (floats ()) in
        consume (Column.to_tensor Nx.float32 d);
        equal (array float_exact) [| 1.; 2.; 3. |] (Column.values Kind.float d));
    test "donating the ragged array a list column took or gave leaves it"
      (fun () ->
        let offsets = Nx.create Nx.int64 [| 3 |] [| 0L; 2L; 3L |] in
        let r = Nx_ragged.v ~offsets (floats ()) in
        let c = Column.of_ragged r in
        consume (Nx_ragged.values r);
        consume (Nx_ragged.offsets r);
        let given = Column.ragged Nx.float32 c in
        consume (Nx_ragged.values given);
        consume (Nx_ragged.offsets given);
        equal
          (array (array float_exact))
          [| [| 1.; 2. |]; [| 3. |] |]
          (Column.values (Kind.list Kind.float) c));
    test "a dead value a column would take raises naming the function"
      (fun () ->
        let dead x =
          ignore (Nx.donate x);
          x
        in
        let refused by value f =
          raises_match
            (Exn.invalid_arg ~sub:(by ^ ": " ^ value ^ " was donated"))
            f
        in
        refused "Column.of_tensor" "float32 [3]" (fun () ->
            Column.of_tensor (dead (floats ())));
        refused "Column.of_tensor" "bit [3]" (fun () ->
            Column.of_tensor ~validity:(dead (three_bits ())) (floats ()));
        let offsets = Nx.create Nx.int64 [| 3 |] [| 0L; 2L; 3L |] in
        let r = Nx_ragged.v ~offsets (floats ()) in
        ignore (dead (Nx_ragged.values r));
        refused "Column.of_ragged" "float32 [3]" (fun () -> Column.of_ragged r);
        let values = Nx.P (dead (floats ())) in
        refused "Column.of_layout" "float32 [3]" (fun () ->
            Column.of_layout (Type.Any Type.float32)
              (Fixed { validity = None; values })));
  ]

let positions_of n =
  Gen.list ~size:(Gen.int_range 0 12) (Gen.int_range (-1) (n + 1))

let sample_positions =
  Gen.bind G.sample (fun (G.Sample (_, vs) as s) ->
      Gen.map (fun is -> (s, is)) (positions_of (Array.length vs)))
  |> Gen.with_pp (fun ppf (s, is) ->
      Format.fprintf ppf "@[<v>%a@,at %a@]" G.pp_sample s
        (Format.pp_print_list ~pp_sep:Format.pp_print_space Format.pp_print_int)
        is)

(* [reversed c] is [c]'s rows sorted by a key that reverses them. *)
let reversed c =
  let n = Column.length c in
  let key = Nx.init Nx.int64 [| n |] (fun i -> Int64.of_int (n - 1 - i.(0))) in
  let t = v [ ("k", Column.of_tensor key); ("x", c) ] in
  column (Error.get_ok Query.(run (of_table t |> sort [ Order.asc "k" ]))) "x"

(* The values under the nulls that a join pads and a sort moves are the same on
   every run of one plan. *)
let deterministic (G.Sample (ty, vs), is) =
  let c = Column.of_options ty vs in
  cover "a null" (nulls vs > 0);
  cover "a padded row" (List.exists (fun i -> i < 0 || i >= Array.length vs) is);
  equal column_w (padded is c) (padded is c);
  equal column_w (reversed c) (reversed c)

(* null_count

   A validity's count is read at the first null_count and kept. The model is a
   column's validity as booleans; columns made from others count their own
   nulls, and two domains that count at once agree with some order of the
   counts. *)

let column_of m = Column.of_tensor ~validity:(bits m) (int64s (Array.length m))
let count_model m = Array.fold_left (fun n v -> if v then n else n + 1) 0 m
let columns = abstract "c"

let masks =
  Gen.with_pp
    (fun ppf m ->
      Array.iter (fun v -> Format.pp_print_char ppf (if v then '1' else '0')) m)
    (Gen.array ~size:(Gen.int_range 0 40) Gen.bool)

(* Positions, outside the rows too. *)
let picks =
  Gen.with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space Format.pp_print_int)
    (Gen.list ~size:(Gen.int_range 0 12) (Gen.int_range (-2) 45))

let take_model is m =
  Array.of_list (List.map (fun i -> i >= 0 && i < Array.length m && m.(i)) is)

(* [column_at m] is [column_of m] with its validity a view from bit 5 of longer
   bits, set around it. *)
let column_at m =
  let n = Array.length m in
  let set =
    Array.init (n + 14) (fun i ->
        if i < 5 || i >= n + 5 then true else m.(i - 5))
  in
  let validity = Nx.slice [ Nx.R (5, n + 5) ] (bits set) in
  Column.of_tensor ~validity (int64s n)

let x_null = Expr.is_null (Col.int "x")
let valid_rows m = Array.make (Array.length m - count_model m) true
let lengths = Gen.with_pp Format.pp_print_int (Gen.int_range 0 45)

let slice_model offset length m =
  let n = Array.length m in
  let lo = min offset n in
  Array.sub m lo (min length (n - lo))

let counts =
  [
    command "of_tensor" (masks @-> makes columns) Fun.id column_of;
    command "of_tensor at bit 5" (masks @-> makes columns) Fun.id column_at;
    command "filter to its nulls"
      (columns ^-> makes columns)
      (fun m -> Array.make (count_model m) false)
      (run_x (Query.filter x_null));
    command "filter to its values"
      (columns ^-> makes columns)
      valid_rows
      (run_x (Query.filter (Expr.not x_null)));
    command "sort, nulls first"
      (columns ^-> makes columns)
      (fun m -> Array.append (Array.make (count_model m) false) (valid_rows m))
      (run_x (Query.sort [ Order.nulls_first (Order.asc "x") ]));
    command "slice"
      (lengths @-> lengths @-> columns ^-> makes columns)
      slice_model
      (fun offset length -> run_x (Query.slice ~offset ~length));
    command "null_count" (columns ^-> returns int) count_model Column.null_count;
    command "take"
      (picks @-> columns ^-> makes columns)
      (fun is m -> take_model (inside (Array.length m) is) m)
      (fun is c -> taken (inside (Column.length c) is) c);
    command "left join" (picks @-> columns ^-> makes columns) take_model padded;
    command "concat"
      (columns ^-> columns ^-> makes columns)
      (fun a b -> Array.append a b)
      (fun a b ->
        let t c = v [ ("x", c) ] in
        column (of_batches [ t a; t b ]) "x");
    command "validity"
      (columns ^-> returns (option (array bool)))
      (fun m -> if count_model m = 0 then None else Some m)
      (fun c ->
        match Column.validity c with
        | Some v when Column.null_count c > 0 -> Some (Nx.to_array v)
        | _ -> None);
  ]

let null_count =
  [
    stateful "is the number of null rows, read once" counts;
    stateful ~domains:2 "is the number of null rows from two domains at once"
      counts;
    test "is 0 for a validity with every row set" (fun () ->
        equal int 0 (Column.null_count (column_of [| true; true; true |])));
    test "counts a validity's rows only, whatever its bits past them" (fun () ->
        (* Two bytes of ones, the second's bits past row 13 set too, and row 2
           cleared. *)
        let bytes = Nx.create Nx.uint8 [| 2 |] [| 0xfb; 0xff |] in
        let validity =
          Nx.slice
            [ Nx.R (0, 13) ]
            (Nx.reshape [| -1 |] (Nx.bitcast Nx.bit bytes))
        in
        let c = Column.of_tensor ~validity (int64s 13) in
        equal int 1 (Column.null_count c);
        let b = Option.get (Column.validity (column (v [ ("x", c) ]) "x")) in
        Nx.Repr.read ~by:"test" b @@ fun a ->
        Rig.Buffer.read (Nx_array.buffer a) Bigarray.int8_unsigned (fun raw ->
            equal int 0 (raw.{1} lsr 5)));
  ]

(* v, of_options, values and options *)

(* Columns at the edges of their types, which the laws below run first: no row,
   no row of a record, every row null, an extension's nulls, integer bounds,
   -0., NaN, infinities, the largest finite float32 and float16, the first and
   last dates, clocks and instants, empty text, NUL and empty lists. *)
let edges =
  let i32 = Int32.(to_int min_int, to_int max_int) in
  Type.
    [
      G.Sample (int8, [||]);
      G.Sample (record [ ("a", Any string) ], [||]);
      G.Sample (string, [| None; None |]);
      G.Sample (ext ~name:"m" float64, [| None |]);
      G.Sample (int8, [| Some (-128); Some 127 |]);
      G.Sample (int16, [| Some (-32768); Some 32767 |]);
      G.Sample (int32, [| Some (fst i32); Some (snd i32) |]);
      G.Sample (int64, [| Some min_int; Some max_int |]);
      G.Sample (uint8, [| Some 0; Some 255 |]);
      G.Sample (uint16, [| Some 0; Some 65535 |]);
      G.Sample (uint32, [| Some 0; Some 0xffff_ffff |]);
      G.Sample (uint64, [| Some 0; Some max_int |]);
      G.Sample
        ( float64,
          [|
            Some (-0.);
            Some Float.nan;
            Some Float.infinity;
            Some Float.neg_infinity;
          |] );
      G.Sample (float32, [| Some (-0.); Some 0x1.fffffep127 |]);
      G.Sample (float16, [| Some (-0.); Some 65504. |]);
      G.Sample (date, [| Some (day (fst i32)); Some (day (snd i32)) |]);
      G.Sample (clock S, [| Some (span 0L); Some (span 86_399_000_000_000L) |]);
      G.Sample
        (wall Ns, [| Some (ticks Int64.min_int); Some (ticks Int64.max_int) |]);
      G.Sample
        ( wall S,
          [|
            Some (ticks (-9_223_372_036_000_000_000L));
            Some (ticks 9_223_372_036_000_000_000L);
          |] );
      G.Sample (string, [| Some ""; Some "\000" |]);
      G.Sample (list string, [| Some [||]; Some [| "" |] |]);
    ]

let has_ext_field fields =
  List.exists (function _, Type.Any (Type.Ext _) -> true | _ -> false) fields

let record_of_ext_record fields =
  let inner = function
    | _, Type.Any (Type.Record fs) -> has_ext_field fs
    | _ -> false
  in
  List.exists inner fields

(* A record of a record with an extension field is drawn in at least one case of
   seven: at 150 cases a run misses it with probability (6/7)^150 = 9e-11. *)
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
    raises_match Exn.invalid_arg (fun () ->
        Column.options (Type.kind ty) (Column.of_options ty vs))

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

let ab = Record.(empty |> add Kind.int "a" (Some 1))
let a_float = Record.(empty |> add Kind.float "a" (Some 1.))
let a_300 = Record.(empty |> add Kind.int "a" (Some 300))

let v_refusals () =
  let open Type in
  expect (refusal (fun () -> Column.v int8 [| 1; 300; 400 |]))
  @@ __POS_OF__ {| Column.v: row 1: int8 does not hold 300 |};
  expect (refusal (fun () -> Column.v float16 [| 65520. |]))
  @@ __POS_OF__ {| Column.v: row 0: float16 does not hold 65520. |};
  expect (refusal (fun () -> Column.of_options string [| None; Some "a\xff" |]))
  @@ __POS_OF__ {| Column.of_options: row 1: string does not hold "a\xff" |};
  expect (refusal (fun () -> Column.v (categorical [| "x" |]) [| "y" |]))
  @@ __POS_OF__ {| Column.v: row 0: categorical["x"] does not hold "y" |};
  expect (refusal (fun () -> Column.v (clock S) [| Time.Span.hours 24 |]))
  @@ __POS_OF__ {| Column.v: row 0: clock[s] does not hold 24h |};
  expect (refusal (fun () -> Column.v (duration S) [| Time.Span.ms 1500 |]))
  @@ __POS_OF__ {| Column.v: row 0: duration[s] does not hold 1s500ms |};
  expect
    (refusal (fun () ->
         Column.v (tensor Nx.float32 [| 2 |]) [| Nx.zeros Nx.float32 [| 3 |] |]))
  @@ __POS_OF__
       {| Column.v: row 0: tensor[float32, 2] does not hold a tensor of shape [3] |};
  expect (refusal (fun () -> Column.v (list int8) [| [| 1 |]; [| 2; 300 |] |]))
  @@ __POS_OF__ {| Column.v: row 1: element 1: int8 does not hold 300 |};
  expect (refusal (fun () -> Column.v (record [ ("b", Any int8) ]) [| ab |]))
  @@ __POS_OF__
       {| Column.v: row 0: record[b int8] does not hold a record of fields [a] |};
  expect
    (refusal (fun () -> Column.v (record [ ("a", Any int8) ]) [| a_float |]))
  @@ __POS_OF__ {| Column.v: row 0: field a: int8 does not hold a float value |};
  expect (refusal (fun () -> Column.v (record [ ("a", Any int8) ]) [| a_300 |]))
  @@ __POS_OF__ {| Column.v: row 0: field a: int8 does not hold 300 |}

let int8s = Column.of_options Type.int8 [| Some 1; None |]

(* [ticks_column ty ts] is the column of [ty] whose ticks are [ts]. *)
let ticks_column ty ts =
  let values = Nx.P (Nx.create Nx.int64 [| Array.length ts |] ts) in
  Result.get_ok (Column.of_layout (Any ty) (Fixed { validity = None; values }))

let read_refusals () =
  let big =
    Column.of_tensor (Nx.create Nx.int64 [| 2 |] [| 0L; Int64.max_int |])
  in
  let wide = Column.of_tensor (Nx.create Nx.uint64 [| 1 |] [| -1L |]) in
  let ext = Type.ext ~name:"m" Type.float64 in
  expect (refusal (fun () -> Column.values Kind.int int8s))
  @@ __POS_OF__ {| Column.values: row 1 is null |};
  expect (refusal (fun () -> Column.options Kind.float int8s))
  @@ __POS_OF__ {| Column.options: int8 is not read as float |};
  expect
    (refusal (fun () ->
         Column.options (Type.kind ext) (Column.of_options ext [| None |])))
  @@ __POS_OF__ {| Column.options: no kind reads ext[m, float64] |};
  expect (refusal (fun () -> Column.options Kind.int big))
  @@ __POS_OF__ {| Column.options: row 1: 9223372036854775807 is outside int |};
  expect (refusal (fun () -> Column.values Kind.int wide))
  @@ __POS_OF__ {| Column.values: row 0: 18446744073709551615 is outside int |};
  expect
    (refusal (fun () ->
         let validity = bits [| false; true; true |] in
         let x = Nx.create Nx.uint64 [| 3 |] [| -1L; 5L; -1L |] in
         Column.options Kind.int (Column.of_tensor ~validity x)))
  @@ __POS_OF__ {| Column.options: row 2: 18446744073709551615 is outside int |};
  let wall = Type.wall Type.S in
  expect
    (refusal (fun () ->
         Column.values (Type.kind wall)
           (ticks_column wall [| 0L; Int64.max_int |])))
  @@ __POS_OF__
       {| Column.values: row 1: datetime[s] tick 9223372036854775807 is outside the range |};
  let duration = Type.duration Type.S in
  expect
    (refusal (fun () ->
         Column.values (Type.kind duration)
           (ticks_column duration [| Int64.min_int |])))
  @@ __POS_OF__
       {| Column.values: row 0: duration[s] tick -9223372036854775808 is outside the range |};
  expect
    (refusal (fun () ->
         let child = Column.of_options Type.int32 [| Some 1; None |] in
         let l =
           Column.Varsize
             { validity = None; offsets = n [| 0L; 1L; 2L |]; child }
         in
         let c = Result.get_ok (Column.of_layout (Any Type.(list int32)) l) in
         Column.values Kind.(list int) c))
  @@ __POS_OF__ {| Column.values: row 1: element 0 is null |}

(* Tensors *)

(* The type of a 1-D tensor of each dtype a scalar type stores. *)
let scalar_types =
  let row name dt ty = (name, Nx.P (Nx.zeros dt [| 2 |]), ty) in
  Type.
    [
      row "bit" Nx.bit (Any bool);
      row "bool" Nx.bool (Any bool);
      row "int8" Nx.int8 (Any int8);
      row "int16" Nx.int16 (Any int16);
      row "int32" Nx.int32 (Any int32);
      row "int64" Nx.int64 (Any int64);
      row "uint8" Nx.uint8 (Any uint8);
      row "uint16" Nx.uint16 (Any uint16);
      row "uint32" Nx.uint32 (Any uint32);
      row "uint64" Nx.uint64 (Any uint64);
      row "float16" Nx.float16 (Any float16);
      row "float32" Nx.float32 (Any float32);
      row "float64" Nx.float64 (Any float64);
    ]

(* 1-D tensors of the dtypes no scalar type stores. *)
let unstored =
  let row name dt = (name, Nx.P (Nx.zeros dt [| 2 |])) in
  [
    row "bfloat16" Nx.bfloat16;
    row "float8_e4m3fn" Nx.float8_e4m3fn;
    row "float8_e5m2" Nx.float8_e5m2;
    row "int4" Nx.int4;
    row "uint4" Nx.uint4;
    row "complex64" Nx.complex64;
    row "complex128" Nx.complex128;
  ]

let of_tensor_refusals () =
  expect (refusal (fun () -> Column.of_tensor (Nx.scalar Nx.int8 1)))
  @@ __POS_OF__ {| Column.of_tensor: a scalar has no rows |};
  expect
    (refusal (fun () ->
         Column.of_tensor ~validity:(bits [| true |]) (Nx.zeros Nx.int8 [| 2 |])))
  @@ __POS_OF__ {| Column.of_tensor: a validity of shape [1] for 2 rows |}

let to_tensor_refusals () =
  expect
    (refusal (fun () ->
         Column.to_tensor Nx.bool (Column.v Type.bool [| true |])))
  @@ __POS_OF__ {| Column.to_tensor: bool is stored as bit, not bool |};
  expect
    (refusal (fun () ->
         Column.to_tensor Nx.int64 (Column.v Type.date [| day 0 |])))
  @@ __POS_OF__ {| Column.to_tensor: date is stored as int32, not int64 |};
  expect (refusal (fun () -> Column.to_tensor Nx.int8 int8s))
  @@ __POS_OF__ {| Column.to_tensor: row 1 is null |};
  expect
    (refusal (fun () ->
         Column.to_tensor Nx.uint8 (Column.v Type.string [| "a" |])))
  @@ __POS_OF__ {| Column.to_tensor: string is not stored one element per row |}

(* Ragged arrays *)

(* The type for a dtype, its values' equality and a generator of them. *)
type dtype =
  | Dtype : string * ('a, 'b) Nx.dtype * ('a -> 'a -> bool) * 'a Gen.t -> dtype

(* The type for a ragged array of a named dtype, with its values' equality. *)
type ragged =
  | Ragged :
      string
      * ('a, 'b) Nx.dtype
      * ('a -> 'a -> bool)
      * ('a, 'b, Nx.host) Nx_ragged.t
      -> ragged

let bits_equal a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let dtypes =
  let int lo hi = Gen.int_range lo hi in
  [
    Dtype ("bit", Nx.bit, Bool.equal, Gen.bool);
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

(* [ragged_gen dt v cell] draws up to six rows of up to four values of shape
   [cell], over values that may hold rows before the first offset and after the
   last. *)
let ragged_gen (type a b) (dt : (a, b) Nx.dtype) (v : a Gen.t) cell =
  let open Gen in
  let* lengths = list ~size:(int_range 0 6) (int_range 0 4) in
  let* before = int_range 0 2 in
  let* after = int_range 0 2 in
  let total = before + List.fold_left ( + ) 0 lengths + after in
  let size = total * Array.fold_left ( * ) 1 cell in
  let+ values = array ~size:(constant size) v in
  let offsets =
    List.fold_left (fun acc l -> (List.hd acc + l) :: acc) [ before ] lengths
    |> List.rev_map Int64.of_int |> Array.of_list
  in
  Nx_ragged.v ~offsets:(n offsets)
    (Nx.create dt (Array.append [| total |] cell) values)

(* Ragged arrays are equal when their rows hold equal values of one cell
   shape. *)
let ragged_w (type a b) (eq : a -> a -> bool) :
    (a, b, Nx.host) Nx_ragged.t Testable.t =
  let cell r =
    Array.sub
      (Nx.shape (Nx_ragged.values r))
      1
      (Nx.ndim (Nx_ragged.values r) - 1)
  in
  let rows r =
    let o = Nx.to_array (Nx_ragged.offsets r)
    and v = Nx.to_array (Nx_ragged.values r) in
    let size = Array.fold_left ( * ) 1 (cell r) in
    Array.init
      (Array.length o - 1)
      (fun i ->
        let first = Int64.to_int o.(i) * size in
        Array.sub v first ((Int64.to_int o.(i + 1) * size) - first))
  in
  Testable.make
    ~pp:(fun ppf r ->
      Format.fprintf ppf "%d rows of lengths %a, cells %a" (Nx_ragged.length r)
        Nx.pp (Nx_ragged.lengths r) Nx.pp_shape (cell r))
    ~equal:(fun a b ->
      let ra = rows a and rb = rows b in
      cell a = cell b
      && Array.length ra = Array.length rb
      && Array.for_all2
           (fun x y -> Array.length x = Array.length y && Array.for_all2 eq x y)
           ra rb)

let cells = [ [||]; [| 2 |]; [| 2; 0 |] ]

let raggeds =
  Gen.bind (Gen.of_list dtypes) (fun (Dtype (name, dt, eq, v)) ->
      Gen.bind (Gen.of_list cells) (fun cell ->
          Gen.map (fun r -> Ragged (name, dt, eq, r)) (ragged_gen dt v cell)))
  |> Gen.with_pp (fun ppf (Ragged (name, _, _, r)) ->
      Format.fprintf ppf "%d rows of %s, values of shape %a"
        (Nx_ragged.length r) name Nx.pp_shape
        (Nx.shape (Nx_ragged.values r)))

(* For each dtype, no row, and an empty row between values before and after
   it. *)
let ragged_edges =
  List.concat_map
    (fun (Dtype (name, dt, eq, _)) ->
      [
        Ragged
          (name, dt, eq, Nx_ragged.v ~offsets:(n [| 0L |]) (Nx.zeros dt [| 0 |]));
        Ragged
          ( name,
            dt,
            eq,
            Nx_ragged.v ~offsets:(n [| 1L; 1L |]) (Nx.zeros dt [| 2 |]) );
      ])
    dtypes

(* Each of the 12 dtypes is drawn in a case of 12: at 300 cases one is missed
   with probability (11/12)^300 = 5e-12; the examples hold each one's no row and
   empty row. *)
let ragged_law (Ragged (name, dt, eq, r)) =
  List.iter (fun (Dtype (d, _, _, _)) -> cover d (String.equal d name)) dtypes;
  cover "no row" (Nx_ragged.length r = 0);
  cover "an empty row"
    (Array.exists (Int64.equal 0L) (Nx.to_array (Nx_ragged.lengths r)));
  cover "values of two axes" (Nx.ndim (Nx_ragged.values r) > 1);
  Law.round_trip (ragged_w eq) column_w Column.of_ragged (Column.ragged dt) r

let bool_raggeds =
  Gen.with_pp
    (fun ppf r -> Format.fprintf ppf "%d rows" (Nx_ragged.length r))
    (ragged_gen Nx.bool Gen.bool [||])

let packed r =
  let bits =
    Nx_ragged.v ~offsets:(Nx_ragged.offsets r)
      (Nx.cast Nx.bit (Nx_ragged.values r))
  in
  equal (ragged_w Bool.equal) bits (Column.ragged Nx.bit (Column.of_ragged r))

let text_or_bytes =
  Gen.pair Gen.bool
    (Gen.array ~size:(Gen.int_range 0 8) (Option.get (G.value Type.string)))

let row_bytes (binary, ss) =
  cover "binary" binary;
  let c =
    if binary then Column.v Type.binary (Array.map Binary.of_string ss)
    else Column.v Type.string ss
  in
  let r = Column.ragged Nx.uint8 c in
  equal (array int)
    (Array.map String.length ss)
    (Array.map Int64.to_int (Nx.to_array (Nx_ragged.lengths r)));
  let bytes = String.concat "" (Array.to_list ss) in
  let o = Int64.to_int (Nx.item [ 0 ] (Nx_ragged.offsets r)) in
  equal (array int)
    (Array.init (String.length bytes) (fun i -> Char.code bytes.[i]))
    (Array.sub (Nx.to_array (Nx_ragged.values r)) o (String.length bytes))

let ext_storage () =
  let ext ty c =
    require_ok
      (Column.of_layout (Any (Type.ext ~name:"m" ty)) (Column.layout c))
  in
  let texts = Column.v Type.string [| "ab"; ""; "é" |] in
  equal (ragged_w Int.equal)
    (Column.ragged Nx.uint8 texts)
    (Column.ragged Nx.uint8 (ext Type.string texts));
  let lists = Column.v Type.(list int32) [| [| 1; 2 |]; [||] |] in
  equal (ragged_w Int32.equal)
    (Column.ragged Nx.int32 lists)
    (Column.ragged Nx.int32 (ext Type.(list int32) lists))

let ragged_refusals () =
  let open Type in
  expect (refusal (fun () -> Column.ragged Nx.int8 (Column.v int8 [| 1 |])))
  @@ __POS_OF__ {| Column.ragged: int8 is not a list of int8, text or bytes |};
  expect
    (refusal (fun () ->
         Column.ragged Nx.int64 (Column.v (list int32) [| [| 1 |] |])))
  @@ __POS_OF__ {| Column.ragged: list[int32] is stored as int32, not int64 |};
  expect (refusal (fun () -> Column.ragged Nx.int8 (Column.v string [| "a" |])))
  @@ __POS_OF__ {| Column.ragged: string is stored as uint8, not int8 |};
  expect
    (refusal (fun () ->
         Column.ragged Nx.uint8 (Column.v (list string) [| [| "a" |] |])))
  @@ __POS_OF__
       {| Column.ragged: list[string] is not a list of uint8, text or bytes |};
  expect
    (refusal (fun () ->
         Column.ragged Nx.uint8
           (Column.of_options binary [| Some (Binary.of_string "a"); None |])))
  @@ __POS_OF__ {| Column.ragged: row 1 is null |};
  expect
    (refusal (fun () ->
         let elements = Column.of_options int32 [| Some 1; Some 2; None |] in
         let l =
           Column.Varsize
             { validity = None; offsets = n [| 0L; 2L; 3L |]; child = elements }
         in
         Column.ragged Nx.int32
           (Result.get_ok (Column.of_layout (Any (list int32)) l))))
  @@ __POS_OF__ {| Column.ragged: an element of row 1 is null |}

let of_ragged_refusals () =
  expect
    (refusal (fun () ->
         Column.of_ragged ~validity:(bits [| true |])
           (Nx_ragged.v
              ~offsets:(n [| 0L; 1L; 1L |])
              (Nx.create Nx.int32 [| 1 |] [| 7l |]))))
  @@ __POS_OF__ {| Column.of_ragged: a validity of shape [1] for 2 rows |};
  expect
    (refusal (fun () ->
         Column.of_ragged
           (Nx_ragged.v
              ~offsets:(n [| 0L; 1L |])
              (Nx.zeros Nx.bfloat16 [| 1 |]))))
  @@ __POS_OF__
       {| Column.of_ragged: no scalar type stores bfloat16; make it 2-D |}

(* of_layout *)

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

(* Rows of a layout that a type holds or not, and the first row it does not
   hold. *)
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

(* An overlong form, a surrogate, a value past U+10FFFF, a sequence cut by the
   row's end, and a last row. *)
let text_edges =
  [
    [ "a"; "\xc0\xaf" ];
    [ "ab\xed\xa0\x80" ];
    [ "\xf4\x90\x80\x80" ];
    [ "\xe2\x82"; "\xac" ];
    [ "é"; "日本"; "\xff" ];
  ]

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

let int8s_layout = fixed Nx.int8 [| 1; 2 |]
let child = Column.v Type.int8 [| 1; 2 |]
let text_child = Column.v Type.uint8 [| 97; 98 |]
let of_layout ty l () = Column.of_layout (Type.Any ty) l

let varsize offsets child =
  Column.Varsize { validity = None; offsets = n offsets; child }

let children ?(length = 2) fields =
  Column.Children { validity = None; length; fields }

let layout_refusals () =
  let open Type in
  let refused ty l = refusal (of_layout ty l) in
  expect (refused int16 int8s_layout)
  @@ __POS_OF__
       {| Column.of_layout: int8 values of shape [2] do not lay out int16 |};
  expect
    (refused (tensor Nx.int8 [| 3 |])
       (Fixed { validity = None; values = P (Nx.zeros Nx.int8 [| 2; 2 |]) }))
  @@ __POS_OF__
       {| Column.of_layout: int8 values of shape [2; 2] do not lay out tensor[int8, 3] |};
  expect
    (refused int8 (Fixed { validity = None; values = P (Nx.scalar Nx.int8 1) }))
  @@ __POS_OF__
       {| Column.of_layout: int8 values of shape [] do not lay out int8 |};
  expect (refused int8 (fixed ~validity:(bits [| true |]) Nx.int8 [| 1; 2 |]))
  @@ __POS_OF__ {| Column.of_layout: a validity of shape [1] for 2 rows |};
  expect
    (refused (list int8)
       (Varsize
          { validity = None; offsets = Nx.zeros Nx.int64 [| 1; 1 |]; child }))
  @@ __POS_OF__
       {| Column.of_layout: offsets of shape [1; 1], not 1-D with an entry |};
  expect (refused (list int8) (varsize [||] child))
  @@ __POS_OF__
       {| Column.of_layout: offsets of shape [0], not 1-D with an entry |};
  expect (refused (list int8) (varsize [| -1L; 0L |] child))
  @@ __POS_OF__ {| Column.of_layout: offsets start at -1 |};
  expect (refused (list int8) (varsize [| 0L; 2L; 1L |] child))
  @@ __POS_OF__ {| Column.of_layout: offsets decrease at row 1 |};
  expect (refused (list int8) (varsize [| 0L; 3L |] child))
  @@ __POS_OF__
       {| Column.of_layout: offsets end at 3, past the child's 2 rows |};
  expect (refused (list int16) (varsize [| 0L; 2L |] child))
  @@ __POS_OF__
       {| Column.of_layout: a child of int8 does not lay out list[int16] |};
  expect (refused string (varsize [| 0L; 2L |] child))
  @@ __POS_OF__ {| Column.of_layout: a child of int8 does not lay out string |};
  expect
    (refused string
       (varsize [| 0L; 1L |] (Column.of_options uint8 [| Some 97; None |])))
  @@ __POS_OF__
       {| Column.of_layout: a child with a null does not lay out string |};
  expect (refused (record [ ("a", Any int8) ]) (children [ ("b", child) ]))
  @@ __POS_OF__ {| Column.of_layout: fields [b] do not lay out record[a int8] |};
  expect (refused (record [ ("a", Any int16) ]) (children [ ("a", child) ]))
  @@ __POS_OF__
       {| Column.of_layout: field a of int8 does not lay out record[a int16] |};
  expect
    (refused (record [ ("a", Any int8) ]) (children ~length:3 [ ("a", child) ]))
  @@ __POS_OF__ {| Column.of_layout: field a of 2 rows for 3 |};
  expect (refused (record []) (children ~length:(-1) []))
  @@ __POS_OF__ {| Column.of_layout: a length of -1 |};
  expect (refused int8 (varsize [| 0L; 2L |] child))
  @@ __POS_OF__ {| Column.of_layout: offsets and a child do not lay out int8 |};
  expect (refused (list int8) (children []))
  @@ __POS_OF__ {| Column.of_layout: fields do not lay out list[int8] |}

(* Text offsets are checked by Nx_ragged.v, whose refusal the .mli names. *)
let text_offsets_refusals () =
  let refused l = refusal (of_layout Type.string l) in
  expect
    (refused
       (Varsize
          {
            validity = None;
            offsets = Nx.zeros Nx.int64 [| 1; 1 |];
            child = text_child;
          }))
  @@ __POS_OF__
       {| Nx_ragged.v: offsets of shape [1; 1], not 1-D with an entry |};
  expect (refused (varsize [||] text_child))
  @@ __POS_OF__ {| Nx_ragged.v: offsets of shape [0], not 1-D with an entry |};
  expect (refused (varsize [| -1L; 0L |] text_child))
  @@ __POS_OF__ {| Nx_ragged.v: offsets start at -1, below 0 |};
  expect (refused (varsize [| 0L; 2L; 1L |] text_child))
  @@ __POS_OF__ {| Nx_ragged.v: offset 2 is 1, below the 2 before it |};
  expect (refused (varsize [| 0L; 3L |] text_child))
  @@ __POS_OF__ {| Nx_ragged.v: offset 1 is 3, past the 2 rows of the values |}

(* Taking and joining rows

   Text rows, some null, the last [len] of which a slice keeps, so that the kept
   rows' offsets start past [0]; and positions into the kept rows, some outside
   them. *)
let text_picks =
  let row =
    Gen.frequency
      [
        (1, Gen.constant None);
        (5, Gen.map Option.some (Option.get (G.value Type.string)));
      ]
  in
  Gen.bind
    (Gen.array ~size:(Gen.int_range 0 12) row)
    (fun vs ->
      let n = Array.length vs in
      Gen.triple (Gen.constant vs) (Gen.int_range 0 n)
        (Gen.list ~size:(Gen.int_range 0 12) (Gen.int_range (-2) (n + 1))))

(* Kept rows with a null and an empty text, taken at positions that hold
   both. *)
let text_edge = ([| Some "x"; None; Some ""; Some "ab" |], 3, [ 0; 1; 2; -1; 3 ])

let kept_rows (vs, len, _) =
  let n = Array.length vs in
  let c = Column.of_options Type.string vs in
  ( Array.sub vs (n - len) len,
    run_x (Query.slice ~offset:(n - len) ~length:len) c )

let text_covers kept is =
  let len = Array.length kept in
  let at i = if i >= 0 && i < len then kept.(i) else None in
  cover "a null row"
    (List.exists (fun i -> i >= 0 && i < len && at i = None) is);
  cover "an empty text" (List.exists (fun i -> at i = Some "") is);
  cover "a text of several bytes"
    (List.exists
       (function Some s -> String.length s > 1 | None -> false)
       (List.map at is));
  at

let take_text ((vs, len, is) as picks) =
  let kept, c = kept_rows picks in
  let is = inside len is in
  let at = text_covers kept is in
  cover "kept rows after the first" (len < Array.length vs && len > 0);
  equal (rows Type.string)
    (Array.of_list (List.map at is))
    (Column.options Kind.string (taken is c))

let join_text ((_, len, is) as picks) =
  let kept, c = kept_rows picks in
  let at = text_covers kept is in
  cover "a position outside the kept rows"
    (List.exists (fun i -> i < 0 || i >= len) is);
  equal (rows Type.string)
    (Array.of_list (List.map at is))
    (Column.options Kind.string (padded is c))

(* List rows, some null, their elements some null, after [pre] elements no row
   holds; and positions into the rows, some outside them. *)
let list_picks =
  let element =
    Gen.frequency
      [ (1, Gen.constant None); (4, Gen.map Option.some (Gen.int_range 0 99)) ]
  in
  let row =
    Gen.frequency
      [
        (1, Gen.constant None);
        (4, Gen.map Option.some (Gen.array ~size:(Gen.int_range 0 4) element));
      ]
  in
  Gen.with_pp
    (fun ppf (_, pre, is) ->
      Format.fprintf ppf "after %d, at %a" pre
        (Format.pp_print_list ~pp_sep:Format.pp_print_space Format.pp_print_int)
        is)
    (Gen.bind
       (Gen.array ~size:(Gen.int_range 0 10) row)
       (fun rows ->
         let n = Array.length rows in
         Gen.triple (Gen.constant rows) (Gen.int_range 0 3)
           (Gen.list ~size:(Gen.int_range 0 12) (Gen.int_range (-2) (n + 1)))))

(* [list_column rows pre] is the [list[int64]] column of [rows], its elements
   after [pre] that no row holds. *)
let list_column rows pre =
  let n = Array.length rows in
  let elements =
    Array.concat
      (Array.make pre (Some (-1))
      :: Array.to_list (Array.map (function Some r -> r | None -> [||]) rows))
  in
  let offsets = Array.make (n + 1) (Int64.of_int pre) in
  Array.iteri
    (fun i r ->
      let l = match r with Some r -> Array.length r | None -> 0 in
      offsets.(i + 1) <- Int64.add offsets.(i) (Int64.of_int l))
    rows;
  let m = Array.length elements in
  let child =
    Column.of_tensor
      ~validity:(bits (Array.map Option.is_some elements))
      (Nx.create Nx.int64 [| m |]
         (Array.map (function Some v -> Int64.of_int v | None -> 0L) elements))
  in
  Result.get_ok
    (Column.of_layout
       (Any (Type.list Type.int64))
       (Varsize
          {
            validity = Some (bits (Array.map Option.is_some rows));
            offsets = Nx.create Nx.int64 [| n + 1 |] offsets;
            child;
          }))

(* [list_rows y] is the rows of the list column [y], read through its layout. *)
let list_rows y =
  match Column.layout y with
  | Varsize { validity; offsets; child } ->
      let o = Nx.to_array offsets and e = Column.options Kind.int child in
      let valid k =
        match validity with
        | Some v -> Nx.item [ k ] (Nx.cast Nx.bool v)
        | None -> true
      in
      Array.init (Column.length y) (fun k ->
          if not (valid k) then None
          else
            let lo = Int64.to_int o.(k) and hi = Int64.to_int o.(k + 1) in
            Some (Array.sub e lo (hi - lo)))
  | Fixed _ | Children _ -> fail "a list column's rows are not a list"

let list_w = array (option (array (option int)))

let null_element_taken rows is =
  List.exists
    (fun i ->
      i >= 0
      && i < Array.length rows
      && match rows.(i) with Some r -> Array.mem None r | None -> false)
    is

let take_lists (rows, pre, is) =
  let is = inside (Array.length rows) is in
  cover "a null element taken" (null_element_taken rows is);
  cover "elements before the first row" (pre > 0);
  equal list_w
    (Array.of_list (List.map (fun i -> rows.(i)) is))
    (list_rows (taken is (list_column rows pre)))

let join_lists (rows, pre, is) =
  let n = Array.length rows in
  cover "a null element taken" (null_element_taken rows is);
  cover "a position outside the rows"
    (List.exists (fun i -> i < 0 || i >= n) is);
  let at i = if i >= 0 && i < n then rows.(i) else None in
  equal list_w
    (Array.of_list (List.map at is))
    (list_rows (padded is (list_column rows pre)))

(* Nested columns: a type of lists and records over int64 and text, and a model
   of its rows with nulls at every depth, a list's elements and a record's
   fields being cells too. *)
type shape = Ints | Texts | Lists of shape | Records of shape list

type cell =
  | Null
  | Int of int
  | Text of string
  | Items of cell array
  | Fields of cell array

let field_name i = "f" ^ string_of_int i

let rec type_of = function
  | Ints -> Type.Any Type.int64
  | Texts -> Type.Any Type.string
  | Lists e ->
      let (Type.Any t) = type_of e in
      Type.Any (Type.list t)
  | Records fs ->
      let field i f = (field_name i, type_of f) in
      Type.Any (Type.record (List.mapi field fs))

let comma ppf () = Format.pp_print_string ppf ", "

let rec pp_shape ppf = function
  | Ints -> Format.pp_print_string ppf "int64"
  | Texts -> Format.pp_print_string ppf "string"
  | Lists e -> Format.fprintf ppf "list[%a]" pp_shape e
  | Records fs ->
      Format.fprintf ppf "{%a}" (Format.pp_print_list ~pp_sep:comma pp_shape) fs

let rec pp_cell ppf = function
  | Null -> Format.pp_print_string ppf "null"
  | Int i -> Format.pp_print_int ppf i
  | Text t -> Format.fprintf ppf "%S" t
  | Items xs -> Format.fprintf ppf "[%a]" pp_cells xs
  | Fields xs -> Format.fprintf ppf "{%a}" pp_cells xs

and pp_cells ppf xs =
  Format.pp_print_seq ~pp_sep:comma pp_cell ppf (Array.to_seq xs)

let cell_w = Testable.make ~pp:pp_cell ~equal:( = )

(* [filler s] is a value of [s] that the rows under a null and the values before
   the first offset hold: no taken row shows it. *)
let rec filler = function
  | Ints -> Int 7
  | Texts -> Text "junk"
  | Lists e -> Items [| filler e |]
  | Records fs -> Fields (Array.of_list (List.map filler fs))

let bits_of cells =
  if Array.mem Null cells then Some (bits (Array.map (( <> ) Null) cells))
  else None

(* [offsets_of pre lengths] is the offsets from [pre] that cut rows of
   [lengths]. *)
let offsets_of pre lengths =
  let o = Array.make (Array.length lengths + 1) (Int64.of_int pre) in
  let cut i l = o.(i + 1) <- Int64.add o.(i) (Int64.of_int l) in
  Array.iteri cut lengths;
  Nx.create Nx.int64 [| Array.length o |] o

(* [build pre s cells] is the column of [s] whose rows are [cells], every list
   and text cutting its values after [pre] values that no row holds. *)
let rec build pre s cells =
  let validity = bits_of cells and n = Array.length cells in
  let layout l = Result.get_ok (Column.of_layout (type_of s) l) in
  match s with
  | Ints ->
      let value = function Int i -> Int64.of_int i | _ -> 0L in
      Column.of_tensor ?validity
        (Nx.create Nx.int64 [| n |] (Array.map value cells))
  | Texts ->
      let texts = Array.map (function Text t -> t | _ -> "junk") cells in
      let bytes =
        String.concat "" (String.make pre 'p' :: Array.to_list texts)
      in
      let m = String.length bytes in
      let child =
        Column.of_tensor
          (Nx.init Nx.uint8 [| m |] (fun i -> Char.code bytes.[i.(0)]))
      in
      let offsets = offsets_of pre (Array.map String.length texts) in
      layout (Varsize { validity; offsets; child })
  | Lists e ->
      let items = function Items xs -> xs | _ -> [| filler e |] in
      let rows = Array.map items cells in
      let before = Array.make pre (filler e) in
      let child = build pre e (Array.concat (before :: Array.to_list rows)) in
      let offsets = offsets_of pre (Array.map Array.length rows) in
      layout (Varsize { validity; offsets; child })
  | Records fs ->
      let field i f =
        let at = function Fields xs -> xs.(i) | _ -> filler f in
        (field_name i, build pre f (Array.map at cells))
      in
      layout (Children { validity; length = n; fields = List.mapi field fs })

let valid validity n =
  match validity with
  | None -> Array.make n true
  | Some v -> Nx.to_array (Nx.cast Nx.bool v)

(* [read s c] is the model of [c]'s rows, read through its layout. *)
let rec read s c =
  let n = Column.length c in
  let leaf f k =
    Array.map (function None -> Null | Some v -> f v) (Column.options k c)
  in
  match (s, Column.layout c) with
  | Ints, _ -> leaf (fun i -> Int i) Kind.int
  | Texts, _ -> leaf (fun t -> Text t) Kind.string
  | Lists e, Varsize { validity; offsets; child } ->
      let ok = valid validity n and o = Nx.to_array offsets in
      let elements = read e child in
      Array.init n (fun i ->
          if not ok.(i) then Null
          else
            let lo = Int64.to_int o.(i) and hi = Int64.to_int o.(i + 1) in
            Items (Array.sub elements lo (hi - lo)))
  | Records fs, Children { validity; fields; _ } ->
      let ok = valid validity n in
      let fields = List.map2 (fun f (_, c) -> read f c) fs fields in
      let fields = Array.of_list fields in
      Array.init n (fun i ->
          if ok.(i) then Fields (Array.map (fun f -> f.(i)) fields) else Null)
  | _ -> fail "a nested column's layout is not its type's"

let rec shape_gen depth =
  let leaf = Gen.of_list [ Ints; Texts ] in
  if depth = 0 then leaf
  else
    let inner = shape_gen (depth - 1) in
    let fields = Gen.list ~size:(Gen.int_range 1 2) inner in
    Gen.frequency
      [
        (1, leaf);
        (2, Gen.map (fun e -> Lists e) inner);
        (1, Gen.map (fun fs -> Records fs) fields);
      ]

(* A list at the top, or a record with a list field. *)
let nested_shape =
  let record (a, b) = Records [ a; Lists b ] in
  Gen.one_of
    [
      Gen.map (fun e -> Lists e) (shape_gen 2);
      Gen.map record (Gen.pair (shape_gen 1) (shape_gen 1));
    ]

let texts = [ ""; "a"; "\xc3\xa9t\xc3\xa9"; "xyz" ]

let rec cell_gen s = Gen.frequency [ (1, Gen.constant Null); (4, value_gen s) ]

and value_gen = function
  | Ints -> Gen.map (fun i -> Int i) (Gen.int_range (-3) 3)
  | Texts -> Gen.map (fun t -> Text t) (Gen.of_list texts)
  | Lists e ->
      let items = Gen.array ~size:(Gen.int_range 0 3) (cell_gen e) in
      Gen.map (fun xs -> Items xs) items
  | Records fs ->
      let cons f rest =
        Gen.map (fun (x, xs) -> x :: xs) (Gen.pair (cell_gen f) rest)
      in
      let fields = List.fold_right cons fs (Gen.constant []) in
      Gen.map (fun xs -> Fields (Array.of_list xs)) fields

(* Positions into [n] rows: drawn, so repeated and out of order, and outside the
   rows too; every row in order; or every row permuted. *)
let positions n =
  let all = List.init n Fun.id in
  Gen.frequency
    [
      (3, Gen.list ~size:(Gen.int_range 0 12) (Gen.int_range (-1) n));
      (1, Gen.constant all);
      (1, Gen.permutation all);
    ]

let pp_picks ppf (s, cells, pre, is) =
  Format.fprintf ppf "%a after %d: [%a] at %a" pp_shape s pre pp_cells cells
    (Format.pp_print_list ~pp_sep:Format.pp_print_space Format.pp_print_int)
    is

let nested_picks =
  let open Gen in
  with_pp pp_picks
    (let* s = nested_shape in
     let* cells = array ~size:(int_range 0 8) (cell_gen s) in
     let+ pre = int_range 0 2 and+ is = positions (Array.length cells) in
     (s, cells, pre, is))

let rec holds p s =
  p s
  ||
  match s with
  | Ints | Texts -> false
  | Lists e -> holds p e
  | Records fs -> List.exists (holds p) fs

let rec below p = function
  | Items xs | Fields xs -> Array.exists (fun x -> p x || below p x) xs
  | Null | Int _ | Text _ -> false

let rec ordered = function
  | a :: (b :: _ as rest) -> a <= b && ordered rest
  | _ -> true

let list_of p = function Lists e -> p e | _ -> false

let has_list = function
  | Records fs -> List.exists (list_of (Fun.const true)) fs
  | _ -> false

(* The shapes, rows and positions every nested law covers. *)
let nested_covers s cells is =
  let n = Array.length cells in
  let at i = if i >= 0 && i < n then cells.(i) else Null in
  let taken = List.map at is in
  let is_list = list_of (Fun.const true) in
  let is_record = function Records _ -> true | _ -> false in
  cover "a list of text" (holds (list_of (( = ) Texts)) s);
  cover "a list of lists" (holds (list_of is_list) s);
  cover "a list of records" (holds (list_of is_record) s);
  cover "a record with a list field" (holds has_list s);
  cover "a null below a taken row" (List.exists (below (( = ) Null)) taken);
  cover "an empty list taken" (List.exists (below (( = ) (Items [||]))) taken);
  cover "an empty text taken" (List.exists (below (( = ) (Text ""))) taken);
  cover "no row" (n = 0);
  cover "every row, in order" (n > 0 && is = List.init n Fun.id);
  cover "a repeated position"
    (List.length (List.sort_uniq compare is) < List.length is);
  cover "positions out of order" (not (ordered is));
  taken

(* A nested column's rows at positions are its rows there, at every depth. *)
let take_nested (s, cells, pre, is) =
  let is = inside (Array.length cells) is in
  let rows_at = nested_covers s cells is in
  cover "values before the first offset" (pre > 0);
  equal (array cell_w) (Array.of_list rows_at)
    (read s (taken is (build pre s cells)))

let join_nested (s, cells, pre, is) =
  let n = Array.length cells in
  let padded_rows = nested_covers s cells is in
  cover "a position outside the rows"
    (List.exists (fun i -> i < 0 || i >= n) is);
  equal (array cell_w)
    (Array.of_list padded_rows)
    (read s (padded is (build pre s cells)))

(* Sorting a table by a key of distinct values moves a nested column's rows with
   their keys: by the drawn positions where they permute the rows. *)
let sort_nested (s, cells, pre, is) =
  let n = Array.length cells in
  let all = List.init n Fun.id in
  let order = if List.sort compare is = all then is else all in
  let keys = Array.make n 0L in
  List.iteri (fun j i -> keys.(i) <- Int64.of_int j) order;
  cover "rows moved" (not (ordered order));
  let k = Column.of_tensor (Nx.create Nx.int64 [| n |] keys) in
  let t = v [ ("k", k); ("x", build pre s cells) ] in
  let sorted =
    Error.get_ok Query.(run (of_table t |> sort [ Order.asc "k" ]))
  in
  equal (array cell_w)
    (Array.of_list (List.map (fun i -> cells.(i)) order))
    (read s (column sorted "x"))

let () =
  exit
    (run "Column"
       [
         group "Column"
           (donation
           @ [
               prop "the values under a plan's nulls are the same on every run"
                 sample_positions deterministic;
             ]);
         group "null_count" null_count;
         group "v"
           [
             prop "values and options read back what v stores" G.sample
               without_nulls;
             prop "stores a float32 at the nearest float32, ties to even"
               ~examples:[ 0x1p-24 +. 0x1p-80; 1. +. 0x1p-24; 1. +. 0x1.8p-23 ]
               (Gen.frequency
                  [ (1, Gen.any_float); (3, Gen.float_range (-3.5e38) 3.5e38) ])
               (fun x ->
                 assume (Type.holds Type.float32 x);
                 let nearest = Int32.float_of_bits (Int32.bits_of_float x) in
                 equal (G.witness Type.float32) nearest
                   (Column.values Kind.float (Column.v Type.float32 [| x |])).(0));
             test
               "stores a float16 at the nearest float16, ties to even, \
                rounding once" (fun () ->
                 (* Binary16 has 11 significant bits: 2048 to 4096 step by 2,
                    and its subnormals by 2^-24. *)
                 let read x =
                   (Column.values Kind.float (Column.v Type.float16 [| x |])).(0)
                 in
                 equal float_exact 2048. (read 2049.);
                 equal float_exact 2052. (read 2051.);
                 equal float_exact 2050. (read (2049. +. 0x1p-14));
                 equal float_exact 0x1p-24 (read (0x1p-25 +. 0x1p-60)));
             test "holds a float32 below its limit and refuses the limit"
               (fun () ->
                 let below = Float.pred 0x1.ffffffp127 in
                 equal float_exact 0x1.fffffep127
                   (Column.values Kind.float
                      (Column.v Type.float32 [| below |])).(0);
                 raises_match Exn.invalid_arg (fun () ->
                     Column.v Type.float32 [| 0x1.ffffffp127 |]));
             test "keeps a float64 NaN's payload" (fun () ->
                 let nan = Int64.float_of_bits 0x7ff8_0000_dead_beefL in
                 let back =
                   Column.values Kind.float (Column.v Type.float64 [| nan |])
                 in
                 equal int64 0x7ff8_0000_dead_beefL
                   (Int64.bits_of_float back.(0)));
             test
               "holds a record field that holds a record with an extension \
                field" (fun () ->
                 let outer = Type.record [ ("r", Any inner_ty) ] in
                 let r v = Record.(add Record.kind "r" v empty) in
                 Law.round_trip column_w pass
                   (Column.values Record.kind)
                   (Column.v outer)
                   (Column.v outer [| r (Some m_null); r None |]));
             test "holds a null field of another kind as a null" (fun () ->
                 (* Type.holds reads only a record's non-null fields. *)
                 let r = Record.(add Kind.float "a" None empty) in
                 let ty = Type.record [ ("a", Any Type.int8) ] in
                 let back = Column.values Record.kind (Column.v ty [| r |]) in
                 equal (option int) None (Record.field Kind.int "a" back.(0)));
             test
               "raises naming the first row whose value the type does not hold"
               v_refusals;
           ];
         group "of_options"
           [
             prop ~count:150 ~examples:edges
               "options reads back what of_options stores" G.sample round_trip;
             prop ~examples:edges
               "a column has its rows' type, length, nulls and validity"
               G.sample observations;
           ];
         group "values and options"
           [
             test
               "raise for a kind that does not read the type, a null or a \
                value outside the kind"
               read_refusals;
           ];
         group "of_tensor"
           [
             cases "reads a 1-D tensor as its dtype's scalar type"
               ~id:(fun (name, _, _) -> name)
               scalar_types
               (fun (_, Nx.P x, ty) ->
                 equal any_w ty (Column.type_ (Column.of_tensor x)));
             test "of a 1-D bit tensor is a bool column over it" (fun () ->
                 let x = Nx.create Nx.bit [| 3 |] [| true; false; true |] in
                 let c = Column.of_tensor x in
                 equal any_w (Any Type.bool) (Column.type_ c);
                 satisfies ~claim:"the tensor's memory" (tensor_w ()) (shares x)
                   (Column.to_tensor Nx.bit c));
             prop "of a 1-D bool tensor is the bool column of its values"
               (Gen.pair (Gen.array Gen.bool) (Gen.int_range 0 9))
               (fun (bs, k) ->
                 let n = Array.length bs in
                 let k = Int.min k n in
                 cover "a view inside a byte" (k mod 8 <> 0 && k < n);
                 let x =
                   Nx.slice [ Nx.R (k, n) ] (Nx.create Nx.bool [| n |] bs)
                 in
                 let c = Column.of_tensor x in
                 equal any_w (Any Type.bool) (Column.type_ c);
                 equal (array bool)
                   (Array.sub bs k (n - k))
                   (Nx.to_array (Column.to_tensor Nx.bit c)));
             test "reads strided, broadcast and transposed views" (fun () ->
                 let x =
                   Nx.create Nx.int32 [| 6 |] [| 0l; 1l; 2l; 3l; 4l; 5l |]
                 in
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
             test "makes a null where the validity is clear" (fun () ->
                 let c =
                   Column.of_tensor ~validity:(three_bits ())
                     (Nx.create Nx.int8 [| 3 |] [| 1; 2; 3 |])
                 in
                 equal
                   (array (option int))
                   [| Some 1; None; Some 3 |]
                   (Column.options Kind.int c));
             cases "raises on a 1-D tensor of a dtype no scalar type stores"
               ~id:fst unstored (fun (_, Nx.P x) ->
                 raises_match Exn.invalid_arg (fun () -> Column.of_tensor x));
             test "raises on a scalar and a validity of another length"
               of_tensor_refusals;
           ];
         group "to_tensor"
           [
             test "is the storage of temporal, categorical and bool values"
               (fun () ->
                 let tensor dt ty vs =
                   Nx.to_array (Column.to_tensor dt (Column.v ty vs))
                 in
                 equal (array int32) [| -1l; 2l |]
                   (tensor Nx.int32 Type.date [| day (-1); day 2 |]);
                 equal (array int64) [| -1L; 3L |]
                   (tensor Nx.int64 (Type.wall Type.Ms)
                      [| ticks (-1_000_000L); ticks 3_000_000L |]);
                 equal (array int32) [| 1l; 0l |]
                   (tensor Nx.int32
                      (Type.categorical [| "x"; "y" |])
                      [| "y"; "x" |]);
                 equal (array bool) [| true; false |]
                   (tensor Nx.bit Type.bool [| true; false |]));
             test "of a tensor column has the cells as rows" (fun () ->
                 let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
                 let c = Column.of_tensor x in
                 equal any_w
                   (Any (Type.tensor Nx.float32 [| 2 |]))
                   (Column.type_ c);
                 let back = Column.to_tensor Nx.float32 c in
                 equal (array int) [| 2; 2 |] (Nx.shape back);
                 equal (array float_exact) [| 1.; 2.; 3.; 4. |]
                   (Nx.to_array back));
             test "of of_tensor reads the tensor's memory" (fun () ->
                 let x = Nx.create Nx.float64 [| 2 |] [| 1.; 2. |] in
                 satisfies ~claim:"the tensor's memory" (tensor_w ()) (shares x)
                   (Column.to_tensor Nx.float64 (Column.of_tensor x)));
             test "raises for another dtype, a null or text" to_tensor_refusals;
           ];
         group "ragged"
           [
             prop ~count:300 ~examples:ragged_edges
               "of of_ragged is the ragged array" raggeds ragged_law;
             prop "of text or bytes is each row's bytes" text_or_bytes row_bytes;
             test "reads an extension of text or a list as its storage"
               ext_storage;
             test "ignores a null element outside the rows" (fun () ->
                 let elements =
                   Column.of_options Type.int32 [| None; Some 1; None |]
                 in
                 let l =
                   Column.Varsize
                     {
                       validity = None;
                       offsets = n [| 1L; 2L |];
                       child = elements;
                     }
                 in
                 let c =
                   Result.get_ok (Column.of_layout (Any Type.(list int32)) l)
                 in
                 let r = Column.ragged Nx.int32 c in
                 equal (array int64) [| 1L; 2L |]
                   (Nx.to_array (Nx_ragged.offsets r)));
             test
               "raises on another type or dtype, a null row or a null element"
               ragged_refusals;
           ];
         group "of_ragged"
           [
             prop "packs bool values into bits" bool_raggeds packed;
             test "is a list of the dtype's type, sharing the buffers"
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
                 satisfies ~claim:"the values' memory" (tensor_w ())
                   (shares (Nx_ragged.values r))
                   (Nx_ragged.values back);
                 satisfies ~claim:"the offsets' memory" (tensor_w ())
                   (shares (Nx_ragged.offsets r))
                   (Nx_ragged.offsets back));
             test "makes a row null where the validity is clear" (fun () ->
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
             test "of values of two axes is a list of tensors" (fun () ->
                 let r =
                   Nx_ragged.v
                     ~offsets:(n [| 0L; 2L |])
                     (Nx.zeros Nx.float32 [| 2; 3 |])
                 in
                 equal any_w
                   (Any Type.(list (tensor Nx.float32 [| 3 |])))
                   (Column.type_ (Column.of_ragged r)));
             test
               "raises on a validity of another length or a dtype without type"
               of_ragged_refusals;
           ];
         group "of_layout"
           [
             prop ~count:150 "reads back a column's layout" G.sample
               layout_round_trip;
             prop ~count:200 ~examples:text_edges
               "refuses the first row of text that is not UTF-8" text_rows
               utf_8_law;
             prop "reads and writes an extension field's storage" G.sample
               ext_in_record;
             test "shares its values" (fun () ->
                 let x = Nx.create Nx.int32 [| 2 |] [| 1l; 2l |] in
                 let c =
                   Column.of_layout (Any Type.date)
                     (Fixed { validity = None; values = P x })
                 in
                 satisfies ~claim:"x's memory" (tensor_w ()) (shares x)
                   (Column.to_tensor Nx.int32 (require_ok c)));
             cases "is the first row the type does not hold, unless it is null"
               ~id:(fun (n, _, _, _) -> n)
               unheld
               (fun (_, ty, l, expected) ->
                 let got =
                   match Column.of_layout ty l with
                   | Ok _ -> None
                   | Error e -> Some e
                 in
                 equal (option (pair int string)) expected got);
             test "raises when the layout does not lay out the type"
               layout_refusals;
             test "raises naming Nx_ragged.v on text offsets it refuses"
               text_offsets_refusals;
           ];
         group "column"
           [
             test "drops a validity with no null" (fun () ->
                 let c = column (v [ ("x", column_of [| true; true |]) ]) "x" in
                 is_none ~pp:Nx.pp (Column.validity c));
           ];
         group "take"
           [
             prop ~count:300 ~examples:[ text_edge ]
               "of text rows is those rows" text_picks take_text;
             prop "of list rows is those rows, null elements included"
               list_picks take_lists;
             prop ~count:200
               "of nested rows is those rows, nulls at every depth included"
               nested_picks take_nested;
           ];
         group "Query.join"
           [
             prop ~count:300 ~examples:[ text_edge ]
               "a left join pads text rows with nulls" text_picks join_text;
             prop "a left join pads list rows with nulls" list_picks join_lists;
             prop ~count:200
               "a left join pads nested rows with nulls at the top only"
               nested_picks join_nested;
           ];
         group "Query.sort"
           [
             prop ~count:200 "moves a nested column's rows with their keys"
               nested_picks sort_nested;
           ];
       ])
