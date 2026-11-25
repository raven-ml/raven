(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module G = Talon_gen

let one name c = v [ (name, c) ]

(* [compute e t] is the column of [e] over [t]'s rows, or the error. *)
let compute e t =
  Result.map
    (fun r -> column r "out")
    (Query.run (Query.select Expr.[ "out" := e ] (Query.of_table t)))

let result e t = require_ok ~pp:Error.pp (compute e t)
let error e t = Format.asprintf "%a" Error.pp (require_error (compute e t))

let rows_are ty expected c =
  equal
    (array (option (G.witness ty)))
    expected
    (Column.options (Type.kind ty) c)

let some vs = Array.map Option.some vs
let pp_any ppf (Type.Any t) = Type.pp ppf t
let date y m d = Option.get (Time.Date.of_civil (y, m, d))

(* [problem e t] is the message of the problem that computing [e] over [t]
   raises, or ["no problem"]. *)
let problem e t =
  match compute e t with _ -> "no problem" | exception Invalid_argument m -> m

(* Ticks *)

let units = Type.[ S; Ms; Us; Ns ]

let unit_name : Type.unit_ -> string = function
  | S -> "s"
  | Ms -> "ms"
  | Us -> "us"
  | Ns -> "ns"

(* [per_s u] is the ticks of the unit [u] in a second. *)
let per_s : Type.unit_ -> int64 = function
  | S -> 1L
  | Ms -> 1_000L
  | Us -> 1_000_000L
  | Ns -> 1_000_000_000L

let ticks_per_day u = Int64.mul 86_400L (per_s u)
let int32_days = (Int64.of_int32 Int32.min_int, Int64.of_int32 Int32.max_int)
let int32_lo, int32_hi = int32_days

(* [floor_div x k] is [x / k] rounded toward negative infinity, for [k > 0]. *)
let floor_div x k =
  let q = Int64.div x k in
  if Int64.rem x k < 0L then Int64.pred q else q

let floor_mod x k = Int64.sub x (Int64.mul (floor_div x k) k)

(* [mul_exact k f] is [k * f] for [f > 0], or [None] if it overflows. *)
let mul_exact k f =
  let r = Int64.mul k f in
  if Int64.div r f = k then Some r else None

(* [add_exact x s] is [x + s], or [None] if it overflows. *)
let add_exact x s =
  let r = Int64.add x s in
  if Int64.compare s 0L >= 0 = (Int64.compare r x >= 0) then Some r else None

(* [stored c] is the storage of each row of the column [c] of fixed-width values
   as int64, [None] under a null. *)
let stored c =
  match Column.layout c with
  | Fixed { validity; values = Nx.P x } -> (
      let x = Nx.to_array (Nx.cast Nx.int64 x) in
      match validity with
      | None -> Array.map Option.some x
      | Some valid ->
          let valid = Nx.to_array (Nx.cast Nx.bool valid) in
          Array.mapi (fun i t -> if valid.(i) then Some t else None) x)
  | _ -> fail "a column of fixed-width values has a fixed-width layout"

(* [ticks_column ty xs] is the column of the temporal type [ty] that stores
   [xs], int32 days for [date], whatever their range. *)
let ticks_column (Type.Any ty as t) xs =
  let n = Array.length xs in
  let validity =
    Nx.cast Nx.bit (Nx.create Nx.bool [| n |] (Array.map Option.is_some xs))
  in
  let ticks = Array.map (Option.value ~default:0L) xs in
  let values =
    match ty with
    | Date -> Nx.P (Nx.create Nx.int32 [| n |] (Array.map Int64.to_int32 ticks))
    | _ -> Nx.P (Nx.create Nx.int64 [| n |] ticks)
  in
  match Column.of_layout t (Fixed { validity = Some validity; values }) with
  | Ok c -> c
  | Error (row, why) -> failf "row %d: %s" row why

(* [cast_ticks ty into xs] is [cast into] of the column of [ty] that stores
   [xs]. *)
let cast_ticks (Type.Any ty as t) (Type.Any into) xs =
  let x = Col.v (Type.kind ty) "x" in
  compute (Expr.cast into x) (one "x" (ticks_column t xs))

(* [agrees ~expected xs r] checks the run [r] over the rows [xs] whose model
   gives [expected], [None] where a row fails: the run fails at the first row
   that fails and is not null, with a message holding [failure], and otherwise
   gives each row's storage, null where [xs] is. *)
let agrees ?(failure = "cannot cast ") ~expected xs r =
  let fails i = Option.is_some xs.(i) && Option.is_none expected.(i) in
  match List.find_opt fails (List.init (Array.length xs) Fun.id) with
  | None ->
      let expected =
        Array.map2 (fun x e -> Option.bind x (fun _ -> e)) xs expected
      in
      equal (array (option int64)) expected (stored (require_ok ~pp:Error.pp r))
  | Some row ->
      contains
        ~sub:(Printf.sprintf ": row %d: %s" row failure)
        (Format.asprintf "%a" Error.pp (require_error r))

let fails xs expected =
  Array.exists2 (fun x e -> x <> None && e = None) xs expected

let before_1970 xs =
  Array.exists (function Some x -> x < 0L | None -> false) xs

(* Text *)

let s = Col.string "s"
let texts vs = one "s" (Column.of_options Type.string vs)

(* Text over few letters, so that partial matches abound. *)
let letters ~min ~max =
  Gen.map (String.concat "")
    (Gen.list ~size:(Gen.int_range min max) (Gen.of_list [ "a"; "b"; "é" ]))

(* [scalars s] is the scalar values of [s], each as its UTF-8 bytes. *)
let scalars s =
  let rec go i acc =
    if i >= String.length s then List.rev acc
    else
      let n = Uchar.utf_decode_length (String.get_utf_8_uchar s i) in
      go (i + n) (String.sub s i n :: acc)
  in
  go 0 []

(* Comparisons *)

let literal_law =
  let gen =
    Gen.pair
      (Gen.list ~size:(Gen.int_range 0 8) (letters ~min:0 ~max:6))
      (letters ~min:0 ~max:4)
  in
  let ops =
    [
      (Expr.( = ), Int.equal 0);
      (Expr.( <> ), fun c -> c <> 0);
      (Expr.( < ), fun c -> c < 0);
      (Expr.( <= ), fun c -> c <= 0);
      (Expr.( > ), fun c -> c > 0);
      (Expr.( >= ), fun c -> c >= 0);
    ]
  in
  prop "text compares with a literal as its bytes do"
    ~examples:[ ([ "ab"; "abb"; "a" ], "ab") ]
    gen
    (fun (vs, lit) ->
      cover "a row equal to the literal" (List.mem lit vs);
      cover "a row the literal is a prefix of"
        (List.exists (fun v -> v <> lit && String.starts_with ~prefix:lit v) vs);
      let vs = Array.of_list (List.map Option.some vs) in
      List.iter
        (fun (op, holds) ->
          let got =
            Column.options Kind.bool
              (result (op s (Expr.string lit)) (texts vs))
          in
          let expected =
            Array.map (Option.map (fun v -> holds (String.compare v lit))) vs
          in
          equal (array (option bool)) expected got)
        ops)

(* Cuts *)

(* Types that talon's comparisons order: numbers, temporal values, text and
   compound values. Floats are listed twice, so that NaN and the infinities are
   drawn in every run, and tensors, which alone take cut's path of compound
   words, three times. *)
let orderable =
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
      Any float32;
      Any float64;
      Any bool;
      Any date;
      Any (clock Ns);
      Any (duration Ms);
      Any (wall Us);
      Any (instant S);
      Any string;
      Any binary;
      Any (categorical [| "b"; "a"; "é"; "" |]);
      Any (list int8);
      Any (record [ ("a", Any int8); ("b", Any string) ]);
      Any (tensor Nx.float32 [| 2 |]);
      Any (tensor Nx.float32 [| 2 |]);
      Any (tensor Nx.int8 [| 1; 2 |]);
    ]

let is_nan : type a. a Type.t -> a -> bool =
 fun ty v ->
  match Kind.provably_equal (Type.kind ty) Kind.float with
  | Some Equal -> Float.is_nan v
  | None -> false

(* [specials ty] is NaN, the infinities and [-0.] for a float type. *)
let specials : type a. a Type.t -> a list =
 fun ty ->
  match Kind.provably_equal (Type.kind ty) Kind.float with
  | Some Equal -> [ Float.nan; Float.nan; infinity; neg_infinity; -0. ]
  | None -> []

let is_tensor : type a. a Type.t -> bool = function
  | Tensor _ -> true
  | _ -> false

(* [unrounded ty] draws floats that a narrow float type holds and rounds, as
   [0.7], which float32 stores as [0.699999988]. *)
let unrounded : type a. a Type.t -> a Gen.t option =
 fun ty ->
  match Kind.provably_equal (Type.kind ty) Kind.float with
  | Some Equal -> Some (Gen.float_range (-1000.) 1000.)
  | None -> None

(* [ascending ty vs] is the edges [vs] as given and as a column of [ty] stores
   them, without NaN, in talon's stored order and without stored repeats. *)
let ascending ty vs =
  let stored = Column.values (Type.kind ty) (Column.v ty vs) in
  let pairs = List.combine (Array.to_list vs) (Array.to_list stored) in
  let pairs = List.filter (fun (_, s) -> not (is_nan ty s)) pairs in
  let pairs =
    List.sort_uniq (fun (_, a) (_, b) -> Type.compare_value ty a b) pairs
  in
  (Array.of_list (List.map fst pairs), Array.of_list (List.map snd pairs))

type cut_case =
  | Cut_case : 'a Type.t * 'a array * 'a array * 'a option array -> cut_case

(* [pp_edges ty] formats edges as given: floats with every digit, before the
   type rounds them. *)
let pp_edges : type a. a Type.t -> Format.formatter -> a array -> unit =
 fun ty ppf vs ->
  match Kind.provably_equal (Type.kind ty) Kind.float with
  | Some Equal -> Array.iter (fun v -> Format.fprintf ppf "%.17g@ " v) vs
  | None -> G.pp_sample ppf (G.Sample (ty, Array.map Option.some vs))

let pp_cut_case ppf (Cut_case (ty, raw, es, xs)) =
  Format.fprintf ppf "@[<v>edges @[%a@]@,stored %a@,rows %a@]" (pp_edges ty) raw
    (pp_edges ty) es G.pp_sample
    (G.Sample (ty, xs))

(* Edges drawn at a type's bounds as well as anywhere, and rows on the edges,
   anywhere and null. *)
let cut_cases =
  let open Gen in
  let* (Type.Any ty) = of_list orderable in
  let value = Option.get (G.value ty) in
  let edge =
    match unrounded ty with
    | None -> value
    | Some raw -> frequency [ (1, value); (2, raw) ]
  in
  let* raw = array ~size:(int_range 2 6) edge in
  let raw, es = ascending ty raw in
  let n = Array.length es in
  let on_edge = map (fun i -> Some es.(i mod n)) nat in
  let rows =
    [ (3, map Option.some value); (1, constant None) ]
    @ (if n = 0 then [] else [ (3, on_edge) ])
    @
    match specials ty with
    | [] -> []
    | vs -> [ (3, map Option.some (of_list vs)) ]
  in
  let+ xs = array ~size:(int_range 0 12) (frequency rows) in
  Cut_case (ty, raw, es, xs)

(* [cut_case ty raw xs] is the case of the edges [raw] and the rows [xs]. *)
let cut_case ty raw xs =
  let raw, es = ascending ty raw in
  Cut_case (ty, raw, es, xs)

let cut_law =
  prop "cut is the bin that a linear scan of the edges finds"
    ~examples:
      [
        cut_case Type.float64 [| 0.; 1. |] [| Some Float.nan; Some 0.5 |];
        cut_case Type.float32 [| 0.7; 1. |] [| Some 0.7; Some 0.9 |];
        cut_case
          (Type.tensor Nx.float32 [| 2 |])
          [|
            Nx.create Nx.float32 [| 2 |] [| 0.; 0. |];
            Nx.create Nx.float32 [| 2 |] [| 1.; 1. |];
          |]
          [| Some (Nx.create Nx.float32 [| 2 |] [| 0.5; 0.5 |]) |];
      ]
    (Gen.with_pp pp_cut_case cut_cases)
    (fun (Cut_case (ty, raw, es, xs)) ->
      assume (Array.length es >= 2);
      let c = Column.of_options ty xs in
      let xs = Column.options (Type.kind ty) c in
      let cmp = Type.compare_value ty and last = Array.length es - 1 in
      let rec scan x i =
        if i >= last then None
        else if cmp es.(i) x <= 0 && cmp x es.(i + 1) < 0 then Some i
        else scan x (i + 1)
      in
      let bin x = scan x 0 in
      let on_edge x = Array.exists (fun e -> cmp e x = 0) es in
      let has p = Array.exists (function Some x -> p x | None -> false) xs in
      cover "a value on an edge" (has on_edge);
      cover "a value between two edges"
        (has (fun x -> Option.is_some (bin x) && not (on_edge x)));
      cover "a value below the first edge" (has (fun x -> cmp x es.(0) < 0));
      cover "a value on or above the last edge"
        (has (fun x -> cmp x es.(last) >= 0));
      cover "a null" (Array.exists Option.is_none xs);
      cover "NaN" (has (is_nan ty));
      cover "an edge that its type rounds"
        (Array.exists2 (fun r e -> Type.compare_value ty r e <> 0) raw es);
      cover "a tensor" (is_tensor ty);
      let x = Col.v (Type.kind ty) "x" in
      equal
        (array (option int))
        (Array.map (fun x -> Option.bind x bin) xs)
        (Column.options Kind.int (result (Expr.cut raw x) (one "x" c))))

let epoch ~ordered =
  Ext.v ~name:"epoch" ~ordered Type.float64 ~dec:Fun.id ~enc:Fun.id

let epochs xs =
  let ty = Type.Any (Type.ext ~name:"epoch" Type.float64) in
  let stored = Column.layout (Column.v Type.float64 xs) in
  Result.get_ok (Column.of_layout ty stored)

let cuts =
  let bins ty es xs =
    let x = Col.v (Type.kind ty) "x" in
    Column.options Kind.int
      (result (Expr.cut es x) (one "x" (Column.of_options ty xs)))
  in
  group "Expr.cut"
    [
      cut_law;
      test
        "bins hold their lower edge; NaN, values on the last edge and nulls \
         are in none" (fun () ->
          equal
            (array (option int))
            [|
              Some 0;
              Some 1;
              Some 1;
              Some 2;
              Some 2;
              None;
              None;
              None;
              None;
              None;
            |]
            (bins Type.float64
               [| -1.; 0.; 1.; infinity |]
               [|
                 Some (-1.);
                 Some (-0.);
                 Some 0.5;
                 Some 1.;
                 Some Float.max_float;
                 Some infinity;
                 Some neg_infinity;
                 Some Float.nan;
                 Some (-2.);
                 None;
               |]));
      test "the edge -0. is the edge 0." (fun () ->
          equal
            (array (option int))
            [| Some 0; Some 0; None |]
            (bins Type.float64 [| -0.; 1. |] (some [| 0.; -0.; -1e-300 |])));
      test "edges take the operand's type, as literals do" (fun () ->
          (* float32 stores 0.7 as 0.699999988, which [x >= float 0.7] holds
             for. *)
          equal
            (array (option int))
            [| Some 0; Some 0; None |]
            (bins Type.float32 [| 0.7; 1. |] (some [| 0.7; 0.9; 1. |])));
      test "integer edges at the ends of their types" (fun () ->
          equal
            (array (option int))
            [| Some 0; Some 0; Some 1; Some 1; None |]
            (bins Type.int64 [| min_int; 0; max_int |]
               (some [| min_int; -1; 0; max_int - 1; max_int |]));
          equal
            (array (option int))
            [| Some 0; Some 0; None |]
            (bins Type.uint8 [| 0; 255 |] (some [| 0; 254; 255 |])));
      test "the edges are copied" (fun () ->
          let es = [| 0; 10 |] in
          let e = Expr.cut es (Col.int "x") in
          es.(1) <- 1;
          equal
            (array (option int))
            [| Some 0 |]
            (Column.options Kind.int
               (result e (one "x" (Column.v Type.int64 [| 5 |])))));
      test "an extension declared ordered cuts its values" (fun () ->
          let e = Expr.cut [| 0.; 10. |] (Ext.col (epoch ~ordered:true) "e") in
          equal
            (array (option int))
            [| None; Some 0; Some 0; None |]
            (Column.options Kind.int
               (result e (one "e" (epochs [| -1.; 0.; 5.; 10. |])))));
      test "edges that do not ascend strictly, or a type that has no order"
        (fun () ->
          let t =
            v
              [
                ("x", Column.v Type.float64 [| 1. |]);
                ("f", Column.v Type.float32 [| 1. |]);
                ("i", Column.v Type.int8 [| 1 |]);
                ("n", Column.v Type.int64 [| 1 |]);
                ("c", Column.v (Type.clock S) [| Time.Span.s 1 |]);
                ("e", epochs [| 1. |]);
              ]
          in
          let x = Col.float "x" and n = Col.int "n" in
          expect
            (String.concat "\n"
               [
                 problem (Expr.cut [| 1.; 1. |] x) t;
                 problem (Expr.cut [| 2; 1 |] n) t;
                 problem (Expr.cut [| -0.; 0. |] x) t;
                 problem (Expr.cut [| 0.; Float.nan |] x) t;
                 problem (Expr.cut [| 0.7; 0.70000001 |] (Col.float "f")) t;
                 problem (Expr.cut [| 0; 1000 |] (Col.int "i")) t;
                 problem
                   (Expr.cut
                      [| Time.Span.s 0; Time.Span.days 1 |]
                      (Col.span "c"))
                   t;
                 problem
                   (Expr.cut [| 0.; 1. |] (Ext.col (epoch ~ordered:false) "e"))
                   t;
               ])
          @@ __POS_OF__
               {|
            select: 1 problem
              "out" := cut […] x
                cut takes edges that ascend strictly, and 1. then 1. do not, as float64 stores them.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            select: 1 problem
              "out" := cut […] n
                cut takes edges that ascend strictly, and 2 then 1 do not, as int64 stores them.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            select: 1 problem
              "out" := cut […] x
                cut takes edges that ascend strictly, and -0. then 0. do not, as float64 stores them.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            select: 1 problem
              "out" := cut […] x
                cut takes no NaN edge.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            select: 1 problem
              "out" := cut […] f
                cut takes edges that ascend strictly, and 0.699999988079071 then 0.699999988079071 do not, as float32 stores them.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            select: 1 problem
              "out" := cut […] i
                int8 does not hold the edge 1000.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            select: 1 problem
              "out" := cut […] c
                clock[s] does not hold the edge 24h.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            select: 1 problem
              "out" := cut […] e
                cut orders values, and the declaration of ext[epoch, float64] is not ~ordered:true.
              input (6 columns): x float64, f float32, i int8, n int64, c clock[s], e ext[epoch, float64]
            |});
    ]

(* Casts *)

(* A cast keeps the values of its operand's kind that its type holds, and fails
   at the first it does not. *)

type pair = Pair : 'a Type.t * 'a Type.t -> pair
type case = Case : 'a Type.t * 'a Type.t * 'a option array -> case

let pairs =
  let all ts =
    List.concat_map (fun a -> List.map (fun b -> Pair (a, b)) ts) ts
  in
  let units = Type.[ S; Ms; Us; Ns ] in
  Type.(
    all [ int8; int16; int32; int64; uint8; uint16; uint32; uint64 ]
    @ all (List.map duration units)
    @ all (List.map clock units)
    @ all (List.map (fun u -> wall u) units)
    @ all [ instant S; instant Ns ]
    @ all [ string; categorical [| "a"; "é"; "" |]; categorical [||] ])

let pp_case ppf (Case (a, b, vs)) =
  Format.fprintf ppf "cast %a of %a" Type.pp b G.pp_sample (G.Sample (a, vs))

let held_casts =
  let gen =
    Gen.bind (Gen.of_list pairs) (fun (Pair (a, b)) ->
        Gen.map (fun vs -> Case (a, b, vs)) (G.options a))
  in
  prop "a cast within a kind keeps the values its type holds"
    ~examples:
      [
        Case (Type.int64, Type.int8, some [| -128; 127 |]);
        Case (Type.string, Type.categorical [| "a"; "b" |], some [| "b"; "a" |]);
        Case (Type.categorical [| "a"; "b" |], Type.string, [| Some "b"; None |]);
        Case (Type.duration S, Type.duration Ms, some [| Time.Span.s 3 |]);
        Case (Type.wall Ms, Type.wall S, some [| Time.of_ns (-2_000_000_000L) |]);
        Case (Type.int8, Type.uint64, some [| -1 |]);
      ]
    (Gen.with_pp pp_case gen)
    (fun (Case (a, b, vs)) ->
      let x = Col.v (Type.kind a) "x" in
      let r = compute (Expr.cast b x) (one "x" (Column.of_options a vs)) in
      let unheld = function Some v -> not (Type.holds b v) | None -> false in
      let first = Array.find_index unheld vs in
      cover "every row held"
        (Option.is_none first && Array.exists Option.is_some vs);
      cover "a row not held" (Option.is_some first);
      match first with
      | None -> rows_are b vs (require_ok ~pp:Error.pp r)
      | Some row ->
          let e = Format.asprintf "%a" Error.pp (require_error r) in
          contains ~sub:(Printf.sprintf ": row %d: cannot cast " row) e)

(* Casts between numeric types *)

(* A value of a numeric column as its storage reads: an integer's int64 bits, a
   float, or a boolean. *)
type number = I of int64 | F of float | B of bool

let pp_number ppf = function
  | I x -> Format.fprintf ppf "%Ld" x
  | F x -> Format.fprintf ppf "%h" x
  | B b -> Format.pp_print_bool ppf b

let number_w =
  Testable.make ~pp:pp_number ~equal:(fun a b ->
      match (a, b) with
      | F x, F y -> Testable.equal float_exact x y
      | _ -> a = b)

(* [numbers ty c] is what the column [c] of the numeric type [ty] stores. *)
let numbers : type a. a Type.t -> Column.t -> number option array =
 fun ty c ->
  match ty with
  | Bool -> Array.map (Option.map (fun b -> B b)) (Column.options Kind.bool c)
  | Float16 | Float32 | Float64 ->
      Array.map (Option.map (fun x -> F x)) (Column.options Kind.float c)
  | _ -> Array.map (Option.map (fun x -> I x)) (stored c)

let rec bit_length m =
  if m = 0L then 0 else 1 + bit_length (Int64.shift_right_logical m 1)

(* [round_int p x] is the integer [x] rounded to [p] significant bits, to
   nearest, ties to even, computed exactly. *)
let round_int p x =
  (* The magnitude, read as unsigned: [Int64.min_int]'s is 2^63. *)
  let m = if x < 0L then Int64.neg x else x in
  let drop = max 0 (bit_length m - p) in
  let r =
    if drop = 0 then Int64.to_float m
    else
      let q = Int64.shift_right_logical m drop in
      let rest = Int64.logand m (Int64.pred (Int64.shift_left 1L drop)) in
      let half = Int64.shift_left 1L (drop - 1) in
      let up = rest > half || (rest = half && Int64.logand q 1L = 1L) in
      Float.ldexp (Int64.to_float (if up then Int64.succ q else q)) drop
  in
  if x < 0L then -.r else r

(* [to_half x] is the double [x] rounded to binary16, to nearest, ties to even:
   11 significant bits, a subnormal's unit 2^-24, and an infinity past 65504. *)
let to_half x =
  if (not (Float.is_finite x)) || x = 0. then x
  else
    let unit = max (snd (Float.frexp x) - 11) (-24) in
    let scaled = Float.ldexp x (-unit) in
    let r =
      if Float.abs (scaled -. Float.trunc scaled) = 0.5 then
        2. *. Float.round (scaled /. 2.)
      else Float.round scaled
    in
    let y = Float.copy_sign (Float.ldexp r unit) x in
    if Float.abs y > 65504. then Float.copy_sign Float.infinity x else y

(* OCaml rounds a double to binary32 once, to nearest, ties to even. *)
let to_single x = Int32.float_of_bits (Int32.bits_of_float x)
let two_63 = Float.ldexp 1. 63

(* [integer lo hi x] is the float [x] as an integer of the range [lo, hi), its
   int64 bits, or [None]. *)
let integer lo hi x =
  if Float.is_integer x && lo <= x && x < hi then
    Some
      (I
         (if x >= two_63 then Int64.of_float (x -. (2. *. two_63))
          else Int64.of_float x))
  else None

(* [convert into v] is the cast of the value [v] to the numeric type [into], as
   the interface states it, or [None] where the cast fails. *)
let convert : type a. a Type.t -> number -> number option =
 fun into v ->
  let to_float p round =
    let y =
      match v with
      | B b -> if b then 1. else 0.
      | I x -> round (round_int p x)
      | F x -> round x
    in
    match v with
    | F x when not (Float.is_finite x) -> Some (F y)
    | _ -> if Float.is_finite y then Some (F y) else None
  in
  let to_int lo hi =
    match v with
    | B b -> Some (I (if b then 1L else 0L))
    | F x -> integer lo hi x
    | I _ -> invalid_arg "convert: an integer cast to an integer"
  in
  match into with
  | Bool -> (
      match v with
      | B b -> Some (B b)
      | I 0L -> Some (B false)
      | I 1L -> Some (B true)
      | F x when x = 0. -> Some (B false)
      | F 1. -> Some (B true)
      | I _ | F _ -> None)
  | Float16 -> to_float 11 to_half
  | Float32 -> to_float 24 to_single
  | Float64 -> to_float 53 Fun.id
  | Int8 -> to_int (-128.) 128.
  | Int16 -> to_int (-32768.) 32768.
  | Int32 -> to_int (-2147483648.) 2147483648.
  | Int64 -> to_int (-.two_63) two_63
  | Uint8 -> to_int 0. 256.
  | Uint16 -> to_int 0. 65536.
  | Uint32 -> to_int 0. 4294967296.
  | Uint64 -> to_int 0. (2. *. two_63)
  | _ -> invalid_arg "convert: not a numeric type"

type numeric = Numeric : 'a Type.t * 'b Type.t * 'a option array -> numeric

let numeric_types =
  Type.
    [
      Any bool;
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
    ]

let is_int (Type.Any t) =
  Option.is_some (Kind.provably_equal (Type.kind t) Kind.int)

(* Pairs of numeric types, but for two integer types, which a cast within a kind
   covers, and rows each source type holds. *)
let numeric_case =
  let open Gen in
  let* (Type.Any a as source) = of_list numeric_types in
  let* (Type.Any b) =
    of_list
      (List.filter (fun t -> not (is_int source && is_int t)) numeric_types)
  in
  let+ vs = G.options a in
  Numeric (a, b, vs)

let pp_numeric ppf (Numeric (a, b, vs)) =
  Format.fprintf ppf "cast %a of %a" Type.pp b G.pp_sample (G.Sample (a, vs))

let numeric_law =
  prop ~count:300
    "a numeric cast keeps exact integers, takes 0 and 1 for bool, and rounds \
     to the nearest float, ties to even"
    ~examples:
      [
        Numeric
          ( Type.int64,
            Type.float32,
            some [| 16_777_217; (1 lsl 60) + (1 lsl 36) + 1 |] );
        Numeric
          ( Type.float64,
            Type.float16,
            some
              [| 1. +. Float.ldexp 1. (-11) +. Float.ldexp 1. (-40); 65519.99 |]
          );
        Numeric (Type.float64, Type.int32, some [| 2.; -0.; 1e9 |]);
        Numeric (Type.float64, Type.int32, some [| 1.; 3.5 |]);
        Numeric (Type.int8, Type.bool, some [| 0; 1 |]);
        Numeric (Type.int8, Type.bool, some [| 2 |]);
        Numeric (Type.float32, Type.int64, some [| Float.nan |]);
        Numeric (Type.float64, Type.float32, some [| Float.infinity; 1e300 |]);
        Numeric (Type.int32, Type.float16, some [| 2049; 70000 |]);
      ]
    (Gen.with_pp pp_numeric numeric_case)
    (fun (Numeric (a, b, vs)) ->
      let source = Column.of_options a vs in
      let values = numbers a source in
      let expected = Array.map (Option.map (convert b)) values in
      let refused =
        Array.find_index (function Some None -> true | _ -> false) expected
      in
      let rounded =
        Array.exists2
          (fun v e ->
            match (v, e) with
            | Some (I x), Some (Some (F y)) ->
                not (Float.abs y <= two_63 && Int64.equal (Int64.of_float y) x)
            | Some (F x), Some (Some (F y)) -> Float.is_finite x && x <> y
            | _ -> false)
          values expected
      in
      cover "a value refused" (Option.is_some refused);
      cover "every value cast"
        (Option.is_none refused && Array.exists Option.is_some values);
      cover "a float rounded" rounded;
      cover "a float to an integer" (is_int (Any b) && not (is_int (Any a)));
      let r =
        compute (Expr.cast b (Col.v (Type.kind a) "x")) (one "x" source)
      in
      match refused with
      | Some row ->
          contains
            ~sub:(Printf.sprintf ": row %d: cannot cast " row)
            (Format.asprintf "%a" Error.pp (require_error r))
      | None ->
          equal
            (array (option number_w))
            (Array.map (Option.map Option.get) expected)
            (numbers b (require_ok ~pp:Error.pp r)))

(* A duration's ticks *)

(* The int64s no float64 holds, past 2^53, and the bounds of int64. *)
let int64_edges =
  [|
    Int64.min_int;
    Int64.max_int;
    0x20_0000_0000_0000L;
    0x20_0000_0000_0001L;
    -0x20_0000_0000_0001L;
    0L;
    -1L;
  |]

let ticks_law =
  let gen =
    Gen.pair (Gen.of_list units)
      (Gen.list ~size:(Gen.int_range 0 24) (Gen.pair Gen.int64 Gen.bool))
  in
  prop "a duration cast to int64 and back keeps its ticks and nulls"
    ~examples:
      (List.map
         (fun u ->
           (u, List.map (fun x -> (x, true)) (Array.to_list int64_edges)))
         units
      @ [ (S, [ (1L, true); (Int64.max_int, false); (3L, true) ]) ])
    (Gen.with_pp
       (fun ppf (u, _) -> Format.pp_print_string ppf (unit_name u))
       gen)
    (fun (u, rows) ->
      let rows = Array.of_list rows in
      let xs = Array.map fst rows and valid = Array.map snd rows in
      let expected =
        Array.map2 (fun x v -> if v then Some x else None) xs valid
      in
      let t = one "x" (ticks_column (Any (Type.duration u)) expected) in
      let ticks = result (Expr.cast Type.int64 (Col.span "x")) t in
      let back =
        result (Expr.cast (Type.duration u) (Col.int "x")) (one "x" ticks)
      in
      equal (array (option int64)) expected (stored ticks);
      equal (array (option int64)) expected (stored back))

let casts =
  group "Expr.cast"
    [
      test "a literal fails at the first row, and over no rows not at all"
        (fun () ->
          let e = Expr.cast Type.int8 (Expr.int 300) in
          let t = one "x" (Column.v Type.int8 [| 1; 2 |]) in
          equal int 0
            (Column.length (result e (one "x" (Column.v Type.int8 [||]))));
          expect (error e t)
          @@ __POS_OF__
               {| select ["out" := cast int8 300]: row 0: cannot cast 300 to int8. |});
      test "an unsigned integer takes floats up to 2^64" (fun () ->
          let c =
            result
              (Expr.cast Type.uint64 (Col.float "x"))
              (one "x" (Column.v Type.float64 [| 1e19 |]))
          in
          (* 10^19 - 2^64: the bits of the uint64 10^19. *)
          equal
            (array (option int64))
            [| Some (-8_446_744_073_709_551_616L) |]
            (stored c));
      test "lists cast element by element" (fun () ->
          let xs = [| Some [| 1; 2 |]; None; Some [||] |] in
          rows_are (Type.list Type.int8) xs
            (result
               (Expr.cast (Type.list Type.int8)
                  (Col.v (Kind.list Kind.int) "x"))
               (one "x" (Column.of_options (Type.list Type.int64) xs))));
      test "records cast field by field and tensors element by element"
        (fun () ->
          let r a = Record.(add Kind.int "a" (Some a) empty) in
          let into = Type.record [ ("a", Any Type.int8) ] in
          let source = Type.record [ ("a", Any Type.int64) ] in
          let fields c =
            Array.map
              (Option.map (Record.field Kind.int "a"))
              (Column.options Record.kind c)
          in
          equal
            (array (option (option int)))
            [| Some (Some 1); Some (Some (-2)) |]
            (fields
               (result
                  (Expr.cast into (Col.v Record.kind "x"))
                  (one "x" (Column.v source [| r 1; r (-2) |]))));
          let tensor dt xs = Nx.create dt [| 2 |] xs in
          let out =
            result
              (Expr.cast
                 (Type.tensor Nx.float32 [| 2 |])
                 (Col.v (Kind.tensor Nx.int64) "x"))
              (one "x"
                 (Column.v
                    (Type.tensor Nx.int64 [| 2 |])
                    [| tensor Nx.int64 [| 1L; 3L |] |]))
          in
          equal
            (array (array float_exact))
            [| [| 1.; 3. |] |]
            (Array.map Nx.to_array (Column.values (Kind.tensor Nx.float32) out)));
      test "a cast's data error names its row, the value and the type"
        (fun () ->
          let failure ty vs into =
            let x = Col.v (Type.kind ty) "x" in
            error (Expr.cast into x) (one "x" (Column.of_options ty vs))
          in
          expect
            (String.concat "\n"
               [
                 failure Type.float64 (some [| 1.; 3.5 |]) Type.int32;
                 failure Type.int16 (some [| 300 |]) Type.int8;
                 failure Type.float32 (some [| Float.nan |]) Type.int64;
                 failure Type.string
                   (some [| "a"; "z" |])
                   (Type.categorical [| "a" |]);
                 failure (Type.duration Ms)
                   (some [| Time.Span.ms 1500 |])
                   (Type.duration S);
                 failure (Type.list Type.int64)
                   [| Some [| 1 |]; None; Some [| 2; 300 |] |]
                   (Type.list Type.int8);
               ])
          @@ __POS_OF__
               {|
            select ["out" := cast int32 x]: row 1: cannot cast 3.5 to int32.
            select ["out" := cast int8 x]: row 0: cannot cast 300 to int8.
            select ["out" := cast int64 x]: row 0: cannot cast nan to int64.
            select ["out" := cast categorical["a"] x]: row 1: cannot cast "z" to categorical["a"].
            select ["out" := cast duration[s] x]: row 0: cannot cast 1s500ms to duration[s].
            select ["out" := cast list[int8] x]: row 2: cannot cast 300 to int8.
            |});
      test "a cast it does not make is a problem naming the function that does"
        (fun () ->
          let t =
            v
              [
                ("s", Column.v Type.string [| "1" |]);
                ("d", Column.v Type.date [| Option.get (Time.Date.of_days 0) |]);
                ("w", Column.v (Type.wall S) [| Time.of_ns 0L |]);
                ("i", Column.v (Type.instant S) [| Time.of_ns 0L |]);
              ]
          in
          let names fn e =
            raises_match (Exn.invalid_arg ~sub:fn) (fun () -> compute (e ()) t)
          in
          names "Str.parse" (fun () -> Expr.cast Type.int64 (Col.string "s"));
          names "Temporal.format" (fun () ->
              Expr.cast Type.string (Col.date "d"));
          names "Temporal.localize" (fun () ->
              Expr.cast (Type.instant S) (Col.wall "w"));
          names "Temporal.wall" (fun () ->
              Expr.cast (Type.wall S) (Col.instant "i")));
      held_casts;
      numeric_law;
      ticks_law;
    ]

(* The days that a datetime[ns] reaches either side of 1970, about 292 years. *)
let ns_days = Int64.div Int64.max_int (ticks_per_day Ns)

let dates_to_datetimes =
  let open Gen in
  let days =
    frequency
      [
        (1, of_list [ int32_lo; Int64.succ int32_lo; Int64.pred int32_hi ]);
        (1, of_list [ int32_hi; Int64.neg (Int64.succ ns_days) ]);
        (1, of_list [ Int64.neg ns_days; -1L; 0L; 1L; ns_days ]);
        (1, of_list [ Int64.succ ns_days ]);
        (2, int64_range (-1_000_000L) 1_000_000L);
        (1, int64_range int32_lo int32_hi);
      ]
  in
  let gen = pair (of_list units) (array ~size:(int_range 0 12) (option days)) in
  prop
    "a date becomes the datetime of its midnight, failing out of the unit's \
     range"
    gen (fun (u, ds) ->
      let expected =
        Array.map
          (fun d -> Option.bind d (fun d -> mul_exact d (ticks_per_day u)))
          ds
      in
      cover "every row in range"
        ((not (fails ds expected)) && Array.exists Option.is_some ds);
      cover "a row out of range" (fails ds expected);
      cover "a date before 1970" (before_1970 ds);
      cover "a null" (Array.exists Option.is_none ds);
      agrees ~expected ds (cast_ticks (Any Type.date) (Any (Type.wall u)) ds))

(* [reach u] is the days either side of 1970 within which a datetime of the unit
   [u] holds every tick of the day, up to a million. *)
let reach u =
  Int64.min 1_000_000L (Int64.pred (Int64.div Int64.max_int (ticks_per_day u)))

let datetimes_to_dates =
  let gen =
    let open Gen in
    let* u = of_list units in
    let day = ticks_per_day u in
    let edges =
      [ Int64.min_int; Int64.succ Int64.min_int; -1L; 0L; 1L ]
      @ [ Int64.pred Int64.max_int; Int64.max_int; Int64.neg day ]
      @ [ Int64.pred (Int64.neg day); Int64.pred day; day ]
    in
    (* The first and last ticks of the date's first and last days, where a unit
       reaches them. *)
    let date_edges =
      List.concat_map
        (fun d ->
          match mul_exact d day with
          | Some t when Int64.compare t (Int64.sub Int64.max_int day) < 0 ->
              [ Int64.pred t; t; Int64.add t (Int64.pred day) ]
          | _ -> [])
        [ int32_lo; int32_hi; Int64.succ int32_hi ]
    in
    let near_midnight =
      let+ d = int64_range (Int64.neg (reach u)) (reach u)
      and+ k = int64_range (-1L) 1L in
      Int64.add (Int64.mul d day) k
    in
    let tick =
      frequency
        [ (2, of_list (edges @ date_edges)); (2, near_midnight); (2, int64) ]
    in
    let+ ts = array ~size:(int_range 0 12) (option tick) in
    (u, ts)
  in
  prop "a datetime becomes the day it falls on, failing out of the date's range"
    gen (fun (u, ts) ->
      let day_of t =
        let d = floor_div t (ticks_per_day u) in
        if d < int32_lo || d > int32_hi then None else Some d
      in
      let expected = Array.map (fun t -> Option.bind t day_of) ts in
      let inside_day t = Int64.rem t (ticks_per_day u) <> 0L in
      cover "a time of day before 1970"
        (Array.exists
           (function Some t -> t < 0L && inside_day t | None -> false)
           ts);
      cover "a day out of the date's range" (fails ts expected);
      cover "a null" (Array.exists Option.is_none ts);
      agrees ~expected ts (cast_ticks (Any (Type.wall u)) (Any Type.date) ts))

let dates_round_trip =
  let gen =
    let open Gen in
    let* u = of_list units in
    let+ ds =
      array ~size:(int_range 0 12)
        (option
           (frequency
              [
                (1, of_list [ Int64.neg (reach u); -1L; 0L; reach u ]);
                (3, int64_range (Int64.neg (reach u)) (reach u));
              ]))
    in
    (u, ds)
  in
  prop "a date cast to a datetime and back is the date" gen (fun (u, ds) ->
      cover "a date before 1970" (before_1970 ds);
      let w = array (option int64) in
      let datetime = Type.Any (Type.wall u) in
      Law.round_trip w w
        (fun ds ->
          stored
            (require_ok ~pp:Error.pp (cast_ticks (Any Type.date) datetime ds)))
        (fun ts ->
          stored
            (require_ok ~pp:Error.pp (cast_ticks datetime (Any Type.date) ts)))
        ds)

let days_floor =
  let gen =
    let open Gen in
    let* u = of_list units in
    let day = ticks_per_day u in
    let edges =
      [ Int64.min_int; Int64.succ Int64.min_int; -1L; 0L; 1L ]
      @ [ Int64.pred Int64.max_int; Int64.max_int ]
    in
    let anywhere =
      let+ d = int64_range (Int64.neg (reach u)) (reach u)
      and+ k = int64_range 0L (Int64.pred day) in
      Int64.add (Int64.mul d day) k
    in
    let tick = frequency [ (1, of_list edges); (3, anywhere); (1, int64) ] in
    let+ ts = array ~size:(int_range 0 12) (option tick) in
    (u, ts)
  in
  prop
    "Temporal.floor (Days 1) is the midnight of a datetime's day, and a cast \
     to a date and back is that midnight"
    gen (fun (u, ts) ->
      let day = ticks_per_day u in
      let midnight t = mul_exact (floor_div t day) day in
      let expected = Array.map (fun t -> Option.bind t midnight) ts in
      cover "a time of day before 1970" (before_1970 ts);
      cover "a midnight below the least tick" (fails ts expected);
      let ty = Type.wall u in
      let x = Col.wall "x" in
      let floor = Expr.Temporal.floor (Days 1) x in
      agrees ~failure:"the 1d period of " ~expected ts
        (compute floor (one "x" (ticks_column (Any ty) ts)));
      (* The cast holds where the midnight and the date both exist. *)
      let defined t =
        let d = floor_div t day in
        Option.is_some (midnight t) && d >= int32_lo && d <= int32_hi
      in
      let kept =
        Array.of_list
          (List.filter
             (function Some t -> defined t | None -> true)
             (Array.to_list ts))
      in
      let t = one "x" (ticks_column (Any ty) kept) in
      equal
        (array (option int64))
        (stored (result floor t))
        (stored (result Expr.(cast ty (cast Type.date x)) t)))

let clocks_and_durations =
  let gen =
    let open Gen in
    let* u0 = of_list units in
    let* u1 = frequency [ (1, constant u0); (1, of_list units) ] in
    let* to_clock = bool in
    let* in_day = frequency [ (1, constant true); (2, constant false) ] in
    let day = ticks_per_day u0 and second = per_s u0 in
    (* Whole seconds of the day, which every unit holds. *)
    let seconds = map (Int64.mul second) (int64_range 0L 86_399L) in
    let value =
      if in_day then seconds
      else if to_clock then
        frequency
          [
            (2, of_list [ Int64.min_int; -1L; 0L; 1L; Int64.pred day; day ]);
            (1, of_list [ Int64.max_int ]);
            (2, map (Int64.mul second) (int64_range (-86_400L) 172_800L));
            (2, int64_range (Int64.neg day) (Int64.mul 2L day));
            (1, int64);
          ]
      else
        frequency
          [
            (1, of_list [ 0L; 1L; Int64.pred day ]);
            (2, map (Int64.mul second) (int64_range 0L 86_399L));
            (2, int64_range 0L (Int64.pred day));
          ]
    in
    let+ xs = array ~size:(int_range 0 10) (option value) in
    (to_clock, u0, u1, xs)
  in
  let rescale u0 u1 x =
    let n0 = per_s u0 and n1 = per_s u1 in
    if n1 >= n0 then mul_exact x (Int64.div n1 n0)
    else
      let k = Int64.div n0 n1 in
      if Int64.rem x k = 0L then Some (Int64.div x k) else None
  in
  (* Four covers share the cases, the rarest drawn in about one case of sixteen,
     so 300 cases reach each in every run. *)
  prop ~count:300
    ~examples:[ (false, Type.Ms, Type.S, [| Some 1500L |]) ]
    "clocks and durations cast both ways, keeping the span and failing outside \
     the day"
    gen
    (fun (to_clock, u0, u1, xs) ->
      let in_day r = r >= 0L && r < ticks_per_day u1 in
      let model x =
        match rescale u0 u1 x with
        | Some r when (not to_clock) || in_day r -> Some r
        | _ -> None
      in
      let expected = Array.map (fun x -> Option.bind x model) xs in
      let outside x = not (in_day x) in
      cover "a duration outside the day, at the clock's unit"
        (to_clock && u0 = u1
        && Array.exists (function Some x -> outside x | None -> false) xs);
      cover "a clock to a duration of a coarser unit, failing"
        ((not to_clock) && per_s u1 < per_s u0 && fails xs expected);
      cover "a duration to a clock, every row kept"
        (to_clock && (not (fails xs expected)) && Array.exists Option.is_some xs);
      cover "a clock to a duration, every row kept"
        ((not to_clock)
        && (not (fails xs expected))
        && Array.exists Option.is_some xs);
      let from, into =
        if to_clock then (Type.duration u0, Type.clock u1)
        else (Type.clock u0, Type.duration u1)
      in
      agrees ~expected xs (cast_ticks (Any from) (Any into) xs))

let temporal_casts =
  let date y m d = Time.Date.to_days (date y m d) in
  let days_of u ts =
    stored
      (require_ok ~pp:Error.pp
         (cast_ticks (Any (Type.wall u)) (Any Type.date) (some ts)))
  in
  let days ds = some (Array.map Int64.of_int ds) in
  group "Expr.cast between temporal types"
    [
      dates_to_datetimes;
      datetimes_to_dates;
      dates_round_trip;
      days_floor;
      clocks_and_durations;
      test "a datetime's day, before 1970 and at the ends of its ticks"
        (fun () ->
          equal
            (array (option int64))
            (days [| date 1969 12 31; date 1970 1 1; date 1970 1 1 |])
            (days_of S [| -1L; 0L; 86_399L |]);
          equal
            (array (option int64))
            (days [| date 1677 9 21; date 2262 4 11 |])
            (days_of Ns [| Int64.min_int; Int64.max_int |]));
      test "dates at the ends of datetime[ns]" (fun () ->
          equal
            (array (option int64))
            [|
              Some
                (Int64.mul (Int64.of_int (date 2262 4 11)) 86_400_000_000_000L);
            |]
            (stored
               (require_ok ~pp:Error.pp
                  (cast_ticks (Any Type.date)
                     (Any (Type.wall Ns))
                     (days [| date 2262 4 11 |]))));
          expect
            (Format.asprintf "%a" Error.pp
               (require_error
                  (cast_ticks (Any Type.date)
                     (Any (Type.wall Ns))
                     (days [| date 1677 9 22; date 1677 9 21 |]))))
          @@ __POS_OF__
               {| select ["out" := cast datetime[ns] x]: row 1: cannot cast 1677-09-21 to datetime[ns]. |});
      test
        "an instant falls on a day only in a zone, so it does not cast to dates"
        (fun () ->
          let t =
            v
              [
                ("z", Column.v (Type.instant S) [| Time.of_ns 0L |]);
                ("d", Column.v Type.date [| Option.get (Time.Date.of_days 0) |]);
              ]
          in
          expect
            (String.concat "\n"
               [
                 problem (Expr.cast Type.date (Col.instant "z")) t;
                 problem (Expr.cast (Type.instant S) (Col.date "d")) t;
               ])
          @@ __POS_OF__
               {|
            select: 1 problem
              "out" := cast date z
                cast does not convert datetime[s, UTC] to date: use Temporal.wall.
              input (2 columns): z datetime[s, UTC], d date
            select: 1 problem
              "out" := cast datetime[s, UTC] d
                cast does not convert date to datetime[s, UTC].
              input (2 columns): z datetime[s, UTC], d date
            |});
    ]

(* Text *)

(* [index s sub i] is the first byte from [i] at which [sub] lies whole in
   [s]. *)
let rec index s sub i =
  let n = String.length sub in
  if i + n > String.length s then None
  else if String.sub s i n = sub then Some i
  else index s sub (i + 1)

let rec pieces_in s i = function
  | [] -> true
  | p :: ps -> (
      match index s p i with
      | Some j -> pieces_in s (j + String.length p) ps
      | None -> false)

let pieces_law =
  let gen =
    Gen.pair
      (Gen.list ~size:(Gen.int_range 0 8) (letters ~min:0 ~max:40))
      (Gen.list ~size:(Gen.int_range 1 3) (letters ~min:1 ~max:3))
  in
  prop "matches finds pieces in order, without overlap" gen (fun (vs, ps) ->
      let hit v = pieces_in v 0 ps in
      cover "a match" (List.exists hit vs);
      cover "a miss" (List.exists (fun v -> not (hit v)) vs);
      let vs = Array.of_list (List.map Option.some vs) in
      let got =
        Column.options Kind.bool
          (result (Expr.Str.matches (Expr.Str.pieces ps) s) (texts vs))
      in
      equal (array (option bool)) (Array.map (Option.map hit) vs) got)

let slices =
  let text = Option.get (G.value Type.string) in
  let gen =
    Gen.triple
      (Gen.array ~size:(Gen.int_range 0 6) (Gen.option text))
      (Gen.int_range (-8) 8) (Gen.int_range 0 8)
  in
  prop "slice and length count the scalar values of String.get_utf_8_uchar"
    ~examples:
      [
        ([| Some "héllo"; Some "ab"; Some ""; None |], 1, 2);
        ([| Some "héllo"; Some "ab" |], -3, 2);
      ]
    gen
    (fun (vs, offset, length) ->
      let multi s = List.exists (fun u -> String.length u > 1) (scalars s) in
      cover "a scalar of several bytes"
        (Array.exists (function Some s -> multi s | None -> false) vs);
      cover "an offset from the end" (offset < 0);
      let slice s =
        let us = scalars s in
        let n = List.length us in
        let p = if offset >= 0 then offset else n + offset in
        String.concat "" (List.filteri (fun i _ -> i >= p && i < p + length) us)
      in
      let t = texts vs in
      equal
        (array (option string))
        (Array.map (Option.map slice) vs)
        (Column.options Kind.string
           (result (Expr.Str.slice ~offset ~length s) t));
      equal
        (array (option int))
        (Array.map (Option.map (fun s -> List.length (scalars s))) vs)
        (Column.options Kind.int (result (Expr.Str.length s) t)))

(* Text that the forms of talon.csv read or refuse, for a type of each form. *)
type parsed = Parsed : 'a Type.t -> parsed

let parse_types =
  Type.
    [
      Parsed bool;
      Parsed int8;
      Parsed int64;
      Parsed float32;
      Parsed float64;
      Parsed date;
      Parsed (wall Us);
      Parsed (instant S);
      Parsed (categorical [| "a"; "x" |]);
    ]

let parse_text =
  Gen.of_list
    ~pp:(fun ppf -> Format.fprintf ppf "%S")
    [
      "true";
      "false";
      "0";
      "1";
      "-1";
      "+1";
      "300";
      "1.5";
      "1e400";
      "-0";
      "nan";
      "inf";
      "2024-03-15";
      "2024-03-15T10:00:00";
      "2024-03-15T10:00:00Z";
      "2024-03-15T10:00:00+02:00";
      "x";
      "a";
      "";
      " 1";
    ]

let parse_law =
  prop "parse reads the text that Column.parse reads"
    (Gen.pair
       (Gen.of_list ~pp:(fun ppf (Parsed ty) -> Type.pp ppf ty) parse_types)
       (Gen.array ~size:(Gen.int_range 0 6) (Gen.option parse_text)))
    (fun (Parsed ty, rows) ->
      let column = Column.parse (Any ty) (Column.of_options Type.string rows) in
      let read c = Column.options (Type.kind ty) c in
      cover "read" (Result.is_ok column);
      cover "refused" (Result.is_error column);
      let by_expr = compute (Expr.Str.parse ty s) (texts rows) in
      match column with
      | Ok c ->
          equal
            (array (option (G.witness ty)))
            (read c)
            (read (require_ok ~pp:Error.pp by_expr))
      | Error (row, _) ->
          contains
            ~sub:(Printf.sprintf ": row %d: " row)
            (Format.asprintf "%a" Error.pp (require_error by_expr)))

let text =
  let bools e vs = Column.options Kind.bool (result e (texts vs)) in
  group "Expr.Str"
    [
      test
        "matches a literal anywhere, a prefix at the start, a suffix at the end"
        (fun () ->
          let vs =
            [| Some "special requests"; Some "requests special"; None |]
          in
          let m p = bools (Expr.Str.matches p s) vs in
          equal
            (array (option bool))
            [| Some true; Some false; None |]
            (m (Expr.Str.pieces [ "special"; "requests" ]));
          equal
            (array (option bool))
            [| Some true; Some false; None |]
            (m (Expr.Str.prefix "spe"));
          equal
            (array (option bool))
            [| Some false; Some true; None |]
            (m (Expr.Str.suffix "cial"));
          equal
            (array (option bool))
            [| Some true; Some true; None |]
            (m (Expr.Str.literal "q")));
      pieces_law;
      slices;
      parse_law;
      test "parse's data error names the text" (fun () ->
          expect
            (error
               (Expr.Str.parse Type.int32 s)
               (texts [| Some "12"; Some "x1" |]))
          @@ __POS_OF__
               {| select ["out" := Str.parse int32 s]: row 1: "x1": not an integer. |});
    ]

(* Trimming *)

(* Unicode's White_Space property, PropList.txt 18.0.0: its 25 scalar values. *)
let white_space =
  [ 0x0009; 0x000A; 0x000B; 0x000C; 0x000D; 0x0020; 0x0085; 0x00A0; 0x1680 ]
  @ List.init 11 (fun i -> 0x2000 + i)
  @ [ 0x2028; 0x2029; 0x202F; 0x205F; 0x3000 ]

(* The scalar values beside each run of White_Space, and the separators and
   zero-width characters outside it. *)
let not_white_space =
  [ 0x0008; 0x000E; 0x001C; 0x001D; 0x001E; 0x001F; 0x0021; 0x0084; 0x0086 ]
  @ [ 0x009F; 0x00A1; 0x167F; 0x1681; 0x180E; 0x1FFF; 0x200B; 0x200C ]
  @ [ 0x200D; 0x2027; 0x202A; 0x202E; 0x2030; 0x205E; 0x2060; 0x2FFF ]
  @ [ 0x3001; 0xFEFF ]

let utf_8 u =
  let b = Buffer.create 4 in
  Buffer.add_utf_8_uchar b (Uchar.of_int u);
  Buffer.contents b

let scalar s =
  Uchar.to_int (Uchar.utf_decode_uchar (String.get_utf_8_uchar s 0))

(* [trimmed s] is [s] without the White_Space scalars at its ends. *)
let trimmed s =
  let rec drop = function
    | u :: us when List.mem (scalar u) white_space -> drop us
    | us -> us
  in
  String.concat "" (List.rev (drop (List.rev (drop (scalars s)))))

let trims vs = Column.options Kind.string (result (Expr.Str.trim s) (texts vs))

(* A scalar value of four bytes of UTF-8, the length no white space has. *)
let four_bytes = utf_8 0x1F600

let trim_law =
  let of_list vs = Gen.of_list ~pp:(fun ppf -> Format.fprintf ppf "%S") vs in
  let token =
    Gen.frequency
      [
        (2, of_list (List.map utf_8 white_space));
        (2, of_list (List.map utf_8 not_white_space));
        (1, of_list [ "a"; "é"; "日" ]);
        (2, of_list [ four_bytes ]);
      ]
  in
  let text =
    Gen.map (String.concat "") (Gen.list ~size:(Gen.int_range 0 8) token)
  in
  prop "trim drops the White_Space scalars at both ends of the text"
    ~examples:
      [
        [| Some (" " ^ four_bytes); Some (four_bytes ^ "\t") |];
        [| Some " \t"; Some "\u{3000}a b\u{3000}" |];
      ]
    (Gen.array ~size:(Gen.int_range 0 6) (Gen.option text))
    (fun vs ->
      let has p = Array.exists (function Some s -> p s | None -> false) vs in
      let ws s = String.length s > 0 && List.mem (scalar s) white_space in
      let ends s =
        let us = scalars s in
        us <> [] && ws (List.hd us) && ws (List.hd (List.rev us))
      in
      cover "white space at both ends and a scalar outside it"
        (has (fun s -> ends s && trimmed s <> ""));
      cover "white space alone" (has (fun s -> s <> "" && trimmed s = ""));
      let starts s = String.starts_with ~prefix:four_bytes s in
      let ends s = String.ends_with ~suffix:four_bytes s in
      cover "a text that trims to a four-byte scalar at its start"
        (has (fun s -> starts (trimmed s) && not (starts s)));
      cover "a text that trims to a four-byte scalar at its end"
        (has (fun s -> ends (trimmed s) && not (ends s)));
      equal
        (array (option string))
        (Array.map (Option.map trimmed) vs)
        (trims vs))

let trimming =
  group "Expr.Str.trim"
    [
      trim_law;
      cases "trims each White_Space scalar at both ends, and keeps it inside"
        ~id:(Printf.sprintf "U+%04X") white_space (fun u ->
          let w = utf_8 u in
          equal
            (array (option string))
            [| Some ("a" ^ w ^ "b"); Some "" |]
            (trims [| Some (w ^ w ^ "a" ^ w ^ "b" ^ w); Some w |]));
      cases "keeps the scalars outside White_Space"
        ~id:(Printf.sprintf "U+%04X") not_white_space (fun u ->
          let n = utf_8 u in
          let kept = n ^ " a " ^ n in
          equal
            (array (option string))
            [| Some kept |]
            (trims [| Some (" " ^ kept ^ "\t") |]));
      test "nulls stay null, and the empty text is kept" (fun () ->
          equal
            (array (option string))
            [| None; Some ""; Some "a" |]
            (trims [| None; Some ""; Some " a\n" |]));
      test "a categorical trims as its text" (fun () ->
          let ty = Type.categorical [| " a"; "a "; "b" |] in
          let t =
            one "s" (Column.of_options ty [| Some " a"; Some "a "; None |])
          in
          equal
            (array (option string))
            [| Some "a"; Some "a"; None |]
            (Column.options Kind.string (result (Expr.Str.trim s) t)));
    ]

(* Joining *)

type part = Text_column of string option array | Literal of string

let pp_part ppf = function
  | Text_column vs -> G.pp_sample ppf (G.Sample (Type.string, vs))
  | Literal s -> Format.fprintf ppf "string %S" s

let concat_law =
  let text = Option.get (G.value Type.string) in
  let gen =
    let open Gen in
    let* n = int_range 0 5 in
    let operand =
      frequency
        [
          ( 3,
            map
              (fun vs -> Text_column vs)
              (array ~size:(constant n) (option text)) );
          (1, map (fun s -> Literal s) text);
        ]
    in
    let+ sep = of_list [ ""; "_"; ", "; "é"; "日本" ]
    and+ ops = list ~size:(int_range 0 4) operand in
    (n, sep, ops)
  in
  let pp ppf (n, sep, ops) =
    Format.fprintf ppf "@[<v>%d rows, separator %S@,%a@]" n sep
      (Format.pp_print_list pp_part)
      ops
  in
  prop "concat joins the texts with the separator, null where one is null"
    ~examples:[ (2, "_", []) ]
    (Gen.with_pp pp gen)
    (fun (n, sep, ops) ->
      let columns =
        List.filter_map
          (function Text_column vs -> Some vs | Literal _ -> None)
          ops
      in
      let name i = Printf.sprintf "c%d" i in
      let t =
        v
          (("rows", Column.v Type.int8 (Array.make n 0))
          :: List.mapi
               (fun i vs -> (name i, Column.of_options Type.string vs))
               columns)
      in
      let k = ref (-1) in
      let es =
        List.map
          (function
            | Text_column _ ->
                incr k;
                Col.string (name !k)
            | Literal s -> Expr.string s)
          ops
      in
      let row i =
        List.map (function Text_column vs -> vs.(i) | Literal s -> Some s) ops
      in
      let expected =
        Array.init n (fun i ->
            let r = row i in
            if List.exists Option.is_none r then None
            else Some (String.concat sep (List.map Option.get r)))
      in
      cover "a row with a null operand" (Array.exists Option.is_none expected);
      cover "a row of two operands or more"
        (List.length ops >= 2 && Array.exists Option.is_some expected);
      cover "a literal and a column"
        (columns <> [] && List.length columns < List.length ops);
      cover "no operand" (ops = [] && n > 0);
      equal
        (array (option string))
        expected
        (Column.options Kind.string (result (Expr.Str.concat sep es) t)))

let joining =
  group "Expr.Str.concat"
    [
      concat_law;
      test "a categorical joins as its text" (fun () ->
          let k = Type.categorical [| "x"; "y" |] in
          let t =
            v
              [
                ("k", Column.of_options k [| Some "y"; None; Some "x" |]);
                ("s", Column.v Type.string [| "1"; "2"; "3" |]);
              ]
          in
          equal
            (array (option string))
            [| Some "y-1"; None; Some "x-3" |]
            (Column.options Kind.string
               (result Expr.(Str.concat "-" [ Col.string "k"; s ]) t)));
      test "null joins to null, and no text to the empty text" (fun () ->
          let t = texts [| Some "a"; None |] in
          equal
            (array (option string))
            [| None; None |]
            (Column.options Kind.string
               (result Expr.(Str.concat "_" [ s; null ]) t));
          equal
            (array (option string))
            [| Some ""; Some "" |]
            (Column.options Kind.string (result (Expr.Str.concat "_" []) t)));
    ]

(* Temporal.add *)

(* Temporal.add over every operand type and every pairing of its unit with a
   span's unit that it takes, ticks drawn at their bounds as well as anywhere.
   The model computes each row in exact int64 arithmetic: a row whose span does
   not scale to the operand's ticks, whose sum overflows, or whose result leaves
   the operand's range fails the run, at the first such row. *)

type operand = Datetime | Duration | Clock | Date

let adds =
  let open Gen in
  let units = Type.[ S; Ms; Us; Ns ] in
  let* op = of_list [ Datetime; Duration; Clock; Date ] in
  let* u = of_list units in
  (* A span finer than the operand's unit is a problem of the verb, and a date's
     unit is the second. *)
  let* du =
    let unit_ = if op = Date then Type.S else u in
    of_list (List.filter (fun d -> per_s d <= per_s unit_) units)
  in
  let edges =
    [ Int64.min_int; Int64.succ Int64.min_int; -1L; 0L; 1L ]
    @ [ Int64.pred Int64.max_int; Int64.max_int ]
  in
  let operand =
    match op with
    | Datetime | Duration -> frequency [ (1, of_list edges); (1, int64) ]
    | Clock ->
        let last = Int64.pred (Int64.mul 86_400L (per_s u)) in
        frequency [ (1, of_list [ 0L; 1L; last ]); (1, int64_range 0L last) ]
    | Date ->
        let lo, hi = int32_days in
        frequency
          [
            (1, of_list [ lo; Int64.succ lo; 0L; Int64.pred hi; hi ]);
            (1, int64_range lo hi);
          ]
  in
  let span =
    frequency
      [
        (2, of_list edges);
        (2, int64);
        ( 1,
          map
            (Int64.mul (Int64.mul 86_400L (per_s du)))
            (int64_range (-1000L) 1000L) );
        (1, int64_range (-1000L) 1000L);
      ]
  in
  let+ rows = array ~size:(int_range 0 12) (pair operand span) in
  (op, u, du, rows)

let model op u du x k =
  let ( let* ) = Option.bind in
  match op with
  | Date ->
      let day = Int64.mul 86_400L (per_s du) in
      let r = Int64.add x (Int64.div k day) in
      let lo, hi = int32_days in
      if Int64.rem k day <> 0L || r < lo || r > hi then None else Some r
  | Datetime | Duration | Clock ->
      let* s = mul_exact k (Int64.div (per_s u) (per_s du)) in
      let* r = add_exact x s in
      let day = Int64.mul 86_400L (per_s u) in
      if op = Clock && (r < 0L || r >= day) then None else Some r

let adds_ticks =
  prop
    "add moves a temporal value by the span's ticks, failing at the first row \
     out of range"
    ~examples:
      [
        (* 2024-02-28 and two days, then 36 hours, which is not whole days. *)
        (Date, Type.S, Type.S, [| (19781L, 172_800L) |]);
        (Date, Type.S, Type.S, [| (19781L, 172_800L); (19781L, 129_600L) |]);
        (* 09:00 and 90 minutes, in milliseconds. *)
        (Clock, Type.Ms, Type.Ms, [| (32_400_000L, 5_400_000L) |]);
      ]
    adds
    (fun (op, u, du, rows) ->
      let column ty dt xs =
        let values = Nx.P (Nx.create dt [| Array.length xs |] xs) in
        match Column.of_layout ty (Fixed { validity = None; values }) with
        | Ok c -> c
        | Error (row, why) -> failf "row %d: %s" row why
      in
      let x = Array.map fst rows and k = Array.map snd rows in
      let a =
        match op with
        | Datetime -> column (Any (Type.instant u)) Nx.int64 x
        | Duration -> column (Any (Type.duration u)) Nx.int64 x
        | Clock -> column (Any (Type.clock u)) Nx.int64 x
        | Date -> column (Any Type.date) Nx.int32 (Array.map Int64.to_int32 x)
      in
      let t =
        v [ ("a", a); ("k", column (Any (Type.duration du)) Nx.int64 k) ]
      in
      let d = Col.span "k" in
      let out =
        match op with
        | Datetime -> compute Expr.(Temporal.add (Col.instant "a") d) t
        | Duration | Clock -> compute Expr.(Temporal.add (Col.span "a") d) t
        | Date -> compute Expr.(Temporal.add (Col.date "a") d) t
      in
      let ticks c =
        match op with
        | Date ->
            Array.map Int64.of_int32 (Nx.to_array (Column.to_tensor Nx.int32 c))
        | _ -> Nx.to_array (Column.to_tensor Nx.int64 c)
      in
      let expected = Array.map2 (model op u du) x k in
      cover "the span's unit is the operand's" (op <> Date && u = du);
      cover "a coarser span" (op <> Date && per_s du < per_s u);
      cover "every row in range"
        (Array.for_all Option.is_some expected && rows <> [||]);
      cover "a row out of range" (Array.exists Option.is_none expected);
      cover "a date" (op = Date && Array.exists Option.is_some expected);
      cover "a clock" (op = Clock && Array.exists Option.is_some expected);
      match Array.find_index Option.is_none expected with
      | None ->
          equal (array int64)
            (Array.map Option.get expected)
            (ticks (require_ok ~pp:Error.pp out))
      | Some row ->
          contains
            ~sub:(Printf.sprintf ": row %d: " row)
            (Format.asprintf "%a" Error.pp (require_error out)))

let literal_span =
  test "a literal span adds to a time of day" (fun () ->
      let t = one "c" (Column.v (Type.clock Ms) [| Time.Span.hours 9 |]) in
      equal
        (array (option int64))
        [| Some (Time.Span.to_ns (Time.Span.minutes 630)) |]
        (Array.map
           (Option.map Time.Span.to_ns)
           (Column.options Kind.span
              (result
                 Expr.(
                   Temporal.add (Col.span "c") (span (Time.Span.minutes 90)))
                 t))))

(* Temporal.diff *)

type diffed =
  | Diffed :
      'a Type.t * 'a Type.t * (int64 option * int64 option) array
      -> diffed

(* Operands of one type, or clocks of two units, and their ticks: at the bounds
   of int64 and anywhere, within the day for clocks, within int32 for dates. *)
let diff_case =
  let open Gen in
  let edges =
    [ Int64.min_int; Int64.succ Int64.min_int; -1L; 0L; 1L; Int64.max_int ]
  in
  let any =
    frequency
      [ (1, of_list edges); (2, int64); (1, int64_range (-1000L) 1000L) ]
  in
  let rows a b = array ~size:(int_range 0 8) (pair (option a) (option b)) in
  let in_day u = int64_range 0L (Int64.pred (ticks_per_day u)) in
  let* u = of_list units in
  let* u1 = of_list units in
  let* kind = int_range 0 3 in
  match kind with
  | 0 ->
      map (fun r -> Diffed (Type.instant u, Type.instant u, r)) (rows any any)
  | 1 ->
      map (fun r -> Diffed (Type.duration u, Type.duration u, r)) (rows any any)
  | 2 ->
      map
        (fun r -> Diffed (Type.clock u, Type.clock u1, r))
        (rows (in_day u) (in_day u1))
  | _ ->
      let days =
        frequency
          [
            (1, of_list [ int32_lo; int32_hi; 0L ]);
            (2, int64_range int32_lo int32_hi);
          ]
      in
      map (fun r -> Diffed (Type.date, Type.date, r)) (rows days days)

let pp_diffed ppf (Diffed (a, b, rows)) =
  let tick ppf = function
    | None -> Format.pp_print_string ppf "null"
    | Some x -> Format.fprintf ppf "%Ld" x
  in
  Format.fprintf ppf "%a - %a: %a" Type.pp a Type.pp b
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       (fun ppf (x, y) -> Format.fprintf ppf "%a - %a" tick x tick y))
    (Array.to_list rows)

(* [unit_of ty] is the unit of the temporal type [ty]'s ticks, a date's in
   seconds. *)
let unit_of : type a. a Type.t -> Type.unit_ = function
  | Type.Datetime { unit_; _ } -> unit_
  | Type.Duration u | Type.Clock u -> u
  | _ -> Type.S

let diff_law =
  prop
    "diff is the span from b to a in their common unit, failing at the first \
     row out of range"
    ~examples:
      [
        Diffed
          ( Type.instant Ns,
            Type.instant Ns,
            [| (Some Int64.max_int, Some (-1L)) |] );
        Diffed (Type.clock S, Type.clock Ns, [| (Some 1L, Some 5L) |]);
      ]
    (Gen.with_pp pp_diffed diff_case)
    (fun (Diffed (ta, tb, rows)) ->
      let date = match ta with Type.Date -> true | _ -> false in
      let ua = unit_of ta and ub = unit_of tb in
      let fine = if per_s ua >= per_s ub then ua else ub in
      let scale u x = Int64.mul x (Int64.div (per_s fine) (per_s u)) in
      let model x y =
        if date then Some (Int64.mul (Int64.sub x y) 86_400L)
        else add_exact (scale ua x) (Int64.neg (scale ub y))
      in
      (* [-y] overflows at [min_int], where the span is out of range unless [x]
         is negative. *)
      let model x y =
        if (not date) && y = Int64.min_int && ua = ub then
          if x < 0L then Some (Int64.sub x y) else None
        else model x y
      in
      let pairs =
        Array.map
          (fun (x, y) ->
            match (x, y) with Some x, Some y -> Some (x, y) | _ -> None)
          rows
      in
      let expected = Array.map (Option.map (fun (x, y) -> model x y)) pairs in
      let flat = Array.map (fun e -> Option.join e) expected in
      let nulls = Array.map (fun e -> Option.map (fun _ -> ()) e) expected in
      cover "a row out of range"
        (Array.exists (fun e -> e = Some None) expected);
      cover "clocks of two units" (ua <> ub);
      cover "dates" date;
      let t =
        v
          [
            ("a", ticks_column (Any ta) (Array.map fst rows));
            ("b", ticks_column (Any tb) (Array.map snd rows));
          ]
      in
      let a = Col.v (Type.kind ta) "a" and b = Col.v (Type.kind tb) "b" in
      agrees ~failure:"" ~expected:flat nulls
        (compute (Expr.Temporal.diff a b) t))

let diff_problem =
  test "diff of operands that do not meet is a problem" (fun () ->
      let t =
        v
          [
            ("a", Column.v (Type.duration S) [| Time.Span.s 1 |]);
            ("b", Column.v (Type.duration Ns) [| Time.Span.s 1 |]);
          ]
      in
      raises_match
        (fun e -> Exn.invalid_arg e)
        (fun () -> compute (Expr.Temporal.diff (Col.span "a") (Col.span "b")) t))

(* Temporal.field *)

let fields =
  [
    `Year;
    `Month;
    `Day;
    `Hour;
    `Minute;
    `Second;
    `Nanosecond;
    `Weekday;
    `Yearday;
  ]

let field_name = function
  | `Year -> "Year"
  | `Month -> "Month"
  | `Day -> "Day"
  | `Hour -> "Hour"
  | `Minute -> "Minute"
  | `Second -> "Second"
  | `Nanosecond -> "Nanosecond"
  | `Weekday -> "Weekday"
  | `Yearday -> "Yearday"

let is_leap y = (y mod 4 = 0 && y mod 100 <> 0) || y mod 400 = 0

let month_length y m =
  if m = 2 then if is_leap y then 29 else 28
  else if List.mem m [ 4; 6; 9; 11 ] then 30
  else 31

(* [civil_fields days ns] is each field of {!fields} of the time [ns]
   nanoseconds into the day [days] after 1970-01-01. 1970-01-01 is a Thursday,
   ISO weekday 4. *)
let civil_fields days ns =
  let y, m, d = Time.Date.to_civil (Option.get (Time.Date.of_days days)) in
  let before = List.init (m - 1) (fun k -> month_length y (k + 1)) in
  let s = ns / 1_000_000_000 in
  [
    y;
    m;
    d;
    s / 3600;
    s / 60 mod 60;
    s mod 60;
    ns mod 1_000_000_000;
    (((days mod 7) + 7 + 3) mod 7) + 1;
    List.fold_left ( + ) d before;
  ]

let days_case =
  Gen.frequency
    [
      (1, Gen.of_list [ int32_lo; int32_hi; 0L; -1L ]);
      (2, Gen.int64_range int32_lo int32_hi);
      (2, Gen.int64_range (-1_000_000L) 1_000_000L);
    ]

let date_fields =
  prop "fields of a date agree with Time.Date.to_civil and the ISO weekday"
    ~examples:
      [
        [|
          0L;
          Int64.of_int (Time.Date.to_days (date 2024 12 31));
          Int64.of_int (Time.Date.to_days (date 2024 3 15));
        |];
      ]
    (Gen.array ~size:(Gen.int_range 0 12) days_case)
    (fun ds ->
      let t = one "d" (ticks_column (Any Type.date) (some ds)) in
      List.iteri
        (fun i f ->
          let expected =
            Array.map
              (fun d -> Some (List.nth (civil_fields (Int64.to_int d) 0) i))
              ds
          in
          if i < 3 || i > 6 then
            equal ~msg:(field_name f)
              (array (option int))
              expected
              (Column.options Kind.int
                 (result (Expr.Temporal.field f (Col.date "d")) t)))
        fields)

(* Ticks of a stamp in its unit, from nanoseconds at the bounds of int64 and
   anywhere, so that each is a stamp. *)
let stamp_ticks u =
  let open Gen in
  let ns =
    frequency
      [
        (1, of_list [ Int64.min_int; Int64.max_int; -1L; 0L; 1L ]);
        (2, int64);
        (1, int64_range (-1_000_000_000_000L) 1_000_000_000_000L);
      ]
  in
  map (fun n -> floor_div n (Int64.div 1_000_000_000L (per_s u))) ns

let clock_fields =
  prop
    "fields of a wall clock and of a clock agree with its day and its time of \
     day"
    ~examples:[ (Type.Ns, [| -1L; Int64.min_int; Int64.max_int |]) ]
    Gen.(
      let* u = of_list units in
      let+ ts = array ~size:(int_range 0 12) (stamp_ticks u) in
      (u, ts))
    (fun (u, ts) ->
      cover "before 1970" (Array.exists (fun t -> t < 0L) ts);
      let ns_per_tick = Int64.div 1_000_000_000L (per_s u) in
      let split t =
        let day = floor_div t (ticks_per_day u) in
        let tod = Int64.sub t (Int64.mul day (ticks_per_day u)) in
        (Int64.to_int day, Int64.to_int (Int64.mul tod ns_per_tick))
      in
      let walls = one "w" (ticks_column (Any (Type.wall u)) (some ts)) in
      let times_of_day =
        Array.map
          (fun t ->
            Int64.sub t
              (Int64.mul (floor_div t (ticks_per_day u)) (ticks_per_day u)))
          ts
      in
      let clocks =
        one "c" (ticks_column (Any (Type.clock u)) (some times_of_day))
      in
      List.iteri
        (fun i f ->
          let expected =
            Array.map
              (fun t ->
                let d, ns = split t in
                Some (List.nth (civil_fields d ns) i))
              ts
          in
          let read e t =
            Column.options Kind.int (result (Expr.Temporal.field f e) t)
          in
          equal
            ~msg:("wall " ^ field_name f)
            (array (option int))
            expected
            (read (Col.wall "w") walls);
          if i >= 3 && i <= 6 then
            equal
              ~msg:("clock " ^ field_name f)
              (array (option int))
              expected
              (read (Col.span "c") clocks))
        fields)

let field_tests =
  [
    test "fields of datetimes at the ends of their ticks" (fun () ->
        let t =
          one "t"
            (Column.v (Type.wall Ns)
               [| Time.of_ns Int64.min_int; Time.of_ns Int64.max_int |])
        in
        let f field =
          Column.options Kind.int
            (result (Expr.Temporal.field field (Col.wall "t")) t)
        in
        equal (array (option int)) (some [| 1677; 2262 |]) (f `Year);
        equal (array (option int)) (some [| 9; 4 |]) (f `Month);
        equal (array (option int)) (some [| 21; 11 |]) (f `Day);
        equal (array (option int)) (some [| 0; 23 |]) (f `Hour));
    test "a date's time field and a clock's calendar field are problems"
      (fun () ->
        let t =
          v
            [
              ("d", Column.v Type.date [| date 2024 3 15 |]);
              ("c", Column.v (Type.clock S) [| Time.Span.s 1 |]);
            ]
        in
        let refused e =
          raises_match (fun e -> Exn.invalid_arg e) (fun () -> compute (e ()) t)
        in
        refused (fun () -> Expr.Temporal.field `Hour (Col.date "d"));
        refused (fun () -> Expr.Temporal.field `Year (Col.span "c")));
  ]

(* Temporal.floor and Temporal.offset *)

let fdiv a b = if a >= 0 then a / b else -((-a + b - 1) / b)
let fmod a b = a - (b * fdiv a b)
let civil d = Time.Date.to_civil (Option.get (Time.Date.of_days d))

let day_of (y, m, d) =
  Option.map Time.Date.to_days (Time.Date.of_civil (y, m, d))

let in_dates d =
  if d >= Int64.to_int int32_lo && d <= Int64.to_int int32_hi then Some d
  else None

(* [floor_day step d] is the first day of the period of the calendar [step] that
   holds the day [d]: periods count from 1970-01-01, weeks from Monday
   1970-01-05 (day 4), and months from January 1970. *)
let floor_day step d =
  match step with
  | Time.Days n -> in_dates (d - fmod d n)
  | Weeks n -> in_dates (d - fmod (d - 4) (7 * n))
  | Months n ->
      let y, m, _ = civil d in
      let k = ((y - 1970) * 12) + m - 1 in
      let k = k - fmod k n in
      day_of (1970 + fdiv k 12, fmod k 12 + 1, 1)
  | Exact _ -> invalid_arg "floor_day"

(* [offset_day step d] is the day [d] moved by the calendar [step], a day past
   the month's end becoming its last. *)
let offset_day step d =
  match step with
  | Time.Days n -> in_dates (d + n)
  | Weeks n -> in_dates (d + (7 * n))
  | Months n ->
      let y, m, dd = civil d in
      let k = (y * 12) + m - 1 + n in
      let y = fdiv k 12 and m = fmod k 12 + 1 in
      day_of (y, m, min dd (month_length y m))
  | Exact _ -> invalid_arg "offset_day"

(* An operand of a temporal type, a step it takes, and its ticks. *)
type moved = Moved : 'a Type.t * Time.step * int64 option array -> moved

let pp_moved ppf (Moved (ty, step, xs)) =
  Format.fprintf ppf "%a by %a: [%s]" Type.pp ty Time.pp_step step
    (String.concat "; "
       (Array.to_list
          (Array.map
             (function None -> "null" | Some x -> Int64.to_string x)
             xs)))

(* Dates, wall clocks and instants of every unit, calendar steps of [-n] to [n]
   units, or [1] to [n], and exact steps of whole ticks. *)
let moved_case ~negative =
  let open Gen in
  let n = if negative then int_range (-14) 14 else int_range 1 14 in
  let calendar =
    one_of
      [
        map (fun n -> Time.Days n) n;
        map (fun n -> Time.Weeks n) n;
        map (fun n -> Time.Months n) n;
      ]
  in
  let exact u =
    let tick = Int64.div 1_000_000_000L (per_s u) in
    let+ k = frequency [ (2, n); (1, of_list [ 60; 3600; 86_400 ]) ] in
    let k = if k = 0 then 1 else k in
    Time.Exact (Time.Span.of_ns (Int64.mul (Int64.of_int k) tick))
  in
  let rows g = array ~size:(int_range 0 8) (option g) in
  let* form = int_range 0 2 in
  let* u = of_list units in
  match form with
  | 0 ->
      let+ step = calendar and+ xs = rows days_case in
      Moved (Type.date, step, xs)
  | 1 ->
      let* step = frequency [ (3, calendar); (1, exact u) ] in
      map (fun xs -> Moved (Type.wall u, step, xs)) (rows (stamp_ticks u))
  | _ ->
      let* step = exact u in
      map (fun xs -> Moved (Type.instant u, step, xs)) (rows (stamp_ticks u))

(* [day_ticks ty] is the ticks of [ty] in a day, and [step_ticks ty d] in the
   span [d]. *)
let day_ticks : type a. a Type.t -> int64 = function
  | Type.Date -> 1L
  | ty -> ticks_per_day (unit_of ty)

let step_ticks ty d =
  Int64.div (Time.Span.to_ns d) (Int64.div 1_000_000_000L (per_s (unit_of ty)))

(* [floored ty step t] is the first value of the period of [step] that holds the
   ticks [t] of [ty], or [None] out of range. *)
let floored ty step t =
  match step with
  | Time.Exact d ->
      let k = step_ticks ty d in
      let r = Int64.sub t (floor_mod t k) in
      if r > t then None else Some r
  | step ->
      let per = day_ticks ty in
      Option.bind
        (floor_day step (Int64.to_int (floor_div t per)))
        (fun d -> mul_exact (Int64.of_int d) per)

(* [offset ty step t] is the ticks [t] of [ty] moved by [step], or [None] out of
   range. *)
let offset ty step t =
  match step with
  | Time.Exact d -> add_exact t (step_ticks ty d)
  | step ->
      (* The time of day stays: [t] moves by whole days, even where its day's
         midnight is below the least tick. *)
      let per = day_ticks ty in
      let day = Int64.to_int (floor_div t per) in
      Option.bind (offset_day step day) (fun d ->
          Option.bind (mul_exact (Int64.of_int (d - day)) per) (add_exact t))

(* [moved_covers ty step xs expected] labels a case of floor or offset. *)
let moved_covers ty step xs expected =
  let exact = match step with Time.Exact _ -> true | _ -> false in
  cover "a date" (match ty with Type.Any Type.Date -> true | _ -> false);
  cover "a wall clock by a calendar step"
    ((match ty with
       | Type.Any (Type.Datetime { clock = Time.Local; _ }) -> true
       | _ -> false)
    && not exact);
  cover "an exact step" exact;
  cover "a row out of range" (fails xs expected);
  cover "before 1970" (before_1970 xs)

let floor_law =
  prop "floor by a step is the first value of the period that holds it"
    ~examples:
      [
        Moved (Type.date, Months 3, some [| 19753L; 19797L |]);
        Moved (Type.date, Weeks 1, some [| 19753L; 19797L |]);
        Moved (Type.wall Ns, Days 1, some [| Int64.min_int |]);
      ]
    (Gen.with_pp pp_moved (moved_case ~negative:false))
    (fun (Moved (ty, step, xs)) ->
      let expected = Array.map (fun x -> Option.bind x (floored ty step)) xs in
      moved_covers (Any ty) step xs expected;
      let x = Col.v (Type.kind ty) "x" in
      agrees ~failure:"" ~expected xs
        (compute
           (Expr.Temporal.floor step x)
           (one "x" (ticks_column (Any ty) xs))))

let offset_law =
  prop "offset moves by a step, a day past the month's end becoming its last"
    ~examples:
      [
        Moved (Type.date, Months 1, some [| 19753L; 19797L |]);
        Moved (Type.date, Days 1, some [| int32_hi |]);
        Moved (Type.wall Ns, Days 0, some [| Int64.min_int |]);
      ]
    (Gen.with_pp pp_moved (moved_case ~negative:true))
    (fun (Moved (ty, step, xs)) ->
      let expected = Array.map (fun x -> Option.bind x (offset ty step)) xs in
      moved_covers (Any ty) step xs expected;
      let x = Col.v (Type.kind ty) "x" in
      agrees ~failure:"" ~expected xs
        (compute
           (Expr.Temporal.offset step x)
           (one "x" (ticks_column (Any ty) xs))))

let floor_tests =
  [
    test "a datetime's day whose midnight is below the least tick has no floor"
      (fun () ->
        let t =
          one "t" (Column.v (Type.wall Ns) [| Time.of_ns Int64.min_int |])
        in
        expect (error (Expr.Temporal.floor (Days 1) (Col.wall "t")) t)
        @@ __POS_OF__
             {| select ["out" := Temporal.floor 1d t]: row 0: the 1d period of 1677-09-21T00:12:43.145224192 is out of range. |});
    test "an exact step that is not whole ticks is a problem" (fun () ->
        let at u ns = one "t" (Column.v (Type.wall u) [| Time.of_ns ns |])
        and t = Col.wall "t"
        and ms n = Time.Exact (Time.Span.ms n) in
        expect
          (String.concat "\n"
             Expr.Temporal.
               [
                 problem (floor (ms 1500) t) (at S 0L);
                 problem (offset (ms 500) t) (at S 0L);
                 problem (floor (ms 1500) t) (at Ms 0L);
               ])
        @@ __POS_OF__
             {|
            select: 1 problem
              "out" := Temporal.floor 1s500ms t
                Temporal.floor moves datetime[s] by multiples of 1s, not 1s500ms.
              input (1 column): t datetime[s]
            select: 1 problem
              "out" := Temporal.offset 500ms t
                Temporal.offset moves datetime[s] by multiples of 1s, not 500ms.
              input (1 column): t datetime[s]
            no problem
            |});
    test "an exact step needs no calendar and floors an instant" (fun () ->
        let t =
          one "t"
            (Column.v (Type.instant Us)
               [| Option.get (Time.of_us 90_000_000L) |])
        in
        let e =
          Expr.Temporal.floor (Exact (Time.Span.minutes 1)) (Col.instant "t")
        in
        equal
          (array (option int64))
          [| Some 60_000_000L |]
          (stored (result e t)));
  ]

(* Temporal.parse and Temporal.format *)

(* Dates, clocks, wall clocks and instants of every unit, at the bounds of their
   ticks and anywhere. *)
let formatted =
  let open Gen in
  let* u = of_list units in
  let* form = of_list [ `Date; `Clock; `Wall; `Instant ] in
  let ty, fmt, tick =
    match form with
    | `Date -> (Type.Any Type.date, "%Y-%m-%d", days_case)
    | `Clock ->
        let last = Int64.pred (ticks_per_day u) in
        ( Type.Any (Type.clock u),
          "%H:%M:%S.%f",
          frequency [ (1, of_list [ 0L; last ]); (2, int64_range 0L last) ] )
    | `Wall -> (Any (Type.wall u), "%Y-%m-%d %H:%M:%S.%f", stamp_ticks u)
    | `Instant -> (Any (Type.instant u), "%Y-%m-%d %H:%M:%S.%f%z", stamp_ticks u)
  in
  let+ xs = array ~size:(int_range 0 12) tick in
  (form, ty, fmt, xs)

let formats_round_trip =
  prop "parse reads back what format writes, at every tick"
    (Gen.with_pp
       (fun ppf (_, ty, fmt, xs) ->
         Format.fprintf ppf "%a in %S: [%s]" pp_any ty fmt
           (String.concat "; " (Array.to_list (Array.map Int64.to_string xs))))
       formatted)
    (fun (form, (Type.Any t as ty), fmt, xs) ->
      cover "a date" (form = `Date);
      cover "a clock" (form = `Clock);
      cover "a wall clock" (form = `Wall);
      cover "an instant" (form = `Instant);
      let text = Expr.Temporal.format fmt (Col.v (Type.kind t) "t") in
      let back =
        result
          (Expr.Temporal.parse fmt t text)
          (one "t" (ticks_column ty (some xs)))
      in
      equal (array (option int64)) (some xs) (stored back))

let parse_tests =
  [
    test "an offset reads as UTC, and an instant writes in UTC" (fun () ->
        let fmt = "%Y-%m-%dT%H:%M:%S%z" and ty = Type.instant S in
        let t = texts [| Some "2024-03-15T10:00:00+02:00" |] in
        let r = Expr.Temporal.(format fmt (parse fmt ty s)) in
        equal
          (array (option string))
          [| Some "2024-03-15T08:00:00Z" |]
          (Column.options Kind.string (result r t)));
    test "format writes the fields its directives name" (fun () ->
        let t =
          one "d"
            (Column.of_options Type.date [| Some (date 2024 3 15); None |])
        in
        equal
          (array (option string))
          [| Some "15/03/2024"; None |]
          (Column.options Kind.string
             (result (Expr.Temporal.format "%d/%m/%Y" (Col.date "d")) t)));
    test "a field the format leaves out is that of 1970-01-01 00:00:00"
      (fun () ->
        let parsed fmt ty text =
          stored (result (Expr.Temporal.parse fmt ty s) (texts [| Some text |]))
        in
        equal
          (array (option int64))
          [| Some 37_800L |]
          (parsed "%H:%M" (Type.wall S) "10:30");
        equal
          (array (option int64))
          [| Some (Int64.of_int (Time.Date.to_days (date 2024 1 1))) |]
          (parsed "%Y" Type.date "2024"));
    test "%f reads one to nine digits" (fun () ->
        let parsed text =
          stored
            (result
               (Expr.Temporal.parse "%S.%f" (Type.wall Ns) s)
               (texts [| Some text |]))
        in
        equal (array (option int64)) [| Some 5_500_000_000L |] (parsed "05.5");
        equal
          (array (option int64))
          [| Some 5_123_456_789L |] (parsed "05.123456789"));
    test "an instant's format needs %z, and a wall clock's takes none"
      (fun () ->
        let t =
          v
            [
              ("s", Column.v Type.string [| "x" |]);
              ("w", Column.v (Type.wall S) [| Time.of_ns 0L |]);
            ]
        in
        let refused e =
          raises_match (fun e -> Exn.invalid_arg e) (fun () -> compute (e ()) t)
        in
        refused (fun () -> Expr.Temporal.parse "%H:%M" (Type.instant S) s);
        refused (fun () -> Expr.Temporal.parse "%H:%M%z" (Type.wall S) s);
        refused (fun () -> Expr.Temporal.format "%H:%M%z" (Col.wall "w")));
    test "a data error names the row and the value" (fun () ->
        let d = date 2024 2 28 in
        expect
          (String.concat "\n"
             [
               error
                 (Expr.Temporal.parse "%d/%m/%Y" Type.date s)
                 (texts [| Some "15/03/2024"; Some "2024-03-15" |]);
               error
                 Expr.(Temporal.add (Col.date "d") (Col.span "x"))
                 (v
                    [
                      ("d", Column.v Type.date [| d |]);
                      ("x", Column.v (Type.duration S) [| Time.Span.hours 36 |]);
                    ]);
             ])
        @@ __POS_OF__
             {|
          select ["out" := Temporal.parse "%d/%m/%Y" date s]: row 1: "2024-03-15": not in the format "%d/%m/%Y".
          select ["out" := Temporal.add d x]: row 0: 36h is not whole days.
          |});
  ]

let () =
  exit
    (run "talon convert"
       [
         group "Expr comparisons" [ literal_law ];
         cuts;
         casts;
         temporal_casts;
         text;
         trimming;
         joining;
         group "Expr.Temporal.add" [ adds_ticks; literal_span ];
         group "Expr.Temporal.diff" [ diff_law; diff_problem ];
         group "Expr.Temporal.field" (date_fields :: clock_fields :: field_tests);
         group "Expr.Temporal.floor" (floor_law :: floor_tests);
         group "Expr.Temporal.offset" [ offset_law ];
         group "Expr.Temporal.parse and format"
           (formats_round_trip :: parse_tests);
       ])
