(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Scalar programs lowered to tolk give the bits nx's eager kernels give.

   A case lowers one program over operands of one dtype. Its rows apply one kind
   each to the same operands: the operands are broadcast along a first axis of
   one row per kind, and the program selects row [r]'s kind by the coordinate,
   so that one compilation checks every kind of a family. Each operand holds its
   dtype's special values, every pair of a few of them, then random elements. *)

open Windtrap
module D = Nx_array.Dtype
module P = Nx.Prim
module K = Nx_kernel.Prog

let dtypes =
  D.
    [
      Any Float64;
      Any Float32;
      Any Float16;
      Any Bfloat16;
      Any Float8_e4m3fn;
      Any Float8_e5m2;
      Any Float4_e2m1fn;
      Any Int64;
      Any Uint64;
      Any Int32;
      Any Uint32;
      Any Int16;
      Any Uint16;
      Any Int8;
      Any Uint8;
      Any Int4;
      Any Uint4;
      Any Complex128;
      Any Complex64;
      Any Bool;
      Any Bit;
    ]

let name (D.Any d) = D.name d
let bits (D.Any d) = D.bits d
let bytes d = max 1 (bits d / 8)
let is_float (D.Any d) = D.is D.Float d
let is_complex (D.Any d) = D.is D.Complex d
let is_bool (D.Any d) = D.is D.Boolean d

let hex b =
  String.concat ""
    (List.init (String.length b) (fun i ->
         Printf.sprintf "%02x" (Char.code b.[String.length b - 1 - i])))

(* Elements

   An element is its bytes in the host's (little-endian) order. *)

let le w (z : int64) =
  String.init w (fun i ->
      Char.chr
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical z (8 * i)) 0xffL)))

(* A float format's special codes: zeros, the least and greatest subnormals, the
   least normal, one and its successor, the greatest finite, the infinity, quiet
   and signalling NaNs with payloads; each of both signs. *)
let format_specials ~e ~m =
  let open Int64 in
  let w = 1 + e + m in
  let bias = sub (shift_left 1L (e - 1)) 1L in
  let exps = shift_left (sub (shift_left 1L e) 1L) m in
  let top = shift_left 1L (m - 1) in
  let mags =
    [
      0L;
      1L;
      sub (shift_left 1L m) 1L;
      shift_left 1L m;
      shift_left bias m;
      logor (shift_left bias m) 1L;
      sub exps 1L;
      exps;
      logor exps top;
      logor exps 1L;
      logor (logor exps top) 1L;
    ]
  in
  let sign = shift_left 1L (w - 1) in
  List.concat_map (fun x -> [ x; logor x sign ]) mags

(* Values at a narrow format's rounding ties and its saturation: 1 between two
   codes, a subnormal tie, the greatest finite and past it. *)
let ties =
  let p = Float.ldexp 1. in
  List.concat_map
    (fun (m, sub, top, past) ->
      [
        1. +. p (-(m + 1));
        1. +. (3. *. p (-(m + 1)));
        p sub;
        1.5 *. p sub;
        top;
        past;
      ])
    [
      (10, -25, 65504., 65520.);
      (7, -134, 0x1.fep127, 0x1.ffp127);
      (3, -10, 448., 464.);
      (2, -17, 57344., 61440.);
      (1, -2, 6., 7.);
      (23, -150, 0x1.fffffep127, 0x1.ffffffp127);
    ]
  @ [ 5.; 2.5; 0.75 ]

(* [x] and the floats one ulp each side of it, as [bits_of] gives them. *)
let around bits_of xs =
  List.concat_map
    (fun x ->
      let b = bits_of x in
      List.concat_map
        (fun b -> [ b; Int64.logor b (bits_of (-0.)) ])
        [ b; Int64.pred b; Int64.succ b ])
    xs

let f32_bits x =
  Int64.logand (Int64.of_int32 (Int32.bits_of_float x)) 0xffffffffL

let int_specials w =
  let open Int64 in
  let min = shift_left 1L (w - 1) and max = sub (shift_left 1L (w - 1)) 1L in
  [
    0L;
    1L;
    -1L;
    min;
    max;
    2L;
    -2L;
    succ min;
    pred max;
    7L;
    -7L;
    add (shift_left 1L 24) 1L;
    add (shift_left 1L 53) 1L;
    add (shift_left 1L 11) 1L;
    add (shift_left 1L 63) 1025L;
    sub (shift_left 1L 32) 1L;
  ]

let codes n = List.init n Int64.of_int

let specials (D.Any dt as d) : string list =
  let w = bytes d in
  let of_list xs = List.map (le w) xs in
  match dt with
  | D.Float64 ->
      of_list (format_specials ~e:11 ~m:52 @ around Int64.bits_of_float ties)
  | D.Float32 -> of_list (format_specials ~e:8 ~m:23 @ around f32_bits ties)
  | D.Float16 -> of_list (format_specials ~e:5 ~m:10)
  | D.Bfloat16 -> of_list (format_specials ~e:8 ~m:7)
  | D.Float8_e4m3fn | D.Float8_e5m2 | D.Uint8 | D.Int8 -> of_list (codes 256)
  | D.Float4_e2m1fn | D.Int4 | D.Uint4 -> of_list (codes 16)
  | D.Bool | D.Bit -> of_list (codes 2)
  | D.Complex64 ->
      let fs = format_specials ~e:8 ~m:23 in
      let fs = List.filteri (fun i _ -> i < 12) fs in
      List.concat_map (fun re -> List.map (fun im -> le 4 re ^ le 4 im) fs) fs
  | D.Complex128 ->
      let fs = format_specials ~e:11 ~m:52 in
      let fs = List.filteri (fun i _ -> i < 12) fs in
      List.concat_map (fun re -> List.map (fun im -> le 8 re ^ le 8 im) fs) fs
  | _ -> of_list (int_specials (bits d))

(* A random element: random bits, or a float of moderate size. *)
let random (D.Any dt as d) st =
  let w = bytes d in
  let any () = String.init w (fun _ -> Char.chr (Random.State.int st 256)) in
  let moderate () = Random.State.float st 8. -. 4. in
  match dt with
  | D.Bool | D.Bit -> le 1 (Int64.of_int (Random.State.int st 2))
  | D.Int4 | D.Uint4 | D.Float4_e2m1fn ->
      le 1 (Int64.of_int (Random.State.int st 16))
  | _ when Random.State.bool st -> any ()
  | D.Float64 -> K.bits D.Float64 (moderate ())
  | D.Float32 -> K.bits D.Float32 (moderate ())
  | D.Float16 -> K.bits D.Float16 (moderate ())
  | D.Bfloat16 -> K.bits D.Bfloat16 (moderate ())
  | D.Complex64 -> K.bits D.Complex64 { re = moderate (); im = moderate () }
  | D.Complex128 -> K.bits D.Complex128 { re = moderate (); im = moderate () }
  | _ -> le w (Int64.of_int (Random.State.int st 21 - 10))

(* Values *)

type value = V : ('v, 's, Nx.host) Nx.t -> value

let truths es = Array.of_list (List.map (fun e -> e.[0] <> '\000') es)

(* The value of dtype [d] whose elements' bits are [es]. *)
let value (D.Any dt) es =
  let n = List.length es in
  let raw = String.concat "" es in
  let bytes = Array.init (String.length raw) (fun i -> Char.code raw.[i]) in
  match dt with
  | D.Bool -> V (Nx.create dt [| n |] (truths es))
  | D.Bit -> V (Nx.create dt [| n |] (truths es))
  | D.Int4 | D.Uint4 | D.Float4_e2m1fn ->
      V (Nx.bitcast dt (Nx.create D.Uint4 [| n |] bytes))
  | _ ->
      let w = String.length raw / max 1 n in
      let u = Nx.create D.Uint8 [| n * w |] bytes in
      if w = 1 then V (Nx.bitcast dt u)
      else V (Nx.bitcast dt (Nx.reshape [| n; w |] u))

(* Operands [x] and [y] of [d]: [x] holds the specials, then [x] and [y] hold
   every pair of the first twelve, then random elements. *)
let operands d seed =
  let st = Random.State.make [| seed |] in
  let sp = specials d in
  let few = List.filteri (fun i _ -> i < 12) sp in
  let rand n = List.init n (fun _ -> random d st) in
  let x =
    sp @ List.concat_map (fun a -> List.map (fun _ -> a) few) few @ rand 64
  in
  let y =
    rand (List.length sp) @ List.concat_map (fun _ -> few) few @ rand 64
  in
  (x, y)

(* Programs

   A program over operands [ins] whose output, at row [r] of the first axis, is
   the value [rows.(r)] computes. *)

let select ~ins rows =
  let nodes = ref [] and n = ref 0 in
  let add nd =
    nodes := nd :: !nodes;
    incr n;
    !n - 1
  in
  Array.iteri (fun i _ -> ignore (add (K.In i))) ins;
  let outs = List.map (fun row -> row add) rows in
  let coord = add (K.Coord 0) in
  let last = List.length outs - 1 in
  let pick =
    List.fold_left
      (fun (acc, r) o ->
        let at =
          add (K.Const (D.Any D.Int64, K.bits D.Int64 (Int64.of_int r)))
        in
        let c = add (K.Op2 (Compare Equal, coord, at)) in
        (add (K.Op3 (Where, c, o, acc)), r - 1))
      (List.nth outs last, last - 1)
      (List.tl (List.rev outs))
    |> fst
  in
  K.v ~ins:(Iarray.of_array ins)
    (Iarray.of_list (List.rev !nodes))
    ~outs:(Iarray.of_list [ pick ])

(* The node [i], of [d], as its bits in a uint64, or as itself where no kind
   reads them: a complex128. *)
let raw64 add (D.Any dt as d) i =
  let cast t i = add (K.Op1 (Cast, D.Any t, i)) in
  let bitcast t i = add (K.Op1 (Bitcast, D.Any t, i)) in
  match dt with
  | D.Bool | D.Bit -> cast D.Uint64 i
  | D.Int4 | D.Uint4 | D.Float4_e2m1fn -> cast D.Uint64 (bitcast D.Uint4 i)
  | D.Complex128 -> bitcast D.Uint64 (cast D.Float64 i)
  | _ -> (
      match bits d with
      | 64 -> bitcast D.Uint64 i
      | 32 -> cast D.Uint64 (bitcast D.Uint32 i)
      | 16 -> cast D.Uint64 (bitcast D.Uint16 i)
      | _ -> cast D.Uint64 (bitcast D.Uint8 i))

(* The dtype a result of [d] is compared at: its codes for a narrow float, which
   its decoded value would quiet. *)
let compared (D.Any dt as d) =
  match dt with
  | D.Float16 | D.Bfloat16 -> D.Any D.Uint16
  | D.Float8_e4m3fn | D.Float8_e5m2 -> D.Any D.Uint8
  | D.Float4_e2m1fn -> D.Any D.Uint4
  | _ -> d

let as_compared add d i =
  let c = compared d in
  if name c = name d then i else add (K.Op1 (Bitcast, c, i))

(* Running *)

type row = { label : string; node : (K.node -> int) -> int }

(* Runs [p] over [loads] at shape [rows; n], eagerly and lowered, and is the
   rows whose elements differ, at most five: the row's label, the element's
   operands and both results. *)
let check (D.Any r) p labels ops =
  let k = List.length labels in
  let n = match ops with (_, es) :: _ -> List.length es | [] -> 1 in
  let s = [| k; n |] in
  let loads =
    List.map
      (fun (V x, _) ->
        P.Plain
          (Nx.broadcast_to s
             (Nx.reshape [| 1; Array.fold_left ( * ) 1 (Nx.shape x) |] x)))
      ops
  in
  let map () =
    let y, () =
      P.eval ~by:"t"
        (P.Map
           {
             layout = Nx_array.Layout.contiguous s;
             prog = p;
             outs = [ r ];
             loads = Iarray.of_list loads;
           })
    in
    y
  in
  let eager = Array.map (K.bits r) (Nx.to_array (map ())) in
  let lowered = Array.map (K.bits r) (Lowered.run map) in
  let bad = ref [] in
  Array.iteri
    (fun i e ->
      if e <> lowered.(i) && List.length !bad < 5 then
        let row = i / n and j = i mod n in
        let operand (_, es) = hex (List.nth es j) in
        bad :=
          Printf.sprintf "%s of %s: eager %s, lowered %s" (List.nth labels row)
            (String.concat " " (List.map operand ops))
            (hex e)
            (hex lowered.(i))
          :: !bad)
    eager;
  List.rev !bad

let no_difference bad = equal (list string) [] bad

(* Kinds *)

let unaries =
  K.
    [
      (Neg, "neg");
      (Recip, "recip");
      (Abs, "abs");
      (Sign, "sign");
      (Sqrt, "sqrt");
      (Exp, "exp");
      (Exp2, "exp2");
      (Log, "log");
      (Log2, "log2");
      (Log1p, "log1p");
      (Expm1, "expm1");
      (Sin, "sin");
      (Cos, "cos");
      (Tan, "tan");
      (Asin, "asin");
      (Acos, "acos");
      (Atan, "atan");
      (Sinh, "sinh");
      (Cosh, "cosh");
      (Tanh, "tanh");
      (Erf, "erf");
      (Floor, "floor");
      (Ceil, "ceil");
      (Round, "round");
      (Trunc, "trunc");
    ]

let binaries =
  K.
    [
      (Add, "add");
      (Sub, "sub");
      (Mul, "mul");
      (Fdiv, "fdiv");
      (Idiv, "idiv");
      (Mod, "mod");
      (Pow, "pow");
      (Atan2, "atan2");
      (Maximum, "maximum");
      (Minimum, "minimum");
      (And, "and");
      (Or, "or");
      (Xor, "xor");
      (Threefry, "threefry");
    ]

let compares =
  K.
    [
      (Equal, "equal");
      (Not_equal, "not_equal");
      (Less, "less");
      (Less_equal, "less_equal");
    ]

let accepts1 k (D.Any x) (D.Any y) = K.accepts1 k x y
let accepts2 k (D.Any x) = K.accepts2 k x
let accepts3 k (D.Any c) (D.Any x) = K.accepts3 k c x

let finish d rows =
  List.map
    (fun (label, f) -> { label; node = (fun add -> as_compared add d (f add)) })
    rows

let run_rows d ~ins rows ops =
  let p = select ~ins (List.map (fun r -> r.node) rows) in
  check (compared d) p (List.map (fun r -> r.label) rows) ops

let unary_rows d =
  List.filter_map
    (fun (k, label) ->
      if accepts1 (K.Unary k) d d then
        Some (label, fun add -> add (K.Op1 (Unary k, d, 0)))
      else None)
    unaries

(* Every kind of two or three operands of [d], each giving [d]: the comparisons
   cast back to [d], Where by [x < y], Fma of [x], [y], [x]. *)
let binary_rows d =
  let bin =
    List.filter_map
      (fun (k, label) ->
        if accepts2 (K.Binary k) d then
          Some (label, fun add -> add (K.Op2 (Binary k, 0, 1)))
        else None)
      binaries
  in
  let cmp =
    List.map
      (fun (c, label) ->
        (label, fun add -> add (K.Op1 (Cast, d, add (K.Op2 (Compare c, 0, 1))))))
      compares
  in
  let where =
    ( "where",
      fun add -> add (K.Op3 (Where, add (K.Op2 (Compare Less, 0, 1)), 0, 1)) )
  in
  let fma =
    if accepts3 K.Fma d d then
      [ ("fma", fun add -> add (K.Op3 (Fma, 0, 1, 0))) ]
    else []
  in
  bin @ cmp @ (where :: fma)

let by_dtype ds =
  let pp ppf (d, seed) = Format.fprintf ppf "(%s, %d)" (name d) seed in
  Gen.with_pp pp (Gen.pair (Gen.of_list ds) (Gen.int_range 0 1_000_000))

let classes =
  [
    ("float", is_float);
    ("complex", is_complex);
    ("boolean", is_bool);
    ("integer", fun d -> not (is_float d || is_complex d || is_bool d));
  ]

(* Covers [d]'s name and class, and each class of [ds]. *)
let cover_classes ds d =
  cover (name d) true;
  List.iter
    (fun (label, is) -> if List.exists is ds then cover label (is d))
    classes

let family title ds rows ~arity =
  prop title ~count:2
    ~examples:(List.map (fun d -> (d, 7)) ds)
    (by_dtype ds)
    (fun (d, seed) ->
      cover_classes ds d;
      let x, y = operands d seed in
      let ops =
        if arity = 1 then [ (value d x, x) ]
        else [ (value d x, x); (value d y, y) ]
      in
      let ins = Array.make arity d in
      no_difference (run_rows d ~ins (finish d (rows d)) ops))

let unary =
  family "unary kinds give eager's bits"
    (List.filter (fun d -> unary_rows d <> []) dtypes)
    unary_rows ~arity:1

let binary =
  family "binary kinds, comparisons, where and fma give eager's bits" dtypes
    binary_rows ~arity:2

(* Casts and bitcasts: one row per destination, compared by its bits in a
   uint64. A complex128 destination, which no uint64 holds, is compared by its
   real part there and whole by {!into_complex128}. *)

let conversion title k ds dsts =
  prop title ~count:2
    ~examples:(List.map (fun d -> (d, 11)) ds)
    (by_dtype ds)
    (fun (d, seed) ->
      cover_classes ds d;
      let x, _ = operands d seed in
      let rows =
        List.map
          (fun t ->
            {
              label = name t;
              node = (fun add -> raw64 add t (add (K.Op1 (k, t, 0))));
            })
          (dsts d)
      in
      no_difference
        (run_rows (D.Any D.Uint64) ~ins:[| d |] rows [ (value d x, x) ]))

(* A cast into complex128, both parts' bits. *)
let into_complex128 =
  prop "casts into complex128 give eager's bits in both parts" ~count:2
    ~examples:(List.map (fun d -> (d, 13)) dtypes)
    (by_dtype dtypes)
    (fun (d, seed) ->
      cover_classes dtypes d;
      let x, _ = operands d seed in
      let (V x) = value d x in
      let map () =
        fst
          (P.eval ~by:"t"
             (P.Map
                {
                  layout = Nx_array.Layout.contiguous (Nx.shape x);
                  prog =
                    K.of_node ~ins:(Iarray.of_list [ d ])
                      (Op1 (Cast, D.Any D.Complex128, 0));
                  outs = [ D.Complex128 ];
                  loads = Iarray.of_list [ P.Plain x ];
                }))
      in
      let bits xs =
        Array.to_list (Array.map (fun v -> hex (K.bits D.Complex128 v)) xs)
      in
      equal (list string) (bits (Nx.to_array (map ()))) (bits (Lowered.run map)))

let cast = conversion "casts give eager's bits" K.Cast dtypes (fun _ -> dtypes)

let bitcast =
  conversion "bitcasts give eager's bits" K.Bitcast
    (List.filter (fun d -> name d <> "complex128") dtypes)
    (fun d -> List.filter (fun t -> accepts1 K.Bitcast d t) dtypes)

(* Shapes and views *)

let map2 p (D.Any a) (D.Any b) shape loads =
  let map () =
    P.eval ~by:"t"
      (P.Map
         {
           layout = Nx_array.Layout.contiguous shape;
           prog = p;
           outs = [ a; b ];
           loads = Iarray.of_list loads;
         })
  in
  let first () = fst (map ()) and second () = fst (snd (map ())) in
  let bits dt xs = Array.to_list (Array.map (fun v -> hex (K.bits dt v)) xs) in
  equal (list string)
    (bits a (Nx.to_array (first ())))
    (bits a (Lowered.run first));
  equal (list string)
    (bits b (Nx.to_array (second ())))
    (bits b (Lowered.run second))

(* x * y + z into int32, and x - z into float16, over three int32 operands. *)
let two_outputs =
  let i32 = D.Any D.Int32 in
  let open K.E in
  let x = input 0 D.Int32 and y = input 1 D.Int32 and z = input 2 D.Int32 in
  K.of_exprs
    ~ins:(Iarray.of_list [ i32; i32; i32 ])
    (Iarray.of_list [ any (fma x y z); any (cast D.Float16 (x - z)) ])

let ints s xs = Nx.create D.Int32 s (Array.map Int32.of_int xs)

(* A boolean is its byte: true where it is not zero. Its kinds compute on the
   byte, so a byte read as a boolean keeps it. *)
let bool_bytes () =
  let u8 = D.Any D.Uint8 and b = D.Any D.Bool in
  let map p ins out x =
    fst
      (P.eval ~by:"t"
         (P.Map
            {
              layout = Nx_array.Layout.contiguous [| 4 |];
              prog =
                K.v ~ins:(Iarray.of_list ins) (Iarray.of_list p)
                  ~outs:(Iarray.of_list [ List.length p - 1 ]);
              outs = [ out ];
              loads = Iarray.of_list [ P.Plain x ];
            }))
  in
  let x = Nx.create D.Uint8 [| 4 |] [| 0; 1; 2; 255 |] in
  let f () =
    let t = Nx.bitcast D.Bool x in
    map [ K.In 0; Op1 (Bitcast, u8, 0) ] [ b ] D.Uint8 t
  in
  equal (array int) (Nx.to_array (f ())) (Lowered.run f)

let shapes =
  group "shapes and views"
    [
      test "a byte read as a boolean keeps its byte" bool_bytes;
      test "a zero-size value" (fun () ->
          let e = ints [| 0 |] [||] in
          map2 two_outputs (D.Any D.Int32) (D.Any D.Float16) [| 0 |]
            [ P.Plain e; P.Plain e; P.Plain e ]);
      test "a one-element value" (fun () ->
          map2 two_outputs (D.Any D.Int32) (D.Any D.Float16) [| 1 |]
            [
              P.Plain (ints [| 1 |] [| -7 |]);
              P.Plain (ints [| 1 |] [| 3 |]);
              P.Plain (ints [| 1 |] [| 2049 |]);
            ]);
      test "transposed, flipped and broadcast operands" (fun () ->
          let x = Nx.transpose (ints [| 3; 2 |] [| 0; 1; 2; 3; 4; 5 |]) in
          let y =
            Nx.flip (ints [| 2; 3 |] [| 2147483647; -2; 3; -4; 5; -6 |])
          in
          let z =
            Nx.broadcast_to [| 2; 3 |] (ints [| 1; 3 |] [| 65519; -2049; 7 |])
          in
          map2 two_outputs (D.Any D.Int32) (D.Any D.Float16) [| 2; 3 |]
            [ P.Plain x; P.Plain y; P.Plain z ]);
    ]

let () =
  exit
    (run "Lower_arith"
       [ unary; binary; cast; into_complex128; bitcast; shapes ])
