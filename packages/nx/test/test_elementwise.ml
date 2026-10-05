(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Element-wise arithmetic, math, comparison and bitwise operations. Each is
   checked against OCaml's own function on every element, over every layout and
   over the values that break arithmetic: NaN, infinities, signed zeros,
   subnormals and the extremes of each type. *)

open Windtrap
open Nx_test

let pp_float ppf x = Format.fprintf ppf "%.17g" x
let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let floats dtype = viewed ~pp:pp_float dtype Gen.any_float

(* Unary float operations *)

type unary = {
  name : string;
  nx : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t;
  ocaml : float -> float;
  exact : bool;  (** correctly rounded, as IEEE requires of it *)
}

let sign x =
  if Float.is_nan x then x else if x > 0. then 1. else if x < 0. then -1. else x

let unary =
  [
    { name = "abs"; nx = Nx.abs; ocaml = Float.abs; exact = true };
    { name = "neg"; nx = Nx.neg; ocaml = Float.neg; exact = true };
    { name = "sign"; nx = Nx.sign; ocaml = sign; exact = true };
    { name = "square"; nx = Nx.square; ocaml = (fun x -> x *. x); exact = true };
    { name = "sqrt"; nx = Nx.sqrt; ocaml = Float.sqrt; exact = true };
    { name = "recip"; nx = Nx.recip; ocaml = (fun x -> 1. /. x); exact = true };
    { name = "trunc"; nx = Nx.trunc; ocaml = Float.trunc; exact = true };
    { name = "ceil"; nx = Nx.ceil; ocaml = Float.ceil; exact = true };
    { name = "floor"; nx = Nx.floor; ocaml = Float.floor; exact = true };
    { name = "round"; nx = Nx.round; ocaml = Float.round; exact = true };
    {
      name = "rsqrt";
      nx = Nx.rsqrt;
      ocaml = (fun x -> 1. /. Float.sqrt x);
      exact = false;
    };
    { name = "log"; nx = Nx.log; ocaml = Float.log; exact = false };
    { name = "log2"; nx = Nx.log2; ocaml = Float.log2; exact = false };
    { name = "exp"; nx = Nx.exp; ocaml = Float.exp; exact = false };
    { name = "log1p"; nx = Nx.log1p; ocaml = Float.log1p; exact = false };
    { name = "expm1"; nx = Nx.expm1; ocaml = Float.expm1; exact = false };
    { name = "exp2"; nx = Nx.exp2; ocaml = Float.exp2; exact = false };
    { name = "sin"; nx = Nx.sin; ocaml = Float.sin; exact = false };
    { name = "cos"; nx = Nx.cos; ocaml = Float.cos; exact = false };
    { name = "tan"; nx = Nx.tan; ocaml = Float.tan; exact = false };
    { name = "asin"; nx = Nx.asin; ocaml = Float.asin; exact = false };
    { name = "acos"; nx = Nx.acos; ocaml = Float.acos; exact = false };
    { name = "atan"; nx = Nx.atan; ocaml = Float.atan; exact = false };
    { name = "sinh"; nx = Nx.sinh; ocaml = Float.sinh; exact = false };
    { name = "cosh"; nx = Nx.cosh; ocaml = Float.cosh; exact = false };
    { name = "tanh"; nx = Nx.tanh; ocaml = Float.tanh; exact = false };
    { name = "asinh"; nx = Nx.asinh; ocaml = Float.asinh; exact = false };
    { name = "acosh"; nx = Nx.acosh; ocaml = Float.acosh; exact = false };
    { name = "atanh"; nx = Nx.atanh; ocaml = Float.atanh; exact = false };
    { name = "erf"; nx = Nx.erf; ocaml = Float.erf; exact = false };
    {
      name = "sigmoid";
      nx = Nx.sigmoid;
      ocaml =
        (fun x ->
          if x < 0. then Float.exp x /. (1. +. Float.exp x)
          else 1. /. (1. +. Float.exp (-.x)));
      exact = false;
    };
  ]

(* A transcendental function is within a few units in the last place of the
   correctly rounded result. *)
let tolerance ~f32 exact =
  if exact then 0.
  else if f32 then 4. *. epsilon_float *. (2. ** 29.)
  else 4. *. epsilon_float

let agrees ~f32 (u : unary) t =
  let rel = tolerance ~f32 u.exact in
  let round = if f32 then to_f32 else Fun.id in
  let r = Ref.of_nx t in
  equal
    (Ref.witness (close ~rel ()))
    { r with data = Array.map (fun x -> round (u.ocaml x)) r.data }
    (Ref.of_nx (u.nx t))

let unary_ops =
  group "unary operations"
    (List.concat_map
       (fun u ->
         [
           prop
             (u.name ^ " at float64 agrees with OCaml's")
             (floats Nx.float64) (agrees ~f32:false u);
           prop
             (u.name ^ " at float32 agrees with OCaml's, rounded")
             (floats Nx.float32) (agrees ~f32:true u);
         ])
       unary
    @ [
        test "sigmoid of a large negative is the subnormal it rounds to"
          (fun () ->
            let sigmoid = List.find (fun u -> u.name = "sigmoid") unary in
            agrees ~f32:true sigmoid
              (Nx.create Nx.float32 [| 3 |] [| -88.8; -100.; -103. |]);
            agrees ~f32:false sigmoid
              (Nx.create Nx.float64 [| 3 |] [| -709.8; -720.; -744. |]));
      ])

let classifiers =
  group "classifiers"
    [
      prop "isnan, isinf and isfinite classify every float" (floats Nx.float64)
        (fun t ->
          let r = Ref.of_nx t in
          let classify f = { r with data = Array.map f r.data } in
          let bools = Ref.witness bool in
          equal bools (classify Float.is_nan) (Ref.of_nx (Nx.isnan t));
          equal bools
            (classify (fun x -> Float.abs x = infinity))
            (Ref.of_nx (Nx.isinf t));
          equal bools (classify Float.is_finite) (Ref.of_nx (Nx.isfinite t)));
      test "isnan, isinf and isfinite of integers say finite" (fun () ->
          let t = Nx.create Nx.int32 [| 2 |] [| 0l; Int32.max_int |] in
          equal (tensor bool) (Nx.zeros Nx.bool [| 2 |]) (Nx.isnan t);
          equal (tensor bool) (Nx.zeros Nx.bool [| 2 |]) (Nx.isinf t);
          equal (tensor bool) (Nx.ones Nx.bool [| 2 |]) (Nx.isfinite t));
      test
        "erfinv inverts erf on (-1, 1), is infinite at the ends and NaN beyond"
        (fun () ->
          let t = Nx.create Nx.float64 [| 5 |] [| -1.; 1.; 1.5; -2.; nan |] in
          equal
            (tensor (close ~rel:0. ()))
            (Nx.create Nx.float64 [| 5 |]
               [| neg_infinity; infinity; nan; nan; nan |])
            (Nx.erfinv t));
      prop "erf of erfinv keeps about seven digits at float32"
        (Gen.array ~size:(Gen.int_range 0 8) (Gen.float_range (-0.999) 0.999))
        (fun xs ->
          let t = Nx.create Nx.float32 [| Array.length xs |] xs in
          equal (tensor (close ~rel:1e-6 ())) t (Nx.erf (Nx.erfinv t)));
      prop "erf of erfinv is the identity at float64"
        (Gen.array ~size:(Gen.int_range 0 8)
           (Gen.float_range (-0.999999) 0.999999))
        (fun xs ->
          let t = Nx.create Nx.float64 [| Array.length xs |] xs in
          equal (tensor (close ~rel:1e-14 ())) t (Nx.erf (Nx.erfinv t)));
    ]

(* Binary float operations, over operands that broadcast. *)

let broadcast_pair dtype =
  let open Gen in
  let* a = floats dtype in
  let s = Nx.shape a in
  let* kind = int_range 0 2 in
  let b_shape =
    match kind with
    | 0 -> s
    | 1 -> [||]
    | _ -> Array.mapi (fun i d -> if i mod 2 = 0 then 1 else d) s
  in
  let+ xs = array ~size:(constant (Ref.numel b_shape)) any_float
  and+ swap = bool in
  let b = Nx.create dtype b_shape xs in
  if swap then (b, a) else (a, b)

type binary = {
  bname : string;
  bnx : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t;
  bocaml : float -> float -> float;
  bexact : bool;
}

let binary =
  [
    { bname = "add"; bnx = Nx.add; bocaml = ( +. ); bexact = true };
    { bname = "sub"; bnx = Nx.sub; bocaml = ( -. ); bexact = true };
    { bname = "mul"; bnx = Nx.mul; bocaml = ( *. ); bexact = true };
    { bname = "div"; bnx = Nx.div; bocaml = ( /. ); bexact = true };
    { bname = "maximum"; bnx = Nx.maximum; bocaml = Float.max; bexact = true };
    { bname = "minimum"; bnx = Nx.minimum; bocaml = Float.min; bexact = true };
    {
      bname = "mod_ (the sign of the dividend, as C's fmod; nx.mli is silent)";
      bnx = Nx.mod_;
      bocaml = Float.rem;
      bexact = false;
    };
    { bname = "pow"; bnx = Nx.pow; bocaml = Float.pow; bexact = false };
    { bname = "atan2"; bnx = Nx.atan2; bocaml = Float.atan2; bexact = false };
    { bname = "hypot"; bnx = Nx.hypot; bocaml = Float.hypot; bexact = false };
  ]

let binary_ops =
  group "binary operations"
    (List.concat_map
       (fun b ->
         let check ~f32 (x, y) =
           let rel = tolerance ~f32 b.bexact in
           let round = if f32 then to_f32 else Fun.id in
           (* An exact operation has IEEE's bits, a zero's sign included. *)
           let witness = if b.bexact then float_exact else close ~rel () in
           equal (Ref.witness witness)
             (Ref.map2
                (fun u v -> round (b.bocaml u v))
                (Ref.of_nx x) (Ref.of_nx y))
             (Ref.of_nx (b.bnx x y))
         in
         [
           prop
             (b.bname ^ " at float64 agrees with OCaml's")
             (broadcast_pair Nx.float64)
             (check ~f32:false);
           prop
             (b.bname ^ " at float32 agrees with OCaml's, rounded")
             (broadcast_pair Nx.float32)
             (check ~f32:true);
         ])
       binary
    @ [
        test "maximum is +0 and minimum -0 of both zeros, in either order"
          (fun () ->
            let check (type b) name (dt : (float, b) Nx.dtype) =
              let v x = Nx.create dt [| 1 |] [| x |] in
              List.iter
                (fun (a, b) ->
                  let msg op = Printf.sprintf "%s %s (%g, %g)" name op a b in
                  equal ~msg:(msg "maximum") float_exact 0.
                    (Nx.item [ 0 ] (Nx.maximum (v a) (v b)));
                  equal ~msg:(msg "minimum") float_exact (-0.)
                    (Nx.item [ 0 ] (Nx.minimum (v a) (v b))))
                [ (-0., 0.); (0., -0.) ]
            in
            check "float16" Nx.float16;
            check "bfloat16" Nx.bfloat16;
            check "float32" Nx.float32;
            check "float64" Nx.float64);
      ])

(* NaN operands *)

(* An element's bits, a NaN's sign and payload included, do not depend on where
   it falls in the run a kernel computes: in a vector body or the scalar loop
   after it, beside a broadcast operand or a strided one. *)

(* A float dtype seen through its bits: an element is [words] words of [width]
   bits, stored as [word], and [special] draws a word whose exponent is all
   ones, NaN or infinity, in either sign. *)
type float_format =
  | F : {
      fname : string;
      dtype : ('a, 'b) Nx.dtype;
      word : ('c, 'd) Nx.dtype;
      width : int;
      words : int;
      special : int64 Gen.t;
    }
      -> float_format

(* The words of an IEEE layout of [e] exponent and [m] fraction bits whose
   exponent is all ones: infinity and the NaNs whose fraction is at least
   [least]. *)
let special ~e ~m ~least =
  let open Gen in
  let ones = Int64.shift_left (Int64.pred (Int64.shift_left 1L e)) m in
  let+ negative = bool
  and+ fraction =
    frequency
      [
        (1, constant 0L);
        (4, int64_range least (Int64.pred (Int64.shift_left 1L m)));
      ]
  in
  let w = Int64.logor ones fraction in
  if negative then Int64.logor (Int64.shift_left 1L (e + m)) w else w

let formats =
  let f fname dtype word ~e ~m ?(least = 1L) words =
    let special = special ~e ~m ~least in
    F { fname; dtype; word; width = 1 + e + m; words; special }
  in
  [
    f "float16" Nx.float16 Nx.int16 ~e:5 ~m:10 1;
    f "bfloat16" Nx.bfloat16 Nx.int16 ~e:8 ~m:7 1;
    (* E4M3 has no infinity, and its one NaN per sign has all ones. *)
    f "float8_e4m3" Nx.float8_e4m3 Nx.int8 ~e:4 ~m:3 ~least:7L 1;
    f "float8_e5m2" Nx.float8_e5m2 Nx.int8 ~e:5 ~m:2 1;
    f "float32" Nx.float32 Nx.int32 ~e:8 ~m:23 1;
    f "float64" Nx.float64 Nx.int64 ~e:11 ~m:52 1;
    f "complex64" Nx.complex64 Nx.int32 ~e:8 ~m:23 2;
    f "complex128" Nx.complex128 Nx.int64 ~e:11 ~m:52 2;
  ]

type nan_op = {
  oname : string;
  arity : int;
  complex : bool;
  run : 'a 'b. ('a, 'b) Nx.t array -> ('a, 'b) Nx.t;
}

(* A polymorphic binary operation, which [nan_ops] spreads over an array. *)
type binary_fn = { f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let nan_ops =
  let binary ?(complex = false) oname { f } =
    { oname; arity = 2; complex; run = (fun x -> f x.(0) x.(1)) }
  in
  [
    binary ~complex:true "add" { f = Nx.add };
    binary ~complex:true "sub" { f = Nx.sub };
    binary ~complex:true "mul" { f = Nx.mul };
    binary ~complex:true "div" { f = Nx.div };
    binary ~complex:true "pow" { f = Nx.pow };
    binary "maximum" { f = Nx.maximum };
    binary "minimum" { f = Nx.minimum };
    binary "mod_" { f = Nx.mod_ };
    binary "atan2" { f = Nx.atan2 };
    binary "hypot" { f = Nx.hypot };
    {
      oname = "fma";
      arity = 3;
      complex = false;
      run = (fun x -> Nx.fma x.(0) x.(1) x.(2));
    };
  ]

(* The operations whose NaN results nx.mli states: a NaN result is the first NaN
   operand. *)
let arithmetic = [ "add"; "sub"; "mul"; "div"; "fma" ]

(* How an operand of [n] elements reaches the kernel: contiguous, every other
   element of a longer run, or one element broadcast to [n]. *)
type laid = Full | Every_other | Scalar

let pp_laid ppf l =
  Format.pp_print_string ppf
    (match l with
    | Full -> "full"
    | Every_other -> "every other"
    | Scalar -> "scalar")

let pp_bits ppf v = Format.fprintf ppf "0x%Lx" v
let bits_exact = Testable.make ~pp:pp_bits ~equal:Int64.equal

(* [n], each operand's layout and its elements' words, and the step between the
   operands' offsets. *)
let nan_case (F f) arity =
  let open Gen in
  let word =
    frequency
      [
        (3, f.special);
        ( 2,
          if f.width = 64 then int64
          else int64_range 0L (Int64.pred (Int64.shift_left 1L f.width)) );
      ]
  in
  let operand n =
    let* laid = of_list ~pp:pp_laid [ Full; Every_other; Scalar ] in
    let size = f.words * if laid = Scalar then 1 else n in
    let+ words = array ~size:(constant size) word in
    (laid, words)
  in
  let pp ppf (n, operands, step) =
    Format.fprintf ppf "n = %d, offset step %d" n step;
    Array.iter
      (fun (laid, words) ->
        Format.fprintf ppf "@ %a: [%a]" pp_laid laid
          (Format.pp_print_seq ~pp_sep:Format.pp_print_space pp_bits)
          (Array.to_seq words))
      operands
  in
  with_pp pp
    (let* n = int_range 0 40 in
     let+ operands = array ~size:(constant arity) (operand n)
     and+ step = int_range 0 15 in
     (n, operands, step))

(* A float dtype's arrays, made from their elements' words and read back as
   words. *)
type ('a, 'b) codec = {
  of_words : int array -> int64 array -> ('a, 'b) Nx.t;
  to_words : ('a, 'b) Nx.t -> int64 array;
  laid_out : n:int -> off:int -> laid * int64 array -> ('a, 'b) Nx.t;
      (** [laid_out ~n ~off operand] is [operand]'s [n] elements, laid out after
          [off] elements of padding. *)
}

let codec (type a b c d) (dtype : (a, b) Nx.dtype) (word : (c, d) Nx.dtype)
    ~width ~words : (a, b) codec =
  let unsigned v =
    if width = 64 then v
    else Int64.logand v (Int64.pred (Int64.shift_left 1L width))
  in
  let signed v =
    if width < 64 && Int64.compare v (Int64.shift_left 1L (width - 1)) >= 0 then
      Int64.sub v (Int64.shift_left 1L width)
    else v
  in
  let of_words shape ws =
    let shape = if words = 1 then shape else Array.append shape [| words |] in
    Nx.create Nx.int64 shape (Array.map signed ws)
    |> Nx.cast word |> Nx.bitcast dtype
  in
  let to_words t =
    Array.map unsigned (Nx.to_array (Nx.cast Nx.int64 (Nx.bitcast word t)))
  in
  let laid_out ~n ~off (laid, ws) =
    match laid with
    | Scalar -> of_words [||] ws
    | Full ->
        let pad = Array.make (off * words) 0L in
        Nx.shrink
          [| (off, off + n) |]
          (of_words [| off + n |] (Array.append pad ws))
    | Every_other ->
        let len = off + (2 * n) in
        let at j =
          let e = (j / words) - off in
          if e >= 0 && e mod 2 = 0 then ws.((e / 2 * words) + (j mod words))
          else 0L
        in
        Nx.slice
          [ Rs (off, len, 2) ]
          (of_words [| len |] (Array.init (len * words) at))
  in
  { of_words; to_words; laid_out }

let all_scalar operands =
  Array.for_all (fun (laid, _) -> laid = Scalar) operands

(* [op]'s result over every offset of its operands, from 0 to 15 elements, is
   word for word the result of each element computed alone. *)
let alone_and_together c ~words op (n, operands, step) =
  let element i (laid, ws) =
    c.of_words [||]
      (if laid = Scalar then ws else Array.sub ws (i * words) words)
  in
  let expected =
    Array.concat
      (List.init
         (if all_scalar operands then 1 else n)
         (fun i -> c.to_words (op.run (Array.map (element i) operands))))
  in
  for off = 0 to 15 do
    let operands =
      Array.mapi
        (fun k o -> c.laid_out ~n ~off:((off + (k * step)) mod 16) o)
        operands
    in
    equal
      ~msg:(Printf.sprintf "at offset %d" off)
      (array bits_exact) expected
      (c.to_words (op.run operands))
  done

(* Long runs: [n] up to 3000, each operand's layout and its elements' words,
   rarely special, and the lengths, from 1 to 40, of the pieces that cut the
   run. *)
let long_case (F f) arity =
  let open Gen in
  let finite =
    if f.width = 64 then int64
    else int64_range 0L (Int64.pred (Int64.shift_left 1L f.width))
  in
  let word = frequency [ (1, f.special); (40, finite) ] in
  let operand n =
    let* laid = of_list ~pp:pp_laid [ Full; Every_other; Scalar ] in
    let size = f.words * if laid = Scalar then 1 else n in
    let+ words = array ~size:(constant size) word in
    (laid, words)
  in
  let pp ppf (n, operands, cuts) =
    Format.fprintf ppf "n = %d, %d pieces at most" n (Array.length cuts);
    Array.iter
      (fun (laid, _) -> Format.fprintf ppf "@ %a" pp_laid laid)
      operands
  in
  with_pp pp
    (let* n = int_range 0 3000 in
     let+ operands = array ~size:(constant arity) (operand n)
     and+ cuts = array ~size:(constant n) (int_range 1 40) in
     (n, operands, cuts))

(* [op]'s result over a long run is word for word its results over the pieces
   that cut it, each a run of at most 40 elements. *)
let whole_and_pieces c op (n, operands, cuts) =
  let whole = Array.map (c.laid_out ~n ~off:0) operands in
  let piece lo hi =
    Array.map2
      (fun (laid, _) t ->
        if laid = Scalar then t else Nx.shrink [| (lo, hi) |] t)
      operands whole
  in
  let rec pieces k lo acc =
    if lo >= n then Array.concat (List.rev acc)
    else
      let hi = min n (lo + cuts.(k)) in
      pieces (k + 1) hi (c.to_words (op.run (piece lo hi)) :: acc)
  in
  if not (all_scalar operands) then
    equal (array bits_exact) (pieces 0 0 []) (c.to_words (op.run whole))

(* The rule itself, on fixed operands: a NaN result is the first NaN operand. q1
   and q2 are quiet NaNs of payloads 1 and 2, q2 negative, and s1 a signaling
   NaN. *)
let first_nan =
  let f32 ws =
    Nx.bitcast Nx.float32 (Nx.create Nx.int32 [| Array.length ws |] ws)
  in
  let f64 ws =
    Nx.bitcast Nx.float64 (Nx.create Nx.int64 [| Array.length ws |] ws)
  in
  let c64 ws =
    Nx.bitcast Nx.complex64 (Nx.create Nx.int32 [| Array.length ws / 2; 2 |] ws)
  in
  let words32 t = Nx.to_array (Nx.bitcast Nx.int32 t) in
  let words64 t = Nx.to_array (Nx.bitcast Nx.int64 t) in
  let w32 =
    array
      (Testable.make
         ~pp:(fun ppf v -> Format.fprintf ppf "0x%lx" v)
         ~equal:Int32.equal)
  in
  let arith =
    [
      ("add", { f = Nx.add });
      ("sub", { f = Nx.sub });
      ("mul", { f = Nx.mul });
      ("div", { f = Nx.div });
    ]
  in
  group "the first NaN operand"
    [
      test "add, sub, mul and div keep the first NaN operand's bits" (fun () ->
          let q1 = 0x7fc00001l and q2 = 0xffc00002l and s1 = 0x7f800001l in
          let one = 0x3f800000l in
          let a = f32 [| q1; q2; one; s1 |] and b = f32 [| q2; q1; q2; q1 |] in
          let q1d = 0x7ff8000000000001L and q2d = 0xfff8000000000002L in
          let s1d = 0x7ff0000000000001L and oned = 0x3ff0000000000000L in
          let ad = f64 [| q1d; q2d; oned; s1d |]
          and bd = f64 [| q2d; q1d; q2d; q1d |] in
          List.iter
            (fun (name, { f }) ->
              equal ~msg:(name ^ " at float32") w32 [| q1; q2; q2; s1 |]
                (words32 (f a b));
              equal ~msg:(name ^ " at float64") (array bits_exact)
                [| q1d; q2d; q2d; s1d |]
                (words64 (f ad bd)))
            arith);
      test "fma keeps the first of its three operands that is NaN" (fun () ->
          let q1 = 0x7fc00001l and q2 = 0xffc00002l and s1 = 0x7f800001l in
          let one = 0x3f800000l in
          equal w32 [| q1; q2; q2; s1 |]
            (words32
               (Nx.fma
                  (f32 [| one; q2; one; one |])
                  (f32 [| q1; q1; one; s1 |])
                  (f32 [| q2; s1; q2; q1 |]))));
      test "a narrow float keeps the first NaN operand's sign" (fun () ->
          let check (type b w) name (dt : (float, b) Nx.dtype)
              (word : (int, w) Nx.dtype) ~neg ~pos =
            let nan w = Nx.bitcast dt (Nx.create word [| 1 |] [| w |]) in
            let sign x =
              Float.sign_bit (Nx.item [ 0 ] (Nx.cast Nx.float32 x))
            in
            List.iter
              (fun (op, { f }) ->
                let msg = Printf.sprintf "%s at %s" op name in
                equal ~msg bool true (sign (f (nan neg) (nan pos)));
                equal ~msg bool false (sign (f (nan pos) (nan neg))))
              arith
          in
          check "float16" Nx.float16 Nx.int16 ~neg:(-511) ~pos:0x7e02;
          check "bfloat16" Nx.bfloat16 Nx.int16 ~neg:(-63) ~pos:0x7fc2;
          check "float8_e4m3" Nx.float8_e4m3 Nx.int8 ~neg:(-1) ~pos:0x7f;
          check "float8_e5m2" Nx.float8_e5m2 Nx.int8 ~neg:(-2) ~pos:0x7d);
      test
        "a NaN part of a complex result is the operands' first NaN part, or \
         else the positive quiet NaN" (fun () ->
          let q1 = 0x7fc00001l and q2 = 0xffc00002l and one = 0x3f800000l in
          let a = c64 [| one; q1 |] and b = c64 [| q2; one |] in
          List.iter
            (fun (name, { f }) ->
              equal ~msg:name w32 [| q1; q1 |] (words32 (f a b)))
            arith;
          let inf = 0x7f800000l and ninf = 0xff800000l in
          equal ~msg:"own NaN" w32 [| 0x7fc00000l; 0l |]
            (words32 (Nx.add (c64 [| inf; 0l |]) (c64 [| ninf; 0l |]))));
    ]

let nan_operands =
  group "NaN operands"
    (List.concat_map
       (fun (F f as format) ->
         List.concat_map
           (fun op ->
             if Nx_dtype.is_complex f.dtype && not op.complex then []
             else
               let c = codec f.dtype f.word ~width:f.width ~words:f.words in
               let alone =
                 prop
                   (Printf.sprintf
                      "%s at %s gives each element the bits it has alone"
                      op.oname f.fname)
                   (nan_case format op.arity)
                   (alone_and_together c ~words:f.words op)
               in
               let long =
                 prop ~count:10
                   (Printf.sprintf
                      "%s at %s gives a long run the bits of its pieces"
                      op.oname f.fname)
                   (long_case format op.arity)
                   (whole_and_pieces c op)
               in
               if List.mem op.oname arithmetic then [ alone; long ]
               else [ alone ])
           nan_ops)
       formats
    @ [ first_nan ])

(* Near zero, and multiply-adds *)

(* [x] and its value under [f], from a sweep around 0 of both signs. *)
let near_zero =
  List.concat_map
    (fun e -> [ Float.ldexp 1. e; -.Float.ldexp 1.5 e ])
    [ -1074; -1022; -600; -60; -30; -27; -12; -1 ]

(* The float32 multiply-add rounded once: the product of float32 values is exact
   in a double, and the sum rounded to odd there rounds once to float32. *)
let fma32 a b c =
  let p = a *. b in
  let r = p +. c in
  let d = r -. p in
  let e = p -. (r -. d) +. (c -. d) in
  let odd = Int64.logand (Int64.bits_of_float r) 1L = 1L in
  to_f32
    (if (not (Float.is_finite r)) || e = 0. || odd then r
     else if e > 0. then Float.succ r
     else Float.pred r)

(* Three operands of one shape, but for the first two, which broadcast. *)
let float_triple dtype =
  let open Gen in
  let* a, b = broadcast_pair dtype in
  let s = Nx.shape (Nx.add a b) in
  let+ c = array ~size:(constant (Ref.numel s)) any_float in
  (a, b, Nx.create dtype s c)

(* [fma_agrees f (a, b, c)] checks [Nx.fma a b c] against [f] on each
   element. *)
let fma_agrees f (a, b, c) =
  let r = Nx.fma a b c in
  let flat x = Nx.to_array (Nx.broadcast_to (Nx.shape r) x) in
  let a = flat a and b = flat b and c = flat c in
  equal (Ref.witness float_exact)
    {
      (Ref.of_nx r) with
      data = Array.init (Array.length c) (fun i -> f a.(i) b.(i) c.(i));
    }
    (Ref.of_nx r)

let near_zero_and_fma =
  let v dt xs = Nx.create dt [| Array.length xs |] xs in
  group "near zero, and multiply-adds"
    [
      test "log1p and expm1 of a value near zero keep its digits" (fun () ->
          let x = v Nx.float64 (Array.of_list near_zero) in
          equal (tensor float_exact) (Nx.map_item Float.log1p x) (Nx.log1p x);
          equal (tensor float_exact) (Nx.map_item Float.expm1 x) (Nx.expm1 x));
      test "log1p and expm1 keep the sign of a zero" (fun () ->
          let x = v Nx.float32 [| 0.; -0. |] in
          equal (tensor float_exact) x (Nx.log1p x);
          equal (tensor float_exact) x (Nx.expm1 x));
      test "log1p is -inf at -1 and NaN below, and expm1 is -1 at -inf"
        (fun () ->
          equal (tensor float_exact)
            (v Nx.float64 [| Float.neg_infinity; Float.nan; Float.infinity |])
            (Nx.log1p (v Nx.float64 [| -1.; -2.; Float.infinity |]));
          equal (tensor float_exact)
            (v Nx.float64 [| -1.; Float.infinity |])
            (Nx.expm1 (v Nx.float64 [| Float.neg_infinity; Float.infinity |])));
      test "log1p and expm1 refuse integers and complex numbers" (fun () ->
          raises_invalid_arg (fun () -> Nx.log1p (Nx.zeros Nx.int32 [| 2 |]));
          raises_invalid_arg (fun () -> Nx.expm1 (Nx.zeros Nx.int32 [| 2 |]));
          raises_invalid_arg (fun () ->
              Nx.log1p (Nx.zeros Nx.complex64 [| 2 |])));
      prop "fma at float64 rounds once, as OCaml's" (float_triple Nx.float64)
        (fma_agrees Float.fma);
      prop "fma at float32 rounds once" (float_triple Nx.float32)
        (fma_agrees fma32);
      prop
        "fma at float16, bfloat16 and the float8 dtypes is float32's, rounded"
        (float_triple Nx.float32) (fun (a, b, c) ->
          let narrowed (type d) name (dt : (float, d) Nx.dtype) =
            let a = Nx.cast dt a and b = Nx.cast dt b and c = Nx.cast dt c in
            let wide t = Nx.cast Nx.float32 t in
            equal ~msg:name
              (tensor (close ~rel:0. ()))
              (Nx.cast dt (Nx.fma (wide a) (wide b) (wide c)))
              (Nx.fma a b c)
          in
          narrowed "float16" Nx.float16;
          narrowed "bfloat16" Nx.bfloat16;
          narrowed "float8_e4m3" Nx.float8_e4m3;
          narrowed "float8_e5m2" Nx.float8_e5m2);
      test "fma keeps the product that its sum cancels" (fun () ->
          let a = v Nx.float32 [| 1. +. 0x1p-12 |] in
          equal (tensor float_exact)
            (v Nx.float32 [| 0x1p-24 |])
            (Nx.fma a a (v Nx.float32 [| -1. -. 0x1p-11 |])));
      test "fma refuses complex numbers and booleans" (fun () ->
          let c = Nx.zeros Nx.complex64 [| 2 |]
          and b = Nx.zeros Nx.bool [| 2 |] in
          raises_invalid_arg (fun () -> Nx.fma c c c);
          raises_invalid_arg (fun () -> Nx.fma b b b));
    ]

(* Checks *)

let index i = String.concat "," (Array.to_list (Array.map string_of_int i))

(* [checked ok] is the index [Nx.check ok] names, or [None]. *)
let checked ok =
  match Nx.check ok index with
  | () -> None
  | exception Invalid_argument m -> Some m

let bools = viewed ~pp:Format.pp_print_bool Nx.bool Gen.bool

let checks =
  group "checks"
    [
      prop "a check names the first false element in C order, over any layout"
        bools (fun ok ->
          let r = Ref.of_nx ok in
          let expected =
            Option.map
              (fun i -> index (Ref.unravel r.shape i))
              (Array.find_index not r.data)
          in
          equal (option string) expected (checked ok));
      test "a check of no element passes" (fun () ->
          equal (option string) None (checked (Nx.zeros Nx.bool [| 0; 3 |])));
      test "a check of a scalar names the empty index" (fun () ->
          equal (option string) (Some "") (checked (Nx.scalar Nx.bool false)));
      test "a passing check makes no message" (fun () ->
          Nx.check (Nx.ones Nx.bool [| 4 |]) (fun _ ->
              fail "a message was made"));
    ]

(* A scalar variant is its operation against a scalar tensor, on either side. *)
let scalar_variants =
  let exact = tensor (close ~rel:0. ()) in
  prop "each scalar variant is its operation with the scalar on its side"
    (Gen.pair (floats Nx.float64) Gen.any_float)
    (fun (t, s) ->
      let c = Nx.scalar Nx.float64 s in
      let right name op op_s = equal ~msg:name exact (op t c) (op_s t s) in
      let left name op rop_s = equal ~msg:name exact (op c t) (rop_s s t) in
      right "add_s" Nx.add Nx.add_s;
      right "sub_s" Nx.sub Nx.sub_s;
      right "mul_s" Nx.mul Nx.mul_s;
      right "div_s" Nx.div Nx.div_s;
      right "pow_s" Nx.pow Nx.pow_s;
      right "mod_s" Nx.mod_ Nx.mod_s;
      right "maximum_s" Nx.maximum Nx.maximum_s;
      right "minimum_s" Nx.minimum Nx.minimum_s;
      left "rsub_s" Nx.sub Nx.rsub_s;
      left "rdiv_s" Nx.div Nx.rdiv_s;
      left "rpow_s" Nx.pow Nx.rpow_s;
      left "rmod_s" Nx.mod_ Nx.rmod_s)

let refusals =
  test "a binary operation refuses shapes that do not broadcast" (fun () ->
      raises_invalid_arg (fun () ->
          Nx.add
            (Nx.zeros Nx.float32 [| 2; 3 |])
            (Nx.zeros Nx.float32 [| 3; 2 |])))

let comparisons =
  let cmp name nx ocaml =
    prop (name ^ " agrees with OCaml's IEEE comparison")
      (broadcast_pair Nx.float64) (fun (x, y) ->
        equal (Ref.witness bool)
          (Ref.map2 ocaml (Ref.of_nx x) (Ref.of_nx y))
          (Ref.of_nx (nx x y)))
  in
  group "comparisons"
    [
      cmp "less" Nx.less (fun (a : float) b -> a < b);
      cmp "less_equal" Nx.less_equal (fun (a : float) b -> a <= b);
      cmp "greater" Nx.greater (fun (a : float) b -> a > b);
      cmp "greater_equal" Nx.greater_equal (fun (a : float) b -> a >= b);
      cmp "equal" Nx.equal (fun (a : float) b -> a = b);
      cmp "not_equal" Nx.not_equal (fun (a : float) b -> a <> b);
      prop "where picks from the first where the condition holds"
        (broadcast_pair Nx.float64) (fun (x, y) ->
          let r = Ref.map2 (fun a b -> (a, b)) (Ref.of_nx x) (Ref.of_nx y) in
          let cond =
            { r with data = Array.map (fun (a, _) -> a < 0.) r.data }
          in
          equal
            (Ref.witness (close ~rel:0. ()))
            {
              r with
              data = Array.map (fun (a, b) -> if a < 0. then a else b) r.data;
            }
            (Ref.of_nx (Nx.where (Nx.create Nx.bool cond.shape cond.data) x y)));
      prop "clamp is minimum of the upper bound and maximum of the lower"
        (Gen.triple (floats Nx.float64)
           (Gen.float_range (-10.) 0.)
           (Gen.float_range 0. 10.))
        (fun (t, lo, hi) ->
          equal
            (tensor (close ~rel:0. ()))
            (Nx.minimum_s (Nx.maximum_s t lo) hi)
            (Nx.clamp ~min:lo ~max:hi t));
      prop "lerp is a + w (b - a)" (broadcast_pair Nx.float64) (fun (a, b) ->
          let w = Nx.scalar Nx.float64 0.25 in
          equal
            (tensor (close ~rel:0. ()))
            (Nx.add a (Nx.mul w (Nx.sub b a)))
            (Nx.lerp a b w));
    ]

(* Integers wrap, as OCaml's fixed-width integers do. *)

let int32_value =
  Gen.frequency
    [
      (4, Gen.map Int32.of_int (Gen.int_range (-9) 9));
      (2, Gen.int32);
      ( 1,
        Gen.of_list
          ~pp:(fun ppf v -> Format.fprintf ppf "%ld" v)
          [ Int32.min_int; Int32.max_int; -1l; 0l ] );
    ]

let pp_int32 ppf v = Format.fprintf ppf "%ld" v
let int32s = viewed ~pp:pp_int32 Nx.int32 int32_value

(* Tensors of one shape, for the laws. *)
let int32_tuple n =
  let open Gen in
  let* s = array ~size:(int_range 0 3) (int_range 0 3) in
  let* steps = layout in
  let one = viewed ~shape:(constant s) ~layout:(constant ~pp:pp_layout steps) in
  list ~size:(constant n) (one ~pp:pp_int32 Nx.int32 int32_value)

let pair_of g =
  Gen.map (function [ a; b ] -> (a, b) | _ -> assert false) (g 2)

let triple_of g =
  Gen.map (function [ a; b; c ] -> (a, b, c) | _ -> assert false) (g 3)

let ints = tensor int32

let int_laws =
  let pairs = pair_of int32_tuple and triples = triple_of int32_tuple in
  group "integer laws"
    [
      prop "add is associative" triples (Law.associative ints Nx.add);
      prop "add is commutative" pairs (Law.commutative ints Nx.add);
      prop "zero is neutral for add" int32s (fun x ->
          Law.neutral ints Nx.add (Nx.zeros_like x) x);
      prop "neg inverts add" int32s (fun x ->
          Law.invertible ints Nx.add (Nx.zeros_like x) Nx.neg x);
      prop "mul is associative" triples (Law.associative ints Nx.mul);
      prop "mul is commutative" pairs (Law.commutative ints Nx.mul);
      prop "one is neutral for mul" int32s (fun x ->
          Law.neutral ints Nx.mul (Nx.ones_like x) x);
      prop "zero absorbs mul" int32s (fun x ->
          Law.absorbing ints Nx.mul (Nx.zeros_like x) x);
      prop "mul distributes over add" triples
        (Law.distributive ints Nx.mul ~over:Nx.add);
      prop "bitwise_and is associative" triples
        (Law.associative ints Nx.bitwise_and);
      prop "bitwise_or is associative" triples
        (Law.associative ints Nx.bitwise_or);
      prop "bitwise_xor is associative" triples
        (Law.associative ints Nx.bitwise_xor);
      prop "bitwise_xor is its own inverse" int32s (fun x ->
          Law.invertible ints Nx.bitwise_xor (Nx.zeros_like x) Fun.id x);
      prop "bitwise_and distributes over bitwise_or" triples
        (Law.distributive ints Nx.bitwise_and ~over:Nx.bitwise_or);
      prop "neg is involutive" int32s (Law.involutive ints Nx.neg);
      prop "bitwise_not is involutive" int32s
        (Law.involutive ints Nx.bitwise_not);
      prop "maximum is associative" triples (Law.associative ints Nx.maximum);
      prop "minimum is commutative" pairs (Law.commutative ints Nx.minimum);
    ]

let int_ops =
  group "integer operations"
    [
      test "shifts refuse a negative count and a float dtype" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.lshift (Nx.zeros Nx.int32 [| 2 |]) (-1));
          raises_invalid_arg (fun () ->
              Nx.rshift (Nx.zeros Nx.int32 [| 2 |]) (-1));
          raises_invalid_arg (fun () ->
              Nx.lshift (Nx.zeros Nx.float32 [| 2 |]) 1));
      prop "logical operations read non-zero as true" (pair_of int32_tuple)
        (fun (a, b) ->
          let truth v = v <> 0l and of_bool c = if c then 1l else 0l in
          let ra = Ref.of_nx a and rb = Ref.of_nx b in
          let both f =
            Ref.map2 (fun x y -> of_bool (f (truth x) (truth y))) ra rb
          in
          equal (Ref.witness int32) (both ( && ))
            (Ref.of_nx (Nx.logical_and a b));
          equal (Ref.witness int32) (both ( || ))
            (Ref.of_nx (Nx.logical_or a b));
          equal (Ref.witness int32) (both ( <> ))
            (Ref.of_nx (Nx.logical_xor a b));
          equal (Ref.witness int32)
            {
              ra with
              data = Array.map (fun v -> of_bool (not (truth v))) ra.data;
            }
            (Ref.of_nx (Nx.logical_not a)));
    ]

(* Integers of every width wrap there, and unsigned ones order, divide and shift
   as unsigned: each operation agrees with int64 arithmetic wrapped to the
   width. Division and remainder by zero give zero. *)

(* Integer power: a negative exponent gives zero, but for bases one and minus
   one. *)
let ipow b e =
  if Int64.compare e 0L < 0 then
    if b = 1L then 1L
    else if b = -1L then if Int64.rem e 2L = 0L then 1L else -1L
    else 0L
  else
    let r = ref 1L in
    for _ = 1 to Int64.to_int e do
      r := Int64.mul !r b
    done;
    !r

let integer_dtypes =
  group "integer dtypes"
    (List.map
       (fun (Int_dtype d) ->
         let values =
           Gen.array ~size:(Gen.int_range 0 8)
             (int_value ~bits:d.bits ~signed:d.signed)
         in
         prop
           (d.name
          ^ " arithmetic, bits and order agree with int64's at its width")
           (Gen.triple values values (Gen.int_range 0 (d.bits - 1)))
           (fun (xs, ys, n) ->
             let len = Int.min (Array.length xs) (Array.length ys) in
             let xs = Array.sub xs 0 len and ys = Array.sub ys 0 len in
             let t v = Nx.create d.dtype [| len |] (Array.map d.of_i64 v) in
             let a = t xs and b = t ys in
             let w = wrap ~bits:d.bits ~signed:d.signed in
             let values f = Array.map (fun v -> d.of_i64 (w v)) f in
             let check msg expected actual =
               equal ~msg (array d.exact) (values expected) (Nx.to_array actual)
             in
             let both f = Array.map2 f xs ys in
             let by_zero f x y = if y = 0L then 0L else f x y in
             let div = if d.signed then Int64.div else Int64.unsigned_div in
             let rem = if d.signed then Int64.rem else Int64.unsigned_rem in
             let cmp = int_compare ~signed:d.signed in
             check "add" (both Int64.add) (Nx.add a b);
             check "sub" (both Int64.sub) (Nx.sub a b);
             check "mul" (both Int64.mul) (Nx.mul a b);
             check "fma"
               (both (fun x y -> Int64.add (Int64.mul x y) x))
               (Nx.fma a b a);
             check "div" (both (by_zero div)) (Nx.div a b);
             check "mod_" (both (by_zero rem)) (Nx.mod_ a b);
             check "maximum"
               (both (fun x y -> if cmp x y >= 0 then x else y))
               (Nx.maximum a b);
             check "minimum"
               (both (fun x y -> if cmp x y <= 0 then x else y))
               (Nx.minimum a b);
             check "bitwise_and" (both Int64.logand) (Nx.bitwise_and a b);
             check "bitwise_or" (both Int64.logor) (Nx.bitwise_or a b);
             check "bitwise_xor" (both Int64.logxor) (Nx.bitwise_xor a b);
             check "neg" (Array.map Int64.neg xs) (Nx.neg a);
             check "recip" (Array.map (by_zero div 1L) xs) (Nx.recip a);
             check "abs"
               (Array.map (fun x -> if d.signed then Int64.abs x else x) xs)
               (Nx.abs a);
             check "bitwise_not" (Array.map Int64.lognot xs) (Nx.bitwise_not a);
             check "lshift"
               (Array.map (fun x -> Int64.shift_left x n) xs)
               (Nx.lshift a n);
             check "rshift"
               (Array.map
                  (fun x ->
                    if d.signed then Int64.shift_right x n
                    else Int64.shift_right_logical x n)
                  xs)
               (Nx.rshift a n);
             let exps =
               Array.map
                 (fun y ->
                   if d.signed then Int64.rem y 6L else Int64.unsigned_rem y 6L)
                 ys
             in
             check "pow" (Array.map2 ipow xs exps) (Nx.pow a (t exps));
             equal ~msg:"less" (array bool)
               (both (fun x y -> cmp x y < 0))
               (Nx.to_array (Nx.less a b));
             equal ~msg:"equal" (array bool) (both ( = ))
               (Nx.to_array (Nx.equal a b))))
       int_dtypes)

let packed_ints =
  let q = Nx.zeros Nx.int4 [| 2 |] and b = Nx.ones Nx.bool [| 2 |] in
  cases "int4 computes nothing, and bool no arithmetic"
    ~name:(fun (n, _) -> n)
    [
      ("int4 add", fun () -> ignore (Nx.add q q));
      ("int4 maximum", fun () -> ignore (Nx.maximum q q));
      ("int4 equal", fun () -> ignore (Nx.equal q q));
      ("bool add", fun () -> ignore (Nx.add b b));
      ("bool neg", fun () -> ignore (Nx.neg b));
      ("bool sum", fun () -> ignore (Nx.sum b));
      ("bool cumsum", fun () -> ignore (Nx.cumsum b));
    ]
    (fun (_, f) -> raises_invalid_arg f)

(* Complex numbers *)

let pp_complex ppf (z : Complex.t) = Format.fprintf ppf "(%g, %g)" z.re z.im
let component = Gen.float_range (-100.) 100.

let complex_value =
  Gen.(
    let+ re = component and+ im = component in
    Complex.{ re; im })

let complexes = viewed ~pp:pp_complex Nx.complex128 complex_value

(* Complex numbers within [rel] of the larger modulus. *)
let complex_close ~rel =
  Testable.make
    ~pp:(fun ppf (z : Complex.t) ->
      Format.fprintf ppf "(%.17g, %.17g)" z.re z.im)
    ~equal:(fun (a : Complex.t) (b : Complex.t) ->
      let same x y = (Float.is_nan x && Float.is_nan y) || x = y in
      (same a.re b.re && same a.im b.im)
      || Complex.norm (Complex.sub a b)
         <= rel *. Float.max (Complex.norm a) (Complex.norm b))

let complex_pair =
  let open Gen in
  let* s = array ~size:(int_range 0 3) (int_range 0 3) in
  let* steps = layout in
  let one = viewed ~shape:(constant s) ~layout:(constant ~pp:pp_layout steps) in
  pair
    (one ~pp:pp_complex Nx.complex128 complex_value)
    (one ~pp:pp_complex Nx.complex128 complex_value)

(* Two float64 tensors of one shape and layout, of any elements. *)
let component_pair =
  let open Gen in
  let* s = array ~size:(int_range 0 3) (int_range 0 3) in
  let* steps = layout in
  let one =
    viewed ~shape:(constant s)
      ~layout:(constant ~pp:pp_layout steps)
      ~pp:pp_float Nx.float64 any_float
  in
  pair one one

(* [components z] is [z]'s components as floats, along a last axis of two. *)
let components z = Nx.bitcast Nx.float64 z

(* [complex] of the parts [real] and [imag] take apart is [z], whatever the bits
   of its components. *)
let reassembles (type c e) name (dt : (Complex.t, c) Nx.dtype)
    (part : (float, e) Nx.dtype) zs =
  prop
    ("complex gives back a " ^ name ^ " from its parts, bit for bit")
    zs
    (fun z ->
      equal Stored.packed (Nx.P z)
        (Nx.P (Nx.complex dt ~re:(Nx.real part z) ~im:(Nx.imag part z))))

(* [conjugate] keeps the real part, negates the imaginary one, and undoes itself
   bit for bit. *)
let conjugates name zs =
  prop ("conjugate negates the imaginary part of a " ^ name) zs (fun z ->
      let r = Ref.of_nx z in
      let map f = { r with data = Array.map f r.data } in
      let floats = Ref.witness float_exact in
      let w = Nx.conjugate z in
      equal ~msg:"real" floats
        (map (fun z -> z.Complex.re))
        (Ref.of_nx (Nx.real Nx.float64 w));
      equal ~msg:"imag" floats
        (map (fun z -> Float.neg z.Complex.im))
        (Ref.of_nx (Nx.imag Nx.float64 w));
      equal ~msg:"twice" Stored.packed (Nx.P z) (Nx.P (Nx.conjugate w)))

let complex_numbers =
  let agree name rel nx ocaml =
    prop (name ^ " agrees with Stdlib.Complex") complex_pair (fun (a, b) ->
        equal
          (Ref.witness (complex_close ~rel))
          (Ref.map2 ocaml (Ref.of_nx a) (Ref.of_nx b))
          (Ref.of_nx (nx a b)))
  in
  let parts t = (Nx.real Nx.float64 t, Nx.imag Nx.float64 t) in
  let floats64 = tensor (close ~rel:0. ()) in
  group "complex numbers"
    [
      agree "add" 0. Nx.add Complex.add;
      agree "sub" 0. Nx.sub Complex.sub;
      agree "mul" 1e-15 Nx.mul Complex.mul;
      (* Division by zero has no one convention: the divisors are non-zero. *)
      prop "div agrees with Stdlib.Complex" complex_pair (fun (a, b) ->
          let rb = Ref.of_nx b in
          assume (Array.for_all (fun z -> z <> Complex.zero) rb.data);
          equal
            (Ref.witness (complex_close ~rel:1e-14))
            (Ref.map2 Complex.div (Ref.of_nx a) rb)
            (Ref.of_nx (Nx.div a b)));
      test "div keeps a zero part where the other part overflows" (fun () ->
          let c re im = Nx.create Nx.complex128 [||] [| { Complex.re; im } |] in
          let q = Nx.item [] (Nx.div (c 0. 0x1p-50) (c 0. (-0x1p-1074))) in
          equal
            (pair float_exact float_exact)
            (Float.neg_infinity, 0.) (q.re, q.im));
      prop
        "real, imag, magnitude, angle and conjugate agree with Stdlib.Complex"
        complexes (fun z ->
          let r = Ref.of_nx z in
          let map f = { r with data = Array.map f r.data } in
          let reals = Ref.witness (close ~rel:1e-15 ()) in
          equal reals
            (map (fun z -> z.Complex.re))
            (Ref.of_nx (Nx.real Nx.float64 z));
          equal reals
            (map (fun z -> z.Complex.im))
            (Ref.of_nx (Nx.imag Nx.float64 z));
          equal reals (map Complex.norm) (Ref.of_nx (Nx.magnitude Nx.float64 z));
          equal reals (map Complex.arg) (Ref.of_nx (Nx.angle Nx.float64 z));
          equal
            (Ref.witness (complex_close ~rel:0.))
            (map Complex.conj)
            (Ref.of_nx (Nx.conjugate z)));
      prop "complex assembles what real and imag take apart" complexes
        (Law.round_trip
           (tensor (complex_close ~rel:0.))
           (pair floats64 floats64) parts
           (fun (re, im) -> Nx.complex Nx.complex128 ~re ~im));
      prop "complex's components read back as floats are re and im"
        component_pair (fun (re, im) ->
          equal (tensor float_exact)
            (Nx.stack ~axis:(-1) [ re; im ])
            (components (Nx.complex Nx.complex128 ~re ~im)));
      test "complex keeps non-finite components, NaN and signed zeros"
        (fun () ->
          let re = Nx.create Nx.float64 [| 4 |] [| 1.; infinity; nan; -0. |]
          and im = Nx.create Nx.float64 [| 4 |] [| infinity; 1.; -0.; nan |] in
          equal (tensor float_exact)
            (Nx.create Nx.float64 [| 4; 2 |]
               [| 1.; infinity; infinity; 1.; nan; -0.; -0.; nan |])
            (components (Nx.complex Nx.complex128 ~re ~im));
          let z =
            Nx.item []
              (Nx.complex Nx.complex64 ~re:(Nx.scalar Nx.float64 1.)
                 ~im:(Nx.scalar Nx.float64 infinity))
          in
          equal (pair float_exact float_exact) (1., infinity) (z.re, z.im));
      test "complex broadcasts re against im" (fun () ->
          let re = Nx.create Nx.float64 [| 2 |] [| 1.; -0. |]
          and im = Nx.create Nx.float64 [| 2; 1 |] [| infinity; -0. |] in
          equal (tensor float_exact)
            (Nx.create Nx.float64 [| 2; 2; 2 |]
               [| 1.; infinity; -0.; infinity; 1.; -0.; -0.; -0. |])
            (components (Nx.complex Nx.complex128 ~re ~im)));
      cases "the sign of a zero imaginary part picks the side of the branch cut"
        ~name:(fun (re, im, _) -> Printf.sprintf "angle (%g, %g)" re im)
        [
          (-1., 0., Float.pi);
          (-1., -0., -.Float.pi);
          (-0., 0., Float.pi);
          (0., 0., 0.);
        ]
        (fun (re, im, expected) ->
          equal (close ~rel:0. ()) expected
            (Nx.item []
               (Nx.angle Nx.float64 (Nx.scalar Nx.complex128 { re; im }))));
      test "real and magnitude read non-finite components exactly" (fun () ->
          let z =
            Nx.create Nx.complex128 [| 3 |]
              [|
                { re = infinity; im = nan };
                { re = -.infinity; im = 0. };
                { re = 1e300; im = 1e300 };
              |]
          in
          equal floats64
            (Nx.create Nx.float64 [| 3 |] [| infinity; -.infinity; 1e300 |])
            (Nx.real Nx.float64 z);
          equal
            (tensor (close ~rel:1e-15 ()))
            (Nx.create Nx.float64 [| 3 |]
               [| infinity; infinity; 1e300 *. Float.sqrt 2. |])
            (Nx.magnitude Nx.float64 z));
      prop
        "abs is the modulus and sign the unit in the same direction, zero at \
         zero"
        complexes (fun z ->
          let r = Ref.of_nx z in
          let map f = { r with data = Array.map f r.data } in
          let unit (w : Complex.t) =
            let m = Complex.norm w in
            if m = 0. then Complex.zero
            else Complex.{ re = w.re /. m; im = w.im /. m }
          in
          equal
            (Ref.witness (complex_close ~rel:1e-15))
            (map (fun w -> Complex.{ re = Complex.norm w; im = 0. }))
            (Ref.of_nx (Nx.abs z));
          equal
            (Ref.witness (complex_close ~rel:1e-15))
            (map unit)
            (Ref.of_nx (Nx.sign z)));
      test "complex numbers refuse order, remainder and rounding" (fun () ->
          let z =
            Nx.create Nx.complex128 [| 2 |]
              Complex.[| { re = 1.; im = 1. }; { re = 2.; im = 0. } |]
          in
          let refuses f =
            raises_match (fun _ -> true) (fun () -> ignore (f ()))
          in
          refuses (fun () -> Nx.less z z);
          refuses (fun () -> Nx.mod_ z z);
          refuses (fun () -> Nx.round z));
      test
        "real, imag, angle and conjugate keep infinities, NaN and signed zeros"
        (fun () ->
          let parts =
            [|
              (1., infinity);
              (infinity, 1.);
              (nan, -0.);
              (-0., nan);
              (neg_infinity, -0.);
              (-1., -0.);
            |]
          in
          let n = Array.length parts in
          let floats f = Nx.create Nx.float64 [| n |] (Array.map f parts) in
          let check (type c) (dt : (Complex.t, c) Nx.dtype) =
            let msg = Nx_dtype.to_string dt in
            let z =
              Nx.create dt [| n |]
                (Array.map (fun (re, im) -> { Complex.re; im }) parts)
            in
            equal ~msg (tensor float_exact) (floats fst) (Nx.real Nx.float64 z);
            equal ~msg (tensor float_exact) (floats snd) (Nx.imag Nx.float64 z);
            equal ~msg floats64
              (floats (fun (re, im) -> Float.atan2 im re))
              (Nx.angle Nx.float64 z);
            equal ~msg (tensor float_exact) (floats fst)
              (Nx.real Nx.float64 (Nx.conjugate z));
            equal ~msg (tensor float_exact)
              (floats (fun (_, im) -> Float.neg im))
              (Nx.imag Nx.float64 (Nx.conjugate z))
          in
          check Nx.complex64;
          check Nx.complex128);
      test "imag and conjugate read a broadcast view" (fun () ->
          let row =
            Nx.create Nx.complex128 [| 3 |]
              [|
                { re = 1.; im = infinity };
                { re = nan; im = -0. };
                { re = -0.; im = 2. };
              |]
          in
          let z = Nx.broadcast_to [| 2; 3 |] row in
          equal (tensor float_exact)
            (Nx.create Nx.float64 [| 2; 3 |]
               [| infinity; -0.; 2.; infinity; -0.; 2. |])
            (Nx.imag Nx.float64 z);
          let pairs =
            Nx.create Nx.float64 [| 3; 2 |]
              [| 1.; neg_infinity; nan; 0.; -0.; -2. |]
          in
          equal (tensor float_exact)
            (Nx.broadcast_to [| 2; 3; 2 |] pairs)
            (components (Nx.conjugate z)));
      reassembles "complex64" Nx.complex64 Nx.float32 Stored.complex64s;
      reassembles "complex128" Nx.complex128 Nx.float64 Stored.complex128s;
      conjugates "complex64" Stored.complex64s;
      conjugates "complex128" Stored.complex128s;
    ]

(* float16 and bfloat16 compute as float32 and round once, which gives the
   correctly rounded result: float32 carries more than twice their precision. *)

type narrow = Narrow : string * (float, 'b) Nx.dtype -> narrow

let narrow_floats =
  let narrow_pair (type b) (dt : (float, b) Nx.dtype) =
    Gen.map
      (fun (a, b) -> (Nx.cast dt a, Nx.cast dt b))
      (broadcast_pair Nx.float32)
  in
  group "narrow floats"
    (List.concat_map
       (fun (Narrow (name, dt)) ->
         let exact = tensor (close ~rel:0. ()) in
         let wide t = Nx.cast Nx.float32 t in
         [
           prop (name ^ " arithmetic is float32's, rounded once")
             (narrow_pair dt) (fun (a, b) ->
               let once f = Nx.cast dt (f (wide a) (wide b)) in
               equal ~msg:"add" exact (once Nx.add) (Nx.add a b);
               equal ~msg:"sub" exact (once Nx.sub) (Nx.sub a b);
               equal ~msg:"mul" exact (once Nx.mul) (Nx.mul a b);
               equal ~msg:"div" exact (once Nx.div) (Nx.div a b);
               equal ~msg:"maximum" exact (once Nx.maximum) (Nx.maximum a b);
               equal ~msg:"less" (tensor bool)
                 (Nx.less (wide a) (wide b))
                 (Nx.less a b);
               equal ~msg:"sqrt" exact
                 (Nx.cast dt (Nx.sqrt (wide a)))
                 (Nx.sqrt a));
           prop
             (name ^ " unary operations are float32's, rounded once")
             (Gen.map (Nx.cast dt) (floats Nx.float32))
             (fun a ->
               List.iter
                 (fun (u : unary) ->
                   equal ~msg:u.name exact (Nx.cast dt (u.nx (wide a))) (u.nx a))
                 unary);
         ])
       [
         Narrow ("float16", Nx.float16);
         Narrow ("bfloat16", Nx.bfloat16);
         Narrow ("float8_e4m3", Nx.float8_e4m3);
         Narrow ("float8_e5m2", Nx.float8_e5m2);
       ]
    @ [
        test "float16 and bfloat16 sums accumulate wider than they store"
          (fun () ->
            equal float_exact 4096.
              (Nx.item [] (Nx.sum (Nx.ones Nx.float16 [| 4096 |])));
            equal float_exact 1024.
              (Nx.item [] (Nx.sum (Nx.ones Nx.bfloat16 [| 1024 |]))));
      ])

(* Booleans *)

let booleans =
  let bools = Gen.array ~size:(Gen.int_range 0 8) Gen.bool in
  group "booleans"
    [
      prop "logical operations, where and min on bool agree with OCaml's"
        (Gen.pair bools bools) (fun (xs, ys) ->
          let n = Int.min (Array.length xs) (Array.length ys) in
          let xs = Array.sub xs 0 n and ys = Array.sub ys 0 n in
          let t v = Nx.create Nx.bool [| n |] v in
          let a = t xs and b = t ys in
          let both f = Array.map2 f xs ys in
          equal ~msg:"logical_and" (array bool) (both ( && ))
            (Nx.to_array (Nx.logical_and a b));
          equal ~msg:"logical_or" (array bool) (both ( || ))
            (Nx.to_array (Nx.logical_or a b));
          equal ~msg:"logical_xor" (array bool) (both ( <> ))
            (Nx.to_array (Nx.logical_xor a b));
          equal ~msg:"logical_not" (array bool) (Array.map not xs)
            (Nx.to_array (Nx.logical_not a));
          equal ~msg:"where" (array bool)
            (both (fun x y -> if x then y else not y))
            (Nx.to_array (Nx.where a b (Nx.logical_not b)));
          if n > 0 then
            equal ~msg:"min" bool (Array.for_all Fun.id xs)
              (Nx.item [] (Nx.min a)));
    ]

(* Data types *)

type packed = D : string * ('a, 'b) Nx.dtype -> packed

let dtypes =
  group "data types"
    [
      cases "each dtype says whether it is float, complex, int or unsigned"
        ~name:(fun (D (name, _), _) -> name)
        [
          (D ("float16", Nx.float16), (true, false, false, false));
          (D ("bfloat16", Nx.bfloat16), (true, false, false, false));
          (D ("float8_e4m3", Nx.float8_e4m3), (true, false, false, false));
          (D ("float8_e5m2", Nx.float8_e5m2), (true, false, false, false));
          (D ("float64", Nx.float64), (true, false, false, false));
          (D ("complex64", Nx.complex64), (false, true, false, false));
          (D ("int4", Nx.int4), (false, false, true, false));
          (D ("uint4", Nx.uint4), (false, false, true, true));
          (D ("uint32", Nx.uint32), (false, false, true, true));
          (D ("uint64", Nx.uint64), (false, false, true, true));
          (D ("bool", Nx.bool), (false, false, false, false));
        ]
        (fun (D (_, dt), expected) ->
          equal (quad bool bool bool bool) expected
            ( Nx_dtype.is_float dt,
              Nx_dtype.is_complex dt,
              Nx_dtype.is_int dt,
              Nx_dtype.is_uint dt ));
      test "narrow integers hold their width's range" (fun () ->
          equal (pair int int) (-8, 7)
            (Nx_dtype.min_value Nx.int4, Nx_dtype.max_value Nx.int4);
          equal (pair int int) (0, 15)
            (Nx_dtype.min_value Nx.uint4, Nx_dtype.max_value Nx.uint4);
          equal (pair int32 int32) (0l, -1l)
            (Nx_dtype.min_value Nx.uint32, Nx_dtype.max_value Nx.uint32);
          equal (pair int64 int64) (0L, -1L)
            (Nx_dtype.min_value Nx.uint64, Nx_dtype.max_value Nx.uint64);
          equal (pair bool bool) (false, true)
            (Nx_dtype.min_value Nx.bool, Nx_dtype.max_value Nx.bool));
    ]

(* Operations large enough to run on the worker pool. *)

(* [nx] over [n] float64 values against [ocaml] on each, as the positions where
   they differ. *)
let agrees nx ocaml n =
  let x = Bigarray.(Array1.create float64 c_layout n) in
  for i = 0 to n - 1 do
    x.{i} <- float_of_int (i mod 1013)
  done;
  let y =
    Bigarray.array1_of_genarray
      (Nx.to_bigarray (nx (Nx.of_bigarray (Bigarray.genarray_of_array1 x))))
  in
  let bad = ref [] in
  for i = n - 1 downto 0 do
    if y.{i} <> ocaml (float_of_int (i mod 1013)) then bad := i :: !bad
  done;
  equal (list int) [] !bad

(* A square root of a million elements, as IEEE rounds it, runs on the pool. *)
let roots () = agrees Nx.sqrt Float.sqrt 1_000_000

let the_pool =
  group "the worker pool"
    [
      test
        "an operation runs on the pool before a fork, in the child and in the \
         parent after it" (fun () ->
          if Sys.win32 then skip ~reason:"no fork on Windows" ();
          roots ();
          let pid = Unix.fork () in
          if pid = 0 then
            Unix._exit (match roots () with () -> 0 | exception _ -> 1)
          else
            let deadline = Unix.gettimeofday () +. 30. in
            let rec wait () =
              match Unix.waitpid [ Unix.WNOHANG ] pid with
              | 0, _ when Unix.gettimeofday () < deadline ->
                  Unix.sleepf 0.01;
                  wait ()
              | 0, _ ->
                  Unix.kill pid Sys.sigkill;
                  ignore (Unix.waitpid [] pid);
                  failf "the child did not finish in 30 s"
              | _, status -> status
            in
            equal ~msg:"the child's exit" bool true (wait () = Unix.WEXITED 0);
            roots ());
      slow "an operation over 720 MB of traffic computes every element"
        (fun () -> agrees (fun x -> Nx.add x x) (fun v -> v +. v) 30_000_000);
    ]

let () =
  exit
    (run "nx elementwise"
       [
         unary_ops;
         near_zero_and_fma;
         checks;
         classifiers;
         binary_ops;
         nan_operands;
         scalar_variants;
         refusals;
         comparisons;
         int_laws;
         int_ops;
         integer_dtypes;
         packed_ints;
         complex_numbers;
         booleans;
         narrow_floats;
         dtypes;
         the_pool;
       ])
