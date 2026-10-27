(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What the suites of Nx_kernel.S's copies, casts and maps and of nx.cpu's
   share: arrays over drawn bytes, the reference cast, the kinds of each
   arity and their checks, and the programs a map is split into. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module B = Rig.Buffer
module Support = Nx_kernels_support

let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = )

(* [a] where the host reads it: [a] itself on the host, a copy elsewhere. *)
let host a =
  if Rig.equal (A.device a) Rig.host then a else A.to_device Rig.host a

(* [a] on [b]'s device, its layout kept. *)
let on (b : Support.backend) a =
  if Rig.equal (A.device a) b.device then a else A.to_device b.device a

(* Fails if [b] declines [k] at [dt], a case it states it computes. *)
let check_declined (b : Support.backend) k (D.Any dt as d) =
  if b.computes k d then
    failf "%s declined a kind it computes at %s" b.name (D.name dt)

let strf = Printf.sprintf

let pp_dtype ppf (D.Any dt) = D.pp ppf dt

let dtypes = Gen.of_list ~pp:pp_dtype D.all

(* Bits *)

(* The bits of [a]'s elements in C order of indices: the bytes of a byte-wide
   dtype's, one code per element of a sub-byte one. *)
let bits_of (type v s) (a : (v, s) A.t) =
  match D.bits (A.dtype a) with
  | 1 -> Array.map Bool.to_int (A.to_array (A.expect D.Bit (A.Any a)))
  | 4 -> A.to_array (Option.get (A.bitcast D.Uint4 a))
  | _ -> A.to_array (Option.get (A.bitcast D.Uint8 a))

(* [bits], as [bits_of] gives them for [d], one string per element. *)
let elements_of (D.Any dt) bits =
  let per = max 1 (D.bits dt / 8) in
  Array.init (Array.length bits / per) (fun i ->
      String.init per (fun j -> Char.chr bits.((i * per) + j)))

(* Bits as [bits_of] gives them for [d], equal where each element has the
   same bits or both are NaN. *)
let same_values d =
  Testable.make ~pp:(Testable.pp (array int)) ~equal:(fun a b ->
      Testable.equal (array (Elements.same d)) (elements_of d a)
        (elements_of d b))

(* Whether [k]'s NaN results are any NaN: all but the moves of bits, and neg
   and abs, which keep a NaN's payload. *)
let any_nan : Support.kind -> bool = function
  | K0 _ | K1 (Copy | Cast | Bitcast | Unary (Neg | Abs)) | K3 Where -> false
  | K1 (Unary _) | K2 _ | K3 Fma -> true

(* The reference cast *)

(* [v] rounded to nearest, ties to even, at [p] significant bits, by integer
   arithmetic: [v] read unsigned unless [signed]. The result is exact in a
   double. *)
let round_int ~signed p v =
  let bit_length m =
    let rec go n m =
      if m = 0L then n else go (n + 1) (Int64.shift_right_logical m 1)
    in
    go 0 m
  in
  let negative = signed && Int64.compare v 0L < 0 in
  let m = if negative then Int64.neg v else v in
  let n = bit_length m in
  let r =
    if n <= p then
      (* [m] below 2^63 converts exactly; 2^63 itself is -2^63's magnitude. *)
      if Int64.compare m 0L < 0 then Float.ldexp 1. 63 else Int64.to_float m
    else
      let s = n - p in
      let q = Int64.shift_right_logical m s in
      let rest = Int64.logand m (Int64.pred (Int64.shift_left 1L s)) in
      let c = Int64.unsigned_compare rest (Int64.shift_left 1L (s - 1)) in
      let up = c > 0 || (c = 0 && Int64.logand q 1L = 1L) in
      Float.ldexp (Int64.to_float (if up then Int64.succ q else q)) s
  in
  if negative then -.r else r

(* The exact value of [x], an element of the integer dtype [dt], read unsigned
   where [dt] is. *)
let int64_of (type v s) (dt : (v, s) D.t) (x : v) : int64 =
  match dt with
  | D.Int64 -> x
  | D.Uint64 -> x
  | D.Int32 -> Int64.of_int32 x
  | D.Uint32 -> Int64.logand (Int64.of_int32 x) 0xFFFF_FFFFL
  | D.Int16 -> Int64.of_int x
  | D.Uint16 -> Int64.of_int x
  | D.Int8 -> Int64.of_int x
  | D.Uint8 -> Int64.of_int x
  | D.Int4 -> Int64.of_int x
  | D.Uint4 -> Int64.of_int x
  | _ -> invalid_arg "int64_of: not an integer dtype"

(* The low [bits] of [v] as a representative: signed or not. *)
let low ~signed bits v =
  let x =
    Int64.to_int (Int64.logand v (Int64.pred (Int64.shift_left 1L bits)))
  in
  if signed && x >= 1 lsl (bits - 1) then x - (1 lsl bits) else x

(* The integer [v] modulo [dt]'s width, as [dt]'s value. *)
let wrap (type w r) (dt : (w, r) D.t) (v : int64) : w =
  match dt with
  | D.Int64 -> v
  | D.Uint64 -> v
  | D.Int32 -> Int64.to_int32 v
  | D.Uint32 -> Int64.to_int32 v
  | D.Int16 -> low ~signed:true 16 v
  | D.Uint16 -> low ~signed:false 16 v
  | D.Int8 -> low ~signed:true 8 v
  | D.Uint8 -> low ~signed:false 8 v
  | D.Int4 -> low ~signed:true 4 v
  | D.Uint4 -> low ~signed:false 4 v
  | _ -> invalid_arg "wrap: not an integer dtype"

(* The integer [v], read unsigned unless [signed], stored into [d]: rounded once
   to a float format, modulo the width to an integer, [v <> 0] to a boolean. *)
let of_integer (type w r) (d : (w, r) D.t) ~signed v : w =
  match D.kind d with
  | D.Float ->
      D.of_float d (round_int ~signed ((D.float_format d).fraction_bits + 1) v)
  | D.Complex ->
      let p = if D.bits d = 64 then 24 else 53 in
      { Complex.re = round_int ~signed p v; im = 0. }
  | D.Boolean -> v <> 0L
  | D.Signed -> wrap d v
  | D.Unsigned -> wrap d v

(* [x], an element of [s], stored into [d] as a cast does. *)
let cast_value (type v s w r) (s : (v, s) D.t) (d : (w, r) D.t) (x : v) : w =
  match D.kind s with
  | D.Float -> D.of_float d x
  | D.Complex -> (
      match D.kind d with
      | D.Boolean -> x.re <> 0. || x.im <> 0.
      | D.Complex ->
          { Complex.re = (D.of_float d x.re).re; im = (D.of_float d x.im).re }
      | _ -> D.of_float d x.re)
  | D.Boolean -> of_integer d ~signed:true (if x then 1L else 0L)
  | D.Signed -> of_integer d ~signed:true (int64_of s x)
  | D.Unsigned -> of_integer d ~signed:false (int64_of s x)

(* The bits of [a]'s float32s: one per element of a float32 array, two of a
   complex64 one, on a last axis. *)
let words (type v s) (a : (v, s) A.t) =
  Option.get (A.bitcast D.Uint32 (Option.get (A.bitcast D.Float32 a)))

(* [a] cast to [d], element by element; a cast to [a]'s own dtype is a copy. A
   float32 that stays one, into or out of complex64's real part, keeps its bits:
   OCaml reads a float32 as a double, which quiets a signalling NaN, so those
   bits are copied here. *)
let reference (type v s w r) (a : (v, s) A.t) (d : (w, r) D.t) : (w, r) A.t =
  let s = A.dtype a in
  match D.equal_witness s d with
  | Some Type.Equal -> A.copy a
  | None ->
      let want =
        A.of_array d
          (L.shape (A.layout a))
          (Array.map (cast_value s d) (A.to_array a))
      in
      let each f =
        List.iter
          (fun i -> f (Iarray.of_array i))
          (indices (Iarray.to_array (L.shape (A.layout a))))
      in
      let re i = Iarray.append i [| 0 |] in
      (match (D.Any s, D.Any d) with
      | D.Any D.Float32, D.Any D.Complex64 ->
          let w = words want and x = words a in
          each (fun i -> A.set w (re i) (A.get x i))
      | D.Any D.Complex64, D.Any D.Float32 ->
          let w = words want and x = words a in
          each (fun i -> A.set w i (A.get x (re i)))
      | _ -> ());
      want

(* Arrays over drawn bytes *)

(* [x]'s low [w] bytes, least significant first. *)
let le w x =
  String.init w (fun i ->
      Char.chr
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical x (8 * i)) 0xFFL)))

let f32_bits x = Int64.of_int32 (Int32.bits_of_float x)

(* Values at the edges of the conversion table: zeros, infinities, NaNs
   signalling and quiet, ties, the narrow formats' largest values and those past
   them, and the integer ranges' bounds. *)
let floats =
  [
    0.;
    -0.;
    infinity;
    neg_infinity;
    1.;
    -1.;
    0.5;
    1.5;
    2.5;
    -2.5;
    127.5;
    -128.5;
    255.5;
    6.;
    7.;
    448.;
    464.;
    57344.;
    61436.;
    61440.;
    61444.;
    65504.;
    65519.99;
    65520.;
    2147483647.;
    2147483648.;
    -2147483649.;
    4294967295.5;
    4294967296.;
    0x1p63;
    -0x1p63;
    0x1p64;
    1e-40;
    5e-324;
    1e300;
    3.4028235e38;
    1e39;
  ]

let specials (type v s) (dt : (v, s) D.t) =
  let f64 = List.map Int64.bits_of_float floats in
  let f32 = List.map f32_bits floats in
  let nan32 = [ 0x7FC00000L; 0xFFC12345L; 0x7F800001L; 0xFFA00002L ] in
  let nan64 =
    [ 0x7FF8000000000000L; 0x7FF0000000000001L; 0xFFF4000000000002L ]
  in
  let ints =
    [
      0L;
      1L;
      -1L;
      Int64.min_int;
      Int64.max_int;
      0x80L;
      0x7FL;
      0x8000L;
      0x7FFFL;
      0x8000_0000L;
      0x7FFF_FFFFL;
      0x100_0001L;
      0x20_0000_0000_0001L;
      0x7FFF_FFFF_FFFF_FC00L;
      -0x800L;
    ]
  in
  match dt with
  | D.Float64 -> List.map (le 8) (f64 @ nan64)
  | D.Float32 -> List.map (le 4) (f32 @ nan32)
  | D.Complex128 ->
      List.concat_map
        (fun x -> [ le 8 x ^ le 8 0L; le 8 0L ^ le 8 x ])
        (f64 @ nan64)
  | D.Complex64 ->
      List.concat_map
        (fun x -> [ le 4 x ^ le 4 0L; le 4 0L ^ le 4 x ])
        (f32 @ nan32)
  | D.Bool -> List.map (le 1) [ 0L; 1L; 2L; 255L ]
  | _ when D.is D.Signed dt || D.is D.Unsigned dt ->
      List.map (le (D.bits dt / 8)) ints
  | _ -> []

type case = Case : ('v, 's) A.t -> case

(* [n] bytes from the seed [seed], with a linear congruential generator. *)
let bytes_of_seed seed n =
  let x = ref (seed land 0xFFFFFFFFFFFF) in
  String.init n (fun _ ->
      x := ((!x * 0x5DEECE66D) + 11) land 0xFFFFFFFFFFFF;
      Char.chr ((!x lsr 24) land 0xFF))

(* A contiguous array of [dt] of shape [s] over the bytes of [seed], every
   seventh element one of [dt]'s specials in turn. *)
let seeded (type v s) (dt : (v, s) D.t) s seed : (v, s) A.t =
  let n = Iarray.fold_left ( * ) 1 s in
  let data = Bytes.of_string (bytes_of_seed seed (max 1 (D.bytes dt n))) in
  (match specials dt with
  | [] -> ()
  | sp ->
      let sp = Array.of_list sp and w = D.bits dt / 8 in
      for i = 0 to (n / 7) - 1 do
        Bytes.blit_string sp.(i mod Array.length sp) 0 data (7 * i * w) w
      done);
  A.v dt (L.contiguous s) (B.of_string (Bytes.to_string data))

let pp_case ppf (Case a) =
  Format.fprintf ppf "%a %a" D.pp (A.dtype a) L.pp (A.layout a)

(* A contiguous array of [dt] of shape [s] over drawn bytes, each element's
   random or one of [dt]'s specials. *)
let drawn (type v s) (dt : (v, s) D.t) s : (v, s) A.t Gen.t =
  let open Gen in
  let n = Iarray.fold_left ( * ) 1 s in
  let bytes w = string_of ~size:(const w) char in
  let+ data =
    if D.bits dt < 8 then bytes (D.bytes dt n)
    else
      let w = D.bits dt / 8 in
      let element =
        match specials dt with
        | [] -> bytes w
        | sp -> frequency [ (3, bytes w); (1, of_list sp) ]
      in
      map (String.concat "") (list ~size:(const n) element)
  in
  A.v dt (L.contiguous s) (B.of_string (if data = "" then "\000" else data))

(* An array of any dtype and shape over drawn bytes, viewed through a movement:
   strided, broadcast, windowed, transposed or reshaped. A drawn movement
   repeats elements a few times in a hundred, so one case in five is
   broadcast along a new leading axis. *)
let case_of dts =
  let open Gen in
  let* (D.Any dt) = dts in
  let* s = shape in
  let* a = drawn dt (Iarray.of_array s) in
  let lead =
    let+ e = int_range 2 3 in
    Some (M.Broadcast (Iarray.of_array (Array.append [| e |] s)))
  in
  let+ m = frequency [ (4, option (movement ~apart:false s)); (1, lead) ] in
  let a = Option.value ~default:a (Option.bind m (fun m -> A.move m a)) in
  Case a

let case = Gen.with_pp pp_case (case_of dtypes)

(* An array of [dt], values from [seed], [h] by [w] and seen through [view]:
   as made, transposed, stepped, or cut into 2x2 windows whose last two axes
   are swapped, as a pooling layer reads them. *)
let large_at dt h w view seed =
  let s =
    match view with 0 -> [| h; w |] | 1 -> [| w; h |] | _ -> [| h; 2 * w |]
  in
  let a = seeded dt (Iarray.of_array s) seed in
  match view with
  | 0 -> Case a
  | 1 -> Case (Option.get (A.move (M.Permute [| 1; 0 |]) a))
  | 2 ->
      let all = { M.start = 0; count = h; step = 1 } in
      Case
        (Option.get
           (A.move (M.Slice [| all; { M.start = 1; count = w; step = 2 } |]) a))
  | _ ->
      let w axis = { M.axis; size = 2; step = 2; dilation = 1 } in
      let v = Option.get (A.move (M.Window [| w 0; w 1 |]) a) in
      Case (Option.get (A.move (M.Permute [| 0; 1; 3; 2 |]) v))

(* Arrays of a few hundred thousand elements at most, in shapes about the
   walk's block and tile sizes, through each view: a job of several threads
   and blocks. *)
let large_of dts =
  let open Gen in
  let* (D.Any dt) = dts in
  let side =
    of_list ~pp:Format.pp_print_int [ 1; 7; 63; 64; 65; 255; 256; 257; 300 ]
  in
  let* h = of_list ~pp:Format.pp_print_int [ 255; 256; 257; 300 ] in
  let* w = side in
  let* view = of_list ~pp:Format.pp_print_int [ 0; 1; 2; 3 ] in
  let+ seed = int in
  large_at dt h w view seed

let large = Gen.with_pp pp_case (large_of dtypes)

(* One large array through each view: the 32 drawn cases of a large law miss
   a view one time in ten thousand. *)
let large_views = List.map (fun v -> large_at D.Float32 257 65 v 1) [ 0; 1; 2; 3 ]

let covers (Case a) =
  let l = A.layout a in
  let n = L.numel l in
  cover "no element" (n = 0);
  cover "one element" (n = 1);
  cover "strided" (n > 1 && not (L.is_contiguous l));
  cover "broadcast" (Iarray.exists (fun s -> s = 0) (L.strides l) && n > 1);
  cover "sub-byte" (D.bits (A.dtype a) < 8)

let covers_large (Case a) =
  let l = A.layout a in
  cover "transposed" (L.rank l = 2 && abs (L.stride l 0) < abs (L.stride l 1));
  cover "stepped" (L.rank l = 2 && L.stride l 1 = 2);
  cover "2x2 windows" (L.rank l = 4);
  cover "several blocks" (L.numel l > 4096)

(* Declines *)

(* Every unary kind, with its name in nx_kinds.h. *)
let unaries =
  Nx_kernel.Prog.
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
      (Asinh, "asinh");
      (Acosh, "acosh");
      (Atanh, "atanh");
      (Erf, "erf");
      (Floor, "floor");
      (Ceil, "ceil");
      (Round, "round");
      (Trunc, "trunc");
    ]

(* Kinds of no, two and three operands *)

module P = Nx_kernel.Prog

let op2s =
  List.map
    (fun b -> P.Binary b)
    P.
      [
        Add; Sub; Mul; Fdiv; Idiv; Mod; Pow; Atan2; Maximum; Minimum; And; Or;
        Xor; Threefry;
      ]
  @ List.map (fun c -> P.Compare c) P.[ Equal; Not_equal; Less; Less_equal ]

let name2 = function
  | P.Binary b -> (
      match b with
      | Add -> "add"
      | Sub -> "sub"
      | Mul -> "mul"
      | Fdiv -> "fdiv"
      | Idiv -> "idiv"
      | Mod -> "mod"
      | Pow -> "pow"
      | Atan2 -> "atan2"
      | Maximum -> "maximum"
      | Minimum -> "minimum"
      | And -> "and"
      | Or -> "or"
      | Xor -> "xor"
      | Threefry -> "threefry")
  | Compare c -> (
      match c with
      | Equal -> "equal"
      | Not_equal -> "not_equal"
      | Less -> "less"
      | Less_equal -> "less_equal")

(* [v]'s low [w] bytes, least significant first. *)
let low_bytes w v =
  List.init w (fun i ->
      Int64.to_int (Int64.logand (Int64.shift_right_logical v (8 * i)) 0xFFL))

(* The bits of nx_kinds.h's kind [name] at each index of [ops], all of the
   dtype [dt], in C order, as [bits_of] gives them: as [dt] for a kind of
   [dt], as a boolean for a comparison. A narrow float computes in float32 and
   rounds once on the store; int4 and uint4 in the 32-bit type of their
   signedness, their result's low bits kept. *)
let rec expected : type v s.
    string -> (v, s) D.t -> compare:bool -> (v, s) A.t array -> int array =
 fun name dt ~compare ops ->
  let shape = L.shape (A.layout ops.(0)) in
  match D.kind dt with
  | D.Float when D.bits dt < 32 ->
      let r =
        expected name D.Float32 ~compare
          (Array.map (fun a -> reference a D.Float32) ops)
      in
      if compare then r
      else
        let b = String.init (Array.length r) (fun i -> Char.chr r.(i)) in
        let b = if b = "" then "\000" else b in
        bits_of
          (reference (A.v D.Float32 (L.contiguous shape) (B.of_string b)) dt)
  | _ ->
  let per idx =
    let idx = Iarray.of_array idx in
    match D.kind dt with
    | D.Float when D.bits dt = 32 ->
        let args =
          Array.map
            (fun a -> Int32.to_int (A.get (words a) idx) land 0xFFFF_FFFF)
            ops
        in
        let r = Nx_kinds_support.f32 name args in
        if compare then [ Bool.to_int (r <> 0) ]
        else low_bytes 4 (Int64.of_int r)
    | D.Float ->
        let r = Nx_kinds_support.f64 name (Array.map (fun a -> A.get a idx) ops) in
        if compare then [ Bool.to_int (r <> 0.) ]
        else low_bytes 8 (Int64.bits_of_float r)
    | D.Boolean ->
        let args = Array.map (fun a -> if A.get a idx then 1L else 0L) ops in
        [ Bool.to_int (Nx_kinds_support.int "u32" name args <> 0L) ]
    | D.Signed | D.Unsigned ->
        let args = Array.map (fun a -> int64_of dt (A.get a idx)) ops in
        let ty =
          (if D.is D.Signed dt then "i" else "u")
          ^ if D.bits dt = 64 then "64" else "32"
        in
        let r =
          match name with
          | "threefry" -> Nx_kinds_support.threefry args.(0) args.(1)
          | "floor" | "ceil" | "round" | "trunc" -> args.(0)
          | _ -> Nx_kinds_support.int ty name args
        in
        if compare then [ Bool.to_int (r <> 0L) ]
        else if D.bits dt < 8 then [ Int64.to_int r land 0xF ]
        else low_bytes (D.bits dt / 8) r
    | D.Complex when D.bits dt = 64 ->
        let part a i =
          Int32.to_int (A.get (words a) (Iarray.append idx [| i |])) land 0xFFFF_FFFF
        in
        let args =
          Array.concat (Array.to_list (Array.map (fun a -> [| part a 0; part a 1 |]) ops))
        in
        let r = Nx_kinds_support.c64 name args in
        if compare then [ Bool.to_int (r.(0) <> 0) ]
        else low_bytes 4 (Int64.of_int r.(0)) @ low_bytes 4 (Int64.of_int r.(1))
    | D.Complex ->
        let args =
          Array.concat
            (Array.to_list
               (Array.map (fun a -> let z = A.get a idx in [| z.Complex.re; z.im |]) ops))
        in
        let r = Nx_kinds_support.c128 name args in
        if compare then [ Bool.to_int (r.(0) <> 0.) ]
        else low_bytes 8 (Int64.bits_of_float r.(0)) @ low_bytes 8 (Int64.bits_of_float r.(1))
  in
  Array.of_list (List.concat_map per (indices (Iarray.to_array shape)))

(* [x]'s shape over drawn bytes, its axes laid out in reverse or in order. *)
let alike (type v s) (x : (v, s) A.t) : (v, s) A.t Gen.t =
  let open Gen in
  let s = L.shape (A.layout x) in
  let r = Iarray.length s in
  let rev = Array.init r (fun i -> r - 1 - i) in
  let* flip = bool in
  if flip && r >= 2 then
    let reversed = Iarray.init r (fun i -> Iarray.get s rev.(i)) in
    let+ y = drawn (A.dtype x) reversed in
    Option.get (A.move (M.Permute (Iarray.of_array rev)) y)
  else drawn (A.dtype x) s

type pair = Pair : ('v, 's) A.t * ('v, 's) A.t -> pair

let pp_pair ppf (Pair (x, y)) =
  Format.fprintf ppf "%a %a, %a" D.pp (A.dtype x) L.pp (A.layout x) L.pp
    (A.layout y)

let pairs =
  Gen.with_pp pp_pair
    (let open Gen in
     let* (Case x) = case_of dtypes in
     let+ y = alike x in
     Pair (x, y))

(* Large views, as the copy laws draw them, against a contiguous operand:
   blocks staged through transposes and jobs of several threads. *)
let large_pairs =
  Gen.with_pp pp_pair
    (let open Gen in
     let* (Case x) =
       large_of
         (of_list ~pp:pp_dtype
            D.[ Any Float32; Any Int8; Any Float64; Any Bfloat16; Any Int4; Any Bit ])
     in
     let+ seed = int in
     Pair (x, seeded (A.dtype x) (L.shape (A.layout x)) seed))

(* The answer of [run] into a seeded destination of [dt] and [shape], checked:
   [Done] with the bytes [want ()], a decline only of a case [b] does not
   claim, and nothing written but on [Done]. A kind outside its domain makes
   no program, or one whose result is not of [dt]. *)
let answers (b : Support.backend) kind (D.Any at) (D.Any dt) shape ~accepted
    ~want (into : 'v 's. ('v, 's) A.t -> A.answer) =
  let dst = on b (seeded dt shape 99) in
  let before = bits_of (host dst) in
  match into dst with
  | A.Done ->
      cover "computed" true;
      equal ~msg:"accepted" bool true accepted;
      equal
        (if any_nan kind then same_values (D.Any dt) else array int)
        (want ()) (bits_of (host dst))
  | A.Declined ->
      cover "declined" true;
      check_declined b kind (D.Any at);
      equal ~msg:"declined writes nothing" (array int) before
        (bits_of (host dst))
  | exception Invalid_argument _ when not accepted ->
      cover "outside its domain" true
  | A.Wrong_dtype when not accepted ->
      cover "outside its domain" true;
      equal ~msg:"refused writes nothing" (array int) before
        (bits_of (host dst))
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

let law_map1 ?(kinds = unaries) (b : Support.backend) (Case x) =
  let module K = (val b.kernels) in
  let dt = A.dtype x and shape = L.shape (A.layout x) in
  let x' = on b x in
  List.iter
    (fun (u, name) ->
      let k = P.Unary u in
      answers b (K1 k) (D.Any dt) (D.Any dt) shape
        ~accepted:(P.accepts1 k dt dt)
        ~want:(fun () -> expected name dt ~compare:false [| x |])
        (fun dst -> Support.map1 (module K) k ~dst x'))
    kinds

let law_map2 ?(kinds = op2s) (b : Support.backend) (Pair (x, y)) =
  let module K = (val b.kernels) in
  let dt = A.dtype x and shape = L.shape (A.layout x) in
  let x' = on b x and y' = on b y in
  List.iter
    (fun k ->
      let compare = match k with P.Compare _ -> true | Binary _ -> false in
      let rd = if compare then D.Any D.Bool else D.Any dt in
      answers b (K2 k) (D.Any dt) rd shape ~accepted:(P.accepts2 k dt)
        ~want:(fun () -> expected (name2 k) dt ~compare [| x; y |])
        (fun dst -> Support.map2 (module K) k ~dst x' y'))
    kinds

(* Programs *)

exception Skip

(* A program over the dtypes [ins], drawn node by node from [st]: one to
   [length] nodes past its In nodes, each a kind whose operands' dtypes it
   takes, a few constants and coordinates among them, every In first; its
   outputs the last node and, sometimes, an earlier one. At most [room]
   operands, outputs and coordinate axes, each axis below [rank]. *)
let program ?(length = 6) st (ins : D.any array) ~room ~rank =
  let nins = Array.length ins in
  let nodes = ref (Array.to_list (Array.init nins (fun k -> P.In k))) in
  let tys = ref (Array.to_list ins) in
  (* Whether each node's bits follow from its loads' and constants': a NaN a
     kind makes is any NaN, so a bitcast reads only such nodes. *)
  let settled = ref (List.init nins (fun _ -> true)) in
  let axes = ref [] in
  let pick l = List.nth l (Random.State.int st (List.length l)) in
  let all_dtypes = Array.of_list D.all in
  let draw () =
    let n = List.length !nodes and ty i = List.nth !tys i in
    let any () = Random.State.int st n in
    let alike i = List.filter (fun j -> ty j = ty i) (List.init n Fun.id) in
    match Random.State.int st 9 with
    | 0 ->
        let (D.Any dt) = all_dtypes.(Random.State.int st (Array.length all_dtypes)) in
        let e = bits_of (seeded dt [| 1 |] (Random.State.bits st)) in
        let e = if D.is D.Boolean dt then [| e.(0) land 1 |] else e in
        P.Const (D.Any dt, String.init (Array.length e) (fun i -> Char.chr e.(i)))
    | 1 ->
        let a = Random.State.int st (max rank 1) in
        if rank = 0 || List.mem a !axes || nins + List.length !axes + 2 > room
        then
          P.Op1 (Copy, ty 0, 0)
        else begin
          axes := a :: !axes;
          P.Coord a
        end
    | 2 ->
        let i = any () in
        P.Op1 (Cast, all_dtypes.(Random.State.int st (Array.length all_dtypes)), i)
    | 3 ->
        let i = pick (List.filter (List.nth !settled) (List.init n Fun.id)) in
        let (D.Any x) = ty i in
        let same = List.filter (fun (D.Any d) -> D.bits d = D.bits x) D.all in
        P.Op1 (Bitcast, pick same, i)
    | 4 | 5 ->
        let i = any () in
        P.Op1 (Unary (fst (pick unaries)), ty i, i)
    | 6 | 7 ->
        let i = any () in
        P.Op2 (pick op2s, i, pick (alike i))
    | _ ->
        let i = any () in
        let conds =
          List.filter
            (fun j -> let (D.Any d) = ty j in D.is D.Boolean d)
            (List.init n Fun.id)
        in
        if Random.State.bool st then P.Op3 (Fma, i, pick (alike i), pick (alike i))
        else if conds = [] then P.Op2 (Compare Less, i, pick (alike i))
        else P.Op3 (Where, pick conds, i, pick (alike i))
  in
  let dtype_of node =
    match node with
    | P.In k -> ins.(k)
    | Coord _ -> D.Any D.Int64
    | Const (d, _) -> d
    | Op1 (_, d, _) -> d
    | Op2 (Compare _, _, _) -> D.Any D.Bool
    | Op2 (_, i, _) | Op3 (_, _, i, _) -> List.nth !tys i
  in
  let has_nan (D.Any d) =
    D.is D.Complex d || (D.is D.Float d && not (D.equal d D.Float4_e2m1fn))
  in
  let settles node =
    let ok i = List.nth !settled i in
    match node with
    | P.In _ | Coord _ | Const _ -> true
    | Op1 (Bitcast, _, i) -> ok i
    | _ when not (has_nan (dtype_of node)) -> true
    | Op1 ((Copy | Cast | Unary (Neg | Abs)), _, i) -> ok i
    | Op3 (Where, _, i, j) -> ok i && ok j
    | _ -> false
  in
  let operands = function
    | P.In _ | Coord _ | Const _ -> []
    | Op1 (_, _, i) -> [ i ]
    | Op2 (_, i, j) -> [ i; j ]
    | Op3 (_, i, j, k) -> [ i; j; k ]
  in
  (* Whether [node]'s kind takes its operands' dtypes [dts], as Prog.v
     states. *)
  let accepts (node : P.node) dts =
    match (node, dts) with
    | Op1 (k, D.Any y, _), [ D.Any x ] -> P.accepts1 k x y
    | Op2 (k, _, _), [ D.Any x; D.Any x' ] -> D.equal x x' && P.accepts2 k x
    | Op3 (k, _, _, _), [ D.Any c; D.Any x; D.Any x' ] ->
        D.equal x x' && P.accepts3 k c x
    | _ -> true
  in
  for _ = 1 to 1 + Random.State.int st length do
    let rec attempt tries =
      let node = draw () in
      if accepts node (List.map (List.nth !tys) (operands node)) then node
      else if tries = 0 then P.Op1 (Copy, List.nth !tys 0, 0)
      else attempt (tries - 1)
    in
    let node = attempt 50 in
    settled := !settled @ [ settles node ];
    nodes := !nodes @ [ node ];
    tys := !tys @ [ dtype_of node ]
  done;
  let n = List.length !nodes in
  let outs =
    if n > nins + 1 && nins + List.length !axes + 2 <= room && Random.State.bool st
    then [| n - 1; nins + Random.State.int st (n - 1 - nins) |]
    else [| n - 1 |]
  in
  ( P.v ~ins:(Iarray.of_array ins) (Iarray.of_list !nodes)
      ~outs:(Iarray.of_array outs),
    Array.of_list !tys )

(* Each node of [p] run on its own through [b]'s kinds of no to three
   operands, over the operands [ops] of [shape]: the arrays of every node.
   Raises [Skip] where a kind is declined. *)
let split (b : Support.backend) (p : P.t) (types : D.any array) shape
    (ops : A.any array) =
  let module K = (val b.kernels) in
  let values = Array.make (Iarray.length p.nodes) ops.(0) in
  let fresh i =
    let (D.Any d) = types.(i) in
    A.Any (A.create b.device d shape)
  in
  let run answer = if answer <> A.Done then raise Skip in
  for i = 0 to Iarray.length p.nodes - 1 do
    values.(i) <-
      (match Iarray.get p.nodes i with
      | In k -> ops.(k)
      | Coord a ->
          let dst = A.create b.device D.Int64 shape in
          run (Support.map0 (module K) (Iota a) ~dst);
          A.Any dst
      | Const (_, bits) ->
          let (A.Any dst) = fresh i in
          run (Support.map0 (module K) (Fill bits) ~dst);
          A.Any dst
      | Op1 (k, _, j) ->
          let (A.Any dst) = fresh i in
          let (A.Any x) = values.(j) in
          run (Support.map1 (module K) k ~dst x);
          A.Any dst
      | Op2 (k, j, l) ->
          let (A.Any dst) = fresh i in
          let (A.Any x) = values.(j) in
          run (Support.map2 (module K) k ~dst x (A.expect (A.dtype x) values.(l)));
          A.Any dst
      | Op3 (k, c, j, l) ->
          let (A.Any c) = values.(c) in
          let (A.Any x) = values.(j) in
          let dst = A.expect (A.dtype x) (fresh i) in
          run (Support.map3 (module K) k ~dst c x (A.expect (A.dtype x) values.(l)));
          A.Any dst)
  done;
  values

(* [p]'s nodes, one per line. *)
let describe (p : P.t) =
  let dname (D.Any d) = D.name d in
  let node = function
    | P.In k -> strf "in %d" k
    | Coord a -> strf "coord %d" a
    | Const (d, b) ->
        strf "const %s 0x%s" (dname d)
          (String.concat "" (List.init (String.length b) (fun i -> strf "%02x" (Char.code b.[i]))))
    | Op1 (Copy, d, i) -> strf "copy %s %d" (dname d) i
    | Op1 (Cast, d, i) -> strf "cast %s %d" (dname d) i
    | Op1 (Bitcast, d, i) -> strf "bitcast %s %d" (dname d) i
    | Op1 (Unary u, d, i) -> strf "%s %s %d" (List.assoc u unaries) (dname d) i
    | Op2 (k, i, j) -> strf "%s %d %d" (name2 k) i j
    | Op3 (Where, c, i, j) -> strf "where %d %d %d" c i j
    | Op3 (Fma, a, b, c) -> strf "fma %d %d %d" a b c
  in
  String.concat "; "
    (List.mapi (fun i n -> strf "%d: %s" i (node n)) (Iarray.to_list p.nodes))

(* The most values [p] holds at once: each from the node that computes it
   through the last node that reads it, or to the end for an output. *)
let held (p : P.t) =
  let n = Iarray.length p.nodes in
  let last = Array.init n Fun.id in
  let read i j = if last.(j) < i then last.(j) <- i in
  for i = 0 to n - 1 do
    match Iarray.get p.nodes i with
    | P.In _ | Coord _ | Const _ -> ()
    | Op1 (_, _, j) -> read i j
    | Op2 (_, j, k) ->
        read i j;
        read i k
    | Op3 (_, j, k, l) ->
        read i j;
        read i k;
        read i l
  done;
  Array.iter (fun o -> last.(o) <- n) (Iarray.to_array p.outs);
  let most = ref 0 in
  for i = 0 to n - 1 do
    let live = ref 0 in
    for v = 0 to i do
      if last.(v) >= i then incr live
    done;
    most := max !most !live
  done;
  !most

(* [x] as a padded load of no padding, filled with zeros: it reads [x] as
   a plain load does. *)
let unpadded (type v s) (x : (v, s) A.t) =
  let none = Iarray.init (L.rank (A.layout x)) (fun _ -> 0) in
  let pad =
    { Nx_kernel.lo = none; hi = none; interior = none; windows = [||] }
  in
  let zero = B.of_string (String.make (D.bytes (A.dtype x) 1) '\000') in
  let fill = A.get (A.v (A.dtype x) (L.contiguous [||]) zero) [||] in
  Nx_kernel.Padded { x; fill; pad }

(* A program's results are its nodes' results, each node run on its own,
   bit for bit, any NaN for a NaN: an intermediate holds its dtype's value
   exactly. With
   [declines], a map that declines a program [declines] does not accept
   fails. [length] bounds the program's nodes past its loads. *)
let law_map_split ?declines ?(labels = true) ?length ?(padded = false)
    (b : Support.backend) (Pair (x, y), seed) =
  let module K = (val b.kernels) in
  let st = Random.State.make [| seed |] in
  let dt = D.Any (A.dtype x) and shape = L.shape (A.layout x) in
  let x = on b x and y = on b y in
  let ins, ops =
    if Random.State.bool st then ([| dt; dt |], [| A.Any x; A.Any y |])
    else ([| dt |], [| A.Any x |])
  in
  (* nx.cpu's loop holds sixteen operands: a program within them is one
     map. *)
  let rank = Iarray.length shape in
  let p, types = program ?length st ins ~room:16 ~rank in
  let load (A.Any x) = if padded then unpadded x else Nx_kernel.Plain x in
  let loads = Array.map load ops in
  match split b p types shape ops with
  | exception Skip -> cover "a node declined" true
  | values -> (
      let dsts =
        Array.map
          (fun o ->
            let (D.Any d) = types.(o) in
            A.Any (on b (seeded d shape (seed + o))))
          (Iarray.to_array p.outs)
      in
      match K.map p ~dsts loads with
      | A.Declined -> (
          cover "map declined" true;
          match declines with
          | Some ok when not (ok p types) ->
              failf "map declined a program it computes: %d nodes"
                (Iarray.length p.nodes)
          | _ -> ())
      | A.Done ->
          cover "computed" true;
          if labels then begin
          cover "several nodes" (Iarray.length p.nodes > Array.length ins + 1);
          let has f = Iarray.exists f p.nodes in
          cover "a coordinate" (has (function P.Coord _ -> true | _ -> false));
          cover "a constant" (has (function P.Const _ -> true | _ -> false));
          cover "a cast" (has (function P.Op1 (Cast, _, _) -> true | _ -> false));
          cover "a where" (has (function P.Op3 (Where, _, _, _) -> true | _ -> false));
          cover "a sub-byte bitcast"
            (has (function
              | P.Op1 (Bitcast, D.Any d, j) ->
                  let (D.Any x) = types.(j) in
                  D.bits x < 8 && not (D.equal x d)
              | _ -> false));
          cover "two outputs" (Array.length (Iarray.to_array p.outs) = 2);
          let axes =
            List.sort_uniq compare
              (List.filter_map
                 (function P.Coord a -> Some a | _ -> None)
                 (List.init (Iarray.length p.nodes) (Iarray.get p.nodes)))
          in
          cover "more than four loop operands"
            (Array.length ins + Iarray.length p.outs + List.length axes > 4);
          let repeated (A.Any x) =
            let l = A.layout x in
            L.numel l > 1 && Iarray.for_all (( = ) 0) (L.strides l)
          in
          cover "a load of one repeated element" (Array.exists repeated ops)
          end;
          if length <> None then
            cover "more than eight values held" (held p > 8);
          Array.iteri
            (fun k o ->
              let (A.Any want) = values.(o) in
              let (A.Any got) = dsts.(k) in
              equal ~msg:(strf "output %d of %s" k (describe p))
                (same_values (D.Any (A.dtype want)))
                (bits_of (host want))
                (bits_of (host got)))
            (Iarray.to_array p.outs)
      | r -> failf "map answered %a" Nx_array_support.pp_answer r)
