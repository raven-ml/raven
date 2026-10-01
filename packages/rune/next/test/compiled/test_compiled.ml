(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The compiled backend against nx.cpu. Each kernel runs on the same operands in
   both backends: strided, broadcast, reversed, offset, in a buffer that starts
   inside another, empty or scalar. Its results agree to the class the lowering
   states for it: bit for bit, within the error of a rounded sum, within a
   budget of units in the last place, or within a measured bound. A refusal
   raises before any work; a program is compiled once per key, from any domain;
   and linear algebra gives non-finite values where nx.cpu raises. The host runs
   the programs; Metal, the full sweeps and the cost figures are slow. *)

open Windtrap
open Nx_test
module B = Nx_device.Buffer
module V = Nx_array.View
module E = Nx_array.Elements
module Scalar = Nx_dtype.Scalar

type ('a, 'b) arr = ('a, 'b) Nx_array.t

let compiled = Nx_backend.kernels Rune_next.Compiled.backend
let cpu = (module Nx_cpu : Nx_backend.S)
let host = Nx_device.host
let numel shape = Array.fold_left ( * ) 1 shape

(* Devices *)

(* A device the compiled programs run on. [flushes] is whether its arithmetic
   reads and writes float32 subnormals as zeros of their sign, as Metal's does,
   bfloat16 ones included since they compute at float32; [budgets] are its
   measured maxima of units in the last place, by row, where they differ from
   the lowering's. *)
type device = {
  device : Nx_device.t;
  name : string;
  float64 : bool;
  flushes : bool;
  budgets : (string * int) list;
}

let on_host =
  {
    device = host;
    name = "host";
    float64 = true;
    flushes = false;
    budgets = [];
  }

let on_metal =
  Option.map
    (fun device ->
      { device; name = "metal"; float64 = false; flushes = true; budgets = [] })
    Metal.device

(* Arrays *)

let pp_ints ppf a =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_seq
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Format.pp_print_int)
    (Array.to_seq a)

let pp_pairs ppf a =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_seq
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       (fun ppf (x, y) -> Format.fprintf ppf "(%d, %d)" x y))
    (Array.to_seq a)

let value (a : ('a, 'b) arr) = Nx.Repr.host a
let shape_of (a : ('a, 'b) arr) = V.shape a.view

let pp_arr ppf (a : ('a, 'b) arr) =
  Format.fprintf ppf "@[<v>%s %a strides %a offset %d, %d bytes in:@ %a@]"
    (Nx_dtype.to_string a.dtype)
    pp_ints (V.shape a.view) pp_ints (V.strides a.view) (V.offset a.view)
    (B.offset a.buffer) Nx.pp (value a)

let buffer d dtype n = B.create d (Scalar.of_dtype dtype) (Int.max 1 n)

(* A new C-contiguous array on [d]. *)
let fresh d dtype shape : ('a, 'b) arr =
  { dtype; view = V.create shape; buffer = buffer d dtype (numel shape) }

(* A copy of [b] on [d], starting as many bytes past a multiple of 16 into its
   memory as [b] does. *)
let copied d b =
  let lead = B.offset b mod 16 in
  let inner = lead * 8 / Scalar.bitsize (B.dtype b) in
  let b' =
    B.view
      (B.create d (B.dtype b) (inner + B.length b))
      ~offset:lead (B.dtype b) (B.length b)
  in
  B.copy ~src:b ~dst:b';
  b'

(* [a] with its buffer on [d], its view unchanged. *)
let moved d (a : ('a, 'b) arr) =
  if Nx_device.equal d host then a else { a with buffer = copied d a.buffer }

(* [a] as a host value, read back from its device. *)
let back (a : ('a, 'b) arr) =
  if Nx_device.equal (B.device a.buffer) host then value a
  else value { a with buffer = copied host a.buffer }

let host_array x =
  match Nx.Repr.v x with
  | Host a -> a
  | Placed _ | Traced _ -> fail "expected a host value"

let array_of x = host_array (Nx.copy x)

(* A run of a kernel: [on] puts an operand where the backend computes, and [dst]
   makes a destination there. *)
type env = {
  on : 'a 'b. ('a, 'b) arr -> ('a, 'b) arr;
  dst : 'a 'b. ('a, 'b) Nx_dtype.t -> int array -> ('a, 'b) arr;
}

let on_cpu = { on = Fun.id; dst = (fun dt s -> fresh host dt s) }

let on_device d =
  { on = (fun a -> moved d.device a); dst = (fun dt s -> fresh d.device dt s) }

(* [both d f] is [f] run by nx.cpu on the host and by the compiled backend on
   [d]. *)
let both d f = (f cpu on_cpu, f compiled (on_device d))

(* Layouts *)

(* Strides and an offset for a shape over a buffer of [length] elements, which
   starts [inner] elements into another. *)
type layout = { strides : int array; offset : int; length : int; inner : int }

(* The axes of [shape] in any order, each reversed or not, with gaps between
   rows, and, where [broadcast], some of stride zero. The buffer starts up to
   three elements into its memory. *)
let layout ?(broadcast = true) shape =
  let open Gen in
  let r = Array.length shape in
  let* order = permutation (List.init r Fun.id) in
  let* reversed = array ~size:(constant r) bool in
  let* gaps = array ~size:(constant r) (int_range 0 2) in
  let* zero =
    array ~size:(constant r)
      (if broadcast then frequency [ (5, constant false); (1, constant true) ]
       else constant false)
  in
  let* before = int_range 0 7 in
  let* start = int_range 0 3 in
  let+ after = int_range 0 3 in
  let inner = start in
  let strides = Array.make r 0 and span = ref 1 in
  List.iter
    (fun axis ->
      if not zero.(axis) then begin
        strides.(axis) <- (if reversed.(axis) then - !span else !span);
        span := !span * (shape.(axis) + gaps.(axis))
      end)
    (List.rev order);
  if numel shape = 0 then { strides; offset = 0; length = before + 1; inner }
  else
    let lo = ref 0 and hi = ref 0 in
    Array.iteri
      (fun d s ->
        if s < 0 then lo := !lo + (s * (shape.(d) - 1))
        else hi := !hi + (s * (shape.(d) - 1)))
      strides;
    let offset = before - !lo in
    { strides; offset; length = offset + !hi + 1 + after; inner }

(* [store dt b i v] writes element [i] of [b] as the bits [v], through the
   unsigned integers of [dt]'s width. *)
let store (type a b) (dt : (a, b) Nx_dtype.t) b : int -> int64 -> unit =
  let n = B.length b in
  match Nx_dtype.itemsize dt with
  | 1 ->
      let set = E.set Nx.uint8 (B.view b ~offset:0 Scalar.UInt8 n) in
      fun i v -> set i (Int64.to_int v land 0xff)
  | 2 ->
      let set = E.set Nx.uint16 (B.view b ~offset:0 Scalar.UInt16 n) in
      fun i v -> set i (Int64.to_int v land 0xffff)
  | 4 ->
      let set = E.set Nx.uint32 (B.view b ~offset:0 Scalar.UInt32 n) in
      fun i v -> set i (Int64.to_int32 v)
  | _ ->
      let set = E.set Nx.uint64 (B.view b ~offset:0 Scalar.UInt64 n) in
      fun i v -> set i v

(* The array of [dtype] and [shape] laid out by [l] over the elements [bits],
   those the view does not reach included. *)
let of_bits dtype shape l bits : ('a, 'b) arr =
  let buffer =
    B.view
      (buffer host dtype (l.inner + l.length))
      ~offset:(l.inner * Nx_dtype.itemsize dtype)
      (Scalar.of_dtype dtype) l.length
  in
  Array.iteri (store dtype buffer) bits;
  { dtype; view = V.create ~offset:l.offset ~strides:l.strides shape; buffer }

(* The bits of each dtype that break arithmetic, as Nx_test draws them. *)
let bits_of (type a b) (dt : (a, b) Nx_dtype.t) : int64 Gen.t =
  let float ~e ~m = Stored.float_bits ~e ~m in
  let int bits signed = int_value ~bits ~signed in
  match dt with
  | Float16 -> float ~e:5 ~m:10
  | BFloat16 -> float ~e:8 ~m:7
  | Float32 -> float ~e:8 ~m:23
  | Float64 -> float ~e:11 ~m:52
  | Float8_e4m3 -> float ~e:4 ~m:3
  | Float8_e5m2 -> float ~e:5 ~m:2
  | Int8 -> int 8 true
  | UInt8 -> int 8 false
  | Int16 -> int 16 true
  | UInt16 -> int 16 false
  | Int32 -> int 32 true
  | UInt32 -> int 32 false
  | Int64 -> int 64 true
  | UInt64 -> int 64 false
  | Bool -> Gen.of_list [ 0L; 1L ]
  | Int4 | UInt4 | Complex64 | Complex128 ->
      invalid_arg "bits_of: no graph holds this dtype"

(* Subnormals of float32 and bfloat16 as signed zeros, for a device that flushes
   them. *)
let flushed (type a b) (dt : (a, b) Nx_dtype.t) bits =
  let flush ~exponent ~sign v =
    if Int64.logand v exponent = 0L then Int64.logand v sign else v
  in
  match dt with
  | Float32 -> Gen.map (flush ~exponent:0x7f80_0000L ~sign:0x8000_0000L) bits
  | BFloat16 -> Gen.map (flush ~exponent:0x7f80L ~sign:0x8000L) bits
  | _ -> bits

(* An array of [dtype] and [shape] under a drawn layout, the elements of its
   buffer drawn from [bits]. The operand of arithmetic, [flush], has no
   subnormal on a device that flushes them; kernels that move or order values
   keep every bit there too. *)
let operand ?broadcast ?bits ?(flush = false) d dtype shape =
  let bits = Option.value bits ~default:(bits_of dtype) in
  let bits = if flush && d.flushes then flushed dtype bits else bits in
  let open Gen in
  with_pp pp_arr
    (let* l = layout ?broadcast shape in
     let+ xs = array ~size:(constant l.length) bits in
     of_bits dtype shape l xs)

(* The host value [x] under a drawn layout without broadcast, the elements its
   view does not reach poisoned with bytes [0x7f]. *)
let laid_out (x : ('a, 'b) Nx.t) =
  let dtype = Nx.dtype x and shape = Nx.shape x in
  Gen.with_pp pp_arr
    (Gen.map
       (fun l ->
         let a =
           of_bits dtype shape l (Array.make l.length 0x7f7f_7f7f_7f7f_7f7fL)
         in
         let get = E.get dtype (elements x) and set = E.set dtype a.buffer in
         for k = 0 to numel shape - 1 do
           let p = ref l.offset in
           Array.iteri
             (fun d i -> p := !p + (i * l.strides.(d)))
             (unravel shape k);
           set !p (get k)
         done;
         a)
       (layout ~broadcast:false shape))

(* The element of [dt] whose bits are [v]. *)
let element_of_bits (type a b) (dt : (a, b) Nx_dtype.t) v : a =
  let one = buffer host dt 1 in
  store dt one 0 v;
  E.get dt one 0

(* The bits of the float [x] rounded to [dt]. *)
let bits_of_float (type b) (dt : (float, b) Nx_dtype.t) x =
  let one = buffer host dt 1 in
  E.set dt one 0 x;
  let get dt' conv =
    conv (E.get dt' (B.view one ~offset:0 (Scalar.of_dtype dt') 1) 0)
  in
  match Nx_dtype.itemsize dt with
  | 2 -> get Nx.uint16 Int64.of_int
  | 4 -> get Nx.uint32 (fun w -> Int64.logand (Int64.of_int32 w) 0xffff_ffffL)
  | _ -> get Nx.uint64 Fun.id

(* Values whose sums and products stay far from overflow at every width, and
   both zeros. *)
let moderate dt =
  Gen.map (bits_of_float dt)
    (Gen.frequency
       [
         (6, Gen.float_range (-2.) 2.);
         ( 1,
           Gen.of_list ~pp:(fun ppf x -> Format.fprintf ppf "%h" x) [ 0.; -0. ]
         );
       ])

(* Shapes *)

let shapes ?(rank = Gen.int_range 0 3) ?(dim = Gen.int_range 0 4) () =
  Gen.bind rank (fun r -> Gen.array ~size:(Gen.constant r) dim)

let shape = shapes ()
let ranked = shapes ~rank:(Gen.int_range 1 3) ()
let nonempty = shapes ~rank:(Gen.int_range 1 3) ~dim:(Gen.int_range 1 4) ()
let axis_of s = Gen.int_range 0 (Array.length s - 1)

let axes_of s =
  Gen.map Array.of_list (Gen.subsequence (List.init (Array.length s) Fun.id))

let without axes s =
  Array.of_list
    (List.filteri (fun i _ -> not (Array.mem i axes)) (Array.to_list s))

(* One draw of each generator of [gs]. *)
let each gs =
  Gen.map Array.of_list
    (Array.fold_right
       (fun g acc -> Gen.map (fun (x, xs) -> x :: xs) (Gen.pair g acc))
       gs (Gen.constant []))

(* Dtypes *)

type dt = D : ('a, 'b) Nx_dtype.t -> dt
type fdt = F : (float, 'b) Nx_dtype.t -> fdt

let pp_dt ppf (D dt) = Format.pp_print_string ppf (Nx_dtype.to_string dt)
let pp_fdt ppf (F dt) = Format.pp_print_string ppf (Nx_dtype.to_string dt)

(* The floats the device's renderer computes: the host's has no 8-bit float,
   Metal's no float64. *)
let floats d =
  [ F Nx.float32; F Nx.float16; F Nx.bfloat16 ]
  @ if d.float64 then [ F Nx.float64 ] else []

let ints =
  [
    D Nx.int8;
    D Nx.uint8;
    D Nx.int16;
    D Nx.uint16;
    D Nx.int32;
    D Nx.uint32;
    D Nx.int64;
    D Nx.uint64;
  ]

let as_dt (F dt) = D dt
let numeric d = ints @ List.map as_dt (floats d)
let every d = (D Nx.bool :: ints) @ List.map as_dt (floats d)

(* Checks: a drawn case, which prints what it computes on, and its assertion. *)
type check = Check of (Format.formatter -> unit) * (unit -> unit)

let check pps f =
  Check
    ( (fun ppf ->
        Format.pp_print_list ~pp_sep:Format.pp_print_cut
          (fun ppf pp -> pp ppf)
          ppf pps),
      f )

let shown a ppf = pp_arr ppf a
let said fmt = Format.dprintf fmt

type per_dtype = { per : 'a 'b. ('a, 'b) Nx_dtype.t -> check Gen.t }
type per_float = { per_float : 'b. (float, 'b) Nx_dtype.t -> check Gen.t }

let checks g = Gen.with_pp (fun ppf (Check (pp, _)) -> pp ppf) g

(* The checks [per] draws for a dtype drawn from [dtypes]. *)
let over dtypes { per } =
  checks (Gen.bind (Gen.of_list ~pp:pp_dt dtypes) (function D dt -> per dt))

let over_floats dtypes { per_float } =
  checks
    (Gen.bind (Gen.of_list ~pp:pp_fdt dtypes) (function F dt -> per_float dt))

(* A row of a family, printed by its name. *)
let rows l =
  Gen.of_list ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name) l

let law ~count name g = prop ~count name g (fun (Check (_, f)) -> f ())

(* Classes *)

(* Exact: eager's bits, every NaN equal to every NaN. *)
let exact e a = Traces.exact (value e) (back a)
let exact_of (e, a) = exact e a

(* [x] with its float32 and bfloat16 subnormals as signed zeros. *)
let flush_subnormals (type a b) (x : (a, b) Nx.t) : (a, b) Nx.t =
  let flush ~exponent ~sign v =
    if Int64.logand v exponent = 0L then Int64.logand v sign else v
  in
  match Nx.dtype x with
  | Float32 ->
      let bits =
        Array.map
          (fun w ->
            Int64.to_int32
              (flush ~exponent:0x7f80_0000L ~sign:0x8000_0000L
                 (Int64.logand (Int64.of_int32 w) 0xffff_ffffL)))
          (Nx.to_array (Nx.bitcast Nx.uint32 x))
      in
      Nx.bitcast Nx.float32 (Nx.create Nx.uint32 (Nx.shape x) bits)
  | BFloat16 ->
      let bits =
        Array.map
          (fun w ->
            Int64.to_int
              (flush ~exponent:0x7f80L ~sign:0x8000L (Int64.of_int w)))
          (Nx.to_array (Nx.bitcast Nx.uint16 x))
      in
      Nx.bitcast Nx.bfloat16 (Nx.create Nx.uint16 (Nx.shape x) bits)
  | _ -> x

(* Exact on [d]: eager's results with their subnormals flushed where [d]'s
   arithmetic flushes those of its results, as Metal's does. *)
let exact_on d (e, a) =
  let e = value e in
  Traces.exact (if d.flushes then flush_subnormals e else e) (back a)

let f64 x = Nx.to_array (Nx.cast Nx.float64 x)

(* The unit roundoff of a float dtype. *)
let roundoff (type b) (dt : (float, b) Nx_dtype.t) =
  match dt with
  | Float64 -> 0x1p-53
  | Float32 -> 0x1p-24
  | Float16 -> 0x1p-11
  | BFloat16 -> 0x1p-8
  | Float8_e4m3 -> 0x1p-4
  | Float8_e5m2 -> 0x1p-3

(* Rounded sum: [a] is within the error of some association of the terms of each
   of [e]'s elements, at most [terms] terms whose magnitudes add up to
   [magnitude]'s element there, and eager's own error: nx.cpu may accumulate a
   narrow dtype at its own width. A zero has eager's sign, NaN is NaN and an
   infinity the same one. *)
let summed ~terms ~magnitude e a =
  let u = roundoff e.Nx_array.dtype and m = Nx.to_array magnitude in
  let ev = f64 (value e) and av = f64 (back a) in
  Array.iteri
    (fun i x ->
      let y = av.(i) in
      let bound =
        2. *. u
        *. ((float_of_int (Int.max 0 (terms - 1)) *. m.(i))
           +. Float.max (Float.abs x) (Float.abs y))
      in
      let within =
        if Float.is_nan x || Float.is_nan y then
          Float.is_nan x && Float.is_nan y
        else if not (Float.is_finite x && Float.is_finite y) then x = y
        else if x = 0. && y = 0. then Float.sign_bit x = Float.sign_bit y
        else Float.abs (x -. y) <= bound
      in
      if not within then
        failf "element %d: %h, eager %h, beyond %h (%d terms of magnitude %h)" i
          y x bound terms m.(i))
    ev

(* The magnitudes of the terms of [x] that [f] sums, as float64: [f] applied to
   their absolute values. *)
let magnitudes f (x : (float, 'b) arr) =
  value (f (host_array (Nx.abs (Nx.cast Nx.float64 (value x)))))

(* The position of each element of a float among the floats of its format, in
   order: its bits as an integer of its width, a negative float's magnitude
   negated. *)
let ranks (x : (float, 'b) Nx.t) =
  let bits, sign =
    match Nx_dtype.itemsize (Nx.dtype x) with
    | 8 -> (Nx.to_array (Nx.bitcast Nx.int64 x), Int64.min_int)
    | 4 ->
        ( Array.map
            (fun w -> Int64.logand (Int64.of_int32 w) 0xffff_ffffL)
            (Nx.to_array (Nx.bitcast Nx.uint32 x)),
          0x8000_0000L )
    | _ ->
        (Array.map Int64.of_int (Nx.to_array (Nx.bitcast Nx.uint16 x)), 0x8000L)
  in
  let magnitude = Int64.pred sign in
  Array.map
    (fun b ->
      if Int64.logand b sign <> 0L then Int64.neg (Int64.logand b magnitude)
      else b)
    bits

(* Units in the last place on [d]: each element of [a] within [budget] of
   eager's [e], its subnormals flushed where [d] flushes them, computed from
   [inputs] elementwise; NaN must be NaN and an infinity the same one. *)
let ulps d ~budget inputs e a =
  let e = if d.flushes then flush_subnormals (value e) else value e in
  let a = back a in
  let ev = Nx.to_array e and av = Nx.to_array a in
  let er = ranks e and ar = ranks a in
  let distance i =
    let x = ev.(i) and y = av.(i) in
    if Float.is_nan x || Float.is_nan y then
      if Float.is_nan x && Float.is_nan y then 0L else Int64.max_int
    else if not (Float.is_finite x && Float.is_finite y) then
      if x = y then 0L else Int64.max_int
    else Int64.abs (Int64.sub ar.(i) er.(i))
  in
  Array.iteri
    (fun i _ ->
      let d = distance i in
      if Int64.compare d (Int64.of_int budget) > 0 then
        failf "%Ld ulps (budget %d) at element %d of (%a): %h, eager %h" d
          budget i
          (Format.pp_print_list
             ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
             (fun ppf x -> Format.fprintf ppf "%h" (f64 (value x)).(i)))
          inputs av.(i) ev.(i))
    ev

(* A transcendental row's budget against nx.cpu on [d]: the device's measured
   maximum, else [stated], the lowering's budget against the correctly rounded
   result, plus nx.cpu's own ulp. At the narrow dtypes both round a float32
   result once, and may round to either side of it. *)
let budget d name stated (dt : (float, 'b) Nx_dtype.t) =
  if Nx_dtype.itemsize dt < 4 then 2
  else Option.value (List.assoc_opt name d.budgets) ~default:(stated + 1)

(* Elementwise *)

let unary k x (module K : Nx_backend.S) env =
  let dst = env.dst x.Nx_array.dtype (shape_of x) in
  K.unary k (env.on x) ~dst;
  dst

let binary k x y (module K : Nx_backend.S) env =
  let dst = env.dst x.Nx_array.dtype (shape_of x) in
  K.binary k (env.on x) (env.on y) ~dst;
  dst

let elementwise d ~count ~heavy =
  let law = law ~count in
  let floats' = List.map as_dt (floats d) in
  let exact_unary =
    law "exact unary"
      (checks
         (Gen.bind
            (rows
               [
                 ("neg", (Nx_backend.Neg, numeric d));
                 ("abs", (Abs, numeric d));
                 ("sign", (Sign, numeric d));
                 ("trunc", (Trunc, numeric d));
                 ("ceil", (Ceil, numeric d));
                 ("floor", (Floor, numeric d));
                 ("round", (Round, numeric d));
                 ("recip", (Recip, numeric d));
                 ("sqrt", (Sqrt, floats'));
               ])
            (fun (name, (k, dtypes)) ->
              over dtypes
                {
                  per =
                    (fun dt ->
                      let open Gen in
                      let* s = shape in
                      let+ x = operand ~flush:true d dt s in
                      check
                        [ said "%s" name; shown x ]
                        (fun () -> exact_on d (both d (unary k x))));
                })))
  and transcendental =
    law "transcendental unary"
      (checks
         (Gen.bind
            (rows
               [
                 ("exp", (Nx_backend.Exp, 4));
                 ("log", (Log, 4));
                 ("sin", (Sin, 4));
                 ("cos", (Cos, 4));
                 ("tan", (Tan, 8));
                 ("asin", (Asin, 8));
                 ("acos", (Acos, 8));
                 ("atan", (Atan, 8));
                 ("sinh", (Sinh, 8));
                 ("cosh", (Cosh, 8));
                 ("tanh", (Tanh, 8));
                 ("erf", (Erf, 8));
               ])
            (fun (name, (k, stated)) ->
              over_floats (floats d)
                {
                  per_float =
                    (fun dt ->
                      let open Gen in
                      let* s = shape in
                      let+ x = operand ~flush:true d dt s in
                      check
                        [ said "%s" name; shown x ]
                        (fun () ->
                          let e, a = both d (unary k x) in
                          ulps d ~budget:(budget d name stated dt) [ x ] e a));
                })))
  and exact_binary =
    law "exact binary"
      (checks
         (Gen.bind
            (rows
               [
                 ("add", (Nx_backend.Add, numeric d));
                 ("sub", (Sub, numeric d));
                 ("mul", (Mul, numeric d));
                 ("maximum", (Maximum, numeric d));
                 ("minimum", (Minimum, numeric d));
                 ("mod", (Mod, numeric d));
                 ("integer division", (Idiv, ints));
                 ("integer power", (Pow, ints));
                 ("division", (Fdiv, floats'));
                 ("and", (And, D Nx.bool :: ints));
                 ("or", (Or, D Nx.bool :: ints));
                 ("xor", (Xor, D Nx.bool :: ints));
               ])
            (fun (name, (k, dtypes)) ->
              over dtypes
                {
                  per =
                    (fun dt ->
                      let open Gen in
                      let* s = shape in
                      let+ x = operand ~flush:true d dt s
                      and+ y = operand ~flush:true d dt s in
                      check
                        [ said "%s" name; shown x; shown y ]
                        (fun () -> exact_on d (both d (binary k x y))));
                })))
  and transcendental_binary =
    law "transcendental binary"
      (checks
         (Gen.bind
            (rows [ ("pow", (Nx_backend.Pow, 16)); ("atan2", (Atan2, 8)) ])
            (fun (name, (k, stated)) ->
              over_floats (floats d)
                {
                  per_float =
                    (fun dt ->
                      let open Gen in
                      let* s = shape in
                      let+ x = operand ~flush:true d dt s
                      and+ y = operand ~flush:true d dt s in
                      check
                        [ said "%s" name; shown x; shown y ]
                        (fun () ->
                          let e, a = both d (binary k x y) in
                          ulps d ~budget:(budget d name stated dt) [ x; y ] e a));
                })))
  and comparisons =
    law "comparisons"
      (checks
         (Gen.bind
            (rows
               [
                 ("equal", Nx_backend.Equal);
                 ("not_equal", Not_equal);
                 ("less", Less);
                 ("less_equal", Less_equal);
               ])
            (fun (name, k) ->
              over (every d)
                {
                  per =
                    (fun dt ->
                      let open Gen in
                      let* s = shape in
                      let+ x = operand ~flush:true d dt s
                      and+ y = operand ~flush:true d dt s in
                      check
                        [ said "%s" name; shown x; shown y ]
                        (fun () ->
                          exact_of
                            (both d (fun (module K : Nx_backend.S) env ->
                                 let dst = env.dst Nx.bool s in
                                 K.compare k (env.on x) (env.on y) ~dst;
                                 dst))));
                })))
  and where =
    law "where"
      (over (every d)
         {
           per =
             (fun dt ->
               let open Gen in
               let* s = shape in
               let+ c = operand d Nx.bool s
               and+ x = operand d dt s
               and+ y = operand d dt s in
               check
                 [ shown c; shown x; shown y ]
                 (fun () ->
                   exact_of
                     (both d (fun (module K : Nx_backend.S) env ->
                          let dst = env.dst dt s in
                          K.where (env.on c) (env.on x) (env.on y) ~dst;
                          dst))));
         })
  and cast =
    law "cast"
      (over (every d)
         {
           per =
             (fun src ->
               Gen.bind
                 (Gen.of_list ~pp:pp_dt (every d))
                 (function
                   | D dst ->
                   let open Gen in
                   let* s = shape in
                   let+ x = operand ~flush:true d src s in
                   check
                     [ said "to %s" (Nx_dtype.to_string dst); shown x ]
                     (fun () ->
                       exact_on d
                         (both d (fun (module K : Nx_backend.S) env ->
                              let y = env.dst dst s in
                              K.cast (env.on x) ~dst:y;
                              y)))));
         })
  and threefry =
    law "threefry"
      (checks
         (let open Gen in
          let* s = shapes ~rank:(int_range 0 2) () in
          let s = Array.append s [| 2 |] in
          let+ key = operand d Nx.int32 s and+ counter = operand d Nx.int32 s in
          check
            [ shown key; shown counter ]
            (fun () ->
              exact_of
                (both d (fun (module K : Nx_backend.S) env ->
                     let dst = env.dst Nx.int32 s in
                     K.threefry (env.on key) (env.on counter) ~dst;
                     dst)))))
  in
  [ exact_unary; exact_binary; comparisons; where; cast; threefry ]
  @ if heavy then [ transcendental; transcendental_binary ] else []

(* Reductions, scans and sorts *)

let reduce k axes x (module K : Nx_backend.S) env =
  let dst = env.dst x.Nx_array.dtype (without axes (shape_of x)) in
  K.reduce k ~axes (env.on x) ~dst;
  dst

let scan k axis x (module K : Nx_backend.S) env =
  let dst = env.dst x.Nx_array.dtype (shape_of x) in
  K.scan k ~axis (env.on x) ~dst;
  dst

(* The magnitudes that bound the rounding of a float sum or product: those of
   the terms added up for a sum, which [sum] gives, and the result's own for a
   product, whose error is relative to it. *)
let magnitude k ~sum e =
  match k with
  | Nx_backend.Sum -> sum ()
  | Prod | Max | Min -> Nx.abs (Nx.cast Nx.float64 (value e))

let pp_axes axes = said "axes %a" pp_ints axes

(* A law of one operand along one of its axes, ascending or not. *)
type along = {
  along : 'a 'b. axis:int -> descending:bool -> ('a, 'b) arr -> unit;
}

let reductions d ~count ~heavy =
  let law = law ~count in
  let exactly =
    [
      ("max", (Nx_backend.Max, numeric d));
      ("min", (Min, numeric d));
      ("integer sum", (Sum, ints));
      ("integer product", (Prod, ints));
    ]
  in
  let rounded = [ ("sum", Nx_backend.Sum); ("product", Prod) ] in
  let reduce_exactly =
    law "reduce exactly"
      (checks
         (Gen.bind (rows exactly) (fun (name, (k, dtypes)) ->
              over dtypes
                {
                  per =
                    (fun dt ->
                      let open Gen in
                      (* The reduced axes of Max and Min are not empty. *)
                      let* s = if k = Max || k = Min then nonempty else shape in
                      let+ axes = axes_of s and+ x = operand d dt s in
                      check
                        [ said "%s" name; pp_axes axes; shown x ]
                        (fun () -> exact_of (both d (reduce k axes x))));
                })))
  and reduce_floats =
    law "reduce floats"
      (checks
         (Gen.bind (rows rounded) (fun (name, k) ->
              over_floats (floats d)
                {
                  per_float =
                    (fun dt ->
                      let open Gen in
                      let* s = shape in
                      let+ axes = axes_of s
                      and+ x = operand ~bits:(moderate dt) d dt s in
                      check
                        [ said "%s" name; pp_axes axes; shown x ]
                        (fun () ->
                          let e, a = both d (reduce k axes x) in
                          let sum () =
                            magnitudes (fun m -> reduce Sum axes m cpu on_cpu) x
                          in
                          let terms =
                            Array.fold_left (fun n i -> n * s.(i)) 1 axes
                          in
                          summed ~terms ~magnitude:(magnitude k ~sum e) e a));
                })))
  and scan_exactly =
    law "scan exactly"
      (checks
         (Gen.bind (rows exactly) (fun (name, (k, dtypes)) ->
              over dtypes
                {
                  per =
                    (fun dt ->
                      let open Gen in
                      let* s = ranked in
                      let+ axis = axis_of s and+ x = operand d dt s in
                      check
                        [ said "%s" name; pp_axes [| axis |]; shown x ]
                        (fun () -> exact_of (both d (scan k axis x))));
                })))
  and scan_floats =
    law "scan floats"
      (checks
         (Gen.bind (rows rounded) (fun (name, k) ->
              over_floats (floats d)
                {
                  per_float =
                    (fun dt ->
                      let open Gen in
                      let* s = ranked in
                      let+ axis = axis_of s
                      and+ x = operand ~bits:(moderate dt) d dt s in
                      check
                        [ said "%s" name; pp_axes [| axis |]; shown x ]
                        (fun () ->
                          let e, a = both d (scan k axis x) in
                          let sum () =
                            magnitudes (fun m -> scan Sum axis m cpu on_cpu) x
                          in
                          summed ~terms:s.(axis) ~magnitude:(magnitude k ~sum e)
                            e a));
                })))
  in
  (* A kernel along one axis of a non-empty operand, ascending or not. *)
  let along name { along } =
    law name
      (over (numeric d)
         {
           per =
             (fun dt ->
               let open Gen in
               let* s = nonempty in
               let+ axis = axis_of s
               and+ descending = bool
               and+ x = operand d dt s in
               check
                 [
                   pp_axes [| axis |]; said "descending %b" descending; shown x;
                 ]
                 (fun () -> along ~axis ~descending x));
         })
  in
  [
    reduce_exactly;
    reduce_floats;
    scan_exactly;
    scan_floats;
    along "argmax and argmin"
      {
        along =
          (fun ~axis ~descending x ->
            exact_of
              (both d (fun (module K : Nx_backend.S) env ->
                   let dst =
                     env.dst Nx.int32 (without [| axis |] (shape_of x))
                   in
                   K.arg_reduce
                     (if descending then Argmax else Argmin)
                     ~axis (env.on x) ~dst;
                   dst)));
      };
  ]
  @
  if heavy then
    [
      along "sort"
        {
          along =
            (fun ~axis ~descending x ->
              exact_of
                (both d (fun (module K : Nx_backend.S) env ->
                     let dst = env.dst x.Nx_array.dtype (shape_of x) in
                     K.sort ~descending ~axis (env.on x) ~dst;
                     dst)));
        };
      along "argsort"
        {
          along =
            (fun ~axis ~descending x ->
              exact_of
                (both d (fun (module K : Nx_backend.S) env ->
                     let dst = env.dst Nx.int32 (shape_of x) in
                     K.argsort ~descending ~axis (env.on x) ~dst;
                     dst)));
        };
    ]
  else []

(* Assembly and indexed access *)

(* Indices along an axis of [n] elements, of [shape]: in range, just out of it,
   and the extremes of int32. *)
let indices ~n shape =
  let open Gen in
  let index =
    frequency
      [
        (8, int_range (-2) (n + 1));
        ( 1,
          of_list ~pp:Format.pp_print_int
            [ Int32.to_int Int32.min_int; Int32.to_int Int32.max_int ] );
      ]
  in
  bind
    (map
       (fun xs -> Nx.create Nx.int32 shape (Array.map Int32.of_int xs))
       (array ~size:(constant (numel shape)) index))
    laid_out

(* Positions along [axis] of a scatter of shape [is] into an axis of [n]:
   distinct along the axis for [unique], any otherwise. *)
let positions ~unique ~axis ~n is =
  if unique then
    Gen.bind
      (Gen.map
         (fun order ->
           let order = Array.of_list order in
           Nx.create Nx.int32 is
             (Array.init (numel is) (fun k ->
                  Int32.of_int order.((unravel is k).(axis)))))
         (Gen.permutation (List.init n Fun.id)))
      laid_out
  else indices ~n is

(* A shape, an axis of it, [unique], and the shape of the indices and updates of
   a scatter along the axis. *)
let scatters =
  let open Gen in
  let* s = ranked in
  let* axis = axis_of s in
  let* unique = bool in
  let+ m = int_range 0 (if unique then s.(axis) else 4) in
  let is = Array.copy s in
  is.(axis) <- m;
  (s, axis, unique, is)

let scatter ~mode ~unique ~axis ~indices ~updates x (module K : Nx_backend.S)
    env =
  let dst = env.dst x.Nx_array.dtype (shape_of x) in
  K.scatter ~mode ~unique ~axis ~indices:(env.on indices)
    ~updates:(env.on updates) (env.on x) ~dst;
  dst

let indexed d ~count =
  let law = law ~count in
  let pad =
    law "pad"
      (over (every d)
         {
           per =
             (fun dt ->
               let open Gen in
               let* s = shape in
               let+ padding =
                 array
                   ~size:(constant (Array.length s))
                   (pair (int_range 0 2) (int_range 0 2))
               and+ fill = bits_of dt
               and+ x = operand d dt s in
               check
                 [ said "padding %a fill 0x%Lx" pp_pairs padding fill; shown x ]
                 (fun () ->
                   let v = element_of_bits dt fill in
                   let out =
                     Array.mapi (fun i (b, a) -> s.(i) + b + a) padding
                   in
                   exact_of
                     (both d (fun (module K : Nx_backend.S) env ->
                          let dst = env.dst dt out in
                          K.pad padding v (env.on x) ~dst;
                          dst))));
         })
  and cat =
    law "cat"
      (over (every d)
         {
           per =
             (fun dt ->
               let open Gen in
               let* s = ranked in
               let* axis = axis_of s in
               let piece =
                 bind (int_range 0 3) (fun n ->
                     let s = Array.copy s in
                     s.(axis) <- n;
                     operand d dt s)
               in
               let+ xs = list ~size:(int_range 1 3) piece in
               check (pp_axes [| axis |] :: List.map shown xs) (fun () ->
                   let out = Array.copy s in
                   out.(axis) <-
                     List.fold_left (fun n x -> n + (shape_of x).(axis)) 0 xs;
                   exact_of
                     (both d (fun (module K : Nx_backend.S) env ->
                          let dst = env.dst dt out in
                          K.cat ~axis (List.map env.on xs) ~dst;
                          dst))));
         })
  and contiguous =
    law "contiguous"
      (over (every d)
         {
           per =
             (fun dt ->
               let open Gen in
               let* s = shape in
               let+ x = operand d dt s in
               check
                 [ shown x ]
                 (fun () ->
                   exact_of
                     (both d (fun (module K : Nx_backend.S) env ->
                          let dst = env.dst dt s in
                          K.contiguous (env.on x) ~dst;
                          dst))));
         })
  and gather =
    law "gather"
      (over (every d)
         {
           per =
             (fun dt ->
               let open Gen in
               let* s = ranked in
               let* axis = axis_of s in
               let* m = int_range 0 4 in
               let is = Array.copy s in
               is.(axis) <- m;
               let+ indices = indices ~n:s.(axis) is and+ x = operand d dt s in
               check
                 [ pp_axes [| axis |]; shown indices; shown x ]
                 (fun () ->
                   exact_of
                     (both d (fun (module K : Nx_backend.S) env ->
                          let dst = env.dst dt is in
                          K.gather ~axis (env.on indices) (env.on x) ~dst;
                          dst))));
         })
  and scatter_exactly =
    law "scatter exactly"
      (checks
         (Gen.bind
            (rows [ ("set", (`Set, every d)); ("integer add", (`Add, ints)) ])
            (fun (name, (mode, dtypes)) ->
              over dtypes
                {
                  per =
                    (fun dt ->
                      let open Gen in
                      let* s, axis, unique, is = scatters in
                      let+ indices = positions ~unique ~axis ~n:s.(axis) is
                      and+ updates = operand d dt is
                      and+ x = operand d dt s in
                      check
                        [
                          said "%s along %d, unique %b" name axis unique;
                          shown indices;
                          shown updates;
                          shown x;
                        ]
                        (fun () ->
                          exact_of
                            (both d
                               (scatter ~mode ~unique ~axis ~indices ~updates x))));
                })))
  and scatter_floats =
    law "scatter add of floats"
      (over_floats (floats d)
         {
           per_float =
             (fun dt ->
               let open Gen in
               let* s, axis, unique, is = scatters in
               let+ indices = positions ~unique ~axis ~n:s.(axis) is
               and+ updates = operand ~bits:(moderate dt) d dt is
               and+ x = operand ~bits:(moderate dt) d dt s in
               check
                 [
                   said "along %d, unique %b" axis unique;
                   shown indices;
                   shown updates;
                   shown x;
                 ]
                 (fun () ->
                   let e, a =
                     both d
                       (scatter ~mode:`Add ~unique ~axis ~indices ~updates x)
                   in
                   let magnitude =
                     magnitudes
                       (fun mx ->
                         let mu =
                           host_array
                             (Nx.abs (Nx.cast Nx.float64 (value updates)))
                         in
                         scatter ~mode:`Add ~unique ~axis ~indices ~updates:mu
                           mx cpu on_cpu)
                       x
                   in
                   summed ~terms:(is.(axis) + 1) ~magnitude e a));
         })
  and update =
    law "update"
      (over (every d)
         {
           per =
             (fun dt ->
               let open Gen in
               let* s = shape in
               let* vs = each (Array.map (fun n -> int_range 0 n) s) in
               let* corner =
                 each (Array.mapi (fun i v -> int_range 0 (s.(i) - v)) vs)
               in
               let+ starts =
                 laid_out
                   (Nx.create Nx.int32
                      [| Array.length s |]
                      (Array.map Int32.of_int corner))
               and+ v = operand d dt vs
               and+ x = operand d dt s in
               check
                 [ shown starts; shown v; shown x ]
                 (fun () ->
                   exact_of
                     (both d (fun (module K : Nx_backend.S) env ->
                          let dst = env.dst dt s in
                          K.update (env.on x) ~starts:(env.on starts) (env.on v)
                            ~dst;
                          dst))));
         })
  in
  [ pad; cat; contiguous; gather; scatter_exactly; scatter_floats; update ]

(* Windows and products *)

(* The parameters of an unfold of [leading @ spatial], and of the fold back. *)
type windows = {
  leading : int array;
  kernel_size : int array;
  stride : int array;
  dilation : int array;
  padding : (int * int) array;
  spatial : int array;
  count : int;
}

let pp_windows w =
  said "leading %a kernel %a stride %a dilation %a padding %a spatial %a"
    pp_ints w.leading pp_ints w.kernel_size pp_ints w.stride pp_ints w.dilation
    pp_pairs w.padding pp_ints w.spatial

(* One or two spatial axes, each at least as long as a window's reach less its
   padding. *)
let windows =
  let open Gen in
  let* k = int_range 1 2 in
  let per g = each (Array.make k g) in
  let* leading = shapes ~rank:(int_range 0 2) ~dim:(int_range 0 2) () in
  let* kernel_size = per (int_range 1 3) in
  let* stride = per (int_range 1 2) in
  let* dilation = per (int_range 1 2) in
  let* padding = per (pair (int_range 0 1) (int_range 0 1)) in
  let+ extra = per (int_range 0 2) in
  let reach i = (dilation.(i) * (kernel_size.(i) - 1)) + 1 in
  let spatial =
    Array.init k (fun i ->
        let b, a = padding.(i) in
        Int.max 1 (reach i - b - a) + extra.(i))
  in
  let count =
    numel
      (Array.init k (fun i ->
           let b, a = padding.(i) in
           ((spatial.(i) + b + a - reach i) / stride.(i)) + 1))
  in
  { leading; kernel_size; stride; dilation; padding; spatial; count }

let patches w = Array.append w.leading [| numel w.kernel_size; w.count |]
let image w = Array.append w.leading w.spatial

let unfold w x (module K : Nx_backend.S) env =
  let dst = env.dst x.Nx_array.dtype (patches w) in
  K.unfold ~kernel_size:w.kernel_size ~stride:w.stride ~dilation:w.dilation
    ~padding:w.padding (env.on x) ~dst;
  dst

let fold w x (module K : Nx_backend.S) env =
  let dst = env.dst x.Nx_array.dtype (image w) in
  K.fold ~output_size:w.spatial ~kernel_size:w.kernel_size ~stride:w.stride
    ~dilation:w.dilation ~padding:w.padding (env.on x) ~dst;
  dst

(* The shapes of a product and of its result, the batch axes broadcast between
   the operands. *)
let products =
  let open Gen in
  let* batch = shapes ~rank:(int_range 0 2) ~dim:(int_range 1 2) () in
  let one_or n = of_list ~pp:Format.pp_print_int [ n; 1 ] in
  let* ba = each (Array.map one_or batch) in
  let* bb = each (Array.map one_or batch) in
  let+ m = int_range 0 3 and+ k = int_range 0 4 and+ n = int_range 0 3 in
  ( Array.append ba [| m; k |],
    Array.append bb [| k; n |],
    Array.append (Array.map2 Int.max ba bb) [| m; n |] )

let matmul out x y (module K : Nx_backend.S) env =
  let dst = env.dst x.Nx_array.dtype out in
  K.matmul (env.on x) (env.on y) ~dst;
  dst

let windows_and_products d ~count =
  let law = law ~count in
  let unfold =
    law "unfold"
      (over (every d)
         {
           per =
             (fun dt ->
               let open Gen in
               let* w = windows in
               let+ x = operand d dt (image w) in
               check
                 [ pp_windows w; shown x ]
                 (fun () -> exact_of (both d (unfold w x))));
         })
  and integer_fold =
    law "fold of integers"
      (over ints
         {
           per =
             (fun dt ->
               let open Gen in
               let* w = windows in
               let+ x = operand d dt (patches w) in
               check
                 [ pp_windows w; shown x ]
                 (fun () -> exact_of (both d (fold w x))));
         })
  and float_fold =
    law "fold of floats"
      (over_floats (floats d)
         {
           per_float =
             (fun dt ->
               let open Gen in
               let* w = windows in
               let+ x = operand ~bits:(moderate dt) d dt (patches w) in
               check
                 [ pp_windows w; shown x ]
                 (fun () ->
                   let e, a = both d (fold w x) in
                   let magnitude =
                     magnitudes (fun m -> fold w m cpu on_cpu) x
                   in
                   summed ~terms:(numel w.kernel_size) ~magnitude e a));
         })
  and integer_matmul =
    law "integer products"
      (over ints
         {
           per =
             (fun dt ->
               let open Gen in
               let* sa, sb, out = products in
               let+ x = operand d dt sa and+ y = operand d dt sb in
               check
                 [ shown x; shown y ]
                 (fun () -> exact_of (both d (matmul out x y))));
         })
  and float_matmul =
    law "float products"
      (over_floats (floats d)
         {
           per_float =
             (fun dt ->
               let open Gen in
               let* sa, sb, out = products in
               let+ x = operand ~bits:(moderate dt) d dt sa
               and+ y = operand ~bits:(moderate dt) d dt sb in
               check
                 [ shown x; shown y ]
                 (fun () ->
                   let e, a = both d (matmul out x y) in
                   let magnitude =
                     magnitudes
                       (fun mx ->
                         let my =
                           host_array (Nx.abs (Nx.cast Nx.float64 (value y)))
                         in
                         matmul out mx my cpu on_cpu)
                       x
                   in
                   summed ~terms:sa.(Array.length sa - 1) ~magnitude e a));
         })
  in
  [ unfold; integer_fold; float_fold; integer_matmul; float_matmul ]

(* Linear algebra *)

(* The dtypes of the factorizations, as the lowering's suite measures them:
   float16 at float32, rounded once. *)
let factor_dtypes d =
  [ F Nx.float32; F Nx.float16 ] @ if d.float64 then [ F Nx.float64 ] else []

let largest x =
  Array.fold_left
    (fun m v -> if Float.is_finite v then Float.max m (Float.abs v) else m)
    0. (f64 x)

(* Each element of [actual] within [limit] of [expected]'s, NaN where that is
   NaN and the same infinity where that is one. *)
let near ~limit expected actual =
  if Nx.shape expected <> Nx.shape actual then
    failf "shape %a, eager %a" pp_ints (Nx.shape actual) pp_ints
      (Nx.shape expected);
  let e = f64 expected and a = f64 actual in
  Array.iteri
    (fun i ei ->
      let ai = a.(i) in
      let within =
        if Float.is_nan ei then Float.is_nan ai
        else if Float.is_finite ei then Float.abs (ai -. ei) <= limit
        else Float.equal ai ei
      in
      if not within then
        failf "element %d: %h, eager %h, beyond %h" i ai ei limit)
    e

(* [near] within [bound] times the largest element of [expected]. *)
let relative ~bound expected actual =
  near ~limit:(bound *. largest expected) expected actual

let to64 x = Nx.cast Nx.float64 x

(* The columns of [q] are orthonormal, within [limit]. *)
let orthonormal ~limit q =
  let q = to64 q and k = Nx.dim (-1) q in
  near ~limit
    (Nx.broadcast_to
       (Array.append (Array.sub (Nx.shape q) 0 (Nx.ndim q - 2)) [| k; k |])
       (Nx.eye Nx.float64 k))
    (Nx.matmul (Nx.matrix_transpose q) q)

(* The first [k] elements of [x] along its [axis]. *)
let first k axis x =
  let axis = Nx.ndim x + axis in
  Nx.shrink
    (Array.mapi (fun i n -> if i = axis then (0, k) else (0, n)) (Nx.shape x))
    x

(* Elements of at most one in magnitude whose squares are normal floats in every
   dtype, or zero. *)
let unit_value =
  Gen.map
    (fun x -> if Float.abs x < 0x1p-6 then 0. else x)
    (Gen.float_range (-1.) 1.)

let matrix dtype shape =
  Gen.map
    (fun xs -> Nx.cast dtype (Nx.create Nx.float64 shape xs))
    (Gen.array ~size:(Gen.constant (numel shape)) unit_value)

(* [x] with one more than its larger size added on the leading diagonal of its
   matrices: of full rank, well-conditioned and diagonally dominant. *)
let conditioned x =
  let m = Nx.dim (-2) x and n = Nx.dim (-1) x in
  Nx.add x
    (Nx.mul_s (Nx.eye ~m:n (Nx.dtype x) m) (float_of_int (Int.max m n + 1)))

let batch = Gen.of_list ~pp:pp_ints [ [||]; [| 2 |] ]
let size = Gen.int_range 1 5

let linalg d ~count =
  let law = law ~count in
  let cholesky =
    law "cholesky"
      (over_floats (factor_dtypes d)
         {
           per_float =
             (fun dt ->
               let open Gen in
               let* b = batch in
               let* n = size in
               let s = Array.append b [| n; n |] in
               let* x = matrix dt s in
               let+ upper = bool
               and+ a =
                 laid_out (conditioned (Nx.matmul x (Nx.matrix_transpose x)))
               in
               check
                 [ said "upper %b" upper; shown a ]
                 (fun () ->
                   let e, r =
                     both d (fun (module K : Nx_backend.S) env ->
                         let dst = env.dst dt s in
                         K.cholesky ~upper (env.on a) ~dst;
                         dst)
                   in
                   relative
                     ~bound:(4. *. float_of_int n *. roundoff dt)
                     (value e) (back r)));
         })
  and qr =
    law "qr"
      (over_floats (factor_dtypes d)
         {
           per_float =
             (fun dt ->
               let open Gen in
               let* b = batch in
               let* m = size in
               let* n = size in
               let* x = matrix dt (Array.append b [| m; n |]) in
               let+ reduced = bool and+ a = laid_out (conditioned x) in
               check
                 [ said "reduced %b" reduced; shown a ]
                 (fun () ->
                   let k = Int.min m n in
                   let rows = if reduced then k else m in
                   let (qe, re), (qa, ra) =
                     both d (fun (module K : Nx_backend.S) env ->
                         let q = env.dst dt (Array.append b [| m; rows |]) in
                         let r = env.dst dt (Array.append b [| rows; n |]) in
                         K.qr ~reduced (env.on a) ~q ~r;
                         (q, r))
                   in
                   let qe = to64 (value qe) and re = to64 (value re) in
                   let qa = to64 (back qa) and ra = to64 (back ra) in
                   (* The signs that eager's and the compiled diagonal of [r]
                      differ by, one where either is zero or past [k]. *)
                   let s =
                     let diagonal r = Nx.diagonal ~axis1:(-2) ~axis2:(-1) r in
                     let s = Nx.mul (diagonal re) (diagonal ra) in
                     let s =
                       Nx.where (Nx.less_s s 0.) (Nx.full_like s (-1.))
                         (Nx.full_like s 1.)
                     in
                     Nx.pad
                       (Array.init (Nx.ndim s) (fun i ->
                            if i = Nx.ndim s - 1 then (0, rows - k) else (0, 0)))
                       1. s
                   in
                   let bound =
                     32. *. float_of_int (Int.max m n) *. roundoff dt
                   in
                   let limit = bound *. Float.max (largest qe) (largest re) in
                   near ~limit (first k (-1) qe)
                     (Nx.mul (first k (-1) qa)
                        (Nx.unsqueeze ~axes:[ -2 ] (first k (-1) s)));
                   near ~limit re (Nx.mul ra (Nx.unsqueeze ~axes:[ -1 ] s));
                   orthonormal ~limit:bound qa));
         })
  and lu =
    law "lu"
      (over_floats (factor_dtypes d)
         {
           per_float =
             (fun dt ->
               let open Gen in
               let* b = batch in
               let* m = size in
               let* n = size in
               let* x = matrix dt (Array.append b [| m; n |]) in
               (* Rows shuffled, so that each step pivots. *)
               let* order = permutation (List.init m Fun.id) in
               let x =
                 Nx.take ~axis:(-2)
                   ~indices:
                     (Nx.create Nx.int32 [| m |]
                        (Array.of_list (List.map Int32.of_int order)))
                   (conditioned x)
               in
               let+ a = laid_out x in
               check
                 [ shown a ]
                 (fun () ->
                   let k = Int.min m n in
                   let (le, pe, oe), (la, pa, oa) =
                     both d (fun (module K : Nx_backend.S) env ->
                         let lu = env.dst dt (Array.append b [| m; n |]) in
                         let pivots =
                           env.dst Nx.int32 (Array.append b [| k |])
                         in
                         let perm = env.dst Nx.int32 (Array.append b [| m |]) in
                         K.lu (env.on a) ~lu ~pivots ~perm;
                         (lu, pivots, perm))
                   in
                   exact pe pa;
                   exact oe oa;
                   relative
                     ~bound:(4. *. float_of_int (Int.max m n) *. roundoff dt)
                     (value le) (back la)));
         })
  and svd =
    law "svd"
      (over_floats (factor_dtypes d)
         {
           per_float =
             (fun dt ->
               let open Gen in
               let* b = batch in
               let* m = int_range 1 4 in
               let* n = int_range 1 4 in
               let* x = matrix dt (Array.append b [| m; n |]) in
               let x = conditioned x in
               let+ full = bool and+ a = laid_out x in
               check
                 [ said "full %b" full; shown a ]
                 (fun () ->
                   let k = Int.min m n in
                   let (_, se, _), (ua, sa, vta) =
                     both d (fun (module K : Nx_backend.S) env ->
                         let u =
                           env.dst dt
                             (Array.append b [| m; (if full then m else k) |])
                         in
                         let s = env.dst Nx.float64 (Array.append b [| k |]) in
                         let vt =
                           env.dst dt
                             (Array.append b [| (if full then n else k); n |])
                         in
                         K.svd (env.on a) ~u ~s ~vt;
                         (u, s, vt))
                   in
                   let bound =
                     32. *. float_of_int (Int.max m n) *. roundoff dt
                   in
                   let se = value se and sa = back sa in
                   relative ~bound se sa;
                   let ua = to64 (back ua) and vta = to64 (back vta) in
                   orthonormal ~limit:bound ua;
                   orthonormal ~limit:bound (Nx.matrix_transpose vta);
                   near
                     ~limit:(bound *. largest se)
                     (to64 x)
                     (Nx.matmul
                        (Nx.mul (first k (-1) ua)
                           (Nx.unsqueeze ~axes:[ -2 ] sa))
                        (first k (-2) vta))));
         })
  and solve =
    law "solve_triangular"
      (over_floats (factor_dtypes d)
         {
           per_float =
             (fun dt ->
               let open Gen in
               let* b = batch in
               let* n = size in
               let* rhs =
                 of_list ~pp:pp_ints [ [| n |]; [| n; 1 |]; [| n; 3 |] ]
               in
               let* x = matrix dt (Array.append b [| n; n |]) in
               let* y = matrix dt (Array.append b rhs) in
               (* Dominant with its diagonal and with a unit one. *)
               let x = conditioned (Nx.div_s x (float_of_int (n + 1))) in
               let+ upper = bool
               and+ transpose = bool
               and+ unit_diag = bool
               and+ a = laid_out x
               and+ y = laid_out y in
               check
                 [
                   said "upper %b transpose %b unit_diag %b" upper transpose
                     unit_diag;
                   shown a;
                   shown y;
                 ]
                 (fun () ->
                   let e, r =
                     both d (fun (module K : Nx_backend.S) env ->
                         let dst = env.dst dt (shape_of y) in
                         K.solve_triangular ~upper ~transpose ~unit_diag
                           (env.on a) (env.on y) ~dst;
                         dst)
                   in
                   relative
                     ~bound:(4. *. float_of_int n *. roundoff dt)
                     (value e) (back r)));
         })
  and never_raises =
    law "raises nothing on any matrix"
      (over_floats (factor_dtypes d)
         {
           per_float =
             (fun dt ->
               let open Gen in
               let* n = size in
               let s = [| n; n |] in
               let+ x = operand ~flush:true d dt s
               and+ y = operand ~flush:true d dt [| n |] in
               check
                 [ shown x; shown y ]
                 (fun () ->
                   let module K = (val compiled) in
                   let { on; dst } = on_device d in
                   K.cholesky ~upper:false (on x) ~dst:(dst dt s);
                   K.solve_triangular ~upper:true ~transpose:false
                     ~unit_diag:false (on x) (on y) ~dst:(dst dt [| n |]);
                   K.qr ~reduced:true (on x) ~q:(dst dt s) ~r:(dst dt s);
                   K.lu (on x) ~lu:(dst dt s) ~pivots:(dst Nx.int32 [| n |])
                     ~perm:(dst Nx.int32 [| n |]);
                   if d.float64 then
                     K.svd (on x) ~u:(dst dt s) ~s:(dst Nx.float64 [| n |])
                       ~vt:(dst dt s)));
         })
  in
  [ cholesky; qr; lu ]
  @ (if d.float64 then [ svd ] else [])
  @ [ solve; never_raises ]

(* Where nx.cpu raises Linalg_error, the compiled programs run their fixed steps
   and give the non-finite values that their steps make of a singular or an
   indefinite matrix. *)

let f32 shape xs = Nx.create Nx.float32 shape xs

let linalg_error kind = function
  | Nx_backend.Linalg_error e -> e.kind = kind
  | _ -> false

let cholesky_of d ~upper x =
  let a = array_of x and s = Nx.shape x in
  raises_match (linalg_error `Not_positive_definite) (fun () ->
      Nx_cpu.cholesky ~upper a ~dst:(fresh host Nx.float32 s));
  let module K = (val compiled) in
  let { on; dst } = on_device d in
  let r = dst Nx.float32 s in
  K.cholesky ~upper (on a) ~dst:r;
  back r

let element x i j = Nx.item [ i; j ] x

let where_eager_raises d =
  let indefinite = f32 [| 2; 2 |] [| 1.; 2.; 2.; 1. |] in
  [
    test
      "cholesky of a matrix that is not positive-definite is NaN from the \
       column of its first pivot that is not positive" (fun () ->
        let l = cholesky_of d ~upper:false indefinite in
        equal float_exact 1. (element l 0 0);
        equal float_exact 2. (element l 1 0);
        is_true ~msg:"the second pivot" (Float.is_nan (element l 1 1));
        let u = cholesky_of d ~upper:true indefinite in
        equal float_exact 2. (element u 0 1);
        is_true ~msg:"the second pivot, upper" (Float.is_nan (element u 1 1)));
    test "cholesky whose last pivot is zero is NaN there" (fun () ->
        let l =
          cholesky_of d ~upper:false (f32 [| 2; 2 |] [| 1.; 1.; 1.; 1. |])
        in
        equal float_exact 1. (element l 1 0);
        is_true (Float.is_nan (element l 1 1)));
    test
      "a triangular solve with a zero on the diagonal is non-finite from its \
       row on" (fun () ->
        let a = array_of (f32 [| 2; 2 |] [| 1.; 0.; 1.; 0. |]) in
        let b = array_of (f32 [| 2 |] [| 1.; 1. |]) in
        let solve (module K : Nx_backend.S) env =
          let dst = env.dst Nx.float32 [| 2 |] in
          K.solve_triangular ~upper:false ~transpose:false ~unit_diag:false
            (env.on a) (env.on b) ~dst;
          dst
        in
        raises_match (linalg_error `Singular) (fun () -> solve cpu on_cpu);
        let x = Nx.to_array (back (solve compiled (on_device d))) in
        equal float_exact 1. x.(0);
        is_false ~msg:"the row of the zero" (Float.is_finite x.(1)));
  ]

(* Refusals *)

(* Refused, naming the backend and the kernel [op] first. *)
let refused op = function
  | Nx_backend.Refused why ->
      String.starts_with ~prefix:("compiled: " ^ op ^ ":") why
  | _ -> false

let bytes_of (a : ('a, 'b) arr) =
  let c = B.bigarray Bigarray.char (copied host a.buffer) in
  String.init (Bigarray.Array1.dim c) (Bigarray.Array1.get c)

(* [refuses d op ~dst f] asserts that [f] raises Refused naming [op], having
   allocated nothing on [d] or the host and left [dst] as it was. The memory of
   buffers collected meanwhile counts as returned, so what [f] leaves allocated
   is at most none. *)
let refuses d op ~dst f =
  let before = bytes_of dst in
  let stats () = (Nx_device.stats d.device, Nx_device.stats host) in
  let d0, h0 = stats () in
  raises_match (refused op) f;
  let d1, h1 = stats () in
  let allocated s s' = Nx_device.Stats.(allocated (diff s s')) in
  at_most ~msg:"bytes allocated on the device" int ~than:0 (allocated d0 d1);
  at_most ~msg:"bytes allocated on the host" int ~than:0 (allocated h0 h1);
  equal ~msg:"the destination" string before (bytes_of dst)

(* An array of [shape] on [d] whose bytes are all [0x5a]. *)
let filled d dtype shape =
  let h = fresh host dtype shape in
  Bigarray.Array1.fill (B.bigarray Bigarray.char h.buffer) '\x5a';
  moved d.device h

let refusals d =
  let module K = (val compiled) in
  [
    test "fft is refused" (fun () ->
        let x = filled d Nx.complex64 [| 4 |] in
        let dst = filled d Nx.complex64 [| 4 |] in
        refuses d "fft" ~dst (fun () ->
            K.fft ~inverse:false ~axes:[| 0 |] x ~dst));
    test "rfft is refused" (fun () ->
        let x = filled d Nx.float32 [| 4 |] in
        let dst = filled d Nx.complex64 [| 3 |] in
        refuses d "rfft" ~dst (fun () -> K.rfft ~axes:[| 0 |] x ~dst));
    test "irfft is refused" (fun () ->
        let x = filled d Nx.complex64 [| 3 |] in
        let dst = filled d Nx.float32 [| 4 |] in
        refuses d "irfft" ~dst (fun () -> K.irfft ~axes:[| 0 |] ~s:None x ~dst));
    test "eig is refused" (fun () ->
        let x = filled d Nx.float32 [| 2; 2 |] in
        let values = filled d Nx.complex128 [| 2 |] in
        refuses d "eig" ~dst:values (fun () -> K.eig x ~values ~vectors:None));
    test "eigh is refused" (fun () ->
        let x = filled d Nx.float32 [| 2; 2 |] in
        let values = filled d Nx.float64 [| 2 |] in
        let vectors = filled d Nx.float32 [| 2; 2 |] in
        refuses d "eigh" ~dst:vectors (fun () ->
            K.eigh x ~values ~vectors:(Some vectors)));
    cases "a dtype no graph holds is refused"
      ~name:(fun (D dt) -> Nx_dtype.to_string dt)
      [ D Nx.int4; D Nx.uint4; D Nx.complex64; D Nx.complex128 ]
      (fun (D dt) ->
        let x = filled d dt [| 3 |] and dst = filled d dt [| 3 |] in
        let y = filled d Nx.float32 [| 3 |] in
        refuses d "unary" ~dst (fun () -> K.unary Neg x ~dst);
        refuses d "contiguous" ~dst (fun () -> K.contiguous x ~dst);
        refuses d "cast" ~dst (fun () -> K.cast y ~dst);
        refuses d "cast" ~dst:y (fun () -> K.cast x ~dst:y));
  ]

(* The refusals of what the host's renderer lacks: the 8-bit floats. *)
let host_refusals =
  let module K = (val compiled) in
  [
    cases "an 8-bit float is refused on the host"
      ~name:(fun (D dt) -> Nx_dtype.to_string dt)
      [ D Nx.float8_e4m3; D Nx.float8_e5m2 ]
      (fun (D dt) ->
        let x = filled on_host dt [| 3 |] and dst = filled on_host dt [| 3 |] in
        refuses on_host "binary" ~dst (fun () -> K.binary Add x x ~dst));
  ]

(* The refusals of what Metal's renderer lacks: float64, which svd's singular
   values are. *)
let metal_refusals d =
  let module K = (val compiled) in
  [
    test "it runs on Metal, whose queues tolk encodes" (fun () ->
        is_true (Nx_backend.runs_on Rune_next.Compiled.backend d.device));
    test "float64 is refused" (fun () ->
        let x = filled d Nx.float64 [| 3 |]
        and dst = filled d Nx.float64 [| 3 |] in
        refuses d "binary" ~dst (fun () -> K.binary Add x x ~dst));
    test "svd is refused, its singular values being float64" (fun () ->
        let x = filled d Nx.float32 [| 2; 2 |] in
        let u = filled d Nx.float32 [| 2; 2 |] in
        let s = filled d Nx.float64 [| 2 |] in
        let vt = filled d Nx.float32 [| 2; 2 |] in
        refuses d "svd" ~dst:u (fun () -> K.svd x ~u ~s ~vt));
  ]

(* Devices *)

let driver name mapping =
  Nx_device.Driver.device ~name ~arch:"test" ~budget:max_int
    (Host_visible { memory = Nx_device.Driver.host_memory; mapping })

(* Devices over the host's memory: one that maps the host's, one that does
   not. *)
let shared = driver "SHARED" (Some Identity)
let own = driver "OWN" None

let devices =
  let runs_on = Nx_backend.runs_on Rune_next.Compiled.backend in
  [
    test "it runs on the host and on a device that shares the host's memory"
      (fun () ->
        is_true ~msg:"host" (runs_on host);
        is_true ~msg:"shared" (runs_on shared));
    test "it does not run on the disk or on a device of its own memory"
      (fun () ->
        is_false ~msg:"disk" (runs_on Nx_device.disk);
        is_false ~msg:"own" (runs_on own));
    test "a device that shares the host's memory computes" (fun () ->
        let d = { on_host with device = shared; name = "shared" } in
        let x = array_of (f32 [| 3 |] [| 1.; -0.; 2.5 |]) in
        exact_of (both d (unary Neg x)));
    test
      "a kernel on a device it does not run on is refused, though nx never \
       calls it there" (fun () ->
        let d = { on_host with device = own; name = "own" } in
        let x = filled d Nx.float32 [| 3 |]
        and dst = filled d Nx.float32 [| 3 |] in
        let module K = (val compiled) in
        refuses d "unary" ~dst (fun () -> K.unary Neg x ~dst));
  ]

(* The cache *)

(* [compiles f] is [f ()] and the number of programs compiled meanwhile: the
   spans the host records for them. *)
let compiles f =
  let p = Nx_device.Profile.start () in
  match f () with
  | y ->
      let compilation = function
        | Nx_device.Profile.Span s ->
            String.starts_with ~prefix:"compile " s.name
        | _ -> false
      in
      (List.length (List.filter compilation (Nx_device.Profile.stop p)), y)
  | exception e ->
      ignore (Nx_device.Profile.stop p);
      raise e

(* A float32 addition of shape [s], each operand [offset] elements into a buffer
   holding [base + i] at element [i]. Each test takes shapes of its own, so that
   its first use of a key is the process's. *)
let addition d s ~offset ~base =
  let x =
    array_of
      (Nx.add_s
         (Nx.arange_f Nx.float32 0. (float_of_int (numel s + offset)) 1.)
         base)
  in
  both d
    (binary Add
       { x with view = V.create ~offset s }
       { x with view = V.create ~offset s })

let cache d =
  [
    test "a second use of a key compiles nothing and reads its new operands"
      (fun () ->
        let first, r =
          compiles (fun () -> addition d [| 5; 7 |] ~offset:0 ~base:0.)
        in
        exact_of r;
        equal ~msg:"the first use" int 1 first;
        let again, r =
          compiles (fun () -> addition d [| 5; 7 |] ~offset:0 ~base:100.)
        in
        exact_of r;
        equal ~msg:"the second use" int 0 again);
    test
      "operands 16 bytes further into their buffers share the key and are read \
       where they are" (fun () ->
        exact_of (addition d [| 5; 9 |] ~offset:0 ~base:0.);
        let again, r =
          compiles (fun () -> addition d [| 5; 9 |] ~offset:4 ~base:0.)
        in
        exact_of r;
        equal int 0 again);
    test "operands at another offset modulo 16 bytes are another key" (fun () ->
        exact_of (addition d [| 5; 11 |] ~offset:0 ~base:0.);
        let again, r =
          compiles (fun () -> addition d [| 5; 11 |] ~offset:1 ~base:0.)
        in
        exact_of r;
        equal int 1 again);
    test "a pad with a fill of -0. after one of 0. keeps its fill's sign"
      (fun () ->
        let x = array_of (f32 [| 2 |] [| 1.; 2. |]) in
        List.iter
          (fun fill ->
            exact_of
              (both d (fun (module K : Nx_backend.S) env ->
                   let dst = env.dst Nx.float32 [| 4 |] in
                   K.pad [| (1, 1) |] fill (env.on x) ~dst;
                   dst)))
          [ 0.; -0. ]);
  ]

(* Domains *)

(* [together fs] runs each of [fs] on a domain of its own, all released at once,
   and is their results. *)
let together fs =
  let go = Atomic.make false in
  let ds =
    List.map
      (fun f ->
        Domain.spawn (fun () ->
            while not (Atomic.get go) do
              Domain.cpu_relax ()
            done;
            f ()))
      fs
  in
  Atomic.set go true;
  List.map Domain.join ds

let domains d =
  [
    test "two domains meeting one new key compile it once and both compute"
      (fun () ->
        let n, results =
          compiles (fun () ->
              together
                [
                  (fun () -> addition d [| 3; 13 |] ~offset:0 ~base:0.);
                  (fun () -> addition d [| 3; 13 |] ~offset:0 ~base:50.);
                ])
        in
        List.iter exact_of results;
        equal ~msg:"compilations" int 1 n);
    test "two domains compile two new keys at once, each computing its own"
      (fun () ->
        let n, results =
          compiles (fun () ->
              together
                [
                  (fun () -> addition d [| 3; 17 |] ~offset:0 ~base:0.);
                  (fun () -> addition d [| 3; 19 |] ~offset:0 ~base:0.);
                ])
        in
        List.iter exact_of results;
        equal ~msg:"compilations" int 2 n);
    test
      "runs of one key from two domains, many more than its links, each read \
       their own operands" (fun () ->
        let runs first =
          List.init 40 (fun i ->
              addition d [| 2; 23 |] ~offset:0 ~base:(float_of_int (first + i)))
        in
        List.iter (List.iter exact_of)
          (together [ (fun () -> runs 0); (fun () -> runs 1000) ]));
    test
      "runs of one key queued with no read between them each give their own \
       result" (fun () ->
        List.iter exact_of
          (List.init 40 (fun i ->
               addition d [| 2; 29 |] ~offset:0 ~base:(float_of_int i))));
  ]

(* Edges *)

let edges d =
  [
    test "a fold whose windows read only padding computes zeros" (fun () ->
        let x = array_of (Nx.create Nx.int8 [| 2; 1 |] [| -128; 0 |]) in
        let w =
          {
            leading = [||];
            kernel_size = [| 2 |];
            stride = [| 1 |];
            dilation = [| 2 |];
            padding = [| (1, 1) |];
            spatial = [| 1 |];
            count = 1;
          }
        in
        exact_of (both d (fold w x)));
    test "an unfold whose windows along an axis read only padding is zeros"
      (fun () ->
        let w =
          {
            leading = [| 1 |];
            kernel_size = [| 2; 1 |];
            stride = [| 1; 2 |];
            dilation = [| 1; 1 |];
            padding = [| (0, 0); (1, 0) |];
            spatial = [| 3; 1 |];
            count = 2;
          }
        in
        let gapped =
          Nx.slice [ A; A; R (0, 1) ]
            (Nx.create Nx.float16 [| 1; 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |])
        in
        exact_of (both d (unfold w (host_array gapped)));
        exact_of (both d (unfold w (array_of gapped))));
    test "a fold and an unfold with no window are zeros and empty" (fun () ->
        let w =
          {
            leading = [||];
            kernel_size = [| 3 |];
            stride = [| 1 |];
            dilation = [| 1 |];
            padding = [| (0, 0) |];
            spatial = [| 1 |];
            count = 0;
          }
        in
        exact_of (both d (fold w (array_of (Nx.zeros Nx.float32 [| 3; 0 |]))));
        exact_of
          (both d (unfold w (array_of (Nx.ones Nx.float32 [| 1 |])))));
    slow "a fold whose windows along an axis read only padding is zeros"
      (fun () ->
        let geometry ~kernel_size ~stride ~dilation ~padding ~spatial x =
          let w =
            {
              leading = [||];
              kernel_size;
              stride;
              dilation;
              padding;
              spatial;
              count = 1;
            }
          in
          exact_of (both d (fold w x))
        in
        geometry ~kernel_size:[| 2; 2 |] ~stride:[| 1; 1 |] ~dilation:[| 2; 2 |]
          ~padding:[| (0, 0); (1, 1) |]
          ~spatial:[| 3; 1 |]
          (array_of
          @@ Nx.create Nx.float32 [| 4; 1 |] [| 1.; 2.; 3.; Float.nan |]);
        geometry ~kernel_size:[| 1; 1 |] ~stride:[| 2; 1 |] ~dilation:[| 1; 2 |]
          ~padding:[| (1, 0); (1, 0) |]
          ~spatial:[| 1; 2 |]
          (array_of @@ Nx.create Nx.float32 [| 1; 3 |] [| 1.; -0.; 3. |]);
        geometry ~kernel_size:[| 2 |] ~stride:[| 1 |] ~dilation:[| 2 |]
          ~padding:[| (1, 1) |]
          ~spatial:[| 1 |]
          (array_of @@ Nx.create Nx.int32 [| 2; 1 |] [| 7l; -3l |]);
        geometry ~kernel_size:[| 1; 3 |] ~stride:[| 2; 1 |] ~dilation:[| 1; 1 |]
          ~padding:[| (1, 0); (0, 0) |]
          ~spatial:[| 1; 5 |]
          (array_of
          @@ Nx.create Nx.float32 [| 3; 3 |]
               [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]);
        geometry ~kernel_size:[| 1; 2 |] ~stride:[| 2; 1 |] ~dilation:[| 1; 1 |]
          ~padding:[| (1, 0); (0, 0) |]
          ~spatial:[| 1; 3 |]
          (array_of @@ Nx.create Nx.int8 [| 2; 2 |] [| 1; -128; 3; 4 |]);
        geometry ~kernel_size:[| 1; 2 |] ~stride:[| 2; 2 |] ~dilation:[| 1; 1 |]
          ~padding:[| (1, 0); (1, 1) |]
          ~spatial:[| 1; 3 |]
          (host_array
             (Nx.broadcast_to [| 2; 2 |]
                (Nx.create Nx.float32 [| 1; 2 |] [| 1.; 2. |]))));
    test "a fold of int8 overlapping windows compiles" (fun () ->
        let x = array_of (Nx.create Nx.int8 [| 2; 2 |] [| 0; -128; 0; 0 |]) in
        let w =
          {
            leading = [||];
            kernel_size = [| 1; 2 |];
            stride = [| 1; 1 |];
            dilation = [| 1; 2 |];
            padding = [| (0, 0); (1, 1) |];
            spatial = [| 2; 1 |];
            count = 2;
          }
        in
        exact_of (both d (fold w x)));
    test
      "an operand whose buffer starts 2 bytes into its memory is read where it \
       is" (fun () ->
        let x =
          of_bits Nx.float16 [| 3; 4 |]
            { strides = [| 1; 3 |]; offset = 0; length = 12; inner = 1 }
            (Array.init 12 (fun i -> if i = 7 then 0x7bffL else 0L))
        in
        exact_of (both d (unary Floor x)));
    test
      "a concatenation of 17 pieces, a kernel of 18 arguments, keeps every bit"
      (fun () ->
        let pieces =
          List.init 17 (fun i ->
              array_of
                (Nx.bitcast Nx.float32
                   (Nx.create Nx.int32 [| 1; 3 |]
                      [| Int32.of_int (i + 1); 0x8000_0000l; 0x7fc0_0001l |])))
        in
        let e, a =
          both d (fun (module K : Nx_backend.S) env ->
              let dst = env.dst Nx.float32 [| 17; 3 |] in
              K.cat ~axis:0 (List.map env.on pieces) ~dst;
              dst)
        in
        Traces.exact
          (Nx.bitcast Nx.int32 (value e))
          (Nx.bitcast Nx.int32 (back a)));
  ]
  @ [
      (* A device that flushes subnormals too: the sign is read from the
         bits. *)
      test "the logarithm of a negative subnormal is NaN" (fun () ->
          let log x = exact_of (both d (unary Log (array_of x))) in
          log
            (Nx.create Nx.float32 [| 3 |] [| -0x1p-149; -0x1p-130; -0x1p-127 |]);
          if d.float64 then
            log
              (Nx.create Nx.float64 [| 3 |]
                 [| -0x1p-1074; -0x1p-1030; -0x1p-1023 |]));
    ]

(* A compiled placement *)

let placement d =
  [
    test "nx computes at a compiled placement, which its results keep"
      (fun () ->
        let p =
          Nx.Placement.device ~backend:Rune_next.Compiled.backend
            (Nx.Device.of_runtime d.device)
        in
        let x = f32 [| 2; 3 |] [| 1.; -0.; 2.5; -3.; 4.; 0.5 |] in
        let y = Nx.mul (Nx.add (Nx.place p x) (Nx.place p x)) (Nx.place p x) in
        is_true ~msg:"placement" (Nx.Placement.equal p (Nx.placement y));
        Traces.exact (Nx.mul (Nx.add x x) x) (Nx.place Nx.Placement.host y));
  ]

(* Cost *)

(* The host time, in microseconds per kernel, of a chain of [n] kernels on [d],
   each [step i x dst] writing a new array of 1024 float32 from the last: once
   issued, and once the device finished them. *)
let chain d n step =
  let x = moved d.device (array_of (Nx.full Nx.float32 [| 1024 |] 0.5)) in
  let run n =
    let y = ref x in
    for i = 1 to n do
      let dst = fresh d.device Nx.float32 [| 1024 |] in
      step i !y dst;
      y := dst
    done;
    !y
  in
  ignore (back (run 64));
  let t0 = Nx_device.Profile.now () in
  let y = run n in
  let t1 = Nx_device.Profile.now () in
  ignore (back y);
  let t2 = Nx_device.Profile.now () in
  let per t = float_of_int t /. 1e3 /. float_of_int n in
  (per (t1 - t0), per (t2 - t0))

let cost d =
  let module K = (val compiled) in
  let one = moved d.device (array_of (Nx.full Nx.float32 [| 1024 |] 1.)) in
  let distinct = [| Nx_backend.Neg; Abs; Sqrt; Exp; Sin; Cos; Floor; Trunc |] in
  let measured name step =
    slow
      ("a chain of " ^ name ^ " costs well under 200 us a kernel")
      (fun () ->
        let issued, finished = chain d 1000 step in
        Printf.printf "%s on %s: %.1f us a kernel issued, %.1f finished\n%!"
          name d.name issued finished;
        less ~msg:"us a kernel, issued" float_exact ~than:200. issued)
  in
  [
    measured "distinct operations" (fun i x dst ->
        K.unary distinct.(i mod Array.length distinct) x ~dst);
    measured "add 1" (fun _ x dst -> K.binary Add x one ~dst);
  ]

(* Suite *)

(* The kernels on [d], [count] cases a law; [heavy] adds the families whose
   programs take up to a second each to compile: the transcendental functions,
   linear algebra, and the sorting networks of sort and argsort. *)
let kernels d ~count ~heavy =
  [
    group "elementwise" (elementwise d ~count ~heavy);
    group "reductions" (reductions d ~count ~heavy);
    group "indexed" (indexed d ~count);
    group "windows and products" (windows_and_products d ~count);
  ]
  @ if heavy then [ group "linear algebra" (linalg d ~count) ] else []

(* Narrow floats accumulate at float32 *)

(* Sums, running sums, products and contractions of float16 and bfloat16 run at
   float32 and round once. Each case has an exact answer that accumulating at
   the narrow width misses: float16 stops counting ones at 2048, bfloat16 at
   256, and a float16 product overflows at 256 * 256. *)
let wide d =
  let check expected (e, a) =
    Traces.exact expected (value e);
    Traces.exact expected (back a)
  in
  let ones dt shape = array_of (Nx.ones dt shape) in
  [
    test "a float16 sum of 4096 ones is 4096" (fun () ->
        check
          (Nx.scalar Nx.float16 4096.)
          (both d (reduce Sum [| 0 |] (ones Nx.float16 [| 4096 |]))));
    test "a bfloat16 sum of 512 ones is 512" (fun () ->
        check
          (Nx.scalar Nx.bfloat16 512.)
          (both d (reduce Sum [| 0 |] (ones Nx.bfloat16 [| 512 |]))));
    test "a float16 running sum of 4096 ones rounds each count once" (fun () ->
        let counts = Array.init 4096 (fun i -> float_of_int (i + 1)) in
        check
          (Nx.cast Nx.float16 (Nx.create Nx.float32 [| 4096 |] counts))
          (both d (scan Sum 0 (ones Nx.float16 [| 4096 |]))));
    test "a float16 product past the float16 range and back is exact" (fun () ->
        let x =
          array_of (Nx.create Nx.float16 [| 3 |] [| 256.; 256.; 0x1p-8 |])
        in
        check (Nx.scalar Nx.float16 256.) (both d (reduce Prod [| 0 |] x)));
    test "a float16 running product overflows only where its value does"
      (fun () ->
        let x =
          array_of (Nx.create Nx.float16 [| 3 |] [| 256.; 256.; 0x1p-8 |])
        in
        check
          (Nx.create Nx.float16 [| 3 |] [| 256.; Float.infinity; 256. |])
          (both d (scan Prod 0 x)));
    test "a float16 contraction of 4096 ones is 4096" (fun () ->
        check
          (Nx.full Nx.float16 [| 1; 1 |] 4096.)
          (both d
             (matmul [| 1; 1 |]
                (ones Nx.float16 [| 1; 4096 |])
                (ones Nx.float16 [| 4096; 1 |]))));
  ]

let contracts d =
  [
    group "narrow floats accumulate at float32" (wide d);
    group "refusals" (refusals d);
    group "linear algebra where nx.cpu raises" (where_eager_raises d);
    group "cache" (cache d);
    group "domains" (domains d);
    group "placement" (placement d);
    group "edges" (edges d);
  ]

let () =
  let metal =
    match on_metal with
    | None ->
        [ slow "no Metal device" (fun () -> skip ~reason:"no Metal device" ()) ]
    | Some m ->
        kernels m ~count:25 ~heavy:true
        @ contracts m
        @ [ group "Metal" (metal_refusals m); group "cost" (cost m) ]
  in
  exit
    (run "Rune_next.Compiled"
       [
         group "host"
           (kernels on_host ~count:1 ~heavy:false
           @ contracts on_host
           @ [
               group "refusals of the host" host_refusals;
               group "devices" devices;
             ]);
         group ~tags:[ "slow" ] "host, swept"
           (kernels on_host ~count:25 ~heavy:true);
         group ~tags:[ "slow" ] "metal" metal;
       ])
