(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reductions, scans and contractions lowered, against nx's eager result on the
   same operands: bit for bit where the lowering is exact, within the bound
   Lower_reduce states elsewhere. *)

open Windtrap
module D = Nx_array.Dtype
module P = Nx.Prim
module K = Nx_kernel

(* Elements

   A case's elements are int64 raws: a float's bits, an integer's value modulo
   its width, a boolean's low bit, and a complex number's two parts as float64
   bits. *)

let numel s = Array.fold_left ( * ) 1 s

(* A float format's exponent and fraction widths. *)
let format (type s) (dt : (float, s) D.t) =
  match dt with
  | D.Float64 -> (11, 52)
  | D.Float32 -> (8, 23)
  | D.Float16 -> (5, 10)
  | D.Bfloat16 -> (8, 7)
  | D.Float8_e4m3fn -> (4, 3)
  | D.Float8_e5m2 -> (5, 2)
  | D.Float4_e2m1fn -> (2, 1)

let width (type v s) (dt : (v, s) D.t) =
  match dt with
  | D.Int64 | D.Uint64 -> 64
  | D.Int32 | D.Uint32 -> 32
  | D.Int16 | D.Uint16 -> 16
  | D.Int8 | D.Uint8 -> 8
  | D.Int4 | D.Uint4 -> 4
  | _ -> 1

let low w r = Int64.to_int r land ((1 lsl w) - 1)

let signed w r =
  let v = low w r in
  if v >= 1 lsl (w - 1) then v - (1 lsl w) else v

let complex raws n =
  let part i = Int64.float_of_bits raws.(i) in
  Array.init n (fun i -> { Complex.re = part (2 * i); im = part ((2 * i) + 1) })

let value (type v s) (dt : (v, s) D.t) shape (raws : int64 array) :
    (v, s, Nx.host) Nx.t =
  let ints : type w r. (w, r) D.t -> (int64 -> w) -> (w, r, Nx.host) Nx.t =
   fun d f -> Nx.create d shape (Array.map f raws)
  in
  match dt with
  | D.Float64 -> Nx.bitcast dt (ints D.Int64 Fun.id)
  | D.Float32 -> Nx.bitcast dt (ints D.Int32 Int64.to_int32)
  | D.Float16 | D.Bfloat16 -> Nx.bitcast dt (ints D.Uint16 (low 16))
  | D.Float8_e4m3fn | D.Float8_e5m2 -> Nx.bitcast dt (ints D.Uint8 (low 8))
  | D.Float4_e2m1fn -> Nx.bitcast dt (ints D.Uint4 (low 4))
  | D.Int64 -> ints dt Fun.id
  | D.Uint64 -> ints dt Fun.id
  | D.Int32 -> ints dt Int64.to_int32
  | D.Uint32 -> ints dt Int64.to_int32
  | D.Int16 -> ints dt (signed 16)
  | D.Uint16 -> ints dt (low 16)
  | D.Int8 -> ints dt (signed 8)
  | D.Uint8 -> ints dt (low 8)
  | D.Int4 -> ints dt (signed 4)
  | D.Uint4 -> ints dt (low 4)
  | D.Bool -> ints dt (fun r -> Int64.logand r 1L = 1L)
  | D.Bit -> ints dt (fun r -> Int64.logand r 1L = 1L)
  | D.Complex128 -> Nx.create dt shape (complex raws (numel shape))
  | D.Complex64 -> Nx.create dt shape (complex raws (numel shape))

(* The number of raws of [n] elements. *)
let raws_of (D.Any dt) n = if D.is D.Complex dt then 2 * n else n

(* Float classes *)

type cls = Number | Zero | Inf | Nan | Subnormal

let float_class (type s) (dt : (float, s) D.t) r =
  let e, f = format dt in
  let ex = low e (Int64.shift_right_logical r f) in
  let fr = Int64.logand r (Int64.pred (Int64.shift_left 1L f)) in
  let top = (1 lsl e) - 1 in
  let zero () = if fr = 0L then Zero else Subnormal in
  match dt with
  | D.Float4_e2m1fn -> if ex = 0 then zero () else Number
  | D.Float8_e4m3fn ->
      if ex = top && fr = 7L then Nan else if ex = 0 then zero () else Number
  | _ ->
      if ex = top then if fr = 0L then Inf else Nan
      else if ex = 0 then zero ()
      else Number

(* A NaN whose quiet bit, its fraction's top bit, is clear. *)
let signalling (type s) (dt : (float, s) D.t) r =
  let _, f = format dt in
  float_class dt r = Nan
  && (match dt with D.Float8_e4m3fn -> false | _ -> true)
  && Int64.logand r (Int64.shift_left 1L (f - 1)) = 0L

let negative (type s) (dt : (float, s) D.t) r =
  let e, f = format dt in
  Int64.logand r (Int64.shift_left 1L (e + f)) <> 0L

(* Generators *)

let among xs = Gen.one_of (List.map Gen.constant xs)

(* [float_raw ~spread dt] draws a float of [dt] whose finite numbers lie within
   [2^±spread], with zeros, infinities, NaNs of any sign and payload, and
   subnormals of weight [subnormal] against the numbers' 10. *)
let float_raw (type s) ~spread ~subnormal (dt : (float, s) D.t) =
  let open Gen in
  let e, f = format dt in
  let bias = (1 lsl (e - 1)) - 1 and top = (1 lsl e) - 1 in
  let mask = Int64.pred (Int64.shift_left 1L f) in
  let field sign ex fr =
    let s = if sign then Int64.shift_left 1L (e + f) else 0L in
    Int64.(logor s (logor (shift_left (of_int ex) f) fr))
  in
  let* cls =
    frequency
      [
        (10, constant Number);
        (2, constant Zero);
        (1, constant Inf);
        (2, constant Nan);
        (subnormal, constant Subnormal);
      ]
  in
  let* sign = bool in
  let* ex = int_range (max 1 (bias - spread)) (min (top - 1) (bias + spread)) in
  let+ fr = map (Int64.logand mask) int64 in
  let some fr = if fr = 0L then 1L else fr in
  match (cls, dt) with
  | Number, _ | (Inf | Nan), D.Float4_e2m1fn | Inf, D.Float8_e4m3fn ->
      field sign ex fr
  | Zero, _ -> field sign 0 0L
  | Subnormal, _ -> field sign 0 (some fr)
  | Inf, _ -> field sign top 0L
  | Nan, D.Float8_e4m3fn -> field sign top mask
  | Nan, _ -> field sign top (some fr)

(* [int_raw dt] draws an integer of [dt]: its extremes, small values, or any
   bits. *)
let int_raw (type v s) (dt : (v, s) D.t) =
  let open Gen in
  let w = width dt in
  let lo, hi =
    match (w, D.is D.Signed dt) with
    | 64, true -> (Int64.min_int, Int64.max_int)
    | 64, false -> (0L, -1L)
    | w, true ->
        (Int64.of_int (-(1 lsl (w - 1))), Int64.of_int ((1 lsl (w - 1)) - 1))
    | w, false -> (0L, Int64.of_int ((1 lsl w) - 1))
  in
  if w = 1 then map Int64.of_int (int_range 0 1)
  else
    frequency
      [
        (1, among [ lo; hi ]);
        (3, map Int64.of_int (int_range (-3) 3));
        (1, int64);
      ]

let raw ~spread ~subnormal (D.Any dt) =
  match D.kind dt with
  | D.Float -> float_raw ~spread ~subnormal dt
  | D.Complex -> float_raw ~spread ~subnormal D.Float64
  | _ -> int_raw dt

let extent =
  Gen.frequency
    [ (1, Gen.constant 0); (2, Gen.constant 1); (6, Gen.int_range 2 4) ]

let shape =
  let open Gen in
  let* r = int_range 1 3 in
  map Array.of_list (list ~size:(constant r) extent)

(* Cases *)

type view = Plain | Transposed | Flipped | Broadcast

type case = {
  dt : D.any;
  m : K.monoid;
  shape : int array;
  view : view;
  axes : int list;
  raws : int64 array;
  carry : int64 array option;
}

(* The shape of the array a view of shape [s] reads. *)
let base view s =
  match view with
  | Plain | Flipped -> s
  | Transposed -> Array.of_list (List.rev (Array.to_list s))
  | Broadcast -> Array.mapi (fun i n -> if i = 0 then 1 else n) s

let viewed view s x =
  match view with
  | Plain -> x
  | Transposed -> Nx.transpose x
  | Flipped -> Nx.flip x
  | Broadcast -> Nx.broadcast_to s x

(* The shape of a scan's carry along [axis]. *)
let carried s axis = Array.mapi (fun i n -> if i = axis then 1 else n) s

let monoid_name : K.monoid -> string = function
  | Sum -> "sum"
  | Prod -> "prod"
  | Max -> "max"
  | Min -> "min"
  | Logsumexp -> "logsumexp"

let pp_case ppf c =
  let (D.Any dt) = c.dt in
  let ints a = String.concat ";" (List.map string_of_int a) in
  let raws a =
    String.concat ";" (Array.to_list (Array.map (Printf.sprintf "%Lx") a))
  in
  Format.fprintf ppf "%s %s [%s] %s axes [%s] raws [%s]%s" (monoid_name c.m)
    (D.name dt)
    (ints (Array.to_list c.shape))
    (match c.view with
    | Plain -> "plain"
    | Transposed -> "transposed"
    | Flipped -> "flipped"
    | Broadcast -> "broadcast")
    (ints c.axes) (raws c.raws)
    (match c.carry with None -> "" | Some a -> " carry [" ^ raws a ^ "]")

(* [case ~spread ~carry dts ms] draws a case of a dtype of the weighted [dts]
   and a monoid of [ms], over one axis where [scan]. *)
let case ?(spread = 4) ?(subnormal = 1) ?(scan = false) dts ms =
  let open Gen in
  let* dt = frequency (List.map (fun (w, d) -> (w, constant d)) dts) in
  let* m = among ms in
  let* s = shape in
  let* view = among [ Plain; Transposed; Flipped; Broadcast ] in
  let r = Array.length s in
  let* keep = list ~size:(constant r) bool in
  let* first = int_range 0 (r - 1) in
  let axes =
    if scan then [ first ]
    else
      List.filter (fun a -> a = first || List.nth keep a) (List.init r Fun.id)
  in
  let* raws =
    array
      ~size:(constant (raws_of dt (numel (base view s))))
      (raw ~spread ~subnormal dt)
  in
  let+ carry =
    if not scan then constant None
    else
      let c = K.carrier dt in
      let n = raws_of c (numel (carried s first)) in
      frequency
        [
          (1, constant None);
          ( 1,
            map Option.some
              (array ~size:(constant n) (raw ~spread ~subnormal c)) );
        ]
  in
  { dt; m; shape = s; view; axes; raws; carry }

let case ?spread ?subnormal ?scan dts ms =
  Gen.with_pp pp_case (case ?spread ?subnormal ?scan dts ms)

let operand (type v s) (dt : (v, s) D.t) c : (v, s, Nx.host) Nx.t =
  viewed c.view c.shape (value dt (base c.view c.shape) c.raws)

let floats =
  D.
    [
      Any Float64;
      Any Float32;
      Any Float16;
      Any Bfloat16;
      Any Float8_e4m3fn;
      Any Float8_e5m2;
      Any Float4_e2m1fn;
    ]

let ints =
  D.
    [
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
    ]

let complexes = D.[ Any Complex64; Any Complex128 ]
let weigh w ds = List.map (fun d -> (w, d)) ds

(* Covers *)

let is_float (D.Any dt) = D.is D.Float dt

let float_raws c (f : 's. (float, 's) D.t -> int64 -> bool) =
  let (D.Any dt) = c.dt in
  match D.kind dt with
  | D.Float -> List.filter (f dt) (Array.to_list c.raws)
  | _ -> []

let cover_floats c =
  let has (f : 's. (float, 's) D.t -> int64 -> bool) = float_raws c f <> [] in
  let nans = float_raws c (fun dt r -> float_class dt r = Nan) in
  cover "a NaN term" (nans <> []);
  cover "NaNs of two bit patterns"
    (List.length (List.sort_uniq compare nans) > 1);
  cover "a signalling NaN" (has signalling);
  cover "a negative NaN"
    (has (fun dt r -> float_class dt r = Nan && negative dt r));
  cover "both zeros"
    (has (fun dt r -> float_class dt r = Zero && negative dt r)
    && has (fun dt r -> float_class dt r = Zero && not (negative dt r)));
  cover "an infinity" (has (fun dt r -> float_class dt r = Inf))

let cover_shapes c =
  cover "a strided or broadcast view" (c.view <> Plain);
  cover "a zero-size operand" (numel c.shape = 0);
  cover "a one-element operand" (numel c.shape = 1);
  cover "several axes" (List.length c.axes > 1)

(* Operations *)

let identity dt = K.Prog.v ~ins:[| D.Any dt |] [| In 0 |] ~outs:[| 0 |]

let reduce_op m axes x =
  let dt = Nx.dtype x in
  P.Reduce
    {
      layout = Nx_array.Layout.contiguous (Nx.shape x);
      axes = Iarray.of_list axes;
      prog = identity dt;
      reductions = [ Monoid (m, 0, dt) ];
      loads = Iarray.of_list [ P.Plain x ];
    }

let scan_op m axis x carry =
  let dt = Nx.dtype x in
  P.Scan
    {
      layout = Nx_array.Layout.contiguous (Nx.shape x);
      axis;
      prog = identity dt;
      reduction = Monoid (m, 0, dt);
      loads = Iarray.of_list [ P.Plain x ];
      carry;
    }

(* [results c] is the eager and the lowered elements of [c]'s reduction or
   scan. *)
let results (type v s) (dt : (v, s) D.t) ~scan c : v array * v array =
  let x = operand dt c in
  if scan then
    let axis = List.hd c.axes in
    let carry =
      Option.map
        (fun raws ->
          let (D.Any cd) = K.carrier (D.Any dt) in
          P.Any (value cd (carried c.shape axis) raws))
        c.carry
    in
    let op () = P.eval ~by:"t" (scan_op c.m axis x carry) in
    (Nx.to_array (op ()), Lowered.run op)
  else
    let op () = fst (P.eval ~by:"t" (reduce_op c.m c.axes x)) in
    (Nx.to_array (op ()), Lowered.run op)

(* Comparisons *)

let hex dt v =
  let b = K.Prog.bits dt v in
  let n = String.length b in
  String.concat ""
    (List.init n (fun i -> Printf.sprintf "%02x" (Char.code b.[n - 1 - i])))

let same_bits dt eager lowered =
  equal (array string) (Array.map (hex dt) eager) (Array.map (hex dt) lowered)

let float_kind x =
  if Float.is_nan x then "nan"
  else if x = Float.infinity then "+inf"
  else if x = Float.neg_infinity then "-inf"
  else if x = 0. && Float.sign_bit x then "-0"
  else if x = 0. then "+0"
  else "number"

(* [near ~zeros ~bound e l] checks [l] against [e]: NaN where [e] is, the same
   infinity, the same zero where [zeros], and otherwise within [bound],
   inclusive. A zero against a number is within the bound or not, as any other
   difference. *)
let near ?(zeros = true) ~bound e l =
  let unsigned k = if zeros || (k <> "+0" && k <> "-0") then k else "0" in
  let k = unsigned (float_kind e) and kl = unsigned (float_kind l) in
  let number k = k = "number" || k = "+0" || k = "-0" || k = "0" in
  if not (number k && number kl && (k = "number" || kl = "number")) then
    equal ~msg:(Printf.sprintf "%h against %h" e l) string k kl
  else
    less
      ~msg:(Printf.sprintf "%h against %h" e l)
      float_exact ~than:(Float.succ bound)
      (Float.abs (e -. l))

(* Bounds

   [γ(k) = k u / (1 - k u)]. A result of [dt] narrower than its carrier rounds
   once more on each side: [u_dt] of each and half a subnormal step. *)

let gamma u k =
  let k = Float.of_int (max 0 k) in
  k *. u /. (1. -. (k *. u))

let carrier_roundoff (D.Any dt) =
  match dt with
  | D.Float64 | D.Complex128 -> Float.ldexp 1. (-53)
  | _ -> Float.ldexp 1. (-24)

let rounding (type s) (dt : (float, s) D.t) e l =
  match dt with
  | D.Float64 | D.Float32 -> 0.
  | _ ->
      let ex, f = format dt in
      let tiny = Float.ldexp 1. (2 - (1 lsl (ex - 1)) - f) in
      (Float.ldexp 1. (-(f + 1)) *. (Float.abs e +. Float.abs l)) +. tiny

(* [members s axes] is, for each element of a value of shape [s] in C order, the
   position of the result a reduction along [axes] gives it to. *)
let members s axes =
  let r = Array.length s in
  Array.init (numel s) (fun i ->
      let rest = ref i and o = ref 0 and m = ref 1 in
      for a = r - 1 downto 0 do
        let k = !rest mod s.(a) in
        rest := !rest / s.(a);
        if not (List.mem a axes) then (
          o := !o + (k * !m);
          m := !m * s.(a))
      done;
      !o)

let coord s axis i =
  let after = ref 1 in
  for a = axis + 1 to Array.length s - 1 do
    after := !after * s.(a)
  done;
  i / !after mod s.(axis)

(* [terms c] is, for each result of [c], the positions of the elements of its
   operand it folds and whether a carry enters it. *)
let terms c ~scan =
  let s = c.shape in
  if scan then
    let axis = List.hd c.axes in
    let slice = members s [ axis ] in
    Array.init (numel s) (fun j ->
        List.filter
          (fun i -> slice.(i) = slice.(j) && coord s axis i <= coord s axis j)
          (List.init (numel s) Fun.id))
  else
    let mem = members s c.axes in
    let kept =
      numel
        (Array.of_list
           (List.filteri (fun a _ -> not (List.mem a c.axes)) (Array.to_list s)))
    in
    Array.init kept (fun o ->
        List.filter (fun i -> mem.(i) = o) (List.init (numel s) Fun.id))

(* [parts dt xs] is the elements [xs] of the float or complex [dt] as float
   arrays: [[xs]], or their real parts then their imaginary parts. *)
let parts (type v s) (dt : (v, s) D.t) (xs : v array) : float array list =
  match D.kind dt with
  | D.Float -> [ xs ]
  | D.Complex ->
      [
        Array.map (fun z -> z.Complex.re) xs;
        Array.map (fun z -> z.Complex.im) xs;
      ]
  | _ -> invalid_arg "parts: not a float"

let rounding_of (type v s) (dt : (v, s) D.t) =
  match D.kind dt with D.Float -> rounding dt | _ -> fun _ _ -> 0.

let sum_abs = List.fold_left (fun a v -> a +. Float.abs v) 0.

(* [check_floats c ~scan xs cs e l] checks each part of the eager results [e]
   against the lowered [l] of a float sum or product of the elements [xs], with
   the carries [cs]: on each side within [γ(n - 1) Σ|x|] of the exact sum of [n]
   terms, a relative [γ(n - 1)] of the exact product. *)
let check_floats c ~scan ~round xs cs e l =
  let ts = terms c ~scan in
  let slice = members c.shape c.axes in
  let u = carrier_roundoff c.dt in
  List.iteri
    (fun p x ->
      let e = List.nth e p and l = List.nth l p in
      Array.iteri
        (fun o members ->
          let carry =
            match cs with
            | Some cs -> [ (List.nth cs p).(slice.(o)) ]
            | None -> []
          in
          let vals = carry @ List.map (fun i -> x.(i)) members in
          let g = gamma u (List.length vals - 1) in
          let bound =
            match c.m with
            | Prod -> 2.01 *. g *. Float.max (Float.abs e.(o)) (Float.abs l.(o))
            | _ -> 2. *. g *. sum_abs vals
          in
          near ~bound:(bound +. round e.(o) l.(o)) e.(o) l.(o))
        ts)
    xs

(* A [Max] or [Min] with a result element and no term is refused. *)
let refused c =
  let kept =
    List.filteri (fun a _ -> not (List.mem a c.axes)) (Array.to_list c.shape)
  in
  (c.m = Max || c.m = Min)
  && List.exists (fun a -> c.shape.(a) = 0) c.axes
  && List.for_all (fun n -> n > 0) kept

let exact ~scan c =
  let (D.Any dt) = c.dt in
  let e, l = results dt ~scan c in
  same_bits dt e l

let bounded ~scan c =
  let (D.Any dt) = c.dt in
  let e, l = results dt ~scan c in
  let xs = parts dt (Nx.to_array (operand dt c)) in
  let cs =
    Option.map
      (fun raws ->
        let (D.Any cd) = K.carrier c.dt in
        parts cd
          (Nx.to_array (value cd (carried c.shape (List.hd c.axes)) raws)))
      c.carry
  in
  check_floats c ~scan ~round:(rounding_of dt) xs cs (parts dt e) (parts dt l)

let is_bool (D.Any dt) = D.is D.Boolean dt
let reduced_empty c = List.exists (fun a -> c.shape.(a) = 0) c.axes
let cover_monoids c ms = List.iter (fun m -> cover (monoid_name m) (c.m = m)) ms

(* Examples every run takes: a one-element and a zero-size operand, and for
   float32 and float16 rows of NaNs of both signs and kinds, zeros of both
   signs, infinities, and all [-0], a NaN carry before the first row. *)

let specials (D.Any d) =
  match d with
  | D.Float32 ->
      Some
        ( [|
            0x7fc00001L;
            0xff800005L;
            0L;
            0x80000000L;
            0x7f800000L;
            0xff800000L;
            0x3f800000L;
            0x7fa00000L;
            0x80000000L;
            0L;
            0x80000000L;
            0xbf800000L;
            0x80000000L;
            0x80000000L;
            0x80000000L;
            0x80000000L;
          |],
          [| 0x7f800009L; 0x3f800000L; 0x80000000L; 0L |] )
  | D.Float16 ->
      Some
        ( [|
            0x7e01L;
            0xfc05L;
            0L;
            0x8000L;
            0x7c00L;
            0xfc00L;
            0x3c00L;
            0x7d00L;
            0x8000L;
            0L;
            0x8000L;
            0xbc00L;
            0x8000L;
            0x8000L;
            0x8000L;
            0x8000L;
          |],
          [| 0x7f800009L; 0x3f800000L; 0x80000000L; 0L |] )
  | _ -> None

let examples ?(scan = false) dts ms =
  let ex dt m shape axes raws carry =
    { dt; m; shape; view = Plain; axes; raws; carry }
  in
  let each dt (m : K.monoid) =
    let zero =
      if (m = Max || m = Min) && not scan then [| 0; 2 |] else [| 2; 0 |]
    in
    let special =
      match specials dt with
      | None -> []
      | Some (raws, carry) ->
          [ ex dt m [| 4; 4 |] [ 1 ] raws (if scan then Some carry else None) ]
    in
    ex dt m [| 1 |] [ 0 ] (Array.make (raws_of dt 1) 1L) None
    :: ex dt m zero [ 1 ] [||] None
    :: special
  in
  List.concat_map (fun dt -> List.concat_map (each dt) ms) dts

(* Reductions *)

let extremes =
  let dts = weigh 3 floats @ weigh 2 ints @ weigh 6 D.[ Any Bool; Any Bit ] in
  prop ~count:25
    ~examples:
      (examples D.[ Any Float32; Any Float16; Any Bool; Any Int8 ] [ Max; Min ])
    "extremes give eager's bits"
    (case dts [ Max; Min ])
    (fun c ->
      assume (not (refused c));
      cover_monoids c [ Max; Min ];
      cover_floats c;
      cover_shapes c;
      cover "a boolean" (is_bool c.dt);
      cover "an integer" (not (is_float c.dt || is_bool c.dt));
      exact ~scan:false c)

let int_folds =
  prop ~count:15
    ~examples:(examples D.[ Any Int8 ] [ Sum; Prod ])
    "integer sums and products wrap as eager's"
    (case (weigh 1 ints) [ Sum; Prod ])
    (fun c ->
      cover_monoids c [ Sum; Prod ];
      cover "an empty reduced extent" (reduced_empty c);
      cover_shapes c;
      exact ~scan:false c)

let float_sums =
  prop ~count:15
    ~examples:(examples D.[ Any Float32; Any Float16 ] [ Sum ])
    "float sums hold their bound"
    (case (weigh 1 floats) [ Sum ])
    (fun c ->
      cover_floats c;
      cover "an empty reduced extent" (reduced_empty c);
      cover "a narrow float"
        (not (List.mem c.dt D.[ Any Float32; Any Float64 ]));
      bounded ~scan:false c)

(* Products of numbers of one binade: no overflow or underflow. *)
let float_products =
  prop ~count:12
    ~examples:(examples D.[ Any Float32 ] [ Prod ])
    "float products hold their bound"
    (case ~spread:0 ~subnormal:0 (weigh 1 floats) [ Prod ])
    (fun c ->
      cover "a NaN term"
        (float_raws c (fun dt r -> float_class dt r = Nan) <> []);
      cover "an empty reduced extent" (reduced_empty c);
      bounded ~scan:false c)

let complex_sums =
  prop ~count:8
    ~examples:(examples D.[ Any Complex64 ] [ Sum ])
    "complex sums add their parts"
    (case (weigh 1 complexes) [ Sum ])
    (fun c ->
      cover "complex64" (c.dt = D.Any D.Complex64);
      bounded ~scan:false c)

(* Scans *)

let extreme_scans =
  let dts = weigh 3 floats @ weigh 2 ints @ weigh 6 D.[ Any Bool; Any Bit ] in
  prop ~count:20
    ~examples:
      (examples ~scan:true
         D.[ Any Float32; Any Float16; Any Bool ]
         [ Max; Min ])
    "extreme scans give eager's bits"
    (case ~scan:true dts [ Max; Min ])
    (fun c ->
      cover_monoids c [ Max; Min ];
      cover "a carry" (c.carry <> None);
      cover "a NaN term"
        (float_raws c (fun dt r -> float_class dt r = Nan) <> []);
      cover "a boolean" (is_bool c.dt);
      exact ~scan:true c)

let int_scans =
  prop ~count:12
    ~examples:(examples ~scan:true D.[ Any Int8 ] [ Sum; Prod ])
    "integer running sums and products wrap as eager's"
    (case ~scan:true (weigh 1 ints) [ Sum; Prod ])
    (fun c ->
      cover_monoids c [ Sum; Prod ];
      cover "a carry" (c.carry <> None);
      exact ~scan:true c)

let float_scans =
  prop ~count:12
    ~examples:(examples ~scan:true D.[ Any Float32; Any Float16 ] [ Sum; Prod ])
    "float running sums and products hold their bound"
    (case ~scan:true ~spread:0 ~subnormal:0 (weigh 1 floats) [ Sum; Prod ])
    (fun c ->
      cover_monoids c [ Sum; Prod ];
      cover "a carry" (c.carry <> None);
      bounded ~scan:true c)

let complex_scans =
  prop ~count:5
    ~examples:(examples ~scan:true D.[ Any Complex64 ] [ Sum ])
    "complex running sums add their parts"
    (case ~scan:true (weigh 1 complexes) [ Sum ])
    (fun c ->
      cover "a carry" (c.carry <> None);
      bounded ~scan:true c)

(* Contractions

   Each operand's axes are batch, free or contracted, in any order, read through
   a transposed view or not. *)

type role = B of int | F of int | C of int

type contraction = {
  cdt : D.any;
  acc : D.any;
  eb : int array;
  ek : int array;
  fa : int array;
  fb : int array;
  ra : role array;
  rb : role array;
  ta : bool;
  tb : bool;
  araws : int64 array;
  braws : int64 array;
  init : int64 array option;
}

let ashape c f r =
  Array.map (function B i -> c.eb.(i) | C i -> c.ek.(i) | F i -> f.(i)) r

let out_shape c = Array.concat [ c.eb; c.fa; c.fb ]

let pairs c pick n =
  let pos r x = Option.get (Array.find_index (( = ) x) r) in
  Iarray.of_list (List.init n (fun i -> (pos c.ra (pick i), pos c.rb (pick i))))

let pp_contraction ppf c =
  let name (D.Any d) = D.name d in
  let ints a = String.concat ";" (Array.to_list (Array.map string_of_int a)) in
  let role = function
    | B i -> "b" ^ string_of_int i
    | C i -> "k" ^ string_of_int i
    | F i -> "f" ^ string_of_int i
  in
  let roles r = String.concat ";" (Array.to_list (Array.map role r)) in
  let raws a =
    String.concat ";" (Array.to_list (Array.map (Printf.sprintf "%Lx") a))
  in
  Format.fprintf ppf
    "%s acc %s a [%s]%s [%s] b [%s]%s [%s] raws a [%s] b [%s]%s" (name c.cdt)
    (name c.acc) (roles c.ra)
    (if c.ta then " transposed" else "")
    (ints (ashape c c.fa c.ra))
    (roles c.rb)
    (if c.tb then " transposed" else "")
    (ints (ashape c c.fb c.rb))
    (raws c.araws) (raws c.braws)
    (match c.init with None -> "" | Some r -> " init [" ^ raws r ^ "]")

let contraction dts accs =
  let open Gen in
  let small =
    frequency [ (1, constant 0); (1, constant 1); (4, int_range 2 3) ]
  in
  let extents n = map Array.of_list (list ~size:(constant n) small) in
  let roles nf eb ek =
    List.init (Array.length eb) (fun i -> B i)
    @ List.init nf (fun i -> F i)
    @ List.init (Array.length ek) (fun i -> C i)
  in
  let* cdt = frequency (List.map (fun (w, d) -> (w, constant d)) dts) in
  let* acc = among (accs cdt) in
  let* eb =
    let* n = int_range 0 1 in
    extents n
  in
  let* ek =
    let* n = int_range 0 2 in
    extents n
  in
  let* fa =
    let* n = int_range 0 1 in
    extents n
  in
  let* fb =
    let* n = int_range 0 1 in
    extents n
  in
  let* ra = map Array.of_list (permutation (roles (Array.length fa) eb ek)) in
  let* rb = map Array.of_list (permutation (roles (Array.length fb) eb ek)) in
  let* ta = bool in
  let* tb = bool in
  let c =
    {
      cdt;
      acc;
      eb;
      ek;
      fa;
      fb;
      ra;
      rb;
      ta;
      tb;
      araws = [||];
      braws = [||];
      init = None;
    }
  in
  let raws s =
    array
      ~size:(constant (raws_of cdt (numel s)))
      (raw ~spread:2 ~subnormal:1 cdt)
  in
  let* araws = raws (ashape c fa ra) in
  let* braws = raws (ashape c fb rb) in
  let+ init =
    frequency [ (1, constant None); (1, map Option.some (raws (out_shape c))) ]
  in
  { c with araws; braws; init }

let contraction dts accs = Gen.with_pp pp_contraction (contraction dts accs)

(* An operand of shape [s], read through a transposed view where [t]. *)
let contracted (type v s) (dt : (v, s) D.t) s t raws : (v, s, Nx.host) Nx.t =
  if not t then value dt s raws
  else Nx.transpose (value dt (Array.of_list (List.rev (Array.to_list s))) raws)

let contract_results (type v s) (dt : (v, s) D.t) c =
  let a = contracted dt (ashape c c.fa c.ra) c.ta c.araws in
  let b = contracted dt (ashape c c.fb c.rb) c.tb c.braws in
  let init = Option.map (value dt (out_shape c)) c.init in
  let batch = pairs c (fun i -> B i) (Array.length c.eb) in
  let contracting = pairs c (fun i -> C i) (Array.length c.ek) in
  let op () =
    P.eval ~by:"t"
      (P.Contract
         {
           order = P.contract_order ~batch ~contracting a b;
           batch;
           contracting;
           acc = c.acc;
           out = dt;
           a;
           b;
           init;
         })
  in
  (a, b, init, Nx.to_array (op ()), Lowered.run op)

let decode dims i =
  let c = Array.make (Array.length dims) 0 and rest = ref i in
  for a = Array.length dims - 1 downto 0 do
    c.(a) <- !rest mod dims.(a);
    rest := !rest / dims.(a)
  done;
  c

let linear dims coords =
  let i = ref 0 in
  Array.iteri (fun a k -> i := (!i * dims.(a)) + k) coords;
  !i

(* [magnitudes c ~term ~start] is, for each result, [start o] plus [term ia ib]
   over its products of [a]'s element [ia] and [b]'s [ib]. *)
let magnitudes c ~term ~start =
  let sa = ashape c c.fa c.ra and sb = ashape c c.fb c.rb in
  let nb = Array.length c.eb and nfa = Array.length c.fa in
  Array.init
    (numel (out_shape c))
    (fun o ->
      let oc = decode (out_shape c) o in
      let at f k = function
        | B i -> oc.(i)
        | F i -> oc.(nb + f + i)
        | C i -> k.(i)
      in
      let sum = ref (start o) in
      for k = 0 to numel c.ek - 1 do
        let kc = decode c.ek k in
        let ia = linear sa (Array.map (at 0 kc) c.ra) in
        let ib = linear sb (Array.map (at nfa kc) c.rb) in
        sum := !sum +. term ia ib
      done;
      !sum)

(* Each part's [|init| + Σ|x·y|] over its products of parts: [ar·br] and [ai·bi]
   in the real part, [ar·bi] and [ai·br] in the imaginary part. *)
let part_magnitudes c xs ys init =
  let start p o =
    match init with None -> 0. | Some i -> Float.abs (List.nth i p).(o)
  in
  let prod x i y j = Float.abs ((List.nth xs x).(i) *. (List.nth ys y).(j)) in
  match xs with
  | [ _ ] -> [ magnitudes c ~term:(fun i j -> prod 0 i 0 j) ~start:(start 0) ]
  | _ ->
      [
        magnitudes c
          ~term:(fun i j -> prod 0 i 0 j +. prod 1 i 1 j)
          ~start:(start 0);
        magnitudes c
          ~term:(fun i j -> prod 0 i 1 j +. prod 1 i 0 j)
          ~start:(start 1);
      ]

(* Integers give eager's bits, floats hold the bound. With an init, a zero
   result's sign follows the order of summation, which the bound leaves open:
   [+0] added last, or a [-0] init first. *)
let contract_check c =
  let (D.Any dt) = c.cdt in
  let a, b, init, e, l = contract_results dt c in
  match D.kind dt with
  | D.Float | D.Complex ->
      let ps = parts dt in
      let mags =
        part_magnitudes c
          (ps (Nx.to_array a))
          (ps (Nx.to_array b))
          (Option.map (fun i -> ps (Nx.to_array i)) init)
      in
      let terms = numel c.ek * List.length mags in
      let g = gamma (carrier_roundoff c.acc) (terms + 1) in
      let round = rounding_of dt in
      List.iteri
        (fun p (e, l) ->
          let mags = List.nth mags p in
          Array.iteri
            (fun o e ->
              near ~zeros:(c.init = None)
                ~bound:((2. *. g *. mags.(o)) +. round e l.(o))
                e l.(o))
            e)
        (List.combine (ps e) (ps l))
  | _ -> same_bits dt e l

let cover_contraction c =
  cover "a batch axis" (c.eb <> [||]);
  cover "two contracted axes" (Array.length c.ek = 2);
  cover "no contracted axis" (c.ek = [||]);
  cover "no contracted term" (numel c.ek = 0);
  cover "an init" (c.init <> None);
  cover "a transposed operand" (c.ta || c.tb)

(* A contraction of no term from an init, every run. *)
let empty dt =
  let c =
    {
      cdt = dt;
      acc = dt;
      eb = [||];
      ek = [| 0 |];
      fa = [| 2 |];
      fb = [||];
      ra = [| F 0; C 0 |];
      rb = [| C 0 |];
      ta = false;
      tb = false;
      araws = [||];
      braws = [||];
      init = None;
    }
  in
  [ c; { c with init = Some (Array.make (raws_of dt 2) 0x80000000L) } ]

let int_contractions =
  let accs d = [ d; D.Any D.Int64 ] in
  prop ~count:12 ~examples:(empty (D.Any D.Int32))
    "integer contractions wrap as eager's"
    (contraction (weigh 1 ints) accs)
    (fun c ->
      cover_contraction c;
      cover "a wider accumulator" (c.acc <> c.cdt);
      contract_check c)

let float_contractions =
  let accs (D.Any d as dt) =
    match d with
    | D.Float64 -> [ dt ]
    | D.Float32 -> [ dt; D.Any D.Float64 ]
    | _ -> [ D.Any D.Float32 ]
  in
  prop ~count:12 ~examples:(empty (D.Any D.Float32))
    "float contractions hold their bound"
    (contraction (weigh 1 floats) accs)
    (fun c ->
      cover_contraction c;
      contract_check c)

let complex_contractions =
  prop ~count:6
    ~examples:(empty (D.Any D.Complex64))
    "complex contractions hold their bound"
    (contraction (weigh 1 complexes) (fun d -> [ d ]))
    (fun c ->
      cover "an init" (c.init <> None);
      contract_check c)

(* Cases *)

let f32 = Nx.create D.Float32

(* A max-pooling window: the padded load's fill enters the windows it
   overlaps. *)
let pooled m fill =
  let x = f32 [| 6 |] [| 1.; Float.nan; -0.; 0.; 3.; Float.neg_infinity |] in
  let pad =
    {
      K.lo = Iarray.of_list [ 1 ];
      hi = Iarray.of_list [ 1 ];
      interior = Iarray.of_list [ 0 ];
      windows =
        Iarray.of_list
          [ { Nx_array.Move.axis = 0; size = 3; step = 2; dilation = 1 } ];
    }
  in
  let op () =
    fst
      (P.eval ~by:"t"
         (P.Reduce
            {
              layout = Nx_array.Layout.contiguous [| 3; 3 |];
              axes = Iarray.of_list [ 1 ];
              prog = identity D.Float32;
              reductions = [ Monoid (m, 0, D.Float32) ];
              loads = Iarray.of_list [ P.Padded { x; fill; pad } ];
            }))
  in
  same_bits D.Float32 (Nx.to_array (op ())) (Lowered.run op)

(* Three reductions of one output in one loop. *)
let several () =
  let x = Nx.create D.Int16 [| 2; 3 |] [| 32767; 1; -2; -32768; 0; 5 |] in
  let op k () =
    let s, (mx, (mn, ())) =
      P.eval ~by:"t"
        (P.Reduce
           {
             layout = Nx_array.Layout.contiguous [| 2; 3 |];
             axes = Iarray.of_list [ 1 ];
             prog = identity D.Int16;
             reductions =
               [
                 Monoid (Sum, 0, D.Int16);
                 Monoid (Max, 0, D.Int16);
                 Monoid (Min, 0, D.Int16);
               ];
             loads = Iarray.of_list [ P.Plain x ];
           })
    in
    List.nth [ s; mx; mn ] k
  in
  List.iter
    (fun k -> equal (array int) (Nx.to_array (op k ())) (Lowered.run (op k)))
    [ 0; 1; 2 ]

(* Expansions: the operations lower through core reductions and maps. *)

let nan_payload = Int32.float_of_bits 0xffa00001l

let arg e () =
  let x =
    f32 [| 2; 4 |] [| 1.; -0.; 0.; 1.; 2.; nan_payload; Float.nan; 7. |]
  in
  let op () =
    fst
      (P.eval ~by:"t"
         (P.Reduce
            {
              layout = Nx_array.Layout.contiguous [| 2; 4 |];
              axes = Iarray.of_list [ 1 ];
              prog = identity D.Float32;
              reductions = [ Arg (e, 0, D.Float32) ];
              loads = Iarray.of_list [ P.Plain x ];
            }))
  in
  same_bits D.Float32
    (Nx.to_array (fst (op ())))
    (Lowered.run (fun () -> fst (op ())));
  equal (array int64)
    (Nx.to_array (snd (op ())))
    (Lowered.run (fun () -> snd (op ())))

(* Float32 sums of 4 terms below 8: their orders differ by at most [2 γ(3) 32 ≈
   1.2e-5] in the mean, more through the variance's squares. *)
let moments () =
  let x = f32 [| 2; 4 |] [| 1.5; -2.; 0.25; 7.; 3.; 3.; 3.; -1. |] in
  let op () =
    fst
      (P.eval ~by:"t"
         (P.Reduce
            {
              layout = Nx_array.Layout.contiguous [| 2; 4 |];
              axes = Iarray.of_list [ 1 ];
              prog = identity D.Float32;
              reductions = [ Moments (0, D.Float32) ];
              loads = Iarray.of_list [ P.Plain x ];
            }))
  in
  equal
    (array (float 1e-4))
    (Nx.to_array (fst (op ())))
    (Lowered.run (fun () -> fst (op ())));
  equal
    (array (float 1e-4))
    (Nx.to_array (snd (op ())))
    (Lowered.run (fun () -> snd (op ())))

(* Float32 exponentials and a logarithm of a sum of 4 terms: a few ulps. *)
let logsumexp () =
  let x =
    f32 [| 2; 4 |] [| 1.5; -2.; 0.25; 7.; Float.neg_infinity; 3.; 3.; -1. |]
  in
  let op () =
    fst
      (P.eval ~by:"t"
         (P.Reduce
            {
              layout = Nx_array.Layout.contiguous [| 2; 4 |];
              axes = Iarray.of_list [ 1 ];
              prog = identity D.Float32;
              reductions = [ Monoid (Logsumexp, 0, D.Float32) ];
              loads = Iarray.of_list [ P.Plain x ];
            }))
  in
  equal (array (float 1e-5)) (Nx.to_array (op ())) (Lowered.run op)

(* A boolean is its byte: the bytes a reduction or a scan of booleans leaves are
   eager's. *)
let bool_bytes ~scan m () =
  let x =
    Nx.bitcast D.Bool
      (Nx.create D.Uint8 [| 3; 3 |] [| 2; 0; 3; 0; 0; 0; 4; 5; 255 |])
  in
  let op () =
    let y =
      if scan then P.eval ~by:"t" (scan_op m 1 x None)
      else fst (P.eval ~by:"t" (reduce_op m [ 1 ] x))
    in
    Nx.bitcast D.Uint8 y
  in
  equal (array int) (Nx.to_array (op ())) (Lowered.run op)

let cases =
  group "cases"
    [
      test "booleans' any keeps eager's bytes" (bool_bytes ~scan:false Max);
      test "booleans' all keeps eager's bytes" (bool_bytes ~scan:false Min);
      test "booleans' running any keeps eager's bytes"
        (bool_bytes ~scan:true Max);
      test "a max-pooling window gives eager's bits" (fun () ->
          pooled Max Float.neg_infinity);
      test "a min-pooling window gives eager's bits" (fun () ->
          pooled Min Float.infinity);
      test "several reductions of one loop" several;
      test "an argmax gives eager's bits and positions" (arg Max);
      test "an argmin gives eager's bits and positions" (arg Min);
      test "moments within float32 sums' differences" moments;
      test "a logsumexp within a few ulps" logsumexp;
    ]

let () =
  exit
    (run "Lower_reduce"
       [
         group "reductions"
           [ extremes; int_folds; float_sums; float_products; complex_sums ];
         group "scans" [ extreme_scans; int_scans; float_scans; complex_scans ];
         group "contractions"
           [ int_contractions; float_contractions; complex_contractions ];
         cases;
       ])
