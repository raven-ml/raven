(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Gathers, scatters, folds, assemblies and sorts lowered, against nx's eager
   result on the same operands: bit for bit where the lowering moves elements
   or computes exactly, within the bound Lower_index states elsewhere. *)

open Windtrap
module D = Nx_array.Dtype
module P = Nx.Prim
module K = Nx_kernel

(* Elements

   A case's elements are int64 raws: a float's bits, an integer's value modulo
   its width, a boolean's low bit, and a complex number's two parts as float64
   bits. *)

let numel s = Array.fold_left ( * ) 1 s

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

let raws_of (D.Any dt) n = if D.is D.Complex dt then 2 * n else n
let name (D.Any d) = D.name d
let is_float (D.Any d) = D.is D.Float d
let is_complex (D.Any d) = D.is D.Complex d
let is_bool (D.Any d) = D.is D.Boolean d
let is_narrow (D.Any d) = D.is D.Float d && D.bits d < 32

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

let negative (type s) (dt : (float, s) D.t) r =
  let e, f = format dt in
  Int64.logand r (Int64.shift_left 1L (e + f)) <> 0L

(* The classes of a case's float raws, a complex one's parts as float64. *)
let classes (D.Any dt) raws =
  let of_ f = Array.to_list (Array.map f raws) in
  match D.kind dt with
  | D.Float -> of_ (fun r -> (float_class dt r, negative dt r))
  | D.Complex ->
      of_ (fun r -> (float_class D.Float64 r, negative D.Float64 r))
  | _ -> []

let cover_floats d raws =
  let cs = classes d raws in
  cover "a NaN" (List.exists (fun (c, _) -> c = Nan) cs);
  cover "both zeros"
    (List.mem (Zero, true) cs && List.mem (Zero, false) cs);
  cover "an infinity" (List.exists (fun (c, _) -> c = Inf) cs)

(* Generators *)

let among xs = Gen.one_of (List.map Gen.constant xs)

(* A float of [dt] within [2^±4], with zeros, infinities and NaNs of any sign
   and payload, and subnormals. *)
let float_raw (type s) (dt : (float, s) D.t) =
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
        (3, constant Zero);
        (1, constant Inf);
        (2, constant Nan);
        (1, constant Subnormal);
      ]
  in
  let* sign = bool in
  let* ex = int_range (max 1 (bias - 4)) (min (top - 1) (bias + 4)) in
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

let raw (D.Any dt) =
  match D.kind dt with
  | D.Float -> float_raw dt
  | D.Complex -> float_raw D.Float64
  | _ -> int_raw dt

(* [raws d n] draws the raws of [n] elements of [d], from a pool of three raws
   half the time, so that equal elements are frequent. *)
let raws d n =
  let open Gen in
  let n = raws_of d n in
  let* pooled = bool in
  if not pooled then array ~size:(constant n) (raw d)
  else
    let* pool = list ~size:(constant 3) (raw d) in
    array ~size:(constant n) (among pool)

let extent =
  Gen.frequency
    [ (1, Gen.constant 0); (2, Gen.constant 1); (6, Gen.int_range 2 4) ]

let shape =
  let open Gen in
  let* r = int_range 1 3 in
  map Array.of_list (list ~size:(constant r) extent)

let all_dtypes = Gen.one_of (List.map Gen.constant D.all)

(* Views *)

type view = Plain | Transposed | Flipped | Broadcast

let views = [ Plain; Transposed; Flipped; Broadcast ]

let view_name = function
  | Plain -> "plain"
  | Transposed -> "transposed"
  | Flipped -> "flipped"
  | Broadcast -> "broadcast"

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

let operand (type v s) (dt : (v, s) D.t) view s raws : (v, s, Nx.host) Nx.t =
  viewed view s (value dt (base view s) raws)

let operand_gen d view s = raws d (numel (base view s))

(* [computed x] is [x] where its index's first coordinate is below its last,
   [x] flipped elsewhere: a value of [x]'s elements that a kernel computes. *)
let computed (type v s) (x : (v, s, _) Nx.t) =
  let dt = Nx.dtype x and s = Nx.shape x in
  let r = Array.length s in
  let prog =
    K.Prog.v
      ~ins:(Iarray.of_list [ D.Any dt; D.Any dt ])
      (Iarray.of_list
         K.Prog.
           [
             In 0;
             In 1;
             Coord 0;
             Coord (r - 1);
             Op2 (Compare Less, 2, 3);
             Op3 (Where, 4, 0, 1);
           ])
      ~outs:(Iarray.of_list [ 5 ])
  in
  let y, () =
    P.eval ~by:"t"
      (P.Map
         {
           layout = Nx_array.Layout.contiguous (Iarray.of_array s);
           prog;
           outs = [ dt ];
           loads = Iarray.of_list [ P.Plain x; P.Plain (Nx.flip x) ];
         })
  in
  y

(* Comparisons *)

let hex dt v =
  let b = K.Prog.bits dt v in
  let n = String.length b in
  String.concat ""
    (List.init n (fun i -> Printf.sprintf "%02x" (Char.code b.[n - 1 - i])))

let same_bits dt eager lowered =
  equal (array string) (Array.map (hex dt) eager) (Array.map (hex dt) lowered)

let is_nan (type v s) (dt : (v, s) D.t) (v : v) =
  match D.kind dt with
  | D.Float -> Float.is_nan v
  | D.Complex -> Float.is_nan v.Complex.re || Float.is_nan v.im
  | _ -> false

(* [same_values] is [same_bits] but that a NaN result may be any NaN. *)
let same_values dt eager lowered =
  let shown v = if is_nan dt v then "nan" else hex dt v in
  equal (array string) (Array.map shown eager) (Array.map shown lowered)

let float_kind x =
  if Float.is_nan x then "nan"
  else if x = Float.infinity then "+inf"
  else if x = Float.neg_infinity then "-inf"
  else if x = 0. && Float.sign_bit x then "-0"
  else if x = 0. then "+0"
  else "number"

(* [near ~bound e l] checks [l] against [e]: a NaN where [e] is one, the same
   infinity, the same zero, and otherwise within [bound], inclusive. *)
let near ~bound e l =
  let msg = Printf.sprintf "%h against %h" e l in
  let k = float_kind e and kl = float_kind l in
  let number k = k = "number" || k = "+0" || k = "-0" in
  if not (number k && number kl && (k = "number" || kl = "number")) then
    equal ~msg string k kl
  else less ~msg float_exact ~than:(Float.succ bound) (Float.abs (e -. l))

(* [γ(k) = k u / (1 - k u)] for the carrier's unit roundoff [u]. *)
let gamma (D.Any dt) k =
  let u =
    match dt with
    | D.Float64 | D.Complex128 -> Float.ldexp 1. (-53)
    | _ -> Float.ldexp 1. (-24)
  in
  let k = Float.of_int (max 0 k) in
  k *. u /. (1. -. (k *. u))

(* A narrow float's result rounds once more on each side: its unit roundoff
   of each and half a subnormal step. *)
let rounding (type s) (dt : (float, s) D.t) e l =
  match dt with
  | D.Float64 | D.Float32 -> 0.
  | _ ->
      let ex, f = format dt in
      let tiny = Float.ldexp 1. (2 - (1 lsl (ex - 1)) - f) in
      (Float.ldexp 1. (-(f + 1)) *. (Float.abs e +. Float.abs l)) +. tiny

(* The float or complex elements [xs] of [dt] as float arrays: [[xs]], or their
   real parts then their imaginary parts. *)
let parts (type v s) (dt : (v, s) D.t) (xs : v array) : float array list =
  match D.kind dt with
  | D.Float -> [ xs ]
  | D.Complex ->
      [
        Array.map (fun z -> z.Complex.re) xs;
        Array.map (fun z -> z.Complex.im) xs;
      ]
  | _ -> invalid_arg "parts: not a float"

let parts64 (type v s) (x : (v, s, Nx.host) Nx.t) =
  match D.kind (Nx.dtype x) with
  | D.Float -> [ Nx.to_array (Nx.cast D.Float64 x) ]
  | D.Complex ->
      [ Nx.to_array (Nx.real D.Float64 x); Nx.to_array (Nx.imag D.Float64 x) ]
  | _ -> invalid_arg "parts64: not a float"

let rounding_of (type v s) (dt : (v, s) D.t) =
  match D.kind dt with D.Float -> rounding dt | _ -> fun _ _ -> 0.

(* [sums dt ~mags ~terms e l] checks each part of the eager and the lowered
   float sums [e] and [l] against each other: each is within [γ(n) m] of the
   exact sum, [n + 1] the terms of an element and [m] the sum of their
   magnitudes, each part's in [mags]. *)
let sums (type v s) (dt : (v, s) D.t) ~mags ~terms (e : v array) (l : v array)
    =
  let round = rounding_of dt in
  let g n = gamma (D.Any dt) n in
  List.iter2
    (fun (e, l) mags ->
      Array.iteri
        (fun o e ->
          let l = l.(o) in
          let bound = (2. *. g terms.(o) *. mags.(o)) +. round e l in
          near ~bound e l)
        e)
    (List.combine (parts dt e) (parts dt l))
    mags

let ints s = String.concat ";" (List.map string_of_int (Array.to_list s))

let raws_text a =
  String.concat ";" (Array.to_list (Array.map (Printf.sprintf "%Lx") a))

let dtype_covers ?(booleans = true) d =
  cover "a float" (is_float d);
  cover "a complex number" (is_complex d);
  if booleans then cover "a boolean" (is_bool d);
  cover "an integer" (not (is_float d || is_complex d || is_bool d))

(* Positions

   A position along an axis of [n] elements: inside it, its neighbours outside
   it, or an int64 extreme. *)

let position n =
  let n = Int64.of_int n in
  Gen.frequency
    [
      (8, Gen.map Int64.of_int (Gen.int_range 0 (max 0 (Int64.to_int n - 1))));
      (2, among [ -1L; n; Int64.add n 1L ]);
      (1, among [ Int64.min_int; Int64.max_int ]);
    ]

let positions n m = Gen.array ~size:(Gen.constant m) (position n)
let outside n p = p < 0L || p >= Int64.of_int n

(* Examples

   Every run takes one example of each dtype, its elements cycling through
   [specials]: both zeros, NaNs of both signs, a signalling one among them,
   infinities, numbers and their repeats; an integer's extremes. *)

let float_specials (type s) (dt : (float, s) D.t) =
  let e, f = format dt in
  let top = (1 lsl e) - 1 and bias = (1 lsl (e - 1)) - 1 in
  let mask = Int64.pred (Int64.shift_left 1L f) in
  let field sign ex fr =
    let s = if sign then Int64.shift_left 1L (e + f) else 0L in
    Int64.(logor s (logor (shift_left (of_int ex) f) fr))
  in
  let nan sign fr =
    match dt with
    | D.Float8_e4m3fn -> field sign top mask
    | D.Float4_e2m1fn -> field sign top 1L
    | _ -> field sign top fr
  in
  let quiet = Int64.logor (Int64.shift_left 1L (f - 1)) 1L in
  let one = field false bias 0L in
  [
    field false 0 0L;
    field true 0 0L;
    one;
    nan false 1L;
    field true bias 1L;
    field false top 0L;
    field true top 0L;
    nan true quiet;
    one;
    field true 0 0L;
  ]

let specials (D.Any dt as d) =
  match D.kind dt with
  | D.Float -> float_specials dt
  | D.Complex ->
      let b = Int64.bits_of_float in
      List.concat_map
        (fun (re, im) -> [ b re; b im ])
        [
          (1., 2.);
          (1., 2.);
          (-0., 0.);
          (0., -0.);
          (Float.nan, 0.);
          (Float.infinity, 1.);
          (1., Int64.float_of_bits 0xfff4000000000001L);
          (Float.neg_infinity, -0.);
        ]
  | _ when is_bool d -> [ 0L; 1L; 1L; 0L ]
  | _ ->
      let w = width dt in
      let lo, hi =
        match (w, D.is D.Signed dt) with
        | 64, true -> (Int64.min_int, Int64.max_int)
        | 64, false -> (0L, -1L)
        | w, true ->
            ( Int64.of_int (-(1 lsl (w - 1))),
              Int64.of_int ((1 lsl (w - 1)) - 1) )
        | w, false -> (0L, Int64.of_int ((1 lsl w) - 1))
      in
      [ lo; hi; 0L; 1L; -1L; 0L ]

(* The raws of [n] elements of [d], cycling through its specials from [i]. *)
let cycled ?(i = 0) d n =
  let sp = Array.of_list (specials d) in
  Array.init (raws_of d n) (fun j -> sp.((i + j) mod Array.length sp))

(* Positions along an axis of [n], with repeats and three outside it. *)
let example_positions n m =
  let ps =
    Int64.
      [| 0L; 0L; 2L; -1L; of_int n; 1L; max_int; of_int (n - 1) |]
  in
  Array.init m (fun j -> ps.(j mod Array.length ps))

(* Gathers *)

type gather = {
  g_dt : D.any;
  g_shape : int array;
  g_view : view;
  g_axis : int;
  g_idx : int array;  (** The positions' shape. *)
  g_pos : int64 array;
  g_raws : int64 array;
  g_computed : bool;  (** Whether a kernel computes the operand. *)
}

let pp_gather ppf c =
  Format.fprintf ppf "%s [%s] %s%s axis %d idx [%s] pos [%s] raws [%s]"
    (name c.g_dt) (ints c.g_shape) (view_name c.g_view)
    (if c.g_computed then " computed" else "")
    c.g_axis
    (ints c.g_idx) (raws_text c.g_pos) (raws_text c.g_raws)

let gather_case =
  let open Gen in
  let* d = all_dtypes in
  let* s = shape in
  let* view = among views in
  let* axis = int_range 0 (Array.length s - 1) in
  let* m = extent in
  let t = Array.mapi (fun i n -> if i = axis then m else n) s in
  let* pos = positions s.(axis) (numel t) in
  let* computed = bool in
  let+ raws = operand_gen d view s in
  {
    g_computed = computed;
    g_dt = d;
    g_shape = s;
    g_view = view;
    g_axis = axis;
    g_idx = t;
    g_pos = pos;
    g_raws = raws;
  }

let gather_example i d s axis m =
  let t = Array.mapi (fun j n -> if j = axis then m else n) s in
  let view = List.nth views (i mod 4) in
  {
    g_dt = d;
    g_shape = s;
    g_view = view;
    g_axis = axis;
    g_idx = t;
    g_pos = example_positions s.(axis) (numel t);
    g_raws = cycled ~i d (numel (base view s));
    g_computed = i mod 3 = 1;
  }

let gather_examples =
  gather_example 0 (D.Any D.Float32) [| 3; 4 |] 1 0
  :: gather_example 0 (D.Any D.Float32) [| 1 |] 0 1
  :: List.mapi (fun i d -> gather_example i d [| 3; 4 |] (i mod 2) 5) D.all

let gathers =
  prop ~count:40 ~examples:gather_examples "gathers give eager's bits"
    (Gen.with_pp pp_gather gather_case)
    (fun c ->
      let (D.Any dt) = c.g_dt in
      dtype_covers c.g_dt;
      cover_floats c.g_dt c.g_raws;
      cover "a strided or broadcast view" (c.g_view <> Plain);
      cover "a computed operand" c.g_computed;
      cover "a position outside the axis"
        (Array.exists (outside c.g_shape.(c.g_axis)) c.g_pos);
      cover "an empty result" (numel c.g_idx = 0);
      cover "a one-element result" (numel c.g_idx = 1);
      let x = operand dt c.g_view c.g_shape c.g_raws in
      let idx = Nx.create D.Int64 c.g_idx c.g_pos in
      let op () =
        let x = if c.g_computed then computed x else x in
        P.eval ~by:"t" (P.Gather { axis = c.g_axis; idx; x })
      in
      same_bits dt (Nx.to_array (op ())) (Lowered.run op))

(* Scatters *)

type scatter = {
  s_dt : D.any;
  s_combine : K.combine;
  s_unique : bool;
  s_shape : int array;
  s_view : view;
  s_axis : int;
  s_idx : int array;
  s_pos : int64 array;
  s_into : int64 array;
  s_updates : int64 array;
}

let combine_name : K.combine -> string = function
  | Set -> "set"
  | Add -> "add"
  | Max -> "max"
  | Min -> "min"

let pp_scatter ppf c =
  Format.fprintf ppf
    "%s %s%s [%s] %s axis %d idx [%s] pos [%s] into [%s] updates [%s]"
    (combine_name c.s_combine) (name c.s_dt)
    (if c.s_unique then " unique" else "")
    (ints c.s_shape) (view_name c.s_view) c.s_axis (ints c.s_idx)
    (raws_text c.s_pos) (raws_text c.s_into) (raws_text c.s_updates)

(* [distinct n t axis pos] is [pos] with each position inside the axis that an
   earlier update of its slice holds moved outside it. *)
let distinct n t axis pos =
  let pos = Array.copy pos in
  let k = t.(axis) in
  let inner = numel (Array.sub t (axis + 1) (Array.length t - axis - 1)) in
  for o = 0 to (Array.length pos / max 1 (k * inner)) - 1 do
    for i = 0 to inner - 1 do
      let seen = Hashtbl.create 8 in
      for a = 0 to k - 1 do
        let j = (((o * k) + a) * inner) + i in
        let p = pos.(j) in
        if not (outside n p) then
          if Hashtbl.mem seen p then pos.(j) <- Int64.of_int (n + a)
          else Hashtbl.add seen p ()
      done
    done
  done;
  pos

let scatter_case dts combines =
  let open Gen in
  let* d = among dts in
  let* combine = among combines in
  let* unique = bool in
  let* s = shape in
  let* view = among views in
  let* axis = int_range 0 (Array.length s - 1) in
  let* m = extent in
  let t = Array.mapi (fun i n -> if i = axis then m else n) s in
  let n = s.(axis) in
  let* pos = positions n (numel t) in
  let pos = if unique then distinct n t axis pos else pos in
  let* into = operand_gen d view s in
  let+ updates = raws d (numel t) in
  {
    s_dt = d;
    s_combine = combine;
    s_unique = unique;
    s_shape = s;
    s_view = view;
    s_axis = axis;
    s_idx = t;
    s_pos = pos;
    s_into = into;
    s_updates = updates;
  }

(* Whether two updates of [c] land on one element. *)
let collide c =
  let t = c.s_idx and axis = c.s_axis in
  let n = c.s_shape.(axis) and k = t.(axis) in
  let inner = numel (Array.sub t (axis + 1) (Array.length t - axis - 1)) in
  let found = ref false in
  for o = 0 to (Array.length c.s_pos / max 1 (k * inner)) - 1 do
    for i = 0 to inner - 1 do
      let seen = Hashtbl.create 8 in
      for a = 0 to k - 1 do
        let p = c.s_pos.((((o * k) + a) * inner) + i) in
        if not (outside n p) then
          if Hashtbl.mem seen p then found := true else Hashtbl.add seen p ()
      done
    done
  done;
  !found

let scattered (type v s) (dt : (v, s) D.t) c =
  let into = operand dt c.s_view c.s_shape c.s_into in
  let updates = value dt c.s_idx c.s_updates in
  let idx = Nx.create D.Int64 c.s_idx c.s_pos in
  let op () =
    P.eval ~by:"t"
      (P.Scatter
         {
           combine = c.s_combine;
           unique = c.s_unique;
           axis = c.s_axis;
           idx;
           updates;
           into;
         })
  in
  (into, updates, Nx.to_array (op ()), Lowered.run op)

let cover_scatter ?booleans c =
  dtype_covers ?booleans c.s_dt;
  cover_floats c.s_dt (Array.append c.s_into c.s_updates);
  cover "a strided or broadcast view" (c.s_view <> Plain);
  cover "a position outside the axis"
    (Array.exists (outside c.s_shape.(c.s_axis)) c.s_pos);
  cover "two updates on one element" (collide c);
  cover "unique" c.s_unique;
  cover "no update" (numel c.s_idx = 0)

let scatter_example i d combine =
  let s = [| 3; 4 |] and axis = i mod 2 in
  let t = Array.mapi (fun j n -> if j = axis then 6 else n) s in
  let unique = i mod 4 = 1 in
  let pos = example_positions s.(axis) (numel t) in
  let view = List.nth views (i mod 4) in
  {
    s_dt = d;
    s_combine = combine;
    s_unique = unique;
    s_shape = s;
    s_view = view;
    s_axis = axis;
    s_idx = t;
    s_pos = (if unique then distinct s.(axis) t axis pos else pos);
    s_into = cycled ~i d (numel (base view s));
    s_updates = cycled ~i:(i + 3) d (numel t);
  }

(* Each dtype, the combines in turn, and an example without updates. *)
let scatter_examples dts combines =
  let none = scatter_example 1 (List.hd dts) (List.hd combines) in
  { none with s_idx = [| 3; 0 |]; s_pos = [||]; s_updates = [||] }
  :: List.mapi
       (fun i d ->
         scatter_example i d (List.nth combines (i mod List.length combines)))
       dts

let scatters_moved =
  prop ~count:40
    ~examples:(scatter_examples D.all [ Set; Max; Min ])
    "a scatter by Set, Max or Min gives eager's bits"
    (Gen.with_pp pp_scatter (scatter_case D.all [ Set; Max; Min ]))
    (fun c ->
      let (D.Any dt) = c.s_dt in
      cover_scatter c;
      List.iter
        (fun k -> cover (combine_name k) (c.s_combine = k))
        [ Set; Max; Min ];
      let _, _, e, l = scattered dt c in
      if c.s_combine = Set then same_bits dt e l else same_values dt e l)

let addable = List.filter (fun d -> not (is_bool d)) D.all

(* The number of updates that land on each element of [c]'s result and the sum
   of the magnitudes of its terms, each part's. *)
let terms c into updates =
  let s = c.s_shape and t = c.s_idx and axis = c.s_axis in
  let n = s.(axis) and k = t.(axis) in
  let inner = numel (Array.sub t (axis + 1) (Array.length t - axis - 1)) in
  let count = Array.make (numel s) 0 in
  let mags = List.map (Array.map Float.abs) (parts64 into) in
  let us = parts64 updates in
  Array.iteri
    (fun j p ->
      if not (outside n p) then begin
        let o = j / (k * inner) and i = j mod inner in
        let target = (((o * n) + Int64.to_int p) * inner) + i in
        count.(target) <- count.(target) + 1;
        List.iter2
          (fun m u -> m.(target) <- m.(target) +. Float.abs u.(j))
          mags us
      end)
    c.s_pos;
  (count, mags)

let scatter_adds =
  prop ~count:40
    ~examples:(scatter_examples addable [ Add ])
    "a scatter by Add wraps integers and holds its float bound"
    (Gen.with_pp pp_scatter (scatter_case addable [ Add ]))
    (fun c ->
      let (D.Any dt) = c.s_dt in
      cover_scatter ~booleans:false c;
      cover "a narrow float" (is_narrow c.s_dt);
      let into, updates, e, l = scattered dt c in
      match D.kind dt with
      | D.Float | D.Complex ->
          let terms, mags = terms c into updates in
          sums dt ~mags ~terms e l
      | _ -> same_bits dt e l)

(* Folds *)

module M = Nx_array.Move

type fold = {
  f_dt : D.any;
  f_shape : int array;
  f_pad : K.pad;
  f_padded : int array;
  f_raws : int64 array;
}

let pp_fold ppf c =
  let a x = ints (Iarray.to_array x) in
  let w (w : M.window) =
    Printf.sprintf "%d:%d/%d*%d" w.axis w.size w.step w.dilation
  in
  Format.fprintf ppf
    "%s [%s] lo [%s] hi [%s] interior [%s] windows [%s] raws [%s]"
    (name c.f_dt) (ints c.f_shape) (a c.f_pad.lo) (a c.f_pad.hi)
    (a c.f_pad.interior)
    (String.concat ";" (List.map w (Iarray.to_list c.f_pad.windows)))
    (raws_text c.f_raws)

(* A padding of a shape of rank [r]: ends that pad or crop, interior padding,
   and windows on some axes half the time. *)
let pad_gen r =
  let open Gen in
  let ends = map Iarray.of_list (list ~size:(constant r) (int_range (-2) 2)) in
  let* lo = ends in
  let* hi = ends in
  let* interior =
    map Iarray.of_list (list ~size:(constant r) (int_range 0 2))
  in
  let* windowed = list ~size:(constant r) bool in
  let* sizes = list ~size:(constant r) (int_range 1 3) in
  let* steps = list ~size:(constant r) (int_range 1 2) in
  let+ dilations = list ~size:(constant r) (int_range 1 2) in
  let windows =
    List.filteri (fun i _ -> List.nth windowed i) (List.init r Fun.id)
    |> List.map (fun axis ->
        {
          M.axis;
          size = List.nth sizes axis;
          step = List.nth steps axis;
          dilation = List.nth dilations axis;
        })
  in
  { K.lo; hi; interior; windows = Iarray.of_list windows }

(* A float dtype's fraction bits. *)
let fraction (D.Any dt) =
  match D.kind dt with D.Float -> Some (snd (format dt)) | _ -> None

let fold_case dts =
  let open Gen in
  let* d = among dts in
  let* s = shape in
  let* pad = pad_gen (Array.length s) in
  match K.Shape.padded pad s with
  | Error _ -> reject ()
  | Ok out ->
      let+ raws = raws d (numel out) in
      { f_dt = d; f_shape = s; f_pad = pad; f_padded = out; f_raws = raws }

let folded (type v s) (x : (v, s, Nx.host) Nx.t) c =
  P.eval ~by:"t"
    (P.Fold { shape = Iarray.of_array c.f_shape; pad = c.f_pad; x })

let cover_fold c =
  let pad = c.f_pad in
  let exists f a = Iarray.exists f a in
  dtype_covers ~booleans:false c.f_dt;
  cover_floats c.f_dt c.f_raws;
  cover "a cropping end"
    (exists (fun n -> n < 0) pad.lo || exists (fun n -> n < 0) pad.hi);
  cover "a padding end"
    (exists (fun n -> n > 0) pad.lo || exists (fun n -> n > 0) pad.hi);
  cover "interior padding" (exists (fun n -> n > 0) pad.interior);
  cover "overlapping windows"
    (exists (fun (w : M.window) -> w.size > 1 && w.step < w.size) pad.windows);
  cover "an empty operand" (numel c.f_padded = 0);
  cover "an empty result" (numel c.f_shape = 0)

let foldable = List.filter (fun d -> not (is_bool d)) D.all

(* Each dtype with ends that pad and crop, interior padding and overlapping
   windows. *)
let fold_examples =
  let a = Iarray.of_list in
  let pad =
    {
      K.lo = a [ 1; -1 ];
      hi = a [ -1; 2 ];
      interior = a [ 1; 0 ];
      windows = a [ { M.axis = 1; size = 2; step = 1; dilation = 1 } ];
    }
  in
  let s = [| 3; 4 |] in
  let out = Result.get_ok (K.Shape.padded pad s) in
  (* An int64 fold of an element no term reaches, a constant [0]. *)
  let reached =
    {
      K.lo = a [ 2; 0; 0 ];
      hi = a [ -1; 0; 0 ];
      interior = a [ 0; 0; 0 ];
      windows = a [ { M.axis = 2; size = 1; step = 1; dilation = 1 } ];
    }
  in
  let r = [| 1; 2; 2 |] in
  let r_out = Result.get_ok (K.Shape.padded reached r) in
  {
    f_dt = D.Any D.Int64;
    f_shape = r;
    f_pad = reached;
    f_padded = r_out;
    f_raws = cycled (D.Any D.Int64) (numel r_out);
  }
  :: List.mapi
       (fun i d ->
         {
           f_dt = d;
           f_shape = s;
           f_pad = pad;
           f_padded = out;
           f_raws = cycled ~i d (numel out);
         })
       foldable

let folds =
  prop ~count:40 ~examples:fold_examples
    "a fold wraps integers and holds its float bound"
    (Gen.with_pp pp_fold (fold_case foldable))
    (fun c ->
      let (D.Any dt) = c.f_dt in
      cover_fold c;
      let x = value dt c.f_padded c.f_raws in
      let op () = folded x c in
      let e = Nx.to_array (op ()) and l = Lowered.run op in
      match D.kind dt with
      | D.Float | D.Complex ->
          (* Each element's terms and their magnitudes' sum, by eager folds
             of ones and of the magnitudes. *)
          let terms =
            Array.map Float.to_int
              (Nx.to_array (folded (Nx.ones D.Float64 c.f_padded) c))
          in
          let mags =
            List.map
              (fun p ->
                Nx.to_array
                  (folded
                     (Nx.abs (Nx.create D.Float64 c.f_padded p))
                     c))
              (parts64 x)
          in
          (* Eager rounds a narrow float's sum at each addition: its
             [terms - 1] roundings of the dtype add to the bound. *)
          let narrow o =
            match fraction c.f_dt with
            | Some f when is_narrow c.f_dt ->
                Float.ldexp 1. (-f) *. Float.of_int terms.(o)
            | _ -> 0.
          in
          List.iter2
            (fun (e, l) mags ->
              Array.iteri
                (fun o e ->
                  let n = max 0 (terms.(o) - 1) in
                  let bound =
                    ((2. *. gamma c.f_dt n) +. narrow o) *. mags.(o)
                    +. rounding_of dt e l.(o)
                  in
                  near ~bound e l.(o))
                e)
            (List.combine (parts dt e) (parts dt l))
            mags
      | _ -> same_bits dt e l)

(* Assemblies *)

type assembly = {
  a_dt : D.any;
  a_shape : int array;
  a_fill : int64 array;
  a_pieces : (M.range array * view * int64 array) list;
}

let pp_range (r : M.range) = Printf.sprintf "%d+%d*%d" r.start r.count r.step

let pp_assembly ppf c =
  Format.fprintf ppf "%s [%s] fill [%s] %s" (name c.a_dt) (ints c.a_shape)
    (raws_text c.a_fill)
    (String.concat " "
       (List.map
          (fun (rs, v, raws) ->
            Printf.sprintf "(%s %s [%s])"
              (String.concat ";" (Array.to_list (Array.map pp_range rs)))
              (view_name v) (raws_text raws))
          c.a_pieces))

let rec all = function
  | [] -> Gen.constant []
  | g :: gs ->
      let open Gen in
      let* x = g in
      let+ xs = all gs in
      x :: xs

let range d =
  let open Gen in
  let* up = bool in
  let* m = int_range 1 3 in
  let step = if up then m else -m in
  if d = 0 then constant { M.start = 0; count = 0; step }
  else
    let* start = int_range 0 (d - 1) in
    let most =
      if step > 0 then ((d - 1 - start) / step) + 1 else (start / -step) + 1
    in
    let+ count =
      frequency [ (1, constant 0); (6, int_range 1 most) ]
    in
    { M.start; count; step }

let region_shape rs = Array.map (fun (r : M.range) -> r.count) rs

let assembly_case =
  let open Gen in
  let* d = all_dtypes in
  let* s = shape in
  let* fill = raws d 1 in
  let piece =
    let* rs = map Array.of_list (all (List.map range (Array.to_list s))) in
    let* v = among views in
    let+ raws = operand_gen d v (region_shape rs) in
    (rs, v, raws)
  in
  let* k = int_range 0 3 in
  let+ pieces = list ~size:(constant k) piece in
  { a_dt = d; a_shape = s; a_fill = fill; a_pieces = pieces }

(* Whether two pieces of [c] share an element. *)
let overlap c =
  let holds (rs : M.range array) idx =
    Array.for_all2
      (fun (r : M.range) i ->
        let o = i - r.start in
        o mod r.step = 0 && o / r.step >= 0 && o / r.step < r.count)
      rs idx
  in
  let n = numel c.a_shape in
  let idx k =
    let a = Array.make (Array.length c.a_shape) 0 in
    let k = ref k in
    for i = Array.length c.a_shape - 1 downto 0 do
      a.(i) <- !k mod c.a_shape.(i);
      k := !k / c.a_shape.(i)
    done;
    a
  in
  List.exists
    (fun k ->
      List.length (List.filter (fun (rs, _, _) -> holds rs (idx k)) c.a_pieces)
      > 1)
    (List.init n Fun.id)

(* Each dtype with a reversed and a strided piece that overlap, an example
   with an empty piece and one without pieces. *)
let assembly_examples =
  let r start count step = { M.start; count; step } in
  let s = [| 3; 4 |] in
  let piece i d v rs =
    (rs, v, cycled ~i d (numel (base v (region_shape rs))))
  in
  let each i d =
    {
      a_dt = d;
      a_shape = s;
      a_fill = cycled ~i:(i + 1) d 1;
      a_pieces =
        [
          piece i d (List.nth views (i mod 4)) [| r 0 3 1; r 3 2 (-1) |];
          piece (i + 5) d Plain [| r 2 2 (-1); r 0 2 2 |];
        ];
    }
  in
  let f32 = D.Any D.Float32 in
  (* Signalling NaN fills of both parts of a complex number, around a piece. *)
  let snan =
    {
      a_dt = D.Any D.Complex128;
      a_shape = [| 2 |];
      a_fill = [| 0x7ff0000000000001L; 0x7ff0000000000001L |];
      a_pieces =
        [ ([| r 0 1 (-1) |], Plain, [| 0x3fb0000000000000L; 1L |]) ];
    }
  in
  let empty = [ piece 0 f32 Plain [| r 0 0 1; r 1 2 1 |] ] in
  snan
  :: { (each 0 f32) with a_pieces = empty }
  :: { (each 0 f32) with a_pieces = [] }
  :: List.mapi each D.all

let assemblies =
  prop ~count:40 ~examples:assembly_examples "an assembly gives eager's bits"
    (Gen.with_pp pp_assembly assembly_case)
    (fun c ->
      let (D.Any dt) = c.a_dt in
      dtype_covers c.a_dt;
      cover_floats c.a_dt
        (Array.concat (c.a_fill :: List.map (fun (_, _, r) -> r) c.a_pieces));
      cover "a reversed region"
        (List.exists
           (fun (rs, _, _) ->
             Array.exists (fun (r : M.range) -> r.step < 0 && r.count > 1) rs)
           c.a_pieces);
      cover "a strided region"
        (List.exists
           (fun (rs, _, _) ->
             Array.exists
               (fun (r : M.range) -> abs r.step > 1 && r.count > 1)
               rs)
           c.a_pieces);
      cover "an empty region"
        (List.exists
           (fun (rs, _, _) -> numel (region_shape rs) = 0)
           c.a_pieces);
      cover "overlapping pieces" (overlap c);
      cover "no piece" (c.a_pieces = []);
      let fill = (Nx.to_array (value dt [| 1 |] c.a_fill)).(0) in
      let pieces =
        List.map
          (fun (rs, v, raws) ->
            (Iarray.of_array rs, operand dt v (region_shape rs) raws))
          c.a_pieces
      in
      let op () =
        P.eval ~by:"t"
          (P.Assemble
             { dtype = dt; shape = Iarray.of_array c.a_shape; fill; pieces })
      in
      same_bits dt (Nx.to_array (op ())) (Lowered.run op))

let () =
  exit
    (run "Lower_index"
       [
         group "gathers and scatters"
           [ gathers; scatters_moved; scatter_adds ];
         group "folds and assemblies" [ folds; assemblies ];
       ])
