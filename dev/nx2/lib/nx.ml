(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('v, 's, 'd) t = ('v, 's, 'd) Value.t
type ('v, 's) dtype = ('v, 's) Nx_array.Dtype.t

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let shape = Prim.shape
let dtype = Prim.dtype

module Dtype = Nx_array.Dtype

let float64 = Dtype.Float64
let float32 = Dtype.Float32
let float16 = Dtype.Float16
let bfloat16 = Dtype.Bfloat16
let float8_e4m3fn = Dtype.Float8_e4m3fn
let float8_e5m2 = Dtype.Float8_e5m2
let float4_e2m1fn = Dtype.Float4_e2m1fn
let int64 = Dtype.Int64
let uint64 = Dtype.Uint64
let int32 = Dtype.Int32
let uint32 = Dtype.Uint32
let int16 = Dtype.Int16
let uint16 = Dtype.Uint16
let int8 = Dtype.Int8
let uint8 = Dtype.Uint8
let int4 = Dtype.Int4
let uint4 = Dtype.Uint4
let complex128 = Dtype.Complex128
let complex64 = Dtype.Complex64
let bool = Dtype.Bool
let bit = Dtype.Bit

type 'd float64_t = (float, Dtype.float64_elt, 'd) t
type 'd float32_t = (float, Dtype.float32_elt, 'd) t
type 'd float16_t = (float, Dtype.float16_elt, 'd) t
type 'd bfloat16_t = (float, Dtype.bfloat16_elt, 'd) t
type 'd float8_e4m3fn_t = (float, Dtype.float8_e4m3fn_elt, 'd) t
type 'd float8_e5m2_t = (float, Dtype.float8_e5m2_elt, 'd) t
type 'd float4_e2m1fn_t = (float, Dtype.float4_e2m1fn_elt, 'd) t
type 'd int64_t = (int64, Dtype.int64_elt, 'd) t
type 'd uint64_t = (int64, Dtype.uint64_elt, 'd) t
type 'd int32_t = (int32, Dtype.int32_elt, 'd) t
type 'd uint32_t = (int32, Dtype.uint32_elt, 'd) t
type 'd int16_t = (int, Dtype.int16_signed_elt, 'd) t
type 'd uint16_t = (int, Dtype.int16_unsigned_elt, 'd) t
type 'd int8_t = (int, Dtype.int8_signed_elt, 'd) t
type 'd uint8_t = (int, Dtype.int8_unsigned_elt, 'd) t
type 'd int4_t = (int, Dtype.int4_elt, 'd) t
type 'd uint4_t = (int, Dtype.uint4_elt, 'd) t
type 'd complex128_t = (Complex.t, Dtype.complex64_elt, 'd) t
type 'd complex64_t = (Complex.t, Dtype.complex32_elt, 'd) t
type 'd bool_t = (bool, Dtype.bool_elt, 'd) t
type 'd bit_t = (bool, Dtype.bit_elt, 'd) t
type host = Devices.host
type 'd devices = 'd Devices.t

let rigs s = List.init (Devices.count s) (Devices.rig s)

module Mesh = struct
  type 'd t = 'd Devices.mesh

  let v s axes = Devices.mesh_v ~by:"Nx.Mesh.v" s axes
end

module Placement = struct
  type 'd t = 'd Devices.placement

  let on = Devices.on

  let device s d =
    match Devices.position s d with
    | Some k -> Devices.one s k
    | None ->
        invalid_argf "Nx.Placement.device: %s is not a device of %a"
          (Rig.name d) Devices.pp s

  let split ~axis s = Devices.split ~by:"Nx.Placement.split" ~axis s
  let mesh m cuts = Devices.mesh ~by:"Nx.Placement.mesh" m cuts
  let devices = Devices.set
  let equal = Devices.equal
  let pp = Devices.pp_placement
end

module type Devices = sig
  type d

  val v : d devices
  val on : d Placement.t
  val split : axis:int -> d Placement.t
end

module Host = struct
  type d = host

  let v = Devices.host
  let on = Devices.on v
  let split ~axis = Devices.split ~by:"Nx.Placement.split" ~axis v
end

let devices ?kernels ds : (module Devices) =
  let v = Devices.mint ~by:"Nx.devices" ?kernels ds in
  (module struct
    type d

    let v = v
    let on = Devices.on v
    let split ~axis = Devices.split ~by:"Nx.Placement.split" ~axis v
  end)

let place p x = Eval.place ~by:"Nx.place" p x
let placement = Prim.at

module Rng = Rng

module Repr = struct
  let of_array s a = Repr.of_array ~by:"Nx.Repr.of_array" s a
  let array = Repr.array
  let of_shards p arrays = Repr.of_shards ~by:"Nx.Repr.of_shards" p arrays
  let shards = Repr.shards
end

(* Constants and arithmetic *)

module D = Nx_array.Dtype
module P = Nx_kernel.Prog

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let bits ~by dt v =
  match P.bits dt v with
  | b -> b
  | exception Invalid_argument e -> invalid_argf "%s: %s" by e

(* [v] as an expression of [dt], naming [by] where [dt] does not hold it. *)
let const ~by dt v =
  match Nx_kernel.Prog.E.const dt v with
  | e -> e
  | exception Invalid_argument e -> invalid_argf "%s: %s" by e

(* [f ()] with an [Invalid_argument] from the array layer renamed [by]: its
   message without the layer's function. *)
let renamed ~by f =
  try f ()
  with Invalid_argument e ->
    let reason =
      match String.index_opt e ':' with
      | Some i -> String.trim (String.sub e (i + 1) (String.length e - i - 1))
      | None -> e
    in
    invalid_argf "%s: %s" by reason

let fill ~by dt shape v =
  let prog = P.of_node ~ins:[||] (Const (D.Any dt, bits ~by dt v)) in
  let layout = renamed ~by (fun () -> Nx_array.Layout.contiguous shape) in
  let x, () =
    Eval.eval ~by
      (Value.Map { layout; prog; outs = Value.[ dt ]; loads = [||] })
  in
  x

let zeros dt shape = fill ~by:"Nx.zeros" dt shape (D.zero dt)
let ones dt shape = fill ~by:"Nx.ones" dt shape (D.one dt)
let full dt shape v = fill ~by:"Nx.full" dt shape v
let scalar dt v = fill ~by:"Nx.scalar" dt [||] v

(* [c] where [x] lies: [x]'s elements are never read. *)
let beside ~by x c =
  match Prim.at x with
  | None -> c
  | Some p -> Eval.eval ~by (Value.Place (p, c))

let full_like x v = beside ~by:"Nx.full_like" x (fill ~by:"Nx.full_like" (dtype x) (shape x) v)
let zeros_like x = beside ~by:"Nx.zeros_like" x (fill ~by:"Nx.zeros_like" (dtype x) (shape x) (D.zero (dtype x)))
let ones_like x = beside ~by:"Nx.ones_like" x (fill ~by:"Nx.ones_like" (dtype x) (shape x) (D.one (dtype x)))

(* The 0-d [v] of [dt] beside [x]. A 0-d value has no axis to cut: beside a
   cut [x] it is whole on each of [x]'s devices. *)
let scalar_beside ~by x dt v =
  let c = fill ~by dt [||] v in
  match Prim.at x with
  | None -> c
  | Some p ->
      let g = Devices.grid p in
      let whole =
        Array.fold_left (fun g (axis, _) -> Grid.uncut g ~axis) g (Grid.cuts g)
      in
      let p = if whole == g then p else Devices.v ~by (Devices.set p) whole in
      Eval.eval ~by (Value.Place (p, c))

let scalar_like x v = scalar_beside ~by:"Nx.scalar_like" x (dtype x) v

(* Leaves *)

module E = P.E

let i64 n = E.const D.Int64 (Int64.of_int n)

(* The value of [dt] and [shape] whose element at each index is [out], an
   expression of no operand, reading the index through [Coord]. *)
let leaf (type v s) ~by (dt : (v, s) D.t) shape (out : (v, s) E.t) =
  let layout = renamed ~by (fun () -> Nx_array.Layout.contiguous shape) in
  let prog = P.of_exprs ~ins:[||] [| E.any out |] in
  let x, () =
    Eval.eval ~by
      (Value.Map { layout; prog; outs = Value.[ dt ]; loads = [||] })
  in
  x

(* The least and greatest [int] [dt] holds, for the dtypes an [int] can leave:
   the integers narrower than 64 bits, uint64 below zero, and booleans. *)
let int_bounds (type v s) (dt : (v, s) D.t) =
  match dt with
  | D.Int32 -> Some (Int32.to_int Int32.min_int, Int32.to_int Int32.max_int)
  | D.Uint32 -> Some (0, (1 lsl 32) - 1)
  | D.Uint64 -> Some (0, max_int)
  | D.Bool | D.Bit -> Some (0, 1)
  | D.Int16 -> Some (D.min_value dt, D.max_value dt)
  | D.Uint16 -> Some (D.min_value dt, D.max_value dt)
  | D.Int8 -> Some (D.min_value dt, D.max_value dt)
  | D.Uint8 -> Some (D.min_value dt, D.max_value dt)
  | D.Int4 -> Some (D.min_value dt, D.max_value dt)
  | D.Uint4 -> Some (D.min_value dt, D.max_value dt)
  | D.Int64 | D.Float64 | D.Float32 | D.Float16 | D.Bfloat16 | D.Float8_e4m3fn
  | D.Float8_e5m2 | D.Float4_e2m1fn | D.Complex128 | D.Complex64 ->
      None

let arange dt start stop step =
  let by = "Nx.arange" in
  if step = 0 then invalid_argf "%s: step 0" by;
  let n =
    if (step > 0 && stop <= start) || (step < 0 && stop >= start) then 0
    else
      let d = stop - start in
      if d > 0 <> (stop > start) then
        invalid_argf "%s: [%d, %d) holds more than %d values" by start stop
          max_int;
      ((d - Int.compare step 0) / step) + 1
  in
  (if n > 0 then
     match int_bounds dt with
     | None -> ()
     | Some (lo, hi) ->
         let last = start + ((n - 1) * step) in
         List.iter
           (fun v ->
             if v < lo || v > hi then
               invalid_argf "%s: %d is outside %s's range [%d, %d]" by v
                 (D.name dt) lo hi)
           [ start; last ]);
  let int64 n = E.const D.Int64 (Int64.of_int n) in
  leaf ~by dt [| n |] E.(cast dt ((coord 0 * int64 step) + int64 start))

(* [start + i step] for the index [i], by one fused multiply-add at float64. *)
let ramp ~start ~step =
  let f64 = E.const D.Float64 in
  E.(fma (cast D.Float64 (coord 0)) (f64 step) (f64 start))

let arange_f dt start stop step =
  let by = "Nx.arange_f" in
  if step = 0. then invalid_argf "%s: step 0" by;
  let length = Float.ceil ((stop -. start) /. step) in
  if (not (Float.is_finite length)) || length >= Float.of_int max_int then
    invalid_argf "%s: [%g, %g) by %g has no finite length" by start stop step;
  let n = if length > 0. then Float.to_int length else 0 in
  leaf ~by dt [| n |] (E.cast dt (ramp ~start ~step))

(* [n] values from [start] to [stop] at float64: [stop] itself last with
   [endpoint], so the ends are exact. *)
let spaced ~by ~endpoint start stop n =
  if n < 0 then invalid_argf "%s: %d values" by n;
  let div = if endpoint then n - 1 else n in
  let step = if div > 0 then (stop -. start) /. Float.of_int div else 0. in
  let v = ramp ~start ~step in
  if not (endpoint && n >= 2) then v
  else
    let final = E.const D.Int64 (Int64.of_int (n - 1)) in
    E.(where (coord 0 = final) (const D.Float64 stop) v)

let linspace dt ?(endpoint = true) start stop n =
  let by = "Nx.linspace" in
  let v = spaced ~by ~endpoint start stop n in
  leaf ~by dt [| Stdlib.max n 0 |] (E.cast dt v)

let logspace dt ?(endpoint = true) ?(base = 10.) start stop n =
  let by = "Nx.logspace" in
  let e = spaced ~by ~endpoint start stop n in
  leaf ~by dt [| Stdlib.max n 0 |] E.(cast dt (const D.Float64 base ** e))

(* Data in and out *)

type 'd packed = P : ('v, 's, 'd) t -> 'd packed

let unpack (type v s d) (dt : (v, s) D.t) (P x : d packed) : (v, s, d) t =
  match D.equal_witness (dtype x) dt with
  | Some Type.Equal -> x
  | None ->
      invalid_argf "Nx.unpack: a value of %s, not %s" (D.name (dtype x))
        (D.name dt)

let on_host a : (_, _, host) t = Repr.of_array Host.v a

let create dt shape vs =
  renamed ~by:"Nx.create" (fun () -> on_host (Nx_array.of_array dt shape vs))

let init dt shape f =
  let by = "Nx.init" in
  let l = renamed ~by (fun () -> Nx_array.Layout.contiguous shape) in
  let r = Array.length shape in
  (* Each index in C order: the last axis runs fastest. *)
  let index j =
    let i = Array.make r 0 and j = ref j in
    for a = r - 1 downto 0 do
      i.(a) <- !j mod shape.(a);
      j := !j / shape.(a)
    done;
    i
  in
  let vs = Array.init (Nx_array.Layout.numel l) (fun j -> f (index j)) in
  renamed ~by (fun () -> on_host (Nx_array.of_array dt shape vs))

let dtype_of_kind (type v s) (k : (v, s) Bigarray.kind) : (v, s) D.t =
  match k with
  | Bigarray.Float64 -> D.Float64
  | Bigarray.Float32 -> D.Float32
  | Bigarray.Float16 -> D.Float16
  | Bigarray.Int64 -> D.Int64
  | Bigarray.Int32 -> D.Int32
  | Bigarray.Int16_signed -> D.Int16
  | Bigarray.Int16_unsigned -> D.Uint16
  | Bigarray.Int8_signed -> D.Int8
  | Bigarray.Int8_unsigned -> D.Uint8
  | Bigarray.Complex64 -> D.Complex128
  | Bigarray.Complex32 -> D.Complex64
  | Bigarray.Char | Bigarray.Int | Bigarray.Nativeint ->
      invalid_argf "Nx.of_bigarray: no dtype stores Bigarray's %s kind"
        (match k with
        | Bigarray.Char -> "char"
        | Bigarray.Int -> "int"
        | _ -> "nativeint")

let of_bigarray b =
  let dt = dtype_of_kind (Bigarray.Genarray.kind b) in
  on_host (Nx_array.copy (Nx_array.of_bigarray dt b))

(* [x], waited for, for the read [by]. A read is not an operation: no
   interpretation receives it. *)
let readable ~by x =
  Prim.alive ~by 0 x;
  match Eval.owner x with
  | Some i -> invalid_argf "%s: the value is traced by %s" by i.name
  | None ->
      let x = Exec.live x in
      Exec.wait ~by x;
      x

let sync x = ignore (readable ~by:"Nx.sync" x)

let read ~by x = Exec.on_host ~by (readable ~by x)

let to_array x = Nx_array.to_array (read ~by:"Nx.to_array" x)

let to_bigarray k x =
  let by = "Nx.to_bigarray" in
  let r = Prim.rank x in
  if r > 16 then invalid_argf "%s: rank %d, above Bigarray's 16" by r;
  Option.get (Nx_array.bigarray k (Nx_array.copy (read ~by x)))

let item i x =
  let by = "Nx.item" in
  let s = shape x in
  let r = Array.length s in
  if List.length i <> r then
    invalid_argf "%s: index %a for %d axes (%s %a)" by pp_shape
      (Array.of_list i) r (D.name (dtype x)) pp_shape s;
  let at a p =
    let d = s.(a) in
    let q = if p < 0 then p + d else p in
    if q < 0 || q >= d then
      invalid_argf "%s: position %d is outside axis %d of extent %d" by p a d;
    { Nx_array.Move.start = q; count = 1; step = 1 }
  in
  let one = List.mapi at i in
  match readable ~by x with
  | Value.Array { a; _ } when Rig.reaches Rig.host (Nx_array.device a) ->
      let start (w : Nx_array.Move.range) = w.start in
      Nx_array.get a (Array.of_list (List.map start one))
  | x ->
      let v = Exec.run ~by (Value.Move (Slice (Iarray.of_list one), x)) in
      Nx_array.get (Exec.on_host ~by v) (Array.make r 0)

let broadcast ~by s x =
  if Prim.has_shape x s then x
  else Eval.eval ~by (Value.Move (Broadcast (Iarray.of_array s), x))

let same_shape = Prim.same_shape

let binary ~by k a b =
  if same_shape a b then Eval.apply2 ~by k (dtype a) a b
  else
    let s = Prim.broadcast_shape ~by (shape a) (shape b) in
    Eval.apply2 ~by k (dtype a) (broadcast ~by s a) (broadcast ~by s b)

let add a b = binary ~by:"Nx.add" (Binary Add) a b
let sub a b = binary ~by:"Nx.sub" (Binary Sub) a b
let mul a b = binary ~by:"Nx.mul" (Binary Mul) a b

(* Integers divide by [Idiv]; floats and complex numbers by [Fdiv], whose
   refusal names booleans. *)
let quotient (type v s) (dt : (v, s) D.t) : P.binary =
  match D.kind dt with D.Signed | D.Unsigned -> Idiv | _ -> Fdiv

let div a b = binary ~by:"Nx.div" (Binary (quotient (dtype a))) a b

let mod_ a b = binary ~by:"Nx.mod_" (Binary Mod) a b
let pow a b = binary ~by:"Nx.pow" (Binary Pow) a b
let atan2 a b = binary ~by:"Nx.atan2" (Binary Atan2) a b
let maximum a b = binary ~by:"Nx.maximum" (Binary Maximum) a b
let minimum a b = binary ~by:"Nx.minimum" (Binary Minimum) a b
let bitwise_and a b = binary ~by:"Nx.bitwise_and" (Binary And) a b
let bitwise_or a b = binary ~by:"Nx.bitwise_or" (Binary Or) a b
let bitwise_xor a b = binary ~by:"Nx.bitwise_xor" (Binary Xor) a b

let comparison ~by k a b =
  if same_shape a b then Eval.apply2 ~by k D.Bool a b
  else
    let s = Prim.broadcast_shape ~by (shape a) (shape b) in
    Eval.apply2 ~by k D.Bool (broadcast ~by s a) (broadcast ~by s b)

let equal a b = comparison ~by:"Nx.equal" (Compare Equal) a b
let not_equal a b = comparison ~by:"Nx.not_equal" (Compare Not_equal) a b
let less a b = comparison ~by:"Nx.less" (Compare Less) a b
let less_equal a b = comparison ~by:"Nx.less_equal" (Compare Less_equal) a b
let greater a b = comparison ~by:"Nx.greater" (Compare Less) b a

let greater_equal a b =
  comparison ~by:"Nx.greater_equal" (Compare Less_equal) b a

(* Scalar forms: the constant [c] of [x]'s dtype, of every set, beside [x]. *)

let constant ~by x c = fill ~by (dtype x) [||] c
let scalar_binary ~by k x c = binary ~by k x (constant ~by x c)
let add_s x c = scalar_binary ~by:"Nx.add_s" (Binary Add) x c
let sub_s x c = scalar_binary ~by:"Nx.sub_s" (Binary Sub) x c
let mul_s x c = scalar_binary ~by:"Nx.mul_s" (Binary Mul) x c
let div_s x c = scalar_binary ~by:"Nx.div_s" (Binary (quotient (dtype x))) x c
let pow_s x c = scalar_binary ~by:"Nx.pow_s" (Binary Pow) x c
let mod_s x c = scalar_binary ~by:"Nx.mod_s" (Binary Mod) x c
let maximum_s x c = scalar_binary ~by:"Nx.maximum_s" (Binary Maximum) x c
let minimum_s x c = scalar_binary ~by:"Nx.minimum_s" (Binary Minimum) x c

let reversed ~by k c x = binary ~by k (constant ~by x c) x
let rsub_s c x = reversed ~by:"Nx.rsub_s" (Binary Sub) c x
let rdiv_s c x = reversed ~by:"Nx.rdiv_s" (Binary (quotient (dtype x))) c x
let rpow_s c x = reversed ~by:"Nx.rpow_s" (Binary Pow) c x

let compared ~by k x c = comparison ~by k x (constant ~by x c)
let compared_by ~by k c x = comparison ~by k (constant ~by x c) x
let equal_s x c = compared ~by:"Nx.equal_s" (Compare Equal) x c
let not_equal_s x c = compared ~by:"Nx.not_equal_s" (Compare Not_equal) x c
let less_s x c = compared ~by:"Nx.less_s" (Compare Less) x c
let less_equal_s x c = compared ~by:"Nx.less_equal_s" (Compare Less_equal) x c
let greater_s x c = compared_by ~by:"Nx.greater_s" (Compare Less) c x

let greater_equal_s x c =
  compared_by ~by:"Nx.greater_equal_s" (Compare Less_equal) c x

let ternary ~by k c x y =
  if same_shape c x && same_shape x y then Eval.apply3 ~by k c x y
  else
    let s =
      Prim.broadcast_shape ~by
        (Prim.broadcast_shape ~by (shape c) (shape x))
        (shape y)
    in
    Eval.apply3 ~by k (broadcast ~by s c) (broadcast ~by s x)
      (broadcast ~by s y)

let where c x y = ternary ~by:"Nx.where" Where c x y
let fma a b c = ternary ~by:"Nx.fma" Fma a b c
let unary ~by k x = Eval.apply1 ~by (Unary k) (dtype x) x
let neg x = unary ~by:"Nx.neg" Neg x
let recip x = unary ~by:"Nx.recip" Recip x
let abs x = unary ~by:"Nx.abs" Abs x
let sign x = unary ~by:"Nx.sign" Sign x
let sqrt x = unary ~by:"Nx.sqrt" Sqrt x
let exp x = unary ~by:"Nx.exp" Exp x
let exp2 x = unary ~by:"Nx.exp2" Exp2 x
let expm1 x = unary ~by:"Nx.expm1" Expm1 x
let log x = unary ~by:"Nx.log" Log x
let log2 x = unary ~by:"Nx.log2" Log2 x
let log1p x = unary ~by:"Nx.log1p" Log1p x
let sin x = unary ~by:"Nx.sin" Sin x
let cos x = unary ~by:"Nx.cos" Cos x
let tan x = unary ~by:"Nx.tan" Tan x
let asin x = unary ~by:"Nx.asin" Asin x
let acos x = unary ~by:"Nx.acos" Acos x
let atan x = unary ~by:"Nx.atan" Atan x
let sinh x = unary ~by:"Nx.sinh" Sinh x
let cosh x = unary ~by:"Nx.cosh" Cosh x
let tanh x = unary ~by:"Nx.tanh" Tanh x
let erf x = unary ~by:"Nx.erf" Erf x
let floor x = unary ~by:"Nx.floor" Floor x
let ceil x = unary ~by:"Nx.ceil" Ceil x
let round x = unary ~by:"Nx.round" Round x
let trunc x = unary ~by:"Nx.trunc" Trunc x

let cast (type v s w r d) (dt : (w, r) D.t) (x : (v, s, d) t) : (w, r, d) t =
  match D.equal_witness (dtype x) dt with
  | Some Type.Equal -> x
  | None -> Eval.apply1 ~by:"Nx.cast" Cast dt x

let bitcast (type v s w r d) (dt : (w, r) D.t) (x : (v, s, d) t) : (w, r, d) t =
  match D.equal_witness (dtype x) dt with
  | Some Type.Equal -> x
  | None -> Eval.eval ~by:"Nx.bitcast" (Value.Bitcast (dt, x))

let copy x = Eval.eval ~by:"Nx.copy" (Value.Copy x)
let donate x = Exec.donate ~by:"Nx.donate" x

(* Shapes, broadcasting and movements *)

let ndim = Prim.rank
let numel x = Array.fold_left ( * ) 1 (shape x)
let nbytes x = ((numel x * D.bits (dtype x)) + 7) / 8

(* An operand in messages, as [float32 [2; 3]]. *)
let pp_value ppf x =
  Format.fprintf ppf "%s %a" (D.name (dtype x)) pp_shape (shape x)

let pp_ints ppf l = pp_shape ppf (Array.of_list l)

let axis ~by x a = Prim.axis ~by (ndim x) a (fun ppf -> pp_value ppf x)
let dim a x = Prim.dim x (axis ~by:"Nx.dim" x a)

(* Distinct axes of [x], or raises naming the repeated one. *)
let axes ~by x l =
  let seen = Array.make (ndim x) false in
  List.map
    (fun a ->
      let a' = axis ~by x a in
      if seen.(a') then invalid_argf "%s: axis %d of %a repeats" by a pp_value x;
      seen.(a') <- true;
      a')
    l

let move ~by mv x = Eval.eval ~by (Value.Move (mv, x))

(* [x] reshaped to, or sliced by, a caller's array, frozen into the
   movement. *)
let reshaped ~by s x = move ~by (Reshape (Iarray.of_array s)) x
let sliced ~by rs x = move ~by (Slice (Iarray.of_array rs)) x

(* [a * b] for extents, or raises naming [by] past an [int]. *)
let times ~by x a b =
  if a <> 0 && b > max_int / a then
    invalid_argf "%s: %a would have more elements than an int counts" by
      pp_value x;
  a * b

let reshape s x =
  let by = "Nx.reshape" in
  let n = numel x and s = Iarray.of_array s in
  let pp_s ppf () = pp_shape ppf (Iarray.to_array s) in
  let unknown = ref None and known = ref 1 in
  Iarray.iteri
    (fun i e ->
      if e < -1 then invalid_argf "%s: extent %d in %a" by e pp_s ();
      if e = -1 then begin
        if !unknown <> None then
          invalid_argf "%s: %a has two unknown extents" by pp_s ();
        unknown := Some i
      end
      else known := times ~by x !known e)
    s;
  let s =
    match !unknown with
    | Some i when !known > 0 && n mod !known = 0 ->
        Iarray.mapi (fun j e -> if j = i then n / !known else e) s
    | Some _ ->
        invalid_argf "%s: %a has %d elements, which %a cannot hold" by pp_value
          x n pp_s ()
    | None ->
        if !known <> n then
          invalid_argf "%s: %a has %d elements, %a has %d" by pp_value x n pp_s
            () !known;
        s
  in
  move ~by (Reshape s) x

let broadcast_to s x =
  let by = "Nx.broadcast_to" in
  let fits =
    Array.for_all (fun e -> e >= 0) s
    && match Prim.merge (shape x) s with Ok s' -> s' = s | Error _ -> false
  in
  if not fits then
    invalid_argf "%s: %a does not broadcast to %a" by pp_value x pp_shape s;
  move ~by (Broadcast (Iarray.of_array s)) x

(* The shape [shapes] broadcast to, raising naming the first that does not
   broadcast with those before it, and the axis where it does not. *)
let broadcast_all ~by shapes =
  let pp_list ppf l =
    Format.pp_print_list
      ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
      pp_shape ppf l
  in
  (* CR: Appending copies the growing diagnostic prefix on every valid shape.
     Cons s' onto before and reverse only for pp_list on failure, keeping
     the ordered message while making this bookkeeping linear. Add a
     broadcast_shapes bench over prebuilt compatible shapes. *)
  let step (s, before) s' =
    if Array.exists (fun e -> e < 0) s' then
      invalid_argf "%s: %a has a negative extent" by pp_shape s';
    match Prim.merge s s' with
    | Ok s -> (s, before @ [ s' ])
    | Error (a, e, e') ->
        (* The axis counted from the end, where every shape aligns. *)
        let a = a - max (Array.length s) (Array.length s') in
        invalid_argf
          "%s: %a does not broadcast with %a: axis %d has %d, neither 1 nor %d"
          by pp_shape s' pp_list before a e' e
  in
  fst (List.fold_left step ([||], []) shapes)

let broadcast_shapes ss = broadcast_all ~by:"Nx.broadcast_shapes" ss

let broadcast_arrays xs =
  let by = "Nx.broadcast_arrays" in
  let s = broadcast_all ~by (List.map shape xs) in
  List.map (broadcast ~by s) xs

let squeeze ?axes:l x =
  let by = "Nx.squeeze" in
  let s = shape x in
  let drop =
    match l with
    | None -> Array.map (fun e -> e = 1) s
    | Some l ->
        let drop = Array.make (Array.length s) false in
        List.iter
          (fun a ->
            if s.(a) <> 1 then
              invalid_argf "%s: axis %d of %a has extent %d" by a pp_value x
                s.(a);
            drop.(a) <- true)
          (axes ~by x l);
        drop
  in
  let kept = List.filteri (fun i _ -> not drop.(i)) (Array.to_list s) in
  reshaped ~by (Array.of_list kept) x

let unsqueeze ~axes:l x =
  let by = "Nx.unsqueeze" in
  let s = shape x in
  let r = Array.length s + List.length l in
  let added = Array.make r false in
  List.iter
    (fun a ->
      let a' =
        Prim.axis ~by r a (fun ppf -> Format.fprintf ppf "a rank %d result" r)
      in
      if added.(a') then invalid_argf "%s: position %d repeats" by a;
      added.(a') <- true)
    l;
  (* [x]'s axes fill the positions not added, in order. *)
  let s' = Array.make r 1 in
  List.iteri
    (fun k i -> s'.(i) <- s.(k))
    (List.filter (fun i -> not added.(i)) (List.init r Fun.id));
  reshaped ~by s' x

let flatten ?(start_dim = 0) ?(end_dim = -1) x =
  let by = "Nx.flatten" in
  (* A 0-d value flattens as the [[1]] it holds. *)
  let x = if ndim x = 0 then move ~by (Reshape [| 1 |]) x else x in
  let s = shape x in
  let a = axis ~by x start_dim and b = axis ~by x end_dim in
  if a > b then
    invalid_argf "%s: start_dim %d comes after end_dim %d in %a" by start_dim
      end_dim pp_value x;
  let merged = Array.fold_left (times ~by x) 1 (Array.sub s a (b - a + 1)) in
  let s' =
    Array.concat
      [
        Array.sub s 0 a;
        [| merged |];
        Array.sub s (b + 1) (Array.length s - b - 1);
      ]
  in
  reshaped ~by s' x

let permute ~by p x = move ~by (Permute (Iarray.of_array p)) x

let transpose ?axes:l x =
  let by = "Nx.transpose" in
  let r = ndim x in
  match l with
  | None -> permute ~by (Array.init r (fun i -> r - 1 - i)) x
  | Some l ->
      if List.length l <> r then
        invalid_argf "%s: axes %a are not a permutation of %a's" by pp_ints l
          pp_value x;
      permute ~by (Array.of_list (axes ~by x l)) x

let moveaxis a b x =
  let by = "Nx.moveaxis" in
  let a = axis ~by x a and b = axis ~by x b in
  let rest = List.filter (( <> ) a) (List.init (ndim x) Fun.id) in
  let p =
    List.filteri (fun i _ -> i < b) rest
    @ (a :: List.filteri (fun i _ -> i >= b) rest)
  in
  permute ~by (Array.of_list p) x

let swapaxes a b x =
  let by = "Nx.swapaxes" in
  let a = axis ~by x a and b = axis ~by x b in
  let p = Array.init (ndim x) Fun.id in
  p.(a) <- b;
  p.(b) <- a;
  permute ~by p x

(* The whole of an axis of extent [d]. *)
let whole d : Nx_array.Move.range = { start = 0; count = d; step = 1 }

(* The region of all of a value of shape [s]. *)
let all_of s = Iarray.init (Array.length s) (fun a -> whole s.(a))

let flip ?axes:l x =
  let by = "Nx.flip" in
  let s = shape x in
  let flipped =
    match l with
    | None -> Array.make (Array.length s) true
    | Some l ->
        let f = Array.make (Array.length s) false in
        List.iter (fun a -> f.(a) <- true) (axes ~by x l);
        f
  in
  let range i d : Nx_array.Move.range =
    if flipped.(i) then { start = max 0 (d - 1); count = d; step = -1 }
    else whole d
  in
  sliced ~by (Array.mapi range s) x

let sliding_window ?axis:(a = -1) ~window ?(step = 1) x =
  let by = "Nx.sliding_window" in
  let axis = axis ~by x a in
  if window < 1 || step < 1 then
    invalid_argf "%s: window %d and step %d over %a; give at least 1" by window
      step pp_value x;
  move ~by (Window [| { axis; size = window; step; dilation = 1 } |]) x

let split ~axis:a n x =
  let by = "Nx.split" in
  let a = axis ~by x a in
  if n < 1 then
    invalid_argf "%s: %d runs of %a; give at least 1" by n pp_value x;
  let s = shape x in
  let d = s.(a) in
  let start = ref 0 in
  List.init n (fun k ->
      let count = (d / n) + if k < d mod n then 1 else 0 in
      let rs = Array.map whole s in
      rs.(a) <- { start = !start; count; step = 1 };
      start := !start + count;
      sliced ~by rs x)

let assemble ~by dtype shape fill pieces =
  let shape = Iarray.of_array shape in
  Eval.eval ~by (Value.Assemble { dtype; shape; fill; pieces })

(* The region of a value of shape [s] that keeps [count] elements from [start]
   along [axis], and the whole of every other axis. *)
let region s axis start count =
  Iarray.init (Array.length s) (fun a ->
      if a <> axis then whole s.(a)
      else { Nx_array.Move.start; count; step = 1 })

let concatenate ~axis:a xs =
  let by = "Nx.concatenate" in
  match xs with
  | [] -> invalid_argf "%s: no values" by
  | x0 :: _ ->
      let a = axis ~by x0 a in
      let s0 = shape x0 in
      let fits s =
        Array.length s = Array.length s0
        && Array.for_all Fun.id (Array.mapi (fun i e -> i = a || e = s0.(i)) s)
      in
      let total = ref 0 in
      let pieces =
        List.map
          (fun x ->
            let s = shape x in
            if not (fits s) then
              invalid_argf "%s: %a and %a differ off axis %d" by pp_value x0
                pp_value x a;
            let r = region s a !total s.(a) in
            total := !total + s.(a);
            (r, x))
          xs
      in
      let s' = Array.copy s0 in
      s'.(a) <- !total;
      assemble ~by (dtype x0) s' (D.zero (dtype x0)) pieces

let stack ?axis:(a = 0) xs =
  let by = "Nx.stack" in
  match xs with
  | [] -> invalid_argf "%s: no values" by
  | x0 :: _ ->
      let s0 = shape x0 in
      let r = Array.length s0 + 1 in
      let a =
        Prim.axis ~by r a (fun ppf -> Format.fprintf ppf "a rank %d result" r)
      in
      List.iter
        (fun x ->
          if not (same_shape x x0) then
            invalid_argf "%s: %a and %a differ" by pp_value x0 pp_value x)
        xs;
      let unit =
        Array.init r (fun i ->
            if i < a then s0.(i) else if i = a then 1 else s0.(i - 1))
      in
      let s' = Array.copy unit in
      s'.(a) <- List.length xs;
      let lifted = Iarray.of_array unit in
      let lift x = move ~by (Reshape lifted) x in
      let pieces = List.mapi (fun k x -> (region unit a k 1, lift x)) xs in
      assemble ~by (dtype x0) s' (D.zero (dtype x0)) pieces

let pad widths v x =
  let by = "Nx.pad" in
  let s = shape x in
  if Array.length widths <> Array.length s then
    invalid_argf "%s: %d widths for %a" by (Array.length widths) pp_value x;
  Array.iter
    (fun (b, e) ->
      if b < 0 || e < 0 then invalid_argf "%s: a negative width (%d, %d)" by b e)
    widths;
  let s' = Array.mapi (fun i d -> fst widths.(i) + d + snd widths.(i)) s in
  let rs =
    Iarray.init (Array.length s) (fun i : Nx_array.Move.range ->
        { start = fst widths.(i); count = s.(i); step = 1 })
  in
  assemble ~by (dtype x) s' v [ (rs, x) ]

let roll ?axis:along k x =
  let by = "Nx.roll" in
  let y, a =
    match along with
    | None -> (move ~by (Reshape [| numel x |]) x, 0)
    | Some a -> (x, axis ~by x a)
  in
  let s = shape y in
  let d = s.(a) in
  let k = if d = 0 then 0 else k mod d in
  let k = if k < 0 then k + d else k in
  let rolled =
    if k = 0 then copy y
    else
      let part start count = move ~by (Slice (region s a start count)) y in
      assemble ~by (dtype y) s
        (D.zero (dtype y))
        [
          (region s a 0 k, part (d - k) k);
          (region s a k (d - k), part 0 (d - k));
        ]
  in
  match along with
  | Some _ -> rolled
  | None -> reshaped ~by (shape x) rolled

(* [x] with its axis [i] repeated [n.(i)] times: whole, end to end, where
   [outer]; element by element otherwise. Each repeated axis gains a unit axis
   beside it, before it where [outer], which is broadcast to the count and
   merged into it. *)
let stretch ~by ~outer n x =
  if Array.for_all (( = ) 1) n then x
  else
    let s = shape x in
    let spread f =
      Array.of_list (List.concat (List.mapi f (Array.to_list s)))
    in
    let beside k i d =
      if n.(i) = 1 then [ d ] else if outer then [ k; d ] else [ d; k ]
    in
    let unit = spread (beside 1) and wide = spread (fun i -> beside n.(i) i) in
    let merged = Array.mapi (fun i d -> times ~by x n.(i) d) s in
    reshaped ~by merged
      (move ~by (Broadcast (Iarray.of_array wide)) (reshaped ~by unit x))

let tile reps x =
  let by = "Nx.tile" in
  let s = shape x in
  let r = Array.length reps and k = Array.length s in
  if r < k then
    invalid_argf "%s: reps %a has fewer entries than %a has axes" by pp_shape
      reps pp_value x;
  if Array.exists (fun n -> n < 0) reps then
    invalid_argf "%s: reps %a has a negative entry" by pp_shape reps;
  let lead = Array.append (Array.make (r - k) 1) s in
  let x = if r = k then x else reshaped ~by lead x in
  stretch ~by ~outer:true reps x

let repeat ?axis:a n x =
  let by = "Nx.repeat" in
  if n < 0 then invalid_argf "%s: count %d is negative" by n;
  let x, a =
    match a with
    | None -> (move ~by (Reshape [| numel x |]) x, 0)
    | Some a -> (x, axis ~by x a)
  in
  stretch ~by ~outer:false
    (Array.init (ndim x) (fun i -> if i = a then n else 1))
    x

(* Axis patterns *)

module Pattern = struct
  include Pattern

  let v s = v ~by:"Nx.Pattern.v" s
  let inverse p = inverse ~by:"Nx.Pattern.inverse" p
end

let rearrange ?(sizes = []) p x =
  let by = "Nx.rearrange" in
  let moves =
    Pattern.moves ~by ~sizes p (shape x) (fun ppf -> pp_value ppf x)
  in
  List.fold_left (fun x mv -> move ~by mv x) x moves

(* Indexing and slicing *)

type 'd index =
  | I of int
  | L of int list
  | T of 'd int64_t
  | R of int * int
  | Rs of int * int * int
  | A
  | N
  | D of 'd int64_t * int

(* An entry of a selection checked against its axis of extent [d]: one written
   position, a range, written positions, positions held in data, a window from a
   start held in data, or a new axis. *)
type 'd pick =
  | At of int
  | Span of Nx_array.Move.range
  | Rows of int list
  | Held of 'd int64_t
  | From of 'd int64_t * int * int
  | New

(* [start] to [stop] by [step] within an axis of extent [d], clipped as a range
   written in the program is: negative ends count from the end. *)
(* A range's count is [1 + (distance - 1) / |step|] for a positive distance
   from its start to its stop: no sum forms past the axis, and a step of
   [min_int], whose magnitude is no int, passes every axis in one. *)
let range d start stop step : Nx_array.Move.range =
  let resolve e = if e < 0 then e + d else e in
  if step > 0 then
    let start = max 0 (min d (resolve start))
    and stop = max 0 (min d (resolve stop)) in
    let count = if stop <= start then 0 else 1 + ((stop - start - 1) / step) in
    { start; count; step }
  else
    let start = max (-1) (min (d - 1) (resolve start))
    and stop = max (-1) (min (d - 1) (resolve stop)) in
    let count =
      if start <= stop then 0
      else if step = min_int then 1
      else 1 + ((start - stop - 1) / -step)
    in
    { start = (if count = 0 then 0 else start); count; step }

(* [idx] checked against [x]: each entry with the axis it selects along, [None]
   for [N]. Every refusal raises here, before anything is computed. *)
let picks ~by idx x =
  let s = shape x in
  let r = Array.length s in
  let addressed =
    List.length (List.filter (function N -> false | _ -> true) idx)
  in
  if addressed > r then
    invalid_argf "%s: %d entries address %d axes; the operand is %a" by
      (List.length idx) addressed pp_value x;
  let next = ref 0 in
  (* The next axis, and a written position checked against it. *)
  let axis () =
    incr next;
    !next - 1
  in
  let written a p =
    let p' = if p < 0 then p + s.(a) else p in
    if p' < 0 || p' >= s.(a) then
      invalid_argf "%s: position %d is outside axis %d of %a" by p a pp_value x;
    p'
  in
  List.map
    (fun entry ->
      match entry with
      | N -> (New, None)
      | I p ->
          let a = axis () in
          (At (written a p), Some a)
      | L ps ->
          let a = axis () in
          (Rows (List.map (written a) ps), Some a)
      | T p -> (Held p, Some (axis ()))
      | R (start, stop) ->
          let a = axis () in
          (Span (range s.(a) start stop 1), Some a)
      | Rs (_, _, 0) -> invalid_argf "%s: a step of 0" by
      | Rs (start, stop, step) ->
          let a = axis () in
          (Span (range s.(a) start stop step), Some a)
      | A ->
          let a = axis () in
          (Span (whole s.(a)), Some a)
      | D (start, n) ->
          let a = axis () in
          if ndim start <> 0 then
            invalid_argf "%s: a window's start %a is not 0-d" by pp_value start;
          if n < 0 || n > s.(a) then
            invalid_argf "%s: a window of %d on axis %d of %a" by n a pp_value x;
          (From (start, n, s.(a)), Some a))
    idx

(* The axes a pick gives the selection. *)
let extents_of (type d) (p : d pick) =
  match p with
  | At _ -> [||]
  | Span r -> [| r.count |]
  | Rows ps -> [| List.length ps |]
  | Held p -> shape p
  | From (_, n, _) -> [| n |]
  | New -> [| 1 |]

(* The selection's shape: each pick's axes, then the axes past them. *)
let selection x picks =
  let s = shape x in
  let used = List.length (List.filter (fun (_, a) -> a <> None) picks) in
  Array.concat
    (List.map (fun (p, _) -> extents_of p) picks
    @ [ Array.sub s used (Array.length s - used) ])

(* The int64 value of written positions [ps]: a value of every set. *)
let written ~by ps =
  assemble ~by D.Int64
    [| List.length ps |]
    0L
    (List.mapi
       (fun j p ->
         ( ([| { Nx_array.Move.start = j; count = 1; step = 1 } |] : _ iarray),
           fill ~by D.Int64 [| 1 |] (Int64.of_int p) ))
       ps)

let const64 n = P.Const (D.Any D.Int64, P.bits D.Int64 (Int64.of_int n))

(* The [n] positions of a window from [start], clamped into [0, d - n]. *)
let window ~by start n d =
  let prog =
    P.v ~ins:[| D.Any D.Int64 |]
      [|
        In 0;
        const64 0;
        Op2 (Binary Maximum, 0, 1);
        const64 (d - n);
        Op2 (Binary Minimum, 2, 3);
        Coord 0;
        Op2 (Binary Add, 4, 5);
      |]
      ~outs:[| 6 |]
  in
  let v, () =
    Eval.eval ~by
      (Value.Map
         {
           layout = Nx_array.Layout.contiguous [| n |];
           prog;
           outs = Value.[ D.Int64 ];
           loads = [| Plain (broadcast ~by [| n |] start) |];
         })
  in
  v

(* [y] read along [axis] at the positions [p], in C order: [y]'s axis of
   extent the number of [p]'s elements. [p] runs along the axis, broadcast
   across the others, which a kernel reads as row takes. *)
let gather_axis ~by axis p y =
  let s = shape y and n = numel p in
  let col = Array.mapi (fun i _ -> if i = axis then n else 1) s in
  let wide = Array.mapi (fun i d -> if i = axis then n else d) s in
  let idx = broadcast ~by wide (reshaped ~by col p) in
  Eval.eval ~by (Value.Gather { axis; idx; x = y })

(* [y] read along [axis] at the positions [p]: [y]'s axis replaced by [p]'s
   axes. *)
let take_axis ~by axis p y =
  let s = shape y and ps = shape p in
  let g = gather_axis ~by axis p y in
  let s' =
    Array.concat
      [
        Array.sub s 0 axis;
        ps;
        Array.sub s (axis + 1) (Array.length s - axis - 1);
      ]
  in
  if Prim.has_shape g s' then g else reshaped ~by s' g

let slice idx x =
  let by = "Nx.slice" in
  let picks = picks ~by idx x in
  let s = shape x in
  let ranges = Array.map whole s in
  List.iter
    (fun (p, a) ->
      match (p, a) with
      | At i, Some a -> ranges.(a) <- { start = i; count = 1; step = 1 }
      | Span r, Some a -> ranges.(a) <- r
      | (At _ | Span _ | Rows _ | Held _ | From _ | New), _ -> ())
    picks;
  let y =
    if
      Array.for_all2
        (fun (r : Nx_array.Move.range) d -> r.count = d && r.step = 1)
        ranges s
    then x
    else sliced ~by ranges x
  in
  (* Each gather keeps the rank: a held axis of [n] positions has extent [n]
     until the reshape below gives it their shape, so no step passes the
     result's rank. *)
  let y =
    List.fold_left
      (fun y (p, a) ->
        match (p, a) with
        | Rows ps, Some a -> gather_axis ~by a (written ~by ps) y
        | Held p, Some a -> gather_axis ~by a p y
        | From (start, n, d), Some a ->
            gather_axis ~by a (window ~by start n d) y
        | (At _ | Span _ | Rows _ | Held _ | From _ | New), _ -> y)
      y (List.rev picks)
  in
  (* Each [At] axis now has extent 1 and goes; each [New] axis comes; each
     held axis takes its positions' shape. *)
  let s' = selection x picks in
  if Prim.has_shape y s' then y else reshaped ~by s' y

let get p x =
  let by = "Nx.get" in
  let s = shape x in
  if List.length p > Array.length s then
    invalid_argf "%s: %d positions for %a" by (List.length p) pp_value x;
  let ranges = Array.map whole s in
  List.iteri
    (fun a i ->
      let i' = if i < 0 then i + s.(a) else i in
      if i' < 0 || i' >= s.(a) then
        invalid_argf "%s: position %d is outside axis %d of %a" by i a pp_value
          x;
      ranges.(a) <- { start = i'; count = 1; step = 1 })
    p;
  let k = List.length p in
  move ~by
    (Reshape (Iarray.of_array (Array.sub s k (Array.length s - k))))
    (sliced ~by ranges x)

let take ?axis:along p x =
  let by = "Nx.take" in
  match along with
  | None -> take_axis ~by 0 p (move ~by (Reshape [| numel x |]) x)
  | Some a -> take_axis ~by (axis ~by x a) p x

(* [x] read along [a] at the positions [p] of [x]'s rank, for the function
   [by]. *)
let take_along ~by a p x =
  let a = axis ~by x a in
  if ndim p <> ndim x then
    invalid_argf "%s: positions %a for %a of another rank" by pp_value p
      pp_value x;
  let sp = shape p and sx = shape x in
  let off s = Array.mapi (fun i d -> if i = a then 1 else d) s in
  let b = Prim.broadcast_shape ~by (off sp) (off sx) in
  let at d = Array.mapi (fun i e -> if i = a then d else e) b in
  Eval.eval ~by
    (Value.Gather
       {
         axis = a;
         idx = broadcast ~by (at sp.(a)) p;
         x = broadcast ~by (at sx.(a)) x;
       })

let take_along_axis ~axis p x = take_along ~by:"Nx.take_along_axis" axis p x

(* Functional updates *)

(* The flat position in [x] of each element of the selection [picks], [-1] where
   a position held in data lies outside its axis: a map over the selection's
   shape, reading each position held in data broadcast to it. A program reads
   at most [Prog.max_operands] operands and outputs, so more positions held in
   data than one map reads chain maps, each reading the address so far, [-1]
   once invalid. Their inputs are placed together first, so the address lies
   where one map would put it. *)
let targets ~by x picks =
  let s = shape x in
  let r = Array.length s in
  let sel = selection x picks in
  let rs = Array.length sel in
  let strides = Array.make r 1 in
  for a = r - 2 downto 0 do
    strides.(a) <- strides.(a + 1) * s.(a + 1)
  done;
  (* The positions held in data, broadcast to the selection, in pick order:
     over selection axes [j, j + k) for [k] axes of their own. *)
  let held j p =
    let ps = shape p in
    let k = Array.length ps in
    let placed =
      Array.init rs (fun i -> if i >= j && i < j + k then ps.(i - j) else 1)
    in
    broadcast ~by sel (reshaped ~by placed p)
  in
  let inputs =
    let j = ref 0 in
    List.concat_map
      (fun (p, a) ->
        match (p, a) with
        | New, _ ->
            incr j;
            []
        | At _, Some _ -> []
        | Span _, Some _ ->
            incr j;
            []
        | Rows ps, Some _ ->
            let v = held !j (written ~by ps) in
            incr j;
            [ v ]
        | Held p, Some _ ->
            let v = held !j p in
            j := !j + ndim p;
            [ v ]
        | From (start, _, _), Some _ ->
            let v = broadcast ~by sel start in
            incr j;
            [ v ]
        | (At _ | Span _ | Rows _ | Held _ | From _), None -> [])
      picks
  in
  let room = P.max_operands - 1 in
  let inputs =
    if List.length inputs <= room then inputs
    else
      let ps = Array.of_list (List.map Prim.at inputs) in
      let shapes = Array.make (Array.length ps) sel in
      match Route.route ~by Elementwise ps shapes with
      | None -> inputs
      | Some route -> List.map (Eval.place ~by route.result) inputs
  in
  let inputs = ref inputs in
  (* The current map's loads and their dtypes, last first. *)
  let loads = ref [] and dtypes = ref [] in
  let ins () = Iarray.of_list (List.rev !dtypes) in
  let read v =
    loads := Value.Plain v :: !loads;
    dtypes := D.Any (dtype v) :: !dtypes;
    E.input (List.length !dtypes - 1) (dtype v)
  in
  let sum = ref (i64 0) and valid = ref None in
  let both ok =
    valid := match !valid with None -> Some ok | Some v -> Some E.(v && ok)
  in
  (* The address so far: the current map's output. *)
  let address () =
    let out =
      match !valid with
      | None -> !sum
      | Some ok -> E.(where ok !sum (i64 (-1)))
    in
    let loads = Iarray.of_list (List.rev !loads) in
    let prog = P.of_exprs ~ins:(ins ()) [| E.any out |] in
    let v, () =
      Eval.eval ~by
        (Value.Map
           {
             layout = Nx_array.Layout.contiguous sel;
             prog;
             outs = Value.[ D.Int64 ];
             loads;
           })
    in
    v
  in
  (* The next position held in data, in a map with room to read it: a full
     map's address carries into a new one, valid where it is not [-1]. *)
  let next () =
    if List.length !loads = room then begin
      let carry = address () in
      loads := [];
      dtypes := [];
      valid := None;
      sum := read carry;
      both E.(i64 0 <= !sum)
    end;
    match !inputs with
    | v :: rest ->
        inputs := rest;
        read v
    | [] -> invalid_arg "Nx.targets: a pick without its input"
  in
  let add a index = sum := E.(!sum + (index * i64 strides.(a))) in
  (* A position held in data, and whether it lies in [0, d). *)
  let checked d =
    let i = next () in
    both E.(i64 0 <= i && i < i64 d);
    i
  in
  let j = ref 0 in
  List.iter
    (fun (p, a) ->
      match (p, a) with
      | New, _ -> incr j
      | At i, Some a -> add a (i64 i)
      | Span g, Some a ->
          add a E.(i64 g.start + (coord !j * i64 g.step));
          incr j
      | Rows _, Some a ->
          add a (checked s.(a));
          incr j
      | Held p, Some a ->
          add a (checked s.(a));
          j := !j + ndim p
      | From (_, n, d), Some a ->
          let st = next () in
          let hi = i64 (d - n) in
          let c = E.(minimum (maximum st (i64 0)) hi) in
          add a E.(c + coord !j);
          incr j
      | (At _ | Span _ | Rows _ | Held _ | From _), None -> ())
    picks;
  let used = List.length (List.filter (fun (_, a) -> a <> None) picks) in
  for a = used to r - 1 do
    add a (E.coord !j);
    incr j
  done;
  address ()

(* The positions a pick held in data, or written as a list, gives its axis, of
   their own shape: [None] for a pick of the program's ranges. *)
let positions ~by = function
  | Held p -> Some p
  | Rows ps -> Some (written ~by ps)
  | From (start, n, d) -> Some (window ~by start n d)
  | At _ | Span _ | New -> None

let set idx v x =
  let by = "Nx.set" in
  let picks = picks ~by idx x in
  List.iter
    (fun (p, _) ->
      match p with
      | Rows ps ->
          if List.length (List.sort_uniq compare ps) <> List.length ps then
            invalid_argf "%s: a list of positions repeats one" by
      | At _ | Span _ | Held _ | From _ | New -> ())
    picks;
  let sel = selection x picks in
  let fits =
    match Prim.broadcast_shape ~by (shape v) sel with
    | s -> s = sel
    | exception Invalid_argument _ -> false
  in
  if not fits then
    invalid_argf "%s: %a does not broadcast to the selection %a" by pp_value v
      pp_shape sel;
  let s = shape x in
  let v = broadcast ~by sel v in
  let whole_axis (p, a) =
    match (p, a) with
    | Span r, Some a -> r.count = s.(a) && r.step = 1
    | New, None -> true
    | (At _ | Span _ | Rows _ | Held _ | From _ | New), _ -> false
  in
  let in_program (p, _) =
    match p with
    | At _ | Span _ | New -> true
    | Rows _ | Held _ | From _ -> false
  in
  match List.partition in_program picks with
  | ranges, [] ->
      (* Positions written in the program alone: one assembly, [x] then [v] at
         the region the ranges keep. *)
      let region = Array.map whole s in
      List.iter
        (fun (p, a) ->
          match (p, a) with
          | At i, Some a -> region.(a) <- { start = i; count = 1; step = 1 }
          | Span r, Some a -> region.(a) <- r
          | (At _ | Span _ | Rows _ | Held _ | From _ | New), _ -> ())
        ranges;
      let counts =
        Array.map (fun (r : Nx_array.Move.range) -> r.count) region
      in
      assemble ~by (dtype x) s
        (D.zero (dtype x))
        [
          (all_of s, x);
          (Iarray.of_array region, reshaped ~by counts v);
        ]
  | ranges, [ ((data, Some a) as pick) ] when List.for_all whole_axis ranges ->
      (* One axis selected by positions, every other whole: a scatter along it,
         the positions broadcast across the others. *)
      let p = Option.get (positions ~by data) in
      let n = numel p in
      let wide = Array.mapi (fun i d -> if i = a then n else d) s in
      let col = Array.mapi (fun i _ -> if i = a then n else 1) s in
      let idx = broadcast ~by wide (reshaped ~by col p) in
      let unique = match fst pick with Held _ -> false | _ -> true in
      Eval.eval ~by
        (Value.Scatter
           {
             combine = Set;
             unique;
             axis = a;
             idx;
             updates = reshaped ~by wide v;
             into = x;
           })
  | _ ->
      (* Any other selection: each selected element's flat position. *)
      let m = Array.fold_left ( * ) 1 sel in
      let line u = move ~by (Reshape [| m |]) u in
      let unique =
        List.for_all
          (fun (p, _) -> match p with Held _ -> false | _ -> true)
          picks
      in
      let updated =
        Eval.eval ~by
          (Value.Scatter
             {
               combine = Set;
               unique;
               axis = 0;
               idx = line (targets ~by x picks);
               updates = line v;
               into = move ~by (Reshape [| numel x |]) x;
             })
      in
      reshaped ~by s updated

type combine = Nx_kernel.combine = Set | Add | Max | Min

let scatter ?(combine = Set) ~axis:a p u x =
  let by = "Nx.scatter" in
  let axis = axis ~by x a in
  Eval.eval ~by
    (Value.Scatter
       { combine; unique = false; axis; idx = p; updates = u; into = x })

(* Reductions and scans *)

(* The distinct axes [l] of [x], sorted: every axis without [l]. *)
let reduced_axes ~by x l =
  match l with
  | None -> Iarray.init (ndim x) Fun.id
  | Some l -> Iarray.of_list (List.sort Int.compare (axes ~by x l))

(* The one-operand reduction [m] of [x] along [axes], rounded to [dt]. *)
let reduce (type v s w r d) ~by m (dt : (w, r) D.t) axes (x : (v, s, d) t) :
    (w, r, d) t =
  let prog = P.of_node ~ins:[| D.Any (dtype x) |] (In 0) in
  let y, () =
    Eval.eval ~by
      (Value.Reduce
         {
           layout = Nx_array.Layout.contiguous (shape x);
           axes;
           prog;
           reductions = Value.[ Monoid (m, 0, dt) ];
           loads = [| Plain x |];
         })
  in
  y

(* [y], reduced from a value of shape [s] along [axes], with each reduced axis
   kept of extent 1 where [keepdims]. *)
let kept ~keepdims s axes y =
  if not keepdims then y
  else reshape (Array.mapi (fun a n -> if Iarray.mem a axes then 1 else n) s) y

let reduction ~by m ?axes ?(keepdims = false) x =
  let s = shape x in
  let axes = reduced_axes ~by x axes in
  kept ~keepdims s axes (reduce ~by m (dtype x) axes x)

let sum ?axes ?keepdims x = reduction ~by:"Nx.sum" Sum ?axes ?keepdims x
let prod ?axes ?keepdims x = reduction ~by:"Nx.prod" Prod ?axes ?keepdims x
let max ?axes ?keepdims x = reduction ~by:"Nx.max" Max ?axes ?keepdims x
let min ?axes ?keepdims x = reduction ~by:"Nx.min" Min ?axes ?keepdims x

(* The sum at [acc], divided there by the count [n], stored in [x]'s dtype: each
   element rounds once to it. *)
let mean (type v s d) ?axes ?(keepdims = false) (x : (v, s, d) t) : (v, s, d) t
    =
  let by = "Nx.mean" in
  let s = shape x and dt = dtype x in
  let axes = reduced_axes ~by x axes in
  let n = Iarray.fold_left (fun n a -> n * s.(a)) 1 axes in
  let within (type w r) (acc : (w, r) D.t) (count : (w, r, d) t) =
    kept ~keepdims s axes (cast dt (div (reduce ~by Sum acc axes x) count))
  in
  match D.kind dt with
  | D.Float ->
      let (Prim.Wide w) = Prim.wide dt in
      within w (scalar w (Float.of_int n))
  | D.Complex -> within dt (scalar dt { Complex.re = Float.of_int n; im = 0. })
  | D.Signed | D.Unsigned | D.Boolean ->
      invalid_argf "%s: %s is neither a float nor a complex dtype" by
        (D.name dt)

(* The inclusive prefix of [m] along [axis] of [x]. *)
let scan_along ~by m axis x =
  let dt = dtype x in
  Eval.eval ~by
    (Value.Scan
       {
         layout = Nx_array.Layout.contiguous (shape x);
         axis;
         prog = P.of_node ~ins:[| D.Any dt |] (In 0);
         reduction = Monoid (m, 0, dt);
         loads = [| Plain x |];
         carry = None;
       })

(* Along [axis], or over the elements in C order, keeping [x]'s shape. *)
let scan ~by m ?axis:along x =
  match along with
  | None ->
      let s = shape x in
      reshaped ~by s (scan_along ~by m 0 (reshaped ~by [| numel x |] x))
  | Some a -> scan_along ~by m (axis ~by x a) x

let cumsum ?axis x = scan ~by:"Nx.cumsum" Sum ?axis x
let cumprod ?axis x = scan ~by:"Nx.cumprod" Prod ?axis x
let cummax ?axis x = scan ~by:"Nx.cummax" Max ?axis x
let cummin ?axis x = scan ~by:"Nx.cummin" Min ?axis x

(* Contraction *)

let contract ?sizes ?acc ?init dt p a b =
  let acc = Option.map (fun dt -> D.Any dt) acc in
  Contraction.contract ~by:"Nx.contract" ?sizes ?acc ?init dt p a b

let einsum ?sizes ?acc p xs =
  let acc = Option.map (fun dt -> D.Any dt) acc in
  Contraction.einsum ~by:"Nx.einsum" ?sizes ?acc p xs

let matmul a b = Contraction.matmul ~by:"Nx.matmul" a b

(* Compositions *)

(* Operand [k] of a map, the load of [x]. *)
let operand k x = E.input k (dtype x)

(* [x] rounded once to [dt]: itself at [dt]. *)
let at (type v s w r) (dt : (v, s) D.t) (x : (w, r) E.t) : (v, s) E.t =
  match D.equal_witness (E.dtype x) dt with
  | Some Type.Equal -> x
  | None -> E.cast dt x

(* The map [out] over [xs] broadcast together, its one result of [dt]: its
   operand [k] reads [xs]' element [k]. One pass over the operands. *)
let elementwise (type v s) ~by (dt : (v, s) D.t) xs (out : (v, s) E.t) =
  let s =
    List.fold_left
      (fun s (Value.Any x) -> Prim.broadcast_shape ~by s (shape x))
      [||] xs
  in
  let load (Value.Any x) = Value.Plain (broadcast ~by s x) in
  let ins = Iarray.of_list (List.map (fun (Value.Any x) -> D.Any (dtype x)) xs) in
  let prog = P.of_exprs ~ins [| E.any out |] in
  let layout = Nx_array.Layout.contiguous s in
  let y, () =
    Eval.eval ~by
      (Value.Map
         {
           layout;
           prog;
           outs = Value.[ dt ];
           loads = Iarray.of_list (List.map load xs);
         })
  in
  y

(* [x <> 0] for [x] of [dt], [0 / 1] for booleans. *)
let not_zero dt x = E.(x <> const dt (D.zero dt))

(* [one] where [t] holds, zero elsewhere, in [dt]. *)
let indicator dt t = E.(where t (const dt (D.one dt)) (const dt (D.zero dt)))

(* One where [k] of the operands' truths holds, zero elsewhere. *)
let logical (type v s d) ~by k (a : (v, s, d) t) (b' : (v, s, d) t) =
  let dt = dtype a in
  match dt with
  | D.Bool -> binary ~by (Binary k) a b'
  | _ ->
  let x = not_zero dt (operand 0 a) and y = not_zero dt (operand 1 b') in
  elementwise ~by dt [ Any a; Any b' ] (indicator dt (E.binary k x y))

let logical_and a b = logical ~by:"Nx.logical_and" And a b
let logical_or a b = logical ~by:"Nx.logical_or" Or a b
let logical_xor a b = logical ~by:"Nx.logical_xor" Xor a b

let logical_not x =
  let dt = dtype x in
  elementwise ~by:"Nx.logical_not" dt [ Any x ]
    (indicator dt E.(operand 0 x = const dt (D.zero dt)))

let integer_or_boolean (type v s) ~by (dt : (v, s) D.t) =
  match D.kind dt with
  | D.Signed | D.Unsigned | D.Boolean -> ()
  | D.Float | D.Complex ->
      invalid_argf "%s: %s is neither an integer nor a boolean dtype" by (D.name dt)

let integer (type v s) ~by (dt : (v, s) D.t) =
  match D.kind dt with
  | D.Signed | D.Unsigned -> ()
  | D.Float | D.Complex | D.Boolean ->
      invalid_argf "%s: %s is not an integer dtype" by (D.name dt)

(* The int64 [v] stored into [dt], modulo its width. *)
let wrapped dt v = E.(cast dt (const D.Int64 v))

let bitwise_not x =
  let by = "Nx.bitwise_not" in
  let dt = dtype x in
  integer_or_boolean ~by dt;
  elementwise ~by dt [ Any x ] E.(operand 0 x lxor wrapped dt (-1L))

let lshift x n =
  let by = "Nx.lshift" in
  let dt = dtype x in
  integer ~by dt;
  if n < 0 then invalid_argf "%s: shift %d" by n;
  let factor = if n < 64 then Int64.shift_left 1L n else 0L in
  elementwise ~by dt [ Any x ] E.(operand 0 x * wrapped dt factor)

(* [x / 2^n] rounded toward negative infinity, so a signed integer keeps its
   sign. Up to 32 bits, at float64, which holds every such integer and its
   quotient exactly: one multiply and a floor. Wider, from the truncated
   quotient and the remainder's sign. Past the width only the sign is left. *)
let rshift x n =
  let by = "Nx.rshift" in
  let dt = dtype x in
  integer ~by dt;
  if n < 0 then invalid_argf "%s: shift %d" by n;
  let bits = D.bits dt in
  let unsigned = D.is D.Unsigned dt in
  let int v = wrapped dt v in
  let y = operand 0 x in
  elementwise ~by dt [ Any x ]
    (if unsigned && n >= bits then int 0L
     else if (not unsigned) && n >= bits - 1 then
       E.(where (y < int 0L) (int (-1L)) (int 0L))
     else if bits <= 32 then
       let scale = E.const D.Float64 (Float.ldexp 1. (-n)) in
       E.(cast dt (floor (cast D.Float64 y * scale)))
     else
       let d = int (Int64.shift_left 1L n) in
       let q = E.(y / d) in
       if unsigned then q
       else E.(q - where (y mod d < int 0L) (int 1L) (int 0L)))

let clamp ?min:lo ?max:hi x =
  let by = "Nx.clamp" in
  if lo = None && hi = None then x
  else
    let dt = dtype x in
    let y = operand 0 x in
    let v = match lo with None -> y | Some lo -> E.maximum y (const ~by dt lo) in
    elementwise ~by dt [ Any x ]
      (match hi with None -> v | Some hi -> E.minimum v (const ~by dt hi))

let isnan x = comparison ~by:"Nx.isnan" (Compare Not_equal) x x

(* A complex value's parts, as views of its bits read as pairs of its parts'
   format: [part 0] its real parts, [part 1] its imaginary parts. *)
type ('s, 'd) parts =
  | Parts : (float, 'r) D.t * (int -> (float, 'r, 'd) t) -> ('s, 'd) parts

let parts (type s d) ~by (z : (Complex.t, s, d) t) : (s, d) parts =
  let split (type r) (f : (float, r) D.t) =
    let pairs : (float, r, d) t = Eval.eval ~by (Value.Bitcast (f, z)) in
    let r = Prim.rank z in
    let part i =
      let one = Array.init (r + 1) (fun a ->
        if a = r then { Nx_array.Move.start = i; count = 1; step = 1 }
        else whole (Prim.dim z a))
      in
      reshaped ~by (shape z) (sliced ~by one pairs)
    in
    Parts (f, part)
  in
  match dtype z with D.Complex64 -> split D.Float32 | D.Complex128 -> split D.Float64

(* Where a float of [dt]'s format is an infinity: [false] in a format with
   none. *)
let infinite (type s) (dt : (float, s) D.t) x =
  if not (D.float_format dt).infinities then E.const D.Bool false
  else E.(abs x = const dt Float.infinity)

(* Where a float of [dt]'s format is neither an infinity nor a NaN. *)
let finite (type s) (dt : (float, s) D.t) x =
  if not (D.float_format dt).infinities then E.(x = x)
  else E.(abs x < const dt Float.infinity)

(* [x]'s float predicate [p], each part's joined by [k] for a complex [x], and
   [other] for the other dtypes, where [x] lies. *)
let predicate (type v s d) ~by
    (p : 'r. (float, 'r) D.t -> (float, 'r) E.t -> (bool, D.bool_elt) E.t) k
    other (x : (v, s, d) t) : d bool_t =
  match D.kind (dtype x) with
  | D.Float -> elementwise ~by D.Bool [ Any x ] (p (dtype x) (operand 0 x))
  | D.Complex ->
      let (Parts (f, part)) = parts ~by x in
      let re = part 0 and im = part 1 in
      elementwise ~by D.Bool [ Any re; Any im ]
        (E.binary k (p f (operand 0 re)) (p f (operand 1 im)))
  | D.Signed | D.Unsigned | D.Boolean ->
      beside ~by x (fill ~by D.Bool (shape x) other)

let isinf x = predicate ~by:"Nx.isinf" (fun dt x -> infinite dt x) Or false x
let isfinite x = predicate ~by:"Nx.isfinite" (fun dt x -> finite dt x) And true x

(* Float functions *)

(* The float function [f w ins] of [xs], of one dtype, computed in its wide
   format [w] and rounded once to that dtype. *)
let float_map (type v s d) ~by (xs : (v, s, d) t list)
    (f : 'w. (float, 'w) D.t -> (float, 'w) E.t array -> (float, 'w) E.t) :
    (v, s, d) t =
  let dt = dtype (List.hd xs) in
  match D.kind dt with
  | D.Float ->
      let (Prim.Wide w) = Prim.wide dt in
      let ins = Array.of_list (List.mapi (fun k x -> at w (operand k x)) xs) in
      elementwise ~by dt (List.map (fun x -> Value.Any x) xs) (at dt (f w ins))
  | D.Complex | D.Signed | D.Unsigned | D.Boolean ->
      invalid_argf "%s: %s is not a float dtype" by (D.name dt)


let square x = binary ~by:"Nx.square" (Binary Mul) x x

let rsqrt x =
  float_map ~by:"Nx.rsqrt" [ x ] (fun _ ins -> E.(recip (sqrt ins.(0))))

(* [sqrt (x^2 + y^2)] with nothing overflowing: at float32, the squares summed
   at float64, which holds them; at float64, [m sqrt (1 + (n / m)^2)] for the
   larger magnitude [m] and the smaller [n], zero for two zeros. An infinity
   where either is, even beside a NaN. *)
let hypotenuse (type w) (w : (float, w) D.t) x y =
  let lit = E.const w in
  let ax = E.abs x and ay = E.abs y in
  let h =
    match w with
    | D.Float32 ->
        let x = E.cast D.Float64 x and y = E.cast D.Float64 y in
        E.(cast w (sqrt (fma x x (y * y))))
    | _ ->
        let m = E.maximum ax ay in
        let r = E.(minimum ax ay / m) in
        let s = E.(sqrt (fma r r (lit 1.))) in
        E.(where (m = lit 0.) (lit 0.) (m * s))
  in
  let inf v = E.(v = lit Float.infinity) in
  E.(where (inf ax || inf ay) (lit Float.infinity) h)

let hypot x y =
  float_map ~by:"Nx.hypot" [ x; y ] (fun w ins -> hypotenuse w ins.(0) ins.(1))

(* Past [big] a square would overflow float32: [log (2 x)] is the function to
   the format's precision there. Below [tiny], [x] is. *)
let big = 0x1p28
let tiny = 0x1p-28

(* log 2 rounded to a double, written out: [Float.log] calls the C library,
   whose last bit may differ by system, and on Windows by domain. *)
let ln2 = 0x1.62e42fefa39efp-1

let asinh x =
  float_map ~by:"Nx.asinh" [ x ] (fun w ins ->
      let lit = E.const w and x = ins.(0) in
      let a = E.abs x in
      let far = E.(log a + lit ln2) in
      let a2 = E.(a * a) in
      let root = E.(sqrt (lit 1. + a2)) in
      let near = E.(log1p (a + (a2 / (lit 1. + root)))) in
      let r = E.(where (lit big <= a) far near) in
      let signed = E.(where (x < lit 0.) (-r) r) in
      E.(where (a < lit tiny) x signed))

let acosh x =
  float_map ~by:"Nx.acosh" [ x ] (fun w ins ->
      let lit = E.const w and x = ins.(0) in
      let far = E.(log x + lit ln2) in
      let root = E.(sqrt (fma x x (lit (-1.)))) in
      let mid = E.(log (x + x - recip (x + root))) in
      let t = E.(x - lit 1.) in
      let near = E.(log1p (t + sqrt (fma t t (t + t)))) in
      let r = E.(where (lit 2. <= x) mid near) in
      let r = E.(where (lit big <= x) far r) in
      E.(where (x < lit 1.) (lit Float.nan) r))

(* [log1p (2a / (1 - a)) / 2] of [a = |x|], the sign put back: a small [a]
   splits the quotient to keep its digits. *)
let atanh x =
  float_map ~by:"Nx.atanh" [ x ] (fun w ins ->
      let lit = E.const w and x = ins.(0) in
      let a = E.abs x in
      let t = E.(a + a) in
      let one_minus = E.(lit 1. - a) in
      let small = E.(t + (t * a / one_minus)) in
      let large = E.(t / one_minus) in
      let q = E.(where (a < lit 0.5) small large) in
      let r = E.(lit 0.5 * log1p q) in
      let signed = E.(where (x < lit 0.) (-r) r) in
      E.(where (a < lit tiny) x signed))

let erfc x = float_map ~by:"Nx.erfc" [ x ] (fun _ i -> Special.erfc i.(0))
let erfinv p = float_map ~by:"Nx.erfinv" [ p ] (fun _ i -> Special.erfinv i.(0))
let ndtr x = float_map ~by:"Nx.ndtr" [ x ] (fun _ i -> Special.ndtr i.(0))

let log_ndtr x =
  float_map ~by:"Nx.log_ndtr" [ x ] (fun _ i -> Special.log_ndtr i.(0))

let ndtri p = float_map ~by:"Nx.ndtri" [ p ] (fun _ i -> Special.ndtri i.(0))
let lgamma x = float_map ~by:"Nx.lgamma" [ x ] (fun _ i -> Special.lgamma i.(0))

let digamma x =
  float_map ~by:"Nx.digamma" [ x ] (fun _ i -> Special.digamma i.(0))

let lbeta a b =
  float_map ~by:"Nx.lbeta" [ a; b ] (fun _ i -> Special.lbeta i.(0) i.(1))

let gammainc a x =
  float_map ~by:"Nx.gammainc" [ a; x ] (fun _ i -> Special.gammainc i.(0) i.(1))

let gammaincc a x =
  float_map ~by:"Nx.gammaincc" [ a; x ] (fun _ i ->
      Special.gammaincc i.(0) i.(1))

let log_gammainc a x =
  float_map ~by:"Nx.log_gammainc" [ a; x ] (fun _ i ->
      Special.log_gammainc i.(0) i.(1))

let log_gammaincc a x =
  float_map ~by:"Nx.log_gammaincc" [ a; x ] (fun _ i ->
      Special.log_gammaincc i.(0) i.(1))

let gammaincinv a p =
  float_map ~by:"Nx.gammaincinv" [ a; p ] (fun _ i ->
      Special.gammaincinv i.(0) i.(1))

let gammainccinv a q =
  float_map ~by:"Nx.gammainccinv" [ a; q ] (fun _ i ->
      Special.gammainccinv i.(0) i.(1))

let betainc a b x =
  float_map ~by:"Nx.betainc" [ a; b; x ] (fun _ i ->
      Special.betainc i.(0) i.(1) i.(2))

let betaincc a b x =
  float_map ~by:"Nx.betaincc" [ a; b; x ] (fun _ i ->
      Special.betaincc i.(0) i.(1) i.(2))

let log_betainc a b x =
  float_map ~by:"Nx.log_betainc" [ a; b; x ] (fun _ i ->
      Special.log_betainc i.(0) i.(1) i.(2))

let log_betaincc a b x =
  float_map ~by:"Nx.log_betaincc" [ a; b; x ] (fun _ i ->
      Special.log_betaincc i.(0) i.(1) i.(2))

let i0e x = float_map ~by:"Nx.i0e" [ x ] (fun _ i -> Special.i0e i.(0))
let i1e x = float_map ~by:"Nx.i1e" [ x ] (fun _ i -> Special.i1e i.(0))

(* Statistics *)

(* The reductions [rs] along [axes] of [out] over [xs], each of shape [s]: its
   operand [k] reads [xs]' element [k]. *)
let reduce_by ~by s axes xs rs out =
  let ins = Iarray.of_list (List.map (fun (Value.Any x) -> D.Any (dtype x)) xs) in
  let prog = P.of_exprs ~ins [| E.any out |] in
  let load (Value.Any x) = Value.Plain x in
  Eval.eval ~by
    (Value.Reduce
       {
         layout = Nx_array.Layout.contiguous s;
         axes;
         prog;
         reductions = rs;
         loads = Iarray.of_list (List.map load xs);
       })

let reduced_count s axes = Iarray.fold_left (fun n a -> n * s.(a)) 1 axes

(* [x]'s elements as booleans reduced by [m] along [axes]: booleans reduce
   as they are, a plain loop. An extreme of no term has no kernel: there,
   whether the count of [true]s is not zero, [Max]'s [false], or whether the
   count of [false]s is zero, [Min]'s [true]. *)
let truth (type v s d) ~by m ?axes ?(keepdims = false) (x : (v, s, d) t) =
  let s = shape x in
  let axes = reduced_axes ~by x axes in
  let y =
    if reduced_count s axes > 0 then
      match D.equal_witness (dtype x) D.Bool with
      | Some Type.Equal -> reduce ~by m D.Bool axes x
      | None ->
          fst
            (reduce_by ~by s axes [ Any x ]
               Value.[ Monoid (m, 0, D.Bool) ]
               (E.cast D.Bool (operand 0 x)))
    else
      let n, () =
        reduce_by ~by s axes [ Any x ]
          Value.[ Monoid (Sum, 0, D.Int64) ]
          E.(cast D.Int64 (cast D.Bool (operand 0 x)))
      in
      match (m : Nx_kernel.monoid) with
      | Max -> not_equal_s n 0L
      | _ -> equal_s n 0L
  in
  kept ~keepdims s axes y

let all ?axes ?keepdims x = truth ~by:"Nx.all" Min ?axes ?keepdims x
let any ?axes ?keepdims x = truth ~by:"Nx.any" Max ?axes ?keepdims x

let count ?axes ?(keepdims = false) m =
  let by = "Nx.count" in
  let s = shape m in
  let axes = reduced_axes ~by m axes in
  let y, () =
    reduce_by ~by s axes [ Any m ]
      Value.[ Monoid (Sum, 0, D.Int64) ]
      (E.cast D.Int64 (operand 0 m))
  in
  kept ~keepdims s axes y

(* The first position of the extreme [e] along [axis], or among all the elements
   in C order. *)
let arg ~by e ?axis:along ?(keepdims = false) x =
  let s = shape x in
  let x, a =
    match along with
    | None -> (reshape [| numel x |] x, 0)
    | Some a -> (x, axis ~by x a)
  in
  let (_, i), () =
    reduce_by ~by (shape x) [| a |] [ Any x ]
      Value.[ Arg (e, 0, dtype x) ]
      (operand 0 x)
  in
  match along with
  | None ->
      if keepdims then reshape (Array.map (fun _ -> 1) s) i else reshape [||] i
  | Some _ -> kept ~keepdims s [| a |] i

let argmax ?axis ?keepdims x = arg ~by:"Nx.argmax" Max ?axis ?keepdims x
let argmin ?axis ?keepdims x = arg ~by:"Nx.argmin" Min ?axis ?keepdims x

(* The variance by the kernels' moments in [x]'s wide format, scaled by [n / (n
   - ddof)] and rooted where [root], then rounded once. *)
let spread (type s d) ~by ~root ?axes ?(keepdims = false) ?(ddof = 0)
    (x : (float, s, d) t) : (float, s, d) t =
  let s = shape x and dt = dtype x in
  let axes = reduced_axes ~by x axes in
  let n = reduced_count s axes in
  if ddof < 0 || (n > 0 && ddof >= n) then
    invalid_argf "%s: ddof %d for %d elements" by ddof n;
  let (Prim.Wide w) = Prim.wide dt in
  let (_, v), () =
    reduce_by ~by s axes [ Any x ]
      Value.[ Moments (0, w) ]
      (operand 0 x)
  in
  let y =
    if ddof = 0 && not root then cast dt v
    else
      let e = operand 0 v in
      let scale = Float.of_int n /. Float.of_int (n - ddof) in
      let e = if ddof = 0 then e else E.(e * const w scale) in
      let e = if root then E.sqrt e else e in
      elementwise ~by dt [ Any v ] (E.cast dt e)
  in
  kept ~keepdims s axes y

let var ?axes ?keepdims ?ddof x =
  spread ~by:"Nx.var" ~root:false ?axes ?keepdims ?ddof x

let std ?axes ?keepdims ?ddof x =
  spread ~by:"Nx.std" ~root:true ?axes ?keepdims ?ddof x

let logsumexp ?axes ?keepdims x =
  reduction ~by:"Nx.logsumexp" Logsumexp ?axes ?keepdims x

let logmeanexp (type s d) ?axes ?(keepdims = false) (x : (float, s, d) t) :
    (float, s, d) t =
  let by = "Nx.logmeanexp" in
  let s = shape x and dt = dtype x in
  let axes = reduced_axes ~by x axes in
  let n = Float.of_int (reduced_count s axes) in
  let (Prim.Wide w) = Prim.wide dt in
  let l = reduce ~by Logsumexp w axes x in
  let y =
    elementwise ~by dt [ Any l ] E.(cast dt (operand 0 l - log (const w n)))
  in
  kept ~keepdims s axes y

(* The affine maps [y -> a y + b] of the recurrence, composed by
   {!Pairwise.scan}: [(a1, b1)] then [(a2, b2)] is [(a1 a2, a2 b1 + b2)],
   one fma, or [b1 + b2] where [b1] is infinite, so that an [a] that
   underflowed to [0] still carries an infinite [b] past it. *)
let average (type s d) ~by ~alpha axis (x : (float, s, d) t) : (float, s, d) t =
  let s = shape x and dt = dtype x in
  let go (type w) (w : (float, w) D.t) =
    let terms =
      let first = E.(coord axis = const D.Int64 0L) in
      let lit = E.const w and t = E.input 0 w in
      let a = E.where first (lit 0.) (lit (1. -. alpha)) in
      let c = E.(where first t (lit alpha * t)) in
      let prog = P.of_exprs ~ins:[| D.Any w |] [| E.any a; E.any c |] in
      let layout = Nx_array.Layout.contiguous s in
      let a, (c, ()) =
        let loads : _ Value.load iarray = [| Value.Plain (cast w x) |] in
        Eval.eval ~by (Value.Map { layout; prog; outs = Value.[ w; w ]; loads })
      in
      [| Value.Any a; Value.Any c |]
    in
    let combine =
      let a1 = E.input 0 w and b1 = E.input 1 w in
      let a2 = E.input 2 w and b2 = E.input 3 w in
      let infinite = E.(abs b1 = const w Float.infinity) in
      E.[| any (a1 * a2); any (where infinite (b1 + b2) (fma a2 b1 b2)) |]
    in
    let y = Pairwise.scan Eval.eval ~by s axis combine terms in
    cast dt (Prim.expect w y.(1))
  in
  let (Prim.Wide w) = Prim.wide dt in
  go w

let ewma ?axis:along ~alpha x =
  let by = "Nx.ewma" in
  if not (alpha > 0. && alpha <= 1.) then
    invalid_argf "%s: alpha %g is outside (0, 1]" by alpha;
  match along with
  | None -> reshape (shape x) (average ~by ~alpha 0 (reshape [| numel x |] x))
  | Some a -> average ~by ~alpha (axis ~by x a) x

(* Activations *)

(* [1 / (1 + t)] above zero and [t / (1 + t)] below, [t = exp (-|x|)]: below
   zero [exp (-x)] would overflow where the result is still a subnormal. *)
(* CR: Share q = t / (1 + t), choosing q below zero and 1 - q otherwise.
   At float64 x = 37, 1 + t rounds to 1 and the reciprocal returns 1;
   the existing golden is the float just below 1. The complement keeps
   that tail and replaces the extra reciprocal with subtraction in this
   Map. Check endpoint classification exactly beside the ULP bound,
   which currently accepts this violation of sigmoid's contract. *)
let sigmoid x =
  float_map ~by:"Nx.sigmoid" [ x ] (fun w ins ->
      let lit = E.const w and x = ins.(0) in
      let t = E.(exp (-abs x)) in
      let d = E.(lit 1. + t) in
      E.(where (x < lit 0.) (t / d) (recip d)))

let softplus x =
  float_map ~by:"Nx.softplus" [ x ] (fun w ins ->
      let x = ins.(0) in
      E.(maximum x (const w 0.) + log1p (exp (-abs x))))

(* [finish w shifted sum] of each [scale x - m] and its lane's sum of
   exponentials, [m] the lane's greatest [scale x], computed in [x]'s wide
   format and rounded once. *)
let normalised (type s d) ~by
    (finish : 'w. (float, 'w) D.t -> (float, 'w) E.t -> (float, 'w) E.t -> (float, 'w) E.t)
    ?(axes = [ -1 ]) ?(scale = 1.) (x : (float, s, d) t) : (float, s, d) t =
  let s = shape x and dt = dtype x in
  let axes = reduced_axes ~by x (Some axes) in
  if numel x = 0 then x
  else
    let (Prim.Wide w) = Prim.wide dt in
    let scaled =
      let x = E.cast w (operand 0 x) in
      if scale = 1. then x else E.(x * const w scale)
    in
    let lanes y = broadcast ~by s (kept ~keepdims:true s axes y) in
    let m, () =
      reduce_by ~by s axes [ Any x ] Value.[ Monoid (Max, 0, w) ] scaled
    in
    let m = lanes m in
    let shifted = E.(scaled - operand 1 m) in
    let sum, () =
      reduce_by ~by s axes [ Any x; Any m ]
        Value.[ Monoid (Sum, 0, w) ]
        (E.exp shifted)
    in
    let sum = lanes sum in
    elementwise ~by dt
      [ Any x; Any m; Any sum ]
      (E.cast dt (finish w shifted (operand 2 sum)))

let softmax ?axes ?scale x =
  normalised ~by:"Nx.softmax" (fun _ d sum -> E.(exp d / sum)) ?axes ?scale x

let log_softmax ?axes ?scale x =
  normalised ~by:"Nx.log_softmax" (fun _ d sum -> E.(d - log sum)) ?axes ?scale x

(* Complex parts *)

let cast_by (type v s w r d) ~by (dt : (w, r) D.t) (x : (v, s, d) t) :
    (w, r, d) t =
  match D.equal_witness (dtype x) dt with
  | Some Type.Equal -> x
  | None -> Eval.apply1 ~by Cast dt x

let real dt z =
  let by = "Nx.real" in
  let (Parts (_, part)) = parts ~by z in
  cast_by ~by dt (part 0)

let imag dt z =
  let by = "Nx.imag" in
  let (Parts (_, part)) = parts ~by z in
  cast_by ~by dt (part 1)

(* [body w re im] of [z]'s parts, computed in the wider of [dt] and the
   parts' format, then stored in [dt]. *)
let of_parts (type r s d) ~by (dt : (float, r) D.t) (z : (Complex.t, s, d) t)
    (body : 'w. (float, 'w) D.t -> (float, 'w) E.t -> (float, 'w) E.t -> (float, 'w) E.t) : (float, r, d) t =
  let (Parts (f, part)) = parts ~by z in
  let (Prim.Wide w) =
    if D.bits dt > D.bits f then Prim.Wide dt else Prim.Wide f
  in
  let re = part 0 and im = part 1 in
  let v = body w (at w (operand 0 re)) (at w (operand 1 im)) in
  elementwise ~by dt [ Any re; Any im ] (at dt v)

let magnitude dt z =
  of_parts ~by:"Nx.magnitude" dt z (fun w re im -> hypotenuse w re im)

let angle dt z = of_parts ~by:"Nx.angle" dt z (fun _ re im -> E.atan2 im re)

let complex (type c a d) (dt : (Complex.t, c) D.t) ~(re : (float, a, d) t)
    ~(im : (float, a, d) t) : (Complex.t, c, d) t =
  let by = "Nx.complex" in
  let s = Prim.broadcast_shape ~by (shape re) (shape im) in
  let r = Array.length s in
  let pairs (type p) (f : (float, p) D.t) =
    let unit = Array.append s [| 1 |] and both = Array.append s [| 2 |] in
    let part x = reshaped ~by unit (broadcast ~by s (cast_by ~by f x)) in
    let pieces = [ (region unit r 0 1, part re); (region unit r 1 1, part im) ] in
    Eval.eval ~by (Value.Bitcast (dt, assemble ~by f both 0. pieces))
  in
  match dt with D.Complex64 -> pairs D.Float32 | D.Complex128 -> pairs D.Float64

(* A complex value's imaginary parts negated: for complex64, its imaginary
   part's sign bit flipped in the value read as uint64, in the host's byte
   order; for complex128, the odd elements of its pairs of parts negated, in
   one map. Any other value is itself. *)
let conjugate (type v s d) (x : (v, s, d) t) : (v, s, d) t =
  let by = "Nx.conjugate" in
  match dtype x with
  | D.Complex64 ->
      let sign = if Sys.big_endian then 0x8000_0000L else Int64.min_int in
      let u = Eval.eval ~by (Value.Bitcast (D.Uint64, x)) in
      let flipped = binary ~by (Binary Xor) u (fill ~by D.Uint64 [||] sign) in
      Eval.eval ~by (Value.Bitcast (D.Complex64, flipped))
  | D.Complex128 ->
      let pairs = Eval.eval ~by (Value.Bitcast (D.Float64, x)) in
      let last = ndim pairs - 1 in
      let y =
        let e = operand 0 pairs in
        elementwise ~by D.Float64 [ Any pairs ]
          E.(where (coord last = const D.Int64 1L) (-e) e)
      in
      Eval.eval ~by (Value.Bitcast (D.Complex128, y))
  | _ -> x

(* Matrices *)

let matrix ~by x =
  if ndim x < 2 then
    invalid_argf "%s: %a has fewer than two axes" by pp_value x

let matrix_transpose x =
  let r = ndim x in
  if r < 2 then x
  else
    let p = Array.init r Fun.id in
    p.(r - 2) <- r - 1;
    p.(r - 1) <- r - 2;
    permute ~by:"Nx.matrix_transpose" p x

(* [x] where column - row holds [keep] against [k] over its last two axes, zero
   elsewhere: the last axis is the column, the one before it the row. *)
let triangle ~by keep k x =
  matrix ~by x;
  let dt = dtype x in
  let r = ndim x in
  let column = E.coord (r - 1) and row = E.coord (r - 2) in
  let d = E.(column - row) and kk = E.const D.Int64 (Int64.of_int k) in
  let inside = match keep with `Below -> E.(d <= kk) | `Above -> E.(kk <= d) in
  elementwise ~by dt [ Any x ] E.(where inside (operand 0 x) (const dt (D.zero dt)))

let tril ?(k = 0) x = triangle ~by:"Nx.tril" `Below k x
let triu ?(k = 0) x = triangle ~by:"Nx.triu" `Above k x

(* The diagonal of [x]'s last two axes [m] and [n], [offset] above the main one,
   as a slice of every [n + 1]th element of the two axes merged: a view where
   they merge without a copy. *)
let last_diagonal ~by offset x =
  let s = shape x in
  let r = Array.length s in
  let m = s.(r - 2) and n = s.(r - 1) in
  let count =
    Stdlib.max 0
      (if offset >= 0 then Stdlib.min m (n - offset)
       else Stdlib.min (m + offset) n)
  in
  let start = if count = 0 then 0 else if offset >= 0 then offset else -offset * n in
  let lead = Array.sub s 0 (r - 2) in
  let flat = reshaped ~by (Array.append lead [| m * n |]) x in
  let ranges = Array.append (Array.map whole lead) [| { Nx_array.Move.start; count; step = n + 1 } |] in
  sliced ~by ranges flat

let diagonal_by ~by ?(offset = 0) ?(axis1 = -2) ?(axis2 = -1) x =
  matrix ~by x;
  let a = axis ~by x axis1 and b = axis ~by x axis2 in
  if a = b then invalid_argf "%s: axis1 and axis2 are both %d" by a;
  let rest = List.filter (fun i -> i <> a && i <> b) (List.init (ndim x) Fun.id) in
  let p = Array.of_list (rest @ [ a; b ]) in
  let last_two = Array.for_all2 ( = ) p (Array.init (ndim x) Fun.id) in
  last_diagonal ~by offset (if last_two then x else permute ~by p x)

let diagonal ?offset ?axis1 ?axis2 x =
  diagonal_by ~by:"Nx.diagonal" ?offset ?axis1 ?axis2 x

let trace ?(offset = 0) x =
  let by = "Nx.trace" in
  reduction ~by Sum ~axes:[ -1 ] (diagonal_by ~by ~offset x)

(* A vector [v] of [n] elements on the [k]th diagonal of an [n + |k|] square:
   [v] padded to the square's extent and read along the rows for [k >= 0],
   along the columns otherwise. *)
(* The [n × m] value whose [k]th diagonal holds [piece]'s elements, zero
   elsewhere: one assembly into the two axes merged, the diagonal every
   [m + 1]th element of them from its first. *)
let on_diagonal ~by dt n m k piece =
  ignore (renamed ~by (fun () -> Nx_array.Layout.contiguous [| n; m |]));
  let start = if k >= 0 then k else -k * m in
  let count = Prim.dim piece 0 in
  let along = { Nx_array.Move.start = (if count = 0 then 0 else start); count; step = m + 1 } in
  let flat = assemble ~by dt [| n * m |] (D.zero dt) [ ([| along |], piece) ] in
  move ~by (Reshape [| n; m |]) flat

let eye ?m ?(k = 0) dt n =
  let by = "Nx.eye" in
  let m = Option.value m ~default:n in
  let count =
    Stdlib.max 0
      (if k >= 0 then Stdlib.min n (m - k) else Stdlib.min (n + k) m)
  in
  on_diagonal ~by dt n m k (fill ~by dt [| count |] (D.one dt))

let diag ?(k = 0) v =
  let by = "Nx.diag" in
  match ndim v with
  | 2 -> diagonal_by ~by ~offset:k v
  | 1 ->
      let n = dim 0 v + Stdlib.abs k in
      on_diagonal ~by (dtype v) n n k v
  | _ -> invalid_argf "%s: %a is neither a vector nor a matrix" by pp_value v

let dot a b = Contraction.dot ~by:"Nx.dot" a b

let vdot a b =
  let by = "Nx.vdot" in
  if numel a <> numel b then
    invalid_argf "%s: %a and %a differ in their number of elements" by pp_value
      a pp_value b;
  let flat x = move ~by (Reshape [| numel x |]) x in
  Contraction.contract ~by (dtype a) (Pattern.v "k, k -> | k")
    (conjugate (flat a)) (flat b)

(* Linear algebra *)

type triangle = Nx_kernel.triangle = Lower | Upper
type factors = Nx_kernel.factors = Reduced | Complete

let routine ~by l = Eval.eval ~by (Value.Linalg l)

(* Refuses a dtype the routines do not take. *)
let routine_dtype ~by x =
  if not (Nx_kernel.full_precision (dtype x)) then
    invalid_argf "%s: %a is not float32, float64, complex64 or complex128" by
      pp_value x

(* Refuses what is not a batch of matrices of a dtype the routines take. *)
let matrices ~by x =
  matrix ~by x;
  routine_dtype ~by x

let square_matrices ~by x =
  matrices ~by x;
  if dim (-1) x <> dim (-2) x then
    invalid_argf "%s: %a is not square" by pp_value x

(* The machine epsilon of the four dtypes the routines take. *)
let epsilon (type v s) (dt : (v, s) D.t) =
  match dt with D.Float32 | D.Complex64 -> 0x1p-23 | _ -> epsilon_float

(* [x]'s batch: its axes before its last two. *)
let batch x = Array.sub (shape x) 0 (ndim x - 2)

(* An empty value of [x]'s dtype where [x] lies: the vectors a routine did not
   compute. *)
let uncomputed ~by x s = beside ~by x (fill ~by (dtype x) s (D.zero (dtype x)))

(* The first [rows] rows and [cols] columns of [x]'s matrices. *)
let corner ~by rows cols x =
  let s = shape x in
  let r = Array.length s in
  let ranges =
    Array.mapi
      (fun i d ->
        if i = r - 2 then { (whole d) with count = rows }
        else if i = r - 1 then { (whole d) with count = cols }
        else whole d)
      s
  in
  sliced ~by ranges x

(* [x]'s strictly lower triangle, ones on its diagonal, zero above. *)
let unit_lower ~by x =
  let dt = dtype x and r = ndim x in
  let column = E.coord (r - 1) and row = E.coord (r - 2) in
  let lit v = E.const dt v in
  elementwise ~by dt [ Any x ]
    E.(
      where (column < row) (operand 0 x)
        (where (column = row) (lit (D.one dt)) (lit (D.zero dt))))

(* [f] of [x]'s magnitudes, a float of [x]'s precision, its result back in
   [x]'s dtype: complex with imaginary parts [+0] for a complex [x]. *)
let magnitudes (type v s d) ~by (x : (v, s, d) t)
    (f : 'r. (float, 'r, d) t -> (float, 'r, d) t) : (v, s, d) t =
  match dtype x with
  | D.Float32 -> f (abs x)
  | D.Float64 -> f (abs x)
  | D.Complex64 -> cast_by ~by D.Complex64 (f (magnitude D.Float32 x))
  | D.Complex128 -> cast_by ~by D.Complex128 (f (magnitude D.Float64 x))
  | dt -> invalid_argf "%s: %s has no magnitudes" by (D.name dt)

(* [a] and [b] with their leading axes broadcast, [b] as columns of [rows]
   rows, and the function that gives a result over those columns [b]'s form:
   a [b] of one axis fewer than [a] is one column, whose axis goes. *)
let columns ~by ~rows a b =
  let ra = ndim a and rb = ndim b in
  let vector = rb = ra - 1 in
  if not (vector || rb = ra) then
    invalid_argf "%s: %a is neither a vector nor a matrix against %a" by
      pp_value b pp_value a;
  let b =
    if vector then reshaped ~by (Array.append (shape b) [| 1 |]) b else b
  in
  if dim (-2) b <> rows then
    invalid_argf "%s: %a has %d rows, not %d" by pp_value b (dim (-2) b) rows;
  let lead = broadcast_all ~by [ batch a; batch b ] in
  let along x =
    broadcast ~by (Array.append lead (Array.sub (shape x) (ndim x - 2) 2)) x
  in
  let back x =
    if vector then reshaped ~by (Array.sub (shape x) 0 (ndim x - 1)) x
    else x
  in
  (along a, along b, back)

let adjoint x = conjugate (matrix_transpose x)

(* The routines read the lower triangle. The Hermitian matrix an upper one
   names is the one the lower triangle of [aᴴ] names. *)
let lower_of triangle a = match triangle with Lower -> a | Upper -> adjoint a

let cholesky ?(triangle = Lower) a =
  let by = "Nx.cholesky" in
  square_matrices ~by a;
  let a = lower_of triangle a in
  routine ~by (Cholesky { triangle; a })

let qr ?(factors = Reduced) a =
  let by = "Nx.qr" in
  matrices ~by a;
  routine ~by (Qr { factors; a })

let lu a =
  let by = "Nx.lu" in
  matrices ~by a;
  let f, _, perm = routine ~by (Lu a) in
  let m = dim (-2) a and n = dim (-1) a in
  let k = Stdlib.min m n in
  let l = unit_lower ~by (corner ~by m k f) in
  let u = triangle ~by `Above 0 (corner ~by k n f) in
  (perm, l, u)

let svd ?(vectors = true) ?(factors = Reduced) a =
  let by = "Nx.svd" in
  matrices ~by a;
  if vectors then
    let s, (u, vh) = routine ~by (Svd { vectors = Vectors factors; a }) in
    (u, s, vh)
  else
    let s, () = routine ~by (Svd { vectors = No_vectors; a }) in
    let lead = batch a in
    let u = uncomputed ~by a (Array.append lead [| dim (-2) a; 0 |]) in
    let vh = uncomputed ~by a (Array.append lead [| 0; dim (-1) a |]) in
    (u, s, vh)

(* The [[…; n; 0]] vectors an eigenvalue routine did not compute. *)
let no_vectors ~by a =
  uncomputed ~by a (Array.append (batch a) [| dim (-1) a; 0 |])

let eig ?(vectors = true) a =
  let by = "Nx.eig" in
  square_matrices ~by a;
  let dtype = dtype a in
  if vectors then routine ~by (Eig { vectors = Vectors (); dtype; a })
  else
    let w, () = routine ~by (Eig { vectors = No_vectors; dtype; a }) in
    (w, no_vectors ~by a)

let eigh ?(vectors = true) ?(triangle = Lower) a =
  let by = "Nx.eigh" in
  square_matrices ~by a;
  let a' = lower_of triangle a in
  if vectors then routine ~by (Eigh { vectors = Vectors (); a = a' })
  else
    let w, () = routine ~by (Eigh { vectors = No_vectors; a = a' }) in
    (w, no_vectors ~by a)

(* The diagonal of [lu]'s factors, and the function that negates [x] where
   each matrix's rows were interchanged an odd number of times and makes a
   zero [+0]. *)
let pivots ~by a =
  let f, steps, _ = routine ~by (Lu a) in
  let k = dim (-1) steps in
  let moved = not_equal steps (arange D.Int64 0 k 1) in
  let odd = equal_s (mod_s (count ~axes:[ -1 ] moved) 2L) 1L in
  let signed x =
    let zero = zeros_like x in
    where (equal x zero) zero (where odd (neg x) x)
  in
  (diagonal_by ~by f, signed)

let det a =
  let by = "Nx.det" in
  renamed ~by (fun () ->
      square_matrices ~by a;
      let d, signed = pivots ~by a in
      signed (reduction ~by Prod ~axes:[ -1 ] d))

(* [z / |z|], zero where [z] is: a real's sign, a complex number's phase. *)
let phase (type v s d) ~by (z : (v, s, d) t) : (v, s, d) t =
  match D.kind (dtype z) with
  | D.Complex ->
      let m = magnitudes ~by z Fun.id in
      let zero = zeros_like z in
      where (equal m zero) zero (div z m)
  | D.Float | D.Signed | D.Unsigned | D.Boolean -> sign z

let slogdet a =
  let by = "Nx.slogdet" in
  renamed ~by (fun () ->
      square_matrices ~by a;
      let d, signed = pivots ~by a in
      let sign = signed (reduction ~by Prod ~axes:[ -1 ] (phase ~by d)) in
      (* A zero pivot beside an infinite one would sum to NaN. *)
      let singular = equal sign (zeros_like sign) in
      let logabs =
        magnitudes ~by d (fun m ->
            let l = reduction ~by Sum ~axes:[ -1 ] (log m) in
            where singular (full_like l Float.neg_infinity) l)
      in
      (sign, logabs))

type transpose = Transpose | Adjoint

let is_complex (type v s d) (x : (v, s, d) t) =
  match D.kind (dtype x) with
  | D.Complex -> true
  | D.Float | D.Signed | D.Unsigned | D.Boolean -> false

let solve_triangular ?(triangle = Lower) ?transpose ?(unit_diag = false) a b =
  let by = "Nx.solve_triangular" in
  square_matrices ~by a;
  let a, b, back = columns ~by ~rows:(dim (-1) a) a b in
  let solved adjoint b =
    routine ~by
      (Solve_triangular { triangle; adjoint; unit_diagonal = unit_diag; a; b })
  in
  back
    (match transpose with
    | None -> solved false b
    | Some Transpose when is_complex a ->
        (* [aᵀ x = b] is [aᴴ x̄ = b̄], and the kernel solves with [aᴴ]. *)
        conjugate (solved true (conjugate b))
    | Some (Transpose | Adjoint) -> solved true b)

(* [a]'s rows interchanged by {!lu}, then [L] and [U] solved in turn: the
   factors' two triangles, read from the one matrix. *)
let solve_by ~by a b =
  square_matrices ~by a;
  let f, _, perm = routine ~by (Lu a) in
  let perm = reshaped ~by (Array.append (shape perm) [| 1 |]) perm in
  let f, b, back = columns ~by ~rows:(dim (-1) a) f b in
  let b = take_along ~by (-2) perm b in
  let solved triangle unit_diagonal b =
    routine ~by
      (Solve_triangular { triangle; adjoint = false; unit_diagonal; a = f; b })
  in
  back (solved Upper false (solved Lower true b))

let solve a b = solve_by ~by:"Nx.solve" a b

let inv a =
  let by = "Nx.inv" in
  square_matrices ~by a;
  let n = dim (-1) a in
  solve_by ~by a (broadcast ~by (shape a) (eye (dtype a) n))

(* Each singular value's reciprocal where it is above [tol] times its
   matrix's greatest, zero at or below, NaN where it is. *)
let reciprocals ~by ~tol s =
  magnitudes ~by s (fun s ->
      let top = reduction ~by Max ~axes:[ -1 ] ~keepdims:true s in
      let above = greater s (mul_s top tol) in
      where (isnan s) s (where above (recip s) (zeros_like s)))

(* The cutoff of a singular value relative to its matrix's greatest, [rtol]
   (default the dtype's machine epsilon) times [max m n]. *)
let cutoff ~by rtol a =
  let rtol = Option.value rtol ~default:(epsilon (dtype a)) in
  if not (rtol >= 0.) then
    invalid_argf "%s: rtol %g is negative or NaN" by rtol;
  rtol *. Float.of_int (Stdlib.max (dim (-2) a) (dim (-1) a))

(* [vh]ᴴ [diag d] [u]ᴴ: the matrices whose singular values are [d]. *)
let resynthesise u d vh =
  matmul (mul (adjoint vh) (unsqueeze ~axes:[ -2 ] d)) (adjoint u)

let lstsq ?rtol a b =
  let by = "Nx.lstsq" in
  renamed ~by (fun () ->
      matrices ~by a;
      let m = dim (-2) a and n = dim (-1) a in
      let tol = cutoff ~by rtol a in
      let a', b', back = columns ~by ~rows:m a b in
      let x, s, rank =
        if Stdlib.min m n = 0 then
          let lead = batch b' in
          let zero dt s = beside ~by b' (fill ~by dt s (D.zero dt)) in
          ( zero (dtype b') (Array.append lead [| n; dim (-1) b' |]),
            zero (dtype a) (Array.append (batch a) [| 0 |]),
            zero D.Int64 (batch a) )
        else
          let u, s, vh = svd a in
          let d = reciprocals ~by ~tol s in
          let kept = count ~axes:[ -1 ] (not_equal d (zeros_like d)) in
          let failed = any ~axes:[ -1 ] (isnan s) in
          let rank = where failed (full_like kept (-1L)) kept in
          (* [V (Σ⁺ (Uᴴ b))] rounds less than [b] by the pseudoinverse. *)
          let c = mul (unsqueeze ~axes:[ -1 ] d) (matmul (adjoint u) b') in
          (matmul (adjoint vh) c, s, rank)
      in
      let r = sub b' (matmul a' x) in
      let residuals =
        magnitudes ~by r (fun r -> reduction ~by Sum ~axes:[ -2 ] (square r))
      in
      (back x, back residuals, rank, s))

let pinv ?rtol ?(hermitian = false) a =
  let by = "Nx.pinv" in
  renamed ~by (fun () ->
      if hermitian then square_matrices ~by a else matrices ~by a;
      let m = dim (-2) a and n = dim (-1) a in
      let tol = cutoff ~by rtol a in
      if Stdlib.min m n = 0 then
        let dt = dtype a in
        beside ~by a
          (fill ~by dt (Array.append (batch a) [| n; m |]) (D.zero dt))
      else if hermitian then begin
        let w, v = eigh a in
        (* An eigenvalue's reciprocal keeps its sign: its magnitude decides. *)
        let kept = reciprocals ~by ~tol w in
        let w' = where (equal kept (zeros_like kept)) kept (recip w) in
        matmul (mul v (unsqueeze ~axes:[ -2 ] w')) (adjoint v)
      end
      else
        let u, s, vh = svd a in
        resynthesise u (reciprocals ~by ~tol s) vh)

(* Norms *)

(* [c · (Σ (m / c)^p)^(1/p)] along [axes], [c] the greatest of the magnitudes
   [m] for a positive [p] and the least for a negative one, so that nothing
   overflows or underflows on the way; [c] itself where it is zero or
   infinite. Of no term the sum is 0, and its power [1/p] is 0 for a positive
   [p] and infinity for a negative one. *)
let power_norm ~by p axes m =
  let sum x = reduction ~by Sum ~axes ~keepdims:true x in
  if List.exists (fun a -> dim a m = 0) axes then
    if p > 0. then sum m else full_like (sum m) Float.infinity
  else
    let k : Nx_kernel.monoid = if p > 0. then Max else Min in
    let c = reduction ~by k ~axes ~keepdims:true m in
    let q = div m c in
    let r =
      if p = 2. then sqrt (sum (square q))
      else pow_s (sum (pow_s q p)) (1. /. p)
    in
    where (logical_or (equal_s c 0.) (isinf c)) c (mul c r)

(* The norm [p] of the vectors of the magnitudes [m] along [axes], kept as
   axes of one. *)
let vector ~by p axes m =
  if Float.is_nan p then invalid_argf "%s: an order of nan" by;
  let sum x = reduction ~by Sum ~axes ~keepdims:true x in
  if p = Float.infinity then reduction ~by Max ~axes ~keepdims:true m
  else if p = Float.neg_infinity then reduction ~by Min ~axes ~keepdims:true m
  else if p = 0. then
    let nonzero = cast_by ~by (dtype m) (not_equal m (zeros_like m)) in
    sum (where (isnan m) m nonzero)
  else if p = 1. then sum m
  else power_norm ~by p axes m

let vector_norm ?(ord = 2.) ?axes:l ?(keepdims = false) x =
  let by = "Nx.vector_norm" in
  renamed ~by (fun () ->
      routine_dtype ~by x;
      let along =
        match l with
        | None -> List.init (ndim x) Fun.id
        | Some l -> axes ~by x l
      in
      let y = magnitudes ~by x (fun m -> vector ~by ord along m) in
      if keepdims then y
      else
        let along = Iarray.of_list (List.sort compare along) in
        reshaped ~by (Prim.reduced (shape x) along) y)

let matrix_norm ?(ord = `Fro) ?(keepdims = false) x =
  let by = "Nx.matrix_norm" in
  renamed ~by (fun () ->
      matrices ~by x;
      let r = ndim x - 2 and c = ndim x - 1 in
      (* The greatest or least sum of a line's magnitudes, [summed] along
         it. *)
      let lines k summed chosen m =
        reduction ~by k ~axes:[ chosen ] ~keepdims:true
          (reduction ~by Sum ~axes:[ summed ] ~keepdims:true m)
      in
      (* The singular values, a [[…; k; 1]] column, reduced by [k] along it. *)
      let singular k =
        let _, s, _ = svd ~vectors:false x in
        let s = unsqueeze ~axes:[ -1 ] s in
        match k with
        | `Sum -> reduction ~by Sum ~axes:[ r ] ~keepdims:true s
        | `Max -> reduction ~by Max ~axes:[ r ] ~keepdims:true s
        | `Min -> reduction ~by Min ~axes:[ r ] ~keepdims:true s
      in
      let y =
        match ord with
        | `Fro -> magnitudes ~by x (fun m -> power_norm ~by 2. [ r; c ] m)
        | `One -> magnitudes ~by x (fun m -> lines Max r c m)
        | `NegOne -> magnitudes ~by x (fun m -> lines Min r c m)
        | `Inf -> magnitudes ~by x (fun m -> lines Max c r m)
        | `NegInf -> magnitudes ~by x (fun m -> lines Min c r m)
        | `Nuc -> singular `Sum
        | `Two -> singular `Max
        | `NegTwo -> singular `Min
      in
      if keepdims then y else reshaped ~by (batch x) y)

(* Order *)

(* [x] along [axis] in the sort order, reversed where [descending], its first
   [k] kept, and their positions. *)
let sorted ~by ?(descending = false) ?k a x =
  Eval.eval ~by (Value.Sort { axis = a; descending; k; x })

let sort ?descending ?axis:(a = -1) x =
  let by = "Nx.sort" in
  sorted ~by ?descending (axis ~by x a) x

let argsort ?descending ?axis:(a = -1) x =
  let by = "Nx.argsort" in
  snd (sorted ~by ?descending (axis ~by x a) x)

let top_k ~k ?axis:(a = -1) x =
  let by = "Nx.top_k" in
  let a = axis ~by x a in
  let n = Prim.dim x a in
  if k < 0 || k > n then
    invalid_argf "%s: k = %d for an axis of %d elements" by k n;
  sorted ~by ~descending:true ~k a x


let signed_of_width = function
  | 8 -> D.Any D.Int8
  | 16 -> D.Any D.Int16
  | 32 -> D.Any D.Int32
  | _ -> D.Any D.Int64

(* The float a key of [w] bits reads [dt]'s values in: [dt] itself at its own
   width, bfloat16 at 16 bits for bfloat16, then float16, float32 and float64.
   float4_e2m1fn's values read as float8_e4m3fn's at 8 bits. *)
let float_of_width (type s) (dt : (float, s) D.t) w =
  match (w, dt) with
  | 16, D.Bfloat16 -> Prim.Wide D.Bfloat16
  | 8, D.Float4_e2m1fn -> Prim.Wide D.Float8_e4m3fn
  | w, dt when D.bits dt = w -> Prim.Wide dt
  | 16, _ -> Prim.Wide D.Float16
  | 32, _ -> Prim.Wide D.Float32
  | _ -> Prim.Wide D.Float64

(* The node of [x]'s key in the unsigned [kd] of [w] bits: an integer's value
   with the sign bit flipped, a float's bits flipped where negative and only its
   sign bit elsewhere, every NaN all ones. With [merge], [-0.]'s key is [+0.]'s,
   as [less] compares numbers. *)
let key_node (type v s k r) ~merge (dt : (v, s) D.t) (kd : (k, r) D.t) w
    (x : (v, s) E.t) : (k, r) E.t =
  let (D.Any s) = signed_of_width w in
  let int v = wrapped s v in
  let sign_bit = Int64.shift_left (-1L) (w - 1) in
  match D.kind dt with
  | D.Unsigned | D.Boolean -> E.cast kd x
  | D.Signed -> E.bitcast kd E.(cast s x lxor int sign_bit)
  | D.Complex -> assert false
  | D.Float ->
      let (Prim.Wide f) = float_of_width dt w in
      let v = at f x in
      let v = if merge then E.(v + const f 0.) else v in
      let i = E.bitcast s v in
      let flip = E.(where (i < int 0L) (int (-1L)) (int sign_bit)) in
      E.bitcast kd E.(where (v <> v) (int (-1L)) (i lxor flip))

let no_complex ~by x =
  if D.is D.Complex (dtype x) then
    invalid_argf "%s: %a: complex numbers have no order key" by pp_value x

let order_key (type k r v s d) (kd : (k, r) D.t) (x : (v, s, d) t) : (k, r, d) t
    =
  let by = "Nx.order_key" in
  let dt = dtype x in
  let w =
    match kd with
    | D.Uint8 -> 8
    | D.Uint16 -> 16
    | D.Uint32 -> 32
    | D.Uint64 -> 64
    | _ ->
        invalid_argf "%s: %s is not uint8, uint16, uint32 or uint64" by
          (D.name kd)
  in
  no_complex ~by x;
  if D.bits dt > w then
    invalid_argf "%s: %s keys are narrower than %s" by (D.name kd) (D.name dt);
  elementwise ~by kd [ Any x ] (key_node ~merge:false dt kd w (operand 0 x))

(* [x]'s uint64 keys, [-0.]'s merged into [+0.]'s where [merge]. *)
let keys64 ~by ~merge x =
  let dt = dtype x in
  elementwise ~by D.Uint64 [ Any x ] (key_node ~merge dt D.Uint64 64 (operand 0 x))

(* Refuses what is not 1-d or 2-d keys. *)
let keyed ~by x =
  let r = ndim x in
  if r <> 1 && r <> 2 then
    invalid_argf "%s: keys %a are neither 1-d nor 2-d" by pp_value x;
  no_complex ~by x

(* Column [j] of the matrix [x], a view. *)
let column ~by j x =
  let n = Prim.dim x 0 in
  let ranges =
    [| whole n; { Nx_array.Move.start = j; count = 1; step = 1 } |]
  in
  move ~by (Reshape [| n |]) (sliced ~by ranges x)

let positions_of ~by x = snd (sorted ~by 0 x)

(* The stable permutation sorting the rows of [keys], one sort per column from
   the last: each keeps the order of the columns after it among its ties. *)
let rows_order ~by keys =
  if ndim keys = 1 then positions_of ~by keys
  else
    let n = Prim.dim keys 0 and w = Prim.dim keys 1 in
    let rec from j p =
      if j < 0 then p
      else
        let q = positions_of ~by (take_axis ~by 0 p (column ~by j keys)) in
        from (j - 1) (take_axis ~by 0 q p)
    in
    if w = 0 then arange D.Int64 0 n 1
    else from (w - 2) (positions_of ~by (column ~by (w - 1) keys))

let lexsort keys =
  let by = "Nx.lexsort" in
  keyed ~by keys;
  rows_order ~by keys

(* Bits of [m]: the rounds of a search among [m] keys, [ceil (log2 (m + 1))]. *)
let rounds m =
  let rec go r = if m lsr r = 0 then r else go (r + 1) in
  go 0

(* How a probe [k] compares with a sought key [x], as uint8: 0 below, 1 equal, 2
   above. *)
let code k x =
  let c v = E.const D.Uint8 v in
  E.(where (k < x) (c 0) (where (k = x) (c 1) (c 2)))

(* The number of keys of [s] before each key of [v] ([`Left]), or before or
   equal to it ([`Right]), by binary lifting: the count [p] takes each step
   [2^(r-1)], …, [1] whose last key [s.(p + step - 1)] still comes before. The
   state is that probe's position [q]; a probe at or past [m] does not advance.
   Rows compare as their first column that differs. *)
let search ~by side s v =
  let ks = keys64 ~by ~merge:true s and kv = keys64 ~by ~merge:true v in
  let m = Prim.dim s 0 in
  let rows = ndim s = 2 in
  let lead = if rows then [| Prim.dim v 0 |] else shape v in
  let before = match side with `Left -> 0 | `Right -> 1 in
  let r = rounds m in
  let step i = 1 lsl (r - 1 - i) in
  let compared probe =
    let c =
      elementwise ~by D.Uint8 [ Any probe; Any kv ]
        (code (operand 0 probe) (operand 1 kv))
    in
    if not rows then c
    else if Prim.dim s 1 = 0 then fill ~by D.Uint8 lead 1
    else
      let differs =
        elementwise ~by D.Bool [ Any c ] E.(operand 0 c <> const D.Uint8 1)
      in
      let j = arg ~by Max ~axis:1 ~keepdims:true differs in
      reshaped ~by lead (take_along_axis ~axis:1 j c)
  in
  let rec round i q =
    let c = compared (take_axis ~by 0 q ks) in
    let last = i = r - 1 in
    let q =
      let e = operand 0 q in
      let up = E.(e < i64 m && operand 1 c <= const D.Uint8 before) in
      let back = i64 (step i - 1) in
      let p = E.(where up (e + i64 1) (e - back)) in
      elementwise ~by D.Int64 [ Any q; Any c ]
        (if last then p else E.add p (i64 (step (i + 1) - 1)))
    in
    if last then q else round (i + 1) q
  in
  if r = 0 then fill ~by D.Int64 lead 0L
  else round 0 (fill ~by D.Int64 lead (Int64.of_int (step 0 - 1)))

let searchsorted ~side s v =
  let by = "Nx.searchsorted" in
  keyed ~by s;
  if ndim s = 2 && (ndim v <> 2 || Prim.dim v 1 <> Prim.dim s 1) then
    invalid_argf "%s: rows %a sought among rows %a" by pp_value v pp_value s;
  search ~by side s v

(* Counted lengths *)

(* The 0-d [n] read on the host, for the function [by]. *)
let length ~by n = Int64.to_int (Nx_array.get (read ~by n) [||])

let sized ~by = function
  | Some s when s < 0 -> invalid_argf "%s: size %d" by s
  | _ -> ()

(* The 1-d [x] cut or padded with [v] to [l] entries. *)
let fit ~by l v x =
  let n = Prim.dim x 0 in
  if l <= n then move ~by (Slice [| { start = 0; count = l; step = 1 } |]) x
  else assemble ~by D.Int64 [| l |] v [ ([| whole n |], x) ]

type 'd groups = {
  ids : 'd int64_t;
  first : 'd int64_t;
  counts : 'd int64_t;
  count : 'd int64_t;
}

(* A scatter into [into] along its one axis. *)
let scattered ~by ?(unique = false) combine idx updates into =
  Eval.eval ~by
    (Value.Scatter { combine; unique; axis = 0; idx; updates; into })

(* One stable sort of the keys; then, in sorted order, a flag where a key's
   order key differs from its predecessor's, the start of its run by a
   cumulative maximum, and the run's first position in the input, which the
   stable sort put at the run's start. Flagged in input order, a cumulative sum
   of those first positions ranks each group by first appearance. Every array
   has [n] entries, padded by construction, until the count cuts them. *)
let unique ?size keys =
  let by = "Nx.unique" in
  keyed ~by keys;
  sized ~by size;
  let n = Prim.dim keys 0 in
  let p = rows_order ~by keys in
  let k = take_axis ~by 0 p (keys64 ~by ~merge:false keys) in
  let i = arange D.Int64 0 n 1 in
  let before =
    take_axis ~by 0
      (elementwise ~by D.Int64 [ Any i ] E.(operand 0 i - i64 1))
      k
  in
  let differs = comparison ~by (Compare Not_equal) k before in
  let differs =
    if ndim keys = 1 then differs else truth ~by Max ~axes:[ 1 ] differs
  in
  let fresh =
    elementwise ~by D.Int64 [ Any i; Any differs ]
      E.(where (operand 0 i = i64 0 || operand 1 differs) (i64 1) (i64 0))
  in
  let start = scan_along ~by Max 0 (binary ~by (Binary Mul) fresh i) in
  let from = take_axis ~by 0 start p in
  let zeros = fill ~by D.Int64 [| n |] 0L in
  let flags = scattered ~by ~unique:true Set p fresh zeros in
  let rank = scan_along ~by Sum 0 flags in
  let of_sorted =
    let r = take_axis ~by 0 from rank in
    elementwise ~by D.Int64 [ Any r ] E.(operand 0 r - i64 1)
  in
  let ids = scattered ~by ~unique:true Set p of_sorted zeros in
  let target =
    elementwise ~by D.Int64 [ Any flags; Any rank ]
      E.(where (operand 0 flags = i64 1) (operand 1 rank - i64 1) (i64 (-1)))
  in
  let first =
    scattered ~by Set target i (fill ~by D.Int64 [| n |] (Int64.of_int n))
  in
  let counts = scattered ~by Add ids (fill ~by D.Int64 [| n |] 1L) zeros in
  let count = reduce ~by Sum D.Int64 [| 0 |] fresh in
  let l = match size with Some l -> l | None -> length ~by count in
  {
    ids;
    first = fit ~by l (Int64.of_int n) first;
    counts = fit ~by l 0L counts;
    count;
  }

(* Each index's run starts at the sum of the counts before it; a scatter of the
   indices at their starts, a cumulative maximum along the result, and [n] past
   the total. A negative count counts zero. *)
let positions ?size c =
  let by = "Nx.positions" in
  if ndim c <> 1 then invalid_argf "%s: %a is not 1-d" by pp_value c;
  integer_or_boolean ~by (dtype c);
  sized ~by size;
  let n = Prim.dim c 0 in
  let counts =
    elementwise ~by D.Int64 [ Any c ] E.(maximum (cast D.Int64 (operand 0 c)) (i64 0))
  in
  let total = reduce ~by Sum D.Int64 [| 0 |] counts in
  let l = match size with Some l -> l | None -> length ~by total in
  let i = arange D.Int64 0 n 1 in
  (* An index of no count goes to its own position below zero, which writes
     nothing, so the targets are distinct. *)
  let starts =
    let ends = scan_along ~by Sum 0 counts in
    let count = operand 1 counts in
    let below = E.(i64 (-1) - operand 2 i) in
    elementwise ~by D.Int64 [ Any ends; Any counts; Any i ]
      E.(where (i64 0 < count) (operand 0 ends - count) below)
  in
  let runs =
    scattered ~by ~unique:true Set starts i (fill ~by D.Int64 [| l |] (-1L))
  in
  let starts = scan_along ~by Max 0 runs and j = arange D.Int64 0 l 1 in
  elementwise ~by D.Int64 [ Any starts; Any j; Any total ]
    E.(where (operand 1 j < operand 2 total) (operand 0 starts) (i64 n))

(* Quantiles and histograms *)

(* The 1-d value of [vs], a value of every set. *)
let vector ~by dt vs =
  let n = Array.length vs in
  let piece j v =
    ( ([| { Nx_array.Move.start = j; count = 1; step = 1 } |] : _ iarray),
      move ~by (Reshape [| 1 |]) (fill ~by dt [||] v) )
  in
  assemble ~by dt [| n |] (D.zero dt) (Array.to_list (Array.mapi piece vs))

(* One sort; per probability the order statistics around [h = q (n - 1)] and one
   map interpolating between them in the wide format. *)
let quantile (type s d) ?axis:along qs (x : (float, s, d) t) : (float, s, d) t =
  let by = "Nx.quantile" in
  Array.iter
    (fun q ->
      if not (q >= 0. && q <= 1.) then
        invalid_argf "%s: probability %h is outside [0, 1]" by q)
    qs;
  let x, a =
    match along with
    | None -> (move ~by (Reshape [| numel x |]) x, 0)
    | Some a -> (x, axis ~by x a)
  in
  let n = Prim.dim x a in
  if n = 0 then
    invalid_argf "%s: no element along axis %d of %a" by a pp_value x;
  let s = fst (sorted ~by a x) in
  let h = Array.map (fun q -> q *. Float.of_int (n - 1)) qs in
  let lo = Array.map (fun h -> Float.to_int (Float.floor h)) h in
  let hi = Array.map (fun l -> Stdlib.min (l + 1) (n - 1)) lo in
  let index l = vector ~by D.Int64 (Array.map Int64.of_int l) in
  let below = take_axis ~by a (index lo) s
  and above = take_axis ~by a (index hi) s in
  let dt = dtype x in
  let (Prim.Wide w) = Prim.wide dt in
  let along =
    Array.mapi (fun i _ -> if i = a then Array.length qs else 1) (shape x)
  in
  let frac =
    reshaped ~by along
      (vector ~by w (Array.map (fun h -> h -. Float.floor h) h))
  in
  let y =
    let lo = at w (operand 0 below) and hi = at w (operand 1 above) in
    let f = operand 2 frac in
    let r = E.(lo + (f * (hi - lo))) in
    let r = E.(where (f = const w 0. || lo = hi) lo r) in
    elementwise ~by dt [ Any below; Any above; Any frac ] (at dt r)
  in
  let r = ndim y in
  permute ~by
    (Array.init r (fun i -> if i = 0 then a else if i <= a then i - 1 else i))
    y

(* Each coordinate's bin, by a search among its edges, the last edge folded into
   the last bin and any other point outside sent to [-1]; the cells' flat
   positions; one adding scatter. *)
let histogram (type s d) ?weights
    (dims : ((float, s, d) t * (float, s, d) t) list) : (float, s, d) t =
  let by = "Nx.histogram" in
  let points =
    match dims with
    | [] -> invalid_argf "%s: no dimension" by
    | (_, x) :: _ -> shape x
  in
  List.iter
    (fun (e, x) ->
      if ndim e <> 1 || Prim.dim e 0 < 2 then
        invalid_argf "%s: edges %a are not 1-d of two or more" by pp_value e;
      if shape x <> points then
        invalid_argf "%s: coordinates %a and %a differ in shape" by pp_value x
          pp_shape points)
    dims;
  Option.iter
    (fun w ->
      if shape w <> points then
        invalid_argf "%s: weights %a for coordinates %a" by pp_value w pp_shape
          points)
    weights;
  let bin (e, x) =
    let m = Prim.dim e 0 in
    let last =
      move ~by (Reshape [||])
        (move ~by (Slice [| { start = m - 1; count = 1; step = 1 } |]) e)
    in
    let found = search ~by `Right e x in
    let last_bin = i64 (m - 2) in
    let j =
      E.(where (operand 1 x = operand 2 last) last_bin (operand 0 found - i64 1))
    in
    elementwise ~by D.Int64 [ Any found; Any x; Any last ]
      E.(where (i64 0 <= j && j <= last_bin) j (i64 (-1)))
  in
  let cell acc (e, x) =
    let nb = Prim.dim e 0 - 1 in
    let b = bin (e, x) in
    let a = operand 0 acc and i = operand 1 b in
    let out = E.(a < i64 0 || i < i64 0) in
    elementwise ~by D.Int64 [ Any acc; Any b ]
      E.(where out (i64 (-1)) ((a * i64 nb) + i))
  in
  let flat =
    match dims with
    | d :: rest -> List.fold_left cell (bin d) rest
    | [] -> assert false
  in
  let bins = Array.of_list (List.map (fun (e, _) -> Prim.dim e 0 - 1) dims) in
  let cells = Array.fold_left ( * ) 1 bins in
  let flat = move ~by (Reshape [| numel flat |]) flat in
  let dt = dtype (snd (List.hd dims)) in
  let h =
    match weights with
    | None ->
        cast_by ~by dt
          (scattered ~by Add flat
             (fill ~by D.Int64 [| numel flat |] 1L)
             (fill ~by D.Int64 [| cells |] 0L))
    | Some w ->
        scattered ~by Add flat
          (move ~by (Reshape [| numel w |]) w)
          (fill ~by dt [| cells |] 0.)
  in
  reshaped ~by bins h

(* Whole values and maps on the host *)

let array_equal a b =
  let by = "Nx.array_equal" in
  if not (same_shape a b) then scalar_beside ~by a D.Bool false
  else truth ~by Min (comparison ~by (Compare Equal) a b)

let map_item f x =
  let by = "Nx.map_item" in
  let vs = Array.map f (Nx_array.to_array (read ~by x)) in
  renamed ~by (fun () -> on_host (Nx_array.of_array (dtype x) (shape x) vs))

(* Windows and segments *)

let one_hot ~num_classes x =
  let by = "Nx.one_hot" in
  integer ~by (dtype x);
  if num_classes < 0 then invalid_argf "%s: %d classes" by num_classes;
  let s = shape x in
  let r = Array.length s in
  let column = reshaped ~by (Array.append s [| 1 |]) x in
  let x = broadcast ~by (Array.append s [| num_classes |]) column in
  elementwise ~by D.Uint8 [ Any x ]
    (indicator D.Uint8 E.(cast D.Int64 (operand 0 x) = coord r))

(* The format a window's sum adds in: a float's compute format
   ({!Prim.wide}), the dtype itself otherwise. *)
let summed (type v s) (dt : (v, s) D.t) =
  match D.kind dt with
  | D.Float ->
      let (Prim.Wide w) = Prim.wide dt in
      D.Any w
  | D.Complex | D.Signed | D.Unsigned | D.Boolean -> D.Any dt

let no_sum (type v s) ~by (dt : (v, s) D.t) =
  if D.is D.Boolean dt then invalid_argf "%s: %s has no sum" by (D.name dt)

(* The extent of a window of [k] taps [dilation] apart: none for no tap. *)
let dilated k dilation = if k = 0 then 0 else (dilation * (k - 1)) + 1

(* The windows along an axis of [n] elements padded by [lo] and [hi]: none
   where the dilated window is longer. *)
let windows_along n (lo, hi) k step dilation =
  let span = dilated k dilation and e = n + lo + hi in
  if e < span then 0 else ((e - span) / step) + 1

(* Refuses a window geometry other than [K > 0] kernel axes, each with a
   positive size, stride and dilation and a padding of no negative width. *)
let geometry ~by ~kernel_size ~stride ~dilation ~padding =
  let k = Array.length kernel_size in
  if k = 0 then invalid_argf "%s: kernel_size has no axis" by;
  let entries name n =
    if n <> k then
      invalid_argf "%s: %d %s for %d kernel axes" by n name k
  in
  entries "strides" (Array.length stride);
  entries "dilations" (Array.length dilation);
  entries "paddings" (Array.length padding);
  let positive name =
    Array.iter (fun n -> if n < 1 then invalid_argf "%s: %s %d" by name n)
  in
  positive "kernel size" kernel_size;
  positive "stride" stride;
  positive "dilation" dilation;
  Array.iter
    (fun (lo, hi) ->
      if lo < 0 || hi < 0 then invalid_argf "%s: padding (%d, %d)" by lo hi)
    padding

let spatial ~by k x =
  if ndim x < k then invalid_argf "%s: %d kernel axes for %a" by k pp_value x

(* The windows of [x]'s last [K] axes as a view of shape
   [[lead…; counts…; kernel_size…]], after padding [x] with zeros, a copy
   where a padding is not zero. *)
let windowed ~by ~kernel_size ~stride ~dilation ~padding x =
  let k = Array.length kernel_size and s = shape x in
  let lead = Array.length s - k in
  let counts =
    Array.init k (fun i ->
        windows_along s.(lead + i) padding.(i) kernel_size.(i) stride.(i)
          dilation.(i))
  in
  let widths =
    Array.mapi (fun a _ -> if a < lead then (0, 0) else padding.(a - lead)) s
  in
  let padded =
    if Array.for_all (( = ) (0, 0)) widths then x
    else
      let s' = Array.mapi (fun a d -> fst widths.(a) + d + snd widths.(a)) s in
      let rs =
        Iarray.init (Array.length s) (fun a : Nx_array.Move.range ->
            { start = fst widths.(a); count = s.(a); step = 1 })
      in
      assemble ~by (dtype x) s' (D.zero (dtype x)) [ (rs, x) ]
  in
  let ws =
    Iarray.init k (fun i : Nx_array.Move.window ->
        {
          axis = lead + i;
          size = kernel_size.(i);
          step = stride.(i);
          dilation = dilation.(i);
        })
  in
  (move ~by (Window ws) padded, counts)

let product = Array.fold_left ( * ) 1

let extract_patches ~kernel_size ~stride ~dilation ~padding x =
  let by = "Nx.extract_patches" in
  geometry ~by ~kernel_size ~stride ~dilation ~padding;
  let k = Array.length kernel_size in
  spatial ~by k x;
  let y, counts = windowed ~by ~kernel_size ~stride ~dilation ~padding x in
  let lead = ndim x - k in
  let taps_first =
    Array.init (lead + (2 * k)) (fun a ->
        if a < lead then a else if a < lead + k then a + k else a - k)
  in
  let s' =
    Array.append (Array.sub (shape x) 0 lead)
      [| product kernel_size; product counts |]
  in
  reshaped ~by s' (permute ~by taps_first y)

let combine_patches ~output_size ~kernel_size ~stride ~dilation ~padding p =
  let by = "Nx.combine_patches" in
  geometry ~by ~kernel_size ~stride ~dilation ~padding;
  let k = Array.length kernel_size in
  if Array.length output_size <> k then
    invalid_argf "%s: %d output extents for %d kernel axes" by
      (Array.length output_size) k;
  Array.iter
    (fun n -> if n < 0 then invalid_argf "%s: output extent %d" by n)
    output_size;
  let dt = dtype p in
  no_sum ~by dt;
  let r = ndim p in
  if r < 2 then invalid_argf "%s: %a holds no patches" by pp_value p;
  let counts =
    Array.init k (fun i ->
        windows_along output_size.(i) padding.(i) kernel_size.(i) stride.(i)
          dilation.(i))
  in
  let s = shape p in
  if s.(r - 2) <> product kernel_size || s.(r - 1) <> product counts then
    invalid_argf "%s: %a, where the geometry gives patches [%d; %d]" by
      pp_value p (product kernel_size) (product counts);
  let lead = Array.sub s 0 (r - 2) in
  let nl = r - 2 in
  (* [p] as the Padded load's shape, [[lead…; counts…; kernel_size…]]. *)
  let y =
    move ~by
      (Reshape (Iarray.of_array (Array.concat [ lead; kernel_size; counts ])))
      p
  in
  let y =
    permute ~by
      (Array.init (nl + (2 * k)) (fun a ->
           if a < nl then a else if a < nl + k then a + k else a - k))
      y
  in
  let pad : Nx_kernel.pad =
    {
      lo =
        Iarray.init (nl + k) (fun a ->
            if a < nl then 0 else fst padding.(a - nl));
      hi =
        Iarray.init (nl + k) (fun a ->
            if a < nl then 0 else snd padding.(a - nl));
      interior = Iarray.init (nl + k) (fun _ -> 0);
      windows =
        Iarray.init k (fun i : Nx_array.Move.window ->
            {
              axis = nl + i;
              size = kernel_size.(i);
              step = stride.(i);
              dilation = dilation.(i);
            });
    }
  in
  let shape = Iarray.of_array (Array.append lead output_size) in
  let (D.Any w) = summed dt in
  let y = cast_by ~by w y in
  cast_by ~by dt (Eval.eval ~by (Value.Fold { shape; pad; x = y }))

(* +0 of [x]'s dtype and of [shape], where [x] lies, or [w] for a constant
   [x]: a sum over no term. *)
let no_term ~by x w shape =
  let zeros = fill ~by (dtype x) shape (D.zero (dtype x)) in
  match Prim.at x with
  | Some _ -> beside ~by x zeros
  | None -> beside ~by w zeros

(* The windows of each output channel over [x], [[n; s…; c]], for [w] laid out
   as [[g; oc; ic; k…]] ([g] groups of [oc] output and [ic] input channels):
   one reduction over the loop [[n; t…; g; oc; ic; k…]] that reads [x], its
   channels split into groups and broadcast over [oc], through a padded load
   whose windows append the taps, and sums the taps and [ic]. [interior]
   zeros go between [x]'s elements. A padding that cuts more than an axis
   holds leaves it empty, with no window. A kernel axis of no tap sums no
   term. *)
let windows_of ~by ~stride ~lo ~hi ~interior ~dilation x w =
  let dt = dtype x in
  let r = ndim x and kw = shape w in
  let k = r - 2 in
  let g = kw.(0) and oc = kw.(1) and ic = kw.(2) in
  let s = shape x in
  let n = s.(0) and spatial = Array.sub s 1 k in
  let taps = Array.sub kw 3 k in
  let extent i =
    let d = spatial.(i) in
    if d = 0 then 0 else d + (interior.(i) * (d - 1))
  in
  let counts =
    Array.init k (fun i ->
        windows_along (extent i) (lo.(i), hi.(i)) taps.(i) stride.(i)
          dilation.(i))
  in
  let out = Array.concat [ [| n |]; counts; [| g * oc |] ] in
  if Array.mem 0 taps then no_term ~by x w out
  else
    let hi =
      Array.init k (fun i -> Stdlib.max hi.(i) (-(extent i + lo.(i))))
    in
    let grouped = Array.concat [ [| n |]; spatial; [| g; 1; ic |] ] in
    let x' = reshaped ~by grouped x in
    let wide = Array.concat [ [| n |]; spatial; [| g; oc; ic |] ] in
    let x' = broadcast ~by wide x' in
    (* [a.(i)] along spatial axis [i], zero along the others of [x']. *)
    let spatially a =
      Iarray.init (k + 4) (fun i -> if i >= 1 && i <= k then a.(i - 1) else 0)
    in
    let windows =
      Iarray.init k (fun i : Nx_array.Move.window ->
          {
            axis = 1 + i;
            size = taps.(i);
            step = stride.(i);
            dilation = dilation.(i);
          })
    in
    let pad : Nx_kernel.pad =
      {
        lo = spatially lo;
        hi = spatially hi;
        interior = spatially interior;
        windows;
      }
    in
    let loop = Array.concat [ [| n |]; counts; [| g; oc; ic |]; taps ] in
    let (D.Any acc) = summed dt in
    let term = E.(at acc (E.input 0 dt) * at acc (E.input 1 dt)) in
    let ins = Iarray.of_list [ D.Any dt; D.Any dt ] in
    let prog = P.of_exprs ~ins [| E.any term |] in
    let y, () =
      Eval.eval ~by
        (Value.Reduce
           {
             layout = renamed ~by (fun () -> Nx_array.Layout.contiguous loop);
             axes = Iarray.init (k + 1) (fun i -> k + 3 + i);
             prog;
             reductions = Value.[ Monoid (Sum, 0, dt) ];
             loads =
               [|
                 Padded { x = x'; fill = D.zero dt; pad };
                 Plain (broadcast ~by loop w);
               |];
           })
    in
    reshaped ~by out y

(* For each spatial axis, the run of values that [padding] keeps: its first
   value and its count, as the interface states them. *)
let run_of padding n k =
  let m = Stdlib.min n k and longest = Stdlib.max n k in
  match padding with
  | `Full -> (0, n + k - 1)
  | `Valid -> (m - 1, longest - m + 1)
  | `Same -> ((if k <= n then (m - 1) / 2 else m / 2), longest)

(* [x]'s batch axes merged into one, with one channel, convolved by the
   kernel as one output and one input channel: value [i] of a kept run from
   [start] reads [x] from [start + i - (k - 1)], which a padding of
   [k - 1 - start] zeros before [x] puts at window [i]. *)
let correlate ?(padding = `Valid) x kernel =
  let by = "Nx.correlate" in
  let dt = dtype x and kr = ndim kernel and r = ndim x in
  if r < kr then
    invalid_argf "%s: %a has fewer axes than the kernel %a" by pp_value x
      pp_value kernel;
  if D.is D.Boolean dt then
    invalid_argf "%s: %s has no products" by (D.name dt);
  let ks = shape kernel and s = shape x in
  if Array.exists (( = ) 0) ks then
    invalid_argf "%s: the kernel %a has an axis of no element" by pp_value
      kernel;
  let lead = Array.sub s 0 (r - kr) and spatial = Array.sub s (r - kr) kr in
  let runs = Array.init kr (fun i -> run_of padding spatial.(i) ks.(i)) in
  let lo = Array.init kr (fun i -> ks.(i) - 1 - fst runs.(i)) in
  let hi =
    Array.init kr (fun i -> snd runs.(i) + ks.(i) - 1 - spatial.(i) - lo.(i))
  in
  let batch = Array.concat [ [| product lead |]; spatial; [| 1 |] ] in
  let w = reshaped ~by (Array.append [| 1; 1; 1 |] ks) kernel in
  let y =
    windows_of ~by ~stride:(Array.make kr 1) ~lo ~hi
      ~interior:(Array.make kr 0) ~dilation:(Array.make kr 1)
      (reshaped ~by batch x) w
  in
  reshaped ~by (Array.append lead (Array.map snd runs)) y

(* Segments and ranges *)

(* What a segment or a range of no row holds: [+0] for [Add], zero for
   [Set], and the least or greatest value in the dtype's order for [Max] and
   [Min]. *)
let identity (type v s) ~by (c : combine) (dt : (v, s) D.t) : v =
  match (c, D.kind dt) with
  | Add, D.Boolean -> invalid_argf "%s: %s has no sum" by (D.name dt)
  | (Set | Add), _ -> D.zero dt
  | Max, D.Complex -> { Complex.re = Float.neg_infinity; im = neg_infinity }
  | Min, D.Complex -> { Complex.re = Float.infinity; im = infinity }
  | Max, _ -> D.min_value dt
  | Min, _ -> D.max_value dt

let rows_of ~by x =
  if ndim x = 0 then invalid_argf "%s: %a has no rows" by pp_value x

let reduce_segments c ~segments ids x =
  let by = "Nx.reduce_segments" in
  if segments < 0 then invalid_argf "%s: %d segments" by segments;
  rows_of ~by x;
  if ndim ids <> 1 || Prim.dim ids 0 <> Prim.dim x 0 then
    invalid_argf "%s: ids %a for the rows of %a" by pp_value ids pp_value x;
  let dt = dtype x and s = shape x in
  let start = identity ~by c dt in
  let into = Array.mapi (fun a d -> if a = 0 then segments else d) s in
  let along = Array.mapi (fun a d -> if a = 0 then d else 1) s in
  let idx = broadcast ~by s (reshaped ~by along ids) in
  Eval.eval ~by
    (Value.Scatter
       {
         combine = c;
         unique = false;
         axis = 0;
         idx;
         updates = x;
         into = fill ~by dt into start;
       })

(* [a] then [b] combined by [c], [a]'s rows before [b]'s: [Add] adds; [Max]
   and [Min] keep the first NaN, else {!maximum}'s or {!minimum}'s, so that
   an extreme is one of its rows. *)
let combined (type v s d) ~by (c : combine) (a : (v, s, d) t) (b : (v, s, d) t)
    : (v, s, d) t =
  match c with
  | Set -> b
  | Add -> binary ~by (Binary Add) a b
  | Max | Min -> (
      let k : P.binary = if c = Max then Maximum else Minimum in
      match D.kind (dtype a) with
      | D.Float | D.Complex ->
          let x = operand 0 a and y = operand 1 b in
          elementwise ~by (dtype a) [ Any a; Any b ]
            E.(where (x <> x) x (where (y <> y) y (binary k x y)))
      | D.Signed | D.Unsigned | D.Boolean -> binary ~by (Binary k) a b)

(* The greatest [k] with [2^k <= v], for [v >= 1]. *)
let top_bit v =
  let rec go k = if v lsr (k + 1) = 0 then k else go (k + 1) in
  go 0

(* The disjoint sparse table: level [k] holds, at each row, its chunk of [2^k]
   rows combined from that row to the chunk's end, its suffix, and from the
   chunk's start to that row, its prefix. A range from [l] to [r] inclusive,
   [l < r], is answered at the one level where [l] and [r] lie in the two
   halves of one chunk of the next level, the highest bit where they differ:
   [l]'s suffix, then [r]'s prefix. Its bits are so a function of [l], [r]
   and their rows alone. Level [k] comes from level [k - 1]: a first half's
   suffix goes on through the second half's total, the suffix of its first
   row, and a second half's prefix starts from the first half's total, the
   prefix of its last row. Rows are padded to whole chunks of the top level;
   no answer reads the padding. *)
let ranges (type v s d) ~by c ~l ~r (x : (v, s, d) t) (none : (v, s, d) t) =
  let n = Prim.dim x 0 and dt = dtype x in
  let rows = Array.sub (shape x) 1 (ndim x - 1) in
  let m = Prim.dim l 0 in
  let per_range t =
    reshaped ~by (Array.append [| m |] (Array.map (fun _ -> 1) rows)) t
  in
  let where c a b = ternary ~by Where (per_range c) a b in
  let li = operand 0 l and ri = operand 1 r in
  let single =
    elementwise ~by D.Bool [ Any l; Any r ] E.(li = ri)
  in
  let res = where single (take_axis ~by 0 r x) none in
  if n < 2 then res
  else
    let top = top_bit (n - 1) in
    let chunk = 1 lsl top in
    let padded = (n + chunk - 1) / chunk * chunk in
    let full = Array.append [| padded |] rows in
    let xp =
      if padded = n then x
      else assemble ~by dt full (D.zero dt) [ (all_of (shape x), x) ]
    in
    let whole_rows = all_of rows in
    let double k suf pre =
      let h = 1 lsl (k - 1) in
      let blocks = padded / (2 * h) in
      let halves = Array.append [| blocks; 2; h |] rows in
      let part half start count t =
        move ~by
          (Slice
             (Iarray.append
                [|
                  whole blocks;
                  { start = half; count = 1; step = 1 };
                  { start; count; step = 1 };
                |]
                whole_rows))
          (reshaped ~by halves t)
      in
      let half i =
        Iarray.append
          [| whole blocks; { start = i; count = 1; step = 1 }; whole h |]
          whole_rows
      in
      let joined first second =
        reshaped ~by full
          (assemble ~by dt halves (D.zero dt)
             [ (half 0, first); (half 1, second) ])
      in
      let ( ++ ) a b = combined ~by c a b in
      ( joined (part 0 0 h suf ++ part 1 0 1 suf) (part 1 0 h suf),
        joined (part 0 0 h pre) (part 0 (h - 1) 1 pre ++ part 1 0 h pre) )
    in
    let rec go k suf pre res =
      let span = i64 (1 lsl k) in
      let here =
        elementwise ~by D.Bool [ Any l; Any r ]
          E.(
            li < ri
            && binary Xor (binary Idiv li span) (binary Idiv ri span) = i64 1)
      in
      let v = combined ~by c (take_axis ~by 0 l suf) (take_axis ~by 0 r pre) in
      let res = where here v res in
      if k = top then res
      else
        let suf, pre = double (k + 1) suf pre in
        go (k + 1) suf pre res
    in
    go 0 xp xp res

let reduce_ranges c ~lo ~hi x =
  let by = "Nx.reduce_ranges" in
  rows_of ~by x;
  if ndim lo <> 1 || not (same_shape lo hi) then
    invalid_argf "%s: bounds %a and %a, not one of each per range" by pp_value
      lo pp_value hi;
  let dt = dtype x and n = Prim.dim x 0 and m = Prim.dim lo 0 in
  let rows = Array.sub (shape x) 1 (ndim x - 1) in
  let start = identity ~by c dt in
  let clip b =
    elementwise ~by D.Int64 [ Any b ]
      E.(minimum (maximum (operand 0 b) (i64 0)) (i64 n))
  in
  let l = clip lo and h = clip hi in
  let r = elementwise ~by D.Int64 [ Any h ] E.(operand 0 h - i64 1) in
  let empty dt v = fill ~by dt (Array.append [| m |] rows) v in
  match c with
  | Set ->
      let some =
        elementwise ~by D.Bool [ Any l; Any h ] E.(operand 0 l < operand 1 h)
      in
      let some =
        reshaped ~by (Array.append [| m |] (Array.map (fun _ -> 1) rows)) some
      in
      ternary ~by Where some (take_axis ~by 0 r x) (empty dt start)
  | Max | Min ->
      (* A NaN extreme is a row's bits, which the selects keep; quieted as
         {!max} returns it, flat so that a complex value's parts have an
         axis to take. *)
      let y = ranges ~by c ~l ~r x (empty dt start) in
      let s = shape y in
      let flat = move ~by (Reshape [| Array.fold_left ( * ) 1 s |]) y in
      reshaped ~by s (Expand.quieted Eval.eval ~by flat)
  | Add ->
      (* Narrow floats sum at float32; [+0] then makes a sum of [-0.] terms
         [+0.], as {!sum}'s is. *)
      let (D.Any w) = summed dt in
      let zero = D.zero w in
      let s = ranges ~by c ~l ~r (cast_by ~by w x) (empty w zero) in
      cast_by ~by dt (binary ~by (Binary Add) s (fill ~by w [||] zero))

(* Maps over segments

   [map_segments] sorts each group's positions by id, pads each id's run to
   whole blocks of [c] rows, calls [f] once on every block, and gathers each
   position's row of [f]'s result back. A block's pad rows repeat a row the
   block holds, so [f] meets only pairs the per-position map meets: a row of
   zeros is a pair that map never evaluates, where [f] may not be finite.

   The helpers below take a group's ids [[| r; j |]], [r] groups of [j]
   positions, and their rows [[| r; j; row... |]]. [call ~g ~c owners rows] is
   [f]'s result and its row shape, after checking its leading axes. *)

(* A block holds as many rows as its padding allows, up to 16: padding costs at
   most [c - 1] rows per segment, which stay below half the positions. Larger
   blocks read each segment's data fewer times, and 16 rows are the tile a GPU's
   tensor core multiplies at once. gpt-oss-20b's gate and up product of 512
   tokens took 2.9 ms in blocks of 16 on an RTX 5000 Ada, 4.1 ms in blocks of
   32 and 7.5 ms in blocks of 8. *)
let largest_block = 16

(* The rows of a block of [j] positions over [segments] segments, 1 when no
   block of 2 or more keeps the padding below half of them: then each position
   is its own block and nothing is sorted. With as many segments as positions
   no block of 2 or more qualifies, since [2 (c - 1) segments >= 2 j > j];
   deciding it first keeps the product below [30 j]. *)
let block ~segments j =
  let rec go c =
    if c < 2 || 2 * (c - 1) * segments < j then c else go (c / 2)
  in
  if segments >= j then 1 else go largest_block

(* The number of windows of devices that cut [t]'s first axis, which must be
   a position axis. *)
let device_windows ~by what ~positional t =
  match Prim.at t with
  | None -> 1
  | Some p ->
      Array.fold_left
        (fun r (a, n) ->
          if a <> 0 || not positional then
            invalid_argf "%s: %s is split over devices along axis %d" by what a;
          r * n)
        1
        (Grid.cuts (Devices.grid p))

(* [x] with each axis of [axes] kept of extent 1. *)
let keep_axes ~by s axes y =
  let kept a d = if List.mem a axes then 1 else d in
  reshaped ~by (Array.mapi kept s) y

(* Per group of [named] along [axes], its first in-range position as a
   position of the group, and whether it has one. [named] has a group's shape,
   [[| r; ... |]] for [r] groups of [j] positions; with no in-range position
   the first is [j], which a gather reads as zero. One maximum finds it: the
   first in-range position has the most positions after it. *)
let first_named ~by ~axes named =
  let s = shape named in
  let r = s.(0) in
  let j = numel named / r in
  let last = j - 1 in
  let after =
    reshaped ~by s (leaf ~by D.Int64 [| r; j |] E.(i64 last - coord 1))
  in
  let most =
    keep_axes ~by s axes
      (reduce ~by Max D.Int64 (Iarray.of_list axes)
         (ternary ~by Where named after (fill ~by D.Int64 [||] (-1L))))
  in
  ( elementwise ~by D.Int64 [ Any most ] E.(i64 last - operand 0 most),
    elementwise ~by D.Bool [ Any most ] E.(i64 0 <= operand 0 most) )

(* [owners], in range already: the clamp writes that range into the program,
   where the gathers by them read it. *)
let in_range ~by ~segments owners =
  let last = i64 (segments - 1) in
  elementwise ~by D.Int64 [ Any owners ]
    E.(minimum (maximum (operand 0 owners) (i64 0)) last)

let units t = Array.make (Array.length t) 1

(* Each position its own block. [gs] is a group's positions as axes,
   [[| r; ... |]], and [x] holds their rows in that shape, with an axis of
   length 1 where positions share a row. An out-of-range position keeps its row
   and takes the owner of the first in-range position that reads it. A row
   that no in-range position reads takes the group's first in-range position
   and its row, or zeros. Rows then vary only along the axes [x] varies along:
   at one token over a few experts, every block reads the token's row. *)
let direct ~by ~segments ~call ~named ~gs ids x =
  let r = Prim.dim ids 0 and j = Prim.dim ids 1 in
  let n = Array.length gs in
  let lead = Array.sub (shape x) 0 n in
  let row = Array.sub (shape x) n (ndim x - n) in
  let where c a b = ternary ~by Where c a b in
  let inner = List.init (n - 1) succ in
  let varies = List.exists (fun a -> lead.(a) > 1) inner in
  let shared = List.filter (fun a -> lead.(a) = 1 && gs.(a) > 1) inner in
  let at, any = first_named ~by ~axes:[ 1 ] named in
  let id = take_along ~by 1 at ids in
  let per_group t =
    reshaped ~by (Array.append [| r |] (Array.make (n - 1) 1)) t
  in
  let by_row t = reshaped ~by (Array.append (shape t) (units row)) t in
  let owners, read =
    if not varies then (where named ids id, per_group any)
    else if shared = [] then (where named ids id, reshaped ~by gs named)
    else
      let at_row, read =
        first_named ~by ~axes:shared (reshaped ~by gs named)
      in
      let flat t = move ~by (Reshape [| r; j |]) (broadcast ~by gs t) in
      let owner = take_along ~by 1 (flat at_row) ids in
      (where named ids (where (flat read) owner id), read)
  in
  let fallback =
    if not varies then scalar_beside ~by x (dtype x) (D.zero (dtype x))
    else
      let rows =
        move ~by
          (Reshape (Iarray.of_array (Array.append [| r; j |] row)))
          (broadcast ~by (Array.append gs row) x)
      in
      move ~by
        (Reshape (Iarray.of_array (Array.append (shape (per_group at)) row)))
        (take_along ~by 1
           (broadcast ~by (Array.append [| r; 1 |] row) (by_row at))
           rows)
  in
  let rows = where (by_row read) x fallback in
  let y, out =
    call ~g:(r * j) ~c:1
      (move ~by (Reshape [| r * j |]) (in_range ~by ~segments owners))
      (move ~by
         (Reshape (Iarray.of_array (Array.append [| r * j; 1 |] row)))
         (broadcast ~by (Array.append gs row) rows))
  in
  (reshaped ~by (Array.append [| r; j |] out) y, out)

(* Each group's positions sorted by id into blocks of [c] rows. *)
let grouped ~by ~segments ~c ~call ~named ids x =
  let r = Prim.dim ids 0 and j = Prim.dim ids 1 in
  let row = Array.sub (shape x) 2 (ndim x - 2) in
  let where c a b = ternary ~by Where c a b in
  let int64 v = fill ~by D.Int64 [||] (Int64.of_int v) in
  let along t p = take_along ~by 1 p t in
  let arith2 f a b =
    elementwise ~by D.Int64 [ Any a; Any b ] (f (operand 0 a) (operand 1 b))
  in
  let iota n step = leaf ~by D.Int64 [| n |] E.(coord 0 * i64 step) in
  let at, any = first_named ~by ~axes:[ 1 ] named in
  let id = along ids at in
  (* Out-of-range positions sort last, as segment [segments], and take no
     slot. *)
  let key = where named ids (int64 segments) in
  let order = snd (sorted ~by 1 key) in
  (* Each segment's run in sorted order, [[| r; segments |]]: its length, first
     position, and slots padded to whole blocks. *)
  let count =
    Eval.eval ~by
      (Value.Scatter
         {
           combine = Add;
           unique = false;
           axis = 1;
           idx = key;
           updates = fill ~by D.Int64 [| r; j |] 1L;
           into = fill ~by D.Int64 [| r; segments |] 0L;
         })
  in
  let sub = arith2 E.( - ) in
  let first = sub (scan_along ~by Sum 1 count) count in
  let padded =
    let carry = i64 (c - 1) in
    elementwise ~by D.Int64 [ Any count ]
      E.(binary Idiv (operand 0 count + carry) (i64 c) * i64 c)
  in
  let ends = scan_along ~by Sum 1 padded in
  let starts = sub ends padded in
  (* Blocks: a bound of the padded runs, whatever the ids. *)
  let blocks = (j + (Stdlib.min segments j * (c - 1)) + c - 1) / c in
  let slots = blocks * c in
  (* A block's owner is the number of runs that end at or before its first slot;
     a block past the last run takes the fallback id. *)
  let ended =
    let s3 = [| r; blocks; segments |] in
    let e = broadcast ~by s3 (move ~by (Reshape [| r; 1; segments |]) ends) in
    let b =
      broadcast ~by s3 (move ~by (Reshape [| 1; blocks; 1 |]) (iota blocks c))
    in
    fst
      (reduce_by ~by s3 [| 2 |] [ Any e; Any b ]
         Value.[ Monoid (Sum, 0, D.Int64) ]
         E.(where (operand 0 e <= operand 1 b) (i64 1) (i64 0)))
  in
  let owners =
    in_range ~by ~segments
      (where
         (elementwise ~by D.Bool [ Any ended ]
            E.(operand 0 ended < i64 segments))
         ended id)
  in
  let slot_owner =
    move ~by (Reshape [| r; slots |])
      (broadcast ~by [| r; blocks; c |]
         (move ~by (Reshape [| r; blocks; 1 |]) owners))
  in
  (* A slot past its run repeats the run's last row, which its block holds. *)
  let offset =
    sub
      (move ~by (Reshape [| 1; slots |]) (iota slots 1))
      (along starts slot_owner)
  in
  let held =
    arith2
      (fun o n -> E.(minimum o (n - i64 1)))
      offset (along count slot_owner)
  in
  let position = along order (arith2 E.( + ) (along first slot_owner) held) in
  let position = where any position (int64 (-1)) in
  let by_row t = reshaped ~by (Array.append (shape t) (units row)) t in
  let rows =
    along x (broadcast ~by (Array.append [| r; slots |] row) (by_row position))
  in
  let y, out =
    call ~g:(r * blocks) ~c
      (move ~by (Reshape [| r * blocks |]) owners)
      (reshaped ~by (Array.append [| r * blocks; c |] row) rows)
  in
  (* Each position's slot: its run's first slot and its rank in the run, and -1
     out of range, which a gather reads as zero. *)
  let rank =
    Eval.eval ~by
      (Value.Scatter
         {
           combine = Set;
           unique = true;
           axis = 1;
           idx = order;
           updates = broadcast ~by [| r; j |] (iota j 1);
           into = fill ~by D.Int64 [| r; j |] 0L;
         })
  in
  let slot = arith2 E.( + ) (along starts key) (sub rank (along first key)) in
  let slot = where named slot (int64 (-1)) in
  let by_out t = reshaped ~by (Array.append (shape t) (units out)) t in
  ( along
      (reshaped ~by (Array.append [| r; slots |] out) y)
      (broadcast ~by (Array.append [| r; j |] out) (by_out slot)),
    out )

let map_segments ~segments ids f x =
  let by = "Nx.map_segments" in
  if segments < 0 then invalid_argf "%s: %d segments" by segments;
  let s = shape ids and xs = shape x in
  let ns = Array.length s in
  if
    Array.length xs < ns
    || not (Array.for_all2 (fun d e -> d = e || d = 1) (Array.sub xs 0 ns) s)
  then
    invalid_argf "%s: %a does not broadcast to the ids %a" by pp_value x
      pp_value ids;
  let row = Array.sub xs ns (Array.length xs - ns) in
  let positional = ns > 0 in
  let r =
    match
      ( device_windows ~by "ids" ~positional ids,
        device_windows ~by "x" ~positional x )
    with
    | 1, r | r, 1 -> r
    | a, b when a = b -> a
    | a, b -> invalid_argf "%s: ids are split in %d windows and x in %d" by a b
  in
  let call ~g ~c owners rows =
    let y = f owners rows in
    let ys = shape y in
    if Array.length ys < 2 || ys.(0) <> g || ys.(1) <> c then
      invalid_argf "%s: f gave %a for %d blocks of %d rows" by pp_value y g c;
    (y, Array.sub ys 2 (Array.length ys - 2))
  in
  let p = numel ids in
  if segments = 0 || p = 0 then
    let y, out =
      call ~g:0 ~c:1
        (fill ~by D.Int64 [| 0 |] 0L)
        (fill ~by (dtype x) (Array.append [| 0; 1 |] row) (D.zero (dtype x)))
    in
    fill ~by (dtype y) (Array.append s out) (D.zero (dtype y))
  else
    let j = p / r in
    let ids = move ~by (Reshape [| r; j |]) ids in
    let named =
      elementwise ~by D.Bool [ Any ids ]
        E.(i64 0 <= operand 0 ids && operand 0 ids < i64 segments)
    in
    let c = block ~segments j in
    let y, out =
      if c = 1 then
        (* A group's positions as axes, and [x]'s rows in that shape,
           unbroadcast. *)
        let gs, lead =
          if not positional then ([| 1; 1 |], [| 1; 1 |])
          else
            let split d = if d = 1 then [| 1; 1 |] else [| r; d / r |] in
            let rest t = Array.sub t 1 (ns - 1) in
            ( Array.append (split s.(0)) (rest s),
              Array.append (split xs.(0)) (rest xs) )
        in
        direct ~by ~segments ~call ~named ~gs ids
          (reshaped ~by (Array.append lead row) x)
      else
        let full = broadcast ~by (Array.append s row) x in
        grouped ~by ~segments ~c ~call ~named ids
          (reshaped ~by (Array.append [| r; j |] row) full)
    in
    let named =
      reshaped ~by (Array.append [| r; j |] (units out)) named
    in
    move ~by
      (Reshape (Iarray.of_array (Array.append s out)))
      (ternary ~by Where named y (fill ~by (dtype y) [||] (D.zero (dtype y))))

(* Convolution *)

(* The arguments of a convolution of [x] by [w], [[o; k…; c / groups]],
   checked: strides, paddings, dilations and groups, all but [x]'s channels. *)
let conv_arguments ~by ?stride ?padding ?dilation ?(groups = 1) x w =
  let r = ndim x in
  if r < 3 then invalid_argf "%s: %a has fewer than three axes" by pp_value x;
  if ndim w <> r then
    invalid_argf "%s: the kernel %a has not the rank of %a" by pp_value w
      pp_value x;
  if D.is D.Boolean (dtype x) then
    invalid_argf "%s: %s has no products" by (D.name (dtype x));
  let k = r - 2 in
  let entries name default = function
    | None -> Array.make k default
    | Some a when Array.length a <> k ->
        invalid_argf "%s: %d %s for %d spatial axes" by (Array.length a) name k
    | Some a -> a
  in
  let stride = entries "strides" 1 stride in
  let dilation = entries "dilations" 1 dilation in
  let padding = entries "paddings" (0, 0) padding in
  let at_least_one name v = if v < 1 then invalid_argf "%s: %s %d" by name v in
  Array.iter (at_least_one "stride") stride;
  Array.iter (at_least_one "dilation") dilation;
  let o = dim 0 w in
  if groups < 1 || o mod groups <> 0 then
    invalid_argf "%s: %d groups for %d output channels" by groups o;
  (stride, padding, dilation, groups)

(* [w], [[o; k…; c / groups]], as [[groups; o / groups; k…; c / groups]]. *)
let by_groups ~by groups w =
  let s = shape w in
  let k = Array.length s - 2 in
  let lead = [| groups; s.(0) / groups |] in
  reshaped ~by (Array.concat [ lead; Array.sub s 1 k; [| s.(k + 1) |] ]) w

let conv ?stride ?padding ?dilation ?groups x w =
  let by = "Nx.conv" in
  let stride, padding, dilation, groups =
    conv_arguments ~by ?stride ?padding ?dilation ?groups x w
  in
  let c = dim (-1) x and cg = dim (-1) w in
  if c <> groups * cg then
    invalid_argf "%s: %a has %d channels, not %d groups of %d" by pp_value x c
      groups cg;
  let k = ndim x - 2 in
  (* [[g; og; k…; cg]] to [[g; og; cg; k…]]. *)
  let p = Array.concat [ [| 0; 1; k + 2 |]; Array.init k (fun i -> 2 + i) ] in
  windows_of ~by ~stride ~lo:(Array.map fst padding)
    ~hi:(Array.map snd padding) ~interior:(Array.make k 0) ~dilation x
    (permute ~by p (by_groups ~by groups w))

(* The transpose is the convolution at stride one of [x] with [stride - 1]
   zeros between its elements, padded by [e - 1 - lo] before, [e] the dilated
   kernel's extent, by [w] reversed along its taps and with its channels
   exchanged. The padding after, [e - 1 - hi] where the least input is the
   formula's, is the one that gives the least input's extent [d]: the padded
   axis is then [d + e - 1] long, which holds [d] windows. *)
let conv_transpose ?stride ?padding ?dilation ?groups x w =
  let by = "Nx.conv_transpose" in
  let stride, padding, dilation, groups =
    conv_arguments ~by ?stride ?padding ?dilation ?groups x w
  in
  let o = dim 0 w in
  if dim (-1) x <> o then
    invalid_argf "%s: %a has not the %d channels of the kernel %a" by pp_value
      x o pp_value w;
  let k = ndim x - 2 in
  let s = shape x and taps = Array.sub (shape w) 1 k in
  let span i = dilated taps.(i) dilation.(i) - 1 in
  (* [x]'s extent along spatial axis [i] once [stride - 1] zeros go between
     its elements. *)
  let spread i =
    let t = s.(1 + i) in
    if t = 0 then 0 else ((t - 1) * stride.(i)) + 1
  in
  let least i =
    let t = s.(1 + i) and lo, hi = padding.(i) in
    let d =
      if t = 0 then 0 else Stdlib.max 0 (spread i + span i - lo - hi)
    in
    if windows_along d (lo, hi) taps.(i) stride.(i) dilation.(i) <> t then
      invalid_argf "%s: no input has a convolution of %a" by pp_value x;
    d
  in
  let target = Array.init k least in
  let lo = Array.init k (fun i -> span i - fst padding.(i)) in
  let hi = Array.init k (fun i -> target.(i) + span i - spread i - lo.(i)) in
  (* [[g; og; k…; cg]] to [[g; cg; og; k…]], the taps reversed. *)
  let p = Array.concat [ [| 0; k + 2; 1 |]; Array.init k (fun i -> 2 + i) ] in
  let w = permute ~by p (by_groups ~by groups w) in
  let taps_axes = List.init k (fun i -> 3 + i) in
  let w = renamed ~by (fun () -> flip ~axes:taps_axes w) in
  windows_of ~by ~stride:(Array.make k 1) ~lo ~hi
    ~interior:(Array.map (fun st -> st - 1) stride)
    ~dilation x w

(* Fourier transforms *)

type fft_norm = Backward | Forward | Ortho

let transform ~by f = Eval.eval ~by (Value.Fft f)

(* The axes [l] of [x] (its last by default), distinct, and the points along
   each: [s]'s, or [points last d] of each extent [d], [last] for the last
   axis. *)
let transform_axes ~by ?(points = fun _ d -> d) x l s =
  if ndim x = 0 then invalid_argf "%s: %a has no axis" by pp_value x;
  let axes = match l with None -> [ ndim x - 1 ] | Some l -> axes ~by x l in
  if axes = [] then invalid_argf "%s: no axis to transform" by;
  let n = List.length axes in
  let s =
    match s with
    | None -> List.mapi (fun i a -> points (i = n - 1) (dim a x)) axes
    | Some s when List.length s <> n ->
        invalid_argf "%s: %d lengths for %d axes" by (List.length s) n
    | Some s ->
        List.iter (fun d -> if d < 0 then invalid_argf "%s: length %d" by d) s;
        s
  in
  (axes, s)

(* [x] cut to [d] elements along each axis [a] of [axes], or extended with
   zeros. *)
let resized ~by x axes sizes =
  let s = shape x in
  if List.for_all2 (fun a d -> s.(a) = d) axes sizes then x
  else
    let target = Array.copy s in
    List.iter2 (fun a d -> target.(a) <- d) axes sizes;
    let kept =
      Iarray.init (Array.length s) (fun i ->
          whole (Stdlib.min s.(i) target.(i)))
    in
    let piece = move ~by (Slice kept) x in
    assemble ~by (dtype x) target (D.zero (dtype x)) [ (kept, piece) ]

(* The factor of a transform of [n] points in all, forward or not, under
   [norm]. *)
let factor norm ~forward n =
  if n = 0 then 1.
  else
    match (norm, forward) with
    | Backward, true | Forward, false -> 1.
    | Backward, false | Forward, true -> 1. /. Float.of_int n
    | Ortho, _ -> 1. /. Float.sqrt (Float.of_int n)

let scaled ~by c x =
  if c = 1. then x else binary ~by (Binary Mul) x (fill ~by (dtype x) [||] c)

(* [z] times the real [c] part by part: a complex product by [c + 0i] makes a
   NaN part of an infinite one. *)
let scaled_parts (type s d) ~by c (z : (Complex.t, s, d) t) :
    (Complex.t, s, d) t =
  let parts (type f) (f : (float, f) D.t) =
    let p = Eval.eval ~by (Value.Bitcast (f, z)) in
    Eval.eval ~by (Value.Bitcast (dtype z, scaled ~by c p))
  in
  if c = 1. then z
  else
    match dtype z with
    | D.Complex64 -> parts D.Float32
    | D.Complex128 -> parts D.Float64

(* A real transform's compute formats: a float format and the complex one of
   its precision. *)
type pair = Pair : (float, 'r) D.t * (Complex.t, 'c) D.t -> pair

(* The pair of the wider of a float [dt]'s compute format ({!Prim.wide}) and a
   complex [c]'s precision. *)
let pair dt c =
  let (Prim.Wide w) = Prim.wide dt in
  if D.bits w > 32 || D.bits c > 64 then Pair (D.Float64, D.Complex128)
  else Pair (D.Float32, D.Complex64)

(* The other axes, ascending, the real one, the last of [axes], and whether it
   is past the others: the kernels' real transforms run along the last of
   ascending axes. *)
let real_axes axes =
  match List.rev axes with
  | [] -> invalid_arg "real_axes"
  | real :: others ->
      let others = List.sort compare others in
      (Iarray.of_list others, real, List.for_all (fun a -> a < real) others)

let complex_by ~by direction ?axes ?s ?(norm = Backward) x =
  let axes, sizes = transform_axes ~by x axes s in
  let x = resized ~by x axes sizes in
  let n = List.fold_left ( * ) 1 sizes in
  let axes = Iarray.of_list (List.sort compare axes) in
  let y = transform ~by (C2c { direction; axes; x }) in
  let forward = direction = Nx_kernel.Forward in
  scaled_parts ~by (factor norm ~forward n) y

let fft ?axes ?s ?norm x =
  complex_by ~by:"Nx.fft" Nx_kernel.Forward ?axes ?s ?norm x

let ifft ?axes ?s ?norm x =
  complex_by ~by:"Nx.ifft" Nx_kernel.Inverse ?axes ?s ?norm x

let rfft dt ?axes ?s ?(norm = Backward) x =
  let by = "Nx.rfft" in
  let axes, sizes = transform_axes ~by x axes s in
  let x = resized ~by x axes sizes in
  let n = List.fold_left ( * ) 1 sizes in
  let (Pair (r, c)) = pair (dtype x) dt in
  let x = cast_by ~by r x in
  let others, real, last = real_axes axes in
  let y =
    if last then
      transform ~by
        (R2c { dtype = c; axes = Iarray.append others [| real |]; x })
    else
      let y = transform ~by (R2c { dtype = c; axes = [| real |]; x }) in
      if Iarray.length others = 0 then y
      else
        transform ~by
          (C2c { direction = Nx_kernel.Forward; axes = others; x = y })
  in
  cast_by ~by dt (scaled_parts ~by (factor norm ~forward:true n) y)

let irfft dt ?axes ?s ?(norm = Backward) z =
  let by = "Nx.irfft" in
  let default last d =
    if not last then d
    else if d = 0 then invalid_argf "%s: %a has no bin to invert" by pp_value z
    else 2 * (d - 1)
  in
  let axes, sizes = transform_axes ~by ~points:default z axes s in
  let n = List.fold_left ( * ) 1 sizes in
  let others, real, last = real_axes axes in
  let last_axis = List.length sizes - 1 in
  let points = List.nth sizes last_axis in
  let bins =
    List.mapi (fun i d -> if i = last_axis then (d / 2) + 1 else d) sizes
  in
  let z = resized ~by z axes bins in
  let (Pair (r, c)) = pair dt (dtype z) in
  let z = cast_by ~by c z in
  let y =
    if last then
      let axes = Iarray.append others [| real |] in
      transform ~by (C2r { dtype = r; n = points; axes; x = z })
    else
      let z =
        if Iarray.length others = 0 then z
        else
          transform ~by
            (C2c { direction = Nx_kernel.Inverse; axes = others; x = z })
      in
      transform ~by (C2r { dtype = r; n = points; axes = [| real |]; x = z })
  in
  cast_by ~by dt (scaled ~by (factor norm ~forward:false n) y)

(* Cosine and sine transforms, each from one complex transform along the last
   axis. *)

type trig_type = Type_1 | Type_2 | Type_3 | Type_4

let along_last x range =
  slice (List.init (ndim x - 1) (fun _ -> A) @ [ range ]) x

let flipped x = flip ~axes:[ -1 ] x

(* [x]'s last axis extended with [n] zeros. *)
let extended_by ~by x n =
  let s = Array.copy (shape x) in
  s.(Array.length s - 1) <- n;
  concatenate ~axis:(-1) [ x; fill ~by (dtype x) s (D.zero (dtype x)) ]

let odd_bins x n = take ~axis:(-1) (arange D.Int64 1 (2 * n) 2) x

(* The [n] elements of [dt] along one axis, [f k] at [k], computed at
   float64. *)
let along_axis ~by dt n f =
  leaf ~by dt [| n |] (at dt (f E.(cast D.Float64 (coord 0))))

(* [x]'s last axis's first element, or its last, times [c]. *)
let scale_end ~by ~first c x =
  let n = dim (-1) x in
  let k = Float.of_int (if first then 0 else n - 1) in
  let lit = E.const D.Float64 in
  let factors i = E.(where (i = lit k) (lit c) (lit 1.)) in
  mul x (along_axis ~by (dtype x) n factors)

let rec cosine : type r c d.
    by:string ->
    trig_type ->
    (float, r) D.t ->
    (Complex.t, c) D.t ->
    (float, r, d) t ->
    (float, r, d) t =
 fun ~by kind f c x ->
  let n = dim (-1) x in
  let spectrum y = fft (cast c y) in
  match kind with
  | Type_1 ->
      let interior = flipped (along_last x (R (1, n - 1))) in
      let mirrored = concatenate ~axis:(-1) [ x; interior ] in
      real f (along_last (spectrum mirrored) (R (0, n)))
  | Type_2 ->
      (* [2 Re (X[k] e^(-iπk/2n))] of the even elements, then the odd ones
         reversed. *)
      let even = along_last x (Rs (0, n, 2)) in
      let odd = flipped (along_last x (Rs (1, n, 2))) in
      let z = spectrum (concatenate ~axis:(-1) [ even; odd ]) in
      let step = Float.pi /. (2. *. Float.of_int n) in
      let theta =
        along_axis ~by f n (fun k -> E.(k * const D.Float64 step))
      in
      let re = mul (real f z) (cos theta) and im = mul (imag f z) (sin theta) in
      mul_s (add re im) 2.
  | Type_3 ->
      let tail = flipped (along_last x (R (1, n))) in
      let mirrored =
        concatenate ~axis:(-1) [ extended_by ~by x ((2 * n) + 1); tail ]
      in
      real f (odd_bins (spectrum mirrored) n)
  | Type_4 -> odd_bins (cosine ~by Type_2 f c (extended_by ~by x n)) n

let sine (type r c d) ~by kind (f : (float, r) D.t) (c : (Complex.t, c) D.t)
    (x : (float, r, d) t) : (float, r, d) t =
  let n = dim (-1) x in
  let lit = E.const D.Float64 in
  let signs =
    along_axis ~by f n (fun k ->
        E.(where (rem k (lit 2.) = lit 0.) (lit 1.) (lit (-1.))))
  in
  match kind with
  | Type_1 ->
      let s = Array.copy (shape x) in
      s.(Array.length s - 1) <- 1;
      let zero = fill ~by f s 0. in
      let odd = concatenate ~axis:(-1) [ zero; x; zero; neg (flipped x) ] in
      neg (imag f (along_last (fft (cast c odd)) (R (1, n + 1))))
  | Type_2 -> flipped (cosine ~by Type_2 f c (mul x signs))
  | Type_3 -> mul signs (cosine ~by Type_3 f c (flipped x))
  | Type_4 -> flipped (cosine ~by Type_4 f c (mul x signs))

type family = Cosine | Sine

(* The orthonormal transform's end weights: [√2] on the way in and [1/√2] on
   the way out, where the plain transform weighs an end point apart. *)
let ortho_ends ~by family kind ~input x =
  let w = if input then Float.sqrt 2. else 1. /. Float.sqrt 2. in
  match (family, kind, input) with
  | Cosine, Type_1, _ ->
      scale_end ~by ~first:false w (scale_end ~by ~first:true w x)
  | Cosine, Type_3, true | Cosine, Type_2, false ->
      scale_end ~by ~first:true w x
  | Sine, Type_3, true | Sine, Type_2, false -> scale_end ~by ~first:false w x
  | _ -> x

let trigonometric (type s d) ~by family ~inverse ?(type_ = Type_2) ?axes
    ?(norm = Backward) (x : (float, s, d) t) : (float, s, d) t =
  renamed ~by (fun () ->
      let axes, _ = transform_axes ~by x axes None in
      if family = Cosine && type_ = Type_1 then
        List.iter
          (fun a ->
            if dim a x = 1 then
              invalid_argf "%s: type 1 along axis %d of one element" by a)
          axes;
      let kind =
        match (inverse, type_) with
        | true, Type_2 -> Type_3
        | true, Type_3 -> Type_2
        | _, k -> k
      in
      let length n =
        match (family, type_) with
        | Cosine, Type_1 -> 2 * (n - 1)
        | Sine, Type_1 -> 2 * (n + 1)
        | _ -> 2 * n
      in
      let ends ~input y =
        match norm with
        | Ortho -> ortho_ends ~by family kind ~input y
        | Backward | Forward -> y
      in
      (* The transform at [f], through complex transforms at [c]. *)
      let run : type r c.
          (float, r) D.t -> (Complex.t, c) D.t -> (float, r, d) t ->
          (float, r, d) t =
       fun f c x ->
        let along x a =
          let y = ends ~input:true (moveaxis a (-1) x) in
          let y =
            match family with
            | Cosine -> cosine ~by kind f c y
            | Sine -> sine ~by kind f c y
          in
          let scale = factor norm ~forward:(not inverse) (length (dim a x)) in
          moveaxis (-1) a (scaled ~by scale (ends ~input:false y))
        in
        List.fold_left along x axes
      in
      if Array.exists (( = ) 0) (shape x) then copy x
      else
        (* At [x]'s compute format, rounded once to its dtype. *)
        let (Pair (f, c)) = pair (dtype x) D.Complex64 in
        cast_by ~by (dtype x) (run f c (cast_by ~by f x)))

let dct ?type_ ?axes ?norm x =
  trigonometric ~by:"Nx.dct" Cosine ~inverse:false ?type_ ?axes ?norm x

let idct ?type_ ?axes ?norm x =
  trigonometric ~by:"Nx.idct" Cosine ~inverse:true ?type_ ?axes ?norm x

let dst ?type_ ?axes ?norm x =
  trigonometric ~by:"Nx.dst" Sine ~inverse:false ?type_ ?axes ?norm x

let idst ?type_ ?axes ?norm x =
  trigonometric ~by:"Nx.idst" Sine ~inverse:true ?type_ ?axes ?norm x

(* Short-time transforms *)

let hann_by ~by dt n =
  if n < 0 then invalid_argf "%s: %d points" by n;
  let step = 2. *. Float.pi /. Float.of_int n in
  let lit = E.const D.Float64 in
  leaf ~by dt [| n |]
    (at dt E.(lit 0.5 - (lit 0.5 * cos (cast D.Float64 (coord 0) * lit step))))

let hann dt n = hann_by ~by:"Nx.hann" dt n

let frame_step ~window = function
  | Some s -> s
  | None -> Stdlib.max 1 (window / 4)

(* [win], or the Hann window, of [window] points. *)
let taper ~by dt ~window = function
  | None -> hann_by ~by dt window
  | Some w ->
      if not (Prim.has_shape w [| window |]) then
        invalid_argf "%s: win %a is not [%d]" by pp_value w window;
      w

let stft dt ~window ?step ?win x =
  let by = "Nx.stft" in
  renamed ~by (fun () ->
      let step = frame_step ~window step in
      if ndim x = 0 then invalid_argf "%s: %a has no axis" by pp_value x;
      if window < 1 || step < 1 then
        invalid_argf "%s: window %d, step %d" by window step;
      if window > dim (-1) x then
        invalid_argf "%s: window %d exceeds the last axis of %a" by window
          pp_value x;
      let w = taper ~by (dtype x) ~window win in
      rfft dt (mul (sliding_window ~window ~step x) w))

let istft dt ~window ?step ?win ?length z =
  let by = "Nx.istft" in
  renamed ~by (fun () ->
      let step = frame_step ~window step in
      let r = ndim z in
      if r < 2 then invalid_argf "%s: %a has fewer than two axes" by pp_value z;
      if window < 1 then invalid_argf "%s: window %d" by window;
      if step < 1 || step > window then
        invalid_argf "%s: step %d is outside [1, %d]" by step window;
      let bins = (window / 2) + 1 in
      if dim (-1) z <> bins then
        invalid_argf "%s: %a has not %d bins" by pp_value z bins;
      (match length with
      | Some l when l < 1 -> invalid_argf "%s: length %d" by l
      | _ -> ());
      let frames = dim (-2) z in
      let points = ((frames - 1) * step) + window in
      let w = taper ~by dt ~window win in
      (* Overlap-add is the adjoint of taking the frames. *)
      let added t =
        combine_patches ~output_size:[| points |] ~kernel_size:[| window |]
          ~stride:[| step |] ~dilation:[| 1 |] ~padding:[| (0, 0) |]
          (matrix_transpose t)
      in
      let signal = added (mul (irfft dt ~s:[ window ] z) w) in
      let envelope = added (broadcast ~by [| frames; window |] (square w)) in
      (* Where every frame's taper is zero both are, and the sample is [0.]. *)
      let zero = zeros_like envelope in
      let safe = where (equal envelope zero) (ones_like envelope) envelope in
      let y = div signal safe in
      match length with None -> y | Some l -> resized ~by y [ r - 2 ] [ l ])

(* Printing *)

let pp_dtype = D.pp

(* Past [elided] elements, an axis longer than [2 * edge] prints its first and
   last [edge] elements around [...]. *)
let elided = 1000
let edge = 3

(* [a]'s elements as nested rows, a 0-d array's as its one element. *)
let pp_elements (type v s) ppf (a : (v, s) Nx_array.t) =
  let dt = Nx_array.dtype a and s = Nx_array.Layout.shape (Nx_array.layout a) in
  let r = Array.length s in
  let cut = Array.fold_left ( * ) 1 s > elided in
  let i = Array.make r 0 in
  let rec row axis =
    if axis = r then D.pp_value dt ppf (Nx_array.get a i)
    else begin
      let d = s.(axis) in
      let shown j = (not cut) || d <= 2 * edge || j < edge || j >= d - edge in
      Format.fprintf ppf "@[<hv 1>[";
      for j = 0 to d - 1 do
        if shown j then begin
          if j > 0 then Format.fprintf ppf ",@ ";
          i.(axis) <- j;
          row (axis + 1)
        end
        else if j = edge then Format.fprintf ppf ",@ ..."
      done;
      Format.fprintf ppf "]@]"
    end
  in
  row 0

let pp ppf x =
  let set ppf p =
    let s = Devices.set p in
    if Devices.number s <> Devices.number Devices.host then
      Format.fprintf ppf " on %a" Devices.pp s
  in
  (* What follows the header, read before anything prints: a failed check
     raises out of the print, as out of any read, with nothing printed. *)
  let rest : Format.formatter -> unit =
    match x with
    | Value.Traced { owner; _ } ->
        fun ppf -> Format.fprintf ppf " traced by %s" owner.name
    | Value.Deferred { node = Value.Node n; _ } ->
        fun ppf -> Format.fprintf ppf " of every set,@ %a" Prim.pp n.op
    | Value.Array _ | Value.Shards _ | Value.Donated _ -> (
        match Prim.death x with
        | Value.Died f -> fun ppf -> Format.fprintf ppf ", donated to %s" f
        | Value.Live | Value.Offered _ -> (
            let lost d why ppf =
              Format.fprintf ppf ", lost with %s: %s" (Rig.name d) why
            in
            let x = Exec.live x in
            match Exec.wait ~by:"Nx.pp" x with
            | exception Rig.Lost (d, why) -> lost d why
            | () -> (
                match Exec.on_host ~by:"Nx.pp" x with
                | a -> fun ppf -> Format.fprintf ppf "@ %a" pp_elements a
                | exception Rig.Lost (d, why) -> lost d why
                | exception Invalid_argument why ->
                    fun ppf -> Format.fprintf ppf ", unread: %s" why)))
  in
  Format.fprintf ppf "@[<hv 2>%s %a" (D.name (dtype x)) pp_shape (shape x);
  Option.iter (set ppf) (Prim.at x);
  rest ppf;
  Format.fprintf ppf "@]"

let to_string x = Format.asprintf "%a" pp x
let print x = Format.printf "%a@." pp x

(* Operations as data *)

module Prim = struct
  type ('v, 's, 'd) form = ('v, 's, 'd) Value.form = {
    dtype : ('v, 's) dtype;
    layout : Nx_array.Layout.t;
    placement : 'd Placement.t option;
  }

  type 'd any = 'd Value.any = Any : ('v, 's, 'd) t -> 'd any
  type 'd load = 'd Value.load =
    | Plain : ('v, 's, 'd) t -> 'd load
    | Padded : {
        x : ('v, 's, 'd) t;
        fill : 'v;
        pad : Nx_kernel.pad;
      }
        -> 'd load

  (* Declared before [reductions] and [outs]: an unannotated [Prim.[ ... ]] is
     the list type declared last, [outs]. *)
  type ('d, 'e) data = ('d, 'e) Value.data =
    | [] : ('d, unit) data
    | ( :: ) : ('v, 's, 'd) Value.t * ('d, 'e) data -> ('d, 'v * 'e) data

  type ('d, 'a) reduction = ('d, 'a) Value.reduction =
    | Monoid :
        Nx_kernel.monoid * int * ('v, 's) dtype
        -> ('d, ('v, 's, 'd) Value.t) reduction
    | Moments :
        int * ('v, 's) dtype
        -> ('d, ('v, 's, 'd) Value.t * ('v, 's, 'd) Value.t) reduction
    | Arg :
        Nx_kernel.extreme * int * ('v, 's) dtype
        -> ( 'd,
             ('v, 's, 'd) Value.t * (int64, Dtype.int64_elt, 'd) Value.t )
           reduction

  type ('d, 'r) reductions = ('d, 'r) Value.reductions =
    | [] : ('d, unit) reductions
    | ( :: ) :
        ('d, 'a) reduction * ('d, 'r) reductions
        -> ('d, 'a * 'r) reductions

  type ('d, 'r) outs = ('d, 'r) Value.outs =
    | [] : ('d, unit) outs
    | ( :: ) : ('v, 's) dtype * ('d, 'r) outs -> ('d, ('v, 's, 'd) t * 'r) outs

  type 'd index = (int64, Dtype.int64_elt, 'd) Value.t

  type ('d, 'r) fft = ('d, 'r) Value.fft =
    | C2c : {
        direction : Nx_kernel.direction;
        axes : int iarray;
        x : (Complex.t, 's, 'd) Value.t;
      }
        -> ('d, (Complex.t, 's, 'd) Value.t) fft
    | R2c : {
        dtype : (Complex.t, 'c) dtype;
        axes : int iarray;
        x : (float, 's, 'd) Value.t;
      }
        -> ('d, (Complex.t, 'c, 'd) Value.t) fft
    | C2r : {
        dtype : (float, 'r) dtype;
        n : int;
        axes : int iarray;
        x : (Complex.t, 's, 'd) Value.t;
      }
        -> ('d, (float, 'r, 'd) Value.t) fft

  type ('d, 'r) linalg = ('d, 'r) Value.linalg =
    | Cholesky : {
        triangle : Nx_kernel.triangle;
        a : ('v, 's, 'd) Value.t;
      }
        -> ('d, ('v, 's, 'd) Value.t) linalg
    | Lu :
        ('v, 's, 'd) Value.t
        -> ('d, ('v, 's, 'd) Value.t * 'd index * 'd index) linalg
    | Qr : {
        factors : Nx_kernel.factors;
        a : ('v, 's, 'd) Value.t;
      }
        -> ('d, ('v, 's, 'd) Value.t * ('v, 's, 'd) Value.t) linalg
    | Svd : {
        vectors :
          ( Nx_kernel.factors,
            ('v, 's, 'd) Value.t * ('v, 's, 'd) Value.t,
            'r )
          Nx_kernel.vectors;
        a : ('v, 's, 'd) Value.t;
      }
        -> ('d, ('v, 's, 'd) Value.t * 'r) linalg
    | Eigh : {
        vectors : (unit, ('v, 's, 'd) Value.t, 'r) Nx_kernel.vectors;
        a : ('v, 's, 'd) Value.t;
      }
        -> ('d, ('v, 's, 'd) Value.t * 'r) linalg
    | Eig : {
        vectors : (unit, (Complex.t, 'c, 'd) Value.t, 'r) Nx_kernel.vectors;
        dtype : (Complex.t, 'c) dtype;
        a : ('v, 's, 'd) Value.t;
      }
        -> ('d, (Complex.t, 'c, 'd) Value.t * 'r) linalg
    | Solve_triangular : {
        triangle : Nx_kernel.triangle;
        adjoint : bool;
        unit_diagonal : bool;
        a : ('v, 's, 'd) Value.t;
        b : ('v, 's, 'd) Value.t;
      }
        -> ('d, ('v, 's, 'd) Value.t) linalg

  type 'r t = 'r Value.prim =
    | Map : {
        layout : Nx_array.Layout.t;
        prog : Nx_kernel.Prog.t;
        outs : ('d, 'r) outs;
        loads : 'd load iarray;
      }
        -> 'r t
    | Reduce : {
        layout : Nx_array.Layout.t;
        axes : int iarray;
        prog : Nx_kernel.Prog.t;
        reductions : ('d, 'r) reductions;
        loads : 'd load iarray;
      }
        -> 'r t
    | Scan : {
        layout : Nx_array.Layout.t;
        axis : int;
        prog : Nx_kernel.Prog.t;
        reduction : ('d, 'r) reduction;
        loads : 'd load iarray;
        carry : 'd any option;
      }
        -> 'r t
    | Gather : {
        axis : int;
        idx : (int64, Dtype.int64_elt, 'd) Value.t;
        x : ('v, 's, 'd) Value.t;
      }
        -> ('v, 's, 'd) Value.t t
    | Scatter : {
        combine : Nx_kernel.combine;
        unique : bool;
        axis : int;
        idx : (int64, Dtype.int64_elt, 'd) Value.t;
        updates : ('v, 's, 'd) Value.t;
        into : ('v, 's, 'd) Value.t;
      }
        -> ('v, 's, 'd) Value.t t
    | Sort : {
        axis : int;
        descending : bool;
        k : int option;
        x : ('v, 's, 'd) Value.t;
      }
        -> (('v, 's, 'd) Value.t * (int64, Dtype.int64_elt, 'd) Value.t) t
    | Fold : {
        shape : int iarray;
        pad : Nx_kernel.pad;
        x : ('v, 's, 'd) Value.t;
      }
        -> ('v, 's, 'd) Value.t t
    | Assemble : {
        dtype : ('v, 's) dtype;
        shape : int iarray;
        fill : 'v;
        pieces : (Nx_array.Move.range iarray * ('v, 's, 'd) Value.t) list;
      }
        -> ('v, 's, 'd) Value.t t
    | Contract : {
        order : Nx_kernel.order;
        batch : (int * int) iarray;
        contracting : (int * int) iarray;
        acc : Nx_array.Dtype.any;
        out : ('v, 's) dtype;
        a : ('a, 'b, 'd) Value.t;
        b : ('c, 'e, 'd) Value.t;
        init : ('v, 's, 'd) Value.t option;
      }
        -> ('v, 's, 'd) Value.t t
    | Fft : ('d, 'r) fft -> 'r t
    | Linalg : ('d, 'r) linalg -> 'r t
    | Copy : ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t t
    | Move : Nx_array.Move.t * ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t t
    | Bitcast : ('w, 'r) dtype * ('v, 's, 'd) Value.t -> ('w, 'r, 'd) Value.t t
    | Place : 'e Placement.t * ('v, 's, 'd) Value.t -> ('v, 's, 'e) Value.t t
    | Check : {
        ok : (bool, Dtype.bool_elt, 'd) Value.t;
        data : ('d, 'e) data;
        fail : int array -> 'e -> exn;
      }
        -> unit t

  type operands = Prim.operands = Operands : 'd any list -> operands

  let name = Prim.name
  let pp = Prim.pp
  let operands = Prim.operands
  let map = Prim.map
  let form = Prim.form
  let results = Prim.results
  let contract_order = Prim.contract_order

  type interpretation = Value.interpretation
  type reach = Value.reach = Values | Extent
  type ('v, 's, +'d) payload = ('v, 's, 'd) Value.payload = ..

  let interpret = Eval.interpret
  let traced = Eval.traced
  let payload = Eval.payload
  let owner = Eval.owner
  let later = Eval.later
  let eval = Eval.eval
  let expand = Eval.expand
end

(* Operators *)

module Infix = struct
  let ( = ) a b = equal a b
  let ( <> ) a b = not_equal a b
  let ( < ) a b = less a b
  let ( <= ) a b = less_equal a b
  let ( > ) a b = greater a b
  let ( >= ) a b = greater_equal a b
  let ( =$ ) x c = equal_s x c
  let ( <>$ ) x c = not_equal_s x c
  let ( <$ ) x c = less_s x c
  let ( <=$ ) x c = less_equal_s x c
  let ( >$ ) x c = greater_s x c
  let ( >=$ ) x c = greater_equal_s x c
  let ( && ) a b = logical_and a b
  let ( || ) a b = logical_or a b
end

let ( + ) a b = add a b
let ( - ) a b = sub a b
let ( * ) a b = mul a b
let ( / ) a b = div a b
let ( ** ) a b = pow a b
let ( ~- ) x = neg x
let ( +$ ) x c = add_s x c
let ( -$ ) x c = sub_s x c
let ( *$ ) x c = mul_s x c
let ( /$ ) x c = div_s x c
