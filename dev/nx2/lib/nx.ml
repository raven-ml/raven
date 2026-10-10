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
  let prog = Prim.program (Const (D.Any dt, bits ~by dt v)) [||] in
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

(* A 0-d value has no axis to cut: beside a cut [x] it is whole on each of
   [x]'s devices. *)
let scalar_like x v =
  let by = "Nx.scalar_like" in
  let c = fill ~by (dtype x) [||] v in
  match Prim.at x with
  | None -> c
  | Some p ->
      let g = Devices.grid p in
      let whole =
        Array.fold_left (fun g (axis, _) -> Grid.uncut g ~axis) g (Grid.cuts g)
      in
      let p = if whole == g then p else Devices.v ~by (Devices.set p) whole in
      Eval.eval ~by (Value.Place (p, c))

(* Leaves *)

(* A program under construction: its nodes, newest first. *)
type program = { mutable nodes : P.node list; mutable count : int }

let node b n =
  b.nodes <- n :: b.nodes;
  b.count <- b.count + 1;
  b.count - 1

let const b dt v = node b (Const (D.Any dt, P.bits dt v))

(* The value of [dt] and [shape] whose element at each index is node [body b] of
   a program with no operand, reading the index through [Coord]. *)
let leaf ~by dt shape body =
  let layout = renamed ~by (fun () -> Nx_array.Layout.contiguous shape) in
  let b = { nodes = []; count = 0 } in
  let out = body b in
  let prog = P.v ~ins:[||] (Array.of_list (List.rev b.nodes)) ~outs:[| out |] in
  let x, () =
    Eval.eval ~by (Value.Map { layout; prog; outs = Value.[ dt ]; loads = [||] })
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
  leaf ~by dt [| n |] (fun b ->
      let i = node b (Coord 0) in
      let v = node b (Op2 (Binary Mul, i, const b D.Int64 (Int64.of_int step))) in
      let v = node b (Op2 (Binary Add, v, const b D.Int64 (Int64.of_int start))) in
      node b (Op1 (Cast, D.Any dt, v)))

(* [start + i step] for the index [i], by one fused multiply-add at float64. *)
let ramp b ~start ~step =
  let i = node b (Op1 (Cast, D.Any D.Float64, node b (Coord 0))) in
  node b
    (Op3 (Fma, i, const b D.Float64 step, const b D.Float64 start))

let arange_f dt start stop step =
  let by = "Nx.arange_f" in
  if step = 0. then invalid_argf "%s: step 0" by;
  let length = Float.ceil ((stop -. start) /. step) in
  if (not (Float.is_finite length)) || length >= Float.of_int max_int then
    invalid_argf "%s: [%g, %g) by %g has no finite length" by start stop step;
  let n = if length > 0. then Float.to_int length else 0 in
  leaf ~by dt [| n |] (fun b -> node b (Op1 (Cast, D.Any dt, ramp b ~start ~step)))

(* [n] values from [start] to [stop] at float64: [stop] itself last with
   [endpoint], so the ends are exact. *)
let spaced ~by b ~endpoint start stop n =
  if n < 0 then invalid_argf "%s: %d values" by n;
  let div = if endpoint then n - 1 else n in
  let step = if div > 0 then (stop -. start) /. Float.of_int div else 0. in
  let v = ramp b ~start ~step in
  if not (endpoint && n >= 2) then v
  else
    let last =
      node b (Op2 (Compare Equal, node b (Coord 0), const b D.Int64 (Int64.of_int (n - 1))))
    in
    node b (Op3 (Where, last, const b D.Float64 stop, v))

let linspace dt ?(endpoint = true) start stop n =
  let by = "Nx.linspace" in
  leaf ~by dt [| Stdlib.max n 0 |] (fun b ->
      node b (Op1 (Cast, D.Any dt, spaced ~by b ~endpoint start stop n)))

let logspace dt ?(endpoint = true) ?(base = 10.) start stop n =
  let by = "Nx.logspace" in
  leaf ~by dt [| Stdlib.max n 0 |] (fun b ->
      let e = spaced ~by b ~endpoint start stop n in
      let v = node b (Op2 (Binary Pow, const b D.Float64 base, e)) in
      node b (Op1 (Cast, D.Any dt, v)))

let eye ?m ?(k = 0) dt n =
  let m = Option.value m ~default:n in
  leaf ~by:"Nx.eye" dt [| n; m |] (fun b ->
      (* [Coord 0] is the column, [Coord 1] the row. *)
      let d = node b (Op2 (Binary Sub, node b (Coord 0), node b (Coord 1))) in
      let on = node b (Op2 (Compare Equal, d, const b D.Int64 (Int64.of_int k))) in
      node b (Op3 (Where, on, const b dt (D.one dt), const b dt (D.zero dt))))

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

(* [x]'s elements in an array on the host, for the read [by]. A read is not an
   operation: no interpretation receives it. *)
let readable ~by x =
  Prim.alive ~by 0 x;
  match Interp.owner x with
  | Some i -> invalid_argf "%s: the value is traced by %s" by i.name
  | None -> Exec.live x

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
  let x = readable ~by x in
  let one = Array.of_list (List.mapi at i) in
  let v = Exec.run ~by (Value.Move (Slice one, x)) in
  Nx_array.get (Exec.on_host ~by v) (Array.make r 0)

let broadcast ~by s x =
  if Prim.has_shape x s then x else Eval.eval ~by (Value.Move (Broadcast s, x))

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
let div (type v s d) (a : (v, s, d) t) (b : (v, s, d) t) =
  let k =
    match D.kind (dtype a) with D.Signed | D.Unsigned -> P.Idiv | _ -> Fdiv
  in
  binary ~by:"Nx.div" (Binary k) a b

let mod_ a b = binary ~by:"Nx.mod_" (Binary Mod) a b
let pow a b = binary ~by:"Nx.pow" (Binary Pow) a b
let atan2 a b = binary ~by:"Nx.atan2" (Binary Atan2) a b
let maximum a b = binary ~by:"Nx.maximum" (Binary Maximum) a b
let minimum a b = binary ~by:"Nx.minimum" (Binary Minimum) a b
let bitwise_and a b = binary ~by:"Nx.bitwise_and" (Binary And) a b
let bitwise_or a b = binary ~by:"Nx.bitwise_or" (Binary Or) a b
let bitwise_xor a b = binary ~by:"Nx.bitwise_xor" (Binary Xor) a b

let comparison ~by k a b =
  if same_shape a b then Eval.apply2 ~by (Compare k) D.Bool a b
  else
    let s = Prim.broadcast_shape ~by (shape a) (shape b) in
    Eval.apply2 ~by (Compare k) D.Bool (broadcast ~by s a) (broadcast ~by s b)

let equal a b = comparison ~by:"Nx.equal" Equal a b
let not_equal a b = comparison ~by:"Nx.not_equal" Not_equal a b
let less a b = comparison ~by:"Nx.less" Less a b
let less_equal a b = comparison ~by:"Nx.less_equal" Less_equal a b
let greater a b = comparison ~by:"Nx.greater" Less b a
let greater_equal a b = comparison ~by:"Nx.greater_equal" Less_equal b a

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

(* [a] as an axis of a value of rank [r], counting from the end where negative.
   [what] names the value in the message. *)
let axis_of ~by r a what =
  let a' = if a < 0 then a + r else a in
  if a' < 0 || a' >= r then invalid_argf "%s: %d is not an axis of %t" by a what
  else a'

let axis ~by x a = axis_of ~by (ndim x) a (fun ppf -> pp_value ppf x)
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

(* [a * b] for extents, or raises naming [by] past an [int]. *)
let times ~by x a b =
  if a <> 0 && b > max_int / a then
    invalid_argf "%s: %a would have more elements than an int counts" by
      pp_value x;
  a * b

let reshape s x =
  let by = "Nx.reshape" in
  let n = numel x in
  let s = Array.copy s in
  let unknown = ref None and known = ref 1 in
  Array.iteri
    (fun i e ->
      if e < -1 then invalid_argf "%s: extent %d in %a" by e pp_shape s;
      if e = -1 then begin
        if !unknown <> None then
          invalid_argf "%s: %a has two unknown extents" by pp_shape s;
        unknown := Some i
      end
      else known := times ~by x !known e)
    s;
  (match !unknown with
  | Some i when !known > 0 && n mod !known = 0 -> s.(i) <- n / !known
  | Some _ ->
      invalid_argf "%s: %a has %d elements, which %a cannot hold" by pp_value x
        n pp_shape s
  | None ->
      if !known <> n then
        invalid_argf "%s: %a has %d elements, %a has %d" by pp_value x n
          pp_shape s !known);
  move ~by (Reshape s) x

let broadcast_to s x =
  let by = "Nx.broadcast_to" in
  let fits =
    Array.for_all (fun e -> e >= 0) s
    && match Prim.merge (shape x) s with Ok s' -> s' = s | Error _ -> false
  in
  if not fits then
    invalid_argf "%s: %a does not broadcast to %a" by pp_value x pp_shape s;
  move ~by (Broadcast (Array.copy s)) x

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
  move ~by (Reshape (Array.of_list kept)) x

let unsqueeze ~axes:l x =
  let by = "Nx.unsqueeze" in
  let s = shape x in
  let r = Array.length s + List.length l in
  let added = Array.make r false in
  List.iter
    (fun a ->
      let a' =
        axis_of ~by r a (fun ppf -> Format.fprintf ppf "a rank %d result" r)
      in
      if added.(a') then invalid_argf "%s: position %d repeats" by a;
      added.(a') <- true)
    l;
  (* [x]'s axes fill the positions not added, in order. *)
  let s' = Array.make r 1 in
  List.iteri
    (fun k i -> s'.(i) <- s.(k))
    (List.filter (fun i -> not added.(i)) (List.init r Fun.id));
  move ~by (Reshape s') x

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
  move ~by (Reshape s') x

let permute ~by p x = move ~by (Permute p) x

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
  move ~by (Slice (Array.mapi range s)) x

let sliding_window ?axis:(a = -1) ~window ?(step = 1) x =
  let by = "Nx.sliding_window" in
  let axis = axis ~by x a in
  let d = Prim.dim x axis in
  if window < 1 || step < 1 then
    invalid_argf "%s: window %d and step %d over %a; give at least 1" by window
      step pp_value x;
  if window > d then
    invalid_argf "%s: window %d exceeds axis %d of %a" by window a pp_value x;
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
      move ~by (Slice rs) x)

let assemble ~by dtype shape fill pieces =
  Eval.eval ~by (Value.Assemble { dtype; shape; fill; pieces })

(* The region of a value of shape [s] that keeps [count] elements from [start]
   along [axis], and the whole of every other axis. *)
let region s axis start count =
  let rs = Array.map whole s in
  rs.(axis) <- { start; count; step = 1 };
  rs

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
        axis_of ~by r a (fun ppf -> Format.fprintf ppf "a rank %d result" r)
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
      let pieces =
        List.mapi (fun k x -> (region unit a k 1, move ~by (Reshape unit) x)) xs
      in
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
    Array.mapi
      (fun i d : Nx_array.Move.range ->
        { start = fst widths.(i); count = d; step = 1 })
      s
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
  | None -> move ~by (Reshape (shape x)) rolled

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
    move ~by (Reshape merged)
      (move ~by (Broadcast wide) (move ~by (Reshape unit) x))

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
  let x = if r = k then x else move ~by (Reshape lead) x in
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
(* CR: Count a nonempty range by division before addition. On an axis
   of extent 3, Rs (0, 3, max_int) currently selects nothing through
   overflow. Use 1 + (distance - 1) / magnitude, handling min_int as
   a singleton reverse range before negating the step. *)
let range d start stop step : Nx_array.Move.range =
  let resolve e = if e < 0 then e + d else e in
  if step > 0 then
    let start = max 0 (min d (resolve start))
    and stop = max 0 (min d (resolve stop)) in
    { start; count = max 0 ((stop - start + step - 1) / step); step }
  else
    let start = max (-1) (min (d - 1) (resolve start))
    and stop = max (-1) (min (d - 1) (resolve stop)) in
    let count = max 0 ((start - stop - step - 1) / -step) in
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
         ( [| { Nx_array.Move.start = j; count = 1; step = 1 } |],
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

(* [y] read along [axis] at the positions [p]: [y]'s axis replaced by [p]'s
   axes. [p] runs along the axis, broadcast across the others, which a kernel
   reads as row takes. *)
let take_axis ~by axis p y =
  let s = shape y and ps = shape p in
  let n = numel p in
  let col = Array.mapi (fun i _ -> if i = axis then n else 1) s in
  let wide = Array.mapi (fun i d -> if i = axis then n else d) s in
  let idx = broadcast ~by wide (move ~by (Reshape col) p) in
  let g = Eval.eval ~by (Value.Gather { axis; idx; x = y }) in
  let s' =
    Array.concat
      [
        Array.sub s 0 axis;
        ps;
        Array.sub s (axis + 1) (Array.length s - axis - 1);
      ]
  in
  if Prim.has_shape g s' then g else move ~by (Reshape s') g

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
    else move ~by (Slice ranges) x
  in
  (* CR: Let slice's final reshape assemble all output axes. On a rank-32
     input, [I 0; T p] with rank-2 p reaches rank 33 before dropping I,
     although the result has rank 32. Factor the rank-preserving Gather
     out of take_axis and use it here for Rows, Held and From, keeping
     reverse order. take_axis can retain its finishing reshape for take. *)
  (* Gathers from the last axis back, so that an axis a gather replaces does not
     renumber the ones still to come. *)
  let y =
    List.fold_left
      (fun y (p, a) ->
        match (p, a) with
        | Rows ps, Some a -> take_axis ~by a (written ~by ps) y
        | Held p, Some a -> take_axis ~by a p y
        | From (start, n, d), Some a -> take_axis ~by a (window ~by start n d) y
        | (At _ | Span _ | Rows _ | Held _ | From _ | New), _ -> y)
      y (List.rev picks)
  in
  (* Each [At] axis now has extent 1 and goes; each [New] axis comes. *)
  let s' = selection x picks in
  if Prim.has_shape y s' then y else move ~by (Reshape s') y

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
    (Reshape (Array.sub s k (Array.length s - k)))
    (move ~by (Slice ranges) x)

let take ?axis p x =
  let by = "Nx.take" in
  match axis with
  | None -> take_axis ~by 0 p (move ~by (Reshape [| numel x |]) x)
  | Some a ->
      take_axis ~by (axis_of ~by (ndim x) a (fun ppf -> pp_value ppf x)) p x

let take_along_axis ~axis:a p x =
  let by = "Nx.take_along_axis" in
  let a = axis_of ~by (ndim x) a (fun ppf -> pp_value ppf x) in
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

(* Functional updates *)

(* The flat position in [x] of each element of the selection [picks], [-1] where
   a position held in data lies outside its axis: one map over the selection's
   shape, reading each position held in data broadcast to it. *)
(* CR: Split this target map when its inputs plus output exceed
   Prog.max_operands: sixteen scalar T indices on a [1; ...; 1] tensor
   already raise. Route all broadcast inputs together before splitting,
   and place them at that result, so placement is independent of chunks.
   Carry one partial flat address, keeping -1 sticky for an invalid index;
   count this carry and the output in each chunk's bound. Keep the
   selection layout, so storage scales with the selection. *)
let targets ~by x picks =
  let s = shape x in
  let r = Array.length s in
  let sel = selection x picks in
  let rs = Array.length sel in
  let strides = Array.make r 1 in
  for a = r - 2 downto 0 do
    strides.(a) <- strides.(a + 1) * s.(a + 1)
  done;
  let nodes = ref [] and count = ref 0 in
  let push n =
    nodes := n :: !nodes;
    incr count;
    !count - 1
  in
  let loads = ref [] in
  let load v =
    loads := Value.Plain v :: !loads;
    List.length !loads - 1
  in
  (* The selection's axis [j] as a coordinate. *)
  let coord j = push (P.Coord (rs - 1 - j)) in
  let sum = ref (push (const64 0)) and valid = ref None in
  let add a index =
    let k = push (const64 strides.(a)) in
    let m = push (P.Op2 (Binary Mul, index, k)) in
    sum := push (P.Op2 (Binary Add, !sum, m))
  in
  (* A position held in data over selection axes [j, j + k): its value at each
     index of the selection, and whether it lies in [0, d). *)
  let held j p d =
    let ps = shape p in
    let k = Array.length ps in
    let placed =
      Array.init rs (fun i -> if i >= j && i < j + k then ps.(i - j) else 1)
    in
    let v = broadcast ~by sel (move ~by (Reshape placed) p) in
    let i = push (P.In (load v)) in
    let lo = push (const64 0) and hi = push (const64 d) in
    let ge = push (P.Op2 (Compare Less_equal, lo, i)) in
    let lt = push (P.Op2 (Compare Less, i, hi)) in
    let ok = push (P.Op2 (Binary And, ge, lt)) in
    (valid :=
       match !valid with
       | None -> Some ok
       | Some v -> Some (push (P.Op2 (Binary And, v, ok))));
    i
  in
  let j = ref 0 in
  List.iter
    (fun (p, a) ->
      match (p, a) with
      | New, _ -> incr j
      | At i, Some a -> add a (push (const64 i))
      | Span g, Some a ->
          let c = coord !j in
          let st = push (const64 g.step) in
          let m = push (P.Op2 (Binary Mul, c, st)) in
          let o = push (const64 g.start) in
          add a (push (P.Op2 (Binary Add, o, m)));
          incr j
      | Rows ps, Some a ->
          add a (held !j (written ~by ps) s.(a));
          incr j
      | Held p, Some a ->
          add a (held !j p s.(a));
          j := !j + ndim p
      | From (start, n, d), Some a ->
          let st = push (P.In (load (broadcast ~by sel start))) in
          let lo = push (const64 0) and hi = push (const64 (d - n)) in
          let c =
            push
              (P.Op2 (Binary Minimum, push (P.Op2 (Binary Maximum, st, lo)), hi))
          in
          add a (push (P.Op2 (Binary Add, c, coord !j)));
          incr j
      | (At _ | Span _ | Rows _ | Held _ | From _), None -> ())
    picks;
  let used = List.length (List.filter (fun (_, a) -> a <> None) picks) in
  for a = used to r - 1 do
    add a (coord !j);
    incr j
  done;
  let out =
    match !valid with
    | None -> !sum
    | Some ok -> push (P.Op3 (Where, ok, !sum, push (const64 (-1))))
  in
  let loads = Array.of_list (List.rev !loads) in
  let prog =
    P.v
      ~ins:(Array.map (fun (Value.Plain v) -> D.Any (dtype v)) loads)
      (Array.of_list (List.rev !nodes))
      ~outs:[| out |]
  in
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
  (* CR: Classify picks by their constructors here. [positions] computes a
     window merely to discard it, then the selected branch reads its start
     again. A donated D start therefore raises, and an ordinary cache update
     computes its window twice. Build positions only in the chosen branch. *)
  match List.partition (fun (p, _) -> positions ~by p = None) picks with
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
        [ (Array.map whole s, x); (region, move ~by (Reshape counts) v) ]
  | ranges, [ ((data, Some a) as pick) ] when List.for_all whole_axis ranges ->
      (* One axis selected by positions, every other whole: a scatter along it,
         the positions broadcast across the others. *)
      let p = Option.get (positions ~by data) in
      let n = numel p in
      let wide = Array.mapi (fun i d -> if i = a then n else d) s in
      let col = Array.mapi (fun i _ -> if i = a then n else 1) s in
      let idx = broadcast ~by wide (move ~by (Reshape col) p) in
      let unique = match fst pick with Held _ -> false | _ -> true in
      Eval.eval ~by
        (Value.Scatter
           {
             combine = Set;
             unique;
             axis = a;
             idx;
             updates = move ~by (Reshape wide) v;
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
      move ~by (Reshape s) updated

type combine = Set | Add | Max | Min

let scatter ?(combine = Set) ~axis:a p u x =
  let by = "Nx.scatter" in
  let combine : Nx_kernel.Spec.combine =
    match combine with Set -> Set | Add -> Add | Max -> Max | Min -> Min
  in
  let axis = axis_of ~by (ndim x) a (fun ppf -> pp_value ppf x) in
  Eval.eval ~by
    (Value.Scatter
       { combine; unique = false; axis; idx = p; updates = u; into = x })

(* Reductions and scans *)

(* [axes] of a value of rank [r], a negative one counted from the end, sorted:
   every axis without [axes]. *)
let reduced_axes ~by r axes =
  match axes with
  | None -> Array.init r Fun.id
  | Some axes ->
      let axes =
        Array.of_list (List.map (fun a -> if a < 0 then a + r else a) axes)
      in
      Array.iter
        (fun a ->
          if a < 0 || a >= r then
            invalid_argf "%s: axis %d of a value of rank %d" by a r)
        axes;
      Array.sort Int.compare axes;
      Array.iteri
        (fun i a ->
          if i > 0 && axes.(i - 1) = a then
            invalid_argf "%s: axis %d repeats" by a)
        axes;
      axes

(* The one-operand reduction [m] of [x] along [axes], rounded to [dt]. *)
let fold (type v s w r d) ~by m (dt : (w, r) D.t) axes (x : (v, s, d) t) :
    (w, r, d) t =
  let prog = Prim.program (In 0) [| D.Any (dtype x) |] in
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
  else reshape (Array.mapi (fun a n -> if Array.mem a axes then 1 else n) s) y

let reduction ~by m ?axes ?(keepdims = false) x =
  let s = shape x in
  let axes = reduced_axes ~by (Array.length s) axes in
  kept ~keepdims s axes (fold ~by m (dtype x) axes x)

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
  let axes = reduced_axes ~by (Array.length s) axes in
  let n = Array.fold_left (fun n a -> n * s.(a)) 1 axes in
  let within (type w r) (acc : (w, r) D.t) (count : (w, r, d) t) =
    kept ~keepdims s axes (cast dt (div (fold ~by Sum acc axes x) count))
  in
  match D.kind dt with
  | D.Float when D.bits dt < 32 ->
      within D.Float32 (scalar D.Float32 (Float.of_int n))
  | D.Float -> within dt (scalar dt (Float.of_int n))
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
         prog = Prim.program (In 0) [| D.Any dt |];
         reduction = Monoid (m, 0, dt);
         loads = [| Plain x |];
       })

(* Along [axis], or over the elements in C order, keeping [x]'s shape. *)
let scan ~by m ?axis x =
  let s = shape x in
  let r = Array.length s in
  match axis with
  | None ->
      let n = Array.fold_left ( * ) 1 s in
      reshape s (scan_along ~by m 0 (reshape [| n |] x))
  | Some axis ->
      let a = if axis < 0 then axis + r else axis in
      if a < 0 || a >= r then
        invalid_argf "%s: axis %d of a value of rank %d" by axis r;
      scan_along ~by m a x

let cumsum ?axis x = scan ~by:"Nx.cumsum" Sum ?axis x
let cumprod ?axis x = scan ~by:"Nx.cumprod" Prod ?axis x
let cummax ?axis x = scan ~by:"Nx.cummax" Max ?axis x
let cummin ?axis x = scan ~by:"Nx.cummin" Min ?axis x

(* Contraction *)

let contract ?sizes ?acc ?init dt p a b =
  let acc = Option.map (fun dt -> D.Any dt) acc in
  Contraction.contract ~by:"Nx.contract" ?sizes ?acc ?init dt p a b

let einsum p a b = Contraction.contract ~by:"Nx.einsum" (dtype a) p a b
let matmul a b = Contraction.matmul ~by:"Nx.matmul" a b

(* Compositions *)

(* The map [body b ins] over [xs] broadcast together, its one result of [dt]:
   [ins] are [xs]'s nodes, in order. One pass over the operands. *)
let elementwise ~by dt xs body =
  let s =
    List.fold_left
      (fun s (Value.Any x) -> Prim.broadcast_shape ~by s (shape x))
      [||] xs
  in
  let load (Value.Any x) = Value.Plain (broadcast ~by s x) in
  let ins = Array.of_list (List.map (fun (Value.Any x) -> D.Any (dtype x)) xs) in
  let b = { nodes = []; count = 0 } in
  let out = body b (Array.mapi (fun i _ -> node b (In i)) ins) in
  let prog = P.v ~ins (Array.of_list (List.rev b.nodes)) ~outs:[| out |] in
  let layout = Nx_array.Layout.contiguous s in
  let y, () =
    Eval.eval ~by
      (Value.Map
         { layout; prog; outs = Value.[ dt ]; loads = Array.of_list (List.map load xs) })
  in
  y

let not_zero b dt x = node b (Op2 (Compare Not_equal, x, const b dt (D.zero dt)))

(* One where [k] of the operands' truths holds, zero elsewhere. *)
let logical ~by k a b' =
  let dt = dtype a in
  elementwise ~by dt [ Any a; Any b' ] (fun b ins ->
      let t = node b (Op2 (Binary k, not_zero b dt ins.(0), not_zero b dt ins.(1))) in
      node b (Op3 (Where, t, const b dt (D.one dt), const b dt (D.zero dt))))

let logical_and a b = logical ~by:"Nx.logical_and" And a b
let logical_or a b = logical ~by:"Nx.logical_or" Or a b
let logical_xor a b = logical ~by:"Nx.logical_xor" Xor a b

let logical_not x =
  let dt = dtype x in
  elementwise ~by:"Nx.logical_not" dt [ Any x ] (fun b ins ->
      let t = node b (Op2 (Compare Equal, ins.(0), const b dt (D.zero dt))) in
      node b (Op3 (Where, t, const b dt (D.one dt), const b dt (D.zero dt))))

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
let int_const b dt v =
  node b (Op1 (Cast, D.Any dt, const b D.Int64 v))

let bitwise_not x =
  let by = "Nx.bitwise_not" in
  let dt = dtype x in
  integer_or_boolean ~by dt;
  elementwise ~by dt [ Any x ] (fun b ins ->
      node b (Op2 (Binary Xor, ins.(0), int_const b dt (-1L))))

let lshift x n =
  let by = "Nx.lshift" in
  let dt = dtype x in
  integer ~by dt;
  if n < 0 then invalid_argf "%s: shift %d" by n;
  let factor = if n < 64 then Int64.shift_left 1L n else 0L in
  elementwise ~by dt [ Any x ] (fun b ins ->
      node b (Op2 (Binary Mul, ins.(0), int_const b dt factor)))

(* A signed integer divided by 2^n rounds toward negative infinity: the
   truncated quotient, less one where the remainder is negative. Past the
   sign bit only the sign is left. *)
let rshift x n =
  let by = "Nx.rshift" in
  let dt = dtype x in
  integer ~by dt;
  if n < 0 then invalid_argf "%s: shift %d" by n;
  let bits = D.bits dt in
  elementwise ~by dt [ Any x ] (fun b ins ->
      let x = ins.(0) in
      match D.kind dt with
      | D.Unsigned ->
          if n >= bits then int_const b dt 0L
          else node b (Op2 (Binary Idiv, x, int_const b dt (Int64.shift_left 1L n)))
      | _ when n >= bits - 1 ->
          let neg = node b (Op2 (Compare Less, x, int_const b dt 0L)) in
          node b (Op3 (Where, neg, int_const b dt (-1L), int_const b dt 0L))
      | _ ->
          let d = int_const b dt (Int64.shift_left 1L n) in
          let q = node b (Op2 (Binary Idiv, x, d)) in
          let r = node b (Op2 (Binary Mod, x, d)) in
          let neg = node b (Op2 (Compare Less, r, int_const b dt 0L)) in
          let one = node b (Op3 (Where, neg, int_const b dt 1L, int_const b dt 0L)) in
          node b (Op2 (Binary Sub, q, one)))

let clamp ?min:lo ?max:hi x =
  let by = "Nx.clamp" in
  if lo = None && hi = None then x
  else
    let dt = dtype x in
    elementwise ~by dt [ Any x ] (fun b ins ->
        let v =
          match lo with
          | None -> ins.(0)
          | Some lo -> node b (Op2 (Binary Maximum, ins.(0), node b (Const (D.Any dt, bits ~by dt lo))))
        in
        match hi with
        | None -> v
        | Some hi -> node b (Op2 (Binary Minimum, v, node b (Const (D.Any dt, bits ~by dt hi)))))

let isnan x = comparison ~by:"Nx.isnan" Not_equal x x

(* A complex value's real and imaginary parts, as views of its bits read as
   pairs of its parts' format. *)
type ('s, 'd) parts =
  | Parts : (float, 'r) D.t * (float, 'r, 'd) t * (float, 'r, 'd) t -> ('s, 'd) parts

let parts (type s d) ~by (z : (Complex.t, s, d) t) : (s, d) parts =
  let split (type r) (f : (float, r) D.t) =
    let pairs : (float, r, d) t = Eval.eval ~by (Value.Bitcast (f, z)) in
    let r = Prim.rank z in
    let part i =
      let one = Array.init (r + 1) (fun a ->
        if a = r then { Nx_array.Move.start = i; count = 1; step = 1 }
        else whole (Prim.dim z a))
      in
      move ~by (Reshape (shape z)) (move ~by (Slice one) pairs)
    in
    Parts (f, part 0, part 1)
  in
  match dtype z with D.Complex64 -> split D.Float32 | D.Complex128 -> split D.Float64

(* The format a float function of [dt] computes in: float32 for the floats
   narrower, [dt] itself otherwise. *)
type wide = Wide : (float, 's) D.t -> wide

let wide (type s) (dt : (float, s) D.t) =
  if D.bits dt < 32 then Wide D.Float32 else Wide dt

(* Where a float of [dt]'s format is an infinity: [false] in a format with
   none. A narrow float compares at float32, where a constant infinity is one:
   no store makes float8_e5m2's. *)
let infinite (type s) b (dt : (float, s) D.t) x =
  if not (D.float_format dt).infinities then const b D.Bool false
  else
    let (Wide w) = wide dt in
    let x = if D.bits dt < 32 then node b (Op1 (Cast, D.Any w, x)) else x in
    let a = node b (Op1 (Unary Abs, D.Any w, x)) in
    node b (Op2 (Compare Equal, a, const b w Float.infinity))

(* Where a float of [dt]'s format is neither an infinity nor a NaN. *)
let finite (type s) b (dt : (float, s) D.t) x =
  let not_nan = node b (Op2 (Compare Equal, x, x)) in
  if not (D.float_format dt).infinities then not_nan
  else
    let (Wide w) = wide dt in
    let x = if D.bits dt < 32 then node b (Op1 (Cast, D.Any w, x)) else x in
    let a = node b (Op1 (Unary Abs, D.Any w, x)) in
    node b (Op2 (Compare Less, a, const b w Float.infinity))

(* [x]'s float predicate [p], each part's joined by [k] for a complex [x], and
   [other] for the other dtypes, where [x] lies. *)
let predicate (type v s d) ~by (p : 'r. program -> (float, 'r) D.t -> int -> int)
    k other (x : (v, s, d) t) : d bool_t =
  match D.kind (dtype x) with
  | D.Float ->
      let dt = dtype x in
      elementwise ~by D.Bool [ Any x ] (fun b ins -> p b dt ins.(0))
  | D.Complex ->
      let (Parts (f, re, im)) = parts ~by x in
      elementwise ~by D.Bool [ Any re; Any im ] (fun b ins ->
          node b (Op2 (Binary k, p b f ins.(0), p b f ins.(1))))
  | D.Signed | D.Unsigned | D.Boolean ->
      beside ~by x (fill ~by D.Bool (shape x) other)

let isinf x = predicate ~by:"Nx.isinf" (fun b dt x -> infinite b dt x) Or false x

let isfinite x =
  predicate ~by:"Nx.isfinite" (fun b dt x -> finite b dt x) And true x

(* Float functions *)

(* The float function [f b w ins] of [xs], of one dtype, computed in its wide
   format [w] and rounded once to that dtype. *)
let float_map (type v s d) ~by (xs : (v, s, d) t list)
    (f : 'w. program -> (float, 'w) D.t -> int array -> int) : (v, s, d) t =
  let dt = dtype (List.hd xs) in
  match D.kind dt with
  | D.Float ->
      let (Wide w) = wide dt in
      let narrow = D.bits dt < 32 in
      elementwise ~by dt
        (List.map (fun x -> Value.Any x) xs)
        (fun b ins ->
          if not narrow then f b w ins
          else
            let ins = Array.map (fun i -> node b (Op1 (Cast, D.Any w, i))) ins in
            node b (Op1 (Cast, D.Any dt, f b w ins)))
  | D.Complex | D.Signed | D.Unsigned | D.Boolean ->
      invalid_argf "%s: %s is not a float dtype" by (D.name dt)

let un1 b k w x = node b (Op1 (Unary k, D.Any w, x))
let bin2 b k x y = node b (Op2 (Binary k, x, y))
let at_least b w x c = node b (Op2 (Compare Less_equal, const b w c, x))
let below b w x c = node b (Op2 (Compare Less, x, const b w c))
let pick b c x y = node b (Op3 (Where, c, x, y))

let square x = binary ~by:"Nx.square" (Binary Mul) x x

let rsqrt x =
  float_map ~by:"Nx.rsqrt" [ x ] (fun b w ins ->
      un1 b Recip w (un1 b Sqrt w ins.(0)))

(* [m sqrt (1 + (n / m)^2)] for the larger magnitude [m] and the smaller [n]:
   nothing squares past [m]. Zero for two zeros, an infinity where either is,
   even beside a NaN. *)
let hypot x y =
  float_map ~by:"Nx.hypot" [ x; y ] (fun b w ins ->
      let ax = un1 b Abs w ins.(0) and ay = un1 b Abs w ins.(1) in
      let m = bin2 b Maximum ax ay and n = bin2 b Minimum ax ay in
      let r = bin2 b Fdiv n m in
      let s = un1 b Sqrt w (node b (Op3 (Fma, r, r, const b w 1.))) in
      let h = bin2 b Mul m s in
      let zero = node b (Op2 (Compare Equal, m, const b w 0.)) in
      let h = pick b zero (const b w 0.) h in
      let inf v = node b (Op2 (Compare Equal, v, const b w Float.infinity)) in
      let either = bin2 b Or (inf ax) (inf ay) in
      pick b either (const b w Float.infinity) h)

(* Past [big] a square would overflow float32: [log (2 x)] is the function to
   the format's precision there. Below [tiny], [x] is. *)
let big = 0x1p28
let tiny = 0x1p-28

let asinh x =
  float_map ~by:"Nx.asinh" [ x ] (fun b w ins ->
      let x = ins.(0) in
      let a = un1 b Abs w x in
      let far = bin2 b Add (un1 b Log w a) (const b w (Float.log 2.)) in
      let a2 = bin2 b Mul a a in
      let root = un1 b Sqrt w (bin2 b Add (const b w 1.) a2) in
      let near =
        un1 b Log1p w
          (bin2 b Add a (bin2 b Fdiv a2 (bin2 b Add (const b w 1.) root)))
      in
      let r = pick b (at_least b w a big) far near in
      let signed = pick b (below b w x 0.) (un1 b Neg w r) r in
      pick b (below b w a tiny) x signed)

let acosh x =
  float_map ~by:"Nx.acosh" [ x ] (fun b w ins ->
      let x = ins.(0) in
      let far = bin2 b Add (un1 b Log w x) (const b w (Float.log 2.)) in
      let root = un1 b Sqrt w (node b (Op3 (Fma, x, x, const b w (-1.)))) in
      let mid =
        un1 b Log w
          (bin2 b Sub (bin2 b Add x x)
             (un1 b Recip w (bin2 b Add x root)))
      in
      let t = bin2 b Sub x (const b w 1.) in
      let near =
        un1 b Log1p w
          (bin2 b Add t
             (un1 b Sqrt w (node b (Op3 (Fma, t, t, bin2 b Add t t)))))
      in
      let r = pick b (at_least b w x 2.) mid near in
      let r = pick b (at_least b w x big) far r in
      pick b (below b w x 1.) (const b w Float.nan) r)

(* [log1p (2a / (1 - a)) / 2] of [a = |x|], the sign put back: a small [a]
   splits the quotient to keep its digits. *)
let atanh x =
  float_map ~by:"Nx.atanh" [ x ] (fun b w ins ->
      let x = ins.(0) in
      let a = un1 b Abs w x in
      let t = bin2 b Add a a in
      let one_minus = bin2 b Sub (const b w 1.) a in
      let small =
        bin2 b Add t (bin2 b Fdiv (bin2 b Mul t a) one_minus)
      in
      let large = bin2 b Fdiv t one_minus in
      let q = pick b (below b w a 0.5) small large in
      let r = bin2 b Mul (const b w 0.5) (un1 b Log1p w q) in
      let signed = pick b (below b w x 0.) (un1 b Neg w r) r in
      pick b (below b w a tiny) x signed)

(* Operations as data *)

module Prim = struct
  type ('v, 's, 'd) form = ('v, 's, 'd) Value.form = {
    dtype : ('v, 's) dtype;
    layout : Nx_array.Layout.t;
    placement : 'd Placement.t option;
  }

  type 'd any = 'd Value.any = Any : ('v, 's, 'd) t -> 'd any
  type 'd load = 'd Value.load = Plain : ('v, 's, 'd) t -> 'd load

  type ('d, 'a) reduction = ('d, 'a) Value.reduction =
    | Monoid :
        Nx_kernel.Spec.monoid * int * ('v, 's) dtype
        -> ('d, ('v, 's, 'd) Value.t) reduction
    | Moments :
        int * ('v, 's) dtype
        -> ('d, ('v, 's, 'd) Value.t * ('v, 's, 'd) Value.t) reduction
    | Arg :
        Nx_kernel.Spec.extreme * int * ('v, 's) dtype
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

  type 'r t = 'r Value.prim =
    | Map : {
        layout : Nx_array.Layout.t;
        prog : Nx_kernel.Prog.t;
        outs : ('d, 'r) outs;
        loads : 'd load array;
      }
        -> 'r t
    | Reduce : {
        layout : Nx_array.Layout.t;
        axes : int array;
        prog : Nx_kernel.Prog.t;
        reductions : ('d, 'r) reductions;
        loads : 'd load array;
      }
        -> 'r t
    | Scan : {
        layout : Nx_array.Layout.t;
        axis : int;
        prog : Nx_kernel.Prog.t;
        reduction : ('d, 'r) reduction;
        loads : 'd load array;
      }
        -> 'r t
    | Gather : {
        axis : int;
        idx : (int64, Dtype.int64_elt, 'd) Value.t;
        x : ('v, 's, 'd) Value.t;
      }
        -> ('v, 's, 'd) Value.t t
    | Scatter : {
        combine : Nx_kernel.Spec.combine;
        unique : bool;
        axis : int;
        idx : (int64, Dtype.int64_elt, 'd) Value.t;
        updates : ('v, 's, 'd) Value.t;
        into : ('v, 's, 'd) Value.t;
      }
        -> ('v, 's, 'd) Value.t t
    | Assemble : {
        dtype : ('v, 's) dtype;
        shape : int array;
        fill : 'v;
        pieces : (Nx_array.Move.range array * ('v, 's, 'd) Value.t) list;
      }
        -> ('v, 's, 'd) Value.t t
    | Contract : {
        spec : Nx_kernel.Spec.contract Nx_kernel.Spec.t;
        out : ('v, 's) dtype;
        a : ('a, 'b, 'd) Value.t;
        b : ('c, 'e, 'd) Value.t;
        init : ('v, 's, 'd) Value.t option;
      }
        -> ('v, 's, 'd) Value.t t
    | Copy : ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t t
    | Move : Nx_array.Move.t * ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t t
    | Bitcast : ('w, 'r) dtype * ('v, 's, 'd) Value.t -> ('w, 'r, 'd) Value.t t
    | Place : 'e Placement.t * ('v, 's, 'd) Value.t -> ('v, 's, 'e) Value.t t
    | Check : {
        ok : (bool, Dtype.bool_elt, 'd) Value.t;
        data : 'd any list;
        fail : int array -> 'd any list -> exn;
      }
        -> unit t

  type operands = Prim.operands = Operands : 'd any list -> operands

  let name = Prim.name
  let pp = Prim.pp
  let operands = Prim.operands
  let map = Prim.map
  let form = Prim.form
  let results = Prim.results

  type interpretation = Value.interpretation
  type reach = Value.reach = Values | Extent
  type ('v, 's, +'d) payload = ('v, 's, 'd) Value.payload = ..

  let interpret = Interp.interpret
  let traced = Interp.traced
  let payload = Interp.payload
  let owner = Interp.owner
  let later = Interp.later
  let eval = Eval.eval
  let expand = Eval.expand
end
