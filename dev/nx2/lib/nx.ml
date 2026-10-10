(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('v, 's, 'd) t = ('v, 's, 'd) Value.t
type ('v, 's) dtype = ('v, 's) Nx_array.Dtype.t

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

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let bits ~by dt v =
  match P.bits dt v with
  | b -> b
  | exception Invalid_argument e -> invalid_argf "%s: %s" by e

let fill ~by dt shape v =
  let prog = Prim.program (Const (D.Any dt, bits ~by dt v)) [||] in
  let layout =
    match Nx_array.Layout.contiguous shape with
    | l -> l
    | exception Invalid_argument e -> invalid_argf "%s: %s" by e
  in
  let x, () =
    Eval.eval ~by
      (Value.Map { layout; prog; outs = Value.[ dt ]; loads = [||] })
  in
  x

let zeros dt shape = fill ~by:"Nx.zeros" dt shape (D.zero dt)
let scalar dt v = fill ~by:"Nx.scalar" dt [||] v

(* Zeros of [x]'s dtype and shape filled where [x] lies: [x]'s elements are
   never read. *)
let zeros_like x =
  let by = "Nx.zeros_like" in
  let dt = dtype x in
  let z = fill ~by dt (shape x) (D.zero dt) in
  match Prim.at x with
  | None -> z
  | Some p -> Eval.eval ~by (Value.Place (p, z))

let broadcast ~by s x =
  if Prim.has_shape x s then x else Eval.eval ~by (Value.Move (Broadcast s, x))

let same_shape = Prim.same_shape

let binary ~by k a b =
  if same_shape a b then Eval.apply2 ~by k (dtype a) a b
  else
    let s = Prim.broadcast_shape ~by (shape a) (shape b) in
    Eval.apply2 ~by k (dtype a) (broadcast ~by s a) (broadcast ~by s b)

let add a b = binary ~by:"Nx.add" (Binary Add) a b
let mul a b = binary ~by:"Nx.mul" (Binary Mul) a b

let less a b =
  let by = "Nx.less" in
  if same_shape a b then Eval.apply2 ~by (Compare Less) D.Bool a b
  else
    let s = Prim.broadcast_shape ~by (shape a) (shape b) in
    Eval.apply2 ~by (Compare Less) D.Bool (broadcast ~by s a)
      (broadcast ~by s b)

let where c x y =
  let by = "Nx.where" in
  if same_shape c x && same_shape x y then Eval.apply3 ~by Where c x y
  else
    let s =
      Prim.broadcast_shape ~by
        (Prim.broadcast_shape ~by (shape c) (shape x))
        (shape y)
    in
    Eval.apply3 ~by Where (broadcast ~by s c) (broadcast ~by s x)
      (broadcast ~by s y)

let cast (type v s w r d) (dt : (w, r) D.t) (x : (v, s, d) t) : (w, r, d) t =
  match D.equal_witness (dtype x) dt with
  | Some Type.Equal -> x
  | None -> Eval.apply1 ~by:"Nx.cast" Cast dt x

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
  let step (s, before) s' =
    if Array.exists (fun e -> e < 0) s' then
      invalid_argf "%s: %a has a negative extent" by pp_shape s';
    match Prim.merge s s' with
    | Ok s -> (s, before @ [ s' ])
    | Error (a, e, e') ->
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

(* Operations as data *)

module Prim = struct
  type ('v, 's, 'd) form = ('v, 's, 'd) Value.form = {
    dtype : ('v, 's) dtype;
    layout : Nx_array.Layout.t;
    placement : 'd Placement.t option;
  }

  type 'd any = 'd Value.any = Any : ('v, 's, 'd) t -> 'd any
  type 'd load = 'd Value.load = Plain : ('v, 's, 'd) t -> 'd load

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
