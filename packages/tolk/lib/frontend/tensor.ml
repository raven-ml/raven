(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop
module D = Dtype

type t = { mutable uop : U.t }

(* Live-tensor registry. Every tensor is recorded in a weak array at
   construction so that a graph rewrite (an in-place assignment, or
   realization rebinding computed nodes onto their buffers) can repoint every
   live handle whose graph contains a rewritten node. Weak slots let the GC
   collect tensors as usual; dead slots are compacted when the array fills. *)
let live = ref (Weak.create 1024)
let live_n = ref 0

let rec register t =
  let cap = Weak.length !live in
  if !live_n < cap then begin
    Weak.set !live !live_n (Some t);
    incr live_n
  end
  else begin
    let kept = ref [] and n = ref 0 in
    for i = cap - 1 downto 0 do
      match Weak.get !live i with
      | Some x ->
          kept := x :: !kept;
          incr n
      | None -> ()
    done;
    (* Grow only when compaction leaves the array mostly full. *)
    let cap' = if 2 * !n >= cap then 2 * cap else cap in
    let fresh = Weak.create cap' in
    List.iteri (fun i x -> Weak.set fresh i (Some x)) !kept;
    live := fresh;
    live_n := !n;
    register t
  end

let live_tensors () =
  let acc = ref [] in
  for i = !live_n - 1 downto 0 do
    match Weak.get !live i with
    | Some t -> acc := t :: !acc
    | None -> ()
  done;
  !acc

let of_uop uop =
  let t = { uop } in
  register t;
  t

let uop t = t.uop
let set_uop t uop = t.uop <- uop

let apply_map mappings =
  if mappings <> [] then begin
    match live_tensors () with
    | [] -> ()
    | ts ->
        let sink = U.sink (List.map (fun t -> t.uop) ts) in
        let sink' = U.substitute ~walk:true mappings sink in
        if sink' != sink then
          List.iteri
            (fun i t ->
              let u = (U.src sink').(i) in
              if t.uop != u then t.uop <- u)
            ts
  end

let dtype t = U.dtype t.uop

let val_dtype t =
  match U.dtype t.uop with
  | D.Val v -> v
  | D.Ptr _ -> invalid_arg "Tensor.val_dtype: pointer dtype"

let device t = U.device_of t.uop
let symbolic_shape t = U.shape t.uop

let shape t =
  List.map
    (fun d ->
      match U.const_int_value d with
      | Some n -> n
      | None -> invalid_arg "Tensor.shape: symbolic dimension")
    (symbolic_shape t)

let ndim t = List.length (symbolic_shape t)
let numel t = List.fold_left ( * ) 1 (shape t)

let resolve_dim ?(extra = false) t dim =
  let total = ndim t + if extra then 1 else 0 in
  let lo = -max 1 total and hi = max 1 total - 1 in
  if dim < lo || dim > hi then
    invalid_arg
      (Printf.sprintf "Tensor.resolve_dim: dim %d out of range [%d, %d]" dim lo
         hi);
  if dim < 0 then dim + total else dim

type scalar = Sint of int | Sfloat of float | Sbool of bool

let scalar_const dt s =
  if D.Val.is_bool dt then
    Const.bool
      (match s with Sbool v -> v | Sint n -> n <> 0 | Sfloat x -> x <> 0.)
  else if D.Val.is_float dt then
    Const.float dt
      (match s with Sfloat x -> x | Sint n -> float_of_int n | Sbool v -> if v then 1. else 0.)
  else
    Const.int dt
      (match s with Sint n -> n | Sfloat x -> int_of_float x | Sbool v -> if v then 1 else 0)

let i n = of_uop (U.const (Const.int D.Val.default_int n))
let f x = of_uop (U.const (Const.float D.Val.default_float x))
let b v = of_uop (U.const (Const.bool v))
let int_ n = U.const (Const.int D.Val.weakint n)

let shape_uop dims =
  match List.map int_ dims with [ d ] -> d | ds -> U.stack ds

let alu_unary op t = of_uop (U.alu_unary ~op ~src:t.uop)
let alu_binary op a b = of_uop (U.alu_binary ~op ~lhs:a.uop ~rhs:b.uop)
let alu_ternary op a b c = of_uop (U.alu_ternary ~op ~a:a.uop ~b:b.uop ~c:c.uop)

let broadcast_shape shapes =
  let max_dim = List.fold_left (fun a s -> max a (List.length s)) 0 shapes in
  let aligned =
    List.map
      (fun s -> List.init (max_dim - List.length s) (fun _ -> 1) @ s)
      shapes
  in
  List.init max_dim (fun idx ->
      let col = List.map (fun s -> List.nth s idx) aligned in
      let dim = if List.mem 0 col then 0 else List.fold_left max 1 col in
      List.iter
        (fun s ->
          if not (s = dim || s = 1) then
            invalid_arg "Tensor.broadcast_shape: incompatible shapes")
        col;
      dim)

let broadcasted_hook : (reverse:bool -> t -> t -> t * t) ref =
  ref (fun ~reverse:_ _ _ ->
      failwith "Tensor.broadcasted: the Op module must be linked to install it")

let broadcasted ?(reverse = false) a b = !broadcasted_hook ~reverse a b
