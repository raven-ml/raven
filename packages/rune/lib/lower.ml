(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
module D = Nx_array.Dtype
module P = Nx.Prim
module Arith = Lower_arith
module Reduce = Lower_reduce

let unlowered what =
  invalid_arg (Printf.sprintf "Rune.Lower.op: %s is not lowered" what)

(* Padded loads

   A padded load spaces each axis's elements by its interior padding, pads or
   crops its ends, then takes its windows: movements that pad with the fill, a
   constant of its bits, which tolk reads as loads gated by their validity. A
   complex value pads each part with the fill's part. *)

(* [interior ~value u k] is [u] with [k.(i)] elements of [value] between
   neighbours along each axis [i]: each element followed by its [k], the last's
   cropped. *)
let interior ~value u k =
  let pad1 u i k =
    let s = Shape.shape u in
    match List.nth s i with
    | Ops.Int n when k > 0 && n > 0 ->
        let r = List.length s in
        let spread =
          Shape.reshape u
            (List.concat
               (List.mapi
                  (fun j d -> if j = i then [ d; Ops.Int 1 ] else [ d ])
                  s))
        in
        let spread =
          Shape.pad ~value spread
            (List.init (r + 1) (fun j ->
                 if j = i + 1 then Some (Ops.Int 0, Ops.Int k) else None))
        in
        let flat =
          Shape.reshape spread
            (List.mapi
               (fun j d -> if j = i then Ops.Int (n * (k + 1)) else d)
               s)
        in
        Shape.shrink flat
          (List.init r (fun j ->
               if j = i then Some (Ops.Int 0, Ops.Int (n + ((n - 1) * k)))
               else None))
    | _ -> u
  in
  let u = ref u in
  Iarray.iteri (fun i k -> u := pad1 !u i k) k;
  !u

(* [ends ~value u lo hi] pads each axis [i] with [lo.(i)] elements of [value]
   before and [hi.(i)] after, cropping where negative. It pads before it crops:
   a crop may reach into the other end's padding, as [lo = -4], [hi = 2] on an
   axis of [3]. *)
let ends ~value u lo hi =
  let pad =
    List.init (Shape.ndim u) (fun i ->
        Some
          ( Ops.Int (Int.max 0 (Iarray.get lo i)),
            Ops.Int (Int.max 0 (Iarray.get hi i)) ))
  in
  let u = Shape.pad ~value u pad in
  Shape.shrink u
    (List.init (Shape.ndim u) (fun i ->
         let n = Lower_bits.extent u i in
         Some
           ( Ops.Int (Int.max 0 (-Iarray.get lo i)),
             Ops.Int (n - Int.max 0 (-Iarray.get hi i)) )))

(* [window u w] is [u]'s windows along [w.axis] in its place, their elements
   along a new last axis. A window longer than the axis gives none, where tolk's
   pool refuses it. *)
let window u (w : Nx_array.Move.window) =
  let r = Shape.ndim u in
  if ((w.size - 1) * w.dilation) + 1 > Lower_bits.extent u w.axis then
    let s = Shape.shape u in
    let none = List.mapi (fun j d -> if j = w.axis then Ops.Int 0 else d) s in
    let empty =
      Shape.shrink u
        (List.init r (fun j ->
             if j = w.axis then Some (Ops.Int 0, Ops.Int 0) else None))
    in
    Shape.expand
      (Shape.reshape empty (none @ [ Ops.Int 1 ]))
      (none @ [ Ops.Int w.size ])
  else
    let last = List.filter (( <> ) w.axis) (List.init r Fun.id) @ [ w.axis ] in
    let pooled =
      Shape.pool ~stride:[ w.step ] ~dilation:[ w.dilation ]
        (Shape.permute u last) [ w.size ]
    in
    Shape.permute pooled
      (List.init r (fun a ->
           if a = w.axis then r - 1 else if a < w.axis then a else a - 1)
      @ [ r ])

let padded (type v s) (dt : (v, s) D.t) (fill : v) (pad : Nx_kernel.pad) u =
  let moved u value =
    let u = ends ~value (interior ~value u pad.interior) pad.lo pad.hi in
    Iarray.fold_left window u pad.windows
  in
  let d = D.Any dt in
  Arith.of_parts d (List.map2 moved (Arith.parts d u) (Arith.fill dt fill))

let load (node : 'v 's 'e. ('v, 's, 'e) Nx.t -> Ops.t) : 'd P.load -> Ops.t =
  function
  | Plain x -> node x
  | Padded { x; fill; pad } -> padded (Nx.dtype x) fill pad (node x)

(* Loops *)

let shape layout = Iarray.to_array (Nx_array.Layout.shape layout)

let body (node : 'v 's 'e. ('v, 's, 'e) Nx.t -> Ops.t) layout prog loads =
  Arith.prog prog (shape layout)
    (Iarray.map (load node) loads |> Iarray.to_array)

(* The dtype of the program's output [k]. *)
let output prog k =
  Nx_kernel.Prog.dtype prog (Iarray.get prog.Nx_kernel.Prog.outs k)

let reduce (node : 'v 's 'e. ('v, 's, 'e) Nx.t -> Ops.t) layout axes prog rs
    loads =
  let ys = body node layout prog loads in
  let axes = Iarray.to_list axes in
  let fold (r, k, dt) = Reduce.reduce r axes ~src:(output prog k) dt ys.(k) in
  Array.of_list (List.concat_map fold (Array.to_list (P.reductions rs)))

let scan (node : 'v 's 'e. ('v, 's, 'e) Nx.t -> Ops.t) layout axis prog r loads
    carry =
  let r, k, dt = P.reduction r in
  let ys = body node layout prog loads in
  let carry = Option.map (fun (P.Any c) -> node c) carry in
  Array.of_list (Reduce.scan r axis ~carry ~src:(output prog k) dt ys.(k))

(* Movements

   A movement is tolk's movement of each part. A slice keeps every [step]th
   element of a run: the run padded to a whole number of steps, split into
   [count] rows of [step] and the first column kept; a negative step flips the
   axis first. *)

let sints s = List.map (fun n -> Ops.Int n) s

let slice1 u i ({ start; count; step } : Nx_array.Move.range) =
  let r = Shape.ndim u in
  let only f = List.init r (fun j -> if j = i then Some (f ()) else None) in
  let n = Lower_bits.extent u i in
  let u, start, step =
    if step < 0 then (Shape.flip u [ i ], n - 1 - start, -step)
    else (u, start, step)
  in
  (* An empty range's start may lie outside an empty axis. *)
  if count = 0 then Shape.shrink u (only (fun () -> (Ops.Int 0, Ops.Int 0)))
  else if step = 1 then
    Shape.shrink u (only (fun () -> (Ops.Int start, Ops.Int (start + count))))
  else
    let stop = Int.min n (start + (count * step)) in
    let run = Shape.shrink u (only (fun () -> (Ops.Int start, Ops.Int stop))) in
    let run =
      Shape.pad run
        (only (fun () -> (Ops.Int 0, Ops.Int ((count * step) - (stop - start)))))
    in
    let s = Shape.shape run in
    let rows =
      Shape.reshape run
        (List.concat
           (List.mapi
              (fun j d ->
                if j = i then [ Ops.Int count; Ops.Int step ] else [ d ])
              s))
    in
    let kept =
      Shape.shrink rows
        (List.init (r + 1) (fun j ->
             if j = i + 1 then Some (Ops.Int 0, Ops.Int 1) else None))
    in
    Shape.reshape kept
      (List.mapi (fun j d -> if j = i then Ops.Int count else d) s)

let movement (m : Nx_array.Move.t) u =
  match m with
  | Reshape s -> Shape.reshape u (sints (Iarray.to_list s))
  | Broadcast s ->
      let s = Iarray.to_list s in
      let ones =
        List.init (List.length s - Shape.ndim u) (fun _ -> Ops.Int 1)
      in
      Shape.expand (Shape.reshape u (ones @ Shape.shape u)) (sints s)
  | Permute p -> Shape.permute u (Iarray.to_list p)
  | Slice rs ->
      let u = ref u in
      Iarray.iteri (fun i r -> u := slice1 !u i r) rs;
      !u
  | Window ws -> Iarray.fold_left window u ws

let move m x u =
  let d = D.Any (Nx.dtype x) in
  Arith.of_parts d (List.map (movement m) (Arith.parts d u))

let bitcast dt x u =
  Arith.bitcast ~src:(D.Any (Nx.dtype x)) (D.Any dt) (Nx.shape x) u

(* Transforms: a result of another precision than its operand's is not
   lowered. *)

let fft (node : 'v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) (type d r)
    (f : (d, r) P.fft) =
  let paired dt u =
    if Arith.dtype (D.Any dt) <> Ops.dtype u then
      unlowered "an Fft of another precision";
    u
  in
  match f with
  | C2c { direction; axes; x } ->
      Lower_fft.c2c direction ~axes:(Iarray.to_list axes) (node x)
  | R2c { dtype; axes; x } ->
      Lower_fft.r2c ~axes:(Iarray.to_list axes) (paired dtype (node x))
  | C2r { dtype; n; axes; x } ->
      Lower_fft.c2r ~axes:(Iarray.to_list axes) ~n (paired dtype (node x))

(* Operations *)

let draws (prog : Nx_kernel.Prog.t) =
  Iarray.exists
    (function Nx_kernel.Prog.Op2 (Binary Threefry, _, _) -> true | _ -> false)
    prog.nodes

let op ?dst ~device ~host (node : 'v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) (type r)
    (o : r P.t) =
  match o with
  | Map { layout; prog; loads; outs = _; at = _ } -> body node layout prog loads
  | Reduce { layout; axes; prog; reductions = rs; loads } ->
      reduce node layout axes prog rs loads
  | Scan { layout; axis; prog; reduction = r; loads; carry } ->
      scan node layout axis prog r loads carry
  | Contract
      { batch; contracting; acc; out; a; b; init; a_scale; b_scale; segments }
    ->
      let scale : _ P.scale -> Reduce.scale = function
        | Exponent c -> Exponent (node c)
        | Value v -> Value (D.Any (Nx.dtype v), node v)
      in
      let operand x s : Reduce.operand =
        let scale = Option.map scale s in
        { dtype = D.Any (Nx.dtype x); node = node x; scale }
      in
      let segments : Ops.t Nx_kernel.segments option =
        match segments with
        | None -> None
        | Some (Rows r) -> Some (Rows { r with offsets = node r.offsets })
        | Some (Contracting c) ->
            Some (Contracting { c with offsets = node c.offsets })
      in
      [|
        Reduce.contract ~batch ~contracting ~segments ~acc ~out:(D.Any out)
          (operand a a_scale) (operand b b_scale) (Option.map node init);
      |]
  | Copy x -> [| node x |]
  | Move (m, x) -> [| move m x (node x) |]
  | Bitcast (dt, x) -> [| bitcast dt x (node x) |]
  | Place (_, x) -> [| node x |]
  | Gather { axis; idx; x } ->
      [| Lower_index.gather (D.Any (Nx.dtype x)) ~axis (node idx) (node x) |]
  | Scatter { combine; unique = _; axis; idx; updates; into } ->
      [|
        Lower_index.scatter ?dst ~host device
          (D.Any (Nx.dtype into))
          combine ~axis ~idx:(node idx) ~updates:(node updates) (node into);
      |]
  | Fold { shape; pad; x } ->
      [| Lower_index.fold (D.Any (Nx.dtype x)) pad shape (node x) |]
  | Assemble { dtype; shape; fill; pieces } ->
      let pieces = List.map (fun (r, p) -> (r, node p)) pieces in
      [| Lower_index.assemble dtype fill shape pieces |]
  | Sort { axis; descending; k; x } ->
      let values, positions =
        Lower_index.sort (D.Any (Nx.dtype x)) ~axis ~descending ~k (node x)
      in
      [| values; positions |]
  | Linalg l -> Lower_linalg.op ~device node l
  | Fft f -> [| fft node f |]
  | Check _ ->
      unlowered (P.name o)
