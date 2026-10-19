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
   crops its ends, then takes its windows: movements, which tolk reads as loads
   gated by their validity, the fill where they read nothing. A complex value
   pads each part with its part of the fill. *)

(* [interior u k] is [u] with [k.(i)] fill elements between neighbours along
   each axis [i]: each element followed by its [k], the last's cropped. *)
let interior c u k =
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
          Shape.pad ~value:c spread
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

(* [ends c u lo hi] pads each axis [i] with [lo.(i)] elements before and
   [hi.(i)] after, cropping where negative. *)
let ends c u lo hi =
  let crop =
    List.mapi
      (fun i d ->
        let n = match d with Ops.Int n -> n | Ops.Sym _ -> assert false in
        let l = Int.max 0 (-Iarray.get lo i)
        and h = Int.max 0 (-Iarray.get hi i) in
        Some (Ops.Int l, Ops.Int (n - h)))
      (Shape.shape u)
  in
  let pad =
    List.init (Shape.ndim u) (fun i ->
        Some
          ( Ops.Int (Int.max 0 (Iarray.get lo i)),
            Ops.Int (Int.max 0 (Iarray.get hi i)) ))
  in
  Shape.pad ~value:c (Shape.shrink u crop) pad

(* [window u w] is [u]'s windows along [w.axis] in its place, their elements
   along a new last axis. *)
let window u (w : Nx_array.Move.window) =
  let r = Shape.ndim u in
  let last = List.filter (( <> ) w.axis) (List.init r Fun.id) @ [ w.axis ] in
  let pooled =
    Shape.pool ~stride:[ w.step ] ~dilation:[ w.dilation ]
      (Shape.permute u last) [ w.size ]
  in
  Shape.permute pooled
    (List.init r (fun a ->
         if a = w.axis then r - 1 else if a < w.axis then a else a - 1)
    @ [ r ])

(* A float part pads its bits: tolk pads a value with zeros and adds the fill,
   which turns a [-0.] fill into [+0.]. *)
let as_bits (c : Dtype.const) u =
  let t = Ops.dtype u in
  if not (Dtype.is_float t) then (c, u, Fun.id)
  else
    let w =
      List.find (fun d -> Dtype.itemsize d = Dtype.itemsize t) Dtype.uints
    in
    let c =
      match c with
      | #Dtype.value as v -> (Dtype.bitcast t w v :> Dtype.const)
      | `Invalid -> c
    in
    (c, Ops.bitcast u w, fun u -> Ops.bitcast u t)

let padded (type v s) (dt : (v, s) D.t) (fill : v) (pad : Nx_kernel.pad) u =
  let part c u =
    let c, u, back = as_bits c u in
    let u = ends c (interior c u pad.interior) pad.lo pad.hi in
    back (Iarray.fold_left window u pad.windows)
  in
  let d = D.Any dt in
  Arith.of_parts d (List.map2 part (Arith.fill dt fill) (Arith.parts d u))

let load (node : 'v 's 'e. ('v, 's, 'e) Nx.t -> Ops.t) : 'd P.load -> Ops.t =
  function
  | Plain x -> node x
  | Padded { x; fill; pad } -> padded (Nx.dtype x) fill pad (node x)

(* Loops *)

let shape layout = Nx_array.Layout.shape layout

let body (node : 'v 's 'e. ('v, 's, 'e) Nx.t -> Ops.t) layout prog loads =
  Arith.prog prog (shape layout)
    (Iarray.map (load node) loads |> Iarray.to_array)

(* The dtype of the program's output [k]. *)
let output prog k =
  Nx_kernel.Prog.dtype prog (Iarray.get prog.Nx_kernel.Prog.outs k)

let monoid (m : Nx_kernel.monoid) : Reduce.monoid option =
  match m with
  | Sum -> Some Sum
  | Prod -> Some Prod
  | Max -> Some Max
  | Min -> Some Min
  | Logsumexp -> None

(* A complex term has no order and no product tolk folds. *)
let direct m src =
  match (m, src) with
  | (Reduce.Prod | Max | Min), D.Any dt when D.is D.Complex dt ->
      unlowered "a complex product or extreme"
  | _ -> ()

(* [reduction r] is the monoid and the output a reduction folds where a graph
   folds it, and its dtype. *)
let reduction (type d a) (r : (d, a) P.reduction) =
  match r with
  | Monoid (m, k, dt) -> Option.map (fun m -> (m, k, D.Any dt)) (monoid m)
  | Moments _ | Arg _ -> None

let rec reductions : type d r.
    (d, r) P.reductions -> (Reduce.monoid * int * D.any) list option = function
  | [] -> Some []
  | r :: rest -> (
      match (reduction r, reductions rest) with
      | Some m, Some ms -> Some (m :: ms)
      | _ -> None)

let reduce (node : 'v 's 'e. ('v, 's, 'e) Nx.t -> Ops.t) layout axes prog rs
    loads =
  match reductions rs with
  | None -> None
  | Some rs ->
      let ys = body node layout prog loads in
      let axes = Iarray.to_list axes in
      let fold (m, k, dt) =
        let src = output prog k in
        direct m src;
        Reduce.reduce m axes ~src dt ys.(k)
      in
      Some (Array.of_list (List.map fold rs))

let scan (node : 'v 's 'e. ('v, 's, 'e) Nx.t -> Ops.t) layout axis prog r loads
    carry =
  match reduction r with
  | None -> None
  | Some (m, k, dt) ->
      let ys = body node layout prog loads in
      let src = output prog k in
      direct m src;
      let carry = Option.map (fun (P.Any c) -> node c) carry in
      Some [| Reduce.scan m axis ~carry ~src dt ys.(k) |]

(* Movements

   A movement is tolk's movement of each part. A slice keeps every [step]th
   element of a run: the run padded to a whole number of steps, split into
   [count] rows of [step] and the first column kept; a negative step flips the
   axis first. *)

let sints s = List.map (fun n -> Ops.Int n) s

let extent u i =
  match List.nth (Shape.shape u) i with
  | Ops.Int n -> n
  | Ops.Sym _ -> invalid_arg "Rune.Lower: a symbolic extent"

let slice1 u i ({ start; count; step } : Nx_array.Move.range) =
  let r = Shape.ndim u in
  let only f = List.init r (fun j -> if j = i then Some (f ()) else None) in
  let n = extent u i in
  let u, start, step =
    if step < 0 then (Shape.flip u [ i ], n - 1 - start, -step)
    else (u, start, step)
  in
  if count = 0 || step = 1 then
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

(* A bitcast of one width is the program of its one kind. *)
let bitcast dt x u =
  let src = Nx.dtype x in
  if D.bits src <> D.bits dt then unlowered "a Bitcast that changes width";
  let prog =
    Nx_kernel.Prog.of_node ~ins:[| D.Any src |] (Op1 (Bitcast, D.Any dt, 0))
  in
  (Arith.prog prog (Nx.shape x) [| u |]).(0)

(* Operations *)

let op (node : 'v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) (type r) (o : r P.t) =
  match o with
  | Map { layout; prog; loads; outs = _ } -> Some (body node layout prog loads)
  | Reduce { layout; axes; prog; reductions = rs; loads } ->
      reduce node layout axes prog rs loads
  | Scan { layout; axis; prog; reduction = r; loads; carry } ->
      scan node layout axis prog r loads carry
  | Contract { batch; contracting; acc; out; a; b; init; order = _ } ->
      let operand x = (D.Any (Nx.dtype x), node x) in
      Some
        [|
          Reduce.contract ~batch ~contracting ~acc ~out:(D.Any out) (operand a)
            (operand b) (Option.map node init);
        |]
  | Copy x -> Some [| node x |]
  | Move (m, x) -> Some [| move m x (node x) |]
  | Bitcast (dt, x) -> Some [| bitcast dt x (node x) |]
  | Gather _ | Scatter _ | Sort _ | Fold _ | Assemble _ | Fft _ | Linalg _
  | Place _ | Check _ ->
      unlowered (P.name o)
