(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module M = Nx_array.Move

(* The outputs of a map whose dtypes are known at run time, and how to read its
   results as values. *)
type 'd outputs =
  | Outputs : ('d, 'r) Value.outs * ('r -> 'd Value.any list) -> 'd outputs

let rec outputs : type d. D.any list -> d outputs = function
  | [] -> Outputs (Value.[], fun () -> [])
  | D.Any dt :: dts ->
      let (Outputs (outs, read)) = outputs dts in
      Outputs (Value.( :: ) (dt, outs), fun (v, r) -> Value.Any v :: read r)

(* The map of [prog] over [loads], of [shape], into [dts]. *)
let map_n (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    prog dts (loads : d Value.load iarray) : d Value.any array =
  let (Outputs (outs, read)) = outputs (Array.to_list dts) in
  let layout = L.contiguous shape in
  Array.of_list (read (apply ~by (Value.Map { layout; prog; outs; loads })))

(* The ranges of [shape] that keep its elements [start + j step], [j < count],
   along [axis] and all others elsewhere. *)
let along_axis shape axis start count step =
  let keep a e =
    if a = axis then { M.start; count; step }
    else { M.start = 0; count = e; step = 1 }
  in
  Iarray.init (Array.length shape) (fun a -> keep a shape.(a))

(* [x]'s elements [start + j step], [j < count], along [axis]. *)
let part (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by axis
    start count step (Value.Any x : d Value.any) : d Value.any =
  let ranges = along_axis (Prim.shape x) axis start count step in
  Value.Any (apply ~by (Value.Move (Slice ranges, x)))

let shape_of (Value.Any x) = Prim.shape x

(* [xs] read plainly. *)
let plain xs =
  Iarray.init (Array.length xs) (fun i ->
      let (Value.Any x) = xs.(i) in
      Value.Plain x)

(* Ladner and Fischer's prefix circuit at its least size, the tree of Brent
   and Kung:

     s     s0  s1  s2  s3  s4
     p     c(s0,s1)  c(s2,s3)          the pairs
     q     the scan of p, recursively
     y     s0  q0  c(q0,s2)  q1  c(q1,s4)

   Output [i]'s depth [f(i)] follows [f(0) = 0], [f(2j+1) = 1 + f(j)] and
   [f(2j) = 2 + f(j-1)], so [f(i) <= max(0, 2⌈log₂(i+1)⌉ - 1)]. *)
let circuit (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by axes
    combine (state : d Value.any array) : d Value.any array =
  let parts start count step =
    Array.mapi (fun k x -> part apply ~by axes.(k) start count step x)
  in
  let rec level state m =
    if m <= 1 then state
    else
      let half = m / 2 and rows = (m + 1) / 2 in
      let p = combine (parts 0 half 2 state) (parts 1 half 2 state) in
      let q = level p half in
      let evens =
        if rows = 1 then [||]
        else combine (parts 0 (rows - 1) 1 q) (parts 2 (rows - 1) 2 state)
      in
      Array.mapi
        (fun k (Value.Any y0) ->
          let dt = Prim.dtype y0 in
          let shape = shape_of state.(k) in
          let piece start count v =
            (along_axis shape axes.(k) start count 2, Prim.expect dt v)
          in
          let pieces =
            piece 0 1 (Value.Any y0)
            :: piece 1 half q.(k)
            :: (if rows = 1 then [] else [ piece 2 (rows - 1) evens.(k) ])
          in
          let fill = D.zero dt and shape = Iarray.of_array shape in
          Value.Any
            (apply ~by (Value.Assemble { dtype = dt; shape; fill; pieces })))
        (parts 0 1 1 state)
  in
  if Array.length state = 0 then state
  else level state (shape_of state.(0)).(axes.(0))

let scan (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    axis combine (state : d Value.any array) : d Value.any array =
  let dts = Array.map (fun (Value.Any x) -> D.Any (Prim.dtype x)) state in
  let along e =
    let s = Array.copy shape in
    s.(axis) <- e;
    s
  in
  (* The map of [exprs] over [xs], of [e] indices along [axis]. *)
  let mapped ins exprs xs =
    let e = (shape_of xs.(0)).(axis) in
    let prog = P.of_exprs ~ins (Iarray.of_array exprs) in
    map_n apply ~by (along e) prog dts (plain xs)
  in
  let combine earlier later =
    mapped (Iarray.of_array (Array.append dts dts)) combine
      (Array.append earlier later)
  in
  circuit apply ~by (Array.make (Array.length state) axis) combine state
