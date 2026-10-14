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
    prog dts (loads : d Value.load array) : d Value.any array =
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
  Array.mapi keep shape

(* [x]'s elements [start + j step], [j < count], along [axis]. *)
let part (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by axis
    start count step (Value.Any x : d Value.any) : d Value.any =
  let ranges = along_axis (Prim.shape x) axis start count step in
  Value.Any (apply ~by (Value.Move (Slice ranges, x)))

let plain (Value.Any x) = Value.Plain x

(* Ladner and Fischer's prefix circuit at its least size, the tree of Brent
   and Kung:

     s     s0  s1  s2  s3  s4
     p     c(s0,s1)  c(s2,s3)          the pairs
     q     the scan of p, recursively
     y     s0  q0  c(q0,s2)  q1  c(q1,s4)

   Output [i]'s depth [f(i)] follows [f(0) = 0], [f(2j+1) = 1 + f(j)] and
   [f(2j) = 2 + f(j-1)], so [f(i) <= max(0, 2⌈log₂(i+1)⌉ - 1)]. *)
let scan (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    axis ?first combine (state : d Value.any array) : d Value.any array =
  let dts = Array.map (fun (Value.Any x) -> D.Any (Prim.dtype x)) state in
  let along e =
    let s = Array.copy shape in
    s.(axis) <- e;
    s
  in
  let parts start count step =
    Array.map (part apply ~by axis start count step)
  in
  let ins = Iarray.of_array (Array.append dts dts) in
  let prog = P.of_exprs ~ins (Iarray.of_array combine) in
  (* The states [combine] makes of the [e] indices of the two. *)
  let combined e earlier later =
    let loads = Array.map plain (Array.append earlier later) in
    map_n apply ~by (along e) prog dts loads
  in
  (* Index [0]'s result, from [state]'s first. *)
  let head first state =
    let s0 = parts 0 1 1 state in
    match first with
    | None -> s0
    | Some first ->
        let ins = Iarray.of_array dts in
        let prog = P.of_exprs ~ins (Iarray.of_array first) in
        map_n apply ~by (along 1) prog dts (Array.map plain s0)
  in
  let rec level first state m =
    if m = 0 || (m = 1 && Option.is_none first) then state
    else if m = 1 then head first state
    else
      let half = m / 2 and rows = (m + 1) / 2 in
      let p = combined half (parts 0 half 2 state) (parts 1 half 2 state) in
      let q = level None p half in
      let evens =
        if rows = 1 then [||]
        else
          combined (rows - 1) (parts 0 (rows - 1) 1 q)
            (parts 2 (rows - 1) 2 state)
      in
      let h = head first state in
      let shape = along m in
      Array.mapi
        (fun k (Value.Any y0) ->
          let dt = Prim.dtype y0 in
          let piece start count v =
            (along_axis shape axis start count 2, Prim.expect dt v)
          in
          let pieces =
            piece 0 1 (Value.Any y0)
            :: piece 1 half q.(k)
            :: (if rows = 1 then [] else [ piece 2 (rows - 1) evens.(k) ])
          in
          let fill = D.zero dt in
          Value.Any
            (apply ~by (Value.Assemble { dtype = dt; shape; fill; pieces })))
        h
  in
  level first state shape.(axis)
