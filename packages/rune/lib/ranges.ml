(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module E = Nx_kernel.Prog.E
module Prim = Nx.Prim

type eval = { eval : 'r. 'r Prim.t -> 'r } [@@unboxed]

let ia = Iarray.of_array
let contiguous s = L.contiguous (ia s)
let plain loads = ia (Array.of_list (List.map (fun x -> Prim.Plain x) loads))

(* The int64 map of [out] over [loads] of shape [shape]. *)
let map e shape (loads : 'd Nx.int64_t list) (out : E.any) : 'd Nx.int64_t =
  let ins = ia (Array.of_list (List.map (fun _ -> D.Any D.Int64) loads)) in
  fst
    (e.eval
       (Map
          {
            layout = contiguous shape;
            prog = P.of_exprs ~ins [| out |];
            outs = Prim.[ D.Int64 ];
            loads = plain loads;
            at = None;
          }))

let ld i = E.input i D.Int64
let i64 n = E.const D.Int64 (Int64.of_int n)

(* A scan of [x]'s maps by [m] along [axis]. *)
let scan e m axis shape (x : 'd Nx.int64_t) (out : E.any) : 'd Nx.int64_t =
  e.eval
    (Scan
       {
         layout = contiguous shape;
         axis;
         prog = P.of_exprs ~ins:(ia [| D.Any D.Int64 |]) [| out |];
         reduction = Monoid (m, 0, D.Int64);
         loads = plain [ x ];
         carry = None;
       })

(* Position [p] belongs to run [j(p)], the first whose end exceeds it, so
   [j(p)] counts the runs whose running greatest end, at least [0], is at
   most [p]: a one added at each below [size], summed along the axis. Run
   [j] reads from [d.{j} = starts.{j} - ends.{j - 1}], so [p] reads
   [p + d.{j(p)}]; a [j(p)] of [k] is past every end. *)
let positions e ~size starts ends =
  let shape = Nx.shape starts in
  let a = Array.length shape - 1 in
  let k = shape.(a) in
  let out = Array.copy shape in
  out.(a) <- size;
  let at = scan e Max a shape ends (E.any (E.maximum (ld 0) (i64 0))) in
  let ones = map e shape [ at ] (E.any (i64 1)) in
  let none =
    fst
      (e.eval
         (Map
            {
              layout = contiguous out;
              prog = P.of_exprs ~ins:(ia [||]) [| E.any (i64 0) |];
              outs = Prim.[ D.Int64 ];
              loads = ia [||];
              at = Some (Nx.beside starts);
            }))
  in
  let count =
    e.eval
      (Scatter
         {
           combine = Add;
           unique = false;
           axis = a;
           idx = at;
           updates = ones;
           into = none;
         })
  in
  let j = scan e Sum a out count (E.any (ld 0)) in
  let before_at = map e shape [ ends ] (E.any E.(coord a - i64 1)) in
  let before = e.eval (Gather { axis = a; idx = before_at; x = ends }) in
  let d = map e shape [ starts; before ] (E.any E.(ld 0 - ld 1)) in
  let dj = e.eval (Gather { axis = a; idx = j; x = d }) in
  map e out [ j; dj ]
    (E.any E.(where (ld 0 < i64 k) (coord a + ld 1) (i64 (-1))))

let compose e ~size starts ends x =
  let pos = positions e ~size starts ends in
  let a = Nx.ndim starts - 1 in
  let cells = Nx.ndim x - a - 1 in
  let shape = Array.copy (Nx.shape x) in
  shape.(a) <- size;
  let idx =
    e.eval
      (Move
         ( Broadcast (ia shape),
           e.eval
             (Move
                ( Reshape (ia (Array.append (Nx.shape pos) (Array.make cells 1))),
                  pos )) ))
  in
  e.eval (Gather { axis = a; idx; x })
