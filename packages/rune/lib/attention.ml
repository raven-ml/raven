(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module E = Nx_kernel.Prog.E
module Prim = Nx.Prim

type eval = { eval : 'r. 'r Prim.t -> 'r } [@@unboxed]

let ia = Iarray.of_array
let contiguous s = L.contiguous (ia s)

(* [x] moved: reshaped, broadcast or its axes permuted. *)
let reshape e s x = e.eval (Move (Reshape (ia s), x))
let broadcast e s x = e.eval (Move (Broadcast (ia s), x))
let permute e p x = e.eval (Move (Permute (ia p), x))

(* The map of the expression [out] over [loads], into [dt]. *)
let map (type v s d) e (dt : (v, s) D.t) shape (loads : d Prim.any list)
    (out : (v, s) E.t) : (v, s, d) Nx.t =
  let ins = List.map (fun (Prim.Any x) -> D.Any (Nx.dtype x)) loads in
  let prog = P.of_exprs ~ins:(ia (Array.of_list ins)) [| E.any out |] in
  let loads = List.map (fun (Prim.Any x) -> Prim.Plain x) loads in
  fst
    (e.eval
       (Map
          {
            layout = contiguous shape;
            prog;
            outs = Prim.[ dt ];
            loads = ia (Array.of_list loads);
            at = None;
          }))

(* [x] with [ones] axes of one element after its own, broadcast to [s]. *)
let spread e ~ones s x =
  broadcast e s (reshape e (Array.append (Nx.shape x) (Array.make ones 1)) x)

(* Each key's position in its pool flattened as [[hk; pages · page; d]], or
   [-1] where its table entry is outside [[0, pages)]: [[lead…; tk]]. *)
let positions e ~pages ~page (table : 'd Nx.int64_t) =
  let s = Nx.shape table in
  let r = Array.length s in
  let lead = Array.sub s 0 (r - 1) and max_pages = s.(r - 1) in
  let t = Array.append lead [| max_pages; page |] in
  let x = spread e ~ones:1 t table in
  let y =
    map e D.Int64 t [ Prim.Any x ]
      E.(
        let x = input 0 D.Int64 in
        let i64 n = const D.Int64 (Int64.of_int n) in
        where
          (i64 0 <= x && x < i64 pages)
          ((x * i64 page) + coord r)
          (i64 (-1)))
  in
  reshape e (Array.append lead [| max_pages * page |]) y

(* The pool [[pages; hk; page; d]] read at [positions], [[lead…; tk]]:
   [[lead…; hk; tk; d]], zero at a position of [-1]. *)
let paged_read e (positions : 'd Nx.int64_t) (pool : ('v, 's, 'd) Nx.t) =
  let r = Nx.ndim positions in
  let lead = Array.sub (Nx.shape positions) 0 (r - 1) in
  let tk = Nx.dim (-1) positions in
  let pages = Nx.dim 0 pool and hk = Nx.dim 1 pool in
  let page = Nx.dim 2 pool and d = Nx.dim 3 pool in
  let flat = reshape e [| hk; pages * page; d |] (permute e [| 1; 0; 2; 3 |] pool) in
  let units = Array.map (fun _ -> 1) lead in
  let x =
    broadcast e
      (Array.append lead [| hk; pages * page; d |])
      (reshape e (Array.append units [| hk; pages * page; d |]) flat)
  in
  let idx =
    broadcast e
      (Array.append lead [| hk; tk; d |])
      (reshape e (Array.append lead [| 1; tk; 1 |]) positions)
  in
  e.eval (Gather { axis = r; idx; x })

let compose (type v s w r a d) e ~prog ~scale ~(acc : (float, a) D.t)
    ~(out : (w, r) D.t) (q : (v, s, d) Nx.t) (kv : (v, s, d) Nx.kv)
    (lengths : d Nx.int64_t option) (loads : d Prim.any iarray) :
    (w, r, d) Nx.t * (float, a, d) Nx.t =
  let r = Nx.ndim q in
  let l = r - 3 in
  let lead = Array.sub (Nx.shape q) 0 l in
  let hq = Nx.dim (-3) q and tq = Nx.dim (-2) q in
  let with_lead tail = Array.append lead tail in
  let k, v, positions =
    match kv with
    | Dense { k; v } -> (k, v, None)
    | Paged { k; v; table } ->
        let p = positions e ~pages:(Nx.dim 0 k) ~page:(Nx.dim 2 k) table in
        (paged_read e p k, paged_read e p v, Some p)
  in
  let hk = Nx.dim (-3) k and tk = Nx.dim (-2) k and dv = Nx.dim (-1) v in
  let g = if hk = 0 then 1 else hq / hk in
  let f = with_lead [| hq; tq; tk |] in
  let dt = Nx.dtype q in
  let i64 n = E.const D.Int64 (Int64.of_int n) in
  (* Values past a row's length are zeroed: a NaN there would reach [p v]. *)
  let v =
    match lengths with
    | None -> v
    | Some len ->
        let s = Nx.shape v in
        let len = spread e ~ones:3 s len and key = r - 2 in
        map e dt s [ Prim.Any v; Prim.Any len ]
          E.(
            where
              (coord key < input 1 D.Int64)
              (input 0 dt) (const dt (D.zero dt)))
  in
  (* The batch pairs: the lead axes and the key heads. *)
  let batch = ia (Array.init (l + 1) (fun a -> (a, a))) in
  let grouped x = reshape e (with_lead [| hk; g; tq; Nx.dim (-1) x |]) x in
  let s =
    e.eval
      (Contract
         {
           batch;
           contracting = ia [| (l + 3, l + 2) |];
           acc = D.Any acc;
           out = acc;
           a = grouped q;
           b = k;
           init = None;
           a_scale = None;
           b_scale = None;
           segments = None;
           sum = false;
         })
  in
  let s = reshape e f s in
  (* The pair program over the score scaled. *)
  let pair =
    P.subst ~ins:prog.P.ins
      (function
        | P.In 0 ->
            Some
              (E.any
                 E.(input 0 acc * const acc scale))
        | _ -> None)
      prog
  in
  let visible, (score, ()) =
    e.eval
      (Map
         {
           layout = contiguous f;
           prog = pair;
           outs = Prim.[ D.Bool; acc ];
           loads =
             ia
               (Array.append [| Prim.Plain s |]
                  (Array.map (fun (Prim.Any x) -> Prim.Plain x) (Iarray.to_array loads)));
           at = None;
         })
  in
  (* A pair is visible where its key is before the row's length and its
     table entry is in range. *)
  let extra =
    (match lengths with Some len -> [ Prim.Any (spread e ~ones:3 f len) ] | None -> [])
    @ match positions with
      | Some p -> [ Prim.Any (broadcast e f (reshape e (with_lead [| 1; 1; tk |]) p)) ]
      | None -> []
  in
  let key = r - 1 in
  let at = 2 + Option.fold ~none:0 ~some:(fun _ -> 1) lengths in
  let m =
    map e acc f
      ([ Prim.Any visible; Prim.Any score ] @ extra)
      E.(
        let seen = input 0 D.Bool in
        let seen =
          match lengths with
          | Some _ -> seen && coord key < input 2 D.Int64
          | None -> seen
        in
        let seen =
          match positions with
          | Some _ -> seen && i64 0 <= input at D.Int64
          | None -> seen
        in
        where seen (input 1 acc) (const acc Float.neg_infinity))
  in
  let lse, () =
    e.eval
      (Reduce
         {
           layout = contiguous f;
           axes = ia [| r - 1 |];
           prog = P.of_exprs ~ins:(ia [| D.Any acc |]) [| E.any (E.input 0 acc) |];
           reductions = Prim.[ Logsumexp (0, acc) ];
           loads = ia [| Prim.Plain m |];
         })
  in
  (* A row whose [lse] is [-∞] has no visible key, or only [-∞] scores: its
     weights and [o] are zero, whatever its values hold. *)
  let row = spread e ~ones:1 f lse in
  let p =
    map e acc f [ Prim.Any m; Prim.Any row ]
      E.(
        let row = input 1 acc in
        where
          (row = const acc Float.neg_infinity)
          (const acc 0.)
          (exp (input 0 acc - row)))
  in
  let o =
    e.eval
      (Contract
         {
           batch;
           contracting = ia [| (l + 3, l + 1) |];
           acc = D.Any acc;
           out;
           a = grouped p;
           b = v;
           init = None;
           a_scale = None;
           b_scale = None;
           segments = None;
           sum = false;
         })
  in
  let os = with_lead [| hq; tq; dv |] in
  let o = reshape e os o in
  let o =
    map e out os [ Prim.Any o; Prim.Any (spread e ~ones:1 os lse) ]
      E.(
        where
          (input 1 acc = const acc Float.neg_infinity)
          (const out (D.zero out)) (input 0 out))
  in
  (o, lse)
