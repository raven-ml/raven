(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx_kernel.Prog
module D = Nx_array.Dtype
module L = Nx_array.Layout
module Prim = Nx.Prim

(* Re-raised with the backtrace of its raise, which a plain [raise] would
   restart at the handler. *)
let reraise e = Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

type ('v, 's, 'd) Prim.payload +=
  | Dual : {
      primal : ('v, 's, 'd) Nx.t;
      tangent : ('v, 's, 'd) Nx.t;
      fresh : bool;
    }
      -> ('v, 's, 'd) Prim.payload

let primal i x =
  match Prim.payload i x with Some (Dual d) -> d.primal | _ -> x

let tangent i x =
  match Prim.payload i x with Some (Dual d) -> Some d.tangent | _ -> None

(* A value of [i] whose primal [y] an [Nx.item] reads, as an eager derivative
   branches on it. [fresh] where [t] is its own memory, in C order. *)
let dual i ~fresh y t =
  Prim.twinned i y (Dual { primal = y; tangent = t; fresh })

let seed i x t = dual i ~fresh:false x t
let tangible x = Scalar.tangible (D.Any (Nx.dtype x))

(* A value of the dtype [dt], checked. *)
let expect (type v s d) (dt : (v, s) D.t) (Prim.Any x : d Prim.any) :
    (v, s, d) Nx.t =
  match D.equal_witness (Nx.dtype x) dt with
  | Some Type.Equal -> x
  | None -> invalid_arg "Rune: a tangent of another dtype"

(* Loads *)

let primal_load i : 'd Prim.load -> 'd Prim.load = function
  | Plain x -> Plain (primal i x)
  | Padded { x; fill; pad } -> Padded { x = primal i x; fill; pad }

let tangent_load i : 'd Prim.load -> 'd Prim.load option = function
  | Plain x -> Option.map (fun t -> Prim.Plain t) (tangent i x)
  | Padded { x; pad; _ } ->
      Option.map
        (fun t -> Prim.Padded { x = t; fill = D.zero (Nx.dtype t); pad })
        (tangent i x)

let load_has i : 'd Prim.load -> bool = function
  | Plain x -> Option.is_some (tangent i x)
  | Padded { x; _ } -> Option.is_some (tangent i x)

let load_dtype : 'd Prim.load -> D.any = function
  | Plain x -> D.Any (Nx.dtype x)
  | Padded { x; _ } -> D.Any (Nx.dtype x)

(* The loads of a loop over primals, then tangents. *)
let jvp_loads i loads =
  let ps = Iarray.to_list (Iarray.map (primal_load i) loads) in
  let ts = List.filter_map (tangent_load i) (Iarray.to_list loads) in
  Iarray.of_list (ps @ ts)

(* Which loads have a tangent. *)
let has i loads =
  Array.init (Iarray.length loads) (fun j -> load_has i (Iarray.get loads j))

(* The same, bit [j] for load [j], of fewer loads than an [int] has bits. *)
let mask i loads =
  let m = ref 0 in
  for j = Iarray.length loads - 1 downto 0 do
    m := (!m lsl 1) lor Bool.to_int (load_has i (Iarray.get loads j))
  done;
  !m

let flags m n = Array.init n (fun j -> (m lsr j) land 1 = 1)
let masked loads = Iarray.length loads < Sys.int_size - 1

(* Outputs of dtypes known at run time *)

type 'd outs = Outs : ('d, 'r) Prim.outs * ('r -> 'd Prim.any list) -> 'd outs

let rec outs_of : type d. D.any list -> d outs = function
  | [] -> Outs ([], fun () -> [])
  | D.Any dt :: rest ->
      let (Outs (o, read)) = outs_of rest in
      Outs (dt :: o, fun (y, r) -> Prim.Any y :: read r)

(* [p]'s outputs at every index of [layout] over [loads], one map. *)
let run_map (type d) ~by layout (p : P.t) (loads : d Prim.load iarray) :
    d Prim.any array =
  let dts k = P.dtype p (Iarray.get p.outs k) in
  let (Outs (outs, read)) = outs_of (List.init (Iarray.length p.outs) dts) in
  Array.of_list
    (read (Prim.eval ~by (Map { layout; prog = p; outs; loads; at = None })))

(* Maps *)

(* A map's derivative: its program's primal and tangent outputs, each output
   that has a tangent followed by it, and which outputs have one. *)
type plan = { prog : P.t; flags : bool array; any : bool }

let plan prog tangents =
  let b = Scalar.start prog ~tangents ~extra:[||] in
  let m = Scalar.outputs b in
  let flags = Array.init m (fun k -> Scalar.tangent b k <> None) in
  let outs =
    List.concat
      (List.init m (fun k ->
           match Scalar.tangent b k with
           | Some t -> [ Scalar.value b k; t ]
           | None -> [ Scalar.value b k ]))
  in
  {
    prog = Scalar.finish b ~outs:(Array.of_list outs);
    flags;
    any = Array.exists Fun.id flags;
  }

(* Plans per domain, found by their program and its loads' tangents: the last
   one found, which a chain of one kind finds again, then a table, emptied when
   it holds [entries]. nx's functions build each kind's one-node program
   once. *)
type entry = { p : P.t; m : int; plan : plan }

type cache = {
  mutable last : entry option;
  table : (P.t * int, plan) Hashtbl.t;
}

let entries = 4096

let cache =
  Domain.DLS.new_key (fun () -> { last = None; table = Hashtbl.create 64 })

let plan_of (prog : P.t) m =
  let c = Domain.DLS.get cache in
  match c.last with
  | Some e when e.p == prog && e.m = m -> e.plan
  | _ ->
      let plan =
        match Hashtbl.find_opt c.table (prog, m) with
        | Some plan -> plan
        | None ->
            if Hashtbl.length c.table >= entries then Hashtbl.reset c.table;
            let plan = plan prog (flags m (Iarray.length prog.ins)) in
            Hashtbl.add c.table (prog, m) plan;
            plan
      in
      c.last <- Some { p = prog; m; plan };
      plan

(* The outputs of a map's derivative and how its result gives the map's, each
   output with a tangent a dual. *)
type ('d, 'r) jouts = J : ('d, 'q) Prim.outs * ('q -> 'r) -> ('d, 'r) jouts

let rec jouts : type d r.
    Prim.interpretation -> (d, r) Prim.outs -> bool array -> int -> (d, r) jouts
    =
 fun i outs flags k ->
  match outs with
  | [] -> J ([], fun () -> ())
  | dt :: rest ->
      let (J (o, f)) = jouts i rest flags (k + 1) in
      if flags.(k) then
        J (dt :: dt :: o, fun (y, (t, q)) -> (dual i ~fresh:true y t, f q))
      else J (dt :: o, fun (y, q) -> (y, f q))

let map_rule (type d r) i ~by layout prog (outs : (d, r) Prim.outs)
    (loads : d Prim.load iarray) : r =
  let p =
    if masked loads then plan_of prog (mask i loads)
    else plan prog (has i loads)
  in
  if not p.any then
    Prim.eval ~by
      (Map
         {
           layout;
           prog;
           outs;
           loads = Iarray.map (primal_load i) loads;
           at = None;
         })
  else
    let jl = jvp_loads i loads in
    match outs with
      | [ dt ] ->
          let y, (t, ()) =
            Prim.eval ~by
              (Map { layout; prog = p.prog; outs = [ dt; dt ]; loads = jl; at = None })
          in
          (dual i ~fresh:true y t, ())
      | _ ->
          let (J (o, f)) = jouts i outs p.flags 0 in
          f
            (Prim.eval ~by
               (Map { layout; prog = p.prog; outs = o; loads = jl; at = None }))

(* Reductions *)

type 'd red = Red : ('d, 'a) Prim.reduction -> 'd red

(* Reductions of dtypes known at run time, read one value each: a mean, of
   [Moments], is all a tangent needs. *)
type 'd reds =
  | Reds : ('d, 'r) Prim.reductions * ('r -> 'd Prim.any list) -> 'd reds

let first_of : type d a. (d, a) Prim.reduction -> a -> d Prim.any = function
  | Monoid _ -> fun y -> Any y
  | Logsumexp _ -> fun y -> Any y
  | Moments _ -> fun (m, _) -> Any m
  | Arg _ -> fun (y, _) -> Any y

let rec reds_of : type d. d red list -> d reds = function
  | [] -> Reds ([], fun () -> [])
  | Red r :: rest ->
      let (Reds (rs, read)) = reds_of rest in
      Reds (r :: rs, fun (y, q) -> first_of r y :: read q)

let rec split n l =
  if n = 0 then ([], l)
  else
    match l with
    | x :: l ->
        let a, b = split (n - 1) l in
        (x :: a, b)
    | [] -> invalid_arg "Rune: split"

let identity dt = P.of_node ~ins:[| dt |] (Op1 (Copy, dt, 0))

(* [x] moved [d] places along [axis], [d] elements of [fill] first and its last
   [d] dropped. *)
let shifted ~axis d fill x =
  let shape = Nx.shape x in
  let pads =
    Array.mapi (fun a _ -> if a = axis then (d, 0) else (0, 0)) shape
  in
  Nx.slice
    (List.init (Array.length shape) (fun a ->
         if a = axis then Nx.R (0, shape.(axis)) else Nx.A))
    (Nx.pad pads fill x)

(* [x] with [first], of extent 1 along [axis], before its slices along it and
   its last slice dropped. *)
let after ~axis first x =
  let n = Nx.dim axis x in
  let r = Nx.ndim x in
  let head =
    Nx.slice
      (List.init r (fun a -> if a = axis then Nx.R (0, n - 1) else Nx.A))
      x
  in
  Nx.concatenate ~axis [ first; head ]

(* [r] with [r_k = a_k r_(k-1) + b_k] along [axis] and [r_(-1) = 0]: the scan of
   the affine maps [r -> a r + b], [(a1, b1)] then [(a2, b2)] being [(a1 a2, a2
   b1 + b2)]. It needs no identity, so no padding multiplies a coefficient, and
   no division, so it holds where [a] has zeros. *)
let linear_scan ~axis a b =
  let compose (a1, b1) (a2, b2) = (Nx.mul a1 a2, Nx.add (Nx.mul a2 b1) b2) in
  snd (Nx.associative_scan ~axis Nx.Ptree.(pair tensor tensor) compose (a, b))

(* The weights of a log-sum-exp that gave [y]: [exp (x - y)], and [1] where [x]
   is [y], as where both are an infinity of one sign, so that the terms at an
   infinite total share its tangent. *)
let weights x y = Nx.where (Nx.equal x y) (Nx.ones_like x) (Nx.exp (Nx.sub x y))

(* At each element of [x], the product of the other elements over [axes]: the
   product before it in their order times the product after it. No division, so
   it is exact at zeros. *)
let others ~axes x =
  let shape = Nx.shape x and r = Nx.ndim x in
  let kept =
    List.filter (fun a -> not (List.mem a axes)) (List.init r Fun.id)
  in
  let order = kept @ axes in
  let moved = Nx.transpose ~axes:order x in
  let k = List.length kept in
  let n = List.fold_left (fun n a -> n * shape.(a)) 1 axes in
  let flat =
    Nx.reshape (Array.of_list (List.map (Array.get shape) kept @ [ n ])) moved
  in
  let one = D.one (Nx.dtype x) in
  let exclusive x = shifted ~axis:k 1 one (Nx.cumprod ~axis:k x) in
  let rev x = Nx.flip ~axes:[ k ] x in
  let o = Nx.mul (exclusive flat) (rev (exclusive (rev flat))) in
  let inverse = Array.make r 0 in
  List.iteri (fun j a -> inverse.(a) <- j) order;
  Nx.transpose ~axes:(Array.to_list inverse) (Nx.reshape (Nx.shape moved) o)

(* A reduction's tangent: the primal results it reads, broadcast over the loop;
   and, given the derivative builder and their nodes, the tangent reductions it
   adds and how their means give its dual result. *)
type ('d, 'a) part = {
  extra : 'd Prim.load list;
  emit :
    Scalar.t ->
    'd Prim.load iarray ->
    int list ->
    (int * (int -> 'd red)) list * ('d Prim.any list -> 'a);
}

let none y = { extra = []; emit = (fun _ _ _ -> ([], fun _ -> y)) }

(* [b]'s node [n] in [dt]. *)
let in_dtype b dt n = if Scalar.dtype b n = dt then n else Scalar.cast b dt n

(* The wider of two dtypes, the first of one width. *)
let wider (D.Any a as x) (D.Any c as y) = if D.bits c > D.bits a then y else x

(* The dtype a reduction of [b]'s node [n] into [dt] computes in: the wider of
   their carriers. *)
let carrier b n dt =
  wider (Nx_kernel.carrier (Scalar.dtype b n)) (Nx_kernel.carrier dt)

let sum dt o = Red (Monoid (Sum, o, dt))
let mean dt o = Red (Moments (o, dt))

let reduce_part (type d a) i ~by ~layout ~axes (r : (d, a) Prim.reduction)
    (y : a) : (d, a) part =
  let loop = Iarray.to_array (L.shape layout) in
  let spread (type v s) (y : (v, s, d) Nx.t) : d Prim.load =
    let kept =
      Array.mapi (fun a e -> if Iarray.mem a axes then 1 else e) loop
    in
    Plain (Nx.broadcast_to loop (Nx.reshape kept y))
  in
  (* Output [k]'s tangent reductions into [dt]: [nodes b v t es] the outputs
     they reduce and how, [finish] what their means give. None where [dt] has no
     derivative. *)
  let reduces (type v s) (dt : (v, s) D.t) k extra
      (nodes :
        Scalar.t -> int -> int -> int list -> (int * (int -> d red)) list)
      (finish : (v, s, d) Nx.t list -> a) : (d, a) part =
    let emit b _ es =
      match Scalar.tangent b k with
      | Some t when Scalar.tangible (D.Any dt) ->
          ( nodes b (Scalar.value b k) t es,
            fun l -> finish (List.map (expect dt) l) )
      | _ -> ([], fun _ -> y)
    in
    { extra; emit }
  in
  (* Tied extremes share the tangent: [v] rounded into [dt] is the extreme where
     it is [y]'s, as rounding keeps order. The ties are counted and their
     tangents summed in [dt]'s carrier, which counts more than a narrow float
     holds. *)
  let ties (type v s) (dt : (v, s) D.t) k (y : (v, s, d) Nx.t)
      (wrap : (v, s, d) Nx.t -> a) : (d, a) part =
    let (D.Any c) = Nx_kernel.carrier (D.Any dt) in
    reduces c k
      [ spread y ]
      (fun b v t es ->
        let dk = Scalar.dtype b t in
        let at = Scalar.equal b (in_dtype b (D.Any dt) v) (List.hd es) in
        let zero = Scalar.const b dk 0. and one = Scalar.const b dk 1. in
        [
          (Scalar.where b at t zero, sum c); (Scalar.where b at one zero, sum c);
        ])
      (function
        | [ n; m ] -> wrap (dual i ~fresh:true y (Nx.cast dt (Nx.div n m)))
        | _ -> wrap y)
  in
  (* The nodes of [v], [t] and [e] in the wider of [v]'s dtype and [dt]. *)
  let widened b dt v t e =
    let w = wider (Scalar.dtype b v) (D.Any dt) in
    (in_dtype b w v, in_dtype b w t, in_dtype b w e, w)
  in
  match r with
  | Monoid (Sum, k, dt) ->
      reduces dt k []
        (fun _ _ t _ -> [ (t, sum dt) ])
        (function [ t ] -> dual i ~fresh:true y t | _ -> y)
  | Monoid (Prod, k, dt) ->
      (* The products of the others, in the reduction's carrier, rounded once
         into [dt]. *)
      let emit b loads _ =
        match Scalar.tangent b k with
        | Some t when Scalar.tangible (D.Any dt) ->
            let (D.Any c) = carrier b (Scalar.value b k) (D.Any dt) in
            let w = D.Any c in
            let prog =
              Scalar.finish b
                ~outs:[| in_dtype b w (Scalar.value b k); in_dtype b w t |]
            in
            let vs = run_map ~by layout prog loads in
            let v = expect c vs.(0) and t = expect c vs.(1) in
            let axes = Iarray.to_list axes in
            let tan = Nx.cast dt (Nx.sum ~axes (Nx.mul (others ~axes v) t)) in
            ([], fun _ -> dual i ~fresh:true y tan)
        | _ -> ([], fun _ -> y)
      in
      { extra = []; emit }
  | Monoid ((Max | Min), k, dt) -> ties dt k y Fun.id
  | Arg (_, k, dt) ->
      let v, pos = y in
      ties dt k v (fun v -> (v, pos))
  | Logsumexp (k, dt) ->
      (* The softmax-weighted mean of the tangents, [Σ w t / Σ w], the weights
         [exp (v - y)]: the rounding of [y] scales both sums, taken in [dt]'s
         carrier, which a narrow float's count of infinite terms fits. *)
      let (D.Any c) = Nx_kernel.carrier (D.Any dt) in
      reduces c k
        [ spread y ]
        (fun b v t es ->
          let v, t, e, w = widened b c v t (List.hd es) in
          let one = Scalar.const b w 1. in
          let u =
            Scalar.where b (Scalar.equal b v e) one
              (Scalar.exp b (Scalar.sub b v e))
          in
          [ (Scalar.mul b u t, sum c); (u, sum c) ])
        (function
          | [ n; d ] -> dual i ~fresh:true y (Nx.cast dt (Nx.div n d)) | _ -> y)
  | Moments (k, dt) ->
      (* The variance's tangent is twice the mean of [(v - m) (t - dm)], [mean
         ((v - m) t) - mean (v - m) dm], exact to first order whatever [m]'s
         rounding. *)
      let m, var = y in
      reduces dt k
        [ spread m ]
        (fun b v t es ->
          let v, t, e, _ = widened b dt v t (List.hd es) in
          let c = Scalar.sub b v e in
          [ (t, mean dt); (Scalar.mul b c t, mean dt); (c, mean dt) ])
        (function
          | [ dm; ct; cm ] ->
              let two = Nx.scalar_like dm (D.of_float dt 2.) in
              let dv = Nx.mul two (Nx.sub ct (Nx.mul cm dm)) in
              (dual i ~fresh:true m dm, dual i ~fresh:true var dv)
          | _ -> y)

let rec reduce_parts : type d r.
    Prim.interpretation ->
    by:string ->
    layout:L.t ->
    axes:int iarray ->
    (d, r) Prim.reductions ->
    r ->
    (d, r) part =
 fun i ~by ~layout ~axes rs y ->
  match rs with
  | [] -> none ()
  | r :: rest ->
      let y0, yq = y in
      let h = reduce_part i ~by ~layout ~axes r y0 in
      let q = reduce_parts i ~by ~layout ~axes rest yq in
      let n = List.length h.extra in
      let emit b loads es =
        let e1, e2 = split n es in
        let r1, f1 = h.emit b loads e1 and r2, f2 = q.emit b loads e2 in
        ( r1 @ r2,
          fun l ->
            let l1, l2 = split (List.length r1) l in
            (f1 l1, f2 l2) )
      in
      { extra = h.extra @ q.extra; emit }

(* The fused derivative of a reduction whose tangents are sums: each sum with a
   tangent followed by the sum of its tangent, in one loop. *)
type ('d, 'r) jreds =
  | JR : ('d, 'q) Prim.reductions * ('q -> 'r) -> ('d, 'r) jreds

let rec jreds : type d r.
    Prim.interpretation ->
    Scalar.t ->
    (d, r) Prim.reductions ->
    int ref ->
    (d, r) jreds =
 fun i b rs next ->
  match rs with
  | [] -> JR ([], fun () -> ())
  | (Monoid (Sum, k, dt) as r) :: rest
    when Scalar.tangent b k <> None && Scalar.tangible (D.Any dt) ->
      let o = !next in
      incr next;
      let (JR (q, f)) = jreds i b rest next in
      JR
        ( r :: Monoid (Sum, o, dt) :: q,
          fun (y, (t, z)) -> (dual i ~fresh:true y t, f z) )
  | r :: rest ->
      let (JR (q, f)) = jreds i b rest next in
      JR (r :: q, fun (y, z) -> (y, f z))

(* The tangents [rs]'s sums reduce, in order. *)
let rec sum_tangents : type d r. Scalar.t -> (d, r) Prim.reductions -> int list
    =
 fun b rs ->
  match rs with
  | [] -> []
  | Monoid (Sum, k, dt) :: rest -> (
      match Scalar.tangent b k with
      | Some t when Scalar.tangible (D.Any dt) -> t :: sum_tangents b rest
      | _ -> sum_tangents b rest)
  | (Monoid _ | Logsumexp _ | Moments _ | Arg _) :: rest -> sum_tangents b rest

(* Whether each of [rs] with a tangent is a sum. *)
let rec sums_only : type d r. Scalar.t -> (d, r) Prim.reductions -> bool =
 fun b rs ->
  let plain k rest = Scalar.tangent b k = None && sums_only b rest in
  match rs with
  | [] -> true
  | Monoid (Sum, _, _) :: rest -> sums_only b rest
  | Monoid ((Prod | Max | Min), k, _) :: rest -> plain k rest
  | Logsumexp (k, _) :: rest -> plain k rest
  | Moments (k, _) :: rest -> plain k rest
  | Arg (_, k, _) :: rest -> plain k rest

let reduce_rule (type d r) i ~by layout axes (prog : P.t)
    (reductions : (d, r) Prim.reductions) (loads : d Prim.load iarray) : r =
  let tangents = has i loads in
  let jl = jvp_loads i loads in
  let b = Scalar.start prog ~tangents ~extra:[||] in
  let k = Scalar.outputs b in
  let touts = sum_tangents b reductions in
  if sums_only b reductions then
    let (JR (rs, f)) = jreds i b reductions (ref k) in
    let outs =
      Array.append (Array.init k (Scalar.value b)) (Array.of_list touts)
    in
    let prog = Scalar.finish b ~outs in
    f
      (Prim.eval ~by
         (Reduce { layout; axes; prog; reductions = rs; loads = jl }))
  else
    let primal_op =
      Prim.Reduce
        {
          layout;
          axes;
          prog;
          reductions;
          loads = Iarray.map (primal_load i) loads;
        }
    in
    let y = Prim.eval ~by primal_op in
    let part = reduce_parts i ~by ~layout ~axes reductions y in
    let extra = Array.of_list part.extra in
    let b = Scalar.start prog ~tangents ~extra:(Array.map load_dtype extra) in
    let base = Array.length (Scalar.ins b) - Array.length extra in
    let es =
      List.init (Array.length extra) (fun e -> Scalar.operand b (base + e))
    in
    let loads = Iarray.append jl (Iarray.of_array extra) in
    match part.emit b loads es with
    | [], finish -> finish []
    | reds, finish ->
        let nodes = Array.of_list (List.map fst reds) in
        let makers = List.map snd reds in
        let (Reds (rs, read)) = reds_of (List.mapi (fun o f -> f o) makers) in
        let prog = Scalar.finish b ~outs:nodes in
        finish
          (read
             (Prim.eval ~by
                (Reduce { layout; axes; prog; reductions = rs; loads })))

(* Scans *)

(* A running tangent computed in a carrier [c]: [f c up v t], [up] casting a
   value into [c], [v] and [t] an output and its tangent in [c]. *)
type ('v, 's, 'd) carried = {
  f :
    'c 'r.
    ('c, 'r) D.t ->
    (('v, 's, 'd) Nx.t -> ('c, 'r, 'd) Nx.t) ->
    ('c, 'r, 'd) Nx.t ->
    ('c, 'r, 'd) Nx.t ->
    ('c, 'r, 'd) Nx.t list;
}

let scan_rule (type d r) i ~by layout axis (prog : P.t)
    (reduction : (d, r) Prim.reduction) (loads : d Prim.load iarray)
    (carry : d Prim.any option) : r =
  let pcarry = Option.map (fun (Prim.Any c) -> Prim.Any (primal i c)) carry in
  let ploads = Iarray.map (primal_load i) loads in
  let y =
    Prim.eval ~by
      (Scan { layout; axis; prog; reduction; loads = ploads; carry = pcarry })
  in
  let ct =
    match carry with
    | Some (Any c) -> Option.map (fun t -> Prim.Any t) (tangent i c)
    | None -> None
  in
  let b = Scalar.start prog ~tangents:(has i loads) ~extra:[||] in
  let jl = jvp_loads i loads in
  (* Output [k] and its tangent in [w], computed into arrays; none where neither
     it nor the carry has a tangent, or [dt] has no derivative. *)
  let parts (type v s c q) (dt : (v, s) D.t) (w : (c, q) D.t) k =
    match (Scalar.tangent b k, ct) with
    | None, None -> None
    | _ when not (Scalar.tangible (D.Any dt)) -> None
    | t, _ ->
        let w' = D.Any w in
        let v = in_dtype b w' (Scalar.value b k) in
        let t =
          match t with
          | Some t -> in_dtype b w' t
          | None -> Scalar.const b w' 0.
        in
        let vs = run_map ~by layout (Scalar.finish b ~outs:[| v; t |]) jl in
        Some (expect w vs.(0), expect w vs.(1))
  in
  (* The tangents [g] computes in output [k]'s carrier or [dt]'s, the wider,
     rounded once into [dt]. *)
  let in_carrier (type v s) (dt : (v, s) D.t) k (g : (v, s, d) carried) =
    let (D.Any c) = carrier b (Scalar.value b k) (D.Any dt) in
    Option.map
      (fun (v, t) -> List.map (Nx.cast dt) (g.f c (Nx.cast c) v t))
      (parts dt c k)
  in
  let shape1 () =
    Array.mapi
      (fun a e -> if a = axis then 1 else e)
      (Iarray.to_array (L.shape layout))
  in
  let scan1 (type v s q) (x : (v, s, d) Nx.t)
      (reduction : (d, q) Prim.reduction) carry : q =
    let prog = identity (D.Any (Nx.dtype x)) in
    Prim.eval ~by
      (Scan { layout; axis; prog; reduction; loads = [| Plain x |]; carry })
  in
  let cast_any (type v s) (dt : (v, s) D.t) (Prim.Any c : d Prim.any) :
      (v, s, d) Nx.t =
    Nx.cast dt c
  in
  (* The running extreme's tangent: the tangent of the first element that takes
     it, or the carry's where no element beats the carry. *)
  let extreme (type v s) (e : Nx_kernel.extreme) (dt : (v, s) D.t) v t =
    let m, pos = scan1 v (Arg (e, 0, dt)) None in
    let g = Prim.eval ~by (Gather { axis; idx = pos; x = t }) in
    match pcarry with
    | None -> g
    | Some c ->
        let c = cast_any dt c in
        let beats = match e with Max -> Nx.less c m | Min -> Nx.less m c in
        let nan = Nx.logical_and (Nx.isnan m) (Nx.logical_not (Nx.isnan c)) in
        let dc =
          match ct with Some t -> cast_any dt t | None -> Nx.zeros_like c
        in
        Nx.where (Nx.logical_or beats nan) g dc
  in
  let single = function [ t ] -> t | _ -> invalid_arg "Rune: one tangent" in
  match reduction with
  | Monoid (Sum, k, dt) -> (
      match parts dt dt k with
      | None -> y
      | Some (_, t) ->
          let carry =
            Option.map
              (fun c ->
                let (D.Any cd) = Nx_kernel.carrier (D.Any dt) in
                Prim.Any (cast_any cd c))
              ct
          in
          dual i ~fresh:true y (scan1 t (Monoid (Sum, 0, dt)) carry))
  | Monoid (Max, k, dt) -> (
      match parts dt dt k with
      | None -> y
      | Some (v, t) -> dual i ~fresh:true y (extreme Max dt v t))
  | Monoid (Min, k, dt) -> (
      match parts dt dt k with
      | None -> y
      | Some (v, t) -> dual i ~fresh:true y (extreme Min dt v t))
  | Monoid (Prod, k, dt) -> (
      (* [dp_i = v_i dp_(i-1) + p_(i-1) t_i], from the carry's tangent. *)
      let f c up v t =
        let first =
          match pcarry with
          | Some x -> cast_any c x
          | None ->
              Nx.ones_at (Nx.beside v) c (shape1 ())
        in
        let tan = linear_scan ~axis v (Nx.mul (after ~axis first (up y)) t) in
        match ct with
        | Some x -> [ Nx.add tan (Nx.mul (cast_any c x) (Nx.cumprod ~axis v)) ]
        | None -> [ tan ]
      in
      match in_carrier dt k { f } with
      | None -> y
      | Some l -> dual i ~fresh:true y (single l))
  | Arg (_, k, dt) -> (
      let value, pos = y in
      match parts dt dt k with
      | None -> y
      | Some (_, t) ->
          let g = Prim.eval ~by (Gather { axis; idx = pos; x = t }) in
          (dual i ~fresh:true value g, pos))
  | Logsumexp (k, dt) -> (
      (* The weighted mean of the tangents up to each element, as a reduction's,
         its two sums each the recurrence [s_i = a_i s_(i-1) + u_i x_i], [a_i]
         the weights' change from [y_(i-1)] to [y_i]. *)
      let f c up v t =
        let y = up y in
        let here = Nx.beside y in
        let lowest =
          Nx.full_at here c (shape1 ()) (D.of_float c Float.neg_infinity)
        in
        let a = weights (after ~axis lowest y) y and u = weights v y in
        [ Nx.div (linear_scan ~axis a (Nx.mul u t)) (linear_scan ~axis a u) ]
      in
      match in_carrier dt k { f } with
      | None -> y
      | Some l -> dual i ~fresh:true y (single l))
  | Moments (k, dt) -> (
      (* The variance's tangent is [2 (C_i - dm_i D_i) / (i + 1)]: [C_i] the sum
         of [(v_j - m_i) t_j] up to [i], updated as the mean moves, [C_i =
         C_(i-1) + (v_i - m_i) t_i + (m_(i-1) - m_i) S_(i-1)], [S] the running
         sum of [t], and [D_i] the same of [1], which the rounding of [m_i]
         keeps from zero. *)
      let m, var = y in
      let f c up v t =
        let m = up m in
        let dm, _ = scan1 t (Moments (0, c)) None in
        let here = Nx.beside v in
        let zero = Nx.zeros_at here c (shape1 ()) in
        let centred t =
          let s = Nx.cumsum ~axis t in
          Nx.cumsum ~axis
            (Nx.add
               (Nx.mul (Nx.sub v m) t)
               (Nx.mul (Nx.sub (after ~axis zero m) m) (after ~axis zero s)))
        in
        let n = Nx.dim axis v and r = Nx.ndim v in
        let along = Array.init r (fun a -> if a = axis then n else 1) in
        let count =
          Nx.reshape along (Nx.cast c (Nx.arange_at here Nx.int32 1 (n + 1) 1))
        in
        let two = Nx.scalar_like v (D.of_float c 2.) in
        let spread =
          Nx.sub (centred t) (Nx.mul dm (centred (Nx.ones_like t)))
        in
        [ dm; Nx.div (Nx.mul two spread) count ]
      in
      match in_carrier dt k { f } with
      | Some [ dm; dv ] -> (dual i ~fresh:true m dm, dual i ~fresh:true var dv)
      | _ -> y)

(* Scatters by extreme: each target's tangent is the mean of the tangents of the
   elements that tie at its extreme, its own and its updates', counted and
   summed in the carrier of their dtype, which counts more than a narrow float
   holds. *)
let scatter_extreme (type v s d) i ~by ~unique ~axis
    (combine : Nx_kernel.combine) (idx : d Nx.int64_t)
    (updates : (v, s, d) Nx.t) (into : (v, s, d) Nx.t) =
  let pu = primal i updates and pi = primal i into in
  let y =
    Prim.eval ~by
      (Scatter { combine; unique; axis; idx; updates = pu; into = pi })
  in
  let at = Prim.eval ~by (Gather { axis; idx; x = y }) in
  let won = Nx.equal pu at and kept = Nx.equal pi y in
  let dt = Nx.dtype pu in
  let (D.Any c) = Nx_kernel.carrier (D.Any dt) in
  let wide x =
    Nx.cast c (Option.value (tangent i x) ~default:(Nx.zeros_like (primal i x)))
  in
  let tu = wide updates and ti = wide into in
  let zero x = Nx.zeros_like x and one x = Nx.ones_like x in
  let add updates into =
    Prim.eval ~by (Scatter { combine = Add; unique; axis; idx; updates; into })
  in
  let count =
    add (Nx.where won (one tu) (zero tu)) (Nx.where kept (one ti) (zero ti))
  in
  let sum = add (Nx.where won tu (zero tu)) (Nx.where kept ti (zero ti)) in
  dual i ~fresh:true y (Nx.cast dt (Nx.div sum count))

(* Complex values and their parts' pairs: bitcasts between them are linear. *)
let same_coordinates (D.Any a) (D.Any b) =
  match (a, b) with
  | D.Complex64, D.Float32 | D.Float32, D.Complex64 -> true
  | D.Complex128, D.Float64 | D.Float64, D.Complex128 -> true
  | _ -> false

(* A factorisation's or a triangular solve's dual results: the routine on the
   primal, and its tangents from the factors. A routine that gives values alone
   takes its vectors from the routine that gives both. *)
let linalg_rule (type d r) name i ~by (l : (d, r) Prim.linalg) : r =
  let eval l = Prim.eval ~by (Linalg l) in
  let dual y t = dual i ~fresh:false y t in
  match l with
  | Cholesky { triangle; a } -> (
      let y = eval (Cholesky { triangle; a = primal i a }) in
      match tangent i a with
      | Some da -> dual y (Factor.cholesky triangle y da)
      | None -> y)
  | Lu a -> (
      let lu, pivots, perm = eval (Lu (primal i a)) in
      match tangent i a with
      | Some da -> (dual lu (Factor.lu lu perm da), pivots, perm)
      | None -> (lu, pivots, perm))
  | Qr { factors; a } -> (
      let q, r = eval (Qr { factors; a = primal i a }) in
      match tangent i a with
      | Some da ->
          let dq, dr = Factor.qr name factors q r da in
          (dual q dq, dual r dr)
      | None -> (q, r))
  | Svd_values { dtype; a } -> (
      let pa = primal i a in
      let s = eval (Svd_values { dtype; a = pa }) in
      match tangent i a with
      | Some da ->
          let _, u, vh = eval (Svd { factors = Reduced; dtype; a = pa }) in
          dual s (Factor.svd_values dtype u vh da)
      | None -> s)
  | Svd { factors; dtype; a } -> (
      let s, u, vh = eval (Svd { factors; dtype; a = primal i a }) in
      match tangent i a with
      | Some da ->
          let ds, du, dvh = Factor.svd name factors dtype s u vh da in
          (dual s ds, dual u du, dual vh dvh)
      | None -> (s, u, vh))
  | Eigh_values { dtype; a } -> (
      let pa = primal i a in
      let w = eval (Eigh_values { dtype; a = pa }) in
      match tangent i a with
      | Some da ->
          let _, v = eval (Eigh { dtype; a = pa }) in
          dual w (Factor.eigh_values dtype v da)
      | None -> w)
  | Eigh { dtype; a } -> (
      let w, v = eval (Eigh { dtype; a = primal i a }) in
      match tangent i a with
      | Some da ->
          let dw, dv = Factor.eigh dtype w v da in
          (dual w dw, dual v dv)
      | None -> (w, v))
  | Eig_values { dtype; a } -> (
      let pa = primal i a in
      let w = eval (Eig_values { dtype; a = pa }) in
      match tangent i a with
      | Some da ->
          let _, v = eval (Eig { dtype; a = pa }) in
          dual w (Factor.eig_values v da)
      | None -> w)
  | Eig { dtype; a } -> (
      let w, v = eval (Eig { dtype; a = primal i a }) in
      match tangent i a with
      | Some da ->
          let dw, dv = Factor.eig w v da in
          (dual w dw, dual v dv)
      | None -> (w, v))
  | Solve_triangular { triangle; adjoint; unit_diagonal; a; b } -> (
      let pa = primal i a and pb = primal i b in
      let x =
        eval
          (Solve_triangular { triangle; adjoint; unit_diagonal; a = pa; b = pb })
      in
      match (tangent i a, tangent i b) with
      | None, None -> x
      | da, db ->
          dual x
            (Factor.solve_triangular ~by triangle ~adjoint ~unit_diagonal pa x
               da db))

let rule (type r) name i ~by (op : r Prim.t) : r =
  (* [x]'s dual from [f] of its primal and of its tangent. *)
  let both ~fresh f x =
    let y = f (primal i x) in
    match tangent i x with Some t -> dual i ~fresh y (f t) | None -> y
  in
  match op with
  | Map { layout; prog; outs; loads; _ } ->
      map_rule i ~by layout prog outs loads
  | Reduce { layout; axes; prog; reductions; loads } ->
      reduce_rule i ~by layout axes prog reductions loads
  | Scan { layout; axis; prog; reduction; loads; carry } ->
      scan_rule i ~by layout axis prog reduction loads carry
  | Gather g ->
      both ~fresh:true (fun x -> Prim.eval ~by (Gather { g with x })) g.x
  | Scatter ({ combine = Add | Set; _ } as s) -> (
      let scatter updates into =
        Prim.eval ~by (Scatter { s with updates; into })
      in
      let pu = primal i s.updates and pi = primal i s.into in
      let y = scatter pu pi in
      match (tangent i s.updates, tangent i s.into) with
      | None, None -> y
      | Some tu, ti ->
          let ti = Option.value ti ~default:(Nx.zeros_like pi) in
          dual i ~fresh:true y (scatter tu ti)
      | None, Some ti -> (
          match s.combine with
          | Add -> dual i ~fresh:false y ti
          | Set | Max | Min ->
              dual i ~fresh:true y (scatter (Nx.zeros_like pu) ti)))
  | Scatter { combine; unique; axis; idx; updates; into } ->
      scatter_extreme i ~by ~unique ~axis combine idx updates into
  | Sort s ->
      let v, idx = Prim.eval ~by (Sort { s with x = primal i s.x }) in
      let v =
        match tangent i s.x with
        | Some t ->
            dual i ~fresh:true v
              (Prim.eval ~by (Gather { axis = s.axis; idx; x = t }))
        | None -> v
      in
      (v, idx)
  | Fold f -> both ~fresh:true (fun x -> Prim.eval ~by (Fold { f with x })) f.x
  | Assemble a ->
      let assemble fill pieces =
        Prim.eval ~by (Assemble { a with fill; pieces })
      in
      let y =
        assemble a.fill (List.map (fun (r, x) -> (r, primal i x)) a.pieces)
      in
      let piece (r, x) =
        match tangent i x with
        | Some t -> (r, t)
        | None -> (r, Nx.zeros_like (primal i x))
      in
      dual i ~fresh:true y (assemble (D.zero a.dtype) (List.map piece a.pieces))
  | Contract c -> (
      let scaled = function
        | Some (Prim.Value s) -> Option.is_some (tangent i s)
        | Some (Exponent _) | None -> false
      in
      if scaled c.a_scale || scaled c.b_scale then
        invalid_arg
          (name
         ^ ": a contraction's scale has a tangent; scales have no derivative");
      let pa = primal i c.a and pb = primal i c.b in
      let init = Option.map (primal i) c.init in
      let y = Prim.eval ~by (Contract { c with a = pa; b = pb; init }) in
      if not (tangible y) then y
      else
        (* [ta · b + a · tb + tinit], each product accumulated onto the terms
           after it. *)
        let ti = Option.bind c.init (tangent i) in
        let product a b init =
          Some (Prim.eval ~by (Contract { c with a; b; init }))
        in
        let t =
          match tangent i c.b with Some tb -> product pa tb ti | None -> ti
        in
        let t =
          match tangent i c.a with Some ta -> product ta pb t | None -> t
        in
        (* With no product, the tangent is the init's own. *)
        let fresh =
          Option.is_some (tangent i c.a) || Option.is_some (tangent i c.b)
        in
        match t with Some t -> dual i ~fresh y t | None -> y)
  | Fft (C2c f) ->
      both ~fresh:true (fun x -> Prim.eval ~by (Fft (C2c { f with x }))) f.x
  | Fft (R2c f) ->
      both ~fresh:true (fun x -> Prim.eval ~by (Fft (R2c { f with x }))) f.x
  | Fft (C2r f) ->
      both ~fresh:true (fun x -> Prim.eval ~by (Fft (C2r { f with x }))) f.x
  | Linalg l -> linalg_rule name i ~by l
  | Copy x -> both ~fresh:true (fun x -> Prim.eval ~by (Copy x)) x
  | Move (mv, x) -> both ~fresh:false (fun x -> Prim.eval ~by (Move (mv, x))) x
  | Bitcast (dt, x) ->
      if same_coordinates (D.Any dt) (D.Any (Nx.dtype x)) then
        both ~fresh:false (fun x -> Prim.eval ~by (Bitcast (dt, x))) x
      else Prim.eval ~by (Bitcast (dt, primal i x))
  | Place (q, x) -> (
      let y = Prim.eval ~by (Place (q, primal i x)) in
      match tangent i x with
      | Some t -> dual i ~fresh:false y (Prim.eval ~by (Place (q, t)))
      | None -> y)
  | Check _ -> Prim.eval ~by (Prim.map (fun x -> primal i x) op)

(* Constructs *)

let owned i (Program.V x) =
  match Prim.owner x with Some j -> j == i | None -> false

(* Custom rules

   The rules over a custom rule ({!Program.Custom}) and a linear call
   ({!Program.Linear}), as {!Custom} calls trace them. *)

module Rules = struct
  open Program

  let sub = Array.sub
  let add (V a) b = V (Nx.add a (like a b))
  let tangible_v (V x) = tangible x
  let primal_v i (V x) = V (primal i x)
  let residual_count = Custom.residual_count

  let tangent_v i (V x) =
    match Prim.payload i x with
    | Some (Dual d) -> Some (V d.tangent)
    | _ -> None

  let dual_v i (V y) t = V (dual i ~fresh:false y (like y t))

  (* The interpretation, its entry point, its tape under reverse mode, and a
     forward derivative of a program from its tangents. *)
  type ctx = {
    i : Prim.interpretation;
    entry : string;
    tape : Prim.interpretation option;
    derive : Program.t -> value array -> (value * value option) array;
  }

  let tracks c ~name captures =
    if Array.exists (owned c.i) captures then
      invalid_arg
        (name
       ^ ": the rule uses a value its own differentiation tracks; pass it as \
          an argument")

  (* A tangent per argument: the differentiation's, or zeros. *)
  let tangents c args =
    Array.map
      (fun a ->
        match tangent_v c.i a with
        | Some t -> t
        | None ->
            let (V p) = primal_v c.i a in
            V (Nx.zeros_like p))
      args

  (* The results [ys] with their tangents [dys] where they have one. *)
  let with_tangents c ys dys =
    Array.map2 (fun y t -> if tangible_v y then dual_v c.i y t else y) ys dys

  (* Under reverse mode the tangents are a tape's variables, which a total
     cannot take. *)
  let taped c (V v) =
    match (Nx.Prim.owner v, c.tape) with
    | Some o, Some tape -> o == tape
    | _ -> false

  let added =
    "Rune.Total.add: a custom_jvp tangent map adds a tangent under reverse \
     mode; a total takes values"

  let no_tangent_added c adds =
    if Array.exists (taped c) adds then invalid_arg added

  let no_forward c =
    invalid_arg
      (c.entry
     ^ ": a custom_vjp rule has no forward derivative; give the function a \
        custom_jvp rule")

  (* A custom rule over the differentiation's values. The results' primals are
     the call performed outward on the arguments' primals, so that every
     differentiation around applies the rule too. Their tangents are the tangent
     map evaluated outward, or a linear call of the pullback over the tangents,
     which the tape records; so a differentiation around differentiates the
     derivative's code. That needs the residuals with their derivatives around:
     the call performed outward is asked for them, and where this call is asked
     for its own, the residuals are derived again, as the primal's code computes
     them. Forward mode has no rule for a pullback's results: where a reverse
     mode inside applies the pullback, it differentiates the rule's code. *)
  let rec custom c ~name ~primal:p ~derivative ~ny ~args ~captures ~residuals
      ~applied ~seeds =
    let r =
      applied_rule c ~name ~primal:p ~derivative ~ny ~args ~captures ~residuals
        ~applied ~seeds
    in
    (* A sum runs from its seed, with the seed's tangent: the primal's additions
       have none. *)
    let base = ny + residual_count p ~ny in
    Array.mapi
      (fun k v ->
        if k < base || k >= base + Array.length seeds then v
        else
          match tangent_v c.i seeds.(k - base) with
          | Some t -> dual_v c.i v t
          | None -> v)
      r

  and applied_rule c ~name ~primal:p ~derivative ~ny ~args ~captures ~residuals
      ~applied ~seeds =
    tracks c ~name captures;
    let pargs = Array.map (primal_v c.i) args in
    let na = Array.length p.adds and nr = residual_count p ~ny in
    let outward ~residuals ~applied =
      perform
        (Custom
           {
             name;
             primal = p;
             derivative;
             ny;
             args = pargs;
             captures;
             residuals;
             applied;
             seeds = Array.map (primal_v c.i) seeds;
           })
    in
    let reverse = Option.is_some c.tape in
    let pullback =
      match derivative with Pullback _ -> true | Tangent _ -> false
    in
    (* The primal derived again computes what the call performed outward
       computed: it offers nothing. *)
    let derive p =
      Program.silenced (fun () -> c.derive p (Array.append args captures))
    in
    let dual (y, t) = match t with Some t -> dual_v c.i y t | None -> y in
    match derivative with
    | Pullback _ when (not reverse) && not applied ->
        if Array.exists tangible_v (sub p.outputs 0 ny) then no_forward c;
        outward ~residuals ~applied
    | Pullback _ when not reverse ->
        (* A reverse mode inside applies the rule. The primals come from the
           call performed outward, so that the reverse modes around apply it
           too; this forward mode has no rule, and takes the tangents of the
           rule's code, the residuals' too where they are asked for. *)
        let out = outward ~residuals:false ~applied in
        let slice = if residuals then ny + nr else ny in
        let derived = derive (prune p ~outputs:(Array.init slice Fun.id)) in
        let ys = Array.init ny (fun k -> dual (out.(k), snd derived.(k))) in
        let res =
          if residuals then Array.map dual (sub derived ny nr)
          else sub out ny nr
        in
        Array.concat [ ys; res; sub out (ny + nr) na ]
    | _ -> (
        let nt =
          match derivative with
          | Tangent t -> Array.length t.adds
          | Pullback _ -> 0
        in
        let ys, res, res_out, adds =
          if residuals then
            let derived =
              sub
                (derive (prune p ~outputs:(Array.init nr (fun k -> ny + k))))
                0 nr
            in
            let out = outward ~residuals:false ~applied:pullback in
            ( sub out 0 ny,
              Array.map fst derived,
              Array.map dual derived,
              sub out (ny + nr) (na + nt) )
          else
            let out = outward ~residuals:true ~applied:pullback in
            let res = sub out ny nr in
            (sub out 0 ny, res, res, sub out (ny + nr) (na + nt))
        in
        let ts = tangents c args in
        match derivative with
        | Tangent _ when reverse && not (Array.exists tangible_v ys) ->
            (* No result has a tangent: the tangent map observes tangents only,
               and a tape has none to observe. *)
            Array.concat [ ys; res_out; adds ]
        | Tangent t ->
            (* A tangent under reverse mode has no value to offer. *)
            if
              reverse && Program.observed t
              && Program.emits_from t
                   (Array.init (Array.length t.params) (fun k ->
                        k < Array.length ts))
            then
              invalid_arg
                "Rune.Probe.emit: a tangent is emitted under reverse mode";
            let out = eval t (Array.append ts res) in
            let own = sub out ny nt in
            if reverse then no_tangent_added c own;
            (* The additions of the tangent maps applied outward, then this
               one's. *)
            let adds =
              Array.mapi
                (fun k v -> if k < na then v else add v own.(k - na))
                adds
            in
            Array.concat [ with_tangents c ys (sub out 0 ny); res_out; adds ]
        | Pullback b ->
            (* The results' tangents are the pullback's linear call over the
               arguments' tangents, which the tape around records. Where the
               tangents are a derived program's variables, as in a loop's step,
               the tape's partial evaluation of the program records it. *)
            if not (Array.exists tangible_v ys) then
              Array.concat [ ys; res_out; adds ]
            else
              let dys =
                perform
                  (Linear
                     {
                       name;
                       apply = None;
                       transpose = Traced b;
                       args = ts;
                       captures = [||];
                       residuals = res;
                       forms = Array.map form ys;
                     })
              in
              Array.concat [ with_tangents c ys dys; res_out; adds ])

  (* A linear call: its results' primals are the call performed outward on the
     operands' primals, and their tangents the call on the arguments' tangents,
     performed outward so that a tape records it as one linear operation, plus
     the derivative of [apply]'s code along the captures' tangents. *)
  let linear c ~name ~apply ~transpose ~args ~captures ~residuals ~forms =
    let ny = Array.length forms in
    let call args =
      perform
        (Linear
           {
             name;
             apply = Some apply;
             transpose;
             args;
             captures = Array.map (primal_v c.i) captures;
             residuals = Array.map (primal_v c.i) residuals;
             forms;
           })
    in
    let out = call (Array.map (primal_v c.i) args) in
    let along_args =
      if not (Array.exists (owned c.i) args) then None
      else
        (* The function applied to tangents and derived again computes what the
           call over primals did: it offers nothing more. *)
        Some (sub (Program.silenced (fun () -> call (tangents c args))) 0 ny)
    in
    let along_captures =
      if not (Array.exists (owned c.i) captures) then None
      else
        let derived =
          Program.silenced (fun () ->
              c.derive apply
                (Array.append (Array.map (primal_v c.i) args) captures))
        in
        Some (Array.map snd (sub derived 0 ny))
    in
    let tangent k =
      let of_captures = Option.bind along_captures (fun ts -> ts.(k)) in
      match (Option.map (fun ts -> ts.(k)) along_args, of_captures) with
      | Some a, Some c -> Some (add a c)
      | Some a, None -> Some a
      | None, c -> c
    in
    Array.mapi
      (fun k y ->
        if k >= ny || not (tangible_v y) then y
        else match tangent k with Some t -> dual_v c.i y t | None -> y)
      out

  (* A custom rule under a differentiation that tracks values it captures and
     none of its arguments. The call is performed outward on the captures'
     primals, so that a differentiation around tracking an argument applies
     the rule; this one's tangents are the derivative of the primal's code
     along the captures, the residuals' too where they are asked for. A sum's
     tangent is its seed's plus those of the additions made onto it, which the
     derived code makes from the identity. *)
  let captured c ~name ~primal:p ~derivative ~ny ~args ~captures ~residuals
      ~applied ~seeds =
    let out =
      perform
        (Custom
           {
             name;
             primal = p;
             derivative;
             ny;
             args;
             captures = Array.map (primal_v c.i) captures;
             residuals;
             applied;
             seeds = Array.map (primal_v c.i) seeds;
           })
    in
    let base = ny + residual_count p ~ny in
    let slice = if residuals then base else ny in
    let derived =
      c.derive (prune p ~outputs:(Array.init slice Fun.id))
        (Array.append args captures)
    in
    let _, forms, feeds = sums [ p ] in
    let sum =
      Array.init (Array.length forms) (fun j ->
          if seeds = [||] then None else tangent_v c.i seeds.(j))
    in
    Array.iteri
      (fun k j ->
        match (sum.(j), snd derived.(slice + k)) with
        | Some a, Some b -> sum.(j) <- Some (add a b)
        | None, b -> sum.(j) <- b
        | Some _, None -> ())
      (List.hd feeds);
    let along k =
      if k < slice then snd derived.(k)
      else if k < base || k >= base + Array.length sum then None
      else sum.(k - base)
    in
    Array.mapi
      (fun k v -> match along k with Some t -> dual_v c.i v t | None -> v)
      out

  (* The tangent map applied to a tape's variables, refusing an addition of one
     where it is made. *)
  let taped_map c d ts =
    let effc (type a) (e : a Effect.t) :
        ((a, _) Effect.Deep.continuation -> _) option =
      match e with
      | Construct (Add (_, x)) when taped c (V x) ->
          Some (fun k -> Effect.Deep.discontinue k (Invalid_argument added))
      | Replace (_, x) when taped c (V x) ->
          Some (fun k -> Effect.Deep.discontinue k (Invalid_argument added))
      | _ -> None
    in
    Effect.Deep.match_with d ts { retc = Fun.id; exnc = reraise; effc }

  (* A custom rule run on the arguments' primals where the call was performed,
     under a differentiation that tracks one of them and is the innermost
     interpretation there, so that no interpretation inside it holds a value
     the rule captures. The tangent map runs on the tangents: values under
     forward mode, the tape's variables under reverse mode. The pullback runs in
     the backward pass, on its cotangents: the tape holds it as a linear call
     with the pullback as its transpose. Neither runs where no differentiation
     applies it: under reverse mode where no result has a tangent, and a
     pullback under forward mode. *)
  let in_place c ({ name; mode; args; rule } : Program.rule) pargs =
    let ys, d = rule pargs in
    tracks c ~name ys;
    let reverse = Option.is_some c.tape in
    let tangible = Array.exists tangible_v ys in
    match mode with
    | Reverse when not reverse ->
        if tangible then no_forward c;
        ys
    | Forward | Reverse when reverse && not tangible -> ys
    | Forward ->
        let ts = tangents c args in
        let dys = if reverse then taped_map c d ts else d ts in
        tracks c ~name dys;
        with_tangents c ys dys
    | Reverse ->
        let dys =
          perform
            (Linear
               {
                 name;
                 apply = None;
                 transpose = Host d;
                 args = tangents c args;
                 captures = [||];
                 residuals = [||];
                 forms = Array.map form ys;
               })
        in
        with_tangents c ys dys

  (* How a differentiation answers a custom rule's call ({!Program.Rule}): in
     place where it tracks an argument whose primal, as every other argument,
     no interpretation traces. *)
  let on_values c (r : Program.rule) =
    let untraced (V x) = Option.is_none (Nx.Prim.owner x) in
    if not (Array.exists (owned c.i) r.args) then None
    else
      let pargs = Array.map (primal_v c.i) r.args in
      if not (Array.for_all untraced pargs) then None
      else Some (fun () -> in_place c r pargs)

  (* A rule applies under a differentiation that tracks one of its arguments.
     One that tracks only values the rule captures differentiates the rule's
     code: where it is evaluated, or, where some differentiation around may
     track an argument, as [captured] does. *)
  let rule c =
    let traced (V x) = Option.is_some (Nx.Prim.owner x) in
    function
    | Custom
        {
          name;
          primal;
          derivative;
          ny;
          args;
          captures;
          residuals;
          applied;
          seeds;
        } ->
        if Array.exists (owned c.i) args then
          Some
            (custom c ~name ~primal ~derivative ~ny ~args ~captures ~residuals
               ~applied ~seeds)
        else if Array.exists (owned c.i) captures && Array.exists traced args
        then
          Some
            (captured c ~name ~primal ~derivative ~ny ~args ~captures
               ~residuals ~applied ~seeds)
        else None
    | Linear
        {
          name;
          apply = Some apply;
          transpose;
          args;
          captures;
          residuals;
          forms;
        } ->
        Some
          (linear c ~name ~apply ~transpose ~args ~captures ~residuals ~forms)
    | _ -> None
end

(* Programs' derivatives

   A loop or a remat over duals is performed outward over its primals and
   tangents with its program's derivative: [eval p] traced under a forward-mode
   interpretation of its own, as JAX derives a jaxpr's. The derivative's
   additions are its primal's, made once that interpretation has returned, so a
   total receives primals, as [Add] of a dual gives it. *)

let pv i (Program.V x) = Program.V (primal i x)
let tv i (Program.V x) = Option.map (fun t -> Program.V t) (tangent i x)
let somes a = Array.of_list (List.filter_map Fun.id (Array.to_list a))

(* The values [vs] put back at the positions where [mask] holds. *)
let spread mask vs =
  let n = ref (-1) in
  Array.map
    (fun h ->
      if h then begin
        incr n;
        Some vs.(!n)
      end
      else None)
    mask

(* [k x t] with [x] and [t] at one type: [x]'s where [t] reads at it, else
   [t]'s. *)
type both = {
  k : 'v 's 'd. ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t -> Program.value;
}

let both (Program.V x) (Program.V t) b =
  let set y = Nx.Placement.devices (Nx.placement y) in
  match Prim.coerce (Nx.dtype x) (set x) t with
  | Some t -> b.k x t
  | None -> (
      match Prim.coerce (Nx.dtype t) (set t) x with
      | Some x -> b.k x t
      | None -> invalid_arg "Rune: a tangent of another dtype or set")

let seeded j v = function
  | Some t -> both v t { k = (fun x t -> Program.V (seed j x t)) }
  | None -> v

let dualv i ~fresh v = function
  | Some t -> both v t { k = (fun y t -> Program.V (dual i ~fresh y t)) }
  | None -> v

(* [ys]'s primals and tangents as [j] holds them. *)
let parts j ys = (Array.map (pv j) ys, Array.map (tv j) ys)

(* Whether a result of the form has a derivative. *)
let differentiable (Program.F f) =
  match Nx.Dtype.kind f.dtype with
  | Float | Complex -> true
  | Signed | Unsigned | Boolean -> false

(* A construct over duals: [Detach], an addition and an emission are performed
   outward on the primal, so that every differentiation around detaches the
   value, a total receives the primal and a probe is offered it; an addition to
   a running value adds the primals, and its tangent the tangents. [Lanes] is
   linear: its tangent is the lanes of the tangent. A custom rule or a linear
   call applies its rule ([Rules]). A scan, a remat, a call, an iterate, a
   switch, a root and a solve are performed with their programs' derivatives. *)
let rec answer ~name ~tape i (c : Program.construct) k =
  match c with
  | Detach x -> Effect.Deep.continue k (Program.perform (Detach (primal i x)))
  | Add (t, x) ->
      Program.add_to t (primal i x);
      Effect.Deep.continue k [||]
  | Accumulate { lanes; running; x } ->
      let y =
        Program.perform
          (Accumulate { lanes; running = primal i running; x = primal i x })
      in
      let y = Program.like running y.(0) in
      (* Linear: the running value's tangent plus the rows of [x]'s, which a
         collect inside the differentiation receives. *)
      let y =
        match (tangent i running, tangent i x) with
        | None, None -> y
        | Some tr, None -> dual i ~fresh:false y tr
        | tr, Some tx ->
            let tr =
              match tr with
              | Some t -> t
              | None ->
                  Program.like running
                    (Loops.filled ~zero:false
                       (Program.form (Program.V (primal i running))))
            in
            dual i ~fresh:false y (Program.added ~lanes tr tx)
      in
      Effect.Deep.continue k [| Program.V y |]
  | Emit _ ->
      let primals = Program.map_operands { f = (fun x -> primal i x) } c in
      ignore (Program.perform primals);
      Effect.Deep.continue k [||]
  | Lanes (a, x) ->
      let lanes x = Program.like x (Program.perform (Lanes (a, x))).(0) in
      let y = lanes (primal i x) in
      let y =
        match tangent i x with
        | Some t -> dual i ~fresh:false y (lanes t)
        | None -> y
      in
      Effect.Deep.continue k [| Program.V y |]
  | Custom _ | Linear _ -> (
      match
        Rules.rule { i; entry = name; tape; derive = derive ~name ~tape i } c
      with
      | Some r -> Effect.Deep.continue k r
      | None -> Effect.Deep.discontinue k Program.Evaluate)
  | Scan { step; length; trips; reverse; init; xs; captures; seeds } ->
      Effect.Deep.continue k
        (scan ~name ~tape i step ~length ~trips ~reverse ~init ~xs ~captures
           ~seeds)
  | Remat { body; args; seeds; differentiated } ->
      Effect.Deep.continue k
        (remat ~name ~tape i body args seeds ~differentiated)
  | Call { body; args; site; seeds } ->
      Effect.Deep.continue k (call ~name ~tape i body args site seeds)
  | Iterate _ -> Effect.Deep.continue k (iterate ~name ~tape i c)
  (* An index no interpretation traces picks the branch here: it runs inside
     this differentiation, as if written inline. *)
  | Switch { index; _ }
    when let (V x) = pv i index in
         Option.is_none (Nx.Prim.owner x) ->
      Effect.Deep.discontinue k Program.Evaluate
  | Switch _ -> Effect.Deep.continue k (switch ~name ~tape i c)
  (* An operator is linear in its arguments: its tangent is the operator of
     their tangents. Its data are values outside the transformations the linear
     solve opens, which track none of them. *)
  | Operator o ->
      let data = Array.map (pv i) o.data in
      let apply args = Program.perform (Operator { o with args; data }) in
      let p = apply (Array.map (pv i) o.args) in
      let ts =
        Array.map2
          (fun a f ->
            match tv i a with Some t -> t | None -> Loops.filled ~zero:true f)
          o.args o.forms
      in
      let t = apply ts in
      Effect.Deep.continue k
        (Array.map2 (fun p t -> dualv i ~fresh:false p (Some t)) p t)
  (* A root's derivative offers nothing. *)
  | Root _ ->
      Effect.Deep.continue k (Program.silenced (fun () -> root ~name ~tape i c))
  | Solve _ ->
      Effect.Deep.continue k
        (Program.silenced (fun () -> solve ~name ~tape i c))
  | Lane_index _ -> Effect.Deep.discontinue k Program.Evaluate
  (* A callback has no derivative: it raises where a tracked argument may reach
     a result that has one, and otherwise runs on the primals. *)
  | Callback cb ->
      let tracked = Array.exists (fun a -> Option.is_some (tv i a)) cb.args in
      if tracked && Array.exists differentiable cb.forms then
        Effect.Deep.discontinue k
          (Invalid_argument
             ("Rune.callback: " ^ name
            ^ " tracks an argument of a call with a float result; a custom \
               rule around the call gives its derivative"))
      else
        let args = Array.map (pv i) cb.args in
        Effect.Deep.continue k (Program.perform (Callback { cb with args }))

(* A scan's derivative is the scan of its step's derivative, whose carry has a
   tangent for each carry that has one at the call or gains one from the step: a
   fixpoint, as JAX's scan finds its nonzero tangents. A carry that gains one
   starts from a zero, [-0.], which keeps the sign of what is added to it. Each
   run of the derived step's parameters, the carry, the row and the captures,
   holds the primals then the tangents. The tangents of a sum's additions are
   one more carry, the sum's tangent, from its seed's: the derived step's
   additions are primals, which a total outside the differentiation receives,
   and a total inside it receives the sums as duals. *)
and scan ~name ~tape i (step : Program.t) ~length ~trips ~reverse ~init ~xs
    ~captures ~seeds =
  let nc = Array.length init in
  let ny = Array.length step.outputs - nc - Array.length step.adds in
  let _, sum_forms, feeds = Program.feeds step in
  let ns = Array.length sum_forms in
  let pinit = Array.map (pv i) init and pxs = Array.map (pv i) xs in
  let pcaps = Array.map (pv i) captures in
  let txs = Array.map (tv i) xs and tcaps = Array.map (tv i) captures in
  let zero v = Loops.filled ~zero:false (Program.form v) in
  let tinit k =
    match tv i init.(k) with Some t -> t | None -> zero pinit.(k)
  in
  let tseed s =
    match if seeds = [||] then None else tv i seeds.(s) with
    | Some t -> t
    | None -> Loops.filled ~zero:false sum_forms.(s)
  in
  (* [has] and [hs] are the carries and sums with a tangent; [pf], [tf] and [sf]
     the forms of the primal and tangent carries and the sums' tangents. *)
  let rec fix has hs pf tf sf =
    let gained = Array.make nc false and gained_s = Array.make ns false in
    let has_y = ref [||] in
    let rows = Array.map Loops.row in
    let forms =
      Array.concat
        [
          pf;
          Loops.pick has tf;
          Loops.pick hs sf;
          rows pxs;
          rows (somes txs);
          Array.map Program.form pcaps;
          Array.map Program.form (somes tcaps);
        ]
    in
    let count m = Array.length (Loops.pick m m) in
    let lengths =
      [
        nc;
        count has;
        count hs;
        Array.length pxs;
        Array.length (somes txs);
        Array.length pcaps;
        Array.length (somes tcaps);
      ]
    in
    let derive vs =
      match Loops.split lengths vs with
      | [ pc; tc; ts; px; tx; pk; tk ] ->
          (* The step's totals run from the trace's running values, a sum with a
             tangent from that tangent's carry, so that its additions chain onto
             it in turn. *)
          let tsums = spread hs ts and derived = ref None in
          let (_ : Program.value array) =
            Program.threaded step (fun running ->
                run ~name ?tape (fun j ->
                    let duals ps ts mask =
                      Array.map2 (seeded j) ps
                        (spread (Array.map Option.is_some mask) ts)
                    in
                    let args =
                      Array.concat
                        [
                          Array.map2 (seeded j) pc (spread has tc);
                          duals px tx txs;
                          duals pk tk tcaps;
                        ]
                    in
                    let totals =
                      Array.mapi
                        (fun k v -> seeded j v tsums.(feeds.(k)))
                        running
                    in
                    let p, t = parts j (Program.eval ~totals step args) in
                    let n = Array.length p - nc - ny in
                    derived :=
                      Some
                        ( Array.sub p 0 nc,
                          Array.sub t 0 nc,
                          Array.sub p nc ny,
                          Array.sub t nc ny,
                          Array.sub p (nc + ny) n,
                          Array.sub t (nc + ny) n );
                    Array.sub p (nc + ny) n))
          in
          let c, tc, y, ty, sums, tsums' = Option.get !derived in
          (* A carry with a tangent whose step gives it none takes a zero. *)
          let tc' =
            Array.init nc (fun k ->
                gained.(k) <- (not has.(k)) && Option.is_some tc.(k);
                match tc.(k) with
                | Some t -> Some t
                | None -> if has.(k) then Some (zero c.(k)) else None)
          in
          let tc' = somes (Loops.pick has tc') in
          (* Each sum's tangent, its running value's: the carry's, plus the
             tangents of the additions made onto it. *)
          let sums_t = Array.make ns None in
          Array.iteri
            (fun k t ->
              let s = feeds.(k) in
              sums_t.(s) <-
                (match t with
                | Some _ -> t
                | None -> if hs.(s) then Some (zero sums.(k)) else None);
              if Option.is_some t && not hs.(s) then gained_s.(s) <- true)
            tsums';
          let sums_t = somes (Loops.pick hs sums_t) in
          has_y := Array.map Option.is_some ty;
          Array.concat [ c; tc'; sums_t; y; somes ty ]
      | _ -> invalid_arg "Rune: a scan's derivative"
    in
    let prog, own = Trace.trace Body ~name forms derive in
    if Array.exists Fun.id gained || Array.exists Fun.id gained_s then
      let has = Array.map2 ( || ) has gained in
      fix has (Array.map2 ( || ) hs gained_s) pf tf sf
    else (prog, own, has, hs, !has_y)
  in
  let has = Array.map (fun v -> Option.is_some (tv i v)) init in
  let hs =
    Array.init ns (fun s -> seeds <> [||] && Option.is_some (tv i seeds.(s)))
  in
  let tf = Array.init nc (fun k -> Program.form (tinit k)) in
  let sf = Array.init ns (fun s -> Program.form (tseed s)) in
  let prog, own, has, hs, has_y =
    fix has hs (Array.map Program.form pinit) tf sf
  in
  let r =
    Program.perform
      (Scan
         {
           step = prog;
           length;
           trips = Option.map (pv i) trips;
           reverse;
           init =
             Array.concat
               [
                 pinit;
                 Loops.pick has (Array.init nc tinit);
                 Loops.pick hs (Array.init ns tseed);
               ];
           xs = Array.append pxs (somes txs);
           captures = Array.concat [ pcaps; somes tcaps; own ];
           seeds = Array.map (pv i) seeds;
         })
  in
  let count m = Array.length (Loops.pick m m) in
  let nt = count has and nts = count hs and nty = count has_y in
  match Loops.split [ nc; nt; nts; ny; nty; ns ] r with
  | [ c; tc; ts; y; ty; sums ] ->
      Array.concat
        [
          Array.map2 (dualv i ~fresh:false) c (spread has tc);
          Array.map2 (dualv i ~fresh:true) y (spread has_y ty);
          Array.map2 (dualv i ~fresh:false) sums (spread hs ts);
        ]
  | _ -> invalid_arg "Rune: a scan's derivative"

(* A remat's derivative is the remat of its function's, as JAX's checkpoint
   differentiates, so that a reverse mode around it sees the remat. The tangents
   of its sums are more outputs: its sums are duals, as its function's running
   totals would be, each from its seed's tangent. *)
and remat ~name ~tape i (body : Program.t) args seeds ~differentiated =
  let pargs = Array.map (pv i) args and targs = Array.map (tv i) args in
  let tseeds = Array.map (tv i) seeds in
  let na = Array.length body.adds in
  let no = Array.length body.outputs - na in
  let has_y = ref [||] in
  let forms =
    Array.concat
      [
        Array.map Program.form pargs;
        Array.map Program.form (somes targs);
        Array.map Program.form (somes tseeds);
      ]
  in
  let count m = Array.length (somes m) in
  let derive vs =
    match Loops.split [ Array.length pargs; count targs; count tseeds ] vs with
    | [ ps; ts; tss ] ->
        let ys = ref [||] in
        let tss =
          if seeds = [||] then Array.make na None
          else spread (Array.map Option.is_some tseeds) tss
        in
        let (_ : Program.value array) =
          Program.threaded body (fun totals ->
              run ~name ?tape (fun j ->
                  let mask = Array.map Option.is_some targs in
                  let args = Array.map2 (seeded j) ps (spread mask ts) in
                  let totals = Array.map2 (seeded j) totals tss in
                  let p, t = parts j (Program.eval ~totals body args) in
                  has_y := Array.map Option.is_some t;
                  ys := Array.append (Array.sub p 0 no) (somes t);
                  Array.sub p no na))
        in
        !ys
    | _ -> invalid_arg "Rune: a remat's derivative"
  in
  (* Traced as the remat's function was: every operation staged. *)
  let prog, own = Trace.trace Staged ~name forms derive in
  let r =
    Program.perform
      (Remat
         {
           body = prog;
           args = Array.concat [ pargs; somes targs; somes tseeds; own ];
           seeds = Array.map (pv i) seeds;
           differentiated;
         })
  in
  let nt = Array.length (Loops.pick !has_y !has_y) in
  match Loops.split [ no; nt; na ] r with
  | [ y; t; adds ] ->
      Array.map2 (dualv i ~fresh:false) (Array.append y adds) (spread !has_y t)
  | _ -> invalid_arg "Rune: a remat's derivative"

(* A call's derivative is the call of its program's derivative, as JAX
   differentiates a compiled call's jaxpr: derived once per derivation at the
   call's site, traced as a compiled function is, and compiled. Its additions
   are duals, as a remat's are. *)
and call ~name ~tape i (body : Program.t) args site seeds =
  let pargs = Array.map (pv i) args and targs = Array.map (tv i) args in
  let tseeds = Array.map (tv i) seeds in
  let mask = Array.map Option.is_some targs in
  let smask = Array.map Option.is_some tseeds in
  let ins = Array.concat [ pargs; somes targs; somes tseeds ] in
  let forms = Array.map Program.form ins in
  let na = Array.length body.adds in
  let no = Array.length body.outputs - na in
  let count m = Array.length (somes m) in
  let made () =
    let has_y = ref [||] in
    let derive vs =
      match
        Loops.split [ Array.length pargs; count targs; count tseeds ] vs
      with
      | [ ps; ts; tss ] ->
          let ys = ref [||] in
          let tss =
            if seeds = [||] then Array.make na None else spread smask tss
          in
          let (_ : Program.value array) =
            Program.threaded body (fun totals ->
                run ~name ?tape (fun j ->
                    let args = Array.map2 (seeded j) ps (spread mask ts) in
                    let totals = Array.map2 (seeded j) totals tss in
                    let p, t = parts j (Program.eval ~totals body args) in
                    has_y := Array.map Option.is_some t;
                    ys := Array.append (Array.sub p 0 no) (somes t);
                    Array.sub p no na))
          in
          !ys
      | _ -> invalid_arg "Rune: a call's derivative"
    in
    let prog, _ = Trace.trace Jit ~name forms derive in
    let totals =
      Array.map (fun (Program.Addition (_, f, _)) -> Program.F f) prog.adds
    in
    ((prog, Array.append forms totals), !has_y, None)
  in
  let reverse = Option.is_some tape && Program.customs site in
  (* The seeds' tangents enter the derivation beside the arguments'. *)
  let derivation =
    Program.Jvp { tangents = Array.append mask smask; reverse }
  in
  let d = Program.derive site derivation forms made in
  let r =
    Program.perform
      (Call
         {
           body = d.program;
           args = ins;
           site = d.at;
           seeds = Array.map (pv i) seeds;
         })
  in
  let nt = Array.length (Loops.pick d.flags d.flags) in
  match Loops.split [ no; nt; na ] r with
  | [ y; t; adds ] ->
      Array.map2 (dualv i ~fresh:false) (Array.append y adds) (spread d.flags t)
  | _ -> invalid_arg "Rune: a call's derivative"

(* A switch's derivative is the switch, on the same index, of its branches'
   derivatives, as JAX derives a cond: each branch's results and their tangents,
   a zero where another branch has one and this one none, and its sums'
   tangents, those its branch does not add to their seeds'. The index has no
   tangent. *)
and switch ~name ~tape i (c : Program.construct) =
  match c with
  | Switch sw -> (
      let branches = sw.branches in
      let pcaps = Array.map (pv i) sw.captures
      and tcaps = Array.map (tv i) sw.captures in
      let tseeds = Array.map (tv i) sw.seeds in
      let _, sum_forms, feeds = Program.sums (Array.to_list branches) in
      let ns = Array.length sum_forms in
      let results = Program.switched branches in
      let no = Array.length results in
      let mask = Array.map Option.is_some tcaps in
      let forms =
        Array.concat
          [
            Array.map Program.form pcaps;
            Array.map Program.form (somes tcaps);
            Array.map Program.form (somes tseeds);
          ]
      in
      let count m = Array.length (somes m) in
      (* Branch [k]'s results, their tangents, and its sums' tangents, the
         primals' additions made on the running values; [keep] the tangents
         every branch outputs, a zero where this one has none. *)
      let derive k ~keep vs =
        match
          Loops.split [ Array.length pcaps; count tcaps; count tseeds ] vs
        with
        | [ ps; ts; tss ] ->
            let b = branches.(k) and feeds = List.nth feeds k in
            let tss =
              if sw.seeds = [||] then Array.make ns None
              else spread (Array.map Option.is_some tseeds) tss
            in
            let out = ref [||] and has = ref [||] in
            let (_ : Program.value array) =
              Program.threaded b (fun totals ->
                  run ~name ?tape (fun j ->
                      let args = Array.map2 (seeded j) ps (spread mask ts) in
                      let na = Array.length b.adds in
                      let totals =
                        Array.mapi
                          (fun a t -> seeded j t tss.(feeds.(a)))
                          totals
                      in
                      let p, t = parts j (Program.eval ~totals b args) in
                      let st = Array.copy tss in
                      Array.iteri (fun a s -> st.(s) <- t.(no + a)) feeds;
                      let t = Array.append (Array.sub t 0 no) st in
                      has := Array.map Option.is_some t;
                      let zero k =
                        Loops.filled ~zero:true
                          (if k < no then results.(k) else sum_forms.(k - no))
                      in
                      out :=
                        Array.append (Array.sub p 0 no)
                          (somes
                             (Array.mapi
                                (fun k t ->
                                  if not keep.(k) then None
                                  else
                                    Some
                                      (match t with
                                      | Some t -> t
                                      | None -> zero k))
                                t));
                      Array.sub p no na))
            in
            (!out, !has)
        | _ -> invalid_arg "Rune: a switch's derivative"
      in
      (* The tangents some branch has, from a first trace of each. *)
      let all = Array.make (no + ns) true in
      let keep =
        Array.fold_left
          (fun acc k ->
            let _ =
              Trace.trace Staged ~name forms (fun vs ->
                  let out, has = derive k ~keep:all vs in
                  Array.iteri (fun j h -> if h then acc.(j) <- true) has;
                  out)
            in
            acc)
          (Array.make (no + ns) false)
          (Array.init (Array.length branches) Fun.id)
      in
      let traced =
        Array.mapi
          (fun k _ ->
            Trace.trace Staged ~name forms (fun vs -> fst (derive k ~keep vs)))
          branches
      in
      let progs, own = Program.unite traced ~params:(Array.length forms) in
      let r =
        Program.perform
          (Switch
             {
               sw with
               branches = progs;
               captures = Array.concat [ pcaps; somes tcaps; somes tseeds; own ];
               seeds = Array.map (pv i) sw.seeds;
             })
      in
      let nt = Array.length (Loops.pick keep keep) in
      match Loops.split [ no; nt; ns ] r with
      | [ y; t; sums ] ->
          Array.map2 (dualv i ~fresh:false) (Array.append y sums)
            (spread keep t)
      | _ -> invalid_arg "Rune: a switch's derivative")
  | _ -> invalid_arg "Rune: a switch's derivative"

(* An iterate's derivative is an iterate whose carry holds the primal carry, the
   tangents of the carries that have one or a step gives one, a fixpoint, the
   tangents of the sums that have one, and the stop. The stop has no derivative:
   it is tested on the primals, once before the loop and after each step, inside
   it, so that every test's additions are chained with their tangents as the
   steps' are, and the loop's own stop reads it. *)
and iterate ~name ~tape i (c : Program.construct) =
  match c with
  | Iterate it -> (
      let nc = Array.length it.init in
      let ny = Array.length it.step.outputs - nc - Array.length it.step.adds in
      let sum_adds, sum_forms, feeds = Program.sums [ it.until; it.step ] in
      let until_feeds = List.nth feeds 0 and step_feeds = List.nth feeds 1 in
      let ns = Array.length sum_adds in
      let pinit = Array.map (pv i) it.init in
      let psc = Array.map (pv i) it.step_captures
      and tsc = Array.map (tv i) it.step_captures in
      let puc = Array.map (pv i) it.until_captures
      and tuc = Array.map (tv i) it.until_captures in
      let zero v = Loops.filled ~zero:false (Program.form v) in
      (* Each sum's running value before the loop, from its seed, and the stop
         at [init]. *)
      let seeds =
        if it.seeds = [||] then Array.map (Loops.filled ~zero:false) sum_forms
        else it.seeds
      in
      let tested =
        derive ~name ~tape i
          ~totals:(Array.map (fun j -> seeds.(j)) until_feeds)
          it.until
          (Array.append it.init it.until_captures)
      in
      let stop0 = fst tested.(0) in
      let sums0 = Array.map (fun v -> (pv i v, tv i v)) seeds in
      Array.iteri (fun k j -> sums0.(j) <- tested.(1 + k)) until_feeds;
      let tinit k =
        match tv i it.init.(k) with Some t -> t | None -> zero pinit.(k)
      in
      let tsum s =
        match snd sums0.(s) with Some t -> t | None -> zero (fst sums0.(s))
      in
      (* [pf], [tf] and [sf] are the forms of the primal and tangent carries and
         the sums' tangents. *)
      let rec fix has hs pf tf sf =
        let gained = Array.make nc false and gained_s = Array.make ns false in
        let has_y = ref [||] in
        let count m = Array.length (Loops.pick m m) in
        let carried =
          Array.concat
            [
              pf; Loops.pick has tf; Loops.pick hs sf; [| Program.form stop0 |];
            ]
        in
        let forms =
          Array.concat
            [
              carried;
              Array.map Program.form psc;
              Array.map Program.form (somes tsc);
              Array.map Program.form puc;
              Array.map Program.form (somes tuc);
            ]
        in
        let lengths =
          [
            nc;
            count has;
            count hs;
            1;
            Array.length psc;
            Array.length (somes tsc);
            Array.length puc;
            Array.length (somes tuc);
          ]
        in
        let step vs =
          match Loops.split lengths vs with
          | [ pc; tc; ts; _; ps; ts_c; pu; tu ] ->
              let tsums = spread hs ts in
              Program.threading sum_adds (fun running ->
                  run ~name ?tape (fun j ->
                      let duals ps ts mask =
                        Array.map2 (seeded j) ps
                          (spread (Array.map Option.is_some mask) ts)
                      in
                      let acc =
                        Array.mapi (fun s v -> seeded j v tsums.(s)) running
                      in
                      let c = Array.map2 (seeded j) pc (spread has tc) in
                      let totals feeds = Array.map (fun s -> acc.(s)) feeds in
                      let out =
                        Program.eval ~totals:(totals step_feeds) it.step
                          (Array.append c (duals ps ts_c tsc))
                      in
                      Array.iteri
                        (fun k s -> acc.(s) <- out.(nc + ny + k))
                        step_feeds;
                      let c' = Array.sub out 0 nc in
                      let test =
                        Program.eval ~totals:(totals until_feeds) it.until
                          (Array.append c' (duals pu tu tuc))
                      in
                      Array.iteri
                        (fun k s -> acc.(s) <- test.(1 + k))
                        until_feeds;
                      let p, t = parts j c' in
                      let yp, yt = parts j (Array.sub out nc ny) in
                      let sp, st = parts j acc in
                      let tc' =
                        Array.init nc (fun k ->
                            gained.(k) <- (not has.(k)) && Option.is_some t.(k);
                            match t.(k) with
                            | Some t -> Some t
                            | None ->
                                if has.(k) then Some (zero p.(k)) else None)
                      in
                      let ts' =
                        Array.init ns (fun s ->
                            gained_s.(s) <-
                              (not hs.(s)) && Option.is_some st.(s);
                            match st.(s) with
                            | Some t -> Some t
                            | None ->
                                if hs.(s) then Some (zero sp.(s)) else None)
                      in
                      has_y := Array.map Option.is_some yt;
                      let carry =
                        Array.concat
                          [
                            p;
                            somes (Loops.pick has tc');
                            somes (Loops.pick hs ts');
                            [| pv j test.(0) |];
                          ]
                      in
                      (sp, Array.concat [ carry; yp; somes yt ])))
          | _ -> invalid_arg "Rune: an iterate's derivative"
        in
        let prog, own = Trace.trace Body ~name forms step in
        if Array.exists Fun.id gained || Array.exists Fun.id gained_s then
          fix
            (Array.map2 ( || ) has gained)
            (Array.map2 ( || ) hs gained_s)
            pf tf sf
        else (prog, own, has, hs, !has_y)
      in
      let has = Array.map (fun v -> Option.is_some (tv i v)) it.init in
      let hs = Array.map (fun (_, t) -> Option.is_some t) sums0 in
      let prog, own, has, hs, has_y =
        fix has hs
          (Array.map Program.form pinit)
          (Array.init nc (fun k -> Program.form (tinit k)))
          (Array.init ns (fun s -> Program.form (tsum s)))
      in
      let init =
        Array.concat
          [
            pinit;
            Loops.pick has (Array.init nc tinit);
            Loops.pick hs (Array.init ns tsum);
            [| stop0 |];
          ]
      in
      let until, uown =
        Trace.trace Body ~name (Array.map Program.form init) (fun vs ->
            [| vs.(Array.length vs - 1) |])
      in
      let r =
        Program.perform
          (Iterate
             {
               it with
               step = prog;
               until;
               init;
               step_captures =
                 Array.concat [ psc; somes tsc; puc; somes tuc; own ];
               until_captures = uown;
               seeds = Array.map fst sums0;
             })
      in
      let count m = Array.length (Loops.pick m m) in
      let nt = count has and nts = count hs and nty = count has_y in
      let stacked = Option.is_some it.rows in
      let lengths =
        [ nc; nt; nts; 1 ]
        @ (if stacked then [ 1; ny; nty ] else [ 0; 0; 0 ])
        @ [ ns ]
      in
      match Loops.split lengths r with
      | [ c; tc; ts; _; trips; y; ty; sums ] ->
          Array.concat
            [
              Array.map2 (dualv i ~fresh:false) c (spread has tc);
              trips;
              Array.map2 (dualv i ~fresh:true) y (spread has_y ty);
              Array.map2 (dualv i ~fresh:false) sums (spread hs ts);
            ]
      | _ -> invalid_arg "Rune: an iterate's derivative")
  | _ -> invalid_arg "Rune: an iterate's derivative"

(* The tangents [outs] hold where they have one, zeros of the primals' forms
   elsewhere. *)
and tangents_or_zeros outs =
  Array.map
    (fun (y, t) ->
      match t with
      | Some t -> t
      | None -> Loops.filled ~zero:true (Program.form y))
    outs

(* A root's derivative, as the implicit function theorem gives it at the
   solution, whether or not it is a zero: the [u] with [J u = -r], [r] the
   residual's tangent along the values it reads from outside, the solution held,
   and [J] its derivative along the solution. The solution is the root performed
   outward, so that a differentiation around takes this rule again, and [u] a
   linear solve, linear in [r], performed outward. *)
and root ~name ~tape i (r : Program.construct) =
  match r with
  | Root ({ system; solution; theta; captures; _ } as rt) ->
      let pth = Array.map (pv i) theta and pcaps = Array.map (pv i) captures in
      let ys =
        Program.perform
          (Root
             {
               rt with
               solution = Array.map (pv i) solution;
               theta = pth;
               captures = pcaps;
             })
      in
      let nx = Array.length ys in
      if not (Array.exists (owned i) theta) then ys
      else
        let r =
          Array.sub
            (derive ~name ~tape i system.residual (Array.append ys theta))
            0 nx
        in
        let b =
          Array.map
            (fun (Program.V t) -> Program.V (Nx.neg t))
            (tangents_or_zeros r)
        in
        let u =
          Program.perform
            (Solve
               {
                 name = rt.name;
                 system;
                 transposed = false;
                 b;
                 data = Array.append ys pth;
                 captures = pcaps;
               })
        in
        Array.map2 (fun y u -> dualv i ~fresh:false y (Some u)) ys u
  | _ -> invalid_arg "Rune: a root's derivative"

(* A linear solve's derivative, as JAX's custom_linear_solve's: of [x = S b],
   [S] solving [J x = b] at the data, [S (db - dJ x)], [dJ] the operator's
   derivative along the data's tangents, its argument [x] held. Both solves are
   performed outward, so that a differentiation around takes this rule again,
   and never differentiates the linear solve's code. *)
and solve ~name ~tape i (s : Program.construct) =
  match s with
  | Solve ({ system; transposed; b; data; captures; _ } as sv) ->
      let pdata = Array.map (pv i) data and pcaps = Array.map (pv i) captures in
      let x =
        Program.perform
          (Solve
             { sv with b = Array.map (pv i) b; data = pdata; captures = pcaps })
      in
      let nx = Array.length x in
      let db =
        Array.map2
          (fun v y ->
            match tv i v with
            | Some t -> t
            | None -> Loops.filled ~zero:true (Program.form y))
          b x
      in
      let along_data = Array.exists (owned i) data in
      if (not along_data) && not (Array.exists (owned i) b) then x
      else
        let rhs =
          if not along_data then db
          else
            let jacobian =
              if transposed then system.transpose ~name else system.jacobian
            in
            let dj =
              tangents_or_zeros
                (Array.sub
                   (derive ~name ~tape i jacobian (Array.append x data))
                   0 nx)
            in
            Array.map2
              (fun d (Program.V j) -> Loops.add d (Program.V (Nx.neg j)))
              db dj
        in
        let u =
          Program.perform
            (Solve { sv with b = rhs; data = pdata; captures = pcaps })
        in
        Array.map2 (fun y u -> dualv i ~fresh:false y (Some u)) x u
  | _ -> invalid_arg "Rune: a linear solve's derivative"

(* [p]'s outputs over [xs], each with its tangent, from a differentiation [j]
   like [i] seeded from [i]'s tangents of [xs]. *)
and derive ?totals ~name ~tape i p xs =
  run ~name ?tape (fun j ->
      let seeded (Program.V x as v) =
        match Prim.payload i x with
        | Some (Dual d) -> Program.V (seed j d.primal d.tangent)
        | _ -> v
      in
      let totals = Option.map (Array.map seeded) totals in
      let out = Program.eval ?totals p (Array.map seeded xs) in
      Array.map
        (fun (Program.V y) ->
          match Prim.payload j y with
          | Some (Dual d) -> (Program.V d.primal, Some (Program.V d.tangent))
          | _ -> (Program.V y, None))
        out)

and run :
    'a.
    name:string ->
    ?tape:Prim.interpretation ->
    (Prim.interpretation -> 'a) ->
    'a =
 fun ~name ?tape f ->
  let owner = ref None in
  let effc (type a) (e : a Effect.t) :
      ((a, _) Effect.Deep.continuation -> _) option =
    match (e, !owner) with
    | Program.Within, _ -> Some (fun k -> Effect.Deep.continue k true)
    | Program.Construct c, Some i
      when List.exists (owned i) (Program.operands c) ->
        Some (answer ~name ~tape i c)
    (* A callback runs outside the differentiation, whatever it reads. *)
    | Program.Construct (Callback _ as c), Some i ->
        Some (answer ~name ~tape i c)
    | Program.Rule r, Some i ->
        let c =
          { Rules.i; entry = name; tape; derive = derive ~name ~tape i }
        in
        Some (fun k -> Effect.Deep.continue k (Rules.on_values c r))
    (* A loop hands a total its chain's end as an addition gives it: the
       primal. *)
    | Program.Replace (t, x), Some i when owned i (Program.V x) ->
        Some
          (fun k ->
            Effect.perform (Program.Replace (t, primal i x));
            Effect.Deep.continue k ())
    | _ -> None
  in
  Effect.Deep.match_with
    (fun () ->
      Prim.interpret ~name Values
        (fun i ~by op -> rule name i ~by op)
        (fun i ->
          owner := Some i;
          f i))
    ()
    { retc = Fun.id; exnc = reraise; effc }

let tangent i x =
  match Prim.payload i x with
  | Some (Dual d) -> Some (if d.fresh then d.tangent else Nx.copy d.tangent)
  | _ -> None

let tangent_of i x =
  match Prim.payload i x with Some (Dual d) -> Some d.tangent | _ -> None
