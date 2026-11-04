(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module E = Nx_kernel.Prog.E
module Prim = Nx.Prim

(* Attention's derivative

   At [(q, k, v)] with the call's [o] and [lse], a pair's weight is
   [p = exp (x - lse)], [x] its score as the pair program rewrites
   [s = scale · q·k], and [p = 0] where it is hidden. Along tangents,
   [ẋ] is the program's tangent at [(s, ṡ = scale (q̇·k + q·k̇))] and the
   loads' tangents:

     l̇se = Σ_j p ẋ                ȯ = Σ_j p ẋ v - l̇se o + Σ_j p v̇

   and its transpose, from cotangents [ō] and [l̄se]:

     δ = ō·o - l̄se                dP = ō·v_j
     s̄ = p (dP - δ)               the program's transpose of [s̄] gives each
                                  read's: [r̄] of [s], [c̄] of each load
     q̄ = scale Σ_j r̄ k_j          k̄_j = scale Σ_i r̄ q_i     v̄_j = Σ_i p ō_i

   summed over a group's heads for the keys and values, and into the pools
   through the table where paged. A row whose [lse] is [-∞] has no visible
   pair: its weights, and so its cotangents, are zeros. These defaults
   compute it as contractions and maps of the whole pair loop. *)

type spec = {
  prog : P.t;
  scale : float;
  paged : bool;
  lengths : bool;
  tangents : bool array;
}

type eval = { eval : 'r. 'r Prim.t -> 'r } [@@unboxed]

let ia = Iarray.of_array
let contiguous s = L.contiguous (ia s)

(* [x] moved: reshaped, broadcast or its axes permuted. *)
let reshape e s x = e.eval (Move (Reshape (ia s), x))
let broadcast e s x = e.eval (Move (Broadcast (ia s), x))
let permute e p x = e.eval (Move (Permute (ia p), x))
let plain (type d) (loads : d Prim.any list) =
  ia (Array.of_list (List.map (fun (Prim.Any x) -> Prim.Plain x) loads))

(* The program [prog]'s one output, of [dt], over [loads]. *)
let map1 (type v s d) e (dt : (v, s) D.t) shape (prog : P.t)
    (loads : d Prim.any list) : (v, s, d) Nx.t =
  fst
    (e.eval
       (Map
          {
            layout = contiguous shape;
            prog;
            outs = Prim.[ dt ];
            loads = plain loads;
            at = None;
          }))

(* The map of the expression [out] over [loads], into [dt]. *)
let map (type v s d) e (dt : (v, s) D.t) shape (loads : d Prim.any list)
    (out : (v, s) E.t) : (v, s, d) Nx.t =
  let ins = List.map (fun (Prim.Any x) -> D.Any (Nx.dtype x)) loads in
  map1 e dt shape (P.of_exprs ~ins:(ia (Array.of_list ins)) [| E.any out |]) loads

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
  let flat =
    reshape e [| hk; pages * page; d |] (permute e [| 1; 0; 2; 3 |] pool)
  in
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

(* The transpose of [paged_read] into [pool]'s shape: each key row's
   cotangent in [y], [[lead…; hk; tk; d]], added into the pool's row its
   position names, summed over the lead indices; a key of position [-1]
   adds nothing. *)
let paged_write e (positions : 'd Nx.int64_t) (pool : ('v, 's, 'd) Nx.t)
    (y : ('v, 's, 'd) Nx.t) =
  let r = Nx.ndim positions in
  let lead = Array.sub (Nx.shape positions) 0 (r - 1) in
  let tk = Nx.dim (-1) positions in
  let pages = Nx.dim 0 pool and hk = Nx.dim 1 pool in
  let page = Nx.dim 2 pool and d = Nx.dim 3 pool in
  let s = Array.append lead [| hk; tk; d |] in
  let p =
    broadcast e s (reshape e (Array.append lead [| 1; tk; 1 |]) positions)
  in
  let dt = Nx.dtype y in
  let idx =
    map e D.Int64 s [ Prim.Any p ]
      E.(maximum (input 0 D.Int64) (const D.Int64 0L))
  in
  let updates =
    map e dt s [ Prim.Any y; Prim.Any p ]
      E.(
        where
          (input 1 D.Int64 < const D.Int64 0L)
          (const dt (D.zero dt)) (input 0 dt))
  in
  let into =
    Nx.copy
      (Nx.broadcast_to
         (Array.append lead [| hk; pages * page; d |])
         (Nx.scalar_like y (D.zero dt)))
  in
  let summed =
    e.eval
      (Scatter
         { combine = Add; unique = false; axis = r; idx; updates; into })
  in
  let summed =
    if Array.length lead = 0 then summed
    else Nx.sum ~axes:(List.init (Array.length lead) Fun.id) summed
  in
  permute e [| 1; 0; 2; 3 |] (reshape e [| hk; pages; page; d |] summed)

(* The programs a call's derivative runs, made once per pair program and
   marked loads: its visibility and its score alone; the score's tangent,
   over the pair program's operands then the tangents of its operand 0 and
   of each marked load; and that tangent's transpose's outputs, one program
   per tangent operand that receives a cotangent, over the pair program's
   operands then the score's cotangent. *)
type plan = {
  visible : P.t;
  score : P.t;
  tangent : P.t option;
  reads : P.t option array;  (** Operand 0's, then each marked load's. *)
}

let plans : plan Memo.t = Memo.create ()

let make_plan (prog : P.t) tangents =
  let n = Iarray.length prog.ins in
  let one k (p : P.t) = P.subst ~outs:[| k |] ~ins:p.ins (fun _ -> None) p in
  let marked = Array.init n (fun i -> i = 0 || tangents.(i - 1)) in
  let b = Scalar.start prog ~tangents:marked ~extra:[||] in
  let tangent =
    match Scalar.tangent b 1 with
    | Some t -> Some (Scalar.finish b ~outs:[| t |])
    | None -> None
  in
  let count = Array.fold_left (fun c t -> if t then c + 1 else c) 1 tangents in
  let reads =
    match tangent with
    | None -> Array.make count None
    | Some tp ->
        let m = Iarray.length tp.ins in
        let acc = P.dtype prog (Iarray.get prog.outs 1) in
        let linear = Array.init m (fun i -> i >= n) in
        let tr, got = Scalar.transpose tp ~linear ~cts:[| (0, acc) |] in
        let next = ref 0 in
        Array.init (m - n) (fun i ->
            match tr with
            | Some tr when got.(n + i) ->
                let k = !next in
                incr next;
                Some (one k tr)
            | _ -> None)
  in
  { visible = one 0 prog; score = one 1 prog; tangent; reads }

let plan spec =
  let n = Iarray.length spec.prog.ins in
  if not (Memo.fits n) then make_plan spec.prog spec.tangents
  else
    let mask = ref 1 in
    Array.iteri (fun i t -> if t then mask := !mask lor (1 lsl (i + 1))) spec.tangents;
    Memo.plain plans spec.prog !mask ~make:(fun k ->
        make_plan k.prog (Array.sub (Memo.operands k) 1 (n - 1)))

(* The pieces both directions read *)

type ('v, 's, 'a, 'd) pieces = {
  lead : int;  (** The number of lead axes. *)
  f : int array;  (** The pair loop's shape. *)
  shape : int array -> int array;
      (** [[lead…]] followed by the given extents. *)
  k : ('v, 's, 'd) Nx.t;  (** Dense. *)
  v : ('v, 's, 'd) Nx.t;
  positions : 'd Nx.int64_t option;
  s : (float, 'a, 'd) Nx.t;  (** Scaled scores. *)
  seen : (bool, D.bool_elt, 'd) Nx.t;
  p : (float, 'a, 'd) Nx.t;  (** Weights, [0] where hidden. *)
}

(* [contract acc spec a b] in [acc], [#] in [spec] standing for the lead
   axes' names. *)
let contract (type a v s w r d) (acc : (float, a) D.t) ~lead spec
    (a : (v, s, d) Nx.t) (b : (w, r, d) Nx.t) =
  let l = String.concat " " (List.init lead (Printf.sprintf "z%d")) in
  let spec =
    String.concat ""
      (List.map
         (fun c -> if c = '#' then l else String.make 1 c)
         (List.of_seq (String.to_seq spec)))
  in
  Nx.contract ~acc acc (Nx.Pattern.v spec) a b

(* [x], [[lead…; hq; tq; d]], as [[lead…; hk; g; tq; d]]. *)
let grouped e pc ~hk x =
  let s = Nx.shape x in
  let r = Array.length s in
  let hq = s.(r - 3) in
  reshape e (pc.shape [| hk; hq / hk; s.(r - 2); s.(r - 1) |]) x

let pieces (type v s a d) e spec ~(acc : (float, a) D.t) (q : (v, s, d) Nx.t)
    (kv : (v, s, d) Nx.kv) (lengths : d Nx.int64_t option)
    (loads : d Prim.any list) (lse : (float, a, d) Nx.t) : (v, s, a, d) pieces
    =
  let r = Nx.ndim q in
  let l = r - 3 in
  let lead = Array.sub (Nx.shape q) 0 l in
  let hq = Nx.dim (-3) q and tq = Nx.dim (-2) q in
  let shape tail = Array.append lead tail in
  let k, v, positions =
    match kv with
    | Dense { k; v } -> (k, v, None)
    | Paged { k; v; table } ->
        let p = positions e ~pages:(Nx.dim 0 k) ~page:(Nx.dim 2 k) table in
        (paged_read e p k, paged_read e p v, Some p)
  in
  let hk = Nx.dim (-3) k and tk = Nx.dim (-2) k in
  let f = shape [| hq; tq; tk |] in
  let gq = reshape e (shape [| hk; hq / hk; tq; Nx.dim (-1) q |]) q in
  let s = reshape e f (contract acc ~lead:l "# h g i d, # h j d -> # h g i j | d" gq k) in
  let s = Nx.mul s (Nx.scalar_like s spec.scale) in
  let pl = plan spec in
  let visible = map1 e D.Bool f pl.visible (Prim.Any s :: loads) in
  let score = map1 e acc f pl.score (Prim.Any s :: loads) in
  (* A pair is seen where its key is before the row's length and its table
     entry is in range. *)
  let extra =
    (match lengths with
    | Some len -> [ Prim.Any (spread e ~ones:3 f len) ]
    | None -> [])
    @
    match positions with
    | Some p -> [ Prim.Any (broadcast e f (reshape e (shape [| 1; 1; tk |]) p)) ]
    | None -> []
  in
  let key = r - 1 in
  let at = 1 + Option.fold ~none:0 ~some:(fun _ -> 1) lengths in
  let seen =
    map e D.Bool f (Prim.Any visible :: extra)
      E.(
        let seen = input 0 D.Bool in
        let seen =
          match lengths with
          | Some _ -> seen && coord key < input 1 D.Int64
          | None -> seen
        in
        match positions with
        | Some _ -> seen && const D.Int64 0L <= input at D.Int64
        | None -> seen)
  in
  let row = spread e ~ones:1 f lse in
  let p =
    map e acc f
      [ Prim.Any seen; Prim.Any score; Prim.Any row ]
      E.(
        let row = input 2 acc in
        where (input 0 D.Bool)
          (where
             (row = const acc Float.neg_infinity)
             (const acc 0.)
             (exp (input 1 acc - row)))
          (const acc 0.))
  in
  { lead = l; f; shape; k; v; positions; s; seen; p }

(* [x] where the pair is seen, [0] elsewhere: a hidden pair's NaN never
   counts. *)
let where_seen e pc x =
  let dt = Nx.dtype x in
  map e dt pc.f
    [ Prim.Any pc.seen; Prim.Any x ]
    E.(where (input 0 D.Bool) (input 1 dt) (const dt (D.zero dt)))

let tangent (type v s w r a d) e spec ~(acc : (float, a) D.t)
    (q : (v, s, d) Nx.t) (kv : (v, s, d) Nx.kv) lengths
    (loads : d Prim.any list) (o : (w, r, d) Nx.t) (lse : (float, a, d) Nx.t)
    (dq : (v, s, d) Nx.t) (dk : (v, s, d) Nx.t) (dv : (v, s, d) Nx.t)
    (dloads : d Prim.any list) : (w, r, d) Nx.t * (float, a, d) Nx.t =
  let pc = pieces e spec ~acc q kv lengths loads lse in
  let l = pc.lead in
  let hk = Nx.dim (-3) pc.k in
  let dk, dv =
    match pc.positions with
    | Some p -> (paged_read e p dk, paged_read e p dv)
    | None -> (dk, dv)
  in
  let raw a b =
    reshape e pc.f
      (contract acc ~lead:l "# h g i d, # h j d -> # h g i j | d" (grouped e pc ~hk a) b)
  in
  let ds = Nx.add (raw dq pc.k) (raw q dk) in
  let ds = Nx.mul ds (Nx.scalar_like ds spec.scale) in
  let dx =
    match (plan spec).tangent with
    | None -> Nx.zeros_like pc.s
    | Some tp ->
        map1 e acc pc.f tp ((Prim.Any pc.s :: loads) @ (Prim.Any ds :: dloads))
  in
  let t = where_seen e pc (Nx.mul pc.p dx) in
  let dlse = Nx.sum ~axes:[ l + 2 ] t in
  let prod a b =
    let y = contract acc ~lead:l "# h g i j, # h j c -> # h g i c | j" (grouped e pc ~hk a) b in
    reshape e (Nx.shape o) y
  in
  let os = Nx.shape o in
  let dlse_o = Nx.mul (spread e ~ones:1 os dlse) (Nx.cast acc o) in
  let dout = Nx.add (Nx.sub (prod t pc.v) dlse_o) (prod pc.p dv) in
  (Nx.cast (Nx.dtype o) dout, dlse)

let cotangent (type v s w r a d) e spec ~(acc : (float, a) D.t)
    (q : (v, s, d) Nx.t) (kv : (v, s, d) Nx.kv) lengths
    (loads : d Prim.any list) (o : (w, r, d) Nx.t) (lse : (float, a, d) Nx.t)
    (bo : (w, r, d) Nx.t) (blse : (float, a, d) Nx.t) =
  let pc = pieces e spec ~acc q kv lengths loads lse in
  let l = pc.lead in
  let hk = Nx.dim (-3) pc.k in
  let bo = Nx.cast acc bo and o = Nx.cast acc o in
  let delta = Nx.sub (Nx.sum ~axes:[ l + 2 ] (Nx.mul bo o)) blse in
  let dp =
    reshape e pc.f
      (contract acc ~lead:l "# h g i c, # h j c -> # h g i j | c" (grouped e pc ~hk bo) pc.v)
  in
  let ds =
    where_seen e pc (Nx.mul pc.p (Nx.sub dp (spread e ~ones:1 pc.f delta)))
  in
  (* The program's transpose: the score's cotangent [ds] read back to the
     operands its tangent was computed from: operand 0, then each marked
     load. *)
  let reads =
    Array.map
      (Option.map (fun one ->
           map1 e acc pc.f one ((Prim.Any pc.s :: loads) @ [ Prim.Any ds ])))
      (plan spec).reads
  in
  let zero () = Nx.zeros_like pc.s in
  let bs = Option.value reads.(0) ~default:(zero ()) in
  let rbar = Nx.mul bs (Nx.scalar_like bs spec.scale) in
  let gr = grouped e pc ~hk rbar in
  let bq =
    reshape e (Nx.shape q) (contract acc ~lead:l "# h g i j, # h j d -> # h g i d | j" gr pc.k)
  in
  let bk = contract acc ~lead:l "# h g i j, # h g i d -> # h j d | g i" gr (grouped e pc ~hk q) in
  let bv =
    contract acc ~lead:l "# h g i j, # h g i c -> # h j c | g i" (grouped e pc ~hk pc.p)
      (grouped e pc ~hk bo)
  in
  let vt = Nx.dtype q in
  let bk, bv =
    match (kv, pc.positions) with
    | Paged { k; v; _ }, Some p ->
        (paged_write e p k (Nx.cast vt bk), paged_write e p v (Nx.cast vt bv))
    | _ -> (Nx.cast vt bk, Nx.cast vt bv)
  in
  let next = ref 1 in
  let bloads =
    List.filter_map Fun.id
      (List.mapi
         (fun i (Prim.Any x) ->
           if not spec.tangents.(i) then None
           else begin
             let c = Option.value reads.(!next) ~default:(zero ()) in
             incr next;
             Some (Prim.Any (Nx.cast (Nx.dtype x) c))
           end)
         loads)
  in
  (Nx.cast vt bq, bk, bv, bloads)

let linear (type d) spec ~transposed (data : d Prim.any array)
    (args : d Prim.any array) : d Prim.any array =
  let e = { eval = (fun op -> Prim.eval ~by:"Nx.attention" op) } in
  let (Prim.Any q) = data.(0) in
  let set = Nx.Placement.devices (Nx.placement q) in
  let like (type v s) (dt : (v, s) D.t) (Prim.Any x) : (v, s, d) Nx.t =
    match Nx.Prim.coerce dt set x with
    | Some y -> y
    | None -> invalid_arg "Rune: an operand of attention's derivative"
  in
  let n = Array.length data in
  let vt = Nx.dtype q in
  let k = like vt data.(1) and v = like vt data.(2) in
  let at = ref 3 in
  let next () =
    let x = data.(!at) in
    incr at;
    x
  in
  let table = if spec.paged then Some (like D.Int64 (next ())) else None in
  let lengths = if spec.lengths then Some (like D.Int64 (next ())) else None in
  let loads = Array.to_list (Array.sub data !at (n - 2 - !at)) in
  let (Prim.Any o) = data.(n - 2) in
  let (Prim.Any lse) = data.(n - 1) in
  let kv : (_, _, d) Nx.kv =
    match table with None -> Dense { k; v } | Some table -> Paged { k; v; table }
  in
  match D.kind (Nx.dtype lse) with
  | Float when not transposed ->
      let acc = Nx.dtype lse in
      let dq = like vt args.(0) and dk = like vt args.(1) in
      let dv = like vt args.(2) in
      let dloads = Array.to_list (Array.sub args 3 (Array.length args - 3)) in
      let o', l' = tangent e spec ~acc q kv lengths loads o lse dq dk dv dloads in
      [| Prim.Any o'; Prim.Any l' |]
  | Float ->
      let acc = Nx.dtype lse in
      let bo = like (Nx.dtype o) args.(0) and bl = like acc args.(1) in
      let bq, bk, bv, bloads = cotangent e spec ~acc q kv lengths loads o lse bo bl in
      Array.of_list (Prim.Any bq :: Prim.Any bk :: Prim.Any bv :: bloads)
  | _ -> invalid_arg "Rune: attention's lse is not a float"
