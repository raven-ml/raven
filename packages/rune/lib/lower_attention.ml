(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
module D = Nx_array.Dtype
module P = Nx_kernel.Prog
module Arith = Lower_arith

(* Attention as kernels of its own

   One kernel folds each query row over its keys in tiles, carrying the row's
   running maximum [m], its sum [l] and its accumulator [acc] of weighted
   values (online softmax, Milakov & Gimelshein 2018; FlashAttention-2's
   forward, Dao 2023, Algorithm 1), so that no score outlives its tile and
   nothing of [[lead…; hq; tq; tk]] is stored. Scores are kept in base 2:
   [x·log₂e], whose exponentials are [exp2]. Per tile of keys:

     m'  = max (m, x_j)          the visible scores' maximum
     μ   = m' where finite, else 0
     α   = 2^(m - μ)             the old state's scale
     p_j = 2^(x_j - μ)           0 at a hidden pair
     l   = α l + Σ p_j
     acc = α acc + Σ p_j v_j

   and at the end [o = acc / l] and [lse = (m + log₂ l) ln 2], [o = +0] and
   [lse = -∞] where [l = 0]. A row's rows of a workgroup are its query rows
   and its group's heads, the heads fastest, so that a tile of keys serves
   each head of a group ([grouped]).

   A pair is visible where its key is before the row's length, its page's
   table entry is in range, and the program's output 0 holds. A tile whose
   pairs are all hidden is skipped ([guard]). *)


(* Problems *)

type pages = { pages : int; page : int; max_pages : int }

(* A capture of the pair program, read at a pair through the strides of its
   own extents: [0] along an axis it broadcasts on. *)
type load = { dt : D.any; strides : int array }

type t = {
  lead : int array;
  hq : int;
  hk : int;
  tq : int;
  tk : int;
  dk : int;
  dv : int;
  paged : pages option;
  lengths : bool;
  prog : P.t;
  scale : float;
  vt : D.any;  (** [q]'s, [k]'s and [v]'s. *)
  acc : D.any;
  out : D.any;
  loads : load array;
  classes : P.t option;
      (** The ranges of the program's output 0 over a tile's pairs
          ({!Nx_kernel.Prog.ranges}), unless it is a constant. *)
}

let numel = Array.fold_left ( * ) 1
let rec log2 n = if n <= 1 then 0 else 1 + log2 (n / 2)
let ceil_div a b = (a + b - 1) / b
let pow2 i = 1 lsl i
let groups a = if a.hk = 0 then 1 else a.hq / a.hk

(* The rows of a group of heads: its query rows times its heads. *)
let grouped a = a.tq * groups a

(* Builder *)

let int = Ops.O.int
let ld buf i = Ops.load (Ops.index buf [ i ]) []
let st buf i v = Ops.store (Ops.index buf [ i ]) v

(* [ld_gated buf i c] is [buf]'s element [i] where [c] holds and [0]
   elsewhere, where it reads nothing. *)
let ld_gated buf i c = Ops.load (Ops.index buf [ Ops.where c i Ops.invalid ]) []
let range ty bound id = Ops.range ~axis_type:ty bound [ id ]
let weak u = Ops.cast u Weak_int
let ( &&: ) = Ops.bitwise_and
let fl dt x = Ops.float ~dtype:dt x
let i64 u = Ops.cast u Int64

(* [guard ~outer id c body] runs [body inside] where [c] holds, [c] the
   same for every thread of a workgroup: a loop of [c]'s trips, [0] or [1]
   ({!Tolk.Op.Range}). [inside buf] is [buf] read or written in it, within
   the guard [outer] reads it through: every buffer the body touches is read
   through [inside], so that no effect or load of the body runs outside the
   loop, where a skipped tile would run it. *)
let guard ?(outer = Fun.id) id c body =
  let g = range Loop (Sym (weak c)) id in
  let inside buf = Ops.after (outer buf) [ g ] in
  Ops.end_ (body inside) [ g ]

(* [reg slot n dt v ~at] is registers of [n] elements of [dt] and the stores
   that set each to [v] at each step of the ranges [at]: a row's, so that a
   thread running several rows starts each anew. *)
let reg slot n dt v ~at =
  let r = Shape.placeholder ~slot ~addrspace:Reg [ max 1 n ] dt in
  let here = Ops.after r at in
  (r, Ops.group (List.init n (fun i -> st here (int i) v)))

(* Slots of the kernels' own buffers, after their arguments'. *)
let slot_m = 100
let slot_l = 101
let slot_s = 102
let slot_vote = 108

(* Arguments

   A kernel takes, in slot order: [o], [lse], for the SIMT kernel [sum]
   ([rows · dv] of the accumulator, value [c] of row [r] at [c · rows + r], so
   that a workgroup's threads touch adjacent elements), then [q], [k], [v],
   the page table, the lengths and the captures where given. *)

type args = {
  o : Ops.t;
  lse : Ops.t;
  sum : Ops.t option;
  q : Ops.t;
  k : Ops.t;
  v : Ops.t;
  table : Ops.t option;
  len : Ops.t option;
  caps : Ops.t array;
}

let args ~scratch a xs =
  let o, lse, sum, rest =
    match (scratch, xs) with
    | true, o :: lse :: sum :: rest -> (o, lse, Some sum, rest)
    | false, o :: lse :: rest -> (o, lse, None, rest)
    | _ -> invalid_arg "Lower_attention: arguments"
  in
  match rest with
  | q :: k :: v :: rest ->
      let table, rest =
        match (a.paged, rest) with
        | Some _, t :: rest -> (Some t, rest)
        | _ -> (None, rest)
      in
      let len, rest =
        match (a.lengths, rest) with
        | true, l :: rest -> (Some l, rest)
        | _ -> (None, rest)
      in
      { o; lse; sum; q; k; v; table; len; caps = Array.of_list rest }
  | _ -> invalid_arg "Lower_attention: arguments"

(* Rows, keys and pairs *)

(* Row [r] of the rows of key head [kh] of lead index [b]: whether it is one,
   its query [t] and head [h], and its index among [o]'s rows. *)
type row = { live : Ops.t; t : Ops.t; h : Ops.t; index : Ops.t }

let row_at a ~b ~kh r =
  let open Ops.O in
  let g = groups a in
  let t = r // int g and h = (kh * int g) + (r % int g) in
  {
    live = r < int (grouped a);
    t;
    h;
    index = (((b * int a.hq) + h) * int a.tq) + t;
  }

(* The keys lead index [b]'s rows read: its length, within [0, tk]. *)
let length a x b =
  match x.len with
  | None -> int a.tk
  | Some lens -> Ops.maximum (int 0) (Ops.minimum (weak (ld lens b)) (int a.tk))

(* Whether key [key] of key head [kh] of lead index [b] is read, where
   [within] holds and its page's entry is in range, and where its row starts
   in [k]'s and [v]'s arrays or pools, in rows. *)
let key_at a x ~inside ~b ~kh ~within key =
  let open Ops.O in
  match (a.paged, x.table) with
  | Some p, Some table ->
      let e =
        ld_gated (inside table)
          ((b * int p.max_pages) + (key // int p.page))
          within
      in
      let e = weak e in
      let ok = within &&: (int 0 <= e) &&: (e < int p.pages) in
      (ok, (((e * int a.hk) + kh) * int p.page) + (key % int p.page))
  | _ -> (within, (((b * int a.hk) + kh) * int a.tk) + key)

let ln2e = Float.log2 (Float.exp 1.)

(* The coordinates of the pair of [row] and [key], as the program reads
   them. *)
let pair_coord a ~b ~(row : row) ~key ax =
  let open Ops.O in
  let lead = Array.length a.lead in
  if ax = lead then i64 row.h
  else if ax = Stdlib.(lead + 1) then i64 row.t
  else if ax = Stdlib.(lead + 2) then i64 key
  else
    let inner =
      numel (Array.sub a.lead Stdlib.(ax + 1) Stdlib.(lead - ax - 1))
    in
    i64 ((b // int inner) % int a.lead.(ax))

(* Each load's parts at the pair of coordinates [coord], read from [bufs]
   where [ok]. *)
let caps_at (loads : load array) bufs ~inside ~coord ~ok =
  let open Ops.O in
  Array.mapi
    (fun c (l : load) ->
      let terms =
        Array.mapi
          (fun ax st -> if st = 0 then int 0 else weak (coord ax) * int st)
          l.strides
      in
      let at = Array.fold_left ( + ) (int 0) terms in
      Arith.parts l.dt (ld_gated (inside bufs.(c)) at ok))
    loads

(* Whether the pair at [coord], read where [ok], of captures' parts [caps]
   and score [s], scaled, in the accumulator, is visible, and its score in
   base 2, [-∞] where hidden. *)
let pair_of a ~coord ~ok ~caps s =
  let acc = Arith.dtype a.acc in
  let outs = Arith.scalar a.prog ~coord (Array.append [| [ s ] |] caps) in
  let seen =
    match outs.(0) with
    | [ v ] ->
        ok &&: Ops.ne v (Ops.const ~dtype:(Ops.dtype v) (`Int Bigint.zero))
    | _ -> invalid_arg "Lower_attention: a mask of parts"
  in
  ( seen,
    Ops.where seen
      (Ops.mul (List.hd outs.(1)) (fl acc ln2e))
      (fl acc Float.neg_infinity) )

(* The pair of [row] and [key], read where [ok], of raw score [raw] in the
   accumulator: whether it is visible, and its score in base 2. *)
let pair a x ~inside ~b ~row ~key ~ok raw =
  let coord = pair_coord a ~b ~row ~key in
  let caps = caps_at a.loads x.caps ~inside ~coord ~ok in
  pair_of a ~coord ~ok ~caps (Ops.mul raw (fl (Arith.dtype a.acc) a.scale))

(* Whether the pair of [row] and [key], read where [ok], is visible. *)
let seen a x ~inside ~b ~row ~key ~ok =
  fst (pair a x ~inside ~b ~row ~key ~ok (fl (Arith.dtype a.acc) 0.))

let score a x ~inside ~b ~row ~key ~ok raw =
  snd (pair a x ~inside ~b ~row ~key ~ok raw)

(* Tile classes

   A tile of rows [r0] to [r1] and keys [k0] to [k1] is classed by the ranges
   of the program's output 0 over the box of its pairs: the lead index, the
   heads of the rows' group, the rows' queries, the keys, each capture that
   varies along lead axes alone at its value there, and any other capture
   over its dtype's whole range. Where output 0 is false at every pair the
   tile is hidden; where true, every pair the length and the table show is
   visible. *)

let classes_of (p : P.t) rank =
  match Iarray.get p.nodes (Iarray.get p.outs 0) with
  | Const _ -> None
  | _ -> Some (P.ranges ~axes:rank ~outs:[| 0 |] p)

let bounds (D.Any dt) =
  if D.is D.Float dt || D.is D.Complex dt then
    let z = Arith.fill dt (D.zero dt) in
    (z, z)
  else (Arith.fill dt (D.min_value dt), Arith.fill dt (D.max_value dt))

(* [(hidden, shown)] of a tile, or [None] for a constant output 0. *)
let classes a x ~inside ~b ~kh ~rows:(r0, r1) ~keys:(k0, k1) =
  match a.classes with
  | None -> None
  | Some rp ->
      let open Ops.O in
      let lead = Array.length a.lead in
      let g = groups a in
      let consts d cs = List.map (fun c -> Ops.const ~dtype:(Arith.dtype d) c) cs in
      let score = fl (Arith.dtype a.acc) 0. in
      let caps =
        Array.mapi
          (fun c (l : load) ->
            let by_lead =
              Array.for_all (( = ) 0) (Array.sub l.strides lead 3)
            in
            if by_lead then
              let at = Array.fold_left ( + ) (int 0)
                  (Array.mapi
                     (fun ax st ->
                       if Stdlib.(ax >= lead || st = 0) then int 0
                       else
                         let inner =
                           numel (Array.sub a.lead Stdlib.(ax + 1) Stdlib.(lead - ax - 1))
                         in
                         (b // int inner) % int a.lead.(ax) * int st)
                     l.strides)
              in
              let v = Arith.parts l.dt (ld (inside x.caps.(c)) at) in
              [ v; v ]
            else
              let lo, hi = bounds l.dt in
              [ consts l.dt lo; consts l.dt hi ])
          a.loads
      in
      let ops = Array.concat ([ [| [ score ]; [ score ] |] ] @ Array.to_list (Array.map Array.of_list caps)) in
      let coord lo hi = [| [ i64 lo ]; [ i64 hi ] |] in
      let coords =
        Array.concat
          (List.init lead (fun ax ->
               let inner =
                 numel (Array.sub a.lead Stdlib.(ax + 1) Stdlib.(lead - ax - 1))
               in
               let c = (b // int inner) % int a.lead.(ax) in
               coord c c)
          @ [
              coord (kh * int g) ((kh * int g) + int Stdlib.(g - 1));
              coord (r0 // int g) (r1 // int g);
              coord k0 k1;
            ])
      in
      let outs = Arith.scalar rp ~coord:(fun _ -> assert false) (Array.append ops coords) in
      let truth = function
        | [ v ] -> Ops.ne v (Ops.const ~dtype:(Ops.dtype v) (`Int Bigint.zero))
        | _ -> invalid_arg "Lower_attention: a class of parts"
      in
      Some (Ops.logical_not (truth outs.(1)), truth outs.(0))

(* [skipping a x ~at ~b ~kh ~rows ~keys ~vote ~ids body] is [body] in guards
   that skip a tile its classes hide, and a tile of unknown class where
   [vote], a workgroup's, finds no visible pair. The vote runs only on a tile
   of unknown class, its result carried out of its guard in a register set
   at each step of the ranges [at]. *)
let skipping a x ~at ~b ~kh ~rows ~keys ~vote ~ids:(id0, id1, id2) body =
  match classes a x ~inside:Fun.id ~b ~kh ~rows ~keys with
  | None -> guard id0 (Ops.O.bool true) body
  | Some (hidden, shown) ->
      guard id0 (Ops.logical_not hidden) (fun inside ->
          let r = Shape.placeholder ~slot:slot_vote ~addrspace:Reg [ 1 ] Bool in
          let set = st (inside (Ops.after r at)) (int 0) shown in
          let voted =
            guard ~outer:inside id1 (Ops.logical_not shown) (fun inside ->
                st (inside (Ops.after r [ set ])) (int 0) (vote inside))
          in
          let run = ld (Ops.after r [ set; voted ]) (int 0) in
          guard ~outer:inside id2 run body)

(* [vote a x ~inside ~b ~kh ~len ~over pairs] is whether a pair of
   [pairs], each a [(row, key)] of a thread, is visible to some thread of
   the ranges [over], [[]] for the thread alone. It reads inside the guard
   of the tile's class, so that only a tile of unknown class votes. *)
let vote a x ~inside ~b ~kh ~len ~over pairs =
  let open Ops.O in
  let any =
    List.fold_left
      (fun any ((row : row), key) ->
        let ok, _ = key_at a x ~inside ~b ~kh ~within:(row.live &&: (key < len)) key in
        Ops.bitwise_or any (seen a x ~inside ~b ~row ~key ~ok))
      (Ops.O.bool false) pairs
  in
  let any = inside any in
  if over = [] then any
  else Ops.ne (Ops.reduce (Ops.cast any Int32) Max over) (Ops.int ~dtype:Int32 0)

(* The state [(m', α, p)] of a tile of base-2 scores [x2] over a row's old
   maximum [m]. *)
let softmax acc m x2 =
  let m' = Array.fold_left Ops.maximum m x2 in
  let ninf = fl acc Float.neg_infinity in
  let mu = Ops.where (Ops.eq m' ninf) (fl acc 0.) m' in
  (m', Ops.exp2 (Ops.sub m mu), Array.map (fun x -> Ops.exp2 (Ops.sub x mu)) x2)

(* A row's [lse] and [o]'s element of accumulator [v] over its sum [l]. *)
let lse_of acc ~m ~l =
  let empty = Ops.eq l (fl acc 0.) in
  let ln2 = fl acc (Float.log 2.) in
  Ops.where empty (fl acc Float.neg_infinity)
    (Ops.mul (Ops.add m (Ops.log2 l)) ln2)

let o_of acc ~l v =
  Ops.where (Ops.eq l (fl acc 0.)) (fl acc 0.) (Ops.alu v Op.Fdiv [ l ])

(* Key splits

   A launch of few workgroups splits each row's keys into [splits] runs of
   tiles, each a workgroup's, which store their state [(m, l, acc)]
   unnormalised: [parts] holds a row's [m] and [l] of run [s] at
   [2 (r · splits + s)] and [+ 1], [sums] its [acc] at
   [(r · splits + s) · dv]. {!combine} folds the runs in order
   (Flash-Decoding, Dao et al. 2023). *)

(* The tiles of run [gs] of [splits], of [tk] keys in tiles of [bk], of a
   row of [len] keys: its first tile and its count. *)
let run_tiles ~tk ~bk ~splits ~gs len =
  let open Ops.O in
  let per = ceil_div (ceil_div tk bk) splits in
  let tiles = (len + int Stdlib.(bk - 1)) // int bk in
  match gs with
  | None -> (int 0, tiles)
  | Some gs ->
      let first = gs * int per in
      (first, Ops.maximum (int 0) (Ops.minimum (tiles - first) (int per)))

(* The SIMT kernel

   A workgroup holds [simt_rows] rows of a group of heads, one a thread, or
   32 for fewer rows ([rows_of]), and takes tiles of [simt_keys] keys,
   whose scores stay in registers. The host has one row a thread, whose
   accumulator's columns are adjacent and added a vector at a time
   ({!simt_width}). *)

let simt_rows = 64

(* The rows a SIMT workgroup holds for [a]: [simt_rows], or a warp's worth
   of 32 threads for fewer rows, a decode step's one query a head, whose
   other threads would hold no row. *)
let rows_of a = if grouped a <= 32 then 32 else simt_rows
let simt_keys = 32

(* The values' columns a SIMT thread adds at a time: one on a GPU, whose
   threads are its vector, and on the host as many as fill one of its
   vector registers. Each column's sum keeps its order at any width. *)
let simt_width (r : Renderer.t) a =
  if r.has_local then 1
  else
    match r.registers () with
    | Some { bytes; _ } -> max 1 (bytes / Dtype.itemsize (Arith.dtype a.acc))
    | None -> 1

let simt ~local ~width ~splits a xs =
  let x = args ~scratch:true a xs in
  let sum = Option.get x.sum in
  let acc = Arith.dtype a.acc in
  let gr = grouped a in
  let bq = if local then rows_of a else 1 in
  let bk = simt_keys in
  let tiles_q = (gr + bq - 1) / bq in
  let rows = numel a.lead * a.hq * a.tq in
  let heads = numel a.lead * a.hk in
  let open Ops.O in
  (* The host's outer loops are its threads' ([Weak]). *)
  let outer = if local then Ops.Axis_type.Global else Weak in
  let gh = range outer (Int heads) 0 in
  (* Row tiles from the last, which under a causal mask read the most keys:
     the longest workgroups start first. *)
  let gtr = range outer (Int tiles_q) 1 in
  let gt = int Stdlib.(tiles_q - 1) - gtr in
  let l = if local then range Local (Int bq) 2 else int 0 in
  let gs = if Stdlib.(splits > 1) then Some (range outer (Int splits) 9) else None in
  let b = gh // int a.hk and kh = gh % int a.hk in
  let row = row_at a ~b ~kh ((gt * int bq) + l) in
  let live = row.live in
  let len = length a x b in
  let first, tiles = run_tiles ~tk:a.tk ~bk ~splits ~gs len in
  let jr = range Loop (Sym tiles) 3 in
  let j = first + jr in
  let at =
    (if local then [ gh; gtr; l ] else [ gh; gtr ]) @ Option.to_list gs
  in
  (* A row's place among the rows of every run. *)
  let part = (row.index * int splits) + Option.value gs ~default:(int 0) in
  let rows = Stdlib.(rows * splits) in
  let m, m_init = reg slot_m 1 acc (fl acc Float.neg_infinity) ~at in
  let ls, l_init = reg slot_l 1 acc (fl acc 0.) ~at in
  let m_j = Ops.after m [ m_init; jr ] and l_j = Ops.after ls [ l_init; jr ] in
  (* A row's accumulators: on a GPU value [c] of row [r] at [c · rows + r],
     so that a workgroup's threads touch adjacent elements; on the host a
     row's values adjacent, so that a thread's run of columns is a
     vector. *)
  let slot c =
    if local then (c * int rows) + part else (part * int a.dv) + c
  in
  (* The accumulator starts at zero: the rows' values, before the loop. *)
  let sum0 =
    Ops.group
      (List.init a.dv (fun c ->
           Ops.store (Ops.index sum [ Ops.where live (slot (int c)) Ops.invalid ]) (fl acc 0.)))
  in
  let sum_j = Ops.after sum (if a.dv = 0 then [ jr ] else [ sum0; jr ]) in
  let key i = (j * int bk) + int i in
  let body inside =
    let keys =
      Array.init bk (fun i ->
          key_at a x ~inside ~b ~kh ~within:(live &&: (key i < len)) (key i))
    in
    let qs = inside x.q and ks = inside x.k and vs = inside x.v in
    (* Scores: Σ_d q_d k_d in the accumulator, each product widened. *)
    let s = Shape.placeholder ~slot:slot_s ~addrspace:Reg [ bk ] acc in
    let here = inside (Ops.after s (jr :: at)) in
    let s_init =
      Ops.group (List.init bk (fun i -> st here (int i) (fl acc 0.)))
    in
    let s =
      if a.dk = 0 then Ops.after s [ s_init ]
      else
        let d = range Loop (Int a.dk) 4 in
        let s_d = Ops.after s [ s_init; d ] in
        let qd = Ops.cast (ld_gated qs ((row.index * int a.dk) + d) live) acc in
        let s_end =
          Ops.end_
            (Ops.group
               (List.init bk (fun i ->
                    let ok, at = keys.(i) in
                    let kd =
                      Ops.cast (ld_gated ks ((at * int a.dk) + d) ok) acc
                    in
                    st s_d (int i) (ld s_d (int i) + (qd * kd)))))
            [ d ]
        in
        Ops.after s [ s_end ]
    in
    let x2 =
      Array.init bk (fun i ->
          score a x ~inside ~b ~row ~key:(key i) ~ok:(fst keys.(i))
            (ld s (int i)))
    in
    let m_old = ld (inside m_j) (int 0) and l_old = ld (inside l_j) (int 0) in
    let m_new, alpha, p = softmax acc m_old x2 in
    let l_new = Array.fold_left ( + ) (l_old * alpha) p in
    (* Values: acc = α acc + Σ p_j v_j, the products added in key order,
       [width] columns at a time: each column's sum is a chain of additions,
       which a vector of columns runs side by side. *)
    let values =
      if a.dv = 0 then []
      else
        let cg = range Loop (Int (ceil_div a.dv width)) 5 in
        let e = if Stdlib.(width = 1) then int 0 else range Upcast (Int width) 10 in
        let sum_c = Ops.after (inside sum_j) [ cg ] in
        let col = (cg * int width) + e in
        let ok_c = live &&: (col < int a.dv) in
        let old = ld_gated sum_c (slot col) ok_c in
        let next =
          Array.fold_left ( + ) (old * alpha)
            (Array.mapi
               (fun i p ->
                 let ok, at = keys.(i) in
                 p
                 * Ops.cast
                     (ld_gated vs ((at * int a.dv) + col) (ok &&: (col < int a.dv)))
                     acc)
               p)
        in
        let ranges = if Stdlib.(width = 1) then [ cg ] else [ e; cg ] in
        [
          Ops.end_
            (Ops.store (Ops.index sum_c [ Ops.where ok_c (slot col) Ops.invalid ]) next)
            ranges;
        ]
    in
    Ops.group
      (values @ [ st (inside m_j) (int 0) m_new; st (inside l_j) (int 0) l_new ])
  in
  (* A thread's vote: whether its row sees a key of the tile. *)
  let vote inside =
    vote a x ~inside ~b ~kh ~len ~over:(if local then [ l ] else [])
      (List.init bk (fun i -> (row, key i)))
  in
  let row_span =
    (gt * int bq, Ops.minimum ((gt * int bq) + int Stdlib.(bq - 1)) (int Stdlib.(gr - 1)))
  in
  let keys = (key 0, key Stdlib.(bk - 1)) in
  let loop =
    Ops.end_
      (skipping a x ~at:(jr :: at) ~b ~kh ~rows:row_span ~keys ~vote ~ids:(6, 7, 8) body)
      [ jr ]
  in
  let m = ld (Ops.after m [ loop ]) (int 0) in
  let lsum = ld (Ops.after ls [ loop ]) (int 0) in
  let sum = Ops.after sum [ loop ] in
  let out = Arith.dtype a.out in
  let at_row i = Ops.where live i Ops.invalid in
  let stores =
    if Stdlib.(splits > 1) then
      Ops.store (Ops.index x.lse [ at_row (int 2 * part) ]) m
      :: Ops.store (Ops.index x.lse [ at_row ((int 2 * part) + int 1) ]) lsum
      :: List.init a.dv (fun c ->
             Ops.store
               (Ops.index x.o [ at_row ((part * int a.dv) + int c) ])
               (ld_gated sum (slot (int c)) live))
    else
      Ops.store (Ops.index x.lse [ at_row row.index ]) (lse_of acc ~m ~l:lsum)
      :: List.init a.dv (fun c ->
             let v = o_of acc ~l:lsum (ld_gated sum (slot (int c)) live) in
             Ops.store
               (Ops.index x.o [ at_row ((row.index * int a.dv) + int c) ])
               (Ops.cast v out))
  in
  let kernel = Ops.kernel_info ~name:"attention" ~opts_to_apply:[] () in
  Ops.sink ~kernel [ Ops.end_ (Ops.group stores) at ]

(* The tensor-core kernel

   For a narrow float on a target whose core of that float's products into
   float32 chains its accumulator into its own operand [A] with each row's
   elements in the same lanes ({!Tc.chain}), as CUDA's [m16n8k16]. A warp
   holds [M] rows of a group of heads, a workgroup [warps] warps, and each
   tile of [keys] keys is copied into workgroup memory, its rows padded by 8
   elements. Per tile, a warp's scores [S = Q Kᵀ] are the core's
   accumulators in registers, [Q]'s operands loaded once; each row's maximum
   goes through workgroup memory to the lanes that share the row; [P],
   rounded once to the values' dtype, is the [A] operand of [P V] from the
   accumulators it chains from; and [V]'s tile is copied after the row
   maximum's barrier, so that its operands' loads wait on it rather than
   live across the softmax. Coordinates of each lane's elements are the
   core's fragments' ({!Tc.frag_coords}). *)

type pick = Mb | Nb | Kb

let picks p (b : Tc.bit) =
  match (p, b) with
  | Mb, M j | Nb, N j | Kb, K j -> Some j
  | _ -> None

(* The coordinate [p] given by the bits [bits] of fragment positions at the
   integer [v]. *)
let ipart p bits v =
  List.fold_left ( + ) 0
    (List.mapi
       (fun i b ->
         match picks p b with
         | Some j when (v lsr i) land 1 = 1 -> 1 lsl j
         | _ -> 0)
       bits)

(* The coordinate [p] of element [e] at [lane] of fragment [f], as a node of
   [lane], and at an integer lane. *)
let coord (f : Tc.fragment) p lane e =
  let terms =
    List.concat
      (List.mapi
         (fun i b ->
           match picks p b with
           | Some j -> [ Ops.O.((lane // int (pow2 i)) % int 2 * int (pow2 j)) ]
           | None -> [])
         f.lanes)
  in
  List.fold_left Ops.add (int (ipart p f.elements e)) terms

let icoord (f : Tc.fragment) p lane e = ipart p f.lanes lane + ipart p f.elements e

let count p bits = List.length (List.filter (fun b -> picks p b <> None) bits)

(* The accumulator and element of [x]'s each element [e] of [y]'s operand [A]
   comes from under the chain [ch], and the key within a step of [y]'s [K]
   of each [K] coordinate. *)
let chained (tc : Tc.t) (ch : Tc.chain) =
  let n, _, k = Tc.dims tc in
  let nbits = count Nb tc.frag_c.elements + count Nb tc.frag_c.lanes in
  let unnamed =
    List.filter
      (fun b -> not (List.mem b ch.k_bits))
      (List.init (count Kb tc.frag_a.elements + count Kb tc.frag_a.lanes) Fun.id)
  in
  (* The key within the step of [K] coordinate [kc]: the accumulator its
     unnamed bits number, then its column. *)
  let key kc =
    let bit i = (kc lsr i) land 1 in
    let col =
      List.fold_left Stdlib.( + ) 0
        (List.mapi (fun i kb -> bit kb lsl i) ch.k_bits)
    in
    let acc =
      List.fold_left Stdlib.( + ) 0 (List.mapi (fun u kb -> bit kb lsl u) unnamed)
    in
    (acc lsl nbits) + col
  in
  let keys = Array.init k key in
  let lanes = Tc.threads tc in
  let elems f = 1 lsl List.length f.Tc.elements in
  let source e =
    let found = ref None in
    for t = 0 to (k / n) - 1 do
      for e' = 0 to elems tc.frag_c - 1 do
        let all = ref true in
        for lane = 0 to lanes - 1 do
          let row = icoord tc.frag_a Mb lane e
          and kc = icoord tc.frag_a Kb lane e in
          let w = keys.(kc) in
          if
            not
              (w lsr nbits = t
              && icoord tc.frag_c Mb lane e' = row
              && icoord tc.frag_c Nb lane e' = w land ((1 lsl nbits) - 1))
          then all := false
        done;
        if !all then found := Some (t, e')
      done
    done;
    match !found with
    | Some te -> te
    | None -> invalid_arg "Lower_attention: a chained operand"
  in
  (Array.init (elems tc.frag_a) source, keys)

(* [key_of keys kc] is the node of the key within a step of [K] coordinate
   [kc], a node: its bits moved as [keys] moves each coordinate's. *)
let key_of keys k kc =
  let open Ops.O in
  let identity = Array.for_all2 ( = ) keys (Array.init k Fun.id) in
  if identity then kc
  else
    let bits = log2 k in
    List.fold_left ( + ) (int 0)
      (List.init bits (fun i ->
           (* Where bit [i] of the coordinate lands in the key. *)
           let dst = log2 keys.(pow2 i) in
           (kc // int (pow2 i)) % int 2 * int (pow2 dst)))

(* A chaining core's fragment facts: its dimensions and lanes, where each
   element of the next product's [A] comes from ({!chained}), and a lane's
   rows of the accumulator: its row bits among the elements number them,
   its row lanes share them with the lanes that differ in a column bit. *)
type facts = {
  n_ : int;
  m_ : int;
  k_ : int;
  lanes : int;
  sources : (int * int) array;
  keymap : int array;
  tsteps : int;
  ea : int;
  ec : int;
  slot_of : int -> int;
  slots : int;
  first_of : int -> int;
}

let facts (tc : Tc.t) =
  let n_, m_, k_ = Tc.dims tc in
  let ch =
    match Tc.chain tc tc A with
    | Some ch -> ch
    | None -> invalid_arg "Lower_attention: a core that does not chain"
  in
  let sources, keymap = chained tc ch in
  let elems (f : Tc.fragment) = 1 lsl List.length f.elements in
  let ec = elems tc.frag_c in
  let slot_of e =
    List.fold_left ( + ) 0
      (List.mapi
         (fun i b ->
           match b with
           | Tc.M _ when (e lsr i) land 1 = 1 ->
               1 lsl count Mb (List.filteri (fun j _ -> j < i) tc.frag_c.elements)
           | _ -> 0)
         tc.frag_c.elements)
  in
  let first_of s =
    Option.get (List.find_opt (fun e -> slot_of e = s) (List.init ec Fun.id))
  in
  {
    n_;
    m_;
    k_;
    lanes = Tc.threads tc;
    sources;
    keymap;
    tsteps = k_ / n_;
    ea = elems tc.frag_a;
    ec;
    slot_of;
    slots = 1 lsl count Mb tc.frag_c.elements;
    first_of;
  }

(* Slots of the tensor-core kernel's buffers. *)
let slot_acc = 103
let slot_ks = 104
let slot_vs = 105
let slot_red = 106
let slot_sum = 107

(* The padding of a row of a tile in workgroup memory, in elements. *)
let pad = 8

(* [stage ~tid ~threads ~n ~at src dst ~width ~first ~cols] copies [n]
   rows of [src], row [i] where [at ~within i] gives [(ok, start)], of
   [width] elements, the [cols] from [first] on, into [dst]'s rows of
   [cols + pad], eight elements a thread at a time; elements past the row,
   or of a row not [ok], are zeros. *)
let stage ~tid ~threads ~n ~at src dst ~width ~first ~cols =
  let open Ops.O in
  let per = ceil_div cols 8 in
  let total = Stdlib.(n * per) in
  List.concat
    (List.init (ceil_div total threads) (fun i ->
         let cid = tid + int Stdlib.(i * threads) in
         let inside_tile = cid < int total in
         let rl = cid // int per and c0 = cid % int per * int 8 in
         let ok, start = at ~within:inside_tile rl in
         List.init 8 (fun e ->
             let col = c0 + int e in
             let src_col = first + col in
             let v =
               ld_gated src ((start * int width) + src_col)
                 (ok &&: (col < int cols) &&: (src_col < int width))
             in
             Ops.store
               (Ops.index dst
                  [ Ops.where inside_tile ((rl * int Stdlib.(cols + pad)) + col) Ops.invalid ])
               v)))

type core = {
  tc : Tc.t;
  warps : int;
  keys : int;  (** A tile's keys. *)
  splits : int;  (** Runs of each row's keys ({!run_tiles}). *)
  cols : int;  (** The values' columns a workgroup takes. *)
}

let core c a xs =
  let x = args ~scratch:false a xs in
  let tc = c.tc and bk = c.keys in
  let acc = Arith.dtype a.acc and vt = Ops.dtype x.q in
  let f = facts tc in
  let n_ = f.n_ and m_ = f.m_ and k_ = f.k_ and lanes = f.lanes in
  let sources = f.sources and keymap = f.keymap and tsteps = f.tsteps in
  let dkp = ceil_div a.dk k_ * k_ in
  let dcp = ceil_div c.cols n_ * n_ in
  let ksteps = dkp / k_ and ntiles = bk / n_ and vsteps = bk / k_ in
  let otiles = dcp / n_ in
  let bq = c.warps * m_ in
  let gr = grouped a in
  let tiles_q = ceil_div gr bq in
  let chunks = max 1 (ceil_div a.dv c.cols) in
  let heads = numel a.lead * a.hk in
  let threads = c.warps * lanes in
  let ea = f.ea and eb = 1 lsl List.length tc.frag_b.elements and ec = f.ec in
  let slot_of = f.slot_of and slots = f.slots and first_of = f.first_of in
  let sharers = 1 lsl count Nb tc.frag_c.lanes in
  let open Ops.O in
  let gh = range Global (Int heads) 0 in
  let gtr = range Global (Int tiles_q) 1 in
  let gt = int Stdlib.(tiles_q - 1) - gtr in
  let gc = range Global (Int chunks) 2 in
  let splits = c.splits in
  let gs = if Stdlib.(splits > 1) then Some (range Global (Int splits) 6) else None in
  let warp = range Local (Int c.warps) 3 in
  let lane = range Warp (Int lanes) 4 in
  let tid = (warp * int lanes) + lane in
  let b = gh // int a.hk and kh = gh % int a.hk in
  let base = (gt * int bq) + (warp * int m_) in
  let row_of r = row_at a ~b ~kh (base + r) in
  let len = length a x b in
  let first, tiles = run_tiles ~tk:a.tk ~bk ~splits ~gs len in
  let jr = range Loop (Sym tiles) 5 in
  let j = first + jr in
  let at = [ gh; gtr; gc ] @ Option.to_list gs @ [ warp; lane ] in
  let part r =
    (r.index * int splits) + Option.value gs ~default:(int 0)
  in
  (* The lane's share among the lanes holding a row: its column lane bits. *)
  let share =
    let nb = ref 0 in
    List.fold_left ( + ) (int 0)
      (List.concat
         (List.mapi
            (fun i bit ->
              match bit with
              | Tc.N _ ->
                  let k = !nb in
                  incr nb;
                  [ (lane // int (pow2 i)) % int 2 * int (pow2 k) ]
              | _ -> [])
            tc.frag_c.lanes))
  in
  (* Q's operands, one per step of its dimension, in [q]'s dtype. *)
  let qa =
    Array.init ksteps (fun kk ->
        Shape.stack
          (List.init ea (fun e ->
               let r = row_of (coord tc.frag_a Mb lane e) in
               let d = int Stdlib.(kk * k_) + coord tc.frag_a Kb lane e in
               ld_gated x.q ((r.index * int a.dk) + d) (r.live &&: (d < int a.dk)))))
  in
  let m, m_init = reg slot_m slots acc (fl acc Float.neg_infinity) ~at in
  let ls, l_init = reg slot_l slots acc (fl acc 0.) ~at in
  let o, o_init = reg slot_acc Stdlib.(otiles * ec) acc (fl acc 0.) ~at in
  let m_j = Ops.after m [ m_init; jr ] and l_j = Ops.after ls [ l_init; jr ] in
  let o_j = Ops.after o [ o_init; jr ] in
  let ks = Shape.placeholder ~slot:slot_ks ~addrspace:Local [ Stdlib.(bk * (dkp + pad)) ] vt in
  let vs = Shape.placeholder ~slot:slot_vs ~addrspace:Local [ Stdlib.(bk * (dcp + pad)) ] vt in
  let red = Shape.placeholder ~slot:slot_red ~addrspace:Local [ Stdlib.(bq * sharers) ] acc in
  let key i = (j * int bk) + i in
  (* [copy inside src dst ~width ~first ~cols] stages the tile's keys of
     [src] ({!stage}). *)
  let copy inside src dst ~width ~first ~cols =
    let at ~within kl =
      let ky = key kl in
      key_at a x ~inside ~b ~kh ~within:(within &&: (ky < len)) ky
    in
    stage ~tid ~threads ~n:bk ~at src dst ~width ~first ~cols
  in
  let first_col = gc * int c.cols in
  let body inside =
    let ks_w = inside ks in
    let ks = Ops.after ks [ Ops.group (copy inside (inside x.k) ks_w ~width:a.dk ~first:(int 0) ~cols:dkp) ] in
    (* S = Q Kᵀ: the accumulator of each tile of [N] keys. *)
    let s =
      Array.init ntiles (fun nt ->
          let rec go kk accum =
            if kk = ksteps then accum
            else
              let bfrag =
                Shape.stack
                  (List.init eb (fun e ->
                       let kl = int Stdlib.(nt * n_) + coord tc.frag_b Nb lane e in
                       let d = int Stdlib.(kk * k_) + coord tc.frag_b Kb lane e in
                       ld ks ((kl * int Stdlib.(dkp + pad)) + d)))
              in
              go Stdlib.(kk + 1)
                (Ops.wmma qa.(kk) bfrag ~acc:accum ~dims:(n_, m_, k_) ~threads:lanes)
          in
          go 0 (Shape.stack (List.init ec (fun _ -> fl acc 0.))))
    in
    let x2 =
      Array.init ntiles (fun nt ->
          Array.init ec (fun e ->
              let r = row_of (coord tc.frag_c Mb lane e) in
              let ky = key (int Stdlib.(nt * n_) + coord tc.frag_c Nb lane e) in
              let ok, _ =
                key_at a x ~inside ~b ~kh ~within:(r.live &&: (ky < len)) ky
              in
              score a x ~inside ~b ~row:r ~key:ky ~ok
                (Ops.index s.(nt) [ Ops.int ~dtype:Int32 e ])))
    in
    let m_old = Array.init slots (fun s -> ld (inside m_j) (int s)) in
    let l_old = Array.init slots (fun s -> ld (inside l_j) (int s)) in
    (* Each row's maximum over its lane's scores, then over its sharers'. *)
    let part s =
      let mine =
        List.concat
          (List.init ntiles (fun nt ->
               List.filter_map
                 (fun e -> if slot_of e = s then Some x2.(nt).(e) else None)
                 (List.init ec Fun.id)))
      in
      List.fold_left Ops.maximum m_old.(s) mine
    in
    let red_at s = ((warp * int m_) + coord tc.frag_c Mb lane (first_of s)) * int sharers in
    let red_w = inside red in
    let red_st = Ops.group (List.init slots (fun s -> st red_w (red_at s + share) (part s))) in
    let red_r = Ops.after red [ red_st ] in
    let mx =
      Array.init slots (fun s ->
          List.fold_left Ops.maximum m_old.(s)
            (List.init sharers (fun i -> ld red_r (red_at s + int i))))
    in
    let mu =
      Array.map
        (fun mx -> Ops.where (Ops.eq mx (fl acc Float.neg_infinity)) (fl acc 0.) mx)
        mx
    in
    let alpha = Array.init slots (fun s -> Ops.exp2 (m_old.(s) - mu.(s))) in
    let p =
      Array.map (Array.mapi (fun e x -> Ops.exp2 (x - mu.(slot_of e)))) x2
    in
    let l_new =
      Array.init slots (fun s ->
          List.fold_left ( + ) (l_old.(s) * alpha.(s))
            (List.concat
               (List.init ntiles (fun nt ->
                    List.filter_map
                      (fun e -> if slot_of e = s then Some p.(nt).(e) else None)
                      (List.init ec Fun.id)))))
    in
    (* V's tile, copied once the row maximum's loads have read [red]. *)
    let vs_w = Ops.after (inside vs) [ red_r ] in
    let vs =
      Ops.after vs [ Ops.group (copy inside (inside x.v) vs_w ~width:a.dv ~first:first_col ~cols:dcp) ]
    in
    (* P, rounded once to the values' dtype, as each step's operand A. *)
    let pa =
      Array.init vsteps (fun kk ->
          Shape.stack
            (List.init ea (fun e ->
                 let t, e' = sources.(e) in
                 Ops.cast p.(Stdlib.((kk * tsteps) + t)).(e') vt)))
    in
    let o1 =
      Array.init otiles (fun ot ->
          let o0 =
            Shape.stack
              (List.init ec (fun e ->
                   ld (inside o_j) (int Stdlib.((ot * ec) + e)) * alpha.(slot_of e)))
          in
          let rec go kk accum =
            if kk = vsteps then accum
            else
              let bfrag =
                Shape.stack
                  (List.init eb (fun e ->
                       let kl =
                         int Stdlib.(kk * k_) + key_of keymap k_ (coord tc.frag_b Kb lane e)
                       in
                       let col = int Stdlib.(ot * n_) + coord tc.frag_b Nb lane e in
                       ld vs ((kl * int Stdlib.(dcp + pad)) + col)))
              in
              go Stdlib.(kk + 1)
                (Ops.wmma pa.(kk) bfrag ~acc:accum ~dims:(n_, m_, k_) ~threads:lanes)
          in
          go 0 o0)
    in
    let o_w = inside o_j in
    Ops.group
      (List.concat
         [
           List.concat
             (List.init otiles (fun ot ->
                  List.init ec (fun e ->
                      st o_w (int Stdlib.((ot * ec) + e))
                        (Ops.index o1.(ot) [ Ops.int ~dtype:Int32 e ]))));
           List.init slots (fun s -> st (inside m_j) (int s) mx.(s));
           List.init slots (fun s -> st (inside l_j) (int s) l_new.(s));
         ])
  in
  (* A lane's vote: whether its rows see a key of the tile. *)
  let vote inside =
    vote a x ~inside ~b ~kh ~len ~over:[ warp; lane ]
      (List.concat
         (List.init ntiles (fun nt ->
              List.init ec (fun e ->
                  ( row_of (coord tc.frag_c Mb lane e),
                    key (int Stdlib.(nt * n_) + coord tc.frag_c Nb lane e) )))))
  in
  let rows =
    (gt * int bq, Ops.minimum ((gt * int bq) + int Stdlib.(bq - 1)) (int Stdlib.(gr - 1)))
  in
  let keys = (key (int 0), key (int Stdlib.(bk - 1))) in
  let loop =
    Ops.end_
      (skipping a x ~at:(jr :: at) ~b ~kh ~rows ~keys ~vote ~ids:(7, 8, 9) body)
      [ jr ]
  in
  let m_out = Ops.after m [ loop ] and l_out = Ops.after ls [ loop ] in
  let o_out = Ops.after o [ loop ] in
  (* Each row's sum over its sharers, through workgroup memory. *)
  let sums = Shape.placeholder ~slot:slot_sum ~addrspace:Local [ Stdlib.(bq * sharers) ] acc in
  let sum_at s = ((warp * int m_) + coord tc.frag_c Mb lane (first_of s)) * int sharers in
  let sums_st =
    Ops.group (List.init slots (fun s -> st sums (sum_at s + share) (ld l_out (int s))))
  in
  let sums_r = Ops.after sums [ sums_st ] in
  let total =
    Array.init slots (fun s ->
        List.fold_left ( + ) (fl acc 0.)
          (List.init sharers (fun i -> ld sums_r (sum_at s + int i))))
  in
  let out = Arith.dtype a.out in
  let split = Stdlib.(splits > 1) in
  let o_stores =
    List.concat
      (List.init otiles (fun ot ->
           List.init ec (fun e ->
               let r = row_of (coord tc.frag_c Mb lane e) in
               let colw = int Stdlib.(ot * n_) + coord tc.frag_c Nb lane e in
               let col = first_col + colw in
               let raw = ld o_out (int Stdlib.((ot * ec) + e)) in
               let ok = r.live &&: (colw < int c.cols) &&: (col < int a.dv) in
               if split then
                 Ops.store
                   (Ops.index x.o [ Ops.where ok ((part r * int a.dv) + col) Ops.invalid ])
                   raw
               else
                 let v = o_of acc ~l:total.(slot_of e) raw in
                 Ops.store
                   (Ops.index x.o [ Ops.where ok ((r.index * int a.dv) + col) Ops.invalid ])
                   (Ops.cast v out))))
  in
  let lse_stores =
    List.concat
      (List.init slots (fun s ->
           let r = row_of (coord tc.frag_c Mb lane (first_of s)) in
           let first = r.live &&: (Ops.eq share (int 0)) &&: (Ops.eq gc (int 0)) in
           let m = ld m_out (int s) in
           if split then
             [
               Ops.store (Ops.index x.lse [ Ops.where first (int 2 * part r) Ops.invalid ]) m;
               Ops.store
                 (Ops.index x.lse [ Ops.where first ((int 2 * part r) + int 1) Ops.invalid ])
                 total.(s);
             ]
           else
             [
               Ops.store (Ops.index x.lse [ Ops.where first r.index Ops.invalid ])
                 (lse_of acc ~m ~l:total.(s));
             ]))
  in
  let kernel = Ops.kernel_info ~name:"attention" ~opts_to_apply:[] () in
  Ops.sink ~kernel [ Ops.end_ (Ops.group (o_stores @ lse_stores)) at ]

(* The runs a row's keys split into: the least power of two at which the
   launch holds the renderer's [stream_threads] threads, as many loads in
   flight as keep memory busy, and at most one a tile of keys. *)
let splits_for ~stream ~workgroups ~threads ~tiles =
  let rec go s =
    if s >= tiles || workgroups * s * threads >= stream then s else go (2 * s)
  in
  max 1 (min (go 1) tiles)

(* The host's combine kernel: {!combine}'s sums, one thread a row, which
   computes the runs' weights once and adds each run's values to its
   columns, [width] columns at a time ({!simt_width}). *)
let host_combine ~splits ~width a o lse sums parts =
  let acc = Arith.dtype a.acc and out = Arith.dtype a.out in
  let rows = numel a.lead * a.hq * a.tq in
  let open Ops.O in
  let row = range Weak (Int rows) 0 in
  let part s = (row * int splits) + int s in
  let ms = List.init splits (fun s -> ld parts (int 2 * part s)) in
  let m = List.fold_left Ops.maximum (fl acc Float.neg_infinity) ms in
  let mu = Ops.where (Ops.eq m (fl acc Float.neg_infinity)) (fl acc 0.) m in
  let ws = List.map (fun ms -> Ops.exp2 (ms - mu)) ms in
  let lsum =
    List.fold_left ( + ) (fl acc 0.)
      (List.mapi (fun s w -> w * ld parts ((int 2 * part s) + int 1)) ws)
  in
  let lse_st = Ops.store (Ops.index lse [ row ]) (lse_of acc ~m ~l:lsum) in
  let cg = range Loop (Int (max 1 (ceil_div a.dv width))) 1 in
  let e = if Stdlib.(width = 1) then int 0 else range Upcast (Int width) 2 in
  let c = (cg * int width) + e in
  let ok = c < int a.dv in
  let v =
    List.fold_left ( + ) (fl acc 0.)
      (List.mapi (fun s w -> w * ld_gated sums ((part s * int a.dv) + c) ok) ws)
  in
  let o_st =
    Ops.end_
      (Ops.store
         (Ops.index o [ Ops.where ok ((row * int a.dv) + c) Ops.invalid ])
         (Ops.cast (o_of acc ~l:lsum v) out))
      (if Stdlib.(width = 1) then [ cg ] else [ e; cg ])
  in
  let kernel = Ops.kernel_info ~name:"attention_combine" ~opts_to_apply:[] () in
  Ops.sink ~kernel [ Ops.end_ (Ops.group [ lse_st; o_st ]) [ row ] ]

(* The combine kernel

   One thread a column of a row folds the row's [splits] runs in order: [M]
   their maximum, each run's [w = 2^(m - M)], [L = Σ w l], [o = Σ w acc / L]
   and, at the row's first column, [lse = (M + log₂ L) ln 2]. Arguments:
   [o], [lse], [sums], [parts]. *)

let combine ~local ~width ~splits a xs =
  match xs with
  | [ o; lse; sums; parts ] when not local ->
      host_combine ~splits ~width a o lse sums parts
  | [ o; lse; sums; parts ] ->
      let acc = Arith.dtype a.acc and out = Arith.dtype a.out in
      let rows = numel a.lead * a.hq * a.tq in
      let cols = max 1 a.dv in
      let bq = if local then simt_rows else 1 in
      let outer = if local then Ops.Axis_type.Global else Weak in
      let open Ops.O in
      let gr = range outer (Int (ceil_div Stdlib.(rows * cols) bq)) 0 in
      let l = if local then range Local (Int bq) 1 else int 0 in
      let i = (gr * int bq) + l in
      let row = i // int cols and c = i % int cols in
      let live = row < int rows in
      let part s = (row * int splits) + int s in
      let ms = List.init splits (fun s -> ld_gated parts (int 2 * part s) live) in
      let m = List.fold_left Ops.maximum (fl acc Float.neg_infinity) ms in
      let mu = Ops.where (Ops.eq m (fl acc Float.neg_infinity)) (fl acc 0.) m in
      let ws = List.map (fun ms -> Ops.exp2 (ms - mu)) ms in
      let lsum =
        List.fold_left ( + ) (fl acc 0.)
          (List.mapi (fun s w -> w * ld_gated parts ((int 2 * part s) + int 1) live) ws)
      in
      let first = live &&: Ops.eq c (int 0) in
      let v =
        List.fold_left ( + ) (fl acc 0.)
          (List.mapi
             (fun s w ->
               w * ld_gated sums ((part s * int a.dv) + c) (live &&: (c < int a.dv)))
             ws)
      in
      let stores =
        [
          Ops.store (Ops.index lse [ Ops.where first row Ops.invalid ]) (lse_of acc ~m ~l:lsum);
          Ops.store
            (Ops.index o [ Ops.where (live &&: (c < int a.dv)) ((row * int a.dv) + c) Ops.invalid ])
            (Ops.cast (o_of acc ~l:lsum v) out);
        ]
      in
      let ranges = if local then [ gr; l ] else [ gr ] in
      let kernel = Ops.kernel_info ~name:"attention_combine" ~opts_to_apply:[] () in
      Ops.sink ~kernel [ Ops.end_ (Ops.group stores) ranges ]
  | _ -> invalid_arg "Lower_attention: the combine kernel's arguments"

(* [plan renderer a] is the tensor-core kernel's shape for [a] on
   [renderer]'s target, if it takes one: a narrow float accumulated in
   float32 on a core of its products that chains ({!core}). It holds [warps]
   of 8 while the launch keeps the renderer's [local_workgroups] workgroups,
   else 4, as many warps as balance a part's units (FlashAttention-2 §3.3:
   query rows split across warps); tiles of 64 keys while the copies of a
   tile's keys and values fit the workgroup's memory, else 32; and the
   values' columns 64 at a time, so that a lane's accumulators stay in its
   registers. *)
let plan (r : Renderer.t) a =
  let vt = Arith.dtype a.vt in
  let chains (tc : Tc.t) =
    match Tc.chain tc tc A with Some ch -> ch.same_lines | None -> false
  in
  let fits (tc : Tc.t) =
    let n, m, k = Tc.dims tc in
    Dtype.equal tc.dtype_in vt
    && Dtype.equal tc.dtype_out Dtype.Float32
    && m = 16 && n = 8 && k = 16 && chains tc
  in
  let narrow = Dtype.equal vt Bfloat16 || Dtype.equal vt Float16 in
  match List.find_opt fits r.tensor_cores with
  | Some tc
    when narrow && r.has_local && r.has_shared
         && Dtype.equal (Arith.dtype a.acc) Float32 ->
      let _, m, k = Tc.dims tc in
      let cols = min (max a.dv 1) 64 in
      let dkp = ceil_div a.dk k * k and dcp = ceil_div cols 8 * 8 in
      let bytes keys = keys * (dkp + pad + dcp + pad) * Dtype.itemsize vt in
      (* The rows' maxima and sums take 4096 bytes beside the tiles. *)
      let fits keys = bytes keys + 4096 <= r.shared_max in
      if not (fits 32) then None
      else
      let keys = if fits 64 then 64 else 32 in
      let workgroups warps =
        numel a.lead * a.hk * ceil_div (grouped a) (warps * m)
        * max 1 (ceil_div a.dv cols)
      in
      let warps = if workgroups 8 >= r.local_workgroups then 8 else 4 in
      (* Few rows take fewer warps: a decode step's one query of each head. *)
      let warps = min warps (ceil_div (grouped a) m) in
      let splits =
        splits_for ~stream:r.stream_threads ~workgroups:(workgroups warps)
          ~threads:(warps * Tc.threads tc) ~tiles:(ceil_div a.tk keys)
      in
      Some { tc; warps; keys; cols; splits }
  | _ -> None

(* The derivative's kernels

   The transposed derivative ({!Attention}), from the cotangents [ō] and
   [l̄se], recomputes each pair's weight from the call's [lse],
   [p = 2^(x·log₂e - lse·log₂e)], [0] where hidden or where [lse] is [-∞],
   and never stores one (FlashAttention-2's backward, Dao 2023, §3.2 and
   Algorithm 2, without its atomics):

     δ   = ō·o - l̄se                    a kernel of one row a thread
     dP  = ō vᵀ        ds = p (dP - δ)   r̄ the pair program's transpose of ds
     q̄  = scale Σ_j r̄ k_j               by query rows, over their keys
     k̄  = scale Σ_i r̄ q_i               by keys, over their group's rows
     v̄  = Σ_i p ō_i

   For a narrow float on a core that chains as {!core}'s, both kernels run
   their products on it: a warp holds [M] query rows ([dq]) or [M] keys
   ([dkdv]) and their operands in registers, the other side's tiles in
   workgroup memory; [p] and [r̄], rounded once to the operands' dtype, are
   the [A] operands of the products they chain into. The keys' kernel
   computes [Sᵀ = K Qᵀ] and [dPᵀ = V ōᵀ], so that its accumulators chain
   along the rows it sums over. Each sum runs in a fixed order, so the bits
   are the shapes' and the target's. Tiles skip as the forward's. *)

(* A derivative kernel's operands: {!args}'s, [o] being the kernel's first
   output and [lse] the call's, then the cotangent [ō] and the rows' [δ]. *)
type dargs = { x : args; bo : Ops.t; delta : Ops.t }

(* [dargs a ~outs xs] reads [xs]: [outs] outputs, then [q], [k], [v], the
   lengths where given, the captures, [lse], [ō] and [δ]. *)
let dargs a ~outs xs =
  let os = List.filteri (fun i _ -> i < outs) xs in
  let rest = List.filteri (fun i _ -> i >= outs) xs in
  match rest with
  | q :: k :: v :: rest ->
      let len, rest =
        match (a.lengths, rest) with
        | true, l :: rest -> (Some l, rest)
        | _ -> (None, rest)
      in
      let nc = Array.length a.loads in
      let caps = Array.of_list (List.filteri (fun i _ -> i < nc) rest) in
      (match List.filteri (fun i _ -> i >= nc) rest with
      | [ lse; bo; delta ] ->
          let x =
            { o = List.hd os; lse; sum = None; q; k; v; table = None; len; caps }
          in
          (Array.of_list os, { x; bo; delta })
      | _ -> invalid_arg "Lower_attention: the derivative's arguments")
  | _ -> invalid_arg "Lower_attention: the derivative's arguments"

(* The pair of [row] and [key], read where [ok], of raw score [raw]: whether
   it is visible, its score in base 2, and [back ds], the score's cotangent
   [ds] read back to the scaled score by the program's transpose [tr], [0]
   where hidden. *)
let pair_back a x ~tr ~inside ~b ~row ~key ~ok raw =
  let acc = Arith.dtype a.acc in
  let coord = pair_coord a ~b ~row ~key in
  let caps = caps_at a.loads x.caps ~inside ~coord ~ok in
  let s = Ops.mul raw (fl acc a.scale) in
  let seen, x2 = pair_of a ~coord ~ok ~caps s in
  let back ds =
    match tr with
    | None -> fl acc 0.
    | Some tr ->
        let outs =
          Arith.scalar tr ~coord (Array.concat [ [| [ s ] |]; caps; [| [ ds ] |] ])
        in
        Ops.where seen (List.hd outs.(0)) (fl acc 0.)
  in
  (seen, x2, back)

(* A pair's weight from its row's [lse], [0] where hidden or [lse] is
   [-∞]. *)
let weight acc ~seen ~x2 ~lse =
  Ops.where
    (seen &&: Ops.ne lse (fl acc Float.neg_infinity))
    (Ops.exp2 (Ops.sub x2 (Ops.mul lse (fl acc ln2e))))
    (fl acc 0.)

(* The kernel of [δ]: one thread a row, [ō·o] summed in column order, minus
   [l̄se]. Arguments: [δ], [o], [ō], [l̄se]. *)
let delta_kernel a xs =
  match xs with
  | [ delta; o; bo; blse ] ->
      let acc = Arith.dtype a.acc in
      let rows = numel a.lead * a.hq * a.tq in
      let bq = simt_rows in
      let open Ops.O in
      let gr = range Global (Int (ceil_div rows bq)) 0 in
      let l = range Local (Int bq) 1 in
      let i = (gr * int bq) + l in
      let live = i < int rows in
      let r = Shape.placeholder ~slot:slot_m ~addrspace:Reg [ 1 ] acc in
      let init = st (Ops.after r [ gr; l ]) (int 0) (fl acc 0.) in
      let c = range Loop (Int a.dv) 2 in
      let r_c = Ops.after r [ init; c ] in
      let at = (i * int a.dv) + c in
      let term = Ops.cast (ld_gated bo at live) acc * Ops.cast (ld_gated o at live) acc in
      let sum = Ops.end_ (st r_c (int 0) (ld r_c (int 0) + term)) [ c ] in
      let dot = ld (Ops.after r [ sum ]) (int 0) in
      let v = dot - Ops.cast (ld_gated blse i live) acc in
      let store = Ops.store (Ops.index delta [ Ops.where live i Ops.invalid ]) v in
      let kernel = Ops.kernel_info ~name:"attention_delta" ~opts_to_apply:[] () in
      Ops.sink ~kernel [ Ops.end_ store [ gr; l ] ]
  | _ -> invalid_arg "Lower_attention: the δ kernel's arguments"

(* [mma tc ~lane ~steps a b ~acc] is [acc] plus the products of [steps]
   steps of [K], step [kk]'s [A] operand [a kk] and [B] element [e] at
   [b kk ~k ~n], [k] and [n] its fragment's coordinates at [lane]. *)
let mma (tc : Tc.t) ~lane ~steps a b ~acc =
  let n_, m_, k_ = Tc.dims tc in
  let eb = 1 lsl List.length tc.frag_b.elements in
  let rec go kk acc =
    if kk = steps then acc
    else
      let bfrag =
        Shape.stack
          (List.init eb (fun e ->
               b kk ~k:(coord tc.frag_b Kb lane e) ~n:(coord tc.frag_b Nb lane e)))
      in
      go (kk + 1) (Ops.wmma (a kk) bfrag ~acc ~dims:(n_, m_, k_) ~threads:(Tc.threads tc))
  in
  go 0 acc

(* [chain_a f vt p kk] is step [kk]'s [A] operand of the accumulators' elements
   [p.(tile).(e)], each rounded once to [vt]. *)
let chain_a f vt p kk =
  Shape.stack
    (List.init f.ea (fun e ->
         let t, e' = f.sources.(e) in
         Ops.cast p.(Stdlib.((kk * f.tsteps) + t)).(e') vt))

(* An accumulator of zeros. *)
let zeros_c f acc = Shape.stack (List.init f.ec (fun _ -> fl acc 0.))

type dcore = {
  dtc : Tc.t;
  qwarps : int;  (* [dq]'s warps, of [M] rows each. *)
  qkeys : int;  (* [dq]'s keys a tile. *)
  kwarps : int;  (* [dkdv]'s warps, of [M] keys each. *)
  krows : int;  (* [dkdv]'s rows a tile. *)
}

(* By rows: [q̄] of the rows a workgroup holds, over each tile of keys. A
   warp holds [M] rows, their [Q] and [ō] operands in registers; the tile's
   keys and values are staged in workgroup memory. Arguments: [q̄], then
   {!dargs}'s. *)
let core_dq c a tr xs =
  let os, d = dargs a ~outs:1 xs in
  let x = d.x in
  let tc = c.dtc and bk = c.qkeys in
  let f = facts tc in
  let n_ = f.n_ and m_ = f.m_ and k_ = f.k_ in
  let acc = Arith.dtype a.acc and vt = Ops.dtype x.q in
  let dkp = ceil_div a.dk k_ * k_ and dvp = ceil_div a.dv k_ * k_ in
  let ksteps = dkp / k_ and csteps = dvp / k_ in
  let ntiles = bk / n_ and vsteps = bk / k_ and dtiles = dkp / n_ in
  let bq = c.qwarps * m_ in
  let gr = grouped a in
  let tiles_q = ceil_div gr bq in
  let heads = numel a.lead * a.hk in
  let threads = c.qwarps * f.lanes in
  let open Ops.O in
  let gh = range Global (Int heads) 0 in
  let gtr = range Global (Int tiles_q) 1 in
  let gt = int Stdlib.(tiles_q - 1) - gtr in
  let warp = range Local (Int c.qwarps) 3 in
  let lane = range Warp (Int f.lanes) 4 in
  let tid = (warp * int f.lanes) + lane in
  let b = gh // int a.hk and kh = gh % int a.hk in
  let base = (gt * int bq) + (warp * int m_) in
  let row_of r = row_at a ~b ~kh (base + r) in
  let len = length a x b in
  let _, tiles = run_tiles ~tk:a.tk ~bk ~splits:1 ~gs:None len in
  let jr = range Loop (Sym tiles) 5 in
  let at = [ gh; gtr; warp; lane ] in
  let operands buf width steps =
    Array.init steps (fun kk ->
        Shape.stack
          (List.init f.ea (fun e ->
               let r = row_of (coord tc.frag_a Mb lane e) in
               let col = int Stdlib.(kk * k_) + coord tc.frag_a Kb lane e in
               ld_gated buf ((r.index * int width) + col) (r.live &&: (col < int width)))))
  in
  let qa = operands x.q a.dk ksteps and oa = operands d.bo a.dv csteps in
  (* Each row slot's [lse] and [δ]. *)
  let per_slot buf =
    Array.init f.slots (fun s ->
        let r = row_of (coord tc.frag_c Mb lane (f.first_of s)) in
        ld_gated buf r.index r.live)
  in
  let lses = per_slot x.lse and deltas = per_slot d.delta in
  let g, g_init = reg slot_acc Stdlib.(dtiles * f.ec) acc (fl acc 0.) ~at in
  let g_j = Ops.after g [ g_init; jr ] in
  let ks = Shape.placeholder ~slot:slot_ks ~addrspace:Local [ Stdlib.(bk * (dkp + pad)) ] vt in
  let vs = Shape.placeholder ~slot:slot_vs ~addrspace:Local [ Stdlib.(bk * (dvp + pad)) ] vt in
  let key i = (jr * int bk) + i in
  let keys_at inside ~within kl =
    let ky = key kl in
    key_at a x ~inside ~b ~kh ~within:(within &&: (ky < len)) ky
  in
  let body inside =
    let tile src dst width cols =
      Ops.after dst
        [
          Ops.group
            (stage ~tid ~threads ~n:bk ~at:(keys_at inside) (inside src) (inside dst)
               ~width ~first:(int 0) ~cols);
        ]
    in
    let ks = tile x.k ks a.dk dkp and vs = tile x.v vs a.dv dvp in
    (* S = Q Kᵀ and dP = ō Vᵀ, each tile of [N] keys an accumulator. *)
    let scores buf w steps ops =
      Array.init ntiles (fun nt ->
          mma tc ~lane ~steps (fun kk -> ops.(kk))
            (fun kk ~k ~n ->
              let kl = int Stdlib.(nt * n_) + n in
              ld buf ((kl * int Stdlib.(w + pad)) + int Stdlib.(kk * k_) + k))
            ~acc:(zeros_c f acc))
    in
    let s = scores ks dkp ksteps qa and dp = scores vs dvp csteps oa in
    let rbar =
      Array.init ntiles (fun nt ->
          Array.init f.ec (fun e ->
              let sl = f.slot_of e in
              let r = row_of (coord tc.frag_c Mb lane e) in
              let ky = key (int Stdlib.(nt * n_) + coord tc.frag_c Nb lane e) in
              let ok, _ = key_at a x ~inside ~b ~kh ~within:(r.live &&: (ky < len)) ky in
              let el u = Ops.index u [ Ops.int ~dtype:Int32 e ] in
              let seen, x2, back =
                pair_back a x ~tr ~inside ~b ~row:r ~key:ky ~ok (el s.(nt))
              in
              let p = weight acc ~seen ~x2 ~lse:lses.(sl) in
              back (Ops.where seen (p * (el dp.(nt) - deltas.(sl))) (fl acc 0.))))
    in
    (* q̄ += r̄ K, [K] read along its keys. *)
    let g1 =
      Array.init dtiles (fun dt ->
          let g0 =
            Shape.stack
              (List.init f.ec (fun e -> ld (inside g_j) (int Stdlib.((dt * f.ec) + e))))
          in
          mma tc ~lane ~steps:vsteps (chain_a f vt rbar)
            (fun kk ~k ~n ->
              let kl = int Stdlib.(kk * k_) + key_of f.keymap k_ k in
              ld ks ((kl * int Stdlib.(dkp + pad)) + int Stdlib.(dt * n_) + n))
            ~acc:g0)
    in
    let g_w = inside g_j in
    Ops.group
      (List.concat
         (List.init dtiles (fun dt ->
              List.init f.ec (fun e ->
                  st g_w (int Stdlib.((dt * f.ec) + e))
                    (Ops.index g1.(dt) [ Ops.int ~dtype:Int32 e ])))))
  in
  let vote inside =
    vote a x ~inside ~b ~kh ~len ~over:[ warp; lane ]
      (List.concat
         (List.init ntiles (fun nt ->
              List.init f.ec (fun e ->
                  ( row_of (coord tc.frag_c Mb lane e),
                    key (int Stdlib.(nt * n_) + coord tc.frag_c Nb lane e) )))))
  in
  let rows = (gt * int bq, Ops.minimum ((gt * int bq) + int Stdlib.(bq - 1)) (int Stdlib.(gr - 1))) in
  let keys = (key (int 0), key (int Stdlib.(bk - 1))) in
  let loop =
    Ops.end_ (skipping a x ~at:(jr :: at) ~b ~kh ~rows ~keys ~vote ~ids:(7, 8, 9) body) [ jr ]
  in
  let g_out = Ops.after g [ loop ] in
  let stores =
    List.concat
      (List.init dtiles (fun dt ->
           List.init f.ec (fun e ->
               let r = row_of (coord tc.frag_c Mb lane e) in
               let col = int Stdlib.(dt * n_) + coord tc.frag_c Nb lane e in
               let ok = r.live &&: (col < int a.dk) in
               let v = ld g_out (int Stdlib.((dt * f.ec) + e)) * fl acc a.scale in
               Ops.store
                 (Ops.index os.(0) [ Ops.where ok ((r.index * int a.dk) + col) Ops.invalid ])
                 (Ops.cast v vt))))
  in
  let kernel = Ops.kernel_info ~name:"attention_dq" ~opts_to_apply:[] () in
  Ops.sink ~kernel [ Ops.end_ (Ops.group stores) at ]

(* By keys: [k̄] and [v̄] of the keys a workgroup holds, over each tile of
   their group's rows. A warp holds [M] keys, their [K] and [V] operands in
   registers; the tile's query rows and their [ō] are staged in workgroup
   memory. Arguments: [k̄], [v̄], then {!dargs}'s. *)
let core_dkdv c a tr xs =
  let os, d = dargs a ~outs:2 xs in
  let x = d.x in
  let tc = c.dtc and br = c.krows in
  let f = facts tc in
  let n_ = f.n_ and m_ = f.m_ and k_ = f.k_ in
  let acc = Arith.dtype a.acc and vt = Ops.dtype x.q in
  let dkp = ceil_div a.dk k_ * k_ and dvp = ceil_div a.dv k_ * k_ in
  let ksteps = dkp / k_ and csteps = dvp / k_ in
  let ntiles = br / n_ and rsteps = br / k_ in
  let dtiles = dkp / n_ and ctiles = dvp / n_ in
  let bkk = c.kwarps * m_ in
  let gr = grouped a in
  let tiles_k = ceil_div a.tk bkk in
  let tiles_r = ceil_div gr br in
  let heads = numel a.lead * a.hk in
  let threads = c.kwarps * f.lanes in
  let open Ops.O in
  let gh = range Global (Int heads) 0 in
  let gt = range Global (Int tiles_k) 1 in
  let warp = range Local (Int c.kwarps) 3 in
  let lane = range Warp (Int f.lanes) 4 in
  let tid = (warp * int f.lanes) + lane in
  let b = gh // int a.hk and kh = gh % int a.hk in
  let k0 = gt * int bkk in
  let base = k0 + (warp * int m_) in
  let len = length a x b in
  (* A tile of keys past the row's length sees no row. *)
  let jr = range Loop (Sym (Ops.where (k0 < len) (int tiles_r) (int 0))) 5 in
  let at = [ gh; gt; warp; lane ] in
  let key_of_m r = base + r in
  let operands buf width steps =
    Array.init steps (fun kk ->
        Shape.stack
          (List.init f.ea (fun e ->
               let ky = key_of_m (coord tc.frag_a Mb lane e) in
               let ok, at = key_at a x ~inside:Fun.id ~b ~kh ~within:(ky < len) ky in
               let col = int Stdlib.(kk * k_) + coord tc.frag_a Kb lane e in
               ld_gated buf ((at * int width) + col) (ok &&: (col < int width)))))
  in
  let ka = operands x.k a.dk ksteps and va = operands x.v a.dv csteps in
  let gk, gk_init = reg slot_acc Stdlib.(dtiles * f.ec) acc (fl acc 0.) ~at in
  let gv, gv_init = reg slot_sum Stdlib.(ctiles * f.ec) acc (fl acc 0.) ~at in
  let gk_j = Ops.after gk [ gk_init; jr ] and gv_j = Ops.after gv [ gv_init; jr ] in
  let qs = Shape.placeholder ~slot:slot_ks ~addrspace:Local [ Stdlib.(br * (dkp + pad)) ] vt in
  let dos = Shape.placeholder ~slot:slot_vs ~addrspace:Local [ Stdlib.(br * (dvp + pad)) ] vt in
  let row i = row_at a ~b ~kh ((jr * int br) + i) in
  let rows_at ~within rl =
    let r = row rl in
    (within &&: r.live, r.index)
  in
  let body inside =
    let tile src dst width cols =
      Ops.after dst
        [
          Ops.group
            (stage ~tid ~threads ~n:br ~at:rows_at (inside src) (inside dst) ~width
               ~first:(int 0) ~cols);
        ]
    in
    let qs = tile x.q qs a.dk dkp and dos = tile d.bo dos a.dv dvp in
    (* Sᵀ = K Qᵀ and dPᵀ = V ōᵀ, each tile of [N] rows an accumulator. *)
    let scores buf w steps ops =
      Array.init ntiles (fun nt ->
          mma tc ~lane ~steps (fun kk -> ops.(kk))
            (fun kk ~k ~n ->
              let rl = int Stdlib.(nt * n_) + n in
              ld buf ((rl * int Stdlib.(w + pad)) + int Stdlib.(kk * k_) + k))
            ~acc:(zeros_c f acc))
    in
    let s = scores qs dkp ksteps ka and dp = scores dos dvp csteps va in
    let pairs =
      Array.init ntiles (fun nt ->
          Array.init f.ec (fun e ->
              let ky = key_of_m (coord tc.frag_c Mb lane e) in
              let r = row (int Stdlib.(nt * n_) + coord tc.frag_c Nb lane e) in
              let ok, _ = key_at a x ~inside ~b ~kh ~within:(r.live &&: (ky < len)) ky in
              let el u = Ops.index u [ Ops.int ~dtype:Int32 e ] in
              let seen, x2, back =
                pair_back a x ~tr ~inside ~b ~row:r ~key:ky ~ok (el s.(nt))
              in
              let lse = ld_gated (inside x.lse) r.index r.live in
              let delta = ld_gated (inside d.delta) r.index r.live in
              let p = weight acc ~seen ~x2 ~lse in
              (p, back (Ops.where seen (p * (el dp.(nt) - delta)) (fl acc 0.)))))
    in
    let p = Array.map (Array.map fst) pairs and rbar = Array.map (Array.map snd) pairs in
    (* v̄ += Pᵀ ō and k̄ += r̄ᵀ Q, the staged rows read along the rows. *)
    let sums g_j tiles buf w ops =
      Array.init tiles (fun t ->
          let g0 =
            Shape.stack
              (List.init f.ec (fun e -> ld (inside g_j) (int Stdlib.((t * f.ec) + e))))
          in
          mma tc ~lane ~steps:rsteps (chain_a f vt ops)
            (fun kk ~k ~n ->
              let rl = int Stdlib.(kk * k_) + key_of f.keymap k_ k in
              ld buf ((rl * int Stdlib.(w + pad)) + int Stdlib.(t * n_) + n))
            ~acc:g0)
    in
    let v1 = sums gv_j ctiles dos dvp p and k1 = sums gk_j dtiles qs dkp rbar in
    let put g_j tiles g1 =
      let g_w = inside g_j in
      List.concat
        (List.init tiles (fun t ->
             List.init f.ec (fun e ->
                 st g_w (int Stdlib.((t * f.ec) + e)) (Ops.index g1.(t) [ Ops.int ~dtype:Int32 e ]))))
    in
    Ops.group (put gv_j ctiles v1 @ put gk_j dtiles k1)
  in
  let vote inside =
    vote a x ~inside ~b ~kh ~len ~over:[ warp; lane ]
      (List.concat
         (List.init ntiles (fun nt ->
              List.init f.ec (fun e ->
                  ( row (int Stdlib.(nt * n_) + coord tc.frag_c Nb lane e),
                    key_of_m (coord tc.frag_c Mb lane e) )))))
  in
  let rows = (jr * int br, Ops.minimum ((jr * int br) + int Stdlib.(br - 1)) (int Stdlib.(gr - 1))) in
  let keys = (k0, k0 + int Stdlib.(bkk - 1)) in
  let loop =
    Ops.end_ (skipping a x ~at:(jr :: at) ~b ~kh ~rows ~keys ~vote ~ids:(7, 8, 9) body) [ jr ]
  in
  let gk_out = Ops.after gk [ loop ] and gv_out = Ops.after gv [ loop ] in
  let stores out g tiles width scale =
    List.concat
      (List.init tiles (fun t ->
           List.init f.ec (fun e ->
               let ky = key_of_m (coord tc.frag_c Mb lane e) in
               let col = int Stdlib.(t * n_) + coord tc.frag_c Nb lane e in
               let ok = (ky < int a.tk) &&: (col < int width) in
               let at = (((gh * int a.tk) + ky) * int width) + col in
               let v = ld g (int Stdlib.((t * f.ec) + e)) in
               let v = match scale with Some s -> v * fl acc s | None -> v in
               Ops.store (Ops.index out [ Ops.where ok at Ops.invalid ]) (Ops.cast v vt))))
  in
  let kernel = Ops.kernel_info ~name:"attention_dkdv" ~opts_to_apply:[] () in
  Ops.sink ~kernel
    [
      Ops.end_
        (Ops.group (stores os.(0) gk_out dtiles a.dk (Some a.scale) @ stores os.(1) gv_out ctiles a.dv None))
        at;
    ]

(* [dplan renderer a] is the derivative's kernels' shape for [a] on
   [renderer]'s target, if it takes them: {!plan}'s core, dense keys, [o] of
   the query's dtype, no load whose cotangent is asked for, and keys and
   values of at most 64 dimensions. A warp then holds its operands, its
   accumulators and those of a tile of 32 keys or rows in registers, up to
   250 of them on sm_89 with no spill, and two workgroups of 4 warps share a
   compute unit. On fudge's RTX 5000 Ada, tiles of 64 spilled and took up to
   2.1 times as long, 8 warps 1.1 to 1.25 times, and keys of 128 dimensions
   spilled and took twice as long as the derivative's default. *)
let dplan (r : Renderer.t) a ~marked =
  let vt = Arith.dtype a.vt in
  match plan r a with
  | Some c
    when Option.is_none a.paged
         && (not (Array.exists Fun.id marked))
         && Dtype.equal (Arith.dtype a.out) vt
         && a.dk > 0 && a.dv > 0 && a.dk <= 64 && a.dv <= 64 ->
      let _, m, _ = Tc.dims c.tc in
      let warps n = max 1 (min 4 (ceil_div n m)) in
      Some
        {
          dtc = c.tc;
          qwarps = warps (grouped a);
          qkeys = 32;
          kwarps = warps a.tk;
          krows = 32;
        }
  | _ -> None

(* Operands *)

(* [invariant u] is, for each axis of [u], whether [u]'s elements do not
   depend on the index along it: an axis an expansion adds, and one of a
   single element. Where a view's movements are not followed, an axis is
   taken to vary. *)
let rec invariant u =
  let s = Shape.shape u in
  let single = List.map (fun d -> Shape.Sint.equal d (Int 1)) s in
  let follow =
    match Ops.op u with
    | Op.Expand -> (
        match Shape.marg u with
        | Expand front ->
            Some
              (List.map (fun _ -> true) front @ invariant (Ops.nth u 0))
        | _ -> None)
    | Op.Permute -> (
        match Shape.marg u with
        | Permute perm ->
            let inner = Array.of_list (invariant (Ops.nth u 0)) in
            Some (List.map (fun i -> inner.(i)) perm)
        | _ -> None)
    | Op.Reshape ->
        (* A reshape that only adds or drops axes of one element. *)
        let src = Ops.nth u 0 in
        let big l =
          List.filter (fun d -> not (Shape.Sint.equal d (Int 1))) l
        in
        let ss = Shape.shape src in
        if
          List.length (big ss) = List.length (big s)
          && List.for_all2 Shape.Sint.equal (big ss) (big s)
        then
          let inner =
            List.filteri
              (fun i _ -> not (Shape.Sint.equal (List.nth ss i) (Int 1)))
              (invariant src)
          in
          let rest = ref inner in
          Some
            (List.map
               (fun d ->
                 if Shape.Sint.equal d (Int 1) then true
                 else
                   match !rest with
                   | x :: tl ->
                       rest := tl;
                       x
                   | [] -> false)
               s)
        else None
    | _ -> None
  in
  match follow with
  | Some f when List.length f = List.length s ->
      List.map2 ( || ) f single
  | _ -> single

(* A kernel's argument starts on 16 bytes, as its parameter assumes: a view
   at another phase is copied. *)
let arg device v =
  let v = if Option.is_none (Ops.device v) then Ops.copy_to_device v device else v in
  let v = Ops.contiguous v in
  let align, phase = Shape.storage_phase v in
  if align >= 16 && phase mod 16 = 0 then v
  else
    let b = Ops.new_buffer device (Shape.max_numel v) (Ops.dtype v) in
    Ops.after b [ Ops.store b v ]

let flat u = Shape.reshape u [ Ops.Int (Shape.max_numel u) ]

(* [capture device dt u] is the argument of the capture [u], of the pair
   loop's shape, cut to one element along each axis it does not vary on,
   and its strides. *)
let capture device dt u =
  (* An axis of no element has nothing to cut. *)
  let inv =
    List.map2
      (fun i d -> i && not (Shape.Sint.equal d (Int 0)))
      (invariant u) (Shape.shape u)
  in
  let s = List.map2 (fun i d -> if i then Ops.Int 1 else d) inv (Shape.shape u) in
  let cut =
    Shape.shrink u (List.map2 (fun i _ -> if i then Some (Ops.Int 0, Ops.Int 1) else None) inv (Shape.shape u))
  in
  let ext = Array.of_list (List.map (function Ops.Int n -> n | Sym _ -> 1) s) in
  let n = Array.length ext in
  let strides = Array.make n 0 in
  let acc = ref 1 in
  for i = n - 1 downto 0 do
    strides.(i) <- (if ext.(i) = 1 then 0 else !acc);
    acc := !acc * ext.(i)
  done;
  (flat (arg device cut), { dt; strides })

(* Lowering *)

let lower ~device ~(renderer : Renderer.t)
    (node : 'v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) (type r) (op : r Nx.Prim.t)
    : Ops.t array =
  match op with
  | Attention { prog; scale; acc; out; q; kv; lengths; loads } ->
      let sq = Nx.shape q in
      let r = Array.length sq in
      let lead = Array.sub sq 0 (r - 3) in
      let hq = sq.(r - 3) and tq = sq.(r - 2) and dk = sq.(r - 1) in
      let k, v, table, paged =
        match kv with
        | Dense { k; v } -> (k, v, None, None)
        | Paged { k; v; table } ->
            let sk = Nx.shape k in
            let max_pages = (Nx.shape table).(r - 3) in
            ( k,
              v,
              Some table,
              Some { pages = sk.(0); page = sk.(2); max_pages } )
      in
      let sk = Nx.shape k and sv = Nx.shape v in
      let hk, tk =
        match paged with
        | None -> (sk.(r - 3), sk.(r - 2))
        | Some p -> (sk.(1), p.max_pages * p.page)
      in
      let dv = sv.(Array.length sv - 1) in
      let caps =
        Array.map
          (fun (Nx.Prim.Any x) -> capture device (D.Any (Nx.dtype x)) (node x))
          (Iarray.to_array loads)
      in
      let a =
        {
          lead;
          hq;
          hk;
          tq;
          tk;
          dk;
          dv;
          paged;
          lengths = Option.is_some lengths;
          prog;
          scale;
          vt = D.Any (Nx.dtype q);
          acc = D.Any acc;
          out = D.Any out;
          loads = Array.map snd caps;
          classes = classes_of prog r;
        }
      in
      let rows = numel lead * hq * tq in
      let o_shape = Arith.dims a.out (Array.append lead [| hq; tq; dv |]) in
      let lse_shape = Arith.dims a.acc (Array.append lead [| hq; tq |]) in
      (* [c] at each element of [s]; an empty buffer where [s] has none. *)
      let fill d c s =
        if List.exists (fun e -> e = Ops.Int 0) s then
          Shape.empty ~device s (Arith.dtype d)
        else
          Shape.expand
            (Shape.reshape (Ops.const ~dtype:(Arith.dtype d) c)
               (List.map (fun _ -> Ops.Int 1) s))
            s
      in
      if rows = 0 || tk = 0 || hk = 0 then
        [|
          fill a.out (`Float 0.) o_shape;
          fill a.acc (`Float Float.neg_infinity) lse_shape;
        |]
      else
        (* An operand of no element, keys of no dimension or values of no
           column, is a buffer of one the kernel never reads. *)
        let operand x =
          if Array.exists (( = ) 0) (Nx.shape x) then
            Ops.new_buffer device 1 (Arith.dtype (D.Any (Nx.dtype x)))
          else flat (arg device (node x))
        in
        let o = Ops.new_buffer device (max 1 (rows * dv)) (Arith.dtype a.out) in
        let lse = Ops.new_buffer device rows (Arith.dtype a.acc) in
        let inputs =
          [ operand q; operand k; operand v ]
          @ Option.to_list (Option.map operand table)
          @ Option.to_list (Option.map operand lengths)
          @ Array.to_list (Array.map fst caps)
        in
        let local = renderer.has_local && renderer.has_shared in
        let acc_t = Arith.dtype a.acc in
        (* [o] and [lse], or with key splits the runs' [sums] and [parts]
           folded by {!combine}. *)
        let launch splits kernel ~scratch =
          if splits = 1 then Shape.custom_kernel (o :: lse :: scratch @ inputs) kernel
          else
            let sums = Ops.new_buffer device (max 1 (rows * splits * dv)) acc_t in
            let parts = Ops.new_buffer device (rows * splits * 2) acc_t in
            let outs = Shape.custom_kernel (sums :: parts :: scratch @ inputs) kernel in
            Shape.custom_kernel
              [ o; lse; List.nth outs 0; List.nth outs 1 ]
              (combine ~local ~width:(simt_width renderer a) ~splits a)
        in
        let outs =
          match plan renderer a with
          | Some c -> launch c.splits (core c a) ~scratch:[]
          | None ->
              (* A GPU's runs fill [stream_threads] threads. The host's
                 threads, a row each, run on its cores, so its runs fill
                 [local_workgroups] of them, as many as leave each core
                 about as many: a decode step's 8 rows ran on 8 of fudge's
                 14 cores. *)
              let splits =
                if not local then
                  splits_for ~stream:renderer.local_workgroups
                    ~workgroups:(numel lead * a.hk * grouped a)
                    ~threads:1 ~tiles:(ceil_div tk simt_keys)
                else
                  splits_for ~stream:renderer.stream_threads
                    ~workgroups:(numel lead * a.hk * ceil_div (grouped a) (rows_of a))
                    ~threads:(rows_of a) ~tiles:(ceil_div tk simt_keys)
              in
              let sum =
                Ops.new_buffer device (max 1 (rows * splits * dv)) acc_t
              in
              launch splits (simt ~local ~width:(simt_width renderer a) ~splits a)
                ~scratch:[ sum ]
        in
        let o =
          if dv = 0 then fill a.out (`Float 0.) o_shape
          else Shape.reshape (List.nth outs 0) o_shape
        in
        [| o; Shape.reshape (List.nth outs 1) lse_shape |]
  | _ -> invalid_arg ("Lower_attention.lower: " ^ Nx.Prim.name op)

(* The derivative's lowering *)

type operand = D.any * int array * Ops.t

let linear ~device ~(renderer : Renderer.t) (spec : Attention.spec) ~transposed
    (data : operand array) (args : operand array) =
  let n = Array.length data in
  let vt, sq, _ = data.(0) in
  let _, sk, _ = data.(1) in
  let _, sv, _ = data.(2) in
  let r = Array.length sq in
  let lead = Array.sub sq 0 (r - 3) in
  let hq = sq.(r - 3) and tq = sq.(r - 2) and dk = sq.(r - 1) in
  let lens = if spec.lengths then Some data.(3) else None in
  let first = 3 + Bool.to_int spec.lengths in
  let loads = Array.sub data first (n - 2 - first) in
  let out, _, o_node = data.(n - 2) in
  let acc, _, lse_node = data.(n - 1) in
  let hk = sk.(r - 3) and tk = sk.(r - 2) in
  let dv = sv.(Array.length sv - 1) in
  if (not transposed) || spec.paged || numel lead * hq * tq = 0 || tk = 0 || hk = 0
  then None
  else
    let caps = Array.map (fun (dt, _, u) -> capture device dt u) loads in
    let a =
      {
        lead;
        hq;
        hk;
        tq;
        tk;
        dk;
        dv;
        paged = None;
        lengths = spec.lengths;
        prog = spec.prog;
        scale = spec.scale;
        vt;
        acc;
        out;
        loads = Array.map snd caps;
        classes = classes_of spec.prog r;
      }
    in
    match dplan renderer a ~marked:spec.tangents with
    | None -> None
    | Some c ->
        let rows = numel lead * hq * tq in
        let nkeys = numel lead * hk * tk in
        let operand (_, _, u) = flat (arg device u) in
        let buf dt m = Ops.new_buffer device (max 1 m) (Arith.dtype dt) in
        let shaped dt s u = Shape.reshape u (Arith.dims dt s) in
        let bo = operand args.(0) and blse = operand args.(1) in
        let delta =
          List.hd
            (Shape.custom_kernel
               [ buf acc rows; flat (arg device o_node); bo; blse ]
               (delta_kernel a))
        in
        let call =
          [ operand data.(0); operand data.(1); operand data.(2) ]
          @ Option.to_list (Option.map operand lens)
          @ Array.to_list (Array.map fst caps)
          @ [ flat (arg device lse_node); bo; delta ]
        in
        let tr = Attention.score_transpose spec in
        let dq = Shape.custom_kernel (buf vt (rows * dk) :: call) (core_dq c a tr) in
        let kv =
          Shape.custom_kernel
            ([ buf vt (nkeys * dk); buf vt (nkeys * dv) ] @ call)
            (core_dkdv c a tr)
        in
        Some
          [|
            shaped vt sq (List.hd dq);
            shaped vt sk (List.nth kv 0);
            shaped vt sv (List.nth kv 1);
          |]
