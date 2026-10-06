(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx.Ptree

(* Stan's windows: an initial buffer, slow windows that double in length, the
   last stretched to the final buffer, and the final buffer; each window says
   whether the geometry is refitted at its end. Too few steps for a slow window
   adapt the step size alone. *)
let schedule n =
  if n < 20 then if n = 0 then [] else [ (n, false) ]
  else
    let first, last, base =
      if 75 + 50 + 25 > n then
        let first = int_of_float (0.15 *. float_of_int n)
        and last = int_of_float (0.1 *. float_of_int n) in
        (first, last, n - first - last)
      else (75, 50, 25)
    in
    let slow_end = n - last in
    let rec slow start size acc =
      if start >= slow_end then List.rev acc
      else
        let size =
          if start + (3 * size) > slow_end then slow_end - start else size
        in
        slow (start + size) (2 * size) ((size, true) :: acc)
    in
    List.filter
      (fun (n, _) -> n > 0)
      ([ (first, false) ] @ slow first base [] @ [ (last, false) ])

let windows schedule =
  let n = List.length schedule in
  let lengths = List.map (fun (n, _) -> Int32.of_int n) schedule in
  let refits = List.map (fun (_, r) -> if r then 1l else 0l) schedule in
  ( Nx.create Nx.int32 [| n |] (Array.of_list lengths),
    Nx.create Nx.int32 [| n |] (Array.of_list refits) )

(* Dual averaging (Hoffman and Gelman 2014), Stan's constants. *)

let da_gamma = 0.05
let da_kappa = 0.75
let da_t0 = 10.

type 'f averaging = {
  mu : (float, 'f) Nx.t;
  s_bar : (float, 'f) Nx.t;
  x_bar : (float, 'f) Nx.t;
  count : (float, 'f) Nx.t; (* steps since the last restart *)
}

type 'f averaging' = 'f averaging

let averaging_ptree (type f) () : f averaging P.t =
  let module S = struct
    type _ t = f averaging'

    let walk c (a : f averaging) : f averaging =
      let open P.Walk in
      let mu = field c "mu" tensor a.mu in
      let s_bar = field c "s_bar" tensor a.s_bar in
      let x_bar = field c "x_bar" tensor a.x_bar in
      let count = field c "count" tensor a.count in
      { mu; s_bar; x_bar; count }
  end in
  P.nest (module S) P.unit

let restart eps =
  {
    mu = Nx.log (Nx.mul_s eps 10.);
    s_bar = Nx.zeros_like eps;
    x_bar = Nx.zeros_like eps;
    count = Nx.zeros (Nx.dtype eps) [||];
  }

let average a ~target stat =
  let dt = Nx.dtype a.mu in
  let count = Nx.add_s a.count 1. in
  let eta = Nx.recip (Nx.add_s count da_t0) in
  let s_bar =
    Nx.add
      (Nx.mul (Nx.sub (Nx.ones_like eta) eta) a.s_bar)
      (Nx.mul eta (Nx.sub target stat))
  in
  let x =
    Nx.sub a.mu (Nx.div (Nx.mul s_bar (Nx.sqrt count)) (Nx.scalar dt da_gamma))
  in
  let x_eta = Nx.pow count (Nx.scalar dt (-.da_kappa)) in
  let x_bar =
    Nx.add (Nx.mul (Nx.sub (Nx.ones_like x_eta) x_eta) a.x_bar) (Nx.mul x_eta x)
  in
  ({ a with s_bar; x_bar; count }, Nx.exp x)

let final a = Nx.exp a.x_bar

(* A window's draws and scores *)

type 'u window = {
  n : Nx.int32_t; (* the window's draws so far *)
  shift : 'u; (* the window's first position *)
  sx : 'u;
  sxx : 'u;
  sg : 'u;
  sgg : 'u;
  xs : 'u; (* the window's draws and scores, for a low-rank fit *)
  gs : 'u;
}

type 'u window' = 'u window

let window_ptree (type u) (u : u P.t) : u window P.t =
  let module S = struct
    type _ t = u window'

    let walk c (w : u window) : u window =
      let open P.Walk in
      let n = field c "n" tensor w.n in
      let shift = field c "shift" (structure u) w.shift in
      let sx = field c "sx" (structure u) w.sx in
      let sxx = field c "sxx" (structure u) w.sxx in
      let sg = field c "sg" (structure u) w.sg in
      let sgg = field c "sgg" (structure u) w.sgg in
      let xs = field c "xs" (structure u) w.xs in
      let gs = field c "gs" (structure u) w.gs in
      { n; shift; sx; sxx; sg; sgg; xs; gs }
  end in
  P.nest (module S) P.unit

let empty u ~buffer x =
  let buffers =
    P.map u
      (fun _ t ->
        let sh = Nx.shape t in
        Nx.zeros (Nx.dtype t)
          (Array.concat
             [ [| sh.(0); buffer |]; Array.sub sh 1 (Array.length sh - 1) ]))
      x
  in
  let z = Rows.zeros u x in
  {
    n = Nx.scalar Nx.int32 0l;
    shift = x;
    sx = z;
    sxx = z;
    sg = z;
    sgg = z;
    xs = buffers;
    gs = buffers;
  }

let reopen u w x =
  {
    w with
    n = Nx.scalar Nx.int32 0l;
    shift = x;
    sx = Rows.zeros u w.sx;
    sxx = Rows.zeros u w.sxx;
    sg = Rows.zeros u w.sg;
    sgg = Rows.zeros u w.sgg;
  }

let sq u x = P.map u (fun _ t -> Nx.mul t t) x

(* The sums are shifted by the window's first position. A buffer of length 0
   records nothing. *)
let record u w x g =
  let dx = P.map2 u (fun _ x sh -> Nx.sub x sh) x w.shift in
  let at buf v =
    P.map2 u
      (fun _ b v ->
        if (Nx.shape b).(1) = 0 then b
        else
          Nx.set
            [ Nx.A; Nx.D (Nx.cast Nx.int64 w.n, 1) ]
            (Nx.unsqueeze ~axes:[ 1 ] v)
            b)
      buf v
  in
  {
    n = Nx.add w.n (Nx.scalar Nx.int32 1l);
    shift = w.shift;
    sx = Rows.add u w.sx dx;
    sxx = Rows.add u w.sxx (sq u dx);
    sg = Rows.add u w.sg g;
    sgg = Rows.add u w.sgg (sq u g);
    xs = at w.xs x;
    gs = at w.gs g;
  }

(* Fisher fits

   The Gaussian that minimises the Fisher divergence to draws and their scores
   (Seyboldt, Carlson and Carpenter 2026): a diagonal scale [sqrt (sd x / sd
   score)] per element, clipped where a score does not vary, then the [rank]
   directions along which the draws' and the scores' covariances, whitened by
   it, disagree most. *)

(* The variance limits of a fit. *)
let variance_low = 1e-20
let variance_high = 1e20

let clip x =
  let dt = Nx.dtype x in
  Nx.clamp
    ~min:(Nx_dtype.of_float dt variance_low)
    ~max:(Nx_dtype.of_float dt variance_high)
    x

let scale u vx vg =
  P.map2 u
    (fun _ vx vg ->
      let r = Nx.sqrt (Nx.div vx vg) in
      Nx.sqrt (clip (Nx.where (Nx.isnan r) (Nx.ones_like r) r)))
    vx vg

(* [rows_matrix u dt x] is [x], each tensor with a leading axis of [L], as an
   [[L; d]] matrix of the float elements; [matrix_rows u like m] reads the [k]
   rows of an [[k; d]] matrix back into [like]'s structure. *)
let rows_matrix (type f) u (dt : (float, f) Nx.dtype) x : (float, f) Nx.t =
  let rows =
    P.fold u
      (fun _ t acc ->
        if not (Rows.float_leaf t) then acc
        else
          let s = Nx.shape t in
          let n =
            Array.fold_left ( * ) 1 (Array.sub s 1 (Array.length s - 1))
          in
          Nx.cast dt (Nx.reshape [| s.(0); n |] t) :: acc)
      x []
  in
  Nx.concatenate ~axis:1 (List.rev rows)

let matrix_rows u like m =
  let k = (Nx.shape m).(0) in
  let offset = ref 0 in
  P.map u
    (fun _ t ->
      let shape = Array.append [| k |] (Nx.shape t) in
      if not (Rows.float_leaf t) then Nx.zeros (Nx.dtype t) shape
      else
        let n = Nx.numel t in
        let cols = Nx.shrink [| (0, k); (!offset, !offset + n) |] m in
        offset := !offset + n;
        Nx.reshape shape (Nx.cast (Nx.dtype t) cols))
    like

(* The regularisation of the covariances a low-rank fit compares. *)
let low_rank_ridge = 1e-5

(* [low_rank u dt ~rank valid m s xs gs] is the Gaussian of mean [m], diagonal
   scale [s], and the [rank] directions where the draws [xs] and scores [gs]
   whose row is [valid], whitened by [s], disagree most. In the span [Q] of
   both, the covariance minimising the Fisher divergence is the geometric mean
   of the draws' covariance [C_x] and the inverse of the scores' [C_g]; its
   eigenvalues farthest from 1 in ratio give the directions. *)
let low_rank (type f) u (dt : (float, f) Nx.dtype) ~rank valid m s xs gs =
  let x = rows_matrix u dt xs and g = rows_matrix u dt gs in
  let l = (Nx.shape x).(0) in
  let valid = Nx.reshape [| l; 1 |] (Nx.cast dt valid) in
  let count = Nx.sum valid in
  let row t =
    rows_matrix u dt (P.map u (fun _ t -> Nx.unsqueeze ~axes:[ 0 ] t) t)
  in
  let x = Nx.mul valid (Nx.div (Nx.sub x (row m)) (row s)) in
  let g_mean =
    Nx.div (Nx.sum ~axes:[ 0 ] ~keepdims:true (Nx.mul valid g)) count
  in
  let g = Nx.mul valid (Nx.mul (Nx.sub g g_mean) (row s)) in
  let q, _ = Nx.qr (Nx.transpose (Nx.concatenate ~axis:0 [ x; g ])) in
  let r = (Nx.shape q).(1) in
  let ridge = Nx.mul_s (Nx.eye dt r) low_rank_ridge in
  let cov a =
    let p = Nx.matmul a q in
    Nx.add (Nx.div (Nx.matmul (Nx.transpose p) p) count) ridge
  in
  let cx = cov x and cg = cov g in
  let power m e =
    let w, v = Nx.eigh m in
    let w = Nx.pow (Nx.cast dt w) (Nx.scalar dt e) in
    Nx.matmul (Nx.mul v (Nx.unsqueeze ~axes:[ 0 ] w)) (Nx.transpose v)
  in
  let half = power cg 0.5 and inv_half = power cg (-0.5) in
  let inner = power (Nx.matmul half (Nx.matmul cx half)) 0.5 in
  let sigma = Nx.matmul inv_half (Nx.matmul inner inv_half) in
  let w, v = Nx.eigh sigma in
  let w = Nx.cast dt w in
  let order = Nx.argsort ~descending:true (Nx.abs (Nx.log w)) in
  let keep = Nx.shrink [| (0, min rank r) |] order in
  let variances = Nx.take ~indices:keep w in
  let directions =
    Nx.transpose (Nx.matmul q (Nx.take ~axis:1 ~indices:keep v))
  in
  let directions, variances =
    if rank <= r then (directions, variances)
    else
      (* Fewer independent draws than directions: the rest change nothing. *)
      ( Nx.concatenate ~axis:0
          [ directions; Nx.zeros dt [| rank - r; (Nx.shape q).(0) |] ],
        Nx.concatenate ~axis:0 [ variances; Nx.ones dt [| rank - r |] ] )
  in
  Gaussian.low_rank u ~mean:m ~scale:s
    ~directions:(matrix_rows u m directions)
    ~variances:(clip variances)

(* [moments u n shift s ss] is the mean and variance of [n] values whose sums,
   shifted by [shift], are [s] and whose sums of squares are [ss]. *)
let moments u n shift s ss =
  let mean =
    P.map2 u
      (fun _ s sh -> Nx.add (Nx.div s (Nx.cast (Nx.dtype s) n)) sh)
      s shift
  in
  let var =
    P.map2 u
      (fun _ s ss ->
        let n = Nx.cast (Nx.dtype s) n in
        let m = Nx.div s n in
        Nx.maximum (Nx.sub (Nx.div ss n) (Nx.mul m m)) (Nx.zeros_like m))
      s ss
  in
  (mean, var)

let per_chain (type u f) (u : u P.t) (dt : (float, f) Nx.dtype) ~rank
    (w : u window) : (u, f) Gaussian.t =
  let mean, vx = moments u w.n w.shift w.sx w.sxx in
  let _, vg = moments u w.n (Rows.zeros u w.sg) w.sg w.sgg in
  let s = scale u vx vg in
  let gp = Gaussian.ptree u in
  if rank = 0 then
    Rune.vmap
      P.(u @-> u @-> returns gp)
      (fun m s -> Gaussian.diagonal u dt ~mean:m ~scale:s)
      mean s
  else
    let buffer = P.fold u (fun _ t _ -> (Nx.shape t).(1)) w.xs 0 in
    let valid = Nx.less (Nx.arange Nx.int32 0 buffer 1) w.n in
    Rune.vmap
      P.(u @-> u @-> u @-> u @-> returns gp)
      (fun m s xs gs -> low_rank u dt ~rank valid m s xs gs)
      mean s w.xs w.gs

(* The chains' draws pooled: the variance across them is the mean of the chains'
   variances plus the variance of their means. *)
let pooled (type u f) (u : u P.t) (dt : (float, f) Nx.dtype) ~rank
    (w : u window) : (u, f) Gaussian.t =
  let pool (mean, var) =
    let m = P.map u (fun _ t -> Nx.mean ~axes:[ 0 ] t) mean in
    let v =
      P.map2 u
        (fun _ v mean ->
          let d = Nx.sub mean (Nx.mean ~axes:[ 0 ] ~keepdims:true mean) in
          Nx.add (Nx.mean ~axes:[ 0 ] v) (Nx.mean ~axes:[ 0 ] (Nx.square d)))
        var mean
    in
    (m, v)
  in
  let mean, vx = pool (moments u w.n w.shift w.sx w.sxx) in
  let _, vg = pool (moments u w.n (Rows.zeros u w.sg) w.sg w.sgg) in
  let s = scale u vx vg in
  if rank = 0 then Gaussian.diagonal u dt ~mean ~scale:s
  else
    let c, buffer =
      P.fold u (fun _ t _ -> ((Nx.shape t).(0), (Nx.shape t).(1))) w.xs (0, 0)
    in
    let flat x =
      P.map u
        (fun _ t ->
          let sh = Nx.shape t in
          Nx.reshape
            (Array.append
               [| c * buffer |]
               (Array.sub sh 2 (Array.length sh - 2)))
            t)
        x
    in
    let draw = Nx.arange Nx.int32 0 (c * buffer) 1 in
    let valid =
      Nx.less (Nx.mod_ draw (Nx.scalar Nx.int32 (Int32.of_int buffer))) w.n
    in
    low_rank u dt ~rank valid mean s (flat w.xs) (flat w.gs)
