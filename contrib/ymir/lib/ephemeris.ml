(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type host = (float, Nx.float64_elt, Nx.host) Nx.t

let day_s = 86400.
let mjd_j2000 = 51544.5
let year_days = 365.25

(* Terms per block, as {!Earth}'s series. *)
let block = 256

(* The model's terms of one power [k] of T, per block: the phases [b] and
   frequencies [c], [[b]], and into the six columns, the weights [a] of
   [cos (b + c T)] and [a c] of [sin (b + c T)], [[b; 6]]. *)
type part = { phases : host; frequencies : host; cos : host; sin : host }

let columns = 6
let width = 5

(* [part terms j b] is the block of [b] of [terms] from [j]. *)
let part terms j b =
  let term x = terms.(j + x) in
  let get f =
    Nx.create Nx.float64 [| b |] (Array.init b (fun x -> f (term x)))
  in
  let weights f =
    Nx.create Nx.float64 [| b; columns |]
      (Array.init (b * columns) (fun x ->
           let r = term (x / columns) in
           if int_of_float r.(1) = x mod columns then f r else 0.))
  in
  {
    phases = get (fun r -> r.(3));
    frequencies = get (fun r -> r.(4));
    cos = weights (fun r -> r.(2));
    sin = weights (fun r -> r.(2) *. r.(4));
  }

(* The blocks of each power of T, 0 to 2. *)
let parts =
  lazy
    (let rows = Earth_tables.ephemeris in
     let row r = Array.sub rows (r * width) width in
     let all = List.init (Array.length rows / width) row in
     let of_power k =
       let terms =
         Array.of_list (List.filter (fun r -> r.(0) = float_of_int k) all)
       in
       let n = Array.length terms in
       List.init
         ((n + block - 1) / block)
         (fun i -> part terms (i * block) (min block (n - (i * block))))
     in
     List.init 3 of_power)

(* [sums t parts] is [(C, S)] over [parts]: [Σ a cos (b + c T)] and
   [Σ a c sin (b + c T)] per column, [[...; 6]], [t] [[...; 1]]. *)
let sums t parts =
  let on = Nx.beside t in
  let batch = Array.sub (Nx.shape t) 0 (Nx.ndim t - 1) in
  let zero =
    Nx.zeros_like (Nx.broadcast_to (Array.append batch [| columns |]) t)
  in
  List.fold_left
    (fun (c, s) p ->
      let place x = Nx.place on x in
      let arg = Nx.add (Nx.mul t (place p.frequencies)) (place p.phases) in
      ( Nx.add c (Nx.matmul (Nx.cos arg) (place p.cos)),
        Nx.add s (Nx.matmul (Nx.sin arg) (place p.sin)) ))
    (zero, zero) parts

let rotation =
  lazy (Nx.create Nx.float64 [| 3; 3 |] Earth_tables.ephemeris_rotation)

(* [rotate on v] is [v], [[...; 3]] on the model's axes, on BCRS's. *)
let rotate on v =
  Nx.matmul v (Nx.matrix_transpose (Nx.place on (Lazy.force rotation)))

(* [slice v lo] is the columns [lo] to [lo + 2] of [v], [[...; 6]]. *)
let slice v lo =
  Nx.slice (List.init (Nx.ndim v - 1) (fun _ -> Nx.A) @ [ Nx.R (lo, lo + 3) ]) v

(* [earth tdb] is [(ph, vh, pb, vb)]: the Earth's heliocentric and
   barycentric positions in au and velocities in au/d on BCRS's axes, at
   TDB [tdb] (eraEpv00). The model's position in each column is
   [Σ_k T^k C_k], T in Julian years, and its velocity the derivative
   [Σ_k (k T^(k-1) C_k − T^k S_k)] per year. *)
let earth (tdb : (Time.tdb, 'd) Time.t) =
  let days =
    Nx.add
      (Nx.sub_s (Nx.cast Nx.float64 tdb.Time.day) mjd_j2000)
      (Nx.div_s tdb.Time.second day_s)
  in
  let t = Nx.unsqueeze ~axes:[ -1 ] (Nx.div_s days year_days) in
  let on = Nx.beside t in
  let (c0, s0), (c1, s1), (c2, s2) =
    match List.map (sums t) (Lazy.force parts) with
    | [ p0; p1; p2 ] -> (p0, p1, p2)
    | _ -> assert false
  in
  let position = Nx.add c0 (Nx.mul t (Nx.add c1 (Nx.mul t c2))) in
  let velocity =
    Nx.add
      (Nx.add (Nx.neg s0) (Nx.sub c1 (Nx.mul t s1)))
      (Nx.mul t (Nx.sub (Nx.mul_s c2 2.) (Nx.mul t s2)))
  in
  let velocity = Nx.div_s velocity year_days in
  let helio x = slice x 0 and bary x = Nx.add (slice x 0) (slice x 3) in
  ( rotate on (helio position),
    rotate on (helio velocity),
    rotate on (bary position),
    rotate on (bary velocity) )
