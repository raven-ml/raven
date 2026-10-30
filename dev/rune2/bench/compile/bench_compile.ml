(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Cold compiles.

   Each sample runs this executable again as [--cold case] over an empty disk
   cache, so a sample is a fresh process's whole first call of a compiled
   function: tracing, lowering, tolk's passes and the kernel compiler, and the
   process's start. A warm process cannot hold this cost: tolk's disk cache
   serves a program after its first compile. A case takes three samples. The
   cases are those whose compile time rune2's lowering or tolk's vectors
   decide.

   [--cold ID] runs one case's first call and prints its milliseconds: run it
   with [CACHEDB] at an empty path and [PARALLEL=1]. *)

module Rune = Rune2

let f64 = Nx.float64

(* [n] host float64 values in [[lo, hi)]. *)
let input ?(n = 1024) lo hi =
  Nx.place Nx.Host.on
    (Nx.add_s (Nx.mul_s (Nx.Rng.uniform f64 [| n |]) (hi -. lo)) lo)

let compile f x = ignore (Sys.opaque_identity (Rune.jit' f x))

(* A synthetic program of [ops] operations shaped as a special function:
   regions selected by [where] over a parameter and a variable that broadcast
   against each other, each region a chain whose intermediates fan out into
   two uses. A region is [region_ops] operations: its selection, its clamp,
   [region_steps] steps of four operations and the select. A compile time that
   grows faster than the program shows from about 4k operations (rune's
   workload). *)
let region_ops = 64
let region_steps = 14

let shaped ops a x =
  let result = ref (Nx.zeros_like x) in
  for r = 0 to (ops / region_ops) - 1 do
    let lo = Float.of_int r /. 8. in
    let inside =
      Nx.logical_and (Nx.greater_equal_s x lo) (Nx.less_s x (lo +. 0.125))
    in
    let xr = Nx.where inside x (Nx.full_like x (lo +. 0.0625)) in
    let t = ref xr and s = ref (Nx.broadcast_to (Nx.shape x) a) in
    for k = 1 to region_steps do
      t := Nx.add (Nx.mul !t xr) !s;
      s := Nx.sub_s (Nx.mul !s !t) (Float.of_int k)
    done;
    result := Nx.where inside !t !result
  done;
  !result

(* A dense solve of 32 float64 equations: an LU factorization with partial
   pivoting and two triangular solves, each a loop of one step per row. *)
let solve () =
  let m = 32 in
  let a =
    Nx.place Nx.Host.on
      (Nx.add
         (Nx.Rng.uniform f64 [| m; m |])
         (Nx.mul_s (Nx.eye f64 m) (Float.of_int m)))
  in
  let b = input ~n:m 0. 1. in
  compile (fun a -> Nx.solve a b) a

(* The eigendecomposition of 64 symmetric float32 matrices of 8 x 8: 56 rounds
   of Jacobi rotations, held in a loop. *)
let eigh () =
  let a = Nx.Rng.uniform Nx.float32 [| 64; 8; 8 |] in
  let a = Nx.place Nx.Host.on (Nx.add a (Nx.matrix_transpose a)) in
  ignore
    (Sys.opaque_identity
       (Rune.jit
          Nx.Ptree.(tensor @-> returns (pair tensor tensor))
          (fun a -> Nx.eigh Nx.float32 a)
          a))

(* A tridiagonal solve of 100 float64 equations by Thomas's elimination as
   three associative scans: of 2 x 2 matrices as four arrays, for the pivots,
   then of affine maps as pairs, for the eliminated and the substituted
   values. A level is read at two or three positions: inlined whole, the
   levels made one kernel that grew some 3 times per level, and tolk stores
   one once its copies are many. *)
let tridiagonal () =
  let n = 100 in
  let matrix (a00, a01, a10, a11) (b00, b01, b10, b11) =
    let open Nx in
    let p00 = add (mul b00 a00) (mul b01 a10)
    and p01 = add (mul b00 a01) (mul b01 a11)
    and p10 = add (mul b10 a00) (mul b11 a10)
    and p11 = add (mul b10 a01) (mul b11 a11) in
    let s =
      maximum (maximum (abs p00) (abs p01)) (maximum (abs p10) (abs p11))
    in
    let s = where (equal s (zeros_like s)) (ones_like s) s in
    (div p00 s, div p01 s, div p10 s, div p11 s)
  in
  let affine (a1, b1) (a2, b2) = Nx.(mul a1 a2, add (mul a2 b1) b2) in
  let pair = Nx.Ptree.(pair tensor tensor) in
  let quad =
    Nx.Ptree.v (fun c (a, b, e, d) ->
        let open Nx.Ptree.Walk in
        let a = index c 0 tensor a in
        let b = index c 1 tensor b in
        let e = index c 2 tensor e in
        let d = index c 3 tensor d in
        (a, b, e, d))
  in
  let shift_down v =
    let first = Nx.slice [ Nx.R (0, 1) ] v in
    let rest = Nx.slice [ Nx.R (0, n - 1) ] v in
    Nx.concatenate ~axis:0 [ Nx.zeros_like first; rest ]
  in
  let sub = input ~n (-1.) 0. and sup = input ~n (-1.) 0. in
  let r = input ~n 0. 1. in
  let solve diag =
    let _, p01, _, p11 =
      Nx.associative_scan quad matrix
        (Nx.zeros_like diag, sup, Nx.neg sub, diag)
    in
    let ratio = Nx.div p01 p11 in
    let den = Nx.sub diag (Nx.mul sub (shift_down ratio)) in
    let _, d =
      Nx.associative_scan pair affine (Nx.neg (Nx.div sub den), Nx.div r den)
    in
    let flip v = Nx.flip ~axes:[ 0 ] v in
    let _, x = Nx.associative_scan pair affine (flip (Nx.neg ratio), flip d) in
    flip x
  in
  compile solve (input ~n 3. 4.)

(* A sampler's start and first steps on a model of 8 groups (eight schools): a
   location, a log scale and 8 effects, under normal and half-Cauchy priors.
   Each of 8 chains starts at the first of 100 uniform candidates whose log
   density is finite, which a check requires of one, with the last candidate's
   terms as its data, then takes 10 gradient steps, each checking each term
   finite. The checks' first failures are read by their data and the failed
   flag, and carried by the loop with their data, as a sampler's first call
   compiles them. *)
let schools () =
  let y = Nx.create f64 [| 8 |] [| 28.; 8.; -3.; 7.; -1.; 1.; 18.; 12. |] in
  let sigma =
    Nx.create f64 [| 8 |] [| 15.; 10.; 16.; 11.; 9.; 11.; 10.; 18. |]
  in
  let normal x m s =
    Nx.add_s (Nx.mul_s (Nx.square (Nx.div_s (Nx.sub_s x m) s)) (-0.5))
      (-.Float.log s)
  in
  (* Of rows [mu; log tau; eta_1 .. eta_8], one density each. *)
  let log_density q =
    let mu = Nx.slice [ Nx.A; Nx.R (0, 1) ] q in
    let log_tau = Nx.slice [ Nx.A; Nx.R (1, 2) ] q in
    let eta = Nx.slice [ Nx.A; Nx.R (2, 10) ] q in
    let tau = Nx.exp log_tau in
    let theta = Nx.add mu (Nx.mul tau eta) in
    let z = Nx.div (Nx.sub y theta) sigma in
    let cauchy = Nx.neg (Nx.log (Nx.add_s (Nx.square (Nx.div_s tau 5.)) 1.)) in
    let terms =
      [
        normal mu 0. 5.;
        Nx.add cauchy log_tau;
        normal eta 0. 1.;
        Nx.sub (Nx.mul_s (Nx.square z) (-0.5)) (Nx.log sigma);
      ]
    in
    List.map (Nx.sum ~axes:[ 1 ]) terms
  in
  let sum ts = List.fold_left Nx.add (List.hd ts) (List.tl ts) in
  let start k =
    let u = Nx.Rng.uniform_at Nx.Host.on ~key:k f64 [| 100; 10 |] in
    let cands = Nx.mul_s (Nx.sub_s u 0.5) 4. in
    let finite = Nx.isfinite (sum (log_density cands)) in
    let last = List.map (Nx.slice [ Nx.I 99 ]) (log_density cands) in
    Nx.check
      Nx.Ptree.(list tensor)
      (Nx.any finite) last
      (fun _ _ -> Failure "no finite start");
    Nx.take ~axis:0 (Nx.argmax (Nx.cast Nx.int32 finite)) cands
  in
  let step q _ =
    let terms = log_density q in
    let lp = sum terms in
    List.iter
      (fun t ->
        Nx.check
          Nx.Ptree.(pair tensor tensor)
          (Nx.isfinite t) (t, lp)
          (fun _ _ -> Failure "a step left the support"))
      terms;
    let g = Rune.grad' (fun q -> Nx.sum (sum (log_density q))) q in
    (Nx.add q (Nx.mul_s g 0.01), lp)
  in
  let sample k =
    let k = Nx.Rng.of_tensor k in
    let q =
      Rune.vmap
        Nx.Ptree.(v Nx.Rng.walk @-> returns tensor)
        start
        (Nx.Rng.split_batch ~n:8 k)
    in
    let q, lps = Rune.scan' ~f:step ~init:q (Nx.zeros f64 [| 10; 1 |]) in
    Nx.add (Nx.sum q) (Nx.sum lps)
  in
  compile sample (Nx.Rng.to_tensor (Nx.Rng.key 0))

(* A scan of 20 steps whose step checks each of its 48 values finite, with the
   value and the step's sum as data: a loop that carries each check's first
   failure and data, as a sampler's step carries its sites' checks. *)
let checked_scan () =
  let n = 48 in
  let step x _ =
    let y = Nx.add (Nx.mul_s x 0.5) (Nx.recip (Nx.add_s (Nx.square x) 1.)) in
    let total = Nx.sum y in
    for i = 0 to n - 1 do
      let v = Nx.slice [ Nx.I i ] y in
      Nx.check
        Nx.Ptree.(pair tensor tensor)
        (Nx.isfinite v) (v, total)
        (fun _ _ -> Failure "a value left the reals")
    done;
    (y, total)
  in
  let run x =
    let y, totals = Rune.scan' ~f:step ~init:x (Nx.zeros f64 [| 20; 1 |]) in
    Nx.add (Nx.sum y) (Nx.sum totals)
  in
  compile run (input ~n 0. 1.)

(* [shaped 4096] of a 2-element state padded to 3: the pad masks the state's
   axis, which the host upcasts whole, so each operation is of 3 lanes, a
   vector of the state's loads and the pad. *)
let shaped_padded () =
  compile
    (fun x -> shaped 4096 (Nx.scalar f64 0.5) (Nx.pad [| (0, 1) |] 0.5 x))
    (input ~n:2 0. 1.)

let cases =
  [
    ("linalg/solve-32", solve);
    ("linalg/eigh-8x8", eigh);
    (* A float64 sine of an unbounded angle, reduced by the bits of 1/(2pi) at
       its exponent: its lowering's size. *)
    ("trig/sin", fun () -> compile Nx.sin (input (-10.) 10.));
    ("scan/tridiagonal-100", tridiagonal);
    ("model/schools-start", schools);
    ("scan/checks-48", checked_scan);
    ( "shaped-4k",
      fun () -> compile (shaped 4096 (Nx.scalar f64 0.5)) (input 0. 1.) );
    ("shaped-4k-3-lanes", shaped_padded);
  ]

let rec remove path =
  if Sys.is_directory path then (
    Array.iter (fun f -> remove (Filename.concat path f)) (Sys.readdir path);
    Sys.rmdir path)
  else Sys.remove path

(* One sample: a fresh process compiling [id] over an empty disk cache. *)
let cold id () =
  let exe = Sys.executable_name in
  let dir = Filename.temp_dir "bench-compile" "" in
  let env =
    Array.append
      [| "CACHEDB=" ^ Filename.concat dir "cache.db"; "PARALLEL=1" |]
      (Unix.environment ())
  in
  let status =
    Fun.protect ~finally:(fun () -> remove dir) @@ fun () ->
    let null = Unix.openfile Filename.null [ Unix.O_WRONLY ] 0 in
    Fun.protect ~finally:(fun () -> Unix.close null) @@ fun () ->
    let pid =
      Unix.create_process_env exe [| exe; "--cold"; id |] env Unix.stdin null
        Unix.stderr
    in
    snd (Unix.waitpid [] pid)
  in
  match status with
  | Unix.WEXITED 0 -> ()
  | _ -> failwith ("bench_compile: the cold compile of " ^ id ^ " failed")

let config =
  Thumper.Config.(
    default |> samples 3 |> warmup 0. |> deadline 300.
    |> metrics [ Thumper.Metric.wall_time ])

let () =
  match Array.to_list Sys.argv with
  | [ _; "--cold"; id ] -> (
      match List.assoc_opt id cases with
      | Some f ->
          let t0 = Unix.gettimeofday () in
          Nx.Rng.with_key (Nx.Rng.key 42) f;
          Printf.printf "%.3f\n" ((Unix.gettimeofday () -. t0) *. 1000.)
      | None ->
          prerr_endline ("bench_compile: no case " ^ id);
          exit 2)
  | _ ->
      Thumper.run "compile" ~config
        ~budgets:[ Thumper.Budget.no_slower_than 0.05 ]
        (List.map (fun (id, _) -> Thumper.bench id (cold id)) cases)
      |> exit
