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
   cases are those whose compile time rune2's lowering decides.

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

let cases =
  [
    ("linalg/solve-32", solve);
    ("linalg/eigh-8x8", eigh);
    (* A float64 sine of an unbounded angle, reduced by the bits of 1/(2pi) at
       its exponent: its lowering's size. *)
    ("trig/sin", fun () -> compile Nx.sin (input (-10.) 10.));
    ( "shaped-4k",
      fun () -> compile (shaped 4096 (Nx.scalar f64 0.5)) (input 0. 1.) );
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
