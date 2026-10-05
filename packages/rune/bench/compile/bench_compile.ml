(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Cold compiles.

   Each call runs this executable again as [--cold case] over an empty disk
   cache, so a sample is a fresh process's whole first call of a compiled
   function: tracing, lowering, tolk's passes and the kernel compiler, and the
   process's start, about 25 ms. A case takes five samples, whose median a
   function's compile is held to. Every case compiles at float64 over a 1-D
   input on the host. *)

let n = 1024
let input lo hi = Nx.add_s (Nx.mul_s (Nx.rand Nx.float64 [| n |]) (hi -. lo)) lo
let compile f x = ignore (Sys.opaque_identity (Rune.jit' f x))
let grad f = Rune.grad' (fun x -> Nx.sum (f x))

(* Synthetic programs of [ops] operations shaped as nx's special functions are:
   regions selected by [where] over a parameter and a variable that broadcast
   against each other, each region a chain whose intermediates fan out into two
   uses. A region is [region_ops] operations: its selection, its clamp,
   [region_steps] steps of four operations and the select. *)
let region_ops = 64
let region_steps = 14

let shaped ops a x =
  let result = ref (Nx.zeros_like x) in
  for r = 0 to (ops / region_ops) - 1 do
    let lo = float_of_int r /. 8. in
    let inside =
      Nx.logical_and (Nx.greater_equal_s x lo) (Nx.less_s x (lo +. 0.125))
    in
    let xr = Nx.where inside x (Nx.full_like x (lo +. 0.0625)) in
    let t = ref xr and s = ref (Nx.broadcast_to (Nx.shape x) a) in
    for k = 1 to region_steps do
      t := Nx.add (Nx.mul !t xr) !s;
      s := Nx.sub_s (Nx.mul !s !t) (float_of_int k)
    done;
    result := Nx.where inside !t !result
  done;
  !result

let shaped_case ops () =
  compile (shaped ops (Nx.scalar Nx.float64 0.5)) (input 0. 1.)

(* A special function of one argument and its derivative, over inputs in [lo,
   hi]. *)
let special name f lo hi =
  [
    ("special/" ^ name, fun () -> compile f (input lo hi));
    ("special/" ^ name ^ "-grad", fun () -> compile (grad f) (input lo hi));
  ]

(* [lbeta] and its derivative in each argument. *)
let lbeta =
  let b = input 0.5 20. in
  let compile2 f = compile (fun a -> f a b) (input 0.5 20.) in
  [
    ("special/lbeta", fun () -> compile2 Nx.lbeta);
    ( "special/lbeta-grad-a",
      fun () ->
        compile2 (fun a b -> Rune.grad' (fun a -> Nx.sum (Nx.lbeta a b)) a) );
    ( "special/lbeta-grad-b",
      fun () ->
        compile2 (fun a b -> Rune.grad' (fun b -> Nx.sum (Nx.lbeta a b)) b) );
  ]

let cases =
  [
    ("erfinv", fun () -> compile Nx.erfinv (input (-1.) 1.));
    ("erfinv-grad", fun () -> compile (grad Nx.erfinv) (input (-1.) 1.));
    ("shaped-1k", shaped_case 1024);
    ("shaped-4k", shaped_case 4096);
    ("shaped-16k", shaped_case 16384);
  ]
  @ special "erfc" Nx.erfc (-6.) 6.
  @ special "ndtr" Nx.ndtr (-10.) 10.
  @ special "log_ndtr" Nx.log_ndtr (-40.) 10.
  @ special "ndtri" Nx.ndtri 0. 1.
  @ special "lgamma" Nx.lgamma (-10.) 20.
  @ special "digamma" Nx.digamma (-10.) 20.
  @ lbeta

let rec remove path =
  if Sys.is_directory path then (
    Array.iter (fun f -> remove (Filename.concat path f)) (Sys.readdir path);
    Sys.rmdir path)
  else Sys.remove path

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
    let pid =
      Unix.create_process_env exe [| exe; "--cold"; id |] env Unix.stdin
        Unix.stdout Unix.stderr
    in
    snd (Unix.waitpid [] pid)
  in
  match status with
  | Unix.WEXITED 0 -> ()
  | _ -> failwith ("bench_compile: the cold compile of " ^ id ^ " failed")

let config =
  Thumper.Config.(
    default |> samples 5 |> warmup 0. |> deadline infinity
    |> metrics [ Thumper.Metric.wall_time ])

let () =
  match Array.to_list Sys.argv with
  | [ _; "--cold"; id ] -> (
      match List.assoc_opt id cases with
      | Some f -> Nx.Rng.with_key (Nx.Rng.key 42) f
      | None ->
          prerr_endline ("bench_compile: no case " ^ id);
          exit 2)
  | _ ->
      Thumper.run "compile" ~config
        ~budgets:[ Thumper.Budget.no_slower_than 0.05 ]
        (List.map (fun (id, _) -> Thumper.bench id (cold id)) cases)
      |> exit
