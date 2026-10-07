(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Cold compiles.

   Each call runs this executable again as [--cold case] over an empty disk
   cache, so a sample is a fresh process's whole first call of a compiled
   function: tracing, lowering, tolk's passes and the kernel compiler, and the
   process's start, about 25 ms. A case takes five samples, whose median a
   function's compile is held to. The special functions and the shaped programs
   compile at float64 over a 1-D input on the host.

   [sinkhorn-64] compiles 64 Sinkhorn iterations unrolled, and their gradient,
   on the host: 1,024 kernels, most of them equal, so scheduling them is about
   half of its compile, and a cost that grew with the square of the kernels
   would show.

   [search/<device>] is the first call of a function compiled with a beam search
   of width 2 on [search_parallel] domains, then [calls] more: one product of
   two float32 square matrices, on the host and on each GPU the machine has. The
   search compiles, loads and times each candidate; the calls run the kernel it
   chose, and take longer if its timings misled it.

   [--cold ID] runs one case's first call, of a row or of [once], and prints its
   milliseconds: run with [CACHEDB] at an empty path and [PARALLEL=1]. The cases
   [once] lists are not rows, since a compile of several seconds, repeated five
   times, would not fit the suite's time: the incomplete gamma's derivatives,
   and its inverse and theirs. *)

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

(* A double-word sum of a million float64 numbers. *)
let wide_sum () =
  let n = 1_000_000 in
  let w =
    Nx_wide.v
      ~lo:(Nx.mul_s (Nx.rand Nx.float64 [| n |]) 0x1p-60)
      (Nx.rand Nx.float64 [| n |])
  in
  let p = Nx_wide.ptree Nx.float64 in
  ignore
    (Sys.opaque_identity
       (Rune.jit Nx.Ptree.(p @-> returns p) (fun w -> Nx_wide.sum w) w))

(* A function of a parameter and a variable, over parameters in [a_lo, a_hi] and
   variables in [lo, hi], and its derivative in the parameter or in the
   variable. *)
let binary f (a_lo, a_hi) (lo, hi) () =
  let x = input lo hi in
  compile (fun a -> f a x) (input a_lo a_hi)

let in_a f a x = Rune.grad' (fun a -> Nx.sum (f a x)) a
let in_x f a x = Rune.grad' (fun x -> Nx.sum (f a x)) x
let shape = (0.1, 50.)
let variable = (0., 60.)
let probability = (0., 1.)

(* The incomplete gamma's long compiles, run one at a time with [--cold]. *)
let once =
  [
    ("special/gammainc-grad-a", binary (in_a Nx.gammainc) shape variable);
    ("special/gammainc-grad-x", binary (in_x Nx.gammainc) shape variable);
    ("special/gammaincinv", binary Nx.gammaincinv shape probability);
    ( "special/gammaincinv-grad-a",
      binary (in_a Nx.gammaincinv) shape probability );
    ( "special/gammaincinv-grad-p",
      binary (in_x Nx.gammaincinv) shape probability );
  ]

(* Sinkhorn iterations in the log domain between two uniform batches, over a
   cost matrix of [sinkhorn_rows] rows, and the gradient of the transport cost
   with respect to the cost. *)
let sinkhorn_rows = 64

let sinkhorn iterations c =
  let n = Nx.dim 0 c in
  let log_weight = -.log (Float.of_int n) in
  let k = Nx.neg c in
  let u = ref (Nx.zeros_like (Nx.slice [ A; R (0, 1) ] c)) in
  let v = ref (Nx.zeros_like (Nx.slice [ R (0, 1); A ] c)) in
  for _ = 1 to iterations do
    u :=
      Nx.neg
        (Nx.sub_s
           (Nx.logsumexp ~axes:[ 1 ] ~keepdims:true (Nx.add k !v))
           log_weight);
    v :=
      Nx.neg
        (Nx.sub_s
           (Nx.logsumexp ~axes:[ 0 ] ~keepdims:true (Nx.add k !u))
           log_weight)
  done;
  Nx.sum (Nx.mul (Nx.exp (Nx.add (Nx.add k !u) !v)) c)

let sinkhorn_case iterations () =
  let c = Nx.rand Nx.float64 [| sinkhorn_rows; sinkhorn_rows |] in
  ignore (Sys.opaque_identity (Rune.jit' (Rune.grad' (sinkhorn iterations)) c))

(* Searches *)

let search_parallel = 8

(* The search's product on the host is of 64 rows, called 100 times; on a GPU,
   of 1,024 rows, called 200 times. *)
type search = { rows : int; calls : int }

let on_host = { rows = 64; calls = 100 }
let on_gpu = { rows = 1024; calls = 200 }

(* The devices a search runs on: the host, and each GPU that opens. *)
let devices =
  [
    ("host", on_host, fun () -> Some Nx.Device.host);
    ("metal", on_gpu, fun () -> Result.to_option (Nx_metal.get 0));
    ("cuda", on_gpu, fun () -> Result.to_option (Nx_cuda.get 0));
    ("nv", on_gpu, fun () -> Result.to_option (Nx_nv.get 0));
    ("amd", on_gpu, fun () -> Result.to_option (Nx_amd.get 0));
  ]

let search_case search open_device () =
  let d = Option.get (open_device ()) in
  let n = search.rows in
  let x = Nx.place (Nx.Placement.on d) (Nx.rand Nx.float32 [| n; n |]) in
  let f =
    Rune.jit' ~beam:2 ~parallel:search_parallel (fun x -> Nx.matmul x x)
  in
  let m = Nx.Device.memory d in
  for _ = 0 to search.calls do
    ignore (Sys.opaque_identity (f x));
    Nx_device.synchronize m
  done

let search_cases =
  List.map
    (fun (name, search, d) -> ("search/" ^ name, search_case search d))
    devices

(* A dense solve of a float64 system of [m] equations: an LU factorization with
   partial pivoting and two triangular solves, each a loop of one step per
   row. *)
let solve m () =
  let a =
    Nx.add
      (Nx.rand Nx.float64 [| m; m |])
      (Nx.mul_s (Nx.eye Nx.float64 m) (float_of_int m))
  in
  let b = Nx.rand Nx.float64 [| m |] in
  ignore (Sys.opaque_identity (Rune.jit' (fun a -> Nx.solve a b) a))

(* [betainc] and [log_betainc] of three inputs, the derivative of [log_betainc]
   in each and of [betainc] in [a], over shapes spread across TOMS 708's
   regions. *)
let betainc =
  let a = Nx.exp (input (-5.) 12.) and b = Nx.exp (input (-5.) 12.) in
  let x = input 0. 1. in
  let compile3 f =
    let g =
      Rune.jit Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor) f
    in
    ignore (Sys.opaque_identity (g a b x))
  in
  let sum f a b x = Nx.sum (f a b x) in
  [
    ("special/betainc", fun () -> compile3 Nx.betainc);
    ( "special/betainc-grad-a",
      fun () ->
        compile3 (fun a b x -> Rune.grad' (fun a -> sum Nx.betainc a b x) a) );
    ("special/log_betainc", fun () -> compile3 Nx.log_betainc);
    ( "special/log_betainc-grad-x",
      fun () ->
        compile3 (fun a b x -> Rune.grad' (fun x -> sum Nx.log_betainc a b x) x)
    );
    ( "special/log_betainc-grad-a",
      fun () ->
        compile3 (fun a b x -> Rune.grad' (fun a -> sum Nx.log_betainc a b x) a)
    );
    ( "special/log_betainc-grad-b",
      fun () ->
        compile3 (fun a b x -> Rune.grad' (fun b -> sum Nx.log_betainc a b x) b)
    );
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
  @ special "i0e" Nx.i0e (-30.) 30.
  @ special "i1e" Nx.i1e (-30.) 30.
  @ lbeta @ betainc
  @ [ ("wide/sum-1e6", wide_sum) ]
  @ List.map
      (fun m -> (Printf.sprintf "linalg/solve-%d" m, solve m))
      [ 32; 128; 512 ]
  @ [ ("sinkhorn-64", sinkhorn_case 64) ]
  @ search_cases
  @ [ ("special/gammainc", binary Nx.gammainc shape variable) ]

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
    default |> samples 5 |> warmup 0. |> deadline infinity
    |> metrics [ Thumper.Metric.wall_time ])

let () =
  match Array.to_list Sys.argv with
  | [ _; "--opens"; name ] ->
      let _, _, open_device = List.find (fun (n, _, _) -> n = name) devices in
      let opens = Option.is_some (open_device ()) in
      exit (if opens then 0 else 1)
  | [ _; "--cold"; id ] -> (
      match List.assoc_opt id (cases @ once) with
      | Some f ->
          let t0 = Unix.gettimeofday () in
          Nx.Rng.with_key (Nx.Rng.key 42) f;
          Printf.printf "%.3f\n" ((Unix.gettimeofday () -. t0) *. 1000.)
      | None ->
          prerr_endline ("bench_compile: no case " ^ id);
          exit 2)
  | _ ->
      (* A search runs on a device that opens, asked of a fresh process: a GPU's
         driver must not be initialized in this one. *)
      let opens id =
        match String.split_on_char '/' id with
        | [ "search"; name ] ->
            Sys.command
              (Filename.quote_command Sys.executable_name [ "--opens"; name ])
            = 0
        | _ -> true
      in
      Thumper.run "compile" ~config
        ~budgets:[ Thumper.Budget.no_slower_than 0.05 ]
        (List.filter_map
           (fun (id, _) ->
             if opens id then Some (Thumper.bench id (cold id)) else None)
           cases)
      |> exit
