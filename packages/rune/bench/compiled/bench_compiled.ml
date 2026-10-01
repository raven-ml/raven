(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rune's Compiled backend, kernel by kernel. A compiled gather costs one
   load per output only where tolk folds the one-hot sum its lowering builds
   into a load at the computed index. Without the fold, the 1Mi row is a sum of
   2^40 terms, so its baseline guards the fold.

   A top_k over a short axis, eagerly and compiled for the host. An axis of at
   most 32 entries is ranked by counting, n * n comparisons a row: compiled,
   that is 3 kernels whatever k; eagerly, it costs more than passes over the
   axis would.

   On Metal and on CUDA, chains of one operation a call on 1024 float32
   elements, each reading the result of the one before: one kernel, and five in
   turn. The device is opened in the measuring worker, which is forked without
   an exec. Metal's compiler service cannot be reached from such a process: it
   answers only from Metal's cache of pipelines. Before measuring, a fresh
   process of this executable ([--warm]) runs every chain's setup, which makes
   each pipeline into that cache. CUDA's driver must not be initialized before
   the fork, so a fresh process ([--cuda]) says whether a CUDA device opens. *)

let kernels = Nx_backend.kernels Rune.compiled

let host_array x =
  match Nx.Repr.v (Nx.copy x) with
  | Host a -> a
  | Placed _ | Traced _ -> invalid_arg "bench_compiled: not a host value"

(* [n] float32 values gathered at [n] uniform indices in [0, n). Compilation
   happens in the warm-up. *)
let gather id n =
  let (module K : Nx_backend.S) = kernels in
  Thumper.bench_with_setup ~id
    ~setup:(fun () ->
      let st = Random.State.make [| 15 |] in
      let x = Nx.init Nx.float32 [| n |] (fun _ -> Random.State.float st 1.) in
      let i =
        Nx.init Nx.int64 [| n |] (fun _ -> Int64.of_int (Random.State.int st n))
      in
      (host_array i, host_array x, host_array (Nx.zeros Nx.float32 [| n |])))
    id
    (fun (i, x, dst) ->
      K.gather ~axis:0 i x ~dst;
      Nx_device.synchronize Nx_device.host)

(* [k] of [n] float32 entries in each of [rows] rows. The compiled function is
   traced and compiled in the setup's first call. *)
let topk ~k ~n ~rows =
  let x () =
    let st = Random.State.make [| 15 |] in
    Nx.init Nx.float32 [| rows; n |] (fun _ -> Random.State.float st 1.)
  in
  let top x = Nx.top_k ~k ~axis:1 x in
  let id = Printf.sprintf "%d-of-%d-%d-rows" k n rows in
  Thumper.group ~id:"topk" "topk"
    [
      Thumper.bench_with_setup ~setup:x (id ^ "-eager") (fun x ->
          ignore (top x);
          Nx_device.synchronize Nx_device.host);
      Thumper.bench_with_setup
        ~setup:(fun () ->
          let f =
            Rune.jit
              Nx.Ptree.(tensor @-> returns (pair tensor tensor))
              top
          in
          let x = x () in
          ignore (f x);
          (f, x))
        (id ^ "-compiled")
        (fun (f, x) ->
          ignore (f x);
          Nx_device.synchronize Nx_device.host);
    ]

(* [n] float64 queries searched among [m] sorted knots, the knots captured by a
   compiled function. Compilation happens in the warm-up. *)
let searchsorted id ~n ~m =
  Thumper.bench_with_setup ~id
    ~setup:(fun () ->
      let st = Random.State.make [| 15 |] in
      let uniform k =
        Nx.init Nx.float64 [| k |] (fun _ -> Random.State.float st 1.)
      in
      let knots = fst (Nx.sort (uniform m)) in
      let f =
        Rune.jit' (fun q -> Nx.searchsorted ~side:`Right knots q)
      in
      (f, uniform n))
    id
    (fun (f, q) ->
      ignore (f q);
      Nx_device.synchronize Nx_device.host)

(* Chains on a GPU *)

type chain = {
  device : Nx_device.t;
  x : (float, Nx_dtype.float32_elt) Nx_array.t;
  mutable a : (float, Nx_dtype.float32_elt) Nx_array.t;
  mutable b : (float, Nx_dtype.float32_elt) Nx_array.t;
  mutable step : int;
}

(* A chain: its name, its setup, and its timed call. *)
type case = { name : string; setup : unit -> chain; call : chain -> unit }

let n = 1024

let chains open_device =
  let (module K) = kernels in
  let array d =
    let buffer = Nx_device.Buffer.create d Nx_dtype.Scalar.Float32 n in
    Nx_device.Buffer.copy
      ~src:
        (Nx_device.Buffer.of_bigarray
           (Bigarray.(Array1.init float32 c_layout n) (fun i ->
                1. +. (float_of_int i /. 1e4))))
      ~dst:buffer;
    {
      Nx_array.dtype = Nx_dtype.Float32;
      view = Nx_array.View.create [| n |];
      buffer;
    }
  in
  let chain name kinds =
    let setup () =
      let device = open_device () in
      let c =
        {
          device;
          x = array device;
          a = array device;
          b = array device;
          step = 0;
        }
      in
      Array.iter (fun k -> K.binary k c.x c.x ~dst:c.a) kinds;
      Nx_device.synchronize device;
      c
    in
    let call c =
      K.binary kinds.(c.step mod Array.length kinds) c.a c.x ~dst:c.b;
      let a = c.a in
      c.a <- c.b;
      c.b <- a;
      c.step <- c.step + 1
    in
    { name; setup; call }
  in
  [
    chain "chain_one_add" Nx_backend.[| Add |];
    chain "chain_distinct" Nx_backend.[| Add; Mul; Sub; Maximum; Minimum |];
  ]

let teardown c = Nx_device.synchronize c.device

let group name open_device =
  Thumper.group ~id:name name
    (List.map
       (fun c ->
         Thumper.bench_with_setup c.name ~setup:c.setup ~teardown c.call)
       (chains open_device))

let run_self flag =
  Sys.command (Filename.quote_command Sys.executable_name [ flag ])

let metal args =
  match Metal.device with
  | None -> []
  | Some open_device ->
      if (not (List.mem "list" args)) && run_self "--warm" <> 0 then
        failwith "the pipelines could not be made";
      [ group "metal" open_device ]

let cuda () =
  if run_self "--cuda" <> 0 then []
  else [ group "cuda" (fun () -> Nx_cuda_device.v 0) ]

let () =
  match (Array.to_list Sys.argv, Metal.device) with
  | [ _; "--warm" ], Some open_device ->
      List.iter (fun c -> teardown (c.setup ())) (chains open_device)
  | [ _; "--cuda" ], _ ->
      exit (if Result.is_ok (Nx_cuda_device.get 0) then 0 else 1)
  | args, _ ->
      Thumper.run "compiled"
        ~budgets:
          [
            Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05;
            Thumper.Budget.no_more_alloc_than 0.01;
          ]
        (Thumper.group ~id:"gather" "gather"
           [ gather "float32-1Mi-from-1Mi-host" (1 lsl 20) ]
        :: topk ~k:4 ~n:32 ~rows:512
        :: Thumper.group ~id:"searchsorted" "searchsorted"
             [
               searchsorted "float64-1e6-into-1e3-host" ~n:1_000_000 ~m:1_000;
               searchsorted "float64-1e6-into-1e6-host" ~n:1_000_000
                 ~m:1_000_000;
             ]
        :: (metal args @ cuda ()))
