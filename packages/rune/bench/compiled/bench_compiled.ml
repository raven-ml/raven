(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Compiled host programs.

   A top_k over a short axis, eagerly and compiled for the host. An axis of at
   most 32 entries is ranked by counting, n * n comparisons a row: compiled,
   that is 3 kernels whatever k; eagerly, it costs more than passes over the
   axis would.

   The launches run on the host and on CUDA. The CUDA device is opened in the
   measuring worker, which is forked without an exec. CUDA's driver must not be
   initialized before the fork, so a fresh process ([--cuda]) says whether a
   CUDA device opens. *)

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
            Rune.jit Nx.Ptree.(tensor @-> returns (pair tensor tensor)) top
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
      let f = Rune.jit' (fun q -> Nx.searchsorted ~side:`Right knots q) in
      (f, uniform n))
    id
    (fun (f, q) ->
      ignore (f q);
      Nx_device.synchronize Nx_device.host)

(* Host programs. Each is compiled in its setup's first call and timed as one
   call of the compiled function. *)

let uniform shape =
  let st = Random.State.make [| 15 |] in
  Nx.init Nx.float32 shape (fun _ -> Random.State.float st 2. -. 1.)

let int64s shape bound =
  let st = Random.State.make [| 16 |] in
  Nx.init Nx.int64 shape (fun _ -> Int64.of_int (Random.State.int st bound))

let compiled_call id signature f inputs =
  Thumper.bench_with_setup ~id
    ~setup:(fun () ->
      let inputs = inputs () in
      let g = Rune.jit signature f in
      ignore (Sys.opaque_identity (inputs g));
      (g, inputs))
    id
    (fun (g, inputs) ->
      ignore (Sys.opaque_identity (inputs g));
      Nx_device.synchronize Nx_device.host)

(* A host kernel's output loop runs in blocks on every core: a 16Mi-element a +
   b * c, and the sums of 4096 rows of 4096. A reduction's own loop stays whole
   in each block. *)
let split =
  let n = 1 lsl 24 in
  Thumper.group ~id:"split" "split"
    [
      compiled_call "fma-float32-16Mi-host"
        Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
        (fun a b c -> Nx.add a (Nx.mul b c))
        (fun () ->
          let a = uniform [| n |] and b = uniform [| n |] in
          let c = uniform [| n |] in
          fun g -> g a b c);
      compiled_call "row-sums-float32-4096x4096-host"
        Nx.Ptree.(tensor @-> returns tensor)
        (fun x -> Nx.sum ~axes:[ 1 ] x)
        (fun () ->
          let x = uniform [| 4096; 4096 |] in
          fun g -> g x);
    ]

(* Gathers are loads at computed indices, and a scatter into a value the call
   consumes stores its rows at their indices: 512 rows of a 32768 x 1024 table,
   8 of 1024 entries in each of 4096 rows, and 16 rows of 1024 written into a
   4096-row cache that each call takes from the one before. *)
let indexed =
  let rows = 32768 and width = 1024 in
  Thumper.group ~id:"indexed" "indexed"
    [
      compiled_call "take-512-rows-of-32768x1024-host"
        Nx.Ptree.(tensor @-> tensor @-> returns tensor)
        (fun table i -> Nx.take ~axis:0 ~indices:i table)
        (fun () ->
          let table = uniform [| rows; width |] in
          let i = int64s [| 512 |] rows in
          fun g -> g table i);
      compiled_call "take-along-8-of-4096x1024-host"
        Nx.Ptree.(tensor @-> tensor @-> returns tensor)
        (fun x i -> Nx.take_along_axis ~axis:1 ~indices:i x)
        (fun () ->
          let x = uniform [| 4096; width |] in
          let i = int64s [| 4096; 8 |] width in
          fun g -> g x i);
      Thumper.bench_with_setup ~id:"scatter-16-rows-into-4096x1024-cache-host"
        ~setup:(fun () ->
          let k = 16 and slots = 4096 in
          let write cache values i =
            Nx.scatter ~unique_indices:true ~axis:0
              ~indices:
                (Nx.broadcast_to [| k; width |] (Nx.reshape [| k; 1 |] i))
              ~values cache
          in
          let g =
            Rune.jit
              Nx.Ptree.(consumes tensor @@ tensor @-> tensor @-> returns tensor)
              write
          in
          let values = uniform [| k; width |] in
          let i =
            Nx.create Nx.int64 [| k |]
              (Array.init k (fun r -> Int64.of_int (r * (slots / k))))
          in
          let cache =
            ref (g (Nx.zeros Nx.float32 [| slots; width |]) values i)
          in
          (g, cache, values, i))
        "scatter-16-rows-into-4096x1024-cache-host"
        (fun (g, cache, values, i) ->
          cache := g !cache values i;
          Nx_device.synchronize Nx_device.host);
    ]

(* Rotary embeddings of 512 positions over 64 heads of 64: the cosines and sines
   of the positions' angles are computed once a position and read by every head,
   which a kernel recomputing them per element would not. *)
let rope =
  let t = 512 and heads = 64 and d = 64 in
  let half = d / 2 in
  let rotate x pos =
    let freqs =
      Nx.exp
        (Nx.mul_s
           (Nx.arange_f Nx.float32 0. (Float.of_int half) 1.)
           (-.log 10000. /. Float.of_int half))
    in
    let angles =
      Nx.mul (Nx.reshape [| t; 1 |] pos) (Nx.reshape [| 1; half |] freqs)
    in
    let c = Nx.reshape [| t; 1; half |] (Nx.cos angles) in
    let s = Nx.reshape [| t; 1; half |] (Nx.sin angles) in
    let x1 = Nx.slice [ A; A; R (0, half) ] x in
    let x2 = Nx.slice [ A; A; R (half, d) ] x in
    Nx.concatenate ~axis:2
      [ Nx.sub (Nx.mul x1 c) (Nx.mul x2 s); Nx.add (Nx.mul x2 c) (Nx.mul x1 s) ]
  in
  Thumper.group ~id:"rope" "rope"
    [
      compiled_call "float32-512-positions-64x64-host"
        Nx.Ptree.(tensor @-> tensor @-> returns tensor)
        rotate
        (fun () ->
          let x = uniform [| t; heads; d |] in
          let pos = Nx.arange_f Nx.float32 0. (Float.of_int t) 1. in
          fun g -> g x pos);
    ]

(* Selections of a value against zero, a chain over 1Mi float32 elements: on
   arm64, each zero sits in a register clang cannot read as a literal. *)
let select_zero =
  let n = 1 lsl 20 in
  let chain x =
    let z = Nx.zeros_like x in
    let neg = Nx.where (Nx.less x z) x z in
    let pos = Nx.where (Nx.greater x z) x z in
    Nx.add (Nx.mul_s neg 0.25) (Nx.where (Nx.less pos z) z pos)
  in
  Thumper.group ~id:"select-zero" "select-zero"
    [
      compiled_call "float32-1Mi-host"
        Nx.Ptree.(tensor @-> returns tensor)
        chain
        (fun () ->
          let x = uniform [| n |] in
          fun g -> g x);
    ]

(* Reverse mode of a two-layer perceptron's loss over a batch of 32 rows of 64
   inputs, 128 hidden units and 10 outputs: the gradient of the compiled loss, a
   forward program that returns the values its backward program reads, then that
   program; and the compiled gradient, one program. *)
let reverse =
  let batch = 32 and inputs = 64 and hidden = 128 and outputs = 10 in
  let params = Nx.Ptree.(pair (pair tensor tensor) (pair tensor tensor)) in
  let loss ((w1, b1), (w2, b2)) x =
    let h = Nx.tanh (Nx.add (Nx.matmul x w1) b1) in
    let y = Nx.add (Nx.matmul h w2) b2 in
    Nx.mean (Nx.mul y y)
  in
  let init () =
    ( ( Nx.mul_s (uniform [| inputs; hidden |]) 0.1,
        Nx.zeros Nx.float32 [| hidden |] ),
      ( Nx.mul_s (uniform [| hidden; outputs |]) 0.1,
        Nx.zeros Nx.float32 [| outputs |] ) )
  in
  let x () = uniform [| batch; inputs |] in
  let timed id step =
    Thumper.bench_with_setup ~id
      ~setup:(fun () ->
        let p = init () and x = x () in
        ignore (Sys.opaque_identity (step p x));
        (p, x))
      id
      (fun (p, x) ->
        ignore (Sys.opaque_identity (step p x));
        Nx_device.synchronize Nx_device.host)
  in
  Thumper.group ~id:"reverse" "reverse"
    [
      timed "grad-of-jit-mlp-float32-host"
        (let l =
           Rune.jit Nx.Ptree.(params @-> tensor @-> returns tensor) loss
         in
         fun p x -> Rune.grad params (fun p -> l p x) p);
      timed "jit-of-grad-mlp-float32-host"
        (Rune.jit
           Nx.Ptree.(params @-> tensor @-> returns params)
           (fun p x -> Rune.grad params (fun p -> loss p x) p));
    ]

(* Launches

   Compiled functions of many small kernels, whose time is mostly the cost of
   launching them: a chain of 64 dependent products of 16 x 16 matrices, one
   kernel each, and a scan of 256 such steps, four kernels each, run as a loop
   of the compiled program. On the host, and on a GPU, where each call
   synchronizes the device. *)

let launch_dim = 16
let chain_kernels = 64
let scan_steps = 256

let product_chain x =
  let x = ref x in
  for _ = 1 to chain_kernels do
    x := Nx.tanh (Nx.matmul !x !x)
  done;
  !x

let product_scan h xs =
  fst
    (Rune.scan'
       ~f:(fun h x ->
         let h = Nx.tanh (Nx.add (Nx.matmul h h) x) in
         (h, Nx.sum h))
       ~init:h xs)

let launches ?(prefix = "") ~place ~sync () =
  let st = Random.State.make [| 15 |] in
  let uniform shape =
    place (Nx.init Nx.float32 shape (fun _ -> Random.State.float st 0.1))
  in
  let h () = uniform [| launch_dim; launch_dim |] in
  [
    Thumper.bench_with_setup
      ~setup:(fun () ->
        let f = Rune.jit' product_chain and x = h () in
        ignore (f x);
        sync ();
        (f, x))
      (Printf.sprintf "%schain-%d" prefix chain_kernels)
      (fun (f, x) ->
        ignore (f x);
        sync ());
    Thumper.bench_with_setup
      ~setup:(fun () ->
        let f =
          Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) product_scan
        in
        let x = h ()
        and xs = uniform [| scan_steps; launch_dim; launch_dim |] in
        ignore (f x xs);
        sync ();
        (f, x, xs))
      (Printf.sprintf "%sscan-%d" prefix scan_steps)
      (fun (f, x, xs) ->
        ignore (f x xs);
        sync ());
  ]

(* The launches on a GPU's device, which opens in the measuring worker. *)
let gpu_launches open_device =
  let device = lazy (open_device ()) in
  launches ~prefix:"jit-"
    ~place:(fun x -> Nx.place (Nx.Placement.on (Lazy.force device)) x)
    ~sync:(fun () ->
      Nx_device.synchronize (Nx.Device.memory (Lazy.force device)))
    ()

let run_self flag =
  Sys.command (Filename.quote_command Sys.executable_name [ flag ])

let cuda () =
  if run_self "--cuda" <> 0 then []
  else
    [
      Thumper.group ~id:"cuda" "cuda"
        (gpu_launches (fun () -> Nx.Device.v (Cuda 0)));
    ]

let () =
  match Array.to_list Sys.argv with
  | [ _; "--cuda" ] ->
      exit (if Result.is_ok (Nx.Device.get (Cuda 0)) then 0 else 1)
  | _ ->
      Thumper.run "compiled"
        ~budgets:
          [
            Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05;
            Thumper.Budget.no_more_alloc_than 0.01;
          ]
        (topk ~k:4 ~n:32 ~rows:512
        :: Thumper.group ~id:"searchsorted" "searchsorted"
             [
               searchsorted "float64-1e6-into-1e3-host" ~n:1_000_000 ~m:1_000;
               searchsorted "float64-1e6-into-1e6-host" ~n:1_000_000
                 ~m:1_000_000;
             ]
        :: split :: indexed :: rope :: select_zero :: reverse
        :: Thumper.group ~id:"launch" "launch"
             (launches ~place:Fun.id
                ~sync:(fun () -> Nx_device.synchronize Nx_device.host)
                ())
        :: cuda ())
