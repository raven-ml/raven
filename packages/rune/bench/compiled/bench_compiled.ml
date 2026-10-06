(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Compiled host programs.

   A top_k over a short axis, eagerly and compiled for the host. An axis of at
   most 32 entries is ranked by counting, n * n comparisons a row: compiled,
   that is 3 kernels whatever k; eagerly, it costs more than passes over the
   axis would.

   The launches, a user's training step ([Lorenz]) and gpt-oss-20b's decode
   products run on the host and on each GPU the machine has: CUDA, NV and AMD
   GPU 0, and the Mac's Metal GPU. *)

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

(* Masks of 10^7 elements, compiled for the host, as [bool], a byte each, and as
   [bit], eight to a byte: a selection of float32 elements by the mask, which
   unpacks a [bit] mask where it reads it; a [bool] mask kept as bits, one pack;
   and writes into a mask the call consumes, of a window of 1024 elements at an
   odd offset and of 16 scattered elements, which store only the bytes they
   change. *)
let masks =
  let n = 10_000_000 in
  let mask () =
    let st = Random.State.make [| 18 |] in
    Nx.init Nx.bool [| n |] (fun _ -> Random.State.bool st)
  in
  let inputs m =
    let x = uniform [| n |] and y = uniform [| n |] in
    fun g -> g m x y
  in
  let signature () =
    Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
  in
  (* Each call consumes the mask the call before returned. *)
  let consumed id f fresh =
    Thumper.bench_with_setup ~id
      ~setup:(fun () ->
        let g = Rune.jit Nx.Ptree.(consumes tensor @@ returns tensor) f in
        let m = ref (g (fresh ())) in
        Nx_device.synchronize Nx_device.host;
        (g, m))
      id
      (fun (g, m) ->
        m := g !m;
        Nx_device.synchronize Nx_device.host)
  in
  let writes (type b) name (dt : (bool, b) Nx.dtype) =
    let window = Nx.ones dt [| 1024 |] in
    let indices =
      Nx.init Nx.int64 [| 16 |] (fun i -> Int64.of_int ((i.(0) * 611_953) + 3))
    in
    let values = Nx.ones dt [| 16 |] in
    [
      consumed
        (Printf.sprintf "set-%s-1e7" name)
        (fun m -> Nx.set [ R (3, 1027) ] window m)
        (fun () -> Nx.cast dt (mask ()));
      consumed
        (Printf.sprintf "scatter-%s-1e7" name)
        (fun m -> Nx.scatter ~axis:0 ~indices ~values m)
        (fun () -> Nx.cast dt (mask ()));
    ]
  in
  Thumper.group ~id:"jit" "jit"
    ([
       compiled_call "where-bool-1e7" (signature ()) Nx.where (fun () ->
           inputs (mask ()));
       compiled_call "where-bit-1e7" (signature ())
         (fun m x y -> Nx.where (Nx.cast Nx.bool m) x y)
         (fun () -> inputs (Nx.cast Nx.bit (mask ())));
       compiled_call "cast-bit-1e7"
         Nx.Ptree.(tensor @-> returns tensor)
         (Nx.cast Nx.bit)
         (fun () ->
           let m = mask () in
           fun g -> g m);
     ]
    @ writes "bool" Nx.bool @ writes "bit" Nx.bit)

(* Transcendental functions of 1Mi elements on the host, which computes [exp2]
   and [sin] as polynomials: [exp] of [[-80, 80]], [log] of [[1e-30, 1e30]], and
   [sin] of [[-30, 30]] and of [[-1e6, 1e6]], whose larger angles take the long
   reduction. *)
let transcendental =
  let n = 1 lsl 20 in
  let row (type b) (dtype : (float, b) Nx.dtype) name f lo hi =
    let id = Printf.sprintf "%s-%s-1Mi-host" name (Nx_dtype.to_string dtype) in
    compiled_call id
      Nx.Ptree.(tensor @-> returns tensor)
      f
      (fun () ->
        let st = Random.State.make [| 17 |] in
        let x =
          Nx.init dtype [| n |] (fun _ ->
              lo +. Random.State.float st (hi -. lo))
        in
        fun g -> g x)
  in
  let rows dtype =
    [
      row dtype "exp" Nx.exp (-80.) 80.;
      row dtype "log" Nx.log 1e-30 1e30;
      row dtype "sin" Nx.sin (-30.) 30.;
      row dtype "sin-far" Nx.sin (-1e6) 1e6;
    ]
  in
  Thumper.group ~id:"transcendental" "transcendental"
    (rows Nx.float32 @ rows Nx.float64)

(* Symmetric eigendecompositions and singular value decompositions of 64 float32
   matrices of 8 x 8, eagerly and compiled for the host. Compiled, an eigh is 56
   rounds of Jacobi rotations and an svd 42, each round a kernel that computes
   its rotations and one for each rotated value, which a loop runs; svd also
   runs two Householder QRs of 8 steps. *)
let factorizations =
  let batch = 64 and n = 8 in
  let a () =
    let x = uniform [| batch; n; n |] in
    Nx.add x (Nx.matrix_transpose x)
  in
  let id = Printf.sprintf "float32-%dx%dx%d" batch n n in
  let svd a =
    let u, s, vt = Nx.svd a in
    (u, (s, vt))
  in
  [
    Thumper.group ~id:"eigh" "eigh"
      [
        Thumper.bench_with_setup ~setup:a (id ^ "-eager") (fun a ->
            ignore (Sys.opaque_identity (Nx.eigh a));
            Nx_device.synchronize Nx_device.host);
        compiled_call (id ^ "-host")
          Nx.Ptree.(tensor @-> returns (pair tensor tensor))
          (fun a -> Nx.eigh a)
          (fun () ->
            let a = a () in
            fun g -> g a);
      ];
    Thumper.group ~id:"svd" "svd"
      [
        Thumper.bench_with_setup ~setup:a (id ^ "-eager") (fun a ->
            ignore (Sys.opaque_identity (svd a));
            Nx_device.synchronize Nx_device.host);
        compiled_call (id ^ "-host")
          Nx.Ptree.(tensor @-> returns (pair tensor (pair tensor tensor)))
          svd
          (fun () ->
            let a = a () in
            fun g -> g a);
      ];
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
        Nx.copy (Nx.zeros Nx.float32 [| hidden |]) ),
      ( Nx.mul_s (uniform [| hidden; outputs |]) 0.1,
        Nx.copy (Nx.zeros Nx.float32 [| outputs |]) ) )
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

(* Finite checks

   A loss-scaled step keeps each update only if every gradient is finite: one
   value of no axes, reduced from all the gradients, that the update of each of
   128 leaves selects by. Most leaves are vectors of 768 or 3072, every fourth a
   768 x 768 matrix. The value is computed once, and the reductions of the small
   leaves share the threads of one workgroup. *)

let finite_leaves = 128

(* The compiled step and its operands, after one call. *)
let finite_setup ~place ~sync () =
  let shape i =
    if i mod 4 = 3 then [| 768; 768 |]
    else [| (if i mod 2 = 0 then 768 else 3072) |]
  in
  let step grads params =
    let finite =
      List.fold_left
        (fun acc g -> Nx.logical_and acc (Nx.all (Nx.isfinite g)))
        (Nx.all (Nx.isfinite (List.hd grads)))
        (List.tl grads)
    in
    List.map2
      (fun g x -> Nx.where finite (Nx.sub x (Nx.mul_s g 1e-4)) x)
      grads params
  in
  let st = Random.State.make [| 15 |] in
  let leaf i =
    place (Nx.init Nx.float32 (shape i) (fun _ -> Random.State.float st 1.))
  in
  let grads = List.init finite_leaves leaf in
  let params = List.init finite_leaves leaf in
  let f =
    Rune.jit
      Nx.Ptree.(list tensor @-> list tensor @-> returns (list tensor))
      step
  in
  ignore (f grads params);
  sync ();
  (f, grads, params)

let finite_checks ~place ~sync id =
  Thumper.bench_with_setup ~setup:(finite_setup ~place ~sync) id
    (fun (f, grads, params) ->
      ignore (f grads params);
      sync ())

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

(* Loops that stop on a condition, and a loop that draws

   Compiled iterates, which test their condition before each trip: Newton's
   iteration for the square roots of 1,024 values, until every residual is
   small, and the scan's step above iterated until a count reaches 256, against
   the scan's 256 steps. A scan of that step that draws a matrix of noise each
   step compiles as one loop. On the host, Newton's iteration of one value
   mapped over 256 lanes from 0.5 to 10,000, whose lanes stop after 4 to 10
   trips, eagerly and compiled. *)

let roots = 1024

let newton a =
  Rune.iterate' ~max:64
    ~until:(fun x -> Nx.less_s (Nx.max (Nx.abs (Nx.sub (Nx.mul x x) a))) 1e-3)
    ~f:(fun x -> Nx.mul_s (Nx.add x (Nx.div a x)) 0.5)
    (Nx.add_s a 1.)

let counted h =
  fst
    (Rune.iterate
       Nx.Ptree.(pair tensor tensor)
       ~max:scan_steps
       ~until:(fun (_, k) -> Nx.greater_equal_s k (Int32.of_int scan_steps))
       ~f:(fun (h, k) -> (Nx.tanh (Nx.matmul h h), Nx.add_s k 1l))
       (h, Nx.scalar Nx.int32 0l))

(* The scan's step above, adding a draw to its row: each step draws from a scope
   of its own. *)
let drawing (k, (h, xs)) =
  Nx.Rng.with_key k (fun () ->
      fst
        (Rune.scan'
           ~f:(fun h x ->
             let noise = Nx.rand Nx.float32 [| launch_dim; launch_dim |] in
             let h = Nx.tanh (Nx.add (Nx.matmul h h) (Nx.add x noise)) in
             (h, Nx.sum h))
           ~init:h xs))

let loop_setups ~place ~sync =
  let st = Random.State.make [| 16 |] in
  let compiled f shape lo hi () =
    let f = Rune.jit' f
    and x =
      place (Nx.init Nx.float32 shape (fun _ -> lo +. Random.State.float st hi))
    in
    ignore (f x);
    sync ();
    (f, x)
  in
  [
    (Printf.sprintf "newton-%d" roots, compiled newton [| roots |] 0.5 4.);
    ( Printf.sprintf "iterate-%d" scan_steps,
      compiled counted [| launch_dim; launch_dim |] 0. 0.1 );
    ( Printf.sprintf "drawing-scan-%d" scan_steps,
      fun () ->
        let uniform shape =
          place (Nx.init Nx.float32 shape (fun _ -> Random.State.float st 0.1))
        in
        let k = Nx.Rng.key 7
        and xs = uniform [| scan_steps; launch_dim; launch_dim |] in
        let f =
          Rune.jit
            Nx.Ptree.(pair Nx.Rng.ptree (pair tensor tensor) @-> returns tensor)
            drawing
        in
        let f h = f (k, (h, xs)) and h = uniform [| launch_dim; launch_dim |] in
        ignore (f h);
        sync ();
        (f, h) );
  ]

let lanes = 256

(* Newton's iteration for the square root of one value, to a relative
   residual. *)
let lane_newton a =
  Rune.iterate' ~max:64
    ~until:(fun x -> Nx.less (Nx.abs (Nx.sub (Nx.mul x x) a)) (Nx.mul_s a 1e-5))
    ~f:(fun x -> Nx.mul_s (Nx.add x (Nx.div a x)) 0.5)
    (Nx.add_s a 1.)

let lane_loops () =
  let a () =
    Nx.init Nx.float32 [| lanes |] (fun i ->
        0.5 *. (20_000. ** (Float.of_int i.(0) /. Float.of_int (lanes - 1))))
  in
  let mapped = Rune.vmap' lane_newton in
  [
    Thumper.bench_with_setup ~setup:a
      (Printf.sprintf "vmap-newton-%d-lanes-eager" lanes) (fun a ->
        ignore (mapped a));
    Thumper.bench_with_setup
      ~setup:(fun () ->
        let f = Rune.jit' mapped and a = a () in
        ignore (f a);
        (f, a))
      (Printf.sprintf "vmap-newton-%d-lanes-compiled" lanes)
      (fun (f, a) -> ignore (f a));
  ]

let loops ?(prefix = "") ~place ~sync () =
  List.map
    (fun (id, setup) ->
      Thumper.bench_with_setup ~setup (prefix ^ id) (fun (f, x) ->
          ignore (f x);
          sync ()))
    (loop_setups ~place ~sync)

(* Lorenz

   The training step of [Lorenz] compiled without a search: at a batch of 1,024
   draws, 256 hidden units, 256 channels and a horizon of 64 on a GPU, and at a
   batch of 128, 64 hidden units, 32 channels and a horizon of 16 on the host,
   where the full step would take seconds. *)

let lorenz_gpu =
  { Lorenz.batch = 1024; hidden = 256; outputs = 256; horizon = 64 }

let lorenz_host =
  { Lorenz.batch = 128; hidden = 64; outputs = 32; horizon = 16 }

(* The step and its operands after [lorenz_warmup] calls: the first compiles it
   for its initial state, the second for the state a step returns, and the rest
   bring a GPU's clock up to what steady training runs at. *)
let lorenz_warmup = 10

let lorenz_setup placement ~sync size () =
  let f, target, key, s = Lorenz.setup (placement ()) size in
  let s = ref s in
  for _ = 1 to lorenz_warmup do
    s := snd (f target key !s)
  done;
  sync ();
  (f, target, key, s)

let lorenz_step ?(prefix = "") ?(suffix = "") placement ~sync size =
  Thumper.bench_with_setup ~setup:(lorenz_setup placement ~sync size)
    (Printf.sprintf "%sstep-batch-%d%s" prefix size.Lorenz.batch suffix)
    (fun (f, target, key, s) ->
      let _, s' = f target key !s in
      s := s';
      sync ())

(* Decode products

   The products of a decode step of gpt-oss-20b's shapes: one float32 activation
   of 2,880 by bfloat16 weights, to the 4,096 queries, the 512 keys (or values)
   and the 32 experts' scores. kaun's decode suite holds the MXFP4 expert
   products. *)

let decode_products = [ ("q", 4096); ("kv", 512); ("router", 32) ]
let decode_dim = 2880

let decode_setup placement ~sync out () =
  let place x = Nx.place (placement ()) x in
  let f =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> returns tensor)
      (fun x w -> Nx.matmul x (Nx.cast Nx.float32 w))
  in
  let x = place (uniform [| 1; decode_dim |]) in
  let w =
    Nx.place (placement ())
      (Nx.cast Nx.bfloat16 (uniform [| decode_dim; out |]))
  in
  ignore (f x w);
  sync ();
  (f, x, w)

let decode ?(prefix = "") ?(suffix = "") placement ~sync () =
  List.map
    (fun (name, out) ->
      Thumper.bench_with_setup ~setup:(decode_setup placement ~sync out)
        (Printf.sprintf "%s%s-%dx%d-bfloat16%s" prefix name decode_dim out
           suffix) (fun (f, x, w) ->
          ignore (f x w);
          sync ()))
    decode_products

(* GPUs

   A GPU's device opens in the measuring worker: a vendor's driver must not be
   initialized before the fork, so a fresh process ([--cuda], [--nv], [--amd])
   says whether it opens. Each GPU runs the launches, the Lorenz step and the
   decode products. *)

type gpu = {
  placement : unit -> Nx.Placement.t;
  place : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t;
  sync : unit -> unit;
}

let on_gpu open_device =
  let device = lazy (open_device ()) in
  let placement () = Nx.Placement.on (Lazy.force device) in
  {
    placement;
    place = (fun x -> Nx.place (placement ()) x);
    sync =
      (fun () -> Nx_device.synchronize (Nx.Device.memory (Lazy.force device)));
  }

let gpu_cases { placement; sync; _ } =
  lorenz_step ~prefix:"lorenz-" placement ~sync lorenz_gpu
  :: decode ~prefix:"decode-" placement ~sync ()

let run_self flag =
  Sys.command (Filename.quote_command Sys.executable_name [ flag ])

let gpus =
  [
    ("cuda", fun () -> Nx_cuda.device 0);
    ("nv", fun () -> Nx_nv.device 0);
    ("amd", fun () -> Nx_amd.device 0);
  ]

let gpu (name, open_device) =
  if run_self ("--" ^ name) <> 0 then []
  else
    let g = on_gpu open_device in
    [
      Thumper.group ~id:name name
        (launches ~prefix:"jit-" ~place:g.place ~sync:g.sync () @ gpu_cases g);
    ]

(* The finite checks, the loops, the Lorenz step and the decode products on the
   Mac's Metal GPU. A forked worker cannot reach Metal's compiler, so a fresh
   process ([--metal]) compiles them into the disk cache, which the worker's
   setups then read, and says whether Metal opens. *)
let on_metal () = on_gpu (fun () -> Nx_metal.device 0)

let metal_setups { placement; place; sync } =
  ignore (finite_setup ~place ~sync ());
  List.iter (fun (_, setup) -> ignore (setup ())) (loop_setups ~place ~sync);
  ignore (lorenz_setup placement ~sync lorenz_gpu ());
  List.iter
    (fun (_, out) -> ignore (decode_setup placement ~sync out ()))
    decode_products

let metal () =
  if run_self "--metal" <> 0 then []
  else
    let ({ place; sync; _ } as g) = on_metal () in
    [
      Thumper.group ~id:"metal" "metal"
        (finite_checks ~place ~sync
           (Printf.sprintf "finite-checks-%d-leaves" finite_leaves)
         :: loops ~prefix:"jit-" ~place ~sync ()
        @ gpu_cases g);
    ]

let suite () =
  topk ~k:4 ~n:32 ~rows:512
  :: Thumper.group ~id:"searchsorted" "searchsorted"
       [
         searchsorted "float64-1e6-into-1e3-host" ~n:1_000_000 ~m:1_000;
         searchsorted "float64-1e6-into-1e6-host" ~n:1_000_000 ~m:1_000_000;
       ]
  :: split :: indexed :: rope :: select_zero :: masks :: transcendental
  :: factorizations
  @ reverse
    :: Thumper.group ~id:"finite" "finite"
         [
           finite_checks ~place:Fun.id
             ~sync:(fun () -> Nx_device.synchronize Nx_device.host)
             (Printf.sprintf "checks-%d-leaves-host" finite_leaves);
         ]
    :: Thumper.group ~id:"launch" "launch"
         (launches ~place:Fun.id
            ~sync:(fun () -> Nx_device.synchronize Nx_device.host)
            ())
    :: Thumper.group ~id:"loop" "loop"
         (loops ~place:Fun.id
            ~sync:(fun () -> Nx_device.synchronize Nx_device.host)
            ()
         @ lane_loops ())
    :: Thumper.group ~id:"lorenz" "lorenz"
         [
           lorenz_step ~suffix:"-host"
             (fun () -> Nx.Placement.host)
             ~sync:(fun () -> Nx_device.synchronize Nx_device.host)
             lorenz_host;
         ]
    :: Thumper.group ~id:"decode" "decode"
         (decode ~suffix:"-host"
            (fun () -> Nx.Placement.host)
            ~sync:(fun () -> Nx_device.synchronize Nx_device.host)
            ())
    :: (List.concat_map gpu gpus @ metal ())

let config = Thumper.Config.(default |> deadline 120.)

let () =
  match Array.to_list Sys.argv with
  | [ _; "--cuda" ] -> exit (if Result.is_ok (Nx_cuda.get 0) then 0 else 1)
  | [ _; "--nv" ] -> exit (if Result.is_ok (Nx_nv.get 0) then 0 else 1)
  | [ _; "--amd" ] -> exit (if Result.is_ok (Nx_amd.get 0) then 0 else 1)
  | [ _; "--metal" ] -> (
      match Nx_metal.get 0 with
      | Error _ -> exit 1
      | Ok _ ->
          metal_setups (on_metal ());
          exit 0)
  | [ _; "--warm" ] ->
      (* Each case once, in as few calls as a trial takes: what the setups
         compile lands in tolk's disk cache, which a measurement then reads. *)
      ignore
        (Thumper.measure
           ~config:Thumper.Config.(config |> samples 3 |> warmup 0.)
           (suite ()))
  | _ ->
      Thumper.run "compiled" ~config
        ~budgets:
          [
            Thumper.Budget.no_slower_than 0.05;
            Thumper.Budget.no_more_alloc_than 0.01;
          ]
        (suite ())
      |> exit
