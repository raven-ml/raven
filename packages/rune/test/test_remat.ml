(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.remat: its function in every mode, bit for bit, and under reverse mode a
   record of its arguments alone. *)

open Windtrap
open Compose
module T = Nx.Ptree
module D = Nx.Dtype

let two () = T.(tensor @-> tensor @-> returns tensor)
let p () = T.(pair tensor tensor)
let objective f (x, y) = Nx.sum (f x y)

type case =
  | Case :
      (float, 's) Nx.dtype * int array * operand * operand * operand * expr
      -> case

let cases =
  let open Gen in
  (let* s = shape in
   let* e = expr ~rank:(Array.length s) in
   let* x = operand s and+ y = operand s and+ t = operand s in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any Nx.float64 ] in
   match D.kind dt with Float -> Case (dt, s, x, y, t, e) | _ -> assert false)
  |> with_pp (fun ppf (Case (dt, s, x, y, _, e)) ->
      Format.fprintf ppf "%a %a, x %a, y %a: %a" Nx.pp_dtype dt pp_shape s
        pp_view x.view pp_view y.view pp e)

(* Outside a jit, [remat s f] and [f] give the same bits and the same failure,
   eagerly and under each transformation and their compositions. *)
let law (Case (dt, s, xo, yo, to_, e)) =
  let x = make dt s xo and y = make dt s yo and t = make dt s to_ in
  cover "a view operand" (xo.view <> Plain || yo.view <> Plain);
  cover "a zero-size shape" (numel s = 0);
  let f x y = eval x y e in
  let r = Rune.remat (two ()) f in
  let tt = outcome_t bits_t and t2 = outcome_t (Testable.pair bits_t bits_t) in
  let same msg g =
    equal ~msg tt
      (outcome (fun () -> bits (g f)))
      (outcome (fun () -> bits (g r)))
  in
  let same2 msg g =
    let both (a, b) = (bits a, bits b) in
    equal ~msg t2
      (outcome (fun () -> both (g f)))
      (outcome (fun () -> both (g r)))
  in
  same "eager" (fun f -> f x y);
  same2 "jvp" (fun f -> Rune.jvp' (fun x -> f x y) x t);
  same2 "grad" (fun f -> Rune.grad (p ()) (objective f) (x, y));
  same2 "grad of grad" (fun f ->
      Rune.grad (p ())
        (fun (x, y) ->
          let gx, gy = Rune.grad (p ()) (objective f) (x, y) in
          Nx.add (Nx.sum (Nx.mul gx gx)) (Nx.sum gy))
        (x, y));
  same "jvp of grad" (fun f ->
      snd (Rune.jvp' (fun x -> fst (Rune.grad (p ()) (objective f) (x, y))) x t));
  let xs = Nx.stack [ x; Nx.neg x ] in
  same "vmap" (fun f -> Rune.vmap' (fun x -> f x y) xs);
  same "grad of vmap" (fun f ->
      Rune.grad' (fun xs -> Nx.sum (Rune.vmap' (fun x -> f x y) xs)) xs);
  same "vmap of grad" (fun f ->
      Rune.vmap' (fun x -> Rune.grad' (fun x -> Nx.sum (f x y)) x) xs)

(* Compiled, a remat'd function gives its function's bits, called and
   differentiated, where every function it applies is computed exactly and its
   derivatives sum no terms. A program is such a function over two operands of
   one dtype and shape, each in its view; its four compiled forms are made once
   and compile at the first case that draws it, and the cases draw a program and
   its operands' elements. Each program runs first as an example, so every run
   compiles each and reaches its covers. *)

type 's ft = (float, 's, Nx.host) Nx.t

type program =
  | Program : {
      dt : (float, 's) Nx.dtype;
      s : int array;
      xv : view;
      yv : view;
      e : expr;
      call : (bool -> 's ft -> 's ft -> 's ft) Lazy.t;
      grad : (bool -> 's ft -> 's ft -> 's ft * 's ft) Lazy.t;
    }
      -> program

let program (type s) (dt : (float, s) Nx.dtype) s xv yv e =
  let e = elementwise (exact e) in
  let f (x : s ft) y = eval x y e in
  let r = Rune.remat (two ()) f in
  (* [remat] picks the remat'd function's compiled form over [f]'s. *)
  let both mk =
    lazy
      (let plain = mk f and remat = mk r in
       fun remat' -> if remat' then remat else plain)
  in
  let call = both (fun f -> Rune.jit (two ()) f) in
  let grad =
    both (fun f ->
        Rune.jit
          T.(tensor @-> tensor @-> returns (pair tensor tensor))
          (fun x y -> Rune.grad (p ()) (objective f) (x, y)))
  in
  Program { dt; s; xv; yv; e; call; grad }

let programs =
  let f32 = Nx.float32 and f64 = Nx.float64 in
  [|
    program f32 [||] Plain Plain (Bin (Mul, X, Un (Sin, Y)));
    program f64 [| 0; 3 |] Plain Flipped (Bin (Add, Un (Square, X), Y));
    program f32 [| 2; 3 |] Strided Broadcast
      (Bin (Maximum, Un (Neg, X), Bin (Mul, Y, Y)));
    program f64 [| 3 |] Flipped Plain (Bin (Where_less, X, Bin (Div, Y, K 2.)));
    program f32 [| 2; 2; 2 |] Broadcast Strided
      (Un (Flip 2, Bin (Sub, X, Un (Exp, Y))));
    program f64 [| 4; 1 |] Plain Plain
      (Bin
         ( Minimum,
           Bin (Mul, Un (Abs, X), Un (Sqrt, Un (Abs, Y))),
           Un (Cumsum 0, Bin (Add, X, Y)) ));
  |]

(* The elements an operand's storage holds in [view] at shape [s]. *)
let storage view s =
  match view with
  | Plain | Flipped -> numel s
  | Strided -> numel s * 2
  | Broadcast -> if Array.length s = 0 then 1 else numel s / Int.max 1 s.(0)

(* Elements as {!Compose.operand} draws them: 3 in 5 from its pool, 2 in 5 in
   [[-4, 4]]. *)
let elements n =
  let open Gen in
  let element =
    frequency
      [
        (3, map (fun i -> pool.(i)) (int_range 0 (Array.length pool - 1)));
        (2, float_range (-4.) 4.);
      ]
  in
  array ~size:(constant n) element

let pp_compiled ppf (i, xs, ys) =
  let (Program p) = programs.(i) in
  let values ppf a =
    Format.pp_print_array
      ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
      (fun ppf v -> Format.fprintf ppf "%h" v)
      ppf a
  in
  Format.fprintf ppf "%a %a, x %a [%a], y %a [%a]: %a" Nx.pp_dtype p.dt pp_shape
    p.s pp_view p.xv values xs pp_view p.yv values ys pp p.e

let compiled_cases =
  let open Gen in
  (let* i = int_range 0 (Array.length programs - 1) in
   let (Program p) = programs.(i) in
   let+ xs = elements (storage p.xv p.s)
   and+ ys = elements (storage p.yv p.s) in
   (i, xs, ys))
  |> with_pp pp_compiled

(* Each program at elements of the pool in turn. *)
let compiled_examples =
  let from k n = Array.init n (fun j -> pool.((k + j) mod Array.length pool)) in
  List.init (Array.length programs) (fun i ->
      let (Program p) = programs.(i) in
      (i, from i (storage p.xv p.s), from (i + 7) (storage p.yv p.s)))

let law_compiled (i, xs, ys) =
  let (Program p) = programs.(i) in
  cover "a view operand" (p.xv <> Plain || p.yv <> Plain);
  cover "a zero-size shape" (numel p.s = 0);
  let x = make p.dt p.s { view = p.xv; values = xs }
  and y = make p.dt p.s { view = p.yv; values = ys } in
  let tt = outcome_t bits_t and t2 = outcome_t (Testable.pair bits_t bits_t) in
  let call remat = outcome (fun () -> bits (Lazy.force p.call remat x y)) in
  equal ~msg:"jit" tt (call false) (call true);
  let grad remat =
    outcome (fun () ->
        let gx, gy = Lazy.force p.grad remat x y in
        (bits gx, bits gy))
  in
  equal ~msg:"grad under jit" t2 (grad false) (grad true)

(* Memory *)

let device = Nx_support.memory 1

module Mem = (val Nx.devices [ device ])

let settled () =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create device 0);
    Rig.free_cache device
  done;
  Rig.allocated device

(* The bytes [keep] holds after a run beyond those held before it, after one
   uncounted run; what it kept is dropped before the next measurement. *)
let held_after kept keep =
  keep ();
  let before = settled () in
  keep ();
  let held = settled () - before in
  ignore (Sys.opaque_identity !kept);
  kept := [];
  held

let f32 = Nx.float32
let three x = Nx.mul (Nx.sin x) (Nx.exp x)

(* Three maps keep their two intermediates; under a remat the record keeps the
   argument alone, which the caller holds, so a pullback holds the result's
   bytes. *)
let test_residuals () =
  let n = 4096 in
  let x = Nx.place Mem.on (Nx.ones f32 [| n |]) in
  let kept = ref [] in
  let keep f () = kept := Rune.vjp' f x :: !kept in
  equal ~msg:"three maps" int (3 * 4 * n) (held_after kept (keep three));
  equal ~msg:"under a remat" int (4 * n)
    (held_after kept (keep (Rune.remat T.(tensor @-> returns tensor) three)));
  (* The pullback computes the same gradient. *)
  let g f = snd (Rune.vjp' f x) (Nx.place Mem.on (Nx.ones f32 [| n |])) in
  equal bits_t
    (bits (g three))
    (bits (g (Rune.remat T.(tensor @-> returns tensor) three)));
  (* What the function computes from a value it captures alone is computed again
     too: the record keeps [x] and [w], which the caller holds, where without a
     remat it keeps the sine, which the exponential's tangent reads. *)
  let w = Nx.place Mem.on (Nx.full f32 [| n |] 0.5) in
  let kept' = ref [] in
  let captured remat () =
    let g w' =
      let f x = Nx.mul x (Nx.exp (Nx.sin w')) in
      let f = if remat then Rune.remat T.(tensor @-> returns tensor) f else f in
      f x
    in
    kept' := Rune.vjp' g w :: !kept'
  in
  equal ~msg:"a capture's intermediates" int
    (2 * 4 * n)
    (held_after kept' (captured false));
  equal ~msg:"under a remat" int (4 * n) (held_after kept' (captured true));
  ignore (Sys.opaque_identity w);
  (* And what it computes from constants alone, under every rule that derives
     the remat again. *)
  let constants remat () =
    let f x =
      Nx.mul x (Nx.exp (Nx.sin (Nx.place Mem.on (Nx.full f32 [| n |] 0.5))))
    in
    let f = if remat then Rune.remat T.(tensor @-> returns tensor) f else f in
    kept' := Rune.vjp' f x :: !kept'
  in
  equal ~msg:"a constant's intermediates" int
    (2 * 4 * n)
    (held_after kept' (constants false));
  equal ~msg:"under a remat, from constants" int (4 * n)
    (held_after kept' (constants true));
  ignore (Sys.opaque_identity x)

(* A step that checkpoints its body: reverse mode stacks each step's carry, the
   remat's argument, where three maps per step would stack two more arrays. *)
let test_scan_residuals () =
  let n = 1024 and rows = 8 in
  let x = Nx.place Mem.on (Nx.ones f32 [| n |]) in
  let xs = Nx.place Mem.on (Nx.zeros f32 [| rows; n |]) in
  let kept = ref [] in
  let scan step x =
    fst
      (Rune.scan T.tensor T.tensor T.unit
         (fun c r -> (Nx.add (step c) r, ()))
         ~init:x xs)
  in
  let keep step () = kept := Rune.vjp' (scan step) x :: !kept in
  let plain = held_after kept (keep three) in
  let checkpointed =
    held_after kept (keep (Rune.remat T.(tensor @-> returns tensor) three))
  in
  (* The result, then per row the carry, and without the remat its sine and
     exponential. *)
  equal ~msg:"plain" int (4 * n * (1 + (3 * rows))) plain;
  equal ~msg:"checkpointed" int (4 * n * (1 + rows)) checkpointed;
  (* Held to the end, so that no measurement sees them collected. *)
  ignore (Sys.opaque_identity (x, xs))

(* Compiled gradients through residual blocks [a + relu (a w1) w2], over
   stacks of weights that are transposed views, which a program reads in place.
   Under remats the backward pass computes every value a block computes again,
   so a compiled gradient keeps of each block its argument alone, an argument
   of the program being one its caller holds. The bytes a compiled function
   holds are its program's storage and its result: under remats no more than
   the blocks' arguments, two blocks' pre-activations and four rows of
   cotangents, whatever a block computes from its weights. Each call runs
   compiled, as an evaluated one would hold nothing, and gives eager's gradient
   but for the order of its sums. *)

let layers = 8
and dim = 32
and batch = 256

let hidden = 8 * dim
let row = 4 * batch * dim
let bound blocks = ((blocks + 4) * row) + (2 * 4 * batch * hidden)

let stack r c =
  Nx.matrix_transpose
    (Nx.mul_s (Nx.Rng.normal ~key:(Nx.Rng.key r) f32 [| layers; c; r |]) 0.05)

let ws = (stack dim hidden, stack hidden dim)

let block ?(prep = Fun.id) (w1, w2) a =
  let h = Nx.matmul a (prep w1) in
  Nx.add a (Nx.matmul (Nx.maximum h (Nx.zeros_like h)) (prep w2))

let rm f = Rune.remat T.(tensor @-> returns tensor) f
let layer (w1, w2) i = (Nx.get [ i ] w1, Nx.get [ i ] w2)

(* The blocks in a chain from [from a], each under a remat where [remat]. *)
let chain ?prep ~from remat ws a =
  Nx.sum
    (List.fold_left
       (fun a i ->
         let b = block ?prep (layer ws i) in
         if remat then rm b a else b a)
       (from a) (List.init layers Fun.id))

let host_settled () =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create Rig.host 0);
    Rig.free_cache Rig.host
  done;
  Rig.allocated Rig.host

let a = Nx.ones f32 [| batch; dim |]

(* The bytes the compiled gradient of [loss remat ws] at [a] holds after a
   warm call, its gradient checked against eager's without remats. *)
let held ?(ws = ws) loss remat =
  let expected = Nx.to_array (Nx.flatten (Rune.grad' (loss false ws) a)) in
  let measure () =
    let g =
      Rune.jit
        T.(pair tensor tensor @-> tensor @-> returns tensor)
        (fun ws a -> Rune.grad' (loss remat ws) a)
    in
    let before = host_settled () and calls = Rune.evaluations () in
    let r = g ws a in
    let held = host_settled () - before in
    Rune_support.Compiled.here ();
    equal ~msg:"evaluated calls" int 0 (Rune.evaluations () - calls);
    equal ~msg:"the gradient"
      (array (float_rel ~rel:1e-5 ~abs:1e-6))
      expected
      (Nx.to_array (Nx.flatten r));
    ignore (Sys.opaque_identity (g, r));
    held
  in
  ignore (measure ());
  measure ()

let at_most ~msg loss blocks =
  let plain = held loss false and recomputed = held loss true in
  less ~msg:(msg ^ ", without remats") int ~than:plain (bound blocks);
  less
    ~msg:
      (Printf.sprintf "%s: %d bytes under remats, %d without" msg recomputed
         plain)
    int ~than:(bound blocks) recomputed

(* From the sine of the argument: each block's argument is a value the program
   computes. *)
let test_compiled () = at_most ~msg:"a chain" (chain ~from:Nx.sin) layers

(* Its backward pass computes one block again at a time, so it holds one
   block's pre-activation beside the blocks' arguments and four rows. *)
let test_one_block () =
  let one = ((layers + 4) * row) + (4 * batch * hidden) in
  let recomputed = held (chain ~from:Nx.sin) true in
  less
    ~msg:(Printf.sprintf "%d bytes under remats" recomputed)
    int ~than:one recomputed

(* Independent chains of remat'd blocks, summed: the backward pass finishes
   one chain before it starts the next, so [k] chains hold what one does and
   a row for each other chain's cotangent. *)
let test_chains () =
  let blocks = 4 in
  let chains k remat ws a =
    List.fold_left
      (fun s c ->
        let from a = Nx.sin (Nx.add_s a (Float.of_int c)) in
        let a =
          List.fold_left
            (fun a i ->
              let b = block (layer ws ((c * blocks) + i)) in
              if remat then rm b a else b a)
            (from a)
            (List.init blocks Fun.id)
        in
        Nx.add s (Nx.sum a))
      (Nx.scalar f32 0.) (List.init k Fun.id)
  in
  let one = held (chains 1) true and two = held (chains 2) true in
  less
    ~msg:(Printf.sprintf "%d bytes for two chains, %d for one" two one)
    int ~than:(one + row + 1) two

(* The first block's argument is the program's: it costs nothing to keep, so
   the chain holds no more than the one from its sine. *)
let test_argument () =
  let from_argument = held (chain ~from:Fun.id) true in
  let from_sine = held (chain ~from:Nx.sin) true in
  less
    ~msg:
      (Printf.sprintf "%d bytes from the argument, %d from its sine"
         from_argument from_sine)
    int ~than:(from_sine + 1) from_argument

(* Blocks that each read the argument keep nothing they compute. *)
let test_over_argument () =
  let summed remat ws a =
    List.fold_left
      (fun s i ->
        let b = block (layer ws i) in
        Nx.add s (Nx.sum (if remat then rm b a else b a)))
      (Nx.scalar f32 0.) (List.init layers Fun.id)
  in
  at_most ~msg:"blocks over the argument" summed layers

(* What a block computes from the weights it captures is computed again. *)
let test_transformed () =
  at_most ~msg:"weights transformed in each block"
    (chain ~prep:Nx.tanh ~from:Nx.sin)
    layers

(* A scan of the blocks over the stacks, its step under a remat, keeps what
   the chain written out keeps: each step's carry. *)
let test_scanned () =
  let scanned remat (w1s, w2s) a =
    let step c w =
      let b = block w in
      ((if remat then rm b c else b c), ())
    in
    Nx.sum
      (fst
         (Rune.scan T.tensor T.(pair tensor tensor) T.unit step ~init:(Nx.sin a)
            (w1s, w2s)))
  in
  at_most ~msg:"a scan over the stacks" scanned layers;
  (* Each row of a transposed stack is read where it lies, as a contiguous
     stack's is. *)
  let contiguous = (Nx.copy (fst ws), Nx.copy (snd ws)) in
  let strided = held scanned true in
  let copied = held ~ws:contiguous scanned true in
  less
    ~msg:(Printf.sprintf "%d bytes over views, %d over copies" strided copied)
    int ~than:(copied + 1) strided

(* A scan over [k] chunks of the batch whose step is the gradient of the
   remat'd chain at its chunk holds the stacked result and one step's
   storage, which the step's values share within a trip: a chunk's part of
   the whole batch's bound. *)
let test_step_gradient () =
  let loss = chain ~from:Nx.sin true in
  let expected = Nx.to_array (Nx.flatten (Rune.grad' (loss ws) a)) in
  List.iter
    (fun k ->
      let measure () =
        let g =
          Rune.jit
            T.(pair tensor tensor @-> tensor @-> returns tensor)
            (fun ws a ->
              let chunks = Nx.reshape [| k; batch / k; dim |] a in
              let (), gs =
                Rune.scan T.unit T.tensor T.tensor
                  (fun () x -> ((), Rune.grad' (loss ws) x))
                  ~init:() chunks
              in
              Nx.reshape [| batch; dim |] gs)
        in
        let before = host_settled () and calls = Rune.evaluations () in
        let r = g ws a in
        let held = host_settled () - before in
        Rune_support.Compiled.here ();
        equal ~msg:"evaluated calls" int 0 (Rune.evaluations () - calls);
        equal ~msg:"the gradient"
          (array (float_rel ~rel:1e-5 ~abs:1e-6))
          expected
          (Nx.to_array (Nx.flatten r));
        ignore (Sys.opaque_identity (g, r));
        held
      in
      ignore (measure ());
      let held = measure () in
      less
        ~msg:(Printf.sprintf "%d chunks: %d bytes" k held)
        int
        ~than:(row + (bound layers / k))
        held)
    [ 1; 2; 4 ]

(* A scan over [k] chunks of the batch whose step sums, in its carry, the
   gradient of the chain at its chunk in each block's weights, separate
   arrays: a step of gradient accumulation. Under remats each trip computes
   its blocks again, its storage held within the trip, so the remats save
   memory at every number of chunks, and a scan of more, smaller chunks holds
   no more than one of the whole batch. On the host what the first call
   holds is its footprint. *)
let test_summed_step () =
  let pd = T.(list (pair tensor tensor)) in
  let s = T.(pair pd tensor @-> returns (pair tensor pd)) in
  let apart =
    List.init layers (fun i ->
        let w1, w2 = layer ws i in
        (Nx.copy w1, Nx.copy w2))
  in
  let loss remat ws x =
    Nx.sum
      (List.fold_left
         (fun a w ->
           let b = block w in
           if remat then rm b a else b a)
         (Nx.sin x) ws)
  in
  let f remat k (ws, a) =
    let chunks = Nx.reshape [| k; batch / k; dim |] a in
    let step (l, sums) x =
      let v, gs = Rune.value_and_grad pd (fun ws -> loss remat ws x) ws in
      let add (s1, s2) (g1, g2) = (Nx.add s1 g1, Nx.add s2 g2) in
      ((Nx.add l v, List.map2 add sums gs), ())
    in
    let zeros (w1, w2) = (Nx.zeros_like w1, Nx.zeros_like w2) in
    fst
      (Rune.scan T.(pair tensor pd) T.tensor T.unit step
         ~init:(Nx.scalar f32 0., List.map zeros ws)
         chunks)
  in
  let flat (l, sums) =
    let row x = Nx.to_array (Nx.flatten x) in
    Array.concat
      (row l :: List.concat_map (fun (s1, s2) -> [ row s1; row s2 ]) sums)
  in
  let expected = flat (f false 1 (apart, a)) in
  let on_host (r, _) = Rig.equal r Rig.host in
  let measure remat k =
    ignore (Rune.jit s (f remat k) (apart, a));
    let g = Rune.jit s (f remat k) in
    let before = host_settled () and calls = Rune.evaluations () in
    let r = g (apart, a) in
    let held = host_settled () - before in
    Rune_support.Compiled.here ();
    equal ~msg:"evaluated calls" int 0 (Rune.evaluations () - calls);
    equal ~msg:"the loss and the sums"
      (array (float_rel ~rel:1e-4 ~abs:1e-4))
      expected (flat r);
    ignore (Sys.opaque_identity (g, r));
    Rune_support.Compiled.here ();
    equal
      ~msg:(Printf.sprintf "%d chunks: the footprint" k)
      int held
      (snd (List.find on_host (Rune.footprint s (f remat k) (apart, a))));
    held
  in
  let whole = measure true 1 in
  List.iter
    (fun k ->
      let plain = measure false k and recomputed = measure true k in
      less
        ~msg:
          (Printf.sprintf "%d chunks: %d bytes under remats, %d without" k
             recomputed plain)
        int ~than:plain recomputed;
      less
        ~msg:
          (Printf.sprintf "%d chunks: %d bytes under remats, %d for one" k
             recomputed whole)
        int ~than:(whole + 1) recomputed)
    [ 1; 2; 4 ]

(* A scan over chunks of 2048 rows whose carry sums the loss and the weights'
   gradients of four remat'd residual layers, 256 to 1024 to 256: each trip
   holds its own storage, which a launch in the trip's body makes no earlier
   than the trip's peak allows, so at 2 and 4 trips the scan holds what it holds
   at one trip and its trip's index, an int a run keeps twice
   ({!Rig_program.footprint}), and less than without remats, as tolk's
   DIVERGENCES D404 states. Both sides of each bound count the same allocations
   ({!Rig.allocation}). *)
let test_deep_scan () =
  let layers = 4 and dim = 256 and hidden = 1024 and n = 2048 in
  let w k r c =
    Nx.mul_s (Nx.Rng.normal ~key:(Nx.Rng.key k) f32 [| r; c |]) 0.02
  in
  let ws =
    List.init layers (fun i ->
        (w (2 * i) dim hidden, w ((2 * i) + 1) hidden dim))
  in
  let pd = T.(list (pair tensor tensor)) in
  let loss remat ws x =
    let layer a w = if remat then rm (block w) a else block w a in
    Nx.sum (List.fold_left layer x ws)
  in
  let f remat (ws, xs) =
    let step (l, sums) x =
      let v, gs = Rune.value_and_grad pd (fun ws -> loss remat ws x) ws in
      let add (s1, s2) (g1, g2) = (Nx.add s1 g1, Nx.add s2 g2) in
      ((Nx.add l v, List.map2 add sums gs), ())
    in
    let zeros (w1, w2) = (Nx.zeros_like w1, Nx.zeros_like w2) in
    fst
      (Rune.scan T.(pair tensor pd) T.tensor T.unit step
         ~init:(Nx.scalar f32 0., List.map zeros ws)
         xs)
  in
  let on_host (r, _) = Rig.equal r Rig.host in
  Rune_support.Compiled.here ();
  let bytes ?(remat = true) g =
    let xs = Nx.Rng.normal ~key:(Nx.Rng.key 99) f32 [| g; n; dim |] in
    snd
      (List.find on_host
         (Rune.footprint
            T.(pair pd tensor @-> returns (pair tensor pd))
            (f remat) (ws, xs)))
  in
  let one = bytes 1 and two = bytes 2 in
  let index = 2 * Rig.allocation Rig.host 8 in
  less
    ~msg:(Printf.sprintf "2 trips: %d bytes, 1 trip %d" two one)
    int
    ~than:(one + index + 1)
    two;
  equal ~msg:"4 trips, as 2" int two (bytes 4);
  less ~msg:"without remats" int ~than:(bytes ~remat:false 2) two

(* The stacks read by rows against separate arrays: each row's gradient is
   its region of the stack's, so the gradient alone holds no more over the
   stacks, with or without remats. Under remats an array apart is the storage
   the remat's call wrote, and a stack's row the region of it the call
   writes instead, so it holds no more under remats either. A step that sums
   it in its carry over [k] chunks adds each row's gradient into its region
   of the carry, in the carry's storage, so it holds no more either, with or
   without remats: under remats each row's add runs once its row is
   computed, the carry being storage its trip is given. *)
let test_stacked_step () =
  let pd = T.(list (pair tensor tensor)) and pt = T.(pair tensor tensor) in
  let apart =
    List.init layers (fun i ->
        let w1, w2 = layer ws i in
        (Nx.copy w1, Nx.copy w2))
  in
  let loss remat ws x =
    Nx.sum
      (List.fold_left
         (fun a w ->
           let b = block w in
           if remat then rm b a else b a)
         (Nx.sin x) ws)
  in
  let rows ws = List.init layers (layer ws) in
  let on_host (r, _) = Rig.equal r Rig.host in
  Rune_support.Compiled.here ();
  let bytes s f x = snd (List.find on_host (Rune.footprint s f x)) in
  (* The results hold the same bytes, in two stacks or in a pair of arrays a
     row, but a stack's allocation may count more than its bytes
     ({!Rig.allocation}): on Windows, the slack of its start on a page. *)
  let slack =
    List.fold_left
      (fun acc w -> acc + Rig.allocation Rig.host (Nx.nbytes w) - Nx.nbytes w)
      0
      [ fst ws; snd ws ]
  in
  let no_more msg ~apart:x ~stacked:y =
    less
      ~msg:(Printf.sprintf "%s: %d bytes stacked, %d apart" msg y x)
      int
      ~than:(x + slack + 1)
      y
  in
  List.iter
    (fun remat ->
      let x =
        bytes
          T.(pair pd tensor @-> returns pd)
          (fun (ws, a) -> Rune.grad pd (fun ws -> loss remat ws a) ws)
          (apart, a)
      and y =
        bytes
          T.(pair pt tensor @-> returns pt)
          (fun (ws, a) -> Rune.grad pt (fun ws -> loss remat (rows ws) a) ws)
          (ws, a)
      in
      no_more (Printf.sprintf "the gradient, remats %b" remat) ~apart:x
        ~stacked:y)
    [ false; true ];
  let chunks k a = Nx.reshape [| k; batch / k; dim |] a in
  let f_apart remat k (ws, a) =
    let step (l, sums) x =
      let v, gs = Rune.value_and_grad pd (fun ws -> loss remat ws x) ws in
      let add (s1, s2) (g1, g2) = (Nx.add s1 g1, Nx.add s2 g2) in
      ((Nx.add l v, List.map2 add sums gs), ())
    in
    let zeros (w1, w2) = (Nx.zeros_like w1, Nx.zeros_like w2) in
    fst
      (Rune.scan T.(pair tensor pd) T.tensor T.unit step
         ~init:(Nx.scalar f32 0., List.map zeros ws)
         (chunks k a))
  in
  let f_stacked remat k ((w1, w2), a) =
    let step (l, (s1, s2)) x =
      let v, (g1, g2) =
        Rune.value_and_grad pt (fun ws -> loss remat (rows ws) x) (w1, w2)
      in
      ((Nx.add l v, (Nx.add s1 g1, Nx.add s2 g2)), ())
    in
    fst
      (Rune.scan T.(pair tensor pt) T.tensor T.unit step
         ~init:(Nx.scalar f32 0., (Nx.zeros_like w1, Nx.zeros_like w2))
         (chunks k a))
  in
  List.iter
    (fun (remat, k) ->
      let x =
        bytes
          T.(pair pd tensor @-> returns (pair tensor pd))
          (f_apart remat k) (apart, a)
      and y =
        bytes
          T.(pair pt tensor @-> returns (pair tensor pt))
          (f_stacked remat k) (ws, a)
      in
      no_more (Printf.sprintf "%d chunks, remats %b" k remat) ~apart:x
        ~stacked:y)
    (List.map (fun k -> (false, k)) [ 2; 4; 8 ]
    @ List.map (fun k -> (true, k)) [ 1; 2; 4; 8 ])

(* Assemblies of the gradients of remat'd functions, each the storage of a
   remat's backward call, with eager's bits: a stack of rows of eight, which
   the calls write in place; a concatenation of three and four, whose second
   starts 12 bytes in, and one along a later axis, whose pieces are no runs,
   both copied; a gradient stacked twice and two gradients each in two
   stacks, each piece a reshape a stack reads once of a value read twice,
   all copied; a scan whose trip stacks two, written in the trip's place;
   and a stack that takes a donated argument's memory, written there. *)
let test_placed_pieces () =
  let f = rm (fun x -> Nx.mul (Nx.sin x) x) in
  let dg x = Rune.grad' (fun x -> Nx.sum (f x)) x in
  let check msg g x =
    equal ~msg bits_t (bits (g x)) (bits (Rune.jit' g x))
  in
  let x n = Nx.Rng.normal ~key:(Nx.Rng.key 9) f32 [| 2; n |] in
  let row i x = Nx.get [ i ] x in
  check "rows of eight"
    (fun x -> Nx.stack [ dg (row 0 x); dg (row 1 x) ])
    (x 8);
  let first n x = Nx.slice [ Nx.R (0, n) ] x in
  check "three and four"
    (fun x ->
      Nx.concatenate ~axis:0 [ dg (first 3 (row 0 x)); dg (first 4 (row 1 x)) ])
    (x 8);
  check "along a later axis"
    (fun x -> Nx.concatenate ~axis:1 [ dg x; dg (Nx.neg x) ])
    (x 4);
  check "one gradient twice"
    (fun x ->
      let y = dg (row 0 x) in
      Nx.stack [ y; y ])
    (x 8);
  let pair = T.(tensor @-> returns (pair tensor tensor)) in
  let crossed x =
    let y = dg (row 0 x) and z = dg (row 1 x) in
    (Nx.stack [ y; z ], Nx.stack [ z; y ])
  in
  let a, b = crossed (x 8) and a', b' = Rune.jit pair crossed (x 8) in
  equal ~msg:"two gradients in two stacks" bits_t (bits a) (bits a');
  equal ~msg:"two gradients in two stacks, the second" bits_t (bits b)
    (bits b');
  check "a trip's stack"
    (fun x ->
      let xs =
        Nx.reshape [| 2; 2; 8 |] (Nx.concatenate ~axis:0 [ x; Nx.neg x ])
      in
      snd
        (Rune.scan T.unit T.tensor T.tensor
           (fun () x -> ((), Nx.stack [ dg (row 0 x); dg (row 1 x) ]))
           ~init:() xs))
    (x 8);
  let stacked x = Nx.stack [ dg (row 0 x); dg (row 1 x) ] in
  let two = T.(tensor @-> tensor @-> returns tensor) in
  let g = Rune.jit two (fun x _ -> stacked x) in
  let expected = bits (stacked (x 8)) in
  let w = Nx.copy (Nx.zeros f32 [| 2; 8 |]) in
  equal ~msg:"into a donated argument" bits_t expected
    (bits (g (x 8) (Nx.donate w)))

(* Donation *)

(* A result that takes a donated argument's memory is stored over it after the
   remat's backward pass reads it: the gradient through the remat plus [b],
   donated, takes [b]'s memory, and plus [a], donated, which the backward pass
   reads at every element, gives the same values either way. *)
let test_donated () =
  let w = (Nx.get [ 0 ] (fst ws), Nx.get [ 0 ] (snd ws)) in
  let grad a = Rune.grad' (fun a -> Nx.sum (rm (block w) a)) a in
  let address x = Nx.Repr.read ~by:"test" x Nx_array.buffer in
  let memory_t =
    Testable.make
      ~pp:(fun ppf b -> Format.fprintf ppf "byte %d" (Rig.Buffer.offset b))
      ~equal:Rig.Buffer.overlaps
  in
  let x () = Nx.Rng.normal ~key:(Nx.Rng.key 5) f32 [| batch; dim |] in
  let b () = Nx.full f32 [| batch; dim |] 0.5 in
  let expected = Nx.to_array (Nx.add (grad (x ())) (b ())) in
  let plus_b =
    Rune.jit
      T.(tensor @-> tensor @-> returns tensor)
      (fun a b -> Nx.add (grad a) b)
  in
  let plus_a =
    Rune.jit T.(tensor @-> returns tensor) (fun a -> Nx.add (grad a) a)
  in
  for call = 0 to 1 do
    let msg = Printf.sprintf "call %d" call in
    let donated = b () in
    let before = address donated in
    let calls = Rune.evaluations () in
    let r = plus_b (x ()) (Nx.donate donated) in
    Rune_support.Compiled.here ();
    equal ~msg:(msg ^ ": compiled") int 0 (Rune.evaluations () - calls);
    equal ~msg:(msg ^ ": in b's memory") memory_t before (address r);
    equal ~msg:(msg ^ ": plus b")
      (array (float_rel ~rel:1e-5 ~abs:1e-6))
      expected (Nx.to_array r);
    let r = plus_a (Nx.donate (x ())) in
    equal ~msg:(msg ^ ": plus a")
      (array (float_rel ~rel:1e-5 ~abs:1e-6))
      (Nx.to_array (Nx.add (grad (x ())) (x ())))
      (Nx.to_array r)
  done

(* Traced at the call *)

module Total = Rune.Total

(* Reverse mode traces [f] once at the call, though its backward pass computes
   [f] again. *)
let test_traced () =
  let calls = ref 0 in
  let r =
    Rune.remat
      T.(tensor @-> returns tensor)
      (fun x ->
        incr calls;
        three x)
  in
  let x = Nx.create f32 [| 2 |] [| 0.5; -1. |] in
  ignore (Rune.grad' (fun x -> Nx.sum (r x)) x);
  equal int 1 !calls

(* An addition counts once, however often the backward pass computes the
   function again: the sum of [x = [0.5; -1.]] once is [-0.5]. *)
let test_total () =
  let t = Total.make () in
  let r =
    Rune.remat
      T.(tensor @-> returns tensor)
      (fun x ->
        Total.add t (Nx.sum x);
        three x)
  in
  let x = Nx.create f32 [| 2 |] [| 0.5; -1. |] in
  let (), total =
    Total.collect t ~init:(Nx.scalar f32 0.) (fun () ->
        ignore (Rune.grad' (fun x -> Nx.sum (r x)) x))
  in
  equal bits_t (bits (Nx.scalar f32 (-0.5))) (bits total)

(* A remat captures an array the differentiation tracks: its gradient is the
   function's. *)
let test_capture () =
  let w = Nx.create Nx.float64 [| 2 |] [| 2.; -3. |] in
  let x = Nx.create Nx.float64 [| 2 |] [| 0.25; 4. |] in
  let g remat =
    Rune.grad'
      (fun w ->
        let f x = Nx.sin (Nx.mul x w) in
        let f =
          if remat then Rune.remat T.(tensor @-> returns tensor) f else f
        in
        Nx.sum (f x))
      w
  in
  equal bits_t (bits (g false)) (bits (g true))

(* A set no kernel library computes on, as a GPU set is outside a compiler:
   a software device and a device that does not run on the host. *)
module Bare =
  (val Nx.devices [ Nx_support.memory 3; Nx_array_support.off_host () ])

(* A remat's program, evaluated now, makes each creation on a set without
   kernels as the host does: its host form placed there. *)
let test_bare () =
  let x = Nx.zeros_at Bare.on Nx.float64 [| 3 |] in
  let check msg want f =
    let y = Rune.remat T.(tensor @-> returns tensor) f x in
    equal ~msg:(msg ^ ": placement") (Nx_support.placement ()) Bare.on
      (Nx.placement y);
    equal ~msg bits_t (bits want) (bits (Nx.place Nx.Host.on y))
  in
  check "eye_at" (Nx.eye Nx.float64 3) (fun x ->
      Nx.eye_at (Nx.placement x) Nx.float64 3);
  check "full_like" (Nx.full Nx.float64 [| 3 |] 2.) (fun x -> Nx.full_like x 2.)

(* A signature of no argument: a structure that is itself a function, and holds
   no array. *)
let test_no_argument () =
  let s = T.returns (T.v (fun _ f -> f)) in
  raises (Invalid_argument "Rune.remat: the signature has no argument")
    (fun () ->
      let (_ : (unit -> unit) -> unit -> unit) = Rune.remat s in
      ())

let () =
  exit
    (run "rune remat"
       [
         group "reverse mode"
           [
             test "the record keeps the arguments" test_residuals;
             test "a checkpointed step stacks its carry alone"
               test_scan_residuals;
           ];
         group "outside a jit"
           [
             prop "remat s f is f, bit for bit, under every transformation"
               cases law;
             test "a call traces f once" test_traced;
             test "a capture's gradient is the function's" test_capture;
             test "an addition counts once" test_total;
             test "creations on a set without kernels are the host's" test_bare;
           ];
         group "inside a jit"
           [
             prop ~examples:compiled_examples
               "remat s f is f, bit for bit, where f is exact" compiled_cases
               law_compiled;
             test "a compiled gradient keeps the blocks' arguments alone"
               test_compiled;
             test "a chain's backward pass computes one block at a time"
               test_one_block;
             test "independent chains are computed one at a time" test_chains;
             test "a block over the program's argument keeps it alone"
               test_argument;
             test "blocks over the argument keep nothing they compute"
               test_over_argument;
             test "weights a block transforms are transformed again"
               test_transformed;
             test "a stack's rows are its pieces' kernels' storage, or copies"
               test_placed_pieces;
             test "a result takes a donated argument after the backward pass"
               test_donated;
           ];
         group "in a loop's step"
           [
             test "a scan's checkpointed step keeps its carry alone"
               test_scanned;
             test "a step that takes a gradient holds a trip's storage"
               test_step_gradient;
             test
               "a step that sums a gradient in its carry holds a trip's storage"
               test_summed_step;
             test
               "a scan of a remat'd chain holds a trip's storage at any number \
                of trips"
               test_deep_scan;
             test "stacks read by rows hold what separate arrays hold"
               test_stacked_step;
           ];
         group "refusals"
           [ test "a signature of no argument raises" test_no_argument ];
       ])
