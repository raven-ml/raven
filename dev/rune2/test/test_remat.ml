(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.remat: its function in every mode, bit for bit, and under reverse mode a
   record of its arguments alone. *)

open Windtrap
open Compose
module Rune = Rune2
module T = Nx.Ptree
module D = Nx.Dtype

let two () = T.(tensor @-> tensor @-> returns tensor)

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
  let p = T.(pair tensor tensor) in
  let objective f (x, y) = Nx.sum (f x y) in
  same2 "grad" (fun f -> Rune.grad p (objective f) (x, y));
  (* Compiled, a function gives eager's bits where every function it applies is
     computed exactly. *)
  let fe x y = eval x y (elementwise (exact e)) in
  let re = Rune.remat (two ()) fe in
  let same_exact msg g =
    equal ~msg tt
      (outcome (fun () -> bits (g fe)))
      (outcome (fun () -> bits (g re)))
  in
  let same2_exact msg g =
    let both (a, b) = (bits a, bits b) in
    equal ~msg t2
      (outcome (fun () -> both (g fe)))
      (outcome (fun () -> both (g re)))
  in
  same2_exact "grad under jit" (fun f ->
      Rune.jit
        T.(tensor @-> tensor @-> returns (pair tensor tensor))
        (fun x y -> Rune.grad p (objective f) (x, y))
        x y);
  same2 "grad of grad" (fun f ->
      Rune.grad p
        (fun (x, y) ->
          let gx, gy = Rune.grad p (objective f) (x, y) in
          Nx.add (Nx.sum (Nx.mul gx gx)) (Nx.sum gy))
        (x, y));
  same "jvp of grad" (fun f ->
      snd (Rune.jvp' (fun x -> fst (Rune.grad p (objective f) (x, y))) x t));
  let xs = Nx.stack [ x; Nx.neg x ] in
  same "vmap" (fun f -> Rune.vmap' (fun x -> f x y) xs);
  same "grad of vmap" (fun f ->
      Rune.grad' (fun xs -> Nx.sum (Rune.vmap' (fun x -> f x y) xs)) xs);
  same "vmap of grad" (fun f ->
      Rune.vmap' (fun x -> Rune.grad' (fun x -> Nx.sum (f x y)) x) xs);
  same_exact "jit" (fun f -> Rune.jit (two ()) f x y)

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
         ~f:(fun c r -> (Nx.add (step c) r, ()))
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

(* A compiled gradient through residual blocks [a + relu (a w1) w2], from the
   sine of the argument: the plain program keeps each block's pre-activations
   for the backward pass, where under remats it keeps each block's argument,
   and computes one block's pre-activations at a time, in the backward pass.
   The bytes a compiled function holds are its program's storage: under remats
   no more than the blocks' arguments, two blocks' pre-activations and four
   rows of cotangents, and no copy of the weights, strided arguments of the
   program that the blocks capture. Each call
   runs compiled, as an evaluated one would hold nothing, and gives eager's
   gradient but for the order of its sums. *)
let test_compiled () =
  let layers = 8 and dim = 32 and batch = 256 in
  let hidden = 8 * dim in
  (* Each stack is a transposed view, which the program reads in place. *)
  let w r c =
    Nx.matrix_transpose
      (Nx.mul_s (Nx.Rng.normal ~key:(Nx.Rng.key r) f32 [| layers; c; r |]) 0.05)
  in
  let ws = (w dim hidden, w hidden dim) in
  let block (w1, w2) a =
    let h = Nx.matmul a w1 in
    Nx.add a (Nx.matmul (Nx.maximum h (Nx.zeros_like h)) w2)
  in
  let loss remat (w1, w2) a =
    Nx.sum
      (List.fold_left
         (fun a i ->
           let w = (Nx.get [ i ] w1, Nx.get [ i ] w2) in
           if remat then Rune.remat T.(tensor @-> returns tensor) (block w) a
           else block w a)
         (Nx.sin a) (List.init layers Fun.id))
  in
  let host = Rig.host in
  let settled () =
    for _ = 1 to 4 do
      Gc.full_major ();
      ignore (Rig.Buffer.create host 0);
      Rig.free_cache host
    done;
    Rig.allocated host
  in
  let a = Nx.ones f32 [| batch; dim |] in
  let expected = Nx.to_array (Nx.flatten (Rune.grad' (loss false ws) a)) in
  let held remat =
    let measure () =
      let g =
        Rune.jit
          T.(pair tensor tensor @-> tensor @-> returns tensor)
          (fun ws a -> Rune.grad' (loss remat ws) a)
      in
      let before = settled () and calls = Rune.evaluations () in
      let r = g ws a in
      let held = settled () - before in
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
  in
  let row = 4 * batch * dim in
  let bound = ((layers + 4) * row) + (2 * 4 * batch * hidden) in
  let plain = held false and recomputed = held true in
  less ~msg:"without remats" int ~than:plain bound;
  less
    ~msg:(Printf.sprintf "%d bytes under remats, %d without" recomputed plain)
    int ~than:bound recomputed

(* Interface *)

module Total = Rune.Total

let test_once () =
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
  equal ~msg:"traced once by reverse mode" int 1 !calls;
  (* An addition counts once, however often the backward pass computes the
     function again. *)
  let t = Total.make () in
  let r =
    Rune.remat
      T.(tensor @-> returns tensor)
      (fun x ->
        Total.add t (Nx.sum x);
        three x)
  in
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
   a memory device and an io device, which does not run on the host. *)
module Bare =
  (val Nx.devices [ Nx_support.memory 3; Nx_array_support.io_device () ])

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

let () =
  exit
    (run "rune2 remat"
       [
         group "modes" [ prop "remat is its function in every mode" cases law ];
         group "memory"
           [
             test "the record keeps the arguments" test_residuals;
             test "a checkpointed step" test_scan_residuals;
             test "a compiled gradient keeps the blocks' arguments alone"
               test_compiled;
           ];
         group "interface"
           [
             test "traced once, additions once" test_once;
             test "a capture" test_capture;
             test "creations on a set without kernels" test_bare;
           ];
       ])
