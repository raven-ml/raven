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

let layers = 8 and dim = 32 and batch = 256
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

(* Donation *)

(* A result that takes a donated argument's memory is stored over it after the
   remat's backward pass reads it: the gradient through the remat plus [b],
   donated, takes [b]'s memory, and plus [a], donated, which the backward pass
   reads at every element, gives the same values either way. *)
let test_donated () =
  let w = (Nx.get [ 0 ] (fst ws), Nx.get [ 0 ] (snd ws)) in
  let grad a = Rune.grad' (fun a -> Nx.sum (rm (block w) a)) a in
  let address x = Nx.Repr.read x Nx_array.buffer in
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

(* Footprints *)

(* Small residual blocks from the argument or its sine, each under a remat
   or not, applied or differentiated, over contiguous weights; a scan of
   their gradient over chunks of the argument; and arguments laid out as
   views, holding no element or donated: programs with and without arenas,
   loops, results, recomputations and donations. *)
type program = Chain of { blocks : int; sine : bool; remat : bool; grad : bool }
| Chunks of { k : int; remat : bool }

type argument = Plain | Transposed | Broadcast | Empty | Donated

let pp_program ppf = function
  | Chain c ->
      Format.fprintf ppf "%d blocks%s%s%s" c.blocks
        (if c.sine then " from the sine" else "")
        (if c.remat then ", remats" else "")
        (if c.grad then ", gradient" else "")
  | Chunks c ->
      Format.fprintf ppf "a scan of the gradient over %d chunks%s" c.k
        (if c.remat then ", remats" else "")

let pp_argument ppf a =
  Format.pp_print_string ppf
    (match a with
    | Plain -> "a plain argument"
    | Transposed -> "a transposed argument"
    | Broadcast -> "a broadcast argument"
    | Empty -> "an empty argument"
    | Donated -> "a donated argument")

let small_ws =
  ( Nx.mul_s (Nx.Rng.normal ~key:(Nx.Rng.key 1) f32 [| 3; 8; 16 |]) 0.1,
    Nx.mul_s (Nx.Rng.normal ~key:(Nx.Rng.key 2) f32 [| 3; 16; 8 |]) 0.1 )

let small_chain ~blocks ~sine ~remat a =
  List.fold_left
    (fun a i ->
      let b = block (layer small_ws i) in
      if remat then rm b a else b a)
    (if sine then Nx.sin a else a)
    (List.init blocks Fun.id)

let small_loss ~blocks ~sine ~remat a =
  Nx.sum (small_chain ~blocks ~sine ~remat a)

let small_f program a =
  match program with
  | Chain { blocks; sine; remat; grad = false } ->
      small_chain ~blocks ~sine ~remat a
  | Chain { blocks; sine; remat; grad = true } ->
      Rune.grad' (small_loss ~blocks ~sine ~remat) a
  | Chunks { k; remat } ->
      let n = (Nx.shape a).(0) in
      let chunks = Nx.reshape [| k; n / k; 8 |] a in
      let (), gs =
        Rune.scan T.unit T.tensor T.tensor
          (fun () x ->
            ((), Rune.grad' (small_loss ~blocks:2 ~sine:true ~remat) x))
          ~init:() chunks
      in
      Nx.reshape [| n; 8 |] gs

let argument kind () =
  let normal shape = Nx.Rng.normal ~key:(Nx.Rng.key 4) f32 shape in
  match kind with
  | Plain -> normal [| 16; 8 |]
  | Transposed -> Nx.matrix_transpose (normal [| 8; 16 |])
  | Broadcast -> Nx.broadcast_to [| 16; 8 |] (normal [| 1; 8 |])
  | Empty -> normal [| 0; 8 |]
  | Donated -> Nx.donate (normal [| 16; 8 |])

let one = T.(tensor @-> returns tensor)

let on_host fp =
  List.fold_left
    (fun n (r, b) -> if Rig.equal r Rig.host then n + b else n)
    0 fp

(* The bytes the first call of a fresh compiled [f] at [x ()] allocates on
   the host, its kernels compiled by an uncounted call of another; a donated
   argument that no result takes is freed by the call, and is added back. *)
let first_call s f x =
  ignore (Rune.jit s f (x ()));
  let g = Rune.jit s f in
  let x = x () in
  let donated = Nx.Prim.donated x in
  let memory = Nx.Repr.read x Nx_array.buffer in
  let before = host_settled () in
  let r = g x in
  let held = host_settled () - before in
  let taken = Rig.Buffer.overlaps memory (Nx.Repr.read r Nx_array.buffer) in
  let freed =
    if donated && not taken then
      Rig.allocation Rig.host (Rig.Buffer.length memory)
    else 0
  in
  ignore (Sys.opaque_identity (g, r, x));
  held + freed

(* The law: on one device, what [footprint] counts is what the first call
   allocates and holds; stand-ins of a C-ordered argument give the same. *)
let law_footprint (program, kind) =
  let f = small_f program in
  let x = argument kind in
  let want = on_host (Rune.footprint one f (x ())) in
  equal ~msg:"footprint" int (first_call one f x) want;
  if kind = Plain then
    let stand_in = Rune.eval_shape T.tensor T.tensor Fun.id (x ()) in
    equal ~msg:"over stand-ins" int want
      (on_host (Rune.footprint one f stand_in))

let footprint_cases =
  let chains =
    List.concat_map
      (fun blocks ->
        List.concat_map
          (fun sine ->
            List.concat_map
              (fun remat ->
                List.map
                  (fun grad -> Chain { blocks; sine; remat; grad })
                  [ false; true ])
              [ false; true ])
          [ false; true ])
      [ 1; 2; 3 ]
  in
  let chunks =
    List.concat_map
      (fun k -> List.map (fun remat -> Chunks { k; remat }) [ false; true ])
      [ 2; 4 ]
  in
  let views =
    List.concat_map
      (fun kind ->
        List.map
          (fun p -> (p, kind))
          [
            Chain { blocks = 2; sine = false; remat = false; grad = false };
            Chain { blocks = 2; sine = true; remat = true; grad = true };
            Chunks { k = 2; remat = true };
          ])
      [ Transposed; Broadcast; Empty; Donated ]
  in
  List.map (fun p -> (p, Plain)) (chains @ chunks) @ views

(* A result that takes a donated argument's memory counts none. *)
let test_footprint_donated () =
  let x () = Nx.full f32 [| 64 |] 2. in
  let f x = Nx.mul_s x 3. in
  equal ~msg:"undonated" int (Rig.allocation Rig.host 256)
    (on_host (Rune.footprint one f (x ())));
  equal ~msg:"donated" int 0 (on_host (Rune.footprint one f (Nx.donate (x ()))));
  equal ~msg:"its first call" int 0
    (first_call one f (fun () -> Nx.donate (x ())))

(* Over two memory devices, a gather of each layer's weights inside each
   layer, under remats or not: a first call fits each device's budget at
   what it holds plus its footprint, and is refused a byte below it, on
   either device. Under remats each device holds less, the gathered layers
   gathered again in the backward pass. *)
let test_footprint_devices () =
  let layers = 4 and d = 64 and batch = 32 in
  let ds = [ Nx_support.memory 2; Nx_support.memory 3 ] in
  let module S = (val Nx.devices ds) in
  let ws =
    Nx.place (S.split ~axis:1)
      (Nx.mul_s (Nx.Rng.normal ~key:(Nx.Rng.key 7) f32 [| layers; d; d |]) 0.05)
  in
  let x = Nx.place (S.split ~axis:0) (Nx.ones f32 [| batch; d |]) in
  let gathered w h =
    let y = Nx.matmul h (Nx.place S.on w) in
    Nx.maximum y (Nx.zeros_like y)
  in
  let s = T.(pair tensor tensor @-> returns tensor) in
  let f remat (ws, x) =
    let loss ws =
      Nx.sum
        (List.fold_left
           (fun h i ->
             let b = gathered (Nx.get [ i ] ws) in
             if remat then rm b h else b h)
           (Nx.sin x) (List.init layers Fun.id))
    in
    Rune.grad' loss ws
  in
  let settled d =
    for _ = 1 to 4 do
      Gc.full_major ();
      ignore (Rig.Buffer.create d 0);
      Rig.free_cache d
    done;
    Rig.allocated d
  in
  (* Whether a fresh first call fits each device's budget at what it holds
     plus [extra d]. *)
  let fits remat extra =
    let g = Rune.jit s (f remat) in
    let held = List.map settled ds in
    List.iter2 (fun d h -> Rig.set_budget d (h + extra d)) ds held;
    Fun.protect
      ~finally:(fun () -> List.iter (fun d -> Rig.set_budget d max_int) ds)
      (fun () ->
        match g (ws, x) with
        | r ->
            ignore (Sys.opaque_identity r);
            true
        | exception Rig.Out_of_memory _ -> false)
  in
  let footprint remat =
    List.filter
      (fun (r, _) -> List.exists (Rig.equal r) ds)
      (Rune.footprint s (f remat) (ws, x))
  in
  List.iter
    (fun remat ->
      let msg = if remat then "under remats" else "without remats" in
      ignore (Rune.jit s (f remat) (ws, x));
      let fp = footprint remat in
      let at d = List.assq d fp in
      equal ~msg:(msg ^ ": fits its footprint") bool true (fits remat at);
      List.iter
        (fun d ->
          let below r = if r == d then at r - 1 else at r in
          equal
            ~msg:(Printf.sprintf "%s: refused a byte below on %s" msg (Rig.name d))
            bool false (fits remat below))
        ds)
    [ false; true ];
  List.iter2
    (fun (r, p) (_, q) ->
      less
        ~msg:
          (Printf.sprintf "%s: %d bytes under remats, %d without" (Rig.name r)
             q p)
        int ~than:p q)
    (footprint false) (footprint true)

let test_footprint_arguments () =
  raises_match (Exn.invalid_arg ~sub:"more than one argument") (fun () ->
      Rune.footprint T.(tensor @-> tensor @-> returns tensor) Nx.add a)

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
   a memory device and a device that does not run on the host. *)
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

let () =
  exit
    (run "rune remat"
       [
         group "modes" [ prop "remat is its function in every mode" cases law ];
         group "memory"
           [
             test "the record keeps the arguments" test_residuals;
             test "a checkpointed step" test_scan_residuals;
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
             test "a scan's checkpointed step keeps its carry alone"
               test_scanned;
             test "a step that takes a gradient plans each trip"
               test_step_gradient;
             test "a result takes a donated argument after the backward pass"
               test_donated;
           ];
         group "footprint"
           [
             Windtrap.cases "footprint is what a first call allocates"
               ~id:(fun (p, a) ->
                 Format.asprintf "%a, %a" pp_program p pp_argument a)
               footprint_cases law_footprint;
             test "a result in a donated argument counts none"
               test_footprint_donated;
             test "over two devices a call fits its footprint, not a byte less"
               test_footprint_devices;
             test "a signature of several arguments raises"
               test_footprint_arguments;
           ];
         group "interface"
           [
             test "traced once, additions once" test_once;
             test "a capture" test_capture;
             test "creations on a set without kernels" test_bare;
           ];
       ])
