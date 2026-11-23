(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.program, Rune.eval_shape and Rune.footprint: what a function traces,
   printed, its result's structure without computing it, and the bytes its first
   compiled call allocates. *)

open Windtrap
module T = Nx.Ptree

let f32 = Nx.float32
let v xs = Nx.create f32 [| Array.length xs |] xs
let invalid ~sub f = raises_match (Exn.invalid_arg ~sub) f
let print p = Format.asprintf "%a" Rune.pp_program p

(* The tangent of each map is computed with its primal in one map of two
   outputs. *)
let test_jvp_fused () =
  let f x = Rune.jvp' (fun x -> Nx.sin (Nx.mul x x)) x x in
  let p = Rune.program T.tensor T.(pair tensor tensor) f (v [| 1.; 2. |]) in
  print_string (print p);
  expect (output ()) @@ __POS_OF__ {|
    program (%0)
      %1 : float32 [2] at CPU, %2 : float32 [2] at CPU = Map [mul(x0, x1); add(mul(x2, x1), mul(x0, x3))] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU) (x2: float32 [2] at CPU) (x3: float32 [2] at CPU)
        with x0 = %0, x1 = %0, x2 = %0, x3 = %0
      %3 : float32 [2] at CPU, %4 : float32 [2] at CPU = Map [sin(x0); mul(x1, cos(x0))] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
        with x0 = %1, x1 = %2
      return %3, %4
    |}

(* A scan carries its step's program. *)
let test_scan () =
  let f x =
    fst
      (Rune.scan'
         (fun c r -> (Nx.add c r, r))
         ~init:x
         (Nx.reshape [| 2; 1 |] x))
  in
  print_string (print (Rune.program T.tensor T.tensor f (v [| 1.; 2. |])));
  expect (output ()) @@ __POS_OF__ {|
    program (%0)
      %1 : float32 [2; 1] at CPU = Move (x0: float32 [2] at CPU)
        with x0 = %0
      %2 : float32 [2] at CPU, %3 : float32 [2; 1] at CPU = scan 2 rows, init (%0), xs (%1), captures ()
        step
          program (%4, %5)
            %6 : float32 [2] at CPU = Move (x0: float32 [1] at CPU)
              with x0 = %5
            %7 : float32 [2] at CPU = Map [add(x0, x1)] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
              with x0 = %4, x1 = %6
            return %7, %5
      return %2
    |}

(* A constant, a detach, a total's addition and a grad's tape, each
   printed. *)
let test_constructs () =
  let w = v [| 0.5; -0. |] in
  let total = Rune.Total.make () in
  let f x =
    Rune.Total.add total x;
    Nx.add (Rune.detach x) (Rune.grad' (fun x -> Nx.sum (Nx.mul x w)) x)
  in
  let p, _ =
    Rune.Total.collect total ~init:(Nx.zeros f32 [| 2 |]) (fun () ->
        Rune.program T.tensor T.tensor f (v [| 1.; 2. |]))
  in
  print_string (print p);
  expect (output ()) @@ __POS_OF__ {|
    c0 : float32 [2] at CPU
    program (%0), totals (%1)
      %2 : float32 [2] at CPU = accumulate %1 + %0
      %3 : float32 [2] at CPU = Map [mul(x0, x1)] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
        with x0 = %0, x1 = c0
      %4 : float32 [] at CPU = Reduce [Sum x0] over [0] (x0: float32 [2] at CPU)
        with x0 = %3
      %5 : float32 [] at CPU = Map [float32:0x0000803f] at CPU
      %6 : float32 [1] at CPU = Move (x0: float32 [] at CPU)
        with x0 = %5
      %7 : float32 [2] at CPU = Move (x0: float32 [1] at CPU)
        with x0 = %6
      %8 : float32 [2] at CPU = Map [x0] (x0: float32 [2] at CPU)
        with x0 = %7
      %9 : float32 [2] at CPU = Map [mul(x1, x0)] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
        with x0 = c0, x1 = %8
      %10 : float32 [2] at CPU = detach %0
      %11 : float32 [2] at CPU = Map [add(x0, x1)] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
        with x0 = %10, x1 = %9
      return %11
      totals %2
    |}

(* An iterate carries its test and its step; nested programs number their
   variables after their parent's. *)
let test_iterate () =
  let f x =
    Rune.iterate' ~max:4
      ~until:(fun c -> Nx.greater_s (Nx.sum c) 10.)
      (fun c -> Nx.mul c (Nx.scalar_like c 2.))
      x
  in
  print_string (print (Rune.program T.tensor T.tensor f (v [| 1.; 2. |])));
  expect (output ()) @@ __POS_OF__ {|
    program (%0)
      %1 : float32 [] at CPU = Map [float32:0x00000040] at CPU
      %2 : float32 [2] at CPU = Move (x0: float32 [] at CPU)
        with x0 = %1
      %3 : float32 [2] at CPU = iterate at most 4, init (%0), captures () and (%2)
        until
          program (%4)
            %5 : float32 [] at CPU = Reduce [Sum x0] over [0] (x0: float32 [2] at CPU)
              with x0 = %4
            %6 : bool [] at CPU = Map [less(float32:0x00002041, x0)] (x0: float32 [] at CPU)
              with x0 = %5
            return %6
        step
          program (%7, %8)
            %9 : float32 [2] at CPU = Map [mul(x0, x1)] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
              with x0 = %7, x1 = %8
            return %9
      return %3
    |}

(* A custom rule carries its primal and its derivative: under grad, the rule's
   call on the primals, then its pullback. *)
let test_custom () =
  let sin_vjp =
    Rune.custom_vjp T.tensor T.tensor (fun x ->
        (Nx.sin x, fun g -> Nx.mul g (Nx.cos x)))
  in
  let f x = Rune.grad' (fun x -> Nx.sum (sin_vjp x)) x in
  print_string (print (Rune.program T.tensor T.tensor f (v [| 1.; 2. |])));
  expect (output ()) @@ __POS_OF__ {|
    program (%0)
      %1 : float32 [2] at CPU, %2 : float32 [2] at CPU = Rune.custom_vjp (%0), captures ()
        primal
          program (%10)
            %11 : float32 [2] at CPU = Map [sin(x0)] (x0: float32 [2] at CPU)
              with x0 = %10
            %12 : float32 [2] at CPU = Map [cos(x0)] (x0: float32 [2] at CPU)
              with x0 = %10
            return %11, %12
        pullback
          program (%13, %14)
            %15 : float32 [2] at CPU = Map [mul(x0, x1)] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
              with x0 = %13, x1 = %14
            return %15
      %3 : float32 [] at CPU = Reduce [Sum x0] over [0] (x0: float32 [2] at CPU)
        with x0 = %1
      %4 : float32 [] at CPU = Map [float32:0x0000803f] at CPU
      %5 : float32 [1] at CPU = Move (x0: float32 [] at CPU)
        with x0 = %4
      %6 : float32 [2] at CPU = Move (x0: float32 [1] at CPU)
        with x0 = %5
      %7 : float32 [2] at CPU = Map [x0] (x0: float32 [2] at CPU)
        with x0 = %6
      %8 : float32 [2] at CPU = Map [mul(x0, x1)] (x0: float32 [2] at CPU) (x1: float32 [2] at CPU)
        with x0 = %7, x1 = %2
      %9 : float32 [2] at CPU = Copy (x0: float32 [2] at CPU)
        with x0 = %8
      return %9
    |}

(* A linear call carries its function and transpose: under grad, the call of the
   tangent map on the primals and the linear call of its transpose. *)
let test_linear () =
  let twice =
    Rune.custom_transpose T.tensor T.tensor
      (fun v -> Nx.mul_s v 2.)
      ~transpose:(fun w -> Nx.mul_s w 3.)
  in
  let f x =
    Rune.grad'
      (fun x ->
        Nx.sum (Rune.custom_jvp T.tensor T.tensor (fun x -> (x, twice)) x))
      x
  in
  print_string (print (Rune.program T.tensor T.tensor f (v [| 1.; 2. |])));
  expect (output ()) @@ __POS_OF__ {|
    program (%0)
      %1 : float32 [2] at CPU = Rune.custom_jvp (%0), captures ()
        primal
          program (%9)
            return %9
        tangent map
          program (%10)
            %11 : float32 [2] at CPU = linear Rune.custom_transpose (%10), captures (), residuals ()
              function
                program (%12)
                  %13 : float32 [2] at CPU = Map [mul(x0, float32:0x00000040)] (x0: float32 [2] at CPU)
                    with x0 = %12
                  return %13
              transpose
                program (%14)
                  %15 : float32 [2] at CPU = Map [mul(x0, float32:0x00004040)] (x0: float32 [2] at CPU)
                    with x0 = %14
                  return %15
            return %11
      %2 : float32 [] at CPU = Reduce [Sum x0] over [0] (x0: float32 [2] at CPU)
        with x0 = %1
      %3 : float32 [] at CPU = Map [float32:0x0000803f] at CPU
      %4 : float32 [1] at CPU = Move (x0: float32 [] at CPU)
        with x0 = %3
      %5 : float32 [2] at CPU = Move (x0: float32 [1] at CPU)
        with x0 = %4
      %6 : float32 [2] at CPU = Map [x0] (x0: float32 [2] at CPU)
        with x0 = %5
      %7 : float32 [2] at CPU = linear Rune.custom_transpose (%6), captures (), residuals ()
        function
          program (%14)
            %15 : float32 [2] at CPU = Map [mul(x0, float32:0x00004040)] (x0: float32 [2] at CPU)
              with x0 = %14
            return %15
        transpose
          program (%12)
            %13 : float32 [2] at CPU = Map [mul(x0, float32:0x00000040)] (x0: float32 [2] at CPU)
              with x0 = %12
            return %13
      %8 : float32 [2] at CPU = Copy (x0: float32 [2] at CPU)
        with x0 = %7
      return %8
    |}

(* Two traces of one function print alike. *)
let test_stable () =
  let f (x, y) = Nx.add (Nx.mul x y) (Nx.scalar_like x 2.) in
  let s = T.(pair tensor tensor) in
  let trace () = print (Rune.program s T.tensor f (v [| 1. |], v [| 3. |])) in
  equal string (trace ()) (trace ())

(* A captured array is data of its program: two traces that differ only in a
   capture's elements print alike, on the host and on another device, while
   the same number written into the program as a literal, or made by the
   traced function, prints apart. *)
module Away = (val Nx.devices [ Result.get_ok (Rig.software "rune-program") ])

let captures_alike (type d) (at : d Nx.Placement.t) =
  let x = Nx.place at (v [| 1.; 2. |]) in
  let trace f = print (Rune.program T.tensor T.tensor f x) in
  let captured c =
    let c = Nx.scalar f32 c in
    fun x -> Nx.mul x (Nx.place (Nx.beside x) c)
  in
  let literal c x = Nx.mul_s x c in
  let made c x = Nx.mul x (Nx.scalar_like x c) in
  equal ~msg:"captured" string (trace (captured 0.5)) (trace (captured 3.));
  not_equal ~msg:"literal" string (trace (literal 0.5)) (trace (literal 3.));
  not_equal ~msg:"made in the trace" string (trace (made 0.5)) (trace (made 3.))

let test_captures_are_data () =
  captures_alike Nx.Host.on;
  captures_alike Away.on

(* The result's structure, dtypes, shapes and placements, without computing;
   an operation on a stand-in raises. *)
let test_eval_shape () =
  let n = ref 0 in
  let f x =
    incr n;
    (Nx.sum ~axes:[ 0 ] x, Nx.cast Nx.int32 x)
  in
  let x = Nx.create f32 [| 2; 3 |] (Array.make 6 1.) in
  let s, c = Rune.eval_shape T.tensor T.(pair tensor tensor) f x in
  equal ~msg:"f runs once" int 1 !n;
  equal (array int) [| 3 |] (Nx.shape s);
  equal (array int) [| 2; 3 |] (Nx.shape c);
  equal ~msg:"dtype" string "int32" (Nx.Dtype.name (Nx.dtype c));
  let at y = Format.asprintf "%a" Nx.Placement.pp (Nx.placement y) in
  equal ~msg:"placement" string (at x) (at s);
  invalid ~sub:"Rune.eval_shape" (fun () -> Nx.to_array s)

(* Footprints *)

(* Residual blocks [a + relu (a w1) w2], each under a remat or not. *)
let block (w1, w2) a =
  let h = Nx.matmul a w1 in
  Nx.add a (Nx.matmul (Nx.maximum h (Nx.zeros_like h)) w2)

let rm f = Rune.remat T.(tensor @-> returns tensor) f
let layer (w1, w2) i = (Nx.get [ i ] w1, Nx.get [ i ] w2)

(* The bytes [d] holds once finalisers have run and its cache is empty. *)
let settled d =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create d 0);
    Rig.free_cache d
  done;
  Rig.allocated d

(* Small residual blocks from the argument or its sine, each under a remat or
   not, applied or differentiated, over contiguous weights; a scan of their
   gradient over chunks of the argument; and arguments laid out as views,
   holding no element or donated: programs with and without arenas, loops,
   results, recomputations and donations. *)
type footprinted =
  | Chain of { blocks : int; sine : bool; remat : bool; grad : bool }
  | Chunks of { k : int; remat : bool }

type argument = Plain | Transposed | Broadcast | Empty | Donated

let pp_footprinted ppf = function
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

let weights =
  lazy
    ( Nx.mul_s (Nx.Rng.normal ~key:(Nx.Rng.key 1) f32 [| 3; 8; 16 |]) 0.1,
      Nx.mul_s (Nx.Rng.normal ~key:(Nx.Rng.key 2) f32 [| 3; 16; 8 |]) 0.1 )

let chain ws ~blocks ~sine ~remat a =
  List.fold_left
    (fun a i ->
      let b = block (layer ws i) in
      if remat then rm b a else b a)
    (if sine then Nx.sin a else a)
    (List.init blocks Fun.id)

let loss ws ~blocks ~sine ~remat a = Nx.sum (chain ws ~blocks ~sine ~remat a)

let compute ws program a =
  match program with
  | Chain { blocks; sine; remat; grad = false } ->
      chain ws ~blocks ~sine ~remat a
  | Chain { blocks; sine; remat; grad = true } ->
      Rune.grad' (loss ws ~blocks ~sine ~remat) a
  | Chunks { k; remat } ->
      let n = (Nx.shape a).(0) in
      let chunks = Nx.reshape [| k; n / k; 8 |] a in
      let (), gs =
        Rune.scan T.unit T.tensor T.tensor
          (fun () x -> ((), Rune.grad' (loss ws ~blocks:2 ~sine:true ~remat) x))
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

(* The bytes the first call of a fresh compiled [f] at [x ()] allocates on the
   host, its kernels compiled by an uncounted call of another; a donated
   argument that no result takes is freed by the call, and is added back. *)
let first_call s f x =
  ignore (Rune.jit s f (x ()));
  let g = Rune.jit s f in
  let x = x () in
  let donated = Nx.Prim.donated x in
  let memory = Nx.Repr.read ~by:"test" x Nx_array.buffer in
  let before = settled Rig.host in
  let r = g x in
  let held = settled Rig.host - before in
  let taken =
    Rig.Buffer.overlaps memory (Nx.Repr.read ~by:"test" r Nx_array.buffer)
  in
  let freed =
    if donated && not taken then
      Rig.allocation Rig.host (Rig.Buffer.length memory)
    else 0
  in
  ignore (Sys.opaque_identity (g, r, x));
  held + freed

(* On one device, what [footprint] counts is what the first call allocates and
   holds; stand-ins of a C-ordered argument give the same. *)
let law_footprint (program, kind) =
  Rune_support.Compiled.here ();
  let f = compute (Lazy.force weights) program in
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
  Rune_support.Compiled.here ();
  let x () = Nx.full f32 [| 64 |] 2. in
  let f x = Nx.mul_s x 3. in
  equal ~msg:"undonated" int
    (Rig.allocation Rig.host 256)
    (on_host (Rune.footprint one f (x ())));
  equal ~msg:"donated" int 0 (on_host (Rune.footprint one f (Nx.donate (x ()))));
  equal ~msg:"its first call" int 0
    (first_call one f (fun () -> Nx.donate (x ())))

(* Over two software devices, a gather of each layer's weights inside each
   layer, under remats or not: a first call fits each device's budget at what it
   holds plus its footprint, and is refused a byte below it, on either device.
   Under remats each device holds less, the gathered layers gathered again in
   the backward pass. *)
let test_footprint_devices () =
  let layers = 4 and dim = 64 and batch = 32 in
  let ds = [ Nx_support.memory 2; Nx_support.memory 3 ] in
  let module S = (val Nx.devices ds) in
  let ws =
    Nx.place (S.split ~axis:1)
      (Nx.mul_s
         (Nx.Rng.normal ~key:(Nx.Rng.key 7) f32 [| layers; dim; dim |])
         0.05)
  in
  let x = Nx.place (S.split ~axis:0) (Nx.ones f32 [| batch; dim |]) in
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
  (* Whether a fresh first call runs within each device's budget at what it
     holds plus [extra d], or is refused for memory. *)
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
            `Runs
        | exception Rig.Out_of_memory _ -> `Refused)
  in
  let outcome =
    Testable.make
      ~pp:(fun ppf o ->
        Format.pp_print_string ppf
          (match o with `Runs -> "runs" | `Refused -> "refused for memory"))
      ~equal:( = )
  in
  Rune_support.Compiled.here ();
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
      equal ~msg:(msg ^ ": fits its footprint") outcome `Runs (fits remat at);
      List.iter
        (fun d ->
          let below r = if r == d then at r - 1 else at r in
          equal
            ~msg:
              (Printf.sprintf "%s: refused a byte below on %s" msg (Rig.name d))
            outcome `Refused (fits remat below))
        ds)
    [ false; true ];
  List.iter2
    (fun (r, p) (_, q) ->
      less
        ~msg:
          (Printf.sprintf "%s: %d bytes under remats, %d without" (Rig.name r) q
             p)
        int ~than:p q)
    (footprint false) (footprint true)

let test_footprint_arguments () =
  raises
    (Invalid_argument "Rune.footprint: the signature has more than one argument")
    (fun () ->
      Rune.footprint
        T.(tensor @-> tensor @-> returns tensor)
        Nx.add (Nx.ones f32 [| 2 |]))

let () =
  exit
    (run "rune program"
       [
         group "program"
           [
             test "a map's tangent is in its map" test_jvp_fused;
             test "a scan carries its step" test_scan;
             test "constants, detach, totals and grad" test_constructs;
             test "an iterate carries its test and step" test_iterate;
             test "a custom rule carries its programs" test_custom;
             test "a linear call carries its programs" test_linear;
             test "two traces print alike" test_stable;
             test "a capture's elements are not the program's"
               test_captures_are_data;
             test "eval_shape" test_eval_shape;
           ];
         group "footprint"
           [
             cases "footprint is what a first call allocates"
               ~id:(fun (p, a) ->
                 Format.asprintf "%a, %a" pp_footprinted p pp_argument a)
               footprint_cases law_footprint;
             test "a result in a donated argument counts none"
               test_footprint_donated;
             test "over two devices a call fits its footprint, not a byte less"
               test_footprint_devices;
             test "a signature of several arguments raises"
               test_footprint_arguments;
           ];
       ])
