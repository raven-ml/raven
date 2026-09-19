(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jit on the Metal device: kernels compile and run on the GPU, data moves
   through copies. Compiled only on macOS. *)

open Windtrap
open Rune_test_support.Support

let test_elementwise_on_metal () =
  let f x = Nx.tanh (Nx.add (Nx.mul x x) x) in
  let g = Rune.jit' ~device:"METAL" f in
  let x = vec32 [| 1.0; -2.0; 0.5 |] in
  check_arr ~msg:"first call" (to_arr (f x)) (g x);
  check_arr ~msg:"replay" (to_arr (f x)) (g x)

let test_matmul_grad_on_metal () =
  let w = Nx.create f32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let f x = Nx.sum (Nx.matmul x w) in
  let g = Rune.jit' ~device:"METAL" (fun x -> Rune.grad' f x) in
  let x = Nx.create f32 [| 2; 3 |] [| 1.0; 0.0; -1.0; 0.5; 2.0; 1.0 |] in
  check_arr ~msg:"grad through metal jit" (to_arr (Rune.grad' f x)) (g x)

(* Multi-kernel compiled traces replay as batched device graphs: the kernels are
   recorded into an indirect command buffer on the first call and later calls
   patch the rebound buffers (fresh outputs, resident inputs) into it instead of
   launching each kernel individually. *)
let test_graph_batched_replay () =
  let w1 =
    Nx.create f32 [| 4; 4 |]
      (Array.init 16 (fun i -> (float_of_int (i mod 5) /. 4.0) -. 0.5))
  in
  let w2 =
    Nx.create f32 [| 4; 4 |]
      (Array.init 16 (fun i -> float_of_int (i mod 3) -. 1.0))
  in
  let f x = Nx.matmul (Nx.tanh (Nx.matmul x w1)) w2 in
  let g = Rune.jit' ~device:"METAL" f in
  let launches0 = !Tolk.Realize.graph_launches in
  List.iteri
    (fun i data ->
      let x = Nx.create f32 [| 2; 4 |] data in
      check_arr
        ~msg:(Printf.sprintf "call %d matches eager" (i + 1))
        (to_arr (f x))
        (g x))
    [
      Array.init 8 (fun i -> float_of_int i /. 8.0);
      Array.init 8 (fun i -> float_of_int (7 - i));
      Array.make 8 (-0.25);
    ];
  is_true ~msg:"every call dispatched a device graph"
    (!Tolk.Realize.graph_launches - launches0 >= 3);
  let x = Nx.create f32 [| 4; 4 |] (Array.init 16 (fun i -> float_of_int i)) in
  check_arr ~msg:"a resident output feeds the next call"
    (to_arr (f (f x)))
    (g (g x))

(* Placed weights. The compiled trace binds their buffers as its constants, and
   the batched replay reads them on every call with nothing uploaded. *)

let delta f =
  let s0 = Rune.jit_stats () in
  let r = f () in
  let s1 = Rune.jit_stats () in
  (r, s1.bytes_to_device - s0.bytes_to_device)

let weights () =
  ( Nx.create f32 [| 4; 4 |]
      (Array.init 16 (fun i -> (float_of_int (i mod 5) /. 4.0) -. 0.5)),
    Nx.create f32 [| 4; 4 |]
      (Array.init 16 (fun i -> float_of_int (i mod 3) -. 1.0)) )

let test_placed_weights_bind () =
  let w1, w2 = weights () in
  let f w1 w2 x = Nx.matmul (Nx.tanh (Nx.matmul x w1)) w2 in
  let p1 = Rune.to_device ~device:"METAL" w1 in
  let p2 = Rune.to_device ~device:"METAL" w2 in
  Gc.full_major ();
  let base = (Rune.jit_stats ()).resident_bytes in
  let g = Rune.jit' ~device:"METAL" (f p1 p2) in
  let launches0 = !Tolk.Realize.graph_launches in
  List.iter
    (fun v ->
      let x = Nx.create f32 [| 2; 4 |] (Array.make 8 v) in
      let y, up = delta (fun () -> g x) in
      equal ~msg:"only the input is uploaded" int (Nx.nbytes x) up;
      check_arr ~msg:"matches eager" (to_arr (f w1 w2 x)) y)
    [ 0.5; -1.0; 2.0 ];
  is_true ~msg:"the calls replayed as device graphs"
    (!Tolk.Realize.graph_launches - launches0 >= 3);
  check_arr ~msg:"a bound weight reads back" (to_arr w1) p1;
  Gc.full_major ();
  equal ~msg:"and keeps its buffer" int 0
    ((Rune.jit_stats ()).resident_bytes - base);
  let x = Nx.create f32 [| 2; 4 |] (Array.make 8 0.25) in
  check_arr ~msg:"after the read" (to_arr (f w1 w2 x)) (g x);
  (* A second compiled function shares the buffers. *)
  let h = Rune.jit' ~device:"METAL" (fun x -> Nx.add (Nx.matmul x p1) x) in
  let y, up = delta (fun () -> h x) in
  equal ~msg:"a second function uploads its input only" int (Nx.nbytes x) up;
  check_arr ~msg:"second function" (to_arr (Nx.add (Nx.matmul x w1) x)) y

let test_bound_input_is_not_donated () =
  let w1, _ = weights () in
  let p = Rune.to_device ~device:"METAL" w1 in
  let g = Rune.jit' ~device:"METAL" (fun x -> Nx.matmul x p) in
  let x = Nx.create f32 [| 2; 4 |] (Array.make 8 1.0) in
  ignore (g x);
  let step = Rune.jit' ~device:"METAL" ~donate:true (fun m -> Nx.mul_s m 2.0) in
  let y, up = delta (fun () -> step p) in
  equal ~msg:"the bound input seeds with no transfer" int 0 up;
  check_arr ~msg:"result" (to_arr (Nx.mul_s w1 2.0)) y;
  check_arr ~msg:"the bound value is still readable" (to_arr w1) p;
  check_arr ~msg:"and still the constant" (to_arr (Nx.matmul x w1)) (g x)

let test_capture_resident_elsewhere () =
  Unix.putenv "RUNE_JIT_FORCE_COPY" "1";
  Fun.protect
    ~finally:(fun () -> Unix.putenv "RUNE_JIT_FORCE_COPY" "0")
    (fun () ->
      let w1, _ = weights () in
      let p = Rune.to_device ~device:"CPU" w1 in
      let g = Rune.jit' ~device:"METAL" (fun x -> Nx.matmul x p) in
      let x = Nx.create f32 [| 2; 4 |] (Array.make 8 1.0) in
      let y, up = delta (fun () -> g x) in
      equal ~msg:"the capture goes through the host and is uploaded" int
        (Nx.nbytes x + Nx.nbytes w1)
        up;
      check_arr ~msg:"result" (to_arr (Nx.matmul x w1)) y)

let tests =
  [
    group "metal device"
      [
        test "element-wise chain matches eager" test_elementwise_on_metal;
        test "grad inside jit matches eager" test_matmul_grad_on_metal;
        test "multi-kernel traces replay as device graphs"
          test_graph_batched_replay;
      ];
    group "placed weights"
      [
        test "placed weights bind and replay as device graphs"
          test_placed_weights_bind;
        test "a bound input is not consumed by donation"
          test_bound_input_is_not_donated;
        test "a capture resident on another device is uploaded"
          test_capture_resident_elsewhere;
      ];
  ]

let () = run "rune jit metal" tests
