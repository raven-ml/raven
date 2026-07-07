(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jit on the CUDA device: kernels compile through NVRTC and run on the GPU,
   data moves through copies. Skipped when no CUDA driver or GPU is present. *)

open Windtrap
open Rune_test_support.Support

let cuda_probe =
  lazy
    (match Tolk_cuda.create "CUDA" with
    | _ -> Ok ()
    | exception Failure msg -> Error msg)

let require_cuda () =
  match Lazy.force cuda_probe with
  | Ok () -> ()
  | Error msg -> skip ~reason:msg ()

let test_elementwise_on_cuda () =
  require_cuda ();
  let f x = Nx.tanh (Nx.add (Nx.mul x x) x) in
  let g = Rune.jit' ~device:"CUDA" f in
  let x = vec32 [| 1.0; -2.0; 0.5 |] in
  check_arr ~msg:"first call" (to_arr (f x)) (g x);
  check_arr ~msg:"replay" (to_arr (f x)) (g x)

let test_matmul_grad_on_cuda () =
  require_cuda ();
  let w = Nx.create f32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let f x = Nx.sum (Nx.matmul x w) in
  let g = Rune.jit' ~device:"CUDA" (fun x -> Rune.grad' f x) in
  let x = Nx.create f32 [| 2; 3 |] [| 1.0; 0.0; -1.0; 0.5; 2.0; 1.0 |] in
  check_arr ~msg:"grad through cuda jit" (to_arr (Rune.grad' f x)) (g x)

(* Captured tensors are uploaded once per compilation and stay resident on the
   device: a later mutation of the capture is not observed (unlike on the CPU
   device, whose buffers alias the tensor's memory). *)
let test_capture_is_uploaded_once () =
  require_cuda ();
  let c = vec32 [| 10.0; 20.0; 30.0 |] in
  let g = Rune.jit' ~device:"CUDA" (fun x -> Nx.add x c) in
  check_arr ~msg:"initial capture" [| 11.0; 21.0; 31.0 |]
    (g (vec32 [| 1.0; 1.0; 1.0 |]));
  Nx.blit (vec32 [| 0.0; 0.0; 0.0 |]) c;
  check_arr ~msg:"capture stays at its compile-time value"
    [| 11.0; 21.0; 31.0 |]
    (g (vec32 [| 1.0; 1.0; 1.0 |]))

(* Captures are compile-time constants: a function that assigns to one fails
   at trace time. Mutable state belongs in the input structure. *)
let test_assign_to_capture_raises () =
  require_cuda ();
  let s = vec32 [| 1.0; 2.0 |] in
  let g =
    Rune.jit' ~device:"CUDA" (fun x ->
        Nx.blit (Nx.add s x) s;
        Nx.mul_s s 10.0)
  in
  raises_match
    (fun exn -> match exn with Rune.Jit_error _ -> true | _ -> false)
    (fun () -> ignore (g (vec32 [| 1.0; 1.0 |])))

(* Device residency: outputs stay on the GPU until read, and unread outputs
   fed back as inputs move no bytes. *)

let delta f =
  let s0 = Rune.jit_stats () in
  let r = f () in
  let s1 = Rune.jit_stats () in
  ( r,
    s1.bytes_to_device - s0.bytes_to_device,
    s1.bytes_from_device - s0.bytes_from_device )

let test_feedback_moves_no_bytes () =
  require_cuda ();
  let f x = Nx.add_s (Nx.mul_s x 2.0) 1.0 in
  let g = Rune.jit' ~device:"CUDA" f in
  let x = vec32 [| 1.0; 2.0; 3.0 |] in
  let h1 = g x in
  let h2, up2, down2 = delta (fun () -> g h1) in
  let h3, up3, down3 = delta (fun () -> g h2) in
  equal ~msg:"feeding h1 back uploads nothing" int 0 up2;
  equal ~msg:"producing h2 downloads nothing" int 0 down2;
  equal ~msg:"feeding h2 back uploads nothing" int 0 up3;
  equal ~msg:"producing h3 downloads nothing" int 0 down3;
  check_arr ~msg:"h3 matches the eager composition" (to_arr (f (f (f x)))) h3;
  check_arr ~msg:"h1 still readable" (to_arr (f x)) h1;
  check_arr ~msg:"h2 still readable" (to_arr (f (f x))) h2

let test_forced_handle_feeds_current_bytes () =
  require_cuda ();
  let g = Rune.jit' ~device:"CUDA" (fun x -> Nx.mul_s x 2.0) in
  let h = g (vec32 [| 1.0; 2.0; 3.0 |]) in
  check_arr ~msg:"reading forces the handle" [| 2.0; 4.0; 6.0 |] h;
  Nx.set_item [ 0 ] 10.0 h;
  let h2, up, _ = delta (fun () -> g h) in
  is_true ~msg:"a forced handle re-uploads" (up > 0);
  check_arr ~msg:"the mutation is observed" [| 20.0; 8.0; 12.0 |] h2

let test_cuda_handle_into_cpu_jit () =
  require_cuda ();
  let gc = Rune.jit' ~device:"CUDA" (fun x -> Nx.mul_s x 2.0) in
  let gp = Rune.jit' (fun x -> Nx.add_s x 1.0) in
  let h = gc (vec32 [| 1.0; 2.0 |]) in
  (* A handle from another device takes the ordinary host path: it forces and
     copies. *)
  check_arr ~msg:"cuda handle read on the cpu device" [| 3.0; 5.0 |] (gp h)

let test_assign_to_resident_leaf () =
  require_cuda ();
  let producer = Rune.jit' ~device:"CUDA" (fun x -> Nx.mul_s x 2.0) in
  let h = producer (vec32 [| 1.0; 2.0 |]) in
  let step =
    Rune.jit' ~device:"CUDA" (fun x ->
        Nx.blit (Nx.mul_s x 2.0) x;
        Nx.sum x)
  in
  let s = step h in
  check_arr ~msg:"sum of the updated leaf" [| 12.0 |] s;
  check_arr ~msg:"the writeback forced h and updated it" [| 4.0; 8.0 |] h

(* Multi-kernel compiled traces replay as batched device execution graphs: the
   kernels are recorded into a CUDA graph on the first call and later calls
   patch the rebound buffers (fresh outputs, resident inputs) into the
   recorded graph instead of launching each kernel individually. *)
let test_graph_batched_replay () =
  require_cuda ();
  let w1 =
    Nx.create f32 [| 4; 4 |]
      (Array.init 16 (fun i -> (float_of_int (i mod 5) /. 4.0) -. 0.5))
  in
  let w2 =
    Nx.create f32 [| 4; 4 |]
      (Array.init 16 (fun i -> float_of_int (i mod 3) -. 1.0))
  in
  (* Two chained matmuls: at least two kernels, so the batch rewrite emits a
     graph call. *)
  let f x = Nx.matmul (Nx.tanh (Nx.matmul x w1)) w2 in
  let g = Rune.jit' ~device:"CUDA" f in
  let launches0 = !Tolk.Realize.graph_launches in
  List.iteri
    (fun i data ->
      let x = Nx.create f32 [| 2; 4 |] data in
      check_arr
        ~msg:(Printf.sprintf "call %d matches eager" (i + 1))
        (to_arr (f x)) (g x))
    [
      Array.init 8 (fun i -> float_of_int i /. 8.0);
      Array.init 8 (fun i -> float_of_int (7 - i));
      Array.make 8 (-0.25);
    ];
  is_true ~msg:"every call dispatched a device execution graph"
    (!Tolk.Realize.graph_launches - launches0 >= 3)

let test_capture_uploaded_once_across_signatures () =
  require_cuda ();
  let n = 256 in
  let c = vec32 (Array.init n float_of_int) in
  let g = Rune.jit' ~device:"CUDA" (fun x -> Nx.add x c) in
  let _, up1, _ = delta (fun () -> g (vec32 (Array.make n 0.0))) in
  let _, up2, _ =
    delta (fun () -> g (Nx.create f32 [| 1; n |] (Array.make n 1.0)))
  in
  equal ~msg:"first compile uploads input and capture" int (2 * n * 4) up1;
  equal ~msg:"second signature re-uploads only the input" int (n * 4) up2

type pair = { u : Nx.float32_t; v : Nx.float32_t }

module Pair = struct
  type t = pair

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { u; v } =
    { u = f u; v = f v }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { u = f p.u q.u; v = f p.v q.v }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { u; v } =
    f u;
    f v
end

let test_pass_through_output_survives () =
  require_cuda ();
  let g =
    Rune.jit2 ~device:"CUDA"
      (module Pair)
      (module Pair)
      (fun p -> { u = p.u; v = Nx.mul_s p.v 2.0 })
  in
  let r1 = g { u = vec32 [| 1.0; 2.0 |]; v = vec32 [| 3.0; 4.0 |] } in
  let r2 = g { u = vec32 [| 5.0; 6.0 |]; v = vec32 [| 7.0; 8.0 |] } in
  check_arr ~msg:"pass-through survives a later call" [| 1.0; 2.0 |] r1.u;
  check_arr ~msg:"second call's pass-through" [| 5.0; 6.0 |] r2.u;
  check_arr ~msg:"second call's computed output" [| 14.0; 16.0 |] r2.v

(* pmap on a duplicated CUDA device tuple: both shards run on the one GPU, so
   the whole multi-device path (per-shard uploads, per-device launches with
   [_device_num] bound, allreduce, gather on read) is exercised without a
   second device. *)

module Single_f32 = struct
  type t = Nx.float32_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

let cuda2 = [ "CUDA:0"; "CUDA:0" ]

let test_pmap_matches_jit_on_cuda () =
  require_cuda ();
  let chain x =
    let y = Nx.tanh (Nx.add (Nx.mul x x) x) in
    let z = Nx.matmul y (Nx.transpose y) in
    Nx.sum z ~axes:[ 1 ]
  in
  let x =
    Nx.create f32 [| 4; 6 |] (Array.init 24 (fun i -> float_of_int i /. 7.0))
  in
  let expect = Rune.jit' ~device:"CUDA" chain x in
  let g = Rune.pmap ~devices:cuda2 (module Single_f32) chain in
  check_arr ~msg:"first call" (to_arr expect) (g x);
  check_arr ~msg:"replay" (to_arr expect) (g x)

let test_pmap_grad_allreduce_on_cuda () =
  require_cuda ();
  let w = Nx.create f32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let loss x = Nx.mean (Nx.matmul x w) in
  let grads x = Rune.grad' loss x in
  let x = Nx.create f32 [| 4; 3 |] (Array.init 12 (fun i -> float_of_int i)) in
  let expect = Rune.jit' ~device:"CUDA" grads x in
  let g = Rune.pmap ~devices:cuda2 (module Single_f32) grads in
  check_arr ~msg:"grad inside pmap on cuda" (to_arr expect) (g x)

let test_pmap_feedback_on_cuda () =
  require_cuda ();
  let g = Rune.pmap ~devices:cuda2 (module Single_f32) (fun x -> Nx.add x x) in
  let x = vec32 (Array.init 8 float_of_int) in
  let y1 = g x in
  let y2, up, _ = delta (fun () -> g y1) in
  equal ~msg:"feedback moves no bytes to device" int 0 up;
  check_arr ~msg:"gathered result"
    (Array.init 8 (fun i -> 4.0 *. float_of_int i))
    y2

let tests =
  [
    group "cuda device"
      [
        test "element-wise chain matches eager" test_elementwise_on_cuda;
        test "grad inside jit matches eager" test_matmul_grad_on_cuda;
        test "captures are uploaded once" test_capture_is_uploaded_once;
        test "assigning to a capture raises" test_assign_to_capture_raises;
        test "multi-kernel traces replay as batched graphs"
          test_graph_batched_replay;
      ];
    group "cuda residency"
      [
        test "feedback moves no bytes" test_feedback_moves_no_bytes;
        test "forced handles feed current bytes"
          test_forced_handle_feeds_current_bytes;
        test "cuda handles read on the cpu device"
          test_cuda_handle_into_cpu_jit;
        test "assigning to a resident leaf forces then writes back"
          test_assign_to_resident_leaf;
        test "pass-through outputs survive later calls"
          test_pass_through_output_survives;
        test "captures upload once across signatures"
          test_capture_uploaded_once_across_signatures;
      ];
    group "cuda pmap"
      [
        test "duplicated device tuple matches jit" test_pmap_matches_jit_on_cuda;
        test "grad inside pmap allreduces on one gpu"
          test_pmap_grad_allreduce_on_cuda;
        test "feedback moves no bytes" test_pmap_feedback_on_cuda;
      ];
  ]

let () = run "rune jit cuda" tests
