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

(* Half precision on the GPU: eager (nx C kernels) vs CUDA-jitted half
   graphs, and the astype-sandwich gradient under a CUDA jit. *)

let sin_data n = Array.init n (fun i -> sin (float_of_int (i + 1)))
let cos_data n = Array.init n (fun i -> 0.5 *. cos (float_of_int (i + 1)))
let half_mat dt r c f = Nx.cast dt (Nx.create f32 [| r; c |] (f (r * c)))

let check_half_on_cuda name ~eps f x =
  let g = Rune.jit' ~device:"CUDA" f in
  check_arr ~eps ~msg:(name ^ " first call") (to_arr (f x)) (g x);
  check_arr ~eps ~msg:(name ^ " replay") (to_arr (f x)) (g x)

let test_half_matmul_on_cuda (type b) name (dt : (float, b) Nx.dtype) ~eps () =
  require_cuda ();
  let b = half_mat dt 8 3 cos_data in
  check_half_on_cuda name ~eps
    (fun a -> Nx.matmul a b)
    (half_mat dt 4 8 sin_data)

let test_half_softmax_on_cuda (type b) name (dt : (float, b) Nx.dtype) ~eps ()
    =
  require_cuda ();
  check_half_on_cuda name ~eps
    (fun x ->
      let e = Nx.exp x in
      Nx.div e (Nx.sum e ~axes:[ 1 ] ~keepdims:true))
    (half_mat dt 3 5 sin_data)

(* fp32 params, half compute, fp32 loss: the gradient of the sandwich must
   come out fp32 and match the all-fp32 gradient within the half dtype's
   tolerance, under a CUDA jit. *)
let test_half_sandwich_grad_on_cuda (type b) name (dt : (float, b) Nx.dtype)
    ~tol () =
  require_cuda ();
  let x = Nx.create f32 [| 4; 3 |] (sin_data 12) in
  let w = Nx.create f32 [| 3; 2 |] (cos_data 6) in
  let sandwich w =
    let xh = Nx.cast dt x and wh = Nx.cast dt w in
    Nx.cast f32 (Nx.mean (Nx.tanh (Nx.matmul xh wh)))
  in
  let jitted = Rune.jit' ~device:"CUDA" (fun w -> Rune.grad' sandwich w) in
  check_arr ~eps:(tol /. 4.0)
    ~msg:(name ^ " cuda grad vs eager sandwich grad")
    (to_arr (Rune.grad' sandwich w))
    (jitted w);
  let reference w = Nx.mean (Nx.tanh (Nx.matmul x w)) in
  check_arr ~eps:tol
    ~msg:(name ^ " cuda grad vs fp32 reference")
    (to_arr (Rune.grad' reference w))
    (jitted w)

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

module Single_bf16 = struct
  type t = Nx.bfloat16_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

(* Reducing over the sharded axis allreduces at bfloat16: each shard's
   partial sum is rounded to bfloat16 before the combine, so allow a couple
   of ulps against the single-device result. *)
let test_pmap_bf16_allreduce_on_cuda () =
  require_cuda ();
  let f x = Nx.sum x ~axes:[ 0 ] in
  let x = half_mat Nx.bfloat16 4 6 sin_data in
  let expect = Rune.jit' ~device:"CUDA" f x in
  let g = Rune.pmap ~devices:cuda2 (module Single_bf16) f in
  check_arr ~eps:0.0625 ~msg:"bf16 allreduce vs single device" (to_arr expect)
    (g x);
  check_arr ~eps:0.0625 ~msg:"replay" (to_arr expect) (g x)

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

(* Donation on the GPU: a state-to-state loop with [donate:true] releases each
   consumed generation's device buffer once its call completes, so resident
   bytes stay bounded at two generations with every handle still reachable and
   no GC; the donated handles raise on read. *)

let raises_donated f =
  raises_match
    (fun exn ->
      match exn with
      | Invalid_argument msg ->
          msg
          = "Rune.jit: this tensor was donated to a jitted call; read or copy \
             it before the call"
      | _ -> false)
    (fun () -> ignore (f ()))

let test_donate_bounds_resident_memory_on_cuda () =
  require_cuda ();
  let n = 4096 in
  let x = vec32 (Array.make n 0.0) in
  let hold = Array.make 10 x in
  let run g =
    let base = (Rune.jit_stats ()).resident_bytes in
    let h = ref (g x) in
    for i = 0 to 9 do
      hold.(i) <- !h;
      h := g !h
    done;
    ((Rune.jit_stats ()).resident_bytes - base, !h)
  in
  let step d = Rune.jit' ~device:"CUDA" ~donate:d (fun x -> Nx.add_s x 1.0) in
  let grew, h = run (step true) in
  is_true ~msg:"donate holds at most two generations" (grew <= 2 * n * 4);
  check_arr ~msg:"donated chain computes the right value" (Array.make n 11.0) h;
  let grew', h' = run (step false) in
  is_true ~msg:"without donate every generation stays resident"
    (grew' >= 10 * n * 4);
  check_arr ~msg:"undonated chain still correct" (Array.make n 11.0) h'

let test_donated_handle_raises_on_cuda () =
  require_cuda ();
  let g = Rune.jit' ~device:"CUDA" ~donate:true (fun x -> Nx.mul_s x 2.0) in
  let h1 = g (vec32 [| 1.0; 2.0 |]) in
  let h2 = g h1 in
  raises_donated (fun () -> to_arr h1);
  raises_donated (fun () -> g h1);
  check_arr ~msg:"the consuming call's output is fine" [| 4.0; 8.0 |] h2

let test_forced_handle_unaffected_by_donate_on_cuda () =
  require_cuda ();
  let g = Rune.jit' ~device:"CUDA" ~donate:true (fun x -> Nx.mul_s x 2.0) in
  let h = g (vec32 [| 1.0; 2.0 |]) in
  check_arr ~msg:"read before the call forces to host" [| 2.0; 4.0 |] h;
  ignore (g h);
  check_arr ~msg:"host bytes survive the donating call" [| 2.0; 4.0 |] h

let test_pmap_donate_on_cuda () =
  require_cuda ();
  let g =
    Rune.pmap ~devices:cuda2 ~donate:true
      (module Single_f32)
      (fun x -> Nx.add x x)
  in
  let x = vec32 (Array.init 8 float_of_int) in
  let y1 = g x in
  let y2, up, _ = delta (fun () -> g y1) in
  equal ~msg:"donated feedback still moves no bytes to device" int 0 up;
  raises_donated (fun () -> to_arr y1);
  check_arr ~msg:"gathered result"
    (Array.init 8 (fun i -> 4.0 *. float_of_int i))
    y2

(* Rng on CUDA. The threefry samplers must be bit-identical to eager
   execution: the Tolk decomposition the GPU kernels compile from is the same
   Random123 function as the eager C kernel. *)

module Key = struct
  type t = Nx.Rng.key

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

let check_bits ~msg expected actual =
  let e = to_arr expected and a = to_arr actual in
  equal ~msg:(msg ^ " (length)") int (Array.length e) (Array.length a);
  is_true ~msg
    (Array.for_all2
       (fun x y -> Int32.bits_of_float x = Int32.bits_of_float y)
       e a)

let raises_jit_error f =
  raises_match
    (fun exn -> match exn with Rune.Jit_error _ -> true | _ -> false)
    (fun () -> ignore (f ()))

let test_rng_uniform_bit_parity_on_cuda () =
  require_cuda ();
  let f key = Nx.Rng.uniform key Nx.float32 [| 1000 |] in
  let k = Nx.Rng.key 42 in
  let g = Rune.jit ~device:"CUDA" (module Key) f in
  check_bits ~msg:"eager == cuda jit, bitwise" (f k) (g k);
  check_bits ~msg:"replay" (f k) (g k)

let test_rng_int_samplers_bit_parity_on_cuda () =
  require_cuda ();
  let k = Nx.Rng.key 9 in
  let fr key = Nx.cast f32 (Nx.Rng.randint key ~high:9 [| 64 |] 3) in
  check_bits ~msg:"randint" (fr k) (Rune.jit ~device:"CUDA" (module Key) fr k);
  let fb key = Nx.cast f32 (Nx.Rng.bernoulli key ~p:0.3 [| 64 |]) in
  check_bits ~msg:"bernoulli" (fb k)
    (Rune.jit ~device:"CUDA" (module Key) fb k)

(* The threefry bits agree exactly; Box-Muller's cos/log/sqrt land within
   float32 ulps of eager (GPU transcendental codegen). *)
let test_rng_normal_matches_eager_on_cuda () =
  require_cuda ();
  let f key = Nx.Rng.normal key Nx.float32 [| 1000 |] in
  let k = Nx.Rng.key 42 in
  check_arr ~msg:"normal" (to_arr (f k)) (Rune.jit ~device:"CUDA" (module Key) f k)

let test_rng_fold_in_driven_steps_on_cuda () =
  require_cuda ();
  let root = Nx.Rng.key 3 in
  let g =
    Rune.jit ~device:"CUDA"
      (module Key)
      (fun key -> Nx.Rng.uniform key Nx.float32 [| 8 |])
  in
  let outs = Array.init 5 (fun i -> to_arr (g (Nx.Rng.fold_in root i))) in
  for i = 0 to 4 do
    for j = i + 1 to 4 do
      is_true ~msg:(Printf.sprintf "steps %d and %d differ" i j)
        (outs.(i) <> outs.(j))
    done
  done;
  is_true ~msg:"reproducible from the root"
    (to_arr (g (Nx.Rng.fold_in root 3)) = outs.(3))

let test_rng_constant_key_raises_on_cuda () =
  require_cuda ();
  let g =
    Rune.jit' ~device:"CUDA" (fun x -> Nx.add x (Nx.rand Nx.float32 [| 3 |]))
  in
  raises_jit_error (fun () -> g (vec32 [| 1.0; 2.0; 3.0 |]))

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
    group "cuda half"
      [
        test "float16 matmul matches eager"
          (test_half_matmul_on_cuda "float16" Nx.float16 ~eps:0.01);
        test "bfloat16 matmul matches eager"
          (test_half_matmul_on_cuda "bfloat16" Nx.bfloat16 ~eps:0.07);
        test "float16 softmax matches eager"
          (test_half_softmax_on_cuda "float16" Nx.float16 ~eps:0.002);
        test "bfloat16 softmax matches eager"
          (test_half_softmax_on_cuda "bfloat16" Nx.bfloat16 ~eps:0.016);
        test "float16 sandwich grad is fp32"
          (test_half_sandwich_grad_on_cuda "float16" Nx.float16 ~tol:0.005);
        test "bfloat16 sandwich grad is fp32"
          (test_half_sandwich_grad_on_cuda "bfloat16" Nx.bfloat16 ~tol:0.02);
      ];
    group "cuda pmap"
      [
        test "duplicated device tuple matches jit" test_pmap_matches_jit_on_cuda;
        test "grad inside pmap allreduces on one gpu"
          test_pmap_grad_allreduce_on_cuda;
        test "bf16 allreduce matches single device"
          test_pmap_bf16_allreduce_on_cuda;
        test "feedback moves no bytes" test_pmap_feedback_on_cuda;
      ];
    group "cuda rng"
      [
        test "uniform is bit-identical to eager"
          test_rng_uniform_bit_parity_on_cuda;
        test "randint and bernoulli are bit-identical to eager"
          test_rng_int_samplers_bit_parity_on_cuda;
        test "normal matches eager" test_rng_normal_matches_eager_on_cuda;
        test "fold_in drives fresh values through a cuda step"
          test_rng_fold_in_driven_steps_on_cuda;
        test "a constant key raises" test_rng_constant_key_raises_on_cuda;
      ];
    group "cuda donation"
      [
        test "donate bounds resident memory at two generations"
          test_donate_bounds_resident_memory_on_cuda;
        test "a donated handle raises on read and re-feed"
          test_donated_handle_raises_on_cuda;
        test "a handle read before the call is unaffected"
          test_forced_handle_unaffected_by_donate_on_cuda;
        test "pmap donation consumes the sharded state"
          test_pmap_donate_on_cuda;
      ];
  ]

let () = run "rune jit cuda" tests
