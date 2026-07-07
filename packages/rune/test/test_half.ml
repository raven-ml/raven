(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Half-precision (float16 / bfloat16) through the transformations: the
   astype-sandwich gradient that underpins fp32-master-weight training,
   eager-vs-jit agreement for half compute graphs, pmap with bfloat16 leaves,
   and vmap over half tensors. *)

open Windtrap
open Rune_test_support.Support

let f16 = Nx.float16
let bf16 = Nx.bfloat16

(* Deterministic inputs in [-1, 1]. *)
let sin_data n = Array.init n (fun i -> sin (float_of_int (i + 1)))
let cos_data n = Array.init n (fun i -> 0.5 *. cos (float_of_int (i + 1)))
let half_mat dt r c f = Nx.cast dt (Nx.create f32 [| r; c |] (f (r * c)))

(* ───── The astype sandwich ─────

   fp32 params, cast to the half dtype, compute (matmul + nonlinearity +
   mean), cast the scalar loss back to fp32. The cast VJP casts the cotangent
   back to the input dtype, so the gradient must come out fp32 and close to
   the all-fp32 gradient. *)

let sandwich_loss (type b) (dt : (float, b) Nx.dtype) x w =
  let xh = Nx.cast dt x and wh = Nx.cast dt w in
  let y = Nx.tanh (Nx.matmul xh wh) in
  Nx.cast f32 (Nx.mean y)

let reference_loss x w = Nx.mean (Nx.tanh (Nx.matmul x w))

let sandwich_inputs () =
  let x = Nx.create f32 [| 4; 3 |] (sin_data 12) in
  let w = Nx.create f32 [| 3; 2 |] (cos_data 6) in
  (x, w)

let test_sandwich_grad (type b) name (dt : (float, b) Nx.dtype) ~tol () =
  let x, w = sandwich_inputs () in
  let v, g = Rune.value_and_grad' (sandwich_loss dt x) w in
  (* The gradient carries the parameter dtype, not the compute dtype. *)
  is_true
    ~msg:(name ^ " grad dtype is float32")
    (Nx_core.Dtype.equal_witness (Nx.dtype g) f32 <> None);
  let v_ref, g_ref = Rune.value_and_grad' (reference_loss x) w in
  check_arr ~eps:tol ~msg:(name ^ " loss vs fp32 reference") (to_arr v_ref) v;
  check_arr ~eps:tol ~msg:(name ^ " grad vs fp32 reference") (to_arr g_ref) g

let test_sandwich_grad_jit (type b) name (dt : (float, b) Nx.dtype) ~tol () =
  let x, w = sandwich_inputs () in
  let eager = Rune.grad' (sandwich_loss dt x) w in
  let jitted = Rune.jit' (fun w -> Rune.grad' (sandwich_loss dt x) w) in
  (* Same graph, so jit may differ from eager only by rounding. *)
  check_arr ~eps:(tol /. 4.0)
    ~msg:(name ^ " jit grad vs eager grad")
    (to_arr eager) (jitted w);
  let g_ref = Rune.grad' (reference_loss x) w in
  check_arr ~eps:tol
    ~msg:(name ^ " jit grad vs fp32 reference")
    (to_arr g_ref) (jitted w)

(* ───── Eager vs jit for half compute graphs ───── *)

let check_eager_vs_jit ~eps ~msg f x =
  let g = Rune.jit' f in
  check_arr ~eps ~msg:(msg ^ " first call") (to_arr (f x)) (g x);
  check_arr ~eps ~msg:(msg ^ " replay") (to_arr (f x)) (g x)

let test_jit_matmul (type b) name (dt : (float, b) Nx.dtype) ~eps () =
  let b = half_mat dt 8 3 cos_data in
  let f a = Nx.matmul a b in
  check_eager_vs_jit ~eps ~msg:(name ^ " matmul") f (half_mat dt 4 8 sin_data)

let softmax_graph x =
  let e = Nx.exp x in
  Nx.div e (Nx.sum e ~axes:[ 1 ] ~keepdims:true)

let test_jit_softmax (type b) name (dt : (float, b) Nx.dtype) ~eps () =
  check_eager_vs_jit ~eps
    ~msg:(name ^ " softmax")
    softmax_graph (half_mat dt 3 5 sin_data)

let layernorm_graph (type b) (dt : (float, b) Nx.dtype) x =
  let mu = Nx.mean x ~axes:[ 1 ] ~keepdims:true in
  let d = Nx.sub x mu in
  let v = Nx.mean (Nx.mul d d) ~axes:[ 1 ] ~keepdims:true in
  Nx.mul d (Nx.rsqrt (Nx.add v (Nx.scalar dt 1e-3)))

let test_jit_layernorm (type b) name (dt : (float, b) Nx.dtype) ~eps () =
  check_eager_vs_jit ~eps
    ~msg:(name ^ " layernorm")
    (layernorm_graph dt) (half_mat dt 3 6 sin_data)

(* ───── pmap with bfloat16 leaves ───── *)

module Single_bf16 = struct
  type t = Nx.bfloat16_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

let devs2 = [ "CPU:1"; "CPU:2" ]

(* Reducing over the sharded axis forces a cross-device allreduce at
   bfloat16. Each device rounds its partial sum to bfloat16 before the
   combine, so allow a couple of ulps against the single-device result. *)
let test_pmap_bf16_allreduce () =
  let f x = Nx.sum x ~axes:[ 0 ] in
  let x = half_mat bf16 4 6 sin_data in
  let expect = Rune.jit' f x in
  let g = Rune.pmap ~devices:devs2 (module Single_bf16) f in
  check_arr ~eps:0.0625 ~msg:"bf16 allreduce vs single device" (to_arr expect)
    (g x);
  check_arr ~eps:0.0625 ~msg:"replay" (to_arr expect) (g x)

(* No cross-device reduce: shards are independent, so the pmap result is
   bit-equal to the single-device one. *)
let test_pmap_bf16_elementwise () =
  let f x = Nx.mul x x in
  let x = half_mat bf16 4 6 sin_data in
  let expect = Rune.jit' f x in
  let g = Rune.pmap ~devices:devs2 (module Single_bf16) f in
  check_arr ~eps:0.0 ~msg:"bf16 elementwise byte-equal" (to_arr expect) (g x)

let test_pmap_bf16_mean_grad () =
  let loss x = Nx.mean (Nx.mul x x) in
  let grads x = Rune.grad' loss x in
  let x = half_mat bf16 4 6 sin_data in
  let expect = Rune.jit' grads x in
  let g = Rune.pmap ~devices:devs2 (module Single_bf16) grads in
  check_arr ~eps:0.0625 ~msg:"bf16 grad allreduce" (to_arr expect) (g x)

(* ───── vmap over half tensors ───── *)

let test_vmap_half (type b) name (dt : (float, b) Nx.dtype) () =
  let x = Nx.create dt [| 2; 3 |] [| 1.0; -2.0; 0.5; 3.0; -0.25; 1.5 |] in
  let y = Rune.vmap' (fun row -> Nx.mul row row) x in
  check_arr ~eps:0.0
    ~msg:(name ^ " vmap square")
    [| 1.0; 4.0; 0.25; 9.0; 0.0625; 2.25 |]
    y;
  let z = Rune.vmap' (fun row -> Nx.sum row) x in
  check_arr ~eps:0.0 ~msg:(name ^ " vmap sum") [| -0.5; 4.25 |] z

let tests =
  [
    group "astype sandwich"
      [
        test "bfloat16 grad is fp32 and matches the fp32 reference"
          (test_sandwich_grad "bfloat16" bf16 ~tol:0.02);
        test "float16 grad is fp32 and matches the fp32 reference"
          (test_sandwich_grad "float16" f16 ~tol:0.005);
        test "bfloat16 sandwich grad under jit"
          (test_sandwich_grad_jit "bfloat16" bf16 ~tol:0.02);
        test "float16 sandwich grad under jit"
          (test_sandwich_grad_jit "float16" f16 ~tol:0.005);
      ];
    group "eager vs jit"
      [
        test "float16 matmul" (test_jit_matmul "float16" f16 ~eps:0.01);
        test "bfloat16 matmul" (test_jit_matmul "bfloat16" bf16 ~eps:0.07);
        test "float16 softmax" (test_jit_softmax "float16" f16 ~eps:0.002);
        test "bfloat16 softmax" (test_jit_softmax "bfloat16" bf16 ~eps:0.016);
        test "float16 layernorm"
          (test_jit_layernorm "float16" f16 ~eps:0.008);
        test "bfloat16 layernorm"
          (test_jit_layernorm "bfloat16" bf16 ~eps:0.06);
      ];
    group "pmap"
      [
        test "bf16 allreduce matches single device" test_pmap_bf16_allreduce;
        test "bf16 elementwise is byte-equal" test_pmap_bf16_elementwise;
        test "bf16 grad allreduces" test_pmap_bf16_mean_grad;
      ];
    group "vmap"
      [
        test "float16 smoke" (test_vmap_half "float16" f16);
        test "bfloat16 smoke" (test_vmap_half "bfloat16" bf16);
      ];
  ]

let () = run "rune half precision" tests
