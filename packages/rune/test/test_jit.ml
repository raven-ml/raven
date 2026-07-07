(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Just-in-time compilation: trace/compile/replay correctness, signature
   retracing, composition with the other transformations, in-place state, and
   trace-time failure modes. *)

open Windtrap
open Rune_test_support.Support

let raises_jit_error f =
  raises_match
    (fun exn -> match exn with Rune.Jit_error _ -> true | _ -> false)
    (fun () -> ignore (f ()))

(* loss p = sum (w * w) + 3 * sum b. d/dw = 2w, d/db = 3, d/dscale = 0. *)
let quadratic p = Nx.add (Nx.sum (Nx.mul p.w p.w)) (Nx.mul_s (Nx.sum p.b) 3.0)

(* Basics *)

let test_elementwise_matches_eager () =
  let f x = Nx.tanh (Nx.add (Nx.mul x x) x) in
  let g = Rune.jit' f in
  let x = vec32 [| 1.0; -2.0; 0.5 |] in
  check_arr ~msg:"first call" (to_arr (f x)) (g x);
  check_arr ~msg:"replay" (to_arr (f x)) (g x)

let test_replay_reads_fresh_inputs () =
  let g = Rune.jit' (fun x -> Nx.mul x x) in
  ignore (g (vec32 [| 1.0; 2.0; 3.0 |]));
  check_arr ~msg:"fresh data" [| 4.0; 9.0; 16.0 |]
    (g (vec32 [| 2.0; 3.0; 4.0 |]))

let test_retrace_on_new_shape () =
  let g = Rune.jit' (fun x -> Nx.sum x) in
  check_arr ~msg:"vector" [| 6.0 |] (g (vec32 [| 1.0; 2.0; 3.0 |]));
  check_arr ~msg:"matrix" [| 10.0 |]
    (g (Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]))

let test_closure_matmul () =
  let w = Nx.create f32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let f x = Nx.matmul x w in
  let g = Rune.jit' f in
  let x = Nx.create f32 [| 2; 3 |] [| 1.0; 0.0; -1.0; 0.5; 2.0; 1.0 |] in
  check_arr ~msg:"matmul" (to_arr (f x)) (g x)

let test_jit2_structured_output () =
  let f p =
    { w = Nx.mul p.w p.w; b = Nx.add p.b p.b; scale = Nx.mul_s p.scale 2.0 }
  in
  let g = Rune.jit2 (module Params) (module Params) f in
  let p = params () in
  let r = g p in
  let e = f p in
  check_arr ~msg:"w" (to_arr e.w) r.w;
  check_arr ~msg:"b" (to_arr e.b) r.b;
  check_arr ~msg:"scale (float64)" (to_arr e.scale) r.scale

(* Composition *)

let test_grad_inside_jit () =
  let step =
    Rune.jit2
      (module Params)
      (module Params)
      (fun p -> Rune.grad (module Params) quadratic p)
  in
  let g = step (params ()) in
  check_arr ~msg:"dw" [| 2.0; -4.0; 6.0 |] g.w;
  check_arr ~msg:"db" [| 3.0 |] g.b;
  check_arr ~msg:"dscale" [| 0.0 |] g.scale;
  (* Replay computes gradients at the new point. *)
  let p2 = { (params ()) with w = vec32 [| 4.0; 5.0; 6.0 |] } in
  let g2 = step p2 in
  check_arr ~msg:"dw at new point" [| 8.0; 10.0; 12.0 |] g2.w

let test_jit_under_grad_is_transparent () =
  let g = Rune.jit' (fun x -> Nx.mul x x) in
  let dx = Rune.grad' (fun x -> Nx.sum (g x)) (vec32 [| 1.0; 2.0; 3.0 |]) in
  check_arr ~msg:"d(sum x^2)" [| 2.0; 4.0; 6.0 |] dx

let test_jit_under_vmap_is_transparent () =
  let g = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
  let y = Rune.vmap' g (Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]) in
  check_arr ~msg:"vmap over jit" [| 2.0; 4.0; 6.0; 8.0 |] y

let test_scan_unrolls_inside_jit () =
  let module C = struct
    type t = Nx.float32_t

    let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

    let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
      f a b

    let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
  end in
  let cumsum xs =
    snd
      (Rune.scan
         (module C)
         ~f:(fun c x ->
           let c = Nx.add c x in
           (c, c))
         ~init:(Nx.scalar f32 0.0) xs)
  in
  let g = Rune.jit' cumsum in
  check_arr ~msg:"cumulative sum" [| 1.0; 3.0; 6.0 |]
    (g (vec32 [| 1.0; 2.0; 3.0 |]))

(* In-place state *)

let test_assign_writes_back_to_leaf () =
  let step =
    Rune.jit
      (module Params)
      (fun p ->
        Nx.blit (Nx.mul_s p.w 2.0) p.w;
        Nx.sum p.w)
  in
  let p = params () in
  let s = step p in
  check_arr ~msg:"sum of updated w" [| 4.0 |] s;
  check_arr ~msg:"w updated in place" [| 2.0; -4.0; 6.0 |] p.w;
  ignore (step p);
  check_arr ~msg:"second step compounds" [| 4.0; -8.0; 12.0 |] p.w

(* Buffer sharing: strided leaves must fall back to copies, views with an offset
   must read the right span, and each call must return tensors with their own
   storage. *)

let test_non_contiguous_input_matches_eager () =
  let f x = Nx.add (Nx.mul x x) x in
  let g = Rune.jit' f in
  let x =
    Nx.transpose (Nx.create f32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |])
  in
  check_arr ~msg:"transposed input" (to_arr (f x)) (g x)

let test_offset_view_input_matches_eager () =
  let f x = Nx.mul_s x 3.0 in
  let g = Rune.jit' f in
  let x =
    Nx.get [ 1 ] (Nx.create f32 [| 3; 4 |] (Array.init 12 float_of_int))
  in
  check_arr ~msg:"offset row view" (to_arr (f x)) (g x)

let test_outputs_have_their_own_storage () =
  let g = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
  let y1 = g (vec32 [| 1.0; 2.0; 3.0 |]) in
  let _y2 = g (vec32 [| 10.0; 20.0; 30.0 |]) in
  check_arr ~msg:"first result unchanged by the second call" [| 2.0; 4.0; 6.0 |]
    y1

(* Mutating a capture between calls has unspecified visibility. This pins the
   CPU device's zero-copy binding, which happens to observe the mutation
   because contiguous captures alias the tensor's memory; other devices keep
   the compile-time value. Not a supported pattern — thread changing values
   as input leaves. *)
let test_cpu_aliasing_observes_closure_mutation () =
  let c = vec32 [| 10.0; 20.0; 30.0 |] in
  let g = Rune.jit' (fun x -> Nx.add x c) in
  check_arr ~msg:"initial capture" [| 11.0; 21.0; 31.0 |]
    (g (vec32 [| 1.0; 1.0; 1.0 |]));
  Nx.blit (vec32 [| 0.0; 0.0; 0.0 |]) c;
  check_arr ~msg:"mutated capture is read through the alias" [| 1.0; 1.0; 1.0 |]
    (g (vec32 [| 1.0; 1.0; 1.0 |]))

(* Captures are compile-time constants: a function that assigns to one fails
   at trace time, on every device. *)
let test_assign_to_capture_raises () =
  let s = vec32 [| 1.0; 2.0 |] in
  let g =
    Rune.jit' (fun x ->
        Nx.blit (Nx.add s x) s;
        Nx.mul_s s 10.0)
  in
  raises_jit_error (fun () -> g (vec32 [| 1.0; 1.0 |]))

(* Sliding windows *)

(* An asymmetric configuration so any axis-ordering mistake shows up: distinct
   kernel, stride, dilation, and padding per spatial dimension. *)
let window_config =
  ( [| 2; 3 |] (* kernel *),
    [| 2; 1 |] (* stride *),
    [| 1; 2 |] (* dilation *),
    [| (1, 0); (2, 1) |] (* padding *) )

let window_input () =
  Nx.create f32 [| 2; 3; 5; 6 |]
    (Array.init (2 * 3 * 5 * 6) (fun i -> float_of_int (i mod 17) -. 8.0))

let test_unfold_matches_eager () =
  let kernel_size, stride, dilation, padding = window_config in
  let f x = Nx.extract_patches ~kernel_size ~stride ~dilation ~padding x in
  let g = Rune.jit' f in
  let x = window_input () in
  equal ~msg:"shape" (array int) (Nx.shape (f x)) (Nx.shape (g x));
  check_arr ~msg:"unfold" (to_arr (f x)) (g x)

let test_fold_matches_eager () =
  let kernel_size, stride, dilation, padding = window_config in
  let output_size = [| 5; 6 |] in
  let f x =
    Nx.combine_patches ~output_size ~kernel_size ~stride ~dilation ~padding
      (Nx.extract_patches ~kernel_size ~stride ~dilation ~padding x)
  in
  let g = Rune.jit' f in
  let x = window_input () in
  check_arr ~msg:"fold of unfold" (to_arr (f x)) (g x)

let test_correlate_matches_eager () =
  let kernel =
    Nx.create f32 [| 3; 3 |]
      [| 1.0; 0.0; -1.0; 2.0; 0.5; -2.0; 1.0; 0.0; -1.0 |]
  in
  let f x = Nx.correlate ~padding:`Same x kernel in
  let g = Rune.jit' f in
  let x = Nx.create f32 [| 6; 7 |] (Array.init 42 (fun i -> float_of_int i)) in
  check_arr ~msg:"correlate same" (to_arr (f x)) (g x)

(* Indexed access *)

(* Row 1 repeats an index so duplicate handling is pinned under jit: [`Set]
   keeps the last update, [`Add] accumulates both on top of [x]'s value. *)
let test_scatter_matches_eager () =
  let idx = Nx.create Nx.int32 [| 2; 2 |] [| 2l; 0l; 1l; 1l |] in
  let f mode x =
    Nx.scatter ~mode ~axis:1 ~indices:idx
      ~values:(Nx.slice [ Nx.A; Nx.R (0, 2) ] x)
      x
  in
  let x = Nx.create f32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let g_set = Rune.jit' (f `Set) and g_add = Rune.jit' (f `Add) in
  check_arr ~msg:"set" (to_arr (f `Set x)) (g_set x);
  check_arr ~msg:"set replay" (to_arr (f `Set x)) (g_set x);
  check_arr ~msg:"add" (to_arr (f `Add x)) (g_add x)

(* Training-step integration: a two-layer MLP trained by a jitted step must
   follow the eager trajectory exactly. *)

type mlp = { w1 : Nx.float32_t; b1 : Nx.float32_t; w2 : Nx.float32_t }

module Mlp = struct
  type t = mlp

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w1; b1; w2 } =
    { w1 = f w1; b1 = f b1; w2 = f w2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { w1 = f p.w1 q.w1; b1 = f p.b1 q.b1; w2 = f p.w2 q.w2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w1; b1; w2 } =
    f w1;
    f b1;
    f w2
end

let test_jitted_training_matches_eager () =
  let xs =
    Nx.create f32 [| 8; 4 |]
      (Array.init 32 (fun i -> float_of_int (i mod 7) /. 7.0))
  in
  let ys =
    Nx.create f32 [| 8; 1 |]
      (Array.init 8 (fun i -> float_of_int (i mod 3) -. 1.0))
  in
  let init () =
    {
      w1 =
        Nx.create f32 [| 4; 5 |]
          (Array.init 20 (fun i -> (0.1 *. float_of_int (i mod 5)) -. 0.2));
      b1 = Nx.zeros f32 [| 5 |];
      w2 =
        Nx.create f32 [| 5; 1 |]
          (Array.init 5 (fun i -> 0.3 -. (0.1 *. float_of_int i)));
    }
  in
  let loss p =
    let h = Nx.tanh (Nx.add (Nx.matmul xs p.w1) p.b1) in
    let d = Nx.sub (Nx.matmul h p.w2) ys in
    Nx.mean (Nx.mul d d)
  in
  let update p =
    let g = Rune.grad (module Mlp) loss p in
    Mlp.map2 (fun w dw -> Nx.sub w (Nx.mul (scalar_like dw 0.1) dw)) p g
  in
  let step = Rune.jit2 (module Mlp) (module Mlp) update in
  let rec train f p n = if n = 0 then p else train f (f p) (n - 1) in
  let jitted = train step (init ()) 5 in
  let eager = train update (init ()) 5 in
  check_arr ~msg:"w1" (to_arr eager.w1) jitted.w1;
  check_arr ~msg:"b1" (to_arr eager.b1) jitted.b1;
  check_arr ~msg:"w2" (to_arr eager.w2) jitted.w2;
  let l0 = scalar (loss (init ())) and l5 = scalar (loss jitted) in
  is_true ~msg:"loss decreased" (l5 < l0)

(* Device residency. Under RUNE_JIT_FORCE_COPY=1 the CPU device takes the
   staged-copy path used by CUDA and Metal: outputs become deferred handles
   that stay on the device until read, and a handle fed back into a compiled
   call seeds its input buffer directly. The transfer counters make the
   no-copy claims observable. The knob is read when the jit closure is
   created, so it is scoped to each test body. *)

let with_force_copy f =
  Unix.putenv "RUNE_JIT_FORCE_COPY" "1";
  Fun.protect f ~finally:(fun () -> Unix.putenv "RUNE_JIT_FORCE_COPY" "0")

(* Run [f] and return its result with the bytes moved to and from the device
   during the run. *)
let delta f =
  let s0 = Rune.jit_stats () in
  let r = f () in
  let s1 = Rune.jit_stats () in
  ( r,
    s1.bytes_to_device - s0.bytes_to_device,
    s1.bytes_from_device - s0.bytes_from_device )

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

let test_feedback_chain_moves_no_bytes () =
  with_force_copy (fun () ->
      let f x = Nx.add_s (Nx.mul_s x 2.0) 1.0 in
      let g = Rune.jit' f in
      let x = vec32 [| 1.0; 2.0; 3.0 |] in
      let h1 = g x in
      let h2, up2, down2 = delta (fun () -> g h1) in
      let h3, up3, down3 = delta (fun () -> g h2) in
      equal ~msg:"feeding h1 back uploads nothing" int 0 up2;
      equal ~msg:"producing h2 downloads nothing" int 0 down2;
      equal ~msg:"feeding h2 back uploads nothing" int 0 up3;
      equal ~msg:"producing h3 downloads nothing" int 0 down3;
      check_arr ~msg:"h3 matches the eager composition" (to_arr (f (f (f x))))
        h3;
      (* Handles from earlier calls keep their own storage (R3). *)
      check_arr ~msg:"h1 still readable" (to_arr (f x)) h1;
      check_arr ~msg:"h2 still readable" (to_arr (f (f x))) h2)

let test_forced_handle_feeds_current_bytes () =
  with_force_copy (fun () ->
      let g = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
      let h = g (vec32 [| 1.0; 2.0; 3.0 |]) in
      check_arr ~msg:"reading forces the handle" [| 2.0; 4.0; 6.0 |] h;
      (* Once forced it is a plain host tensor: mutations are honored and
         feeding it back re-uploads the current bytes. *)
      Nx.set_item [ 0 ] 10.0 h;
      let h2, up, _ = delta (fun () -> g h) in
      is_true ~msg:"a forced handle re-uploads" (up > 0);
      check_arr ~msg:"the mutation is observed" [| 20.0; 8.0; 12.0 |] h2)

let test_same_handle_as_two_leaves () =
  with_force_copy (fun () ->
      let g =
        Rune.jit2
          (module Pair)
          (module Pair)
          (fun p -> { u = Nx.add p.u p.v; v = Nx.mul p.u p.v })
      in
      let h = Rune.jit' (fun x -> Nx.mul_s x 3.0) (vec32 [| 1.0; 2.0 |]) in
      let r, up, _ = delta (fun () -> g { u = h; v = h }) in
      equal ~msg:"resident duplicate leaves upload nothing" int 0 up;
      check_arr ~msg:"u" [| 6.0; 12.0 |] r.u;
      check_arr ~msg:"v" [| 9.0; 36.0 |] r.v)

let test_duplicate_outputs_share_one_handle () =
  with_force_copy (fun () ->
      let g =
        Rune.jit2
          (module Pair)
          (module Pair)
          (fun p ->
            let y = Nx.add p.u p.v in
            { u = y; v = y })
      in
      let r = g { u = vec32 [| 1.0 |]; v = vec32 [| 2.0 |] } in
      is_true ~msg:"both leaves are one handle" (r.u == r.v);
      check_arr ~msg:"readable" [| 3.0 |] r.u;
      check_arr ~msg:"readable through the other leaf" [| 3.0 |] r.v)

let test_cross_jit_feedback () =
  with_force_copy (fun () ->
      let g1 = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
      let g2 = Rune.jit' (fun x -> Nx.add_s x 1.0) in
      let h = g1 (vec32 [| 1.0; 2.0 |]) in
      let r, up, _ = delta (fun () -> g2 h) in
      equal ~msg:"a distinct jitted closure seeds the handle too" int 0 up;
      check_arr ~msg:"value" [| 3.0; 5.0 |] r)

let test_cross_signature_feedback () =
  with_force_copy (fun () ->
      let g = Rune.jit' (fun x -> Nx.sum ~axes:[ 0 ] x) in
      let h1 = g (Nx.create f32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]) in
      (* h1 has a new shape: feeding it back compiles a second signature,
         still without forcing the handle. *)
      let h2, up, down = delta (fun () -> g h1) in
      equal ~msg:"the retrace uploads nothing" int 0 up;
      equal ~msg:"the retrace downloads nothing" int 0 down;
      check_arr ~msg:"value" [| 21.0 |] h2;
      check_arr ~msg:"h1 still readable" [| 5.0; 7.0; 9.0 |] h1)

let test_pass_through_output_survives () =
  with_force_copy (fun () ->
      let g =
        Rune.jit2
          (module Pair)
          (module Pair)
          (fun p -> { u = p.u; v = Nx.mul_s p.v 2.0 })
      in
      let r1 = g { u = vec32 [| 1.0; 2.0 |]; v = vec32 [| 3.0; 4.0 |] } in
      let r2 = g { u = vec32 [| 5.0; 6.0 |]; v = vec32 [| 7.0; 8.0 |] } in
      check_arr ~msg:"pass-through survives a later call" [| 1.0; 2.0 |] r1.u;
      check_arr ~msg:"first call's computed output" [| 6.0; 8.0 |] r1.v;
      check_arr ~msg:"second call's pass-through" [| 5.0; 6.0 |] r2.u;
      check_arr ~msg:"second call's computed output" [| 14.0; 16.0 |] r2.v)

let test_assign_to_resident_leaf () =
  with_force_copy (fun () ->
      let producer = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
      let h = producer (vec32 [| 1.0; 2.0 |]) in
      let step =
        Rune.jit' (fun x ->
            Nx.blit (Nx.mul_s x 2.0) x;
            Nx.sum x)
      in
      let s = step h in
      check_arr ~msg:"sum of the updated leaf" [| 12.0 |] s;
      check_arr ~msg:"the writeback forced h and updated it" [| 4.0; 8.0 |] h)

let test_grad_over_jit_with_deferred_arg () =
  with_force_copy (fun () ->
      let g = Rune.jit' (fun x -> Nx.mul x x) in
      let h = g (vec32 [| 1.0; 2.0; 3.0 |]) in
      (* Under grad the jitted function runs eagerly; the handle forces on its
         first operation. *)
      let dx = Rune.grad' (fun x -> Nx.sum (g x)) h in
      check_arr ~msg:"gradient at the deferred point" [| 2.0; 8.0; 18.0 |] dx)

let test_vmap_over_jit_with_deferred_arg () =
  with_force_copy (fun () ->
      let g = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
      let h = g (Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]) in
      let y = Rune.vmap' g h in
      check_arr ~msg:"vmap over jit at a deferred point"
        [| 4.0; 8.0; 12.0; 16.0 |]
        y)

let test_dispatch_on_handle_reads_no_bytes () =
  with_force_copy (fun () ->
      let g = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
      let h = g (vec32 [| 1.0; 2.0 |]) in
      (* Signature dispatch uses only metadata: replaying on a handle must not
         force it. *)
      let _h2, _, down = delta (fun () -> g h) in
      equal ~msg:"dispatching on a handle downloads nothing" int 0 down)

let test_capture_uploaded_once_across_signatures () =
  with_force_copy (fun () ->
      let n = 256 in
      let c = vec32 (Array.init n float_of_int) in
      let g = Rune.jit' (fun x -> Nx.add x c) in
      let _, up1, _ = delta (fun () -> g (vec32 (Array.make n 0.0))) in
      (* A new input shape compiles a second signature; the capture's device
         copy is shared, so only the input is uploaded. *)
      let _, up2, _ =
        delta (fun () -> g (Nx.create f32 [| 1; n |] (Array.make n 1.0)))
      in
      equal ~msg:"first compile uploads input and capture" int (2 * n * 4) up1;
      equal ~msg:"second signature re-uploads only the input" int (n * 4) up2)

let test_dropped_handles_are_reclaimed () =
  with_force_copy (fun () ->
      let n = 1024 in
      let g = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
      let x = vec32 (Array.make n 1.0) in
      let base = (Rune.jit_stats ()).resident_bytes in
      let _, _, down =
        delta (fun () ->
            for _ = 1 to 50 do
              ignore (g x)
            done)
      in
      equal ~msg:"unread outputs download nothing" int 0 down;
      (* Collect the dropped handles; the next calls drain their buffers. *)
      Gc.full_major ();
      ignore (g x);
      Gc.full_major ();
      ignore (g x);
      let s = Rune.jit_stats () in
      is_true ~msg:"resident bytes are bounded after gc"
        (s.resident_bytes - base <= 3 * n * 4))

(* Donation. [donate:true] consumes resident input handles: their device
   buffers return to the allocator once the call completes, so a
   state-to-state loop holds ~2 generations of device memory instead of one
   per call, without any GC. A donated handle raises on read; host tensors,
   already-read handles, and written-back leaves are unaffected. *)

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

let test_donate_bounds_resident_memory () =
  with_force_copy (fun () ->
      let n = 4096 in
      let step d = Rune.jit' ~donate:d (fun x -> Nx.add_s x 1.0) in
      let x = vec32 (Array.make n 0.0) in
      (* Every handle stays reachable, so nothing here depends on the GC. *)
      let hold = Array.make 10 x in
      let run g =
        let base = (Rune.jit_stats ()).resident_bytes in
        let h = ref (g x) in
        for i = 0 to 9 do
          hold.(i) <- !h;
          h := g !h
        done;
        let r = (Rune.jit_stats ()).resident_bytes - base in
        (!h, r)
      in
      let h, grew = run (step true) in
      is_true ~msg:"donate holds at most two generations" (grew <= 2 * n * 4);
      check_arr ~msg:"donated chain computes the right value"
        (Array.make n 11.0) h;
      let h', grew' = run (step false) in
      is_true ~msg:"without donate every generation stays resident"
        (grew' >= 10 * n * 4);
      check_arr ~msg:"undonated chain still correct" (Array.make n 11.0) h')

let test_donated_handle_raises_on_read () =
  with_force_copy (fun () ->
      let g = Rune.jit' ~donate:true (fun x -> Nx.mul_s x 2.0) in
      let h1 = g (vec32 [| 1.0; 2.0 |]) in
      let h2 = g h1 in
      (* h1 was donated to the second call: its storage is gone. *)
      raises_donated (fun () -> to_arr h1);
      check_arr ~msg:"the consuming call's output is fine" [| 4.0; 8.0 |] h2)

let test_donated_handle_refeed_raises () =
  with_force_copy (fun () ->
      let g = Rune.jit' ~donate:true (fun x -> Nx.mul_s x 2.0) in
      let h1 = g (vec32 [| 1.0; 2.0 |]) in
      ignore (g h1);
      (* Seeding a donated handle forces it, which raises the same error. *)
      raises_donated (fun () -> g h1))

let test_donate_duplicate_leaves_once () =
  with_force_copy (fun () ->
      let g =
        Rune.jit2 ~donate:true
          (module Pair)
          (module Pair)
          (fun p -> { u = Nx.add p.u p.v; v = Nx.mul p.u p.v })
      in
      let h = Rune.jit' (fun x -> Nx.mul_s x 3.0) (vec32 [| 1.0; 2.0 |]) in
      let base = (Rune.jit_stats ()).resident_bytes in
      let r = g { u = h; v = h } in
      (* One handle behind two leaves donates once; only the two fresh
         outputs remain resident. *)
      is_true ~msg:"the duplicate handle was released once"
        ((Rune.jit_stats ()).resident_bytes - base <= 2 * 2 * 4);
      check_arr ~msg:"u" [| 6.0; 12.0 |] r.u;
      check_arr ~msg:"v" [| 9.0; 36.0 |] r.v;
      raises_donated (fun () -> to_arr h))

let test_forced_handle_unaffected_by_donate () =
  with_force_copy (fun () ->
      let g = Rune.jit' ~donate:true (fun x -> Nx.mul_s x 2.0) in
      let h = g (vec32 [| 1.0; 2.0 |]) in
      check_arr ~msg:"read before the call forces to host" [| 2.0; 4.0 |] h;
      ignore (g h);
      (* Already on the host: donation does not touch its bytes. *)
      check_arr ~msg:"host bytes survive the donating call" [| 2.0; 4.0 |] h)

let test_host_input_unaffected_by_donate () =
  with_force_copy (fun () ->
      let g = Rune.jit' ~donate:true (fun x -> Nx.mul_s x 2.0) in
      let x = vec32 [| 1.0; 2.0 |] in
      ignore (g x);
      check_arr ~msg:"a host tensor is never consumed" [| 1.0; 2.0 |] x)

let test_donate_false_leaves_handle_readable () =
  with_force_copy (fun () ->
      let g = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
      let h1 = g (vec32 [| 1.0; 2.0 |]) in
      ignore (g h1);
      check_arr ~msg:"default keeps the input handle alive" [| 2.0; 4.0 |] h1)

let test_donated_writeback_leaf_survives () =
  with_force_copy (fun () ->
      let producer = Rune.jit' (fun x -> Nx.mul_s x 2.0) in
      let h = producer (vec32 [| 1.0; 2.0 |]) in
      let step =
        Rune.jit' ~donate:true (fun x ->
            Nx.blit (Nx.mul_s x 2.0) x;
            Nx.sum x)
      in
      let s = step h in
      (* The writeback forces the leaf before donation applies: the handle
         holds the updated host value, and its device storage is released by
         the force rather than the donation. *)
      check_arr ~msg:"sum of the updated leaf" [| 12.0 |] s;
      check_arr ~msg:"the written-back leaf is not consumed" [| 4.0; 8.0 |] h)

(* Failure modes *)

let test_data_dependent_read_raises () =
  let g = Rune.jit' (fun x -> if Nx.item [ 0 ] x > 0.0 then x else Nx.neg x) in
  raises_jit_error (fun () -> g (vec32 [| 1.0; 2.0 |]))

let test_unsupported_op_raises () =
  let g = Rune.jit' (fun x -> Nx.cholesky x) in
  raises_jit_error (fun () ->
      g (Nx.create f32 [| 2; 2 |] [| 4.0; 2.0; 2.0; 3.0 |]))

let tests =
  [
    group "jit basics"
      [
        test "element-wise chain matches eager" test_elementwise_matches_eager;
        test "replay reads fresh input data" test_replay_reads_fresh_inputs;
        test "a new shape retraces" test_retrace_on_new_shape;
        test "closure-captured weights (matmul)" test_closure_matmul;
        test "jit2 returns structured outputs" test_jit2_structured_output;
      ];
    group "composition"
      [
        test "grad inside jit matches eager grad" test_grad_inside_jit;
        test "jit under grad runs eagerly" test_jit_under_grad_is_transparent;
        test "jit under vmap runs eagerly" test_jit_under_vmap_is_transparent;
        test "scan unrolls into the trace" test_scan_unrolls_inside_jit;
      ];
    group "sliding windows"
      [
        test "unfold matches eager" test_unfold_matches_eager;
        test "fold of unfold matches eager" test_fold_matches_eager;
        test "correlate matches eager" test_correlate_matches_eager;
      ];
    group "indexed access"
      [ test "scatter matches eager" test_scatter_matches_eager ];
    group "training"
      [
        test "jitted training follows the eager trajectory"
          test_jitted_training_matches_eager;
      ];
    group "state"
      [
        test "assign writes back to a leaf" test_assign_writes_back_to_leaf;
        test "cpu aliasing observes closure mutation (unspecified)"
          test_cpu_aliasing_observes_closure_mutation;
        test "assigning to a capture raises" test_assign_to_capture_raises;
        test "non-contiguous inputs fall back to copies"
          test_non_contiguous_input_matches_eager;
        test "offset views read the right span"
          test_offset_view_input_matches_eager;
        test "outputs have their own storage"
          test_outputs_have_their_own_storage;
      ];
    group "residency"
      [
        test "feedback chain moves no bytes" test_feedback_chain_moves_no_bytes;
        test "forced handles feed current bytes"
          test_forced_handle_feeds_current_bytes;
        test "the same handle can seed two leaves"
          test_same_handle_as_two_leaves;
        test "duplicate output leaves share one handle"
          test_duplicate_outputs_share_one_handle;
        test "handles feed other jitted closures" test_cross_jit_feedback;
        test "handles feed new signatures without forcing"
          test_cross_signature_feedback;
        test "pass-through outputs survive later calls"
          test_pass_through_output_survives;
        test "assigning to a resident leaf forces then writes back"
          test_assign_to_resident_leaf;
        test "grad over jit forces deferred arguments"
          test_grad_over_jit_with_deferred_arg;
        test "vmap over jit forces deferred arguments"
          test_vmap_over_jit_with_deferred_arg;
        test "signature dispatch never forces"
          test_dispatch_on_handle_reads_no_bytes;
        test "captures upload once across signatures"
          test_capture_uploaded_once_across_signatures;
        test "dropped handles are reclaimed" test_dropped_handles_are_reclaimed;
      ];
    group "donation"
      [
        test "donate bounds resident memory at two generations"
          test_donate_bounds_resident_memory;
        test "a donated handle raises on read"
          test_donated_handle_raises_on_read;
        test "re-feeding a donated handle raises"
          test_donated_handle_refeed_raises;
        test "duplicate leaves donate once" test_donate_duplicate_leaves_once;
        test "a handle read before the call is unaffected"
          test_forced_handle_unaffected_by_donate;
        test "host inputs are unaffected" test_host_input_unaffected_by_donate;
        test "donate:false is the unchanged default"
          test_donate_false_leaves_handle_readable;
        test "a written-back leaf survives donation"
          test_donated_writeback_leaf_survives;
      ];
    group "errors"
      [
        test "reading a traced value raises" test_data_dependent_read_raises;
        test "unsupported operations raise" test_unsupported_op_raises;
      ];
  ]

let () = run "rune jit" tests
