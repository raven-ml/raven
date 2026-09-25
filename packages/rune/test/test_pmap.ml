(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Multi-device parallel jit: pmap numerics against jit, replicate-vs-shard
   placements, gradients (cross-device allreduce), device residency and
   gather-on-read, placement errors, and the under-transformation fallback. Runs
   on CPU device instances, so no GPU is needed. *)

open Windtrap
open Rune_test_support.Support

let devs2 = [ Rune.device "CPU:1"; Rune.device "CPU:2" ]
let devs4 = List.map Rune.device [ "CPU:1"; "CPU:2"; "CPU:3"; "CPU:4" ]
let arange n = Array.init n (fun i -> float_of_int (i + 1) /. 7.0)
let m46 () = Nx.create f32 [| 4; 6 |] (arange 24)
let m86 () = Nx.create f32 [| 8; 6 |] (arange 48)

(* An elementwise + matmul + reduce chain over one tensor. The matmul combines
   the batch-sharded value with its own transpose (mismatched shard axes),
   exercising the cross-device realignment path on top of the plain
   allreduce. *)
let chain x =
  let y = Nx.tanh (Nx.add (Nx.mul x x) x) in
  let z = Nx.matmul y (Nx.transpose y) in
  Nx.sum z ~axes:[ 1 ]

(* Numerics vs jit *)

let test_matches_jit_2dev () =
  let x = m46 () in
  let expect = Rune.jit' chain x in
  let g = Rune.pmap ~devices:devs2 Nx.Ptree.(tensor @-> returns tensor) chain in
  check_arr ~msg:"first call" (to_arr expect) (g x);
  check_arr ~msg:"replay" (to_arr expect) (g x)

let test_matches_jit_4dev () =
  let x = m86 () in
  let expect = Rune.jit' chain x in
  let g = Rune.pmap ~devices:devs4 Nx.Ptree.(tensor @-> returns tensor) chain in
  check_arr ~msg:"4 devices" (to_arr expect) (g x)

(* No cross-device reduce: each device computes its shard independently, so the
   result is byte-equal to the single-device one. *)
let test_elementwise_byte_equal () =
  let f x = Nx.tanh (Nx.add (Nx.mul x x) x) in
  let x = m46 () in
  let expect = Rune.jit' f x in
  let g = Rune.pmap ~devices:devs2 Nx.Ptree.(tensor @-> returns tensor) f in
  check_arr ~eps:0.0 ~msg:"byte-equal" (to_arr expect) (g x)

let test_shard_axis_1 () =
  let f x = Nx.add (Nx.mul x x) x in
  let x = m46 () in
  let expect = Rune.jit' f x in
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:[ Some 1 ]
      Nx.Ptree.(tensor @-> returns tensor)
      f
  in
  check_arr ~eps:0.0 ~msg:"axis 1 shards" (to_arr expect) (g x)

let test_retrace_on_new_shape () =
  let g =
    Rune.pmap ~devices:devs2
      Nx.Ptree.(tensor @-> returns tensor)
      (fun x -> Nx.sum x)
  in
  check_arr ~msg:"first shape" [| 36.0 |]
    (g (vec32 (Array.init 8 (fun i -> float_of_int (i + 1)))));
  check_arr ~msg:"retraced shape" [| 10.0 |]
    (g (Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]))

(* Replicate-vs-shard: the data-parallel shape. Params are replicated, the batch
   is sharded, and the mean loss reduces over the sharded axis — a cross-device
   allreduce. *)

type dp = { w : Nx.float32_t; x : Nx.float32_t; t : Nx.float32_t }

module Dp = struct
  type _ t = dp

  let walk c { w; x; t } =
    let open Nx.Ptree.Walk in
    let w = field c "w" tensor w in
    let x = field c "x" tensor x in
    let t = field c "t" tensor t in
    { w; x; t }
end

let dp_ptree = Nx.Ptree.instantiate (module Dp)

let dp_loss p =
  let d = Nx.sub (Nx.matmul p.x p.w) p.t in
  Nx.mean (Nx.mul d d)

(* The parameters replicated, the batch and targets split on axis 0: one
   argument each. *)
let dp_axes = [ None; Some 0; Some 0 ]
let dp_signature r = Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns r)
let dp_args f w x t = f { w; x; t }
let dp_call g p = g p.w p.x p.t

let dp_input () =
  {
    w = Nx.create f32 [| 3; 2 |] (arange 6);
    x = Nx.create f32 [| 4; 3 |] (arange 12);
    t = Nx.create f32 [| 4; 2 |] (arange 8);
  }

let test_dp_loss_matches_jit () =
  let p = dp_input () in
  let expect = Rune.jit Nx.Ptree.(dp_ptree @-> returns tensor) dp_loss p in
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:dp_axes
      (dp_signature Nx.Ptree.tensor)
      (dp_args dp_loss)
  in
  check_arr ~msg:"mean loss over sharded batch" (to_arr expect) (dp_call g p)

(* Gradients: value_and_grad inside the pmapped function. Differentiating a mean
   over the sharded batch makes every parameter gradient a cross-device
   allreduce, which multi_pm inserts automatically — the DDP path. *)

let test_grad_inside_pmap () =
  let grads p = snd (Rune.value_and_grad dp_ptree dp_loss p) in
  let p = dp_input () in
  let expect = Rune.jit Nx.Ptree.(dp_ptree @-> returns dp_ptree) grads p in
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:dp_axes (dp_signature dp_ptree)
      (dp_args grads)
  in
  let got = dp_call g p in
  check_arr ~msg:"dw (allreduced)" (to_arr expect.w) got.w;
  check_arr ~msg:"dx (sharded)" (to_arr expect.x) got.x;
  check_arr ~msg:"dt (sharded)" (to_arr expect.t) got.t

(* Gradients through keepdims reductions: reduce ~keepdims:true realizes a
   per-shard buffer whose broadcast back against the sharded operand is the
   softmax / layer-norm shape. Differentiated inside pmap, each gradient must
   match the single-device jit gradient. *)

let keepdims_input () =
  {
    w = Nx.create f32 [| 8; 8 |] (arange 64);
    x = Nx.create f32 [| 8; 8 |] (Array.init 64 (fun i -> sin (float_of_int i)));
    t = Nx.create f32 [| 8; 8 |] (arange 64);
  }

let check_keepdims_grad loss =
  let grads p = snd (Rune.value_and_grad dp_ptree loss p) in
  let p = keepdims_input () in
  let expect = Rune.jit Nx.Ptree.(dp_ptree @-> returns dp_ptree) grads p in
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:dp_axes (dp_signature dp_ptree)
      (dp_args grads)
  in
  let got = dp_call g p in
  check_arr ~msg:"dw" (to_arr expect.w) got.w;
  check_arr ~msg:"dx" (to_arr expect.x) got.x

let test_grad_max_keepdims () =
  check_keepdims_grad (fun s ->
      let h = Nx.add s.x s.w in
      Nx.mean (Nx.exp (Nx.sub h (Nx.max h ~axes:[ -1 ] ~keepdims:true))))

let test_grad_sum_keepdims () =
  check_keepdims_grad (fun s ->
      let h = Nx.add s.x s.w in
      Nx.mean (Nx.mul h (Nx.sum h ~axes:[ -1 ] ~keepdims:true)))

let test_grad_mean_keepdims () =
  check_keepdims_grad (fun s ->
      let h = Nx.add s.x s.w in
      let d = Nx.sub h (Nx.mean h ~axes:[ -1 ] ~keepdims:true) in
      Nx.mean (Nx.mul d d))

(* Residency: outputs stay per-device; feeding an unread output back into a
   matching placement moves no bytes; reading gathers shards correctly. *)

let test_feedback_moves_no_bytes () =
  let g =
    Rune.pmap ~devices:devs2
      Nx.Ptree.(tensor @-> returns tensor)
      (fun x -> Nx.add x x)
  in
  let x = vec32 (Array.init 8 (fun i -> float_of_int i)) in
  let y1 = g x in
  Rune.reset_jit_stats ();
  let y2 = g y1 in
  let s = Rune.jit_stats () in
  equal ~msg:"feedback call moves no bytes to device" int 0 s.bytes_to_device;
  (* gather on read: shards reassemble to 4x *)
  check_arr ~eps:0.0 ~msg:"gathered result"
    (Array.init 8 (fun i -> 4.0 *. float_of_int i))
    y2

let test_replicated_feedback () =
  (* w -> w * 2 with w replicated: the replicated output seeds the replicated
     input directly on the next call. *)
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:[ None ]
      Nx.Ptree.(tensor @-> returns tensor)
      (fun w -> Nx.mul_s w 2.0)
  in
  let w = vec32 [| 1.0; 2.0; 3.0 |] in
  let w1 = g w in
  Rune.reset_jit_stats ();
  let w2 = g w1 in
  let s = Rune.jit_stats () in
  equal ~msg:"replicated feedback moves no bytes" int 0 s.bytes_to_device;
  check_arr ~eps:0.0 ~msg:"replicated read" [| 4.0; 8.0; 12.0 |] w2

(* An output split on axis 1 reads, prints and takes part in operations that
   create constants beside it, which run on the host. *)
let test_split_output_in_eager_code () =
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:[ Some 1 ]
      Nx.Ptree.(tensor @-> returns tensor)
      (fun x -> Nx.add x x)
  in
  let y = g (Nx.create f32 [| 2; 4 |] (Array.init 8 float_of_int)) in
  let h = Nx.place Nx.Placement.host y in
  equal ~msg:"printed in C order" string (Nx.to_string h) (Nx.to_string y);
  check_arr ~eps:0.0 ~msg:"tril" (to_arr (Nx.tril h)) (Nx.tril y);
  check_arr ~eps:0.0 ~msg:"gathered rows"
    (to_arr (Nx.slice [ Nx.L [ 1; 0 ] ] h))
    (Nx.slice [ Nx.L [ 1; 0 ] ] y)

let test_mismatched_placement_forces () =
  (* An output sharded on axis 0 fed into an axis-1 placement is forced to the
     host and re-split, not seeded. *)
  let f x = Nx.add x x in
  let g0 =
    Rune.pmap ~devices:devs2 ~in_axes:[ Some 0 ]
      Nx.Ptree.(tensor @-> returns tensor)
      f
  in
  let g1 =
    Rune.pmap ~devices:devs2 ~in_axes:[ Some 1 ]
      Nx.Ptree.(tensor @-> returns tensor)
      f
  in
  let x = m46 () in
  let y = g0 x in
  check_arr ~eps:0.0 ~msg:"re-split result matches"
    (to_arr (Nx.add (Nx.add x x) (Nx.add x x)))
    (g1 y)

(* A window write on the sharded axis: each device writes the part of the window
   that falls in its shard, for a static and for a traced start. *)
module Win = struct
  type win = { x : Nx.float32_t; v : Nx.float32_t; pos : Nx.int32_t }
  type _ t = win

  let walk c { x; v; pos } =
    let open Nx.Ptree.Walk in
    let x = field c "x" tensor x in
    let v = field c "v" tensor v in
    let pos = field c "pos" tensor pos in
    { x; v; pos }
end

let win_ptree = Nx.Ptree.instantiate (module Win)

let test_set_window_on_mapped_axis () =
  let f (w : Win.win) = Nx.set [ Nx.R (1, 3); Nx.A ] w.v w.x in
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:[ Some 0; None; None ]
      Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
      (fun x v pos -> f { Win.x; v; pos })
  in
  let w =
    {
      Win.x = m46 ();
      v = Nx.full f32 [| 2; 6 |] 9.0;
      pos = Nx.scalar Nx.int32 1l;
    }
  in
  check_arr ~eps:0.0 ~msg:"window spanning both shards"
    (to_arr (f w))
    (g w.x w.v w.pos)

let test_set_traced_window_on_mapped_axis () =
  let f (w : Win.win) = Nx.set [ Nx.D (w.pos, 2); Nx.A ] w.v w.x in
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:[ Some 0; None; None ]
      Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
      (fun x v pos -> f { Win.x; v; pos })
  in
  let w =
    {
      Win.x = m46 ();
      v = Nx.full f32 [| 2; 6 |] 9.0;
      pos = Nx.scalar Nx.int32 1l;
    }
  in
  check_arr ~eps:0.0 ~msg:"traced window spanning both shards"
    (to_arr (f w))
    (g w.x w.v w.pos)

(* A placed capture is on one device: a pmap reads it back and replicates it, as
   it does a host capture, and a function that bound it keeps its buffer. *)
let test_host_is_not_a_device () =
  raises_match
    (function Invalid_argument _ -> true | _ -> false)
    (fun () ->
      Rune.pmap
        ~devices:[ Rune.device "CPU"; Rune.device "CPU:1" ]
        Nx.Ptree.(tensor @-> returns tensor)
        Fun.id)

let test_placed_capture_is_replicated () =
  let w = Nx.create f32 [| 6 |] (arange 6) in
  let p = Nx.place (Nx.Placement.device (Rune.device "CPU:1")) w in
  let bound =
    Rune.jit' ~devices:[ Rune.device "CPU:1" ] (fun x -> Nx.mul x p)
  in
  let x = m46 () in
  check_arr ~msg:"bound" (to_arr (Nx.mul x w)) (bound x);
  let g =
    Rune.pmap ~devices:devs2
      Nx.Ptree.(tensor @-> returns tensor)
      (fun x -> Nx.mul x p)
  in
  check_arr ~msg:"pmap" (to_arr (Nx.mul x w)) (g x);
  check_arr ~msg:"bound, after the pmap read it" (to_arr (Nx.mul x w)) (bound x)

(* A compiled function runs on one device: an output of a pmap, on several,
   raises as its input instead of being read through the host. *)
let test_split_output_into_jit_raises () =
  let g =
    Rune.pmap ~devices:devs2
      Nx.Ptree.(tensor @-> returns tensor)
      (fun x -> Nx.mul_s x 2.0)
  in
  let y = g (m46 ()) in
  raises_match
    (function
      | Invalid_argument msg ->
          String.starts_with ~prefix:"Rune.jit: the argument at 0 is on sharded"
            msg
      | _ -> false)
    (fun () -> Rune.jit' (fun x -> Nx.add_s x 1.0) y)

(* A value on one device of a split storage, which nx makes from a cut inside
   one shard, views that device's shard: it reads that shard, enters a
   replicated input by value rather than as the whole storage, and cannot be
   consumed, which would release every shard. *)
let test_one_shard_of_a_split_storage () =
  let g =
    Rune.pmap ~devices:devs4
      Nx.Ptree.(tensor @-> returns tensor)
      (fun x -> Nx.add x x)
  in
  let x = m86 () in
  let y = g x in
  let v =
    match y with
    | Nx_effect.Placed r ->
        Nx_effect.placed
          (Nx.Placement.device (List.nth devs4 1))
          f32 r.r_view r.r_cell
    | _ -> fail "expected a placed value"
  in
  let rows = Nx.slice [ Nx.R (2, 4) ] (Nx.add x x) in
  check_arr ~eps:0.0 ~msg:"reads its device's shard" (to_arr rows) v;
  let h =
    Rune.pmap ~devices:devs4 ~in_axes:[ None ]
      Nx.Ptree.(tensor @-> returns tensor)
      (fun v -> Nx.mul_s v 3.0)
  in
  check_arr ~eps:0.0 ~msg:"enters a replicated input by value"
    (to_arr (Nx.mul_s rows 3.0))
    (h v);
  let f =
    Rune.jit
      Nx.Ptree.(consumes tensor @@ returns tensor)
      (fun v -> Nx.add_s v 1.0)
  in
  raises_match
    (function
      | Invalid_argument msg ->
          msg
          = "Rune.jit: the argument at 0 is a view of one shard of a split \
             storage, so it cannot be consumed; pass Nx.copy of it"
      | _ -> false)
    (fun () -> f v);
  check_arr ~eps:0.0 ~msg:"the split value stays" (to_arr (Nx.add x x)) y

let test_pass_through_output () =
  let g =
    Rune.pmap ~devices:devs2
      Nx.Ptree.(tensor @-> returns tensor)
      (fun x ->
        ignore (Nx.sum x);
        x)
  in
  let x = m46 () in
  check_arr ~eps:0.0 ~msg:"pass-through gathers the input" (to_arr x) (g x)

(* Consumption: a consumed argument's resident multi-device handle has every
   per-device shard buffer released once the call completes, and the consumed
   handle raises on read. A handle whose placement mismatches is read through
   the host and consumed all the same. *)

let raises_consumed f =
  raises_match
    (fun exn ->
      match exn with
      | Invalid_argument msg ->
          msg
          = "this value was consumed at 0 in a compiled call's arguments; use \
             the value the call returned"
      | _ -> false)
    (fun () -> ignore (f ()))

let test_consume_sharded_state () =
  let n = 1024 in
  let g =
    Rune.pmap ~devices:devs2
      Nx.Ptree.(consumes tensor @@ returns tensor)
      (fun x -> Nx.add_s x 1.0)
  in
  let x = vec32 (Array.make n 0.0) in
  let base = (Rune.jit_stats ()).resident_bytes in
  let h1 = g x in
  let h2 = g h1 in
  let h3 = g h2 in
  (* Consumption bounds the loop at two generations even though h1 and h2 stay
     reachable; only h3's shards remain resident. *)
  is_true ~msg:"sharded state loop holds at most two generations"
    ((Rune.jit_stats ()).resident_bytes - base <= 2 * n * 4);
  raises_consumed (fun () -> to_arr h1);
  raises_consumed (fun () -> to_arr h2);
  check_arr ~eps:0.0 ~msg:"the live generation reads correctly"
    (Array.make n 3.0) h3

let test_consume_replicated_releases_all_shards () =
  let n = 512 in
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:[ None ]
      Nx.Ptree.(consumes tensor @@ returns tensor)
      (fun w -> Nx.mul_s w 2.0)
  in
  (* Retire the handles earlier tests dropped unread, so their release cannot
     land inside the window measured below. *)
  full_major ();
  let base = (Rune.jit_stats ()).resident_bytes in
  let w1 = g (vec32 (Array.make n 1.0)) in
  (* A replicated handle owns one full-size buffer per device. *)
  is_true ~msg:"one replicated generation is resident"
    ((Rune.jit_stats ()).resident_bytes - base >= 2 * n * 4);
  let w2 = g w1 in
  is_true ~msg:"consuming releases every replica"
    ((Rune.jit_stats ()).resident_bytes - base <= 2 * n * 4);
  raises_consumed (fun () -> to_arr w1);
  check_arr ~eps:0.0 ~msg:"value" (Array.make n 4.0) w2

let test_mismatched_placement_is_consumed () =
  let f x = Nx.add x x in
  let g0 =
    Rune.pmap ~devices:devs2 ~in_axes:[ Some 0 ]
      Nx.Ptree.(tensor @-> returns tensor)
      f
  in
  let g1 =
    Rune.pmap ~devices:devs2 ~in_axes:[ Some 1 ]
      Nx.Ptree.(consumes tensor @@ returns tensor)
      f
  in
  let x = m46 () in
  let y = g0 x in
  (* The axis-1 call reads y through the host to re-split it, and consumes
     it. *)
  check_arr ~eps:0.0 ~msg:"re-split result matches"
    (to_arr (Nx.add (Nx.add x x) (Nx.add x x)))
    (g1 y);
  raises_consumed (fun () -> to_arr y)

(* A pmap copies every capture: one over the storage its consumed argument
   reaches raises before the call, and nothing is consumed. *)
let test_a_capture_of_consumed_storage_raises () =
  let w = Nx.place (Nx.Placement.device (Rune.device "CPU:1")) (m46 ()) in
  let g =
    Rune.pmap ~devices:devs2
      Nx.Ptree.(consumes tensor @@ returns tensor)
      (fun x -> Nx.add x w)
  in
  raises_match
    (function
      | Invalid_argument msg ->
          String.starts_with
            ~prefix:
              "Rune.jit: the argument at 0 and a capture of the function reach \
               one storage"
            msg
      | _ -> false)
    (fun () -> g w);
  check_arr ~eps:0.0 ~msg:"the value is not consumed" (to_arr (m46 ())) w

(* Errors *)

let test_empty_devices () =
  raises_match Exn.invalid_arg (fun () ->
      let g =
        Rune.pmap ~devices:[] Nx.Ptree.(tensor @-> returns tensor) Fun.id
      in
      ignore (g (vec32 [| 1.0; 2.0 |])))

let test_mixed_backends () =
  raises_match Exn.invalid_arg (fun () ->
      let g =
        Rune.pmap
          ~devices:[ Rune.device "CPU:1"; Rune.device "CUDA" ]
          Nx.Ptree.(tensor @-> returns tensor)
          Fun.id
      in
      ignore (g (vec32 [| 1.0; 2.0 |])))

let test_non_divisible_axis () =
  let g =
    Rune.pmap ~devices:devs2 Nx.Ptree.(tensor @-> returns tensor) Fun.id
  in
  raises_match Exn.invalid_arg (fun () ->
      ignore (g (vec32 [| 1.0; 2.0; 3.0 |])))

let test_in_axes_arity () =
  raises_match
    (function
      | Invalid_argument msg ->
          msg
          = "Rune.pmap: in_axes has 2 entries but the signature has 1 arguments"
      | _ -> false)
    (fun () ->
      let g =
        Rune.pmap ~devices:devs2 ~in_axes:[ Some 0; None ]
          Nx.Ptree.(tensor @-> returns tensor)
          Fun.id
      in
      ignore (g (vec32 [| 1.0; 2.0 |])))

let test_axis_out_of_range () =
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:[ Some 3 ]
      Nx.Ptree.(tensor @-> returns tensor)
      Fun.id
  in
  raises_match Exn.invalid_arg (fun () -> ignore (g (vec32 [| 1.0; 2.0 |])))

(* Under an enclosing transformation the pmapped function runs eagerly, so grad
   over pmap differentiates the plain function. *)

let test_grad_over_pmap_runs_eagerly () =
  let g =
    Rune.pmap ~devices:devs2
      Nx.Ptree.(tensor @-> returns tensor)
      (fun x -> Nx.sum (Nx.mul x x))
  in
  let x = vec32 [| 1.0; 2.0; 3.0; 4.0 |] in
  let dx = Rune.grad Nx.Ptree.tensor (fun x -> g x) x in
  check_arr ~msg:"grad over pmap" [| 2.0; 4.0; 6.0; 8.0 |] dx

(* Two outputs that both need a cross-device reduction: the gradient of a
   replicated parameter (summed over the sharded batch) and the loss itself. The
   gradient comes from a gather, so its buffer spans the whole vocabulary while
   the loss spans one element; a schedule that sizes the first from the second
   silently truncates it to one element and writes every lane to index zero.
   Every element of the gradient is checked, not just its sum, because a
   truncated buffer can still total correctly by accident.

   With one occurrence of each id in the batch, [d(sum (w[ids] * m))/dw] is
   [m]'s value at every entry. *)

module Grad_and_loss = struct
  type grad_and_loss = { g : Nx.float32_t; loss : Nx.float32_t }
  type _ t = grad_and_loss

  let walk c { g; loss } =
    let open Nx.Ptree.Walk in
    let g = field c "g" tensor g in
    let loss = field c "loss" tensor loss in
    { g; loss }
end

let grad_and_loss_ptree = Nx.Ptree.instantiate (module Grad_and_loss)

module Weights_mask_ids = struct
  type weights_mask_ids = {
    w : Nx.float32_t;
    m : Nx.float32_t;
    ids : Nx.int32_t;
  }

  type _ t = weights_mask_ids

  let walk c { w; m; ids } =
    let open Nx.Ptree.Walk in
    let w = field c "w" tensor w in
    let m = field c "m" tensor m in
    let ids = field c "ids" tensor ids in
    { w; m; ids }
end

let weights_mask_ids_ptree = Nx.Ptree.instantiate (module Weights_mask_ids)

let test_two_collective_outputs () =
  let vocab, dim, rows, cols = (8, 4, 4, 2) in
  let w = Nx.full f32 [| vocab; dim |] 0.02 in
  let m = Nx.full f32 [| rows; cols; dim |] 0.5 in
  let ids =
    Nx.reshape [| rows; cols |] (Nx.arange Nx.int32 0 (rows * cols) 1)
  in
  let step { Weights_mask_ids.w; m; ids } =
    let loss, g =
      Rune.value_and_grad Nx.Ptree.tensor
        (fun w ->
          let e =
            Nx.reshape [| rows; cols; dim |]
              (Nx.take w ~axis:0 ~indices:(Nx.reshape [| rows * cols |] ids))
          in
          Nx.sum (Nx.mul e m))
        w
    in
    { Grad_and_loss.g; loss }
  in
  let f =
    Rune.pmap ~devices:devs2 ~in_axes:[ None; None; Some 0 ]
      Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns grad_and_loss_ptree)
      (fun w m ids -> step { Weights_mask_ids.w; m; ids })
  in
  let out = f w m ids in
  check_arr ~msg:"gradient of the replicated parameter"
    (Array.make (vocab * dim) 0.5)
    out.Grad_and_loss.g;
  check_arr ~msg:"loss" [| 0.32 |] out.Grad_and_loss.loss

(* Dropout under pmap + grad: [fold_in_axis] folds each device's own index into
   a replicated key, so the per-device dropout masks decorrelate. At [x = 1],
   the gradient of [sum (0.5 * x**2 * mask)] recovers the mask while retaining
   the sharded input dependency from which pmap derives the output placement.
   The mask itself is a tape constant, so this exercises the same backward path
   as data-parallel dropout. *)

let test_pmap_dropout_grad_decorrelates () =
  let key = Nx.Rng.key 7 in
  let mask_grad x key =
    snd
      (Rune.value_and_grad Nx.Ptree.tensor
         (fun x ->
           let m =
             Nx.cast f32
               (Nx.Rng.bernoulli (Nx.Rng.fold_in_axis key)
                  (Nx.broadcast_to [| 2; 16 |] (Nx.scalar f32 0.5)))
           in
           Nx.mul_s (Nx.sum (Nx.mul (Nx.mul x x) m)) 0.5)
         x)
  in
  let g =
    Rune.pmap ~devices:devs2 ~in_axes:[ Some 0; None ]
      Nx.Ptree.(tensor @-> tensor @-> returns tensor)
      mask_grad
  in
  let masks = g (Nx.ones f32 [| 2; 16 |]) key in
  for i = 0 to 1 do
    check_arr ~eps:0.0
      ~msg:(Printf.sprintf "device %d mask is fold_in key %d" i i)
      (to_arr
         (Nx.slice [ Nx.I i ]
            (Nx.cast f32
               (Nx.Rng.bernoulli (Nx.Rng.fold_in key i)
                  (Nx.broadcast_to [| 2; 16 |] (Nx.scalar f32 0.5))))))
      (Nx.slice [ Nx.I i ] masks)
  done;
  is_true ~msg:"per-device dropout masks are decorrelated"
    (to_arr (Nx.slice [ Nx.I 0 ] masks) <> to_arr (Nx.slice [ Nx.I 1 ] masks))

(* The DP microbench: a 2-layer MLP train step (value_and_grad + SGD inside
   pmap), params replicated, batch sharded over 2 devices. The 10-step loss
   trajectory matches single-device jit at the same effective batch. *)

type mlp = {
  w1 : Nx.float32_t;
  b1 : Nx.float32_t;
  w2 : Nx.float32_t;
  b2 : Nx.float32_t;
  xb : Nx.float32_t;
  yb : Nx.float32_t;
}

module Mlp = struct
  type _ t = mlp

  let walk c { w1; b1; w2; b2; xb; yb } =
    let open Nx.Ptree.Walk in
    let w1 = field c "w1" tensor w1 in
    let b1 = field c "b1" tensor b1 in
    let w2 = field c "w2" tensor w2 in
    let b2 = field c "b2" tensor b2 in
    let xb = field c "xb" tensor xb in
    let yb = field c "yb" tensor yb in
    { w1; b1; w2; b2; xb; yb }
end

let mlp_ptree = Nx.Ptree.instantiate (module Mlp)

let mlp_loss s =
  let h = Nx.relu (Nx.add (Nx.matmul s.xb s.w1) s.b1) in
  let p = Nx.add (Nx.matmul h s.w2) s.b2 in
  let d = Nx.sub p s.yb in
  Nx.mean (Nx.mul d d)

let sgd (type a b) (w : (a, b) Nx.t) (g : (a, b) Nx.t) : (a, b) Nx.t =
  Nx.sub w (Nx.mul g (scalar_like w 0.05))

let mlp_step s =
  let l, g = Rune.value_and_grad mlp_ptree mlp_loss s in
  ( {
      (Nx.Ptree.map2 mlp_ptree (fun _ w g -> sgd w g) s g) with
      xb = s.xb;
      yb = s.yb;
    },
    l )

let mlp_init () =
  let rng i n =
    Array.init n (fun j -> sin (float_of_int ((i * 7919) + j)) *. 0.5)
  in
  {
    w1 = Nx.create f32 [| 4; 8 |] (rng 1 32);
    b1 = Nx.create f32 [| 8 |] (rng 2 8);
    w2 = Nx.create f32 [| 8; 2 |] (rng 3 16);
    b2 = Nx.create f32 [| 2 |] (rng 4 2);
    xb = Nx.create f32 [| 16; 4 |] (rng 5 64);
    yb = Nx.create f32 [| 16; 2 |] (rng 6 32);
  }

let trajectory step0 =
  let s = ref (mlp_init ()) in
  Array.init 10 (fun _ ->
      let s', l = step0 !s in
      s := s';
      scalar l)

let test_dp_training_matches_jit () =
  let jit_losses =
    trajectory
      (Rune.jit
         Nx.Ptree.(mlp_ptree @-> returns (pair mlp_ptree tensor))
         mlp_step)
  in
  (* The parameters replicated, the batch split: one argument per tensor. *)
  let in_axes = [ None; None; None; None; Some 0; Some 0 ] in
  let pmap_step () =
    let g =
      Rune.pmap ~devices:devs2 ~in_axes
        Nx.Ptree.(
          tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor
          @-> returns (pair mlp_ptree tensor))
        (fun w1 b1 w2 b2 xb yb -> mlp_step { w1; b1; w2; b2; xb; yb })
    in
    fun s -> g s.w1 s.b1 s.w2 s.b2 s.xb s.yb
  in
  let pmap_losses = trajectory (pmap_step ()) in
  Array.iteri
    (fun i l ->
      equal
        ~msg:(Printf.sprintf "loss at step %d" i)
        (float 1e-6) l pmap_losses.(i))
    jit_losses;
  (* Late steps run entirely on resident state: nothing moves to the devices. *)
  Rune.reset_jit_stats ();
  let pstep = pmap_step () in
  let s0 = mlp_init () in
  let s1, _ = pstep s0 in
  Rune.reset_jit_stats ();
  let s2, _ = pstep s1 in
  let stats = Rune.jit_stats () in
  equal ~msg:"resident training step moves no bytes to device" int 0
    stats.bytes_to_device;
  ignore (Sys.opaque_identity s2)

(* A multi-device trace declines to stage a scan (its probe answer is [false]):
   the recurrence and its gradient unroll into each shard's trace as they did
   before staging existed. The scan folds the columns, so sharding the rows
   commutes with it and the concatenated shard gradients equal the single-device
   gradient. *)
let test_grad_through_scan_inside_pmap () =
  let loss x =
    let rows = (Nx.shape x).(0) in
    let _, ys =
      Rune.scan'
        ~f:(fun c col ->
          let c = Nx.tanh (Nx.add c col) in
          (c, Nx.mul c c))
        ~init:(Nx.zeros f32 [| rows |]) (Nx.transpose x)
    in
    Nx.sum ys
  in
  let grads x = Rune.grad Nx.Ptree.tensor loss x in
  let x = m46 () in
  let expect = Rune.jit Nx.Ptree.(tensor @-> returns tensor) grads x in
  let g = Rune.pmap ~devices:devs2 Nx.Ptree.(tensor @-> returns tensor) grads in
  check_arr ~msg:"sharded grads" (to_arr expect) (g x)

let tests =
  [
    group "numerics"
      [
        test "elementwise+matmul+reduce matches jit on 2 devices"
          test_matches_jit_2dev;
        test "elementwise+matmul+reduce matches jit on 4 devices"
          test_matches_jit_4dev;
        test "elementwise chain is byte-equal to jit"
          test_elementwise_byte_equal;
        test "sharding along axis 1" test_shard_axis_1;
        test "new shape retraces" test_retrace_on_new_shape;
      ];
    group "placement"
      [
        test "replicated params + sharded batch mean loss matches jit"
          test_dp_loss_matches_jit;
        test "grad inside pmap allreduces like jit" test_grad_inside_pmap;
        test "grad through max keepdims matches jit" test_grad_max_keepdims;
        test "grad through sum keepdims matches jit" test_grad_sum_keepdims;
        test "grad through mean keepdims matches jit" test_grad_mean_keepdims;
        test "window write on the mapped axis" test_set_window_on_mapped_axis;
        test "traced window write on the mapped axis"
          test_set_traced_window_on_mapped_axis;
      ];
    group "residency"
      [
        test "a split output in eager code" test_split_output_in_eager_code;
        test "feedback call moves no bytes" test_feedback_moves_no_bytes;
        test "replicated outputs feed back without transfer"
          test_replicated_feedback;
        test "mismatched placement forces and re-splits"
          test_mismatched_placement_forces;
        test "pass-through outputs gather on read" test_pass_through_output;
        test "a placed capture is read back and replicated"
          test_placed_capture_is_replicated;
        test "a split output into jit raises" test_split_output_into_jit_raises;
        test "one shard of a split storage" test_one_shard_of_a_split_storage;
      ];
    group "consumption"
      [
        test "sharded state loop is bounded at two generations"
          test_consume_sharded_state;
        test "replicated consumption releases every replica"
          test_consume_replicated_releases_all_shards;
        test "a mismatched placement is read and consumed"
          test_mismatched_placement_is_consumed;
        test "a capture of consumed storage raises"
          test_a_capture_of_consumed_storage_raises;
      ];
    group "errors"
      [
        test "empty devices raises" test_empty_devices;
        test "the host is not a pmap device" test_host_is_not_a_device;
        test "mixed backends raise" test_mixed_backends;
        test "non-divisible shard axis raises" test_non_divisible_axis;
        test "in_axes arity mismatch raises" test_in_axes_arity;
        test "shard axis out of range raises" test_axis_out_of_range;
      ];
    group "composition"
      [
        test "grad over pmap runs eagerly" test_grad_over_pmap_runs_eagerly;
        test "dropout under pmap+grad decorrelates masks"
          test_pmap_dropout_grad_decorrelates;
        test "two collectively reduced outputs" test_two_collective_outputs;
        test "grad through a scan unrolls into the shard traces"
          test_grad_through_scan_inside_pmap;
      ];
    group "training"
      [
        test "data-parallel MLP training follows the jit trajectory"
          test_dp_training_matches_jit;
      ];
  ]

let () = run "rune pmap" tests
