(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Compiled functions over values split or copied across devices: numerics
   against one device, gradients (cross-device allreduce), residency and
   gather-on-read, consumption, placement errors, and the under-transformation
   fallback. Runs on CPU device instances, so no GPU is needed. *)

open Windtrap
open Rune_test_support.Support

let devs2 = [ Rune.device "CPU:1"; Rune.device "CPU:2" ]
let devs4 = List.map Rune.device [ "CPU:1"; "CPU:2"; "CPU:3"; "CPU:4" ]
let rows ds x = Nx.place (Nx.Placement.sharded ~axis:0 ds) x
let arange n = Array.init n (fun i -> float_of_int (i + 1) /. 7.0)
let m46 () = Nx.create f32 [| 4; 6 |] (arange 24)
let m86 () = Nx.create f32 [| 8; 6 |] (arange 48)
let placement = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

(* An elementwise + matmul + reduce chain over one tensor. The matmul combines
   the batch-split value with its own transpose, gathered to every device first
   ([gather]), on top of the plain allreduce. *)
let chain ?(gather = Fun.id) x =
  let y = Nx.tanh (Nx.add (Nx.mul x x) x) in
  let z = Nx.matmul y (gather (Nx.transpose y)) in
  Nx.sum z ~axes:[ 1 ]

(* Numerics against one device *)

let test_matches_jit_2dev () =
  let x = m46 () in
  let expect = Rune.jit' chain x in
  let g =
    Rune.jit' (chain ~gather:(Nx.place (Nx.Placement.replicated devs2)))
  in
  check_arr ~msg:"first call" (to_arr expect) (g (rows devs2 x));
  check_arr ~msg:"replay" (to_arr expect) (g (rows devs2 x))

let test_matches_jit_4dev () =
  let x = m86 () in
  let expect = Rune.jit' chain x in
  let g =
    Rune.jit' (chain ~gather:(Nx.place (Nx.Placement.replicated devs4)))
  in
  check_arr ~msg:"4 devices" (to_arr expect) (g (rows devs4 x))

(* No cross-device reduce: each device computes its slice independently, so the
   result is byte-equal to the single-device one. *)
let test_elementwise_byte_equal () =
  let f x = Nx.tanh (Nx.add (Nx.mul x x) x) in
  let x = m46 () in
  let expect = Rune.jit' f x in
  check_arr ~eps:0.0 ~msg:"byte-equal" (to_arr expect)
    (Rune.jit' f (rows devs2 x))

let test_split_axis_1 () =
  let f x = Nx.add (Nx.mul x x) x in
  let x = m46 () in
  let expect = Rune.jit' f x in
  check_arr ~eps:0.0 ~msg:"split on axis 1" (to_arr expect)
    (Rune.jit' f (Nx.place (Nx.Placement.sharded ~axis:1 devs2) x))

let test_retrace_on_new_shape () =
  let g = Rune.jit' (fun x -> Nx.sum x) in
  check_arr ~msg:"first shape" [| 36.0 |]
    (g (rows devs2 (vec32 (Array.init 8 (fun i -> float_of_int (i + 1))))));
  check_arr ~msg:"retraced shape" [| 10.0 |]
    (g (rows devs2 (Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |])))

(* The data-parallel shape: the parameters enter from the host as a copy on each
   device, the batch and targets are split on axis 0, and the mean loss reduces
   over the split axis, a cross-device allreduce. *)

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

let dp_split p = { p with x = rows devs2 p.x; t = rows devs2 p.t }

let dp_input () =
  {
    w = Nx.create f32 [| 3; 2 |] (arange 6);
    x = Nx.create f32 [| 4; 3 |] (arange 12);
    t = Nx.create f32 [| 4; 2 |] (arange 8);
  }

let test_dp_loss_matches_jit () =
  let p = dp_input () in
  let g = Rune.jit Nx.Ptree.(dp_ptree @-> returns tensor) dp_loss in
  check_arr ~msg:"mean loss over a split batch" (to_arr (g p)) (g (dp_split p))

(* Gradients: value_and_grad inside the compiled function. Differentiating a
   mean over the split batch makes every parameter gradient a cross-device
   allreduce, which tolk's rewrite inserts: the DDP path. *)

let test_grad_over_a_split_batch () =
  let grads p = snd (Rune.value_and_grad dp_ptree dp_loss p) in
  let p = dp_input () in
  let g = Rune.jit Nx.Ptree.(dp_ptree @-> returns dp_ptree) grads in
  let expect = g p in
  let got = g (dp_split p) in
  check_arr ~msg:"dw (allreduced)" (to_arr expect.w) got.w;
  equal ~msg:"dw is a copy on each device" placement
    (Nx.Placement.replicated devs2)
    (Nx.placement got.w);
  check_arr ~msg:"dx (split)" (to_arr expect.x) got.x;
  check_arr ~msg:"dt (split)" (to_arr expect.t) got.t

(* Gradients through keepdims reductions: reduce ~keepdims:true realizes a
   per-slice buffer whose broadcast back against the split operand is the
   softmax / layer-norm shape. Each gradient must match the single-device
   one. *)

let keepdims_input () =
  {
    w = Nx.create f32 [| 8; 8 |] (arange 64);
    x = Nx.create f32 [| 8; 8 |] (Array.init 64 (fun i -> sin (float_of_int i)));
    t = Nx.create f32 [| 8; 8 |] (arange 64);
  }

let check_keepdims_grad loss =
  let grads p = snd (Rune.value_and_grad dp_ptree loss p) in
  let p = keepdims_input () in
  let g = Rune.jit Nx.Ptree.(dp_ptree @-> returns dp_ptree) grads in
  let expect = g p in
  let got = g (dp_split p) in
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

(* A window write on the split axis: each device writes the part of the window
   that falls in its slice, for a static and for a traced start. *)
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

let check_window msg f =
  let g = Rune.jit Nx.Ptree.(win_ptree @-> returns tensor) f in
  let w =
    {
      Win.x = m46 ();
      v = Nx.full f32 [| 2; 6 |] 9.0;
      pos = Nx.scalar Nx.int32 1l;
    }
  in
  check_arr ~eps:0.0 ~msg (to_arr (f w)) (g { w with x = rows devs2 w.x })

let test_set_window_on_the_split_axis () =
  check_window "window spanning both slices" (fun (w : Win.win) ->
      Nx.set [ Nx.R (1, 3); Nx.A ] w.v w.x)

let test_set_traced_window_on_the_split_axis () =
  check_window "traced window spanning both slices" (fun (w : Win.win) ->
      Nx.set [ Nx.D (w.pos, 2); Nx.A ] w.v w.x)

(* A pool of [slots] rows of 4 by 2 values, split by [axis] over four devices,
   receives rows 4 and 1 of [values]. *)
let pool_rows ~axis =
  let pool = Nx.create f32 [| 8; 4; 2 |] (arange 64) in
  let slots = Nx.create Nx.int32 [| 2 |] [| 4l; 1l |] in
  let values =
    Nx.create f32 [| 2; 4; 2 |]
      (Array.init 16 (fun i -> 100. +. float_of_int i))
  in
  let write slots values pool =
    let indices =
      Nx.broadcast_to [| 2; 4; 2 |] (Nx.reshape [| 2; 1; 1 |] slots)
    in
    Nx.scatter ~unique_indices:true ~axis:0 ~indices ~values pool
  in
  let split x = Nx.place (Nx.Placement.sharded ~axis devs4) x in
  let values = if axis = 0 then values else split values in
  (write, split pool, slots, values)

(* An indexed write into a split destination: each device writes the updates
   that land in its slice, off the split axis and along it. *)
let test_indexed_write_into_a_split_value () =
  List.iter
    (fun axis ->
      let write, pool, slots, values = pool_rows ~axis in
      let msg = Printf.sprintf "split along axis %d" axis in
      let y = Rune.jit' (fun pool -> write slots values pool) pool in
      equal ~msg:(msg ^ ": placement") placement (Nx.placement pool)
        (Nx.placement y);
      check_arr ~eps:0.0 ~msg (to_arr (write slots values pool)) y)
    [ 0; 1 ]

(* Residency: outputs stay on their devices; feeding an output back moves no
   bytes; reading gathers the slices. *)

(* An output split on axis 1 reads, prints and takes part in operations that
   create constants beside it. *)
let test_split_output_in_eager_code () =
  let y =
    Rune.jit'
      (fun x -> Nx.add x x)
      (Nx.place
         (Nx.Placement.sharded ~axis:1 devs2)
         (Nx.create f32 [| 2; 4 |] (Array.init 8 float_of_int)))
  in
  let h = Nx.place Nx.Placement.host y in
  equal ~msg:"printed in C order" string (Nx.to_string h) (Nx.to_string y);
  check_arr ~eps:0.0 ~msg:"tril" (to_arr (Nx.tril h)) (Nx.tril y);
  check_arr ~eps:0.0 ~msg:"gathered rows"
    (to_arr (Nx.slice [ Nx.L [ 1; 0 ] ] h))
    (Nx.slice [ Nx.L [ 1; 0 ] ] y)

(* Linear algebra and windows act along their own axes: over a value split on
   its batch axis, eager and compiled code agree on the elements and the
   placement. *)
let test_batch_split_operations () =
  let agree msg f x =
    let compiled = Rune.jit' f (rows devs2 x) in
    let eager = f (rows devs2 x) in
    equal ~msg:(msg ^ ": placement") placement (Nx.placement compiled)
      (Nx.placement eager);
    check_arr ~eps:1e-5 ~msg (to_arr compiled) eager
  in
  let batch = Nx.reshape [| 2; 3; 3 |] (Nx.arange Nx.float32 0 18 1) in
  let spd =
    Nx.add
      (Nx.matmul batch (Nx.transpose ~axes:[ 0; 2; 1 ] batch))
      (Nx.mul_s (Nx.eye Nx.float32 3) 10.)
  in
  agree "cholesky" Nx.cholesky spd;
  let w = Nx.ones Nx.float32 [| 2; 2 |] in
  agree "correlate"
    (fun i -> Nx.correlate i w)
    (Nx.reshape [| 2; 1; 4; 4 |] (Nx.arange Nx.float32 0 32 1))

(* A roll along the split axis by one slice of two cuts each slice whole, which
   a compiled program copies to both devices: eager and compiled agree on the
   elements and the placement. *)
let test_roll_by_a_slice () =
  let x = Nx.reshape [| 8; 6 |] (Nx.arange Nx.float32 0 48 1) in
  let roll = Nx.roll ~axis:0 4 in
  let compiled = Rune.jit' roll (rows devs2 x) in
  let eager = roll (rows devs2 x) in
  equal ~msg:"placement" placement (Nx.placement compiled) (Nx.placement eager);
  equal ~msg:"elements" (array float_exact) (Nx.to_array compiled)
    (Nx.to_array eager);
  equal ~msg:"the roll" (array float_exact)
    (Nx.to_array (roll x))
    (Nx.to_array eager);
  (* Over four devices, two whole slices combine as copies on all four. *)
  let pair s =
    Nx.add (Nx.slice [ Nx.R (0, 2) ] s) (Nx.slice [ Nx.R (2, 4) ] s)
  in
  let compiled = Rune.jit' pair (rows devs4 x) in
  let eager = pair (rows devs4 x) in
  equal ~msg:"four devices: placement" placement (Nx.placement compiled)
    (Nx.placement eager);
  equal ~msg:"four devices: on all four" placement
    (Nx.Placement.replicated devs4)
    (Nx.placement eager);
  equal ~msg:"four devices: elements" (array float_exact) (Nx.to_array compiled)
    (Nx.to_array eager)

let test_feedback_moves_no_bytes () =
  let g = Rune.jit' (fun x -> Nx.add x x) in
  let x = vec32 (Array.init 8 (fun i -> float_of_int i)) in
  let y1 = g (rows devs2 x) in
  Rune.reset_jit_stats ();
  let y2 = g y1 in
  let s = Rune.jit_stats () in
  equal ~msg:"feedback call moves no bytes to device" int 0 s.bytes_to_device;
  (* gather on read: the slices reassemble to 4x *)
  check_arr ~eps:0.0 ~msg:"gathered result"
    (Array.init 8 (fun i -> 4.0 *. float_of_int i))
    y2

(* w -> w * 2 with w entering from the host as a copy on each device: the output
   is a copy on each, and seeds the next call directly. *)
let test_replicated_feedback () =
  let g = Rune.jit' ~devices:devs2 (fun w -> Nx.mul_s w 2.0) in
  let w1 = g (vec32 [| 1.0; 2.0; 3.0 |]) in
  equal ~msg:"a copy on each device" placement
    (Nx.Placement.replicated devs2)
    (Nx.placement w1);
  Rune.reset_jit_stats ();
  let w2 = g w1 in
  let s = Rune.jit_stats () in
  equal ~msg:"replicated feedback moves no bytes" int 0 s.bytes_to_device;
  check_arr ~eps:0.0 ~msg:"replicated read" [| 4.0; 8.0; 12.0 |] w2

(* An output split on axis 0 placed split on axis 1 moves between the devices
   before the call, and the call runs over the new split. *)
let test_a_moved_output_feeds_a_call () =
  let g = Rune.jit' (fun x -> Nx.add x x) in
  let x = m46 () in
  let y = g (rows devs2 x) in
  let z = g (Nx.place (Nx.Placement.sharded ~axis:1 devs2) y) in
  equal ~msg:"split as its input" placement
    (Nx.Placement.sharded ~axis:1 devs2)
    (Nx.placement z);
  check_arr ~eps:0.0 ~msg:"value" (to_arr (Nx.add (Nx.add x x) (Nx.add x x))) z

let test_pass_through_output () =
  let g =
    Rune.jit' (fun x ->
        ignore (Nx.sum x);
        x)
  in
  let x = m46 () in
  check_arr ~eps:0.0 ~msg:"pass-through gathers the input" (to_arr x)
    (g (rows devs2 x))

(* A capture on the program's devices is bound where it lives; one on other
   devices raises instead of being read back. *)
let test_placed_capture_is_bound () =
  let w = Nx.create f32 [| 6 |] (arange 6) in
  let x = m46 () in
  let on1 = Nx.place (Nx.Placement.device (Rune.device "CPU:1")) w in
  raises_match
    (function
      | Invalid_argument msg ->
          String.starts_with ~prefix:"Rune.jit: a captured value is on CPU:1"
            msg
      | _ -> false)
    (fun () -> Rune.jit' (fun x -> Nx.mul x on1) (rows devs2 x));
  let copies = Nx.place (Nx.Placement.replicated devs2) w in
  let g = Rune.jit' (fun x -> Nx.mul x copies) in
  check_arr ~msg:"bound" (to_arr (Nx.mul x w)) (g (rows devs2 x));
  let x = rows devs2 x in
  Rune.reset_jit_stats ();
  check_arr ~msg:"again" (to_arr (Nx.mul (m46 ()) w)) (g x);
  equal ~msg:"the capture moves nothing" int 0
    (Rune.jit_stats ()).bytes_to_device

(* A movement of a split output is a view of every slice: its elements are the
   host movement's, read through per-slice views with offsets, negative strides
   and zero strides. *)
let test_moved_split_output () =
  let x = m86 () in
  let y = Rune.jit' (fun x -> Nx.add x x) (rows devs4 x) and h = Nx.add x x in
  let split axis = Nx.Placement.sharded ~axis devs4 in
  List.iter
    (fun (what, move, p) ->
      let m = move y in
      equal ~msg:(what ^ ": placement") placement p (Nx.placement m);
      check_arr ~eps:0.0 ~msg:what (to_arr (move h)) m)
    [
      ("transpose", (fun t -> Nx.transpose t), split 1);
      ("reshape", Nx.reshape [| 4; 2; 6 |], split 0);
      ("whole rows", Nx.slice [ Nx.A; Nx.R (1, 4) ], split 0);
      ("flipped columns", Nx.flip ~axes:[ 1 ], split 0);
      ( "broadcast",
        (fun t -> Nx.broadcast_to [| 8; 3; 6 |] (Nx.reshape [| 8; 1; 6 |] t)),
        split 0 );
      ( "reversed window of the transpose",
        (fun t ->
          Nx.flip ~axes:[ 0 ] (Nx.transpose (Nx.slice [ Nx.A; Nx.R (2, 5) ] t))),
        split 1 );
      ( "a row inside one slice",
        (fun t -> Nx.slice [ Nx.I 5; Nx.R (1, 5) ] t),
        Nx.Placement.device (List.nth devs4 2) );
    ]

(* A cut inside one slice of a split value is a view of that slice on its
   device: it reads that slice, runs a program on that device alone, and cannot
   be consumed, which would release every slice. *)
let test_one_slice_of_a_split_storage () =
  let x = m86 () in
  let y = Rune.jit' (fun x -> Nx.add x x) (rows devs4 x) in
  let v = Nx.slice [ Nx.R (2, 4) ] y in
  equal ~msg:"on its slice's device" placement
    (Nx.Placement.device (List.nth devs4 1))
    (Nx.placement v);
  let two = Nx.slice [ Nx.R (2, 4) ] (Nx.add x x) in
  check_arr ~eps:0.0 ~msg:"reads its device's slice" (to_arr two) v;
  equal ~msg:"item" float_exact
    (Nx.item [ 5; 2 ] (Nx.add x x))
    (Nx.item [ 5; 2 ] y);
  let tripled, up, _ =
    let s0 = Rune.jit_stats () in
    let r = Rune.jit' (fun v -> Nx.mul_s v 3.0) v in
    (r, (Rune.jit_stats ()).bytes_to_device - s0.bytes_to_device, ())
  in
  check_arr ~eps:0.0 ~msg:"a program on its device"
    (to_arr (Nx.mul_s two 3.0))
    tripled;
  equal ~msg:"reads the slice in place" int 0 up;
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

(* Consumption: a consumed argument's per-device buffers are released once the
   call completes, and the consumed value raises on read. *)

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

let test_consume_split_state () =
  let n = 1024 in
  let g =
    Rune.jit
      Nx.Ptree.(consumes tensor @@ returns tensor)
      (fun x -> Nx.add_s x 1.0)
  in
  let x = rows devs2 (vec32 (Array.make n 0.0)) in
  let base = (Rune.jit_stats ()).resident_bytes in
  let h1 = g x in
  let h2 = g h1 in
  let h3 = g h2 in
  (* Consumption bounds the loop at two generations even though h1 and h2 stay
     reachable; only h3's slices remain resident beside x's. *)
  is_true ~msg:"split state loop holds at most two generations"
    ((Rune.jit_stats ()).resident_bytes - base <= 2 * n * 4);
  raises_consumed (fun () -> to_arr h1);
  raises_consumed (fun () -> to_arr h2);
  check_arr ~eps:0.0 ~msg:"the live generation reads correctly"
    (Array.make n 3.0) h3

let test_consume_replicated_releases_every_copy () =
  let n = 512 in
  let g =
    Rune.jit ~devices:devs2
      Nx.Ptree.(consumes tensor @@ returns tensor)
      (fun w -> Nx.mul_s w 2.0)
  in
  (* Retire the values earlier tests dropped unread, so their release cannot
     land inside the window measured below. *)
  full_major ();
  let base = (Rune.jit_stats ()).resident_bytes in
  let w1 = g (vec32 (Array.make n 1.0)) in
  (* A copy on each device owns one full-size buffer per device. *)
  is_true ~msg:"one replicated generation is resident"
    ((Rune.jit_stats ()).resident_bytes - base >= 2 * n * 4);
  let w2 = g w1 in
  is_true ~msg:"consuming releases every copy"
    ((Rune.jit_stats ()).resident_bytes - base <= 2 * n * 4);
  raises_consumed (fun () -> to_arr w1);
  check_arr ~eps:0.0 ~msg:"value" (Array.make n 4.0) w2

(* A consumed argument over several devices lends its storage, its buffer on
   every device, to the result that continues it: a carry split or copied, and a
   pool an indexed write lands in, which the program never copies. *)
let test_consumed_storage_is_lent () =
  let lent f x =
    let before = (Rune.jit_stats ()).reused_bytes in
    let y = f x in
    (y, (Rune.jit_stats ()).reused_bytes - before)
  in
  let step =
    Rune.jit
      Nx.Ptree.(consumes tensor @@ returns tensor)
      (fun x -> Nx.add_s (Nx.mul_s x 2.0) 1.0)
  in
  List.iter
    (fun (msg, p, bytes) ->
      let y, reused = lent step (Nx.place p (m86 ())) in
      equal ~msg:(msg ^ ": placement") placement p (Nx.placement y);
      equal ~msg:(msg ^ ": every buffer is lent") int bytes reused;
      check_arr ~eps:0.0 ~msg (to_arr (Nx.add_s (Nx.mul_s (m86 ()) 2.0) 1.0)) y)
    [
      ("split", Nx.Placement.sharded ~axis:0 devs4, 48 * 4);
      ("copied", Nx.Placement.replicated devs4, 4 * 48 * 4);
    ];
  List.iter
    (fun axis ->
      let write, pool, slots, values = pool_rows ~axis in
      let msg = Printf.sprintf "a pool split along axis %d" axis in
      let expected = to_arr (write slots values pool) in
      let f =
        Rune.jit
          Nx.Ptree.(consumes tensor @@ returns tensor)
          (write slots values)
      in
      let y, reused = lent f (Nx.copy pool) in
      equal ~msg:(msg ^ ": the pool is lent") int (64 * 4) reused;
      check_arr ~eps:0.0 ~msg expected y;
      (* Written and only read, the pool is filled in the program. *)
      let g =
        Rune.jit
          Nx.Ptree.(consumes tensor @@ returns tensor)
          (fun pool -> Nx.sum ~axes:[ 1; 2 ] (write slots values pool))
      in
      check_arr ~msg:(msg ^ ", only read")
        (to_arr (Nx.sum ~axes:[ 1; 2 ] (write slots values pool)))
        (g (Nx.copy pool)))
    [ 0; 1 ]

(* A capture over the storage a consumed argument reaches raises before the
   call, and nothing is consumed. *)
let test_a_capture_of_consumed_storage_raises () =
  let w = Nx.place (Nx.Placement.replicated devs2) (m46 ()) in
  let g =
    Rune.jit Nx.Ptree.(consumes tensor @@ returns tensor) (fun x -> Nx.add x w)
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
      Rune.jit' ~devices:[] Fun.id (vec32 [| 1.0; 2.0 |]))

let test_host_in_a_device_list () =
  raises_match Exn.invalid_arg (fun () ->
      Rune.jit'
        ~devices:[ Rune.device "CPU"; Rune.device "CPU:1" ]
        Fun.id
        (vec32 [| 1.0; 2.0 |]))

let test_mixed_backends () =
  raises_match Exn.invalid_arg (fun () ->
      Rune.jit'
        ~devices:[ Rune.device "CPU:1"; Rune.device "CUDA" ]
        Fun.id
        (vec32 [| 1.0; 2.0 |]))

let test_a_split_that_does_not_divide () =
  raises_match Exn.invalid_arg (fun () ->
      Rune.jit' ~devices:devs2
        (Nx.place (Nx.Placement.sharded ~axis:0 devs2))
        (vec32 [| 1.0; 2.0; 3.0 |]))

let test_a_split_axis_out_of_range () =
  raises_match Exn.invalid_arg (fun () ->
      Rune.jit' ~devices:devs2
        (Nx.place (Nx.Placement.sharded ~axis:3 devs2))
        (vec32 [| 1.0; 2.0 |]))

(* Under an enclosing transformation a compiled function runs eagerly, so grad
   over it differentiates the plain function. *)

let test_grad_over_a_program_runs_eagerly () =
  let g = Rune.jit' ~devices:devs2 (fun x -> Nx.sum (Nx.mul x x)) in
  let x = vec32 [| 1.0; 2.0; 3.0; 4.0 |] in
  let dx = Rune.grad Nx.Ptree.tensor (fun x -> g x) x in
  check_arr ~msg:"grad over a program" [| 2.0; 4.0; 6.0; 8.0 |] dx

(* Two outputs that both need a cross-device reduction: the gradient of a
   parameter on every device (summed over the split batch) and the loss itself.
   The gradient comes from a gather, so its buffer spans the whole vocabulary
   while the loss spans one element; a schedule that sizes the first from the
   second silently truncates it to one element and writes every lane to index
   zero. Every element of the gradient is checked, not just its sum, because a
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

let test_two_collective_outputs () =
  let vocab, dim, n, cols = (8, 4, 4, 2) in
  let w = Nx.full f32 [| vocab; dim |] 0.02 in
  let m = Nx.full f32 [| n; cols; dim |] 0.5 in
  let ids = Nx.reshape [| n; cols |] (Nx.arange Nx.int32 0 (n * cols) 1) in
  let step w m ids =
    let loss, g =
      Rune.value_and_grad Nx.Ptree.tensor
        (fun w ->
          let e =
            Nx.reshape [| n; cols; dim |]
              (Nx.take w ~axis:0 ~indices:(Nx.reshape [| n * cols |] ids))
          in
          Nx.sum (Nx.mul e m))
        w
    in
    { Grad_and_loss.g; loss }
  in
  let f =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns grad_and_loss_ptree)
      step
  in
  let out = f w m (rows devs2 ids) in
  check_arr ~msg:"gradient of the parameter on every device"
    (Array.make (vocab * dim) 0.5)
    out.Grad_and_loss.g;
  check_arr ~msg:"loss" [| 0.32 |] out.Grad_and_loss.loss

(* Dropout per device: [vmap] over an axis split one slice per device, with
   [fold_in_axis] folding each lane's index into the key the lanes share, so the
   devices' masks decorrelate. At [x = 1] the gradient of [sum (0.5 * x**2 *
   mask)] recovers the mask; the mask is a tape constant, so this exercises the
   backward path of data-parallel dropout. *)

let test_dropout_per_device_decorrelates () =
  let key = Nx.Rng.key 7 in
  let draw k =
    Nx.Rng.bernoulli k (Nx.broadcast_to [| 16 |] (Nx.scalar f32 0.5))
  in
  let mask_grad rows key =
    Rune.vmap'
      (fun x ->
        snd
          (Rune.value_and_grad Nx.Ptree.tensor
             (fun x ->
               let m = Nx.cast f32 (draw (Nx.Rng.fold_in_axis key)) in
               Nx.mul_s (Nx.sum (Nx.mul (Nx.mul x x) m)) 0.5)
             x))
      rows
  in
  let g =
    Rune.jit Nx.Ptree.(tensor @-> Nx.Rng.ptree @-> returns tensor) mask_grad
  in
  let masks = g (rows devs2 (Nx.ones f32 [| 2; 16 |])) key in
  for i = 0 to 1 do
    check_arr ~eps:0.0
      ~msg:(Printf.sprintf "lane %d mask is fold_in key %d" i i)
      (to_arr (Nx.cast f32 (draw (Nx.Rng.fold_in key i))))
      (Nx.slice [ Nx.I i ] masks)
  done;
  is_true ~msg:"per-device dropout masks are decorrelated"
    (to_arr (Nx.slice [ Nx.I 0 ] masks) <> to_arr (Nx.slice [ Nx.I 1 ] masks))

(* A program over several devices declines to stage a scan (its probe answer is
   [false]): the recurrence and its gradient unroll into the program. The scan
   folds the columns, so splitting the rows commutes with it and the gathered
   gradient equals the single-device one. *)
let test_grad_through_scan_over_devices () =
  let loss x =
    let n = (Nx.shape x).(0) in
    let _, ys =
      Rune.scan'
        ~f:(fun c col ->
          let c = Nx.tanh (Nx.add c col) in
          (c, Nx.mul c c))
        ~init:(Nx.zeros f32 [| n |]) (Nx.transpose x)
    in
    Nx.sum ys
  in
  let grads x = Rune.grad Nx.Ptree.tensor loss x in
  let x = m46 () in
  let g = Rune.jit Nx.Ptree.(tensor @-> returns tensor) grads in
  check_arr ~msg:"split grads" (to_arr (g x)) (g (rows devs2 x))

(* The DP microbench: a 2-layer MLP train step (value_and_grad + SGD inside the
   compiled function), the parameters entering from the host as a copy on each
   device and the batch split over 2 devices. The 10-step loss trajectory
   matches one device at the same effective batch. *)

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

let mlp_split s = { s with xb = rows devs2 s.xb; yb = rows devs2 s.yb }

let trajectory ?(start = Fun.id) step0 =
  let s = ref (start (mlp_init ())) in
  Array.init 10 (fun _ ->
      let s', l = step0 !s in
      s := s';
      scalar l)

let test_dp_training_matches_jit () =
  let step () =
    Rune.jit Nx.Ptree.(mlp_ptree @-> returns (pair mlp_ptree tensor)) mlp_step
  in
  let one = trajectory (step ()) in
  let split = trajectory ~start:mlp_split (step ()) in
  Array.iteri
    (fun i l ->
      equal ~msg:(Printf.sprintf "loss at step %d" i) (float 1e-6) l split.(i))
    one;
  (* Later steps run entirely on resident state: nothing moves to the
     devices. *)
  let pstep = step () in
  let s1, _ = pstep (mlp_split (mlp_init ())) in
  Rune.reset_jit_stats ();
  let s2, _ = pstep s1 in
  let stats = Rune.jit_stats () in
  equal ~msg:"resident training step moves no bytes to device" int 0
    stats.bytes_to_device;
  ignore (Sys.opaque_identity s2)

let tests =
  [
    group "numerics"
      [
        test "elementwise+matmul+reduce matches one device on 2"
          test_matches_jit_2dev;
        test "elementwise+matmul+reduce matches one device on 4"
          test_matches_jit_4dev;
        test "an elementwise chain is byte-equal to one device"
          test_elementwise_byte_equal;
        test "split along axis 1" test_split_axis_1;
        test "a new shape retraces" test_retrace_on_new_shape;
      ];
    group "placement"
      [
        test "a mean loss over a split batch matches one device"
          test_dp_loss_matches_jit;
        test "grad over a split batch allreduces" test_grad_over_a_split_batch;
        test "grad through max keepdims matches one device"
          test_grad_max_keepdims;
        test "grad through sum keepdims matches one device"
          test_grad_sum_keepdims;
        test "grad through mean keepdims matches one device"
          test_grad_mean_keepdims;
        test "a window write on the split axis"
          test_set_window_on_the_split_axis;
        test "an indexed write into a split value"
          test_indexed_write_into_a_split_value;
        test "a traced window write on the split axis"
          test_set_traced_window_on_the_split_axis;
      ];
    group "residency"
      [
        test "a split output in eager code" test_split_output_in_eager_code;
        test "operations over a batch split" test_batch_split_operations;
        test "a roll by one slice" test_roll_by_a_slice;
        test "a feedback call moves no bytes" test_feedback_moves_no_bytes;
        test "copies feed back without transfer" test_replicated_feedback;
        test "a moved output feeds a call" test_a_moved_output_feeds_a_call;
        test "pass-through outputs gather on read" test_pass_through_output;
        test "a placed capture is bound on the devices"
          test_placed_capture_is_bound;
        test "a moved split output" test_moved_split_output;
        test "one slice of a split storage" test_one_slice_of_a_split_storage;
      ];
    group "consumption"
      [
        test "a split state loop is bounded at two generations"
          test_consume_split_state;
        test "consumed storage is lent on every device"
          test_consumed_storage_is_lent;
        test "consuming copies releases every copy"
          test_consume_replicated_releases_every_copy;
        test "a capture of consumed storage raises"
          test_a_capture_of_consumed_storage_raises;
      ];
    group "errors"
      [
        test "empty devices raise" test_empty_devices;
        test "the host in a device list raises" test_host_in_a_device_list;
        test "mixed backends raise" test_mixed_backends;
        test "a split that does not divide raises"
          test_a_split_that_does_not_divide;
        test "a split axis out of range raises" test_a_split_axis_out_of_range;
      ];
    group "composition"
      [
        test "grad over a program runs eagerly"
          test_grad_over_a_program_runs_eagerly;
        test "dropout per device decorrelates masks"
          test_dropout_per_device_decorrelates;
        test "two collectively reduced outputs" test_two_collective_outputs;
        test "grad through a scan unrolls into the program"
          test_grad_through_scan_over_devices;
      ];
    group "training"
      [
        test "data-parallel MLP training follows one device"
          test_dp_training_matches_jit;
      ];
  ]

let () = run "rune device lists" tests
