(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jit on the Metal device: kernels compile and run on the GPU, data moves
   through copies. Compiled only on macOS. *)

open Windtrap
open Rune_test_support.Support

(* A virtual GPU, such as a CI runner's paravirtual device, runs every kernel
   but offers no queue capability, so compiled calls replay kernel by kernel
   there. Batched submission is checked only where the device has it. *)
let metal_queues =
  lazy (Option.is_some (Tolk.Device.queue (Tolk.Device.get "METAL")))

let queues_used ~msg dispatched =
  if Lazy.force metal_queues then is_true ~msg dispatched

let test_elementwise_on_metal () =
  let f x = Nx.tanh (Nx.add (Nx.mul x x) x) in
  let g = Rune.jit' ~device:"METAL" f in
  let x = vec32 [| 1.0; -2.0; 0.5 |] in
  check_arr ~msg:"first call" (to_arr (f x)) (g x);
  check_arr ~msg:"replay" (to_arr (f x)) (g x)

(* Metal has one device: its name has no index. *)
let test_one_metal_device () =
  let metal = Rune.device "METAL" in
  is_true ~msg:"index 0 is the device" (Rune.device "metal:0" == metal);
  is_true ~msg:"its backend's devices"
    (List.equal ( == ) [ metal ] (Rune.devices "METAL"));
  raises_match
    (function Invalid_argument _ -> true | _ -> false)
    (fun () -> Rune.device "METAL:1")

let test_matmul_grad_on_metal () =
  let w = Nx.create f32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let f x = Nx.sum (Nx.matmul x w) in
  let g = Rune.jit' ~device:"METAL" (fun x -> Rune.grad' f x) in
  let x = Nx.create f32 [| 2; 3 |] [| 1.0; 0.0; -1.0; 0.5; 2.0; 1.0 |] in
  check_arr ~msg:"grad through metal jit" (to_arr (Rune.grad' f x)) (g x)

(* Multi-kernel compiled traces replay as batched compiled queues: the kernels are
   recorded into an indirect command buffer on the first call and later calls
   patch the rebound buffers (fresh outputs, resident inputs) into it instead of
   launching each kernel individually. *)
let test_queue_batched_replay () =
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
  let launches0 = !Tolk.Realize.queue_submissions in
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
  queues_used ~msg:"every call dispatched a compiled queue"
    (!Tolk.Realize.queue_submissions - launches0 >= 3);
  let x = Nx.create f32 [| 4; 4 |] (Array.init 16 (fun i -> float_of_int i)) in
  check_arr ~msg:"a resident output feeds the next call"
    (to_arr (f (f x)))
    (g (g x))

(* A staged scan's body replays as one compiled queue per iteration, the slot
   buffers rebound between iterations patched into it, instead of launching its
   kernels one by one. *)
let test_scan_body_replays_as_a_queue () =
  let w =
    Nx.create f32 [| 4; 4 |]
      (Array.init 16 (fun i -> (float_of_int (i mod 5) /. 4.0) -. 0.5))
  in
  let fold xs =
    Rune.scan'
      ~f:(fun c x ->
        let c =
          Nx.tanh
            (Nx.add
               (Nx.matmul (Nx.reshape [| 1; 4 |] c) w |> Nx.reshape [| 4 |])
               x)
        in
        (c, Nx.sum (Nx.mul c c)))
      ~init:(Nx.zeros f32 [| 4 |]) xs
  in
  let f xs = snd (fold xs) in
  let g = Rune.jit' ~device:"METAL" f in
  let xs =
    Nx.create f32 [| 6; 4 |] (Array.init 24 (fun i -> float_of_int i /. 24.0))
  in
  check_arr ~msg:"first call" (to_arr (f xs)) (g xs);
  let launches0 = !Tolk.Realize.queue_submissions in
  check_arr ~msg:"replay" (to_arr (f xs)) (g xs);
  queues_used ~msg:"one queue submission per iteration"
    (!Tolk.Realize.queue_submissions - launches0 >= 6)

(* A linked queue keeps its intermediates' buffers alive, so it must not
   outlive the compiled function it belongs to. *)
let test_command_storage_released_with_its_function () =
  let run c =
    let g =
      Rune.jit' ~device:"METAL" (fun x ->
          Nx.add_s (Nx.matmul (Nx.tanh (Nx.matmul x x)) x) c)
    in
    let x = Nx.create f32 [| 4; 4 |] (Array.init 16 float_of_int) in
    ignore (to_arr (g x));
    ignore (to_arr (g x))
  in
  run 0.;
  full_major ();
  let base = !Tolk.Device.Buffer.mem_used in
  let launches0 = !Tolk.Realize.queue_submissions in
  for i = 1 to 4 do
    run (float_of_int i)
  done;
  queues_used ~msg:"the calls recorded compiled queues"
    (!Tolk.Realize.queue_submissions - launches0 >= 8);
  full_major ();
  equal ~msg:"no command storage outlives its function" int base
    !Tolk.Device.Buffer.mem_used

(* Placed weights. The compiled trace binds their buffers as its constants, and
   the batched replay reads them on every call with nothing uploaded. *)

let delta f =
  let s0 = Rune.jit_stats () in
  let r = f () in
  let s1 = Rune.jit_stats () in
  (r, s1.bytes_to_device - s0.bytes_to_device)

let on_metal x = Nx.place (Nx.Placement.device (Rune.device "METAL")) x

let weights () =
  ( Nx.create f32 [| 4; 4 |]
      (Array.init 16 (fun i -> (float_of_int (i mod 5) /. 4.0) -. 0.5)),
    Nx.create f32 [| 4; 4 |]
      (Array.init 16 (fun i -> float_of_int (i mod 3) -. 1.0)) )

let test_placed_weights_bind () =
  let w1, w2 = weights () in
  let f w1 w2 x = Nx.matmul (Nx.tanh (Nx.matmul x w1)) w2 in
  let p1 = on_metal w1 in
  let p2 = on_metal w2 in
  let g = Rune.jit' ~device:"METAL" (f p1 p2) in
  let launches0 = !Tolk.Realize.queue_submissions in
  List.iter
    (fun v ->
      let x = Nx.create f32 [| 2; 4 |] (Array.make 8 v) in
      let y, up = delta (fun () -> g x) in
      equal ~msg:"only the input is uploaded" int (Nx.nbytes x) up;
      check_arr ~msg:"matches eager" (to_arr (f w1 w2 x)) y)
    [ 0.5; -1.0; 2.0 ];
  queues_used ~msg:"the calls replayed as compiled queues"
    (!Tolk.Realize.queue_submissions - launches0 >= 3);
  check_arr ~msg:"a bound weight reads back" (to_arr w1) p1;
  is_true ~msg:"and keeps its buffer" (bound_by 1 p1);
  let x = Nx.create f32 [| 2; 4 |] (Array.make 8 0.25) in
  check_arr ~msg:"after the read" (to_arr (f w1 w2 x)) (g x);
  (* A second compiled function shares the buffers. *)
  let h = Rune.jit' ~device:"METAL" (fun x -> Nx.add (Nx.matmul x p1) x) in
  let y, up = delta (fun () -> h x) in
  equal ~msg:"a second function uploads its input only" int (Nx.nbytes x) up;
  check_arr ~msg:"second function" (to_arr (Nx.add (Nx.matmul x w1) x)) y

(* One float32 tensor as a tree. *)
module Single = struct
  type t = Nx.float32_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) x = f x

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) x = f x
end

let raises_donated f =
  raises_match
    (function
      | Invalid_argument msg ->
          String.starts_with ~prefix:"this value was donated" msg
      | _ -> false)
    (fun () -> ignore (f ()))

(* A dtype Metal cannot hold raises where it would be placed: at the upload, and
   at an eager operation whose result has it. *)
let test_unsupported_dtype_raises_at_placement () =
  let cannot_hold f =
    raises_match
      (function
        | Invalid_argument msg ->
            String.ends_with ~suffix:"cannot hold float64" msg
        | _ -> false)
      f
  in
  cannot_hold (fun () -> on_metal (Nx.create f64 [| 3 |] [| 1.; 2.; 3. |]));
  let p = on_metal (vec32 [| 1.0; 2.0; 3.0 |]) in
  cannot_hold (fun () -> Nx.cast Nx.float64 p);
  cannot_hold (fun () -> Nx.cast Nx.float64 (Nx.sum p))

(* A view of part of a placed storage binds it at any element offset, whatever
   the dtype's width, and a strided one is movement in the program. *)
let test_placed_views_bind () =
  let metal = Nx.Placement.device (Rune.device "METAL") in
  let check (type a b) ~msg (dt : (a, b) Nx.dtype) =
    let n = 4097 in
    let x =
      Nx.cast dt
        (Nx.create f32 [| n |]
           (Array.init n (fun i -> float_of_int (i mod 61))))
    in
    let p = Nx.place metal x in
    let f v = Nx.add v v in
    let view = Nx.slice [ Nx.R (1, n) ] p in
    let y, up = delta (fun () -> Rune.jit' ~device:"METAL" f view) in
    equal ~msg:(msg ^ ": nothing is uploaded") int 0 up;
    let expected = f (Nx.slice [ Nx.R (1, n) ] x) in
    is_true ~msg:(msg ^ ": value")
      (Nx.item [] (Nx.all (Nx.equal (Nx.place Nx.Placement.host y) expected)))
  in
  check ~msg:"float32" f32;
  check ~msg:"bfloat16" Nx.bfloat16;
  check ~msg:"int8" Nx.int8;
  let w1, _ = weights () in
  let p = Nx.place metal w1 in
  let x = Nx.create f32 [| 2; 4 |] (Array.make 8 1.0) in
  let g = Rune.jit' ~device:"METAL" (fun w -> Nx.matmul x w) in
  let y, up = delta (fun () -> g (Nx.matrix_transpose p)) in
  equal ~msg:"a transpose: only the capture is uploaded" int (Nx.nbytes x) up;
  check_arr ~msg:"a transpose" (to_arr (Nx.matmul x (Nx.matrix_transpose w1))) y

(* A compiled function refuses a dtype its device cannot hold before it runs: a
   host input, a value it computes, and a value it captures. *)
let test_unsupported_dtype_raises_before_a_call () =
  let x = Nx.create f64 [| 3 |] [| 1.; 2.; 3. |] in
  let (), up =
    delta (fun () ->
        raises_match
          (function
            | Invalid_argument msg ->
                msg
                = "Rune.jit: input leaf 0 is float64, which METAL cannot hold"
            | _ -> false)
          (fun () -> Rune.jit' ~device:"METAL" (fun x -> Nx.mul_s x 2.0) x))
  in
  equal ~msg:"an input: nothing is uploaded" int 0 up;
  let cannot_hold f =
    raises_match
      (function
        | Rune.Jit_error msg ->
            String.ends_with ~suffix:"float64, which METAL cannot hold" msg
        | _ -> false)
      f
  in
  cannot_hold (fun () ->
      Rune.jit' ~device:"METAL"
        (fun x -> Nx.cast Nx.float32 (Nx.mul_s (Nx.cast Nx.float64 x) 2.0))
        (vec32 [| 1.0; 2.0 |]));
  cannot_hold (fun () ->
      Rune.jit' ~device:"METAL"
        (fun y -> Nx.add y (Nx.cast Nx.float32 x))
        (vec32 [| 1.0; 2.0; 3.0 |]))

(* On the default device, a dtype it cannot hold waits for the captures: one on
   CPU:1, which holds float64, moves the program there, whichever the trace
   meets first. *)
let test_capture_moves_past_a_dtype () =
  is_true ~msg:"METAL is the default"
    (Rune.default_device () == Rune.device "METAL");
  let cpu1 = Nx.Placement.device (Rune.device "CPU:1") in
  let w = Nx.place cpu1 (vec32 [| 1.0; 2.0; 3.0 |]) in
  let x = vec32 [| 4.0; 5.0; 6.0 |] in
  let expected = [| 5.0; 7.0; 9.0 |] in
  let on_cpu1 ~msg y =
    is_true ~msg:(msg ^ ": on CPU:1") (Nx.Placement.equal cpu1 (Nx.placement y));
    check_arr ~msg expected y
  in
  on_cpu1 ~msg:"the input's float64 first"
    (Rune.jit'
       (fun x ->
         let a = Nx.cast Nx.float64 x in
         let b = Nx.cast Nx.float64 w in
         Nx.cast f32 (Nx.add a b))
       x);
  on_cpu1 ~msg:"the capture's float64 first"
    (Rune.jit'
       (fun x ->
         let b = Nx.cast Nx.float64 w in
         let a = Nx.cast Nx.float64 x in
         Nx.cast f32 (Nx.add a b))
       x);
  on_cpu1 ~msg:"a float64 input"
    (Nx.cast f32
       (Rune.jit'
          (fun x -> Nx.add x (Nx.cast Nx.float64 w))
          (Nx.cast Nx.float64 x)));
  raises_match
    (function Rune.Jit_error _ -> true | _ -> false)
    (fun () -> Rune.jit' (fun x -> Nx.cast f32 (Nx.cast Nx.float64 x)) x)

let test_bound_input_is_not_donated () =
  let w1, _ = weights () in
  let p = on_metal w1 in
  let g = Rune.jit' ~device:"METAL" (fun x -> Nx.matmul x p) in
  let x = Nx.create f32 [| 2; 4 |] (Array.make 8 1.0) in
  ignore (g x);
  let step =
    Rune.jit_step ~device:"METAL"
      (module Nx.Ptree)
      (module Single)
      (fun _ m -> Nx.mul_s m 2.0)
      (Nx.Ptree.list [])
  in
  let y, up = delta (fun () -> step p) in
  equal ~msg:"the bound input seeds with no transfer" int 0 up;
  check_arr ~msg:"result" (to_arr (Nx.mul_s w1 2.0)) y;
  check_arr ~msg:"the bound value is still readable" (to_arr w1) p;
  check_arr ~msg:"and still the constant" (to_arr (Nx.matmul x w1)) (g x)

(* A step reads placed weights and consumes its state: the weights stay resident
   and readable, the state is written over its own storage. *)
let test_step_reads_weights_consumes_state () =
  let w1, _ = weights () in
  let w = on_metal w1 in
  let f w x = Nx.add_s (Nx.mul x x) (Nx.item [] (Nx.mean w)) in
  let step =
    Rune.jit_step ~device:"METAL"
      (module Single)
      (module Single)
      (fun w x -> Nx.add (Nx.mul x x) (Nx.mean w))
  in
  let x0 =
    Nx.create f32 [| 4; 4 |] (Array.init 16 (fun i -> float_of_int i /. 16.0))
  in
  let x1 = step w x0 in
  let before = (Rune.jit_stats ()).reused_bytes in
  let x2, up = delta (fun () -> step w x1) in
  equal ~msg:"resident leaves upload nothing" int 0 up;
  equal ~msg:"the state is written over its own storage" int (Nx.nbytes x0)
    ((Rune.jit_stats ()).reused_bytes - before);
  check_arr ~eps:1e-5 ~msg:"two steps" (to_arr (f w1 (f w1 x0))) x2;
  raises_donated (fun () -> to_arr x1);
  check_arr ~msg:"the weights are readable" (to_arr w1) w

(* A call returns while its kernels may still run; a read waits for them. *)
let test_read_after_call_waits () =
  let n = 512 in
  let f x = Nx.add_s (Nx.matmul (Nx.tanh x) (Nx.transpose x)) 1.0 in
  let g = Rune.jit' ~device:"METAL" f in
  let x =
    Nx.create f32 [| n; n |]
      (Array.init (n * n) (fun i -> float_of_int (i mod 13) /. 13.0))
  in
  check_arr ~eps:1e-2 ~msg:"right after one call" (to_arr (f x)) (g x);
  let step =
    Rune.jit' ~device:"METAL" (fun x -> Nx.add_s (Nx.mul_s x 0.5) 1.0)
  in
  let h = ref (on_metal (vec32 (Array.make 4096 0.0))) in
  for _ = 1 to 50 do
    h := step !h
  done;
  let expected = 2.0 -. (2.0 *. (0.5 ** 50.0)) in
  check_arr ~eps:1e-6 ~msg:"after fifty unread calls" (Array.make 4096 expected)
    !h

module Pair = struct
  type t = { u : Nx.float32_t; v : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p =
    { u = f p.u; v = f p.v }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { u = f p.u q.u; v = f p.v q.v }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) p =
    f p.u;
    f p.v
end

(* Two programs take turns on one consumed state, with no wait between their
   calls: each consumed buffer is reused only by work queued after the kernels
   that read it. *)
let test_two_programs_alternate () =
  let n = 64 in
  let mix (p : Pair.t) =
    let a = Nx.tanh (Nx.matmul p.u p.v) in
    let scale = Nx.add_s (Nx.sum ~axes:[ 1 ] ~keepdims:true (Nx.abs a)) 1.0 in
    {
      Pair.u = Nx.add p.u (Nx.div a scale);
      v = Nx.sub p.v (Nx.mul_s (Nx.transpose a) 0.1);
    }
  in
  let fold (p : Pair.t) =
    let m = Nx.mean ~axes:[ 0 ] ~keepdims:true (Nx.matmul p.v p.u) in
    let u = Nx.mul_s (Nx.sin (Nx.add p.u m)) 0.5 in
    { Pair.u; v = Nx.add (Nx.mul_s p.v 0.9) (Nx.matmul u u) }
  in
  let compile f =
    Rune.jit_step ~device:"METAL"
      (module Nx.Ptree)
      (module Pair)
      (fun _ p -> f p)
      (Nx.Ptree.list [])
  in
  let mix' = compile mix and fold' = compile fold in
  let init k =
    Nx.create f32 [| n; n |]
      (Array.init (n * n) (fun i -> sin (float_of_int ((k * i) + 1)) /. 8.0))
  in
  let p = { Pair.u = init 3; v = init 7 } in
  let e = ref p and h = ref p in
  for i = 1 to 12 do
    let f, f' = if i mod 2 = 0 then (fold, fold') else (mix, mix') in
    e := f !e;
    h := f' !h
  done;
  check_arr ~eps:1e-3 ~msg:"u" (to_arr !e.Pair.u) !h.Pair.u;
  check_arr ~eps:1e-3 ~msg:"v" (to_arr !e.Pair.v) !h.Pair.v

(* Programs run in turn share the device's arena. A program recorded as a device
   queue over a smaller arena is re-patched onto the grown one, and a second
   program of the same size allocates no arena of its own. *)
let test_programs_share_an_arena () =
  let program ~n act =
    let f x =
      let a = act (Nx.matmul x (Nx.transpose x)) in
      Nx.sum ~axes:[ 1 ] (Nx.matmul a a)
    in
    let x k =
      Nx.create f32 [| n; 8 |]
        (Array.init (n * 8) (fun i -> sin (float_of_int ((k * i) + 1)) /. 4.0))
    in
    (f, Rune.jit' ~device:"METAL" f, x)
  in
  let device_bytes () =
    Option.value ~default:0
      (Hashtbl.find_opt Tolk.Helpers.Global_counters.mem_used_per_device "METAL")
  in
  let f, f', x = program ~n:256 Nx.tanh in
  check_arr ~eps:1e-2 ~msg:"small" (to_arr (f (x 1))) (f' (x 1));
  check_arr ~eps:1e-2 ~msg:"small, replayed" (to_arr (f (x 2))) (f' (x 2));
  let n = 1024 in
  let g, g', y = program ~n Nx.sin in
  check_arr ~eps:1e-2 ~msg:"large" (to_arr (g (y 1))) (g' (y 1));
  check_arr ~eps:1e-2 ~msg:"small, on the grown arena"
    (to_arr (f (x 3)))
    (f' (x 3));
  let h, h', _ = program ~n (fun a -> Nx.mul_s (Nx.sin a) 0.5) in
  let before = device_bytes () in
  check_arr ~eps:1e-2 ~msg:"another large" (to_arr (h (y 2))) (h' (y 2));
  is_true ~msg:"it allocates no arena of its own"
    (device_bytes () - before < n * n * 4);
  for k = 4 to 6 do
    check_arr ~eps:1e-2 ~msg:"small, in turn" (to_arr (f (x k))) (f' (x k));
    check_arr ~eps:1e-2 ~msg:"large, in turn" (to_arr (g (y k))) (g' (y k));
    check_arr ~eps:1e-2 ~msg:"other large, in turn"
      (to_arr (h (y (k + 3))))
      (h' (y (k + 3)))
  done

let test_capture_resident_elsewhere () =
  let w1, _ = weights () in
  let p = Nx.place (Nx.Placement.device (Rune.device "CPU:1")) w1 in
  let g = Rune.jit' ~device:"METAL" (fun x -> Nx.matmul x p) in
  let x = Nx.create f32 [| 2; 4 |] (Array.make 8 1.0) in
  let (), up =
    delta (fun () ->
        raises_match
          (function
            | Invalid_argument msg ->
                String.starts_with
                  ~prefix:"Rune.jit: a captured value is on CPU:1" msg
            | _ -> false)
          (fun () -> g x))
  in
  equal ~msg:"nothing is uploaded" int 0 up

(* A weight over a mapped file is placed by reading the file. *)
let test_place_from_a_mapped_file () =
  let n = 4096 in
  let path = Filename.temp_file "rune_metal_mapped_" ".bin" in
  Fun.protect
    ~finally:(fun () ->
      full_major ();
      try Sys.remove path with Sys_error _ -> ())
    (fun () ->
      let values = Array.init n (fun i -> float_of_int (i mod 97) /. 8.0) in
      let oc = open_out_bin path in
      let bytes = Bytes.create (4 * n) in
      Array.iteri
        (fun i v -> Bytes.set_int32_le bytes (4 * i) (Int32.bits_of_float v))
        values;
      output_bytes oc bytes;
      close_out oc;
      let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
      let stat = Unix.fstat fd in
      let mapping =
        Nx_buffer.of_bigarray1
          (Bigarray.array1_of_genarray
             (Unix.map_file fd Bigarray.int8_unsigned Bigarray.c_layout false
                [| -1 |]))
      in
      Unix.close fd;
      Nx_buffer.register_file
        { path; size = 4 * n; mtime = stat.st_mtime; inode = stat.st_ino }
        mapping;
      let w =
        Nx.of_buffer
          (Nx_buffer.reinterpret Nx_buffer.Float32 mapping)
          ~shape:[| 64; 64 |]
      in
      let placed = on_metal (Nx.matrix_transpose w) in
      let g = Rune.jit' ~device:"METAL" (fun x -> Nx.matmul x placed) in
      let x = Nx.create f32 [| 2; 64 |] (Array.make 128 0.5) in
      check_arr ~msg:"matches eager"
        (to_arr (Nx.matmul x (Nx.matrix_transpose w)))
        (g x))

(* Metal flushes float32 subnormals to zero when it compares floats. A compiled
   sort keeps them, in order, as eager does, and -0 ties with 0. Adding 0 on the
   host clears the sign of zero, which the compiled values do not keep. *)
let test_sort_keeps_subnormals () =
  let x =
    vec32
      [| 1e-45; -1e-45; 0.; -0.; 1e-40; -1e-40; 3.4028235e38; -3.4028235e38 |]
  in
  let sort x =
    let values, indices = Nx.sort ~axis:0 x in
    Nx.stack [ values; Nx.cast f32 indices ]
  in
  let unsigned_zero t = Array.map (fun v -> v +. 0.) (to_arr t) in
  equal ~msg:"sorted values over their positions" (array float_exact)
    (unsigned_zero (sort x))
    (unsigned_zero (Rune.jit' ~device:"METAL" sort x))

(* Reads and moves keep a placed value where it is (RFC 0005, Laws 3 and 4), and
   a loop whose state starts on the host compiles once. *)

let read_bytes f =
  let s0 = Rune.jit_stats () in
  let r = f () in
  (r, (Rune.jit_stats ()).bytes_from_device - s0.bytes_from_device)

let test_item_on_resident_logits () =
  let vocab = 4096 in
  let w =
    Nx.create f32 [| 8; vocab |]
      (Array.init (8 * vocab) (fun i -> float_of_int (i mod 17) /. 17.0))
  in
  let placed = on_metal w in
  let head = Rune.jit' ~device:"METAL" (fun h -> Nx.matmul h placed) in
  let h = Nx.create f32 [| 1; 8 |] (Array.init 8 float_of_int) in
  let logits = head h in
  let v, down = read_bytes (fun () -> Nx.item [ 0; 5 ] logits) in
  equal ~msg:"the element" (float 1e-5) (Nx.item [ 0; 5 ] (Nx.matmul h w)) v;
  equal ~msg:"four bytes move" int 4 down;
  is_true ~msg:"the logits stay resident" (bound_by 0 logits);
  let step = Rune.jit' ~device:"METAL" (fun l -> Nx.mul_s l 2.0) in
  let (_ : Nx.float32_t), up = delta (fun () -> step logits) in
  equal ~msg:"and feed a call with no upload" int 0 up

let test_move_to_host_keeps_its_source () =
  let w1, _ = weights () in
  let p = on_metal w1 in
  let h = Nx.place Nx.Placement.host p in
  is_true ~msg:"a host copy"
    (Nx.Placement.equal Nx.Placement.host (Nx.placement h));
  check_arr ~msg:"its elements" (to_arr w1) h;
  check_arr ~msg:"the source is still readable" (to_arr w1) p;
  is_true ~msg:"and resident" (bound_by 0 p);
  let g = Rune.jit' ~device:"METAL" (fun x -> Nx.mul_s x 2.0) in
  let (_ : Nx.float32_t), up = delta (fun () -> g p) in
  equal ~msg:"and feeds a call with no upload" int 0 up

let test_mixed_placements_raise () =
  let p = on_metal (vec32 [| 1.0; 2.0 |]) in
  let q =
    Nx.place (Nx.Placement.device (Rune.device "CPU:1")) (vec32 [| 3.0; 4.0 |])
  in
  let (), down =
    read_bytes (fun () ->
        raises_match
          (function
            | Invalid_argument msg ->
                String.ends_with ~suffix:"place one of them" msg
            | _ -> false)
          (fun () -> Nx.add p q))
  in
  equal ~msg:"nothing is read" int 0 down;
  let y = Nx.add p (vec32 [| 1.0; 1.0 |]) in
  is_true ~msg:"a host operand joins"
    (Nx.Placement.equal (Nx.placement p) (Nx.placement y));
  check_arr ~msg:"and the result is right" [| 2.0; 3.0 |] y

let test_host_started_loop_compiles_once () =
  let traces = ref 0 in
  let step =
    Rune.jit_step
      (module Nx.Ptree)
      (module Single)
      (fun _ x ->
        incr traces;
        Nx.add_s (Nx.mul_s x 0.5) 1.0)
      (Nx.Ptree.list [])
  in
  let x = ref (vec32 (Array.make 64 0.0)) in
  for _ = 1 to 4 do
    x := step !x
  done;
  equal ~msg:"one trace for the host state and the placed ones" int 1 !traces;
  is_true ~msg:"the state is on Metal"
    (Nx.Placement.equal
       (Nx.Placement.device (Rune.device "METAL"))
       (Nx.placement !x));
  check_arr ~msg:"four steps" (Array.make 64 (2.0 -. (2.0 *. (0.5 ** 4.0)))) !x

(* Without the promise of unique indices, thousands of updates aimed at one row
   land in index order on the GPU too: the last [`Set] wins, and [`Add] sums in
   the order that fixes its rounding. *)
let test_scatter_duplicates_on_metal () =
  let updates = 4096 and width = 8 in
  let indices =
    Nx.create Nx.int32 [| updates; width |] (Array.make (updates * width) 1l)
  in
  let values =
    Nx.create f32 [| updates; width |]
      (Array.init (updates * width) (fun i -> float_of_int (i + 1)))
  in
  let t = Nx.zeros f32 [| 2; width |] in
  List.iter
    (fun (name, mode) ->
      let f t = Nx.scatter ~mode ~axis:0 ~indices ~values t in
      check_arr ~msg:name (to_arr (f t)) (Rune.jit' ~device:"METAL" f t))
    [ ("set", `Set); ("add", `Add) ]

(* Radix selection on the GPU picks the positions eager does, NaN, both zeros
   and both infinities included, with a run of ties at the threshold. *)
let test_top_k_on_metal () =
  let st = Random.State.make [| 9 |] in
  let pool = [| Float.nan; -0.; 0.; Float.infinity; Float.neg_infinity |] in
  let data =
    Nx.init Nx.float64 [| 2; 4096 |] (fun _ ->
        match Random.State.int st 4 with
        | 0 -> pool.(Random.State.int st (Array.length pool))
        | 1 -> float_of_int (Random.State.int st 5)
        | _ -> Random.State.float st 8. -. 4.)
  in
  let check (type a b) name (x : (a, b) Nx.t) =
    List.iter
      (fun k ->
        let indices x = Nx.cast f32 (snd (Nx.top_k ~k x)) in
        check_arr
          ~msg:(Printf.sprintf "%s top %d" name k)
          (to_arr (indices x))
          (Rune.jit' ~device:"METAL" indices x))
      [ 17; 512; 2000 ]
  in
  check "float32" (Nx.cast f32 data);
  check "bfloat16" (Nx.cast Nx.bfloat16 data);
  check "int32"
    (Nx.init Nx.int32 [| 2; 4096 |] (fun _ ->
         Int32.of_int (Random.State.int st 200 - 100)));
  (* The GPU flushes subnormals in arithmetic; the keys never pass through it,
     so rows of zeros and subnormals rank as eagerly. *)
  let tiny = [| 0.; -0.; 1e-45; -1e-45; 1e-40; 6e-8; 1e-39 |] in
  let subnormals =
    Nx.init Nx.float64 [| 2; 4096 |] (fun _ ->
        tiny.(Random.State.int st (Array.length tiny)))
  in
  check "float32 subnormals" (Nx.cast f32 subnormals);
  check "bfloat16 subnormals" (Nx.cast Nx.bfloat16 subnormals)

(* Reading a strided view of a placed value copies bits: a signalling NaN and
   its payload survive the read, as they do for a contiguous one. *)
let test_placed_view_keeps_nan_bits () =
  let bits =
    Nx.create Nx.int32 [| 2; 3 |]
      [| 0x7F800001l; 0xFFC00123l; 1l; 0x80000000l; 0x7FA00000l; -1l |]
  in
  let placed = on_metal (Nx.bitcast f32 bits) in
  equal ~msg:"a transposed view" (array int32)
    (Nx.to_array (Nx.transpose bits))
    (Nx.to_array (Nx.bitcast Nx.int32 (Nx.transpose placed)))

(* Metal sorts key and position packed in one int64, as the CPU does. *)
let test_sort_matches_eager () =
  let pieces = [ ([| 2; 513; 3 |], 1); ([| 32_768 |], 0) ] in
  check_sort_pieces f32 pieces Nx.float32;
  check_sort_pieces f32 pieces Nx.bfloat16;
  check_sort_pieces f32 pieces Nx.int32

(* A running maximum or minimum compares keys, not floats, so Metal's flush of
   subnormals in comparisons leaves them in order. *)
let test_scans_keep_subnormals () =
  let x = vec32 [| -1e-40; 1e-40; -2e-40; 3e-40; 0.5; -1e-45 |] in
  List.iter
    (fun (name, f) ->
      equal ~msg:name (array float_exact)
        (to_arr (f x))
        (to_arr (Rune.jit' ~device:"METAL" f x)))
    [ ("cummax", Nx.cummax ~axis:0); ("cummin", Nx.cummin ~axis:0) ]

(* A value with no elements has no storage: an empty input, output or capture
   compiles and replays on Metal, next to values that do have elements. *)
let test_empty_values () =
  let v = vec32 [| 1.0; 2.0; 3.0 |] in
  let empty = Nx.zeros f32 [| 0 |] in
  let check name f x =
    let g = Rune.jit' ~device:"METAL" f in
    for call = 1 to 2 do
      let msg = Printf.sprintf "%s, call %d" name call in
      check_arr ~msg (to_arr (f x)) (g x)
    done
  in
  check "an empty output" (fun v -> Nx.mul_s (Nx.slice [ Nx.R (1, 1) ] v) 2.0) v;
  check "an empty input" (fun x -> Nx.mul_s x 2.0) empty;
  check "an empty input returned" Fun.id empty;
  check "an empty capture returned" (fun _ -> empty) v;
  check "a sum over an empty slice"
    (fun x -> Nx.add (Nx.sum (Nx.slice [ Nx.R (1, 1) ] x)) x)
    v

let tests =
  [
    group "metal device"
      [
        test "Metal has one device" test_one_metal_device;
        test "element-wise chain matches eager" test_elementwise_on_metal;
        test "duplicate scatter updates land in order"
          test_scatter_duplicates_on_metal;
        test "bitcast reads on the GPU the bits eager reads"
          (check_bitcast_matches_eager ~device:"METAL");
        slow "top_k over a row of 2^20 entries on the GPU"
          (check_top_k_long_row ~device:"METAL");
        slow "top_k selects on the GPU what it selects eagerly"
          test_top_k_on_metal;
        slow "sort matches eager" test_sort_matches_eager;
        test "grad inside jit matches eager" test_matmul_grad_on_metal;
        test "multi-kernel traces replay as compiled queues"
          test_queue_batched_replay;
        test "a staged scan body replays as a compiled queue"
          test_scan_body_replays_as_a_queue;
        test "command storage is released with its function"
          test_command_storage_released_with_its_function;
        test "a read after a call waits for it" test_read_after_call_waits;
        test "programs run in turn share an arena" test_programs_share_an_arena;
        test "two programs alternate on one consumed state"
          test_two_programs_alternate;
        test "sort keeps subnormals" test_sort_keeps_subnormals;
        test "empty values have no storage" test_empty_values;
        test "scans keep subnormals" test_scans_keep_subnormals;
      ];
    group "placed weights"
      [
        test "a placed view is read bit for bit" test_placed_view_keeps_nan_bits;
        test "a dtype Metal cannot hold raises at placement"
          test_unsupported_dtype_raises_at_placement;
        test "placed weights bind and replay as compiled queues"
          test_placed_weights_bind;
        test "a bound input is not consumed by donation"
          test_bound_input_is_not_donated;
        test "a step reads its weights and consumes its state"
          test_step_reads_weights_consumes_state;
        test "a capture resident on another device raises"
          test_capture_resident_elsewhere;
        test "placed views bind at any offset" test_placed_views_bind;
        test "a dtype Metal cannot hold raises before a compiled call"
          test_unsupported_dtype_raises_before_a_call;
        test "a capture moves a program past a dtype"
          test_capture_moves_past_a_dtype;
        test "a weight over a mapped file is placed from the file"
          test_place_from_a_mapped_file;
      ];
    group "reads, moves and loops"
      [
        test "item on resident logits reads one element"
          test_item_on_resident_logits;
        test "a move to the host keeps its source"
          test_move_to_host_keeps_its_source;
        test "mixed placements raise" test_mixed_placements_raise;
        test "a loop whose state starts on the host compiles once"
          test_host_started_loop_compiles_once;
      ];
  ]

let () = run "rune jit metal" tests
