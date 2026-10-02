(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Time of one compiled MoE block at the shapes of gpt-oss-20b, on random packed
   weights: no checkpoint is read.

   Usage: bench.exe [--jit DEVICE] [--tokens N] [--steps N]. Every step feeds
   fresh random tokens, so routing varies, and reads the output back, so the
   device has finished. *)

let experts = 32
let k = 4
let width = 2880
let intermediate = 2880
let limit = 7.0

let random_bytes shape draw =
  let n = Array.fold_left ( * ) 1 shape in
  let a = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n in
  for i = 0 to n - 1 do
    Bigarray.Array1.unsafe_set a i (draw ())
  done;
  Nx.of_bigarray (Bigarray.reshape (Bigarray.genarray_of_array1 a) shape)

let random_floats ~scale shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create Nx.float32 shape
    (Array.init n (fun _ -> scale *. (Random.float 2.0 -. 1.0)))

let packed ~inputs ~outputs =
  let groups = inputs / 32 in
  let scales =
    random_bytes [| experts; outputs; groups |] (fun () -> 118 + Random.int 6)
  in
  Moe.Quant
    (Nx_quant.mxfp4 ~scales
       (random_bytes
          [| experts; outputs; inputs / 2 |]
          (fun () -> Random.bits () land 255)))

let seconds f =
  let t0 = Unix.gettimeofday () in
  let v = f () in
  (v, Unix.gettimeofday () -. t0)

(* The devices [s] names, as in ["CUDA:0,CUDA:1"], opened. *)
let devices_of s =
  Result.fold ~ok:Nx.Device.all ~error:failwith (Nx.Device.of_string s)

let () =
  let jit = ref "METAL" in
  let tokens = ref 1 and steps = ref 20 in
  Arg.parse
    [
      ("--jit", Arg.Set_string jit, "Device (default METAL)");
      ("--tokens", Arg.Set_int tokens, "Tokens per call (default 1)");
      ("--steps", Arg.Set_int steps, "Timed calls (default 20)");
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "bench.exe [--jit DEVICE] [--tokens N] [--steps N]";
  Random.init 0;
  let router =
    {
      Kaun.Linear.w = random_floats ~scale:0.05 [| width; experts |];
      b = Some (random_floats ~scale:0.05 [| experts |]);
    }
  in
  let p, building =
    seconds (fun () ->
        {
          Moe.gate_up = packed ~inputs:width ~outputs:(2 * intermediate);
          gate_up_bias =
            random_floats ~scale:0.05 [| experts; 2 * intermediate |];
          down = packed ~inputs:intermediate ~outputs:width;
          down_bias = random_floats ~scale:0.05 [| experts; width |];
        })
  in
  Printf.printf "random weights built in %.1f s\n%!" building;
  let device = List.hd (devices_of !jit) in
  let f =
    Rune.jit' (fun x ->
        Moe.apply ~limit p (Moe.route ~k (Kaun.Linear.apply router x)) x)
  in
  let run () =
    let x =
      Nx.place (Nx.Placement.on [ device ])
        (random_floats ~scale:1.0 [| !tokens; width |])
    in
    let y, t = seconds (fun () -> Nx.to_array (f x)) in
    if not (Array.for_all Float.is_finite y) then failwith "non-finite output";
    t
  in
  Printf.printf "first call (trace, compile, upload, run): %.2f s\n%!" (run ());
  Printf.printf "second call: %.1f ms\n%!" (1e3 *. run ());
  let before = Nx_device.stats device in
  let times = Array.init !steps (fun _ -> run ()) in
  Array.sort compare times;
  let stats = Nx_device.Stats.diff before (Nx_device.stats device) in
  Printf.printf
    "%s, %d tokens per call, %d calls: min %.1f ms, median %.1f ms, max %.1f ms\n"
    !jit !tokens !steps
    (1e3 *. times.(0))
    (1e3 *. times.(!steps / 2))
    (1e3 *. times.(!steps - 1));
  Printf.printf "bytes to device %d, from device %d\n%!"
    (Nx_device.Stats.bytes_in stats)
    (Nx_device.Stats.bytes_out stats)
