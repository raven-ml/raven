(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Learning rate schedules control how the learning rate changes over training.

   A schedule is simply a function [int -> float]: given a 1-based step number,
   it returns the learning rate. This example evaluates several schedules and
   prints their values, then uses warmup + cosine decay in an optimization
   loop. *)

module S = Vega.Schedule

let print_schedule name s steps =
  Printf.printf "  %-30s" name;
  List.iter (fun step -> Printf.printf "  %3d:%.6f" step (s step)) steps;
  Printf.printf "\n"

let sample = [ 1; 25; 50; 75; 100 ]

let () =
  Printf.printf "--- Schedule values at steps %s ---\n"
    (String.concat ", " (List.map string_of_int sample));

  print_schedule "constant 0.01" (S.constant 0.01) sample;

  print_schedule "cosine_decay"
    (S.cosine_decay ~init_value:0.01 ~decay_steps:100 ())
    sample;

  print_schedule "warmup_cosine_decay"
    (S.warmup_cosine_decay ~init_value:0.0 ~peak_value:0.01 ~warmup_steps:25
       ~decay_steps:75 ())
    sample;

  print_schedule "one_cycle"
    (S.one_cycle ~max_value:0.01 ~total_steps:100 ())
    sample;

  print_schedule "piecewise_constant"
    (S.piecewise_constant ~boundaries:[ 30; 70 ] ~values:[ 0.01; 0.001; 0.0001 ])
    sample;

  Printf.printf "\n";

  (* join: sequence two schedules end-to-end *)
  Printf.printf "--- join: linear warmup then cosine decay ---\n";
  let joined =
    S.join
      [
        (20, S.linear ~init_value:0.0 ~end_value:0.01 ~steps:20);
        (80, S.cosine_decay ~init_value:0.01 ~decay_steps:80 ());
      ]
  in
  print_schedule "join [warmup; cosine]" joined sample;

  Printf.printf "\n";

  (* Use warmup + cosine decay in an optimization loop *)
  Printf.printf "--- Adam with warmup_cosine_decay (100 steps) ---\n";
  let lr =
    S.warmup_cosine_decay ~init_value:0.0 ~peak_value:0.01 ~warmup_steps:20
      ~decay_steps:80 ()
  in
  let tx = Vega.adam lr in
  let param = ref (Nx.create Nx.float32 [| 2 |] [| 5.0; -3.0 |]) in
  let st = ref (Vega.init tx !param) in
  for i = 1 to 100 do
    let p, s = Vega.step !st ~grad:!param ~param:!param in
    param := p;
    st := s;
    if i mod 20 = 0 then
      Printf.printf "  step %3d  lr=%.6f  x = %s\n" i (lr i)
        (Nx.data_to_string !param)
  done
