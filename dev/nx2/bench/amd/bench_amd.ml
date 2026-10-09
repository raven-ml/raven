(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The AMD rows against this machine's baseline. A call runs a row's kernel a
   fixed count of times in one submission and returns once they are done: its
   wall time is their span on the GPU and a submission's round trip.
   gate_amd.exe prints the device times themselves. *)

module S = Nx_amd_support
module D = Nx_array.Dtype

(* Runs per call: about 2 ms of a memory-bound row at 500 GB/s; a peak row's run
   spans about a millisecond itself. *)
let count (p : Rows.prepared) =
  match p.work with
  | Bytes 0 -> 1000
  | Bytes n -> Int.max 1 (1_000_000_000 / n)
  | Flops _ -> 1

let case (r : Rows.row) =
  let setup () =
    let g = S.gpu () in
    let p = r.make g in
    (g, p, count p)
  in
  Thumper.bench_with_setup ~setup r.name (fun (g, (p : Rows.prepared), count) ->
      S.device_time g p.run ~count)

(* The binding's host share of a call: the plan of a contraction alone, and the
   submission of 64 launches of the smallest one, its wait outside the timed
   call. *)
let host =
  let bf16 = D.Any D.Bfloat16 and f32 = D.Any D.Float32 in
  let planner () =
    let a, b, y, _ = Rows.plan (S.gpu ()) bf16 f32 bf16 4096 4096 4096 in
    S.planner ~a ~b ~y
      ~batch:[ (0, 0) ]
      ~contracting:[ (2, 2) ]
      ~acc:(D.code D.Float32) ()
  in
  let plan =
    Thumper.bench_with_setup ~setup:planner "plan/contract-bf16-4096" (fun p ->
        p ())
  in
  let setup () =
    let g = S.gpu () in
    let _, _, _, run = Rows.plan g bf16 f32 bf16 1 1 1 in
    (g, run)
  in
  let fill =
    Thumper.bench_with_setup ~setup "fill/contract-64" (fun (g, run) ->
        S.enqueue g ~count:64 run)
  in
  [ plan; fill ]

let () =
  (* No device opens in this process: thumper's workers open the GPU after it
     forks them. *)
  if Rig_amd_amdgpu.count () = 0 then begin
    print_endline "nx.amd bench: no AMD GPU";
    exit 0
  end;
  S.hold_gpu ();
  exit (Thumper.run "nx_amd" (host @ List.map case Rows.all))
