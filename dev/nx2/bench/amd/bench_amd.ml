(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The AMD rows against this machine's baseline. A call runs a row's kernel a
   fixed count of times in one submission and returns once they are done: its
   wall time is their span on the GPU and a submission's round trip.
   gate_amd.exe prints the device times themselves. *)

module S = Nx_amd_support

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

let () =
  (* No device opens in this process: thumper's workers open the GPU after it
     forks them. *)
  if Rig_amd_amdgpu.count () = 0 then begin
    print_endline "nx.amd bench: no AMD GPU";
    exit 0
  end;
  S.hold_gpu ();
  exit (Thumper.run "nx_amd" (List.map case Rows.all))
