(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The CUDA rows against this machine's baseline. A call runs a row's kernel a
   fixed count of times behind a hold and returns once they are done: its wall
   time is their span on the GPU and a submission's round trip. gate_cuda.exe
   prints the device times themselves. *)

module S = Nx_cuda_support

(* Runs per call: about 2 ms of a memory-bound row at 500 GB/s; a peak row's run
   spans 1-2 ms itself. *)
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
  (* No CUDA library starts in this process: thumper's workers open the GPU
     after it forks them. *)
  if not (Sys.file_exists "/dev/nvidiactl") then begin
    print_endline "nx.cuda bench: no NVIDIA GPU";
    exit 0
  end;
  S.hold_gpu ();
  exit (Thumper.run "nx_cuda" (List.map case Rows.all))
