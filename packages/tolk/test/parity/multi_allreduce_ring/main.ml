(* Parity case: RING=2-forced ring allreduce on 4 devices.

   [a] is sharded on axis 0 across 4 CPU devices and summed over axis 0.
   RING=2 forces the ring strategy: the reduce-scatter walks each chunk
   around the ring accumulating at each hop, then the allgather chains
   copies device to device — visible as per-chunk shrink/copy/add
   kernels, unlike the naive path's one combine per device.
   LATE_ALLREDUCE=0 (set by the dune rule alongside RING) expands the
   allreduce inline during the multi rewrite so its kernels are visible
   in the schedule.

   Backends are limited to cpu and cuda: kernel-name counters are shared
   across backends, so the reference must be generated with exactly the
   backends the OCaml side renders.

   Paired with main.py. Run `uv run main.py` to regenerate *.expected. *)

open Tolk_uop
module U = Uop

let backends =
  List.filter
    (fun (name, _) -> name = "cpu" || name = "cuda")
    Helpers.all_backends

let devices = [ "CPU:0"; "CPU:1"; "CPU:2"; "CPU:3" ]

let build () =
  let a = Helpers.mk_param_multi ~idx:0 ~devices ~axis:0 [ 8; 64 ] in
  Helpers.wrap_sink [ U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 0 ] ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
