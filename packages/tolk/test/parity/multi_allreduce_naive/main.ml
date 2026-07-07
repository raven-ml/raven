(* Parity case: naive allreduce from a reduce over the sharded axis.

   2 devices, [a] sharded on axis 0, summed over axis 0. The per-shard
   reduce yields one partial per device; with 2 devices the allreduce
   takes the naive path (copy every shard to each device and add).
   LATE_ALLREDUCE=0 (set by the dune rule) expands the allreduce inline
   during the multi rewrite so its kernels are visible in the schedule.

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

let devices = [ "CPU:0"; "CPU:1" ]

let build () =
  let a = Helpers.mk_param_multi ~idx:0 ~devices ~axis:0 [ 8; 16 ] in
  Helpers.wrap_sink [ U.reduce_axis ~src:a ~op:Ops.Add ~axes:[ 0 ] ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
