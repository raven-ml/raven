(* Parity case: late (default) allreduce as a precompiled function.

   Same graph as multi_allreduce_naive but with the default
   LATE_ALLREDUCE=1: the ALLREDUCE survives the multi rewrite and
   scheduling wraps it into a precompiled allreduce function. Only the
   per-shard reduce and the copy kernel around the opaque allreduce call
   appear in the extracted schedule.

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
