(* Parity case: sharded + replicated elementwise add on 2 devices.

   [a] is sharded on axis 0 across CPU:0/CPU:1, [b] is replicated (multi
   device, no axis). The replicated input is sharded symbolically, so the
   kernel indexes it with the [_device_num] variable.

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
  let a = Helpers.mk_param_multi ~idx:0 ~devices ~axis:0 [ 8; 8 ] in
  let b = Helpers.mk_param_multi ~idx:1 ~devices [ 16; 8 ] in
  Helpers.wrap_sink [ U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
