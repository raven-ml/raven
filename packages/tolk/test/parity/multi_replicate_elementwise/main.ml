(* Parity case: replicate (copy-to-tuple) + elementwise on replicated input.

   [a] is a single-device param sharded onto 2 devices (a copy to the
   device tuple followed by a symbolic shrink wrapped in MULTI), [b] is
   replicated by a copy to the device tuple. The broadcast copy becomes
   per-device copies in an MSTACK, and the shard's shrink is moved before
   the MSTACK with the [_device_num] variable substituted per device.

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

(* shard(devices, axis=0): copy to the device tuple, shrink each device's
   slice with the symbolic [_device_num] offset, wrap in MULTI. *)
let shard_axis0 src ~rows ~cols =
  let ndev = List.length devices in
  let sz = rows / ndev in
  let copied = U.copy ~src ~device:(Multi devices) () in
  let dnum =
    U.variable ~name:"_device_num" ~min_val:0 ~max_val:(ndev - 1) ()
  in
  let off = U.alu_binary ~op:Ops.Mul ~lhs:dnum ~rhs:(Helpers.idx sz) in
  let sharded =
    U.shrink ~src:copied
      ~offset:(U.stack [ off; Helpers.idx 0 ])
      ~size:(U.stack [ Helpers.idx sz; Helpers.idx cols ])
  in
  U.multi ~src:sharded ~axis:0

let build () =
  let a = shard_axis0 (Helpers.mk_param ~idx:0 [ 16; 8 ]) ~rows:16 ~cols:8 in
  let b =
    U.copy ~src:(Helpers.mk_param ~idx:1 [ 16; 8 ]) ~device:(Multi devices) ()
  in
  Helpers.wrap_sink [ U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
