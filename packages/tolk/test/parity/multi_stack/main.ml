open Tolk_uop
module U = Uop

let backends = List.filter (fun (name, _) -> name = "cpu" || name = "cuda") Helpers.all_backends

let build () =
  let a = Helpers.mk_param ~idx:0 ~device:"CPU:0" [2; 4] in
  let b = Helpers.mk_param ~idx:1 ~device:"CPU:1" [2; 4] in
  let shards = U.unshard ~src:(U.mstack [a; b]) ~axes:[0] () in
  Helpers.wrap_sink [ U.alu_binary ~op:Ops.Add ~lhs:shards
    ~rhs:(U.const (Const.float Dtype.float32 1.0)) ]

let () =
  Helpers.dump_tensor ~backends ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
