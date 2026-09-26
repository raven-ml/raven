open Tolk_uop
module U = Uop

let backends = List.filter (fun (name, _) -> name = "cpu" || name = "cuda") Helpers.all_backends

let build () =
  let a = Helpers.mk_param ~idx:0 ~dtype:Dtype.int32 [2; 4] in
  let wide = U.alu_binary ~op:Ops.Mul ~lhs:(U.cast ~src:a ~dtype:Dtype.weakint)
    ~rhs:(U.const (Const.int Dtype.weakint 2147483648)) in
  let reshaped = U.reshape ~src:wide ~shape:(U.stack [U.const_int 4; U.const_int 2]) in
  let transposed = U.permute ~src:reshaped ~order:[1; 0] in
  Helpers.wrap_sink [ U.cast ~src:transposed ~dtype:Dtype.int64 ]

let () =
  Helpers.dump_tensor ~backends ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
