(* Parity case: C = A @ B, fp8e4m3 inputs, float32 accumulate, M=16 N=8 K=32.

   Rendered for both SM80 and SM90: the heuristic optimizer engages the
   8x16x32 fp8 tensor core on SM90 (mma.sync kernel) while SM80, which has no
   fp8 tensor core, renders a plain reduce loop. *)

open Tolk_uop
module U = Uop

let backends =
  [
    ("cuda_sm80", Tolk.Cstyle.cuda Tolk.Gpu_target.SM80);
    ("cuda_sm90", Tolk.Cstyle.cuda Tolk.Gpu_target.SM90);
  ]

let build () =
  let m, n, k = (16, 8, 32) in
  let a = Helpers.mk_param ~idx:0 ~dtype:Dtype.fp8e4m3 [ m; k ] in
  let b = Helpers.mk_param ~idx:1 ~dtype:Dtype.fp8e4m3 [ k; n ] in
  (* dot: a.reshape(M,1,K) * b.permute(1,0).reshape(1,N,K), summed over K. *)
  let ar = U.reshape ~src:a ~shape:(Helpers.mk_shape [ m; 1; k ]) in
  let ae = U.broadcast_to ~src:ar ~shape:(Helpers.mk_shape [ m; n; k ]) in
  let bt = U.permute ~src:b ~order:[ 1; 0 ] in
  let br = U.reshape ~src:bt ~shape:(Helpers.mk_shape [ 1; n; k ]) in
  let be = U.broadcast_to ~src:br ~shape:(Helpers.mk_shape [ m; n; k ]) in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:ae ~rhs:be in
  let mulf = U.cast ~src:mul ~dtype:Dtype.float32 in
  let red = U.reduce_axis ~src:mulf ~op:Ops.Add ~axes:[ 2 ] in
  Helpers.wrap_sink [ red ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
