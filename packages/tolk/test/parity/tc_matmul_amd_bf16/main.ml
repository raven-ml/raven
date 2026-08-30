(* Parity case: C = A @ B, bfloat16 inputs, float32 accumulate, M=N=K=16.

   Rendered for gfx1100 (RDNA3) and gfx942 (CDNA3): the heuristic optimizer
   engages the 16x16x16 bfloat16 tensor core — a WMMA kernel on RDNA3, an
   MFMA kernel (with the CDNA string rewrite) on CDNA3. *)

open Tolk_uop
module U = Uop

let backends =
  [
    ("amd_gfx1100", Tolk.Cstyle.amd Tolk.Gpu_target.RDNA3);
    ("amd_gfx942", Tolk.Cstyle.amd Tolk.Gpu_target.CDNA3);
  ]

let build () =
  let m, n, k = (16, 16, 16) in
  let a = Helpers.mk_param ~idx:0 ~dtype:Dtype.bfloat16 [ m; k ] in
  let b = Helpers.mk_param ~idx:1 ~dtype:Dtype.bfloat16 [ k; n ] in
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
