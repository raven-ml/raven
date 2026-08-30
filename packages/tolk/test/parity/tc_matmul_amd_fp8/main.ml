(* Parity case: C = A @ B, fp8e4m3 inputs, float32 accumulate, M=N=16 K=128.

   Rendered for gfx950 (CDNA4) only: the heuristic optimizer engages the
   16x16x128 fp8 tensor core (scaled MFMA kernel via the CDNA string
   rewrite). *)

open Tolk_uop
module U = Uop

let backends = [ ("amd_gfx950", Tolk.Cstyle.amd Tolk.Gpu_target.CDNA4) ]

let build () =
  let m, n, k = (16, 16, 128) in
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
