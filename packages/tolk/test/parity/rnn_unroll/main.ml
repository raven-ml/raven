(* Parity case: an unrolled recurrence, forward only.

   Two steps of [h <- x@W + h@U] over 4x4 matrices, each step contributing a
   squared-magnitude scalar loss to a running sum. Each matmul is a
   broadcast-multiply contracted over the trailing axis; every hidden state
   feeds both the next step and its own loss term, so the chain fuses into a
   small number of kernels whose split is decided by rangeify.

   Paired with main.py. Run `uv run main.py` to regenerate *.expected. *)

open Tolk_uop
module U = Uop

let batch = 4
let dim = 4
let horizon = 2
let add a b = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b
let mul a b = U.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:b
let reshape src shape = U.reshape ~src ~shape:(Helpers.mk_shape shape)
let transpose src = U.permute ~src ~order:[ 1; 0 ]

(* out[m; n] = Σ_k lhs[m; k] · rhs[k; n] *)
let matmul ~m ~k ~n lhs rhs =
  let ae =
    U.broadcast_to ~src:(reshape lhs [ m; 1; k ]) ~shape:(Helpers.mk_shape [ m; n; k ])
  in
  let be =
    U.broadcast_to
      ~src:(reshape (transpose rhs) [ 1; n; k ])
      ~shape:(Helpers.mk_shape [ m; n; k ])
  in
  U.reduce_axis ~src:(mul ae be) ~op:Ops.Add ~axes:[ 2 ]

let build () =
  let b = batch and d = dim in
  let w_in = Helpers.mk_param ~idx:0 [ d; d ] in
  let w_rec = Helpers.mk_param ~idx:1 [ d; d ] in
  let h0 = Helpers.mk_param ~idx:2 [ b; d ] in
  let rec loop t h acc =
    if t = horizon then acc
    else
      let x = Helpers.mk_param ~idx:(3 + t) [ b; d ] in
      let h = add (matmul ~m:b ~k:d ~n:d x w_in) (matmul ~m:b ~k:d ~n:d h w_rec) in
      let loss = U.reduce_axis ~src:(mul h h) ~op:Ops.Add ~axes:[ 0; 1 ] in
      loop (t + 1) h (match acc with None -> Some loss | Some a -> Some (add a loss))
  in
  match loop 0 h0 None with
  | Some acc -> Helpers.wrap_sink [ acc ]
  | None -> invalid_arg "horizon must be positive"

let () =
  Helpers.dump_tensor ~stages:[ Helpers.Stage7 ] ~out_dir:Sys.argv.(1) (build ())
