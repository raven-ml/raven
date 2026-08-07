(* Parity case: two reduce accumulators in one kernel.

   [out = Aᵀ@B + C@D] over 16x16. The two contractions fuse into one kernel,
   but only the first has its contraction axis mapped onto the local index,
   so it is staged through shared memory and group-reduced while the second
   stays an ordinary loop. Each reduce therefore needs its own accumulator
   register; giving them the same one makes the closing add read a single
   accumulator twice and doubles the result.

   CPU is excluded: without local dimensions there is no group reduce, so
   the kernel has one accumulator and the case proves nothing there.

   Paired with main.py. Run `uv run main.py` to regenerate *.expected. *)

open Tolk_uop
module U = Uop

let n = 16
let reshape src shape = U.reshape ~src ~shape:(Helpers.mk_shape shape)
let transpose src = U.permute ~src ~order:[ 1; 0 ]

let contract ae be =
  let expand src = U.broadcast_to ~src ~shape:(Helpers.mk_shape [ n; n; n ]) in
  U.reduce_axis
    ~src:(U.alu_binary ~op:Ops.Mul ~lhs:(expand ae) ~rhs:(expand be))
    ~op:Ops.Add ~axes:[ 2 ]

(* out[m; n] = Σ_k lhs[m; k] · rhs[k; n] *)
let matmul_nn lhs rhs =
  contract (reshape lhs [ n; 1; n ]) (reshape (transpose rhs) [ 1; n; n ])

(* out[m; n] = Σ_k lhs[k; m] · rhs[k; n] *)
let matmul_tn lhs rhs =
  contract
    (reshape (transpose lhs) [ n; 1; n ])
    (reshape (transpose rhs) [ 1; n; n ])

let build () =
  let p i = Helpers.mk_param ~idx:i [ n; n ] in
  Helpers.wrap_sink
    [
      U.alu_binary ~op:Ops.Add
        ~lhs:(matmul_tn (p 0) (p 1))
        ~rhs:(matmul_nn (p 2) (p 3));
    ]

let () =
  Helpers.dump_tensor ~backends:Helpers.gpu_backends ~stages:[ Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
