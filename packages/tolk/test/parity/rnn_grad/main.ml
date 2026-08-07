(* Parity case: the reverse pass of an unrolled recurrence.

   The forward chain is the one in [rnn_unroll] — two steps of
   [h <- x@W + h@U] over 4x4 matrices under a squared-magnitude loss — and
   this case adds the gradient sweep back through it, which is the graph a
   training step compiles. It exercises the three matmul orientations
   together ([a@b] forward, [a@b'] for the gradient flowing back through a
   weight, [a'@b] for a weight's own gradient), keeps every forward hidden
   state live across both sweeps, and accumulates each weight gradient as a
   sum of one contraction per step.

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

let contract ~m ~n ~k ae be =
  let expand src = U.broadcast_to ~src ~shape:(Helpers.mk_shape [ m; n; k ]) in
  U.reduce_axis ~src:(mul (expand ae) (expand be)) ~op:Ops.Add ~axes:[ 2 ]

(* out[m; n] = Σ_k lhs[m; k] · rhs[k; n] *)
let matmul_nn ~m ~k ~n lhs rhs =
  contract ~m ~n ~k (reshape lhs [ m; 1; k ]) (reshape (transpose rhs) [ 1; n; k ])

(* out[m; n] = Σ_k lhs[m; k] · rhs[n; k] *)
let matmul_nt ~m ~k ~n lhs rhs =
  contract ~m ~n ~k (reshape lhs [ m; 1; k ]) (reshape rhs [ 1; n; k ])

(* out[m; n] = Σ_k lhs[k; m] · rhs[k; n] *)
let matmul_tn ~m ~k ~n lhs rhs =
  contract ~m ~n ~k
    (reshape (transpose lhs) [ m; 1; k ])
    (reshape (transpose rhs) [ 1; n; k ])

let build () =
  let b = batch and d = dim in
  let w_in = Helpers.mk_param ~idx:0 [ d; d ] in
  let w_rec = Helpers.mk_param ~idx:1 [ d; d ] in
  let h0 = Helpers.mk_param ~idx:2 [ b; d ] in
  let xs = List.init horizon (fun t -> Helpers.mk_param ~idx:(3 + t) [ b; d ]) in
  let h = Array.make (horizon + 1) h0 in
  List.iteri
    (fun t x ->
      h.(t + 1) <-
        add (matmul_nn ~m:b ~k:d ~n:d x w_in) (matmul_nn ~m:b ~k:d ~n:d h.(t) w_rec))
    xs;
  let two =
    U.expand
      ~src:(U.const (Const.float Dtype.float32 2.0))
      ~dims:(Helpers.mk_shape [ b; d ])
  in
  let g = Array.make (horizon + 1) h0 in
  g.(horizon) <- mul two h.(horizon);
  for t = horizon - 1 downto 0 do
    let carried = matmul_nt ~m:b ~k:d ~n:d g.(t + 1) w_rec in
    g.(t) <- (if t = 0 then carried else add (mul two h.(t)) carried)
  done;
  let sum_steps operands =
    List.mapi (fun t o -> matmul_tn ~m:d ~k:b ~n:d o g.(t + 1)) operands
    |> function
    | [] -> invalid_arg "horizon must be positive"
    | first :: rest -> List.fold_left add first rest
  in
  Helpers.wrap_sink
    [ sum_steps xs; sum_steps (List.init horizon (fun t -> h.(t))); g.(0) ]

let () =
  Helpers.dump_tensor ~stages:[ Helpers.Stage7 ] ~out_dir:Sys.argv.(1) (build ())
