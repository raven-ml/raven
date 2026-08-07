(* Parity case: a contraction sum whose iteration space exceeds 2^62.

   Twelve independent contractions over a batch axis of 32 are summed into
   one 8x8 result, which fuses into a single kernel carrying twelve reduce
   axes on top of the output: 2^66 points. This is the shape a weight
   gradient takes when a recurrence is unrolled — one contraction per step,
   all accumulated — reduced to the smallest graph that reaches it.

   The host-threading heuristic sizes its thread count from the product of
   the full shape, so a 63-bit reconstruction of that product wraps and
   silently drops the thread split. This pins that the split survives an
   iteration space that does not fit in a machine word.

   CPU only — threading is a host-renderer feature — and CPU_COUNT is
   pinned so the chosen thread count does not follow the machine.

   Paired with main.py. Run `uv run main.py` to regenerate *.expected. *)

open Tolk_uop
module U = Uop

let batch = 32
let dim = 8
let terms = 12
let backends = List.filter (fun (name, _) -> name = "cpu") Helpers.all_backends

(* out[m; n] = Σ_k lhs[k; m] · rhs[k; n] *)
let matmul_tn ~m ~k ~n lhs rhs =
  let view src shape =
    U.broadcast_to
      ~src:(U.reshape ~src:(U.permute ~src ~order:[ 1; 0 ])
              ~shape:(Helpers.mk_shape shape))
      ~shape:(Helpers.mk_shape [ m; n; k ])
  in
  let ae = view lhs [ m; 1; k ] and be = view rhs [ 1; n; k ] in
  U.reduce_axis
    ~src:(U.alu_binary ~op:Ops.Mul ~lhs:ae ~rhs:be)
    ~op:Ops.Add ~axes:[ 2 ]

let build () =
  let b = batch and d = dim in
  let term t =
    matmul_tn ~m:d ~k:b ~n:d
      (Helpers.mk_param ~idx:(2 * t) [ b; d ])
      (Helpers.mk_param ~idx:((2 * t) + 1) [ b; d ])
  in
  let acc =
    List.fold_left
      (fun acc t -> U.alu_binary ~op:Ops.Add ~lhs:acc ~rhs:(term t))
      (term 0)
      (List.init (terms - 1) (fun i -> i + 1))
  in
  Helpers.wrap_sink [ acc ]

let () =
  Helpers.dump_tensor ~backends ~stages:[ Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
