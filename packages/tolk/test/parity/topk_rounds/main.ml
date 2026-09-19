(* Parity case: top-2 of 4 router logits by selection rounds, then a softmax.

   The graph [Nx.top_k] emits for a small k, in reference ops. A round takes
   the maximum over the entries still free (NaNs aside), marks the entries
   equal to it, and [argmax] picks one of them; the pick leaves the free set
   by comparing positions. The two picks are concatenated, gather their
   logits, and a softmax over the k logits gives the routing weights. The
   reference's own [topk] is a full sort followed by a shrink; this case pins
   the rounds. A round costs four kernels: whether any entry is live, the
   maximum, and the two reduces of [argmax] (a maximum, then the last position
   holding it); the second round first writes its candidate mask out as int32.
   Then one kernel for the concatenation, one for the gather and three for the
   softmax. Fourteen kernels.

   Backends are limited to cpu and metal: kernel-name counters are shared
   across backends, so the reference must be generated with exactly the
   backends the OCaml side renders.

   Paired with main.py. Run `uv run main.py` to regenerate *.expected. *)

open Tolk_frontend
module D = Tolk_uop.Dtype

let backends =
  List.filter
    (fun (name, _) -> name = "cpu" || name = "metal")
    Helpers.all_backends

let ones shape = List.map (fun _ -> 1) shape

(* A scalar as the nx frontend hands it to a binary op: a constant broadcast to
   the shape of its peer. *)
let scalar value dtype shape =
  let const = Creation.full ~buffer:false ~dtype [] value in
  Movement.expand (Movement.reshape const (ones shape)) shape

let tokens = 3
let experts = 4
let k = 2

let build () =
  let param idx ?dtype shape = Tensor.of_uop (Helpers.mk_param ~idx ?dtype shape) in
  let shape = [ tokens; experts ] and row = [ tokens; 1 ] in
  let x = param 0 shape in
  let position =
    Movement.broadcast_to
      (Movement.reshape (param 1 ~dtype:D.int32 [ experts ]) [ 1; experts ])
      shape
  in
  let low = scalar (Sfloat Float.neg_infinity) D.float32 shape in
  let real =
    Elementwise.bitwise_xor (Elementwise.ne x x) (scalar (Sbool true) D.bool shape)
  in
  let pick free =
    let live = Elementwise.bitwise_and free real in
    let greatest =
      Movement.reshape
        (Reduce.max ~axis:[ 1 ] (Elementwise.where live x low))
        row
    in
    let best =
      Elementwise.bitwise_and live
        (Elementwise.eq x (Movement.broadcast_to greatest shape))
    in
    let some =
      Reduce.max ~axis:[ 1 ]
        (Elementwise.ne live (scalar (Sbool false) D.bool shape))
    in
    let chosen =
      Elementwise.where
        (Movement.broadcast_to (Movement.reshape some row) shape)
        best free
    in
    Dtype_ops.cast
      (Op.argmax ~axis:1 ~keepdim:true (Dtype_ops.cast chosen D.int32))
      D.int32
  in
  let rec rounds i free picks =
    if i = k then List.rev picks
    else
      let index = pick free in
      let taken = Elementwise.ne position (Movement.broadcast_to index shape) in
      rounds (i + 1) (Elementwise.bitwise_and free taken) (index :: picks)
  in
  let picks = rounds 0 (scalar (Sbool true) D.bool shape) [] in
  let indices = Op.cat ~dim:1 (List.hd picks) (List.tl picks) in
  let logits = Op.gather x ~dim:1 indices in
  let top =
    Movement.broadcast_to
      (Movement.reshape (Reduce.max ~axis:[ 1 ] logits) row)
      [ tokens; k ]
  in
  let e = Elementwise.exp (Elementwise.sub logits top) in
  let total =
    Movement.reshape (Reduce.sum ~axis:[ 1 ] ~dtype:D.float32 e) row
  in
  let weights = Elementwise.div e (Movement.broadcast_to total [ tokens; k ]) in
  Helpers.wrap_sink [ Tensor.uop indices; Tensor.uop weights ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
