(* Parity case: masked softmax whose normaliser includes a per-head sink.

   Scores [heads; queries; keys] at float32, masked to -inf, and one sink
   logit per head: a key of value zero that has no column. The shift is the
   maximum over the row and the sink, and the sink's exponential joins the
   sum. A row with every key masked has a finite shift, the sink, so its
   weights are 0 / 1 and not 0 / 0. Three kernels: shift, total, quotient,
   each applying the mask itself; the sink's term fuses into the total.

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

let heads = 2
let queries = 3
let keys = 4

let build () =
  let param idx ?dtype shape = Tensor.of_uop (Helpers.mk_param ~idx ?dtype shape) in
  let shape = [ heads; queries; keys ] and row = [ heads; queries; 1 ] in
  let scores = param 0 shape in
  let sink = Movement.reshape (param 1 [ heads; 1 ]) [ heads; 1; 1 ] in
  let mask = param 2 ~dtype:D.bool [ queries; keys ] in
  let scores =
    Elementwise.where
      (Movement.broadcast_to mask shape)
      scores
      (scalar (Sfloat Float.neg_infinity) D.float32 shape)
  in
  let sink = Movement.broadcast_to sink row in
  let top =
    Elementwise.maximum
      (Movement.reshape (Reduce.max ~axis:[ 2 ] scores) row)
      sink
  in
  let e =
    Elementwise.exp (Elementwise.sub scores (Movement.broadcast_to top shape))
  in
  let total =
    Movement.reshape (Reduce.sum ~axis:[ 2 ] ~dtype:D.float32 e) row
  in
  let total = Elementwise.add total (Elementwise.exp (Elementwise.sub sink top)) in
  Helpers.wrap_sink
    [ Tensor.uop (Elementwise.div e (Movement.broadcast_to total shape)) ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
