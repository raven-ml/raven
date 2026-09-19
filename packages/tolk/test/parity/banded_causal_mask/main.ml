(* Parity case: sliding-window causal mask built from positions.

   Each query carries its position p, and a key at column j is seen when
   [j <= p] and [j > p - window]: a band of [window] keys ending at the query.
   The mask selects scores against -inf. The columns are a buffer, as a
   traced function captures its [arange]. One kernel: both comparisons fold
   into the select and no mask is written out.

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

let batch = 1
let queries = 3
let keys = 8
let window = 4

let build () =
  let param idx ?dtype shape = Tensor.of_uop (Helpers.mk_param ~idx ?dtype shape) in
  let shape = [ batch; queries; keys ] in
  let pos = param 0 ~dtype:D.int32 [ batch; queries ] in
  let column = param 1 ~dtype:D.int32 [ keys ] in
  let scores = param 2 shape in
  let key =
    Movement.broadcast_to (Movement.reshape column [ 1; 1; keys ]) shape
  in
  let query =
    Movement.broadcast_to (Movement.reshape pos [ batch; queries; 1 ]) shape
  in
  let causal = Elementwise.le key query in
  let recent =
    Elementwise.lt
      (Elementwise.sub query (scalar (Sint window) D.int32 shape))
      key
  in
  Helpers.wrap_sink
    [
      Tensor.uop
        (Elementwise.where
           (Elementwise.bitwise_and causal recent)
           scores
           (scalar (Sfloat Float.neg_infinity) D.float32 shape));
    ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
