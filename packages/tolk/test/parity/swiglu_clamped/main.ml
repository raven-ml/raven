(* Parity case: clamped SwiGLU over interleaved gate and linear features.

   The expert activation: features [...; 2 * half] alternate gate and
   linear, read as pairs [-1; half; 2] and split by a shrink of the last
   axis. The gate is clamped above, the linear feature on both sides, and the
   result is [gate * sigmoid (1.702 * gate) * (linear + 1)], the sigmoid
   written as [1 / (1 + 2 ** (x * -1 / ln 2))]. One kernel reading the
   interleaved buffer at strides of two.

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

let tokens = 2
let k = 2
let half = 8
let limit = 7.0

let build () =
  let out = [ tokens; k; 1; half ] in
  let f v = scalar (Sfloat v) D.float32 out in
  let h = Tensor.of_uop (Helpers.mk_param ~idx:0 [ tokens; k; 1; 2 * half ]) in
  let pairs = Movement.reshape h [ tokens * k; half; 2 ] in
  let feature i =
    let column =
      Movement.shrink pairs [ (0, tokens * k); (0, half); (i, i + 1) ]
    in
    Movement.reshape (Movement.reshape column [ tokens * k; half ]) out
  in
  let gate = Elementwise.minimum (feature 0) (f limit) in
  let linear =
    Elementwise.minimum (Elementwise.maximum (feature 1) (f (-.limit))) (f limit)
  in
  let x = Elementwise.mul gate (f 1.702) in
  let sigmoid =
    Elementwise.reciprocal
      (Elementwise.add
         (Elementwise.exp2 (Elementwise.mul x (f (-1.0 /. Float.log 2.0))))
         (f 1.0))
  in
  Helpers.wrap_sink
    [
      Tensor.uop
        (Elementwise.mul
           (Elementwise.mul gate sigmoid)
           (Elementwise.add linear (f 1.0)));
    ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
