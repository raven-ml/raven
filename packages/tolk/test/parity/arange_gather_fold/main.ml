(* Parity case: embedding-style gather of an arange folds at schedule time.

   The gpt2 positional-embedding shape: [weight[arange(32)[:, :13]]] written
   as the one-hot [eq] + [where] + [sum] gather. The arange must never
   materialize as an int buffer — the whole gather folds into a single
   kernel that reads the weight rows directly.

   Backends are limited to cpu and cuda: kernel-name counters are shared
   across backends, so the reference must be generated with exactly the
   backends the OCaml side renders.

   Paired with main.py. Run `uv run main.py` to regenerate *.expected. *)

open Tolk_frontend
module D = Tolk_uop.Dtype

let backends =
  List.filter
    (fun (name, _) -> name = "cpu" || name = "cuda")
    Helpers.all_backends

let build () =
  let w = Tensor.of_uop (Helpers.mk_param ~idx:0 [ 32; 4 ]) in
  let allpos = Movement.reshape (Op.arange 32) [ 1; -1 ] in
  let pos = Movement.shrink allpos [ (0, 1); (0, 13) ] in
  let arange = Op.arange 32 in
  let one_hot =
    Movement.unsqueeze (Elementwise.eq arange (Movement.unsqueeze pos (-1))) (-1)
  in
  (* The +1 keeps the store from being a raw vector-load passthrough,
     which renders differently and is unrelated to the gather fold. *)
  let out =
    Elementwise.add
      (Reduce.sum ~axis:[ -2 ] ~dtype:D.Val.float32
         (Elementwise.where one_hot w (Tensor.i 0)))
      (Tensor.f 1.0)
  in
  Helpers.wrap_sink [ Tensor.uop out ]

let () =
  Helpers.dump_tensor ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (build ())
