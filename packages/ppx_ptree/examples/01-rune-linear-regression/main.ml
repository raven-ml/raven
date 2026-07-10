(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Fit [y = x @ w + b] with a derived parameter tree. The generated [map],
   [map2], and [iter] functions make [Params] directly usable by Rune. *)

module Params = struct
  type t = { w : Nx.float32_t; b : Nx.float32_t } [@@deriving ptree]
end

let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  let expected_w = Nx.create Nx.float32 [| 3; 1 |] [| 2.0; -1.0; 0.5 |] in
  let expected_b = 0.3 in
  let x = Nx.randn Nx.float32 [| 64; 3 |] in
  let noise = Nx.mul_s (Nx.randn Nx.float32 [| 64; 1 |]) 0.01 in
  let y = Nx.add_s (Nx.add (Nx.matmul x expected_w) noise) expected_b in

  let loss (params : Params.t) =
    let prediction = Nx.add (Nx.matmul x params.w) params.b in
    Nx.mean (Nx.square (Nx.sub prediction y))
  in

  (* [Params] is both the input and output tree. The first call traces and
     compiles the gradient and update; later calls replay the compiled step. *)
  let learning_rate = 0.1 in
  let step =
    Rune.jit2
      (module Params)
      (module Params)
      (fun params ->
        let gradients = Rune.grad (module Params) loss params in
        Params.
          {
            w = Nx.sub params.w (Nx.mul_s gradients.w learning_rate);
            b = Nx.sub params.b (Nx.mul_s gradients.b learning_rate);
          })
  in

  let params =
    ref
      Params.
        { w = Nx.zeros Nx.float32 [| 3; 1 |]; b = Nx.zeros Nx.float32 [| 1 |] }
  in
  for iteration = 1 to 200 do
    params := step !params;
    if iteration mod 40 = 0 then
      Printf.printf "step %3d  loss %.6f\n%!" iteration
        (Nx.item [] (loss !params))
  done;

  Printf.printf "\nw (expected about [2.0; -1.0; 0.5]):\n  %s\n"
    (Nx.data_to_string !params.w);
  Printf.printf "b (expected about 0.3):\n  %s\n" (Nx.data_to_string !params.b)
