(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Fit [y = x @ w + b] with a derived structure. The weights are the positions
   of [Params.t]'s payload ['a]; it holds no fixed array, so its set is
   anonymous. The deriver writes [Params.walk], and [Nx.Ptree.instantiate]
   describes the record at arrays. *)

module Params = struct
  type ('a, _) t = { w : 'a; b : 'a } [@@deriving ptree]
end

let params = Nx.Ptree.instantiate (module Params)

let () =
  let expected_w = Nx.create Nx.float32 [| 3; 1 |] [| 2.0; -1.0; 0.5 |] in
  let expected_b = 0.3 in
  let x = Nx.Rng.normal ~key:(Nx.Rng.key 0) Nx.float32 [| 64; 3 |] in
  let noise =
    Nx.mul_s (Nx.Rng.normal ~key:(Nx.Rng.key 1) Nx.float32 [| 64; 1 |]) 0.01
  in
  let y = Nx.add_s (Nx.add (Nx.matmul x expected_w) noise) expected_b in

  let loss (p : (_, _) Params.t) =
    let prediction = Nx.add (Nx.matmul x p.w) p.b in
    Nx.mean (Nx.square (Nx.sub prediction y))
  in

  (* The first call traces and compiles the gradient and the update; later calls
     replay the compiled program. *)
  let learning_rate = 0.1 in
  let step =
    Rune.jit
      Nx.Ptree.(params @-> returns params)
      (fun p ->
        let g = Rune.grad params loss p in
        Params.
          {
            w = Nx.sub p.w (Nx.mul_s g.w learning_rate);
            b = Nx.sub p.b (Nx.mul_s g.b learning_rate);
          })
  in

  let p =
    ref
      Params.
        { w = Nx.zeros Nx.float32 [| 3; 1 |]; b = Nx.zeros Nx.float32 [| 1 |] }
  in
  for iteration = 1 to 200 do
    (* Donating the parameters lets the step write the new ones into their
       memory; the old [!p] is dead after the call. *)
    p := step (Nx.Ptree.donate params !p);
    if iteration mod 40 = 0 then
      Printf.printf "step %3d  loss %.6f\n%!" iteration (Nx.item [] (loss !p))
  done;

  Printf.printf "\nw (expected about [2.0; -1.0; 0.5]):\n  %s\n"
    (Nx.to_string !p.w);
  Printf.printf "b (expected about 0.3):\n  %s\n" (Nx.to_string !p.b)
