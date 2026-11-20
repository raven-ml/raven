(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sampling and transforms have derivatives in their positions and
   parameters. Here the angle that turns the grey photograph onto a turned
   copy is found by gradient descent on the difference of the two, through
   sampling. *)

let () =
  let img = Mynd.convert Nx.float32 (Nx_io.load_image "../chelsea.png") in
  let h = 75 and w = 113 in
  let grey =
    Mynd.Sample.resize ~edge:`Replicate `Triangle (h, w) (Mynd.Colour.luma img)
  in

  (* [warp a x] reads [x] at its pixels' centres turned by [a] radians about
     the image's centre. *)
  let c = Mynd.Sample.centres Nx.float32 (h, w) in
  let centre =
    Nx.create Nx.float32 [| 2 |]
      [| Float.of_int w /. 2.; Float.of_int h /. 2. |]
  in
  let about a =
    let open Mynd.Transform in
    Nx.matmul (translation centre)
      (Nx.matmul (rotation a) (translation (Nx.neg centre)))
  in
  let warp a x =
    Mynd.Sample.at ~edge:`Replicate (`Cubic (-0.5)) x
      (Mynd.Transform.apply (about a) c)
  in
  let target = warp (Nx.scalar Nx.float32 0.2) grey in

  (* The loss is the mean squared difference over the middle of the two
     images, which every angle the descent visits reads from inside the
     photograph. *)
  let inner = Nx.slice [ Nx.R (15, 60); Nx.R (20, 93) ] in
  let loss a =
    Nx.mean (Nx.square (Nx.sub (inner (warp a grey)) (inner target)))
  in
  let a = ref (Nx.scalar Nx.float32 0.) in
  for step = 1 to 30 do
    let l, g = Rune.value_and_grad' loss !a in
    if step = 1 || step mod 10 = 0 then
      Printf.printf "step %2d: loss %.4f, angle %.4f\n" step (Nx.item [] l)
        (Nx.item [] !a);
    a := Nx.sub !a (Nx.mul_s g 0.3)
  done;
  Printf.printf "angle found: %.4f radians\n" (Nx.item [] !a)
