(* Attention maps: one panel per layer and head, sharing one colour bar. *)

open Hugin

let tokens = [| "the"; "cat"; "sat"; "on"; "the"; "mat" |]

let () =
  let t = Array.length tokens in
  let logits = Nx.Rng.normal (Nx.Rng.key 7) Nx.float32 [| 3; 4; t; t |] in
  let causal = Nx.init Nx.bool [| t; t |] (fun i -> i.(1) <= i.(0)) in
  let masked =
    Nx.where
      (Nx.broadcast_to [| 3; 4; t; t |] causal)
      (Nx.mul_s logits 2.)
      (Nx.full Nx.float32 [| 3; 4; t; t |] Float.neg_infinity)
  in
  let a = Nx.softmax ~axes:[ 3 ] masked in
  rect ~fy:(dim ~title:"layer" 0) ~fx:(dim ~title:"head" 1)
    ~y:(dim ~labels:tokens 2) ~x:(dim ~labels:tokens 3)
    ~fill:(num ~title:"attention" a) ()
  |> coord (Coord.cartesian ~aspect:1. ())
  |> save ~size:(Size.figure 520. 380.) "attention.png"
