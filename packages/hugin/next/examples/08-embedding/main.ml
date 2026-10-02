(* An embedding: a few thousand points coloured by digit, and a hundred thousand
   drawn as one image of their dots. *)

open Hugin_next

let digits = Array.init 10 string_of_int

let embedding n =
  let keys = Nx.Rng.split ~n:2 (Nx.Rng.key 8) in
  let labels = Nx.Rng.randint keys.(0) ~high:10 [| n |] in
  let angle = Nx.mul_s (Nx.cast Nx.float32 labels) (2. *. Float.pi /. 10.) in
  let centre =
    Nx.stack ~axis:1 [ Nx.mul_s (Nx.cos angle) 6.; Nx.mul_s (Nx.sin angle) 6. ]
  in
  let e = Nx.add centre (Nx.Rng.normal keys.(1) Nx.float32 [| n; 2 |]) in
  (e, labels)

let figure n =
  let e, labels = embedding n in
  dot
    ~x:(num Nx.(slice [ A; I 0 ] e))
    ~y:(num Nx.(slice [ A; I 1 ] e))
    ~fill:(cat ~title:(Text.v "digit") ~labels:digits labels)
    ~opacity:(const 0.3) ()

let () =
  grid
    [
      [
        figure 3_000 |> title (Text.v "3,000 points");
        figure 100_000 |> title (Text.v "100,000 points");
      ];
    ]
  |> save ~size:(Size.figure 640. 300.) "embedding.png"
