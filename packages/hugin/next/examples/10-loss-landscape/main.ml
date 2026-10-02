(* A loss landscape: filled contours on a log scale, and the path an optimiser
   took across it. *)

open Hugin_next

let () =
  let alphas = Nx.linspace Nx.float64 (-2.) 2. 81 in
  let betas = Nx.linspace Nx.float64 (-1.5) 1.5 61 in
  let a = Nx.reshape [| 1; 81 |] alphas and b = Nx.reshape [| 61; 1 |] betas in
  (* Rosenbrock's valley, shifted to stay positive. *)
  let loss =
    Nx.add_s
      (Nx.add
         (Nx.square (Nx.sub_s a 1.))
         (Nx.mul_s (Nx.square (Nx.sub b (Nx.square a))) 10.))
      0.05
  in
  let steps = 40 in
  let path =
    Nx.init Nx.float64 [| steps; 2 |] (fun i ->
        let t = float i.(0) /. float (steps - 1) in
        if i.(1) = 0 then -1.6 +. (2.55 *. t)
        else 1.2 -. (2.1 *. t *. (1. -. t)) -. (0.25 *. t))
  in
  let px = num Nx.(slice [ A; I 0 ] path)
  and py = num Nx.(slice [ A; I 1 ] path) in
  layer
    [
      contour
        ~x:(num ~title:(Text.v "α") alphas)
        ~y:(num ~title:(Text.v "β") Nx.(slice [ A; N ] betas))
        ~fill:(num ~scale:(Scale.log ()) ~title:(Text.v "loss") loss)
        ();
      line ~x:px ~y:py ~stroke:(const Color.white) ();
      dot ~x:px ~y:py ~size:(const 9.) ~fill:(const Color.white) ();
    ]
  |> save "landscape.png"
