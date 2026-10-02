(* A batch of predictions: each image in a panel captioned by its true and
   predicted class, the mistakes framed. *)

open Hugin_next

let classes = [| "ring"; "bar"; "dot"; "wave" |]

(* [picture k] is a 32 × 32 RGB image of class [k mod 4]. Its background drifts
   from blue towards red with [k], so each row of the grid has a tint of its
   own: an image is painted from its pixels, never through a colour scale. *)
let picture k =
  let c = k mod 4 and hue = float k /. 16. in
  Nx.init Nx.float32 [| 32; 32; 3 |] (fun i ->
      let y = float i.(0) -. 15.5 and x = float i.(1) -. 15.5 in
      let r = Float.sqrt ((x *. x) +. (y *. y)) in
      let ink =
        match c with
        | 0 -> Float.abs (r -. 10.) < 2.5
        | 1 -> Float.abs x < 3.
        | 2 -> r < 6.
        | _ -> Float.abs (y -. (6. *. Float.sin (x /. 4.))) < 2.
      in
      let base = [| 0.15 +. (0.5 *. hue); 0.2; 0.55 -. (0.3 *. hue) |] in
      if ink then 1. -. (0.2 *. base.(i.(2))) else base.(i.(2)))

let () =
  let n = 12 in
  let batch = Nx.stack (List.init n picture) in
  let truth = Nx.init Nx.int32 [| n |] (fun i -> Int32.of_int (i.(0) mod 4)) in
  let pred =
    Nx.init Nx.int32 [| n |] (fun i ->
        Int32.of_int
          (if i.(0) = 5 || i.(0) = 10 then (i.(0) + 1) mod 4 else i.(0) mod 4))
  in
  let name c = classes.(Int32.to_int c) in
  let captions =
    Array.map2
      (fun t p -> name t ^ " → " ^ name p)
      (Nx.to_array truth) (Nx.to_array pred)
  in
  layer
    [
      image ~fx:(dim ~scale:(Scale.band ~wrap:4 ()) ~labels:captions 0) batch;
      rect
        ~fx:(dim ~valid:(Nx.not_equal truth pred) 0)
        ~stroke:(const Color.red) ();
    ]
  |> save ~size:(Size.figure 420. 360.) "images.png"
