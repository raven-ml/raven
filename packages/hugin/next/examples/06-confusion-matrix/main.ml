(* A confusion matrix: cells coloured by recall, each with its count in the
   colour that reads on it. *)

open Hugin_next

let classes = [| "plane"; "car"; "bird"; "cat"; "deer"; "dog" |]

let counts =
  let n = Array.length classes in
  Nx.init Nx.int32 [| n; n |] (fun i ->
      let t = i.(0) and p = i.(1) in
      if t = p then Int32.of_int (80 + (7 * t mod 15))
      else
        Int32.of_int
          ((t * 7) + (p * 3 mod 11 mod if abs (t - p) = 1 then 13 else 4)))

let () =
  let m = counts in
  let f = Nx.cast Nx.float64 m in
  let recall =
    num ~title:(Text.v "recall") Nx.(div f (sum ~axes:[ 1 ] ~keepdims:true f))
  in
  let predicted = dim ~title:(Text.v "predicted") ~labels:classes 1
  and truth = dim ~title:(Text.v "true") ~labels:classes 0 in
  layer
    [
      rect ~x:predicted ~y:truth ~fill:recall ();
      text ~x:predicted ~y:truth ~text:(num m)
        ~fill:(map_range Color.contrast recall)
        ();
    ]
  |> coord (Coord.cartesian ~aspect:1. ())
  |> save "confusion.png"
