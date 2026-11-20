(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* One transform moves an image, its labels and their boxes together: the
   photograph turned by 30° about its centre, with labels drawn by hand on
   the cat's eyes and nose. *)

open Hugin

let names = [| "left eye"; "right eye"; "nose" |]

(* The labels: [k + 1] inside the ellipse [ellipses.(k)], of centre
   [(cx, cy)] and radii [(rx, ry)] in pixels, [0] elsewhere. *)
let ellipses =
  [|
    ((171., 115.), (36., 31.));
    ((316., 137.), (29., 26.));
    ((262., 244.), (30., 15.));
  |]

let labels h w =
  Nx.init Nx.int32 [| h; w; 1 |] (fun i ->
      let x = Float.of_int i.(1) +. 0.5 and y = Float.of_int i.(0) +. 0.5 in
      let inside ((cx, cy), (rx, ry)) =
        let u = (x -. cx) /. rx and v = (y -. cy) /. ry in
        (u *. u) +. (v *. v) < 1.
      in
      let rec find k =
        if k = Array.length ellipses then 0l
        else if inside ellipses.(k) then Int32.of_int (k + 1)
        else find (k + 1)
      in
      find 0)

(* [about c m] is the transform [m] about the position [c]: move [c] to the
   origin, apply [m], move it back. [Nx.matmul a b] applies [b] first. *)
let about c m =
  let open Mynd.Transform in
  Nx.matmul (translation c) (Nx.matmul m (translation (Nx.neg c)))

(* [corners b] is the four corners of the box [b], [[4; 2]]. *)
let corners b =
  let v i = Nx.item [ i ] b in
  Nx.create Nx.float32 [| 4; 2 |] [| v 0; v 1; v 2; v 1; v 0; v 3; v 2; v 3 |]

(* [hull p] is the least box holding the positions [p], [[n; 2]]. *)
let hull p =
  Nx.concatenate ~axis:0 [ Nx.min ~axes:[ 0 ] p; Nx.max ~axes:[ 0 ] p ]

let shape x =
  let dims = Array.to_list (Array.map string_of_int (Nx.shape x)) in
  "[" ^ String.concat "; " dims ^ "]"

let floats fmt x =
  String.concat " "
    (List.map (Printf.sprintf fmt) (Array.to_list (Nx.to_array x)))

let count m = Int64.to_int (Nx.item [] (Nx.count m))

let () =
  let img = Mynd.convert Nx.float32 (Nx_io.load_image "../chelsea.png") in
  let h = Nx.dim 0 img and w = Nx.dim 1 img in
  let l = labels h w in

  (* [m] maps the photograph's positions to the turned image's. A warp reads,
     at each of its own centres, the input at the position that maps there:
     the centres mapped by [m]'s inverse. *)
  let c =
    Nx.create Nx.float32 [| 2 |]
      [| Float.of_int w /. 2.; Float.of_int h /. 2. |]
  in
  let m =
    about c (Mynd.Transform.rotation (Nx.scalar Nx.float32 (Float.pi /. 6.)))
  in
  let p =
    Mynd.Transform.apply (Nx.inv m) (Mynd.Sample.centres Nx.float32 (h, w))
  in
  let turned = Mynd.Sample.at ~edge:`Zero (`Cubic (-0.5)) img p in

  (* Labels are read at the nearest pixel, so every value is a label. *)
  let turned_l = Mynd.Sample.nearest ~edge:`Zero l p in
  Printf.printf "turned image %s, labels %s\n" (shape turned) (shape turned_l);

  (* A box under the transform: its corners mapped by [m], and their hull. A
     pixel of the turned labels holds a label where its centre maps into one
     of that label's pixels, which lie in the box, so its centre lies in the
     hull. Its cell reaches half a pixel past its centre, so the box of the
     turned label lies within the hull grown by half a pixel. *)
  let boxes = ref [] in
  Array.iteri
    (fun k name ->
      let mask x = Nx.equal_s x (Int32.of_int (k + 1)) in
      let before = Mynd.Box.of_mask Nx.float32 (mask l) in
      let corners' = Mynd.Transform.apply m (corners before) in
      let hl = hull corners' in
      let after = Mynd.Box.of_mask Nx.float32 (mask turned_l) in
      let v x i = Nx.item [ i ] x in
      let inside =
        v after 0 >= v hl 0 -. 0.5
        && v after 1 >= v hl 1 -. 0.5
        && v after 2 <= v hl 2 +. 0.5
        && v after 3 <= v hl 3 +. 0.5
      in
      Printf.printf "%s: %d pixels, %d after\n" name
        (count (mask l))
        (count (mask turned_l));
      Printf.printf "  box %s, its corners' hull after %s\n"
        (floats "%g" before) (floats "%.1f" hl);
      Printf.printf "  box after %s, within the hull grown by 1/2: %b\n"
        (floats "%g" after) inside;
      boxes := (after, hl) :: !boxes)
    names;

  let rects fmt bs =
    let b = Nx.stack (List.map fmt bs) in
    let side i = num (Nx.slice [ Nx.A; Nx.I i ] b) in
    (side 0, side 1, side 2, side 3)
  in
  let x, y, x2, y2 = rects fst !boxes in
  let hx, hy, hx2, hy2 = rects snd !boxes in
  let overlay =
    Nx.concatenate ~axis:(-1)
      [
        Nx.full Nx.float32 [| h; w; 3 |] 1.;
        Nx.mul_s (Nx.cast Nx.float32 (Nx.greater_s turned_l 0l)) 0.4;
      ]
  in
  layer
    [
      image turned;
      image overlay;
      rect ~x:hx ~y:hy ~x2:hx2 ~y2:hy2 ~stroke:(const (Color.v 1. 0.85 0.)) ();
      rect ~x ~y ~x2 ~y2 ~stroke:(const Color.white) ();
    ]
  |> save ~size:(Size.figure 451. 300.) "warps.png"
