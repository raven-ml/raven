(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A mask of the photograph's light pixels, opened and closed by a disk,
   split into components, and its small components removed. *)

open Hugin

(* Footprints, openings and closings, from doc/morphology.md. *)
let disk r =
  let k = (2 * r) + 1 in
  Nx.init Nx.bool [| k; k |] (fun i ->
      let dy = i.(0) - r and dx = i.(1) - r in
      (dy * dy) + (dx * dx) <= r * r)

let opening fp x = Mynd.Filter.dilate fp (Mynd.Filter.erode fp x)
let closing fp x = Mynd.Filter.erode fp (Mynd.Filter.dilate fp x)

(* Areas and small objects, from doc/morphology.md: [areas c m] is each
   pixel's component area. *)
let areas c m =
  let s = Nx.shape m in
  let r = Array.length s in
  let lead = Array.sub s 0 (r - 3) in
  let flat x = Nx.reshape (Array.append lead [| s.(r - 3) * s.(r - 2) |]) x in
  let l = flat (Mynd.Mask.components c m) in
  let ones = Nx.ones_like l in
  let counts = Nx.scatter ~combine:Nx.Add ~axis:(-1) l ones (Nx.zeros_like l) in
  Nx.reshape s (Nx.take_along_axis ~axis:(-1) l counts)

let remove_small c n m =
  Nx.logical_and m (Nx.greater_s (areas c m) (Int64.of_int n))

(* [pieces m] is the number of 8-connected components of a mask: a
   component's label is the least row-major index of its pixels, so exactly
   one of its pixels has its own index as its label. *)
let pieces m =
  let h = Nx.dim (-3) m and w = Nx.dim (-2) m in
  let index = Nx.reshape [| h; w; 1 |] (Nx.arange Nx.int64 0 (h * w) 1) in
  Nx.item [] (Nx.count (Nx.equal (Mynd.Mask.components `Eight m) index))

let count m = Nx.item [] (Nx.count m)

let () =
  let img = Mynd.convert Nx.float32 (Nx_io.load_image "../chelsea.png") in
  let mask = Nx.greater_s (Mynd.Colour.luma img) 0.5 in
  let fp = disk 2 in
  let opened = opening fp mask in
  let closed = closing fp mask in
  Printf.printf "luma above 0.5: %Ld pixels in %Ld pieces\n" (count mask)
    (pieces mask);
  Printf.printf "opened by a disk of radius 2: %Ld pixels in %Ld pieces\n"
    (count opened) (pieces opened);
  Printf.printf "closed: %Ld pixels in %Ld pieces\n" (count closed)
    (pieces closed);

  (* Each pixel's area is the size of its piece. Removing the pieces of at
     most 100 pixels keeps the pixels of larger area. *)
  let cleaned = remove_small `Eight 100 mask in
  Printf.printf
    "without pieces of at most 100 pixels: %Ld pixels in %Ld pieces\n"
    (count cleaned) (pieces cleaned);

  (* The largest piece is the pixels of the largest area, since no other
     piece has as many, and its box is theirs. *)
  let a = areas `Eight mask in
  let largest = Nx.max a in
  let biggest = Nx.equal a largest in
  let b = Mynd.Box.of_mask Nx.float32 biggest in
  Printf.printf "largest piece: %Ld pixels, box [%s]\n" (Nx.item [] largest)
    (String.concat ", "
       (List.map (Printf.sprintf "%g") (Array.to_list (Nx.to_array b))));

  let panel x name = image (Nx.cast Nx.float32 x) |> title name in
  grid
    [
      [
        panel mask "luma above 0.5";
        panel opened "opened";
        panel cleaned "without small pieces";
      ];
    ]
  |> save ~size:(Size.figure 900. 240.) "morphology.png"
