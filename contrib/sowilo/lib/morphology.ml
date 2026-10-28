(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type kernel_shape = Rect | Cross | Ellipse

let structuring_element shape (kh, kw) =
  if kh <= 0 || kw <= 0 || kh mod 2 = 0 || kw mod 2 = 0 then
    invalid_arg "structuring_element: dimensions must be positive odd integers";
  match shape with
  | Rect -> Nx.ones Nx.uint8 [| kh; kw |]
  | Cross ->
      let center_h = kh / 2 and center_w = kw / 2 in
      let h_line = Nx.ones Nx.uint8 [| 1; kw |] in
      let h_padded =
        Nx.pad [| (center_h, kh - center_h - 1); (0, 0) |] 0 h_line
      in
      let v_line = Nx.ones Nx.uint8 [| kh; 1 |] in
      let v_padded =
        Nx.pad [| (0, 0); (center_w, kw - center_w - 1) |] 0 v_line
      in
      Nx.maximum h_padded v_padded
  | Ellipse ->
      let cx = float (kw / 2) and cy = float (kh / 2) in
      let rx = cx +. 0.5 and ry = cy +. 0.5 in
      let data =
        Array.init (kh * kw) (fun idx ->
            let y = float (idx / kw) -. cy in
            let x = float (idx mod kw) -. cx in
            if (x *. x /. (rx *. rx)) +. (y *. y /. (ry *. ry)) <= 1.0 then 1
            else 0)
      in
      Nx.create Nx.uint8 [| kh; kw |] data

(* A kernel's extents and active (non-zero) positions, read once to OCaml: a
   composite operation reads them once for all its erosions and dilations. *)
type neighborhood = { kh : int; kw : int; positions : (int * int) list }

let neighborhood kernel =
  let kshape = Nx.shape kernel in
  let kh = kshape.(0) and kw = kshape.(1) in
  let k = Nx.to_array kernel in
  let positions = ref [] in
  for i = 0 to kh - 1 do
    for j = 0 to kw - 1 do
      if k.((i * kw) + j) <> 0 then positions := (i, j) :: !positions
    done
  done;
  match !positions with
  | [] ->
      invalid_arg "structuring element must have at least one non-zero element"
  | ps -> { kh; kw; positions = List.rev ps }

let morph_reduce op slices =
  match slices with
  | [] -> failwith "empty slice list"
  | first :: rest -> List.fold_left (fun acc s -> op acc s) first rest

let morph_op (type a b d) ~op { kh; kw; positions } (img : (a, b, d) Nx.t) :
    (a, b, d) Nx.t =
  let pad_h = kh / 2 and pad_w = kw / 2 in
  let reduce =
    match op with
    | `Min -> morph_reduce Nx.minimum
    | `Max -> morph_reduce Nx.maximum
  in
  let dt = Nx.dtype img in
  (* For erosion, pad with max so boundary doesn't create false minima. For
     dilation, pad with min (zeros). *)
  let pad_val : a =
    match op with
    | `Max -> Nx.Dtype.zero dt
    | `Min -> Nx.Dtype.max_value dt
  in
  Helpers.with_batch
    (fun img ->
      let shape = Nx.shape img in
      let h = shape.(1) and w = shape.(2) in
      let padding = [| (0, 0); (pad_h, pad_h); (pad_w, pad_w); (0, 0) |] in
      let padded = Nx.pad padding pad_val img in
      let slices =
        List.map
          (fun (dy, dx) ->
            Nx.slice [ Nx.A; Nx.R (dy, dy + h); Nx.R (dx, dx + w); Nx.A ] padded)
          positions
      in
      reduce slices)
    img

let erode ~kernel img = morph_op ~op:`Min (neighborhood kernel) img
let dilate ~kernel img = morph_op ~op:`Max (neighborhood kernel) img

let opening ~kernel img =
  let n = neighborhood kernel in
  morph_op ~op:`Max n (morph_op ~op:`Min n img)

let closing ~kernel img =
  let n = neighborhood kernel in
  morph_op ~op:`Min n (morph_op ~op:`Max n img)

let morphological_gradient ~kernel img =
  let n = neighborhood kernel in
  Nx.sub (morph_op ~op:`Max n img) (morph_op ~op:`Min n img)
