(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type kernel_shape = Rect | Cross | Ellipse

let structuring_element shape (kh, kw) =
  if kh <= 0 || kw <= 0 || kh mod 2 = 0 || kw mod 2 = 0 then
    invalid_arg "structuring_element: dimensions must be positive odd integers";
  match shape with
  | Rect -> Rune.ones Rune.uint8 [| kh; kw |]
  | Cross ->
      let center_h = kh / 2 and center_w = kw / 2 in
      let h_line = Rune.ones Rune.uint8 [| 1; kw |] in
      let h_padded =
        Rune.pad [| (center_h, kh - center_h - 1); (0, 0) |] 0 h_line
      in
      let v_line = Rune.ones Rune.uint8 [| kh; 1 |] in
      let v_padded =
        Rune.pad [| (0, 0); (center_w, kw - center_w - 1) |] 0 v_line
      in
      Rune.maximum h_padded v_padded
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
      Rune.create Rune.uint8 [| kh; kw |] data

(* Find active positions (non-zero) in a kernel *)
let active_positions kernel =
  let kshape = Rune.shape kernel in
  let kh = kshape.(0) and kw = kshape.(1) in
  let positions = ref [] in
  for i = 0 to kh - 1 do
    for j = 0 to kw - 1 do
      if Rune.item [ i; j ] kernel <> 0 then positions := (i, j) :: !positions
    done
  done;
  match !positions with
  | [] ->
      invalid_arg "structuring element must have at least one non-zero element"
  | ps -> List.rev ps

let morph_reduce op slices =
  match slices with
  | [] -> failwith "empty slice list"
  | first :: rest -> List.fold_left (fun acc s -> op acc s) first rest

let morph_op (type a b) ~op ~kernel (img : (a, b) Rune.t) : (a, b) Rune.t =
  let kshape = Rune.shape kernel in
  let kh = kshape.(0) and kw = kshape.(1) in
  let pad_h = kh / 2 and pad_w = kw / 2 in
  let positions = active_positions kernel in
  let reduce =
    match op with
    | `Min -> morph_reduce Rune.minimum
    | `Max -> morph_reduce Rune.maximum
  in
  let dt = Rune.dtype img in
  (* For erosion, pad with max so boundary doesn't create false minima. For
     dilation, pad with min (zeros). *)
  let pad_val : a =
    match op with
    | `Max -> Nx_core.Dtype.zero dt
    | `Min -> Nx_core.Dtype.max_value dt
  in
  Helpers.with_batch
    (fun img ->
      let shape = Rune.shape img in
      let h = shape.(1) and w = shape.(2) in
      let padding = [| (0, 0); (pad_h, pad_h); (pad_w, pad_w); (0, 0) |] in
      let padded = Rune.pad padding pad_val img in
      let slices =
        List.map
          (fun (dy, dx) ->
            Rune.slice
              [ Rune.A; Rune.R (dy, dy + h); Rune.R (dx, dx + w); Rune.A ]
              padded)
          positions
      in
      reduce slices)
    img

let erode ~kernel img = morph_op ~op:`Min ~kernel img
let dilate ~kernel img = morph_op ~op:`Max ~kernel img
let opening ~kernel img = dilate ~kernel (erode ~kernel img)
let closing ~kernel img = erode ~kernel (dilate ~kernel img)

let morphological_gradient ~kernel img =
  Rune.sub (dilate ~kernel img) (erode ~kernel img)
