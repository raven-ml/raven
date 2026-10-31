(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type kernel_shape = Rect | Cross | Ellipse

let structuring_element shape (kh, kw) =
  if kh <= 0 || kw <= 0 || kh mod 2 = 0 || kw mod 2 = 0 then
    invalid_arg "structuring_element: dimensions must be positive odd integers";
  match shape with
  | Rect -> Nx.full Nx.bool [| kh; kw |] true
  | Cross ->
      let ch = kh / 2 and cw = kw / 2 in
      Nx.init Nx.bool [| kh; kw |] (fun idx -> idx.(0) = ch || idx.(1) = cw)
  | Ellipse ->
      let cx = float (kw / 2) and cy = float (kh / 2) in
      let rx = cx +. 0.5 and ry = cy +. 0.5 in
      Nx.init Nx.bool [| kh; kw |] (fun idx ->
          let y = float idx.(0) -. cy and x = float idx.(1) -. cx in
          (x *. x /. (rx *. rx)) +. (y *. y /. (ry *. ry)) <= 1.0)

(* A kernel's [true] entries as offsets from its centre, row [kh / 2] and
   column [kw / 2], read once to OCaml: a composite operation reads them once
   for all its erosions and dilations. The image is padded by [kh / 2] rows
   and [kw / 2] columns on each side, which holds the offsets, from
   [-(kh / 2)] to [kh - 1 - kh / 2], and their reflections. *)
type neighborhood = { pad_h : int; pad_w : int; offsets : (int * int) list }

let neighborhood kernel =
  let kshape = Nx.shape kernel in
  let rank = Array.length kshape in
  if rank <> 2 then
    invalid_arg
      (Printf.sprintf "structuring element must have rank 2, got rank %d" rank);
  let kh = kshape.(0) and kw = kshape.(1) in
  let k = Nx.to_array kernel in
  let offsets = ref [] in
  for i = kh - 1 downto 0 do
    for j = kw - 1 downto 0 do
      if k.((i * kw) + j) then
        offsets := (i - (kh / 2), j - (kw / 2)) :: !offsets
    done
  done;
  if !offsets = [] then
    invalid_arg "structuring element must have at least one true element";
  { pad_h = kh / 2; pad_w = kw / 2; offsets = !offsets }

(* The neighbourhood reflected about its centre: the pixel [dy] rows below
   becomes the one [dy] rows above, and likewise for columns. *)
let reflect n =
  { n with offsets = List.map (fun (dy, dx) -> (-dy, -dx)) n.offsets }

let morph_reduce op slices =
  match slices with
  | [] -> failwith "empty slice list"
  | first :: rest -> List.fold_left (fun acc s -> op acc s) first rest

let morph_op (type a b d) ~op { pad_h; pad_w; offsets } (img : (a, b, d) Nx.t) :
    (a, b, d) Nx.t =
  let reduce =
    match op with
    | `Min -> morph_reduce Nx.minimum
    | `Max -> morph_reduce Nx.maximum
  in
  let dt = Nx.dtype img in
  (* The padding is the identity of the reduction, so that no pixel outside
     the image is an extreme: the dtype's largest value for erosion and its
     least for dilation. *)
  let pad_val : a =
    match op with
    | `Max -> Nx.Dtype.min_value dt
    | `Min -> Nx.Dtype.max_value dt
  in
  Helpers.with_batch
    (fun img ->
      let shape = Nx.shape img in
      let h = shape.(1) and w = shape.(2) in
      let padding = [| (0, 0); (pad_h, pad_h); (pad_w, pad_w); (0, 0) |] in
      let padded = Nx.pad padding pad_val img in
      let slice (dy, dx) =
        let y = pad_h + dy and x = pad_w + dx in
        Nx.slice [ Nx.A; Nx.R (y, y + h); Nx.R (x, x + w); Nx.A ] padded
      in
      reduce (List.map slice offsets))
    img

let erode ~kernel img = morph_op ~op:`Min (neighborhood kernel) img
let dilate ~kernel img = morph_op ~op:`Max (neighborhood kernel) img

(* The dilation by the reflected element is the adjoint of the erosion, which
   makes the opening at most the image and idempotent; for a symmetric
   element it is the plain dilation. Closing is dual. *)
let opening ~kernel img =
  let n = neighborhood kernel in
  morph_op ~op:`Max (reflect n) (morph_op ~op:`Min n img)

let closing ~kernel img =
  let n = neighborhood kernel in
  morph_op ~op:`Min (reflect n) (morph_op ~op:`Max n img)

(* Where a neighbourhood holds a pixel of the image, the erosion is at most
   the dilation, so the difference does not wrap; on booleans it is the
   dilation without the erosion. *)
let morphological_gradient (type a b d) ~kernel (img : (a, b, d) Nx.t) :
    (a, b, d) Nx.t =
  let n = neighborhood kernel in
  let dilated = morph_op ~op:`Max n img and eroded = morph_op ~op:`Min n img in
  match Nx.Dtype.kind (Nx.dtype img) with
  | Nx.Dtype.Boolean -> Nx.logical_and dilated (Nx.logical_not eroded)
  | _ -> Nx.sub dilated eroded
