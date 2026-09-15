(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module V = Hugin_vg

let color c =
  let r, g, b, a = Color.to_rgba c in
  V.Color.v ~a r g b

let face (font : Theme.font) =
  match font.weight with `Bold -> V.Font.bold | `Normal -> V.Font.regular

let text_measurer ~(font : Theme.font) s =
  match V.Font.bounds (face font) ~size:font.size s with
  | Some b -> (V.Box.width b, V.Box.height b)
  | None -> (0., 0.)

let split points = (Array.map fst points, Array.map snd points)

(* Markers *)

let marker_path shape size =
  let hs = size /. 2. in
  let line xs ys = V.Path.polyline xs ys in
  match shape with
  | Spec.Circle -> V.Path.circle 0. 0. hs
  | Spec.Square -> V.Path.rect (-.hs) (-.hs) size size
  | Spec.Triangle -> V.Path.polygon [| 0.; hs; -.hs |] [| -.hs; hs; hs |]
  | Spec.Plus ->
      V.Path.append
        (line [| -.hs; hs |] [| 0.; 0. |])
        (line [| 0.; 0. |] [| -.hs; hs |])
  | Spec.Star ->
      let d = hs *. 0.707 in
      List.fold_left V.Path.append V.Path.empty
        [
          line [| -.hs; hs |] [| 0.; 0. |];
          line [| 0.; 0. |] [| -.hs; hs |];
          line [| -.d; d |] [| -.d; d |];
          line [| d; -.d |] [| -.d; d |];
        ]

(* A marker centred on the origin. Plus and star are stroke-only, drawn in the
   fill color when one is given. *)
let marker shape size fill stroke =
  let path = marker_path shape size in
  let width = V.Stroke.v (Float.max 1. (size *. 0.15)) in
  match shape with
  | Spec.Plus | Spec.Star ->
      let c =
        match (fill, stroke) with
        | Some c, _ | None, Some c -> c
        | None, None -> Color.black
      in
      V.Picture.stroke width (color c) path
  | Spec.Circle | Spec.Square | Spec.Triangle ->
      V.Picture.group
        [
          (match fill with
          | Some c -> V.Picture.fill (color c) path
          | None -> V.Picture.empty);
          (match stroke with
          | Some c -> V.Picture.stroke width (color c) path
          | None -> V.Picture.empty);
        ]

(* Primitives *)

let rec primitive = function
  | Scene.Path { points; close; fill; stroke; line_width; dash } ->
      let xs, ys = split points in
      let path =
        if close then V.Path.polygon xs ys else V.Path.polyline xs ys
      in
      V.Picture.group
        [
          (match fill with
          | Some c -> V.Picture.fill (color c) path
          | None -> V.Picture.empty);
          (match stroke with
          | Some c ->
              V.Picture.stroke
                (V.Stroke.v ~dash:(Array.of_list dash) line_width)
                (color c) path
          | None -> V.Picture.empty);
        ]
  | Scene.Markers
      { points; shape; size; sizes = None; fill; fills = None; stroke } ->
      let xs, ys = split points in
      V.Picture.stamp (marker shape size fill stroke) xs ys
  | Scene.Markers { points; shape; size; sizes; fill; fills; stroke } ->
      V.Picture.group
        (Array.to_list
           (Array.mapi
              (fun i (x, y) ->
                let size = match sizes with Some s -> s.(i) | None -> size in
                let fill =
                  match fills with Some f -> Some f.(i) | None -> fill
                in
                V.Picture.transform (V.Affine.translate x y)
                  (marker shape size fill stroke))
              points))
  | Scene.Text { x; y; content; font; color = c; anchor; baseline; angle } ->
      let f = face font in
      let b =
        Option.value
          (V.Font.bounds f ~size:font.size content)
          ~default:(V.Box.v 0. 0. 0. 0.)
      in
      let dx =
        match anchor with
        | `Start -> -.b.x0
        | `Middle -> -.(b.x0 +. (V.Box.width b /. 2.))
        | `End -> -.b.x1
      in
      let dy =
        match baseline with
        | `Top -> -.b.y0
        | `Middle -> -.(b.y0 +. (V.Box.height b /. 2.))
        | `Bottom -> -.b.y1
      in
      let m =
        if angle = 0. then V.Affine.translate x y
        else V.Affine.(translate x y * rotate angle)
      in
      V.Picture.transform m
        (V.Picture.text f ~size:font.size (color c) ~x:dx ~y:dy content)
  | Scene.Image { x; y; w; h; data } -> V.Picture.image ~x ~y ~w ~h data
  | Scene.Clip { x; y; w; h; children } ->
      V.Picture.clip (V.Path.rect x y w h)
        (V.Picture.group (List.map primitive children))
  | Scene.Group children -> V.Picture.group (List.map primitive children)

let picture (scene : Scene.t) =
  V.Picture.group (List.map primitive scene.primitives)

(* Entry points *)

let raster ~width ~height scene =
  Hugin_vg_raster.render ~width:(int_of_float width)
    ~height:(int_of_float height) (picture scene)

let render_png filename ~width ~height scene =
  Nx_io.save_image filename (raster ~width ~height scene)

let render_to_buffer ~width ~height scene =
  Nx_io.encode_png (raster ~width ~height scene)

let write filename s =
  let oc = open_out_bin filename in
  Fun.protect ~finally:(fun () -> close_out oc) (fun () -> output_string oc s)

let render_pdf filename ~width ~height scene =
  write filename (Hugin_vg_pdf.render ~width ~height (picture scene))

let render_svg ~width ~height scene =
  Hugin_vg_svg.render ~width ~height (picture scene)
