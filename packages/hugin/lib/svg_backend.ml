(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* SVG backend *)

(* Text measurer *)

let text_measurer ~(font : Theme.font) s =
  let w = float (String.length s) *. font.size *. 0.6 in
  let h = font.size in
  (w, h)

(* Helpers *)

let color_to_rgb_string c =
  let r, g, b, _ = Color.to_rgba c in
  Printf.sprintf "rgb(%d,%d,%d)"
    (Float.to_int (r *. 255.))
    (Float.to_int (g *. 255.))
    (Float.to_int (b *. 255.))

let add_fill buf = function
  | None -> Buffer.add_string buf " fill=\"none\""
  | Some c ->
      Printf.bprintf buf " fill=\"%s\"" (color_to_rgb_string c);
      let a = Color.alpha c in
      if a < 1. then Printf.bprintf buf " fill-opacity=\"%.3g\"" a

let add_stroke buf = function
  | None -> Buffer.add_string buf " stroke=\"none\""
  | Some c ->
      Printf.bprintf buf " stroke=\"%s\"" (color_to_rgb_string c);
      let a = Color.alpha c in
      if a < 1. then Printf.bprintf buf " stroke-opacity=\"%.3g\"" a

let text_anchor_string = function
  | `Start -> "start"
  | `Middle -> "middle"
  | `End -> "end"

let dominant_baseline_string = function
  | `Top -> "text-before-edge"
  | `Middle -> "central"
  | `Bottom -> "text-after-edge"

let escape_xml s =
  let buf = Buffer.create (String.length s) in
  String.iter
    (function
      | '<' -> Buffer.add_string buf "&lt;"
      | '>' -> Buffer.add_string buf "&gt;"
      | '&' -> Buffer.add_string buf "&amp;"
      | '"' -> Buffer.add_string buf "&quot;"
      | c -> Buffer.add_char buf c)
    s;
  Buffer.contents buf

(* Marker shapes *)

let marker_path shape size =
  let hs = size /. 2. in
  match shape with
  | Spec.Circle ->
      Printf.sprintf "M %g 0 A %g %g 0 1 1 %g 0 A %g %g 0 1 1 %g 0 Z" (-.hs) hs
        hs hs hs hs (-.hs)
  | Spec.Square ->
      Printf.sprintf "M %g %g L %g %g L %g %g L %g %g Z" (-.hs) (-.hs) hs (-.hs)
        hs hs (-.hs) hs
  | Spec.Triangle ->
      Printf.sprintf "M 0 %g L %g %g L %g %g Z" (-.hs) hs hs (-.hs) hs
  | Spec.Plus ->
      Printf.sprintf "M %g 0 L %g 0 M 0 %g L 0 %g" (-.hs) hs (-.hs) hs
  | Spec.Star ->
      let d = hs *. 0.707 in
      Printf.sprintf
        "M %g 0 L %g 0 M 0 %g L 0 %g M %g %g L %g %g M %g %g L %g %g" (-.hs) hs
        (-.hs) hs (-.d) (-.d) d d d (-.d) (-.d) d

(* Primitive rendering — ids threaded through to avoid global state *)

type ids = { mutable clip_id : int; mutable marker_id : int }

let fresh_clip ids =
  ids.clip_id <- ids.clip_id + 1;
  Printf.sprintf "clip-%d" ids.clip_id

let fresh_marker ids =
  ids.marker_id <- ids.marker_id + 1;
  Printf.sprintf "marker-%d" ids.marker_id

let rec render_primitive ids buf = function
  | Scene.Path { points; close; fill; stroke; line_width; dash } ->
      if Array.length points < 2 then ()
      else begin
        Buffer.add_string buf "<path d=\"";
        Array.iteri
          (fun i (x, y) ->
            if i = 0 then Printf.bprintf buf "M %g %g" x y
            else Printf.bprintf buf " L %g %g" x y)
          points;
        if close then Buffer.add_string buf " Z";
        Buffer.add_char buf '"';
        add_fill buf fill;
        add_stroke buf stroke;
        if line_width > 0. then
          Printf.bprintf buf " stroke-width=\"%g\"" line_width;
        begin match dash with
        | [] -> ()
        | ds ->
            Buffer.add_string buf " stroke-dasharray=\"";
            List.iteri
              (fun i d ->
                if i > 0 then Buffer.add_char buf ',';
                Printf.bprintf buf "%g" d)
              ds;
            Buffer.add_char buf '"'
        end;
        Buffer.add_string buf "/>\n"
      end
  | Scene.Markers { points; shape; size; sizes; fill; fills; stroke } ->
      let stroke_only =
        match shape with Spec.Plus | Spec.Star -> true | _ -> false
      in
      begin match (fills, sizes) with
      | None, None ->
          let id = fresh_marker ids in
          let d = marker_path shape size in
          Printf.bprintf buf "<defs><symbol id=\"%s\"><path d=\"%s\"" id d;
          if stroke_only then begin
            Buffer.add_string buf " fill=\"none\"";
            let stroke_c = match fill with Some _ -> fill | None -> stroke in
            add_stroke buf stroke_c;
            Printf.bprintf buf " stroke-width=\"%g\""
              (Float.max 1. (size *. 0.15))
          end
          else begin
            add_fill buf fill;
            add_stroke buf stroke;
            if stroke <> None then Buffer.add_string buf " stroke-width=\"1\""
          end;
          Buffer.add_string buf "/></symbol></defs>\n";
          Array.iter
            (fun (x, y) ->
              Printf.bprintf buf "<use href=\"#%s\" x=\"%g\" y=\"%g\"/>\n" id x
                y)
            points
      | _ ->
          Array.iteri
            (fun i (x, y) ->
              let s = match sizes with Some ss -> ss.(i) | None -> size in
              let f =
                match fills with Some fs -> Some fs.(i) | None -> fill
              in
              let d = marker_path shape s in
              Printf.bprintf buf "<path d=\"%s\" transform=\"translate(%g,%g)\""
                d x y;
              if stroke_only then begin
                Buffer.add_string buf " fill=\"none\"";
                let stroke_c = match f with Some _ -> f | None -> stroke in
                add_stroke buf stroke_c;
                Printf.bprintf buf " stroke-width=\"%g\""
                  (Float.max 1. (s *. 0.15))
              end
              else begin
                add_fill buf f;
                add_stroke buf stroke;
                if stroke <> None then
                  Buffer.add_string buf " stroke-width=\"1\""
              end;
              Buffer.add_string buf "/>\n")
            points
      end
  | Scene.Text { x; y; content; font; color; anchor; baseline; angle } ->
      Printf.bprintf buf "<text x=\"%g\" y=\"%g\"" x y;
      Printf.bprintf buf " font-family=\"%s\"" font.Theme.family;
      Printf.bprintf buf " font-size=\"%g\"" font.size;
      begin match font.weight with
      | `Bold -> Buffer.add_string buf " font-weight=\"bold\""
      | `Normal -> ()
      end;
      Printf.bprintf buf " fill=\"%s\"" (color_to_rgb_string color);
      let a = Color.alpha color in
      if a < 1. then Printf.bprintf buf " fill-opacity=\"%.3g\"" a;
      Printf.bprintf buf " text-anchor=\"%s\"" (text_anchor_string anchor);
      Printf.bprintf buf " dominant-baseline=\"%s\""
        (dominant_baseline_string baseline);
      if angle <> 0. then
        Printf.bprintf buf " transform=\"rotate(%g,%g,%g)\""
          (angle *. -180. /. Float.pi)
          x y;
      Printf.bprintf buf ">%s</text>\n" (escape_xml content)
  | Scene.Image { x; y; w; h; data } ->
      let b64 = Image_util.nx_to_png_base64 data in
      Printf.bprintf buf "<image x=\"%g\" y=\"%g\" width=\"%g\" height=\"%g\"" x
        y w h;
      Printf.bprintf buf " href=\"data:image/png;base64,%s\"/>\n" b64
  | Scene.Clip { x; y; w; h; children } ->
      let id = fresh_clip ids in
      Printf.bprintf buf
        "<defs><clipPath id=\"%s\"><rect x=\"%g\" y=\"%g\" width=\"%g\" \
         height=\"%g\"/></clipPath></defs>\n"
        id x y w h;
      Printf.bprintf buf "<g clip-path=\"url(#%s)\">\n" id;
      List.iter (render_primitive ids buf) children;
      Buffer.add_string buf "</g>\n"
  | Scene.Group children ->
      Buffer.add_string buf "<g>\n";
      List.iter (render_primitive ids buf) children;
      Buffer.add_string buf "</g>\n"

(* Entry points *)

let render (scene : Scene.t) =
  let ids = { clip_id = 0; marker_id = 0 } in
  let buf = Buffer.create 4096 in
  Printf.bprintf buf "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  Printf.bprintf buf
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%g\" height=\"%g\" \
     viewBox=\"0 0 %g %g\">\n"
    scene.width scene.height scene.width scene.height;
  List.iter (render_primitive ids buf) scene.primitives;
  Buffer.add_string buf "</svg>\n";
  Buffer.contents buf

let render_to_file filename scene =
  let s = render scene in
  let oc = open_out filename in
  output_string oc s;
  close_out oc
