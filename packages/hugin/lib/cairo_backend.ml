(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Color helpers *)

let set_color cr c =
  let r, g, b, a = Color.to_rgba c in
  Ucairo.set_source_rgba cr r g b a

let set_font cr (font : Theme.font) =
  let weight =
    match font.weight with `Normal -> Ucairo.Normal | `Bold -> Ucairo.Bold
  in
  Ucairo.select_font_face cr font.family weight;
  Ucairo.set_font_size cr font.size

(* Text measurer *)

let text_measurer cr ~font s =
  set_font cr font;
  let ext = Ucairo.text_extents cr s in
  (ext.width, ext.height)

(* Marker rendering *)

let draw_marker cr shape size (px, py) =
  let hs = size /. 2. in
  match shape with
  | Spec.Circle ->
      Ucairo.arc cr px py ~r:hs ~a1:0. ~a2:(2. *. Float.pi);
      Ucairo.Path.close cr
  | Spec.Square -> Ucairo.rectangle cr (px -. hs) (py -. hs) ~w:size ~h:size
  | Spec.Triangle ->
      Ucairo.move_to cr px (py -. hs);
      Ucairo.line_to cr (px +. hs) (py +. hs);
      Ucairo.line_to cr (px -. hs) (py +. hs);
      Ucairo.Path.close cr
  | Spec.Plus ->
      Ucairo.move_to cr (px -. hs) py;
      Ucairo.line_to cr (px +. hs) py;
      Ucairo.move_to cr px (py -. hs);
      Ucairo.line_to cr px (py +. hs)
  | Spec.Star ->
      Ucairo.move_to cr (px -. hs) py;
      Ucairo.line_to cr (px +. hs) py;
      Ucairo.move_to cr px (py -. hs);
      Ucairo.line_to cr px (py +. hs);
      let d = hs *. 0.707 in
      Ucairo.move_to cr (px -. d) (py -. d);
      Ucairo.line_to cr (px +. d) (py +. d);
      Ucairo.move_to cr (px +. d) (py -. d);
      Ucairo.line_to cr (px -. d) (py +. d)

(* Primitive rendering *)

let rec render_primitive cr = function
  | Scene.Path { points; close; fill; stroke; line_width; dash } ->
      if Array.length points < 2 then ()
      else begin
        let x0, y0 = points.(0) in
        Ucairo.move_to cr x0 y0;
        for i = 1 to Array.length points - 1 do
          let x, y = points.(i) in
          Ucairo.line_to cr x y
        done;
        if close then Ucairo.Path.close cr;
        begin match fill with
        | Some c ->
            set_color cr c;
            if stroke <> None then Ucairo.fill_preserve cr else Ucairo.fill cr
        | None -> ()
        end;
        begin match stroke with
        | Some c ->
            set_color cr c;
            Ucairo.set_line_width cr line_width;
            (match dash with
            | [] -> Ucairo.set_dash cr [||]
            | ds -> Ucairo.set_dash cr (Array.of_list ds));
            Ucairo.stroke cr
        | None -> ()
        end
      end
  | Scene.Markers { points; shape; size; sizes; fill; fills; stroke } ->
      let stroke_only =
        match shape with Spec.Plus | Spec.Star -> true | _ -> false
      in
      Array.iteri
        (fun i pt ->
          let s = match sizes with Some ss -> ss.(i) | None -> size in
          let f = match fills with Some fs -> Some fs.(i) | None -> fill in
          Ucairo.Path.clear cr;
          draw_marker cr shape s pt;
          if stroke_only then begin
            let c =
              match f with
              | Some c -> c
              | None -> ( match stroke with Some c -> c | None -> Color.black)
            in
            set_color cr c;
            Ucairo.set_line_width cr (Float.max 1. (s *. 0.15));
            Ucairo.stroke cr
          end
          else begin
            begin match f with
            | Some c ->
                set_color cr c;
                if stroke <> None then Ucairo.fill_preserve cr
                else Ucairo.fill cr
            | None -> ()
            end;
            begin match stroke with
            | Some c ->
                set_color cr c;
                Ucairo.set_line_width cr (Float.max 1. (s *. 0.15));
                Ucairo.stroke cr
            | None -> ()
            end
          end)
        points
  | Scene.Text { x; y; content; font; color; anchor; baseline; angle } ->
      set_font cr font;
      set_color cr color;
      let ext = Ucairo.text_extents cr content in
      let dx =
        match anchor with
        | `Start -> -.ext.x_bearing
        | `Middle -> -.(ext.x_bearing +. (ext.width /. 2.))
        | `End -> -.(ext.x_bearing +. ext.width)
      in
      let dy =
        match baseline with
        | `Top -> -.ext.y_bearing
        | `Middle -> -.(ext.y_bearing +. (ext.height /. 2.))
        | `Bottom -> -.(ext.y_bearing +. ext.height)
      in
      Ucairo.save cr;
      Ucairo.translate cr x y;
      if angle <> 0. then Ucairo.rotate cr angle;
      Ucairo.move_to cr dx dy;
      Ucairo.show_text cr content;
      Ucairo.restore cr
  | Scene.Image { x; y; w; h; data } ->
      let img_surface = Image_util.nx_to_cairo_surface data in
      let img_w = (Nx.shape data).(1) and img_h = (Nx.shape data).(0) in
      Ucairo.save cr;
      Ucairo.translate cr x y;
      Ucairo.scale cr (w /. float img_w) (h /. float img_h);
      Ucairo.set_source_surface cr img_surface ~x:0. ~y:0.;
      Ucairo.paint cr;
      Ucairo.restore cr;
      Ucairo.Surface.finish img_surface
  | Scene.Clip { x; y; w; h; children } ->
      Ucairo.save cr;
      Ucairo.rectangle cr x y ~w ~h;
      Ucairo.clip cr;
      List.iter (render_primitive cr) children;
      Ucairo.restore cr
  | Scene.Group children -> List.iter (render_primitive cr) children

(* Scene rendering *)

let render_scene cr (scene : Scene.t) =
  Ucairo.set_antialias cr Ucairo.Antialias_default;
  Ucairo.set_line_cap cr Ucairo.Round;
  Ucairo.set_line_join cr Ucairo.Join_round;
  List.iter (render_primitive cr) scene.primitives

(* Entry points *)

let render_to_png filename ~width ~height (scene : Scene.t) =
  let w = int_of_float width and h = int_of_float height in
  let surface = Ucairo.Image.create ~w ~h in
  let cr = Ucairo.create surface in
  render_scene cr scene;
  Ucairo.Png.write surface filename;
  Ucairo.Surface.finish surface

let render_to_pdf filename ~width ~height (scene : Scene.t) =
  let surface = Ucairo.Pdf.create filename ~w:width ~h:height in
  let cr = Ucairo.create surface in
  render_scene cr scene;
  Ucairo.Surface.finish surface

let render_to_buffer ~width ~height (scene : Scene.t) =
  let w = int_of_float width and h = int_of_float height in
  let surface = Ucairo.Image.create ~w ~h in
  let cr = Ucairo.create surface in
  render_scene cr scene;
  let buf = Buffer.create 4096 in
  Ucairo.Png.write_to_stream surface (Buffer.add_string buf);
  Ucairo.Surface.finish surface;
  Buffer.contents buf

let show_interactive ~theme ~width ~height prepared =
  let w = int_of_float width and h = int_of_float height in
  let csdl = Cairo_sdl.create ~width:w ~height:h ~title:"Hugin" in

  let render_current () =
    let cr = Cairo_sdl.context csdl in
    let cw = float (Cairo_sdl.width csdl) in
    let ch = float (Cairo_sdl.height csdl) in
    let tm = text_measurer cr in
    let scene =
      Resolve.resolve_prepared ~text_measurer:tm ~theme ~width:cw ~height:ch
        prepared
    in
    render_scene cr scene;
    Cairo_sdl.present csdl
  in

  render_current ();

  let ev = Usdl.Event.create () in
  let quit = ref false in
  while not !quit do
    if not (Usdl.Event.wait ev) then quit := true
    else begin
      match Usdl.Event.typ ev with
      | `Quit -> quit := true
      | `Window_event -> begin
          match Usdl.Event.window_event_id ev with
          | `Resized | `Size_changed ->
              Cairo_sdl.resize csdl;
              render_current ()
          | `Exposed -> render_current ()
          | `Close -> quit := true
          | _ -> ()
        end
      | `Key_down ->
          let keycode = Usdl.Event.keycode ev in
          if keycode = Usdl.Keycode.escape || keycode = Usdl.Keycode.q then
            quit := true
      | _ -> ()
    end
  done;
  Cairo_sdl.destroy csdl
