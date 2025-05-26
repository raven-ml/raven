let render_line2d cr (t_ctx : Transforms.context2d) (l : Artist.line2d) =
  if Nx.size l.xdata < 2 then () else Cairo.save cr;
  Render_utils.set_source_color cr l.color;
  Cairo.set_line_width cr l.linewidth;
  (match l.linestyle with
  | Dashed -> Cairo.set_dash cr ~ofs:0.0 [| 6.0; 6.0 |]
  | Dotted -> Cairo.set_dash cr ~ofs:0.0 [| 1.0; 3.0 |]
  | DashDot -> Cairo.set_dash cr ~ofs:0.0 [| 6.0; 3.0; 1.0; 3.0 |]
  | Solid | None -> Cairo.set_dash cr ~ofs:0.0 [||]);

  let x0 = Nx.get_item [ 0 ] l.xdata in
  let y0 = Nx.get_item [ 0 ] l.ydata in
  let px0, py0 = Transforms.transform t_ctx ~x:x0 ~y:y0 in
  Cairo.move_to cr px0 py0;

  for i = 1 to Nx.size l.xdata - 1 do
    let x = Nx.get_item [ i ] l.xdata in
    let y = Nx.get_item [ i ] l.ydata in
    let px, py = Transforms.transform t_ctx ~x ~y in
    Cairo.line_to cr px py
  done;
  Cairo.stroke cr;
  Cairo.restore cr

let render_image cr (t_ctx : Transforms.context2d) (img : Artist.image) =
  try
    let surface =
      match img.data with
      | Artist.Uint8_image u8_data ->
          Render_utils.to_cairo_surface ?cmap:img.cmap u8_data
      | Artist.Float32_image f32_data ->
          Render_utils.float32_to_cairo_surface ?cmap:img.cmap f32_data
    in
    let img_w_pix = float (Cairo.Image.get_width surface) in
    let img_h_pix = float (Cairo.Image.get_height surface) in

    if img_w_pix <= 0. || img_h_pix <= 0. then
      failwith "Image has zero width or height";

    Cairo.save cr;
    let pb = t_ctx.pixel_bounds in
    if pb.width <= 0. || pb.height <= 0. then (
      Printf.eprintf
        "Warning: Skipping image rendering due to zero-sized pixel bounds.\n%!";
      Cairo.restore cr)
    else (
      Cairo.rectangle cr pb.left pb.top ~w:pb.width ~h:pb.height;
      Cairo.clip cr;
      let scale_x = pb.width /. img_w_pix in
      let scale_y = pb.height /. img_h_pix in
      Cairo.translate cr pb.left pb.top;
      Cairo.scale cr scale_x scale_y;
      Cairo.set_source_surface cr surface ~x:0. ~y:0.;
      let pattern = Cairo.get_source cr in
      Cairo.Pattern.set_filter pattern BEST;
      Cairo.paint cr;
      Cairo.restore cr)
  with ex -> (
    Printexc.print_backtrace stderr;
    Printf.eprintf "Unexpected error rendering image: %s\n"
      (Printexc.to_string ex);
    try Cairo.restore cr with _ -> ())

let render_line3d cr (t_ctx : Transforms.context3d) (l : Artist.line3d) =
  if Nx.size l.xdata < 2 then () else Cairo.save cr;
  Render_utils.set_source_color cr l.color;
  Cairo.set_line_width cr l.linewidth;
  (match l.linestyle with
  | Dashed -> Cairo.set_dash cr ~ofs:0.0 [| 6.0; 6.0 |]
  | Dotted -> Cairo.set_dash cr ~ofs:0.0 [| 1.0; 3.0 |]
  | DashDot -> Cairo.set_dash cr ~ofs:0.0 [| 6.0; 3.0; 1.0; 3.0 |]
  | Solid | None -> Cairo.set_dash cr ~ofs:0.0 [||]);

  let first_visible_point = ref true in
  for i = 0 to Nx.size l.xdata - 1 do
    let x = Nx.get_item [ i ] l.xdata in
    let y = Nx.get_item [ i ] l.ydata in
    let z = Nx.get_item [ i ] l.zdata in
    match Transforms.transform3d t_ctx ~x ~y ~z with
    | Some (px, py) ->
        if !first_visible_point then Cairo.move_to cr px py
        else Cairo.line_to cr px py;
        first_visible_point := false
    | None ->
        if not !first_visible_point then Cairo.stroke cr;
        first_visible_point := true
  done;

  if not !first_visible_point then Cairo.stroke cr;
  Cairo.restore cr

let render_bar cr (t_ctx : Transforms.context2d) (b : Artist.bar) =
  let n = Nx.size b.x in
  if n = 0 || n <> Nx.size b.height then (
    Printf.eprintf
      "Warning: Bar data size mismatch (x vs height). Skipping render.\n%!";
    ())
  else Cairo.save cr;
  Render_utils.set_source_color cr b.color;
  let half_width = b.width /. 2.0 in
  for i = 0 to n - 1 do
    let x_center = Nx.get_item [ i ] b.x in
    let height = Nx.get_item [ i ] b.height in
    let y_bottom = b.bottom in
    let y_top = y_bottom +. height in

    let x_left = x_center -. half_width in
    let x_right = x_center +. half_width in

    let px_bl, py_bl = Transforms.transform t_ctx ~x:x_left ~y:y_bottom in
    let px_br, _py_br = Transforms.transform t_ctx ~x:x_right ~y:y_bottom in
    let _px_tl, py_tl = Transforms.transform t_ctx ~x:x_left ~y:y_top in

    if
      Float.is_finite px_bl && Float.is_finite py_bl && Float.is_finite px_br
      && Float.is_finite py_tl
    then
      let bar_pixel_width = px_br -. px_bl in
      let bar_pixel_height = py_bl -. py_tl in

      let rect_x = Float.min px_bl px_br in
      let rect_y = Float.min py_bl py_tl in
      let rect_w = Float.abs bar_pixel_width in
      let rect_h = Float.abs bar_pixel_height in

      if rect_w > 0. && rect_h > 0. then (
        Cairo.rectangle cr rect_x rect_y ~w:rect_w ~h:rect_h;
        Cairo.fill cr)
      else ()
  done;
  Cairo.restore cr

let render_text cr (t_ctx : Transforms.context2d) (t : Artist.text) =
  Cairo.save cr;
  let px, py = Transforms.transform t_ctx ~x:t.x ~y:t.y in

  if not (Float.is_finite px && Float.is_finite py) then Cairo.restore cr
  else (
    Render_utils.set_source_color cr t.color;
    Cairo.select_font_face cr "Sans" ~slant:Upright ~weight:Normal;
    Cairo.set_font_size cr t.fontsize;
    Cairo.move_to cr px py;
    Cairo.show_text cr t.content;
    Cairo.restore cr)

let render_scatter cr (t_ctx : Transforms.context2d) (s : Artist.scatter) =
  let n = Nx.size s.xdata in
  if n = 0 || n <> Nx.size s.ydata then (
    Printf.eprintf
      "Warning: Scatter data size mismatch (x vs y). Skipping render.\n%!";
    ())
  else Cairo.save cr;
  Render_utils.set_source_color cr s.c;
  let default_line_width = 1.0 in

  let radius = sqrt (s.s /. Float.pi) in
  let side = sqrt s.s in
  let plus_arm_half_length = sqrt s.s /. 2.0 in
  let triangle_height = sqrt s.s *. 1.2 in
  let triangle_base = triangle_height *. 0.866 *. 2. /. sqrt 3. in

  for i = 0 to n - 1 do
    let x = Nx.get_item [ i ] s.xdata in
    let y = Nx.get_item [ i ] s.ydata in
    let px, py = Transforms.transform t_ctx ~x ~y in

    if Float.is_finite px && Float.is_finite py then
      match s.marker with
      | Artist.Circle ->
          Cairo.arc cr px py ~r:radius ~a1:0. ~a2:(2. *. Float.pi);
          Cairo.fill cr
      | Artist.Point | Artist.Pixel ->
          Cairo.rectangle cr (px -. 0.5) (py -. 0.5) ~w:1.0 ~h:1.0;
          Cairo.fill cr
      | Artist.Square ->
          let x_tl = px -. (side /. 2.0) in
          let y_tl = py -. (side /. 2.0) in
          Cairo.rectangle cr x_tl y_tl ~w:side ~h:side;
          Cairo.fill cr
      | Artist.Triangle ->
          let half_base = triangle_base /. 2.0 in
          let half_height = triangle_height /. 2.0 in
          Cairo.move_to cr px (py -. half_height);
          Cairo.line_to cr (px +. half_base) (py +. half_height);
          Cairo.line_to cr (px -. half_base) (py +. half_height);
          Cairo.fill cr
      | Artist.Plus ->
          Cairo.set_line_width cr default_line_width;
          Cairo.move_to cr (px -. plus_arm_half_length) py;
          Cairo.line_to cr (px +. plus_arm_half_length) py;
          Cairo.move_to cr px (py -. plus_arm_half_length);
          Cairo.line_to cr px (py +. plus_arm_half_length);
          Cairo.stroke cr
      | Artist.Star ->
          Cairo.set_line_width cr default_line_width;
          Cairo.move_to cr (px -. plus_arm_half_length) py;
          Cairo.line_to cr (px +. plus_arm_half_length) py;
          Cairo.move_to cr px (py -. plus_arm_half_length);
          Cairo.line_to cr px (py +. plus_arm_half_length);
          Cairo.stroke cr
      | Artist.None -> ()
    else ()
  done;
  Cairo.restore cr

let render_errorbar cr (t_ctx : Transforms.context2d) (line : Artist.line2d)
    (style : Artist.errorbar_style) =
  Cairo.save cr;
  let n = Nx.size line.xdata in
  if n = 0 then (
    Cairo.restore cr;
    ())
  else
    let has_yerr = Option.is_some style.yerr in
    let has_xerr = Option.is_some style.xerr in

    Render_utils.set_source_color cr style.color;
    Cairo.set_line_width cr style.linewidth;
    for i = 0 to n - 1 do
      let x = Nx.get_item [ i ] line.xdata in
      let y = Nx.get_item [ i ] line.ydata in
      let px, py = Transforms.transform t_ctx ~x ~y in

      if Float.is_finite px && Float.is_finite py then (
        (if has_yerr then
           let y_err_val = Nx.get_item [ i ] (Option.get style.yerr) in
           let _px_low, py_low =
             Transforms.transform t_ctx ~x ~y:(y -. y_err_val)
           in
           let _px_high, py_high =
             Transforms.transform t_ctx ~x ~y:(y +. y_err_val)
           in
           if Float.is_finite py_low && Float.is_finite py_high then (
             Cairo.move_to cr px py_low;
             Cairo.line_to cr px py_high;
             Cairo.stroke cr;
             if style.capsize > 0. then (
               let half_cap = style.capsize /. 2.0 in
               Cairo.move_to cr (px -. half_cap) py_low;
               Cairo.line_to cr (px +. half_cap) py_low;
               Cairo.stroke cr;
               Cairo.move_to cr (px -. half_cap) py_high;
               Cairo.line_to cr (px +. half_cap) py_high;
               Cairo.stroke cr)));

        if has_xerr then
          let x_err_val = Nx.get_item [ i ] (Option.get style.xerr) in
          let px_low, _py_low =
            Transforms.transform t_ctx ~x:(x -. x_err_val) ~y
          in
          let px_high, _py_high =
            Transforms.transform t_ctx ~x:(x +. x_err_val) ~y
          in
          if Float.is_finite px_low && Float.is_finite px_high then (
            Cairo.move_to cr px_low py;
            Cairo.line_to cr px_high py;
            Cairo.stroke cr;
            if style.capsize > 0. then (
              let half_cap = style.capsize /. 2.0 in
              Cairo.move_to cr px_low (py -. half_cap);
              Cairo.line_to cr px_low (py +. half_cap);
              Cairo.stroke cr;
              Cairo.move_to cr px_high (py -. half_cap);
              Cairo.line_to cr px_high (py +. half_cap);
              Cairo.stroke cr)))
    done;

    let marker_color = line.color in
    let marker_type = line.marker in
    let default_marker_radius = 3.0 in
    let default_marker_side = default_marker_radius *. 1.8 in
    let default_marker_linewidth = 1.0 in

    Render_utils.set_source_color cr marker_color;
    for i = 0 to n - 1 do
      let x = Nx.get_item [ i ] line.xdata in
      let y = Nx.get_item [ i ] line.ydata in
      let px, py = Transforms.transform t_ctx ~x ~y in

      if Float.is_finite px && Float.is_finite py then
        match marker_type with
        | Artist.None -> ()
        | Artist.Circle ->
            Cairo.arc cr px py ~r:default_marker_radius ~a1:0.
              ~a2:(2. *. Float.pi);
            Cairo.fill cr
        | Artist.Point | Artist.Pixel ->
            Cairo.rectangle cr (px -. 0.5) (py -. 0.5) ~w:1.0 ~h:1.0;
            Cairo.fill cr
        | Artist.Square ->
            let half_side = default_marker_side /. 2.0 in
            Cairo.rectangle cr (px -. half_side) (py -. half_side)
              ~w:default_marker_side ~h:default_marker_side;
            Cairo.fill cr
        | Artist.Triangle ->
            let half_side = default_marker_side /. 2.0 in
            let height = default_marker_side *. sqrt 3. /. 2.0 in
            Cairo.move_to cr px (py -. (height *. 2.0 /. 3.0));
            Cairo.line_to cr (px +. half_side) (py +. (height /. 3.0));
            Cairo.line_to cr (px -. half_side) (py +. (height /. 3.0));
            Cairo.Path.close cr;
            Cairo.fill cr
        | Artist.Plus ->
            let half_arm = default_marker_side /. 2.0 in
            Cairo.set_line_width cr default_marker_linewidth;
            Cairo.move_to cr (px -. half_arm) py;
            Cairo.line_to cr (px +. half_arm) py;
            Cairo.move_to cr px (py -. half_arm);
            Cairo.line_to cr px (py +. half_arm);
            Cairo.stroke cr
        | Artist.Star ->
            let half_arm = default_marker_side /. 2.0 in
            Cairo.set_line_width cr default_marker_linewidth;
            Cairo.move_to cr (px -. half_arm) py;
            Cairo.line_to cr (px +. half_arm) py;
            Cairo.move_to cr px (py -. half_arm);
            Cairo.line_to cr px (py +. half_arm);
            Cairo.stroke cr
    done;

    Cairo.restore cr

let render_step cr (t_ctx : Transforms.context2d) (st : Artist.step) =
  if Nx.size st.xdata < 2 then () else Cairo.save cr;
  Render_utils.set_source_color cr st.color;
  Cairo.set_line_width cr st.linewidth;
  (match st.linestyle with
  | Dashed -> Cairo.set_dash cr ~ofs:0.0 [| 6.0; 6.0 |]
  | Dotted -> Cairo.set_dash cr ~ofs:0.0 [| 1.0; 3.0 |]
  | DashDot -> Cairo.set_dash cr ~ofs:0.0 [| 6.0; 3.0; 1.0; 3.0 |]
  | Solid | None -> Cairo.set_dash cr ~ofs:0.0 [||]);

  let x0 = Nx.get_item [ 0 ] st.xdata in
  let y0 = Nx.get_item [ 0 ] st.ydata in
  let px0, py0 = Transforms.transform t_ctx ~x:x0 ~y:y0 in

  if Float.is_finite px0 && Float.is_finite py0 then Cairo.move_to cr px0 py0
  else ();
  let last_valid = ref (Float.is_finite px0 && Float.is_finite py0) in

  for i = 0 to Nx.size st.xdata - 2 do
    let x1 = Nx.get_item [ i + 1 ] st.xdata in
    let y1 = Nx.get_item [ i + 1 ] st.ydata in
    let px1, py1 = Transforms.transform t_ctx ~x:x1 ~y:y1 in
    let current_valid = Float.is_finite px1 && Float.is_finite py1 in

    if current_valid then
      if !last_valid then (
        let prev_x = Nx.get_item [ i ] st.xdata in
        let prev_y = Nx.get_item [ i ] st.ydata in
        let prev_px, prev_py = Transforms.transform t_ctx ~x:prev_x ~y:prev_y in

        match st.where with
        | Post ->
            Cairo.line_to cr px1 prev_py;
            Cairo.line_to cr px1 py1
        | Pre ->
            Cairo.line_to cr prev_px py1;
            Cairo.line_to cr px1 py1
        | Mid ->
            let mid_px = (prev_px +. px1) /. 2.0 in
            Cairo.line_to cr mid_px prev_py;
            Cairo.line_to cr mid_px py1;
            Cairo.line_to cr px1 py1)
      else Cairo.move_to cr px1 py1;
    last_valid := current_valid
  done;
  Cairo.stroke cr;
  Cairo.restore cr

let render_fill_between cr (t_ctx : Transforms.context2d)
    (fb : Artist.fill_between) =
  Cairo.save cr;
  Render_utils.set_source_color cr fb.color;

  let n = Nx.size fb.xdata in
  if n < 2 then (
    Cairo.restore cr;
    ())
  else
    let is_valid i =
      try
        let x = Nx.get_item [ i ] fb.xdata in
        let y1 = Nx.get_item [ i ] fb.y1data in
        let y2 = Nx.get_item [ i ] fb.y2data in
        let px, py1 = Transforms.transform t_ctx ~x ~y:y1 in
        let _, py2 = Transforms.transform t_ctx ~x ~y:y2 in

        let finite =
          Float.is_finite px && Float.is_finite py1 && Float.is_finite py2
        in
        let where_cond =
          match fb.where with
          | None -> true
          | Some w -> Nx.get_item [ i ] w > 0.
        in
        finite && where_cond
      with _ -> false
    in

    let draw_polygon path_y1 path_y2_rev =
      if List.length path_y1 >= 2 then
        match path_y1 with
        | (px_start, py_start) :: rest_y1 ->
            Cairo.move_to cr px_start py_start;
            List.iter (fun (px, py) -> Cairo.line_to cr px py) rest_y1;
            List.iter (fun (px, py) -> Cairo.line_to cr px py) path_y2_rev;
            Cairo.Path.close cr;
            Cairo.fill cr
        | _ -> ()
    in

    let i = ref 0 in
    while !i < n do
      while !i < n && not (is_valid !i) do
        incr i
      done;

      if !i < n then (
        let segment_y1_points = ref [] in
        let segment_y2_points_rev = ref [] in

        while !i < n && is_valid !i do
          let x = Nx.get_item [ !i ] fb.xdata in
          let y1 = Nx.get_item [ !i ] fb.y1data in
          let y2 = Nx.get_item [ !i ] fb.y2data in
          let px, py1 = Transforms.transform t_ctx ~x ~y:y1 in
          let _, py2 = Transforms.transform t_ctx ~x ~y:y2 in
          segment_y1_points := (px, py1) :: !segment_y1_points;
          segment_y2_points_rev := (px, py2) :: !segment_y2_points_rev;
          incr i
        done;

        draw_polygon (List.rev !segment_y1_points) !segment_y2_points_rev)
    done;

    Cairo.restore cr

let render_artist cr t_ctx artist =
  match (artist, t_ctx) with
  | Artist.Line3D l, Transforms.Ctx3D ctx3d -> render_line3d cr ctx3d l
  | Artist.Line2D l, Transforms.Ctx2D ctx2d -> render_line2d cr ctx2d l
  | Artist.Image img, Transforms.Ctx2D ctx2d -> render_image cr ctx2d img
  | Artist.Bar b, Transforms.Ctx2D ctx2d -> render_bar cr ctx2d b
  | Artist.Text t, Transforms.Ctx2D ctx2d -> render_text cr ctx2d t
  | Artist.Scatter s, Transforms.Ctx2D ctx2d -> render_scatter cr ctx2d s
  | Artist.Errorbar (l, style), Transforms.Ctx2D ctx2d ->
      render_errorbar cr ctx2d l style
  | Artist.Step st, Transforms.Ctx2D ctx2d -> render_step cr ctx2d st
  | Artist.FillBetween fb, Transforms.Ctx2D ctx2d ->
      render_fill_between cr ctx2d fb
  | _, _ ->
      Printf.eprintf
        "Warning: Mismatched artist type and transform context type. Skipping \
         render.\n\
         %!"
