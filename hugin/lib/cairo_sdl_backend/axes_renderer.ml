type data_limits_full = {
  xlim : float * float;
  ylim : float * float;
  zlim : float * float;
}

let render_axes_2d cr fig_width fig_height (ax : Axes.t) =
  let axes_abs_bounds =
    Transforms.
      {
        left = ax.left *. fig_width;
        top = fig_height *. (1. -. ax.bottom -. ax.height);
        width = ax.width *. fig_width;
        height = ax.height *. fig_height;
      }
  in

  let xlim, ylim = Axes.get_final_data_bounds ax in
  let data_lims_2d : Transforms.data_limits =
    { xmin = fst xlim; xmax = snd xlim; ymin = fst ylim; ymax = snd ylim }
  in

  let is_image_view, image_artist_opt =
    match ax.artists with
    | [ Artist.Image img ]
      when ax.title = "" && ax.xlabel = "" && ax.ylabel = "" ->
        (true, Some img)
    | _ -> (false, None)
  in

  if axes_abs_bounds.width <= 0. || axes_abs_bounds.height <= 0. then
    Printf.eprintf "Warning: Axes '%s' has zero size. Skipping.\n%!" ax.title
  else if
    not
      (List.for_all Float.is_finite
         [
           data_lims_2d.xmin;
           data_lims_2d.xmax;
           data_lims_2d.ymin;
           data_lims_2d.ymax;
         ])
  then
    Printf.eprintf "Warning: Invalid data limits for axes '%s'. Skipping.\n%!"
      ax.title
  else if data_lims_2d.xmax <= data_lims_2d.xmin then
    Printf.eprintf
      "Warning: Invalid X data range (%.2f, %.2f) for axes '%s'. Skipping.\n%!"
      data_lims_2d.xmin data_lims_2d.xmax ax.title
  else (
    Cairo.save cr;
    Cairo.set_antialias cr ANTIALIAS_DEFAULT;

    if is_image_view then (
      match image_artist_opt with
      | None ->
          Printf.eprintf
            "Internal Error: is_image_view true but no image found.\n%!"
      | Some img ->
          let shape = img.shape in
          let img_h, img_w =
            match shape with
            | [| h; w; _ |] | [| h; w |] -> (float h, float w)
            | _ -> raise (Invalid_argument "Bad image shape for aspect ratio")
          in
          if img_w <= 0. || img_h <= 0. then
            raise (Failure "Image has zero dimension");

          let img_aspect = img_w /. img_h in
          let available_w = axes_abs_bounds.width in
          let available_h = axes_abs_bounds.height in
          let available_aspect =
            if available_h > 0. then available_w /. available_h else 1.0
          in

          let final_w, final_h =
            if img_aspect > available_aspect then
              (available_w, available_w /. img_aspect)
            else (available_h *. img_aspect, available_h)
          in

          let final_l =
            axes_abs_bounds.left +. ((available_w -. final_w) /. 2.0)
          in
          let final_t =
            axes_abs_bounds.top +. ((available_h -. final_h) /. 2.0)
          in

          let final_bounds =
            Transforms.
              {
                left = final_l;
                top = final_t;
                width = final_w;
                height = final_h;
              }
          in

          if final_bounds.width < 1. || final_bounds.height < 1. then
            Printf.eprintf
              "Warning: Image area for axes '%s' too small after aspect ratio \
               calculation. Skipping.\n\
               %!"
              ax.title
          else
            let transform_ctx =
              Transforms.create_transform_context data_lims_2d final_bounds
                ax.xscale ax.yscale
            in

            Cairo.save cr;
            Artist_renderer.render_artist cr (Ctx2D transform_ctx)
              (Artist.Image img);
            Cairo.restore cr)
    else
      let xticks =
        match ax.xticks with
        | Some t -> t
        | None ->
            Ticks.generate_linear_ticks ~min_val:data_lims_2d.xmin
              ~max_val:data_lims_2d.xmax ~max_ticks:8
      in
      let y_tick_min = Float.min data_lims_2d.ymin data_lims_2d.ymax in
      let y_tick_max = Float.max data_lims_2d.ymin data_lims_2d.ymax in
      let yticks =
        if y_tick_min >= y_tick_max then []
        else
          match ax.yticks with
          | Some t -> t
          | None ->
              Ticks.generate_linear_ticks ~min_val:y_tick_min
                ~max_val:y_tick_max ~max_ticks:8
      in

      let plot_bounds, padding, transform_ctx =
        let bounds, pad =
          Layout.calculate_axes_layout cr ax xticks yticks axes_abs_bounds
        in
        let ctx =
          Transforms.create_transform_context data_lims_2d bounds ax.xscale
            ax.yscale
        in
        (bounds, pad, ctx)
      in

      if plot_bounds.width < 1. || plot_bounds.height < 1. then
        Printf.eprintf
          "Warning: Plot area for axes '%s' too small after layout \
           calculation. Skipping decorations/artists.\n\
           %!"
          ax.title
      else (
        Cairo.save cr;
        Cairo.rectangle cr plot_bounds.left plot_bounds.top ~w:plot_bounds.width
          ~h:plot_bounds.height;
        Cairo.clip cr;
        List.iter
          (fun artist ->
            Artist_renderer.render_artist cr (Ctx2D transform_ctx) artist)
          ax.artists;
        Cairo.restore cr;

        Cairo.set_source_rgb cr 0.0 0.0 0.0;
        Cairo.set_line_width cr 1.0;
        Cairo.rectangle cr plot_bounds.left plot_bounds.top ~w:plot_bounds.width
          ~h:plot_bounds.height;
        Cairo.stroke cr;

        if ax.grid_visible then (
          Cairo.save cr;
          Render_utils.set_source_color cr Artist.Color.lightgray;
          Cairo.set_line_width cr 0.5;
          List.iter
            (fun tick_x ->
              let px1, _py1 =
                Transforms.transform transform_ctx ~x:tick_x
                  ~y:data_lims_2d.ymin
              in
              if
                px1 >= plot_bounds.left -. 1e-6
                && px1 <= plot_bounds.left +. plot_bounds.width +. 1e-6
              then (
                Cairo.move_to cr px1 plot_bounds.top;
                Cairo.line_to cr px1 (plot_bounds.top +. plot_bounds.height);
                Cairo.stroke cr))
            xticks;
          List.iter
            (fun tick_y ->
              let _px1, py1 =
                Transforms.transform transform_ctx ~x:data_lims_2d.xmin
                  ~y:tick_y
              in
              if
                py1 >= plot_bounds.top -. 1e-6
                && py1 <= plot_bounds.top +. plot_bounds.height +. 1e-6
              then (
                Cairo.move_to cr plot_bounds.left py1;
                Cairo.line_to cr (plot_bounds.left +. plot_bounds.width) py1;
                Cairo.stroke cr))
            yticks;
          Cairo.restore cr);

        Cairo.save cr;
        Cairo.select_font_face cr Layout.Defaults.font_face;
        Cairo.set_font_size cr Layout.Defaults.tick_font_size;
        Render_utils.set_source_color cr Artist.Color.black;
        let tick_len = Layout.Defaults.tick_length in
        let tick_label_gap = Layout.Defaults.tick_label_gap in

        List.iter
          (fun tick_x ->
            let px, _ =
              Transforms.transform transform_ctx ~x:tick_x ~y:data_lims_2d.ymin
            in
            if
              px >= plot_bounds.left -. 1e-6
              && px <= plot_bounds.left +. plot_bounds.width +. 1e-6
            then (
              Cairo.move_to cr px (plot_bounds.top +. plot_bounds.height);
              Cairo.rel_line_to cr 0.0 tick_len;
              Cairo.stroke cr;
              let label = Ticks.format_tick_value tick_x in
              let te = Cairo.text_extents cr label in
              Cairo.move_to cr
                (px -. (te.Cairo.width /. 2.0))
                (plot_bounds.top +. plot_bounds.height +. tick_len
               +. tick_label_gap +. te.Cairo.height);
              Cairo.show_text cr label))
          xticks;

        List.iter
          (fun tick_y ->
            let _, py =
              Transforms.transform transform_ctx ~x:data_lims_2d.xmin ~y:tick_y
            in
            if
              py >= plot_bounds.top -. 1e-6
              && py <= plot_bounds.top +. plot_bounds.height +. 1e-6
            then (
              Cairo.move_to cr plot_bounds.left py;
              Cairo.rel_line_to cr (-.tick_len) 0.0;
              Cairo.stroke cr;
              let label = Ticks.format_tick_value tick_y in
              let te = Cairo.text_extents cr label in
              Cairo.move_to cr
                (plot_bounds.left -. tick_len -. tick_label_gap
               -. te.Cairo.width)
                (py +. (te.Cairo.height /. 2.0));
              Cairo.show_text cr label))
          yticks;
        Cairo.restore cr;

        Cairo.save cr;
        Render_utils.set_source_color cr Artist.Color.black;
        let base_font cr =
          Cairo.select_font_face cr Layout.Defaults.font_face
        in

        if ax.xlabel <> "" then (
          base_font cr;
          Cairo.set_font_size cr Layout.Defaults.label_font_size;
          let te = Cairo.text_extents cr ax.xlabel in
          let x =
            plot_bounds.left +. ((plot_bounds.width -. te.Cairo.width) /. 2.0)
          in
          let y =
            axes_abs_bounds.top +. axes_abs_bounds.height -. padding.bottom
            +. Layout.Defaults.tick_length +. Layout.Defaults.tick_label_gap
            +. te.height +. Layout.Defaults.axis_label_gap
          in
          Cairo.move_to cr x y;
          Cairo.show_text cr ax.xlabel);

        if ax.ylabel <> "" then (
          base_font cr;
          Cairo.set_font_size cr Layout.Defaults.label_font_size;
          let te = Cairo.text_extents cr ax.ylabel in
          let x =
            axes_abs_bounds.left +. padding.left -. Layout.Defaults.tick_length
            -. Layout.Defaults.tick_label_gap -. te.Cairo.height
            -. Layout.Defaults.axis_label_gap
          in
          let y =
            plot_bounds.top +. ((plot_bounds.height +. te.Cairo.width) /. 2.0)
          in
          Cairo.save cr;
          Cairo.move_to cr x y;
          Cairo.rotate cr (-.Float.pi /. 2.0);
          Cairo.show_text cr ax.ylabel;
          Cairo.restore cr);

        if ax.title <> "" then (
          Cairo.select_font_face cr Layout.Defaults.font_face ~weight:Bold;
          Cairo.set_font_size cr Layout.Defaults.title_font_size;
          let te = Cairo.text_extents cr ax.title in
          let x =
            plot_bounds.left +. ((plot_bounds.width -. te.Cairo.width) /. 2.0)
          in
          let y =
            axes_abs_bounds.top +. padding.top -. Layout.Defaults.title_gap
          in
          Cairo.move_to cr x y;
          Cairo.show_text cr ax.title);
        Cairo.restore cr);

      Cairo.restore cr)

let render_axes_3d cr fig_width fig_height (ax : Axes.t) =
  let axes_abs_bounds =
    Transforms.
      {
        left = ax.left *. fig_width;
        top = fig_height *. (1. -. ax.bottom -. ax.height);
        width = ax.width *. fig_width;
        height = ax.height *. fig_height;
      }
  in

  if axes_abs_bounds.width <= 0. || axes_abs_bounds.height <= 0. then
    Printf.eprintf "Warning: 3D Axes '%s' has zero size. Skipping.\n%!" ax.title
  else
    let (xmin, xmax), (ymin, ymax) = Axes.get_final_data_bounds ax in
    let zmin, zmax = Axes.get_final_z_bounds ax in

    if not (List.for_all Float.is_finite [ xmin; xmax; ymin; ymax; zmin; zmax ])
    then
      Printf.eprintf
        "Warning: Invalid data limits for 3D axes '%s'. Skipping.\n%!" ax.title
    else if xmax <= xmin || ymax <= ymin || zmax <= zmin then
      Printf.eprintf
        "Warning: Invalid data range for 3D axes '%s'. Skipping.\n%!" ax.title
    else
      let plot_bounds = axes_abs_bounds in

      if plot_bounds.width < 1. || plot_bounds.height < 1. then
        Printf.eprintf
          "Warning: Plot area for 3D axes '%s' too small. Skipping.\n%!"
          ax.title
      else
        let transform_ctx_3d =
          Transforms.create_transform_context3d ~xmin ~xmax ~ymin ~ymax ~zmin
            ~zmax ~elev:ax.elev ~azim:ax.azim plot_bounds
        in
        let transform_ctx = Transforms.Ctx3D transform_ctx_3d in
        let trans3d = Transforms.transform3d transform_ctx_3d in

        Cairo.save cr;
        Cairo.set_antialias cr ANTIALIAS_DEFAULT;

        let xticks =
          match ax.xticks with
          | Some t -> t
          | None ->
              Ticks.generate_linear_ticks ~min_val:xmin ~max_val:xmax
                ~max_ticks:5
        in
        let yticks =
          match ax.yticks with
          | Some t -> t
          | None ->
              Ticks.generate_linear_ticks ~min_val:ymin ~max_val:ymax
                ~max_ticks:5
        in
        let zticks =
          match ax.zticks with
          | Some t -> t
          | None ->
              Ticks.generate_linear_ticks ~min_val:zmin ~max_val:zmax
                ~max_ticks:5
        in

        if ax.grid_visible then (
          Cairo.save cr;
          Render_utils.set_source_color cr Artist.Color.lightgray;
          Cairo.set_line_width cr 0.5;
          List.iter
            (fun x ->
              match
                (trans3d ~x ~y:ymin ~z:zmin, trans3d ~x ~y:ymax ~z:zmin)
              with
              | Some p1, Some p2 ->
                  Cairo.move_to cr (fst p1) (snd p1);
                  Cairo.line_to cr (fst p2) (snd p2);
                  Cairo.stroke cr
              | _, _ -> ())
            xticks;
          List.iter
            (fun y ->
              match
                (trans3d ~x:xmin ~y ~z:zmin, trans3d ~x:xmax ~y ~z:zmin)
              with
              | Some p1, Some p2 ->
                  Cairo.move_to cr (fst p1) (snd p1);
                  Cairo.line_to cr (fst p2) (snd p2);
                  Cairo.stroke cr
              | _, _ -> ())
            yticks;
          List.iter
            (fun x ->
              match
                (trans3d ~x ~y:ymin ~z:zmin, trans3d ~x ~y:ymin ~z:zmax)
              with
              | Some p1, Some p2 ->
                  Cairo.move_to cr (fst p1) (snd p1);
                  Cairo.line_to cr (fst p2) (snd p2);
                  Cairo.stroke cr
              | _, _ -> ())
            xticks;
          List.iter
            (fun z ->
              match
                (trans3d ~x:xmin ~y:ymin ~z, trans3d ~x:xmax ~y:ymin ~z)
              with
              | Some p1, Some p2 ->
                  Cairo.move_to cr (fst p1) (snd p1);
                  Cairo.line_to cr (fst p2) (snd p2);
                  Cairo.stroke cr
              | _, _ -> ())
            zticks;
          List.iter
            (fun y ->
              match
                (trans3d ~x:xmin ~y ~z:zmin, trans3d ~x:xmin ~y ~z:zmax)
              with
              | Some p1, Some p2 ->
                  Cairo.move_to cr (fst p1) (snd p1);
                  Cairo.line_to cr (fst p2) (snd p2);
                  Cairo.stroke cr
              | _, _ -> ())
            yticks;
          List.iter
            (fun z ->
              match
                (trans3d ~x:xmin ~y:ymin ~z, trans3d ~x:xmin ~y:ymax ~z)
              with
              | Some p1, Some p2 ->
                  Cairo.move_to cr (fst p1) (snd p1);
                  Cairo.line_to cr (fst p2) (snd p2);
                  Cairo.stroke cr
              | _, _ -> ())
            zticks;
          Cairo.restore cr);
        Cairo.save cr;
        Render_utils.set_source_color cr Artist.Color.gray;
        Cairo.set_line_width cr 0.8;
        let corners =
          [|
            (xmin, ymin, zmin);
            (xmax, ymin, zmin);
            (xmax, ymax, zmin);
            (xmin, ymax, zmin);
            (xmin, ymin, zmax);
            (xmax, ymin, zmax);
            (xmax, ymax, zmax);
            (xmin, ymax, zmax);
          |]
        in
        let edges =
          [|
            (0, 1);
            (1, 2);
            (2, 3);
            (3, 0);
            (4, 5);
            (5, 6);
            (6, 7);
            (7, 4);
            (0, 4);
            (1, 5);
            (2, 6);
            (3, 7);
          |]
        in
        Array.iter
          (fun (i, j) ->
            let x1, y1, z1 = corners.(i) in
            let x2, y2, z2 = corners.(j) in
            match (trans3d ~x:x1 ~y:y1 ~z:z1, trans3d ~x:x2 ~y:y2 ~z:z2) with
            | Some (px1, py1), Some (px2, py2) ->
                Cairo.move_to cr px1 py1;
                Cairo.line_to cr px2 py2;
                Cairo.stroke cr
            | _, _ -> ())
          edges;
        Cairo.restore cr;

        (* Draw Ticks and Labels - Simplified positioning *)
        Cairo.save cr;
        Cairo.select_font_face cr Layout.Defaults.font_face;
        Cairo.set_font_size cr Layout.Defaults.tick_font_size;
        Render_utils.set_source_color cr Artist.Color.black;
        let tick_len_3d = 3.0 in
        (* Shorter ticks for 3D *)
        let label_offset = 15.0 in

        (* X Ticks along edge Y=ymin, Z=zmin *)
        List.iter
          (fun x ->
            match trans3d ~x ~y:ymin ~z:zmin with
            | Some (px, py) ->
                (* Simple vertical tick *)
                Cairo.move_to cr px py;
                Cairo.rel_line_to cr 0. tick_len_3d;
                Cairo.stroke cr;
                let label = Ticks.format_tick_value x in
                let te = Cairo.text_extents cr label in
                Cairo.move_to cr
                  (px -. (te.Cairo.width /. 2.))
                  (py +. label_offset);
                Cairo.show_text cr label
            | None -> ())
          xticks;
        (* Y Ticks along edge X=xmax, Z=zmin *)
        List.iter
          (fun y ->
            match trans3d ~x:xmax ~y ~z:zmin with
            | Some (px, py) ->
                (* Simple horizontal tick *)
                Cairo.move_to cr px py;
                Cairo.rel_line_to cr tick_len_3d 0.;
                Cairo.stroke cr;
                let label = Ticks.format_tick_value y in
                let te = Cairo.text_extents cr label in
                Cairo.move_to cr
                  (px +. label_offset -. (te.Cairo.width /. 2.))
                  (py +. (te.Cairo.height /. 2.));
                Cairo.show_text cr label
            | None -> ())
          yticks;
        (* Z Ticks along edge X=xmax, Y=ymax *)
        List.iter
          (fun z ->
            match trans3d ~x:xmax ~y:ymax ~z with
            | Some (px, py) ->
                (* Simple angled tick - approximate *)
                Cairo.move_to cr px py;
                Cairo.rel_line_to cr (-.tick_len_3d *. 0.707)
                  (-.tick_len_3d *. 0.707);
                Cairo.stroke cr;
                let label = Ticks.format_tick_value z in
                let te = Cairo.text_extents cr label in
                Cairo.move_to cr (px -. label_offset)
                  (py -. (label_offset *. 0.5) +. (te.Cairo.height /. 2.));
                Cairo.show_text cr label
            | None -> ())
          zticks;
        Cairo.restore cr;

        (* Render artists within the plot area *)
        Cairo.save cr;
        Cairo.rectangle cr plot_bounds.left plot_bounds.top ~w:plot_bounds.width
          ~h:plot_bounds.height;
        Cairo.clip cr;
        List.iter
          (fun artist -> Artist_renderer.render_artist cr transform_ctx artist)
          ax.artists;
        Cairo.restore cr;

        (* Render Axis Labels - Simplified positioning *)
        Cairo.save cr;
        Cairo.select_font_face cr Layout.Defaults.font_face;
        Cairo.set_font_size cr Layout.Defaults.label_font_size;
        Render_utils.set_source_color cr Artist.Color.black;
        let label_pos_offset = 35.0 in

        (if ax.xlabel <> "" then
           match trans3d ~x:((xmin +. xmax) /. 2.) ~y:ymin ~z:zmin with
           | Some (px, py) ->
               let te = Cairo.text_extents cr ax.xlabel in
               Cairo.move_to cr
                 (px -. (te.Cairo.width /. 2.))
                 (py +. label_pos_offset);
               Cairo.show_text cr ax.xlabel
           | None -> ());
        (if ax.ylabel <> "" then
           match trans3d ~x:xmax ~y:((ymin +. ymax) /. 2.) ~z:zmin with
           | Some (px, py) ->
               let te = Cairo.text_extents cr ax.ylabel in
               Cairo.move_to cr
                 (px +. (label_pos_offset *. 0.7))
                 (py +. (te.Cairo.height /. 2.));
               Cairo.show_text cr ax.ylabel
           | None -> ());
        (match ax.zlabel with
        | Some zlabel when zlabel <> "" -> (
            match trans3d ~x:xmax ~y:ymax ~z:((zmin +. zmax) /. 2.) with
            | Some (px, py) ->
                let te = Cairo.text_extents cr zlabel in
                Cairo.move_to cr (px -. label_pos_offset)
                  (py -. (label_pos_offset *. 0.7) +. (te.Cairo.height /. 2.));
                Cairo.show_text cr zlabel
            | None -> ())
        | _ -> ());

        Cairo.restore cr;

        (* Render Title - Centered at top *)
        if ax.title <> "" then (
          Cairo.save cr;
          Cairo.select_font_face cr Layout.Defaults.font_face ~weight:Bold;
          Cairo.set_font_size cr Layout.Defaults.title_font_size;
          Render_utils.set_source_color cr Artist.Color.black;
          let te = Cairo.text_extents cr ax.title in
          let title_x =
            plot_bounds.left +. ((plot_bounds.width -. te.Cairo.width) /. 2.0)
          in
          let title_y =
            plot_bounds.top +. Layout.Defaults.title_font_size +. 5.0
          in
          (* Approx position *)
          Cairo.move_to cr title_x title_y;
          Cairo.show_text cr ax.title;
          Cairo.restore cr);

        Cairo.restore cr

let render_axes cr fig_width fig_height ax =
  match ax.Axes.projection with
  | Axes.TwoD -> render_axes_2d cr fig_width fig_height ax
  | Axes.ThreeD -> render_axes_3d cr fig_width fig_height ax
