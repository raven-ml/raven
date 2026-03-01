(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Prepared.t + Theme.t -> Scene.t resolution

   Layout and pixel-coordinate work. All data-only processing (histogram
   binning, auto-coloring, bounds) is done in Prepared. *)

type text_measurer = font:Theme.font -> string -> float * float

(* Region in device pixels *)
type region = { rx : float; ry : float; rw : float; rh : float }

(* Scale-aware coord transform *)

let data_to_pixel_x sx region v =
  let u = sx.Scale.to_unit v in
  region.rx +. (u *. region.rw)

let data_to_pixel_y sy region v =
  let u = sy.Scale.to_unit v in
  region.ry +. region.rh -. (u *. region.rh)

(* Dash pattern *)

let dash_of_style = function
  | `Solid -> []
  | `Dashed -> [ 6.; 4. ]
  | `Dotted -> [ 2.; 3. ]
  | `Dash_dot -> [ 6.; 3.; 2.; 3. ]

(* Resolution helpers *)

let resolve_color ?default_alpha color alpha =
  let c = Option.value ~default:Color.black color in
  match (alpha, default_alpha) with
  | Some a, _ | None, Some a -> Color.with_alpha a c
  | None, None -> c

let resolve_line_width sf (theme : Theme.t) line_width =
  Option.value ~default:theme.line_width line_width *. sf

let resolve_dash sf line_style =
  let dash = match line_style with Some s -> dash_of_style s | None -> [] in
  List.map (fun d -> d *. sf) dash

(* Emit mark primitives *)

let step_transform step points n =
  match step with
  | None -> points
  | Some `Post ->
      if n < 2 then points
      else begin
        let out = Array.make ((2 * n) - 1) (0., 0.) in
        let k = ref 0 in
        for i = 0 to n - 2 do
          let px, py = points.(i) in
          let px_next, _ = points.(i + 1) in
          out.(!k) <- (px, py);
          incr k;
          out.(!k) <- (px_next, py);
          incr k
        done;
        out.(!k) <- points.(n - 1);
        Array.sub out 0 (!k + 1)
      end
  | Some `Pre ->
      if n < 2 then points
      else begin
        let out = Array.make ((2 * n) - 1) (0., 0.) in
        let k = ref 0 in
        out.(!k) <- points.(0);
        incr k;
        for i = 1 to n - 1 do
          let _, py_prev = points.(i - 1) in
          let px, py = points.(i) in
          out.(!k) <- (px, py_prev);
          incr k;
          out.(!k) <- (px, py);
          incr k
        done;
        Array.sub out 0 !k
      end
  | Some `Mid ->
      if n < 2 then points
      else begin
        let out = Array.make ((3 * n) - 2) (0., 0.) in
        let k = ref 0 in
        for i = 0 to n - 2 do
          let px, py = points.(i) in
          let px_next, py_next = points.(i + 1) in
          let mx = (px +. px_next) /. 2. in
          out.(!k) <- (px, py);
          incr k;
          out.(!k) <- (mx, py);
          incr k;
          out.(!k) <- (mx, py_next);
          incr k
        done;
        out.(!k) <- points.(n - 1);
        Array.sub out 0 (!k + 1)
      end

let emit_line_mark sx sy plot_area theme ~x ~y ~color ~line_width ~line_style
    ~step ~marker ~alpha =
  let n = (Nx.shape x).(0) in
  let color = resolve_color color alpha in
  let sf = theme.Theme.scale_factor in
  let lw = resolve_line_width sf theme line_width in
  let scaled_dash = resolve_dash sf line_style in
  (* Split line into finite-value segments *)
  let segments = ref [] in
  let current = ref [] in
  let all_finite_points = ref [] in
  for i = 0 to n - 1 do
    let xv = Nx.item [ i ] x in
    let yv = Nx.item [ i ] y in
    if Float.is_finite xv && Float.is_finite yv then begin
      let px = data_to_pixel_x sx plot_area xv in
      let py = data_to_pixel_y sy plot_area yv in
      let pt = (px, py) in
      current := pt :: !current;
      all_finite_points := pt :: !all_finite_points
    end
    else
      match !current with
      | [] -> ()
      | _ ->
          segments := Array.of_list (List.rev !current) :: !segments;
          current := []
  done;
  (match !current with
  | [] -> ()
  | _ -> segments := Array.of_list (List.rev !current) :: !segments);
  let segments = List.rev !segments in
  let paths =
    List.map
      (fun points ->
        let n_pts = Array.length points in
        let points = step_transform step points n_pts in
        Scene.Path
          {
            points;
            close = false;
            fill = None;
            stroke = Some color;
            line_width = lw;
            dash = scaled_dash;
          })
      segments
  in
  match marker with
  | Some shape ->
      let finite_points = Array.of_list (List.rev !all_finite_points) in
      let ms = theme.marker_size *. sf in
      let markers =
        Scene.Markers
          {
            points = finite_points;
            shape;
            size = ms;
            sizes = None;
            fill = Some color;
            fills = None;
            stroke = None;
          }
      in
      paths @ [ markers ]
  | None -> paths

let emit_point_mark sx sy plot_area theme ~x ~y ~color ~color_by ~size ~size_by
    ~marker ~alpha =
  let n = (Nx.shape x).(0) in
  let color = resolve_color color alpha in
  let shape = Option.value ~default:Spec.Circle marker in
  let ms =
    (match size with Some s -> s | None -> theme.Theme.marker_size)
    *. theme.scale_factor
  in
  (* Collect only finite points *)
  let valid = Array.make n false in
  let num_valid = ref 0 in
  for i = 0 to n - 1 do
    let xv = Nx.item [ i ] x in
    let yv = Nx.item [ i ] y in
    if Float.is_finite xv && Float.is_finite yv then begin
      valid.(i) <- true;
      incr num_valid
    end
  done;
  let nv = !num_valid in
  let points = Array.make nv (0., 0.) in
  let vi = ref 0 in
  for i = 0 to n - 1 do
    if valid.(i) then begin
      let px = data_to_pixel_x sx plot_area (Nx.item [ i ] x) in
      let py = data_to_pixel_y sy plot_area (Nx.item [ i ] y) in
      points.(!vi) <- (px, py);
      incr vi
    end
  done;
  let sizes =
    match size_by with
    | Some sb ->
        let sb_lo, sb_hi = Prepared.nx_finite_range sb in
        let sb_range = if sb_hi = sb_lo then 1. else sb_hi -. sb_lo in
        let arr = Array.make nv ms in
        let vi = ref 0 in
        for i = 0 to n - 1 do
          if valid.(i) then begin
            let sv = (Nx.item [ i ] sb -. sb_lo) /. sb_range in
            arr.(!vi) <- (ms *. 0.5) +. (ms *. Float.sqrt sv);
            incr vi
          end
        done;
        Some arr
    | None -> None
  in
  let fills =
    match color_by with
    | Some cb ->
        let cb_lo, cb_hi = Prepared.nx_finite_range cb in
        let cb_range = if cb_hi = cb_lo then 1. else cb_hi -. cb_lo in
        let arr = Array.make nv Color.black in
        let vi = ref 0 in
        for i = 0 to n - 1 do
          if valid.(i) then begin
            let cv = (Nx.item [ i ] cb -. cb_lo) /. cb_range in
            let c = Cmap.eval theme.sequential cv in
            arr.(!vi) <-
              (match alpha with Some a -> Color.with_alpha a c | None -> c);
            incr vi
          end
        done;
        Some arr
    | None -> None
  in
  let fill = if fills <> None then None else Some color in
  let stroke = Some color in
  [ Scene.Markers { points; shape; size = ms; sizes; fill; fills; stroke } ]

let emit_bar_mark sx sy plot_area theme ~x ~height ~width ~bottom ~color ~alpha
    =
  let n = (Nx.shape x).(0) in
  let color = resolve_color color alpha in
  let w = Option.value ~default:0.8 width in
  let prims = ref [] in
  for i = 0 to n - 1 do
    let xi = Nx.item [ i ] x in
    let hi = Nx.item [ i ] height in
    if Float.is_finite xi && Float.is_finite hi then
      let x0 = data_to_pixel_x sx plot_area (xi -. (w /. 2.)) in
      let x1 = data_to_pixel_x sx plot_area (xi +. (w /. 2.)) in
      let y0 = data_to_pixel_y sy plot_area bottom in
      let y1 = data_to_pixel_y sy plot_area (bottom +. hi) in
      let lx = Float.min x0 x1 and rx = Float.max x0 x1 in
      let ty = Float.min y0 y1 and by = Float.max y0 y1 in
      prims :=
        Scene.Path
          {
            points = [| (lx, ty); (rx, ty); (rx, by); (lx, by) |];
            close = true;
            fill = Some color;
            stroke = None;
            line_width = 0.;
            dash = [];
          }
        :: !prims
  done;
  List.rev !prims

let emit_image_mark sx sy plot_area ~data ~extent =
  match extent with
  | Some (xmin, xmax, ymin, ymax) ->
      let px0 = data_to_pixel_x sx plot_area xmin in
      let px1 = data_to_pixel_x sx plot_area xmax in
      let py0 = data_to_pixel_y sy plot_area ymax in
      let py1 = data_to_pixel_y sy plot_area ymin in
      let x = Float.min px0 px1 in
      let y = Float.min py0 py1 in
      let w = Float.abs (px1 -. px0) in
      let h = Float.abs (py1 -. py0) in
      [ Scene.Image { x; y; w; h; data } ]
  | None ->
      let shape = Nx.shape data in
      let img_h = float shape.(0) and img_w = float shape.(1) in
      let aspect = img_w /. img_h in
      let plot_aspect = plot_area.rw /. plot_area.rh in
      let w, h =
        if aspect > plot_aspect then (plot_area.rw, plot_area.rw /. aspect)
        else (plot_area.rh *. aspect, plot_area.rh)
      in
      let x = plot_area.rx +. ((plot_area.rw -. w) /. 2.) in
      let y = plot_area.ry +. ((plot_area.rh -. h) /. 2.) in
      [ Scene.Image { x; y; w; h; data } ]

let emit_text_mark sx sy plot_area theme ~x ~y ~content ~color ~font_size =
  let color = Option.value ~default:Color.black color in
  let size = Option.value ~default:theme.Theme.font_label.size font_size in
  let px = data_to_pixel_x sx plot_area x in
  let py = data_to_pixel_y sy plot_area y in
  [
    Scene.Text
      {
        x = px;
        y = py;
        content;
        font =
          {
            family = theme.font_label.family;
            size = size *. theme.scale_factor;
            weight = `Normal;
          };
        color;
        anchor = `Start;
        baseline = `Bottom;
        angle = 0.;
      };
  ]

let emit_hline_mark sy plot_area theme ~y:yv ~color ~line_width ~line_style
    ~alpha =
  let color = resolve_color color alpha in
  let sf = theme.Theme.scale_factor in
  let lw = resolve_line_width sf theme line_width in
  let dash = resolve_dash sf line_style in
  let py = data_to_pixel_y sy plot_area yv in
  [
    Scene.Path
      {
        points = [| (plot_area.rx, py); (plot_area.rx +. plot_area.rw, py) |];
        close = false;
        fill = None;
        stroke = Some color;
        line_width = lw;
        dash;
      };
  ]

let emit_vline_mark sx plot_area theme ~x:xv ~color ~line_width ~line_style
    ~alpha =
  let color = resolve_color color alpha in
  let sf = theme.Theme.scale_factor in
  let lw = resolve_line_width sf theme line_width in
  let dash = resolve_dash sf line_style in
  let px = data_to_pixel_x sx plot_area xv in
  [
    Scene.Path
      {
        points = [| (px, plot_area.ry); (px, plot_area.ry +. plot_area.rh) |];
        close = false;
        fill = None;
        stroke = Some color;
        line_width = lw;
        dash;
      };
  ]

let emit_abline_mark sx sy plot_area theme ~slope ~intercept ~color ~line_width
    ~line_style ~alpha =
  let color = resolve_color color alpha in
  let sf = theme.Theme.scale_factor in
  let lw = resolve_line_width sf theme line_width in
  let dash = resolve_dash sf line_style in
  let x0 = sx.Scale.lo and x1 = sx.Scale.hi in
  let y0v = (slope *. x0) +. intercept in
  let y1v = (slope *. x1) +. intercept in
  let px0 = data_to_pixel_x sx plot_area x0 in
  let py0 = data_to_pixel_y sy plot_area y0v in
  let px1 = data_to_pixel_x sx plot_area x1 in
  let py1 = data_to_pixel_y sy plot_area y1v in
  [
    Scene.Path
      {
        points = [| (px0, py0); (px1, py1) |];
        close = false;
        fill = None;
        stroke = Some color;
        line_width = lw;
        dash;
      };
  ]

let emit_fill_between_segment sx sy plot_area color indices x y1 y2 =
  let n_seg = List.length indices in
  if n_seg = 0 then []
  else
    let points = Array.make (2 * n_seg) (0., 0.) in
    let k = ref 0 in
    List.iter
      (fun i ->
        let xv = Nx.item [ i ] x in
        let yv = Nx.item [ i ] y1 in
        if Float.is_finite xv && Float.is_finite yv then begin
          points.(!k) <-
            (data_to_pixel_x sx plot_area xv, data_to_pixel_y sy plot_area yv);
          incr k
        end)
      indices;
    let forward_count = !k in
    List.iter
      (fun i ->
        let xv = Nx.item [ i ] x in
        let yv = Nx.item [ i ] y2 in
        if Float.is_finite xv && Float.is_finite yv then begin
          points.(!k) <-
            (data_to_pixel_x sx plot_area xv, data_to_pixel_y sy plot_area yv);
          incr k
        end)
      (List.rev indices);
    let total = !k in
    if total < 3 || forward_count = 0 then []
    else
      [
        Scene.Path
          {
            points = Array.sub points 0 total;
            close = true;
            fill = Some color;
            stroke = None;
            line_width = 0.;
            dash = [];
          };
      ]

let emit_fill_between_mark sx sy plot_area ~x ~y1 ~y2 ~where ~color ~alpha =
  let n = (Nx.shape x).(0) in
  let color = resolve_color ~default_alpha:0.3 color alpha in
  match where with
  | None ->
      let indices = List.init n Fun.id in
      emit_fill_between_segment sx sy plot_area color indices x y1 y2
  | Some mask ->
      (* Split into contiguous runs where mask > 0 *)
      let segments = ref [] in
      let current = ref [] in
      for i = 0 to n - 1 do
        if Nx.item [ i ] mask > 0. then current := i :: !current
        else
          match !current with
          | [] -> ()
          | seg ->
              segments := List.rev seg :: !segments;
              current := []
      done;
      (match !current with
      | [] -> ()
      | seg -> segments := List.rev seg :: !segments);
      List.concat_map
        (fun seg -> emit_fill_between_segment sx sy plot_area color seg x y1 y2)
        (List.rev !segments)

let emit_hspan_mark sy plot_area ~y0 ~y1 ~color ~alpha =
  let color = resolve_color ~default_alpha:0.2 color alpha in
  let py0 = data_to_pixel_y sy plot_area y0 in
  let py1 = data_to_pixel_y sy plot_area y1 in
  let top = Float.min py0 py1 and bot = Float.max py0 py1 in
  [
    Scene.Path
      {
        points =
          [|
            (plot_area.rx, top);
            (plot_area.rx +. plot_area.rw, top);
            (plot_area.rx +. plot_area.rw, bot);
            (plot_area.rx, bot);
          |];
        close = true;
        fill = Some color;
        stroke = None;
        line_width = 0.;
        dash = [];
      };
  ]

let emit_vspan_mark sx plot_area ~x0 ~x1 ~color ~alpha =
  let color = resolve_color ~default_alpha:0.2 color alpha in
  let px0 = data_to_pixel_x sx plot_area x0 in
  let px1 = data_to_pixel_x sx plot_area x1 in
  let left = Float.min px0 px1 and right = Float.max px0 px1 in
  [
    Scene.Path
      {
        points =
          [|
            (left, plot_area.ry);
            (right, plot_area.ry);
            (right, plot_area.ry +. plot_area.rh);
            (left, plot_area.ry +. plot_area.rh);
          |];
        close = true;
        fill = Some color;
        stroke = None;
        line_width = 0.;
        dash = [];
      };
  ]

let emit_errorbar_mark sx sy plot_area theme ~x ~y ~yerr ~xerr ~color
    ~line_width ~cap_size ~alpha =
  let n = (Nx.shape x).(0) in
  let color = resolve_color color alpha in
  let sf = theme.Theme.scale_factor in
  let lw = resolve_line_width sf theme line_width in
  let cap =
    (match cap_size with Some s -> s | None -> theme.marker_size *. 0.5) *. sf
  in
  let prims = ref [] in
  let make_path pts =
    Scene.Path
      {
        points = pts;
        close = false;
        fill = None;
        stroke = Some color;
        line_width = lw;
        dash = [];
      }
  in
  for i = 0 to n - 1 do
    let xv = Nx.item [ i ] x in
    let yv = Nx.item [ i ] y in
    if Float.is_finite xv && Float.is_finite yv then begin
      let px = data_to_pixel_x sx plot_area xv in
      let py = data_to_pixel_y sy plot_area yv in
      let y_lo, y_hi =
        match yerr with
        | `Symmetric e ->
            let ev = Nx.item [ i ] e in
            (yv -. ev, yv +. ev)
        | `Asymmetric (elo, ehi) ->
            (yv -. Nx.item [ i ] elo, yv +. Nx.item [ i ] ehi)
      in
      let py_lo = data_to_pixel_y sy plot_area y_lo in
      let py_hi = data_to_pixel_y sy plot_area y_hi in
      prims := make_path [| (px, py_lo); (px, py_hi) |] :: !prims;
      prims := make_path [| (px -. cap, py_hi); (px +. cap, py_hi) |] :: !prims;
      prims := make_path [| (px -. cap, py_lo); (px +. cap, py_lo) |] :: !prims;
      begin match xerr with
      | Some xerr_val ->
          let x_lo, x_hi =
            match xerr_val with
            | `Symmetric e ->
                let ev = Nx.item [ i ] e in
                (xv -. ev, xv +. ev)
            | `Asymmetric (elo, ehi) ->
                (xv -. Nx.item [ i ] elo, xv +. Nx.item [ i ] ehi)
          in
          let px_lo = data_to_pixel_x sx plot_area x_lo in
          let px_hi = data_to_pixel_x sx plot_area x_hi in
          prims := make_path [| (px_lo, py); (px_hi, py) |] :: !prims;
          prims :=
            make_path [| (px_lo, py -. cap); (px_lo, py +. cap) |] :: !prims;
          prims :=
            make_path [| (px_hi, py -. cap); (px_hi, py +. cap) |] :: !prims
      | None -> ()
      end
    end
  done;
  List.rev !prims

let emit_heatmap_mark sx sy plot_area theme ~data ~cmap ~annotate ~vmin ~vmax
    ~fmt =
  let shape = Nx.shape data in
  let rows = shape.(0) and cols = shape.(1) in
  let frows = float rows in
  let lo = ref Float.infinity and hi = ref Float.neg_infinity in
  for r = 0 to rows - 1 do
    for c = 0 to cols - 1 do
      let v = Nx.item [ r; c ] data in
      if Float.is_finite v then begin
        if v < !lo then lo := v;
        if v > !hi then hi := v
      end
    done
  done;
  let vlo = Option.value ~default:!lo vmin in
  let vhi = Option.value ~default:!hi vmax in
  let vrange = if vhi = vlo then 1. else vhi -. vlo in
  let cmap = Option.value ~default:theme.Theme.sequential cmap in
  let sf = theme.Theme.scale_factor in
  let prims = ref [] in
  for r = 0 to rows - 1 do
    for c = 0 to cols - 1 do
      let v = Nx.item [ r; c ] data in
      let t = Float.max 0. (Float.min 1. ((v -. vlo) /. vrange)) in
      let cell_color = Cmap.eval cmap t in
      let x0 = data_to_pixel_x sx plot_area (float c) in
      let x1 = data_to_pixel_x sx plot_area (float (c + 1)) in
      let y0 = data_to_pixel_y sy plot_area (frows -. float r) in
      let y1 = data_to_pixel_y sy plot_area (frows -. float (r + 1)) in
      let lx = Float.min x0 x1 and rx = Float.max x0 x1 in
      let ty = Float.min y0 y1 and by = Float.max y0 y1 in
      prims :=
        Scene.Path
          {
            points = [| (lx, ty); (rx, ty); (rx, by); (lx, by) |];
            close = true;
            fill = Some cell_color;
            stroke = None;
            line_width = 0.;
            dash = [];
          }
        :: !prims;
      if annotate then begin
        let text =
          match fmt with Some f -> f v | None -> Printf.sprintf "%.2g" v
        in
        let text_color =
          if Color.lightness cell_color > 0.65 then Color.black else Color.white
        in
        let cx = (lx +. rx) /. 2. in
        let cy = (ty +. by) /. 2. in
        let font_size =
          Float.max (8. *. sf)
            (Float.min
               (Float.abs (rx -. lx) *. 0.4)
               (Float.abs (by -. ty) *. 0.4))
        in
        prims :=
          Scene.Text
            {
              x = cx;
              y = cy;
              content = text;
              font =
                {
                  family = theme.font_tick.family;
                  size = font_size;
                  weight = `Normal;
                };
              color = text_color;
              anchor = `Middle;
              baseline = `Middle;
              angle = 0.;
            }
          :: !prims
      end
    done
  done;
  List.rev !prims

let emit_contour_mark sx sy plot_area theme ~data ~x0 ~x1 ~y0 ~y1 ~levels
    ~filled ~cmap ~color ~line_width ~alpha =
  let sf = theme.Theme.scale_factor in
  let contours = Prepared.prepare_contour ~x0 ~x1 ~y0 ~y1 ~data ~levels in
  let n_levels = List.length contours in
  let prims = ref [] in
  List.iteri
    (fun i cp ->
      let t = if n_levels <= 1 then 0.5 else float i /. float (n_levels - 1) in
      let c =
        match color with
        | Some c -> c
        | None ->
            let cmap = Option.value ~default:theme.Theme.sequential cmap in
            Cmap.eval cmap t
      in
      let c = match alpha with Some a -> Color.with_alpha a c | None -> c in
      let lw = resolve_line_width sf theme line_width in
      List.iter
        (fun seg ->
          let points =
            Array.map
              (fun (dx, dy) ->
                ( data_to_pixel_x sx plot_area dx,
                  data_to_pixel_y sy plot_area dy ))
              seg
          in
          if filled then
            prims :=
              Scene.Path
                {
                  points;
                  close = true;
                  fill = Some c;
                  stroke = None;
                  line_width = 0.;
                  dash = [];
                }
              :: !prims
          else
            prims :=
              Scene.Path
                {
                  points;
                  close = false;
                  fill = None;
                  stroke = Some c;
                  line_width = lw;
                  dash = [];
                }
              :: !prims)
        cp.Prepared.paths)
    contours;
  List.rev !prims

let emit_mark sx sy plot_area theme = function
  | Spec.Line
      { x; y; color; line_width; line_style; step; marker; label = _; alpha } ->
      emit_line_mark sx sy plot_area theme ~x ~y ~color ~line_width ~line_style
        ~step ~marker ~alpha
  | Spec.Point
      { x; y; color; color_by; size; size_by; marker; label = _; alpha } ->
      emit_point_mark sx sy plot_area theme ~x ~y ~color ~color_by ~size
        ~size_by ~marker ~alpha
  | Spec.Bar { x; height; width; bottom; color; label = _; alpha } ->
      emit_bar_mark sx sy plot_area theme ~x ~height ~width ~bottom ~color
        ~alpha
  | Spec.Hist _ ->
      failwith
        "resolve: Spec.Hist reached emit_mark; should have been normalized to \
         Bar by Prepared.compile"
  | Spec.Image { data; extent } -> emit_image_mark sx sy plot_area ~data ~extent
  | Spec.Text_mark { x; y; content; color; font_size } ->
      emit_text_mark sx sy plot_area theme ~x ~y ~content ~color ~font_size
  | Spec.Hline { y; color; line_width; line_style; label = _; alpha } ->
      emit_hline_mark sy plot_area theme ~y ~color ~line_width ~line_style
        ~alpha
  | Spec.Vline { x; color; line_width; line_style; label = _; alpha } ->
      emit_vline_mark sx plot_area theme ~x ~color ~line_width ~line_style
        ~alpha
  | Spec.Abline
      { slope; intercept; color; line_width; line_style; label = _; alpha } ->
      emit_abline_mark sx sy plot_area theme ~slope ~intercept ~color
        ~line_width ~line_style ~alpha
  | Spec.Fill_between { x; y1; y2; where; color; alpha; label = _ } ->
      emit_fill_between_mark sx sy plot_area ~x ~y1 ~y2 ~where ~color ~alpha
  | Spec.Hspan { y0; y1; color; alpha; label = _ } ->
      emit_hspan_mark sy plot_area ~y0 ~y1 ~color ~alpha
  | Spec.Vspan { x0; x1; color; alpha; label = _ } ->
      emit_vspan_mark sx plot_area ~x0 ~x1 ~color ~alpha
  | Spec.Errorbar
      { x; y; yerr; xerr; color; line_width; cap_size; label = _; alpha } ->
      emit_errorbar_mark sx sy plot_area theme ~x ~y ~yerr ~xerr ~color
        ~line_width ~cap_size ~alpha
  | Spec.Heatmap { data; cmap; annotate; vmin; vmax; fmt } ->
      emit_heatmap_mark sx sy plot_area theme ~data ~cmap ~annotate ~vmin ~vmax
        ~fmt
  | Spec.Imshow _ ->
      failwith
        "resolve: Spec.Imshow reached emit_mark; should have been normalized \
         to Image by Prepared.compile"
  | Spec.Contour
      {
        data;
        x0;
        x1;
        y0;
        y1;
        levels;
        filled;
        cmap;
        color;
        line_width;
        label = _;
        alpha;
      } ->
      emit_contour_mark sx sy plot_area theme ~data ~x0 ~x1 ~y0 ~y1 ~levels
        ~filled ~cmap ~color ~line_width ~alpha

(* Axis primitives *)

let scaled_font (theme : Theme.t) (f : Theme.font) =
  { f with size = f.size *. theme.scale_factor }

let emit_axes ~text_measurer sx sy plot_area (theme : Theme.t) ~xticks ~yticks
    (pp : Prepared.panel) =
  let sf = theme.scale_factor in
  let prims = ref [] in
  let axis_color = theme.axis.color in
  let lw = theme.axis.width *. sf in
  let tl = theme.tick_length *. sf in
  List.iter
    (fun (v, label) ->
      let px = data_to_pixel_x sx plot_area v in
      let by = plot_area.ry +. plot_area.rh in
      prims :=
        Scene.Path
          {
            points = [| (px, by); (px, by +. tl) |];
            close = false;
            fill = None;
            stroke = Some axis_color;
            line_width = lw;
            dash = [];
          }
        :: !prims;
      let font = scaled_font theme theme.font_tick in
      prims :=
        Scene.Text
          {
            x = px;
            y = by +. tl +. (8. *. sf);
            content = label;
            font;
            color = axis_color;
            anchor = `Middle;
            baseline = `Top;
            angle = 0.;
          }
        :: !prims)
    xticks;

  (* Y ticks *)
  List.iter
    (fun (v, label) ->
      let py = data_to_pixel_y sy plot_area v in
      let lx = plot_area.rx in
      prims :=
        Scene.Path
          {
            points = [| (lx -. tl, py); (lx, py) |];
            close = false;
            fill = None;
            stroke = Some axis_color;
            line_width = lw;
            dash = [];
          }
        :: !prims;
      let font = scaled_font theme theme.font_tick in
      prims :=
        Scene.Text
          {
            x = lx -. tl -. (8. *. sf);
            y = py;
            content = label;
            font;
            color = axis_color;
            anchor = `End;
            baseline = `Middle;
            angle = 0.;
          }
        :: !prims)
    yticks;

  (* Grid *)
  let show_grid = Option.value ~default:(theme.grid <> None) pp.grid_visible in
  begin match theme.grid with
  | Some grid_line when show_grid ->
      List.iter
        (fun (v, _) ->
          let px = data_to_pixel_x sx plot_area v in
          prims :=
            Scene.Path
              {
                points =
                  [| (px, plot_area.ry); (px, plot_area.ry +. plot_area.rh) |];
                close = false;
                fill = None;
                stroke = Some grid_line.color;
                line_width = grid_line.width *. sf;
                dash = grid_line.dash;
              }
            :: !prims)
        xticks;
      List.iter
        (fun (v, _) ->
          let py = data_to_pixel_y sy plot_area v in
          prims :=
            Scene.Path
              {
                points =
                  [| (plot_area.rx, py); (plot_area.rx +. plot_area.rw, py) |];
                close = false;
                fill = None;
                stroke = Some grid_line.color;
                line_width = grid_line.width *. sf;
                dash = grid_line.dash;
              }
            :: !prims)
        yticks
  | _ -> ()
  end;

  (* Axis border *)
  let lx = plot_area.rx and ty = plot_area.ry in
  let rx = lx +. plot_area.rw and by = ty +. plot_area.rh in
  prims :=
    Scene.Path
      {
        points = [| (lx, ty); (rx, ty); (rx, by); (lx, by) |];
        close = true;
        fill = None;
        stroke = Some axis_color;
        line_width = lw;
        dash = [];
      }
    :: !prims;

  (* Title *)
  begin match pp.title with
  | Some s ->
      let font = scaled_font theme theme.font_title in
      let cx = plot_area.rx +. (plot_area.rw /. 2.) in
      prims :=
        Scene.Text
          {
            x = cx;
            y = plot_area.ry -. (theme.title_gap *. sf);
            content = s;
            font;
            color = axis_color;
            anchor = `Middle;
            baseline = `Bottom;
            angle = 0.;
          }
        :: !prims
  | None -> ()
  end;

  (* X label *)
  begin match pp.x.label with
  | Some s ->
      let font = scaled_font theme theme.font_label in
      let cx = plot_area.rx +. (plot_area.rw /. 2.) in
      let _, tick_h =
        text_measurer ~font:(scaled_font theme theme.font_tick) "0"
      in
      let y =
        plot_area.ry +. plot_area.rh +. tl +. tick_h +. (theme.label_gap *. sf)
      in
      prims :=
        Scene.Text
          {
            x = cx;
            y;
            content = s;
            font;
            color = axis_color;
            anchor = `Middle;
            baseline = `Top;
            angle = 0.;
          }
        :: !prims
  | None -> ()
  end;

  (* Y label *)
  begin match pp.y.label with
  | Some s ->
      let font = scaled_font theme theme.font_label in
      let tick_font = scaled_font theme theme.font_tick in
      let max_ytick_w =
        List.fold_left
          (fun acc (_, label) ->
            let w, _ = text_measurer ~font:tick_font label in
            Float.max acc w)
          0. yticks
      in
      let _, label_h = text_measurer ~font s in
      let x =
        plot_area.rx -. tl -. max_ytick_w -. (8. *. sf)
        -. (theme.label_gap *. sf) -. (label_h /. 2.)
      in
      let y = plot_area.ry +. (plot_area.rh /. 2.) in
      prims :=
        Scene.Text
          {
            x;
            y;
            content = s;
            font;
            color = axis_color;
            anchor = `Middle;
            baseline = `Middle;
            angle = Float.pi /. 2.;
          }
        :: !prims
  | None -> ()
  end;

  List.rev !prims

(* Legend *)

type legend_kind =
  | Legend_line of Spec.line_style option * Spec.marker option
  | Legend_point of Spec.marker
  | Legend_bar
  | Legend_ref_line of Spec.line_style option

let mark_label = function
  | Spec.Line { label; _ }
  | Spec.Point { label; _ }
  | Spec.Bar { label; _ }
  | Spec.Hist { label; _ }
  | Spec.Hline { label; _ }
  | Spec.Vline { label; _ }
  | Spec.Abline { label; _ }
  | Spec.Fill_between { label; _ }
  | Spec.Hspan { label; _ }
  | Spec.Vspan { label; _ }
  | Spec.Errorbar { label; _ }
  | Spec.Contour { label; _ } ->
      label
  | Spec.Image _ | Spec.Text_mark _ | Spec.Heatmap _ | Spec.Imshow _ -> None

let mark_legend_kind = function
  | Spec.Line { line_style; marker; _ } -> Legend_line (line_style, marker)
  | Spec.Point { marker; _ } ->
      Legend_point (Option.value ~default:Spec.Circle marker)
  | Spec.Bar _ | Spec.Hist _ | Spec.Fill_between _ | Spec.Hspan _ | Spec.Vspan _
    ->
      Legend_bar
  | Spec.Hline { line_style; _ }
  | Spec.Vline { line_style; _ }
  | Spec.Abline { line_style; _ } ->
      Legend_ref_line line_style
  | Spec.Errorbar _ -> Legend_ref_line None
  | Spec.Contour { filled = true; _ } -> Legend_bar
  | Spec.Contour _ -> Legend_ref_line None
  | _ -> Legend_bar

let emit_legend ~text_measurer ~loc ~ncol plot_area theme marks =
  let sf = theme.Theme.scale_factor in
  let entries =
    List.filter_map
      (fun m ->
        match mark_label m with
        | Some label ->
            let color =
              match Prepared.mark_color m with
              | Some c -> c
              | None -> Color.black
            in
            Some (label, color, mark_legend_kind m)
        | None -> None)
      marks
  in
  if entries = [] then []
  else begin
    let font = scaled_font theme theme.font_tick in
    let swatch_size = 10. *. sf in
    let gap = 4. *. sf in
    let line_h = Float.max swatch_size (font.size *. 1.2) in
    let margin = 8. *. sf in
    let ncol = max 1 ncol in
    let n_entries = List.length entries in
    let nrows = (n_entries + ncol - 1) / ncol in
    (* Compute per-column max label width *)
    let col_widths = Array.make ncol 0. in
    List.iteri
      (fun i (label, _, _) ->
        let col = i mod ncol in
        let w, _ = text_measurer ~font label in
        col_widths.(col) <- Float.max col_widths.(col) w)
      entries;
    let col_gap = 12. *. sf in
    let col_w i = swatch_size +. gap +. col_widths.(i) in
    let legend_w =
      let total = ref 0. in
      for i = 0 to ncol - 1 do
        total := !total +. col_w i
      done;
      !total +. (col_gap *. float (ncol - 1))
    in
    let legend_h = (float nrows *. (line_h +. gap)) -. gap in
    let loc = Option.value ~default:Spec.Upper_right loc in
    (* x0 is the right edge of the legend area *)
    let x0, y0 =
      match loc with
      | Spec.Upper_right ->
          (plot_area.rx +. plot_area.rw -. margin, plot_area.ry +. margin)
      | Spec.Upper_left ->
          (plot_area.rx +. margin +. legend_w, plot_area.ry +. margin)
      | Spec.Lower_right ->
          ( plot_area.rx +. plot_area.rw -. margin,
            plot_area.ry +. plot_area.rh -. margin -. legend_h )
      | Spec.Lower_left ->
          ( plot_area.rx +. margin +. legend_w,
            plot_area.ry +. plot_area.rh -. margin -. legend_h )
      | Spec.Center ->
          ( plot_area.rx +. ((plot_area.rw +. legend_w) /. 2.),
            plot_area.ry +. ((plot_area.rh -. legend_h) /. 2.) )
      | Spec.Right ->
          ( plot_area.rx +. plot_area.rw -. margin,
            plot_area.ry +. ((plot_area.rh -. legend_h) /. 2.) )
      | Spec.Upper_center ->
          ( plot_area.rx +. ((plot_area.rw +. legend_w) /. 2.),
            plot_area.ry +. margin )
      | Spec.Lower_center ->
          ( plot_area.rx +. ((plot_area.rw +. legend_w) /. 2.),
            plot_area.ry +. plot_area.rh -. margin -. legend_h )
    in
    (* Background box *)
    let inner_pad = 6. *. sf in
    let bg_x = x0 -. legend_w -. inner_pad in
    let bg_y = y0 -. inner_pad in
    let bg_w = legend_w +. (2. *. inner_pad) in
    let bg_h = legend_h +. (2. *. inner_pad) in
    let bg =
      Scene.Path
        {
          points =
            [|
              (bg_x, bg_y);
              (bg_x +. bg_w, bg_y);
              (bg_x +. bg_w, bg_y +. bg_h);
              (bg_x, bg_y +. bg_h);
            |];
          close = true;
          fill = Some (Color.with_alpha 0.85 theme.background);
          stroke = Some (Color.with_alpha 0.3 theme.axis.color);
          line_width = 1. *. sf;
          dash = [];
        }
    in
    (* Compute column x-offsets (from right edge of legend) *)
    let col_offsets = Array.make ncol 0. in
    let acc = ref 0. in
    for c = ncol - 1 downto 0 do
      col_offsets.(c) <- !acc;
      acc := !acc +. col_w c +. col_gap
    done;
    let prims = ref [ bg ] in
    List.iteri
      (fun i (label, color, kind) ->
        let row = i / ncol in
        let col = i mod ncol in
        let y = y0 +. (float row *. (line_h +. gap)) in
        let y_mid = y +. (swatch_size /. 2.) in
        let cx0 = x0 -. col_offsets.(col) in
        begin match kind with
        | Legend_line (line_style, marker) ->
            prims :=
              Scene.Path
                {
                  points = [| (cx0 -. swatch_size, y_mid); (cx0, y_mid) |];
                  close = false;
                  fill = None;
                  stroke = Some color;
                  line_width = theme.line_width *. sf;
                  dash = resolve_dash sf line_style;
                }
              :: !prims;
            begin match marker with
            | Some shape ->
                let ms = 6. *. sf in
                prims :=
                  Scene.Markers
                    {
                      points = [| (cx0 -. (swatch_size /. 2.), y_mid) |];
                      shape;
                      size = ms;
                      sizes = None;
                      fill = Some color;
                      fills = None;
                      stroke = None;
                    }
                  :: !prims
            | None -> ()
            end
        | Legend_point marker ->
            let ms = 8. *. sf in
            prims :=
              Scene.Markers
                {
                  points = [| (cx0 -. (swatch_size /. 2.), y_mid) |];
                  shape = marker;
                  size = ms;
                  sizes = None;
                  fill = Some color;
                  fills = None;
                  stroke = None;
                }
              :: !prims
        | Legend_bar ->
            prims :=
              Scene.Path
                {
                  points =
                    [|
                      (cx0 -. swatch_size, y);
                      (cx0, y);
                      (cx0, y +. swatch_size);
                      (cx0 -. swatch_size, y +. swatch_size);
                    |];
                  close = true;
                  fill = Some color;
                  stroke = None;
                  line_width = 0.;
                  dash = [];
                }
              :: !prims
        | Legend_ref_line line_style ->
            prims :=
              Scene.Path
                {
                  points = [| (cx0 -. swatch_size, y_mid); (cx0, y_mid) |];
                  close = false;
                  fill = None;
                  stroke = Some color;
                  line_width = theme.line_width *. sf;
                  dash = resolve_dash sf line_style;
                }
              :: !prims
        end;
        prims :=
          Scene.Text
            {
              x = cx0 -. swatch_size -. gap;
              y = y_mid;
              content = label;
              font;
              color = theme.axis.color;
              anchor = `End;
              baseline = `Middle;
              angle = 0.;
            }
          :: !prims)
      entries;
    List.rev !prims
  end

(* Colorbar for color_by *)

let emit_colorbar plot_area (theme : Theme.t) ~height_frac (lo, hi) =
  let sf = theme.scale_factor in
  let font = scaled_font theme theme.font_tick in
  let bar_w = 16. *. sf in
  let bar_gap = 12. *. sf in
  let bar_x = plot_area.rx +. plot_area.rw +. bar_gap in
  let bar_y = plot_area.ry in
  let bar_h = plot_area.rh *. height_frac in
  (* Vertical gradient: series of thin horizontal strips *)
  let n_strips = 64 in
  let strip_h = bar_h /. float n_strips in
  let strips =
    List.init n_strips (fun i ->
        let t = 1. -. (float i /. float (n_strips - 1)) in
        let c = Cmap.eval theme.sequential t in
        let sy = bar_y +. (float i *. strip_h) in
        Scene.Path
          {
            points =
              [|
                (bar_x, sy);
                (bar_x +. bar_w, sy);
                (bar_x +. bar_w, sy +. strip_h +. 1.);
                (bar_x, sy +. strip_h +. 1.);
              |];
            close = true;
            fill = Some c;
            stroke = None;
            line_width = 0.;
            dash = [];
          })
  in
  (* Border around colorbar *)
  let border =
    Scene.Path
      {
        points =
          [|
            (bar_x, bar_y);
            (bar_x +. bar_w, bar_y);
            (bar_x +. bar_w, bar_y +. bar_h);
            (bar_x, bar_y +. bar_h);
          |];
        close = true;
        fill = None;
        stroke = Some theme.axis.color;
        line_width = theme.axis.width *. sf;
        dash = [];
      }
  in
  (* Tick labels along the right edge *)
  let ticks = Ticks.generate `Linear ~lo ~hi () in
  let range = hi -. lo in
  let range = if range = 0. then 1. else range in
  let label_x = bar_x +. bar_w +. (6. *. sf) in
  let tick_prims =
    List.filter_map
      (fun (v, label) ->
        let t = (v -. lo) /. range in
        if t < -0.01 || t > 1.01 then None
        else
          let py = bar_y +. bar_h -. (t *. bar_h) in
          Some
            (Scene.Text
               {
                 x = label_x;
                 y = py;
                 content = label;
                 font;
                 color = theme.axis.color;
                 anchor = `Start;
                 baseline = `Middle;
                 angle = 0.;
               }))
      ticks
  in
  strips @ [ border ] @ tick_prims

(* Size guide for size_by *)

let emit_size_guide plot_area (theme : Theme.t) ~y_offset (lo, hi) =
  let sf = theme.scale_factor in
  let font = scaled_font theme theme.font_tick in
  let guide_gap = 12. *. sf in
  let guide_x = plot_area.rx +. plot_area.rw +. guide_gap in
  let max_r = 12. *. sf in
  (* Three representative sizes: max, mid, min *)
  let values = [| hi; (lo +. hi) /. 2.; lo |] in
  let range = hi -. lo in
  let range = if range = 0. then 1. else range in
  let prims = ref [] in
  let cy = ref (plot_area.ry +. y_offset +. max_r +. (4. *. sf)) in
  Array.iter
    (fun v ->
      let t = (v -. lo) /. range in
      let size = ((max_r *. 0.3) +. (max_r *. 0.7 *. Float.sqrt t)) *. 2. in
      let cx = guide_x +. max_r in
      prims :=
        Scene.Markers
          {
            points = [| (cx, !cy) |];
            shape = Spec.Circle;
            size;
            sizes = None;
            fill = Some (Color.with_alpha 0.2 theme.axis.color);
            fills = None;
            stroke = Some theme.axis.color;
          }
        :: !prims;
      let label = Printf.sprintf "%.4g" v in
      let label_x = cx +. max_r +. (6. *. sf) in
      prims :=
        Scene.Text
          {
            x = label_x;
            y = !cy;
            content = label;
            font;
            color = theme.axis.color;
            anchor = `Start;
            baseline = `Middle;
            angle = 0.;
          }
        :: !prims;
      cy := !cy +. (max_r *. 2.) +. (8. *. sf))
    values;
  List.rev !prims

(* Compute layout padding *)

let compute_layout ~text_measurer (theme : Theme.t) (pp : Prepared.panel) xticks
    yticks =
  let sf = theme.scale_factor in
  let tick_font = scaled_font theme theme.font_tick in
  let label_font = scaled_font theme theme.font_label in
  let title_font = scaled_font theme theme.font_title in
  let tl = theme.tick_length *. sf in

  (* Left padding: y-tick labels + gap + optional ylabel *)
  let max_ytick_w =
    List.fold_left
      (fun acc (_, label) ->
        let w, _ = text_measurer ~font:tick_font label in
        Float.max acc w)
      0. yticks
  in
  let left = (theme.padding *. sf) +. max_ytick_w +. tl +. (8. *. sf) in
  let left =
    match pp.y.label with
    | Some s ->
        let _, h = text_measurer ~font:label_font s in
        left +. h +. (theme.label_gap *. sf)
    | None -> left
  in

  (* Bottom padding: x-tick labels + gap + optional xlabel *)
  let _, tick_h = text_measurer ~font:tick_font "0" in
  let bottom = (theme.padding *. sf) +. tick_h +. tl +. (8. *. sf) in
  let bottom =
    match pp.x.label with
    | Some s ->
        let _, h = text_measurer ~font:label_font s in
        bottom +. h +. (theme.label_gap *. sf)
    | None -> bottom
  in

  (* Top padding: title *)
  let top = theme.padding *. sf in
  let top =
    match pp.title with
    | Some s ->
        let _, h = text_measurer ~font:title_font s in
        top +. h +. (theme.title_gap *. sf)
    | None -> top
  in

  (* Right padding — extra space for colorbar / size guide *)
  let right =
    let base = theme.padding *. sf in
    let colorbar_w =
      match pp.colorbar_range with
      | Some (lo, hi) ->
          let bar_w = 16. *. sf in
          let bar_gap = 12. *. sf in
          let ticks = Ticks.generate `Linear ~lo ~hi () in
          let max_label_w =
            List.fold_left
              (fun acc (_, label) ->
                let w, _ = text_measurer ~font:tick_font label in
                Float.max acc w)
              0. ticks
          in
          bar_gap +. bar_w +. (6. *. sf) +. max_label_w +. (4. *. sf)
      | None -> 0.
    in
    let size_guide_w =
      match pp.size_by_range with
      | Some (lo, hi) ->
          let guide_gap = 12. *. sf in
          let max_r = 12. *. sf in
          let mid = (lo +. hi) /. 2. in
          let max_label_w =
            List.fold_left
              (fun acc v ->
                let w, _ =
                  text_measurer ~font:tick_font (Printf.sprintf "%.4g" v)
                in
                Float.max acc w)
              0. [ lo; mid; hi ]
          in
          guide_gap +. (max_r *. 2.) +. (6. *. sf) +. max_label_w +. (4. *. sf)
      | None -> 0.
    in
    base +. Float.max colorbar_w size_guide_w
  in

  (left, top, right, bottom)

(* Resolve a single prepared panel *)

let resolve_panel ~text_measurer theme region (pp : Prepared.panel) =
  let theme = Option.value ~default:theme pp.theme_override in

  let sx, xticks = Axis.make_scale_and_ticks pp.x in
  let sy, yticks = Axis.make_scale_and_ticks pp.y in

  let left, top, right, bottom =
    compute_layout ~text_measurer theme pp xticks yticks
  in

  let plot_area =
    {
      rx = region.rx +. left;
      ry = region.ry +. top;
      rw = Float.max 1. (region.rw -. left -. right);
      rh = Float.max 1. (region.rh -. top -. bottom);
    }
  in

  (* Background *)
  let bg =
    Scene.Path
      {
        points =
          [|
            (region.rx, region.ry);
            (region.rx +. region.rw, region.ry);
            (region.rx +. region.rw, region.ry +. region.rh);
            (region.rx, region.ry +. region.rh);
          |];
        close = true;
        fill = Some theme.background;
        stroke = None;
        line_width = 0.;
        dash = [];
      }
  in

  (* Axes decorations *)
  let axes_prims =
    emit_axes ~text_measurer sx sy plot_area theme ~xticks ~yticks pp
  in

  (* Data marks inside clip region *)
  let data_prims = List.concat_map (emit_mark sx sy plot_area theme) pp.marks in
  let clipped_data =
    Scene.Clip
      {
        x = plot_area.rx;
        y = plot_area.ry;
        w = plot_area.rw;
        h = plot_area.rh;
        children = data_prims;
      }
  in

  (* Legend *)
  let legend_prims =
    emit_legend ~text_measurer ~loc:pp.legend_loc ~ncol:pp.legend_ncol plot_area
      theme pp.marks
  in

  (* Colorbar for color_by *)
  let has_both = pp.colorbar_range <> None && pp.size_by_range <> None in
  let colorbar_prims =
    match pp.colorbar_range with
    | Some range ->
        let height_frac = if has_both then 0.55 else 1. in
        emit_colorbar plot_area theme ~height_frac range
    | None -> []
  in

  (* Size guide for size_by *)
  let size_guide_prims =
    match pp.size_by_range with
    | Some range ->
        let y_offset = if has_both then plot_area.rh *. 0.6 else 0. in
        emit_size_guide plot_area theme ~y_offset range
    | None -> []
  in

  [ bg; clipped_data ] @ axes_prims @ legend_prims @ colorbar_prims
  @ size_guide_prims

(* Resolve a prepared grid layout *)

let resolve_grid ~resolve_prepared ~text_measurer theme region rows gap =
  let nrows = List.length rows in
  let ncols =
    List.fold_left (fun acc row -> max acc (List.length row)) 0 rows
  in
  if nrows = 0 || ncols = 0 then []
  else begin
    let cell_w = (region.rw -. (gap *. float (ncols - 1))) /. float ncols in
    let cell_h = (region.rh -. (gap *. float (nrows - 1))) /. float nrows in
    let prims = ref [] in
    List.iteri
      (fun ri row ->
        List.iteri
          (fun ci prepared ->
            let cell_region =
              {
                rx = region.rx +. (float ci *. (cell_w +. gap));
                ry = region.ry +. (float ri *. (cell_h +. gap));
                rw = cell_w;
                rh = cell_h;
              }
            in
            let p =
              resolve_prepared ~text_measurer theme cell_region prepared
            in
            prims := List.rev_append p !prims)
          row)
      rows;
    List.rev !prims
  end

(* Grid-level decorations *)

let emit_grid_decorations ~text_measurer theme region
    (gd : Prepared.grid_decorations) all_marks =
  let sf = theme.Theme.scale_factor in
  let color = theme.axis.color in
  let prims = ref [] in
  let r = ref region in

  (* Title: above grid *)
  begin match gd.gd_title with
  | Some s ->
      let font = scaled_font theme theme.font_title in
      let _, title_h = text_measurer ~font s in
      let title_gap = theme.title_gap *. sf in
      prims :=
        Scene.Text
          {
            x = !r.rx +. (!r.rw /. 2.);
            y = !r.ry +. title_h;
            content = s;
            font;
            color;
            anchor = `Middle;
            baseline = `Bottom;
            angle = 0.;
          }
        :: !prims;
      let used = title_h +. title_gap in
      r := { !r with ry = !r.ry +. used; rh = !r.rh -. used }
  | None -> ()
  end;

  (* Xlabel: below grid *)
  begin match gd.gd_xlabel with
  | Some s ->
      let font = scaled_font theme theme.font_label in
      let _, label_h = text_measurer ~font s in
      let label_gap = theme.label_gap *. sf in
      let used = label_h +. label_gap in
      prims :=
        Scene.Text
          {
            x = !r.rx +. (!r.rw /. 2.);
            y = !r.ry +. !r.rh -. label_gap;
            content = s;
            font;
            color;
            anchor = `Middle;
            baseline = `Bottom;
            angle = 0.;
          }
        :: !prims;
      r := { !r with rh = !r.rh -. used }
  | None -> ()
  end;

  (* Ylabel: left of grid, rotated *)
  begin match gd.gd_ylabel with
  | Some s ->
      let font = scaled_font theme theme.font_label in
      let _, label_h = text_measurer ~font s in
      let label_gap = theme.label_gap *. sf in
      let used = label_h +. label_gap in
      prims :=
        Scene.Text
          {
            x = !r.rx +. (label_h /. 2.);
            y = !r.ry +. (!r.rh /. 2.);
            content = s;
            font;
            color;
            anchor = `Middle;
            baseline = `Middle;
            angle = Float.pi /. 2.;
          }
        :: !prims;
      r := { !r with rx = !r.rx +. used; rw = !r.rw -. used }
  | None -> ()
  end;

  (* Shared legend *)
  let legend_prims =
    match gd.gd_legend_loc with
    | Some loc ->
        emit_legend ~text_measurer ~loc:(Some loc)
          ~ncol:gd.Prepared.gd_legend_ncol !r theme all_marks
    | None -> []
  in

  (List.rev !prims, legend_prims, !r)

(* Top-level resolve from Prepared.t *)

let rec resolve_tree ~text_measurer theme region = function
  | Prepared.Panel pp -> resolve_panel ~text_measurer theme region pp
  | Prepared.Grid { rows; gap } ->
      let gap_px = gap *. Float.min region.rw region.rh in
      resolve_grid ~resolve_prepared:resolve_tree ~text_measurer theme region
        rows gap_px
  | Prepared.Decorated_grid { decorations; inner; all_marks } ->
      let theme = Option.value ~default:theme decorations.gd_theme_override in
      let dec_prims, legend_prims, grid_region =
        emit_grid_decorations ~text_measurer theme region decorations all_marks
      in
      dec_prims
      @ resolve_tree ~text_measurer theme grid_region inner
      @ legend_prims

let resolve_prepared ~text_measurer ~theme ~width ~height prepared =
  let region = { rx = 0.; ry = 0.; rw = width; rh = height } in
  let primitives = resolve_tree ~text_measurer theme region prepared in
  { Scene.width; height; primitives }

(* Convenience: compile + resolve in one step *)

let resolve ~text_measurer ~theme ~width ~height spec =
  let prepared = Prepared.compile ~theme spec in
  resolve_prepared ~text_measurer ~theme ~width ~height prepared
