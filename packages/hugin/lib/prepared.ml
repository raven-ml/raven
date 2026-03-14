(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Data-only compilation: Spec.t -> Prepared.t

   Compiles once per dataset. Separates data-dependent work (collecting
   decorations, histogram binning, auto-coloring, data bounds) from
   layout-dependent work (pixel coordinates, text measurement) which lives in
   Resolve. *)

(* Data bounds *)

let nx_finite_range (arr : Nx.float32_t) =
  let n = (Nx.shape arr).(0) in
  let lo = ref Float.infinity and hi = ref Float.neg_infinity in
  for i = 0 to n - 1 do
    let v = Nx.item [ i ] arr in
    if Float.is_finite v then begin
      if v < !lo then lo := v;
      if v > !hi then hi := v
    end
  done;
  (!lo, !hi)

let expand_range scale lo hi =
  if lo = hi then (lo -. 1., hi +. 1.)
  else
    match scale with
    | `Log ->
        let lo_log = Float.log10 (Float.max 1e-10 lo) in
        let hi_log = Float.log10 (Float.max 1e-10 hi) in
        let pad = (hi_log -. lo_log) *. 0.05 in
        (Float.pow 10. (lo_log -. pad), Float.pow 10. (hi_log +. pad))
    | `Sqrt ->
        let lo = Float.max 0. lo in
        let pad = (hi -. lo) *. 0.05 in
        (Float.max 0. (lo -. pad), hi +. pad)
    | `Asinh | `Symlog _ | `Linear ->
        let pad = (hi -. lo) *. 0.05 in
        (lo -. pad, hi +. pad)

let mark_x_range = function
  | Spec.Line { x; _ } | Spec.Point { x; _ } -> Some (nx_finite_range x)
  | Spec.Bar { x; width; _ } ->
      let lo, hi = nx_finite_range x in
      let w = (match width with Some w -> w | None -> 0.8) /. 2. in
      Some (lo -. w, hi +. w)
  | Spec.Hist { x; _ } -> Some (nx_finite_range x)
  | Spec.Image { extent = Some (xmin, xmax, _, _); _ } ->
      Some (Float.min xmin xmax, Float.max xmin xmax)
  | Spec.Image _ -> None
  | Spec.Text_mark { x; _ } -> Some (x, x)
  | Spec.Hline _ -> None
  | Spec.Vline { x; _ } -> Some (x, x)
  | Spec.Abline _ -> None
  | Spec.Fill_between { x; _ } -> Some (nx_finite_range x)
  | Spec.Errorbar { x; xerr; _ } ->
      let lo, hi = nx_finite_range x in
      let lo, hi =
        match xerr with
        | Some (`Symmetric e) ->
            let _, emax = nx_finite_range e in
            (lo -. emax, hi +. emax)
        | Some (`Asymmetric (elo, ehi)) ->
            let _, emlo = nx_finite_range elo in
            let _, emhi = nx_finite_range ehi in
            (lo -. emlo, hi +. emhi)
        | None -> (lo, hi)
      in
      Some (lo, hi)
  | Spec.Hspan _ -> None
  | Spec.Vspan { x0; x1; _ } -> Some (Float.min x0 x1, Float.max x0 x1)
  | Spec.Heatmap { data; _ } ->
      let shape = Nx.shape data in
      let cols = float shape.(1) in
      Some (0., cols)
  | Spec.Imshow _ -> None
  | Spec.Contour { x0; x1; _ } -> Some (Float.min x0 x1, Float.max x0 x1)

let mark_y_range = function
  | Spec.Line { y; _ } | Spec.Point { y; _ } -> Some (nx_finite_range y)
  | Spec.Bar { height; bottom; _ } ->
      let lo, hi = nx_finite_range height in
      Some (Float.min bottom (bottom +. lo), Float.max bottom (bottom +. hi))
  | Spec.Hist _ -> None
  | Spec.Image { extent = Some (_, _, ymin, ymax); _ } ->
      Some (Float.min ymin ymax, Float.max ymin ymax)
  | Spec.Image _ -> None
  | Spec.Text_mark { y; _ } -> Some (y, y)
  | Spec.Hline { y; _ } -> Some (y, y)
  | Spec.Vline _ -> None
  | Spec.Abline _ -> None
  | Spec.Fill_between { y1; y2; _ } ->
      let lo1, hi1 = nx_finite_range y1 in
      let lo2, hi2 = nx_finite_range y2 in
      Some (Float.min lo1 lo2, Float.max hi1 hi2)
  | Spec.Errorbar { y; yerr; _ } ->
      let lo, hi = nx_finite_range y in
      let lo, hi =
        match yerr with
        | `Symmetric e ->
            let _, emax = nx_finite_range e in
            (lo -. emax, hi +. emax)
        | `Asymmetric (elo, ehi) ->
            let _, emlo = nx_finite_range elo in
            let _, emhi = nx_finite_range ehi in
            (lo -. emlo, hi +. emhi)
      in
      Some (lo, hi)
  | Spec.Hspan { y0; y1; _ } -> Some (Float.min y0 y1, Float.max y0 y1)
  | Spec.Vspan _ -> None
  | Spec.Heatmap { data; _ } ->
      let shape = Nx.shape data in
      let rows = float shape.(0) in
      Some (0., rows)
  | Spec.Imshow _ -> None
  | Spec.Contour { y0; y1; _ } -> Some (Float.min y0 y1, Float.max y0 y1)

let union_range a b =
  match (a, b) with
  | None, x | x, None -> x
  | Some (a0, a1), Some (b0, b1) -> Some (Float.min a0 b0, Float.max a1 b1)

let compute_data_bounds ~xscale ~yscale marks =
  let xr =
    List.fold_left (fun acc m -> union_range acc (mark_x_range m)) None marks
  in
  let yr =
    List.fold_left (fun acc m -> union_range acc (mark_y_range m)) None marks
  in
  let xlo, xhi =
    match xr with Some (a, b) -> expand_range xscale a b | None -> (0., 1.)
  in
  let ylo, yhi =
    match yr with Some (a, b) -> expand_range yscale a b | None -> (0., 1.)
  in
  (xlo, xhi, ylo, yhi)

(* Collect decorations from spec tree *)

type collected = {
  marks : Spec.mark list;
  x : Axis.config;
  y : Axis.config;
  title : string option;
  grid_visible : bool option;
  frame_visible : bool option;
  legend_loc : Spec.legend_loc option;
  legend_ncol : int;
  theme_override : Theme.t option;
}

let empty_collected =
  {
    marks = [];
    x = Axis.empty_config;
    y = Axis.empty_config;
    title = None;
    grid_visible = None;
    frame_visible = None;
    legend_loc = None;
    legend_ncol = 1;
    theme_override = None;
  }

let rec collect c = function
  | Spec.Mark m -> { c with marks = m :: c.marks }
  | Spec.Layers ts -> List.fold_left collect c ts
  | Spec.Decorated { inner; decorations } ->
      let c = collect c inner in
      List.fold_left apply_decoration c decorations
  | Spec.Grid _ -> c

and apply_decoration c = function
  | Spec.Title s when c.title = None -> { c with title = Some s }
  | Spec.Xlabel s when c.x.label = None ->
      { c with x = { c.x with label = Some s } }
  | Spec.Ylabel s when c.y.label = None ->
      { c with y = { c.y with label = Some s } }
  | Spec.Xlim (lo, hi) when c.x.lim = None ->
      { c with x = { c.x with lim = Some (lo, hi) } }
  | Spec.Ylim (lo, hi) when c.y.lim = None ->
      { c with y = { c.y with lim = Some (lo, hi) } }
  | Spec.Xscale s when c.x.scale = None ->
      { c with x = { c.x with scale = Some s } }
  | Spec.Yscale s when c.y.scale = None ->
      { c with y = { c.y with scale = Some s } }
  | Spec.Xinvert -> { c with x = { c.x with invert = true } }
  | Spec.Yinvert -> { c with y = { c.y with invert = true } }
  | Spec.Grid_visible v when c.grid_visible = None ->
      { c with grid_visible = Some v }
  | Spec.Legend (loc, ncol) when c.legend_loc = None ->
      { c with legend_loc = Some loc; legend_ncol = ncol }
  | Spec.Xticks t when c.x.ticks = None ->
      { c with x = { c.x with ticks = Some t } }
  | Spec.Yticks t when c.y.ticks = None ->
      { c with y = { c.y with ticks = Some t } }
  | Spec.With_theme t when c.theme_override = None ->
      { c with theme_override = Some t }
  | Spec.Xtick_format f when c.x.tick_format = None ->
      { c with x = { c.x with tick_format = Some f } }
  | Spec.Ytick_format f when c.y.tick_format = None ->
      { c with y = { c.y with tick_format = Some f } }
  | Spec.Frame v when c.frame_visible = None ->
      { c with frame_visible = Some v }
  | _ -> c

(* Auto-coloring *)

let mark_color = function
  | Spec.Line { color; _ }
  | Spec.Point { color; _ }
  | Spec.Bar { color; _ }
  | Spec.Hist { color; _ }
  | Spec.Text_mark { color; _ }
  | Spec.Hline { color; _ }
  | Spec.Vline { color; _ }
  | Spec.Abline { color; _ }
  | Spec.Fill_between { color; _ }
  | Spec.Hspan { color; _ }
  | Spec.Vspan { color; _ }
  | Spec.Errorbar { color; _ }
  | Spec.Contour { color; _ } ->
      color
  | Spec.Image _ | Spec.Heatmap _ | Spec.Imshow _ -> None

let auto_color (theme : Theme.t) marks =
  let n_palette = Array.length theme.palette in
  List.mapi
    (fun i m ->
      match mark_color m with
      | Some _ -> m
      | None -> (
          let c = theme.palette.(i mod n_palette) in
          match m with
          | Spec.Line r -> Spec.Line { r with color = Some c }
          | Spec.Point r -> Spec.Point { r with color = Some c }
          | Spec.Bar r -> Spec.Bar { r with color = Some c }
          | Spec.Hist r -> Spec.Hist { r with color = Some c }
          | Spec.Hline r -> Spec.Hline { r with color = Some c }
          | Spec.Vline r -> Spec.Vline { r with color = Some c }
          | Spec.Abline r -> Spec.Abline { r with color = Some c }
          | Spec.Fill_between r -> Spec.Fill_between { r with color = Some c }
          | Spec.Hspan r -> Spec.Hspan { r with color = Some c }
          | Spec.Vspan r -> Spec.Vspan { r with color = Some c }
          | Spec.Errorbar r -> Spec.Errorbar { r with color = Some c }
          | Spec.Contour r -> Spec.Contour { r with color = Some c }
          | m -> m))
    marks

(* Histogram normalization — convert Hist to Bar *)

let normalize_hist marks =
  List.map
    (fun m ->
      match m with
      | Spec.Hist { x; bins; density; color; label } ->
          let xmin, xmax = nx_finite_range x in
          let edges =
            match bins with
            | `Num num_bins ->
                Array.init (num_bins + 1) (fun i ->
                    xmin +. ((xmax -. xmin) *. float i /. float num_bins))
            | `Edges e -> e
          in
          let num_bins = Array.length edges - 1 in
          let n = (Nx.shape x).(0) in
          let counts = Array.make num_bins 0. in
          let binned = ref 0 in
          for i = 0 to n - 1 do
            let v = Nx.item [ i ] x in
            if Float.is_finite v && v >= edges.(0) && v <= edges.(num_bins) then begin
              incr binned;
              let bin = ref 0 in
              while !bin < num_bins - 1 && v >= edges.(!bin + 1) do
                incr bin
              done;
              counts.(!bin) <- counts.(!bin) +. 1.
            end
          done;
          if density then begin
            let total =
              let b = float !binned in
              if b = 0. then 1. else b
            in
            for i = 0 to num_bins - 1 do
              let w = edges.(i + 1) -. edges.(i) in
              counts.(i) <- counts.(i) /. (total *. w)
            done
          end;
          let bar_x =
            Nx.init Float32 [| num_bins |] (fun idx ->
                let i = idx.(0) in
                (edges.(i) +. edges.(i + 1)) /. 2.)
          in
          let bar_h =
            Nx.init Float32 [| num_bins |] (fun idx -> counts.(idx.(0)))
          in
          let w = if num_bins > 0 then edges.(1) -. edges.(0) else 1. in
          Spec.Bar
            {
              x = bar_x;
              height = bar_h;
              width = Some w;
              bottom = 0.;
              color;
              label;
              alpha = None;
            }
      | m -> m)
    marks

(* Guide ranges *)

let color_by_range marks =
  List.fold_left
    (fun acc m ->
      match m with
      | Spec.Point { color_by = Some cb; _ } ->
          let lo, hi = nx_finite_range cb in
          union_range acc (Some (lo, hi))
      | _ -> acc)
    None marks

let size_by_range marks =
  List.fold_left
    (fun acc m ->
      match m with
      | Spec.Point { size_by = Some sb; _ } ->
          let lo, hi = nx_finite_range sb in
          union_range acc (Some (lo, hi))
      | _ -> acc)
    None marks

(* Collect marks from all panels in a spec tree *)

let rec collect_all_marks = function
  | Spec.Mark m -> [ m ]
  | Spec.Layers ts -> List.concat_map collect_all_marks ts
  | Spec.Decorated { inner; _ } -> collect_all_marks inner
  | Spec.Grid { rows; _ } ->
      List.concat_map (List.concat_map collect_all_marks) rows

(* Grid-level decorations *)

type grid_decorations = {
  gd_title : string option;
  gd_xlabel : string option;
  gd_ylabel : string option;
  gd_legend_loc : Spec.legend_loc option;
  gd_legend_ncol : int;
  gd_theme_override : Theme.t option;
}

let extract_grid_decorations decorations =
  let d =
    {
      gd_title = None;
      gd_xlabel = None;
      gd_ylabel = None;
      gd_legend_loc = None;
      gd_legend_ncol = 1;
      gd_theme_override = None;
    }
  in
  List.fold_left
    (fun d dec ->
      match dec with
      | Spec.Title s when d.gd_title = None -> { d with gd_title = Some s }
      | Spec.Xlabel s when d.gd_xlabel = None -> { d with gd_xlabel = Some s }
      | Spec.Ylabel s when d.gd_ylabel = None -> { d with gd_ylabel = Some s }
      | Spec.Legend (loc, ncol) when d.gd_legend_loc = None ->
          { d with gd_legend_loc = Some loc; gd_legend_ncol = ncol }
      | Spec.With_theme t when d.gd_theme_override = None ->
          { d with gd_theme_override = Some t }
      | _ -> d)
    d decorations

(* Prepared panel — all data-only work done *)

type panel = {
  marks : Spec.mark list;
  x : Axis.t;
  y : Axis.t;
  title : string option;
  legend_loc : Spec.legend_loc option;
  legend_ncol : int;
  grid_visible : bool option;
  frame_visible : bool option;
  theme_override : Theme.t option;
  colorbar_range : (float * float) option;
  size_by_range : (float * float) option;
}

type t =
  | Panel of panel
  | Grid of { rows : t list list; gap : float }
  | Decorated_grid of {
      decorations : grid_decorations;
      inner : t;
      all_marks : Spec.mark list;
    }

(* Imshow: rasterize float32 data to uint8 RGB via stretch + colormap *)

let apply_stretch stretch v =
  match stretch with
  | `Linear -> v
  | `Log -> Float.log10 (1. +. (9. *. v)) /. Float.log10 10.
  | `Sqrt -> Float.sqrt (Float.max 0. v)
  | `Asinh ->
      let a = 10. in
      Float.asinh (a *. v) /. Float.asinh a
  | `Power a -> Float.pow (Float.max 0. v) a

let rasterize_imshow ~stretch ~cmap ~vmin ~vmax (data : Nx.float32_t) =
  let shape = Nx.shape data in
  let rows = shape.(0) and cols = shape.(1) in
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
  let vlo = match vmin with Some v -> v | None -> !lo in
  let vhi = match vmax with Some v -> v | None -> !hi in
  let vrange = if vhi = vlo then 1. else vhi -. vlo in
  let rgb = Nx.zeros Nx.uint8 [| rows; cols; 3 |] in
  for r = 0 to rows - 1 do
    for c = 0 to cols - 1 do
      let v = Nx.item [ r; c ] data in
      let t = Float.max 0. (Float.min 1. ((v -. vlo) /. vrange)) in
      let t = apply_stretch stretch t in
      let t = Float.max 0. (Float.min 1. t) in
      let color = Cmap.eval cmap t in
      let cr, cg, cb, _ = Color.to_rgba color in
      Nx.set_item [ r; c; 0 ] (int_of_float (cr *. 255.)) rgb;
      Nx.set_item [ r; c; 1 ] (int_of_float (cg *. 255.)) rgb;
      Nx.set_item [ r; c; 2 ] (int_of_float (cb *. 255.)) rgb
    done
  done;
  rgb

let normalize_imshow (theme : Theme.t) marks =
  List.map
    (fun m ->
      match m with
      | Spec.Imshow { data; stretch; cmap; vmin; vmax } ->
          let cmap = match cmap with Some c -> c | None -> theme.sequential in
          let rgb = rasterize_imshow ~stretch ~cmap ~vmin ~vmax data in
          Spec.Image { data = rgb; extent = None }
      | m -> m)
    marks

(* Contour tracing via marching squares *)

type contour_paths = { level : float; paths : (float * float) array list }

(* Join 2-point segments that share endpoints into connected polylines. Marching
   squares produces one segment per cell edge crossing. Segments from adjacent
   cells share exact floating-point endpoints (deterministic lerp), so we chain
   them with exact equality via a hashtable. *)
let join_segments segments =
  let n = List.length segments in
  if n = 0 then []
  else
    let segs = Array.of_list segments in
    let visited = Array.make n false in
    let adj = Hashtbl.create (2 * n) in
    Array.iteri
      (fun i (a, b) ->
        let add pt =
          let cur = try Hashtbl.find adj pt with Not_found -> [] in
          Hashtbl.replace adj pt (i :: cur)
        in
        add a;
        add b)
      segs;
    let find_unvisited_neighbor pt =
      match Hashtbl.find adj pt with
      | exception Not_found -> None
      | neighbors ->
          let rec scan = function
            | [] -> None
            | j :: rest -> if visited.(j) then scan rest else Some j
          in
          scan neighbors
    in
    let chains = ref [] in
    for start = 0 to n - 1 do
      if not visited.(start) then begin
        visited.(start) <- true;
        let a0, b0 = segs.(start) in
        (* front: backward extensions (cons'd, so in chain order). back: forward
           extensions (cons'd, so reversed). *)
        let front = ref [ a0 ] in
        let back = ref [ b0 ] in
        (* Extend forward from b0 *)
        let cur = ref b0 in
        let go = ref true in
        while !go do
          match find_unvisited_neighbor !cur with
          | None -> go := false
          | Some j ->
              visited.(j) <- true;
              let a, b = segs.(j) in
              let next = if a = !cur then b else a in
              back := next :: !back;
              cur := next
        done;
        (* Extend backward from a0 *)
        cur := a0;
        go := true;
        while !go do
          match find_unvisited_neighbor !cur with
          | None -> go := false
          | Some j ->
              visited.(j) <- true;
              let a, b = segs.(j) in
              let next = if a = !cur then b else a in
              front := next :: !front;
              cur := next
        done;
        (* front is in chain order; back is reversed *)
        chains := Array.of_list (!front @ List.rev !back) :: !chains
      end
    done;
    List.rev !chains

let trace_contours ~rows ~cols (data : Nx.float32_t) levels =
  let get r c =
    if r >= 0 && r < rows && c >= 0 && c < cols then Nx.item [ r; c ] data
    else 0.
  in
  List.map
    (fun level ->
      let segments = ref [] in
      for r = 0 to rows - 2 do
        for c = 0 to cols - 2 do
          let v00 = get r c in
          let v10 = get r (c + 1) in
          let v11 = get (r + 1) (c + 1) in
          let v01 = get (r + 1) c in
          let b0 = if v00 >= level then 1 else 0 in
          let b1 = if v10 >= level then 1 else 0 in
          let b2 = if v11 >= level then 1 else 0 in
          let b3 = if v01 >= level then 1 else 0 in
          let case = b0 lor (b1 lsl 1) lor (b2 lsl 2) lor (b3 lsl 3) in
          let lerp va vb =
            let d = vb -. va in
            if Float.abs d < 1e-30 then 0.5 else (level -. va) /. d
          in
          let fc = float c and fr = float r in
          let top = (fc +. lerp v00 v10, fr) in
          let right = (fc +. 1., fr +. lerp v10 v11) in
          let bottom = (fc +. lerp v01 v11, fr +. 1.) in
          let left = (fc, fr +. lerp v00 v01) in
          let add a b = segments := (a, b) :: !segments in
          begin match case with
          | 0 | 15 -> ()
          | 1 | 14 -> add top left
          | 2 | 13 -> add top right
          | 3 | 12 -> add left right
          | 4 | 11 -> add right bottom
          | 5 ->
              let center = (v00 +. v10 +. v11 +. v01) /. 4. in
              if center >= level then begin
                add top right;
                add bottom left
              end
              else begin
                add top left;
                add bottom right
              end
          | 6 | 9 -> add top bottom
          | 7 | 8 -> add bottom left
          | 10 ->
              let center = (v00 +. v10 +. v11 +. v01) /. 4. in
              if center >= level then begin
                add top left;
                add bottom right
              end
              else begin
                add top right;
                add bottom left
              end
          | _ -> ()
          end
        done
      done;
      let paths = join_segments !segments in
      { level; paths })
    levels

let prepare_contour ~x0 ~x1 ~y0 ~y1 ~data ~levels =
  let shape = Nx.shape data in
  let rows = shape.(0) and cols = shape.(1) in
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
  let vlo = !lo and vhi = !hi in
  let level_values =
    match levels with
    | `Values a -> Array.to_list a
    | `Num n ->
        let range = vhi -. vlo in
        if range = 0. then [ vlo ]
        else
          List.init n (fun i ->
              vlo +. (range *. (float (i + 1) /. float (n + 1))))
  in
  let contours = trace_contours ~rows ~cols data level_values in
  (* Map grid coords to data coords *)
  let xscale = (x1 -. x0) /. float (cols - 1) in
  let yscale = (y1 -. y0) /. float (rows - 1) in
  List.map
    (fun cp ->
      let paths =
        List.map
          (fun seg ->
            Array.map
              (fun (gc, gr) -> (x0 +. (gc *. xscale), y0 +. (gr *. yscale)))
              seg)
          cp.paths
      in
      { cp with paths })
    contours

(* Compile a spec tree into a prepared tree *)

let compile_panel theme spec =
  let c = collect empty_collected spec in
  let c = { c with marks = List.rev c.marks } in
  let theme = Option.value ~default:theme c.theme_override in
  let marks =
    normalize_hist (normalize_imshow theme (auto_color theme c.marks))
  in
  let xscale = Option.value ~default:`Linear c.x.scale in
  let yscale = Option.value ~default:`Linear c.y.scale in
  let xlo, xhi, ylo, yhi = compute_data_bounds ~xscale ~yscale marks in
  let x = Axis.resolve ~data_lo:xlo ~data_hi:xhi c.x in
  let y = Axis.resolve ~data_lo:ylo ~data_hi:yhi c.y in
  Panel
    {
      marks;
      x;
      y;
      title = c.title;
      legend_loc = c.legend_loc;
      legend_ncol = c.legend_ncol;
      grid_visible = c.grid_visible;
      frame_visible = c.frame_visible;
      theme_override = c.theme_override;
      colorbar_range = color_by_range marks;
      size_by_range = size_by_range marks;
    }

let rec compile ~theme spec =
  match spec with
  | Spec.Grid { rows; gap } ->
      let rows = List.map (List.map (compile ~theme)) rows in
      Grid { rows; gap }
  | Spec.Decorated { inner = Spec.Grid _ as g; decorations } ->
      let gd = extract_grid_decorations decorations in
      let theme = Option.value ~default:theme gd.gd_theme_override in
      let all_marks = auto_color theme (collect_all_marks g) in
      let inner = compile ~theme g in
      Decorated_grid { decorations = gd; inner; all_marks }
  | spec -> compile_panel theme spec
