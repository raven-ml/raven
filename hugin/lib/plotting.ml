let plot ?color ?linewidth ?linestyle ?marker ?label ~x ~y ax =
  let artist = Artist.line2d ?color ?linewidth ?linestyle ?marker ?label x y in
  Axes.add_artist artist ax

let plot_y ?color ?linewidth ?linestyle ?marker ?label ~y ax =
  let n_elements =
    match Ndarray.shape y with
    | [| size |] -> size
    | _ ->
        Printf.eprintf
          "Warning: Plotting.plot_y requires a non-empty 1D Ndarray. Skipping.\n\
           %!";
        0
  in
  if n_elements <= 0 then ax
  else
    let x_data =
      Ndarray.create Ndarray.float32 [| n_elements |]
        (Array.init n_elements (fun i -> float_of_int i))
    in
    plot ?color ?linewidth ?linestyle ?marker ?label ~x:x_data ~y ax

let plot3d ?color ?linewidth ?linestyle ?marker ?label ~x ~y ~z ax =
  if ax.Axes.projection <> Axes.ThreeD then
    Printf.eprintf "Warning: Calling plot3d on a 2D Axes object.\n%!";
  let artist =
    Artist.line3d ?color ?linewidth ?linestyle ?marker ?label x y z
  in
  Axes.add_artist artist ax

let scatter ?s ?c ?marker ?label ~x ~y ax =
  let artist = Artist.scatter ?s ?c ?marker ?label x y in
  Axes.add_artist artist ax

let bar ?width ?(bottom = 0.0) ?color ?label ~x ~height ax =
  let artist = Artist.bar ?width ~bottom ?color ?label ~height x in
  let ax = Axes.add_artist artist ax in
  let current_ymin_opt, _ = ax.Axes.ylim in
  if current_ymin_opt = None then Axes.set_ylim ~min:bottom ax else ax

let hist ?(bins = `Num 10) ?range ?(density = false) ?color ?label ~x ax =
  let x_min, x_max =
    match range with
    | Some (min_val, max_val) -> (min_val, max_val)
    | None ->
        let min_val = Ndarray.min x |> Ndarray.get_item [||] in
        let max_val = Ndarray.max x |> Ndarray.get_item [||] in
        (min_val, max_val)
  in
  let edges =
    match bins with
    | `Num n ->
        Array.init (n + 1) (fun i ->
            x_min +. ((x_max -. x_min) *. float_of_int i /. float_of_int n))
    | `Edges arr -> arr
  in
  let num_bins = Array.length edges - 1 in
  if num_bins <= 0 then invalid_arg "Plotting.hist: Must have at least one bin";
  let counts = Array.make num_bins 0. in
  let total_count = ref 0. in

  Ndarray.iter
    (fun v ->
      if v >= x_min && v <= x_max then (
        total_count := !total_count +. 1.;
        let bin_idx = ref (-1) in
        for i = 0 to num_bins - 1 do
          if
            v >= edges.(i)
            && (v < edges.(i + 1) || (i = num_bins - 1 && v <= edges.(i + 1)))
          then bin_idx := i
        done;
        if !bin_idx >= 0 then counts.(!bin_idx) <- counts.(!bin_idx) +. 1.0))
    x;

  let bar_heights, _bar_widths, bar_centers =
    let heights = Array.make num_bins 0. in
    let widths = Array.make num_bins 0. in
    let centers = Array.make num_bins 0. in
    for i = 0 to num_bins - 1 do
      let width = edges.(i + 1) -. edges.(i) in
      widths.(i) <- width;
      centers.(i) <- edges.(i) +. (width /. 2.0);
      heights.(i) <-
        (if density then
           if width > 1e-9 && !total_count > 0. then
             counts.(i) /. !total_count /. width
           else 0.
         else counts.(i))
    done;
    (heights, widths, centers)
  in

  let height = Ndarray.create Ndarray.float32 [| num_bins |] bar_heights in
  let bar_centers = Ndarray.create Ndarray.float32 [| num_bins |] bar_centers in

  let bar_artist = Artist.bar ?color ?label ~height bar_centers in
  let modified_ax = Axes.add_artist bar_artist ax in
  let current_ymin_opt, _ = modified_ax.Axes.ylim in
  if current_ymin_opt = None then Axes.set_ylim ~min:0.0 modified_ax
  else modified_ax

let imshow ?cmap ?aspect ?extent ~data ax =
  let artist = Artist.image ?extent ?cmap ?aspect data in
  Axes.add_artist artist ax

let text ?color ?fontsize ~x ~y text_content ax =
  let artist = Artist.text ?color ?fontsize ~x ~y text_content in
  Axes.add_artist artist ax

let errorbar ?yerr ?xerr ?(ecolor = Artist.Color.black) ?(elinewidth = 1.0)
    ?(capsize = 3.0) ?(fmt : Artist.plot_style option) ?label ~x ~y ax =
  let default_color = Artist.Color.blue in
  let default_linewidth = 1.5 in
  let default_linestyle = Artist.Solid in
  let default_marker = Artist.Circle in

  let final_color, final_linewidth, final_linestyle, final_marker =
    match fmt with
    | None ->
        (default_color, default_linewidth, default_linestyle, default_marker)
    | Some style ->
        ( Option.value style.fmt_color ~default:default_color,
          Option.value style.fmt_linewidth ~default:default_linewidth,
          Option.value style.fmt_linestyle ~default:default_linestyle,
          Option.value style.fmt_marker ~default:default_marker )
  in

  let base_line : Artist.line2d =
    Artist.
      {
        xdata = x;
        ydata = y;
        color = final_color;
        linewidth = final_linewidth;
        linestyle = final_linestyle;
        marker = final_marker;
        label;
      }
  in

  let error_style : Artist.errorbar_style =
    { yerr; xerr; color = ecolor; linewidth = elinewidth; capsize }
  in

  let n = Ndarray.size x in
  let validate_err name arr_opt =
    match arr_opt with
    | Some arr when Ndarray.size arr <> n ->
        invalid_arg
          (Printf.sprintf
             "Plotting.errorbar: %s length (%d) must match x/y data length (%d)"
             name (Ndarray.size arr) n)
    | _ -> ()
  in
  validate_err "yerr" yerr;
  validate_err "xerr" xerr;

  let artist = Artist.Errorbar (base_line, error_style) in
  Axes.add_artist artist ax

let step ?color ?linewidth ?linestyle ?where ?label ~x ~y ax =
  let artist = Artist.step ?color ?linewidth ?linestyle ?where ?label x y in
  Axes.add_artist artist ax

let fill_between ?color ?where ?interpolate ?label ~x ~y1 ~y2 ax =
  let artist = Artist.fill_between ?color ?where ?interpolate ?label x y1 y2 in
  Axes.add_artist artist ax

let matshow ?cmap ?aspect ?extent ?(origin = `upper) ~data ax =
  let img_artist = Artist.image ?cmap ?aspect ?extent data in
  let _ = Axes.add_artist img_artist ax in

  let shape = Ndarray.shape data in
  let rows, cols =
    match shape with
    | [| r; c |] -> (float r, float c)
    | _ -> invalid_arg "matshow requires 2D data"
  in
  let default_extent = (-0.5, cols -. 0.5, rows -. 0.5, -0.5) in
  let final_extent = Option.value extent ~default:default_extent in
  let d0, d1, d2, d3 = final_extent in
  let _ = Axes.set_xlim ~min:d0 ~max:d1 ax in
  let ymin, ymax = if origin = `upper then (d2, d3) else (d3, d2) in
  let _ = Axes.set_ylim ~min:ymin ~max:ymax ax in

  let xticks = List.init (int_of_float cols) float_of_int in
  let yticks = List.init (int_of_float rows) float_of_int in
  let _ = Axes.set_xticks xticks ax in
  let _ = Axes.set_yticks yticks ax in

  ax
