type t = {
  projection : projection;
  left : float;
  bottom : float;
  width : float;
  height : float;
  mutable title : string;
  mutable xlabel : string;
  mutable ylabel : string;
  mutable zlabel : string option;
  mutable artists : Artist.t list;
  mutable xlim : float option * float option;
  mutable ylim : float option * float option;
  mutable zlim : float option * float option;
  mutable xticks : float list option;
  mutable yticks : float list option;
  mutable zticks : float list option;
  mutable xscale : scale;
  mutable yscale : scale;
  mutable grid_visible : bool;
  mutable grid_which : [ `major | `minor | `both ];
  mutable grid_axis : [ `x | `y | `both ];
  mutable elev : float;
  mutable azim : float;
}

and projection = TwoD | ThreeD
and scale = Linear | Log

let create ?(projection = TwoD) ?(elev = 30.) ?(azim = -60.) ~left ~bottom
    ~width ~height () =
  {
    projection;
    left;
    bottom;
    width;
    height;
    title = "";
    xlabel = "";
    ylabel = "";
    zlabel = None;
    artists = [];
    xlim = (None, None);
    ylim = (None, None);
    zlim = (None, None);
    xticks = None;
    yticks = None;
    zticks = None;
    xscale = Linear;
    yscale = Linear;
    grid_visible = false;
    grid_which = `major;
    grid_axis = `both;
    elev = (if projection = ThreeD then elev else 30.);
    azim = (if projection = ThreeD then azim else -60.);
  }

let add_artist artist ax =
  ax.artists <- artist :: ax.artists;
  ax

let cla ax =
  ax.artists <- [];
  ax.title <- "";
  ax.xlabel <- "";
  ax.ylabel <- "";
  ax.zlabel <- None;
  ax.xlim <- (None, None);
  ax.ylim <- (None, None);
  ax.zlim <- (None, None);
  ax

let set_title title ax =
  ax.title <- title;
  ax

let set_xlabel label ax =
  ax.xlabel <- label;
  ax

let set_ylabel label ax =
  ax.ylabel <- label;
  ax

let set_zlabel label ax =
  if ax.projection = ThreeD then ax.zlabel <- Some label else ();
  ax

let set_xlim ?min ?max ax =
  let current_min, current_max = ax.xlim in
  let n_min = match min with Some v -> Some v | None -> current_min in
  let n_max = match max with Some v -> Some v | None -> current_max in
  ax.xlim <- (n_min, n_max);
  ax

let set_ylim ?min ?max ax =
  let current_min, current_max = ax.ylim in
  let n_min = match min with Some v -> Some v | None -> current_min in
  let n_max = match max with Some v -> Some v | None -> current_max in
  ax.ylim <- (n_min, n_max);
  ax

let set_zlim ?min ?max ax =
  if ax.projection = ThreeD then
    let current_min, current_max = ax.zlim in
    let n_min = match min with Some v -> Some v | None -> current_min in
    let n_max = match max with Some v -> Some v | None -> current_max in
    ax.zlim <- (n_min, n_max)
  else ();
  ax

let set_xscale scale ax =
  ax.xscale <- scale;
  ax

let set_yscale scale ax =
  ax.yscale <- scale;
  ax

let set_xticks ticks ax =
  ax.xticks <- Some ticks;
  ax

let set_yticks ticks ax =
  ax.yticks <- Some ticks;
  ax

let set_zticks ticks ax =
  if ax.projection = ThreeD then ax.zticks <- Some ticks else ();
  ax

let set_elev elev ax =
  if ax.projection = ThreeD then ax.elev <- elev else ();
  ax

let set_azim azim ax =
  if ax.projection = ThreeD then ax.azim <- azim else ();
  ax

let grid ?(which = `major) ?(axis = `both) visible ax =
  ax.grid_visible <- visible;
  ax.grid_which <- which;
  ax.grid_axis <- axis;
  ax

let calculate_data_bounds (ax : t) : (float * float * float * float) option =
  let min_x = ref Float.infinity in
  let max_x = ref Float.neg_infinity in
  let min_y = ref Float.infinity in
  let max_y = ref Float.neg_infinity in
  let has_finite_data = ref false in

  let update_bounds x y =
    if Float.is_finite x && Float.is_finite y then (
      has_finite_data := true;
      min_x := min !min_x x;
      max_x := max !max_x x;
      min_y := min !min_y y;
      max_y := max !max_y y)
  in

  let array_finite_bounds arr =
    let current_min = ref Float.infinity in
    let current_max = ref Float.neg_infinity in
    let found_finite = ref false in
    Nx.iter_item
      (fun v ->
        if Float.is_finite v then (
          found_finite := true;
          current_min := min !current_min v;
          current_max := max !current_max v))
      arr;
    if !found_finite then Some (!current_min, !current_max) else None
  in

  let update_bounds_from_arrays arr_x arr_y =
    match (array_finite_bounds arr_x, array_finite_bounds arr_y) with
    | Some (x_min, x_max), Some (y_min, y_max) ->
        has_finite_data := true;
        min_x := min !min_x x_min;
        max_x := max !max_x x_max;
        min_y := min !min_y y_min;
        max_y := max !max_y y_max
    | _ -> ()
  in

  List.iter
    (fun artist ->
      match artist with
      | Artist.Line2D l -> update_bounds_from_arrays l.xdata l.ydata
      | Artist.Line3D l -> update_bounds_from_arrays l.xdata l.ydata
      | Artist.Scatter s -> update_bounds_from_arrays s.xdata s.ydata
      | Artist.Bar b ->
          let n = Nx.size b.x in
          if n > 0 && n = Nx.size b.height then
            for i = 0 to n - 1 do
              let x_center = Nx.get_item [ i ] b.x in
              let height = Nx.get_item [ i ] b.height in
              let x_left = x_center -. (b.width /. 2.0) in
              let x_right = x_center +. (b.width /. 2.0) in
              let y_bottom = b.bottom in
              let y_top = y_bottom +. height in
              update_bounds x_left y_bottom;
              update_bounds x_right y_top
            done
      | Artist.Image img -> (
          match img.extent with
          | Some (l, r, b, t) ->
              update_bounds l b;
              update_bounds r t
          | None ->
              let rows, cols =
                match img.shape with
                | [| r; c |] -> (float r, float c)
                | [| r; c; _ |] -> (float r, float c)
                | _ -> (0., 0.)
              in
              if rows > 0. && cols > 0. then (
                update_bounds (-0.5) (-0.5);
                update_bounds (cols -. 0.5) (rows -. 0.5)))
      | Artist.Errorbar (line, style) -> (
          update_bounds_from_arrays line.xdata line.ydata;
          let n = Nx.size line.xdata in
          (match style.yerr with
          | Some yerr when Nx.size yerr = n ->
              for i = 0 to n - 1 do
                let x = Nx.get_item [ i ] line.xdata in
                let y = Nx.get_item [ i ] line.ydata in
                let dy = Nx.get_item [ i ] yerr in
                if Float.is_finite x && Float.is_finite y && Float.is_finite dy
                then (
                  update_bounds x (y -. dy);
                  update_bounds x (y +. dy))
              done
          | _ -> ());
          match style.xerr with
          | Some xerr when Nx.size xerr = n ->
              for i = 0 to n - 1 do
                let x = Nx.get_item [ i ] line.xdata in
                let y = Nx.get_item [ i ] line.ydata in
                let dx = Nx.get_item [ i ] xerr in
                if Float.is_finite x && Float.is_finite y && Float.is_finite dx
                then (
                  update_bounds (x -. dx) y;
                  update_bounds (x +. dx) y)
              done
          | _ -> ())
      | Artist.Step st -> update_bounds_from_arrays st.xdata st.ydata
      | Artist.FillBetween fb ->
          update_bounds_from_arrays fb.xdata fb.y1data;
          update_bounds_from_arrays fb.xdata fb.y2data
      | Artist.Text _ -> ())
    ax.artists;

  if !has_finite_data then Some (!min_x, !max_x, !min_y, !max_y) else None

let get_final_data_bounds ?(xpad = 0.05) ?(ypad = 0.05) (ax : t) :
    (float * float) * (float * float) =
  let calculated_bounds = calculate_data_bounds ax in

  let default_xmin, default_xmax, default_ymin, default_ymax =
    (0.0, 1.0, 0.0, 1.0)
  in

  let c_xmin, c_xmax, c_ymin, c_ymax =
    match calculated_bounds with
    | Some bounds -> bounds
    | None -> (default_xmin, default_xmax, default_ymin, default_ymax)
  in

  let user_xmin_opt, user_xmax_opt = ax.xlim in
  let user_ymin_opt, user_ymax_opt = ax.ylim in

  let final_xmin =
    match user_xmin_opt with
    | Some x -> x
    | None ->
        if calculated_bounds = None then default_xmin
        else
          let dx = (c_xmax -. c_xmin) *. xpad in
          if dx = 0.0 then c_xmin -. 0.5 else c_xmin -. dx
  in
  let final_xmax =
    match user_xmax_opt with
    | Some x -> x
    | None ->
        if calculated_bounds = None then default_xmax
        else
          let dx = (c_xmax -. c_xmin) *. xpad in
          if dx = 0.0 then c_xmax +. 0.5 else c_xmax +. dx
  in

  let final_ymin =
    match user_ymin_opt with
    | Some y -> y
    | None ->
        if calculated_bounds = None then default_ymin
        else
          let dy = (c_ymax -. c_ymin) *. ypad in
          if dy = 0.0 then c_ymin -. 0.5 else c_ymin -. dy
  in
  let final_ymax =
    match user_ymax_opt with
    | Some y -> y
    | None ->
        if calculated_bounds = None then default_ymax
        else
          let dy = (c_ymax -. c_ymin) *. ypad in
          if dy = 0.0 then c_ymax +. 0.5 else c_ymax +. dy
  in

  let final_xmin, final_xmax =
    if final_xmin > final_xmax then (default_xmin, default_xmax)
    else (final_xmin, final_xmax)
  in
  let final_ymin, final_ymax =
    if final_ymin > final_ymax then (default_ymin, default_ymax)
    else (final_ymin, final_ymax)
  in

  ((final_xmin, final_xmax), (final_ymin, final_ymax))

let calculate_z_bounds (ax : t) : (float * float) option =
  if ax.projection <> ThreeD then None
  else
    let min_z = ref Float.infinity in
    let max_z = ref Float.neg_infinity in
    let has_finite_z = ref false in

    let update_z_bounds z =
      if Float.is_finite z then (
        has_finite_z := true;
        min_z := min !min_z z;
        max_z := max !max_z z)
    in

    let array_finite_z_bounds arr =
      let current_min = ref Float.infinity in
      let current_max = ref Float.neg_infinity in
      let found_finite = ref false in
      Nx.iter_item
        (fun v ->
          if Float.is_finite v then (
            found_finite := true;
            current_min := min !current_min v;
            current_max := max !current_max v))
        arr;
      if !found_finite then Some (!current_min, !current_max) else None
    in

    List.iter
      (fun artist ->
        match artist with
        | Artist.Line3D l -> (
            match array_finite_z_bounds l.zdata with
            | Some (z_min, z_max) ->
                update_z_bounds z_min;
                update_z_bounds z_max
            | None -> ())
        | _ -> ())
      ax.artists;

    if !has_finite_z then Some (!min_z, !max_z) else None

let get_final_z_bounds ?(zpad = 0.05) (ax : t) : float * float =
  if ax.projection <> ThreeD then (0.0, 1.0)
  else
    let calculated_z_bounds = calculate_z_bounds ax in
    let default_zmin, default_zmax = (0.0, 1.0) in

    let c_zmin, c_zmax =
      match calculated_z_bounds with
      | Some bounds -> bounds
      | None -> (default_zmin, default_zmax)
    in

    let user_zmin_opt, user_zmax_opt = ax.zlim in

    let final_zmin =
      match user_zmin_opt with
      | Some z -> z
      | None ->
          if calculated_z_bounds = None then default_zmin
          else
            let dz = (c_zmax -. c_zmin) *. zpad in
            if dz = 0.0 then c_zmin -. 0.5 else c_zmin -. dz
    in
    let final_zmax =
      match user_zmax_opt with
      | Some z -> z
      | None ->
          if calculated_z_bounds = None then default_zmax
          else
            let dz = (c_zmax -. c_zmin) *. zpad in
            if dz = 0.0 then c_zmax +. 0.5 else c_zmax +. dz
    in

    if final_zmin > final_zmax then (default_zmin, default_zmax)
    else (final_zmin, final_zmax)
