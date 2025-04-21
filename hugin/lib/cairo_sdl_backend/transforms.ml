module Matrix = struct
  type t = float array array

  let make_ident () : t =
    Array.init 4 (fun r -> Array.init 4 (fun c -> if r = c then 1. else 0.))

  let mat_mul (a : t) (b : t) : t =
    let res = Array.make_matrix 4 4 0. in
    for r = 0 to 3 do
      for c = 0 to 3 do
        for k = 0 to 3 do
          res.(r).(c) <- res.(r).(c) +. (a.(r).(k) *. b.(k).(c))
        done
      done
    done;
    res

  let transform_vec (m : t) (v : float * float * float * float) :
      float * float * float * float =
    let x, y, z, w = v in
    let x' =
      (m.(0).(0) *. x)
      +. (m.(0).(1) *. y)
      +. (m.(0).(2) *. z)
      +. (m.(0).(3) *. w)
    in
    let y' =
      (m.(1).(0) *. x)
      +. (m.(1).(1) *. y)
      +. (m.(1).(2) *. z)
      +. (m.(1).(3) *. w)
    in
    let z' =
      (m.(2).(0) *. x)
      +. (m.(2).(1) *. y)
      +. (m.(2).(2) *. z)
      +. (m.(2).(3) *. w)
    in
    let w' =
      (m.(3).(0) *. x)
      +. (m.(3).(1) *. y)
      +. (m.(3).(2) *. z)
      +. (m.(3).(3) *. w)
    in
    (x', y', z', w')

  let make_translation tx ty tz : t =
    let m = make_ident () in
    m.(0).(3) <- tx;
    m.(1).(3) <- ty;
    m.(2).(3) <- tz;
    m

  let make_scale sx sy sz : t =
    let m = make_ident () in
    m.(0).(0) <- sx;
    m.(1).(1) <- sy;
    m.(2).(2) <- sz;
    m

  let make_rotation_x angle_rad : t =
    let m = make_ident () in
    let c = cos angle_rad in
    let s = sin angle_rad in
    m.(1).(1) <- c;
    m.(1).(2) <- -.s;
    m.(2).(1) <- s;
    m.(2).(2) <- c;
    m

  let make_rotation_y angle_rad : t =
    let m = make_ident () in
    let c = cos angle_rad in
    let s = sin angle_rad in
    m.(0).(0) <- c;
    m.(0).(2) <- s;
    m.(2).(0) <- -.s;
    m.(2).(2) <- c;
    m

  let make_rotation_z angle_rad : t =
    let m = make_ident () in
    let c = cos angle_rad in
    let s = sin angle_rad in
    m.(0).(0) <- c;
    m.(0).(1) <- -.s;
    m.(1).(0) <- s;
    m.(1).(1) <- c;
    m

  let make_perspective ~fovy_rad ~aspect ~znear ~zfar : t =
    let m = Array.make_matrix 4 4 0. in
    if fovy_rad <= 0. || aspect <= 0. || znear <= 0. || zfar <= znear then
      failwith "Invalid perspective parameters";
    let f = 1.0 /. tan (fovy_rad /. 2.0) in
    m.(0).(0) <- f /. aspect;
    m.(1).(1) <- f;
    m.(2).(2) <- (zfar +. znear) /. (znear -. zfar);
    m.(2).(3) <- 2.0 *. zfar *. znear /. (znear -. zfar);
    m.(3).(2) <- -1.0;
    m
end

type data_limits = { xmin : float; xmax : float; ymin : float; ymax : float }
type pixel_bounds = { left : float; top : float; width : float; height : float }

type context2d = {
  data_lims : data_limits;
  pixel_bounds : pixel_bounds;
  x_log : bool;
  y_log : bool;
  y_inverted : bool;
}

type context3d = {
  xmin : float;
  xmax : float;
  ymin : float;
  ymax : float;
  zmin : float;
  zmax : float;
  pixel_bounds : pixel_bounds;
  mvp_matrix : Matrix.t;
  view_matrix : Matrix.t;
  model_matrix : Matrix.t;
  znear : float;
}

type context = Ctx2D of context2d | Ctx3D of context3d

let create_transform_context data_lims pixel_bounds x_scale y_scale =
  {
    data_lims;
    pixel_bounds;
    x_log = (match x_scale with Axes.Log -> true | _ -> false);
    y_log = (match y_scale with Axes.Log -> true | _ -> false);
    y_inverted = true;
  }

let transform (ctx : context2d) ~x ~y =
  let { xmin; xmax; ymin; ymax } : data_limits = ctx.data_lims in
  let { left; top; width; height } = ctx.pixel_bounds in

  let pixel_x =
    if ctx.x_log then
      if x <= 0. || xmin <= 0. || xmax <= 0. then Float.nan
      else
        let log_min = log10 xmin in
        let log_max = log10 xmax in
        let log_range = log_max -. log_min in
        if Float.equal log_range 0. then left +. (width /. 2.)
        else left +. ((log10 x -. log_min) /. log_range *. width)
    else
      let range_x = xmax -. xmin in
      if Float.equal range_x 0. then left +. (width /. 2.)
      else left +. ((x -. xmin) /. range_x *. width)
  in

  let y_norm =
    if ctx.y_log then
      if y <= 0. || ymin <= 0. || ymax <= 0. then Float.nan
      else
        let log_min = log10 ymin in
        let log_max = log10 ymax in
        let log_range = log_max -. log_min in
        if Float.equal log_range 0. then 0.5
        else (log10 y -. log_min) /. log_range
    else
      let y_data_min = Float.min ymin ymax in
      let y_data_max = Float.max ymin ymax in
      let range_y = y_data_max -. y_data_min in
      if Float.equal range_y 0. then 0.5
      else
        let norm = (y -. y_data_min) /. range_y in
        if ymax < ymin then 1.0 -. norm else norm
  in

  let pixel_y =
    if ctx.y_inverted then top +. ((1.0 -. y_norm) *. height)
    else top +. (y_norm *. height)
  in

  (pixel_x, pixel_y)

let create_transform_context3d ~xmin ~xmax ~ymin ~ymax ~zmin ~zmax ~elev ~azim
    pixel_bounds =
  let deg_to_rad d = d *. Float.pi /. 180. in
  let elev_rad = deg_to_rad elev in
  let azim_rad = deg_to_rad azim in

  let xrange = xmax -. xmin in
  let yrange = ymax -. ymin in
  let zrange = zmax -. zmin in
  let max_range = Float.max (Float.max xrange yrange) zrange in
  let scale_factor =
    if max_range > 0. && Float.is_finite max_range then 2.0 /. max_range
    else 1.0
  in

  let center_x = (xmin +. xmax) /. 2.0 in
  let center_y = (ymin +. ymax) /. 2.0 in
  let center_z = (zmin +. zmax) /. 2.0 in
  let model_trans_mat =
    Matrix.make_translation (-.center_x) (-.center_y) (-.center_z)
  in
  let model_scale_mat =
    Matrix.make_scale scale_factor scale_factor scale_factor
  in
  let model_matrix = Matrix.mat_mul model_scale_mat model_trans_mat in
  let camera_dist = 3.0 in
  let view_trans_mat = Matrix.make_translation 0. 0. (-.camera_dist) in
  let rot_y = Matrix.make_rotation_y azim_rad in
  let rot_x = Matrix.make_rotation_x elev_rad in
  let view_rot_mat = Matrix.mat_mul rot_x rot_y in
  let view_matrix = Matrix.mat_mul view_trans_mat view_rot_mat in
  let aspect =
    if pixel_bounds.height > 0. then pixel_bounds.width /. pixel_bounds.height
    else 1.0
  in
  let fovy_rad = deg_to_rad 45.0 in
  let znear = 0.1 in
  let zfar = camera_dist +. 4.0 in
  let proj_mat = Matrix.make_perspective ~fovy_rad ~aspect ~znear ~zfar in
  let mv_matrix = Matrix.mat_mul view_matrix model_matrix in
  let mvp_matrix = Matrix.mat_mul proj_mat mv_matrix in

  {
    xmin;
    xmax;
    ymin;
    ymax;
    zmin;
    zmax;
    pixel_bounds;
    mvp_matrix;
    view_matrix;
    model_matrix;
    znear;
  }

let transform3d (ctx : context3d) ~x ~y ~z : (float * float) option =
  let x_model, y_model, z_model, w_model =
    Matrix.transform_vec ctx.model_matrix (x, y, z, 1.0)
  in
  let _x_view, _y_view, z_view, _w_view =
    Matrix.transform_vec ctx.view_matrix (x_model, y_model, z_model, w_model)
  in

  if z_view > -.ctx.znear then None
  else
    let x_clip, y_clip, z_clip, w_clip =
      Matrix.transform_vec ctx.mvp_matrix (x, y, z, 1.0)
    in

    if w_clip <= 1e-6 then None
    else
      let ndc_x = x_clip /. w_clip in
      let ndc_y = y_clip /. w_clip in
      let ndc_z = z_clip /. w_clip in

      if
        ndc_x < -1.0 || ndc_x > 1.0 || ndc_y < -1.0 || ndc_y > 1.0
        || ndc_z < -1.0 || ndc_z > 1.0
      then None
      else
        let px =
          ctx.pixel_bounds.left
          +. ((ndc_x +. 1.0) *. 0.5 *. ctx.pixel_bounds.width)
        in
        let py =
          ctx.pixel_bounds.top
          +. ((1.0 -. ((ndc_y +. 1.0) *. 0.5)) *. ctx.pixel_bounds.height)
        in
        Some (px, py)
