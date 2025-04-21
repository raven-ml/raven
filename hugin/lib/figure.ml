type t = {
  width : int;
  height : int;
  mutable axes : Axes.t list;
  facecolor : Artist.color;
}

let create ?(width = 640) ?(height = 480) () =
  { width; height; axes = []; facecolor = Artist.Color.white }

let add_axes ~left ~bottom ~width ~height ?(projection = Axes.TwoD) fig =
  let ax = Axes.create ~projection ~left ~bottom ~width ~height () in
  fig.axes <- ax :: fig.axes;
  ax

let add_subplot ?(nrows = 1) ?(ncols = 1) ?(index = 1) ?(projection = Axes.TwoD)
    fig =
  if index < 1 || index > nrows * ncols then
    invalid_arg "Figure.add_subplot: index out of range";
  if nrows < 1 || ncols < 1 then
    invalid_arg "Figure.add_subplot: nrows and ncols must be >= 1";

  let num = float_of_int index in
  let rows = float_of_int nrows in
  let cols = float_of_int ncols in

  let row = ceil (num /. cols) -. 1. in
  let col = float_of_int ((index - 1) mod ncols) in

  let h_pad_total = 0.15 in
  let v_pad_total = 0.15 in
  let h_space = 0.05 in
  let v_space = 0.05 in

  let bottom_margin = v_pad_total /. 2. in
  let top_margin = v_pad_total /. 2. in
  let left_margin = h_pad_total /. 2. in
  let right_margin = h_pad_total /. 2. in

  let plot_area_h = 1.0 -. left_margin -. right_margin in
  let plot_area_v = 1.0 -. bottom_margin -. top_margin in

  let cell_width = (plot_area_h -. (max 0. (cols -. 1.) *. h_space)) /. cols in
  let cell_height = (plot_area_v -. (max 0. (rows -. 1.) *. v_space)) /. rows in

  let sub_left = left_margin +. (col *. (cell_width +. h_space)) in
  let sub_bottom =
    bottom_margin +. ((rows -. 1. -. row) *. (cell_height +. v_space))
  in
  let sub_width = cell_width in
  let sub_height = cell_height in

  let clip v = max 0.0 (min 1.0 v) in
  add_axes fig ~left:(clip sub_left) ~bottom:(clip sub_bottom)
    ~width:(clip sub_width) ~height:(clip sub_height) ~projection

let clf fig =
  fig.axes <- [];
  fig
