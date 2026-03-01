(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Artist = Artist
module Axes = Axes
module Figure = Figure
module Plotting = Plotting

type figure = Figure.t
type axes = Axes.t

let apply_decorations ?title ?xlabel ?ylabel ?zlabel ax =
  let ax = match title with Some t -> Axes.set_title t ax | None -> ax in
  let ax = match xlabel with Some l -> Axes.set_xlabel l ax | None -> ax in
  let ax = match ylabel with Some l -> Axes.set_ylabel l ax | None -> ax in
  let ax = match zlabel with Some l -> Axes.set_zlabel l ax | None -> ax in
  ax

let plot ?title ?xlabel ?ylabel ?color ?linewidth ?linestyle ?marker ?label x y
    =
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  let ax = Plotting.plot ?color ?linewidth ?linestyle ?marker ?label ~x ~y ax in
  let _ = apply_decorations ?title ?xlabel ?ylabel ax in
  fig

let plot_y ?title ?xlabel ?ylabel ?color ?linewidth ?linestyle ?marker ?label
    y_data =
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  let ax =
    Plotting.plot_y ?color ?linewidth ?linestyle ?marker ?label ~y:y_data ax
  in
  let _ = apply_decorations ?title ?xlabel ?ylabel ax in
  fig

let plot3d ?title ?xlabel ?ylabel ?zlabel ?color ?linewidth ?linestyle ?marker
    ?label x y z =
  let fig = Figure.create () in
  let ax = Figure.add_subplot ~projection:ThreeD fig in
  let ax =
    Plotting.plot3d ?color ?linewidth ?linestyle ?marker ?label ~x ~y ~z ax
  in
  let _ = apply_decorations ?title ?xlabel ?ylabel ?zlabel ax in
  fig

let scatter ?title ?xlabel ?ylabel ?s ?s_data ?c ?marker ?label x y =
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  let ax = Plotting.scatter ?s ?s_data ?c ?marker ?label ~x ~y ax in
  let _ = apply_decorations ?title ?xlabel ?ylabel ax in
  fig

let hist ?bins ?range ?density ?title ?xlabel ?ylabel ?color ?label x_data =
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  let ax = Plotting.hist ?bins ?range ?density ?color ?label ~x:x_data ax in
  let _ = apply_decorations ?title ?xlabel ?ylabel ax in
  fig

let bar ?width ?bottom ?title ?xlabel ?ylabel ?color ?label ~height x =
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  let ax = Plotting.bar ?width ?bottom ?color ?label ~height ~x ax in
  let _ = apply_decorations ?title ?xlabel ?ylabel ax in
  fig

let imshow ?cmap ?aspect ?extent ?title ?xlabel ?ylabel data =
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  let ax = Plotting.imshow ?cmap ?aspect ?extent ~data ax in
  let _ = apply_decorations ?title ?xlabel ?ylabel ax in
  fig

let step ?title ?xlabel ?ylabel ?color ?linewidth ?linestyle ?where ?label x y =
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  let ax = Plotting.step ?color ?linewidth ?linestyle ?where ?label ~x ~y ax in
  let _ = apply_decorations ?title ?xlabel ?ylabel ax in
  fig

let fill_between ?title ?xlabel ?ylabel ?color ?where ?interpolate ?label x y1
    y2 =
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  let ax =
    Plotting.fill_between ?color ?where ?interpolate ?label ~x ~y1 ~y2 ax
  in
  let _ = apply_decorations ?title ?xlabel ?ylabel ax in
  fig

let errorbar ?title ?xlabel ?ylabel ?yerr ?xerr ?ecolor ?elinewidth ?capsize
    ?fmt (* This is now Artist.plot_style option *) ?label x y =
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  let ax =
    Plotting.errorbar ?yerr ?xerr ?ecolor ?elinewidth ?capsize ?fmt ?label ~x ~y
      ax (* Pass the plot_style fmt directly *)
  in
  let _ = apply_decorations ?title ?xlabel ?ylabel ax in
  fig

let matshow ?cmap ?aspect ?extent ?origin ?title data =
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  (* Apply decorations *before* calling matshow, as matshow might set specific
     labels/titles *)
  let ax_decorated = apply_decorations ?title ax in
  (* matshow sets its own x/y labels based on matrix indices, so don't pass
     xlabel/ylabel *)
  let _ = Plotting.matshow ?cmap ?aspect ?extent ?origin ~data ax_decorated in
  fig

(* Add this alias definition, likely near the `figure` and `subplot` aliases *)
let add_axes = Figure.add_axes
let figure = Figure.create
let subplot = Figure.add_subplot

let show fig =
  let canvas =
    Cairo_sdl_backend.create_canvas ~width:fig.Figure.width
      ~height:fig.Figure.height ()
  in
  Cairo_sdl_backend.render fig canvas;
  Cairo_sdl_backend.show canvas

let savefig ?dpi ?format filename fig =
  let canvas =
    Cairo_sdl_backend.create_canvas ~width:fig.Figure.width
      ~height:fig.Figure.height ()
  in
  Cairo_sdl_backend.render fig canvas;
  Cairo_sdl_backend.save ?dpi ?format fig filename canvas

let render ?(format = "png") fig = Cairo_sdl_backend.save_to_buffer ~format fig

let base64_encode_string input =
  let alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
  in
  let len = String.length input in
  let out_len = (len + 2) / 3 * 4 in
  let out = Bytes.create out_len in
  let rec loop i j =
    if i < len then begin
      let b0 = Char.code (String.unsafe_get input i) in
      let b1 =
        if i + 1 < len then Char.code (String.unsafe_get input (i + 1)) else 0
      in
      let b2 =
        if i + 2 < len then Char.code (String.unsafe_get input (i + 2)) else 0
      in
      Bytes.unsafe_set out j (String.unsafe_get alphabet (b0 lsr 2));
      Bytes.unsafe_set out (j + 1)
        (String.unsafe_get alphabet (((b0 land 3) lsl 4) lor (b1 lsr 4)));
      Bytes.unsafe_set out (j + 2)
        (if i + 1 < len then
           String.unsafe_get alphabet (((b1 land 0xf) lsl 2) lor (b2 lsr 6))
         else '=');
      Bytes.unsafe_set out (j + 3)
        (if i + 2 < len then String.unsafe_get alphabet (b2 land 0x3f) else '=');
      loop (i + 3) (j + 4)
    end
  in
  loop 0 0;
  Bytes.unsafe_to_string out

let pp_figure fmt fig =
  let image_data = render fig in
  let base64_data = base64_encode_string image_data in
  Format.fprintf fmt "![figure](data:image/png;base64,%s)" base64_data
