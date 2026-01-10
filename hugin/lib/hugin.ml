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

let scatter ?title ?xlabel ?ylabel ?s ?c ?marker ?label x y =
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in
  let ax = Plotting.scatter ?s ?c ?marker ?label ~x ~y ax in
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

let render ?(format = "png") fig =
  let canvas =
    Cairo_sdl_backend.create_canvas ~width:fig.Figure.width
      ~height:fig.Figure.height ()
  in
  Cairo_sdl_backend.render fig canvas;
  Cairo_sdl_backend.save_to_buffer ~format fig
