(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Usdl

type canvas = { csdl : Cairo_sdl.t; mutable current_figure : Figure.t option }

let create_canvas ~width ~height () =
  let csdl = Cairo_sdl.create ~width ~height ~title:"Hugin Plot" in
  { csdl; current_figure = None }

let render figure canvas =
  canvas.current_figure <- Some figure;
  let cr = Cairo_sdl.context canvas.csdl in
  let target_width = Cairo_sdl.width canvas.csdl in
  let target_height = Cairo_sdl.height canvas.csdl in
  Cairo.set_source_rgb cr 1.0 1.0 1.0;
  Cairo.paint cr;
  Figure_renderer.render_figure cr (float target_width) (float target_height)
    figure;
  Cairo_sdl.redraw canvas.csdl

let show canvas =
  match canvas.current_figure with
  | None -> Printf.eprintf "No figure has been rendered to show.\n%!"
  | Some figure_to_show ->
      let event = Usdl.create_event () in
      let quit = ref false in

      render figure_to_show canvas;

      while not !quit do
        match Usdl.wait_event (Some event) with
        | Error msg ->
            Printf.eprintf "SDL_WaitEvent error: %s\n%!" msg;
            quit := true
        | Ok true -> (
            match Usdl.get_event_type event with
            | `Quit -> quit := true
            | `Window_event -> (
                match Usdl.get_window_event_id event with
                | `Resized | `Size_changed ->
                    Printf.printf "Window resized/size_changed event\n%!";
                    Cairo_sdl.resize canvas.csdl;
                    render figure_to_show canvas
                | `Exposed ->
                    Printf.printf "Window exposed event\n%!";
                    render figure_to_show canvas
                | `Close -> quit := true
                | _ -> ())
            | `Key_down ->
                let keycode = Usdl.get_event_keycode event in
                if keycode = Usdl.Keycode.escape || keycode = Usdl.Keycode.q
                then quit := true
            | _ -> ())
        | Ok false -> quit := true
      done;
      Cairo_sdl.cleanup canvas.csdl;
      canvas.current_figure <- None

let save ?dpi ?(format = "png") figure filename _canvas =
  let _ = dpi in
  let target_width = figure.Figure.width in
  let target_height = figure.Figure.height in

  let surface = Cairo.Image.(create ARGB32 ~w:target_width ~h:target_height) in
  let cr = Cairo.create surface in

  Cairo.set_source_rgb cr 1.0 1.0 1.0;
  Cairo.paint cr;

  Figure_renderer.render_figure cr (float target_width) (float target_height)
    figure;

  (match String.lowercase_ascii format with
  | "png" -> Cairo.PNG.write surface filename
  | _ ->
      Printf.eprintf
        "Warning: Saving format '%s' not supported. Only PNG is implemented.\n\
         %!"
        format);

  Cairo.Surface.finish surface

let save_to_buffer ?(format = "png") figure =
  if format <> "png" then failwith "Only PNG format is supported";
  let target_width = figure.Figure.width in
  let target_height = figure.Figure.height in
  let surface = Cairo.Image.(create ARGB32 ~w:target_width ~h:target_height) in
  let cr = Cairo.create surface in
  Cairo.set_source_rgb cr 1.0 1.0 1.0;
  Cairo.paint cr;
  Figure_renderer.render_figure cr (float target_width) (float target_height)
    figure;
  let buf = Buffer.create 1024 in
  let write_func s = Buffer.add_string buf s in
  (match String.lowercase_ascii format with
  | "png" -> Cairo.PNG.write_to_stream surface write_func
  | _ ->
      Printf.eprintf
        "Warning: Saving format '%s' not supported. Only PNG is implemented.\n\
         %!"
        format);
  Cairo.Surface.finish surface;
  Buffer.contents buf
