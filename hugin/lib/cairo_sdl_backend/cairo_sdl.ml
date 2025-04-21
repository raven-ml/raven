open Usdl

type t = {
  window : Window.t;
  renderer : Renderer.t;
  mutable sdl_surface : Surface.t;
  mutable cairo_surface : Cairo.Surface.t;
  mutable context : Cairo.context;
  mutable width : int;
  mutable height : int;
}

let create ~width ~height ~title =
  match Usdl.init Init.video with
  | Error e ->
      Printf.eprintf "SDL init failed: %s\n%!" e;
      exit 1
  | Ok () -> (
      if not (Usdl.set_hint Hint.render_scale_quality "1") then
        Printf.printf "Warning: Failed to set render scale quality hint.\n%!";

      let window_flags =
        Window_flags.windowed lor Window_flags.resizable
        lor Window_flags.allow_highdpi
      in
      match Usdl.create_window ~title ~w:width ~h:height window_flags with
      | Error e ->
          Printf.eprintf "Window creation failed: %s\n%!" e;
          Usdl.quit ();
          exit 1
      | Ok window -> (
          let renderer_flags =
            Renderer_flags.accelerated lor Renderer_flags.presentvsync
          in
          match Usdl.create_renderer ~flags:renderer_flags window with
          | Error e ->
              Printf.eprintf "Renderer creation failed: %s\n%!" e;
              Usdl.destroy_window window;
              Usdl.quit ();
              exit 1
          | Ok renderer -> (
              match Usdl.get_renderer_output_size renderer with
              | Error e ->
                  Printf.eprintf "Failed to get renderer output size: %s\n%!" e;
                  Usdl.destroy_renderer renderer;
                  Usdl.destroy_window window;
                  Usdl.quit ();
                  exit 1
              | Ok (actual_width, actual_height) -> (
                  Printf.printf
                    "Window Size: %d x %d, Drawable Size: %d x %d\n%!" width
                    height actual_width actual_height;

                  let fmt = Pixel.format_argb8888 in
                  match
                    Usdl.create_rgb_surface_with_format ~w:actual_width
                      ~h:actual_height ~depth:32 fmt
                  with
                  | Error e ->
                      Printf.eprintf "SDL surface creation failed: %s\n%!" e;
                      Usdl.destroy_renderer renderer;
                      Usdl.destroy_window window;
                      Usdl.quit ();
                      exit 1
                  | Ok sdl_surface ->
                      let pixels = Usdl.get_surface_pixels sdl_surface in
                      let stride = Usdl.get_surface_pitch sdl_surface in
                      let cairo_surface =
                        Cairo.Image.create_for_data8 pixels Cairo.Image.ARGB32
                          ~w:actual_width ~h:actual_height ~stride
                      in
                      let context = Cairo.create cairo_surface in
                      {
                        window;
                        renderer;
                        sdl_surface;
                        cairo_surface;
                        context;
                        width = actual_width;
                        height = actual_height;
                      }))))

let context t = t.context
let width t = t.width
let height t = t.height

let redraw t =
  Cairo.Surface.flush t.cairo_surface;
  Cairo.Surface.finish t.cairo_surface;

  match Usdl.create_texture_from_surface t.renderer t.sdl_surface with
  | Error e -> Printf.eprintf "Texture creation failed: %s\n%!" e
  | Ok texture ->
      (match Usdl.render_clear t.renderer with
      | Ok () -> ()
      | Error e -> Printf.eprintf "Render clear failed: %s\n%!" e);
      (match Usdl.render_copy t.renderer texture with
      | Ok () -> ()
      | Error e -> Printf.eprintf "Render copy failed: %s\n%!" e);
      Usdl.render_present t.renderer;
      Usdl.destroy_texture texture;

      let pixels = Usdl.get_surface_pixels t.sdl_surface in
      let stride = Usdl.get_surface_pitch t.sdl_surface in
      t.cairo_surface <-
        Cairo.Image.create_for_data8 pixels Cairo.Image.ARGB32 ~w:t.width
          ~h:t.height ~stride;
      t.context <- Cairo.create t.cairo_surface

let resize t =
  match Usdl.get_renderer_output_size t.renderer with
  | Error e ->
      Printf.eprintf "Failed to get renderer output size on resize: %s\n%!" e
  | Ok (new_width, new_height) ->
      if new_width = t.width && new_height = t.height then ()
      else if new_width <= 0 || new_height <= 0 then ()
      else (
        Printf.printf "Resizing surface to %d x %d\n%!" new_width new_height;
        Cairo.Surface.finish t.cairo_surface;
        Usdl.free_surface t.sdl_surface;

        let fmt = Pixel.format_argb8888 in
        match
          Usdl.create_rgb_surface_with_format ~w:new_width ~h:new_height
            ~depth:32 fmt
        with
        | Error e -> Printf.eprintf "SDL surface recreate failed: %s\n%!" e
        | Ok new_sdl_surface ->
            let pixels = Usdl.get_surface_pixels new_sdl_surface in
            let stride = Usdl.get_surface_pitch new_sdl_surface in
            let new_cairo_surface =
              Cairo.Image.create_for_data8 pixels Cairo.Image.ARGB32
                ~w:new_width ~h:new_height ~stride
            in
            let new_context = Cairo.create new_cairo_surface in

            (* Update state *)
            t.sdl_surface <- new_sdl_surface;
            t.cairo_surface <- new_cairo_surface;
            t.context <- new_context;
            t.width <- new_width;
            t.height <- new_height)

let cleanup t =
  Cairo.Surface.finish t.cairo_surface;
  Usdl.free_surface t.sdl_surface;
  Usdl.destroy_renderer t.renderer;
  Usdl.destroy_window t.window;
  Usdl.quit ()
