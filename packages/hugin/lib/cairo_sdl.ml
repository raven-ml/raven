(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Cairo-SDL integration: shared ARGB8888 surface *)

type t = {
  window : Usdl.Window.t;
  renderer : Usdl.Renderer.t;
  mutable surface : Usdl.Surface.t;
  mutable cairo_surface : Ucairo.surface;
  mutable context : Ucairo.t;
  mutable width : int;
  mutable height : int;
}

let make_cairo_context surface =
  let pixels = Usdl.Surface.pixels surface in
  let stride = Usdl.Surface.pitch surface in
  let total = Bigarray.Array1.dim pixels in
  let h = total / stride in
  let w = stride / 4 in
  let cs = Ucairo.Image.create_for_data8 pixels ~w ~h ~stride in
  let cr = Ucairo.create cs in
  (cs, cr, w, h)

let create ~width ~height ~title =
  Usdl.init ();
  let window = Usdl.Window.create ~title ~w:width ~h:height in
  let renderer = Usdl.Renderer.create window in
  let ow, oh = Usdl.Renderer.output_size renderer in
  let surface = Usdl.Surface.create_argb8888 ~w:ow ~h:oh in
  let cairo_surface, context, w, h = make_cairo_context surface in
  { window; renderer; surface; cairo_surface; context; width = w; height = h }

let context t = t.context
let width t = t.width
let height t = t.height

let present t =
  Ucairo.Surface.flush t.cairo_surface;
  let tex = Usdl.Texture.of_surface t.renderer t.surface in
  Usdl.Renderer.clear t.renderer;
  Usdl.Renderer.copy t.renderer tex;
  Usdl.Renderer.present t.renderer;
  Usdl.Texture.destroy tex

let resize t =
  let nw, nh = Usdl.Renderer.output_size t.renderer in
  if nw <> t.width || nh <> t.height then
    begin if nw > 0 && nh > 0 then begin
      Ucairo.Surface.finish t.cairo_surface;
      Usdl.Surface.destroy t.surface;
      let surface = Usdl.Surface.create_argb8888 ~w:nw ~h:nh in
      let cairo_surface, context, w, h = make_cairo_context surface in
      t.surface <- surface;
      t.cairo_surface <- cairo_surface;
      t.context <- context;
      t.width <- w;
      t.height <- h
    end
    end

let destroy t =
  Ucairo.Surface.finish t.cairo_surface;
  Usdl.Surface.destroy t.surface;
  Usdl.Renderer.destroy t.renderer;
  Usdl.Window.destroy t.window;
  Usdl.quit ()
