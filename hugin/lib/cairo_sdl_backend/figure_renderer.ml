let render_figure cr target_width target_height (fig : Figure.t) =
  Cairo.save cr;
  Cairo.set_antialias cr ANTIALIAS_DEFAULT;
  Cairo.set_line_cap cr ROUND;
  Cairo.set_line_join cr JOIN_ROUND;
  Render_utils.set_source_color cr fig.facecolor;
  Cairo.paint cr;
  List.iter
    (Axes_renderer.render_axes cr target_width target_height)
    (List.rev fig.axes);
  Cairo.restore cr
