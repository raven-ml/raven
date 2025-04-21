(** hugin: Functional Plotting and Visualization Library for OCaml

    Provides composable artists, axes, figures, and high-level plotting APIs for
    creating scientific and statistical graphics using Ndarray. *)

(** Artist: Visual element constructors and style specifications. *)
module Artist : sig
  type color = { r : float; g : float; b : float; a : float }
  (** [color] is an RGBA color record with components in [0.0,1.0]. *)

  type cmap
  (** [cmap] is colormap mapping normalized [0.0,1.0] to RGBA colors. *)

  (** Predefined colors. *)
  module Color : sig
    val blue : color
    val green : color
    val red : color
    val cyan : color
    val magenta : color
    val yellow : color
    val black : color
    val white : color
    val lightgray : color
    val gray : color
    val darkgray : color
    val orange : color
  end

  module Colormap : sig
    val viridis : cmap
    val plasma : cmap
    val inferno : cmap
    val magma : cmap
    val cividis : cmap
    val coolwarm : cmap
    val gray : cmap
  end

  (**Patterns for stroked lines. *)
  type line_style =
    | Solid  (** continuous line .*)
    | Dashed  (** repeating dash pattern .*)
    | DashDot  (** dash-dot sequence .*)
    | Dotted  (** dotted line .*)
    | None  (** no line. *)

  (** Shapes for data point markers. *)
  type marker_style =
    | Circle  (** circular markers. *)
    | Point  (** single-pixel points. *)
    | Pixel  (** same as [Point]. *)
    | Square  (** square markers. *)
    | Triangle  (** triangular markers. *)
    | Plus  (** plus-sign markers. *)
    | Star  (** star-shaped markers. *)
    | None  (** no markers. *)

  type plot_style
  (** Abstract type representing the style (color, marker, line) for a plot
      element. Used by functions like [Plotting.errorbar] to style the central
      data points. *)

  val plot_style :
    ?color:color ->
    ?linewidth:float ->
    ?linestyle:line_style ->
    ?marker:marker_style ->
    unit ->
    plot_style
  (** [plot_style ?color ?linewidth ?linestyle ?marker ()]

      Create a reusable style for line and marker artists.

      {2 Parameters}
      - [color]: stroke/marker color.
      - [linewidth]: width of stroked lines.
      - [linestyle]: pattern for stroked lines.
      - [marker]: shape for data point markers.

      {2 Returns}
      - a [plot_style] object to style artists like errorbar, hist, etc. *)

  (** Position of step relative to the x-coordinate in step plots. *)
  type step_where =
    | Pre  (** step change at the x coordinate before drawing segment. *)
    | Post  (** step change at the x coordinate after drawing segment. *)
    | Mid  (** step change at midpoint between coordinates. *)

  type t
  (** A visual element to be drawn on an Axes. *)

  val line2d :
    ?color:color ->
    ?linewidth:float ->
    ?linestyle:line_style ->
    ?marker:marker_style ->
    ?label:string ->
    Ndarray.float32_t ->
    Ndarray.float32_t ->
    t
  (** [line2d ?color ?linewidth ?linestyle ?marker ?label x y]

      Create a 2D polyline connecting points (x.[i], y.[i]).

      {2 Parameters}
      - [?color]: line/marker color.
      - [?linewidth]: width of the line.
      - [?linestyle]: pattern for the line.
      - [?marker]: marker style at each data point.
      - [?label]: legend entry.
      - [x]: 1D float32 array of x coordinates.
      - [y]: 1D float32 array of y coordinates.

      {2 Returns}
      - an artist [t] for rendering on an axes.

      {2 Raises}
      - [Invalid_argument] if x and y lengths differ. *)

  val line3d :
    ?color:color ->
    ?linewidth:float ->
    ?linestyle:line_style ->
    ?marker:marker_style ->
    ?label:string ->
    Ndarray.float32_t ->
    Ndarray.float32_t ->
    Ndarray.float32_t ->
    t
  (** [line3d ?color ?linewidth ?linestyle ?marker ?label x y z]

      Create a 3D polyline connecting points (x.[i], y.[i], z.[i]).

      {2 Parameters}
      - [?color]: line/marker color.
      - [?linewidth]: width of the line.
      - [?linestyle]: pattern for the line.
      - [?marker]: marker style at each data point.
      - [?label]: legend entry.
      - [x]: 1D float32 ndarray of x coordinates.
      - [y]: 1D float32 ndarray of y coordinates.
      - [z]: 1D float32 ndarray of z coordinates.

      {2 Returns}
      - an artist [t] representing the configured 3D line.

      {2 Raises}
      - [Invalid_argument] if lengths of [x], [y], and [z] differ.

      {2 Examples}
      {[
        let a3d = line3d x_arr y_arr z_arr in
        Axes.add_artist a3d ax3d
      ]} *)

  val scatter :
    ?s:float ->
    ?c:color ->
    ?marker:marker_style ->
    ?label:string ->
    Ndarray.float32_t ->
    Ndarray.float32_t ->
    t
  (** [scatter ?s ?c ?marker ?label x y]

      Create a scatter plot artist for points (x.[i], y.[i]).

      {2 Parameters}
      - [?s]: marker size in points (default 6.0).
      - [?c]: marker color (default [Artist.Color.blue]).
      - [?marker]: marker style (default [Circle]).
      - [?label]: legend entry.
      - [x]: 1D float32 ndarray of x coordinates.
      - [y]: 1D float32 ndarray of y coordinates.

      {2 Returns}
      - an artist [t] representing the scatter plot.

      {2 Raises}
      - [Invalid_argument] if lengths of [x] and [y] differ.

      {2 Examples}
      {[
        let sc =
          scatter ~s:10. ~c:Color.red ~marker:Circle ~label:"pts" ~x x_arr ~y
            y_arr
        in
        Axes.add_artist sc axes
      ]} *)

  val bar :
    ?width:float ->
    ?bottom:float ->
    ?color:color ->
    ?label:string ->
    height:Ndarray.float32_t ->
    Ndarray.float32_t ->
    t
  (** [bar ?width ?bottom ?color ?label ~height x]

      Create a bar plot artist for (x, height) data.

      {2 Parameters}
      - [?width]: width of each bar (default 0.8).
      - [?bottom]: y-coordinate of the bottom of the bars (default 0.0).
      - [?color]: bar color.
      - [?label]: legend entry.
      - [height]: 1D float32 ndarray of heights.
      - [x]: 1D float32 ndarray of x coordinates.

      {2 Returns}
      - an artist [t] representing the bar plot.

      {2 Raises}
      - [Invalid_argument] if lengths of [x] and [height] differ.

      {2 Examples}
      {[
        let b =
          bar ~width:0.5 ~bottom:0. ~color:Color.green ~label:"bars"
            ~height:height_arr x_arr
        in
        Axes.add_artist b axes
      ]} *)

  val step :
    ?color:color ->
    ?linewidth:float ->
    ?linestyle:line_style ->
    ?where:step_where ->
    ?label:string ->
    Ndarray.float32_t ->
    Ndarray.float32_t ->
    t
  (** [step ?color ?linewidth ?linestyle ?where ?label x y]

      Create a step plot artist for (x, y) data.

      {2 Parameters}
      - [?color]: line color.
      - [?linewidth]: line width in points.
      - [?linestyle]: pattern for stepped line.
      - [?where]: position of step relative to x ([Pre], [Post], [Mid]).
      - [?label]: legend entry.
      - [x]: 1D float32 ndarray of x coordinates.
      - [y]: 1D float32 ndarray of y coordinates.

      {2 Returns}
      - an artist [t] representing the step plot.

      {2 Raises}
      - [Invalid_argument] if lengths of [x] and [y] differ.

      {2 Examples}
      {[
        let st = step x_arr y_arr ~where:Mid in
        Axes.add_artist st axes
      ]} *)

  val fill_between :
    ?color:color ->
    ?where:Ndarray.float32_t ->
    ?interpolate:bool ->
    ?label:string ->
    Ndarray.float32_t ->
    Ndarray.float32_t ->
    Ndarray.float32_t ->
    t
  (** [fill_between ?color ?where ?interpolate ?label x y1 y2]

      Create an artist filling the area between two curves y1(x) and y2(x).

      {2 Parameters}
      - [?color]: fill color.
      - [?where]: 1D float32 mask; true values fill region.
      - [?interpolate]: whether to interpolate missing segments (default false).
      - [?label]: legend entry.
      - [x]: 1D float32 ndarray of x coordinates.
      - [y1]: 1D float32 ndarray of lower bound values.
      - [y2]: 1D float32 ndarray of upper bound values.

      {2 Returns}
      - an artist [t] representing the filled region.

      {2 Raises}
      - [Invalid_argument] if lengths of [x], [y1], and [y2] differ.

      {2 Examples}
      {[
        let fb = fill_between x_arr y_lo y_hi in
        Axes.add_artist fb axes
      ]} *)

  val text :
    ?color:color -> ?fontsize:float -> x:float -> y:float -> string -> t
  (** [text ?color ?fontsize ~x ~y content]

      Create a text annotation at the specified coordinates.

      {2 Parameters}
      - [?color]: text color (default [Artist.Color.black]).
      - [?fontsize]: font size in points (default 12.0).
      - [x]: x-coordinate in data units.
      - [y]: y-coordinate in data units.
      - [content]: text string to display.

      {2 Returns}
      - a [t] representing the text artist.

      {2 Notes}
      - Text is anchored at (x,y); alignment options not configurable.

      {2 Examples}
      {[
        let tx = text ~x:0.5 ~y:0.5 "Hello" in
        Axes.add_artist tx axes
      ]} *)

  val image :
    ?extent:float * float * float * float ->
    ?cmap:cmap ->
    ?aspect:string ->
    Ndarray.uint8_t ->
    t
  (** [image ?extent ?cmap ?aspect data]

      Create an image artist from a uint8 ndarray.

      {2 Parameters}
      - [?extent]: (xmin, xmax, ymin, ymax) for positioning; defaults to data
        indices.
      - [?cmap]: colormap for interpreting scalar data; ignored for RGB(A)
        arrays.
      - [?aspect]: aspect ratio, e.g., "auto" or "equal" (default "auto").
      - [data]: uint8 ndarray of shape [|H;W|], [|H;W;1|], [|H;W;3|], or
        [|H;W;4|].

      {2 Returns}
      - a [t] representing the image artist.

      {2 Raises}
      - [Invalid_argument] if [data] rank or channel count is unsupported.

      {2 Examples}
      {[
        let img = image ~extent:(0., 10., 0., 5.) img_arr in
        Axes.add_artist img axes
      ]} *)
end

(** Module for axes-level operations (the plotting area). *)
module Axes : sig
  type t
  (** Abstract type representing an Axes object within a figure. *)

  (** Type representing the projection of the axes. *)
  type projection = TwoD | ThreeD

  (** Type representing axis scale. *)
  type scale = Linear | Log

  val set_title : string -> t -> t
  (** [set_title title axes]

      Set the title text for the axes.

      {2 Parameters}
      - [title]: string to display as the axes title.
      - [axes]: axes instance to modify.

      {2 Returns}
      - updated axes with the new title.

      {2 Examples}
      {[
        let ax = set_title "My Plot" ax in
        ...
      ]} *)

  val set_xlabel : string -> t -> t
  (** [set_xlabel label axes]

      Set the x-axis label text for the axes.

      {2 Parameters}
      - [label]: string to display on the x-axis.
      - [axes]: axes instance to modify.

      {2 Returns}
      - updated axes with the new x-axis label.

      {2 Examples}
      {[
        let ax = set_xlabel "Time (s)" ax in
        ...
      ]} *)

  val set_ylabel : string -> t -> t
  (** [set_ylabel label axes]

      Set the y-axis label text for the axes.

      {2 Parameters}
      - [label]: string to display on the y-axis.
      - [axes]: axes instance to modify.

      {2 Returns}
      - updated axes with the new y-axis label.

      {2 Examples}
      {[
        let ax = set_ylabel "Amplitude" ax in
        ...
      ]} *)

  val set_zlabel : string -> t -> t
  (** [set_zlabel label axes]

      Set the z-axis label text for 3D axes.

      {2 Parameters}
      - [label]: string to display on the z-axis.
      - [axes]: 3D axes instance to modify.

      {2 Returns}
      - updated axes with the new z-axis label.

      {2 Examples}
      {[
        let ax3d = set_zlabel "Depth" ax3d in
        ...
      ]} *)

  val set_xlim : ?min:float -> ?max:float -> t -> t
  (** [set_xlim ?min ?max axes]

      Set the visible range for the x-axis.

      {2 Parameters}
      - [?min]: lower x-axis limit; [Float.nan] for automatic.
      - [?max]: upper x-axis limit; [Float.nan] for automatic.
      - [axes]: axes instance to modify.

      {2 Returns}
      - updated axes with specified x-axis limits.

      {2 Examples}
      {[
        let ax = set_xlim ~min:0. ~max:10. ax in
        ...
      ]} *)

  val set_ylim : ?min:float -> ?max:float -> t -> t
  (** [set_ylim ?min ?max axes]

      Set the visible range for the y-axis.

      {2 Parameters}
      - [?min]: lower y-axis limit; [Float.nan] for automatic.
      - [?max]: upper y-axis limit; [Float.nan] for automatic.
      - [axes]: axes instance to modify.

      {2 Returns}
      - updated axes with specified y-axis limits.

      {2 Examples}
      {[
        let ax = set_ylim ~min:-1. ~max:1. ax in
        ...
      ]} *)

  val set_zlim : ?min:float -> ?max:float -> t -> t
  (** [set_zlim ?min ?max axes]

      Set the visible range for the z-axis in 3D plots.

      {2 Parameters}
      - [?min]: lower z-axis limit; [Float.nan] for automatic.
      - [?max]: upper z-axis limit; [Float.nan] for automatic.
      - [axes]: 3D axes instance to modify.

      {2 Returns}
      - updated axes with specified z-axis limits.

      {2 Examples}
      {[
        let ax3d = set_zlim ~min:0. ~max:5. ax3d in
        ...
      ]} *)

  val set_xscale : scale -> t -> t
  (** [set_xscale scale axes]

      Set the x-axis scaling (linear or logarithmic).

      {2 Parameters}
      - [scale]: [Linear] or [Log].
      - [axes]: axes instance to modify.

      {2 Returns}
      - updated axes with new x-axis scale.

      {2 Examples}
      {[
        let ax = set_xscale Log ax in
        ...
      ]} *)

  val set_yscale : scale -> t -> t
  (** [set_yscale scale axes]

      Set the y-axis scaling (linear or logarithmic).

      {2 Parameters}
      - [scale]: [Linear] or [Log].
      - [axes]: axes instance to modify.

      {2 Returns}
      - updated axes with new y-axis scale.

      {2 Examples}
      {[
        let ax = set_yscale Log ax in
        ...
      ]} *)

  val set_xticks : float list -> t -> t
  (** [set_xticks ticks axes]

      Set manual tick positions on the x-axis.

      {2 Parameters}
      - [ticks]: list of positions (floats).
      - [axes]: axes instance to modify.

      {2 Returns}
      - updated axes with specified x-axis ticks.

      {2 Examples}
      {[
        let ax = set_xticks [0.;1.;2.;3.] ax in
        ...
      ]} *)

  val set_yticks : float list -> t -> t
  (** [set_yticks ticks axes]

      Set manual tick positions on the y-axis.

      {2 Parameters}
      - [ticks]: list of positions.
      - [axes]: axes instance.

      {2 Returns}
      - updated axes with specified y-axis ticks.

      {2 Examples}
      {[
        let ax = set_yticks [0.;0.5;1.] ax in
        ...
      ]} *)

  val set_zticks : float list -> t -> t
  (** [set_zticks ticks axes]

      Set manual tick positions on the z-axis (3D plots).

      {2 Parameters}
      - [ticks]: list of positions.
      - [axes]: 3D axes instance.

      {2 Returns}
      - updated axes with specified z-axis ticks.

      {2 Examples}
      {[
        let ax3d = set_zticks [0.;5.;10.] ax3d in
        ...
      ]} *)

  val set_elev : float -> t -> t
  (** [set_elev angle axes]

      Set elevation angle for 3D axes in degrees.

      {2 Parameters}
      - [angle]: elevation angle in degrees.
      - [axes]: 3D axes instance.

      {2 Returns}
      - updated axes with new elevation.

      {2 Examples}
      {[
        let ax3d = set_elev 30. ax3d in
        ...
      ]} *)

  val set_azim : float -> t -> t
  (** [set_azim angle axes]

      Set azimuth angle for 3D axes in degrees.

      {2 Parameters}
      - [angle]: azimuth angle in degrees.
      - [axes]: 3D axes instance.

      {2 Returns}
      - updated axes with new azimuth.

      {2 Examples}
      {[
        let ax3d = set_azim 45. ax3d in
        ...
      ]} *)

  val grid :
    ?which:[ `major | `minor | `both ] ->
    ?axis:[ `x | `y | `both ] ->
    bool ->
    t ->
    t
  (** [grid ?which ?axis visible axes]

      Toggle grid lines on the axes.

      {2 Parameters}
      - [?which]: [ `major ], [ `minor ], or [ `both ] grid lines.
      - [?axis]: axes to apply ([ `x ], [ `y ], or [ `both ]).
      - [visible]: [true] to show, [false] to hide.
      - [axes]: axes instance.

      {2 Returns}
      - updated axes with grid state changed.

      {2 Notes}
      - Defaults: [which=`major], [axis=`both].

      {2 Examples}
      {[
        let ax = grid ~visible:true axes in
        ...
      ]} *)

  val add_artist : Artist.t -> t -> t
  (** [add_artist artist axes]

      Add a custom artist to the axes.

      {2 Parameters}
      - [artist]: an [Artist.t] element.
      - [axes]: axes instance to draw onto.

      {2 Returns}
      - updated axes with the artist added.

      {2 Examples}
      {[
        let a = Artist.line2d x_arr y_arr in
        let ax = add_artist a ax in
        ...
      ]} *)

  val cla : t -> t
  (** [cla axes]

      Clear all artists, titles, and labels from the axes.

      {2 Parameters}
      - [axes]: axes instance to clear.

      {2 Returns}
      - cleared axes ready for fresh plotting.

      {2 Examples}
      {[
        let ax = cla ax in
        ...
      ]} *)
end

(** Module for figure-level operations (the top-level canvas). *)
module Figure : sig
  type t
  (** Abstract type representing a Figure, which contains Axes. *)

  val create : ?width:int -> ?height:int -> unit -> t
  (** [create ?width ?height ()]

      Create a new figure canvas.

      {2 Parameters}
      - [?width]: figure width in pixels (default 800).
      - [?height]: figure height in pixels (default 600).

      {2 Returns}
      - a new [Figure.t] instance.

      {2 Examples}
      {[
        let fig = create ~width:1024 ~height:768 () in
        ...
      ]} *)

  val add_axes :
    left:float ->
    bottom:float ->
    width:float ->
    height:float ->
    ?projection:Axes.projection ->
    t ->
    Axes.t
  (** [add_axes ~left ~bottom ~width ~height ?projection fig]

      Add custom-positioned axes to the figure.

      {2 Parameters}
      - [left]: distance from left edge (0.0 to 1.0 fraction).
      - [bottom]: distance from bottom edge.
      - [width]: width of axes as fraction of figure.
      - [height]: height of axes as fraction of figure.
      - [?projection]: [TwoD] or [ThreeD] (default [TwoD]).
      - [fig]: parent figure.

      {2 Returns}
      - new [Axes.t] placed on the figure.

      {2 Examples}
      {[
        let ax = add_axes ~left:0.1 ~bottom:0.1 ~width:0.8 ~height:0.8 fig in
        ...
      ]} *)

  val add_subplot :
    ?nrows:int ->
    ?ncols:int ->
    ?index:int ->
    ?projection:Axes.projection ->
    t ->
    Axes.t
  (** [add_subplot ?nrows ?ncols ?index ?projection fig]

      Add a subplot in a grid layout to the figure.

      {2 Parameters}
      - [?nrows]: number of rows in grid (default 1).
      - [?ncols]: number of columns (default 1).
      - [?index]: position index (1-based, default 1).
      - [?projection]: [TwoD] or [ThreeD] axes type.
      - [fig]: parent figure.

      {2 Returns}
      - new [Axes.t] for the subplot.

      {2 Examples}
      {[
        let ax = add_subplot ~nrows:2 ~ncols:2 ~index:3 fig in
        ...
      ]} *)

  val clf : t -> t
  (** [clf fig]

      Clear all axes from the figure, resetting to empty canvas.

      {2 Parameters}
      - [fig]: figure to clear.

      {2 Returns}
      - cleared figure with no axes.

      {2 Examples}
      {[
        let fig = clf fig in
        ...
      ]} *)
end

(** Module containing functions to add standard plot types to an Axes. *)
module Plotting : sig
  val plot :
    ?color:Artist.color ->
    ?linewidth:float ->
    ?linestyle:Artist.line_style ->
    ?marker:Artist.marker_style ->
    ?label:string ->
    x:Ndarray.float32_t ->
    y:Ndarray.float32_t ->
    Axes.t ->
    Axes.t
  (** [plot ?color ?linewidth ?linestyle ?marker ?label ~x ~y axes]

      Plot y versus x on the specified axes.

      {2 Parameters}
      - [?color]: line/marker color.
      - [?linewidth]: width of the line in points.
      - [?linestyle]: dash pattern for the line.
      - [?marker]: marker style at data points.
      - [?label]: legend entry for the plotted data.
      - [x]: 1D float32 ndarray of x coordinates.
      - [y]: 1D float32 ndarray of y coordinates.
      - [axes]: target axes to draw the plot on.

      {2 Returns}
      - updated axes with the plotted line and markers.

      {2 Raises}
      - [Invalid_argument] if [x] and [y] lengths differ.

      {2 Examples}
      {[
        let ax = plot x_arr y_arr ax in
        ...
      ]} *)

  val plot_y :
    ?color:Artist.color ->
    ?linewidth:float ->
    ?linestyle:Artist.line_style ->
    ?marker:Artist.marker_style ->
    ?label:string ->
    y:Ndarray.float32_t ->
    Axes.t ->
    Axes.t
  (** [plot_y ?color ?linewidth ?linestyle ?marker ?label ~y axes]

      Plot y data against integer indices [0..N-1] on the given axes.

      {2 Parameters}
      - [?color]: line/marker color.
      - [?linewidth]: line width.
      - [?linestyle]: dash pattern.
      - [?marker]: point marker style.
      - [?label]: legend entry.
      - [y]: 1D float32 ndarray of y values.
      - [axes]: target axes.

      {2 Returns}
      - axes with the plotted data.

      {2 Examples}
      {[
        let ax = plot_y y_arr ax in
        ...
      ]} *)

  val plot3d :
    ?color:Artist.color ->
    ?linewidth:float ->
    ?linestyle:Artist.line_style ->
    ?marker:Artist.marker_style ->
    ?label:string ->
    x:Ndarray.float32_t ->
    y:Ndarray.float32_t ->
    z:Ndarray.float32_t ->
    Axes.t ->
    Axes.t
  (** [plot3d ?color ?linewidth ?linestyle ?marker ?label ~x ~y ~z axes]

      Plot a 3D line through points (x.[i], y.[i], z.[i]).

      {2 Parameters}
      - [?color]: line/marker color.
      - [?linewidth]: line thickness.
      - [?linestyle]: dash pattern.
      - [?marker]: marker style.
      - [?label]: legend entry.
      - [x], [y], [z]: 1D float32 ndarrays of coordinates.
      - [axes]: 3D axes for plotting.

      {2 Returns}
      - axes with the 3D line artist added.

      {2 Raises}
      - [Invalid_argument] if lengths of [x], [y], [z] mismatch.

      {2 Examples}
      {[
        let ax3d = plot3d x_arr y_arr z_arr ax3d in
        ...
      ]} *)

  val scatter :
    ?s:float ->
    ?c:Artist.color ->
    ?marker:Artist.marker_style ->
    ?label:string ->
    x:Ndarray.float32_t ->
    y:Ndarray.float32_t ->
    Axes.t ->
    Axes.t
  (** [scatter ?s ?c ?marker ?label ~x ~y axes]

      Create a scatter plot of points (x, y) on the axes.

      {2 Parameters}
      - [?s]: marker size.
      - [?c]: marker color.
      - [?marker]: style of marker.
      - [?label]: legend label.
      - [x], [y]: coordinate arrays.
      - [axes]: target axes.

      {2 Returns}
      - axes with scatter artist added.

      {2 Raises}
      - [Invalid_argument] on length mismatch.

      {2 Examples}
      {[
        let ax = scatter x_arr y_arr ax in
        ...
      ]} *)

  val bar :
    ?width:float ->
    ?bottom:float ->
    ?color:Artist.color ->
    ?label:string ->
    x:Ndarray.float32_t ->
    height:Ndarray.float32_t ->
    Axes.t ->
    Axes.t
  (** [bar ?width ?bottom ?color ?label ~x ~height axes]

      Create a bar chart with bars centered at x and heights specified.

      {2 Parameters}
      - [?width]: bar width.
      - [?bottom]: baseline for bars.
      - [?color]: bar fill color.
      - [?label]: legend label.
      - [x]: positions for bars.
      - [height]: heights of bars.
      - [axes]: target axes.

      {2 Returns}
      - axes with bar artists added.

      {2 Raises}
      - [Invalid_argument] if [x] and [height] lengths differ.

      {2 Examples}
      {[
        let ax = bar ~x x_arr ~height h_arr ax in
        ...
      ]} *)

  val hist :
    ?bins:[ `Num of int | `Edges of float array ] ->
    ?range:float * float ->
    ?density:bool ->
    ?color:Artist.color ->
    ?label:string ->
    x:Ndarray.float32_t ->
    Axes.t ->
    Axes.t
  (** [hist ?bins ?range ?density ?color ?label ~x axes]

      Plot a histogram of the data in x.

      {2 Parameters}
      - [?bins]: number of bins or explicit edges.
      - [?range]: (min, max) data range to include.
      - [?density]: plot probability density instead of counts.
      - [?color]: fill color for bars.
      - [?label]: legend label.
      - [x]: 1D float32 array of data values.
      - [axes]: target axes.

      {2 Returns}
      - axes with histogram bars added.

      {2 Raises}
      - [Invalid_argument] on invalid range or bins.

      {2 Examples}
      {[
        let ax = hist ~bins:(`Num 20) ~range:(0.,1.) ~density:true ~x x_arr ax in
        ...
      ]} *)

  val step :
    ?color:Artist.color ->
    ?linewidth:float ->
    ?linestyle:Artist.line_style ->
    ?where:Artist.step_where ->
    ?label:string ->
    x:Ndarray.float32_t ->
    y:Ndarray.float32_t ->
    Axes.t ->
    Axes.t
  (** [step ?color ?linewidth ?linestyle ?where ?label ~x ~y axes]

      Create a step plot connecting points with horizontal segments.

      {2 Parameters}
      - [?color]: line color.
      - [?linewidth]: line thickness.
      - [?linestyle]: dash pattern.
      - [?where]: step alignment ([Pre], [Post], [Mid]).
      - [?label]: legend entry.
      - [x], [y]: 1D float32 arrays of coordinates.
      - [axes]: target axes.

      {2 Returns}
      - axes with step plot added.

      {2 Raises}
      - [Invalid_argument] if [x] and [y] lengths differ.

      {2 Examples}
      {[
        let ax = step x_arr y_arr ~where:Mid ax in
        ...
      ]} *)

  val fill_between :
    ?color:Artist.color ->
    ?where:Ndarray.float32_t ->
    ?interpolate:bool ->
    ?label:string ->
    x:Ndarray.float32_t ->
    y1:Ndarray.float32_t ->
    y2:Ndarray.float32_t ->
    Axes.t ->
    Axes.t
  (** [fill_between ?color ?where ?interpolate ?label ~x ~y1 ~y2 axes]

      Shade region between curves y1(x) and y2(x).

      {2 Parameters}
      - [?color]: fill color.
      - [?where]: boolean mask array; optional.
      - [?interpolate]: fill missing segments (default false).
      - [?label]: legend entry.
      - [x], [y1], [y2]: coordinate arrays.
      - [axes]: target axes.

      {2 Returns}
      - axes with shaded region added.

      {2 Raises}
      - [Invalid_argument] if lengths of [x], [y1], [y2] mismatch.

      {2 Examples}
      {[
        let ax = fill_between ~x x_arr ~y1 y_low ~y2 y_high ax in
        ...
      ]} *)

  val errorbar :
    ?yerr:Ndarray.float32_t ->
    ?xerr:Ndarray.float32_t ->
    ?ecolor:Artist.color ->
    ?elinewidth:float ->
    ?capsize:float ->
    ?fmt:Artist.plot_style ->
    ?label:string ->
    x:Ndarray.float32_t ->
    y:Ndarray.float32_t ->
    Axes.t ->
    Axes.t
  (** [errorbar ?yerr ?xerr ?ecolor ?elinewidth ?capsize ?fmt ?label ~x ~y axes]

      Add error bars to data points on the axes.

      {2 Parameters}
      - [?yerr]: symmetric y-error values.
      - [?xerr]: symmetric x-error values.
      - [?ecolor]: error bar line color.
      - [?elinewidth]: error bar line width.
      - [?capsize]: width of error bar caps in points.
      - [?fmt]: central line/marker style as [plot_style].
      - [?label]: legend entry for central data.
      - [x], [y]: data coordinate arrays.
      - [axes]: target axes.

      {2 Returns}
      - axes with error bar artists added.

      {2 Raises}
      - [Invalid_argument] if array lengths mismatch.

      {2 Examples}
      {[
        let ax = errorbar x_arr y_arr ~yerr y_err ax in
        ...
      ]} *)

  val imshow :
    ?cmap:Artist.cmap ->
    ?aspect:string ->
    ?extent:float * float * float * float ->
    data:('a, 'b) Ndarray.t ->
    Axes.t ->
    Axes.t
  (** [imshow ?cmap ?aspect ?extent ~data axes]

      Display a 2D or 3D image array on the axes.

      {2 Parameters}
      - [?cmap]: colormap for single-channel arrays.
      - [?aspect]: aspect ratio option ("auto" or "equal").
      - [?extent]: (xmin,xmax,ymin,ymax) image bounds.
      - [data]: ndarray of shape [|H;W|], [|H;W;3|], or [|H;W;4|].
      - [axes]: target axes.

      {2 Returns}
      - axes with image artist.

      {2 Raises}
      - [Invalid_argument] on unsupported array rank or shape.

      {2 Examples}
      {[
        let ax = imshow img_arr ax in
        ...
      ]} *)

  val matshow :
    ?cmap:Artist.cmap ->
    ?aspect:string ->
    ?extent:float * float * float * float ->
    ?origin:[ `upper | `lower ] ->
    data:('a, 'b) Ndarray.t ->
    Axes.t ->
    Axes.t
  (** [matshow ?cmap ?aspect ?extent ?origin ~data axes]

      Display a 2D numeric matrix with grid-aligned cells.

      {2 Parameters}
      - [?cmap]: colormap for mapping values to colors.
      - [?aspect]: aspect ratio for display.
      - [?extent]: (xmin,xmax,ymin,ymax) bounds.
      - [?origin]: data origin placement ([`upper] or [`lower]).
      - [data]: 2D ndarray of numeric values.
      - [axes]: target axes.

      {2 Returns}
      - axes with matrix display.

      {2 Examples}
      {[
        let ax = matshow ~data m_arr ax in
        ...
      ]} *)

  val text :
    ?color:Artist.color ->
    ?fontsize:float ->
    x:float ->
    y:float ->
    string ->
    Axes.t ->
    Axes.t
  (** [text ?color ?fontsize ~x ~y content axes]

      Add a text annotation at data coordinates.

      {2 Parameters}
      - [?color]: text color (default black).
      - [?fontsize]: text size in points (default 12.0).
      - [x]: x-coordinate.
      - [y]: y-coordinate.
      - [content]: text string to display.
      - [axes]: target axes.

      {2 Returns}
      - axes with text annotation added.

      {2 Examples}
      {[
        let ax = text ~x:1. ~y:2. "Label" ax in
        ...
      ]} *)
end

(** {1 Top-Level API}

    These functions create a Figure and Axes implicitly, plot the data, apply
    common optional decorations, and return the Figure. *)

type figure = Figure.t
(** Type alias for Figure.t *)

type axes = Axes.t
(** Type alias for Axes.t *)

val plot :
  ?title:string ->
  ?xlabel:string ->
  ?ylabel:string ->
  ?color:Artist.color ->
  ?linewidth:float ->
  ?linestyle:Artist.line_style ->
  ?marker:Artist.marker_style ->
  ?label:string ->
  Ndarray.float32_t ->
  Ndarray.float32_t ->
  figure
(** [plot ?title ?xlabel ?ylabel ?color ?linewidth ?linestyle ?marker ?label x
     y]

    Create a new figure and plot y versus x.

    {2 Parameters}
    - [?title]: title displayed atop the figure.
    - [?xlabel]: label for the x-axis.
    - [?ylabel]: label for the y-axis.
    - [?color]: line and marker color.
    - [?linewidth]: thickness of the line.
    - [?linestyle]: dash pattern of the line.
    - [?marker]: marker style at data points.
    - [?label]: legend entry for the plotted data.
    - [x]: 1D float32 ndarray of x coordinates.
    - [y]: 1D float32 ndarray of y coordinates.

    {2 Returns}
    - a [figure] containing the created plot.

    {2 Examples}
    {[
      let fig = plot x_arr y_arr in
      savefig fig "plot.png"
    ]} *)

val plot_y :
  ?title:string ->
  ?xlabel:string ->
  ?ylabel:string ->
  ?color:Artist.color ->
  ?linewidth:float ->
  ?linestyle:Artist.line_style ->
  ?marker:Artist.marker_style ->
  ?label:string ->
  Ndarray.float32_t ->
  figure
(** [plot_y ?title ?xlabel ?ylabel ?color ?linewidth ?linestyle ?marker ?label
     y]

    Create a new figure and plot the data [y] against indices [0..N-1].

    {2 Parameters}
    - [?title]: figure title.
    - [?xlabel]: x-axis label.
    - [?ylabel]: y-axis label.
    - [?color]: line/marker color.
    - [?linewidth]: thickness of the line.
    - [?linestyle]: dash pattern of the line.
    - [?marker]: marker style at data points.
    - [?label]: legend entry for the series.
    - [y]: 1D float32 ndarray of y values.

    {2 Returns}
    - a [figure] containing the plotted series.

    {2 Examples}
    {[
      let fig = plot_y y_arr in
      savefig fig "plot_y.png"
    ]} *)

val plot3d :
  ?title:string ->
  ?xlabel:string ->
  ?ylabel:string ->
  ?zlabel:string ->
  ?color:Artist.color ->
  ?linewidth:float ->
  ?linestyle:Artist.line_style ->
  ?marker:Artist.marker_style ->
  ?label:string ->
  Ndarray.float32_t ->
  Ndarray.float32_t ->
  Ndarray.float32_t ->
  figure
(** [plot3d ?title ?xlabel ?ylabel ?zlabel ?color ?linewidth ?linestyle ?marker
     ?label x y z]

    Create a new figure, add 3D axes, and plot a 3D line through points.

    {2 Parameters}
    - [?title]: figure title.
    - [?xlabel], [?ylabel], [?zlabel]: axis labels.
    - [?color]: line/marker color.
    - [?linewidth]: line width.
    - [?linestyle]: dash pattern.
    - [?marker]: marker style.
    - [?label]: legend entry.
    - [x], [y], [z]: 1D float32 ndarrays of coordinates.

    {2 Returns}
    - a [figure] with the 3D line plotted.

    {2 Examples}
    {[
      let fig = plot3d x_arr y_arr z_arr in
      savefig fig "plot3d.png"
    ]} *)

val scatter :
  ?title:string ->
  ?xlabel:string ->
  ?ylabel:string ->
  ?s:float ->
  ?c:Artist.color ->
  ?marker:Artist.marker_style ->
  ?label:string ->
  Ndarray.float32_t ->
  Ndarray.float32_t ->
  figure
(** [scatter ?title ?xlabel ?ylabel ?s ?c ?marker ?label x y]

    Create a new figure and scatter plot points (x,y).

    {2 Parameters}
    - [?title]: figure title.
    - [?xlabel]: x-axis label.
    - [?ylabel]: y-axis label.
    - [?s]: marker size in points.
    - [?c]: marker color.
    - [?marker]: marker style.
    - [?label]: legend entry.
    - [x], [y]: coordinate arrays.

    {2 Returns}
    - a [figure] containing the scatter plot.

    {2 Raises}
    - [Invalid_argument] if lengths of [x] and [y] differ.

    {2 Examples}
    {[
      let fig = scatter x_arr y_arr in
      savefig fig "scatter.png"
    ]} *)

val hist :
  ?bins:[ `Num of int | `Edges of float array ] ->
  ?range:float * float ->
  ?density:bool ->
  ?title:string ->
  ?xlabel:string ->
  ?ylabel:string ->
  ?color:Artist.color ->
  ?label:string ->
  Ndarray.float32_t ->
  figure
(** [hist ?bins ?range ?density ?title ?xlabel ?ylabel ?color ?label x]

    Create a new figure and plot a histogram of the data in x.

    {2 Parameters}
    - [?bins]: number of bins or explicit edges.
    - [?range]: (min, max) interval for data inclusion.
    - [?density]: if true, plot probability density instead of counts.
    - [?title]: figure title.
    - [?xlabel]: x-axis label.
    - [?ylabel]: y-axis label.
    - [?color]: bar fill color.
    - [?label]: legend entry.
    - [x]: 1D float32 ndarray of data values.

    {2 Returns}
    - a [figure] containing the histogram plot.

    {2 Examples}
    {[
      let fig = hist ~bins:(`Num 50) ~range:(0., 1.) ~density:true data in
      savefig fig "hist.png"
    ]} *)

val bar :
  ?width:float ->
  ?bottom:float ->
  ?title:string ->
  ?xlabel:string ->
  ?ylabel:string ->
  ?color:Artist.color ->
  ?label:string ->
  height:Ndarray.float32_t ->
  Ndarray.float32_t ->
  figure
(** [bar ?width ?bottom ?title ?xlabel ?ylabel ?color ?label ~height x]

    Create a new figure and render a bar chart.

    {2 Parameters}
    - [?width]: width of each bar (default 0.8).
    - [?bottom]: baseline for bars (default 0.0).
    - [?title]: figure title.
    - [?xlabel]: label for x-axis.
    - [?ylabel]: label for y-axis.
    - [?color]: bar fill color.
    - [?label]: legend entry.
    - [~height]: 1D float32 ndarray of bar heights.
    - [x]: 1D float32 ndarray of bar positions.

    {2 Returns}
    - a [figure] containing the bar chart.

    {2 Examples}
    {[
      let fig = bar ~height:h_arr x_arr in
      savefig fig "bar.png"
    ]} *)

val imshow :
  ?cmap:Artist.cmap ->
  ?aspect:string ->
  ?extent:float * float * float * float ->
  ?title:string ->
  ?xlabel:string ->
  ?ylabel:string ->
  ('a, 'b) Ndarray.t ->
  figure
(** [imshow ?cmap ?aspect ?extent ?title ?xlabel ?ylabel data]

    Create a new figure and display an image from a data array.

    {2 Parameters}
    - [?cmap]: colormap for single-channel (2D) data.
    - [?aspect]: aspect ratio mode ("auto" or "equal").
    - [?extent]: bounding box (xmin, xmax, ymin, ymax).
    - [?title]: figure title.
    - [?xlabel], [?ylabel]: axis labels.
    - [data]: ndarray of shape [|H;W|], [|H;W;3|], or [|H;W;4|].

    {2 Returns}
    - a [figure] with the image displayed.

    {2 Raises}
    - [Invalid_argument] if [data] rank or shape is unsupported.

    {2 Examples}
    {[
      let fig = imshow img_arr in
      savefig fig "image.png"
    ]} *)

(* Added Top-Level wrappers for new plot types *)
val step :
  ?title:string ->
  ?xlabel:string ->
  ?ylabel:string ->
  ?color:Artist.color ->
  ?linewidth:float ->
  ?linestyle:Artist.line_style ->
  ?where:Artist.step_where ->
  ?label:string ->
  Ndarray.float32_t ->
  Ndarray.float32_t ->
  figure
(** [step ?title ?xlabel ?ylabel ?color ?linewidth ?linestyle ?where ?label x y]

    Create a new figure and draw a step plot for data y vs x.

    {2 Parameters}
    - [?title]: figure title.
    - [?xlabel], [?ylabel]: axis labels.
    - [?color]: line color.
    - [?linewidth]: line width.
    - [?linestyle]: dash pattern.
    - [?where]: step alignment relative to x.
    - [?label]: legend entry.
    - [x], [y]: coordinate arrays.

    {2 Returns}
    - a [figure] containing the step plot.

    {2 Raises}
    - [Invalid_argument] on length mismatch.

    {2 Examples}
    {[
      let fig = step x_arr y_arr ~where:Mid in
      savefig fig "step.png"
    ]} *)

val fill_between :
  ?title:string ->
  ?xlabel:string ->
  ?ylabel:string ->
  ?color:Artist.color ->
  ?where:Ndarray.float32_t ->
  ?interpolate:bool ->
  ?label:string ->
  Ndarray.float32_t ->
  Ndarray.float32_t ->
  Ndarray.float32_t ->
  figure
(** [fill_between ?title ?xlabel ?ylabel ?color ?where ?interpolate ?label x y1
     y2]

    Create a new figure and shade the area between two curves.

    {2 Parameters}
    - [?title]: figure title.
    - [?xlabel], [?ylabel]: axis labels.
    - [?color]: fill color.
    - [?where]: mask array selecting regions to fill.
    - [?interpolate]: interpolate across missing regions.
    - [?label]: legend entry.
    - [x]: x-coordinate array.
    - [y1], [y2]: arrays defining lower and upper curves.

    {2 Returns}
    - a [figure] containing the shaded region.

    {2 Raises}
    - [Invalid_argument] on length mismatch.

    {2 Examples}
    {[
      let fig = fill_between x_arr y_lo y_hi in
      savefig fig "area.png"
    ]} *)

val errorbar :
  ?title:string ->
  ?xlabel:string ->
  ?ylabel:string ->
  ?yerr:Ndarray.float32_t ->
  ?xerr:Ndarray.float32_t ->
  ?ecolor:Artist.color ->
  ?elinewidth:float ->
  ?capsize:float ->
  ?fmt:Artist.plot_style ->
  ?label:string ->
  Ndarray.float32_t ->
  Ndarray.float32_t ->
  figure
(** [errorbar ?title ?xlabel ?ylabel ?yerr ?xerr ?ecolor ?elinewidth ?capsize
     ?fmt ?label x y]

    Create a new figure and plot data with error bars.

    {2 Parameters}
    - [?title]: figure title.
    - [?xlabel], [?ylabel]: axis labels.
    - [?yerr]: array of y-error values.
    - [?xerr]: array of x-error values.
    - [?ecolor]: error bar color.
    - [?elinewidth]: error bar line width.
    - [?capsize]: cap size in points.
    - [?fmt]: plot style for central points/line.
    - [?label]: legend entry.
    - [x], [y]: data coordinate arrays.

    {2 Returns}
    - a [figure] containing the error bar plot.

    {2 Raises}
    - [Invalid_argument] if array lengths mismatch.

    {2 Examples}
    {[
      let fig = errorbar x_arr y_arr ~yerr:y_err in
      savefig fig "errorbar.png"
    ]} *)

val matshow :
  ?cmap:Artist.cmap ->
  ?aspect:string ->
  ?extent:float * float * float * float ->
  ?origin:[ `upper | `lower ] ->
  ?title:string ->
  ('a, 'b) Ndarray.t ->
  figure
(** [matshow ?cmap ?aspect ?extent ?origin ?title data]

    Create a new figure and display a 2D matrix with cell coloring.

    {2 Parameters}
    - [?cmap]: colormap for mapping values.
    - [?aspect]: aspect ratio mode.
    - [?extent]: (xmin,xmax,ymin,ymax) plot bounds.
    - [?origin]: [`upper] or [`lower] y-axis origin placement.
    - [?title]: figure title.
    - [data]: 2D ndarray of numeric values.

    {2 Returns}
    - a [figure] containing the matrix image.

    {2 Examples}
    {[
      let fig = matshow matrix in
      savefig fig "matrix.png"
    ]} *)

(** {1 Core Figure/Axes Management} *)

val figure : ?width:int -> ?height:int -> unit -> figure
(** [figure ?width ?height ()]

    Create a new figure canvas with optional size.

    {2 Parameters}
    - [?width]: width in pixels (default 800).
    - [?height]: height in pixels (default 600).
    - [()]: unit argument.

    {2 Returns}
    - a new [figure].

    {2 Examples}
    {[
      let fig = figure ~width:1024 ~height:768 () in
      ...
    ]} *)

val subplot :
  ?nrows:int ->
  ?ncols:int ->
  ?index:int ->
  ?projection:Axes.projection ->
  figure ->
  axes
(** [subplot ?nrows ?ncols ?index ?projection fig]

    Add a subplot in a grid layout to the figure.

    {2 Parameters}
    - [?nrows]: number of rows (default 1).
    - [?ncols]: number of columns (default 1).
    - [?index]: subplot index (1-based, default 1).
    - [?projection]: [TwoD] or [ThreeD] axes type.
    - [fig]: parent figure.

    {2 Returns}
    - created [axes].

    {2 Examples}
    {[
      let ax = subplot ~nrows:2 ~ncols:2 ~index:3 fig in
      ...
    ]} *)

val add_axes :
  left:float ->
  bottom:float ->
  width:float ->
  height:float ->
  ?projection:Axes.projection ->
  figure ->
  axes
(** [add_axes ~left ~bottom ~width ~height ?projection fig]

    Add custom-positioned axes to the figure.

    {2 Parameters}
    - [~left]: left margin (fraction of figure width).
    - [~bottom]: bottom margin (fraction of figure height).
    - [~width], [~height]: size of axes (fractions).
    - [?projection]: [TwoD] or [ThreeD].
    - [fig]: parent figure.

    {2 Returns}
    - created [axes].

    {2 Examples}
    {[
      let ax = add_axes ~left:0.1 ~bottom:0.1 ~width:0.8 ~height:0.8 fig in
      ...
    ]} *)

(** {1 Display and Saving} *)

val show : figure -> unit
(** [show figure]

    Render and display the figure in an interactive window.

    {2 Parameters}
    - [figure]: the figure to display.

    {2 Notes}
    - Blocks execution until the window is closed.

    {2 Examples}
    {[
      let fig = plot x_arr y_arr in
      show fig
    ]} *)

val savefig : ?dpi:int -> ?format:string -> string -> figure -> unit
(** [savefig ?dpi ?format filename fig]

    Save the figure to a file.

    {2 Parameters}
    - [?dpi]: resolution in dots per inch (default 100).
    - [?format]: output format (e.g., "png", "pdf"); inferred from extension by
      default.
    - [filename]: destination file path.
    - [fig]: figure to save.

    {2 Notes}
    - Supported formats depend on the Cairo and writer backends.

    {2 Examples}
    {[
      let fig = plot x_arr y_arr in
      savefig ~dpi:300 "highres.png" fig
    ]} *)
