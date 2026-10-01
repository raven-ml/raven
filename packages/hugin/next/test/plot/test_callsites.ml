(* The call sites of RFC 0016's Guide, against hugin_next.mli. This library is
   compiled and never run: it checks that the interface states what the
   benchmark figures need. Munin and talon values that do not exist yet are
   stubbed with the types the RFC gives them. *)

module Munin = struct
  module Run = struct
    type t

    let history (_ : t) (_ : string) : Nx.float64_t * Nx.float64_t =
      assert false
  end
end

module Talon = struct
  module Type = struct
    type t = Categorical of string array | Float
    type any = Any of t
  end

  module Column = struct
    type t

    let type_ (_ : t) : Type.any = assert false
    let to_tensor (_ : ('a, 'b) Nx.dtype) (_ : t) : ('a, 'b) Nx.t = assert false
  end

  type t

  let column (_ : t) (_ : string) : Column.t = assert false
end

open Hugin_next

(* A first figure. *)

let first losses = save "loss.svg" (line ~y:(num losses) ())

(* Data enters through lifts: losses : [5; T]. *)

let seeds steps losses =
  line ~x:(num steps) ~y:(num losses) ~stroke:(dim ~title:(Text.v "seed") 0) ()

let step_axis losses =
  line ~x:(index ~title:(Text.v "step") (-1)) ~y:(num losses) ()

(* Scales are named, and composition decides sharing. *)

let log_loss step loss =
  line ~x:(num step) ~y:(num ~scale:(Scale.log ()) loss) ()

let sweep rates finals losses =
  let lr = Scale.log ~name:"lr" ~scheme:Scheme.viridis () in
  let f =
    grid
      [
        [
          line ~y:(num losses) ~stroke:(num ~scale:lr rates) ();
          dot ~x:(num ~scale:lr rates) ~y:(num finals) ();
        ];
      ]
  in
  let fitted = Resolved.scale (resolve f) lr in
  line ~y:(num losses) ~stroke:(num ~scale:fitted rates) ()

(* Composing figures. The Guide stacks these bars with [stack], which comes with
   the data-and-transforms RFC. *)

let compose a b = (grid [ [ a; b ] ], grid [ [ a ]; [ b ] ], span ~cols:2 a)

let bars classes cls score metrics metric =
  rect ~x:(cat ~labels:classes cls) ~y:(num score)
    ~fill:(cat ~labels:metrics metric)
    ()

(* Training dashboard. *)

let dashboard run =
  let step, loss = Munin.Run.history run "train/loss" in
  let vstep, acc = Munin.Run.history run "val/accuracy" in
  let epochs, _ = Munin.Run.history run "epoch" in
  let at t = Nx.take ~indices:(Nx.argmax acc) t in
  let losses =
    layer
      [
        rule ~x:(num epochs) ~opacity:(const 0.15) ();
        line
          ~x:(num ~title:(Text.v "step") step)
          ~y:(num ~scale:(Scale.log ()) ~title:(Text.v "loss") loss)
          ~opacity:(const 0.3) ();
        line ~x:(num step) ~y:(num (Nx.ewma ~alpha:0.02 loss)) ();
      ]
  and accuracy =
    layer
      [
        dot ~x:(num vstep) ~y:(num ~title:(Text.v "val. accuracy") acc) ();
        text
          ~x:(num (at vstep))
          ~y:(num (at acc))
          ~text:(num (at acc))
          ~dy:6. ();
      ]
  in
  grid [ [ losses ]; [ accuracy ] ] |> share [ ("x", `Shared) ]

(* Confusion matrix. *)

let confusion (m : Nx.int32_t) classes =
  let counts = Nx.cast Nx.float64 m in
  let recall =
    num ~title:(Text.v "recall")
      Nx.(div counts (sum ~axes:[ 1 ] ~keepdims:true counts))
  in
  let predicted = dim ~title:(Text.v "predicted") ~labels:classes 1
  and truth = dim ~title:(Text.v "true") ~labels:classes 0 in
  layer
    [
      rect ~x:predicted ~y:truth ~fill:recall ();
      text ~x:predicted ~y:truth ~text:(num m)
        ~fill:(map_range Color.contrast recall)
        ();
    ]
  |> coord (Coord.cartesian ~aspect:1. ())

(* Attention grid: a : [12; 12; T; T]. *)

let attention (a : Nx.float32_t) tokens =
  rect
    ~fy:(dim ~title:(Text.v "layer") 0)
    ~fx:(dim ~title:(Text.v "head") 1)
    ~y:(dim ~labels:tokens 2) ~x:(dim ~labels:tokens 3)
    ~fill:(num ~title:(Text.v "attention") a)
    ()

(* Embedding scatter, e : [100_000; 2], and its density. nx counts points into
   the cells of given edges, so the density computes its edges where the data
   lives. *)

let embedding (e : Nx.float32_t) (labels : Nx.int32_t) digits =
  dot
    ~x:(num Nx.(slice [ A; I 0 ] e))
    ~y:(num Nx.(slice [ A; I 1 ] e))
    ~fill:(cat ~labels:digits labels)
    ~opacity:(const 0.3) ()

let density (x : Nx.float32_t) (y : Nx.float32_t) =
  let edges v =
    let lo = Nx.min v in
    Nx.add lo (Nx.mul (Nx.sub (Nx.max v) lo) (Nx.linspace Nx.float32 0. 1. 201))
  in
  let xe = edges x and ye = edges y in
  let counts = Nx.histogram [ (xe, x); (ye, y) ] in
  rect
    ~x:(num Nx.(slice [ R (0, 200); N ] xe))
    ~x2:(num Nx.(slice [ R (1, 201); N ] xe))
    ~y:(num Nx.(slice [ N; R (0, 200) ] ye))
    ~y2:(num Nx.(slice [ N; R (1, 201) ] ye))
    ~fill:(num ~scale:(Scale.log ()) counts)
    ()

(* Image grid: batch : [64; 32; 32; 3] in [0, 1]. *)

let predictions (batch : Nx.float32_t) ~(truth : Nx.int32_t)
    ~(pred : Nx.int32_t) classes =
  let name c = classes.(Int32.to_int c) in
  let caption t p = name t ^ " → " ^ name p in
  let captions = Array.map2 caption (Nx.to_array truth) (Nx.to_array pred) in
  layer
    [
      image ~fx:(dim ~scale:(Scale.band ~wrap:8 ()) ~labels:captions 0) batch;
      rect
        ~fx:(dim ~valid:(Nx.not_equal truth pred) 0)
        ~stroke:(const Color.red) ();
    ]

(* Faceted talon plot. *)

let by_optimizer t =
  let open Talon in
  let codes name =
    let c = column t name in
    match Column.type_ c with
    | Type.Any (Type.Categorical labels) ->
        cat ~labels (Column.to_tensor Nx.int32 c)
    | _ -> invalid_arg name
  in
  line
    ~x:(num (Column.to_tensor Nx.int64 (column t "step")))
    ~y:
      (num ~title:(Text.v "loss")
         (Column.to_tensor Nx.float64 (column t "loss")))
    ~stroke:(codes "optimizer") ~fx:(codes "batch_size") ()

(* Loss landscape: loss : [nb; na], alphas : [na], betas : [nb], path : [n; 2].
   betas gets an axis of its own so that it varies along the rows of loss. *)

let landscape ~(alphas : Nx.float64_t) ~(betas : Nx.float64_t)
    (loss : Nx.float64_t) (path : Nx.float64_t) =
  let px = num Nx.(slice [ A; I 0 ] path)
  and py = num Nx.(slice [ A; I 1 ] path) in
  layer
    [
      contour ~x:(num alphas)
        ~y:(num Nx.(slice [ A; N ] betas))
        ~fill:(num ~scale:(Scale.log ()) ~title:(Text.v "loss") loss)
        ();
      line ~x:px ~y:py ();
      dot ~x:px ~y:py ~size:(const 9.) ();
    ]

(* Paper figure. *)

let figure_3 ~a ~b ~c ~d =
  let panel s f = title ~align:`Left (Text.bold (Text.v s)) f in
  let font file =
    In_channel.(with_open_bin file input_all) |> Font.of_string |> Result.get_ok
  in
  let theme =
    Theme.v ~size:8.
      ~fonts:[ font "paper-regular.ttf"; font "paper-bold.ttf" ]
      ()
  in
  let fig =
    grid [ [ panel "(a)" a; panel "(b)" b ]; [ panel "(c)" c; panel "(d)" d ] ]
  in
  save ~theme
    ~size:(Size.figure (Size.mm 180.) (Size.mm 110.))
    "figure-3.pdf" fig

let eta = Text.(concat [ v "step size η"; sub (v "0") ])

(* Rendering in stages. *)

type shown = { r : Resolved.t; l : Layout.t; d : Drawing.t }

let redraw prev ~view ~size fig =
  let r = resolve ~prev:prev.r ~view fig in
  let l = layout ~prev:prev.l size r in
  { r; l; d = draw ~prev:prev.d ~density:2. l }

let to_png f =
  let d = render (Size.figure 360. 240.) f in
  List.iter (Format.eprintf "%a@." pp_warning) (Drawing.warnings d);
  Hugin_next_vg_raster.png ~density:2. (Drawing.renderable d)

let to_svg f =
  Hugin_next_vg_svg.render
    (Drawing.renderable (render (Size.panels 120. 80.) f))

let quietly f = save ~warn:ignore ~density:(Size.dpi 300.) "figure.png" f

(* View values: a slider read by bind, and a zoom. *)

let smoothing = View.number "alpha" ~init:0.02

let live loss =
  bind smoothing (fun alpha -> line ~y:(num (Nx.ewma ~alpha loss)) ())

let zoomed f =
  let k = View.zoom (Scale.linear ~name:"x" ()) in
  render ~view:View.(set k (Some (0., 100.)) empty) (Size.figure 360. 240.) f

(* Guides as figures. *)

let with_guides f =
  layer [ f; axis ~grid:true "y"; legend ~side:`Bottom "color" ]
  |> share [ ("color", `Independent) ]
  |> name "main"

(* Extending Hugin: Fehu's cart. The page is y-down, so the pole's tip lies
   above its foot at a smaller y. *)

let cart ~(x : Nx.float32_t) ~(angle : Nx.float32_t) =
  let theta = Role.value ~name:"angle" in
  Mark.v ~name:"fehu.cart"
    [
      Mark.bind Role.x (num x);
      Mark.bind Role.y (const 0.2);
      Mark.bind theta (num angle);
    ]
    (fun rows ->
      let xs, ys = Mark.points rows and a = Option.get (Mark.get rows theta) in
      let pole k =
        Picture.stroke (Stroke.v 2.) Color.black
          (Path.polyline
             [| xs.(k); xs.(k) +. (30. *. sin a.(k)) |]
             [| ys.(k); ys.(k) -. (30. *. cos a.(k)) |])
      in
      Picture.group (List.init (Mark.length rows) pole))

(* Built-in marks as a user writes them with Mark.v. *)

let fills_or_accent rows =
  match Mark.get rows Role.fill with
  | Some cs -> cs
  | None -> Array.make (Mark.length rows) (Theme.accent (Mark.theme rows))

let my_dot ?fill ~x ~y () =
  let fill = Option.to_list (Option.map (Mark.bind Role.fill) fill) in
  Mark.v ~name:"my_dot" ~reduce:Mark.raster
    ([ Mark.bind Role.x x; Mark.bind Role.y y ] @ fill)
    (fun rows ->
      let xs, ys = Mark.points rows in
      let d = 0.5 *. Theme.size (Mark.theme rows) in
      let area = Float.pi *. d *. d /. 4. in
      let glyph =
        Picture.fill Color.black (Symbol.path `Fill area Symbol.circle)
      in
      Picture.stamp ~fills:(fills_or_accent rows) xs ys glyph)

let my_rect ?fill ~x ~y () =
  let length = Scale.linear ~zero:true () in
  let fill = Option.to_list (Option.map (Mark.bind Role.fill) fill) in
  Mark.v ~name:"my_rect" ~reduce:Mark.cells
    ([ Mark.bind ~imply:length Role.x x; Mark.bind ~imply:length Role.y y ]
    @ fill)
    (fun rows ->
      let x0, x1 = Mark.extent rows `X and y0, y1 = Mark.extent rows `Y in
      let fills = fills_or_accent rows in
      let cell i =
        let finite = Float.is_finite in
        if finite x0.(i) && finite x1.(i) && finite y0.(i) && finite y1.(i) then
          let b = Box2.of_pts (P2.v x0.(i) y0.(i)) (P2.v x1.(i) y1.(i)) in
          Picture.fill fills.(i) (Mark.project rows (Path.rect b))
        else Picture.empty
      in
      Picture.group (List.init (Mark.length rows) cell))

let my_line ?curve ?x ~y () =
  let curve = Option.value curve ~default:Curve.linear in
  let x =
    match x with
    | Some x -> Mark.bind Role.x x
    | None -> Mark.bind Role.x (index (-1))
  in
  Mark.v ~name:"my_line" ~reduce:Mark.m4
    [ x; Mark.bind Role.y y ]
    (fun rows ->
      let th = Mark.theme rows in
      let pen = Stroke.v (0.15 *. Theme.size th) in
      let series s =
        if Mark.length s < 2 then Mark.warn rows "a series of one row";
        let xs, ys = Mark.points s in
        Picture.stroke pen (Theme.accent th) (Curve.path curve xs ys)
      in
      Picture.group (List.map series (Mark.series rows)))

let my_text ~x ~y ~text () =
  Mark.v ~name:"my_text"
    [ Mark.bind Role.x x; Mark.bind Role.y y; Mark.bind Role.text text ]
    (fun rows ->
      let xs, ys = Mark.points rows
      and ts = Option.get (Mark.get rows Role.text) in
      let ink = Theme.ink (Mark.theme rows) in
      let label i = Mark.text rows ink (P2.v xs.(i) ys.(i)) ts.(i) in
      Picture.group (List.init (Mark.length rows) label))

(* Images over a leading datum axis: px : [n; h; w; c]. Channels of shape [n]
   give the mark one row per image. *)

let my_image ?fx (px : Nx.uint8_t) =
  let shape = Nx.shape px in
  let per_image v = num (Nx.full Nx.float64 [| shape.(0) |] v) in
  let pixels = Scale.linear ~nice:false () in
  let rows_down = Scale.linear ~nice:false ~reverse:true () in
  let fx = Option.to_list (Option.map (Mark.bind Role.fx) fx) in
  Mark.v ~name:"my_image"
    ~coord:(Coord.cartesian ~aspect:1. ())
    ([
       Mark.bind ~imply:pixels ~guide:false Role.x (per_image 0.);
       Mark.bind Role.x2 (per_image (float shape.(2)));
       Mark.bind ~imply:rows_down ~guide:false Role.y (per_image 0.);
       Mark.bind Role.y2 (per_image (float shape.(1)));
     ]
    @ fx)
    (fun rows ->
      let at = Coord.point (Mark.projection rows) in
      let x0, x1 = Mark.extent rows `X and y0, y1 = Mark.extent rows `Y in
      let index = Mark.index rows in
      let image i =
        let box = Box2.of_pts (at x0.(i) y0.(i)) (at x1.(i) y1.(i)) in
        Picture.image box (Nx.slice [ I index.(i) ] px)
      in
      Picture.group (List.init (Mark.length rows) image))

(* Filled contours: the bands between the fill scale's levels, each painted with
   the colour of its midpoint through the scale's range. The fields of a panel
   are consecutive blocks of rows, dropped samples included. *)

let strictly_monotone a =
  let sign i = Float.compare a.(i + 1) a.(i) in
  Array.for_all Float.is_finite a
  && Seq.for_all
       (fun i -> sign i <> 0 && sign i = sign 0)
       (Seq.init (max 0 (Array.length a - 1)) Fun.id)

let my_contour ?x ?y ~fill () =
  let x =
    match x with
    | Some x -> Mark.bind Role.x x
    | None -> Mark.bind Role.x (index (-1))
  and y =
    match y with
    | Some y -> Mark.bind Role.y y
    | None -> Mark.bind Role.y (index (-2))
  in
  Mark.v ~name:"my_contour"
    [ x; y; Mark.bind Role.fill fill ]
    (fun rows ->
      let open Hugin_next_gg_kit in
      let shape = Mark.shape rows in
      let n = shape.(Array.length shape - 2)
      and m = shape.(Array.length shape - 1) in
      let normalized r = Option.get (Mark.normalized rows r) in
      let us = normalized Role.fill
      and col_x = normalized Role.x
      and row_y = normalized Role.y in
      let color = Option.get (Mark.range rows Role.fill) in
      let levels =
        Option.get (Mark.ticks rows Role.fill)
        |> Array.to_list |> List.cons 0. |> List.cons 1.
        |> List.sort_uniq Float.compare
        |> Array.of_list
      in
      let draw_field k =
        let first = k * n * m in
        let xs = Array.init m (fun j -> col_x.(first + j))
        and ys = Array.init n (fun i -> row_y.(first + (i * m))) in
        if strictly_monotone xs && strictly_monotone ys then
          let z =
            Nx.create Nx.float64 [| n; m |] (Array.sub us first (n * m))
          in
          let f = Field2.v ~xs ~ys z in
          let band l =
            let lo = levels.(l) and hi = levels.(l + 1) in
            let path = Pgon2.to_path (Field2.isoband ~lo ~hi f) in
            Picture.fill (color ((lo +. hi) /. 2.)) (Mark.project rows path)
          in
          Picture.group (List.init (Array.length levels - 1) band)
        else (
          Mark.warn rows "a field whose positions are not strictly monotone";
          Picture.empty)
      in
      Picture.group (List.init (Mark.length rows / (n * m)) draw_field))

(* A mark that draws its series as several stamps keeps per-instance provenance
   by tagging each stamp with its rows. *)

let ticks_mark ~x ~y () =
  Mark.v ~name:"ticks"
    [ Mark.bind Role.x x; Mark.bind Role.y y ]
    (fun rows ->
      let pen = Stroke.v 1. in
      let glyph =
        Picture.stroke pen Color.black (Symbol.path `Stroke 9. Symbol.plus)
      in
      let stamp s =
        let xs, ys = Mark.points s in
        Picture.tag
          { Picture.id = Mark.id s; rows = Picture.Rows (Mark.index s) }
          (Picture.stamp xs ys glyph)
      in
      Picture.group (List.map stamp (Mark.series rows)))
