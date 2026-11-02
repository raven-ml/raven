(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin
module C = Matrix_charts
module G = Matrix_grid
module Float_text = Munin_float_text.Float_text

type axis = Step | Step_metric | Time

type view = {
  axis : axis;
  smooth : int;
  log : bool;
  window : (float * float) option;
}

type spec = { key : string; width : int; height : int; x_axis : bool }

(* Every plot has these margins, so that plots stacked in one column share their
   columns whatever their labels: room on the left for a y label of at most
   [label_cells] cells and its tick, on the right for half of an x label. *)
let left = 12
let right = 5
let label_cells = 10

(* The x of samples *)

let steps s i = Float.of_int (Metric.step s i)

(* A sample pairs with the step metric's latest sample at its step. *)
let paired h key s =
  match History.step_metric h key with
  | None -> steps s
  | Some m -> (
      match History.series h m with
      | exception Invalid_argument _ -> fun _ -> Float.nan
      | ms -> (
          let at = Hashtbl.create (Metric.length ms) in
          for i = 0 to Metric.length ms - 1 do
            Hashtbl.replace at (Metric.step ms i) (Metric.value ms i)
          done;
          fun i ->
            match Hashtbl.find_opt at (Metric.step s i) with
            | Some v -> v
            | None -> Float.nan))

let x_of axis h key s =
  match axis with
  | Step -> steps s
  | Step_metric -> paired h key s
  | Time ->
      let at =
        Fact.active_at (Fact.pauses h) (Run.started_at (History.run h))
      in
      fun i -> at (Metric.time s i)

let bound x =
  if not (x > 0.) then 1.
  else
    let x = 1.1 *. x in
    let p = 10. ** Float.floor (Float.log10 x) in
    let pick = List.find (fun m -> m *. p >= x) [ 1.; 2.; 5.; 10. ] in
    pick *. p

let series h key =
  match History.series h key with
  | s -> Some s
  | exception Invalid_argument _ -> None

(* Tracks

   A track is one series reduced for a plot, with what it takes to extend it as
   a followed history grows. A follow's series extends the one before unless it
   read the log again from its start or read a resume ([History.follow]). A
   step's x never changes, and an active time changes only at a resume; a step
   metric's pairing can change as its samples arrive, so on that axis a track is
   reduced whole. *)

type track = { curve : Curve.t; length : int; resumes : int }

let reusable axis h prev s =
  axis <> Step_metric
  && (not (History.restarted h))
  && prev.resumes = List.length (History.resumes h)
  && prev.length <= Metric.length s

let track ?prev ?tau axis plot h key s =
  let x = x_of axis h key s in
  let curve =
    match prev with
    | Some p when reusable axis h p s ->
        if p.length = Metric.length s then p.curve
        else Curve.extend p.curve ~x s
    | _ -> Curve.reduce ?tau plot ~x s
  in
  { curve; length = Metric.length s; resumes = List.length (History.resumes h) }

(* The greatest x of a series, kept the same way. *)
type top = { top : float; seen : int; again : int (* resumes *) }

let top ?prev axis h key s =
  let x = x_of axis h key s in
  let from, m =
    match prev with
    | Some p
      when axis <> Step_metric
           && (not (History.restarted h))
           && p.again = List.length (History.resumes h)
           && p.seen <= Metric.length s ->
        (p.seen, p.top)
    | _ -> (0, 0.)
  in
  let m = ref m in
  for i = from to Metric.length s - 1 do
    let v = x i in
    if Float.is_finite v then m := Float.max !m v
  done;
  { top = !m; seen = Metric.length s; again = List.length (History.resumes h) }

(* The x domain: on the step axis to the run's declared [steps], or past them
   when a sample is, else to the bound of the greatest x. *)
let domain_of axis h tops =
  let t = List.fold_left (fun m t -> Float.max m t.top) 0. tops in
  match (axis, Run.steps (History.run h)) with
  | Step, Some n when n > 0 && t <= Float.of_int n -> (0., Float.of_int n)
  | _ -> (0., bound t)

let domain axis ~key hs =
  match hs with
  | [] -> (0., 1.)
  | h :: _ ->
      domain_of axis h
        (List.filter_map
           (fun h -> Option.map (top axis h key) (series h key))
           hs)

(* Domains of y *)

let widen = function
  | None -> (0., 1.)
  | Some (lo, hi) when lo < hi -> (lo, hi)
  | Some (v, _) ->
      let d = if v = 0. then 0.5 else Float.abs v *. 0.05 in
      (v -. d, v +. d)

let union a b =
  match (a, b) with
  | None, e | e, None -> e
  | Some (a, b), Some (c, d) -> Some (Float.min a c, Float.max b d)

let points_extent pts =
  Array.fold_left
    (fun e (_, y) -> if Float.is_finite y then union e (Some (y, y)) else e)
    None pts

(* Prepared charts *)

let fractions = [| 100.; 30.; 10. |]

(* [round_125 x] is the 1, 2 or 5 times a power of ten nearest [x] on a log
   scale. *)
let round_125 x =
  let p = 10. ** Float.floor (Float.log10 x) in
  let m = x /. p in
  let pick =
    if m < 1.5 then 1. else if m < 3.5 then 2. else if m < 7.5 then 5. else 10.
  in
  pick *. p

(* The difference rows' pairs, made in [prepare] ({!pair}). *)
type paired = {
  points : (float * float) array; (* per dot column: its middle, the mean *)
  last : (int * float * float) option; (* the latest pair: step, x, difference *)
}

type prepared = {
  view : view;
  spec : spec;
  ids : string list; (* the focus's and marked runs' ids *)
  lo : float;
  hi : float;
  tau : float option;
  dots : int;
  tops : (string * top) list;
  focus : track option; (* raw *)
  smooth : track option;
  marked : (char * string * track) list;
  unpaired : int;
  latest : float option; (* the focus's greatest x *)
  resumes : float list; (* their steps, on the step axis *)
  y : float * float;
  log : bool;
  differences : (char * paired) list;
  paired_at : (char * (int * int * int)) list;
      (* the series' lengths and resumes each pairing read *)
      (* the focus's level minus each marked run's, per dot column *)
}

let lo p = p.lo
let hi p = p.hi
let tau p = p.tau
let unpaired p = p.unpaired
let latest p = p.latest
let spec p = p.spec
let log_y p = p.log

(* Labels. An x label is a step count, a duration on the time axis, or a value;
   a y label at an extreme of the drawn values is a value, a round tick between
   them is printed as it is. *)
let x_label axis v =
  match axis with
  | Time -> Text.duration v
  | Step | Step_metric ->
      if Float.is_integer v && Float.abs v < 1e15 then
        Text.count (Float.to_int v)
      else Float_text.g ~precision:12 v

let y_label ~extreme v =
  Text.cut label_cells
    (if extreme then Text.value v
     else if Float.is_integer v && Float.abs v < 1e15 then
       Text.count (Float.to_int v)
     else Float_text.g ~precision:12 v)

(* The y ticks of a plot of [rows] over [(lo, hi)]: both extremes, so that the
   region the run is in is labelled, and between them multiples of a 1, 2 or 5
   step about a third of the plot apart, none closer than two rows to another;
   on a log scale the powers of ten between the extremes. *)
let y_ticks ~log (lo, hi) rows =
  if rows < 2 || not (hi > lo) then [ lo ]
  else
    let pos v =
      let f =
        if log then
          (Float.log10 v -. Float.log10 lo) /. (Float.log10 hi -. Float.log10 lo)
        else (v -. lo) /. (hi -. lo)
      in
      f *. Float.of_int (rows - 1)
    in
    (* A product such as [4. *. 0.2] is not [0.8]: each tick is snapped to the
       float nearest its twelve significant digits, the same on every system. *)
    let snap v =
      Option.value
        (Float_text.of_string (Float_text.g ~precision:12 v))
        ~default:v
    in
    let candidates =
      if log then
        let a = Float.ceil (Float.log10 lo)
        and b = Float.floor (Float.log10 hi) in
        List.init
          (max 0 (Float.to_int (b -. a) + 1))
          (fun i -> snap (10. ** (a +. Float.of_int i)))
      else
        let raw = (hi -. lo) /. Float.of_int (max 2 (rows / 3)) in
        let p = 10. ** Float.floor (Float.log10 raw) in
        let step =
          List.find (fun k -> k *. p >= raw) [ 1.; 2.; 5.; 10. ] *. p
        in
        let a = Float.ceil (lo /. step) and b = Float.floor (hi /. step) in
        List.init
          (max 0 (Float.to_int (b -. a) + 1))
          (fun i -> snap ((a +. Float.of_int i) *. step))
    in
    let far a b = Float.abs (pos a -. pos b) >= 2. in
    let inner =
      List.fold_left
        (fun kept v ->
          if
            v > lo && v < hi && far v lo && far v hi
            && List.for_all (far v) kept
          then v :: kept
          else kept)
        [] candidates
    in
    (lo :: List.rev inner) @ [ hi ]

let dash = function 'a' -> `Dotted | _ -> `Dashed

let figure ~faint ~axis ~x_axis ~labels ~rows ~lo ~hi ~y ~log ~resumes ~cursor
    ~marked ~raw ~focus =
  let resume_lines =
    List.map
      (fun x -> C.vline ~dash:`Dotted ~color:Ansi.Color.bright_black x)
      resumes
  in
  let cursor =
    match cursor with Some x -> [ C.vline ~dash:`Dotted x ] | None -> []
  in
  let marked =
    List.map
      (fun (l, level) ->
        C.line ~dots:`Braille ~dash:(dash l)
          ~color:(C.const (Paint.mark_color l))
          ~x:fst ~y:snd level)
      marked
  in
  let raw =
    match raw with
    | Some r ->
        [
          C.attrs Ansi.Attr.dim
            (C.line ~color:(C.const Ansi.Color.default_fg) ~x:fst ~y:snd r);
        ]
    | None -> []
  in
  let focus =
    C.line ~dots:`Braille
      ~color:(C.const Ansi.Color.default_fg)
      ~x:fst ~y:snd focus
  in
  let focus = if faint then C.attrs Ansi.Attr.dim focus else focus in
  let ylo, yhi = y in
  let ticks = y_ticks ~log y rows in
  let ys =
    if log then C.Scale.log ~domain:y ~ticks ()
    else C.Scale.linear ~domain:y ~ticks ()
  in
  let y_format _ v = y_label ~extreme:(v = ylo || v = yhi) v in
  C.layer
    (resume_lines @ cursor @ marked @ raw
    @ [
        focus;
        (if x_axis then
           C.axis ~spacing:16 ~format:(fun _ v -> x_label axis v) "x"
         else C.axis ~show:false "x");
        (if labels then C.axis ~format:y_format "y" else C.axis ~show:false "y");
      ])
  |> C.scale "x" (C.Scale.linear ~domain:(lo, hi) ())
  |> C.scale "y" ys |> C.margins ~left ~right

let theme light =
  let base = if light then C.Theme.light else C.Theme.dark in
  {
    base with
    C.Theme.axis = Paint.dim;
    label = Paint.dim;
    title = Paint.plain;
    grid = Paint.dim;
  }

(* The dot columns of a plot in a region, laid out as [draw] lays it out: the x
   domain sets the x labels, and the y labels fit in the left margin whatever
   they are, so the plot is the one the series are reduced for. *)
let dots ~axis ~lo ~hi spec =
  let fig =
    figure ~faint:false ~axis ~x_axis:spec.x_axis ~labels:spec.x_axis
      ~rows:(spec.height - 1) ~lo ~hi ~y:(0., 1.) ~log:false ~resumes:[]
      ~cursor:None ~marked:[] ~raw:None ~focus:[||]
  in
  2
  * (C.Layout.plot (C.layout ~width:spec.width ~height:(spec.height - 1) fig))
      .width

(* Difference rows

   The focus's sample minus a marked run's at the same step, for each step both
   logged with finite values: the marked run's sample at a step is its latest
   one there, as on the step metric's axis. The differences are averaged per dot
   column of the focus's x, and the latest is kept with its step. Two runs that
   log at different cadences, or one that ends inside a column, are compared
   only where both logged. *)

let pair axis plot focus key marked_h =
  match (series focus key, series marked_h key) with
  | Some fs, Some ms ->
      let at = Hashtbl.create (Metric.length ms) in
      for i = 0 to Metric.length ms - 1 do
        Hashtbl.replace at (Metric.step ms i) (Metric.value ms i)
      done;
      let x = x_of axis focus key fs in
      let sums = Array.make plot.Curve.dots 0.
      and counts = Array.make plot.Curve.dots 0 in
      let last = ref None in
      for i = 0 to Metric.length fs - 1 do
        let step = Metric.step fs i and v = Metric.value fs i in
        match Hashtbl.find_opt at step with
        | Some o when Float.is_finite v && Float.is_finite o ->
            let d = v -. o and xi = x i in
            if not (Float.is_nan xi) then begin
              let k = Curve.column plot xi in
              if k >= 0 && k < plot.dots then begin
                sums.(k) <- sums.(k) +. d;
                counts.(k) <- counts.(k) + 1
              end;
              match !last with
              | Some (s, _, _) when s > step -> ()
              | _ -> last := Some (step, xi, d)
            end
        | _ -> ()
      done;
      let width = (plot.hi -. plot.lo) /. Float.of_int plot.dots in
      let points =
        Array.of_list
          (List.filter_map Fun.id
             (List.init plot.dots (fun k ->
                  if counts.(k) = 0 then None
                  else
                    Some
                      ( plot.lo +. ((Float.of_int k +. 0.5) *. width),
                        sums.(k) /. Float.of_int counts.(k) ))))
      in
      { points; last = !last }
  | _ -> { points = [||]; last = None }

let prepare ?prev view spec focus marked =
  let key = spec.key in
  let ids =
    List.map (fun h -> Run.id (History.run h)) (focus :: List.map snd marked)
  in
  let prev =
    match prev with
    | Some p when p.view = view && p.spec = spec && p.ids = ids -> Some p
    | _ -> None
  in
  let all = ('-', focus) :: marked in
  let tops =
    List.filter_map
      (fun (_, h) ->
        let id = Run.id (History.run h) in
        Option.map
          (fun s ->
            let prev = Option.bind prev (fun p -> List.assoc_opt id p.tops) in
            (id, top ?prev view.axis h key s))
          (series h key))
      all
  in
  let lo, hi =
    match view.window with
    | Some w -> w
    | None -> domain_of view.axis focus (List.map snd tops)
  in
  let tau =
    if view.smooth = 0 then None
    else Some (round_125 ((hi -. lo) /. fractions.(view.smooth - 1)))
  in
  let dots = max 2 (dots ~axis:view.axis ~lo ~hi spec) in
  let plot = { Curve.lo; hi; dots } in
  let same =
    Option.map
      (fun p -> p.lo = lo && p.hi = hi && p.tau = tau && p.dots = dots)
      prev
    = Some true
  in
  let prev = if same then prev else None in
  let reduce ?tau get h =
    Option.map
      (fun s -> track ?prev:(Option.bind prev get) ?tau view.axis plot h key s)
      (series h key)
  in
  let focus_track = reduce (fun p -> p.focus) focus in
  let smooth =
    match tau with
    | Some tau -> reduce ~tau (fun p -> p.smooth) focus
    | None -> None
  in
  let marked =
    List.filter_map
      (fun (l, h) ->
        let id = Run.id (History.run h) in
        Option.map
          (fun t -> (l, id, t))
          (reduce ?tau
             (fun p ->
               Option.map
                 (fun (_, _, t) -> t)
                 (List.find_opt (fun (l', _, _) -> l' = l) p.marked))
             h))
      marked
  in
  let unpaired =
    match (view.axis, series focus key) with
    | Step_metric, Some s ->
        let x = x_of view.axis focus key s in
        let n = ref 0 in
        for i = 0 to Metric.length s - 1 do
          if Float.is_nan (x i) then incr n
        done;
        !n
    | _ -> 0
  in
  let extent =
    let e = Option.bind focus_track (fun t -> Curve.extent t.curve) in
    let e = union e (Option.bind smooth (fun t -> Curve.extent t.curve)) in
    List.fold_left
      (fun e (_, _, t) -> union e (points_extent (Curve.level t.curve)))
      e marked
  in
  let log =
    view.log && match extent with Some (lo, _) -> lo > 0. | None -> false
  in
  let resumes =
    if view.axis = Step then
      List.map
        (fun (r : History.resume) -> Float.of_int r.step)
        (History.resumes focus)
    else []
  in
  (* Pairing reads both series whole: a preparation's pairs are kept while its
     plot, both series' lengths and their resumes stay. *)
  let differences =
    List.map
      (fun (l, h) ->
        let lengths =
          ( Option.fold ~none:0 ~some:Metric.length (series focus key),
            Option.fold ~none:0 ~some:Metric.length (series h key),
            List.length (History.resumes focus)
            + List.length (History.resumes h) )
        in
        let kept =
          if History.restarted focus || History.restarted h then None
          else
            Option.bind prev (fun p ->
                match
                  (List.assoc_opt l p.differences, List.assoc_opt l p.paired_at)
                with
                | Some d, Some n when n = lengths -> Some d
                | _ -> None)
        in
        ( l,
          (match kept with
          | Some d -> d
          | None -> pair view.axis plot focus key h),
          lengths ))
      (List.tl all)
  in
  {
    view;
    spec;
    ids;
    lo;
    hi;
    tau;
    dots;
    tops;
    focus = focus_track;
    smooth;
    marked;
    unpaired;
    latest = Option.map (fun t -> t.top) (List.assoc_opt (List.hd ids) tops);
    resumes;
    y = widen extent;
    log;
    differences = List.map (fun (l, d, _) -> (l, d)) differences;
    paired_at = List.map (fun (l, _, n) -> (l, n)) differences;
  }

type placed = { px : int; pw : int }

let inside g x y = x >= 0 && y >= 0 && x < G.width g && y < G.height g

(* The marks of samples that are not finite, in [row]: a cell per marked column,
   [×] (or [↑], [↓]) where a span starts and a dotted rule along it. A span runs
   from one marked column to the next while no finite sample lies between them,
   so a metric that stays NaN reads as one span whether it logs every step or
   every thousand. *)
let marks ?(blank_only = false) p l (plot : G.region) g ~x ~row =
  let cells = Hashtbl.create 16 in
  Array.iter
    (fun (k, e) ->
      let col = min (plot.width - 1) (k * plot.width / max 1 p.dots) in
      let glyph =
        match e with `Nan -> "×" | `Pos_inf -> "↑" | `Neg_inf -> "↓"
      in
      match Hashtbl.find_opt cells col with
      | Some "×" -> ()
      | _ -> Hashtbl.replace cells col glyph)
    (match p.focus with Some t -> Curve.events t.curve | None -> [||]);
  let finite = Hashtbl.create 64 in
  let pts =
    match p.focus with Some t -> Curve.points t.curve | None -> [||]
  in
  Array.iter
    (fun (px, py) ->
      if Float.is_finite py then
        Hashtbl.replace finite
          (C.Layout.position l "x" (C.Layout.Num px) - plot.x)
          ())
    pts;
  let cols =
    List.sort compare (Hashtbl.fold (fun c _ acc -> c :: acc) cells [])
  in
  let rule = Hashtbl.create 64 in
  let rec spans = function
    | a :: (b :: _ as rest) ->
        let clear = ref true in
        for c = a + 1 to b - 1 do
          if Hashtbl.mem finite c then clear := false
        done;
        if !clear then
          for c = a + 1 to b - 1 do
            Hashtbl.replace rule c ()
          done;
        spans rest
    | _ -> ()
  in
  spans cols;
  let put col glyph =
    let gx = x + plot.x + col in
    if inside g gx row then begin
      let i = G.idx g ~x:gx ~y:row in
      if (not blank_only) || G.is_empty g i || G.get_text g i = " " then
        G.draw_text ~style:Paint.bad g ~x:gx ~y:row ~text:glyph
    end
  in
  Hashtbl.iter (fun col () -> put col "┄") rule;
  List.iter
    (fun col ->
      put col
        (if Hashtbl.mem cells (col - 1) || Hashtbl.mem rule (col - 1) then "┄"
         else Hashtbl.find cells col))
    cols

let draw ?cursor ?(inline = false) ?(faint = false) ~light p g ~x ~y =
  let labels = p.spec.x_axis in
  let width = p.spec.width and rows = p.spec.height - 1 in
  (* Inline, the plot takes the region's top row and the marks go in its blank
     cells. *)
  let y = if inline then y - 1 else y in
  if width < 2 || rows < 1 then None
  else begin
    let focus_pts, raw =
      match (p.smooth, p.focus) with
      | Some s, f ->
          (Curve.points s.curve, Option.map (fun t -> Curve.points t.curve) f)
      | None, Some f -> (Curve.points f.curve, None)
      | None, None -> ([||], None)
    in
    let levels =
      List.map (fun (l, _, t) -> (l, Curve.level t.curve)) p.marked
    in
    let fig =
      figure ~faint ~axis:p.view.axis ~x_axis:p.spec.x_axis ~labels ~rows
        ~lo:p.lo ~hi:p.hi ~y:p.y ~log:p.log ~resumes:p.resumes ~cursor
        ~marked:levels ~raw ~focus:focus_pts
    in
    let l = C.layout ~theme:(theme light) ~width ~height:rows fig in
    let plot = C.Layout.plot l in
    C.draw l g ~x ~y:(y + 1);
    let cell (px, py) =
      ( C.Layout.position l "x" (C.Layout.Num px),
        C.Layout.position l "y" (C.Layout.Num py) )
    in
    let in_region gx gy =
      gx >= x && gx < x + width && gy > y && gy <= y + rows && inside g gx gy
    in
    let is_mark i =
      List.exists
        (fun (c, _) -> Ansi.Color.equal (G.get_fg g i) (Paint.mark_color c))
        levels
    in
    (* A letter goes where no line inks, on its own line's ink, or at a line's
       end over another marked run's; never over the focus's. *)
    let can ?(at_end = false) c (cx, cy) =
      let gx = x + cx and gy = y + 1 + cy in
      in_region gx gy
      &&
      let i = G.idx g ~x:gx ~y:gy in
      G.is_empty g i
      || G.get_text g i = " "
      || Ansi.Color.equal (G.get_fg g i) (Paint.mark_color c)
      || (at_end && is_mark i)
    in
    let letter ?at_end c (cx, cy) =
      if can ?at_end c (cx, cy) then
        G.draw_text ~style:(Paint.mark c) g ~x:(x + cx)
          ~y:(y + 1 + cy)
          ~text:(String.make 1 c)
    in
    let is_letter (cx, cy) =
      let gx = x + cx and gy = y + 1 + cy in
      in_region gx gy
      && List.exists
           (fun (c, _) -> G.get_text g (G.idx g ~x:gx ~y:gy) = String.make 1 c)
           levels
    in
    List.iter
      (fun (c, level) ->
        let n = Array.length level in
        if n > 0 then begin
          List.iter (fun k -> letter c (cell level.(k))) [ n / 3; 2 * n / 3 ];
          let cx, cy = cell level.(n - 1) in
          let cx = if cx + 1 < width then cx + 1 else cx in
          (* The end letter goes in the end cell, or beside it when another
             letter or the focus holds it, so that a run that ends under the
             focus is still named where it ends. *)
          Option.iter (letter ~at_end:true c)
            (List.find_opt
               (fun q -> (not (is_letter q)) && can ~at_end:true c q)
               [
                 (cx, cy);
                 (cx, cy - 1);
                 (cx, cy + 1);
                 (cx + 1, cy - 1);
                 (cx + 1, cy + 1);
                 (cx, cy - 2);
                 (cx, cy + 2);
               ])
        end)
      levels;
    (* The focus's latest sample: a dot in the empty cell after it, where the
       eye looks for "now"; none when it is not finite, which its mark says. *)
    (let n = Array.length focus_pts in
     if n > 0 && Float.is_finite (snd focus_pts.(n - 1)) then
       let cx, cy = cell focus_pts.(n - 1) in
       let gx = x + cx + 1 and gy = y + 1 + cy in
       if gx >= x + plot.x && gx < x + plot.x + plot.width && in_region gx gy
       then begin
         let i = G.idx g ~x:gx ~y:gy in
         if G.is_empty g i || G.get_text g i = " " then
           G.draw_text ~style:Paint.bold g ~x:gx ~y:gy ~text:"•"
       end);
    marks ~blank_only:inline p l plot g ~x ~row:(if inline then y + 1 else y);
    Some { px = x + plot.x; pw = plot.width }
  end

type versus = {
  last : float;
  step : int;
  at : float;
  scale : float;
  zero_row : int;
}

let draw_versus ~light p c g ~(under : placed) ~y ~rows =
  match List.assoc_opt c p.differences with
  | None | Some { last = None; _ } | Some { points = [||]; _ } -> None
  | Some { points = d; last = Some (step, at, last) } ->
      let big =
        Array.fold_left (fun a (_, v) -> Float.max a (Float.abs v)) 0. d
      in
      let big = if big > 0. then big else 1. in
      let fig =
        C.layer
          [
            C.line ~dots:`Braille
              ~color:(C.const (Paint.mark_color c))
              ~x:fst ~y:snd d;
            C.axis ~show:false "x";
            C.axis ~show:false "y";
          ]
        |> C.scale "x" (C.Scale.linear ~domain:(p.lo, p.hi) ())
        |> C.scale "y" (C.Scale.linear ~domain:(-.big, big) ~nice:false ())
        |> C.margins ~left ~right
      in
      let l =
        C.layout ~theme:(theme light)
          ~width:(under.pw + left + right)
          ~height:rows fig
      in
      let plot = C.Layout.plot l in
      C.draw l g ~x:(under.px - plot.x) ~y;
      (* The zero rule, in the cells of zero's row that no dot inks. *)
      let zy =
        y + min (rows - 1) (max 0 (C.Layout.position l "y" (C.Layout.Num 0.)))
      in
      for cx = under.px to under.px + under.pw - 1 do
        if inside g cx zy then begin
          let i = G.idx g ~x:cx ~y:zy in
          if G.is_empty g i || G.get_text g i = " " then
            G.draw_text ~style:Paint.dim g ~x:cx ~y:zy ~text:"┄"
        end
      done;
      Some { last; step; at; scale = big; zero_row = zy }

type trend = {
  t_axis : axis;
  t_width : int;
  t_domain : (float * float) option;
  t_top : top;
  t_lo : float;
  t_hi : float;
  t_track : track;
  text : string;
}

let text t = t.text

let render ~width ~lo ~hi ~domain (t : track) =
  let d =
    match domain with Some d -> d | None -> widen (Curve.extent t.curve)
  in
  let fig =
    C.layer
      [
        C.line ~dots:`Braille
          ~color:(C.const Ansi.Color.default_fg)
          ~x:fst ~y:snd (Curve.points t.curve);
        C.axis ~show:false "x";
        C.axis ~show:false "y";
      ]
    |> C.scale "x" (C.Scale.linear ~domain:(lo, hi) ())
    |> C.scale "y" (C.Scale.linear ~domain:d ())
  in
  let g = G.create ~width ~height:1 () in
  C.draw (C.layout ~width ~height:1 fig) g ~x:0 ~y:0;
  Array.iter
    (fun (k, _) -> G.draw_text g ~x:(min (width - 1) (k / 2)) ~y:0 ~text:"×")
    (Curve.events t.curve);
  let s = G.to_text ~trim:false g in
  match String.index_opt s '\n' with Some i -> String.sub s 0 i | None -> s

let trend ?prev ?domain axis ~width h key =
  match series h key with
  | None -> None
  | Some s ->
      let prev =
        match prev with
        | Some p
          when p.t_axis = axis && p.t_width = width && p.t_domain = domain ->
            Some p
        | _ -> None
      in
      let t_top = top ?prev:(Option.map (fun p -> p.t_top) prev) axis h key s in
      let lo, hi =
        match axis with
        | Time -> (0., Float.max 1. t_top.top)
        | _ -> domain_of axis h [ t_top ]
      in
      let prev =
        match prev with
        | Some p when p.t_lo = lo && p.t_hi = hi -> Some p
        | _ -> None
      in
      let plot = { Curve.lo; hi; dots = 2 * width } in
      let t =
        track ?prev:(Option.map (fun p -> p.t_track) prev) axis plot h key s
      in
      Some
        {
          t_axis = axis;
          t_width = width;
          t_domain = domain;
          t_top;
          t_lo = lo;
          t_hi = hi;
          t_track = t;
          text = render ~width ~lo ~hi ~domain t;
        }
