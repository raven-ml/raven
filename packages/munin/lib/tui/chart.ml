(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin
module C = Matrix_charts
module G = Matrix_grid

type axis = Step | Step_metric | Time

type view = {
  axis : axis;
  smooth : int;
  log : bool;
  window : (float * float) option;
}

type spec = { key : string; width : int; height : int; x_axis : bool }
type drawn = { log : bool; range : (float * float) option }

let palette = function
  | 'a' -> Ansi.Color.cyan
  | 'b' -> Ansi.Color.magenta
  | _ -> Ansi.Color.yellow

(* The width of the y labels' column: margins that the labels of matrix's ticks,
   which take an offset or a factor note past seven digits, fit in, so that the
   plot's width depends on the region alone. *)
let left = 8

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
}

let lo p = p.lo
let hi p = p.hi
let tau p = p.tau
let unpaired p = p.unpaired
let latest p = p.latest
let spec p = p.spec

let figure ~x_axis ~lo ~hi ~y ~log ~resumes ~cursor ~marked ~raw ~focus =
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
        C.line ~dots:`Braille ~dash:`Dotted
          ~color:(C.const (palette l))
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
  let ys =
    if log then C.Scale.log ~domain:y () else C.Scale.linear ~domain:y ()
  in
  C.layer
    (resume_lines @ cursor @ marked @ raw
    @ [
        focus;
        (if x_axis then C.axis "x" else C.axis ~show:false "x");
        C.axis "y";
      ])
  |> C.scale "x" (C.Scale.linear ~domain:(lo, hi) ())
  |> C.scale "y" ys |> C.margins ~left

let layout ?cursor ~light p =
  let focus_pts, raw =
    match (p.smooth, p.focus) with
    | Some s, f ->
        (Curve.points s.curve, Option.map (fun t -> Curve.points t.curve) f)
    | None, Some f -> (Curve.points f.curve, None)
    | None, None -> ([||], None)
  in
  let marked = List.map (fun (l, _, t) -> (l, Curve.level t.curve)) p.marked in
  let fig =
    figure ~x_axis:p.spec.x_axis ~lo:p.lo ~hi:p.hi ~y:p.y ~log:p.log
      ~resumes:p.resumes ~cursor ~marked ~raw ~focus:focus_pts
  in
  let theme = if light then C.Theme.light else C.Theme.dark in
  (C.layout ~theme ~width:p.spec.width ~height:(p.spec.height - 1) fig, marked)

(* The dot columns of a plot in a region, which the margins fix. *)
let dots spec =
  let fig =
    figure ~x_axis:spec.x_axis ~lo:0. ~hi:1. ~y:(0., 1.) ~log:false ~resumes:[]
      ~cursor:None ~marked:[] ~raw:None ~focus:[||]
  in
  2
  * (C.Layout.plot (C.layout ~width:spec.width ~height:(spec.height - 1) fig))
      .width

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
  let dots = max 2 (dots spec) in
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
  }

let draw ?cursor ~light p g ~x ~y =
  let width = p.spec.width and rows = p.spec.height - 1 in
  if width < 2 || rows < 2 then { log = false; range = None }
  else begin
    let l, levels = layout ?cursor ~light p in
    C.draw l g ~x ~y:(y + 1);
    let plot = C.Layout.plot l in
    (* Letters, at the end of each level line and in cells along it that hold no
       dot of another line. *)
    let cell (px, py) =
      ( C.Layout.position l "x" (C.Layout.Num px),
        C.Layout.position l "y" (C.Layout.Num py) )
    in
    (* A letter along a line goes where no other line inks; a line's end letter
       may also replace another marked run's ink, never the focus's. *)
    let marked_ink i =
      List.exists (fun (c, _) -> G.get_fg g i = palette c) levels
    in
    let letter ?(at_end = false) c (cx, cy) =
      let gx = x + cx and gy = y + 1 + cy in
      if gx >= x && gx < x + width && gy > y && gy <= y + rows then begin
        let i = G.idx g ~x:gx ~y:gy in
        let blank = G.is_empty g i || G.get_text g i = " " in
        if blank || G.get_fg g i = palette c || (at_end && marked_ink i) then
          G.draw_text
            ~style:(Ansi.Style.make ~fg:(palette c) ())
            g ~x:gx ~y:gy ~text:(String.make 1 c)
      end
    in
    List.iter
      (fun (c, level) ->
        let n = Array.length level in
        if n > 0 then begin
          List.iter
            (fun k -> letter c (cell level.(k)))
            [ n / 4; n / 2; 3 * n / 4 ];
          let cx, cy = cell level.(n - 1) in
          let cx = if x + cx + 1 < x + width then cx + 1 else cx in
          letter ~at_end:true c (cx, cy)
        end)
      levels;
    (* The focus's samples that are not finite, in the row above the plot. *)
    Option.iter
      (fun t ->
        Array.iter
          (fun (k, e) ->
            let glyph =
              match e with `Nan -> "×" | `Pos_inf -> "↑" | `Neg_inf -> "↓"
            in
            let gx = x + plot.x + (k / 2) in
            let i = G.idx g ~x:gx ~y in
            if not ((not (G.is_empty g i)) && G.get_text g i = "×") then
              G.draw_text
                ~style:(Ansi.Style.make ~fg:Ansi.Color.red ())
                g ~x:gx ~y ~text:glyph)
          (Curve.events t.curve))
      p.focus;
    (* A plot too short for two ticks states its range. *)
    { log = p.log; range = (if plot.height < 4 then Some p.y else None) }
  end

(* Trends *)

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
