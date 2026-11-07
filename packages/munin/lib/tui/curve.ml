(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Metric = Munin.Metric

type event = [ `Nan | `Pos_inf | `Neg_inf ]
type plot = { lo : float; hi : float; dots : int }

(* Matrix maps a value to [(v - lo) * (1 / (hi - lo))] and bins position [p] of
   an [l]-cell axis to [min (l - 1) ⌊p l⌋], with every position below one cell
   in the first (matrix_charts.ml, [positions] and [bin]). *)
let column p x =
  if x < p.lo then -1
  else if x > p.hi then p.dots
  else
    let q = (x -. p.lo) *. (1. /. (p.hi -. p.lo)) *. Float.of_int p.dots in
    if q < 1. then 0 else min (p.dots - 1) (Float.to_int q)

(* M4

   Per dot column, the samples of each stretch of finite values are reduced to
   the first, the least, the greatest and the last, in log order. A sample that
   is not finite or a step back closes the stretch and breaks the line with a
   NaN point, which matrix drops. The regions left and right of the domain are a
   column each, so the segments that cross the plot's edges keep their ends. *)

type point = { x : float; y : float; i : int }

type stretch = {
  col : int;
  first : point;
  least : point;
  greatest : point;
  last : point;
}

type m4 = {
  out : (float * float) list; (* emitted points, newest first *)
  cur : stretch option;
  prev_x : float; (* x of the last sample fed, NaN before any *)
}

let m4_empty = { out = []; cur = None; prev_x = Float.nan }

let emit_stretch out s =
  let mid =
    if s.least.i <= s.greatest.i then [ s.least; s.greatest ]
    else [ s.greatest; s.least ]
  in
  (* A sample that is both the first and the least, say, is emitted once. *)
  let rec distinct = function
    | a :: (b :: _ as rest) when a.i = b.i -> distinct rest
    | a :: rest -> a :: distinct rest
    | [] -> []
  in
  List.fold_left
    (fun out p -> (p.x, p.y) :: out)
    out
    (distinct ((s.first :: mid) @ [ s.last ]))

let close m = match m.cur with None -> m.out | Some s -> emit_stretch m.out s

let break out =
  match out with
  | [] -> []
  | (_, y) :: _ when Float.is_nan y -> out
  | out -> (Float.nan, Float.nan) :: out

(* [feed plot m p] adds the sample [p], whose x is not NaN. *)
let feed plot m p =
  let back = p.x < m.prev_x in
  if not (Float.is_finite p.y) then
    { out = break (close m); cur = None; prev_x = p.x }
  else if back then
    let s =
      { col = column plot p.x; first = p; least = p; greatest = p; last = p }
    in
    { out = break (close m); cur = Some s; prev_x = p.x }
  else
    let col = column plot p.x in
    match m.cur with
    | Some s when s.col = col ->
        let least = if p.y < s.least.y then p else s.least in
        let greatest = if p.y > s.greatest.y then p else s.greatest in
        { m with cur = Some { s with least; greatest; last = p }; prev_x = p.x }
    | _ ->
        let s = { col; first = p; least = p; greatest = p; last = p } in
        { out = close m; cur = Some s; prev_x = p.x }

let m4_points m = Array.of_list (List.rev (close m))

(* Smoothing

   The debiased mean of the Charts section, by its recurrence. Samples at one
   step are pooled into their mean, so a step's smoothed value is known once a
   later step arrives; until then it is [pending]. [x0] is the step before the
   first, [x_1 - (x_2 - x_1)], known at the second step. A step back starts the
   mean again. *)

type ema = {
  tau : float;
  y1 : float; (* the first step's mean *)
  num : float; (* in units of [q] *)
  den : float;
  q : float; (* 1, or 1/4 once a difference passes the largest float *)
  last_x : float; (* the last closed step, NaN before any *)
  closed : int; (* closed steps since the start *)
  pending : (float * float * int * int * int) option;
      (* step, sum scaled by 2^-shift, shift, count, index of its last sample
         ([Mean]) *)
}

let ema_empty tau =
  {
    tau;
    y1 = Float.nan;
    num = 0.;
    den = 0.;
    q = 1.;
    last_x = Float.nan;
    closed = 0;
    pending = None;
  }

(* [weight tau g] is [1 - exp (-g / tau)], accurate for small [g / tau]. *)
let weight tau g = -.Float.expm1 (-.g /. tau)

(* [close_step e (x, y)] is [e] with the step [x] of mean [y] closed, and the
   step's smoothed value. Where a difference, or the value, would pass the
   largest float, the sum goes on in quarters: a quarter of a difference of two
   finite values is at most half the largest float, and the weights that [den]
   sums add up to at most 1, so [num] stays below it too. The value is clamped
   to the finite floats, as a weighted mean of finite values is. *)
let rec close_step e x y =
  if e.closed = 0 then ({ e with y1 = y; last_x = x; closed = 1 }, y)
  else
    let q = e.q in
    let g = x -. e.last_x in
    let decay = exp (-.g /. e.tau) and a = weight e.tau g in
    (* At the second step the first's weight is known: its gap is this one. *)
    let num, den = if e.closed = 1 then (0., a) else (e.num, e.den) in
    let num = (num *. decay) +. (a *. ((q *. y) -. (q *. e.y1))) in
    let den = (den *. decay) +. a in
    let s = ((q *. e.y1) +. (num /. den)) /. q in
    if q = 1. && not (Float.is_finite num && Float.is_finite s) then
      close_step { e with q = 0.25; num = e.num *. 0.25 } x y
    else
      ( { e with num; den; last_x = x; closed = e.closed + 1 },
        Float.min Float.max_float (Float.max (-.Float.max_float) s) )

(* Curves *)

type t = {
  plot : plot;
  ema : ema option;
  n : int; (* samples read *)
  m4 : m4;
  sums : float array;
      (* per dot column, the sum of finite values scaled by 2^-shift ([Mean]) *)
  shifts : int array;
  counts : int array;
  events : event option array;
  extent : (float * float) option;
}

let in_plot plot c = c >= 0 && c < plot.dots

(* [add_value c p] is [c] with [p] reduced and counted in its column. *)
let add_value c p =
  let col = column c.plot p.x in
  let m4 = feed c.plot c.m4 p in
  if Float.is_finite p.y && in_plot c.plot col then begin
    Mean.add c.sums c.shifts col p.y;
    c.counts.(col) <- c.counts.(col) + 1;
    let extent =
      match c.extent with
      | None -> Some (p.y, p.y)
      | Some (lo, hi) -> Some (Float.min lo p.y, Float.max hi p.y)
    in
    { c with m4; extent }
  end
  else { c with m4 }

let mark c x y =
  let col = column c.plot x in
  if in_plot c.plot col then
    let e =
      if Float.is_nan y then `Nan else if y > 0. then `Pos_inf else `Neg_inf
    in
    match c.events.(col) with Some `Nan -> () | _ -> c.events.(col) <- Some e

(* [add c ~x s i] reads the sample [i] of [s]. The arrays of [c] are its own
   copies, written in place. *)
let add c ~x s i =
  let xi = x i and y = Metric.value s i in
  if Float.is_nan xi then c
  else begin
    if not (Float.is_finite y) then mark c xi y;
    match c.ema with
    | None -> add_value c { x = xi; y; i }
    | Some e -> (
        if not (Float.is_finite y) then c
        else
          match e.pending with
          | Some (px, sum, shift, n, _) when px = xi ->
              let sum, shift = Mean.plus sum shift y in
              let pending = Some (px, sum, shift, n + 1, i) in
              { c with ema = Some { e with pending } }
          | Some (px, sum, shift, n, pi) ->
              let e, s = close_step e px (Mean.get sum shift n) in
              let c = add_value c { x = px; y = s; i = pi } in
              let e = if xi < px then ema_empty e.tau else e in
              { c with ema = Some { e with pending = Some (xi, y, 0, 1, i) } }
          | None ->
              { c with ema = Some { e with pending = Some (xi, y, 0, 1, i) } })
  end

let copy c =
  {
    c with
    sums = Array.copy c.sums;
    shifts = Array.copy c.shifts;
    counts = Array.copy c.counts;
    events = Array.copy c.events;
  }

let extend c ~x s =
  let c = ref (copy c) in
  for i = !c.n to Metric.length s - 1 do
    c := add !c ~x s i
  done;
  { !c with n = Metric.length s }

let reduce ?tau plot ~x s =
  let c =
    {
      plot;
      ema = Option.map ema_empty tau;
      n = 0;
      m4 = m4_empty;
      sums = Array.make plot.dots 0.;
      shifts = Array.make plot.dots 0;
      counts = Array.make plot.dots 0;
      events = Array.make plot.dots None;
      extent = None;
    }
  in
  extend c ~x s

(* The pending step of a smoothed curve, closed as if a later step had come. *)
let settled c =
  match c.ema with
  | Some ({ pending = Some (px, sum, shift, n, pi); _ } as e) ->
      let c = copy c in
      let _, s = close_step e px (Mean.get sum shift n) in
      add_value c { x = px; y = s; i = pi }
  | _ -> c

let points c = m4_points (settled c).m4
let extent c = (settled c).extent

let level c =
  let c = settled c in
  let pts = ref [] in
  for k = c.plot.dots - 1 downto 0 do
    if c.counts.(k) > 0 then
      let x =
        c.plot.lo
        +. (Float.of_int k +. 0.5)
           /. Float.of_int c.plot.dots *. (c.plot.hi -. c.plot.lo)
      in
      pts := (x, Mean.get c.sums.(k) c.shifts.(k) c.counts.(k)) :: !pts
  done;
  Array.of_list !pts

let events c =
  let l = ref [] in
  for k = c.plot.dots - 1 downto 0 do
    match c.events.(k) with Some e -> l := (k, e) :: !l | None -> ()
  done;
  Array.of_list !l
