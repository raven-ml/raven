(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Metric = Munin.Metric

type grid = { lo : float; hi : float; dots : int; dot : float -> int }
type event = [ `Nan | `Pos_inf | `Neg_inf ]

(* Sums that stay finite. A column's sum is kept scaled by 2^-shift: [shift]
   is 0 until a sum would pass the largest float, and then the sum is scaled
   down by a power of two, exactly. *)

let scale_step = 64

let rec plus sum shift x =
  let t = sum +. Float.ldexp x (-shift) in
  if Float.is_finite t then (t, shift)
  else plus (Float.ldexp sum (-scale_step)) (shift + scale_step) x

(* The values summed are finite, and so is their mean: clamping a rounding
   past the largest float only brings it closer. A NaN stays NaN. *)
let[@inline] clamp x =
  if x > Float.max_float then Float.max_float
  else if x < -.Float.max_float then -.Float.max_float
  else x

let[@inline] mean_of sum shift n = clamp (Float.ldexp (sum /. Float.of_int n) shift)

(* A column's finite values: their count, scaled sum, sum of logarithms, and
   how many are at or below 0, which have no logarithm. *)
type sums = {
  count : int array;
  sum : float array;
  shift : int array;
  logs : float array;
  nonpos : int array;
}

let sums dots =
  {
    count = Array.make dots 0;
    sum = Array.make dots 0.;
    shift = Array.make dots 0;
    logs = Array.make dots 0.;
    nonpos = Array.make dots 0;
  }

let copy_sums s =
  {
    count = Array.copy s.count;
    sum = Array.copy s.sum;
    shift = Array.copy s.shift;
    logs = Array.copy s.logs;
    nonpos = Array.copy s.nonpos;
  }

(* The functions a sample passes through are inlined, so that its floats stay
   unboxed: a call boxes each float it is given and returns. *)

let rec scale_sum s k y =
  s.sum.(k) <- Float.ldexp s.sum.(k) (-scale_step);
  s.shift.(k) <- s.shift.(k) + scale_step;
  let t = s.sum.(k) +. Float.ldexp y (-s.shift.(k)) in
  if Float.is_finite t then s.sum.(k) <- t else scale_sum s k y

let[@inline] add_sum s k y =
  let t = s.sum.(k) +. Float.ldexp y (-s.shift.(k)) in
  if Float.is_finite t then s.sum.(k) <- t else scale_sum s k y

let[@inline] add s k y =
  s.count.(k) <- s.count.(k) + 1;
  add_sum s k y;
  if y > 0. then s.logs.(k) <- s.logs.(k) +. Float.log y
  else s.nonpos.(k) <- s.nonpos.(k) + 1

(* The mean of column [k], with [extra] counted in it. *)
let column_mean s ~log k extra =
  let n = s.count.(k) + match extra with Some _ -> 1 | None -> 0 in
  if n = 0 then Float.nan
  else if log then
    let logs, nonpos =
      match extra with
      | Some y when y > 0. -> (s.logs.(k) +. Float.log y, s.nonpos.(k))
      | Some _ -> (s.logs.(k), s.nonpos.(k) + 1)
      | None -> (s.logs.(k), s.nonpos.(k))
    in
    if nonpos > 0 then Float.nan else Float.exp (logs /. Float.of_int n)
  else
    let sum, shift =
      match extra with
      | Some y -> plus s.sum.(k) s.shift.(k) y
      | None -> (s.sum.(k), s.shift.(k))
    in
    mean_of sum shift n

(* Smoothing

   The debiased mean by its recurrence. The step [x_i] of mean [y_i] weighs
   [(1 - exp (-(x_i - x_{i-1}) / tau)) exp (-(x - x_i) / tau)] at [x], the
   step before the first being [x_1 - (x_2 - x_1)], known at the second
   step. [den] sums the weights and [num] the weighted [(y_i - y_1) q], so
   that the smoothed value is [y_1 + num / (den q)]. The weights sum to at
   most 1, so [num] stays below twice the largest float times [q]: [q] is
   1, or 1/4 once a difference passes the largest float.

   A series is read sample by sample, so the state is mutated in place: its
   floats sit in a record of floats alone, which holds them unboxed, and a
   step's result is left in [value] and [closed_x] rather than returned. A
   reduction keeps a copy, which [extend] copies again before reading. *)

type mean = {
  tau : float;
  mutable y1 : float;
  mutable num : float;
  mutable den : float;
  mutable q : float;
  mutable last_x : float;  (* the last closed step *)
  mutable open_x : float;  (* the open step *)
  mutable open_sum : float;  (* its sum, scaled by 2^-[open_shift] *)
  mutable closed_x : float;  (* the step [feed] closed last *)
  mutable value : float;  (* its smoothed value *)
}

type counts = {
  mutable steps : int;  (* steps closed since the start *)
  mutable open_shift : int;
  mutable open_n : int;  (* the open step's samples; 0 when none is open *)
}

type ema = { m : mean; c : counts }

let ema_empty tau =
  {
    m =
      {
        tau;
        y1 = Float.nan;
        num = 0.;
        den = 0.;
        q = 1.;
        last_x = Float.nan;
        open_x = Float.nan;
        open_sum = 0.;
        closed_x = Float.nan;
        value = Float.nan;
      };
    c = { steps = 0; open_shift = 0; open_n = 0 };
  }

let copy_ema e = { m = { e.m with tau = e.m.tau }; c = { e.c with steps = e.c.steps } }

(* A lower step starts the mean again; the open step stays. *)
let restart e =
  e.m.y1 <- Float.nan;
  e.m.num <- 0.;
  e.m.den <- 0.;
  e.m.q <- 1.;
  e.m.last_x <- Float.nan;
  e.c.steps <- 0

(* [close e] closes the open step, at [x] of mean [y], its smoothed value
   in [e.m.value]. A step's weight is [1 - exp (-g / tau)] for its gap [g],
   which [expm1] gives accurately for small [g / tau]. *)
let rec close e =
  let m = e.m and c = e.c in
  let x = m.open_x in
  let y = mean_of m.open_sum c.open_shift c.open_n in
  if c.steps = 0 then begin
    m.y1 <- y;
    m.last_x <- x;
    c.steps <- 1;
    m.value <- y
  end
  else
    let q = m.q in
    let g = x -. m.last_x in
    let decay = exp (-.g /. m.tau) and a = -.Float.expm1 (-.g /. m.tau) in
    (* At the second step the first's weight is known: its gap is this one. *)
    let num = if c.steps = 1 then 0. else m.num in
    let den = if c.steps = 1 then a else m.den in
    let num = (num *. decay) +. (a *. ((q *. y) -. (q *. m.y1))) in
    let den = (den *. decay) +. a in
    let s = ((q *. m.y1) +. (num /. den)) /. q in
    if q = 1. && not (Float.is_finite num && Float.is_finite s) then begin
      m.q <- 0.25;
      m.num <- m.num *. 0.25;
      close e
    end
    else begin
      m.num <- num;
      m.den <- den;
      m.last_x <- x;
      c.steps <- c.steps + 1;
      m.value <- clamp s
    end

let rec scale_open m c y =
  m.open_sum <- Float.ldexp m.open_sum (-scale_step);
  c.open_shift <- c.open_shift + scale_step;
  let t = m.open_sum +. Float.ldexp y (-c.open_shift) in
  if Float.is_finite t then m.open_sum <- t else scale_open m c y

(* [feed e x y] adds the finite [y] at [x]: a new step closes the open one,
   whose x and smoothed value it leaves in [e.m.closed_x] and [e.m.value],
   and is [true] then; a lower step starts the mean again. *)
let[@inline] feed e x y =
  let m = e.m and c = e.c in
  if c.open_n > 0 && m.open_x = x then begin
    let t = m.open_sum +. Float.ldexp y (-c.open_shift) in
    if Float.is_finite t then m.open_sum <- t else scale_open m c y;
    c.open_n <- c.open_n + 1;
    false
  end
  else begin
    let closed = c.open_n > 0 in
    if closed then begin
      close e;
      m.closed_x <- m.open_x;
      if x < m.open_x then restart e
    end;
    m.open_x <- x;
    m.open_sum <- y;
    c.open_shift <- 0;
    c.open_n <- 1;
    closed
  end

(* The open step's value, as if a later step had come. *)
let provisional e =
  if e.c.open_n = 0 then None
  else
    let e' = copy_ema e in
    close e';
    Some (e.m.open_x, e'.m.value)

(* Reductions *)

(* The finite sample nearest the domain on one side: the greatest x on the
   left, the least on the right, the later of two at one x. [x] is NaN while
   there is none. *)
type side = { mutable x : float; mutable y : float }

let no_side () = { x = Float.nan; y = Float.nan }
let copy_side s = { x = s.x; y = s.y }

let[@inline] near ~left s x y =
  if Float.is_nan s.x || (left && x >= s.x) || ((not left) && x <= s.x) then begin
    s.x <- x;
    s.y <- y
  end

let side_opt s = if Float.is_nan s.x then None else Some (s.x, s.y)

type t = {
  grid : grid;
  smooth : bool;
  n : int;
  raw : sums;
  lo : float array;
  hi : float array;
  events : event option array;
  before : side;
  after : side;
  smoothed : sums;  (* the closed steps' smoothed values *)
  ema : ema;
  s_before : side;
  s_after : side;
}

let grid r = r.grid
let length r = r.n
let smoothed r = r.smooth

(* The dot column of [x]: [-1] left of the domain, [dots] right of it. *)
let[@inline] column (g : grid) x = if x < g.lo then -1 else if x > g.hi then g.dots else g.dot x

let mark r k y =
  if k >= 0 && k < r.grid.dots then
    match r.events.(k) with
    | Some `Nan -> ()
    | _ ->
        r.events.(k) <-
          (if Float.is_nan y then Some `Nan else if y > 0. then Some `Pos_inf else Some `Neg_inf)

(* The value [feed] closed, at its column or as an edge. *)
let[@inline] smooth_point r =
  let x = r.ema.m.closed_x and s = r.ema.m.value in
  let k = column r.grid x in
  if k < 0 then near ~left:true r.s_before x s
  else if k >= r.grid.dots then near ~left:false r.s_after x s
  else add r.smoothed k s

(* [read r ~x s i] adds sample [i] to [r], whose arrays and state no other
   reduction shares. *)
let read r ~x s i =
  let xi = x i and y = Metric.value s i in
  if Float.is_nan xi then ()
  else
    let k = column r.grid xi in
    if not (Float.is_finite y) then mark r k y
    else begin
      if k < 0 then near ~left:true r.before xi y
      else if k >= r.grid.dots then near ~left:false r.after xi y
      else begin
        add r.raw k y;
        (* As [Float.min] and [Float.max], [-0.] below [0.]. *)
        let lo = r.lo.(k) and hi = r.hi.(k) in
        if y < lo || (y = lo && Float.sign_bit y) then r.lo.(k) <- y;
        if y > hi || (y = hi && not (Float.sign_bit y)) then r.hi.(k) <- y
      end;
      if r.smooth && feed r.ema xi y then smooth_point r
    end

(* [fill r ~x s] is [r], whose arrays and state no other reduction shares,
   with the samples of [s] past those it read. *)
let fill r ~x s =
  for i = r.n to Metric.length s - 1 do
    read r ~x s i
  done;
  { r with n = Metric.length s }

let extend r ~x s =
  fill
    {
      r with
      raw = copy_sums r.raw;
      lo = Array.copy r.lo;
      hi = Array.copy r.hi;
      events = Array.copy r.events;
      before = copy_side r.before;
      after = copy_side r.after;
      smoothed = copy_sums r.smoothed;
      ema = copy_ema r.ema;
      s_before = copy_side r.s_before;
      s_after = copy_side r.s_after;
    }
    ~x s

let make ?tau grid ~x s =
  let dots = grid.dots in
  let smooth = tau <> None in
  fill
    {
      grid;
      smooth;
      n = 0;
      raw = sums dots;
      lo = Array.make dots Float.infinity;
      hi = Array.make dots Float.neg_infinity;
      events = Array.make dots None;
      before = no_side ();
      after = no_side ();
      smoothed = sums (if smooth then dots else 0);
      ema = ema_empty (Option.value tau ~default:1.);
      s_before = no_side ();
      s_after = no_side ();
    }
    ~x s

(* Reading *)

let mean r ~smooth ~log k =
  if not (smooth && r.smooth) then column_mean r.raw ~log k None
  else
    let extra =
      match provisional r.ema with
      | Some (x, s) when column r.grid x = k -> Some s
      | _ -> None
    in
    column_mean r.smoothed ~log k extra

let extent r k = if r.raw.count.(k) = 0 then None else Some (r.lo.(k), r.hi.(k))
let event r k = r.events.(k)

let edges r ~smooth =
  if not (smooth && r.smooth) then (side_opt r.before, side_opt r.after)
  else
    let before = copy_side r.s_before and after = copy_side r.s_after in
    (match provisional r.ema with
    | Some (x, s) ->
        let k = column r.grid x in
        if k < 0 then near ~left:true before x s
        else if k >= r.grid.dots then near ~left:false after x s
    | None -> ());
    (side_opt before, side_opt after)

(* The smoothed value reads every sample up to the latest at or before [v],
   those past [v] included: they close steps and start the mean again. No
   later sample shares that sample's step, since its x would be at most [v].
   A latest sample that is not finite is the value, smoothed or not. *)
let at ?tau ~x s v =
  let latest = ref (-1) in
  for i = 0 to Metric.length s - 1 do
    if x i <= v then latest := i
  done;
  if !latest < 0 then None
  else
    let y = Metric.value s !latest in
    match tau with
    | None -> Some y
    | Some _ when not (Float.is_finite y) -> Some y
    | Some tau ->
        let e = ema_empty tau in
        for i = 0 to !latest do
          let xi = x i and y = Metric.value s i in
          if (not (Float.is_nan xi)) && Float.is_finite y then ignore (feed e xi y)
        done;
        Option.map snd (provisional e)

(* Two series' latest common step *)

(* The scan holds each series' finite values by step, the last in log order
   at each. *)
let common_scan a b =
  let finite s =
    let t = Hashtbl.create (Metric.length s) in
    for i = 0 to Metric.length s - 1 do
      let v = Metric.value s i in
      if Float.is_finite v then Hashtbl.replace t (Metric.step s i) v
    done;
    t
  in
  let fa = finite a and fb = finite b in
  Hashtbl.fold
    (fun step va acc ->
      match (Hashtbl.find_opt fb step, acc) with
      | Some _, Some (s, _, _) when step <= s -> acc
      | Some vb, _ -> Some (step, va, vb)
      | None, _ -> acc)
    fa None

(* Ordered series are walked back from their ends: a step above the other's
   latest can be common with no earlier sample of the other, so it is passed.
   The first finite sample met at a step is its last in log order. *)
let common_walk a b =
  let rec go i j =
    if i < 0 || j < 0 then None
    else
      let va = Metric.value a i and vb = Metric.value b j in
      if not (Float.is_finite va) then go (i - 1) j
      else if not (Float.is_finite vb) then go i (j - 1)
      else
        let sa = Metric.step a i and sb = Metric.step b j in
        if sa = sb then Some (sa, va, vb) else if sa > sb then go (i - 1) j else go i (j - 1)
  in
  go (Metric.length a - 1) (Metric.length b - 1)

let common a b =
  if Metric.ordered a && Metric.ordered b then common_walk a b else common_scan a b
