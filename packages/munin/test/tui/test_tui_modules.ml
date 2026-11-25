(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The modules behind the dashboard's stop, whose laws the dashboard's
   interface leaves to them: the reduction, the drawing of a chart, the
   numbers, the facts of a run and the watch of its arrivals. *)

open Windtrap
open Munin
module Reduce = Munin_tui.Reduce
module Plot = Munin_tui.Plot
module Watch = Munin_tui.Watch
module Numbers = Munin_tui.Numbers

let rec remove path =
  match (Unix.lstat path).st_kind with
  | Unix.S_DIR ->
      Array.iter (fun e -> remove (Filename.concat path e)) (Sys.readdir path);
      Unix.rmdir path
  | _ -> Sys.remove path
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> ()

(* The reference for round numbers: the C library's reading of the decimal
   [m] e [k], correctly rounded. *)
let decimal_text m k = float_of_string (Printf.sprintf "%de%d" m k)

(* The store of the runs the laws write, one or two a case, which the
   process removes as it ends. *)
let laws =
  lazy
    (let dir = Filename.temp_dir "munin_tui" "" in
     at_exit (fun () -> remove dir);
     Store.open_ ~root:(Filename.concat dir "laws") ())

(* Laws of the reduction *)

(* [pairs l] is each sample of [l] with the next. *)
let pairs l =
  match l with
  | [] -> []
  | _ :: rest ->
      List.combine (List.filteri (fun k _ -> k < List.length rest) l) rest

(* The laws' runs, written from the format's text. *)

(* [series_of samples] is the series of a run that logged [samples], each a step
   and a value, under one key. *)
let series_of samples =
  let store = Lazy.force laws in
  let id, _ = Fixture.write_log store (List.map Fixture.sample samples) in
  let h = History.read ~series:[ "m" ] (Option.get (Store.run store id)) in
  History.series h "m"

(* Samples: steps that mostly increase, sometimes repeat and now and then
   decrease, and values among which NaN, the infinities and -0. appear. *)
let gen_value =
  Gen.frequency
    [
      (20, Gen.float_range (-50.) 50.);
      (1, Gen.constant Float.nan);
      (1, Gen.constant Float.infinity);
      (1, Gen.constant Float.neg_infinity);
      (1, Gen.constant (-0.));
    ]

(* Values whose sums and differences pass the largest float, among the rest. *)
let gen_big_value =
  Gen.frequency
    [
      (6, gen_value);
      (1, Gen.constant Float.max_float);
      (1, Gen.constant (-.Float.max_float));
    ]

(* Values above 0, so that every mean has a logarithm. *)
let gen_positive_value =
  Gen.frequency
    [ (8, Gen.float_range 1e-3 1e3); (1, Gen.constant Float.nan) ]

let gen_samples ?(value = gen_value) () =
  let open Gen in
  let* deltas =
    list ~size:(int_range 0 400)
      (frequency
         [ (12, int_range 1 30); (3, constant 0); (1, int_range (-200) (-1)) ])
  in
  let* values = list ~size:(constant (List.length deltas)) value in
  let steps =
    List.rev
      (snd
         (List.fold_left
            (fun (s, acc) d ->
              let s = max 0 (s + d) in
              (s, s :: acc))
            (0, []) deltas))
  in
  constant (List.combine steps values)

(* A grid over samples: the whole of their steps or a window of them, some
   dot columns, whether every fifth sample, from the third, has a NaN x, and
   smoothing's time constant. *)
type grid_case = { lo : float; hi : float; dots : int; holes : bool; tau : float }

let gen_grid samples =
  let open Gen in
  let last = List.fold_left (fun m (s, _) -> max m s) 0 samples in
  (* A third of the grids have few columns, so that a column holds many
     samples. *)
  let* dots = frequency [ (1, int_range 1 32); (2, int_range 1 320) ] in
  let* holes = frequency [ (3, constant false); (1, constant true) ] in
  let* tau = float_range 0.5 500. in
  let* window = bool in
  let* lo = if window then int_range 0 (max 0 (last - 1)) else constant 0 in
  let+ hi =
    if window then int_range (lo + 1) (last + 1) else constant (last + 1)
  in
  { lo = Float.of_int lo; hi = Float.of_int hi; dots; holes; tau }

let pp_samples ppf samples =
  Format.pp_print_list ~pp_sep:Format.pp_print_space
    (fun ppf (st, v) -> Format.fprintf ppf "%d:%h" st v)
    ppf samples

let pp_case ppf (samples, g) =
  Format.fprintf ppf "@[%a@ %d dots over [%g, %g]%s, tau %g@]" pp_samples
    samples g.dots g.lo g.hi
    (if g.holes then " with holes" else "")
    g.tau

let gen_case ?value () =
  Gen.(
    with_pp pp_case
      (let* samples = gen_samples ?value () in
       let+ g = gen_grid samples in
       (samples, g)))

let hole g i = g.holes && i mod 5 = 2

(* [points g samples] is each sample's x and value. *)
let points g samples =
  List.mapi
    (fun i (st, v) -> ((if hole g i then Float.nan else Float.of_int st), v))
    samples

let column g x =
  min (g.dots - 1)
    (Float.to_int ((x -. g.lo) *. (1. /. (g.hi -. g.lo)) *. Float.of_int g.dots))

let in_domain g x = x >= g.lo && x <= g.hi

(* [x_of g s i] is the x of [s]'s sample [i]: its step, or NaN for a hole. *)
let x_of g s i = if hole g i then Float.nan else Float.of_int (Metric.step s i)

let reduce ?(smooth = true) g s =
  let grid = { Reduce.lo = g.lo; hi = g.hi; dots = g.dots; dot = column g } in
  Reduce.make ?tau:(if smooth then Some g.tau else None) grid ~x:(x_of g s) s

let event_name = function
  | Some `Nan -> "NaN"
  | Some `Pos_inf -> "+inf"
  | Some `Neg_inf -> "-inf"
  | None -> "-"

(* [half_mean l] is half the mean of [l], each value divided by the count and
   by 2 first, so that no partial sum's rounding passes the largest float. *)
let half_mean l =
  let n = Float.of_int (List.length l) in
  List.fold_left (fun a v -> a +. (v /. n /. 2.)) 0. l

let geometric l =
  if List.exists (fun v -> not (v > 0.)) l then None
  else
    let n = Float.of_int (List.length l) in
    Some (exp (List.fold_left (fun a v -> a +. (log v /. n)) 0. l))

(* [by_column g pts] is the finite values of each dot column. *)
let by_column g pts =
  let columns = Array.make g.dots [] in
  List.iter
    (fun (x, v) ->
      if Float.is_finite v && in_domain g x then
        let k = column g x in
        columns.(k) <- v :: columns.(k))
    pts;
  columns

(* The smoothing of the module's doc, from its definition: the finite samples
   pooled into steps of one x, started again where x decreases, each step's
   value the mean of the steps up to it, the step [x_i] weighted by
   [(1 - exp (-(x_i - x_(i-1)) / tau)) exp (-(x - x_i) / tau)], the step
   before the first [x_1 - (x_2 - x_1)]. Each step is its x, its smoothed
   value quartered, so that no sum passes the largest float, and the indices
   of its samples. *)
let smoothed_steps ~tau pts =
  let groups =
    List.rev
      (List.fold_left
         (fun acc (i, (x, v)) ->
           if Float.is_nan x || not (Float.is_finite v) then acc
           else
             match acc with
             | (x', vs, is) :: rest when x' = x -> (x, v :: vs, i :: is) :: rest
             | _ -> (x, [ v ], [ i ]) :: acc)
         []
         (List.mapi (fun i p -> (i, p)) pts))
  in
  (* The segments, each the steps from one start, oldest first. *)
  let segments =
    List.rev
      (List.fold_left
         (fun acc ((x, _, _) as step) ->
           match acc with
           | ((px, _, _) :: _ as seg) :: rest when x >= px -> (step :: seg) :: rest
           | _ -> [ step ] :: acc)
         [] groups)
    |> List.map List.rev
  in
  let smooth seg =
    let xs = Array.of_list (List.map (fun (x, _, _) -> x) seg) in
    let ys = Array.of_list (List.map (fun (_, vs, _) -> half_mean vs /. 2.) seg) in
    let x0 = if Array.length xs > 1 then xs.(0) -. (xs.(1) -. xs.(0)) else xs.(0) -. 1. in
    List.mapi
      (fun k (x, _, is) ->
        let w =
          Array.init (k + 1) (fun i ->
              let prev = if i = 0 then x0 else xs.(i - 1) in
              -.Float.expm1 (-.(xs.(i) -. prev) /. tau) *. exp (-.(x -. xs.(i)) /. tau))
        in
        let den = Array.fold_left ( +. ) 0. w in
        let s = ref 0. in
        Array.iteri (fun i wi -> s := !s +. (wi /. den *. ys.(i))) w;
        (x, !s, is))
      seg
  in
  List.concat_map smooth segments

(* A key's finite values whose sum passes the largest float. *)
let sum_passes vs = not (Float.is_finite (List.fold_left ( +. ) 0. vs))

let scale_of samples =
  List.fold_left
    (fun m (_, v) -> if Float.is_finite v then Float.max m (Float.abs v) else m)
    0. samples

let close scale = float_rel ~rel:1e-9 ~abs:(Float.max 1e-12 (1e-9 *. scale))
let defined v = if Float.is_nan v then None else Some v

let cover_shapes samples g =
  cover "a window cuts the samples"
    (List.exists
       (fun (st, _) -> not (in_domain g (Float.of_int st)))
       samples);
  cover "steps decrease"
    (List.exists (fun ((a, _), (b, _)) -> b < a) (pairs samples));
  cover "samples share a step"
    (List.exists (fun ((a, _), (b, _)) -> a = b) (pairs samples));
  cover "a dot column holds many samples" (List.length samples > 4 * g.dots);
  cover "a sample's x is NaN" (g.holes && List.length samples > 2)

let means =
  prop "a column's mean is the mean of its finite samples" (gen_case ~value:gen_big_value ())
    (fun (samples, g) ->
      cover_shapes samples g;
      let s = series_of samples in
      let r = reduce g s in
      let columns = by_column g (points g samples) in
      cover "a column's values sum past the largest float"
        (Array.exists sum_passes columns);
      cover "a column holds a value at or below 0 and one above"
        (Array.exists
           (fun vs ->
             List.exists (fun v -> v <= 0.) vs && List.exists (fun v -> v > 0.) vs)
           columns);
      equal int (List.length samples) (Reduce.length r);
      Array.iteri
        (fun k vs ->
          let msg = Printf.sprintf "column %d" k in
          let scale = scale_of (List.map (fun v -> (0, v)) vs) in
          equal ~msg (option (close scale))
            (if vs = [] then None else Some (half_mean vs))
            (Option.map
               (fun m -> m /. 2.)
               (defined (Reduce.mean r ~smooth:false ~log:false k)));
          equal ~msg (option (float_rel ~rel:1e-9 ~abs:0.))
            (if vs = [] then None else geometric vs)
            (defined (Reduce.mean r ~smooth:false ~log:true k));
          equal ~msg
            (option (pair float_exact float_exact))
            (if vs = [] then None
             else
               Some
                 ( List.fold_left Float.min Float.infinity vs,
                   List.fold_left Float.max Float.neg_infinity vs ))
            (Reduce.extent r k))
        columns)

(* The smoothed values in each column, quartered. *)
let smoothed_columns g pts =
  let columns = Array.make g.dots [] in
  List.iter
    (fun (x, s, _) ->
      if in_domain g x then
        let k = column g x in
        columns.(k) <- s :: columns.(k))
    (smoothed_steps ~tau:g.tau pts);
  columns

let smoothed_means =
  prop "a column's smoothed mean is the mean of its steps' smoothed values"
    (gen_case ~value:gen_big_value ())
    (fun (samples, g) ->
      cover_shapes samples g;
      let s = series_of samples in
      let r = reduce g s in
      let columns = smoothed_columns g (points g samples) in
      cover "a step's mean differs from another's past the largest float"
        (List.exists (fun (_, v) -> v = Float.max_float) samples
        && List.exists (fun (_, v) -> v = -.Float.max_float) samples);
      let scale = scale_of samples in
      Array.iteri
        (fun k ss ->
          let n = Float.of_int (List.length ss) in
          equal
            ~msg:(Printf.sprintf "column %d" k)
            (option (close (scale /. 4.)))
            (if ss = [] then None
             else Some (List.fold_left (fun a v -> a +. (v /. n)) 0. ss))
            (Option.map
               (fun m -> m /. 4.)
               (defined (Reduce.mean r ~smooth:true ~log:false k))))
        columns)

let geometric_means =
  prop "a column's geometric mean is that of its finite samples, smoothed or not"
    (gen_case ~value:gen_positive_value ())
    (fun (samples, g) ->
      cover_shapes samples g;
      let s = series_of samples in
      let r = reduce g s in
      let pts = points g samples in
      let quartered = Array.map (List.map (fun v -> v *. 4.)) (smoothed_columns g pts) in
      Array.iteri
        (fun k vs ->
          let msg = Printf.sprintf "column %d" k in
          equal ~msg (option (float_rel ~rel:1e-9 ~abs:0.))
            (if vs = [] then None else geometric vs)
            (defined (Reduce.mean r ~smooth:false ~log:true k));
          equal ~msg (option (float_rel ~rel:1e-9 ~abs:0.))
            (if quartered.(k) = [] then None else geometric quartered.(k))
            (defined (Reduce.mean r ~smooth:true ~log:true k)))
        (by_column g pts))

let unsmoothed =
  prop "a reduction without a smoothed form reads its raw means"
    (gen_case ())
    (fun (samples, g) ->
      let s = series_of samples in
      let r = reduce ~smooth:false g s in
      equal bool false (Reduce.smoothed r);
      for k = 0 to g.dots - 1 do
        List.iter
          (fun log ->
            equal float_exact
              (Reduce.mean r ~smooth:false ~log k)
              (Reduce.mean r ~smooth:true ~log k))
          [ false; true ]
      done)

let events =
  let one = { lo = 0.; hi = 2.; dots = 1; holes = false; tau = 1. } in
  prop "each dot column with a sample that is not finite is marked, NaN first"
    ~examples:
      [
        ([ (0, Float.infinity); (0, Float.nan); (1, 1.) ], one);
        ([ (0, Float.nan); (1, Float.neg_infinity) ], one);
      ]
    (gen_case ())
    (fun (samples, g) ->
      let s = series_of samples in
      let r = reduce g s in
      let expected = Array.make g.dots None in
      List.iter
        (fun (x, v) ->
          if (not (Float.is_finite v)) && in_domain g x then
            let k = column g x in
            let e =
              if Float.is_nan v then `Nan
              else if v > 0. then `Pos_inf
              else `Neg_inf
            in
            match expected.(k) with
            | Some `Nan -> ()
            | _ -> expected.(k) <- Some e)
        (points g samples);
      let nan = Array.make g.dots false and inf = Array.make g.dots false in
      List.iter
        (fun (x, v) ->
          if in_domain g x && not (Float.is_finite v) then
            let k = column g x in
            if Float.is_nan v then nan.(k) <- true else inf.(k) <- true)
        (points g samples);
      cover "a column holds NaN and an infinity"
        (Array.exists2 ( && ) nan inf);
      equal (array string)
        (Array.map event_name expected)
        (Array.init g.dots (fun k -> event_name (Reduce.event r k))))

(* [nearest ~left l] is the point of [l], in log order, nearest the domain on
   one side: the greatest x on the left, the least on the right, the later of
   two at one x. *)
let nearest ~left l =
  List.fold_left
    (fun best (x, y) ->
      match best with
      | Some (bx, _) when (left && x < bx) || ((not left) && x > bx) -> best
      | _ -> Some (x, y))
    None l

let edges =
  prop "the edges are the finite samples nearest the domain on each side"
    (gen_case ~value:gen_big_value ())
    (fun (samples, g) ->
      let s = series_of samples in
      let r = reduce g s in
      let pts = points g samples in
      let finite =
        List.filter (fun (x, v) -> (not (Float.is_nan x)) && Float.is_finite v) pts
      in
      let left = List.filter (fun (x, _) -> x < g.lo) finite in
      let right = List.filter (fun (x, _) -> x > g.hi) finite in
      cover "a sample lies left of the domain" (left <> []);
      cover "a sample lies right of the domain" (right <> []);
      let side = option (pair float_exact float_exact) in
      let raw_left, raw_right = Reduce.edges r ~smooth:false in
      equal ~msg:"raw left" side (nearest ~left:true left) raw_left;
      equal ~msg:"raw right" side (nearest ~left:false right) raw_right;
      let steps = List.map (fun (x, s, _) -> (x, s)) (smoothed_steps ~tau:g.tau pts) in
      let quarter = Option.map (fun (x, y) -> (x, y /. 4.)) in
      let side = option (pair float_exact (close (scale_of samples /. 4.))) in
      let smooth_left, smooth_right = Reduce.edges r ~smooth:true in
      equal ~msg:"smoothed left" side
        (nearest ~left:true (List.filter (fun (x, _) -> x < g.lo) steps))
        (quarter smooth_left);
      equal ~msg:"smoothed right" side
        (nearest ~left:false (List.filter (fun (x, _) -> x > g.hi) steps))
        (quarter smooth_right))

(* A case for [at]: a third of the time its series ends with a sample that is
   not finite, at the latest step, and the x is at or past that step. *)
let gen_at_case =
  Gen.(
    with_pp
      (fun ppf ((samples, g), v) -> Format.fprintf ppf "%a@ at %g" pp_case (samples, g) v)
      (let* samples, g = gen_case ~value:gen_big_value () in
       let last = List.fold_left (fun m (s, _) -> max m s) 0 samples in
       let* ends = frequency [ (2, constant false); (1, constant true) ] in
       if ends then
         let* y = one_of [ constant Float.nan; constant Float.infinity; constant Float.neg_infinity ] in
         let+ v = int_range last (last + 1) in
         ((samples @ [ (last, y) ], g), Float.of_int v)
       else
         let+ v = int_range (-1) (last + 1) in
         ((samples, g), Float.of_int v)))

let at =
  prop "a value at an x is that of the latest sample at or before it"
    gen_at_case
    (fun ((samples, g), v) ->
      let s = series_of samples in
      let x = x_of g s in
      let pts = points g samples in
      let latest =
        List.fold_left
          (fun acc (i, (xi, _)) -> if xi <= v then Some i else acc)
          None
          (List.mapi (fun i p -> (i, p)) pts)
      in
      let value = Option.map (fun i -> snd (List.nth pts i)) latest in
      cover "a sample before it in the log lies past the x"
        (match latest with
        | Some i -> List.exists (fun (xi, _) -> xi > v) (List.filteri (fun j _ -> j < i) pts)
        | None -> false);
      cover "the latest sample is not finite"
        (match value with Some y -> not (Float.is_finite y) | None -> false);
      equal ~msg:"raw" (option float_exact) value (Reduce.at ~x s v);
      match value with
      | Some y when not (Float.is_finite y) ->
          equal ~msg:"smoothed" (option float_exact) value (Reduce.at ~tau:g.tau ~x s v)
      | _ ->
          let step =
            Option.map
              (fun i ->
                let _, s, _ =
                  List.find (fun (_, _, is) -> List.mem i is) (smoothed_steps ~tau:g.tau pts)
                in
                s)
              latest
          in
          equal ~msg:"smoothed"
            (option (close (scale_of samples /. 4.)))
            step
            (Option.map (fun y -> y /. 4.) (Reduce.at ~tau:g.tau ~x s v)))

(* Two series' latest common step: their samples, each sorted by step half
   the time, so that pairs of ordered series and series out of order are both
   drawn often. *)
let gen_sorted_samples =
  Gen.(
    let* samples = gen_samples () in
    let+ sorted = bool in
    if sorted then List.stable_sort (fun (a, _) (b, _) -> Int.compare a b) samples else samples)

let ordered_steps samples =
  let rec go = function a :: (b :: _ as rest) -> a <= b && go rest | _ -> true in
  go (List.map fst samples)

let common =
  prop "the latest common step is the largest step both have a finite sample at"
    Gen.(
      with_pp
        (fun ppf (a, b) -> Format.fprintf ppf "@[%a@ and %a@]" pp_samples a pp_samples b)
        (pair gen_sorted_samples gen_sorted_samples))
    (fun (a, b) ->
      let finite l = List.filter (fun (_, v) -> Float.is_finite v) l in
      let last_at l s = snd (List.find (fun (t, _) -> t = s) (List.rev (finite l))) in
      let both = List.filter (fun (s, _) -> List.mem_assoc s (finite b)) (finite a) in
      let expected =
        match List.map fst both with
        | [] -> None
        | steps ->
            let s = List.fold_left max min_int steps in
            Some (s, (last_at a s, last_at b s))
      in
      let ordered = ordered_steps a && ordered_steps b in
      cover "both series are ordered, with a common step" (ordered && expected <> None);
      cover "a series is out of order, with a common step" ((not ordered) && expected <> None);
      let sa = series_of a and sb = series_of b in
      equal ~msg:"ordered" (pair bool bool) (ordered_steps a, ordered_steps b)
        (Metric.ordered sa, Metric.ordered sb);
      equal (option (pair int (pair float_exact float_exact))) expected
        (Option.map (fun (s, va, vb) -> (s, (va, vb))) (Reduce.common sa sb)))

(* A read of a sample boxes its x, from [x], and its value, from
   [Metric.value]: 4 words. Smoothing boxes the x of the step it closes for
   the grid's [dot], and a readout reads a sample twice. A build that inlines
   across modules reads the value unboxed; the dev profile, which compiles
   with [-opaque], does not, so the bound is the dev profile's. *)
let words_per_sample = 8.

(* A reduction is made at every resize and extended at every poll, and a
   readout is read at every poll, from series of millions of samples: each
   costs its dot columns and a few words a sample, never per-sample records.
   The words a sample are the difference between two lengths, so that the
   columns and the series' own cost cancel. *)
let words_a_sample =
  test "a reduction and a readout allocate a few words a sample" (fun () ->
      let n = 4000 in
      let case len =
        let s = series_of (List.init len (fun i -> (i, Float.of_int (i mod 7) -. 3.))) in
        let hi = Float.of_int (2 * n) in
        let dot x = min 99 (Float.to_int (x /. hi *. 100.)) in
        let grid = { Reduce.lo = 0.; hi; dots = 100; dot } in
        let x i = Float.of_int (Metric.step s i) in
        (s, grid, x)
      in
      let words f =
        f ();
        let before = Gc.minor_words () in
        f ();
        Gc.minor_words () -. before
      in
      let per what f =
        let a = words (f (case n)) and b = words (f (case (2 * n))) in
        at_most ~msg:what (Windtrap.float 1e-9) ~than:words_per_sample
          ((b -. a) /. Float.of_int n)
      in
      per "words a sample of a smoothed reduction" (fun (s, grid, x) () ->
          ignore (Sys.opaque_identity (Reduce.make ~tau:100. grid ~x s)));
      per "words a sample of a raw reduction" (fun (s, grid, x) () ->
          ignore (Sys.opaque_identity (Reduce.make grid ~x s)));
      per "words a sample of a smoothed readout" (fun (s, _, x) () ->
          ignore (Sys.opaque_identity (Reduce.at ~tau:100. ~x s 1e9))))

let extend =
  prop "extending a reduction equals reducing the longer series"
    Gen.(
      with_pp
        (fun ppf (c, cut) -> Format.fprintf ppf "@[%a@ cut at %d@]" pp_case c cut)
        (let* samples, g = gen_case ~value:gen_big_value () in
         let+ cut = int_range 0 (List.length samples) in
         ((samples, g), cut)))
    (fun ((samples, g), cut) ->
      let store = Lazy.force laws in
      let first = List.filteri (fun k _ -> k < cut) samples in
      let rest = List.filteri (fun k _ -> k >= cut) samples in
      let id, log = Fixture.write_log store (List.map Fixture.sample first) in
      let h = History.read ~series:[ "m" ] (Option.get (Store.run store id)) in
      Out_channel.with_open_gen [ Open_append; Open_binary ] 0o644 log
        (fun oc ->
          List.iter (fun r -> output_string oc (Fixture.sample r ^ "\n")) rest);
      let h' = History.follow h in
      let s0 = History.series h "m" and s1 = History.series h' "m" in
      let whole = reduce g s1 in
      let grown = Reduce.extend (reduce g s0) ~x:(x_of g s1) s1 in
      equal int (Reduce.length whole) (Reduce.length grown);
      for k = 0 to g.dots - 1 do
        let msg = Printf.sprintf "column %d" k in
        List.iter
          (fun (smooth, log) ->
            equal ~msg float_exact
              (Reduce.mean whole ~smooth ~log k)
              (Reduce.mean grown ~smooth ~log k))
          [ (false, false); (false, true); (true, false); (true, true) ];
        equal ~msg
          (option (pair float_exact float_exact))
          (Reduce.extent whole k) (Reduce.extent grown k);
        equal ~msg string
          (event_name (Reduce.event whole k))
          (event_name (Reduce.event grown k))
      done;
      let side = option (pair float_exact float_exact) in
      List.iter
        (fun smooth ->
          let wl, wr = Reduce.edges whole ~smooth
          and gl, gr = Reduce.edges grown ~smooth in
          equal side wl gl;
          equal side wr gr)
        [ false; true ])

(* Laws of the charts, through the drawing the screen calls *)

let theme depth =
  Munin_tui.Theme.make { scheme = `Dark; depth; background = None }

let truecolor = theme `Truecolor

let fg (s : Matrix.Ansi.Style.t) = Option.get s.fg

let chart ?(smooth = false) ?(log = false) ?(guides = `Both) x =
  {
    Plot.x;
    log;
    smooth;
    guides;
    now = None;
    cursor = None;
    extents = false;
    fill = false;
    lines = [];
  }

(* [drawn th c ~width ~height runs] is the grid [c] draws at [width] by
   [height] cells, each of [runs] a role, a colour and a series reduced on
   [c]'s dot columns at [reduced], the drawing's own size by default, each
   sample at its step. *)
let drawn ?reduced th (c : Plot.t) ~width ~height runs =
  let rw, rh = Option.value reduced ~default:(width, height) in
  let lines =
    match Plot.dots th c ~width:rw ~height:rh with
    | None -> []
    | Some (dots, dot) ->
        let lo, hi = c.x in
        let tau = if c.smooth then Some (Plot.tau c.x) else None in
        List.map
          (fun (role, colour, s) ->
            let x i = Float.of_int (Metric.step s i) in
            let reduction = Reduce.make ?tau { lo; hi; dots; dot } ~x s in
            { Plot.role; colour; reductions = [ reduction ] })
          runs
  in
  let g = Matrix.Grid.create ~width ~height () in
  Plot.draw th { c with lines } g;
  g

let braille_dots =
  [ (0x01, 0, 0); (0x02, 0, 1); (0x04, 0, 2); (0x40, 0, 3);
    (0x08, 1, 0); (0x10, 1, 1); (0x20, 1, 2); (0x80, 1, 3) ]

(* [ink ?colour g] is the braille dots of [g]'s cells, of those in [colour]
   when given, each as its column and row of dots in [g]. *)
let ink ?colour g =
  let acc = ref [] in
  for y = 0 to Matrix.Grid.height g - 1 do
    for x = 0 to Matrix.Grid.width g - 1 do
      let i = Matrix.Grid.idx g ~x ~y in
      let cp = Matrix.Grid.get_codepoint g i in
      let mine =
        match colour with
        | None -> true
        | Some c -> Matrix.Ansi.Color.equal (Matrix.Grid.get_fg g i) c
      in
      if cp >= 0x2800 && cp <= 0x28FF && mine then
        List.iter
          (fun (bit, dx, dy) ->
            if (cp - 0x2800) land bit <> 0 then
              acc := ((2 * x) + dx, (4 * y) + dy) :: !acc)
          braille_dots
    done
  done;
  !acc

(* [braille g i] is the dots of [g]'s cell [i] as their bits, [0] for a
   cell of another glyph. *)
let braille g i =
  let cp = Matrix.Grid.get_codepoint g i in
  if cp >= 0x2800 && cp <= 0x28FF then cp - 0x2800 else 0

let colour = Windtrap.Testable.make ~pp:Matrix.Ansi.Color.pp ~equal:Matrix.Ansi.Color.equal

(* [rows ink] is, for each dot column inked, its least and greatest row,
   by column. *)
let rows ink =
  let t = Hashtbl.create 64 in
  List.iter
    (fun (c, r) ->
      match Hashtbl.find_opt t c with
      | Some (a, b) -> Hashtbl.replace t c (min a r, max b r)
      | None -> Hashtbl.replace t c (r, r))
    ink;
  List.sort compare (Hashtbl.fold (fun c ab acc -> (c, ab) :: acc) t [])

(* [cells g p] is the cells of [g] whose text satisfies [p], as columns. *)
let cells g p =
  let acc = ref [] in
  for y = 0 to Matrix.Grid.height g - 1 do
    for x = 0 to Matrix.Grid.width g - 1 do
      if p (Matrix.Grid.get_text g (Matrix.Grid.idx g ~x ~y)) then acc := x :: !acc
    done
  done;
  !acc

let text g =
  String.concat "\n"
    (List.init (Matrix.Grid.height g) (fun y ->
         String.concat ""
           (List.init (Matrix.Grid.width g) (fun x ->
                Matrix.Grid.get_text g (Matrix.Grid.idx g ~x ~y)))))

let holds sub s =
  let n = String.length sub in
  let rec go i = i + n <= String.length s && (String.sub s i n = sub || go (i + 1)) in
  go 0

let gen_guides = Gen.of_list [ `Both; `Y; `None ]

let pp_guides ppf g =
  Format.pp_print_string ppf
    (match g with `Both -> "both axes" | `Y -> "the y axis" | `None -> "no axis")

(* Finite values, the extremes and -0. among them. *)
let gen_finite =
  Gen.frequency
    [
      (20, Gen.float_range (-50.) 50.);
      (1, Gen.constant Float.max_float);
      (1, Gen.constant (-.Float.max_float));
      (1, Gen.constant (-0.));
      (1, Gen.constant 1e-300);
    ]

(* A chart's size and guides, and samples at increasing steps. *)
type draw_case = {
  width : int;
  height : int;
  guides : [ `Both | `Y | `None ];
  samples : (int * float) list;
}

let pp_draw_case ppf c =
  Format.fprintf ppf "@[%dx%d, %a,@ %a@]" c.width c.height pp_guides c.guides
    pp_samples c.samples

let gen_draw_case ?(value = gen_finite) ?(min_height = 1) () =
  let open Gen in
  with_pp pp_draw_case
    (let* width = int_range 1 140 in
     let* height = int_range min_height 30 in
     let* guides = gen_guides in
     let* deltas = list ~size:(int_range 1 60) (int_range 0 400) in
     let* values = list ~size:(constant (List.length deltas)) value in
     let steps =
       List.rev
         (snd (List.fold_left (fun (s, acc) d -> (s + d, (s + d) :: acc)) (0, []) deltas))
     in
     constant { width; height; guides; samples = List.combine steps values })

let last_step samples = List.fold_left (fun m (s, _) -> max m s) 0 samples
let domain samples = (0., Float.of_int (last_step samples + 1))
let me = Munin_tui.Plot.Cursor
let mine = fg (Munin_tui.Theme.run truecolor 0)

(* A run drawn in the cursor's colour without its spikes: an overlay. A
   spike's dot takes the cursor's line's colour in a cell it shares with
   the line, so a law of the line alone reads it from this run. *)
let plain = Munin_tui.Plot.Overlay 'a'

(* The value a y label names, with its suffix. *)
let label_value s =
  let n = String.length s in
  let scaled d = Option.map (fun v -> v *. d) (float_of_string_opt (String.sub s 0 (n - 1))) in
  if n = 0 then None
  else
    match s.[n - 1] with
    | 'k' -> scaled 1e3
    | 'M' -> scaled 1e6
    | 'G' -> scaled 1e9
    | _ -> float_of_string_opt s

(* [y_labels g] is the labels left of [g]'s y axis, top to bottom. *)
let y_labels g =
  List.filter_map
    (fun y ->
      let label =
        String.trim
          (String.concat ""
             (List.init (min 6 (Matrix.Grid.width g)) (fun x ->
                  Matrix.Grid.get_text g (Matrix.Grid.idx g ~x ~y))))
      in
      if label = "" then None else Some label)
    (List.init (Matrix.Grid.height g) Fun.id)

(* Law 1 and 2: a line is drawn in the dot columns its samples were reduced
   to, joined from its first sample's to its last's, whatever its values. A
   flat series at a value no y label gives exactly has a plot without y
   labels, which must lie where the plot with them does. *)
(* The cases the law has to draw, each often enough that every run reaches
   it: any samples; a plot one or two rows high with both axes, where the
   tick search mostly finds no exact y label, of a flat series at a value no
   tick names; and samples at one step, which fall in one dot column. *)
let gen_shape =
  Gen.(
    with_pp
      (fun ppf s ->
        match s with
        | `Any -> Format.pp_print_string ppf "any"
        | `Low h -> Format.fprintf ppf "flat, %d rows" h
        | `One_step -> Format.pp_print_string ppf "one step")
      (one_of [ constant `Any; map (fun h -> `Low h) (int_range 1 2); constant `One_step ]))

let binned =
  prop "a line covers the dot columns from its first sample's to its last's"
    (Gen.pair (gen_draw_case ()) gen_shape)
    (fun (c, shape) ->
      let c =
        match shape with
        | `Any -> c
        | `Low height ->
            {
              c with
              guides = `Both;
              height;
              samples = List.map (fun (s, _) -> (s, 12.3456789)) c.samples;
            }
        | `One_step ->
            let s = fst (List.hd c.samples) in
            { c with samples = List.map (fun (_, v) -> (s, v)) c.samples }
      in
      let x = domain c.samples in
      let ch = chart ~guides:c.guides x in
      let s = series_of c.samples in
      let g = drawn truecolor ch ~width:c.width ~height:c.height [ (me, 0, s) ] in
      let expected =
        match Plot.dots truecolor ch ~width:c.width ~height:c.height with
        | None -> []
        | Some (_, dot) ->
            let first = dot (Float.of_int (fst (List.hd c.samples)))
            and last = dot (Float.of_int (last_step c.samples)) in
            List.init (last - first + 1) (fun i -> first + i)
      in
      cover "one dot column" (List.length expected = 1);
      cover "several dot columns" (List.length expected > 1);
      cover "both axes, no y label" (c.guides = `Both && y_labels g = []);
      let shown = List.map fst (rows (ink ~colour:mine g)) in
      (* A plot's left edge is the axis's: shift the ink to the plot. *)
      let shift = match (shown, expected) with a :: _, b :: _ -> a - b | _ -> 0 in
      equal (list int) expected (List.map (fun k -> k - shift) shown))

(* Law 2: a column's value is its samples' mean, and the y scale keeps
   order, so a rising series draws a line that never falls. *)
let rising =
  prop "a series that rises draws a line that never falls"
    (Gen.pair (gen_draw_case ~value:(Gen.float_range 1e-3 1e3) ()) Gen.bool)
    (fun (c, log) ->
      let values = List.sort Float.compare (List.map snd c.samples) in
      let samples = List.combine (List.map fst c.samples) values in
      let ch = chart ~log ~guides:c.guides (domain samples) in
      let g =
        drawn truecolor ch ~width:c.width ~height:c.height
          [ (plain, 0, series_of samples) ]
      in
      let rec check = function
        | (k, (top, bottom)) :: ((k', (top', bottom')) :: _ as rest) ->
            let msg = Printf.sprintf "dot columns %d and %d" k k' in
            at_most ~msg int ~than:top top';
            at_most ~msg int ~than:bottom bottom';
            check rest
        | _ -> ()
      in
      check (rows (ink ~colour:mine g)))

(* Law 3: a lone spike is drawn in its column, faint, smoothed or not. The
   spike follows a quarter of the samples. Under smoothing the line has
   settled on them and takes at most 1 - exp (-1 / tau) of it, a seventh at
   the least tau, 200 steps over 30; without it the line is the column's
   mean, which takes at most a quarter of it, since every dot column holds
   four samples or more. A plot of six rows then holds it more than three
   dot rows from the line. *)
let spike =
  let open Gen in
  let gen =
    let* smooth = bool in
    let* width = int_range 30 140 in
    let* n = if smooth then int_range 200 400 else int_range (8 * width) ((8 * width) + 200) in
    let* base = float_range (-10.) 10. in
    let* jump = float_range 10. 1000. in
    let* up = bool in
    let* at = int_range (n / 4) (n - 1) in
    let* height = int_range 6 30 in
    let+ guides = gen_guides in
    (smooth, n, base, (if up then jump else -.jump), at, width, height, guides)
  in
  prop "a spike is a faint dot in its column, smoothed or not"
    (with_pp
       (fun ppf (smooth, n, base, jump, at, w, h, guides) ->
         Format.fprintf ppf "%s%d samples of %g, %+g at %d, %dx%d, %a"
           (if smooth then "smoothed, " else "")
           n base jump at w h pp_guides guides)
       gen)
    (fun (smooth, n, base, jump, at, width, height, guides) ->
      cover "smoothed" smooth;
      cover "not smoothed" (not smooth);
      let samples = List.init n (fun i -> (i, if i = at then base +. jump else base)) in
      let ch = chart ~smooth ~guides (domain samples) in
      let g = drawn truecolor ch ~width ~height [ (me, 0, series_of samples) ] in
      let faint = fg (Munin_tui.Theme.faint_run truecolor 0) in
      match Plot.dots truecolor ch ~width ~height with
      | None -> fail "no plot"
      | Some (_, dot) ->
          let line = rows (ink ~colour:mine g) in
          let shift = match line with (k, _) :: _ -> k - dot 0. | [] -> 0 in
          let column = dot (Float.of_int at) + shift in
          Windtrap.not_equal ~msg:"the faint dots in the spike's column"
            (Windtrap.list Windtrap.int) []
            (List.filter_map
               (fun (k, r) -> if k = column then Some r else None)
               (ink ~colour:faint g)))

(* Law 6: from the first sample that is not finite, the plot is hatched and
   labelled, and the line stops. *)
let non_finite =
  let open Gen in
  let gen =
    let* c = gen_draw_case ~value:(float_range (-5.) 5.) ~min_height:4 () in
    let* width = int_range 30 140 in
    let* guides = of_list [ `Both; `Y ] in
    let* after = int_range 1 20 in
    let+ e = of_list [ Float.nan; Float.infinity; Float.neg_infinity ] in
    (* The tail spans the steps before it again, so that the hatch is
       wide enough to show its pattern beside its label. *)
    let last = last_step c.samples in
    let gap = max 1 ((last + 1) / after) in
    let tail = List.init after (fun i -> (last + 1 + (i * gap), e)) in
    let tail = tail @ [ ((2 * last) + 2 + after, e) ] in
    { c with width; guides; samples = c.samples @ tail }
  in
  prop "from the first sample that is not finite, a labelled hatch and no line"
    (with_pp pp_draw_case gen)
    (fun c ->
      let ch = chart ~guides:c.guides (domain c.samples) in
      let g = drawn truecolor ch ~width:c.width ~height:c.height [ (me, 0, series_of c.samples) ] in
      let hatch = cells g (( = ) "╱") in
      let line = List.map (fun (k, _) -> k / 2) (ink ~colour:mine g) in
      if hatch = [] then fail "no hatch";
      let shown = text g in
      let e = snd (List.nth c.samples (List.length c.samples - 1)) in
      equal ~msg:"the labels shown" Windtrap.(list string)
        [ "× " ^ Munin_tui.Numbers.non_finite e ]
        (List.filter (fun s -> holds s shown) [ "× NaN"; "× inf"; "× -inf" ]);
      at_most ~msg:"the line's last cell, the hatch's first" Windtrap.int
        ~than:(List.fold_left min max_int hatch)
        (List.fold_left max 0 line))

(* Law 7: the cursor's run is drawn last, its spikes included: a cell it
   shares with another run's line shows the dots of both, in the cursor's
   run's colour. The other run logs the cursor's values in reverse, so the
   two share a y domain and each draws alone as it draws with the other. *)
let cursor_wins =
  prop "the cursor's run colours a cell it shares"
    (Gen.with_pp
       (fun ppf (c, other) ->
         Format.fprintf ppf "%a, %s" pp_draw_case c
           (match other with `Context -> "a grey run" | `Overlay -> "an overlay"))
       (Gen.pair (gen_draw_case ()) (Gen.of_list [ `Context; `Overlay ])))
    (fun (c, other) ->
      let steps, values = List.split c.samples in
      let s = series_of c.samples
      and r = series_of (List.combine steps (List.rev values)) in
      let role = match other with `Context -> Munin_tui.Plot.Context | `Overlay -> Overlay 'a' in
      let ch = chart ~guides:c.guides (domain c.samples) in
      let draw runs = drawn truecolor ch ~width:c.width ~height:c.height runs in
      let g = draw [ (role, 1, r); (me, 0, s) ]
      and theirs = draw [ (role, 1, r) ]
      and ours = draw [ (me, 0, s) ] in
      let spike = fg (Munin_tui.Theme.faint_run truecolor 0) in
      let shared = ref 0 and spiked = ref 0 in
      for y = 0 to c.height - 1 do
        for x = 0 to c.width - 1 do
          let i = Matrix.Grid.idx g ~x ~y in
          let a = braille theirs i and b = braille ours i in
          if a lor b <> 0 then begin
            if a <> 0 && b <> 0 then begin
              incr shared;
              if Matrix.Ansi.Color.equal (Matrix.Grid.get_fg ours i) spike then incr spiked
            end;
            let expected = Matrix.Grid.get_fg (if b <> 0 then ours else theirs) i in
            equal ~msg:(Printf.sprintf "cell %d,%d: dots, colour" x y) (pair int colour)
              (a lor b, expected)
              (braille g i, Matrix.Grid.get_fg g i)
          end
        done
      done;
      cover "a cell both runs ink" (!shared > 0);
      cover ~share:0.05 "a spike in a cell the other run inks" (!spiked > 0);
      cover "a grey run" (other = `Context);
      cover "an overlay" (other = `Overlay))

(* Law 8: without colour, each overlay ends in its letter. *)
let letters =
  prop "without colour an overlay ends in its letter, with colour it does not"
    (gen_draw_case ~min_height:3 ())
    (fun c ->
      let s = series_of c.samples in
      let ch = chart ~guides:c.guides (domain c.samples) in
      let runs = [ (Munin_tui.Plot.Overlay 'd', 1, s); (me, 0, s) ] in
      let width = c.width + 10 in
      let letter th = cells (drawn th ch ~width ~height:c.height runs) (( = ) "d") in
      equal ~msg:"in colour" (list int) [] (letter truecolor);
      (* The line's last cell, and the cell right of it within the plot,
         whose place in the grid the line's first dot column gives. *)
      let mono = theme `Mono in
      let line = rows (ink (drawn mono ch ~width ~height:c.height runs)) in
      cover "a line" (line <> []);
      match (line, Plot.dots mono ch ~width ~height:c.height) with
      | (first, _) :: _, Some (dots, dot) ->
          let offset = first - dot (Float.of_int (fst (List.hd c.samples))) in
          let last = fst (List.nth line (List.length line - 1)) / 2 in
          equal ~msg:"without colour" (list int)
            [ min (last + 1) ((offset + dots - 1) / 2) ]
            (letter mono)
      | _ -> not_equal ~msg:"without colour" (list int) [] (letter mono))

(* Rule 1: a y label is a number, never one cut short. A near-constant
   series about a round value, with a step far below the labels' digits,
   is labelled with values that read exactly in six cells, or not at all. *)
let exact_labels =
  let open Gen in
  let gen =
    let* k = int_range 1 999 in
    let* e = int_range (-6) 9 in
    let* negative = bool in
    let* rel = float_range 1e-9 1e-2 in
    let* n = int_range 2 200 in
    let* width = int_range 20 140 in
    let* height = int_range 2 30 in
    let+ log = bool in
    let base = Float.of_int k *. (10. ** Float.of_int e) in
    let base = if negative && not log then -.base else base in
    (base, Float.abs base *. rel /. Float.of_int n, n, width, height, log)
  in
  prop "a y label is an exact number in six cells"
    (with_pp
       (fun ppf (base, step, n, w, h, log) ->
         Format.fprintf ppf "%d samples from %g by %g, %dx%d%s" n base step w h
           (if log then ", log" else ""))
       gen)
    (fun (base, step, n, width, height, log) ->
      let samples = List.init n (fun i -> (i, base +. (Float.of_int i *. step))) in
      let ch = chart ~log ~guides:`Y (domain samples) in
      let g = drawn truecolor ch ~width ~height [ (me, 0, series_of samples) ] in
      let labels = y_labels g in
      cover "labelled" (labels <> []);
      let values =
        List.map
          (fun l ->
            match label_value l with
            | Some v -> v
            | None -> fail (Printf.sprintf "the label %S is not a number" l))
          labels
      in
      let lo = List.fold_left (fun m (_, v) -> Float.min m v) Float.infinity samples
      and hi = List.fold_left (fun m (_, v) -> Float.max m v) Float.neg_infinity samples in
      let below, above =
        if log then (lo /. 10., hi *. 1.25)
        else
          let margin = Float.max ((hi -. lo) *. 0.05) (Float.abs lo *. 1e-12) in
          (lo -. margin, hi +. margin)
      in
      List.iter
        (fun v ->
          if v < below || v > above then
            fail (Printf.sprintf "the label %g is outside [%g, %g]" v below above))
        values;
      equal ~msg:"the labels, top to bottom" Windtrap.(list float_exact)
        (List.sort_uniq (fun a b -> Float.compare b a) values)
        values)

(* Law 1's x domain of a chart: from 0 to the greatest end of the runs
   drawn that have samples of the chart's metric. A run that may still log
   ends at its latest step rounded up to 1, 2 or 5 times a power of ten,
   capped at its [steps]; any other at its largest step. These runs have no
   writer a reader can probe: one is writing when a sample arrived, silent
   when none did, or ended. *)
type life = Writing | Ended | Silent

let life_name = function Writing -> "writing" | Ended -> "ended" | Silent -> "silent"

(* A run that logged samples at step 0 and [last], of the metric [m] when
   [has], else of another. *)
type law_run = { life : life; steps : int option; last : int; has : bool }

let law_run ?steps ?(has = true) life last = { life; steps; last; has }

let pp_law_run ppf r =
  Format.fprintf ppf "%s %d of %s%s" (life_name r.life) r.last
    (match r.steps with Some n -> string_of_int n | None -> "-")
    (if r.has then "" else " without m")

let facts r =
  let store = Lazy.force laws in
  let key = if r.has then "m" else "n" in
  let samples =
    List.map
      (fun st -> Fixture.values st [ (key, 1.) ])
      (List.sort_uniq compare [ 0; r.last ])
  in
  let ended =
    match r.life with Ended -> [ Fixture.finished ] | Writing | Silent -> []
  in
  let declare = [ Fixture.declare "m"; Fixture.declare "n" ] in
  let id, _ = Fixture.write_log store ?steps:r.steps ~declare (samples @ ended) in
  let run = Option.get (Store.run store id) in
  (* An arrival at the poll's time: the run reads [writing]. *)
  let watch =
    match r.life with
    | Writing -> Some (Watch.observe (Watch.start ~samples:0) ~now:0. ~samples:2)
    | Ended | Silent -> None
  in
  Munin_tui.Facts.of_run ~now:0. ~history:(Some (History.read run)) ~hint:None ~watch
    ~pauses:[] run

let x_domain =
  cases
    ~id:(fun (name, _, _) -> name)
    "the x domain ends at a run's last step, a growing run's rounded up"
    [
      ("ended short of its steps", [ law_run Ended ~steps:3000 2812 ], 2812.);
      ("ended without steps", [ law_run Ended 2812 ], 2812.);
      ("ended past its steps", [ law_run Ended ~steps:100 150 ], 150.);
      ("ended at step 0", [ law_run Ended 0 ], 1.);
      ("silent short of its steps", [ law_run Silent ~steps:3000 80 ], 80.);
      ("writing short of its steps", [ law_run Writing ~steps:3000 80 ], 100.);
      ("writing, capped at its steps", [ law_run Writing ~steps:3000 2900 ], 3000.);
      ("writing at its steps", [ law_run Writing ~steps:100 100 ], 100.);
      ("writing past its steps", [ law_run Writing ~steps:100 150 ], 200.);
      ("writing, a power of ten", [ law_run Writing 10_000 ], 10_000.);
      ("writing, just past one", [ law_run Writing 10_001 ], 20_000.);
      ("writing, to a five", [ law_run Writing 2_001 ], 5_000.);
      ("writing, to a two", [ law_run Writing 1_500 ], 2_000.);
      ("writing, one step", [ law_run Writing 1 ], 1.);
      ("writing at step 0", [ law_run Writing ~steps:3000 0 ], 1.);
      ("the greatest of two", [ law_run Writing ~steps:300 10; law_run Ended 400 ], 400.);
      ("an overlay without the metric", [ law_run Ended 10; law_run Ended ~has:false 150 ], 10.);
      ("no run with the metric", [ law_run Ended ~has:false 150 ], 1.);
    ]
    (fun (_, runs, expected) ->
      equal (pair float_exact float_exact) (0., expected) (Plot.domain ~key:"m" (List.map facts runs)))

(* Zoom's step cursor: its rule shows on every row of the plot that no
   dots take, the raw extent's included, and its step label on the x axis is a chip of the step padded by a blank
   each side, which the tick labels keep a blank cell from. The cursor stays
   left of the axis's end, where [step] is the chip's neighbour. *)
let zoom_cursor =
  (* Many noisy samples a dot column, so that the raw extent is tall. *)
  let gen =
    let open Gen in
    let* n = int_range 300 800 in
    let* values = list ~size:(constant n) (float_range (-50.) 50.) in
    let* width = int_range 60 100 in
    let* height = int_range 6 30 in
    let* at = float_range 0. 0.8 in
    let samples = List.mapi (fun i v -> (i, v)) values in
    constant ({ width; height; guides = `Both; samples }, at)
  in
  let gen =
    Gen.with_pp
      (fun ppf (c, at) -> Format.fprintf ppf "%dx%d, %d samples, cursor at %g" c.width c.height
         (List.length c.samples) at)
      gen
  in
  prop "zoom's cursor rule shows where no dots are, its label clear of the ticks" gen
    (fun (c, at) ->
      let lo, hi = domain c.samples in
      let v = Float.floor (lo +. (at *. (hi -. lo))) in
      let ch = { (chart (lo, hi)) with cursor = Some v; extents = true } in
      let g = drawn truecolor ch ~width:c.width ~height:c.height [ (me, 0, series_of c.samples) ] in
      let at x y = Matrix.Grid.idx g ~x ~y in
      let rule = fg (Munin_tui.Theme.fg truecolor Mid) in
      let axis_row =
        match
          List.find_opt
            (fun y -> List.exists (fun x -> Matrix.Grid.get_text g (at x y) = "└") (List.init c.width Fun.id))
            (List.init c.height Fun.id)
        with
        | Some y -> y
        | None -> fail "no x axis"
      in
      let ruled x y =
        Matrix.Grid.get_text g (at x y) = "│" && Matrix.Ansi.Color.equal (Matrix.Grid.get_fg g (at x y)) rule
      in
      (* The rule's column, from the chart drawn without a line. *)
      let alone = drawn truecolor ch ~width:c.width ~height:c.height [] in
      let columns =
        List.filter
          (fun x ->
            List.exists
              (fun y ->
                let i = Matrix.Grid.idx alone ~x ~y in
                Matrix.Grid.get_text alone i = "│"
                && Matrix.Ansi.Color.equal (Matrix.Grid.get_fg alone i) rule)
              (List.init axis_row Fun.id))
          (List.init c.width Fun.id)
      in
      let x = match columns with [ x ] -> x | _ -> failf "the rule's columns: %d" (List.length columns) in
      let bare = drawn truecolor { ch with cursor = None } ~width:c.width ~height:c.height
          [ (me, 0, series_of c.samples) ] in
      let band = fg (Munin_tui.Theme.faint_run truecolor 0) in
      cover "the raw extent on the rule's column"
        (List.exists
           (fun y -> Matrix.Ansi.Color.equal (Matrix.Grid.get_fg bare (Matrix.Grid.idx bare ~x ~y)) band)
           (List.init axis_row Fun.id));
      List.iter
        (fun y ->
          if not (ruled x y || braille g (at x y) <> 0) then
            failf "row %d of the rule's column shows %S" y (Matrix.Grid.get_text g (at x y)))
        (List.init axis_row Fun.id);
      (* The chip: the cells of the label row on the cursor's background. *)
      let row = axis_row + 1 in
      let chip = Option.get (Munin_tui.Theme.bg truecolor Raise) in
      let on x = Matrix.Ansi.Color.equal (Matrix.Grid.get_bg g (at x row)) chip in
      let xs = List.filter on (List.init c.width Fun.id) in
      let first = List.fold_left min c.width xs and last = List.fold_left max (-1) xs in
      if xs = [] then fail "no step label";
      let text a b = String.concat "" (List.init (b - a + 1) (fun x -> Matrix.Grid.get_text g (at (a + x) row))) in
      equal ~msg:"the chip" string (" " ^ Munin_tui.Numbers.count (Float.to_int v) ^ " ") (text first last);
      let blank x = x < 0 || x >= c.width || String.trim (Matrix.Grid.get_text g (at x row)) = "" in
      equal ~msg:"the cells beside the chip are blank" (list bool) [ true; true ]
        [ blank (first - 1); blank (last + 1) ])

(* [x_end r] is the end law 1 gives one run. *)
let x_end r =
  let l = Float.of_int r.last in
  match (r.life, r.steps) with
  | (Ended | Silent), _ -> Float.max 1. l
  | Writing, Some n when r.last <= n -> Float.min (Float.of_int n) (Munin_tui.Facts.bound l)
  | Writing, _ -> Munin_tui.Facts.bound l

(* A run's steps and last step: short of them, at them, or past them. *)
let gen_law_run =
  let open Gen in
  let* life = of_list [ Writing; Ended; Silent ] in
  let* has = frequency [ (3, constant true); (1, constant false) ] in
  let* steps = option (int_range 1 5000) in
  let+ last =
    match steps with
    | None -> int_range 0 5000
    | Some n -> frequency [ (3, int_range 0 n); (1, constant n); (1, int_range n (2 * n)) ]
  in
  { life; steps; last; has }

let gen_law_runs =
  Gen.(
    with_pp
      (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_law_run)
      (list ~size:(int_range 1 3) gen_law_run))

let x_domain_law =
  prop "the x domain is the furthest end of the runs with the metric"
    ~examples:
      [
        [ law_run Ended ~has:false 150 ];
        [ law_run Writing 2500 ];
        [ law_run Writing ~steps:3000 2900 ];
      ]
    gen_law_runs (fun runs ->
      let short r = match r.steps with Some n -> r.last < n | None -> false in
      let past r = match r.steps with Some n -> r.last > n | None -> false in
      let rounds_past r =
        match r.steps with
        | Some n -> r.last <= n && Munin_tui.Facts.bound (Float.of_int r.last) > Float.of_int n
        | None -> false
      in
      let some p = List.exists p runs in
      cover "a writing run short of its steps" (some (fun r -> r.life = Writing && short r));
      cover "a writing run's rounding passes its steps" (some (fun r -> r.life = Writing && rounds_past r));
      cover "a writing run without steps" (some (fun r -> r.life = Writing && r.steps = None));
      cover "an ended run short of its steps" (some (fun r -> r.life = Ended && short r));
      cover "a run past its steps" (some past);
      cover "a silent run" (some (fun r -> r.life = Silent));
      cover "a run without the metric beside one with it"
        (some (fun r -> r.has) && some (fun r -> not r.has));
      cover "no run with the metric" (not (some (fun r -> r.has)));
      let ends = List.map x_end (List.filter (fun r -> r.has) runs) in
      let expected = match ends with [] -> 1. | _ -> List.fold_left Float.max 0. ends in
      equal (pair float_exact float_exact) (0., expected) (Plot.domain ~key:"m" (List.map facts runs)))

(* Zoom's window: [n] halvings of the domain [(0, hi)], the part that holds
   the cursor, or the nearer end for a cursor outside. It lies inside the
   domain, so zooming on an ended run's last step shows its last samples. *)
let zoom_window =
  let gen =
    Gen.(
      let* hi =
        frequency [ (3, map Float.of_int (int_range 1 100_000)); (1, float_range 1e-3 1e9) ]
      in
      let* narrow = int_range 0 30 in
      let+ at =
        frequency
          [
            (5, float_range 0. hi);
            (3, constant hi);
            (1, constant 0.);
            (1, float_range (-.hi) 0.);
            (1, float_range hi (2. *. hi));
          ]
      in
      (hi, narrow, at))
    |> Gen.with_pp (fun ppf (hi, narrow, at) ->
           Format.fprintf ppf "(0, %h), %d halvings, at %h" hi narrow at)
  in
  prop "zoom's window lies in the domain and holds the cursor" gen (fun (hi, narrow, at) ->
      cover "the cursor at the domain's end" (at = hi);
      cover "the cursor outside the domain" (at < 0. || at > hi);
      cover "narrowed more than once" (narrow > 1);
      let lo', hi' = Plot.window (0., hi) ~at ~narrow in
      at_least ~msg:"its start" float_exact ~than:0. lo';
      at_most ~msg:"its end" float_exact ~than:hi hi';
      let w = Float.ldexp hi (-narrow) in
      equal ~msg:"its width" (float_rel ~rel:1e-9 ~abs:(1e-12 *. hi)) w (hi' -. lo');
      let at = Float.min hi (Float.max 0. at) in
      at_least ~msg:"the cursor, from its start" float_exact ~than:lo' at;
      at_most ~msg:"the cursor, to its end" float_exact ~than:hi' at)

(* Law 9: a live run's latest step has its [now] rule wherever it is in the
   x domain, which is closed: a run whose latest step is the domain's end,
   as one past 1,000 steps without [steps] reaches at step 1,000, has it. *)
let now_rule =
  cases ~id:(fun v -> Printf.sprintf "now at %g" v) "the now rule marks the latest step"
    [ 0.; 500.; 1000. ]
    (fun v ->
      let samples = List.init 11 (fun i -> (i * 100, Float.of_int i)) in
      let ch = { (chart (0., 1000.)) with now = Some v } in
      let width = 60 and height = 10 in
      let g = drawn truecolor ch ~width ~height [ (me, 0, series_of samples) ] in
      let dot =
        match Plot.dots truecolor ch ~width ~height with
        | Some (_, dot) -> dot
        | None -> fail "no plot"
      in
      (* The line starts at step 0: its first dot column is [dot 0.] in the
         plot, which places the plot in the grid. *)
      let shift =
        match rows (ink ~colour:mine g) with
        | (k, _) :: _ -> k - dot 0.
        | [] -> fail "no line"
      in
      equal ~msg:"the rule's cells" (list int)
        [ (dot v + shift) / 2 ]
        (List.sort_uniq compare (cells g (( = ) "┊")));
      let rec count t i =
        if i + 3 > String.length t then 0
        else (if String.sub t i 3 = "now" then 1 else 0) + count t (i + 1)
      in
      equal ~msg:"the now labels" int 1 (count (text g) 0))

(* A rule is context and dots are data: on the [now] rule's or zoom's
   cursor rule's column, a cell that dots take, a line's, a spike's or the
   raw extent's, shows them as it shows without the rule, and a cell no
   dots take shows the rule. With two runs at zoom, the cursor's raw extent
   shares the rule's column with the other run's line, and the cursor's run,
   drawn last, colours the cells they share. *)
let lines_over_rules =
  let gen =
    let open Gen in
    let* role =
      one_of
        (List.map constant
           [ `Context; `Overlay; `Cursor; `Band_and `Context; `Band_and `Overlay ])
    in
    let band = match role with `Band_and _ -> true | _ -> false in
    (* Many samples a dot column and a gentle slope make a tall extent. *)
    let* n = if band then int_range 400 1000 else int_range 20 400 in
    let* slope = if band then float_range (-0.01) 0.01 else float_range (-5.) 5. in
    let* noise = list ~size:(constant n) (float_range (-1.) 1.) in
    let* width = if band then int_range 30 60 else int_range 30 120 in
    let* height = int_range 6 24 in
    let* at = float_range 0.05 0.95 in
    let* zoom = if band then constant true else bool in
    let+ off = float_range (-8.) 8. in
    let samples = List.mapi (fun i e -> (i, (slope *. Float.of_int i) +. (10. *. e))) noise in
    (samples, width, height, at, zoom, role, off)
  in
  let name = function
    | `Context -> "grey run"
    | `Overlay -> "overlay"
    | `Cursor -> "cursor's run"
    | `Band_and `Context -> "cursor's run and a grey run"
    | `Band_and `Overlay -> "cursor's run and an overlay"
  in
  let pp ppf (s, w, h, at, zoom, role, off) =
    Format.fprintf ppf "%dx%d, %d samples, %s rule at %g, %s, the other %+g" w h (List.length s)
      (if zoom then "cursor" else "now") at (name role) off
  in
  prop "dots win the cells they share with a rule" (Gen.with_pp pp gen)
    (fun (samples, width, height, at, zoom, role, off) ->
      let lo, hi = domain samples in
      let v = Float.floor (lo +. (at *. (hi -. lo))) in
      let base = { (chart (lo, hi)) with extents = zoom } in
      let ch = if zoom then { base with cursor = Some v } else { base with now = Some v } in
      let other r =
        match r with
        | `Context -> (Munin_tui.Plot.Context, 0, fg (Munin_tui.Theme.faint truecolor Dim))
        | `Overlay -> (Munin_tui.Plot.Overlay 'a', 1, fg (Munin_tui.Theme.run truecolor 1))
      in
      let shifted = List.map (fun (i, y) -> (i, y +. off)) samples in
      let runs, theirs =
        match role with
        | (`Context | `Overlay) as r ->
            let role, colour, _ = other r in
            ([ (role, colour, series_of samples) ], None)
        | `Cursor -> ([ (me, 0, series_of samples) ], None)
        | `Band_and r ->
            let role, colour, line = other r in
            ([ (role, colour, series_of shifted); (me, 0, series_of samples) ], Some line)
      in
      let g = drawn truecolor ch ~width ~height runs
      and bare = drawn truecolor base ~width ~height runs
      and alone = drawn truecolor ch ~width ~height []
      and no_band = drawn truecolor { base with extents = false } ~width ~height runs in
      let band = fg (Munin_tui.Theme.faint_run truecolor 0) in
      let rule_glyph s = s = "│" || s = "┊" in
      let cell = pair string colour in
      let shared = ref 0 and ruled = ref 0 and taken = ref 0 in
      for y = 0 to height - 1 do
        for x = 0 to width - 1 do
          let i = Matrix.Grid.idx g ~x ~y in
          let shown g = (Matrix.Grid.get_text g i, Matrix.Grid.get_fg g i) in
          let msg = Printf.sprintf "cell %d,%d" x y in
          if rule_glyph (Matrix.Grid.get_text alone i) then
            if braille bare i <> 0 then begin
              incr shared;
              (* The band takes a cell the other run's line crosses. *)
              (match theirs with
               | Some line
                 when Matrix.Ansi.Color.equal (Matrix.Grid.get_fg no_band i) line
                      && braille no_band i <> 0
                      && Matrix.Ansi.Color.equal (Matrix.Grid.get_fg bare i) band ->
                   incr taken
               | _ -> ());
              equal ~msg cell (shown bare) (shown g)
            end
            else if String.trim (Matrix.Grid.get_text bare i) = "" then begin
              incr ruled;
              equal ~msg cell (shown alone) (shown g)
            end
        done
      done;
      cover "dots on the rule" (!shared > 0);
      cover "the rule where no dots are" (!ruled > 0);
      cover "a grey run's line" (role = `Context);
      cover "an overlay's line" (role = `Overlay);
      cover "zoom's cursor rule" zoom;
      (* About a quarter of the cases: a demand of a share would need 400
         cases to decide it; presence is reached on every run. *)
      cover "the band takes a cell another run's line crosses, on the rule"
        (!taken > 0))

(* The x axis's program labels, the [step] title and zoom's cursor, keep
   clear of each other: the row reads [step] whole or not at all, wherever
   the cursor is, and whole while the cursor is far from it. *)
let x_labels_clear =
  let gen =
    let open Gen in
    let* last = int_range 5 2000 in
    let* width = int_range 30 140 in
    let* height = int_range 6 20 in
    let+ at =
      frequency
        [
          (1, float_range 0. 1.);
          (1, float_range 0. 0.5);
          (2, float_range 0.85 1.);
          (1, float_range 0.95 1.);
        ]
    in
    (last, width, height, at)
  in
  let pp ppf (last, w, h, at) = Format.fprintf ppf "%d steps, %dx%d, cursor at %g" last w h at in
  prop "the step title and the cursor's chip keep clear of each other" (Gen.with_pp pp gen)
    (fun (last, width, height, at) ->
      let samples = List.init (last + 1) (fun i -> (i, Float.of_int (i mod 7))) in
      let lo, hi = (0., Float.of_int last) in
      let v = Float.round (at *. hi) in
      let ch = { (chart (lo, hi)) with cursor = Some v; extents = true } in
      let g = drawn truecolor ch ~width ~height [ (me, 0, series_of samples) ] in
      let row y = String.concat "" (List.init width (fun x -> Matrix.Grid.get_text g (Matrix.Grid.idx g ~x ~y))) in
      let axis_row =
        match List.find_opt (fun y -> holds "└" (row y)) (List.init height Fun.id) with
        | Some y -> y
        | None -> fail "no x axis"
      in
      let words = List.filter (( <> ) "") (String.split_on_char ' ' (row (axis_row + 1))) in
      let lettered w = String.exists (fun c -> String.contains "step" c) w in
      let far = at <= 0.5 && width >= 60 in
      cover "the cursor near the title" (at >= 0.95);
      cover "the cursor far from the title" far;
      match List.filter lettered words with
      | [ "step" ] -> ()
      | [] when not far -> ()
      | ws -> failf "the label row %S reads %s" (row (axis_row + 1)) (String.concat ", " ws))

(* Round numbers are the floats nearest their decimal values, on every
   system: law 1's end is the least of 1, 2 or 5 times a power of ten at or
   above the largest step, tried at powers of ten and the floats beside
   them, up to 1e22, the last power of ten a float holds exactly. *)
let bounds =
  cases
    ~id:(fun (x, _) -> Printf.sprintf "%.17g" x)
    "a step rounds up to 1, 2 or 5 times a power of ten"
    [
      (0., 1.);
      (1., 1.);
      (Float.pred 1., 1.);
      (Float.succ 1., 2.);
      (Float.pred 10., 10.);
      (10., 10.);
      (Float.succ 10., 20.);
      (999.99999999999989, 1000.);
      (1000., 1000.);
      (1000.0000000000001, 2000.);
      (1500., 2000.);
      (2001., 5000.);
      (Float.pred 1e15, 1e15);
      (1e15, 1e15);
      (Float.succ 1e15, 2e15);
      (Float.pred 1e22, 1e22);
      (1e22, 1e22);
      (Float.succ 1e22, 2e22);
      (3e22, 5e22);
      (Float.max_float, Float.max_float);
      (Float.infinity, Float.max_float);
    ]
    (fun (x, expected) -> equal float_exact expected (Munin_tui.Facts.bound x))

(* A y axis's ticks are round numbers: the powers of ten of a log domain,
   the multiples of a 1-2-5 step of a linear one. *)
let smoothables =
  cases ~id:fst "a key is smoothed by default when it starts with train/"
    [ ("train/loss", true); ("val/loss", false); ("train", false);
      ("val/train/loss", false) ]
    (fun (key, expected) -> equal bool expected (Plot.smoothable key))

let taus =
  cases
    ~id:(fun ((lo, hi), _) -> Printf.sprintf "(%g, %g)" lo hi)
    "smoothing's time constant is a thirtieth of the domain"
    [ ((0., 300.), 10.); ((100., 400.), 10.); ((0., 1.), 1. /. 30.) ]
    (fun (domain, expected) ->
      equal (float_rel ~rel:1e-15 ~abs:0.) expected (Plot.tau domain))

(* A linear axis's ticks: in the domain, ascending, the multiples of one
   step of 1, 2 or 5 times a power of ten, at most [most], each labelled. *)
let tick_steps =
  let gen =
    Gen.(
      let* lo = float_range (-1e4) 1e4 in
      let* e = int_range (-8) 8 in
      let* width = float_range 1. 10. in
      let+ most = int_range 2 12 in
      (lo, lo +. (width *. (10. ** Float.of_int e)), most))
  in
  prop "a linear axis's ticks are the domain's multiples of a round step"
    (Gen.with_pp
       (fun ppf (lo, hi, most) -> Format.fprintf ppf "(%h, %h), %d" lo hi most)
       gen)
    (fun (lo, hi, most) ->
      let ticks = Plot.y_ticks ~log:false ~most (lo, hi) in
      cover "ticks" (List.length ticks >= 2);
      at_most ~msg:"their count" int ~than:most (List.length ticks);
      List.iter
        (fun t -> if t < lo || t > hi then failf "the tick %h is outside" t)
        ticks;
      equal ~msg:"ascending" (list float_exact)
        (List.sort_uniq Float.compare ticks) ticks;
      if ticks <> [] then begin
        not_equal ~msg:"labelled" (option unit) None
          (Option.map ignore (Numbers.tick ticks));
        match ticks with
        | a :: b :: _ ->
            (* Some step [m] × 10^k gives each tick as the float nearest its
               multiple n × m × 10^k, the [n]s consecutive. *)
            let k0 = Numbers.decade (b -. a) in
            let multiples (m, k) =
              let unit = decimal_text m k in
              let ns = List.map (fun t -> Float.to_int (Float.round (t /. unit))) ticks in
              List.for_all2 (fun t n -> decimal_text (n * m) k = t) ticks ns
              && List.for_all2 (fun n n' -> n' = n + 1)
                   (List.filteri (fun i _ -> i < List.length ns - 1) ns)
                   (List.tl ns)
            in
            let steps =
              List.concat_map
                (fun k -> List.map (fun m -> (m, k)) [ 1; 2; 5 ])
                [ k0 - 1; k0; k0 + 1 ]
            in
            satisfies ~claim:"the multiples of 1, 2 or 5 times a power of ten"
              (list float_exact)
              (fun _ -> List.exists multiples steps)
              ticks
        | _ -> ()
      end)

let ticks_of =
  cases
    ~id:(fun (log, most, (lo, hi), _) ->
      Printf.sprintf "%s %d over [%.17g, %.17g]" (if log then "log" else "linear") most lo hi)
    "the ticks are the domain's round numbers"
    [
      (true, 10, (1e-3, 1e5), [ 1e-3; 1e-2; 0.1; 1.; 10.; 100.; 1e3; 1e4; 1e5 ]);
      (true, 10, (Float.succ 1e-3, Float.pred 1e5), [ 1e-2; 0.1; 1.; 10.; 100.; 1e3; 1e4 ]);
      (true, 3, (1e-3, 1e5), [ 1e-3; 1.; 1e3 ]);
      (false, 4, (0., 0.3), [ 0.; 0.1; 0.2; 0.3 ]);
      (false, 5, (0., 1.), [ 0.; 0.5; 1. ]);
      (false, 5, (0., 2.1e6), [ 0.; 5e5; 1e6; 1.5e6; 2e6 ]);
      (false, 5, (-0.003, 0.003), [ -0.002; 0.; 0.002 ]);
      (* The float below 1 and 1, where a step's quotients pass 2^52 and
         several of its multiples round to 1. *)
      (false, 3, (Float.pred 1., 1.), [ 1. ]);
      (* Both quotients by the step 1e-113 round outward, while its
         multiples in the domain are three. *)
      (false, 3, (Float.succ 5e-113, Float.pred 9e-113), [ 6e-113; 7e-113; 8e-113 ]);
    ]
    (fun (log, most, domain, expected) ->
      equal (list float_exact) expected (Plot.y_ticks ~log ~most domain))

(* Law 5's choice: y is on a log scale when every finite sample drawn is
   above 0 and the greatest is a hundred times the least or more. [logged
   values] is what a poll knows of a run that logged [values] under [m]. *)
let logged values =
  let store = Lazy.force laws in
  let id, _ = Fixture.write_log store (List.mapi (fun st v -> Fixture.sample (st, v)) values) in
  let r = Option.get (Store.run store id) in
  Munin_tui.Facts.of_run ~now:0. ~history:(Some (History.read r)) ~hint:None ~watch:None
    ~pauses:[] r

let log_choice =
  cases
    ~id:(fun (name, _, _) -> name)
    "y is on a log scale when every value is above 0 over two decades"
    [
      ("two decades", [ [ 1.; 100. ] ], true);
      ("short of two", [ [ 1.; 99. ] ], false);
      ("below one", [ [ 0.01; 1. ] ], true);
      ("a zero", [ [ 0.; 100. ] ], false);
      ("a negative", [ [ -1.; 100. ] ], false);
      ("a NaN aside", [ [ 1.; Float.nan; 1000. ] ], true);
      ("over two runs", [ [ 1.; 10. ]; [ 50.; 200. ] ], true);
      ("one run of zeros", [ [ 1.; 1000. ]; [ 0. ] ], false);
      ("no sample", [ [] ], false);
    ]
    (fun (_, runs, expected) ->
      equal bool expected (Plot.log (List.map logged runs) "m"))

(* A value prints in four significant digits, counted after rounding: a
   value that rounds up to the next power of ten keeps four, and its notation
   is that of the rounded value. *)
(* [cut w s] is [s] when it fits, else a prefix of [s] and […] in at most
   [w] cells, and [""] for [w <= 0]: over wide glyphs, combining marks and
   tabs. *)
let cut_law =
  let gen =
    let open Gen in
    let piece = one_of (List.map constant [ "a"; "b"; "漢"; "字"; "é"; "\t"; "é"; " " ]) in
    let* pieces = list ~size:(int_range 0 12) piece in
    let s = String.concat "" pieces in
    let+ w = int_range (-1) (Munin_tui.Numbers.width s + 1) in
    (w, s)
  in
  let gen = Gen.with_pp (fun ppf (w, s) -> Format.fprintf ppf "%d, %S" w s) gen in
  prop "a cut text fits, and ends in … when it is cut" gen (fun (w, s) ->
      let module N = Munin_tui.Numbers in
      let r = N.cut w s in
      let ell = "…" in
      at_most ~msg:"its width" int ~than:(max 0 w) (N.width r);
      cover "whole" (N.width s <= w);
      if N.width s <= w then equal ~msg:"whole" string s r
      else if w <= 0 then equal ~msg:"no room" string "" r
      else begin
        if not (String.ends_with ~suffix:ell r) then failf "%S does not end in …" r;
        let p = String.sub r 0 (String.length r - String.length ell) in
        if not (String.starts_with ~prefix:p s) then failf "%S is not a prefix of %S" p s;
        cover "cut" true;
        cover "a wide glyph left out at the edge" (N.width p < w - 1)
      end)

let values_printed =
  cases ~id:fst "a value prints in four significant digits"
    [
      ("0.254", (0.254, "0.2540"));
      ("2150", (2150., "2150"));
      ("10", (10., "10"));
      ("10.5", (10.5, "10.50"));
      ("-1.5", (-1.5, "-1.500"));
      ("5.116e-05", (5.116e-5, "5.116e-05"));
      ("9.9996", (9.9996, "10.00"));
      ("0.99996", (0.99996, "1.000"));
      ("999.96", (999.96, "1.000k"));
      ("999999.6", (999999.6, "1.000e+06"));
      ("9.99996e-05", (9.99996e-5, "0.0001000"));
      ("9.998e-05", (9.998e-5, "9.998e-05"));
      ("NaN", (Float.nan, "NaN"));
      ("inf", (Float.infinity, "inf"));
      ("-inf", (Float.neg_infinity, "-inf"));
      ("-0.", (-0., "0"));
      ("348513", (348513., "348513"));
      ("-999999", (-999999., "-999999"));
      ("348500", (348500., "348500"));
      ("1234.56", (1234.56, "1.235k"));
      ("348513.5", (348513.5, "348.5k"));
      ("1234567", (1234567., "1.235e+06"));
      ("0.0015", (0.0015, "0.001500"));
    ]
    (fun (_, (x, expected)) -> equal string expected (Munin_tui.Numbers.value x))

(* A printed value reads back to the value, or shows that it is rounded: by a
   decimal point, a suffix or an exponent. A whole value of six digits or
   fewer reads back, NaN and the infinities as their words, and [-0.] as
   [0]. *)
let rounding_shown =
  let whole n = Gen.map Float.of_int (Gen.int_range (-n) n) in
  let gen =
    Gen.frequency
      [
        (4, whole 999_999);
        (4, whole 1_000_000_000);
        (4, Gen.float_range (-1e6) 1e6);
        (2, Gen.any_float);
        (2, Gen.one_of (List.map Gen.constant [ Float.nan; Float.infinity; Float.neg_infinity ]));
        (2, Gen.one_of (List.map Gen.constant [ 0.; -0. ]));
      ]
  in
  let read = function
    | "NaN" -> Some Float.nan
    | "inf" -> Some Float.infinity
    | "-inf" -> Some Float.neg_infinity
    | s -> Munin_float_text.Float_read.of_string s
  in
  prop "a printed value reads back or shows it is rounded" gen (fun x ->
      let s = Munin_tui.Numbers.value x in
      let exact =
        match read s with
        | Some y -> Float.equal y x
        | None -> false
      in
      let marked = String.exists (fun c -> String.contains ".kMGe" c) s in
      let six = Float.is_integer x && Float.abs x < 1e6 in
      cover "whole, six digits or fewer" six;
      cover "whole, past six digits" (Float.is_integer x && not six);
      cover "rounded, from a thousand" ((not exact) && Float.abs x >= 1000. && Float.abs x < 1e6);
      cover ~share:0.05 "not finite" (not (Float.is_finite x));
      cover ~share:0.1 "negative" (x < 0.);
      cover ~share:0.03 "-0." (Float.sign_bit x && x = 0.);
      if (six || not (Float.is_finite x)) && not exact then
        failf "%h prints %s, which does not read back" x s;
      if not (exact || marked) then failf "%h prints %s, rounded without a mark" x s)

(* A difference fills a strip's column of twelve cells: its widest forms are
   a signed value, as [-4.941e-324] with its sign, and [+100.00 pt]. *)
let difference_cells = 11

let difference_fits =
  let gen =
    Gen.(
      let value =
        frequency
          [
            (4, float_range (-1e3) 1e3);
            (2, float_range (-1e-6) 1e-6);
            (2, float_range 0. 1.);
            (2, any_float);
            (1, constant 0.);
            (1, constant Float.max_float);
            (1, constant (-.Float.max_float));
            (1, constant 5e-324);
          ]
      in
      triple (of_list [ "loss"; "val/acc" ]) value value)
  in
  prop "a difference takes at most eleven cells"
    ~examples:
      [
        ("loss", 1., 1e-300);
        ("loss", Float.max_float, -.Float.max_float);
        ("loss", 1., 5e-324);
      ]
    (Gen.with_pp
       (fun ppf (k, v, o) -> Format.fprintf ppf "%s %h against %h" k v o)
       gen)
    (fun (key, value, other) ->
      match Munin_tui.Numbers.difference key ~value ~other with
      | None -> ()
      | Some s ->
          cover "relative" (String.ends_with ~suffix:"%" s);
          cover "absolute"
            (not
               (String.ends_with ~suffix:"%" s
               || String.ends_with ~suffix:"pt" s));
          at_most ~msg:s int ~than:difference_cells (Munin_tui.Numbers.width s))

(* The sentences of [Numbers.difference] and their examples. *)
let differences =
  cases
    ~id:(fun (name, _, _, _, _) -> name)
    "a difference is relative, in points, or the difference itself"
    [
      ("+60%", "loss", 1.6, 1., Some "+60%");
      ("-6.0%", "loss", 0.94, 1., Some "-6.0%");
      ("+0.14 pt", "val/acc", 0.9414, 0.94, Some "+0.14 pt");
      ("same in the digits shown", "loss", 1.0001, 1., None);
      ("not finite", "loss", Float.nan, 1., None);
      ("against zero", "loss", 0.9, 0., Some "+0.9000");
      ("four digits before the point", "loss", 100.99, 1., Some "+9999%");
      ("past four digits", "loss", 1., 1e-300, Some "+1");
      ( "past the largest float",
        "loss",
        Float.max_float,
        -.Float.max_float,
        Some "+inf" );
    ]
    (fun (_, key, value, other, expected) ->
      equal (option string) expected
        (Munin_tui.Numbers.difference key ~value ~other))

let decimals =
  prop "decimal m k is the float nearest m × 10^k"
    ~examples:[ (1, 22); (1, 23); (1, -324); (2, 308); (1, 309); (5, -325) ]
    Gen.(
      with_pp
        (fun ppf (m, k) -> Format.fprintf ppf "%de%d" m k)
        (pair (int_range (-1_000_000) 1_000_000) (int_range (-340) 330)))
    (fun (m, k) -> equal float_exact (decimal_text m k) (Numbers.decimal m k))

let decades =
  prop "decade x is the k with 10^k <= |x| < 10^(k+1)"
    ~examples:[ 1.; 10.; 1e22; Float.pred 1e22; 5e-324; Float.max_float; 0.1 ]
    Gen.float
    (fun x ->
      assume (x <> 0.);
      let k = Numbers.decade x in
      at_most ~msg:"10^k" float_exact ~than:(Float.abs x) (Numbers.decimal 1 k);
      less ~msg:"10^(k+1)" float_exact ~than:(Numbers.decimal 1 (k + 1))
        (Float.abs x))

(* [bound]'s candidates are 1, 2 and 5 times the powers of ten: the least at
   or above [x] is among those of its decade and the next. *)
let rounds =
  prop "bound x is the least 1, 2 or 5 times a power of ten at or above x"
    Gen.(frequency [ (3, float_range 0. 1e6); (1, float) ])
    (fun x ->
      let b = Munin_tui.Facts.bound x in
      if not (x > 0.) then equal ~msg:"at or below 0" float_exact 1. b
      else
        let k = Numbers.decade x in
        let candidates =
          List.concat_map
            (fun k -> List.map (fun m -> decimal_text m k) [ 1; 2; 5 ])
            [ k; k + 1 ]
        in
        let above = List.filter (fun c -> c >= x) candidates in
        let expected =
          match List.filter Float.is_finite above with
          | [] -> Float.max_float
          | l -> List.fold_left Float.min Float.infinity l
        in
        equal float_exact expected b)

let counts =
  prop "a count prints with a separator every three digits"
    ~examples:[ 0; 999; 1000; -1000; max_int; min_int ]
    Gen.int
    (fun n ->
      let s = Numbers.count n in
      let digits = String.concat "" (String.split_on_char ',' s) in
      equal ~msg:"without separators" string (string_of_int n) digits;
      let groups =
        String.split_on_char ','
          (if n < 0 then String.sub s 1 (String.length s - 1) else s)
      in
      equal ~msg:"the groups after the first" (list int)
        (List.map (fun _ -> 3) (List.tl groups))
        (List.map String.length (List.tl groups)))

let elisions =
  let gen =
    Gen.(
      let segment = string_of ~size:(int_range 1 8) (of_list [ 'a'; 'b'; '_' ]) in
      let* key = map (String.concat "/") (list ~size:(int_range 1 4) segment) in
      let+ w = int_range 0 (String.length key + 2) in
      (w, key))
  in
  prop "an elided key fits, and keeps its last segment when it can"
    (Gen.with_pp (fun ppf (w, k) -> Format.fprintf ppf "%d, %S" w k) gen)
    (fun (w, key) ->
      let e = Numbers.elide w key in
      at_most ~msg:e int ~than:w (Numbers.width e);
      if String.length key <= w then equal ~msg:"whole" string key e
      else
        let last =
          match String.rindex_opt key '/' with
          | Some i -> String.sub key i (String.length key - i)
          | None -> key
        in
        cover "the last segment kept" (String.ends_with ~suffix:last e);
        cover "cut" (w > 0 && String.ends_with ~suffix:"…" e))

(* Ticks at the multiples of a 1, 2 or 5 step: those [tick] labels read
   back exactly in six cells, in one notation. *)
let tick_law =
  let gen =
    Gen.(
      let* m = of_list [ 1; 2; 5 ] in
      let* k = int_range (-12) 12 in
      let* first = int_range (-20) 20 in
      let+ n = int_range 1 6 in
      List.init n (fun i -> decimal_text ((first + i) * m) k))
  in
  prop "a tick label gives its tick exactly in six cells"
    (Gen.with_pp
       (fun ppf xs ->
         Format.pp_print_list ~pp_sep:Format.pp_print_space
           (fun ppf x -> Format.fprintf ppf "%g" x)
           ppf xs)
       gen)
    (fun xs ->
      match Numbers.tick xs with
      | None -> ()
      | Some label ->
          cover "labelled" true;
          List.iter
            (fun x ->
              let l = label x in
              at_most ~msg:l int ~than:6 (Numbers.width l);
              equal ~msg:l (option float_exact) (Some x) (label_value l))
            xs)

(* The rest of the module's sentences, by their examples. *)
let metrics =
  cases ~id:(fun (k, x, _) -> Printf.sprintf "%s %g" k x)
    "an accuracy in [0;1] prints as a percentage, any other metric as a value"
    [ ("val/acc", 0.954, "95.40%"); ("loss", 0.5, "0.5000"); ("acc", 1.5, "1.500") ]
    (fun (key, x, expected) -> equal string expected (Numbers.metric key x))

let givens =
  cases ~id:Fun.id "a given value prints in the fewest digits that read back"
    [ "0.1"; "0.0004213752718364"; "5e-324"; "NaN"; "-inf" ]
    (fun text ->
      equal string text (Numbers.given (float_of_string text)))

let compacts =
  cases ~id:snd "an axis prints a step compactly"
    [ (500., "500"); (1500., "1.5k"); (12500., "12.5k"); (3e6, "3M") ]
    (fun (x, expected) -> equal string expected (Numbers.compact x))

let percents_of =
  cases
    ~id:(fun (s, t, _) -> Printf.sprintf "%d of %d" s t)
    "progress is rounded down, 100% once the steps are reached"
    [ (19950, 20000, "99%"); (20000, 20000, "100%"); (20001, 20000, "100%");
      (5, 0, "") ]
    (fun (s, t, expected) -> equal string expected (Numbers.percent_of s t))

let percents =
  cases ~id:snd "a percentage has one decimal below 10"
    [ (7.5, "7.5%"); (71., "71%") ]
    (fun (x, expected) -> equal string expected (Numbers.percent x))

let gibs =
  test "bytes print in GiB with one decimal" (fun () ->
      equal string "1.5" (Numbers.gib (1.5 *. 1073741824.)))

(* 51,720 s is 14:22 on 1970-01-01, in UTC. *)
let clocks =
  cases ~id:(fun (name, _, _) -> name)
    "a time of day has its date unless it is today's"
    [ ("today", 51720., "14:22"); ("another day", 200000., "Jan 1 14:22") ]
    (fun (_, now, expected) ->
      setenv "TZ" (Some "UTC");
      equal string expected (Numbers.clock ~now 51720.))

let dates =
  test "a date gives the time to the second" (fun () ->
      setenv "TZ" (Some "UTC");
      equal string "Jan 1 14:22:05" (Numbers.date 51725.))

let widths =
  cases ~id:fst "a width is the cells a text takes"
    [
      ("a", ("a", 1));
      ("a wide glyph", ("漢", 2));
      ("a combining mark", ("é", 1));
      ("empty", ("", 0));
    ]
    (fun (_, (text, cells)) -> equal int cells (Numbers.width text))

let spans =
  cases ~id:fst "a span prints in two units, or seconds below a minute"
    [
      ("4m 10s", (250., "4m 10s"));
      ("45s", (45., "45s"));
      ("a minute", (60., "1m 00s"));
      ("negative", (-5., "0s"));
    ]
    (fun (_, (x, expected)) ->
      equal string expected (Munin_tui.Numbers.span x))

(* An axis's labels, in one notation: plain while every tick fits six
   cells, else the suffix, else the exponent. *)
let tick_labels =
  let labels xs = Option.map (fun l -> List.map l xs) (Munin_tui.Numbers.tick xs) in
  cases
    ~id:(fun (xs, _) -> String.concat " " (List.map (Printf.sprintf "%g") xs))
    "an axis's labels are exact, in one notation"
    [
      ([ 12500. ], Some [ "12500" ]);
      ([ 1e6 ], Some [ "1M" ]);
      ([ 1.5e6 ], Some [ "1.5M" ]);
      ([ 2e6 ], Some [ "2M" ]);
      ([ 0.; 500000.; 1e6; 1.5e6; 2e6 ], Some [ "0"; "500k"; "1M"; "1.5M"; "2M" ]);
      ([ 0.; 0.0125; 0.025 ], Some [ "0"; "0.0125"; "0.025" ]);
      ([ -250.; 0.; 250. ], Some [ "-250"; "0"; "250" ]);
      ([ 0.; 1e-5; 2e-5 ], Some [ "0"; "1e-05"; "2e-05" ]);
      ([ 1e-5; 1e-4; 1e-3 ], Some [ "1e-05"; "1e-04"; "1e-03" ]);
      ([ 0.51234 ], None);
      ([ Float.nan ], None);
    ]
    (fun (xs, expected) -> equal (option (list string)) expected (labels xs))

(* A filled chart, as a machine's strip, inks each of its line's dot
   columns from the line down to the plot's floor, without a gap. *)
let filled =
  prop "a filled line inks each of its columns down to the floor"
    (gen_draw_case ~value:(Gen.float_range (-50.) 50.) ())
    (fun c ->
      let ch = { (chart ~guides:`None (domain c.samples)) with fill = true } in
      let g = drawn truecolor ch ~width:c.width ~height:c.height [ (plain, 0, series_of c.samples) ] in
      let ink = ink ~colour:mine g in
      let floor = (4 * c.height) - 1 in
      cover "several columns" (List.length (rows ink) > 1);
      List.iter
        (fun (k, (top, bottom)) ->
          equal ~msg:(Printf.sprintf "dot column %d's lowest row" k) int floor bottom;
          equal ~msg:(Printf.sprintf "dot column %d's dots" k) int (bottom - top + 1)
            (List.length (List.filter (fun (k', _) -> k' = k) ink)))
        (rows ink))

(* A reduction made for a plot of other dot columns draws no line. *)
let matching =
  prop "a reduction made for another plot's dot columns draws no line"
    (Gen.pair (gen_draw_case ()) (Gen.int_range 1 20))
    (fun (c, more) ->
      let ch = chart ~guides:c.guides (domain c.samples) in
      let runs = [ (me, 0, series_of c.samples) ] in
      let width = c.width + 2 and height = c.height + 2 in
      let own = drawn truecolor ch ~width ~height runs in
      if ink ~colour:mine own = [] then fail "no line at its own size";
      let n w = Option.map fst (Plot.dots truecolor ch ~width:w ~height) in
      (* A wider region may keep the plot's width, giving up a guide. *)
      assume (n width <> n (width + more));
      let other = drawn ~reduced:(width + more, height) truecolor ch ~width ~height runs in
      equal (list (pair int int)) [] (ink ~colour:mine other))

(* Law 4: smoothing weighs steps, not samples, so a curve logged every step
   and every 50 steps smooths to the same line. Logged every 50 steps, a
   sample stands for the 50 steps before it, which moves the line by at most
   25 steps' change: for a decay of rate at least 2,500 steps over at most
   that many steps, 1.6% of its range, under a dot row of a 12-row chart.
   With a row each for the means of different samples in a column and for
   rounding, the lines agree within 2 dot rows in every column. *)
let cadence_rows = 2

let cadence =
  let open Gen in
  let gen =
    let* n = map (fun k -> 50 * k) (int_range 100 400) in
    let* rate = float_range 2500. (Float.of_int n) in
    let* scale = float_range 0.1 100. in
    let* offset = float_range (-10.) 10. in
    let+ width = int_range 60 140 in
    (n, rate, scale, offset, width)
  in
  prop ~count:20 "a curve logged every step or every 50 smooths to the same line"
    (with_pp
       (fun ppf (n, rate, scale, offset, width) ->
         Format.fprintf ppf "%g + %g exp (-s / %g) to %d, %d columns" offset scale rate
           n width)
       gen)
    (fun (n, rate, scale, offset, width) ->
      let f s = offset +. (scale *. exp (-.Float.of_int s /. rate)) in
      let every k = List.init ((n / k) + 1) (fun i -> (i * k, f (i * k))) in
      let ch = chart ~smooth:true ~guides:`Both (0., Float.of_int n) in
      let line k = rows (ink ~colour:mine (drawn truecolor ch ~width ~height:12 [ (plain, 0, series_of (every k)) ])) in
      let a = line 1 and b = line 50 in
      equal ~msg:"dot columns" Windtrap.(list int) (List.map fst a) (List.map fst b);
      List.iter2
        (fun (k, (top, bottom)) (_, (top', bottom')) ->
          let msg = Printf.sprintf "dot column %d" k in
          at_most ~msg Windtrap.int ~than:cadence_rows (abs (top - top'));
          at_most ~msg Windtrap.int ~than:cadence_rows (abs (bottom - bottom')))
        a b)

(* Theme *)

module Theme = Munin_tui.Theme

let depths = [ `Truecolor; `Ansi256; `Ansi16; `Ansi8; `Mono; `Plain ]

let depth_name = function
  | `Truecolor -> "direct colour"
  | `Ansi256 -> "256 colours"
  | `Ansi16 -> "16 colours"
  | `Ansi8 -> "8 colours"
  | `Mono -> "no colour"
  | `Plain -> "plain"

let colourless = function `Mono | `Plain -> true | _ -> false
let reversed (st : Matrix.Ansi.Style.t) = Matrix.Ansi.Attr.mem Inverse st.attrs

let style =
  Testable.make ~pp:Matrix.Ansi.Style.pp ~equal:Matrix.Ansi.Style.equal

let token_name : Theme.token -> string = function
  | Raise -> "Raise"
  | Sel -> "Sel"
  | Line -> "Line"
  | Line2 -> "Line2"
  | Dim -> "Dim"
  | Mid -> "Mid"
  | Fg -> "Fg"
  | Hi -> "Hi"
  | Live -> "Live"
  | Fail -> "Fail"
  | Warn -> "Warn"

let tokens = Theme.[ Raise; Sel; Line; Line2; Dim; Mid; Fg; Hi; Live; Fail; Warn ]

let coloured_depths =
  cases ~id:depth_name "a look without colour keeps bold and faint alone"
    depths
    (fun depth ->
      let t = theme depth in
      equal ~msg:"coloured" bool (not (colourless depth)) (Theme.coloured t);
      equal ~msg:"the cursor's row reversed" bool (colourless depth)
        (reversed (Theme.sel t));
      if colourless depth then begin
        let coloured_tokens =
          List.filter
            (fun tok -> (Theme.fg t tok).fg <> None || Theme.bg t tok <> None)
            tokens
        in
        equal ~msg:"tokens with a colour" (list string) []
          (List.map token_name coloured_tokens)
      end)

(* With 16 or 8 colours a token is an entry of the 16 or the terminal's
   default. *)
let sixteen_entries =
  cases ~id:depth_name "with 16 or 8 colours a token is one of the 16"
    [ `Ansi16; `Ansi8 ]
    (fun depth ->
      let t = theme depth in
      let entry (c : Matrix.Ansi.Color.t option) =
        match Option.map Matrix.Ansi.Color.intent c with
        | None | Some Default -> true
        | Some (Indexed i) -> i < 16
        | Some Rgb -> false
      in
      let names p = List.map token_name (List.filter p tokens) in
      equal ~msg:"text not of the 16" (list string) []
        (names (fun tok -> not (entry (Theme.fg t tok).fg)));
      equal ~msg:"backgrounds not of the 16" (list string) []
        (names (fun tok -> not (entry (Theme.bg t tok)))))

let run_colours =
  prop "a run's colour is taken modulo the run colours"
    Gen.(
      with_pp
        (fun ppf (d, i) -> Format.fprintf ppf "%s, %d" (depth_name d) i)
        (pair (of_list depths) (int_range (-50) 50)))
    (fun (depth, i) ->
      let t = theme depth in
      equal style (Theme.run t i) (Theme.run t (i + Theme.runs)))

let schemes =
  test "a dark and a light terminal take different colours" (fun () ->
      let look scheme =
        Theme.make { scheme; depth = `Truecolor; background = None }
      in
      not_equal ~msg:"text" (option string)
        (Option.map Matrix.Ansi.Color.to_hex (Theme.fg (look `Dark) Fg).fg)
        (Option.map Matrix.Ansi.Color.to_hex (Theme.fg (look `Light) Fg).fg))

(* What a poll knows of a run of [experiment] that declared [declared],
   each with a sample at step 0, and gave [params]. *)
let known ?(experiment = "law") ?(params = []) declared =
  let store = Lazy.force laws in
  let id, _ =
    Fixture.write_log store ~experiment ~params
      ~declare:
        (List.map (fun (key, goal) -> Fixture.declare ?goal key) declared)
      [ Fixture.values 0 (List.map (fun (k, _) -> (k, 1.)) declared) ]
  in
  let r = Option.get (Store.run store id) in
  Munin_tui.Facts.of_run ~now:0. ~history:(Some (History.read r)) ~hint:None
    ~watch:None ~pauses:[] r

let key_metrics =
  cases
    ~id:(fun (name, _, _) -> name)
    "an experiment's key metric is the first loss with a goal outside train/, \
     else the first goal outside it, else the first goal, else the first"
    [
      ( "a loss outside train/",
        [ ("train/loss", Some `Minimize); ("val/acc", Some `Maximize);
          ("val/loss", Some `Minimize) ],
        Some "val/loss" );
      ( "a goal outside train/",
        [ ("train/loss", Some `Minimize); ("val/acc", Some `Maximize) ],
        Some "val/acc" );
      ( "a goal",
        [ ("train/loss", Some `Minimize); ("train/acc", Some `Maximize) ],
        Some "train/acc" );
      ("the first", [ ("b", None); ("a", None) ], Some "a");
      ("no metric", [], None);
    ]
    (fun (_, declared, expected) ->
      equal (list (option string)) [ expected ]
        (List.map
           (fun (e : Munin_tui.Facts.experiment) -> e.key)
           (Munin_tui.Facts.experiments [ known declared ])))

let varying =
  test "an experiment's varying params differ between its runs, a missing one \
        counting" (fun () ->
      let run params = known ~params [ ("m", None) ] in
      let runs =
        [
          run [ ("lr", `Float 0.1); ("batch", `Int 64); ("opt", `String "sgd") ];
          run [ ("lr", `Float 0.2); ("batch", `Int 64) ];
        ]
      in
      equal (list (list string)) [ [ "lr"; "opt" ] ]
        (List.map
           (fun (e : Munin_tui.Facts.experiment) ->
             List.sort String.compare e.varying)
           (Munin_tui.Facts.experiments runs)))

let experiment_order =
  test "experiments come in the order of their first run" (fun () ->
      let run experiment = known ~experiment [ ("m", None) ] in
      equal (list string) [ "a"; "b" ]
        (List.map
           (fun (e : Munin_tui.Facts.experiment) -> e.name)
           (Munin_tui.Facts.experiments [ run "a"; run "b"; run "a" ])))

(* A run's glyph, for each state with and without a sample that is not
   finite: the facts of a run that logged a NaN, and of one that did not,
   in each state. *)
let glyphs =
  let facts values =
    let store = Lazy.force laws in
    let id, _ =
      Fixture.write_log store (List.mapi (fun st v -> Fixture.sample (st, v)) values)
    in
    let r = Option.get (Store.run store id) in
    Munin_tui.Facts.of_run ~now:0. ~history:(Some (History.read r))
      ~hint:None ~watch:None ~pauses:[] r
  in
  let row (state, glyph) =
    let marked =
      match state with Watch.Failed | Crashed | Killed -> glyph | _ -> "×"
    in
    [
      (Watch.word state, state, false, glyph);
      (Watch.word state ^ ", NaN", state, true, marked);
    ]
  in
  cases
    ~id:(fun (name, _, _, _) -> name)
    "a run's glyph is its state's, or × for a sample that is not finite"
    (List.concat_map row
       [
         (Watch.Starting, "○");
         (Running, "○");
         (Live, "●");
         (Stalled, "◌");
         (Writing, "●");
         (Silent, "◌");
         (Crashed, "✗");
         (Finished, "✓");
         (Failed, "✗");
         (Killed, "■");
       ])
    (fun (_, state, nan, expected) ->
      let f = facts (if nan then [ 1.; Float.nan ] else [ 1.; 2. ]) in
      equal string expected (Munin_tui.Facts.glyph { f with state }))

(* The state words, against a model of the arrivals a dashboard times. *)
type arrivals = {
  now : float;
  samples : int;
  times : float list; (* newest first *)
}

let model_word (status : Run.status) a =
  let gap =
    let rec gaps = function
      | x :: (y :: _ as r) -> (x -. y) :: gaps r
      | _ -> []
    in
    match
      List.sort Float.compare (gaps (List.filteri (fun i _ -> i < 32) a.times))
    with
    | [] -> None
    | g ->
        let n = List.length g in
        Some
          (if n mod 2 = 1 then List.nth g (n / 2)
           else (List.nth g ((n / 2) - 1) +. List.nth g (n / 2)) /. 2.)
  in
  let stalled =
    match (gap, a.times) with
    | Some g, t :: _ -> a.now -. t > Float.max (3. *. g) 10.
    | _ -> false
  in
  match status with
  | `Running ->
      if a.samples = 0 then "starting"
      else if a.times = [] then "running"
      else if stalled then "stalled"
      else "live"
  | `Silent -> if a.times <> [] && not stalled then "writing" else "silent"
  | `Crashed -> "crashed"
  | `Finished -> "finished"
  | `Failed _ -> "failed"
  | `Killed -> "killed"

let watch_test =
  let w = abstract "w" in
  let status =
    Gen.of_list
      ~pp:(fun ppf s -> Format.pp_print_string ppf (Fixture.status_name s))
      (* Running thrice, since three of the five words are a running run's. *)
      [ `Running; `Running; `Running; `Silent; `Crashed; `Finished; `Failed "e"; `Killed ]
  in
  let pp_step ppf (dt, n) = Format.fprintf ppf "+%gs %+d" dt n in
  let observe =
    command "observe"
      (Gen.with_pp pp_step
         Gen.(
           pair
             (map Float.of_int
                (frequency [ (4, int_range 0 20); (1, int_range 60 300) ]))
             (int_range (-1) 3))
      @-> w ^-> makes w)
      (fun (dt, d) a ->
        let now = a.now +. dt in
        let samples = max 0 (a.samples + d) in
        let times =
          if samples > a.samples then
            List.filteri (fun i _ -> i < 32) (now :: a.times)
          else a.times
        in
        { now; samples; times })
      (* A resume that orphans samples lowers the count a poll reads. *)
      (fun (dt, d) (w, now, samples) ->
        let now = now +. dt and samples = max 0 (samples + d) in
        (Watch.observe w ~now ~samples, now, samples))
  in
  stateful ~count:300 ~steps:60 "the state words follow the arrivals"
    [
      command "start"
        (Gen.int_range 0 3 @-> makes w)
        (fun samples -> { now = 0.; samples; times = [] })
        (fun samples -> (Watch.start ~samples, 0., samples));
      (* Listed three times, so that a value's arrivals pile up. *)
      observe;
      observe;
      observe;
      command "word"
        (status @-> w ^-> returns string)
        (fun st a ->
          let w = model_word st a in
          List.iter
            (fun x -> cover x (w = x))
            [ "starting"; "running"; "live"; "stalled"; "writing"; "silent" ];
          w)
        (fun st (w, now, _) -> Watch.word (Watch.state st (Some w) ~now));
      command "last_arrival"
        (w ^-> returns (option float_exact))
        (fun a -> match a.times with t :: _ -> Some t | [] -> None)
        (fun (w, _, _) -> Watch.last_arrival w);
    ]

(* Which states a writer may still log in, and which ended badly. *)
let live_broken =
  cases ~id:Watch.word "a state is live while a writer may log, broken when it failed"
    Watch.
      [
        Starting; Running; Live; Stalled; Writing; Silent; Crashed; Finished;
        Failed; Killed;
      ]
    (fun st ->
      let live = List.mem st Watch.[ Starting; Running; Live; Writing ] in
      let broken = List.mem st Watch.[ Failed; Crashed ] in
      equal (pair bool bool) (live, broken) (Watch.live st, Watch.broken st))

(* Before a poll watched it, a run's state is its status alone. *)
let unwatched =
  cases
    ~id:(fun (s, _) -> Fixture.status_name s)
    "a run no poll watched reads its status"
    [
      (`Running, "running");
      (`Silent, "silent");
      (`Crashed, "crashed");
      (`Finished, "finished");
      (`Failed "e", "failed");
      (`Killed, "killed");
    ]
    (fun (status, word) ->
      equal string word (Watch.word (Watch.state status None ~now:0.)))

let () =
  exit
    (run "munin.tui.modules"
       [
         group "Reduce"
           [
             means;
             smoothed_means;
             geometric_means;
             unsmoothed;
             events;
             edges;
             extend;
             words_a_sample;
             common;
             at;
           ];
         group "Plot"
           [
             x_domain;
             x_domain_law;
             zoom_window;
             log_choice;
             smoothables;
             taus;
             ticks_of;
             tick_steps;
             matching;
             binned;
             rising;
             cadence;
             filled;
             exact_labels;
             spike;
             zoom_cursor;
             non_finite;
             now_rule;
             x_labels_clear;
             cursor_wins;
             lines_over_rules;
             letters;
           ];
         group "Numbers"
           [
             cut_law;
             values_printed;
             rounding_shown;
             differences;
             difference_fits;
             tick_labels;
             tick_law;
             spans;
             decimals;
             decades;
             counts;
             elisions;
             metrics;
             givens;
             compacts;
             percents_of;
             percents;
             gibs;
             clocks;
             dates;
             widths;
           ];
         group "Facts"
           [ bounds; rounds; glyphs; key_metrics; varying; experiment_order ];
         group "Theme" [ coloured_depths; sixteen_entries; run_colours; schemes ];
         group "Watch" [ watch_test; unwatched; live_broken ];
       ])
