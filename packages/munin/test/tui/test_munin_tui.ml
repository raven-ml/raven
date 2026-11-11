(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Munin
module Reduce = Munin_tui.Reduce
module Plot = Munin_tui.Plot
module Watch = Munin_tui.Watch
module T = Matrix_test

(* Stores *)

(* The stores live under a directory that the tests make the home, so that a
   screen names one as [~/store], and times print in UTC. *)
let home = Filename.temp_dir "munin_tui" ""

let () =
  Unix.putenv "HOME" home;
  Unix.putenv "TZ" "UTC"

let rec remove path =
  match (Unix.lstat path).st_kind with
  | Unix.S_DIR ->
      Array.iter (fun e -> remove (Filename.concat path e)) (Sys.readdir path);
      Unix.rmdir path
  | _ -> Sys.remove path
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> ()

(* The stores that tests share stay until the end, with the writers that hold
   their runs: those let go first, since Windows removes no file a process holds
   open. Fixture's own [at_exit] runs after this one. *)
let () =
  at_exit (fun () ->
      Fixture.release ();
      remove home)

let child = Filename.concat (Sys.getcwd ()) "../support/child.exe"

(* [with_store ?name runs f] is [f store], [store] a store of [runs] at
   [~/name], which it removes after with the writers that hold its runs. The
   root is joined with [/], which Windows also takes, so that a screen names it
   as [~/name] there too. *)
let with_store ?(name = "store") runs f =
  let root = home ^ "/" ^ name in
  let store = Fixture.make ~child root runs in
  Fun.protect
    ~finally:(fun () ->
      Fixture.release ();
      remove root)
    (fun () -> f store)

let id_of r = r.Fixture.id

(* A test whose expected screen shows a crashed run: on a system whose writers
   record no boot, such as Windows, the fixture holds no crashed run. *)
let needs_crashed () =
  if not (Fixture.boots ~child home) then
    skip ~reason:"no run reads crashed where writers record no boot identity" ()

(* Screens *)

let second = 1_000_000_000

(* The dashboard's wall clock is the fixture's time plus the session's. *)
let elapsed = ref (fun () -> 0)
let clock () = Fixture.now +. (Float.of_int (!elapsed ()) /. 1e9)

(* [feed store r ~step] appends [r]'s sample at [step], written now. *)
let feed store r ~step = Fixture.feed store r ~step ~time:(clock ())

type key = Text of string | Key of Matrix.Input.Keystroke.key

let send term s = function
  | Text t -> T.input s (T.Terminal.text term t)
  | Key k -> T.input s (T.Terminal.key term k)

(* [show ~cols ~rows store screen keys] is the screen after the dashboard
   started on [screen] and read [store] once, then [script] ran, a second
   passed, and the dashboard took [keys], each followed by a second. *)
let show ?(script = ignore) ?profile ~cols ~rows store screen keys =
  let term = T.Terminal.create ?profile ~cols ~rows () in
  (elapsed := fun () -> 0);
  let app = Munin_tui.app ~clock store screen in
  Mosaic_test.run term app @@ fun s ->
  (elapsed := fun () -> T.now s);
  T.advance s second;
  script s;
  T.advance s second;
  List.iter
    (fun k ->
      send term s k;
      T.advance s second)
    keys;
  T.check_presented s;
  T.Terminal.screen term

let golden name text =
  expect_file ~file:("packages/munin/test/tui/golden/" ^ name ^ ".txt") text

(* A terminal of 16 colours: matrix's emulator, from an environment that
   names no more. *)
let sixteen = { T.Profile.vte with env = [ ("TERM", "xterm") ] }

(* A terminal whose user asked for no colour. *)
let no_colour = { T.Profile.vte with env = ("NO_COLOR", "1") :: T.Profile.vte.env }

(* The main store's screens, after tidy-otter logged its step 12,400 while the
   dashboard watched, so that an arrival makes it live. The goldens hold the
   styles, so that a golden sees colour, bold, faint and reverse video. *)
let screen_test name ~cols ~rows ?(start = Munin_tui.Runs) ?(crashed = false)
    ?profile keys =
  test name (fun () ->
      if crashed then needs_crashed ();
      with_store Fixture.runs @@ fun store ->
      let script _ = feed store Fixture.tidy_otter ~step:12400 in
      golden name
        (T.Screen.styled (show ~script ?profile ~cols ~rows store start keys)))

let char c = Text (String.make 1 c)
let times n k = List.init n (fun _ -> k)

(* The dashboard opens with its cursor on tidy-otter, the newest run, and its
   chart on train/loss. amber-fox is the second run and dusty-mole the fifth:
   overlaying both draws a run whose losses turn NaN against the cursor's. *)
let overlay_two =
  [ char 'j'; Text " "; char 'j'; char 'j'; char 'j'; Text " "; char 'g' ]

let screens =
  group "screens"
    [
      screen_test "dashboard-80x24" ~cols:80 ~rows:24 ~crashed:true [];
      screen_test "dashboard-120x40" ~cols:120 ~rows:40 ~crashed:true [];
      screen_test "dashboard-160x48" ~cols:160 ~rows:48 ~crashed:true [];
      screen_test "overlays-120x40" ~cols:120 ~rows:40 ~crashed:true
        overlay_two;
      screen_test "overlays-120x40-no-colour" ~cols:120 ~rows:40 ~crashed:true
        ~profile:no_colour overlay_two;
      screen_test "filter-120x40" ~cols:120 ~rows:40 ~crashed:true
        [ char '/'; Text "cifar" ];
      (* A kept filter that keeps no run: the pane holds it, and the centre
         says why it is empty. *)
      screen_test "filter-none-120x40" ~cols:120 ~rows:40 ~crashed:true
        [ char '/'; char 'q'; Key Enter ];
      screen_test "dashboard-120x40-16-colours" ~cols:120 ~rows:40 ~crashed:true
        ~profile:sixteen [];
      screen_test "strips-120x40" ~cols:120 ~rows:40
        (overlay_two @ [ Key Enter ]);
      (* Without colour the cursor's strip is reverse video, its plot too. *)
      screen_test "strips-120x40-no-colour" ~cols:120 ~rows:40 ~profile:no_colour
        (overlay_two @ [ Key Enter ]);
      (* The last strips: a live run's machine, each filled to its floor. *)
      screen_test "strips-machine-120x40" ~cols:120 ~rows:40
        (overlay_two @ [ Key Enter ] @ times 20 (Key Down));
      (* Zoom's step cursor starts on the latest val/loss sample, and each
         move left goes back one sample. *)
      screen_test "zoom-120x40" ~cols:120 ~rows:40
        (overlay_two @ [ Key Enter; Key Enter ] @ times 8 (Key Left));
      screen_test "info-120x40" ~cols:120 ~rows:40 [ char 'i' ];
      screen_test "help-120x40" ~cols:120 ~rows:40 ~crashed:true [ char '?' ];
      screen_test "help-80x24" ~cols:80 ~rows:24 ~crashed:true [ char '?' ];
      screen_test "dashboard-100x30" ~cols:100 ~rows:30 ~crashed:true [];
      screen_test "small-50x14" ~cols:50 ~rows:14
        ~start:(Munin_tui.Run (Some (id_of Fixture.tidy_otter)))
        [];
      test "empty-120x40" (fun () ->
          with_store ~name:"empty" [] @@ fun store ->
          golden "empty-120x40"
            (T.Screen.styled (show ~cols:120 ~rows:40 store Munin_tui.Runs [])));
    ]

(* States *)

(* The run pane's first lines, for a run in each state: its glyph, name and
   state word, and what it says of its time. tidy-otter reads running until a
   sample arrives, live once one has, and stalled once its usual gap of 23 s,
   from two arrivals, has passed three times. far-node and lost-node have no
   writer, so their lock is not probed. *)
(* [chars s] is the UTF-8 characters of [s], one per cell for the text the
   screens print. *)
let chars s =
  let n = String.length s in
  let rec go i acc =
    if i >= n then List.rev acc
    else
      let c = Char.code s.[i] in
      let k =
        if c < 0x80 then 1 else if c < 0xE0 then 2 else if c < 0xF0 then 3 else 4
      in
      let k = min k (n - i) in
      go (i + k) (String.sub s i k :: acc)
  in
  go 0 []

(* [from col row] is [row] from its cell [col]. *)
let from col row =
  String.concat "" (List.filteri (fun i _ -> i >= col) (chars row))

(* The run pane's heading, [³ run], starts its column, and its first two
   lines follow a blank one. *)
let run_pane screen =
  let row, col =
    match T.Screen.find screen "³" with
    | (row, col) :: _ -> (row, col)
    | [] -> fail "no run pane"
  in
  String.trim (from col (T.Screen.row screen (row + 2)))
  ^ " / "
  ^ String.trim (from col (T.Screen.row screen (row + 3)))

let states_test =
  test "states" (fun () ->
      needs_crashed ();
      let first ?(script = fun _ _ -> ()) runs r =
        with_store runs @@ fun store ->
        let screen =
          show ~script:(script store) ~cols:160 ~rows:48 store
            (Munin_tui.Run (Some (id_of r)))
            []
        in
        run_pane screen
      in
      let tidy = Fixture.tidy_otter in
      let arrives store _ = feed store tidy ~step:12400 in
      let stalls store s =
        feed store tidy ~step:12400;
        T.advance s (23 * second);
        feed store tidy ~step:12450;
        T.advance s (250 * second)
      in
      let lines =
        [
          first Fixture.states Fixture.warm_start;
          first Fixture.runs tidy;
          first Fixture.runs tidy ~script:arrives;
          first Fixture.runs tidy ~script:stalls;
          first Fixture.states Fixture.far_node ~script:(fun store _ ->
              feed store Fixture.far_node ~step:3200);
          first Fixture.states Fixture.lost_node;
          first Fixture.runs Fixture.mlp_base;
          first Fixture.runs Fixture.amber_fox;
          first Fixture.runs Fixture.eager_hawk;
          first Fixture.states Fixture.quiet_heron;
        ]
      in
      golden "states" (String.concat "\n" lines ^ "\n"))

(* Logs written from the format's text, for the tests that need a run of their
   own: a start, the metric "m", then [records]. *)
let law_runs = ref 0

let metric_record ?(goal = "") ?(step_metric = "") key =
  Printf.sprintf {|{"type":"metric","time":1.0,"key":"%s"%s%s}|} key
    (if goal = "" then "" else Printf.sprintf {|,"goal":"%s"|} goal)
    (if step_metric = "" then ""
     else Printf.sprintf {|,"step_metric":"%s"|} step_metric)

let write_log store ?goal ?steps ?declare ?(started = 1) ?(experiment = "law") records =
  let declare = Option.value declare ~default:[ metric_record ?goal "m" ] in
  incr law_runs;
  let id = Printf.sprintf "20261101T000000Z-%08x" !law_runs in
  let dir = Filename.concat (Store.root store) "runs" in
  List.iter
    (fun d -> if not (Sys.file_exists d) then Sys.mkdir d 0o755)
    [ Store.root store; dir ];
  Sys.mkdir (Filename.concat dir id) 0o755;
  let start =
    Printf.sprintf
      {|{"type":"start","time":%d.0,"format":1,"experiment":"%s",%s"params":{},"provenance":{"command":["t"],"cwd":"/","pid":1,"ocaml":"5","raven":"t","os":"t","arch":"t","seeds":{},"env":{}}}|}
      started experiment
      (match steps with
      | Some n -> Printf.sprintf {|"steps":%d,|} n
      | None -> "")
  in
  let log = Filename.concat (Filename.concat dir id) "events.jsonl" in
  Out_channel.with_open_bin log (fun oc ->
      List.iter
        (fun l -> output_string oc (l ^ "\n"))
        ((start :: declare) @ records));
  (id, log)

let sample_record (step, v) =
  Printf.sprintf {|{"type":"metrics","time":2.0,"step":%d,"values":{"m":%s}}|}
    step (Fixture.num v)

let resume_record step =
  Printf.sprintf
    {|{"type":"resume","time":3.0,"step":%d,"provenance":{"command":["t"],"cwd":"/","pid":1,"ocaml":"5","raven":"t","os":"t","arch":"t","seeds":{},"env":{}}}|}
    step

(* Laws of the reduction *)

(* [pairs l] is each sample of [l] with the next. *)
let pairs l =
  match l with
  | [] -> []
  | _ :: rest ->
      List.combine (List.filteri (fun k _ -> k < List.length rest) l) rest

(* The laws' runs, written from the format's text. *)
let laws = lazy (Store.open_ ~root:(Filename.concat home "laws") ())

(* [series_of samples] is the series of a run that logged [samples], each a step
   and a value, under one key. *)
let series_of samples =
  let store = Lazy.force laws in
  let id, _ = write_log store (List.map sample_record samples) in
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
  let* dots = int_range 1 320 in
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
  prop "each dot column with a sample that is not finite is marked, NaN first"
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
      cover "a column holds NaN and an infinity"
        (List.exists (fun (_, v) -> Float.is_nan v) samples
        && List.exists
             (fun (_, v) -> Float.is_finite v = false && not (Float.is_nan v))
             samples);
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
      let id, log = write_log store (List.map sample_record first) in
      let h = History.read ~series:[ "m" ] (Option.get (Store.run store id)) in
      Out_channel.with_open_gen [ Open_append; Open_binary ] 0o644 log
        (fun oc ->
          List.iter (fun r -> output_string oc (sample_record r ^ "\n")) rest);
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
          [ (me, 0, series_of samples) ]
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
      Windtrap.not_equal ~msg:"the labels shown" (Windtrap.list Windtrap.string) []
        (List.filter (fun s -> holds s shown) [ "× NaN"; "× inf"; "× -inf" ]);
      at_most ~msg:"the line's last cell, the hatch's first" Windtrap.int
        ~than:(List.fold_left min max_int hatch)
        (List.fold_left max 0 line))

(* Law 7: the cursor's run is drawn last and wins every cell it shares. *)
let cursor_wins =
  prop "the cursor's run wins the cells an overlay shares with it"
    (gen_draw_case ())
    (fun c ->
      let s = series_of c.samples in
      let ch = chart ~guides:c.guides (domain c.samples) in
      let g =
        drawn truecolor ch ~width:c.width ~height:c.height
          [ (Munin_tui.Plot.Overlay 'a', 1, s); (me, 0, s) ]
      in
      let other = fg (Munin_tui.Theme.run truecolor 1) in
      equal (list (pair int int)) [] (ink ~colour:other g))

(* Law 8: without colour, each overlay ends in its letter. *)
let letters =
  prop "without colour an overlay ends in its letter, with colour it does not"
    (gen_draw_case ~min_height:3 ())
    (fun c ->
      let s = series_of c.samples in
      let ch = chart ~guides:c.guides (domain c.samples) in
      let runs = [ (Munin_tui.Plot.Overlay 'd', 1, s); (me, 0, s) ] in
      let letter th = cells (drawn th ch ~width:(c.width + 10) ~height:c.height runs) (( = ) "d") in
      equal ~msg:"in colour" (list int) [] (letter truecolor);
      if letter (theme `Mono) = [] then fail "no letter without colour")

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

(* Law 1's x domain: from 0 to the greatest end of the runs drawn, a run's
   [steps], or, without them or past them, its largest step rounded up to 1,
   2 or 5 times a power of ten. [facts ?steps last] is what a poll knows of a
   run that logged its samples up to the step [last]. *)
let facts ?steps last =
  let store = Lazy.force laws in
  let id, _ =
    write_log store ?steps (List.map (fun st -> sample_record (st, 1.)) (List.init (last + 1) Fun.id))
  in
  let r = Option.get (Store.run store id) in
  Munin_tui.Facts.of_run ~now:0. ~history:(Some (History.read r)) ~hint:None ~watch:None
    ~pauses:[] r

let x_domain =
  cases
    ~id:(fun (name, _, _) -> name)
    "the x domain is the runs' steps, else their largest step rounded up"
    [
      ("declared, not reached", [ (Some 100, 50) ], 100.);
      ("declared, reached", [ (Some 100, 100) ], 100.);
      ("past the declared", [ (Some 100, 150) ], 200.);
      ("a power of ten", [ (None, 10_000) ], 10_000.);
      ("just past one", [ (None, 10_001) ], 20_000.);
      ("to a five", [ (None, 2_001) ], 5_000.);
      ("to a two", [ (None, 1_500) ], 2_000.);
      ("one step", [ (None, 1) ], 1.);
      ("the greatest of two", [ (Some 300, 10); (None, 400) ], 500.);
    ]
    (fun (_, runs, expected) ->
      let fs = List.map (fun (steps, last) -> facts ?steps last) runs in
      equal (pair float_exact float_exact) (0., expected) (Plot.domain fs))

(* Law 9: a live run's latest step has its [now] rule wherever it is in the
   x domain, which is closed: a run whose latest step is the domain's end,
   as one past 1,000 steps without [steps] reaches at step 1,000, has it. *)
let now_rule =
  cases ~id:(fun v -> Printf.sprintf "now at %g" v) "the now rule marks the latest step"
    [ 0.; 500.; 1000. ]
    (fun v ->
      let samples = List.init 11 (fun i -> (i * 100, Float.of_int i)) in
      let ch = { (chart (0., 1000.)) with now = Some v } in
      let g = drawn truecolor ch ~width:60 ~height:10 [ (me, 0, series_of samples) ] in
      let text =
        String.concat "\n"
          (List.init (Matrix.Grid.height g) (fun y ->
               String.concat ""
                 (List.init (Matrix.Grid.width g) (fun x ->
                      Matrix.Grid.get_text g (Matrix.Grid.idx g ~x ~y)))))
      in
      let rec count i =
        if i + 3 > String.length text then 0
        else (if String.sub text i 3 = "now" then 1 else 0) + count (i + 1)
      in
      equal ~msg:"the now labels" int 1 (count 0))

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
  let id, _ = write_log store (List.mapi (fun st v -> sample_record (st, v)) values) in
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
      ("348513", (348513., "348513"));
      ("-999999", (-999999., "-999999"));
      ("348500", (348500., "348500"));
      ("1234.56", (1234.56, "1.235k"));
      ("348513.5", (348513.5, "348.5k"));
      ("1234567", (1234567., "1.235e+06"));
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
        (2, whole 1_000_000_000);
        (2, Gen.float_range (-1e6) 1e6);
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
      let g = drawn truecolor ch ~width:c.width ~height:c.height [ (me, 0, series_of c.samples) ] in
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
      let line k = rows (ink ~colour:mine (drawn truecolor ch ~width ~height:12 [ (me, 0, series_of (every k)) ])) in
      let a = line 1 and b = line 50 in
      equal ~msg:"dot columns" Windtrap.(list int) (List.map fst a) (List.map fst b);
      List.iter2
        (fun (k, (top, bottom)) (_, (top', bottom')) ->
          let msg = Printf.sprintf "dot column %d" k in
          at_most ~msg Windtrap.int ~than:cadence_rows (abs (top - top'));
          at_most ~msg Windtrap.int ~than:cadence_rows (abs (bottom - bottom')))
        a b)

let values_record step kvs =
  Printf.sprintf {|{"type":"metrics","time":2.0,"step":%d,"values":{%s}}|} step
    (String.concat ","
       (List.map
          (fun (k, v) -> Printf.sprintf {|"%s":%s|} k (Fixture.num v))
          kvs))

(* Runs: the filter and the best value of each experiment *)

(* [before row sep] is [row] up to the first [sep] in it. *)
let before row sep =
  let n = String.length sep in
  let rec find i =
    if i + n > String.length row then None
    else if String.sub row i n = sep then Some (String.sub row 0 i)
    else find (i + 1)
  in
  find 0

(* The runs pane: the text of each row before the pane's rule. *)
let pane screen y =
  let row = T.Screen.row screen y in
  match before row "│" with Some l -> l | None -> row

(* The names of the runs pane's rows, and whether a row holds a star. *)
let shown screen =
  let rec go y acc =
    if y >= T.Screen.rows screen - 1 then List.rev acc
    else
      let fields =
        List.filter (( <> ) "") (String.split_on_char ' ' (pane screen y))
      in
      match
        List.find_opt
          (fun f -> List.exists (fun r -> r.Fixture.name = f) Fixture.runs)
          fields
      with
      | Some name -> go (y + 1) ((name, List.mem "★" fields) :: acc)
      | None -> go (y + 1) acc
  in
  go 1 []

let filter =
  cases "the filter keeps the rows that match each word"
    ~id:(fun (f, _) -> f)
    [
      ( "cifar",
        [
          "tidy-otter";
          "amber-fox";
          "brisk-wren";
          "calm-lynx";
          "dusty-mole";
          "eager-hawk";
        ] );
      ("MNIST", [ "mlp-wide"; "mlp-base" ]);
      ("lr=0.001", [ "amber-fox" ]);
      ("warmup=2000", [ "brisk-wren"; "calm-lynx" ]);
      ("sweep failed", [ "eager-hawk" ]);
      ("mlp crashed", [ "mlp-base" ]);
      ("20261031", [ "mlp-wide"; "mlp-base" ]);
      ("nothing", []);
    ]
    (fun (f, expected) ->
      if List.mem "mlp-base" expected then needs_crashed ();
      with_store Fixture.runs @@ fun store ->
      let screen =
        show ~cols:160 ~rows:48 store Munin_tui.Runs
          [ char '/'; Text f; Key Enter ]
      in
      equal (list string) expected (List.map fst (shown screen)))

let stars =
  test "the star is the best of each experiment among the rows shown" (fun () ->
      with_store Fixture.runs @@ fun store ->
      let starred keys =
        let screen = show ~cols:160 ~rows:48 store Munin_tui.Runs keys in
        List.filter_map
          (fun (n, star) -> if star then Some n else None)
          (shown screen)
      in
      equal (list string) [ "amber-fox"; "mlp-wide" ] (starred []);
      equal (list string) [ "calm-lynx" ]
        (starred [ char '/'; Text "warmup=2000"; Key Enter ]))

(* A run 50 steps short of its 20,000 is at 99.75%: it reads 99% on Run and on
   Runs, never 100% before its end. *)
let short_of_steps =
  test "a run short of its steps never reads 100%" (fun () ->
      with_store ~name:"short" [] @@ fun store ->
      let id, _ =
        write_log store ~goal:"minimize" ~steps:20000
          [ sample_record (0, 1.); sample_record (19950, 0.5) ]
      in
      let run = show ~cols:120 ~rows:40 store (Munin_tui.Run (Some id)) [] in
      let runs = show ~cols:120 ~rows:40 store Munin_tui.Runs [] in
      let found screen s = T.Screen.find screen s <> [] in
      equal (list bool)
        [ true; false; true; false ]
        [
          found run "99%";
          found run "100%";
          found runs " 99%";
          found runs "100%";
        ])

(* Runs that log as programs do: the cursor's every 10 steps with the value
   [step / 1000], the overlaid run [marked] samples of [0]. A strip compares
   the two at the latest step both logged, so its difference is the cursor's
   value there. *)
let paired =
  cases "a strip compares the run with its overlay at the same step"
    ~id:(fun (name, _, _) -> name)
    [
      ("the overlay logs every 100 steps", List.init 21 (fun k -> k * 100), "+0.9000");
      ( "the overlay ends in the cursor's last column",
        List.init 99 (fun k -> k * 10) @ [ 989 ],
        "+0.9800" );
    ]
    (fun (_, marked, value) ->
      with_store ~name:"paired" [] @@ fun store ->
      ignore
        (write_log store ~goal:"minimize" ~started:1
           (List.map (fun st -> sample_record (st, 0.)) marked));
      ignore
        (write_log store ~goal:"minimize" ~started:2
           (List.init 100 (fun k ->
                sample_record (k * 10, Float.of_int (k * 10) /. 1000.))));
      (* The cursor's run, the newest, is the first row: overlay the second,
         come back and open the strips. *)
      let screen =
        show ~cols:120 ~rows:40 store Munin_tui.Runs
          [ char 'j'; Text " "; char 'k'; Key Enter ]
      in
      equal bool true (T.Screen.find screen value <> []))

(* A metric whose every sample a resume orphans has no statistics, and Details
   lists the run all the same. *)
let all_orphans =
  test "the run's facts list a run whose metric is all orphans" (fun () ->
      with_store ~name:"orphaned" [] @@ fun store ->
      let id, _ =
        write_log store
          ~declare:
            [
              metric_record ~goal:"minimize" "m";
              metric_record ~goal:"maximize" "val/acc";
            ]
          [
            sample_record (100, 1.);
            values_record 900 [ ("val/acc", 0.5) ];
            resume_record 500;
            sample_record (600, 0.5);
            {|{"type":"end","time":4.0,"status":"finished"}|};
          ]
      in
      let screen =
        show ~cols:120 ~rows:40 store (Munin_tui.Run (Some id)) [ char 'i' ]
      in
      equal bool true (T.Screen.find screen "metrics" <> []))

(* Below 100 columns the nav line counts the cursor's run among the runs
   kept. Past 30 runs every experiment but the cursor's starts folded, and a
   folded header's cursor is on its experiment's first run, which has no line
   of its own: three experiments of 11 runs, the newest open, and 11 moves
   down reach the second's header, whose first run is the 12th kept. *)
let nav_count =
  test "the nav line counts a folded experiment's run among the runs kept"
    (fun () ->
      with_store ~name:"folded" [] @@ fun store ->
      List.iteri
        (fun e experiment ->
          for k = 1 to 11 do
            ignore (write_log store ~experiment ~started:((e * 11) + k) [])
          done)
        [ "first"; "second"; "third" ];
      let screen = show ~cols:90 ~rows:30 store Munin_tui.Runs (times 11 (char 'j')) in
      (* The nav line's count: what follows its [›], up to a double space. *)
      let count =
        match T.Screen.find screen "›  " with
        | (y, x) :: _ ->
            let rest = String.trim (from (x + 1) (T.Screen.row screen y)) in
            Some (Option.value (before rest "  ") ~default:rest)
        | [] -> None
      in
      equal (option string) (Some "12 of 33 · second") count)

(* What the dashboard holds *)

(* A run that a sample arrived from stays live through depths, cursor moves and
   the runs looked at in between: the dashboard holds what it read of a run
   with its arrivals while it shows the run. Each case logs a new step of
   tidy-otter while the dashboard watches, then takes the keys, and reads
   tidy-otter's glyph back in the runs pane, [●] for live. *)
let held =
  let store =
    lazy (Fixture.make ~child (Filename.concat home "held") Fixture.runs)
  in
  let step = ref 12400 in
  let key_gen =
    Gen.with_pp
      (fun ppf ks ->
        Format.pp_print_list ~pp_sep:Format.pp_print_space
          (fun ppf -> function
            | Text t -> Format.fprintf ppf "%S" t
            | Key Enter -> Format.pp_print_string ppf "Enter"
            | Key Escape -> Format.pp_print_string ppf "Esc"
            | Key _ -> Format.pp_print_string ppf "key")
          ppf ks)
      Gen.(
        list ~size:(int_range 0 12)
          (frequency
             [
               (3, constant (Key Enter));
               (3, constant (Key Escape));
               (2, constant (Text "j"));
               (2, constant (Text "k"));
               (1, constant (Text "i"));
               (1, constant (Text "]"));
             ]))
  in
  prop ~count:30 "a live run stays live through depths and cursor moves"
    key_gen (fun keys ->
      let store = Lazy.force store in
      step := !step + 50;
      let script _ = feed store Fixture.tidy_otter ~step:!step in
      cover "the keys open a run" (List.mem (Key Enter) keys);
      let screen =
        show ~script ~cols:120 ~rows:40 store Munin_tui.Runs
          (keys @ times 4 (Key Escape))
      in
      let glyph =
        let rec find y =
          if y >= T.Screen.rows screen then "absent"
          else
            let rec after = function
              | "tidy-otter" :: g :: _ -> Some g
              | _ :: rest -> after rest
              | [] -> None
            in
            match
              after
                (List.filter (( <> ) "")
                   (String.split_on_char ' ' (pane screen y)))
            with
            | Some g -> g
            | None -> find (y + 1)
        in
        find 1
      in
      equal string "●" glyph)

(* The cursor leaves an ended run and comes back: its charts were kept, whole,
   so the dashboard reads nothing of the run, and the log moved away meanwhile
   is never missed. Any fold of it would fail and name the run. *)
let kept_pane =
  test "a cursor back on an ended run reads nothing of it" (fun () ->
      with_store Fixture.runs @@ fun store ->
      let log =
        Filename.concat
          (Run.dir (Option.get (Store.run store (id_of Fixture.amber_fox))))
          "events.jsonl"
      in
      let term = T.Terminal.create ~cols:160 ~rows:48 () in
      (elapsed := fun () -> 0);
      let app = Munin_tui.app ~clock store Munin_tui.Runs in
      let screen =
        Mosaic_test.run term app @@ fun s ->
        (elapsed := fun () -> T.now s);
        T.advance s second;
        List.iter
          (fun k ->
            send term s k;
            T.advance s second)
          [ char 'j'; char 'j' ];
        (* tidy-otter, amber-fox, brisk-wren: the cursor left amber-fox. *)
        Sys.rename log (log ^ ".away");
        List.iter
          (fun k ->
            send term s k;
            T.advance s second)
          [ char 'k' ];
        T.advance s (3 * second);
        T.Terminal.screen term
      in
      Sys.rename (log ^ ".away") log;
      equal (list (pair int int)) [] (T.Screen.find screen "cannot read");
      equal bool true (T.Screen.find screen "amber-fox  finished" <> []))

(* A run logs its last samples and ends after its chart was prepared, and the
   cursor leaves it before a poll reads them: coming back draws them, since a
   chart prepared before the end is not kept. *)
let ended_away =
  test "a cursor back on a run that ended away draws its last samples"
    (fun () ->
      with_store Fixture.runs @@ fun store ->
      let s =
        Session.start ~store ~experiment:"late" ~name:"late-end" ~steps:100
          ~provenance:(Provenance.detect ~cwd:home ())
          ()
      in
      (* A metric of no smoothing, so that the legend gives the raw value. *)
      let loss = Session.metric s ~goal:`Minimize "val/loss" in
      for step = 1 to 50 do
        Session.log loss ~step 1.0
      done;
      let term = T.Terminal.create ~cols:160 ~rows:48 () in
      (elapsed := fun () -> 0);
      let app = Munin_tui.app ~clock store Munin_tui.Runs in
      let screen =
        Mosaic_test.run term app @@ fun t ->
        (elapsed := fun () -> T.now t);
        T.advance t second;
        for step = 51 to 100 do
          Session.log loss ~step 9.0
        done;
        Session.finish s;
        (* late-end, the newest run, is the first row: the cursor leaves it
           before a poll reads its last samples. *)
        send term t (char 'j');
        T.advance t (2 * second);
        send term t (char 'k');
        T.advance t (3 * second);
        T.Terminal.screen term
      in
      (* The last 50 samples, all 9, are what the legend gives at the run's
         last step. *)
      equal bool true (T.Screen.find screen "late-end 9 " <> []);
      equal bool true (T.Screen.find screen "late-end  finished" <> []))

(* A run ends between two timer polls after logging its last samples, while
   the dashboard draws it as a trend only (the cursor is on another run) or as
   the cursor's strips with no readout of its metric: what it draws of the run
   is what it draws of the same run read whole after its end. *)
let ended_between =
  let late ~between keys =
    with_store Fixture.runs @@ fun store ->
    let s =
      Session.start ~store ~experiment:"late" ~name:"late-end" ~steps:100
        ~provenance:(Provenance.detect ~cwd:home ())
        ()
    in
    let loss = Session.metric s ~goal:`Minimize "val/loss" in
    let finish () =
      for step = 51 to 100 do
        Session.log loss ~step 9.0
      done;
      Session.finish s
    in
    for step = 1 to 50 do
      Session.log loss ~step 1.0
    done;
    if not between then finish ();
    let term = T.Terminal.create ~cols:160 ~rows:48 () in
    (elapsed := fun () -> 0);
    let app = Munin_tui.app ~clock store Munin_tui.Runs in
    Mosaic_test.run term app @@ fun t ->
    (elapsed := fun () -> T.now t);
    T.advance t second;
    List.iter
      (fun k ->
        send term t k;
        T.advance t second)
      keys;
    T.advance t (2 * second);
    if between then finish ();
    T.advance t (3 * second);
    T.Terminal.screen term
  in
  (* The trend is in the runs pane's row; the strip in the centre's rows. *)
  let trend screen =
    match T.Screen.find screen "late-end" with
    | (y, _) :: _ -> pane screen y
    | [] -> fail "no row of late-end"
  in
  let strip screen =
    match T.Screen.find screen "▌val/loss" with
    | (y, x) :: _ ->
        let part y =
          let r = from x (T.Screen.row screen y) in
          Option.value (before r "│") ~default:r
        in
        (* A strip is two rows high. *)
        part y ^ "\n" ^ part (y + 1)
    | [] -> fail "no strip of val/loss"
  in
  cases ~id:(fun (n, _, _) -> n) "a run that ends between polls is drawn as if read whole"
    [ ("trend", [ char 'j' ], trend); ("strip", [ Key Enter ], strip) ]
    (fun (_, keys, part) ->
      equal string
        (part (late ~between:false keys))
        (part (late ~between:true keys)))

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
            [ "running"; "live"; "stalled"; "writing"; "silent" ];
          w)
        (fun st (w, now, _) -> Watch.word (Watch.state st (Some w) ~now));
    ]

(* The depths *)

(* The text typed in the filter is drawn from characters that no run of the
   fixture has in a name, id, experiment, tag, state or parameter, so a filter
   with text keeps no run. Enter and [i] then have no run to act on. *)
let no_run = [ 'q'; '?'; 'j' ]

type depth_state = {
  depth : int; (* 0 charts, 1 strips, 2 zoom *)
  help : bool;
  typing : bool;
  info : bool;
  filter : bool; (* the kept filter has text *)
  typed : bool; (* the filter being typed has text *)
}

let keeps_none st = if st.typing then st.typed else st.filter

let model_key st k =
  match k with
  | _ when st.typing -> (
      match k with
      | Key Enter -> { st with typing = false; filter = st.typed }
      | Key Escape -> { st with typing = false; filter = false }
      | _ -> { st with typed = true })
  | _ when st.help -> (
      match k with Text "?" | Key Escape -> { st with help = false } | _ -> st)
  | _ when st.info -> (
      match k with Text "i" | Key Escape -> { st with info = false } | _ -> st)
  | Text "?" -> { st with help = true }
  | Key Escape when st.depth = 0 -> { st with filter = false }
  | Key Escape -> { st with depth = st.depth - 1 }
  | (Key Enter | Text "i") when keeps_none st -> st
  | Key Enter -> { st with depth = min 2 (st.depth + 1) }
  | Text "i" -> { st with info = true }
  | Text "/" -> { st with typing = true; typed = st.filter }
  | _ -> st

(* What the dashboard shows, from its status bar: the keys of help, of the
   filter and of the run's facts, or else the depth its crumbs set in bold. *)
let observed screen =
  let row = T.Screen.rows screen - 1 in
  let last = T.Screen.row screen row in
  let bold col =
    let _, style = T.Screen.cell screen ~row ~col in
    Matrix.Ansi.Attr.mem Bold style.Matrix.Ansi.Style.attrs
  in
  (* The crumbs follow blanks, one column each, so a byte offset in the row
     is their column; each crumb is 9 columns after the one before. *)
  let crumbs = before last "charts › strips › zoom" in
  if holds "↑↓ scroll" last then "info"
  else if holds "esc close" last then "help"
  else if holds "enter keep" last then "typing"
  else
    match crumbs with
    | None -> "unknown: " ^ String.trim last
    | Some lead -> (
        let at = String.length lead in
        match List.find_opt (fun (d, _) -> bold (at + (9 * d))) [ (0, "charts"); (1, "strips"); (2, "zoom") ] with
        | Some (_, name) -> name
        | None -> "unknown: " ^ String.trim last)

let model_name st =
  if st.help then "help"
  else if st.typing then "typing"
  else if st.info then "info"
  else match st.depth with 0 -> "charts" | 1 -> "strips" | _ -> "zoom"

let depth_test =
  let store =
    lazy (Fixture.make ~child (Filename.concat home "depths") Fixture.runs)
  in
  let dash = abstract "d" in
  let key_gen =
    Gen.of_list
      ~pp:(fun ppf k ->
        Format.pp_print_string ppf
          (match k with
          | Text t -> Printf.sprintf "%S" t
          | Key Enter -> "Enter"
          | Key Escape -> "Esc"
          | Key _ -> "key"))
      [
        Key Enter; Key Escape; Text "i"; Text "?"; Text "/"; Text "q"; Text "j";
      ]
  in
  let start =
    { depth = 0; help = false; typing = false; info = false; filter = false; typed = false }
  in
  let word = Gen.(string_of ~size:(int_range 1 3) (of_list no_run)) in
  let typed k =
    List.mem k (Key Enter :: Key Escape :: List.map (fun c -> Text (String.make 1 c)) no_run)
  in
  stateful ~count:60 ~steps:12
    "Enter goes deeper, Esc back, q is typed in the filter"
    [
      command "runs" (Gen.unit @-> makes dash) (fun () -> start) (fun () -> []);
      command "run"
        (Gen.unit @-> makes dash)
        (fun () -> start)
        (fun () -> [ `Start_run ]);
      command "key"
        ~pre:(fun k st -> if st.typing then typed k else k <> Text "q")
        (key_gen @-> dash ^-> makes dash)
        (fun k st -> model_key st k)
        (fun k keys -> `Key k :: keys);
      (* A filter kept with text, which keeps no run. *)
      command "filter"
        ~pre:(fun _ st -> not (st.typing || st.help || st.info))
        (word @-> dash ^-> makes dash)
        (fun _ st -> { st with filter = true })
        (fun w keys ->
          let chars = List.of_seq (String.to_seq w) in
          (`Key (Key Enter) :: List.rev_map (fun c -> `Key (Text (String.make 1 c))) chars)
          @ (`Key (Text "/") :: keys));
      command "screen"
        (dash ^-> returns string)
        (fun st ->
          cover "a kept filter keeps no run" (st.filter && not st.typing);
          model_name st)
        (fun keys ->
          let keys = List.rev keys in
          let start, keys =
            match keys with
            | `Start_run :: rest ->
                (Munin_tui.Run (Some (id_of Fixture.tidy_otter)), rest)
            | l -> (Munin_tui.Runs, l)
          in
          let keys =
            List.filter_map
              (function `Key k -> Some k | `Start_run -> None)
              keys
          in
          let term = T.Terminal.create ~cols:120 ~rows:40 () in
          (elapsed := fun () -> 0);
          let app = Munin_tui.app ~clock (Lazy.force store) start in
          Mosaic_test.run term app @@ fun s ->
          (elapsed := fun () -> T.now s);
          T.advance s second;
          List.iter (send term s) keys;
          T.settle s;
          observed (T.Terminal.screen term));
    ]

(* Where the runs pane is hidden, the filter shows it while it is typed:
   the text typed is in its line, and Enter keeps it. *)
let hidden_filter =
  test "the filter shows the runs pane where it is hidden" (fun () ->
      with_store ~name:"hidden" Fixture.runs @@ fun store ->
      List.iter
        (fun (what, cols, rows, hide) ->
          let typing = show ~cols ~rows store Munin_tui.Runs (hide @ [ char '/'; Text "cifar" ]) in
          let lines =
            List.filter (holds "/ cifar")
              (List.init (T.Screen.rows typing) (T.Screen.row typing))
          in
          equal ~msg:(what ^ ": lines with the filter typed") int 1 (List.length lines);
          let kept = show ~cols ~rows store Munin_tui.Runs (hide @ [ char '/'; Text "cifar"; Key Enter ]) in
          equal ~msg:(what ^ ": after Enter") string "charts" (observed kept))
        [ ("80 columns", 80, 24, []); ("hidden by 1", 120, 40, [ char '1' ]) ])

(* The strips scroll *)

(* At 120 columns the strips own the centre: a strip's first row holds [▌]
   at column 35 for the cursor's, or a blank, then its label. *)
let strip_col = 35

(* [strips screen] is the label of each strip whose two rows are both above
   the status bar, by row, and the cursor's. *)
let strips screen =
  let last = T.Screen.rows screen - 2 in
  let at row col = fst (T.Screen.cell screen ~row ~col) in
  let label row =
    String.trim
      (String.concat "" (List.init 16 (fun i -> at row (strip_col + 1 + i))))
  in
  let firsts =
    List.filter
      (fun row ->
        List.mem (at row strip_col) [ " "; "▌" ]
        && not (List.mem (at row (strip_col + 1)) [ " "; ""; "↓"; "↑" ]))
      (List.init last Fun.id)
  in
  let cursor = List.find_opt (fun row -> at row strip_col = "▌") firsts in
  (List.map label firsts, Option.map label cursor)

(* The strips scroll the least that shows the cursor's strip whole, from the
   strips shown: a move to a strip shown leaves them where they are. *)
let strips_scroll =
  let store =
    lazy (Fixture.make ~child (Filename.concat home "strips") Fixture.runs)
  in
  let move = Gen.of_list ~pp:(fun ppf k -> Format.pp_print_string ppf (if k = "j" then "down" else "up")) [ "j"; "k" ] in
  (* Twelve moves down pass the strips 20 rows show, so every case scrolls;
     the moves drawn after go either way. *)
  let down = List.init 12 (fun _ -> "j") in
  prop ~count:15 "a strip's move scrolls the least that shows it"
    (Gen.list ~size:(Gen.int_range 0 16) move)
    (fun moves ->
      let moves = down @ moves in
      let term = T.Terminal.create ~cols:120 ~rows:20 () in
      (elapsed := fun () -> 0);
      let start = Munin_tui.Run (Some (id_of Fixture.tidy_otter)) in
      let app = Munin_tui.app ~clock (Lazy.force store) start in
      Mosaic_test.run term app @@ fun s ->
      (elapsed := fun () -> T.now s);
      T.advance s second;
      send term s (Key Enter);
      T.advance s second;
      let scrolled = ref false in
      ignore
        (List.fold_left
           (fun (shown, _) k ->
             send term s (Text k);
             T.advance s second;
             let screen = T.Terminal.screen term in
             let shown', cursor = strips screen in
             (match cursor with
             | None -> fail ("the cursor's strip is not shown whole:\n" ^ T.Screen.text screen)
             | Some c ->
                 if List.mem c shown then equal ~msg:("after " ^ k) (list string) shown shown'
                 else scrolled := true);
             (shown', cursor))
           (strips (T.Terminal.screen term))
           moves);
      cover "a move scrolls" !scrolled)

(* The data path *)

(* One poll at a time. A poll asked for while one runs waits for it, and the
   asks made meanwhile make one poll after it. The polls run off the loop:
   here each waits in a queue until the test releases the oldest. A tick
   asks for one, and so does a key that moves the cursor, once the first
   poll chose it; a poll's result may ask for another, for what the screen
   it chose lacks. *)
type phase = Idle | Reading | Again

type path = { phase : phase; cursor : int; chosen : bool }

let data_path_test =
  let store =
    lazy (Fixture.make ~child (Filename.concat home "path") Fixture.runs)
  in
  let last = List.length Fixture.runs - 1 in
  let path = abstract "p" in
  (* The system: the calls so far, replayed on a fresh dashboard each time,
     and the number of polls that wait after the last. *)
  let replay calls =
    let term = T.Terminal.create ~cols:120 ~rows:40 () in
    let held = Queue.create () in
    (elapsed := fun () -> 0);
    let app = Munin_tui.app ~clock (Lazy.force store) Munin_tui.Runs in
    Mosaic_test.run term app
      ~process_perform:(fun job -> Queue.push job held)
      ~quiesce:ignore
    @@ fun s ->
    (elapsed := fun () -> T.now s);
    T.settle s;
    List.iter
      (function
        | `Tick -> T.advance s second
        | `Key k -> send term s k
        | `Release ->
            if not (Queue.is_empty held) then (Queue.pop held) ();
            T.settle s)
      (List.rev calls);
    Queue.length held
  in
  let call c calls =
    calls := c :: !calls;
    replay !calls
  in
  let waiting p = match p.phase with Idle -> 0 | Reading | Again -> 1 in
  let ask p = p := { !p with phase = (if !p.phase = Idle then Reading else Again) } in
  let key = Gen.of_list ~pp:Format.pp_print_string [ "j"; "k" ] in
  let tick =
    command "tick"
      (path ^-> returns int)
      (fun p ->
        ask p;
        waiting !p)
      (call `Tick)
  in
  stateful ~count:30 ~steps:10
    "one poll runs at a time, and the asks meanwhile make one after it"
    [
      command "start"
        (Gen.unit @-> makes path)
        (fun () -> ref { phase = Reading; cursor = 0; chosen = false })
        (fun () -> ref []);
      (* Listed twice, so that asks pile up while a poll runs. *)
      tick;
      tick;
      command "key"
        (key @-> path ^-> returns int)
        (fun k p ->
          let cursor = max 0 (min last (!p.cursor + if k = "j" then 1 else -1)) in
          if !p.chosen && cursor <> !p.cursor then begin
            p := { !p with cursor };
            ask p
          end;
          waiting !p)
        (fun k -> call (`Key (Text k)));
      (* The first poll chooses the cursor, so it asks for another; asks made
         while a poll ran make one after it; any other poll may ask for what
         the screen it read lacks. *)
      command "release"
        (path ^-> judges int)
        (fun p outcome ->
          let n = match outcome with Ok n -> n | Error e -> raise e in
          (match !p.phase with
          | Idle -> equal ~msg:"nothing waits" int 0 n
          | Again -> equal ~msg:"the asks made one poll" int 1 n
          | Reading when not !p.chosen -> equal ~msg:"the cursor chosen" int 1 n
          | Reading -> less ~msg:"one poll at most" int ~than:2 n);
          cover "a poll runs after asks made meanwhile" (!p.phase = Again);
          let chosen = !p.chosen || !p.phase <> Idle in
          p := { !p with phase = (if n = 0 then Idle else Reading); chosen })
        (call `Release);
    ]

(* The status bar's hints *)

(* [hints screen] is the status bar's hints, each its keys, in bold, and its
   word; hints are three spaces or more apart, and the crumbs, which hold
   [›], are no hint. *)
let hints screen =
  let row = T.Screen.rows screen - 1 in
  let cells =
    List.init (T.Screen.cols screen) (fun col ->
        let ch, style = T.Screen.cell screen ~row ~col in
        (ch, Matrix.Ansi.Attr.mem Bold style.Matrix.Ansi.Style.attrs))
  in
  (* Fields: runs of cells broken by three blanks or more. *)
  let rec fields acc field blanks = function
    | [] -> List.rev (if field = [] then acc else List.rev field :: acc)
    | ((" " | ""), _) :: rest when blanks >= 2 ->
        fields (if field = [] then acc else List.rev field :: acc) [] 0 rest
    | ((" " | ""), _) as c :: rest -> fields acc (c :: field) (blanks + 1) rest
    | c :: rest -> fields acc (c :: field) 0 rest
  in
  let text cs = String.trim (String.concat "" (List.map fst cs)) in
  List.filter_map
    (fun field ->
      let label = List.filter snd field and word = List.filter (fun c -> not (snd c)) field in
      let all = text field in
      if all = "" || before all "›" <> None then None else Some (text label, text word))
    (fields [] [] 0 cells)

(* The keys that press a hint's label: both ways for a pair of arrows, since
   one way may be at its end. *)
let presses = function
  | "↑↓" -> [ Key Down; Key Up ]
  | "←→" -> [ Key Right; Key Left ]
  | "[ ]" -> [ Text "]"; Text "[" ]
  | "+ −" -> [ Text "+"; Text "-" ]
  | "enter" -> [ Key Enter ]
  | "esc" -> [ Key Escape ]
  | "space" -> [ Text " " ]
  | l -> [ Text l ]

(* What a hint's word says its key does, from where the screen is: the screen
   it leads to, or a change of what the screen shows. *)
let meaning ~from word =
  match word with
  | ("charts" | "strips" | "zoom") when word <> from -> word
  | "info" | "quit" -> word
  | "filter" -> "typing"
  | "keys" -> "help"
  | "keep the filter" | "clear it" | "close" -> "charts"
  | _ -> "changed"

let hint_test =
  let store =
    lazy (Fixture.make ~child (Filename.concat home "hints") Fixture.runs)
  in
  (* [session keys] is the screen after [keys], and whether the dashboard
     quit. *)
  let session keys =
    let term = T.Terminal.create ~cols:160 ~rows:48 () in
    (elapsed := fun () -> 0);
    let app = Munin_tui.app ~clock (Lazy.force store) Munin_tui.Runs in
    Mosaic_test.run term app @@ fun s ->
    (elapsed := fun () -> T.now s);
    T.advance s second;
    List.iter
      (fun k ->
        send term s k;
        if not (T.ended s) then T.advance s second)
      keys;
    (T.Terminal.screen term, T.ended s)
  in
  let help = lazy (T.Screen.text (fst (session [ char '?' ]))) in
  let holds sub s = before s sub <> None in
  (* A filter of [q] keeps no run. *)
  let none = [ char '/'; char 'q'; Key Enter ] in
  cases "each hint on the status bar does what help says" ~id:(fun (name, _, _) -> name)
    [
      ("charts", "charts", []);
      ("strips", "strips", [ Key Enter ]);
      ("zoom", "zoom", [ Key Enter; Key Enter ]);
      ("typing", "typing", [ char '/' ]);
      ("info", "info", [ char 'i' ]);
      ("help", "help", [ char '?' ]);
      ("charts, no run kept", "charts", none);
      ("strips, no run kept", "strips", Key Enter :: none);
      ("typing, no run kept", "typing", [ char '/'; char 'q' ]);
    ]
    (fun (_, from, keys) ->
      let screen, _ = session keys in
      equal ~msg:"the screen" string from (observed screen);
      let hints = hints screen in
      if hints = [] then fail "no hint on the status bar";
      (* The depths' keys are help's. *)
      if List.mem from [ "charts"; "strips"; "zoom" ] then
        equal ~msg:"keys help lacks" (list string) []
          (List.filter_map
             (fun (label, _) ->
               if holds label (Lazy.force help) then None else Some label)
             hints);
      let outcome k =
        let after, ended = session (keys @ [ k ]) in
        if ended then "quit"
        else
          let o = observed after in
          if o <> from then o
          else if T.Screen.diff screen after <> None then "changed"
          else "unchanged"
      in
      let does (label, _) =
        let outcomes = List.map outcome (presses label) in
        Option.value ~default:"unchanged"
          (List.find_opt (( <> ) "unchanged") outcomes)
      in
      let pairs f = List.map (fun (l, w) -> (l ^ " " ^ w, f (l, w))) hints in
      equal
        (list (pair string string))
        (pairs (fun (_, w) -> meaning ~from w))
        (pairs does))

(* [y] asks for the other scale of the metric shown, and of no other: zoom's
   title says [log] while the metric is drawn on one. [m] rises over three
   decades and [w] within one, so [m] starts on a log scale and [w] on a
   linear one. *)
let y_per_metric =
  let zoom_m = [ Key Enter; Key Enter ] and zoom_w = [ Key Enter; Key Down; Key Enter ] in
  cases
    ~id:(fun (name, _, _, _) -> name)
    "y switches the scale of the metric shown alone"
    [
      ("m", zoom_m, "m", true);
      ("w", zoom_w, "w", false);
      ("y on m", zoom_m @ [ char 'y' ], "m", false);
      ("y on w", zoom_w @ [ char 'y' ], "w", true);
      ("y on m, then w", zoom_m @ [ char 'y'; Key Escape; Key Down; Key Enter ], "w", false);
      ("y on w, then m", zoom_w @ [ char 'y'; Key Escape; Key Up; Key Enter ], "m", true);
    ]
    (fun (_, keys, key, log) ->
      with_store ~name:"scales" [] @@ fun store ->
      ignore
        (write_log store
           ~declare:[ metric_record ~goal:"minimize" "m"; metric_record "w" ]
           (List.init 100 (fun i ->
                let t = Float.of_int i /. 99. in
                values_record i [ ("m", 10. ** (3. *. t)); ("w", 1. +. t) ])));
      let screen = show ~cols:120 ~rows:40 store Munin_tui.Runs keys in
      (* Zoom's title: the metric, then its notes. *)
      let notes =
        match T.Screen.find screen (key ^ "   ") with
        | (y, x) :: _ -> Some (String.trim (from (x + String.length key) (T.Screen.row screen y)))
        | [] -> None
      in
      let rec parts n =
        match before n " · " with
        | Some a ->
            let k = String.length a + String.length " · " in
            String.trim a :: parts (String.sub n k (String.length n - k))
        | None -> [ String.trim n ]
      in
      equal (option bool) (Some log) (Option.map (fun n -> List.mem "log" (parts n)) notes))

let () =
  exit
    (run "munin.tui"
       [
         screens;
         states_test;
         group "reduce"
           [
             words_a_sample;
             common;
             means;
             smoothed_means;
             geometric_means;
             unsmoothed;
             events;
             edges;
             at;
             extend;
           ];
         group "charts"
           [
             binned; rising; spike; non_finite; cursor_wins; letters; exact_labels; tick_labels; x_domain; now_rule; bounds;
             ticks_of;
             log_choice; y_per_metric; values_printed; rounding_shown;
             filled;
             matching; cadence;
           ];
         group "runs" [ filter; stars; nav_count ];
         group "run" [ all_orphans; short_of_steps; paired ];
         group "held" [ held; kept_pane; ended_away; ended_between; watch_test ];
         depth_test;
         hidden_filter;
         strips_scroll;
         data_path_test;
         hint_test;
       ])
