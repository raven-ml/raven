(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Munin
module Curve = Munin_tui.Curve
module T = Matrix_test

(* Stores *)

(* The stores live under a directory that the tests make the home, so that a
   screen names one as [~/store], and times print in UTC. The screens draw for
   a terminal of 24-bit colour, whatever terminal runs the tests, as the
   goldens state: below 256 colours the cursor row also shows a marker. *)
let home = Filename.temp_dir "munin_tui" ""

let depth = `Truecolor

let () =
  Unix.putenv "HOME" home;
  Unix.putenv "TZ" "UTC";
  Munin_tui.set_depth depth

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
let show ?(script = ignore) ~cols ~rows store screen keys =
  let term = T.Terminal.create ~cols ~rows () in
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

(* The main store's screens, after tidy-otter logged its step 12,400 while the
   dashboard watched, so that an arrival makes it live. *)
let screen_test name ~cols ~rows ?(start = Munin_tui.Runs) ?(crashed = false)
    ?depth:(d = depth) keys =
  test name (fun () ->
      if crashed then needs_crashed ();
      Munin_tui.set_depth d;
      Fun.protect ~finally:(fun () -> Munin_tui.set_depth depth) @@ fun () ->
      with_store Fixture.runs @@ fun store ->
      let script _ = feed store Fixture.tidy_otter ~step:12400 in
      golden name (T.Screen.text (show ~script ~cols ~rows store start keys)))

let char c = Text (String.make 1 c)
let times n k = List.init n (fun _ -> k)

(* On Runs, the focus metric starts on train/loss, the first by name of the
   metrics with a goal that the most rows have. Five moves right, through the
   metrics any row has by name, make it val/loss: train/lr, train/samples_s,
   val/acc, val/ece, val/loss. amber-fox is the second row and dusty-mole the
   fifth. *)
let to_val_loss = times 5 (char 'l')

(* amber-fox, and dusty-mole, whose losses turn NaN: the attention line names a
   marked run's NaN. *)
let mark_two =
  [ char 'j'; Text " "; char 'j'; char 'j'; char 'j'; Text " "; char 'g' ]

let open_run = to_val_loss @ mark_two @ [ Key Enter ]

let screens =
  group "screens"
    [
      screen_test "runs-80x24" ~cols:80 ~rows:24 ~crashed:true to_val_loss;
      screen_test "runs-160x48" ~cols:160 ~rows:48 ~crashed:true to_val_loss;
      screen_test "runs-80x24-filter" ~cols:80 ~rows:24 ~crashed:true
        (to_val_loss @ [ char '/'; Text "cifar" ]);
      (* Below 256 colours the cursor row is reverse video, marked [›]. *)
      screen_test "runs-80x24-16-colours" ~cols:80 ~rows:24 ~crashed:true
        ~depth:`Ansi16 to_val_loss;
      screen_test "run-80x24" ~cols:80 ~rows:24 open_run;
      screen_test "run-160x48" ~cols:160 ~rows:48 open_run;
      (* Two marked runs of the sweep that run under the focus: the difference
         rows show what the overlay hides. *)
      screen_test "run-120x40-difference" ~cols:120 ~rows:40
        (to_val_loss
        @ [
            char 'j';
            Text " ";
            char 'j';
            char 'j';
            Text " ";
            char 'g';
            Key Enter;
          ]);
      (* Zoom's step cursor starts on the latest val/loss sample, 12,000, and
         each move left goes back one sample, 500 steps. *)
      screen_test "zoom-80x24" ~cols:80 ~rows:24
        (open_run @ [ Key Enter ] @ times 8 (Key Left));
      screen_test "zoom-160x48" ~cols:160 ~rows:48
        (open_run @ [ Key Enter ] @ times 8 (Key Left));
      screen_test "details-80x24" ~cols:80 ~rows:24 (open_run @ [ char 'i' ]);
      screen_test "help-80x24" ~cols:80 ~rows:24 ~crashed:true [ char '?' ];
      screen_test "small-39x9" ~cols:39 ~rows:9
        ~start:(Munin_tui.Run (Some (id_of Fixture.tidy_otter)))
        [];
      test "empty-80x24" (fun () ->
          with_store ~name:"empty" [] @@ fun store ->
          golden "empty-80x24"
            (T.Screen.text (show ~cols:80 ~rows:24 store Munin_tui.Runs [])));
    ]

(* States *)

(* The first line of Run, for a run in each state. tidy-otter reads running
   until a sample arrives, live once one has, and stalled once its usual gap of
   23 s, from two arrivals, has passed three times. far-node and lost-node have
   no writer, so their lock is not probed. *)
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
        T.Screen.row screen 0
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

let write_log store ?goal ?steps ?declare ?(started = 1) records =
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
      {|{"type":"start","time":%d.0,"format":1,"experiment":"law",%s"params":{},"provenance":{"command":["t"],"cwd":"/","pid":1,"ocaml":"5","raven":"t","os":"t","arch":"t","seeds":{},"env":{}}}|}
      started
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

(* Orphans *)

(* [before row sep] is the text of [row] before its first [sep], if any. *)
let before row sep =
  let n = String.length sep in
  let rec find i =
    if i + n > String.length row then None
    else if String.sub row i n = sep then Some (String.sub row 0 i)
    else find (i + 1)
  in
  find 0

(* The greatest label of a y axis in the text of a chart: the numbers before its
   ticks. *)
let top_tick text =
  String.split_on_char '\n' text
  |> List.filter_map (fun row ->
      Option.bind (before row "┤") (fun l ->
          float_of_string_opt (String.trim l)))
  |> List.fold_left Float.max Float.neg_infinity

(* A run that logs a spike at step 8, then resumes from step 7: the spike is an
   orphan, which sets no domain and is counted in Zoom's table. *)
let orphans =
  test "an orphan sets no domain and is counted" (fun () ->
      with_store ~name:"orphans" [] @@ fun store ->
      let samples =
        List.init 9 (fun step ->
            (step, if step = 8 then 9000. else 1. /. Float.of_int (step + 1)))
      in
      let id, _ =
        write_log store ~goal:"minimize"
          (List.map sample_record samples
          @ [ resume_record 7 ]
          @ List.map sample_record [ (8, 0.11); (9, 0.1) ]
          @ [ {|{"type":"end","time":4.0,"status":"finished"}|} ])
      in
      let run = Option.get (Store.run store id) in
      let h = History.read run in
      let g =
        Munin_tui.Chart.draw ~width:60 ~height:12 ~lo:0. ~hi:10. ~key:"m" h []
      in
      at_most float_exact ~than:1.5 (top_tick (Matrix_grid.to_text g));
      let screen =
        show ~cols:160 ~rows:48 store (Munin_tui.Run (Some id)) [ Key Enter ]
      in
      let orphans_column =
        match T.Screen.find screen "orphans" with
        | [ (row, _) ] ->
            (* The focus's row, whose last column is the orphans'. *)
            let fields =
              List.filter (( <> ) "")
                (String.split_on_char ' ' (T.Screen.row screen (row + 1)))
            in
            List.nth fields (List.length fields - 1)
        | found -> failf "orphans found %d times" (List.length found)
      in
      equal string "1" orphans_column)

(* The braille dots of each line of [g]'s text: U+2801 to U+28FF, E2 A0..A3 xx
   in UTF-8. *)
let dots_per_line g =
  let dots line =
    let n = ref 0 in
    for i = 0 to String.length line - 3 do
      let b k = Char.code line.[i + k] in
      if
        b 0 = 0xE2
        && b 1 >= 0xA0
        && b 1 <= 0xA3
        && (b 1, b 2) <> (0xA0, 0x80)
      then incr n
    done;
    !n
  in
  List.map dots (String.split_on_char '\n' (Matrix_grid.to_text g))

(* The y axis labels of [g], top to bottom. *)
let tick_labels g =
  let ticked l =
    let rec at i =
      if i + 3 > String.length l then None
      else if String.sub l i 3 = "┤" then Some (String.trim (String.sub l 0 i))
      else at (i + 1)
    in
    at 0
  in
  List.filter_map ticked (String.split_on_char '\n' (Matrix_grid.to_text g))

(* A metric that stays at the largest float, or at its negation, spans no
   values: its y domain widens around the value and stays finite. *)
let flat_extreme =
  cases "a chart of a value at the largest float draws it"
    ~id:(Printf.sprintf "%h")
    [ Float.max_float; -.Float.max_float ]
    (fun v ->
      with_store ~name:"flat-extreme" [] @@ fun store ->
      let id, _ =
        write_log store (List.init 9 (fun step -> sample_record (step, v)))
      in
      let h = History.read (Option.get (Store.run store id)) in
      let g =
        Munin_tui.Chart.draw ~width:60 ~height:12 ~lo:0. ~hi:10. ~key:"m" h []
      in
      not_equal (list int) [] (List.filter (( < ) 0) (dots_per_line g)))

(* A metric at the largest float, then at its negation, then at 0: its y domain
   is wider than the largest float. The line runs from the top row to the
   bottom one, and the y axis is labelled at multiples of a step of 5e307 on 27
   rows, as on any other domain. *)
let wide_extreme =
  test "a chart wider than the largest float draws across its rows" (fun () ->
      with_store ~name:"wide-extreme" [] @@ fun store ->
      let v step =
        match step with
        | 0 -> Float.max_float
        | 1 -> -.Float.max_float
        | _ -> 0.
      in
      let id, _ =
        write_log store (List.init 9 (fun step -> sample_record (step, v step)))
      in
      let h = History.read (Option.get (Store.run store id)) in
      let g =
        Munin_tui.Chart.draw ~width:60 ~height:28 ~lo:0. ~hi:10. ~key:"m" h []
      in
      at_least int ~than:20
        (List.length (List.filter (( < ) 0) (dots_per_line g)));
      equal (list string)
        [
          "1.798e+308";
          "1.5e+308";
          "1e+308";
          "5e+307";
          "0";
          "-5e+307";
          "-1e+308";
          "-1.5e+308";
          "-1.798e+3…";
        ]
        (tick_labels g))

(* The marks the medium carries *)

(* Three runs of one sweep against the focus, train/loss over steps 2,000 to
   20,000: the case a cap on marked runs must survive. Two marked runs keep
   their letters and their own cells. A third, calm-lynx, falls in the cells of
   the first, amber-fox, over most of the plot, so without colour no reader can
   tell the two lines apart between their letters. [shared] measures it on the
   level lines at the chart's resolution, 70 cells of 2 dot columns by 12 rows
   of 4 dots: a change that keeps a third line apart fails this test and reopens
   the cap. *)
let shared store a b =
  let level r =
    let h =
      History.read ~series:[ "train/loss" ]
        (Option.get (Store.run store (id_of r)))
    in
    let s = History.series h "train/loss" in
    Curve.level
      (Curve.reduce
         { lo = 2000.; hi = 20000.; dots = 140 }
         ~x:(fun i -> Float.of_int (Metric.step s i))
         s)
  in
  let la = level a and lb = level b in
  let all = Array.append la lb in
  let lo = Array.fold_left (fun m (_, y) -> Float.min m y) Float.infinity all in
  let hi =
    Array.fold_left (fun m (_, y) -> Float.max m y) Float.neg_infinity all
  in
  let row y =
    Float.to_int (Float.of_int 12 *. (y -. lo) /. (hi -. lo +. 1e-12))
  in
  let cell x = Float.to_int ((x -. 2000.) /. 18000. *. 70.) in
  let cells l =
    List.sort_uniq compare
      (Array.to_list (Array.map (fun (x, y) -> (cell x, row y)) l))
  in
  let ca = cells la and cb = cells lb in
  let common = List.filter (fun c -> List.mem c cb) ca in
  Float.of_int (List.length common) /. Float.of_int (max 1 (List.length ca))

let demonstration =
  test "marks-demonstration" (fun () ->
      with_store Fixture.runs @@ fun store ->
      let history r =
        History.read ~series:[ "train/loss" ]
          (Option.get (Store.run store (id_of r)))
      in
      let focus = history Fixture.tidy_otter in
      let others =
        [ Fixture.amber_fox; Fixture.brisk_wren; Fixture.calm_lynx ]
      in
      let chart n =
        let marked =
          List.filteri (fun k _ -> k < n) others
          |> List.mapi (fun k r -> (Char.chr (Char.code 'a' + k), history r))
        in
        Matrix_grid.to_text
          (Munin_tui.Chart.draw ~width:78 ~height:14 ~lo:2000. ~hi:20000.
             ~key:"train/loss" focus marked)
      in
      let charts = List.map chart [ 1; 2; 3 ] in
      let has c text = String.contains text c in
      equal (list bool) [ true; true ]
        [ has 'a' (List.nth charts 1); has 'b' (List.nth charts 1) ];
      less (float 1e-9) ~than:0.5
        (shared store Fixture.amber_fox Fixture.brisk_wren);
      greater (float 1e-9) ~than:0.5
        (shared store Fixture.amber_fox Fixture.calm_lynx);
      golden "marks-demonstration"
        (String.concat "\n"
           (List.mapi
              (fun k text ->
                Printf.sprintf
                  "train/loss, steps 2,000 to 20,000, tidy-otter against %d \
                   marked run%s\n\
                   %s"
                  (k + 1)
                  (if k = 0 then "" else "s")
                  text)
              charts)))

(* Laws of the charts *)

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

let step_x s i = Float.of_int (Metric.step s i)

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

let gen_samples ?(decrease = true) ?(value = gen_value) () =
  let open Gen in
  let* deltas =
    list ~size:(int_range 0 400)
      (frequency
         ([ (12, int_range 1 30); (3, constant 0) ]
         @ if decrease then [ (1, int_range (-200) (-1)) ] else []))
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

let pp_samples ppf samples =
  Format.pp_print_list ~pp_sep:Format.pp_print_space
    (fun ppf (st, v) -> Format.fprintf ppf "%d:%h" st v)
    ppf samples

let pp_plot ppf (cells, lo, hi) =
  Format.fprintf ppf "%d cells over [%g, %g]" cells lo hi

let finite_extent samples =
  List.fold_left
    (fun (lo, hi) (_, v) ->
      if Float.is_finite v then (Float.min lo v, Float.max hi v) else (lo, hi))
    (Float.infinity, Float.neg_infinity)
    samples

(* [draw ~cells ~lo ~hi ~ylo ~yhi points] is the text of a solid braille line
   through [points] in a plot of [cells] by 6 cells. *)
let draw ~cells ~lo ~hi ~ylo ~yhi points =
  let open Matrix_charts in
  let fig =
    layer
      [
        line ~dots:`Braille ~x:fst ~y:snd points;
        axis ~show:false "x";
        axis ~show:false "y";
      ]
    |> scale "x" (Scale.linear ~domain:(lo, hi) ())
    |> scale "y" (Scale.linear ~domain:(ylo, yhi) ())
  in
  let l = layout ~width:cells ~height:6 fig in
  equal int cells (Layout.plot l).width;
  let g = Matrix_grid.create ~width:cells ~height:6 () in
  draw l g ~x:0 ~y:0;
  Matrix_grid.to_text g

(* Every sample in log order, broken where a value is not finite, which matrix
   drops, and where the step decreases. *)
let every_sample samples =
  let _, rows =
    List.fold_left
      (fun (prev, acc) (step, v) ->
        let x = Float.of_int step in
        let acc = if x < prev then (Float.nan, Float.nan) :: acc else acc in
        (x, (x, v) :: acc))
      (Float.neg_infinity, []) samples
  in
  Array.of_list (List.rev rows)

let gen_plot samples =
  let open Gen in
  let last = List.fold_left (fun m (s, _) -> max m s) 0 samples in
  let* cells = int_range 1 160 in
  let* window = bool in
  let* lo = if window then int_range 0 (max 0 (last - 1)) else constant 0 in
  let+ hi =
    if window then int_range (lo + 1) (last + 1) else constant (last + 1)
  in
  (cells, Float.of_int lo, Float.of_int hi)

let m4 =
  prop "a reduced line inks the dots of every sample's line"
    Gen.(
      with_pp
        (fun ppf (s, p) ->
          Format.fprintf ppf "@[%a@ %a@]" pp_samples s pp_plot p)
        (let* samples = gen_samples () in
         let+ plot = gen_plot samples in
         (samples, plot)))
    (fun (samples, (cells, lo, hi)) ->
      let ylo, yhi = finite_extent samples in
      assume (ylo < yhi);
      let s = series_of samples in
      let n = List.length samples in
      cover "a dot column holds one sample" (n > 0 && n < cells);
      cover "a dot column holds many samples" (n > 8 * cells);
      cover "a window cuts the samples"
        (List.exists
           (fun (st, _) -> Float.of_int st < lo || Float.of_int st >= hi)
           samples);
      cover "steps decrease"
        (List.exists (fun ((a, _), (b, _)) -> b < a) (pairs samples));
      let c = Curve.reduce { lo; hi; dots = 2 * cells } ~x:(step_x s) s in
      equal string
        (draw ~cells ~lo ~hi ~ylo ~yhi (every_sample samples))
        (draw ~cells ~lo ~hi ~ylo ~yhi (Curve.points c)))

let column ~lo ~hi ~dots x =
  min (dots - 1)
    (Float.to_int ((x -. lo) *. (1. /. (hi -. lo)) *. Float.of_int dots))

(* [half_mean l] is half the mean of [l], each value divided by the count and
   by 2 first, so that no partial sum's rounding passes the largest float. *)
let half_mean l =
  let n = Float.of_int (List.length l) in
  List.fold_left (fun a v -> a +. (v /. n /. 2.)) 0. l

(* A quarter of the smoothed value at a distinct step [x_i], from the formula of
   the Charts section: samples at one step count as their mean. Quartered, so
   that no difference of values, nor their weighted sum, passes the largest
   float. *)
let quarter_smoothed ~tau samples x =
  let groups =
    List.fold_left
      (fun acc (st, v) ->
        if not (Float.is_finite v) then acc
        else
          match acc with
          | (st', vs) :: rest when st' = st -> (st, v :: vs) :: rest
          | _ -> (st, [ v ]) :: acc)
      [] samples
    |> List.rev
    |> List.map (fun (st, vs) -> (Float.of_int st, half_mean vs /. 2.))
  in
  let xs = Array.of_list (List.map fst groups) in
  let ys = Array.of_list (List.map snd groups) in
  let m = Array.length xs in
  let x0 = if m > 1 then xs.(0) -. (xs.(1) -. xs.(0)) else xs.(0) -. 1. in
  let num = ref 0. and den = ref 0. in
  Array.iteri
    (fun k xk ->
      if xk <= x then begin
        let prev = if k = 0 then x0 else xs.(k - 1) in
        let w =
          (1. -. exp (-.(xk -. prev) /. tau)) *. exp (-.(x -. xk) /. tau)
        in
        num := !num +. (w *. (ys.(k) -. ys.(0)));
        den := !den +. w
      end)
    xs;
  ys.(0) +. (!num /. !den)

(* A key's finite values whose sum, at some step, passes the largest float. *)
let sum_passes vs =
  not (Float.is_finite (List.fold_left ( +. ) 0. vs))

let ema =
  prop "a smoothed point is the debiased mean of the samples before it"
    Gen.(
      with_pp
        (fun ppf (s, tau, cells) ->
          Format.fprintf ppf "@[%a@ tau %g, %d cells@]" pp_samples s tau cells)
        (let* samples = gen_samples ~decrease:false ~value:gen_big_value () in
         let* tau = float_range 0.5 500. in
         let+ cells = int_range 1 160 in
         (samples, tau, cells)))
    (fun (samples, tau, cells) ->
      assume (List.exists (fun (_, v) -> Float.is_finite v) samples);
      let s = series_of samples in
      let last = List.fold_left (fun m (st, _) -> max m st) 0 samples in
      cover "samples share a step"
        (List.exists (fun ((a, _), (b, _)) -> a = b) (pairs samples));
      cover "the first two samples share a step"
        (match samples with (a, _) :: (b, _) :: _ -> a = b | _ -> false);
      cover "the samples at a step sum past the largest float"
        (List.exists
           (fun (st, _) ->
             sum_passes
               (List.filter_map
                  (fun (st', v) ->
                    if st' = st && Float.is_finite v then Some v else None)
                  samples))
           samples);
      let finite = List.filter (fun (_, v) -> Float.is_finite v) samples in
      cover "a step's mean differs from the first's past the largest float"
        (match finite with
        | [] -> false
        | (first, _) :: _ ->
            let y1 =
              half_mean
                (List.filter_map
                   (fun (st, v) -> if st = first then Some v else None)
                   finite)
            in
            List.exists
              (fun (_, v) -> Float.abs ((v /. 2.) -. y1) > Float.max_float /. 2.)
              finite);
      let scale =
        List.fold_left (fun m (_, v) -> Float.max m (Float.abs v)) 0. finite
      in
      let plot =
        { Curve.lo = 0.; hi = Float.of_int (last + 1); dots = 2 * cells }
      in
      let c = Curve.reduce ~tau plot ~x:(step_x s) s in
      (* Every column with a finite sample holds a smoothed point. *)
      let columns l =
        List.sort_uniq compare
          (List.map (column ~lo:plot.lo ~hi:plot.hi ~dots:plot.dots) l)
      in
      equal (list int)
        (columns (List.map (fun (st, _) -> Float.of_int st) finite))
        (columns
           (List.filter
              (fun x -> not (Float.is_nan x))
              (Array.to_list (Array.map fst (Curve.points c)))));
      Array.iter
        (fun (x, y) ->
          if not (Float.is_nan x) then
            equal
              (float_rel ~rel:1e-9 ~abs:(Float.max 1e-12 (1e-12 *. scale)))
              (quarter_smoothed ~tau samples x)
              (y /. 4.))
        (Curve.points c))

let events =
  prop "each dot column with a sample that is not finite is marked"
    Gen.(
      with_pp
        (fun ppf (s, p) ->
          Format.fprintf ppf "@[%a@ %a@]" pp_samples s pp_plot p)
        (let* samples = gen_samples () in
         let+ plot = gen_plot samples in
         (samples, plot)))
    (fun (samples, (cells, lo, hi)) ->
      let s = series_of samples in
      let dots = 2 * cells in
      let c = Curve.reduce { lo; hi; dots } ~x:(step_x s) s in
      let expected = Hashtbl.create 16 in
      List.iter
        (fun (st, v) ->
          let x = Float.of_int st in
          if (not (Float.is_finite v)) && x >= lo && x <= hi then begin
            let k = column ~lo ~hi ~dots x in
            let e =
              if Float.is_nan v then `Nan
              else if v > 0. then `Pos_inf
              else `Neg_inf
            in
            match Hashtbl.find_opt expected k with
            | Some `Nan -> ()
            | _ -> Hashtbl.replace expected k e
          end)
        samples;
      cover "a column holds NaN and an infinity"
        (Hashtbl.length expected > 0
        && List.exists (fun (_, v) -> Float.is_nan v) samples
        && List.exists
             (fun (_, v) -> Float.is_finite v = false && not (Float.is_nan v))
             samples);
      let name = function
        | `Nan -> "NaN"
        | `Pos_inf -> "+inf"
        | `Neg_inf -> "-inf"
      in
      let sorted l =
        List.sort compare (List.map (fun (k, e) -> (k, name e)) l)
      in
      equal
        (list (pair int string))
        (sorted (List.of_seq (Hashtbl.to_seq expected)))
        (sorted (Array.to_list (Curve.events c))))

let level =
  prop "a level point is the mean of its column's finite values"
    Gen.(
      with_pp
        (fun ppf (s, p) ->
          Format.fprintf ppf "@[%a@ %a@]" pp_samples s pp_plot p)
        (let* samples = gen_samples ~value:gen_big_value () in
         let+ plot = gen_plot samples in
         (samples, plot)))
    (fun (samples, (cells, lo, hi)) ->
      let s = series_of samples in
      let dots = 2 * cells in
      let c = Curve.reduce { lo; hi; dots } ~x:(step_x s) s in
      let columns = Array.make dots [] in
      List.iter
        (fun (st, v) ->
          let x = Float.of_int st in
          if Float.is_finite v && x >= lo && x <= hi then
            let k = column ~lo ~hi ~dots x in
            columns.(k) <- v :: columns.(k))
        samples;
      let columns = List.filter (( <> ) []) (Array.to_list columns) in
      cover "a column's values sum past the largest float"
        (List.exists sum_passes columns);
      let level = Array.to_list (Curve.level c) in
      equal ~msg:"columns" int (List.length columns) (List.length level);
      (* Each column within the rounding of its own values. *)
      List.iteri
        (fun k (vs, (_, y)) ->
          let scale =
            List.fold_left (fun m v -> Float.max m (Float.abs v)) 0. vs
          in
          equal
            ~msg:(Printf.sprintf "column %d" k)
            (float_rel ~rel:1e-9 ~abs:(Float.max 1e-12 (1e-12 *. scale)))
            (half_mean vs) (y /. 2.))
        (List.combine columns level))

let extend =
  prop "extending a reduction equals reducing the longer series"
    Gen.(
      with_pp
        (fun ppf (s, cut, p) ->
          Format.fprintf ppf "@[%a@ cut at %d,@ %a@]" pp_samples s cut pp_plot p)
        (let* samples = gen_samples () in
         let* cut = int_range 0 (List.length samples) in
         let+ plot = gen_plot samples in
         (samples, cut, plot)))
    (fun (samples, cut, (cells, lo, hi)) ->
      let store = Lazy.force laws in
      let first = List.filteri (fun k _ -> k < cut) samples in
      let rest = List.filteri (fun k _ -> k >= cut) samples in
      let id, log = write_log store (List.map sample_record first) in
      let h = History.read ~series:[ "m" ] (Option.get (Store.run store id)) in
      Out_channel.with_open_gen [ Open_append; Open_binary ] 0o644 log
        (fun oc ->
          List.iter (fun r -> output_string oc (sample_record r ^ "\n")) rest);
      let h' = History.follow h in
      let plot = { Curve.lo; hi; dots = 2 * cells } in
      let s0 = History.series h "m" and s1 = History.series h' "m" in
      let whole = Curve.reduce plot ~x:(step_x s1) s1 in
      let grown =
        Curve.extend (Curve.reduce plot ~x:(step_x s0) s0) ~x:(step_x s1) s1
      in
      let pts = array (pair float_exact float_exact) in
      equal pts (Curve.points whole) (Curve.points grown);
      equal pts (Curve.level whole) (Curve.level grown))

(* The x axis *)

let values_record step kvs =
  Printf.sprintf {|{"type":"metrics","time":2.0,"step":%d,"values":{%s}}|} step
    (String.concat ","
       (List.map
          (fun (k, v) -> Printf.sprintf {|"%s":%s|} k (Fixture.num v))
          kvs))

(* Records of "m" and of its step metric "e", each at a step that rises or
   stays, holding either or both. *)
let gen_paired =
  let open Gen in
  list ~size:(int_range 1 60)
    (let* d = frequency [ (3, int_range 1 5); (1, constant 0) ] in
     let* which = int_range 0 2 in
     let+ v = float_range 0. 100. in
     (d, which, v))

let pairing =
  prop "a sample's x on the step metric is its latest sample at that step"
    gen_paired (fun records ->
      let store = Lazy.force laws in
      let step = ref 0 in
      let lines =
        List.map
          (fun (d, which, v) ->
            step := !step + d;
            let kvs =
              match which with
              | 0 -> [ ("m", v) ]
              | 1 -> [ ("e", v +. 1000.) ]
              | _ -> [ ("m", v); ("e", v +. 1000.) ]
            in
            (!step, kvs, values_record !step kvs))
          records
      in
      let id, _ =
        write_log store
          ~declare:[ metric_record "e"; metric_record ~step_metric:"e" "m" ]
          (List.map (fun (_, _, l) -> l) lines)
      in
      let h = History.read (Option.get (Store.run store id)) in
      let s = History.series h "m" in
      (* The latest "e" at each step, from the records in log order. *)
      let e = Hashtbl.create 16 in
      List.iter
        (fun (st, kvs, _) ->
          Option.iter (Hashtbl.replace e st) (List.assoc_opt "e" kvs))
        lines;
      let expected =
        List.init (Metric.length s) (fun i ->
            Option.value
              (Hashtbl.find_opt e (Metric.step s i))
              ~default:Float.nan)
      in
      cover "a sample has no pair" (List.exists Float.is_nan expected);
      cover "a sample pairs"
        (List.exists (fun x -> not (Float.is_nan x)) expected);
      equal (list float_exact) expected
        (List.init (Metric.length s) (Munin_tui.Chart.paired h "m")))

(* The least of 1, 2 or 5 times a power of ten at or above [1.1 *. x]. *)
let bound x =
  if not (x > 0.) then 1.
  else
    let c =
      List.concat_map
        (fun k ->
          List.map (fun m -> m *. (10. ** Float.of_int k)) [ 1.; 2.; 5. ])
        (List.init 30 (fun k -> k - 10))
    in
    List.fold_left
      (fun b c -> if c >= 1.1 *. x && c < b then c else b)
      Float.infinity c

let domain =
  prop
    "the x domain ends at the declared steps, else at a round bound past every \
     x"
    Gen.(
      let* steps = option (int_range 0 100_000) in
      let+ last = int_range 0 200_000 in
      (steps, last))
    (fun (steps, last) ->
      let store = Lazy.force laws in
      let id, _ =
        write_log store ?steps
          [ sample_record (0, 1.); sample_record (last, 2.) ]
      in
      let h = History.read (Option.get (Store.run store id)) in
      cover "steps declared"
        (match steps with Some n -> n > 0 && last <= n | None -> false);
      cover "a sample past the declared steps"
        (match steps with Some n -> n > 0 && last > n | None -> false);
      cover "no steps" (steps = None);
      let expected =
        match steps with
        | Some n when n > 0 && last <= n -> (0., Float.of_int n)
        | _ -> (0., bound (Float.of_int last))
      in
      equal
        (pair float_exact float_exact)
        expected
        (Munin_tui.Chart.domain h "m"))

(* Log y, and its refusal where a value is not above 0: the Zoom of "m". *)
let log_y =
  cases "log y where every value is above 0"
    ~id:(fun (name, _, _) -> name)
    [
      ("positive", [ 1.; 10.; 100. ], "log y");
      ("a zero", [ 0.; 10.; 100. ], "linear: values <= 0");
      ("a negative", [ -1.; 10.; 100. ], "linear: values <= 0");
    ]
    (fun (_, values, note) ->
      with_store ~name:"log" [] @@ fun store ->
      let id, _ =
        write_log store (List.mapi (fun i v -> sample_record (i, v)) values)
      in
      let screen =
        show ~cols:120 ~rows:40 store (Munin_tui.Run (Some id))
          [ Key Enter; char 'L' ]
      in
      equal bool true (T.Screen.find screen note <> []))

(* Runs: the filter and the best value of each experiment *)

(* The names of Runs's rows, and whether a row holds a star: the rows from the
   header to the first blank row, those that name a run. *)
let shown screen =
  let rec go y acc =
    if y >= T.Screen.rows screen then List.rev acc
    else
      match
        List.filter (( <> ) "")
          (String.split_on_char ' ' (T.Screen.row screen y))
      with
      | [] -> List.rev acc
      | fields -> (
          match
            List.find_opt
              (fun f -> List.exists (fun r -> r.Fixture.name = f) Fixture.runs)
              fields
          with
          | Some name ->
              go (y + 1)
                (( name,
                   List.exists (fun f -> String.ends_with ~suffix:"*" f) fields
                 )
                :: acc)
          | None -> go (y + 1) acc)
  in
  go 3 []

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
        let screen =
          show ~cols:160 ~rows:48 store Munin_tui.Runs (to_val_loss @ keys)
        in
        List.filter_map
          (fun (n, star) -> if star then Some n else None)
          (shown screen)
      in
      equal (list string) [ "amber-fox"; "mlp-wide" ] (starred []);
      equal (list string) [ "calm-lynx" ]
        (starred [ char '/'; Text "warmup=2000"; Key Enter ]))

let metric_filter =
  test "the filter on Run keeps the metrics that match each word" (fun () ->
      with_store Fixture.runs @@ fun store ->
      let screen =
        show ~cols:80 ~rows:24 store
          (Munin_tui.Run (Some (id_of Fixture.tidy_otter)))
          [ char '/'; Text "VAL acc"; Key Enter ]
      in
      let found s = T.Screen.find screen s <> [] in
      equal (list bool) [ true; false; false ]
        [ found "val/acc"; found "val/loss"; found "train/acc" ])

(* Zoom counts a metric's samples that are not finite together, NaN and
   infinities alike. *)
let non_finite_count =
  test "Zoom counts NaN and infinities together" (fun () ->
      with_store ~name:"counts" [] @@ fun store ->
      let values =
        [
          1.;
          Float.nan;
          Float.infinity;
          Float.nan;
          2.;
          Float.neg_infinity;
          Float.nan;
        ]
      in
      let id, _ =
        write_log store (List.mapi (fun i v -> sample_record (i, v)) values)
      in
      let screen =
        show ~cols:160 ~rows:48 store (Munin_tui.Run (Some id)) [ Key Enter ]
      in
      let row =
        match T.Screen.find screen "not finite" with
        | [ (row, _) ] ->
            List.filter (( <> ) "")
              (String.split_on_char ' ' (T.Screen.row screen (row + 1)))
        | found -> failf "not finite found %d times" (List.length found)
      in
      (* The focus's row ends with its samples, non-finite ones and orphans. *)
      let n = List.length row in
      equal (list string) [ "7"; "5"; "0" ]
        [ List.nth row (n - 3); List.nth row (n - 2); List.nth row (n - 1) ])

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

(* Runs that log as programs do: the focus every 10 steps with the value [step /
   1000], the marked run [marked] samples of [0]. The difference rows pair the
   two at the same step, so their latest value is the focus's at the latest step
   both logged, and its verdict and step are that step's. *)
let paired =
  cases "a difference row compares the two runs at the same step"
    ~id:(fun (name, _, _, _) -> name)
    [
      ( "the marked run logs every 100 steps",
        List.init 21 (fun k -> k * 100),
        "+0.9000",
        "worse at 900" );
      ( "the marked run ends in the focus's last column",
        List.init 99 (fun k -> k * 10) @ [ 989 ],
        "+0.9800",
        "worse at 980" );
    ]
    (fun (_, marked, value, verdict) ->
      with_store ~name:"paired" [] @@ fun store ->
      ignore
        (write_log store ~goal:"minimize" ~started:1
           (List.map (fun st -> sample_record (st, 0.)) marked));
      ignore
        (write_log store ~goal:"minimize" ~started:2
           (List.init 100 (fun k ->
                sample_record (k * 10, Float.of_int (k * 10) /. 1000.))));
      (* The focus, the newest, is the first row: mark the second, come back and
         open the first. *)
      let screen =
        show ~cols:120 ~rows:40 store Munin_tui.Runs
          [ char 'j'; Text " "; char 'k'; Key Enter ]
      in
      let found s = T.Screen.find screen s <> [] in
      equal (list bool) [ true; true ] [ found value; found verdict ])

(* The focus logs the largest float at step 0 and the marked run its negation,
   a difference past the largest float; later steps differ by [step / 1000].
   The row draws with its scale, its latest value and its verdict. *)
let paired_past =
  test "a difference row draws where a difference passes the largest float"
    (fun () ->
      with_store ~name:"paired-past" [] @@ fun store ->
      let values first =
        List.init 100 (fun k ->
            sample_record
              (k * 10, if k = 0 then first else Float.of_int (k * 10) /. 1000.))
      in
      ignore
        (write_log store ~goal:"minimize" ~started:1
           (List.init 100 (fun k ->
                sample_record
                  (k * 10, if k = 0 then -.Float.max_float else 0.))));
      ignore
        (write_log store ~goal:"minimize" ~started:2 (values Float.max_float));
      let screen =
        show ~cols:120 ~rows:40 store Munin_tui.Runs
          [ char 'j'; Text " "; char 'k'; Key Enter ]
      in
      let found s = T.Screen.find screen s <> [] in
      let rows s = List.map fst (T.Screen.find screen s) in
      equal (list bool) [ true; true ]
        [ found "+0.9900"; found "worse at 990" ];
      (* The scale's row of the difference row holds the line's dots. *)
      match rows "±1.798e+308" with
      | [ r ] ->
          equal (list int) [ r ]
            (List.sort_uniq compare (List.filter (( = ) r) (rows "⠉⠉⠉")))
      | l -> failf "the scale found %d times" (List.length l))

(* The focus logs one value at every step and the marked run another: every
   difference is the same, and the row draws it at its top. 1e308 against
   -1e308 differs by more than the largest float, and 1e-323 against 5e-324 by
   the least subnormal. *)
let paired_wide =
  cases "a difference row draws a constant difference at its top"
    ~id:(fun (f, m) -> Printf.sprintf "%g against %g" f m)
    [ (1e308, -1e308); (1e-323, 5e-324) ]
    (fun (focus, marked) ->
      with_store ~name:"paired-wide" [] @@ fun store ->
      let flat v = List.init 100 (fun k -> sample_record (k * 10, v)) in
      ignore (write_log store ~goal:"minimize" ~started:1 (flat marked));
      ignore (write_log store ~goal:"minimize" ~started:2 (flat focus));
      let screen =
        show ~cols:120 ~rows:40 store Munin_tui.Runs
          [ char 'j'; Text " "; char 'k'; Key Enter ]
      in
      (* The row's first line holds its top dots. *)
      match T.Screen.find screen "focus minus each marked run" with
      | [ (r, _) ] ->
          equal (list int) [ r + 1 ]
            (List.sort_uniq compare
               (List.filter
                  (( = ) (r + 1))
                  (List.map fst (T.Screen.find screen "⠉⠉⠉"))))
      | l -> failf "the caption found %d times" (List.length l))

(* The focus logs 1e308 at every step and the marked run -1e308 over the first
   half and -1e307 over the second: the mean differences 2e308 and 1.1e308 are
   drawn in proportion under a scale of 2e308, which rounds to infinity. *)
let paired_split =
  test "a difference row past the largest float keeps its proportions"
    (fun () ->
      with_store ~name:"paired-split" [] @@ fun store ->
      let records f = List.init 100 (fun k -> sample_record (k * 10, f k)) in
      ignore
        (write_log store ~goal:"minimize" ~started:1
           (records (fun k -> if k < 50 then -1e308 else -1e307)));
      ignore
        (write_log store ~goal:"minimize" ~started:2
           (records (fun _ -> 1e308)));
      let screen =
        show ~cols:120 ~rows:40 store Munin_tui.Runs
          [ char 'j'; Text " "; char 'k'; Key Enter ]
      in
      let found s = T.Screen.find screen s <> [] in
      equal (list bool) [ true; true ]
        [ found "±inf"; found "+1.100e+308" ])

(* A metric from 0 to 4.15526391128405e-310, subnormal, on 22 rows: the line
   runs between its extremes, and the y axis is labelled at multiples of a step
   of 1e-310, as on any other domain. *)
let subnormal_ticks =
  test "a chart of subnormal values ticks at round multiples" (fun () ->
      with_store ~name:"subnormal-ticks" [] @@ fun store ->
      let hi = 4.15526391128405e-310 in
      let id, _ =
        write_log store
          (List.init 9 (fun step ->
               sample_record (step, if step mod 2 = 0 then 0. else hi)))
      in
      let h = History.read (Option.get (Store.run store id)) in
      let g =
        Munin_tui.Chart.draw ~width:60 ~height:23 ~lo:0. ~hi:10. ~key:"m" h []
      in
      at_least int ~than:8
        (List.length (List.filter (( < ) 0) (dots_per_line g)));
      equal (list string)
        [ "4.155e-310"; "3e-310"; "2e-310"; "1e-310"; "0" ]
        (tick_labels g))

(* A step metric that reaches 1.08e308: the x domain is bounded within the
   finite floats and the chart draws. *)
let step_metric_extreme =
  test "a chart on a step metric near the largest float draws" (fun () ->
      with_store ~name:"step-metric-extreme" [] @@ fun store ->
      let record k =
        let v = Float.of_int k in
        values_record k [ ("m", v); ("e", v *. 1.2e307) ]
      in
      ignore
        (write_log store
           ~declare:[ metric_record "e"; metric_record ~step_metric:"e" "m" ]
           (List.init 10 record));
      let screen =
        show ~cols:120 ~rows:40 store Munin_tui.Runs
          [ Key Enter; char 'j'; char 'x' ]
      in
      let found s = T.Screen.find screen s <> [] in
      (* The x axis, the row under the plot's corner, labels e's values. *)
      let x_labels =
        match T.Screen.find screen "└┬" with
        | [ (r, _) ] -> T.Screen.row screen (r + 1)
        | _ -> ""
      in
      let big w =
        String.ends_with ~suffix:"e+307" w || String.ends_with ~suffix:"e+308" w
      in
      equal (list bool) [ false; true ]
        [
          found "cannot read";
          List.exists big (String.split_on_char ' ' x_labels);
        ])

(* The marks change and the poll that prepares the new chart has not returned:
   no difference row draws the run a letter named before. The reads are held on
   a queue of the test's until it releases them. *)
let remarked =
  test "a difference row never draws a run no longer marked" (fun () ->
      with_store ~name:"remarked" [] @@ fun store ->
      let flat v = List.init 100 (fun k -> sample_record (k * 10, v)) in
      ignore (write_log store ~goal:"minimize" ~started:1 (flat 1.));
      ignore (write_log store ~goal:"minimize" ~started:2 (flat 0.));
      ignore
        (write_log store ~goal:"minimize" ~started:3
           (List.init 100 (fun k ->
                sample_record (k * 10, Float.of_int (k * 10) /. 1000.))));
      let queue = Queue.create () and holding = ref true in
      let session = ref None in
      let process_perform job =
        match (!holding, !session) with
        | false, Some s -> T.perform s job
        | _ -> Queue.push job queue
      in
      let release s =
        holding := false;
        Queue.iter (T.perform s) queue;
        Queue.clear queue
      in
      let term = T.Terminal.create ~cols:120 ~rows:40 () in
      (elapsed := fun () -> 0);
      let app = Munin_tui.app ~clock store Munin_tui.Runs in
      Mosaic_test.run ~process_perform term app @@ fun s ->
      session := Some s;
      (elapsed := fun () -> T.now s);
      release s;
      T.advance s second;
      let keys ks =
        List.iter
          (fun k ->
            send term s k;
            T.advance s second)
          ks
      in
      (* Rows: the focus, the run of 0s, the run of 1s. Mark the run of 0s as a
         and open the focus. *)
      keys [ char 'j'; Text " "; char 'k'; Key Enter ];
      let before = T.Terminal.screen term in
      (* From Run: focus the run of 0s and unmark it, focus the run of 1s and
         mark it, a again, and come back to the focus, while no read returns. *)
      holding := true;
      keys [ char ']'; Text " "; char ']'; Text " "; char '['; char '[' ];
      let held = T.Terminal.screen term in
      release s;
      T.advance s second;
      let after = T.Terminal.screen term in
      let found screen v = T.Screen.find screen v <> [] in
      (* At step 990 the focus is 0.99: 0.99 above the run of 0s, 0.01 below the
         run of 1s. *)
      equal (list bool) [ true; false; true ]
        [ found before "+0.9900"; found held "+0.9900"; found after "-0.01000" ])

(* A metric whose every sample a resume orphans has no statistics, and Details
   lists the run all the same. *)
let all_orphans =
  test "Details lists a run whose metric is all orphans" (fun () ->
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
        show ~cols:80 ~rows:24 store (Munin_tui.Run (Some id)) [ char 'i' ]
      in
      equal bool true (T.Screen.find screen "metrics" <> []))

(* What the dashboard holds *)

(* A run that a sample arrived from stays live through screens, cursor moves and
   the runs opened in between: the dashboard holds what it read of a run while a
   screen of its stack needs it, and its arrivals with it. Each case logs a new
   step of tidy-otter while the dashboard watches, then takes the keys, and
   reads tidy-otter's word back on Runs. *)
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
  prop ~count:30 "a live run stays live through screens and cursor moves"
    key_gen (fun keys ->
      let store = Lazy.force store in
      step := !step + 50;
      let script _ = feed store Fixture.tidy_otter ~step:!step in
      cover "the keys open a run" (List.mem (Key Enter) keys);
      let screen =
        show ~script ~cols:120 ~rows:40 store Munin_tui.Runs
          (keys @ times 4 (Key Escape))
      in
      let word =
        match T.Screen.find screen "tidy-otter" with
        | (row, _) :: _ -> (
            match
              List.filter (( <> ) "")
                (String.split_on_char ' ' (T.Screen.row screen row))
            with
            | ("›" | "a" | "b") :: ("●" | "○" | "◌" | "✓" | "✗" | "■") :: w :: _
            | ("●" | "○" | "◌" | "✓" | "✗" | "■") :: w :: _
            | w :: _ ->
                w
            | [] -> "")
        | [] -> "absent"
      in
      equal string "live" word)

(* The cursor leaves an ended run and comes back: the pane's chart was kept, so
   the dashboard reads nothing of the run, and the log moved away meanwhile is
   never missed. Any fold of it would fail and name the run. *)
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
      (* The pane names amber-fox and draws its chart, a y axis labelled. *)
      equal bool true (T.Screen.find screen "amber-fox  ✓ finished" <> []);
      equal bool true (T.Screen.find screen "┤" <> []))

(* A run logs its last samples and ends after its pane's chart was prepared, and
   the cursor leaves it before a poll reads them: coming back draws them, since
   a chart prepared before the end is not kept. *)
let ended_away =
  test "a cursor back on a run that ended away draws its last samples"
    (fun () ->
      with_store Fixture.runs @@ fun store ->
      let s =
        Session.start ~store ~experiment:"late" ~name:"late-end" ~steps:100
          ~provenance:(Provenance.detect ~cwd:home ())
          ()
      in
      let loss = Session.metric s ~goal:`Minimize "train/loss" in
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
      (* The first 50 samples, all 1, span a y axis from 0.95 to 1.05; the last
         50, all 9, stretch it to its top label, the greatest value drawn. *)
      equal bool true (T.Screen.find screen "9.000 ┤" <> []);
      equal bool true (T.Screen.find screen "late-end  ✓ finished" <> []))

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
      [ `Running; `Silent; `Crashed; `Finished; `Failed "e"; `Killed ]
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
        (Munin_tui.Watch.observe w ~now ~samples, now, samples))
  in
  stateful ~count:300 ~steps:60 "the state words follow the arrivals"
    [
      command "start"
        (Gen.int_range 0 3 @-> makes w)
        (fun samples -> { now = 0.; samples; times = [] })
        (fun samples -> (Munin_tui.Watch.start ~now:0. ~samples, 0., samples));
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
        (fun st (w, now, _) -> Munin_tui.Watch.word st (Some w) ~now);
    ]

(* The screen stack *)

type screen_name = Runs_s | Run_s | Zoom_s | Details_s

type stack = {
  screens : screen_name list; (* the top first *)
  help : bool;
  typing : bool;
}

let model_key st k =
  let top = List.hd st.screens in
  let push s = { st with screens = s :: st.screens } in
  match k with
  | _ when st.typing -> (
      match k with
      | Key Enter | Key Escape -> { st with typing = false }
      | _ -> st)
  | _ when st.help -> (
      match k with Text "?" | Key Escape -> { st with help = false } | _ -> st)
  | Text "?" -> { st with help = true }
  | Key Escape -> (
      match st.screens with
      | _ :: (_ :: _ as rest) -> { st with screens = rest }
      | _ -> st)
  | Key Enter -> (
      match top with Runs_s -> push Run_s | Run_s -> push Zoom_s | _ -> st)
  | Text "i" -> ( match top with Details_s -> st | _ -> push Details_s)
  | Text "/" -> (
      match top with Runs_s | Run_s -> { st with typing = true } | _ -> st)
  | _ -> st

(* The screen shown, from its key line, which names each screen's keys: Help
   over it by its first section; typing when the key line holds the filter,
   which on Runs says how Enter and Esc end it. *)
let observed screen =
  let rows = T.Screen.rows screen in
  let last = String.trim (T.Screen.row screen (rows - 1)) in
  let has s = T.Screen.find screen s <> [] in
  let starts p s =
    String.length s >= String.length p && String.sub s 0 (String.length p) = p
  in
  let holds sub s =
    let n = String.length sub in
    let rec go i =
      i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
    in
    go 0
  in
  if has "Everywhere" then "help"
  else if starts "/" last then
    if holds "enter keep" last then "runs typing" else "run typing"
  else if holds "enter open" last then "runs"
  else if holds "enter zoom" last then "run"
  else if starts "esc run" last then "zoom"
  else if starts "↑↓ scroll" last then "details"
  else "unknown: " ^ last

let model_name st =
  if st.help then "help"
  else
    let top =
      match List.hd st.screens with
      | Runs_s -> "runs"
      | Run_s -> "run"
      | Zoom_s -> "zoom"
      | Details_s -> "details"
    in
    if st.typing then top ^ " typing" else top

let stack_test =
  let store =
    lazy (Fixture.make ~child (Filename.concat home "stack") Fixture.runs)
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
  stateful ~count:60 ~steps:12
    "Enter opens, Esc goes back, q is typed in the filter"
    [
      command "runs"
        (Gen.unit @-> makes dash)
        (fun () -> { screens = [ Runs_s ]; help = false; typing = false })
        (fun () -> []);
      command "run"
        (Gen.unit @-> makes dash)
        (fun () ->
          { screens = [ Run_s; Runs_s ]; help = false; typing = false })
        (fun () -> [ `Start_run ]);
      command "key"
        ~pre:(fun k st -> k <> Text "q" || st.typing)
        (key_gen @-> dash ^-> makes dash)
        (fun k st -> model_key st k)
        (fun k keys -> `Key k :: keys);
      command "screen"
        (dash ^-> returns string)
        model_name
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

let () =
  exit
    (run "munin.tui"
       [
         screens;
         states_test;
         group "charts"
           [
             m4;
             ema;
             events;
             level;
             extend;
             pairing;
             domain;
             log_y;
             orphans;
             flat_extreme;
             subnormal_ticks;
             wide_extreme;
             demonstration;
           ];
         group "runs" [ filter; stars; metric_filter ];
         group "run"
           [
             non_finite_count;
             all_orphans;
             short_of_steps;
             paired;
             paired_past;
             paired_wide;
             paired_split;
             step_metric_extreme;
             remarked;
           ];
         group "held" [ held; kept_pane; ended_away; watch_test ];
         stack_test;
       ])
