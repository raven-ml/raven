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

let rec remove path =
  match (Unix.lstat path).st_kind with
  | Unix.S_DIR ->
      Array.iter (fun e -> remove (Filename.concat path e)) (Sys.readdir path);
      Unix.rmdir path
  | _ -> Sys.remove path
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> ()

let child = Filename.concat (Sys.getcwd ()) "../support/child.exe"

(* [with_store ?name runs f] is [f store], [store] a store of [runs] at
   [~/name], the home being the test's own directory, and lets the writers
   that hold its runs go after. The root is joined with [/], which Windows
   also takes, so that a screen names it as [~/name] there too. *)
let with_store ?(name = "store") runs f =
  let home = temp_dir () in
  setenv "HOME" (Some home);
  let store = Fixture.make ~child (home ^ "/" ^ name) runs in
  Fun.protect ~finally:Fixture.release (fun () -> f store)

(* The directory of the stores that the cases of one law share, which the
   process removes as it ends, after the writers that hold their runs let
   go, since Windows removes no file a process holds open. *)
let shared =
  lazy
    (let dir = Filename.temp_dir "munin_tui" "" in
     at_exit (fun () ->
         Fixture.release ();
         remove dir);
     dir)

(* [shared_store name runs] is a store of [runs] in [shared]. *)
let shared_store name runs =
  Fixture.make ~child (Filename.concat (Lazy.force shared) name) runs

let id_of r = r.Fixture.id

(* A test whose expected screen shows a crashed run: on a system whose writers
   record no boot, such as Windows, the fixture holds no crashed run. *)
let needs_crashed () =
  if not (Fixture.boots ~child (Lazy.force shared)) then
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
  setenv "TZ" (Some "UTC");
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

let char c = Text (String.make 1 c)
let times n k = List.init n (fun _ -> k)

(* [before row sep] is [row] up to the first [sep] in it. *)
let before row sep =
  let n = String.length sep in
  let rec find i =
    if i + n > String.length row then None
    else if String.sub row i n = sep then Some (String.sub row 0 i)
    else find (i + 1)
  in
  find 0

(* [main_screen ~cols ~rows keys] is the main store's screen after [keys],
   tidy-otter having logged its step 12,400 while the dashboard watched, so
   that an arrival makes it live. *)
let main_screen ?(start = Munin_tui.Runs) ?profile ~cols ~rows keys =
  with_store Fixture.runs @@ fun store ->
  let script _ = feed store Fixture.tidy_otter ~step:12400 in
  show ~script ?profile ~cols ~rows store start keys

(* A screen's text, compared with its golden: the claim is the layout. *)
let layout_test ~golden:file ?start ?(crashed = true) ~cols ~rows keys name =
  test name (fun () ->
      if crashed then needs_crashed ();
      golden file (T.Screen.text (main_screen ?start ~cols ~rows keys)))

(* A screen's text and styles, compared with its golden: the claim is what
   the styles show. *)
let style_test ~golden:file ?profile ~cols ~rows keys name =
  test name (fun () ->
      needs_crashed ();
      golden file
        (T.Screen.styled (main_screen ?profile ~cols ~rows keys)))

(* The dashboard opens with its cursor on tidy-otter, the newest run, and its
   chart on train/loss. amber-fox is the second run and dusty-mole the fifth:
   overlaying both draws a run whose losses turn NaN against the cursor's. *)
let overlay_two =
  [ char 'j'; Text " "; char 'j'; char 'j'; char 'j'; Text " "; char 'g' ]

let screens =
  group "screens"
    [
      layout_test ~golden:"dashboard-80x24" ~cols:80 ~rows:24 []
        "below 100 columns a line names the run and one line of values \
         takes the hero's place";
      layout_test ~golden:"dashboard-100x30" ~cols:100 ~rows:30 []
        "at 100 columns and 30 lines both panes, the hero and two small \
         charts show";
      layout_test ~golden:"dashboard-120x40" ~cols:120 ~rows:40 []
        "at 120 columns the runs pane takes 34 columns and the run pane 28";
      layout_test ~golden:"dashboard-160x48" ~cols:160 ~rows:48 []
        "from 150 columns the panes take 42 and 34, the hero four values \
         and three small charts";
      style_test ~golden:"overlays-120x40" ~cols:120 ~rows:40 overlay_two
        "overlays draw in their colours beside the cursor's run";
      style_test ~golden:"overlays-120x40-no-colour" ~profile:no_colour
        ~cols:120 ~rows:40 overlay_two
        "without colour an overlay's line and legend end in its letter";
      layout_test ~golden:"strips-120x40" ~cols:120 ~rows:40
        (overlay_two @ [ Key Enter ])
        "at strips each metric is a strip with its value and its \
         difference to the first overlay";
      (* Zoom's step cursor starts on the latest val/loss sample, and each
         move left goes back one sample. *)
      style_test ~golden:"zoom-120x40" ~cols:120 ~rows:40
        (overlay_two @ [ Key Enter; Key Enter ] @ times 8 (Key Left))
        "zoom draws the raw extent faint behind the line, a step cursor and \
         a readout";
      layout_test ~golden:"info-120x40" ~cols:120 ~rows:40 [ char 'i' ]
        "i shows everything about the run over the centre";
      layout_test ~golden:"help-80x24" ~cols:80 ~rows:24 [ char '?' ]
        "? shows the keys";
      layout_test ~golden:"small-50x14" ~crashed:false ~cols:50 ~rows:14
        ~start:(Munin_tui.Run (Some (id_of Fixture.tidy_otter)))
        []
        "below 60 columns a line per live run asks for a larger terminal";
      test "a store without runs shows how a program starts one" (fun () ->
          with_store ~name:"empty" [] @@ fun store ->
          golden "empty-120x40"
            (T.Screen.text (show ~cols:120 ~rows:40 store Munin_tui.Runs [])));
    ]

(* A terminal of 16 colours draws the text a 24-bit one draws: colour
   repeats what the text says. *)
let sixteen_colours =
  test "a terminal of 16 colours shows the 24-bit screen's text" (fun () ->
      with_store Fixture.[ short_running; short_amber ] @@ fun store ->
      let text profile =
        T.Screen.text (show ?profile ~cols:120 ~rows:40 store Munin_tui.Runs [])
      in
      equal string (text None) (text (Some sixteen)))

(* Without colour the cursor's row of the runs pane and the cursor's strip,
   two rows from its mark to the screen's last column but its margin, are
   reverse video. *)
let reverse_video =
  test "without colour the cursor's row and strip are reverse video" (fun () ->
      with_store Fixture.[ short_running; short_amber ] @@ fun store ->
      let sc =
        show ~profile:no_colour ~cols:120 ~rows:40 store Munin_tui.Runs
          [ Key Enter ]
      in
      let plain row cols =
        List.filter
          (fun col ->
            let _, style = T.Screen.cell sc ~row ~col in
            not (Matrix.Ansi.Attr.mem Inverse style.Matrix.Ansi.Style.attrs))
          cols
      in
      let at text =
        match T.Screen.find sc text with
        | p :: _ -> p
        | [] -> failf "no %s" text
      in
      let range a b = List.init (b - a + 1) (fun i -> a + i) in
      let row, _ = at "tidy-otter" in
      equal ~msg:"the runs pane's row, cells not reversed" (list int) []
        (plain row (range 0 32));
      (* The strip's mark, right of the runs pane's 34 columns. *)
      let y, x =
        match List.filter (fun (_, x) -> x >= 34) (T.Screen.find sc "▌") with
        | p :: _ -> p
        | [] -> fail "no strip's mark"
      in
      List.iter
        (fun r ->
          equal
            ~msg:(Printf.sprintf "the strip's row %d, cells not reversed" r)
            (list int) []
            (plain r (range x (T.Screen.cols sc - 2))))
        [ y; y + 1 ])

(* A kept filter that keeps no run: the pane holds it, and the centre says
   why it is empty, on the pane's heading row. *)
let filter_none =
  test "a kept filter that keeps no run stays, and the centre says so"
    (fun () ->
      with_store Fixture.[ short_running; short_amber ] @@ fun store ->
      let sc =
        show ~cols:120 ~rows:40 store Munin_tui.Runs
          [ char '/'; char 'q'; Key Enter ]
      in
      let row text = List.map fst (T.Screen.find sc text) in
      equal ~msg:"the centre's sentence" (list int) (row "¹ runs")
        (row "no run matches the filter");
      let words y =
        List.filter (( <> ) "")
          (String.split_on_char ' '
             (Option.value
                (before (T.Screen.row sc y) "│")
                ~default:(T.Screen.row sc y)))
      in
      equal ~msg:"the pane's filter line" (list (list string))
        [ [ "/"; "q"; "0"; "of"; "2" ] ]
        (List.map words (row "/ q")))

(* [holds sub s] is whether [s] holds [sub]. *)
let holds sub s =
  let n = String.length sub in
  let rec go i = i + n <= String.length s && (String.sub s i n = sub || go (i + 1)) in
  go 0

let braille_dots =
  [ (0x01, 0, 0); (0x02, 0, 1); (0x04, 0, 2); (0x40, 0, 3);
    (0x08, 1, 0); (0x10, 1, 1); (0x20, 1, 2); (0x80, 1, 3) ]

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

(* [run_pane_name screen] is the name the run pane's heading gives. *)
let run_pane_name screen =
  List.nth_opt
    (List.filter (( <> ) "") (String.split_on_char ' ' (run_pane screen)))
    1

(* [run_pane_heading screen] is the run pane's first line: the run's glyph,
   name and state word. *)
let run_pane_heading screen = before (run_pane screen) " / "

(* [legend_entry screen name] is the legend's entry of the run [name], up to
   the gap before the next entry. *)
let legend_entry screen name =
  match T.Screen.find screen ("━ " ^ name) with
  | (y, x) :: _ ->
      let r = from x (T.Screen.row screen y) in
      Some (String.trim (Option.value (before r "   ") ~default:r))
  | [] -> None

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
          first [ tidy ] tidy;
          first [ tidy ] tidy ~script:arrives;
          first [ tidy ] tidy ~script:stalls;
          first Fixture.states Fixture.far_node ~script:(fun store _ ->
              feed store Fixture.far_node ~step:3200);
          first Fixture.states Fixture.lost_node;
          first [ Fixture.mlp_base ] Fixture.mlp_base;
          first [ Fixture.amber_fox ] Fixture.amber_fox;
          first [ Fixture.eager_hawk ] Fixture.eager_hawk;
          first Fixture.states Fixture.quiet_heron;
        ]
      in
      golden "states" (String.concat "\n" lines ^ "\n"))

(* A run whose writer holds its lock may log again after a pause, so a
   stalled run keeps its rounded end and its axis does not change when
   samples arrive again. *)
let stalled_end =
  test "a stalled run keeps its rounded end" (fun () ->
      let r =
        Fixture.sweep_run ~id:"20261101T120000Z-5a11ed00" ~name:"paused"
          ~started:(Fixture.now -. 100.) ~pid:4242 ~commit:Fixture.commit ~dirty:false ~tags:[]
          Fixture.base ~upto:150 ~ended:None ~status:`Running ()
      in
      with_store ~name:"stalled" [ r ] @@ fun store ->
      let run = Option.get (Store.run store r.id) in
      (* Samples arrived at 0 and 1 s, and none for 999 s since. *)
      let watch =
        Watch.observe (Watch.observe (Watch.start ~samples:0) ~now:0. ~samples:2) ~now:1.
          ~samples:4
      in
      let f =
        Munin_tui.Facts.of_run ~now:1000. ~history:(Some (History.read run)) ~hint:None
          ~watch:(Some watch) ~pauses:[] run
      in
      equal ~msg:"its state" string "stalled" (Watch.word f.state);
      equal ~msg:"its end, 150 rounded up" float_exact 200. f.x_end)

(* Runs: the filter and the best value of each experiment *)

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

(* The rows read the main store and write nothing to it. *)
let filter_store = lazy (shared_store "filter" Fixture.runs)

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
      ("lr<0.002", [ "tidy-otter"; "amber-fox"; "brisk-wren" ]);
      ( "lr>2e-3",
        [ "calm-lynx"; "dusty-mole"; "eager-hawk"; "mlp-wide"; "mlp-base" ] );
      ("nothing", []);
    ]
    (fun (f, expected) ->
      if List.mem "mlp-base" expected then needs_crashed ();
      let store = Lazy.force filter_store in
      let screen =
        show ~cols:100 ~rows:30 store Munin_tui.Runs
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
        Fixture.write_log store ~goal:`Minimize ~steps:20000
          [ Fixture.sample (0, 1.); Fixture.sample (19950, 0.5) ]
      in
      let run = show ~cols:120 ~rows:40 store (Munin_tui.Run (Some id)) [] in
      let runs = show ~cols:120 ~rows:40 store Munin_tui.Runs [] in
      (* The screen's percentages: the run has no accuracy, so they are
         its progress. *)
      let percents screen =
        List.concat_map
          (fun y ->
            List.filter
              (String.ends_with ~suffix:"%")
              (String.split_on_char ' ' (T.Screen.row screen y)))
          (List.init (T.Screen.rows screen) Fun.id)
      in
      equal ~msg:"on the run" (list string) [ "99%" ] (percents run);
      equal ~msg:"on the runs" (list string) [ "99%" ] (percents runs))

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
        (Fixture.write_log store ~goal:`Minimize ~started:1
           (List.map (fun st -> Fixture.sample (st, 0.)) marked));
      ignore
        (Fixture.write_log store ~goal:`Minimize ~started:2
           (List.init 100 (fun k ->
                Fixture.sample (k * 10, Float.of_int (k * 10) /. 1000.))));
      (* The cursor's run, the newest, is the first row: overlay the second,
         come back and open the strips. *)
      let screen =
        show ~cols:120 ~rows:40 store Munin_tui.Runs
          [ char 'j'; Text " "; char 'k'; Key Enter ]
      in
      (* The strip's last word, left of the run pane's border. *)
      let difference =
        match T.Screen.find screen "▌m" with
        | (y, x) :: _ ->
            let r = from x (T.Screen.row screen y) in
            let r = Option.value (before r "│") ~default:r in
            List.nth_opt
              (List.rev (List.filter (( <> ) "") (String.split_on_char ' ' r)))
              0
        | [] -> None
      in
      equal (option string) (Some value) difference)

(* A metric whose every sample a resume orphans has no statistics, and Details
   lists the run all the same. *)
let all_orphans =
  test "the run's facts list a run whose metric is all orphans" (fun () ->
      with_store ~name:"orphaned" [] @@ fun store ->
      let id, _ =
        Fixture.write_log store
          ~declare:
            [
              Fixture.declare ~goal:`Minimize "m";
              Fixture.declare ~goal:`Maximize "val/acc";
            ]
          [
            Fixture.sample (100, 1.);
            Fixture.values 900 [ ("val/acc", 0.5) ];
            Fixture.resume 500;
            Fixture.sample (600, 0.5);
            Fixture.finished;
          ]
      in
      let screen =
        show ~cols:120 ~rows:40 store (Munin_tui.Run (Some id)) [ char 'i' ]
      in
      (* The keys of the info's metrics: the first word of each row of its
         value column, from [metrics] to [notes]. *)
      let row label =
        match T.Screen.find screen label with
        | p :: _ -> p
        | [] -> failf "no %s" label
      in
      let y, x = row "metrics" and y_end, _ = row "notes" in
      let keys =
        List.filter_map
          (fun y ->
            let r = from (x + 12) (T.Screen.row screen y) in
            let r = Option.value (before r "│") ~default:r in
            match String.split_on_char ' ' (String.trim r) with
            | "" :: _ | [] -> None
            | k :: _ -> Some k)
          (List.init (y_end - y) (fun i -> y + i))
      in
      equal (list string) [ "m" ] keys)

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
            ignore (Fixture.write_log store ~experiment ~started:((e * 11) + k) [])
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
  (* Each case starts a writer and runs a session of up to 18 s of polls,
     about 50 ms: the law's time is its 30 sessions. *)
  prop ~count:30 "a live run stays live through depths and cursor moves"
    key_gen (fun keys ->
      (* Two runs, so that j and k move the cursor off tidy-otter and
         back. *)
      with_store Fixture.[ short_running; short_amber ] @@ fun store ->
      let script _ = feed store Fixture.short_running ~step:650 in
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
      with_store Fixture.[ short_running; short_amber; short_brisk ] @@ fun store ->
      let log =
        Filename.concat
          (Run.dir (Option.get (Store.run store (id_of Fixture.short_amber))))
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
      equal (option string) (Some "✓ amber-fox  finished")
        (run_pane_heading screen))

(* A run logs its last samples and ends after its chart was prepared, and the
   cursor leaves it before a poll reads them: coming back draws them, since a
   chart prepared before the end is not kept. *)
let ended_away =
  test "a cursor back on a run that ended away draws its last samples"
    (fun () ->
      with_store [ Fixture.short_amber ] @@ fun store ->
      let s =
        Session.start ~store ~experiment:"late" ~name:"late-end" ~steps:100
          ~provenance:(Provenance.detect ~cwd:"/" ())
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
      equal ~msg:"the legend" (option string) (Some "━ late-end 9")
        (legend_entry screen "late-end");
      equal ~msg:"the run pane" (option string) (Some "✓ late-end  finished")
        (run_pane_heading screen))

(* A run ends between two timer polls after logging its last samples, while
   the dashboard draws it as a trend only (the cursor is on another run) or as
   the cursor's strips with no readout of its metric: what it draws of the run
   is what it draws of the same run read whole after its end. *)
let ended_between =
  let late ~between keys =
    with_store [ Fixture.short_amber ] @@ fun store ->
    let s =
      Session.start ~store ~experiment:"late" ~name:"late-end" ~steps:100
        ~provenance:(Provenance.detect ~cwd:"/" ())
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
  (* One run: a session reads and draws every run of its store, and the
     depths need one. *)
  let store =
    lazy
      (let root = Filename.concat (Lazy.force shared) "depths" in
       let store = Store.open_ ~root () in
       let id, _ =
         Fixture.write_log store (List.init 20 (fun i -> Fixture.sample (i, 1.)))
       in
       (store, id))
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
    {
      depth = 0;
      help = false;
      typing = false;
      info = false;
      filter = false;
      typed = false;
    }
  in
  let word = Gen.(string_of ~size:(int_range 1 3) (of_list no_run)) in
  let typed k =
    List.mem k
      (Key Enter :: Key Escape
      :: List.map (fun c -> Text (String.make 1 c)) no_run)
  in
  (* A dashboard is the keys sent so far, newest first, replayed on a fresh
     session at each look, and its model the state they lead to. Both are
     cells that the keys change, so that a look reads the latest. *)
  let key =
    command "key"
      ~pre:(fun k st -> if !st.typing then typed k else k <> Text "q")
      (key_gen @-> dash ^-> returns unit)
      (fun k st -> st := model_key !st k)
      (fun k keys -> keys := `Key k :: !keys)
  in
  let look keys =
    let store, id = Lazy.force store in
    let start, keys =
      match List.rev !keys with
      | `Start_run :: rest -> (Munin_tui.Run (Some id), rest)
      | l -> (Munin_tui.Runs, l)
    in
    let keys =
      List.filter_map (function `Key k -> Some k | `Start_run -> None) keys
    in
    let term = T.Terminal.create ~cols:120 ~rows:40 () in
    (elapsed := fun () -> 0);
    let app = Munin_tui.app ~clock store start in
    Mosaic_test.run term app @@ fun s ->
    (elapsed := fun () -> T.now s);
    T.advance s second;
    List.iter (send term s) keys;
    T.settle s;
    observed (T.Terminal.screen term)
  in
  let screen =
    command "screen" (dash ^-> returns string)
      (fun st -> model_name !st)
      look
  in
  stateful ~count:40 ~steps:10
    "Enter goes deeper, Esc back, q is typed in the filter"
    [
      command "runs"
        (Gen.unit @-> makes dash)
        (fun () -> ref start)
        (fun () -> ref []);
      command "run"
        (Gen.unit @-> makes dash)
        (fun () -> ref start)
        (fun () -> ref [ `Start_run ]);
      (* Listed four and two times, so that a look follows several keys. *)
      key;
      key;
      key;
      key;
      (* A filter kept with text, which keeps no run. *)
      command "filter"
        ~pre:(fun _ st -> not (!st.typing || !st.help || !st.info))
        (word @-> dash ^-> returns unit)
        (fun _ st -> st := { !st with filter = true })
        (fun w keys ->
          let chars = List.of_seq (String.to_seq w) in
          keys :=
            (`Key (Key Enter)
            :: List.rev_map (fun c -> `Key (Text (String.make 1 c))) chars)
            @ (`Key (Text "/") :: !keys));
      screen;
      screen;
    ]

(* [after_keys keys] is what the dashboard of one run shows after [keys],
   from its status bar. *)
let after_keys keys =
  with_store ~name:"after-keys" [] @@ fun store ->
  ignore (Fixture.write_log store (List.init 20 (fun i -> Fixture.sample (i, 1.))));
  observed (show ~cols:120 ~rows:40 store Munin_tui.Runs keys)

(* The depths in a fixed order: each Enter one deeper, to zoom, and each Esc
   one back. *)
let depth_keys =
  cases ~id:fst "Enter goes one depth deeper and Esc one back"
    [
      ("Enter", ([ Key Enter ], "strips"));
      ("Enter twice", ([ Key Enter; Key Enter ], "zoom"));
      ("Enter at zoom", ([ Key Enter; Key Enter; Key Enter ], "zoom"));
      ("Esc at zoom", ([ Key Enter; Key Enter; Key Escape ], "strips"));
      ("Esc at strips", ([ Key Enter; Key Escape ], "charts"));
    ]
    (fun (_, (keys, depth)) -> equal string depth (after_keys keys))

let shown_over =
  cases ~id:fst "? and i show over the depth, and close as they opened"
    [
      ("?", ([ char '?' ], "help"));
      ("? ?", ([ char '?'; char '?' ], "charts"));
      ("? Esc", ([ char '?'; Key Escape ], "charts"));
      ("Enter ? Esc", ([ Key Enter; char '?'; Key Escape ], "strips"));
      ("i", ([ char 'i' ], "info"));
      ("i i", ([ char 'i'; char 'i' ], "charts"));
      ("Enter i Esc", ([ Key Enter; char 'i'; Key Escape ], "strips"));
    ]
    (fun (_, (keys, shown)) -> equal string shown (after_keys keys))

let typed_q =
  cases ~id:fst "q is typed in the filter"
    [
      ("/ q", ([ char '/'; char 'q' ], "typing"));
      ("/ q Enter", ([ char '/'; char 'q'; Key Enter ], "charts"));
    ]
    (fun (_, (keys, shown)) -> equal string shown (after_keys keys))

(* Six runs of one experiment, [r1] the oldest, each with samples of [m]. *)
let six store =
  List.iter
    (fun k ->
      ignore
        (Fixture.write_log store ~started:k
           ~name:(Printf.sprintf "r%d" k)
           (List.init 20 (fun i -> Fixture.sample (i, Float.of_int (i * k))))))
    [ 1; 2; 3; 4; 5; 6 ]

(* The names of the runs pane's rows, top to bottom. *)
let listed screen =
  List.filter_map
    (fun y ->
      List.find_opt
        (fun w -> String.length w = 2 && w.[0] = 'r')
        (String.split_on_char ' ' (pane screen y)))
    (List.init (T.Screen.rows screen) Fun.id)

(* The names the legend gives, on the row of its first entry: the word
   after each [━ ]. *)
let legend screen =
  match List.filter (fun (_, x) -> x > 42) (T.Screen.find screen "━ ") with
  | (y, _) :: _ ->
      let rec names r =
        match before r "━ " with
        | None -> []
        | Some head ->
            let rest =
              String.sub r (String.length head + String.length "━ ")
                (String.length r - String.length head - String.length "━ ")
            in
            List.hd (String.split_on_char ' ' rest) :: names rest
      in
      names (from 42 (T.Screen.row screen y))
  | [] -> []

let six_keys =
  cases
    ~id:(fun (name, _, _) -> name)
    "each key does what the keys table says"
    [
      ("G the last run", [ char 'G' ], `Cursor "r1");
      ("g the first run", [ char 'G'; char 'g' ], `Cursor "r6");
      ("] the next run at strips", [ Key Enter; char ']' ], `Cursor "r5");
      ( "[ the previous run at strips",
        [ Key Enter; char ']'; char '[' ],
        `Cursor "r6" );
      ( "o orders by name",
        [ char 'o'; char 'o' ],
        `Listed [ "r1"; "r2"; "r3"; "r4"; "r5"; "r6" ] );
      ("e folds the experiment", [ char 'e' ], `Listed []);
      (* Five overlays asked for: the fifth, r1, is refused. *)
      ( "space overlays four runs at most",
        List.concat (times 5 [ char 'j'; Text " " ]) @ [ char 'g' ],
        `Legend [ "r2"; "r3"; "r4"; "r5"; "r6" ] );
      ( "c clears the overlays",
        [ char 'j'; Text " "; char 'k'; char 'c' ],
        `Legend [ "r6" ] );
      ( "Esc at charts clears the overlays",
        [ char 'j'; Text " "; char 'k'; Key Escape ],
        `Legend [ "r6" ] );
    ]
    (fun (_, keys, expected) ->
      with_store ~name:"six" [] @@ fun store ->
      six store;
      let sc = show ~cols:160 ~rows:48 store Munin_tui.Runs keys in
      match expected with
      | `Cursor name -> equal (option string) (Some name) (run_pane_name sc)
      | `Listed names -> equal (list string) names (listed sc)
      | `Legend names ->
          equal (list string) names (List.sort String.compare (legend sc)))

(* [run] reads the store before it asks for a terminal. *)
let unknown_run =
  test "run is an error for a run the store lacks" (fun () ->
      with_store [] @@ fun store ->
      is_error (Munin_tui.run store (Munin_tui.Run (Some "absent"))))

(* [tree root] is each file under [root], with its size and modification
   time. *)
let tree root =
  let rec walk path acc =
    let st = Unix.lstat path in
    match st.st_kind with
    | Unix.S_DIR ->
        Array.fold_left
          (fun acc e -> walk (Filename.concat path e) acc)
          acc (Sys.readdir path)
    | _ -> (path, (st.st_size, st.st_mtime)) :: acc
  in
  List.sort compare (walk root [])

let never_writes =
  test "the dashboard never writes the store" (fun () ->
      with_store Fixture.[ short_running; short_amber ] @@ fun store ->
      let before = tree (Store.root store) in
      ignore
        (show ~cols:120 ~rows:40 store Munin_tui.Runs
           [
             Key Enter; Key Enter; char '+'; Key Escape; char 'i'; Key Escape;
             char '?'; Key Escape; Text " "; char 'r'; char 's'; char 'y';
             char 'o'; char 'e'; char '/'; char 'x'; Key Escape;
           ]);
      equal
        (list (pair string (pair int float_exact)))
        before
        (tree (Store.root store)))

(* Where the runs pane is hidden, the filter shows it while it is typed:
   the text typed is in its line, and Enter keeps it. *)
let hidden_filter =
  test "the filter shows the runs pane where it is hidden" (fun () ->
      with_store ~name:"hidden" Fixture.[ short_running; short_amber ] @@ fun store ->
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

(* The strips of a run that recorded readings end with its machine's, live
   or ended: the readings the run pane gives, in its order. *)
let machine_strips =
  cases
    ~id:(fun (name, _) -> name)
    "a run's strips end with its machine's readings, live or ended"
    [ ("live", Fixture.short_running); ("ended", Fixture.short_amber) ]
    (fun (_, r) ->
      with_store [ r ] @@ fun store ->
      (* An arrival makes a running run live; an ended run takes none. *)
      let script _ =
        if r.Fixture.status = `Running then feed store r ~step:650
      in
      let start = Munin_tui.Run (Some (id_of r)) in
      let screen =
        show ~script ~cols:120 ~rows:40 store start
          (Key Enter :: times 30 (Key Down))
      in
      let labels, _ = strips screen in
      let last = List.filteri (fun i _ -> i >= List.length labels - 4) labels in
      equal (list string) [ "GPU busy"; "VRAM"; "CPU busy"; "RAM" ] last)

(* The strips scroll the least that shows the cursor's strip whole, from the
   strips shown: a move to a strip shown leaves them where they are. *)
let strips_scroll =
  (* One run, whose strips are its own beside any other. *)
  let store =
    lazy
      (shared_store "strips"
         [ Fixture.short_running ])
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
      let start = Munin_tui.Run (Some (id_of Fixture.short_running)) in
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

(* [strip_ink screen] is, for each strip under the legend whose two rows
   are above the status bar, its label and its braille dots, each its dot
   column, its dot row and its colour. *)
let strip_ink screen =
  let last = T.Screen.rows screen - 2 in
  let at row col = fst (T.Screen.cell screen ~row ~col) in
  let label row = String.trim (String.concat "" (List.init 16 (fun i -> at row (strip_col + 1 + i)))) in
  let rows = List.init last Fun.id in
  let legend = Option.value (List.find_opt (fun row -> at row strip_col = "━") rows) ~default:last in
  let firsts =
    List.filter
      (fun row ->
        row > legend
        && List.mem (at row strip_col) [ " "; "▌" ]
        && not (List.mem (at row (strip_col + 1)) [ " "; ""; "↓"; "↑" ]))
      rows
  in
  let dots ~top row col =
    let ch, style = T.Screen.cell screen ~row ~col in
    let cp = if ch = "" then 0 else Uchar.to_int (Uchar.utf_decode_uchar (String.get_utf_8_uchar ch 0)) in
    if cp < 0x2800 || cp > 0x28FF then []
    else
      List.filter_map
        (fun (bit, dx, dy) ->
          if (cp - 0x2800) land bit = 0 then None
          else Some ((2 * col) + dx, (4 * (row - top)) + dy, style.Matrix.Ansi.Style.fg))
        braille_dots
  in
  let cols = List.init (T.Screen.cols screen) Fun.id in
  let ink top = List.concat_map (fun row -> List.concat_map (dots ~top row) cols) [ top; top + 1 ] in
  List.map (fun row -> (label row, ink row)) firsts

(* [last_dot ink] is the rightmost dot column of [ink]. *)
let last_dot ink = List.fold_left (fun e (c, _, _) -> max e c) (-1) ink

(* Law 1: the strips share one x axis, so that a column is one moment in
   each. The cursor's run ended at step 3,000 with [m], [n] and readings of
   its GPU's memory; the overlay ended at 5,000 with [m] alone. The cursor's
   line ends in one dot column in every strip, though the overlay widens
   neither [n]'s nor the machine's. The cursor's colour is that of [n],
   which the overlay does not draw. *)
let shared_axis =
  test "the strips share one x axis" (fun () ->
      with_store ~name:"shared" [] @@ fun store ->
      let declare = [ Fixture.declare "m"; Fixture.declare "n" ] in
      let ended = Fixture.finished in
      ignore
        (Fixture.write_log store ~declare ~started:1
           (List.init 51 (fun k -> Fixture.values (k * 100) [ ("m", 1.) ]) @ [ ended ]));
      let cursor k =
        [
          Fixture.values (k * 100) [ ("m", 0.); ("n", 0.) ];
          Fixture.reading ~time:2. (k * 100) [ ("dev/CUDA:0/used_bytes", 1e9) ];
        ]
      in
      ignore (Fixture.write_log store ~declare ~started:2 (List.concat (List.init 31 cursor) @ [ ended ]));
      (* The cursor's run, the newest, is the first row: overlay the second,
         come back and open the strips. *)
      let screen =
        show ~cols:120 ~rows:40 store Munin_tui.Runs [ char 'j'; Text " "; char 'k'; Key Enter ]
      in
      let strips = strip_ink screen in
      equal ~msg:"the strips" (list string) [ "m"; "n"; "VRAM" ] (List.map fst strips);
      let colour = match List.assoc "n" strips with (_, _, c) :: _ -> c | [] -> None in
      let mine key = List.filter (fun (_, _, c) -> c = colour) (List.assoc key strips) in
      let n = last_dot (mine "n") in
      equal ~msg:"the cursor's last dot column in m" int n (last_dot (mine "m"));
      equal ~msg:"its last in VRAM" int n (last_dot (List.assoc "VRAM" strips)))

(* A reading is drawn at the step it carries, the run's latest step when it
   was read, so its column is the column of a metric's sample at that step.
   The run reads its GPU's memory at step 0 before its first sample of [m],
   then at each of [m]'s steps from 300 to 600; [n] spans the steps 0 to
   1,000. The machine's strip starts in [n]'s first dot column and ends in
   [m]'s last. *)
let reading_steps =
  test "a reading is drawn at its step" (fun () ->
      with_store ~name:"reading-steps" [] @@ fun store ->
      let declare = [ Fixture.declare "m"; Fixture.declare "n" ] in
      let vram = "dev/CUDA:0/used_bytes" in
      let n = List.init 101 (fun k -> Fixture.values (k * 10) [ ("n", 1.) ]) in
      let paired k =
        [ Fixture.values (k * 10) [ ("m", 1.) ]; Fixture.reading ~time:2. (k * 10) [ (vram, 1e9) ] ]
      in
      let read = (Fixture.reading ~time:2. 0 [ (vram, 1e9) ] :: n) @ List.concat (List.init 31 (fun k -> paired (30 + k))) in
      let id, _ = Fixture.write_log store ~declare (read @ [ Fixture.finished ]) in
      let strips = strip_ink (show ~cols:120 ~rows:40 store (Munin_tui.Run (Some id)) [ Key Enter ]) in
      equal ~msg:"the strips" (list string) [ "m"; "n"; "VRAM" ] (List.map fst strips);
      let first ink = List.fold_left (fun e (c, _, _) -> min e c) max_int ink in
      let ink key = List.assoc key strips in
      equal ~msg:"its first dot column, n's" int (first (ink "n")) (first (ink "VRAM"));
      equal ~msg:"its last, m's" int (last_dot (ink "m")) (last_dot (ink "VRAM")))

(* [resumed_log store ~crashed] is the id of a run that read 20 GB of GPU
   memory from step 450 to 700, its records timed [crashed], then resumed
   from step 400 at 3 s and read 1 GB from there to 1,000. *)
let resumed_log store ~crashed =
  let vram = "dev/CUDA:0/used_bytes" in
  let attempt ~time ~from ~upto gb =
    List.concat
      (List.init (((upto - from) / 50) + 1) (fun k ->
           let step = from + (k * 50) in
           [
             Fixture.values ~time step [ ("m", 1.) ];
             Fixture.reading ~time step [ (vram, gb *. 1e9) ];
           ]))
  in
  let records =
    attempt ~time:crashed ~from:450 ~upto:700 20.
    @ [ Fixture.resume 400 ]
    @ attempt ~time:3.5 ~from:400 ~upto:1000 1.
    @ [ Fixture.finished ]
  in
  fst (Fixture.write_log store records)

(* A resume abandons the history after its step: the strips draw the
   history the run kept: [resumed_log]'s machine strip draws a constant
   1 GB, every dot column at one height. The log's order decides, so the
   resumed writer's clock may be behind the crashed one's: the resume is
   logged at 3 s, after readings logged at 5 s. *)
let resumed_readings =
  cases "a resume leaves the readings it abandoned out of the strips"
    ~id:(fun (name, _) -> name)
    [ ("the resumed writer's clock ahead", 2.); ("the resumed writer's clock behind", 5.) ]
    (fun (_, crashed) ->
      with_store ~name:"resumed" [] @@ fun store ->
      let id = resumed_log store ~crashed in
      let strips = strip_ink (show ~cols:120 ~rows:40 store (Munin_tui.Run (Some id)) [ Key Enter ]) in
      let tops = Hashtbl.create 64 in
      List.iter
        (fun (c, r, _) ->
          let top = Option.value (Hashtbl.find_opt tops c) ~default:max_int in
          Hashtbl.replace tops c (min top r))
        (List.assoc "VRAM" strips);
      let heights = List.sort_uniq compare (Hashtbl.fold (fun _ r acc -> r :: acc) tops []) in
      equal ~msg:"the top dot rows of its columns" (list int)
        [ List.hd heights ] heights)

(* Zoom's cursor walks the samples the chart draws: on the VRAM of
   [resumed_log]'s run, twelve moves left from the latest step reach step
   400, the resumed attempt's first, and one more stays there, below the
   abandoned steps. *)
let resumed_cursor =
  test "zoom's cursor skips the readings a resume abandoned" (fun () ->
      with_store ~name:"resumed-cursor" [] @@ fun store ->
      let id = resumed_log store ~crashed:2. in
      let zoom = [ Key Enter; Key Down; Key Enter ] in
      let at lefts =
        let screen = show ~cols:120 ~rows:40 store (Munin_tui.Run (Some id)) (zoom @ times lefts (Key Left)) in
        List.find_opt (fun s -> holds ("at step " ^ s) (T.Screen.text screen)) [ "400"; "700"; "1,000" ]
      in
      equal ~msg:"the start" (option string) (Some "1,000") (at 0);
      equal ~msg:"twelve moves left" (option string) (Some "400") (at 12);
      equal ~msg:"one more" (option string) (Some "400") (at 13))

(* The strips list shows whole strips: at any height and after any moves,
   the strips from a first, in order, right under the legend, the cursor's
   among them, as many as fit, and a line that says how many follow. A
   strip takes three lines with the gap under it, the machine's header one
   more over its first strip shown, and the line of those that follow two
   with its gap. *)
let strips_whole =
  (* One run, whose strips are its own beside any other. *)
  let store =
    lazy
      (shared_store "whole" [ Fixture.short_running ])
  in
  let start = Munin_tui.Run (Some (id_of Fixture.short_running)) in
  let every =
    lazy (fst (strips (show ~cols:120 ~rows:100 (Lazy.force store) start [ Key Enter ])))
  in
  let machine = [ "GPU busy"; "VRAM"; "CPU busy"; "RAM" ] in
  let gen =
    Gen.(
      with_pp
        (fun ppf (rows, downs) -> Format.fprintf ppf "%d rows, %d down" rows downs)
        (pair (int_range 16 40) (int_range 0 24)))
  in
  prop ~count:20
    "the strips shown are whole, as many as fit, and say how many follow"
    ~examples:[ (16, 0); (16, 24) ]
    gen
    (fun (rows, downs) ->
      let every = Lazy.force every in
      let n = List.length every in
      let screen =
        show ~cols:120 ~rows (Lazy.force store) start (Key Enter :: times downs (Key Down))
      in
      let at row col = fst (T.Screen.cell screen ~row ~col) in
      let status = rows - 1 in
      let lines = List.init status Fun.id in
      let legend = Option.get (List.find_opt (fun row -> at row strip_col = "━") lines) in
      let label row = String.trim (String.concat "" (List.init 16 (fun i -> at row (strip_col + 1 + i)))) in
      let firsts =
        List.filter
          (fun row ->
            row > legend
            && List.mem (at row strip_col) [ " "; "▌" ]
            && not (List.mem (at row (strip_col + 1)) [ " "; ""; "↓"; "↑" ]))
          lines
      in
      let shown = List.map label firsts in
      let k = List.length shown in
      let first = match shown with [] -> n | s :: _ -> Option.value (List.find_index (( = ) s) every) ~default:n in
      let follows = n - first - k in
      let header = List.exists (fun s -> List.mem s machine) shown in
      cover "strips are hidden above" (first > 0);
      cover "strips follow" (follows > 0);
      cover "the machine's header is shown" header;
      if not (List.exists (fun row -> at row strip_col = "▌") firsts) then
        fail ("the cursor's strip is not shown:\n" ^ T.Screen.text screen);
      equal ~msg:"the strips, from the first shown" (list string)
        (List.filteri (fun i _ -> i >= first && i < first + k) every)
        shown;
      List.iter
        (fun row -> at_most ~msg:("the second row of " ^ label row) int ~than:(status - 1) (row + 1))
        firsts;
      let top = if header && List.mem (List.hd shown) machine then legend + 3 else legend + 2 in
      equal ~msg:"the first strip's row" int top (List.hd firsts);
      let mores = List.filter (fun row -> holds " more" (T.Screen.row screen row)) lines in
      let expected = if follows = 0 then [] else [ Printf.sprintf "+%d more" follows ] in
      equal ~msg:"the line of those that follow" (list string) expected
        (List.map (fun row -> String.trim (String.concat "" (List.init 10 (fun i -> at row (strip_col + i))))) mores);
      (* The lines no strip, gap or line of those that follow takes cannot
         hold the strip after the last shown, nor, when none follows, the
         one before the first. The last strip's gap is room when none
         follows. *)
      let last = List.nth firsts (k - 1) in
      let room = status - if follows > 0 then last + 4 else last + 2 in
      let need =
        if follows = 0 then 3
        else
          let next_header = (not header) && List.mem (List.nth every (first + k)) machine in
          (if follows = 1 then 1 else 3) + if next_header then 1 else 0
      in
      if first > 0 || follows > 0 then less ~msg:"the lines left" int ~than:need room)

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
  (* Three runs of a few samples, so that the cursor's moves meet both ends
     of the list: the newest without an end, the others ended. *)
  let store =
    lazy
      (let root = Filename.concat (Lazy.force shared) "path" in
       let store = Store.open_ ~root () in
       let samples = List.init 20 (fun i -> Fixture.sample (i, 1.)) in
       List.iter
         (fun (started, ended) ->
           ignore
             (Fixture.write_log store ~started
                (samples @ if ended then [ Fixture.finished ] else [])))
         [ (1, true); (2, true); (3, false) ];
       store)
  in
  let last = 2 in
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
  (* Each call replays every call before it on a fresh dashboard, so a
     program of [steps] calls costs steps²/2 sessions: few steps and more
     programs reach the cover at less cost. *)
  stateful ~count:80 ~steps:8
    "one poll runs at a time, and the asks meanwhile make one after it"
    [
      command "start"
        (Gen.unit @-> makes path)
        (fun () -> ref { phase = Reading; cursor = 0; chosen = false })
        (fun () -> ref []);
      (* Listed thrice, so that asks pile up while a poll runs. *)
      tick;
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

(* A run without a metric sample: one without an end says the charts are
   to come, an ended one that it logged none, each on the row of the runs
   pane's heading, as the hero is. *)
let no_sample =
  cases
    ~id:(fun (name, _, _) -> name)
    "a run without a metric sample says why the charts are missing"
    [
      ("without an end", [], "no metric sample yet");
      ("ended", [ Fixture.finished ], "the run logged no metric");
    ]
    (fun (_, records, says) ->
      with_store ~name:"no-sample" [] @@ fun store ->
      ignore (Fixture.write_log store ~declare:[] records);
      let screen = show ~cols:120 ~rows:40 store Munin_tui.Runs [] in
      let row text = match T.Screen.find screen text with (y, _) :: _ -> Some y | [] -> None in
      equal (option int) (row "¹ runs") (row says);
      equal ~msg:"the sentence ends there" (option string) (Some "")
        (Option.map
           (fun y ->
             (* What follows the sentence, up to the run pane's border. *)
             let r = T.Screen.row screen y in
             let at = String.length (Option.get (before r says)) + String.length says in
             let tail = String.sub r at (String.length r - at) in
             String.trim (Option.value (before tail "│") ~default:tail))
           (row says)))

(* [r] with no other run of the experiment to draw: the legend has no entry
   for them. *)
let no_others =
  test "r with no other run to draw adds no legend entry" (fun () ->
      with_store ~name:"no-others" [] @@ fun store ->
      ignore (Fixture.write_log store (List.init 20 (fun i -> Fixture.sample (i, Float.of_int i))));
      let before_r = show ~cols:120 ~rows:40 store Munin_tui.Runs [] in
      let after_r = show ~cols:120 ~rows:40 store Munin_tui.Runs [ char 'r' ] in
      let others sc = List.length (T.Screen.find sc "others") in
      equal ~msg:"before r, its hint" int 1 (others before_r);
      equal ~msg:"after r" int 0 (others after_r))

(* Runs of one name, listed or drawn, each end in the end of its id, the
   shortest that tells them apart, at least four characters; a run of a
   name of its own has none. The ids differ in their last character. *)
let same_names =
  test "runs of one name are told apart by their ids" (fun () ->
      with_store ~name:"same-names" [] @@ fun store ->
      let run ?name () =
        fst (Fixture.write_log store ?name (List.init 20 (fun i -> Fixture.sample (i, Float.of_int i))))
      in
      let ids = List.init 3 (fun _ -> run ~name:"cnn-adam" ()) in
      let lone = run ~name:"lone" () in
      (* The cursor on the newest, lone; down to each cnn-adam to overlay
         it, the last under the cursor. *)
      let keys = [ char 'j'; Text " "; char 'j'; Text " "; char 'j'; Text " " ] in
      let screen = show ~cols:160 ~rows:48 store Munin_tui.Runs keys in
      let tail id = String.sub id (String.length id - 4) 4 in
      List.iter
        (fun id ->
          let label = "cnn-adam " ^ tail id in
          (* The legend's entry, whole, and the runs pane's row, left of
             column 42, whose name gives way to the id's end. *)
          let pane =
            List.filter
              (fun (_, x) -> x < 42)
              (T.Screen.find screen (" " ^ tail id))
          in
          (* Each logged 19 at the latest step. *)
          equal ~msg:label (option string)
            (Some ("━ " ^ label ^ " 19"))
            (legend_entry screen label);
          equal ~msg:(tail id) int 1 (List.length pane))
        ids;
      equal ~msg:"lone" (list (pair int int)) []
        (T.Screen.find screen ("lone " ^ tail lone)))

(* Two runs of one name whose ids share their last five characters take
   six: the fewest that tell them apart. The strips' header keeps the
   suffix whole and cuts the name. *)
let long_suffixes =
  test "same-named runs whose ids end alike take longer suffixes" (fun () ->
      with_store ~name:"long-suffixes" [] @@ fun store ->
      let run id =
        ignore
          (Fixture.write_log store ~name:"cnn-adam" ~id
             (List.init 20 (fun i -> Fixture.sample (i, Float.of_int i))))
      in
      run "20261101T000000Z-00a12345";
      run "20261101T000000Z-00b12345";
      (* Overlay the newest, then the cursor to the other, and strips. *)
      let screen = show ~cols:120 ~rows:40 store Munin_tui.Runs [ Text " "; char 'j'; Key Enter ] in
      List.iter
        (fun label ->
          equal ~msg:label (option string) (Some ("━ " ^ label))
            (legend_entry screen label))
        [ "cnn-adam a12345"; "cnn-adam b12345" ];
      (* The strips' header names the overlay after [vs], up to the run
         pane's border. *)
      let vs =
        List.map
          (fun (y, x) ->
            let r = from x (T.Screen.row screen y) in
            String.trim (Option.value (before r "│") ~default:r))
          (T.Screen.find screen "vs ")
      in
      equal ~msg:"the header's overlay" (list string) [ "vs cn… b12345" ] vs)

(* [Run None] puts the cursor on the newest run without an end that has
   not crashed. A run written without a writer file and without an end
   reads silent: it has no end. *)
let run_none_newest =
  test "Run None opens on the newest run without an end" (fun () ->
      with_store ~name:"run-none" [] @@ fun store ->
      let samples = List.init 20 (fun i -> Fixture.sample (i, Float.of_int i)) in
      ignore (Fixture.write_log store ~name:"going" ~started:1 samples);
      ignore
        (Fixture.write_log store ~name:"ended" ~started:2
           (samples @ [ Fixture.finished ]));
      let screen = show ~cols:120 ~rows:40 store (Munin_tui.Run None) [] in
      equal (option string) (Some "going") (run_pane_name screen))

(* With every run ended, [Run None] waits, and takes the first run that
   starts. *)
let run_none_waits =
  test "Run None waits for a run to start" (fun () ->
      with_store ~name:"run-none-waits" [] @@ fun store ->
      let samples = List.init 20 (fun i -> Fixture.sample (i, Float.of_int i)) in
      let ended = Fixture.finished in
      ignore (Fixture.write_log store ~name:"ended" ~started:1 (samples @ [ ended ]));
      let term = T.Terminal.create ~cols:120 ~rows:40 () in
      (elapsed := fun () -> 0);
      let app = Munin_tui.app ~clock store (Munin_tui.Run None) in
      let waiting, started =
        Mosaic_test.run term app @@ fun s ->
        (elapsed := fun () -> T.now s);
        T.advance s second;
        let waiting = T.Terminal.screen term in
        ignore (Fixture.write_log store ~name:"late" ~started:2 samples);
        T.advance s (2 * second);
        (waiting, T.Terminal.screen term)
      in
      let says screen text =
        List.map (fun (y, _) -> String.trim (T.Screen.row screen y))
          (T.Screen.find screen text)
      in
      equal ~msg:"before a run starts" (list string)
        [ "waiting for a run in ~/run-none-waits" ]
        (says waiting "waiting for a run");
      equal ~msg:"once one starts" (option string) (Some "late")
        (run_pane_name started))

(* Cut text *)

(* A run whose texts outgrow their columns: a long name, host, command,
   directory, param keys and error. *)
let long_run =
  let w =
    { Fixture.seed = 5; lr = 1e-3; warmup = 100; floor = 0.3; val_floor = 0.4; overfit = 0.;
      rate = 2.; nan_from = None }
  in
  {
    (Fixture.sweep_run ~id:"20261101T020000Z-0c0c0c0c" ~name:"conservative-baseline"
       ~started:1793500000.0 ~pid:9610 ~commit:Fixture.commit ~dirty:false ~tags:[] w ~upto:1130
       ~ended:
         (Some
            ( `Failed
                "Rig.Lost (CUDA:0, \"out of memory\") while allocating 2.0 GiB for the activations of conv12",
              false ))
       ())
    with
    host = "build-node-with-a-long-name.cluster.internal";
    command =
      [ "/home/ada/src/resnet/_build/default/examples/train/train.exe"; "--learning-rate"; "0.001" ];
    cwd = "/home/ada/src/resnet/experiments/cifar/a/rather/deep/directory";
    params =
      [ ("step_size", `Float 0.001); ("max_iterations", `Int 50); ("batch_size", `Int 64) ];
  }

(* The info's value column, from [label]'s row to [next]'s: the rows' text
   between the column and the next border, joined without blanks. *)
let info_values screen label next =
  let y, x = match T.Screen.find screen label with p :: _ -> p | [] -> failf "no %s" label in
  let y_end = match T.Screen.find screen next with (y', _) :: _ -> y' | [] -> failf "no %s" next in
  let value y =
    let rest = from (x + 12) (T.Screen.row screen y) in
    let rest = Option.value (before rest "│") ~default:rest in
    String.concat ""
      (List.filter (fun c -> not (List.mem c [ " "; "█"; "▀"; "▄" ])) (chars rest))
  in
  String.concat "" (List.init (y_end - y) (fun i -> value (y + i)))

let no_blanks s = String.concat "" (String.split_on_char ' ' s)

(* Rule: a cut text ends in […] and keeps a blank before the next column; a
   number and the info's facts are never cut. The screens of [long_run],
   drawn once. *)
let long_screens =
  lazy
    (with_store ~name:"long" [ long_run; Fixture.short_amber ] @@ fun store ->
     let at = Munin_tui.Run (Some long_run.id) in
     List.map
       (fun (name, cols, rows, keys) -> (name, show ~cols ~rows store at keys))
       [
         ("160x48", 160, 48, []);
         ("120x40", 120, 40, []);
         ("info 120x40", 120, 40, [ char 'i' ]);
         ("info 80x24", 80, 24, [ char 'i' ]);
       ])

let long_screen name = List.assoc name (Lazy.force long_screens)
let found name sub = List.length (T.Screen.find (long_screen name) sub)

let info_pid =
  test "info gives the pid whole" (fun () ->
      List.iter
        (fun name -> equal ~msg:name int 1 (found name "pid 9610"))
        [ "info 120x40"; "info 80x24" ])

let info_command =
  test "info gives the command and the directory whole" (fun () ->
      List.iter
        (fun name ->
          equal ~msg:name string
            (no_blanks (String.concat "" long_run.command ^ long_run.cwd))
            (info_values (long_screen name) "command" "host"))
        [ "info 120x40"; "info 80x24" ])

let banner_whole =
  test "the banner gives the error whole, after the step" (fun () ->
      equal ~msg:"its start" int 1 (found "120x40" "at step 1,100: Rig.Lost");
      equal ~msg:"its end" int 1 (found "120x40" "conv12"))

let param_keys =
  test "a param key takes the longest key's width" (fun () ->
      equal ~msg:"the longest" int 1 (found "160x48" "max_iterations ");
      equal ~msg:"a shorter" int 1 (found "160x48" "step_size "))

let cut_blank =
  test "a cut text keeps a blank before the next column" (fun () ->
      List.iter
        (fun (name, sc) ->
          List.iter
            (fun (y, x) ->
              let next = from (x + 1) (T.Screen.row sc y) in
              match chars next with
              | [] | (" " | "│") :: _ -> ()
              | c :: _ -> failf "%s, row %d: … touches %S" name y c)
            (T.Screen.find sc "…"))
        (Lazy.force long_screens))

(* A run whose values outgrow their columns: sampled floats as params, and
   four metrics with a goal under long keys. A number shows whole: the run
   pane prints a param by the value rule, info gives it exactly, and the line
   of values keeps the metrics that fit. *)
let value_screens =
  let keys =
    [
      "validation/loss_total";
      "validation/accuracy_top1";
      "train/loss_total";
      "train/accuracy_top1";
    ]
  in
  let goal k = if holds "loss" k then `Minimize else `Maximize in
  let value k s = if holds "loss" k then 1. /. s else 0.9 -. (0.5 /. s) in
  lazy
    (with_store ~name:"long-values" [] @@ fun store ->
     ignore
       (Fixture.write_log store
          ~params:
            [
              ("learning_rate", `Float 0.0004213752718364);
              ("weight_decay", `Float 0.00031622776601683794);
            ]
          ~declare:(List.map (fun k -> Fixture.declare ~goal:(goal k) k) keys)
          (List.init 40 (fun i ->
               let s = Float.of_int (i + 1) in
               Fixture.values ((i + 1) * 100)
                 (List.map (fun k -> (k, value k s)) keys))));
     List.map
       (fun (name, cols, rows, keys) ->
         (name, show ~cols ~rows store Munin_tui.Runs keys))
       [
         ("120x40", 120, 40, []);
         ("80x24", 80, 24, []);
         ("info", 120, 40, [ char 'i' ]);
         ("help", 80, 24, [ char '?' ]);
       ])

let value_screen name = List.assoc name (Lazy.force value_screens)

let pane_params =
  test "the run pane prints a param's number as values print" (fun () ->
      (* Each number in the pane, up to the blank after it. *)
      let sc = value_screen "120x40" in
      let numbers =
        List.filter_map
          (fun (y, x) ->
            if x < 90 then None
            else
              let rest = from x (T.Screen.row sc y) in
              Some (List.hd (String.split_on_char ' ' rest)))
          (T.Screen.find sc "0.000")
      in
      equal (list string) [ "0.0004214"; "0.0003162" ] numbers)

let info_params =
  test "info gives a param's number as logged" (fun () ->
      equal int 1
        (List.length (T.Screen.find (value_screen "info") "0.0004213752718364")))

let values_line =
  test "the line of values keeps the values that fit whole" (fun () ->
      let sc = value_screen "80x24" in
      match List.sort compare (T.Screen.find sc "train/") with
      | (y, _) :: _ ->
          equal ~msg:"its cut texts" (list (pair int int)) []
            (List.filter (fun (y', _) -> y' = y) (T.Screen.find sc "…"))
      | [] -> fail "no line of values")

let help_whole =
  test "help at 80 columns cuts no text" (fun () ->
      let sc = value_screen "help" in
      let top =
        match T.Screen.find sc "keys" with
        | (y, _) :: _ -> y
        | [] -> fail "no help"
      in
      equal (list (pair int int)) []
        (List.filter (fun (y, _) -> y >= top) (T.Screen.find sc "…")))

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
  (* Two runs, so that the keys that move the cursor change the screen. A
     key's outcome is a session replayed from the start, one or two a hint:
     about a hundred sessions, the test's time. *)
  let store =
    lazy
      (shared_store "hints"
         [ Fixture.short_running; Fixture.short_amber ])
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

(* The status bar at any width: the hints that fit, leaving from the last,
   and [? keys] always. *)
let hints_fit =
  let store =
    lazy (shared_store "hints-fit" Fixture.[ short_running; short_amber ])
  in
  let bar cols = hints (show ~cols ~rows:24 (Lazy.force store) Munin_tui.Runs []) in
  let all = lazy (bar 200) in
  cases ~id:string_of_int "the status bar keeps ? keys and drops hints from the last"
    [ 60; 70; 80; 90; 100; 120; 160 ]
    (fun cols ->
      let shown = bar cols in
      let all = Lazy.force all in
      let keys = ("?", "keys") in
      if not (List.mem keys all) then fail "no ? keys at 200 columns";
      let others = List.filter (( <> ) keys) all in
      let rec prefix = function
        | [], _ -> true
        | x :: a, y :: b -> x = y && prefix (a, b)
        | _ :: _, [] -> false
      in
      equal ~msg:"the last hint" (option (pair string string)) (Some keys)
        (List.nth_opt shown (List.length shown - 1));
      let rest = List.filter (( <> ) keys) shown in
      if not (prefix (rest, others)) then
        failf "%s is not a prefix of %s"
          (String.concat ", " (List.map fst rest))
          (String.concat ", " (List.map fst others)))

(* A hint shows once the bar has room for it whole: at the width that holds
   the first three hints and [? keys] exactly, all four show, and one column
   less drops the third. *)
let hints_exact =
  let store =
    lazy (shared_store "hints-exact" Fixture.[ short_running; short_amber ])
  in
  let screen cols = show ~cols ~rows:24 (Lazy.force store) Munin_tui.Runs [] in
  test "a hint that fits whole shows" (fun () ->
      let wide = screen 200 in
      let all = hints wide in
      let others = List.filter (( <> ) ("?", "keys")) all in
      let w (k, word) = Munin_tui.Numbers.width (k ^ " " ^ word) in
      let row = String.trim (T.Screen.row wide (T.Screen.rows wide - 1)) in
      let dropped = List.filteri (fun i _ -> i >= 3) others in
      let exact =
        Munin_tui.Numbers.width row - List.fold_left (fun n h -> n + 3 + w h) 0 dropped
      in
      (* The bar's padding, a column each side. *)
      let cols = exact + 2 in
      let first n = List.filteri (fun i _ -> i < n) others @ [ ("?", "keys") ] in
      equal ~msg:(Printf.sprintf "at %d columns" cols) (list (pair string string)) (first 3)
        (hints (screen cols));
      equal ~msg:(Printf.sprintf "at %d columns" (cols - 1)) (list (pair string string)) (first 2)
        (hints (screen (cols - 1))))

(* Below 100 columns, a run without another metric with a goal has no line
   of values: its tabs follow the line that names it, a blank row apart. *)
let no_values_line =
  test "a run without values has no line of them" (fun () ->
      with_store ~name:"no-values" [] @@ fun store ->
      ignore
        (Fixture.write_log store ~declare:[ Fixture.declare "score" ]
           (List.init 20 (fun i -> Fixture.values i [ ("score", Float.of_int i) ])));
      let screen = show ~cols:80 ~rows:24 store Munin_tui.Runs [] in
      let row text = match T.Screen.find screen text with (y, _) :: _ -> Some y | [] -> None in
      equal (option int) (Option.map (( + ) 2) (row "‹ ")) (row "score"))

(* A strip's key takes the longest key's width, up to a share of the
   centre that leaves the plot its room. *)
let strip_keys =
  test "a strip's key is whole when the centre has room for it" (fun () ->
      with_store ~name:"strip-keys" [] @@ fun store ->
      let keys = [ "train/samples_per_s"; "m" ] in
      ignore
        (Fixture.write_log store ~declare:(List.map (fun k -> Fixture.declare k) keys)
           (List.init 20 (fun i -> Fixture.values i (List.map (fun k -> (k, Float.of_int i)) keys))));
      let screen = show ~cols:120 ~rows:40 store Munin_tui.Runs [ Key Enter ] in
      equal ~msg:"whole" int 1
        (List.length (T.Screen.find screen "train/samples_per_s "));
      equal ~msg:"cut" (list (pair int int)) []
        (T.Screen.find screen "train/samples_…"))

(* Zoom on a reading of the machine gives its values in the reading's unit,
   as its strip does: RAM in GiB, a busy share in percent. *)
let zoom_units =
  cases ~id:(fun (name, _, _) -> name) "zoom gives a reading in its unit"
    [ ("mem_used_bytes", [], " GiB"); ("cpu_busy_pct", [ Key Up ], "%") ]
    (fun (key, up, unit) ->
      with_store [ Fixture.short_running ] @@ fun store ->
      let script _ = feed store Fixture.short_running ~step:650 in
      (* The machine's strips end the list: the last is RAM, CPU busy before. *)
      let keys = [ Key Enter ] @ times 30 (Key Down) @ up @ [ Key Enter ] in
      let screen = show ~script ~cols:120 ~rows:40 store Munin_tui.Runs keys in
      let rows = List.init (T.Screen.rows screen) (T.Screen.row screen) in
      let with_text t = List.filter (holds t) rows in
      not_equal ~msg:"the zoom's title" (list string) [] (with_text key);
      equal ~msg:"raw byte counts" (list string) [] (with_text "e+");
      at_least ~msg:"rows of values in its unit" int ~than:2
        (List.length (with_text unit)))

(* The banner of a run whose metrics went non-finite names what each
   metric's first such sample was: [a] goes first, at step 5, [b] at 7. *)
let banner_names =
  cases ~id:(fun (name, _, _) -> name) "the banner names what the metrics went"
    [
      ("NaN", (Float.nan, Float.nan), "× a NaN from step 5; 2 metrics went NaN");
      ("inf", (Float.infinity, Float.infinity), "× a inf from step 5; 2 metrics went inf");
      ("-inf", (Float.neg_infinity, Float.neg_infinity), "× a -inf from step 5; 2 metrics went -inf");
      ("inf, then NaN", (Float.infinity, Float.nan), "× a inf from step 5; 2 metrics went NaN or inf");
    ]
    (fun (_, (a, b), says) ->
      with_store ~name:"banner" [] @@ fun store ->
      let value k i = if k = "a" && i >= 5 then a else if k = "b" && i >= 7 then b else Float.of_int i in
      ignore
        (Fixture.write_log store ~declare:[ Fixture.declare "a"; Fixture.declare "b" ]
           (List.init 10 (fun i -> Fixture.values i [ ("a", value "a" i); ("b", value "b" i) ])));
      let screen = show ~cols:160 ~rows:48 store Munin_tui.Runs [] in
      if T.Screen.find screen says = [] then
        failf "no %S in:\n%s" says
          (String.concat "\n" (List.init (T.Screen.rows screen) (T.Screen.row screen))))

(* Zoom's window is numbers, so its line wraps rather than cut one: after
   one [+] on a run of two million steps it reads from 1,000,000. *)
let zoom_window_whole =
  test "zoom's window gives its steps whole" (fun () ->
      with_store ~name:"window" [] @@ fun store ->
      ignore
        (Fixture.write_log store
           (List.init 41 (fun i -> Fixture.sample (i * 50_000, Float.of_int i)))
        );
      let screen = show ~cols:120 ~rows:40 store Munin_tui.Runs [ Key Enter; Key Enter; char '+' ] in
      let y, x = match T.Screen.find screen "showing" with p :: _ -> p | [] -> fail "no window" in
      (* The panel's lines from [showing] down to the status bar. *)
      let lines =
        List.init (T.Screen.rows screen - 1 - y) (fun k -> String.trim (from x (T.Screen.row screen (y + k))))
      in
      equal string "showing 1,000,000 – 2,000,000"
        (String.concat " " (List.filter (( <> ) "") lines)))

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
        (Fixture.write_log store
           ~declare:[ Fixture.declare ~goal:`Minimize "m"; Fixture.declare "w" ]
           (List.init 100 (fun i ->
                let t = Float.of_int i /. 99. in
                Fixture.values i [ ("m", 10. ** (3. *. t)); ("w", 1. +. t) ])));
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
         group "dashboard"
           [
             never_writes;
             info_pid;
             info_command;
             banner_whole;
             param_keys;
             cut_blank;
             pane_params;
             info_params;
             values_line;
             help_whole;
             strip_keys;
             hints_fit;
             hints_exact;
             ended_between;
           ];
         group "screen"
           [
             screens;
             sixteen_colours;
             reverse_video;
             filter_none;
             nav_count;
             hidden_filter;
             filter;
             stars;
             same_names;
             long_suffixes;
             banner_names;
             no_values_line;
             no_sample;
             no_others;
             paired;
             machine_strips;
             strips_whole;
             strips_scroll;
             all_orphans;
           ];
         group "charts"
           [
             stalled_end;
             shared_axis;
             reading_steps;
             resumed_readings;
             zoom_window_whole;
             zoom_units;
           ];
         group "states" [ states_test ];
         group "numbers" [ short_of_steps ];
         group "keys"
           [
             depth_test;
             depth_keys;
             shown_over;
             typed_q;
             six_keys;
             hint_test;
             y_per_metric;
             resumed_cursor;
           ];
         group "running"
           [
             run_none_newest;
             run_none_waits;
             unknown_run;
             held;
             kept_pane;
             ended_away;
             data_path_test;
           ];
       ])
