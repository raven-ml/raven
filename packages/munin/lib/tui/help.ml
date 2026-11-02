(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module G = Matrix_grid

let everywhere =
  [
    ("q ^C", "quit");
    ("?", "these keys");
    ("esc", "back, close, clear");
    ("i", "every fact about the run");
    ("y", "copy the run's id");
    ("space", "mark the run: a, then b");
  ]

let keys = function
  | `Runs ->
      ( "Runs",
        [
          ("↑↓ j k", "run");
          ("g G", "first, last run");
          ("←→ h l", "the metric shown");
          ("enter", "open the run");
          ("/", "filter: name, word, key=value");
          ("o O", "order, reverse");
        ] )
  | `Run ->
      ( "Run",
        [
          ("↑↓ j k", "metric");
          ("g G", "first, last metric");
          ("enter", "zoom the chart");
          ("[ ]", "previous, next run");
          ("/", "filter the metrics");
          ("s", "smoothing: off, three τ");
          ("x", "x: step, step metric, time");
          ("L", "log y");
          ("+ - 0", "narrow, widen, reset x");
          ("< >", "pan x");
        ] )
  | `Zoom ->
      ( "Zoom",
        [
          ("←→ h l", "step cursor, a sample");
          ("↑↓ j k", "metric");
          ("[ ]", "previous, next run");
          ("s x L", "smoothing, x, log y");
          ("+ - 0 < >", "x window");
        ] )
  | `Details ->
      ( "Details",
        [
          ("↑↓ j k", "scroll");
          ("g G", "top, bottom");
          ("[ ]", "previous, next run");
        ] )

let words =
  [
    ("● live", "samples arrive");
    ("○ running", "no sample seen yet");
    ("○ starting", "no sample logged yet");
    ("◌ stalled", "no sample for three gaps");
    ("● writing", "lock not probed, arriving");
    ("◌ silent", "lock not probed, quiet");
    ("✓ finished", "ended well");
    ("✗ failed", "ended with an error");
    ("✗ crashed", "writer gone without an end");
    ("■ killed", "ended by a signal");
    ("× ↑ ↓", "NaN, +inf, -inf at a step");
    ("*", "best of its experiment");
    ("━ a b", "the focus, marked runs");
    ("•", "the latest sample");
    ("!", "what needs attention");
  ]

let col_w = 46
let key_w = 14

let lines groups =
  List.concat_map
    (fun (title, l) ->
      [ `Title title ] @ List.map (fun r -> `Row r) l @ [ `Blank ])
    groups

let paint ~screen ~light ~scroll g ~width ~height =
  let veil =
    if light then Ansi.Color.of_rgba 255 255 255 170
    else Ansi.Color.of_rgba 0 0 0 150
  in
  G.fill_rect g ~x:0 ~y:0 ~width ~height ~color:veil;
  let left_groups = [ ("Everywhere", everywhere); keys screen ] in
  let right_groups = [ ("Words", words) ] in
  let two = width >= (2 * col_w) + 8 in
  let left, right =
    if two then (lines left_groups, lines right_groups)
    else (lines (left_groups @ right_groups), [])
  in
  let need = max (List.length left) (List.length right) + 2 in
  let bw = if two then (2 * col_w) + 6 else min (width - 2) (col_w + 4) in
  let bh = min (height - 2) (need + 1) in
  let bx = (width - bw) / 2 and by = max 0 ((height - 1 - bh) / 2) in
  G.clear_rect g ~x:bx ~y:by ~width:bw ~height:bh;
  G.fill_rect g ~x:bx ~y:by ~width:bw ~height:bh
    ~color:
      (if light then Ansi.Color.of_rgb 0xf6 0xf6 0xf6
       else Ansi.Color.of_rgb 0x26 0x26 0x26);
  Paint.text ~style:Paint.bold g ~x:(bx + 2) ~y:by ~w:(bw - 4) "keys";
  Paint.right g ~x:(bx + 2) ~y:by ~w:(bw - 4)
    [ (Paint.plain, "esc"); (Paint.dim, " close") ];
  let room = bh - 3 in
  (* A column that does not fit scrolls. *)
  let column x ls =
    let at = max 0 (min scroll (List.length ls - room)) in
    List.iteri
      (fun i line ->
        let y = by + 2 + i - at in
        if y >= by + 2 && y < by + 2 + room then
          match line with
          | `Title t -> Paint.text ~style:Paint.bold g ~x ~y ~w:col_w t
          | `Row (k, a) ->
              (* A state word in its own style, a key in the accent. *)
              let style =
                match String.index_opt k ' ' with
                | Some i
                  when snd
                         (Paint.state
                            (String.sub k (i + 1) (String.length k - i - 1)))
                       != Paint.plain ->
                    snd
                      (Paint.state
                         (String.sub k (i + 1) (String.length k - i - 1)))
                | _ -> Paint.accent
              in
              Paint.text ~style g ~x ~y ~w:key_w k;
              Paint.text ~style:Paint.dim g ~x:(x + key_w) ~y ~w:(col_w - key_w)
                a
          | `Blank -> ())
      ls;
    if List.length ls > room then
      Paint.text ~style:Paint.dim g ~x
        ~y:(by + bh - 1)
        ~w:col_w
        (if at + room < List.length ls then "↓ more" else "↑ more")
  in
  column (bx + 2) left;
  if two then column (bx + 4 + col_w) right
