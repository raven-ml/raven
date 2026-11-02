(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Help: the keys, with the screens they act on, and the words and glyphs of the
   screens. *)

type screen = [ `Runs | `Run | `Zoom | `Details ]

(* Each key, what it does, and the screens it acts on. *)
let keys : (string * string * screen list) list =
  [
    ("↑↓ j k", "move: run, metric, line", [ `Runs; `Run; `Zoom; `Details ]);
    ("←→ h l", "sideways: metric, step cursor", [ `Runs; `Run; `Zoom ]);
    ("g G", "first, last", [ `Runs; `Run; `Zoom; `Details ]);
    ("enter", "open: run, metric", [ `Runs; `Run ]);
    ("esc", "back; clear the filter", [ `Runs; `Run; `Zoom; `Details ]);
    ("space", "mark the run, at most 2", [ `Runs; `Run; `Zoom ]);
    ("/", "filter the list on screen", [ `Runs; `Run ]);
    ("o O", "order by next column, reverse", [ `Runs ]);
    ("i", "details of the run", [ `Runs; `Run; `Zoom ]);
    ("y", "copy the run's id", [ `Runs; `Run; `Zoom; `Details ]);
    ("?", "this help", [ `Runs; `Run; `Zoom; `Details ]);
    ("q ^C", "quit", [ `Runs; `Run; `Zoom; `Details ]);
    ("[ ]", "previous, next run", [ `Run; `Zoom; `Details ]);
    ("s", "smoothing: off, τ = span/100, span/30, span/10", [ `Run; `Zoom ]);
    ("x", "x axis: step, step metric, time", [ `Run; `Zoom ]);
    ("L", "log y, where every value shown is above 0", [ `Run; `Zoom ]);
    ( "+ - 0",
      "narrow, widen, reset the x window at the latest step",
      [ `Run; `Zoom ] );
    ("< >", "pan the x window", [ `Run; `Zoom ]);
  ]

let name = function
  | `Runs -> "Runs"
  | `Run -> "Run"
  | `Zoom -> "Zoom"
  | `Details -> "Details"

let lines ~screen ~width =
  let entry (k, d, _) = Text.pad 9 k ^ d in
  let short, long = List.partition (fun (_, d, _) -> Text.width d <= 30) keys in
  let half = width / 2 in
  let rec pairs = function
    | a :: b :: rest -> (Text.pad half (entry a) ^ entry b) :: pairs rest
    | [ a ] -> [ entry a ]
    | [] -> []
  in
  let not_here =
    List.filter_map
      (fun (k, _, on) -> if List.mem screen on then None else Some k)
      keys
  in
  [ Text.pad (width - 39) "KEYS" ^ "each key means the same on every screen" ]
  @ pairs short @ List.map entry long
  @ [
      "";
      "STATES   starting  no sample yet            writing   not probed, \
       arriving";
      "         running   no arrival seen yet      silent    not probed, none \
       seen";
      "         live      lock held, arriving      crashed   its writer left \
       no end";
      "         stalled   held, none for 3 gaps    finished  failed  killed  \
       its end";
      "";
      "MARKS    ━ focus   a b marked runs   × ↑ ↓ NaN, +inf, -inf   * best of \
       its";
      "         experiment   ▸ cursor   ! needs attention";
    ]
  @
  match not_here with
  | [] -> []
  | l -> [ Printf.sprintf "Not on %s: %s" (name screen) (String.concat " " l) ]

let paint ~screen g ~width ~height =
  Matrix_grid.clear g;
  let ls = lines ~screen ~width in
  List.iteri
    (fun i l -> if i < height - 2 then Paint.text g ~x:0 ~y:i ~w:width l)
    ls;
  Paint.text g ~x:0 ~y:(height - 1) ~w:width "esc close  q quit"
