(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module S = Mosaic.Shortcut

type group = Move | Look | Compare | Find | Panes
type mode = Screen | Filtering | Info | Help
type depth = [ `Charts | `Strips | `Zoom ]
type place = [ depth | `Bare ]

type row = {
  keys : (S.t * Model.action) list;
  label : string;
  alt : string;
  group : group;
  help : string;
  hints : (place * string) list;
  on_run : bool;
}

let mode (m : Model.t) =
  match m.layer with
  | Screen -> Screen
  | Filtering _ -> Filtering
  | Info _ -> Info
  | Help -> Help

let depth (m : Model.t) : depth =
  match m.depth with Charts -> `Charts | Strips _ -> `Strips | Zoom _ -> `Zoom

let row ?(hints = []) ?(alt = "") ?(on_run = false) group label help keys =
  { keys; label; alt; group; help; hints; on_run }
let everywhere w = [ (`Charts, w); (`Strips, w); (`Zoom, w); (`Bare, w) ]
let c = S.char

let screen =
  [
    row Move "↑↓" "run; metric in strips" ~alt:"k j" ~on_run:true
      ~hints:[ (`Charts, "run"); (`Strips, "metric") ]
      [ (S.up, Vertical (-1)); (S.down, Vertical 1); (c 'k', Vertical (-1)); (c 'j', Vertical 1) ];
    row Move "←→" "metric; zoom step, ⇧ ×10" ~alt:"h l" ~on_run:true
      ~hints:[ (`Charts, "metric"); (`Zoom, "step") ]
      [
        (S.left, Horizontal (-1));
        (S.right, Horizontal 1);
        (c 'h', Horizontal (-1));
        (c 'l', Horizontal 1);
        (S.key ~shift:true Left, Horizontal (-10));
        (S.key ~shift:true Right, Horizontal 10);
        (c 'H', Horizontal (-10));
        (c 'L', Horizontal 10);
      ];
    row Move "[ ]" "previous, next run" ~on_run:true
      ~hints:[ (`Strips, "run") ]
      [ (c '[', Run (-1)); (c ']', Run 1) ];
    row Move "g G" "first, last run" ~on_run:true [ (c 'g', First); (c 'G', Last) ];
    row Move "enter" "charts › strips › zoom" ~on_run:true
      ~hints:[ (`Charts, "strips"); (`Strips, "zoom") ]
      [ (S.enter, Deeper) ];
    row Move "esc"
      "back a depth, or clear"
      ~hints:[ (`Strips, "charts"); (`Zoom, "strips") ]
      [ (S.escape, Back) ];
    row Compare "space" "overlay the run, 4 at most" ~on_run:true
      ~hints:[ (`Charts, "overlay") ]
      [ (S.space, Overlay) ];
    row Compare "r" "the other runs, in grey" [ (c 'r', Others) ];
    row Compare "c" "clear overlays" [ (c 'c', Clear) ];
    row Look "s" "smoothing on, off" ~on_run:true ~hints:[ (`Zoom, "smooth") ]
      [ (c 's', Smooth) ];
    row Look "y" "other y scale, this metric" ~on_run:true [ (c 'y', Scale) ];
    row Look "+ −" "narrow, widen the steps" ~on_run:true
      ~hints:[ (`Zoom, "zoom") ]
      [ (c '+', Narrow); (c '-', Widen) ];
    row Find "/" "filter: name, tag, lr<0.05" ~hints:[ (`Charts, "filter") ] [ (c '/', Filter) ];
    row Find "o" "order: started, best, name" [ (c 'o', Order) ];
    row Find "e" "fold, open the experiment" ~on_run:true [ (c 'e', Fold) ];
    row Panes "i" "everything about the run" ~on_run:true [ (c 'i', Info) ];
    row Panes "1 3" "runs, run pane" [ (c '1', Toggle_runs); (c '3', Toggle_run) ];
    row Panes "?" "these keys" ~hints:(everywhere "keys") [ (c '?', Help) ];
    row Panes "q" "quit; Ctrl+C too" ~hints:[ (`Bare, "quit") ] [ (c 'q', Quit) ];
  ]

let filtering =
  [
    row Find "enter" "keep the filter" ~hints:(everywhere "keep the filter") [];
    row Find "esc" "clear the filter" ~hints:(everywhere "clear it") [ (S.escape, Clear_filter) ];
    row Move "↑↓" "the run" ~on_run:true ~hints:(everywhere "run") [ (S.up, Run (-1)); (S.down, Run 1) ];
  ]

let info =
  [
    row Move "↑↓" "scroll" ~hints:(everywhere "scroll") [];
    row Panes "esc" "close" ~hints:(everywhere "close") [ (S.escape, Close); (c 'i', Close) ];
    row Panes "q" "quit; Ctrl+C too" ~hints:(everywhere "quit") [ (c 'q', Quit) ];
  ]

let help =
  [
    row Panes "esc" "close" ~hints:(everywhere "close") [ (S.escape, Close); (c '?', Close) ];
    row Panes "q" "quit; Ctrl+C too" ~hints:(everywhere "quit") [ (c 'q', Quit) ];
  ]

let table = function
  | Screen -> screen
  | Filtering -> filtering
  | Info -> info
  | Help -> help

let bindings mode = List.concat_map (fun r -> r.keys) (table mode)

let opens_help r = List.exists (fun (_, a) -> a = Model.Help) r.keys

let hints m =
  let d : place = match Model.frame m with Panes _ -> (depth m :> place) | _ -> `Bare in
  let run = Model.focus m <> None in
  let shown =
    List.filter
      (fun r -> (run || not r.on_run) && List.mem_assoc d r.hints)
      (table (mode m))
  in
  let hint r = (r.label, List.assoc d r.hints) in
  let help, others = List.partition opens_help shown in
  (List.map hint others, Option.map hint (List.nth_opt help 0))

let group_name = function
  | Move -> "move"
  | Look -> "look"
  | Compare -> "compare"
  | Find -> "find"
  | Panes -> "panes"
