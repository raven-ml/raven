(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin
module Float_print = Munin_float_text.Float_print
module Th = Theme
open Mosaic
module F = Text.Fragment
module M = Model

let frag = Centre.frag
let line = Centre.line
let cell = Centre.cell
let split = Centre.split
let full = size_wh (pct 100) (pct 100)
let one = size_wh auto (px 1)
let shrinks = size_wh (px 0) (px 1)
let id (f : Facts.run) = Run.id f.run
let name (f : Facts.run) = Facts.name f.run

(* The store's directory, with [~] for the home directory. *)
let tilde path =
  match Sys.getenv_opt "HOME" with
  | Some h when h <> "" && String.starts_with ~prefix:h path ->
      "~" ^ String.sub path (String.length h) (String.length path - String.length h)
  | _ -> path

let root (m : M.t) = tilde (Store.root m.store)
let say th s = box ~padding:(padding_lrtb 2 2 1 0) ~flex_grow:1. [ line [ frag th Dim s ] ]

(* Top bar *)

let top_bar th (m : M.t) =
  let all = Data.runs m.data in
  let wide = match m.size with Some (w, _) -> w >= 100 | None -> false in
  let count p = List.length (List.filter p all) in
  let live = count (fun (f : Facts.run) -> Watch.live f.state) in
  let failed = count (fun (f : Facts.run) -> Watch.broken f.state) in
  let nan = count (fun (f : Facts.run) -> (not (Watch.broken f.state)) && f.non_finite <> None) in
  let experiments =
    List.length (List.sort_uniq String.compare (List.map (fun (f : Facts.run) -> Run.experiment f.run) all))
  in
  let hosts =
    List.sort_uniq compare (List.map (fun (f : Facts.run) -> (Run.provenance f.run).hostname) all)
  in
  let part tok s = [ frag th tok s; frag th Dim "   " ] in
  let right =
    List.concat
        ((if wide then
            [
              part Dim
                (Printf.sprintf "%d runs · %d experiment%s" (List.length all) experiments
                   (if experiments = 1 then "" else "s"));
            ]
          else [])
        @ (if live > 0 then [ part Live (Printf.sprintf "● %d live" live) ] else [])
        @ (if failed > 0 then [ part Fail (Printf.sprintf "✗ %d failed" failed) ] else [])
        @ (if nan > 0 then [ part Fail (Printf.sprintf "× %d NaN" nan) ] else [])
        @ match hosts with [ Some h ] when wide -> [ [ frag th Dim h ] ] | _ -> [])
  in
  (* Without runs, the store's directory follows the name. *)
  let rest = if all = [] then line [ frag th Dim (root m) ] else Centre.right right in
  box ~flex_direction:Row ~size:one ~flex_shrink:0. ?background:(Th.bg th Raise)
    ~padding:(padding_xy 1 0) ~gap:(gap 2)
    [ line ~grow:false [ frag ~bold:true th Mid "ᛗ munin" ]; rest ]

(* Below 100 columns, the cursor's run on a line of its own. *)
let nav_line th (m : M.t) =
  match M.focus m with
  | None -> empty
  | Some f ->
      let kept = M.kept m in
      let i = Option.value (List.find_index (fun k -> id k = id f) kept) ~default:0 in
      let progress =
        match (f.latest, Run.steps f.run) with
        | Some s, Some n when s < n -> [ frag th Dim (Numbers.count s ^ "/" ^ Numbers.count n) ]
        | _ -> []
      in
      box ~padding:(padding_xy 1 0) ~flex_shrink:0.
        [
          split
            [
              frag th Dim "‹ ";
              Centre.glyph th f;
              frag ~bold:true th Hi (" " ^ M.label m f);
              frag th Dim " ›";
              frag th Dim
                (Printf.sprintf "  %d of %d · %s" (i + 1) (List.length kept) (Run.experiment f.run));
            ]
            progress;
        ]

(* Runs pane *)

(* A line of the pane as plain data, so that an unchanged line costs a
   comparison. *)
type row =
  | Head of {
      experiment : string;
      count : int;
      folded : bool;
      here : int option;  (** The cursor's colour, when its run is folded here. *)
      right : (Th.token * string) option;
    }
  | Line of {
      here : bool;
      colour : int;
      drawn : bool;
      name : string;
      suffix : string;  (** The end of its id, when its name is shared. *)
      glyph : string;
      glyph_tok : Th.token;
      name_tok : Th.token;
      best : (Th.token * bool * string) option;
      star : bool;
    }

(* The cells of a run's name, its id's end included. *)
let name_w = 11

(* A row; one shown draws its run's trend. *)
let row ?trend (look, width, row) =
  let th = Th.make look in
  let sel here = if here then Some Theme.Cursor else None in
  let bar here colour = cell 1 (if here then [ F.text ~style:(Th.run th colour) "▌" ] else []) in
  match row with
  | Head h ->
      box ~flex_direction:Row ~size:one ?surface:(sel (h.here <> None))
        [
          bar (h.here <> None) (Option.value h.here ~default:0);
          line
            [
              frag th Dim (if h.folded then "▸ " else "▾ ");
              frag ~bold:(not h.folded) th (if h.folded then Mid else Fg) h.experiment;
              frag th Dim (Printf.sprintf "  %d" h.count);
            ];
          rich_text ~wrap:`None ~flex_shrink:0.
            (match h.right with
            | Some (tok, s) when width >= 26 -> [ frag th tok s; F.text " " ]
            | _ -> []);
        ]
  | Line r ->
      let swatch =
        if r.here || r.drawn then F.text ~style:(Th.run th r.colour) "■"
        else frag th Line2 "·"
      in
      let rest =
        if width < 26 then []
        else
          [
            canvas ~flex_grow:1. ~min_size:(size_wh (px 0) (px 1))
              ~on_size:(fun ~w ~h -> Some (M.Sized (Trend, w, h)))
              (match trend with Some p -> Centre.draw th p | None -> fun _ ~dt:_ -> ());
            Centre.rcell 8
              (match r.best with
              | Some (tok, bold, s) -> [ frag ~bold th tok s ]
              | None -> []);
            cell 2 (if r.star then [ F.text " "; frag th Warn "★" ] else []);
          ]
      in
      box ~flex_direction:Row ~size:one ?surface:(sel r.here)
        ([
           bar r.here r.colour;
           cell 2 [ F.text " "; swatch ];
           cell (name_w + 1)
             (let room = name_w - (if r.suffix = "" then 0 else 1 + Numbers.width r.suffix) in
              [ F.text " "; frag ~bold:r.here th r.name_tok (Numbers.cut room r.name) ]
              @ if r.suffix = "" then [] else [ frag th Dim (" " ^ r.suffix) ]);
           (* A blank after the name, so that a cut name's […] keeps off the
              glyph. *)
           cell 3 [ F.text " "; frag ~bold:true th r.glyph_tok r.glyph ];
         ]
        @ rest)

let row_view k = row k

(* Each line of the runs pane, with its run's id. *)
let rows (m : M.t) =
  let focus = Option.map id (M.focus m) in
  let overlaid o = List.exists (fun (_, x) -> x = o) (M.Overlays.runs m.overlays) in
  let colour o = Option.value (M.colour m o) ~default:0 in
  let lines = M.lines m in
  let suffixes = M.suffixes m in
  let stars = Hashtbl.create 8 in
  let star e runs =
    match Hashtbl.find_opt stars e with
    | Some s -> s
    | None ->
        let s = Option.bind (M.key m e) (fun key -> Runs.star runs ~key) in
        Hashtbl.replace stars e s;
        s
  in
  let runs_of = Hashtbl.create 8 in
  List.iter
    (function
      | Runs.Header { experiment; runs; _ } -> Hashtbl.replace runs_of experiment runs
      | Run _ -> ())
    lines;
  List.map
    (function
      | Runs.Header { experiment; runs; folded } ->
          let count p = List.length (List.filter p runs) in
          let failed = count (fun (f : Facts.run) -> Watch.broken f.state) in
          let live = count (fun (f : Facts.run) -> Watch.live f.state) in
          let here =
            match focus with
            | Some c when folded && List.exists (fun f -> id f = c) runs -> Some (colour c)
            | _ -> None
          in
          let right =
            if not folded then Option.map (fun k -> (Th.Dim, Numbers.elide 12 k)) (M.key m experiment)
            else if failed > 0 then Some (Fail, Printf.sprintf "✗ %d failed" failed)
            else if live > 0 then Some (Live, Printf.sprintf "● %d live" live)
            else Some (Dim, Printf.sprintf "✓ %d" (List.length runs))
          in
          (Head { experiment; count = List.length runs; folded; here; right }, None)
      | Run f ->
          let o = id f in
          let here = Some o = focus and drawn = overlaid o in
          let e = Run.experiment f.run in
          let name_tok : Th.token =
            if here then Hi
            else if drawn || Watch.live f.state || Watch.broken f.state || f.non_finite <> None then Fg
            else Mid
          in
          let starred =
            star e (Option.value (Hashtbl.find_opt runs_of e) ~default:[]) = Some o
          in
          let best =
            Option.bind (M.key m e) (fun key ->
                match Facts.best f key with
                | Some b ->
                    Some ((if starred then Th.Hi else Mid), starred, Numbers.metric key b.value)
                | None ->
                    Option.map
                      (fun (s : History.key_stats) ->
                        let v = s.stats.last.value in
                        ((if Float.is_finite v then Th.Mid else Fail), false, Numbers.metric key v))
                      (Facts.stats f key))
          in
          ( Line
              {
                here;
                colour = colour o;
                drawn;
                name = name f;
                suffix = Option.value (M.Smap.find_opt o suffixes) ~default:"";
                glyph = Facts.glyph f;
                glyph_tok = Centre.state_token f;
                name_tok;
                best;
                star = starred;
              },
            Some o ))
    lines

let filter_line th (m : M.t) =
  let shown = List.length (M.kept m) and all = List.length (Data.runs m.data) in
  let count = [ frag th Dim (Printf.sprintf "%d of %d" shown all) ] in
  match m.layer with
  | Filtering e ->
      box ~flex_direction:Row ~size:one ~flex_shrink:0. ?background:(Th.bg th Raise)
        ~padding:(padding_xy 1 0) ~margin:(margin_lrtb 1 1 0 0)
        [
          cell 2 [ frag ~bold:true th Warn "/ " ];
          input ~flex_grow:1. ~focused:true ~state:e
            ~on_change:(fun e -> M.Edited e)
            ~on_submit:(fun _ -> M.Submitted)
            ();
          rich_text ~wrap:`None ~flex_shrink:0. count;
        ]
  | _ when Runs.text m.filter <> "" ->
      box ~flex_direction:Row ~size:one ~flex_shrink:0. ?background:(Th.bg th Raise)
        ~padding:(padding_xy 1 0) ~margin:(margin_lrtb 1 1 0 0)
        [ split [ frag ~bold:true th Warn "/ "; frag th Hi (Runs.text m.filter) ] count ]
  | _ -> empty

let runs_pane th (m : M.t) w =
  let width = w in
  let trends = M.trends m in
  box ~flex_direction:Column ~size:(size_wh (px w) (pct 100)) ~flex_shrink:0. ~border:true
    ~border_sides:[ `Right ] ?border_color:(Th.bg th Line) ~padding:(padding_lrtb 0 0 1 0)
    [
      box ~padding:(padding_xy 1 0) ~flex_shrink:0.
        [ line [ frag th Dim "¹ "; frag ~bold:true th Mid "runs" ] ];
      filter_line th m;
      scroll_box ~flex_grow:1. ~min_size:(size_wh (px 0) (px 0)) ~margin:(margin_lrtb 0 0 1 0)
        ~show_scrollbars:false
        ~scroll:(Scroll.at m.top)
        ~on_scroll:(fun s -> M.Rows_scrolled s)
        ~on_size:(fun ~w ~h -> Some (M.Sized (Rows, w, h)))
        (List.map
           (fun (r, run) ->
             match Option.bind run (fun o -> List.assoc_opt o trends) with
             | Some c -> row ~trend:(M.plot m c) (m.look, width, r)
             | None -> memo ~key:(m.look, width, r) row_view)
           (rows m));
    ]

(* Run pane *)

(* A gauge one row high: [━━━━╸─────]. *)
let glyphs = { Progress_bar.filled = "━"; half = "╸"; empty = "─" }

let gauge th label frac value =
  box ~flex_direction:Row ~size:one ~flex_shrink:0. ~gap:(gap 1)
    [
      cell 4 [ frag th Dim label ];
      progress_bar ~flex_grow:1. ~value:(Float.min 1. (Float.max 0. frac))
        ?filled_color:(Th.bg th Mid) ?empty_color:(Th.bg th Line2) ~glyphs ();
      rich_text ~wrap:`None ~flex_shrink:0. [ frag th Fg value ];
    ]

(* A machine's reading's latest finite value. *)
let reading (f : Facts.run) k =
  match Facts.stats f k with
  | Some s when Float.is_finite s.stats.last.value -> Some s.stats.last.value
  | _ -> None

(* A memory's size in GiB, without a decimal when it has none to show. *)
let whole b =
  let s = Numbers.gib b in
  if String.ends_with ~suffix:".0" s then String.sub s 0 (String.length s - 2) else s

let device (f : Facts.run) =
  List.find_map
    (fun (k, (s : History.key_stats)) ->
      if s.kind = `Reading && String.starts_with ~prefix:"dev/" k then
        Option.map (fun j -> String.sub k 4 (j - 4)) (String.index_from_opt k 4 '/')
      else None)
    f.stats

let machine th (f : Facts.run) =
  let dk d k = "dev/" ^ d ^ "/" ^ k in
  let dev = device f in
  let gpu =
    match dev with
    | None -> []
    | Some d -> (
        (match reading f (dk d "busy_pct") with
        | Some p -> [ gauge th "GPU" (p /. 100.) (Numbers.percent p) ]
        | None -> [])
        @
        match (reading f (dk d "used_bytes"), reading f (dk d "memory_bytes")) with
        | Some u, Some t when t > 0. -> [ gauge th "VRAM" (u /. t) (Numbers.gib u ^ "/" ^ whole t ^ "G") ]
        | _ -> [])
  in
  let host =
    (match reading f "sys/cpu_busy_pct" with
    | Some p -> [ gauge th "CPU" (p /. 100.) (Numbers.percent p) ]
    | None -> [])
    @
    match (reading f "sys/mem_used_bytes", reading f "sys/mem_total_bytes") with
    | Some u, Some t when t > 0. -> [ gauge th "RAM" (u /. t) (Numbers.gib u ^ "/" ^ whole t ^ "G") ]
    | _ -> []
  in
  let heat =
    match dev with
    | None -> []
    | Some d -> (
        match (reading f (dk d "temperature_c"), reading f (dk d "power_w")) with
        | None, None -> []
        | temp, power ->
            [
              line
                ((match temp with
                 | Some c ->
                     [ frag ~bold:true th (if c > 80. then Warn else Fg) (Float_print.f 0 c ^ "°C") ]
                 | None -> [])
                @
                match power with
                | Some p -> [ frag th Dim ("  " ^ Float_print.f 0 p ^ " W") ]
                | None -> []);
            ])
  in
  match gpu @ host with
  | [] -> []
  | rows -> (Centre.section th "machine" :: rows) @ heat @ [ line [] ]

(* The share of the run pane's width, in percent, that param keys may take,
   leaving the rest to the values. *)
let key_share = 50

let params th (m : M.t) (f : Facts.run) =
  let all = Run.params f.run in
  if all = [] then []
  else
    let vary =
      match Data.experiment m.data (Run.experiment f.run) with
      | Some e -> e.varying
      | None -> []
    in
    let varies, others = List.partition (fun (k, _) -> List.mem k vary) all in
    let shown = varies @ List.filteri (fun i _ -> i < 3) others in
    let shown = List.filteri (fun i _ -> i < 6) shown in
    let longest = List.fold_left (fun w (k, _) -> max w (Numbers.width k)) 0 shown in
    let param (k, v) =
      let varies = List.mem k vary in
      (* A number is never cut: it prints by the value rule, and the key
         and [varies] give way to it. *)
      let number = match v with `Int _ | `Float _ -> true | _ -> false in
      box ~flex_direction:Row ~size:one ~flex_shrink:0. ~gap:(gap 1)
        [
          rich_text ~wrap:`None ~truncate:true ~size:(size_wh (px longest) (px 1))
            ~max_size:(size_wh (pct key_share) (px 1)) ~flex_shrink:1. ~min_size:shrinks
            [ frag th Dim k ];
          rich_text ~wrap:`None ~truncate:true ~flex_grow:1.
            ~flex_shrink:(if number then 0. else 1.) ~min_size:shrinks
            [ frag ~bold:varies th (if varies then Hi else Fg) (Facts.printed v) ];
          (if varies then
             rich_text ~wrap:`None ~truncate:true ~min_size:shrinks
               [ frag th Dim "varies" ]
           else empty);
        ]
    in
    let more = List.length all - List.length shown in
    (Centre.section th "params" :: List.map param shown)
    @ [ (if more > 0 then line [ frag th Dim (Printf.sprintf "+%d more   i: all" more) ] else empty); line [] ]

let source th (f : Facts.run) =
  let p = Run.provenance f.run in
  let commit =
    match Facts.commit p with
    | Some c ->
        line
          ([ frag th Fg c ]
          @ match p.git with Some { dirty = true; _ } -> [ frag th Warn " dirty" ] | _ -> [])
    | None -> line [ frag th Dim "no commit" ]
  in
  let command =
    match p.command with
    | prog :: args -> [ line [ frag th Dim (String.concat " " (Filename.basename prog :: args)) ] ]
    | [] -> []
  in
  (Centre.section th "source" :: commit :: command)

let state th (m : M.t) (f : Facts.run) =
  let now = Data.time m.data in
  let ago t = Numbers.span (now -. t) ^ " ago" in
  let at = Numbers.count (Option.value f.latest ~default:0) in
  (* The name gives way to the state word. *)
  let first =
    box ~flex_direction:Row ~size:one ~flex_shrink:0.
      [
        line ~grow:false [ Centre.glyph th f; frag ~bold:true th Hi (" " ^ M.label m f) ];
        rich_text ~wrap:`None ~flex_shrink:0.
          [ frag th (Centre.state_token f) ("  " ^ Watch.word f.state) ];
      ]
  in
  let second =
    let dim s = [ frag th Dim s ] in
    Centre.groups
      (if Watch.live f.state then
         match f.last_arrival with
         | Some a -> [ dim ("last sample " ^ ago a) ]
         | None -> [ dim ("last write " ^ ago (Run.updated_at f.run)) ]
       else if Watch.broken f.state || f.state = Killed then
         [ dim (Numbers.span f.active); dim ("· stopped at " ^ at) ]
       else [ dim (Numbers.span f.active); dim ("· " ^ at ^ " steps") ])
  in
  let progress =
    match (f.latest, Run.steps f.run) with
    | Some s, Some n when n > 0 ->
        let tok : Th.token = if Watch.broken f.state then Fail else Live in
        let frac = Float.of_int s /. Float.of_int n in
        let rate =
          match f.rate with
          | Some x when Watch.live f.state && s < n && x > 0. ->
              let left = Float.of_int (n - s) /. x in
              [
                line
                  [
                    frag ~bold:true th Fg (Numbers.span left);
                    frag th Dim (" left · ends " ^ Numbers.clock ~now (now +. left));
                  ];
                line
                  [
                    frag ~bold:true th Fg (Float_print.g ~precision:2 x);
                    frag th Dim (" steps/s · " ^ Numbers.span f.active ^ " in");
                  ];
              ]
          | _ -> []
        in
        [
          box ~flex_direction:Row ~size:one ~flex_shrink:0. ~gap:(gap 1)
            [
              progress_bar ~flex_grow:1. ~value:(Float.min 1. frac) ?filled_color:(Th.bg th tok)
                ?empty_color:(Th.bg th Line2) ~glyphs ();
              rich_text ~wrap:`None ~flex_shrink:0. [ frag th Fg (Numbers.percent_of s n) ];
            ];
          line [ frag ~bold:true th Fg (Numbers.count s); frag th Dim (" of " ^ Numbers.count n ^ " steps") ];
        ]
        @ rate @ [ line [] ]
    | _ -> []
  in
  [ first; second; line [] ] @ progress

let run_pane th (m : M.t) (f : Facts.run) =
  [ line [ frag th Dim "³ "; frag ~bold:true th Mid "run" ]; line [] ]
  @ state th m f @ params th m f
  @ (if Watch.live f.state then machine th f else [])
  @ source th f

let side_pane th (m : M.t) side =
  match (side, M.focus m) with
  | M.Off, _ | _, None -> empty
  | Beside w, Some f ->
      box ~flex_direction:Column ~size:(size_wh (px w) (pct 100)) ~flex_shrink:0. ~border:true
        ~border_sides:[ `Left ] ?border_color:(Th.bg th Line) ~padding:(padding_lrtb 1 1 1 0)
        ~overflow:{ x = Hidden; y = Hidden }
        (run_pane th m f)
  | Over w, Some f ->
      box ~flex_direction:Column ~position:Absolute ~inset:(inset_lrtb (-1) 0 0 0)
        ~size:(size_wh (px w) (pct 100)) ~z_index:1 ~background:(Th.ground th) ~border:true
        ~border_sides:[ `Left ] ?border_color:(Th.bg th Line) ~padding:(padding_lrtb 1 1 1 0)
        ~overflow:{ x = Hidden; y = Hidden }
        (run_pane th m f)

(* Info *)

let label_w = 12

(* A fact of the info, whole: it wraps at its words. *)
let para frags =
  rich_text ~wrap:`Word ~flex_grow:1. ~flex_shrink:1. ~min_size:(size_wh (px 0) auto) frags

let info_sections th (m : M.t) (f : Facts.run) =
  let field l x =
    box ~flex_direction:Row ~flex_shrink:0. [ Centre.label label_w [ frag th Dim l ]; para x ]
  in
  let line = para in
  let plain s = [ frag th Fg s ] in
  let r = f.run in
  let p = Run.provenance r in
  let run =
    [
      field "id" (plain (Run.id r));
      field "experiment" (plain (Run.experiment r));
      field "started" (plain (Numbers.date (Run.started_at r)));
      (match Run.ended_at r with
      | Some e -> field "ended" (plain (Numbers.date e))
      | None -> field "last write" (plain (Numbers.date (Run.updated_at r))));
      field "steps"
        (match (f.latest, Run.steps r) with
        | Some s, Some n -> [ frag th Fg (Numbers.count s); frag th Dim (" of " ^ Numbers.count n) ]
        | Some s, None -> plain (Numbers.count s)
        | None, Some n -> [ frag th Dim ("none of " ^ Numbers.count n) ]
        | None, None -> [ frag th Dim "none" ]);
    ]
    @ (match Run.parent r with Some x -> [ field "parent" (plain x) ] | None -> [])
    @ match Run.status r with `Failed e -> [ field "error" [ frag th Fail e ] ] | _ -> []
  in
  let source =
    [
      (* An argument wraps whole, so that a command copied from the screen
         keeps its arguments. *)
      box ~flex_direction:Row ~flex_shrink:0.
        [
          Centre.label label_w [ frag th Dim "command" ];
          Centre.groups (List.map (fun a -> [ frag th Fg a ]) p.command);
        ];
      field "in" (plain p.cwd);
      box ~flex_direction:Row ~flex_shrink:0.
        [
          Centre.label label_w [ frag th Dim "host" ];
          Centre.groups
            [
              [ frag th Fg (Option.value p.hostname ~default:"—") ];
              [ frag th Dim (Printf.sprintf "· pid %d" p.pid) ];
            ];
        ];
      field "commit"
        ([ frag th Fg (Option.value (Facts.commit p) ~default:"—") ]
        @
        match p.git with
        | Some { dirty = true; _ } -> [ frag th Warn "  dirty: the work tree had changes" ]
        | _ -> []);
      field "built" (plain (Printf.sprintf "OCaml %s · raven %s · %s %s" p.ocaml p.raven p.os p.arch));
    ]
  in
  (* The keys take the longest's width and a blank, up to a share of the
     row. *)
  let params =
    let all = List.sort compare (Run.params r) in
    let longest = List.fold_left (fun w (k, _) -> max w (Numbers.width k)) 0 all in
    let key k =
      rich_text ~wrap:`None ~truncate:true
        ~size:(size_wh (px (max (label_w - 1) longest)) (px 1))
        ~max_size:(size_wh (pct key_share) (px 1)) ~flex_shrink:1. ~min_size:shrinks
        ~margin:(margin_lrtb 0 1 0 0) [ frag th Dim k ]
    in
    List.map
      (fun (k, v) -> box ~flex_direction:Row ~flex_shrink:0. [ key k; para (plain (Facts.value v)) ])
      all
  in
  (* The values wrap under each other, right of the key. *)
  let metric (k, (s : History.key_stats)) =
    let arrow = match Centre.arrow s.goal with "" -> "" | a -> " " ^ a in
    box ~flex_direction:Row ~flex_shrink:0.
      [
        rich_text ~wrap:`None ~flex_shrink:0. [ frag th Dim (k ^ arrow ^ "  ") ];
        Centre.groups
      ([
         [
           frag th Fg (Numbers.metric k s.stats.last.value);
           frag th Dim (" at " ^ Numbers.count s.stats.last.step ^ "  ");
         ];
       ]
      @ (match Option.bind s.goal (fun g -> Metric.best g s.stats) with
        | Some b ->
            [
              [
                frag th Dim "best ";
                frag th Fg (Numbers.metric k b.value);
                frag th Dim (" at " ^ Numbers.count b.step ^ "  ");
              ];
            ]
        | None -> [])
      @
      match s.stats.non_finite with
      | Some (n, x) ->
          [ [ frag th Fail (Printf.sprintf "%s not finite from %s" (Numbers.count n) (Numbers.count x.step)) ] ]
      | None -> []);
      ]
  in
  let metrics = List.map metric (List.filter (fun (_, (s : History.key_stats)) -> s.kind = `Metric) f.stats) in
  let none = [ line [ frag th Dim "—" ] ] in
  let listed g = function [] -> none | l -> List.map g l in
  let log =
    match Data.history m.data (id f) with
    | None -> [ ("uses", [ line [ frag th Dim "…" ] ]) ]
    | Some h ->
        [
          ("uses", listed (fun (n, v) -> line (plain (Printf.sprintf "%s v%d" n v))) (History.uses h));
          ( "files",
            listed
              (fun (x : History.file) ->
                line [ frag th Fg x.key; frag th Dim (Printf.sprintf " at %s: %s" (Numbers.count x.step) x.path) ])
              (History.files h) );
          ( "resumes",
            listed
              (fun (x : History.resume) ->
                line [ frag th Fg ("at step " ^ Numbers.count x.step); frag th Dim (", " ^ Numbers.date x.time) ])
              (History.resumes h) );
        ]
        @
        match History.damaged h with
        | 0 -> []
        | n -> [ ("damaged", [ line [ frag th Fail (Printf.sprintf "%d lines of the log" n) ] ]) ]
  in
  [
    ("run", run);
    ("source", source);
    ("params", if params = [] then none else params);
    ("metrics", if metrics = [] then none else metrics);
    ("notes", [ line (plain (Option.value (Run.notes r) ~default:"—")) ]);
    ("tags", [ line (plain (match Run.tags r with [] -> "—" | l -> String.concat ", " l)) ]);
  ]
  @ log

let info th (m : M.t) scroll (f : Facts.run) =
  let section (title, rows) =
    box ~flex_direction:Row ~flex_shrink:0. ~margin:(margin_lrtb 0 0 0 1)
      [
        cell 10 [ frag th Mid title ];
        box ~flex_direction:Column ~flex_grow:1. ~min_size:(size_wh (px 0) auto)
          ~margin:(margin_lrtb 2 0 0 0) rows;
      ]
  in
  box ~flex_direction:Column ~position:Absolute ~inset:(inset 0) ~z_index:1
    ~background:(Th.ground th) ~padding:(padding_lrtb 1 1 1 0)
    [
      split
        [
          Centre.glyph th f;
          frag ~bold:true th Hi (" " ^ M.label m f);
          frag th Dim ("  " ^ Run.experiment f.run ^ "  everything");
        ]
        [ frag ~bold:true th Hi "esc"; frag th Dim " close" ];
      scroll_box ~flex_grow:1. ~min_size:(size_wh (px 0) (px 0)) ~margin:(margin_lrtb 0 0 1 0)
        ~focused:true ~scroll
        ~on_scroll:(fun s -> M.Info_scrolled s)
        (List.map section (info_sections th m f));
    ]

(* Status bar *)

(* The blank columns between the bar's items, and each side of the bar. *)
let hint_gap = 3
let bar_padding = 1

let status_bar th (m : M.t) =
  let bar items =
    box ~flex_direction:Row ~size:one ~flex_shrink:0. ?background:(Th.bg th Raise)
      ~padding:(padding_xy bar_padding 0) ~overflow:{ x = Hidden; y = Hidden } items
  in
  match m.notice with
  | Some n -> bar [ line [ frag ~bold:true th Fail n ] ]
  | None ->
      let sep = " › " in
      let crumbs =
        match (m.layer, M.frame m) with
        | Screen, Panes _ -> [ (`Charts, "charts"); (`Strips, "strips"); (`Zoom, "zoom") ]
        | _ -> []
      in
      let crumbs_w =
        List.fold_left (fun w (_, word) -> w + Numbers.width word) 0 crumbs
        + (Numbers.width sep * max 0 (List.length crumbs - 1))
      in
      let crumb_frags =
        let on = Keys.depth m in
        List.concat
          (List.mapi
             (fun i (d, word) ->
               (if i = 0 then [] else [ frag th Line2 sep ])
               @ [ frag ~bold:(d = on) th (if d = on then Hi else Dim) word ])
             crumbs)
      in
      let hint (k, w) = [ frag ~bold:true th Hi k; frag th Dim (" " ^ w) ] in
      let hint_w (k, w) = Numbers.width k + 1 + Numbers.width w in
      let others, help = Keys.hints m in
      (* The items' widths and the gaps between them: the crumbs and the
         help key always, then the hints that fit, whole, leaving from the
         last. *)
      let fixed = (if crumbs = [] then [] else [ crumbs_w ]) @ Option.to_list (Option.map hint_w help) in
      let used = List.fold_left ( + ) 0 fixed + (hint_gap * max 0 (List.length fixed - 1)) in
      let room =
        match m.size with Some (w, _) -> w - (2 * bar_padding) - used | None -> 0
      in
      let rec fit room ~alone = function
        | h :: rest ->
            let cost = hint_w h + if alone then 0 else hint_gap in
            if cost > room then [] else h :: fit (room - cost) ~alone:false rest
        | [] -> []
      in
      let items =
        (if crumbs = [] then [] else [ crumb_frags ])
        @ List.map hint (fit room ~alone:(fixed = []) others)
        @ Option.to_list (Option.map hint help)
      in
      let gap = F.text (String.make hint_gap ' ') in
      bar [ line (List.concat (List.mapi (fun i x -> if i = 0 then x else gap :: x) items)) ]

(* Help *)

(* The keys in two columns, so that they fit 24 rows. *)
let help th =
  let rows = Keys.table Screen in
  let group g =
    let mine = List.filter (fun (r : Keys.row) -> r.group = g) rows in
    box ~flex_direction:Column ~flex_shrink:0. ~margin:(margin_lrtb 0 0 0 1)
      (line [ frag ~bold:true th Mid (Keys.group_name g) ]
      :: List.map
           (fun (r : Keys.row) ->
             (* The words wrap rather than cut, on a terminal narrower than
                help. *)
             box ~flex_direction:Row ~flex_shrink:0.
               [
                 cell 9
                   [ frag ~bold:true th Hi r.label; frag th Hi (if r.alt = "" then "" else " " ^ r.alt) ];
                 rich_text ~wrap:`Word ~flex_grow:1. ~flex_shrink:1. ~min_size:(size_wh (px 0) auto)
                   [ frag th Fg r.help ];
               ])
           mine)
  in
  let column groups =
    box ~flex_direction:Column ~flex_grow:1. ~flex_basis:(px 0) ~min_size:(size_wh (px 0) auto)
      (List.map group groups)
  in
  modal ~on_close:(M.Key Close)
    (box ~flex_direction:Column ~size:(size_wh (px 84) auto) ~max_size:(size_wh (pct 100) (pct 100))
       ~border:true ~border_style:Border.rounded ?border_color:(Th.bg th Line2)
       ?background:(Th.bg th Raise) ~title:"keys" ~title_style:(Th.fg ~bold:true th Hi) ~padding:(padding_xy 2 1)
       [
         box ~flex_direction:Row ~gap:(gap 4) ~flex_shrink:0.
           [ column [ Keys.Move; Look; Compare ]; column [ Find; Panes ] ];
         line [ frag th Dim "munin reads the store once a second and never writes to it" ];
       ])

(* The other screens *)

let empty_store th (m : M.t) =
  let code =
    [
      "let store = Munin.Store.open_ () in";
      "Munin.Session.within";
      "  (Munin.Session.start ~store ~experiment:\"mnist\" ())";
      "@@ fun s -> ...";
    ]
  in
  box ~flex_grow:1. ~justify_content:Center ~align_items:Center
    [
      box ~flex_direction:Column ~size:(size_wh (px 60) auto) ~max_size:(size_wh (pct 100) auto)
        [
          line [ frag ~bold:true th Hi "No runs yet" ];
          line [ frag th Dim ("in " ^ root m) ];
          line [];
          line [ frag th Mid "A run shows up here as soon as a program starts one:" ];
          line [];
          box ~flex_direction:Column ?background:(Th.bg th Raise) ~padding:(padding_xy 2 1)
            (List.map (fun c -> line [ frag th Fg c ]) code);
          line [];
          line [ frag th Dim "Another store:  munin --root DIR" ];
        ];
    ]

(* Too small: one line per live run, then the size it needs. *)
let tiny th (m : M.t) =
  let h = match m.size with Some (_, h) -> h | None -> 0 in
  let live = List.filter (fun (f : Facts.run) -> Watch.live f.state) (Data.runs m.data) in
  let run (f : Facts.run) =
    rich_text ~wrap:`None ~truncate:true
      [
        Centre.glyph th f;
        frag ~bold:true th Hi (" " ^ M.label m f);
        frag th Dim (match f.latest with Some s -> "  " ^ Numbers.count s | None -> "");
      ]
  in
  box ~flex_direction:Column ~size:full ~justify_content:Center ~align_items:Center
    (List.map run (List.filteri (fun i _ -> i < h - 2) live)
    @ [ rich_text ~wrap:`None ~truncate:true [ frag th Dim "make the terminal larger" ] ])

(* The screen *)

let error_line th (m : M.t) =
  match Data.errors m.data with
  | [] -> empty
  | (id, e) :: _ ->
      let what = if id = "" then "the listing: " ^ e else id ^ ": " ^ e in
      box ~padding:(padding_xy 1 0) ~flex_shrink:0. [ line [ frag th Dim what ] ]

(* The runs pane stays when the filter keeps no run: it holds the filter
   being typed, which Enter keeps. *)
let body th (m : M.t) (p : M.panes) =
  let runs = match p.runs with Some w -> runs_pane th m w | None -> empty in
  let row = box ~flex_direction:Row ~flex_grow:1. ~min_size:(size_wh (px 0) (px 0)) in
  match M.focus m with
  | None -> row [ runs; say th "no run matches the filter" ]
  | Some f ->
      let over =
        match (m.layer, p.run) with
        | Info s, _ -> info th m s f
        | _, Over _ -> side_pane th m p.run
        | _ -> empty
      in
      row
        [
          runs;
          box ~flex_direction:Column ~flex_grow:1. ~position:Relative ~min_size:(size_wh (px 0) (px 0))
            [ Centre.view th m p f; over ];
          (match p.run with Beside _ -> side_pane th m p.run | Over _ | Off -> empty);
        ]

let view (m : M.t) =
  let th = Th.make m.look in
  (* The cursor's rows are on the theme's Cursor surface, in munin's
     selection: a background, or reverse video without colour. *)
  let screen inner =
    themed (Theme.with_style Cursor (Th.sel th))
      (box ~flex_direction:Column ~size:full
         ([ top_bar th m ] @ inner @ [ error_line th m; status_bar th m ]
         @ match m.layer with Help -> [ help th ] | _ -> []))
  in
  match M.frame m with
  | Unsized -> empty
  | Tiny -> tiny th m
  | Reading -> screen [ say th "reading the store…" ]
  | Waiting_run -> screen [ say th ("waiting for a run in " ^ root m) ]
  | Empty -> screen [ empty_store th m ]
  | Panes p -> screen ((if p.nav then [ nav_line th m ] else []) @ [ body th m p ])
