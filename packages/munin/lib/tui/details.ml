(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin

let label_w = 12
let field l v = [ (Paint.dim, Text.pad label_w l) ] @ v
let plain s = [ (Paint.plain, s) ]

(* Each section: its name and its rows. *)
let sections r ~stats h ~marked ~wide =
  let p = Run.provenance r in
  let value = Runs_screen.value_text in
  let step = Fact.latest_step stats in
  let run =
    [
      field "id" (plain (Run.id r));
      field "experiment" (plain (Run.experiment r));
      field "started" (plain (Text.date (Run.started_at r)));
      (match Run.ended_at r with
      | Some e -> field "ended" (plain (Text.date e))
      | None -> field "last write" (plain (Text.date (Run.updated_at r))));
      field "steps"
        (match (step, Run.steps r) with
        | Some s, Some t ->
            [ (Paint.plain, Text.count s); (Paint.dim, " of " ^ Text.count t) ]
        | Some s, None -> plain (Text.count s)
        | None, Some t -> [ (Paint.dim, "none of " ^ Text.count t) ]
        | None, None -> [ (Paint.dim, "none") ]);
    ]
    @
    match Run.parent r with
    | Some x -> [ field "parent" (plain x) ]
    | None -> []
  in
  let source =
    [
      field "command" (plain (String.concat " " p.command));
      field "in" (plain p.cwd);
      field "host"
        [
          (Paint.plain, Option.value p.hostname ~default:"—");
          (Paint.dim, Printf.sprintf " · pid %d" p.pid);
        ];
      field "commit"
        [
          (Paint.plain, Option.value (Fact.commit p) ~default:"—");
          ( Paint.dim,
            match p.git with
            | Some { dirty = true; _ } -> "  the work tree had changes"
            | _ -> "" );
        ];
      field "built"
        (plain
           (Printf.sprintf "OCaml %s · raven %s · %s %s" p.ocaml p.raven p.os
              p.arch));
    ]
  in
  (* A param a marked run gives another value, or none, shows it beside. *)
  let params =
    List.map
      (fun (k, x) ->
        field k
          ((Paint.plain, value x)
          :: List.concat_map
               (fun (l, mr) ->
                 match List.assoc_opt k (Run.params mr) with
                 | Some x' when x' = x -> []
                 | Some x' ->
                     [
                       (Paint.mark l, Printf.sprintf "  %c " l);
                       (Paint.plain, value x');
                     ]
                 | None ->
                     [
                       (Paint.mark l, Printf.sprintf "  %c " l); (Paint.dim, "—");
                     ])
               marked))
      (List.sort compare (Run.params r))
  in
  let metric_stats =
    List.filter (fun (_, (s : History.key_stats)) -> s.kind = `Metric) stats
  in
  let name_w =
    2
    + List.fold_left
        (fun a (k, (s : History.key_stats)) ->
          max a (Text.width k + Text.width (Fact.arrow s.goal) + 1))
        0 metric_stats
  in
  (* Below 100 columns a metric's row leaves out its sample count, which Zoom
     gives. *)
  let metrics =
    List.map
      (fun (k, (s : History.key_stats)) ->
        let arrow = match Fact.arrow s.goal with "" -> "" | a -> " " ^ a in
        [
          (Paint.dim, Text.pad name_w (k ^ arrow));
          (Paint.plain, Text.pad_left 9 (Text.value s.stats.last.value));
        ]
        @ [ (Paint.dim, Text.pad 11 (" at " ^ Text.count s.stats.last.step)) ]
        @ (match Option.bind s.goal (fun g -> Metric.best g s.stats) with
          | Some b ->
              [
                (Paint.dim, "best ");
                (Paint.plain, Text.value b.value);
                (Paint.dim, Text.pad 11 (" at " ^ Text.count b.step));
              ]
          | None -> [ (Paint.plain, String.make 25 ' ') ])
        @ (if wide then
             [
               ( Paint.dim,
                 Printf.sprintf "%s samples" (Text.count s.stats.count) );
             ]
           else [])
        @
        match s.stats.non_finite with
        | Some (n, f) ->
            [
              ( Paint.bad,
                Printf.sprintf "%s not finite from %s" (Text.count n)
                  (Text.count f.step) );
            ]
        | None -> [])
      metric_stats
  in
  let listed f = function
    | [] -> [ [ (Paint.dim, "—") ] ]
    | l -> List.map f l
  in
  let log =
    match h with
    | None ->
        [
          ("uses", [ [ (Paint.dim, "…") ] ]);
          ("files", [ [ (Paint.dim, "…") ] ]);
          ("resumes", [ [ (Paint.dim, "…") ] ]);
        ]
    | Some h -> (
        [
          ( "uses",
            listed
              (fun (n, v) -> plain (Printf.sprintf "%s v%d" n v))
              (History.uses h) );
          ( "files",
            listed
              (fun (f : History.file) ->
                [
                  (Paint.plain, f.key);
                  ( Paint.dim,
                    Printf.sprintf " at %s: %s" (Text.count f.step) f.path );
                ])
              (History.files h) );
          ( "resumes",
            listed
              (fun (x : History.resume) ->
                [
                  (Paint.plain, "at step " ^ Text.count x.step);
                  (Paint.dim, ", " ^ Text.date x.time);
                ])
              (History.resumes h) );
        ]
        @
        match History.damaged h with
        | 0 -> []
        | n ->
            [
              ( "damaged",
                [ [ (Paint.bad, Printf.sprintf "%d lines of the log" n) ] ] );
            ])
  in
  [
    ("run", run);
    ("source", source);
    ("params", if params = [] then [ [ (Paint.dim, "—") ] ] else params);
    ("metrics", if metrics = [] then [ [ (Paint.dim, "—") ] ] else metrics);
    ("notes", [ plain (Option.value (Run.notes r) ~default:"—") ]);
    ( "tags",
      [ plain (match Run.tags r with [] -> "—" | t -> String.concat ", " t) ] );
  ]
  @ log

let section_w = 10

let paint ~word ~scroll r ~stats h ~marked g ~width ~height =
  let cw = min (width - 4) 104 in
  let x = (width - cw) / 2 in
  Paint.spans g ~x ~y:0 ~w:cw
    (Paint.word word
    @ [
        (Paint.plain, "  ");
        (Paint.bold, Runs_screen.name r);
        (Paint.dim, "  " ^ Run.experiment r);
      ]);
  Paint.right g ~x ~y:0 ~w:cw [ (Paint.plain, "esc"); (Paint.dim, " close") ];
  let lines =
    List.concat_map
      (fun (title, rows) ->
        List.mapi (fun i row -> ((if i = 0 then title else ""), row)) rows
        @ [ ("", []) ])
      (sections r ~stats h ~marked ~wide:(width >= 100))
  in
  let room = height - 4 in
  let n = List.length lines in
  let scroll = max 0 (min scroll (n - room)) in
  List.iteri
    (fun i (title, row) ->
      let y = 2 + i - scroll in
      if y >= 2 && y < 2 + room then begin
        Paint.text ~style:Paint.dim g ~x ~y ~w:section_w title;
        Paint.spans g ~x:(x + section_w + 2) ~y ~w:(cw - section_w - 2) row
      end)
    lines;
  if n > room then
    Paint.right g ~x ~y:(height - 1) ~w:cw
      [
        ( Paint.dim,
          Printf.sprintf "%d–%d of %d" (scroll + 1) (min n (scroll + room)) n );
      ];
  Paint.hints g ~x ~y:(height - 1) ~w:(cw - 16)
    [ ("↑↓", "scroll"); ("[ ]", "run"); ("y", "copy id") ]
