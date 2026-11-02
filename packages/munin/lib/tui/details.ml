(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin

let label l = Text.pad 11 l

let lines ~word r ~stats h ~marked =
  let p = Run.provenance r in
  let name = Runs_screen.name r in
  let value = Runs_screen.value_text in
  let step = Fact.latest_step stats in
  let ended =
    match Run.ended_at r with
    | Some e -> [ label "" ^ "ended " ^ Text.date e ]
    | None -> []
  in
  let params =
    List.mapi
      (fun i (k, x) ->
        let others =
          List.filter_map
            (fun (l, mr) ->
              match List.assoc_opt k (Run.params mr) with
              | Some x' when x' <> x ->
                  Some (Printf.sprintf "%c %s" l (value x'))
              | None -> Some (Printf.sprintf "%c —" l)
              | _ -> None)
            marked
        in
        (if i = 0 then Text.pad 9 "PARAMS" else String.make 9 ' ')
        ^ (if others = [] then "  " else "≠ ")
        ^ Printf.sprintf "%-9s %-12s %s" k (value x) (String.concat "  " others))
      (List.sort compare (Run.params r))
  in
  let metrics =
    List.mapi
      (fun i (k, (st : History.key_stats)) ->
        let best =
          match Option.bind st.goal (fun g -> Metric.best g st.stats) with
          | Some b ->
              Printf.sprintf "best %s @%s" (Text.value b.value)
                (Text.short b.step)
          | None -> ""
        in
        (if i = 0 then label "METRICS" else label "")
        ^ Printf.sprintf "%s %10s  %-22s %4d samples%s"
            (Text.pad 16 (Text.elide 16 k))
            (Text.value st.stats.last.value)
            best st.stats.count
            (match st.stats.non_finite with
            | Some (n, f) ->
                Printf.sprintf ", %d not finite from %s" n (Text.count f.step)
            | None -> ""))
      (List.filter (fun (_, (s : History.key_stats)) -> s.kind = `Metric) stats)
  in
  let resumes, damaged, uses, files =
    match h with
    | None -> ("—", 0, "—", [])
    | Some h ->
        ( (match History.resumes h with
          | [] -> "no resume"
          | l ->
              String.concat ", "
                (List.map
                   (fun (x : History.resume) ->
                     Printf.sprintf "resumed at %s, %s" (Text.count x.step)
                       (Text.date x.time))
                   l)),
          History.damaged h,
          (match History.uses h with
          | [] -> "—"
          | l ->
              String.concat ", "
                (List.map (fun (n, v) -> Printf.sprintf "%s v%d" n v) l)),
          List.map
            (fun (f : History.file) ->
              Printf.sprintf "%s at %s: %s" f.key (Text.count f.step) f.path)
            (History.files h) )
  in
  [
    Printf.sprintf "%s  %s  %s  %s" name word (Run.experiment r) (Run.id r);
    "";
    label "RUN"
    ^ Printf.sprintf "started %s%s"
        (Text.date (Run.started_at r))
        (match (step, Run.steps r) with
        | Some s, Some t ->
            Printf.sprintf ", step %s of %s" (Text.count s) (Text.count t)
        | Some s, None -> ", step " ^ Text.count s
        | None, _ -> "");
  ]
  @ ended
  @ [
      label ""
      ^ Printf.sprintf "tags %s · parent %s · %s · %d damaged lines"
          (match Run.tags r with [] -> "—" | t -> String.concat ", " t)
          (Option.value (Run.parent r) ~default:"—")
          resumes damaged;
      label "SOURCE"
      ^ Printf.sprintf "git %s · %s, pid %d"
          (Option.value (Fact.commit ~dirty:" dirty" p) ~default:"—")
          (Option.value p.hostname ~default:"—")
          p.pid;
      label "" ^ String.concat " " p.command;
      label ""
      ^ Printf.sprintf "in %s · OCaml %s · raven %s · %s %s" p.cwd p.ocaml
          p.raven p.os p.arch;
    ]
  @ params @ metrics
  @ [
      label "NOTES" ^ Option.value (Run.notes r) ~default:"—";
      label "USES" ^ uses;
    ]
  @
  match files with
  | [] -> [ label "FILES" ^ "—" ]
  | f :: rest -> (label "FILES" ^ f) :: List.map (fun f -> label "" ^ f) rest

let hint = "↑↓ scroll  [ ] run  y copy id  esc back  ? help  q quit"

let paint ~word ~scroll r ~stats h ~marked g ~width ~height =
  let ls = lines ~word r ~stats h ~marked in
  let room = height - 2 in
  let scroll = max 0 (min scroll (List.length ls - room)) in
  List.iteri
    (fun i l ->
      if i >= scroll && i - scroll < room then
        Paint.text g ~x:0 ~y:(i - scroll) ~w:width l)
    ls;
  Paint.text g ~x:0 ~y:(height - 1) ~w:width hint
