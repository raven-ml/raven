(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

(* Helpers *)

let format_timestamp t =
  let tm = Unix.localtime t in
  Printf.sprintf "%04d-%02d-%02d %02d:%02d:%02d" (1900 + tm.tm_year)
    (1 + tm.tm_mon) tm.tm_mday tm.tm_hour tm.tm_min tm.tm_sec

let format_elapsed secs =
  let secs = int_of_float secs in
  let h = secs / 3600 in
  let m = secs mod 3600 / 60 in
  let s = secs mod 60 in
  Printf.sprintf "%02d:%02d:%02d" h m s

let short_hash s = if String.length s > 7 then String.sub s 0 7 else s

(* View *)

let view ~(run : Munin.Run.t) ~(status : Theme.run_status) ~elapsed_secs
    ~(metric_defs : (string * Munin.Run.metric_def) list)
    ~(latest_metrics : (string * Munin.Run.metric) list)
    ~(step_metrics : string list) ~(best_for_tag : string -> float option) =
  let prov = Munin.Run.provenance run in
  let status_color = Theme.status_color status in
  let run_section =
    [ Overview.section_header "Run" ]
    @ (match Munin.Run.name run with
      | Some name -> [ Overview.kv_row "Name" name ]
      | None -> [])
    @ [ Overview.kv_row "ID" (Munin.Run.id run) ]
    @ [ Overview.kv_row "Experiment" (Munin.Run.experiment_name run) ]
    @ (match Munin.Run.group run with
      | Some g -> [ Overview.kv_row "Group" g ]
      | None -> [])
    @ [
        box ~flex_direction:Row ~gap:(gap 1)
          ~size:{ width = pct 100; height = auto }
          [
            text ~style:Overview.key_style "Status";
            box ~flex_grow:1.0 ~size:{ width = auto; height = auto } [];
            text ~style:(Ansi.Style.make ~fg:status_color ()) "\u{25CF} ";
            text
              ~style:(Ansi.Style.make ~bold:true ~fg:status_color ())
              (Theme.status_label status);
          ];
        Overview.kv_row "Started" (format_timestamp (Munin.Run.started_at run));
        Overview.kv_row "Duration" (format_elapsed elapsed_secs);
      ]
    @
    match Munin.Run.tags run with
    | [] -> []
    | tags -> [ Overview.kv_row "Tags" (String.concat ", " tags) ]
  in
  let prov_section =
    [ Overview.section_header "Provenance" ]
    @ [
        Overview.kv_row "Command" (String.concat " " prov.command);
        Overview.kv_row "Directory" prov.cwd;
      ]
    @ (match prov.hostname with
      | Some h -> [ Overview.kv_row "Hostname" h ]
      | None -> [])
    @ [ Overview.kv_row "PID" (string_of_int prov.pid) ]
    @
    match prov.git_commit with
    | Some hash ->
        let dirty =
          match prov.git_dirty with
          | Some true -> " (dirty)"
          | Some false -> " (clean)"
          | None -> ""
        in
        [ Overview.kv_row "Git" (short_hash hash ^ dirty) ]
    | None -> []
  in
  let params = Munin.Run.params run in
  let params_section =
    if params = [] then []
    else
      [ Overview.section_header "Params" ]
      @ List.map
          (fun (k, v) -> Overview.kv_row k (Overview.format_value v))
          params
  in
  let metrics =
    List.filter
      (fun (k, _) -> (not (Overview.is_sys k)) && not (List.mem k step_metrics))
      latest_metrics
  in
  let summary_section =
    if metrics = [] then []
    else
      [ Overview.section_header "Summary" ]
      @ List.map
          (fun (k, m) ->
            Overview.format_summary_value ~metric_defs ~best_for_tag k m)
          metrics
  in
  let notes_section =
    match Munin.Run.notes run with
    | Some notes when notes <> "" ->
        [
          Overview.section_header "Notes"; text ~style:Overview.val_style notes;
        ]
    | _ -> []
  in
  box ~flex_direction:Column ~padding:(padding 2) ~gap:(gap 1)
    ~size:{ width = pct 100; height = pct 100 }
    (run_section @ prov_section @ params_section @ summary_section
   @ notes_section)
