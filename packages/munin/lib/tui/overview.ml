(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

(* Styles *)

let section_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:12) ~dim:true ()

let key_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:16) ()
let val_style = Ansi.Style.make ~fg:Ansi.Color.white ()

(* Helpers *)

let is_sys tag = String.length tag > 4 && String.sub tag 0 4 = "sys/"

let format_value (v : Munin.Value.t) =
  match v with
  | `Bool b -> string_of_bool b
  | `Int i -> string_of_int i
  | `Float f ->
      if Float.is_integer f && Float.abs f < 1e9 then Printf.sprintf "%.0f" f
      else Printf.sprintf "%g" f
  | `String s -> s

let format_float f =
  if Float.is_integer f && Float.abs f < 1e6 then Printf.sprintf "%.0f" f
  else if Float.abs f >= 0.01 && Float.abs f < 1e6 then Printf.sprintf "%.4f" f
  else Printf.sprintf "%.4g" f

(* View *)

let section_header label =
  box
    ~size:{ width = pct 100; height = auto }
    [ text ~style:section_style (Printf.sprintf "\u{2500}\u{2500} %s " label) ]

let kv_row k v =
  box ~flex_direction:Row ~gap:(gap 1)
    ~size:{ width = pct 100; height = auto }
    [
      text ~style:key_style k;
      box ~flex_grow:1.0 ~size:{ width = auto; height = auto } [];
      text ~style:val_style v;
    ]

let badge_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:10) ~dim:true ()

let format_summary_value ~(metric_defs : (string * Munin.Run.metric_def) list)
    ~(best_for_tag : string -> float option) k (m : Munin.Run.metric) =
  let def = List.assoc_opt k metric_defs in
  let value =
    match (def, best_for_tag k) with
    | Some { summary = `Min | `Max; _ }, Some bv -> format_float bv
    | _ -> format_float m.value
  in
  let summary_badge =
    match def with
    | Some { summary = `Min; _ } -> " (min)"
    | Some { summary = `Max; _ } -> " (max)"
    | Some { summary = `Mean; _ } -> " (mean)"
    | Some { summary = `None; _ } -> " (none)"
    | Some { summary = `Last; _ } | None -> ""
  in
  let goal_arrow =
    match def with
    | Some { goal = Some `Minimize; _ } -> " \u{2193}"
    | Some { goal = Some `Maximize; _ } -> " \u{2191}"
    | _ -> ""
  in
  box ~flex_direction:Row ~gap:(gap 0)
    ~size:{ width = pct 100; height = auto }
    [
      text ~style:key_style k;
      box ~flex_grow:1.0 ~size:{ width = auto; height = auto } [];
      text ~style:val_style value;
      text ~style:badge_style (summary_badge ^ goal_arrow);
    ]

let view ~(run : Munin.Run.t)
    ~(latest_metrics : (string * Munin.Run.metric) list)
    ~(step_metrics : string list)
    ~(metric_defs : (string * Munin.Run.metric_def) list)
    ~(best_for_tag : string -> float option) =
  let params = Munin.Run.params run in
  let metrics =
    List.filter
      (fun (k, _) -> (not (is_sys k)) && not (List.mem k step_metrics))
      latest_metrics
  in
  let sections =
    (if params <> [] then
       [ section_header "Params" ]
       @ List.map (fun (k, v) -> kv_row k (format_value v)) params
     else [])
    @
    if metrics <> [] then
      [ section_header "Summary" ]
      @ List.map
          (fun (k, m) -> format_summary_value ~metric_defs ~best_for_tag k m)
          metrics
    else []
  in
  if sections = [] then box ~size:{ width = px 0; height = px 0 } []
  else
    box ~flex_direction:Column ~gap:(gap 0)
      ~size:{ width = pct 100; height = auto }
      sections
