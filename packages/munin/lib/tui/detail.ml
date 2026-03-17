(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic
module Charts = Matrix_charts

(* Exponential moving average: alpha in (0,1]. *)
let ema alpha history =
  match history with
  | [] -> []
  | (s0, v0) :: rest ->
      let rec loop acc prev = function
        | [] -> List.rev acc
        | (s, v) :: xs ->
            let smoothed = (alpha *. v) +. ((1. -. alpha) *. prev) in
            loop ((s, smoothed) :: acc) smoothed xs
      in
      (s0, v0) :: loop [] v0 rest

(* Statistics *)

type stats = {
  last : float;
  best : float option;
  best_step : int option;
  mean : float;
  count : int;
}

let compute_stats history ~best =
  match history with
  | [] -> None
  | _ ->
      let count = ref 0 in
      let sum = ref 0. in
      let last = ref 0. in
      List.iter
        (fun (_, v) ->
          incr count;
          sum := !sum +. v;
          last := v)
        history;
      let n = !count in
      let mean = !sum /. float_of_int n in
      let best_step =
        match best with
        | None -> None
        | Some bv ->
            let rec find = function
              | [] -> None
              | (s, v) :: _ when Float.equal v bv -> Some s
              | _ :: rest -> find rest
            in
            find history
      in
      Some { last = !last; best; best_step; mean; count = n }

let format_step step =
  if step < 1000 then string_of_int step
  else
    let s = string_of_int step in
    let len = String.length s in
    let buf = Buffer.create (len + ((len - 1) / 3)) in
    for i = 0 to len - 1 do
      if i > 0 && (len - i) mod 3 = 0 then Buffer.add_char buf ',';
      Buffer.add_char buf s.[i]
    done;
    Buffer.contents buf

let view_stats_row stats =
  let parts =
    [ Printf.sprintf "Last: %.4g" stats.last ]
    @ (match (stats.best, stats.best_step) with
      | Some bv, Some bs ->
          [ Printf.sprintf "Best: %.4g (step %s)" bv (format_step bs) ]
      | Some bv, None -> [ Printf.sprintf "Best: %.4g" bv ]
      | None, _ -> [])
    @ [
        Printf.sprintf "Mean: %.4g" stats.mean;
        Printf.sprintf "Samples: %s" (format_step stats.count);
      ]
  in
  box ~justify_content:Center ~align_items:Center
    ~size:{ width = pct 100; height = auto }
    [ text ~style:Theme.muted_style (String.concat "  \u{00B7}  " parts) ]

let meta_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:12) ()

let view_metric_def_row (metric_def : Munin.Run.metric_def) =
  let parts =
    (match metric_def.goal with
      | Some `Minimize -> [ "Goal: minimize" ]
      | Some `Maximize -> [ "Goal: maximize" ]
      | None -> [])
    @ (match metric_def.summary with
      | `Last -> []
      | `Min -> [ "Summary: min" ]
      | `Max -> [ "Summary: max" ]
      | `Mean -> [ "Summary: mean" ]
      | `None -> [ "Summary: none" ])
    @
    match metric_def.step_metric with
    | Some sm -> [ Printf.sprintf "Step metric: %s" sm ]
    | None -> []
  in
  if parts = [] then None
  else
    Some
      (box ~justify_content:Center ~align_items:Center
         ~size:{ width = pct 100; height = auto }
         [ text ~style:meta_style (String.concat "  \u{00B7}  " parts) ])

(* View *)

let view ~tag ~history_for_tag ~best ~size ~smooth ~metric_def =
  let history = history_for_tag tag in
  let display_history =
    match smooth with None -> history | Some alpha -> ema alpha history
  in
  let title =
    match Theme.last_value history with
    | None -> tag
    | Some v -> Printf.sprintf "%s [%.4f]" tag v
  in
  let title = if Option.is_some smooth then title ^ " (EMA)" else title in
  let stats = compute_stats history ~best in
  let stats_rows =
    (match stats with Some s -> [ view_stats_row s ] | None -> [])
    @
    match metric_def with
    | Some md -> (
        match view_metric_def_row md with Some row -> [ row ] | None -> [])
    | None -> []
  in
  box ~flex_direction:Column ~gap:(gap 1) ~align_items:Center ~size
    ([
       box ~border:true ~title ~padding:(padding 1)
         ~size:{ width = pct 100; height = pct 100 }
         ~flex_grow:1.0
         [
           canvas
             ~size:{ width = pct 100; height = pct 100 }
             (fun c ~delta:_ ->
               let width = Canvas.width c in
               let height = Canvas.height c in
               let grid = Canvas.grid c in
               match smooth with
               | None ->
                   Theme.draw_metric_chart ~compact:false history grid ~width
                     ~height
               | Some _ ->
                   if history = [] then ()
                   else
                     let to_arr h =
                       Array.of_list
                         (List.map
                            (fun (step, value) -> (float_of_int step, value))
                            h)
                     in
                     let raw_data = to_arr history in
                     let smooth_data = to_arr display_history in
                     let chart =
                       Charts.empty ()
                       |> Charts.with_frame
                            (Charts.manual_frame ~margins:(1, 1, 1, 4) ())
                       |> Charts.with_axes
                            ~x:
                              (Charts.Axis.default |> Charts.Axis.with_ticks 6
                              |> Charts.Axis.with_style Theme.axis_style)
                            ~y:
                              (Charts.Axis.default |> Charts.Axis.with_ticks 4
                              |> Charts.Axis.with_style Theme.axis_style
                              |> Charts.Axis.with_format (fun _ v ->
                                  Printf.sprintf "%.4g" v))
                       |> Charts.with_grid
                            (Charts.Gridlines.default
                            |> Charts.Gridlines.with_style Theme.grid_style
                            |> Charts.Gridlines.with_x true
                            |> Charts.Gridlines.with_y true)
                       |> Charts.line ~id:"raw" ~resolution:`Braille2x4
                            ~style:
                              (Ansi.Style.make
                                 ~fg:(Ansi.Color.grayscale ~level:8)
                                 ())
                            ~x:fst ~y:snd raw_data
                       |> Charts.line ~id:"smooth" ~resolution:`Braille2x4
                            ~style:(Ansi.Style.make ~fg:Ansi.Color.cyan ())
                            ~x:fst ~y:snd smooth_data
                     in
                     ignore (Charts.draw chart grid ~width ~height));
         ];
     ]
    @ stats_rows)
