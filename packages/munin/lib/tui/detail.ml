(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

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

let view ~tag ~history_for_tag ~best ~size ~smooth =
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
  box ~flex_direction:Column ~gap:(gap 1) ~align_items:Center ~size
    [
      box ~border:true ~title ~padding:(padding 1)
        ~size:{ width = pct 100; height = pct 100 }
        ~flex_grow:1.0
        [
          canvas
            ~size:{ width = pct 100; height = pct 100 }
            (fun c ~delta:_ ->
              Theme.draw_metric_chart ~compact:false display_history
                (Canvas.grid c) ~width:(Canvas.width c)
                ~height:(Canvas.height c));
        ];
      (match best with
      | None -> box [] ~size:{ width = px 0; height = px 0 }
      | Some value ->
          box ~justify_content:Center ~align_items:Center
            ~size:{ width = pct 100; height = auto }
            [ text (Printf.sprintf "Best: %.4f" value) ]);
    ]
