(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module G = Matrix_grid

let text ?(style = Ansi.Style.default) g ~x ~y ~w s =
  let w = min w (G.width g - x) in
  if y >= 0 && y < G.height g && x >= 0 && w > 0 then
    G.draw_text ~style g ~x ~y ~text:(Text.cut w s)

let plain = Ansi.Style.default
let bold = Ansi.Style.make ~bold:true ()
let dim = Ansi.Style.make ~dim:true ()
let reverse = Ansi.Style.make ~inverse:true ()
let alert = Ansi.Style.make ~fg:Ansi.Color.red ~bold:true ()

let word = function
  | "live" | "writing" -> Ansi.Style.make ~fg:Ansi.Color.green ()
  | "stalled" | "silent" | "running" | "starting" ->
      Ansi.Style.make ~fg:Ansi.Color.yellow ()
  | "failed" | "crashed" -> alert
  | "killed" -> dim
  | _ -> plain
