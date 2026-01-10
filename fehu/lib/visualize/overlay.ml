(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ctx = {
  step_idx : int;
  episode_idx : int;
  info : Fehu.Info.t;
  action : Fehu.Space.Value.t option;
  value : float option;
  log_prob : float option;
  reward : float;
  done_ : bool;
}

type t = Fehu.Render.image -> ctx -> Fehu.Render.image

let identity image _ = image

let compose overlays =
 fun image ctx ->
  List.fold_left (fun img overlay -> overlay img ctx) image overlays

let fst3 (a, _, _) = a
let snd3 (_, b, _) = b
let thd3 (_, _, c) = c

module Glyph = struct
  type t = { width : int; rows : int array }

  let make rows =
    let height = Array.length rows in
    if height <> 7 then invalid_arg "Glyph.make: expected 7 rows";
    let width = String.length rows.(0) in
    let rows_bits =
      Array.map
        (fun row ->
          if String.length row <> width then
            invalid_arg "Glyph.make: inconsistent row width";
          let bits = ref 0 in
          for idx = 0 to width - 1 do
            let bit = if row.[idx] <> '0' then 1 else 0 in
            bits := (!bits lsl 1) lor bit
          done;
          !bits)
        rows
    in
    { width; rows = rows_bits }

  let table =
    let open struct
      let glyph c rows = (c, make rows)
    end in
    [
      glyph '0'
        [| "01110"; "10001"; "10011"; "10101"; "11001"; "10001"; "01110" |];
      glyph '1'
        [| "00100"; "01100"; "00100"; "00100"; "00100"; "00100"; "01110" |];
      glyph '2'
        [| "01110"; "10001"; "00001"; "00010"; "00100"; "01000"; "11111" |];
      glyph '3'
        [| "01110"; "10001"; "00001"; "00110"; "00001"; "10001"; "01110" |];
      glyph '4'
        [| "00010"; "00110"; "01010"; "10010"; "11111"; "00010"; "00010" |];
      glyph '5'
        [| "11111"; "10000"; "11110"; "00001"; "00001"; "10001"; "01110" |];
      glyph '6'
        [| "00111"; "01000"; "10000"; "11110"; "10001"; "10001"; "01110" |];
      glyph '7'
        [| "11111"; "00001"; "00010"; "00100"; "01000"; "01000"; "01000" |];
      glyph '8'
        [| "01110"; "10001"; "10001"; "01110"; "10001"; "10001"; "01110" |];
      glyph '9'
        [| "01110"; "10001"; "10001"; "01111"; "00001"; "00010"; "11100" |];
      glyph 'A'
        [| "01110"; "10001"; "10001"; "11111"; "10001"; "10001"; "10001" |];
      glyph 'B'
        [| "11110"; "10001"; "10001"; "11110"; "10001"; "10001"; "11110" |];
      glyph 'C'
        [| "01110"; "10001"; "10000"; "10000"; "10000"; "10001"; "01110" |];
      glyph 'D'
        [| "11100"; "10010"; "10001"; "10001"; "10001"; "10010"; "11100" |];
      glyph 'E'
        [| "11111"; "10000"; "10000"; "11110"; "10000"; "10000"; "11111" |];
      glyph 'F'
        [| "11111"; "10000"; "10000"; "11110"; "10000"; "10000"; "10000" |];
      glyph 'G'
        [| "01110"; "10001"; "10000"; "10000"; "10011"; "10001"; "01111" |];
      glyph 'H'
        [| "10001"; "10001"; "10001"; "11111"; "10001"; "10001"; "10001" |];
      glyph 'I'
        [| "01110"; "00100"; "00100"; "00100"; "00100"; "00100"; "01110" |];
      glyph 'J'
        [| "00001"; "00001"; "00001"; "00001"; "10001"; "10001"; "01110" |];
      glyph 'K'
        [| "10001"; "10010"; "10100"; "11000"; "10100"; "10010"; "10001" |];
      glyph 'L'
        [| "10000"; "10000"; "10000"; "10000"; "10000"; "10000"; "11111" |];
      glyph 'M'
        [| "10001"; "11011"; "10101"; "10101"; "10001"; "10001"; "10001" |];
      glyph 'N'
        [| "10001"; "11001"; "10101"; "10011"; "10001"; "10001"; "10001" |];
      glyph 'O'
        [| "01110"; "10001"; "10001"; "10001"; "10001"; "10001"; "01110" |];
      glyph 'P'
        [| "11110"; "10001"; "10001"; "11110"; "10000"; "10000"; "10000" |];
      glyph 'Q'
        [| "01110"; "10001"; "10001"; "10001"; "10101"; "10010"; "01101" |];
      glyph 'R'
        [| "11110"; "10001"; "10001"; "11110"; "10100"; "10010"; "10001" |];
      glyph 'S'
        [| "01111"; "10000"; "10000"; "01110"; "00001"; "00001"; "11110" |];
      glyph 'T'
        [| "11111"; "00100"; "00100"; "00100"; "00100"; "00100"; "00100" |];
      glyph 'U'
        [| "10001"; "10001"; "10001"; "10001"; "10001"; "10001"; "01110" |];
      glyph 'V'
        [| "10001"; "10001"; "10001"; "10001"; "10001"; "01010"; "00100" |];
      glyph 'W'
        [| "10001"; "10001"; "10001"; "10101"; "10101"; "11011"; "10001" |];
      glyph 'X'
        [| "10001"; "10001"; "01010"; "00100"; "01010"; "10001"; "10001" |];
      glyph 'Y'
        [| "10001"; "10001"; "01010"; "00100"; "00100"; "00100"; "00100" |];
      glyph 'Z'
        [| "11111"; "00001"; "00010"; "00100"; "01000"; "10000"; "11111" |];
      glyph ' '
        [| "00000"; "00000"; "00000"; "00000"; "00000"; "00000"; "00000" |];
      glyph ':'
        [| "00000"; "00100"; "00100"; "00000"; "00100"; "00100"; "00000" |];
      glyph '.'
        [| "00000"; "00000"; "00000"; "00000"; "00000"; "00100"; "00100" |];
      glyph '/'
        [| "00001"; "00010"; "00100"; "01000"; "10000"; "00000"; "00000" |];
      glyph '-'
        [| "00000"; "00000"; "00000"; "11111"; "00000"; "00000"; "00000" |];
      glyph '_'
        [| "00000"; "00000"; "00000"; "00000"; "00000"; "11111"; "00000" |];
      glyph '+'
        [| "00000"; "00100"; "00100"; "11111"; "00100"; "00100"; "00000" |];
    ]

  let lookup =
    let table =
      let tbl = Hashtbl.create 64 in
      List.iter (fun (ch, glyph) -> Hashtbl.add tbl ch glyph) table;
      tbl
    in
    fun ch ->
      let key = Char.uppercase_ascii ch in
      match Hashtbl.find_opt table key with
      | Some glyph -> glyph
      | None -> Hashtbl.find table ' '
end

let with_rgb8 image k =
  let converted, buffer = Utils.convert_to_rgb8 image in
  k converted buffer;
  converted

let draw_text buffer ~width ~x ~y ~r ~g ~b text =
  let cursor_x = ref x in
  String.iter
    (fun ch ->
      let glyph = Glyph.lookup ch in
      for row = 0 to Array.length glyph.rows - 1 do
        let row_bits = glyph.rows.(row) in
        for col = 0 to glyph.width - 1 do
          let mask = 1 lsl (glyph.width - 1 - col) in
          if row_bits land mask <> 0 then
            Utils.set_pixel_rgb8 ~buffer ~width ~x:(!cursor_x + col)
              ~y:(y + row) ~r ~g ~b
        done
      done;
      cursor_x := !cursor_x + glyph.width + 1)
    text

let text ?(pos = (10, 20)) ?(color = (255, 255, 255)) to_string image ctx =
  let label = to_string ctx in
  if String.length label = 0 then image
  else
    with_rgb8 image (fun converted buffer ->
        let x, y = pos in
        draw_text buffer ~width:converted.width ~x ~y ~r:(fst3 color)
          ~g:(snd3 color) ~b:(thd3 color) label)

let clamp_unit value =
  if value < 0. then 0. else if value > 1. then 1. else value

let bar ?pos ?(size = (120, 8)) ?(color = (40, 200, 120))
    ?(background = (20, 20, 20)) ~value image ctx =
  let v = clamp_unit (value ctx) in
  with_rgb8 image (fun converted buffer ->
      let width = converted.width in
      let height = converted.height in
      let x, y =
        match pos with
        | Some (px, py) -> (px, py)
        | None ->
            let default_y = height - snd size - 10 in
            (10, max 0 default_y)
      in
      let bar_width, bar_height = size in
      Utils.fill_rect_rgb8 ~buffer ~width ~x ~y ~w:bar_width ~h:bar_height
        ~r:(fst3 background) ~g:(snd3 background) ~b:(thd3 background);
      let filled =
        int_of_float (Float.round (v *. Float.of_int bar_width))
        |> max 0 |> min bar_width
      in
      Utils.fill_rect_rgb8 ~buffer ~width ~x ~y ~w:filled ~h:bar_height
        ~r:(fst3 color) ~g:(snd3 color) ~b:(thd3 color))
