(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module G = Matrix_grid
module S = Ansi.Style
module C = Ansi.Color

type depth = Mono | Ansi16 | Ansi256 | Truecolor

let depth =
  let env v = Option.value (Sys.getenv_opt v) ~default:"" in
  if env "NO_COLOR" <> "" then Mono
  else
    match String.lowercase_ascii (env "COLORTERM") with
    | "truecolor" | "24bit" -> Truecolor
    | _ ->
        let term = env "TERM" in
        let n = String.length term in
        let rec has i =
          i + 8 <= n && (String.sub term i 8 = "256color" || has (i + 1))
        in
        if has 0 then Ansi256 else Ansi16

let tinted =
  match depth with Ansi256 | Truecolor -> true | Mono | Ansi16 -> false

(* Styles *)

let plain = S.default
let dim = S.make ~dim:true ()
let bold = S.make ~bold:true ()
let reverse = S.make ~inverse:true ()
let ok = S.make ~fg:C.green ()
let bad = S.make ~fg:C.red ()
let warn = S.make ~fg:C.yellow ()
let accent = S.make ~fg:C.cyan ()
let alert = S.make ~fg:C.red ~bold:true ()
let mark_color = function 'a' -> C.cyan | 'b' -> C.magenta | _ -> C.yellow
let mark c = S.make ~fg:(mark_color c) ~bold:true ()
let with_ style overlay = S.merge ~base:style ~overlay

let selected ~light =
  if tinted then
    S.make
      ~bg:(if light then C.of_rgb 0xe4 0xe8 0xee else C.of_rgb 0x2c 0x32 0x3c)
      ()
  else reverse

let state = function
  | "live" | "writing" -> ("●", ok)
  | "running" | "starting" -> ("○", ok)
  | "finished" -> ("✓", dim)
  | "killed" -> ("■", dim)
  | "failed" | "crashed" -> ("✗", bad)
  | "stalled" | "silent" -> ("◌", warn)
  | _ -> (" ", plain)

let word w =
  let glyph, style = state w in
  [ (style, glyph ^ " "); (with_ style bold, w) ]

(* Drawing *)

type spans = (S.t * string) list

let put ?(style = plain) g ~x ~y ~w s =
  let w = min w (G.width g - x) in
  if w <= 0 || x < 0 || y < 0 || y >= G.height g then 0
  else begin
    let s = Text.cut w s in
    G.draw_text ~style g ~x ~y ~text:s;
    Text.width s
  end

let text ?style g ~x ~y ~w s = ignore (put ?style g ~x ~y ~w s)

let spans ?(base = plain) g ~x ~y ~w l =
  let right = x + w in
  ignore
    (List.fold_left
       (fun x (style, s) ->
         if x >= right then x
         else x + put ~style:(with_ base style) g ~x ~y ~w:(right - x) s)
       x l)

let width l = List.fold_left (fun n (_, s) -> n + Text.width s) 0 l

let right ?base g ~x ~y ~w l =
  let n = width l in
  spans ?base g ~x:(x + max 0 (w - n)) ~y ~w:(min w n) l

let fill g ~x ~y ~w style =
  if w > 0 && y >= 0 && y < G.height g then
    G.draw_text ~style g ~x ~y ~text:(String.make w ' ')

let sep = (dim, " · ")

let join parts =
  let rec go = function
    | [] -> []
    | [ x ] -> x
    | x :: rest -> x @ [ sep ] @ go rest
  in
  go (List.filter (( <> ) []) parts)

let fit ?(drop = []) ~w head parts =
  let line ps = head @ join (List.map snd ps) in
  let rec go ps order =
    if width (line ps) <= w then line ps
    else
      match order with
      | i :: rest -> go (List.filter (fun (j, _) -> j <> i) ps) rest
      | [] -> (
          match List.rev ps with
          | [] -> line []
          | _ :: rest -> go (List.rev rest) [])
  in
  go (List.mapi (fun i p -> (i, p)) parts) drop

let hints g ~x ~y ~w l =
  let item (k, a) = [ (plain, k); (dim, " " ^ a) ] in
  let gap = [ (dim, "   ") ] in
  let last = item ("?", "keys") in
  let rec take acc = function
    | [] -> acc
    | h :: t ->
        let next = acc @ [ item h ] in
        if width (List.concat_map (fun i -> i @ gap) next) + width last <= w
        then take next t
        else acc
  in
  spans g ~x ~y ~w (List.concat_map (fun i -> i @ gap) (take [] l) @ last)

let key k =
  match String.rindex_opt k '/' with
  | Some i ->
      (String.sub k 0 (i + 1), String.sub k (i + 1) (String.length k - i - 1))
  | None -> ("", k)

let key_spans ?(style = plain) k =
  let p, rest = key k in
  [ (with_ style dim, p); (style, rest) ]

let path w s =
  if Text.width s <= w then s
  else if w <= 1 then Text.cut w s
  else
    (* Byte offsets are cells for the ASCII paths this cuts. *)
    let n = String.length s in
    let tail = String.sub s (n - (w - 1)) (w - 1) in
    let tail =
      match String.index_opt tail '/' with
      | Some i when i < String.length tail - 1 ->
          String.sub tail i (String.length tail - i)
      | _ -> tail
    in
    "…" ^ tail

let centre g ~y ~w l =
  let n = width l in
  spans g ~x:(max 0 ((w - n) / 2)) ~y ~w l
