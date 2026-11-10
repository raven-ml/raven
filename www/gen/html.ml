(* Escaping, and the few pieces of HTML the generated parts share. *)

let sprintf = Printf.sprintf
let concat_map f l = String.concat "" (List.map f l)

let add_escaped buf = function
  | '&' -> Buffer.add_string buf "&amp;"
  | '<' -> Buffer.add_string buf "&lt;"
  | '>' -> Buffer.add_string buf "&gt;"
  | '"' -> Buffer.add_string buf "&quot;"
  | '\'' -> Buffer.add_string buf "&#39;"
  | c -> Buffer.add_char buf c

(* [esc s] is [s] as HTML text or as an attribute value. Every string read
   from the repository goes through it. *)
let esc s =
  let buf = Buffer.create (String.length s + 16) in
  String.iter (add_escaped buf) s;
  Buffer.contents buf

(* A library's glyph in its colour, or an empty slot of the same width. *)
let glyph ?size (e : Registry.entry) =
  let size = match size with Some s -> sprintf "; font-size: %s" s | None -> "" in
  match e.glyph with
  | Some (rune, colour) -> sprintf {|<span class="glyph" style="color: %s%s" aria-hidden="true">%s</span>|} colour size rune
  | None -> sprintf {|<span class="glyph" style="%s" aria-hidden="true"></span>|} size

let github = "https://github.com/raven-ml/raven"
