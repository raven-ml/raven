(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_next_gg

type t = {
  font : Font.t;
  size : float;
  text : string;
  glyphs : Font.glyph array;
  xs : float array;
  ys : float array;
  clusters : int array;
}

let err fmt = Printf.ksprintf invalid_arg ("Run.v: " ^^ fmt)

(* The byte index of each Unicode character of [text], valid UTF-8. *)
let char_starts text =
  let rec loop i acc =
    if i >= String.length text then Array.of_list (List.rev acc)
    else
      loop
        (i + Uchar.utf_decode_length (String.get_utf_8_uchar text i))
        (i :: acc)
  in
  loop 0 []

(* [starts_char text i] for an index [i] past the checks of clusters, which make
   it non-negative. *)
let starts_char text i =
  if String.length text = 0 then i = 0
  else i < String.length text && Char.code text.[i] land 0xC0 <> 0x80

let v ?ys ?clusters ~font ~size ~text ~glyphs ~xs () =
  let glyphs = Array.copy glyphs and xs = Array.copy xs in
  let ys = Option.map Array.copy ys
  and clusters = Option.map Array.copy clusters in
  let n = Array.length glyphs in
  let length what a =
    if Array.length a <> n then
      err "%d glyphs but %d %s" n (Array.length a) what
  in
  length "xs" xs;
  Option.iter (length "ys") ys;
  Option.iter (length "clusters") clusters;
  Array.iter
    (fun g ->
      if g < 0 || g >= Font.glyph_count font then
        err "glyph %d not in [0, %d]" g (Font.glyph_count font - 1))
    glyphs;
  if not (size >= 0. && Float.is_finite size) then err "invalid size %g" size;
  let finite what a =
    Array.iter
      (fun x -> if not (Float.is_finite x) then err "%s holds %g" what x)
      a
  in
  finite "xs" xs;
  Option.iter (finite "ys") ys;
  if not (String.is_valid_utf_8 text) then err "text is not valid UTF-8";
  if n = 0 && text <> "" then err "no glyph renders %S" text;
  let clusters =
    match clusters with
    | Some c ->
        Array.iteri
          (fun i k ->
            if i = 0 && k <> 0 then err "clusters start at %d" k;
            if i > 0 && k < c.(i - 1) then err "clusters decrease at glyph %d" i;
            if not (starts_char text k) then
              err "cluster %d does not start a character" k)
          c;
        c
    | None ->
        let c = char_starts text in
        if Array.length c <> n then
          err "%d glyphs for %d characters" n (Array.length c);
        c
  in
  let ys = match ys with Some ys -> ys | None -> Array.make n 0. in
  { font; size; text; glyphs; xs; ys; clusters }

let font r = r.font
let size r = r.size
let text r = r.text
let length r = Array.length r.glyphs

let check fn r i =
  if i < 0 || i >= Array.length r.glyphs then
    invalid_arg
      (Printf.sprintf "Run.%s: index %d not in [0, %d]" fn i
         (Array.length r.glyphs - 1))

let glyph r i =
  check "glyph" r i;
  r.glyphs.(i)

let x r i =
  check "x" r i;
  r.xs.(i)

let y r i =
  check "y" r i;
  r.ys.(i)

let cluster r i =
  check "cluster" r i;
  r.clusters.(i)

(* A size is not negative, so scaling keeps a box's minimum corner its
   minimum. *)
let bounds r =
  let s = r.size in
  let minx = ref infinity and miny = ref infinity in
  let maxx = ref neg_infinity and maxy = ref neg_infinity in
  for i = 0 to Array.length r.glyphs - 1 do
    match Font.ink r.font r.glyphs.(i) with
    | None -> ()
    | Some b ->
        let x = r.xs.(i) and y = r.ys.(i) in
        minx := Float.min !minx (x +. (s *. Box2.minx b));
        miny := Float.min !miny (y +. (s *. Box2.miny b));
        maxx := Float.max !maxx (x +. (s *. Box2.maxx b));
        maxy := Float.max !maxy (y +. (s *. Box2.maxy b))
  done;
  if !minx > !maxx then None
  else Some (Box2.of_pts (P2.v !minx !miny) (P2.v !maxx !maxy))

let equal r r' =
  r == r'
  || Array.length r.glyphs = Array.length r'.glyphs
     && Font.equal r.font r'.font && Float.equal r.size r'.size
     && String.equal r.text r'.text
     && Array.for_all2 Int.equal r.glyphs r'.glyphs
     && Array.for_all2 Float.equal r.xs r'.xs
     && Array.for_all2 Float.equal r.ys r'.ys
     && Array.for_all2 Int.equal r.clusters r'.clusters

let pp ppf r =
  Format.fprintf ppf "@[<1>(run %a %g %a" Font.pp r.font r.size Literal.pp
    r.text;
  Array.iteri
    (fun i g ->
      Format.fprintf ppf "@ %d@@%g,%g#%d" g r.xs.(i) r.ys.(i) r.clusters.(i))
    r.glyphs;
  Format.fprintf ppf ")@]"
