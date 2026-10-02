(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_gg
open Hugin_font

(* Styles *)

type style = {
  bold : bool;
  italic : bool;
  k : float;
  d : float;
  color : Color.t option;
}

let plain = { bold = false; italic = false; k = 1.; d = 0.; color = None }

let style_equal a b =
  a == b
  || Bool.equal a.bold b.bold
     && Bool.equal a.italic b.italic
     && Float.equal a.k b.k && Float.equal a.d b.d
     && Option.equal Color.equal a.color b.color

let style_compare a b =
  let c = Bool.compare a.bold b.bold in
  if c <> 0 then c
  else
    let c = Bool.compare a.italic b.italic in
    if c <> 0 then c
    else
      let c = Float.compare a.k b.k in
      if c <> 0 then c
      else
        let c = Float.compare a.d b.d in
        if c <> 0 then c else Option.compare Color.compare a.color b.color

(* Texts

   A text is its spans in order: the maximal sequences of characters of one
   style, as valid UTF-8, none empty. Structural equality on this normal form is
   the equality of meaning. *)

type span = { chars : string; style : style }
type t = span list

(* [normalize spans] merges the consecutive spans of equal styles. *)
let normalize spans =
  let rec group style parts = function
    | s :: rest when style_equal s.style style ->
        group style (s.chars :: parts) rest
    | rest -> (String.concat "" (List.rev parts), rest)
  in
  let rec loop acc = function
    | [] -> List.rev acc
    | s :: rest ->
        let chars, rest = group s.style [ s.chars ] rest in
        loop ({ chars; style = s.style } :: acc) rest
  in
  loop [] spans

let repair s =
  if String.is_valid_utf_8 s then s
  else
    let b = Buffer.create (String.length s + 8) in
    let rec loop i =
      if i < String.length s then begin
        let d = String.get_utf_8_uchar s i in
        Buffer.add_utf_8_uchar b (Uchar.utf_decode_uchar d);
        loop (i + Uchar.utf_decode_length d)
      end
    in
    loop 0;
    Buffer.contents b

let v s = if s = "" then [] else [ { chars = repair s; style = plain } ]
let concat ts = normalize (List.concat ts)
let restyle f t = normalize (List.map (fun s -> { s with style = f s.style }) t)
let bold t = restyle (fun st -> { st with bold = true }) t
let italic t = restyle (fun st -> { st with italic = true }) t

let scale s t =
  if not (Float.is_finite s && s > 0.) then
    invalid_arg (Printf.sprintf "Text.scale: %g is not finite and positive" s);
  restyle (fun st -> { st with k = st.k *. s; d = st.d *. s }) t

let color c t =
  restyle
    (fun st ->
      match st.color with None -> { st with color = Some c } | Some _ -> st)
    t

let sup t =
  restyle (fun st -> { st with k = 0.7 *. st.k; d = (0.7 *. st.d) -. 0.4 }) t

let sub t =
  restyle (fun st -> { st with k = 0.7 *. st.k; d = (0.7 *. st.d) +. 0.2 }) t

let equal t t' =
  List.equal
    (fun s s' -> String.equal s.chars s'.chars && style_equal s.style s'.style)
    t t'

let compare t t' =
  List.compare
    (fun s s' ->
      let c = String.compare s.chars s'.chars in
      if c <> 0 then c else style_compare s.style s'.style)
    t t'

let pp_span ppf s =
  let st = s.style in
  if style_equal st plain then Literal.pp ppf s.chars
  else begin
    Format.fprintf ppf "@[<1>(%a" Literal.pp s.chars;
    if st.bold then Format.fprintf ppf "@ bold";
    if st.italic then Format.fprintf ppf "@ italic";
    if not (Float.equal st.k 1.) then Format.fprintf ppf "@ size %g" st.k;
    if not (Float.equal st.d 0.) then Format.fprintf ppf "@ shift %g" st.d;
    Option.iter (Format.fprintf ppf "@ %a" Color.pp) st.color;
    Format.fprintf ppf ")@]"
  end

let pp ppf t =
  Format.fprintf ppf "@[<1>(text";
  List.iter (Format.fprintf ppf "@ %a" pp_span) t;
  Format.fprintf ppf ")@]"

(* Characters *)

let is_newline u =
  match Uchar.to_int u with
  | 0x0A | 0x0B | 0x0C | 0x0D | 0x85 | 0x2028 | 0x2029 -> true
  | _ -> false

(* The Default_Ignorable_Code_Point property of Unicode 16.0. *)
let is_default_ignorable u =
  let c = Uchar.to_int u in
  if c < 0xAD then false
  else if c < 0x2000 then
    c = 0xAD || c = 0x34F || c = 0x61C
    || (0x115F <= c && c <= 0x1160)
    || (0x17B4 <= c && c <= 0x17B5)
    || (0x180B <= c && c <= 0x180F)
  else if c < 0x10000 then
    (0x200B <= c && c <= 0x200F)
    || (0x202A <= c && c <= 0x202E)
    || (0x2060 <= c && c <= 0x206F)
    || c = 0x3164
    || (0xFE00 <= c && c <= 0xFE0F)
    || c = 0xFEFF || c = 0xFFA0
    || (0xFFF0 <= c && c <= 0xFFF8)
  else
    (0x1BCA0 <= c && c <= 0x1BCA3)
    || (0x1D173 <= c && c <= 0x1D17A)
    || (0xE0000 <= c && c <= 0xE0FFF)

module Layout = struct
  type halign = [ `Left | `Center | `Right ]
  type valign = [ `Top | `Cap | `Middle | `Baseline | `Bottom ]
  type placed = { color : Color.t option; at : P2.t; run : Run.t }

  type t = {
    box : Box2.t;
    ink : Box2.t option;
    runs : placed array;
    missing : Uchar.t list;
  }

  let err fmt = Printf.ksprintf invalid_arg ("Text.Layout.v: " ^^ fmt)

  (* Faces *)

  let slant_rank asked slant =
    match (asked, slant) with
    | `Italic, `Italic | `Oblique, `Oblique | `Normal, `Normal -> 0
    | `Italic, `Oblique | `Oblique, `Italic | `Normal, `Oblique -> 1
    | `Italic, `Normal | `Oblique, `Normal | `Normal, `Italic -> 2

  (* The rank of the weight [x] of a face for the weight [w] asked for, smaller
     first. Weights are in [1, 1000], so each group of the order has its own
     interval of ranks, all below [3001]. *)
  let weight_rank w x =
    if 400 <= w && w <= 500 then
      if w <= x && x <= 500 then x else if x < w then 2000 - x else 2000 + x
    else if w < 400 then if x <= w then 1000 - x else 1000 + x
    else if x >= w then x
    else 2000 - x

  (* The indices of [faces] from best to worst for a request. *)
  let ranking faces ~bold ~italic =
    let p = faces.(0) in
    let w = if bold then 700 else Font.weight p in
    let s = if italic then `Italic else Font.slant p in
    let rank f =
      (4000 * slant_rank s (Font.slant f)) + weight_rank w (Font.weight f)
    in
    let ranks = Array.map rank faces in
    let order = Array.init (Array.length faces) Fun.id in
    Array.stable_sort (fun i j -> Int.compare ranks.(i) ranks.(j)) order;
    order

  (* Setting characters

     A glyph item is a character set in a face, with its size [s] and its
     baseline [e] below the line's. *)

  type item = {
    u : Uchar.t;
    face : int;
    glyph : Font.glyph;
    s : float;
    e : float;
    color : Color.t option;
  }

  let is_space it = Uchar.to_int it.u = 0x20

  (* The items of [text], the end of each of its paragraphs (the lines that
     newlines end) in order, and the characters no face has. *)
  let items faces ~size text =
    let rankings = Array.make 4 [||] in
    let ranking st =
      let r = (Bool.to_int st.bold * 2) + Bool.to_int st.italic in
      if Array.length rankings.(r) = 0 then
        rankings.(r) <- ranking faces ~bold:st.bold ~italic:st.italic;
      rankings.(r)
    in
    let items = Dynarray.create () and ends = ref [] and missing = ref [] in
    let after_cr = ref false in
    let set st s e u =
      let r = ranking st in
      let rec find i =
        if i = Array.length r then begin
          if not (List.exists (Uchar.equal u) !missing) then
            missing := u :: !missing;
          (r.(0), 0)
        end
        else
          match Font.glyph faces.(r.(i)) u with
          | 0 -> find (i + 1)
          | g -> (r.(i), g)
      in
      let face, glyph = find 0 in
      Dynarray.add_last items { u; face; glyph; s; e; color = st.color }
    in
    let span { chars; style = st } =
      let s = st.k *. size and e = st.d *. size in
      let rec loop i =
        if i < String.length chars then begin
          let d = String.get_utf_8_uchar chars i in
          let u = Uchar.utf_decode_uchar d in
          if not (is_default_ignorable u) then begin
            if not (Float.is_finite s && s > 0.) then
              err "size %g of U+%04X is not finite and positive" s
                (Uchar.to_int u);
            let c = Uchar.to_int u in
            if c = 0x0A && !after_cr then ()
            else if is_newline u then ends := Dynarray.length items :: !ends
            else set st s e u;
            after_cr := c = 0x0D
          end;
          loop (i + Uchar.utf_decode_length d)
        end
      in
      loop 0
    in
    List.iter span text;
    ends := Dynarray.length items :: !ends;
    (Dynarray.to_array items, List.rev !ends, List.rev !missing)

  (* Breaking lines *)

  type line = { first : int; last : int; w : float }

  (* [break faces width items xs lo hi add] calls [add] on the lines of the
     paragraph [items.(lo)] to [items.(hi - 1)], each as the items it draws and
     its width, and sets the x of those items from their line's start in
     [xs]. *)
  let break faces width items xs lo hi add =
    let advance it = Font.advance faces.(it.face) it.glyph *. it.s in
    let next_pen it pen j =
      if j = hi then pen
      else
        let it' = items.(j) in
        if
          it.face = it'.face && Float.equal it.s it'.s && Float.equal it.e it'.e
        then pen +. (Font.kerning faces.(it.face) it.glyph it'.glyph *. it.s)
        else pen
    in
    let rec line first = scan first first 0. first 0. (-1) first 0.
    (* The line from [first] has its items before [i] placed, its glyphs ending
       at [last] with width [w], and its last end point that fitted at [fit],
       [-1] for none, where its glyphs ended at [fit_last] with width
       [fit_w]. *)
    and scan first i pen last w fit fit_last fit_w =
      if i = hi then
        if w <= width || fit < 0 then add { first; last; w }
        else begin
          add { first; last = fit_last; w = fit_w };
          line fit
        end
      else begin
        let it = items.(i) in
        xs.(i) <- pen;
        let after = pen +. advance it in
        let last, w = if is_space it then (last, w) else (i + 1, after) in
        let j = i + 1 in
        if j < hi && is_space it && (not (is_space items.(j))) && last > lo then
          if w <= width then scan first j (next_pen it after j) last w j last w
          else if fit < 0 then begin
            add { first; last; w };
            line j
          end
          else begin
            add { first; last = fit_last; w = fit_w };
            line fit
          end
        else scan first j (next_pen it after j) last w fit fit_last fit_w
      end
    in
    line lo

  (* Layouts *)

  let finite what x =
    if not (Float.is_finite x) then err "%s is not finite" what

  (* [place faces items xs ~x ~y i j] is the run of the items from [i] up to [j]
     excluded, on a line that starts at [x] with its baseline at [y], and the
     run's ink. *)
  let place faces items xs ~x ~y i j =
    let it = items.(i) in
    let n = j - i in
    let text = Buffer.create (2 * n) in
    let glyphs = Array.make n 0 and rxs = Array.make n 0. in
    for k = 0 to n - 1 do
      let it = items.(i + k) in
      finite "a glyph's origin" (x +. xs.(i + k));
      Buffer.add_utf_8_uchar text it.u;
      glyphs.(k) <- it.glyph;
      rxs.(k) <- xs.(i + k) -. xs.(i)
    done;
    let ox = x +. xs.(i) and oy = y +. it.e in
    finite "a glyph's origin" oy;
    let run =
      Run.v ~font:faces.(it.face) ~size:it.s ~text:(Buffer.contents text)
        ~glyphs ~xs:rxs ()
    in
    let ink =
      match Run.bounds run with
      | exception Invalid_argument _ -> err "the ink is not finite"
      | None -> None
      | Some b ->
          let x0 = Box2.minx b +. ox and x1 = Box2.maxx b +. ox in
          let y0 = Box2.miny b +. oy and y1 = Box2.maxy b +. oy in
          List.iter (finite "the ink") [ x0; x1; y0; y1 ];
          Some (Box2.of_pts (P2.v x0 y0) (P2.v x1 y1))
    in
    ({ color = it.color; at = P2.v ox oy; run }, ink)

  let v ?(width = infinity) ?(halign = `Left) ?(valign = `Baseline) ~fonts ~size
      text =
    let faces = Array.of_list fonts in
    if Array.length faces = 0 then err "no fonts";
    if not (Float.is_finite size && size > 0.) then
      err "size %g is not finite and positive" size;
    if not (width >= 0.) then err "width %g is negative or nan" width;
    let items, ends, missing = items faces ~size text in
    let xs = Array.make (Array.length items) 0. in
    let lines = Dynarray.create () in
    let rec paragraphs lo = function
      | [] -> ()
      | hi :: ends ->
          break faces width items xs lo hi (Dynarray.add_last lines);
          paragraphs hi ends
    in
    paragraphs 0 ends;
    let lines = Dynarray.to_array lines in
    let n = Array.length lines in
    let p = faces.(0) in
    let ascent = Array.make n (Font.ascent p *. size) in
    let descent = Array.make n (Font.descent p *. size) in
    let cap_size = ref 0. in
    Array.iteri
      (fun l { first; last; _ } ->
        for i = first to last - 1 do
          let { face; s; e; _ } = items.(i) in
          let f = faces.(face) in
          ascent.(l) <- Float.max ascent.(l) ((Font.ascent f *. s) -. e);
          descent.(l) <- Float.max descent.(l) ((Font.descent f *. s) +. e);
          if l = 0 && e = 0. then cap_size := Float.max !cap_size s
        done)
      lines;
    let cap = Font.cap_height p *. if !cap_size > 0. then !cap_size else size in
    let baselines = Array.make n 0. in
    for l = 1 to n - 1 do
      baselines.(l) <-
        baselines.(l - 1)
        +. descent.(l - 1)
        +. (Font.line_gap p *. size)
        +. ascent.(l)
    done;
    (* Here and below, [0. -. x] keeps a zero offset at [0.] rather than [-0.],
       which [pp] would print. *)
    let dy =
      match valign with
      | `Top -> ascent.(0)
      | `Cap -> cap
      | `Middle -> 0.5 *. (cap -. baselines.(n - 1))
      | `Baseline -> 0. -. baselines.(n - 1)
      | `Bottom -> 0. -. (baselines.(n - 1) +. descent.(n - 1))
    in
    let runs = Dynarray.create () and ink = ref None in
    let minx = ref infinity and miny = ref infinity in
    let maxx = ref neg_infinity and maxy = ref neg_infinity in
    Array.iteri
      (fun l { first; last; w } ->
        let x =
          match halign with
          | `Left -> 0.
          | `Center -> 0. -. (w /. 2.)
          | `Right -> 0. -. w
        in
        let y = baselines.(l) +. dy in
        let top = y -. ascent.(l) and bottom = y +. descent.(l) in
        List.iter (finite "a line's box") [ x; x +. w; top; bottom ];
        minx := Float.min !minx x;
        maxx := Float.max !maxx (x +. w);
        miny := Float.min !miny top;
        maxy := Float.max !maxy bottom;
        let same it it' =
          it'.face = it.face && Float.equal it'.s it.s && Float.equal it'.e it.e
          && (it'.color == it.color
             || Option.equal Color.equal it'.color it.color)
        in
        let rec runs_from i =
          if i < last then begin
            let j = ref (i + 1) in
            while !j < last && same items.(i) items.(!j) do
              incr j
            done;
            let placed, b = place faces items xs ~x ~y i !j in
            Dynarray.add_last runs placed;
            Option.iter
              (fun b ->
                ink :=
                  Some (match !ink with None -> b | Some u -> Box2.union u b))
              b;
            runs_from !j
          end
        in
        runs_from first)
      lines;
    {
      box = Box2.of_pts (P2.v !minx !miny) (P2.v !maxx !maxy);
      ink = !ink;
      runs = Dynarray.to_array runs;
      missing;
    }

  let box l = l.box
  let ink l = l.ink
  let missing l = l.missing

  let fold f acc l =
    let acc = ref acc in
    for i = 0 to Array.length l.runs - 1 do
      let ({ color; at; run } : placed) = Array.unsafe_get l.runs i in
      acc := f !acc color at run
    done;
    !acc

  let equal l l' =
    l == l'
    || Box2.equal l.box l'.box
       && Array.length l.runs = Array.length l'.runs
       && Array.for_all2
            (fun (a : placed) (b : placed) ->
              Option.equal Color.equal a.color b.color
              && P2.equal a.at b.at && Run.equal a.run b.run)
            l.runs l'.runs

  let pp ppf l =
    Format.fprintf ppf "@[<v 1>(layout %a" Box2.pp l.box;
    Array.iter
      (fun ({ color; at; run } : placed) ->
        Format.fprintf ppf "@ @[<1>(%a@ %a@ %a)@]" P2.pp at
          (Format.pp_print_option
             ~none:(fun ppf () -> Format.pp_print_char ppf '-')
             Color.pp)
          color Run.pp run)
      l.runs;
    Format.fprintf ppf ")@]"
end
