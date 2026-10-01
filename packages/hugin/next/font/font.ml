(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_next_gg

type glyph = int
type slant = [ `Normal | `Italic | `Oblique ]
type error = Io of string | Malformed of string | Unsupported of string

let pp_error ppf = function
  | Io msg -> Format.pp_print_string ppf msg
  | Malformed msg -> Format.fprintf ppf "malformed font: %s" msg
  | Unsupported msg -> Format.fprintf ppf "unsupported font: %s" msg

exception Decode_error of error

let malformed fmt =
  Printf.ksprintf (fun msg -> raise (Decode_error (Malformed msg))) fmt

let unsupported fmt =
  Printf.ksprintf (fun msg -> raise (Decode_error (Unsupported msg))) fmt

(* Tables

   A table is a range of the font's bytes. Reads are relative to its start and
   fail as malformed past its end, so that a decoded font never reads outside
   what it validated. Positions are sums of offsets read from the file, never
   negative. *)

type range = File | Table of string | Glyph of int
type table = { s : string; range : range; off : int; len : int }

(* [past t i] fails on a read of [t] that would end at its byte [i]. Messages
   give positions in the file. *)
let past t i =
  match t.range with
  | File -> malformed "the file ends before byte %d" i
  | Table tag -> malformed "the '%s' table ends before byte %d" tag (t.off + i)
  | Glyph g -> malformed "the data of glyph %d ends before byte %d" g (t.off + i)

(* [check_end t stop] checks that [t] holds its bytes before [stop]. *)
let check_end t stop = if stop > t.len then past t stop

let u8 t i =
  check_end t (i + 1);
  Char.code (String.unsafe_get t.s (t.off + i))

let i8 t i =
  let v = u8 t i in
  if v >= 0x80 then v - 0x100 else v

let u16 t i =
  check_end t (i + 2);
  String.get_uint16_be t.s (t.off + i)

let i16 t i =
  check_end t (i + 2);
  String.get_int16_be t.s (t.off + i)

let i32 t i =
  check_end t (i + 4);
  Int32.to_int (String.get_int32_be t.s (t.off + i))

let u32 t i = i32 t i land 0xFFFF_FFFF

let directory s =
  let file = { s; range = File; off = 0; len = String.length s } in
  (match u32 file 0 with
  | 0x00010000 | 0x74727565 -> ()
  | 0x4F54544F -> unsupported "CFF outlines"
  | 0x74746366 -> unsupported "font collection"
  | 0x774F4646 | 0x774F4632 -> unsupported "WOFF compression"
  | _ -> malformed "not an OpenType file");
  let tables = Hashtbl.create 16 in
  for i = u16 file 4 - 1 downto 0 do
    let r = 12 + (16 * i) in
    let off = u32 file (r + 8) and len = u32 file (r + 12) in
    let tag = String.sub s r 4 in
    if off + len > String.length s then
      malformed "the '%s' table extends past the end of the file" tag;
    Hashtbl.replace tables tag { s; range = Table tag; off; len }
  done;
  tables

(* Character maps *)

type cmap =
  | Segments of {
      sub : table;
      ends : int array;
      starts : int array;
      deltas : int array;
      offsets : int array;
    }
    (* [offsets.(i)] is the position in [sub] of the glyph id of segment [i]'s
       start, or [-1] if the segment adds its delta to characters. *)
  | Groups of { starts : int array; ends : int array; glyphs : int array }

(* A table starting at [pos] of [t], ending with it; past its end it is empty,
   and every read of it fails. *)
let subtable (t : table) pos = { t with off = t.off + pos; len = t.len - pos }

(* Format 4 segments, the last of which, starting at U+FFFF, maps nothing. *)
let segments sub =
  let seg2 = u16 sub 6 in
  if seg2 mod 2 = 1 then malformed "'cmap' format 4: odd segment count";
  let n = seg2 / 2 in
  let ends = Array.init n (fun i -> u16 sub (14 + (2 * i))) in
  let starts = Array.init n (fun i -> u16 sub (16 + seg2 + (2 * i))) in
  let deltas = Array.init n (fun i -> u16 sub (16 + (2 * seg2) + (2 * i))) in
  let offsets =
    Array.init n (fun i ->
        let pos = 16 + (3 * seg2) + (2 * i) in
        match u16 sub pos with
        | 0 -> -1
        | ro ->
            let start = starts.(i) and stop = ends.(i) in
            if start <= stop && start <> 0xFFFF then
              check_end sub (pos + ro + (2 * (stop - start)) + 2);
            pos + ro)
  in
  for i = 1 to n - 1 do
    if ends.(i) <= ends.(i - 1) then
      malformed "'cmap' format 4: segments not in increasing order"
  done;
  Segments { sub; ends; starts; deltas; offsets }

let groups sub =
  let n = u32 sub 12 in
  check_end sub (16 + (12 * n));
  let field k i = u32 sub (16 + (12 * i) + k) in
  let starts = Array.init n (field 0) in
  let ends = Array.init n (field 4) in
  let glyphs = Array.init n (field 8) in
  for i = 0 to n - 1 do
    if starts.(i) > ends.(i) || (i > 0 && starts.(i) <= ends.(i - 1)) then
      malformed "'cmap' format 12: groups not in increasing order"
  done;
  Groups { starts; ends; glyphs }

(* The Unicode subtable to read: format 12 over format 4, then the Windows
   platform over the Unicode one. *)
let cmap t =
  let best = ref None and best_rank = ref (-1) in
  for i = 0 to u16 t 2 - 1 do
    let r = 4 + (8 * i) in
    let platform = u16 t r and encoding = u16 t (r + 2) in
    let sub = subtable t (u32 t (r + 4)) in
    let unicode =
      platform = 0 || (platform = 3 && (encoding = 1 || encoding = 10))
    in
    let rank =
      match u16 sub 0 with
      | 12 -> 2 + Bool.to_int (platform = 3)
      | 4 -> Bool.to_int (platform = 3)
      | _ -> -1
    in
    if unicode && rank > !best_rank then begin
      best := Some sub;
      best_rank := rank
    end
  done;
  match !best with
  | None -> unsupported "no Unicode character map of format 4 or 12"
  | Some sub -> if u16 sub 0 = 12 then groups sub else segments sub

(* [search n f] is the least [i] in [0;n[ with [f i], or [n] if there is none;
   [f] is monotone. *)
let search n f =
  let rec loop lo hi =
    if lo >= hi then lo
    else
      let mid = (lo + hi) / 2 in
      if f mid then loop lo mid else loop (mid + 1) hi
  in
  loop 0 n

let lookup cmap c =
  match cmap with
  | Segments { sub; ends; starts; deltas; offsets } ->
      let i = search (Array.length ends) (fun i -> ends.(i) >= c) in
      if i = Array.length ends || starts.(i) > c || starts.(i) = 0xFFFF then 0
      else if offsets.(i) < 0 then (c + deltas.(i)) land 0xFFFF
      else
        let g = u16 sub (offsets.(i) + (2 * (c - starts.(i)))) in
        if g = 0 then 0 else (g + deltas.(i)) land 0xFFFF
  | Groups { starts; ends; glyphs } ->
      let i = search (Array.length ends) (fun i -> ends.(i) >= c) in
      if i = Array.length ends || starts.(i) > c then 0
      else glyphs.(i) + (c - starts.(i))

(* Kerning *)

(* A [GPOS] pair adjustment subtable, positions relative to the table. *)
type pairs =
  | Glyph_pairs of {
      coverage : int;
      format1 : int;
      format2 : int;
      sets : int array;
    }
  | Class_pairs of {
      coverage : int;
      format1 : int;
      format2 : int;
      classes1 : int;
      classes2 : int;
      count2 : int;
      records : int;
    }

(* [Kern] holds the position and the pair count of each horizontal format 0
   subtable of a [kern] table. *)
type kerning =
  | No_kerning
  | Gpos of { gpos : table; lookups : pairs list list }
  | Kern of { kern : table; subtables : (int * int) list }

let value_size format =
  let n = ref 0 in
  for bit = 0 to 7 do
    if format land (1 lsl bit) <> 0 then incr n
  done;
  2 * !n

(* The x advance in the value record at [pos] of format [format]. *)
let x_advance t pos format =
  if format land 0x4 = 0 then 0
  else
    i16 t
      (pos
      + (2 * Bool.to_int (format land 1 <> 0))
      + (2 * Bool.to_int (format land 2 <> 0)))

(* [check_coverage t pos] checks the coverage table at [pos] and is the largest
   coverage index it gives, or [-1] if it covers nothing. *)
let check_coverage t pos =
  match u16 t pos with
  | 1 ->
      let n = u16 t (pos + 2) in
      for i = 0 to n - 1 do
        let g = u16 t (pos + 4 + (2 * i)) in
        if i > 0 && g <= u16 t (pos + 2 + (2 * i)) then
          malformed "'GPOS': coverage not in increasing order"
      done;
      n - 1
  | 2 ->
      let n = u16 t (pos + 2) in
      let last = ref (-1) in
      for i = 0 to n - 1 do
        let r = pos + 4 + (6 * i) in
        let start = u16 t r and stop = u16 t (r + 2) in
        if start > stop || (i > 0 && start <= u16 t (r - 4)) then
          malformed "'GPOS': coverage ranges not in increasing order";
        last := Int.max !last (u16 t (r + 4) + stop - start)
      done;
      !last
  | f -> malformed "'GPOS': coverage format %d" f

let coverage_index t pos g =
  match u16 t pos with
  | 1 ->
      let n = u16 t (pos + 2) in
      let i = search n (fun i -> u16 t (pos + 4 + (2 * i)) >= g) in
      if i < n && u16 t (pos + 4 + (2 * i)) = g then i else -1
  | _ ->
      let n = u16 t (pos + 2) in
      let i = search n (fun i -> u16 t (pos + 6 + (6 * i)) >= g) in
      let r = pos + 4 + (6 * i) in
      if i < n && u16 t r <= g then u16 t (r + 4) + g - u16 t r else -1

let check_classes t pos count =
  match u16 t pos with
  | 1 ->
      for i = 0 to u16 t (pos + 4) - 1 do
        if u16 t (pos + 6 + (2 * i)) >= count then
          malformed "'GPOS': class beyond the class count"
      done
  | 2 ->
      for i = 0 to u16 t (pos + 2) - 1 do
        let r = pos + 4 + (6 * i) in
        if u16 t r > u16 t (r + 2) || (i > 0 && u16 t r <= u16 t (r - 4)) then
          malformed "'GPOS': class ranges not in increasing order";
        if u16 t (r + 4) >= count then
          malformed "'GPOS': class beyond the class count"
      done
  | f -> malformed "'GPOS': class definition format %d" f

let class_of t pos g =
  match u16 t pos with
  | 1 ->
      let start = u16 t (pos + 2) in
      if g >= start && g < start + u16 t (pos + 4) then
        u16 t (pos + 6 + (2 * (g - start)))
      else 0
  | _ ->
      let n = u16 t (pos + 2) in
      let i = search n (fun i -> u16 t (pos + 6 + (6 * i)) >= g) in
      let r = pos + 4 + (6 * i) in
      if i < n && u16 t r <= g then u16 t (r + 4) else 0

let pair_subtable t pos =
  let coverage = pos + u16 t (pos + 2) in
  let format1 = u16 t (pos + 4) and format2 = u16 t (pos + 6) in
  let last = check_coverage t coverage in
  match u16 t pos with
  | 1 ->
      let count = u16 t (pos + 8) in
      if last >= count then malformed "'GPOS': coverage beyond the pair sets";
      let record = 2 + value_size format1 + value_size format2 in
      let sets =
        Array.init count (fun i ->
            let set = pos + u16 t (pos + 10 + (2 * i)) in
            let n = u16 t set in
            check_end t (set + 2 + (n * record));
            for j = 1 to n - 1 do
              if
                u16 t (set + 2 + (j * record))
                <= u16 t (set + 2 + ((j - 1) * record))
              then malformed "'GPOS': pairs not in increasing order"
            done;
            set)
      in
      Glyph_pairs { coverage; format1; format2; sets }
  | 2 ->
      let classes1 = pos + u16 t (pos + 8)
      and classes2 = pos + u16 t (pos + 10) in
      let count1 = u16 t (pos + 12) and count2 = u16 t (pos + 14) in
      if count1 = 0 || count2 = 0 then malformed "'GPOS': empty class matrix";
      check_classes t classes1 count1;
      check_classes t classes2 count2;
      let records = pos + 16 in
      let size = value_size format1 + value_size format2 in
      check_end t (records + (count1 * count2 * size));
      Class_pairs
        { coverage; format1; format2; classes1; classes2; count2; records }
  | f -> malformed "'GPOS': pair adjustment format %d" f

(* The pair adjustment subtables of the lookups that [kern] features refer to,
   each lookup once, in lookup order, or [None] if no feature is [kern]. *)
let gpos t =
  let features = u16 t 6 and lookup_list = u16 t 8 in
  let lookup_count = u16 t lookup_list in
  let has_kern = ref false in
  let kern = Array.make lookup_count false in
  for i = 0 to u16 t features - 1 do
    let r = features + 2 + (6 * i) in
    if u32 t r = 0x6B65726E (* "kern" *) then begin
      has_kern := true;
      let f = features + u16 t (r + 4) in
      for j = 0 to u16 t (f + 2) - 1 do
        let l = u16 t (f + 4 + (2 * j)) in
        if l >= lookup_count then malformed "'GPOS': lookup %d does not exist" l;
        kern.(l) <- true
      done
    end
  done;
  let lookups = ref [] in
  for l = lookup_count - 1 downto 0 do
    if kern.(l) then begin
      let pos = lookup_list + u16 t (lookup_list + 2 + (2 * l)) in
      let kind = u16 t pos in
      let subtables =
        List.init (u16 t (pos + 4)) (fun i -> pos + u16 t (pos + 6 + (2 * i)))
        |> List.filter_map (fun st ->
            match kind with
            | 2 -> Some (pair_subtable t st)
            | 9 when u16 t (st + 2) = 2 ->
                Some (pair_subtable t (st + u32 t (st + 4)))
            | _ -> None)
      in
      lookups := subtables :: !lookups
    end
  done;
  if !has_kern then Some (Gpos { gpos = t; lookups = !lookups }) else None

let pair_x_advance t g g' = function
  | Glyph_pairs { coverage; format1; format2; sets } ->
      let i = coverage_index t coverage g in
      if i < 0 then None
      else
        let set = sets.(i) in
        let record = 2 + value_size format1 + value_size format2 in
        let n = u16 t set in
        let j = search n (fun j -> u16 t (set + 2 + (j * record)) >= g') in
        let r = set + 2 + (j * record) in
        if j < n && u16 t r = g' then Some (x_advance t (r + 2) format1)
        else None
  | Class_pairs
      { coverage; format1; format2; classes1; classes2; count2; records } ->
      if coverage_index t coverage g < 0 then None
      else
        let c1 = class_of t classes1 g and c2 = class_of t classes2 g' in
        let size = value_size format1 + value_size format2 in
        Some (x_advance t (records + (size * ((c1 * count2) + c2))) format1)

(* The horizontal, non-minimum, non-cross-stream format 0 subtables of a version
   0 [kern] table. *)
let kern t =
  if u16 t 0 <> 0 then No_kerning
  else
    let rec subtables i pos acc =
      if i = 0 then List.rev acc
      else
        let coverage = u16 t (pos + 4) in
        let n = u16 t (pos + 6) in
        let acc =
          if coverage land 0xFF07 = 1 then begin
            check_end t (pos + 14 + (6 * n));
            for j = 1 to n - 1 do
              let key k =
                (u16 t (pos + 14 + (6 * k)) lsl 16)
                lor u16 t (pos + 16 + (6 * k))
              in
              if key j <= key (j - 1) then
                malformed "'kern': pairs not in increasing order"
            done;
            (pos + 14, n) :: acc
          end
          else acc
        in
        subtables (i - 1) (pos + u16 t (pos + 2)) acc
    in
    Kern { kern = t; subtables = subtables (u16 t 2) 4 [] }

let kern_value t (pos, n) g g' =
  let key = (g lsl 16) lor g' in
  let entry j = (u16 t (pos + (6 * j)) lsl 16) lor u16 t (pos + 2 + (6 * j)) in
  let j = search n (fun j -> entry j >= key) in
  if j < n && entry j = key then i16 t (pos + 4 + (6 * j)) else 0

(* Glyph outlines *)

type shape =
  | Empty
  | Simple of {
      ends : int array;
      on : Bytes.t;
      xs : float array;
      ys : float array;
    }
  | Composite of (int * Affine.t) list (* Components and their maps. *)

let simple g t contours =
  let ends = Array.init contours (fun i -> u16 t (10 + (2 * i))) in
  for i = 1 to contours - 1 do
    if ends.(i) <= ends.(i - 1) then
      malformed "glyph %d: contour ends not in increasing order" g
  done;
  let n = if contours = 0 then 0 else ends.(contours - 1) + 1 in
  let flags = Bytes.create n in
  let p = ref (12 + (2 * contours) + u16 t (10 + (2 * contours))) in
  let i = ref 0 in
  while !i < n do
    let f = u8 t !p in
    incr p;
    let repeat =
      if f land 8 <> 0 then (
        let r = u8 t !p in
        incr p;
        r)
      else 0
    in
    if !i + repeat >= n then
      malformed "glyph %d: flags repeat past its points" g;
    Bytes.fill flags !i (repeat + 1) (Char.unsafe_chr f);
    i := !i + repeat + 1
  done;
  let coords short same =
    let v = ref 0 in
    Array.init n (fun i ->
        let f = Char.code (Bytes.unsafe_get flags i) in
        if f land short <> 0 then begin
          let d = u8 t !p in
          incr p;
          v := if f land same <> 0 then !v + d else !v - d
        end
        else if f land same = 0 then begin
          v := !v + i16 t !p;
          p := !p + 2
        end;
        float !v)
  in
  let xs = coords 0x02 0x10 in
  let ys = coords 0x04 0x20 in
  Simple { ends; on = flags; xs; ys }

let f2dot14 t pos = float (i16 t pos) /. 16384.

let composite g t =
  let rec components pos acc =
    let flags = u16 t pos and c = u16 t (pos + 2) in
    if flags land 0x2 = 0 then
      unsupported "glyph %d: component placed by point matching" g;
    let words = flags land 0x1 <> 0 in
    let dx, dy, pos =
      if words then (float (i16 t (pos + 4)), float (i16 t (pos + 6)), pos + 8)
      else (float (i8 t (pos + 4)), float (i8 t (pos + 5)), pos + 6)
    in
    let xx, yx, xy, yy, pos =
      if flags land 0x8 <> 0 then
        let s = f2dot14 t pos in
        (s, 0., 0., s, pos + 2)
      else if flags land 0x40 <> 0 then
        (f2dot14 t pos, 0., 0., f2dot14 t (pos + 2), pos + 4)
      else if flags land 0x80 <> 0 then
        ( f2dot14 t pos,
          f2dot14 t (pos + 2),
          f2dot14 t (pos + 4),
          f2dot14 t (pos + 6),
          pos + 8 )
      else (1., 0., 0., 1., pos)
    in
    (* With [SCALED_COMPONENT_OFFSET] the offset goes through the scale. *)
    let x0, y0 =
      if flags land 0x800 <> 0 then
        ((xx *. dx) +. (xy *. dy), (yx *. dx) +. (yy *. dy))
      else (dx, dy)
    in
    let acc = (c, { Affine.xx; yx; xy; yy; x0; y0 }) :: acc in
    if flags land 0x20 <> 0 then components pos acc else List.rev acc
  in
  Composite (components 10 [])

type glyphs = { glyf : table; loca : int array }

let shape glyphs g =
  let start = glyphs.loca.(g) and stop = glyphs.loca.(g + 1) in
  if stop = start then Empty
  else
    let t =
      {
        glyphs.glyf with
        range = Glyph g;
        off = glyphs.glyf.off + start;
        len = stop - start;
      }
    in
    let contours = i16 t 0 in
    if contours >= 0 then simple g t contours else composite g t

(* The points of every glyph, composites summed, checked against the 65535 of
   TrueType's point indices; this also rejects cycles of components. *)
let validate_glyphs glyphs n =
  let points = Array.make n (-1) in
  let rec count g =
    match points.(g) with
    | -2 -> malformed "glyph %d: composite contains itself" g
    | -1 ->
        points.(g) <- -2;
        let p =
          match shape glyphs g with
          | Empty -> 0
          | Simple { xs; _ } -> Array.length xs
          | Composite cs ->
              List.fold_left
                (fun acc (c, _) ->
                  if c >= n then
                    malformed "glyph %d: component %d does not exist" g c;
                  acc + count c)
                0 cs
        in
        if p > 0xFFFF then malformed "glyph %d: more than 65535 points" g;
        points.(g) <- p;
        p
    | p -> p
  in
  for g = 0 to n - 1 do
    ignore (count g)
  done

(* Fonts *)

type t = {
  data : string;
  upem : float;
  glyph_count : int;
  advances : float array;
  glyphs : glyphs;
  cmap : cmap;
  kerning : kerning;
  ascent : float;
  descent : float;
  line_gap : float;
  cap_height : float;
  x_height : float;
  italic_angle : float;
  bounds : Box2.t;
  family : string;
  postscript_name : string;
  weight : int;
  slant : slant;
}

let check_glyph fn f g =
  if g < 0 || g >= f.glyph_count then
    invalid_arg
      (Printf.sprintf "Font.%s: glyph %d not in [0, %d]" fn g (f.glyph_count - 1))

let glyph_count f = f.glyph_count

let glyph f u =
  let g = lookup f.cmap (Uchar.to_int u) in
  if g < f.glyph_count then g else 0

let advance f g =
  check_glyph "advance" f g;
  f.advances.(g)

let kerning f g g' =
  check_glyph "kerning" f g;
  check_glyph "kerning" f g';
  let units =
    match f.kerning with
    | No_kerning -> 0
    | Gpos { gpos; lookups } ->
        let rec first = function
          | [] -> 0
          | st :: sts -> (
              match pair_x_advance gpos g g' st with
              | Some v -> v
              | None -> first sts)
        in
        List.fold_left (fun acc subtables -> acc + first subtables) 0 lookups
    | Kern { kern; subtables } ->
        List.fold_left (fun acc st -> acc + kern_value kern st g g') 0 subtables
  in
  float units /. f.upem

(* Outlines are emitted contour by contour from the on-curve point found first,
   or the midpoint of the first two points when a contour has none; between two
   off-curve points lies an implied on-curve midpoint. *)
let contours upem (m : Affine.t) ends on xs ys path =
  (* [0. -. y] keeps the baseline at [0.] rather than [-0.]. *)
  let pt x y =
    let x' = (m.xx *. x) +. (m.xy *. y) +. m.x0 in
    let y' = (m.yx *. x) +. (m.yy *. y) +. m.y0 in
    P2.v (x' /. upem) ((0. -. y') /. upem)
  in
  let path = ref path and start = ref 0 in
  Array.iter
    (fun stop ->
      let n = stop - !start + 1 and base = !start in
      start := stop + 1;
      if n >= 2 then begin
        let on i = Char.code (Bytes.get on (base + (i mod n))) land 1 <> 0 in
        let x i = xs.(base + (i mod n)) and y i = ys.(base + (i mod n)) in
        let first =
          let rec find i =
            if i >= n then -1 else if on i then i else find (i + 1)
          in
          find 0
        in
        let sx, sy, first =
          if first >= 0 then (x first, y first, first)
          else ((x 0 +. x 1) /. 2., (y 0 +. y 1) /. 2., 0)
        in
        path := Path.move_to (pt sx sy) !path;
        let ctrl = ref None in
        let curve_to ex ey =
          match !ctrl with
          | None -> path := Path.line_to (pt ex ey) !path
          | Some (cx, cy) ->
              path := Path.quad_to (pt cx cy) (pt ex ey) !path;
              ctrl := None
        in
        (* A line back to the start is left to [close]. *)
        for i = first + 1 to first + n do
          if on i then
            begin if i < first + n || Option.is_some !ctrl then
              curve_to (x i) (y i)
            end
          else begin
            (match !ctrl with
            | Some (cx, cy) -> curve_to ((cx +. x i) /. 2.) ((cy +. y i) /. 2.)
            | None -> ());
            ctrl := Some (x i, y i)
          end
        done;
        (match !ctrl with Some _ -> curve_to sx sy | None -> ());
        path := Path.close !path
      end)
    ends;
  !path

let outline f g =
  check_glyph "outline" f g;
  let rec emit g m path =
    match shape f.glyphs g with
    | Empty -> path
    | Simple { ends; on; xs; ys } -> contours f.upem m ends on xs ys path
    | Composite cs ->
        List.fold_left (fun path (c, cm) -> emit c Affine.(m * cm) path) path cs
  in
  emit g Affine.id Path.empty

(* Names *)

(* Characters 0x80 to 0xFF of Mac OS Roman. *)
let mac_roman =
  [|
    0x00C4; 0x00C5; 0x00C7; 0x00C9; 0x00D1; 0x00D6; 0x00DC; 0x00E1; 0x00E0;
    0x00E2; 0x00E4; 0x00E3; 0x00E5; 0x00E7; 0x00E9; 0x00E8; 0x00EA; 0x00EB;
    0x00ED; 0x00EC; 0x00EE; 0x00EF; 0x00F1; 0x00F3; 0x00F2; 0x00F4; 0x00F6;
    0x00F5; 0x00FA; 0x00F9; 0x00FB; 0x00FC; 0x2020; 0x00B0; 0x00A2; 0x00A3;
    0x00A7; 0x2022; 0x00B6; 0x00DF; 0x00AE; 0x00A9; 0x2122; 0x00B4; 0x00A8;
    0x2260; 0x00C6; 0x00D8; 0x221E; 0x00B1; 0x2264; 0x2265; 0x00A5; 0x00B5;
    0x2202; 0x2211; 0x220F; 0x03C0; 0x222B; 0x00AA; 0x00BA; 0x03A9; 0x00E6;
    0x00F8; 0x00BF; 0x00A1; 0x00AC; 0x221A; 0x0192; 0x2248; 0x2206; 0x00AB;
    0x00BB; 0x2026; 0x00A0; 0x00C0; 0x00C3; 0x00D5; 0x0152; 0x0153; 0x2013;
    0x2014; 0x201C; 0x201D; 0x2018; 0x2019; 0x00F7; 0x25CA; 0x00FF; 0x0178;
    0x2044; 0x20AC; 0x2039; 0x203A; 0xFB01; 0xFB02; 0x2021; 0x00B7; 0x201A;
    0x201E; 0x2030; 0x00C2; 0x00CA; 0x00C1; 0x00CB; 0x00C8; 0x00CD; 0x00CE;
    0x00CF; 0x00CC; 0x00D3; 0x00D4; 0xF8FF; 0x00D2; 0x00DA; 0x00DB; 0x00D9;
    0x0131; 0x02C6; 0x02DC; 0x00AF; 0x02D8; 0x02D9; 0x02DA; 0x00B8; 0x02DD;
    0x02DB; 0x02C7;
  |]
[@@ocamlformat "disable"]

let decode_name t ~utf_16 pos len =
  check_end t (pos + len);
  let b = Buffer.create len in
  let s = String.sub t.s (t.off + pos) len in
  let i = ref 0 in
  while !i < len do
    if utf_16 then begin
      let d = String.get_utf_16be_uchar s !i in
      Buffer.add_utf_8_uchar b (Uchar.utf_decode_uchar d);
      i := !i + Uchar.utf_decode_length d
    end
    else begin
      let c = Char.code s.[!i] in
      Buffer.add_utf_8_uchar b
        (Uchar.of_int (if c < 0x80 then c else mac_roman.(c - 0x80)));
      incr i
    end
  done;
  Buffer.contents b

(* The name [id] of the US English Windows Unicode record, else of the first
   Windows Unicode record, else of the first Macintosh Roman one. *)
let name t id =
  let strings = u16 t 4 in
  let us = ref None and windows = ref None and mac = ref None in
  for i = u16 t 2 - 1 downto 0 do
    let r = 6 + (12 * i) in
    if u16 t (r + 6) = id then begin
      let platform = u16 t r and encoding = u16 t (r + 2) in
      let record = Some (strings + u16 t (r + 10), u16 t (r + 8)) in
      if platform = 3 && (encoding = 1 || encoding = 10) then begin
        windows := record;
        if u16 t (r + 4) = 0x409 then us := record
      end
      else if platform = 1 && encoding = 0 then mac := record
    end
  done;
  match (!us, !windows, !mac) with
  | Some (pos, len), _, _ | None, Some (pos, len), _ ->
      Some (decode_name t ~utf_16:true pos len)
  | None, None, Some (pos, len) -> Some (decode_name t ~utf_16:false pos len)
  | None, None, None -> None

(* Loading *)

let decode s =
  let tables = directory s in
  let find tag = Hashtbl.find_opt tables tag in
  let need tag =
    match find tag with Some t -> t | None -> malformed "no '%s' table" tag
  in
  let head = need "head" and hhea = need "hhea" and maxp = need "maxp" in
  if u32 head 12 <> 0x5F0F3CF5 then malformed "'head': wrong magic number";
  let units = u16 head 18 in
  if units < 16 || units > 16384 then
    malformed "'head': %d units per em not in [16, 16384]" units;
  let upem = float units in
  let glyph_count = u16 maxp 4 in
  if glyph_count = 0 then malformed "'maxp': no glyphs";
  let hmetrics = u16 hhea 34 in
  if hmetrics = 0 then malformed "'hhea': no horizontal metrics";
  let hmtx = need "hmtx" in
  let advances =
    Array.init glyph_count (fun g ->
        float (u16 hmtx (4 * Int.min g (hmetrics - 1))) /. upem)
  in
  let glyf =
    match find "glyf" with
    | Some t -> t
    | None -> unsupported "no 'glyf' outlines"
  in
  let loca = need "loca" in
  let long =
    match i16 head 50 with
    | 0 -> false
    | 1 -> true
    | f -> malformed "'head': loca format %d" f
  in
  let loca =
    Array.init (glyph_count + 1) (fun i ->
        if long then u32 loca (4 * i) else 2 * u16 loca (2 * i))
  in
  for i = 0 to glyph_count - 1 do
    if loca.(i + 1) < loca.(i) then
      malformed "'loca': offsets not in increasing order"
  done;
  if loca.(glyph_count) > glyf.len then
    malformed "'loca': glyph data past the 'glyf' table";
  let glyphs = { glyf; loca } in
  validate_glyphs glyphs glyph_count;
  let cmap = cmap (need "cmap") in
  let kerning =
    match (Option.bind (find "GPOS") gpos, find "kern") with
    | Some k, _ -> k
    | None, Some t -> kern t
    | None, None -> No_kerning
  in
  let os2 = find "OS/2" in
  let fs_selection = match os2 with Some t -> u16 t 62 | None -> 0 in
  (* The ascender, descender and line gap at [v], [v + 2] and [v + 4] of
     [vt]. *)
  let vt, v =
    match os2 with
    | Some t when fs_selection land 0x80 <> 0 -> (t, 68)
    | _ -> (hhea, 4)
  in
  let em t pos = float (i16 t pos) /. upem in
  let ascent = em vt v in
  let descent = -.em vt (v + 2) in
  let line_gap = em vt (v + 4) in
  let os2_height pos =
    match os2 with
    | Some t when u16 t 0 >= 2 && i16 t pos > 0 -> Some (em t pos)
    | _ -> None
  in
  let name_table = find "name" in
  let name id = Option.bind name_table (fun t -> name t id) in
  let family =
    match name 16 with Some n -> n | None -> Option.value (name 1) ~default:""
  in
  let f =
    {
      data = s;
      upem;
      glyph_count;
      advances;
      glyphs;
      cmap;
      kerning;
      ascent;
      descent;
      line_gap;
      cap_height = 0.;
      x_height = 0.;
      italic_angle =
        (match find "post" with
        | Some t -> float (i32 t 4) /. 65536. *. Float.pi /. 180.
        | None -> 0.);
      bounds =
        Box2.of_pts
          (P2.v (em head 36) (0. -. em head 42))
          (P2.v (em head 40) (0. -. em head 38));
      family;
      postscript_name = Option.value (name 6) ~default:"";
      weight =
        (match os2 with
        | Some t -> Int.max 1 (Int.min 1000 (u16 t 4))
        | None -> 400);
      slant =
        (match os2 with
        | Some _ when fs_selection land 0x1 <> 0 -> `Italic
        | Some _ when fs_selection land 0x200 <> 0 -> `Oblique
        | Some _ -> `Normal
        | None -> if u16 head 44 land 0x2 <> 0 then `Italic else `Normal);
    }
  in
  (* The top of the ink of the glyph [u] maps to, if it maps to one with ink. *)
  let ink_top u =
    match glyph f (Uchar.of_char u) with
    | 0 -> None
    | g -> Option.map (fun b -> 0. -. Box2.miny b) (Path.bounds (outline f g))
  in
  let height pos u default =
    match os2_height pos with
    | Some h -> h
    | None -> ( match ink_top u with Some h -> h | None -> default)
  in
  { f with cap_height = height 88 'H' ascent; x_height = height 86 'x' 0.5 }

let of_string s =
  match decode s with f -> Ok f | exception Decode_error e -> Error e

let of_file path =
  match In_channel.with_open_bin path In_channel.input_all with
  | s -> of_string s
  | exception Sys_error msg -> Error (Io msg)

let bundled s =
  match of_string s with
  | Ok f -> f
  | Error e -> Format.kasprintf failwith "Font: bundled face: %a" pp_error e

let regular = bundled Font_data.inter_regular
let bold = bundled Font_data.inter_bold

(* Identity *)

let bytes f = f.data
let equal f f' = f == f' || String.equal f.data f'.data
let compare f f' = if f == f' then 0 else String.compare f.data f'.data
let family f = f.family
let postscript_name f = f.postscript_name
let weight f = f.weight
let slant f = f.slant

let pp ppf f =
  Format.fprintf ppf "@[<1>(font %S@ %d@ %s)@]" f.family f.weight
    (match f.slant with
    | `Normal -> "normal"
    | `Italic -> "italic"
    | `Oblique -> "oblique")

(* Metrics *)

let ascent f = f.ascent
let descent f = f.descent
let line_gap f = f.line_gap
let cap_height f = f.cap_height
let x_height f = f.x_height
let italic_angle f = f.italic_angle
let bounds f = f.bounds
