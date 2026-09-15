(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* TrueType parsing: glyf outlines, hmtx advances, cmap and GPOS pair kerning.
   Everything is read on demand from the font bytes except the character map and
   the kerning lookups, which are decoded once. *)

exception Malformed of string
exception Unsupported of string

let malformed fmt = Printf.ksprintf (fun s -> raise (Malformed s)) fmt
let unsupported fmt = Printf.ksprintf (fun s -> raise (Unsupported s)) fmt

(* Big-endian readers *)

let u8 s i =
  if i < 0 || i >= String.length s then malformed "read past end of font";
  Char.code (String.unsafe_get s i)

let u16 s i = (u8 s i lsl 8) lor u8 s (i + 1)

let i16 s i =
  let v = u16 s i in
  if v >= 0x8000 then v - 0x10000 else v

let u32 s i = (u16 s i lsl 16) lor u16 s (i + 2)

(* Value records in GPOS *)

let value_record_size fmt =
  let n = ref 0 in
  for b = 0 to 7 do
    if fmt land (1 lsl b) <> 0 then incr n
  done;
  2 * !n

(* [x_advance s off fmt] is the xAdvance field of the value record at [off] with
   format [fmt], or 0 when the record has none. *)
let x_advance s off fmt =
  if fmt land 0x4 = 0 then 0
  else begin
    let skip =
      (if fmt land 0x1 <> 0 then 2 else 0) + if fmt land 0x2 <> 0 then 2 else 0
    in
    i16 s (off + skip)
  end

(* Coverage and class definition tables *)

(* [coverage_index s off g] is the coverage index of glyph [g] in the table at
   [off], or -1. *)
let coverage_index s off g =
  match u16 s off with
  | 1 ->
      let n = u16 s (off + 2) in
      let lo = ref 0 and hi = ref (n - 1) and found = ref (-1) in
      while !found < 0 && !lo <= !hi do
        let mid = (!lo + !hi) / 2 in
        let v = u16 s (off + 4 + (2 * mid)) in
        if v = g then found := mid
        else if v < g then lo := mid + 1
        else hi := mid - 1
      done;
      !found
  | 2 ->
      let n = u16 s (off + 2) in
      let lo = ref 0 and hi = ref (n - 1) and found = ref (-1) in
      while !found < 0 && !lo <= !hi do
        let mid = (!lo + !hi) / 2 in
        let r = off + 4 + (6 * mid) in
        let start = u16 s r and stop = u16 s (r + 2) in
        if g < start then hi := mid - 1
        else if g > stop then lo := mid + 1
        else found := u16 s (r + 4) + (g - start)
      done;
      !found
  | f -> unsupported "coverage format %d" f

let class_of s off g =
  match u16 s off with
  | 1 ->
      let start = u16 s (off + 2) and n = u16 s (off + 4) in
      if g >= start && g < start + n then u16 s (off + 6 + (2 * (g - start)))
      else 0
  | 2 ->
      let n = u16 s (off + 2) in
      let lo = ref 0 and hi = ref (n - 1) and found = ref 0 and go = ref true in
      while !go && !lo <= !hi do
        let mid = (!lo + !hi) / 2 in
        let r = off + 4 + (6 * mid) in
        let start = u16 s r and stop = u16 s (r + 2) in
        if g < start then hi := mid - 1
        else if g > stop then lo := mid + 1
        else begin
          found := u16 s (r + 4);
          go := false
        end
      done;
      !found
  | f -> unsupported "class definition format %d" f

(* Pair positioning subtables *)

type pair_pos =
  | Pairs of { coverage : int; fmt1 : int; fmt2 : int; sets : int }
      (** Format 1: [sets] is the offset of the pair set offsets. *)
  | Classes of {
      coverage : int;
      fmt1 : int;
      fmt2 : int;
      class1 : int;
      class2 : int;
      class2_count : int;
      records : int;
    }  (** Format 2: [records] is the offset of the class pair matrix. *)

type t = {
  data : string;
  tables : (string, int * int) Hashtbl.t;
  units_per_em : int;
  ascender : int;
  descender : int;
  num_glyphs : int;
  num_hmetrics : int;
  hmtx : int;
  loca : int;
  long_loca : bool;
  glyf : int;
  cmap : (int, int) Hashtbl.t;
  kern : pair_pos list list;  (** One list of subtables per kern lookup. *)
  family : string;
  weight : int;
}

(* Table directory *)

let parse_directory s =
  let tables = Hashtbl.create 16 in
  let num = u16 s 4 in
  for i = 0 to num - 1 do
    let r = 12 + (16 * i) in
    let tag = String.sub s r 4 in
    let off = u32 s (r + 8) and len = u32 s (r + 12) in
    if off + len > String.length s then malformed "%s table past end" tag;
    Hashtbl.replace tables tag (off, len)
  done;
  tables

(* Character map *)

let parse_cmap s off =
  let map = Hashtbl.create 1024 in
  let n = u16 s (off + 2) in
  let best = ref None in
  for i = 0 to n - 1 do
    let r = off + 4 + (8 * i) in
    let platform = u16 s r and encoding = u16 s (r + 2) in
    let sub = off + u32 s (r + 4) in
    let format = u16 s sub in
    let rank =
      match (platform, encoding, format) with
      | 3, 10, 12 -> 3
      | 3, 1, 4 -> 2
      | 0, _, (4 | 12) -> 1
      | _ -> 0
    in
    match !best with
    | Some (r', _) when r' >= rank -> ()
    | _ -> if rank > 0 then best := Some (rank, sub)
  done;
  begin match !best with
  | None -> unsupported "no Unicode cmap subtable"
  | Some (_, sub) -> (
      match u16 s sub with
      | 4 ->
          let seg_count = u16 s (sub + 6) / 2 in
          let ends = sub + 14 in
          let starts = ends + (2 * seg_count) + 2 in
          let deltas = starts + (2 * seg_count) in
          let range_offsets = deltas + (2 * seg_count) in
          for i = 0 to seg_count - 1 do
            let stop = u16 s (ends + (2 * i)) in
            let start = u16 s (starts + (2 * i)) in
            let delta = u16 s (deltas + (2 * i)) in
            let ro_addr = range_offsets + (2 * i) in
            let ro = u16 s ro_addr in
            if start <= stop && start <> 0xFFFF then
              for c = start to stop do
                let g =
                  if ro = 0 then (c + delta) land 0xFFFF
                  else begin
                    let g = u16 s (ro_addr + ro + (2 * (c - start))) in
                    if g = 0 then 0 else (g + delta) land 0xFFFF
                  end
                in
                if g <> 0 then Hashtbl.replace map c g
              done
          done
      | 12 ->
          let groups = u32 s (sub + 12) in
          for i = 0 to groups - 1 do
            let r = sub + 16 + (12 * i) in
            let start = u32 s r and stop = u32 s (r + 4) in
            let g0 = u32 s (r + 8) in
            for c = start to stop do
              Hashtbl.replace map c (g0 + c - start)
            done
          done
      | f -> unsupported "cmap format %d" f)
  end;
  map

(* Kerning lookups *)

let parse_pair_pos s off =
  match u16 s off with
  | 1 ->
      Pairs
        {
          coverage = off + u16 s (off + 2);
          fmt1 = u16 s (off + 4);
          fmt2 = u16 s (off + 6);
          sets = off + 10;
        }
  | 2 ->
      Classes
        {
          coverage = off + u16 s (off + 2);
          fmt1 = u16 s (off + 4);
          fmt2 = u16 s (off + 6);
          class1 = off + u16 s (off + 8);
          class2 = off + u16 s (off + 10);
          class2_count = u16 s (off + 14);
          records = off + 16;
        }
  | f -> unsupported "pair positioning format %d" f

let parse_gpos s off =
  let feature_list = off + u16 s (off + 6) in
  let lookup_list = off + u16 s (off + 8) in
  let kern_lookups = ref [] in
  let n = u16 s feature_list in
  for i = 0 to n - 1 do
    let r = feature_list + 2 + (6 * i) in
    if String.sub s r 4 = "kern" then begin
      let f = feature_list + u16 s (r + 4) in
      let count = u16 s (f + 2) in
      for j = 0 to count - 1 do
        let idx = u16 s (f + 4 + (2 * j)) in
        if not (List.mem idx !kern_lookups) then
          kern_lookups := idx :: !kern_lookups
      done
    end
  done;
  let lookup idx =
    let l = lookup_list + u16 s (lookup_list + 2 + (2 * idx)) in
    let typ = u16 s l and count = u16 s (l + 4) in
    List.init count (fun i ->
        let st = l + u16 s (l + 6 + (2 * i)) in
        match typ with
        | 2 -> Some (parse_pair_pos s st)
        | 9 ->
            if u16 s (st + 2) = 2 then
              Some (parse_pair_pos s (st + u32 s (st + 4)))
            else None
        | _ -> None)
    |> List.filter_map Fun.id
  in
  List.sort compare !kern_lookups |> List.map lookup

(* Name and weight *)

let utf16be_to_utf8 s off len =
  let b = Buffer.create len in
  let i = ref 0 in
  while !i + 1 < len do
    let c = u16 s (off + !i) in
    if c >= 0xD800 && c <= 0xDBFF && !i + 3 < len then begin
      let lo = u16 s (off + !i + 2) in
      let u = 0x10000 + ((c - 0xD800) lsl 10) + (lo - 0xDC00) in
      Buffer.add_utf_8_uchar b (Uchar.of_int u);
      i := !i + 4
    end
    else begin
      if Uchar.is_valid c then Buffer.add_utf_8_uchar b (Uchar.of_int c);
      i := !i + 2
    end
  done;
  Buffer.contents b

let parse_family s off =
  let count = u16 s (off + 2) in
  let strings = off + u16 s (off + 4) in
  let best = ref None in
  for i = 0 to count - 1 do
    let r = off + 6 + (12 * i) in
    let platform = u16 s r and name_id = u16 s (r + 6) in
    let len = u16 s (r + 8) and so = u16 s (r + 10) in
    if name_id = 1 then
      match (platform, !best) with
      | 3, _ -> best := Some (utf16be_to_utf8 s (strings + so) len)
      | 1, None -> best := Some (String.sub s (strings + so) len)
      | _ -> ()
  done;
  Option.value !best ~default:""

(* Loading *)

let of_string s =
  if String.length s < 12 then malformed "too short";
  let version = u32 s 0 in
  if version = 0x4F54544F then unsupported "CFF outlines";
  if version <> 0x00010000 && version <> 0x74727565 then
    malformed "not a TrueType font";
  let tables = parse_directory s in
  let find tag =
    match Hashtbl.find_opt tables tag with
    | Some (off, _) -> off
    | None -> malformed "missing %s table" tag
  in
  let head = find "head" and hhea = find "hhea" and maxp = find "maxp" in
  let long_loca =
    match i16 s (head + 50) with
    | 0 -> false
    | 1 -> true
    | f -> malformed "loca format %d" f
  in
  let num_glyphs = u16 s (maxp + 4) in
  let cmap = parse_cmap s (find "cmap") in
  let kern =
    match Hashtbl.find_opt tables "GPOS" with
    | Some (off, _) -> parse_gpos s off
    | None -> []
  in
  let family =
    match Hashtbl.find_opt tables "name" with
    | Some (off, _) -> parse_family s off
    | None -> ""
  in
  let weight =
    match Hashtbl.find_opt tables "OS/2" with
    | Some (off, _) -> u16 s (off + 4)
    | None -> 400
  in
  {
    data = s;
    tables;
    units_per_em = u16 s (head + 18);
    ascender = i16 s (hhea + 4);
    descender = i16 s (hhea + 6);
    num_glyphs;
    num_hmetrics = u16 s (hhea + 34);
    hmtx = find "hmtx";
    loca = find "loca";
    long_loca;
    glyf = find "glyf";
    cmap;
    kern;
    family;
    weight;
  }

(* Glyph access *)

let glyph_of_uchar t u = Option.value (Hashtbl.find_opt t.cmap u) ~default:0

let advance t g =
  let g = if g >= t.num_glyphs then 0 else g in
  let i = if g < t.num_hmetrics then g else t.num_hmetrics - 1 in
  u16 t.data (t.hmtx + (4 * i))

(* [glyph_range t g] is the offset and length of glyph [g]'s data. *)
let glyph_range t g =
  let g = if g >= t.num_glyphs then 0 else g in
  let s = t.data in
  let a, b =
    if t.long_loca then (u32 s (t.loca + (4 * g)), u32 s (t.loca + (4 * g) + 4))
    else (2 * u16 s (t.loca + (2 * g)), 2 * u16 s (t.loca + (2 * g) + 2))
  in
  (t.glyf + a, b - a)

(* [bounds t g] is the glyph's bounding box in font units, or [None] for an
   empty glyph. *)
let bounds t g =
  let off, len = glyph_range t g in
  if len < 10 then None
  else
    let s = t.data in
    Some (i16 s (off + 2), i16 s (off + 4), i16 s (off + 6), i16 s (off + 8))

let kerning t g1 g2 =
  let s = t.data in
  let rec subtables = function
    | [] -> 0
    | Pairs { coverage; fmt1; fmt2; sets } :: rest ->
        let ci = coverage_index s coverage g1 in
        if ci < 0 then subtables rest
        else begin
          let base = sets - 10 in
          let set = base + u16 s (sets + (2 * ci)) in
          let count = u16 s set in
          let size = 2 + value_record_size fmt1 + value_record_size fmt2 in
          let lo = ref 0 and hi = ref (count - 1) and found = ref None in
          while !found = None && !lo <= !hi do
            let mid = (!lo + !hi) / 2 in
            let r = set + 2 + (size * mid) in
            let g = u16 s r in
            if g = g2 then found := Some (x_advance s (r + 2) fmt1)
            else if g < g2 then lo := mid + 1
            else hi := mid - 1
          done;
          match !found with Some v -> v | None -> subtables rest
        end
    | Classes { coverage; fmt1; fmt2; class1; class2; class2_count; records }
      :: rest ->
        if coverage_index s coverage g1 < 0 then subtables rest
        else begin
          let c1 = class_of s class1 g1 and c2 = class_of s class2 g2 in
          let size = value_record_size fmt1 + value_record_size fmt2 in
          x_advance s (records + (size * ((c1 * class2_count) + c2))) fmt1
        end
  in
  List.fold_left (fun acc lookup -> acc + subtables lookup) 0 t.kern

(* Outlines *)

(* Emits glyph [g]'s contours through [move], [line], [quad] and [close] in font
   units, with the affine [(a, b, c, d, dx, dy)] applied. Composite glyphs
   recurse into their components. *)
let rec outline t g (a, b, c, d, dx, dy) ~move ~line ~quad ~close ~depth =
  let off, len = glyph_range t g in
  if len >= 10 then begin
    let s = t.data in
    let contours = i16 s off in
    let tx x y = (a *. x) +. (c *. y) +. dx
    and ty x y = (b *. x) +. (d *. y) +. dy in
    if contours >= 0 then begin
      let ends = off + 10 in
      let n_points =
        if contours = 0 then 0 else u16 s (ends + (2 * (contours - 1))) + 1
      in
      let ins_len = u16 s (ends + (2 * contours)) in
      let flags = Bytes.create n_points in
      let p = ref (ends + (2 * contours) + 2 + ins_len) in
      let i = ref 0 in
      while !i < n_points do
        let f = u8 s !p in
        incr p;
        Bytes.set flags !i (Char.chr f);
        incr i;
        if f land 8 <> 0 then begin
          let repeat = u8 s !p in
          incr p;
          for _ = 1 to repeat do
            if !i < n_points then Bytes.set flags !i (Char.chr f);
            incr i
          done
        end
      done;
      let xs = Array.make n_points 0. and ys = Array.make n_points 0. in
      let v = ref 0 in
      for i = 0 to n_points - 1 do
        let f = Char.code (Bytes.get flags i) in
        if f land 2 <> 0 then begin
          let dx = u8 s !p in
          incr p;
          v := if f land 16 <> 0 then !v + dx else !v - dx
        end
        else if f land 16 = 0 then begin
          v := !v + i16 s !p;
          p := !p + 2
        end;
        xs.(i) <- float !v
      done;
      v := 0;
      for i = 0 to n_points - 1 do
        let f = Char.code (Bytes.get flags i) in
        if f land 4 <> 0 then begin
          let dy = u8 s !p in
          incr p;
          v := if f land 32 <> 0 then !v + dy else !v - dy
        end
        else if f land 32 = 0 then begin
          v := !v + i16 s !p;
          p := !p + 2
        end;
        ys.(i) <- float !v
      done;
      let start = ref 0 in
      for ci = 0 to contours - 1 do
        let stop = u16 s (ends + (2 * ci)) in
        let n = stop - !start + 1 in
        if n > 0 then begin
          let on i =
            Char.code (Bytes.get flags (!start + (i mod n))) land 1 <> 0
          in
          let px i = xs.(!start + (i mod n))
          and py i = ys.(!start + (i mod n)) in
          (* Start on an on-curve point, or on the midpoint of two off-curve
             points when the contour has none. *)
          let first_on =
            let rec find i =
              if i >= n then -1 else if on i then i else find (i + 1)
            in
            find 0
          in
          let sx, sy, first =
            if first_on >= 0 then (px first_on, py first_on, first_on)
            else ((px 0 +. px 1) /. 2., (py 0 +. py 1) /. 2., 0)
          in
          move (tx sx sy) (ty sx sy);
          let ctrl = ref None in
          let emit_to x y =
            (match !ctrl with
            | None -> line (tx x y) (ty x y)
            | Some (qx, qy) -> quad (tx qx qy) (ty qx qy) (tx x y) (ty x y));
            ctrl := None
          in
          (* Walk once around the contour back to the start point. *)
          for i = first + 1 to first + n do
            let x = px i and y = py i in
            if on i then emit_to x y
            else begin
              (match !ctrl with
              | Some (qx, qy) ->
                  let mx = (qx +. x) /. 2. and my = (qy +. y) /. 2. in
                  quad (tx qx qy) (ty qx qy) (tx mx my) (ty mx my)
              | None -> ());
              ctrl := Some (x, y)
            end
          done;
          (match !ctrl with
          | Some (qx, qy) -> quad (tx qx qy) (ty qx qy) (tx sx sy) (ty sx sy)
          | None -> ());
          close ()
        end;
        start := stop + 1
      done
    end
    else begin
      if depth > 8 then malformed "composite glyph nesting too deep";
      let p = ref (off + 10) in
      let continue = ref true in
      while !continue do
        let flags = u16 s !p and gi = u16 s (!p + 2) in
        p := !p + 4;
        let arg1, arg2 =
          if flags land 1 <> 0 then begin
            let a1 = i16 s !p and a2 = i16 s (!p + 2) in
            p := !p + 4;
            (a1, a2)
          end
          else begin
            let a1 = u8 s !p and a2 = u8 s (!p + 1) in
            p := !p + 2;
            ( (if a1 >= 128 then a1 - 256 else a1),
              if a2 >= 128 then a2 - 256 else a2 )
          end
        in
        if flags land 2 = 0 then
          unsupported "composite glyph with point matching";
        let f2dot14 o = float (i16 s o) /. 16384. in
        let ca, cb, cc, cd =
          if flags land 8 <> 0 then begin
            let sc = f2dot14 !p in
            p := !p + 2;
            (sc, 0., 0., sc)
          end
          else if flags land 0x40 <> 0 then begin
            let sx = f2dot14 !p and sy = f2dot14 (!p + 2) in
            p := !p + 4;
            (sx, 0., 0., sy)
          end
          else if flags land 0x80 <> 0 then begin
            let m =
              (f2dot14 !p, f2dot14 (!p + 2), f2dot14 (!p + 4), f2dot14 (!p + 6))
            in
            p := !p + 8;
            m
          end
          else (1., 0., 0., 1.)
        in
        let cdx = float arg1 and cdy = float arg2 in
        (* Compose the component transform under the parent's. *)
        let m =
          ( (a *. ca) +. (c *. cb),
            (b *. ca) +. (d *. cb),
            (a *. cc) +. (c *. cd),
            (b *. cc) +. (d *. cd),
            (a *. cdx) +. (c *. cdy) +. dx,
            (b *. cdx) +. (d *. cdy) +. dy )
        in
        outline t gi m ~move ~line ~quad ~close ~depth:(depth + 1);
        continue := flags land 0x20 <> 0
      done
    end
  end

let outline t g ~move ~line ~quad ~close =
  outline t g (1., 0., 0., 1., 0., 0.) ~move ~line ~quad ~close ~depth:0
