(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

exception Malformed of string
exception Unsupported of string

let malformed fmt = Printf.ksprintf (fun m -> raise (Malformed m)) fmt
let unsupported fmt = Printf.ksprintf (fun m -> raise (Unsupported m)) fmt

(* Bytes

   Reads are relative to the table's start and fail past its end, so a
   position read from the file, however large or negative, reads nothing
   outside the table. *)

type tbl = { s : string; off : int; len : int; tag : string }

let need t i stop =
  if i < 0 || stop > t.len then
    malformed "the '%s' table ends before byte %d" t.tag (t.off + stop)

let u8 t i =
  need t i (i + 1);
  Char.code (String.unsafe_get t.s (t.off + i))

let u16 t i =
  need t i (i + 2);
  String.get_uint16_be t.s (t.off + i)

let u32 t i =
  need t i (i + 4);
  Int32.to_int (String.get_int32_be t.s (t.off + i)) land 0xFFFF_FFFF

(* [uint t i size] is the big-endian unsigned integer of [size] bytes at
   [i]. *)
let uint t i size =
  let v = ref 0 in
  for k = 0 to size - 1 do
    v := (!v lsl 8) lor u8 t (i + k)
  done;
  !v

(* INDEX data: item [k] is the bytes from [base + offs.(k)] to
   [base + offs.(k + 1)]. CFF counts items in 16 bits and CFF2 in 32. *)

type index = { base : int; offs : int array }

let empty = { base = 0; offs = [| 0 |] }
let count ix = Array.length ix.offs - 1
let item ix k = (ix.base + ix.offs.(k), ix.base + ix.offs.(k + 1))

(* [index t ~cff2 what pos] is the INDEX at [pos] and the position after it. *)
let index t ~cff2 what pos =
  let n, pos = if cff2 then (u32 t pos, pos + 4) else (u16 t pos, pos + 2) in
  if n = 0 then (empty, pos)
  else
    let size = u8 t pos in
    if size < 1 || size > 4 then
      malformed "'%s': the %s INDEX has offsets of %d bytes" t.tag what size;
    let first = pos + 1 in
    (* The offsets lie in the table before any array is made for them. *)
    need t first (first + ((n + 1) * size));
    let offs = Array.init (n + 1) (fun k -> uint t (first + (k * size)) size) in
    if offs.(0) <> 1 then
      malformed "'%s': the %s INDEX starts at offset %d" t.tag what offs.(0);
    for k = 0 to n - 1 do
      if offs.(k + 1) < offs.(k) then
        malformed "'%s': the %s INDEX's offsets decrease" t.tag what
    done;
    let base = first + ((n + 1) * size) - 1 in
    need t base (base + offs.(n));
    ({ base; offs }, base + offs.(n))

(* DICT data

   A DICT is a list of operators with their operands. Operators are numbered
   as bytes, [12 x] as [1200 + x]. No operand this decoder reads is real, so a
   real is read as [nan], which no integer operand accepts. *)

let dict_stack = 513 (* CFF2's limit, which CFF's DICTs stay under. *)

let dict t (a, b) =
  let ops = ref [] and args = ref [] and n = ref 0 in
  let push v =
    if !n >= dict_stack then
      malformed "'%s': a DICT has more than %d operands" t.tag dict_stack;
    args := v :: !args;
    incr n
  in
  let i = ref a in
  while !i < b do
    let b0 = u8 t !i in
    if b0 <= 21 || (b0 >= 22 && b0 <= 27) || b0 = 31 then begin
      let op, len = if b0 = 12 then (1200 + u8 t (!i + 1), 2) else (b0, 1) in
      ops := (op, Array.of_list (List.rev !args)) :: !ops;
      args := [];
      n := 0;
      i := !i + len
    end
    else if b0 = 28 then begin
      let v = u16 t (!i + 1) in
      push (float (if v >= 0x8000 then v - 0x10000 else v));
      i := !i + 3
    end
    else if b0 = 29 then begin
      let v = u32 t (!i + 1) in
      push (float (if v >= 0x8000_0000 then v - 0x1_0000_0000 else v));
      i := !i + 5
    end
    else if b0 = 30 then begin
      (* A real's nibbles end at the first [0xf]. *)
      let rec skip j =
        let v = u8 t j in
        if v land 0xF = 0xF || v lsr 4 = 0xF then j + 1 else skip (j + 1)
      in
      i := skip (!i + 1);
      push Float.nan
    end
    else if b0 <= 246 then begin
      push (float (b0 - 139));
      incr i
    end
    else if b0 <= 250 then begin
      push (float (((b0 - 247) * 256) + u8 t (!i + 1) + 108));
      i := !i + 2
    end
    else if b0 <= 254 then begin
      push (float (-((b0 - 251) * 256) - u8 t (!i + 1) - 108));
      i := !i + 2
    end
    else malformed "'%s': a DICT holds the reserved byte 255" t.tag
  done;
  if !i > b then
    malformed "'%s': a DICT's last operand runs past its end" t.tag;
  List.rev !ops

let ints t op args =
  Array.map
    (fun v ->
      if Float.is_integer v then Float.to_int v
      else malformed "'%s': operator %d takes integers" t.tag op)
    args

(* [find t d op k] is the [k] integer operands of [op] in [d], if present. *)
let find t d op k =
  match List.assoc_opt op d with
  | None -> None
  | Some args ->
      let v = ints t op args in
      if Array.length v <> k then
        malformed "'%s': operator %d takes %d operands" t.tag op k;
      Some v

let offset t d op = Option.map (fun v -> v.(0)) (find t d op 1)

(* Decoded tables *)

type t = {
  tbl : tbl;
  cff2 : bool;
  charstrings : index;
  global : index;
  locals : index array; (* Per font DICT. *)
  vsindex : int array; (* Per font DICT, CFF2. *)
  fd_of : int array; (* Per glyph. *)
  regions : int array option; (* Per ItemVariationData, CFF2. *)
}

(* [private_dict t ~cff2 d] is the local subroutines and the vsindex of the
   Private DICT that the font DICT [d] names, or none. *)
let private_dict t ~cff2 d =
  match find t d 18 2 with
  | None -> (empty, 0)
  | Some [| size; off |] ->
      if size < 0 then malformed "'%s': a Private DICT of size %d" t.tag size;
      need t off (off + size);
      let p = dict t (off, off + size) in
      let local =
        match offset t p 19 with
        | None -> empty
        | Some rel -> fst (index t ~cff2 "Subrs" (off + rel))
      in
      (local, Option.value (offset t p 22) ~default:0)
  | Some _ -> assert false

(* [select t ~cff2 ~glyphs ~fds pos] is the font DICT of each glyph from the
   FDSelect at [pos]. *)
let select t ~cff2 ~glyphs ~fds pos =
  let fd_of = Array.make glyphs 0 in
  let check fd =
    if fd >= fds then malformed "'%s': FDSelect names font DICT %d" t.tag fd;
    fd
  in
  let ranges n ~head ~size ~first ~fd =
    if n = 0 then malformed "'%s': FDSelect has no ranges" t.tag;
    let at k = pos + head + (k * size) in
    if first (at 0) <> 0 then
      malformed "'%s': FDSelect starts after glyph 0" t.tag;
    for k = 0 to n - 1 do
      let lo = first (at k) and hi = first (at (k + 1)) in
      if hi <= lo || hi > glyphs then
        malformed "'%s': FDSelect ranges out of order" t.tag;
      Array.fill fd_of lo (hi - lo) (check (fd (at k)))
    done;
    if first (at n) <> glyphs then
      malformed "'%s': FDSelect ends at glyph %d of %d" t.tag (first (at n))
        glyphs
  in
  (match u8 t pos with
  | 0 ->
      need t (pos + 1) (pos + 1 + glyphs);
      for g = 0 to glyphs - 1 do
        fd_of.(g) <- check (u8 t (pos + 1 + g))
      done
  | 3 ->
      ranges (u16 t (pos + 1)) ~head:3 ~size:3 ~first:(u16 t)
        ~fd:(fun i -> u8 t (i + 2))
  | 4 when cff2 ->
      ranges (u32 t (pos + 1)) ~head:5 ~size:6 ~first:(u32 t)
        ~fd:(fun i -> u16 t (i + 4))
  | f -> malformed "'%s': FDSelect format %d" t.tag f);
  fd_of

(* [regions t pos] is the number of regions of each ItemVariationData of the
   VariationStore at [pos]. *)
let regions t pos =
  let store = pos + 2 in
  if u16 t store <> 1 then malformed "'%s': ItemVariationStore format" t.tag;
  let n = u16 t (store + 6) in
  Array.init n (fun k -> u16 t (store + u32 t (store + 8 + (4 * k)) + 4))

let decode ~cff2 ~glyphs s ~off ~len =
  let tbl = { s; off; len; tag = (if cff2 then "CFF2" else "CFF ") } in
  let t = tbl in
  let major = u8 t 0 in
  if major <> (if cff2 then 2 else 1) then
    unsupported "'%s' version %d" t.tag major;
  let top, global =
    if cff2 then
      let hdr = u8 t 2 and size = u16 t 3 in
      need t hdr (hdr + size);
      let global = fst (index t ~cff2 "global Subrs" (hdr + size)) in
      (dict t (hdr, hdr + size), global)
    else
      let _, pos = index t ~cff2 "Name" (u8 t 2) in
      let tops, pos = index t ~cff2 "Top DICT" pos in
      let _, pos = index t ~cff2 "String" pos in
      if count tops <> 1 then
        malformed "'%s': %d fonts in one table" t.tag (count tops);
      (dict t (item tops 0), fst (index t ~cff2 "global Subrs" pos))
  in
  (match find t top 1206 1 with
  | Some [| 2 |] | None -> ()
  | Some v -> unsupported "'%s': charstrings of type %d" t.tag v.(0));
  if List.mem_assoc 1220 top then unsupported "'%s': a synthetic font" t.tag;
  let charstrings =
    match offset t top 17 with
    | Some pos -> fst (index t ~cff2 "CharStrings" pos)
    | None -> malformed "'%s': no CharStrings" t.tag
  in
  if count charstrings <> glyphs then
    malformed "'%s': %d charstrings for %d glyphs" t.tag (count charstrings)
      glyphs;
  let cid = cff2 || List.mem_assoc 1230 top in
  let fds =
    if not cid then [| top |]
    else
      match offset t top 1236 with
      | None -> malformed "'%s': no FDArray" t.tag
      | Some pos ->
          let ix = fst (index t ~cff2 "FDArray" pos) in
          if count ix = 0 then malformed "'%s': no font DICT" t.tag;
          Array.init (count ix) (fun k -> dict t (item ix k))
  in
  let privates = Array.map (private_dict t ~cff2) fds in
  let fd_of =
    match offset t top 1237 with
    | Some pos when cid ->
        select t ~cff2 ~glyphs ~fds:(Array.length fds) pos
    | Some _ | None ->
        if cid && not cff2 then malformed "'%s': no FDSelect" t.tag;
        Array.make glyphs 0
  in
  let regions =
    if cff2 then Option.map (regions t) (offset t top 24) else None
  in
  let vsindex = Array.map snd privates in
  Option.iter
    (fun r ->
      Array.iter
        (fun v ->
          if v < 0 || v >= Array.length r then
            malformed "'%s': vsindex %d names no ItemVariationData" t.tag v)
        vsindex)
    regions;
  {
    tbl;
    cff2;
    charstrings;
    global;
    locals = Array.map fst privates;
    vsindex;
    fd_of;
    regions;
  }

(* Charstrings *)

type sink = {
  move : float -> float -> unit;
  line : float -> float -> unit;
  curve : float -> float -> float -> float -> float -> float -> unit;
  close : unit -> unit;
}

let max_stems = 96 (* Type 2's limit on stem hints. *)
let max_depth = 10 (* The nesting of subroutines. *)
let max_bytes = 65535 (* The most bytes a charstring holds. *)
let transient = 32 (* Elements of the transient array. *)

type state = {
  c : t;
  g : int;
  local : index;
  stack : float array;
  mutable sp : int;
  mutable x : float;
  mutable y : float;
  mutable started : bool; (* A contour is open. *)
  mutable cleared : bool; (* A stack-clearing operator ran, so no width. *)
  mutable stems : int;
  mutable k : int; (* Regions a blend blends, CFF2. *)
  mutable ran : int; (* Bytes run, subroutines expanded. *)
  store : float option array;
  sink : sink;
}

exception End

let fail st fmt = Printf.ksprintf (fun m -> malformed "glyph %d: %s" st.g m) fmt

let push st v =
  if st.sp >= Array.length st.stack then
    fail st "more than %d arguments on the stack" (Array.length st.stack);
  st.stack.(st.sp) <- v;
  st.sp <- st.sp + 1

let pop st =
  if st.sp = 0 then fail st "an operator takes more arguments than it has";
  st.sp <- st.sp - 1;
  st.stack.(st.sp)

let int_arg st v =
  if Float.is_integer v && Float.abs v < 1e9 then Float.to_int v
  else fail st "an integer argument is %g" v

(* Arithmetic results are 16.16 numbers, as charstring numbers are; a result
   out of their range is an overflow, which Type 2 leaves undefined. *)
let fixed st v =
  let r = Float.round (v *. 65536.) /. 65536. in
  if Float.is_nan r || r < -32768. || r >= 32768. then
    fail st "an arithmetic result %g out of range" v;
  r

(* The first stack-clearing operator of a CFF charstring takes the glyph's
   width as one more argument, which outlines do not need: [width st odd]
   drops it if [odd sp] says it is there. *)
let width st odd =
  if (not st.cleared) && not st.c.cff2 then begin
    if st.sp > 0 && odd st.sp then begin
      Array.blit st.stack 1 st.stack 0 (st.sp - 1);
      st.sp <- st.sp - 1
    end;
    st.cleared <- true
  end

let clear st =
  st.sp <- 0;
  st.cleared <- true

let close st =
  if st.started then begin
    st.sink.close ();
    st.started <- false
  end

let move st dx dy =
  close st;
  st.x <- st.x +. dx;
  st.y <- st.y +. dy;
  st.sink.move st.x st.y;
  st.started <- true

let drawing st = if not st.started then fail st "it draws before a moveto"

let line st dx dy =
  drawing st;
  st.x <- st.x +. dx;
  st.y <- st.y +. dy;
  st.sink.line st.x st.y

let curve st dxa dya dxb dyb dxc dyc =
  drawing st;
  let xa = st.x +. dxa and ya = st.y +. dya in
  let xb = xa +. dxb and yb = ya +. dyb in
  st.x <- xb +. dxc;
  st.y <- yb +. dyc;
  st.sink.curve xa ya xb yb st.x st.y

let arg st i = st.stack.(i)

let args st ok what =
  if not (ok st.sp) then fail st "%s with %d arguments" what st.sp

(* [lines st ~horizontal] draws lines that alternate between horizontal and
   vertical, starting as [horizontal] says. *)
let lines st ~horizontal =
  args st (fun n -> n >= 1) "a lineto";
  for i = 0 to st.sp - 1 do
    let v = arg st i in
    if horizontal = (i mod 2 = 0) then line st v 0. else line st 0. v
  done

(* [bends st ~vertical] draws the curves of [vhcurveto] and [hvcurveto]:
   each starts along one axis and ends along the other, the last one's final
   argument, if any, being its free end coordinate. *)
let bends st ~vertical =
  args st (fun n -> n >= 4 && (n mod 4 = 0 || n mod 4 = 1)) "a curve";
  let n = st.sp in
  let i = ref 0 and vertical = ref vertical in
  while n - !i >= 4 do
    let a k = arg st (!i + k) in
    let last = if n - !i = 5 then a 4 else 0. in
    if !vertical then curve st 0. (a 0) (a 1) (a 2) (a 3) last
    else curve st (a 0) 0. (a 1) (a 2) last (a 3);
    i := !i + if n - !i = 5 then 5 else 4;
    vertical := not !vertical
  done

(* [flats st ~vertical] draws [vvcurveto] and [hhcurveto] curves, which start
   and end along one axis, the first possibly leaving it. *)
let flats st ~vertical =
  args st (fun n -> n >= 4 && (n mod 4 = 0 || n mod 4 = 1)) "a curve";
  let start = st.sp mod 4 in
  let off = if start = 1 then arg st 0 else 0. in
  let i = ref start in
  while !i < st.sp do
    let a k = arg st (!i + k) in
    let off' = if !i = start then off else 0. in
    if vertical then curve st off' (a 0) (a 1) (a 2) 0. (a 3)
    else curve st (a 0) off' (a 1) (a 2) (a 3) 0.;
    i := !i + 4
  done

let bias n = if n < 1240 then 107 else if n < 33900 then 1131 else 32768

let truth b = if b then 1. else 0.

let rec exec st (a, b) depth =
  let t = st.c.tbl and cff2 = st.c.cff2 in
  let i = ref a in
  let ran n =
    st.ran <- st.ran + n;
    if st.ran > max_bytes then
      fail st "it runs more than %d bytes of charstrings" max_bytes;
    i := !i + n
  in
  let reserved () =
    (* CFF2 ignores an operator it does not know and clears the stack. *)
    if cff2 then clear st else fail st "it holds a reserved operator"
  in
  let returned = ref false in
  while (not !returned) && !i < b do
    let b0 = u8 t !i in
    if b0 >= 32 then begin
      if b0 <= 246 then begin
        push st (float (b0 - 139));
        ran 1
      end
      else if b0 <= 250 then begin
        push st (float (((b0 - 247) * 256) + u8 t (!i + 1) + 108));
        ran 2
      end
      else if b0 <= 254 then begin
        push st (float (-((b0 - 251) * 256) - u8 t (!i + 1) - 108));
        ran 2
      end
      else begin
        let v = u32 t (!i + 1) in
        let v = if v >= 0x8000_0000 then v - 0x1_0000_0000 else v in
        push st (float v /. 65536.);
        ran 5
      end
    end
    else if b0 = 28 then begin
      let v = u16 t (!i + 1) in
      push st (float (if v >= 0x8000 then v - 0x10000 else v));
      ran 3
    end
    else if b0 = 12 then begin
      let b1 = u8 t (!i + 1) in
      ran 2;
      escape st b1
    end
    else begin
      ran 1;
      match b0 with
      | 1 | 3 | 18 | 23 ->
          (* hstem, vstem, hstemhm, vstemhm *)
          width st (fun n -> n mod 2 = 1);
          stems st;
          clear st
      | 19 | 20 ->
          (* hintmask, cntrmask, after implicit vstems *)
          width st (fun n -> n mod 2 = 1);
          stems st;
          clear st;
          let bytes = (st.stems + 7) / 8 in
          if !i + bytes > b then fail st "a hint mask runs past its end";
          ran bytes
      | 21 ->
          width st (fun n -> n > 2);
          args st (( = ) 2) "rmoveto";
          move st (arg st 0) (arg st 1);
          clear st
      | 22 ->
          width st (fun n -> n > 1);
          args st (( = ) 1) "hmoveto";
          move st (arg st 0) 0.;
          clear st
      | 4 ->
          width st (fun n -> n > 1);
          args st (( = ) 1) "vmoveto";
          move st 0. (arg st 0);
          clear st
      | 5 ->
          args st (fun n -> n >= 2 && n mod 2 = 0) "rlineto";
          for k = 0 to (st.sp / 2) - 1 do
            line st (arg st (2 * k)) (arg st ((2 * k) + 1))
          done;
          clear st
      | 6 ->
          lines st ~horizontal:true;
          clear st
      | 7 ->
          lines st ~horizontal:false;
          clear st
      | 8 ->
          args st (fun n -> n >= 6 && n mod 6 = 0) "rrcurveto";
          for k = 0 to (st.sp / 6) - 1 do
            let a j = arg st ((6 * k) + j) in
            curve st (a 0) (a 1) (a 2) (a 3) (a 4) (a 5)
          done;
          clear st
      | 24 ->
          args st (fun n -> n >= 8 && (n - 2) mod 6 = 0) "rcurveline";
          let curves = (st.sp - 2) / 6 in
          for k = 0 to curves - 1 do
            let a j = arg st ((6 * k) + j) in
            curve st (a 0) (a 1) (a 2) (a 3) (a 4) (a 5)
          done;
          line st (arg st (6 * curves)) (arg st ((6 * curves) + 1));
          clear st
      | 25 ->
          args st (fun n -> n >= 8 && (n - 6) mod 2 = 0) "rlinecurve";
          let lines = (st.sp - 6) / 2 in
          for k = 0 to lines - 1 do
            line st (arg st (2 * k)) (arg st ((2 * k) + 1))
          done;
          let a j = arg st ((2 * lines) + j) in
          curve st (a 0) (a 1) (a 2) (a 3) (a 4) (a 5);
          clear st
      | 26 ->
          flats st ~vertical:true;
          clear st
      | 27 ->
          flats st ~vertical:false;
          clear st
      | 30 ->
          bends st ~vertical:true;
          clear st
      | 31 ->
          bends st ~vertical:false;
          clear st
      | 10 | 29 ->
          let subrs = if b0 = 10 then st.local else st.c.global in
          let n = count subrs in
          let k = int_arg st (pop st) + bias n in
          if k < 0 || k >= n then fail st "it calls subroutine %d of %d" k n;
          if depth >= max_depth then
            fail st "subroutines nest deeper than %d" max_depth;
          exec st (item subrs k) (depth + 1)
      | 11 when not cff2 -> returned := true
      | 14 when not cff2 ->
          width st (fun n -> n = 1 || n = 5);
          if st.sp = 4 then
            unsupported "glyph %d: an accented glyph built by endchar" st.g;
          args st (( = ) 0) "endchar";
          close st;
          raise End
      | 15 when cff2 ->
          let ivd = int_arg st (pop st) in
          (match st.c.regions with
          | Some r when ivd >= 0 && ivd < Array.length r -> st.k <- r.(ivd)
          | _ -> fail st "vsindex %d names no ItemVariationData" ivd);
          clear st
      | 16 when cff2 ->
          if Option.is_none st.c.regions then
            fail st "it blends without a VariationStore";
          let n = int_arg st (pop st) in
          let base = st.sp - (n * (st.k + 1)) in
          if n < 0 || base < 0 then fail st "a blend of %d values" n;
          (* The default instance keeps the defaults and drops the deltas. *)
          st.sp <- base + n
      | _ -> reserved ()
    end
  done;
  if !i > b then fail st "a number runs past the end of its charstring"

and stems st =
  if st.sp mod 2 = 1 then fail st "a stem hint lacks its width";
  st.stems <- st.stems + (st.sp / 2);
  if (not st.c.cff2) && st.stems > max_stems then
    fail st "more than %d stem hints" max_stems

and escape st b1 =
  let cff2 = st.c.cff2 in
  let unary f =
    let v = pop st in
    push st (f v)
  and binary f =
    let b = pop st in
    let a = pop st in
    push st (f a b)
  in
  let fx = fixed st in
  match b1 with
  | 35 ->
      args st (( = ) 13) "flex";
      let a = arg st in
      curve st (a 0) (a 1) (a 2) (a 3) (a 4) (a 5);
      curve st (a 6) (a 7) (a 8) (a 9) (a 10) (a 11);
      clear st
  | 34 ->
      args st (( = ) 7) "hflex";
      let a = arg st in
      curve st (a 0) 0. (a 1) (a 2) (a 3) 0.;
      curve st (a 4) 0. (a 5) (-.a 2) (a 6) 0.;
      clear st
  | 36 ->
      args st (( = ) 9) "hflex1";
      let a = arg st in
      curve st (a 0) (a 1) (a 2) (a 3) (a 4) 0.;
      curve st (a 5) 0. (a 6) (a 7) (a 8) (-.(a 1 +. a 3 +. a 7));
      clear st
  | 37 ->
      args st (( = ) 11) "flex1";
      let a = arg st in
      let dx = a 0 +. a 2 +. a 4 +. a 6 +. a 8
      and dy = a 1 +. a 3 +. a 5 +. a 7 +. a 9 in
      let dx6, dy6 =
        if Float.abs dx > Float.abs dy then (a 10, -.dy) else (-.dx, a 10)
      in
      curve st (a 0) (a 1) (a 2) (a 3) (a 4) (a 5);
      curve st (a 6) (a 7) (a 8) (a 9) dx6 dy6;
      clear st
  | _ when cff2 -> clear st
  | 0 -> clear st (* dotsection, a no-op *)
  | 3 -> binary (fun a b -> truth (a <> 0. && b <> 0.))
  | 4 -> binary (fun a b -> truth (a <> 0. || b <> 0.))
  | 5 -> unary (fun a -> truth (a = 0.))
  | 9 -> unary Float.abs
  | 10 -> binary (fun a b -> fx (a +. b))
  | 11 -> binary (fun a b -> fx (a -. b))
  | 12 ->
      binary (fun a b ->
          if b = 0. then fail st "it divides by zero" else fx (a /. b))
  | 14 -> unary (fun a -> fx (-.a))
  | 15 -> binary (fun a b -> truth (a = b))
  | 18 -> ignore (pop st)
  | 20 ->
      let k = int_arg st (pop st) in
      let v = pop st in
      if k < 0 || k >= transient then fail st "it puts at %d" k;
      st.store.(k) <- Some v
  | 21 -> (
      let k = int_arg st (pop st) in
      match if k < 0 || k >= transient then None else st.store.(k) with
      | Some v -> push st v
      | None -> fail st "it gets %d before putting it" k)
  | 22 ->
      let v2 = pop st in
      let v1 = pop st in
      let s2 = pop st in
      let s1 = pop st in
      push st (if v1 <= v2 then s1 else s2)
  | 23 -> unsupported "glyph %d: the random operator" st.g
  | 24 -> binary (fun a b -> fx (a *. b))
  | 26 ->
      unary (fun a ->
          if a < 0. then fail st "a square root of %g" a else fx (Float.sqrt a))
  | 27 ->
      let v = pop st in
      push st v;
      push st v
  | 28 ->
      let b = pop st in
      let a = pop st in
      push st b;
      push st a
  | 29 ->
      let k = int_arg st (pop st) in
      let k = if k < 0 then 0 else k in
      if k >= st.sp then fail st "it indexes %d of %d" k st.sp;
      push st st.stack.(st.sp - 1 - k)
  | 30 ->
      let j = int_arg st (pop st) in
      let n = int_arg st (pop st) in
      if n < 0 || n > st.sp then fail st "it rolls %d of %d" n st.sp;
      if n > 0 then begin
        let base = st.sp - n in
        let old = Array.sub st.stack base n in
        for k = 0 to n - 1 do
          st.stack.(base + k) <- old.((((k - j) mod n) + n) mod n)
        done
      end
  | _ -> fail st "it holds the reserved operator 12 %d" b1

let run c g sink =
  let fd = c.fd_of.(g) in
  let st =
    {
      c;
      g;
      local = c.locals.(fd);
      stack = Array.make (if c.cff2 then 513 else 48) 0.;
      sp = 0;
      x = 0.;
      y = 0.;
      started = false;
      cleared = false;
      stems = 0;
      k = (match c.regions with Some r -> r.(c.vsindex.(fd)) | None -> 0);
      ran = 0;
      store = Array.make transient None;
      sink;
    }
  in
  match exec st (item c.charstrings g) 0 with
  | () ->
      if not c.cff2 then fail st "its charstring does not end with endchar";
      close st
  | exception End -> ()

(* Writing charstrings *)

let add_byte b v = Buffer.add_char b (Char.unsafe_chr v)

(* [number b v] writes [v], a multiple of 2⁻¹⁶, exactly: an integer in the
   fewest bytes, a fraction as a 16.16 number, and a value out of their range
   as a sum of in-range parts with [add]. *)
let rec number b v =
  if Float.is_integer v && Float.abs v <= 32767. then begin
    let n = Float.to_int v in
    if n >= -107 && n <= 107 then add_byte b (n + 139)
    else if n >= 108 && n <= 1131 then begin
      add_byte b (((n - 108) lsr 8) + 247);
      add_byte b ((n - 108) land 0xFF)
    end
    else if n <= -108 && n >= -1131 then begin
      add_byte b (((-n - 108) lsr 8) + 251);
      add_byte b ((-n - 108) land 0xFF)
    end
    else begin
      add_byte b 28;
      Buffer.add_uint16_be b (n land 0xFFFF)
    end
  end
  else if v >= -32768. && v < 32768. then begin
    add_byte b 255;
    Buffer.add_int32_be b (Int32.of_float (Float.round (v *. 65536.)))
  end
  else begin
    let part = if v > 0. then 32767. else -32767. in
    number b part;
    number b (v -. part);
    add_byte b 12;
    add_byte b 10
  end

(* At most this many arguments go to one operator, so that a sum's parts fit
   on CFF's stack of 48 beside them. *)
let batch = 42

(* [encoder g] is a sink that writes what it is given as glyph [g]'s
   charstring, and a function that ends and returns it. *)
let encoder g =
  let b = Buffer.create 256 and args = Buffer.create 64 in
  let x = ref 0. and y = ref 0. in
  let op = ref 0 and n = ref 0 in
  let flush () =
    if !n > 0 then begin
      Buffer.add_buffer b args;
      add_byte b !op;
      Buffer.clear args;
      n := 0
    end
  in
  let point x' y' =
    number args (x' -. !x);
    number args (y' -. !y);
    x := x';
    y := y'
  in
  (* [segment o k] starts an argument batch of operator [o], [k] arguments
     a segment. *)
  let segment o k =
    if !op <> o || !n + k > batch then flush ();
    op := o;
    n := !n + k
  in
  let sink =
    {
      move =
        (fun x' y' ->
          flush ();
          segment 21 2;
          point x' y';
          flush ());
      line =
        (fun x' y' ->
          segment 5 2;
          point x' y');
      curve =
        (fun xa ya xb yb xc yc ->
          segment 8 6;
          point xa ya;
          point xb yb;
          point xc yc);
      close = flush;
    }
  in
  let finish () =
    flush ();
    add_byte b 14;
    if Buffer.length b > max_bytes then
      unsupported "glyph %d: its outline takes more than %d bytes of charstring"
        g max_bytes;
    Buffer.contents b
  in
  (sink, finish)

let charstring c g =
  let sink, finish = encoder g in
  run c g sink;
  finish ()

let check c g sink =
  let e, finish = encoder g in
  run c g
    {
      move =
        (fun x y ->
          e.move x y;
          sink.move x y);
      line =
        (fun x y ->
          e.line x y;
          sink.line x y);
      curve =
        (fun xa ya xb yb x y ->
          e.curve xa ya xb yb x y;
          sink.curve xa ya xb yb x y);
      close =
        (fun () ->
          e.close ();
          sink.close ());
    };
  ignore (finish ())

(* Writing tables *)

let index_bytes items =
  let b = Buffer.create 256 in
  let n = List.length items in
  Buffer.add_uint16_be b n;
  if n > 0 then begin
    let total = List.fold_left (fun a s -> a + String.length s) 1 items in
    let size =
      if total < 0x100 then 1
      else if total < 0x10000 then 2
      else if total < 0x1000000 then 3
      else 4
    in
    add_byte b size;
    let put v =
      for k = size - 1 downto 0 do
        add_byte b ((v lsr (8 * k)) land 0xFF)
      done
    in
    put 1;
    ignore
      (List.fold_left
         (fun o s ->
           let o = o + String.length s in
           put o;
           o)
         1 items);
    List.iter (Buffer.add_string b) items
  end;
  Buffer.contents b

(* An offset in a DICT is written in five bytes, so that a DICT's length does
   not depend on the offsets it holds. *)
let int5 b v =
  add_byte b 29;
  Buffer.add_int32_be b (Int32.of_int v)

let small b v = number b (float v)

let escaped b op =
  add_byte b 12;
  add_byte b op

(* A FontName holds printable ASCII other than the PostScript delimiters. *)
let font_name s =
  let ok c = c > ' ' && c <= '~' && not (String.contains "[](){}<>/%" c) in
  let s = String.of_seq (Seq.filter ok (String.to_seq s)) in
  let s = if String.length s > 127 then String.sub s 0 127 else s in
  if s = "" then "Font" else s

let std_strings = 391 (* SIDs below this name standard strings. *)

let table ~name charstrings =
  let n = Array.length charstrings in
  let top ~charset ~fdselect ~chars ~fdarray =
    let b = Buffer.create 64 in
    small b std_strings;
    small b (std_strings + 1);
    small b 0;
    escaped b 30;
    small b n;
    escaped b 34;
    int5 b charset;
    add_byte b 15;
    int5 b fdselect;
    escaped b 37;
    int5 b chars;
    add_byte b 17;
    int5 b fdarray;
    escaped b 36;
    Buffer.contents b
  in
  let header = "\001\000\004\004" in
  let names = index_bytes [ font_name name ] in
  let strings = index_bytes [ "Adobe"; "Identity" ] in
  let globals = index_bytes [] in
  let charset =
    if n <= 1 then "\002"
    else
      let b = Buffer.create 5 in
      add_byte b 2;
      Buffer.add_uint16_be b 1;
      Buffer.add_uint16_be b (n - 2);
      Buffer.contents b
  in
  let fdselect =
    let b = Buffer.create 8 in
    add_byte b 3;
    Buffer.add_uint16_be b 1;
    Buffer.add_uint16_be b 0;
    add_byte b 0;
    Buffer.add_uint16_be b n;
    Buffer.contents b
  in
  let chars = index_bytes (Array.to_list charstrings) in
  (* defaultWidthX 0, nominalWidthX 0 *)
  let private_dict = "\139\020\139\021" in
  let font_dict p =
    let b = Buffer.create 12 in
    int5 b (String.length private_dict);
    int5 b p;
    add_byte b 18;
    Buffer.contents b
  in
  let fdarray p = index_bytes [ font_dict p ] in
  let tops dummy = index_bytes [ dummy ] in
  let zero = top ~charset:0 ~fdselect:0 ~chars:0 ~fdarray:0 in
  let start =
    String.length header + String.length names
    + String.length (tops zero)
    + String.length strings + String.length globals
  in
  let charset_at = start in
  let fdselect_at = charset_at + String.length charset in
  let chars_at = fdselect_at + String.length fdselect in
  let fdarray_at = chars_at + String.length chars in
  let private_at = fdarray_at + String.length (fdarray 0) in
  String.concat ""
    [
      header;
      names;
      tops
        (top ~charset:charset_at ~fdselect:fdselect_at ~chars:chars_at
           ~fdarray:fdarray_at);
      strings;
      globals;
      charset;
      fdselect;
      chars;
      fdarray private_at;
      private_dict;
    ]
