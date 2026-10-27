(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every layout is in canonical form: an axis of extent 1 has stride 0, and a
   layout with no element has offset 0 and every stride 0. [finish] puts any
   layout in it; [contiguous] builds C order in it directly.

   C reads the fields in this order (nx_array_stubs.c): the two change
   together. *)

type t = {
  shape : int iarray;
  strides : int iarray;
  offset : int;
  flags : int;
  lo : int;
  hi : int;
}

let max_rank = Shape.max_rank
let max_numel = Shape.max_numel
let fits = Shape.fits
let contiguous_flag = 1
let distinct_flag = 2
let empty_flag = 4
let rank l = Iarray.length l.shape
let flags l = l.flags
let offset l = l.offset
let span l = (l.lo, l.hi)
let shape l = l.shape
let strides l = l.strides
let unsafe_dim l i = Iarray.unsafe_get l.shape i
let unsafe_stride l i = Iarray.unsafe_get l.strides i
let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let dim l i =
  if i < 0 || i >= rank l then
    invalid_argf "Layout.dim: axis %d of %d" i (rank l);
  unsafe_dim l i

let stride l i =
  if i < 0 || i >= rank l then
    invalid_argf "Layout.stride: axis %d of %d" i (rank l);
  unsafe_stride l i

let numel l =
  if flags l land empty_flag <> 0 then 0
  else
    let n = ref 1 in
    for i = 0 to rank l - 1 do
      n := !n * unsafe_dim l i
    done;
    !n

let is_contiguous l = flags l land contiguous_flag <> 0
let is_distinct l = flags l land distinct_flag <> 0

let ints_equal (a : int iarray) (b : int iarray) =
  let n = Iarray.length a in
  n = Iarray.length b
  &&
  let i = ref 0 in
  while !i < n && Iarray.unsafe_get a !i = Iarray.unsafe_get b !i do
    incr i
  done;
  !i = n

let equal l l' =
  l == l'
  || l.offset = l'.offset
     && ints_equal l.shape l'.shape
     && ints_equal l.strides l'.strides

(* The fields [equal] compares, mixed into one int: no tuple to hash. *)
let hash l =
  let h = ref l.offset in
  for i = 0 to rank l - 1 do
    h := (!h * 31) + unsafe_dim l i;
    h := (!h * 31) + unsafe_stride l i
  done;
  Hashtbl.hash !h

(* Building *)

let negative fn = invalid_argf "%s: the layout reaches a negative position" fn

let past fn =
  invalid_argf "%s: the layout's span ends past max_numel (%d)" fn max_numel

(* Whether [s] has an extent 0. A loop: Iarray.mem allocates the closure of
   its own. *)
let has_zero s =
  let r = Iarray.length s and i = ref 0 in
  while !i < r && Iarray.unsafe_get s !i <> 0 do
    incr i
  done;
  !i < r

(* [strides] in canonical form for [shape], which has an extent 0 if [empty]:
   the stride of an axis of extent 1 is 0, and so is every stride if [empty].
   It is [strides] itself where they already are. *)
let canonical ~empty shape strides =
  let r = Iarray.length shape in
  let i = ref 0 in
  while
    !i < r
    && (Iarray.unsafe_get strides !i = 0
       || not (empty || Iarray.unsafe_get shape !i = 1))
  do
    incr i
  done;
  if !i = r then strides
  else begin
    let st = Shape.zeros r in
    if not empty then
      for i = 0 to r - 1 do
        if Iarray.unsafe_get shape i <> 1 then
          Array.unsafe_set st i (Iarray.unsafe_get strides i)
      done;
    Shape.freeze st
  end

(* The layout of [shape], [strides] and [offset] in canonical form, with its
   flags and span. Extents are non-negative and their product is admitted
   (Shape.numel); a stride of an axis of extent above 1 times the extent fits
   in an int. The span is checked against [max_numel]. *)
let finish fn shape strides offset =
  let r = Iarray.length shape in
  let empty = has_zero shape in
  let strides = canonical ~empty shape strides in
  if empty then
    {
      shape;
      strides;
      offset = 0;
      flags = contiguous_flag lor distinct_flag lor empty_flag;
      lo = 0;
      hi = 0;
    }
  else begin
    (* The span, from the offset and each axis's reach; C order, where stride i
       is the product of the later extents, or 0 for an axis of extent 1. The
       offset and every reach lie within the span, so each is bounded first:
       then their sum, of at most 33 terms below 2^53, fits. *)
    if offset < 0 then negative fn;
    if offset >= max_numel then past fn;
    let lo = ref offset and hi = ref offset in
    let contiguous = ref true and run = ref 1 in
    for i = r - 1 downto 0 do
      let d = Iarray.unsafe_get shape i and st = Iarray.unsafe_get strides i in
      let reach = (d - 1) * st in
      if reach <= -max_numel then negative fn;
      if reach >= max_numel then past fn;
      if reach < 0 then lo := !lo + reach else hi := !hi + reach;
      if st <> if d = 1 then 0 else !run then contiguous := false;
      run := !run * d
    done;
    (* A position counts elements from a buffer's first byte: a negative one
       lies outside every buffer. With [0 <= lo] and [hi < max_numel], every
       reach below sums to at most [hi - 1 - lo]. *)
    if !lo < 0 then negative fn;
    if !hi >= max_numel then past fn;
    (* Distinct: each axis of extent above 1 has a stride above the reach of the
       axes of smaller stride, ties broken by axis. *)
    let distinct = ref true in
    for j = 0 to r - 1 do
      let dj = Iarray.unsafe_get shape j in
      if dj > 1 then begin
        let sj = abs (Iarray.unsafe_get strides j) and below = ref 0 in
        for k = 0 to r - 1 do
          let dk = Iarray.unsafe_get shape k
          and sk = abs (Iarray.unsafe_get strides k) in
          if dk > 1 && (sk < sj || (sk = sj && k < j)) then
            below := !below + ((dk - 1) * sk)
        done;
        if sj <= !below then distinct := false
      end
    done;
    let flags =
      (if !contiguous then contiguous_flag else 0)
      lor if !distinct then distinct_flag else 0
    in
    { shape; strides; offset; flags; lo = !lo; hi = !hi + 1 }
  end

(* Constructors *)

(* A C-order layout is canonical, contiguous and distinct, with span [[0, n)]:
   [contiguous] builds it without [finish]'s general checks. *)
let contiguous shape =
  let r = Iarray.length shape in
  Shape.check_rank "Layout.contiguous" r;
  let n = Shape.numel "Layout.contiguous" shape in
  let strides = Shape.zeros r in
  let run = ref 1 in
  for i = r - 1 downto 0 do
    let d = Iarray.unsafe_get shape i in
    if n > 0 && d > 1 then Array.unsafe_set strides i !run;
    run := !run * d
  done;
  let strides = Shape.freeze strides in
  let flags = contiguous_flag lor distinct_flag in
  if n = 0 then
    { shape; strides; offset = 0; flags = flags lor empty_flag; lo = 0; hi = 0 }
  else { shape; strides; offset = 0; flags; lo = 0; hi = n }

let v ?(offset = 0) ~strides s =
  let r = Iarray.length s in
  Shape.check_rank "Layout.v" r;
  if Iarray.length strides <> r then
    invalid_argf "Layout.v: %d strides for %d axes" (Iarray.length strides) r;
  ignore (Shape.numel "Layout.v" s);
  for i = 0 to r - 1 do
    let d = Iarray.get s i and st = Iarray.get strides i in
    (* As in Shape.numel, the division runs only for factors past 2^31. *)
    if
      d > 1
      && (st = min_int
         || (abs st lor (d - 1) >= 1 lsl 31 && abs st > max_int / (d - 1)))
    then
      invalid_argf
        "Layout.v: axis %d has extent %d and stride %d: its reach overflows" i d
        st
  done;
  finish "Layout.v" s strides offset

(* Movements. Each writes the strides of the result, of shape [s'], from [l]'s;
   [finish] makes them canonical. A stride is formed only for an axis of extent
   above 1, where the result's span, inside [l]'s, bounds it. *)

let moved = finish "Layout.move"

(* The first axis of [s] from [i] whose extent is not 1, or [s]'s rank. *)
let past_ones s i =
  let i = ref i in
  while !i < Iarray.length s && Iarray.unsafe_get s !i = 1 do
    incr i
  done;
  !i

(* Strides for [s'] over the same elements in the same C order as [l], if
   strides express it. Axes of extent 1 are left out on both sides; the others
   are matched in groups of equal product, and a group of [l]'s axes must lay
   out one run. *)
let reshape l s' =
  let r = rank l in
  let strides = Shape.zeros (Iarray.length s') in
  (* [i] and [j] walk [l]'s axes and [s']'s, skipping extents of 1. *)
  let i = ref (past_ones l.shape 0) and j = ref (past_ones s' 0) in
  let runs = ref true in
  while !runs && !i < r do
    (* Grow a group of each side's axes until their products are equal.
       Move.shape checked that the shapes have one number of elements, so
       neither side runs out first. *)
    let i0 = !i and j0 = !j and po = ref 1 and pn = ref 1 and last = ref !i in
    while not (!po = !pn && !i > i0) do
      if !po <= !pn then begin
        let a = !i in
        if a > i0 && unsafe_stride l !last <> unsafe_stride l a * unsafe_dim l a
        then runs := false;
        po := !po * unsafe_dim l a;
        last := a;
        i := past_ones l.shape (a + 1)
      end
      else begin
        pn := !pn * Iarray.unsafe_get s' !j;
        j := past_ones s' (!j + 1)
      end
    done;
    (* The group of [l] lays out one run: the new axes split it in C order. *)
    let st = ref (unsafe_stride l !last) in
    for k = !j - 1 downto j0 do
      let d = Iarray.unsafe_get s' k in
      if d > 1 then begin
        Array.unsafe_set strides k !st;
        st := !st * d
      end
    done
  done;
  if !runs then Some (moved s' (Shape.freeze strides) (offset l)) else None

let move m l =
  let s' = Move.shape m l.shape in
  let r = rank l and r' = Iarray.length s' in
  let st = Shape.zeros r' in
  if numel l = 0 || has_zero s' then Some (moved s' (Shape.freeze st) 0)
  else
    match m with
    | Move.Reshape _ -> reshape l s'
    | Broadcast _ ->
        for i = 0 to r' - 1 do
          let j = i - r' + r in
          st.(i) <- (if j < 0 then 0 else unsafe_stride l j)
        done;
        Some (moved s' (Shape.freeze st) (offset l))
    | Permute p ->
        for i = 0 to r' - 1 do
          st.(i) <- unsafe_stride l (Iarray.unsafe_get p i)
        done;
        Some (moved s' (Shape.freeze st) (offset l))
    | Slice rs ->
        let offset = ref (offset l) in
        for i = 0 to r - 1 do
          let x = Iarray.unsafe_get rs i and s = unsafe_stride l i in
          offset := !offset + (x.start * s);
          st.(i) <- (if x.count > 1 then s * x.step else 0)
        done;
        Some (moved s' (Shape.freeze st) !offset)
    | Window ws ->
        (* Axis [w.axis] steps from window to window; the appended axis [r + j]
           steps within window [j]. *)
        for i = 0 to r - 1 do
          st.(i) <- unsafe_stride l i
        done;
        Iarray.iteri
          (fun j (w : Move.window) ->
            let s = unsafe_stride l w.axis in
            st.(w.axis) <-
              (if Iarray.unsafe_get s' w.axis > 1 then s * w.step else 0);
            st.(r + j) <- (if w.size > 1 then s * w.dilation else 0))
          ws;
        Some (moved s' (Shape.freeze st) (offset l))

(* Coalescing *)

external coalesce_into : t array -> int array -> int = "nx_array_coalesce"
[@@noalloc]

(* The coalescer refuses what is checked here first, so it answers [NX_OK]. *)
let coalesce ls =
  let n = Array.length ls in
  if n < 1 || n > Shape.max_operands then
    invalid_argf "Layout.coalesce: %d layouts, not 1 to %d" n
      Shape.max_operands;
  if not (Array.for_all (fun l -> ints_equal ls.(0).shape l.shape) ls) then
    invalid_arg "Layout.coalesce: layouts of different shapes";
  let out = Array.make (1 + max_rank + (n * (1 + max_rank))) 0 in
  let e = coalesce_into ls out in
  if e <> 0 then invalid_argf "Layout.coalesce: the coalescer answered %d" e;
  (* The coalescer built each layout from checked ones: [finish] computes its
     span and flags. The layouts share one shape. *)
  let r = out.(0) in
  let shape = Shape.freeze (Array.sub out 1 r) in
  Array.init n (fun k ->
      let at = 1 + r + (k * (1 + r)) in
      finish "Layout.coalesce" shape
        (Shape.freeze (Array.sub out (at + 1) r))
        out.(at))

let pp ppf l =
  Format.fprintf ppf "{shape = %a; strides = %a; offset = %d}" Shape.pp l.shape
    Shape.pp l.strides l.offset
