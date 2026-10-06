(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A1 = Bigarray.Array1
module B = Nx_device.Buffer

type int64s = (int64, Bigarray.int64_elt, Bigarray.c_layout) A1.t
type float64s = (float, Bigarray.float64_elt, Bigarray.c_layout) A1.t
type buf = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) A1.t

(* Each reader reads the [len] bytes at [pos] of [b] as its type's text, and
   raises [Invalid] when they are not that text or their value is outside the
   type. A valid text allocates nothing, except a float that the exact fast path
   does not cover. Values that an OCaml [int] may not hold are stored into an
   array at an index, never returned, so that they are never boxed. *)

exception Invalid of string

let invalid why = raise_notrace (Invalid why)
let[@inline] get (b : buf) i = Char.unsafe_chr (A1.unsafe_get b i)
let is_digit c = '0' <= c && c <= '9'
let digit (b : buf) i = A1.unsafe_get b i - 48
let sub_string b pos len = String.init len (fun k -> get b (pos + k))

let rec equals_from ~caseless b pos s i =
  i = String.length s
  ||
  let c = get b (pos + i) in
  let c = if caseless then Char.lowercase_ascii c else c in
  c = String.unsafe_get s i && equals_from ~caseless b pos s (i + 1)

let equals ~caseless b pos len s =
  len = String.length s && equals_from ~caseless b pos s 0

let is_minus b pos len = len > 0 && get b pos = '-'

let after_sign b pos len =
  if len > 0 && (get b pos = '-' || get b pos = '+') then pos + 1 else pos

let check_digits b first stop =
  if first = stop then invalid "not an integer";
  for i = first to stop - 1 do
    if not (is_digit (get b i)) then invalid "not an integer"
  done

(* Booleans and numbers *)

let bool b pos len =
  if equals ~caseless:false b pos len "true" then true
  else if equals ~caseless:false b pos len "false" then false
  else invalid "not true or false"

(* [int ~min ~max] reads a range of at most 2^32 values. *)
let int ~min ~max b pos len =
  let first = after_sign b pos len and stop = pos + len in
  check_digits b first stop;
  (* Past 2^40 the value is out of every range, so it stops growing. *)
  let v = ref 0 in
  for i = first to stop - 1 do
    if !v < 1 lsl 40 then v := (10 * !v) + digit b i
  done;
  let v = if is_minus b pos len then - !v else !v in
  if v < min || v > max then invalid "out of range";
  v

(* [greater b first s i] is [true] iff the digits at [first] are greater than
   the digits [s] from [i], of equal length. *)
let rec greater b first s i =
  i < String.length s
  &&
  let c = get b (first + i) and d = String.unsafe_get s i in
  c > d || (c = d && greater b first s (i + 1))

(* [wide ~max ~min b pos len a i] reads an integer whose magnitude is at most
   the digits [max], or [min] when negative, and stores it modulo 2^64. Digit
   strings of equal length compare as their values. *)
let wide ~max ~min b pos len (a : int64s) i =
  let first = after_sign b pos len and stop = pos + len in
  check_digits b first stop;
  let first = ref first in
  while !first < stop - 1 && get b !first = '0' do
    incr first
  done;
  let first = !first in
  let limit = if is_minus b pos len then min else max in
  let n = stop - first and m = String.length limit in
  if n > m || (n = m && greater b first limit 0) then invalid "out of range";
  let v = ref 0L in
  for k = first to stop - 1 do
    v := Int64.add (Int64.mul !v 10L) (Int64.of_int (digit b k))
  done;
  A1.unsafe_set a i (if is_minus b pos len then Int64.neg !v else !v)

let int64 b pos len a i =
  wide ~max:"9223372036854775807" ~min:"9223372036854775808" b pos len a i

(* A minus is allowed only before zero. *)
let uint64 b pos len a i =
  wide ~max:"18446744073709551615" ~min:"0" b pos len a i

let pow10 = Array.init 23 (fun e -> float_of_string ("1e" ^ string_of_int e))

(* [float] reads a decimal number, [inf], [infinity] or [nan] as the nearest
   float64. The fast path is exact: a significand of at most 2^53 and a power of
   ten of at most 10^22 are exact doubles, and the product or quotient rounds
   once. Past 2^53 the significand stops growing, which leaves the fast path. *)
let float b pos len (a : float64s) k =
  let stop = pos + len in
  let first = after_sign b pos len in
  let rest = stop - first in
  if
    equals ~caseless:true b first rest "inf"
    || equals ~caseless:true b first rest "infinity"
  then A1.unsafe_set a k (if is_minus b pos len then neg_infinity else infinity)
  else if equals ~caseless:true b first rest "nan" then A1.unsafe_set a k nan
  else begin
    let i = ref first and m = ref 0 and exp = ref 0 in
    let frac = ref false and any = ref false in
    while
      !i < stop
      &&
      let c = get b !i in
      is_digit c || (c = '.' && not !frac)
    do
      if get b !i = '.' then frac := true
      else begin
        any := true;
        if !m <= 1 lsl 53 then begin
          m := (10 * !m) + digit b !i;
          if !frac then decr exp
        end
      end;
      incr i
    done;
    if not !any then invalid "not a number";
    if !i < stop && (get b !i = 'e' || get b !i = 'E') then begin
      incr i;
      let negative = !i < stop && get b !i = '-' in
      if !i < stop && (negative || get b !i = '+') then incr i;
      if not (!i < stop && is_digit (get b !i)) then invalid "not a number";
      let e = ref 0 in
      while !i < stop && is_digit (get b !i) do
        if !e < 100_000 then e := (10 * !e) + digit b !i;
        incr i
      done;
      exp := if negative then !exp - !e else !exp + !e
    end;
    if !i <> stop then invalid "not a number";
    if !m <= 1 lsl 53 && -22 <= !exp && !exp <= 22 then begin
      let v = Float.of_int !m in
      let v = if !exp >= 0 then v *. pow10.(!exp) else v /. pow10.(- !exp) in
      A1.unsafe_set a k (if is_minus b pos len then -.v else v)
    end
    else A1.unsafe_set a k (float_of_string (sub_string b pos len))
  end

(* Rounding to a narrower type: [float] rounds once to float64, and a cast to
   the narrower type rounds again. Only a float64 halfway between two values of
   that type can round wrong then, so such a float64 moves one step towards the
   text's value, found by comparing their exact digits, unless the text is the
   float64 itself. *)

type width = { p : int; emin : int; max : float }

(* The width of the float dtype [dt], read from its limits: [emin] is the
   exponent of its least normal value. *)
let float_width dt =
  {
    p = Nx_dtype.precision dt;
    emin = snd (Float.frexp (Nx_dtype.min_normal dt)) - 1;
    max = Nx_dtype.max_finite dt;
  }

let half = float_width Nx_dtype.float16
let single = float_width Nx_dtype.float32

(* [nearest w v] is the value of width [w] nearest to [v], ties to even. *)
let nearest w v =
  let biased =
    Int64.to_int (Int64.shift_right_logical (Int64.bits_of_float v) 52)
  in
  let e = Int.max ((biased land 0x7FF) - 1023) w.emin in
  let ulp = Float.ldexp 1. (e - w.p + 1) in
  let q = v /. ulp in
  let r = Float.round q in
  let r =
    if Float.abs (r -. q) = 0.5 && Float.rem r 2. <> 0. then
      r -. Float.copy_sign 1. q
    else r
  in
  let r = r *. ulp in
  if Float.abs r > w.max then Float.copy_sign Float.infinity v else r

(* [magnitude s] is the significant digits of the decimal number [s], without
   leading and trailing zeros, and the exponent [e] of its value [0.digits ×
   10^e], ignoring its sign. The digits of zero are empty. *)
let magnitude s =
  let n = String.length s and digits = Buffer.create 32 in
  let i = ref (if n > 0 && (s.[0] = '-' || s.[0] = '+') then 1 else 0) in
  let point = ref (-1) and exp = ref 0 in
  while !i < n && (is_digit s.[!i] || s.[!i] = '.') do
    if s.[!i] = '.' then point := Buffer.length digits
    else Buffer.add_char digits s.[!i];
    incr i
  done;
  if !i < n then begin
    incr i;
    let negative = s.[!i] = '-' in
    if negative || s.[!i] = '+' then incr i;
    while !i < n do
      if !exp < 100_000 then exp := (10 * !exp) + Char.code s.[!i] - 48;
      incr i
    done;
    if negative then exp := - !exp
  end;
  let d = Buffer.contents digits in
  let point = if !point < 0 then String.length d else !point in
  let lead = ref 0 and stop = ref (String.length d) in
  while !lead < !stop && d.[!lead] = '0' do
    incr lead
  done;
  while !stop > !lead && d.[!stop - 1] = '0' do
    decr stop
  done;
  if !lead = !stop then ("", 0)
  else (String.sub d !lead (!stop - !lead), point - !lead + !exp)

let compare_magnitude (d0, e0) (d1, e1) =
  match (d0, d1) with
  | "", "" -> 0
  | "", _ -> -1
  | _, "" -> 1
  | _ -> if e0 <> e1 then Int.compare e0 e1 else String.compare d0 d1

(* [narrow w] reads the text as [float] does, and stores a float64 that rounds
   to the value of width [w] nearest to the text, ties to even. It allocates
   when the float64 nearest to the text is halfway between two values of [w],
   whose digits it then compares exactly. *)
let narrow w b pos len a k =
  float b pos len a k;
  let x = A1.unsafe_get a k in
  if Float.is_finite x && nearest w (Float.pred x) <> nearest w (Float.succ x)
  then
    (* 1100 digits write every float64 exactly. *)
    let text = magnitude (sub_string b pos len) in
    match compare_magnitude text (magnitude (Printf.sprintf "%.1100e" x)) with
    | 0 -> ()
    | c ->
        A1.unsafe_set a k
          (if c > 0 = (x > 0.) then Float.succ x else Float.pred x)

(* Dates and datetimes *)

let two_digits b i =
  let c0 = get b i and c1 = get b (i + 1) in
  if is_digit c0 && is_digit c1 then (10 * digit b i) + digit b (i + 1) else -1

let is_leap y = (y mod 4 = 0 && y mod 100 <> 0) || y mod 400 = 0

let days_in_month y m =
  match m with
  | 2 -> if is_leap y then 29 else 28
  | 4 | 6 | 9 | 11 -> 30
  | _ -> 31

(* Howard Hinnant's days_from_civil. *)
let days_from_civil y m d =
  let y = if m <= 2 then y - 1 else y in
  let era = (if y >= 0 then y else y - 399) / 400 in
  let yoe = y - (era * 400) in
  let doy = (((153 * ((m + 9) mod 12)) + 2) / 5) + d - 1 in
  let doe = (yoe * 365) + (yoe / 4) - (yoe / 100) + doy in
  (era * 146097) + doe - 719468

(* Howard Hinnant's civil_from_days, the inverse of [days_from_civil]. *)
let civil_of_days days =
  let z = days + 719468 in
  let era = (if z >= 0 then z else z - 146096) / 146097 in
  let doe = z - (era * 146097) in
  let yoe = (doe - (doe / 1460) + (doe / 36524) - (doe / 146096)) / 365 in
  let doy = doe - ((365 * yoe) + (yoe / 4) - (yoe / 100)) in
  let mp = ((5 * doy) + 2) / 153 in
  let d = doy - (((153 * mp) + 2) / 5) + 1 in
  let m = if mp < 10 then mp + 3 else mp - 9 in
  let y = yoe + (era * 400) in
  ((if m <= 2 then y + 1 else y), m, d)

(* [day_at b pos y] reads [-MM-DD] at [pos], whose six bytes exist, as the days
   since 1970-01-01 of that day of the year [y]. *)
let day_at b pos y =
  let m = two_digits b (pos + 1) and d = two_digits b (pos + 4) in
  if m < 0 || d < 0 || get b pos <> '-' || get b (pos + 3) <> '-' then
    invalid "not YYYY-MM-DD";
  if m < 1 || m > 12 || d < 1 || d > days_in_month y m then
    invalid "not a day of the calendar";
  days_from_civil y m d

(* [date_at b pos] reads [YYYY-MM-DD] at [pos], whose ten bytes exist, as its
   days since 1970-01-01. *)
let date_at b pos =
  let y0 = two_digits b pos and y1 = two_digits b (pos + 2) in
  if y0 < 0 || y1 < 0 then invalid "not YYYY-MM-DD";
  day_at b (pos + 4) ((100 * y0) + y1)

(* A year outside [0000] to [9999] is signed and of at least four digits, as
   [Time.Date.pp] writes it: [-0044-03-15], [+12345-01-01]. Past 2^40 the year
   stops growing, past any range. [civil_days] is the days of any such date,
   which a datetime's ticks bound, and [date] those that {!Time.Date} holds. *)
let civil_days b pos len =
  if len = 10 then date_at b pos
  else begin
    let stop = pos + len in
    if len < 11 || (get b pos <> '+' && get b pos <> '-') then
      invalid "not YYYY-MM-DD";
    let y = ref 0 in
    for i = pos + 1 to stop - 7 do
      if not (is_digit (get b i)) then invalid "not YYYY-MM-DD";
      if !y < 1 lsl 40 then y := (10 * !y) + digit b i
    done;
    day_at b (stop - 6) (if get b pos = '-' then - !y else !y)
  end

let date b pos len =
  let days = civil_days b pos len in
  if Option.is_none (Time.Date.of_days days) then invalid "out of range";
  days

let floor_div a b = Int64.(if rem a b < 0L then pred (div a b) else div a b)

let per_second : Type.unit_ -> int = function
  | S -> 1
  | Ms -> 1_000
  | Us -> 1_000_000
  | Ns -> 1_000_000_000

let not_whole : Type.unit_ -> string = function
  | S -> "not a whole number of seconds"
  | Ms -> "not a whole number of milliseconds"
  | Us -> "not a whole number of microseconds"
  | Ns -> "not a whole number of nanoseconds"

(* [ticks u secs ns] is [secs] seconds and [ns] nanoseconds, [0 <= ns < 10^9],
   after 1970-01-01 00:00:00 as ticks of [u]. A negative [secs] with a fraction
   counts from [secs + 1], so that the least instants do not wrap. *)
let ticks u secs ns =
  let per = per_second u in
  let div = 1_000_000_000 / per in
  if ns mod div <> 0 then invalid (not_whole u);
  let secs, frac =
    if Int64.compare secs 0L < 0 && ns > 0 then
      (Int64.succ secs, (ns / div) - per)
    else (secs, ns / div)
  in
  let per = Int64.of_int per in
  let whole = Int64.mul secs per in
  let v = Int64.add whole (Int64.of_int frac) in
  if Int64.div whole per <> secs || frac >= 0 <> (Int64.compare v whole >= 0)
  then invalid "out of range";
  v

(* The most days whose seconds int64 holds. *)
let max_days = Int64.to_int (Int64.div Int64.max_int 86400L)

(* [seconds days s] is [days] days and [s] seconds, [|s| < 2 * 86400], as int64
   seconds. A negative [days] counts from [days + 1], so that the least seconds
   do not wrap. *)
let seconds days s =
  let days, s = if days < 0 then (days + 1, s - 86400) else (days, s) in
  if Int.abs days > max_days then invalid "out of range";
  let d = Int64.mul (Int64.of_int days) 86400L and s = Int64.of_int s in
  let r = Int64.add d s in
  if Int64.compare s 0L >= 0 <> (Int64.compare r d >= 0) then
    invalid "out of range";
  r

(* [fraction b i stop] reads one to nine digits of a fraction of a second at
   [!i], as nanoseconds, or is [-1]. *)
let fraction b i stop =
  let first = !i and ns = ref 0 in
  while !i < stop && !i - first < 9 && is_digit (get b !i) do
    ns := (10 * !ns) + digit b !i;
    incr i
  done;
  if !i = first then -1
  else begin
    for _ = !i - first to 8 do
      ns := 10 * !ns
    done;
    !ns
  end

(* [offset b i stop] reads [Z] or [±hh:mm] at [!i] as seconds east of UTC, or is
   [None]. *)
let offset b i stop =
  if !i < stop && get b !i = 'Z' then begin
    incr i;
    Some 0
  end
  else if !i + 6 > stop then None
  else
    let s = get b !i in
    let oh = two_digits b (!i + 1) and om = two_digits b (!i + 4) in
    if (s <> '+' && s <> '-') || oh < 0 || om < 0 || get b (!i + 3) <> ':' then
      None
    else begin
      if oh > 23 || om > 59 then invalid "not an offset";
      i := !i + 6;
      let o = (3600 * oh) + (60 * om) in
      Some (if s = '-' then -o else o)
    end

let form ~zoned =
  invalid
    (if zoned then "not YYYY-MM-DDThh:mm:ss with an offset"
     else "not YYYY-MM-DDThh:mm:ss without an offset")

(* [datetime u ~zoned] reads a datetime, with an offset iff [zoned], as ticks of
   [u] since 1970-01-01 00:00:00, in UTC when it has an offset. *)
let datetime u ~zoned b pos len (a : int64s) k =
  let stop = pos + len in
  (* A signed year makes the date longer than ten bytes. *)
  let rec date_end i =
    if i >= stop || get b i = 'T' || get b i = ' ' then i else date_end (i + 1)
  in
  let dlen =
    if len > 0 && (get b pos = '+' || get b pos = '-') then
      date_end (pos + 1) - pos
    else 10
  in
  if len < dlen + 9 then form ~zoned;
  let days = civil_days b pos dlen in
  let p = pos + dlen in
  let t = get b p in
  let hh = two_digits b (p + 1) and mm = two_digits b (p + 4) in
  let ss = two_digits b (p + 7) in
  if
    (t <> 'T' && t <> ' ')
    || hh < 0 || mm < 0 || ss < 0
    || get b (p + 3) <> ':'
    || get b (p + 6) <> ':'
  then form ~zoned;
  if hh > 23 || mm > 59 || ss > 59 then invalid "not a time of day";
  let i = ref (p + 9) and ns = ref 0 in
  if !i < stop && get b !i = '.' then begin
    incr i;
    ns := fraction b i stop;
    if !ns < 0 then form ~zoned
  end;
  let offset =
    if not zoned then 0
    else match offset b i stop with Some o -> o | None -> form ~zoned
  in
  if !i <> stop then form ~zoned;
  let secs = seconds days ((3600 * hh) + (60 * mm) + ss - offset) in
  A1.unsafe_set a k (ticks u secs !ns)

(* [directed fmt ty] reads a text in the format [fmt] of [Temporal.parse] as a
   value of [ty], a date, a clock or a datetime: days, or ticks. *)
let directed fmt (Type.Any ty) b pos len (a : int64s) k =
  let stop = pos + len and i = ref pos in
  let mismatch () = invalid (Printf.sprintf "not in the format %S" fmt) in
  let byte c = if !i < stop && get b !i = c then incr i else mismatch () in
  let digits n =
    if !i + n > stop then mismatch ();
    let v = ref 0 in
    for j = !i to !i + n - 1 do
      if not (is_digit (get b j)) then mismatch ();
      v := (10 * !v) + digit b j
    done;
    i := !i + n;
    !v
  in
  let year () =
    match if !i < stop then get b !i else ' ' with
    | ('+' | '-') as sign ->
        incr i;
        let first = !i and y = ref 0 in
        while !i < stop && is_digit (get b !i) do
          if !y < 1 lsl 40 then y := (10 * !y) + digit b !i;
          incr i
        done;
        if !i - first < 4 then mismatch ();
        if sign = '-' then - !y else !y
    | _ -> digits 4
  in
  let y = ref 1970 and mo = ref 1 and d = ref 1 and h = ref 0 and mi = ref 0 in
  let s = ref 0 and ns = ref 0 and off = ref 0 in
  let j = ref 0 in
  while !j < String.length fmt do
    (match fmt.[!j] with
    | '%' -> (
        incr j;
        match fmt.[!j] with
        | 'Y' -> y := year ()
        | 'm' -> mo := digits 2
        | 'd' -> d := digits 2
        | 'H' -> h := digits 2
        | 'M' -> mi := digits 2
        | 'S' -> s := digits 2
        | 'f' ->
            ns := fraction b i stop;
            if !ns < 0 then mismatch ()
        | 'z' -> (
            match offset b i stop with
            | Some o -> off := o
            | None -> mismatch ())
        | c -> byte c)
    | c -> byte c);
    incr j
  done;
  if !i <> stop then mismatch ();
  if !mo < 1 || !mo > 12 || !d < 1 || !d > days_in_month !y !mo then
    invalid "not a day of the calendar";
  if !h > 23 || !mi > 59 || !s > 59 then invalid "not a time of day";
  let days = days_from_civil !y !mo !d in
  let tod = (3600 * !h) + (60 * !mi) + !s in
  A1.unsafe_set a k
    (match ty with
    | Date ->
        if Option.is_none (Time.Date.of_days days) then invalid "out of range";
        Int64.of_int days
    | Clock u -> ticks u (Int64.of_int tod) !ns
    | Datetime { unit_; _ } -> ticks unit_ (seconds days (tod - !off)) !ns
    | _ -> assert false (* Binding checked [ty]. *))

(* Parsing *)

let by = "Column.parse"
let err fmt = Format.kasprintf invalid_arg fmt

(* [rows r valid read] calls [read b pos len i] on the bytes of each row [i] of
   [r] that the validity [valid] holds, in order, and is the first row where
   [read] raises [Invalid] with the reason. *)
let rows r valid read =
  let n = Nx_ragged.length r in
  let scan (o : int64s) b is_null =
    let i = ref 0 in
    try
      while !i < n do
        if not (is_null !i) then begin
          let pos = Int64.to_int (A1.unsafe_get o !i) in
          read b pos (Int64.to_int (A1.unsafe_get o (!i + 1)) - pos) !i
        end;
        incr i
      done;
      None
    with Invalid why -> Some (!i, why)
  in
  Strings.reading ~by (Nx_ragged.offsets r) @@ fun o ->
  Strings.reading ~by (Nx_ragged.values r) @@ fun b ->
  let o = B.bigarray Bigarray.int64 o
  and b = B.bigarray Bigarray.int8_unsigned b in
  match valid with
  | None -> scan o b (Fun.const false)
  | Some m ->
      Strings.reading ~by m @@ fun m ->
      let m = B.bigarray Bigarray.int8_unsigned m in
      scan o b (fun i -> (A1.unsafe_get m (i lsr 3) lsr (i land 7)) land 1 = 0)

(* [fixed kind zero r valid read] is the array that [read] fills at each row
   [rows] reads, [zero] under a null. *)
let fixed kind zero r valid read =
  let a = A1.create kind Bigarray.c_layout (Nx_ragged.length r) in
  A1.fill a zero;
  match rows r valid (fun b pos len i -> read b pos len a i) with
  | Some e -> Error e
  | None -> Ok (Nx.of_bigarray (Bigarray.genarray_of_array1 a))

let text_rows c =
  match (Column.type_ c, Column.data c) with
  | (Any String | Any Binary), Bytes r -> r
  | Any t, _ -> err "Column.parse: %a is neither string nor binary" Type.pp t

let parse (Type.Any ty as any) c =
  let r = text_rows c in
  let valid = Column.validity c in
  let column d = Column.with_data any d c in
  let cast dt x = column (Fixed (P (Nx.cast dt x))) in
  let keep x = column (Fixed (P x)) in
  let ints read = fixed Bigarray.int64 0L r valid read in
  let of_int read =
    ints (fun b pos len a i ->
        A1.unsafe_set a i (Int64.of_int (read b pos len)))
  in
  let floats read = fixed Bigarray.float64 0. r valid read in
  let small dt ~min ~max = Result.map (cast dt) (of_int (int ~min ~max)) in
  match ty with
  | Bool ->
      let read b pos len a i =
        A1.unsafe_set a i (Bool.to_int (bool b pos len))
      in
      Result.map (cast Nx.bit) (fixed Bigarray.int8_unsigned 0 r valid read)
  | Int8 -> small Nx.int8 ~min:(-0x80) ~max:0x7F
  | Int16 -> small Nx.int16 ~min:(-0x8000) ~max:0x7FFF
  | Int32 -> small Nx.int32 ~min:(-0x8000_0000) ~max:0x7FFF_FFFF
  | Uint8 -> small Nx.uint8 ~min:0 ~max:0xFF
  | Uint16 -> small Nx.uint16 ~min:0 ~max:0xFFFF
  | Uint32 -> small Nx.uint32 ~min:0 ~max:0xFFFF_FFFF
  | Int64 -> Result.map keep (ints int64)
  | Uint64 ->
      let bits x = column (Fixed (P (Nx.bitcast Nx.uint64 x))) in
      Result.map bits (ints uint64)
  | Float16 -> Result.map (cast Nx.float16) (floats (narrow half))
  | Float32 -> Result.map (cast Nx.float32) (floats (narrow single))
  | Float64 -> Result.map keep (floats float)
  | Date -> Result.map (cast Nx.int32) (of_int date)
  | Datetime { unit_; zone } ->
      Result.map keep (ints (datetime unit_ ~zoned:(zone <> None)))
  | Categorical d ->
      let codes = Hashtbl.create (Iarray.length d) in
      Iarray.iteri (fun i s -> Hashtbl.add codes s i) d;
      let code b pos len =
        match Hashtbl.find codes (sub_string b pos len) with
        | k -> k
        | exception Not_found -> invalid "not in the dictionary"
      in
      Result.map (cast Nx.int32) (of_int code)
  | String -> (
      match Strings.utf_8 ~by ?mask:valid r with
      | Some e -> Error e
      | None -> Ok (column (Bytes r)))
  | Binary -> Ok (column (Bytes r))
  | Clock _ | Duration _ | List _ | Record _ | Tensor _ | Ext _ ->
      err "Column.parse: %a has no text form" Type.pp ty

let parse_with fmt (Type.Any ty as any) c =
  let valid = Column.validity c in
  let column x =
    let x = match ty with Date -> Nx.P (Nx.cast Nx.int32 x) | _ -> Nx.P x in
    Column.with_data any (Fixed x) c
  in
  Result.map column
    (fixed Bigarray.int64 0L (text_rows c) valid (directed fmt any))

(* Formats *)

let format_with fmt c =
  let (Type.Any ty) = Column.type_ c in
  let n = Column.length c in
  let ticks =
    match Column.data c with
    | Fixed (P x) -> Nx.to_array (Nx.cast Nx.int64 x)
    | _ -> assert false
  in
  let valid = Option.map Nx.to_array (Column.validity c) in
  (* [fields v] is the days, the second of the day and the nanosecond of the
     second of the value [v]. *)
  let fields v =
    match ty with
    | Date -> (Int64.to_int v, 0, 0)
    | Clock u | Datetime { unit_ = u; _ } ->
        let per = Int64.of_int (per_second u) in
        let secs = floor_div v per
        and ns_per_tick = 1_000_000_000 / per_second u in
        let days = floor_div secs 86400L in
        let sod = Int64.sub secs (Int64.mul days 86400L) in
        let ns =
          Int64.to_int (Int64.sub v (Int64.mul secs per)) * ns_per_tick
        in
        (Int64.to_int days, Int64.to_int sod, ns)
    | _ -> assert false (* Binding checked [ty]. *)
  in
  let b = Buffer.create (16 * n) in
  let write v =
    let days, sod, ns = fields v in
    let y, m, d = civil_of_days days in
    let j = ref 0 in
    while !j < String.length fmt do
      (match fmt.[!j] with
      | '%' -> (
          incr j;
          match fmt.[!j] with
          | 'Y' when 0 <= y && y <= 9999 -> Printf.bprintf b "%04d" y
          | 'Y' -> Printf.bprintf b "%+05d" y
          | 'm' -> Printf.bprintf b "%02d" m
          | 'd' -> Printf.bprintf b "%02d" d
          | 'H' -> Printf.bprintf b "%02d" (sod / 3600)
          | 'M' -> Printf.bprintf b "%02d" (sod / 60 mod 60)
          | 'S' -> Printf.bprintf b "%02d" (sod mod 60)
          | 'f' -> Printf.bprintf b "%09d" ns
          | 'z' -> Buffer.add_char b 'Z'
          | c -> Buffer.add_char b c)
      | c -> Buffer.add_char b c);
      incr j
    done
  in
  let offsets = Array.make (n + 1) 0L in
  for i = 0 to n - 1 do
    if Option.fold ~none:true ~some:(fun v -> v.(i)) valid then write ticks.(i);
    offsets.(i + 1) <- Int64.of_int (Buffer.length b)
  done;
  let text = Buffer.contents b in
  let bytes =
    Nx.init Nx.uint8 [| String.length text |] (fun i -> Char.code text.[i.(0)])
  in
  let offsets = Nx.create Nx.int64 [| n + 1 |] offsets in
  Column.with_data (Any Type.string) (Bytes (Nx_ragged.v ~offsets bytes)) c

(* Printing *)

let pp_text ppf s = Format.pp_print_string ppf s

(* [text x p emin b] writes at the start of [b], 32 bytes or more, the text of
   [x], a float of [p] significant bits and least normal exponent [emin], and is
   its length: the fewest significant digits that read back as [x] at that
   width, the nearest of them to [x], ties to even, without an exponent from
   [1e-7] up to [1e21] ([150], [0.0015]) and as C's [%e] writes them otherwise
   ([1e+21], [1.5e-08]), or [nan], [inf] or [-inf]. *)
external text :
  (float[@unboxed]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  Bytes.t ->
  (int[@untagged]) = "talon_float_text_byte" "talon_float_text"
[@@noalloc]

let double = float_width Nx_dtype.float64

let width : type a. a Type.t -> width = function
  | Float16 -> half
  | Float32 -> single
  | _ -> double

(* [float_text ty x] is the canonical text of [x] at the width of [ty]. *)
let float_text ty x =
  let w = width ty and b = Bytes.create 32 in
  let x = if Float.is_finite x then nearest w x else x in
  Bytes.sub_string b 0 (text x w.p w.emin b)

(* [add_date b days] writes the day [days] after 1970-01-01 as [Time.Date.pp]
   does. *)
let add_date b days =
  let y, m, d = civil_of_days days in
  if y >= 0 && y <= 9999 then Printf.bprintf b "%04d-%02d-%02d" y m d
  else Printf.bprintf b "%+05d-%02d-%02d" y m d

(* [add_datetime u ~zoned b ticks] writes the ticks [ticks] of [u] as
   [YYYY-MM-DDThh:mm:ss], the fewest fraction digits that are exact, and [Z]
   when [zoned]. *)
let add_datetime u ~zoned b ticks =
  let per = Int64.of_int (per_second u) in
  let secs = floor_div ticks per in
  let frac = Int64.to_int (Int64.sub ticks (Int64.mul secs per)) in
  let days = floor_div secs 86400L in
  let sod = Int64.to_int (Int64.sub secs (Int64.mul days 86400L)) in
  add_date b (Int64.to_int days);
  Printf.bprintf b "T%02d:%02d:%02d" (sod / 3600) (sod / 60 mod 60) (sod mod 60);
  if frac > 0 then begin
    let digits =
      Printf.sprintf "%09d" (frac * (1_000_000_000 / per_second u))
    in
    let n = ref 9 in
    while digits.[!n - 1] = '0' do
      decr n
    done;
    Buffer.add_char b '.';
    Buffer.add_substring b digits 0 !n
  end;
  if zoned then Buffer.add_char b 'Z'

let pp_instant ~zoned ppf t =
  let b = Buffer.create 32 in
  add_datetime Ns ~zoned b (Time.to_ns t);
  pp_text ppf (Buffer.contents b)

let pp : type a. a Type.t -> Format.formatter -> a -> unit =
 fun ty ppf v ->
  match (ty, Type.kind ty) with
  | Binary, _ -> pp_text ppf (v :> string)
  | Datetime { zone; _ }, _ -> pp_instant ~zoned:(zone <> None) ppf v
  | (Clock _ | Duration _ | List _ | Record _ | Tensor _ | Ext _), k ->
      Type.pp_lit k ppf v
  | _, Bool -> Format.pp_print_bool ppf v
  | _, Int -> Format.pp_print_int ppf v
  | _, Float -> pp_text ppf (float_text ty v)
  | _, String -> pp_text ppf v
  | _, Date -> Time.Date.pp ppf v
  | _, k -> Type.pp_lit k ppf v

(* Writing *)

let fixed_writer : type a b c.
    a Type.t -> (b, c) Nx.t -> Buffer.t -> int -> unit =
 fun ty x ->
  let ints () = Nx.to_array (Nx.cast Nx.int64 x) in
  match ty with
  | Bool ->
      let v = ints () in
      fun b i -> Buffer.add_string b (if v.(i) = 0L then "false" else "true")
  | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 ->
      let v = ints () in
      fun b i -> Buffer.add_string b (Int64.to_string v.(i))
  | Uint64 ->
      let v = Nx.to_array (Nx.bitcast Nx.int64 x) in
      fun b i -> Printf.bprintf b "%Lu" v.(i)
  | Float16 | Float32 | Float64 ->
      let v = Nx.to_array (Nx.cast Nx.float64 x) in
      let w = width ty and s = Bytes.create 32 in
      fun b i -> Buffer.add_subbytes b s 0 (text v.(i) w.p w.emin s)
  | Categorical d ->
      let v = ints () in
      fun b i -> Buffer.add_string b (Iarray.get d (Int64.to_int v.(i)))
  | Date ->
      let v = ints () in
      fun b i -> add_date b (Int64.to_int v.(i))
  | Datetime { unit_; zone } ->
      let v = ints () and zoned = Option.is_some zone in
      fun b i -> add_datetime unit_ ~zoned b v.(i)
  | _ -> err "Column.print: %a is not a type that Column.parse reads" Type.pp ty

let print c =
  let (Any ty) = Column.type_ c in
  match (ty, Column.data c) with
  | (String | Binary), _ -> c
  | _, Fixed (P x) ->
      let write = fixed_writer ty x and n = Column.length c in
      let valid = Option.map Nx.to_array (Column.validity c) in
      let b = Buffer.create (8 * n) and offsets = Array.make (n + 1) 0L in
      for i = 0 to n - 1 do
        (match valid with Some v when not v.(i) -> () | _ -> write b i);
        offsets.(i + 1) <- Int64.of_int (Buffer.length b)
      done;
      let values =
        A1.create Bigarray.int8_unsigned Bigarray.c_layout (Buffer.length b)
      in
      String.iteri
        (fun i ch -> A1.unsafe_set values i (Char.code ch))
        (Buffer.contents b);
      let r =
        Nx_ragged.v
          ~offsets:(Nx.create Nx.int64 [| n + 1 |] offsets)
          (Nx.of_bigarray (Bigarray.genarray_of_array1 values))
      in
      Column.with_data (Any Type.string) (Bytes r) c
  | _ -> err "Column.print: %a is not a type that Column.parse reads" Type.pp ty

(* Cells *)

let is_null c i =
  match Column.validity c with None -> false | Some v -> not (Nx.item [ i ] v)

let row_int64 x i = Nx.item [] (Nx.cast Nx.int64 (Nx.get [ i ] x))
let row_float x i = Nx.item [] (Nx.cast Nx.float64 (Nx.get [ i ] x))

(* [pp_ticks ty pp_value of_ticks ppf ticks] writes the value of [ticks] through
   [of_ticks], or the ticks themselves when they are outside {!Time}'s range. *)
let pp_ticks ty pp_value of_ticks ppf ticks =
  match of_ticks ticks with
  | Some v -> pp_value ppf v
  | None -> Format.fprintf ppf "%a tick %Ld" Type.pp ty ticks

let span_of_ticks : Type.unit_ -> int64 -> Time.span option = function
  | S -> Time.Span.of_s
  | Ms -> Time.Span.of_ms
  | Us -> Time.Span.of_us
  | Ns -> fun n -> Some (Time.Span.of_ns n)

let instant_of_ticks : Type.unit_ -> int64 -> Time.instant option = function
  | S -> Time.of_s
  | Ms -> Time.of_ms
  | Us -> Time.of_us
  | Ns -> fun n -> Some (Time.of_ns n)

let rec pp_cell c i ppf =
  if is_null c i then pp_text ppf "∅"
  else
    let (Any ty) = Column.type_ c in
    pp_row ty c i ppf

and pp_row : type a. a Type.t -> Column.t -> int -> Format.formatter -> unit =
 fun ty c i ppf ->
  match (ty, Column.data c) with
  | Ext { storage; _ }, _ -> pp_row storage c i ppf
  | (String | Binary), Bytes r ->
      let o = Nx_ragged.offsets r and values = Nx_ragged.values r in
      let pos = Int64.to_int (Nx.item [ i ] o) in
      let len = Int64.to_int (Nx.item [ i + 1 ] o) - pos in
      let byte k = Char.chr (Nx.item [ pos + k ] values) in
      Type.pp_quoted ppf (String.init len byte)
  | List _, List { offsets; child } ->
      let first = Int64.to_int (Nx.item [ i ] offsets) in
      let stop = Int64.to_int (Nx.item [ i + 1 ] offsets) in
      let elements = List.init (stop - first) (( + ) first) in
      Type.pp_list (fun ppf j -> pp_cell child j ppf) ppf elements
  | Record _, _ -> pp_text ppf "<record>"
  | Tensor _, _ -> pp_text ppf "<tensor>"
  | _, Fixed (P x) -> (
      match (ty, Type.kind ty) with
      | Int64, _ -> Format.fprintf ppf "%Ld" (row_int64 x i)
      | Uint64, _ ->
          let bits = Nx.item [] (Nx.bitcast Nx.int64 (Nx.get [ i ] x)) in
          Format.fprintf ppf "%Lu" bits
      | Categorical d, _ ->
          Type.pp_quoted ppf (Iarray.get d (Int64.to_int (row_int64 x i)))
      | Date, _ ->
          let of_days n = Time.Date.of_days (Int64.to_int n) in
          pp_ticks ty Time.Date.pp of_days ppf (row_int64 x i)
      | (Clock u | Duration u), _ ->
          pp_ticks ty Time.Span.pp (span_of_ticks u) ppf (row_int64 x i)
      | Datetime { unit_; zone }, _ ->
          let pp_value = pp_instant ~zoned:(zone <> None) in
          pp_ticks ty pp_value (instant_of_ticks unit_) ppf (row_int64 x i)
      | _, Bool -> pp ty ppf (row_int64 x i <> 0L)
      | _, Int -> pp ty ppf (Int64.to_int (row_int64 x i))
      | _, Float -> pp ty ppf (row_float x i)
      | _ -> assert false)
  | _ -> assert false

(* Floats of a table column *)

let significant = 6
let max_decimals = 6

(* Past 10^16 a float64's integer digits are no longer all its own. *)
let max_fixed = 1e16

(* [exponent x] is the decimal exponent of [x] rounded to six significant
   digits: [2] for [99.99996], which rounds to [100.000]. *)
let exponent x =
  let s = Printf.sprintf "%.*e" (significant - 1) x in
  let e = String.index s 'e' in
  int_of_string (String.sub s (e + 1) (String.length s - e - 1))

let pp_floats xs =
  let decimals = ref 0 and scientific = ref false in
  Array.iter
    (fun x ->
      if Float.is_finite x && x <> 0. then begin
        let d = significant - 1 - exponent x in
        if d > max_decimals || Float.abs x >= max_fixed then scientific := true
        else decimals := Int.max !decimals d
      end)
    xs;
  let decimals = !decimals and scientific = !scientific in
  fun ppf x ->
    if Float.is_nan x then pp_text ppf "nan"
    else if not (Float.is_finite x) then
      pp_text ppf (if x > 0. then "inf" else "-inf")
    else if scientific then Format.fprintf ppf "%.*e" (significant - 1) x
    else Format.fprintf ppf "%.*f" decimals x
