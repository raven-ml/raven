(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A1 = Bigarray.Array1
module B = Nx_device.Buffer

type int64s = (int64, Bigarray.int64_elt, Bigarray.c_layout) A1.t
type float64s = (float, Bigarray.float64_elt, Bigarray.c_layout) A1.t

(* Each reader reads the [len] bytes at [pos] of [b] as its type's text, and
   raises [Invalid] when they are not that text or their value is outside the
   type. A valid text allocates nothing, except a float that the exact fast path
   does not cover. Values that an OCaml [int] may not hold are stored into an
   array at an index, never returned, so that they are never boxed. *)

exception Invalid of string

let invalid why = raise_notrace (Invalid why)
let[@inline] get b i = Char.unsafe_chr (A1.unsafe_get b i)
let is_digit c = '0' <= c && c <= '9'
let digit b i = A1.unsafe_get b i - 48
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

let half = { p = 11; emin = -14; max = 65504. }
let single = { p = 24; emin = -126; max = 0x1.fffffep127 }

(* [nearest w v] is the value of width [w] nearest to [v], for a [v] that is not
   halfway between two of them. *)
let nearest w v =
  let biased =
    Int64.to_int (Int64.shift_right_logical (Int64.bits_of_float v) 52)
  in
  let e = Int.max ((biased land 0x7FF) - 1023) w.emin in
  let ulp = Float.ldexp 1. (e - w.p + 1) in
  let r = Float.round (v /. ulp) *. ulp in
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

let int_pow10 = Array.init 19 (fun e -> int_of_float pow10.(e))

let too_many precision =
  invalid (Printf.sprintf "more than %d digits" precision)

(* [decimal ~precision ~scale] reads a decimal number without an exponent, exact
   at [scale] digits after the point and of at most [precision <= 18] digits, as
   its unscaled value. *)
let decimal ~precision ~scale b pos len =
  let stop = pos + len in
  let limit = int_pow10.(precision) in
  let i = ref (after_sign b pos len) and v = ref 0 and frac = ref (-1) in
  let any = ref false in
  while !i < stop do
    let c = get b !i in
    if c = '.' && !frac < 0 then frac := 0
    else if is_digit c then begin
      any := true;
      if !frac >= 0 then incr frac;
      if !frac <= scale then begin
        v := (10 * !v) + digit b !i;
        if !v >= limit then too_many precision
      end
      else if c <> '0' then
        invalid (Printf.sprintf "more than %d digits after the point" scale)
    end
    else invalid "not a decimal number";
    incr i
  done;
  if not !any then invalid "not a decimal number";
  for _ = Int.max 0 !frac + 1 to scale do
    if !v >= limit / 10 then too_many precision;
    v := 10 * !v
  done;
  if is_minus b pos len then - !v else !v

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
   stops growing, which is out of range. *)
let date b pos len =
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
    let days = day_at b (stop - 6) (if get b pos = '-' then - !y else !y) in
    if Option.is_none (Time.Date.of_days days) then invalid "out of range";
    days
  end

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

let form ~zoned =
  invalid
    (if zoned then "not YYYY-MM-DDThh:mm:ss with an offset"
     else "not YYYY-MM-DDThh:mm:ss without an offset")

(* [datetime u ~zoned] reads a datetime, with an offset iff [zoned], as ticks of
   [u] since 1970-01-01 00:00:00, in UTC when it has an offset. *)
let datetime u ~zoned b pos len (a : int64s) k =
  let stop = pos + len in
  if len < 19 then form ~zoned;
  let days = date_at b pos in
  let t = get b (pos + 10) in
  let hh = two_digits b (pos + 11) and mm = two_digits b (pos + 14) in
  let ss = two_digits b (pos + 17) in
  if
    (t <> 'T' && t <> ' ')
    || hh < 0 || mm < 0 || ss < 0
    || get b (pos + 13) <> ':'
    || get b (pos + 16) <> ':'
  then form ~zoned;
  if hh > 23 || mm > 59 || ss > 59 then invalid "not a time of day";
  let i = ref (pos + 19) and ns = ref 0 in
  if !i < stop && get b !i = '.' then begin
    incr i;
    let first = !i in
    while !i < stop && !i - first < 9 && is_digit (get b !i) do
      ns := (10 * !ns) + digit b !i;
      incr i
    done;
    if !i = first then form ~zoned;
    for _ = !i - first to 8 do
      ns := 10 * !ns
    done
  end;
  let offset =
    if not zoned then 0
    else if !i < stop && get b !i = 'Z' then begin
      incr i;
      0
    end
    else if !i + 6 <= stop then begin
      let s = get b !i in
      let oh = two_digits b (!i + 1) and om = two_digits b (!i + 4) in
      if (s <> '+' && s <> '-') || oh < 0 || om < 0 || get b (!i + 3) <> ':'
      then form ~zoned;
      if oh > 23 || om > 59 then invalid "not an offset";
      i := !i + 6;
      let o = (3600 * oh) + (60 * om) in
      if s = '-' then -o else o
    end
    else form ~zoned
  in
  if !i <> stop then form ~zoned;
  let secs = (86400 * days) + (3600 * hh) + (60 * mm) + ss - offset in
  let per = per_second u in
  let div = 1_000_000_000 / per in
  if !ns mod div <> 0 then invalid (not_whole u);
  if per < 1_000_000_000 then
    A1.unsafe_set a k (Int64.of_int ((secs * per) + (!ns / div)))
  else begin
    if secs > 9223372036 || secs < -9223372037 then invalid "out of range";
    (* In this range the ticks wrap at most once, away from the sign of
       [secs]. *)
    let v =
      Int64.add
        (Int64.mul (Int64.of_int secs) 1_000_000_000L)
        (Int64.of_int !ns)
    in
    if secs >= 0 <> (Int64.compare v 0L >= 0) then invalid "out of range";
    A1.unsafe_set a k v
  end

(* Parsing *)

let by = "Column.parse"
let err fmt = Format.kasprintf invalid_arg fmt

(* [rows r valid read] calls [read b pos len i] on the bytes of each row [i] of
   [r] that [valid] holds, in order, and is the first row where [read] raises
   [Invalid] with the reason. *)
let rows r valid read =
  let n = Nx_ragged.length r in
  let scan o b is_null =
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
      scan o b (fun i -> A1.unsafe_get m i = 0)

(* [fixed kind zero r valid read] is the array that [read] fills at each row
   [rows] reads, [zero] under a null. *)
let fixed kind zero r valid read =
  let a = A1.create kind Bigarray.c_layout (Nx_ragged.length r) in
  A1.fill a zero;
  match rows r valid (fun b pos len i -> read b pos len a i) with
  | Some e -> Error e
  | None -> Ok (Nx.of_bigarray (Bigarray.genarray_of_array1 a))

let parse (Type.Any ty as any) c =
  let r =
    match (Column.type_ c, Column.data c) with
    | (Any String | Any Binary), Bytes r -> r
    | Any t, _ -> err "Column.parse: %a is neither string nor binary" Type.pp t
  in
  let valid = Column.valid c in
  let column d = Column.make any ?valid ~length:(Column.length c) d in
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
      let read b pos len = Bool.to_int (bool b pos len) in
      Result.map (cast Nx.bool) (of_int read)
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
  | Decimal { precision; scale } ->
      Result.map keep (of_int (decimal ~precision ~scale))
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

(* Printing *)

let pp_text ppf s = Format.pp_print_string ppf s
let rec pow10_64 k = if k = 0 then 1L else Int64.mul 10L (pow10_64 (k - 1))

(* [at_scale scale d] is [d], which is exact at [scale], written with [scale]
   digits after the point. *)
let at_scale scale d =
  let u = Decimal.unscaled d and s = Decimal.scale d in
  let unscaled =
    if s <= scale then Int64.mul u (pow10_64 (scale - s))
    else Int64.div u (pow10_64 (s - scale))
  in
  Decimal.v ~unscaled ~scale

let ns_per_second = 1_000_000_000L
let floor_div a b = Int64.(if rem a b < 0L then pred (div a b) else div a b)

(* [pp_instant ~zoned] writes [YYYY-MM-DDThh:mm:ss], the fewest fraction digits
   that are exact, and [Z] when [zoned]. *)
let pp_instant ~zoned ppf t =
  let ns = Time.to_ns t in
  let secs = floor_div ns ns_per_second in
  let frac = Int64.to_int (Int64.sub ns (Int64.mul secs ns_per_second)) in
  let days = Int64.to_int (floor_div secs 86400L) in
  let sod =
    Int64.to_int (Int64.sub secs (Int64.mul (Int64.of_int days) 86400L))
  in
  Format.fprintf ppf "%a" Time.Date.pp (Option.get (Time.Date.of_days days));
  Format.fprintf ppf "T%02d:%02d:%02d" (sod / 3600)
    (sod / 60 mod 60)
    (sod mod 60);
  if frac > 0 then begin
    let digits = Printf.sprintf "%09d" frac and n = ref 9 in
    while digits.[!n - 1] = '0' do
      decr n
    done;
    Format.fprintf ppf ".%s" (String.sub digits 0 !n)
  end;
  if zoned then pp_text ppf "Z"

let pp : type a. a Type.t -> Format.formatter -> a -> unit =
 fun ty ppf v ->
  match (ty, Type.kind ty) with
  | Decimal { scale; _ }, _ -> Decimal.pp ppf (at_scale scale v)
  | Binary, _ -> pp_text ppf (v :> string)
  | Datetime { zone; _ }, _ -> pp_instant ~zoned:(zone <> None) ppf v
  | (Clock _ | Duration _ | List _ | Record _ | Tensor _ | Ext _), k ->
      Type.pp_lit k ppf v
  | _, Bool -> Format.pp_print_bool ppf v
  | _, Int -> Format.pp_print_int ppf v
  | _, String -> pp_text ppf v
  | _, Date -> Time.Date.pp ppf v
  | _, k -> Type.pp_lit k ppf v

(* Cells *)

let is_null c i =
  match Column.validity c with
  | None -> false
  | Some v ->
      not (Nx.item [ 0 ] (Nx_bits.to_bool (Nx_bits.sub v ~offset:i ~length:1)))

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
      | Decimal { scale; _ }, _ ->
          Decimal.pp ppf (Decimal.v ~unscaled:(row_int64 x i) ~scale)
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

(* [exponent_of s] is the exponent of the [%e] text [s]. *)
let exponent_of s =
  let e = String.index s 'e' in
  int_of_string (String.sub s (e + 1) (String.length s - e - 1))

(* [exponent x] is the decimal exponent of [x] rounded to six significant
   digits: [2] for [99.99996], which rounds to [100.000]. *)
let exponent x = exponent_of (Printf.sprintf "%.*e" (significant - 1) x)

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
