(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let day_s = 86_400

(* Instants *)

(* [shift u d] is [u + d], or [None] beyond [int64]'s range. *)
let shift u d =
  let v = Int64.add u (Int64.of_int d) in
  if Bool.equal (d >= 0) (Int64.compare v u >= 0) then Some v else None

(* [sub_sat u d] is [u - d], clamped to [int64]'s range. *)
let sub_sat u d =
  match shift u (-d) with
  | Some v -> v
  | None ->
      if (d > 0) [@mutate off "subtracting 0 never overflows"] then
        Int64.min_int
      else Int64.max_int

(* Calendar *)

(* [month_start y m] is the day that starts month [m] of year [y], counted from
   1970-01-01. Rules only ask for years near one 400-year cycle, well within the
   range of dates. *)
let month_start y m =
  Time.Date.to_days (Option.get (Time.Date.of_civil (y, m, 1)))

(* The Gregorian calendar repeats every 400 years, 146_097 days. *)
let cycle_days = 146_097

(* Moving [days] by whole cycles keeps it within the range of dates. *)
let starts_month days =
  let date = Option.get (Time.Date.of_days (days mod cycle_days)) in
  let _, _, d = Time.Date.to_civil date in
  d = 1

(* [year_near days] is within two years of the year of the day [days] since
   1970: a Gregorian year averages 146_097 / 400 days. *)
let year_near days = 1970 + (days * 400 / cycle_days)

(* Rules, the POSIX TZ strings of TZif footers *)

(* The dates of a rule: [Jn], day 1 to 365 with February 29 never counted; [n],
   day 0 to 365; and [Mm.w.d], weekday [d] of week [w] of month [m]. *)
type date =
  | Julian of int
  | Yday of int
  | Weekday of { month : int; week : int; day : int }

(* A change happens [time] seconds after the midnight that starts [date], on the
   clock in effect before it. *)
type change = { date : date; time : int }
type daylight = { dst : int; start : change; stop : change }

(* Offsets of standard and daylight saving time, in seconds east of UTC. *)
type rule = { std : int; daylight : daylight option }

let day_of_date year = function
  | Julian n ->
      if n < 60 then month_start year 1 + n - 1 else month_start year 3 + n - 60
  | Yday n -> month_start year 1 + n
  | Weekday { month; week; day } ->
      let first = month_start year month in
      let next =
        if month = 12 then month_start (year + 1) 1
        else month_start year (month + 1)
      in
      (* 1970-01-01 is a Thursday, weekday 4. *)
      let d =
        first + ((((day - first - 4) mod 7) + 7) mod 7) + (7 * (week - 1))
      in
      if d >= next then d - 7 else d

let cycle_s = cycle_days * day_s

(* [reduce u] is [u] less whole cycles, in (-cycle_s, cycle_s), and a year near
   it. *)
let reduce u =
  let r = Int64.to_int (Int64.rem u (Int64.of_int cycle_s)) in
  (r, year_near (r / day_s))

(* The changes of [d] from three years before [year] to three years after, as
   [(time, offset)] in time order. [year] is within two years of the instant's,
   and rule hours reach 167, so a year's changes can fall in the next year: the
   seven years surround the instant with changes. At equal times a stop precedes
   a start, so that daylight saving time all year never shows a standard time of
   no length. *)
let changes std d year =
  let at y c offset = (day_of_date y c.date * day_s) + c.time - offset in
  let events = ref [] in
  for y = year - 3 to year + 3 do
    events :=
      (at y d.stop d.dst, 0, std) :: (at y d.start std, 1, d.dst) :: !events
  done;
  let by_time (t, o, _) (t', o', _) =
    match Int.compare t t' with 0 -> Int.compare o o' | c -> c
  in
  List.map (fun (t, _, offset) -> (t, offset)) (List.sort by_time !events)

let rule_offset rule u =
  match rule.daylight with
  | None -> rule.std
  | Some d ->
      let r, year = reduce u in
      List.fold_left
        (fun o (t, o') -> if t <= r then o' else o)
        rule.std (changes rule.std d year)

(* [rule_next rule u] is the first instant after [u] where [rule] changes the
   offset, and the offset from there. Each event is checked against the rule's
   offset at its instant: under daylight saving time all year, the last stop of
   the window has no start to cancel it, yet changes nothing. *)
let rule_next rule u =
  match rule.daylight with
  | None -> None
  | Some d ->
      let r, year = reduce u in
      let current = rule_offset rule u in
      let rec find = function
        | [] -> None
        | (t, _) :: rest
          when (t <= r) [@mutate off "an event at r already sets current"] ->
            find rest
        | (t, _) :: rest -> (
            match shift u (t - r) with
            | None -> None
            | Some v ->
                let o = rule_offset rule v in
                if o = current then find rest else Some (v, o))
      in
      find (changes rule.std d year)

(* Zones *)

type zone = {
  name : string;
  first : int;
  times : int64 array;
  offsets : int array;
  rule : rule option;
  lowest : int;
  highest : int;
}

let utc =
  {
    name = "UTC";
    first = 0;
    times = [||];
    offsets = [||];
    rule = None;
    lowest = 0;
    highest = 0;
  }

let name z = z.name

(* [count z t] is the number of transitions of [z] at or before [t]. *)
let count z t =
  let rec search lo hi =
    if lo >= hi then lo
    else
      let mid = (lo + hi) / 2 in
      if Int64.compare z.times.(mid) t <= 0 then search (mid + 1) hi
      else search lo mid
  in
  search 0 (Array.length z.times)

let offset_s z t =
  let n = Array.length z.times in
  match count z t with
  | 0 when n > 0 -> z.first
  | i when i < n -> z.offsets.(i - 1)
  | _ -> (
      match z.rule with
      | Some rule -> rule_offset rule t
      | None -> if n = 0 then z.first else z.offsets.(n - 1))

(* [next_change z t] is the first instant after [t] where the offset of [z]
   changes, and the offset from there. *)
let next_change z t =
  let n = Array.length z.times in
  let current = offset_s z t in
  let rec explicit i =
    if i < n then
      if z.offsets.(i) <> current then Some (z.times.(i), z.offsets.(i))
      else explicit (i + 1)
    else
      match z.rule with
      | None -> None
      | Some rule ->
          rule_next rule (if n = 0 then t else Int64.max t z.times.(n - 1))
  in
  explicit (count z t)

type local =
  | Unique of int
  | Ambiguous of { before : int; after : int }
  | Gap of { before : int; after : int }

(* The clock reads [t] at [t - o] for offsets [o] of [z], so from [t - highest]
   to [t - lowest]. [local] walks the pieces of constant offset over that
   window: a piece of offset [o] holds a reading if it holds [t - o], and a
   boundary [b] from [o] to [o'] skips [t] if [t - o >= b > t - o']. Clamping [t
   - o] to [int64]'s range gives the offsets beyond it. The window starts with
   the clock at or before [t] and ends with it at or after [t], so it holds a
   reading or a skip. Without a reading, the clock reads less than [t] up to the
   first skip and more than [t] from the last, so [after] comes from the first
   skip and [before] from the last. *)
let local z t =
  let hi = sub_sat t z.lowest in
  let rec walk start o ((readings, latest, earliest) as found) skips =
    let next =
      match next_change z start with
      | Some (b, _) as next when Int64.compare b hi <= 0 -> next
      | _ -> None
    in
    let u = sub_sat t o in
    let reads =
      Int64.compare start u <= 0
      && match next with None -> true | Some (b, _) -> Int64.compare u b < 0
    in
    let found =
      if reads then (readings + 1, min latest o, max earliest o) else found
    in
    match next with
    | None -> (found, skips)
    | Some (b, o') ->
        let below =
          Int64.compare (sub_sat t o') b < 0
            [@@mutate off "the clock reads t at b"]
        in
        let skips =
          if Int64.compare u b >= 0 && below then
            match skips with
            | None -> Some (o, o')
            | Some (_, after) -> Some (o, after)
          else skips
        in
        walk b o' found skips
  in
  let lo = sub_sat t z.highest in
  match walk lo (offset_s z lo) (0, max_int, min_int) None with
  | (1, o, _), _ -> Unique o
  | (0, _, _), Some (before, after) -> Gap { before; after }
  | (0, _, _), None -> assert false
  | (_, after, before), _ -> Ambiguous { before; after }

(* Decoding TZif files, RFC 9636 *)

exception Malformed of int * int * string

let malformed first last fmt =
  Format.kasprintf (fun msg -> raise (Malformed (first, last, msg))) fmt

type decoder = { s : string; mutable pos : int }

let need d len what =
  if len > String.length d.s - d.pos then
    malformed d.pos
      (d.pos + len - 1)
      "the file ends after %d bytes, inside %s" (String.length d.s) what

let u8 d =
  let v = String.get_uint8 d.s d.pos in
  d.pos <- d.pos + 1;
  v

let i32 d =
  let v = Int32.to_int (String.get_int32_be d.s d.pos) in
  d.pos <- d.pos + 4;
  v

let time d size =
  if size = 4 then Int64.of_int (i32 d)
  else
    let v = String.get_int64_be d.s d.pos in
    d.pos <- d.pos + 8;
    v

type header = {
  version : int; (* 1 to 4: a later version is read as version 4. *)
  later : bool; (* The version is above 4. *)
  isutcnt : int;
  isstdcnt : int;
  leapcnt : int;
  timecnt : int;
  typecnt : int;
  charcnt : int;
}

let header d =
  need d 44 "a header";
  let at = d.pos in
  if String.sub d.s at 4 <> "TZif" then
    malformed at (at + 3) "the header does not start with \"TZif\"";
  let version, later =
    match d.s.[at + 4] with
    | '\000' -> (1, false)
    | '2' .. '4' as c -> (Char.code c - Char.code '0', false)
    | '5' .. '\255' -> (4, true)
    | c -> malformed (at + 4) (at + 4) "unknown version %C" c
  in
  d.pos <- at + 20;
  let counts = Array.init 6 (fun _ -> i32 d land 0xFFFF_FFFF) in
  let check field ok msg =
    let first = at + 20 + (4 * field) in
    if not ok then malformed first (first + 3) "%s" msg
  in
  let isutcnt = counts.(0) and isstdcnt = counts.(1) and typecnt = counts.(4) in
  check 0
    (isutcnt = 0 || isutcnt = typecnt)
    "the count of UT/local indicators is neither 0 nor the count of time types";
  check 1
    (isstdcnt = 0 || isstdcnt = typecnt)
    "the count of standard/wall indicators is neither 0 nor the count of time \
     types";
  check 4 (typecnt <> 0) "the count of time types is 0";
  check 5 (counts.(5) <> 0) "the count of designation bytes is 0";
  {
    version;
    later;
    isutcnt;
    isstdcnt;
    leapcnt = counts.(2);
    timecnt = counts.(3);
    typecnt;
    charcnt = counts.(5);
  }

let block_length h size =
  (h.timecnt * (size + 1))
  + (h.typecnt * 6) + h.charcnt
  + (h.leapcnt * (size + 4))
  + h.isstdcnt + h.isutcnt

(* [footer_rule ~version ~at tz] parses the POSIX TZ string [tz], found at byte
   [at], with RFC 9636's extension of rule hours from version 3 on. *)
let footer_rule ~version ~at tz =
  let len = String.length tz in
  let i = ref 0 in
  let fail fmt =
    let p = at + max 0 (min !i (len - 1)) in
    malformed p p ("footer: " ^^ fmt)
  in
  let peek () = if !i < len then Some tz.[!i] else None in
  let accept c =
    let here = peek () = Some c in
    if here then incr i;
    here
  in
  let expect c = if not (accept c) then fail "expected %C" c in
  let number ~what ~lo ~hi =
    let start = !i in
    while !i < len && tz.[!i] >= '0' && tz.[!i] <= '9' do
      incr i
    done;
    if !i = start then fail "expected the %s" what;
    let digits = String.sub tz start (!i - start) in
    match int_of_string_opt digits with
    | Some n when lo <= n && n <= hi -> n
    | _ -> fail "the %s %s is not in [%d, %d]" what digits lo hi
  in
  let designation () =
    let quoted = accept '<' in
    let start = !i in
    let is_char = function
      | 'A' .. 'Z' | 'a' .. 'z' -> true
      | '0' .. '9' | '+' | '-' -> quoted
      | _ -> false
    in
    while !i < len && is_char tz.[!i] do
      incr i
    done;
    let abbr = String.sub tz start (!i - start) in
    if quoted then expect '>';
    if abbr = "" then fail "expected a designation";
    if String.length abbr < 3 then
      fail "the designation %S has fewer than 3 characters" abbr;
    abbr
  in
  (* [[+|-]hh[:mm[:ss]]], in seconds. *)
  let clock ~signed ~max_hours =
    let sign =
      if signed && accept '-' then -1
      else (
        if signed then ignore (accept '+');
        1)
    in
    let h = number ~what:"hour" ~lo:0 ~hi:max_hours in
    let m, s =
      if accept ':' then
        let m = number ~what:"minute" ~lo:0 ~hi:59 in
        let s = if accept ':' then number ~what:"second" ~lo:0 ~hi:59 else 0 in
        (m, s)
      else (0, 0)
    in
    sign * ((h * 3600) + (m * 60) + s)
  in
  let change () =
    let date =
      if accept 'J' then Julian (number ~what:"day" ~lo:1 ~hi:365)
      else if accept 'M' then (
        let month = number ~what:"month" ~lo:1 ~hi:12 in
        expect '.';
        let week = number ~what:"week" ~lo:1 ~hi:5 in
        expect '.';
        let day = number ~what:"weekday" ~lo:0 ~hi:6 in
        Weekday { month; week; day })
      else Yday (number ~what:"day" ~lo:0 ~hi:365)
    in
    let time =
      if not (accept '/') then 7200
      else if version >= 3 then clock ~signed:true ~max_hours:167
      else clock ~signed:false ~max_hours:24
    in
    { date; time }
  in
  (* A POSIX offset counts seconds west of UTC. *)
  let offset () = -clock ~signed:true ~max_hours:24 in
  if len = 0 then None
  else
    let std =
      ignore (designation ());
      offset ()
    in
    if !i = len then Some { std; daylight = None }
    else
      let name = designation () in
      let dst =
        match peek () with None | Some ',' -> std + 3600 | Some _ -> offset ()
      in
      if !i = len then fail "daylight saving time %S has no rule" name;
      expect ',';
      let start = change () in
      expect ',';
      let stop = change () in
      if !i <> len then fail "unexpected %C" tz.[!i];
      Some { std; daylight = Some { dst; start; stop } }

let decode ~name s =
  let d = { s; pos = 0 } in
  let h1 = header d in
  let h, size =
    if h1.version = 1 then (h1, 4)
    else begin
      let length = block_length h1 4 in
      need d length "the version 1 data block";
      d.pos <- d.pos + length;
      let at = d.pos in
      let h = header d in
      if s.[at + 4] <> s.[4] then
        malformed (at + 4) (at + 4)
          "the second header's version %C differs from the first's, %C"
          s.[at + 4]
          s.[4];
      (h, 8)
    end
  in
  need d (block_length h size) "the data block";
  let times_at = d.pos in
  let time_bytes k =
    let at = times_at + (k * size) in
    (at, at + size - 1)
  in
  let times = Array.init h.timecnt (fun _ -> time d size) in
  for k = 1 to h.timecnt - 1 do
    if Int64.compare times.(k) times.(k - 1) <= 0 then
      let first, last = time_bytes k in
      malformed first last "transition %d is not after the previous one" k
  done;
  let types =
    Array.init h.timecnt (fun k ->
        let ty = u8 d in
        if ty >= h.typecnt then
          malformed (d.pos - 1) (d.pos - 1)
            "transition %d has time type %d, beyond the %d time types" k ty
            h.typecnt;
        ty)
  in
  let records =
    Array.init h.typecnt (fun k ->
        let at = d.pos in
        let offset = i32 d in
        let is_dst = u8 d in
        let idx = u8 d in
        if offset = -0x8000_0000 then
          malformed at (at + 3) "time type %d has the UT offset -2^31" k;
        if is_dst > 1 then
          malformed (at + 4) (at + 4)
            "time type %d has the DST indicator %d, not 0 or 1" k is_dst;
        (at + 5, offset, idx))
  in
  let chars = String.sub s d.pos h.charcnt in
  d.pos <- d.pos + h.charcnt;
  let type_offsets =
    Array.mapi
      (fun k (at, offset, idx) ->
        if idx >= h.charcnt then
          malformed at at
            "time type %d has the designation index %d, beyond the %d \
             designation bytes"
            k idx h.charcnt;
        if not (String.contains_from chars idx '\000') then
          malformed at at "time type %d has a designation with no NUL" k;
        offset)
      records
  in
  let leaps_at = d.pos in
  let leaps =
    Array.init h.leapcnt (fun _ ->
        let occurrence = time d size in
        (occurrence, i32 d))
  in
  let bad_leap k fmt =
    let at = leaps_at + (k * (size + 4)) in
    malformed at (at + size + 3) fmt
  in
  Array.iteri
    (fun k (occurrence, corr) ->
      if k = 0 && Int64.compare occurrence 0L < 0 then
        bad_leap k "the first leap second occurs before 1970";
      if k > 0 && Int64.compare occurrence (fst leaps.(k - 1)) <= 0 then
        bad_leap k "leap second %d does not occur after the previous one" k;
      (* The correction before the record, unspecified before the first one of a
         version 4 table truncated at the start. *)
      let previous =
        if k > 0 then Some (snd leaps.(k - 1))
        else if abs corr = 1 then Some 0
        else if h.version >= 4 then None
        else
          bad_leap k "the first leap-second correction is %d, not 1 or -1" corr
      in
      let expiry = h.version >= 4 && k = h.leapcnt - 1 in
      match previous with
      | None -> ()
      | Some p when expiry && corr = p -> ()
      | Some p when abs (corr - p) <> 1 ->
          bad_leap k
            "leap second %d has the correction %d, which does not differ from \
             the previous one, %d, by 1"
            k corr p
      | Some p ->
          (* Less the smaller of the corrections around it, a leap second's UNIX
             leap time is the midnight that starts the next month. *)
          let start = Int64.sub occurrence (Int64.of_int (min p corr)) in
          let month_start =
            Int64.rem start (Int64.of_int day_s) = 0L
            && starts_month
                 (Int64.to_int (Int64.div start (Int64.of_int day_s)))
          in
          if not month_start then
            bad_leap k "leap second %d does not end a month" k)
    leaps;
  let indicators count what =
    Array.init count (fun k ->
        let v = u8 d in
        if v > 1 then
          malformed (d.pos - 1) (d.pos - 1)
            "time type %d has the %s indicator %d, not 0 or 1" k what v;
        v = 1)
  in
  let isstd = indicators h.isstdcnt "standard/wall" in
  let isut_at = d.pos in
  let isut = indicators h.isutcnt "UT/local" in
  Array.iteri
    (fun k ut ->
      if ut && not (h.isstdcnt > 0 && isstd.(k)) then
        malformed (isut_at + k) (isut_at + k)
          "time type %d is in UT but not in standard time" k)
    isut;
  (* Transition times count leap seconds when the file has a leap-second table:
     the correction in effect at each is removed. *)
  let corr_at k t =
    let rec last i =
      if i < 0 then None
      else if Int64.compare (fst leaps.(i)) t <= 0 then Some (snd leaps.(i))
      else last (i - 1)
    in
    match last (h.leapcnt - 1) with
    | Some c -> c
    | None when h.leapcnt = 0 || abs (snd leaps.(0)) = 1 -> 0
    | None ->
        let first, last = time_bytes k in
        malformed first last
          "transition %d precedes the leap-second table, whose correction is \
           unknown there"
          k
  in
  let times =
    Array.mapi (fun k t -> Int64.sub t (Int64.of_int (corr_at k t))) times
  in
  let rule =
    if h.version = 1 then begin
      if d.pos <> String.length s then
        malformed d.pos
          (String.length s - 1)
          "bytes follow the data block of a version 1 file";
      None
    end
    else begin
      need d 1 "the footer";
      if s.[d.pos] <> '\n' then
        malformed d.pos d.pos "the footer does not start with a newline";
      let at = d.pos + 1 in
      match String.index_from_opt s at '\n' with
      | None ->
          malformed d.pos
            (String.length s - 1)
            "the footer does not end with a newline"
      | Some e ->
          let tz = String.sub s at (e - at) in
          Option.iter
            (fun i -> malformed (at + i) (at + i) "the footer holds a NUL byte")
            (String.index_opt tz '\000');
          if (not h1.later) && e + 1 <> String.length s then
            malformed (e + 1) (String.length s - 1) "bytes follow the footer";
          footer_rule ~version:h.version ~at tz
    end
  in
  (* A transition at a positive leap second, 23:59:60, falls in the POSIX second
     of one at 23:59:59, and replaces it. Leap seconds are months apart, so no
     other transitions meet. *)
  let kept =
    Array.of_list
      (List.filter
         (fun k ->
           k = h.timecnt - 1 || Int64.compare times.(k) times.(k + 1) < 0)
         (List.init h.timecnt Fun.id))
  in
  let times = Array.map (Array.get times) kept in
  let offsets = Array.map (fun k -> type_offsets.(types.(k))) kept in
  (* Some zic versions write a footer that disagrees with the last transition,
     as zic 2022g does for America/Ojinaga in slim files. As tzcode reads such a
     file, the last transition's offset holds until the rule first changes the
     offset after it. *)
  let times, offsets, rule =
    let n = Array.length times in
    match rule with
    | Some r when n > 0 && rule_offset r times.(n - 1) <> offsets.(n - 1) -> (
        match rule_next r times.(n - 1) with
        | Some (u, o) ->
            (Array.append times [| u |], Array.append offsets [| o |], rule)
        | None -> (times, offsets, None))
    | _ -> (times, offsets, rule)
  in
  let all =
    Array.to_list type_offsets
    @
    match rule with
    | None -> []
    | Some { std; daylight = None } -> [ std ]
    | Some { std; daylight = Some d } -> [ std; d.dst ]
  in
  {
    name;
    first = type_offsets.(0);
    times;
    offsets;
    rule;
    lowest = List.fold_left min max_int all;
    highest = List.fold_left max min_int all;
  }

(* Databases *)

type db = string

(* A [Sys_error] about a path starts with it. *)
let sys_error path msg =
  let prefix = path ^ ": " in
  let msg =
    if String.starts_with ~prefix msg then
      let n = String.length prefix in
      String.sub msg n (String.length msg - n)
    else msg
  in
  Error.v ~file:path msg

let of_dir dir =
  match Sys.is_directory dir with
  | true -> Ok dir
  | false -> Error (Error.v ~file:dir "not a directory")
  | exception Sys_error msg -> Error (sys_error dir msg)

let system () =
  match Sys.getenv_opt "TZDIR" with
  | Some dir when dir <> "" -> of_dir dir
  | Some _ | None -> of_dir "/usr/share/zoneinfo"

(* Windows opens a device, in any directory, for a segment whose part before its
   first dot names one. tz names no zone so. *)
let is_device segment =
  match String.uppercase_ascii (List.hd (String.split_on_char '.' segment)) with
  | "CON" | "PRN" | "AUX" | "NUL" -> true
  | stem when String.length stem = 4 -> (
      match (String.sub stem 0 3, stem.[3]) with
      | ("COM" | "LPT"), '0' .. '9' -> true
      | _ -> false)
  | _ -> false

(* The characters of tz's zone names. Without "." and ".." segments and device
   names, a name designates a path under the database's directory on every
   system. *)
let is_name name =
  let is_char = function
    | 'A' .. 'Z' | 'a' .. 'z' | '0' .. '9' | '.' | '-' | '_' | '+' -> true
    | _ -> false
  in
  let is_segment s =
    s <> "" && s <> "." && s <> ".." && String.for_all is_char s
    && not (is_device s)
  in
  List.for_all is_segment (String.split_on_char '/' name)

let find dir name =
  if not (is_name name) then
    Error (Error.v (Printf.sprintf "%S is not a zone name" name))
  else
    let path = Filename.concat dir name in
    match Sys.is_directory path with
    | true -> Error (Error.v ~file:path "a directory, not a zone")
    | false | (exception Sys_error _) -> (
        match In_channel.with_open_bin path In_channel.input_all with
        | exception Sys_error msg -> Error (sys_error path msg)
        | s -> (
            match decode ~name s with
            | zone -> Ok zone
            | exception Malformed (first, last, msg) ->
                Error (Error.v ~file:path ~bytes:(first, last) msg)))
