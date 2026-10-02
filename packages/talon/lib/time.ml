(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let ns_per_us = 1_000L
let ns_per_ms = 1_000_000L
let ns_per_s = 1_000_000_000L
let ns_per_min = 60_000_000_000L
let ns_per_h = 3_600_000_000_000L
let ns_per_day = 86_400_000_000_000L

(* [scale u n] is [n * u] for [u > 0], or [None] beyond [int64]'s range. *)
let scale u n =
  if
    Int64.compare n (Int64.div Int64.max_int u) > 0
    || Int64.compare n (Int64.div Int64.min_int u) < 0
  then None
  else Some (Int64.mul n u)

(* [floor_div n u] is [n / u] for [u > 0], rounded toward negative infinity. *)
let floor_div n u =
  let q = Int64.div n u in
  if Int64.compare (Int64.rem n u) 0L < 0 then Int64.pred q else q

(* Dates *)

type date = int

let min_days = Int32.to_int Int32.min_int
let max_days = Int32.to_int Int32.max_int
let is_leap y = (y mod 4 = 0 && y mod 100 <> 0) || y mod 400 = 0

let days_in_month y = function
  | 2 -> if is_leap y then 29 else 28
  | 4 | 6 | 9 | 11 -> 30
  | _ -> 31

(* Howard Hinnant's [days_from_civil]: a year starts on March 1, so that the
   leap day ends it, and an era is a 400-year cycle of 146_097 days. *)
let days_of_civil y m d =
  let y = if m <= 2 then y - 1 else y in
  let era =
    (if (y >= 0) [@mutate off "-399 / 400 is 0 too"] then y else y - 399) / 400
  in
  let yoe = y - (era * 400) in
  let doy = (((153 * ((m + 9) mod 12)) + 2) / 5) + d - 1 in
  let doe = (yoe * 365) + (yoe / 4) - (yoe / 100) + doy in
  (era * 146_097) + doe - 719_468

(* Howard Hinnant's [civil_from_days], the inverse of [days_of_civil]. *)
let civil_of_days days =
  let z = days + 719_468 in
  let era =
    (if (z >= 0) [@mutate off "-146_096 / 146_097 is 0 too"] then z
     else z - 146_096)
    / 146_097
  in
  let doe = z - (era * 146_097) in
  let yoe = (doe - (doe / 1460) + (doe / 36_524) - (doe / 146_096)) / 365 in
  let doy = doe - ((365 * yoe) + (yoe / 4) - (yoe / 100)) in
  let mp = ((5 * doy) + 2) / 153 in
  let d = doy - (((153 * mp) + 2) / 5) + 1 in
  let m = if mp < 10 then mp + 3 else mp - 9 in
  let y = yoe + (era * 400) in
  ((if m <= 2 then y + 1 else y), m, d)

let min_year, _, _ = civil_of_days min_days
let max_year, _, _ = civil_of_days max_days

module Date = struct
  type t = date

  let of_days n = if n < min_days || n > max_days then None else Some n
  let to_days t = t

  (* Bounding the year first keeps [days_of_civil] from overflowing. *)
  let of_civil (y, m, d) =
    if m < 1 || m > 12 || d < 1 || d > days_in_month y m then None
    else if y < min_year || y > max_year then None
    else of_days (days_of_civil y m d)

  let to_civil = civil_of_days
  let equal = Int.equal
  let compare = Int.compare

  let pp ppf t =
    let y, m, d = to_civil t in
    if y >= 0 && y <= 9999 then Format.fprintf ppf "%04d-%02d-%02d" y m d
    else Format.fprintf ppf "%+05d-%02d-%02d" y m d
end

(* Instants *)

type instant = int64

let of_ns n = n
let of_us = scale ns_per_us
let of_ms = scale ns_per_ms
let of_s = scale ns_per_s
let to_ns t = t
let to_us t = floor_div t ns_per_us
let to_ms t = floor_div t ns_per_ms
let to_s t = floor_div t ns_per_s
let equal = Int64.equal
let compare = Int64.compare

let pp ppf t =
  let days = floor_div t ns_per_day in
  let ns = Int64.to_int (Int64.sub t (Int64.mul days ns_per_day)) in
  let s = ns / 1_000_000_000 and frac = ns mod 1_000_000_000 in
  Format.fprintf ppf "%aT%02d:%02d:%02d" Date.pp (Int64.to_int days) (s / 3600)
    (s / 60 mod 60)
    (s mod 60);
  if frac mod 1_000_000 = 0 then (
    if frac <> 0 then Format.fprintf ppf ".%03d" (frac / 1_000_000))
  else if frac mod 1000 = 0 then Format.fprintf ppf ".%06d" (frac / 1000)
  else Format.fprintf ppf ".%09d" frac

(* Spans *)

type span = int64

module Span = struct
  type t = span

  let make name u n =
    match scale u (Int64.of_int n) with
    | Some d -> d
    | None ->
        invalid_arg (Printf.sprintf "Time.Span.%s: %d is out of range" name n)

  let ns = Int64.of_int
  let us = make "us" ns_per_us
  let ms = make "ms" ns_per_ms
  let s = make "s" ns_per_s
  let minutes = make "minutes" ns_per_min
  let hours = make "hours" ns_per_h
  let days = make "days" ns_per_day
  let of_ns n = n
  let of_us = scale ns_per_us
  let of_ms = scale ns_per_ms
  let of_s = scale ns_per_s
  let to_ns d = d
  let to_us d = floor_div d ns_per_us
  let to_ms d = floor_div d ns_per_ms
  let to_s d = floor_div d ns_per_s
  let equal = Int64.equal
  let compare = Int64.compare

  let units =
    [
      (ns_per_h, "h");
      (ns_per_min, "m");
      (ns_per_s, "s");
      (ns_per_ms, "ms");
      (ns_per_us, "us");
      (1L, "ns");
    ]

  (* The magnitude is read unsigned, so that [Int64.abs Int64.min_int], which is
     [Int64.min_int], reads as 2{^63}. *)
  let pp ppf d =
    let rec components m = function
      | [] -> ()
      | (u, name) :: units ->
          let n = Int64.unsigned_div m u in
          if not (Int64.equal n 0L) then Format.fprintf ppf "%Lu%s" n name;
          components (Int64.unsigned_rem m u) units
    in
    if Int64.compare d 0L < 0 then Format.pp_print_char ppf '-';
    if Int64.equal d 0L then Format.pp_print_string ppf "0s"
    else components (Int64.abs d) units
end

(* Steps *)

type step = Months of int | Weeks of int | Days of int | Exact of span

let pp_step ppf = function
  | Months n -> Format.fprintf ppf "%dmo" n
  | Weeks n -> Format.fprintf ppf "%dw" n
  | Days n -> Format.fprintf ppf "%dd" n
  | Exact d -> Span.pp ppf d
