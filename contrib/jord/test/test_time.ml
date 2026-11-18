(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord

let instant s =
  match Time.of_rfc3339 s with Ok t -> t | Error e -> fail e

(* The instants, from Python's calendar.timegm where it reaches (years 1 to
   9999), and 366 days before year 1 for year 0, a leap year of the proleptic
   calendar. A fraction rounds within a float64's spacing. *)
let times =
  cases ~id:fst "RFC 3339's date-times, as POSIX's seconds"
    [
      ("1970-01-01T00:00:00Z", 0.);
      ("1969-12-31T23:59:59Z", -1.);
      ("2024-06-29T10:20:21.024000Z", 1719656421.024);
      ("2024-06-29 17:00:44.921157+00:00", 1719680444.921157);
      ("2024-06-29t00:00:00z", 1719619200.);
      (* RFC 3339's unknown local offset (§4.3) reads as Z. *)
      ("2024-06-29T00:00:00-00:00", 1719619200.);
      ("2024-06-29T23:30:00-01:00", 1719707400.);
      ("2024-01-01T00:30:00+01:00", 1704065400.);
      ("2024-03-01T00:00:00+00:01", 1709251140.);
      ("2000-02-29T00:00:00Z", 951782400.);
      ("1900-03-01T00:00:00Z", -2203891200.);
      ("0000-01-01T00:00:00Z", -62167219200.);
      ("9999-12-31T23:59:59Z", 253402300799.);
    ]
    (fun (s, want) -> equal (float 5e-7) want (instant s))

let leap_seconds =
  test "a leap second reads as the next minute's second 0" (fun () ->
      let next = instant "1999-01-01T00:00:00Z" in
      equal float_exact next (instant "1998-12-31T23:59:60Z");
      equal float_exact (next +. 0.5) (instant "1998-12-31T23:59:60.5Z");
      equal float_exact next (instant "1999-01-01T00:59:60+01:00");
      (* RFC 3339 §5.8's leap second, and in Pacific Standard Time. *)
      equal float_exact
        (instant "1990-12-31T23:59:60Z")
        (instant "1990-12-31T15:59:60-08:00");
      equal float_exact
        (instant "1991-01-01T00:00:00Z")
        (instant "1990-12-31T23:59:60Z"))

(* Every day of four centuries is 86400 seconds after the day before, holds
   its instants until the next midnight, and the day after a month's last is
   refused. *)
let calendar =
  test "each day of 1900 to 2299 follows the one before" (fun () ->
      let last y m =
        match m with
        | 2 ->
            if (y mod 4 = 0 && y mod 100 <> 0) || y mod 400 = 0 then 29
            else 28
        | 4 | 6 | 9 | 11 -> 30
        | _ -> 31
      in
      let day y m d = Printf.sprintf "%04d-%02d-%02dT00:00:00Z" y m d in
      let prev = ref (instant (day 1899 12 31)) in
      for y = 1900 to 2299 do
        for m = 1 to 12 do
          for d = 1 to last y m do
            let t = instant (day y m d) in
            equal ~msg:(day y m d) float_exact (!prev +. 86400.) t;
            equal ~msg:(day y m d) (triple int int int) (y, m, d) (Time.date t);
            equal ~msg:(day y m d) (triple int int int) (y, m, d)
              (Time.date (t +. 86399.999));
            equal ~msg:(day y m d) float_exact t (Time.of_date (y, m, d));
            prev := t
          done;
          is_true ~msg:(day y m (last y m + 1))
            (Result.is_error (Time.of_rfc3339 (day y m (last y m + 1))))
        done
      done)

let dates =
  test "an instant's day, before 1970 and at a leap second" (fun () ->
      let date s = Time.date (instant s) in
      equal (triple int int int) (1969, 12, 31) (date "1969-12-31T23:59:59Z");
      equal (triple int int int) (0, 1, 1) (date "0000-01-01T00:00:00Z");
      equal (triple int int int) (9999, 12, 31) (date "9999-12-31T23:59:59Z");
      equal (triple int int int) (1999, 1, 1)
        (date "1998-12-31T23:59:60.5Z");
      List.iter
        (fun t ->
          raises_match (Exn.invalid_arg ~sub:"Jord.Time.date") (fun () ->
              Time.date t))
        [
          Float.nan;
          Float.infinity;
          Float.neg_infinity;
          1e30;
          0x1p61 *. 86400.;
        ];
      (* The last day it takes: civil_from_days's shift does not overflow. *)
      let y, _, _ = Time.date (Float.pred 0x1p61 *. 86400.) in
      is_true (y > 0))

let of_date =
  test "of_date is the day's first instant, and refuses other days" (fun () ->
      equal float_exact 0. (Time.of_date (1970, 1, 1));
      equal float_exact (-62167219200.) (Time.of_date (0, 1, 1));
      let big = (1 lsl 28) - 1 in
      equal (triple int int int) (big, 12, 31)
        (Time.date (Time.of_date (big, 12, 31)));
      List.iter
        (fun day ->
          raises_match (Exn.invalid_arg ~sub:"Jord.Time.of_date") (fun () ->
              Time.of_date day))
        [
          (2023, 2, 29);
          (2024, 13, 1);
          (2024, 0, 1);
          (2024, 1, 0);
          (1 lsl 28, 1, 1);
          (-(1 lsl 28), 1, 1);
        ])

(* Week 1 of [y] starts on the Monday before its first Thursday: by
   definition, from each day's weekday, with no week arithmetic. *)
let weekday z = (((z + 3) mod 7) + 7) mod 7 (* Monday 0; 1970-01-01 Thu. *)

let first_monday y =
  let jan1 = int_of_float (Time.of_date (y, 1, 1) /. 86400.) in
  let thursday = jan1 + ((3 - weekday jan1 + 7) mod 7) in
  thursday - 3

let week z =
  let y, _, _ = Time.date (Float.of_int (z * 86400)) in
  let y = List.find (fun y -> first_monday y <= z) [ y + 1; y; y - 1 ] in
  (y, ((z - first_monday y) / 7) + 1)

let calendar_days =
  test "day_of_year and iso_week are ISO 8601's" (fun () ->
      (* ISO 8601:2004's examples: 1985-102 and 1985-W15-5 are 1985-04-12. *)
      let t = Time.of_date (1985, 4, 12) +. 43200. in
      equal int 102 (Time.day_of_year t);
      equal (pair int int) (1985, 15) (Time.iso_week t);
      List.iter
        (fun (day, w) ->
          equal (pair int int) w (Time.iso_week (Time.of_date day)))
        [
          (* ISO 8601:2004, 3.2.2, NOTE 2. *)
          ((1995, 1, 1), (1994, 52));
          ((1996, 12, 31), (1997, 1));
          ((2008, 12, 29), (2009, 1));
          ((2010, 1, 3), (2009, 53));
          ((2010, 1, 4), (2010, 1));
          ((2020, 12, 31), (2020, 53));
          ((2021, 1, 3), (2020, 53));
          ((1970, 1, 1), (1970, 1));
        ];
      equal int 366 (Time.day_of_year (Time.of_date (2024, 12, 31)));
      equal int 365 (Time.day_of_year (Time.of_date (2023, 12, 31)));
      equal int 1 (Time.day_of_year (Time.of_date (1970, 1, 1)));
      (* The instant before 1970 lies on 1969-12-31. *)
      equal int 365 (Time.day_of_year (-0.5));
      raises_match (Exn.invalid_arg ~sub:"Jord.Time.iso_week") (fun () ->
          Time.iso_week Float.nan))

let calendar_law =
  test "each day from 1890 to 2110 is its year's day and week" (fun () ->
      let first = int_of_float (Time.of_date (1890, 1, 1) /. 86400.) in
      let last = int_of_float (Time.of_date (2110, 12, 31) /. 86400.) in
      for z = first to last do
        let t = Float.of_int (z * 86400) +. 86399.5 in
        let y, _, _ = Time.date t in
        let jan1 = int_of_float (Time.of_date (y, 1, 1) /. 86400.) in
        if Time.day_of_year t <> z - jan1 + 1 then failf "day %d" z;
        if Time.iso_week t <> week z then failf "week of day %d" z
      done)

let refusals =
  cases ~id:Fun.id "texts that are not date-times"
    [
      "2024-06-29";
      "2024-06-29T10:20:21";
      "2024-06-29T10:20:21.Z";
      "2024-06-29X10:20:21Z";
      "2024-06-29T10:20:21Zx";
      "2024-13-01T00:00:00Z";
      "2023-02-29T00:00:00Z";
      "2024-06-29T24:00:00Z";
      "2024-06-29T10:60:00Z";
      "2024-06-29T10:20:61Z";
      "2024-06-29T10:20:60Z";
      "1998-12-31T23:58:60Z";
      "1998-12-31T23:59:60+01:00";
      "2024-06-00T00:00:00Z";
      "2024-00-29T00:00:00Z";
      "2024-06-29T10:20:21+0100";
      "2024-06-29T10:20:21+24:00";
      "2024-06-29T10:20:21+01:60";
      "+024-06-29T10:20:21Z";
    ]
    (fun s ->
      match Time.of_rfc3339 s with
      | Ok _ -> failf "%S read" s
      | Error e -> contains ~sub:"RFC 3339" e)

let () =
  exit
    (run "jord.time"
       [
         times;
         leap_seconds;
         calendar;
         dates;
         of_date;
         calendar_days;
         calendar_law;
         refusals;
       ])
