(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Tz = Talon.Tz
module Error = Talon.Error

(* Fixtures *)

(* The database of golden/tz/zoneinfo/, written by gen/tz.py. *)
let db () = Error.get_ok (Tz.of_dir "golden/tz/zoneinfo")
let zone name = Error.get_ok (Tz.find (db ()) name)
let hour = 3600
let day = 86_400

(* [posix y m d h min] is the POSIX second of that UTC time, by Howard Hinnant's
   days-from-civil. *)
let posix y m d h min =
  let y = if m <= 2 then y - 1 else y in
  let era = (if y >= 0 then y else y - 399) / 400 in
  let yoe = y - (era * 400) in
  let doy = (((153 * ((m + 9) mod 12)) + 2) / 5) + d - 1 in
  let days =
    (era * 146_097) + (yoe * 365) + (yoe / 4) - (yoe / 100) + doy - 719_468
  in
  Int64.of_int ((days * day) + (h * hour) + (min * 60))

(* zdump's transitions of the zones copied from the system, from 1800 to 2200:
   [(zone, instant, offset before, offset after)]. *)
let zdump =
  In_channel.with_open_text "golden/tz/transitions.txt" In_channel.input_lines
  |> List.map (fun line ->
      Scanf.sscanf line "%s %Ld %d %d" (fun zone u before after ->
          (zone, u, before, after)))

(* The transitions of each fixture zone, from zdump for those copied from the
   system or built from their sources, and from the spec for hand-made ones. *)
let transitions = function
  | "slim/Europe/Paris" | "right/Europe/Paris" | "v5/Europe/Paris" ->
      List.filter_map
        (fun (z, u, b, a) ->
          if z = "Europe/Paris" then Some (u, b, a) else None)
        zdump
  | "hand/Version1" -> [ (1000L, 3600, 7200) ]
  | "hand/Truncated" -> [ (1_500_000_000L, 0, 3600) ]
  | "hand/AllYearDst" -> []
  | "hand/Rules" ->
      let at y m d h min s = Int64.add (posix y m d h min) (Int64.of_int s) in
      [
        (at 2023 12 31 0 30 15, 3600, 7200);
        (at 2024 2 29 19 45 0, 7200, 3600);
        (at 2024 12 29 0 30 15, 3600, 7200);
        (at 2025 2 28 19 45 0, 7200, 3600);
      ]
  | "hand/Days" ->
      [
        (posix 2024 2 23 20 0, 7200, 3600);
        (posix 2024 2 29 2 0, 3600, 7200);
        (posix 2025 2 21 20 0, 7200, 3600);
        (posix 2025 3 1 2 0, 3600, 7200);
      ]
  | "hand/Julian" ->
      [
        (posix 2024 2 27 23 0, 3600, 7200);
        (posix 2024 2 29 22 0, 7200, 3600);
        (posix 2025 2 27 23 0, 3600, 7200);
        (posix 2025 2 28 22 0, 7200, 3600);
      ]
  | "hand/LeapAtEpoch" -> [ (-1L, 0, 3600) ]
  | "hand/Skips" -> [ (1000L, 0, 3600); (1500L, 3600, 0); (2000L, 0, 3600) ]
  | "hand/Folds" -> [ (1000L, 7200, 3600); (1500L, 3600, 0) ]
  | "hand/FoldGap" -> [ (1000L, 3600, 0); (1100L, 0, 7200) ]
  | "hand/GapFoldGap" -> [ (0L, 0, 3600); (10L, 3600, -3600); (20L, -3600, 0) ]
  | "hand/StdDisagrees" -> [ (1000L, 3600, 7200) ]
  | "hand/Disagrees" ->
      [ (15_638_400L, 0, -21_600); (posix 1971 3 14 8 0, -21_600, -18_000) ]
  | "hand/PermDisagrees" -> [ (1000L, 0, -10_800) ]
  | "hand/LeapPair" -> [ (78_796_799L, 0, 7200) ]
  | "hand/Fixed" | "Etc/GMT+5" -> []
  | name ->
      List.filter_map
        (fun (z, u, b, a) -> if z = name then Some (u, b, a) else None)
        zdump

let zones =
  [
    "Europe/Paris";
    "America/New_York";
    "Australia/Lord_Howe";
    "America/Nuuk";
    "Asia/Jerusalem";
    "UTC";
    "Etc/GMT+5";
    "slim/Europe/Paris";
    "right/Europe/Paris";
    "v5/Europe/Paris";
    "hand/Version1";
    "hand/AllYearDst";
    "hand/Truncated";
    "hand/Rules";
    "hand/Days";
    "hand/Julian";
    "hand/LeapAtEpoch";
    "hand/Skips";
    "hand/Folds";
    "hand/FoldGap";
    "hand/GapFoldGap";
    "hand/StdDisagrees";
    "hand/Disagrees";
    "hand/PermDisagrees";
    "hand/LeapPair";
    "hand/Fixed";
  ]

(* Every offset a zone takes, from its transitions, or its only offset. *)
let offsets name =
  match transitions name with
  | [] when name = "hand/AllYearDst" -> [ -14_400 ]
  | [] when name = "hand/Fixed" -> [ 19_800 ]
  | [] when name = "Etc/GMT+5" -> [ -18_000 ]
  | [] -> [ 0 ]
  | rows ->
      List.sort_uniq Int.compare
        (List.concat_map (fun (_, b, a) -> [ b; a ]) rows)

(* Instants *)

(* [t - o], clamped to [int64]'s range, as the contract of [local] reads offsets
   beyond it. *)
let sub_sat t o =
  let v = Int64.sub t (Int64.of_int o) in
  if o > 0 && Int64.compare v t > 0 then Int64.min_int
  else if o < 0 && Int64.compare v t < 0 then Int64.max_int
  else v

let extremes =
  Int64.
    [
      min_int;
      add min_int 1L;
      add min_int 86_400L;
      sub max_int 86_400L;
      sub max_int 1L;
      max_int;
    ]

let pp_instant ppf t = Format.fprintf ppf "%Ld" t

(* Instants near the transitions of [name], within its years, anywhere, and at
   the ends of [int64]. Near a transition, half are wall-clock times within
   three hours of the clock's reading on either side of it, where its gaps and
   folds lie. *)
let instant name =
  let near =
    match transitions name with
    | [] -> []
    | rows ->
        let around u spread =
          Gen.map
            (fun d -> Int64.add u (Int64.of_int d))
            (Gen.int_range (-spread) spread)
        in
        [
          (3, Gen.bind (Gen.of_list rows) (fun (u, _, _) -> around u (2 * day)));
          ( 3,
            Gen.bind (Gen.of_list rows) (fun (u, before, after) ->
                Gen.bind
                  (Gen.of_list [ before; after ])
                  (fun o -> around (Int64.add u (Int64.of_int o)) (3 * hour)))
          );
        ]
  in
  Gen.with_pp pp_instant
    (Gen.frequency
       (near
       @ [
           (2, Gen.int64_range (-5_500_000_000L) 7_300_000_000L);
           (1, Gen.int64);
           (1, Gen.of_list extremes);
         ]))

let zone_instant =
  Gen.with_pp
    (fun ppf (name, t) -> Format.fprintf ppf "%s at %Ld" name t)
    (Gen.bind (Gen.of_list zones) (fun name ->
         Gen.map (fun t -> (name, t)) (instant name)))

let local =
  Testable.make
    ~pp:(fun ppf -> function
      | Tz.Unique o -> Format.fprintf ppf "Unique %d" o
      | Ambiguous { before; after } ->
          Format.fprintf ppf "Ambiguous { before = %d; after = %d }" before
            after
      | Gap { before; after } ->
          Format.fprintf ppf "Gap { before = %d; after = %d }" before after)
    ~equal:( = )

(* UTC *)

let utc =
  group "utc"
    [
      test "is named UTC" (fun () -> equal string "UTC" (Tz.name Tz.utc));
      prop "has the offset 0 at every instant" (instant "UTC") (fun t ->
          equal int 0 (Tz.offset_s Tz.utc t));
      prop "reads every local time once, at offset 0" (instant "UTC") (fun t ->
          equal local (Unique 0) (Tz.local Tz.utc t));
      test "agrees with the UTC file at the ends of int64" (fun () ->
          List.iter
            (fun t ->
              equal int (Tz.offset_s Tz.utc t) (Tz.offset_s (zone "UTC") t))
            extremes);
    ]

(* Offsets *)

let sweep name =
  let z = zone name in
  List.iter
    (fun (u, before, after) ->
      equal int before
        ~msg:(Printf.sprintf "%Ld - 1" u)
        (Tz.offset_s z (Int64.pred u));
      equal int after ~msg:(Printf.sprintf "%Ld" u) (Tz.offset_s z u))
    (transitions name)

let agree a b =
  prop (Printf.sprintf "%s agrees with %s" a b) (instant b) (fun t ->
      equal int (Tz.offset_s (zone b) t) (Tz.offset_s (zone a) t))

let cycle = Int64.of_int (146_097 * day)

let offset_s =
  group "offset_s"
    [
      cases ~name:Fun.id "changes at each transition"
        (List.filter (fun name -> transitions name <> []) zones)
        sweep;
      test "is the first time type's before the first transition" (fun () ->
          let first name =
            match transitions name with
            | (_, before, _) :: _ -> before
            | [] -> 0
          in
          List.iter
            (fun name ->
              equal int (first name) ~msg:name
                (Tz.offset_s (zone name) Int64.min_int))
            (List.filter
               (fun name ->
                 not
                   (List.mem name
                      [
                        "UTC";
                        "Etc/GMT+5";
                        "hand/AllYearDst";
                        "hand/Rules";
                        "hand/Days";
                        "hand/Julian";
                        "hand/Fixed";
                      ]))
               zones));
      test "follows the footer before and after a file without transitions"
        (fun () ->
          let z = zone "hand/Rules" in
          equal int 7200 (Tz.offset_s z (posix 1000 1 15 0 0));
          equal int 3600 (Tz.offset_s z (posix 1000 7 15 0 0)));
      cases ~name:Fun.id "persists the last offset without a rule"
        [ "hand/Version1"; "hand/StdDisagrees" ] (fun name ->
          let z = zone name in
          equal int 7200 (Tz.offset_s z Int64.max_int);
          equal int 7200 (Tz.offset_s z (posix 2400 7 1 0 0)));
      prop "has a file's only offset without transitions or a rule"
        (instant "hand/Fixed") (fun t ->
          let z = zone "hand/Fixed" in
          equal int 19_800 (Tz.offset_s z t);
          equal local (Unique 19_800) (Tz.local z t));
      prop "reads a POSIX offset as seconds west of UTC" (instant "Etc/GMT+5")
        (fun t -> equal int (-18_000) (Tz.offset_s (zone "Etc/GMT+5") t));
      test "follows a disagreeing footer from its first change of offset"
        (fun () ->
          let z = zone "hand/Disagrees" in
          equal int (-21_600) (Tz.offset_s z (posix 1970 11 1 6 59));
          equal int (-21_600) (Tz.offset_s z (posix 1970 11 1 7 0));
          equal int (-18_000) (Tz.offset_s z (posix 1972 7 1 0 0));
          equal int (-21_600) (Tz.offset_s z (posix 1972 12 1 0 0)));
      test "keeps the last offset under a disagreeing rule that never changes"
        (fun () ->
          let z = zone "hand/PermDisagrees" in
          equal int (-10_800) (Tz.offset_s z (posix 1974 1 1 3 0));
          equal int (-10_800) (Tz.offset_s z Int64.max_int));
      prop "follows a footer with no transitions everywhere"
        (instant "hand/AllYearDst") (fun t ->
          equal int (-14_400) (Tz.offset_s (zone "hand/AllYearDst") t));
      test "removes leap seconds from transition times" (fun () ->
          let z = zone "hand/Truncated" in
          equal int 0 (Tz.offset_s z 1_499_999_999L);
          equal int 3600 (Tz.offset_s z 1_500_000_000L));
      agree "slim/Europe/Paris" "Europe/Paris";
      agree "right/Europe/Paris" "Europe/Paris";
      agree "v5/Europe/Paris" "Europe/Paris";
      prop "repeats every 400 years after the last transition"
        (Gen.with_pp
           (fun ppf (name, t) -> Format.fprintf ppf "%s at %Ld" name t)
           (Gen.pair (Gen.of_list zones)
              (Gen.frequency
                 [
                   (3, Gen.int64_range 4_200_000_000L 20_000_000_000L);
                   ( 1,
                     Gen.int64_range 4_200_000_000L
                       (Int64.sub Int64.max_int cycle) );
                 ])))
        (fun (name, t) ->
          let z = zone name in
          equal int (Tz.offset_s z t) (Tz.offset_s z (Int64.add t cycle)));
    ]

(* Local times *)

(* The offsets [o] of [name] at which its clock reads [t]. *)
let readings name t =
  let z = zone name in
  List.filter (fun o -> Tz.offset_s z (sub_sat t o) = o) (offsets name)

let complete (name, t) =
  let z = zone name in
  match (Tz.local z t, readings name t) with
  | Unique o, [ o' ] -> equal int o' o
  | Ambiguous { before; after }, (_ :: _ :: _ as r) ->
      equal int (List.fold_left max min_int r) before;
      equal int (List.fold_left min max_int r) after
  | Gap { before; after }, [] ->
      less int ~than:after before;
      less int ~than:after (Tz.offset_s z (sub_sat t after));
      greater int ~than:before (Tz.offset_s z (sub_sat t before))
  | l, r ->
      failf "local is %a, but the clock reads t at the offsets [%s]"
        (Testable.pp local) l
        (String.concat "; " (List.map string_of_int r))

(* Local times every 50 seconds around the transitions of the zones whose
   transitions are closer than the offsets they change. *)
let close_calls =
  List.concat_map
    (fun name -> List.init 241 (fun i -> (name, Int64.of_int ((i - 80) * 50))))
    [ "hand/Skips"; "hand/Folds"; "hand/FoldGap"; "hand/GapFoldGap" ]

(* [complete], labelling the cases it depends on. *)
let completeness (name, t) =
  let l = Tz.local (zone name) t in
  cover "a gap" (match l with Gap _ -> true | _ -> false);
  cover "a fold" (match l with Ambiguous _ -> true | _ -> false);
  cover "an end of int64" (List.mem t extremes);
  complete (name, t)

(* The wall-clock time at [u] reads back to [u]'s offset, which lies between the
   earliest and the latest readings when the clock reads it again. *)
let round_trip (name, u) =
  let z = zone name in
  let o = Tz.offset_s z u in
  let t = sub_sat u (-o) in
  assume (Int64.equal (sub_sat t o) u);
  match Tz.local z t with
  | Unique o' -> equal int o o'
  | Ambiguous { before; after } ->
      at_least int ~than:after o;
      at_most int ~than:before o
  | Gap _ as l ->
      failf "the clock reads %Ld at %Ld, yet local is %a" t u
        (Testable.pp local) l

let wall name y m d h min expected =
  let label = Printf.sprintf "%s %04d-%02d-%02d %02d:%02d" name y m d h min in
  ( label,
    fun () -> equal local expected (Tz.local (zone name) (posix y m d h min)) )

let walls =
  [
    wall "Europe/Paris" 2024 7 1 12 0 (Unique 7200);
    wall "Europe/Paris" 2024 3 31 2 30 (Gap { before = 3600; after = 7200 });
    wall "Europe/Paris" 2024 10 27 2 30
      (Ambiguous { before = 7200; after = 3600 });
    wall "slim/Europe/Paris" 2063 3 25 2 0 (Gap { before = 3600; after = 7200 });
    wall "slim/Europe/Paris" 2063 10 28 2 59
      (Ambiguous { before = 7200; after = 3600 });
    wall "slim/Europe/Paris" 2063 10 28 3 0 (Unique 3600);
    wall "America/New_York" 2024 3 10 2 30
      (Gap { before = -18_000; after = -14_400 });
    wall "America/New_York" 2024 11 3 1 30
      (Ambiguous { before = -14_400; after = -18_000 });
    wall "Australia/Lord_Howe" 2024 4 7 1 45
      (Ambiguous { before = 39_600; after = 37_800 });
    wall "Australia/Lord_Howe" 2024 10 6 2 15
      (Gap { before = 37_800; after = 39_600 });
    wall "Australia/Lord_Howe" 2100 1 1 0 0 (Unique 39_600);
    wall "America/Nuuk" 2090 3 25 23 30 (Gap { before = -7200; after = -3600 });
    wall "America/Nuuk" 2090 10 28 23 30
      (Ambiguous { before = -3600; after = -7200 });
    wall "Asia/Jerusalem" 2090 3 24 2 30 (Gap { before = 7200; after = 10_800 });
    wall "hand/Skips" 1970 1 1 0 50 (Gap { before = 0; after = 3600 });
    wall "hand/Folds" 1970 1 1 1 20 (Ambiguous { before = 7200; after = 0 });
    wall "hand/FoldGap" 1970 1 1 2 0 (Gap { before = 0; after = 7200 });
    wall "hand/GapFoldGap" 1970 1 1 0 0 (Gap { before = -3600; after = 3600 });
    wall "hand/Disagrees" 1970 8 1 12 0 (Unique (-21_600));
    wall "hand/LeapPair" 1972 7 1 1 0 (Gap { before = 0; after = 7200 });
    wall "hand/Version1" 1970 1 1 1 16 (Unique 3600);
    wall "hand/Version1" 1970 1 1 1 17 (Gap { before = 3600; after = 7200 });
    wall "hand/Version1" 1970 1 1 2 16 (Gap { before = 3600; after = 7200 });
    wall "hand/Version1" 1970 1 1 2 17 (Unique 7200);
  ]

let local_times =
  group "local"
    [
      cases ~name:fst "reads the wall clock" walls (fun (_, f) -> f ());
      prop ~count:400 "names exactly the offsets at which the clock reads t"
        ~examples:close_calls zone_instant completeness;
      prop ~count:400 "reads every instant's wall-clock time back" zone_instant
        round_trip;
      test "answers at the ends of int64" (fun () ->
          List.iter
            (fun name -> List.iter (fun t -> complete (name, t)) extremes)
            zones);
    ]

(* Databases *)

let to_string e = Format.asprintf "%a" Error.pp e

(* Error paths join with the system's separator; the baselines use '/'. *)
let portable s =
  if Sys.win32 then String.map (function '\\' -> '/' | c -> c) s else s

let pp_result = function Ok _ -> "Ok" | Error e -> portable (to_string e)
let pp_find db name = pp_result (Result.map Tz.name (Tz.find db name))

let malformed () =
  Sys.readdir "golden/tz/malformed"
  |> Array.to_list |> List.sort String.compare
  |> List.iter (fun case ->
      let db =
        Error.get_ok (Tz.of_dir (Filename.concat "golden/tz/malformed" case))
      in
      print_endline (pp_find db "Zone"));
  expect (output ())
  @@ __POS_OF__
       {|
    golden/tz/malformed/after-footer/Zone: byte 134: bytes follow the footer
    golden/tz/malformed/designation/Zone: byte 118: time type 1 has the designation index 8, beyond the 8 designation bytes
    golden/tz/malformed/designation-nul/Zone: byte 118: time type 1 has a designation with no NUL
    golden/tz/malformed/footer-digits/Zone: byte 134: footer: the hour 123 is not in [0, 24]
    golden/tz/malformed/footer-end/Zone: bytes 127-132: the footer does not end with a newline
    golden/tz/malformed/footer-hour-v2/Zone: byte 146: footer: the hour 26 is not in [0, 24]
    golden/tz/malformed/footer-hour-v3/Zone: byte 147: footer: the hour 168 is not in [0, 167]
    golden/tz/malformed/footer-no-rule/Zone: byte 135: footer: daylight saving time "CCC" has no rule
    golden/tz/malformed/footer-nul/Zone: byte 131: the footer holds a NUL byte
    golden/tz/malformed/footer-short/Zone: byte 130: footer: the designation "BB" has fewer than 3 characters
    golden/tz/malformed/footer-start/Zone: byte 127: the footer does not start with a newline
    golden/tz/malformed/footer-syntax/Zone: byte 133: footer: the designation "x" has fewer than 3 characters
    golden/tz/malformed/indicator/Zone: byte 128: time type 1 has the standard/wall indicator 2, not 0 or 1
    golden/tz/malformed/isdst/Zone: byte 117: time type 1 has the DST indicator 2, not 0 or 1
    golden/tz/malformed/isutcnt/Zone: bytes 74-77: the count of UT/local indicators is neither 0 nor the count of time types
    golden/tz/malformed/leap-equal-v4/Zone: bytes 139-150: leap second 1 has the correction 1, which does not differ from the previous one, 1, by 1
    golden/tz/malformed/leap-expiry-v2/Zone: bytes 139-150: leap second 1 has the correction 1, which does not differ from the previous one, 1, by 1
    golden/tz/malformed/leap-first/Zone: bytes 127-138: the first leap-second correction is 2, not 1 or -1
    golden/tz/malformed/leap-month/Zone: bytes 127-138: leap second 0 does not end a month
    golden/tz/malformed/leap-negative/Zone: bytes 127-138: the first leap second occurs before 1970
    golden/tz/malformed/leap-order/Zone: bytes 139-150: leap second 1 does not occur after the previous one
    golden/tz/malformed/leap-step/Zone: bytes 139-150: leap second 1 has the correction 3, which does not differ from the previous one, 1, by 1
    golden/tz/malformed/leap-unknown/Zone: bytes 98-105: transition 0 precedes the leap-second table, whose correction is unknown there
    golden/tz/malformed/no-types/Zone: bytes 90-93: the count of time types is 0
    golden/tz/malformed/not-ascending/Zone: bytes 106-113: transition 1 is not after the previous one
    golden/tz/malformed/second-magic/Zone: bytes 54-57: the header does not start with "TZif"
    golden/tz/malformed/truncated/Zone: bytes 98-126: the file ends after 114 bytes, inside the data block
    golden/tz/malformed/type-index/Zone: byte 106: transition 0 has time type 2, beyond the 2 time types
    golden/tz/malformed/ut-no-std/Zone: byte 128: time type 1 is in UT but not in standard time
    golden/tz/malformed/ut-not-std/Zone: byte 130: time type 1 is in UT but not in standard time
    golden/tz/malformed/utoff/Zone: bytes 113-116: time type 1 has the UT offset -2^31
    golden/tz/malformed/v1-after-data/Zone: byte 69: bytes follow the data block of a version 1 file
    golden/tz/malformed/version/Zone: byte 4: unknown version '1'
    golden/tz/malformed/versions-differ/Zone: byte 58: the second header's version '2' differs from the first's, '3'
    |}

(* The error of a rejected name holds no path, so it is printed as is. *)
let names () =
  List.iter
    (fun name ->
      print_endline
        (match Tz.find (db ()) name with
        | Ok _ -> "Ok"
        | Error e -> to_string e))
    [
      "";
      "/UTC";
      "UTC/";
      "Europe//Paris";
      ".";
      "..";
      "./UTC";
      "../zoneinfo/UTC";
      "Europe/../UTC";
      "Europe\\Paris";
      "UTC\000";
      "Europe/Par is";
      "CON";
      "prn";
      "Aux.zone";
      "Europe/nul.txt";
      "com1";
      "LPT9.x.y";
      "COM0";
    ];
  expect (output ())
  @@ __POS_OF__
       {|
    "" is not a zone name
    "/UTC" is not a zone name
    "UTC/" is not a zone name
    "Europe//Paris" is not a zone name
    "." is not a zone name
    ".." is not a zone name
    "./UTC" is not a zone name
    "../zoneinfo/UTC" is not a zone name
    "Europe/../UTC" is not a zone name
    "Europe\\Paris" is not a zone name
    "UTC\000" is not a zone name
    "Europe/Par is" is not a zone name
    "CON" is not a zone name
    "prn" is not a zone name
    "Aux.zone" is not a zone name
    "Europe/nul.txt" is not a zone name
    "com1" is not a zone name
    "LPT9.x.y" is not a zone name
    "COM0" is not a zone name
    |}

(* Names that a Windows device name only begins are names. *)
let not_zones () =
  List.iter
    (fun name -> print_endline (pp_find (db ()) name))
    [
      "Europe";
      "zone.tab";
      "tiny";
      "Europe/Nowhere";
      "CONSOLE";
      "COM";
      "COM10";
      "LPTX";
      "NULL.zone";
      ".AUX";
    ];
  expect (output ())
  @@ __POS_OF__
       {|
    golden/tz/zoneinfo/Europe: a directory, not a zone
    golden/tz/zoneinfo/zone.tab: bytes 0-3: the header does not start with "TZif"
    golden/tz/zoneinfo/tiny: bytes 0-43: the file ends after 2 bytes, inside a header
    golden/tz/zoneinfo/Europe/Nowhere: No such file or directory
    golden/tz/zoneinfo/CONSOLE: No such file or directory
    golden/tz/zoneinfo/COM: No such file or directory
    golden/tz/zoneinfo/COM10: No such file or directory
    golden/tz/zoneinfo/LPTX: No such file or directory
    golden/tz/zoneinfo/NULL.zone: No such file or directory
    golden/tz/zoneinfo/.AUX: No such file or directory
    |}

let write path content =
  Out_channel.with_open_bin path (fun oc -> output_string oc content)

let paris () =
  In_channel.with_open_bin "golden/tz/zoneinfo/Europe/Paris"
    In_channel.input_all

(* A tree with links of every kind, and a file that is not TZif. *)
let tree () =
  if Sys.win32 then
    skip
      ~reason:
        "links to directories need Unix.symlink ~to_dir:true, and the \
         baselines mask paths joined with '/'"
      ();
  let dir = temp_dir () in
  let at = Filename.concat dir in
  Unix.mkdir (at "Real") 0o755;
  write (at "Real/Paris") (paris ());
  write (at "Real/notes") "not a zone";
  Unix.symlink "Real/Paris" (at "Link");
  Unix.symlink "." (at "Cycle");
  Unix.symlink (at "Real") (at "Dir");
  Unix.symlink "nowhere" (at "Dangling");
  Unix.symlink "B" (at "A");
  Unix.symlink "A" (at "B");
  dir

let links () =
  let dir = tree () in
  let db = Error.get_ok (Tz.of_dir dir) in
  let masked s =
    let prefix = dir ^ "/" in
    if String.starts_with ~prefix s then
      let n = String.length prefix in
      "$DIR/" ^ String.sub s n (String.length s - n)
    else s
  in
  List.iter
    (fun name -> print_endline (masked (pp_find db name)))
    [
      "Real/Paris";
      "Link";
      "Dir/Paris";
      "Cycle/Cycle/Real/Paris";
      "Dangling";
      "A";
      "Real/notes";
    ];
  expect (output ())
  @@ __POS_OF__
       {|
    Ok
    Ok
    Ok
    Ok
    $DIR/Dangling: No such file or directory
    $DIR/A: Too many levels of symbolic links
    $DIR/Real/notes: bytes 0-43: the file ends after 10 bytes, inside a header
    |}

let link_name () =
  let db = Error.get_ok (Tz.of_dir (tree ())) in
  equal string "Link" (Tz.name (Error.get_ok (Tz.find db "Link")))

let dir_link () =
  let dir = tree () in
  let db = Error.get_ok (Tz.of_dir (Filename.concat dir "Dir")) in
  equal string "Paris" (Tz.name (Error.get_ok (Tz.find db "Paris")))

let needs_permissions () =
  if Sys.win32 then
    skip ~reason:"permission bits do not stop reading on Windows" ();
  if Unix.geteuid () = 0 then skip ~reason:"root reads every file" ()

let unreadable () =
  needs_permissions ();
  let dir = temp_dir () in
  let path = Filename.concat dir "Secret" in
  write path (paris ());
  Unix.chmod path 0o000;
  let db = Error.get_ok (Tz.of_dir dir) in
  equal string (path ^ ": Permission denied") (pp_find db "Secret")

let unreadable_dir () =
  needs_permissions ();
  let dir = temp_dir () in
  let closed = Filename.concat dir "Closed" in
  Unix.mkdir closed 0o755;
  write (Filename.concat closed "Zone") (paris ());
  Unix.chmod closed 0o000;
  Fun.protect
    ~finally:(fun () -> Unix.chmod closed 0o755)
    (fun () ->
      let db = Error.get_ok (Tz.of_dir dir) in
      equal string
        (Filename.concat closed "Zone" ^ ": Permission denied")
        (pp_find db "Closed/Zone"))

let find =
  group "find"
    [
      test "names a zone as it is asked for" (fun () ->
          equal string "America/New_York" (Tz.name (zone "America/New_York"));
          equal string "slim/Europe/Paris" (Tz.name (zone "slim/Europe/Paris")));
      test "rejects names that leave the directory or that tz never uses" names;
      test "reports a name that designates no zone" not_zones;
      test "reports a path through a file" (fun () ->
          ignore (require_error (Tz.find (db ()) "UTC/Paris")));
      test "reports each malformed file" malformed;
      test "follows symbolic links, and reports dead ones" links;
      test "names a zone read through a link by the link" link_name;
      test "reports a file it cannot read" unreadable;
      test "reports a file in a directory it cannot read" unreadable_dir;
    ]

let of_dir =
  group "of_dir"
    [
      test "rejects a path that is not a directory" (fun () ->
          expect (pp_result (Tz.of_dir "golden/tz/transitions.txt"))
          @@ __POS_OF__ {| golden/tz/transitions.txt: not a directory |});
      test "rejects a missing directory" (fun () ->
          expect (pp_result (Tz.of_dir "missing"))
          @@ __POS_OF__ {| missing: No such file or directory |});
      test "follows the directory it is given when it is a link" dir_link;
    ]

let system =
  group "system"
    [
      test "reads TZDIR" (fun () ->
          setenv "TZDIR" (Some "golden/tz/zoneinfo");
          let z = Error.get_ok (Tz.find (Error.get_ok (Tz.system ())) "UTC") in
          equal int 0 (Tz.offset_s z 0L));
      test "fails when TZDIR names no directory" (fun () ->
          setenv "TZDIR" (Some "missing");
          expect (pp_result (Tz.system ()))
          @@ __POS_OF__ {| missing: No such file or directory |});
      test "reads /usr/share/zoneinfo when TZDIR is empty" (fun () ->
          if not (Sys.file_exists "/usr/share/zoneinfo") then
            skip ~reason:"no /usr/share/zoneinfo" ();
          setenv "TZDIR" (Some "");
          let db = Error.get_ok (Tz.system ()) in
          let z = Error.get_ok (Tz.find db "Europe/Paris") in
          equal int 7200 (Tz.offset_s z (posix 2024 7 1 0 0)));
    ]

let () = exit (run "tz" [ utc; offset_s; local_times; find; of_dir; system ])
