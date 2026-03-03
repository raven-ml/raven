(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon

let () =
  let df =
    create
      [
        ("id", Col.string [| "a"; "b"; "c"; "d" |]);
        ("age", Col.int32 [| 20l; 30l; 40l; 50l |]);
        ("height_cm", Col.float64 [| 170.; 180.; 165.; 175. |]);
        ("is_member", Col.bool [| true; false; true; false |]);
        ("note", Col.string [| "x"; "y"; "z"; "" |]);
      ]
  in

  let numeric = select_columns df `Numeric in
  let floats = select_columns df `Float in
  let names = column_names df in
  let by_prefix =
    List.filter (fun n -> String.starts_with ~prefix:"he" n) names
  in
  let by_suffix =
    List.filter (fun n -> String.ends_with ~suffix:"_cm" n) names
  in
  let numeric_except_id =
    List.filter (fun n -> n <> "id") (select_columns df `Numeric)
  in

  Printf.printf "numeric: [%s]\n" (String.concat ", " numeric);
  Printf.printf "float:   [%s]\n" (String.concat ", " floats);
  Printf.printf "prefix 'he': [%s]\n" (String.concat ", " by_prefix);
  Printf.printf "suffix '_cm': [%s]\n" (String.concat ", " by_suffix);
  Printf.printf "numeric except id: [%s]\n"
    (String.concat ", " numeric_except_id)
