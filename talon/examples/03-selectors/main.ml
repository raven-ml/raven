open Talon

let () =
  let df =
    create
      [
        ("id", Col.string_list [ "a"; "b"; "c"; "d" ]);
        ("age", Col.int32_list [ 20l; 30l; 40l; 50l ]);
        ("height_cm", Col.float64_list [ 170.; 180.; 165.; 175. ]);
        ("is_member", Col.bool_list [ true; false; true; false ]);
        ("note", Col.string_list [ "x"; "y"; "z"; "" ]);
      ]
  in

  let numeric = Cols.numeric df in
  let floats = Cols.float df in
  let by_prefix = Cols.with_prefix df "he" in
  let by_suffix = Cols.with_suffix df "_cm" in
  let re = Re.compile (Re.Perl.re "^(age|height_.*)$") in
  let by_regex = Cols.matching df re in
  let numeric_except_id = Cols.except df ("id" :: numeric) in

  Printf.printf "numeric: [%s]\n" (String.concat ", " numeric);
  Printf.printf "float:   [%s]\n" (String.concat ", " floats);
  Printf.printf "prefix 'he': [%s]\n" (String.concat ", " by_prefix);
  Printf.printf "suffix '_cm': [%s]\n" (String.concat ", " by_suffix);
  Printf.printf "regex ^(age|height_.*)$: [%s]\n" (String.concat ", " by_regex);
  Printf.printf "numeric except id: [%s]\n"
    (String.concat ", " numeric_except_id)
