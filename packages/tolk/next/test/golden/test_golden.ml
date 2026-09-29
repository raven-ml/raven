open Windtrap

let golden contents =
  let file = temp_file ~suffix:".golden" () in
  Out_channel.with_open_bin file (fun oc -> output_string oc contents);
  file

let header = "# tinygrad 79af1ca70e7021f504919c4ff5631245acc33ed6\n"
let not_a_golden f = raises_match (Exn.failure ~substring:"not a golden") f

let texts =
  group "text"
    [
      test "is the file without its header line and final newline" (fun () ->
          equal text "a\nb\n" (Golden.text (golden (header ^ "a\nb\n\n"))));
      test "of an empty body is empty" (fun () ->
          equal text "" (Golden.text (golden (header ^ "\n"))));
      test "rejects a file without the header" (fun () ->
          not_a_golden (fun () -> Golden.text (golden "a\n")));
      test "rejects a file without a body" (fun () ->
          not_a_golden (fun () -> Golden.text (golden header)));
      test "rejects a file without a final newline" (fun () ->
          not_a_golden (fun () -> Golden.text (golden (header ^ "a"))));
    ]

let lattice () = golden (header ^ "a\tb\tlub\nint\tint\tint\nint\thalf\thalf\n")

let tables =
  group "table"
    [
      test "has one row per line after the columns" (fun () ->
          let rows = Golden.table (lattice ()) in
          equal (list string) [ "int"; "half" ]
            (List.map (fun row -> Golden.cell row "lub") rows));
      test "keeps empty cells" (fun () ->
          let rows = Golden.table (golden (header ^ "a\tb\n\t\n")) in
          equal (list string) [ "" ]
            (List.map (fun row -> Golden.cell row "b") rows));
      test "rejects a row with too few cells" (fun () ->
          raises_match (Exn.failure ~substring:"line 3: 1 cells for 2 columns")
            (fun () -> Golden.table (golden (header ^ "a\tb\nx\n"))));
      test "rejects a table without rows" (fun () ->
          raises_match (Exn.failure ~substring:"no row") (fun () ->
              Golden.table (golden (header ^ "a\tb\n"))));
      test "cell rejects an unknown column" (fun () ->
          let row = List.hd (Golden.table (lattice ())) in
          raises_match (Exn.invalid_arg ~substring:"no column \"c\"") (fun () ->
              Golden.cell row "c"));
      test "key names a row by the given columns, in order" (fun () ->
          let row = List.nth (Golden.table (lattice ())) 1 in
          equal string "b=half a=int" (Golden.key [ "b"; "a" ] row));
    ]

let () = exit (run "golden" [ texts; tables ])
