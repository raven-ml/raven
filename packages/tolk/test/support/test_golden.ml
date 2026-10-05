open Windtrap

let golden contents =
  let file = temp_file ~suffix:".golden" () in
  Out_channel.with_open_bin file (fun oc -> output_string oc contents);
  file

let header = "# tinygrad 79af1ca70e7021f504919c4ff5631245acc33ed6\n"
let cells column file = List.map (fun cell -> cell column) (Golden.rows file)

let rejects substring contents =
  raises_match (Exn.failure ~substring) (fun () ->
      Golden.rows (golden contents))

let texts =
  group "text"
    [
      group "matches its body"
        [ Golden.text "listing.golden" (fun () -> "line 1\nline 2\n") ];
      group "fails on another text"
        [
          xfail ~reason:"the body ends with a newline"
            (Golden.text "listing.golden" (fun () -> "line 1\nline 2"));
        ];
    ]

let tables =
  group "table"
    [
      test "rows are the lines after the columns, in order" (fun () ->
          equal (list string) [ "int"; "half" ] (cells "lub" "lattice.golden"));
      test "columns are the names on the first line, in order" (fun () ->
          equal (list string) [ "a"; "b"; "lub" ]
            (Golden.columns "lattice.golden"));
      test "an empty cell is kept" (fun () ->
          equal (list string) [ "" ]
            (cells "b" (golden (header ^ "a\tb\nx\t\n"))));
      test "a file without the header is not a golden" (fun () ->
          rejects "not a golden" "a\tb\nx\ty\n");
      test "a file without a final newline is not a golden" (fun () ->
          rejects "not a golden" (header ^ "a\tb\nx\ty"));
      test "a table needs a row" (fun () ->
          rejects "no row" (header ^ "a\tb\n"));
      test "a row needs a cell per column" (fun () ->
          rejects "line 3: 1 cells for 2 columns" (header ^ "a\tb\nx\n"));
      test "cell rejects an unknown column" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"no column \"c\"") (fun () ->
              cells "c" "lattice.golden"));
    ]

let path_ends_with name = equal (list string) [ "lattice.golden"; name ]

let cases =
  group "cases"
    [
      Golden.cases "lattice.golden" (fun cell ->
          path_ends_with ("a=" ^ cell "a") (List.tl (current_test ())));
      group "keyed"
        [
          Golden.cases ~key:[ "b"; "a" ] "lattice.golden" (fun cell ->
              path_ends_with
                ("b=" ^ cell "b" ^ " a=" ^ cell "a")
                (List.tl (List.tl (current_test ()))));
        ];
    ]

(* Graphs: a golden's storage has tinygrad's slots. chained_functions calls one
   body three times, each with call-local storage, which its schedule makes into
   buffers of fresh slots. *)

let slots u =
  let open Tolk in
  List.filter_map
    (fun n -> match Ops.arg n with Ops.Param p -> Some p.slot | _ -> None)
    (Ops.toposort ~calls:Enter u)

let schedules_apart () =
  let open Tolk in
  let big = Golden.sink "../schedule/schedule/chained_functions.golden" in
  let linear, _ = Schedule.create_linear_with_vars big in
  let held = Ops.backward_slice_with_self big in
  let made =
    List.concat_map slots
      (List.filter
         (fun n -> Ops.op n = Buffer && not (Ops.Nodes.mem n held))
         (Ops.toposort ~calls:Enter linear))
  in
  not_equal (list int) [] made;
  equal (list int) [] (List.filter (fun k -> List.mem k (slots big)) made)

let graphs =
  group "sink"
    [
      test
        "a schedule of a golden makes storage in slots the golden does not name"
        schedules_apart;
    ]

let () = exit (run "Golden" [ texts; tables; cases; graphs ])
