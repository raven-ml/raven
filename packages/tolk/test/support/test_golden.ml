open Windtrap

let golden contents =
  let file = temp_file ~suffix:".golden" () in
  Out_channel.with_open_bin file (fun oc -> output_string oc contents);
  file

let header = "# tinygrad 6b7b018dae9ec20264f2a4b6a1322da1bb49b141\n"
let cells column file = List.map (fun cell -> cell column) (Golden.rows file)

let rejects substring contents =
  raises_match (Exn.failure ~sub:substring) (fun () ->
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
          raises_match (Exn.invalid_arg ~sub:"no column \"c\"") (fun () ->
              cells "c" "lattice.golden"));
    ]

(* A golden's table is one test, named by its file, that checks each row. *)
let cases =
  group "cases"
    [
      Golden.cases "lattice.golden" (fun _ ->
          equal (list string) [ "cases"; "lattice.golden" ] (current_test ()));
      group "keyed"
        [
          Golden.cases ~key:[ "b"; "a" ] "lattice.golden" (fun _ ->
              equal (list string)
                [ "cases"; "keyed"; "lattice.golden" ]
                (current_test ()));
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
  let held = Ops.backward_slice_with_self ~calls:Skip big in
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

(* Tolk's output: a [.tolk] beside its golden. Each test makes its own pair in
   a directory of the test, and a file name is absolute there, so the suite's
   executable is not what locates it. *)

let hash body = Digest.(to_hex (string body))
let write path text =
  Out_channel.with_open_bin path (fun oc -> output_string oc text)

let golden_contents body = header ^ body ^ "\n"
let tolk_contents ~line body = line ^ "\n" ^ body ^ "\n"

(* [pair ?golden ?tolk ()] is the name of a golden, with its files in a fresh
   directory. *)
let pair ?golden ?tolk () =
  let dir = temp_dir ~prefix:"tolk" () in
  let name = Filename.concat dir "x.golden" in
  Option.iter (fun b -> write name (golden_contents b)) golden;
  Option.iter (fun t -> write (Filename.concat dir "x.tolk") t) tolk;
  name

let tolk_rejects substring file =
  raises_match (Exn.failure ~sub:substring) (fun () -> Golden.body file)

let tolk_tests =
  group "tolk"
    [
      group "prefers x.tolk beside x.golden"
        [
          (* The fixture's hash is a literal in the file: it is the same on
             every machine and across runs. *)
          Golden.text "kept_diff.golden" (fun () -> "tolk's\nbody");
          test "Golden.body reads the .tolk" (fun () ->
              equal string "tolk's\nbody" (Golden.body "kept_diff.golden"));
          test "a .tolk with no .golden needs no hash" (fun () ->
              equal string "ours"
                (Golden.body (pair ~tolk:"# tolk D73\nours\n" ())));
        ];
      group "fails when x.tolk does not match x.golden"
        [
          test "a golden that moved fails, naming the entries and hashes"
            (fun () ->
              let file =
                pair ~golden:"new upstream"
                  ~tolk:
                    (tolk_contents
                       ~line:("# tolk D81 D96 golden=" ^ hash "old upstream")
                       "ours")
                  ()
              in
              List.iter
                (fun sub -> tolk_rejects sub file)
                [
                  "moved beneath D81 D96";
                  "golden=" ^ hash "old upstream";
                  hash "new upstream";
                ]);
          test "a .tolk equal to its golden fails" (fun () ->
              tolk_rejects "delete"
                (pair ~golden:"same"
                   ~tolk:
                     (tolk_contents ~line:("# tolk D1 golden=" ^ hash "same")
                        "same")
                   ()));
          test "a .tolk naming no entry fails" (fun () ->
              tolk_rejects "names no entry"
                (pair ~golden:"upstream"
                   ~tolk:
                     (tolk_contents ~line:("# tolk golden=" ^ hash "upstream")
                        "ours")
                   ()));
          test "a .tolk of no golden carrying a hash fails" (fun () ->
              tolk_rejects "does not exist"
                (pair ~tolk:"# tolk D1 golden=abc\nours\n" ()));
          test "a .tolk beside a golden without its hash fails" (fun () ->
              tolk_rejects "no golden= hash"
                (pair ~golden:"upstream" ~tolk:"# tolk D1\nours\n" ()));
          test "an unknown token in the header fails" (fun () ->
              tolk_rejects "neither a D-number"
                (pair ~tolk:"# tolk D1 note\nours\n" ()));
          group "the checks fail the test"
            [
              xfail ~reason:"x.tolk has a stale hash"
                (Golden.text "stale.golden" (fun () -> "tolk's body"));
              xfail ~reason:"a table's stale .tolk is a failing test"
                (Golden.cases "stale_table.golden" (fun _ -> ()));
            ];
        ];
    ]

(* Under [-u] a run writes the [.tolk] of each golden test it runs. A run under
   [-u] rewrites files for good, so the test starts the executable again, with
   a fixture directory outside the build tree. It takes the name of the golden
   from [update_var], and exits. *)

let update_var = "TOLK_GOLDEN_UPDATE_DIR"

let () =
  match Sys.getenv_opt update_var with
  | None -> ()
  | Some dir ->
      let file = Filename.concat dir "x.golden" in
      let actual =
        In_channel.with_open_bin (Filename.concat dir "actual")
          In_channel.input_all
      in
      exit
        (run "update"
           [ group "update" [ Golden.text file (fun () -> actual) ] ])

(* A directory no build tree holds: windtrap reads a path under [_build] as
   dune's copy of a source file. *)
let scratch_base () =
  if Sys.win32 then Filename.get_temp_dir_name () else "/tmp"

let rerun_updating ?golden ?tolk actual =
  let dir = Filename.temp_dir ~temp_dir:(scratch_base ()) "tolk_golden" "" in
  Fun.protect
    ~finally:(fun () ->
      Array.iter
        (fun f -> Sys.remove (Filename.concat dir f))
        (Sys.readdir dir);
      Sys.rmdir dir)
    (fun () ->
      Option.iter
        (fun b -> write (Filename.concat dir "x.golden") (golden_contents b))
        golden;
      Option.iter (write (Filename.concat dir "x.tolk")) tolk;
      write (Filename.concat dir "actual") actual;
      let log = Filename.concat dir "log" in
      let out = Unix.openfile log [ O_WRONLY; O_CREAT; O_TRUNC ] 0o644 in
      let env =
        Array.append
          [| update_var ^ "=" ^ dir; "WINDTRAP_PROJECT_ROOT=" ^ dir |]
          (Array.of_list
             (List.filter
                (fun v -> not (String.starts_with ~prefix:"CI=" v))
                (Array.to_list (Unix.environment ()))))
      in
      let pid =
        Unix.create_process_env Sys.executable_name
          [| Sys.executable_name; "-u" |]
          env Unix.stdin out out
      in
      let _, status = Unix.waitpid [] pid in
      Unix.close out;
      let read name =
        let p = Filename.concat dir name in
        if Sys.file_exists p then
          Some (In_channel.with_open_bin p In_channel.input_all)
        else None
      in
      (status, read "x.tolk", read "log"))

let exits_zero (status, _, log) =
  if status <> Unix.WEXITED 0 then
    failf "the run under -u failed:@\n%s" (Option.value log ~default:"")

let tolk_of (_, tolk, _) = tolk

let update_tests =
  group "tolk -u"
    [
      test "writes a new .tolk with the hash and no entry" (fun () ->
          let result = rerun_updating ~golden:"upstream" "ours" in
          exits_zero result;
          equal (option string)
            (Some ("# tolk golden=" ^ hash "upstream" ^ "\nours\n"))
            (tolk_of result));
      test "keeps the entries and rewrites the hash and body" (fun () ->
          let result =
            rerun_updating ~golden:"moved"
              ~tolk:"# tolk D81 D96 golden=stale\nours\n"
              "tolk's new"
          in
          exits_zero result;
          equal (option string)
            (Some
               ("# tolk D81 D96 golden=" ^ hash "moved" ^ "\ntolk's new\n"))
            (tolk_of result));
      test "writes nothing when the output is the golden's" (fun () ->
          let result = rerun_updating ~golden:"same" "same" in
          exits_zero result;
          equal (option string) None (tolk_of result));
      test "a .tolk with no golden stays without a hash" (fun () ->
          let result =
            rerun_updating ~tolk:"# tolk D73\nold\n" "ours"
          in
          exits_zero result;
          equal (option string) (Some "# tolk D73\nours\n") (tolk_of result));
    ]

let () =
  exit (run "Golden" [ texts; tables; cases; graphs; tolk_tests; update_tests ])
