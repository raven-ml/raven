open Windtrap
open Tolk

let hash body = Digest.(to_hex (string body))

let write path text =
  Out_channel.with_open_bin path (fun oc -> output_string oc text)

(* [file contents] is a fresh file holding [contents]. *)
let file contents =
  let path = temp_file ~suffix:".golden" () in
  write path contents;
  path

let golden_contents body = body ^ "\n"
let tolk_contents ~line body = line ^ "\n" ^ body ^ "\n"

(* [pair ?golden ?tolk ()] is the name of a golden, [x.golden], in a fresh
   directory, beside [x.tolk]: either file holds the given contents, or is
   absent. The name is absolute, so the suite's executable does not locate
   it. *)
let pair ?golden ?tolk () =
  let dir = temp_dir ~prefix:"tolk" () in
  let name = Filename.concat dir "x.golden" in
  Option.iter (fun b -> write name (golden_contents b)) golden;
  Option.iter (fun t -> write (Filename.concat dir "x.tolk") t) tolk;
  name

let tolk_of file = Filename.remove_extension file ^ ".tolk"

let fails_naming subs f =
  List.iter (fun sub -> raises_match (Exn.failure ~sub) f) subs

(* Under [-u] a run writes the [.tolk] of each golden test it runs, for good, so
   these tests start the executable again, in a fixture directory outside the
   build tree, named by [update_var]. That run checks [x.golden] against the
   text in [actual], or, where there is none, against the graph [sink] reads
   from it: what a suite does to number its output like the recorded one. *)

let update_var = "TOLK_GOLDEN_UPDATE_DIR"

let () =
  match Sys.getenv_opt update_var with
  | None -> ()
  | Some dir ->
      let file = Filename.concat dir "x.golden" in
      let actual = Filename.concat dir "actual" in
      let test =
        if Sys.file_exists actual then
          Golden.text file (fun () ->
              In_channel.with_open_bin actual In_channel.input_all)
        else Golden.graph file (fun () -> Golden.sink file)
      in
      exit (run "rerun" [ group "rerun" [ test ] ])

type output = Text of string | Sink

(* [rerun ~argv ?golden ?tolk output] runs the check of [output] against a
   golden with the arguments [argv], and is its exit code, the [.tolk] it leaves
   and its report. In the temporary directory, which no build tree holds:
   windtrap reads a path under [_build] as dune's copy of a source file. dune
   points TMPDIR at a directory of its own inside the one it was given, which
   opam's sandbox lets a build write where it makes /tmp read-only. *)
let rerun ~argv ?golden ?tolk output =
  let dir = Filename.temp_dir "tolk_golden" "" in
  Fun.protect
    ~finally:(fun () ->
      Array.iter (fun f -> Sys.remove (Filename.concat dir f)) (Sys.readdir dir);
      Sys.rmdir dir)
    (fun () ->
      Option.iter
        (fun b -> write (Filename.concat dir "x.golden") (golden_contents b))
        golden;
      Option.iter (write (Filename.concat dir "x.tolk")) tolk;
      (match output with
      | Text actual -> write (Filename.concat dir "actual") actual
      | Sink -> ());
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
          (Array.append [| Sys.executable_name |] argv)
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
      let code = match status with WEXITED n -> n | _ -> -1 in
      (code, read "x.tolk", Option.value (read "log") ~default:""))

(* [leaves expected result] checks that the run passed and left the [.tolk]
   [expected]. *)
let leaves expected (code, tolk, log) =
  equal ~msg:log int 0 code;
  equal (option string) expected tolk

let update = rerun ~argv:[| "-u" |]

(* Goldens *)

let goldens =
  group "goldens"
    [
      test
        "a relative name is read beside the executable, whatever the working \
         directory" (fun () ->
          chdir (temp_dir ());
          equal text "line 1\nline 2\n" (Golden.body "listing.golden"));
      test "a file that cannot be read raises Sys_error" (fun () ->
          raises_match (Exn.sys_error ?sub:None) (fun () ->
              Golden.body (pair ())));
      cases "a file that is not a golden raises Failure naming it" ~id:fst
        [
          ("an empty golden", (Some "", None, `Golden));
          ("a golden without its final newline", (Some "line", None, `Golden));
          ( "a .tolk whose first line is not # tolk",
            (None, Some "# tlok D1\nours\n", `Tolk) );
          ( "a .tolk with two golden= hashes",
            (None, Some "# tolk D1 golden=a golden=b\nours\n", `Tolk) );
          ("a .tolk of a header alone", (None, Some "# tolk D1\n", `Tolk));
        ]
        (fun (_, (contents, tolk, faulty)) ->
          let name =
            match contents with Some c -> file c | None -> pair ?tolk ()
          in
          let faulty = if faulty = `Golden then name else tolk_of name in
          fails_naming [ faulty ] (fun () -> Golden.body name));
    ]

(* Tolk's output *)

let tolk_tests =
  let upstream = "upstream" in
  group "Tolk's output"
    [
      test
        "a .tolk beside its golden, naming an entry and the golden's hash, is \
         read" (fun () ->
          equal string "ours"
            (Golden.body
               (pair ~golden:upstream
                  ~tolk:
                    (tolk_contents
                       ~line:("# tolk D1 golden=" ^ hash upstream)
                       "ours")
                  ())));
      test "a .tolk with no .golden carries no hash" (fun () ->
          equal string "ours" (Golden.body (pair ~tolk:"# tolk D73\nours\n" ())));
      cases "reading a .tolk raises Failure naming it and what is wrong" ~id:fst
        (let h = hash upstream in
         [
           ( "a header naming no entry",
             (Some upstream, "# tolk golden=" ^ h, "ours", []) );
           ( "a header token neither an entry nor a hash",
             (None, "# tolk D1 note", "ours", [ "note" ]) );
           ( "no hash beside a golden",
             (Some upstream, "# tolk D1", "ours", [ h ]) );
           ( "a hash of a golden that moved",
             ( Some upstream,
               "# tolk D81 D96 golden=" ^ hash "old upstream",
               "ours",
               [ "D81 D96"; hash "old upstream"; h ] ) );
           ( "a hash and no golden",
             (None, "# tolk D1 golden=abc", "ours", [ "abc" ]) );
           ( "a body equal to the golden's",
             (Some upstream, "# tolk D1 golden=" ^ h, upstream, []) );
         ])
        (fun (_, (golden, line, body, data)) ->
          let name = pair ?golden ~tolk:(tolk_contents ~line body) () in
          fails_naming (tolk_of name :: data) (fun () -> Golden.body name));
      test "a table whose golden moved names the new hash" (fun () ->
          let name =
            pair ~golden:"a\nx"
              ~tolk:(tolk_contents ~line:"# tolk D1 golden=stale" "a\ny")
              ()
          in
          let message = require_some (Golden.problem name) in
          contains ~sub:(hash "a\nx") message);
      test
        "under -u text writes a differing output as the .tolk, keeping its \
         entries" (fun () ->
          leaves
            (Some ("# tolk D81 D96 golden=" ^ hash "moved" ^ "\ntolk's new\n"))
            (update ~golden:"moved" ~tolk:"# tolk D81 D96 golden=stale\nours\n"
               (Text "tolk's new")));
      test "under -u a new .tolk names no entry" (fun () ->
          leaves
            (Some ("# tolk golden=" ^ hash upstream ^ "\nours\n"))
            (update ~golden:upstream (Text "ours")));
      test "under -u a .tolk of no golden stays without a hash" (fun () ->
          leaves (Some "# tolk D73\nours\n")
            (update ~tolk:"# tolk D73\nold\n" (Text "ours")));
      test "under -u an output equal to its golden writes nothing" (fun () ->
          leaves None (update ~golden:"same" (Text "same")));
      test
        "under -u an output equal to its golden fails beside a .tolk, which \
         stays" (fun () ->
          let tolk =
            tolk_contents ~line:("# tolk D1 golden=" ^ hash "same") "old"
          in
          let code, left, _ = update ~golden:"same" ~tolk (Text "same") in
          equal int 1 code;
          equal (option string) (Some tolk) left);
    ]

(* updating *)

let updating =
  group "updating"
    [
      test
        "is false without -u, where a differing output fails and writes nothing"
        (fun () ->
          let code, left, _ =
            rerun ~argv:[||] ~golden:"upstream" (Text "ours")
          in
          equal int 1 code;
          equal (option string) None left);
    ]

(* body *)

let body =
  group "body"
    [
      test "is the .tolk's body where one exists" (fun () ->
          equal string "tolk's\nbody" (Golden.body "kept_diff.golden"));
      test "is the golden's body otherwise" (fun () ->
          equal string "line 1\nline 2\n" (Golden.body "listing.golden"));
    ]

(* text *)

let texts =
  group "text"
    [
      group "passes on its golden's body"
        [ Golden.text "listing.golden" (fun () -> "line 1\nline 2\n") ];
      group "fails on another text"
        [
          xfail ~reason:"the body ends with a newline"
            (Golden.text "listing.golden" (fun () -> "line 1\nline 2"));
        ];
      group "compares with the .tolk where one exists"
        [ Golden.text "kept_diff.golden" (fun () -> "tolk's\nbody") ];
      group "fails where the .tolk's golden moved"
        [
          xfail ~reason:"stale.tolk carries another hash"
            (Golden.text "stale.golden" (fun () -> "tolk's body"));
        ];
      (* The suite declares it, so declaring reads nothing. *)
      group "reads its golden when the test runs"
        [
          xfail ~reason:"absent.golden does not exist"
            (Golden.text "absent.golden" (fun () -> ""));
        ];
    ]

(* graph, sink, upstream *)

(* kept_graph.golden is a sink of nothing; its .tolk, a sink of an empty
   stack. *)
let kept = Ops.sink [ Ops.v Stack ]

let graphs =
  group "graph"
    [
      group "compares with the .tolk where one exists"
        [ Golden.graph "kept_graph.golden" (fun () -> kept) ];
    ]

let sinks =
  group "sink"
    [
      test "is the .tolk's graph where one exists" (fun () ->
          equal Uops.uop kept (Golden.sink "kept_graph.golden"));
      test "raises Failure naming a golden whose body is not a graph" (fun () ->
          let name = pair ~golden:"not a graph" () in
          fails_naming [ name ] (fun () -> Golden.sink name));
      test "under -u reads a .tolk whose golden moved, which graph rewrites"
        (fun () ->
          let moved = "0 Ops.STACK dtypes.void []" in
          let ours = "0 Ops.SINK dtypes.void []" in
          leaves
            (Some
               ("# tolk D81 golden=" ^ hash moved ^ "\n"
               ^ Graph.to_string (Graph.of_string ours)
               ^ "\n"))
            (update ~golden:moved
               ~tolk:("# tolk D81 golden=stale\n" ^ ours ^ "\n")
               Sink));
    ]

let upstreams =
  group "upstream"
    [
      test "is the golden's graph, whatever .tolk sits beside it" (fun () ->
          equal Uops.uop (Ops.sink []) (Golden.upstream "kept_graph.golden"));
    ]

(* cases *)

(* lattice.golden's column [lub] is the least upper bound of [a] and [b], of int
   and half. *)
let lub cell =
  let half = cell "a" = "half" || cell "b" = "half" in
  equal string (if half then "half" else "int") (cell "lub")

(* A table and its .tolk, where rows y and z differ. *)
let differing ?(golden = Some "a\tb\nx\t1\ny\t2\nz\t3")
    ?(tolk = "a\tb\nx\t1\ny\t9\nz\t8") () =
  let line =
    match golden with
    | Some g -> "# tolk D1 golden=" ^ hash g
    | None -> "# tolk D1"
  in
  pair ?golden ~tolk:(tolk_contents ~line tolk) ()

let row_y_and_z = [ "a=y"; "a=z" ]

let cases_tests =
  group "cases"
    [
      group "runs check on each row" [ Golden.cases "lattice.golden" lub ];
      group "identifies a row by its key cells"
        [
          Golden.cases ~key:[ "b"; "a" ] "lattice.golden" lub;
          group "two rows of one key fail the table"
            [
              xfail ~reason:"column b is int on both rows"
                (Golden.cases ~key:[ "b" ] "lattice.golden" lub);
            ];
        ];
      group "is one failing test where a .tolk is stale"
        [
          xfail ~reason:"stale_table.tolk carries another hash"
            (Golden.cases "stale_table.golden" ignore);
        ];
      test "raises Sys_error when applied to a file that cannot be read"
        (fun () ->
          raises_match (Exn.sys_error ?sub:None) (fun () ->
              Golden.cases (pair ()) ignore));
      group "holds a listed row to the .tolk"
        [
          Golden.cases "differs.golden"
            ~differs:[ ("D1", row_y_and_z) ]
            (fun cell ->
              let tolk = [ ("x", "1"); ("y", "9"); ("z", "8") ] in
              equal string (List.assoc (cell "a") tolk) (cell "b"));
        ];
      test "is a table of rows where differs is exact" (fun () ->
          equal (option string) None
            (Golden.problem ~differs:[ ("D1", row_y_and_z) ] (differing ())));
      cases
        "is one failing test naming what is wrong where differs is not exact"
        ~id:fst
        [
          ( "a row that differs is not listed",
            (`Pair, [ ("D1", [ "a=y" ]) ], [ "a=z" ]) );
          ( "a listed row equals the golden",
            (`Pair, [ ("D1", "a=x" :: row_y_and_z) ], [ "a=x" ]) );
          ( "a pattern holds on no row",
            (`Pair, [ ("D1", "a=w" :: row_y_and_z) ], [ "a=w" ]) );
          ( "an entry is not in the header",
            (`Pair, [ ("D2", row_y_and_z) ], [ "D2" ]) );
          ( "the tables have different numbers of rows",
            (`More_rows, [ ("D1", "a=w" :: row_y_and_z) ], [ "x.golden" ]) );
          ( "the golden is missing",
            (`No_golden, [ ("D1", row_y_and_z) ], [ "x.golden" ]) );
          ( "the .tolk is missing",
            (`No_tolk, [ ("D1", row_y_and_z) ], [ "x.golden" ]) );
        ]
        (fun (_, (files, differs, named)) ->
          let name =
            match files with
            | `Pair -> differing ()
            | `More_rows -> differing ~tolk:"a\tb\nx\t1\ny\t9\nz\t8\nw\t4" ()
            | `No_golden -> differing ~golden:None ()
            | `No_tolk -> pair ~golden:"a\tb\nx\t1\ny\t2\nz\t3" ()
          in
          let message = require_some (Golden.problem ~differs name) in
          List.iter (fun sub -> contains ~sub message) named);
    ]

(* columns, rows *)

let columns =
  group "columns"
    [
      test "is the names on the first line, in order" (fun () ->
          equal (list string) [ "a"; "b"; "lub" ]
            (Golden.columns "lattice.golden"));
    ]

let cells column file = List.map (fun cell -> cell column) (Golden.rows file)

let rows =
  group "rows"
    [
      test "is the lines after the columns, in order" (fun () ->
          equal (list string) [ "int"; "half" ] (cells "lub" "lattice.golden"));
      test "reads the .tolk where one exists" (fun () ->
          equal (list string) [ "1"; "9"; "8" ] (cells "b" "differs.golden"));
      test "keeps an empty cell" (fun () ->
          equal (list string) [ "" ] (cells "b" (file "a\tb\nx\t\n")));
      test "raises Failure naming a table with no row" (fun () ->
          let name = file "a\tb\n" in
          fails_naming [ name ] (fun () -> Golden.rows name));
      test "raises Failure naming a line whose cells are not one per column"
        (fun () ->
          let name = file "a\tb\nx\n" in
          fails_naming [ name; "line 2" ] (fun () -> Golden.rows name));
      test
        "gives a cell function that raises Invalid_argument naming an unknown \
         column" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"\"c\"") (fun () ->
              cells "c" "lattice.golden"));
    ]

(* Graphs: a golden's storage has tinygrad's slots. chained_functions calls one
   body three times, each with call-local storage, which its schedule makes into
   buffers of fresh slots. *)

let slots u =
  List.filter_map
    (fun n -> match Ops.arg n with Ops.Param p -> Some p.slot | _ -> None)
    (Ops.toposort ~calls:Enter u)

let schedules_apart () =
  let big = Golden.sink "../schedule/schedule/chained_functions.golden" in
  let linear, _ = Schedule.create_linear_with_vars ~cores:Cores.host big in
  let held = Ops.backward_slice_with_self ~calls:Skip big in
  let made =
    List.concat_map slots
      (List.filter
         (fun n -> Ops.op n = Buffer && not (Ops.Nodes.mem n held))
         (Ops.toposort ~calls:Enter linear))
  in
  not_equal (list int) [] made;
  equal (list int) [] (List.filter (fun k -> List.mem k (slots big)) made)

let schedules =
  group "sink"
    [
      test
        "a schedule of a golden makes storage in slots the golden does not name"
        schedules_apart;
    ]

let () =
  exit
    (run "Golden"
       [
         goldens;
         tolk_tests;
         updating;
         body;
         texts;
         graphs;
         sinks;
         upstreams;
         cases_tests;
         columns;
         rows;
         schedules;
       ])
