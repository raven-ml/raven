(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Checks that the regressions ledger maps every old rune test.

   [check_regressions ledger list...] reads the ledger's rows whose source is an
   old test, [| old: <file> <path>; <path>; ... | <behaviour> | <outcome> |],
   and one listing per old test file, [<file>.list] such as [test_grad.ml.list],
   holding the paths of its tests, one per line, as the suite's [-l] prints
   them; a listing of the one line [*] stands for a file that is a single test.
   A path in a row is a suite's full path, or a group's path followed by [› *],
   or [*] for the whole file, and [\|] stands for a bar in a test's name. A
   row's outcome is a rune.next test's path (it holds [›]), a cram test
   ([<path>.t]) or [dropped: <reason>].

   A row may name a test that no longer exists: it records what became of a
   deleted test. It prints every test no row maps and every row whose outcome is
   neither, and exits 1 if there is one. *)

let read_lines path =
  In_channel.with_open_text path In_channel.input_all
  |> String.split_on_char '\n' |> List.map String.trim
  |> List.filter (fun l -> l <> "")

let starts_with ~prefix s =
  String.length s >= String.length prefix
  && String.sub s 0 (String.length prefix) = prefix

let ends_with ~suffix s =
  let n = String.length s and k = String.length suffix in
  n >= k && String.sub s (n - k) k = suffix

(* The cells of a table row, without the outer bars; [\|] is a bar inside a
   cell. *)
let cells line =
  let cells = ref [] and cell = Buffer.create 64 in
  let n = String.length line in
  let rec go i =
    if i < n then
      match line.[i] with
      | '\\' when i + 1 < n && line.[i + 1] = '|' ->
          Buffer.add_char cell '|';
          go (i + 2)
      | '|' ->
          cells := String.trim (Buffer.contents cell) :: !cells;
          Buffer.clear cell;
          go (i + 1)
      | c ->
          Buffer.add_char cell c;
          go (i + 1)
  in
  go 0;
  match List.rev !cells with "" :: cells -> cells | _ -> []

type row = { line : int; file : string; paths : string list; outcome : string }

let rows ledger =
  List.concat
    (List.mapi
       (fun i line ->
         match cells line with
         | source :: _ :: outcome :: _ when starts_with ~prefix:"old: " source
           -> (
             let source = String.sub source 5 (String.length source - 5) in
             match String.index_opt source ' ' with
             | None -> [ { line = i + 1; file = source; paths = []; outcome } ]
             | Some k ->
                 let file = String.sub source 0 k in
                 let rest =
                   String.sub source (k + 1) (String.length source - k - 1)
                 in
                 let paths =
                   List.map String.trim (String.split_on_char ';' rest)
                 in
                 [ { line = i + 1; file; paths; outcome } ])
         | _ -> [])
       (In_channel.with_open_text ledger In_channel.input_all
       |> String.split_on_char '\n'))

(* Whether the row path [p] covers the test path [t]. *)
let covers p t =
  p = "*" || p = t
  || ends_with ~suffix:" › *" p
     && starts_with ~prefix:(String.sub p 0 (String.length p - 1)) t

let contains ~sub s =
  let n = String.length s and k = String.length sub in
  let rec at i = i + k <= n && (String.sub s i k = sub || at (i + 1)) in
  at 0

let valid_outcome o =
  starts_with ~prefix:"dropped: " o
  || contains ~sub:"›" o || ends_with ~suffix:".t" o

let () =
  match Array.to_list Sys.argv with
  | _ :: ledger :: listings ->
      let rows = rows ledger in
      let errors = ref 0 in
      let error fmt =
        incr errors;
        Printf.printf (fmt ^^ "\n")
      in
      List.iter
        (fun r ->
          if not (valid_outcome r.outcome) then
            error
              "REGRESSIONS.md:%d: %s: the outcome is neither a rune.next test \
               nor dropped: %s"
              r.line r.file r.outcome)
        rows;
      let listed = Hashtbl.create 64 in
      List.iter
        (fun listing ->
          let file = Filename.remove_extension (Filename.basename listing) in
          let tests = read_lines listing in
          Hashtbl.replace listed file ();
          let mine = List.filter (fun r -> r.file = file) rows in
          List.iter
            (fun t ->
              if
                t <> "*"
                && not
                     (List.exists
                        (fun r -> List.exists (fun p -> covers p t) r.paths)
                        mine)
              then error "%s: %s: no row maps this test" file t)
            tests;
          if tests = [ "*" ] && mine = [] then
            error "%s: no row maps this file" file)
        listings;
      List.iter
        (fun r ->
          if not (Hashtbl.mem listed r.file) then
            error "REGRESSIONS.md:%d: %s is no old test file" r.line r.file)
        rows;
      if !errors > 0 then (
        Printf.printf "%d problems\n" !errors;
        exit 1)
  | _ ->
      prerr_endline "usage: check_regressions LEDGER LISTING...";
      exit 2
