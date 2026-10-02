(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Talon's answers to the benchmark suite: writes one answer, or checks every
   answer at the CI size against the committed ones. bench_h2o.exe and
   bench_tpch.exe time them. *)

open Talon_next

let usage =
  {|usage: runner answer DATA ID FILE
       runner check ANSWERS DATA
       runner questions WORKLOAD/SIZE

ID is a question's id, such as groupby/1e7/q03 or tpch/sf1/q21, read from
the data generated under DATA.

answer    writes the question's canonical answer to the CSV file FILE.
check     runs every question at the CI size (groupby/1e6, join/1e6,
          tpch/sf0.1) and compares its answer with the committed one under
          ANSWERS, as answers.py does.
questions prints the ids of a workload's questions, one per line.|}

let fail fmt =
  Printf.ksprintf
    (fun s ->
      prerr_endline s;
      exit 1)
    fmt

(* Workloads *)

(* CI checks a family of workloads at one size. *)
type family = { ci : string; workload : data:string -> string -> Workload.t }

let families : (string * family) list = []

let family name =
  match List.assoc_opt name families with
  | Some f -> f
  | None ->
      fail "unknown workload %S; expected one of %s" name
        (String.concat ", " (List.map fst families))

let workload ~data id =
  match String.split_on_char '/' id with
  | [ name; size ] -> (family name).workload ~data size
  | _ -> fail "%S is not a workload id, such as groupby/1e7" id

let question ~data id =
  let cut = Option.value ~default:0 (String.rindex_opt id '/') in
  let w = workload ~data (String.sub id 0 cut) in
  let name = String.sub id (cut + 1) (String.length id - cut - 1) in
  match
    List.find_opt (fun (q : Workload.question) -> q.name = name) w.questions
  with
  | Some q -> (w, q)
  | None -> fail "%s has no question %S" w.id name

let run query = Error.get_ok (Query.run query)

(* Answers *)

let answer ~data id file =
  let w, q = question ~data id in
  let got = run (q.query (Workload.load w)) in
  Answer.write file (Answer.canonical ~ordered:w.ordered got)

let check ~answers ~data =
  let committed = Answer.read_index answers in
  let ran = ref [] and failures = ref 0 in
  let check_workload (w : Workload.t) =
    let tables = Workload.load w in
    let check_question (q : Workload.question) =
      let id = w.id ^ "/" ^ q.name in
      ran := id :: !ran;
      let problems =
        match List.assoc_opt id committed with
        | None -> [ "no committed answer" ]
        | Some (rows, stride) -> (
            try
              let got =
                Answer.canonical ~ordered:w.ordered (run (q.query tables))
              in
              let expected =
                Answer.read (schema got) (Answer.path answers id)
              in
              Answer.check ~expected ~rows ~stride got
            with Failure e | Invalid_argument e -> [ e ])
      in
      Printf.printf "%-20s %s\n%!" id (if problems = [] then "ok" else "FAIL");
      List.iter (Printf.printf "  %s\n%!") problems;
      if problems <> [] then incr failures
    in
    List.iter check_question w.questions
  in
  List.iter (fun (_, f) -> check_workload (f.workload ~data f.ci)) families;
  let gone = List.filter (fun (id, _) -> not (List.mem id !ran)) committed in
  List.iter (fun (id, _) -> Printf.printf "%-20s no question\n%!" id) gone;
  if !failures > 0 || gone <> [] then
    fail "%d questions disagree, %d committed answers have no question"
      !failures (List.length gone)

let questions id =
  let w = workload ~data:"" id in
  List.iter
    (fun (q : Workload.question) -> print_endline (w.id ^ "/" ^ q.name))
    w.questions

let () =
  match List.tl (Array.to_list Sys.argv) with
  | [ "answer"; data; id; file ] -> answer ~data id file
  | [ "check"; answers; data ] -> check ~answers ~data
  | [ "questions"; id ] -> questions id
  | _ -> fail "%s" usage
