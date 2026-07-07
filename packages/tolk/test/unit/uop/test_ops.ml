open Windtrap
open Tolk_uop

let rec read_all ic b =
  match input_line ic with
  | line ->
      Buffer.add_string b line;
      Buffer.add_char b '\n';
      read_all ic b
  | exception End_of_file -> Buffer.contents b

let tinygrad_init root =
  Filename.concat root "_tinygrad/tinygrad/uop/__init__.py"

let find_repo_root () =
  let rec loop dir =
    if Sys.file_exists (tinygrad_init dir) then dir
    else
      let parent = Filename.dirname dir in
      if String.equal parent dir then
        invalid_arg "could not find vendored tinygrad checkout"
      else loop parent
  in
  loop (Sys.getcwd ())

let run_tinygrad_groups root =
  let code =
    {|
import sys
root = sys.argv[1]
sys.path.insert(0, root + "/_tinygrad")
from tinygrad.uop import Ops, GroupOp

groups = [
  "Unary",
  "Binary",
  "Ternary",
  "ALU",
  "Broadcastable",
  "Elementwise",
  "Defines",
  "Irreducible",
  "Movement",
  "Commutative",
  "Associative",
  "Idempotent",
  "Reduce",
  "Comparison",
  "All",
]

def names(group):
  return ",".join(op.name for op in Ops if op in group)

print("Ops=" + ",".join(op.name for op in Ops))
for group in groups:
  print(group + "=" + names(getattr(GroupOp, group)))
|}
  in
  let command =
    String.concat " "
      [ "python3"; "-c"; Filename.quote code; Filename.quote root ]
  in
  let ic, oc, ec = Unix.open_process_full command (Unix.environment ()) in
  close_out oc;
  let stdout = read_all ic (Buffer.create 1024) in
  let stderr = read_all ec (Buffer.create 256) in
  match Unix.close_process_full (ic, oc, ec) with
  | Unix.WEXITED 0 -> stdout
  | Unix.WEXITED code ->
      failwith
        (Printf.sprintf "tinygrad GroupOp probe exited %d:\n%s" code stderr)
  | Unix.WSIGNALED signal ->
      failwith
        (Printf.sprintf "tinygrad GroupOp probe signaled %d:\n%s" signal stderr)
  | Unix.WSTOPPED signal ->
      failwith
        (Printf.sprintf "tinygrad GroupOp probe stopped %d:\n%s" signal stderr)

let split_names s = if String.equal s "" then [] else String.split_on_char ',' s

let parse_table text =
  let parse_line line =
    match String.index_opt line '=' with
    | None -> invalid_arg ("bad tinygrad parity row: " ^ line)
    | Some i ->
        let key = String.sub line 0 i in
        let value = String.sub line (i + 1) (String.length line - i - 1) in
        key, split_names value
  in
  text |> String.split_on_char '\n'
  |> List.filter (fun line -> not (String.equal line ""))
  |> List.map parse_line

let expect_names table key got =
  let want =
    match List.assoc_opt key table with
    | Some names -> names
    | None -> invalid_arg ("missing tinygrad parity row: " ^ key)
  in
  is_true ~msg:("tinygrad " ^ key ^ " parity") (got = want)

let names ops = List.map Ops.name ops

let op_order_matches_vendored_tinygrad () =
  let table = find_repo_root () |> run_tinygrad_groups |> parse_table in
  expect_names table "Ops" (names Ops.Group.all);
  expect_names table "All" (names Ops.Group.all)

let group_memberships_match_vendored_tinygrad () =
  let table = find_repo_root () |> run_tinygrad_groups |> parse_table in
  let groups =
    [
      "Unary", names Ops.Group.unary;
      "Binary", names Ops.Group.binary;
      "Ternary", names Ops.Group.ternary;
      "ALU", names Ops.Group.alu;
      "Broadcastable", names Ops.Group.broadcastable;
      "Elementwise", names Ops.Group.elementwise;
      "Defines", names Ops.Group.defines;
      "Irreducible", names Ops.Group.irreducible;
      "Movement", names Ops.Group.movement;
      "Commutative", names Ops.Group.commutative;
      "Associative", names Ops.Group.associative;
      "Idempotent", names Ops.Group.idempotent;
      "Reduce", names Ops.Group.reduce;
      "Comparison", names Ops.Group.comparison;
    ]
  in
  List.iter (fun (key, got) -> expect_names table key got) groups

let group_predicates_match_group_lists () =
  let open Ops.Group in
  let agrees name group pred =
    is_true ~msg:(name ^ " predicate parity")
      (List.for_all (fun op -> Bool.equal (pred op) (mem op group)) all)
  in
  agrees "Unary" unary is_unary;
  agrees "Binary" binary is_binary;
  agrees "Ternary" ternary is_ternary;
  agrees "ALU" alu is_alu;
  agrees "Broadcastable" broadcastable is_broadcastable;
  agrees "Elementwise" elementwise is_elementwise;
  agrees "Defines" defines is_define;
  agrees "Irreducible" irreducible is_irreducible;
  agrees "Movement" movement is_movement;
  agrees "Commutative" commutative is_commutative;
  agrees "Associative" associative is_associative;
  agrees "Idempotent" idempotent is_idempotent;
  agrees "Reduce" reduce is_reduce;
  agrees "Comparison" comparison is_comparison

let () =
  run "tolk.uop.ops_parity"
    [
      group "Ops parity"
        [
          test "op order matches vendored tinygrad"
            op_order_matches_vendored_tinygrad;
          test "GroupOp memberships match vendored tinygrad"
            group_memberships_match_vendored_tinygrad;
          test "predicates match public groups"
            group_predicates_match_group_lists;
        ];
    ]
