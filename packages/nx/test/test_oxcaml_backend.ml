(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Checks that nx-oxcaml still implements every member of the backend contract.

   nx-oxcaml implements the [nx.backend] virtual library, so a member added to
   [Backend_intf.S] must be added there too. It has its own dune-project and
   needs an OxCaml switch, so it is excluded from this build and never
   typechecked against the contract it implements — the break surfaces only when
   someone builds it by hand.

   This compares names, not types. It catches a member added to the contract and
   left unimplemented, which is the drift that happens in practice; a changed
   signature on an existing member still needs an OxCaml build to catch.

   The two sides are read differently, because they fail differently. A
   declaration missed on the contract side drops a requirement silently, so
   every declaration there must be understood or the check errors out. A
   definition missed on the implementation side only over-reports, so that side
   guards just the constructs that could hide a definition. *)

let contract_module_type = "Nx_core.Backend_intf.S"

(* Blank out comments so a declaration quoted in prose is not read as real. Line
   structure and columns are preserved. *)
let blank_comments src =
  let n = String.length src in
  let out = Bytes.of_string src in
  let blank i = if Bytes.get out i <> '\n' then Bytes.set out i ' ' in
  let depth = ref 0 and in_string = ref false and i = ref 0 in
  let starts j a b = j + 1 < n && src.[j] = a && src.[j + 1] = b in
  while !i < n do
    if !depth > 0 then
      if starts !i '(' '*' then (
        blank !i;
        blank (!i + 1);
        incr depth;
        i := !i + 2)
      else if starts !i '*' ')' then (
        blank !i;
        blank (!i + 1);
        decr depth;
        i := !i + 2)
      else (
        blank !i;
        incr i)
    else if !in_string then
      if src.[!i] = '\\' then i := !i + 2
      else (
        if src.[!i] = '"' then in_string := false;
        incr i)
    else if src.[!i] = '"' then (
      in_string := true;
      incr i)
    else if starts !i '(' '*' then (
      blank !i;
      blank (!i + 1);
      depth := 1;
      i := !i + 2)
    else incr i
  done;
  if !depth <> 0 then failwith "unterminated comment";
  Bytes.to_string out

let read path =
  let ic = open_in_bin path in
  let src = really_input_string ic (in_channel_length ic) in
  close_in ic;
  String.split_on_char '\n' (blank_comments src)

(* Every failure here means a maintainer has to come and teach this checker
   something, so say where. *)
let bail path lineno fmt =
  Printf.ksprintf
    (fun msg ->
      Printf.eprintf "%s:%d: %s\n" path (lineno + 1) msg;
      exit 2)
    fmt

let starts_with prefix line = String.starts_with ~prefix line
let drop n s = String.sub s n (String.length s - n)

let is_ident_char c =
  (c >= 'a' && c <= 'z')
  || (c >= 'A' && c <= 'Z')
  || (c >= '0' && c <= '9')
  || c = '_' || c = '\''

(* The identifier starting at [pos], or [None] if there is not one there. *)
let ident_at line pos =
  if pos >= String.length line || not (is_ident_char line.[pos]) then None
  else begin
    let stop = ref pos in
    while !stop < String.length line && is_ident_char line.[!stop] do
      incr stop
    done;
    Some (String.sub line pos (!stop - pos))
  end

let value_name path lineno line pos =
  match ident_at line pos with
  | Some name -> name
  | None -> bail path lineno "cannot read the value name from %S" line

(* [type ('a, 'b) t = ...] and [type context] both name the type last, after any
   parameters. [rest] starts just after the [type] keyword. *)
let type_name path lineno rest =
  let rest = String.trim rest in
  let after c =
    match String.index_opt rest c with
    | Some i -> String.trim (drop (i + 1) rest)
    | None -> bail path lineno "cannot read the type name from %S" rest
  in
  let named =
    if starts_with "(" rest then after ')'
    else if starts_with "'" rest then after ' '
    else rest
  in
  match ident_at named 0 with
  | Some name -> name
  | None -> bail path lineno "cannot read the type name from %S" rest

(* Declarations written at exactly [indent], with the module paths of any
   [include]s. Anything else at that indent is an error: a member hidden behind
   a construct this checker does not understand would go unchecked. *)
let declarations path ~indent lines =
  let members = ref [] and includes = ref [] in
  let pad = String.make indent ' ' in
  let declared line =
    String.length line > indent
    && String.sub line 0 indent = pad
    && line.[indent] <> ' '
  in
  List.iteri
    (fun lineno line ->
      if declared line then
        let rest = drop indent line in
        if starts_with "val " rest then
          members := ("val", value_name path lineno rest 4) :: !members
        else if starts_with "type " rest then
          members := ("type", type_name path lineno (drop 5 rest)) :: !members
        else if starts_with "include " rest then
          includes := String.trim (drop 8 rest) :: !includes
        else
          bail path lineno
            "cannot tell what %S declares; a member hidden here would go \
             unchecked, so teach this checker about it"
            (String.trim rest))
    lines;
  (List.rev !members, List.rev !includes)

(* The contract is the body of [module type S]. *)
let contract path =
  let lines = read path in
  let rec body = function
    | [] -> bail path 0 "no [module type S = sig] here; has the contract moved?"
    | line :: rest when starts_with "module type S = sig" line ->
        let rec until_end = function
          | [] -> bail path 0 "[module type S] is never closed"
          | "end" :: _ -> []
          | line :: rest -> line :: until_end rest
        in
        until_end rest
    | _ :: rest -> body rest
  in
  let members, includes = declarations path ~indent:2 (body lines) in
  if includes <> [] then
    bail path 0 "[module type S] includes %s; teach this checker about it"
      (String.concat ", " includes);
  members

(* The virtual library's interface is the contract plus whatever nx_backend.mli
   declares on top of it. *)
let virtual_interface ~contract_path ~mli_path =
  let extra, includes = declarations mli_path ~indent:0 (read mli_path) in
  if includes <> [ contract_module_type ] then
    bail mli_path 0 "expected [include %s], found %s" contract_module_type
      (match includes with
      | [] -> "no include"
      | includes -> String.concat ", " includes);
  contract contract_path @ extra

(* Names defined at the top level of an implementation. *)
let definitions path =
  let names = ref [] and last_keyword = ref "" in
  let value lineno line pos =
    (* [let () = ...] is a side effect, not a definition. *)
    if not (starts_with "()" (drop pos line)) then
      names := value_name path lineno line pos :: !names
  in
  let typ lineno line pos =
    names := type_name path lineno (drop pos line) :: !names
  in
  List.iteri
    (fun lineno line ->
      if starts_with "let rec " line then (
        last_keyword := "let";
        value lineno line 8)
      else if starts_with "let " line then (
        last_keyword := "let";
        value lineno line 4)
      else if starts_with "external " line then (
        last_keyword := "let";
        value lineno line 9)
      else if starts_with "type " line then (
        last_keyword := "type";
        typ lineno line 5)
      else if starts_with "and " line then
        if !last_keyword = "type" then typ lineno line 4
        else value lineno line 4
      else if starts_with "include " line || starts_with "module " line then
        bail path lineno
          "%S may define members this checker cannot enumerate; teach it about \
           them or drop the construct"
          (String.trim line))
    (read path);
  !names

let () =
  match Sys.argv with
  | [| _; contract_path; mli_path; impl_path |] ->
      let required = virtual_interface ~contract_path ~mli_path in
      let defined = definitions impl_path in
      let missing =
        List.filter (fun (_, name) -> not (List.mem name defined)) required
      in
      if missing = [] then
        Printf.printf "%s implements all %d members of %s.\n" impl_path
          (List.length required) contract_module_type
      else begin
        Printf.eprintf
          "%s no longer implements %s.\n\n\
           Missing:\n\
           %s\n\n\
           nx-oxcaml implements the nx.backend virtual library but is built with\n\
           a separate OxCaml toolchain, so this build does not compile it. Every\n\
           member added to the contract has to be implemented there too.\n"
          impl_path contract_module_type
          (String.concat "\n"
             (List.map
                (fun (kind, name) -> Printf.sprintf "  %s %s" kind name)
                missing));
        exit 1
      end
  | argv ->
      Printf.eprintf
        "usage: %s BACKEND_INTF_ML NX_BACKEND_MLI IMPLEMENTATION_ML\n" argv.(0);
      exit 2
