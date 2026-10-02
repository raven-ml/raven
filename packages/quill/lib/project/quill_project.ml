(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type notebook = { title : string; path : string }

type toc_item =
  | Notebook of notebook * toc_item list
  | Section of string
  | Separator

type config = {
  title : string option;
  authors : string list;
  description : string option;
  output : string option;
  edit_url : string option;
}

type t = { title : string; root : string; toc : toc_item list; config : config }

let default_config =
  {
    title = None;
    authors = [];
    description = None;
    output = None;
    edit_url = None;
  }

(* ───── Config parser ───── *)

let trim = String.trim

let leading_spaces s =
  let len = String.length s in
  let rec loop i = if i < len && s.[i] = ' ' then loop (i + 1) else i in
  loop 0

let is_comment_or_blank s =
  let s = trim s in
  String.length s = 0 || s.[0] = '#'

let is_separator s = trim s = "---"

let is_section s =
  let s = trim s in
  let len = String.length s in
  len >= 2 && s.[0] = '[' && s.[len - 1] = ']'

let parse_section s =
  let s = trim s in
  String.sub s 1 (String.length s - 2)

let parse_kv s =
  match String.index_opt s '=' with
  | None -> None
  | Some i ->
      let key = trim (String.sub s 0 i) in
      let value = trim (String.sub s (i + 1) (String.length s - i - 1)) in
      Some (key, value)

let is_toc_entry s = parse_kv s <> None || is_section s || is_separator s

let parse_metadata (cfg : config) key value =
  match key with
  | "title" -> { cfg with title = Some value }
  | "authors" ->
      let authors = List.map trim (String.split_on_char ',' value) in
      { cfg with authors }
  | "description" -> { cfg with description = Some value }
  | "output" -> { cfg with output = Some value }
  | "edit-url" -> { cfg with edit_url = Some value }
  | _ -> cfg

(* TOC parser: builds a tree from indented lines.

   We collect items at each indent level. When indentation increases, new items
   become children of the previous notebook. When it decreases, we close the
   current group. *)

type toc_entry =
  | E_notebook of string * string * int (* title, path, indent *)
  | E_section of string
  | E_separator

let collect_toc_entries lines =
  let entries = ref [] in
  List.iter
    (fun line ->
      if is_comment_or_blank line then ()
      else if is_separator line then entries := E_separator :: !entries
      else if is_section line then
        entries := E_section (parse_section line) :: !entries
      else
        let indent = leading_spaces line in
        match parse_kv line with
        | Some (title, path) ->
            entries := E_notebook (title, path, indent) :: !entries
        | None -> ())
    lines;
  List.rev !entries

let rec build_toc entries =
  match entries with
  | [] -> ([], [])
  | entry :: rest -> (
      match entry with
      | E_separator ->
          let siblings, remaining = build_toc rest in
          (Separator :: siblings, remaining)
      | E_section title ->
          let siblings, remaining = build_toc rest in
          (Section title :: siblings, remaining)
      | E_notebook (title, path, indent) ->
          let children, after_children = collect_children (indent + 1) rest in
          let nb = { title; path } in
          let siblings, remaining = build_toc after_children in
          (Notebook (nb, children) :: siblings, remaining))

and collect_children min_indent entries =
  match entries with
  | E_notebook (_, _, indent) :: _ when indent >= min_indent ->
      let item, rest = take_one_child min_indent entries in
      let more_children, remaining = collect_children min_indent rest in
      (item :: more_children, remaining)
  | _ -> ([], entries)

and take_one_child min_indent entries =
  match entries with
  | E_notebook (title, path, indent) :: rest when indent >= min_indent ->
      let children, remaining = collect_children (indent + 1) rest in
      let nb = { title; path } in
      (Notebook (nb, children), remaining)
  | _ -> failwith "take_one_child: expected notebook entry"

let parse_config source =
  let lines = String.split_on_char '\n' source in
  (* Split into metadata lines and TOC lines *)
  let in_metadata = ref true in
  let meta_lines = ref [] in
  let toc_lines = ref [] in
  List.iter
    (fun line ->
      if !in_metadata then
        if is_comment_or_blank line then ()
        else if
          is_toc_entry (String.trim line) && not (is_comment_or_blank line)
        then (
          (* Check if this is a key=value that looks like metadata or TOC *)
          match parse_kv line with
          | Some (_, _) when (not (is_section line)) && leading_spaces line = 0
            ->
              (* Could be metadata or a TOC entry. Heuristic: if the value looks
                 like a file path (contains . or /), it's TOC *)
              let trimmed = trim line in
              let value =
                match String.index_opt trimmed '=' with
                | Some i ->
                    trim
                      (String.sub trimmed (i + 1)
                         (String.length trimmed - i - 1))
                | None -> ""
              in
              if
                String.contains value '/'
                || String.contains value '.'
                   && String.length value > 0
                   && value <> ""
              then (
                in_metadata := false;
                toc_lines := line :: !toc_lines)
              else if value = "" then (
                (* Empty value at indent 0: could be a placeholder TOC entry or
                   a metadata key with no value. If we haven't seen any TOC
                   entries yet, check if the key is a known metadata key *)
                let key =
                  match String.index_opt trimmed '=' with
                  | Some i -> trim (String.sub trimmed 0 i)
                  | None -> trimmed
                in
                match key with
                | "title" | "authors" | "description" | "output" | "edit-url" ->
                    meta_lines := line :: !meta_lines
                | _ ->
                    in_metadata := false;
                    toc_lines := line :: !toc_lines)
              else meta_lines := line :: !meta_lines
          | _ ->
              in_metadata := false;
              toc_lines := line :: !toc_lines)
        else meta_lines := line :: !meta_lines
      else toc_lines := line :: !toc_lines)
    lines;
  let config =
    List.fold_left
      (fun cfg line ->
        match parse_kv line with
        | Some (key, value) -> parse_metadata cfg key value
        | None -> cfg)
      default_config (List.rev !meta_lines)
  in
  let toc_entries = collect_toc_entries (List.rev !toc_lines) in
  let toc, _ = build_toc toc_entries in
  Ok (config, toc)

(* ───── Title from filename ───── *)

let title_of_filename path =
  let base = Filename.basename path in
  let name = Filename.remove_extension base in
  (* Strip leading digits and separators *)
  let len = String.length name in
  let start = ref 0 in
  while
    !start < len
    &&
    let c = name.[!start] in
    (c >= '0' && c <= '9') || c = '-' || c = '_'
  do
    incr start
  done;
  let name =
    if !start >= len then name else String.sub name !start (len - !start)
  in
  (* Replace dashes and underscores with spaces *)
  let buf = Buffer.create (String.length name) in
  String.iter
    (fun c ->
      match c with
      | '-' | '_' -> Buffer.add_char buf ' '
      | c -> Buffer.add_char buf c)
    name;
  let result = Buffer.contents buf in
  (* Capitalize first letter *)
  if String.length result > 0 then
    let first = Char.uppercase_ascii result.[0] in
    let rest = String.sub result 1 (String.length result - 1) in
    String.make 1 first ^ rest
  else result

(* ───── Queries ───── *)

let rec all_notebooks toc =
  List.concat_map
    (fun item ->
      match item with
      | Notebook (nb, children) -> nb :: all_notebooks children
      | Section _ | Separator -> [])
    toc

let is_placeholder nb = nb.path = ""

let notebooks project =
  List.filter (fun nb -> not (is_placeholder nb)) (all_notebooks project.toc)

let notebooks_array project = Array.of_list (notebooks project)

let find_notebook_index project nb =
  let nbs = notebooks_array project in
  let rec loop i =
    if i >= Array.length nbs then None
    else if nbs.(i).path = nb.path then Some i
    else loop (i + 1)
  in
  loop 0

let prev_notebook project nb =
  match find_notebook_index project nb with
  | Some i when i > 0 -> Some (notebooks_array project).(i - 1)
  | _ -> None

let next_notebook project nb =
  let nbs = notebooks_array project in
  match find_notebook_index project nb with
  | Some i when i < Array.length nbs - 1 -> Some nbs.(i + 1)
  | _ -> None

let number toc nb =
  let rec search counter = function
    | [] -> None
    | Notebook (n, children) :: rest ->
        incr counter;
        if n.path = nb.path then Some [ !counter ]
        else
          begin match search (ref 0) children with
          | Some sub -> Some (!counter :: sub)
          | None -> search counter rest
          end
    | Section _ :: rest ->
        counter := 0;
        search counter rest
    | Separator :: rest -> search counter rest
  in
  match search (ref 0) toc with Some ns -> ns | None -> []

let number_string = function
  | [] -> ""
  | ns -> String.concat "." (List.map string_of_int ns)
