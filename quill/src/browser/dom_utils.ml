open Brr
open Brr_ext

let log fmt =
  Printf.ksprintf (fun s -> Console.(log [ Jstr.v ("[dom_utils] " ^ s) ])) fmt

let node_list_to_list (nl : Jv.t) : El.t list =
  let len = Jv.Int.get nl "length" in
  let rec loop i acc =
    if i >= len then List.rev acc
    else
      let item = Jv.call nl "item" [| Jv.of_int i |] in
      if Jv.is_null item then loop (i + 1) acc
      else loop (i + 1) (El.of_jv item :: acc)
  in
  loop 0 []

let get_element_inline_id (el : El.t) : int option =
  let id_str = El.prop El.Prop.id el |> Jstr.to_string in
  match String.split_on_char '-' id_str with
  | [ _type; id_num ] -> int_of_string_opt id_num
  | _ -> None

let get_current_range () : Range.t option =
  match Window.get_selection G.window with
  | None -> None
  | Some sel ->
      if Selection.range_count sel > 0 then Some (Selection.get_range_at sel 0)
      else None

let parse_id_from_string ~(prefix : string) (id_str : string) : int option =
  if String.starts_with ~prefix id_str then
    let id_part =
      String.sub id_str (String.length prefix)
        (String.length id_str - String.length prefix)
    in
    try Some (int_of_string id_part) with Failure _ -> None
  else None

let rec find_ancestor (predicate : El.t -> bool) ?(stop_at : El.t option)
    (node : El.t) : El.t option =
  let check_stop current =
    match stop_at with
    | Some stop_el when Jv.equal (El.to_jv current) (El.to_jv stop_el) -> true
    | _ -> false
  in
  if check_stop node then None
  else if predicate node then Some node
  else
    match El.parent node with
    | Some parent -> find_ancestor predicate ?stop_at parent
    | None -> None

let get_element_id_with_prefix (prefix : string) (el : El.t) : int option =
  El.prop El.Prop.id el |> Jstr.to_string |> parse_id_from_string ~prefix

let get_element_block_id el = get_element_id_with_prefix "block-" el
let get_element_run_id el = get_element_id_with_prefix "run-" el
let get_element_codeblock_id el = get_element_id_with_prefix "codeblock-" el

let has_prefix_id prefix el =
  get_element_id_with_prefix prefix el |> Option.is_some

let find_block_parent ?stop_at node =
  find_ancestor (has_prefix_id "block-") ?stop_at node

let find_run_ancestor node = find_ancestor (has_prefix_id "run-") node

let get_element_tag node =
  El.tag_name node |> Jstr.to_string |> String.uppercase_ascii

let find_codeblock_ancestor node =
  find_ancestor
    (fun el -> get_element_tag el = "CODE" && has_prefix_id "codeblock-" el)
    node

let has_class (el : El.t) class_name : bool =
  let class_list = El.prop (El.Prop.jstr (Jstr.of_string "class")) el in
  Jstr.find_sub ~sub:(Jstr.v class_name) class_list |> Option.is_some

let is_inline_id id_str =
  List.exists
    (fun prefix -> String.starts_with ~prefix id_str)
    [ "run-"; "codespan-"; "emph-"; "strong-"; "seq-" ]

let find_inline_ancestor node =
  find_ancestor
    (fun el ->
      let id_str = El.prop El.Prop.id el |> Jstr.to_string in
      is_inline_id id_str || has_class el "inline-text")
    node

let contains_range_start (range : Range.t) (node : El.t) : bool =
  let container_jv = Range.start_container range in
  let target_jv = El.to_jv node in
  let rec is_descendant current_jv =
    if Jv.equal current_jv target_jv then true
    else
      match Jv.find current_jv "parentNode" with
      | Some parent_jv
        when not (Jv.is_null parent_jv || Jv.is_undefined parent_jv) ->
          is_descendant parent_jv
      | _ -> false
  in
  if Jv.is_null container_jv || Jv.is_undefined container_jv then false
  else is_descendant container_jv

let get_node_value node : string =
  Jv.find (El.to_jv node) "nodeValue"
  |> Option.map Jv.to_jstr
  |> Option.value ~default:Jstr.empty
  |> Jstr.to_string

let get_text_content node : string =
  if Jv.get (El.to_jv node) "nodeType" |> Jv.to_int = 3 then get_node_value node
  else if El.is_el node then El.text_content node |> Jstr.to_string
  else ""

let inner_text node : string =
  if El.is_el node then Brr_ext.El.inner_text node |> Jstr.to_string else ""

let is_span_with_text node text =
  if El.is_el node then get_element_tag node = "SPAN" && inner_text node = text
  else false

let rec find_first_text_node (node : El.t) : El.t option =
  if El.is_txt node then Some node
  else
    let children = El.children node in
    let rec search = function
      | [] -> None
      | child :: rest -> (
          match find_first_text_node child with
          | Some text -> Some text
          | None -> search rest)
    in
    search children

let find_next_block_element (el : El.t) : El.t option =
  let rec find_next sibling_opt =
    match sibling_opt with
    | None -> None
    | Some sibling ->
        if El.is_el sibling && get_element_block_id sibling |> Option.is_some
        then Some sibling
        else find_next (El.next_sibling sibling)
  in
  find_next (El.next_sibling el)

let focus_element_start (el : El.t) : unit =
  match El.find_first_by_selector ~root:el (Jstr.v ".inline-text") with
  | Some text_node -> (
      match Window.get_selection G.window with
      | Some sel ->
          let range = Document.create_range G.document in
          Range.set_start range (El.to_jv text_node) 0;
          Range.collapse range true;
          Selection.remove_all_ranges sel;
          Selection.add_range sel range;
          El.scroll_into_view el
      | None -> log "focus_element_start: No selection object")
  | None -> log "focus_element_start: No .inline-text node found in element"

let get_caret_offset_within (element : El.t) : int =
  match Window.get_selection G.window with
  | None -> 0
  | Some sel ->
      if Selection.range_count sel > 0 then (
        let range = Selection.get_range_at sel 0 in
        let pre_caret_range = Range.clone range in
        Range.select_node_contents pre_caret_range (El.to_jv element);
        Range.set_end pre_caret_range
          (Range.end_container range)
          (Range.end_offset range);
        let text = Range.to_string pre_caret_range in
        Jstr.length text)
      else 0

let get_selection_offsets_within (element : El.t) : int * int =
  match Window.get_selection G.window with
  | None -> (0, 0)
  | Some sel ->
      if Selection.range_count sel > 0 then (
        let range = Selection.get_range_at sel 0 in
        (* Calculate start offset *)
        let start_range = Document.create_range G.document in
        Range.select_node_contents start_range (El.to_jv element);
        Range.set_end start_range
          (Range.start_container range)
          (Range.start_offset range);
        let start_offset = Jstr.length (Range.to_string start_range) in
        (* Calculate end offset *)
        let end_range = Document.create_range G.document in
        Range.select_node_contents end_range (El.to_jv element);
        Range.set_end end_range
          (Range.end_container range)
          (Range.end_offset range);
        let end_offset = Jstr.length (Range.to_string end_range) in
        (start_offset, end_offset))
      else (0, 0)

(* Type to represent the result of DOM traversal *)
type traverse_result =
  | Found of { node : El.t; position : int }
  | Not_found of int

(* Recursively traverse the DOM to find the text node at a given offset *)
let rec traverse (node : El.t) (offset : int) : traverse_result =
  if El.is_txt node then
    let text = El.text_content node in
    let length = Jstr.length text in
    if offset <= length then Found { node; position = offset }
    else Not_found length
  else
    let child_nodes = Jv.get (El.to_jv node) "childNodes" in
    let children = node_list_to_list child_nodes in
    let rec loop cumulative children =
      match children with
      | [] -> Not_found cumulative
      | child :: rest -> (
          match traverse child (offset - cumulative) with
          | Found res -> Found res
          | Not_found len -> loop (cumulative + len) rest)
    in
    loop 0 children

(* Find the text node and position for a given character index *)
let get_text_node_at_position (root : El.t) (index : int) : El.t * int =
  match traverse root index with
  | Found { node; position } -> (node, position)
  | Not_found _ -> (root, index)

let set_caret_offset_within (context : El.t) (offset : int) : unit =
  match Window.get_selection G.window with
  | None -> ()
  | Some sel ->
      log "Restoring caret position: %d" offset;
      let node, position = get_text_node_at_position context offset in
      let new_range = Document.create_range G.document in
      Range.set_start new_range (El.to_jv node) position;
      Range.set_end new_range (El.to_jv node) position;
      Selection.remove_all_ranges sel;
      Selection.add_range sel new_range
