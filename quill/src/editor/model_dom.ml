open Model
open Dom_utils
open Js_of_ocaml

let rec parse_block el range =
  let tag = Js.to_string el##.tagName in
  let id_s = Js.to_string el##.id in
  match String.split_on_char '-' id_s with
  | [ "block"; _ ] -> (
      let visible_text = Js.to_string el##.innerText in
      let focused =
        match range with
        | Some r -> is_within_range r (el :> Dom.node Js.t)
        | None -> false
      in
      match tag with
      | "P" ->
          if String.trim visible_text = "" then
            Some { id = next_block_id (); content = Blank_line (); focused }
          else Some (block_of_md (get_text_content el))
      | "H1" | "H2" | "H3" | "H4" | "H5" | "H6" ->
          Some (block_of_md (get_text_content el))
      | "PRE" ->
          let code = Js.to_string el##.innerText in
          Some
            {
              id = next_block_id ();
              content = Codeblock { code; output = None };
              focused;
            }
      | "DIV" ->
          let children = nodeList_to_list el##.childNodes in
          let blocks =
            List.filter_map
              (fun node ->
                match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
                | Some child_el -> parse_block child_el range
                | None -> None)
              children
          in
          Some { id = next_block_id (); content = Blocks blocks; focused }
      | _ -> None)
  | _ -> None

let parse_dom root range =
  let editor_div = root##querySelector (Js.string "#editor") in
  let children = nodeList_to_list editor_div##.childNodes in
  Model.next_block_id_ref := 0;
  Model.next_run_id_ref := 0;
  let blocks =
    List.filter_map
      (fun node ->
        match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
        | Some el -> parse_block el range
        | None -> None)
      children
  in
  blocks
