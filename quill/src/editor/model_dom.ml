open Model
open Js_of_ocaml

let nodeList_to_list nl =
  let len = nl##.length in
  let rec go i acc =
    if i >= len then List.rev acc
    else
      let acc' =
        match nl##item i |> Js.Opt.to_option with
        | Some node -> node :: acc
        | None -> acc
      in
      go (i + 1) acc'
  in
  go 0 []

let is_span_with_text node text =
  match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
  | Some el when Js.to_string el##.tagName = "SPAN" ->
      Js.to_string el##.innerText = text
  | _ -> false

let get_element_tag node =
  match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
  | Some el -> Some (Js.to_string el##.tagName)
  | None -> None

let get_inner_text node =
  match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
  | Some el -> Js.to_string el##.innerText
  | None -> ""

let parse_run span_el range : inline =
  let children = nodeList_to_list span_el##.childNodes in
  let content, focused =
    match children with
    | [ node ] when node##.nodeType = Dom.TEXT ->
        let text =
          Js.Opt.case node##.nodeValue
            (fun _ -> "")
            (fun node_value -> Js.to_string node_value)
        in
        let is_focused =
          match range with
          | Some r -> Dom_utils.is_within_range r node
          | None -> false
        in
        (Run text, is_focused)
    | [ span1; middle; span2 ] ->
        if is_span_with_text span1 "*" && is_span_with_text span2 "*" then
          match get_element_tag middle with
          | Some "EM" ->
              let inner_text = get_inner_text middle in
              let is_focused =
                match range with
                | Some r -> Dom_utils.is_within_range r middle
                | None -> false
              in
              (Emph (Run inner_text), is_focused)
          | _ -> (Run (Js.to_string span_el##.innerText), false)
        else if is_span_with_text span1 "**" && is_span_with_text span2 "**"
        then
          match get_element_tag middle with
          | Some "STRONG" ->
              let inner_text = get_inner_text middle in
              let is_focused =
                match range with
                | Some r -> Dom_utils.is_within_range r middle
                | None -> false
              in
              (Strong (Run inner_text), is_focused)
          | _ -> (Run (Js.to_string span_el##.innerText), false)
        else (Run (Js.to_string span_el##.innerText), false)
    | _ ->
        let text = Js.to_string span_el##.innerText in
        let is_focused =
          match range with
          | Some r -> Dom_utils.is_within_range r (span_el :> Dom.node Js.t)
          | None -> false
        in
        (Run text, is_focused)
  in
  incr next_id;
  { id = !next_id; content; focused }

let parse_inlines (el : Dom_html.element Js.t)
    (range : Dom_html.range Js.t option) : inline =
  let run_spans =
    let children = nodeList_to_list el##.childNodes in
    List.filter_map
      (fun node ->
        match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
        | Some span_el
          when Js.to_string span_el##.tagName = "SPAN"
               && not (Js.to_bool (span_el##hasAttribute (Js.string "hidden")))
          ->
            Some span_el
        | _ -> None)
      children
  in
  let runs = List.map (fun span_el -> parse_run span_el range) run_spans in
  incr next_id;
  { id = !next_id; content = Seq runs; focused = false }

let rec parse_block el range =
  let tag = Js.to_string el##.tagName in
  let id_s = Js.to_string el##.id in
  match String.split_on_char '-' id_s with
  | [ "block"; id_str ] -> (
      let id = int_of_string id_str in
      let visible_text = Js.to_string el##.innerText in
      let focused =
        match range with
        | Some r -> Dom_utils.is_within_range r (el :> Dom.node Js.t)
        | None -> false
      in
      match tag with
      | "P" ->
          if String.trim visible_text = "" then
            Some { id; content = Blank_line (); focused }
          else
            let inline = parse_inlines el range in
            Some { id; content = Paragraph inline; focused }
      | "H1" | "H2" | "H3" | "H4" | "H5" | "H6" ->
          let level = int_of_string (String.sub tag 1 1) in
          let inline = parse_inlines el range in
          Some { id; content = Heading (level, inline); focused }
      | "PRE" ->
          let code = Js.to_string el##.innerText in
          Some { id; content = Codeblock code; focused }
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
          Some { id; content = Blocks blocks; focused }
      | _ -> None)
  | _ -> None

let parse_dom root range =
  let editor_div = root##querySelector (Js.string "#editor") in
  let children = nodeList_to_list editor_div##.childNodes in
  Model.next_id := 0;
  let blocks =
    List.filter_map
      (fun node ->
        match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
        | Some el -> parse_block el range
        | None -> None)
      children
  in
  blocks
