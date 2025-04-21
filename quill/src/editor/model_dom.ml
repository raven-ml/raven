open Model
open Dom_utils
open Js_of_ocaml

let rec parse_inline (el : Dom.node Js.t) (range : Dom_html.range Js.t option) :
    inline =
  let is_focused =
    match range with Some r -> is_within_range r el | None -> false
  in
  match get_element_tag el with
  | Some "SPAN" ->
      let children = nodeList_to_list el##.childNodes in
      let content =
        match children with
        | [ node ] when node##.nodeType = Dom.TEXT ->
            Run (get_text_content node)
        | [ span1; middle; span2 ] ->
            if is_span_with_text span1 "*" && is_span_with_text span2 "*" then
              match get_element_tag middle with
              | Some "EM" -> Emph (parse_inline middle range)
              | _ -> Run (get_text_content el)
            else if is_span_with_text span1 "**" && is_span_with_text span2 "**"
            then
              match get_element_tag middle with
              | Some "STRONG" -> Strong (parse_inline middle range)
              | _ -> Run (get_text_content el)
            else Run (get_text_content el)
        | _ -> Run (get_text_content el)
      in
      incr next_id;
      { id = !next_id; content; focused = is_focused }
  | Some "EM" ->
      let children = nodeList_to_list el##.childNodes in
      let inlines =
        List.filter_map
          (fun node ->
            match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
            | Some child_el when Js.to_string child_el##.tagName = "SPAN" ->
                Some (parse_inline (child_el :> Dom.node Js.t) range)
            | _ -> None)
          children
      in
      let content =
        match inlines with [ single ] -> single.content | _ -> Seq inlines
      in
      incr next_id;
      { id = !next_id; content; focused = is_focused }
  | Some "STRONG" ->
      let children = nodeList_to_list el##.childNodes in
      let inlines =
        List.filter_map
          (fun node ->
            match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
            | Some child_el when Js.to_string child_el##.tagName = "SPAN" ->
                Some (parse_inline (child_el :> Dom.node Js.t) range)
            | _ -> None)
          children
      in
      let content =
        match inlines with [ single ] -> single.content | _ -> Seq inlines
      in
      incr next_id;
      { id = !next_id; content; focused = is_focused }
  | _ ->
      incr next_id;
      {
        id = !next_id;
        content = Run (get_text_content el);
        focused = is_focused;
      }

let parse_run span_el range : inline = parse_inline span_el range

let parse_inlines el range : inline =
  let children = nodeList_to_list el##.childNodes in
  let inlines =
    List.filter_map
      (fun node ->
        match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
        | Some span_el
          when Js.to_string span_el##.tagName = "SPAN"
               && not (Js.to_bool (span_el##hasAttribute (Js.string "hidden")))
          ->
            Some (parse_inline (span_el :> Dom.node Js.t) range)
        | _ -> None)
      children
  in
  incr next_id;
  { id = !next_id; content = Seq inlines; focused = false }

let rec parse_block el range =
  let tag = Js.to_string el##.tagName in
  let id_s = Js.to_string el##.id in
  match String.split_on_char '-' id_s with
  | [ "block"; id_str ] -> (
      let id = int_of_string id_str in
      let visible_text = Js.to_string el##.innerText in
      let focused =
        match range with
        | Some r -> is_within_range r (el :> Dom.node Js.t)
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
