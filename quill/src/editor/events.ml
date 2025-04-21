open Js_of_ocaml

let handle_selectionchange mounted_app _ev =
  let sel = Dom_html.window##getSelection in
  if not (Js.to_bool sel##.isCollapsed) then Js._false
  else
    match Dom_utils.get_current_range () with
    | None -> Js._false
    | Some range -> (
        let node = range##.startContainer in
        match Dom_utils.map_node_to_indices node with
        | None -> Js._false
        | Some (block_idx, inline_idx) ->
            let msg = Update.Focus_inline (block_idx, inline_idx) in
            Vdom_blit.process mounted_app msg;
            Js._false)

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

let rec block_of_element (el : Dom_html.element Js.t) =
  let open Model in
  let tag = Js.to_string el##.tagName in
  let id_s = Js.to_string el##.id in
  match String.split_on_char '-' id_s with
  | [ "block"; id_str ] -> (
      let id = int_of_string id_str in
      let visible_text = Js.to_string el##.innerText in
      let text =
        Js.Opt.get el##.textContent (fun () -> Js.string "") |> Js.to_string
      in
      match tag with
      | "P" ->
          if String.trim visible_text = "" then
            Some { id; content = Blank_line (); focused = false }
          else
            let inline = Model.inline_of_md text in
            Some { id; content = Paragraph inline; focused = false }
      | "H1" | "H2" | "H3" | "H4" | "H5" | "H6" ->
          let level = int_of_string (String.sub tag 1 1) in
          let inline = Model.inline_of_md visible_text in
          Some { id; content = Heading (level, inline); focused = false }
      | "PRE" ->
          let code = Js.to_string el##.innerText in
          Some { id; content = Codeblock code; focused = false }
      | "DIV" ->
          let children = nodeList_to_list el##.childNodes in
          let blocks =
            List.filter_map
              (fun node ->
                match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
                | Some child_el -> block_of_element child_el
                | None -> None)
              children
          in
          Some { id; content = Blocks blocks; focused = false }
      | _ -> None)
  | _ -> None

let document_of_dom root =
  let editor_div = root##querySelector (Js.string "#editor") in
  let children = nodeList_to_list editor_div##.childNodes in
  Model.next_id := 0;
  let blocks =
    List.filter_map
      (fun node ->
        match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
        | Some el -> block_of_element el
        | None -> None)
      children
  in
  blocks

let handle_input mounted_app dom_el _ev =
  let before = Dom_utils.save_cursor () in
  let msg = Update.Set_document (document_of_dom (Obj.magic dom_el)) in
  Vdom_blit.process mounted_app msg;
  Vdom_blit.after_redraw mounted_app (fun () -> Dom_utils.restore_cursor before);
  Js._false

let setup_event_listeners dom_el mounted_app =
  let _ =
    Dom_html.addEventListener dom_el (Dom.Event.make "input")
      (Dom_html.handler (handle_input mounted_app dom_el))
      Js._false
  in

  let _ =
    Dom_html.addEventListener Dom_html.document
      (Dom.Event.make "selectionchange")
      (Dom_html.handler (handle_selectionchange mounted_app))
      Js._false
  in
  ()
