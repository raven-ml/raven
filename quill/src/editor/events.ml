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
  let tag = Js.to_string el##.tagName |> String.lowercase_ascii in
  let id_s = Js.to_string el##.id in
  match String.split_on_char '-' id_s with
  | [ "block"; idx ] -> (
      let blk = int_of_string idx in
      match tag with
      | "p" ->
          let txt =
            Js.Opt.get el##.textContent (fun () -> Js.string "") |> Js.to_string
          in
          Some
            Model.
              {
                id = blk;
                content = Paragraph (inline_of_md txt);
                focused = false;
              }
      | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" ->
          let level = int_of_string (String.sub tag 1 1) in
          let txt =
            Js.Opt.get el##.textContent (fun () -> Js.string "") |> Js.to_string
          in
          Some
            {
              id = blk;
              content = Heading (level, Model.inline_of_md txt);
              focused = false;
            }
      | "pre" ->
          let code = el##.innerText |> Js.to_string in
          Some { id = blk; content = Codeblock code; focused = false }
      | "div" ->
          let children =
            nodeList_to_list el##.childNodes
            |> List.filter_map (fun node ->
                   match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
                   | Some el -> block_of_element el
                   | None -> None)
          in
          Some { id = blk; content = Blocks children; focused = false }
      | _ -> None)
  | _ -> None

let document_of_dom root =
  let children = nodeList_to_list root##.childNodes in
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
