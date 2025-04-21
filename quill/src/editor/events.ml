open Js_of_ocaml

let handle_selectionchange mounted_app _ev =
  let sel = Dom_html.window##getSelection in
  if not (Js.to_bool sel##.isCollapsed) then Js._false
  else
    match Dom_utils.get_current_range () with
    | None -> Js._false
    | Some range -> (
        let node = range##.startContainer in
        match Dom_utils.map_node_to_inline_id node with
        | None -> Js._false
        | Some inline_id ->
            let msg = Update.Focus_inline_by_id inline_id in
            Vdom_blit.process mounted_app msg;
            Js._false)

let handle_input mounted_app dom_el _ev =
  let pos = Dom_utils.save_cursor () in
  let range = Dom_utils.get_current_range () in
  let new_document = Model_dom.parse_dom (Obj.magic dom_el) range in
  let msg = Update.Set_document new_document in
  Vdom_blit.process mounted_app msg;
  Vdom_blit.after_redraw mounted_app (fun () -> Dom_utils.restore_cursor pos);
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
