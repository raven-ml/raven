open Js_of_ocaml

let log fmt =
  Printf.ksprintf
    (fun s ->
      Js_of_ocaml.Console.console##log (Js_of_ocaml.Js.string ("[events] " ^ s)))
    fmt

let send_execution_request mounted_app block_id code =
  let xhr = XmlHttpRequest.create () in
  xhr##_open (Js.string "POST") (Js.string "/api/execute") Js._true;
  xhr##setRequestHeader (Js.string "Content-Type")
    (Js.string "application/json");
  xhr##.responseType := Js.string "json";
  let data = Js.Unsafe.obj [| ("code", Js.Unsafe.inject (Js.string code)) |] in
  xhr##.onreadystatechange :=
    Js.wrap_callback (fun () ->
        if xhr##.readyState = XmlHttpRequest.DONE then
          if xhr##.status = 200 then
            let response = Js.Unsafe.get xhr "response" in
            let status = Js.to_string (Js.Unsafe.get response "status") in
            let result_str =
              if status <> "success" then
                Js.to_string (Js.Unsafe.get response "error")
              else Js.to_string (Js.Unsafe.get response "output")
            in
            Vdom_blit.process mounted_app
              (Update.Set_codeblock_output (block_id, result_str))
          else log "Execution request failed with status: %d" xhr##.status);
  let json_string = Js._JSON##stringify data in
  xhr##send (Js.some json_string)

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

let handle_keydown mounted_app dom_el ev =
  let key = Js.Optdef.case ev##.key (fun () -> "") Js.to_string in
  let meta_key = Js.to_bool (Js.Unsafe.get ev (Js.string "metaKey")) in
  let ctrl_key = Js.to_bool (Js.Unsafe.get ev (Js.string "ctrlKey")) in
  if key = "Enter" && (meta_key || ctrl_key) then
    match Dom_utils.get_current_range () with
    | None ->
        log "handle_keydown: No range found";
        Js._false
    | Some range -> (
        if not (Js.to_bool range##.collapsed) then (
          log "handle_keydown: Range is not collapsed";
          Js._false)
        else
          (* Prevent default behavior *)
          match Dom_utils.find_codeblock range##.startContainer with
          | None ->
              log "handle_keydown: Not in a code block";
              Js._false
          | Some codeblock_id ->
              let block_id =
                int_of_string
                  (String.sub codeblock_id 10 (String.length codeblock_id - 10))
              in
              let code_el =
                Dom_html.getElementById ("codeblock-" ^ string_of_int block_id)
              in
              let code = Js.to_string code_el##.innerText in
              send_execution_request mounted_app block_id code;
              Js._false)
  else if key = "Enter" then
    match Dom_utils.get_current_range () with
    | None ->
        log "handle_keydown: No range found";
        Js._false
    | Some range -> (
        if not (Js.to_bool range##.collapsed) then (
          log "handle_keydown: Range is not collapsed";
          Js._false (* Text selected *))
        else
          let container = range##.startContainer in
          let block_el_opt =
            let rec find_block node =
              if node = (dom_el :> Dom.node Js.t) then None
              else
                match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
                | Some el
                  when Dom_utils.starts_with (Js.to_string el##.id) "block-" ->
                    Some el
                | _ -> (
                    match Js.Opt.to_option node##.parentNode with
                    | Some parent -> find_block parent
                    | None -> None)
            in
            find_block container
          in
          match block_el_opt with
          | None ->
              log "handle_keydown: No block element found";
              Js._false
          | Some block_el ->
              let tag = Js.to_string block_el##.tagName in
              if tag = "P" then (
                match Dom_utils.map_node_to_inline_id container with
                | None ->
                    log "handle_keydown: No inline ID found";
                    Js._false
                | Some run_id ->
                    let offset = range##.startOffset in
                    let block_id =
                      int_of_string
                        (String.sub
                           (Js.to_string block_el##.id)
                           6
                           (String.length (Js.to_string block_el##.id) - 6))
                    in
                    let parent =
                      Js.Opt.get block_el##.parentNode (fun () -> assert false)
                    in
                    let children =
                      Dom_utils.nodeList_to_list parent##.childNodes
                    in
                    let rec find_index node lst i =
                      match lst with
                      | [] -> -1
                      | hd :: tl ->
                          if hd = (block_el :> Dom.node Js.t) then i
                          else find_index node tl (i + 1)
                    in
                    let index = find_index block_el children 0 in
                    Vdom_blit.process mounted_app
                      (Update.Split_block (block_id, run_id, offset));
                    Vdom_blit.after_redraw mounted_app (fun () ->
                        let children = parent##.childNodes in
                        if index + 1 < children##.length then
                          match
                            Js.Opt.to_option (children##item (index + 1))
                          with
                          | Some next_el ->
                              let range = Dom_html.document##createRange in
                              range##setStart next_el 0;
                              range##collapse Js._true;
                              let sel = Dom_html.window##getSelection in
                              sel##removeAllRanges;
                              sel##addRange range
                          | None -> ()
                        else ());
                    Js._false (* Prevent default behavior *))
              else if tag = "PRE" then
                let code_el =
                  block_el##querySelector (Js.string "code[id^='codeblock-']")
                in
                match Js.Opt.to_option code_el with
                | None ->
                    log "handle_keydown: No code element found";
                    Js._false
                | Some code_el ->
                    let content = Dom_utils.get_inner_text code_el in
                    if content = "```" then (
                      let block_id =
                        int_of_string
                          (String.sub
                             (Js.to_string block_el##.id)
                             6
                             (String.length (Js.to_string block_el##.id) - 6))
                      in
                      let new_block_id = !Model.next_block_id_ref in
                      Vdom_blit.process mounted_app
                        (Update.Replace_block_codeblock block_id);
                      Vdom_blit.after_redraw mounted_app (fun () ->
                          let new_paragraph_id =
                            "block-" ^ string_of_int new_block_id
                          in
                          match
                            Dom_html.getElementById_opt new_paragraph_id
                          with
                          | None -> ()
                          | Some new_paragraph_el ->
                              let range = Dom_html.document##createRange in
                              range##setStart
                                (new_paragraph_el :> Dom.node Js.t)
                                0;
                              range##collapse Js._false;
                              let sel = Dom_html.window##getSelection in
                              sel##removeAllRanges;
                              sel##addRange range);
                      Js._false (* Prevent default behavior *))
                    else Js._false (* Let browser add newline in code block *)
              else Js._false)
  else Js._true (* Non-Enter keys *)

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

  let _ =
    Dom_html.addEventListener dom_el Dom_html.Event.keydown
      (Dom_html.handler (handle_keydown mounted_app dom_el))
      Js._false
  in

  ()
