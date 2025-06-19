open Brr
open Brr_ext
open Dom_utils

let log fmt =
  Printf.ksprintf (fun s -> Console.(log [ Jstr.v ("[events] " ^ s) ])) fmt

(* Selection Handling *)

(* Helper to find block parent with ID *)
let find_block_parent_with_id ~stop_at el =
  let rec find el =
    if Jv.equal (El.to_jv el) (El.to_jv stop_at) then None
    else
      match get_element_block_id el with
      | Some id -> Some (el, id)
      | None -> 
          match El.parent el with
          | None -> None
          | Some parent -> find parent
  in
  find el

let handle_selectionchange mounted_app dom_el (_ev : Ev.Type.void Ev.t) =
  match Window.get_selection G.window with
  | None -> log "selectionchange: No selection object"
  | Some sel ->
      if Selection.is_collapsed sel then
        match El.find_first_by_selector (Jstr.v "#editor") ~root:dom_el with
        | None -> log "selectionchange: Could not find editor element"
        | Some editor_div -> (
            (* Get the current range and find the containing element *)
            if Selection.range_count sel > 0 then
              let range = Selection.get_range_at sel 0 in
                let start_container = Range.start_container range in
                let start_offset = Range.start_offset range in
                (* Find the containing block element *)
                match find_block_parent_with_id ~stop_at:editor_div (El.of_jv start_container) with
                | None -> log "selectionchange: No block parent found"
                | Some (_block_el, block_id) ->
                    log "selectionchange: Found block %d at offset %d" block_id start_offset;
                    (* Calculate offset within the block *)
                    let block_offset = 
                      (* For now, use a simple offset calculation *)
                      start_offset
                    in
                    (* Create a selection position *)
                    let pos = { Quill.View.block_id = Quill.Document.block_id_of_int block_id; offset = block_offset } in
                    let cmd = Quill.Command.Set_selection (Quill.View.collapsed_at pos) in
                    Vdom_blit.process mounted_app (Update.Execute_command cmd)
            else log "selectionchange: No range found")
      else log "selectionchange: Selection is not collapsed"

(* Input Handling *)

let handle_input mounted_app (dom_el : El.t) (_ev : Ev.Input.t Ev.t) =
  (* For now, we'll parse the DOM and update the document *)
  (* This is a simplified approach - in a real implementation, 
     we would track the actual changes more precisely *)
  let new_document = Model_dom.parse_dom dom_el in
  let markdown = Quill.Markdown.serialize new_document in
  Vdom_blit.process mounted_app (Update.Set_document_markdown markdown)

(* Text insertion handling *)
let handle_text_insertion mounted_app text =
  match Window.get_selection G.window with
  | None -> false
  | Some sel ->
      if not (Selection.is_collapsed sel) then false
      else
        if Selection.range_count sel > 0 then
            let _range = Selection.get_range_at sel 0 in
            let model : Model.t = Vdom_blit.get mounted_app in
            match Quill.Engine.get_view model.engine with
            | { selection = Some sel; _ } ->
                let cmd = Quill.Command.Insert_text (sel.focus.block_id, sel.focus.offset, text) in
                Vdom_blit.process mounted_app (Update.Execute_command cmd);
                true
            | _ -> false
        else false

(* Remove Handling *)
let handle_remove ~is_backspace mounted_app =
  let model : Model.t = Vdom_blit.get mounted_app in
  match Quill.Engine.get_view model.engine with
  | { selection = Some sel; _ } when Quill.View.is_collapsed sel ->
      if is_backspace && sel.focus.offset > 0 then
        let cmd = Quill.Command.Delete_range (sel.focus.block_id, sel.focus.offset - 1, sel.focus.offset) in
        Vdom_blit.process mounted_app (Update.Execute_command cmd);
        true
      else if not is_backspace then
        let cmd = Quill.Command.Delete_range (sel.focus.block_id, sel.focus.offset, sel.focus.offset + 1) in
        Vdom_blit.process mounted_app (Update.Execute_command cmd);
        true
      else false
  | { selection = Some sel; _ } ->
      (* Handle range deletion *)
      let cmd = Quill.Command.Delete_range (sel.anchor.block_id, sel.anchor.offset, sel.focus.offset) in
      Vdom_blit.process mounted_app (Update.Execute_command cmd);
      true
  | _ -> false

(* Keydown Handling *)

let handle_execute_code ~code_execution_handler mounted_app =
  let model : Model.t = Vdom_blit.get mounted_app in
  match Quill.Engine.get_view model.engine with
  | { selection = Some sel; _ } ->
      log "keydown (exec): Executing block %d" (sel.focus.block_id :> int);
      (* Get the block to find its code *)
      (match Quill.Document.find_block (Quill.Engine.get_document model.engine) sel.focus.block_id with
       | Some block ->
           (match block.content with
            | Codeblock { code; _ } ->
                code_execution_handler sel.focus.block_id code;
                true
            | _ -> false)
       | None -> false)
  | _ -> false

let handle_enter mounted_app =
  let model : Model.t = Vdom_blit.get mounted_app in
  match Quill.Engine.get_view model.engine with
  | { selection = Some sel; _ } when Quill.View.is_collapsed sel ->
      let cmd = Quill.Command.Split_block (sel.focus.block_id, sel.focus.offset) in
      Vdom_blit.process mounted_app (Update.Execute_command cmd);
      true
  | _ -> false

let handle_keydown ~code_execution_handler mounted_app (_dom_el : El.t)
    (ev : Ev.Keyboard.t Ev.t) =
  let evt = Ev.as_type ev in
  let key = Jstr.to_string (Ev.Keyboard.key evt) in
  let meta_key = Ev.Keyboard.meta_key evt in
  let ctrl_key = Ev.Keyboard.ctrl_key evt in
  let shift_key = Ev.Keyboard.shift_key evt in
  let should_prevent =
    match key with
    | "Enter" when meta_key || ctrl_key ->
        handle_execute_code ~code_execution_handler mounted_app
    | "Enter" -> handle_enter mounted_app
    | "Backspace" -> handle_remove ~is_backspace:true mounted_app
    | "Delete" -> handle_remove ~is_backspace:false mounted_app
    | "ArrowLeft" ->
        let cmd = Quill.Command.Move_cursor `Left in
        Vdom_blit.process mounted_app (Update.Execute_command cmd);
        true
    | "ArrowRight" ->
        let cmd = Quill.Command.Move_cursor `Right in
        Vdom_blit.process mounted_app (Update.Execute_command cmd);
        true
    | "ArrowUp" ->
        let cmd = Quill.Command.Move_cursor `Up in
        Vdom_blit.process mounted_app (Update.Execute_command cmd);
        true
    | "ArrowDown" ->
        let cmd = Quill.Command.Move_cursor `Down in
        Vdom_blit.process mounted_app (Update.Execute_command cmd);
        true
    | "Home" ->
        let cmd = Quill.Command.Move_cursor `Start in
        Vdom_blit.process mounted_app (Update.Execute_command cmd);
        true
    | "End" ->
        let cmd = Quill.Command.Move_cursor `End in
        Vdom_blit.process mounted_app (Update.Execute_command cmd);
        true
    | "b" when meta_key || ctrl_key ->
        let cmd = Quill.Command.Toggle_inline_style `Bold in
        Vdom_blit.process mounted_app (Update.Execute_command cmd);
        true
    | "i" when meta_key || ctrl_key ->
        let cmd = Quill.Command.Toggle_inline_style `Italic in
        Vdom_blit.process mounted_app (Update.Execute_command cmd);
        true
    | "Tab" when not shift_key ->
        (* Indent *)
        let model = Vdom_blit.get mounted_app in
        (match Quill.Engine.get_view model.engine with
        | { selection = Some sel; _ } ->
            let cmd = Quill.Command.Indent sel.focus.block_id in
            Vdom_blit.process mounted_app (Update.Execute_command cmd);
            true
        | _ -> false)
    | "Tab" when shift_key ->
        (* Outdent *)
        let model = Vdom_blit.get mounted_app in
        (match Quill.Engine.get_view model.engine with
        | { selection = Some sel; _ } ->
            let cmd = Quill.Command.Outdent sel.focus.block_id in
            Vdom_blit.process mounted_app (Update.Execute_command cmd);
            true
        | _ -> false)
    | _ when String.length key = 1 && not (meta_key || ctrl_key) ->
        (* Regular character input *)
        handle_text_insertion mounted_app key
    | _ -> false
  in
  if should_prevent then (
    log "keydown: Preventing default for key '%s'" key;
    Ev.prevent_default ev)
  else log "keydown: Allowing default for key '%s'" key

let setup_event_listeners ~code_execution_handler (dom_el : El.t) mounted_app =
  let _selection_listener =
    Ev.listen Ev.selectionchange
      (fun ev -> handle_selectionchange mounted_app dom_el ev)
      (Document.as_target G.document)
  in
  let _keydown_listener =
    Ev.listen Ev.keydown
      (fun ev -> handle_keydown ~code_execution_handler mounted_app dom_el ev)
      (El.as_target dom_el)
  in
  (* Disable default input event since we handle everything through commands *)
  let _input_listener =
    Ev.listen Ev.input
      (fun ev -> Ev.prevent_default ev)
      (El.as_target dom_el)
  in
  ()