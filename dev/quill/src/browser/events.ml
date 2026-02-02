(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Brr
open Brr_ext
open Dom_utils

let log fmt =
  Printf.ksprintf (fun s -> Console.(log [ Jstr.v ("[events] " ^ s) ])) fmt

(* ───── Selection Handling ───── *)

let handle_selectionchange mounted_app dom_el (_ev : Ev.Type.void Ev.t) =
  match Window.get_selection G.window with
  | None -> log "selectionchange: No selection object"
  | Some sel ->
      if Selection.is_collapsed sel then
        match El.find_first_by_selector (Jstr.v "#editor") ~root:dom_el with
        | None -> log "selectionchange: Could not find editor element"
        | Some editor_div -> (
            let offset = get_caret_offset_within editor_div in
            (* Parse current DOM to get the model *)
            let model = Vdom_blit.get mounted_app in
            match Model_dom.find_in_blocks model.Model.document offset with
            | Some ("inline", id) ->
                log "selectionchange: Focusing inline %d" id;
                Vdom_blit.process mounted_app (Update.Focus_inline_by_id id)
            | Some ("block", id) ->
                log "selectionchange: Focusing block %d" id;
                Vdom_blit.process mounted_app (Update.Focus_block_by_id id)
            | Some _ ->
                log
                  "selectionchange: Found non-inline/block element at offset %d"
                  offset
            | None ->
                log "selectionchange: No element found at offset %d" offset)
      else log "selectionchange: Selection is not collapsed"

(* ───── Input Handling ───── *)

let handle_input mounted_app (dom_el : El.t) (_ev : Ev.Input.t Ev.t) =
  let new_document = Model_dom.parse_dom dom_el in
  let msg = Update.Set_document new_document in
  let offset = get_caret_offset_within dom_el in
  Vdom_blit.process mounted_app msg;
  Vdom_blit.after_redraw mounted_app (fun () ->
      set_caret_offset_within dom_el offset)

(* ───── Remove Handling ───── *)

let handle_remove ~is_backspace mounted_app dom_el =
  let editor_div =
    match El.find_first_by_selector (Jstr.v "#editor") ~root:dom_el with
    | Some ed -> ed
    | None -> failwith "Could not find editor element"
  in
  let model = Vdom_blit.get mounted_app in
  let md = Quill_editor.Document.to_markdown model.Model.document in
  let start_offset, end_offset = get_selection_offsets_within editor_div in
  let new_md, new_offset =
    if start_offset = end_offset then (* Collapsed selection *)
      if is_backspace then (
        log "handle_remove: Backspace pressed";
        if start_offset > 0 then
          let new_md =
            String.sub md 0 (start_offset - 1)
            ^ String.sub md start_offset (String.length md - start_offset)
          in
          (new_md, start_offset - 1)
        else (md, start_offset))
      else if
        (* Delete *)
        log "handle_remove: Delete pressed";
        start_offset < String.length md
      then
        let new_md =
          String.sub md 0 start_offset
          ^ String.sub md (start_offset + 1)
              (String.length md - start_offset - 1)
        in
        (new_md, start_offset)
      else (md, start_offset)
    else (
      (* Non-collapsed selection *)
      log "handle_remove: Non-collapsed selection";
      let new_md =
        String.sub md 0 start_offset
        ^ String.sub md end_offset (String.length md - end_offset)
      in
      (new_md, start_offset))
  in
  if new_md <> md then (
    let new_document = Quill_editor.Document.of_markdown new_md in
    Vdom_blit.process mounted_app (Update.Set_document new_document);
    Vdom_blit.after_redraw mounted_app (fun () ->
        set_caret_offset_within editor_div new_offset));
  true (* Prevent default behavior *)

(* ───── Keydown Handling ───── *)

let handle_execute_code ~code_execution_handler =
  let range_opt = get_current_range () in
  let start_node_opt =
    Option.map (fun range -> El.of_jv (Range.start_container range)) range_opt
  in
  let codeblock_opt = Option.bind start_node_opt find_codeblock_ancestor in
  match codeblock_opt with
  | None ->
      log "keydown (exec): Not inside a codeblock element";
      false
  | Some code_el -> (
      match get_element_codeblock_id code_el with
      | None ->
          log "keydown (exec): Could not parse codeblock ID";
          false
      | Some block_id ->
          let code = inner_text code_el in
          log "keydown (exec): Performing effect for block %d" block_id;
          code_execution_handler block_id code;
          true)

let handle_enter mounted_app dom_el =
  match El.find_first_by_selector (Jstr.v "#editor") ~root:dom_el with
  | None ->
      log "Could not find editor element";
      false
  | Some editor_div ->
      let start_offset, end_offset = get_selection_offsets_within editor_div in
      let md = Jstr.to_string (El.text_content editor_div) in
      let inserted_text = "\n\n" in
      let new_md =
        if start_offset = end_offset then
          (* Collapsed selection: Insert two newlines at caret position *)
          String.sub md 0 start_offset
          ^ inserted_text
          ^ String.sub md start_offset (String.length md - start_offset)
        else
          (* Non-collapsed selection: Remove selected text and insert two
             newlines *)
          String.sub md 0 start_offset
          ^ inserted_text
          ^ String.sub md end_offset (String.length md - end_offset)
      in
      let new_offset = start_offset + String.length inserted_text in
      let doc = Quill_editor.Document.of_markdown (String.trim new_md) in
      let doc = Quill_editor.Document.normalize_blanklines doc in
      Vdom_blit.process mounted_app (Update.Set_document doc);
      Vdom_blit.after_redraw mounted_app (fun () ->
          set_caret_offset_within editor_div new_offset);
      true

let handle_enter_in_codeblock mounted_app dom_el =
  match El.find_first_by_selector (Jstr.v "#editor") ~root:dom_el with
  | None ->
      log "Could not find editor element";
      false
  | Some editor_div ->
      let start_offset, end_offset = get_selection_offsets_within editor_div in
      let md = Jstr.to_string (El.text_content editor_div) in
      let inserted_text = "\n" in
      let new_md =
        if start_offset = end_offset then
          (* Collapsed selection: Insert one newline at caret position *)
          String.sub md 0 start_offset
          ^ inserted_text
          ^ String.sub md start_offset (String.length md - start_offset)
        else
          (* Non-collapsed selection: Remove selected text and insert one
             newline *)
          String.sub md 0 start_offset
          ^ inserted_text
          ^ String.sub md end_offset (String.length md - end_offset)
      in
      let new_offset = start_offset + String.length inserted_text in
      let doc = Quill_editor.Document.of_markdown (String.trim new_md) in
      Vdom_blit.process mounted_app (Update.Set_document doc);
      Vdom_blit.after_redraw mounted_app (fun () ->
          set_caret_offset_within editor_div new_offset);
      true

let handle_enter_key mounted_app dom_el =
  let range_opt = get_current_range () in
  let start_node_opt =
    Option.map (fun range -> El.of_jv (Range.start_container range)) range_opt
  in
  let parent_opt =
    Option.bind start_node_opt (find_block_parent ~stop_at:dom_el)
  in
  match parent_opt with
  | None ->
      log "keydown (enter): Not inside a known block element.";
      false
  | Some block_el -> (
      let tag = get_element_tag block_el in
      match tag with
      | "PRE" -> handle_enter_in_codeblock mounted_app dom_el
      | _ -> handle_enter mounted_app dom_el)

let handle_keydown ~code_execution_handler mounted_app (dom_el : El.t)
    (ev : Ev.Keyboard.t Ev.t) =
  let evt = Ev.as_type ev in
  let key = Jstr.to_string (Ev.Keyboard.key evt) in
  let meta_key = Ev.Keyboard.meta_key evt in
  let ctrl_key = Ev.Keyboard.ctrl_key evt in
  let should_prevent =
    match key with
    | "Enter" when meta_key || ctrl_key ->
        handle_execute_code ~code_execution_handler
    | "Enter" -> handle_enter_key mounted_app dom_el
    | "Backspace" -> handle_remove ~is_backspace:true mounted_app dom_el
    | "Delete" -> handle_remove ~is_backspace:false mounted_app dom_el
    | _ -> false
  in
  if should_prevent then (
    log "keydown: Preventing default for key '%s'" key;
    Ev.prevent_default ev)
  else log "keydown: Allowing default for key '%s'" key

let setup_event_listeners ~code_execution_handler (dom_el : El.t) mounted_app =
  let _input_listener =
    Ev.listen Ev.input
      (fun ev -> handle_input mounted_app dom_el ev)
      (El.as_target dom_el)
  in
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
  ()
