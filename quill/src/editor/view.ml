open Update
open Vdom

let span ?key ?a l = elt "span" ?key ?a l
let p ?key ?a l = elt "p" ?key ?a l
let div ?key ?a l = elt "div" ?key ?a l
let tabindex = int_attr "tabindex"

(* Helper functions for inline rendering *)
let rec inline_to_plain (inline : Quill.Document.inline) =
  match inline.content with
  | Run s -> s
  | Emph inner | Strong inner -> inline_to_plain inner
  | Code_span s -> s
  | Seq inlines -> String.concat "" (List.map inline_to_plain inlines)
  | Break `Hard -> "\n"
  | Break `Soft -> " "
  | Image { alt; _ } -> inline_to_plain alt
  | Link { text; _ } -> inline_to_plain text
  | Raw_html _ -> ""

let rec wrap_inline ~focused (inline : Quill.Document.inline) =
  let id = Printf.sprintf "inline-%d" (inline.id :> int) in
  match inline.content with
  | Run s ->
      span ~key:id
        ~a:[ attr "id" id; attr "class" "inline-text" ]
        [ (if s = "" then elt "br" [] else text s) ]
  | Code_span s ->
      elt "code" ~key:id
        ~a:[ attr "id" id ]
        [
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "`" ];
          span
            ~a:[ attr "class" "inline-content inline-text" ]
            [ (if s = "" then elt "br" [] else text s) ];
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "`" ];
        ]
  | Emph inner ->
      elt "em" ~key:id
        ~a:[ attr "id" id ]
        [
          span
            ~a:
              [
                bool_prop "hidden" (not focused);
                bool_prop "contenteditable" false;
              ]
            [ text "*" ];
          wrap_inline ~focused inner;
          span
            ~a:
              [
                bool_prop "hidden" (not focused);
                bool_prop "contenteditable" false;
              ]
            [ text "*" ];
        ]
  | Strong inner ->
      elt "strong" ~key:id
        ~a:[ attr "id" id ]
        [
          span
            ~a:
              [
                bool_prop "hidden" (not focused);
                bool_prop "contenteditable" false;
              ]
            [ text "**" ];
          wrap_inline ~focused inner;
          span
            ~a:
              [
                bool_prop "hidden" (not focused);
                bool_prop "contenteditable" false;
              ]
            [ text "**" ];
        ]
  | Seq inlines ->
      span ~key:id ~a:[ attr "id" id ] (List.map (wrap_inline ~focused) inlines)
  | Break typ -> (
      match typ with
      | `Hard ->
          span ~key:id
            ~a:[ attr "id" id ]
            [ text "\n" (* Hard breaks render as newlines *) ]
      | `Soft ->
          span ~key:id
            ~a:[ attr "id" id ]
            [ text " " (* Soft breaks render as a space *) ])
  | Image { alt; src } ->
      if focused then
        span ~key:id
          ~a:[ attr "id" id ]
          [
            text "![";
            wrap_inline ~focused alt;
            text "](";
            text src;
            text ")";
          ]
      else
        let alt_text = inline_to_plain alt in
        elt "img" ~key:id
          ~a:[ attr "id" id; attr "src" src; attr "alt" alt_text ]
          []
  | Link { text = text'; href } ->
      if focused then
        span ~key:id
          ~a:[ attr "id" id ]
          [
            text "[";
            wrap_inline ~focused text';
            text "](";
            text href;
            text ")";
          ]
      else
        elt "a" ~key:id
          ~a:[ attr "id" id; attr "href" href ]
          [ wrap_inline ~focused text' ]
  | Raw_html html ->
      if focused then span ~key:id ~a:[ attr "id" id ] [ text html ]
      else
        (* For safety, render as text *)
        span ~key:id ~a:[ attr "id" id ] [ text html ]

let newline = span ~a:[ bool_prop "hidden" true ] [ text "\n" ]
let with_newline el = fragment [ el; newline ]

let rec render_block (view_state : Quill.View.t) (block : Quill.Document.block) =
  let block_id = (block.id :> int) in
  let id = Printf.sprintf "block-%d" block_id in
  let is_focused = 
    match view_state.focused_block with
    | Some fid -> fid = block.id
    | None -> false
  in
  let focused_inline_id = view_state.focused_inline in
  let inline_focused = 
    match focused_inline_id with
    | Some _ -> true  (* TODO: check if inline is within this block *)
    | None -> false
  in
  
  match block.content with
  | Paragraph inline ->
      p ~key:id ~a:[ attr "id" id ] [ wrap_inline ~focused:inline_focused inline ] |> with_newline
  | Codeblock { code; language; output } ->
      let code_id = Printf.sprintf "codeblock-%d" block_id in
      let lang_attr = match language with
        | Some lang -> [ attr "data-language" lang ]
        | None -> []
      in
      let code_el = elt "code" ~a:(attr "id" code_id :: lang_attr) [ text code ] in
      let pre_node =
        elt "pre" ~key:id
          ~a:[ attr "id" id ]
          [
            span ~a:[ bool_prop "hidden" true ] [ text "```\n" ];
            code_el;
            span ~a:[ bool_prop "hidden" true ] [ text "\n```\n" ];
          ]
      in
      (match output with
      | None -> pre_node
      | Some output_blocks ->
          let output_id = Printf.sprintf "output-%d" block_id in
          let start_comment =
            span
              ~a:[ bool_prop "hidden" true ]
              [ text "<!-- quill=output_start -->\n" ]
          in
          let end_comment =
            span
              ~a:[ bool_prop "hidden" true ]
              [ text "<!-- quill=output_end -->\n" ]
          in
          let output_children = List.map (render_block view_state) output_blocks in
          let output_el =
            div ~key:output_id
              ~a:
                [
                  attr "id" output_id;
                  attr "class" "execution-output";
                  bool_prop "contenteditable" false;
                ]
              output_children
          in
          fragment [ pre_node; start_comment; output_el; end_comment ])
  | Heading (level, inline) ->
      let tag = Printf.sprintf "h%d" level in
      let pre =
        span
          ~a:[ bool_prop "hidden" (not is_focused) ]
          [ text (String.make level '#' ^ " ") ]
      in
      elt tag ~key:id ~a:[ attr "id" id ] [ pre; wrap_inline ~focused:inline_focused inline ] |> with_newline
  | Blank_line ->
      p
        ~key:id
        ~a:
          [
            attr "id" id; bool_prop "hidden" true;
          ]
        [ text "\n" ]
  | Block_quote blocks ->
      let children = List.map (render_block view_state) blocks in
      elt "blockquote" ~key:id ~a:[ attr "id" id ] children |> with_newline
  | Thematic_break ->
      elt "hr" ~key:id ~a:[ attr "id" id ] [] |> with_newline
  | List (list_type, _spacing, items) ->
      let tag = match list_type with 
        | Unordered _ -> "ul" 
        | Ordered _ -> "ol" 
      in
      let list_items =
        List.mapi
          (fun i item_blocks ->
            let item_id = Printf.sprintf "%s-item-%d" id i in
            let children = List.map (render_block view_state) item_blocks in
            elt "li" ~key:item_id ~a:[ attr "id" item_id ] children)
          items
      in
      elt tag ~key:id ~a:[ attr "id" id ] list_items |> with_newline
  | Html_block html ->
      (* For safety, we render HTML blocks as text in the editor *)
      div ~key:id
        ~a:[ attr "id" id; attr "class" "html-block" ]
        [ elt "code" [ text html ] ]
      |> with_newline

let render_execution_result (block_id : Quill.Document.block_id) (result : Quill.Execution.execution_result) =
  let result_text = match result.error with
    | None -> result.output
    | Some err -> "Error: " ^ err
  in
  div 
    ~a:[ attr "class" "execution-result"; attr "data-block-id" (string_of_int (block_id :> int)) ]
    [ text result_text ]

let view (model : Model.t) : msg Vdom.vdom =
  let document = Quill.Engine.get_document model.engine in
  let view_state = Quill.Engine.get_view model.engine in
  
  let blocks = Quill.Document.get_blocks document in
  let editor_content =
    match blocks with
    | [] -> p []
    | _ -> 
        let rendered_blocks = List.map (fun block ->
          let block_vdom = render_block view_state block in
          (* Add execution results if any *)
          match Quill.Engine.get_block_result model.engine block.id with
          | None -> block_vdom
          | Some result -> 
              fragment [ block_vdom; render_execution_result block.id result ]
        ) blocks in
        fragment rendered_blocks
  in
  fragment
    [
      div
        ~a:[ attr "id" "editor"; attr "contentEditable" "true" ]
        [ editor_content ];
      div ~a:[ attr "id" "debug" ] [ View_debug.view model ];
    ]