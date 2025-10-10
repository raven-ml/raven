open Quill_markdown
open Update
open Vdom

let span ?key ?a l = elt "span" ?key ?a l
let p ?key ?a l = elt "p" ?key ?a l
let tabindex = int_attr "tabindex"

(* Helper functions for inline rendering *)
let get_segments (inline : inline) =
  match inline.inline_content with Seq rs -> rs | _ -> [ inline ]

and inline_to_plain (inline : inline) =
  inline_content_to_plain inline.inline_content

let rec wrap_run ~block (run : inline) =
  let focused = run.focused in
  match run.inline_content with
  | Run s ->
      let id = "run-" ^ string_of_int run.id in
      span ~key:id
        ~a:[ attr "id" id; attr "class" "inline-text" ]
        [ (if s = "" then elt "br" [] else text s) ]
  | Code_span s ->
      let id = "codespan-" ^ string_of_int run.id in
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
      let id = "emph-" ^ string_of_int run.id in
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
          fragment @@ render_inline_content ~block inner;
          span
            ~a:
              [
                bool_prop "hidden" (not focused);
                bool_prop "contenteditable" false;
              ]
            [ text "*" ];
        ]
  | Strong inner ->
      let id = "strong-" ^ string_of_int run.id in
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
          fragment @@ render_inline_content ~block inner;
          span
            ~a:
              [
                bool_prop "hidden" (not focused);
                bool_prop "contenteditable" false;
              ]
            [ text "**" ];
        ]
  | Seq inlines ->
      let id = "seq-" ^ string_of_int run.id in
      span ~key:id ~a:[ attr "id" id ] (List.map (wrap_run ~block) inlines)
  | Break typ -> (
      let id = "break-" ^ string_of_int run.id in
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
      let id = "image-" ^ string_of_int run.id in
      if focused then
        span ~key:id
          ~a:[ attr "id" id ]
          [
            text "![";
            fragment (render_inline_content ~block alt);
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
      let id = "link-" ^ string_of_int run.id in
      if focused then
        span ~key:id
          ~a:[ attr "id" id ]
          [
            text "[";
            fragment (render_inline_content ~block text');
            text "](";
            text href;
            text ")";
          ]
      else
        elt "a" ~key:id
          ~a:[ attr "id" id; attr "href" href ]
          (render_inline_content ~block text')
  | Raw_html html ->
      let id = "rawhtml-" ^ string_of_int run.id in
      if focused then span ~key:id ~a:[ attr "id" id ] [ text html ]
      else
        (* Assuming Vdom supports raw HTML; adjust as needed *)
        span ~key:id ~a:[ attr "id" id; attr "dangerouslySetInnerHTML" html ] []

and render_inline_content ~block (inline : inline) =
  match inline.inline_content with
  | Seq items -> List.map (wrap_run ~block) items
  | _ -> [ wrap_run ~block inline ]

let newline = span ~a:[ bool_prop "hidden" true ] [ text "\n" ]
let with_newline el = fragment [ el; newline ]

let paragraph block_id inline =
  let id = Printf.sprintf "block-%d" block_id in
  let children = render_inline_content ~block:block_id inline in
  p ~key:id ~a:[ attr "id" id ] children |> with_newline

let heading block_id level inline =
  let id = Printf.sprintf "block-%d" block_id in
  let tag = Printf.sprintf "h%d" level in
  let segs = get_segments inline in
  let children = List.map (fun run -> wrap_run ~block:block_id run) segs in
  let pre =
    span
      ~a:[ bool_prop "hidden" (not inline.focused) ]
      [ text (String.make level '#' ^ " ") ]
  in
  elt tag ~key:id ~a:[ attr "id" id ] (pre :: children) |> with_newline

let blank_line block_id =
  p
    ~key:(Printf.sprintf "block-%d" block_id)
    ~a:
      [
        attr "id" (Printf.sprintf "block-%d" block_id); bool_prop "hidden" true;
      ]
    [ text "\n" ]

let rec codeblock block_id content =
  let id = Printf.sprintf "block-%d" block_id in
  let code_id = Printf.sprintf "codeblock-%d" block_id in
  let code_el = elt "code" ~a:[ attr "id" code_id ] [ text content.code ] in
  let pre_node =
    (* Capture the <pre> node without the trailing newline wrapper *)
    elt "pre" ~key:id
      ~a:[ attr "id" id ]
      [
        span ~a:[ bool_prop "hidden" true ] [ text "```\n" ];
        code_el;
        span ~a:[ bool_prop "hidden" true ] [ text "\n```\n" ];
        (* Ensure PRE block has trailing newline in source *)
      ]
  in
  match content.output with
  | None -> pre_node
  | Some output_block ->
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
      let output_el =
        div ~key:output_id
          ~a:
            [
              attr "id" output_id;
              attr "class" "execution-output";
              bool_prop "contenteditable" false;
            ]
          [ block output_block.id output_block.content ]
      in
      fragment [ pre_node; start_comment; output_el; end_comment ]

and blocks block_id blocks =
  let id = Printf.sprintf "block-%d" block_id in
  let children = List.map (fun b -> block b.id b.content) blocks in
  div ~key:id ~a:[ attr "id" id ] children

and block_quote block_id blocks =
  let id = Printf.sprintf "block-%d" block_id in
  let children = List.map (fun b -> block b.id b.content) blocks in
  elt "blockquote" ~key:id ~a:[ attr "id" id ] children |> with_newline

and thematic_break block_id =
  let id = Printf.sprintf "block-%d" block_id in
  elt "hr" ~key:id ~a:[ attr "id" id ] [] |> with_newline

and list block_id list_type _spacing items =
  let id = Printf.sprintf "block-%d" block_id in
  let tag = match list_type with Unordered _ -> "ul" | Ordered _ -> "ol" in
  let list_items =
    List.mapi
      (fun i item_blocks ->
        let item_id = Printf.sprintf "%s-item-%d" id i in
        let children = List.map (fun b -> block b.id b.content) item_blocks in
        elt "li" ~key:item_id ~a:[ attr "id" item_id ] children)
      items
  in
  elt tag ~key:id ~a:[ attr "id" id ] list_items |> with_newline

and html_block block_id html =
  let id = Printf.sprintf "block-%d" block_id in
  (* For safety, we render HTML blocks as text in the editor *)
  div ~key:id
    ~a:[ attr "id" id; attr "class" "html-block" ]
    [ elt "code" [ text html ] ]
  |> with_newline

and link_reference_definition block_id _ld =
  (* Link reference definitions are not typically shown in the rendered
     output *)
  let id = Printf.sprintf "block-%d" block_id in
  div ~key:id ~a:[ attr "id" id; bool_prop "hidden" true ] []

and block block_id block_content =
  match block_content with
  | Paragraph inline -> paragraph block_id inline
  | Codeblock code -> codeblock block_id code
  | Heading (level, inline) -> heading block_id level inline
  | Blocks bs -> blocks block_id bs
  | Blank_line () -> blank_line block_id
  | Block_quote bs -> block_quote block_id bs
  | Thematic_break -> thematic_break block_id
  | List (list_type, spacing, items) -> list block_id list_type spacing items
  | Html_block html -> html_block block_id html
  | Link_reference_definition ld -> link_reference_definition block_id ld

let view (model : Model.t) : msg Vdom.vdom =
  let editor_content =
    match model.document with
    | [] -> p []
    | _ -> fragment (List.map (fun b -> block b.id b.content) model.document)
  in
  fragment
    [
      div
        ~a:[ attr "id" "editor"; attr "contentEditable" "true" ]
        [ editor_content ];
      div ~a:[ attr "id" "debug" ] [ View_debug.view model ];
    ]
