open Model
open Update
open Vdom

let span ?key ?a l = elt "span" ?key ?a l
let p ?key ?a l = elt "p" ?key ?a l
let tabindex = int_attr "tabindex"

let get_segments (inline : inline) =
  match inline.inline_content with Seq rs -> rs | _ -> [ inline ]

let rec inline_content_to_plain (inline_content : inline_content) =
  match inline_content with
  | Run s -> s
  | Emph i -> inline_to_plain i
  | Strong i -> inline_to_plain i
  | Code_span s -> s
  | Seq items -> String.concat "" (List.map inline_to_plain items)

and inline_to_plain (inline : inline) =
  inline_content_to_plain inline.inline_content

let rec wrap_run ~block (run : inline) =
  let id = "run-" ^ string_of_int run.id in
  let focused = run.focused in
  match run.inline_content with
  | Run s -> span ~key:id ~a:[ attr "id" id ] [ text s ]
  | Code_span s ->
      span ~key:id
        [
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "`" ];
          span ~a:[ attr "id" id ] [ elt "code" [ text s ] ];
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "`" ];
        ]
  | Emph inner ->
      span ~key:(string_of_int run.id)
        ~a:[ attr "id" id ]
        [
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "*" ];
          elt "em" ~a:[ attr "id" id ] (render_inline_content ~block inner);
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "*" ];
        ]
  | Strong inner ->
      span ~key:(string_of_int run.id)
        ~a:[ attr "id" id ]
        [
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "**" ];
          elt "strong" (render_inline_content ~block inner);
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "**" ];
        ]
  | Seq inlines -> fragment (List.map (wrap_run ~block) inlines)

and render_inline_content ~block (inline : inline) =
  match inline.inline_content with
  | Seq items -> List.map (wrap_run ~block) items
  | _ -> [ wrap_run ~block inline ]

let paragraph block_id inline =
  (* on empty paragraphs, render br *)
  match inline.inline_content with
  | Run "" ->
      let id = Printf.sprintf "block-%d" block_id in
      p ~key:id ~a:[ attr "id" id ] [ elt "br" [] ]
  | _ ->
      let children = render_inline_content ~block:block_id inline in
      let id = Printf.sprintf "block-%d" block_id in
      p ~key:id ~a:[ attr "id" id ] children

let codeblock block_id content =
  let id = Printf.sprintf "block-%d" block_id in
  let code_id = Printf.sprintf "codeblock-%d" block_id in
  let code_el = elt "code" ~a:[ attr "id" code_id ] [ text content.code ] in
  let pre_el = elt "pre" ~key:id ~a:[ attr "id" id ] [ code_el ] in
  match content.output with
  | None -> pre_el
  | Some output ->
      let output_id = Printf.sprintf "output-%d" block_id in
      fragment
        [
          pre_el;
          div ~key:output_id
            ~a:[ attr "id" output_id; attr "class" "execution-output" ]
            [ text output ];
        ]

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
  elt tag ~key:id ~a:[ attr "id" id ] (pre :: children)

let blank_line block_id =
  p
    ~key:(Printf.sprintf "block-%d" block_id)
    ~a:[ attr "id" (Printf.sprintf "block-%d" block_id) ]
    []

let rec blocks block_id blocks =
  let id = Printf.sprintf "block-%d" block_id in
  let children = List.map (fun b -> block b.id b.content) blocks in
  div ~key:id ~a:[ attr "id" id ] children

and block block_id block_content =
  match block_content with
  | Paragraph inline -> paragraph block_id inline
  | Codeblock code -> codeblock block_id code
  | Heading (level, inline) -> heading block_id level inline
  | Blocks bs -> blocks block_id bs
  | Blank_line () -> blank_line block_id

let view (model : model) : msg Vdom.vdom =
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
      div ~a:[ attr "id" "debug" ] [ Debug_view.view model ];
    ]
