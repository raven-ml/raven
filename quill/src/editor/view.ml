open Model
open Update
open Vdom

let span ?key ?a l = elt "span" ?key ?a l
let p ?key ?a l = elt "p" ?key ?a l
let tabindex = int_attr "tabindex"

let get_segments (inline : inline) =
  match inline.content with Seq rs -> rs | _ -> [ inline ]

let rec inline_content_to_plain (inline_content : inline_content) =
  match inline_content with
  | Run s -> s
  | Emph i -> inline_content_to_plain i
  | Strong i -> inline_content_to_plain i
  | Seq items -> String.concat "" (List.map inline_to_plain items)

and inline_to_plain (inline : inline) = inline_content_to_plain inline.content

let wrap_run ~block ~run_j (run : inline) =
  let id = Printf.sprintf "block-%d-run-%d" block run_j in
  let txt = inline_to_plain run in
  let focused = run.focused in
  let attrs = [ attr "id" id; tabindex 0 ] in
  match run.content with
  | Run s -> span ~key:id ~a:attrs [ text s ]
  | Emph _ ->
      span ~key:id ~a:attrs
        [
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "*" ];
          elt "em" [ text txt ];
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "*" ];
        ]
  | Strong _ ->
      span ~key:id ~a:attrs
        [
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "**" ];
          elt "strong" [ text txt ];
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "**" ];
        ]
  | Seq _ -> failwith "unexpected Seq in wrap_run"

let paragraph block_id inline =
  let segs = get_segments inline in
  let children =
    List.mapi (fun j run -> wrap_run ~block:block_id ~run_j:j run) segs
  in
  let id = Printf.sprintf "block-%d" block_id in
  p ~key:id ~a:[ attr "id" id ] children

let codeblock block_id code =
  let id = Printf.sprintf "block-%d" block_id in
  elt "pre" ~key:id
    [
      span ~a:[ bool_prop "hidden" true ] [ text "```" ];
      elt "code" ~a:[ attr "id" id ] [ text code ];
      span ~a:[ bool_prop "hidden" true ] [ text "```" ];
    ]

let heading block_id level inline =
  let id = Printf.sprintf "block-%d" block_id in
  let tag = Printf.sprintf "h%d" level in
  let segs = get_segments inline in
  let children =
    List.mapi (fun j run -> wrap_run ~block:block_id ~run_j:j run) segs
  in
  let pre =
    span ~a:[ bool_prop "hidden" true ] [ text (String.make level '#' ^ " ") ]
  in
  elt tag ~key:id ~a:[ attr "id" id ] (pre :: children)

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
  | Blank_line () -> fragment []

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
