open Model
open Update
open Vdom

let span ?key ?a l = elt "span" ?key ?a l
let p ?key ?a l = elt "p" ?key ?a l
let tabindex = int_attr "tabindex"
let get_segments = function Seq rs -> rs | r -> [ r ]

let rec inline_to_plain = function
  | Run s -> s
  | Emph i -> inline_to_plain i
  | Strong i -> inline_to_plain i
  | Seq items -> String.concat "" (List.map inline_to_plain items)

let wrap_run ?focused ~block ~run_j run =
  let id = Printf.sprintf "block-%d-run-%d" block run_j in
  let txt =
    match run with
    | Run s -> s
    | Emph _ | Strong _ | Seq _ -> inline_to_plain run
  in
  let focused =
    match focused with
    | Some (b, r) when b = block && r = run_j -> true
    | _ -> false
  in
  let attrs = [ attr "id" id; tabindex 0 ] in
  match run with
  | Run s -> span ~key:id ~a:attrs [ text s ]
  | Emph _ ->
      span ~key:id ~a:attrs
        [
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "*" ];
          elt "em" [ text txt ];
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "*" ];
        ]
  | Strong _ ->
      (* wrap in <strong> *)
      span ~key:id ~a:attrs
        [
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "**" ];
          elt "strong" [ text txt ];
          span ~a:[ bool_prop "hidden" (not focused) ] [ text "**" ];
        ]
  | Seq _ ->
      (* Seq should have been flattened by get_segments *)
      failwith "unexpected Seq in wrap_run"

let paragraph ?focused block_id inline =
  let segs = get_segments inline in
  let children =
    List.mapi (fun j run -> wrap_run ?focused ~block:block_id ~run_j:j run) segs
  in
  let id = Printf.sprintf "block-%d" block_id in
  p ~key:id ~a:[ attr "id" id ] children

let codeblock block_id code =
  (* with pre code *)
  let id = Printf.sprintf "block-%d" block_id in
  elt "pre" ~key:id
    [
      span ~a:[ bool_prop "hidden" true ] [ text "```" ];
      elt "code" ~a:[ attr "id" id ] [ text code ];
      span ~a:[ bool_prop "hidden" true ] [ text "```" ];
    ]

let rec blocks block_id blocks =
  let id = Printf.sprintf "block-%d" block_id in
  let children = List.map (fun b -> block b.id b.content) blocks in
  div ~key:id ~a:[ attr "id" id ] children

and block block_id block_content =
  match block_content with
  | Paragraph inline -> paragraph block_id inline
  | Codeblock code -> codeblock block_id code
  | Blocks bs -> blocks block_id bs

let view (model : model) : msg Vdom.vdom =
  fragment (List.map (fun b -> block b.id b.content) model.document)
