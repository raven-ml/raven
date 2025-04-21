[@@@warning "-69"]

open Model
open Vdom
open Js_of_ocaml
open Js_browser

let log fmt =
  Printf.ksprintf
    (fun s -> Js_of_ocaml.Console.console##log (Js.string ("[main] " ^ s)))
    fmt

let get_current_range () =
  let selection = Dom_html.window##getSelection in
  if selection##.rangeCount > 0 then Some (selection##getRangeAt 0) else None

let starts_with str prefix =
  let len = String.length prefix in
  if String.length str < len then false else String.sub str 0 len = prefix

let map_node_to_indices (node : Dom.node Js.t) : (int * int) option =
  let rec find_span (node : Dom.node Js.t) =
    if node##.nodeType = Dom.ELEMENT then
      let el : Dom_html.element Js.t = Js.Unsafe.coerce node in
      let id = Js.to_string el##.id in
      if starts_with id "block-" then Some el
      else
        match Js.Opt.to_option el##.parentNode with
        | Some parent -> find_span parent
        | None -> None
    else
      match Js.Opt.to_option node##.parentNode with
      | Some parent -> find_span parent
      | None -> None
  in
  match find_span node with
  | Some span_el ->
      let id = Js.to_string span_el##.id in
      let parts = String.split_on_char '-' id in
      if List.length parts >= 4 then (* Expecting "block-i-run-j" *)
        let block_idx = int_of_string (List.nth parts 1) in
        let inline_idx = int_of_string (List.nth parts 3) in
        Some (block_idx, inline_idx)
      else None
  | None -> None

let handle_selectionchange _ev =
  log "[selectionchange]";
  let sel = Dom_html.window##getSelection in
  if not (Js.to_bool sel##.isCollapsed) then None
  else
    match get_current_range () with
    | None -> None
    | Some range -> (
        let node = range##.startContainer in
        match map_node_to_indices node with
        | None -> None
        | Some (block_idx, inline_idx) ->
            Some (Update.Focus_inline (block_idx, inline_idx)))

let child_nodes (el : Dom.node Js.t) : Dom.node Js.t list =
  let rec loop acc ndx =
    match Js.Opt.to_option ndx with
    | None -> List.rev acc
    | Some n -> loop (n :: acc) n##.nextSibling
  in
  loop [] el##.firstChild

let document_of_dom (root : Dom_html.element Js.t) : block list =
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
  in

  let ps = nodeList_to_list (root##getElementsByTagName (Js.string "p")) in

  ps
  |> List.filter_map (fun p_el ->
         let p_el = Js.Unsafe.coerce p_el in
         let id_s = Js.to_string p_el##.id in
         match String.split_on_char '-' id_s with
         | [ "block"; idx ] ->
             let blk = int_of_string idx in
             let txt =
               Js.Opt.get p_el##.textContent (fun () -> Js.string "")
               |> Js.to_string
             in
             Some { id = blk; content = block_content_of_md txt }
         | _ -> None)

let handle_input root _ev = Some (Update.Set_document (document_of_dom root))
let app = simple_app ~init ~update:Update.update ~view:View.view ()

type cursor_pos = { block : int; run : int; offs : int }

let save_cursor () =
  let sel = Dom_html.window##getSelection in
  if Js.to_bool sel##.isCollapsed then
    match get_current_range () with
    | None -> None
    | Some range -> (
        let container = range##.startContainer in
        let offset = range##.startOffset in
        match map_node_to_indices container with
        | None -> None
        | Some (blk, run) ->
            log "[save_cursor] block=%d, run=%d, offs=%d" blk run offset;
            Some { block = blk; run; offs = offset })
  else None

let restore_cursor (pos : cursor_pos option) : unit =
  match pos with
  | None -> ()
  | Some { block; run; offs } -> (
      log "[restore_cursor] block=%d, run=%d, offs=%d" block run offs;
      let id = Printf.sprintf "block-%d-run-%d" block run in
      match Dom_html.getElementById_opt id with
      | None -> ()
      | Some span_el ->
          let children = child_nodes (span_el :> Dom.node Js.t) in
          log "[restore] %d children under %s" (List.length children) id;
          List.iteri
            (fun i ch ->
              let ch = Js.Unsafe.coerce ch in
              log "[restore] child %d: nodeType=%d, nodeName=%s, nodeValue=%s" i
                ch##.nodeType
                (Js.to_string (Js.Unsafe.coerce ch##.nodeName))
                (Js.Opt.case ch##.nodeValue (fun () -> "<none>") Js.to_string))
            children;
          Js.Opt.iter span_el##.firstChild (fun node ->
              let len =
                Js.Opt.get node##.nodeValue (fun () -> Js.string "")
                |> Js.to_string |> String.length
              in
              let o = if offs > len then len else offs in

              let range = Dom_html.document##createRange in
              ignore (range##setStart node o);
              ignore (range##collapse Js._true);

              log "[restore_cursor] setting range";
              Js_of_ocaml.Console.console##log range;
              let sel = Dom_html.window##getSelection in
              sel##removeAllRanges;
              sel##addRange range))

let setup_editor editor =
  editor##setAttribute (Js.string "contentEditable") (Js.string "true");
  editor##setAttribute (Js.string "tabindex") (Js.string "0")

let () =
  match Document.get_element_by_id Js_browser.document "editor" with
  | None ->
      Js_of_ocaml.Console.console##error (Js.string "No #editor element found")
  | Some container ->
      setup_editor (Obj.magic container);
      let mounted_app = Vdom_blit.run ~container app in
      let dom_el = Vdom_blit.dom mounted_app in

      let _ =
        Dom_html.addEventListener (Obj.magic dom_el) (Dom.Event.make "input")
          (Dom_html.handler (fun ev ->
               let before = save_cursor () in
               (match handle_input (Obj.magic dom_el) ev with
               | Some msg -> Vdom_blit.process mounted_app msg
               | None -> ());
               ignore
                 (Dom_html.window##setTimeout
                    (Js.wrap_callback (fun _ -> restore_cursor before))
                    (Js.number_of_float 0.));
               Js._false))
          Js._false
      in

      let _ =
        Dom_html.addEventListener Dom_html.document
          (Dom.Event.make "selectionchange")
          (Dom_html.handler (fun ev ->
               (match handle_selectionchange ev with
               | Some msg -> Vdom_blit.process mounted_app msg
               | None -> ());
               Js._false))
          Js._false
      in

      Element.append_child (Document.body document) dom_el
