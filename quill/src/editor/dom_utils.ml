open Js_of_ocaml

let get_current_range () =
  let selection = Dom_html.window##getSelection in
  if selection##.rangeCount > 0 then Some (selection##getRangeAt 0) else None

let is_within_range (range : Dom_html.range Js.t) (node : Dom.node Js.t) : bool
    =
  let container = range##.startContainer in
  let rec is_descendant current =
    if current = node then true
    else
      match Js.Opt.to_option current##.parentNode with
      | Some parent -> is_descendant parent
      | None -> false
  in
  is_descendant container

let starts_with str prefix =
  let len = String.length prefix in
  if String.length str < len then false else String.sub str 0 len = prefix

let map_node_to_indices node =
  let rec find_span node =
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
      if List.length parts = 2 then (* "block-i" *)
        let block_idx = int_of_string (List.nth parts 1) in
        Some (block_idx, 0)
      else if List.length parts = 4 && List.nth parts 2 = "run" then
        (* "block-i-run-j" *)
        let block_idx = int_of_string (List.nth parts 1) in
        let inline_idx = int_of_string (List.nth parts 3) in
        Some (block_idx, inline_idx)
      else None
  | None -> None

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
        | Some (blk, run) -> Some { block = blk; run; offs = offset })
  else None

let restore_cursor pos =
  match pos with
  | None -> ()
  | Some { block; run; offs } -> (
      let id = Printf.sprintf "block-%d-run-%d" block run in
      match Dom_html.getElementById_opt id with
      | None -> ()
      | Some span_el ->
          Js.Opt.iter span_el##.firstChild (fun node ->
              let len =
                Js.Opt.get node##.nodeValue (fun () -> Js.string "")
                |> Js.to_string |> String.length
              in
              let o = if offs > len then len else offs in

              let range = Dom_html.document##createRange in
              ignore (range##setStart node o);
              ignore (range##collapse Js._true);
              Js_of_ocaml.Console.console##log range;
              let sel = Dom_html.window##getSelection in
              sel##removeAllRanges;
              sel##addRange range))
