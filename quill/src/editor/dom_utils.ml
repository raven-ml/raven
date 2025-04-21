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

let get_text_content node =
  if node##.nodeType = Dom.TEXT then
    Js.Opt.get node##.nodeValue (fun () -> Js.string "") |> Js.to_string
  else
    match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
    | Some el ->
        Js.Opt.get el##.textContent (fun () -> Js.string "") |> Js.to_string
    | None -> ""

let is_span_with_text node text =
  match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
  | Some el when Js.to_string el##.tagName = "SPAN" ->
      Js.to_string el##.innerText = text
  | _ -> false

let get_element_tag node =
  match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
  | Some el -> Some (Js.to_string el##.tagName)
  | None -> None

let get_inner_text node =
  match Js.Opt.to_option (Dom_html.CoerceTo.element node) with
  | Some el -> Js.to_string el##.innerText
  | None -> ""

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

type cursor_pos =
  | Inline_pos of { inline_id : int; offs : int }
  | Codeblock_pos of { block_id : int; offs : int }

let map_node_to_inline_id node =
  let rec find_run_span node =
    if node##.nodeType = Dom.ELEMENT then
      let el : Dom_html.element Js.t = Js.Unsafe.coerce node in
      let id = Js.to_string el##.id in
      if starts_with id "run-" then Some el
      else
        match Js.Opt.to_option el##.parentNode with
        | Some parent -> find_run_span parent
        | None -> None
    else
      match Js.Opt.to_option node##.parentNode with
      | Some parent -> find_run_span parent
      | None -> None
  in
  match find_run_span node with
  | Some span_el ->
      let id = Js.to_string span_el##.id in
      let parts = String.split_on_char '-' id in
      if List.length parts >= 2 then
        let inline_id = int_of_string (List.nth parts 1) in
        Some inline_id
      else None
  | None -> None

let find_codeblock node =
  let rec find parent =
    if parent##.nodeType = Dom.ELEMENT then
      let el : Dom_html.element Js.t = Js.Unsafe.coerce parent in
      let id = Js.to_string el##.id in
      if starts_with id "codeblock-" then Some id
      else
        match Js.Opt.to_option parent##.parentNode with
        | Some grandparent -> find grandparent
        | None -> None
    else
      match Js.Opt.to_option parent##.parentNode with
      | Some grandparent -> find grandparent
      | None -> None
  in
  if node##.nodeType = Dom.TEXT then
    match Js.Opt.to_option node##.parentNode with
    | Some parent -> find parent
    | None -> None
  else find node

let save_cursor () =
  let sel = Dom_html.window##getSelection in
  if Js.to_bool sel##.isCollapsed then
    match get_current_range () with
    | None -> None
    | Some range -> (
        let container = range##.startContainer in
        let offset = range##.startOffset in
        match find_codeblock container with
        | Some codeblock_id ->
            let block_id =
              int_of_string
                (String.sub codeblock_id 10 (String.length codeblock_id - 10))
            in
            Some (Codeblock_pos { block_id; offs = offset })
        | None -> (
            match map_node_to_inline_id container with
            | Some inline_id -> Some (Inline_pos { inline_id; offs = offset })
            | None -> None))
  else None

let restore_cursor pos =
  match pos with
  | None -> ()
  | Some (Inline_pos { inline_id; offs }) -> (
      let id = "run-" ^ string_of_int inline_id in
      match Dom_html.getElementById_opt id with
      | None -> ()
      | Some span_el ->
          Js.Opt.iter span_el##.firstChild (fun node ->
              let len =
                Js.Opt.get node##.nodeValue (fun () -> Js.string "")
                |> Js.to_string |> String.length
              in
              let o = min offs len in
              let range = Dom_html.document##createRange in
              ignore (range##setStart node o);
              ignore (range##collapse Js._true);
              let sel = Dom_html.window##getSelection in
              sel##removeAllRanges;
              sel##addRange range))
  | Some (Codeblock_pos { block_id; offs }) -> (
      let code_id = "codeblock-" ^ string_of_int block_id in
      match Dom_html.getElementById_opt code_id with
      | None -> ()
      | Some code_el ->
          Js.Opt.iter code_el##.firstChild (fun node ->
              let len =
                Js.Opt.get node##.nodeValue (fun () -> Js.string "")
                |> Js.to_string |> String.length
              in
              let o = min offs len in
              let range = Dom_html.document##createRange in
              ignore (range##setStart node o);
              ignore (range##collapse Js._true);
              let sel = Dom_html.window##getSelection in
              sel##removeAllRanges;
              sel##addRange range))
