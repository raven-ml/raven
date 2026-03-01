(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic
open Quill

(* Palette *)

let prompt_fg = Ansi.Color.cyan
let output_fg = Ansi.Color.of_rgb 170 175 185
let error_fg = Ansi.Color.of_rgb 210 100 100
let hint_fg = Ansi.Color.of_rgb 80 80 92
let accent = Ansi.Color.yellow
let completion_bg = Ansi.Color.of_rgb 40 40 50

(* Constants *)

let textarea_id = "repl-input"
let prompt_str = "# "
let prompt_width = 2

(* Helpers *)

let clamp lo hi x = if x < lo then lo else if x > hi then hi else x

let starts_with ~prefix s =
  let lp = String.length prefix and ls = String.length s in
  lp <= ls && String.sub s 0 lp = prefix

let is_ident_start = function
  | 'a' .. 'z' | 'A' .. 'Z' | '_' -> true
  | _ -> false

let is_ident_char = function
  | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' | '\'' -> true
  | _ -> false

let take_first n xs =
  let rec loop acc n xs =
    if n <= 0 then List.rev acc
    else
      match xs with [] -> List.rev acc | x :: tl -> loop (x :: acc) (n - 1) tl
  in
  loop [] n xs

let unique_sorted strings =
  let sorted = List.sort String.compare strings in
  let rec dedup acc = function
    | a :: (b :: _ as tl) when String.equal a b -> dedup acc tl
    | x :: tl -> dedup (x :: acc) tl
    | [] -> List.rev acc
  in
  dedup [] sorted

let collect_identifiers s =
  let tbl = Hashtbl.create 64 in
  let n = String.length s in
  let i = ref 0 in
  while !i < n do
    if is_ident_start s.[!i] then begin
      let j = ref (!i + 1) in
      while !j < n && is_ident_char s.[!j] do
        incr j
      done;
      let token = String.sub s !i (!j - !i) in
      if String.length token >= 2 then Hashtbl.replace tbl token ();
      i := !j
    end
    else incr i
  done;
  Hashtbl.fold (fun key () acc -> key :: acc) tbl []

let ocaml_keywords =
  [
    "and";
    "as";
    "begin";
    "class";
    "done";
    "else";
    "end";
    "exception";
    "external";
    "false";
    "for";
    "fun";
    "function";
    "if";
    "in";
    "include";
    "let";
    "match";
    "module";
    "mutable";
    "of";
    "open";
    "rec";
    "sig";
    "struct";
    "then";
    "true";
    "try";
    "type";
    "val";
    "when";
    "with";
  ]

let lowercase_codepoint i =
  if i >= Char.code 'A' && i <= Char.code 'Z' then i + 32 else i

let lp n = Toffee.Style.Length_percentage.length (Float.of_int n)

let padding_lrtb ~l ~r ~t ~b =
  Toffee.Geometry.Rect.make ~left:(lp l) ~right:(lp r) ~top:(lp t)
    ~bottom:(lp b)

(* Syntax highlighting *)

let highlight_source source =
  try
    Tree_sitter_ocaml.highlight_ocaml source
    |> Syntax_theme.apply Syntax_theme.default ~content:source
  with _ -> []

(* Cursor / byte offset helpers *)

let utf8_codepoint_offsets s =
  let len = String.length s in
  let rec prev_start i =
    if i <= 0 then 0
    else if Char.code s.[i] land 0xC0 = 0x80 then prev_start (i - 1)
    else i
  in
  let rec loop acc i =
    if i <= 0 then Array.of_list (0 :: acc)
    else
      let j = prev_start (i - 1) in
      loop (i :: acc) j
  in
  loop [] len

let grapheme_byte_offsets s = utf8_codepoint_offsets s

let cursor_byte_of code cursor =
  let offsets = grapheme_byte_offsets code in
  let max_cursor = Array.length offsets - 1 in
  let cursor = clamp 0 max_cursor cursor in
  (cursor, offsets.(cursor))

let cursor_of_byte code byte =
  let offsets = grapheme_byte_offsets code in
  let byte = clamp 0 (String.length code) byte in
  let rec loop i =
    if i >= Array.length offsets then Array.length offsets - 1
    else if offsets.(i) >= byte then i
    else loop (i + 1)
  in
  loop 0

let cursor_line code cursor =
  let _, cursor_byte = cursor_byte_of code cursor in
  let line = ref 0 in
  for i = 0 to cursor_byte - 1 do
    if code.[i] = '\n' then incr line
  done;
  !line

let line_count s =
  let n = ref 1 in
  String.iter (fun c -> if c = '\n' then incr n) s;
  !n

(* History *)

let history_sep = "(**)"

let history_path () =
  let base =
    match Sys.getenv_opt "XDG_DATA_HOME" with
    | Some dir -> dir
    | None -> (
        match Sys.getenv_opt "HOME" with
        | Some home -> Filename.concat home ".local/share"
        | None -> "/tmp")
  in
  Filename.concat (Filename.concat base "quill") "history.ml"

let load_history path =
  if not (Sys.file_exists path) then [||]
  else
    let ic = open_in path in
    let buf = Buffer.create 4096 in
    (try
       while true do
         Buffer.add_char buf (input_char ic)
       done
     with End_of_file -> close_in ic);
    let content = Buffer.contents buf in
    let lines = String.split_on_char '\n' content in
    let rec loop acc curr = function
      | [] ->
          let entry = String.concat "\n" (List.rev curr) in
          let entries = if String.trim entry = "" then acc else entry :: acc in
          Array.of_list (List.rev entries)
      | l :: ls ->
          if String.trim l = history_sep then
            let entry = String.concat "\n" (List.rev curr) in
            let acc = if String.trim entry = "" then acc else entry :: acc in
            loop acc [] ls
          else loop acc (l :: curr) ls
    in
    loop [] [] lines

let save_history path entries count =
  let dir = Filename.dirname path in
  (if not (Sys.file_exists dir) then
     try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ());
  let oc = open_out path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () ->
      for i = 0 to count - 1 do
        if i > 0 then Printf.fprintf oc "\n%s\n" history_sep;
        output_string oc entries.(i)
      done;
      output_char oc '\n')

(* Model *)

type execution_state = Idle | Executing

type completion = {
  prefix : string;
  cursor_byte : int;
  replace_start_byte : int;
  items : string list;
  selected : int;
}

type model = {
  kernel : Kernel.t;
  event_queue : Kernel.event Queue.t;
  execution_state : execution_state;
  input : string;
  input_cursor : int;
  input_cursor_override : int option;
  input_selection : (int * int) option;
  draft : string;
  completion : completion option;
  completion_popup_open : bool;
  history : string array;
  history_count : int;
  history_cursor : int;
  history_path : string;
  pending_outputs : Cell.output list;
  pending_cell_id : Cell.id option;
  counter : int;
  type_info : Kernel.type_info option;
  ctrl_x_pending : bool;
}

type msg =
  | Input_changed of string
  | Input_submitted of string
  | Cursor_changed of int * (int * int) option
  | History_prev
  | History_next
  | Trigger_completion
  | Next_completion
  | Prev_completion
  | Accept_completion
  | Dismiss_completion
  | Execution_tick
  | Execution_done
  | Show_type
  | Dismiss_type
  | Chord_prefix
  | Edit_external
  | Clear_screen
  | Interrupt
  | Quit

(* Completion *)

let find_prefix_at_cursor code ~cursor ~selection =
  match selection with
  | Some _ -> None
  | None ->
      let cursor, cursor_byte = cursor_byte_of code cursor in
      let len = String.length code in
      let at_ident_end =
        cursor_byte = len || not (is_ident_char code.[cursor_byte])
      in
      if not at_ident_end then None
      else
        let i = ref (cursor_byte - 1) in
        while
          !i >= 0 && (is_ident_char code.[!i] || Char.equal code.[!i] '.')
        do
          decr i
        done;
        let start = !i + 1 in
        let token =
          if cursor_byte > start then String.sub code start (cursor_byte - start)
          else ""
        in
        let replace_start_byte, prefix =
          match String.rindex_opt token '.' with
          | Some dot ->
              ( start + dot + 1,
                String.sub token (dot + 1) (String.length token - dot - 1) )
          | None -> (start, token)
        in
        Some (cursor, cursor_byte, replace_start_byte, prefix)

let selected_completion_item c =
  match c.items with
  | [] -> None
  | items -> Some (List.nth items (c.selected mod List.length items))

let index_of item items =
  let rec loop i = function
    | [] -> None
    | x :: _ when String.equal x item -> Some i
    | _ :: tl -> loop (i + 1) tl
  in
  loop 0 items

let cycle_completion c delta =
  let len = List.length c.items in
  if len = 0 then c
  else { c with selected = (c.selected + delta + len) mod len }

let build_completion ?(force = false) m =
  match
    find_prefix_at_cursor m.input ~cursor:m.input_cursor
      ~selection:m.input_selection
  with
  | None -> None
  | Some (_, cursor_byte, replace_start_byte, prefix) -> (
      if (not force) && String.length prefix = 0 then None
      else
        let kernel_items =
          try
            List.map
              (fun (c : Kernel.completion_item) -> c.label)
              (m.kernel.complete ~code:m.input ~pos:cursor_byte)
          with _ -> []
        in
        let items =
          unique_sorted
            (kernel_items @ collect_identifiers m.input @ ocaml_keywords)
          |> List.filter (fun item ->
              (String.length prefix = 0 || starts_with ~prefix item)
              && not (String.equal item prefix))
          |> take_first 200
        in
        match items with
        | [] -> None
        | _ ->
            Some
              { prefix; cursor_byte; replace_start_byte; items; selected = 0 })

let preserve_selection prev next =
  match (prev, next) with
  | Some prev, Some next -> (
      match selected_completion_item prev with
      | Some item -> (
          match index_of item next.items with
          | Some idx -> Some { next with selected = idx }
          | None -> Some next)
      | None -> Some next)
  | _, x -> x

let recompute_completion m =
  let force = m.completion_popup_open in
  let next = build_completion ~force m in
  { m with completion = preserve_selection m.completion next }

let ghost_text m =
  match m.completion with
  | None -> None
  | Some c when String.length c.prefix = 0 -> None
  | Some c -> (
      match selected_completion_item c with
      | None -> None
      | Some item when starts_with ~prefix:c.prefix item ->
          let suffix =
            String.sub item (String.length c.prefix)
              (String.length item - String.length c.prefix)
          in
          if String.length suffix = 0 then None else Some suffix
      | Some _ -> None)

let replace_range_at_byte s ~start_byte ~end_byte text =
  let len = String.length s in
  let start_byte = clamp 0 len start_byte in
  let end_byte = clamp start_byte len end_byte in
  String.sub s 0 start_byte ^ text ^ String.sub s end_byte (len - end_byte)

let apply_completion m c choice =
  let code =
    replace_range_at_byte m.input ~start_byte:c.replace_start_byte
      ~end_byte:c.cursor_byte choice
  in
  let cursor =
    cursor_of_byte code (c.replace_start_byte + String.length choice)
  in
  {
    m with
    input = code;
    input_cursor = cursor;
    input_cursor_override = Some cursor;
    input_selection = None;
    completion_popup_open = false;
    completion = None;
  }
  |> recompute_completion

(* External editor *)

let editor_cmd () =
  match Sys.getenv_opt "VISUAL" with
  | Some e -> Some e
  | None -> Sys.getenv_opt "EDITOR"

(* Phrase completeness *)

let is_complete m =
  match m.kernel.is_complete with
  | Some f -> f m.input
  | None ->
      let trimmed = String.trim m.input in
      trimmed <> "" && String.ends_with ~suffix:";;" trimmed

(* History helpers *)

let add_history entry m =
  let trimmed = String.trim entry in
  if trimmed = "" then m
  else if
    m.history_count > 0 && String.equal m.history.(m.history_count - 1) trimmed
  then { m with history_cursor = m.history_count }
  else
    let history =
      if m.history_count >= Array.length m.history then begin
        let cap = max 256 (m.history_count * 2) in
        let arr = Array.make cap "" in
        Array.blit m.history 0 arr 0 m.history_count;
        arr
      end
      else m.history
    in
    history.(m.history_count) <- trimmed;
    {
      m with
      history;
      history_count = m.history_count + 1;
      history_cursor = m.history_count + 1;
    }

(* Event draining *)

let drain_events m =
  let rec loop outputs =
    match Queue.pop m.event_queue with
    | Kernel.Output { output; _ } -> loop (output :: outputs)
    | Kernel.Finished _ -> loop outputs
    | Kernel.Status_changed _ -> loop outputs
    | exception Queue.Empty -> outputs
  in
  let new_outputs = List.rev (loop []) in
  { m with pending_outputs = m.pending_outputs @ new_outputs }

(* Static output rendering *)

let render_output_item output =
  match output with
  | Cell.Stdout s when s <> "" ->
      Some (text ~style:(Ansi.Style.make ~fg:output_fg ()) s)
  | Cell.Stderr s when s <> "" ->
      Some (text ~style:(Ansi.Style.make ~fg:error_fg ()) s)
  | Cell.Error s ->
      Some (text ~style:(Ansi.Style.make ~fg:error_fg ~bold:true ()) s)
  | Cell.Display { mime; data } ->
      Some
        (text
           ~style:(Ansi.Style.make ~fg:hint_fg ~italic:true ())
           (Printf.sprintf "[%s · %d bytes]" mime (String.length data)))
  | _ -> None

let commit_input source =
  let spans = highlight_source source in
  Cmd.static_commit
    (box ~flex_direction:Column
       ~size:{ width = pct 100; height = auto }
       [
         box ~flex_direction:Row
           [
             text
               ~style:(Ansi.Style.make ~fg:prompt_fg ~bold:true ())
               prompt_str;
             code ~spans source;
           ];
       ])

let commit_outputs outputs =
  let items = List.filter_map render_output_item outputs in
  if items = [] then Cmd.none
  else
    Cmd.static_commit
      (box ~flex_direction:Column
         ~size:{ width = pct 100; height = auto }
         items)

(* Init *)

let init ~create_kernel () =
  let event_queue = Queue.create () in
  let on_event ev = Queue.push ev event_queue in
  let kernel = create_kernel ~on_event in
  let hpath = history_path () in
  let entries = load_history hpath in
  let count = Array.length entries in
  let history =
    if count = 0 then Array.make 256 ""
    else
      let cap = max 256 (count * 2) in
      let arr = Array.make cap "" in
      Array.blit entries 0 arr 0 count;
      arr
  in
  let banner =
    Cmd.static_commit
      (box ~flex_direction:Column
         ~size:{ width = pct 100; height = auto }
         [
           text
             ~style:(Ansi.Style.make ~bold:true ~fg:prompt_fg ())
             "Quill \xe2\x80\x94 OCaml Interactive Toplevel";
           text
             ~style:(Ansi.Style.make ~fg:hint_fg ())
             "All Raven packages pre-loaded. Ctrl+D to exit.";
           text "";
         ])
  in
  ( {
      kernel;
      event_queue;
      execution_state = Idle;
      input = "";
      input_cursor = 0;
      input_cursor_override = None;
      input_selection = None;
      draft = "";
      completion = None;
      completion_popup_open = false;
      history;
      history_count = count;
      history_cursor = count;
      history_path = hpath;
      pending_outputs = [];
      pending_cell_id = None;
      counter = 0;
      type_info = None;
      ctrl_x_pending = false;
    },
    banner )

(* Update *)

let handle_submit m =
  let trimmed = String.trim m.input in
  if trimmed = "" then (m, Cmd.none)
  else if m.execution_state = Executing then (m, Cmd.none)
  else
    let cell_id = Cell.fresh_id () in
    let input_cmd = commit_input m.input in
    let m = add_history trimmed m in
    save_history m.history_path m.history m.history_count;
    let exec_cmd =
      Cmd.perform (fun dispatch ->
          m.kernel.execute ~cell_id ~code:trimmed;
          dispatch Execution_done)
    in
    ( {
        m with
        input = "";
        input_cursor = 0;
        input_cursor_override = Some 0;
        input_selection = None;
        draft = "";
        execution_state = Executing;
        pending_outputs = [];
        pending_cell_id = Some cell_id;
        counter = m.counter + 1;
        completion = None;
        completion_popup_open = false;
        type_info = None;
      },
      Cmd.batch [ input_cmd; exec_cmd ] )

let update msg m =
  let m =
    match msg with Chord_prefix -> m | _ -> { m with ctrl_x_pending = false }
  in
  match msg with
  | Input_changed v ->
      let m =
        {
          m with
          input = v;
          type_info = None;
          completion_popup_open =
            (match m.input_selection with
            | Some _ -> false
            | None -> m.completion_popup_open);
        }
      in
      (recompute_completion m, Cmd.none)
  | Input_submitted _ -> handle_submit m
  | Cursor_changed (cursor, selection) ->
      let m =
        {
          m with
          input_cursor = cursor;
          input_cursor_override = None;
          input_selection = selection;
          completion_popup_open =
            (match selection with
            | Some _ -> false
            | None -> m.completion_popup_open);
        }
      in
      (recompute_completion m, Cmd.none)
  | History_prev ->
      if m.history_cursor > 0 then
        let draft =
          if m.history_cursor = m.history_count then m.input else m.draft
        in
        let cursor = m.history_cursor - 1 in
        let entry = m.history.(cursor) in
        ( {
            m with
            input = entry;
            draft;
            history_cursor = cursor;
            input_cursor = String.length entry;
            input_cursor_override = Some (String.length entry);
            input_selection = None;
            completion_popup_open = false;
            completion = None;
          },
          Cmd.none )
      else (m, Cmd.none)
  | History_next ->
      if m.history_cursor < m.history_count - 1 then
        let cursor = m.history_cursor + 1 in
        let entry = m.history.(cursor) in
        ( {
            m with
            input = entry;
            history_cursor = cursor;
            input_cursor = String.length entry;
            input_cursor_override = Some (String.length entry);
            input_selection = None;
            completion_popup_open = false;
            completion = None;
          },
          Cmd.none )
      else if m.history_cursor = m.history_count - 1 then
        ( {
            m with
            input = m.draft;
            history_cursor = m.history_count;
            input_cursor = String.length m.draft;
            input_cursor_override = Some (String.length m.draft);
            input_selection = None;
            completion_popup_open = false;
            completion = None;
          },
          Cmd.none )
      else (m, Cmd.none)
  | Trigger_completion ->
      if Option.is_some m.input_selection then (m, Cmd.none)
      else
        (recompute_completion { m with completion_popup_open = true }, Cmd.none)
  | Next_completion -> (
      match m.completion with
      | None -> (m, Cmd.none)
      | Some c ->
          ( {
              m with
              completion = Some (cycle_completion c 1);
              completion_popup_open = true;
            },
            Cmd.none ))
  | Prev_completion -> (
      match m.completion with
      | None -> (m, Cmd.none)
      | Some c ->
          ( {
              m with
              completion = Some (cycle_completion c (-1));
              completion_popup_open = true;
            },
            Cmd.none ))
  | Accept_completion -> (
      match m.completion with
      | None -> (m, Cmd.none)
      | Some c -> (
          match selected_completion_item c with
          | None ->
              ( recompute_completion { m with completion_popup_open = false },
                Cmd.none )
          | Some choice -> (apply_completion m c choice, Cmd.none)))
  | Dismiss_completion ->
      (recompute_completion { m with completion_popup_open = false }, Cmd.none)
  | Execution_tick ->
      let m = drain_events m in
      (m, Cmd.none)
  | Execution_done ->
      let m = drain_events m in
      let out_cmd = commit_outputs m.pending_outputs in
      ( {
          m with
          execution_state = Idle;
          pending_outputs = [];
          pending_cell_id = None;
        },
        Cmd.batch [ out_cmd; Cmd.focus textarea_id ] )
  | Show_type -> (
      match m.kernel.type_at with
      | Some f ->
          let _, cursor_byte = cursor_byte_of m.input m.input_cursor in
          let info = try f ~code:m.input ~pos:cursor_byte with _ -> None in
          ({ m with type_info = info }, Cmd.none)
      | None -> (m, Cmd.none))
  | Dismiss_type -> ({ m with type_info = None }, Cmd.none)
  | Chord_prefix -> ({ m with ctrl_x_pending = true }, Cmd.none)
  | Edit_external -> (
      match editor_cmd () with
      | None -> (m, Cmd.none)
      | Some editor ->
          let tmp = Filename.temp_file "quill" ".ml" in
          let oc = open_out tmp in
          output_string oc m.input;
          close_out oc;
          let cmd =
            Cmd.perform (fun dispatch ->
                ignore
                  (Sys.command
                     (Printf.sprintf "%s %s" editor (Filename.quote tmp)));
                let ic = open_in tmp in
                let buf = Buffer.create 256 in
                (try
                   while true do
                     Buffer.add_char buf (input_char ic)
                   done
                 with End_of_file -> close_in ic);
                (try Sys.remove tmp with Sys_error _ -> ());
                dispatch (Input_changed (Buffer.contents buf)))
          in
          (m, cmd))
  | Clear_screen -> (m, Cmd.static_clear)
  | Interrupt ->
      if m.execution_state = Executing then begin
        m.kernel.interrupt ();
        (m, Cmd.none)
      end
      else if m.input <> "" then
        let commit =
          Cmd.static_commit
            (box ~flex_direction:Row
               ~size:{ width = pct 100; height = auto }
               [
                 text
                   ~style:(Ansi.Style.make ~fg:prompt_fg ~bold:true ())
                   prompt_str;
                 text ~style:(Ansi.Style.make ~fg:hint_fg ()) m.input;
               ])
        in
        ( {
            m with
            input = "";
            input_cursor = 0;
            input_cursor_override = Some 0;
            input_selection = None;
            draft = "";
            completion = None;
            completion_popup_open = false;
            history_cursor = m.history_count;
          },
          Cmd.batch [ commit; Cmd.focus textarea_id ] )
      else (m, Cmd.none)
  | Quit ->
      m.kernel.shutdown ();
      (m, Cmd.quit)

(* View *)

let completion_panel m =
  if not (m.completion_popup_open && m.execution_state = Idle) then empty
  else
    match m.completion with
    | None ->
        box ~border:true ~border_color:hint_fg ~padding:(padding 1)
          [
            text
              ~style:(Ansi.Style.make ~fg:hint_fg ())
              "No suggestions at cursor.";
          ]
    | Some c ->
        box ~border:true ~border_color:hint_fg
          ~padding:(padding_lrtb ~l:1 ~r:1 ~t:0 ~b:0)
          ~flex_direction:Column ~gap:(gap 0)
          ~margin:(margin_lrtb prompt_width 0 0 0)
          ~size:{ width = auto; height = auto }
          ~max_size:{ width = px 60; height = auto }
          [
            box ~flex_direction:Column ~gap:(gap 0)
              (take_first 10 c.items
              |> List.mapi (fun i item ->
                  let selected = i = c.selected in
                  let prefix = if selected then "> " else "  " in
                  text
                    ~style:
                      (if selected then
                         Ansi.Style.make ~fg:Ansi.Color.black ~bg:accent
                           ~bold:true ()
                       else Ansi.Style.make ~fg:output_fg ~bg:completion_bg ())
                    (prefix ^ item)));
          ]

let type_info_panel m =
  match m.type_info with
  | None -> empty
  | Some info ->
      box ~border:true ~border_color:prompt_fg
        ~padding:(padding_lrtb ~l:1 ~r:1 ~t:0 ~b:0)
        ~margin:(margin_lrtb prompt_width 0 0 0)
        ~size:{ width = pct 100; height = auto }
        [ text ~style:(Ansi.Style.make ~fg:prompt_fg ()) info.typ ]

let streaming_output_view m =
  if m.execution_state = Idle then empty
  else
    let items = List.filter_map render_output_item m.pending_outputs in
    box ~flex_direction:Column
      ~size:{ width = pct 100; height = auto }
      ~max_size:{ width = pct 100; height = px 12 }
      ([
         box ~flex_direction:Row ~gap:(gap 1) ~align_items:Center
           [
             spinner ~color:prompt_fg ();
             text ~style:(Ansi.Style.make ~fg:hint_fg ()) "Evaluating...";
           ];
       ]
      @ items)

let hint_bar m =
  let hints =
    match m.execution_state with
    | Idle ->
        "Enter: submit \xc2\xb7 Ctrl+Enter: newline \xc2\xb7 Tab: complete \
         \xc2\xb7 Ctrl+T: type \xc2\xb7 Ctrl+D: exit"
    | Executing -> "Ctrl+C: interrupt"
  in
  text ~style:(Ansi.Style.make ~fg:hint_fg ()) hints

let view m =
  let spans = highlight_source m.input in
  let gt = ghost_text m in
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = auto }
    [
      box ~flex_direction:Row
        ~size:{ width = pct 100; height = auto }
        [
          text ~style:(Ansi.Style.make ~fg:prompt_fg ~bold:true ()) prompt_str;
          textarea ~id:textarea_id ~autofocus:true ~value:m.input ~spans
            ?cursor:m.input_cursor_override ~selection:m.input_selection
            ?ghost_text:gt
            ~ghost_text_color:(Ansi.Color.grayscale ~level:10)
            ~cursor_style:`Line ~cursor_color:prompt_fg ~wrap:`None
            ~size:{ width = pct 100; height = auto }
            ~flex_grow:1.
            ~on_input:(fun v -> Some (Input_changed v))
            ~on_submit:(fun v -> Some (Input_submitted v))
            ~on_cursor:(fun ~cursor ~selection ->
              Some (Cursor_changed (cursor, selection)))
            ();
        ];
      completion_panel m;
      type_info_panel m;
      streaming_output_view m;
      hint_bar m;
    ]

(* Key handling *)

let on_key m ev =
  let data = Event.Key.data ev in
  let md = data.modifier in
  (* Ctrl+X prefix: second key of chord *)
  if m.ctrl_x_pending then begin
    match data.key with
    | Char c
      when md.ctrl && lowercase_codepoint (Uchar.to_int c) = Char.code 'e' ->
        Event.Key.prevent_default ev;
        Some Edit_external
    | _ -> None
  end
  else
    match data.key with
    (* Completion keys -- checked first *)
    | Escape when m.completion_popup_open ->
        Event.Key.prevent_default ev;
        Some Dismiss_completion
    | Enter
      when m.completion_popup_open
           && not (md.ctrl || md.alt || md.super || md.shift) ->
        Event.Key.prevent_default ev;
        Some Accept_completion
    | Tab when m.completion_popup_open && md.shift ->
        Event.Key.prevent_default ev;
        Some Prev_completion
    | Tab when m.completion_popup_open ->
        Event.Key.prevent_default ev;
        Some Accept_completion
    | Char c
      when m.completion_popup_open && md.ctrl
           && lowercase_codepoint (Uchar.to_int c) = Char.code 'n' ->
        Event.Key.prevent_default ev;
        Some Next_completion
    | Char c
      when m.completion_popup_open && md.ctrl
           && lowercase_codepoint (Uchar.to_int c) = Char.code 'p' ->
        Event.Key.prevent_default ev;
        Some Prev_completion
    (* Ctrl+Space: trigger completion *)
    | Char c when md.ctrl && Uchar.to_int c = Char.code ' ' ->
        Event.Key.prevent_default ev;
        Some Trigger_completion
    (* Tab: trigger completion *)
    | Tab when Option.is_none m.input_selection ->
        Event.Key.prevent_default ev;
        Some Trigger_completion
    (* Enter: submit if complete, otherwise newline *)
    | Enter when not (md.ctrl || md.shift || md.alt || md.super) ->
        if m.execution_state = Executing then begin
          Event.Key.prevent_default ev;
          None
        end
        else if is_complete m then begin
          Event.Key.prevent_default ev;
          Some (Input_submitted m.input)
        end
        else None
    (* Up at first line: history prev *)
    | Up
      when (not (md.ctrl || md.alt || md.super))
           && cursor_line m.input m.input_cursor = 0
           && Option.is_none m.input_selection ->
        Event.Key.prevent_default ev;
        Some History_prev
    (* Down at last line: history next *)
    | Down
      when (not (md.ctrl || md.alt || md.super))
           && cursor_line m.input m.input_cursor = line_count m.input - 1
           && Option.is_none m.input_selection ->
        Event.Key.prevent_default ev;
        Some History_next
    (* Ctrl+L: clear screen *)
    | Char c
      when md.ctrl && lowercase_codepoint (Uchar.to_int c) = Char.code 'l' ->
        Event.Key.prevent_default ev;
        Some Clear_screen
    (* Ctrl+C: interrupt or cancel *)
    | Char c
      when md.ctrl && lowercase_codepoint (Uchar.to_int c) = Char.code 'c' ->
        Event.Key.prevent_default ev;
        Some Interrupt
    (* Ctrl+D on empty: quit *)
    | Char c
      when md.ctrl
           && lowercase_codepoint (Uchar.to_int c) = Char.code 'd'
           && m.input = "" && m.execution_state = Idle ->
        Event.Key.prevent_default ev;
        Some Quit
    (* Ctrl+X: start chord prefix *)
    | Char c
      when md.ctrl && lowercase_codepoint (Uchar.to_int c) = Char.code 'x' ->
        Event.Key.prevent_default ev;
        Some Chord_prefix
    (* Ctrl+T: show type *)
    | Char c
      when md.ctrl && lowercase_codepoint (Uchar.to_int c) = Char.code 't' ->
        Event.Key.prevent_default ev;
        Some Show_type
    | _ -> None

(* Subscriptions *)

let subscriptions m =
  Sub.batch
    [
      (if m.execution_state = Executing then
         Sub.on_tick (fun ~dt:_ -> Execution_tick)
       else Sub.none);
      Sub.on_key_all (on_key m);
    ]

(* Entry points *)

let run ~create_kernel =
  let matrix =
    Matrix.create ~mode:`Primary ~target_fps:(Some 60.) ~cursor_visible:true
      ~mouse_enabled:false ()
  in
  let init () = init ~create_kernel () in
  Mosaic.run ~matrix { init; update; view; subscriptions }

let run_pipe ~create_kernel =
  let event_queue = Queue.create () in
  let on_event ev = Queue.push ev event_queue in
  let (kernel : Kernel.t) = create_kernel ~on_event in
  let buf = Buffer.create 4096 in
  (try
     while true do
       Buffer.add_char buf (input_char stdin)
     done
   with End_of_file -> ());
  let code = Buffer.contents buf in
  if String.trim code <> "" then begin
    let cell_id = Cell.fresh_id () in
    kernel.execute ~cell_id ~code;
    let rec drain () =
      match Queue.pop event_queue with
      | Kernel.Output { output; _ } ->
          (match output with
          | Cell.Stdout s -> print_string s
          | Cell.Stderr s -> Printf.eprintf "%s" s
          | Cell.Error s -> Printf.eprintf "%s" s
          | Cell.Display { mime; _ } -> Printf.eprintf "[%s output]\n%!" mime);
          drain ()
      | Kernel.Finished _ -> drain ()
      | Kernel.Status_changed _ -> drain ()
      | exception Queue.Empty -> ()
    in
    drain ()
  end;
  kernel.shutdown ()
