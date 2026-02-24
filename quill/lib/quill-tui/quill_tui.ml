(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic
open Quill

(* ───── Model ───── *)

type mode = Normal | Editing

type footer_msg_kind = Info | Warning | Error | Confirm

type footer_msg = {
  kind : footer_msg_kind;
  text : string;
  created_at : float;
}

type model = {
  session : Session.t;
  kernel : Kernel.t;
  event_queue : Kernel.event Queue.t;
  path : string;
  focus : int;
  mode : mode;
  dirty : bool;
  footer_msg : footer_msg option;
  last_mtime : float;
  reload_acc : float;
  confirm_quit : bool;
  show_help : bool;
  clock : float;
  viewport_width : int;
  viewport_height : int;
}

type msg =
  | Focus_next
  | Focus_prev
  | Execute_focused
  | Execute_all
  | Interrupt
  | Insert_code_below
  | Insert_text_below
  | Delete_focused
  | Toggle_cell_kind
  | Move_up
  | Move_down
  | Clear_focused
  | Clear_all
  | Save
  | Quit
  | Tick of float
  | Dismiss_message
  | Toggle_help
  | Resize of int * int
  | Enter_edit
  | Exit_edit
  | Edit_source of string
  | Submit_edit of string

(* ───── Palette ───── *)

let chrome_bg = Ansi.Color.of_rgb 24 24 30
let accent = Ansi.Color.of_rgb 218 165 80
let accent_dim = Ansi.Color.of_rgb 140 110 60
let border_focused = Ansi.Color.of_rgb 120 120 140
let border_unfocused = Ansi.Color.of_rgb 50 50 58
let label_fg = Ansi.Color.of_rgb 100 100 115
let hint_fg = Ansi.Color.of_rgb 80 80 92
let output_fg = Ansi.Color.of_rgb 170 175 185
let output_dim_fg = Ansi.Color.of_rgb 120 125 135
let warning_fg = Ansi.Color.of_rgb 210 180 100
let error_fg = Ansi.Color.of_rgb 210 100 100
let error_bg = Ansi.Color.of_rgb 50 30 30
let info_fg = Ansi.Color.of_rgb 150 160 175
let overlay_bg = Ansi.Color.of_rgb 12 12 16
let cell_bg_focused = Ansi.Color.of_rgb 30 30 38
let reload_interval = 1.0
let template = "# Untitled\n\n```ocaml\n\n```\n"
let scroll_box_id = "notebook-scroll"
let textarea_id = "cell-editor"
let help_scroll_id = "footer-help-scroll"
let lp n = Toffee.Style.Length_percentage.length (Float.of_int n)

let padding_lrtb ~l ~r ~t ~b =
  Toffee.Geometry.Rect.make ~left:(lp l) ~right:(lp r) ~top:(lp t)
    ~bottom:(lp b)

(* ───── Helpers ───── *)

let read_file path =
  let ic = open_in path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

let write_file path content =
  let oc = open_out path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc content)

let get_mtime path =
  try (Unix.stat path).Unix.st_mtime with Unix.Unix_error _ -> 0.

let drain_events event_queue session =
  let rec loop session =
    match Queue.pop event_queue with
    | Kernel.Output { cell_id; output } ->
        loop (Session.apply_output cell_id output session)
    | Kernel.Finished { cell_id; success } ->
        loop (Session.finish_execution cell_id ~success session)
    | Kernel.Status_changed _ -> loop session
    | exception Queue.Empty -> session
  in
  loop session

let focused_cell m = Doc.nth m.focus (Session.doc m.session)
let cell_count m = Doc.length (Session.doc m.session)
let char_eq c u = Uchar.equal u (Uchar.of_char c)

let with_footer_message m kind text =
  { m with footer_msg = Some { kind; text; created_at = m.clock } }

let clear_footer_message m = { m with footer_msg = None }

let clear_confirm_message m =
  match m.footer_msg with
  | Some { kind = Confirm; _ } -> clear_footer_message m
  | _ -> m

let footer_message_timeout kind =
  match kind with
  | Info -> Some 3.
  | Warning -> Some 5.
  | Error | Confirm -> None

let expire_footer_message m =
  match m.footer_msg with
  | Some ({ kind; created_at; _ } as footer_msg) -> (
      match footer_message_timeout kind with
      | Some timeout when m.clock -. created_at >= timeout ->
          { m with footer_msg = None }
      | _ -> { m with footer_msg = Some footer_msg })
  | None -> m

let is_navigation_msg msg =
  match msg with Focus_next | Focus_prev | Move_up | Move_down -> true | _ -> false

let should_clear_error_msg msg =
  if is_navigation_msg msg then false
  else
    match msg with
    | Tick _ | Dismiss_message | Toggle_help -> false
    | _ -> true

(* ───── Init ───── *)

let init ~create_kernel ~path () =
  let event_queue = Queue.create () in
  let on_event ev = Queue.push ev event_queue in
  let kernel = create_kernel ~on_event in
  let md =
    if Sys.file_exists path then read_file path
    else (
      write_file path template;
      template)
  in
  let doc = Quill_markdown.of_string md in
  let session = Session.create doc in
  let last_mtime = get_mtime path in
  ( {
      session;
      kernel;
      event_queue;
      path;
      focus = 0;
      mode = Normal;
      dirty = false;
      footer_msg = None;
      last_mtime;
      reload_acc = 0.;
      confirm_quit = false;
      show_help = false;
      clock = 0.;
      viewport_width = 120;
      viewport_height = 32;
    },
    Cmd.set_title (Printf.sprintf "Quill - %s" (Filename.basename path)) )

(* ───── File reload ───── *)

let check_reload m =
  let mtime = get_mtime m.path in
  if mtime > m.last_mtime then
    let md = read_file m.path in
    let doc = Quill_markdown.of_string md in
    let session = Session.reload doc m.session in
    let n = Doc.length (Session.doc session) in
    let focus = if n > 0 then min m.focus (n - 1) else 0 in
    { m with session; focus; last_mtime = mtime; dirty = false }
  else m

(* ───── Execute helpers ───── *)

let execute_cell m id source =
  let session = Session.checkpoint m.session in
  let session = Session.clear_outputs id session in
  let session = Session.mark_running id session in
  m.kernel.execute ~cell_id:id ~code:source;
  let session = drain_events m.event_queue session in
  clear_footer_message { m with session; dirty = true }

let execute_all_cells m =
  let session = Session.clear_all_outputs m.session in
  let session = ref session in
  List.iter
    (fun cell ->
      match cell with
      | Cell.Code { id; source; _ } ->
          session := Session.mark_running id !session;
          m.kernel.execute ~cell_id:id ~code:source;
          session := drain_events m.event_queue !session
      | Cell.Text _ -> ())
    (Doc.cells (Session.doc !session));
  clear_footer_message { m with session = !session; dirty = true }

(* ───── Update ───── *)

let tick_model m dt =
  let session = drain_events m.event_queue m.session in
  let m = { m with session; clock = m.clock +. dt } in
  expire_footer_message m

let update_toggle_help m =
  let show_help = not m.show_help in
  let cmd =
    if show_help then Cmd.focus help_scroll_id
    else match m.mode with Editing -> Cmd.focus textarea_id | Normal -> Cmd.focus scroll_box_id
  in
  ({ m with show_help }, cmd)

let update_save m =
  let session = Session.checkpoint m.session in
  let m = { m with session } in
  let doc = Session.doc m.session in
  let content = Quill_markdown.to_string_with_outputs doc in
  write_file m.path content;
  let last_mtime = get_mtime m.path in
  (with_footer_message { m with dirty = false; last_mtime } Info "Saved", Cmd.none)

let update_quit m =
  if m.dirty && not m.confirm_quit then
    ( with_footer_message { m with confirm_quit = true } Confirm
        "Unsaved changes. Press q again to quit, s to save.",
      Cmd.none )
  else (
    m.kernel.shutdown ();
    (m, Cmd.quit))

let update_editing msg m =
  match msg with
  | Toggle_help -> update_toggle_help m
  | Dismiss_message ->
      ({ m with confirm_quit = false; footer_msg = None }, Cmd.none)
  | Resize (width, height) ->
      ({ m with viewport_width = width; viewport_height = height }, Cmd.none)
  | Exit_edit ->
      let session = Session.checkpoint m.session in
      ({ m with mode = Normal; session }, Cmd.focus scroll_box_id)
  | Edit_source source -> (
      match focused_cell m with
      | Some cell ->
          let cell_id = Cell.id cell in
          let session = Session.update_source cell_id source m.session in
          ({ m with session; dirty = true }, Cmd.none)
      | None -> (m, Cmd.none))
  | Submit_edit _ -> (
      match focused_cell m with
      | Some (Cell.Code { id; source; _ }) ->
          let session = Session.checkpoint m.session in
          let m = { m with session; mode = Normal } in
          let m = execute_cell m id source in
          (m, Cmd.focus scroll_box_id)
      | _ ->
          let session = Session.checkpoint m.session in
          ({ m with mode = Normal; session }, Cmd.focus scroll_box_id))
  | Save -> update_save m
  | Quit ->
      let session = Session.checkpoint m.session in
      update_quit { m with session; mode = Normal }
  | Interrupt ->
      m.kernel.interrupt ();
      (m, Cmd.none)
  | Tick dt ->
      let m = tick_model m dt in
      ({ m with reload_acc = m.reload_acc +. dt }, Cmd.none)
  | _ -> (m, Cmd.none)

let update_normal msg m =
  match msg with
  | Toggle_help -> update_toggle_help m
  | Dismiss_message ->
      ({ m with confirm_quit = false; footer_msg = None }, Cmd.none)
  | Resize (width, height) ->
      ({ m with viewport_width = width; viewport_height = height }, Cmd.none)
  | Focus_next ->
      let n = cell_count m in
      let focus = if n > 0 then min (m.focus + 1) (n - 1) else 0 in
      ({ m with focus }, Cmd.none)
  | Focus_prev -> ({ m with focus = max (m.focus - 1) 0 }, Cmd.none)
  | Execute_focused -> (
      match focused_cell m with
      | Some (Cell.Code { id; source; _ }) ->
          (execute_cell m id source, Cmd.none)
      | Some (Cell.Text _) ->
          (with_footer_message m Error "Cannot execute a text cell", Cmd.none)
      | None -> (with_footer_message m Warning "No cell to execute", Cmd.none))
  | Execute_all -> (execute_all_cells m, Cmd.none)
  | Interrupt ->
      m.kernel.interrupt ();
      (m, Cmd.none)
  | Insert_code_below ->
      let pos = m.focus + 1 in
      let cell = Cell.code "" in
      let session = Session.insert_cell ~pos cell m.session in
      let n = Doc.length (Session.doc session) in
      ({ m with session; focus = min pos (n - 1); dirty = true }, Cmd.none)
  | Insert_text_below ->
      let pos = m.focus + 1 in
      let cell = Cell.text "" in
      let session = Session.insert_cell ~pos cell m.session in
      let n = Doc.length (Session.doc session) in
      ({ m with session; focus = min pos (n - 1); dirty = true }, Cmd.none)
  | Delete_focused -> (
      match focused_cell m with
      | Some cell ->
          let session = Session.remove_cell (Cell.id cell) m.session in
          let n = Doc.length (Session.doc session) in
          let focus = if n > 0 then min m.focus (n - 1) else 0 in
          ({ m with session; focus; dirty = true }, Cmd.none)
      | None -> (m, Cmd.none))
  | Toggle_cell_kind -> (
      match focused_cell m with
      | Some cell ->
          let cell_id = Cell.id cell in
          let kind =
            match cell with Cell.Code _ -> `Text | Cell.Text _ -> `Code
          in
          let session = Session.set_cell_kind cell_id kind m.session in
          ({ m with session; dirty = true }, Cmd.none)
      | None -> (m, Cmd.none))
  | Move_up -> (
      match focused_cell m with
      | Some cell when m.focus > 0 ->
          let cell_id = Cell.id cell in
          let pos = m.focus - 1 in
          let session = Session.move_cell cell_id ~pos m.session in
          ({ m with session; focus = pos; dirty = true }, Cmd.none)
      | _ -> (m, Cmd.none))
  | Move_down -> (
      match focused_cell m with
      | Some cell when m.focus < cell_count m - 1 ->
          let cell_id = Cell.id cell in
          let pos = m.focus + 1 in
          let session = Session.move_cell cell_id ~pos m.session in
          ({ m with session; focus = pos; dirty = true }, Cmd.none)
      | _ -> (m, Cmd.none))
  | Clear_focused -> (
      match focused_cell m with
      | Some cell ->
          let session = Session.clear_outputs (Cell.id cell) m.session in
          ({ m with session; dirty = true }, Cmd.none)
      | None -> (m, Cmd.none))
  | Clear_all ->
      let session = Session.clear_all_outputs m.session in
      ({ m with session; dirty = true }, Cmd.none)
  | Save -> update_save m
  | Quit -> update_quit m
  | Tick dt ->
      let m = tick_model m dt in
      let reload_acc = m.reload_acc +. dt in
      if reload_acc >= reload_interval then
        let m = check_reload { m with reload_acc = 0. } in
        (m, Cmd.none)
      else ({ m with reload_acc }, Cmd.none)
  | Enter_edit -> (
      match focused_cell m with
      | Some (Cell.Code _) -> ({ m with mode = Editing }, Cmd.focus textarea_id)
      | _ -> (m, Cmd.none))
  | _ -> (m, Cmd.none)

let update msg m =
  let m =
    if should_clear_error_msg msg then
      match m.footer_msg with
      | Some { kind = Error; _ } -> clear_footer_message m
      | _ -> m
    else m
  in
  let m =
    match msg with
    | Quit | Tick _ | Toggle_help | Resize _ -> m
    | _ -> clear_confirm_message { m with confirm_quit = false }
  in
  match m.mode with
  | Editing -> update_editing msg m
  | Normal -> update_normal msg m

(* ───── View Components ───── *)

let running_count m =
  List.fold_left
    (fun acc cell ->
      match cell with
      | Cell.Code { id; _ } ->
          if Session.cell_status id m.session = Session.Running then acc + 1
          else acc
      | _ -> acc)
    0
    (Doc.cells (Session.doc m.session))

let has_running m = running_count m > 0

let view_header m =
  let n = cell_count m in
  let left =
    box ~flex_direction:Row ~gap:(gap 1) ~align_items:Center
      [
        text ~style:(Ansi.Style.make ~fg:label_fg ~italic:true ()) "quill";
        text
          ~style:(Ansi.Style.make ~fg:Ansi.Color.white ~bold:true ())
          (Filename.basename m.path);
      ]
  in
  let center =
    let rc = running_count m in
    if rc > 0 then
      box ~flex_direction:Row ~gap:(gap 1) ~align_items:Center
        [
          spinner ~frame_set:Spinner.dots ~color:accent ();
          text
            ~style:(Ansi.Style.make ~fg:accent ())
            (Printf.sprintf "%d running" rc);
        ]
    else
      text
        ~style:(Ansi.Style.make ~fg:label_fg ())
        (Printf.sprintf "%d cells" n)
  in
  let right =
    if m.dirty then
      text ~style:(Ansi.Style.make ~fg:accent ~bold:true ()) "\xe2\x97\x8f"
    else empty
  in
  box ~background:chrome_bg ~flex_direction:Row ~justify_content:Space_between
    ~align_items:Center
    ~size:{ width = pct 100; height = auto }
    ~padding:(padding_lrtb ~l:2 ~r:2 ~t:0 ~b:0)
    [ left; center; right ]

let view_error_bar msg =
  box ~background:error_bg ~border:true ~border_sides:[ `Left ]
    ~border_style:Border.heavy ~border_color:error_fg
    ~size:{ width = pct 100; height = auto }
    ~padding:(padding_lrtb ~l:1 ~r:1 ~t:0 ~b:0)
    [ text ~style:(Ansi.Style.make ~fg:error_fg ()) msg ]

let trim_trailing_newlines s =
  let len = String.length s in
  let i = ref (len - 1) in
  while !i >= 0 && (s.[!i] = '\n' || s.[!i] = '\r') do
    decr i
  done;
  if !i = len - 1 then s else String.sub s 0 (!i + 1)

let view_output output =
  match output with
  | Cell.Stdout s ->
      text ~style:(Ansi.Style.make ~fg:output_fg ()) (trim_trailing_newlines s)
  | Cell.Stderr s ->
      text
        ~style:(Ansi.Style.make ~fg:warning_fg ~italic:true ())
        ("\xe2\x96\xb6 " ^ trim_trailing_newlines s)
  | Cell.Error s -> view_error_bar s
  | Cell.Display { mime; data } ->
      if String.starts_with ~prefix:"text/" mime then
        text ~style:(Ansi.Style.make ~fg:output_fg ()) data
      else
        text
          ~style:(Ansi.Style.make ~fg:output_dim_fg ~italic:true ())
          (Printf.sprintf "[%s \xc2\xb7 %d bytes]" mime (String.length data))

let view_code_cell ~index ~is_focused ~is_editing ~status source outputs =
  let border_color = if is_focused then border_focused else border_unfocused in
  let num = index + 1 in
  let title =
    if is_editing then Printf.sprintf " %d \xe2\x9c\x8e " num
    else
      let status_indicator =
        match status with
        | Session.Running -> " \xe2\x80\xa6"
        | Session.Queued -> " \xe2\x97\x8b"
        | Session.Idle -> if outputs <> [] then " \xe2\x9c\x93" else ""
      in
      Printf.sprintf " %d%s " num status_indicator
  in
  let source_view =
    if is_editing then
      box
        ~padding:(padding_lrtb ~l:1 ~r:1 ~t:0 ~b:0)
        ~size:{ width = pct 100; height = auto }
        [
          textarea ~id:textarea_id ~value:source ~text_color:output_fg
            ~background_color:cell_bg_focused ~focused_text_color:output_fg
            ~focused_background_color:cell_bg_focused ~cursor_style:`Line
            ~cursor_color:accent
            ~size:{ width = pct 100; height = auto }
            ~on_input:(fun s -> Some (Edit_source s))
            ~on_submit:(fun s -> Some (Submit_edit s))
            ();
        ]
    else
      let highlights =
        Tree_sitter_ocaml.highlight_ocaml source
        |> Syntax_theme.apply Syntax_theme.default ~content:source
      in
      box
        ~padding:(padding_lrtb ~l:1 ~r:1 ~t:1 ~b:0)
        ~size:{ width = pct 100; height = auto }
        [ code ~highlights source ]
  in
  let status_row =
    match status with
    | Session.Running ->
        box ~flex_direction:Row ~gap:(gap 1) ~align_items:Center
          ~padding:(padding_lrtb ~l:1 ~r:1 ~t:0 ~b:0)
          ~size:{ width = pct 100; height = auto }
          [
            spinner ~frame_set:Spinner.dots ~color:accent ();
            text
              ~style:(Ansi.Style.make ~fg:accent_dim ~italic:true ())
              "evaluating";
          ]
    | _ -> empty
  in
  let output_section =
    if outputs = [] then empty
    else
      box ~flex_direction:Column ~border:true ~border_sides:[ `Top ]
        ~border_style:Border.single ~border_color:border_unfocused
        ~size:{ width = pct 100; height = auto }
        ~padding:(padding 1)
        (List.map view_output outputs)
  in
  box ~flex_direction:Column ~border:true ~border_color
    ~border_style:Border.rounded ~title ~title_alignment:`Left
    ?background:(if is_focused then Some cell_bg_focused else None)
    ~size:{ width = pct 100; height = auto }
    [ source_view; status_row; output_section ]

let view_text_cell ~is_focused source =
  box
    ?background:(if is_focused then Some cell_bg_focused else None)
    ~size:{ width = pct 100; height = auto }
    ~padding:(padding_lrtb ~l:2 ~r:2 ~t:0 ~b:0)
    [ markdown source ]

let view_cell ~index ~focus ~mode m cell =
  let is_focused = index = focus in
  match cell with
  | Cell.Code { id; source; outputs; _ } ->
      let status = Session.cell_status id m.session in
      let is_editing = is_focused && mode = Editing in
      view_code_cell ~index ~is_focused ~is_editing ~status source outputs
  | Cell.Text { source; _ } -> view_text_cell ~is_focused source

let view_cells m =
  let cells = Doc.cells (Session.doc m.session) in
  if cells = [] then
    [
      box ~flex_direction:Column ~align_items:Center ~justify_content:Center
        ~flex_grow:1.
        ~size:{ width = pct 100; height = pct 100 }
        [
          text
            ~style:(Ansi.Style.make ~fg:label_fg ~italic:true ())
            "empty notebook";
          box
            ~size:{ width = auto; height = auto }
            ~padding:(padding_lrtb ~l:0 ~r:0 ~t:1 ~b:0)
            [
              text
                ~style:(Ansi.Style.make ~fg:hint_fg ())
                "press a to add a code cell, or t for text";
            ];
        ];
    ]
  else
    List.mapi
      (fun index cell -> view_cell ~index ~focus:m.focus ~mode:m.mode m cell)
      cells

type footer_width_tier = Wide | Medium | Compact | Tiny

type footer_action = {
  key : string;
  label : string;
}

let footer_width_tier m =
  if m.viewport_width >= 120 then Wide
  else if m.viewport_width >= 80 then Medium
  else if m.viewport_width >= 60 then Compact
  else Tiny

let rec take n xs =
  if n <= 0 then []
  else
    match xs with
    | [] -> []
    | x :: tl -> x :: take (n - 1) tl

let focused_kind_label m =
  match focused_cell m with
  | Some (Cell.Code _) -> "code"
  | Some (Cell.Text _) -> "text"
  | None -> "none"

let footer_mode_label m =
  match m.mode with Normal -> "NORMAL" | Editing -> "EDIT"

let footer_kernel_label m =
  let rc = running_count m in
  if rc > 0 then Printf.sprintf "running %d" rc else "idle"

let footer_actions m =
  if m.confirm_quit then
    [
      { key = "q"; label = "Confirm Quit" };
      { key = "s"; label = "Save" };
      { key = "Esc"; label = "Cancel" };
      { key = "?"; label = "Help" };
    ]
  else
    match m.mode with
    | Editing ->
        [
          { key = "Esc"; label = "Exit" };
          { key = "Ctrl-Enter"; label = "Run" };
          { key = "Ctrl-S"; label = "Save" };
          { key = "Ctrl-C"; label = "Interrupt" };
          { key = "?"; label = "Help" };
        ]
    | Normal ->
        if has_running m then
          [
            { key = "Ctrl-C"; label = "Interrupt" };
            { key = "j/k"; label = "Navigate" };
            { key = "s"; label = "Save" };
            { key = "q"; label = "Quit" };
            { key = "?"; label = "Help" };
          ]
        else
          match focused_cell m with
          | Some (Cell.Code _) ->
              [
                { key = "Enter"; label = "Run" };
                { key = "e"; label = "Edit" };
                { key = "a"; label = "+Code" };
                { key = "t"; label = "+Text" };
                { key = "s"; label = "Save" };
                { key = "?"; label = "Help" };
              ]
          | Some (Cell.Text _) ->
              [
                { key = "m"; label = "To Code" };
                { key = "a"; label = "+Code" };
                { key = "t"; label = "+Text" };
                { key = "s"; label = "Save" };
                { key = "?"; label = "Help" };
              ]
          | None ->
              [
                { key = "a"; label = "+Code" };
                { key = "t"; label = "+Text" };
                { key = "s"; label = "Save" };
                { key = "?"; label = "Help" };
              ]

let footer_action_limit tier =
  match tier with Wide -> 4 | Medium -> 3 | Compact -> 2 | Tiny -> 1

let footer_action_label tier label =
  match (tier, label) with
  | Medium, "Interrupt" -> "Stop"
  | Medium, "Navigate" -> "Nav"
  | Medium, "Confirm Quit" -> "Confirm"
  | Medium, "To Code" -> "ToCode"
  | Compact, "Save" -> "Save"
  | Compact, "Interrupt" -> "Stop"
  | Compact, "Navigate" -> "Nav"
  | Compact, "Confirm Quit" -> "Confirm"
  | Compact, "To Code" -> "Code"
  | Compact, "+Code" -> "+C"
  | Compact, "+Text" -> "+T"
  | Compact, "Help" -> "?"
  | _ -> label

let truncate_text max_len s =
  if String.length s <= max_len then s
  else String.sub s 0 (max 0 (max_len - 1)) ^ "…"

let footer_message_view tier m =
  match m.footer_msg with
  | None -> None
  | Some { kind; text = msg; _ } ->
      let fg, prefix =
        match kind with
        | Info -> (info_fg, "INFO")
        | Warning -> (warning_fg, "WARN")
        | Error -> (error_fg, "ERROR")
        | Confirm -> (warning_fg, "CONFIRM")
      in
      let max_len =
        match tier with Wide -> 32 | Medium -> 22 | Compact -> 14 | Tiny -> 8
      in
      Some (fg, Printf.sprintf "%s:%s" prefix (truncate_text max_len msg))

let footer_status_text tier m =
  let total = cell_count m in
  let focus =
    if total = 0 then "cell 0/0"
    else Printf.sprintf "cell %d/%d" (m.focus + 1) total
  in
  let kernel = footer_kernel_label m in
  let dirty = if m.dirty then "modified" else "saved" in
  match tier with
  | Wide ->
      Printf.sprintf "%s %s %s %s" focus (focused_kind_label m) dirty kernel
  | Medium -> Printf.sprintf "%s %s %s" focus (focused_kind_label m) kernel
  | Compact -> Printf.sprintf "%s %s" focus kernel
  | Tiny -> ""

let view_footer_actions tier m =
  let key_style = Ansi.Style.make ~fg:label_fg ~bold:true () in
  let desc_style = Ansi.Style.make ~fg:hint_fg () in
  let actions =
    if tier = Tiny then [ { key = "?"; label = "Help" } ]
    else take (footer_action_limit tier) (footer_actions m)
  in
  let view_action action =
    let label = footer_action_label tier action.label in
    box ~flex_direction:Row ~gap:(gap 0) ~align_items:Center
      ~size:{ width = auto; height = auto }
      [
        text ~style:key_style (Printf.sprintf "[%s]" action.key);
        text ~style:desc_style (Printf.sprintf " %s" label);
      ]
  in
  box ~flex_direction:Row ~gap:(gap 1) ~align_items:Center
    ~size:{ width = auto; height = auto }
    (List.map view_action actions)

let view_footer m =
  let tier = footer_width_tier m in
  let mode_style =
    Ansi.Style.make
      ~fg:(match m.mode with Editing -> accent | Normal -> label_fg)
      ~bold:true ()
  in
  let desc_style = Ansi.Style.make ~fg:hint_fg () in
  let status_text = footer_status_text tier m in
  let status_node =
    if status_text = "" then empty
    else text ~style:desc_style (Printf.sprintf " %s" status_text)
  in
  let message_node =
    match footer_message_view tier m with
    | Some (fg, msg) -> text ~style:(Ansi.Style.make ~fg ~bold:true ()) (" | " ^ msg)
    | None -> empty
  in
  let left =
    box ~flex_direction:Row ~gap:(gap 0) ~align_items:Center
      ~size:{ width = auto; height = auto }
      [
        text ~style:mode_style (Printf.sprintf "[%s]" (footer_mode_label m));
        status_node;
        message_node;
      ]
  in
  let right = view_footer_actions tier m in
  box ~background:chrome_bg ~flex_direction:Row ~justify_content:Space_between
    ~align_items:Center
    ~size:{ width = pct 100; height = auto }
    ~padding:(padding_lrtb ~l:2 ~r:2 ~t:0 ~b:0)
    [ left; right ]

let view_footer_help_overlay m =
  if not m.show_help then empty
  else
    let section_title title =
      text ~style:(Ansi.Style.make ~fg:accent ~bold:true ()) title
    in
    let item key desc =
      box ~flex_direction:Row ~gap:(gap 1) ~align_items:Center
        ~size:{ width = pct 100; height = auto }
        [
          text ~style:(Ansi.Style.make ~fg:label_fg ~bold:true ())
            (Printf.sprintf "[%s]" key);
          text ~style:(Ansi.Style.make ~fg:hint_fg ()) desc;
        ]
    in
    let panel_width = if m.viewport_width < 80 then pct 96 else pct 82 in
    let panel_height = if m.viewport_height < 24 then pct 86 else pct 72 in
    box ~position:Absolute ~inset:(inset 0) ~z_index:20 ~background:overlay_bg
      ~justify_content:Center ~align_items:Center
      ~size:{ width = pct 100; height = pct 100 }
      [
        box ~border:true ~border_style:Border.rounded ~border_color:border_focused
          ~background:chrome_bg ~flex_direction:Column ~gap:(gap 1)
          ~size:{ width = panel_width; height = panel_height }
          ~padding:(padding_lrtb ~l:1 ~r:1 ~t:0 ~b:1)
          [
            box ~flex_direction:Row ~justify_content:Space_between
              ~align_items:Center
              ~size:{ width = pct 100; height = auto }
              [
                text ~style:(Ansi.Style.make ~fg:Ansi.Color.white ~bold:true ())
                  "Keybindings";
                text ~style:(Ansi.Style.make ~fg:hint_fg ()) "Esc or ? to close";
              ];
            scroll_box ~id:help_scroll_id ~scroll_y:true ~scroll_x:false
              ~flex_grow:1. ~size:{ width = pct 100; height = auto }
              ~padding:(padding_lrtb ~l:1 ~r:1 ~t:0 ~b:0) ~flex_direction:Column
              ~gap:(gap 1)
              [
                box ~flex_direction:Column ~gap:(gap 1)
                  [
                    section_title "Navigation";
                    item "j / k" "Focus next / previous cell";
                    item "Up / Down" "Focus next / previous cell";
                    item "J / K" "Move cell down / up";
                  ];
                box ~flex_direction:Column ~gap:(gap 1)
                  [
                    section_title "Execution";
                    item "Enter" "Run focused code cell";
                    item "Ctrl-A" "Run all code cells";
                    item "Ctrl-C" "Interrupt execution";
                  ];
                box ~flex_direction:Column ~gap:(gap 1)
                  [
                    section_title "Cell management";
                    item "a / t" "Insert code / text cell";
                    item "d" "Delete focused cell";
                    item "m" "Toggle focused cell kind";
                    item "c / Ctrl-L" "Clear focused / all outputs";
                  ];
                box ~flex_direction:Column ~gap:(gap 1)
                  [
                    section_title "File and session";
                    item "s / Ctrl-S" "Save notebook";
                    item "q" "Quit (double press if modified)";
                    item "Esc" "Dismiss footer message";
                    item "?" "Toggle this help panel";
                  ];
                box ~flex_direction:Column ~gap:(gap 1)
                  [
                    section_title "Editing mode";
                    item "Esc" "Exit editor";
                    item "Ctrl-Enter" "Submit and run code cell";
                    item "Ctrl-S" "Save notebook";
                  ];
              ];
          ];
      ]

let view m =
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = pct 100 }
    [
      view_header m;
      scroll_box ~id:scroll_box_id ~scroll_y:true ~scroll_x:false ~flex_grow:1.
        ~autofocus:true
        ~size:{ width = pct 100; height = auto }
        ~flex_direction:Column ~gap:(gap 1)
        ~padding:(padding_lrtb ~l:1 ~r:1 ~t:1 ~b:1)
        (view_cells m);
      view_footer m;
      view_footer_help_overlay m;
    ]

(* ───── Subscriptions ───── *)

let subscriptions model =
  Sub.batch
    [
      Sub.on_tick (fun ~dt -> Tick dt);
      Sub.on_resize (fun ~width ~height -> Resize (width, height));
      (* Use on_key_all for all bindings because the scroll_box consumes
         j/k/Up/Down via its scroll bar before on_key sees them. *)
      Sub.on_key_all (fun ev ->
          let data = Event.Key.data ev in
          if model.show_help then
            match data.key with
            | Escape -> Some Toggle_help
            | Char c when char_eq '?' c -> Some Toggle_help
            | _ -> None
          else
            match model.mode with
            | Editing -> (
                if data.modifier.ctrl then
                  match data.key with
                  | Char c when char_eq 'a' c -> Some Execute_all
                  | Char c when char_eq 's' c -> Some Save
                  | Char c when char_eq 'c' c -> Some Interrupt
                  | Char c when char_eq 'l' c -> Some Clear_all
                  | _ -> None
                else
                  match data.key with
                  | Char c when char_eq '?' c -> Some Toggle_help
                  | _ -> None)
            | Normal -> (
                if data.modifier.ctrl then
                  match data.key with
                  | Char c when char_eq 'a' c -> Some Execute_all
                  | Char c when char_eq 's' c -> Some Save
                  | Char c when char_eq 'c' c -> Some Interrupt
                  | Char c when char_eq 'l' c -> Some Clear_all
                  | _ -> None
                else
                  match data.key with
                  | Char c when char_eq 'j' c -> Some Focus_next
                  | Char c when char_eq 'k' c -> Some Focus_prev
                  | Char c when char_eq 'J' c -> Some Move_down
                  | Char c when char_eq 'K' c -> Some Move_up
                  | Char c when char_eq 'e' c -> Some Enter_edit
                  | Char c when char_eq 'i' c -> Some Enter_edit
                  | Char c when char_eq 'a' c -> Some Insert_code_below
                  | Char c when char_eq 't' c -> Some Insert_text_below
                  | Char c when char_eq 'd' c -> Some Delete_focused
                  | Char c when char_eq 'm' c -> Some Toggle_cell_kind
                  | Char c when char_eq 'c' c -> Some Clear_focused
                  | Char c when char_eq 's' c -> Some Save
                  | Char c when char_eq 'q' c -> Some Quit
                  | Char c when char_eq '?' c -> Some Toggle_help
                  | Down -> Some Focus_next
                  | Up -> Some Focus_prev
                  | Enter -> Some Execute_focused
                  | Escape -> Some Dismiss_message
                  | _ -> None));
      (* Escape in editing mode: textarea does not consume it, so on_key
         works. *)
      Sub.on_key (fun ev ->
          match model.mode with
          | Editing when not model.show_help -> (
              match (Event.Key.data ev).key with
              | Escape -> Some Exit_edit
              | _ -> None)
          | Editing | Normal -> None);
    ]

(* ───── Run ───── *)

let run ~create_kernel path =
  let init () = init ~create_kernel ~path () in
  run { init; update; view; subscriptions }
