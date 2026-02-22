(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic
open Quill

(* ───── Model ───── *)

type model = {
  session : Session.t;
  kernel : Kernel.t;
  event_queue : Kernel.event Queue.t;
  path : string;
  focus : int;
  dirty : bool;
  last_error : string option;
  last_mtime : float;
  reload_acc : float;
  confirm_quit : bool;
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
  | Dismiss_error

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
let cell_bg_focused = Ansi.Color.of_rgb 30 30 38
let reload_interval = 1.0
let template = "# Untitled\n\n```ocaml\n\n```\n"
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
      dirty = false;
      last_error = None;
      last_mtime;
      reload_acc = 0.;
      confirm_quit = false;
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
  { m with session; dirty = true; last_error = None }

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
  { m with session = !session; dirty = true; last_error = None }

(* ───── Update ───── *)

let update msg m =
  let m =
    match msg with
    | Quit | Tick _ | Dismiss_error -> m
    | _ -> { m with confirm_quit = false }
  in
  match msg with
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
          ({ m with last_error = Some "Cannot execute a text cell" }, Cmd.none)
      | None -> ({ m with last_error = Some "No cell to execute" }, Cmd.none))
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
  | Save ->
      let session = Session.checkpoint m.session in
      let m = { m with session } in
      let doc = Session.doc m.session in
      let content = Quill_markdown.to_string_with_outputs doc in
      write_file m.path content;
      let last_mtime = get_mtime m.path in
      ({ m with dirty = false; last_error = None; last_mtime }, Cmd.none)
  | Quit ->
      if m.dirty && not m.confirm_quit then
        ( {
            m with
            confirm_quit = true;
            last_error =
              Some "Unsaved changes. Press q again to quit, s to save.";
          },
          Cmd.none )
      else (
        m.kernel.shutdown ();
        (m, Cmd.quit))
  | Tick dt ->
      let reload_acc = m.reload_acc +. dt in
      if reload_acc >= reload_interval then
        let m = check_reload { m with reload_acc = 0. } in
        (m, Cmd.none)
      else ({ m with reload_acc }, Cmd.none)
  | Dismiss_error -> ({ m with last_error = None }, Cmd.none)

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
          spinner ~preset:Dots ~color:accent ();
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
    else null
  in
  box ~background:chrome_bg ~flex_direction:Row ~justify_content:Space_between
    ~align_items:Center
    ~size:{ width = pct 100; height = auto }
    ~padding:(padding_lrtb ~l:2 ~r:2 ~t:0 ~b:0)
    [ left; center; right ]

let view_error_bar msg =
  box ~background:error_bg ~border:true ~border_sides:[ `Left ]
    ~border_style:Grid.Border.heavy ~border_color:error_fg
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

let view_code_cell ~index ~is_focused ~status source outputs =
  let border_color = if is_focused then border_focused else border_unfocused in
  let num = index + 1 in
  let status_indicator =
    match status with
    | Session.Running -> " \xe2\x80\xa6"
    | Session.Queued -> " \xe2\x97\x8b"
    | Session.Idle -> if outputs <> [] then " \xe2\x9c\x93" else ""
  in
  let title = Printf.sprintf " %d%s " num status_indicator in
  let source_view =
    box
      ~padding:(padding_lrtb ~l:1 ~r:1 ~t:1 ~b:0)
      ~size:{ width = pct 100; height = auto }
      [ code ~filetype:"ocaml" source ]
  in
  let status_row =
    match status with
    | Session.Running ->
        box ~flex_direction:Row ~gap:(gap 1) ~align_items:Center
          ~padding:(padding_lrtb ~l:1 ~r:1 ~t:0 ~b:0)
          ~size:{ width = pct 100; height = auto }
          [
            spinner ~preset:Dots ~color:accent ();
            text
              ~style:(Ansi.Style.make ~fg:accent_dim ~italic:true ())
              "evaluating";
          ]
    | _ -> null
  in
  let output_section =
    if outputs = [] then null
    else
      box ~flex_direction:Column ~border:true ~border_sides:[ `Top ]
        ~border_style:Grid.Border.single ~border_color:border_unfocused
        ~size:{ width = pct 100; height = auto }
        ~padding:(padding 1)
        (List.map view_output outputs)
  in
  box ~flex_direction:Column ~border:true ~border_color
    ~border_style:Grid.Border.rounded ~title ~title_alignment:`Left
    ?background:(if is_focused then Some cell_bg_focused else None)
    ~size:{ width = pct 100; height = auto }
    [ source_view; status_row; output_section ]

let view_text_cell ~is_focused source =
  box
    ?background:(if is_focused then Some cell_bg_focused else None)
    ~size:{ width = pct 100; height = auto }
    ~padding:(padding_lrtb ~l:2 ~r:2 ~t:0 ~b:0)
    [ markdown source ]

let view_cell ~index ~focus m cell =
  let is_focused = index = focus in
  match cell with
  | Cell.Code { id; source; outputs; _ } ->
      let status = Session.cell_status id m.session in
      view_code_cell ~index ~is_focused ~status source outputs
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
    List.mapi (fun index cell -> view_cell ~index ~focus:m.focus m cell) cells

let view_footer m =
  let key_style = Ansi.Style.make ~fg:label_fg ~bold:true () in
  let desc_style = Ansi.Style.make ~fg:hint_fg () in
  let sep = text ~style:desc_style " \xc2\xb7 " in
  let key k d =
    fragment
      [
        text ~style:key_style k; text ~style:desc_style (Printf.sprintf " %s" d);
      ]
  in
  let keys_row =
    if has_running m then
      box ~flex_direction:Row
        ~size:{ width = pct 100; height = auto }
        ~flex_wrap:Wrap
        [
          key "Ctrl-C" "interrupt";
          sep;
          key "j/k" "navigate";
          sep;
          key "Enter" "run";
          sep;
          key "s" "save";
          sep;
          key "q" "quit";
        ]
    else
      box ~flex_direction:Row
        ~size:{ width = pct 100; height = auto }
        ~flex_wrap:Wrap
        [
          key "j/k" "navigate";
          sep;
          key "J/K" "move";
          sep;
          key "Enter" "run";
          sep;
          key "Ctrl-A" "run all";
          sep;
          key "a" "code";
          sep;
          key "t" "text";
          sep;
          key "d" "delete";
          sep;
          key "s" "save";
          sep;
          key "q" "quit";
        ]
  in
  let error_row =
    match m.last_error with Some e -> view_error_bar e | None -> null
  in
  box ~flex_direction:Column ~background:chrome_bg
    ~size:{ width = pct 100; height = auto }
    ~padding:(padding_lrtb ~l:2 ~r:2 ~t:0 ~b:0)
    [ error_row; keys_row ]

let view m =
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = pct 100 }
    [
      view_header m;
      scroll_box ~scroll_y:true ~scroll_x:false ~flex_grow:1. ~autofocus:true
        ~size:{ width = pct 100; height = auto }
        ~flex_direction:Column ~gap:(gap 1)
        ~padding:(padding_lrtb ~l:1 ~r:1 ~t:1 ~b:1)
        (view_cells m);
      view_footer m;
    ]

(* ───── Subscriptions ───── *)

let subscriptions _model =
  Sub.batch
    [
      Sub.on_tick (fun ~dt -> Tick dt);
      Sub.on_key_all (fun ev ->
          let data = Mosaic_ui.Event.Key.data ev in
          if data.modifier.ctrl then
            match data.key with
            | Char c when char_eq 'a' c -> Some Execute_all
            | Char c when char_eq 's' c -> Some Save
            | Char c when char_eq 'c' c -> Some Interrupt
            | Char c when char_eq 'l' c -> Some Clear_all
            | _ -> None
          else None);
      Sub.on_key (fun ev ->
          match (Mosaic_ui.Event.Key.data ev).key with
          | Char c when char_eq 'j' c -> Some Focus_next
          | Char c when char_eq 'k' c -> Some Focus_prev
          | Char c when char_eq 'J' c -> Some Move_down
          | Char c when char_eq 'K' c -> Some Move_up
          | Char c when char_eq 'a' c -> Some Insert_code_below
          | Char c when char_eq 't' c -> Some Insert_text_below
          | Char c when char_eq 'd' c -> Some Delete_focused
          | Char c when char_eq 'm' c -> Some Toggle_cell_kind
          | Char c when char_eq 'c' c -> Some Clear_focused
          | Char c when char_eq 's' c -> Some Save
          | Char c when char_eq 'q' c -> Some Quit
          | Down -> Some Focus_next
          | Up -> Some Focus_prev
          | Enter -> Some Execute_focused
          | Escape -> Some Dismiss_error
          | _ -> None);
    ]

(* ───── Run ───── *)

let run ~create_kernel path =
  let init () = init ~create_kernel ~path () in
  Mosaic_unix.run ~exit_on_ctrl_c:false { init; update; view; subscriptions }
