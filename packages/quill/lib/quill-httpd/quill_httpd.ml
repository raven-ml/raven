(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Quill

let err_file_not_found : _ format = "Error: %s not found\n%!"

(* ───── File I/O ───── *)

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

(* ───── Server state ───── *)

type state = {
  mutable session : Session.t;
  mutable kernel : Kernel.t;
  path : string;
  mutex : Mutex.t;
  mutable ws : Httpd.ws option;
}

let locked st f =
  Mutex.lock st.mutex;
  Fun.protect ~finally:(fun () -> Mutex.unlock st.mutex) f

let send st msg =
  let preview =
    if String.length msg > 120 then String.sub msg 0 120 ^ "…" else msg
  in
  Printf.eprintf "[ws:send] %s\n%!" preview;
  match st.ws with Some ws -> Httpd.ws_send ws msg | None -> ()

let send_undo_redo st =
  send st
    (Protocol.undo_redo_to_json
       ~can_undo:(Session.can_undo st.session)
       ~can_redo:(Session.can_redo st.session))

let cells_with_status st =
  List.map
    (fun c -> (c, Session.cell_status (Cell.id c) st.session))
    (Doc.cells (Session.doc st.session))

let send_notebook st =
  send st
    (Protocol.notebook_to_json ~cells:(cells_with_status st)
       ~can_undo:(Session.can_undo st.session)
       ~can_redo:(Session.can_redo st.session))

(* ───── Execution ───── *)

let execute_cell_ids st cell_ids =
  locked st (fun () ->
      List.iter
        (fun cell_id ->
          st.session <- Session.mark_queued cell_id st.session;
          send st (Protocol.cell_status_to_json ~cell_id Session.Queued))
        cell_ids);
  List.iter
    (fun cell_id ->
      let source =
        locked st (fun () ->
            match Doc.find cell_id (Session.doc st.session) with
            | Some (Cell.Code { source; _ }) ->
                st.session <- Session.clear_outputs cell_id st.session;
                st.session <- Session.mark_running cell_id st.session;
                send st (Protocol.cell_status_to_json ~cell_id Session.Running);
                Some source
            | _ -> None)
      in
      match source with
      | Some code -> st.kernel.execute ~cell_id ~code
      | None -> ())
    cell_ids

(* ───── Kernel event handler ───── *)

let on_kernel_event st = function
  | Kernel.Output { cell_id; output } ->
      let kind =
        match output with
        | Cell.Stdout _ -> "stdout"
        | Stderr _ -> "stderr"
        | Error _ -> "error"
        | Display _ -> "display"
      in
      Printf.eprintf "[kernel] output %s for %s\n%!" kind cell_id;
      locked st (fun () ->
          st.session <- Session.apply_output cell_id output st.session;
          send st (Protocol.cell_output_to_json ~cell_id output))
  | Kernel.Finished { cell_id; success } ->
      Printf.eprintf "[kernel] finished %s success=%b\n%!" cell_id success;
      locked st (fun () ->
          st.session <- Session.finish_execution cell_id ~success st.session;
          match Doc.find cell_id (Session.doc st.session) with
          | Some cell ->
              let status = Session.cell_status cell_id st.session in
              send st (Protocol.cell_updated_to_json cell status)
          | None ->
              Printf.eprintf "[kernel] cell %s not found after finish!\n%!"
                cell_id)
  | Kernel.Status_changed _ -> ()

(* ───── Client message handler ───── *)

let execute_async st cell_ids =
  st.session <- Session.checkpoint st.session;
  ignore (Thread.create (fun () -> execute_cell_ids st cell_ids) () : Thread.t)

let handle_client_msg st = function
  | Protocol.Update_source { cell_id; source } ->
      st.session <- Session.update_source cell_id source st.session
  | Protocol.Checkpoint ->
      st.session <- Session.checkpoint st.session;
      send_undo_redo st
  | Protocol.Execute_cell { cell_id } -> execute_async st [ cell_id ]
  | Protocol.Execute_cells { cell_ids } -> execute_async st cell_ids
  | Protocol.Execute_all ->
      let cell_ids =
        List.filter_map
          (fun c ->
            match c with Cell.Code { id; _ } -> Some id | Text _ -> None)
          (Doc.cells (Session.doc st.session))
      in
      execute_async st cell_ids
  | Protocol.Interrupt | Protocol.Complete _ ->
      assert false (* dispatched by [handle_msg] before reaching here *)
  | Protocol.Insert_cell { pos; kind } ->
      let cell =
        match kind with `Code -> Cell.code "" | `Text -> Cell.text ""
      in
      st.session <- Session.insert_cell ~pos cell st.session;
      let status = Session.cell_status (Cell.id cell) st.session in
      send st (Protocol.cell_inserted_to_json ~pos cell status);
      send_undo_redo st
  | Protocol.Delete_cell { cell_id } ->
      st.session <- Session.remove_cell cell_id st.session;
      send st (Protocol.cell_deleted_to_json ~cell_id);
      send_undo_redo st
  | Protocol.Move_cell { cell_id; pos } ->
      st.session <- Session.move_cell cell_id ~pos st.session;
      send st (Protocol.cell_moved_to_json ~cell_id ~pos);
      send_undo_redo st
  | Protocol.Set_cell_kind { cell_id; kind } ->
      st.session <- Session.set_cell_kind cell_id kind st.session;
      (match Doc.find cell_id (Session.doc st.session) with
      | Some cell ->
          let status = Session.cell_status cell_id st.session in
          send st (Protocol.cell_updated_to_json cell status)
      | None -> ());
      send_undo_redo st
  | Protocol.Clear_outputs { cell_id } -> (
      st.session <- Session.clear_outputs cell_id st.session;
      match Doc.find cell_id (Session.doc st.session) with
      | Some cell ->
          let status = Session.cell_status cell_id st.session in
          send st (Protocol.cell_updated_to_json cell status)
      | None -> ())
  | Protocol.Clear_all_outputs ->
      st.session <- Session.clear_all_outputs st.session;
      send_notebook st
  | Protocol.Save ->
      st.session <- Session.checkpoint st.session;
      let content =
        Quill_markdown.to_string_with_outputs (Session.doc st.session)
      in
      write_file st.path content;
      send st (Protocol.saved_to_json ())
  | Protocol.Undo ->
      st.session <- Session.undo st.session;
      send_notebook st
  | Protocol.Redo ->
      st.session <- Session.redo st.session;
      send_notebook st

(* ───── WebSocket handler ───── *)

let client_msg_name = function
  | Protocol.Update_source _ -> "update_source"
  | Checkpoint -> "checkpoint"
  | Execute_cell _ -> "execute_cell"
  | Execute_cells _ -> "execute_cells"
  | Execute_all -> "execute_all"
  | Interrupt -> "interrupt"
  | Insert_cell _ -> "insert_cell"
  | Delete_cell _ -> "delete_cell"
  | Move_cell _ -> "move_cell"
  | Set_cell_kind _ -> "set_cell_kind"
  | Clear_outputs _ -> "clear_outputs"
  | Clear_all_outputs -> "clear_all_outputs"
  | Save -> "save"
  | Undo -> "undo"
  | Redo -> "redo"
  | Complete _ -> "complete"

let handle_msg st msg =
  Printf.eprintf "[ws:recv] %s\n%!" (client_msg_name msg);
  match msg with
  | Protocol.Interrupt -> st.kernel.interrupt ()
  | Protocol.Complete { request_id; code; pos } ->
      let items = st.kernel.complete ~code ~pos in
      locked st (fun () ->
          send st (Protocol.completions_to_json ~request_id items))
  | msg -> locked st (fun () -> handle_client_msg st msg)

let ws_handler st _req ws =
  Printf.eprintf "[ws] connection opened\n%!";
  locked st (fun () ->
      st.ws <- Some ws;
      send_notebook st);
  let rec loop () =
    match Httpd.ws_recv ws with
    | Some msg -> (
        match Protocol.client_msg_of_json msg with
        | Ok client_msg ->
            handle_msg st client_msg;
            loop ()
        | Error err ->
            locked st (fun () -> send st (Protocol.error_to_json err));
            loop ())
    | None ->
        Printf.eprintf "[ws] connection closed\n%!";
        locked st (fun () ->
            (* Only clear if we're still the active connection *)
            match st.ws with
            | Some w when w == ws -> st.ws <- None
            | _ -> ())
  in
  loop ()

(* ───── Entry point ───── *)

let serve ?(addr = "127.0.0.1") ?(port = 8888) path =
  if not (Sys.file_exists path) then (
    Printf.eprintf err_file_not_found path;
    exit 1);
  let md = read_file path in
  let doc = Quill_markdown.of_string md in
  let session = Session.create doc in
  let mutex = Mutex.create () in
  let st =
    {
      session;
      kernel =
        {
          execute = (fun ~cell_id:_ ~code:_ -> ());
          interrupt = ignore;
          complete = (fun ~code:_ ~pos:_ -> []);
          type_at = None;
          diagnostics = None;
          status = (fun () -> Kernel.Starting);
          shutdown = ignore;
        };
      path;
      mutex;
      ws = None;
    }
  in
  let on_event ev = on_kernel_event st ev in
  st.kernel <- Quill_raven.create ~on_event;
  let server = Httpd.create ~addr ~port () in
  Httpd.route server GET "/" (fun _req ->
      Httpd.response
        ~headers:[ ("Content-Type", "text/html; charset=utf-8") ]
        Assets.index_html);
  Httpd.static server ~prefix:"/assets/" ~loader:Assets.lookup ();
  Httpd.websocket server "/ws" (ws_handler st);
  Printf.printf "Quill web: http://%s:%d\n%!" addr port;
  Httpd.run server;
  st.kernel.shutdown ()
