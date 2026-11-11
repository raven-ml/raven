(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin

type screen = Runs | Run of string option
type model = Model.t
type msg = Model.msg

(* The poll: one in flight. A poll asked for while one runs runs after it.
   A timer's poll lists the store; another reads what the screen lacks. *)
let poll (m : Model.t) ~list =
  match m.poll with
  | Reading | Again -> ({ m with poll = Again }, Mosaic.Cmd.none)
  | Idle ->
      let store = m.store and wants = Model.wants m and d = m.data and now = m.clock () in
      ( { m with poll = Reading },
        Mosaic.Cmd.perform (fun dispatch ->
            dispatch (Model.Polled (Data.poll store ~now ~list wants d))) )

(* A change that changes what the screen draws asks for a poll. *)
let after (before : Model.t) (m : Model.t) =
  if Data.same (Model.wants before) (Model.wants m) then (m, Mosaic.Cmd.none)
  else poll m ~list:false

(* A background lighter than mid-grey is a light terminal's. *)
let scheme (r, g, b) =
  let luma = (0.2126 *. Float.of_int r) +. (0.7152 *. Float.of_int g) +. (0.0722 *. Float.of_int b) in
  if luma > 127.5 then `Light else `Dark

let update (msg : Model.msg) (m : Model.t) =
  match msg with
  | Tick -> poll m ~list:true
  | Polled d ->
      (* What was read can change what the screen draws: a cursor chosen, a
         metric found. Then what it lacks is read at once. *)
      let again = m.poll = Again in
      let m' = Model.polled d { m with poll = Idle } in
      if again then poll m' ~list:false else after m m'
  | Key Quit -> (m, Mosaic.Cmd.quit)
  | Key a -> after m (Model.act a m)
  | Resized (w, h) -> after m (Model.resized w h m)
  | Sized (s, w, h) -> after m (Model.sized s w h m)
  | Edited e -> after m (Model.edited e m)
  | Submitted -> after m (Model.act Keep_filter m)
  | Info_scrolled s -> (Model.scrolled_info s m, Mosaic.Cmd.none)
  | Rows_scrolled s -> after m (Model.scrolled_rows s m)
  | Strips_scrolled s -> (Model.scrolled_strips s m, Mosaic.Cmd.none)
  | Styling s -> (Model.styled s m, Mosaic.Cmd.none)
  | Palette p ->
      let look = { m.look with scheme = scheme p.background; background = Some p.background } in
      (Model.palette look m, Mosaic.Cmd.none)

let subscriptions (m : Model.t) =
  let keys = List.map (fun (k, a) -> (k, Model.Key a)) (Keys.bindings (Keys.mode m)) in
  Mosaic.Sub.batch
    [
      Mosaic.Sub.every ~id:"poll" 1. (fun _ -> Model.Tick);
      Mosaic.Sub.on_keys keys;
      Mosaic.Sub.on_resize (fun ~width ~height -> Model.Resized (width, height));
      Mosaic.Sub.on_palette (fun p -> Some (Model.Palette p));
      Mosaic.Sub.on_styling (fun s -> Model.Styling s);
    ]

let init ~clock store screen () =
  let cursor : Model.cursor =
    match screen with Runs -> Unchosen | Run (Some id) -> On id | Run None -> Waiting
  in
  let m = Model.make ~store ~clock ~scheme:`Dark ~cursor in
  let m, read = poll m ~list:true in
  (m, Mosaic.Cmd.batch [ read; Mosaic.Cmd.query_palette ])

let app ~clock store screen =
  { Mosaic.init = init ~clock store screen; update; view = View.view; subscriptions }

module Reduce = Reduce
module Plot = Plot
module Numbers = Numbers
module Facts = Facts
module Theme = Theme
module Watch = Watch

let run store screen =
  match screen with
  | Run (Some id) when Store.run store id = None ->
      Error (Printf.sprintf "no run %s in %s" id (Store.root store))
  | _ when not (Unix.isatty Unix.stdin && Unix.isatty Unix.stdout) ->
      Error "the dashboard needs a terminal"
  | _ ->
      let config = Matrix.config ~mouse:`Off ~fps:10. () in
      Mosaic.run ~config (app ~clock:Unix.gettimeofday store screen);
      Ok ()
