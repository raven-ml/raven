(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Event = Event
module Run = Run

let discover ?base_dir () =
  let dir = Option.value base_dir ~default:(Env.base_dir ()) in
  if not (Sys.file_exists dir) then []
  else
    Sys.readdir dir |> Array.to_list
    |> List.filter_map (fun entry ->
           let run_dir = Filename.concat dir entry in
           if Sys.is_directory run_dir then Run.load run_dir else None)
    |> List.sort (fun a b -> compare (Run.created_at b) (Run.created_at a))

let latest ?base_dir () =
  match discover ?base_dir () with
  | [] -> None
  | h :: _ -> Some h

let create_run = Run.create
