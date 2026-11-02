(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Artifact versions across runs. One run publishes a dataset, a second uses its
   latest version and publishes a result under an alias, and the store tells
   which run produced and which used each version. *)

open Munin

let write path text =
  Out_channel.with_open_bin path (fun oc -> output_string oc text)

let prepare store =
  Session.within
    (Session.start ~store ~experiment:"pipeline" ~name:"prepare-data" ())
  @@ fun s ->
  let path = Filename.temp_file "measurements" ".csv" in
  write path "wavelength,flux\n450.0,1.23\n550.0,2.45\n650.0,1.87\n";
  Session.log_artifact s ~name:"measurements"
    ~metadata:[ ("rows", `Int 3) ]
    path

let analyse store =
  Session.within
    (Session.start ~store ~experiment:"pipeline" ~name:"analyse" ())
  @@ fun s ->
  let data = Option.get (Store.artifact store "measurements" `Latest) in
  Session.use_artifact s data;
  let rows =
    In_channel.with_open_bin (Artifact.path data) In_channel.input_all
  in
  let path = Filename.temp_file "result" ".txt" in
  write path
    (Printf.sprintf "rows read: %d\npeak: 550.0\n"
       (List.length (String.split_on_char '\n' (String.trim rows)) - 1));
  let result = Session.log_artifact s ~name:"analysis" path in
  Store.set_alias store result "reviewed";
  Session.id s

let () =
  let store = Store.open_ () in
  let data = prepare store in
  Printf.printf "published %s %d (%s)\n" (Artifact.name data)
    (Artifact.version data) (Artifact.digest data);
  let analysis_run = analyse store in
  let reviewed =
    Option.get (Store.artifact store "analysis" (`Alias "reviewed"))
  in
  Printf.printf "analysis %d, alias reviewed, produced by %s\n"
    (Artifact.version reviewed)
    (Artifact.run reviewed);
  let uses =
    History.uses (History.read (Option.get (Store.run store analysis_run)))
  in
  List.iter
    (fun (name, n) -> Printf.printf "%s used %s %d\n" analysis_run name n)
    uses;
  print_endline "versions:";
  List.iter
    (fun a ->
      Printf.printf "  %s %d, %d bytes, from %s\n" (Artifact.name a)
        (Artifact.version a) (Artifact.size_bytes a) (Artifact.run a))
    (Store.artifacts store)
