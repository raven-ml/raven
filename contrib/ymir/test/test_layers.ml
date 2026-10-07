(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Layers: only the module holding the solves names jera, so every other
   function of the geometry is nx arithmetic, and no module names ymir.fits,
   so a transform links no file reader. *)

open Windtrap

(* [modules ()] is each library module with the modules it names, from
   [ocamldep -modules]. *)
let modules () =
  In_channel.with_open_text "layers.deps" In_channel.input_lines
  |> List.filter_map (fun line ->
      match String.index_opt line ':' with
      | None -> None
      | Some i ->
          let file = Filename.basename (String.sub line 0 i) in
          let rest = String.sub line (i + 1) (String.length line - i - 1) in
          let names =
            String.split_on_char ' ' rest |> List.filter (fun s -> s <> "")
          in
          Some (file, names))

let () =
  exit
    (run "Layers"
       [
         test "only solve.ml names Jera" (fun () ->
             let naming =
               List.filter_map
                 (fun (file, names) ->
                   if List.mem "Jera" names then Some file else None)
                 (modules ())
             in
             equal (list string) [ "solve.ml" ] naming);
         test "no module names Ymir_fits" (fun () ->
             let naming =
               List.filter_map
                 (fun (file, names) ->
                   if List.mem "Ymir_fits" names then Some file else None)
                 (modules ())
             in
             equal (list string) [] naming);
       ])
