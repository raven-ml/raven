(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reading a store: its runs with their status, provenance and params, the
   newest run's metrics, and the artifact versions. Run the other examples first
   to fill the store. *)

open Munin

let status : Run.status -> string = function
  | `Running -> "running"
  | `Silent -> "silent"
  | `Crashed -> "crashed"
  | `Finished -> "finished"
  | `Failed e -> "failed: " ^ e
  | `Killed -> "killed"

let () =
  let store = Store.open_ () in
  let runs = Store.runs store in
  Printf.printf "store %s: %d runs\n" (Store.root store) (List.length runs);
  List.iter
    (fun r ->
      Printf.printf "  %s  %-12s %-20s %s\n" (Run.id r) (Run.experiment r)
        (Option.value ~default:"-" (Run.name r))
        (status (Run.status r)))
    runs;
  (match runs with
  | [] -> ()
  | r :: _ ->
      let p = Run.provenance r in
      Printf.printf "\nnewest run %s\n" (Run.id r);
      Printf.printf "  command: %s\n" (String.concat " " p.command);
      Printf.printf "  host: %s\n" (Option.value ~default:"-" p.hostname);
      Option.iter
        (fun (g : Provenance.git) ->
          Printf.printf "  git: %s%s\n" g.commit
            (if g.dirty then " (dirty)" else ""))
        p.git;
      List.iter
        (fun (k, v) ->
          Printf.printf "  param %s = %s\n" k (Format.asprintf "%a" Value.pp v))
        (Run.params r);
      let h = History.read r in
      List.iter
        (fun key ->
          let st = Option.get (History.stats h key) in
          Printf.printf "  %s: %d samples, last %g at step %d\n" key st.count
            st.last.value st.last.step)
        (History.metrics h @ History.readings h));
  let versions = Store.artifacts store in
  Printf.printf "\nartifacts: %d versions\n" (List.length versions);
  List.iter
    (fun a ->
      Printf.printf "  %s %d (%d bytes)\n" (Artifact.name a)
        (Artifact.version a) (Artifact.size_bytes a))
    versions
