(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next

let question (w : Workload.t) (q : Workload.question) =
  Thumper.group q.name
    [
      Thumper.bench_with_setup
        ~setup:(fun () -> q.query (Workload.load w))
        "talon"
        (fun query -> Error.get_ok (Query.run query));
    ]

let size ~data workload size =
  let w = workload ~data size in
  match Workload.missing w with
  | [] -> Some (Thumper.group size (List.map (question w) w.questions))
  | _ ->
      Printf.eprintf "%s: no data under %s, skipped\n%!" w.id data;
      None

(* The slowest questions take seconds per call at the gate sizes, past thumper's
   default deadline of 10 s per case. *)
let config = Thumper.Config.(default |> deadline 1800.)

let run suite ~sizes families =
  match Sys.getenv_opt "TALON_BENCH_DATA" with
  | None | Some "" ->
      prerr_endline
        "TALON_BENCH_DATA is unset: set it to the directory the data scripts \
         wrote";
      exit 2
  | Some data ->
      let family (name, workload) =
        Thumper.group name (List.filter_map (size ~data workload) sizes)
      in
      Thumper.run ~config suite (List.map family families)
