(* Plan problems raise when a verb is applied, before any data is read; failures
   in data are [Error] values. *)

open Talon

let flights =
  Talon.v
    [
      ("carrier", Column.v Type.string [| "AA"; "AA"; "DL" |]);
      ("origin", Column.v Type.string [| "JFK"; "LGA"; "JFK" |]);
      ("dep_delay", Column.v Type.int64 [| 4; 130; -2 |]);
    ]

let problems () =
  Query.(
    of_table flights
    |> aggregate ~by:[ "carier" ]
         Expr.
           [
             "mean_delay" := mean (Col.int "dep_dly");
             "late" := mean (Col.float "carrier");
           ])

let failure q =
  match Query.run q with
  | Ok _ -> ()
  | Error e -> Format.printf "%a@.@." Error.pp e

let () =
  (match problems () with
  | _ -> ()
  | exception Invalid_argument m -> Format.printf "%s@.@." m);
  failure
    Query.(
      of_table flights
      |> derive Expr.[ "delay" := cast Type.int8 (Col.int "dep_delay") ]);
  failure
    Query.(
      of_table flights
      |> aggregate ~by:[ "carrier" ]
           Expr.[ "origin" := only (Col.string "origin") ])
