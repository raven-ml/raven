(* Load a CSV and a Parquet file, filter, aggregate, join and sort. *)

open Talon

let ( let* ) = Result.bind
let delay = Col.int "dep_delay"

let late_by_carrier flights carriers =
  Query.(
    of_source flights
    |> filter Expr.(delay > int 15)
    |> aggregate ~by:[ "carrier" ]
         Expr.[ "mean_delay" := mean delay; "flights" := rows ]
    |> join ~on:(Join.keys [ "carrier" ]) ~each_left:One (of_source carriers)
    |> sort [ Order.desc "mean_delay" ])

let main () =
  let* flights = Talon_csv.file ~nulls:[ "NA" ] "flights.csv" in
  let* carriers = Talon_parquet.file "carriers.parquet" in
  let q = late_by_carrier flights carriers in

  (* Printing a query reads no data. The optimized plan is the one a run runs:
     it reads two of the CSV file's columns. *)
  Format.printf "%a@.@." Query.pp q;
  Format.printf "%a@.@." Query.pp (Query.optimize q);

  let* t = Query.run q in
  Format.printf "%a@." Talon.pp t;
  Ok ()

let () = Error.get_ok (main ())
