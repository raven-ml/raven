(* From a CSV file to a feature matrix and labels as nx tensors. *)

open Talon

let ( let* ) = Result.bind
let delay = Col.int "dep_delay"
let standardize x = Expr.((x -. over (mean x)) /. over (std x))

let features flights =
  Query.(
    of_source flights
    |> filter Expr.(not (is_null delay))
    |> select
         Expr.
           [
             "distance" := standardize (cast Type.float64 (Col.int "distance"));
             "carrier_delay" := over ~by:[ "carrier" ] (mean delay);
             "month" := Col.int "month";
             "late" := Col.int "arr_delay" > int 15;
           ])

let main () =
  let* flights = Talon_csv.file ~nulls:[ "NA" ] "flights.csv" in
  let* t = Query.run (features flights) in

  (* Shuffle every column at once, then copy the features into one matrix. *)
  let t = Talon.take (Nx.Rng.permutation (Nx.Rng.key 0) (Talon.rows t)) t in
  Format.printf "%a@.@."
    (Talon.pp_with { Talon.limits with head = 3; tail = 2 })
    t;
  let x =
    Talon.to_tensor Nx.float32 [ "distance"; "carrier_delay"; "month" ] t
  in
  let y = Column.to_tensor Nx.bit (Talon.column t "late") in
  Format.printf "x: %a@.y: %a@." Nx.pp_shape (Nx.shape x) Nx.pp_shape
    (Nx.shape y);
  Ok ()

let () = Error.get_ok (main ())
