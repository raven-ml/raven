(* Build a table from OCaml values, inspect it, and query it. *)

open Talon

let day d = Option.get (Time.Date.of_civil (2024, 3, d))

let readings =
  Talon.v
    [
      ( "station",
        Column.v Type.string [| "oslo"; "oslo"; "lima"; "lima"; "pune" |] );
      ("day", Column.v Type.date (Array.map day [| 1; 2; 1; 2; 1 |]));
      ( "temp",
        Column.of_options Type.float32
          [| Some (-3.5); Some (-1.25); Some 22.; None; Some 31.5 |] );
      ( "rain",
        Column.of_tensor
          (Nx.create Nx.bool [| 5 |] [| false; true; false; false; true |]) );
    ]

let () =
  Format.printf "%a@.@." Talon.pp readings;

  (* A column reads back as OCaml values, a null as [None]. *)
  let temp = Talon.column readings "temp" in
  Format.printf "temp: %d null of %d@." (Column.null_count temp)
    (Column.length temp);
  Column.options Kind.float temp
  |> Array.iter (function
    | Some t -> Format.printf "  %g@." t
    | None -> Format.printf "  null@.");
  Format.printf "@.";

  (* A query describes a table; running it computes the rows. *)
  let warm =
    Query.(
      of_table readings
      |> filter Expr.(Col.float "temp" > float 0.)
      |> derive
           Expr.[ "temp_f" := (Col.float "temp" *. float 1.8) +. float 32. ])
  in
  Format.printf "%a@.@." Query.pp warm;
  let t = Error.get_ok (Query.run warm) in
  Format.printf "%a@.@." Talon.pp t;

  (* Values leave as OCaml values or as an nx tensor. *)
  let label =
    Expr.(
      const (Printf.sprintf "%s: %.1f°C")
      $ Col.string "station" $ Col.float "temp")
  in
  Array.iter print_endline (Error.get_ok (Query.values label warm));
  Format.printf "%a@." Nx.pp (Talon.to_tensor Nx.float32 [ "temp"; "temp_f" ] t)
