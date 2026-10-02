(* Join kinds, conditions and assertions. *)

open Talon

let trials =
  Talon.v
    [
      ("subject", Column.v Type.string [| "s1"; "s1"; "s2"; "s3"; "s3" |]);
      ("trial", Column.v Type.int16 [| 1; 2; 1; 1; 2 |]);
      ("rt", Column.v Type.float64 [| 0.412; 0.388; 0.501; 0.450; 0.433 |]);
    ]

let subjects =
  Talon.v
    [
      ("id", Column.v Type.string [| "s1"; "s2"; "s4" |]);
      ("age", Column.v Type.uint8 [| 24; 31; 27 |]);
    ]

let show name q =
  match Query.run q with
  | Ok t -> Format.printf "%s:@.%a@.@." name Talon.pp t
  | Error e -> Format.printf "%s:@.%a@.@." name Error.pp e

let () =
  let trials = Query.of_table trials and subjects = Query.of_table subjects in
  let on = Join.eq "subject" "id" in
  show "inner" Query.(trials |> join ~on subjects);
  show "left" Query.(trials |> join ~kind:Left ~on subjects);
  show "full" Query.(trials |> join ~kind:Full ~on subjects);
  show "semi" Query.(trials |> join ~kind:Semi ~on subjects);
  show "anti" Query.(trials |> join ~kind:Anti ~on subjects);

  (* An assertion on the number of matches fails the run, naming the row. *)
  show "each_left:One" Query.(trials |> join ~on ~each_left:One subjects)
