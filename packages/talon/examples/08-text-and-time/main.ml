(* Text and temporal expressions over a table of recording sessions. *)

open Talon

let at s = Option.get (Time.of_s (Int64.of_int s))

let sessions =
  Talon.v
    [
      ( "file",
        Column.v Type.string
          [| "sub-01_ses-01.nwb"; "sub-01_ses-02.nwb"; "sub-02_ses-01.nwb" |] );
      ( "start",
        Column.v
          (Type.datetime ~zone:"UTC" Type.Ms)
          [| at 1710495000; at 1710581523; at 1711010000 |] );
      ( "logged",
        Column.v Type.string [| "15/03/2024"; "16/03/2024"; "21/03/2024" |] );
    ]

let file = Col.string "file"
let start = Col.instant "start"
let show q = Format.printf "%a@.@." Talon.pp (Error.get_ok (Query.run q))

let () =
  (* Text counts and slices Unicode scalar values. *)
  show
    Query.(
      of_table sessions
      |> select
           Expr.
             [
               "subject" := Str.slice ~offset:0 ~length:6 file;
               "parts" := Str.split "_" (Str.replace ".nwb" ~by:"" file);
               "first" := Str.matches (Str.prefix "sub-01") file;
               "length" := Str.length file;
             ]);

  (* Calendar fields, arithmetic on instants, and spans between them. *)
  show
    Query.(
      of_table sessions
      |> select
           Expr.
             [
               "weekday" := Temporal.field `Weekday start;
               "day" := Temporal.floor (Time.Days 1) start;
               "end" := Temporal.add start (span (Time.Span.minutes 90));
               "since" := Temporal.diff start (over (min start));
             ]);

  (* Text read and written in a format. *)
  show
    Query.(
      of_table sessions
      |> select
           Expr.
             [
               "logged" :=
                 Temporal.parse "%d/%m/%Y" Type.date (Col.string "logged");
               "label" := Temporal.format "%Y-%m-%d %H:%M" start;
             ])
