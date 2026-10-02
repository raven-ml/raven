(* Selectors and the compositions of Kit over a table of measurements. *)

open Talon

let batch name vs =
  Talon.v
    [
      ("site", Column.v Type.string (Array.map (fun _ -> name) vs));
      ("ph", Column.of_options Type.float64 (Array.map fst vs));
      ("temp", Column.v Type.float32 (Array.map snd vs));
      ("grade", Column.v Type.string [| "a"; "b"; "a" |]);
    ]

let north = batch "north" [| (Some 7.1, 12.5); (None, 13.); (Some 6.8, 11.75) |]

let south =
  batch "south" [| (Some 7.4, 18.); (Some 7.2, 17.5); (Some 7.4, 18.) |]

let show q = Format.printf "%a@.@." Talon.pp (Error.get_ok (Query.run q))

let () =
  let all = Query.(of_table north |> append (of_table south)) in

  (* One output per float column, read through one function. *)
  let standardize x = Expr.((x -. over (mean x)) /. over (std x)) in
  show
    Query.(
      all
      |> select
           Expr.
             [
               keep Sel.(names [ "site" ]);
               across Kind.float
                 Sel.(of_kind Kind.float)
                 (fun n x -> n := standardize x);
             ]);
  show (Kit.null_count all);
  show (Kit.describe all);
  show (Kit.value_counts "site" all);
  show (Kit.distinct (Kit.drop Sel.(names [ "ph" ]) all));
  show
    (Kit.top_k 2
       [ Order.desc "celsius" ]
       (Kit.rename [ ("temp", "celsius") ] all));

  (* A categorical column takes its dictionary from the data, then splits into
     one boolean column per category. *)
  let graded = Error.get_ok (Kit.categorize [ "grade" ] all) in
  show (Kit.one_hot "grade" graded)
