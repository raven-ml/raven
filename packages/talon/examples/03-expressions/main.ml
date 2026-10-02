(* Typed expressions over frames: row values, reductions, and reductions
   broadcast back to rows with [over]. *)

open Talon

let losses =
  Talon.v
    [
      ("run", Column.v Type.string [| "a"; "a"; "a"; "a"; "b"; "b"; "b"; "b" |]);
      ("step", Column.v Type.int32 [| 100; 200; 300; 400; 100; 200; 300; 400 |]);
      ( "loss",
        Column.of_options Type.float64
          [|
            Some 2.31;
            Some 1.87;
            Some 1.98;
            Some 1.55;
            Some 2.40;
            None;
            Some 1.71;
            Some 1.52;
          |] );
    ]

let loss = Col.float "loss"
let step = Col.int "step"
let per_run e = Expr.over ~by:[ "run" ] ~order:[ Order.asc "step" ] e
let show q = Format.printf "%a@.@." Talon.pp (Error.get_ok (Query.run q))

let () =
  (* Row expressions: one value per row. *)
  show
    Query.(
      of_table losses
      |> derive
           Expr.
             [
               "change" := loss -. per_run (shift 1 loss);
               "rank" := per_run (rank loss);
               "late" := step >= int 300 && not (is_null loss);
             ]);

  (* Reductions: one value per group. *)
  show
    Query.(
      of_table losses
      |> aggregate ~by:[ "run" ]
           Expr.
             [
               "evals" := count loss;
               "best" := min loss;
               "best_step" := first (if_ (loss = over (min loss)) step null);
               "improved" := last loss < first loss;
             ]);

  (* With [~by:[]], one group of every row. *)
  show
    Query.(
      of_table losses
      |> aggregate ~by:[]
           Expr.
             [ "rows" := rows; "mean" := mean loss; "p90" := quantile 0.9 loss ])
