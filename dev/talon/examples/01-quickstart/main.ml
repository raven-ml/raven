open Talon

let () =
  (* Create a simple dataframe *)
  let df =
    create
      [
        ("name", Col.string_list [ "Alice"; "Bob"; "Charlie"; "Dana" ]);
        ("age", Col.int32_list [ 25l; 30l; 35l; 28l ]);
        ("height", Col.float64_list [ 1.70; 1.80; 1.75; 1.65 ]);
        ("weight", Col.float64_list [ 65.0; 82.0; 77.0; 55.0 ]);
        ("active", Col.bool_list [ true; false; true; true ]);
      ]
  in

  Printf.printf "== quickstart ==\n";
  Printf.printf "shape: %d rows x %d cols\n" (num_rows df) (num_columns df);

  (* Add a BMI column: weight / (height^2) *)
  let df =
    with_column df "bmi" Nx.float64
      Row.(
        map2 (number "weight") (number "height") ~f:(fun w h -> w /. (h ** 2.)))
  in

  (* Row-wise “fitness score”: BMI inverse + activity boost *)
  let df =
    with_column df "fitness" Nx.float64
      Row.(
        map2 (number "bmi") (bool "active") ~f:(fun bmi active ->
            (1. /. bmi) +. if active then 0.2 else 0.))
  in

  (* Column aggregations (operate on a single column) *)
  let avg_bmi = Agg.Float.mean df "bmi" in
  Printf.printf "avg BMI: %.3f\n" avg_bmi;

  (* Show the head *)
  print ~max_rows:10 df;
  ()
