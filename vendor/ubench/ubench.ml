type quota =
  | Time_limit of float
  (* seconds *)
  | Iteration_limit of int

type measurement = {
  time_ns : float;
  runs : int;
  minor_words : float;
  major_words : float;
  promoted_words : float;
  (* Extensible for custom predictors *)
  custom_predictors : (string * float) list;
}

type predictor =
  | Runs
  | Time_ns
  | Minor_words
  | Major_words
  | Promoted_words
  | Custom of string

type responder =
  | Time_per_run
  | Memory_per_run
  | Total_time
  | Allocation_rate
  | Custom_responder of string

type regression_result = {
  responder : responder;
  predictors : predictor list;
  coefficients : float array;
  r_squared : float;
  confidence_intervals : (float * float) array option;
}

type statistics = {
  avg : float;
  min : float;
  max : float;
  std_dev : float;
  ci95_lower : float;
  ci95_upper : float;
}

type bench_data = {
  measurements : measurement list;
  time_stats : statistics;
  memory_stats : statistics;
  regressions : regression_result list;
  total_time : float;
  total_runs : int;
}

type analysis_result = {
  name : string;
  measurements : measurement list;
  time_stats : statistics;
  memory_stats : statistics;
  regressions : regression_result list;
  total_time : float;
  total_runs : int;
}

type benchmark_config = {
  quota : quota;
  warmup_iterations : int;
  min_measurements : int;
  stabilize_gc : bool;
  geometric_scale : float;
  fork_benchmarks : bool;
  regressions : (responder * predictor list) list;
}

type output_format = Pretty_table | JSON | CSV

(* Simple matrix operations for OLS regression *)
module Matrix = struct
  type t = float array array

  let create rows cols = Array.make_matrix rows cols 0.0

  let transpose m =
    let rows = Array.length m in
    let cols = Array.length m.(0) in
    let result = create cols rows in
    for i = 0 to rows - 1 do
      for j = 0 to cols - 1 do
        result.(j).(i) <- m.(i).(j)
      done
    done;
    result

  let multiply a b =
    let rows_a = Array.length a in
    let cols_a = Array.length a.(0) in
    let cols_b = Array.length b.(0) in
    let result = create rows_a cols_b in
    for i = 0 to rows_a - 1 do
      for j = 0 to cols_b - 1 do
        for k = 0 to cols_a - 1 do
          result.(i).(j) <- result.(i).(j) +. (a.(i).(k) *. b.(k).(j))
        done
      done
    done;
    result

  let solve_normal_equations xtx xty =
    (* Solve (X'X)β = X'y using Gaussian elimination *)
    let n = Array.length xtx in
    let augmented = Array.make_matrix n (n + 1) 0.0 in

    (* Create augmented matrix [X'X | X'y] *)
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        augmented.(i).(j) <- xtx.(i).(j)
      done;
      augmented.(i).(n) <- xty.(i)
    done;

    (* Gaussian elimination with partial pivoting *)
    for i = 0 to n - 1 do
      (* Find pivot *)
      let max_row = ref i in
      for k = i + 1 to n - 1 do
        if abs_float augmented.(k).(i) > abs_float augmented.(!max_row).(i) then
          max_row := k
      done;

      (* Swap rows *)
      if !max_row <> i then (
        let temp = augmented.(i) in
        augmented.(i) <- augmented.(!max_row);
        augmented.(!max_row) <- temp);

      (* Make all rows below this one 0 in current column *)
      for k = i + 1 to n - 1 do
        if augmented.(i).(i) <> 0. then
          let factor = augmented.(k).(i) /. augmented.(i).(i) in
          for j = i to n do
            augmented.(k).(j) <-
              augmented.(k).(j) -. (factor *. augmented.(i).(j))
          done
      done
    done;

    (* Back substitution *)
    let solution = Array.make n 0.0 in
    for i = n - 1 downto 0 do
      solution.(i) <- augmented.(i).(n);
      for j = i + 1 to n - 1 do
        solution.(i) <- solution.(i) -. (augmented.(i).(j) *. solution.(j))
      done;
      if augmented.(i).(i) <> 0. then
        solution.(i) <- solution.(i) /. augmented.(i).(i)
    done;

    solution
end

let extract_predictor_value measurement = function
  | Runs -> float measurement.runs
  | Time_ns -> measurement.time_ns
  | Minor_words -> measurement.minor_words
  | Major_words -> measurement.major_words
  | Promoted_words -> measurement.promoted_words
  | Custom name -> (
      try List.assoc name measurement.custom_predictors with Not_found -> 0.0)

let extract_responder_value measurement = function
  | Time_per_run -> measurement.time_ns /. float measurement.runs
  | Memory_per_run -> measurement.minor_words /. float measurement.runs
  | Total_time -> measurement.time_ns
  | Allocation_rate -> measurement.minor_words /. measurement.time_ns *. 1e9
  | Custom_responder name -> (
      try List.assoc name measurement.custom_predictors with Not_found -> 0.0)

let ordinary_least_squares measurements ~responder ~predictors =
  let n = List.length measurements in
  let p = List.length predictors in

  if n <= p then
    {
      responder;
      predictors;
      coefficients = [||];
      r_squared = 0.0;
      confidence_intervals = None;
    }
  else
    (* Build design matrix X and response vector y *)
    let x_matrix = Array.make_matrix n p 0.0 in
    let y_vector = Array.make n 0.0 in

    List.iteri
      (fun i meas ->
        List.iteri
          (fun j pred -> x_matrix.(i).(j) <- extract_predictor_value meas pred)
          predictors;
        y_vector.(i) <- extract_responder_value meas responder)
      measurements;

    (* Compute X'X and X'y *)
    let xt = Matrix.transpose x_matrix in
    let xtx = Matrix.multiply xt x_matrix in
    let xty = Array.make p 0.0 in
    for i = 0 to p - 1 do
      for j = 0 to n - 1 do
        xty.(i) <- xty.(i) +. (xt.(i).(j) *. y_vector.(j))
      done
    done;

    try
      let coefficients = Matrix.solve_normal_equations xtx xty in

      (* Compute R-squared *)
      let y_mean = Array.fold_left ( +. ) 0.0 y_vector /. float n in
      let ss_tot =
        Array.fold_left
          (fun acc y -> acc +. ((y -. y_mean) ** 2.0))
          0.0 y_vector
      in

      let ss_res = ref 0.0 in
      List.iteri
        (fun _i meas ->
          let predicted =
            Array.fold_left
              (fun acc (j, coef) ->
                acc
                +. (coef *. extract_predictor_value meas (List.nth predictors j)))
              0.0
              (Array.mapi (fun j coef -> (j, coef)) coefficients)
          in
          let residual = extract_responder_value meas responder -. predicted in
          ss_res := !ss_res +. (residual ** 2.0))
        measurements;

      let r_squared =
        if ss_tot = 0.0 then 1.0 else 1.0 -. (!ss_res /. ss_tot)
      in

      {
        responder;
        predictors;
        coefficients;
        r_squared;
        confidence_intervals = None;
      }
    with _ ->
      {
        responder;
        predictors;
        coefficients = [||];
        r_squared = 0.0;
        confidence_intervals = None;
      }

let default_config =
  {
    quota = Time_limit 1.0;
    warmup_iterations = 3;
    min_measurements = 10;
    stabilize_gc = true;
    geometric_scale = 1.5;
    fork_benchmarks = false;
    regressions = [ (Time_per_run, [ Runs ]); (Memory_per_run, [ Runs ]) ];
  }

let stabilize_gc () =
  let rec loop failsafe last_heap_live_words =
    if failsafe <= 0 then
      failwith "unable to stabilize the number of live words in the major heap";
    Gc.compact ();
    let stat = Gc.stat () in
    if stat.live_words <> last_heap_live_words then
      loop (failsafe - 1) stat.live_words
  in
  loop 10 0

let mean values =
  let sum = Array.fold_left ( +. ) 0. values in
  sum /. float (Array.length values)

let std_deviation values =
  if Array.length values < 2 then 0.0
  else
    let avg = mean values in
    let variance =
      Array.fold_left
        (fun acc x ->
          let diff = x -. avg in
          acc +. (diff *. diff))
        0. values
    in
    sqrt (variance /. float (Array.length values - 1))

let confidence_interval_95 values =
  if Array.length values < 3 then
    let avg = mean values in
    (avg, avg)
  else
    let sorted = Array.copy values in
    Array.sort Float.compare sorted;
    let n = Array.length sorted in
    let lower_idx = max 0 (n * 25 / 1000) in
    let upper_idx = min (n - 1) (n * 975 / 1000) in
    (sorted.(lower_idx), sorted.(upper_idx))

let compute_statistics values =
  if Array.length values = 0 then
    {
      avg = 0.;
      min = 0.;
      max = 0.;
      std_dev = 0.;
      ci95_lower = 0.;
      ci95_upper = 0.;
    }
  else
    let avg = mean values in
    let min_val = Array.fold_left min Float.max_float values in
    let max_val = Array.fold_left max Float.min_float values in
    let std_dev = std_deviation values in
    let ci95_lower, ci95_upper = confidence_interval_95 values in
    { avg; min = min_val; max = max_val; std_dev; ci95_lower; ci95_upper }

let measure_one_batch f batch_size =
  for _ = 1 to 3 do
    f ()
  done;

  let minor1, promoted1, major1 = Gc.counters () in
  let start_time = Unix.gettimeofday () in

  for _ = 1 to batch_size do
    ignore (f ())
  done;

  let end_time = Unix.gettimeofday () in
  let minor2, promoted2, major2 = Gc.counters () in

  {
    time_ns = (end_time -. start_time) *. 1e9;
    runs = batch_size;
    minor_words = minor2 -. minor1;
    major_words = major2 -. major1;
    promoted_words = promoted2 -. promoted1;
    custom_predictors = [];
  }

let run_bench_with_config config f : bench_data =
  let measurements = ref [] in
  let total_time = ref 0. in
  let total_runs = ref 0 in
  let measurement_count = ref 0 in
  let batch_size = ref 1 in
  let start_time = Unix.gettimeofday () in

  for _ = 1 to config.warmup_iterations do
    ignore (f ())
  done;

  let should_continue () =
    let elapsed = Unix.gettimeofday () -. start_time in
    let min_measurements_met = !measurement_count >= config.min_measurements in
    match config.quota with
    | Time_limit max_time ->
        (not min_measurements_met)
        || (min_measurements_met && elapsed < max_time)
    | Iteration_limit max_iter -> !total_runs < max_iter
  in

  while should_continue () do
    if config.stabilize_gc then stabilize_gc ();

    let measurement = measure_one_batch f !batch_size in
    measurements := measurement :: !measurements;
    total_time := !total_time +. measurement.time_ns;
    total_runs := !total_runs + measurement.runs;
    incr measurement_count;

    let next_batch =
      int_of_float (float !batch_size *. config.geometric_scale)
    in
    batch_size := max next_batch (!batch_size + 1);

    (match config.quota with
    | Iteration_limit max_iter ->
        batch_size := min !batch_size (max_iter - !total_runs)
    | Time_limit _ -> ());

    if !batch_size <= 0 then batch_size := 1
  done;

  let measurements_list = List.rev !measurements in
  let time_values =
    Array.of_list
      (List.map (fun m -> m.time_ns /. float m.runs) measurements_list)
  in
  let memory_values =
    Array.of_list
      (List.map (fun m -> m.minor_words /. float m.runs) measurements_list)
  in

  (* Perform regression analysis *)
  let regressions =
    List.map
      (fun (resp, preds) ->
        ordinary_least_squares measurements_list ~responder:resp
          ~predictors:preds)
      config.regressions
  in

  {
    measurements = measurements_list;
    time_stats = compute_statistics time_values;
    memory_stats = compute_statistics memory_values;
    regressions;
    total_time = !total_time;
    total_runs = !total_runs;
  }

let run_benchmark_in_fork name f config =
  let read_fd, write_fd = Unix.pipe () in
  match Unix.fork () with
  | 0 -> (
      (* Child process *)
      Unix.close read_fd;
      try
        let result = run_bench_with_config config f in
        let marshalled = Marshal.to_string result [] in
        let oc = Unix.out_channel_of_descr write_fd in
        output_string oc marshalled;
        close_out oc;
        exit 0
      with e ->
        Printf.eprintf "Benchmark %s failed: %s\n" name (Printexc.to_string e);
        exit 1)
  | child_pid ->
      (* Parent process *)
      Unix.close write_fd;
      let _, status = Unix.waitpid [] child_pid in
      let ic = Unix.in_channel_of_descr read_fd in
      let result =
        match status with
        | WEXITED 0 -> ( try Some (Marshal.from_channel ic) with _ -> None)
        | _ -> None
      in
      close_in ic;
      result

let insert_underscores s =
  let len = String.length s in
  let rec aux i acc =
    if i <= 0 then acc
    else if i <= 3 then String.sub s 0 i ^ acc
    else aux (i - 3) ("_" ^ String.sub s (i - 3) 3 ^ acc)
  in
  if len <= 3 then s else aux len ""

let format_time_ns ns =
  let format_with_underscores value unit =
    let s = Printf.sprintf "%.2f" value in
    match String.split_on_char '.' s with
    | [ int_part; dec_part ] ->
        insert_underscores int_part ^ "." ^ dec_part ^ unit
    | _ -> s ^ unit
  in
  if ns < 1e3 then format_with_underscores ns "ns"
  else if ns < 1e6 then format_with_underscores (ns /. 1e3) "μs"
  else if ns < 1e9 then format_with_underscores (ns /. 1e6) "ms"
  else format_with_underscores (ns /. 1e9) "s"

let format_words w =
  if w < 1e3 then Printf.sprintf "%.2fw" w
  else if w < 1e6 then Printf.sprintf "%.2fkw" (w /. 1e3)
  else Printf.sprintf "%.2fMw" (w /. 1e6)

let format_number n =
  if n < 1e3 then Printf.sprintf "%.0f" n
  else if n < 1e6 then Printf.sprintf "%.1fk" (n /. 1e3)
  else if n < 1e9 then Printf.sprintf "%.1fM" (n /. 1e6)
  else Printf.sprintf "%.1fG" (n /. 1e9)

let string_of_responder = function
  | Time_per_run -> "time_per_run"
  | Memory_per_run -> "memory_per_run"
  | Total_time -> "total_time"
  | Allocation_rate -> "allocation_rate"
  | Custom_responder s -> s

let string_of_predictor = function
  | Runs -> "runs"
  | Time_ns -> "time_ns"
  | Minor_words -> "minor_words"
  | Major_words -> "major_words"
  | Promoted_words -> "promoted_words"
  | Custom s -> s

let print_regression_analysis results =
  Printf.printf "\n=== Regression Analysis ===\n";
  List.iter
    (fun result ->
      Printf.printf "\n--- %s ---\n" result.name;
      List.iter
        (fun reg ->
          if Array.length reg.coefficients > 0 then (
            Printf.printf "%s ~ " (string_of_responder reg.responder);
            List.iteri
              (fun i pred ->
                let coef = reg.coefficients.(i) in
                let sign =
                  if i = 0 then "" else if coef >= 0. then " + " else " - "
                in
                let abs_coef = abs_float coef in
                Printf.printf "%s%.6f*%s" sign abs_coef
                  (string_of_predictor pred))
              reg.predictors;
            Printf.printf " (R² = %.4f)\n" reg.r_squared))
        result.regressions)
    results

let print_json (results : analysis_result list) =
  let regression_to_json reg =
    let coeffs_json =
      String.concat ","
        (Array.to_list (Array.map (Printf.sprintf "%.6f") reg.coefficients))
    in
    let preds_json =
      String.concat ","
        (List.map (fun p -> "\"" ^ string_of_predictor p ^ "\"") reg.predictors)
    in
    Printf.sprintf
      {|{"responder":"%s","predictors":[%s],"coefficients":[%s],"r_squared":%.6f}|}
      (string_of_responder reg.responder)
      preds_json coeffs_json reg.r_squared
  in

  let result_to_json (r : analysis_result) =
    let regressions_json =
      String.concat "," (List.map regression_to_json r.regressions)
    in
    Printf.sprintf
      {|{"name":"%s","time_per_run_ns":%.2f,"time_ci95":[%.2f,%.2f],"memory_per_run":%.2f,"total_runs":%d,"regressions":[%s]}|}
      r.name r.time_stats.avg r.time_stats.ci95_lower r.time_stats.ci95_upper
      r.memory_stats.avg r.total_runs regressions_json
  in

  let results_json = String.concat ",\n  " (List.map result_to_json results) in
  Printf.printf "[\n  %s\n]\n" results_json

let print_csv (results : analysis_result list) =
  Printf.printf
    "name,time_per_run_ns,time_ci95_lower,time_ci95_upper,memory_per_run,total_runs,r_squared\n";
  List.iter
    (fun (r : analysis_result) ->
      let time_regression =
        try
          List.find
            (fun (reg : regression_result) -> reg.responder = Time_per_run)
            r.regressions
        with Not_found ->
          {
            responder = Time_per_run;
            predictors = [];
            coefficients = [||];
            r_squared = 0.0;
            confidence_intervals = None;
          }
      in
      Printf.printf "%s,%.2f,%.2f,%.2f,%.2f,%d,%.4f\n" r.name r.time_stats.avg
        r.time_stats.ci95_lower r.time_stats.ci95_upper r.memory_stats.avg
        r.total_runs time_regression.r_squared)
    results

let print_pretty_table results =
  if results = [] then (
    Printf.printf "No benchmark results to display.\n";
    exit 0);

  (* ANSI color codes *)
  let reset = "\x1b[0m" in
  let bold = "\x1b[1m" in
  let green = "\x1b[32m" in
  let cyan = "\x1b[36m" in
  let colorize code text = code ^ text ^ reset in

  let strip_ansi_codes s =
    let len = String.length s in
    let buf = Buffer.create len in
    let i = ref 0 in
    while !i < len do
      if !i + 1 < len && s.[!i] = '\x1b' && s.[!i + 1] = '[' then (
        try
          let j = String.index_from s (!i + 2) 'm' in
          i := j + 1
        with Not_found ->
          Buffer.add_char buf s.[!i];
          incr i)
      else (
        Buffer.add_char buf s.[!i];
        incr i)
    done;
    Buffer.contents buf
  in

  (* Count visual width of string (handling UTF-8 properly) *)
  let visual_width s =
    let s = strip_ansi_codes s in
    let len = String.length s in
    let count = ref 0 in
    let i = ref 0 in
    while !i < len do
      let c = Char.code s.[!i] in
      if c < 0x80 then (
        (* ASCII character *)
        incr count;
        incr i)
      else if c < 0xE0 then (
        (* 2-byte UTF-8 *)
        incr count;
        i := !i + 2)
      else if c < 0xF0 then (
        (* 3-byte UTF-8 *)
        incr count;
        i := !i + 3)
      else (
        (* 4-byte UTF-8 *)
        incr count;
        i := !i + 4)
    done;
    !count
  in

  let pad_left s width =
    let visible_len = visual_width s in
    if visible_len >= width then s
    else String.make (width - visible_len) ' ' ^ s
  in

  let pad_right s width =
    let visible_len = visual_width s in
    if visible_len >= width then s
    else s ^ String.make (width - visible_len) ' '
  in

  (* Find the fastest time and lowest memory for color coding *)
  let fastest_time =
    List.fold_left
      (fun acc r -> min acc r.time_stats.avg)
      Float.max_float results
  in
  let lowest_memory =
    List.fold_left
      (fun acc r -> min acc r.memory_stats.avg)
      Float.max_float results
  in

  (* Sort results by time *)
  let sorted_results =
    List.sort (fun r1 r2 -> compare r1.time_stats.avg r2.time_stats.avg) results
  in

  (* Create row data with coloring *)
  let rows_data =
    List.map
      (fun r ->
        let vs_fastest = r.time_stats.avg /. fastest_time *. 100.0 in
        ( r,
          [
            r.name;
            (if r.time_stats.avg = fastest_time then colorize green
             else fun x -> x)
              (format_time_ns r.time_stats.avg);
            (if r.memory_stats.avg = lowest_memory then colorize cyan
             else fun x -> x)
              (format_words r.memory_stats.avg);
            (if vs_fastest = 100.0 then colorize green else fun x -> x)
              (Printf.sprintf "%.2f%%" vs_fastest);
          ] ))
      sorted_results
  in

  let headers = [ "Name"; "Time/Run"; "mWd/Run"; "vs Fastest" ] in

  (* Calculate column widths based on actual data *)
  let widths =
    List.fold_left
      (fun acc (_, row_strs_colored) ->
        List.map2
          (fun w s_colored -> max w (visual_width s_colored))
          acc row_strs_colored)
      (List.map visual_width headers)
      rows_data
  in

  (* Unicode box drawing characters *)
  let top_left = "\u{250C}" in
  let top_mid = "\u{252C}" in
  let top_right = "\u{2510}" in
  let mid_left = "\u{251C}" in
  let mid_mid = "\u{253C}" in
  let mid_right = "\u{2524}" in
  let bot_left = "\u{2514}" in
  let bot_mid = "\u{2534}" in
  let bot_right = "\u{2518}" in
  let hline = "\u{2500}" in
  let vline = "\u{2502}" in

  let repeat_str s n =
    let buf = Buffer.create (n * String.length s) in
    for _ = 1 to n do
      Buffer.add_string buf s
    done;
    Buffer.contents buf
  in

  let make_border left mid right =
    left
    ^ String.concat mid (List.map (fun w -> repeat_str hline (w + 2)) widths)
    ^ right
  in

  let top_border = make_border top_left top_mid top_right in
  let separator = make_border mid_left mid_mid mid_right in
  let bottom_border = make_border bot_left bot_mid bot_right in

  Printf.printf "%s\n" top_border;

  (* Print headers *)
  let print_header_row headers =
    let padded_headers =
      List.mapi
        (fun i s ->
          let w = List.nth widths i in
          let padded_s = if i = 0 then pad_right s w else pad_left s w in
          colorize bold padded_s)
        headers
    in
    let row_str = String.concat (" " ^ vline ^ " ") padded_headers in
    Printf.printf "%s %s %s\n" vline row_str vline
  in
  print_header_row headers;
  Printf.printf "%s\n" separator;

  (* Print data rows *)
  let print_data_row (_result_record, row_strings_colored) =
    let padded_colored_row =
      List.mapi
        (fun i s_colored ->
          let w = List.nth widths i in
          if i = 0 then pad_right s_colored w else pad_left s_colored w)
        row_strings_colored
    in
    let row_str = String.concat (" " ^ vline ^ " ") padded_colored_row in
    Printf.printf "%s %s %s\n" vline row_str vline
  in
  List.iter print_data_row rows_data;
  Printf.printf "%s\n" bottom_border

open Cmdliner

let quota_conv =
  let parse s =
    if String.contains s 's' then
      try
        let time_str = String.sub s 0 (String.index s 's') in
        Ok (Time_limit (float_of_string time_str))
      with _ -> Error (`Msg "Invalid time format")
    else if String.contains s 'x' then
      try
        let iter_str = String.sub s 0 (String.index s 'x') in
        Ok (Iteration_limit (int_of_string iter_str))
      with _ -> Error (`Msg "Invalid iteration format")
    else
      try Ok (Time_limit (float_of_string s))
      with _ -> Error (`Msg "Invalid quota format")
  in
  let print fmt quota =
    match quota with
    | Time_limit t -> Format.fprintf fmt "%.1fs" t
    | Iteration_limit i -> Format.fprintf fmt "%dx" i
  in
  Arg.conv (parse, print)

let output_format_conv =
  let parse = function
    | "table" | "pretty" -> Ok Pretty_table
    | "json" -> Ok JSON
    | "csv" -> Ok CSV
    | s -> Error (`Msg ("Invalid format: " ^ s))
  in
  let print fmt = function
    | Pretty_table -> Format.fprintf fmt "table"
    | JSON -> Format.fprintf fmt "json"
    | CSV -> Format.fprintf fmt "csv"
  in
  Arg.conv (parse, print)

let quota_arg =
  let doc = "Time limit (e.g., '5s') or iteration limit (e.g., '1000x')" in
  Arg.(value & opt quota_conv (Time_limit 1.0) & info [ "q"; "quota" ] ~doc)

let output_format_arg =
  let doc = "Output format: table, json, or csv" in
  Arg.(
    value & opt output_format_conv Pretty_table & info [ "f"; "format" ] ~doc)

let fork_arg =
  let doc = "Run each benchmark in a separate process" in
  Arg.(value & flag & info [ "fork" ] ~doc)

let warmup_arg =
  let doc = "Number of warmup iterations" in
  Arg.(value & opt int 3 & info [ "w"; "warmup" ] ~doc)

let stabilize_gc_arg =
  let doc = "Stabilize GC between measurements" in
  Arg.(value & flag & info [ "gc" ] ~doc)

let verbose_arg =
  let doc = "Show regression analysis" in
  Arg.(value & flag & info [ "v"; "verbose" ] ~doc)

type benchmark = string * (unit -> unit)

let create name f = (name, f)

let run_with_cli_config quota output_format fork_benchmarks warmup stabilize_gc
    _verbose benchmarks =
  let config =
    {
      quota;
      warmup_iterations = warmup;
      min_measurements = 10;
      stabilize_gc;
      geometric_scale = 1.5;
      fork_benchmarks;
      regressions = [ (Time_per_run, [ Runs ]); (Memory_per_run, [ Runs ]) ];
    }
  in

  Printf.printf "Running %d benchmarks...\n%!" (List.length benchmarks);

  let results =
    List.mapi
      (fun i (name, f) ->
        Printf.printf "[%d/%d] Running %s..." (i + 1) (List.length benchmarks)
          name;
        flush_all ();

        let result =
          if config.fork_benchmarks then (
            match run_benchmark_in_fork name f config with
            | Some r -> r
            | None ->
                Printf.printf " FAILED.\n%!";
                {
                  measurements = [];
                  time_stats = compute_statistics [||];
                  memory_stats = compute_statistics [||];
                  regressions = [];
                  total_time = 0.;
                  total_runs = 0;
                })
          else run_bench_with_config config f
        in

        Printf.printf " Done.\n%!";
        {
          name;
          measurements = result.measurements;
          time_stats = result.time_stats;
          memory_stats = result.memory_stats;
          regressions = result.regressions;
          total_time = result.total_time;
          total_runs = result.total_runs;
        })
      benchmarks
  in

  Printf.printf "\nBenchmark Results:\n";
  (match output_format with
  | Pretty_table -> print_pretty_table results
  | JSON -> print_json results
  | CSV -> print_csv results);

  results

let cli_term =
  Term.(
    const run_with_cli_config $ quota_arg $ output_format_arg $ fork_arg
    $ warmup_arg $ stabilize_gc_arg $ verbose_arg)

let run ?(config = default_config) ?(output_format = Pretty_table) benchmarks =
  run_with_cli_config config.quota output_format config.fork_benchmarks
    config.warmup_iterations config.stabilize_gc false benchmarks

(* CLI command *)
let make_cli_command benchmarks =
  let info = Cmd.info "ubench" ~doc:"Universal benchmarking tool" in
  Cmd.v info Term.(const (fun () -> ignore (run benchmarks)) $ const ())

(* Convenience functions *)
let with_time_limit seconds config = { config with quota = Time_limit seconds }

let with_iteration_limit iters config =
  { config with quota = Iteration_limit iters }

let with_warmup warmup config = { config with warmup_iterations = warmup }

let with_gc_stabilization enabled config =
  { config with stabilize_gc = enabled }

let with_fork enabled config = { config with fork_benchmarks = enabled }
let with_regressions regressions config = { config with regressions }
