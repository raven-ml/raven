type quota =
  | Time_limit of float
  (* seconds *)
  | Iteration_limit of int
  | Variance_limit of float
(* coefficient of variation threshold *)

type measurement = {
  time_ns : float; (* CPU time (user + system) *)
  wall_ns : float; (* Wall clock time *)
  utime_ns : float; (* User CPU time *)
  stime_ns : float; (* System CPU time *)
  cutime_ns : float; (* Child user CPU time *)
  cstime_ns : float; (* Child system CPU time *)
  cycles : float; (* CPU cycles (estimated from time if TSC unavailable) *)
  runs : int;
  minor_words : float;
  major_words : float;
  promoted_words : float;
  minor_collections : int;
  major_collections : int;
  compactions : int;
  (* Extensible for custom predictors *)
  custom_predictors : (string * float) list;
}

type predictor =
  | One (* Constant predictor for overhead estimation *)
  | Runs
  | Time_ns
  | Wall_ns
  | Cycles
  | User_time
  | System_time
  | Child_time
  | Minor_words
  | Major_words
  | Promoted_words
  | Minor_collections
  | Major_collections
  | Compactions
  | Custom of string

type responder =
  | Time_per_run
  | Wall_per_run
  | Memory_per_run
  | Total_time
  | Total_wall
  | Allocation_rate
  | Custom_responder of string

type regression_result = {
  responder : responder;
  predictors : predictor list;
  coefficients : float array;
  r_squared : float;
  adjusted_r_squared : float;
  intercept : float option;
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
  wall_stats : statistics;
  memory_stats : statistics;
  regressions : regression_result list;
  total_time : float;
  total_runs : int;
}

type analysis_result = {
  name : string;
  measurements : measurement list;
  time_stats : statistics;
  wall_stats : statistics;
  memory_stats : statistics;
  regressions : regression_result list;
  total_time : float;
  total_runs : int;
}

type benchmark_mode =
  | Latency of int (* Fixed number of iterations *)
  | Throughput of float (* Minimum CPU time in seconds *)

type progress_info = {
  name : string;
  current_measurement : int;
  total_measurements : int option;
  elapsed_time : float;
  estimated_remaining : float option;
}

module Config = struct
  type t = {
    mode : benchmark_mode;
    quota : quota;
    warmup_iterations : int;
    min_measurements : int;
    stabilize_gc : bool;
    geometric_scale : float;
    fork_benchmarks : bool;
    regressions : (responder * predictor list * bool) list;
        (* bool = include intercept *)
    custom_measurer : ((unit -> unit) -> int -> measurement) option;
    ascii_only : bool;
    null_loop_subtraction : bool;
    min_cpu : float; (* Minimum CPU time for reliable results *)
    repeat : int; (* Number of times to repeat the benchmark *)
    progress_callback : (progress_info -> unit) option;
  }

  let default =
    {
      mode = Throughput 1.0;
      quota = Time_limit 0.3;
      warmup_iterations = 1;
      min_measurements = 3;
      stabilize_gc = false;
      geometric_scale = 1.3;
      fork_benchmarks = false;
      regressions =
        [
          (Time_per_run, [ One; Runs ], false); (Memory_per_run, [ Runs ], true);
        ];
      custom_measurer = None;
      ascii_only = false;
      null_loop_subtraction = true;
      min_cpu = 0.002;
      repeat = 1;
      progress_callback = None;
    }

  let time_limit secs c = { c with quota = Time_limit secs }
  let iteration_limit iters c = { c with quota = Iteration_limit iters }
  let variance_limit cv c = { c with quota = Variance_limit cv }
  let warmup n c = { c with warmup_iterations = n }
  let min_measurements n c = { c with min_measurements = n }
  let gc_stabilization enabled c = { c with stabilize_gc = enabled }
  let fork enabled c = { c with fork_benchmarks = enabled }
  let ascii_only enabled c = { c with ascii_only = enabled }
  let geometric_scale factor c = { c with geometric_scale = factor }
  let regressions regs c = { c with regressions = regs }
  let custom_measurer m c = { c with custom_measurer = Some m }
  let progress_callback cb c = { c with progress_callback = Some cb }
  let ( |> ) c f = f c
  let build c = c
end

let default_config = Config.default

open Config

type output_format = Pretty_table | JSON | CSV

exception Matrix_singular of string

(* Simple matrix operations for OLS regression *)
module Matrix = struct
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

      (* Check for near-zero pivot (numerical instability) *)
      if abs_float augmented.(i).(i) < 1e-10 then
        raise (Matrix_singular (Printf.sprintf "Near-zero pivot at row %d" i));

      (* Make all rows below this one 0 in current column *)
      for k = i + 1 to n - 1 do
        let factor = augmented.(k).(i) /. augmented.(i).(i) in
        for j = i to n do
          augmented.(k).(j) <- augmented.(k).(j) -. (factor *. augmented.(i).(j))
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
      if abs_float augmented.(i).(i) < 1e-10 then
        raise
          (Matrix_singular (Printf.sprintf "Near-zero diagonal at row %d" i));
      solution.(i) <- solution.(i) /. augmented.(i).(i)
    done;

    solution
end

let extract_predictor_value measurement = function
  | One -> 1.0 (* Constant value for overhead estimation *)
  | Runs -> float measurement.runs
  | Time_ns -> measurement.time_ns
  | Wall_ns -> measurement.wall_ns
  | Cycles -> measurement.cycles
  | User_time -> measurement.utime_ns
  | System_time -> measurement.stime_ns
  | Child_time -> measurement.cutime_ns +. measurement.cstime_ns
  | Minor_words -> measurement.minor_words
  | Major_words -> measurement.major_words
  | Promoted_words -> measurement.promoted_words
  | Minor_collections -> float measurement.minor_collections
  | Major_collections -> float measurement.major_collections
  | Compactions -> float measurement.compactions
  | Custom name -> (
      try List.assoc name measurement.custom_predictors with Not_found -> 0.0)

let extract_responder_value measurement = function
  | Time_per_run -> measurement.time_ns /. float measurement.runs
  | Wall_per_run -> measurement.wall_ns /. float measurement.runs
  | Memory_per_run -> measurement.minor_words /. float measurement.runs
  | Total_time -> measurement.time_ns
  | Total_wall -> measurement.wall_ns
  | Allocation_rate -> measurement.minor_words /. measurement.time_ns *. 1e9
  | Custom_responder name -> (
      try List.assoc name measurement.custom_predictors with Not_found -> 0.0)

let ordinary_least_squares ?(intercept = true) measurements ~responder
    ~predictors =
  let n = List.length measurements in
  let p_orig = List.length predictors in
  let p = p_orig + if intercept then 1 else 0 in

  if n <= p then
    {
      responder;
      predictors;
      coefficients = [||];
      r_squared = 0.0;
      adjusted_r_squared = 0.0;
      intercept = None;
      confidence_intervals = None;
    }
  else
    (* Build design matrix X and response vector y *)
    let x_matrix = Array.make_matrix n p 0.0 in
    let y_vector = Array.make n 0.0 in

    List.iteri
      (fun i meas ->
        (* Add intercept column if requested *)
        if intercept then x_matrix.(i).(0) <- 1.0;
        List.iteri
          (fun j pred ->
            let col_idx = j + if intercept then 1 else 0 in
            x_matrix.(i).(col_idx) <- extract_predictor_value meas pred)
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
      let all_coefficients = Matrix.solve_normal_equations xtx xty in

      (* Separate intercept from other coefficients *)
      let intercept_value, coefficients =
        if intercept then
          (Some all_coefficients.(0), Array.sub all_coefficients 1 p_orig)
        else (None, all_coefficients)
      in

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
            (* Start with intercept if present *)
            let base = match intercept_value with Some b -> b | None -> 0.0 in
            (* Add predictor contributions *)
            Array.fold_left
              (fun acc (j, coef) ->
                acc
                +. (coef *. extract_predictor_value meas (List.nth predictors j)))
              base
              (Array.mapi (fun j coef -> (j, coef)) coefficients)
          in
          let residual = extract_responder_value meas responder -. predicted in
          ss_res := !ss_res +. (residual ** 2.0))
        measurements;

      let r_squared =
        if ss_tot = 0.0 then 1.0 else 1.0 -. (!ss_res /. ss_tot)
      in

      (* Calculate adjusted R² *)
      let adjusted_r_squared =
        if n - p <= 0 then r_squared
        else 1.0 -. ((1.0 -. r_squared) *. float (n - 1) /. float (n - p))
      in

      {
        responder;
        predictors;
        coefficients;
        r_squared;
        adjusted_r_squared;
        intercept = intercept_value;
        confidence_intervals = None;
      }
    with
    | Matrix_singular msg ->
        Printf.eprintf "Warning: Matrix singular in regression: %s\n" msg;
        {
          responder;
          predictors;
          coefficients = [||];
          r_squared = 0.0;
          adjusted_r_squared = 0.0;
          intercept = None;
          confidence_intervals = None;
        }
    | e ->
        Printf.eprintf "Warning: Regression failed: %s\n" (Printexc.to_string e);
        {
          responder;
          predictors;
          coefficients = [||];
          r_squared = 0.0;
          adjusted_r_squared = 0.0;
          intercept = None;
          confidence_intervals = None;
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

(* Bootstrap resampling for more robust confidence intervals *)
let bootstrap_confidence_interval ?(n_resamples = 1000) ?(confidence = 0.95)
    values compute_statistic =
  if Array.length values < 3 then
    let stat = compute_statistic values in
    (stat, stat)
  else
    let n = Array.length values in
    let bootstrap_stats = Array.make n_resamples 0.0 in

    (* Random seed for reproducibility *)
    Random.self_init ();

    (* Perform bootstrap resampling *)
    for i = 0 to n_resamples - 1 do
      let resample = Array.init n (fun _ -> values.(Random.int n)) in
      bootstrap_stats.(i) <- compute_statistic resample
    done;

    (* Sort bootstrap statistics *)
    Array.sort Float.compare bootstrap_stats;

    (* Calculate confidence interval indices *)
    let alpha = 1.0 -. confidence in
    let lower_idx = int_of_float (float n_resamples *. (alpha /. 2.0)) in
    let upper_idx =
      int_of_float (float n_resamples *. (1.0 -. (alpha /. 2.0)))
    in

    (bootstrap_stats.(lower_idx), bootstrap_stats.(upper_idx))

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
    (* Use bootstrap for more robust confidence intervals when we have enough
       samples *)
    let ci95_lower, ci95_upper =
      if Array.length values >= 20 then
        bootstrap_confidence_interval ~n_resamples:1000 ~confidence:0.95 values
          mean
      else confidence_interval_95 values
    in
    { avg; min = min_val; max = max_val; std_dev; ci95_lower; ci95_upper }

(* Statistical functions for t-tests and comparison *)

(* log_gamma x computes the logarithm of the Gamma function at x using Lanczos
   method. It is assumed x > 0. *)
let log_gamma =
  let c =
    [|
      1.000000000000000174663;
      5716.400188274341379136;
      -14815.30426768413909044;
      14291.49277657478554025;
      -6348.160217641458813289;
      1301.608286058321874105;
      -108.1767053514369634679;
      2.605696505611755827729;
      -0.7423452510201416151527e-2;
      0.5384136432509564062961e-7;
      -0.4023533141268236372067e-8;
    |]
  in
  let c_last = Array.length c - 1 in
  let g = float (c_last - 1) in
  let sqrt2pi = sqrt (8. *. atan 1.) in
  let rec sum i den s =
    if i > 0 then sum (i - 1) (den -. 1.) (s +. (c.(i) /. den)) else c.(0) +. s
  in
  fun x ->
    assert (x > 0.);
    let xg = x +. g in
    let xg_5 = xg -. 0.5 in
    log (sqrt2pi *. sum c_last xg 0.) +. ((x -. 0.5) *. log xg_5) -. xg_5

(* Helper for betai function *)
let max_tiny x = max 1e-30 x (* to avoid null divisors *)
let betai_cf_eps = epsilon_float

(* Continued fraction expansion for incomplete beta function *)
let betai_cf x a b =
  let apb = a +. b and ap1 = a +. 1. and am1 = a -. 1. in
  let rec lentz m c d f =
    let m2 = 2. *. m in
    (* Even rec step d_2m *)
    let cf_d2m = m *. (b -. m) *. x /. ((am1 +. m2) *. (a +. m2)) in
    let d = 1. /. max_tiny (1. +. (cf_d2m *. d))
    and c = max_tiny (1. +. (cf_d2m /. c)) in
    let f = f *. d *. c in
    (* Odd rec step d_2m+1 *)
    let cf_d2m1 = -.(a +. m) *. (apb +. m) *. x /. ((a +. m2) *. (ap1 +. m2)) in
    let d = 1. /. max_tiny (1. +. (cf_d2m1 *. d))
    and c = max_tiny (1. +. (cf_d2m1 /. c)) in
    let delta = c *. d in
    let f = f *. delta in
    if abs_float (delta -. 1.) < betai_cf_eps then f else lentz (m +. 1.) c d f
  in
  (* Initialize Lentz's method *)
  let d2 = 1. /. max_tiny (1. -. (apb *. x /. ap1)) in
  lentz 1. 1. d2 d2

(* Incomplete beta function I_x(a,b) *)
let betai x a b =
  assert (a > 0. && b > 0.);
  if x < 0. || x > 1. then invalid_arg "betai";
  if x = 0. then 0.
  else if x = 1. then 1.
  else
    let m =
      exp
        (log_gamma (a +. b)
        -. log_gamma a -. log_gamma b
        +. (a *. log x)
        +. (b *. log (1. -. x)))
    in
    if x < (a +. 1.) /. (a +. b +. 2.) then m *. betai_cf x a b /. a
    else 1. -. (m *. betai_cf (1. -. x) b a /. b)

(* Complement of Student's t distribution: 1 - A(t|nu) *)
let cpl_student_t t nu = betai (nu /. (nu +. (t *. t))) (0.5 *. nu) 0.5

(* Check whether two rates are significantly different using t-test *)
let different_rates significance n1 r1 s1 n2 r2 s2 =
  assert (n1 > 0 && n2 > 0);
  if n1 = 1 && n2 = 1 then true
    (* no info about distribution, assume different *)
  else
    let df = float (n1 + n2 - 2) (* degrees of freedom *)
    and n1 = float n1
    and n2 = float n2 in
    let sD = sqrt ((s1 +. s2) /. df *. ((1. /. n1) +. (1. /. n2))) in
    let t = (r1 -. r2) /. sD in
    cpl_student_t t df <= significance

(* Measure null loop overhead *)
let measure_null_loop batch_size =
  let minor1, promoted1, major1 = Gc.counters () in
  let gc1 = Gc.quick_stat () in
  let tms1 = Unix.times () in
  let wall_start = Unix.gettimeofday () in

  for _ = 1 to batch_size do
    ignore (ignore ())
  done;

  let wall_end = Unix.gettimeofday () in
  let tms2 = Unix.times () in
  let minor2, promoted2, major2 = Gc.counters () in
  let gc2 = Gc.quick_stat () in

  let utime = tms2.Unix.tms_utime -. tms1.Unix.tms_utime in
  let stime = tms2.Unix.tms_stime -. tms1.Unix.tms_stime in
  let cutime = tms2.Unix.tms_cutime -. tms1.Unix.tms_cutime in
  let cstime = tms2.Unix.tms_cstime -. tms1.Unix.tms_cstime in
  (* Estimate cycles from wall time and typical CPU frequency (3 GHz) *)
  let estimated_cycles = (wall_end -. wall_start) *. 3e9 in

  {
    time_ns = (utime +. stime) *. 1e9;
    wall_ns = (wall_end -. wall_start) *. 1e9;
    utime_ns = utime *. 1e9;
    stime_ns = stime *. 1e9;
    cutime_ns = cutime *. 1e9;
    cstime_ns = cstime *. 1e9;
    cycles = estimated_cycles;
    runs = batch_size;
    minor_words = minor2 -. minor1;
    major_words = major2 -. major1;
    promoted_words = promoted2 -. promoted1;
    minor_collections = gc2.minor_collections - gc1.minor_collections;
    major_collections = gc2.major_collections - gc1.major_collections;
    compactions = gc2.compactions - gc1.compactions;
    custom_predictors = [];
  }

(* Subtract measurement b from a, ensuring non-negative results *)
let subtract_measurements a b =
  let pos_sub x y = if x > y then x -. y else 0. in
  let pos_sub_int x y = if x > y then x - y else 0 in
  {
    time_ns = pos_sub a.time_ns b.time_ns;
    wall_ns = pos_sub a.wall_ns b.wall_ns;
    utime_ns = pos_sub a.utime_ns b.utime_ns;
    stime_ns = pos_sub a.stime_ns b.stime_ns;
    cutime_ns = pos_sub a.cutime_ns b.cutime_ns;
    cstime_ns = pos_sub a.cstime_ns b.cstime_ns;
    cycles = pos_sub a.cycles b.cycles;
    runs = a.runs;
    minor_words = pos_sub a.minor_words b.minor_words;
    major_words = pos_sub a.major_words b.major_words;
    promoted_words = pos_sub a.promoted_words b.promoted_words;
    minor_collections = pos_sub_int a.minor_collections b.minor_collections;
    major_collections = pos_sub_int a.major_collections b.major_collections;
    compactions = pos_sub_int a.compactions b.compactions;
    custom_predictors = a.custom_predictors;
  }

(* Enhanced measurement with wall time and Unix.times *)
let measure_one_batch ?(null_loop_subtraction = true) f batch_size =
  (* Warmup to reduce noise *)
  for _ = 1 to 3 do
    f ()
  done;

  let minor1, promoted1, major1 = Gc.counters () in
  let gc1 = Gc.quick_stat () in
  let tms1 = Unix.times () in
  let wall_start = Unix.gettimeofday () in

  for _ = 1 to batch_size do
    ignore (f ())
  done;

  let wall_end = Unix.gettimeofday () in
  let tms2 = Unix.times () in
  let minor2, promoted2, major2 = Gc.counters () in
  let gc2 = Gc.quick_stat () in

  let utime = tms2.Unix.tms_utime -. tms1.Unix.tms_utime in
  let stime = tms2.Unix.tms_stime -. tms1.Unix.tms_stime in
  let cutime = tms2.Unix.tms_cutime -. tms1.Unix.tms_cutime in
  let cstime = tms2.Unix.tms_cstime -. tms1.Unix.tms_cstime in
  (* Estimate cycles from wall time and typical CPU frequency (3 GHz) *)
  let estimated_cycles = (wall_end -. wall_start) *. 3e9 in

  let raw_measurement =
    {
      time_ns = (utime +. stime) *. 1e9;
      wall_ns = (wall_end -. wall_start) *. 1e9;
      utime_ns = utime *. 1e9;
      stime_ns = stime *. 1e9;
      cutime_ns = cutime *. 1e9;
      cstime_ns = cstime *. 1e9;
      cycles = estimated_cycles;
      runs = batch_size;
      minor_words = minor2 -. minor1;
      major_words = major2 -. major1;
      promoted_words = promoted2 -. promoted1;
      minor_collections = gc2.minor_collections - gc1.minor_collections;
      major_collections = gc2.major_collections - gc1.major_collections;
      compactions = gc2.compactions - gc1.compactions;
      custom_predictors = [];
    }
  in

  if null_loop_subtraction then
    let null_measurement = measure_null_loop batch_size in
    subtract_measurements raw_measurement null_measurement
  else raw_measurement

let run_bench_with_config config f : bench_data =
  (* Validate config *)
  if config.geometric_scale <= 1.0 then
    failwith
      (Printf.sprintf "Invalid geometric_scale: %.2f (must be > 1.0)"
         config.geometric_scale);

  let measurements = ref [] in
  let total_time = ref 0. in
  let total_runs = ref 0 in
  let measurement_count = ref 0 in
  let batch_size = ref 1 in
  let start_time = Sys.time () in

  (* Use custom measurer if provided *)
  let measure_batch =
    match config.custom_measurer with
    | Some measurer -> measurer f
    | None -> measure_one_batch f
  in

  for _ = 1 to config.warmup_iterations do
    ignore (f ())
  done;

  let should_continue () =
    let elapsed = Sys.time () -. start_time in
    let min_measurements_met = !measurement_count >= config.min_measurements in
    match config.quota with
    | Time_limit max_time ->
        (not min_measurements_met)
        || (min_measurements_met && elapsed < max_time)
    | Iteration_limit max_iter -> !total_runs < max_iter
    | Variance_limit cv_threshold ->
        if !measurement_count < config.min_measurements then true
        else
          let time_values =
            List.map (fun m -> m.time_ns /. float m.runs) !measurements
          in
          let values_array = Array.of_list time_values in
          if Array.length values_array = 0 then true
          else
            let avg = mean values_array in
            let std_dev = std_deviation values_array in
            if avg = 0. then false else std_dev /. avg > cv_threshold
  in

  while should_continue () do
    if config.stabilize_gc then stabilize_gc ();

    let measurement = measure_batch !batch_size in
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
    | Time_limit _ | Variance_limit _ -> ());

    if !batch_size <= 0 then batch_size := 1
  done;

  let measurements_list = List.rev !measurements in
  let time_values =
    Array.of_list
      (List.map
         (fun m -> if m.runs = 0 then 0. else m.time_ns /. float m.runs)
         measurements_list)
  in
  let wall_values =
    Array.of_list
      (List.map
         (fun m -> if m.runs = 0 then 0. else m.wall_ns /. float m.runs)
         measurements_list)
  in
  let memory_values =
    Array.of_list
      (List.map
         (fun m -> if m.runs = 0 then 0. else m.minor_words /. float m.runs)
         measurements_list)
  in

  (* Perform regression analysis *)
  let regressions =
    List.map
      (fun (resp, preds, intercept) ->
        ordinary_least_squares ~intercept measurements_list ~responder:resp
          ~predictors:preds)
      config.regressions
  in

  {
    measurements = measurements_list;
    time_stats = compute_statistics time_values;
    wall_stats = compute_statistics wall_values;
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
  | Wall_per_run -> "wall_per_run"
  | Memory_per_run -> "memory_per_run"
  | Total_time -> "total_time"
  | Total_wall -> "total_wall"
  | Allocation_rate -> "allocation_rate"
  | Custom_responder s -> s

let string_of_predictor = function
  | One -> "one"
  | Runs -> "runs"
  | Time_ns -> "time_ns"
  | Wall_ns -> "wall_ns"
  | Cycles -> "cycles"
  | User_time -> "user_time"
  | System_time -> "system_time"
  | Child_time -> "child_time"
  | Minor_words -> "minor_words"
  | Major_words -> "major_words"
  | Promoted_words -> "promoted_words"
  | Minor_collections -> "minor_collections"
  | Major_collections -> "major_collections"
  | Compactions -> "compactions"
  | Custom s -> s

let print_regression_analysis results =
  Printf.printf "\n=== Regression Analysis ===\n";
  List.iter
    (fun (result : analysis_result) ->
      Printf.printf "\n--- %s ---\n" result.name;
      List.iter
        (fun reg ->
          if Array.length reg.coefficients > 0 then (
            Printf.printf "%s ~ " (string_of_responder reg.responder);
            (* Print intercept if present *)
            (match reg.intercept with
            | Some intercept -> Printf.printf "%.6f" intercept
            | None -> ());
            (* Print predictor terms *)
            List.iteri
              (fun i pred ->
                let coef = reg.coefficients.(i) in
                let sign =
                  if i = 0 && reg.intercept = None then ""
                  else if coef >= 0. then " + "
                  else " - "
                in
                let abs_coef = abs_float coef in
                Printf.printf "%s%.6f*%s" sign abs_coef
                  (string_of_predictor pred))
              reg.predictors;
            Printf.printf " (R² = %.4f, Adj R² = %.4f)\n" reg.r_squared
              reg.adjusted_r_squared))
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
    let intercept_json =
      match reg.intercept with
      | Some i -> Printf.sprintf ",\"intercept\":%.6f" i
      | None -> ""
    in
    Printf.sprintf
      {|{"responder":"%s","predictors":[%s],"coefficients":[%s],"r_squared":%.6f,"adjusted_r_squared":%.6f%s}|}
      (string_of_responder reg.responder)
      preds_json coeffs_json reg.r_squared reg.adjusted_r_squared intercept_json
  in

  let result_to_json (r : analysis_result) =
    let regressions_json =
      String.concat "," (List.map regression_to_json r.regressions)
    in
    Printf.sprintf
      {|{"name":"%s","time_stats":{"avg":%.2f,"min":%.2f,"max":%.2f,"std_dev":%.2f,"ci95_lower":%.2f,"ci95_upper":%.2f},"wall_stats":{"avg":%.2f,"min":%.2f,"max":%.2f,"std_dev":%.2f,"ci95_lower":%.2f,"ci95_upper":%.2f},"memory_stats":{"avg":%.2f,"min":%.2f,"max":%.2f,"std_dev":%.2f,"ci95_lower":%.2f,"ci95_upper":%.2f},"total_time_ns":%.2f,"total_runs":%d,"regressions":[%s]}|}
      r.name r.time_stats.avg r.time_stats.min r.time_stats.max
      r.time_stats.std_dev r.time_stats.ci95_lower r.time_stats.ci95_upper
      r.wall_stats.avg r.wall_stats.min r.wall_stats.max r.wall_stats.std_dev
      r.wall_stats.ci95_lower r.wall_stats.ci95_upper r.memory_stats.avg
      r.memory_stats.min r.memory_stats.max r.memory_stats.std_dev
      r.memory_stats.ci95_lower r.memory_stats.ci95_upper r.total_time
      r.total_runs regressions_json
  in

  let results_json = String.concat ",\n  " (List.map result_to_json results) in
  Printf.printf "[\n  %s\n]\n" results_json

let print_csv (results : analysis_result list) =
  Printf.printf
    "name,time_avg,time_min,time_max,time_std_dev,time_ci95_lower,time_ci95_upper,wall_avg,wall_min,wall_max,wall_std_dev,wall_ci95_lower,wall_ci95_upper,memory_avg,memory_min,memory_max,memory_std_dev,memory_ci95_lower,memory_ci95_upper,total_runs,time_r_squared,time_adjusted_r_squared\n";
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
            adjusted_r_squared = 0.0;
            intercept = None;
            confidence_intervals = None;
          }
      in
      Printf.printf
        "%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.4f,%.4f\n"
        r.name r.time_stats.avg r.time_stats.min r.time_stats.max
        r.time_stats.std_dev r.time_stats.ci95_lower r.time_stats.ci95_upper
        r.wall_stats.avg r.wall_stats.min r.wall_stats.max r.wall_stats.std_dev
        r.wall_stats.ci95_lower r.wall_stats.ci95_upper r.memory_stats.avg
        r.memory_stats.min r.memory_stats.max r.memory_stats.std_dev
        r.memory_stats.ci95_lower r.memory_stats.ci95_upper r.total_runs
        time_regression.r_squared time_regression.adjusted_r_squared)
    results

(* Tabulate results for cross-benchmark comparison with statistical
   significance *)
type comparison_result = {
  baseline : analysis_result;
  compared : analysis_result;
  speedup : float;
  speedup_ci_lower : float;
  speedup_ci_upper : float;
  significant : bool;
  p_value : float option;
}

let compare ?(confidence = 0.95) baseline compared =
  let n1 = List.length baseline.measurements in
  let n2 = List.length compared.measurements in

  if n1 = 0 || n2 = 0 then
    {
      baseline;
      compared;
      speedup = nan;
      speedup_ci_lower = nan;
      speedup_ci_upper = nan;
      significant = false;
      p_value = None;
    }
  else
    let rates1 =
      List.map
        (fun m -> float m.runs /. (m.wall_ns /. 1e9))
        baseline.measurements
    in
    let rates2 =
      List.map
        (fun m -> float m.runs /. (m.wall_ns /. 1e9))
        compared.measurements
    in

    let rates1_array = Array.of_list rates1 in
    let rates2_array = Array.of_list rates2 in

    let avg1 = mean rates1_array in
    let avg2 = mean rates2_array in
    let var1 =
      if n1 > 1 then
        let sum_sq =
          Array.fold_left
            (fun acc r ->
              let diff = r -. avg1 in
              acc +. (diff *. diff))
            0. rates1_array
        in
        sum_sq /. float (n1 - 1)
      else 0.
    in
    let var2 =
      if n2 > 1 then
        let sum_sq =
          Array.fold_left
            (fun acc r ->
              let diff = r -. avg2 in
              acc +. (diff *. diff))
            0. rates2_array
        in
        sum_sq /. float (n2 - 1)
      else 0.
    in

    let speedup = avg2 /. avg1 in
    let is_significant =
      different_rates (1. -. confidence) n1 avg1 var1 n2 avg2 var2
    in

    let speedup_ci_lower, speedup_ci_upper =
      if n1 > 1 && n2 > 1 then
        let se1 = sqrt (var1 /. float n1) in
        let se2 = sqrt (var2 /. float n2) in
        let z = 1.96 in
        let lower = (avg2 -. (z *. se2)) /. (avg1 +. (z *. se1)) in
        let upper = (avg2 +. (z *. se2)) /. (avg1 -. (z *. se1)) in
        (lower, upper)
      else (speedup, speedup)
    in

    {
      baseline;
      compared;
      speedup;
      speedup_ci_lower;
      speedup_ci_upper;
      significant = is_significant;
      p_value = None;
    }

let print_comparison comp =
  Printf.printf "\n=== Benchmark Comparison ===\n";
  Printf.printf "Baseline: %s\n" comp.baseline.name;
  Printf.printf "Compared: %s\n" comp.compared.name;

  if comp.speedup = nan then Printf.printf "Cannot compare: insufficient data\n"
  else (
    Printf.printf "\n%s is %.2fx %s than %s\n" comp.compared.name
      (abs_float comp.speedup)
      (if comp.speedup > 1.0 then "faster" else "slower")
      comp.baseline.name;

    Printf.printf "95%% CI: [%.2fx, %.2fx]\n" comp.speedup_ci_lower
      comp.speedup_ci_upper;

    if comp.significant then
      Printf.printf "This difference is statistically significant (p < %.2f)\n"
        (1. -. 0.95)
    else Printf.printf "This difference is NOT statistically significant\n";

    Printf.printf "\nBaseline: %s (±%.2f%%)\n"
      (format_time_ns comp.baseline.time_stats.avg)
      (comp.baseline.time_stats.std_dev /. comp.baseline.time_stats.avg *. 100.);

    Printf.printf "Compared: %s (±%.2f%%)\n"
      (format_time_ns comp.compared.time_stats.avg)
      (comp.compared.time_stats.std_dev /. comp.compared.time_stats.avg *. 100.))

let tabulate ?(confidence = 0.95) ?(cpu_selector = `Process)
    (results : analysis_result list) =
  if confidence < 0. || confidence > 1. then
    invalid_arg "Ubench.tabulate: confidence < 0. or > 1.";
  let len = List.length results in
  if len = 0 then invalid_arg "Ubench.tabulate: empty list of results";

  (* Select CPU time basis *)
  let cpu_time m =
    match cpu_selector with
    | `Process -> m.time_ns (* user + system *)
    | `User -> m.utime_ns
    | `System -> m.stime_ns
    | `Children -> m.cutime_ns +. m.cstime_ns
    | `All -> m.time_ns +. m.cutime_ns +. m.cstime_ns
  in

  (* Compute rates for each benchmark *)
  let comp_rates_cpu (results : analysis_result) =
    let n = List.length results.measurements in
    if n = 0 then (results.name, 0, nan, 0.)
    else
      let rates =
        List.map
          (fun m -> float m.runs /. (cpu_time m /. 1e9))
          results.measurements
      in
      let rates_array = Array.of_list rates in
      let avg_rate = mean rates_array in
      let variance =
        if n > 1 then
          let sum_sq =
            Array.fold_left
              (fun acc r ->
                let diff = r -. avg_rate in
                acc +. (diff *. diff))
              0. rates_array
          in
          sum_sq /. float (n - 1)
        else 0.
      in
      (results.name, n, avg_rate, variance)
  in

  let rates = List.map comp_rates_cpu results in
  let sorted_rates =
    List.sort (fun (_, _, r1, _) (_, _, r2, _) -> Float.compare r1 r2) rates
  in

  (* Determine if displaying as rate or time *)
  let display_as_rate =
    let _, _, median_rate, _ = List.nth sorted_rates (len / 2) in
    median_rate > 1.
  in

  (* Build comparison table *)
  Printf.printf "\n=== Cross-Benchmark Comparison ===\n";
  Printf.printf "%-20s %15s" ""
    (if display_as_rate then "Rate (/s)" else "Time (s)");
  List.iter (fun (name, _, _, _) -> Printf.printf " %12s" name) sorted_rates;
  Printf.printf "\n";

  (* Print each benchmark row *)
  List.iter
    (fun (row_name, row_n, row_rate, row_var) ->
      Printf.printf "%-20s" row_name;

      (* Print rate or time *)
      if display_as_rate then Printf.printf " %15.2f" row_rate
      else
        Printf.printf " %15.6f" (if row_rate > 0. then 1. /. row_rate else 0.);

      (* Print comparisons *)
      List.iter
        (fun (col_name, col_n, col_rate, col_var) ->
          if row_name = col_name then Printf.printf " %12s" "--"
          else if row_n = 0 || col_n = 0 || row_rate = nan || col_rate = nan
          then Printf.printf " %12s" "N/A"
          else
            let ratio = row_rate /. col_rate in
            let pct = (ratio -. 1.) *. 100. in
            let is_significant =
              different_rates (1. -. confidence) row_n row_rate row_var col_n
                col_rate col_var
            in
            if is_significant then Printf.printf " %11.0f%%" pct
            else Printf.printf " [%9.0f%%]" pct)
        sorted_rates;
      Printf.printf "\n")
    sorted_rates;

  Printf.printf
    "\n\
     Note: [brackets] indicate difference is not statistically significant (p \
     > %.2f)\n"
    (1. -. confidence);
  flush stdout

let print_pretty_table ?(ascii_only = false) ?(sort_by_wall = true) results =
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

  (* Find the fastest times and lowest memory for color coding *)
  let fastest_wall =
    List.fold_left
      (fun acc r -> min acc r.wall_stats.avg)
      Float.max_float results
  in
  let fastest_cpu =
    List.fold_left
      (fun acc r -> min acc r.time_stats.avg)
      Float.max_float results
  in
  let lowest_memory =
    List.fold_left
      (fun acc r -> min acc r.memory_stats.avg)
      Float.max_float results
  in

  (* Sort results by wall-clock time unless disabled. *)
  let sorted_results =
    if sort_by_wall then
      List.sort
        (fun r1 r2 -> Float.compare r1.wall_stats.avg r2.wall_stats.avg)
        results
    else results
  in

  (* Create row data with coloring *)
  let rows_data =
    List.map
      (fun r ->
        let wall_avg = r.wall_stats.avg in
        let cpu_avg = r.time_stats.avg in
        let vs_fastest =
          if fastest_wall = 0. then Float.infinity else wall_avg /. fastest_wall
        in
        let speedup =
          if wall_avg = 0. then Float.infinity else fastest_wall /. wall_avg
        in
        ( r,
          [
            r.name;
            (if wall_avg = fastest_wall then colorize green else fun x -> x)
              (format_time_ns wall_avg);
            (if cpu_avg = fastest_cpu then colorize green else fun x -> x)
              (format_time_ns cpu_avg);
            (if r.memory_stats.avg = lowest_memory then colorize cyan
             else fun x -> x)
              (format_words r.memory_stats.avg);
            (if speedup >= 1.0 then colorize green else fun x -> x)
              (Printf.sprintf "%.2fx" speedup);
            (if vs_fastest = 1.0 then colorize green else fun x -> x)
              (Printf.sprintf "%.0f%%" (vs_fastest *. 100.0));
          ] ))
      sorted_results
  in

  let headers =
    [ "Name"; "Wall/Run"; "CPU/Run"; "mWd/Run"; "Speedup"; "vs Fastest" ]
  in

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

  (* Box drawing characters (Unicode or ASCII) *)
  let ( top_left,
        top_mid,
        top_right,
        mid_left,
        mid_mid,
        mid_right,
        bot_left,
        bot_mid,
        bot_right,
        hline,
        vline ) =
    if ascii_only then ("+", "+", "+", "+", "+", "+", "+", "+", "+", "-", "|")
    else
      ( "\u{250C}",
        "\u{252C}",
        "\u{2510}",
        "\u{251C}",
        "\u{253C}",
        "\u{2524}",
        "\u{2514}",
        "\u{2534}",
        "\u{2518}",
        "\u{2500}",
        "\u{2502}" )
  in

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

(* Command-line argument parsing using standard library Arg module *)

let parse_quota s =
  if String.contains s 's' then
    try
      let time_str = String.sub s 0 (String.index s 's') in
      Time_limit (float_of_string time_str)
    with _ -> failwith ("Invalid time format: " ^ s)
  else if String.contains s 'x' then
    try
      let iter_str = String.sub s 0 (String.index s 'x') in
      Iteration_limit (int_of_string iter_str)
    with _ -> failwith ("Invalid iteration format: " ^ s)
  else if String.contains s '%' then
    try
      let cv_str = String.sub s 0 (String.index s '%') in
      Variance_limit (float_of_string cv_str /. 100.0)
    with _ -> failwith ("Invalid variance format: " ^ s)
  else
    try Time_limit (float_of_string s)
    with _ -> failwith ("Invalid quota format: " ^ s)

let parse_output_format = function
  | "table" | "pretty" -> Pretty_table
  | "json" -> JSON
  | "csv" -> CSV
  | s ->
      failwith ("Invalid output format: " ^ s ^ ". Must be: table, json, or csv")

(* Internal representation *)
type benchmark_impl = string * (unit -> unit)

(* Public abstract type that can be either a single benchmark or a group *)
type benchmark = Single of benchmark_impl | Group of string * benchmark list

let bench name f = Single (name, fun () -> ignore (f ()))
let create name f = bench name f (* Keep create for backwards compatibility *)

(* Create a benchmark group with a name prefix *)
let group name benchmarks = Group (name, benchmarks)
let create_group = group (* Keep create_group for backwards compatibility *)

(* Flatten benchmarks into a list of concrete benchmarks with prefixed names *)
let rec flatten ?(prefix = "") benchmark =
  match benchmark with
  | Single (name, f) ->
      let full_name = if prefix = "" then name else prefix ^ "/" ^ name in
      [ (full_name, f) ]
  | Group (name, children) ->
      let new_prefix = if prefix = "" then name else prefix ^ "/" ^ name in
      List.concat (List.map (flatten ~prefix:new_prefix) children)

(* Convert a list of benchmarks to a flat list of implementations *)
let flatten_benchmarks benchmarks =
  List.concat (List.map (flatten ~prefix:"") benchmarks)

let bench_with_setup :
    string ->
    setup:(unit -> 'a) ->
    teardown:('a -> unit) ->
    f:('a -> 'b) ->
    benchmark =
 fun name ~setup ~teardown ~f ->
  Single
    ( name,
      fun () ->
        let resource = setup () in
        Fun.protect
          ~finally:(fun () -> teardown resource)
          (fun () -> ignore (f resource)) )

let create_with_setup = bench_with_setup (* Keep for backwards compatibility *)

let bench_param :
    string -> (param:'a -> 'b) -> params:(string * 'a) list -> benchmark list =
 fun base_name f ~params ->
  List.map
    (fun (param_name, param_value) ->
      let name = Printf.sprintf "%s[%s]" base_name param_name in
      Single (name, fun () -> ignore (f ~param:param_value)))
    params

let create_param = bench_param (* Keep for backwards compatibility *)

let run_with_cli_config quota output_format fork_benchmarks warmup stabilize_gc
    verbose ascii_only sort_by_wall benchmarks =
  let config =
    {
      mode = Throughput 1.0;
      quota;
      warmup_iterations = warmup;
      min_measurements = 3;
      stabilize_gc;
      geometric_scale = 1.3;
      fork_benchmarks;
      regressions =
        [
          (Time_per_run, [ One; Runs ], false); (Memory_per_run, [ Runs ], true);
        ];
      custom_measurer = None;
      ascii_only;
      null_loop_subtraction = true;
      min_cpu = 0.002;
      repeat = 1;
      progress_callback = None;
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
                  wall_stats = compute_statistics [||];
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
          wall_stats = result.wall_stats;
          memory_stats = result.memory_stats;
          regressions = result.regressions;
          total_time = result.total_time;
          total_runs = result.total_runs;
        })
      benchmarks
  in

  Printf.printf "\nBenchmark Results:\n";
  (match output_format with
  | Pretty_table ->
      print_pretty_table ~ascii_only:config.ascii_only ~sort_by_wall results
  | JSON -> print_json results
  | CSV -> print_csv results);

  (* Print regression analysis if verbose mode *)
  if verbose then print_regression_analysis results;

  results

let run_silent ?(config = default_config) benchmarks =
  let benchmarks = flatten_benchmarks benchmarks in
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
                  wall_stats = compute_statistics [||];
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
          wall_stats = result.wall_stats;
          memory_stats = result.memory_stats;
          regressions = result.regressions;
          total_time = result.total_time;
          total_runs = result.total_runs;
        })
      benchmarks
  in
  results

let run_and_print ?(config = default_config) ?(output_format = Pretty_table)
    ?(verbose = false) ?(sort_by_wall = true) benchmarks =
  let results = run_silent ~config benchmarks in

  Printf.printf "\nBenchmark Results:\n";
  (match output_format with
  | Pretty_table ->
      print_pretty_table ~ascii_only:config.ascii_only ~sort_by_wall results
  | JSON -> print_json results
  | CSV -> print_csv results);

  if verbose then print_regression_analysis results;
  results

let run ?(config = default_config) ?(output_format = Pretty_table)
    ?(verbose = false) ?(quota = config.quota)
    ?(warmup = config.warmup_iterations) ?(format = output_format)
    ?(sort_by_wall = true) benchmarks =
  let config =
    (config |> fun c -> { c with quota }) |> fun c ->
    { c with warmup_iterations = warmup }
  in
  run_and_print ~config ~output_format:format ~verbose ~sort_by_wall benchmarks

(* CLI using standard library Arg module *)
let run_cli benchmarks =
  let benchmarks = flatten_benchmarks benchmarks in
  let quota = ref (Time_limit 0.3) in
  let output_format = ref Pretty_table in
  let fork_benchmarks = ref false in
  let warmup = ref 1 in
  let stabilize_gc = ref false in
  let verbose = ref false in
  let ascii_only = ref false in
  let help = ref false in

  let spec =
    [
      ( "-q",
        Arg.String (fun s -> quota := parse_quota s),
        "QUOTA  Time limit (e.g., '5s'), iteration limit (e.g., '1000x'), or \
         variance limit (e.g., '1%')" );
      ( "--quota",
        Arg.String (fun s -> quota := parse_quota s),
        "QUOTA  Time limit (e.g., '5s'), iteration limit (e.g., '1000x'), or \
         variance limit (e.g., '1%')" );
      ( "-f",
        Arg.String (fun s -> output_format := parse_output_format s),
        "FORMAT  Output format: table, json, or csv (default: table)" );
      ( "--format",
        Arg.String (fun s -> output_format := parse_output_format s),
        "FORMAT  Output format: table, json, or csv (default: table)" );
      ( "--fork",
        Arg.Set fork_benchmarks,
        "  Run each benchmark in a separate process" );
      ("-w", Arg.Set_int warmup, "N  Number of warmup iterations (default: 3)");
      ( "--warmup",
        Arg.Set_int warmup,
        "N  Number of warmup iterations (default: 3)" );
      ("--gc", Arg.Set stabilize_gc, "  Stabilize GC between measurements");
      ("-v", Arg.Set verbose, "  Show regression analysis");
      ("--verbose", Arg.Set verbose, "  Show regression analysis");
      ( "--ascii-only",
        Arg.Set ascii_only,
        "  Use ASCII characters only (no Unicode)" );
      ("-h", Arg.Set help, "  Display this help message");
      ("--help", Arg.Set help, "  Display this help message");
    ]
  in

  let usage_msg =
    "ubench - Universal benchmarking tool for OCaml\n\nUsage: " ^ Sys.argv.(0)
    ^ " [OPTIONS]\n\nOptions:"
  in

  (* Parse arguments *)
  let anon_arg s =
    Printf.eprintf "Warning: unexpected argument '%s' ignored\n" s
  in

  try
    Arg.parse spec anon_arg usage_msg;

    if !help then (
      Arg.usage spec usage_msg;
      exit 0);

    (* Run benchmarks with parsed configuration *)
    ignore
      (run_with_cli_config !quota !output_format !fork_benchmarks !warmup
         !stabilize_gc !verbose !ascii_only true benchmarks)
  with
  | Arg.Bad msg ->
      Printf.eprintf "%s\n" msg;
      exit 1
  | Arg.Help msg ->
      Printf.printf "%s\n" msg;
      exit 0
  | Failure msg ->
      Printf.eprintf "Error: %s\n" msg;
      exit 1

(* Convenience functions *)
let with_time_limit seconds config = { config with quota = Time_limit seconds }

let with_iteration_limit iters config =
  { config with quota = Iteration_limit iters }

let with_warmup warmup config = { config with warmup_iterations = warmup }

let with_gc_stabilization enabled config =
  { config with stabilize_gc = enabled }

let with_fork enabled config = { config with fork_benchmarks = enabled }
let with_regressions regressions config = { config with regressions }

let with_variance_limit cv_threshold config =
  { config with quota = Variance_limit cv_threshold }

let with_custom_measurer measurer config =
  { config with custom_measurer = Some measurer }

let with_ascii_only enabled config = { config with ascii_only = enabled }
let with_min_measurements n config = { config with min_measurements = n }

let with_progress_callback callback config =
  { config with progress_callback = Some callback }

(* Save benchmark results to a JSON file *)
let save_results_json filename results =
  let oc = open_out filename in
  let json =
    let result_to_json (r : analysis_result) =
      Printf.sprintf
        {|{"name":"%s","time_stats":{"avg":%.2f,"min":%.2f,"max":%.2f,"std_dev":%.2f,"ci95_lower":%.2f,"ci95_upper":%.2f},"wall_stats":{"avg":%.2f,"min":%.2f,"max":%.2f,"std_dev":%.2f,"ci95_lower":%.2f,"ci95_upper":%.2f},"memory_stats":{"avg":%.2f,"min":%.2f,"max":%.2f,"std_dev":%.2f,"ci95_lower":%.2f,"ci95_upper":%.2f},"total_time_ns":%.2f,"total_runs":%d}|}
        r.name r.time_stats.avg r.time_stats.min r.time_stats.max
        r.time_stats.std_dev r.time_stats.ci95_lower r.time_stats.ci95_upper
        r.wall_stats.avg r.wall_stats.min r.wall_stats.max r.wall_stats.std_dev
        r.wall_stats.ci95_lower r.wall_stats.ci95_upper r.memory_stats.avg
        r.memory_stats.min r.memory_stats.max r.memory_stats.std_dev
        r.memory_stats.ci95_lower r.memory_stats.ci95_upper r.total_time
        r.total_runs
    in
    Printf.sprintf "[%s]" (String.concat "," (List.map result_to_json results))
  in
  output_string oc json;
  close_out oc

(* Save benchmark results to a CSV file *)
let save_results_csv filename results =
  let oc = open_out filename in
  Printf.fprintf oc
    "name,time_avg,time_min,time_max,time_std_dev,time_ci95_lower,time_ci95_upper,wall_avg,wall_min,wall_max,wall_std_dev,wall_ci95_lower,wall_ci95_upper,memory_avg,memory_min,memory_max,memory_std_dev,memory_ci95_lower,memory_ci95_upper,total_runs\n";
  List.iter
    (fun (r : analysis_result) ->
      Printf.fprintf oc
        "%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d\n"
        r.name r.time_stats.avg r.time_stats.min r.time_stats.max
        r.time_stats.std_dev r.time_stats.ci95_lower r.time_stats.ci95_upper
        r.wall_stats.avg r.wall_stats.min r.wall_stats.max r.wall_stats.std_dev
        r.wall_stats.ci95_lower r.wall_stats.ci95_upper r.memory_stats.avg
        r.memory_stats.min r.memory_stats.max r.memory_stats.std_dev
        r.memory_stats.ci95_lower r.memory_stats.ci95_upper r.total_runs)
    results;
  close_out oc
