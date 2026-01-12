(** Ubench - Micro-benchmarking library for OCaml.

    Ubench provides high-precision benchmarking with statistical analysis,
    memory tracking, and regression analysis. It measures execution time and
    memory allocation patterns, offering configurable quotas, warmup phases, and
    output formats.

    {1 Quick Start}

    Create and run a simple benchmark:
    {[
      let bench = Ubench.create "my_function" (fun () -> expensive_computation ()) in
      let results = Ubench.run [bench] in
      (* Results printed automatically in table format *)
    ]}

    Create a benchmarking executable with CLI:
    {[
      (* mybench.ml *)
      let benchmarks = [
        Ubench.create "test1" (fun () -> ...);
        Ubench.create "test2" (fun () -> ...);
      ]

      let () = Ubench.run_cli benchmarks
      (* Run with: ./mybench -q 5s --format json --verbose *)
    ]}

    {1 Key Features}

    - High-precision timing using CPU time (not wall clock)
    - Memory allocation tracking (minor/major/promoted words)
    - Statistical analysis with confidence intervals
    - Regression analysis with optional intercept terms
    - Configurable execution quotas (time, iteration, or variance limits)
    - GC stabilization between measurements
    - Process isolation with fork mode
    - Multiple output formats (table, JSON, CSV)
    - Support for custom measurement functions
    - ASCII-only output mode for compatibility

    {1 Measurement Process}

    Benchmarks run in batches with geometrically increasing sizes to balance
    measurement overhead with precision. Each measurement includes warmup
    iterations, optional GC stabilization, and captures both timing and memory
    allocation data.

    The library computes statistics including mean, standard deviation, min/max
    values, and 95% confidence intervals for both time and memory metrics. *)

(** {1 Types} *)

type progress_info = {
  name : string;  (** Name of the current benchmark *)
  current_measurement : int;  (** Current measurement number *)
  total_measurements : int option;  (** Total measurements if known *)
  elapsed_time : float;  (** Time elapsed since benchmark started *)
  estimated_remaining : float option;
      (** Estimated remaining time if available *)
}
(** Progress information for benchmark execution. *)

(** Execution quota controlling when benchmarks terminate. *)
type quota =
  | Time_limit of float  (** Maximum execution time in seconds *)
  | Iteration_limit of int  (** Maximum number of function calls *)
  | Variance_limit of float
      (** Stop when coefficient of variation falls below threshold *)

type measurement = {
  time_ns : float;  (** CPU time (user + system) in nanoseconds *)
  wall_ns : float;  (** Wall clock time in nanoseconds *)
  utime_ns : float;  (** User CPU time in nanoseconds *)
  stime_ns : float;  (** System CPU time in nanoseconds *)
  cutime_ns : float;  (** Child user CPU time in nanoseconds *)
  cstime_ns : float;  (** Child system CPU time in nanoseconds *)
  cycles : float;  (** CPU cycles (estimated from time if TSC unavailable) *)
  runs : int;  (** Number of function calls in this batch *)
  minor_words : float;  (** Minor heap words allocated *)
  major_words : float;  (** Major heap words allocated *)
  promoted_words : float;  (** Words promoted from minor to major heap *)
  minor_collections : int;  (** Number of minor GC collections *)
  major_collections : int;  (** Number of major GC collections *)
  compactions : int;  (** Number of heap compactions *)
  custom_predictors : (string * float) list;  (** User-defined metrics *)
}
(** Single measurement containing detailed timing and allocation data. *)

(** Predictor variables for regression analysis. *)
type predictor =
  | One  (** Constant predictor for overhead estimation *)
  | Runs  (** Number of function calls *)
  | Time_ns  (** CPU time *)
  | Wall_ns  (** Wall clock time *)
  | Cycles  (** CPU cycles *)
  | User_time  (** User CPU time *)
  | System_time  (** System CPU time *)
  | Child_time  (** Child process time *)
  | Minor_words  (** Minor heap allocation *)
  | Major_words  (** Major heap allocation *)
  | Promoted_words  (** Promoted heap allocation *)
  | Minor_collections  (** Number of minor GC collections *)
  | Major_collections  (** Number of major GC collections *)
  | Compactions  (** Number of heap compactions *)
  | Custom of string  (** Custom predictor by name *)

(** Response variables for regression analysis. *)
type responder =
  | Time_per_run  (** Average CPU time per function call *)
  | Wall_per_run  (** Average wall time per function call *)
  | Memory_per_run  (** Average memory allocation per call *)
  | Total_time  (** Total CPU time *)
  | Total_wall  (** Total wall time *)
  | Allocation_rate  (** Allocation rate (words/second) *)
  | Custom_responder of string  (** Custom response variable *)

type regression_result = {
  responder : responder;  (** Dependent variable *)
  predictors : predictor list;  (** Independent variables *)
  coefficients : float array;  (** Regression coefficients *)
  r_squared : float;  (** Coefficient of determination (0.0 to 1.0) *)
  adjusted_r_squared : float;
      (** Adjusted R² that penalizes for additional predictors *)
  intercept : float option;  (** Intercept term if regression includes one *)
  confidence_intervals : (float * float) array option;
      (** 95% confidence intervals for coefficients *)
}
(** Result of regression analysis. *)

type statistics = {
  avg : float;  (** Arithmetic mean *)
  min : float;  (** Minimum value *)
  max : float;  (** Maximum value *)
  std_dev : float;  (** Standard deviation *)
  ci95_lower : float;  (** Lower bound of 95% confidence interval *)
  ci95_upper : float;  (** Upper bound of 95% confidence interval *)
}
(** Statistical summary of measurements. *)

type bench_data = {
  measurements : measurement list;  (** Raw measurement data *)
  time_stats : statistics;  (** Time per run statistics *)
  wall_stats : statistics;  (** Wall-clock time per run statistics *)
  memory_stats : statistics;  (** Memory per run statistics *)
  regressions : regression_result list;  (** Regression analysis results *)
  total_time : float;  (** Total benchmark execution time *)
  total_runs : int;  (** Total function calls *)
}
(** Internal benchmark result data. *)

type analysis_result = {
  name : string;  (** Benchmark name *)
  measurements : measurement list;  (** Raw measurement data *)
  time_stats : statistics;  (** Time per run statistics *)
  wall_stats : statistics;  (** Wall-clock time per run statistics *)
  memory_stats : statistics;  (** Memory per run statistics *)
  regressions : regression_result list;  (** Regression analysis results *)
  total_time : float;  (** Total benchmark execution time *)
  total_runs : int;  (** Total function calls *)
}
(** Complete benchmark analysis result. *)

(** Benchmark execution mode. *)
type benchmark_mode =
  | Latency of int  (** Fixed number of iterations *)
  | Throughput of float  (** Minimum CPU time in seconds *)

(** Output format for benchmark results. *)
type output_format =
  | Pretty_table  (** Human-readable table with colors and formatting *)
  | JSON  (** Machine-readable JSON format *)
  | CSV  (** Comma-separated values *)

type benchmark
(** Abstract type representing a benchmark or group of benchmarks. *)

(** {1 Configuration} *)

(** Fluent configuration builder module for ergonomic config creation. *)
module Config : sig
  type t = {
    mode : benchmark_mode;  (** Latency or throughput mode *)
    quota : quota;  (** Execution limit *)
    warmup_iterations : int;  (** Number of warmup calls before measurement *)
    min_measurements : int;  (** Minimum number of measurements to collect *)
    stabilize_gc : bool;  (** Whether to compact GC between measurements *)
    geometric_scale : float;  (** Batch size scaling factor (must be > 1.0) *)
    fork_benchmarks : bool;
        (** Whether to run benchmarks in separate processes *)
    regressions : (responder * predictor list * bool) list;
        (** Regression analyses with intercept flag *)
    custom_measurer : ((unit -> unit) -> int -> measurement) option;
        (** Custom measurement function *)
    ascii_only : bool;  (** Use ASCII characters only in output *)
    null_loop_subtraction : bool;  (** Subtract null loop overhead *)
    min_cpu : float;  (** Minimum CPU time for reliable results *)
    repeat : int;  (** Number of times to repeat the benchmark *)
    progress_callback : (progress_info -> unit) option;
        (** Optional callback for progress updates *)
  }
  (** Benchmark configuration options. *)

  val default : t
  (** Default benchmark configuration.

      Uses 1 second time limit, 3 warmup iterations, 10 minimum measurements, GC
      stabilization enabled, 1.5x geometric scaling, no process forking, and
      regression analysis for time_per_run vs runs and memory_per_run vs runs
      with intercept terms. *)

  val time_limit : float -> t -> t
  (** Set time limit quota in seconds *)

  val iteration_limit : int -> t -> t
  (** Set iteration limit quota *)

  val variance_limit : float -> t -> t
  (** Set variance limit quota (coefficient of variation) *)

  val warmup : int -> t -> t
  (** Set number of warmup iterations *)

  val min_measurements : int -> t -> t
  (** Set minimum number of measurements *)

  val gc_stabilization : bool -> t -> t
  (** Enable/disable GC stabilization *)

  val fork : bool -> t -> t
  (** Enable/disable process forking *)

  val ascii_only : bool -> t -> t
  (** Enable/disable ASCII-only output *)

  val geometric_scale : float -> t -> t
  (** Set geometric scaling factor for batch sizes *)

  val regressions : (responder * predictor list * bool) list -> t -> t
  (** Set regression analyses to perform *)

  val custom_measurer : ((unit -> unit) -> int -> measurement) -> t -> t
  (** Set custom measurement function *)

  val progress_callback : (progress_info -> unit) -> t -> t
  (** Set progress callback function *)

  val ( |> ) : t -> (t -> t) -> t
  (** Pipe operator for chaining configuration *)

  val build : t -> t
  (** Finalize configuration (identity function for clarity) *)
end

val default_config : Config.t
(** Default benchmark configuration.

    Alias for [Config.default]. *)

(** {1 Core Functions} *)

exception Matrix_singular of string
(** Raised when matrix operations encounter numerical instability *)

val save_results_json : string -> analysis_result list -> unit
(** [save_results_json filename results] saves benchmark results to a JSON file.
*)

val save_results_csv : string -> analysis_result list -> unit
(** [save_results_csv filename results] saves benchmark results to a CSV file.
*)

(** Abstract benchmark type that can represent individual benchmarks or groups
*)

val bench : string -> (unit -> 'a) -> benchmark
(** [bench name f] creates a benchmark with the given [name] and function [f].

    The function [f] should perform the operation to be measured. It will be
    called multiple times during benchmarking. The result is automatically
    ignored. *)

val group : string -> benchmark list -> benchmark
(** [group name benchmarks] creates a benchmark group with the given name. All
    benchmarks in the group will have their names prefixed with the group name
    when executed. Groups can be nested to create hierarchies. *)

val create : string -> (unit -> 'a) -> benchmark
(** Alias for [bench]. Kept for backwards compatibility. *)

val create_group : string -> benchmark list -> benchmark
(** Alias for [group]. Kept for backwards compatibility. *)

val bench_with_setup :
  string ->
  setup:(unit -> 'a) ->
  teardown:('a -> unit) ->
  f:('a -> 'b) ->
  benchmark
(** [bench_with_setup name ~setup ~teardown ~f] creates a benchmark with
    resource management.

    The [setup] function is called once before measurements begin to initialize
    resources. The [teardown] function is called once after measurements
    complete to clean up. The function [f] receives the resource from [setup]
    and is measured multiple times. The result is automatically ignored. *)

val bench_param :
  string -> (param:'a -> 'b) -> params:(string * 'a) list -> benchmark list
(** [bench_param base_name f ~params] creates parameterized benchmarks.

    Generates one benchmark for each parameter value, with names of the form
    "base_name[param_name]". Useful for testing performance across different
    input sizes or configurations. The result is automatically ignored. *)

val create_with_setup :
  string ->
  setup:(unit -> 'a) ->
  teardown:('a -> unit) ->
  f:('a -> 'b) ->
  benchmark
(** [create_with_setup name ~setup ~teardown ~f] creates a benchmark with
    resource management.

    The [setup] function is called once before measurements begin, [teardown] is
    called after measurements complete, and [f] receives the resource from
    setup. This is useful for benchmarks that require expensive initialization
    or cleanup.

    Example:
    {[
      let bench =
        create_with_setup "file_read"
          ~setup:(fun () -> open_in "large_file.txt")
          ~teardown:close_in
          ~f:(fun ic -> really_input_string ic 1024 |> ignore)
    ]} *)

val create_param :
  string -> (param:'a -> 'b) -> params:(string * 'a) list -> benchmark list
(** [create_param base_name f ~params] creates multiple benchmarks for different
    parameter values.

    This generates one benchmark for each parameter in [params], with names
    formatted as "base_name[param_name]". Useful for testing performance across
    different input sizes or configurations.

    Example:
    {[
      let benches =
        create_param "sort"
          (fun ~param -> Array.sort compare param)
          ~params:
            [
              ("small", Array.make 10 0);
              ("medium", Array.make 100 0);
              ("large", Array.make 1000 0);
            ]
    ]} *)

val run :
  ?config:Config.t ->
  ?output_format:output_format ->
  ?verbose:bool ->
  ?quota:quota ->
  ?warmup:int ->
  ?format:output_format ->
  ?sort_by_wall:bool ->
  benchmark list ->
  analysis_result list
(** [run ?config ?output_format ?verbose ?quota ?warmup ?format ?sort_by_wall
    benchmarks]
    executes the given benchmarks and returns analysis results.

    Results are automatically printed using the specified output format. The
    function performs statistical analysis and regression modeling on the
    collected measurements. The shorthand parameters allow quick overrides
    without building a full config.

    @param config Benchmark configuration (defaults to {!default_config})
    @param output_format Output format (defaults to [Pretty_table])
    @param verbose Whether to print regression analysis (defaults to [false])
    @param quota Quick override for execution quota
    @param warmup Quick override for warmup iterations
    @param format Alias for output_format for convenience
    @param sort_by_wall Sort table output by wall-clock time (defaults to [true])
*)

val run_silent : ?config:Config.t -> benchmark list -> analysis_result list
(** [run_silent ?config benchmarks] executes benchmarks without printing
    results.

    Useful when you want to process results programmatically or apply custom
    formatting. Progress information is still printed to stdout. *)

val run_and_print :
  ?config:Config.t ->
  ?output_format:output_format ->
  ?verbose:bool ->
  ?sort_by_wall:bool ->
  benchmark list ->
  analysis_result list
(** [run_and_print ?config ?output_format ?verbose ?sort_by_wall benchmarks]
    executes
    benchmarks and prints results.

    Equivalent to [run] but with explicit naming to distinguish from
    [run_silent]. *)

val with_time_limit : float -> Config.t -> Config.t
(** [with_time_limit seconds config] sets a time limit quota. *)

val with_iteration_limit : int -> Config.t -> Config.t
(** [with_iteration_limit iters config] sets an iteration limit quota. *)

val with_variance_limit : float -> Config.t -> Config.t
(** [with_variance_limit cv_threshold config] sets a variance limit quota. *)

val with_warmup : int -> Config.t -> Config.t
(** [with_warmup iterations config] sets the number of warmup iterations. *)

val with_min_measurements : int -> Config.t -> Config.t
(** [with_min_measurements n config] sets the minimum number of measurements. *)

val with_gc_stabilization : bool -> Config.t -> Config.t
(** [with_gc_stabilization enabled config] enables or disables GC stabilization.
*)

val with_fork : bool -> Config.t -> Config.t
(** [with_fork enabled config] enables or disables process forking for
    benchmarks. *)

val with_ascii_only : bool -> Config.t -> Config.t
(** [with_ascii_only enabled config] enables or disables ASCII-only output. *)

val with_custom_measurer :
  ((unit -> unit) -> int -> measurement) -> Config.t -> Config.t
(** [with_custom_measurer measurer config] sets a custom measurement function.
*)

val with_regressions :
  (responder * predictor list * bool) list -> Config.t -> Config.t
(** [with_regressions regressions config] sets the regression analyses to
    perform. Each tuple contains (responder, predictors, include_intercept). *)

val with_progress_callback : (progress_info -> unit) -> Config.t -> Config.t
(** [with_progress_callback callback config] sets a progress callback function.

    The callback is invoked periodically during benchmark execution with
    information about the current progress. Useful for GUI integration or custom
    progress reporting. *)

(** {1 CLI Integration} *)

val run_cli : benchmark list -> unit
(** [run_cli benchmarks] runs benchmarks with command-line interface.

    Parses command-line arguments using the standard library Arg module and
    executes benchmarks accordingly. Supports options for quota, output format,
    forking, warmup iterations, GC stabilization, verbose output, and ASCII-only
    mode. Call this as the main entry point for a benchmarking executable.

    Available options:
    - [-q QUOTA] or [--quota QUOTA]: Time (5s), iteration (1000x), or variance
      (1%) limit
    - [-f FORMAT] or [--format FORMAT]: Output format (table, json, csv)
    - [--fork]: Run benchmarks in separate processes
    - [-w N] or [--warmup N]: Number of warmup iterations
    - [--gc]: Stabilize GC between measurements
    - [-v] or [--verbose]: Show regression analysis
    - [--ascii-only]: Use ASCII characters only
    - [-h] or [--help]: Display help message *)

(** {1 Advanced Analysis} *)

val ordinary_least_squares :
  ?intercept:bool ->
  measurement list ->
  responder:responder ->
  predictors:predictor list ->
  regression_result
(** [ordinary_least_squares ?intercept measurements ~responder ~predictors]
    performs OLS regression.

    @param intercept Whether to include an intercept term (defaults to [true])
    @param measurements List of measurements for analysis
    @param responder Dependent variable to predict
    @param predictors Independent variables for prediction
    @return Regression results including coefficients, R², and adjusted R² *)

val compute_statistics : float array -> statistics
(** [compute_statistics values] computes statistical summary of values.

    Returns mean, min, max, standard deviation, and 95% confidence interval. For
    empty arrays, returns zeros for all statistics. *)

val stabilize_gc : unit -> unit
(** [stabilize_gc ()] compacts the heap until the number of live words
    stabilizes.

    Useful for reducing measurement noise due to GC activity. May fail if heap
    cannot be stabilized within 10 attempts. *)

val measure_one_batch :
  ?null_loop_subtraction:bool -> (unit -> unit) -> int -> measurement
(** [measure_one_batch ?null_loop_subtraction f batch_size] measures [f] called
    [batch_size] times.

    @param null_loop_subtraction
      Whether to subtract null loop overhead (defaults to [true])

    Includes warmup iterations and captures timing and allocation data. Uses CPU
    time rather than wall-clock time for better precision. Null loop subtraction
    improves accuracy by removing measurement overhead. *)

val run_bench_with_config : Config.t -> (unit -> unit) -> bench_data
(** [run_bench_with_config config f] benchmarks function [f] with given config.

    Handles adaptive batch sizing, quota checking, and measurement collection.
    Returns raw benchmark data for further analysis. *)

(** {1 Output Formatting} *)

val print_pretty_table :
  ?ascii_only:bool -> ?sort_by_wall:bool -> analysis_result list -> unit
(** [print_pretty_table ?ascii_only ?sort_by_wall results] prints results as
    formatted table.

    @param ascii_only Use ASCII characters only (defaults to [false])
    @param sort_by_wall Sort rows by wall-clock time (defaults to [true])

    Uses colors and Unicode box-drawing characters unless [ascii_only] is true.
    Shows time per run, memory allocation, and comparison to fastest. *)

val print_json : analysis_result list -> unit
(** [print_json results] prints results in JSON format.

    Includes complete statistics and regression analysis results. *)

val print_csv : analysis_result list -> unit
(** [print_csv results] prints results in CSV format.

    Includes all statistics fields and regression R² values. *)

val print_regression_analysis : analysis_result list -> unit
(** [print_regression_analysis results] prints detailed regression analysis.

    Shows regression equations with coefficients, R², and adjusted R² values. *)

(** {1 Cross-Benchmark Comparison} *)

type comparison_result = {
  baseline : analysis_result;  (** Baseline benchmark *)
  compared : analysis_result;  (** Benchmark being compared *)
  speedup : float;  (** Speedup factor (compared/baseline) *)
  speedup_ci_lower : float;  (** Lower bound of 95% CI for speedup *)
  speedup_ci_upper : float;  (** Upper bound of 95% CI for speedup *)
  significant : bool;  (** Whether difference is statistically significant *)
  p_value : float option;  (** P-value if computed *)
}
(** Result of comparing two benchmarks. *)

val compare :
  ?confidence:float -> analysis_result -> analysis_result -> comparison_result
(** [compare ?confidence baseline compared] performs statistical comparison.

    @param confidence
      Confidence level for significance testing (defaults to 0.95)
    @return Comparison result with speedup factor and statistical significance

    The speedup factor is the ratio of performance (compared/baseline). Values
    greater than 1.0 indicate the compared benchmark is faster. Statistical
    significance is determined using Student's t-test. *)

val print_comparison : comparison_result -> unit
(** [print_comparison comp] prints a human-readable comparison summary.

    Shows speedup factor, confidence intervals, statistical significance, and
    timing statistics for both benchmarks. *)

val tabulate :
  ?confidence:float ->
  ?cpu_selector:[ `Process | `User | `System | `Children | `All ] ->
  analysis_result list ->
  unit
(** [tabulate ?confidence ?cpu_selector results] displays cross-benchmark
    comparison table.

    @param confidence
      Confidence level for statistical significance (defaults to [0.95])
    @param cpu_selector
      Which CPU time to use for comparison (defaults to [`Process])

    Shows pairwise comparisons between benchmarks with percentage differences.
    Statistical significance is determined using Student's t-test.
    Non-significant differences are shown in brackets. *)

(** {1 Statistical Functions} *)

val log_gamma : float -> float
(** [log_gamma x] computes the natural logarithm of the gamma function.

    Uses the Lanczos approximation for numerical stability. *)

val betai : float -> float -> float -> float
(** [betai x a b] computes the incomplete beta function I_x(a,b).

    Used in Student's t-distribution calculations. *)

val cpl_student_t : float -> float -> float
(** [cpl_student_t t nu] computes the complement of Student's t CDF.

    Returns P(T > t) where T follows a t-distribution with nu degrees of
    freedom. *)

val different_rates :
  float -> int -> float -> float -> int -> float -> float -> bool
(** [different_rates significance n1 r1 s1 n2 r2 s2] tests if two rates differ.

    @param significance Significance level (e.g., 0.05 for 95% confidence)
    @param n1 Number of measurements for first benchmark
    @param r1 Mean rate for first benchmark
    @param s1 Variance for first benchmark
    @param n2 Number of measurements for second benchmark
    @param r2 Mean rate for second benchmark
    @param s2 Variance for second benchmark
    @return
      [true] if rates are statistically different at given significance level *)

(** {1 Utility Functions} *)

val format_time_ns : float -> string
(** [format_time_ns ns] formats nanoseconds as human-readable string.

    Automatically selects appropriate units (ns, μs, ms, s) with underscores for
    thousands separation. *)

val format_words : float -> string
(** [format_words w] formats word count as human-readable string.

    Automatically selects appropriate units (w, kw, Mw). *)

val format_number : float -> string
(** [format_number n] formats number with appropriate suffix.

    Uses k for thousands, M for millions, G for billions. *)
