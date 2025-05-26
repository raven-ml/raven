let string_of_dtype : type a b. (a, b) Nx.dtype -> string = function
  | Float32 -> "float32"
  | Float64 -> "float64"
  | _ -> "other" (* Only float32 and float64 are used here *)

(* Helper for binary operations: takes two arrays *)
let binary_op_bench : type a b.
    ((a, b) Nx.t -> (a, b) Nx.t -> (a, b) Nx.t) ->
    int ->
    (a, b) Nx.dtype ->
    unit ->
    unit =
 fun op size dtype ->
  let shape = [| size; size |] in
  (* TODO: Fix Nx.rand and use it instead of ones *)
  let a = Nx.astype dtype (Nx.ones Nx.float32 shape) in
  let b = Nx.astype dtype (Nx.ones Nx.float32 shape) in
  fun () -> op a b |> ignore

(* Helper for unary operations: takes one array *)
let unary_op_bench : type a b.
    ((a, b) Nx.t -> (a, b) Nx.t) -> int -> (a, b) Nx.dtype -> unit -> unit =
 fun op size dtype ->
  let shape = [| size; size |] in
  (* TODO: Fix Nx.rand and use it instead of ones *)
  let a = Nx.astype dtype (Nx.ones Nx.float32 shape) in
  fun () -> op a |> ignore

(* Helper for reduction operations: reduces array to scalar/smaller array *)
let reduction_op_bench : type a b c d.
    ((a, b) Nx.t -> (c, d) Nx.t) -> int -> (a, b) Nx.dtype -> unit -> unit =
 fun op size dtype ->
  let shape = [| size; size |] in
  (* TODO: Fix Nx.rand and use it instead of ones *)
  let a = Nx.astype dtype (Nx.ones Nx.float32 shape) in
  fun () -> op a |> ignore

(* Helper for matrix operations like matmul *)
let matmul_bench : type a b. int -> (a, b) Nx.dtype -> unit -> unit =
 fun size dtype ->
  (* TODO: Fix Nx.rand and use it instead of ones *)
  let a = Nx.astype dtype (Nx.ones Nx.float32 [| size; size |]) in
  let b = Nx.astype dtype (Nx.ones Nx.float32 [| size; size |]) in
  fun () -> Nx.matmul a b |> ignore

(* List of operations to benchmark *)
let operations : type a b.
    int -> (a, b) Nx.dtype -> (string * (unit -> unit)) list =
 fun size dtype ->
  List.concat
    [
      (* Binary operations *)
      [
        ("Addition", binary_op_bench Nx.add size dtype);
        ("Multiplication", binary_op_bench Nx.mul size dtype);
        (* Unary operations *)
        ("Square", unary_op_bench Nx.square size dtype);
      ];
      (* Matrix operations - skip for large sizes *)
      (if size <= 100 then [ ("MatMul", matmul_bench size dtype) ] else []);
      (* Reductions *)
      [ ("Sum", reduction_op_bench Nx.sum size dtype) ];
    ]

let float_operations : type b.
    int -> (float, b) Nx.dtype -> (string * (unit -> unit)) list =
 fun size dtype ->
  [
    (* Float-specific unary operations *)
    ("Sqrt", unary_op_bench Nx.sqrt size dtype);
    ("Exp", unary_op_bench Nx.exp size dtype);
  ]

(* Generate benchmark tests for all combinations *)
let tests ~sizes =
  let tests_on_dtype (type a b) (dtype : (a, b) Nx.dtype) =
    List.concat_map
      (fun size ->
        let ops = operations size dtype in
        List.map
          (fun (op_name, bench_fun) ->
            let name =
              Printf.sprintf "%s on %dx%d %s" op_name size size
                (string_of_dtype dtype)
            in
            Ubench.create name bench_fun)
          ops)
      sizes
  in
  let tests_float_on_dtype (type b) (dtype : (float, b) Nx.dtype) =
    List.concat_map
      (fun size ->
        let ops = float_operations size dtype in
        List.map
          (fun (op_name, bench_fun) ->
            let name =
              Printf.sprintf "%s on %dx%d %s" op_name size size
                (string_of_dtype dtype)
            in
            Ubench.create name bench_fun)
          ops)
      sizes
  in
  List.concat
    [
      tests_on_dtype Float32;
      tests_on_dtype Float64;
      tests_float_on_dtype Float32;
      tests_float_on_dtype Float64;
    ]

(* Run the benchmarks *)
let () =
  print_endline "# Nx Benchmarks";
  let tests = tests ~sizes:[ 50; 100; 500 ] in
  let results = Ubench.run ~warmup:1 ~trials:3 ~min_time:0.01 tests in
  Ubench.print_report results
