let string_of_dtype : type a b. (a, b) Ndarray.dtype -> string = function
  | Float32 -> "float32"
  | Float64 -> "float64"
  | _ -> "other" (* Only float32 and float64 are used here *)

(* Helper for binary operations: takes two arrays *)
let binary_op_bench : type a b.
    ((a, b) Ndarray.t -> (a, b) Ndarray.t -> (a, b) Ndarray.t) ->
    int ->
    (a, b) Ndarray.dtype ->
    unit ->
    unit =
 fun op size dtype ->
  let shape = [| size; size |] in
  let a = Ndarray.astype dtype (Ndarray.rand Ndarray.float32 shape) in
  let b = Ndarray.astype dtype (Ndarray.rand Ndarray.float32 shape) in
  fun () -> op a b |> ignore

(* Helper for unary operations: takes one array *)
let _unary_op_bench : type a b.
    ((a, b) Ndarray.t -> (a, b) Ndarray.t) ->
    int ->
    (a, b) Ndarray.dtype ->
    unit ->
    unit =
 fun op size dtype ->
  let shape = [| size; size |] in
  let a = Ndarray.astype dtype (Ndarray.rand Ndarray.float32 shape) in
  fun () -> op a |> ignore

(* List of operations to benchmark *)
let operations : type a b.
    int -> (a, b) Ndarray.dtype -> (string * (unit -> unit)) list =
 fun size dtype ->
  [
    ("Addition", binary_op_bench Ndarray.add size dtype);
    (* ("Subtraction", binary_op_bench Ndarray.sub size dtype);
       ("Multiplication", binary_op_bench Ndarray.mul size dtype); ("Division",
       binary_op_bench Ndarray.div size dtype); ("Power", binary_op_bench
       Ndarray.pow size dtype); *)
  ]

let float_operations : type b.
    int -> (float, b) Ndarray.dtype -> (string * (unit -> unit)) list =
 fun _size _dtype ->
  [ (* ("Square Root", unary_op_bench Ndarray.sqrt size dtype); *)
    (* ("Mean", unary_op_bench Ndarray.mean size dtype); *) ]

(* Generate benchmark tests for all combinations *)
let tests ~sizes =
  let tests_on_dtype (type a b) (dtype : (a, b) Ndarray.dtype) =
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
  let tests_float_on_dtype (type b) (dtype : (float, b) Ndarray.dtype) =
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
  print_endline "# Ndarray Benchmarks";
  let tests = tests ~sizes:[ 50; 100; 500; 1000; 2000 ] in
  let results = Ubench.run ~warmup:0 ~trials:1 ~min_time:1. tests in
  Ubench.print_report results
