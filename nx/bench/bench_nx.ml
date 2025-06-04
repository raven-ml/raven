open Nx

(* Helper to create test arrays *)
let make_array dtype shape =
  (* TODO: Fix Nx.rand and use it instead of ones *)
  Nx.astype dtype (Nx.ones float32 shape)

(* Benchmark functions *)
let bench_add : type a b. int -> (a, b) dtype -> unit -> unit =
 fun size dtype () ->
  let a = make_array dtype [| size; size |] in
  let b = make_array dtype [| size; size |] in
  Nx.add a b |> ignore

let bench_mul : type a b. int -> (a, b) dtype -> unit -> unit =
 fun size dtype () ->
  let a = make_array dtype [| size; size |] in
  let b = make_array dtype [| size; size |] in
  Nx.mul a b |> ignore

let bench_square : type a b. int -> (a, b) dtype -> unit -> unit =
 fun size dtype () ->
  let a = make_array dtype [| size; size |] in
  Nx.square a |> ignore

let bench_sqrt : type b. int -> (float, b) dtype -> unit -> unit =
 fun size dtype () ->
  let a = make_array dtype [| size; size |] in
  Nx.sqrt a |> ignore

let bench_exp : type b. int -> (float, b) dtype -> unit -> unit =
 fun size dtype () ->
  let a = make_array dtype [| size; size |] in
  Nx.exp a |> ignore

let bench_sum : type a b. int -> (a, b) dtype -> unit -> unit =
 fun size dtype () ->
  let a = make_array dtype [| size; size |] in
  Nx.sum a |> ignore

let bench_matmul : type a b. int -> (a, b) dtype -> unit -> unit =
 fun size dtype () ->
  let a = make_array dtype [| size; size |] in
  let b = make_array dtype [| size; size |] in
  Nx.matmul a b |> ignore

let bench_conv2d : type a b. int -> int -> (a, b) dtype -> unit -> unit =
 fun size kernel_size dtype () ->
  let input = make_array dtype [| 1; 3; size; size |] in
  let kernel = make_array dtype [| 16; 3; kernel_size; kernel_size |] in
  Nx.convolve2d ~padding_mode:`Same input kernel |> ignore

(* Generate benchmarks *)
let make_benchmarks () =
  let sizes = [50; 100] in  (* Reduced for faster runs *)
  let dtype_name : type a b. (a, b) dtype -> string = function
    | Float32 -> "f32"
    | Float64 -> "f64"
    | _ -> "other"
  in
  
  let bench_for_dtype : type a b. (a, b) dtype -> _ =
    fun dtype ->
      List.concat_map (fun size ->
        let name s = Printf.sprintf "%s %dx%d %s" s size size (dtype_name dtype) in
        List.concat [
          (* Basic operations *)
          [ Ubench.create (name "add") (bench_add size dtype);
            Ubench.create (name "mul") (bench_mul size dtype);
            Ubench.create (name "square") (bench_square size dtype);
            Ubench.create (name "sum") (bench_sum size dtype);
          ];
          
          (* Float-specific operations *)
          (match dtype with
           | Float32 -> 
               [ Ubench.create (name "sqrt") (bench_sqrt size Float32);
                 Ubench.create (name "exp") (bench_exp size Float32); ]
           | Float64 ->
               [ Ubench.create (name "sqrt") (bench_sqrt size Float64);
                 Ubench.create (name "exp") (bench_exp size Float64); ]
           | _ -> []);
          
          (* Matrix operations - skip large sizes *)
          (if size < 100 then
             [ Ubench.create (name "matmul") (bench_matmul size dtype) ]
           else []);
          
          (* Convolution - skip large sizes *)
          (if size < 100 then
             [ Ubench.create (name "conv2d-3x3") (bench_conv2d size 3 dtype);
               Ubench.create (name "conv2d-5x5") (bench_conv2d size 5 dtype); ]
           else []);
        ]
      ) sizes
  in
  
  List.concat [
    bench_for_dtype Float32;
    bench_for_dtype Float64;
  ]

(* Run benchmarks *)
let () =
  print_endline "# Nx Benchmarks\n";
  let benchmarks = make_benchmarks () in
  let results = Ubench.run ~warmup:1 ~trials:2 ~min_time:0.001 benchmarks in
  Ubench.print_report results