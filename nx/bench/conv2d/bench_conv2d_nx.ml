(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Benchmarks for Nx.correlate and Nx.extract_patches *)

let backend_name = "Nx"

(* correlate: (leading..., spatial...) with kernel of rank K *)
let correlate_configs =
  [
    (* (label, input_shape, kernel_shape) *)
    ("1D 1k", [| 100 |], [| 5 |]);
    ("1D 10k batched", [| 16; 10000 |], [| 5 |]);
    ("2D 64x64", [| 64; 64 |], [| 3; 3 |]);
    ("2D 256x256", [| 256; 256 |], [| 3; 3 |]);
    ("2D batch 8x64x64", [| 8; 64; 64 |], [| 3; 3 |]);
    ("2D batch 8x256x256", [| 8; 256; 256 |], [| 3; 3 |]);
  ]

let extract_patches_configs =
  [
    (* (label, input_shape, kernel_size, stride) *)
    ("2D 64x64 k3 s1", [| 1; 1; 64; 64 |], [| 3; 3 |], [| 1; 1 |]);
    ("2D 64x64 k3 s2", [| 1; 1; 64; 64 |], [| 3; 3 |], [| 2; 2 |]);
    ("2D 256x256 k3 s1", [| 1; 1; 256; 256 |], [| 3; 3 |], [| 1; 1 |]);
    ("2D 8x3x64x64 k3 s1", [| 8; 3; 64; 64 |], [| 3; 3 |], [| 1; 1 |]);
  ]

let build_benchmarks () =
  let benchmarks = ref [] in
  List.iter
    (fun (label, input_shape, kernel_shape) ->
      let x = Nx.rand Nx.Float32 input_shape in
      let k = Nx.rand Nx.Float32 kernel_shape in
      let name = Printf.sprintf "correlate %s f32 (%s)" label backend_name in
      benchmarks :=
        Ubench.bench name (fun () -> ignore (Nx.correlate x k)) :: !benchmarks)
    correlate_configs;
  List.iter
    (fun (label, input_shape, kernel_size, stride) ->
      let x = Nx.rand Nx.Float32 input_shape in
      let k = Array.length kernel_size in
      let dilation = Array.make k 1 in
      let padding = Array.make k (0, 0) in
      let name =
        Printf.sprintf "extract_patches %s f32 (%s)" label backend_name
      in
      benchmarks :=
        Ubench.bench name (fun () ->
            ignore
              (Nx.extract_patches ~kernel_size ~stride ~dilation ~padding x))
        :: !benchmarks)
    extract_patches_configs;
  List.rev !benchmarks

let default_config () =
  let open Ubench.Config in
  default |> time_limit 1.0 |> warmup 1 |> min_measurements 5
  |> geometric_scale 1.3 |> gc_stabilization false |> build

let () =
  let benchmarks = build_benchmarks () in
  let config = default_config () in
  ignore (Ubench.run ~config benchmarks)
