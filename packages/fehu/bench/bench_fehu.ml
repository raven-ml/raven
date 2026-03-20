(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Gae = Fehu.Gae
module Buffer = Fehu.Buffer

let gamma = 0.99
let lambda = 0.95

let make_arrays n =
  let rewards = Array.init n (fun i -> Float.of_int (i mod 5)) in
  let values = Array.init n (fun i -> Float.of_int (i mod 10) *. 0.1) in
  let terminated = Array.init n (fun i -> i mod 50 = 49) in
  let truncated = Array.init n (fun i -> i mod 100 = 99) in
  let next_values = Array.init n (fun i -> Float.of_int ((i + 1) mod 10) *. 0.1) in
  (rewards, values, terminated, truncated, next_values)

let gae_benchmarks () =
  let sizes = [ ("256", 256); ("1024", 1024); ("4096", 4096) ] in
  let benches = ref [] in
  List.iter
    (fun (label, n) ->
      let rewards, values, terminated, truncated, next_values = make_arrays n in
      benches :=
        Thumper.bench
          (Printf.sprintf "compute n=%s" label)
          (fun () ->
            Thumper.consume
              (Gae.compute ~rewards ~values ~terminated ~truncated ~next_values
                 ~gamma ~lambda))
        :: !benches)
    sizes;
  List.rev !benches

let gae_from_values_benchmarks () =
  let sizes = [ ("256", 256); ("1024", 1024); ("4096", 4096) ] in
  let benches = ref [] in
  List.iter
    (fun (label, n) ->
      let rewards, values, terminated, truncated, _ = make_arrays n in
      benches :=
        Thumper.bench
          (Printf.sprintf "compute_from_values n=%s" label)
          (fun () ->
            Thumper.consume
              (Gae.compute_from_values ~rewards ~values ~terminated ~truncated
                 ~last_value:0.0 ~gamma ~lambda))
        :: !benches)
    sizes;
  List.rev !benches

let returns_benchmarks () =
  let sizes = [ ("256", 256); ("1024", 1024); ("4096", 4096) ] in
  let benches = ref [] in
  List.iter
    (fun (label, n) ->
      let rewards, _, terminated, truncated, _ = make_arrays n in
      benches :=
        Thumper.bench
          (Printf.sprintf "returns n=%s" label)
          (fun () ->
            Thumper.consume (Gae.returns ~rewards ~terminated ~truncated ~gamma))
        :: !benches)
    sizes;
  List.rev !benches

let normalize_benchmarks () =
  let sizes = [ ("256", 256); ("1024", 1024); ("4096", 4096) ] in
  let benches = ref [] in
  List.iter
    (fun (label, n) ->
      let arr = Array.init n (fun i -> Float.of_int i *. 0.01) in
      benches :=
        Thumper.bench
          (Printf.sprintf "normalize n=%s" label)
          (fun () -> Thumper.consume (Gae.normalize arr))
        :: !benches)
    sizes;
  List.rev !benches

let fill_buffer capacity =
  let buf : (float, float) Buffer.t = Buffer.create ~capacity in
  for i = 0 to capacity - 1 do
    Buffer.add buf
      {
        Buffer.observation = Float.of_int i;
        action = Float.of_int (i mod 4);
        reward = Float.of_int (i mod 10) *. 0.1;
        next_observation = Float.of_int (i + 1);
        terminated = i mod 50 = 49;
        truncated = i mod 100 = 99;
      }
  done;
  buf

let buffer_add_benchmarks () =
  let capacity = 10000 in
  let buf = fill_buffer capacity in
  let tr =
    {
      Buffer.observation = 0.0;
      action = 0.0;
      reward = 1.0;
      next_observation = 1.0;
      terminated = false;
      truncated = false;
    }
  in
  [
    Thumper.bench "add (full buffer, cap=10000)" (fun () ->
        Buffer.add buf tr);
  ]

let buffer_create_benchmarks () =
  let sizes = [ ("100", 100); ("1000", 1000); ("10000", 10000) ] in
  List.map
    (fun (label, n) ->
      Thumper.bench
        (Printf.sprintf "create+fill cap=%s" label)
        (fun () -> Thumper.consume (fill_buffer n)))
    sizes

let build_benchmarks () =
  [
    Thumper.group "GAE" (gae_benchmarks ());
    Thumper.group "GAE from values" (gae_from_values_benchmarks ());
    Thumper.group "Returns" (returns_benchmarks ());
    Thumper.group "Normalize" (normalize_benchmarks ());
    Thumper.group "Buffer add" (buffer_add_benchmarks ());
    Thumper.group "Buffer create" (buffer_create_benchmarks ());
  ]

let () =
  let benchmarks = build_benchmarks () in
  Thumper.run "fehu" benchmarks
