(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Indicative runtime-throughput benchmarks. Each workload is compiled once
   through the capture-and-replay JIT (or, for the copy case, allocated once)
   and then replayed K times; the timed region is execution only, with compile
   and scheduling held outside the loop.

   The four workloads and their throughput units:
     matmul       a @ b, float32          GFLOP/s   (2*M*N*K flops)
     elementwise  a + b * c over a buffer  GB/s      (3 reads + 1 write)
     reduce       sum over a buffer        GB/s      (1 read)
     copy         host -> device buffer    GB/s      (bytes uploaded)

   Run the Python companion against an explicit tinygrad checkout for a
   like-for-like CPU comparison. Repeat alternating runs on a quiet host.
   The backend defaults to CPU unless [DEV] selects another device; its
   name is recorded in every row. Allocation deltas cover the calling domain
   and include timing-loop overhead; storage bytes count live logical buffers,
   not native allocator reservations or backend object handles. *)

module Dtype = Tolk_uop.Dtype
module Dev = Tolk.Device
module El = Tolk_frontend.Elementwise
module Rd = Tolk_frontend.Reduce
module Op = Tolk_frontend.Op
module Run = Tolk_frontend.Run
module Jit = Tolk_frontend.Jit

let keep x = ignore (Sys.opaque_identity x)
let now () = Thumper_clock.elapsed_ns ()

let f32_bytes = 4

(* A zero-filled float32 tensor of [shape]. Values are immaterial to timing. *)
let input shape =
  let n = List.fold_left ( * ) 1 shape in
  Run.of_bytes ~dtype:Dtype.float32 ~shape (Bytes.make (n * f32_bytes) '\000')

(* Adaptive replay timing. One estimate call sizes K so the timed run spans
   roughly [target_s]; K is clamped so tiny and huge workloads both behave.
   Returns median and minimum per-replay nanoseconds and the K used. *)
let target_s = 1.5
let min_k = 5
let max_k = 5000

type timing = {
  median_ns : float;
  min_ns : float;
  k : int;
  allocated_words : float;
  promoted_words : float;
  major_words : float;
  storage_bytes_before : int;
  storage_bytes_after : int;
}

let time_replay call =
  let est =
    let t0 = now () in
    call ();
    Int64.to_float (Int64.sub (now ()) t0) /. 1e9
  in
  let k =
    int_of_float (Float.max 1. (target_s /. Float.max est 1e-9))
    |> max min_k |> min max_k
  in
  let samples = Array.make k 0. in
  let storage_bytes_before = Tolk.Helpers.Global_counters.mem_used () in
  let minor, promoted, major = Gc.counters () in
  for i = 0 to k - 1 do
    let t0 = now () in
    call ();
    samples.(i) <- Int64.to_float (Int64.sub (now ()) t0)
  done;
  let minor_after, promoted_after, major_after = Gc.counters () in
  let storage_bytes_after = Tolk.Helpers.Global_counters.mem_used () in
  Array.sort Float.compare samples;
  let per_call words = words /. float_of_int k in
  {
    median_ns = samples.(k / 2); min_ns = samples.(0); k;
    allocated_words = per_call
        ((minor_after -. minor) +. (major_after -. major) -. (promoted_after -. promoted));
    promoted_words = per_call (promoted_after -. promoted);
    major_words = per_call (major_after -. major);
    storage_bytes_before; storage_bytes_after;
  }

(* A replayable compute workload: build the graph from realized inputs, warm
   and capture through the JIT, then time execution-only replays. *)
let time_compute ~build inputs =
  let jit = Jit.create ~outputs:(fun tensor -> [tensor]) (fun ins ~vars:_ -> Run.realize (build ins)) in
  Array.iter (fun t -> keep (Run.realize t)) inputs;
  let dev = Run.device () in
  let call () =
    keep (Jit.call jit inputs);
    Dev.synchronize dev
  in
  call ();
  call ();
  time_replay call

(* Host -> device upload: allocate the buffer and host bytes once, then time
   the copyin primitive. *)
let time_copy n =
  let dev = Run.device () in
  let buf = Dev.create_buffer ~size:n ~dtype:Dtype.float32 dev in
  Dev.Buffer.ensure_allocated buf;
  let host = Bytes.make (n * f32_bytes) '\000' in
  let call () =
    Dev.Buffer.copyin buf host;
    Dev.synchronize dev
  in
  call ();
  time_replay call

(* Rows *)

type row = {
  bench : string;
  size : string;
  unit_ : string;
  amount : float;  (* flops or bytes moved per replay *)
  timing : timing;
}

(* Throughput is [amount /. ns]: one flop per ns is a GFLOP/s, one byte per ns
   is a GB/s, so both units share the same conversion. *)
let per_ns amount ns = amount /. ns

let matmul_row n =
  let a = input [ n; n ] and b = input [ n; n ] in
  let timing =
    time_compute ~build:(fun ins -> Op.matmul ins.(0) ins.(1)) [| a; b |]
  in
  let flops = 2.0 *. float_of_int n *. float_of_int n *. float_of_int n in
  {
    bench = "matmul";
    size = string_of_int n;
    unit_ = "GFLOP/s";
    amount = flops;
    timing;
  }

let elementwise_row n =
  let a = input [ n ] and b = input [ n ] and c = input [ n ] in
  let timing =
    time_compute
      ~build:(fun ins -> El.add ins.(0) (El.mul ins.(1) ins.(2)))
      [| a; b; c |]
  in
  let bytes = 4.0 *. float_of_int n *. float_of_int f32_bytes in
  {
    bench = "elementwise";
    size = "16M";
    unit_ = "GB/s";
    amount = bytes;
    timing;
  }

let reduce_row n =
  let x = input [ n ] in
  let timing =
    time_compute ~build:(fun ins -> Rd.sum ins.(0)) [| x |]
  in
  let bytes = float_of_int n *. float_of_int f32_bytes in
  {
    bench = "reduce";
    size = "16M";
    unit_ = "GB/s";
    amount = bytes;
    timing;
  }

let copy_row n =
  let timing = time_copy n in
  let bytes = float_of_int n *. float_of_int f32_bytes in
  {
    bench = "copy";
    size = "16M";
    unit_ = "GB/s";
    amount = bytes;
    timing;
  }

(* Output *)

let write_file path contents =
  let oc = open_out path in
  output_string oc contents;
  close_out oc

let row_json backend r =
  Printf.sprintf
    {|{"bench":"%s","size":"%s","backend":"%s","unit":"%s","median":%.6f,"peak":%.6f,"k":%d,"allocated_words_per_replay":%.0f,"promoted_words_per_replay":%.0f,"major_words_per_replay":%.0f,"storage_bytes_before":%d,"storage_bytes_after":%d}|}
    r.bench r.size backend r.unit_ (per_ns r.amount r.timing.median_ns)
    (per_ns r.amount r.timing.min_ns) r.timing.k
    r.timing.allocated_words r.timing.promoted_words r.timing.major_words
    r.timing.storage_bytes_before r.timing.storage_bytes_after

let print_table backend rows =
  Printf.printf "\ntolk runtime throughput (backend %s)\n" backend;
  Printf.printf "%-12s  %-6s  %10s  %10s  %8s  %6s\n" "bench" "size" "median"
    "peak" "unit" "k";
  List.iter
    (fun r ->
      Printf.printf "%-12s  %-6s  %10.2f  %10.2f  %8s  %6d\n" r.bench r.size
        (per_ns r.amount r.timing.median_ns)
        (per_ns r.amount r.timing.min_ns)
        r.unit_ r.timing.k)
    rows

let () =
  if Sys.getenv_opt "DEV" = None then Unix.putenv "DEV" "CPU";
  let out_dir = if Array.length Sys.argv > 1 then Sys.argv.(1) else "." in
  let rows =
    [
      matmul_row 512;
      matmul_row 1024;
      elementwise_row (16 * 1024 * 1024);
      reduce_row (16 * 1024 * 1024);
      copy_row (16 * 1024 * 1024);
    ]
  in
  let backend = Dev.name (Run.device ()) in
  print_table backend rows;
  let json =
    "[\n  "
    ^ String.concat ",\n  " (List.map (row_json backend) rows)
    ^ "\n]\n"
  in
  write_file (Filename.concat out_dir "tolk_runtime.json") json;
  Printf.printf "\nwrote %d rows to %s/tolk_runtime.json\n" (List.length rows)
    out_dir
