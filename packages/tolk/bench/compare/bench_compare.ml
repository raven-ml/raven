(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Comparative compile-pipeline timing. For each workload graph (see
   {!Tolk_bench_graphs}) every pipeline stage is timed on an identical graph
   and emitted as JSON, so the counterpart reference driver can be joined
   against it stage by stage.

   The measurement is warm-once then median-and-min of N samples on a
   monotonic clock, matching the reference driver. Stages 2-6 are pure graph
   transforms timed with the render-only CPU renderer (the same renderer the
   parity goldens bless through). Stage 7 (compile) shells out to clang via
   the device renderer's compiler; it uses fewer samples and its own,
   compilable source.

   Two files are written under the output directory:
     tolk.json         one row per (workload, stage) with the timing schema
     tolk.verify.json  per-workload kernel count and first-kernel source, for
                       the same-graph cross-check against the reference. *)

open Tolk
module U = Tolk_uop.Uop
module Graphs = Tolk_bench_graphs.Graphs

let optimize = true
let keep x = ignore (Sys.opaque_identity x)

(* Render-only renderer for stages 2-6: byte-identical to the parity "cpu"
   goldens. *)
let ren = Cstyle.clang_no_abi Gpu_target.X86_64

(* Device renderer + compiler for stage 7: the host clang path the CPU device
   actually uses, so the compiled source is a valid translation unit. *)
let device_ren = Device.renderer (Tolk_cpu.create "CPU:bench_compare")

let clang_compiler =
  match Renderer.compiler device_ren with
  | Some c -> c
  | None -> failwith "CPU device renderer has no compiler"

let kernel_name processed =
  match U.as_kernel_info processed with Some ki -> ki.name | None -> "kernel"

let render_kernel r k =
  let processed = Codegen.full_rewrite_to_sink ~optimize r k in
  Renderer.render r ~name:(kernel_name processed) (Linearizer.linearize processed)

(* Timing *)

let n_stage = 20
let n_compile = 5

let median_min samples =
  let a = Array.copy samples in
  Array.sort Float.compare a;
  let n = Array.length a in
  (a.(n / 2), a.(0))

let time ~n f =
  keep (f ());
  let samples =
    Array.init n (fun _ ->
        let t0 = Thumper_clock.elapsed_ns () in
        let r = f () in
        let t1 = Thumper_clock.elapsed_ns () in
        keep r;
        Int64.to_float (Int64.sub t1 t0) /. 1e6)
  in
  median_min samples

(* Stage rows *)

type row = {
  workload : string;
  size : string;
  stage : string;
  ms_median : float;
  ms_min : float;
  n_kernels : int;
  src_bytes : int;
}

let measure w =
  let workload = Graphs.name w and size = Graphs.size w in
  let sink = Graphs.sink w in
  let kg = Rangeify.get_kernel_graph sink in
  let ks = Graphs.kernels kg in
  let n_kernels = List.length ks in
  let codegen = List.map (Codegen.full_rewrite_to_sink ~optimize ren) ks in
  let programs =
    List.map (fun p -> (kernel_name p, Linearizer.linearize p)) codegen
  in
  let render_srcs =
    List.map (fun (name, prog) -> Renderer.render ren ~name prog) programs
  in
  let src_bytes =
    List.fold_left (fun acc s -> acc + String.length s) 0 render_srcs
  in
  let device_srcs = List.map (render_kernel device_ren) ks in
  let compile_bytes =
    List.fold_left (fun acc s -> acc + String.length s) 0 device_srcs
  in
  let first_kernel_src =
    match render_srcs with s :: _ -> String.trim s | [] -> ""
  in
  let stage stage ~bytes (ms_median, ms_min) =
    { workload; size; stage; ms_median; ms_min; n_kernels; src_bytes = bytes }
  in
  let rows =
    [
      stage "rangeify" ~bytes:src_bytes
        (time ~n:n_stage (fun () -> Rangeify.get_kernel_graph sink));
      stage "schedule" ~bytes:src_bytes
        (time ~n:n_stage (fun () ->
             Schedule.memory_plan_rewrite (Schedule.create_schedule kg) []));
      stage "codegen" ~bytes:src_bytes
        (time ~n:n_stage (fun () ->
             List.iter
               (fun k -> keep (Codegen.full_rewrite_to_sink ~optimize ren k))
               ks));
      stage "linearize" ~bytes:src_bytes
        (time ~n:n_stage (fun () ->
             List.iter (fun p -> keep (Linearizer.linearize p)) codegen));
      stage "render" ~bytes:src_bytes
        (time ~n:n_stage (fun () ->
             List.iter
               (fun (name, prog) -> keep (Renderer.render ren ~name prog))
               programs));
      stage "compile" ~bytes:compile_bytes
        (time ~n:n_compile (fun () ->
             List.iter
               (fun s -> keep (Compiler.compile_cached clang_compiler s))
               device_srcs));
    ]
  in
  (rows, (workload, n_kernels, first_kernel_src))

(* JSON emission *)

let escape s =
  let buf = Buffer.create (String.length s + 16) in
  String.iter
    (fun c ->
      match c with
      | '"' -> Buffer.add_string buf "\\\""
      | '\\' -> Buffer.add_string buf "\\\\"
      | '\n' -> Buffer.add_string buf "\\n"
      | '\r' -> Buffer.add_string buf "\\r"
      | '\t' -> Buffer.add_string buf "\\t"
      | c when Char.code c < 0x20 ->
          Buffer.add_string buf (Printf.sprintf "\\u%04x" (Char.code c))
      | c -> Buffer.add_char buf c)
    s;
  Buffer.contents buf

let row_json r =
  Printf.sprintf
    {|{"workload":"%s","size":"%s","stage":"%s","ms_median":%.6f,"ms_min":%.6f,"n_kernels":%d,"src_bytes":%d}|}
    (escape r.workload) (escape r.size) (escape r.stage) r.ms_median r.ms_min
    r.n_kernels r.src_bytes

let write_file path contents =
  let oc = open_out path in
  output_string oc contents;
  close_out oc

let () =
  let out_dir = if Array.length Sys.argv > 1 then Sys.argv.(1) else "." in
  let results = List.map measure Graphs.all in
  let rows = List.concat_map fst results in
  let timings =
    "[\n  " ^ String.concat ",\n  " (List.map row_json rows) ^ "\n]\n"
  in
  write_file (Filename.concat out_dir "tolk.json") timings;
  let verify_entry (workload, n_kernels, src) =
    Printf.sprintf {|  "%s": {"n_kernels":%d,"first_kernel_src":"%s"}|}
      (escape workload) n_kernels (escape src)
  in
  let verify =
    "{\n"
    ^ String.concat ",\n" (List.map (fun (_, v) -> verify_entry v) results)
    ^ "\n}\n"
  in
  write_file (Filename.concat out_dir "tolk.verify.json") verify;
  Printf.printf "wrote %d rows for %d workloads to %s/tolk.json\n"
    (List.length rows) (List.length results) out_dir
