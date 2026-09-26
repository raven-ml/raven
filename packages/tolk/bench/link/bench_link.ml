(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Metal linking microbenchmark: real compiled chains of 1024-element adds.
   Compilation, execution, validation and collection are outside link_ns.
   Materialization is reported separately because some backend allocations
   remain lazy after linking. Storage-root counts are NOT native allocation
   counts: LRU reuse and native ICB/pipeline objects are not exposed here. *)

open Tolk
open Tolk_uop
module U = Uop
module B = Device.Buffer

let calls = ref 64
let samples = ref 11
let elements = 1024
let now = Thumper_clock.elapsed_ns
let elapsed start = Int64.to_float (Int64.sub (now ()) start)
let keep value = ignore (Sys.opaque_identity value)

let param slot =
  U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int elements) ()

let kernel device =
  let dst = param 0 and src = param 1 in
  let r = U.range ~size:(U.const_int elements) ~axis:0 ~kind:Axis_type.Global () in
  let load = U.load ~src:(U.index ~ptr:src ~idxs:[r] ()) () in
  let value = U.alu_binary ~op:Ops.Add ~lhs:load ~rhs:(U.const_int 1) in
  let store = U.store ~dst:(U.index ~ptr:dst ~idxs:[r] ()) ~value () in
  let kernel_info = U.{name = "link_chain"; applied_opts = [];
    opts_to_apply = None; estimates = None; beam = 0} in
  Codegen.to_program ~optimize:false (Device.renderer device)
    (U.sink ~kernel_info [U.end_ ~value:store ~ranges:[r]])

let template device program ~eager =
  let args = Array.init 2 (fun slot ->
      let p = U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int elements)
          ~device:(U.Single (Device.name device)) () in
      if eager then U.with_tag "lt_input" p else p) in
  U.linear (List.init !calls (fun i ->
      U.call ~body:program ~args:[args.(i mod 2); args.(1 - (i mod 2))]
        ~info:{grad_fxn = None; name = Some (U.Label (string_of_int i));
          precompile = false; precompile_backward = false; aux = None; dtype = Dtype.void}))

let roots linked =
  let seen = Hashtbl.create 16 in
  U.toposort linked |> List.filter_map (fun node ->
      match U.as_buffer node with
      | Some {buffer = {buffer = Some [buffer]; _}; _} ->
          let root = B.base buffer in
          if Hashtbl.mem seen (B.id root) then None
          else (Hashtbl.add seen (B.id root) (); Some root)
      | _ -> None)

type measurement = {
  link_ns : float; materialize_ns : float;
  minor_words : float; promoted_words : float; major_words : float;
  minor_collections : float; major_collections : float;
  storage_roots : float; storage_bytes : float;
}

let measure ~ctx ~eager compiled inputs =
  Gc.full_major ();
  let before = Gc.quick_stat () in
  (* quick_stat is sampled at collection boundaries in OCaml 5. Its word
     deltas can be zero for a complete link that fits in the minor heap. *)
  let minor_before, promoted_before, major_before = Gc.counters () in
  let start = now () in
  let linked = Realize.link_linear ~ctx ~allow_cache:eager compiled in
  let link_ns = elapsed start in
  let minor_after, promoted_after, major_after = Gc.counters () in
  let after = Gc.quick_stat () in
  let backing = roots linked in
  let start = now () in
  List.iter B.ensure_allocated backing;
  let materialize_ns = elapsed start in
  let backing = List.filter (fun b ->
      not (Array.exists (fun input -> B.id input = B.id b) inputs)) backing in
  linked, {
    link_ns; materialize_ns;
    minor_words = minor_after -. minor_before;
    promoted_words = promoted_after -. promoted_before;
    major_words = major_after -. major_before;
    minor_collections = float_of_int (after.minor_collections - before.minor_collections);
    major_collections = float_of_int (after.major_collections - before.major_collections);
    storage_roots = float_of_int (List.length backing);
    storage_bytes = float_of_int (List.fold_left (fun total b -> total + B.nbytes b) 0 backing);
  }

let median values =
  let values = Array.copy values in
  Array.sort Float.compare values;
  values.(Array.length values / 2)

let run device program inputs ~eager =
  let input_uops = Array.map U.from_buffer inputs in
  let ctx = Realize.exec_context ~input_uops ~update_stats:false () in
  let to_program device = Codegen.to_program ~beam_device:device (Device.renderer device) in
  let compiled = Realize.compile_linear ~device ~profile:false ~to_program
      (template device program ~eager) in
  let tags = U.toposort compiled |> List.filter (fun u -> U.node_tag u = Some "lt_input") in
  if eager && tags = [] then failwith "benchmark did not preserve lt_input bindings";
  let zero = Bytes.make (elements * 4) '\000' in
  let replay linked =
    Array.iter (fun b -> B.copyin b zero) inputs;
    Realize.run_linear ~device ~to_program ~jit:true ~wait:true
      ~update_stats:false ~input_uops linked;
    Device.synchronize device;
    Array.iteri (fun slot b ->
        let expected = if slot = (!calls - 1) mod 2 then !calls else !calls - 1 in
        let bytes = B.as_bytes b in
        for i = 0 to elements - 1 do
          if Bytes.get_int32_le bytes (4 * i) <> Int32.of_int expected then
            failwith "linked chain produced an incorrect result"
        done) inputs in
  (* The retained control remains live across every subsequent independent link.
     Replaying it after the new link exercises the ownership being measured. *)
  let retained = if eager then None else
      Some (Realize.link_linear ~ctx ~allow_cache:false compiled) in
  Option.iter replay retained;
  let sample () =
    let linked, measurement = measure ~ctx ~eager compiled inputs in
    replay linked;
    Option.iter replay retained;
    keep linked;
    measurement in
  for i = 1 to 3 do ignore i; keep (sample ()) done;
  let results = Array.init !samples (fun _ -> sample ()) in
  let metric f = median (Array.map f results) in
  Printf.printf
    "{\"backend\":%S,\"mode\":%S,\"calls\":%d,\"elements\":%d,\"samples\":%d,\"hcq_cache_thresh\":%d,\"ocaml\":%S,\"ocamlrunparam\":%S,\"link_ns\":%.0f,\"materialize_ns\":%.0f,\"minor_words\":%.0f,\"promoted_words\":%.0f,\"major_words\":%.0f,\"minor_collections\":%.0f,\"major_collections\":%.0f,\"linked_storage_roots\":%.0f,\"linked_storage_bytes\":%.0f}\n%!"
    (Device.name device) (if eager then "eager_lt_input" else "retained_independent")
    !calls elements !samples (Helpers.Context_var.get Helpers.hcq_cache_thresh)
    Sys.ocaml_version (Option.value (Sys.getenv_opt "OCAMLRUNPARAM") ~default:"")
    (metric (fun r -> r.link_ns)) (metric (fun r -> r.materialize_ns))
    (metric (fun r -> r.minor_words)) (metric (fun r -> r.promoted_words))
    (metric (fun r -> r.major_words)) (metric (fun r -> r.minor_collections))
    (metric (fun r -> r.major_collections)) (metric (fun r -> r.storage_roots))
    (metric (fun r -> r.storage_bytes))

let () =
  Arg.parse ["--calls", Arg.Set_int calls, "Number of top-level calls (64 or 128)";
    "--samples", Arg.Set_int samples, "Measured links per mode (default 11)"]
    (fun arg -> invalid_arg ("unexpected argument: " ^ arg)) "bench_link";
  if !calls < Helpers.Context_var.get Helpers.hcq_cache_thresh || !calls < 2 || !samples < 1
  then invalid_arg "calls must reach HCQ_CACHE_THRESH, and samples must be positive";
  let device = Tolk_metal.create "METAL:link-bench" in
  let inputs = Array.init 2 (fun _ ->
      let b = Device.create_buffer ~size:elements ~dtype:Dtype.int32 device in
      B.ensure_allocated b; b) in
  let program = kernel device in
  run device program inputs ~eager:true;
  run device program inputs ~eager:false
