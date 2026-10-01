(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk
module B = Nx_device.Buffer
module A = Nx_amd_device

(* The GPU as the compiler encodes its packets. *)
let gpu a =
  let p = A.props a in
  let counting (c : A.counting) =
    {
      Ops_amd.counters =
        List.map
          (fun (ct : A.counter) ->
            {
              Ops_amd.block = ct.block;
              event = ct.event;
              register = ct.register;
              instances = ct.instances;
              engines = ct.engines;
              arrays = ct.arrays;
              wgps = ct.wgps;
              offset = ct.offset;
            })
          c.counters;
      size = c.size;
      wgp_active = c.wgp_active;
    }
  in
  let profiling (pr : A.profiling) =
    {
      Ops_amd.slots = pr.slots;
      counting = Option.map counting pr.counting;
      tracing =
        Option.map
          (fun (t : A.tracing) ->
            { Ops_amd.window = t.window; engines = t.engines })
          pr.tracing;
    }
  in
  {
    Ops_amd.target = p.target;
    gc = p.gc;
    sdma = p.sdma;
    xccs = p.xccs;
    shader_engines = p.shader_engines;
    compute_units = p.compute_units;
    scratch_slots_per_cu = p.scratch_slots_per_cu;
    aql = A.aql a;
    compute_ring = B.nbytes (A.compute a).ring;
    copy_rings = List.map (fun (q : A.queue) -> B.nbytes q.ring) (A.sdma a);
    profiling = Option.map profiling (A.profiling a);
  }

let queue a = function
  | "COMPUTE:0" -> A.compute a
  | q -> (
      match String.split_on_char ':' q with
      | [ "COPY"; i ] -> List.nth (A.sdma a) (int_of_string i)
      | _ -> invalid_arg (q ^ " is no AMD queue"))

(* A program's code object, loaded on the device, which also grows the device's
   scratch memory for its kernel: the AQL queue's descriptor holds it, as
   tinygrad's AQL queue grows it when it encodes a dispatch. *)
let program d a ~binary ~name =
  match Nx_device.Program.load d ~binary ~name with
  | Error why -> failwith why
  | Ok p ->
      let k = Option.get (A.kernel p) in
      ignore (A.scratch a k.private_segment);
      k.code

(* The profile request work is encoded for: the counters it counts and whether
   it traces. A batch writes the device's profiling of its request, whose trace
   buffers every request shares, so it links and runs only while its request is
   the profile's. *)
let request () = (Nx_device.Profile.counters (), Nx_device.Profile.traced ())

let pp_request = function
  | [], false -> "no counters or traces"
  | [], true -> "traces"
  | counters, traced ->
      Printf.sprintf "counters [%s]%s"
        (String.concat "; " counters)
        (if traced then " and traces" else "")

let check fn encoded =
  let asked = request () in
  if asked <> encoded then
    invalid_arg
      (Printf.sprintf
         "Tolk_engine.%s: the batch was encoded for %s, and the profile asks \
          for %s"
         fn (pp_request encoded) (pp_request asked))

(* The device's profiling, for a batch linked under the request it was encoded
   for. *)
let profiled a encoded =
  check "link" encoded;
  Option.get (A.profiling a)

let counted a encoded = Option.get (profiled a encoded).counting
let traced a encoded = Option.get (profiled a encoded).tracing

(* The storage of the placeholders AMD's commands name: those of the device
   [name]. *)
let placeholder name d a encoded u =
  let on_device =
    match Ops.device u with
    | Some (Single n) | Some (Multi [ n ]) -> n = name
    | _ -> false
  in
  if not on_device then None
  else
    Option.map
      (function
        | Ops_amd.Ring q -> (queue a q).ring
        | Write_ptr q -> (queue a q).write_ptr
        | Put q -> (queue a q).put
        | Doorbell q -> (queue a q).doorbell
        | Program { binary; name } -> program d a ~binary ~name
        | Scratch n -> A.scratch a n
        | Log -> (profiled a encoded).log
        | Samples -> (counted a encoded).samples
        | Traces -> (traced a encoded).traces
        | Trace_ends -> (traced a encoded).ends)
      (Ops_amd.storage u)

(* The queues address the memory nx.device says the device reaches: other memory
   is copied through the host's. *)
let reaches d devices n =
  match List.assoc_opt n devices with
  | Some d' -> Nx_device.reaches d d'
  | None -> false

(* Each submission flushes the host data path before its host program rings a
   doorbell, so that its copy queues, which have no flush of their own, read
   what the host wrote to mapped memory. *)
let queues ~host devices name d =
  Option.map
    (fun a ->
      let encoded = request () in
      ( Ops_amd.queues ~host:(Lazy.force host) ~reaches:(reaches d devices)
          (gpu a),
        placeholder name d a encoded,
        fun () ->
          check "run" encoded;
          A.flush_hdp a ))
    (A.of_device d)
