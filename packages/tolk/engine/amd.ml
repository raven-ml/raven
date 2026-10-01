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

(* The profiling of a batch that profiles, which the profile being taken asks
   for since the batch was compiled. *)
let mismatch () =
  invalid_arg
    "Tolk_engine.link: the batch profiles its kernels' runs as the profile \
     being taken does not"

let profiled a = match A.profiling a with Some p -> p | None -> mismatch ()

let samples a =
  match (profiled a).counting with Some c -> c.samples | None -> mismatch ()

let traced a =
  match (profiled a).tracing with Some t -> t | None -> mismatch ()

(* The storage of the placeholders AMD's commands name: those of the device
   [name]. *)
let placeholder name d a u =
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
        | Log -> (profiled a).log
        | Samples -> samples a
        | Traces -> (traced a).traces
        | Trace_ends -> (traced a).ends)
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
      ( Ops_amd.queues ~host:(Lazy.force host) ~reaches:(reaches d devices)
          (gpu a),
        placeholder name d a,
        fun () -> A.flush_hdp a ))
    (A.of_device d)
