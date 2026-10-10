(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Rig_metal

include Rig_gpu_support.Make (struct
  module D = Rig_metal

  let class_ = "METAL"
  let present () = Sys.file_exists "/System/Library/Frameworks/Metal.framework"
  let open_ () = Rig_metal.open_ 0
end)

(* A device's ring, by hand *)

type ring

external ring : int -> ring = "rig_metal_test_ring"
external commit : ring -> last:bool -> int = "rig_metal_test_commit"

external complete : ring -> int -> failed:bool -> unit
  = "rig_metal_test_complete"

external word : ring -> int = "rig_metal_test_word"
external times : ring -> int -> int * int = "rig_metal_test_times"
external failure : ring -> string option = "rig_metal_test_failure"
external sleep : ring -> string option = "rig_metal_test_sleep"
external stop : ring -> bool = "rig_metal_test_stop"

(* Fills *)

type arg =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external failing_arg : int -> arg = "rig_metal_test_failing"
external failing_fill : unit -> nativeint = "rig_metal_test_failing_fill"

external dispatch_arg : nativeint -> nativeint -> int -> int -> int -> arg
  = "rig_metal_test_dispatch"

external split_arg : arg -> nativeint -> int -> int -> unit
  = "rig_metal_test_split"

external dispatch_fill : unit -> nativeint = "rig_metal_test_dispatch_fill"

external execute_arg : nativeint -> int -> arg = "rig_metal_test_execute"

external execute_fill : unit -> nativeint = "rig_metal_test_execute_fill"

type fill = { fn : nativeint; arg : arg }

let part f =
  let arg = Rig.Buffer.of_bigarray f.arg in
  {
    Rig.Submission.queue = "COMPUTE:0";
    after = [||];
    work = Fill { fill = f.fn; arg; ring_units = 0; segment_bytes = 0 };
  }

let failing code = { fn = failing_fill (); arg = failing_arg code }

let dispatch ~pipeline ?(offset = 0) args ~groups ~threads =
  let arg =
    dispatch_arg
      (Nativeint.of_int pipeline)
      (Rig_metal.locate args).handle offset groups threads
  in
  { fn = dispatch_fill (); arg }

let split f d k ~times =
  split_arg f.arg (Rig_metal.capability d).split k times

let execute (b : Rig_metal_abi.icb) =
  { fn = execute_fill (); arg = execute_arg b.handle (Array.length b.commands) }

external watching_arg : unit -> arg * nativeint = "rig_metal_test_watching"
external watching_fill : unit -> nativeint = "rig_metal_test_watching_fill"

let watching () =
  let arg, slot = watching_arg () in
  ({ fn = watching_fill (); arg }, slot)

external resize : nativeint -> groups:int -> threads:int -> unit
  = "rig_metal_test_resize"

(* Probes *)

external macos : unit -> bool = "rig_metal_test_macos"

let macos = macos ()

let fixture ~dir f =
  In_channel.with_open_bin
    (Filename.concat dir (f ^ ".metallib"))
    In_channel.input_all

external weak : nativeint -> nativeint = "rig_metal_test_weak"
external alive : nativeint -> bool = "rig_metal_test_alive"
external uptime : unit -> int = "rig_metal_test_uptime"

(* Conformance *)

let binary () =
  (fixture ~dir:"../metal/fixtures" "fill", [ "fill"; "step"; "spin"; "bump"; "copy" ])

let second () = Some (Rig_metal.open_ 0)
let launch_binary () = Some (fixture ~dir:"../metal/fixtures" "launch")
let images = Rig_gpu_support.loader (fun () -> fst (binary ()))

(* [kernel] of [image] on COMPUTE:0 over [groups] threadgroups of [threads]
   threads, with [params] bytes of parameters [store] stores: the buffers' GPU
   addresses themselves, as a support names no slot of the caller's run. *)
let launching image kernel ~params ~groups ~threads store =
  let module Run = Rig.Submission.Run in
  let part =
    {
      Rig.Submission.queue = "COMPUTE:0";
      after = [||];
      work = Launch { image; kernel; params; refs = [||] };
    }
  in
  let block run b =
    Run.groups run b groups 1 1;
    Run.threads run b threads 1 1;
    store run b
  in
  { Rig_gpu_support.part; block }

(* The kernel [copy] has a thread per word and no bound: the grid is the
   words, in threadgroups of the most threads up to 256 that divide them. *)
let copy_words t ~dst ~src =
  let module Run = Rig.Submission.Run in
  let words = Rig.Buffer.length src / 4 in
  let rec threads k = if words mod k = 0 then k else threads (k - 1) in
  let threads = threads (Int.min words 256) in
  let w =
    launching (images t.d) "copy" ~params:16 ~groups:(words / threads) ~threads
      (fun run k ->
        Run.int64 run k 0 (Rig.Buffer.address dst);
        Run.int64 run k 8 (Rig.Buffer.address src))
  in
  (w, src)

(* The kernel [spin] runs [c] steps of a generator on one thread, then writes
   its state at [out], here a word of its own. A step takes about 29 ns on an M1
   Max: one per 10 ns leaves room for faster GPUs. *)
let spin t ~ns =
  let module Run = Rig.Submission.Run in
  let c = Int.min ((ns / 10) + 1) 0xffff_ffff in
  let out = Rig_gpu_support.arguments t.d (String.make 8 '\000') in
  let w =
    launching (images t.d) "spin" ~params:16 ~groups:1 ~threads:1 (fun run k ->
        Run.int64 run k 0 (Rig.Buffer.address out);
        Run.int32 run k 8 c)
  in
  (w, out)
