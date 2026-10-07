(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Thread traces: the program's refusals, the write pointer's units, and a GFX9
   trace of one wave. *)

open Windtrap
open Device_amd_abi

let gpu target =
  {
    Gpu.target;
    gc = target;
    sdma = (6, 0, 0);
    xccs = 1;
    shader_engines = 4;
    compute_units = 32;
    scratch_slots = 32;
  }

let recording =
  group "recording"
    [
      cases ~name:string_of_int "a size of no whole pages is refused"
        [ 0; 4097; -4096 ] (fun size ->
          raises_match (Exn.invalid_arg ~substring:"Thread_trace.start")
            (fun () -> Thread_trace.start (gpu (11, 0, 0)) ~size (fun _ -> 0)));
      test "a GFX 11.0 write pointer counts from address 0" (fun () ->
          equal int 0x100
            (Thread_trace.length
               (gpu (11, 0, 0))
               ~buffer:0x10_0000
               ((0x10_0000 + 0x100) / 32)));
      test "a GFX12 write pointer counts from the buffer" (fun () ->
          equal int 0x100
            (Thread_trace.length
               (gpu (12, 0, 1))
               ~buffer:0x10_0000 (0x100 / 32)));
    ]

(* A GFX9 trace: a wave start of compute unit 5, SIMD 2 and slot 7, a packet of
   delta 5 (20 cycles), the wave's end, and zeros. *)
let gfx9_trace =
  let b = Bytes.make 16 '\000' in
  let wave kind = kind lor (5 lsl 6) lor (7 lsl 10) lor (2 lsl 14) in
  Bytes.set_int32_le b 0 (Int32.of_int (wave 3));
  Bytes.set_uint16_le b 4 (5 lsl 4);
  Bytes.set_uint16_le b 6 (wave 6);
  Bytes.to_string b

let decoding =
  group "decoding"
    [
      test "a GFX9 trace's wave" (fun () ->
          equal
            (list (list int))
            [ [ 5; 2; 7; 0; 20 ] ]
            (List.map
               (fun (w : Thread_trace.wave) ->
                 [ w.cu; w.simd; w.slot; w.start; w.stop ])
               (Thread_trace.waves (gpu (9, 4, 2)) gfx9_trace)));
      test "a trace without markers has no clock" (fun () ->
          is_none (Thread_trace.clock (gpu (9, 4, 2)) gfx9_trace));
      test "an empty trace has no waves" (fun () ->
          equal int 0 (List.length (Thread_trace.waves (gpu (11, 0, 0)) "")));
    ]

let () = exit (run "device_amd_abi.thread_trace" [ recording; decoding ])
