(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The formatting helpers behind the DEBUG=2 execution line, against the strings
   the reference produces for the same arguments, and the allocation counter. *)

open Windtrap
open Tolk
module G = Helpers.Global_counters

let () = Device.register "CPU" Tolk_cpu.create
let device = Device.get "CPU:helpers-test"

let formatting =
  group "formatting"
    [
      test "time_to_str picks the unit above ten of the next" (fun () ->
          equal string "   12.50s " (Helpers.time_to_str 12.5);
          equal string "   50.00ms" (Helpers.time_to_str 0.05);
          equal string "     5.00us" (Helpers.time_to_str ~w:9 5e-6);
          equal string "10000.00us" (Helpers.time_to_str 0.01);
          equal string "10000.00ms" (Helpers.time_to_str 10.0);
          equal string "    12.30ms" (Helpers.time_to_str ~w:9 0.0123));
      test "size_to_str" (fun () ->
          equal string "5 B" (Helpers.size_to_str 5);
          equal string "1023 B" (Helpers.size_to_str 1023);
          equal string "1.50 KB" (Helpers.size_to_str 1536);
          equal string "1.00 MB" (Helpers.size_to_str (1 lsl 20));
          equal string "1.50 GB"
            (Helpers.size_to_str ((1 lsl 30) + (1 lsl 29))));
      test "colored" (fun () ->
          equal string "\027[31mx\027[0m" (Helpers.colored "x" (Some "red"));
          equal string "\027[91mx\027[0m" (Helpers.colored "x" (Some "RED"));
          equal string "\027[46mx\027[0m"
            (Helpers.colored ~background:true "x" (Some "cyan"));
          equal string "x" (Helpers.colored "x" None));
      test "ansilen ignores escape sequences" (fun () ->
          equal int 10
            (Helpers.ansilen (Helpers.colored "batched 26" (Some "cyan")));
          equal int 2 (Helpers.ansilen "a\027[Kb"));
    ]

let counters =
  group "Global_counters"
    [
      test "mem_used follows allocation and release" (fun () ->
          let before = !G.mem_used in
          let buf =
            Device.create_buffer ~size:1000 ~dtype:Tolk_uop.Dtype.float32 device
          in
          equal int before !G.mem_used;
          Device.Buffer.ensure_allocated buf;
          equal int (before + 4000) !G.mem_used;
          Device.Buffer.deallocate buf;
          equal int before !G.mem_used);
      test "reset leaves mem_used alone" (fun () ->
          let buf =
            Device.create_buffer ~size:10 ~dtype:Tolk_uop.Dtype.float32 device
          in
          Device.Buffer.ensure_allocated buf;
          let used = !G.mem_used in
          G.kernel_count := 3;
          G.reset ();
          equal int 0 !G.kernel_count;
          equal int used !G.mem_used;
          Device.Buffer.deallocate buf);
    ]

let () = run __FILE__ [ formatting; counters ]
