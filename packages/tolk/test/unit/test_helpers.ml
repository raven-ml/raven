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
          Helpers.Context_var.with_context [ B (Helpers.no_color, 0) ] (fun () ->
            equal string "\027[31mx\027[0m" (Helpers.colored "x" (Some "red"));
            equal string "\027[91mx\027[0m" (Helpers.colored "x" (Some "RED"));
            equal string "\027[46mx\027[0m"
              (Helpers.colored ~background:true "x" (Some "cyan"));
            equal string "x" (Helpers.colored "x" None)));
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

module H = Tolk.Helpers
module Target = Tolk_uop.Target

let with_dev value f =
  H.Context_var.with_context
    [ B (H.dev, List.map Target.of_string (String.split_on_char ';' value)) ] f

let with_env key value f =
  let old = Option.value (Sys.getenv_opt key) ~default:"" in
  Unix.putenv key value;
  Fun.protect ~finally:(fun () -> Unix.putenv key old) f

let select candidates =
  Tolk.Helpers.select_first_inited ~message:"No interface is available" candidates

let () =
  run __FILE__
    [ formatting; counters;
      test "target strings preserve architecture and interface spelling" (fun () ->
          let t = Target.of_string "remote:host:2+nv:cuda:sm_89" in
          equal string "NV" t.device;
          equal string "CUDA" t.renderer;
          equal string "sm_89" t.arch;
          equal string "remote:host" t.interface;
          equal string "2" t.indices;
          equal string "remote:host:2+NV:CUDA:sm_89" (Target.to_string t));
      test "target strings normalize empty fields without inventing defaults" (fun () ->
          List.iter (fun (input, output) ->
              equal string output (Target.to_string (Target.of_string input)))
            [ "", ""; "cpu::", "CPU"; "+nv:cuda:", "NV:CUDA";
              "PCI:+", "PCI+"; ":0,2+nv", ":0,2+NV";
              "::Apple7", "::Apple7" ]);
      test "target strings reject excess separators" (fun () ->
          List.iter (fun s -> raises_match (function Invalid_argument _ -> true | _ -> false)
              (fun () -> Target.of_string s))
            [ "PCI+NV+CUDA"; "CPU:CLANG:arm64:extra" ]);
      test "per-backend targets and defaults do not leak between contexts" (fun () ->
          with_dev "CPU:CLANG;PCI:2,0+NV:CUDA:sm_89" (fun () ->
              equal string "CPU:CLANG:arm64" (Target.to_string (H.target ~arch:"arm64" "CPU"));
              equal string "PCI:2,0+NV:CUDA:sm_89"
                (Target.to_string (H.target ~arch:"sm_90" "NV:1"));
              equal string "AMD::gfx1100" (Target.to_string (H.target ~arch:"gfx1100" "AMD"));
              raises Exit (fun () -> with_dev "NV:PTX" (fun () ->
                  equal string "PTX" (H.target "NV").renderer;
                  raise Exit));
              equal string "CUDA" (H.target "NV").renderer));
      test "the first matching wildcard supplies the target" (fun () ->
          with_dev "PCI+;NV:CUDA" (fun () ->
              equal string "PCI+NV" (Target.to_string (H.target "NV"))));
      test "DEV selects one interface without trying alternatives" (fun () ->
          with_dev "CPU;PCI+NV" (fun () ->
              equal string "PCI"
                (H.select_interface ~device:"NV:1"
                   [ "NVK", (fun () -> fail "NVK should not be attempted");
                     "PCI", (fun () -> "PCI") ])));
      test "an unknown interface fails before initialization" (fun () ->
          with_dev "USB+AMD" (fun () ->
              raises (Invalid_argument "AMD has no interface \"USB\"") (fun () ->
                  H.select_interface ~device:"AMD" [ "KFD", (fun () -> fail "unexpected initialization") ])));
      test "mock interfaces require explicit selection" (fun () ->
          let candidates = [ "MOCK", (fun () -> "mock"); "PCI", (fun () -> "pci") ] in
          with_dev "NV" (fun () -> equal string "pci" (H.select_interface ~device:"NV" candidates));
          with_dev "MOCK+NV" (fun () -> equal string "mock" (H.select_interface ~device:"NV" candidates)));
      test "legacy target settings point to DEV" (fun () ->
          with_dev "NV" (fun () ->
              with_env "NV_IFACE" "PCI" (fun () ->
                  raises (Invalid_argument "NV_IFACE=PCI is deprecated, use DEV=PCI+NV instead")
                    (fun () -> H.select_interface ~device:"NV" []));
              with_env "NV_CC" "CUDA" (fun () ->
                  raises (Invalid_argument "NV_CC=CUDA is deprecated, use DEV=NV:CUDA instead")
                    (fun () -> H.target "NV"))));
      test "prefers the kernel driver without initializing PCI" (fun () ->
          let calls = ref [] in
          let create name () = calls := name :: !calls; name in
          equal string "KFD" (select [ create "KFD"; create "PCI" ]);
          equal (list string) [ "KFD" ] !calls);
      test "falls back after driver initialization fails" (fun () ->
          let calls = ref [] in
          let driver () = calls := "NVK" :: !calls; failwith "driver unavailable" in
          let pci () = calls := "PCI" :: !calls; "PCI" in
          equal string "PCI" (select [ driver; pci ]);
          equal (list string) [ "PCI"; "NVK" ] !calls);
      test "preserves the error from an explicitly selected interface" (fun () ->
          raises (Failure "PCI unavailable") (fun () ->
              select [ (fun () -> failwith "PCI unavailable") ]));
      test "reports failures from every attempted interface" (fun () ->
          raises
            (Failure
               "No interface is available\nFailure(\"driver unavailable\")\nFailure(\"PCI unavailable\")")
            (fun () ->
              select
                [ (fun () -> failwith "driver unavailable");
                  (fun () -> failwith "PCI unavailable") ]));
      test "does not retry after the selected runtime fails" (fun () ->
          let pci_attempted = ref false in
          let driver () = fun () -> failwith "queue initialization failed" in
          let pci () = pci_attempted := true; fun () -> () in
          let open_runtime = select [ driver; pci ] in
          raises (Failure "queue initialization failed") open_runtime;
          equal bool false !pci_attempted);
      test "does not treat cancellation as an unavailable driver" (fun () ->
          raises Sys.Break (fun () -> select [ (fun () -> raise Sys.Break); (fun () -> ()) ]));
    ]
