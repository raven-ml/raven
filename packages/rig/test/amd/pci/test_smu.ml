(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The power manager's messages, against amdgpu's message headers. *)

open Windtrap
open Rig_amd_pci

let strf = Printf.sprintf

(* The IDs of amdgpu's message headers (smu_v13_0_0_ppsmc.h,
   smu_v13_0_6_ppsmc.h, smu_v13_0_12_ppsmc.h, smu_v14_0_2_ppsmc.h) and clock
   enumerations, by the MP1 versions amdgpu drives with each. *)
let messages =
  [
    ((14, 0, 3), "PPSMC_MSG_SetSoftMinByFreq", Some 0x19);
    ((14, 0, 3), "PPSMC_MSG_GetDpmFreqByIndex", Some 0x1f);
    ((14, 0, 3), "PPCLK_GFXCLK", Some 0);
    ((13, 0, 10), "PPSMC_MSG_Mode1Reset", Some 0x2f);
    ((13, 0, 14), "PPSMC_MSG_GfxDriverReset", Some 3);
    ((13, 0, 14), "PPCLK_UCLK", Some 3);
    ((13, 0, 14), "PPCLK_GFXCLK", None);
    ((13, 0, 12), "PPSMC_MSG_McaBankDumpDW", Some 0x37);
    ((13, 0, 4), "PPSMC_MSG_GetSmuVersion", None);
  ]

let power =
  group ~timeout:10. "power manager"
    [
      cases "a message has its header's ID"
        ~id:(fun ((a, b, c), n, _) -> strf "%s of MP1 %d.%d.%d" n a b c)
        messages
        (fun (mp1, name, id) -> equal (option int) id (Smu.message mp1 name));
      cases "a clock's DPM runs iff its feature bit is enabled"
        ~id:(fun ((a, b, c), clock, features, _) ->
          strf "%s of MP1 %d.%d.%d, features 0x%x" clock a b c features)
        (* The DPM feature bits of smu14_driver_if_v14_0.h (UCLK 3, FCLK 4,
           SOCCLK 5, GFXCLK 1) and smu_v13_0_6_pmfw.h (UCLK 6, GFXCLK 3). *)
        [
          ((14, 0, 3), "PPCLK_UCLK", (1 lsl 3) lor (1 lsl 5), true);
          ((14, 0, 3), "PPCLK_SOCCLK", (1 lsl 3) lor (1 lsl 5), true);
          ((14, 0, 3), "PPCLK_FCLK", (1 lsl 3) lor (1 lsl 5), false);
          ((14, 0, 3), "PPCLK_GFXCLK", (1 lsl 3) lor (1 lsl 5), false);
          ((14, 0, 3), "PPCLK_UCLK", 0, false);
          ((13, 0, 6), "PPCLK_UCLK", 1 lsl 6, true);
          ((13, 0, 6), "PPCLK_GFXCLK", 1 lsl 6, false);
        ]
        (fun (mp1, clock, features, on) ->
          equal bool on (Smu.dpm mp1 ~features clock));
      cases "a clock's name must start with PPCLK_" ~id:Fun.id
        (* UCLK is shorter than the prefix, and XXXXXXUCLK is as long as UCLK
           with it. *)
        [ ""; "UCLK"; "XXXXXXUCLK"; "ppclk_UCLK" ]
        (fun clock ->
          raises_match (Exn.invalid_arg ~sub:"Smu.dpm") (fun () ->
              Smu.dpm (14, 0, 3) ~features:(-1) clock));
      test "a clock's request holds the clock above the value" (fun () ->
          equal int 0x2_0010 (Smu.clock_request ~clock:2 0x10));
    ]

let () = exit (run "rig_amd_pci.smu" [ power ])
