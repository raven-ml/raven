(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt entries, decoded. *)

open Windtrap
open Rig_amd_pci

let gfx9 = (9, 4, 3)
let gfx11 = (11, 0, 0)
let gfx12 = (12, 0, 1)

(* An entry as the IH v6 lays it out: client in bits 0-7 of word 0, source in
   8-15, ring 16-23, VMID 24-27; PASID in bits 0-15 of word 3, node 16-23; four
   context words from word 4. Client and source IDs are soc15_ih_clientid.h's
   and the ivsrcid headers'. *)
let entry ?(ctx = [| 0; 0; 0; 0 |]) ?(vmid = 0) ?(pasid = 0) client source =
  [|
    client lor (source lsl 8) lor (vmid lsl 24);
    0;
    0;
    pasid;
    ctx.(0);
    ctx.(1);
    ctx.(2);
    ctx.(3);
  |]

let grbm_cp = 0x14
let soc21_gfx = 0xa
let soc15_sdma0 = 0x8
let soc15_utcl2 = 0x1b
let sdma4 = (4, 4, 2)
let sdma7 = (7, 0, 1)

let report =
  Testable.structural ~pp:(fun ppf -> function
    | Ih.Page_fault -> Format.pp_print_string ppf "Page_fault"
    | Ih.Fault s -> Format.fprintf ppf "Fault %S" s)

let decoded = option report

let interrupts =
  group ~timeout:10. "interrupts"
    [
      cases "a release reports nothing" ~id:fst
        [
          ("GFX12 end of pipe", (gfx12, sdma7, entry grbm_cp 0xb5));
          ("GFX12 copy trap", (gfx12, sdma7, entry soc21_gfx 0x31));
          ("GFX9 end of pipe", (gfx9, sdma4, entry grbm_cp 0xb5));
          ("GFX9 copy trap", (gfx9, sdma4, entry soc15_sdma0 0xe0));
        ]
        (fun (_, (gc, sdma, e)) -> equal decoded None (Ih.decode ~gc ~sdma e));
      cases "a page walker's fault is a page fault" ~id:fst
        [
          ("GFX12", (gfx12, sdma7, entry soc21_gfx 0));
          ("GFX9", (gfx9, sdma4, entry soc15_utcl2 0));
        ]
        (fun (_, (gc, sdma, e)) ->
          equal decoded (Some Ih.Page_fault) (Ih.decode ~gc ~sdma e));
      test "an error names its client, source and words" (fun () ->
          equal decoded
            (Some
               (Ih.Fault
                  "interrupt client=GRBM_CP src=CP_PRIV_REG_FAULT(184) ring=0 \
                   vmid=3(0) pasid=7 node=0 ctx=[0x1, 0x2, 0x3, 0x4]"))
            (Ih.decode ~gc:gfx11 ~sdma:(6, 0, 0)
               (entry ~vmid:3 ~pasid:7 ~ctx:[| 1; 2; 3; 4 |] grbm_cp 0xb8)));
      cases "a shader error names its kind" ~id:fst
        [
          ( "GFX12 illegal instruction",
            (gfx12, [| 1 lsl 21; 2 lsl 6; 0; 0 |], "ILLEGAL_INST") );
          ( "GFX9 memory violation",
            (gfx9, [| 2 lsl 26; 2 lsl 4; 0; 0 |], "MEMVIOL") );
        ]
        (fun (_, (gc, ctx, kind)) ->
          match Ih.decode ~gc ~sdma:sdma7 (entry ~ctx grbm_cp 0xef) with
          | Some (Ih.Fault s) -> ends_with ~suffix:("shader error " ^ kind) s
          | r -> failf "%a" (Testable.pp decoded) r);
      cases "an interrupt that reports no error is none" ~id:fst
        [
          ( "a shader's other interrupt",
            (gfx12, entry ~ctx:[| 0; 1 lsl 6; 0; 0 |] grbm_cp 0xef) );
          ("an idle GC", (gfx9, entry grbm_cp 0xe9));
          ("a client with no sources", (gfx12, entry 0x1e 0));
        ]
        (fun (_, (gc, e)) -> equal decoded None (Ih.decode ~gc ~sdma:sdma7 e));
      test "an entry of another length raises" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Ih.decode ~gc:gfx12 ~sdma:sdma7 [| 0 |]));
    ]

let () = exit (run "rig_amd_pci.ih" [ interrupts ])
