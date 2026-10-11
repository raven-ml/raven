(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Boot_defs

let strf = Printf.sprintf

type report = Page_fault | Fault of string

(* Decoding *)

(* The bits [lo] to [hi] of [v]. *)
let bits v lo hi = (v lsr lo) land ((1 lsl (hi - lo + 1)) - 1)

(* The sources whose interrupts report an error, as the kernel driver treats
   them; the others wake a sleep and report nothing. *)
let errors =
  [
    "CP_BAD_OPCODE_ERROR";
    "CP_ECC_ERROR";
    "CP_FUE_ERROR";
    "CP_GDS_ALLOC_ERROR";
    "CP_GPF";
    "CP_PM4_PKT_RSVD_BIT_ERROR";
    "CP_PRIV_INSTR_FAULT";
    "CP_PRIV_REG_FAULT";
    "CP_WAIT_MEM_SEM_FAULT";
    "GRBM_RD_TIMEOUT_ERROR";
    "RLC_GC_FED_INTERRUPT";
    "SDMA_ATOMIC_TIMEOUT";
    "SDMA_DOORBELL_INVALID";
    "SDMA_ECC";
    "SDMA_FROZEN";
    "SDMA_PAGE_FAULT";
    "SDMA_PAGE_NULL";
    "SDMA_PAGE_TIMEOUT";
    "SDMA_POLL_TIMEOUT";
    "SDMA_QUEUE_HANG";
    "SDMA_SEM_INCOMPLETE_TIMEOUT";
    "SDMA_SEM_WAIT_FAIL_TIMEOUT";
    "SDMA_SRAM_ECC";
    "SDMA_VM_HOLE";
    "SDMA_XNACK";
    "UTCL2_DATA_POISONING";
  ]

(* A shader's interrupt of encoding 2 reports an error: its type indexes
   these. *)
let shader_errors = [| "EDC_FUE"; "ILLEGAL_INST"; "MEMVIOL"; "EDC_FED" |]
let shader_error_encoding = 2
let major (m, _, _) = m

(* The block whose sources an interrupt of [client] names, and the client's
   name. SOC21 GPUs (GC 11 on) report GC's and SDMA's through two clients;
   SOC15's through GC's, its shader engines' and each SDMA engine's. *)
let block ~gc ~sdma client =
  if major gc >= 11 then
    let block =
      if
        client = D.soc21_ih_clientid_grbm_cp || client = D.soc21_ih_clientid_gfx
      then Some (strf "GFX_%d_" (major gc))
      else None
    in
    (block, D.soc21_client_name client)
  else
    let gfx =
      D.
        [
          soc15_ih_clientid_grbm_cp;
          soc15_ih_clientid_se0sh;
          soc15_ih_clientid_se1sh;
          soc15_ih_clientid_se2sh;
          soc15_ih_clientid_se3sh;
        ]
    in
    let sdmas =
      D.
        [
          soc15_ih_clientid_sdma0;
          soc15_ih_clientid_sdma1;
          soc15_ih_clientid_sdma2;
          soc15_ih_clientid_sdma3;
          soc15_ih_clientid_sdma4;
          soc15_ih_clientid_sdma5;
          soc15_ih_clientid_sdma6;
          soc15_ih_clientid_sdma7;
        ]
    in
    let block =
      if List.mem client gfx then Some "GFX_9_"
      else if List.mem client sdmas then Some (strf "SDMA0_%d_" (major sdma))
      else None
    in
    (block, D.soc15_client_name client)

let source block id =
  List.find_map
    (fun (b, i, name) ->
      if i = id && String.starts_with ~prefix:block b then Some name else None)
    D.ih_sources

let decode ~gc ~sdma e =
  if Array.length e <> 8 then
    invalid_arg
      (strf "Ih.decode: an entry of %d words, expected 8" (Array.length e));
  let client = bits e.(0) 0 7 and id = bits e.(0) 8 15 in
  let block, client_name = block ~gc ~sdma client in
  let name =
    Option.value ~default:"" (Option.bind block (fun b -> source b id))
  in
  let line () =
    strf
      "interrupt client=%s src=%s(%d) ring=%d vmid=%d(%d) pasid=%d node=%d \
       ctx=[0x%x, 0x%x, 0x%x, 0x%x]"
      (if client_name = "" then string_of_int client else client_name)
      name id
      (bits e.(0) 16 23)
      (bits e.(0) 24 27)
      (bits e.(0) 31 31)
      (bits e.(3) 0 15)
      (bits e.(3) 16 23)
      e.(4) e.(5) e.(6) e.(7)
  in
  let soc21 = major gc >= 11 in
  if name = "UTCL2_FAULT" || ((not soc21) && client = D.soc15_ih_clientid_utcl2)
  then Some Page_fault
  else if name = "SQ_INTERRUPT_ID" then
    let encoding = if soc21 then bits e.(5) 6 7 else bits e.(4) 26 27 in
    if encoding <> shader_error_encoding then None
    else
      (* The kernel's SQ_INTERRUPT_WORD_WAVE: its error type in bits 21-24 of
         the first context word on SOC21, in bits 4-7 of the second on SOC15. *)
      let kind = if soc21 then bits e.(4) 21 24 else bits e.(5) 4 7 in
      let kind =
        if kind < Array.length shader_errors then shader_errors.(kind)
        else string_of_int kind
      in
      Some (Fault (strf "%s: shader error %s" (line ()) kind))
  else if List.mem name errors then Some (Fault (line ()))
  else None
