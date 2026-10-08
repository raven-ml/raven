(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Messages *)

let message mp1 name = List.assoc_opt name (D.smu_messages mp1)
let clock_request ~clock v = (clock lsl 16) lor v

(* A clock's DPM feature is FEATURE_DPM_ and the clock's name past PPCLK_. *)
let dpm mp1 ~features clock =
  let name = "FEATURE_DPM_" ^ String.sub clock 6 (String.length clock - 6) in
  match message mp1 name with
  | Some bit -> (features lsr bit) land 1 = 1
  | None -> false

let dotted (a, b, c) = strf "%d.%d.%d" a b c

type t = {
  r : Regs.t;
  gmc : Gmc.t;
  mp0 : Discovery.version;
  mp1 : Discovery.version;
  table : int; (* the driver table's physical address *)
  levels : (int, int list) Hashtbl.t; (* each clock's frequencies, read once *)
}

let make r gmc ~table =
  let l = Regs.layout_of r in
  let mp1 = Regs.version l D.mp1_hwid in
  if D.smu_messages mp1 = [] then
    invalid_argf "Rig_amd_pci.open_: MP1 %s has no messages" (dotted mp1);
  {
    r;
    gmc;
    mp0 = Regs.version l D.mp0_hwid;
    mp1;
    table;
    levels = Hashtbl.create 4;
  }

let id s name =
  match message s.mp1 name with
  | Some id -> id
  | None -> invalid_argf "Rig_amd_pci.open_: the power manager has no %s" name

(* The message registers: response, argument and ID. The debug port is a second
   set the firmware answers on its own. *)
let port = ("mmMP1_SMN_C2PMSG_90", "mmMP1_SMN_C2PMSG_82", "mmMP1_SMN_C2PMSG_66")

let debug_port =
  ("mmMP1_SMN_C2PMSG_54", "mmMP1_SMN_C2PMSG_53", "mmMP1_SMN_C2PMSG_75")

let done_ = 1
let default_ms = 10_000

(* Sends message [m] with [param] and waits for the answer, as the kernel's
   smu_cmn_send_smc_msg_with_param does; the argument register then holds the
   reply. *)
(* [ask s m param] sends message [m] and is [Ok reply], or [Error answer] for
   any answer but done. *)
let ask ?(debug = false) ?(ms = default_ms) s m param =
  let resp, arg, cmd = if debug then debug_port else port in
  Regs.write ~value:0 s.r resp [];
  Regs.write ~value:param s.r arg [];
  Regs.write ~value:m s.r cmd [];
  Regs.wait ~ms s.r (strf "the power manager's message 0x%x" m) (fun () ->
      Regs.read s.r resp <> 0);
  let answer = Regs.read s.r resp in
  if answer = done_ then Ok (Regs.read s.r arg) else Error answer

let send ?debug ?ms s m param =
  match ask ?debug ?ms s m param with
  | Ok reply -> reply
  | Error answer ->
      raise
        (Regs.Stuck
           (strf "the power manager's message 0x%x answered 0x%x" m answer))

let lo32 v = v land 0xffff_ffff
let hi32 v = (v lsr 32) land 0xffff_ffff

let start s =
  let mc = Gmc.mc s.gmc s.table in
  ignore (send s (id s "PPSMC_MSG_SetDriverDramAddrHigh") (hi32 mc));
  ignore (send s (id s "PPSMC_MSG_SetDriverDramAddrLow") (lo32 mc));
  ignore (send s (id s "PPSMC_MSG_EnableAllSmuFeatures") 0)

(* A power manager that runs answers within 100 ms. *)
let alive_ms = 100

let alive s =
  match send ~ms:alive_ms s (id s "PPSMC_MSG_GetSmuVersion") 0 with
  | _ -> true
  | exception Regs.Stuck _ -> false

(* The enabled features, as the low and high words the power manager reports:
   GetRunningSmuFeatures on SMU 13.0.0, 13.0.7 and 14, GetEnabledSmuFeatures on
   13.0.6 and 13.0.12 (smu_cmn_get_enabled_mask). *)
let features s =
  let word half =
    let names =
      [
        "PPSMC_MSG_GetRunningSmuFeatures" ^ half;
        "PPSMC_MSG_GetEnabledSmuFeatures" ^ half;
      ]
    in
    match List.find_map (message s.mp1) names with
    | Some m -> send s m 0
    | None ->
        (* Regs.layout refuses a power manager that reports no features. *)
        assert false
  in
  word "Low" lor (word "High" lsl 32)

(* A clock's frequencies, from the count its last index answers: at most 16, the
   most levels of a clock SMU 11, 13 and 14 hold (MAX_DPM_LEVELS of smu_v11_0.h,
   smu_v13_0.h and smu_v14_0.h). *)
let max_levels = 16

let frequencies s clock =
  match Hashtbl.find_opt s.levels clock with
  | Some l -> l
  | None ->
      let by_index = id s "PPSMC_MSG_GetDpmFreqByIndex" in
      let q i = send s by_index (clock_request ~clock i) land 0x7fff_ffff in
      let n = q 0xff in
      if n > max_levels then
        raise
          (Regs.Stuck
             (strf "the power manager counts %d levels of clock %d, past %d" n
                clock max_levels));
      let l = List.init n q in
      Hashtbl.replace s.levels clock l;
      l

(* A power manager may refuse a clock's soft minimum within 20 ms; its maximum
   then still holds the clock below. GFX9's graphics clock is the firmware's
   alone: the table has no such clock. *)
let soft_min_ms = 20

(* A clock whose DPM is off, as before the power manager's features are enabled,
   runs at its boot frequency, which no request changes. *)
let clocks s level =
  let gc = Regs.version (Regs.layout_of s.r) D.gc_hwid in
  let features = features s in
  List.iter
    (fun name ->
      match message s.mp1 name with
      | Some clock when dpm s.mp1 ~features name -> (
          match frequencies s clock with
          | [] -> ()
          | l ->
              let v =
                match level with
                | `Lowest -> List.hd l
                | `Highest -> List.nth l (List.length l - 1)
              in
              (try
                 ignore
                   (send ~ms:soft_min_ms s
                      (id s "PPSMC_MSG_SetSoftMinByFreq")
                      (clock_request ~clock v))
               with Regs.Stuck _ -> ());
              if gc >= (10, 0, 0) then
                ignore
                  (send s
                     (id s "PPSMC_MSG_SetSoftMaxByFreq")
                     (clock_request ~clock v)))
      | Some _ | None -> ())
    [ "PPCLK_UCLK"; "PPCLK_FCLK"; "PPCLK_SOCCLK"; "PPCLK_GFXCLK" ]

(* Reset *)

(* The debug port's mode 1 reset, of the power managers of GPUs whose MP0 is
   13.0.0, 13.0.7, 13.0.10 or 14 and later. *)
let debug_mode1 = 2

(* The GPU answers no register access while it resets: 1 s on SMU 14, 500 ms on
   earlier ones (smu_v14_0_2_mode1_reset, SMU13_MODE1_RESET_WAIT_TIME_IN_MS of
   smu_v13_0.h). *)
let after_reset_ms s = if s.mp1 >= (14, 0, 0) then 1_000 else 500
let answer_ms = 2_000
let amd = 0x1002

let reset s =
  let debug =
    s.mp0 >= (14, 0, 0)
    || List.mem s.mp0 [ (13, 0, 0); (13, 0, 7); (13, 0, 10) ]
  in
  let driver = List.mem s.mp0 [ (13, 0, 6); (13, 0, 12); (13, 0, 15) ] in
  (* The firmware resets before it answers, so no answer is awaited. *)
  let resp, arg, cmd = if debug then debug_port else port in
  let m, param =
    if debug then (debug_mode1, 0)
    else if driver then (id s "PPSMC_MSG_GfxDriverReset", 1)
    else (id s "PPSMC_MSG_Mode1Reset", 0)
  in
  Regs.write ~value:0 s.r resp [];
  Regs.write ~value:param s.r arg [];
  Regs.write ~value:m s.r cmd [];
  if not (Gmc.hive s.gmc) then begin
    Regs.pause s.r (after_reset_ms s);
    (* Configuration reads fail fast on a GPU still in reset, where register
       reads would stall the bus. *)
    let fn = Regs.fn s.r in
    Regs.wait ~ms:answer_ms s.r
      "the GPU after its reset; a power cycle recovers it" (fun () ->
        Rig_pci.Function.config16 fn 0 = amd)
  end

(* Machine-check banks *)

(* A bank is 16 64-bit registers, each read as two 32-bit halves. *)
let bank_registers = 16

let banks s =
  let read ~uncorrectable =
    let count, dump =
      if uncorrectable then
        ("PPSMC_MSG_QueryValidMcaCount", "PPSMC_MSG_McaBankDumpDW")
      else ("PPSMC_MSG_QueryValidMcaCeCount", "PPSMC_MSG_McaBankCeDumpDW")
    in
    match (message s.mp1 count, message s.mp1 dump) with
    | Some count, Some dump ->
        let half bank i = send s dump ((bank lsl 16) lor i) in
        List.init (send s count 0) (fun bank ->
            List.init bank_registers (fun i ->
                (half bank ((i * 8) + 4) lsl 32) lor half bank (i * 8)))
    | _ -> []
  in
  read ~uncorrectable:true @ read ~uncorrectable:false
  |> List.map (fun regs -> String.concat " " (List.map (strf "0x%x") regs))
  |> String.concat "; "
