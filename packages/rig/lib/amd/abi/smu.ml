(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Boot_defs

(* Messages *)

let message mp1 name = List.assoc_opt name (D.smu_messages mp1)
let clock_request ~clock v = (clock lsl 16) lor v

(* A clock's DPM feature is FEATURE_DPM_ and the clock's name past PPCLK_. *)
let dpm mp1 ~features clock =
  let prefix = "PPCLK_" in
  if not (String.starts_with ~prefix clock) then
    invalid_arg (Printf.sprintf "Smu.dpm: clock %S is not a PPCLK_ name" clock);
  let n = String.length prefix in
  let name = "FEATURE_DPM_" ^ String.sub clock n (String.length clock - n) in
  match message mp1 name with
  | Some bit -> (features lsr bit) land 1 = 1
  | None -> false
