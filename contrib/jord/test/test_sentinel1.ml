(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord.Sentinel1

(* STAC's sar extension: HH, VV, HV and VH. *)
let names =
  cases "each polarisation's name reads back" ~id:snd
    [ (HH, "HH"); (HV, "HV"); (VH, "VH"); (VV, "VV") ]
    (fun (p, n) ->
      equal string n (name p);
      equal (option string) (Some n) (Option.map name (of_name n)))

let others =
  cases "a name of no polarisation reads as none" ~id:(Printf.sprintf "%S")
    [ ""; "vv"; "V"; "VVH"; "RH"; " VV" ]
    (fun s -> equal (option string) None (Option.map name (of_name s)))

let () =
  exit (run "jord.sentinel1" [ group "polarisations" [ names; others ] ])
