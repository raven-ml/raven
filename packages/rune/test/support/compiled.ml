(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Whether Clang compiles [d]'s programs: a host's, a software device's or
   another machine's host's. *)
let with_clang d =
  Rig.runs_on_host d
  ||
  match Tolk_engine.target d with
  | t -> t.Tolk.Helpers.Target.renderer = "CLANG"
  | exception (Tolk.Device.Uncomputable _ | Invalid_argument _) -> false

let here ?(on = Rig.host) () =
  match Tolk_engine.compiles on with
  | Error reason when with_clang on -> Windtrap.skip ~reason ()
  | Ok () | Error _ -> ()

let set s = List.iter (fun on -> here ~on ()) (Nx.Devices.to_list s)
