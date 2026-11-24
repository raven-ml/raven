(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let () =
  let cc = Tolk.Setting.value Tolk.Setting.cc in
  match Tolk_engine.compiles Rig.host with
  | Ok () ->
      Printf.printf "%s compiles for %s\n" cc (Tolk_engine.target Rig.host).arch
  | Error why ->
      Printf.eprintf "The host compiles nothing: %s\n" why;
      exit 1
