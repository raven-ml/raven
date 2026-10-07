(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.metal.device off macOS, where Metal does not exist: the library links, and
   no device opens. *)

open Windtrap

let () =
  exit
    (run "nx.metal.device off macOS"
       [
         test "there is no device" (fun () ->
             equal int 0 (Nx_metal_device.count ()));
         test "get refuses, naming macOS" (fun () ->
             equal string "METAL: Metal exists on macOS only"
               (require_error (Nx_metal_device.get 0));
             equal string "METAL:1: Metal exists on macOS only"
               (require_error (Nx_metal_device.get 1)));
         test "msg_send is 0" (fun () ->
             equal nativeint 0n Nx_metal_device.msg_send);
       ])
