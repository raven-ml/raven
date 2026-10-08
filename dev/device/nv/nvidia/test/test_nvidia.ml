(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module N = Device_nv_nvidia

let () =
  exit
  @@ run "device_nv_nvidia"
       [
         group ~timeout:10. "numbering"
           [
             test "the GPUs are NVIDIA's display functions in bus order"
               (fun () ->
                 equal (list string)
                   [
                     "0000:01:00.0";
                     "0000:0a:00.0";
                     "ffff:00:00.0";
                     "10000:00:01.0";
                   ]
                   (N.gpus_at "fixtures"));
             test "a machine without PCI functions has no GPU" (fun () ->
                 equal (list string) [] (N.gpus_at "no-such-directory"));
             test "names GPU 0 NV and GPU i NV:i" (fun () ->
                 equal (list string) [ "NV"; "NV:1"; "NV:7" ]
                   (List.map N.device_name [ 0; 1; 7 ]));
           ];
         group ~timeout:10. "opening"
           [
             test "a negative GPU raises" (fun () ->
                 raises_match Exn.invalid_arg (fun () -> N.open_ (-1));
                 raises_match Exn.invalid_arg (fun () -> N.device_name (-1)));
             test "a GPU past the count is an error naming the count" (fun () ->
                 let n = N.count () in
                 let e = require_error (N.open_ n) in
                 contains ~sub:(Printf.sprintf "has %d NVIDIA GPUs" n) e);
           ];
       ]
