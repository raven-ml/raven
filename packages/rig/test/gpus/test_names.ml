(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Devices by name, on the CPU alone: the host's and software devices'
   names, the strings that name no device, and opens's answers. A GPU's
   names are the GPU suite's (test_gpus). *)

open Windtrap

let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal
let opened = result device string

(* The refusal of a string that names no device, as the .mli states it. *)
let no_name s =
  Printf.sprintf
    "%S names no device: a name is KIND or KIND:N, where KIND is one of CPU, \
     METAL, CUDA, NV, AMD, NV-PCI, AMD-PCI, in any case, and N a number with \
     no leading zero"
    s

let host () =
  List.iter
    (fun n -> equal ~msg:n opened (Ok Rig.host) (Rig_gpus.open_ n))
    [ "CPU"; "cpu"; "CPU:0"; "cPu:0" ]

let software () =
  let d = require_ok ~pp:Format.pp_print_string (Rig_gpus.open_ "cpu:2") in
  equal ~msg:"its name" string "CPU:2" (Rig.name d);
  equal ~msg:"Rig.software's device" opened (Rig.software "CPU:2") (Ok d);
  equal ~msg:"its name opens it" opened (Ok d) (Rig_gpus.open_ (Rig.name d));
  equal ~msg:"a larger N" (result string string) (Ok "CPU:17")
    (Result.map Rig.name (Rig_gpus.open_ "CPU:17"))

let not_names =
  [
    "";
    ":";
    "CPU:";
    "CUDA:";
    "CUDA:01";
    "CPU:00";
    "CPU:01";
    "CUDA:-1";
    "CUDA:+1";
    "CUDA:x";
    "CUDA:1x";
    "CPU:0:1";
    "CUDA:99999999999999999999";
    "VULKAN";
    "VULKAN:0";
    " cuda";
    "cuda ";
    "CUDA :1";
    "HOST";
  ]

let refused s =
  equal ~msg:"open_" opened (Error (no_name s)) (Rig_gpus.open_ s);
  equal ~msg:"opens" (result unit string) (Error (no_name s))
    (Rig_gpus.opens s)

let open_all () =
  let names = result (list string) string in
  let open_all s = Result.map (List.map Rig.name) (Rig_gpus.open_all s) in
  equal ~msg:"one name" names (Ok [ "CPU" ]) (open_all "cpu");
  equal ~msg:"in order, blanks trimmed" names
    (Ok [ "CPU:3"; "CPU"; "CPU:2" ])
    (open_all "CPU:3, cpu ,cPu:2");
  equal ~msg:"the first that opens nothing" names (Error (no_name "VULKAN"))
    (open_all "CPU:2,VULKAN,HOST");
  equal ~msg:"an empty name" names (Error (no_name ""))
    (open_all "CPU:2,,CPU:3");
  equal ~msg:"no name" names (Error (no_name "")) (open_all "");
  equal ~msg:"two names of one device" names
    (Error "\"cpu\" and \"CPU:0\" name one device, CPU")
    (open_all "cpu,CPU:2,CPU:0")

let opens_cpu () =
  List.iter
    (fun n -> equal ~msg:n (result unit string) (Ok ()) (Rig_gpus.opens n))
    [ "CPU"; "cpu:0"; "CPU:3" ]

(* The refusal of a GPU this machine has not: GPU 4096 of NVIDIA's kernel
   path, which its opener refuses from the PCI bus's listing before it opens
   the kernel driver. *)
let opens_refusal () =
  let why = function Ok _ -> "" | Error why -> why in
  let here = why (Rig_gpus.open_ "NV:4096") in
  let child = why (Rig_gpus.opens "nv:4096") in
  if here = "" then fail "NV:4096 opened";
  equal ~msg:"the child's refusal" string here child

let () =
  if Sys.win32 then exit 0;
  exit
  @@ run "rig_gpus names"
       [
         group "open_"
           [
             test "CPU, in any case and with :0, names the host" host;
             test "CPU:N names the software device of that name" software;
             cases "a string that names no device is refused, naming the kinds"
               ~id:(Printf.sprintf "%S")
               not_names refused;
           ];
         group "open_all"
           [
             test "a comma list opens each name, refusing the first bad one"
               open_all;
           ];
         group "opens"
           [
             test "a child opens the host's and a software device's names"
               opens_cpu;
             test "a child's refusal is opens's" opens_refusal;
           ];
       ]
