(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Devices by the names the command line gives them: CPU is the host, CPU:k a
   test device over the host's memory, METAL the Metal device, and CUDA or
   CUDA:i a CUDA device. *)

let test_device name =
  Nx_device.Driver.device ~name ~arch:"test" ~budget:max_int
    (Host_visible
       { memory = Nx_device.Driver.host_memory; mapping = Some Identity })

let of_name name =
  match String.split_on_char ':' (String.uppercase_ascii name) with
  | [ "CPU" ] -> Nx_device.host
  | [ "CPU"; _ ] -> test_device (String.uppercase_ascii name)
  | [ "METAL" ] -> Metal.device ()
  | [ "CUDA" ] -> Nx_cuda_device.v 0
  | [ "CUDA"; i ] -> Nx_cuda_device.v (int_of_string i)
  | _ -> failwith (name ^ ": not a device (CPU, CPU:k, METAL, CUDA, CUDA:i)")

(* [parse s] is the devices [s] names: a CPU device count ([4] is CPU:1 to
   CPU:4) or a comma-separated list of names. *)
let parse s =
  match int_of_string_opt s with
  | Some n when n > 0 ->
      List.init n (fun i -> test_device (Printf.sprintf "CPU:%d" (i + 1)))
  | Some _ -> failwith "--devices: the device count must be positive"
  | None -> List.map (fun n -> of_name (String.trim n)) (String.split_on_char ',' s)
