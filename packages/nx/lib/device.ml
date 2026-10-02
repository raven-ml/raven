(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = Nx_device.t
type want = Host | Cpu of int | Metal | Cuda of int | Nv of int | Amd of int

let host = Nx_device.host
let name = Nx_device.name
let pp = Nx_device.pp

(* Test devices: ["CPU:k"] holds its values in the host's memory, which it maps
   as it is, and loads no programs, so the host computes on it. Each [k] is one
   device, made by its first use. *)

let tests = Hashtbl.create 4
let tests_lock = Mutex.create ()

let test k =
  Mutex.protect tests_lock @@ fun () ->
  match Hashtbl.find_opt tests k with
  | Some d -> d
  | None ->
      let d =
        Nx_device.Driver.device
          ~name:(Printf.sprintf "CPU:%d" k)
          ~arch:"test" ~budget:max_int
          (Host_visible
             { memory = Nx_device.Driver.host_memory; mapping = Some Identity })
      in
      Hashtbl.add tests k d;
      d

let get = function
  | Host | Cpu 0 -> Ok host
  | Cpu k when k < 0 -> invalid_arg (Printf.sprintf "Nx.Device.get: CPU:%d" k)
  | Cpu k -> Ok (test k)
  | Metal -> Device_metal.get ()
  | Cuda i -> Nx_cuda_device.get i
  | Nv i -> Nx_nv_device.get i
  | Amd i -> Nx_amd_device.get i

let open_ w = match get w with Ok d -> d | Error e -> failwith e
let cpu k = open_ (Cpu k)
let metal () = open_ Metal
let cuda i = open_ (Cuda i)
let nv i = open_ (Nv i)
let amd i = open_ (Amd i)

(* [first_of what ws] is the first device of [ws] that opens. *)
let first_of what ws =
  if ws = [] then invalid_arg (what ^ ": no device");
  let rec go errors = function
    | [] ->
        failwith (what ^ ": none opens: " ^ String.concat "; " (List.rev errors))
    | w :: ws -> (
        match get w with Ok d -> d | Error e -> go (e :: errors) ws)
  in
  go [] ws

let first ws = first_of "Nx.Device.first" ws
let gpu () = first_of "Nx.Device.gpu" [ Metal; Cuda 0; Nv 0; Amd 0 ]

let all ws =
  let opened = List.map get ws in
  match
    List.filter_map (function Error e -> Some e | Ok _ -> None) opened
  with
  | [] -> List.map Result.get_ok opened
  | errors -> failwith ("Nx.Device.all: " ^ String.concat "; " errors)

(* Names *)

let want_of_name s =
  let index i =
    match int_of_string_opt i with Some i when i >= 0 -> Some i | _ -> None
  in
  match String.split_on_char ':' (String.uppercase_ascii (String.trim s)) with
  | [ "CPU" ] -> Some Host
  | [ "CPU"; k ] -> Option.map (fun k -> Cpu k) (index k)
  | [ "METAL" ] -> Some Metal
  | [ "CUDA" ] -> Some (Cuda 0)
  | [ "CUDA"; i ] -> Option.map (fun i -> Cuda i) (index i)
  | [ "NV" ] -> Some (Nv 0)
  | [ "NV"; i ] -> Option.map (fun i -> Nv i) (index i)
  | [ "AMD" ] -> Some (Amd 0)
  | [ "AMD"; i ] -> Option.map (fun i -> Amd i) (index i)
  | _ -> None

let of_string s =
  let rec go acc = function
    | [] -> Ok (List.rev acc)
    | n :: ns -> (
        match want_of_name n with
        | Some w -> go (w :: acc) ns
        | None ->
            Error
              (Printf.sprintf
                 "%S is not a device; devices are CPU, CPU:k, METAL, CUDA:i, \
                  NV:i and AMD:i"
                 (String.trim n)))
  in
  go [] (String.split_on_char ',' s)
