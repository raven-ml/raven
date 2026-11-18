(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type gpu = {
  name : string;
  bus : string option;
  open_ : unit -> (Rig.t, string) result;
}

type path = { kind : string; gpus : unit -> (gpu list, string) result }

(* A path and how it opens its GPU [i]. Its kind is its GPU 0's device's name,
   which making its opener reads without I/O. *)
type way = { path : path; open_gpu : int -> (Rig.t, string) result }

let gpu (o : _ Rig_edge.opener) bus =
  { name = o.name; bus; open_ = (fun () -> Rig.open_ o) }

let way (opener : int -> _ Rig_edge.opener) gpus =
  {
    path = { kind = (opener 0).name; gpus };
    open_gpu = (fun i -> Rig.open_ (opener i));
  }

let listed buses (opener : int -> _ Rig_edge.opener) =
  way opener (fun () ->
      Result.map (List.mapi (fun i b -> gpu (opener i) (Some b))) (buses ()))

let metal () =
  let gpus n = List.init n (fun i -> gpu (Rig_metal.gpu i) None) in
  Result.map gpus (Rig_metal.count ())

let ways ?firmware () =
  [
    way Rig_metal.gpu metal;
    listed Rig_cuda.buses Rig_cuda.gpu;
    listed Rig_nv_nvidia.buses Rig_nv_nvidia.gpu;
    listed Rig_amd_amdgpu.buses Rig_amd_amdgpu.gpu;
    listed Rig_nv_pci.buses (Rig_nv_pci.gpu ?firmware);
    listed Rig_amd_pci.buses (Rig_amd_pci.gpu ?firmware);
  ]

let paths ?firmware () = List.map (fun w -> w.path) (ways ?firmware ())

(* A refusal names what refused, unless its reason starts with that name, as
   the driver-less paths' do. *)
let named name why =
  if String.starts_with ~prefix:(name ^ ":") why then why
  else name ^ ": " ^ why

(* Each path is listed once the paths before it have opened no GPU. *)
let first ?firmware () =
  let rec gpus refused gs rest =
    match gs with
    | [] -> paths_from refused rest
    | g :: gs -> (
        match g.open_ () with
        | Ok d -> Ok d
        | Error why -> gpus (named g.name why :: refused) gs rest)
  and paths_from refused = function
    | [] when refused = [] -> Error "no path of rig sees a GPU here"
    | [] -> Error ("no GPU opens: " ^ String.concat "; " (List.rev refused))
    | p :: rest -> (
        match p.gpus () with
        | Ok gs -> gpus refused gs rest
        | Error why -> paths_from (named p.kind why :: refused) rest)
  in
  paths_from [] (paths ?firmware ())

(* Names *)

let strf = Printf.sprintf

(* [N] of [KIND:N]: decimal digits with no sign and no leading zero. *)
let index s =
  let digit c = c >= '0' && c <= '9' in
  if s = "" || not (String.for_all digit s) then None
  else if s.[0] = '0' && s <> "0" then None
  else int_of_string_opt s

type device = Cpu of int | Gpu of way * int

let device name =
  let ways = ways () in
  let kind, i =
    match String.index_opt name ':' with
    | None -> (name, Some 0)
    | Some k ->
        ( String.sub name 0 k,
          index (String.sub name (k + 1) (String.length name - k - 1)) )
  in
  let kind = String.uppercase_ascii kind in
  let is_kind w = w.path.kind = kind in
  match (i, List.find_opt is_kind ways) with
  | Some i, _ when kind = "CPU" -> Ok (Cpu i)
  | Some i, Some w -> Ok (Gpu (w, i))
  | _ ->
      let kinds = "CPU" :: List.map (fun w -> w.path.kind) ways in
      Error
        (strf
           "%S names no device: a name is KIND or KIND:N, where KIND is one of \
            %s, in any case, and N a number with no leading zero"
           name (String.concat ", " kinds))

let open_ name =
  match device name with
  | Error _ as e -> e
  | Ok (Cpu 0) -> Ok Rig.host
  | Ok (Cpu k) -> Rig.software (strf "CPU:%d" k)
  | Ok (Gpu (w, i)) -> w.open_gpu i

(* The child writes ['o'], or ['e'] and its refusal, then leaves without
   running this process's exit, whatever its open or its write raised: an
   answer it could not write reads as its exit. *)
let in_child name r w =
  (try
     Unix.close r;
     let answer =
       match open_ name with
       | Ok _ -> "o"
       | Error why -> "e" ^ why
       | exception e -> "e" ^ Printexc.to_string e
     in
     let oc = Unix.out_channel_of_descr w in
     output_string oc answer;
     flush oc
   with _ -> ());
  Unix._exit 0

let rec reap pid =
  match Unix.waitpid [] pid with
  | _, status -> status
  | exception Unix.Unix_error (Unix.EINTR, _, _) -> reap pid

(* The child's answer, read to its end. The child is reaped and the pipe
   closed even when the read raises. *)
let answer pid r =
  let ic = Unix.in_channel_of_descr r in
  match In_channel.input_all ic with
  | answer ->
      close_in ic;
      (reap pid, answer)
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      close_in_noerr ic;
      ignore (reap pid);
      Printexc.raise_with_backtrace e bt

let opens name =
  match device name with
  | Error why -> Error why
  | Ok _ -> (
      let r, w = Unix.pipe ~cloexec:true () in
      match Unix.fork () with
      | exception e ->
          let bt = Printexc.get_raw_backtrace () in
          Unix.close r;
          Unix.close w;
          Printexc.raise_with_backtrace e bt
      | 0 -> in_child name r w
      | pid -> (
          Unix.close w;
          let died why =
            Error (strf "%s: the child that opened it %s" name why)
          in
          match answer pid r with
          | Unix.WEXITED 0, "o" -> Ok ()
          | Unix.WEXITED 0, a when String.starts_with ~prefix:"e" a ->
              Error (String.sub a 1 (String.length a - 1))
          | Unix.WEXITED n, _ -> died (strf "exited %d" n)
          | (Unix.WSIGNALED s | Unix.WSTOPPED s), _ ->
              died ("was killed by " ^ Sys.signal_to_string s)))
