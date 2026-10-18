(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Failure walks of GPUs through NVIDIA's kernel driver. Each operation runs in
   a forked child that alone carries a limit: [k] more files, for [k] from 0 to
   the files the operation takes, so that the open of each file it takes fails
   once; or, for a process's first open, too little address space for the
   reservation of the GPU's addresses. The child first opens the GPU
   and runs the operation once, which makes what the process keeps for good.
   After each failure the outcome is one rig_nv.mli or rig_nv_nvidia.mli states,
   the process's files and the GPU's mappings are back where they were, the
   operation succeeds once the limit is lifted, and the GPU opens again. The suite forks
   before any call to NVIDIA's driver. *)

open Windtrap
module N = Rig_nv
module P = Rig_nv_nvidia
module S = Rig_nv_nvidia_support
module H = Rig_gpu_support.Host

external set_space : int -> int = "rig_nv_nvidia_test_set_space"

let strf = Printf.sprintf
let timeout = 60.
let page = 4096

(* OCaml can enumerate sysfs and collect within 64 MiB; the first GPU
   reservation needs 64 GiB, so the limit still refuses that reservation. *)
let runtime_headroom = 64 * 1024 * 1024

(* The most files an operation is walked through. *)
let most_files = 16

(* The process *)

let lines path = In_channel.with_open_text path In_channel.input_lines

let mapped () =
  let size l = Scanf.sscanf_opt l "VmSize: %d kB" (fun k -> k * 1024) in
  Option.get (List.find_map size (lines "/proc/self/status"))

let checked = function 0 -> () | e -> failwith (strf "setrlimit: errno %d" e)

(* Limits *)

type limit = Files of int | Space of int

let pp_limit ppf = function
  | Files k -> Format.fprintf ppf "%d more files" k
  | Space n -> Format.fprintf ppf "%d more bytes of address space" n

(* [under lim f] is [f ()] with the process held to [lim]. *)
let under lim f =
  match lim with
  | Files k -> S.with_limit (S.limit_for k) f
  | Space n ->
      checked (set_space (mapped () + n));
      Fun.protect ~finally:(fun () -> checked (set_space (-1))) f

(* [in_child f] is [f ()] in a forked child, its result sent back through a
   pipe. [f] calls no verb: a failure in the child would end the child's copy of
   the run. *)
let in_child (f : unit -> 'a) : ('a, string) result =
  let r, w = Unix.pipe ~cloexec:true () in
  Rig_support.needs_fork ();
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let result = try Ok (f ()) with e -> Error (Printexc.to_string e) in
      let oc = Unix.out_channel_of_descr w in
      Marshal.to_channel oc result [];
      close_out oc;
      Unix._exit 0
  | pid ->
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      let result = Marshal.from_channel ic in
      close_in ic;
      ignore (Unix.waitpid [] pid);
      result

(* Operations *)

type op = Open | Alloc of Rig_edge.memory | Map_host | Grow

let op_name = function
  | Open -> "open"
  | Alloc Device -> "alloc of device memory"
  | Alloc Pinned -> "alloc of pinned memory"
  | Alloc Mapped -> "alloc of mapped memory"
  | Map_host -> "map_host"
  | Grow -> "a growth of the channels"

(* The operations other than an open act on an open GPU [g], and [map_host] on
   [host], host memory starting on a page. A growth asks for channels whose
   rings hold twice the entries of the last it asked for, [want], from more
   than the 262,144 a ring holds at open. *)
type on = { g : N.t option; host : int; mutable want : int }

let host_bytes = 16 * page

(* [take on op] is what [op] answered, and the function that gives back what it
   took. *)
let take on op =
  let region g = function
    | Some r -> ("Some", fun () -> N.free g r)
    | None -> ("None", ignore)
  in
  match (op, on.g) with
  | Open, _ -> (
      match Rig_gpu_support.opened (P.gpu 0) with
      | Ok g -> ("Ok", fun () -> N.stop g ~fault:None)
      | Error e -> ("Error " ^ e, ignore))
  | Alloc kind, Some g -> region g (N.alloc g kind page)
  | Map_host, Some g ->
      region g (N.map_host g on.host host_bytes)
  | Grow, Some g -> (
      on.want <- 2 * on.want;
      let nops = [| (0, Array.make (2 * on.want) 0) |] in
      match Rig_nv_support.Edge.room g nops with
      | "GROW" -> (
          match N.grow g with
          | Rig_edge.Grown -> ("Grown", ignore)
          | Short n -> (strf "Short %d" n, ignore))
      | a -> ("room " ^ a, ignore))
  | (Alloc _ | Map_host | Grow), None -> invalid_arg "no open GPU"

let answer on op =
  match take on op with
  | r -> r
  | exception N.Fault why -> ("Fault " ^ why, ignore)

let run_once on op =
  let got, give = answer on op in
  give ();
  got

type report = {
  got : string;
  files : int * int;
  maps : string list * string list;
  again : string;
  reopened : bool;
}

(* In a child: opens the GPU, and runs [op] once, unless [cold], which leaves an
   open the process's first; then runs [op] under [lim], and with the limit
   lifted. *)
let attempt ?(cold = false) op lim =
  let g =
    if op = Open then None
    else Some (Result.get_ok (Rig_gpu_support.opened (P.gpu 0)))
  in
  let host = H.pages host_bytes in
  Fun.protect ~finally:(fun () -> H.free_pages host host_bytes) @@ fun () ->
  let on = { g; host; want = 140_000 } in
  if not cold then ignore (run_once on op);
  let files = S.files () and maps = S.mappings () in
  let got, give = under lim (fun () -> answer on op) in
  give ();
  let files = (files, S.files ()) and maps = (maps, S.mappings ()) in
  let again = run_once on op in
  Option.iter (N.stop ~fault:None) g;
  let reopened =
    Result.is_ok
      (Result.map (N.stop ~fault:None) (Rig_gpu_support.opened (P.gpu 0)))
  in
  { got; files; maps; again; reopened }

let succeeded = function "Ok" | "Some" | "Grown" -> true | _ -> false

(* What the .mli allows [op] to answer when a limit stops it: an open's [Error];
   an allocation's or a mapping's [None], or [Fault] for a failure that is no
   refusal; a growth's [Short], since a shortage keeps the device's
   channels. *)
let allowed op got =
  match op with
  | Open -> String.starts_with ~prefix:"Error " got
  | Alloc _ | Map_host ->
      got = "None" || String.starts_with ~prefix:"Fault " got
  | Grow -> String.starts_with ~prefix:"Short " got

(* A failure leaves the process's files and the GPU's mappings as they were.
   An open that succeeds keeps its device's timeline word for the process. *)
let check op lim r =
  let at = Format.asprintf "%s with %a" (op_name op) pp_limit lim in
  if not (succeeded r.got) then begin
    equal
      ~msg:(strf "%s: %S is an answer the .mli states" at r.got)
      bool true (allowed op r.got);
    equal ~msg:(at ^ ": files") int (fst r.files) (snd r.files);
    equal ~msg:(at ^ ": GPU mappings") (list string) (fst r.maps) (snd r.maps)
  end;
  equal ~msg:(at ^ ": again") string "succeeded"
    (if succeeded r.again then "succeeded" else r.again);
  equal ~msg:(at ^ ": opens again") bool true r.reopened

let need_gpu () =
  if Result.value (P.count ()) ~default:0 = 0 then
    skip ~reason:"the machine has no NVIDIA GPU" ()

(* Walks [op] through 0, 1, … more files, up to the first that lets it
   succeed. *)
let walk_files op () =
  need_gpu ();
  let rec go k =
    if k > most_files then failf "%s takes more than %d files" (op_name op) k;
    let lim = Files k in
    let r =
      require_ok ~pp:Format.pp_print_string (in_child (fun () -> attempt op lim))
    in
    check op lim r;
    if not (succeeded r.got) then go (k + 1)
  in
  go 0

(* A process's first open, with too little address space for the GPU. *)
let walk_space () =
  need_gpu ();
  let lim = Space runtime_headroom in
  let r =
    require_ok ~pp:Format.pp_print_string
      (in_child (fun () -> attempt ~cold:true Open lim))
  in
  check Open lim r;
  contains ~sub:"reserving the GPU's addresses" r.got;
  equal ~msg:"refused" bool false (succeeded r.got)

(* An allocation of twice the GPU's memory is refused, filling nothing. *)
let too_large () =
  need_gpu ();
  let got =
    require_ok ~pp:Format.pp_print_string
      (in_child (fun () ->
           let g = Result.get_ok (Rig_gpu_support.opened (P.gpu 0)) in
           let got = N.alloc g Device (2 * (N.facts g).budget) in
           Option.iter (N.free g) got;
           N.stop g ~fault:None;
           Option.is_some got))
  in
  equal ~msg:"given" bool false got

(* What [op] answers with no file left. *)
let starved op =
  require_ok ~pp:Format.pp_print_string
    (in_child (fun () -> (attempt op (Files 0)).got))

let open_starved () =
  need_gpu ();
  let got = starved Open in
  equal ~msg:got bool false
    (String.ends_with ~suffix:"the machine has 0 NVIDIA GPUs" got)

let alloc_starved op () =
  need_gpu ();
  equal string "None" (starved op)

let ops = [ Open; Alloc Device; Alloc Pinned; Alloc Mapped; Map_host; Grow ]

let () =
  exit
    (run "rig_nv_nvidia walk"
       [
         group ~timeout
           "an operation under a limit answers as its .mli states and leaves \
            nothing"
           (List.map
              (fun op -> test (op_name op ^ ", file by file") (walk_files op))
              ops
           @ [
               test "a first open, short of address space" walk_space;
               test "an allocation of twice the GPU's memory is refused"
                 too_large;
             ]);
         group ~timeout "a process out of files"
           (test "fails an open naming the cause, not the machine's GPUs"
              open_starved
           :: List.map
                (fun op ->
                  test (op_name op ^ " answers None") (alloc_starved op))
                [ Alloc Pinned; Alloc Mapped; Map_host ]);
       ])
