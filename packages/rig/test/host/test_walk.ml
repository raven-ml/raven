(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A failure walk of linking. A child process of the suite limits its address
   space to [n] bytes above what it maps, so that the mapping of an image's
   executable memory fails while the limit is below the image's size, and loads
   an image on the host, for [n] from none to more than the image needs. The
   load answers an image or [Error], the failure leaves no mapping behind, and
   the load succeeds once the limit is lifted. On Linux only: the limit bounds
   no mapping elsewhere. *)

open Windtrap
module S = Rig_host_support

let strf = Printf.sprintf
let page = 4096

(* The room walked, in bytes: none, up to more than the fixture's image. *)
let rooms = [ 0; page; 4 * page; 16 * page; 256 * page ]

let exe =
  let e = Sys.executable_name in
  if Filename.is_relative e then Filename.concat (Sys.getcwd ()) e else e

let obj =
  lazy
    (S.fixture
       ~dir:(Filename.concat Filename.parent_dir_name "fixtures")
       "affine")

let lines path = In_channel.with_open_text path In_channel.input_lines

(* The process's status, read into [status] without a channel. The child reads
   it right before it sets its limit, and nothing it frees after may return
   memory to the system, as musl's C library does with a free block: a channel's
   buffer, the C library's memory, is freed at the channel's close, after its
   bytes were counted, and so is a block too large for the minor heap, which the
   collector frees once unreachable. [status], made as the program starts, stays
   reachable. *)
let status = Bytes.create 8192

let read_status () =
  let fd = Unix.openfile "/proc/self/status" [ O_RDONLY; O_CLOEXEC ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  let rec go at =
    if at = Bytes.length status then failwith "/proc/self/status: too long";
    match Unix.read fd status at (Bytes.length status - at) with
    | 0 -> at
    | n -> go (at + n)
  in
  go 0

(* The bytes the process maps: the [VmSize] of its status, ["VmSize: N kB"],
   parsed in place. *)
let mapped () =
  let n = read_status () and key = "VmSize:" in
  let rec find i =
    if i + String.length key > n then failwith "no VmSize in /proc/self/status"
    else if Bytes.sub_string status i (String.length key) = key then
      i + String.length key
    else find (i + 1)
  in
  let rec kib i k =
    match Bytes.get status i with
    | (' ' | '\t') when k = 0 -> kib (i + 1) k
    | '0' .. '9' as c -> kib (i + 1) ((10 * k) + Char.code c - Char.code '0')
    | _ -> k
  in
  1024 * kib (find 0) 0

let mappings () = List.length (lines "/proc/self/maps")

(* Collects, then drains the host: an unreachable image unloads at a drain, once
   the host's work before it is done. *)
let collect () =
  for _ = 1 to 3 do
    Gc.full_major ();
    ignore (Sys.opaque_identity (Rig.Buffer.create Rig.host 0))
  done

let checked = function 0 -> () | e -> failwith (strf "setrlimit: errno %d" e)

(* [linked ()] is what a load of the fixture on the host answers. *)
let linked () =
  match Rig.Image.load Rig.host (Lazy.force obj) with
  | Ok p ->
      ignore (Sys.opaque_identity p);
      "linked"
  | Error _ -> "refused"

(* The child: links once uncounted, then under a limit [room] bytes above what
   it maps, and prints what the link answered, whether its mappings came back
   once collected, and what a link answers with the limit lifted. It collects
   between counting its mappings and reading what it maps, so that no memory of
   the count returns to the system under the limit ([status]). *)
let child room =
  ignore (linked ());
  collect ();
  let before = mappings () in
  collect ();
  checked (Rig_support.set_address_space (mapped () + room));
  let got = linked () in
  checked (Rig_support.set_address_space max_int);
  collect ();
  let back = mappings () = before in
  Printf.printf "%s\n%s\nagain: %s\n" got
    (if back then "mappings: as before" else "mappings: other")
    (linked ())

let walked room () =
  if not (Sys.file_exists "/proc/self/maps") then
    skip ~reason:"the limit bounds no mapping off Linux" ();
  ignore (Lazy.force obj);
  let args = [| exe; "walk"; string_of_int room |] in
  let ic = Unix.open_process_args_in exe args in
  let out = In_channel.input_all ic in
  (match Unix.close_process_in ic with
  | WEXITED 0 -> ()
  | _ -> failf "the child failed: %s" out);
  match String.split_on_char '\n' (String.trim out) with
  | [ got; back; again ] ->
      if room = 0 then equal ~msg:"with no room" string "refused" got;
      if room = List.nth rooms (List.length rooms - 1) then
        equal ~msg:"with room" string "linked" got;
      equal string "mappings: as before" back;
      equal string "again: linked" again
  | _ -> failf "the child said: %s" out

let () =
  match Array.to_list Sys.argv with
  | [ _; "walk"; room ] -> child (int_of_string room)
  | _ ->
      exit
        (run "rig_host.walk"
           [
             group "linking"
               [
                 cases
                   "a load whose mapping the system refuses is Error and \
                    leaves no mapping, with room of"
                   ~id:(strf "%d bytes") rooms (fun room -> walked room ());
               ];
           ])
