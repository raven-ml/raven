(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A failure walk of linking. A child process of the suite limits its address
   space to [n] bytes above what it maps, so that the mapping of an image's
   executable memory fails while the limit is below the image's size, and
   loads an image on the host, for [n] from none to more than the image needs.
   The load answers an image or [Error], the failure leaves no mapping behind,
   and the load succeeds once the limit is lifted. On Linux only: the limit
   bounds no mapping elsewhere. *)

open Windtrap
module S = Rig_host_support

let strf = Printf.sprintf
let timeout = 60.
let page = 4096

(* The room walked, in bytes: none, up to more than the fixture's image. *)
let rooms = [ 0; page; 4 * page; 16 * page; 256 * page ]

let exe =
  let e = Sys.executable_name in
  if Filename.is_relative e then Filename.concat (Sys.getcwd ()) e else e

let obj =
  S.fixture ~dir:Filename.(concat (dirname (dirname exe)) "fixtures") "affine"

let lines path = In_channel.with_open_text path In_channel.input_lines

(* The bytes the process maps: the [VmSize] of its status. *)
let mapped () =
  let field l = Scanf.sscanf_opt l "VmSize: %d kB" (fun k -> k * 1024) in
  match List.find_map field (lines "/proc/self/status") with
  | Some n -> n
  | None -> failwith "no VmSize in /proc/self/status"

let mappings () = List.length (lines "/proc/self/maps")

(* Collects, then drains the host: an unreachable image unloads at a drain,
   once the host's work before it is done. *)
let collect () =
  for _ = 1 to 3 do
    Gc.full_major ();
    ignore (Sys.opaque_identity (Rig.Buffer.create Rig.host 0))
  done

let checked = function 0 -> () | e -> failwith (strf "setrlimit: errno %d" e)

(* [linked ()] is what a load of the fixture on the host answers. *)
let linked () =
  match Rig.Image.load Rig.host obj with
  | Ok p ->
      ignore (Sys.opaque_identity p);
      "linked"
  | Error _ -> "refused"

(* The child: links once uncounted, then under a limit [room] bytes above what
   it maps, and prints what the link answered, whether its mappings came back
   once collected, and what a link answers with the limit lifted. *)
let child room =
  ignore (linked ());
  collect ();
  let before = mappings () in
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
             group ~timeout
               "a link whose mapping fails answers Error and leaves nothing"
               (List.map
                  (fun room ->
                    test (strf "%d bytes of room" room) (walked room))
                  rooms);
           ])
