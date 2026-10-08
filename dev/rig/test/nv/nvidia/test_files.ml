(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Opens in a fresh process. Under a limit on the process's files they reach a
   failure after each file an open takes; with an address the path reserves
   taken, the open fails and gives back what it reserved. Each runs in a forked
   child alone, so nothing else on the machine runs short of files; the suite
   forks before any call to NVIDIA's driver. *)

open Windtrap
module P = Rig_nv_nvidia
module S = Rig_nv_nvidia_support

(* A failed open at the limit of [k] more files: the process's open files before
   it, after it, and after a second open at the same limit. *)
type failed = { k : int; before : int; after : int; again : int; why : string }

let pp_failed ppf f =
  Format.fprintf ppf "{k %d; files %d -> %d -> %d; %S}" f.k f.before f.after
    f.again f.why

(* [f ()] in a forked child, its result sent back through a pipe. [f] calls no
   verb: a failure in the child would end the child's copy of the run. *)
let in_child (f : unit -> 'a) : ('a, string) result =
  let r, w = Unix.pipe ~cloexec:true () in
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

let open_ () =
  match P.open_ 0 with
  | Ok g -> Ok g
  | Error why -> Error why
  | exception e -> Error ("raised " ^ Printexc.to_string e)

(* The files an open takes in a fresh process. *)
let taken () =
  let base = S.files () in
  match open_ () with
  | Error why -> failwith why
  | Ok g ->
      let n = S.files () - base in
      Rig_nv.stop g;
      n

(* Opens at the limits of 0, 1, 2, … more files until one succeeds, each failed
   one repeated at its limit: the failures and the files the process holds after
   the open that succeeded. *)
let limited () =
  let base = S.files () in
  let rec go k failures =
    if k > 64 then (List.rev failures, None)
    else
      let limit = S.limit_for k in
      let before = S.files () in
      match S.with_limit limit open_ with
      | Ok g ->
          let n = S.files () - base in
          Rig_nv.stop g;
          (List.rev failures, Some n)
      | Error why ->
          let after = S.files () in
          let again =
            match S.with_limit limit open_ with
            | Ok g ->
                Rig_nv.stop g;
                -1
            | Error _ -> S.files ()
          in
          go (k + 1) ({ k; before; after; again; why } :: failures)
  in
  go 0 []

let files () =
  if P.count () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  S.hold_gpu ();
  let fresh = require_ok ~msg:"an open in a fresh process" (in_child taken) in
  let failures, opened =
    require_ok ~msg:"opens under limits" (in_child limited)
  in
  let opened = require_some ~msg:"an open under 64 more files" opened in
  greater int ~msg:"failed opens before the first" ~than:0
    (List.length failures);
  List.iter
    (fun f ->
      let msg = Format.asprintf "%a" pp_failed f in
      not_equal int ~msg (-1) f.again;
      equal int ~msg f.after f.again;
      equal bool ~msg:(msg ^ " is an Error") false
        (String.starts_with ~prefix:"raised " f.why))
    failures;
  equal int ~msg:"files held after the open" fresh opened

(* The last page below 2^40, inside the addresses the path reserves. *)
let page = 4096
let taken_page = (1 lsl 40) - page

(* An open with [taken_page] mapped, then one after it is unmapped: their
   results, as [Ok ()] or the error. *)
let reserved () =
  if not (S.occupy taken_page page) then failwith "the page was mapped already";
  let first = Result.map Rig_nv.stop (open_ ()) in
  S.vacate taken_page page;
  let second = Result.map Rig_nv.stop (open_ ()) in
  (first, second)

let addresses () =
  if P.count () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  S.hold_gpu ();
  let first, second = require_ok (in_child reserved) in
  ignore (require_error ~msg:"the open with an address taken" first : string);
  equal (result unit string) ~msg:"the next open" (Ok ()) second

let () =
  S.hold_gpu ();
  exit
    (run "rig_nv_nvidia files"
       [
         group ~timeout:60. "files"
           [
             test
               "a failed open under a file limit takes no file, and the open \
                that succeeds holds what a fresh one does"
               files;
             test
               "an open refused its addresses gives them back for the next one"
               addresses;
           ];
       ])
