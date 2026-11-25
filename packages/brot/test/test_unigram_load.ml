(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The first Unigram file a process loads. Reading a score's text uses a table
   of powers of ten that brot shares between domains, so two domains that load
   their first file at the same moment must both get it. This executable loads
   nothing before its one test, so the test meets brot as a program does on its
   first load. *)

open Windtrap
open Brot

let file =
  {|{"model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["a",-12.130167007446289],["b",-4.5750198364257812]]}}|}

(* What tokenizers 0.23.1 reads off the two scores. *)
let expected = [ "-0x1.842a53fffffffp+3"; "-0x1.24cd1ffffffffp+2" ]

let scores t =
  let member name = function
    | Jsont.Object (members, _) -> Jsont.Json.find_mem name members
    | _ -> None
  in
  match
    Option.bind (member "model" (to_json t)) (fun (_, m) -> member "vocab" m)
  with
  | Some (_, Jsont.Array (_ :: entries, _)) ->
      List.map
        (function
          | Jsont.Array ([ _; Jsont.Number (score, _) ], _) ->
              Printf.sprintf "%h" score
          | _ -> "bad entry")
        entries
  | _ -> [ "no vocab" ]

(* Both domains wait until both are ready, then load, so that their first reads
   of a score start together. *)
let test_two_domains () =
  let path = temp_file ~suffix:".json" () in
  Out_channel.with_open_bin path (fun oc -> Out_channel.output_string oc file);
  let ready = Atomic.make 0 in
  let load () =
    Atomic.incr ready;
    while Atomic.get ready < 2 do
      Domain.cpu_relax ()
    done;
    Result.map scores (from_file path)
  in
  let other = Domain.spawn load in
  let mine = load () in
  let theirs = Domain.join other in
  equal
    (pair (result (list string) string) (result (list string) string))
    (Ok expected, Ok expected) (mine, theirs)

let () =
  exit
    (run "from_file's first load"
       [
         group "from_file"
           [
             test
               "two domains that load their first file both read its scores as \
                tokenizers reads them"
               test_two_domains;
           ];
       ])
