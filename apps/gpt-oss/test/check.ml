(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What the validators share: fixtures read as JSON, and checks printed one a
   line and counted. *)

(* Fixtures *)

let json_of_file path =
  let text = In_channel.with_open_bin path In_channel.input_all in
  match Jsont_bytesrw.decode_string Jsont.json text with
  | Ok j -> j
  | Error e -> failwith (path ^ ": " ^ e)

let decode codec j =
  match Jsont.Json.decode codec j with
  | Ok v -> v
  | Error e -> failwith ("fixture: " ^ e)

let mem name j = decode (Jsont.mem name Jsont.json) j
let find name j = decode (Jsont.mem name ~absent:None (Jsont.some Jsont.json)) j

(* The members of an object, in the file's order. *)
let members j =
  List.rev
    (decode (Jsont.fold_object Jsont.json (fun _ n v acc -> (n, v) :: acc) []) j)

let list j = decode (Jsont.list Jsont.json) j
let number j = decode Jsont.number j
let string j = decode Jsont.string j
let string_option j = decode (Jsont.option Jsont.string) j
let bool j = decode Jsont.bool j
let floats j = decode (Jsont.array Jsont.number) j
let ints j = decode (Jsont.array Jsont.int) j

(* Checks *)

let checks = ref 0
let failures = ref []

let report name status detail =
  Printf.printf "%-60s %s%s\n%!" name status detail

let check name ok detail =
  incr checks;
  if not ok then failures := name :: !failures;
  report name (if ok then "ok" else "FAIL") detail

(* A difference the precision explains: printed, never a failure. *)
let note name ok detail = report name (if ok then "ok" else "note") detail

(* Prints the count and exits with 1 if a check failed. *)
let finish () =
  match List.rev !failures with
  | [] -> Printf.printf "%d checks passed\n" !checks
  | first :: _ as all ->
      Printf.printf "%d of %d checks failed; the first: %s\n" (List.length all)
        !checks first;
      exit 1
