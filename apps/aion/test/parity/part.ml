(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What every part shares: its inputs, under AION_PARITY as the README's Tests
   section lays them out, and its verdict, the count of its failing checks. *)

let printf = Printf.printf

let dir =
  lazy
    (match Sys.getenv_opt "AION_PARITY" with
    | Some d when d <> "" -> d
    | _ ->
        prerr_endline
          "parity: AION_PARITY is unset: set it to the directory of the \
           README's Tests section";
        exit 1)

let path p = Filename.concat (Lazy.force dir) p
let load p = Nx_io.load_safetensors (path p)

let get dt name a =
  match Nx_io.Archive.find name a with
  | Some p -> Nx.unpack dt p
  | None -> failwith ("no tensor " ^ name)

let host x = Nx.place Nx.Host.on x
let floats t = Nx.to_array (Nx.cast Nx.float64 (host t))
let ints t = Array.map Int64.to_int (Nx.to_array (Nx.cast Nx.int64 (host t)))

let take rows t =
  let p =
    Nx.create Nx.int64 [| Array.length rows |] (Array.map Int64.of_int rows)
  in
  Nx.take ~axis:0 p t

(* Verdict *)

let failures = ref 0
let fail () = incr failures

let report name (c : Spread.check) =
  if c.failures <> [] then fail ();
  printf "  %s%s\n%!"
    (if c.failures = [] then "" else "FAIL ")
    (Format.asprintf "%a" (Spread.pp name) c)

let finish () =
  printf "\nfailing checks: %d\n" !failures;
  exit (if !failures = 0 then 0 else 1)
