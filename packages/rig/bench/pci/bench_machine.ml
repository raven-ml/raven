(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A machine's functions, as every open finds its function among them, over
   trees of files laid out as Linux's [/sys/bus/pci].

   [functions] lists a machine of 4 or 64 GPUs: the files of each function, then
   the sort into bus order. [open-stop] opens the first of two GPUs with a
   driver that starts nothing and stops it: the listing, the take of its files
   and their release. *)

module Machine = Rig_pci.Machine
module Driver = Rig_pci.Driver
open Rig_pci_support

let strf = Printf.sprintf

(* Errors *)

let err_listed why = strf "functions: %s" why
let err_open why = strf "open-stop: %s" why

(* Machines *)

let vendor = 0x1002
let counts = [ 4; 64 ]

(* [n] GPUs, one per bus, in no bus order: a listing sorts them. *)
let tree n =
  Tree.make
    (List.init n (fun i ->
         Tree.gpu (Machine.address ~domain:0 ~bus:(n - i) ~device:0 ~fn:0)))

let functions m =
  match Machine.functions m with
  | Ok ids -> ids
  | Error why -> failwith (err_listed why)

let functions_row n =
  Thumper.bench_with_setup
    ~setup:(fun () -> tree n)
    ~teardown:Tree.remove (string_of_int n)
    (fun root -> functions (Machine.at root))

(* Opening *)

let is_gpu (id : Machine.id) = id.vendor = vendor && id.class_ lsr 16 = 0x03

let gpus =
  Driver.make ~name:"AMD-PCI" ~memory_bar:(Some 0) ~siblings:Driver.Remove
    ~nodes:(fun ~root:_ _ -> [])
    ~unreleased:(fun ~root:_ _ -> None)
    ~teardown_ms:2000
    ~reset:(fun _ -> Ok ())
    ~ready:(fun _ -> Ok ())
    is_gpu

let open_stop root =
  let m = Machine.at root in
  match
    Driver.open_ gpus m 0 (fun h _ ->
        Driver.set_stop h (fun () -> `Clean);
        Ok h)
  with
  | Ok h -> Driver.stop h
  | Error why -> failwith (err_open why)

let open_row =
  Thumper.bench_with_setup
    ~setup:(fun () -> tree 2)
    ~teardown:Tree.remove "open-stop" open_stop

let () =
  exit
    (Thumper.run "rig_pci_machine"
       [ Thumper.group "functions" (List.map functions_row counts); open_row ])
