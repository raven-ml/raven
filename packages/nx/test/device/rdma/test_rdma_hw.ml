(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Copies between two machines over their adapters: this machine's adapter 0 and
   the adapter 0 of the machine NX_RDMA_TEST names, whose server's key is in the
   file NX_RDMA_TEST_KEY. The adapters' own buffers are memory they describe, so
   the copies between them cross the fabric. *)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar

let machines =
  lazy
    (match
       (Sys.getenv_opt "NX_RDMA_TEST", Sys.getenv_opt "NX_RDMA_TEST_KEY")
     with
    | Some target, Some key_file -> (
        let host, port =
          match String.rindex_opt target ':' with
          | Some i ->
              ( String.sub target 0 i,
                int_of_string
                  (String.sub target (i + 1) (String.length target - i - 1)) )
          | None -> (target, Nx_remote_device.default_port)
        in
        let key = In_channel.with_open_bin key_file In_channel.input_all in
        match Nx_remote_device.connect ~port ~key host with
        | Error why -> Error why
        | Ok far -> (
            match (Nx_rdma_device.get 0, Nx_rdma_device.get ~host:far 0) with
            | Ok here, Ok there -> Ok (here, there)
            | Error why, _ | _, Error why -> Error why))
    | _ -> Error "NX_RDMA_TEST and NX_RDMA_TEST_KEY name no other machine")

let adapters () =
  match Lazy.force machines with Ok p -> p | Error why -> skip ~reason:why ()

let pattern n =
  String.init n (fun i -> Char.chr (((i * 31) + (i lsr 10)) land 0xff))

let fill b s =
  B.copy
    ~src:
      (B.of_bigarray
         (Bigarray.Array1.init Bigarray.char Bigarray.c_layout (String.length s)
            (String.get s)))
    ~dst:b

let read b =
  let ba =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (B.nbytes b)
  in
  B.copy ~src:b ~dst:(B.of_bigarray ba);
  String.init (Bigarray.Array1.dim ba) (Bigarray.Array1.get ba)

let test_names () =
  let here, there = adapters () in
  equal ~msg:"this machine's" string "RDMA" (Nx_device.name here);
  is_true ~msg:"the other's"
    (String.starts_with ~prefix:"RDMA@" (Nx_device.name there))

let test_copies () =
  let here, there = adapters () in
  List.iter
    (fun n ->
      let a = B.create here S.UInt8 n and b = B.create there S.UInt8 n in
      let bytes = pattern n in
      fill a bytes;
      B.copy ~src:a ~dst:b;
      equal
        ~msg:(Printf.sprintf "%d bytes there" n)
        bool true
        (String.equal bytes (read b));
      let c = B.create here S.UInt8 n in
      B.copy ~src:b ~dst:c;
      equal
        ~msg:(Printf.sprintf "%d bytes back" n)
        bool true
        (String.equal bytes (read c)))
    [ 1; 4096; 4097; 3 lsl 20 ]

let () =
  exit
    (run "nx.rdma.device (hardware)"
       [
         test "names" test_names;
         test "copies between the machines' adapters" test_copies;
       ])
