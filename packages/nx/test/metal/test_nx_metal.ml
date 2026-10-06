(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Metal GPUs as nx devices, on real hardware: what Nx_metal opens is the Mac's
   GPU with no eager backend. Tests needing a GPU skip without one. *)

open Windtrap

let device = Testable.make ~pp:Nx.Device.pp ~equal:Nx.Device.equal

let gpu () =
  match Nx_metal.get 0 with Ok d -> d | Error e -> skip ~reason:e ()

let opening =
  group "opening"
    [
      test "device 0 is the Mac's GPU, one device" (fun () ->
          let d = gpu () in
          equal device d (Nx_metal.device 0);
          equal string "METAL" (Nx.Device.name d);
          not_equal device Nx.Device.host d);
      test
        "a GPU past the Mac's one is an Error, and device raises it as Failure"
        (fun () ->
          let why = require_error (Nx_metal.get 1) in
          raises (Failure why) (fun () -> ignore (Nx_metal.device 1)));
      test "a negative index raises Invalid_argument" (fun () ->
          raises_match Exn.invalid_arg (fun () -> ignore (Nx_metal.get (-1)));
          raises_match Exn.invalid_arg (fun () -> ignore (Nx_metal.device (-1))));
    ]

let computing =
  group "computing"
    [
      test "values placed on it read back" (fun () ->
          let p = Nx.Placement.on (gpu ()) in
          let x = Nx.create Nx.float32 [| 3 |] [| 1.; -0.; 3.5 |] in
          let y = Nx.place p x in
          equal bool true (Nx.Placement.equal p (Nx.placement y));
          equal (array float_exact) (Nx.to_array x) (Nx.to_array y));
      test "a bitcast of bytes placed on it is the host's, a view of them"
        (fun () ->
          let bytes =
            Nx.init Nx.uint8 [| 4; 6 |] (fun i ->
                ((i.(0) * 61) + (i.(1) * 7)) land 255)
          in
          let x = Nx.place (Nx.Placement.on (gpu ())) bytes in
          let storage t =
            match Nx.Repr.v t with
            | Nx.Repr.Placed r -> Nx.Repr.Placed.storage r
            | _ -> failf "a value that is not placed"
          in
          let codes = Nx.bitcast Nx.uint4 x in
          let back = Nx.bitcast Nx.uint8 codes in
          equal (array int)
            (Nx.to_array (Nx.bitcast Nx.uint4 bytes))
            (Nx.to_array codes);
          equal (array int) (Nx.to_array bytes) (Nx.to_array back);
          equal ~msg:"one storage" (pair bool bool) (true, true)
            (storage codes == storage x, storage back == storage x));
      test "an eager operation raises, naming the remedies" (fun () ->
          let x =
            Nx.place (Nx.Placement.on (gpu ())) (Nx.ones Nx.float32 [| 2 |])
          in
          raises_match (Exn.invalid_arg ~substring:"has no eager kernels")
            (fun () -> ignore (Nx.exp x)));
    ]

let () = exit (run "nx.metal" [ opening; computing ])
