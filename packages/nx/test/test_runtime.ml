(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Values placed on runtime devices, over test runtimes whose memory is host
   memory. A runtime device holds a value's elements as stored, gives each
   operation the host's result placed where its operand is, and raises
   Out_of_memory with itself when its runtime cannot allocate. *)

open Windtrap
open Nx_test
open Stored

type bytes_ba =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(* A runtime over host memory. *)
let runtime ?(budget = max_int) name =
  let memory : (nativeint, bytes_ba) Hashtbl.t = Hashtbl.create 16 in
  let alloc n =
    let ba =
      Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n
    in
    let a = Nx_device.Buffer.host_address (Nx_device.Buffer.of_bigarray ba) in
    Hashtbl.add memory a ba;
    Some { Nx_device.host = Some a; device = a; handle = a }
  in
  let free (m : Nx_device.memory) = Hashtbl.remove memory m.device in
  Nx_device.make ~name ~arch:"test" ~budget ~memory:{ alloc; free } ()

let r1 = runtime "R1"
let r2 = runtime "R2"
let d1 = Nx.Device.of_runtime r1
let d2 = Nx.Device.of_runtime r2
let host x = Nx.place Nx.Placement.host x

(* A value of [shape] on one runtime device, copied on both, or split in two
   along an axis that halves evenly. *)
let placements shape =
  let halves =
    List.filter
      (fun a -> shape.(a) mod 2 = 0)
      (List.init (Array.length shape) Fun.id)
  in
  Gen.of_list ~pp:Nx.Placement.pp
    ([
       Nx.Placement.device d1;
       Nx.Placement.device d2;
       Nx.Placement.replicated [ d1; d2 ];
     ]
    @ List.map (fun axis -> Nx.Placement.sharded ~axis [ d1; d2 ]) halves)

let placed tensors =
  Gen.with_pp
    (fun ppf (t, p) -> Format.fprintf ppf "%a at %a" Nx.pp t Nx.Placement.pp p)
    (Gen.bind tensors (fun t ->
         Gen.map (fun p -> (t, p)) (placements (Nx.shape t))))

(* The bytes each runtime received while [f] ran, and its result. *)
let received f =
  let before = List.map Nx_device.stats [ r1; r2 ] in
  let y = f () in
  let bytes_in r s = Nx_device.Stats.(bytes_in (diff s (Nx_device.stats r))) in
  (y, List.map2 bytes_in [ r1; r2 ] before)

(* The bytes of the window of [x] that [d] holds at [p], if any. *)
let window_bytes p x d =
  if List.exists (Nx.Device.equal d) (Nx.Placement.devices p) then
    Nx.nbytes (Nx.copy (Nx.shrink (Nx.Placement.window p (Nx.shape x) d) x))
  else 0

let round_trips =
  group "placing"
    (List.map
       (fun (Case c) ->
         prop
           (c.name
          ^ " values read back bit for bit, each device receiving its window")
           (placed c.tensors) (fun (x, p) ->
             cover "split" (List.length (Nx.Placement.devices p) = 2);
             let y, bytes = received (fun () -> Nx.place p x) in
             equal Devices.placement p (Nx.placement y);
             equal ~msg:"bytes received" (list int)
               (List.map (window_bytes p x) [ d1; d2 ])
               bytes;
             equal packed (Nx.P x) (Nx.P (host y))))
       every)

(* Operations whose result, at a placement on one device or copied, lies where
   their operand does. *)
let operations =
  [
    ("neg", Nx.neg);
    ("x + x", fun x -> Nx.add x x);
    ("exp", Nx.exp);
    ("sum", fun x -> Nx.sum x);
    ("transpose", fun x -> Nx.transpose x);
    ("flip", fun x -> Nx.flip x);
    ("flatten", fun x -> Nx.flatten x);
  ]

let computing =
  group "computing"
    [
      prop
        "an operation gives the host's result on the elements placed, placed \
         where its operand is"
        (Gen.triple
           (Gen.of_list
              ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name)
              operations)
           (float_tensors Nx.float32 ~e:8 ~m:23)
           (Gen.of_list ~pp:Nx.Placement.pp
              [ Nx.Placement.device d1; Nx.Placement.replicated [ d1; d2 ] ]))
        (fun ((_, f), x, p) ->
          let y = f (Nx.place p x) in
          equal Devices.placement p (Nx.placement y);
          equal packed (Nx.P (f (Nx.copy x))) (Nx.P (host y)));
      test
        "an allocation the runtime cannot make raises Out_of_memory with the \
         device and its bytes" (fun () ->
          let small = Nx.Device.of_runtime (runtime ~budget:16 "SMALL") in
          let on_small = Nx.Placement.device small in
          let out_of_memory n = function
            | Nx.Device.Out_of_memory (d, m) -> Nx.Device.equal d small && m = n
            | _ -> false
          in
          raises_match (out_of_memory 400) (fun () ->
              Nx.place on_small (Nx.zeros Nx.float32 [| 100 |]));
          let x = Nx.place on_small (Nx.zeros Nx.float32 [| 4 |]) in
          raises_match (out_of_memory 16) (fun () -> Nx.add x x));
    ]

let devices =
  group "devices"
    [
      test "each runtime is one device, named as it, and the host's is the host"
        (fun () ->
          is_true (Nx.Device.of_runtime Nx_device.host == Nx.Device.host);
          is_true (Nx.Device.of_runtime r1 == d1);
          equal string "R1" (Nx.Device.name d1);
          is_false (Nx.Device.equal d1 d2));
      test "a placement mixing runtime devices with the host is refused"
        (fun () ->
          raises_invalid_arg (fun () ->
              Nx.Placement.replicated [ d1; Nx.Device.host ]));
    ]

let () = exit (run "nx runtime devices" [ devices; round_trips; computing ])
