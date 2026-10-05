(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* AMD GPUs as nx devices through the kernel driver, on real hardware: what
   Nx_amd opens is the runtime's GPU computed by nx.amd's backend, and only a
   call of [get_pci] or [device_pci] reaches PCI. The backend copies and casts
   every served dtype bit for bit as the host does, under every layout and for
   every value of the dtypes of 8 and 16 bits, from two domains at once, and
   refuses the rest naming the remedies. Tests needing a GPU skip without
   one. *)

open Windtrap
open Nx_test

let device = Testable.make ~pp:Nx.Device.pp ~equal:Nx.Device.equal
let memory = Testable.make ~pp:Nx_device.pp ~equal:Nx_device.equal
let gpu () = match Nx_amd.get 0 with Ok d -> d | Error e -> skip ~reason:e ()

let opening =
  group "opening"
    [
      test "device i is the runtime's GPU i of the kernel driver" (fun () ->
          let d = gpu () in
          let m = Result.get_ok (Nx_amd_device.get ~interface:Kernel 0) in
          equal device d (Nx_amd.device 0);
          equal memory m (Nx.Device.memory d);
          equal string (Nx_device.name m) (Nx.Device.name d);
          not_equal device Nx.Device.host d);
      test "get fails without touching PCI where the kernel driver is absent"
        (fun () ->
          if Sys.file_exists "/dev/kfd" then skip ~reason:"/dev/kfd exists" ();
          let why = require_error (Nx_amd.get 0) in
          raises (Failure why) (fun () -> ignore (Nx_amd.device 0)));
      test "the changes to the machine are the runtime's" (fun () ->
          let n = Nx_amd_device.count () in
          List.iter
            (fun (call, f, g) ->
              equal ~msg:call (result unit string) (g n) (f n))
            [
              ("detach", Nx_amd.detach, Nx_amd_device.detach);
              ("attach", Nx_amd.attach, Nx_amd_device.attach);
              ("reset", Nx_amd.reset, Nx_amd_device.reset);
              ( "fetch_firmware",
                Nx_amd.fetch_firmware,
                Nx_amd_device.fetch_firmware );
            ]);
      test "a negative index raises Invalid_argument" (fun () ->
          List.iter
            (fun f -> raises_match Exn.invalid_arg (fun () -> ignore (f (-1))))
            [ Nx_amd.get; Nx_amd.get_pci ];
          List.iter
            (fun f -> raises_match Exn.invalid_arg (fun () -> ignore (f (-1))))
            [ Nx_amd.device; Nx_amd.device_pci ]);
    ]

(* Computing *)

let on_gpu x = Nx.place (Nx.Placement.on (gpu ())) x
let host x = Nx.place Nx.Placement.host x

(* [f x] on the GPU, read back, and on the host, bit for bit. *)
let same f x =
  equal Stored.packed (f (Nx.P x))
    ((fun (Nx.P y) -> Nx.P (host y)) (f (Nx.P (on_gpu x))))

type dtype = Dtype : ('a, 'b) Nx.dtype -> dtype

let pp_dtype ppf (Dtype d) = Nx_dtype.pp ppf d

(* The dtypes the backend serves. *)
let served =
  [
    Dtype Nx.bool;
    Dtype Nx.int8;
    Dtype Nx.uint8;
    Dtype Nx.int16;
    Dtype Nx.uint16;
    Dtype Nx.int32;
    Dtype Nx.uint32;
    Dtype Nx.int64;
    Dtype Nx.uint64;
    Dtype Nx.float16;
    Dtype Nx.bfloat16;
    Dtype Nx.float32;
    Dtype Nx.float64;
    Dtype Nx.float8_e4m3;
    Dtype Nx.float8_e5m2;
  ]

let cast (Dtype d) (Nx.P x) = Nx.P (Nx.cast d x)
let copy (Nx.P x) = Nx.P (Nx.copy x)

let served_case (Stored.Case c) =
  List.exists (fun (Dtype d) -> Nx_dtype.to_string d = c.name) served

let conformance =
  group "conformance"
    (List.concat_map
       (fun (Stored.Case c) ->
         [
           prop (c.name ^ " values of every layout copy as on the host")
             c.tensors (fun x ->
               cover "strided" (not (Nx.is_c_contiguous x));
               cover "contiguous" (Nx.is_c_contiguous x && Nx.numel x > 1);
               same copy x);
           prop
             (c.name ^ " values of every layout cast as on the host")
             (Gen.pair c.tensors (Gen.of_list ~pp:pp_dtype served))
             (fun (x, d) ->
               cover "strided" (not (Nx.is_c_contiguous x));
               same (cast d) x);
         ])
       (List.filter served_case Stored.every))

(* Every value of each dtype of 8 or 16 bits, from the integers of its width. *)
let every_value (Dtype d) =
  match Nx_dtype.itemsize d with
  | 1 ->
      Nx.P (Nx.bitcast d (Nx.create Nx.uint8 [| 256 |] (Array.init 256 Fun.id)))
  | _ ->
      Nx.P
        (Nx.bitcast d
           (Nx.create Nx.uint16 [| 65536 |] (Array.init 65536 Fun.id)))

let placed f (Nx.P x) = (fun (Nx.P y) -> Nx.P (host y)) (f (Nx.P (on_gpu x)))

let sweep =
  let narrow =
    List.filter
      (fun (Dtype d) ->
        Nx_dtype.itemsize d <= 2 && Nx_dtype.to_string d <> "bool")
      served
  in
  group "every value"
    (List.map
       (fun (Dtype s as src) ->
         test
           (Printf.sprintf "every %s cast to each dtype as on the host"
              (Nx_dtype.to_string s))
           (fun () ->
             let x = every_value src in
             List.iter
               (fun d ->
                 let msg = Format.asprintf "to %a" pp_dtype d in
                 equal ~msg Stored.packed (cast d x) (placed (cast d) x))
               served))
       narrow)

(* A value of each served dtype: 64 elements of bytes drawn once, read as it;
   for bool, the bytes' low bits, which bool holds as they are. *)
let sample (Dtype d) =
  let k = Nx_dtype.itemsize d in
  let bytes =
    Nx.create Nx.uint8 [| 64; k |]
      (Array.init (64 * k) (fun i -> i * 37 mod 256))
  in
  match d with
  | Bool ->
      Nx.P
        (Nx.cast Nx.bool
           (Nx.reshape [| 64 |] (Nx.bitwise_and bytes (Nx.ones_like bytes))))
  | _ when k = 1 -> Nx.P (Nx.bitcast d (Nx.reshape [| 64 |] bytes))
  | _ -> Nx.P (Nx.bitcast d bytes)

(* The host's casts of the samples, computed once: a program on two domains
   replays the reference along each order of its calls. *)
let host_casts =
  lazy
    (List.map
       (fun s -> (s, List.map (fun d -> (d, cast d (sample s))) served))
       served)

let domains =
  let dtypes = Gen.of_list ~pp:pp_dtype served in
  let commands =
    [
      command "cast"
        (dtypes @-> dtypes @-> returns Stored.packed)
        (fun s d -> List.assq d (List.assq s (Lazy.force host_casts)))
        (fun s d -> placed (cast d) (sample s));
    ]
  in
  (* First in the run, before any other test loads a key: each program runs 50
     times, and its casts load fresh keys from both domains at once. *)
  group "domains"
    [
      stateful ~tags:[ "slow" ] ~count:10 ~domains:2
        "casts from two domains at once" commands;
      stateful "casts from one domain" commands;
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
      test "the results of a cast are on the GPU" (fun () ->
          let p = Nx.Placement.on (gpu ()) in
          let y = Nx.cast Nx.int32 (Nx.place p (Nx.ones Nx.float32 [| 4 |])) in
          equal bool true (Nx.Placement.equal p (Nx.placement y)));
      test "an operation it has no kernel for raises, naming the remedies"
        (fun () ->
          let x = on_gpu (Nx.ones Nx.float32 [| 2 |]) in
          raises_match
            (Exn.invalid_arg
               ~substring:"no unary kernels. Compile it with Rune.jit")
            (fun () -> ignore (Nx.exp x)));
      test "a dtype it does not serve raises, naming it" (fun () ->
          let x = on_gpu (Nx.ones Nx.complex64 [| 2 |]) in
          raises_match (Exn.invalid_arg ~substring:"no complex dtypes")
            (fun () -> ignore (Nx.cast Nx.float32 x)));
      test "a value over a bare GPU buffer computes once placed on the device"
        (fun () ->
          let d = gpu () in
          let b =
            Nx_device.Buffer.create (Nx.Device.memory d) Nx_dtype.Scalar.Float32
              4
          in
          let x = Nx.of_buffer Nx.float32 [| 4 |] b in
          raises_match (Exn.invalid_arg ~substring:"has no eager kernels")
            (fun () -> ignore (Nx.cast Nx.int32 x));
          let y = Nx.place (Nx.Placement.on d) x in
          equal (array int) [| 4 |] (Nx.shape (Nx.cast Nx.int32 y)));
      test "a value of no element casts to one of no element" (fun () ->
          let y = Nx.cast Nx.int8 (on_gpu (Nx.zeros Nx.float32 [| 0; 3 |])) in
          equal (array int) [| 0; 3 |] (Nx.shape y));
    ]

let () = exit (run "nx.amd" [ domains; opening; computing; conformance; sweep ])
