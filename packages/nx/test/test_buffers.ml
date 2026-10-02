(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Values over runtime buffers: a value of a buffer and the buffer of a value's
   elements, for file formats, and each device's buffer and view of a value, for
   compiled calls. Every dtype, under every layout, on the host and on test
   runtimes over host memory. *)

open Windtrap
open Nx_test
module B = Nx_device.Buffer

let d1 = Devices.d1
let d2 = Devices.d2
let r1 = Nx.Device.memory d1
let placement = Devices.placement
let disk = Nx.Placement.on (Nx.Device.make Nx_device.disk)

let device =
  Testable.make
    ~pp:(fun ppf d -> Format.pp_print_string ppf (Nx_device.name d))
    ~equal:Nx_device.equal

(* Where a value is: on the host, or placed on [d1]. *)
type where = { name : string; at : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let wheres =
  Gen.of_list
    ~pp:(fun ppf w -> Format.pp_print_string ppf w.name)
    [
      { name = "on the host"; at = Fun.id };
      { name = "on a device"; at = (fun x -> Nx.place (Nx.Placement.on d1) x) };
    ]

(* Whether [x]'s elements are one run of its storage in C order that starts on a
   byte. *)
let one_run x =
  let v = view x in
  let bits = Nx_dtype.Scalar.(bitsize (of_dtype (Nx.dtype x))) in
  Nx.numel x > 0
  && Nx_array.View.is_c_contiguous v
  && Nx_array.View.offset v * bits mod 8 = 0

let runtime_of x =
  match Nx.Placement.devices (Nx.placement x) with
  | [ d ] -> Nx.Device.memory d
  | _ -> fail "expected a value on one device"

let traced () =
  let module N = struct
    type (_, _) Nx.Repr.node += Nothing : ('a, 'b) Nx.Repr.node
  end in
  Nx.Repr.Traced.v ~context:Nx.Placement.host Nx.Placement.host Nx.float32
    [| 2 |] N.Nothing

let escaped =
  Invalid_argument
    "a traced tensor has no bytes; it was used outside the trace that made it"

(* to_buffer *)

let round_trip (Stored.Case c) =
  prop
    (c.name
   ^ ": of_buffer reads back to_buffer's elements, on the value's device")
    (Gen.pair c.tensors wheres) (fun (t, where) ->
      let x = where.at t in
      let b = Nx.to_buffer x in
      equal ~msg:"length" int (Nx.numel x) (B.length b);
      equal ~msg:"device" device (runtime_of x) (B.device b);
      equal ~msg:"elements" Stored.packed (Nx.P t)
        (Nx.P (Nx.of_buffer (Nx.dtype x) (Nx.shape x) b)))

let own_storage (Stored.Case c) =
  prop
    (c.name
   ^ ": to_buffer is the value's storage exactly when its elements are one run \
      of it") (Gen.pair c.tensors wheres) (fun (t, where) ->
      let x = where.at t in
      cover "one run" (one_run x);
      cover "not one run" (Nx.numel x > 0 && not (one_run x));
      if Nx.numel x > 0 then
        equal ~msg:"shares the storage" bool (one_run x)
          (share_memory (Nx.to_buffer x) (storage x)))

let to_buffer =
  group "to_buffer"
    (List.map round_trip Stored.every
    @ List.map own_storage Stored.every
    @ [
        test "a value on the disk in one run is its file's bytes" (fun () ->
            let x = Runtimes.on_disk (Nx.arange Nx.int32 0 6 1) in
            let b = Nx.to_buffer x in
            equal device Nx_device.disk (B.device b);
            is_true (share_memory b (storage x)));
        test "a value on the disk in no run is copied to the host" (fun () ->
            let t = Nx.reshape [| 2; 3 |] (Nx.arange Nx.int32 0 6 1) in
            let x = Nx.transpose (Runtimes.on_disk t) in
            let b = Nx.to_buffer x in
            equal device Nx_device.host (B.device b);
            equal (array int32)
              (Nx.to_array (Nx.transpose t))
              (Nx.to_array (Nx.of_buffer Nx.int32 [| 3; 2 |] b)));
        test "a value on a device in no run is copied on that device" (fun () ->
            let t = Nx.reshape [| 2; 3 |] (Nx.arange Nx.int32 0 6 1) in
            let x = Nx.transpose (Nx.place (Nx.Placement.on d1) t) in
            let b = Nx.to_buffer x in
            equal device r1 (B.device b);
            is_false (share_memory b (storage x));
            equal (array int32)
              (Nx.to_array (Nx.transpose t))
              (Nx.to_array (Nx.of_buffer Nx.int32 [| 3; 2 |] b)));
        test "an empty value gives an empty buffer on its device" (fun () ->
            let x =
              Nx.place (Nx.Placement.on d1) (Nx.zeros Nx.int8 [| 0; 3 |])
            in
            let b = Nx.to_buffer x in
            equal int 0 (B.length b);
            equal device r1 (B.device b));
        test "refuses a value copied on several devices" (fun () ->
            let x =
              Nx.place
                (Nx.Placement.replicated [ d1; d2 ])
                (Nx.ones Nx.int8 [| 4 |])
            in
            raises_invalid_arg (fun () -> Nx.to_buffer x));
        test "refuses a value split over several devices" (fun () ->
            let x =
              Nx.place
                (Nx.Placement.sharded ~axis:0 [ d1; d2 ])
                (Nx.ones Nx.int8 [| 4 |])
            in
            raises_invalid_arg (fun () -> Nx.to_buffer x));
        test "refuses a traced value" (fun () ->
            raises
              (Invalid_argument "Nx.to_buffer: a traced value has no buffer")
              (fun () -> Nx.to_buffer (traced ())));
        test "refuses a consumed value, naming where it was consumed" (fun () ->
            let x = Nx.place (Nx.Placement.on d1) (Nx.ones Nx.int8 [| 4 |]) in
            Devices.consume (Devices.storage_of x) ~path:"0.weights";
            raises_match (Exn.invalid_arg ~substring:"0.weights") (fun () ->
                Nx.to_buffer x));
      ])

(* Read *)

let reads_exactly (Stored.Case c) =
  prop
    (c.name
   ^ ": a read is a host buffer of exactly the value's elements, the value's \
      storage when they are one run of it on the host")
    (Gen.pair c.tensors wheres) (fun (t, where) ->
      let x = where.at t in
      let b = Nx.Op.eval (Read { by = "test"; x }) in
      equal ~msg:"device" device Nx_device.host (B.device b);
      equal ~msg:"elements" Stored.packed (Nx.P t)
        (Nx.P (Nx.of_buffer (Nx.dtype x) (Nx.shape x) b));
      let on_host = Nx.Placement.equal (Nx.placement x) Nx.Placement.host in
      if on_host && Nx.numel x > 0 then
        equal ~msg:"shares the storage" bool (one_run x)
          (share_memory b (storage x)))

(* [f ()] while a consuming call holds [b]'s memory exclusively. *)
let while_consumed b f =
  B.Claim.read b;
  is_true ~msg:"exclusive" (B.Claim.try_exclusive b);
  Fun.protect
    ~finally:(fun () ->
      B.Claim.finish b;
      B.Claim.release b)
    f

let in_use = Exn.invalid_arg ~substring:"in use by a consuming call"

let reads =
  group "Read"
    (List.map reads_exactly Stored.every
    @ [
        test "a read that gathers a value refuses memory a consuming call holds"
          (fun () ->
            let x =
              Nx.transpose (Nx.reshape [| 2; 3 |] (Nx.arange Nx.int32 0 6 1))
            in
            while_consumed (storage x) (fun () ->
                raises_match in_use (fun () ->
                    Nx.Op.eval (Read { by = "test"; x }))));
      ])

(* of_buffer *)

let of_buffer =
  let ints n = Nx_array.Elements.create Nx.int32 n in
  group "of_buffer"
    [
      test "reads a buffer of the host as a host value, in place" (fun () ->
          let b = Nx.to_buffer (Nx.arange Nx.int32 0 6 1) in
          let x = Nx.of_buffer Nx.int32 [| 2; 3 |] b in
          equal placement Nx.Placement.host (Nx.placement x);
          is_true (share_memory b (storage x));
          equal (array int32) [| 0l; 1l; 2l; 3l; 4l; 5l |] (Nx.to_array x));
      test "reads a buffer of a device as a value on that device, in place"
        (fun () ->
          let b = B.create r1 Nx_dtype.Scalar.Int32 6 in
          B.copy ~src:(Nx.to_buffer (Nx.arange Nx.int32 0 6 1)) ~dst:b;
          let x = Nx.of_buffer Nx.int32 [| 3; 2 |] b in
          equal placement (Nx.Placement.on d1) (Nx.placement x);
          is_true (share_memory b (storage x));
          equal (array int32) [| 0l; 1l; 2l; 3l; 4l; 5l |] (Nx.to_array x));
      test "reads a file's bytes as a value on the disk" (fun () ->
          let x = Runtimes.on_disk (Nx.arange Nx.int32 0 6 1) in
          equal placement disk (Nx.placement x));
      test "an empty buffer is an empty value" (fun () ->
          equal (array int) [| 0; 5 |]
            (Nx.shape (Nx.of_buffer Nx.int32 [| 0; 5 |] (ints 0))));
      test "refuses a shape of another number of elements" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.of_buffer Nx.int32 [| 2; 3 |] (ints 5)));
      test "refuses a buffer of another format" (fun () ->
          raises_match
            (Exn.invalid_arg ~substring:"int32 buffer read as float32")
            (fun () -> Nx.of_buffer Nx.float32 [| 4 |] (ints 4)));
    ]

(* shards and of_shards *)

(* Tensors on [d1] alone, copied on [d1] and [d2], or split between them, and on
   the host. *)
let placed tensors =
  Gen.frequency
    [
      (3, Runtimes.placed [ d1; d2 ] tensors);
      (1, Gen.map (fun t -> (t, Nx.Placement.host)) tensors);
    ]

let shards_round_trip (Stored.Case c) =
  prop (c.name ^ ": of_shards over shards is the value, at its placement")
    (placed c.tensors) (fun (t, p) ->
      let x = Nx.place p t in
      let buffers, v = Nx.shards x in
      let y = Nx.of_shards p (Nx.dtype x) v buffers in
      equal ~msg:"placement" placement p (Nx.placement y);
      equal ~msg:"elements" Stored.packed (Nx.P t) (Nx.P y))

let one_per_device (Stored.Case c) =
  prop
    (c.name ^ ": shards is one buffer per device, each in its device's memory")
    (placed c.tensors) (fun (t, p) ->
      let buffers, _ = Nx.shards (Nx.place p t) in
      equal (list device)
        (List.map Nx.Device.memory (Nx.Placement.devices p))
        (List.map B.device buffers))

let same_handles x y =
  List.for_all2 ( == ) (fst (Nx.shards x)) (fst (Nx.shards y))

let shards =
  let x () =
    Nx.place
      (Nx.Placement.sharded ~axis:0 [ d1; d2 ])
      (Nx.reshape [| 4; 3 |] (Nx.arange Nx.int32 0 12 1))
  in
  group "shards"
    (List.map shards_round_trip Stored.every
    @ List.map one_per_device Stored.every
    @ [
        test "gives the same handles at every call" (fun () ->
            let x = x () in
            is_true (same_handles x x));
        test "gives a view the handles of the value it views" (fun () ->
            let x = x () in
            is_true (same_handles x (Nx.flip ~axes:[ 1 ] x)));
        test "a host value is its one buffer and view" (fun () ->
            let t =
              Nx.transpose (Nx.create Nx.int8 [| 2; 3 |] [| 1; 2; 3; 4; 5; 6 |])
            in
            let buffers, v = Nx.shards t in
            equal int 1 (List.length buffers);
            equal (array int) [| 1; 3 |] (Nx_array.View.strides v));
        test "a cut inside one shard gives the buffer of that shard's device"
          (fun () ->
            let x = x () in
            let row = Nx.slice [ R (3, 4) ] x in
            equal placement (Nx.Placement.on d2) (Nx.placement row);
            is_true
              (List.for_all2 ( == )
                 (fst (Nx.shards row))
                 [ List.nth (fst (Nx.shards x)) 1 ]));
        test "a split value gives each device its shard's view" (fun () ->
            let _, v = Nx.shards (x ()) in
            equal (array int) [| 2; 3 |] (Nx_array.View.shape v));
        test "refuses a traced value as every read does" (fun () ->
            raises escaped (fun () -> Nx.shards (traced ())));
      ])

let of_shards =
  let p = Nx.Placement.replicated [ d1; d2 ] in
  let view = Nx_array.View.create [| 4 |] in
  let on r ?(scalar = Nx_dtype.Scalar.Int32) n = B.create r scalar n in
  let r2 = Nx.Device.memory d2 in
  let refuses what buffers =
    test what (fun () ->
        raises_invalid_arg (fun () -> Nx.of_shards p Nx.int32 view buffers))
  in
  group "of_shards"
    [
      refuses "refuses fewer buffers than devices" [ on r1 4 ];
      refuses "refuses more buffers than devices" [ on r1 4; on r2 4; on r2 4 ];
      refuses "refuses a buffer on another device than its own"
        [ on r1 4; on r1 4 ];
      refuses "refuses buffers of another format"
        [
          on r1 ~scalar:Nx_dtype.Scalar.Float32 4;
          on r2 ~scalar:Nx_dtype.Scalar.Float32 4;
        ];
      refuses "refuses buffers of different lengths" [ on r1 4; on r2 5 ];
      refuses "refuses a view past the buffers' elements" [ on r1 3; on r2 3 ];
      test "refuses a host buffer under a view past its elements" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.of_shards Nx.Placement.host Nx.int32 view
                [ Nx_array.Elements.create Nx.int32 3 ]));
      test "reads a host buffer at the host placement as a host value"
        (fun () ->
          let b = Nx_array.Elements.create Nx.int32 4 in
          let x = Nx.of_shards Nx.Placement.host Nx.int32 view [ b ] in
          equal placement Nx.Placement.host (Nx.placement x);
          is_true (share_memory b (storage x)));
    ]

let () =
  exit (run "nx buffers" [ reads; to_buffer; of_buffer; shards; of_shards ])
