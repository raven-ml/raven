(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What a driver and a compiler do with NVIDIA's formats, on the last kernel of
   a cubin of 128 kernels.

   [qmd/launch] describes one kernel's launch (its rules, descriptor, sizes,
   addresses, local memory and release), as tolk does per kernel at link, and
   [launch] its parts. [qmd/encode] fills that descriptor's holes with integers,
   as a launch would without a compiler. [method] rows encode what a driver's
   ring writer sends per submission, a release and one placement (a 16-byte copy
   and the copy engine's release); [release-template] is the release with its
   value left, as the writer takes it once. [gpfifo] encodes one ring entry.
   [fill-into-bytes] rows copy the encoder's template into bytes allocated
   beforehand and write its holes, as a writer filling a template does: the
   floor of the encoder above them. [cubin/elf-of-string] reads the ELF object
   alone: the floor of [cubin/of-string]. *)

module Abi = Rig_nv_abi
module Packet = Abi.Packet
module Method = Abi.Method
module Qmd = Abi.Qmd

let gpu ~compute_class ~sass_version =
  {
    Abi.Gpu.compute_class;
    sass_version;
    gpcs = 12;
    tpcs_per_gpc = 6;
    sms_per_tpc = 2;
    warps_per_sm = 48;
    shared_window = 0x7294_0000_0000;
    local_window = 0x7293_0000_0000;
    local = (fun _ -> Ok ());
  }

(* An RTX 5000 Ada and an RTX 5090. *)
let ada = gpu ~compute_class:0xc9c0 ~sass_version:0x89
let blackwell = gpu ~compute_class:0xcec0 ~sass_version:0xa4

(* The suite's fixture, made as test/nv/abi/fixtures/README.md says. *)

let cubin_bytes =
  In_channel.with_open_bin "../../../test/nv/abi/fixtures/many_sm89.cubin"
    In_channel.input_all

let of_string obj () =
  match Abi.Cubin.of_string obj with Ok c -> c | Error e -> failwith e

let cubin = of_string cubin_bytes ()

(* The last kernel, behind every other one. *)
let name = List.nth (Abi.Cubin.kernels cubin) 127
let kernel = Option.get (Abi.Cubin.kernel cubin name)

(* Every address the GPU reaches is below 2^40. *)
let program = 0x10_0000_0000 + kernel.code
let bank0 = 0x20_0000_0000
let signal = 0x30_0000_0040
let local_per_thread = 0x400
let launch g = Result.get_ok (Abi.Launch.make g kernel)

(* Launch descriptors *)

(* A launch whose grid is left for each launch, as tolk's are. *)
let describe g =
  let q =
    Qmd.make (launch g)
    |> Qmd.set_dim (Block X) 256 |> Qmd.set_dim (Block Y) 1
    |> Qmd.set_dim (Block Z) 1
    |> Qmd.patch_dim (Grid X) 4096
    |> Qmd.patch_dim (Grid Y) 1 |> Qmd.patch_dim (Grid Z) 1
    |> Qmd.set_program program |> Qmd.set_bank 0 bank0
    |> Qmd.set_local_memory local_per_thread
  in
  Qmd.structure (Option.get (Qmd.release System signal 1 q))

let encode_structure s = Abi.Structure.encode Int64.of_int s
let encode p = Rig_packet.encode Int64.of_int p

(* The floor of an encoder whose result is [filled]: a writer that holds
   [template], [filled] with its holes zero, copies it into bytes allocated
   beforehand and writes each hole's word, as a driver fills a template. A hole
   is the offset and width of its word; the floor writes a 64-bit word as two
   32-bit ones, from values read beforehand, and checks it writes [filled]. *)
let fill_into_bytes ~template ~filled holes =
  let split (at, width) =
    if width = 8 then [ (at, 4); (at + 4, 4) ] else [ (at, width) ]
  in
  let words = Array.of_list (List.concat_map split holes) in
  let at = Array.map fst words and width = Array.map snd words in
  let value i =
    match width.(i) with
    | 1 -> String.get_uint8 filled at.(i)
    | 2 -> String.get_uint16_le filled at.(i)
    | _ -> Int32.to_int (String.get_int32_le filled at.(i))
  in
  let values = Array.init (Array.length words) value in
  let n = String.length filled in
  let b = Bytes.create n in
  let fill () =
    Bytes.blit_string template 0 b 0 n;
    for i = 0 to Array.length at - 1 do
      let at = Array.unsafe_get at i and v = Array.unsafe_get values i in
      match Array.unsafe_get width i with
      | 1 -> Bytes.set_uint8 b at v
      | 2 -> Bytes.set_uint16_le b at v
      | _ -> Bytes.set_int32_le b at (Int32.of_int v)
    done;
    b
  in
  if Bytes.to_string (fill ()) <> filled then
    failwith "a floor writes other bytes";
  fill

(* The floor of [Rig_packet.encode value p], from [p]'s template with the values
   [known] leaves. *)
let packet_floor ?(known = fun _ -> None) value p =
  let template, holes = Rig_packet.template known p in
  let width = function Packet.Dword _ | W32 _ -> 4 | W64 _ -> 8 in
  fill_into_bytes ~template
    ~filled:(Rig_packet.encode value p)
    (List.map (fun (i, w) -> (4 * i, width w)) holes)

(* The floor of [encode_structure s]. *)
let structure_floor (s : int Abi.Structure.t) =
  let width bits =
    if bits <= 8 then 1
    else if bits <= 16 then 2
    else if bits <= 32 then 4
    else 8
  in
  fill_into_bytes ~template:s.bytes ~filled:(encode_structure s)
    (List.map
       (fun (h : int Abi.Structure.hole) -> (h.at, width h.bits))
       s.holes)

let qmd =
  let launch label g =
    Thumper.bench ("launch/" ^ label) (fun () -> describe (Thumper.black_box g))
  in
  let s = describe ada in
  Thumper.group "qmd"
    [
      launch "ada" ada;
      launch "blackwell" blackwell;
      Thumper.bench "encode/ada" (fun () ->
          encode_structure (Thumper.black_box s));
      Thumper.bench "fill-into-bytes/ada" (structure_floor s);
    ]

let launch_group =
  let q = Abi.Qmd.make (launch ada) in
  let p = Abi.Structure.encode Fun.id (Abi.Qmd.parameters q) in
  Thumper.group "launch"
    [
      Thumper.bench "make/ada" (fun () ->
          Abi.Launch.make (Thumper.black_box ada) kernel);
      Thumper.bench "driver-parameters/ada" (fun () ->
          Abi.Qmd.parameters (Thumper.black_box q));
      Thumper.bench "driver-parameters-into-bytes/ada"
        (fill_into_bytes ~template:p ~filled:p []);
    ]

(* Methods and ring entries *)

(* The ring writer's values: the timeline word it knows, the value it releases
   on each submission. *)
type v = Word | V

let value = function Word -> Int64.of_int signal | V -> 0x1234L
let known = function Word -> Some (value Word) | V -> None

(* One placement: a 16-byte copy, then the copy engine's release. *)
let placement () =
  Method.copy ~dst:0x50_0000_0000 ~src:0x60_0000_0000 16
  @ Method.copy_release System signal 0x1234

let methods =
  Thumper.group "method"
    [
      Thumper.bench "release-encode" (fun () ->
          Rig_packet.encode value
            (Method.release System (Thumper.black_box Word) V));
      Thumper.bench "release-template" (fun () ->
          Rig_packet.template known
            (Method.release System (Thumper.black_box Word) V));
      Thumper.bench "release-fill-into-bytes"
        (packet_floor ~known value (Method.release System Word V));
      Thumper.bench "placement-encode" (fun () ->
          encode (Thumper.black_box placement ()));
      Thumper.bench "placement-fill-into-bytes"
        (packet_floor Int64.of_int (placement ()));
    ]

let gpfifo =
  let words = 0x40 in
  Thumper.group "gpfifo"
    [
      Thumper.bench "entry-encode" (fun () ->
          encode
            (Abi.Gpfifo.entry 0 ~offset:0x100 ~words:(Thumper.black_box words)));
      Thumper.bench "entry-fill-into-bytes"
        (packet_floor Int64.of_int (Abi.Gpfifo.entry 0 ~offset:0x100 ~words));
    ]

(* Cubins *)

let cubin_group =
  let base = 0x10_0000_0000 in
  Thumper.group "cubin"
    [
      Thumper.bench "of-string/nv-128-kernels" (of_string cubin_bytes);
      Thumper.bench "elf-of-string/nv-128-kernels" (fun () ->
          Rig_elf.of_string ~align:128 cubin_bytes);
      Thumper.bench "kernel/last-of-128" (fun () ->
          Abi.Cubin.kernel (Thumper.black_box cubin) name);
      Thumper.bench "patches/128-relocations" (fun () ->
          Abi.Cubin.patches (Thumper.black_box cubin) ~base);
    ]

let () =
  exit
    (Thumper.run "rig_nv_abi"
       [ qmd; launch_group; methods; gpfifo; cubin_group ])
