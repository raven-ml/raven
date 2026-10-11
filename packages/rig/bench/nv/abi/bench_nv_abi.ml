(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What a driver and a compiler do with NVIDIA's formats, on the last kernel of
   a cubin of 128 kernels.

   [qmd/launch] describes one kernel's launch (its rules, descriptor, sizes,
   addresses and local memory), as tolk does per kernel at link, and [launch]
   its parts. [qmd/encode] fills that descriptor's patches with integers, as a
   launch would without a compiler. [method] rows encode what a driver's ring
   writer sends per submission, a release and one placement (a 16-byte copy and
   the copy engine's release); [release-template] is the release with its value
   left, as the writer takes it once. [gpfifo] encodes one ring entry.
   [fill-into-bytes] rows copy the encoder's template into bytes allocated
   beforehand and write its patches, as a writer filling a template does: the
   floor of the encoder above them. [cubin/elf-of-string] reads the ELF object
   alone: the floor of [cubin/of-string]. *)

module Abi = Rig_nv_abi
module Template = Abi.Template
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

(* The suite's fixture, made by test/nv/abi/fixtures/gen.sh. *)

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

(* A launch whose grid is left for each launch, as tolk's are, and which
   releases a value when it completes. *)
let values =
  {
    Qmd.program = Int program;
    local = Int local_per_thread;
    banks = [ (0, Int bank0) ];
    grid = (Slot 0, Slot 1, Slot 2);
    block = (Int 256, Int 1, Int 1);
    shared = None;
    chain = None;
    releases = [ Release (Int signal, Int 1) ];
  }

let grid = function 0 -> 4096L | _ -> 1L
let describe g = Qmd.make (launch g) values

(* The floor of [Template.fill value t]: a writer that holds [t]'s bytes copies
   them into bytes allocated beforehand and writes each patch's word, as a
   driver fills a template. The floor writes a 64-bit word as two 32-bit ones,
   from values read beforehand, and checks it writes what [fill] does. *)
let fill_into_bytes value (t : Template.t) =
  let filled = Template.fill value t in
  let width bits =
    if bits <= 8 then 1
    else if bits <= 16 then 2
    else if bits <= 32 then 4
    else 8
  in
  let split (p : Template.patch) =
    let at = p.at and width = width p.bits in
    if width = 8 then [ (at, 4); (at + 4, 4) ] else [ (at, width) ]
  in
  let words = Array.of_list (List.concat_map split t.patches) in
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
    Bytes.blit_string t.bytes 0 b 0 n;
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

let qmd =
  let launch label g =
    Thumper.bench ("launch/" ^ label) (fun () -> describe (Thumper.black_box g))
  in
  let t = describe ada in
  Thumper.group "qmd"
    [
      launch "ada" ada;
      launch "blackwell" blackwell;
      Thumper.bench "encode/ada" (fun () ->
          Template.fill grid (Thumper.black_box t));
      Thumper.bench "fill-into-bytes/ada" (fill_into_bytes grid t);
    ]

let launch_group =
  let l = launch ada in
  let p = Qmd.parameters l values in
  Thumper.group "launch"
    [
      Thumper.bench "make/ada" (fun () ->
          Abi.Launch.make (Thumper.black_box ada) kernel);
      Thumper.bench "driver-parameters/ada" (fun () ->
          Qmd.parameters (Thumper.black_box l) values);
      Thumper.bench "driver-parameters-into-bytes/ada" (fill_into_bytes grid p);
    ]

(* Methods and ring entries *)

(* The ring writer's values: the timeline word it knows, the value it releases
   on each submission, by slot. *)
let word = Template.Int signal
let v = Template.Slot 0
let value _ = 0x1234L

(* One placement: a 16-byte copy, then the copy engine's release. *)
let placement () =
  Template.concat
    [
      Method.copy ~dst:(Int 0x50_0000_0000) ~src:(Int 0x60_0000_0000) (Int 16);
      Method.copy_release (Int signal) (Int 0x1234);
    ]

let methods =
  Thumper.group "method"
    [
      Thumper.bench "release-encode" (fun () ->
          (Method.release (Thumper.black_box word) (Int 0x1234)).bytes);
      Thumper.bench "release-template" (fun () ->
          Method.release (Thumper.black_box word) v);
      Thumper.bench "release-fill-into-bytes"
        (fill_into_bytes value (Method.release word v));
      Thumper.bench "placement-encode" (fun () ->
          (Thumper.black_box placement ()).Template.bytes);
      Thumper.bench "placement-fill-into-bytes"
        (fill_into_bytes value (placement ()));
    ]

let gpfifo =
  let words = 0x40 in
  Thumper.group "gpfifo"
    [
      Thumper.bench "entry-encode" (fun () ->
          (Abi.Gpfifo.entry (Int 0) ~offset:0x100
             ~words:(Thumper.black_box words))
            .bytes);
      Thumper.bench "entry-fill-into-bytes"
        (fill_into_bytes value (Abi.Gpfifo.entry (Slot 0) ~offset:0x100 ~words));
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
