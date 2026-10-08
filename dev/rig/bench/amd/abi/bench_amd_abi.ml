(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What a driver and a compiler do with AMD's formats, on one kernel of a code
   object of 128 kernels.

   [pm4/dispatch] describes one kernel's run (its registers, DISPATCH_DIRECT,
   the run's acquire and flush), as a compiler does once per kernel. [encode]
   and [template] interpret that run with every value known and with the
   launch's values left as holes. [register/find] finds eleven COMPUTE registers
   by name, as a caller that programs them does. [fill-into-bytes] rows copy a
   template into bytes allocated beforehand and write its holes, as a driver
   does per launch: the floor of the encoders above them. *)

open Rig_amd_abi
module S = Rig_amd_abi_support

(* An MI300X, a Radeon PRO W7900 and a Radeon AI PRO R9700. *)
let gfx9 =
  S.gpu ~target:(9, 4, 2) ~sdma:(4, 4, 2) ~xccs:8 ~shader_engines:4
    ~compute_units:38 (9, 4, 3)

let gfx11 =
  S.gpu ~target:(11, 0, 0) ~sdma:(6, 0, 0) ~xccs:1 ~shader_engines:6
    ~compute_units:96 (11, 0, 0)

let gfx12 =
  S.gpu ~target:(12, 0, 1) ~sdma:(7, 0, 1) ~xccs:1 ~shader_engines:4
    ~compute_units:64 (12, 0, 1)

(* The suite's fixture, made as test/amd/abi/fixtures/README.md says. *)

let code_object =
  In_channel.with_open_bin "../../../test/amd/abi/fixtures/many_gfx1201.hsaco"
    In_channel.input_all

let of_string obj () =
  match Code_object.of_string obj with Ok co -> co | Error e -> failwith e

(* The last kernel, behind every other one. *)
let co = of_string code_object ()
let name = List.nth (Code_object.kernels co) 127
let kernel = Option.get (Code_object.kernel co name)

(* GFX9's compilers have kernels read their scratch's descriptor. *)
let kernel_gfx9 = { kernel with private_segment_buffer = true }

(* Runs *)

(* The values of a run, a release and an AQL packet: an eager launch knows the
   program and the scratch, and leaves the arguments, the dispatch packet, the
   grid and the signal for each launch. *)
type arg = Known of int | Args | Dispatch_packet | Groups of int | Signal

let known = function
  | Known n -> Some (Int64.of_int n)
  | Args | Dispatch_packet | Groups _ | Signal -> None

let program = 0x7f00_0000_0100
let scratch = 0x7f10_0000_0000
let groups = [| 4096; 1; 1 |]

let value a =
  Int64.of_int
    (match a with
    | Known n -> n
    | Args -> 0x7f20_0000_0000
    | Dispatch_packet -> 0x7f20_0000_1000
    | Groups i -> groups.(i)
    | Signal -> 0x1234)

let run g k =
  Pm4.run g
    (Pm4.dispatch g k ~program:(Known program) ~scratch:(Known scratch)
       ~args:Args ~packet:Dispatch_packet
       ~threads:(Known 256, Known 1, Known 1)
       ~groups:(Groups 0, Groups 1, Groups 2)
       ())

let encode p = Rig_packet.encode value p

(* [p]'s template and its holes' words, as (word index, word) pairs, and a
   function copying the template into [b] and writing the words over it. *)
let fill_into_bytes p =
  let t, holes = Rig_packet.template known p in
  let words =
    List.concat_map
      (fun (i, w) ->
        let s = encode [ w ] in
        List.init
          (String.length s / 4)
          (fun j -> (i + j, Int32.to_int (String.get_int32_le s (4 * j)))))
      holes
  in
  let index = Array.of_list (List.map fst words) in
  let word = Array.of_list (List.map snd words) in
  let b = Bytes.create (String.length t) in
  fun () ->
    Bytes.blit_string t 0 b 0 (String.length t);
    for i = 0 to Array.length index - 1 do
      Bytes.set_int32_le b
        (4 * Array.unsafe_get index i)
        (Int32.of_int (Array.unsafe_get word i))
    done;
    b

let release g =
  Pm4.release_mem g System ~interrupt:0 (Known 0x7f30_0000_0040)
    (Data_64 Signal)

let pm4 =
  let dispatch label g k =
    Thumper.bench ("dispatch/" ^ label) (fun () ->
        run (Thumper.black_box g) (Thumper.black_box k))
  in
  Thumper.group "pm4"
    [
      dispatch "gfx9" gfx9 kernel_gfx9;
      dispatch "gfx11" gfx11 kernel;
      dispatch "gfx12" gfx12 kernel;
      Thumper.bench "encode/gfx12" (fun () ->
          encode (run (Thumper.black_box gfx12) kernel));
      Thumper.bench "template/gfx12" (fun () ->
          Rig_packet.template known (run (Thumper.black_box gfx12) kernel));
      Thumper.bench "fill-into-bytes/gfx12" (fill_into_bytes (run gfx12 kernel));
      Thumper.bench "release-encode/gfx12" (fun () ->
          encode (release (Thumper.black_box gfx12)));
      Thumper.bench "release-fill-into-bytes/gfx12"
        (fill_into_bytes (release gfx12));
    ]

(* Registers *)

(* The COMPUTE registers a dispatch sets. *)
let dispatch_registers =
  Array.map
    (fun n -> "regCOMPUTE_" ^ n)
    [|
      "PGM_LO";
      "PGM_RSRC1";
      "PGM_RSRC3";
      "TMPRING_SIZE";
      "DISPATCH_SCRATCH_BASE_LO";
      "RESTART_X";
      "USER_DATA_0";
      "RESOURCE_LIMITS";
      "START_X";
      "DISPATCH_INITIATOR";
      "TMPRING_SIZE";
    |]

let register =
  let find label g =
    Thumper.bench ("find/compute/" ^ label) (fun () ->
        let g = Thumper.black_box g in
        Array.iter
          (fun n -> ignore (Sys.opaque_identity (Register.find g n)))
          dispatch_registers)
  in
  Thumper.group "register"
    [ find "gfx9" gfx9; find "gfx11" gfx11; find "gfx12" gfx12 ]

(* AQL *)

let aql_dispatch descriptor =
  Aql.dispatch kernel ~descriptor ~args:Args ~threads:(256, 1, 1)
    ~grid:(Groups 0, Groups 1, Groups 2)

let aql =
  let descriptor = Known (program + kernel.descriptor) in
  Thumper.group "aql"
    [
      Thumper.bench "dispatch" (fun () ->
          aql_dispatch (Thumper.black_box descriptor));
      Thumper.bench "encode" (fun () ->
          encode (aql_dispatch (Thumper.black_box descriptor)));
      Thumper.bench "fill-into-bytes"
        (fill_into_bytes (aql_dispatch descriptor));
    ]

(* Code objects and scratch *)

let code =
  Thumper.group "code-object"
    [
      Thumper.bench "of-string/amd-128-kernels" (of_string code_object);
      Thumper.bench "kernel/last-of-128" (fun () ->
          Code_object.kernel (Thumper.black_box co) name);
    ]

let scratch =
  Thumper.group "scratch"
    [
      Thumper.bench "descriptor/gfx12" (fun () ->
          Scratch.descriptor (Thumper.black_box gfx12) ~base:scratch
            (Scratch.size gfx12 kernel.private_segment));
    ]

let () =
  exit (Thumper.run "rig_amd_abi" [ pm4; register; aql; code; scratch ])
