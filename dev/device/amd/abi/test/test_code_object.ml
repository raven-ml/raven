(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Code objects of the ELF library's fixtures (elf/test/fixtures/README.md): a
   linked code object, the relocatable object it was linked from, whose code
   takes relocations a loader does not apply, and a host object. *)

open Windtrap
open Device_amd_abi

let gpu target =
  {
    Gpu.target;
    gc = target;
    sdma = (6, 0, 0);
    xccs = 1;
    shader_engines = 6;
    compute_units = 48;
    scratch_slots = 32;
  }

let fixture name =
  In_channel.with_open_bin
    ("../../../elf/test/fixtures/" ^ name)
    In_channel.input_all

let code_object name =
  match Code_object.of_string (fixture name) with
  | Ok co -> co
  | Error e -> failf "%s: %s" name e

(* The image of [co], as the module's preamble writes it. *)
let image co =
  let o = Code_object.elf co in
  let b = Bytes.make (Code_object.size co) '\000' in
  let put (s : Device_elf.section) =
    match s.offset with
    | Some off -> Bytes.blit_string o.file s.at b off s.length
    | None -> ()
  in
  Iarray.iter put o.sections;
  let patch (off, p) = Bytes.blit_string p 0 b off (String.length p) in
  List.iter patch (Code_object.patches co);
  Bytes.to_string b

let reading =
  group "reading"
    [
      test "a kernel's code is its symbol" (fun () ->
          let co = code_object "amd_gfx1100.hsaco" in
          equal (list string) [ "add" ] (Code_object.kernels co);
          let k = require_some (Code_object.kernel co "add") in
          equal (option int)
            (Device_elf.symbol (Code_object.elf co) "add")
            (Some k.entry));
      test "a kernel's descriptor is its image's bytes" (fun () ->
          let co = code_object "amd_gfx1100.hsaco" in
          let k = require_some (Code_object.kernel co "add") in
          let img = image co in
          equal int
            (String.get_int32_le img (k.descriptor + 4) |> Int32.to_int)
            k.private_segment;
          equal int
            (Int64.to_int (String.get_int64_le img (k.descriptor + 16)))
            (k.entry - k.descriptor));
      test "the image is whole 32-bit words" (fun () ->
          let co = code_object "amd_gfx1100.hsaco" in
          equal int 0 (Code_object.size co mod 4));
      test "a gfx1100 object runs on gfx1100 alone" (fun () ->
          let co = code_object "amd_gfx1100.hsaco" in
          equal string "gfx1100" (Code_object.target co);
          equal (pair bool bool) (true, false)
            ( Code_object.runs_on co (gpu (11, 0, 0)),
              Code_object.runs_on co (gpu (11, 0, 1)) ));
      test "an absent kernel is none" (fun () ->
          is_none (Code_object.kernel (code_object "amd_gfx1100.hsaco") "sub"));
    ]

let refusals =
  group "refusals"
    [
      test "a relocation of another kind than REL64 is refused" (fun () ->
          let r = Code_object.of_string (fixture "amd_gfx1100.o") in
          contains ~sub:"R_AMDGPU_REL64"
            (Result.fold ~ok:(fun _ -> "") ~error:Fun.id r));
      test "a host object is refused" (fun () ->
          is_error (Code_object.of_string (fixture "host_x86_64.o")));
      test "bytes that are no ELF object are refused" (fun () ->
          is_error (Code_object.of_string "not an object"));
    ]

let () = exit (run "device_amd_abi.code_object" [ reading; refusals ])
