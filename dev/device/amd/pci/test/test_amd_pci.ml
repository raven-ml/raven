(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Discovery = Device_amd_pci.Discovery

let strf = Printf.sprintf

let fixture name =
  In_channel.with_open_bin ("fixtures/" ^ name) In_channel.input_all

(* The R9700's blocks as the amdgpu driver lists them (fixtures/r9700.txt):
   hardware ID, instance, version and segment bases. *)
let listing =
  fixture "r9700.txt" |> String.split_on_char '\n'
  |> List.filter (( <> ) "")
  |> List.map (fun line ->
      match String.split_on_char ' ' line with
      | hw :: inst :: version :: _harvest :: bases ->
          let version =
            match List.map int_of_string (String.split_on_char '.' version) with
            | [ a; b; c ] -> (a, b, c)
            | _ -> failwith ("a version in r9700.txt: " ^ line)
          in
          ( int_of_string hw,
            int_of_string inst,
            version,
            Array.of_list (List.map int_of_string bases) )
      | _ -> failwith ("a line of r9700.txt: " ^ line))

let table name =
  match Discovery.of_string (fixture name) with
  | Ok d -> d
  | Error why -> failf "%s: %s" name why

let version = triple int int int
let bases = array int

(* Discovery *)

let gc = 11
let sdma0 = 42
let mp0 = 255
let mp1 = 1
let umc = 150

let listed name =
  test
    (strf "%s states every block's version and bases as amdgpu lists them" name)
  @@ fun () ->
  let d = table name in
  List.iter
    (fun (hw, inst, v, b) ->
      equal
        ~msg:(strf "the bases of block %d instance %d" hw inst)
        (list (pair int bases))
        [ (inst, b) ]
        (List.filter (fun (i, _) -> i = inst) (Discovery.live d hw));
      if inst = 0 then
        equal
          ~msg:(strf "the version of block %d" hw)
          (option version) (Some v) (Discovery.version d hw))
    listing

let discovery =
  group ~timeout:10. "discovery"
    [
      cases ~name:Fun.id "place" [ "offset"; "bytes" ] (function
        | "offset" -> equal int (64 * 1024) Discovery.offset
        | _ -> equal int (10 * 1024) Discovery.bytes);
      listed "r9700.bin";
      listed "r9700_wide.bin";
      test "the R9700's blocks of a boot have their versions" (fun () ->
          let d = table "r9700.bin" in
          List.iter
            (fun (b, v) ->
              equal ~msg:(Discovery.name b) (option version) (Some v)
                (Discovery.version d b))
            [
              (gc, (12, 0, 1));
              (sdma0, (7, 0, 1));
              (mp0, (14, 0, 3));
              (mp1, (14, 0, 3));
            ]);
      test "the R9700's GC table states its shape" (fun () ->
          let g = (table "r9700.bin").gc in
          equal (list int) [ 4; 2; 8; 32; 16; 65536 ]
            Discovery.
              [ g.engines; g.arrays; g.units; g.scratch_slots; g.waves; g.lds ]);
      test "a fused instance is harvested and not live" (fun () ->
          let d = table "r9700_fused.bin" in
          equal (list (pair int (list int))) [ (umc, [ 7 ]) ] d.harvested;
          equal (list int) [ 0; 1; 2; 3; 4; 5; 6 ]
            (List.map fst (Discovery.live d umc)));
      test "a table without fused instances has none" (fun () ->
          equal (list (pair int (list int))) [] (table "r9700.bin").harvested);
      cases
        ~name:(fun (b, _) -> strf "block %d" b)
        "names"
        [
          (gc, "GC"); (sdma0, "SDMA0"); (mp0, "MP0"); (mp1, "MP1"); (0x7fff, "");
        ]
        (fun (b, n) -> equal string n (Discovery.name b));
    ]

(* A table damaged anywhere is refused or read, and reading never raises. *)

let contains ~sub s =
  let n = String.length sub in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = sub || at (i + 1))
  in
  at 0

let damaged =
  let tables = [ "r9700.bin"; "r9700_wide.bin"; "r9700_fused.bin" ] in
  let length = Discovery.bytes in
  let gen =
    let open Gen in
    let+ name = of_list tables
    and+ cut = int_range 0 length
    and+ flips =
      list ~size:(int_range 0 3)
        (pair (int_range 0 (length - 1)) (int_range 1 255))
    in
    (name, cut, flips)
  in
  prop ~timeout:30. "a damaged table is refused or read, never raised" gen
    (fun (name, cut, flips) ->
      let b = Bytes.of_string (fixture name) in
      List.iter
        (fun (i, x) -> Bytes.set_uint8 b i (Bytes.get_uint8 b i lxor x))
        flips;
      let r = Discovery.of_string (Bytes.sub_string b 0 cut) in
      let refused sub =
        match r with Error e -> contains ~sub e | Ok _ -> false
      in
      cover "read" (Result.is_ok r);
      cover "refused by a checksum" (refused "checksum");
      cover "refused by a field outside" (refused "outside"))

let () = exit (run "device_amd_pci" [ discovery; damaged ])
