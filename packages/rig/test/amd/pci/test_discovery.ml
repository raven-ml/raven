(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Discovery tables: the R9700's, and tables damaged anywhere. *)

open Windtrap
open Rig_amd_pci

let strf = Printf.sprintf

let fixture name =
  In_channel.with_open_bin ("../fixtures/" ^ name) In_channel.input_all

(* The R9700's blocks as the amdgpu driver lists them (../fixtures/r9700.txt):
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

(* A table refused for one defect names the field that holds it and the field's
   byte offset. discovery.h lays the binary out with its checksum at byte 8,
   over the bytes from 10 to its size at 10, and its tables' offsets and
   checksums from 12, 8 bytes a table: the IP table's at 12 and 14, the GC
   table's at 20 and 22, the harvest table's at 28 and 30. The IP table holds
   its signature at 0, its size at 6, its dies at 12 and the first die's offset
   at 16; a die its ID at 0. The GC table holds its ID at 0, its version at 4
   and its size at 8; the harvest table its signature at 0, over 136 bytes. *)

(* A field's byte offset, from the table's 16-bit words. *)
type place = (int -> int) -> int

let binary n : place = fun _ -> n
let ip n : place = fun u16 -> u16 12 + n
let gc_table n : place = fun u16 -> u16 20 + n
let harvest n : place = fun u16 -> u16 28 + n
let first_die : place = fun u16 -> u16 (u16 12 + 16)

(* The table [name] with each 16-bit word of [edits] written, then the IP, GC
   and harvest tables' checksums made good, then the words of [after] written,
   then the binary's checksum made good unless [binary] is [false]; and the
   table's 16-bit words before the edits. *)
let edited ?(binary = true) ?(after = []) name edits =
  let s = fixture name in
  let t = Bytes.of_string s in
  let u16 at = Bytes.get_uint16_le t at
  and was at = String.get_uint16_le s at in
  let write ((at : place), v) = Bytes.set_uint16_le t (at was) v in
  let sum at n =
    let s = ref 0 in
    for i = at to at + n - 1 do
      s := !s + Bytes.get_uint8 t i
    done;
    !s land 0xffff
  in
  let ih = was 12 and gc = was 20 and hv = was 28 in
  List.iter write edits;
  Bytes.set_uint16_le t 14 (sum ih (u16 (ih + 6)));
  Bytes.set_uint16_le t 22 (sum gc (u16 (gc + 8)));
  if hv <> 0 then Bytes.set_uint16_le t 30 (sum hv 136);
  List.iter write after;
  if binary then Bytes.set_uint16_le t 8 (sum 10 (u16 10 - 10));
  (Bytes.to_string t, was)

let refusals =
  let case ?binary ?after ?(name = "r9700.bin") label edits ~field ~(at : place)
      =
    (label, (fun () -> edited ?binary ?after name edits), field, at)
  in
  cases
    ~name:(fun (label, _, _, _) -> label)
    "a defect is refused, naming its field and its byte offset"
    [
      case "the binary's signature"
        [ (binary 0, 0) ]
        ~field:"signature" ~at:(binary 0);
      case "the binary's checksum" ~binary:false
        [ (binary 8, 0) ]
        ~field:"checksum" ~at:(binary 8);
      case "no IP table" [ (binary 12, 0) ] ~field:"IP table" ~at:(binary 12);
      case "the IP table's signature"
        [ (ip 0, 0) ]
        ~field:"signature" ~at:(ip 0);
      case "the IP table's checksum" []
        ~after:[ (binary 14, 0) ]
        ~field:"checksum" ~at:(binary 14);
      case "more dies than the IP table's header holds"
        [ (ip 12, 0xffff) ]
        ~field:"dies" ~at:(ip 12);
      case "a die whose ID is not its position"
        [ (first_die, 1) ]
        ~field:"ID" ~at:first_die;
      case "no GC table" [ (binary 20, 0) ] ~field:"GC table" ~at:(binary 20);
      case "the GC table's ID" [ (gc_table 0, 0) ] ~field:"ID" ~at:(gc_table 0);
      case "the GC table's checksum" []
        ~after:[ (binary 22, 0) ]
        ~field:"checksum" ~at:(binary 22);
      case "a GC table of version 3"
        [ (gc_table 4, 3) ]
        ~field:"version" ~at:(gc_table 4);
      case "the harvest table's signature" ~name:"r9700_fused.bin"
        [ (harvest 0, 0) ]
        ~field:"signature" ~at:(harvest 0);
      case "the harvest table's checksum" ~name:"r9700_fused.bin" []
        ~after:[ (binary 30, 0) ]
        ~field:"checksum" ~at:(binary 30);
    ]
    (fun (_, table, field, at) ->
      let t, was = table () in
      let why = require_error (Discovery.of_string t) in
      Windtrap.contains ~sub:field why;
      Windtrap.contains ~sub:(strf "at byte %d" (at was)) why)

let dies =
  group ~timeout:10. "dies"
    [
      test "a GPU of one die has die 0" (fun () ->
          equal (list int) [ 0 ] (Discovery.aids (table "r9700.bin")));
      test "a die is live with all its SDMA instances or a pair" (fun () ->
          let d = table "r9700.bin" in
          let sdma = List.init 16 (fun i -> (i, [| 0 |])) in
          let d =
            {
              d with
              bases = (42, sdma) :: List.remove_assoc 42 d.bases;
              harvested = [ (42, [ 8; 9; 12 ]) ];
            }
          in
          equal (list int) [ 0; 1; 2 ] (Discovery.aids d));
    ]

let () =
  exit (run "rig_amd_pci.discovery" [ discovery; damaged; refusals; dies ])
