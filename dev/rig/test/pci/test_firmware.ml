(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Firmware images by digest, in the directories a lookup is given. *)

open Windtrap
open Rig_pci
open Rig_pci_support

let image = "raven firmware image\n"
let pinned = "ce1c62cec35ab52e7ceef75a48b8cf0743b671f581f066288ec274636805997a"

(* Digests *)

(* BLAKE2b with 32 bytes of output, as Python's hashlib computes it. *)
let test_digest =
  cases "digest is the hexadecimal BLAKE2b-256 digest"
    ~name:(fun (name, _, _) -> name)
    [
      ( "empty",
        "",
        "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8" );
      ( "abc",
        "abc",
        "bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319" );
      ("the image", image, pinned);
      ( "a million bytes",
        String.make 1_000_000 'a',
        "0741850f36cba4259628355d1073e24ddb9ca0e1bfac36fd39ae5dc2101e23a4" );
    ]
    (fun (_, s, d) -> equal string d (Firmware.digest s))

(* Files *)

let rec mkdir_p d =
  if not (Sys.file_exists d) then begin
    mkdir_p (Filename.dirname d);
    Sys.mkdir d 0o755
  end

let write dir file s =
  let path = Filename.concat dir file in
  mkdir_p (Filename.dirname path);
  Out_channel.with_open_bin path (fun oc -> output_string oc s)

let rec remove path =
  if Sys.is_directory path then begin
    Array.iter (fun f -> remove (Filename.concat path f)) (Sys.readdir path);
    Sys.rmdir path
  end
  else Sys.remove path

(* Every file the suite writes is under one directory beside it in _build,
   cleared at the start of a run, since a killed run leaves it, and removed at
   exit. *)
let root =
  lazy
    (let d = Filename.concat (Sys.getcwd ()) "firmware.tmp" in
     if Sys.file_exists d then remove d;
     Sys.mkdir d 0o755;
     at_exit (fun () -> if Sys.file_exists d then remove d);
     d)

let temp_dir () = Filename.temp_dir ~temp_dir:(Lazy.force root) "d" ""
let name = "amdgpu/image.bin"

let found =
  result
    (Testable.contramap
       (fun (i : Firmware.image) -> (i.path, i.contents))
       (pair string string))
    string

(* The image, found in [dir]. *)
let in_ dir = Ok { Firmware.path = Filename.concat dir name; contents = image }

let test_first () =
  let a = temp_dir () and b = temp_dir () in
  write b name image;
  equal ~msg:"from the directory that holds it" found (in_ b)
    (Firmware.find [ a; b ] name ~digest:pinned);
  write a name image;
  equal ~msg:"from the first that holds it" found (in_ a)
    (Firmware.find [ a; b ] name ~digest:pinned)

let test_other_digest () =
  let a = temp_dir () and b = temp_dir () in
  write a name "another image\n";
  write b name image;
  equal ~msg:"another digest is skipped" found (in_ b)
    (Firmware.find [ a; b ] name ~digest:pinned)

let test_missing () =
  let a = temp_dir () and b = temp_dir () in
  write a name "another image\n";
  match Firmware.find [ a; b ] name ~digest:pinned with
  | Ok _ -> fail "a file with another digest was loaded"
  | Error why ->
      List.iter
        (fun sub -> contains ~msg:"names" ~sub why)
        [ name; pinned; a; b; Filename.concat a name ]

(* A file that cannot be read is no image, and a refusal says why. *)
let test_unreadable () =
  if Unix.geteuid () = 0 then
    skip ~reason:"root reads a file whatever its mode" ();
  let a = temp_dir () and b = temp_dir () in
  write a name image;
  Unix.chmod (Filename.concat a name) 0o000;
  (match Firmware.find [ a ] name ~digest:pinned with
  | Ok _ -> fail "an unreadable file was loaded"
  | Error why ->
      contains ~msg:"names the file and its cause"
        ~sub:("reading " ^ Filename.concat a name ^ ": ")
        why);
  write b name image;
  equal ~msg:"the next directory gives it" found (in_ b)
    (Firmware.find [ a; b ] name ~digest:pinned);
  Unix.chmod (Filename.concat a name) 0o644

let test_no_directory () =
  is_error ~msg:"no directory holds anything"
    (Firmware.find [] name ~digest:pinned)

(* The files under [d] and their contents, in order. *)
let rec files d =
  Sys.readdir d |> Array.to_list |> List.sort compare
  |> List.concat_map (fun f ->
      let path = Filename.concat d f in
      if Sys.is_directory path then (path, "/") :: files path
      else [ (path, In_channel.with_open_bin path In_channel.input_all) ])

let test_writes_nothing () =
  let a = temp_dir () and b = temp_dir () in
  write a name "another image\n";
  write b name image;
  let before = files (Lazy.force root) in
  ignore
    (Firmware.find [ a; b ] name ~digest:pinned
      : (Firmware.image, string) result);
  ignore
    (Firmware.find [ a; b ] "amdgpu/missing.bin" ~digest:pinned
      : (Firmware.image, string) result);
  equal (list (pair string string)) before (files (Lazy.force root))

let test_compressed () =
  let a = temp_dir () in
  write a (name ^ ".zst") image;
  write a (name ^ ".xz") image;
  is_error ~msg:"only the file of that name"
    (Firmware.find [ a ] name ~digest:pinned)

(* Reads *)

(* An image found once is not read again while its file keeps its identity: a
   file made unreadable since, which keeps its device, inode, size and
   modification time, still gives it. *)
let test_read_once () =
  if Unix.geteuid () = 0 then
    skip ~reason:"root reads a file whatever its mode" ();
  let a = temp_dir () in
  write a name image;
  equal ~msg:"read" found (in_ a) (Firmware.find [ a ] name ~digest:pinned);
  Unix.chmod (Filename.concat a name) 0o000;
  equal ~msg:"not read again" found (in_ a)
    (Firmware.find [ a ] name ~digest:pinned);
  Unix.chmod (Filename.concat a name) 0o644

(* A file whose identity changed since a find gave it is verified again: one
   rewritten with another image is skipped, one removed is not found. *)
let test_changed () =
  let a = temp_dir () and b = temp_dir () in
  write a name image;
  write b name image;
  equal ~msg:"read" found (in_ a) (Firmware.find [ a ] name ~digest:pinned);
  write a name "another image, longer\n";
  is_error ~msg:"rewritten" (Firmware.find [ a ] name ~digest:pinned);
  equal ~msg:"skipped" found (in_ b)
    (Firmware.find [ a; b ] name ~digest:pinned);
  Sys.remove (Filename.concat b name);
  is_error ~msg:"removed" (Firmware.find [ b ] name ~digest:pinned)

(* An image found under one digest is no image of another. *)
let test_other_pin () =
  let a = temp_dir () in
  write a name image;
  equal ~msg:"read" found (in_ a) (Firmware.find [ a ] name ~digest:pinned);
  is_error ~msg:"another pinned digest"
    (Firmware.find [ a ] name ~digest:(Firmware.digest "another image\n"))

let () =
  exit
  @@ run "rig_pci.firmware"
       [
         group ~timeout:patience "digest" [ test_digest ];
         group ~timeout:patience "find"
           [
             test "the first directory holding the image gives it" test_first;
             test "a file with another digest is skipped" test_other_digest;
             test
               "an image nowhere is refused, naming the image, its digest, the \
                directories and the files with another digest"
               test_missing;
             test "an unreadable file is refused with its cause" test_unreadable;
             test "no directory holds nothing" test_no_directory;
             test "a lookup writes nothing" test_writes_nothing;
             test "a compressed file is not the image" test_compressed;
           ];
         group ~timeout:patience "reads"
           [
             test "an unchanged file is read once" test_read_once;
             test "a changed file is verified again" test_changed;
             test "an image found under one digest is no image of another"
               test_other_pin;
           ];
       ]
