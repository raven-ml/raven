(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Firmware images by digest, in the directories a lookup is given. *)

open Windtrap
open Device_pci

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

(* Every file the suite writes is under one directory, removed at exit. *)
let root =
  lazy
    (let d = Filename.temp_dir "firmware" "" in
     at_exit (fun () -> remove d);
     d)

let temp_dir () = Filename.temp_dir ~temp_dir:(Lazy.force root) "d" ""
let name = "amdgpu/image.bin"
let found = result string string

let test_first () =
  let a = temp_dir () and b = temp_dir () in
  write b name image;
  equal ~msg:"from the directory that holds it" found (Ok image)
    (Firmware.find [ a; b ] name ~digest:pinned);
  write a name image;
  equal ~msg:"from the first that holds it" found (Ok image)
    (Firmware.find [ a; b ] name ~digest:pinned)

let test_other_digest () =
  let a = temp_dir () and b = temp_dir () in
  write a name "another image\n";
  write b name image;
  equal ~msg:"another digest is skipped" found (Ok image)
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

let test_no_directory () =
  is_error ~msg:"no directory holds anything"
    (Firmware.find [] name ~digest:pinned)

let test_compressed () =
  let a = temp_dir () in
  write a (name ^ ".zst") image;
  write a (name ^ ".xz") image;
  is_error ~msg:"only the file of that name"
    (Firmware.find [ a ] name ~digest:pinned)

let () =
  exit
  @@ run "device_pci Firmware"
       [
         test_digest;
         group "find"
           [
             test "the first directory holding the image gives it" test_first;
             test "a file with another digest is skipped" test_other_digest;
             test
               "an image nowhere is refused, naming the image, its digest, the \
                directories and the files with another digest"
               test_missing;
             test "no directory holds nothing" test_no_directory;
             test "a compressed file is not the image" test_compressed;
           ];
       ]
