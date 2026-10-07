(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Firmware images by digest, in directories and the cache. Every lookup in this
   process sees a cache of its own; the lookups that need another environment
   run in a child process. *)

open Windtrap
open Device_pci

(* The child: [TEST_FIRMWARE_FIND=name] prints what [find] gives for the image
   below, in the environment it was given. *)

let image = "raven firmware image\n"
let digest = "f4bd2d0ec861f4d7ce351f995c824a6b18368a3d9b0b053366c50336923ed291"

let () =
  match Sys.getenv_opt "TEST_FIRMWARE_FIND" with
  | None -> ()
  | Some name ->
      (match Firmware.find name ~sha256:digest with
      | Ok s when s = image -> print_string "the image"
      | Ok _ -> print_string "other bytes"
      | Error _ -> print_string "none");
      exit 0

(* SHA-256 *)

let test_sha256 =
  cases "sha256 is the hexadecimal digest"
    ~name:(fun (name, _, _) -> name)
    [
      ( "empty",
        "",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" );
      ( "abc",
        "abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" );
      ( "two blocks",
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" );
      ( "55 bytes, the most one block pads",
        String.make 55 'a',
        "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318" );
      ( "56 bytes, padded to two blocks",
        String.make 56 'a',
        "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a" );
      ( "63 bytes",
        String.make 63 'a',
        "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34" );
      ( "64 bytes, one whole block",
        String.make 64 'a',
        "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb" );
      ( "65 bytes",
        String.make 65 'a',
        "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0" );
      ( "a million bytes",
        String.make 1_000_000 'a',
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0" );
    ]
    (fun (_, s, d) -> equal string d (Firmware.sha256 s))

(* Files *)

let of_hex h =
  String.init
    (String.length h / 2)
    (fun i -> Char.chr (int_of_string ("0x" ^ String.sub h (2 * i) 2)))

(* The image and another one, compressed by xz and zstd. *)
let xz =
  of_hex
    "fd377a585a000004e6d6b44604c0191521011600000000000000000009e390b5010014726176656e206669726d7761726520696d6167650a00000000f2d3364f7087ea610001351576936aef1fb6f37d010000000004595a"

let zst =
  of_hex "28b52ffd2415a90000726176656e206669726d7761726520696d6167650a4a856cd1"

let other_xz =
  of_hex
    "fd377a585a000004e6d6b44604c0120e2101160000000000000000009dc166a901000d616e6f7468657220696d6167650a000000dcfa83d9586b935500012e0e009139cc1fb6f37d010000000004595a"

let other_zst = of_hex "28b52ffd0458710000616e6f7468657220696d6167650a7fd26a3b"

(* Names no system holds in /lib/firmware. *)
let name file = Printf.sprintf "raven-test-%d/%s" (Unix.getpid ()) file

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

(* The cache of this process: [RAVEN_CACHE_ROOT] names it. *)
let with_cache f =
  let root = temp_dir () in
  Unix.putenv "RAVEN_CACHE_ROOT" root;
  f (Filename.concat root "firmware")

let found = result string string

let has s sub =
  let n = String.length sub in
  let rec go i =
    i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
  in
  go 0

(* [find]'s [Error] for an image no place holds, rather than a file with another
   digest. *)
let missing why = has why "none of"

let test_dir () =
  with_cache @@ fun _ ->
  let dir = temp_dir () in
  write dir (name "plain.bin") image;
  equal ~msg:"the image" found (Ok image)
    (Firmware.find ~dir (name "plain.bin") ~sha256:digest);
  write dir (name "other.bin") "another image\n";
  match Firmware.find ~dir (name "other.bin") ~sha256:digest with
  | Error why -> contains ~msg:"names the file" ~sub:(name "other.bin") why
  | Ok _ -> fail "a file with another digest was loaded"

let test_nowhere () =
  with_cache @@ fun cache ->
  let dir = temp_dir () in
  let refused ~msg places r =
    match r with
    | Ok _ -> fail "an image nowhere was found"
    | Error why ->
        List.iter
          (fun sub -> contains ~msg ~sub why)
          ([ name "absent.bin"; digest ] @ places)
  in
  refused ~msg:"with a directory, naming the image, its digest and the places"
    [ dir; "/lib/firmware"; cache ]
    (Firmware.find ~dir (name "absent.bin") ~sha256:digest);
  refused ~msg:"without" [ "/lib/firmware"; cache ]
    (Firmware.find (name "absent.bin") ~sha256:digest)

let test_cache () =
  with_cache @@ fun cache ->
  write cache (name "cached.bin") image;
  equal ~msg:"an image of the cache" found (Ok image)
    (Firmware.find (name "cached.bin") ~sha256:digest);
  write cache (name "stale.bin") "another image\n";
  match Firmware.find (name "stale.bin") ~sha256:digest with
  | Error why -> contains ~msg:"another digest is skipped" ~sub:"none of" why
  | Ok _ -> fail "a file with another digest was loaded"

let test_dir_first () =
  with_cache @@ fun cache ->
  let dir = temp_dir () in
  write cache (name "both.bin") image;
  write dir (name "both.bin") "another image\n";
  is_error ~msg:"the directory's file, refused"
    (Firmware.find ~dir (name "both.bin") ~sha256:digest);
  equal ~msg:"the cache's, when the directory lacks it" found (Ok image)
    (Firmware.find ~dir:(temp_dir ()) (name "both.bin") ~sha256:digest)

(* Without the library, the compressed file is not read: the image is
   missing. *)
let test_compressed =
  cases "compressed files of a directory"
    ~name:(fun (file, _, _) -> file)
    [
      ("image.bin.xz", xz, Ok image);
      ("image.bin.zst", zst, Ok image);
      ("other.bin.xz", other_xz, Error ());
      ("other.bin.zst", other_zst, Error ());
    ]
    (fun (file, bytes, expected) ->
      with_cache @@ fun _ ->
      let dir = temp_dir () in
      write dir (name file) bytes;
      let plain = name (Filename.remove_extension file) in
      match (Firmware.find ~dir plain ~sha256:digest, expected) with
      | Error why, _ when missing why ->
          skip ~reason:"the system has no decompression library" ()
      | Ok s, Ok e -> equal ~msg:"decompressed" string e s
      | Error why, Error () -> contains ~msg:"names the file" ~sub:plain why
      | Ok _, _ -> fail "an image with another digest was loaded"
      | Error why, _ -> fail why)

(* The cache's location, in a child whose environment holds [env] alone. *)
let find_in env file =
  let out, w = Unix.pipe ~cloexec:true () in
  let env = Array.of_list (("TEST_FIRMWARE_FIND=" ^ name file) :: env) in
  let pid =
    Unix.create_process_env Sys.executable_name [| Sys.executable_name |] env
      Unix.stdin w Unix.stderr
  in
  Unix.close w;
  let ic = Unix.in_channel_of_descr out in
  let s = In_channel.input_all ic in
  close_in ic;
  ignore (Unix.waitpid [] pid);
  s

let test_location =
  cases "the cache is found"
    ~name:(fun (n, _) -> n)
    [
      ("under RAVEN_CACHE_ROOT", `Raven);
      ("under XDG_CACHE_HOME without RAVEN_CACHE_ROOT", `Xdg);
      ("under HOME without either", `Home);
      ("under RAVEN_CACHE_ROOT alone when both are set", `Both);
    ]
    (fun (_, where) ->
      let home = temp_dir () and xdg = temp_dir () and raven = temp_dir () in
      let file = "located.bin" in
      let home_env = "HOME=" ^ home in
      let env, at, expected =
        match where with
        | `Raven ->
            ( [ home_env; "RAVEN_CACHE_ROOT=" ^ raven ],
              raven ^ "/firmware",
              "the image" )
        | `Xdg ->
            ( [ home_env; "XDG_CACHE_HOME=" ^ xdg ],
              xdg ^ "/raven/firmware",
              "the image" )
        | `Home -> ([ home_env ], home ^ "/.cache/raven/firmware", "the image")
        | `Both ->
            ( [ home_env; "XDG_CACHE_HOME=" ^ xdg; "RAVEN_CACHE_ROOT=" ^ raven ],
              xdg ^ "/raven/firmware",
              "none" )
      in
      write at (name file) image;
      equal string expected (find_in env file))

let test_writes_nothing () =
  with_cache @@ fun cache ->
  is_error ~msg:"an image nowhere"
    (Firmware.find (name "absent.bin") ~sha256:digest);
  equal ~msg:"no cache made" bool false (Sys.file_exists cache)

let () =
  exit
  @@ run "device_pci Firmware"
       [
         test_sha256;
         group "find"
           [
             test "a directory's image, refused with another digest" test_dir;
             test "an image nowhere is refused, naming where it looked"
               test_nowhere;
             test "the cache's image, skipped with another digest" test_cache;
             test "the directory comes before the cache" test_dir_first;
             test_compressed;
             test_location;
             test "a lookup writes nothing" test_writes_nothing;
           ];
       ]
