(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nothing here touches the network: each test seeds a temporary cache with the
   files a loader would otherwise download, and runs with no [curl] on [PATH]
   or a stand-in that fails. *)

open Windtrap

(* A cache root of [RAVEN_CACHE_ROOT], and the directory [datasets/name] in
   it. *)
let dataset_dir root name =
  let dir = Filename.concat (Filename.concat root "datasets") name in
  if not (Sys.file_exists (Filename.dirname dir)) then
    Sys.mkdir (Filename.dirname dir) 0o755;
  Sys.mkdir dir 0o755;
  dir

let fresh_root () =
  let root = temp_dir () in
  setenv "RAVEN_CACHE_ROOT" (Some root);
  setenv "PATH" (Some (temp_dir ()));
  root

(* MNIST *)

let rows = 2
let cols = 3
let images = 2
let image_magic = 2051
let label_magic = 2049

(* Pixel [p] in C order of (image, row, column) of the training images:
   distinct, from 0 to 255. The test images are [255 - pixel p]. *)
let pixel p = p * 255 / ((images * rows * cols) - 1)
let test_pixel p = 255 - pixel p

let int32_be n =
  String.init 4 (fun k -> Char.chr ((n lsr (8 * (3 - k))) land 0xff))

let write dir name header body =
  Out_channel.with_open_bin (Filename.concat dir name) (fun oc ->
      List.iter (fun n -> output_string oc (int32_be n)) header;
      output_string oc body)

(* Writes the four files of an MNIST-format dataset into [dir]: the images of
   each split by [pixel] and [test_pixel], the training labels counting from
   [first_label] and the test labels from [first_label + images]. *)
let seed_mnist ?(image_magic = image_magic) ?(label_magic = label_magic)
    ~first_label dir =
  let bytes f = String.init (images * rows * cols) (fun p -> Char.chr (f p)) in
  let labels first = String.init images (fun i -> Char.chr (first + i)) in
  List.iter
    (fun (split, f, first) ->
      write dir
        (split ^ "-images-idx3-ubyte")
        [ image_magic; images; rows; cols ]
        (bytes f);
      write dir
        (split ^ "-labels-idx1-ubyte")
        [ label_magic; images ] (labels first))
    [
      ("train", pixel, first_label); ("t10k", test_pixel, first_label + images);
    ]

let pixels f = Array.init (images * rows * cols) (fun p -> float_of_int (f p))

let channels_last () =
  seed_mnist ~first_label:0 (dataset_dir (fresh_root ()) "mnist");
  let train_x, train_y, test_x, test_y =
    Kaun_datasets.mnist ~normalize:false ()
  in
  List.iter
    (fun (split, f, x) ->
      equal ~msg:(split ^ " shape") (array int)
        [| images; rows; cols; 1 |]
        (Nx.shape x);
      equal
        ~msg:(split ^ " pixels in C order of (image, row, column)")
        (array float_exact) (pixels f) (Nx.to_array x))
    [ ("train", pixel, train_x); ("test", test_pixel, test_x) ];
  equal ~msg:"train labels" (array int64) [| 0L; 1L |] (Nx.to_array train_y);
  equal ~msg:"test labels" (array int64) [| 2L; 3L |] (Nx.to_array test_y)

(* Each file seeded as its download, gzip-compressed: the loader decompresses
   it. *)
let reads_gzip () =
  let dir = dataset_dir (fresh_root ()) "mnist" in
  seed_mnist ~first_label:0 dir;
  Array.iter
    (fun name ->
      let path = Filename.concat dir name in
      let data = In_channel.with_open_bin path In_channel.input_all in
      Out_channel.with_open_bin (path ^ ".gz") (fun oc ->
          output_string oc (Lz.Gzip.encode_string data));
      Sys.remove path)
    (Sys.readdir dir);
  let _, train_y, test_x, _ = Kaun_datasets.mnist ~normalize:false () in
  equal ~msg:"test pixels" (array float_exact) (pixels test_pixel)
    (Nx.to_array test_x);
  equal ~msg:"train labels" (array int64) [| 0L; 1L |] (Nx.to_array train_y)

(* The seeded bytes span 0 to 255 in both splits. *)
let normalizes () =
  seed_mnist ~first_label:0 (dataset_dir (fresh_root ()) "mnist");
  let train_x, _, test_x, _ = Kaun_datasets.mnist () in
  List.iter
    (fun (split, x) ->
      Array.iter
        (satisfies ~msg:split ~claim:"in [0, 1]" float_exact (fun v ->
             0. <= v && v <= 1.))
        (Nx.to_array x))
    [ ("train", train_x); ("test", test_x) ]

let download_error () =
  ignore (fresh_root ());
  raises_match (fun e -> Exn.failure e) (fun () -> Kaun_datasets.mnist ())

(* curl reads ~/.curlrc before its arguments unless [-q] comes first (man curl,
   [-q]); an option there, such as [-o], would act on the download. A [curl] alone on [PATH] writes its arguments to a log, one per
   line, and fails as curl does on a 404. *)
let reads_no_curlrc () =
  if Sys.win32 then skip ~reason:"the curl stand-in is a shell script" ();
  ignore (fresh_root ());
  let bin = temp_dir ~prefix:"bin" () in
  let log = Filename.concat (temp_dir ~prefix:"curl" ()) "arguments" in
  let curl = Filename.concat bin "curl" in
  Out_channel.with_open_bin curl (fun oc ->
      Printf.fprintf oc "#!/bin/sh\nprintf '%%s\\n' \"$@\" > %s\nexit 22\n"
        (Filename.quote log));
  Unix.chmod curl 0o755;
  setenv "PATH" (Some bin);
  raises_match (fun e -> Exn.failure e) (fun () -> Kaun_datasets.mnist ());
  let first =
    if Sys.file_exists log then
      In_channel.with_open_bin log In_channel.input_line
    else None
  in
  equal ~msg:"curl's first argument" (option string) (Some "-q") first

(* A [curl] alone on [PATH] that appends its arguments to [log] and writes
   "part" where it is told to; then, while [kill] exists, kills the process
   that ran it, as a Ctrl-C or a crash mid-download would, and otherwise fails
   as curl does on a 404. *)
let partial_curl ~log ~kill =
  let bin = temp_dir ~prefix:"bin" () in
  let curl = Filename.concat bin "curl" in
  Out_channel.with_open_bin curl (fun oc ->
      output_string oc
        (String.concat "\n"
           [
             "#!/bin/sh";
             Printf.sprintf "printf '%%s\\n' \"$@\" >> %s" (Filename.quote log);
             "out=";
             "while [ $# -gt 0 ]; do [ \"$1\" = -o ] && out=$2; shift; done";
             "if [ -n \"$out\" ]; then printf part > \"$out\"; else printf part; fi";
             Printf.sprintf "[ -e %s ] && kill -9 $PPID" (Filename.quote kill);
             "exit 22";
             "";
           ]));
  Unix.chmod curl 0o755;
  setenv "PATH" (Some bin)

let lines path =
  if Sys.file_exists path then
    List.filter (( <> ) "")
      (String.split_on_char '\n'
         (In_channel.with_open_bin path In_channel.input_all))
  else []

(* A download killed midway leaves nothing at its cache path, and the next
   load downloads the file again. *)
let killed_download () =
  if Sys.win32 then skip ~reason:"the curl stand-in is a shell script" ();
  let root = fresh_root () in
  let dir = temp_dir ~prefix:"curl" () in
  let log = Filename.concat dir "arguments" in
  let kill = Filename.concat dir "kill" in
  Out_channel.with_open_bin kill ignore;
  partial_curl ~log ~kill;
  (match Unix.fork () with
  | 0 ->
      (try ignore (Kaun_datasets.mnist ()) with _ -> ());
      Unix._exit 0
  | pid ->
      let ended = function
        | Unix.WEXITED n -> Printf.sprintf "exited %d" n
        | Unix.WSIGNALED n -> Printf.sprintf "killed by signal %d" n
        | Unix.WSTOPPED n -> Printf.sprintf "stopped by signal %d" n
      in
      equal ~msg:"the loader" string (ended (Unix.WSIGNALED Sys.sigkill))
        (ended (snd (Unix.waitpid [] pid))));
  let gz =
    Filename.concat
      (Filename.concat (Filename.concat root "datasets") "mnist")
      "train-images-idx3-ubyte.gz"
  in
  equal ~msg:"the cache path" bool false (Sys.file_exists gz);
  Sys.remove kill;
  let runs = List.length (List.filter (( = ) "-q") (lines log)) in
  raises_match (fun e -> Exn.failure e) (fun () -> Kaun_datasets.mnist ());
  let urls =
    List.filter
      (fun a -> String.length a > 8 && String.sub a 0 8 = "https://")
      (lines log)
  in
  equal ~msg:"the cache path after a failed download" bool false
    (Sys.file_exists gz);
  equal ~msg:"the next load's first download" (option string)
    (Some
       "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz")
    (List.nth_opt urls runs)

let selects_fashion () =
  let root = fresh_root () in
  seed_mnist ~first_label:0 (dataset_dir root "mnist");
  seed_mnist ~first_label:5 (dataset_dir root "fashion-mnist");
  let labels fashion =
    let _, train_y, _, _ = Kaun_datasets.mnist ~fashion () in
    Nx.to_array train_y
  in
  equal ~msg:"~fashion:true" (array int64) [| 5L; 6L |] (labels true);
  equal ~msg:"~fashion:false" (array int64) [| 0L; 1L |] (labels false)

let wrong_magic (_, seed) =
  seed (dataset_dir (fresh_root ()) "mnist");
  raises_match (fun e -> Exn.failure e) (fun () -> Kaun_datasets.mnist ())

(* ustar archives, by POSIX's pax, "ustar Interchange Format" *)

let record = 512

(* A member's header: [name] after [prefix], [size] in octal, [typeflag],
   magic "ustar" and version "00", and the checksum: the sum of the header's
   bytes, its own field counted as spaces. *)
let ustar_header ?(prefix = "") ?(size_field = "") ~typeflag ~size name =
  let h = Bytes.make record '\000' in
  let put off s = Bytes.blit_string s 0 h off (String.length s) in
  put 0 name;
  put 100 "0000644";
  put 124 (if size_field = "" then Printf.sprintf "%011o" size else size_field);
  put 148 "        ";
  Bytes.set h 156 typeflag;
  put 257 "ustar\000";
  put 263 "00";
  put 345 prefix;
  let sum = ref 0 in
  Bytes.iter (fun c -> sum := !sum + Char.code c) h;
  put 148 (Printf.sprintf "%06o\000 " !sum);
  Bytes.to_string h

(* A member: its header, then [data] padded with zeros to a whole record. *)
let member ?prefix ?size_field ?(typeflag = '0') name data =
  let n = String.length data in
  ustar_header ?prefix ?size_field ~typeflag ~size:n name
  ^ data
  ^ String.make ((record - (n mod record)) mod record) '\000'

(* Two zero records end an archive. *)
let ustar members = String.concat "" members ^ String.make (2 * record) '\000'

let gzip_file tar =
  let path = Filename.concat (temp_dir ()) "a.tar.gz" in
  Out_channel.with_open_bin path (fun oc ->
      output_string oc (Lz.Gzip.encode_string tar));
  path

(* The regular files [read_tar_gz] gives, each read in pieces of [piece] bytes
   up to [limit] bytes: name, size and the bytes taken. *)
let read_back ?(limit = max_int) ~piece path =
  let got = ref [] in
  Kaun_datasets.read_tar_gz path (fun name size take ->
      let b = Buffer.create size in
      let rec go () =
        let want = min piece (limit - Buffer.length b) in
        if want > 0 then
          match take want with
          | "" -> ()
          | s ->
              Buffer.add_string b s;
              go ()
      in
      go ();
      got := (name, size, Buffer.contents b) :: !got);
  List.rev !got

(* A member drawn: a regular file of [size] bytes, or a directory or a
   symbolic link, which the reader skips. *)
type drawn = File of int | Dir | Link

let pp_drawn ppf = function
  | File n -> Format.fprintf ppf "file of %d bytes" n
  | Dir -> Format.pp_print_string ppf "directory"
  | Link -> Format.pp_print_string ppf "link"

(* Sizes at the bounds of a record, of a read, and of lz's 256 KiB run. *)
let size =
  Gen.frequency
    [
      (3, Gen.of_list [ 0; 1; 511; 512; 513; 1024; 65535; 65536; 65537 ]);
      (3, Gen.int_range 0 4096);
      (1, Gen.int_range 200_000 300_000);
    ]

let drawn =
  Gen.frequency
    [
      (8, Gen.map (fun n -> File n) size);
      (1, Gen.constant Dir);
      (1, Gen.constant Link);
    ]

(* The bytes of file [i] of [n] bytes, distinct from file to file. *)
let data i n = String.init n (fun k -> Char.chr (((k * 7) + (i * 13)) land 255))

let round_trip =
  let gen =
    Gen.triple
      (Gen.list ~size:(Gen.int_range 0 5) (Gen.with_pp pp_drawn drawn))
      (Gen.of_list [ 1; 511; 512; 513; 70_000 ])
      (Gen.of_list [ max_int; 0; 100; 512 ])
  in
  prop "gives each regular file's bytes, in order, up to what f takes" gen
    (fun (members, piece, limit) ->
      let files =
        List.concat
          (List.mapi
             (fun i -> function
               | File n -> [ (Printf.sprintf "d/f%d" i, data i n) ]
               | Dir | Link -> [])
             members)
      in
      let tar =
        ustar
          (List.mapi
             (fun i -> function
               | File n -> member (Printf.sprintf "d/f%d" i) (data i n)
               | Dir -> member ~typeflag:'5' (Printf.sprintf "d%d/" i) ""
               | Link -> member ~typeflag:'2' (Printf.sprintf "l%d" i) "")
             members)
      in
      cover "a file a multiple of 512 bytes long"
        (List.exists
           (fun (_, d) -> d <> "" && String.length d mod 512 = 0)
           files);
      cover "a file past 64 KiB"
        (List.exists (fun (_, d) -> String.length d > 65536) files);
      cover "an archive past lz's 256 KiB run" (String.length tar > 262_144);
      cover "a file f stops short of"
        (List.exists (fun (_, d) -> String.length d > limit) files);
      cover "a skipped member"
        (List.exists (fun m -> m = Dir || m = Link) members);
      let expected =
        List.map
          (fun (name, d) ->
            let k = min limit (String.length d) in
            (name, String.length d, String.sub d 0 k))
          files
      in
      equal (list (triple string int string)) expected
        (read_back ~limit ~piece (gzip_file tar)))

(* An archive whose end is followed by zero records past lz's 256 KiB run, as
   GNU tar pads to its blocking factor: its gzip trailer is decoded only by a
   reader that reads past the archive's end. *)
let padded members = ustar members ^ String.make (600 * record) '\000'

(* gzip's trailer: the data's CRC-32, then its length, 8 bytes. *)
let flip_crc gz =
  let b = Bytes.of_string gz in
  let i = Bytes.length b - 8 in
  Bytes.set b i (Char.chr (Char.code (Bytes.get b i) lxor 1));
  Bytes.to_string b

let corrupt_archives =
  let tar () = ustar [ member "a" "first"; member "b" (String.make 700 'b') ] in
  [
    ("cut in a member's data", fun () -> String.sub (tar ()) 0 (3 * record));
    ( "a header whose checksum does not match",
      fun () ->
        let t = Bytes.of_string (tar ()) in
        Bytes.set t 0 'A';
        Bytes.to_string t );
    ( "a size field that is not octal",
      fun () -> ustar [ member ~size_field:"0000000009" "a" "" ] );
    ("a pax extended header", fun () -> ustar [ member ~typeflag:'x' "x" "" ]);
    ("a GNU long name", fun () -> ustar [ member ~typeflag:'L' "L" "" ]);
    ("no end", fun () -> member "a" "first");
  ]

let refuses_archive (_, tar) =
  let path = gzip_file (tar ()) in
  raises_match (Exn.failure ~sub:path) (fun () ->
      ignore (read_back ~piece:512 path))

(* The archive's every byte reached f, yet its checksum fails. *)
let refuses_crc () =
  let src = gzip_file (padded [ member "a" "first" ]) in
  let gz = flip_crc (In_channel.with_open_bin src In_channel.input_all) in
  Out_channel.with_open_bin src (fun oc -> output_string oc gz);
  raises_match (Exn.failure ~sub:"incorrect data check") (fun () ->
      ignore (read_back ~piece:512 src))

(* CIFAR-10 *)

(* An entry is one label byte followed by a 3 x 32 x 32 image. *)
let cifar10_entry = 1 + (3 * 32 * 32)
let cifar10_batch = 10_000
let batches_dir = "cifar-10-batches-bin/"

(* [seed_cifar10 tar] puts [tar], gzip-compressed, where cifar10 downloads its
   archive. *)
let seed_cifar10 tar =
  let dir = dataset_dir (fresh_root ()) "cifar10" in
  Out_channel.with_open_bin
    (Filename.concat dir "cifar-10-binary.tar.gz")
    (fun oc -> output_string oc (Lz.Gzip.encode_string tar))

(* data_batch_1.bin holds [entries] entries, read first. *)
let refuses_batch_size entries =
  seed_cifar10
    (ustar
       [
         member (batches_dir ^ "data_batch_1.bin")
           (String.make (entries * cifar10_entry) '\001');
       ]);
  raises_match (Exn.failure ~sub:"data_batch_1.bin") (fun () ->
      Kaun_datasets.cifar10 ())

let refuses_missing () =
  seed_cifar10 (ustar [ member "readme.html" "" ]);
  raises_match (Exn.failure ~sub:"data_batch_1.bin") (fun () ->
      Kaun_datasets.cifar10 ())

(* Nothing a failed load read is kept: the next load fails too. gzip's
   checksum fails as zlib words it. *)
let refuses_crc_twice () =
  let tar = padded [ member (batches_dir ^ "readme.html") "" ] in
  let dir = dataset_dir (fresh_root ()) "cifar10" in
  Out_channel.with_open_bin
    (Filename.concat dir "cifar-10-binary.tar.gz")
    (fun oc -> output_string oc (flip_crc (Lz.Gzip.encode_string tar)));
  let load () =
    raises_match (Exn.failure ~sub:"incorrect data check") (fun () ->
        Kaun_datasets.cifar10 ())
  in
  load ();
  load ()

let () =
  exit
    (run "kaun datasets"
       [
         group "mnist"
           [
             test
               "reads each split's images channels last and its labels as int64"
               channels_last;
             test "gives images in [0, 1] by default" normalizes;
             test "decompresses each split's gzip files" reads_gzip;
             test "reads Fashion-MNIST when fashion is true" selects_fashion;
             test "raises Failure on a download error" download_error;
             test "runs curl without the user's configuration" reads_no_curlrc;
             test
               "leaves nothing at the cache path of a download killed \
                midway, and downloads it again"
               killed_download;
             cases "raises Failure on a file of the wrong magic number" ~id:fst
               [
                 ( "images",
                   fun dir ->
                     seed_mnist ~image_magic:label_magic ~first_label:0 dir );
                 ( "labels",
                   fun dir ->
                     seed_mnist ~label_magic:image_magic ~first_label:0 dir );
               ]
               wrong_magic;
           ];
         group "read_tar_gz"
           [
             round_trip;
             cases "raises Failure naming the archive" ~id:fst corrupt_archives
               refuses_archive;
             test "raises Failure when gzip's checksum fails" refuses_crc;
           ];
         group "cifar10"
           [
             cases
               "raises Failure naming a batch file of another size than 10,000 \
                entries"
               ~id:(Printf.sprintf "%d entries")
               [ 0; 1; cifar10_batch - 1; cifar10_batch + 1 ]
               refuses_batch_size;
             test "raises Failure naming a batch the archive lacks"
               refuses_missing;
             test "raises Failure on every load of a corrupt archive"
               refuses_crc_twice;
           ];
       ])
