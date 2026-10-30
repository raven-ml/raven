(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* NumPy formats. A value saved as npy, and an archive saved as npz, loads as
   itself, bit for bit, over every dtype npy holds, views and the values that
   break codecs; files numpy and other zip writers wrote load as they meant; an
   aligned little-endian array loads over the file's pages and any other into
   memory of its own; a file cut short or corrupted raises [Failure] naming it,
   and never anything else. *)

open Windtrap
open Io_support

let save_npy path (Nx.P x) = Nx_io.save_npy path x

(* A file the format describes in the header [dict], as numpy writes it. *)
let npy_file dict data =
  let used = 10 + String.length dict + 1 in
  let dict = dict ^ String.make ((64 - (used mod 64)) mod 64) ' ' ^ "\n" in
  "\x93NUMPY\001\000"
  ^ String.init 2 (fun i ->
      Char.chr ((String.length dict lsr (8 * i)) land 0xff))
  ^ dict ^ data

(* A float32 vector of 2^17 + 3 elements from a drawn start: its data spans many
   pages and ends inside one, where every other drawn value fits in the file's
   first page. *)
let long = (1 lsl 17) + 3

let long_case =
  Gen.map
    (fun start ->
      Case
        ( D.Float32,
          [| long |],
          Array.init long (fun i -> Float.of_int ((start + i) mod 4096) /. 8.),
          Plain ))
    (Gen.int_range 0 4095)

(* Values of every dtype npy holds, and one in ten a long vector. *)
let npy_cases =
  Gen.with_pp
    (fun ppf (Case (dt, s, _, view)) ->
      Format.fprintf ppf "%a %a %a" D.pp dt pp_shape s pp_view view)
    (Gen.frequency [ (9, case npy_dtypes); (1, long_case) ])

let npy =
  let loads name expected () =
    equal packed expected (Nx_io.load_npy (fixture name))
  in
  group "npy"
    [
      prop "a value saved as npy loads as itself, bit for bit" npy_cases
        (fun (Case (_, s, _, _) as c) ->
          covers c;
          cover "a value of 2^17 + 3 elements" (s = [| long |]);
          Law.round_trip packed string
            (fun x -> saved ".npy" save_npy x)
            Nx_io.load_npy (value c));
      cases "files numpy wrote load" ~id:fst
        [
          ( "fortran.npy",
            loads "fortran.npy"
              (Nx.P (Nx.reshape [| 2; 3 |] (Nx.arange_f Nx.float32 1. 7. 1.)))
          );
          ( "big_endian.npy",
            loads "big_endian.npy"
              (Nx.P (Nx.create Nx.int16 [| 3 |] [| 1; 256; -2 |])) );
          ( "version2.npy",
            loads "version2.npy" (Nx.P (Nx.arange Nx.int32 7 10 1)) );
          ( "version3.npy",
            loads "version3.npy"
              (Nx.P (Nx.create Nx.bool [| 3 |] [| true; false; true |])) );
        ]
        (fun (_, check) -> check ());
      cases
        "an aligned little-endian file loads over its pages, and a big-endian \
         one into memory of its own"
        ~id:fst
        [
          ("C order", ("version2.npy", Over_file));
          ("Fortran order", ("fortran.npy", Over_file));
          ("big-endian", ("big_endian.npy", Own_memory));
        ]
        (fun (_, (name, expected)) ->
          equal memory expected (memory_of (Nx_io.load_npy (fixture name))));
      test "a complex big-endian element keeps its real part first" (fun () ->
          let data =
            "\x3f\xf0\000\000\000\000\000\000\xc0\000\000\000\000\000\000\000"
          in
          let path =
            file
              (npy_file
                 "{'descr': '>c16', 'fortran_order': False, 'shape': (1,), }"
                 data)
          in
          equal packed
            (Nx.P (Nx.create Nx.complex128 [| 1 |] [| { re = 1.; im = -2. } |]))
            (Nx_io.load_npy path));
      cases "a file that is no npy file fails, naming it" ~id:fst
        [
          ("a text file", fun () -> "hello, nx\n");
          ("an npz file", fun () -> read (fixture "savez.npz"));
        ]
        (fun (_, contents) ->
          let path = file (contents ()) in
          fails ~path (fun () -> Nx_io.load_npy path));
      test "a dtype nx lacks fails, naming the file" (fun () ->
          let path = fixture "unicode.npy" in
          fails ~path (fun () -> Nx_io.load_npy path));
      test "a byte past the data, or every proper prefix, fails" (fun () ->
          let whole =
            read (saved ".npy" save_npy (Nx.P (Nx.arange Nx.int16 0 5 1)))
          in
          let path = file (whole ^ "\000") in
          fails ~path (fun () -> Nx_io.load_npy path);
          prefixes_fail Nx_io.load_npy whole);
      test "any header byte flipped loads or fails" (fun () ->
          let whole =
            read (saved ".npy" save_npy (Nx.P (Nx.arange Nx.int16 0 5 1)))
          in
          flips_load_or_fail Nx_io.load_npy ~upto:128 whole);
      cases "a malformed header fails, naming the file" ~id:fst
        [
          ( "a field twice",
            "{'descr': '<i2', 'descr': '<i2', 'fortran_order': False, 'shape': \
             (1,), }" );
          ( "an unknown field",
            "{'descr': '<i2', 'fortran_order': False, 'shape': (1,), 'x': 1, }"
          );
          ("no shape", "{'descr': '<i2', 'fortran_order': False, }");
          ( "a negative extent",
            "{'descr': '<i2', 'fortran_order': False, 'shape': (-1,), }" );
          ( "a byte-order mark on a wide dtype",
            "{'descr': '|i2', 'fortran_order': False, 'shape': (1,), }" );
          ( "33 axes",
            "{'descr': '<i2', 'fortran_order': False, 'shape': ("
            ^ String.concat ", " (List.init 33 (fun _ -> "1"))
            ^ "), }" );
          ( "an extent past 2^53",
            "{'descr': '<i2', 'fortran_order': False, 'shape': (0, \
             9007199254740993), }" );
        ]
        (fun (_, dict) ->
          let path = file (npy_file dict "\000\000") in
          fails ~path (fun () -> Nx_io.load_npy path));
      cases "a dtype npy lacks is refused, naming the save"
        ~id:(fun (D.Any dt) -> D.name dt)
        D.
          [
            Any Bfloat16;
            Any Float8_e4m3fn;
            Any Float8_e5m2;
            Any Float4_e2m1fn;
            Any Int4;
            Any Uint4;
            Any Bit;
          ]
        (fun (D.Any dt) ->
          invalid ~by:"Nx_io.save_npy" (fun () ->
              Nx_io.save_npy (temp_file ()) (Nx.zeros dt [| 2 |])));
      test "a bit value's refusal names the cast to bool" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Nx.cast Nx.bool") (fun () ->
              Nx_io.save_npy (temp_file ()) (Nx.zeros Nx.bit [| 8 |])));
    ]

(* npz *)

let save_npz path l = Nx_io.save_npz path (Archive.of_list l)

(* [xorshift n] is gen/generate.py's noise: the low byte of each of [n] states
   of Marsaglia's xorshift32 from 2463534242. *)
let xorshift n =
  let mask = 0xffff_ffff and x = ref 2463534242 in
  Array.init n (fun _ ->
      x := !x lxor ((!x lsl 13) land mask);
      x := !x lxor (!x lsr 17);
      x := !x lxor ((!x lsl 5) land mask);
      !x land 0xff)

(* [without_zip64_end s] is the archive [s] without its ZIP64 end record and
   locator, as Python's zipfile, and so numpy, writes an archive of 65535
   entries: the classic end record's 16-bit count is then the count itself. *)
let without_zip64_end s =
  let eocd = String.length s - 22 in
  if String.sub s (eocd - 20) 4 <> "PK\006\007" then s
  else
    let record = Int64.to_int (String.get_int64_le s (eocd - 12)) in
    String.sub s 0 record ^ String.sub s eocd 22

let npz =
  let one = Nx.P (Nx.zeros Nx.int8 [| 1 |]) in
  group "npz"
    [
      prop "an archive saved as npz loads as itself"
        (Gen.with_pp pp_entries (archives npy_dtypes))
        (fun entries ->
          List.iter (fun (_, c) -> covers c) entries;
          cover "an empty archive" (entries = []);
          cover "several entries" (List.length entries >= 2);
          Law.round_trip archive string
            (saved ".npz" Nx_io.save_npz)
            Nx_io.load_npz (archive_of entries));
      cases "archives numpy wrote load, deflated or stored" ~id:fst
        [
          ( "archive.npz",
            [
              ("a", Nx.P (Nx.create Nx.float32 [| 1; 2 |] [| 1.5; -2. |]));
              ("b", Nx.P (Nx.arange Nx.int64 3 6 1));
            ] );
          ( "savez.npz",
            [
              ("a", Nx.P (Nx.create Nx.uint16 [| 3 |] [| 1; 2; 65535 |]));
              ("b", Nx.P (Nx.create Nx.bool [| 1; 2 |] [| true; false |]));
            ] );
        ]
        (fun (name, entries) ->
          equal archive (Archive.of_list entries)
            (Nx_io.load_npz (fixture name)));
      test
        "an archive of 65535 entries, the most a 16-bit count holds, loads as \
         itself, with or without a ZIP64 end record" (fun () ->
          let a =
            Archive.of_list (List.init 65535 (fun i -> (string_of_int i, one)))
          in
          let whole = read (saved ".npz" Nx_io.save_npz a) in
          equal archive a (Nx_io.load_npz (file whole));
          equal archive a (Nx_io.load_npz (file (without_zip64_end whole))));
      cases
        "entries of other names than an npy file's are ignored, whatever they \
         hold"
        ~id:fst
        [
          ( "a text file",
            ( "other_names.npz",
              ("w", Nx.P (Nx.create Nx.float64 [| 2 |] [| 1.5; -2. |])) ) );
          ( "a directory and a file compressed with bzip2",
            ( "skipped_entries.npz",
              ("weights/w", Nx.P (Nx.create Nx.int16 [| 2 |] [| 3; -4 |])) ) );
        ]
        (fun (_, (name, entry)) ->
          equal archive
            (Archive.of_list [ entry ])
            (Nx_io.load_npz (fixture name)));
      test "an entry named .npy alone names no array, and fails naming it"
        (fun () ->
          let path = fixture "suffix_only.npz" in
          fails ~path ~naming:{|".npy"|} (fun () -> Nx_io.load_npz path));
      cases "a file that is no npz file fails, naming it" ~id:fst
        [
          ("a text file", fun () -> "hello, nx\n");
          ("an npy file", fun () -> read (fixture "version2.npy"));
        ]
        (fun (_, contents) ->
          let path = file (contents ()) in
          fails ~path (fun () -> Nx_io.load_npz path));
      test "an entry numpy deflated with a stored block inside loads" (fun () ->
          let path = fixture "stored_block.npz" in
          equal ~msg:"deflated" int 8 (String.get_uint16_le (read path) 8);
          let w = Array.append (Array.make 1024 0) (xorshift 20_000) in
          equal archive
            (Archive.of_list
               [ ("w", Nx.P (Nx.create Nx.uint8 [| Array.length w |] w)) ])
            (Nx_io.load_npz path));
      (* Noise over two of the encoder's blocks of 65535 bytes, which it stores,
         after a run of zeros, which it codes. *)
      prop ~count:20
        "an entry of a run of zeros then noise, saved as npz, loads as itself"
        (Gen.triple
           (Gen.int_range 65_536 100_000)
           (Gen.int_range 131_070 160_000)
           Gen.nat)
        (fun (run, noise, seed) ->
          let random = Random.State.make [| seed |] in
          let t =
            Nx.init Nx.uint8
              [| run + noise |]
              (fun i -> if i.(0) < run then 0 else Random.State.int random 256)
          in
          let path = saved ".npz" save_npz [ ("w", Nx.P t) ] in
          equal ~msg:"deflated" int 8 (String.get_uint16_le (read path) 8);
          equal packed (Nx.P t) (find (Nx_io.load_npz path) "w"));
      test
        "a deflated entry whose data starts at no multiple of its element size \
         loads" (fun () ->
          equal packed
            (Nx.P (Nx.create Nx.float64 [| 5 |] [| 0.; 1.5; 3.; 4.5; 6. |]))
            (find (Nx_io.load_npz (fixture "unaligned.npz")) "w"));
      cases "an invalid name is refused" ~id:Fun.id
        [ "/w"; "a//b"; "./w"; "a/../b"; "\xff"; "a\\b"; "c:w" ] (fun name ->
          invalid ~by:"Nx_io.save_npz" (fun () ->
              save_npz (temp_file ()) [ (name, one) ]));
      test "a dtype npy lacks is refused, naming the entry" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"Nx_io.save_npz: w: npy has no bfloat16")
            (fun () ->
              save_npz (temp_file ())
                [ ("w", Nx.P (Nx.zeros Nx.bfloat16 [| 1 |])) ]));
      (let s = Random.State.make [| 0x4e58 |] in
       let random _ = Random.State.int s 256 in
       cases
         "an incompressible entry is stored, over the file, and a compressible \
          one deflated, into memory of its own"
         ~id:fst
         [
           ("stored", (Nx.init Nx.uint8 [| 70_000 |] random, 0, Over_file));
           ("deflated", (Nx.zeros Nx.uint8 [| 70_000 |], 8, Own_memory));
         ]
         (fun (name, (t, method_, lies)) ->
           let path = saved ".npz" save_npz [ (name, Nx.P t) ] in
           (* The compression method of the archive's first entry. *)
           equal ~msg:"method" int method_ (String.get_uint16_le (read path) 8);
           let loaded = find (Nx_io.load_npz path) name in
           equal packed (Nx.P t) loaded;
           equal memory lies (memory_of loaded)));
      test "a deflated entry's donation writes in place" (fun () ->
          let n = 70_000 in
          let path =
            saved ".npz" save_npz [ ("w", Nx.P (Nx.zeros Nx.uint8 [| n |])) ]
          in
          equal ~msg:"deflated" int 8 (String.get_uint16_le (read path) 8);
          let load () = Nx.unpack Nx.uint8 (find (Nx_io.load_npz path) "w") in
          let y, grew = grown Rig.host load (fun x -> Nx.neg (Nx.donate x)) in
          less ~msg:"no memory of the result's size" int ~than:n grew;
          equal (array int) (Array.make n 0) (Nx.to_array y));
      test "an entry whose checksum does not match fails" (fun () ->
          let s = read (saved ".npz" save_npz [ ("w", one) ]) in
          (* The CRC-32 of the central directory's first entry. *)
          let rec crc i =
            if String.sub s i 4 = "PK\001\002" then i + 16 else crc (i + 1)
          in
          let path = file (flip_byte s (crc 0)) in
          fails ~path (fun () -> Nx_io.load_npz path));
      test
        "a deflated entry that declares more data than deflate can expand to \
         fails" (fun () ->
          let s = read (fixture "unaligned.npz") in
          let rec find signature i =
            if String.sub s i 4 = signature then i else find signature (i + 1)
          in
          (* The uncompressed size of the local and the central header. *)
          let b = Bytes.of_string s in
          Bytes.set_int32_le b (find "PK\003\004" 0 + 22) 0x40000000l;
          Bytes.set_int32_le b (find "PK\001\002" 0 + 24) 0x40000000l;
          let path = file (Bytes.to_string b) in
          fails ~path ~naming:"declares more data" (fun () ->
              Nx_io.load_npz path));
      test "every proper prefix fails, and any byte flipped loads or fails"
        (fun () ->
          let whole =
            read
              (saved ".npz" save_npz
                 [ ("a", Nx.P (Nx.arange Nx.int16 0 5 1)); ("b", one) ])
          in
          prefixes_fail Nx_io.load_npz whole;
          flips_load_or_fail Nx_io.load_npz whole);
    ]

let () = exit (run "nx.io NumPy" [ npy; npz ])
