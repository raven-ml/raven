(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Formats. A value saved in a format loads as itself, bit for bit, over every
   dtype the format holds, views and the values that break codecs; files other
   tools wrote load as they meant; a file cut short or corrupted raises
   [Failure] naming it, and never anything else. *)

open Windtrap
open Io_support
module Archive = Nx_io.Archive

(* An archive compares as its named values, in name order. *)
let listed a =
  List.map (fun n -> (n, Option.get (Archive.find n a))) (Archive.names a)

let archive = Testable.contramap listed (list (pair string packed))
let find a name = Option.get (Archive.find name a)

(* [over_file x] is [true] iff [x] reads memory borrowed from elsewhere, as the
   host's borrow of a file's pages is. *)
let over_file (Nx.P x) =
  match Nx.Repr.array x with
  | Some a -> Rig.Buffer.is_borrowed (Nx_array.buffer a)
  | None -> false

let cut s n = String.sub s 0 (String.length s - n)

let saved suffix save x =
  let path = temp_file ~suffix () in
  save path x;
  path

(* Durable saves *)

let saves =
  let x = Nx.arange Nx.int32 0 4 1 in
  group "saves"
    [
      test "a save leaves the file alone in its directory" (fun () ->
          let dir = temp_dir () in
          let path = Filename.concat dir "x.npy" in
          Nx_io.save_npy path x;
          Nx_io.save_npy path x;
          equal (array string) [| "x.npy" |] (Sys.readdir dir));
      test "without overwrite, a save to a path that names a file fails"
        (fun () ->
          let path = file "old" in
          fails ~path (fun () -> Nx_io.save_npy ~overwrite:false path x);
          equal string "old" (read path);
          let fresh = Filename.concat (temp_dir ()) "x.npy" in
          Nx_io.save_npy ~overwrite:false fresh x;
          equal packed (Nx.P x) (Nx_io.load_npy fresh));
      test "a save into a missing directory fails, naming the path" (fun () ->
          let path = Filename.concat (temp_dir ()) "missing/x.npy" in
          fails ~path (fun () -> Nx_io.save_npy path x));
      test "a file that cannot be opened fails, naming it" (fun () ->
          let path = Filename.concat (temp_dir ()) "absent.npy" in
          fails ~path (fun () -> Nx_io.load_npy path);
          fails ~path (fun () -> Nx_io.load_npz path);
          fails ~path (fun () -> Nx_io.load_safetensors path);
          fails ~path (fun () -> Nx_io.load_gguf path);
          fails ~path (fun () -> Nx_io.load_txt path Nx.float32));
      test
        "a refusal before the write names the save, and an error during it \
         keeps its own message" (fun () ->
          let by = "Nx_io.save_npy" in
          raises
            (Invalid_argument (by ^ ": the value is dead"))
            (fun () ->
              Fail.reading ~by
                (fun _ -> invalid_arg "Nx.Repr.read: the value is dead")
                (fun () -> ()));
          raises (Invalid_argument "codec: a bad image") (fun () ->
              Fail.reading ~by
                (fun f -> f ())
                (fun () -> invalid_arg "codec: a bad image")));
      test "a save of a dead value names the save" (fun () ->
          let x = Nx.create Nx.int32 [| 2 |] [| 1l; 2l |] in
          ignore (Nx.neg (Nx.donate x));
          invalid ~by:"Nx_io.save_npy" (fun () ->
              Nx_io.save_npy (temp_file ()) x));
      (let device = Nx_support.memory 3 in
       let module Mem = (val Nx.devices [ device ]) in
       (* Values in memory of their own: one made from OCaml data, and one
          computed on a memory device, since a placement onto it from the host
          shares the host's memory. *)
       let on_host () =
         Nx.create Nx.float32 [| 1 lsl 16 |] (Array.make (1 lsl 16) 1.)
       in
       let on_mem () =
         let x = Nx.place Mem.on (on_host ()) in
         Nx.add x (Nx.zeros_like x)
       in
       let donated d make save =
         (* The bytes [d] holds once collected memory has returned. *)
         let settled () =
           for _ = 1 to 4 do
             Gc.full_major ();
             ignore (Rig.Buffer.create d 0);
             Rig.free_cache d
           done;
           Rig.allocated d
         in
         let saved_then_donated x =
           save x;
           Nx.neg (Nx.donate x)
         in
         ignore (saved_then_donated (make ()));
         let x = make () in
         let before = settled () in
         let y = saved_then_donated x in
         less ~msg:"no memory of the result's size" int ~than:(4 lsl 16)
           (Rig.allocated d - before);
         equal (array float_exact) [| -1. |]
           (Nx.to_array (Nx.slice [ Nx.R (0, 1) ] y))
       in
       let npy x = Nx_io.save_npy (temp_file ()) x in
       let npz x =
         Nx_io.save_npz (temp_file ()) (Archive.of_list [ ("x", Nx.P x) ])
       in
       let safetensors x =
         Nx_io.save_safetensors (temp_file ())
           (Archive.of_list [ ("x", Nx.P x) ])
       in
       cases ~name:fst
         "a save leaves its value's memory to the value's next donation"
         [
           ("npy on the host", fun () -> donated Rig.host on_host npy);
           ("npz on the host", fun () -> donated Rig.host on_host npz);
           ( "safetensors on the host",
             fun () -> donated Rig.host on_host safetensors );
           ("npy on a memory device", fun () -> donated device on_mem npy);
         ]
         (fun (_, check) -> check ()));
      test "a value of another set or of every set saves" (fun () ->
          let module M = (val Nx.devices [ Nx_support.memory 0 ]) in
          let x = Nx.arange Nx.int32 0 6 1 in
          let on_m = Nx.place M.on x in
          equal packed (Nx.P x)
            (Nx_io.load_npy (saved ".npy" Nx_io.save_npy on_m));
          equal packed (Nx.P x)
            (Nx_io.load_npy
               (saved ".npy" Nx_io.save_npy (Nx.arange Nx.int32 0 6 1))));
    ]

(* NumPy *)

let npy_dtypes =
  D.
    [
      Any Float16;
      Any Float32;
      Any Float64;
      Any Int8;
      Any Int16;
      Any Int32;
      Any Int64;
      Any Uint8;
      Any Uint16;
      Any Uint32;
      Any Uint64;
      Any Complex64;
      Any Complex128;
      Any Bool;
    ]

let save_npy path (Nx.P x) = Nx_io.save_npy path x

(* A file the format describes in the header [dict], as numpy writes it. *)
let npy_file dict data =
  let used = 10 + String.length dict + 1 in
  let dict = dict ^ String.make ((64 - (used mod 64)) mod 64) ' ' ^ "\n" in
  "\x93NUMPY\001\000"
  ^ String.init 2 (fun i ->
      Char.chr ((String.length dict lsr (8 * i)) land 0xff))
  ^ dict ^ data

let npy =
  let loads name expected () =
    equal packed expected (Nx_io.load_npy (fixture name))
  in
  group "npy"
    [
      prop "a value saved as npy loads as itself, bit for bit" (case npy_dtypes)
        (fun c ->
          covers c;
          Law.round_trip packed string
            (fun x -> saved ".npy" save_npy x)
            Nx_io.load_npy (value c));
      cases ~name:fst "files numpy wrote load"
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
      test
        "an aligned little-endian file loads over its pages, and a big-endian \
         one into memory of its own" (fun () ->
          is_true ~msg:"C order"
            (over_file (Nx_io.load_npy (fixture "version2.npy")));
          is_true ~msg:"Fortran order"
            (over_file (Nx_io.load_npy (fixture "fortran.npy")));
          is_false ~msg:"big-endian"
            (over_file (Nx_io.load_npy (fixture "big_endian.npy"))));
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
      cases ~name:fst "a malformed header fails, naming the file"
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
      cases
        ~name:(fun (D.Any dt) -> D.name dt)
        "a dtype npy lacks is refused, naming the save"
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
          raises_match (Exn.invalid_arg ~substring:"Nx.cast Nx.bool") (fun () ->
              Nx_io.save_npy (temp_file ()) (Nx.zeros Nx.bit [| 8 |])));
    ]

(* npz *)

let entry_names = [ "w"; "layer.0/weight"; "a b"; "é"; "日本"; "deep/er/name" ]

(* Archives of one to three values of [dts], under distinct names. *)
let archives dts =
  let open Gen in
  let* names = subsequence entry_names in
  let names =
    match names with
    | [] -> [ "w" ]
    | n :: ns -> n :: List.filteri (fun i _ -> i < 2) ns
  in
  let+ cs = list ~size:(constant (List.length names)) (case dts) in
  List.combine names cs

let archive_of entries =
  Archive.of_list (List.map (fun (n, c) -> (n, value c)) entries)

let pp_entries ppf l =
  Format.pp_print_list
    (fun ppf (n, c) ->
      Format.fprintf ppf "%s: %a" n
        (fun ppf (Case (dt, s, _, v)) ->
          Format.fprintf ppf "%a %a %a" D.pp dt pp_shape s pp_view v)
        c)
    ppf l

let save_npz path l = Nx_io.save_npz path (Archive.of_list l)

let npz =
  let one = Nx.P (Nx.zeros Nx.int8 [| 1 |]) in
  group "npz"
    [
      prop "an archive saved as npz loads as itself, each entry alone too"
        (Gen.with_pp pp_entries (archives npy_dtypes))
        (fun entries ->
          List.iter (fun (_, c) -> covers c) entries;
          let a = archive_of entries in
          let path = saved ".npz" Nx_io.save_npz a in
          equal archive a (Nx_io.load_npz path);
          List.iter
            (fun (name, _) ->
              equal ~msg:name packed (find a name)
                (Nx_io.load_npz_entry ~name path))
            entries);
      test "an archive numpy wrote loads" (fun () ->
          equal archive
            (Archive.of_list
               [
                 ("a", Nx.P (Nx.create Nx.float32 [| 1; 2 |] [| 1.5; -2. |]));
                 ("b", Nx.P (Nx.arange Nx.int64 3 6 1));
               ])
            (Nx_io.load_npz (fixture "archive.npz")));
      test
        "a deflated entry whose data starts at no multiple of its element size \
         loads" (fun () ->
          equal packed
            (Nx.P (Nx.create Nx.float64 [| 5 |] [| 0.; 1.5; 3.; 4.5; 6. |]))
            (Nx_io.load_npz_entry ~name:"w" (fixture "unaligned.npz")));
      cases ~name:Fun.id "an invalid name is refused"
        [ "/w"; "a//b"; "./w"; "a/../b"; "\xff"; "a\\b"; "c:w" ] (fun name ->
          invalid ~by:"Nx_io.save_npz" (fun () ->
              save_npz (temp_file ()) [ (name, one) ]));
      test "a dtype npy lacks is refused, naming the entry" (fun () ->
          raises_match
            (Exn.invalid_arg ~substring:"Nx_io.save_npz: w: npy has no bfloat16")
            (fun () ->
              save_npz (temp_file ())
                [ ("w", Nx.P (Nx.zeros Nx.bfloat16 [| 1 |])) ]));
      test
        "an incompressible entry is stored, over the file, and a compressible \
         one deflated" (fun () ->
          let s = Random.State.make [| 0x4e58 |] in
          let random _ = Random.State.int s 256 in
          List.iter
            (fun (name, t, method_) ->
              let path = saved ".npz" save_npz [ (name, Nx.P t) ] in
              (* The compression method of the archive's first entry. *)
              equal ~msg:name int method_ (String.get_uint16_le (read path) 8);
              let loaded = Nx_io.load_npz_entry ~name path in
              equal packed (Nx.P t) loaded;
              if method_ = 0 then
                is_true ~msg:"over the file" (over_file loaded))
            [
              ("stored", Nx.init Nx.uint8 [| 70_000 |] random, 0);
              ("deflated", Nx.zeros Nx.uint8 [| 70_000 |], 8);
            ]);
      test "a deflated entry's donation writes in place" (fun () ->
          let n = 70_000 in
          let path =
            saved ".npz" save_npz [ ("w", Nx.P (Nx.zeros Nx.uint8 [| n |])) ]
          in
          equal ~msg:"deflated" int 8 (String.get_uint16_le (read path) 8);
          let load () =
            Nx.unpack Nx.uint8 (Nx_io.load_npz_entry ~name:"w" path)
          in
          let y, grew = host_grown load (fun x -> Nx.neg (Nx.donate x)) in
          less ~msg:"no memory of the result's size" int ~than:n grew;
          equal (array int) (Array.make n 0) (Nx.to_array y));
      test "a missing entry fails, naming the file" (fun () ->
          let path = saved ".npz" save_npz [ ("w", one) ] in
          fails ~path ~naming:"absent" (fun () ->
              Nx_io.load_npz_entry ~name:"absent" path));
      test "an entry whose checksum does not match fails" (fun () ->
          let s = read (saved ".npz" save_npz [ ("w", one) ]) in
          (* The CRC-32 of the central directory's first entry. *)
          let rec crc i =
            if String.sub s i 4 = "PK\001\002" then i + 16 else crc (i + 1)
          in
          let path = file (flip_byte s (crc 0)) in
          fails ~path (fun () -> Nx_io.load_npz_entry ~name:"w" path));
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
              Nx_io.load_npz_entry ~name:"w" path));
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

(* SafeTensors *)

let safetensors_dtypes =
  D.
    [
      Any Bool;
      Any Uint8;
      Any Int8;
      Any Float8_e5m2;
      Any Float8_e4m3fn;
      Any Int16;
      Any Uint16;
      Any Float16;
      Any Bfloat16;
      Any Int32;
      Any Uint32;
      Any Float32;
      Any Float64;
      Any Int64;
      Any Uint64;
    ]

let save_safetensors path l = Nx_io.save_safetensors path (Archive.of_list l)

let with_header header =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int (String.length header));
  Bytes.to_string b ^ header

(* [raw_safetensors ?pad entries] is a SafeTensors file holding [entries], each
   a name, a dtype tag, a shape and the entry's bytes, in order. The data
   section starts on a multiple of 8, [pad] bytes later. *)
let raw_safetensors ?(pad = 0) entries =
  let offset = ref 0 in
  let field (name, dtype, shape, data) =
    let start = !offset in
    offset := start + String.length data;
    let shape = String.concat "," (List.map string_of_int shape) in
    Printf.sprintf {|"%s":{"dtype":"%s","shape":[%s],"data_offsets":[%d,%d]}|}
      name dtype shape start !offset
  in
  let json = "{" ^ String.concat "," (List.map field entries) ^ "}" in
  let padding = ((8 - (String.length json mod 8)) mod 8) + pad in
  with_header (json ^ String.make padding ' ')
  ^ String.concat "" (List.map (fun (_, _, _, data) -> data) entries)

let pattern n seed =
  String.init n (fun i -> Char.chr (((i * 37) + seed) land 255))

(* Entries of every dtype, each with the dtype nx loads it at, in descending
   element size, so that with [pad = 0] each sits on a multiple of its element
   size and with [pad = 1] none of the wide ones does. *)
let typed_entries =
  [
    ("u64", "U64", "uint64", [ 2 ], pattern 16 1);
    ("u32", "U32", "uint32", [ 2; 2 ], pattern 16 2);
    ("bf16", "BF16", "bfloat16", [ 3 ], pattern 6 3);
    ("f16", "F16", "float16", [ 1 ], pattern 2 4);
    ("f8_e4m3", "F8_E4M3", "float8_e4m3fn", [ 3 ], pattern 3 5);
    ("f8_e5m2", "F8_E5M2", "float8_e5m2", [ 3 ], pattern 3 6);
    ("bool", "BOOL", "bool", [ 4 ], "\000\001\002\255");
    ("empty", "F32", "float32", [ 0; 3 ], "");
    ("scales", "F8_E8M0", "uint8", [ 2; 3 ], pattern 6 7);
    ("nibbles", "F4", "uint8", [ 2; 4 ], pattern 4 8);
    ("sixes", "F6_E2M3", "uint8", [ 4 ], pattern 3 9);
    ("other sixes", "F6_E3M2", "uint8", [ 8 ], pattern 6 10);
  ]

let typed_file ?pad () =
  raw_safetensors ?pad
    (List.map
       (fun (n, tag, _, shape, data) -> (n, tag, shape, data))
       typed_entries)

(* A value's dtype, shape and bytes as stored. *)
let stored (Nx.P x) =
  let a = Option.get (Nx.Repr.array (Nx.copy x)) in
  let b = Nx_array.buffer a in
  ( D.name (Nx.dtype x),
    Nx.shape x,
    String.init (Rig.Buffer.length b) (fun i ->
        let s = Bytes.create 1 in
        Rig.Buffer.blit_to_bytes b i s 0 1;
        Bytes.get s 0) )

let payload path =
  let s = read path in
  let start = 8 + Int64.to_int (String.get_int64_le s 0) in
  String.sub s start (String.length s - start)

let u8_entry name = raw_safetensors [ (name, "U8", [ 1 ], "\042") ]

let safetensors =
  group "safetensors"
    [
      prop "an archive saved as SafeTensors loads as itself"
        (Gen.with_pp pp_entries (archives safetensors_dtypes))
        (fun entries ->
          List.iter (fun (_, c) -> covers c) entries;
          let a = archive_of entries in
          equal archive a
            (Nx_io.load_safetensors
               (saved ".safetensors" Nx_io.save_safetensors a)));
      cases ~name:fst
        "files numpy and torch wrote load, and save again as they are"
        [
          ("f16", [| 0x0000; 0x0001; 0x3C00; 0x7C00; 0x7E01 |]);
          ("bf16", [| 0x0000; 0x0001; 0x3F80; 0x7F80; 0x7FC1 |]);
        ]
        (fun (dtype, bits) ->
          let path = fixture (dtype ^ "_bit_exact.safetensors") in
          let entry = dtype ^ "_tensor" in
          let (Nx.P t as p) = find (Nx_io.load_safetensors path) entry in
          equal (array int) bits (Nx.to_array (Nx.bitcast Nx.uint16 t));
          let again = temp_file () in
          save_safetensors again [ (entry, p) ];
          equal string (payload path) (payload again));
      test "an entry of any dtype loads its bytes as stored, at any offset"
        (fun () ->
          List.iter
            (fun pad ->
              let a = Nx_io.load_safetensors (file (typed_file ~pad ())) in
              List.iter
                (fun (name, tag, dtype, shape, data) ->
                  let narrow = List.mem tag [ "F4"; "F6_E2M3"; "F6_E3M2" ] in
                  let shape =
                    if narrow then [| String.length data |]
                    else Array.of_list shape
                  in
                  equal ~msg:name
                    (triple string (array int) string)
                    (dtype, shape, data)
                    (stored (find a name)))
                typed_entries)
            [ 0; 1 ]);
      test
        "an aligned entry loads over the file's pages, and a misaligned one \
         into memory of its own" (fun () ->
          let u32 pad =
            find (Nx_io.load_safetensors (file (typed_file ~pad ()))) "u32"
          in
          is_true ~msg:"aligned" (over_file (u32 0));
          is_false ~msg:"one byte past" (over_file (u32 1)));
      test
        "a misaligned entry's donation writes in place, and an aligned one's \
         leaves the file's pages" (fun () ->
          let n = 1 lsl 14 in
          (* [n] float32 ones, little-endian. *)
          let ones =
            String.concat "" (List.init n (fun _ -> "\000\000\128\063"))
          in
          let negated pad =
            let path =
              file (raw_safetensors ~pad [ ("x", "F32", [ n ], ones) ])
            in
            let load () =
              Nx.unpack Nx.float32 (find (Nx_io.load_safetensors path) "x")
            in
            let y, grew = host_grown load (fun x -> Nx.neg (Nx.donate x)) in
            equal (array float_exact) (Array.make n (-1.)) (Nx.to_array y);
            grew
          in
          less ~msg:"misaligned: no memory of the result's size" int
            ~than:(4 * n) (negated 1);
          at_least ~msg:"aligned: a fresh result" int ~than:(4 * n)
            (negated 0));
      test "a header's JSON string escapes are decoded" (fun () ->
          equal (list string)
            [ "aé🚀\"\\/\b\012\r\n\t" ]
            (Archive.names
               (Nx_io.load_safetensors
                  (file (u8_entry {|aé🚀\"\\\/\b\f\r\n\t|})))));
      test "metadata of strings is accepted and ignored" (fun () ->
          let json =
            {|{"__metadata__":{"format":"pt"},"w":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}|}
          in
          let pad = String.make ((8 - (String.length json mod 8)) mod 8) ' ' in
          equal (list string) [ "w" ]
            (Archive.names
               (Nx_io.load_safetensors
                  (file (with_header (json ^ pad) ^ "\042")))));
      test
        "a save orders entries by dtype, as the reference implementation does, \
         then by name" (fun () ->
          let path =
            saved ".safetensors" save_safetensors
              [
                ("a", Nx.P (Nx.zeros Nx.uint8 [| 1 |]));
                ("b", Nx.P (Nx.full Nx.float64 [| 1 |] 1.));
                ("c", Nx.P (Nx.full Nx.int64 [| 1 |] 2L));
                ("d", Nx.P (Nx.full Nx.float32 [| 1 |] 3.));
              ]
          in
          let p = payload path in
          equal int 21 (String.length p);
          equal string "\002\000\000\000\000\000\000\000" (String.sub p 0 8);
          equal string "\000\000\000\000\000\000\240\063" (String.sub p 8 8));
      cases
        ~name:(fun (D.Any dt) -> D.name dt)
        "a dtype SafeTensors lacks is refused, naming the entry"
        D.
          [
            Any Complex64;
            Any Complex128;
            Any Int4;
            Any Uint4;
            Any Bit;
            Any Float4_e2m1fn;
          ]
        (fun (D.Any dt) ->
          raises_match
            (Exn.invalid_arg ~substring:"Nx_io.save_safetensors: w: ")
            (fun () ->
              save_safetensors (temp_file ())
                [ ("w", Nx.P (Nx.zeros dt [| 2 |])) ]));
      test
        "every proper prefix fails, and any header byte flipped loads or fails"
        (fun () ->
          let whole = typed_file () in
          prefixes_fail Nx_io.load_safetensors whole;
          let header = String.length whole - 64 in
          flips_load_or_fail Nx_io.load_safetensors ~upto:header whole);
      (let whole = typed_file () in
       let with_length n =
         let b = Bytes.of_string whole in
         Bytes.set_int64_le b 0 n;
         Bytes.to_string b
       in
       let twice = [ ("w", "U8", [ 1 ], "a"); ("w", "U8", [ 1 ], "b") ] in
       cases ~name:fst "a malformed file or no regular file fails, naming it"
         [
           ("a byte past the data", fun () -> file (whole ^ "\000"));
           ("a header past the end", fun () -> file (with_length 8_000L));
           ("a header over 100 MB", fun () -> file (with_length 100_000_001L));
           ( "a header length's high bit",
             fun () -> file (with_length Int64.min_int) );
           ( "a shape nested in arrays",
             fun () ->
               let json =
                 {|{"w":{"dtype":"U8","shape":[[1]],"data_offsets":[0,1]}}|}
               in
               let pad =
                 String.make ((8 - (String.length json mod 8)) mod 8) ' '
               in
               file (with_header (json ^ pad) ^ "a") );
           ( "a header nested a million deep",
             fun () ->
               let n = 1_000_000 in
               let json =
                 "{\"w\":" ^ String.make n '[' ^ String.make n ']' ^ "}"
               in
               let pad =
                 String.make ((8 - (String.length json mod 8)) mod 8) ' '
               in
               file (with_header (json ^ pad)) );
           ( "a header that is no JSON",
             fun () -> file (with_header "{\"w\":" ^ "\000") );
           ("a name twice", fun () -> file (raw_safetensors twice));
           ( "an empty name",
             fun () -> file (raw_safetensors [ ("", "U8", [ 1 ], "a") ]) );
           ( "an unknown dtype",
             fun () ->
               file (raw_safetensors [ ("w", "C64", [ 1 ], "abcdefgh") ]) );
           ( "a size that is not the shape's",
             fun () -> file (raw_safetensors [ ("w", "U16", [ 1 ], "a") ]) );
           ( "a negative extent",
             fun () -> file (raw_safetensors [ ("w", "U8", [ -1 ], "") ]) );
           ( "a name that is no JSON string",
             fun () -> file (u8_entry {|\ud800|}) );
           ("a directory", fun () -> temp_dir ());
         ]
         (fun (_, path) ->
           let path = path () in
           fails ~path (fun () -> Nx_io.load_safetensors path)));
    ]

(* GGUF *)

module Gguf = Nx_io.Gguf

let le n bytes =
  String.init bytes (fun i -> Char.chr ((n lsr (8 * i)) land 0xff))

let le64 n =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 n;
  Bytes.to_string b

let u64 n = le64 (Int64.of_int n)
let gstring s = u64 (String.length s) ^ s

(* [encode v] is the type tag of [v] and its encoding, by the specification's
   table of value types. An empty array is written as one of [UINT8]. *)
let rec encode : Gguf.value -> int * string = function
  | Uint8 n -> (0, le n 1)
  | Int8 n -> (1, le n 1)
  | Uint16 n -> (2, le n 2)
  | Int16 n -> (3, le n 2)
  | Uint32 n -> (4, le n 4)
  | Int32 n -> (5, le n 4)
  | Float32 x -> (6, le (Int32.to_int (Int32.bits_of_float x)) 4)
  | Bool b -> (7, if b then "\001" else "\000")
  | String s -> (8, gstring s)
  | Array a ->
      let tag = if a = [||] then 0 else fst (encode a.(0)) in
      let items = Array.to_list (Array.map (fun v -> snd (encode v)) a) in
      (9, le tag 4 ^ u64 (Array.length a) ^ String.concat "" items)
  | Uint64 n -> (10, le64 n)
  | Int64 n -> (11, le64 n)
  | Float64 x -> (12, le64 (Int64.bits_of_float x))

let kv key v =
  let tag, payload = encode v in
  gstring key ^ le tag 4 ^ payload

let align a n = (n + a - 1) / a * a

(* [gguf ?magic ?version ?alignment ?shift kvs tensors] is a GGUF file of the
   encoded key-values [kvs] and of [tensors], each a name, a type tag, a logical
   shape and its bytes. Each tensor's data starts on a multiple of [alignment],
   [shift] bytes later, and the file ends where the last ends. *)
let gguf ?(magic = "GGUF") ?(version = 3) ?(alignment = 32) ?(shift = 0) kvs
    tensors =
  let offsets, _ =
    List.fold_left
      (fun (offsets, next) (_, _, _, data) ->
        let off = align alignment next + shift in
        (off :: offsets, off + String.length data))
      ([], 0) tensors
  in
  let offsets = List.rev offsets in
  let info (name, tag, shape, _) off =
    gstring name
    ^ le (List.length shape) 4
    ^ String.concat "" (List.rev_map u64 shape)
    ^ le tag 4 ^ u64 off
  in
  let header =
    magic ^ le version 4
    ^ u64 (List.length tensors)
    ^ u64 (List.length kvs)
    ^ String.concat "" kvs
    ^ String.concat "" (List.map2 info tensors offsets)
  in
  let data = Buffer.create 256 in
  List.iter2
    (fun (_, _, _, bytes) off ->
      Buffer.add_string data (String.make (off - Buffer.length data) '\000');
      Buffer.add_string data bytes)
    tensors offsets;
  let start = align alignment (String.length header) in
  header
  ^ String.make (start - String.length header) '\000'
  ^ Buffer.contents data

(* Metadata of every value type, at the bounds of each. *)
let metadata : (string * Gguf.value) list =
  [
    ("general.architecture", String "llama");
    ("u8", Uint8 255);
    ("i8", Int8 (-128));
    ("u16", Uint16 65535);
    ("i16", Int16 (-32768));
    ("u32", Uint32 0xFFFF_FFFF);
    ("i32", Int32 (-0x8000_0000));
    ("u64", Uint64 (-1L));
    ("i64", Int64 Int64.min_int);
    ("f32", Float32 (-1.5));
    ("f64", Float64 0.1);
    ("true", Bool true);
    ("false", Bool false);
    ("text", String "é🚀\000");
    ("empty text", String "");
    ("tokens", Array [| String "a"; String ""; String "bc" |]);
    ("ids", Array [| Int32 1; Int32 (-2) |]);
    ("empty", Array [||]);
    ("nested", Array [| Array [| Uint8 1 |]; Array [| Uint8 2; Uint8 3 |] |]);
  ]

let q8_0 = 8

(* Tensors, each a name, a type tag, a logical shape and its bytes, with the
   dtype and the shape nx loads it at, and its type. A Q8_0 row of 64 elements
   is two blocks of 34 bytes. *)
let gguf_tensors =
  [
    ("f32", 0, [ 2; 3 ], pattern 24 1, "float32", [| 2; 3 |], Gguf.F32);
    ("f16", 1, [ 3 ], pattern 6 2, "float16", [| 3 |], Gguf.F16);
    ("q8_0", q8_0, [ 2; 64 ], pattern 136 3, "uint8", [| 2; 68 |], Gguf.Q8_0);
    ("empty", 0, [ 0; 3 ], "", "float32", [| 0; 3 |], Gguf.F32);
  ]

let gguf_file ?version ?alignment ?shift () =
  let kvs =
    match alignment with
    | None -> metadata
    | Some a -> ("general.alignment", Gguf.Uint32 a) :: metadata
  in
  gguf ?version ?alignment ?shift
    (List.map (fun (k, v) -> kv k v) kvs)
    (List.map
       (fun (n, tag, shape, data, _, _, _) -> (n, tag, shape, data))
       gguf_tensors)

let gguf_value =
  Testable.structural ~pp:(fun ppf _ -> Format.pp_print_string ppf "<value>")

let gguf_group =
  group "gguf"
    [
      test "metadata of every value type loads as it was written" (fun () ->
          let g = Nx_io.load_gguf (file (gguf_file ())) in
          equal ~msg:"version" int 3 (Gguf.version g);
          equal (list (pair string gguf_value)) metadata (Gguf.metadata g));
      cases
        ~name:(fun (v, a, s) ->
          Printf.sprintf "version %d, alignment %s, shift %d" v
            (Option.fold ~none:"default" ~some:string_of_int a)
            s)
        "a tensor loads its bytes as stored, at any offset of the file"
        [
          (3, None, 0);
          (2, None, 0);
          (3, Some 64, 0);
          (3, Some 1, 0);
          (3, Some 1, 1);
        ]
        (fun (version, alignment, shift) ->
          let g =
            Nx_io.load_gguf (file (gguf_file ~version ?alignment ~shift ()))
          in
          equal ~msg:"version" int version (Gguf.version g);
          List.iter
            (fun (name, _, shape, data, dtype, shape', ty) ->
              equal ~msg:name
                (triple string (array int) string)
                (dtype, shape', data)
                (stored (find (Gguf.tensors g) name));
              let info = Gguf.info name g in
              is_true ~msg:"type" (info.dtype = ty);
              equal ~msg:"logical shape" (array int) (Array.of_list shape)
                info.shape)
            gguf_tensors);
      test "metadata of arrays nested a million deep loads" (fun () ->
          (* Each level is an array of one array: its element type, then its
             count. The innermost is an empty array of uint8. *)
          let n = 1_000_000 in
          let deep =
            gstring "k" ^ le 9 4
            ^ String.concat "" (List.init n (fun _ -> le 9 4 ^ u64 1))
            ^ le 0 4 ^ u64 0
          in
          let g =
            Nx_io.load_gguf (file (gguf [ deep ] [ ("w", 0, [ 1 ], "abcd") ]))
          in
          let rec depth d : Gguf.value -> int = function
            | Array [| v |] -> depth (d + 1) v
            | Array [||] -> d + 1
            | _ -> -1
          in
          equal int (n + 1) (depth 0 (List.assoc "k" (Gguf.metadata g))));
      test "every proper prefix fails, naming the file" (fun () ->
          prefixes_fail Nx_io.load_gguf (gguf_file ()));
      test "any header byte flipped loads or fails" (fun () ->
          flips_load_or_fail Nx_io.load_gguf ~upto:600 (gguf_file ()));
      test "an absent tensor's info fails" (fun () ->
          let g = Nx_io.load_gguf (file (gguf_file ())) in
          raises_match (Exn.failure ~substring:"absent") (fun () ->
              Gguf.info "absent" g));
      (let one = [ ("w", 0, [ 1 ], "abcd") ] in
       let raw ?magic ?version ?alignment ?shift kvs tensors () =
         file (gguf ?magic ?version ?alignment ?shift kvs tensors)
       in
       let count n = "GGUF" ^ le 3 4 ^ u64 n ^ u64 0 in
       cases ~name:fst "a malformed file fails, naming it"
         [
           ("an empty file", fun () -> file "");
           ("a bad magic", raw ~magic:"GGUG" [] one);
           ("version 1", raw ~version:1 [] one);
           ("version 4", raw ~version:4 [] one);
           ("a big-endian file", raw ~version:0x03000000 [] one);
           ("a tensor count past the end", fun () -> file (count max_int));
           ("a tensor count's high bit", fun () -> file (count (-1)));
           ("a key twice", raw [ kv "k" (Uint8 1); kv "k" (Uint8 2) ] one);
           ("a tensor twice", raw [] (one @ one));
           ("a tensor with an empty name", raw [] [ ("", 0, [ 1 ], "abcd") ]);
           ("a value type unknown", raw [ gstring "k" ^ le 13 4 ^ "\000" ] one);
           ("a bool of 2", raw [ gstring "k" ^ le 7 4 ^ "\002" ] one);
           ( "a string past the end",
             raw [ gstring "k" ^ le 8 4 ^ u64 max_int ] one );
           ("a tensor type unknown", raw [] [ ("w", 4, [ 1 ], "abcd") ]);
           ( "a row not a whole number of blocks",
             raw [] [ ("w", q8_0, [ 31 ], pattern 34 0) ] );
           ("a dimension's high bit", raw [] [ ("w", 0, [ -1 ], "abcd") ]);
           ( "dimensions whose product overflows",
             raw [] [ ("w", 0, [ 1 lsl 40; 1 lsl 40 ], "abcd") ] );
           ("an offset off the alignment", raw ~shift:1 [] one);
           ( "an alignment not a power of two",
             raw ~alignment:24 [ kv "general.alignment" (Uint32 24) ] one );
           ( "an alignment of zero",
             raw [ kv "general.alignment" (Uint32 0) ] one );
           ( "an alignment not a uint32",
             raw ~alignment:64 [ kv "general.alignment" (Uint64 64L) ] one );
           ("data cut short", fun () -> file (cut (gguf [] one) 1));
           ("a directory", fun () -> temp_dir ());
         ]
         (fun (_, path) ->
           let path = path () in
           fails ~path (fun () -> Nx_io.load_gguf path)));
    ]

(* Text *)

let txt_dtypes = List.filter (fun (D.Any dt) -> not (D.is D.Complex dt)) D.all

(* Two elements of [dt] are one value: floats by their bits, any two NaNs
   alike. *)
let same_float a b =
  (Float.is_nan a && Float.is_nan b)
  || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let same_elements (type v s) (dt : (v, s) D.t) (a : v array) (b : v array) =
  let same : v -> v -> bool =
    match D.kind dt with
    | D.Float -> same_float
    | D.Complex -> fun _ _ -> false
    | D.Signed | D.Unsigned | D.Boolean -> ( = )
  in
  Array.length a = Array.length b && Array.for_all2 same a b

(* Text keeps a 0-d value and a matrix of one row or column as a vector. *)
let text_shape s =
  match s with [||] -> [| 1 |] | [| 1; n |] | [| n; 1 |] -> [| n |] | s -> s

let text_cases =
  case ~rank:2
    ~kinds:[ Single; Nan; Negative_zero; Bounds; Many ]
    ~views:[ Plain; Transposed; Flipped ]
    txt_dtypes

let txt =
  group "text"
    [
      prop "a vector or a matrix saved as text loads as itself, NaN aside"
        text_cases (fun (Case (dt, _, _, _) as c) ->
          let x = Nx.unpack dt (value c) in
          covers ~empty:false ~broadcast:false c;
          cover "a sub-byte dtype" (D.bits dt < 8);
          let path = temp_file () in
          Nx_io.save_txt path x;
          let y = Nx_io.load_txt path dt in
          equal ~msg:"shape" (array int) (text_shape (Nx.shape x)) (Nx.shape y);
          is_true ~msg:"elements"
            (same_elements dt (Nx.to_array x) (Nx.to_array y)));
      test
        "a file numpy's savetxt wrote loads and saves back as written, its \
         floats in %.18e notation from their exact value" (fun () ->
          let numpy = fixture "savetxt.txt" and path = temp_file () in
          Nx_io.save_txt ~sep:"," ~newline:"\r\n" ~comments:"##"
            ~header:"alpha beta gamma" ~footer:"generated by numpy" path
            (Nx_io.load_txt ~sep:"," ~comments:"##" numpy Nx.float64);
          equal string (read numpy) (read path));
      cases ~name:fst
        "integers are written in decimal, unsigned ones as unsigned"
        [
          ( "1 0",
            fun p ->
              Nx_io.save_txt p (Nx.create Nx.bool [| 2 |] [| true; false |]) );
          ( "-9223372036854775808 9223372036854775807",
            fun p ->
              Nx_io.save_txt p
                (Nx.create Nx.int64 [| 2 |] [| Int64.min_int; Int64.max_int |])
          );
          ( "18446744073709551615 0",
            fun p ->
              Nx_io.save_txt p (Nx.create Nx.uint64 [| 2 |] [| -1L; 0L |]) );
          ( "4294967295",
            fun p -> Nx_io.save_txt p (Nx.create Nx.uint32 [| 1 |] [| -1l |]) );
          ( "-8 7",
            fun p -> Nx_io.save_txt p (Nx.create Nx.int4 [| 2 |] [| -8; 7 |]) );
        ]
        (fun (expected, save) ->
          let path = temp_file () in
          save path;
          equal string (expected ^ "\n") (read path));
      test "rows are skipped, counted and commented" (fun () ->
          let path = file "# a\n1 2\n\n3   4\n# b\n5 6\n" in
          equal (array int32)
            [| 1l; 2l; 3l; 4l; 5l; 6l |]
            (Nx.to_array (Nx_io.load_txt path Nx.int32));
          equal (array int) [| 3; 2 |] (Nx.shape (Nx_io.load_txt path Nx.int32));
          equal (array int32) [| 3l; 4l |]
            (Nx.to_array (Nx_io.load_txt ~skiprows:2 ~max_rows:1 path Nx.int32)));
      test "a float field takes a sign, a point, an exponent or a word"
        (fun () ->
          let path = file "+1e+3 .5 5. -2E-1 NaN -Infinity inf\n" in
          let x = Nx_io.load_txt path Nx.float64 in
          equal (array float_exact)
            [| 1000.; 0.5; 5.; -0.2; Float.neg_infinity; Float.infinity |]
            (Array.of_list
               (List.filter
                  (fun v -> not (Float.is_nan v))
                  (Array.to_list (Nx.to_array x))));
          is_true ~msg:"NaN" (Float.is_nan (Nx.to_array x).(4));
          equal (array int32) [| 7l; -7l |]
            (Nx.to_array (Nx_io.load_txt (file "+7 -7\n") Nx.int32)));
      test "a separator of several characters splits fields" (fun () ->
          let path = file "1::2::3\n" in
          equal (array int) [| 1; 2; 3 |]
            (Nx.to_array (Nx_io.load_txt ~sep:"::" path Nx.int16)));
      test "a bool field is true, false or a number" (fun () ->
          let path = file "true FALSE 0 2.5 -0.0\n" in
          equal (array bool)
            [| true; false; false; true; false |]
            (Nx.to_array (Nx_io.load_txt path Nx.bool)));
      cases ~name:fst "a field outside its dtype or no literal of it fails"
        [
          ("256 in uint8", ("1 256\n", D.Any D.Uint8));
          ("-1 in uint32", ("-1\n", D.Any D.Uint32));
          ("2^64 in uint64", ("18446744073709551616\n", D.Any D.Uint64));
          ("2^63 in int64", ("9223372036854775808\n", D.Any D.Int64));
          ("8 in int4", ("8\n", D.Any D.Int4));
          ("a word in float32", ("1 x\n", D.Any D.Float32));
          ("hexadecimal in int32", ("0x10\n", D.Any D.Int32));
          ("an underscore in int32", ("1_0\n", D.Any D.Int32));
          ("a hexadecimal float", ("0x1p3\n", D.Any D.Float64));
          ("a NaN with a payload", ("nan(1)\n", D.Any D.Float64));
          ("an exponent without digits", ("1e\n", D.Any D.Float64));
          ("a lone point", (".\n", D.Any D.Float64));
          ("a float in int8", ("1.0\n", D.Any D.Int8));
          ("rows of two lengths", ("1 2\n3\n", D.Any D.Float32));
          ("no row", ("# only\n\n", D.Any D.Float32));
        ]
        (fun (_, (text, D.Any dt)) ->
          let path = file text in
          fails ~path (fun () -> Nx_io.load_txt path dt));
      test "a complex dtype, three axes or a bad count are refused" (fun () ->
          let path = file "1\n" in
          invalid ~by:"Nx_io.load_txt" (fun () ->
              Nx_io.load_txt path Nx.complex64);
          invalid ~by:"Nx_io.load_txt" (fun () ->
              Nx_io.load_txt ~skiprows:(-1) path Nx.int8);
          invalid ~by:"Nx_io.load_txt" (fun () ->
              Nx_io.load_txt ~max_rows:0 path Nx.int8);
          invalid ~by:"Nx_io.save_txt" (fun () ->
              Nx_io.save_txt (temp_file ()) (Nx.zeros Nx.complex128 [| 2 |]));
          invalid ~by:"Nx_io.save_txt" (fun () ->
              Nx_io.save_txt (temp_file ()) (Nx.zeros Nx.int8 [| 1; 1; 1 |])));
    ]

let () = exit (run "nx.io" [ saves; npy; npz; safetensors; gguf_group; txt ])
