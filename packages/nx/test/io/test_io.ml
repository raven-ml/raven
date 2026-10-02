(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tensor I/O. Each format reads back what it writes, bit for bit where it is
   lossless, over every dtype it holds, every layout and the bit patterns that
   break codecs: NaN payloads, infinities, signed zeros and subnormals. Files
   that other tools wrote, in support/ by gen/generate.py, pin the formats
   themselves. *)

open Windtrap
open Nx_test
open Stored

(* Files *)

let read path = In_channel.with_open_bin path In_channel.input_all

let write path contents =
  Out_channel.with_open_bin path (fun oc -> output_string oc contents)

let file suffix contents =
  let path = temp_file ~suffix () in
  write path contents;
  path

let saved suffix save x =
  let path = temp_file ~suffix () in
  save path x;
  path

let missing name = Filename.concat (temp_dir ()) name
let fixture name = Filename.concat "support" name
let cut s n = String.sub s 0 (String.length s - n)

let flip_byte s i =
  String.mapi (fun j c -> if i = j then Char.chr (Char.code c lxor 1) else c) s

let le32 n = String.init 4 (fun i -> Char.chr ((n lsr (8 * i)) land 0xff))

(* [fails ?naming f] asserts that [f ()] raises [Failure] with a message that
   contains [naming]. *)
let fails ?naming f =
  raises_match (Exn.failure ?substring:naming) (fun () -> ignore (f ()))

let unix_error err f =
  raises_match
    (function Unix.Unix_error (e, _, _) -> e = err | _ -> false)
    (fun () -> ignore (f ()))

(* An archive compares as a set of named tensors. *)
let entries = slist (pair string packed) (fun (a, _) (b, _) -> compare a b)

let listed a =
  List.map
    (fun name -> (name, Option.get (Nx_io.Archive.find name a)))
    (Nx_io.Archive.names a)

let find a name = Option.get (Nx_io.Archive.find name a)

let round_trips ~save ~load cases =
  List.map
    (fun (Case c) ->
      prop (c.name ^ " tensors round trip bit for bit") c.tensors (fun t ->
          cover "an empty tensor" (Nx.numel t = 0);
          cover "a scalar" (Nx.ndim t = 0);
          cover "a view"
            ((not (Nx.is_c_contiguous t)) || Nx_array.View.offset (view t) <> 0);
          Law.round_trip packed string (saved "" save) load (Nx.P t)))
    cases

(* Archives of tensors of [cases], each under a distinct name of [names]. *)
let archives names cases =
  let entry =
    Gen.bind (Gen.of_list cases) (fun (Case c) ->
        Gen.map (fun t -> Nx.P t) c.tensors)
  in
  let entries_of chosen =
    let size = Gen.constant (List.length chosen) in
    Gen.map (List.combine chosen) (Gen.list ~size entry)
  in
  Gen.with_pp (Testable.pp entries)
    (Gen.bind (Gen.subsequence names) entries_of)

let archive_round_trip ~save ~load names cases =
  prop "an archive of named tensors round trips as a set" (archives names cases)
    (fun l ->
      cover "an empty archive" (l = []);
      cover "several entries" (List.length l >= 2);
      Law.round_trip entries string (saved "" save)
        (fun path -> listed (load path))
        l)

(* NumPy *)

let npy_cases =
  (bool :: ints) @ [ float16; float32; float64; complex64; complex128 ]

let save_npy path (Nx.P t) = Nx_io.save_npy path t
let save_npz path l = Nx_io.save_npz path (Nx_io.Archive.of_list l)

let npy =
  let loads name expected =
    (name, fun () -> equal packed expected (Nx_io.load_npy (fixture name)))
  in
  group "npy"
    [
      group "round trip"
        (round_trips ~save:save_npy ~load:Nx_io.load_npy npy_cases);
      cases ~name:fst "files written by numpy load"
        [
          loads "fortran.npy"
            (Nx.P (Nx.reshape [| 2; 3 |] (Nx.arange Nx.float32 1 7 1)));
          loads "big_endian.npy"
            (Nx.P (Nx.create Nx.int16 [| 3 |] [| 1; 256; -2 |]));
          loads "version2.npy" (Nx.P (Nx.arange Nx.int32 7 10 1));
          loads "version3.npy"
            (Nx.P (Nx.create Nx.bool [| 3 |] [| true; false; true |]));
        ]
        (fun (_, check) -> check ());
      test "a tensor of 2^17 + 3 elements round trips" (fun () ->
          let t = Nx.P (Nx.arange Nx.float32 0 ((1 lsl 17) + 3) 1) in
          equal packed t (Nx_io.load_npy (saved "" save_npy t)));
      test "a byte past the payload or a dtype nx lacks fails" (fun () ->
          let s = read (saved "" save_npy (Nx.P (Nx.zeros Nx.int8 [| 3 |]))) in
          fails (fun () -> Nx_io.load_npy (file "" (s ^ "\000")));
          fails (fun () -> Nx_io.load_npy (fixture "unicode.npy")));
      test "a save refuses memory a consuming call holds" (fun () ->
          let x = Nx.arange Nx.int32 0 6 1 in
          let b = Nx.to_buffer x in
          Nx_device.Buffer.Claim.read b;
          is_true (Nx_device.Buffer.Claim.try_exclusive b);
          Fun.protect
            ~finally:(fun () ->
              Nx_device.Buffer.Claim.finish b;
              Nx_device.Buffer.Claim.release b)
            (fun () ->
              fails ~naming:"in use by a consuming call" (fun () ->
                  Nx_io.save_npy (temp_file ~suffix:".npy" ()) x)));
      test "a save that cannot read its tensor names itself" (fun () ->
          let run : type r. r Nx.Op.t -> r = function
            | Read { by; _ } -> invalid_arg by
            | op -> Nx.Op.eval op
          in
          let refusing = { Nx.Op.run; claims = (fun _ -> true) } in
          let path = temp_file ~suffix:".npy" () in
          fails ~naming:"Nx_io.save_npy" (fun () ->
              Nx.Op.intercept refusing (fun () ->
                  Nx_io.save_npy path (Nx.zeros Nx.int8 [| 3 |]))));
    ]

let npz =
  let names = [ "w"; "layer.0/weight"; "a b"; "é"; "日本"; "deep/er/name" ] in
  let one = Nx.P (Nx.zeros Nx.int8 [| 1 |]) in
  group "npz"
    [
      (* load_npz_entry reads what load_npz reads, and fails on another name. *)
      archive_round_trip ~save:save_npz names npy_cases ~load:(fun path ->
          let archive = Nx_io.load_npz path in
          let entry name p = equal packed p (Nx_io.load_npz_entry ~name path) in
          List.iter (fun (name, p) -> entry name p) (listed archive);
          fails (fun () -> Nx_io.load_npz_entry ~name:"absent" path);
          archive);
      test "an archive written by numpy loads" (fun () ->
          equal entries
            [
              ("a", Nx.P (Nx.create Nx.float32 [| 1; 2 |] [| 1.5; -2. |]));
              ("b", Nx.P (Nx.arange Nx.int64 3 6 1));
            ]
            (listed (Nx_io.load_npz (fixture "archive.npz"))));
      cases ~name:Fun.id "an invalid name is refused"
        [ "/w"; "a//b"; "./w"; "a/../b"; "\xff" ] (fun name ->
          fails (fun () -> save_npz (temp_file ()) [ (name, one) ]));
      test "an incompressible entry is stored and a compressible one deflated"
        (fun () ->
          let s = Random.State.make [| 0x4e58 |] in
          let random _ = Random.State.int s 256 in
          List.iter
            (fun (name, t, method_) ->
              let path = saved "" save_npz [ (name, Nx.P t) ] in
              (* The compression method of the archive's first entry. *)
              equal ~msg:name int method_ (String.get_uint16_le (read path) 8);
              equal packed (Nx.P t) (Nx_io.load_npz_entry ~name path))
            [
              ("stored", Nx.init Nx.uint8 [| 70_000 |] random, 0);
              ("deflated", Nx.zeros Nx.uint8 [| 70_000 |], 8);
            ]);
      test
        "a deflated entry whose data starts at no multiple of its element size \
         loads" (fun () ->
          equal packed
            (Nx.P (Nx.create Nx.float64 [| 5 |] [| 0.; 1.5; 3.; 4.5; 6. |]))
            (Nx_io.load_npz_entry ~name:"w" (fixture "unaligned.npz")));
      test "an entry whose checksum does not match fails" (fun () ->
          let s = read (saved "" save_npz [ ("w", one) ]) in
          (* The CRC-32 of the central directory's first entry. *)
          let rec crc i =
            if String.sub s i 4 = "PK\001\002" then i + 16 else crc (i + 1)
          in
          let path = file "" (flip_byte s (crc 0)) in
          fails (fun () -> Nx_io.load_npz_entry ~name:"w" path));
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
          let path = file "" (Bytes.to_string b) in
          fails ~naming:"declares more data" (fun () ->
              Nx_io.load_npz_entry ~name:"w" path));
    ]

(* Compression *)

let gzip s = Compress_deflate.Gzip.compress s

let gunzipped src =
  let dst = temp_file () in
  Nx_io.gunzip ~src ~dst;
  read dst

let compression =
  group "compression"
    [
      prop "gunzip decompresses gzip members to their concatenation"
        ~examples:[ [ "\144\144\144\144" ] ]
        (Gen.list ~size:(Gen.int_range 1 4) Gen.string)
        (fun l ->
          let members = String.concat "" (List.map gzip l) in
          equal string (String.concat "" l) (gunzipped (file "" members)));
      test "gunzip streams data many times longer than its buffers" (fun () ->
          let r = Random.State.make [| 7 |] in
          let data =
            String.init
              (3 * 1024 * 1024)
              (fun _ -> Char.chr (Char.code 'a' + Random.State.int r 16))
          in
          let half = String.length data / 2 in
          let members =
            gzip (String.sub data 0 half)
            ^ gzip (String.sub data half (String.length data - half))
          in
          equal string data (gunzipped (file "" members)));
      test "gunzip decompresses a file written by Python's gzip" (fun () ->
          equal string "hello, nx gzip!\n" (gunzipped (fixture "hello.gz")));
      cases ~name:fst
        "a malformed member fails and leaves the destination as it was"
        [
          ("a checksum", fun s -> flip_byte s (String.length s - 8));
          ("a partial member after it", fun s -> s ^ String.sub s 0 12);
          ("a size field", fun s -> cut s 4 ^ le32 99);
          ("no member", fun _ -> "");
        ]
        (fun (_, damage) ->
          let src = file "" (damage (read (fixture "hello.gz"))) in
          let dst = file "" "sentinel" in
          fails (fun () -> Nx_io.gunzip ~src ~dst);
          equal string "sentinel" (read dst));
    ]

(* SafeTensors *)

let safetensors_cases =
  (bool :: ints)
  @ [ float16; bfloat16; float32; float64; float8_e4m3; float8_e5m2 ]

let save_safetensors path p =
  Nx_io.save_safetensors path (Nx_io.Archive.of_list [ ("t", p) ])

let save_safetensors_list path l =
  Nx_io.save_safetensors path (Nx_io.Archive.of_list l)

(* Not inlined, so that once it returns nothing but its result keeps the file
   open. *)
let[@inline never] load_entry path name =
  find (Nx_io.load_safetensors path) name

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
    ("f8_e4m3", "F8_E4M3", "float8_e4m3", [ 3 ], pattern 3 5);
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

(* An entry loads its bytes as stored, at its dtype, or as bytes at [uint8]: of
   its shape, or of shape [[| n |]] for the formats narrower than a byte. *)
let loads_typed_entries path =
  let archive = Nx_io.load_safetensors path in
  let names = List.map (fun (name, _, _, _, _) -> name) typed_entries in
  equal (slist string compare) names (List.map fst (listed archive));
  List.iter
    (fun (name, tag, dtype, shape, data) ->
      let narrow = List.mem tag [ "F4"; "F6_E2M3"; "F6_E3M2" ] in
      let shape =
        if narrow then [| String.length data |] else Array.of_list shape
      in
      equal ~msg:name
        (triple string (array int) string)
        (dtype, shape, data)
        (storage (find archive name)))
    typed_entries

let payload path =
  let s = read path in
  let start = 8 + Int64.to_int (String.get_int64_le s 0) in
  String.sub s start (String.length s - start)

let safetensors =
  let json_names =
    [ "é🚀"; "tab\tnew\nline"; "quote\"back\\slash/"; "\x00\x01\x1f" ]
  in
  let u8_entry name =
    with_header
      ("{\"" ^ name ^ {|":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}|})
    ^ "\042"
  in
  let disk = Nx.Placement.on (Nx.Device.make Nx_device.disk) in
  let on_disk archive =
    List.for_all
      (fun (_, Nx.P t) -> Nx.Placement.equal disk (Nx.placement t))
      (listed archive)
  in
  let reads f =
    let read () = Nx_device.Stats.bytes_out (Nx_device.stats Nx_device.disk) in
    let before = read () in
    let y = f () in
    (y, read () - before)
  in
  group "safetensors"
    [
      group "round trip"
        (round_trips ~save:save_safetensors
           ~load:(fun path -> load_entry path "t")
           safetensors_cases);
      archive_round_trip ~save:save_safetensors_list
        ~load:Nx_io.load_safetensors
        ("w" :: "model.layers.0.weight" :: json_names)
        safetensors_cases;
      cases ~name:fst
        "files written by numpy and torch load, and save again as they are"
        [
          ("f16", [| 0x0000; 0x0001; 0x3C00; 0x7C00; 0x7E01 |]);
          ("bf16", [| 0x0000; 0x0001; 0x3F80; 0x7F80; 0x7FC1 |]);
        ]
        (fun (dtype, bits) ->
          let path = fixture (dtype ^ "_bit_exact.safetensors") in
          let entry = dtype ^ "_tensor" in
          let (Nx.P t as p) = load_entry path entry in
          equal (array int) bits (Nx.to_array (Nx.bitcast Nx.uint16 t));
          let again = temp_file () in
          save_safetensors_list again [ (entry, p) ];
          equal string (payload path) (payload again));
      test "a save that cannot read its traced tensor names itself" (fun () ->
          let module N = struct
            type (_, _) Nx.Repr.node += Node : ('a, 'b) Nx.Repr.node
          end in
          let t =
            Nx.Repr.Traced.v ~context:Nx.Placement.host Nx.Placement.host
              Nx.float32 [| 3 |] N.Node
          in
          let run : type r. r Nx.Op.t -> r = function
            | Read { by; _ } -> invalid_arg by
            | Contiguous x -> x
            | op -> Nx.Op.eval op
          in
          let refusing = { Nx.Op.run; claims = (fun _ -> true) } in
          let path = temp_file ~suffix:".safetensors" () in
          raises_match (Exn.invalid_arg ~substring:"Nx_io.save_safetensors")
            (fun () ->
              Nx.Op.intercept refusing (fun () ->
                  save_safetensors path (Nx.P t))));
      test "a header's JSON string escapes are decoded" (fun () ->
          let name = "aé🚀\"\\/\b\012\r\n\t" in
          let p = Nx.P (Nx.create Nx.uint8 [| 1 |] [| 42 |]) in
          equal entries
            [ (name, p) ]
            (listed
               (Nx_io.load_safetensors
                  (file ""
                     (u8_entry {|\u0061\u00e9\uD83D\uDE80\"\\\/\b\f\r\n\t|})))));
      cases ~name:Fun.id "a name that is no JSON string is refused"
        [
          {|\ud800|}; {|\udc00|}; {|\ud800\u0041|}; {|\u12xz|}; {|\q|}; "a\001b";
        ] (fun name ->
          fails (fun () -> Nx_io.load_safetensors (file "" (u8_entry name))));
      test
        "an entry of any dtype loads its bytes as stored, on the disk, at any \
         offset of the file" (fun () ->
          List.iter
            (fun pad ->
              let path = file "" (typed_file ~pad ()) in
              loads_typed_entries path;
              is_true ~msg:"on the disk" (on_disk (Nx_io.load_safetensors path)))
            [ 0; 1 ]);
      test
        "a load reads the header alone, and a use of an entry its file's pages \
         where they are aligned to its elements, and its bytes otherwise"
        (fun () ->
          let used pad =
            let path = file "" (typed_file ~pad ()) in
            let header =
              String.length (read path) - String.length (payload path)
            in
            let archive, bytes =
              reads (fun () -> Nx_io.load_safetensors path)
            in
            equal ~msg:"the header" int header bytes;
            let (Nx.P t) = find archive "u32" in
            snd (reads (fun () -> Nx.to_array (Nx.bitcast Nx.int32 t)))
          in
          equal ~msg:"an aligned entry" int 0 (used 0);
          equal ~msg:"an entry one byte past" int 16 (used 1));
      test
        "an entry of a file truncated since it loaded fails at use, naming it"
        (fun () ->
          let path = temp_file () in
          save_safetensors path (Nx.P (Nx.arange Nx.int32 0 1024 1));
          let (Nx.P t) = load_entry path "t" in
          Unix.truncate path 100;
          raises_match (Exn.sys_error ~substring:path) (fun () -> Nx.to_array t));
      (let whole = typed_file () in
       let with_length n =
         let b = Bytes.of_string whole in
         Bytes.set_int64_le b 0 n;
         Bytes.to_string b
       in
       let contents s () = file "" s in
       let fifo () =
         let path = missing "fifo" in
         if Sys.win32 then skip ~reason:"no FIFO on Windows" ();
         Unix.mkfifo path 0o600;
         path
       in
       let twice = [ ("w", "U8", [ 1 ], "a"); ("w", "U8", [ 1 ], "b") ] in
       cases ~name:fst "a malformed file or no regular file fails, naming it"
         [
           ("data cut short", contents (cut whole 5));
           ("a byte past the data", contents (whole ^ "\000"));
           ("a header past the end", contents (with_length 8_000L));
           ("a header over 100 MB", contents (with_length 100_000_001L));
           ("a header length's high bit", contents (with_length Int64.min_int));
           ( "a header that is no JSON",
             contents (with_header "{\"w\":" ^ "\000") );
           ("a name twice", contents (raw_safetensors twice));
           ( "an empty name",
             contents (raw_safetensors [ ("", "U8", [ 1 ], "a") ]) );
           ("a directory", fun () -> temp_dir ());
           ("a FIFO", fifo);
         ]
         (fun (_, path) ->
           let path = path () in
           fails ~naming:path (fun () -> Nx_io.load_safetensors path)));
      test "a refused rename keeps the temporary file and names it" (fun () ->
          let dir = temp_dir () in
          let path = Filename.concat dir "w.safetensors" in
          Sys.mkdir path 0o755;
          write (Filename.concat path "occupant") "";
          let t = Nx.P (Nx.arange Nx.float32 0 3 1) in
          let message =
            match save_safetensors path t with
            | () -> fail "the rename over a directory was not refused"
            | exception Failure message -> message
          in
          match
            List.filter (( <> ) "w.safetensors")
              (Array.to_list (Sys.readdir dir))
          with
          | [ name ] ->
              let kept = Filename.concat dir name in
              contains ~sub:path message;
              contains ~sub:kept message;
              equal packed t (load_entry kept "t")
          | names -> failf "%d files kept, one expected" (List.length names));
      test
        "a file whose tensors are alive is replaced and deleted, the tensors \
         outliving their archive unchanged, and saved again from the disk"
        (fun () ->
          let path = temp_file () in
          let first = Nx.P (Nx.arange Nx.int32 0 65536 1) in
          let second = Nx.P (Nx.arange Nx.int32 1 65537 1) in
          save_safetensors path first;
          let loaded = load_entry path "t" in
          (match save_safetensors path second with
          | () -> equal ~msg:"the new file" packed second (load_entry path "t")
          | exception Failure reason when Sys.win32 -> skip ~reason ());
          (match Sys.remove path with
          | () -> ()
          | exception Sys_error reason when Sys.win32 -> skip ~reason ());
          Gc.full_major ();
          equal ~msg:"the tensor loaded first" packed first loaded;
          let again = temp_file () in
          save_safetensors again loaded;
          equal ~msg:"saved from the disk" packed first (load_entry again "t"));
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
   shape and the tensor's bytes. Each tensor's data starts on a multiple of
   [alignment], [shift] bytes later, and the file ends where the last ends. *)
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
    let dims = List.rev_map u64 shape in
    gstring name
    ^ le (List.length shape) 4
    ^ String.concat "" dims ^ le tag 4 ^ u64 off
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

let raw_tensors =
  List.map (fun (n, tag, shape, data, _, _, _) -> (n, tag, shape, data))

let gguf_file ?version ?alignment ?shift () =
  let kvs =
    match alignment with
    | None -> metadata
    | Some a -> ("general.alignment", Gguf.Uint32 a) :: metadata
  in
  gguf ?version ?alignment ?shift
    (List.map (fun (k, v) -> kv k v) kvs)
    (raw_tensors gguf_tensors)

let rec pp_value ppf : Gguf.value -> unit = function
  | Uint8 n -> Format.fprintf ppf "Uint8 %d" n
  | Int8 n -> Format.fprintf ppf "Int8 %d" n
  | Uint16 n -> Format.fprintf ppf "Uint16 %d" n
  | Int16 n -> Format.fprintf ppf "Int16 %d" n
  | Uint32 n -> Format.fprintf ppf "Uint32 %d" n
  | Int32 n -> Format.fprintf ppf "Int32 %d" n
  | Uint64 n -> Format.fprintf ppf "Uint64 %Lu" n
  | Int64 n -> Format.fprintf ppf "Int64 %Ld" n
  | Float32 x -> Format.fprintf ppf "Float32 %h" x
  | Float64 x -> Format.fprintf ppf "Float64 %h" x
  | Bool b -> Format.fprintf ppf "Bool %b" b
  | String s -> Format.fprintf ppf "String %S" s
  | Array a ->
      Format.fprintf ppf "Array [|%a|]"
        (Format.pp_print_array
           ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
           pp_value)
        a

let gguf_value = Testable.structural ~pp:pp_value
let gguf_metadata = list (pair string gguf_value)

let pp_dtype ppf (d : Gguf.dtype) =
  Format.pp_print_string ppf
    (match d with F32 -> "F32" | F16 -> "F16" | Q8_0 -> "Q8_0" | _ -> "other")

let tensor_info =
  Testable.contramap
    (fun (i : Gguf.tensor_info) -> (i.dtype, i.shape))
    (pair (Testable.structural ~pp:pp_dtype) (array int))

(* The file's tensors load as their bytes, at their dtypes and shapes, with
   their descriptions. *)
let loads_gguf_tensors (g : Gguf.t) =
  let names = List.map (fun (n, _, _, _, _, _, _) -> n) gguf_tensors in
  equal ~msg:"names" (slist string compare) names
    (Nx_io.Archive.names (Gguf.tensors g));
  List.iter
    (fun (name, _, shape, data, dtype, stored, ty) ->
      equal ~msg:name
        (triple string (array int) string)
        (dtype, stored, data)
        (storage (find (Gguf.tensors g) name));
      equal ~msg:name tensor_info
        { dtype = ty; shape = Array.of_list shape }
        (Gguf.info name g))
    gguf_tensors

(* The GGUF files in the caches of the Hugging Face hub and llama.cpp. *)
let cached_gguf () =
  let home = Option.value (Sys.getenv_opt "HOME") ~default:"" in
  let dirs =
    List.map (Filename.concat home)
      [
        ".cache/huggingface/hub"; ".cache/llama.cpp"; "Library/Caches/llama.cpp";
      ]
  in
  let rec find dir =
    match Sys.readdir dir with
    | exception Sys_error _ -> None
    | names ->
        Array.sort compare names;
        Array.to_seq names
        |> Seq.find_map (fun name ->
            let path = Filename.concat dir name in
            if Filename.check_suffix name ".gguf" then Some path
            else if Sys.is_directory path then find path
            else None)
  in
  List.find_map find dirs

let gguf_group =
  let disk = Nx.Placement.on (Nx.Device.make Nx_device.disk) in
  let off_disk (g : Gguf.t) =
    List.filter_map
      (fun (name, Nx.P t) ->
        if Nx.Placement.equal disk (Nx.placement t) then None else Some name)
      (listed (Gguf.tensors g))
  in
  group "gguf"
    [
      test "metadata of every value type loads as it was written" (fun () ->
          let g = Nx_io.load_gguf (file "" (gguf_file ())) in
          equal ~msg:"version" int 3 (Gguf.version g);
          equal gguf_metadata metadata (Gguf.metadata g));
      cases
        ~name:(fun (v, a, s) ->
          Printf.sprintf "version %d, alignment %s, shift %d" v
            (Option.fold ~none:"default" ~some:string_of_int a)
            s)
        "a tensor loads its bytes as stored, on the disk, at any offset of the \
         file"
        [
          (3, None, 0);
          (2, None, 0);
          (3, Some 64, 0);
          (3, Some 1, 0);
          (3, Some 1, 1);
        ]
        (fun (version, alignment, shift) ->
          let g =
            Nx_io.load_gguf (file "" (gguf_file ~version ?alignment ~shift ()))
          in
          equal ~msg:"version" int version (Gguf.version g);
          loads_gguf_tensors g;
          equal ~msg:"off the disk" (list string) [] (off_disk g));
      test "every proper prefix of a file fails, naming it" (fun () ->
          let whole = gguf_file () in
          for n = 0 to String.length whole - 1 do
            let path = file "" (String.sub whole 0 n) in
            fails ~naming:path (fun () -> Nx_io.load_gguf path)
          done);
      (let one = [ ("w", 0, [ 1 ], "abcd") ] in
       let raw ?magic ?version ?alignment ?shift kvs tensors () =
         file "" (gguf ?magic ?version ?alignment ?shift kvs tensors)
       in
       let count n = "GGUF" ^ le 3 4 ^ u64 n ^ u64 0 in
       let fifo () =
         let path = missing "fifo" in
         if Sys.win32 then skip ~reason:"no FIFO on Windows" ();
         Unix.mkfifo path 0o600;
         path
       in
       cases ~name:fst "a malformed file or no regular file fails, naming it"
         [
           ("an empty file", fun () -> file "" "");
           ("a bad magic", raw ~magic:"GGUG" [] one);
           ("version 1", raw ~version:1 [] one);
           ("version 4", raw ~version:4 [] one);
           ("a big-endian file", raw ~version:0x03000000 [] one);
           ("a tensor count past the end", fun () -> file "" (count max_int));
           ("a tensor count's high bit", fun () -> file "" (count (-1)));
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
           ("data cut short", fun () -> file "" (cut (gguf [] one) 1));
           ("a directory", fun () -> temp_dir ());
           ("a FIFO", fifo);
         ]
         (fun (_, path) ->
           let path = path () in
           fails ~naming:path (fun () -> Nx_io.load_gguf path)));
      test "a file in the model caches loads" (fun () ->
          match cached_gguf () with
          | None -> skip ~reason:"no GGUF file in the model caches" ()
          | Some path ->
              let g = Nx_io.load_gguf path in
              (match
                 List.assoc_opt "general.architecture" (Gguf.metadata g)
               with
              | Some (String _) -> ()
              | v ->
                  failf "general.architecture is %a"
                    (Format.pp_print_option pp_value)
                    v);
              equal ~msg:"off the disk" (list string) [] (off_disk g);
              List.iter
                (fun (name, Nx.P t) ->
                  let i = Gguf.info name g in
                  let n = Array.length i.shape in
                  equal ~msg:name (array int)
                    (Array.sub i.shape 0 (n - 1))
                    (Array.sub (Nx.shape t) 0 (n - 1)))
                (listed (Gguf.tensors g)));
    ]

(* Text *)

let txt_cases = (bool :: ints) @ [ float16; bfloat16; float32; float64 ]

(* Whether [t] loads back from text at its own shape: a vector, or a matrix of
   at least two rows and two columns. *)
let txt_shaped t =
  Nx.numel t > 0
  &&
  match Nx.shape t with
  | [| _ |] -> true
  | [| r; c |] -> r >= 2 && c >= 2
  | _ -> false

let save_txt path t = Nx_io.save_txt path t
let matrices = Gen.such_that txt_shaped float64s
let int32s shape l = Nx.create Nx.int32 shape (Array.of_list l)
let strings = Gen.of_list ~pp:(fun ppf s -> Format.fprintf ppf "%S" s)

let comment_text =
  Gen.string_of ~size:(Gen.int_range 0 40)
    (Gen.frequency [ (9, Gen.char_range ' ' '~'); (1, Gen.constant '\n') ])

let txt =
  group "text"
    [
      group "round trip"
        (List.map
           (fun (Case c) ->
             prop
               (c.name
              ^ " tensors round trip, NaN keeping no sign, a single row or \
                 column as a vector")
               (Gen.such_that
                  (fun t -> Nx.numel t > 0 && Nx.ndim t <= 2)
                  c.tensors)
               (fun t ->
                 cover "a matrix" (Nx.ndim t = 2 && txt_shaped t);
                 cover "a scalar" (Nx.ndim t = 0);
                 let loaded = Nx_io.load_txt (saved "" save_txt t) c.dtype in
                 let t =
                   if txt_shaped t then t else Nx.reshape [| Nx.numel t |] t
                 in
                 equal c.values t loaded))
           txt_cases);
      prop
        "a separator, a newline and a header and footer under a comment prefix \
         read back with the same separator and prefix"
        (Gen.quad matrices
           (Gen.pair
              (strings [ " "; ","; "\t"; ";"; ", "; "||" ])
              (strings [ "\n"; "\r\n" ]))
           (Gen.pair comment_text comment_text)
           (strings [ "#"; "%%"; "//" ]))
        (fun (t, (sep, newline), (header, footer), comments) ->
          let path = temp_file () in
          Nx_io.save_txt ~sep ~newline ~comments:(comments ^ " ") ~header
            ~footer path t;
          equal (tensor float_exact) t
            (Nx_io.load_txt ~sep ~comments path Nx.float64));
      test
        "blank lines are skipped, ~append adds rows, ~skiprows skips lines and \
         ~max_rows bounds the rows read" (fun () ->
          let path = file "" "\n1 2\n  \n" in
          Nx_io.save_txt ~append:true path (int32s [| 2 |] [ 3l; 4l ]);
          equal (tensor int32)
            (int32s [| 2; 2 |] [ 1l; 2l; 3l; 4l ])
            (Nx_io.load_txt path Nx.int32);
          equal (tensor int32)
            (int32s [| 2 |] [ 3l; 4l ])
            (Nx_io.load_txt ~skiprows:2 ~max_rows:1 path Nx.int32));
      test
        "a file numpy's savetxt wrote loads and saves back as written, its \
         floats in %.18e notation from their exact value, NaN and the \
         infinities as nan, inf and -inf" (fun () ->
          let numpy = fixture "savetxt.txt" and path = temp_file () in
          Nx_io.save_txt ~sep:"," ~newline:"\r\n" ~comments:"##"
            ~header:"alpha beta gamma" ~footer:"generated by numpy" path
            (Nx_io.load_txt ~sep:"," ~comments:"##" numpy Nx.float64);
          equal string (read numpy) (read path));
      cases
        ~name:(fun (_, Nx.P t) -> Nx_dtype.to_string (Nx.dtype t))
        "integers are written in decimal, unsigned ones as unsigned"
        [
          ("1 0", Nx.P (Nx.create Nx.bool [| 2 |] [| true; false |]));
          ( "-9223372036854775808 9223372036854775807",
            Nx.P (Nx.create Nx.int64 [| 2 |] [| Int64.min_int; Int64.max_int |])
          );
          ( "4294967295 2147483648",
            Nx.P (Nx.create Nx.uint32 [| 2 |] [| -1l; Int32.min_int |]) );
          ( "18446744073709551615 9223372036854775808",
            Nx.P (Nx.create Nx.uint64 [| 2 |] [| -1L; Int64.min_int |]) );
        ]
        (fun (text, (Nx.P t as p)) ->
          equal string (text ^ "\n") (read (saved "" save_txt t));
          equal packed p (Nx.P (Nx_io.load_txt (file "" text) (Nx.dtype t))));
      cases ~name:fst "text that is no int8 tensor fails to load"
        [
          ("no data", "# only a comment\n\n");
          ("a word", "1 two 3\n");
          ("rows of different lengths", "1 2\n3\n");
          ("a value out of range", "300\n");
        ]
        (fun (_, text) ->
          fails (fun () -> Nx_io.load_txt (file "" text) Nx.int8));
      test
        "a negative unsigned value, a tensor of three dimensions, a negative \
         ~skiprows and a ~max_rows below one are refused" (fun () ->
          let path = file "" "1 2\n" and negative = file "" "-1\n" in
          fails (fun () -> Nx_io.load_txt negative Nx.uint32);
          fails (fun () -> Nx_io.load_txt negative Nx.uint64);
          fails (fun () ->
              Nx_io.save_txt path (Nx.zeros Nx.int32 [| 2; 2; 2 |]));
          fails (fun () -> Nx_io.load_txt ~skiprows:(-1) path Nx.int32);
          fails (fun () -> Nx_io.load_txt ~max_rows:0 path Nx.int32));
    ]

(* Images *)

(* The layouts that keep an image an image of at least one row. *)
let image_layout =
  let steps =
    [
      "flipped";
      "every other row";
      "without its first row";
      "without its last row";
    ]
  in
  let keeps (l : layout) = List.mem l.name steps in
  Gen.with_pp pp_layout
    (Gen.list ~size:(Gen.int_range 0 3)
       (Gen.of_list (List.filter keeps layout_steps)))

(* Shapes of [channels] channels, [0] for no channel axis. *)
let image_shape ~size channels =
  let open Gen in
  let* h = size in
  let* w = size in
  let+ c = of_list channels in
  if c = 0 then [| h; w |] else [| h; w; c |]

let images channels =
  viewed
    ~shape:(image_shape ~size:(Gen.int_range 5 8) channels)
    ~layout:image_layout ~pp:Format.pp_print_int Nx.uint8 (Gen.int_range 0 255)

(* Images whose channel [c] is [a + 60 c + sy y + sx x] clamped to a byte. *)
let smooth channels =
  let open Gen in
  let slope = int_range (-6) 6 in
  let+ s = with_pp pp_shape (image_shape ~size:(int_range 1 24) channels)
  and+ a, sy, sx = triple (int_range 0 255) slope slope in
  Nx.init Nx.uint8 s (fun i ->
      let c = if Array.length s = 3 then i.(2) else 0 in
      Int.max 0 (Int.min 255 (a + (60 * c) + (sy * i.(0)) + (sx * i.(1)))))

let errors a b =
  Array.map2 (fun x y -> abs (x - y)) (Nx.to_array a) (Nx.to_array b)

let flat t = Nx.reshape [| Nx.dim 0 t; Nx.dim 1 t |] t

(* The pixels Pillow decodes from the fixture image [name]. *)
let pillow name =
  let npy = Filename.remove_extension name ^ ".npy" in
  Nx.unpack Nx.uint8 (Nx_io.load_npy (fixture npy))

let refuse_image name shape =
  fails (fun () -> Nx_io.save_image (missing name) (Nx.zeros Nx.uint8 shape))

(* The chunks of the PNG file [png], as their types and data, in order. *)
let chunks png =
  let rec loop i acc =
    if i >= String.length png then List.rev acc
    else
      let n = Int32.to_int (String.get_int32_be png i) in
      let chunk = (String.sub png (i + 4) 4, String.sub png (i + 8) n) in
      loop (i + 12 + n) (chunk :: acc)
  in
  loop 8 []

let chunk_data ty png =
  List.filter_map
    (fun (ty', data) -> if ty = ty' then Some data else None)
    (chunks png)

(* [png] without its chunks of type [ty]. *)
let without ty png =
  let b = Buffer.create (String.length png) in
  Buffer.add_string b (String.sub png 0 8);
  let rec loop i =
    if i < String.length png then begin
      let n = Int32.to_int (String.get_int32_be png i) in
      if String.sub png (i + 4) 4 <> ty then
        Buffer.add_string b (String.sub png i (12 + n));
      loop (i + 12 + n)
    end
  in
  loop 8;
  Buffer.contents b

let be32 n = String.init 4 (fun i -> Char.chr ((n lsr (8 * (3 - i))) land 0xff))
let tiny = Nx.full Nx.uint8 [| 2; 3; 3 |] 9

let png_chunks_group =
  group "PNG chunks"
    [
      test "encode_png writes neither pHYs nor sRGB by default" (fun () ->
          let png = Nx_io.encode_png tiny in
          equal (list string) [] (chunk_data "pHYs" png);
          equal (list string) [] (chunk_data "sRGB" png));
      cases ~name:fst "~dpi writes the pixels per metre on both axes, in metres"
        [
          ("72 dpi is 2835 pixels per metre", (72., 2835));
          ("144 dpi is 5669 pixels per metre", (144., 5669));
          ("0.0127 dpi rounds up to 1", (0.0127, 1));
          ("the largest PNG integer", (2147483647. *. 0.0254, 2147483647));
        ]
        (fun (_, (dpi, ppm)) ->
          equal (list string)
            [ be32 ppm ^ be32 ppm ^ "\001" ]
            (chunk_data "pHYs" (Nx_io.encode_png ~dpi tiny)));
      test "~srgb writes one sRGB chunk with the perceptual intent" (fun () ->
          equal (list string) [ "\000" ]
            (chunk_data "sRGB" (Nx_io.encode_png ~srgb:true tiny)));
      test "the chunks precede the image data" (fun () ->
          let png = Nx_io.encode_png ~dpi:72. ~srgb:true tiny in
          let rec before_idat = function
            | [] | ("IDAT", _) :: _ -> []
            | (ty, _) :: rest -> ty :: before_idat rest
          in
          equal
            (slist string String.compare)
            [ "IHDR"; "pHYs"; "sRGB" ]
            (before_idat (chunks png)));
      prop
        "with ~dpi and ~srgb the file is that of encode_png with the two \
         chunks added, and loads back the same"
        (Gen.pair (images [ 0; 1; 3; 4 ]) (Gen.float_range 1. 1000.))
        (fun (t, dpi) ->
          let plain = Nx_io.encode_png t in
          let png = Nx_io.encode_png ~dpi ~srgb:true t in
          equal string plain (without "sRGB" (without "pHYs" png));
          equal (tensor int)
            (Nx_io.load_image (file ".png" plain))
            (Nx_io.load_image (file ".png" png)));
      cases ~name:fst "~dpi outside the range of a PNG integer is refused"
        [
          ("zero", 0.);
          ("a negative", -72.);
          ("one rounding to zero pixels per metre", 0.0126);
          ("one rounding past the largest PNG integer", 2147483648. *. 0.0254);
          ("an infinity", infinity);
          ("nan", nan);
        ]
        (fun (_, dpi) ->
          raises_match (Exn.invalid_arg ~substring:"dpi") (fun () ->
              ignore (Nx_io.encode_png ~dpi tiny)));
    ]

let images_group =
  group "images"
    [
      prop
        "save_image writes the bytes of encode_png, which load back exactly, \
         gray in three equal channels, alpha dropped, and colour in grayscale \
         within one level of 0.299 R + 0.587 G + 0.114 B"
        (images [ 0; 1; 3; 4 ])
        (fun t ->
          let path = temp_file ~suffix:".png" () in
          Nx_io.save_image path t;
          equal string (Nx_io.encode_png t) (read path);
          let colour = Nx_io.load_image path in
          let gray = Nx_io.load_image ~grayscale:true path in
          let floats t = Ref.map Float.of_int (Ref.of_nx t) in
          match Nx.shape t with
          | [| h; w; 3 |] ->
              equal (tensor int) t colour;
              let luma p =
                [| (0.299 *. p.(0)) +. (0.587 *. p.(1)) +. (0.114 *. p.(2)) |]
              in
              equal
                (Ref.witness (close ~abs:1. ~rel:0. ()))
                (Ref.along ~axis:2 ~length:1 luma (floats t))
                (floats (Nx.reshape [| h; w; 1 |] gray))
          | [| _; _; 4 |] ->
              equal (tensor int) (Nx.slice [ A; A; R (0, 3) ] t) colour
          | [| h; w |] | [| h; w; _ |] ->
              equal (tensor int) (flat t) gray;
              equal (tensor int)
                (Nx.broadcast_to [| h; w; 3 |] (Nx.reshape [| h; w; 1 |] gray))
                colour
          | _ -> assert false);
      prop
        "a smooth image returns from JPEG at quality 90 within a mean error of \
         8 levels"
        (smooth [ 0; 1; 3 ])
        (fun t ->
          let path = temp_file ~suffix:".jpg" () in
          Nx_io.save_image path t;
          let colour = Nx.shape t = [| Nx.dim 0 t; Nx.dim 1 t; 3 |] in
          let back = Nx_io.load_image ~grayscale:(not colour) path in
          let t = if colour then t else flat t in
          equal (array int) (Nx.shape t) (Nx.shape back);
          let e = errors t back in
          let mean = Array.fold_left ( + ) 0 e / Array.length e in
          at_most int ~than:8 mean);
      cases ~name:fst "files other encoders wrote load as Pillow decodes them"
        (List.map
           (fun n -> ("png_" ^ n ^ ".png", 0))
           [ "filters"; "gray1"; "rgb16"; "palette_adam7" ]
        @ List.map
            (fun n -> ("jpeg_" ^ n ^ ".jpg", 3))
            [ "baseline"; "progressive"; "gray"; "cmyk"; "restart" ])
        (fun (name, tolerance) ->
          let expected = pillow name in
          let grayscale = Nx.ndim expected = 2 in
          let t = Nx_io.load_image ~grayscale (fixture name) in
          equal ~msg:"shape" (array int) (Nx.shape expected) (Nx.shape t);
          at_most ~msg:"largest error" int ~than:tolerance
            (Array.fold_left Int.max 0 (errors expected t)));
      cases ~name:Fun.id
        "the extension selects the encoder, whatever its case, and the \
         contents the decoder, whatever the extension"
        [ ".png"; ".PNG"; ".jpg"; ".JPG"; ".jpeg"; ".JpEg" ] (fun ext ->
          let path = temp_file ~suffix:ext () in
          Nx_io.save_image path (Nx.full Nx.uint8 [| 4; 4; 3 |] 7);
          let png = String.lowercase_ascii ext = ".png" in
          starts_with ~affix:(if png then "\137PNG" else "\xff\xd8") (read path);
          let other = file (if png then ".jpg" else ".png") (read path) in
          equal (tensor int) (Nx_io.load_image path) (Nx_io.load_image other));
      cases
        ~name:(fun (name, _, _) -> name)
        "an extension or a shape the format does not hold is refused"
        [
          ("a BMP extension", "x.bmp", [| 2; 2; 3 |]);
          ("no extension", "x", [| 2; 2; 3 |]);
          ("a vector", "x.png", [| 4 |]);
          ("two channels", "x.png", [| 2; 2; 2 |]);
          ("no rows", "x.png", [| 0; 3; 3 |]);
          ("four channels in JPEG", "x.jpg", [| 2; 2; 4 |]);
          ("no columns in JPEG", "x.jpg", [| 3; 0 |]);
        ]
        (fun (_, name, shape) ->
          refuse_image name shape;
          if name = "x.png" then
            fails (fun () -> Nx_io.encode_png (Nx.zeros Nx.uint8 shape)));
      test "a PNG chunk whose checksum does not match fails" (fun () ->
          let png = flip_byte (read (fixture "png_filters.png")) 30 in
          fails ~naming:"PNG chunk CRC mismatch" (fun () ->
              Nx_io.load_image (file "" png)));
      test "an image of noise, which deflate stores uncompressed, round trips"
        (fun () ->
          let noise =
            Nx.init Nx.uint8 [| 64; 64; 3 |] (fun _ -> Random.int 256)
          in
          equal (tensor int) noise
            (Nx_io.load_image (saved ".png" Nx_io.save_image noise)));
    ]

(* Every format *)

(* A format's writer of a fixed value and its reader, a file it reads, and how
   it refuses a path: [Failure], or the [Unix.Unix_error] it is given. *)
type format = {
  name : string;
  suffix : string;
  save : ?overwrite:bool -> string -> unit;
  load : string -> unit;
  sample : unit -> string;
  refused : Unix.error -> (unit -> unit) -> unit;
  overwrite : bool;  (** [save] takes [~overwrite] *)
  binary : bool;  (** text and a prefix of a file are no file of it *)
}

let format ?(suffix = "") ?(overwrite = true) ?(binary = true)
    ?(refused = fun _ f -> fails f) ?sample name
    (save : ?overwrite:bool -> string -> unit) load =
  let sample =
    match sample with
    | Some sample -> sample
    | None -> fun () -> read (saved suffix (fun path () -> save path) ())
  in
  { name; suffix; save; load; sample; refused; overwrite; binary }

let formats =
  let v = Nx.arange Nx.int32 0 2 1 and gray = Nx.zeros Nx.uint8 [| 2; 2 |] in
  let image suffix =
    format ~suffix ~refused:unix_error suffix
      (fun ?overwrite path -> Nx_io.save_image ?overwrite path gray)
      (fun path -> ignore (Nx_io.load_image path))
  in
  [
    format "npy"
      (fun ?overwrite p -> Nx_io.save_npy ?overwrite p v)
      (fun p -> ignore (Nx_io.load_npy p));
    format "npz"
      (fun ?overwrite p ->
        Nx_io.save_npz ?overwrite p (Nx_io.Archive.of_list [ ("v", Nx.P v) ]))
      (fun p -> ignore (Nx_io.load_npz p));
    format "safetensors"
      (fun ?overwrite p ->
        Nx_io.save_safetensors ?overwrite p
          (Nx_io.Archive.of_list [ ("v", Nx.P v) ]))
      (fun p -> ignore (Nx_io.load_safetensors p));
    format ~overwrite:false ~binary:false "text"
      (fun ?overwrite:_ p -> Nx_io.save_txt p v)
      (fun p -> ignore (Nx_io.load_txt p Nx.int32));
    image ".png";
    image ".jpg";
    format ~overwrite:false ~refused:unix_error
      ~sample:(fun () -> read (fixture "hello.gz"))
      "gunzip"
      (fun ?overwrite:_ dst -> Nx_io.gunzip ~src:(fixture "hello.gz") ~dst)
      (fun src -> Nx_io.gunzip ~src ~dst:(temp_file ()));
  ]

(* The dtypes each format refuses, which text refuses to load as well. A binary
   format leaves the file it was to replace as it was, and nothing beside it,
   which nx_io.mli states for SafeTensors. *)
let refusals =
  let z dtype = Nx.P (Nx.zeros dtype [| 2 |]) in
  let bf16 = [ z Nx.bfloat16 ]
  and f8 = [ z Nx.float8_e4m3; z Nx.float8_e5m2 ] in
  let complex = [ z Nx.complex64; z Nx.complex128 ] in
  let int4 = [ z Nx.int4; z Nx.uint4 ] in
  let saves ?naming save p =
    let dir = temp_dir () in
    let path = Filename.concat dir "old" in
    write path "old";
    fails ?naming (fun () -> save path p);
    equal ~msg:"the file" string "old" (read path);
    equal ~msg:"the directory" (array string) [| "old" |] (Sys.readdir dir)
  in
  let text (Nx.P t) =
    fails (fun () -> Nx_io.save_txt (temp_file ()) t);
    fails (fun () -> Nx_io.load_txt (file "" "0 0\n") (Nx.dtype t))
  in
  List.concat_map
    (fun (format, dtypes, check) ->
      List.map
        (fun (Nx.P t as p) ->
          let dtype = Nx_dtype.to_string (Nx.dtype t) in
          (format ^ " refuses " ^ dtype, fun () -> check p))
        dtypes)
    [
      ("npy", bf16 @ f8 @ int4, saves save_npy);
      ( "npz",
        bf16 @ f8 @ int4,
        saves ~naming:"entry w" (fun path p -> save_npz path [ ("entry w", p) ])
      );
      ("safetensors", complex @ int4, saves ~naming:"t" save_safetensors);
      ("text", f8 @ complex @ int4, text);
    ]

let every_format =
  let name f = f.name in
  let with_overwrite = List.filter (fun f -> f.overwrite) formats in
  group "every format"
    [
      cases ~name "a missing file or directory is refused" formats (fun f ->
          let path = Filename.concat (missing "d") ("x" ^ f.suffix) in
          f.refused ENOENT (fun () -> f.load path);
          f.refused ENOENT (fun () -> f.save path));
      cases ~name "a save replaces an existing file" formats (fun f ->
          let path = file f.suffix "old" and fresh = missing ("x" ^ f.suffix) in
          f.save path;
          f.save fresh;
          equal string (read fresh) (read path));
      cases ~name
        "~overwrite:false writes a new file and refuses an existing one, \
         leaving it as it was"
        with_overwrite (fun f ->
          let path = missing ("x" ^ f.suffix) in
          f.save ~overwrite:false path;
          f.load path;
          let old = file f.suffix "old" in
          f.refused EEXIST (fun () -> f.save ~overwrite:false old);
          equal string "old" (read old));
      cases ~name "an empty file, a text file or a file cut short fails to load"
        formats (fun f ->
          let s = f.sample () in
          let damaged =
            if f.binary then
              [ ""; "hello, nx\n"; String.sub s 0 (String.length s / 2) ]
            else [ "" ]
          in
          List.iter
            (fun c -> fails (fun () -> f.load (file f.suffix c)))
            damaged);
      cases ~name:fst
        "a dtype the format does not hold is refused, the previous file kept"
        refusals (fun (_, check) -> check ());
    ]

(* Malformed streams *)

(* [f ()] returns or raises [Failure], as the decoders promise for a stream that
   is not theirs: never another exception, a crash or a read out of bounds. *)
let returns_or_fails f = match f () with _ -> () | exception Failure _ -> ()

(* A stream damaged at a drawn position: one bit flipped there, or cut short
   there. *)
type damage = Flip of int * int | Cut of int

let pp_damage ppf = function
  | Flip (at, bit) -> Format.fprintf ppf "Flip (%d, %d)" at bit
  | Cut at -> Format.fprintf ppf "Cut %d" at

let damage n =
  Gen.with_pp pp_damage
    (Gen.map
       (fun (at, bit, flip) -> if flip then Flip (at, bit) else Cut at)
       (Gen.triple (Gen.int_range 0 (n - 1)) (Gen.int_range 0 7) Gen.bool))

let damaged s = function
  | Flip (at, bit) ->
      String.mapi
        (fun j c ->
          if j = at then Char.chr (Char.code c lxor (1 lsl bit)) else c)
        s
  | Cut at -> String.sub s 0 at

let bytes = Gen.string_of ~size:(Gen.int_range 0 512) Gen.char

let pixels =
  Nx.init Nx.uint8 [| 13; 17; 3 |] (fun i ->
      ((((i.(0) * 17) + i.(1)) * 3) + i.(2)) * 37 mod 256)

(* A PNG and a JPEG of [pixels], made once. *)
let png = lazy (Nx_io.encode_png pixels)
let jpeg = lazy (read (saved ".jpg" Nx_io.save_image pixels))

let damaged_image name stream suffix =
  prop
    ("load_image of a damaged " ^ name ^ " loads or fails")
    (Gen.bind Gen.unit (fun () -> damage (String.length (Lazy.force stream))))
    (fun d ->
      returns_or_fails (fun () ->
          Nx_io.load_image (file suffix (damaged (Lazy.force stream) d))))

let malformed =
  group "malformed streams"
    [
      prop "gunzip of a gzip header and any bytes decompresses or fails" bytes
        (fun b ->
          let src =
            file ".gz" ("\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff" ^ b)
          in
          returns_or_fails (fun () -> Nx_io.gunzip ~src ~dst:(temp_file ())));
      test
        "an npz entry of zeros then noise, deflated with a stored block, loads \
         back" (fun () ->
          let t =
            Nx.init Nx.uint8 [| 300_000 |] (fun i ->
                if i.(0) < 150_000 then 0 else Random.int 256)
          in
          let path = saved "" save_npz [ ("w", Nx.P t) ] in
          equal int 8 (String.get_uint16_le (read path) 8);
          equal packed (Nx.P t) (Nx_io.load_npz_entry ~name:"w" path));
      damaged_image "PNG" png ".png";
      damaged_image "JPEG" jpeg ".jpg";
      prop "load_image of an image signature and any bytes loads or fails"
        (Gen.pair Gen.bool bytes) (fun (is_png, b) ->
          let signature = if is_png then "\137PNG\r\n\026\n" else "\xff\xd8" in
          returns_or_fails (fun () ->
              Nx_io.load_image (file "" (signature ^ b))));
    ]

let () =
  exit
    (run "nx.io"
       [
         npy;
         npz;
         compression;
         safetensors;
         gguf_group;
         txt;
         images_group;
         png_chunks_group;
         every_format;
         malformed;
       ])
