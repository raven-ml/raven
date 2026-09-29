(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tensor I/O. Each format reads back what it writes, bit for bit where it is
   lossless, over every dtype it holds, every layout and the bit patterns that
   break codecs: NaN payloads, infinities, signed zeros and subnormals. Files
   that other tools wrote, in fixtures/ with generate.py, pin the formats
   themselves. *)

open Windtrap
open Nx_test

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
let fixture name = Filename.concat "fixtures" name
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

(* Tensors compare bit for bit: dtype, shape and the bytes of the elements in
   row-major order. *)

let storage (Nx.P t) =
  let bytes = Bytes.create (Nx.nbytes t) in
  Nx_buffer.blit_to_bytes (Nx.to_buffer t) bytes;
  (Nx_dtype.to_string (Nx.dtype t), Nx.shape t, Bytes.unsafe_to_string bytes)

let pp_packed ppf (Nx.P t as p) =
  let _, _, bytes = storage p in
  Format.fprintf ppf "%a (bytes %S)" Nx.pp t bytes

let packed =
  Testable.make ~pp:pp_packed ~equal:(fun a b -> storage a = storage b)

(* An archive compares as a set of named tensors. *)
let entries = slist (pair string packed) (fun (a, _) (b, _) -> compare a b)
let listed archive = List.of_seq (Hashtbl.to_seq archive)

(* Dtypes *)

(* A dtype, tensors of it under every layout, and the witness of their values
   that a text file keeps: every NaN equal to every NaN. *)
type case =
  | Case : {
      name : string;
      dtype : ('a, 'b) Nx.dtype;
      tensors : ('a, 'b) Nx.t Gen.t;
      values : ('a, 'b) Nx.t testable;
    }
      -> case

let case name dtype tensors values = Case { name; dtype; tensors; values }
let shape = Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 4)
let pp_bits ppf v = Format.fprintf ppf "0x%Lx" v
let ones n = Int64.pred (Int64.shift_left 1L n)

(* The bit patterns of a float format with [e] exponent and [m] significand
   bits: signed zeros, the least and greatest subnormals, the least normal, the
   greatest finite, infinities, quiet and signalling NaNs with payloads, and any
   pattern. *)
let float_bits ~e ~m =
  let inf = Int64.shift_left (ones e) m in
  let nans =
    [ Int64.logor inf (Int64.shift_left 1L (m - 1)); Int64.succ inf ]
  in
  let corners =
    [ 0L; 1L; ones m; Int64.shift_left 1L m; Int64.pred inf; inf ] @ nans
  in
  let sign = Int64.logor (Int64.shift_left 1L (e + m)) in
  let mask = if e + m = 63 then -1L else ones (1 + e + m) in
  Gen.frequency
    [
      (1, Gen.of_list ~pp:pp_bits (corners @ List.map sign corners));
      (2, Gen.with_pp pp_bits (Gen.map (Int64.logand mask) Gen.int64));
    ]

(* Tensors of [dtype] whose elements are the bits [bits] draws, read through the
   integer dtype [name] of the same width. *)
let from_bits name dtype bits =
  match List.find (fun (Int_dtype d) -> d.name = name) int_dtypes with
  | Int_dtype u ->
      let drawn =
        let open Gen in
        let* s = shape in
        let* steps = layout in
        let+ xs = array ~size:(constant (Ref.numel s)) bits in
        (steps, Ref.create s xs)
      in
      let pp ppf (steps, r) =
        Format.fprintf ppf "%a: %a" pp_layout steps (Ref.pp pp_bits) r
      in
      Gen.map
        (fun (steps, (r : int64 Ref.t)) ->
          let words = Nx.create u.dtype r.shape (Array.map u.of_i64 r.data) in
          lay_out steps (Nx.bitcast dtype words))
        (Gen.with_pp pp drawn)

let float_tensors dtype ~e ~m =
  from_bits ("uint" ^ string_of_int (1 + e + m)) dtype (float_bits ~e ~m)

let float_case name dtype ~e ~m =
  case name dtype (float_tensors dtype ~e ~m) (tensor float_exact)

let ints =
  List.map
    (fun (Int_dtype d) ->
      let value = int_value ~bits:d.bits ~signed:d.signed in
      case d.name d.dtype (from_bits d.name d.dtype value) (tensor d.exact))
    int_dtypes

let bool =
  let pp = Format.pp_print_bool in
  case "bool" Nx.bool (viewed ~pp Nx.bool Gen.bool) (tensor bool)

let float16 = float_case "float16" Nx.float16 ~e:5 ~m:10
let bfloat16 = float_case "bfloat16" Nx.bfloat16 ~e:8 ~m:7
let float32 = float_case "float32" Nx.float32 ~e:8 ~m:23
let float64s = float_tensors Nx.float64 ~e:11 ~m:52
let float64 = case "float64" Nx.float64 float64s (tensor float_exact)
let float8_e4m3 = float_case "float8_e4m3" Nx.float8_e4m3 ~e:4 ~m:3
let float8_e5m2 = float_case "float8_e5m2" Nx.float8_e5m2 ~e:5 ~m:2

let complex_exact =
  Testable.contramap
    (fun (c : Complex.t) -> (c.re, c.im))
    (pair float_exact float_exact)

(* A complex64 element is two float32 bit patterns, real part first. *)
let complex64 =
  let f32 = float_bits ~e:8 ~m:23 in
  let word (re, im) = Int64.logor (Int64.shift_left im 32) re in
  let tensors =
    from_bits "int64" Nx.complex64 (Gen.map word (Gen.pair f32 f32))
  in
  case "complex64" Nx.complex64 tensors (tensor complex_exact)

let complex128 =
  let pp ppf (c : Complex.t) = Format.fprintf ppf "%h%+hi" c.re c.im in
  let value (re, im) = { Complex.re; im } in
  let values = Gen.map value (Gen.pair Gen.any_float Gen.any_float) in
  case "complex128" Nx.complex128
    (viewed ~pp Nx.complex128 values)
    (tensor complex_exact)

let round_trips ~save ~load cases =
  List.map
    (fun (Case c) ->
      prop (c.name ^ " tensors round trip bit for bit") c.tensors (fun t ->
          cover "an empty tensor" (Nx.numel t = 0);
          cover "a scalar" (Nx.ndim t = 0);
          cover "a view" ((not (Nx.is_c_contiguous t)) || Nx.offset t <> 0);
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
let save_npz path l = Nx_io.save_npz path l

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
          Hashtbl.iter entry archive;
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
        [ ""; "/w"; "a//b"; "./w"; "a/../b"; "\xff" ] (fun name ->
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
      test "an entry whose checksum does not match fails" (fun () ->
          let s = read (saved "" save_npz [ ("w", one) ]) in
          (* The CRC-32 of the central directory's first entry. *)
          let rec crc i =
            if String.sub s i 4 = "PK\001\002" then i + 16 else crc (i + 1)
          in
          let path = file "" (flip_byte s (crc 0)) in
          fails (fun () -> Nx_io.load_npz_entry ~name:"w" path));
    ]

(* Compression *)

let crc32 s =
  let crc = ref 0xFFFFFFFF in
  String.iter
    (fun c ->
      crc := !crc lxor Char.code c;
      for _ = 1 to 8 do
        crc := (!crc lsr 1) lxor (!crc land 1 * 0xEDB88320)
      done)
    s;
  !crc lxor 0xFFFFFFFF

(* A gzip member around the DEFLATE data of [deflate s]. *)
let gzip s =
  let z = Nx_io.deflate s in
  "\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"
  ^ String.sub z 2 (String.length z - 6)
  ^ le32 (crc32 s)
  ^ le32 (String.length s land 0xFFFFFFFF)

let gunzipped src =
  let dst = temp_file () in
  Nx_io.gunzip ~src ~dst;
  read dst

let lines = String.concat "" (List.init 200 (Printf.sprintf "line %d\n"))

let compression =
  group "compression"
    [
      prop
        "inflate inverts deflate, whose zlib frame and Adler-32 it checks, on \
         short strings and on long ones past the window and block sizes"
        (let long = Gen.int_range 60_000 200_000 in
         Gen.frequency
           [
             (18, Gen.string);
             (1, Gen.string_of ~size:long (Gen.char_range 'a' 'd'));
             (1, Gen.string_of ~size:long Gen.char);
           ])
        (Law.round_trip string string Nx_io.deflate Nx_io.inflate);
      cases ~name:fst "inflate reads streams written by Python's zlib"
        [
          ("empty", "");
          ("fixed", "hello, nx zlib!\n");
          ("stored", "stored block\n");
          ("dynamic", lines);
        ]
        (fun (name, data) ->
          let z = read (fixture ("zlib_" ^ name ^ ".z")) in
          equal string data (Nx_io.inflate z));
      cases ~name:fst "inflate refuses what is no zlib stream"
        [
          ("the empty string", fun _ -> "");
          ("text", fun _ -> "hello");
          ( "a checksum that does not match",
            fun z -> flip_byte z (String.length z - 1) );
          ("a stream cut short", fun z -> cut z 5);
        ]
        (fun (_, damage) ->
          let z = read (fixture "zlib_dynamic.z") in
          fails (fun () -> Nx_io.inflate (damage z)));
      prop "gunzip decompresses gzip members to their concatenation"
        ~examples:[ [ "\144\144\144\144" ] ]
        (Gen.list ~size:(Gen.int_range 1 4) Gen.string)
        (fun l ->
          let members = String.concat "" (List.map gzip l) in
          equal string (String.concat "" l) (gunzipped (file "" members)));
      test "gunzip decompresses a file written by Python's gzip" (fun () ->
          equal string "hello, nx gzip!\n" (gunzipped (fixture "hello.gz")));
      cases ~name:fst
        "a malformed member fails and leaves the destination as it was"
        [
          ("a checksum", fun s -> flip_byte s (String.length s - 8));
          ("a partial member after it", fun s -> s ^ String.sub s 0 12);
          ("a size field", fun s -> cut s 4 ^ le32 99);
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

let save_safetensors path p = Nx_io.save_safetensors path [ ("t", p) ]

(* Not inlined, so that once it returns nothing but its result keeps the file
   mapped. *)
let[@inline never] load_entry path name =
  Hashtbl.find (Nx_io.load_safetensors path) name

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
        (storage (Hashtbl.find archive name)))
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
  let file_range archive name =
    let (Nx.P t) = Hashtbl.find archive name in
    Option.map
      (fun ((file : Nx_buffer.file), offset) -> (file.path, offset))
      (Nx_buffer.file_range (Nx.to_buffer t))
  in
  group "safetensors"
    [
      group "round trip"
        (round_trips ~save:save_safetensors
           ~load:(fun path -> load_entry path "t")
           safetensors_cases);
      archive_round_trip
        ~save:(fun path l -> Nx_io.save_safetensors path l)
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
          Nx_io.save_safetensors again [ (entry, p) ];
          equal string (payload path) (payload again));
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
        "an entry of any dtype loads its bytes as stored, a view of the file \
         when aligned and a copy otherwise" (fun () ->
          let path = file "" (typed_file ()) in
          loads_typed_entries path;
          let start =
            String.length (read path) - String.length (payload path)
          in
          let archive = Nx_io.load_safetensors path in
          let range = option (pair string int) in
          equal ~msg:"u64" range (Some (path, start)) (file_range archive "u64");
          equal ~msg:"bf16" range
            (Some (path, start + 32))
            (file_range archive "bf16");
          let path = file "" (typed_file ~pad:1 ()) in
          loads_typed_entries path;
          equal ~msg:"a copy" range None
            (file_range (Nx_io.load_safetensors path) "u64"));
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
         outliving their archive unchanged" (fun () ->
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
          equal ~msg:"the tensor loaded first" packed first loaded);
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
        "a smooth image returns from JPEG within a mean error of 8 levels \
         (nx_io.mli states no quality)"
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
          fails (fun () -> Nx_io.load_image (file "" png)));
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
      (fun ?overwrite p -> Nx_io.save_npz ?overwrite p [ ("v", Nx.P v) ])
      (fun p -> ignore (Nx_io.load_npz p));
    format "safetensors"
      (fun ?overwrite p ->
        Nx_io.save_safetensors ?overwrite p [ ("v", Nx.P v) ])
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
  let saves save p =
    let dir = temp_dir () in
    let path = Filename.concat dir "old" in
    write path "old";
    fails (fun () -> save path p);
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
      ("safetensors", complex @ int4, saves save_safetensors);
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
      cases ~name:fst "an archive refuses a name given twice"
        [
          ("npz", save_npz);
          ("safetensors", fun path l -> Nx_io.save_safetensors path l);
        ]
        (fun (_, save) ->
          let p = Nx.P (Nx.zeros Nx.int8 [| 1 |]) in
          fails (fun () -> save (temp_file ()) [ ("w", p); ("w", p) ]));
    ]

let () =
  exit
    (run "nx.io"
       [ npy; npz; compression; safetensors; txt; images_group; every_format ])
