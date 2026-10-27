(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* SafeTensors. An archive saved as SafeTensors loads as itself, bit for bit,
   over every dtype the format holds, its entries ordered as the reference
   implementation orders them; files numpy and torch wrote load and save again
   as they are; an entry of a dtype nx lacks loads as its bytes; an aligned
   entry loads over the file's pages and a misaligned one into memory of its
   own; a malformed file raises [Failure] naming it, and never anything else. *)

open Windtrap
open Io_support

let save_safetensors path l = Nx_io.save_safetensors path (Archive.of_list l)

let with_header header =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int (String.length header));
  Bytes.to_string b ^ header

(* [with_json json data] is the SafeTensors file of the header [json], padded to
   a multiple of 8 bytes, and the data [data]. *)
let with_json json data =
  with_header (json ^ String.make ((8 - (String.length json mod 8)) mod 8) ' ')
  ^ data

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

(* Entries of every dtype, each with the dtype nx loads it at, in descending
   element size, so that with [pad = 0] each sits on a multiple of its element
   size and with [pad = 1] none of the wide ones does. *)
let typed_entries =
  [
    ("u64", "U64", "uint64", [ 2 ], pattern 16 1);
    ("c64", "C64", "complex64", [ 2 ], pattern 16 11);
    ("u32", "U32", "uint32", [ 2; 2 ], pattern 16 2);
    ("bf16", "BF16", "bfloat16", [ 3 ], pattern 6 3);
    ("f16", "F16", "float16", [ 1 ], pattern 2 4);
    ("f8_e4m3", "F8_E4M3", "float8_e4m3fn", [ 3 ], pattern 3 5);
    ("f8_e5m2", "F8_E5M2", "float8_e5m2", [ 3 ], pattern 3 6);
    ("bool", "BOOL", "bool", [ 4 ], "\000\001\002\255");
    ("empty", "F32", "float32", [ 0; 3 ], "");
    ("scales", "F8_E8M0", "uint8", [ 2; 3 ], pattern 6 7);
    ("e4m3fnuz", "F8_E4M3FNUZ", "uint8", [ 3 ], pattern 3 12);
    ("e5m2fnuz", "F8_E5M2FNUZ", "uint8", [ 1; 2 ], pattern 2 13);
    ("nibbles", "F4", "uint8", [ 2; 4 ], pattern 4 8);
    ("sixes", "F6_E2M3", "uint8", [ 4 ], pattern 3 9);
    ("other sixes", "F6_E3M2", "uint8", [ 8 ], pattern 6 10);
  ]

let typed_file ?pad () =
  raw_safetensors ?pad
    (List.map
       (fun (n, tag, _, shape, data) -> (n, tag, shape, data))
       typed_entries)

let payload path =
  let s = read path in
  let start = 8 + Int64.to_int (String.get_int64_le s 0) in
  String.sub s start (String.length s - start)

let u8_entry name = raw_safetensors [ (name, "U8", [ 1 ], "\042") ]

(* Names a JSON string must escape, or may: a tab, a newline, a quote, a
   backslash, a slash, control bytes, and characters past ASCII. *)
let escaped_names =
  [ "a\tb"; "a\nb"; "say \"w\""; "a\\b"; "a/b"; "\001\031"; "é🚀" ]

let names = entry_names @ escaped_names

let safetensors =
  group "safetensors"
    [
      prop "an archive saved as SafeTensors loads as itself"
        (Gen.with_pp pp_entries (archives ~names safetensors_dtypes))
        (fun entries ->
          List.iter (fun (_, c) -> covers c) entries;
          cover "an empty archive" (entries = []);
          cover "a name JSON escapes"
            (List.exists (fun (n, _) -> List.mem n escaped_names) entries);
          Law.round_trip archive string
            (saved ".safetensors" Nx_io.save_safetensors)
            Nx_io.load_safetensors (archive_of entries));
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
      test
        "a complex64 file torch wrote loads bit for bit, and saves again as it \
         is" (fun () ->
          let path = fixture "c64_bit_exact.safetensors" in
          let p = find (Nx_io.load_safetensors path) "c64_tensor" in
          let bits =
            [
              0x3F800000;
              0x40000000;
              0x80000000;
              0xFF800000;
              0x7FC00000;
              0x3F000000;
            ]
          in
          let le32 n =
            String.init 4 (fun i -> Char.chr ((n lsr (8 * i)) land 0xff))
          in
          equal
            (triple string (array int) string)
            ("complex64", [| 3 |], String.concat "" (List.map le32 bits))
            (stored p);
          let again = temp_file () in
          save_safetensors again [ ("c64_tensor", p) ];
          equal string (payload path) (payload again));
      test "the FNUZ float8 entries torch wrote load as their bytes" (fun () ->
          let a = Nx_io.load_safetensors (fixture "fnuz.safetensors") in
          List.iter
            (fun name ->
              equal ~msg:name
                (triple string (array int) string)
                ("uint8", [| 4 |], "\000\001\128\127")
                (stored (find a name)))
            [ "e4m3fnuz"; "e5m2fnuz" ]);
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
      cases ~name:fst
        "an aligned entry loads over the file's pages, and a misaligned one \
         into memory of its own"
        [ ("aligned", (0, Over_file)); ("one byte past", (1, Own_memory)) ]
        (fun (_, (pad, expected)) ->
          let u32 =
            find (Nx_io.load_safetensors (file (typed_file ~pad ()))) "u32"
          in
          equal memory expected (memory_of u32));
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
            let y, grew = grown Rig.host load (fun x -> Nx.neg (Nx.donate x)) in
            equal (array float_exact) (Array.make n (-1.)) (Nx.to_array y);
            grew
          in
          less ~msg:"misaligned: no memory of the result's size" int
            ~than:(4 * n) (negated 1);
          at_least ~msg:"aligned: a fresh result" int ~than:(4 * n) (negated 0));
      cases ~name:fst "a header's JSON string escapes are decoded"
        [
          ("short escapes", ({|aé🚀\"\\\/\b\f\r\n\t|}, "aé🚀\"\\/\b\012\r\n\t"));
          ("\\u escapes", ({|\u0061\u00e9\u00E9|}, "aéé"));
          ("a surrogate pair", ({|\uD83D\uDE80\ud83d\ude80|}, "🚀🚀"));
        ]
        (fun (_, (json, name)) ->
          equal (list string) [ name ]
            (Archive.names (Nx_io.load_safetensors (file (u8_entry json)))));
      cases ~name:fst
        "a name that is no JSON string or no Unicode text fails, naming the \
         file"
        [
          ("a lone high surrogate", {|\ud800|});
          ("a lone low surrogate", {|\udc00|});
          ("a high surrogate before a letter", {|\ud800A|});
          ("a high surrogate before another escape", {|\ud800\u0041|});
          ("a \\u escape of no hexadecimal", {|\u12xz|});
          ("an unknown escape", {|\q|});
          ("a raw control byte", "a\001b");
          ("a byte of no UTF-8 character", "a\xffb");
        ]
        (fun (_, json) ->
          let path = file (u8_entry json) in
          fails ~path (fun () -> Nx_io.load_safetensors path));
      test
        "an empty entry at a non-empty one's offset loads, whichever the \
         header lists first" (fun () ->
          let json =
            {|{"w":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"e":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}|}
          in
          equal archive
            (Archive.of_list
               [
                 ("e", Nx.P (Nx.zeros Nx.uint8 [| 0 |]));
                 ("w", Nx.P (Nx.create Nx.uint8 [| 2 |] [| 1; 2 |]));
               ])
            (Nx_io.load_safetensors (file (with_json json "\001\002"))));
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
                ("e", Nx.P (Nx.full Nx.complex64 [| 1 |] { re = 2.; im = 0.5 }));
              ]
          in
          (* int64 2, float64 1, complex64 2 + 0.5i, float32 3, uint8 0. *)
          equal string
            "\002\000\000\000\000\000\000\000\000\000\000\000\000\000\240\063\000\000\000\064\000\000\000\063\000\000\064\064\000"
            (payload path));
      cases
        ~name:(fun (D.Any dt) -> D.name dt)
        "a dtype SafeTensors lacks is refused, naming the entry"
        D.[ Any Complex128; Any Int4; Any Uint4; Any Bit; Any Float4_e2m1fn ]
        (fun (D.Any dt) ->
          raises_match
            (Exn.invalid_arg ~substring:"Nx_io.save_safetensors: w: ")
            (fun () ->
              save_safetensors (temp_file ())
                [ ("w", Nx.P (Nx.zeros dt [| 2 |])) ]));
      cases ~name:fst
        "a name the format's reader refuses is refused, and nothing is written"
        [
          ("a name that is not UTF-8", "a\xffb");
          ("the metadata's name", "__metadata__");
        ]
        (fun (_, name) ->
          let dir = temp_dir () in
          let path = Filename.concat dir "x.safetensors" in
          raises_match
            (Exn.invalid_arg
               ~substring:(Printf.sprintf "Nx_io.save_safetensors: %S" name))
            (fun () ->
              save_safetensors path [ (name, Nx.P (Nx.zeros Nx.uint8 [| 1 |])) ]);
          equal (array string) [||] (Sys.readdir dir));
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
       cases ~name:fst "a malformed file fails, naming it"
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
           ( "a field twice in an entry",
             fun () ->
               file
                 (with_json
                    {|{"w":{"dtype":"U8","dtype":"I8","shape":[1],"data_offsets":[0,1]}}|}
                    "a") );
           ( "an empty name",
             fun () -> file (raw_safetensors [ ("", "U8", [ 1 ], "a") ]) );
           ( "an unknown dtype",
             fun () ->
               file (raw_safetensors [ ("w", "C128", [ 1 ], "abcdefgh") ]) );
           ( "a size that is not the shape's",
             fun () -> file (raw_safetensors [ ("w", "U16", [ 1 ], "a") ]) );
           ( "a negative extent",
             fun () -> file (raw_safetensors [ ("w", "U8", [ -1 ], "") ]) );
           ("a text file", fun () -> file "hello, nx\n");
           ("an npy file", fun () -> file (read (fixture "version2.npy")));
         ]
         (fun (_, path) ->
           let path = path () in
           fails ~path (fun () -> Nx_io.load_safetensors path)));
    ]

let () = exit (run "nx.io SafeTensors" [ safetensors ])
