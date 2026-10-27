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
         ]
         (fun (_, path) ->
           let path = path () in
           fails ~path (fun () -> Nx_io.load_safetensors path)));
    ]

let () = exit (run "nx.io SafeTensors" [ safetensors ])
