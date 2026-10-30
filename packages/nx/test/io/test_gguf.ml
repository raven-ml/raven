(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GGUF. A file's metadata of every value type loads as it was written, and its
   arrays as their bytes as stored, a scalar type at its dtype and a block
   format at uint8 with a row's bytes for its last extent, at any alignment of
   the file; a malformed file raises [Failure] naming it, and never anything
   else. *)

open Windtrap
open Io_support
module Gguf = Nx_io.Gguf

let le n bytes =
  String.init bytes (fun i -> Char.chr ((n lsr (8 * i)) land 0xff))

let le64 n =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 n;
  Bytes.to_string b

let u64 n = le64 (Int64.of_int n)
let gstring s = u64 (String.length s) ^ s

(* [cut s n] is [s] without its last [n] bytes. *)
let cut s n = String.sub s 0 (String.length s - n)

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

(* [laid_out ?magic ?version ?alignment ?shift kvs tensors] is a GGUF file of
   the encoded key-values [kvs] and of [tensors], each a name, a type tag, a
   logical shape and its bytes, with the position in the file of each tensor's
   bytes. Each tensor's data starts on a multiple of [alignment], [shift] bytes
   later, and the file ends where the last ends. *)
let laid_out ?(magic = "GGUF") ?(version = 3) ?(alignment = 32) ?(shift = 0) kvs
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
  ( header
    ^ String.make (start - String.length header) '\000'
    ^ Buffer.contents data,
    List.map (( + ) start) offsets )

let gguf ?magic ?version ?alignment ?shift kvs tensors =
  fst (laid_out ?magic ?version ?alignment ?shift kvs tensors)

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

(* [gguf_layout ?version ?alignment ?shift ()] is the file of [metadata] and
   [gguf_tensors], and the position of each tensor's bytes. *)
let gguf_layout ?version ?alignment ?shift () =
  let kvs =
    match alignment with
    | None -> metadata
    | Some a -> ("general.alignment", Gguf.Uint32 a) :: metadata
  in
  laid_out ?version ?alignment ?shift
    (List.map (fun (k, v) -> kv k v) kvs)
    (List.map
       (fun (n, tag, shape, data, _, _, _) -> (n, tag, shape, data))
       gguf_tensors)

let gguf_file ?version ?alignment ?shift () =
  fst (gguf_layout ?version ?alignment ?shift ())

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
      Format.fprintf ppf "@[<hv 2>Array [|%a|]@]"
        (Format.pp_print_seq
           ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ")
           pp_value)
        (Array.to_seq a)

let gguf_value = Testable.structural ~pp:pp_value

(* The tensor types, with their names. *)
let gguf_dtypes =
  Gguf.
    [
      (F32, "F32");
      (F16, "F16");
      (BF16, "BF16");
      (F64, "F64");
      (I8, "I8");
      (I16, "I16");
      (I32, "I32");
      (I64, "I64");
      (Q4_0, "Q4_0");
      (Q4_1, "Q4_1");
      (Q5_0, "Q5_0");
      (Q5_1, "Q5_1");
      (Q8_0, "Q8_0");
      (Q8_1, "Q8_1");
      (Q2_K, "Q2_K");
      (Q3_K, "Q3_K");
      (Q4_K, "Q4_K");
      (Q5_K, "Q5_K");
      (Q6_K, "Q6_K");
      (Q8_K, "Q8_K");
      (IQ2_XXS, "IQ2_XXS");
      (IQ2_XS, "IQ2_XS");
      (IQ3_XXS, "IQ3_XXS");
      (IQ1_S, "IQ1_S");
      (IQ4_NL, "IQ4_NL");
      (IQ3_S, "IQ3_S");
      (IQ2_S, "IQ2_S");
      (IQ4_XS, "IQ4_XS");
      (IQ1_M, "IQ1_M");
      (TQ1_0, "TQ1_0");
      (TQ2_0, "TQ2_0");
      (MXFP4, "MXFP4");
    ]

let gguf_dtype =
  Testable.make ~equal:( = ) ~pp:(fun ppf d ->
      Format.pp_print_string ppf (List.assoc d gguf_dtypes))

let gguf_group =
  group "gguf"
    [
      test "metadata of every value type loads as it was written" (fun () ->
          let g = Nx_io.load_gguf (file (gguf_file ())) in
          equal ~msg:"version" int 3 (Gguf.version g);
          equal (list (pair string gguf_value)) metadata (Gguf.metadata g));
      cases
        "a tensor loads its bytes as stored, at any offset of the file, over \
         the file's pages where the offset is a multiple of its element's size \
         and into memory of its own elsewhere"
        ~id:(fun (v, a, s) ->
          Printf.sprintf "version %d, alignment %s, shift %d" v
            (Option.fold ~none:"default" ~some:string_of_int a)
            s)
        [
          (3, None, 0);
          (2, None, 0);
          (3, Some 64, 0);
          (3, Some 1, 0);
          (3, Some 1, 1);
          (3, Some 1, 2);
          (3, Some 1, 3);
        ]
        (fun (version, alignment, shift) ->
          let contents, positions = gguf_layout ~version ?alignment ~shift () in
          let g = Nx_io.load_gguf (file contents) in
          equal ~msg:"version" int version (Gguf.version g);
          List.iter2
            (fun (name, _, shape, data, dtype, shape', ty) position ->
              let (Nx.P x as t) = find (Gguf.tensors g) name in
              equal ~msg:name
                (triple string (array int) string)
                (dtype, shape', data) (stored t);
              let info = Gguf.info name g in
              equal ~msg:"type" gguf_dtype ty info.dtype;
              equal ~msg:"logical shape" (array int) (Array.of_list shape)
                info.shape;
              let size = Int.max 1 (D.bits (Nx.dtype x) / 8) in
              if data <> "" then
                equal ~msg:(name ^ " lies") memory
                  (if position mod size = 0 then Over_file else Own_memory)
                  (memory_of t))
            gguf_tensors positions);
      cases "a scalar type loads at its dtype"
        ~id:(fun (_, tag, ty, _) ->
          Printf.sprintf "%s, tag %d" (List.assoc ty gguf_dtypes) tag)
        [
          ("f32", 0, Gguf.F32, D.Any D.Float32);
          ("f16", 1, Gguf.F16, D.Any D.Float16);
          ("i8", 24, Gguf.I8, D.Any D.Int8);
          ("i16", 25, Gguf.I16, D.Any D.Int16);
          ("i32", 26, Gguf.I32, D.Any D.Int32);
          ("i64", 27, Gguf.I64, D.Any D.Int64);
          ("f64", 28, Gguf.F64, D.Any D.Float64);
          ("bf16", 30, Gguf.BF16, D.Any D.Bfloat16);
        ]
        (fun (name, tag, ty, D.Any dt) ->
          let data = pattern (D.bytes dt 6) 5 in
          let g =
            Nx_io.load_gguf (file (gguf [] [ (name, tag, [ 2; 3 ], data) ]))
          in
          equal ~msg:"type" gguf_dtype ty (Gguf.info name g).dtype;
          equal
            (triple string (array int) string)
            (D.name dt, [| 2; 3 |], data)
            (stored (find (Gguf.tensors g) name)));
      (* Each block format's elements and bytes per block, as ggml's
         ggml-common.h lays out its block structures, under its ggml_type
         tag. *)
      cases
        "a block format loads as its rows' bytes, at uint8, and refuses a row \
         of no whole number of blocks"
        ~id:(fun (tag, ty, _, _) ->
          Printf.sprintf "%s, tag %d" (List.assoc ty gguf_dtypes) tag)
        [
          (2, Gguf.Q4_0, 32, 18);
          (3, Gguf.Q4_1, 32, 20);
          (6, Gguf.Q5_0, 32, 22);
          (7, Gguf.Q5_1, 32, 24);
          (8, Gguf.Q8_0, 32, 34);
          (9, Gguf.Q8_1, 32, 36);
          (10, Gguf.Q2_K, 256, 84);
          (11, Gguf.Q3_K, 256, 110);
          (12, Gguf.Q4_K, 256, 144);
          (13, Gguf.Q5_K, 256, 176);
          (14, Gguf.Q6_K, 256, 210);
          (15, Gguf.Q8_K, 256, 292);
          (16, Gguf.IQ2_XXS, 256, 66);
          (17, Gguf.IQ2_XS, 256, 74);
          (18, Gguf.IQ3_XXS, 256, 98);
          (19, Gguf.IQ1_S, 256, 50);
          (20, Gguf.IQ4_NL, 32, 18);
          (21, Gguf.IQ3_S, 256, 110);
          (22, Gguf.IQ2_S, 256, 82);
          (23, Gguf.IQ4_XS, 256, 136);
          (29, Gguf.IQ1_M, 256, 56);
          (34, Gguf.TQ1_0, 256, 54);
          (35, Gguf.TQ2_0, 256, 66);
          (39, Gguf.MXFP4, 32, 17);
        ]
        (fun (tag, ty, elements, bytes) ->
          (* Two rows of three blocks. *)
          let data = pattern (6 * bytes) tag in
          let g =
            Nx_io.load_gguf
              (file (gguf [] [ ("w", tag, [ 2; 3 * elements ], data) ]))
          in
          let info = Gguf.info "w" g in
          equal ~msg:"type" gguf_dtype ty info.dtype;
          equal ~msg:"logical shape" (array int)
            [| 2; 3 * elements |]
            info.shape;
          equal
            (triple string (array int) string)
            ("uint8", [| 2; 3 * bytes |], data)
            (stored (find (Gguf.tensors g) "w"));
          (* A row one element past a whole number of blocks. *)
          let path =
            file
              (gguf []
                 [ ("w", tag, [ 2; elements + 1 ], pattern (4 * bytes) tag) ])
          in
          fails ~path ~naming:(Printf.sprintf "not of blocks of %d" elements)
            (fun () -> Nx_io.load_gguf path));
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
          raises_match (Exn.failure ~sub:"absent") (fun () ->
              Gguf.info "absent" g));
      (let one = [ ("w", 0, [ 1 ], "abcd") ] in
       let raw ?magic ?version ?alignment ?shift kvs tensors () =
         file (gguf ?magic ?version ?alignment ?shift kvs tensors)
       in
       let count n = "GGUF" ^ le 3 4 ^ u64 n ^ u64 0 in
       cases "a malformed file fails, naming it" ~id:fst
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
         ]
         (fun (_, path) ->
           let path = path () in
           fails ~path (fun () -> Nx_io.load_gguf path)));
    ]

let () = exit (run "nx.io GGUF" [ gguf_group ])
