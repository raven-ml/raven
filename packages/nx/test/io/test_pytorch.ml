(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* PyTorch checkpoints. Files torch 2.15 wrote load as they mean: every dtype,
   views, shared storages and tied weights, protocols 1 to 5, the legacy format,
   a big-endian and a compressed file; pickles written here byte by byte load as
   pickletools defines their opcodes, or are refused by the rules
   [Nx_io.load_pytorch] states; [Nx_io.Pytorch.tensors] names tensors by their
   paths and refuses what has none. The pt_*.pt files hold the bytes [pattern n
   seed] or small ranges, so their values are stated here. *)

open Windtrap
open Io_support
module Pt = Nx_io.Pytorch

(* Values *)

let rec equal_value (a : Pt.value) (b : Pt.value) =
  match (a, b) with
  | Float x, Float y ->
      Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
  | Complex x, Complex y -> x = y
  | Tensor x, Tensor y -> same_value x y
  | Tuple xs, Tuple ys | List xs, List ys | Set xs, Set ys ->
      List.equal equal_value xs ys
  | Dict xs, Dict ys ->
      List.equal
        (fun (k, v) (k', v') -> equal_value k k' && equal_value v v')
        xs ys
  | Shared (i, x), Shared (j, y) -> i = j && equal_value x y
  | Tensor _, _ | _, Tensor _ | Shared _, _ | _, Shared _ -> false
  | _ -> a = b

let rec pp_value ppf (v : Pt.value) =
  let list ppf =
    Format.pp_print_list
      ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
      pp_value ppf
  in
  match v with
  | Null -> Format.fprintf ppf "None"
  | Bool b -> Format.fprintf ppf "%b" b
  | Int i -> Format.fprintf ppf "%d" i
  | Long s -> Format.fprintf ppf "%SL" s
  | Float f -> Format.fprintf ppf "%h" f
  | Complex c -> Format.fprintf ppf "(%h+%hj)" c.re c.im
  | String s -> Format.fprintf ppf "%S" s
  | Bytes s -> Format.fprintf ppf "b%S" s
  | Tuple l -> Format.fprintf ppf "(%a)" list l
  | List l -> Format.fprintf ppf "[%a]" list l
  | Set l -> Format.fprintf ppf "{%a}" list l
  | Dict kv ->
      Format.fprintf ppf "{%a}"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
           (fun ppf (k, v) -> Format.fprintf ppf "%a: %a" pp_value k pp_value v))
        kv
  | Tensor x -> pp_packed ppf x
  | Global g -> Format.fprintf ppf "<global %s>" g
  | Object o -> Format.fprintf ppf "<object %s>" o
  | Shared (i, v) -> Format.fprintf ppf "#%d=%a" i pp_value v
  | Ref i -> Format.fprintf ppf "#%d" i

let value = Testable.make ~equal:equal_value ~pp:pp_value

let field k = function
  | Pt.Dict kv -> (
      match List.assoc_opt (Pt.String k) kv with
      | Some v -> v
      | None -> failf "no key %S" k)
  | v -> failf "%a is no dict" pp_value v

let tensor = function
  | Pt.Tensor x -> x
  | v -> failf "%a is no tensor" pp_value v

let f32 shape a = Pt.Tensor (Nx.P (Nx.create Nx.float32 shape a))
let floats n f = Array.init n (fun i -> Float.of_int (f i))

(* Pickles, byte by byte *)

let le n v = String.init n (fun i -> Char.chr ((v lsr (8 * i)) land 0xff))
let str s = "X" ^ le 4 (String.length s) ^ s
let int1 n = "K" ^ le 1 n
let global m n = "c" ^ m ^ "\n" ^ n ^ "\n"
let tuple items = "(" ^ String.concat "" items ^ "t"

let dict entries =
  "}("
  ^ String.concat "" (List.concat_map (fun (k, v) -> [ k; v ]) entries)
  ^ "u"

let odict = global "collections" "OrderedDict" ^ ")R"
let pickle ?(proto = 2) body = "\x80" ^ le 1 proto ^ body ^ "."

let pid ?(cls = "FloatStorage") key n =
  tuple [ str "storage"; global "torch" cls; str key; str "cpu"; int1 n ] ^ "Q"

(* [rebuild key n] is [_rebuild_tensor_v2] of a vector of the [n] elements of
   storage [key]. *)
let rebuild ?(cls = "FloatStorage") ?(hooks = odict) ?(meta = "") key n =
  global "torch._utils" "_rebuild_tensor_v2"
  ^ "(" ^ pid ~cls key n ^ int1 0 ^ int1 n ^ "\x85" ^ int1 1 ^ "\x85" ^ "\x89"
  ^ hooks ^ meta ^ "tR"

(* Zip archives as torch writes them *)

(* [zip entries] is a zip archive of [entries], each stored with its data at a
   multiple of 64 bytes, padded by an extra field as torch pads them. [crc]
   gives each entry's stated CRC-32 from its true one. *)
let zip ?(crc = fun _ c -> c) entries =
  let buf = Buffer.create 4096 in
  let central = Buffer.create 1024 in
  List.iter
    (fun (name, data) ->
      let off = Buffer.length buf in
      let used = off + 30 + String.length name in
      let pad = (64 - (used mod 64)) mod 64 in
      let pad = if pad > 0 && pad < 4 then pad + 64 else pad in
      let extra =
        if pad = 0 then ""
        else le 2 0x4246 ^ le 2 (pad - 4) ^ String.make (pad - 4) '\000'
      in
      let c = crc name (Lz.Crc32.string data) in
      let common =
        le 2 20 ^ le 2 0 ^ le 2 0 ^ le 2 0 ^ le 2 0 ^ le 4 c
        ^ le 4 (String.length data)
        ^ le 4 (String.length data)
        ^ le 2 (String.length name)
      in
      Buffer.add_string buf
        ("PK\003\004" ^ common
        ^ le 2 (String.length extra)
        ^ name ^ extra ^ data);
      Buffer.add_string central
        ("PK\001\002" ^ le 2 20 ^ common ^ le 2 0 ^ le 2 0 ^ le 2 0 ^ le 2 0
       ^ le 4 0 ^ le 4 off ^ name))
    entries;
  let cd = Buffer.length buf in
  Buffer.add_buffer buf central;
  let n = List.length entries in
  Buffer.add_string buf
    ("PK\005\006" ^ le 2 0 ^ le 2 0 ^ le 2 n ^ le 2 n
    ^ le 4 (Buffer.length central)
    ^ le 4 cd ^ le 2 0);
  Buffer.contents buf

(* [checkpoint p storages] is a torch zip of the pickle [p] and [storages], each
   a key and its bytes. *)
let checkpoint ?crc ?byteorder p storages =
  zip ?crc
    (("archive/data.pkl", p)
     ::
     (match byteorder with
     | None -> []
     | Some b -> [ ("archive/byteorder", b) ])
    @ List.map (fun (k, b) -> ("archive/data/" ^ k, b)) storages)

let floats_bytes l =
  String.concat ""
    (List.map
       (fun f -> le 4 (Int32.to_int (Int32.bits_of_float f) land 0xffffffff))
       l)

let load s = Nx_io.load_pytorch (file ~suffix:".pt" s)
let load_pickle ?(storages = []) p = load (checkpoint p storages)

let refused ~naming s =
  let path = file ~suffix:".pt" s in
  fails ~path ~naming (fun () -> Nx_io.load_pytorch path)

let four = [ ("0", floats_bytes [ 1.; 2.; 3.; 4. ]) ]

(* Files torch wrote *)

let torch_dtypes =
  (* PyTorch's name, nx's dtype or [None], and the element's bytes. *)
  D.
    [
      ("float64", Some (Any Float64), 8);
      ("float32", Some (Any Float32), 4);
      ("float16", Some (Any Float16), 2);
      ("bfloat16", Some (Any Bfloat16), 2);
      ("int64", Some (Any Int64), 8);
      ("int32", Some (Any Int32), 4);
      ("int16", Some (Any Int16), 2);
      ("int8", Some (Any Int8), 1);
      ("uint8", Some (Any Uint8), 1);
      ("bool", Some (Any Bool), 1);
      ("complex128", Some (Any Complex128), 16);
      ("complex64", Some (Any Complex64), 8);
      ("uint16", Some (Any Uint16), 2);
      ("uint32", Some (Any Uint32), 4);
      ("uint64", Some (Any Uint64), 8);
      ("float8_e4m3fn", Some (Any Float8_e4m3fn), 1);
      ("float8_e5m2", Some (Any Float8_e5m2), 1);
      ("float8_e4m3fnuz", None, 1);
      ("float8_e5m2fnuz", None, 1);
      ("float8_e8m0fnu", None, 1);
      ("float4_e2m1fn_x2", None, 1);
      ("complex32", None, 4);
      ("bits8", None, 1);
      ("bits16", None, 2);
      ("bits1x8", None, 1);
      ("bits2x4", None, 1);
      ("bits4x2", None, 1);
    ]

let views_expected =
  let base = Array.init 24 Float.of_int in
  [
    ("base", f32 [| 4; 6 |] base);
    ("offset", f32 [| 6 |] (floats 6 (fun i -> i + 5)));
    ("transposed", f32 [| 6; 4 |] (floats 24 (fun k -> (k mod 4 * 6) + (k / 4))));
    ("strided", f32 [| 8 |] (floats 8 (fun i -> 1 + (3 * i))));
    ("tied", f32 [| 6; 4 |] base);
    ("expanded", f32 [| 4; 3 |] (floats 12 (fun k -> k mod 3)));
    ("empty", f32 [| 0; 3 |] [||]);
    ("scalar", f32 [||] [| 2.5 |]);
    ("param", f32 [| 2 |] [| 1.; 1. |]);
  ]

let shared_expected =
  let t = Pt.Tensor (Nx.P (Nx.create Nx.int64 [| 2 |] [| 0L; 1L |])) in
  Pt.List [ t; t ]

let meta ~proto : Pt.value =
  let s k = Pt.String k in
  Dict
    [
      (s "none", Null);
      (s "true", Bool true);
      (s "false", Bool false);
      (s "int", Int 7);
      (s "neg", Int (-(1 lsl 40)));
      (s "big", Long "\x00\x00\x00\x00\x00\x00\x00\x00\x40");
      (s "negbig", Long "\x00\x00\x00\x00\x00\x00\x00\x00\xc0");
      (s "float", Float 1.5);
      (s "inf", Float Float.infinity);
      (s "negzero", Float (-0.));
      (s "complex", Complex { re = 1.; im = -2. });
      (s "str", String "\xc3\xa9\xe6\x97\xa5");
      (s "bytes", Bytes "\x00\xff");
      (s "bytearray", Bytes "ab");
      (s "empty_bytes", Bytes "");
      (s "tuple", Tuple [ Int 1; String "a" ]);
      (s "list", List [ Int 1; Int 2 ]);
      (s "set", Set [ Int 3; Int 4 ]);
      (s "frozenset", Set [ Int 5 ]);
      (s "odict", Dict [ (s "a", Int 1) ]);
      (s "counter", Dict [ (s "x", Int 2) ]);
      (s "size", Tuple [ Int 2; Int 3 ]);
      (s "dtype", Global "torch.float32");
      ( s "namespace",
        Object
          (if proto < 2 then "copyreg._reconstructor" else "argparse.Namespace")
      );
      ( s "keys",
        Dict
          [
            (Int 0, s "zero");
            (Float 1.5, s "half");
            (Tuple [ Int 1; Int 2 ], s "pair");
          ] );
    ]

(* The byte of [s] where the data of its entry [name] starts. *)
let data_of s name =
  let rec find i =
    if
      String.sub s i 4 = "PK\003\004"
      && String.sub s (i + 30) (String.length name) = name
    then
      i + 30 + String.get_uint16_le s (i + 26) + String.get_uint16_le s (i + 28)
    else find (i + 1)
  in
  find 0

let torch_files =
  group "files torch wrote"
    [
      test "every dtype loads at nx's dtype, or as bytes with an axis of them"
        (fun () ->
          let ckpt = Nx_io.load_pytorch (fixture "pt_dtypes.pt") in
          List.iteri
            (fun seed (name, dt, k) ->
              let bytes = pattern (6 * k) seed in
              let bytes =
                if name = "bool" then
                  String.map (fun c -> Char.chr (Char.code c land 1)) bytes
                else bytes
              in
              let expected =
                match dt with
                | Some (D.Any dt) -> (D.name dt, [| 2; 3 |], bytes)
                | None ->
                    ( "uint8",
                      (if k = 1 then [| 2; 3 |] else [| 2; 3; k |]),
                      bytes )
              in
              equal ~msg:name
                (triple string (array int) string)
                expected
                (stored (tensor (field name ckpt))))
            torch_dtypes);
      test "views keep their offsets and strides, over one storage mapped once"
        (fun () ->
          let ckpt = Nx_io.load_pytorch (fixture "pt_views.pt") in
          List.iter
            (fun (name, expected) ->
              equal ~msg:name value expected (field name ckpt))
            views_expected;
          equal ~msg:"one tensor twice" value shared_expected
            (field "shared" ckpt);
          equal ~msg:"an aligned stored storage" memory Over_file
            (memory_of (tensor (field "base" ckpt))));
      cases "Python's data loads at protocols 1 to 5" ~id:string_of_int
        [ 1; 2; 3; 4; 5 ] (fun proto ->
          let ckpt =
            Nx_io.load_pytorch
              (fixture (Printf.sprintf "pt_protocol%d.pt" proto))
          in
          equal value (f32 [| 4 |] [| 0.; 1.; 2.; 3. |]) (field "w" ckpt);
          equal value (meta ~proto) (field "meta" ckpt));
      test "the legacy format loads" (fun () ->
          let path = fixture "pt_legacy.pt" in
          let ckpt = Nx_io.load_pytorch path in
          List.iter
            (fun (name, expected) ->
              equal ~msg:name value expected (field name ckpt))
            [
              ("w", f32 [| 4; 6 |] (Array.init 24 Float.of_int));
              ( "t",
                f32 [| 6; 4 |] (floats 24 (fun k -> (k mod 4 * 6) + (k / 4))) );
              ("o", f32 [| 6 |] (floats 6 (fun i -> i + 5)));
              ( "own",
                Pt.Tensor (Nx.P (Nx.create Nx.int64 [| 3 |] [| 0L; 1L; 2L |]))
              );
            ]);
      test "a big-endian file loads swapped, into memory of its own" (fun () ->
          let ckpt = Nx_io.load_pytorch (fixture "pt_big.pt") in
          equal value (f32 [| 2 |] [| 1.5; -2. |]) (field "f" ckpt);
          equal value
            (Pt.Tensor
               (Nx.P (Nx.create Nx.complex64 [| 1 |] [| { re = 1.; im = 2. } |])))
            (field "c" ckpt);
          equal value
            (Pt.Tensor (Nx.P (Nx.create Nx.int16 [| 2 |] [| 1; -2 |])))
            (field "i" ckpt);
          equal memory Own_memory (memory_of (tensor (field "f" ckpt))));
      test "compressed storages load into memory of their own" (fun () ->
          let ckpt = Nx_io.load_pytorch (fixture "pt_deflated.pt") in
          List.iter
            (fun (name, expected) ->
              equal ~msg:name value expected (field name ckpt))
            views_expected;
          equal memory Own_memory (memory_of (tensor (field "base" ckpt))));
      test
        "a storage one tensor reads is its own, and one several read is shared"
        (fun () ->
          let path = fixture "pt_deflated.pt" in
          let load name () =
            Nx.unpack Nx.float32 (tensor (field name (Nx_io.load_pytorch path)))
          in
          let donated x = Nx.neg (Nx.donate x) in
          let _, grew = grown Rig.host (load "param") donated in
          less ~msg:"param's donation writes in place" int ~than:8 grew;
          let _, grew = grown Rig.host (load "base") donated in
          less ~msg:"base's donation writes new memory" int ~than:grew 95);
      test "a file written with CRCs off loads, its CRCs 0" (fun () ->
          let path = fixture "pt_nocrc.pt" in
          let s = read path in
          let rec cd i =
            if String.sub s i 4 = "PK\001\002" then i else cd (i + 1)
          in
          equal ~msg:"the first entry's CRC" int 0
            (String.get_int32_le s (cd 0 + 16) |> Int32.to_int);
          equal value
            (f32 [| 4 |] [| 0.; 1.; 2.; 3. |])
            (field "w" (Nx_io.load_pytorch path)));
      test "a whole module loads as an opaque value" (fun () ->
          equal value (Object "torch.nn.modules.linear.Linear")
            (Nx_io.load_pytorch (fixture "pt_module.pt")));
      test "a storage whose CRC-32 does not match fails, mapped as it is"
        (fun () ->
          let s = read (fixture "pt_views.pt") in
          let path =
            file ~suffix:".pt" (flip_byte s (data_of s "pt_views/data/0" + 5))
          in
          fails ~path ~naming:"CRC-32" (fun () -> Nx_io.load_pytorch path));
      test "every proper prefix fails, and any byte flipped loads or fails"
        (fun () ->
          let whole = read (fixture "pt_legacy.pt") in
          prefixes_fail Nx_io.load_pytorch whole;
          flips_load_or_fail Nx_io.load_pytorch whole;
          flips_load_or_fail Nx_io.load_pytorch (read (fixture "pt_views.pt")));
      cases "files torch reads otherwise, or with a bit set, are refused"
        ~id:fst
        [
          ("a conjugate bit", ("pt_conj.pt", "metadata"));
          ("a TorchScript archive", ("pt_script.pt", "TorchScript"));
        ]
        (fun (_, (name, naming)) ->
          let path = fixture name in
          fails ~path ~naming (fun () -> Nx_io.load_pytorch path));
      test "the tar format is refused" (fun () ->
          let header =
            String.make 257 '\000' ^ "ustar\00000" ^ String.make 245 '\000'
          in
          refused ~naming:"tar" (header ^ String.make 512 '\000'));
    ]

(* Opcodes *)

(* Pickles of protocols 1 to 5 and the values pickletools gives them. *)
let opcodes : (string * string * Pt.value) list =
  [
    ("BININT", "J" ^ le 4 0xfffffffb, Int (-5));
    ("BININT1", "K\xff", Int 255);
    ("BININT2", "M\x01\x02", Int 513);
    ("LONG1 of -2^62, min_int", "\x8a\x08" ^ le 7 0 ^ "\xc0", Int min_int);
    ("LONG1 of 2^62", "\x8a\x09" ^ le 7 0 ^ "\x40\x00", Long (le 7 0 ^ "\x40"));
    ("LONG1 with bytes that only extend its sign", "\x8a\x03\x05\x00\x00", Int 5);
    ("LONG4", "\x8b" ^ le 4 9 ^ le 8 0 ^ "\x01", Long (le 8 0 ^ "\x01"));
    ("LONG1 of no byte", "\x8a\x00", Int 0);
    ("INT of a bool", "I01\n", Bool true);
    ("INT", "I-7\n", Int (-7));
    ( "LONG",
      "L-12345678901234567890L\n",
      Long "\x2e\xf5\xe0\x14\x73\x56\xab\x54\xff" );
    ("BINFLOAT", "G\x3f\xf8" ^ le 6 0, Float 1.5);
    ("BINSTRING", "T" ^ le 4 3 ^ "abc", String "abc");
    ("SHORT_BINSTRING", "U\x02hi", String "hi");
    ("BINUNICODE8", "\x8d" ^ le 8 2 ^ "ok", String "ok");
    ( "SHORT_BINUNICODE of a lone surrogate",
      "\x8c\x03\xed\xa0\x80",
      String "\xed\xa0\x80" );
    ("BINBYTES", "B" ^ le 4 1 ^ "z", Bytes "z");
    ("BINBYTES8", "\x8e" ^ le 8 1 ^ "z", Bytes "z");
    ("BYTEARRAY8", "\x96" ^ le 8 2 ^ "ab", Bytes "ab");
    ("READONLY_BUFFER", "C\x01a\x98", Bytes "a");
    ( "NONE, NEWTRUE, NEWFALSE",
      "N\x88\x89\x87",
      Tuple [ Null; Bool true; Bool false ] );
    ( "EMPTY_TUPLE, TUPLE1, TUPLE2, TUPLE3",
      ")K\x01\x85K\x02K\x03\x86\x87",
      Tuple [ Tuple []; Tuple [ Int 1 ]; Tuple [ Int 2; Int 3 ] ] );
    ("TUPLE", "(K\x01K\x02t", Tuple [ Int 1; Int 2 ]);
    ( "EMPTY_LIST, APPEND, APPENDS",
      "]K\x01a(K\x02K\x03e",
      List [ Int 1; Int 2; Int 3 ] );
    ("LIST", "(K\x01l", List [ Int 1 ]);
    ( "EMPTY_DICT, SETITEM, SETITEMS",
      "}K\x01K\x02s(K\x03K\x04u",
      Dict [ (Int 1, Int 2); (Int 3, Int 4) ] );
    ("DICT", "(K\x01K\x02d", Dict [ (Int 1, Int 2) ]);
    ("EMPTY_SET, ADDITEMS", "\x8f(K\x01K\x02\x90", Set [ Int 1; Int 2 ]);
    ("FROZENSET", "(K\x01\x91", Set [ Int 1 ]);
    ("BINPUT, BINGET", "]q\x00h\x00\x86", Tuple [ Shared (0, List []); Ref 0 ]);
    ( "LONG_BINPUT, LONG_BINGET",
      "]r" ^ le 4 7 ^ "j" ^ le 4 7 ^ "\x86",
      Tuple [ Shared (0, List []); Ref 0 ] );
    ("MEMOIZE", "]\x94h\x00\x86", Tuple [ Shared (0, List []); Ref 0 ]);
    ("GLOBAL", global "torch" "float16", Global "torch.float16");
    ("STACK_GLOBAL", "\x8c\x05torch\x8c\x04bool\x93", Global "torch.bool");
    ("REDUCE", global "argparse" "Namespace" ^ ")R", Object "argparse.Namespace");
    ( "NEWOBJ",
      global "argparse" "Namespace" ^ ")\x81",
      Object "argparse.Namespace" );
    ("NEWOBJ_EX", global "m" "C" ^ ")}\x92", Object "m.C");
    ("BUILD", global "m" "C" ^ ")R}b", Object "m.C");
    ("FRAME", "\x95" ^ le 8 2 ^ "K\x01", Int 1);
    ( "OrderedDict, its attributes dropped",
      odict ^ "K\x01K\x02s}(" ^ str "_metadata" ^ "}ub",
      Dict [ (Int 1, Int 2) ] );
    ( "a call of the callable a call made",
      global "builtins" "getattr" ^ "(" ^ global "m" "C" ^ str "f" ^ "tR)R",
      Object "builtins.getattr" );
    ( "_codecs.encode",
      global "_codecs" "encode" ^ "(" ^ str "\xc3\xa9" ^ str "latin1" ^ "tR",
      Bytes "\xe9" );
    ("bytes()", global "builtins" "bytes" ^ ")R", Bytes "");
    ( "complex",
      global "builtins" "complex" ^ "(G" ^ le 8 0 ^ "K\x02tR",
      Complex { re = 0.; im = 2. } );
  ]

let refusals : (string * string * string) list =
  [
    ("POP", "N0N", "POP");
    ("POP_MARK", "(N1N", "POP_MARK");
    ("DUP", "N2\x86", "DUP");
    ("protocol 0's PUT", "]p0\n", "protocol-0");
    ("protocol 0's FLOAT", "F1.5\n", "protocol-0");
    ("protocol 0's PERSID", "P0\n", "protocol-0");
    ("INST", "(im\nC\n", "INST");
    ("OBJ", "(" ^ global "m" "C" ^ "o", "INST");
    ("EXT1", "\x82\x01", "EXT1");
    ("NEXT_BUFFER", "\x97", "NEXT_BUFFER");
    ("an INT with a leading zero", "I007\n", "decimal");
    ("PROTO 6", "\x80\x06N", "protocol 6");
    ("an opcode across its frame's end", "\x95" ^ le 8 1 ^ "K\x01", "frame");
    ("a string that is not UTF-8", "\x8c\x01\xff", "UTF-8");
    ("a memo entry never put", "h\x05", "memo");
    ("a MARK left at STOP", "(N", "MARK");
    ("two values at STOP", "NN", "2 values");
    ("a list holding itself", "]q\x00h\x00a", "holds");
    ("a second state", global "m" "C" ^ ")R}b}b", "second state");
    ("a key given twice, 1 and True", "}(K\x01N\x88Nu", "twice");
    ( "a key 1 and 1+0j",
      "}(K\x01N" ^ global "__builtin__" "complex" ^ "(K\x01K\x00tRNu",
      "twice" );
    ("a NaN key", "}(G\x7f\xf8" ^ le 6 0 ^ "Nu", "NaN");
    ("a list as a key", "}(]Nu", "equality");
    ( "a set element given twice",
      global "__builtin__" "set" ^ "((K\x01G\x3f\xf0" ^ le 6 0 ^ "ltR",
      "twice" );
    ( "OrderedDict with arguments",
      global "collections" "OrderedDict" ^ "(K\x01tR",
      "does not take" );
    ( "_codecs.encode of a character latin-1 lacks",
      global "_codecs" "encode" ^ "(" ^ str "\xe6\x97\xa5" ^ str "latin1" ^ "tR",
      "latin-1" );
    ("a call of a value that is not callable", "K\x01)R", "callable");
  ]

let opcode_cases =
  group "opcodes"
    [
      cases "each opcode builds what pickletools defines"
        ~id:(fun (n, _, _) -> n)
        opcodes
        (fun (_, body, expected) ->
          equal value expected (load_pickle (pickle ~proto:4 body)));
      test "protocol 1 has no PROTO opcode" (fun () ->
          equal value (Int 1) (load_pickle ("K\x01" ^ ".")));
      test "Python 2's names read as Python 3's below protocol 3" (fun () ->
          let p proto = pickle ~proto (global "__builtin__" "set") in
          equal value (Global "builtins.set") (load_pickle (p 2));
          equal value (Global "__builtin__.set") (load_pickle (p 3)));
      cases "a pickle breaking the language is refused, naming the rule"
        ~id:(fun (n, _, _) -> n)
        refusals
        (fun (_, body, naming) -> refused ~naming (checkpoint (pickle body) []));
      test "bytes after STOP are refused" (fun () ->
          refused ~naming:"after its STOP" (checkpoint (pickle "N" ^ "N") []));
    ]

(* Tensors *)

let tensor_cases =
  let p ?storages body = load_pickle ?storages (pickle body) in
  group "tensors"
    [
      test "a tensor is a view of its storage" (fun () ->
          equal value
            (Dict [ (String "w", f32 [| 4 |] [| 1.; 2.; 3.; 4. |]) ])
            (p ~storages:four (dict [ (str "w", rebuild "0" 4) ])));
      cases "a tensor call that holds more than torch writes is refused" ~id:fst
        [
          ( "BUILD on a tensor (torch's loader then runs set_)",
            ( dict
                [
                  ( str "w",
                    rebuild "0" 4
                    ^ tuple
                        [ pid "1" 4; int1 0; int1 4 ^ "\x85"; int1 1 ^ "\x85" ]
                    ^ "b" );
                ],
              "BUILD" ) );
          ( "a negative bit",
            ( dict
                [
                  (str "w", rebuild ~meta:(dict [ (str "neg", "\x88") ]) "0" 4);
                ],
              "metadata" ) );
          ( "a backend's layout key",
            ( dict
                [
                  ( str "w",
                    rebuild
                      ~meta:(dict [ (str "npu_format_/FRACTAL_NZ", "\x88") ])
                      "0" 4 );
                ],
              "metadata" ) );
          ( "backward hooks",
            ( dict [ (str "w", rebuild ~hooks:(dict [ (int1 0, "N") ]) "0" 4) ],
              "hooks" ) );
          ( "a view past its storage",
            ( dict
                [
                  ( str "w",
                    global "torch._utils" "_rebuild_tensor"
                    ^ tuple
                        [ pid "0" 4; int1 1; int1 4 ^ "\x85"; int1 1 ^ "\x85" ]
                    ^ "R" );
                ],
              "past" ) );
          ( "a stride past OCaml's int",
            ( dict
                [
                  ( str "w",
                    global "torch._utils" "_rebuild_tensor"
                    ^ tuple
                        [
                          pid "0" 4;
                          int1 0;
                          int1 2 ^ "\x85";
                          "\x8a\x08" ^ le 7 0 ^ "\x20\x85";
                        ]
                    ^ "R" );
                ],
              "past" ) );
          ( "a storage class torch lacks",
            (dict [ (str "w", rebuild ~cls:"NoStorage" "0" 4) ], "storage class")
          );
          ( "a dtype torch lacks",
            ( dict
                [
                  ( str "w",
                    global "torch._utils" "_rebuild_tensor_v3"
                    ^ tuple
                        [
                          pid ~cls:"UntypedStorage" "0" 16;
                          int1 0;
                          int1 4 ^ "\x85";
                          int1 1 ^ "\x85";
                          "\x89";
                          odict;
                          global "torch" "float7";
                        ]
                    ^ "R" );
                ],
              "dtype" ) );
          ( "a key named with two dtypes",
            ( tuple [ rebuild "0" 4; rebuild ~cls:"IntStorage" "0" 4 ],
              "two dtypes" ) );
          ( "a persistent id that names no storage",
            (tuple [ str "module" ] ^ "Q", "no storage") );
        ]
        (fun (_, (body, naming)) ->
          refused ~naming (checkpoint (pickle body) four));
      cases "a storage's record must be the length its id states, and be there"
        ~id:fst
        [
          ("too long", ([ ("0", floats_bytes [ 1.; 2.; 3.; 4.; 5. ]) ], "bytes"));
          ("missing", ([], "no archive/data/0"));
        ]
        (fun (_, (storages, naming)) ->
          refused ~naming (checkpoint (pickle (rebuild "0" 4)) storages));
      test "a storage whose stated CRC-32 is 0 is read unchecked" (fun () ->
          let s =
            checkpoint
              ~crc:(fun n c -> if n = "archive/data/0" then 0 else c)
              (pickle (rebuild "0" 4))
              four
          in
          equal value (f32 [| 4 |] [| 1.; 2.; 3.; 4. |]) (load s));
      test
        "a big-endian file's untyped storage read wider than a byte is refused"
        (fun () ->
          let body =
            global "torch._utils" "_rebuild_tensor_v3"
            ^ tuple
                [
                  pid ~cls:"UntypedStorage" "0" 4;
                  int1 0;
                  int1 2 ^ "\x85";
                  int1 1 ^ "\x85";
                  "\x89";
                  odict;
                  global "torch" "uint16";
                ]
            ^ "R"
          in
          refused ~naming:"byte order"
            (checkpoint ~byteorder:"big" (pickle body) [ ("0", "abcd") ]));
    ]

(* Calls outside the vocabulary *)

let opaque_cases =
  let setitem = global "operator" "setitem" in
  group "opaque calls"
    [
      test "a call given values nothing else reaches loads as its name"
        (fun () ->
          equal value
            (Dict [ (String "o", Object "m.f") ])
            (load_pickle ~storages:four
               (pickle
                  (dict
                     [
                       ( str "o",
                         global "m" "f"
                         ^ tuple [ "]" ^ rebuild "0" 4 ^ "a" ]
                         ^ "R" );
                     ]))));
      cases "a value or storage given to one and reachable elsewhere is refused"
        ~id:fst
        [
          ( "a list it is given, also a dict's value",
            dict
              [
                (str "a", "]q\x00" ^ rebuild "0" 4 ^ "a");
                ( str "b",
                  setitem ^ tuple [ "h\x00"; int1 0; rebuild "1" 4 ] ^ "R" );
              ] );
          ( "a storage a returned tensor reads",
            dict
              [
                (str "a", rebuild "0" 4);
                (str "b", global "m" "f" ^ tuple [ pid "0" 4 ] ^ "R");
              ] );
          ( "an object a setter is given, kept as the setter's result is",
            dict
              [
                (str "o", global "m" "S" ^ ")Rq\x00");
                (str "r", global "m" "setter" ^ tuple [ "h\x00"; dict [] ] ^ "R");
              ] );
          ( "a list given to two calls",
            global "m" "f" ^ tuple [ "]q\x00" ] ^ "R" ^ global "m" "g"
            ^ tuple [ "h\x00" ] ^ "R\x86" );
          ( "an OrderedDict's state that a returned dict also holds",
            dict [ (str "d", "}q\x00"); (str "o", odict ^ "h\x00b") ] );
          ( "a state that a returned dict also holds",
            dict [ (str "d", "}q\x00"); (str "o", global "m" "C" ^ ")Rh\x00b") ]
          );
        ]
        (fun (_, body) ->
          refused ~naming:"reachable elsewhere"
            (checkpoint (pickle body)
               (four @ [ ("1", floats_bytes [ 0.; 0.; 0.; 0. ]) ])));
      test "a bare global loads as its name" (fun () ->
          equal value (Global "m.C") (load_pickle (pickle (global "m" "C"))));
    ]

(* Naming *)

let naming =
  let t = f32 [| 1 |] [| 1. |] in
  let names v = Archive.names (Pt.tensors v) in
  group "Pytorch.tensors"
    [
      test "names a state dict's tensors by its keys" (fun () ->
          let ckpt = Nx_io.load_pytorch (fixture "pt_views.pt") in
          equal (list string)
            [
              "base";
              "empty";
              "expanded";
              "offset";
              "param";
              "scalar";
              "shared.0";
              "shared.1";
              "strided";
              "tied";
              "transposed";
            ]
            (names ckpt);
          equal packed
            (tensor (field "base" ckpt))
            (find (Pt.tensors ckpt) "base"));
      test "joins keys and indices by dots" (fun () ->
          equal (list string) [ "7"; "a.0.b"; "a.1" ]
            (names
               (Dict
                  [
                    (String "a", List [ Dict [ (String "b", t) ]; t ]);
                    (Int 7, t);
                  ])));
      cases "refuses a value whose tensors have no names" ~id:fst
        [
          ("an object", Pt.Dict [ (String "m", Object "torch.nn.Linear") ]);
          ("a tensor under a global's key", Dict [ (Global "m.C", t) ]);
          ("a tensor in a set", Set [ t ]);
          ( "two tensors of one name",
            Dict [ (String "a.b", t); (String "a", Dict [ (String "b", t) ]) ]
          );
          ("a tensor at the root", t);
        ]
        (fun (_, v) ->
          raises_match (Exn.failure ~sub:"Nx_io.Pytorch.tensors") (fun () ->
              ignore (Pt.tensors v)));
      test "a whole module's tensors have no names" (fun () ->
          raises_match (Exn.failure ~sub:"torch.nn.modules.linear.Linear")
            (fun () ->
              ignore (Pt.tensors (Nx_io.load_pytorch (fixture "pt_module.pt")))));
    ]

(* Integers, keys and sharing *)

(* [decimal_of_twos b] is the decimal of the integer whose two's complement,
   little-endian, is [b]: the magnitude divided by ten until it is zero. *)
let decimal_of_twos b =
  let n = String.length b in
  let negative = n > 0 && Char.code b.[n - 1] land 0x80 <> 0 in
  let mag = Array.init n (fun i -> Char.code b.[i]) in
  if negative then begin
    let carry = ref 1 in
    for i = 0 to n - 1 do
      let v = (lnot mag.(i) land 0xff) + !carry in
      mag.(i) <- v land 0xff;
      carry := v lsr 8
    done
  end;
  let digits = Buffer.create 32 in
  while Array.exists (( <> ) 0) mag do
    let rest = ref 0 in
    for i = n - 1 downto 0 do
      let v = (!rest lsl 8) lor mag.(i) in
      mag.(i) <- v / 10;
      rest := v mod 10
    done;
    Buffer.add_char digits (Char.chr (Char.code '0' + !rest))
  done;
  let d = Buffer.contents digits in
  let d =
    String.init (String.length d) (fun i -> d.[String.length d - 1 - i])
  in
  if d = "" then "0" else if negative then "-" ^ d else d

(* Byte strings that reach [int]'s bounds and their neighbours, and lengths 0, 8
   and 9. *)
(* Byte strings at [int]'s bounds and their neighbours, and of lengths 0, 8
   and 9, which the law runs on as examples, so every seed reaches them. *)
let bounds =
  [
    "";
    le 7 0xffffffffffffff ^ "\x3f" (* max_int *);
    le 7 0 ^ "\x40" (* max_int + 1 *);
    le 7 0 ^ "\xc0" (* min_int *);
    le 7 0xffffffffffffff ^ "\xbf" (* min_int - 1 *);
    le 8 0 ^ "\x01";
    "\xff";
  ]

let twos_bytes = Gen.string_of ~size:(Gen.int_range 0 12) Gen.char

(* The forms of one integer a key may take: an int, a float, a complex and a
   bool, where Python's value is that integer. *)
type form = As_int | As_float | As_complex | As_bool

(* Integers by their two's complement, with the float that is exactly each one,
   if any. *)
let integers =
  [
    ("0", le 1 0, Some 0.);
    ("1", le 1 1, Some 1.);
    ("-1", "\xff", Some (-1.));
    ("2^53", le 7 0x20000000000000, Some 0x1p53);
    ("2^53+1", le 7 0x20000000000001, None);
    ("-(2^53+1)", le 7 0xdfffffffffffff ^ "\xff", None);
    ("max_int", le 7 0xffffffffffffff ^ "\x3f", None);
    ("min_int, -2^62", le 7 0 ^ "\xc0", Some (-0x1p62));
    ("2^62", le 7 0 ^ "\x40", Some 0x1p62);
    ("-2^62-1", le 7 0xffffffffffffff ^ "\xbf", None);
    ("2^64", le 8 0 ^ "\x01", Some 0x1p64);
  ]

let key_of (name, bytes, exact) form =
  let long = "\x8a" ^ le 1 (String.length bytes) ^ bytes in
  let float f =
    "G"
    ^ String.init 8 (fun i ->
        Char.chr
          (Int64.to_int
             (Int64.shift_right_logical (Int64.bits_of_float f) (8 * (7 - i)))
          land 0xff))
  in
  match (form, exact) with
  | As_int, _ -> Some long
  | As_float, Some f -> Some (float f)
  | As_complex, Some f ->
      Some (global "builtins" "complex" ^ "(" ^ float f ^ "G" ^ le 8 0 ^ "tR")
  | As_bool, _ when name = "0" -> Some "\x89"
  | As_bool, _ when name = "1" -> Some "\x88"
  | _ -> None

let laws =
  group "laws"
    [
      prop "LONG1 of any bytes and LONG of their decimal are one integer"
        ~examples:bounds twos_bytes (fun b ->
          let n = String.length b in
          cover "no byte" (n = 0);
          cover "8 bytes" (n = 8);
          cover "9 bytes" (n = 9);
          let long1 = load_pickle (pickle ~proto:4 ("\x8a" ^ le 1 n ^ b)) in
          (match long1 with
          | Int i ->
              cover "max_int" (i = max_int);
              cover "min_int" (i = min_int)
          | _ -> ());
          let text =
            load_pickle (pickle ~proto:4 ("L" ^ decimal_of_twos b ^ "L\n"))
          in
          equal value long1 text);
      prop "keys that are one integer in Python are one key, and others two"
        (let open Gen in
         let boundary = List.filteri (fun i _ -> i = 7 || i = 8) integers in
         let integer =
           frequency [ (3, of_list boundary); (2, of_list integers) ]
         in
         let keyed x =
           let forms = [ As_int; As_float; As_complex; As_bool ] in
           of_list
             (List.filter_map
                (fun f -> Option.map (fun k -> (f, k)) (key_of x f))
                forms)
         in
         let* x = integer in
         let* same = bool in
         let* y = if same then constant x else integer in
         let* fk = keyed x in
         let+ gk = keyed y in
         ((x, fk), (y, gk)))
        (fun (((a, _, _), (f, k)), ((b, _, _), (g, k'))) ->
          cover "one integer in two forms" (a = b && f <> g);
          cover "two integers" (a <> b);
          cover "-2^62 or 2^62 in two forms"
            ((a = "min_int, -2^62" || a = "2^62") && b = a && f <> g);
          let s = checkpoint (pickle ("}(" ^ k ^ "N" ^ k' ^ "Nu")) [] in
          if a = b then refused ~naming:"twice" s else ignore (load s));
    ]

(* [doubled d] is a tuple holding the tuple below it twice, [d] levels deep over
   [leaf]: 5 bytes a level, 2^d paths. *)
let doubled ~leaf d =
  leaf ^ "q\x00" ^ String.concat "" (List.init d (fun _ -> "h\x00\x86q\x00"))

(* [dict_of_tuples keys] is a dict whose keys are tuples of ints below 3001,
   [keys] then [(0, 1, ..., 3000)], each of value [None]. *)
let dict_of_tuples keys =
  let int i = "M" ^ le 2 i in
  let tuple t = "(" ^ String.concat "" (List.map int t) ^ "t" in
  pickle
    ("}("
    ^ String.concat "" (List.map (fun t -> tuple t ^ "N") keys)
    ^ tuple (List.init 3001 Fun.id)
    ^ "Nu")

(* [colliding n] is [n] distinct 4-tuples [(a, b, c, d)] with [29791 a + 961 b +
   31 c + d] constant. The keys of a load number the ints 0 to 3000 in that
   order when the dict ends with [(0, ..., 3000)], so these tuples all collide
   under a hash that folds [h * 31 + x], as nx.io's did. *)
let colliding n =
  let target = 1000 * 29791 in
  let out = ref [] and count = ref 0 in
  (try
     for a = 1000 downto 0 do
       for b = 0 to 3000 do
         (* The [c] whose [d] lies in 0 to 3000. *)
         let rest = target - (a * 29791) - (b * 961) in
         for
           c = Int.max 0 ((rest - 3000 + 30) / 31) to Int.min 3000 (rest / 31)
         do
           let d = rest - (c * 31) in
           if d >= 0 && d <= 3000 then begin
             out := [ a; b; c; d ] :: !out;
             incr count;
             if !count = n then raise Exit
           end
         done
       done
     done
   with Exit -> ());
  !out

(* The least of three times [f ()] takes, in seconds. *)
let best f =
  List.fold_left Float.min Float.infinity
    (List.init 3 (fun _ ->
         let t = Unix.gettimeofday () in
         ignore (f ());
         Unix.gettimeofday () -. t))

let review =
  group "sizes a hostile file chooses"
    [
      test "keys a file makes collide load as fast as as many ordinary ones"
        (fun () ->
          let n = 20_000 in
          let crafted = colliding n in
          equal ~msg:"distinct crafted keys" int n
            (List.length (List.sort_uniq compare crafted));
          let ordinary =
            List.init n (fun i -> [ i mod 3001; i / 3001; 7; 11 ])
          in
          let time keys =
            let path =
              file ~suffix:".pt" (checkpoint (dict_of_tuples keys) [])
            in
            best (fun () -> Nx_io.load_pytorch path)
          in
          let t = time ordinary in
          less ~msg:"seconds, against five times an ordinary file's" float_exact
            ~than:((5. *. t) +. 0.05)
            (time crafted));
      test "a key that doubles 40 times loads in linear time, shared" (fun () ->
          match load_pickle (pickle ("}" ^ doubled ~leaf:")" 40 ^ "Ns")) with
          | Dict [ (Tuple [ Shared (0, _); Ref 0 ], Null) ] -> ()
          | v -> failf "%a" pp_value v);
      test "tensors refuses a shared value that holds a tensor" (fun () ->
          let v =
            load_pickle ~storages:four
              (pickle (doubled ~leaf:(rebuild "0" 4) 40))
          in
          raises_match (Exn.failure ~sub:"holds a tensor") (fun () ->
              ignore (Pt.tensors v)));
      test "tensors of a value that doubles 40 times, with no tensor, is empty"
        (fun () ->
          let v = load_pickle (pickle (doubled ~leaf:")" 40)) in
          equal (list string) [] (Archive.names (Pt.tensors v)));
      test "a shared list is Shared at its first path and a Ref after"
        (fun () ->
          equal value
            (Dict
               [ (String "a", Shared (0, List [ Int 1 ])); (String "b", Ref 0) ])
            (load_pickle
               (pickle (dict [ (str "a", "]q\x01K\x01a"); (str "b", "h\x01") ]))));
      test "a key nested 10^6 tuples deep loads, under an 8 MiB stack"
        (fun () ->
          let p = pickle ("})" ^ String.make 1_000_000 '\x85' ^ "Ns") in
          let rec depth n = function
            | Pt.Tuple [ t ] -> depth (n + 1) t
            | Tuple [] -> n
            | _ -> -1
          in
          match load_pickle p with
          | Dict [ (k, Null) ] -> equal int 1_000_000 (depth 0 k)
          | _ -> failf "not a dict of one key");
      test "a LONG4 of 10^6 bytes loads as its bytes" (fun () ->
          let b = String.make 999_999 '\x01' ^ "\x02" in
          equal value (Long b)
            (load_pickle (pickle ("\x8b" ^ le 4 1_000_000 ^ b))));
      test "LONG text of 4300 digits loads, and of 4301 is refused" (fun () ->
          let digits n = "1" ^ String.make (n - 1) '0' in
          (match load_pickle (pickle ("L" ^ digits 4300 ^ "L\n")) with
          | Long _ -> ()
          | v -> failf "%a" pp_value v);
          refused ~naming:"4300"
            (checkpoint (pickle ("L" ^ digits 4301 ^ "L\n")) []));
      cases "byte counts that pass an int's range are refused" ~id:fst
        [
          ( "an offset times a dtype's width",
            ( global "torch._utils" "_rebuild_tensor_v3"
              ^ tuple
                  [
                    pid ~cls:"UntypedStorage" "0" 4;
                    "\x8a\x08" ^ le 7 0 ^ "\x20";
                    int1 1 ^ "\x85";
                    int1 1 ^ "\x85";
                    "\x89";
                    odict;
                    global "torch" "complex32";
                  ]
              ^ "R",
              [ ("0", "abcd") ],
              "past" ) );
          ( "a storage's count times its width",
            ( tuple
                [
                  str "storage";
                  global "torch" "ComplexDoubleStorage";
                  str "0";
                  str "cpu";
                  "\x8a\x08" ^ le 7 0 ^ "\x08";
                ]
              ^ "Q",
              [ ("0", "") ],
              "more bytes" ) );
        ]
        (fun (_, (body, storages, naming)) ->
          refused ~naming (checkpoint (pickle body) storages));
    ]

(* The legacy format, byte by byte: its magic number, version, system
   information, value and keys, then each storage's count and elements. *)
let legacy value keys storages =
  pickle ("\x8a\x0a" ^ "\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19")
  ^ pickle "M\xe9\x03" ^ pickle "}" ^ value
  ^ pickle ("](" ^ String.concat "" (List.map str keys) ^ "e")
  ^ String.concat ""
      (List.map (fun b -> le 8 (String.length b / 4) ^ b) storages)

let legacy_pid ?(view = "N") key n =
  tuple
    [
      str "storage";
      global "torch" "FloatStorage";
      str key;
      str "cpu";
      int1 n;
      view;
    ]
  ^ "Q"

let legacy_tensor ?view key n size =
  global "torch._utils" "_rebuild_tensor_v2"
  ^ "(" ^ legacy_pid ?view key n ^ int1 0 ^ int1 size ^ "\x85" ^ int1 1 ^ "\x85"
  ^ "\x89" ^ odict ^ "tR"

let legacy_views =
  let view k off size = tuple [ str k; int1 off; int1 size ] in
  let base = floats_bytes [ 1.; 2.; 3.; 4. ] in
  group "legacy views"
    [
      test "a view of a storage reads its elements" (fun () ->
          equal value
            (Dict
               [
                 (String "a", f32 [| 2 |] [| 1.; 2. |]);
                 (String "b", f32 [| 2 |] [| 3.; 4. |]);
               ])
            (load
               (legacy
                  (pickle
                     (dict
                        [
                          (str "a", legacy_tensor ~view:(view "v0" 0 2) "0" 4 2);
                          (str "b", legacy_tensor ~view:(view "v1" 2 2) "0" 4 2);
                        ]))
                  [ "0" ] [ base ])));
      cases "a view key that torch would read otherwise is refused" ~id:fst
        [
          ( "named again at another offset",
            (view "v0" 0 2, view "v0" 2 2, "view \"v0\"") );
          ("naming a storage", (view "v0" 0 2, view "0" 2 2, "storage's key"));
          ("past its storage", (view "v0" 0 2, view "v1" 3 2, "does not lie"));
        ]
        (fun (_, (a, b, naming)) ->
          refused ~naming
            (legacy
               (pickle
                  (dict
                     [
                       (str "a", legacy_tensor ~view:a "0" 4 2);
                       (str "b", legacy_tensor ~view:b "0" 4 2);
                     ]))
               [ "0" ] [ base ]));
    ]

let () =
  exit
    (run "nx.io PyTorch"
       [
         torch_files;
         opcode_cases;
         tensor_cases;
         opaque_cases;
         naming;
         laws;
         review;
         legacy_views;
       ])
