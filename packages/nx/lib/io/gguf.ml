(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type value =
  | Uint8 of int
  | Int8 of int
  | Uint16 of int
  | Int16 of int
  | Uint32 of int
  | Int32 of int
  | Uint64 of int64
  | Int64 of int64
  | Float32 of float
  | Float64 of float
  | Bool of bool
  | String of string
  | Array of value array

type dtype =
  | F32
  | F16
  | BF16
  | F64
  | I8
  | I16
  | I32
  | I64
  | Q4_0
  | Q4_1
  | Q5_0
  | Q5_1
  | Q8_0
  | Q8_1
  | Q2_K
  | Q3_K
  | Q4_K
  | Q5_K
  | Q6_K
  | Q8_K
  | IQ2_XXS
  | IQ2_XS
  | IQ3_XXS
  | IQ1_S
  | IQ4_NL
  | IQ3_S
  | IQ2_S
  | IQ4_XS
  | IQ1_M
  | TQ1_0
  | TQ2_0
  | MXFP4

type tensor_info = { dtype : dtype; shape : int array }

type t = {
  version : int;
  metadata : (string * value) list;
  tensors : Archive.t;
  infos : tensor_info Archive.Names.t;
}

open Error
module B = Nx_device.Buffer
module A = Bigarray.Array1

let magic = "GGUF"

(* A file's alignment without a [general.alignment] key. *)
let default_alignment = 32

(* The magic, the version and the two counts. *)
let min_header = 24

(* Tensor types *)

type kind = K : ('a, 'b) Nx_dtype.t -> kind

(* How a tensor type stores elements: as an nx dtype, or in blocks of [elements]
   elements and [bytes] bytes. *)
type storage = Scalar of kind | Blocks of { elements : int; bytes : int }

(* The tags of ggml's [ggml_type]. The gaps are types since removed. *)
let dtype_of_tag = function
  | 0 -> F32
  | 1 -> F16
  | 2 -> Q4_0
  | 3 -> Q4_1
  | 6 -> Q5_0
  | 7 -> Q5_1
  | 8 -> Q8_0
  | 9 -> Q8_1
  | 10 -> Q2_K
  | 11 -> Q3_K
  | 12 -> Q4_K
  | 13 -> Q5_K
  | 14 -> Q6_K
  | 15 -> Q8_K
  | 16 -> IQ2_XXS
  | 17 -> IQ2_XS
  | 18 -> IQ3_XXS
  | 19 -> IQ1_S
  | 20 -> IQ4_NL
  | 21 -> IQ3_S
  | 22 -> IQ2_S
  | 23 -> IQ4_XS
  | 24 -> I8
  | 25 -> I16
  | 26 -> I32
  | 27 -> I64
  | 28 -> F64
  | 29 -> IQ1_M
  | 30 -> BF16
  | 34 -> TQ1_0
  | 35 -> TQ2_0
  | 39 -> MXFP4
  | tag -> fail_msg "tensor type %d is unknown" tag

(* The block sizes are those of ggml's block structures: a Q8_0 block is a
   float16 scale and 32 int8 values, 34 bytes for 32 elements. *)
let storage = function
  | F32 -> Scalar (K Float32)
  | F16 -> Scalar (K Float16)
  | BF16 -> Scalar (K BFloat16)
  | F64 -> Scalar (K Float64)
  | I8 -> Scalar (K Int8)
  | I16 -> Scalar (K Int16)
  | I32 -> Scalar (K Int32)
  | I64 -> Scalar (K Int64)
  | Q4_0 -> Blocks { elements = 32; bytes = 18 }
  | Q4_1 -> Blocks { elements = 32; bytes = 20 }
  | Q5_0 -> Blocks { elements = 32; bytes = 22 }
  | Q5_1 -> Blocks { elements = 32; bytes = 24 }
  | Q8_0 -> Blocks { elements = 32; bytes = 34 }
  | Q8_1 -> Blocks { elements = 32; bytes = 36 }
  | Q2_K -> Blocks { elements = 256; bytes = 84 }
  | Q3_K -> Blocks { elements = 256; bytes = 110 }
  | Q4_K -> Blocks { elements = 256; bytes = 144 }
  | Q5_K -> Blocks { elements = 256; bytes = 176 }
  | Q6_K -> Blocks { elements = 256; bytes = 210 }
  | Q8_K -> Blocks { elements = 256; bytes = 292 }
  | IQ2_XXS -> Blocks { elements = 256; bytes = 66 }
  | IQ2_XS -> Blocks { elements = 256; bytes = 74 }
  | IQ3_XXS -> Blocks { elements = 256; bytes = 98 }
  | IQ1_S -> Blocks { elements = 256; bytes = 50 }
  | IQ4_NL -> Blocks { elements = 32; bytes = 18 }
  | IQ3_S -> Blocks { elements = 256; bytes = 110 }
  | IQ2_S -> Blocks { elements = 256; bytes = 82 }
  | IQ4_XS -> Blocks { elements = 256; bytes = 136 }
  | IQ1_M -> Blocks { elements = 256; bytes = 56 }
  | TQ1_0 -> Blocks { elements = 256; bytes = 54 }
  | TQ2_0 -> Blocks { elements = 256; bytes = 66 }
  | MXFP4 -> Blocks { elements = 32; bytes = 17 }

(* Header decoding *)

(* The header's bytes and the position of the next one. *)
type cursor = {
  bytes : (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) A.t;
  mutable pos : int;
}

let remaining c = A.dim c.bytes - c.pos

(* [need c n] fails unless [n] bytes are left. Each item a count counts takes a
   byte at least, so this also bounds what a count allocates. *)
let need c n =
  if n > remaining c then
    fail_msg "the file ends at byte %d, inside its header" (A.dim c.bytes)

(* The unsigned little-endian integer of the next [n <= 4] bytes. *)
let uint c n =
  need c n;
  let v = ref 0 in
  for i = n - 1 downto 0 do
    v := (!v lsl 8) lor A.unsafe_get c.bytes (c.pos + i)
  done;
  c.pos <- c.pos + n;
  !v

let sint c n =
  let shift = Sys.int_size - (8 * n) in
  (uint c n lsl shift) asr shift

let int64 c =
  let low = uint c 4 in
  let high = uint c 4 in
  Int64.logor (Int64.shift_left (Int64.of_int high) 32) (Int64.of_int low)

(* A uint64 length or count, of at most [remaining c] items. *)
let count c =
  let pos = c.pos in
  let n = int64 c in
  if Int64.compare n 0L < 0 || Int64.compare n (Int64.of_int (remaining c)) > 0
  then fail_msg "the count %Lu at byte %d exceeds the header" n pos;
  Int64.to_int n

(* A uint64 dimension or offset. *)
let size c =
  let pos = c.pos in
  let n = int64 c in
  if Int64.compare n 0L < 0 || Int64.compare n (Int64.of_int max_int) > 0 then
    fail_msg "the size %Lu at byte %d is too large" n pos;
  Int64.to_int n

(* The next [n] bytes. *)
let chars c n =
  need c n;
  let at i = Char.unsafe_chr (A.unsafe_get c.bytes (c.pos + i)) in
  let s = String.init n at in
  c.pos <- c.pos + n;
  s

let string c = chars c (count c)

(* The tags of the specification's [gguf_metadata_value_type]. *)
let rec value c tag =
  match tag with
  | 0 -> Uint8 (uint c 1)
  | 1 -> Int8 (sint c 1)
  | 2 -> Uint16 (uint c 2)
  | 3 -> Int16 (sint c 2)
  | 4 -> Uint32 (uint c 4)
  | 5 -> Int32 (sint c 4)
  | 6 -> Float32 (Int32.float_of_bits (Int32.of_int (uint c 4)))
  | 7 -> (
      match uint c 1 with
      | 0 -> Bool false
      | 1 -> Bool true
      | b -> fail_msg "the bool at byte %d is %d" (c.pos - 1) b)
  | 8 -> String (string c)
  | 9 ->
      let elements = uint c 4 in
      if elements > 12 then fail_msg "value type %d is unknown" elements;
      let n = count c in
      Array (Array.init n (fun _ -> value c elements))
  | 10 -> Uint64 (int64 c)
  | 11 -> Int64 (int64 c)
  | 12 -> Float64 (Int64.float_of_bits (int64 c))
  | tag -> fail_msg "value type %d is unknown" tag

let version c =
  match uint c 4 with
  | (2 | 3) as v -> v
  | v ->
      let swapped =
        ((v land 0xff) lsl 24)
        lor ((v land 0xff00) lsl 8)
        lor ((v lsr 8) land 0xff00)
        lor (v lsr 24)
      in
      if swapped >= 1 && swapped <= 3 then
        failwith "the file is big-endian, which is not supported";
      fail_msg "version %d is not supported, only 2 and 3" v

let alignment metadata =
  match List.assoc_opt "general.alignment" metadata with
  | None -> default_alignment
  | Some (Uint32 a) when a > 0 && a land (a - 1) = 0 -> a
  | Some (Uint32 a) -> fail_msg "the alignment %d is not a power of two" a
  | Some _ -> failwith "general.alignment is not a uint32"

let metadata c n =
  let keys = Hashtbl.create n in
  let kv _ =
    let key = string c in
    if Hashtbl.mem keys key then fail_msg "the key %S is given twice" key;
    Hashtbl.add keys key ();
    let tag = uint c 4 in
    (key, value c tag)
  in
  Array.to_list (Array.init n kv)

(* A tensor's name, type, dimensions in file order and offset. *)
let tensor_info c =
  let name = string c in
  let rank = uint c 4 in
  need c rank;
  let dims = Array.init rank (fun _ -> size c) in
  let dtype = dtype_of_tag (uint c 4) in
  let offset = size c in
  (name, dtype, dims, offset)

(* Tensors *)

let mul name a b =
  if b <> 0 && a > max_int / b then fail_msg "tensor %S is too large" name;
  a * b

(* The dtype, the shape and the size in bytes of tensor [name] of logical
   [shape] as nx loads it. A block format's rows, along the last axis, are their
   bytes. *)
let layout name dtype shape =
  let rank = Array.length shape in
  let numel = Array.fold_left (mul name) 1 shape in
  match storage dtype with
  | Scalar (K kind) -> (K kind, shape, mul name numel (Nx_dtype.itemsize kind))
  | Blocks { elements; bytes } ->
      let row = if rank = 0 then 1 else shape.(rank - 1) in
      if row mod elements <> 0 then
        fail_msg "tensor %S has rows of %d elements, not of blocks of %d" name
          row elements;
      let row_bytes = row / elements * bytes in
      let rows = if row = 0 then 0 else numel / row in
      let stored = if rank = 0 then [| row_bytes |] else Array.copy shape in
      if rank > 0 then stored.(rank - 1) <- row_bytes;
      (K Nx_dtype.UInt8, stored, mul name rows row_bytes)

let read_tensors file ~data_start ~alignment infos =
  let file_len = B.length file in
  Array.fold_left
    (fun (tensors, infos) (name, dtype, dims, offset) ->
      if name = "" then failwith "a tensor has an empty name";
      if Archive.mem name tensors then fail_msg "tensor %S is named twice" name;
      if offset mod alignment <> 0 then
        fail_msg "tensor %S is at offset %d, not a multiple of %d" name offset
          alignment;
      let shape = Array.of_list (List.rev (Array.to_list dims)) in
      let K kind, stored, len = layout name dtype shape in
      if offset > file_len - data_start || len > file_len - data_start - offset
      then fail_msg "the file ends before tensor %S's data" name;
      let off = data_start + offset in
      let tensor = Storage.mapped file kind stored ~off ~len in
      ( Archive.add name tensor tensors,
        Archive.Names.add name { dtype; shape } infos ))
    (Archive.empty, Archive.Names.empty)
    infos

let read file =
  let file_len = B.length file in
  if file_len < min_header then
    fail_msg "%d bytes is too short for a GGUF file" file_len;
  let bytes =
    match B.borrow Nx_device.host file with
    | Ok b -> Storage.bytes b
    | Error why -> failwith why
  in
  let c = { bytes; pos = 0 } in
  if chars c 4 <> magic then failwith "the file does not start with GGUF";
  let version = version c in
  let n_tensors = count c in
  let n_kv = count c in
  let metadata = metadata c n_kv in
  let alignment = alignment metadata in
  let infos = Array.init n_tensors (fun _ -> tensor_info c) in
  let data_start = (c.pos + alignment - 1) / alignment * alignment in
  let tensors, infos = read_tensors file ~data_start ~alignment infos in
  { version; metadata; tensors; infos }

let load path =
  match B.of_file path with
  | Error why -> failwith why
  | Ok file -> (
      try read file with
      | Failure msg -> fail_msg "%s: %s" path msg
      | Sys_error msg -> failwith msg)

(* Contents *)

let version g = g.version
let metadata g = g.metadata
let tensors g = g.tensors

let info name g =
  match Archive.Names.find_opt name g.infos with
  | Some info -> info
  | None ->
      Printf.ksprintf failwith "Nx_io.Gguf.info: %s: no tensor in the file" name
