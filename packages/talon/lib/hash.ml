(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Bigarray.Array1

type words = Nx.host Nx.int64_t

(* xxHash64 of each row, in C ([talon_hash.c]). *)

external hash_words :
  (int64, Bigarray.int64_elt, Bigarray.c_layout) A.t ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  ('a, 'b, Bigarray.c_layout) A.t ->
  unit = "talon_hash_words_byte" "talon_hash_words"
[@@noalloc]

external hash_bytes :
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) A.t ->
  (int64, Bigarray.int64_elt, Bigarray.c_layout) A.t ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  ('a, 'b, Bigarray.c_layout) A.t ->
  unit = "talon_hash_bytes_byte" "talon_hash_bytes"
[@@noalloc]

let by = "Expr.hash"

(* [of_matrix seed x] is the hash of each row of the [[n; w]] words [x]. The
   rows are consecutive once read, so each row is its own [w] words. *)
let of_matrix seed (x : words) =
  let n = Nx.dim 0 x and w = Nx.dim 1 x in
  let a = Strings.fresh Nx.int64 n in
  Strings.fill a (fun out ->
      Strings.reading ~by x Bigarray.int64 (fun words ->
          hash_words words w n seed out));
  Strings.value a

let of_bytes seed (r : Strings.bytes) =
  let n = Nx_ragged.length r in
  let a = Strings.fresh Nx.int64 n in
  Strings.fill a (fun out ->
      Strings.reading ~by (Nx_ragged.offsets r) Bigarray.int64 (fun o ->
          Strings.reading ~by (Nx_ragged.values r) Bigarray.int8_unsigned
            (fun v -> hash_bytes v o n seed out)));
  Strings.value a

(* A value's word

   Every value hashes as one or more little-endian 64-bit words of its value,
   whatever its storage width, as its kind reads it. These widths are the
   format's: they never change. *)

let null_word = 0x9E37_79B1_85EB_CA87L
let canonical_nan = 0x7FF8_0000_0000_0000L

let float x =
  let d = Nx.cast Nx.float64 x in
  let bits = Nx.bitcast Nx.int64 d in
  let bits = Nx.where (Nx.equal_s d 0.) (Nx.zeros_like bits) bits in
  Nx.where (Nx.isnan d) (Nx.scalar Nx.int64 canonical_nan) bits

let word (type a) (ty : a Type.t) (Nx.P x : Nx.host Nx.packed) : words =
  match ty with
  | Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Date ->
      Nx.cast Nx.int64 x
  | Uint64 -> Nx.bitcast Nx.int64 x
  | Float16 | Float32 | Float64 -> float x
  | _ -> (
      match Kernels.ns_per_tick ty with
      | Some ns -> Nx.mul_s (Nx.cast Nx.int64 x) ns
      | None -> invalid_arg "Hash.word: not a hashed kind")

(* The hash of each row

   [hash64 seed ty c] is [c]'s rows hashed, with a null as the null word and,
   for a record, its fields' hashes as its words. Binding admits only these
   types. *)

let rec hash64 : type a. int -> a Type.t -> Column.t -> words =
 fun seed ty c ->
  let h =
    match (ty, Column.data c) with
    | Categorical d, Fixed (P x) ->
        let dict = Column.v Type.string (Iarray.to_array d) in
        let table =
          match Column.data dict with
          | Bytes r -> of_bytes seed r
          | _ -> assert false
        in
        let n = Column.length c in
        if Nx.dim 0 table = 0 then Nx.zeros Nx.int64 [| n |]
        else
          (* A code past the table, under a null, reads zero. *)
          Nx.take (Nx.cast Nx.int64 x) table
    | (String | Binary), Bytes r -> of_bytes seed r
    | Record fields, Fields cs ->
        let hs =
          List.map2
            (fun (_, Type.Any t) c -> hash64 seed t c)
            fields cs
        in
        let n = Column.length c in
        let matrix =
          match hs with
          | [] -> Nx.zeros Nx.int64 [| n; 0 |]
          | hs -> Nx.stack ~axis:1 hs
        in
        of_matrix seed matrix
    | _, Fixed p ->
        let w = word ty p in
        of_matrix seed (Nx.reshape [| Nx.dim 0 w; 1 |] w)
    | _ -> assert false (* Binding admits these types only. *)
  in
  match Column.validity c with
  | None -> h
  | Some v ->
      let null =
        of_matrix seed (Nx.create Nx.int64 [| 1; 1 |] [| null_word |])
      in
      Nx.where (Nx.cast Nx.bool v) h null

let column ~seed c =
  let (Type.Any ty) = Column.type_ c in
  let h = Nx.rshift (Nx.bitcast Nx.uint64 (hash64 seed ty c)) 2 in
  Column.make (Any Type.int64) ~length:(Column.length c)
    (Fixed (Nx.P (Nx.bitcast Nx.int64 h)))
