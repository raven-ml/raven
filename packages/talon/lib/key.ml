(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type use = Identity | Order

(* A float's word is its order key after [-0.] becomes [0.], so that every NaN
   is one word, the greatest. *)
let float x =
  Nx.order_key Nx.uint64 (Nx.where (Nx.equal_s x 0.) (Nx.zeros_like x) x)

(* A complex number's words are its real and imaginary parts', on a last
   axis. *)
let parts re im = Nx.stack ~axis:(Nx.ndim re) [ float re; float im ]

let element : type a b. (a, b) Nx.t -> Nx.uint64_t =
 fun x ->
  match Nx.dtype x with
  | Float16 -> float x
  | Float32 -> float x
  | Float64 -> float x
  | BFloat16 -> float x
  | Float8_e4m3 -> float x
  | Float8_e5m2 -> float x
  | Complex64 -> parts (Nx.real Nx.float32 x) (Nx.imag Nx.float32 x)
  | Complex128 -> parts (Nx.real Nx.float64 x) (Nx.imag Nx.float64 x)
  | _ -> Nx.order_key Nx.uint64 x

(* [code use r] numbers the rows of [r] by their elements. *)
let code use r =
  let code =
    match use with Identity -> Nx_ragged.ids r | Order -> Nx_ragged.rank r
  in
  Nx.cast Nx.uint64 code

(* [rows use x] is the code of each row of the [[n; w]] words [x]. *)
let rows use x =
  let offsets = Nx.arange Nx.int64 0 (Nx.dim 0 x + 1) 1 in
  code use (Nx_ragged.v ~offsets x)

let rec value use c =
  let word =
    match Column.data c with
    | Fixed (P x) when Nx.ndim x = 1 -> element x
    | Fixed (P x) -> rows use (Nx.flatten ~start_dim:1 (element x))
    | Bytes r -> code use r
    | List { offsets; child } ->
        let words =
          match words use child with [ w ] -> w | ws -> Nx.stack ~axis:1 ws
        in
        code use (Nx_ragged.v ~offsets words)
    | Fields [] -> Nx.zeros Nx.uint64 [| Column.length c |]
    | Fields cs -> rows use (Nx.stack ~axis:1 (List.concat_map (words use) cs))
  in
  match Column.valid c with
  | None -> word
  | Some v -> Nx.where v word (Nx.zeros_like word)

(* [words use c] is [c]'s null flag, if it has a null, and its value word. *)
and words use c =
  match Column.valid c with
  | None -> [ value use c ]
  | Some v -> [ Nx.cast Nx.uint64 (Nx.logical_not v); value use c ]

let identity = function
  | [] -> invalid_arg "Key.identity: no column"
  | cs -> Nx.stack ~axis:1 (List.concat_map (words Identity) cs)

(* [of_ids ids] is the groups that [ids], numbered in order of first appearance
   from 0, give their rows. A scatter keeps the last update in index order, so
   over the rows reversed it keeps each group's first row. *)
let of_ids ids : Nx.groups =
  let n = Nx.dim 0 ids in
  let k = if n = 0 then 0 else Int64.to_int (Nx.item [] (Nx.max ids)) + 1 in
  let first =
    Nx.scatter ~axis:0 ~indices:(Nx.flip ids)
      ~values:(Nx.flip (Nx.arange Nx.int64 0 n 1))
      (Nx.zeros Nx.int64 [| k |])
  and counts = Nx.reduce_segments `Add ~segments:k ids (Nx.ones_like ids) in
  { ids; first; counts }

let groups cs =
  match cs with
  | [ c ] when Option.is_none (Column.valid c) -> (
      match Column.data c with
      | Fixed (P x) when Nx.ndim x = 1 -> Nx.unique (identity cs)
      | _ -> of_ids (Nx.bitcast Nx.int64 (value Identity c)))
  | _ -> Nx.unique (identity cs)

let order = function
  | [] -> invalid_arg "Key.order: no key"
  | ks ->
      let key (c, (o : Order.t)) =
        let w = value Order c in
        let w = if o.desc then Nx.bitwise_not w else w in
        match Column.valid c with
        | None -> [ w ]
        | Some v ->
            let null = if o.nulls_first then v else Nx.logical_not v in
            [ Nx.cast Nx.uint64 null; w ]
      in
      Nx.stack ~axis:1 (List.concat_map key ks)
