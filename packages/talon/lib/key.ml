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

let element : type a b. (a, b, Nx.host) Nx.t -> Nx.host Nx.uint64_t =
 fun x ->
  match (Nx.Dtype.kind (Nx.dtype x), Nx.dtype x) with
  | Float, _ -> float x
  | _, Complex64 -> parts (Nx.real Nx.float32 x) (Nx.imag Nx.float32 x)
  | _, Complex128 -> parts (Nx.real Nx.float64 x) (Nx.imag Nx.float64 x)
  | _ -> Nx.order_key Nx.uint64 x

(* Rows of elements of any dtype. *)
type rows = Rows : ('v, 's, Nx.host) Nx_ragged.t -> rows

(* [code use r] numbers the rows of [r] by their elements. Codes are at least
   [0], so their bits are their words. *)
let code use r =
  let code =
    match use with
    | Identity -> (Nx_ragged.unique ~size:0 r).ids
    | Order -> Nx_ragged.rank r
  in
  Nx.bitcast Nx.uint64 code

(* [matrix x] is the [[n; w]] words [x] as [n] rows. *)
let matrix x =
  Nx_ragged.v ~offsets:(Nx.arange Nx.int64 0 (Nx.dim 0 x + 1) 1) x

let rec value use c =
  let word =
    match Column.data c with
    | Fixed (P x) when Nx.ndim x = 1 -> element x
    | Fields [] -> Nx.zeros Nx.uint64 [| Column.length c |]
    | _ ->
        let (Rows r) = rows use c in
        code use r
  in
  (* The values under a null are any: its word is 0, or for [Identity] the
     word of a fixed value's zero, which keeps the words of a column of small
     values within a small span, which {!Nx.unique} numbers by a direct table
     on nx.cpu. *)
  let zero () =
    match (use, Column.data c) with
    | Identity, Fixed (P x) when Nx.ndim x = 1 ->
        element (Nx.zeros_at (Nx.beside x) (Nx.dtype x) [| 1 |])
    | _ -> Nx.zeros_like word
  in
  match Column.validity c with
  | None -> word
  | Some v -> Nx.where (Nx.cast Nx.bool v) word (zero ())

(* [rows use c] is the rows of elements that code the values of [c], a
   compound column: a tensor, text, a list or a record of a field or more. *)
and rows use c =
  match Column.data c with
  | Fixed (P x) -> Rows (matrix (Nx.flatten ~start_dim:1 (element x)))
  | Bytes r -> Rows r
  | List { offsets; child } ->
      let words =
        match words use child with [ w ] -> w | ws -> Nx.stack ~axis:1 ws
      in
      Rows (Nx_ragged.v ~offsets words)
  | Fields cs ->
      Rows (matrix (Nx.stack ~axis:1 (List.concat_map (words use) cs)))

(* [words use c] is [c]'s null flag, if it has a validity, and its value
   word. *)
and words use c =
  match Column.validity c with
  | None -> [ value use c ]
  | Some v -> [ Nx.cast Nx.uint64 (Nx.logical_not v); value use c ]

(* A lone word is its own row, a view, where stacking would copy it. *)
let identity = function
  | [] -> invalid_arg "Key.identity: no column"
  | cs -> (
      match List.concat_map (words Identity) cs with
      | [ w ] -> Nx.reshape [| Nx.dim 0 w; 1 |] w
      | ws -> Nx.stack ~axis:1 ws)

(* A single compound column without a null is grouped by its rows, which
   gives the groups with their numbering. *)
let groups ?size cs =
  match cs with
  | [ c ] when Column.known_zero c -> (
      match Column.data c with
      | Fixed (P x) when Nx.ndim x = 1 -> Nx.unique ?size (identity cs)
      | Fields [] -> Nx.unique ?size (identity cs)
      | _ ->
          let (Rows r) = rows Identity c in
          Nx_ragged.unique ?size r)
  | _ -> Nx.unique ?size (identity cs)

let order = function
  | [] -> invalid_arg "Key.order: no key"
  | ks ->
      let key (c, (o : Order.t)) =
        let w = value Order c in
        let w = if o.desc then Nx.bitwise_not w else w in
        match Column.validity c with
        | None -> [ w ]
        | Some v ->
            let null = if o.nulls_first then v else Nx.logical_not v in
            [ Nx.cast Nx.uint64 null; w ]
      in
      Nx.stack ~axis:1 (List.concat_map key ks)
