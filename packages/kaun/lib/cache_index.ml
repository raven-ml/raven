(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type tokens =
  | Whole of Nx.int32_t
  | Tabled of { row : Nx.int32_t option; pos : Nx.int32_t; table : Nx.int32_t }

type t = { tokens : tokens; window : int option }

let invalid fmt = Printf.ksprintf invalid_arg fmt

(* Lanes are padded on the left, so the last column is every lane's last
   token. *)
let left_padded ~seq lens =
  let batch = Array.length lens in
  Nx.create Nx.int32 [| batch; seq |]
    (Array.init (batch * seq) (fun t ->
         let b = t / seq and i = t mod seq in
         Int32.of_int (max (-1) (i - (seq - lens.(b))))))

let whole ?lens ~batch ~seq () =
  if batch <= 0 || seq <= 0 then
    invalid
      "Cache_index.whole: batch and seq must be positive, got batch=%d seq=%d"
      batch seq;
  let lens = Option.value lens ~default:(Array.make batch seq) in
  if Array.length lens <> batch then
    invalid "Cache_index.whole: %d lengths for %d lanes" (Array.length lens)
      batch;
  Array.iter
    (fun n ->
      if n < 0 || n > seq then
        invalid
          "Cache_index.whole: a lane of %d tokens does not fit %d positions" n
          seq)
    lens;
  { tokens = Whole (left_padded ~seq lens); window = None }

let rows ~context lens =
  let batch = Array.length lens in
  if batch = 0 then invalid_arg "Cache_index.rows: no lanes";
  if context <= 0 then
    invalid "Cache_index.rows: context must be positive, got %d" context;
  Array.iter
    (fun n ->
      if n < 0 || n > context then
        invalid
          "Cache_index.rows: a lane of %d tokens does not fit a context of %d" n
          context)
    lens;
  let seq = Array.fold_left max 1 lens in
  let table =
    Nx.create Nx.int32 [| batch; context |]
      (Array.init (batch * context) Int32.of_int)
  in
  {
    tokens = Tabled { row = None; pos = left_padded ~seq lens; table };
    window = None;
  }

let make ?row ~pos ~table () =
  (match (Nx.shape pos, Nx.shape table) with
  | [| b; s |], [| r; c |] when b > 0 && s > 0 && r > 0 && c > 0 -> (
      match row with
      | None when r = b -> ()
      | Some row when Nx.shape row = [| b |] -> ()
      | None ->
          invalid "Cache_index.make: a table of %d rows for %d lanes needs ~row"
            r b
      | Some _ -> invalid "Cache_index.make: row must have shape [%d]" b)
  | _ ->
      invalid_arg
        "Cache_index.make: pos must have shape [batch; seq] and table [rows; \
         context], neither of them empty");
  { tokens = Tabled { row; pos; table }; window = None }

let window w index =
  if w <= 0 then invalid "Cache_index.window: window must be positive, got %d" w;
  { index with window = Some w }

let raw index = match index.tokens with Whole pos | Tabled { pos; _ } -> pos
let batch index = Nx.dim 0 (raw index)
let seq index = Nx.dim 1 (raw index)

let context index =
  match index.tokens with
  | Whole pos -> Nx.dim 1 pos
  | Tabled { table; _ } -> Nx.dim 1 table

let inside ~below t =
  Nx.logical_and (Nx.greater_equal_s t 0l) (Nx.less_s t (Int32.of_int below))

(* Positions, with every token of a lane whose row is outside the table as
   padding. *)
let pos index =
  match index.tokens with
  | Whole pos | Tabled { row = None; pos; _ } -> pos
  | Tabled { row = Some row; pos; table } ->
      let named = inside ~below:(Nx.dim 0 table) row in
      Nx.where
        (Nx.reshape [| Nx.dim 0 pos; 1 |] named)
        pos (Nx.full_like pos (-1l))

let positions index =
  Nx.clamp ~min:0l ~max:(Int32.of_int (context index - 1)) (raw index)

let advance index =
  match index.tokens with
  | Whole _ -> invalid_arg "Cache_index.advance: a whole index keeps nothing"
  | Tabled { row; table; _ } ->
      (* Any negative position is padding: a lane of it advances to 0. *)
      let last =
        Nx.maximum_s (Nx.max ~axes:[ 1 ] ~keepdims:true (pos index)) (-1l)
      in
      { index with tokens = Tabled { row; pos = Nx.add_s last 1l; table } }

(* The table of each lane. *)
let lanes ~row table =
  match row with
  | None -> table
  | Some row ->
      let last = Int32.of_int (Nx.dim 0 table - 1) in
      Nx.take ~axis:0 ~indices:(Nx.clamp ~min:0l ~max:last row) table

(* Masks *)

(* [keys] broadcasts to [batch; 1; m]: the position each key holds. *)
let sees ?window ~keys pos =
  let query = Nx.reshape [| Nx.dim 0 pos; Nx.dim 1 pos; 1 |] pos in
  let causal = Nx.less_equal keys query in
  match window with
  | None -> causal
  | Some w ->
      Nx.logical_and causal (Nx.greater keys (Nx.sub_s query (Int32.of_int w)))

let column ~context =
  Nx.reshape [| 1; context |] (Nx.arange Nx.int32 0 context 1)

let mask index =
  let pos = pos index and window = index.window in
  match index.tokens with
  | Whole _ ->
      let keys = Nx.reshape [| Nx.dim 0 pos; 1; Nx.dim 1 pos |] pos in
      Nx.logical_and (Nx.greater_equal_s keys 0l) (sees ?window ~keys pos)
  | Tabled { table; _ } ->
      let context = Nx.dim 1 table in
      sees ?window ~keys:(Nx.reshape [| 1; 1; context |] (column ~context)) pos

(* Pools *)

let tail pool =
  let shape = Nx.shape pool in
  if Array.length shape = 0 then
    invalid_arg "Cache_index.extend: a pool has a slot axis";
  Array.sub shape 1 (Array.length shape - 1)

let ones tail = Array.map (fun _ -> 1) tail

(* [pool] with [values] at the slot the table names at each token's own column.
   A token that has none writes the scratch row. *)
let write ~pos ~table values pool =
  let batch = Nx.dim 0 pos and seq = Nx.dim 1 pos in
  let tail = tail pool in
  let slots = Nx.dim 0 pool - 1 and context = Nx.dim 1 table in
  let tokens = batch * seq in
  let own =
    Nx.take_along_axis ~axis:1
      ~indices:(Nx.clamp ~min:0l ~max:(Int32.of_int (context - 1)) pos)
      table
  in
  let addressed =
    Nx.logical_and (inside ~below:context pos) (inside ~below:slots own)
  in
  let target = Nx.where addressed own (Nx.full_like own (Int32.of_int slots)) in
  let indices =
    Nx.broadcast_to
      (Array.append [| tokens |] tail)
      (Nx.reshape (Array.append [| tokens |] (ones tail)) target)
  in
  let values =
    Nx.reshape (Array.append [| tokens |] tail) (Nx.contiguous values)
  in
  Nx.scatter ~unique_indices:true ~axis:0 ~indices ~values pool

(* [pool] at each lane's columns, zero at the columns no token of the lane sees
   and at unallocated ones. *)
let read ?window ~pos ~table pool =
  let tail = tail pool in
  let slots = Nx.dim 0 pool - 1 in
  let batch = Nx.dim 0 table and context = Nx.dim 1 table in
  let column = column ~context in
  let allocated = inside ~below:slots table in
  let seen =
    let last = Nx.max ~axes:[ 1 ] ~keepdims:true pos in
    let upto = Nx.less_equal column last in
    match window with
    | None -> upto
    | Some w ->
        (* The least position of the lane's tokens, or none: padding is -1. *)
        let first =
          Nx.min ~axes:[ 1 ] ~keepdims:true
            (Nx.where
               (Nx.greater_equal_s pos 0l)
               pos
               (Nx.full_like pos (Int32.of_int context)))
        in
        Nx.logical_and upto
          (Nx.greater column (Nx.sub_s first (Int32.of_int w)))
  in
  let slot =
    Nx.where allocated table (Nx.full_like table (Int32.of_int slots))
  in
  let live =
    Nx.reshape
      (Array.append [| batch * context |] (ones tail))
      (Nx.logical_and allocated seen)
  in
  let win =
    Nx.take ~axis:0 ~indices:(Nx.reshape [| batch * context |] slot) pool
  in
  Nx.reshape
    (Array.append [| batch; context |] tail)
    (Nx.where live win (Nx.zeros (Nx.dtype pool) [| 1 |]))

let extend index values pool =
  let tail = tail pool in
  if Nx.shape values <> Array.append [| batch index; seq index |] tail then
    invalid
      "Cache_index.extend: values must have shape [%d; %d] then the pool's"
      (batch index) (seq index);
  match index.tokens with
  | Whole _ -> (values, pool)
  | Tabled { row; table; _ } ->
      let pos = pos index and table = lanes ~row table in
      let pool = write ~pos ~table values pool in
      (read ?window:index.window ~pos ~table pool, pool)

(* Traversals *)

let map f index =
  let tokens =
    match index.tokens with
    | Whole pos -> Whole (f pos)
    | Tabled { row; pos; table } ->
        let row = Option.map f row in
        let pos = f pos in
        let table = f table in
        Tabled { row; pos; table }
  in
  { index with tokens }

let map2 f a b =
  if a.window <> b.window then
    invalid_arg "Cache_index.map2: the indices differ in their window";
  let tokens =
    match (a.tokens, b.tokens) with
    | Whole pos, Whole pos' -> Whole (f pos pos')
    | Tabled a, Tabled b when Option.is_some a.row = Option.is_some b.row ->
        let row =
          match (a.row, b.row) with
          | Some row, Some row' -> Some (f row row')
          | _ -> None
        in
        let pos = f a.pos b.pos in
        let table = f a.table b.table in
        Tabled { row; pos; table }
    | _ ->
        invalid_arg "Cache_index.map2: the indices were not built the same way"
  in
  { a with tokens }

let iter f index =
  match index.tokens with
  | Whole pos -> f pos
  | Tabled { row; pos; table } ->
      Option.iter f row;
      f pos;
      f table
