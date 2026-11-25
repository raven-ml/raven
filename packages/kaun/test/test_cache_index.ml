(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

(* OCaml's [min_int] and [max_int] stand for int64's extremes. *)
let wide n =
  if n = max_int then Int64.max_int
  else if n = min_int then Int64.min_int
  else Int64.of_int n

let narrow n =
  if n = Int64.max_int then max_int
  else if n = Int64.min_int then min_int
  else Int64.to_int n

let int64s shape a = Nx.create Nx.int64 shape (Array.map wide a)
let flat t = Nx.to_array (Nx.reshape [| -1 |] t)
let ints t = Array.map narrow (flat t)

(* An int64 matrix from its rows. *)
let matrix rows =
  int64s
    [| Array.length rows; Array.length rows.(0) |]
    (Array.concat (Array.to_list rows))

(* The index of tokens at [pos] in sequences held at [table]. *)
let index_at ~pos ~table =
  Cache_index.make ~pos:(matrix pos) ~table:(matrix table) ()

(* Values numbered by token: token [i] of a lane of [n] holds [from + i + 1], so
   what a column reads names the token that stored it. *)
let numbered ~batch ~from n =
  Nx.create Nx.float32 [| batch; n; 1 |]
    (Array.init (batch * n) (fun i -> float_of_int (from + (i mod n) + 1)))

let index_s = Nx.Ptree.v Cache_index.walk

let visits index =
  List.map
    (Format.asprintf "%a" Nx.Ptree.pp_visit)
    (Nx.Ptree.visits index_s index)

let leaves index =
  List.map
    (fun (Nx.P x) -> Nx.to_array (Nx.cast Nx.float64 x))
    (fst (Nx.Ptree.flatten index_s index))

(* [a] and [b] are one index: the same reports and the same tensors. *)
let same ~msg a b =
  equal ~msg:(msg ^ ": reports") (list string) (visits a) (visits b);
  equal ~msg:(msg ^ ": tensors")
    (list (array float_exact))
    (leaves a) (leaves b)

(* The pairs of [batch; 1; seq; context], as an attention reads them. *)
let grid ~batch ~seq ~context : Nx.host Nx.pair =
  let at n shape =
    Nx.reshape shape (Nx.create Nx.int64 [| n |] (Array.init n Int64.of_int))
  in
  {
    q = at seq [| 1; 1; seq; 1 |];
    k = at context [| 1; 1; 1; context |];
    len = Nx.full Nx.int64 [| batch; 1; 1; 1 |] (Int64.of_int context);
    tq = seq;
  }

(* [visible index] at every pair of an attention over the mask's columns. *)
let visible_at index =
  let context = Nx.dim 2 (Cache_index.mask index) in
  Nx.to_array
    (Cache_index.visible index
       (grid ~batch:(Cache_index.batch index) ~seq:(Cache_index.seq index)
          ~context))

(* Drafts are trees: token [i] continues token [parents.(i)], a root the
   table. *)

let depths parents =
  let rec depth i = if parents.(i) < 0 then 0 else 1 + depth parents.(i) in
  Array.init (Array.length parents) depth

(* [sees.(i).(j)]: token [j] is token [i] or one of its ancestors. *)
let tree_sees parents =
  let rec path i = if i < 0 then [] else i :: path parents.(i) in
  let n = Array.length parents in
  Array.init n (fun i -> Array.init n (fun j -> List.mem j (path i)))

(* Each lane's [sees], as a tensor of [lanes; seq; seq]. *)
let cube lanes =
  let n = Array.length lanes.(0) in
  Nx.create Nx.bool
    [| Array.length lanes; n; n |]
    (Array.concat (List.concat_map Array.to_list (Array.to_list lanes)))

let sees_of trees = cube (Array.map tree_sees trees)

(* Drawn tabled indices *)

(* Two lanes of three tokens over tables of four columns, tables of blocks of
   two positions two columns wide, and pools of 18 slots. Sequence [r]'s table
   names slots [4r] to [4r + 3] in some order, its table of blocks slots [2r]
   and [2r + 1], and draft token [i] of lane [b] stores at [12 + 3b + i]. Any of
   these may instead be [-1] or a slot outside the pool. With [row], lanes name
   three sequences, [-1], or a fourth outside the table. *)

let lanes = 2
let tokens = 3
let m = 2
let pool_slots = 18

type tabled = {
  row : int array option;
  pos : int array array;
  table : int array array;
  blocks : int array array;
  every : bool; (* Read in blocks of [m]. *)
  draft : (int array array * bool array array array) option;
      (* Each lane's slots and sees. *)
  window : int option;
  slots : int; (* The pool's. *)
}

let index_of t =
  let row = Option.map (int64s [| lanes |]) t.row in
  let index =
    Cache_index.make ?row
      ~every:[ (m, matrix t.blocks) ]
      ~pos:(matrix t.pos) ~table:(matrix t.table) ()
  in
  let index = if t.every then Cache_index.every m index else index in
  let index =
    match t.draft with
    | None -> index
    | Some (slots, sees) ->
        Cache_index.draft ~slots:(matrix slots) ~sees:(cube sees) index
  in
  match t.window with None -> index | Some w -> Cache_index.window w index

let pp_tabled ppf t =
  let ints a = String.concat "," (Array.to_list (Array.map string_of_int a)) in
  let rows a = String.concat " | " (Array.to_list (Array.map ints a)) in
  let opt pp = Option.fold ~none:"none" ~some:pp in
  Format.fprintf ppf
    "row %s pos %s table %s blocks %s every %b draft %s window %s slots %d"
    (opt ints t.row) (rows t.pos) (rows t.table) (rows t.blocks) t.every
    (opt
       (fun (s, sees) ->
         let bits r =
           String.concat ""
             (Array.to_list (Array.map (fun b -> if b then "1" else "0") r))
         in
         "slots " ^ rows s ^ " sees "
         ^ String.concat " | "
             (Array.to_list
                (Array.map
                   (fun l ->
                     String.concat "," (Array.to_list (Array.map bits l)))
                   sees)))
       t.draft)
    (opt string_of_int t.window)
    t.slots

(* The model of a drawn index, from the laws. *)

(* Lane [b]'s sequence, [None] where its row addresses nothing. *)
let sequence t b =
  let r = match t.row with None -> b | Some row -> row.(b) in
  if r >= 0 && r < Array.length t.table then Some r else None

(* Whether a lane's row addresses nothing. *)
let lost t = List.exists (fun b -> sequence t b = None) [ 0; 1 ]

(* Lane [b]'s positions: padding throughout where its row addresses nothing. *)
let lane_pos t b =
  match sequence t b with Some _ -> t.pos.(b) | None -> Array.make tokens (-1)

(* The table lane [b] reads, in blocks under [every]. *)
let lane_table t b =
  let of_row r = if t.every then t.blocks.(r) else t.table.(r) in
  match sequence t b with
  | Some r -> of_row r
  | None -> Array.make (Array.length (of_row 0)) (-1)

let width t = Array.length (lane_table t 0)

(* The position column [j] stands at: its block's last under [every]. *)
let stands t j = if t.every then ((j + 1) * m) - 1 else j

(* The column that stands at position [p], if one does. *)
let own_column t p =
  if p < 0 then None
  else if not t.every then Some p
  else if (p + 1) mod m = 0 then Some (p / m)
  else None

let real pos = List.filter (fun p -> p >= 0) (Array.to_list pos)
let last pos = List.fold_left max (-1) (real pos)
let least pos = List.fold_left min max_int (real pos)

(* Whether a token at [query] sees a key at [key]: at or before it, and under
   the window among the last [w] positions. *)
let within t ~query key =
  key >= 0 && key <= query
  && match t.window with None -> true | Some w -> key > query - w

let addresses t slot = slot >= 0 && slot < t.slots

(* The slot token [i] of lane [b] stores at (law 3), [None] where it stores
   nothing: padding, a position past the last column, a token that closes no
   block, and a target outside the pool (laws 2 and 5). *)
let target t b i =
  let p = (lane_pos t b).(i) in
  let slot =
    match (t.draft, own_column t p) with
    | Some (slots, _), _ when p >= 0 -> Some slots.(b).(i)
    | None, Some c when c < width t -> Some (lane_table t b).(c)
    | _ -> None
  in
  match slot with Some s when addresses t s -> Some s | _ -> None

(* The tokens aimed at each slot. *)
let writers t =
  let w = Array.make t.slots [] in
  for b = 0 to lanes - 1 do
    for i = 0 to tokens - 1 do
      Option.iter (fun s -> w.(s) <- (b, i) :: w.(s)) (target t b i)
    done
  done;
  w

let collides t = Array.exists (fun w -> List.length w > 1) (writers t)

(* Whether some token of [t] satisfies [f b i p], [p] its position. *)
let some_token t f =
  List.exists
    (fun b ->
      List.exists (fun i -> f b i (lane_pos t b).(i)) (List.init tokens Fun.id))
    [ 0; 1 ]

(* Every number of [t]'s tensors. *)
let numbers t =
  let flat a = List.concat_map Array.to_list (Array.to_list a) in
  Option.fold ~none:[] ~some:Array.to_list t.row
  @ flat t.pos @ flat t.table @ flat t.blocks
  @ Option.fold ~none:[] ~some:(fun (s, _) -> flat s) t.draft

(* Whether a lane has padding after a real token. *)
let padded_after t =
  List.exists
    (fun b ->
      let pos = lane_pos t b in
      List.exists
        (fun i ->
          pos.(i) >= 0
          && Array.exists (fun p -> p < 0) (Array.sub pos i (tokens - i)))
        (List.init tokens Fun.id))
    [ 0; 1 ]

(* Whether a lane's greatest position is not its last token's. *)
let greatest_inside t =
  List.exists
    (fun b ->
      let pos = lane_pos t b in
      last pos > pos.(tokens - 1))
    [ 0; 1 ]

(* Two values per token, and a pool whose rows all differ from them. *)
let values () =
  Nx.create Nx.float32 [| lanes; tokens; 2 |]
    (Array.init
       (lanes * tokens * 2)
       (fun k -> float_of_int ((100 * ((k / 2) + 1)) + (k mod 2))))

let pool t =
  Nx.create Nx.float32 [| t.slots; 2 |]
    (Array.init (2 * t.slots) (fun k -> -.float_of_int (k + 1)))

(* The pool after the call's stores: a slot no token aims at keeps its row, a
   slot one token aims at holds that token's values, and a slot two tokens aim
   at holds, element by element, an unspecified one of theirs (law 4), which
   [actual] names. *)
let stored t actual =
  let before = flat (pool t) and v = flat (values ()) and w = writers t in
  Array.init (2 * t.slots) (fun k ->
      let row (b, i) = v.((((b * tokens) + i) * 2) + (k mod 2)) in
      match w.(k / 2) with
      | [] -> before.(k)
      | [ token ] -> row token
      | aimed ->
          let rows = List.map row aimed in
          if List.mem actual.(k) rows then actual.(k) else List.hd rows)

(* The pages the laws give: lane [b]'s table at each column a token of the lane
   may see, [-1] at a column standing past every position (under a draft, at or
   past the least), or below every token's window; under a draft, then the slot
   each token stores at, [-1] for padding. *)
let expected_pages t =
  Array.init lanes (fun b ->
      let pos = lane_pos t b in
      let least = least pos in
      let upto =
        if t.draft = None then last pos else min (least - 1) (last pos)
      in
      let hidden j =
        let s = stands t j in
        s > upto
        || match t.window with Some w -> s <= least - w | None -> false
      in
      let table =
        Array.mapi
          (fun j slot -> if hidden j then -1 else slot)
          (lane_table t b)
      in
      match t.draft with
      | None -> table
      | Some (slots, _) ->
          Array.append table
            (Array.mapi (fun i s -> if pos.(i) >= 0 then s else -1) slots.(b)))

(* Which columns each token sees under {!Cache_index.mask} and
   {!Cache_index.visible}, [lanes; tokens; context]: a table column at or before
   the token's position and within its window (under a draft, also before the
   lane's least position), and a draft token that [sees] names, at or before the
   position, storing at a slot. The mask hides a hole; [visible] shows it
   without a draft. *)
let expected_sees t =
  let width = width t in
  let context = width + if t.draft = None then 0 else tokens in
  let mask = Array.make (lanes * tokens * context) false in
  let visible = Array.copy mask in
  for b = 0 to lanes - 1 do
    let pos = lane_pos t b in
    for i = 0 to tokens - 1 do
      for j = 0 to context - 1 do
        let k = (((b * tokens) + i) * context) + j in
        if j < width then begin
          let key = stands t j in
          let before = t.draft = None || key < least pos in
          let seen = before && within t ~query:pos.(i) key in
          let hole = (lane_table t b).(j) < 0 in
          mask.(k) <- seen && not hole;
          visible.(k) <- (if t.draft = None then seen else mask.(k))
        end
        else begin
          let slots, sees = Option.get t.draft in
          let token = j - width in
          let seen =
            sees.(b).(i).(token)
            && slots.(b).(token) >= 0
            && within t ~query:pos.(i) pos.(token)
          in
          mask.(k) <- seen;
          visible.(k) <- seen
        end
      done
    done
  done;
  (mask, visible)

(* Whether a token may see a hole: a table column it sees but for its entry,
   which is [-1]. *)
let sees_a_hole t =
  let mask, visible = expected_sees { t with draft = None } in
  t.draft = None && Array.exists2 (fun m v -> v && not m) mask visible

(* [f] over [xs], drawn in order. *)
let rec each f = function
  | [] -> Gen.constant []
  | x :: xs ->
      let open Gen in
      let* y = f x in
      let+ ys = each f xs in
      y :: ys

(* Every negative number reads as [-1] does: [-1], [-2] or int64's least. *)
let negative =
  Gen.frequency
    [
      (3, Gen.constant (-1)); (1, Gen.constant (-2)); (1, Gen.constant min_int);
    ]

let tabled_gen =
  let open Gen in
  (* [slot], or one that addresses nothing: negative, or past the pool. *)
  let entry slot =
    frequency
      [
        (6, constant slot);
        (1, negative);
        ( 1,
          frequency
            [ (3, constant (pool_slots + (slot mod 2))); (1, constant max_int) ]
        );
      ]
  in
  let run ~width r =
    let* order = permutation (List.init width (fun j -> (width * r) + j)) in
    let+ entries = each entry order in
    Array.of_list entries
  in
  let rows ~width =
    let* r0 = run ~width 0 in
    let* r1 = run ~width 1 in
    let+ r2 = run ~width 2 in
    [| r0; r1; r2 |]
  in
  (* A position past the last column from 4, or padding. *)
  let position =
    frequency [ (2, negative); (8, int_range 0 5); (1, constant max_int) ]
  in
  (* [n] tokens from [p0], padded on the left, or positions in any order. *)
  let tokens_from =
    frequency
      [
        ( 2,
          let* n = int_range 0 tokens in
          let+ p0 = int_range 0 5 in
          Array.init tokens (fun i ->
              if i < tokens - n then -1 else p0 + i - (tokens - n)) );
        (1, array ~size:(constant tokens) position);
      ]
  in
  (* A tree of three tokens from [p0], or of two after a token of padding, or
     tokens at any positions; each sees the tree's paths, those and more, or
     every token. *)
  let draft_lane b =
    let* p0 = int_range 0 4 in
    let* p2 = int_range 0 1 in
    let* padded = frequency [ (3, constant false); (1, constant true) ] in
    let parents = if padded then [| -1; -1; 1 |] else [| -1; 0; p2 |] in
    let tree = tree_sees parents in
    let* pos =
      frequency
        [
          ( 2,
            constant
              (Array.mapi
                 (fun i d -> if padded && i = 0 then -1 else p0 + d)
                 (depths parents)) );
          (1, array ~size:(constant tokens) position);
        ]
    in
    let* sees =
      frequency
        [
          (2, constant tree);
          ( 1,
            let+ extra =
              array ~size:(constant tokens) (array ~size:(constant tokens) bool)
            in
            Array.map2 (Array.map2 ( || )) tree extra );
          (1, constant (Array.make_matrix tokens tokens true));
        ]
    in
    let+ slots = each entry (List.init tokens (fun i -> 12 + (3 * b) + i)) in
    (pos, Array.of_list slots, sees)
  in
  let sequence =
    frequency
      [
        (2, constant 0);
        (2, constant 1);
        (2, constant 2);
        (1, negative);
        (1, frequency [ (3, constant 3); (1, constant max_int) ]);
      ]
  in
  let* row =
    frequency
      [
        (1, constant None);
        (1, map Option.some (array ~size:(constant lanes) sequence));
      ]
  in
  let sequences = if row = None then lanes else 3 in
  let* table = rows ~width:4 in
  let* blocks = rows ~width:2 in
  let* kind =
    frequency
      [ (2, constant `Positions); (1, constant `Blocks); (1, constant `Draft) ]
  in
  let* window =
    frequency [ (1, constant None); (1, map Option.some (int_range 1 4)) ]
  in
  let+ pos, draft =
    match kind with
    | `Draft ->
        let+ p0, s0, e0 = draft_lane 0 and+ p1, s1, e1 = draft_lane 1 in
        ([| p0; p1 |], Some ([| s0; s1 |], [| e0; e1 |]))
    | `Positions | `Blocks ->
        let+ l0 = tokens_from and+ l1 = tokens_from in
        ([| l0; l1 |], None)
  in
  {
    row;
    pos;
    table = Array.sub table 0 sequences;
    blocks = Array.sub blocks 0 sequences;
    every = kind = `Blocks;
    draft;
    window;
    slots = pool_slots;
  }

let drawn = Gen.with_pp pp_tabled tabled_gen

(* Laws *)

(* A renaming [sigma] of the pool's slots, in the tables and in the pool, reads
   the same columns and stores each row at its renamed slot. *)
let law_renaming (t, sigma) =
  assume (not (collides t));
  cover "blocks" t.every;
  cover "a draft" (t.draft <> None);
  let sigma = Array.of_list sigma in
  let inverse = Array.make t.slots 0 in
  Array.iteri (fun s u -> inverse.(u) <- s) sigma;
  let rename = Array.map (fun s -> if addresses t s then sigma.(s) else s) in
  let renamed =
    {
      t with
      table = Array.map rename t.table;
      blocks = Array.map rename t.blocks;
      draft = Option.map (fun (s, p) -> (Array.map rename s, p)) t.draft;
    }
  in
  let pool = pool t in
  let seen, pool' = Cache_index.extend (index_of t) (values ()) pool in
  let seen_renamed, pool_renamed =
    Cache_index.extend (index_of renamed) (values ())
      (Nx.take ~axis:0 (int64s [| t.slots |] inverse) pool)
  in
  equal ~msg:"seen" (array float_exact) (flat seen) (flat seen_renamed);
  equal ~msg:"each slot's row at its renamed slot" (array float_exact)
    (flat pool')
    (flat (Nx.take ~axis:0 (int64s [| t.slots |] sigma) pool_renamed))

let renaming_gen =
  Gen.pair drawn (Gen.permutation (List.init pool_slots Fun.id))

(* [extend] compiled, the index an input: one program per structure. *)
let compiled_extend =
  lazy
    (Rune.jit
       Nx.Ptree.(index_s @-> tensor @-> tensor @-> returns (pair tensor tensor))
       Cache_index.extend)

(* [pool'] read through [pages], [lanes; context] flat: zero at a page that
   addresses nothing. *)
let read_through t pages pool' =
  Array.init
    (2 * Array.length pages)
    (fun k ->
      let p = pages.(k / 2) in
      if addresses t p then pool'.((2 * p) + (k mod 2)) else 0.)

let law_reads t =
  Compiled.here ();
  let expected = expected_pages t in
  cover "a lane whose row addresses nothing" (lost t);
  cover "a hole a token may see" (sees_a_hole t);
  cover "a page outside the pool"
    (Array.exists (Array.exists (fun p -> p >= t.slots)) expected);
  cover "a draft token that stores nothing"
    (match t.draft with
    | None -> false
    | Some (slots, _) ->
        some_token t (fun b i p -> p >= 0 && not (addresses t slots.(b).(i))));
  let index = index_of t in
  let pages = ints (Option.get (Cache_index.pages index)) in
  let seen, pool' = Cache_index.extend index (values ()) (pool t) in
  equal ~msg:"eager" (array float_exact)
    (read_through t pages (flat pool'))
    (flat seen);
  let seen, pool' = Lazy.force compiled_extend index (values ()) (pool t) in
  equal ~msg:"compiled" (array float_exact)
    (read_through t pages (flat pool'))
    (flat seen)

let law_stores t =
  Compiled.here ();
  let tabled = t.draft = None in
  let entry b p f =
    match own_column t p with
    | Some c when c < width t -> f (lane_table t b).(c)
    | _ -> false
  in
  cover "a padded token" (some_token t (fun _ _ p -> p < 0));
  cover "a lane whose row addresses nothing" (lost t);
  cover "a position past the last column"
    (tabled
    && some_token t (fun _ _ p ->
        match own_column t p with Some c -> c >= width t | None -> false));
  cover "a token that closes a block"
    (t.every && some_token t (fun b _ p -> entry b p (fun _ -> true)));
  cover "a token that closes no block"
    (t.every && some_token t (fun _ _ p -> p >= 0 && own_column t p = None));
  cover "an unallocated target"
    (tabled && some_token t (fun b _ p -> entry b p (fun s -> s < 0)));
  cover "a target outside the pool"
    (tabled && some_token t (fun b _ p -> entry b p (fun s -> s >= t.slots)));
  cover "a draft token whose slot addresses nothing"
    (match t.draft with
    | None -> false
    | Some (slots, _) ->
        some_token t (fun b i p -> p >= 0 && not (addresses t slots.(b).(i))));
  cover "a draft token at or past the table's width"
    (t.draft <> None
    && some_token t (fun _ _ p -> p >= Array.length t.table.(0)));
  cover "a number below -1" (List.exists (fun n -> n < -1) (numbers t));
  cover "int64's greatest" (List.mem max_int (numbers t));
  cover "two tokens aimed at one slot" (collides t);
  cover "a pool of one slot" (t.slots = 1);
  cover "a pool of no slot" (t.slots = 0);
  let index = index_of t in
  let eager = flat (Cache_index.store index (values ()) (pool t)) in
  equal ~msg:"eager" (array float_exact) (stored t eager) eager;
  let _, compiled = Lazy.force compiled_extend index (values ()) (pool t) in
  let compiled = flat compiled in
  equal ~msg:"compiled" (array float_exact) (stored t compiled) compiled

(* A pool of one slot, whose slot axis has extent 1, read by position and in
   blocks (8989862ab3); a pool of no slot; two lanes of one sequence storing one
   position. *)
let store_examples =
  let one =
    {
      row = None;
      pos = [| [| 0; -1; 1 |]; [| -1; 0; 1 |] |];
      table = [| [| -1; 0; 2; 3 |]; [| 5; 6; 7; 0 |] |];
      blocks = [| [| 0; 1 |]; [| 2; 3 |] |];
      every = false;
      draft = None;
      window = None;
      slots = 1;
    }
  in
  let three =
    {
      one with
      row = Some [| 0; 0 |];
      table = [| [| 0; 1; 2; 3 |]; [| 4; 5; 6; 7 |]; [| 8; 9; 10; 11 |] |];
      blocks = [| [| 0; 1 |]; [| 2; 3 |]; [| 4; 5 |] |];
    }
  in
  [
    one;
    { one with every = true; pos = [| [| 0; 1; 2 |]; [| -1; -1; -1 |] |] };
    { three with row = Some [| 1; -1 |]; slots = 0 };
    {
      three with
      pos = [| [| -1; 0; 1 |]; [| -1; -1; 1 |] |];
      slots = pool_slots;
    };
  ]

(* One slot, so the pool's slot axis has extent 1: a token that stores nothing
   leaves the slot as a token that stores there left it, eagerly and compiled
   (8989862ab3). *)
let one_slot_rows =
  [
    ("padding after the token", [| 0; -1 |], [| 0 |], 10.);
    ("padding before the token", [| -1; 0 |], [| 0 |], 20.);
    ("an unallocated column before it", [| 0; 1 |], [| -1; 0 |], 20.);
  ]

let test_one_slot (_, pos, table, expected) =
  Compiled.here ();
  let values =
    Nx.create Nx.float32 [| 1; 2; 2; 4 |]
      (Array.init 16 (fun i -> if i < 8 then 10. else 20.))
  in
  let pool = Nx.zeros Nx.float32 [| 1; 2; 4 |] in
  let stored pos =
    let table = int64s [| 1; Array.length table |] table in
    snd (Cache_index.extend (Cache_index.make ~pos ~table ()) values pool)
  in
  let pos = int64s [| 1; 2 |] pos in
  equal ~msg:"eager" (array float_exact) (Array.make 8 expected)
    (flat (stored pos));
  equal ~msg:"compiled" (array float_exact) (Array.make 8 expected)
    (flat (Rune.jit Nx.Ptree.(tensor @-> returns tensor) stored pos))

let law_pages t =
  cover "blocks" t.every;
  cover "a draft" (t.draft <> None);
  cover "a padded draft token"
    (t.draft <> None && some_token t (fun _ _ p -> p < 0));
  cover "a window" (t.window <> None);
  cover "a hole a token may see" (sees_a_hole t);
  cover "a lane whose row addresses nothing" (lost t);
  cover "padding after a real token" (padded_after t);
  cover "a lane whose greatest position is not its last token"
    (greatest_inside t);
  cover "a number below -1" (List.exists (fun n -> n < -1) (numbers t));
  cover "int64's greatest" (List.mem max_int (numbers t));
  (* Every negative entry reads as [-1] does. *)
  let unallocated = Array.map (fun p -> if p < 0 then -1 else p) in
  equal (array int)
    (unallocated (Array.concat (Array.to_list (expected_pages t))))
    (unallocated (ints (Option.get (Cache_index.pages (index_of t)))))

(* Steps at other positions, over other tables and under other windows replay
   the first step's program: no trace and no lowering of their own
   (5b514a0ab2). *)
let test_one_program () =
  Compiled.here ();
  let traces = ref 0 in
  let step =
    Rune.jit
      Nx.Ptree.(
        index_s @-> tensor @-> tensor
        @-> returns (pair tensor (pair tensor tensor)))
      (fun index values pool ->
        incr traces;
        let seen, pool = Cache_index.extend index values pool in
        (seen, (pool, Cache_index.mask index)))
  in
  let call (pos, table, w) =
    let index =
      Cache_index.window w (index_at ~pos:[| pos |] ~table:[| table |])
    in
    let values = numbered ~batch:1 ~from:0 2 in
    let pool = Nx.create Nx.float32 [| 4; 1 |] [| -1.; -2.; -3.; -4. |] in
    let msg what = Printf.sprintf "window %d: %s" w what in
    let seen, pool' = Cache_index.extend index values pool in
    let seen_c, (pool_c, mask_c) = step index values pool in
    equal ~msg:(msg "seen") (array float_exact) (flat seen) (flat seen_c);
    equal ~msg:(msg "pool") (array float_exact) (flat pool') (flat pool_c);
    equal ~msg:(msg "mask") (array bool)
      (flat (Cache_index.mask index))
      (flat mask_c)
  in
  call ([| 0; 1 |], [| 3; 0; 2; 1 |], 1);
  let lowered = Rune.lowerings () in
  List.iter call
    [
      ([| 2; 3 |], [| 3; 0; 2; 1 |], 3);
      ([| 1; 2 |], [| 1; -1; 0; 2 |], max_int);
      ([| -1; 3 |], [| 2; 3; 1; 0 |], 2);
    ];
  equal ~msg:"traces" int 1 !traces;
  equal ~msg:"lowerings after the first step" int lowered (Rune.lowerings ())

(* make *)

(* Lanes 0 and 1 hold positions 0 and 1 of sequence 0: lane 1 reads what lane 0
   stores in the same call. *)
let test_lanes_of_one_sequence () =
  let index =
    Cache_index.make
      ~row:(int64s [| 2 |] [| 0; 0 |])
      ~pos:(int64s [| 2; 1 |] [| 0; 1 |])
      ~table:(int64s [| 1; 3 |] [| 2; 0; 1 |])
      ()
  in
  let seen, _ =
    Cache_index.extend index
      (Nx.create Nx.float32 [| 2; 1; 1 |] [| 10.; 20. |])
      (Nx.zeros Nx.float32 [| 3; 1 |])
  in
  equal (array float_exact) [| 10.; 0.; 0.; 10.; 20.; 0. |] (flat seen)

let refuses fn (_, f) =
  raises_match (Exn.invalid_arg ~sub:("Cache_index." ^ fn ^ ":")) f

let three_blocks = int64s [| 1; 3 |] [| 0; 1; 2 |]

let make_refusals =
  let table = int64s [| 1; 4 |] [| 0; 1; 2; 3 |] in
  let pos = int64s [| 1; 1 |] [| 0 |] in
  [
    ( "positions of one axis",
      fun () -> Cache_index.make ~pos:(int64s [| 1 |] [| 0 |]) ~table () );
    ( "an empty axis",
      fun () -> Cache_index.make ~pos:(int64s [| 1; 0 |] [||]) ~table () );
    ( "a table of one row for two lanes, without row",
      fun () -> Cache_index.make ~pos:(int64s [| 2; 1 |] [| 0; 0 |]) ~table ()
    );
    ( "a row of two lanes for one",
      fun () -> Cache_index.make ~row:(int64s [| 2 |] [| 0; 0 |]) ~pos ~table ()
    );
    ( "a block of one position",
      fun () -> Cache_index.make ~every:[ (1, three_blocks) ] ~pos ~table () );
    ( "two pairs with one m",
      fun () ->
        Cache_index.make
          ~every:[ (4, three_blocks); (4, three_blocks) ]
          ~pos ~table () );
    ( "a table of blocks of two rows for one",
      fun () ->
        Cache_index.make
          ~every:[ (4, int64s [| 2; 3 |] (Array.make 6 0)) ]
          ~pos ~table () );
  ]

(* rows *)

let test_rows_runs () =
  let values =
    Nx.create Nx.float32 [| 2; 3; 1 |] [| 10.; 11.; 12.; 20.; 21.; 22. |]
  in
  equal (array float_exact)
    [| 11.; 12.; 0.; 0.; 20.; 21.; 22.; 0. |]
    (flat
       (Cache_index.store
          (Cache_index.rows ~context:4 [| 2; 3 |])
          values
          (Nx.zeros Nx.float32 [| 8; 1 |])))

let test_rows_blocks () =
  let index =
    Cache_index.every 4 (Cache_index.rows ~every:[ 4 ] ~context:10 [| 4; 4 |])
  in
  equal (array float_exact)
    [| 4.; 0.; 0.; 4.; 0.; 0. |]
    (flat
       (Cache_index.store index
          (numbered ~batch:2 ~from:0 4)
          (Nx.zeros Nx.float32 [| 6; 1 |])))

(* Blocks of 4 over a context of 10: block 2 holds positions 8 to 11, and no
   position of the context closes it. *)
let test_rows_last_block () =
  let index =
    Cache_index.every 4 (Cache_index.rows ~every:[ 4 ] ~context:10 [| 10 |])
  in
  let pool =
    Cache_index.store index
      (numbered ~batch:1 ~from:0 10)
      (Nx.zeros Nx.float32 [| 3; 1 |])
  in
  equal ~msg:"stored" (array float_exact) [| 4.; 8.; 0. |] (flat pool);
  equal ~msg:"seen by position 9" (array bool) [| true; true; false |]
    (flat (Nx.slice [ I 0; I 9 ] (Cache_index.mask index)))

let rows_refusals =
  [
    ("no lane", fun () -> Cache_index.rows ~context:4 [||]);
    ("a context of 0", fun () -> Cache_index.rows ~context:0 [| 0 |]);
    ("a negative length", fun () -> Cache_index.rows ~context:4 [| -1 |]);
    ("a length past the context", fun () -> Cache_index.rows ~context:4 [| 5 |]);
    ( "a block of one position",
      fun () -> Cache_index.rows ~every:[ 1 ] ~context:4 [| 2 |] );
    ( "blocks of 4 twice",
      fun () -> Cache_index.rows ~every:[ 4; 4 ] ~context:4 [| 2 |] );
  ]

(* whole *)

let test_whole_keeps_nothing () =
  let index = Cache_index.whole ~batch:2 ~seq:3 () in
  let values = numbered ~batch:2 ~from:0 3 in
  let pool = Nx.create Nx.float32 [| 4; 1 |] [| -1.; -2.; -3.; -4. |] in
  let seen, pool' = Cache_index.extend index values pool in
  equal ~msg:"seen" (array float_exact) (flat values) (flat seen);
  equal ~msg:"extend's pool" (array float_exact) (flat pool) (flat pool');
  equal ~msg:"store" (array float_exact) (flat pool)
    (flat (Cache_index.store index values pool))

let whole_refusals =
  [
    ("a batch of 0", fun () -> Cache_index.whole ~batch:0 ~seq:2 ());
    ("a seq of 0", fun () -> Cache_index.whole ~batch:1 ~seq:0 ());
    ( "a length past seq",
      fun () -> Cache_index.whole ~lens:[| 3 |] ~batch:1 ~seq:2 () );
    ( "a negative length",
      fun () -> Cache_index.whole ~lens:[| -1 |] ~batch:1 ~seq:2 () );
  ]

(* packed *)

let test_packed_positions () =
  equal (array int) [| 0; 0; 1; 0; 1 |]
    (ints (Cache_index.positions (Cache_index.packed ~seq:5 [| [| 2; 2 |] |])))

let test_packed_singletons () =
  let lens = [| 3; 5; 0 |] in
  let whole = Cache_index.whole ~lens ~batch:3 ~seq:5 ()
  and packed = Cache_index.packed ~seq:5 (Array.map (fun n -> [| n |]) lens) in
  let mask index = flat (Cache_index.mask index) in
  let seen index =
    flat
      (fst
         (Cache_index.extend index
            (numbered ~batch:3 ~from:0 5)
            (Nx.zeros Nx.float32 [| 0; 1 |])))
  in
  equal ~msg:"mask" (array bool) (mask whole) (mask packed);
  equal ~msg:"mask under a window" (array bool)
    (mask (Cache_index.window 2 whole))
    (mask (Cache_index.window 2 packed));
  equal ~msg:"seen" (array float_exact) (seen whole) (seen packed);
  equal ~msg:"positions" (array int)
    (ints (Cache_index.positions whole))
    (ints (Cache_index.positions packed))

let test_packed_program () =
  Compiled.here ();
  let traces = ref 0 in
  let mask =
    Rune.jit
      Nx.Ptree.(index_s @-> returns tensor)
      (fun index ->
        incr traces;
        Cache_index.mask index)
  in
  List.iter
    (fun lens ->
      let index = Cache_index.packed ~seq:6 lens in
      equal ~msg:"mask" (array bool)
        (flat (Cache_index.mask index))
        (flat (mask index)))
    [ [| [| 2; 4 |]; [| 6 |] |]; [| [| 1; 1; 3 |]; [| 3; 2 |] |] ];
  equal ~msg:"traces" int 1 !traces

let packed_refusals =
  [
    ("no lane", fun () -> Cache_index.packed ~seq:5 [||]);
    ("a seq of 0", fun () -> Cache_index.packed ~seq:0 [| [||] |]);
    ( "a negative length",
      fun () -> Cache_index.packed ~seq:5 [| [| 1 |]; [| -1 |] |] );
    ( "lengths summing past seq",
      fun () -> Cache_index.packed ~seq:5 [| [| 3; 3 |] |] );
  ]

(* window *)

let test_window_replaces () =
  let index = Cache_index.rows ~context:4 [| 3 |] in
  same ~msg:"window 2 over window 1"
    (Cache_index.window 2 index)
    (Cache_index.window 2 (Cache_index.window 1 index))

(* Each kind of index, under windows of one past its largest position, half of
   [max_int] and [max_int]. *)
let widest_rows =
  let rows () = Cache_index.rows ~every:[ 2 ] ~context:6 [| 4; 2 |] in
  [
    ("rows", rows);
    ("a lane of padding", fun () -> Cache_index.rows ~context:6 [| 4; 0 |]);
    ("rows in blocks", fun () -> Cache_index.every 2 (rows ()));
    ("an advanced index", fun () -> Cache_index.advance (rows ()));
    ( "a padded whole index",
      fun () -> Cache_index.whole ~lens:[| 3; 5 |] ~batch:2 ~seq:5 () );
    ( "a packed index",
      fun () -> Cache_index.packed ~seq:6 [| [| 2; 3 |]; [| 6 |] |] );
    ( "a draft",
      fun () ->
        Cache_index.draft
          ~slots:(int64s [| 1; 3 |] [| 19; 18; 17 |])
          ~sees:(sees_of [| [| -1; 0; 0 |] |])
          (index_at
             ~pos:[| [| 2; 3; 3 |] |]
             ~table:[| [| 7; 2; 11; 0; 5; 9; 1; 10; 3; 6; 4; 8 |] |]) );
    ( "a selection",
      fun () ->
        Cache_index.select
          (int64s [| 1; 2; 3 |] [| 1; 3; 0; 4; 2; 0 |])
          (index_at ~pos:[| [| 2; 4 |] |] ~table:[| [| 3; 0; 4; -1; 2; 1 |] |])
    );
  ]

let test_widest (_, make) =
  Nx.Rng.with_key (Nx.Rng.key 49) @@ fun () ->
  let index = make () in
  let batch = Cache_index.batch index and seq = Cache_index.seq index in
  let columns = Nx.dim 2 (Cache_index.mask index) in
  let q = Nx.Rng.normal Nx.float32 [| batch; 4; seq; 2 |] in
  let k = Nx.Rng.normal Nx.float32 [| batch; 2; columns; 2 |] in
  let v = Nx.Rng.normal Nx.float32 [| batch; 2; columns; 3 |] in
  let values = Nx.Rng.normal Nx.float32 [| batch; seq; 3 |] in
  let pool = Nx.Rng.normal Nx.float32 [| 20; 3 |] in
  let attended index =
    flat
      (Attention.attend ~mask:(Cache_index.visible index) q (Nx.Dense { k; v }))
  in
  let seen index = flat (fst (Cache_index.extend index values pool)) in
  let largest =
    Int64.to_int (Nx.item [] (Nx.max (Cache_index.positions index)))
  in
  List.iter
    (fun w ->
      let widest = Cache_index.window w index in
      let msg what = Printf.sprintf "window %d: %s" w what in
      equal ~msg:(msg "mask") (array bool)
        (flat (Cache_index.mask index))
        (flat (Cache_index.mask widest));
      equal ~msg:(msg "seen") (array float_exact) (seen index) (seen widest);
      equal ~msg:(msg "attention") (array float_exact) (attended index)
        (attended widest))
    [ largest + 1; max_int / 2; max_int ]

(* Two indices stacked along a new leading axis map; under a window, they do
   not. *)
let test_window_vmap () =
  let positions =
    Rune.vmap Nx.Ptree.(index_s @-> returns tensor) Cache_index.positions
  in
  let a = index_at ~pos:[| [| 1; 7 |] |] ~table:[| [| 3; -1; 0; 1 |] |] in
  let b = index_at ~pos:[| [| 0; -1 |] |] ~table:[| [| 5; 2; 4; 9 |] |] in
  let stacked = Nx.Ptree.map2 index_s (fun _ x y -> Nx.stack [ x; y ]) a b in
  equal ~msg:"without a window" (array int)
    (Array.append
       (ints (Cache_index.positions a))
       (ints (Cache_index.positions b)))
    (ints (positions stacked));
  raises_match (Exn.invalid_arg ~sub:"window") (fun () ->
      positions (Cache_index.window 2 stacked))

let window_refusals =
  let rows () = Cache_index.rows ~context:4 [| 2 |] in
  [
    ("0", fun () -> Cache_index.window 0 (rows ()));
    ("-1", fun () -> Cache_index.window (-1) (rows ()));
  ]

(* select *)

(* [k] columns per token among [0] to [4], negative numbers and int64's
   greatest: some outside the context, some chosen twice. *)
let columns_gen =
  let open Gen in
  let* k = int_range 1 3 in
  let column =
    frequency [ (1, negative); (6, int_range 0 4); (1, constant max_int) ]
  in
  let+ chosen = array ~size:(constant (lanes * tokens * k)) column in
  (k, chosen)

let selection_gen = Gen.pair drawn columns_gen

(* A selection reads, at each chosen column its token sees without a selection,
   what the read without it gives there, and zero elsewhere. *)
let law_selection (t, (k, chosen)) =
  let t = { t with draft = None } in
  let plain = index_of t in
  let selected =
    Cache_index.select (int64s [| lanes; tokens; k |] chosen) plain
  in
  let context = Cache_index.context plain in
  let mask = flat (Cache_index.mask plain) in
  let seen = flat (fst (Cache_index.extend plain (values ()) (pool t))) in
  let _, visible = expected_sees t in
  let token x = x / k in
  let lane x = token x / tokens in
  let inside x = chosen.(x) >= 0 && chosen.(x) < context in
  let sees x = inside x && mask.((token x * context) + chosen.(x)) in
  let choices = List.init (Array.length chosen) Fun.id in
  cover "a window" (t.window <> None);
  cover "blocks" t.every;
  cover "a column outside the context"
    (List.exists (fun x -> not (inside x)) choices);
  cover "a column chosen twice"
    (List.exists
       (fun x ->
         List.exists
           (fun y -> y <> x && token y = token x && chosen.(y) = chosen.(x))
           choices)
       choices);
  cover "a chosen hole the token would see"
    (List.exists
       (fun x ->
         inside x && visible.((token x * context) + chosen.(x)) && not (sees x))
       choices);
  equal ~msg:"mask" (array bool)
    (Array.init (Array.length chosen) sees)
    (flat (Cache_index.mask selected));
  equal ~msg:"seen" (array float_exact)
    (Array.init
       (2 * Array.length chosen)
       (fun y ->
         let x = y / 2 in
         if sees x then
           seen.((((lane x * context) + chosen.(x)) * 2) + (y mod 2))
         else 0.))
    (flat (fst (Cache_index.extend selected (values ()) (pool t))))

let law_selection_stores (t, (k, chosen)) =
  let t = { t with draft = None } in
  cover "blocks" t.every;
  let plain = index_of t in
  let selected =
    Cache_index.select (int64s [| lanes; tokens; k |] chosen) plain
  in
  equal (array float_exact)
    (flat (Cache_index.store plain (values ()) (pool t)))
    (flat (snd (Cache_index.extend selected (values ()) (pool t))))

let test_select_replaces () =
  let index = Cache_index.rows ~context:4 [| 2 |] in
  let chosen = int64s [| 1; 2; 1 |] [| 0; 1 |] in
  same ~msg:"the later selection"
    (Cache_index.select chosen index)
    (Cache_index.select chosen
       (Cache_index.select (int64s [| 1; 2; 2 |] [| 1; 0; 0; 1 |]) index))

(* On a whole index a chosen column is a token of the lane, or a block it
   closes. Tokens hold their number. *)
let whole_selections =
  let pairs n p = Array.concat (List.init n (fun _ -> p)) in
  [
    ( "a whole index, lane 0 padded by one token",
      fun () ->
        let index = Cache_index.whole ~lens:[| 2; 3 |] ~batch:2 ~seq:3 () in
        let columns = int64s [| 2; 3; 3 |] (pairs 6 [| 2; 0; 1 |]) in
        ( Cache_index.select columns index,
          numbered ~batch:2 ~from:0 3,
          [|
            0.;
            0.;
            0.;
            0.;
            0.;
            2.;
            3.;
            0.;
            2.;
            0.;
            1.;
            0.;
            0.;
            1.;
            2.;
            3.;
            1.;
            2.;
          |] ) );
    ( "a packed lane of two sequences of two",
      fun () ->
        let index = Cache_index.packed ~seq:5 [| [| 2; 2 |] |] in
        let columns = int64s [| 1; 5; 2 |] [| 0; 0; 1; 1; 1; 2; 1; 3; 2; 4 |] in
        ( Cache_index.select columns index,
          numbered ~batch:1 ~from:0 5,
          [| 0.; 0.; 2.; 2.; 2.; 3.; 0.; 4.; 0.; 5. |] ) );
    ( "blocks of 4, lane 0 padded by four tokens",
      fun () ->
        let index =
          Cache_index.every 4
            (Cache_index.whole ~lens:[| 6; 10 |] ~batch:2 ~seq:10 ())
        in
        let columns =
          Nx.broadcast_to [| 2; 10; 2 |] (int64s [| 1; 1; 2 |] [| 1; 0 |])
        in
        ( Cache_index.select columns index,
          numbered ~batch:2 ~from:0 10,
          Array.concat
            [
              pairs 7 [| 0.; 0. |];
              pairs 3 [| 0.; 8. |];
              pairs 3 [| 0.; 0. |];
              pairs 4 [| 0.; 4. |];
              pairs 3 [| 8.; 4. |];
            ] ) );
  ]

let test_whole_selection (_, make) =
  let index, values, expected = make () in
  let seen, _ =
    Cache_index.extend index values (Nx.zeros Nx.float32 [| 0; 1 |])
  in
  equal (array float_exact) expected (flat seen)

(* 8989862ab3: a broadcast selection on a whole index failed. *)
let test_select_broadcast () =
  let columns =
    Nx.broadcast_to [| 1; 3; 2 |] (int64s [| 1; 1; 2 |] [| 0; 2 |])
  in
  let seen, _ =
    Cache_index.extend
      (Cache_index.select columns (Cache_index.whole ~batch:1 ~seq:3 ()))
      (numbered ~batch:1 ~from:0 3)
      (Nx.zeros Nx.float32 [| 0; 1 |])
  in
  equal (array float_exact) [| 1.; 0.; 1.; 0.; 1.; 3. |] (flat seen)

(* Positions 2 and 4 over a shuffled table whose column 3 is unallocated; the
   columns choose positions after the token's, outside the context and the
   hole. *)
let test_select_compiled () =
  Compiled.here ();
  let index =
    index_at ~pos:[| [| 2; 4 |] |] ~table:[| [| 3; 0; 4; -1; 2; 1 |] |]
  in
  let pool = Nx.create Nx.float32 [| 5; 1 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let values = Nx.create Nx.float32 [| 1; 2; 1 |] [| 30.; 50. |] in
  let extend = Lazy.force compiled_extend in
  List.iter
    (fun chosen ->
      let selected = Cache_index.select (int64s [| 1; 2; 5 |] chosen) index in
      let seen, pool' = Cache_index.extend selected values pool in
      let seen_c, pool_c = extend selected values pool in
      equal ~msg:"seen" (array float_exact) (flat seen) (flat seen_c);
      equal ~msg:"pool" (array float_exact) (flat pool') (flat pool_c))
    [ [| 1; 3; 2; -1; 1; 5; 6; 0; 4; 0 |]; [| 0; 1; 2; 3; 4; 4; 3; 2; 1; 0 |] ]

let select_refusals =
  let rows () = Cache_index.rows ~context:4 [| 2 |] in
  [
    ( "columns of two axes",
      fun () -> Cache_index.select (int64s [| 1; 2 |] [| 0; 1 |]) (rows ()) );
    ( "no column per token",
      fun () -> Cache_index.select (int64s [| 1; 2; 0 |] [||]) (rows ()) );
    ( "a draft",
      fun () ->
        Cache_index.select
          (int64s [| 1; 1; 1 |] [| 0 |])
          (Cache_index.draft
             ~slots:(int64s [| 1; 1 |] [| 5 |])
             ~sees:(Nx.full Nx.bool [| 1; 1; 1 |] true)
             (Cache_index.rows ~context:4 [| 1 |])) );
  ]

(* draft *)

(* Positions 0 and 1 stored over a table of 4 columns; a root at 2 and two
   children at 3. Column 2's slot holds NaN, which the draft does not read. *)
let test_draft_seen () =
  let pool =
    Nx.create Nx.float32 [| 7; 1 |] [| 2.; 0.; nan; 1.; 0.; 0.; 0. |]
  in
  let index =
    Cache_index.draft
      ~slots:(int64s [| 1; 3 |] [| 6; 4; 5 |])
      ~sees:(sees_of [| [| -1; 0; 0 |] |])
      (index_at ~pos:[| [| 2; 3; 3 |] |] ~table:[| [| 3; 0; 2; 1 |] |])
  in
  let values = Nx.create Nx.float32 [| 1; 3; 1 |] [| 30.; 40.; 41. |] in
  equal (array float_exact)
    [| 1.; 2.; 0.; 0.; 30.; 40.; 41. |]
    (flat (fst (Cache_index.extend index values pool)))

let test_draft_replaces () =
  let index =
    Cache_index.window 2
      (index_at ~pos:[| [| 2; 3; 3 |] |] ~table:[| [| 3; 0; 2; 1 |] |])
  in
  let draft slots parents =
    Cache_index.draft
      ~slots:(int64s [| 1; 3 |] slots)
      ~sees:(sees_of [| parents |])
  in
  same ~msg:"the later draft, under the window"
    (draft [| 5; 6; 4 |] [| -1; 0; 1 |] index)
    (draft [| 5; 6; 4 |] [| -1; 0; 1 |]
       (draft [| 6; 4; 5 |] [| -1; 0; 0 |] index))

let draft_refusals =
  let one = int64s [| 1; 1 |] [| 0 |] in
  let sees = Nx.full Nx.bool [| 1; 1; 1 |] true in
  let rows () = Cache_index.rows ~context:3 [| 1 |] in
  [
    ( "a whole index",
      fun () ->
        Cache_index.draft ~slots:one ~sees
          (Cache_index.whole ~batch:1 ~seq:1 ()) );
    ( "a packed index",
      fun () ->
        Cache_index.draft ~slots:one ~sees
          (Cache_index.packed ~seq:1 [| [| 1 |] |]) );
    ( "slots of another shape",
      fun () ->
        Cache_index.draft ~slots:(int64s [| 1; 2 |] [| 0; 1 |]) ~sees (rows ())
    );
    ( "sees of another shape",
      fun () ->
        Cache_index.draft ~slots:one
          ~sees:(Nx.full Nx.bool [| 1; 1 |] true)
          (rows ()) );
    ( "a selection",
      fun () ->
        Cache_index.draft ~slots:one ~sees
          (Cache_index.select (int64s [| 1; 1; 1 |] [| 0 |]) (rows ())) );
    ( "blocks",
      fun () ->
        Cache_index.draft ~slots:one ~sees
          (Cache_index.every 2
             (Cache_index.rows ~every:[ 2 ] ~context:4 [| 1 |])) );
  ]

(* every *)

(* Lane 0 holds 6 tokens padded by 4, lane 1 holds 10: block 0 closes at
   position 3, block 1 at 7, and block 2 at 11, past both lanes. *)
let test_every_whole () =
  let index =
    Cache_index.every 4
      (Cache_index.whole ~lens:[| 6; 10 |] ~batch:2 ~seq:10 ())
  in
  let seen, _ =
    Cache_index.extend index
      (numbered ~batch:2 ~from:0 10)
      (Nx.zeros Nx.float32 [| 0; 1 |])
  in
  equal (array float_exact) [| 8.; 0.; 0.; 4.; 8.; 0. |] (flat seen)

(* A toy compressed stream: each token stores its value at its position; the
   token that closes a block of 4 reads its block's positions through a
   selection and stores their sum as the block's entry; each token then sums the
   entries it sees. *)

type stream = {
  index : Nx.host Cache_index.t;
  x : Nx.host Nx.float32_t;
  y : Nx.host Nx.float32_t;
  sources : Nx.host Nx.float32_t;
  entries : Nx.host Nx.float32_t;
}

let stream_s =
  Nx.Ptree.v (fun c (s : stream) ->
      let open Nx.Ptree.Walk in
      let i = field c "index" Cache_index.walk s.index in
      let x = field c "x" tensor s.x in
      let y = field c "y" tensor s.y in
      let sources = field c "sources" tensor s.sources in
      let entries = field c "entries" tensor s.entries in
      { index = i; x; y; sources; entries })

let stream s =
  let batch = Cache_index.batch s.index and seq = Cache_index.seq s.index in
  let pos = Cache_index.positions s.index in
  let first = Nx.reshape [| batch; seq; 1 |] (Nx.sub pos (Nx.mod_s pos 4L)) in
  let block =
    Nx.add first (Nx.reshape [| 1; 1; 4 |] (Nx.arange Nx.int64 0 4 1))
  in
  let own, sources =
    Cache_index.extend (Cache_index.select block s.index) s.x s.sources
  in
  let blocks = Cache_index.every 4 s.index in
  let seen, entries =
    Cache_index.extend blocks (Nx.sum ~axes:[ 2 ] own) s.entries
  in
  let context = Cache_index.context blocks in
  let y =
    Nx.sum ~axes:[ 2 ]
      (Nx.where (Cache_index.mask blocks)
         (Nx.reshape [| batch; 1; context |] seen)
         (Nx.zeros Nx.float32 [| 1 |]))
  in
  { s with y; sources; entries }

(* Blocks sum positions plus one, so block j holds 16 j + 10, and position t
   sees it once (j + 1) * 4 <= t + 1. *)
let stream_outputs =
  List.init 12 (fun t ->
      List.fold_left ( +. ) 0.
        (List.init 3 (fun j ->
             if (j + 1) * 4 <= t + 1 then float_of_int ((16 * j) + 10) else 0.)))

let stream_entries = [| 42.; 10.; 26. |]

(* A sequence of 12 positions fed in calls, each a list of lanes of one
   sequence, each lane a list of positions, over shuffled tables. *)
let feed_stream ?(step = stream) calls =
  let table = int64s [| 1; 12 |] [| 5; 11; 0; 7; 2; 9; 4; 1; 10; 3; 8; 6 |] in
  let blocks = int64s [| 1; 3 |] [| 1; 2; 0 |] in
  let ys, s =
    List.fold_left
      (fun (ys, s) lanes ->
        let batch = List.length lanes and seq = List.length (List.hd lanes) in
        let pos = int64s [| batch; seq |] (Array.of_list (List.concat lanes)) in
        let index =
          Cache_index.make
            ~row:(int64s [| batch |] (Array.make batch 0))
            ~every:[ (4, blocks) ]
            ~pos ~table ()
        in
        let x =
          Nx.add_s (Nx.cast Nx.float32 (Nx.reshape [| batch; seq; 1 |] pos)) 1.
        in
        let s =
          step { s with index; x; y = Nx.zeros Nx.float32 [| batch; seq |] }
        in
        (List.rev_append (Array.to_list (flat s.y)) ys, s))
      ( [],
        {
          index = Cache_index.rows ~context:1 [| 1 |];
          x = Nx.zeros Nx.float32 [| 1; 1; 1 |];
          y = Nx.zeros Nx.float32 [| 1; 1 |];
          sources = Nx.zeros Nx.float32 [| 12; 1 |];
          entries = Nx.zeros Nx.float32 [| 3; 1 |];
        } )
      calls
  in
  (List.rev ys, s.entries)

let feedings =
  [
    ("whole", [ [ List.init 12 Fun.id ] ]);
    ( "chunks of 3 that split every block",
      List.init 4 (fun c -> [ List.init 3 (fun i -> (3 * c) + i) ]) );
    ("token by token", List.init 12 (fun t -> [ [ t ] ]));
    ( "one-token lanes that close blocks 1 and 2 in one call",
      [ [ [ 0; 1; 2; 3 ] ]; List.init 8 (fun i -> [ 4 + i ]) ] );
  ]

let test_stream (_, calls) =
  let ys, entries = feed_stream calls in
  equal ~msg:"outputs" (list float_exact) stream_outputs ys;
  equal ~msg:"entries" (array float_exact) stream_entries (flat entries)

let test_stream_whole () =
  let s =
    stream
      {
        index = Cache_index.whole ~batch:1 ~seq:12 ();
        x = numbered ~batch:1 ~from:0 12;
        y = Nx.zeros Nx.float32 [| 1; 12 |];
        sources = Nx.zeros Nx.float32 [| 0; 1 |];
        entries = Nx.zeros Nx.float32 [| 0; 1 |];
      }
  in
  equal (list float_exact) stream_outputs (Array.to_list (flat s.y))

let test_stream_compiled () =
  Compiled.here ();
  let step = Rune.jit Nx.Ptree.(stream_s @-> returns stream_s) stream in
  let ys, entries =
    feed_stream ~step
      [ [ [ 0; 1; 2 ] ]; [ [ 3; 4; 5 ] ]; [ [ 6; 7; 8 ] ]; [ [ 9; 10; 11 ] ] ]
  in
  equal ~msg:"outputs" (list float_exact) stream_outputs ys;
  equal ~msg:"entries" (array float_exact) stream_entries (flat entries)

let every_identities =
  let rows () = Cache_index.rows ~every:[ 4 ] ~context:12 [| 4 |] in
  let four () = Cache_index.every 4 (rows ()) in
  let whole () = Cache_index.whole ~batch:1 ~seq:3 () in
  [
    ("every 1 of rows", fun () -> (rows (), Cache_index.every 1 (rows ())));
    ( "every 1 of a whole index",
      fun () -> (whole (), Cache_index.every 1 (whole ())) );
    ( "every 4 of rows read in blocks of 4",
      fun () -> (four (), Cache_index.every 4 (four ())) );
  ]

let test_every_identity (_, make) =
  let index, every = make () in
  same ~msg:"every" index every

let test_every_keeps_the_window () =
  let rows = Cache_index.rows ~every:[ 4 ] ~context:12 [| 4 |] in
  same ~msg:"every 4 under window 3"
    (Cache_index.window 3 (Cache_index.every 4 rows))
    (Cache_index.every 4 (Cache_index.window 3 rows))

let every_refusals =
  let rows () = Cache_index.rows ~every:[ 4; 128 ] ~context:12 [| 4 |] in
  [
    ("m of 0", fun () -> Cache_index.every 0 (rows ()));
    ("no table for blocks of 8", fun () -> Cache_index.every 8 (rows ()));
    ( "blocks of 4 read already",
      fun () -> Cache_index.every 128 (Cache_index.every 4 (rows ())) );
    ( "a selection",
      fun () ->
        Cache_index.every 4
          (Cache_index.select (int64s [| 1; 4; 1 |] [| 0; 1; 2; 3 |]) (rows ()))
    );
    ( "a draft",
      fun () ->
        Cache_index.every 4
          (Cache_index.draft
             ~slots:(int64s [| 1; 4 |] [| 0; 1; 2; 3 |])
             ~sees:(Nx.full Nx.bool [| 1; 4; 4 |] true)
             (rows ())) );
    ( "a packed index",
      fun () -> Cache_index.every 2 (Cache_index.packed ~seq:4 [| [| 2; 2 |] |])
    );
  ]

(* advance *)

(* The next position, clamped to the table's last column by [positions]. A lane
   at int64's greatest position has no next one. *)
let law_advance t =
  assume (not (some_token t (fun _ _ p -> p = max_int)));
  cover "a lane whose greatest position is not its last token"
    (greatest_inside t);
  cover "a lane of padding"
    (List.exists (fun b -> last (lane_pos t b) < 0) [ 0; 1 ]);
  cover "a next position inside the table"
    (List.exists
       (fun b -> last (lane_pos t b) + 1 < Array.length t.table.(0))
       [ 0; 1 ]);
  let index = Cache_index.advance (index_of t) in
  let last_column = Array.length t.table.(0) - 1 in
  equal ~msg:"seq" int 1 (Cache_index.seq index);
  equal ~msg:"positions" (array int)
    (Array.init lanes (fun b -> min last_column (last (lane_pos t b) + 1)))
    (ints (Cache_index.positions index))

(* Lanes 0 and 1 hold positions 0 and 1 of one sequence: each lane's next token
   follows its own. *)
let test_advance_per_lane () =
  let index =
    Cache_index.make
      ~row:(int64s [| 2 |] [| 0; 0 |])
      ~pos:(int64s [| 2; 1 |] [| 0; 1 |])
      ~table:(int64s [| 1; 4 |] [| 0; 1; 2; 3 |])
      ()
  in
  equal (array int) [| 1; 2 |]
    (ints (Cache_index.positions (Cache_index.advance index)))

(* A padded token sees nothing, and one at position 0 sees column 0. *)
let test_advance_padding () =
  let index = index_at ~pos:[| [| -1; -1 |] |] ~table:[| [| 0; 1 |] |] in
  equal (array bool) [| true; false |]
    (flat (Cache_index.mask (Cache_index.advance index)))

(* The next token is at position 4. Under window 2 it sees positions 3 and 4;
   blocks of 2 stand at 1, 3 and 5. *)
let advance_keeps =
  let rows every = Cache_index.rows ~every ~context:6 [| 4 |] in
  [
    ( "a window",
      (fun () -> Cache_index.window 2 (rows [])),
      [| false; false; false; true; true; false |] );
    ( "blocks",
      (fun () -> Cache_index.every 2 (rows [ 2 ])),
      [| true; true; false |] );
    ( "blocks under a window",
      (fun () -> Cache_index.window 2 (Cache_index.every 2 (rows [ 2 ]))),
      [| false; true; false |] );
  ]

let test_advance_keeps (_, make, expected) =
  equal (array bool) expected
    (flat (Cache_index.mask (Cache_index.advance (make ()))))

let advance_drops =
  let rows () = Cache_index.rows ~context:4 [| 2 |] in
  [
    ( "the selection",
      "columns: None",
      fun () -> Cache_index.select (int64s [| 1; 2; 1 |] [| 0; 1 |]) (rows ())
    );
    ( "the draft",
      "draft: None",
      fun () ->
        Cache_index.draft
          ~slots:(int64s [| 1; 2 |] [| 4; 5 |])
          ~sees:(sees_of [| [| -1; 0 |] |])
          (rows ()) );
  ]

let test_advance_drops (_, report, make) =
  mem string report (visits (Cache_index.advance (make ())))

let advance_refusals =
  [
    ( "a whole index",
      fun () -> Cache_index.advance (Cache_index.whole ~batch:1 ~seq:2 ()) );
    ( "a packed index",
      fun () -> Cache_index.advance (Cache_index.packed ~seq:2 [| [| 2 |] |]) );
  ]

(* The call's tokens *)

let counted f (_, make, expected) = equal int expected (f (make ()))

let batches =
  [
    ("rows", (fun () -> Cache_index.rows ~context:6 [| 3; 5 |]), 2);
    ("whole", (fun () -> Cache_index.whole ~batch:3 ~seq:4 ()), 3);
    ( "packed",
      (fun () -> Cache_index.packed ~seq:5 [| [| 1 |]; [||]; [| 2; 3 |] |]),
      3 );
    ( "make",
      (fun () ->
        index_at ~pos:[| [| 0; 1 |]; [| 0; 1 |] |] ~table:[| [| 0 |]; [| 1 |] |]),
      2 );
  ]

let seqs =
  [
    ( "rows: the longest length",
      (fun () -> Cache_index.rows ~context:6 [| 3; 5 |]),
      5 );
    ("whole", (fun () -> Cache_index.whole ~batch:3 ~seq:4 ()), 4);
    ( "packed",
      (fun () -> Cache_index.packed ~seq:5 [| [| 1 |]; [| 2; 3 |] |]),
      5 );
    ( "make",
      (fun () -> index_at ~pos:[| [| 0; 1; 2 |] |] ~table:[| [| 0 |] |]),
      3 );
  ]

let contexts =
  [
    ( "a table's width",
      (fun () -> index_at ~pos:[| [| 0 |] |] ~table:[| Array.init 6 Fun.id |]),
      6 );
    ( "a draft: the table's width plus seq",
      (fun () ->
        Cache_index.draft
          ~slots:(int64s [| 1; 3 |] [| 4; 5; 6 |])
          ~sees:(sees_of [| [| -1; 0; 0 |] |])
          (index_at ~pos:[| [| 2; 3; 3 |] |] ~table:[| [| 3; 0; 2; 1 |] |])),
      7 );
    ("a whole index: seq", (fun () -> Cache_index.whole ~batch:1 ~seq:5 ()), 5);
    ( "blocks: the width of the table of blocks",
      (fun () ->
        Cache_index.every 4 (Cache_index.rows ~every:[ 4 ] ~context:10 [| 2 |])),
      3 );
    ( "a whole index in blocks of 4: (seq + 3) / 4",
      (fun () -> Cache_index.every 4 (Cache_index.whole ~batch:1 ~seq:10 ())),
      3 );
  ]

let positions_rows =
  [
    ( "a table's clamped to 0 and its last column",
      (fun () -> index_at ~pos:[| [| -1; 1; 7 |] |] ~table:[| [| 0; 1 |] |]),
      [| 0; 1; 1 |] );
    ( "a whole index's with padding at 0",
      (fun () -> Cache_index.whole ~lens:[| 1 |] ~batch:1 ~seq:3 ()),
      [| 0; 0; 0 |] );
    ( "in blocks, clamped to the positions' table",
      (fun () ->
        Cache_index.every 4
          (Cache_index.make
             ~every:[ (4, three_blocks) ]
             ~pos:(int64s [| 1; 3 |] [| 9; 10; 11 |])
             ~table:(int64s [| 1; 10 |] (Array.init 10 Fun.id))
             ())),
      [| 9; 9; 9 |] );
  ]

let test_positions (_, make, expected) =
  equal (array int) expected (ints (Cache_index.positions (make ())))

(* Pools *)

(* Values of another shape than [[| 1; 2; 2 |]], for a lane of two tokens and a
   pool of [[| 4; 2 |]]. *)
let wrong_values =
  [
    ("another batch", [| 2; 2; 2 |]);
    ("another seq", [| 1; 3; 2 |]);
    ("another width", [| 1; 2; 3 |]);
  ]

let refuses_values fn f (_, shape) =
  raises_match
    (Exn.invalid_arg ~sub:("Cache_index." ^ fn ^ ":"))
    (fun () ->
      f
        (Cache_index.rows ~context:4 [| 2 |])
        (Nx.zeros Nx.float32 shape)
        (Nx.zeros Nx.float32 [| 4; 2 |]))

(* [a] and [b] address alike: the same pages, positions, masks, reads and
   stores. *)
let address_alike ~msg a b =
  let shape = [| Cache_index.batch a; Cache_index.seq a; 2 |] in
  let values =
    Nx.create Nx.float32 shape
      (Array.init (Array.fold_left ( * ) 1 shape) float_of_int)
  in
  let pool =
    Nx.create Nx.float32 [| pool_slots; 2 |]
      (Array.init (2 * pool_slots) (fun k -> -.float_of_int (k + 1)))
  in
  let msg what = msg ^ ": " ^ what in
  equal ~msg:(msg "pages")
    (option (array int))
    (Option.map ints (Cache_index.pages a))
    (Option.map ints (Cache_index.pages b));
  equal ~msg:(msg "positions") (array int)
    (ints (Cache_index.positions a))
    (ints (Cache_index.positions b));
  equal ~msg:(msg "mask") (array bool)
    (flat (Cache_index.mask a))
    (flat (Cache_index.mask b));
  equal ~msg:(msg "visible") (array bool) (visible_at a) (visible_at b);
  let seen, pool' = Cache_index.extend a values pool in
  let seen_b, pool_b = Cache_index.extend b values pool in
  equal ~msg:(msg "seen") (array float_exact) (flat seen) (flat seen_b);
  equal ~msg:(msg "extend's pool") (array float_exact) (flat pool')
    (flat pool_b);
  equal ~msg:(msg "store") (array float_exact)
    (flat (Cache_index.store a values pool))
    (flat (Cache_index.store b values pool))

let law_resolved (t, (k, chosen)) =
  cover "a draft" (t.draft <> None);
  cover "blocks" t.every;
  cover "a window" (t.window <> None);
  let index = index_of t in
  address_alike ~msg:"the index" index (Cache_index.resolve index);
  if t.draft = None then begin
    let selected =
      Cache_index.select (int64s [| lanes; tokens; k |] chosen) index
    in
    address_alike ~msg:"a selection" selected (Cache_index.resolve selected)
  end

(* Compiled, a resolved index stores where the laws say and reads its pool
   through the index's pages. *)
let law_resolved_compiled t =
  Compiled.here ();
  cover "a draft" (t.draft <> None);
  cover "blocks" t.every;
  cover "a window" (t.window <> None);
  let index = index_of t in
  let pages = ints (Option.get (Cache_index.pages index)) in
  let seen, pool' =
    Lazy.force compiled_extend (Cache_index.resolve index) (values ()) (pool t)
  in
  let pool' = flat pool' in
  equal ~msg:"stored" (array float_exact) (stored t pool') pool';
  equal ~msg:"seen" (array float_exact) (read_through t pages pool') (flat seen)

let resolved_kinds =
  let rows () = Cache_index.rows ~every:[ 2 ] ~context:6 [| 3; 2 |] in
  [
    ("rows", rows);
    ("blocks of 2", fun () -> Cache_index.every 2 (rows ()));
    ( "blocks under a window",
      fun () -> Cache_index.window 3 (Cache_index.every 2 (rows ())) );
    ( "a row per lane",
      fun () ->
        Cache_index.window 2
          (Cache_index.make
             ~row:(int64s [| 3 |] [| 1; -1; 0 |])
             ~pos:(int64s [| 3; 2 |] [| 2; 3; 0; 1; -1; 0 |])
             ~table:(int64s [| 2; 4 |] [| 4; 5; 6; 7; 8; -1; 10; 11 |])
             ()) );
  ]

let test_resolved_kind (_, make) =
  let index = make () in
  address_alike ~msg:"resolved" index (Cache_index.resolve index)

(* What {!Cache_index.window}, {!Cache_index.advance}, {!Cache_index.every} to
   another block size, {!Cache_index.select} and {!Cache_index.draft} make from
   a resolved index. *)
let made_from_resolved =
  let rows () =
    Cache_index.window 2 (Cache_index.rows ~every:[ 2 ] ~context:6 [| 3; 2 |])
  in
  [
    ("another window", Cache_index.window 3);
    ("advance", Cache_index.advance);
    ("every 2", Cache_index.every 2);
    ( "a selection",
      Cache_index.select (int64s [| 2; 3; 1 |] [| 0; 1; 2; 5; 0; 1 |]) );
    ( "a draft",
      fun index ->
        Cache_index.draft
          ~slots:(int64s [| 2; 1 |] [| 11; 10 |])
          ~sees:(Nx.full Nx.bool [| 2; 1; 1 |] true)
          (Cache_index.advance index) );
  ]
  |> List.map (fun (id, f) -> (id, rows, f))

let test_made_from_resolved (_, rows, f) =
  address_alike ~msg:"made from it"
    (f (rows ()))
    (f (Cache_index.resolve (rows ())))

let test_unresolved (_, rows, f) =
  mem string "resolved: None" (visits (f (Cache_index.resolve (rows ()))))

let resolved_unchanged =
  [
    ("whole", fun () -> Cache_index.whole ~lens:[| 2; 3 |] ~batch:2 ~seq:3 ());
    ("packed", fun () -> Cache_index.packed ~seq:3 [| [| 1; 2 |]; [| 3 |] |]);
  ]

let test_resolved_unchanged (_, make) =
  same ~msg:"resolved" (make ()) (Cache_index.resolve (make ()))

(* The kernels a compiled [f] launches at a call after its first, on a software
   device. *)
let launches f =
  ignore (f ());
  let _, events = Rig.Profile.take (fun () -> ignore (f ())) in
  List.concat_map
    (function
      | Rig.Profile.Work { name; _ } when name <> "copies" ->
          String.split_on_char ',' name
      | _ -> [])
    events

(* Compiled, a store through a resolved index launches the store alone, and an
   attention under [visible] through [pages] launches what it launches over a
   table given as an argument. *)
let test_resolved_launches () =
  Compiled.here ();
  let device = Result.get_ok (Rig.software "kaun-test-resolve") in
  let module D = (val Nx.devices [ device ]) in
  let index_s = Nx.Ptree.v Cache_index.walk in
  let index =
    Cache_index.place D.on
      (Cache_index.window 2
         (Cache_index.advance (Cache_index.rows ~context:8 [| 3 |])))
  in
  let r =
    Rune.jit Nx.Ptree.(index_s @-> returns index_s) Cache_index.resolve index
  in
  let placed t = Nx.place D.on t in
  let values = placed (Nx.ones Nx.float32 [| 1; 1; 2; 2 |]) in
  let pool () = placed (Nx.ones Nx.float32 [| 8; 2; 2 |]) in
  let store =
    Rune.jit
      Nx.Ptree.(index_s @-> tensor @-> tensor @-> returns tensor)
      Cache_index.store
  in
  equal ~msg:"a store" int 1
    (List.length
       (launches (fun () -> Nx.to_array (store r values (Nx.donate (pool ()))))));
  let q = placed (Nx.ones Nx.float32 [| 1; 4; 1; 2 |]) in
  let paged pool = Nx.reshape [| 8; 2; 1; 2 |] pool in
  let attend_index =
    Rune.jit
      Nx.Ptree.(index_s @-> tensor @-> tensor @-> tensor @-> returns tensor)
      (fun index q k v ->
        let table = Option.get (Cache_index.pages index) in
        Attention.attend
          ~mask:(Cache_index.visible index)
          q
          (Nx.Paged { k = paged k; v = paged v; table }))
  and attend_table =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> tensor @-> tensor @-> returns tensor)
      (fun table q k v ->
        Attention.attend q (Nx.Paged { k = paged k; v = paged v; table }))
  in
  let table = Option.get (Cache_index.pages r) in
  let k = pool () and v = pool () in
  equal ~msg:"an attention" int
    (List.length (launches (fun () -> Nx.to_array (attend_table table q k v))))
    (List.length (launches (fun () -> Nx.to_array (attend_index r q k v))))

let law_store_is_extend t =
  cover "a draft" (t.draft <> None);
  cover "blocks" t.every;
  let index = index_of t in
  equal (array float_exact)
    (flat (snd (Cache_index.extend index (values ()) (pool t))))
    (flat (Cache_index.store index (values ()) (pool t)))

let no_pages =
  [
    ("a whole index", fun () -> Cache_index.whole ~batch:1 ~seq:2 ());
    ("a packed index", fun () -> Cache_index.packed ~seq:2 [| [| 1; 1 |] |]);
    ( "a selection",
      fun () ->
        Cache_index.select
          (int64s [| 1; 2; 1 |] [| 0; 1 |])
          (Cache_index.rows ~context:4 [| 2 |]) );
  ]

let test_no_pages (_, make) = is_none ~pp:Nx.pp (Cache_index.pages (make ()))

(* mask and visible *)

(* An index made by a constructor, and what the reference needs: a column stands
   at [(j + 1) stride - 1]; a whole index at stride 1 instead has its tokens as
   columns, which see their own sequence ([runs]). *)
type constructed = {
  case : string;
  index : Nx.host Cache_index.t;
  context : int;
  stride : int;
  runs : bool;
  window : int option;
  pos : int array; (* Each token's position, [-1] for padding. *)
}

type shown =
  | Constructed of constructed
  | Drawn of tabled
  | Selected of tabled * int * int array (* [k] columns per token. *)

let pp_shown ppf = function
  | Constructed c ->
      Format.fprintf ppf "%s context %d stride %d window %s" c.case c.context
        c.stride
        (Option.fold ~none:"none" ~some:string_of_int c.window)
  | Drawn t -> pp_tabled ppf t
  | Selected (t, _, chosen) ->
      Format.fprintf ppf "%a columns %s" pp_tabled t
        (String.concat "," (Array.to_list (Array.map string_of_int chosen)))

(* A lane of [len] tokens padded on the left of [seq] positions. *)
let left_padded ~seq len = Array.init seq (fun i -> max (-1) (i - (seq - len)))

let constructed_gen =
  let open Gen in
  let* batch = int_range 1 3 in
  let lens n = array ~size:(constant batch) (int_range 0 n) in
  let padded ~seq ls =
    Array.concat (Array.to_list (Array.map (left_padded ~seq) ls))
  in
  let* kind =
    frequency
      [ (1, constant `Rows); (1, constant `Whole); (1, constant `Packed) ]
  in
  let* case, index, context, stride, runs, pos =
    match kind with
    | `Rows ->
        let* context = int_range 1 9 in
        let* ls = lens context in
        let* blocks = bool in
        let+ advanced = bool in
        let every = if blocks then [ 3 ] else [] in
        let rows = Cache_index.rows ~every ~context ls in
        let seq = Array.fold_left max 1 ls in
        let index, pos =
          if advanced then
            (Cache_index.advance rows, Array.map (fun l -> max 0 l) ls)
          else (rows, padded ~seq ls)
        in
        ( (if advanced then "advanced rows" else "rows"),
          (if blocks then Cache_index.every 3 index else index),
          (if blocks then (context + 2) / 3 else context),
          (if blocks then 3 else 1),
          false,
          pos )
    | `Whole ->
        let* seq = int_range 1 9 in
        let* ls = lens seq in
        let+ blocks = bool in
        let index = Cache_index.whole ~lens:ls ~batch ~seq () in
        if blocks then
          ( "whole in blocks",
            Cache_index.every 2 index,
            (seq + 1) / 2,
            2,
            false,
            padded ~seq ls )
        else ("whole", index, seq, 1, true, padded ~seq ls)
    | `Packed ->
        let* seq = int_range 1 9 in
        let* cut = int_range 0 seq in
        let+ n = int_range 0 (seq - cut) in
        let lane =
          Array.concat
            [
              Array.make (seq - cut - n) (-1);
              Array.init cut Fun.id;
              Array.init n Fun.id;
            ]
        in
        ( "packed",
          Cache_index.packed ~seq (Array.make batch [| cut; n |]),
          seq,
          1,
          true,
          Array.concat (List.init batch (fun _ -> lane)) )
  in
  let+ window =
    frequency [ (1, constant None); (1, map Option.some (int_range 1 4)) ]
  in
  let index =
    Option.fold ~none:index ~some:(fun w -> Cache_index.window w index) window
  in
  Constructed { case; index; context; stride; runs; window; pos }

let shown_gen =
  Gen.with_pp pp_shown
    (Gen.frequency
       [
         (3, constructed_gen);
         (3, Gen.map (fun t -> Drawn t) tabled_gen);
         ( 1,
           Gen.map
             (fun (t, (k, chosen)) ->
               Selected ({ t with draft = None }, k, chosen))
             (Gen.pair tabled_gen columns_gen) );
       ])

(* What each query of [c] sees: the columns at or before its position, under a
   window only the last [w] of them. A constructor's index has no hole, so the
   mask and [visible] agree. *)
let constructed_sees c =
  let batch = Cache_index.batch c.index and seq = Cache_index.seq c.index in
  let seen = Array.make (batch * seq * c.context) false in
  for b = 0 to batch - 1 do
    for i = 0 to seq - 1 do
      let query = c.pos.((b * seq) + i) in
      for j = 0 to c.context - 1 do
        let key, related =
          if not c.runs then (((j + 1) * c.stride) - 1, true)
          else
            (* The key is token [j] of the lane: seen if one sequence, whose
               tokens start together, holds both. *)
            let p = c.pos.((b * seq) + j) in
            (p, p >= 0 && j - p = i - query)
        in
        seen.((((b * seq) + i) * c.context) + j) <-
          (related && key >= 0 && key <= query
          && match c.window with None -> true | Some w -> key > query - w)
      done
    done
  done;
  seen

(* [s]'s kind, index, mask and [visible]. Under a selection a token sees a
   chosen column inside the context that it sees without the selection. *)
let shown_case = function
  | Constructed c -> (c.case, c.index, constructed_sees c, constructed_sees c)
  | Drawn t ->
      let mask, visible = expected_sees t in
      ( (if t.draft = None then "a table" else "a draft"),
        index_of t,
        mask,
        visible )
  | Selected (t, k, chosen) ->
      let plain, _ = expected_sees t and context = width t in
      let sees =
        Array.mapi
          (fun x c -> c >= 0 && c < context && plain.((x / k * context) + c))
          chosen
      in
      let columns = int64s [| lanes; tokens; k |] chosen in
      ("a selection", Cache_index.select columns (index_of t), sees, sees)

let cover_shown s =
  let case, _, mask, visible = shown_case s in
  List.iter
    (fun c -> cover c (case = c))
    [
      "rows";
      "advanced rows";
      "whole";
      "whole in blocks";
      "packed";
      "a table";
      "a draft";
      "a selection";
    ];
  cover "under a window"
    (match s with
    | Constructed c -> c.window <> None
    | Drawn t | Selected (t, _, _) -> t.window <> None);
  cover "a hole a token may see" (Array.exists2 ( <> ) mask visible);
  let named f =
    match s with
    | Drawn ({ draft = Some (_, sees); _ } as t) ->
        List.exists
          (fun b ->
            let pos = lane_pos t b in
            List.exists
              (fun i ->
                List.exists
                  (fun j -> sees.(b).(i).(j) && f pos.(i) pos.(j))
                  (List.init tokens Fun.id))
              (List.init tokens Fun.id))
          [ 0; 1 ]
    | _ -> false
  in
  cover "sees names a later token" (named (fun q k -> q >= 0 && k > q));
  cover "sees names a padded token" (named (fun q k -> q >= 0 && k < 0))

(* A draft at positions 2, 3 and padding whose tokens all see each other: none
   sees a later position or padding. *)
let all_seeing =
  Drawn
    {
      row = None;
      pos = [| [| 2; 3; -1 |]; [| -1; -1; -1 |] |];
      table = [| [| 3; 0; 2; 1 |]; [| 4; 5; 6; 7 |] |];
      blocks = [| [| 0; 1 |]; [| 2; 3 |] |];
      every = false;
      draft =
        Some
          ( [| [| 12; 13; 14 |]; [| 15; 16; 17 |] |],
            Array.make 2 (Array.make_matrix tokens tokens true) );
      window = None;
      slots = pool_slots;
    }

let law_mask s =
  cover_shown s;
  let _, index, mask, _ = shown_case s in
  equal (array bool) mask (flat (Cache_index.mask index))

let law_visible s =
  cover_shown s;
  let _, index, _, visible = shown_case s in
  equal (array bool) visible (visible_at index)

(* [mask index] and [visible index] at every pair of an attention over the
   mask's columns. *)
let masks index =
  let context = Nx.dim 2 (Cache_index.mask index) in
  ( Cache_index.mask index,
    Cache_index.visible index
      (grid ~batch:(Cache_index.batch index) ~seq:(Cache_index.seq index)
         ~context) )

(* Indices of one structure and shape that differ in their values: positions,
   tables, lengths, windows from 1 to int64's greatest, drafts, choices. *)
let compiled_kinds =
  let rows lens = Cache_index.rows ~every:[ 2 ] ~context:6 lens in
  let table pos table = index_at ~pos ~table in
  let windows ws index = List.map (fun w -> Cache_index.window w index) ws in
  let widths = [ 1; 3; max_int ] in
  let whole lens = Cache_index.whole ~lens ~batch:2 ~seq:5 () in
  let draft slots trees =
    Cache_index.draft
      ~slots:(int64s [| 1; 3 |] slots)
      ~sees:(sees_of trees)
      (table [| [| 2; 3; 3 |] |] [| [| 3; 0; -1; 1 |] |])
  in
  let selection chosen =
    Cache_index.select
      (int64s [| 1; 2; 3 |] chosen)
      (table [| [| 2; 4 |] |] [| [| 3; 0; 4; -1; 2; 1 |] |])
  in
  [
    ( "tables",
      fun () ->
        [
          table [| [| 0; 1; 2 |] |] [| [| 3; 0; 2; -1 |] |];
          table [| [| 3; -1; 1 |] |] [| [| 1; 2; 3; 0 |] |];
        ] );
    ( "tables under windows",
      fun () ->
        windows widths (table [| [| 2; 0; 3 |] |] [| [| 3; -1; 2; 1 |] |]) );
    ( "rows in blocks",
      fun () ->
        [
          Cache_index.every 2 (rows [| 4; 2 |]);
          Cache_index.every 2 (rows [| 1; 5 |]);
        ] );
    ( "blocks under windows",
      fun () -> windows widths (Cache_index.every 2 (rows [| 4; 2 |])) );
    ("whole", fun () -> [ whole [| 3; 5 |]; whole [| 5; 0 |] ]);
    ("whole under windows", fun () -> windows widths (whole [| 3; 5 |]));
    ( "whole in blocks",
      fun () ->
        [
          Cache_index.every 2 (whole [| 3; 5 |]);
          Cache_index.every 2 (whole [| 1; 4 |]);
        ] );
    ( "packed under windows",
      fun () ->
        [
          Cache_index.window 2
            (Cache_index.packed ~seq:6 [| [| 2; 3 |]; [| 6 |] |]);
          Cache_index.window max_int
            (Cache_index.packed ~seq:6 [| [| 1; 1; 3 |]; [| 3; 2 |] |]);
        ] );
    ( "advanced under windows",
      fun () ->
        [
          Cache_index.advance (Cache_index.window 2 (rows [| 4; 2 |]));
          Cache_index.advance (Cache_index.window max_int (rows [| 1; 5 |]));
        ] );
    ( "drafts",
      fun () ->
        [
          draft [| 6; 4; 5 |] [| [| -1; 0; 0 |] |];
          draft [| 5; -1; 6 |] [| [| -1; 0; 1 |] |];
        ] );
    ( "selections under windows",
      fun () ->
        [
          Cache_index.window 2 (selection [| 1; 3; 0; 4; 2; 0 |]);
          Cache_index.window max_int (selection [| 5; -1; 2; 0; 0; 4 |]);
        ] );
  ]

(* Kinds of 70 columns with tokens near the end, one at a decode step's shape: a
   token per lane. *)
let wide_kinds =
  let table pos table = index_at ~pos ~table in
  let windows ws index = List.map (fun w -> Cache_index.window w index) ws in
  let seventy =
    [|
      Array.init 70 (fun j -> j * 37 mod 70);
      Array.init 70 (fun j -> if j = 50 then -1 else 70 + j);
    |]
  in
  [
    ( "70 columns under windows",
      fun () ->
        windows [ 1; 3; 20; max_int ]
          (table [| [| 66; 67; 68 |]; [| 40; 69; -1 |] |] seventy) );
    ( "a decode step over 70 columns under windows",
      fun () ->
        windows [ 1; 3; 20; max_int ] (table [| [| 68 |]; [| 69 |] |] seventy)
    );
    ( "70 blocks of 2 under windows",
      fun () ->
        let lanes n =
          Array.init 2 (fun b -> Array.init n (fun j -> (b * n) + j))
        in
        windows [ 1; 3; 20; max_int ]
          (Cache_index.every 2
             (Cache_index.make
                ~every:[ (2, matrix (lanes 70)) ]
                ~pos:(matrix [| [| 136; 137; 138 |]; [| 80; 139; -1 |] |])
                ~table:(matrix (lanes 140))
                ())) );
  ]

(* An attention under [visible index] over drawn queries, keys and values in
   [[-1, 1)], four query heads on two key heads of two features. *)
let attended index q k v =
  Attention.attend ~mask:(Cache_index.visible index) q (Nx.Dense { k; v })

(* Compiled, an attention reads [visible] per pair inside its kernel. Both runs
   take, in float64, the softmax of the same scores over the same keys. With n
   <= 70 keys, d = 2 features and |q|, |k| <= sqrt 2, a score differs by at most
   6u, so a weight by about (6 ln 2 + E) u, E = 2 the ulps of {!Nx.exp2}. The
   compiled kernel adds (2E + 1) u for each tile of 16 keys or more where the
   maximum rises, at most 4, and one rescale of the sum a tile
   (lower_attention.mli, "Numerics"). Each output, a convex combination of
   values of magnitude at most 1, then differs by at most about (4n + 60) u <=
   340u, u = 2^-53, about 3.8e-14; [1e-13] leaves room. A key the mask drops or
   adds moves an output by far more. *)
let test_compiled_masks (_, make) =
  Compiled.here ();
  Nx.Rng.with_key (Nx.Rng.key 50) @@ fun () ->
  let indices = make () in
  let compiled =
    Rune.jit
      Nx.Ptree.(
        index_s @-> tensor @-> tensor @-> tensor
        @-> returns (pair (pair tensor tensor) tensor))
      (fun index q k v -> (masks index, attended index q k v))
  in
  let draw shape =
    Nx.sub_s (Nx.mul_s (Nx.Rng.uniform Nx.float64 shape) 2.) 1.
  in
  List.iteri
    (fun n index ->
      let batch = Cache_index.batch index and seq = Cache_index.seq index in
      let columns = Nx.dim 2 (Cache_index.mask index) in
      let q = draw [| batch; 4; seq; 2 |] in
      let k = draw [| batch; 2; columns; 2 |] in
      let v = draw [| batch; 2; columns; 3 |] in
      let mask, visible = masks index in
      let (mask', visible'), attention' = compiled index q k v in
      let msg what = Printf.sprintf "index %d: %s" n what in
      equal ~msg:(msg "mask") (array bool) (flat mask) (flat mask');
      equal ~msg:(msg "visible") (array bool) (Nx.to_array visible)
        (Nx.to_array visible');
      equal ~msg:(msg "attention")
        (array (float 1e-13))
        (flat (attended index q k v))
        (flat attention'))
    indices

(* Indices of three tokens whose keys are not 7 wide. *)
let narrow_keys =
  [
    ( "a table of six columns",
      fun () ->
        index_at ~pos:[| [| 0; 1; 2 |] |] ~table:[| Array.init 6 Fun.id |] );
    ("a whole index", fun () -> Cache_index.whole ~batch:1 ~seq:3 ());
    ( "a selection of two columns",
      fun () ->
        Cache_index.select
          (int64s [| 1; 3; 2 |] [| 0; 1; 0; 1; 0; 1 |])
          (Cache_index.rows ~context:7 [| 3 |]) );
  ]

let test_narrow_keys (_, make) =
  let q = Nx.zeros Nx.float32 [| 1; 2; 3; 2 |] in
  let k = Nx.zeros Nx.float32 [| 1; 2; 7; 2 |] in
  raises_match (Exn.invalid_arg ~sub:"Nx.attention") (fun () ->
      Attention.attend
        ~mask:(Cache_index.visible (make ()))
        q
        (Nx.Dense { k; v = k }))

(* Structure *)

let drafted index =
  Cache_index.draft
    ~slots:(int64s [| 1; 1 |] [| 3 |])
    ~sees:(Nx.full Nx.bool [| 1; 1; 1 |] true)
    index

let made () =
  Cache_index.make ~row:(int64s [| 1 |] [| 0 |])
    ~pos:(int64s [| 1; 1 |] [| 0 |])
    ~table:(int64s [| 2; 2 |] [| 0; 1; 2; 3 |])
    ()

(* The reports walk's .mli lists, per kind of index. *)
let reports =
  [
    ( "whole",
      (fun () -> Cache_index.whole ~batch:2 ~seq:3 ()),
      [
        "tokens: case \"whole\"";
        "tokens.pos: a leaf";
        "every: int 1";
        "window: None";
        "columns: None";
        "draft: None";
        "resolved: None";
      ] );
    ( "packed",
      (fun () -> Cache_index.packed ~seq:3 [| [| 1; 2 |] |]),
      [
        "tokens: case \"packed\"";
        "tokens.pos: a leaf";
        "every: int 1";
        "window: None";
        "columns: None";
        "draft: None";
        "resolved: None";
      ] );
    ( "rows in blocks of 4 under a window",
      (fun () ->
        Cache_index.window 2
          (Cache_index.every 4
             (Cache_index.rows ~every:[ 4; 2 ] ~context:8 [| 3; 2 |]))),
      [
        "tokens: case \"tabled\"";
        "tokens.row: None";
        "tokens.pos: a leaf";
        "tokens.table: a leaf";
        "tokens.blocks: length 2";
        "tokens.blocks.0: int 4";
        "tokens.blocks.0: a leaf";
        "tokens.blocks.1: int 2";
        "tokens.blocks.1: a leaf";
        "every: int 4";
        "window: Some";
        "window: a leaf";
        "columns: None";
        "draft: None";
        "resolved: None";
      ] );
    ( "a row and a selection",
      (fun () -> Cache_index.select (int64s [| 1; 1; 1 |] [| 0 |]) (made ())),
      [
        "tokens: case \"tabled\"";
        "tokens.row: Some";
        "tokens.row: a leaf";
        "tokens.pos: a leaf";
        "tokens.table: a leaf";
        "tokens.blocks: length 0";
        "every: int 1";
        "window: None";
        "columns: Some";
        "columns: a leaf";
        "draft: None";
        "resolved: None";
      ] );
    ( "a draft",
      (fun () -> drafted (made ())),
      [
        "tokens: case \"tabled\"";
        "tokens.row: Some";
        "tokens.row: a leaf";
        "tokens.pos: a leaf";
        "tokens.table: a leaf";
        "tokens.blocks: length 0";
        "every: int 1";
        "window: None";
        "columns: None";
        "draft: Some";
        "draft.slots: a leaf";
        "draft.sees: a leaf";
        "resolved: None";
      ] );
    ( "resolved under a window",
      (fun () ->
        Cache_index.resolve
          (Cache_index.window 2 (Cache_index.rows ~context:4 [| 2 |]))),
      [
        "tokens: case \"tabled\"";
        "tokens.row: None";
        "tokens.pos: a leaf";
        "tokens.table: a leaf";
        "tokens.blocks: length 0";
        "every: int 1";
        "window: Some";
        "window: a leaf";
        "columns: None";
        "draft: None";
        "resolved: Some";
        "resolved.stores: a leaf";
        "resolved.pages: Some";
        "resolved.pages: a leaf";
      ] );
    ( "resolved with a selection",
      (fun () ->
        Cache_index.resolve
          (Cache_index.select (int64s [| 1; 1; 1 |] [| 0 |]) (made ()))),
      [
        "tokens: case \"tabled\"";
        "tokens.row: Some";
        "tokens.row: a leaf";
        "tokens.pos: a leaf";
        "tokens.table: a leaf";
        "tokens.blocks: length 0";
        "every: int 1";
        "window: None";
        "columns: Some";
        "columns: a leaf";
        "draft: None";
        "resolved: Some";
        "resolved.stores: a leaf";
        "resolved.pages: None";
      ] );
  ]

let test_reports (_, make, expected) =
  equal (list string) expected (visits (make ()))

(* Pairs of indices that differ in what a compiled program depends on. *)
let differing =
  let rows () = Cache_index.rows ~every:[ 4 ] ~context:8 [| 3; 2 |] in
  let one () = Cache_index.rows ~context:4 [| 1 |] in
  [
    ("a window", fun () -> (rows (), Cache_index.window 2 (rows ())));
    ("blocks read", fun () -> (rows (), Cache_index.every 4 (rows ())));
    ( "a block size",
      fun () -> (rows (), Cache_index.rows ~every:[ 2 ] ~context:8 [| 3; 2 |])
    );
    ( "whole or tabled",
      fun () -> (Cache_index.whole ~batch:2 ~seq:3 (), rows ()) );
    ( "whole or packed",
      fun () ->
        ( Cache_index.whole ~batch:1 ~seq:3 (),
          Cache_index.packed ~seq:3 [| [| 3 |] |] ) );
    ("a draft", fun () -> (one (), drafted (one ())));
    ("resolved", fun () -> (one (), Cache_index.resolve (one ())));
  ]

let test_differing (_, make) =
  let a, b = make () in
  not_equal (list string) (visits a) (visits b)

let test_window_size_is_a_value () =
  let rows = Cache_index.rows ~context:8 [| 3; 2 |] in
  equal (list string)
    (visits (Cache_index.window 2 rows))
    (visits (Cache_index.window 3 rows))

let round_trips =
  [
    ( "blocks and a window",
      fun () ->
        Cache_index.window 2
          (Cache_index.rows ~every:[ 4 ] ~context:8 [| 3; 2 |]) );
    ( "a row and a selection",
      fun () ->
        Cache_index.select
          (int64s [| 1; 1; 1 |] [| 1 |])
          (Cache_index.make ~row:(int64s [| 1 |] [| 1 |])
             ~pos:(int64s [| 1; 1 |] [| 1 |])
             ~table:(int64s [| 2; 2 |] [| 0; 1; 2; 3 |])
             ()) );
    ("a draft", fun () -> drafted (Cache_index.rows ~context:4 [| 1 |]));
    ("a packing", fun () -> Cache_index.packed ~seq:4 [| [| 2; 1 |] |]);
    ( "resolved",
      fun () ->
        Cache_index.resolve
          (Cache_index.window 2 (Cache_index.rows ~context:4 [| 3 |])) );
  ]

let test_round_trip (_, make) =
  let index = make () in
  let tensors, _ = Nx.Ptree.flatten index_s index in
  let copy =
    Nx.Ptree.rebuild index_s ~like:index
      (List.map (fun (Nx.P x) -> Nx.P (Nx.copy x)) tensors)
  in
  same ~msg:"the copy" index copy

(* place *)

let two =
  lazy
    (Nx.devices
       (List.map
          (fun n -> Result.get_ok (Rig.software n))
          [ "kaun-index-0"; "kaun-index-1" ]))

(* An index placed cut along axis 0 of two devices: its tensors lie whole on
   each, and a window made after lies beside its positions. *)
let test_place () =
  let module Two = (val Lazy.force two : Nx.Devices) in
  let placement =
    Testable.make ~pp:Nx.Placement.pp ~equal:(fun a b -> Nx.Placement.equal a b)
  in
  let cut = Two.split ~axis:0 in
  let rows = Cache_index.rows ~context:4 [| 2; 2 |] in
  let whole_on_each ~msg index =
    List.iter
      (fun (Nx.P x) -> equal ~msg placement Two.on (Nx.placement x))
      (fst (Nx.Ptree.flatten (Nx.Ptree.v Cache_index.walk) index))
  in
  whole_on_each ~msg:"placed rows" (Cache_index.place cut rows);
  whole_on_each ~msg:"a placed window"
    (Cache_index.place cut (Cache_index.window 2 rows));
  whole_on_each ~msg:"the window of a placed index"
    (Cache_index.window 2 (Cache_index.place cut rows))

let () =
  let id = fst and id3 (i, _, _) = i in
  exit
    (run "kaun cache index"
       [
         group "laws"
           [
             prop "a renaming of the slots reads the same columns" renaming_gen
               law_renaming;
             prop ~count:150 ~examples:store_examples
               "what addresses nothing reads as zero, eagerly and compiled"
               drawn law_reads;
             prop ~count:200 ~examples:store_examples
               "a token stores at its own column and nowhere else, eagerly and \
                compiled"
               drawn law_stores;
             cases
               ~id:(fun (i, _, _, _) -> i)
               "a pool of one slot keeps what its token stored" one_slot_rows
               test_one_slot;
             prop ~count:200 "pages are -1 at the columns never read" drawn
               law_pages;
             test
               "one compiled program serves every position, allocation and \
                window size"
               test_one_program;
           ];
         group "make"
           [
             test "lanes of one sequence see each other's tokens"
               test_lanes_of_one_sequence;
             cases ~id "raises on a wrong shape, an empty axis or a bad m"
               make_refusals (refuses "make");
           ];
         group "rows"
           [
             test "sequence b stores position j at slot b * context + j"
               test_rows_runs;
             test "sequence b's blocks are slots b * c to b * c + c - 1"
               test_rows_blocks;
             test "the last block closes only past the context"
               test_rows_last_block;
             cases ~id "raises on a bad lane, context or m" rows_refusals
               (refuses "rows");
           ];
         group "whole"
           [
             test "nothing is read and nothing is kept" test_whole_keeps_nothing;
             cases ~id "raises on a batch, seq or length that does not fit"
               whole_refusals (refuses "whole");
           ];
         group "packed"
           [
             test "each sequence's positions start at 0" test_packed_positions;
             test "one sequence per lane masks, extends and rotates as whole"
               test_packed_singletons;
             test "one compiled program serves every packing"
               test_packed_program;
             cases ~id "raises on a bad lane, seq or length" packed_refusals
               (refuses "packed");
           ];
         group "window"
           [
             test "replaces the window the index had" test_window_replaces;
             cases ~id "a window wider than every position hides nothing"
               widest_rows test_widest;
             test "Rune.vmap refuses an index with a window" test_window_vmap;
             cases ~id "raises on a size that is not positive" window_refusals
               (refuses "window");
           ];
         group "select"
           [
             prop ~count:150
               "a chosen column reads and masks as without a selection"
               selection_gen law_selection;
             prop "what is stored does not change" selection_gen
               law_selection_stores;
             test "replaces the selection the index had" test_select_replaces;
             cases ~id "on a whole index a column is a token of the lane"
               whole_selections test_whole_selection;
             test "a selection may be a broadcast view" test_select_broadcast;
             test "compiled, a selection reads and stores as eager"
               test_select_compiled;
             cases ~id "raises on a wrong shape or a draft" select_refusals
               (refuses "select");
           ];
         group "draft"
           [
             test "seen is the table before the least position, then values"
               test_draft_seen;
             test "keeps the window and replaces the draft the index had"
               test_draft_replaces;
             cases ~id "raises on a wrong shape or an index it cannot draft"
               draft_refusals (refuses "draft");
           ];
         group "every"
           [
             test "on a whole index a block is read from its closing token"
               test_every_whole;
             cases ~id "a block's entry does not depend on how it was fed"
               feedings test_stream;
             test "a whole index gives the stream's outputs" test_stream_whole;
             test "a compiled stream stores and reads as eager"
               test_stream_compiled;
             cases ~id "is the index when it reads blocks of m already"
               every_identities test_every_identity;
             test "keeps the window" test_every_keeps_the_window;
             cases ~id "raises on m or an index it cannot read in blocks"
               every_refusals (refuses "every");
           ];
         group "advance"
           [
             prop "the next token is one past its lane's greatest position"
               drawn law_advance;
             test "where two lanes name one sequence it is per lane"
               test_advance_per_lane;
             test "a lane of padding advances to position 0"
               test_advance_padding;
             cases ~id:id3 "keeps the window and the blocks" advance_keeps
               test_advance_keeps;
             cases ~id:id3 "drops the selection and the draft" advance_drops
               test_advance_drops;
             cases ~id "raises on a whole index" advance_refusals
               (refuses "advance");
           ];
         group "batch"
           [
             cases ~id:id3 "is the number of lanes" batches
               (counted Cache_index.batch);
           ];
         group "seq"
           [
             cases ~id:id3 "is the number of tokens per lane" seqs
               (counted Cache_index.seq);
           ];
         group "context"
           [
             cases ~id:id3 "is the number of columns" contexts
               (counted Cache_index.context);
           ];
         group "positions"
           [
             cases ~id:id3 "are clamped to the positions' table" positions_rows
               test_positions;
           ];
         group "extend"
           [
             cases ~id "raises on values of another shape" wrong_values
               (refuses_values "extend" Cache_index.extend);
           ];
         group "resolve"
           [
             prop "a resolved index addresses as the index" selection_gen
               law_resolved;
             prop "compiled, a resolved index stores and reads as the laws say"
               drawn law_resolved_compiled;
             cases ~id "every kind of index resolves" resolved_kinds
               test_resolved_kind;
             cases ~id:id3
               "an index made from it addresses as one made from the index"
               made_from_resolved test_made_from_resolved;
             cases ~id:id3 "an index made from it is not resolved"
               made_from_resolved test_unresolved;
             cases ~id "a whole or packed index is unchanged" resolved_unchanged
               test_resolved_unchanged;
             test "a layer computes none of a resolved index's addressing"
               test_resolved_launches;
           ];
         group "store"
           [
             prop "is extend's pool'" drawn law_store_is_extend;
             cases ~id "raises on values of another shape" wrong_values
               (refuses_values "store" Cache_index.store);
           ];
         group "pages"
           [
             cases ~id
               "is None on a whole or packed index and under a selection"
               no_pages test_no_pages;
           ];
         group "mask"
           [
             prop ~count:400 ~examples:[ all_seeing ]
               "is which columns each token sees" shown_gen law_mask;
           ];
         group "visible"
           [
             prop ~count:400 ~examples:[ all_seeing ]
               "holds where mask holds, and at a hole" shown_gen law_visible;
             cases ~id
               "compiled, mask, visible and attention under it agree with eager"
               (compiled_kinds @ wide_kinds)
               test_compiled_masks;
             cases ~id "an attention raises unless its keys are context wide"
               narrow_keys test_narrow_keys;
           ];
         group "walk"
           [
             cases ~id:id3 "reports the index's structure in order" reports
               test_reports;
             cases ~id "indices a program tells apart report differently"
               differing test_differing;
             test "a window's size is a value of the index"
               test_window_size_is_a_value;
             cases ~id "rebuilds an index from its tensors" round_trips
               test_round_trip;
           ];
         group "place"
           [ test "an index lies whole on each device of at" test_place ];
       ])
