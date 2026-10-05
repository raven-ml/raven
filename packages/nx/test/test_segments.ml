(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Maps over segments, against the per-position map: each position's row goes
   through the function its id selects, and an id out of range gives zeros. The
   function records what it receives, which the contract states: owners in
   range, blocks of rows of in-range positions with the block's id, and a number
   of blocks that reads no id. *)

open Windtrap
open Nx_test

let int64s v = Nx.create Nx.int64 [| Array.length v |] v
let numel s = Array.fold_left ( * ) 1 s

(* Results compare bits, so -0 and +0 differ and NaN equals its own bits. *)
let bits =
  Testable.contramap
    (fun t -> Nx.to_array (Nx.bitcast Nx.int64 t))
    (array int64)

(* What [f] received in one call: its owners and its rows, by block. *)
type call = { owners : int array; rows : float array array array }

(* [row_fn owners rows] maps each row [v] of owner [o] to [[| sum v * (o + 1); o
   |]], a function of the row and its owner alone, exact on small integers. *)
let row_fn owners rows =
  let g = Nx.dim 0 rows and c = Nx.dim 1 rows in
  let total =
    if Nx.ndim rows = 2 then rows
    else Nx.sum ~axes:(List.init (Nx.ndim rows - 2) (fun a -> a + 2)) rows
  in
  let o = Nx.reshape [| g; 1 |] (Nx.cast Nx.float64 owners) in
  Nx.stack ~axis:2
    [ Nx.mul total (Nx.add_s o 1.); Nx.broadcast_to [| g; c |] o ]

(* [recorded ()] is [row_fn], which records each call, and the calls. *)
let recorded () =
  let calls = ref [] in
  let f owners rows =
    let g = Nx.dim 0 rows and c = Nx.dim 1 rows in
    let w = numel (Array.sub (Nx.shape rows) 2 (Nx.ndim rows - 2)) in
    let flat = Nx.to_array rows in
    let call =
      {
        owners = Array.map Int64.to_int (Nx.to_array owners);
        rows =
          Array.init g (fun b ->
              Array.init c (fun i -> Array.sub flat (((b * c) + i) * w) w));
      }
    in
    calls := call :: !calls;
    row_fn owners rows
  in
  (f, calls)

(* [expected ~segments ids x] is the per-position map of [row_fn]. *)
let expected ~segments ids x =
  let s = Nx.shape ids in
  let p = numel s in
  let ns = Array.length s in
  let row = Array.sub (Nx.shape x) ns (Nx.ndim x - ns) in
  let rows =
    Nx.to_array
      (Nx.reshape [| p; numel row |] (Nx.broadcast_to (Array.append s row) x))
  in
  let ids = Nx.to_array (Nx.reshape [| p |] ids) in
  let out =
    Array.concat
      (List.init p (fun i ->
           let id = ids.(i) in
           if id >= 0L && id < Int64.of_int segments then
             let o = Int64.to_float id in
             let total =
               Array.fold_left ( +. ) 0.
                 (Array.sub rows (i * numel row) (numel row))
             in
             [| total *. (o +. 1.); o |]
           else [| 0.; 0. |]))
  in
  Nx.create Nx.float64 (Array.append s [| 2 |]) out

(* [in_range ~segments ids x] is each in-range position's id and row. *)
let in_range ~segments ids x =
  let s = Nx.shape ids in
  let p = numel s in
  let ns = Array.length s in
  let row = Array.sub (Nx.shape x) ns (Nx.ndim x - ns) in
  let w = numel row in
  let rows =
    Nx.to_array (Nx.reshape [| p; w |] (Nx.broadcast_to (Array.append s row) x))
  in
  let ids = Nx.to_array (Nx.reshape [| p |] ids) in
  List.filter_map
    (fun i ->
      let id = ids.(i) in
      if id >= 0L && id < Int64.of_int segments then
        Some (Int64.to_int id, Array.sub rows (i * w) w)
      else None)
    (List.init p Fun.id)

let same_bits a b =
  Array.length a = Array.length b
  && Array.for_all2
       (fun u v -> Int64.bits_of_float u = Int64.bits_of_float v)
       a b

(* [received ~segments ids x call] checks what [f] received against the
   contract, for one group of [p] positions. *)
let received ~segments ids x call =
  let p = Nx.numel ids in
  let g = Array.length call.rows in
  let c = if g = 0 then 1 else Array.length call.rows.(0) in
  let named = in_range ~segments ids x in
  at_most ~msg:"rows per block" ~than:16 int c;
  if segments = 0 || p = 0 then equal ~msg:"blocks" int 0 g
  else
    at_most ~msg:"blocks"
      ~than:((p + (min segments p * (c - 1)) + c - 1) / c)
      int g;
  Array.iteri
    (fun b o ->
      if o < 0 || o >= segments then failf "block %d: owner %d out of range" b o;
      Array.iteri
        (fun i v ->
          let held = named = [] && Array.for_all (fun e -> e = 0.) v in
          let of_owner =
            List.exists (fun (id, r) -> id = o && same_bits r v) named
          in
          if not (held || of_owner) then
            failf "block %d (owner %d), row %d is no in-range row of its owner"
              b o i)
        call.rows.(b))
    call.owners;
  List.iter
    (fun (id, r) ->
      let found = ref false in
      Array.iteri
        (fun b o ->
          if o = id && Array.exists (same_bits r) call.rows.(b) then
            found := true)
        call.owners;
      if not !found then failf "a row of id %d reaches no block of its id" id)
    named;
  (g, c)

(* Draws *)

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map Int64.to_string a)))

type draw = {
  segments : int;
  shape : int array;  (** positions *)
  ids : int64 array;
  broadcast : bool array;  (** a leading axis of [x] of size 1 *)
  row : int array;
  poison : float option;  (** the rows no in-range position reads *)
}

let pp_draw ppf d =
  Format.fprintf ppf "segments %d, ids %a of shape %a, broadcast %s, row %a%s"
    d.segments pp_ints d.ids pp_shape d.shape
    (String.concat ""
       (Array.to_list (Array.map (fun b -> if b then "1" else ".") d.broadcast)))
    pp_shape d.row
    (match d.poison with
    | Some v -> Printf.sprintf ", %g out of range" v
    | None -> "")

let extremes =
  [ Int64.min_int; Int64.max_int; 0x1_0000_0000L; 0x1_0000_0001L; -2L ]

let draw_with ?(segments = Gen.int_range 0 4) shapes =
  let open Gen in
  let* segments = segments in
  let* shape = shapes in
  let id =
    frequency
      [ (8, map Int64.of_int (int_range (-1) segments)); (1, of_list extremes) ]
  in
  let* ids = array ~size:(constant (numel shape)) id in
  let* broadcast = array ~size:(constant (Array.length shape)) bool in
  let* row = one_of [ constant [||]; map (fun w -> [| w |]) (int_range 0 2) ] in
  let+ poison =
    option (of_list [ Float.nan; Float.infinity; Float.neg_infinity ])
  in
  { segments; shape; ids; broadcast; row; poison }

let draw =
  let open Gen in
  with_pp pp_draw
    (draw_with
       (one_of
          [
            constant [||];
            map (fun n -> [| n |]) (int_range 0 40);
            array ~size:(constant 2) (int_range 0 6);
            array ~size:(constant 3) (int_range 0 3);
          ]))

(* [tensors d] is [d]'s ids and rows: small integers, distinct per row of x, and
   NaN where [d] poisons a row that no in-range position reads. *)
let tensors d =
  let ids = Nx.reshape d.shape (int64s d.ids) in
  let lead = Array.mapi (fun a n -> if d.broadcast.(a) then 1 else n) d.shape in
  let w = numel d.row in
  (* The row of x each position reads, and the rows an in-range one reads. *)
  let row_of p =
    let i = ref p and u = ref 0 and stride = ref 1 in
    for a = Array.length d.shape - 1 downto 0 do
      let n = d.shape.(a) in
      if lead.(a) > 1 then u := !u + (!i mod n * !stride);
      stride := !stride * lead.(a);
      i := !i / n
    done;
    !u
  in
  let read = Array.make (numel lead) false in
  Array.iteri
    (fun p id ->
      if id >= 0L && id < Int64.of_int d.segments then read.(row_of p) <- true)
    d.ids;
  let x =
    Nx.init Nx.float64 (Array.append lead d.row) (fun i ->
        let u = ref 0 in
        Array.iteri
          (fun a n -> if a < Array.length lead then u := (!u * lead.(a)) + n)
          i;
        match d.poison with
        | Some v when not read.(!u) -> v
        | _ -> float_of_int ((!u * (w + 1)) + 1))
  in
  (ids, x)

let groups ~segments ids = numel (Nx.shape ids) > 2 * segments

let law d =
  let ids, x = tensors d in
  let f, calls = recorded () in
  let y = Nx.map_segments ~segments:d.segments ids f x in
  equal ~msg:"result" bits (expected ~segments:d.segments ids x) y;
  (match !calls with
  | [ call ] ->
      let g, c = received ~segments:d.segments ids x call in
      (* The blocks read no id: zeros of the same shape give the same. *)
      let f', calls' = recorded () in
      ignore (Nx.map_segments ~segments:d.segments (Nx.zeros_like ids) f' x);
      let call' = List.hd !calls' in
      equal ~msg:"blocks of other ids" int g (Array.length call'.rows);
      if g > 0 then
        equal ~msg:"rows of other ids" int c (Array.length call'.rows.(0));
      cover "rows in blocks of 2 or more" (c >= 2);
      cover "a block per position" (c = 1 && g > 0);
      (* Positions that read one row of x, along a broadcast axis, beside
         positions that read rows of their own. *)
      let along f =
        List.exists Fun.id
          (List.mapi
             (fun a n -> n > 1 && f d.broadcast.(a))
             (Array.to_list d.shape))
      in
      cover "a block per position, one row for all"
        (c = 1 && along Fun.id && not (along not));
      cover "a block per position, rows shared and own"
        (c = 1 && along Fun.id && along not);
      cover "a block per position, rows shared along two axes"
        (c = 1
        && List.length
             (List.filter Fun.id
                (List.mapi
                   (fun a n -> n > 1 && d.broadcast.(a))
                   (Array.to_list d.shape)))
           >= 2)
  | calls -> failf "f called %d times" (List.length calls));
  let p = numel d.shape in
  cover "no position" (p = 0);
  cover "no segment" (d.segments = 0 && p > 0);
  cover "a scalar id" (d.shape = [||]);
  cover "a broadcast row" (Array.exists Fun.id d.broadcast && p > 1);
  let poisoned f =
    Option.fold ~none:false ~some:f d.poison
    && Array.exists (fun i -> i < 0L) d.ids
  in
  cover "NaN out of range" (poisoned Float.is_nan);
  cover "an infinity out of range"
    (poisoned (fun v -> Float.abs v = Float.infinity));
  cover "an extreme id" (Array.exists (fun i -> List.mem i extremes) d.ids);
  cover "a row of width zero" (numel d.row = 0 && p > 0);
  cover "grouped, an empty last segment"
    (groups ~segments:d.segments ids
    && d.segments > 0
    && not (Array.mem (Int64.of_int (d.segments - 1)) d.ids))

(* Chosen draws: a last segment no position names, beside a NaN row at -1, on
   both sides of a block per position; then a block per position over one row
   shared by every position, all at -1, over a row per token, one token's ids
   all at -1, and over rows shared along two axes, adjacent or not. *)
let examples =
  let draw ?(broadcast = [| false |]) ?shape segments ids =
    {
      segments;
      shape = Option.value shape ~default:[| Array.length ids |];
      ids = Array.map Int64.of_int ids;
      broadcast;
      row = [| 2 |];
      poison = Some Float.nan;
    }
  in
  [
    draw 3 [| 0; 1; -1; 1; 0; -1; 1; 0; 0; 1 |];
    draw 3 [| 1; -1 |];
    draw 2 [| -1; -1; -1; -1; -1; -1; -1 |];
    draw 2 [| -1; -1 |];
    draw ~broadcast:[| true |] 4 [| -1; -1; -1 |];
    draw ~broadcast:[| false; true |] ~shape:[| 3; 3 |] 5
      [| 1; -1; -1; 3; -1; -1; -1; -1; -1 |];
    draw ~broadcast:[| false; true; true |] ~shape:[| 3; 2; 2 |] 6
      [| -1; 2; -1; 5; -1; -1; -1; -1; 0; -1; -1; 1 |];
    draw ~broadcast:[| true; false; true |] ~shape:[| 2; 3; 2 |] 6
      [| -1; -1; 4; -1; -1; -1; -1; 3; -1; -1; -1; -1 |];
  ]

let maps =
  group "map_segments"
    [
      prop ~examples "is the per-position map, and f receives what it promises"
        draw law;
      test "an integer group with no in-range position gives f rows of zeros"
        (fun () ->
          List.iter
            (fun n ->
              let ids = Nx.full Nx.int64 [| n |] (-1L) in
              let x = Nx.add_s (Nx.arange Nx.int32 0 n 1) 5l in
              let seen = ref [||] in
              let y =
                Nx.map_segments ~segments:2 ids
                  (fun _ rows ->
                    seen := Nx.to_array rows;
                    rows)
                  x
              in
              equal ~msg:"rows" (array int32)
                (Array.make (Array.length !seen) 0l)
                !seen;
              equal ~msg:"result" (array int32) (Array.make n 0l)
                (Nx.to_array y))
            [ 3; 9 ]);
      test "f's result may have other row axes than x" (fun () ->
          let ids = int64s [| 1L; 0L; 1L |]
          and x = Nx.ones Nx.float32 [| 3; 2 |] in
          let y =
            Nx.map_segments ~segments:2 ids
              (fun _ rows ->
                Nx.broadcast_to
                  [| Nx.dim 0 rows; Nx.dim 1 rows; 4; 5 |]
                  (Nx.reshape
                     [| Nx.dim 0 rows; Nx.dim 1 rows; 1; 1 |]
                     (Nx.sum ~axes:[ 2 ] rows)))
              x
          in
          equal (array int) [| 3; 4; 5 |] (Nx.shape y));
      test "map_segments refuses negative segments, misshapen x and f's result"
        (fun () ->
          let ids = Nx.zeros Nx.int64 [| 2; 3 |] in
          let x = Nx.zeros Nx.float32 [| 2; 3; 4 |] in
          let id _ rows = rows in
          raises_invalid_arg (fun () -> Nx.map_segments ~segments:(-1) ids id x);
          raises_invalid_arg (fun () ->
              Nx.map_segments ~segments:1 ids id (Nx.zeros Nx.float32 [| 2 |]));
          raises_invalid_arg (fun () ->
              Nx.map_segments ~segments:1 ids id
                (Nx.zeros Nx.float32 [| 2; 2; 4 |]));
          raises_invalid_arg (fun () ->
              Nx.map_segments ~segments:1 ids
                (fun _ rows -> Nx.reshape [| -1 |] rows)
                x);
          raises_invalid_arg (fun () ->
              Nx.map_segments ~segments:1 ids
                (fun _ rows -> Nx.concatenate ~axis:0 [ rows; rows ])
                x));
    ]

(* Dtypes *)

type fdt = F : (float, 'b) Nx.dtype -> fdt

let float_dtypes =
  [
    F Nx.float16;
    F Nx.bfloat16;
    F Nx.float32;
    F Nx.float64;
    F Nx.float8_e4m3;
    F Nx.float8_e5m2;
  ]

(* [scaled owners rows] multiplies each row by its owner plus one, element by
   element, at the rows' dtype: a row's result reads that row and its owner
   alone. *)
let scaled owners rows =
  let shape = Array.mapi (fun a n -> if a = 0 then n else 1) (Nx.shape rows) in
  Nx.mul rows (Nx.reshape shape (Nx.add_s (Nx.cast (Nx.dtype rows) owners) 1.))

(* [per_position ~segments f ids x] is [f] on each in-range position's row
   alone, a block of one row, and zeros elsewhere: no row out of range reaches
   [f]. *)
let per_position ~segments f ids x =
  let s = Nx.shape ids in
  let p = numel s and ns = Array.length s in
  let row = Array.sub (Nx.shape x) ns (Nx.ndim x - ns) in
  let ids = Nx.reshape [| p |] ids in
  let rows =
    Nx.reshape (Array.append [| p |] row)
      (Nx.broadcast_to (Array.append s row) x)
  in
  let one i =
    let id = Nx.item [ i ] ids in
    if id >= 0L && id < Int64.of_int segments then
      Nx.reshape row
        (f
           (Nx.reshape [| 1 |] (Nx.slice [ I i ] ids))
           (Nx.reshape (Array.append [| 1; 1 |] row) (Nx.slice [ I i ] rows)))
    else Nx.zeros (Nx.dtype x) row
  in
  Nx.reshape (Array.append s row) (Nx.stack (List.init p one))

(* Routes on either side of a block per position, each row that no in-range
   position reads NaN: rows of their own, a row per token of 3 positions, blocks
   of 2 among extreme ids, and blocks of 4 with no in-range position. *)
let dtype_routes =
  let nan = Float.nan in
  let rows shape v = Nx.create Nx.float64 shape v in
  [
    ( "a block per position, rows of their own",
      4,
      int64s [| 3L; -1L; 0L |],
      rows [| 3; 2 |] [| 1.; 2.; nan; nan; 5.; 6. |] );
    ( "a block per position, a row per token",
      5,
      Nx.reshape [| 3; 3 |]
        (int64s [| 1L; -1L; -1L; 3L; -1L; 4L; -1L; -1L; -1L |]),
      rows [| 3; 1; 2 |] [| 1.; 2.; 3.; 4.; nan; nan |] );
    ( "blocks of 2, extreme ids",
      2,
      int64s
        [|
          0L; 1L; -1L; 1L; 0L; Int64.max_int; 1L; 0L; 0L; Int64.min_int; 1L; 2L;
        |],
      Nx.init Nx.float64 [| 12; 2 |] (fun i ->
          if List.mem i.(0) [ 2; 5; 9; 11 ] then nan
          else float_of_int ((2 * i.(0)) + i.(1) + 1)) );
    ( "blocks of 4, no position in range",
      1,
      Nx.full Nx.int64 [| 8 |] (-1L),
      Nx.full Nx.float64 [| 8; 3 |] nan );
  ]

let dtypes =
  cases
    ~name:(fun (F dt, (route, _, _, _)) ->
      Printf.sprintf "%s, %s" (Nx_dtype.to_string dt) route)
    "map_segments is the per-position map at every float dtype"
    (List.concat_map
       (fun dt -> List.map (fun r -> (dt, r)) dtype_routes)
       float_dtypes)
    (fun (F dt, (_, segments, ids, x)) ->
      let x = Nx.cast dt x in
      equal bits
        (Nx.cast Nx.float64 (per_position ~segments scaled ids x))
        (Nx.cast Nx.float64 (Nx.map_segments ~segments ids scaled x)))

(* Segment counts up to the largest int: every id below one is in range, and the
   rule that sizes blocks reads only shapes. *)
let large_segments =
  cases ~name:string_of_int
    "a segment count up to the largest int gives the per-position map"
    [ 1 lsl 40; max_int / 30; (max_int / 30) + 1; 1 lsl 58; max_int ]
    (fun segments ->
      let ids =
        int64s [| 0L; 5L; -1L; Int64.max_int; Int64.shift_left 1L 40 |]
      in
      let x = Nx.create Nx.float64 [| 5; 1 |] [| 1.; 2.; 3.; 4.; 5. |] in
      equal bits (expected ~segments ids x)
        (Nx.map_segments ~segments ids row_fn x))

(* Devices *)

let host t = Nx.place Nx.Placement.host t
let pair = [ Devices.d1; Devices.d2 ]

(* [routes n] is [n] positions over 3 segments, the second half's ids all -1,
   and their rows, with a NaN row at -1: with 12 positions each device's group
   holds a block per position, with 24 blocks of 2. *)
let routes n =
  let ids =
    int64s
      (Array.init n (fun i ->
           if i >= n / 2 then -1L else [| 2L; 0L; -1L; 2L; 1L; 0L |].(i mod 6)))
  in
  let x =
    Nx.set [ I 2 ]
      (Nx.full Nx.float64 [| 4 |] Float.nan)
      (Nx.reshape [| n; 4 |]
         (Nx.arange_f Nx.float64 1. (float_of_int ((4 * n) + 1)) 1.))
  in
  (ids, x)

(* Draws whose positions split over two devices, a first axis of even length,
   and which of ids and x split: x splits only along an axis of its own. A map
   with no position or no segment is [empty_split]'s. *)
let split_draw =
  let open Gen in
  let shapes =
    one_of
      [
        map (fun n -> [| 2 * n |]) (int_range 1 20);
        (let+ n = int_range 1 3 and+ m = int_range 1 6 in
         [| 2 * n; m |]);
      ]
  in
  let layout =
    of_list ~pp:Format.pp_print_string [ "both split"; "ids split"; "x split" ]
  in
  let+ d = draw_with ~segments:(int_range 1 4) shapes and+ l = layout in
  (d, l)

let split_draw =
  Gen.with_pp
    (fun ppf (d, l) -> Format.fprintf ppf "%a, %s" pp_draw d l)
    split_draw

(* Each device's group is its window of positions, and its blocks the matching
   half of f's: each half holds what [received] promises its group. *)
let split_law (d, layout) =
  let ids, x = tensors d in
  let split t = Nx.place (Nx.Placement.sharded ~axis:0 pair) t in
  let copies t = Nx.place (Nx.Placement.replicated pair) t in
  let own = not d.broadcast.(0) in
  let ids', x' =
    match layout with
    | "ids split" -> (split ids, copies x)
    | "x split" when own -> (copies ids, split x)
    | _ -> (split ids, if own then split x else copies x)
  in
  let f, calls = recorded () in
  let y = Nx.map_segments ~segments:d.segments ids' f x' in
  equal ~msg:"result" bits (expected ~segments:d.segments ids x) (host y);
  match !calls with
  | [ call ] ->
      let g = Array.length call.rows in
      equal ~msg:"blocks, even" int 0 (g mod 2);
      let half = d.shape.(0) / 2 in
      let named = ref [] in
      for i = 0 to 1 do
        let window t = Nx.slice [ R (i * half, (i + 1) * half) ] t in
        let ids = window ids in
        let sub a = Array.sub a (i * g / 2) (g / 2) in
        let call = { owners = sub call.owners; rows = sub call.rows } in
        ignore
          (received ~segments:d.segments ids (if own then window x else x) call);
        named :=
          List.exists
            (fun v -> v >= 0L && v < Int64.of_int d.segments)
            (Array.to_list (Nx.to_array ids))
          :: !named
      done;
      cover "a group with no in-range position beside one with"
        (List.mem true !named && List.mem false !named);
      cover "a block per position" (g > 0 && Nx.numel ids / 2 <= 2 * d.segments);
      cover "rows in blocks" (Nx.numel ids / 2 > 2 * d.segments)
  | calls -> failf "f called %d times" (List.length calls)

(* No segment, or no position, split over two devices. *)
let empty_split =
  cases
    ~name:(fun (segments, shape) ->
      Format.asprintf "%d segments, ids of shape %a" segments pp_shape shape)
    "an empty map split over two devices is zeros, and f sees no block"
    [ (0, [| 2 |]); (0, [| 4; 3 |]); (2, [| 2; 0 |]); (2, [| 0 |]) ]
    (fun (segments, shape) ->
      let split t = Nx.place (Nx.Placement.sharded ~axis:0 pair) t in
      let ids = Nx.zeros Nx.int64 shape in
      let x = Nx.full Nx.float64 (Array.append shape [| 2 |]) Float.nan in
      let f, calls = recorded () in
      let y = Nx.map_segments ~segments (split ids) f (split x) in
      equal ~msg:"result" bits (expected ~segments ids x) (host y);
      equal ~msg:"blocks" (list int) [ 0 ]
        (List.map (fun c -> Array.length c.rows) !calls))

let placements =
  let map ids x = Nx.map_segments ~segments:3 ids row_fn x in
  let split ?(axis = 0) t = Nx.place (Nx.Placement.sharded ~axis pair) t in
  let copies t = Nx.place (Nx.Placement.replicated pair) t in
  let layouts =
    [
      ("both split", fun ids x -> (split ids, split x));
      ("ids split", fun ids x -> (split ids, copies x));
      ("x split", fun ids x -> (copies ids, split x));
    ]
  in
  group "map_segments over devices"
    [
      prop
        "split over two devices, a map is one device's, and each group's \
         blocks hold what f is promised"
        split_draw split_law;
      empty_split;
      cases
        ~name:(fun (n, (l, _)) -> Printf.sprintf "%d positions, %s" n l)
        "positions split over two devices give one device's result, a device \
         with every id -1 included"
        (List.concat_map
           (fun n -> List.map (fun l -> (n, l)) layouts)
           [ 12; 24 ])
        (fun (n, (_, place)) ->
          let ids, x = routes n in
          let ids', x' = place ids x in
          equal bits (expected ~segments:3 ids x) (host (map ids' x')));
      test "map_segments refuses a split along a row axis or unequal splits"
        (fun () ->
          let ids, x = routes 12 in
          let four = [ Devices.d1; Devices.d2; Devices.d3; Devices.d4 ] in
          raises_invalid_arg (fun () -> map ids (split ~axis:1 x));
          raises_invalid_arg (fun () ->
              map
                (split ~axis:1 (Nx.reshape [| 6; 2 |] ids))
                (Nx.reshape [| 6; 2; 4 |] x));
          raises_invalid_arg (fun () ->
              map (Nx.place (Nx.Placement.sharded ~axis:0 four) ids) (split x)));
    ]

let () =
  exit
    (run "nx segments"
       [
         maps;
         group "map_segments at dtypes and sizes" [ dtypes; large_segments ];
         placements;
       ])
