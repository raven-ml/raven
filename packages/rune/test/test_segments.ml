(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Maps over segments under rune: compiled, mapped and differentiated, each
   against the per-position map, which applies [f] to each in-range position's
   row alone and gives zeros elsewhere. Positions number either side of a block
   per position: 3 over 4 segments, and 24 over 3, in blocks of 2 or more. *)

open Windtrap

let ints v = Nx.create Nx.int64 [| Array.length v |] (Array.map Int64.of_int v)

let floats dt shape seed =
  let s = Random.State.make [| seed |] in
  Nx.init dt shape (fun _ -> Random.State.float s 2. -. 1.)

(* [product w owners rows] multiplies each block's rows by its owner's matrix of
   [w] [[| segments; d; e |]]. *)
let product w owners rows =
  Nx.tanh (Nx.matmul rows (Nx.take ~axis:0 ~indices:owners w))

(* [normalised w owners rows] is [product] of the rows divided by their norms:
   zero divided by zero at a row of zeros, whose derivative is NaN. *)
let normalised w owners rows =
  product w owners
    (Nx.div rows
       (Nx.sqrt (Nx.sum ~axes:[ -1 ] ~keepdims:true (Nx.square rows))))

(* [reference f ~segments w ids x] is the per-position map of [f w]: a
   position's row alone, through its owner's matrix, or zeros out of range. The
   rows of positions out of range reach no operation. *)
let reference f ~segments w ids x =
  let s = Nx.shape ids and d = Nx.dim (-1) x in
  let n = Array.fold_left ( * ) 1 s and e = Nx.dim (-1) w in
  let ids = Nx.reshape [| n |] ids in
  let x = Nx.reshape [| n; d |] (Nx.broadcast_to (Array.append s [| d |]) x) in
  let named i =
    let id = Nx.item [ i ] ids in
    id >= 0L && id < Int64.of_int segments
  in
  let row i =
    let id = Nx.slice [ I i ] ids in
    Nx.reshape [| e |]
      (f w (Nx.reshape [| 1 |] id)
         (Nx.reshape [| 1; 1; d |] (Nx.slice [ I i ] x)))
  in
  if n = 0 then Nx.zeros (Nx.dtype x) (Array.append s [| e |])
  else
    Nx.reshape (Array.append s [| e |])
      (Nx.stack
         (List.init n (fun i ->
              if named i then row i else Nx.zeros (Nx.dtype x) [| e |])))

let mapped f ~segments w ids x = Nx.map_segments ~segments ids (f w) x

(* Routes: 3 positions over 4 segments, a block each, and 24 over 3, in blocks,
   whose last segment no position names. A route's rows have [lead] leading
   axes, its ids' shape unless positions share rows; a row that no in-range
   position reads holds NaN. *)
type route = {
  name : string;
  segments : int;
  ids : Nx.int64_t;
  lead : int array option;
}

let routes =
  [
    {
      name = "a block per position";
      segments = 4;
      ids = ints [| 3; -1; 0 |];
      lead = None;
    };
    {
      name = "blocks of positions, the last segment empty";
      segments = 3;
      ids =
        ints
          (Array.init 24 (fun i ->
               match i mod 7 with 0 -> -1 | 3 -> 5 | k -> k mod 2));
      lead = None;
    };
  ]

(* A block per position over rows that positions share: one token's row read by
   3 positions all at -1, and a row per token of 3, the last token's ids all at
   -1. *)
let shared_one =
  {
    name = "one row shared by positions all at -1";
    segments = 4;
    ids = ints [| -1; -1; -1 |];
    lead = Some [| 1 |];
  }

let shared_per_token =
  {
    name = "a row per token, one token's ids all at -1";
    segments = 5;
    ids = Nx.reshape [| 3; 3 |] (ints [| 1; -1; -1; 3; -1; -1; -1; -1; -1 |]);
    lead = Some [| 3; 1 |];
  }

(* Routes at the edges: extreme ids on either side of a block per position, no
   position, no segment, one position, one segment in blocks of 8, and a row per
   token of 3 in blocks of 4, one token's ids all at -1. *)
let ids64 shape v = Nx.create Nx.int64 shape v

let edges =
  [
    {
      name = "extreme ids, a block per position";
      segments = 4;
      ids = ids64 [| 4 |] [| Int64.min_int; 2L; Int64.max_int; 4L |];
      lead = None;
    };
    {
      name = "extreme ids, in blocks";
      segments = 2;
      ids =
        ids64 [| 16 |]
          (Array.init 16 (fun i ->
               match i mod 8 with
               | 0 -> Int64.min_int
               | 3 -> Int64.max_int
               | 5 -> 2L
               | 6 -> -1L
               | k -> Int64.of_int (k mod 2)));
      lead = None;
    };
    {
      name = "no position";
      segments = 3;
      ids = ids64 [| 0 |] [||];
      lead = None;
    };
    {
      name = "no segment";
      segments = 0;
      ids = ints [| -1; 0; 2 |];
      lead = None;
    };
    { name = "one position"; segments = 3; ids = ints [| 2 |]; lead = None };
    {
      name = "one segment, in blocks";
      segments = 1;
      ids = ints (Array.init 24 (fun i -> if i mod 5 = 2 then -1 else 0));
      lead = None;
    };
    {
      name = "a row per token, in blocks, one token's ids all at -1";
      segments = 2;
      ids =
        Nx.reshape [| 8; 3 |]
          (ints
             (Array.init 24 (fun i ->
                  if i / 3 = 3 then -1 else [| 1; 0; -1; 1 |].(i mod 4))));
      lead = Some [| 8; 1 |];
    };
  ]

let d = 8
let e = 6

(* [rows ?poison dt r] is [r]'s rows, [poison], NaN by default, where no
   in-range position reads a row. *)
let rows ?(poison = Float.nan) dt r =
  let s = Nx.shape r.ids in
  let lead = Option.value r.lead ~default:s in
  let shared =
    List.filter
      (fun a -> lead.(a) = 1 && s.(a) > 1)
      (List.init (Array.length s) Fun.id)
  in
  let named =
    Nx.cast Nx.int32
      (Nx.logical_and
         (Nx.greater_equal_s r.ids 0L)
         (Nx.less_s r.ids (Int64.of_int r.segments)))
  in
  let read =
    Nx.reshape
      (Array.append lead [| 1 |])
      (Nx.greater_s (Nx.max ~axes:shared ~keepdims:true named) 0l)
  in
  let shape = Array.append lead [| d |] in
  Nx.where read (floats dt shape 3) (Nx.full dt shape poison)

let weights dt r = floats dt [| r.segments; d; e |] 5

(* A loss weighting each value differently, so that a gradient tells its
   positions apart. *)
let weighted y =
  Nx.sum
    (Nx.mul y
       (Nx.reshape (Nx.shape y)
          (Nx.sin (Nx.arange_f (Nx.dtype y) 0. (float_of_int (Nx.numel y)) 1.))))

let close () = Oracle.tensor ~rel:1e-5 ~abs:1e-6 ()
let pair = Nx.Ptree.(pair tensor tensor)
let grads = Nx.Ptree.(pair tensor tensor @-> returns (pair tensor tensor))

let values =
  group "values"
    [
      cases
        ~name:(fun r -> r.name)
        "compiled, a map is the per-position map"
        (routes @ [ shared_one; shared_per_token ] @ edges)
        (fun r ->
          let w = weights Nx.float32 r and x = rows Nx.float32 r in
          let compiled =
            Rune.jit
              Nx.Ptree.(tensor @-> tensor @-> returns tensor)
              (mapped product ~segments:r.segments w)
              r.ids x
          in
          equal (close ())
            (reference product ~segments:r.segments w r.ids x)
            compiled;
          equal (close ())
            (mapped product ~segments:r.segments w r.ids x)
            compiled);
    ]

(* Derivatives *)

let derivatives =
  group "derivatives"
    [
      cases
        ~name:(fun r -> r.name)
        "the gradient in the rows and the matrices is the per-position map's, \
         and a finite difference's"
        (routes @ [ shared_one; shared_per_token ] @ edges)
        (fun r ->
          let w = weights Nx.float64 r and x = rows Nx.float64 r in
          let loss map (x, w) = weighted (map ~segments:r.segments w r.ids x) in
          let gx, gw = Rune.grad pair (loss (mapped product)) (x, w) in
          let ex, ew = Rune.grad pair (loss (reference product)) (x, w) in
          equal ~msg:"rows" (close ()) ex gx;
          equal ~msg:"matrices" (close ()) ew gw;
          let cx, cw =
            Rune.jit grads (Rune.grad pair (loss (mapped product))) (x, w)
          in
          equal ~msg:"compiled rows" (close ()) ex cx;
          equal ~msg:"compiled matrices" (close ()) ew cw;
          (* Along a direction of the matrices: rows out of range hold NaN. *)
          let v = floats Nx.float64 (Nx.shape w) 9 in
          let fd =
            Oracle.central ~eps:1e-6 (fun w -> loss (mapped product) (x, w)) w v
          in
          equal ~msg:"finite difference" (float 1e-6) (Nx.item [] fd)
            (Oracle.dot gw v));
      cases
        ~name:(fun r -> r.name)
        "a NaN row out of range adds zero, and a row of zeros is never \
         differentiated"
        (routes @ [ shared_per_token ] @ edges)
        (fun r ->
          let w = weights Nx.float64 r and x = rows Nx.float64 r in
          let loss map (x, w) =
            weighted (map normalised ~segments:r.segments w r.ids x)
          in
          let near = Oracle.tensor ~rel:1e-9 () in
          let ex, ew = Rune.grad pair (loss reference) (x, w) in
          let gx, gw = Rune.grad pair (loss mapped) (x, w) in
          equal ~msg:"rows" near ex gx;
          equal ~msg:"matrices" near ew gw;
          let cx, cw = Rune.jit grads (Rune.grad pair (loss mapped)) (x, w) in
          equal ~msg:"compiled rows" near ex cx;
          equal ~msg:"compiled matrices" near ew cw);
      cases
        ~name:(fun r -> r.name)
        "the tangent is the per-position map's, eager and compiled"
        (routes @ [ shared_one; shared_per_token ] @ edges)
        (fun r ->
          let w = weights Nx.float64 r and x = rows Nx.float64 r in
          let t = floats Nx.float64 (Nx.shape x) 11 in
          let f map x = map ~segments:r.segments w r.ids x in
          let y, dy = Rune.jvp' (f (mapped product)) x t in
          let y', dy' = Rune.jvp' (f (reference product)) x t in
          equal ~msg:"primal" (close ()) y' y;
          equal ~msg:"tangent" (close ()) dy' dy;
          equal ~msg:"compiled tangent" (close ()) dy'
            (Rune.jit' (fun x -> snd (Rune.jvp' (f (mapped product)) x t)) x));
    ]

(* Infinite rows *)

let infinite =
  cases
    ~name:(fun (r, v) -> Printf.sprintf "%s, %g" r.name v)
    "an infinite row out of range adds zero to both gradients, eager and \
     compiled"
    (List.concat_map
       (fun r -> [ (r, Float.infinity); (r, Float.neg_infinity) ])
       (routes @ [ shared_one; shared_per_token ] @ edges))
    (fun (r, poison) ->
      let w = weights Nx.float64 r and x = rows ~poison Nx.float64 r in
      let loss map (x, w) =
        weighted (map product ~segments:r.segments w r.ids x)
      in
      let near = Oracle.tensor ~rel:1e-9 () in
      let ex, ew = Rune.grad pair (loss reference) (x, w) in
      let gx, gw = Rune.grad pair (loss mapped) (x, w) in
      equal ~msg:"rows" near ex gx;
      equal ~msg:"matrices" near ew gw;
      let cx, cw = Rune.jit grads (Rune.grad pair (loss mapped)) (x, w) in
      equal ~msg:"compiled rows" near ex cx;
      equal ~msg:"compiled matrices" near ew cw)

(* Dtypes *)

type fdt = F : (float, 'b) Nx.dtype -> fdt

(* [scaled v owners rows] multiplies each row by its owner's vector of [v], [[|
   segments; 1; d |]], element by element: one rounding per element, at any
   dtype. *)
let scaled v owners rows = Nx.mul rows (Nx.take ~axis:0 ~indices:owners v)

let dtypes =
  cases
    ~name:(fun (F dt, r) -> Format.asprintf "%a, %s" Nx.pp_dtype dt r.name)
    "at every float dtype, a map is the per-position map bit for bit, eager \
     and compiled"
    (List.concat_map
       (fun dt ->
         List.map
           (fun r -> (dt, r))
           (routes @ [ shared_one; shared_per_token ] @ edges))
       [
         F Nx.float16;
         F Nx.bfloat16;
         F Nx.float32;
         F Nx.float64;
         F Nx.float8_e4m3;
         F Nx.float8_e5m2;
       ])
    (fun (F dt, r) ->
      let v = floats dt [| r.segments; 1; d |] 5 and x = rows dt r in
      let expected = reference scaled ~segments:r.segments v r.ids x in
      equal ~msg:"eager" (Oracle.tensor ()) expected
        (mapped scaled ~segments:r.segments v r.ids x);
      equal ~msg:"compiled" (Oracle.tensor ()) expected
        (Rune.jit
           Nx.Ptree.(tensor @-> tensor @-> returns tensor)
           (mapped scaled ~segments:r.segments v)
           r.ids x))

(* Maps *)

let lanes =
  cases
    ~name:(fun r -> r.name)
    "a map over ids and rows is each lane's map, eager and compiled" routes
    (fun r ->
      let w = weights Nx.float32 r in
      let lane i =
        ( Nx.where (Nx.equal_s r.ids (-1L)) r.ids
            (Nx.mod_s
               (Nx.add_s r.ids (Int64.of_int i))
               (Int64.of_int r.segments)),
          floats Nx.float32 [| Nx.dim 0 r.ids; d |] (20 + i) )
      in
      let ids = Nx.stack (List.init 3 (fun i -> fst (lane i)))
      and xs = Nx.stack (List.init 3 (fun i -> snd (lane i))) in
      let f (ids, x) = mapped product ~segments:r.segments w ids x in
      let s = Nx.Ptree.(pair tensor tensor @-> returns tensor) in
      let expected = Nx.stack (List.init 3 (fun i -> f (lane i))) in
      equal ~msg:"eager" (Oracle.tensor ()) expected (Rune.vmap s f (ids, xs));
      equal ~msg:"compiled" (close ()) expected
        (Rune.jit s (Rune.vmap s f) (ids, xs)))

(* Three lanes of [r]: its ids, its in-range ids moved to the next segment, and
   ids all at -1, each with its rows, NaN where no in-range position reads one
   and shifted by the lane. *)
let lanes_of r =
  let segments = Int64.of_int r.segments in
  let named =
    Nx.logical_and (Nx.greater_equal_s r.ids 0L) (Nx.less_s r.ids segments)
  in
  let moved =
    if r.segments = 0 then r.ids
    else Nx.where named (Nx.mod_s (Nx.add_s r.ids 1L) segments) r.ids
  in
  List.mapi
    (fun i ids ->
      (ids, Nx.add_s (rows Nx.float32 { r with ids }) (float_of_int i)))
    [ r.ids; moved; Nx.full_like r.ids (-1L) ]

let lanes_poisoned =
  cases
    ~name:(fun r -> r.name)
    "a map over lanes is each lane's map bit for bit, a lane all at -1 and NaN \
     rows out of range included, over ids and rows, rows alone and ids alone"
    (routes @ [ shared_one; shared_per_token ] @ edges)
    (fun r ->
      let w = weights Nx.float32 r in
      let lanes = lanes_of r in
      let ids = Nx.stack (List.map fst lanes)
      and xs = Nx.stack (List.map snd lanes) in
      let f (ids, x) = mapped product ~segments:r.segments w ids x in
      let s = Nx.Ptree.(pair tensor tensor @-> returns tensor) in
      let each g = Nx.stack (List.map g lanes) in
      let exact = Oracle.tensor () in
      equal ~msg:"eager" exact (each f) (Rune.vmap s f (ids, xs));
      equal ~msg:"compiled" (close ()) (each f)
        (Rune.jit s (Rune.vmap s f) (ids, xs));
      let ids0, x0 = List.hd lanes in
      equal ~msg:"rows alone" exact
        (each (fun (_, x) -> f (ids0, x)))
        (Rune.vmap' (fun x -> f (ids0, x)) xs);
      equal ~msg:"ids alone" exact
        (each (fun (ids, _) -> f (ids, x0)))
        (Rune.vmap' (fun ids -> f (ids, x0)) ids))

(* Devices *)

let devices = List.map Nx.Device.cpu [ 1; 2 ]

let placements =
  cases
    ~name:(fun r -> r.name)
    "compiled, positions split over two devices give one device's map, a \
     device with every id -1 included"
    routes
    (fun r ->
      let n = Nx.dim 0 r.ids in
      let ids =
        Nx.concatenate ~axis:0 [ r.ids; Nx.full Nx.int64 [| n |] (-1L) ]
      in
      let r = { r with ids } in
      let w = weights Nx.float32 r and x = rows Nx.float32 r in
      let split t = Nx.place (Nx.Placement.sharded ~axis:0 devices) t in
      let map =
        Rune.jit
          Nx.Ptree.(tensor @-> tensor @-> returns tensor)
          (mapped product ~segments:r.segments w)
      in
      equal (close ()) (map r.ids x)
        (Nx.place Nx.Placement.host (map (split r.ids) (split x))))

(* A row per token of 3 positions, the second device's tokens all at -1:
   [segments] 8 over 4 tokens gives a block per position on each device, 3 over
   8 blocks of 2. *)
let shared_split =
  cases
    ~name:(fun (segments, tokens) ->
      Printf.sprintf "%d segments, %d tokens" segments tokens)
    "a row per token split over two devices gives one device's map, eager and \
     compiled"
    [ (8, 4); (3, 8) ]
    (fun (segments, tokens) ->
      let ids =
        Nx.reshape [| tokens; 3 |]
          (ints
             (Array.init (3 * tokens) (fun i ->
                  if i >= 3 * tokens / 2 then -1
                  else [| 2; -1; 0; 1; -1 |].(i mod 5) mod segments)))
      in
      let r = { name = ""; segments; ids; lead = Some [| tokens; 1 |] } in
      let w = weights Nx.float32 r and x = rows Nx.float32 r in
      let split t = Nx.place (Nx.Placement.sharded ~axis:0 devices) t in
      let host t = Nx.place Nx.Placement.host t in
      let map = mapped product ~segments w in
      let compiled =
        Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) map
      in
      let expected = reference product ~segments w ids x in
      equal ~msg:"eager" (close ()) expected (host (map (split ids) (split x)));
      equal ~msg:"compiled" (close ()) expected
        (host (compiled (split ids) (split x))))

let () =
  exit
    (run "Rune.map_segments"
       [
         values;
         derivatives;
         group "infinities" [ infinite ];
         group "dtypes" [ dtypes ];
         group "maps" [ lanes; lanes_poisoned ];
         group "placements" [ placements; shared_split ];
       ])
