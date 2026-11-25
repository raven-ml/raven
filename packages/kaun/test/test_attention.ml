(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

let flat t = Nx.to_array (Nx.reshape [| -1 |] t)
let f64 t = Nx.cast Nx.float64 t
let numel = Array.fold_left ( * ) 1
let exactly = array float_exact
let int64s shape a = Nx.create Nx.int64 shape (Array.map Int64.of_int a)
let normal seed dt shape = Nx.Rng.normal ~key:(Nx.Rng.key seed) dt shape

(* Float64 values in [-1, 1] that need no key, for gradient checks. *)
let waves seed shape =
  Nx.create Nx.float64 shape
    (Array.init (numel shape) (fun i ->
         Float.sin ((Float.of_int i *. 2.3) +. seed)))

(* A mask that reads a boolean array at each pair. *)
let captured m (_ : Nx.host Nx.pair) = m

(* Lanes padded after their real tokens see their own causally. *)
let causal_within (p : Nx.host Nx.pair) = Nx.less_equal p.k p.q
let grads_ok = function Ok () -> () | Error m -> fail m

(* Two float64 computations of one attention that order their sums apart: a
   compiled call and eager, or grouped heads and repeated ones. Each lies within
   Nx.attention's bound of the exact result, about (2 d + 4) u a M with u =
   2^-53: under 1e-13 for these inputs (d <= 6, scores a < 10, values M < 4), so
   the two differ by less than 1e-12 after the projections' few roundings. *)
let reordered = array (float 1e-12)

(* Every element is zero, of either sign. *)
let all_zero ~msg a =
  satisfies ~msg ~claim:"every element is zero" exactly
    (Array.for_all (fun x -> x = 0.))
    a

(* A layer's leaves by path. *)

let shapes p =
  List.rev
    (Nx.Ptree.fold
       (Nx.Ptree.instantiate (module Attention))
       (fun path t acc -> (Nx.Ptree.Path.to_string path, Nx.shape t) :: acc)
       p [])

let values p =
  List.rev
    (Nx.Ptree.fold
       (Nx.Ptree.instantiate (module Attention))
       (fun path t acc -> (Nx.Ptree.Path.to_string path, flat (f64 t)) :: acc)
       p [])

(* A layer drawn from [seed] with biases drawn too, so that the output
   projection of zero is not zero. *)
let layer ?q_dim ?kv_dim seed ~embed_dim dt =
  Attention.make ~bias_init:(Init.normal ~stddev:1.) ?q_dim ?kv_dim
    ~key:(Nx.Rng.key seed) ~embed_dim dt

(* Identity projections without bias: [apply] is the core over [x]'s head
   slices. *)
let identity dim =
  let id () = { Linear.w = Nx.eye Nx.float32 dim; b = None } in
  { Attention.q = id (); k = id (); v = id (); out = id () }

(* 2 key-value heads serving 4 query heads of head_dim 2, and the layer with
   each key-value head's projection rows repeated for its query heads. *)
let grouped_pair dt =
  let p =
    Attention.make ~key:(Nx.Rng.key 28) ~bias:false ~kv_dim:4 ~embed_dim:8 dt
  in
  let rows = int64s [| 8 |] [| 0; 1; 0; 1; 2; 3; 2; 3 |] in
  let repeat (l : _ Linear.t) = { l with w = Nx.take ~axis:0 rows l.w } in
  (p, { p with k = repeat p.k; v = repeat p.v })

(* make *)

let test_make_shapes (q_dim, kv_dim) =
  let p =
    Attention.make ?q_dim ?kv_dim ~key:(Nx.Rng.key 1) ~embed_dim:6 Nx.float32
  in
  let q = Option.value q_dim ~default:6 in
  let kv = Option.value kv_dim ~default:6 in
  equal
    (list (pair string (array int)))
    [
      ("q.w", [| q; 6 |]);
      ("q.b", [| q |]);
      ("k.w", [| kv; 6 |]);
      ("k.b", [| kv |]);
      ("v.w", [| kv; 6 |]);
      ("v.b", [| kv |]);
      ("out.w", [| 6; q |]);
      ("out.b", [| 6 |]);
    ]
    (shapes p)

let dims_id (q_dim, kv_dim) =
  let dim = function None -> "default" | Some d -> string_of_int d in
  Printf.sprintf "q_dim %s, kv_dim %s" (dim q_dim) (dim kv_dim)

type projection_args = Defaults | Initializers | No_bias

let projection_args_id = function
  | Defaults -> "Linear.make's defaults"
  | Initializers -> "a w_init and a bias_init"
  | No_bias -> "no bias"

let test_make_projections args =
  let w_init, bias_init, bias =
    match args with
    | Defaults -> (None, None, None)
    | Initializers ->
        (Some (Init.normal ~stddev:2.), Some (Init.uniform ~scale:1.), None)
    | No_bias -> (None, None, Some false)
  in
  let key = Nx.Rng.key 2 in
  let keys = Nx.Rng.split ~n:4 key in
  let proj i ~inputs ~outputs =
    Linear.make ?w_init ?bias_init ?bias ~key:keys.(i) ~inputs ~outputs
      Nx.float32
  in
  let expected =
    {
      Attention.q = proj 0 ~inputs:6 ~outputs:8;
      k = proj 1 ~inputs:6 ~outputs:4;
      v = proj 2 ~inputs:6 ~outputs:4;
      out = proj 3 ~inputs:8 ~outputs:6;
    }
  in
  equal
    (list (pair string exactly))
    (values expected)
    (values
       (Attention.make ?w_init ?bias_init ?bias ~q_dim:8 ~kv_dim:4 ~key
          ~embed_dim:6 Nx.float32))

let test_make_refuses (embed_dim, q_dim, kv_dim) =
  raises_match (Exn.invalid_arg ~sub:"Attention.make") (fun () ->
      Attention.make ?q_dim ?kv_dim ~key:(Nx.Rng.key 0) ~embed_dim Nx.float32)

let refused_dims_id (e, q, kv) =
  let dim = function None -> "default" | Some d -> string_of_int d in
  Printf.sprintf "embed_dim %d, q_dim %s, kv_dim %s" e (dim q) (dim kv)

(* init *)

let test_init_is_make () =
  let key = Nx.Rng.key 3 in
  equal
    (list (pair string exactly))
    (values (Attention.make ~key ~embed_dim:8 Nx.float32))
    (values (Attention.init ~key ~embed_dim:8))

let test_init_glorot () =
  let p = Attention.init ~key:(Nx.Rng.key 4) ~embed_dim:8 in
  (* Glorot-uniform over 8 inputs and 8 outputs: U(-l, l), l = sqrt (6 / 16). *)
  let limit = Float.sqrt (6. /. 16.) in
  List.iter
    (fun (name, (l : (Nx.host Nx.float32_t, Nx.host) Linear.t)) ->
      satisfies ~msg:(name ^ ".w") ~claim:"every weight within sqrt (6 / 16)"
        exactly
        (Array.for_all (fun x -> Float.abs x <= limit))
        (flat l.w);
      all_zero ~msg:(name ^ ".b") (flat (require_some l.b)))
    [ ("q", p.q); ("k", p.k); ("v", p.v); ("out", p.out) ]

(* apply *)

let test_apply_shape shape =
  let p = Attention.init ~key:(Nx.Rng.key 5) ~embed_dim:8 in
  equal (array int) shape
    (Nx.shape (Attention.apply ~head_dim:4 p (Nx.zeros Nx.float32 shape)))

let shape_id s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

(* apply against its pieces: x's batch axes folded into one, queries and keys
   rotated at positions 0 to seq - 1, attend over the split heads, merged
   through the output projection. *)

type layer_mask = No_mask | Causal_tokens | Lane_masks

type layer_case = {
  dtype : int;  (** float32, bfloat16, float16 or float64 *)
  lead : int array;
  seq : int;
  head_dim : int;
  kv_heads : int;
  groups : int;
  embed : int;
  mask : layer_mask;
  lens : int array option;
  rope : bool;
  seed : int;
}

let dtype_names = [| "float32"; "bfloat16"; "float16"; "float64" |]

let pp_layer_case ppf c =
  Format.fprintf ppf
    "%s lead %s seq %d head_dim %d kv_heads %d groups %d embed %d mask %s lens \
     %s rope %b seed %d"
    dtype_names.(c.dtype) (shape_id c.lead) c.seq c.head_dim c.kv_heads c.groups
    c.embed
    (match c.mask with
    | No_mask -> "none"
    | Causal_tokens -> "causal"
    | Lane_masks -> "per lane")
    (match c.lens with None -> "none" | Some l -> shape_id l)
    c.rope c.seed

let layer_gen =
  let open Gen in
  let* dtype = int_range 0 3 in
  let* rank = int_range 0 2 in
  let* lead =
    array ~size:(constant rank)
      (frequency [ (1, constant 0); (2, int_range 1 2) ])
  in
  let* seq = int_range 1 4 in
  let* head_dim = of_list [ 2; 4 ] in
  let* kv_heads = int_range 1 2 in
  let* groups = int_range 1 3 in
  let* embed = int_range 1 6 in
  let* mask = of_list [ No_mask; Causal_tokens; Lane_masks ] in
  let* lens = option (array ~size:(constant (numel lead)) (int_range 0 seq)) in
  let* rope = bool in
  let+ seed = int_range 0 1000 in
  {
    dtype;
    lead;
    seq;
    head_dim;
    kv_heads;
    groups;
    embed;
    mask;
    lens;
    rope;
    seed;
  }

let check_pieces (type b) c (dt : (float, b) Nx.dtype) =
  let heads = c.kv_heads * c.groups and batch = numel c.lead in
  let p =
    layer c.seed ~q_dim:(heads * c.head_dim) ~kv_dim:(c.kv_heads * c.head_dim)
      ~embed_dim:c.embed dt
  in
  let shape = Array.append c.lead [| c.seq; c.embed |] in
  let x = normal (c.seed + 1) dt shape in
  let mask =
    match c.mask with
    | No_mask -> None
    | Causal_tokens -> Some Nx.causal
    | Lane_masks ->
        let shown =
          Nx.less_s
            (Nx.Rng.uniform
               ~key:(Nx.Rng.key (c.seed + 2))
               Nx.float32
               [| batch; 1; c.seq; c.seq |])
            0.6
        in
        Some (captured shown)
  in
  let lengths = Option.map (int64s c.lead) c.lens in
  let rope =
    if c.rope then Some (Rope.make ~head_dim:c.head_dim ~context:8 ()) else None
  in
  let pieces =
    let x = Nx.reshape [| batch; c.seq; c.embed |] x in
    let pos = Nx.reshape [| 1; c.seq |] (Nx.arange Nx.int64 0 c.seq 1) in
    let rotate t =
      match rope with None -> t | Some r -> Rope.apply r ~pos t
    in
    let q = rotate (Attention.split ~head_dim:c.head_dim p.q x) in
    let k = rotate (Attention.split ~head_dim:c.head_dim p.k x) in
    let v = Attention.split ~head_dim:c.head_dim p.v x in
    let lengths = Option.map (Nx.reshape [| batch |]) lengths in
    Nx.reshape shape
      (Attention.merge p.out
         (Attention.attend ?mask ?lengths q (Nx.Dense { k; v })))
  in
  equal exactly
    (flat (f64 pieces))
    (flat (f64 (Attention.apply ~head_dim:c.head_dim ?mask ?lengths ?rope p x)))

let law_pieces c =
  (* A batch of no lane compares no value, so every other cover needs a lane. *)
  let lanes = numel c.lead > 0 in
  cover "a batch of no lane" (not lanes);
  cover "float16" (lanes && c.dtype = 2);
  cover "bfloat16" (lanes && c.dtype = 1);
  cover "float64" (lanes && c.dtype = 3);
  cover "grouped heads" (lanes && c.groups > 1);
  cover "no batch axis" (c.lead = [||]);
  cover "two batch axes" (lanes && Array.length c.lead = 2);
  cover "causal" (lanes && c.mask = Causal_tokens);
  cover "a mask per lane" (lanes && c.mask = Lane_masks);
  cover "lengths" (lanes && c.lens <> None);
  cover "rope" (lanes && c.rope);
  match c.dtype with
  | 0 -> check_pieces c Nx.float32
  | 1 -> check_pieces c Nx.bfloat16
  | 2 -> check_pieces c Nx.float16
  | _ -> check_pieces c Nx.float64

let test_heads_attend_apart () =
  (* Identity projections, embed 2, 2 heads of dimension 1: head [h] is the core
     over column [h] of [x] alone. *)
  let x0 = [| 0.5; -1.0 |] and x1 = [| 1.0; 2.0 |] and x2 = [| -0.5; 0.0 |] in
  let x =
    Nx.create Nx.float32 [| 3; 2 |]
      [| x0.(0); x0.(1); x1.(0); x1.(1); x2.(0); x2.(1) |]
  in
  let head h =
    let col = Nx.create Nx.float32 [| 3; 1 |] [| x0.(h); x1.(h); x2.(h) |] in
    flat (Attention.scaled_dot_product_attention col col col)
  in
  let h0 = head 0 and h1 = head 1 in
  equal exactly
    [| h0.(0); h1.(0); h0.(1); h1.(1); h0.(2); h1.(2) |]
    (flat (Attention.apply ~head_dim:1 (identity 2) x))

let test_grouped_equals_repeated () =
  let grouped, repeated = grouped_pair Nx.float64 in
  let x = normal 29 Nx.float64 [| 2; 5; 8 |] in
  let at p = flat (Attention.apply ~head_dim:2 ~mask:Nx.causal p x) in
  equal reordered (at repeated) (at grouped)

let test_apply_causal () =
  let p = layer 6 ~embed_dim:8 Nx.float32 in
  let x = normal 7 Nx.float32 [| 2; 4; 8 |] in
  let later = Nx.add_s (Nx.slice [ A; R (3, 4) ] x) 10. in
  let x' = Nx.concatenate ~axis:1 [ Nx.slice [ A; R (0, 3) ] x; later ] in
  let first3 x =
    flat
      (Nx.slice
         [ A; R (0, 3) ]
         (Attention.apply ~head_dim:4 ~mask:Nx.causal p x))
  in
  equal exactly (first3 x) (first3 x')

let test_apply_no_key () =
  let p = layer 8 ~embed_dim:4 Nx.float32 in
  let x = normal 9 Nx.float32 [| 2; 3; 4 |] in
  (* Token 0 sees no key; the others see every key. *)
  let shown =
    Nx.create Nx.bool [| 3; 3 |]
      [| false; false; false; true; true; true; true; true; true |]
  in
  let y = Attention.apply ~head_dim:2 ~mask:(captured shown) p x in
  let b = flat (require_some p.out.b) in
  equal exactly (Array.append b b) (flat (Nx.slice [ A; I 0 ] y))

let test_apply_unmasked () =
  let p = layer 10 ~embed_dim:8 Nx.float32 in
  let x = normal 11 Nx.float32 [| 2; 5; 8 |] in
  let every = captured (Nx.ones Nx.bool [| 5; 5 |]) in
  equal exactly
    (flat (Attention.apply ~head_dim:2 ~mask:every p x))
    (flat (Attention.apply ~head_dim:2 p x))

(* Lanes padded after their real tokens, the padding NaN. A real token's output
   is the one it has in its lane cut to its real tokens. [lead] is x's batch
   axes and [lens] one length per lane. *)
let padded ~lead ~lens =
  let x = normal 12 Nx.float64 (Array.append lead [| 4; 4 |]) in
  let x = Nx.reshape [| numel lead; 4; 4 |] x in
  let lane b =
    let real = Nx.slice [ R (b, b + 1); R (0, lens.(b)) ] x in
    let pad = Nx.full Nx.float64 [| 1; 4 - lens.(b); 4 |] Float.nan in
    Nx.concatenate ~axis:1 [ real; pad ]
  in
  let x = Nx.concatenate ~axis:0 (List.init (numel lead) lane) in
  (Nx.reshape (Array.append lead [| 4; 4 |]) x, int64s lead lens)

let real ~lens y =
  let y = Nx.reshape [| Array.length lens; 4; 4 |] y in
  Array.concat
    (List.mapi
       (fun b n -> flat (Nx.slice [ I b; R (0, n) ] y))
       (Array.to_list lens))

let alone ~causal ~lens p x =
  let x = Nx.reshape [| Array.length lens; 4; 4 |] x in
  let mask = if causal then Some Nx.causal else None in
  Array.concat
    (List.mapi
       (fun b n ->
         flat
           (Attention.apply ~head_dim:2 ?mask p
              (Nx.slice [ R (b, b + 1); R (0, n) ] x)))
       (Array.to_list lens))

let test_lengths_hide_padding (lead, lens, causal) =
  let p = layer 13 ~embed_dim:4 Nx.float64 in
  let x, lengths = padded ~lead ~lens in
  let mask = if causal then Some causal_within else None in
  equal exactly (alone ~causal ~lens p x)
    (real ~lens (Attention.apply ~head_dim:2 ?mask ~lengths p x))

let lengths_id (lead, lens, causal) =
  Printf.sprintf "batch axes %s, lengths %s, %s" (shape_id lead) (shape_id lens)
    (if causal then "causal" else "no mask")

let test_lengths_compiled () =
  Compiled.here ();
  let p = layer 13 ~embed_dim:4 Nx.float64 in
  let lens = [| 2; 4 |] in
  let x, lengths = padded ~lead:[| 2 |] ~lens in
  let both x =
    Nx.stack
      [
        Attention.apply ~head_dim:2 ~lengths p x;
        Attention.apply ~head_dim:2 ~mask:causal_within ~lengths p x;
      ]
  in
  let y = Rune.jit' both x in
  equal reordered
    (Array.append
       (alone ~causal:false ~lens p x)
       (alone ~causal:true ~lens p x))
    (Array.append (real ~lens (Nx.get [ 0 ] y)) (real ~lens (Nx.get [ 1 ] y)))

let test_causal_alignment () =
  let p = layer 14 ~embed_dim:4 Nx.float64 in
  let x = normal 15 Nx.float64 [| 3; 4; 4 |] in
  let lens = [| 2; 3; 4 |] in
  (* Token i of a lane of [len] real tokens sees the keys j <= i - (4 - len). *)
  let shifted =
    Nx.create Nx.bool [| 3; 1; 4; 4 |]
      (Array.init 48 (fun e ->
           let b = e / 16 and i = e / 4 mod 4 and j = e mod 4 in
           j <= i - (4 - lens.(b))))
  in
  let lengths = int64s [| 3 |] lens in
  equal exactly
    (flat (Attention.apply ~head_dim:2 ~mask:(captured shifted) ~lengths p x))
    (flat (Attention.apply ~head_dim:2 ~mask:Nx.causal ~lengths p x))

type grad_layer = Plain | Causal_mask | Rotated | Grouped

let grad_layer_id = function
  | Plain -> "no mask"
  | Causal_mask -> "causal"
  | Rotated -> "causal, rotated"
  | Grouped -> "causal, grouped heads"

let test_apply_gradients row =
  let p, x =
    match row with
    | Grouped -> (fst (grouped_pair Nx.float64), waves 1. [| 2; 3; 8 |])
    | _ -> (layer 16 ~embed_dim:4 Nx.float64, waves 1. [| 3; 4 |])
  in
  let mask = if row = Plain then None else Some Nx.causal in
  let rope =
    if row = Rotated then Some (Rope.make ~head_dim:2 ~context:64 ()) else None
  in
  let loss p =
    let y = Attention.apply ~head_dim:2 ?mask ?rope p x in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune.check_grads (Nx.Ptree.instantiate (module Attention)) loss p)

(* Each row is a call that differs from a valid one in one argument, and the
   part of the message that names it. *)
let apply_refusals =
  let p () = Attention.init ~key:(Nx.Rng.key 17) ~embed_dim:4 in
  let x () = Nx.zeros Nx.float32 [| 2; 4 |] in
  let apply ?(head_dim = 2) ?mask ?lengths p x =
    ignore (Attention.apply ~head_dim ?mask ?lengths p x)
  in
  [
    ( "x is a float8 format",
      "float8_e4m3fn is not",
      fun () ->
        apply
          (Nx.Ptree.cast (module Attention) Nx.float8_e4m3fn (p ()))
          (Nx.zeros Nx.float8_e4m3fn [| 2; 4 |]) );
    ( "x has fewer than 2 axes",
      "at least sequence and feature axes",
      fun () -> apply (p ()) (Nx.zeros Nx.float32 [| 4 |]) );
    ( "x's last axis is not embed_dim",
      "last axis has size 3",
      fun () -> apply (p ()) (Nx.zeros Nx.float32 [| 2; 3 |]) );
    ( "head_dim is not positive",
      "head_dim must be positive",
      fun () -> apply ~head_dim:0 (p ()) (x ()) );
    ( "head_dim does not divide the widths",
      "head_dim 3 does not divide",
      fun () -> apply ~head_dim:3 (p ()) (x ()) );
    ( "the key and value projections differ in width",
      "differ in width",
      fun () ->
        let v =
          Linear.make ~key:(Nx.Rng.key 18) ~inputs:4 ~outputs:2 Nx.float32
        in
        apply { (p ()) with v } (x ()) );
    ( "kv_heads does not divide heads",
      "2 key-value heads do not divide 3",
      fun () ->
        apply
          (Attention.make ~q_dim:6 ~kv_dim:4 ~key:(Nx.Rng.key 19) ~embed_dim:4
             Nx.float32)
          (x ()) );
    ( "lengths is not of x's batch axes",
      "lengths must have shape [3]",
      fun () ->
        apply
          ~lengths:(Nx.zeros Nx.int64 [| 2 |])
          (p ())
          (Nx.zeros Nx.float32 [| 3; 2; 4 |]) );
    ( "the mask does not broadcast",
      "does not broadcast",
      fun () ->
        apply ~mask:(captured (Nx.ones Nx.bool [| 3; 3 |])) (p ()) (x ()) );
  ]

(* split *)

(* Head h of token s holds features h * head_dim to (h + 1) * head_dim - 1 of
   the token's projection. *)
let test_split head_dim =
  let l = Linear.make ~key:(Nx.Rng.key 20) ~inputs:4 ~outputs:6 Nx.float32 in
  let x = normal 21 Nx.float32 [| 2; 3; 4 |] in
  let y = flat (Linear.apply l x) and heads = 6 / head_dim in
  let expected =
    Array.init
      (2 * heads * 3 * head_dim)
      (fun i ->
        let e = i mod head_dim and s = i / head_dim mod 3 in
        let h = i / (head_dim * 3) mod heads
        and b = i / (head_dim * 3 * heads) in
        y.((((b * 3) + s) * 6) + (h * head_dim) + e))
  in
  let got = Attention.split ~head_dim l x in
  equal
    (pair (array int) exactly)
    ([| 2; heads; 3; head_dim |], expected)
    (Nx.shape got, flat got)

let split_refusals =
  let split ~head_dim x =
    let l = Linear.make ~key:(Nx.Rng.key 22) ~inputs:4 ~outputs:6 Nx.float32 in
    ignore (Attention.split ~head_dim l x)
  in
  [
    ( "x does not have three axes",
      "x must have shape [batch; seq; embed]",
      fun () -> split ~head_dim:2 (Nx.zeros Nx.float32 [| 2; 4 |]) );
    ( "head_dim is not positive",
      "head_dim must be positive",
      fun () -> split ~head_dim:0 (Nx.zeros Nx.float32 [| 1; 2; 4 |]) );
    ( "head_dim does not divide the width",
      "head_dim 4 does not divide",
      fun () -> split ~head_dim:4 (Nx.zeros Nx.float32 [| 1; 2; 4 |]) );
  ]

(* attend *)

(* Lanes that share dense rows: lane [i] of [batch] reads row [i / (batch /
   rows)], and its result is the one it has over a copy of that row. *)
let shared_rows ~batch ~rows ~m =
  let q = normal 31 Nx.float64 [| batch; 4; 2; 3 |] in
  let k = normal 32 Nx.float64 [| rows; 2; m; 3 |] in
  let v = normal 33 Nx.float64 [| rows; 2; m; 2 |] in
  let lanes = batch / rows in
  let copy t =
    Nx.take ~axis:0
      (int64s [| batch |] (Array.init batch (fun i -> i / lanes)))
      t
  in
  let sinks = normal 34 Nx.float64 [| 4 |] in
  let f k v q = Attention.attend ~sinks q (Nx.Dense { k; v }) in
  (q, f k v, f (copy k) (copy v))

let test_shared_rows (batch, rows, m) =
  Compiled.here ();
  let q, shared, copied = shared_rows ~batch ~rows ~m in
  let expected = flat (copied q) in
  equal ~msg:"eager" exactly expected (flat (shared q));
  equal ~msg:"compiled" reordered expected (flat (Rune.jit' shared q))

let shared_id (batch, rows, m) =
  Printf.sprintf "%d lanes over %d rows of %d keys" batch rows m

module Qkvs = struct
  type t = {
    q : Nx.host Nx.float64_t;
    k : Nx.host Nx.float64_t;
    v : Nx.host Nx.float64_t;
    sinks : Nx.host Nx.float64_t;
  }

  let walk c { q; k; v; sinks } =
    let open Nx.Ptree.Walk in
    let q = field c "q" tensor q in
    let k = field c "k" tensor k in
    let v = field c "v" tensor v in
    let sinks = field c "sinks" tensor sinks in
    { q; k; v; sinks }
end

let qkvs = Nx.Ptree.v Qkvs.walk

let qkvs_of ~q ~k ~v ~sinks =
  {
    Qkvs.q = waves 1. q;
    k = waves 2. k;
    v = waves 3. v;
    sinks = waves 4. sinks;
  }

let test_shared_rows_gradients () =
  let p =
    qkvs_of ~q:[| 4; 2; 3; 2 |] ~k:[| 2; 1; 4; 2 |] ~v:[| 2; 1; 4; 3 |]
      ~sinks:[| 2 |]
  in
  let loss { Qkvs.q; k; v; sinks } =
    let y = Attention.attend ~sinks q (Nx.Dense { k; v }) in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune.check_grads qkvs loss p)

(* Keys past a lane's length are NaN. *)
let poison_past n t =
  let s = Nx.shape t in
  let r = Array.length s in
  let rows = List.init (r - 2) (fun _ -> Nx.A) @ [ Nx.R (n, s.(r - 2)) ] in
  let pad = Array.copy s in
  pad.(r - 2) <- s.(r - 2) - n;
  Nx.set rows (Nx.full (Nx.dtype t) pad Float.nan) t

(* Rows of their own: over [batch * n] dense rows, query [i] of lane [b] reads
   row [b * n + i] alone, and its result is the one it has as a lane of one
   query over that row, its mask read at its own pair, its bias, dropout keys
   and length its lane's. A length past the [m] keys reads as [m] in
   [Nx.causal]. 4 query heads over 2 key heads of 3 features, values of 2. *)
type own = {
  dtype : int;
  batch : int;
  n : int;
  m : int;
  mask_heads : bool;
  bias_heads : bool;
  lens : int array;
  seed : int;
}

let pp_own ppf c =
  Format.fprintf ppf
    "%s batch %d n %d m %d mask heads %b bias heads %b lens %s seed %d"
    dtype_names.(c.dtype) c.batch c.n c.m c.mask_heads c.bias_heads
    (shape_id c.lens) c.seed

(* Both sides read each row through Nx.attention alone: the results round once
   to the dtype from the same exact values, so they differ by at most two
   roundings of values below 4, the dropout's doubling included. *)
let own_tolerance = [| 0x1p-21; 0x1p-5; 0x1p-8; 0x1p-50 |]

let own_rows (type b) ?(compiled = false) (dt : (float, b) Nx.dtype) c =
  let { batch; n; m; seed; _ } = c in
  let q = normal seed dt [| batch; 4; n; 3 |] in
  let k = normal (seed + 1) dt [| batch * n; 2; m; 3 |] in
  let v = normal (seed + 2) dt [| batch * n; 2; m; 2 |] in
  let sinks = normal (seed + 3) dt [| 4 |] in
  let width heads = if heads then 4 else 1 in
  let bias =
    normal (seed + 4) Nx.float32 [| batch; width c.bias_heads; n; m |]
  in
  let chosen =
    Nx.less_s
      (Nx.Rng.uniform
         ~key:(Nx.Rng.key (seed + 5))
         Nx.float32
         [| batch; width c.mask_heads; n; m |])
      0.8
  in
  let lengths = int64s [| batch |] c.lens in
  let mask c (p : Nx.host Nx.pair) = Nx.logical_and (Nx.causal p) c in
  let pairs = batch * n * m in
  (* A key per pair, where there is a pair. *)
  let dropout, key =
    if pairs = 0 then (None, None)
    else
      ( Some 0.5,
        Some
          (Nx.Rng.reshape [| batch; 1; n; m |]
             (Nx.Rng.split_batch ~n:pairs (Nx.Rng.key (seed + 6)))) )
  in
  let attend q =
    Attention.attend ~mask:(mask chosen) ~lengths ~bias ~sinks ?dropout ?key q
      (Nx.Dense { k; v })
  in
  let alone b i =
    let at t = Nx.slice [ R (b, b + 1); A; R (i, i + 1) ] t in
    let row t = Nx.slice [ R ((b * n) + i, (b * n) + i + 1) ] t in
    let shifted (p : Nx.host Nx.pair) =
      mask (at chosen) { p with q = Nx.add_s p.q (Int64.of_int i); tq = n }
    in
    let key =
      Option.map (fun k -> Nx.Rng.of_tensor (at (Nx.Rng.to_tensor k))) key
    in
    flat
      (f64
         (Attention.attend ~mask:shifted
            ~lengths:(int64s [| 1 |] [| c.lens.(b) |])
            ~bias:(at bias) ~sinks ?dropout ?key (at q)
            (Nx.Dense { k = row k; v = row v })))
  in
  (* Lane-major, heads before queries, as the result lies. *)
  let expected =
    Array.concat
      (List.concat_map
         (fun b ->
           let queries = Array.init n (alone b) in
           List.init 4 (fun h ->
               Array.concat
                 (List.init n (fun i -> Array.sub queries.(i) (h * 2) 2))))
         (List.init batch Fun.id))
  in
  let within = array (float own_tolerance.(c.dtype)) in
  equal ~msg:"eager" within expected (flat (f64 (attend q)));
  if compiled then
    equal ~msg:"compiled" within expected (flat (f64 (Rune.jit' attend q)))

let law_own_rows c =
  let queries = c.batch * c.n > 0 in
  cover "no query" (not queries);
  cover "no key" (queries && c.m = 0);
  cover "float16" (queries && c.dtype = 2);
  cover "bfloat16" (queries && c.dtype = 1);
  cover "a mask with heads" (queries && c.mask_heads);
  cover "a bias with heads" (queries && c.bias_heads);
  cover "a length past the keys"
    (queries && Array.exists (fun l -> l > c.m) c.lens);
  match c.dtype with
  | 0 -> own_rows Nx.float32 c
  | 1 -> own_rows Nx.bfloat16 c
  | 2 -> own_rows Nx.float16 c
  | _ -> own_rows Nx.float64 c

let own_gen =
  let open Gen in
  let* dtype = int_range 0 3 in
  let* batch = frequency [ (1, constant 0); (6, int_range 1 2) ] in
  let* n = frequency [ (1, constant 0); (6, int_range 2 3) ] in
  let* m = frequency [ (1, constant 0); (6, int_range 1 4) ] in
  let* mask_heads = bool in
  let* bias_heads = bool in
  let* lens = list ~size:(constant batch) (int_range 0 6) in
  let+ seed = int_range 0 1000 in
  { dtype; batch; n; m; mask_heads; bias_heads; lens = Array.of_list lens; seed }

let test_own_rows_compiled () =
  Compiled.here ();
  own_rows ~compiled:true Nx.float64
    {
      dtype = 3;
      batch = 2;
      n = 3;
      m = 4;
      mask_heads = true;
      bias_heads = false;
      lens = [| 3; 9 |];
      seed = 41;
    }

let test_own_rows_gradients () =
  let p =
    qkvs_of ~q:[| 2; 2; 3; 2 |] ~k:[| 6; 1; 4; 2 |] ~v:[| 6; 1; 4; 3 |]
      ~sinks:[| 2 |]
  in
  let hidden =
    Nx.create Nx.bool [| 2; 1; 3; 4 |]
      (Array.init 24 (fun t -> t mod 5 <> 2))
  in
  let loss { Qkvs.q; k; v; sinks } =
    let y =
      Attention.attend ~mask:(captured hidden) ~sinks q (Nx.Dense { k; v })
    in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune.check_grads qkvs loss p)

let test_attend_lengths () =
  let q = normal 35 Nx.float32 [| 2; 2; 3; 4 |] in
  (* Lane 0 holds 1 key, lane 1 all 5. *)
  let lane0 t = poison_past 1 (Nx.slice [ R (0, 1) ] t) in
  let keys seed =
    let t = normal seed Nx.float32 [| 2; 2; 5; 4 |] in
    Nx.concatenate ~axis:0 [ lane0 t; Nx.slice [ R (1, 2) ] t ]
  in
  let k = keys 36 and v = keys 37 in
  let lane i n =
    let keys t = Nx.slice [ R (i, i + 1); A; R (0, n) ] t in
    flat
      (Attention.scaled_dot_product_attention
         (Nx.slice [ R (i, i + 1) ] q)
         (keys k) (keys v))
  in
  equal exactly
    (Array.append (lane 0 1) (lane 1 5))
    (flat
       (Attention.attend
          ~lengths:(int64s [| 2 |] [| 1; 5 |])
          q
          (Nx.Dense { k; v })))

(* Two keys of score 0, a bias of -8.03125 on the second, and values 0 and 1:
   the result is sigmoid (-8.03125). A bias rounded to bfloat16, 8 or 8.0625,
   would be off by 3%, against bfloat16's rounding of the result, under 0.4%. *)
let test_bias_keeps_its_bits () =
  let bf16 = Nx.bfloat16 in
  let q = Nx.zeros bf16 [| 1; 1; 1; 2 |] in
  let k = Nx.zeros bf16 [| 1; 1; 2; 2 |] in
  let v = Nx.create bf16 [| 1; 1; 2; 1 |] [| 0.; 1. |] in
  let bias = Nx.create Nx.float32 [| 1; 1; 1; 2 |] [| 0.; -8.03125 |] in
  let y = Attention.attend ~bias q (Nx.Dense { k; v }) in
  let y = Nx.item [ 0; 0; 0; 0 ] y in
  let want = 1. /. (1. +. exp 8.03125) in
  equal (float (want *. 0x1p-8)) want y

(* Dropout of the weights: queries of two heads over one key-value head of five
   keys. Under the mask, rows 0 and 2 see one key each, row 1 every key, row 3
   none. *)
let drop_visible =
  lazy
    (Nx.create Nx.bool [| 1; 1; 4; 5 |]
       (Array.concat
          [
            [| true; false; false; false; false |];
            [| true; true; true; true; true |];
            [| false; false; true; false; false |];
            [| false; false; false; false; false |];
          ]))

let drop_bias () =
  Nx.create Nx.float32 [| 1; 1; 1; 5 |] [| 0.; 0.5; -1.; 0.25; 2. |]

let drop_sinks () = Nx.create Nx.float64 [| 2 |] [| -0.5; 1.5 |]

(* One key per pair, shared by the heads. *)
let drop_keys =
  lazy
    (Nx.Rng.reshape [| 1; 1; 4; 5 |] (Nx.Rng.split_batch ~n:20 (Nx.Rng.key 6)))

let drop_inputs () =
  ( normal 1 Nx.float64 [| 1; 2; 4; 4 |],
    normal 2 Nx.float64 [| 1; 1; 5; 4 |],
    normal 3 Nx.float64 [| 1; 1; 5; 3 |] )

let drop_mask () =
  Dropout.mask ~rate:0.5 ~key:(Lazy.force drop_keys)
    (Nx.zeros Nx.float64 [| 1; 2; 4; 5 |])

(* What [attend] is given, of which [mask] and [sinks] may be left out. *)
type drop_case = { masked : bool; sinked : bool }

let drop_cases =
  List.concat_map
    (fun masked -> List.map (fun sinked -> { masked; sinked }) [ true; false ])
    [ true; false ]

let drop_case_id c =
  Printf.sprintf "%s, %s"
    (if c.masked then "masked" else "unmasked")
    (if c.sinked then "sinks" else "no sinks")

let dropped c ~bias ~sinks q k v =
  let mask =
    if c.masked then Some (captured (Lazy.force drop_visible)) else None
  in
  let sinks = if c.sinked then Some sinks else None in
  Attention.attend ?mask ~bias ?sinks ~dropout:0.5 ~key:(Lazy.force drop_keys) q
    (Nx.Dense { k; v })

(* The weights in full: the softmax over the visible scores, biased, and a last
   column of the sinks, or of a score so low that its weight is exactly 0 beside
   any visible key and 1 where none is visible; then that column dropped, the
   weights dropped, and the values weighed. *)
let dropped_reference c ~bias ~sinks q k v =
  let s = Nx.mul_s (Nx.matmul q (Nx.swapaxes 2 3 k)) 0.5 in
  let s = Nx.add s (f64 bias) in
  let s =
    if not c.masked then s
    else
      Nx.where (Lazy.force drop_visible) s (Nx.scalar_like s Float.neg_infinity)
  in
  let column =
    if c.sinked then Nx.reshape [| 1; 2; 1; 1 |] sinks
    else Nx.full Nx.float64 [| 1; 1; 1; 1 |] (-1e30)
  in
  let column = Nx.broadcast_to [| 1; 2; 4; 1 |] column in
  let w = Nx.softmax (Nx.concatenate ~axis:3 [ s; column ]) in
  let w = Nx.slice [ A; A; A; R (0, 5) ] w in
  let w = Nx.where (drop_mask ()) (Nx.mul_s w 2.) (Nx.scalar_like w 0.) in
  Nx.matmul w v

(* Which of a row's cases the mask and keys give it: it sees no key, keeps none,
   all or some of those it sees. *)
let drop_row_cases () =
  let vis =
    Nx.to_array (Nx.broadcast_to [| 1; 2; 4; 5 |] (Lazy.force drop_visible))
  in
  let z = Nx.to_array (drop_mask ()) in
  let case r =
    let seen = ref 0 and kept = ref 0 in
    for j = 0 to 4 do
      if vis.((5 * r) + j) then begin
        incr seen;
        if z.((5 * r) + j) then incr kept
      end
    done;
    if !seen = 0 then "sees none"
    else if !kept = 0 then "keeps none"
    else if !kept = !seen then "keeps all"
    else "keeps some"
  in
  List.sort_uniq compare (List.init 8 case)

(* The reference rounds a few times in float64 more than attend: 1e-12 is
   thousands of ulps of results below 10. *)
let drop_close = array (float 1e-12)

let test_dropout_values c =
  equal ~msg:"the rows reach each case" (list string)
    [ "keeps all"; "keeps none"; "keeps some"; "sees none" ]
    (drop_row_cases ());
  let q, k, v = drop_inputs () in
  let bias = drop_bias () and sinks = drop_sinks () in
  equal drop_close
    (flat (dropped_reference c ~bias ~sinks q k v))
    (flat (dropped c ~bias ~sinks q k v))

let test_dropout_gradients c =
  let q, k, v = drop_inputs () in
  let bias = drop_bias () and sinks = drop_sinks () in
  let d f x = Rune.grad Nx.Ptree.tensor (fun x -> Nx.sum (f x)) x in
  let grad ?(close = drop_close) name g =
    equal ~msg:name close
      (flat (g (dropped_reference c)))
      (flat (g (dropped c)))
  in
  grad "q" (fun f -> d (fun q -> f ~bias ~sinks q k v) q);
  grad "k" (fun f -> d (fun k -> f ~bias ~sinks q k v) k);
  grad "v" (fun f -> d (fun v -> f ~bias ~sinks q k v) v);
  (* The bias is float32: each gradient rounds to it once from float64 values
     1e-12 apart, so they differ by at most a float32 ulp, under 1e-6 for
     gradients below 4. *)
  grad
    ~close:(array (float 1e-6))
    "bias"
    (fun f -> d (fun bias -> f ~bias ~sinks q k v) bias);
  if c.sinked then
    grad "sinks" (fun f -> d (fun sinks -> f ~bias ~sinks q k v) sinks)

let test_dropout_compiled () =
  Compiled.here ();
  let q, k, v = drop_inputs () in
  let bias = drop_bias () and sinks = drop_sinks () in
  let c = { masked = true; sinked = true } in
  let f q = dropped c ~bias ~sinks q k v in
  let with_grad q =
    let g = Rune.grad Nx.Ptree.tensor (fun q -> Nx.sum (f q)) q in
    Nx.concatenate ~axis:0 [ Nx.reshape [| -1 |] (f q); Nx.reshape [| -1 |] g ]
  in
  equal reordered (flat (with_grad q)) (flat (Rune.jit' with_grad q))

let test_dropout_zero shared =
  let q, k, v = drop_inputs () in
  let q = if shared then Nx.concatenate ~axis:0 [ q; q ] else q in
  let kv = Nx.Dense { k; v } in
  let mask =
    if shared then None else Some (captured (Lazy.force drop_visible))
  in
  equal exactly
    (flat (Attention.attend ?mask q kv))
    (flat (Attention.attend ?mask ~dropout:0. q kv))

(* Keys of five slots in pages of three, read through a table in reverse. *)
let test_dropout_paged () =
  let q, k, v = drop_inputs () in
  let pad x = Nx.pad [| (0, 0); (0, 0); (0, 1); (0, 0) |] 0. x in
  let paged x =
    let c = Nx.dim 3 x in
    let pages = Nx.reshape [| 2; 1; 3; c |] (pad x) in
    Nx.concatenate ~axis:0
      [ Nx.slice [ R (1, 2) ] pages; Nx.slice [ R (0, 1) ] pages ]
  in
  let table = int64s [| 1; 2 |] [| 1; 0 |] in
  let mask =
    captured
      (Nx.pad
         [| (0, 0); (0, 0); (0, 0); (0, 1) |]
         false (Lazy.force drop_visible))
  in
  let keys =
    Nx.Rng.reshape [| 1; 1; 4; 6 |] (Nx.Rng.split_batch ~n:24 (Nx.Rng.key 6))
  in
  let attend kv = Attention.attend ~mask ~dropout:0.5 ~key:keys q kv in
  equal exactly
    (flat (attend (Nx.Dense { k = pad k; v = pad v })))
    (flat (attend (Nx.Paged { k = paged k; v = paged v; table })))

(* A rate of 0.25 scales by 4/3, which bfloat16 does not hold, so rounding once
   more is seen. *)
let test_dropout_rounds_twice () =
  let bf16 = Nx.bfloat16 in
  let q = Nx.cast bf16 (normal 38 Nx.float32 [| 1; 2; 3; 4 |]) in
  let k = Nx.cast bf16 (normal 39 Nx.float32 [| 1; 1; 6; 4 |]) in
  let v = Nx.cast bf16 (normal 40 Nx.float32 [| 1; 1; 6; 3 |]) in
  let key = Nx.Rng.key 41 in
  let z = Dropout.mask ~rate:0.25 ~key (Nx.zeros bf16 [| 1; 2; 3; 6 |]) in
  let o, _ =
    Nx.attention ~keep:(fun _ -> z) ~acc:Nx.float32 ~kv:(Nx.Dense { k; v }) q
  in
  let expected = Nx.cast bf16 (Nx.mul_s (Nx.cast Nx.float32 o) (1. /. 0.75)) in
  equal exactly
    (flat (f64 expected))
    (flat (f64 (Attention.attend ~dropout:0.25 ~key q (Nx.Dense { k; v }))))

let test_attend_gradients () =
  let p =
    qkvs_of ~q:[| 1; 4; 3; 2 |] ~k:[| 1; 2; 4; 2 |] ~v:[| 1; 2; 4; 3 |]
      ~sinks:[| 4 |]
  in
  let loss { Qkvs.q; k; v; sinks } =
    let y =
      Attention.attend ~mask:Nx.causal ~scale:0.8 ~sinks q (Nx.Dense { k; v })
    in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune.check_grads qkvs loss p)

(* Four query heads of one lane over two key-value heads, or four lanes of two
   heads over rows of two. *)
let attend_refusals =
  let t shape = Nx.zeros Nx.float32 shape in
  let dense ~rows ~heads ~m ~d =
    Nx.Dense { k = t [| rows; heads; m; d |]; v = t [| rows; heads; m; d |] }
  in
  let q () = t [| 1; 4; 2; 2 |] and kv () = dense ~rows:1 ~heads:2 ~m:2 ~d:2 in
  let lanes () = t [| 4; 2; 1; 3 |] in
  let rows n = dense ~rows:n ~heads:2 ~m:5 ~d:3 in
  let attend ?mask ?lengths ?scale ?bias ?sinks ?dropout ?key q kv =
    ignore
      (Attention.attend ?mask ?lengths ?scale ?bias ?sinks ?dropout ?key q kv)
  in
  [
    ( "q is a float8 format",
      "float8_e4m3fn is not",
      fun () ->
        let x = Nx.zeros Nx.float8_e4m3fn [| 1; 1; 1; 2 |] in
        attend x (Nx.Dense { k = x; v = x }) );
    ( "q does not have four axes",
      "q must have shape",
      fun () -> attend (t [| 4; 2; 2 |]) (kv ()) );
    ( "k's d is not q's",
      "k must have shape",
      fun () ->
        attend (q ())
          (Nx.Dense { k = t [| 1; 2; 2; 3 |]; v = t [| 1; 2; 2; 2 |] }) );
    ("0 rows", "rows dividing 4", fun () -> attend (lanes ()) (rows 0));
    ( "3 rows of 4 lanes",
      "rows dividing 4",
      fun () -> attend (lanes ()) (rows 3) );
    ( "8 rows of 4 lanes",
      "rows dividing 4",
      fun () -> attend (lanes ()) (rows 8) );
    ( "6 rows of 4 lanes of 2 queries",
      "rows dividing 4 or one per query, 8",
      fun () -> attend (t [| 4; 2; 2; 3 |]) (rows 6) );
    ( "kv_heads does not divide heads",
      "3 key-value heads do not divide 4",
      fun () -> attend (q ()) (dense ~rows:1 ~heads:3 ~m:2 ~d:2) );
    ( "d is 0 without scale",
      "no feature need a scale",
      fun () ->
        attend
          (t [| 1; 1; 2; 0 |])
          (Nx.Dense { k = t [| 1; 1; 2; 0 |]; v = t [| 1; 1; 2; 2 |] }) );
    ( "sinks are not [heads]",
      "sinks must have shape [4]",
      fun () -> attend ~sinks:(t [| 2 |]) (q ()) (kv ()) );
    ( "lengths are not [batch]",
      "lengths must have shape [1]",
      fun () -> attend ~lengths:(Nx.zeros Nx.int64 [||]) (q ()) (kv ()) );
    ( "the mask does not broadcast",
      "does not broadcast",
      fun () ->
        attend ~mask:(captured (Nx.ones Nx.bool [| 3; 3 |])) (q ()) (kv ()) );
    ( "the bias does not broadcast",
      "a bias of shape [4] does not broadcast",
      fun () -> attend ~bias:(t [| 4 |]) (q ()) (kv ()) );
    ( "dropout is 1",
      "dropout must be in [0, 1), got 1",
      fun () -> attend ~dropout:1. (q ()) (kv ()) );
    ( "dropout is NaN",
      "dropout must be in [0, 1), got nan",
      fun () -> attend ~dropout:Float.nan (q ()) (kv ()) );
    ( "the dropout keys do not broadcast",
      "dropout keys of batch shape [3]",
      fun () ->
        attend ~dropout:0.5
          ~key:(Nx.Rng.split_batch ~n:3 (Nx.Rng.key 1))
          (q ()) (kv ()) );
    ( "shared rows and a mask",
      "shared by several lanes",
      fun () -> attend ~mask:Nx.causal (lanes ()) (rows 2) );
    ( "shared rows and lengths",
      "shared by several lanes",
      fun () ->
        attend ~lengths:(Nx.full Nx.int64 [| 4 |] 5L) (lanes ()) (rows 2) );
    ( "shared rows and a bias",
      "shared by several lanes",
      fun () -> attend ~bias:(t [| 5 |]) (lanes ()) (rows 2) );
    ( "shared rows and a positive dropout",
      "shared by several lanes",
      fun () -> attend ~dropout:0.5 (lanes ()) (rows 2) );
  ]

(* merge *)

let test_merge_projects () =
  let l = Linear.make ~key:(Nx.Rng.key 42) ~inputs:6 ~outputs:4 Nx.float32 in
  let y = normal 43 Nx.float32 [| 2; 3; 5; 2 |] in
  let ya = flat y in
  (* Token s's features are its heads' in head order. *)
  let joined =
    Nx.create Nx.float32 [| 2; 5; 6 |]
      (Array.init 60 (fun i ->
           let f = i mod 6 and s = i / 6 mod 5 and b = i / 30 in
           ya.((((((b * 3) + (f / 2)) * 5) + s) * 2) + (f mod 2))))
  in
  equal exactly (flat (Linear.apply l joined)) (flat (Attention.merge l y))

let test_merge_undoes_split head_dim =
  let eye = { Linear.w = Nx.eye Nx.float32 4; b = None } in
  let x = normal 44 Nx.float32 [| 2; 3; 4 |] in
  equal exactly (flat x)
    (flat (Attention.merge eye (Attention.split ~head_dim eye x)))

let merge_refusals =
  let l () = Linear.make ~key:(Nx.Rng.key 45) ~inputs:6 ~outputs:4 Nx.float32 in
  [
    ( "y does not have four axes",
      "y must have shape [batch; heads; seq; d]",
      fun () -> ignore (Attention.merge (l ()) (Nx.zeros Nx.float32 [| 2; 6 |]))
    );
    ( "heads * d is not the input width",
      "is 4 in a and 6 in b",
      fun () ->
        ignore (Attention.merge (l ()) (Nx.zeros Nx.float32 [| 1; 2; 3; 2 |]))
    );
  ]

(* scaled_dot_product_attention *)

(* The core against its definition in float64, within the bound nx.mli states
   for Nx.attention, over drawn dtypes, grouped heads, masks, scales and sinks,
   by attend over grouped heads and by scaled_dot_product_attention over
   repeated ones. Inputs are drawn in [-1, 1], queries and keys times [amp],
   rounded to the dtype, and the reference reads the rounded values. *)

type drawn_mask = Unmasked | Causal | Drawn

type case = {
  dtype : int;  (** float32, bfloat16, float16 or float64 *)
  batch : int;
  hk : int;
  groups : int;
  n : int;
  m : int;
  d : int;
  dv : int;
  mask : drawn_mask;
  sinks : float option;  (** drawn in [-s, s] *)
  scale : float option;
  amp : float;
  seed : float;
}

let pp_case ppf c =
  Format.fprintf ppf
    "%s batch %d hk %d groups %d n %d m %d d %d dv %d mask %s sinks %s scale \
     %s amp %g seed %g"
    dtype_names.(c.dtype) c.batch c.hk c.groups c.n c.m c.d c.dv
    (match c.mask with
    | Unmasked -> "none"
    | Causal -> "causal"
    | Drawn -> "drawn")
    (match c.sinks with Some s -> Printf.sprintf "±%g" s | None -> "none")
    (match c.scale with Some s -> string_of_float s | None -> "default")
    c.amp c.seed

let case_gen =
  let open Gen in
  let* dtype = int_range 0 3 in
  let* batch = frequency [ (1, constant 0); (8, int_range 1 2) ] in
  let* hk = frequency [ (1, constant 0); (6, int_range 1 2) ] in
  let* groups = int_range 1 3 in
  let* n = frequency [ (1, constant 0); (6, int_range 1 5) ] in
  let* m = frequency [ (1, constant 0); (6, int_range 1 7) ] in
  let* scale = frequency [ (2, constant None); (1, constant (Some 2.5)) ] in
  (* Keys of no feature need a scale, and give a row uniform weights. *)
  let* d =
    match scale with
    | None -> int_range 1 4
    | Some _ -> frequency [ (1, constant 0); (1, int_range 1 4) ]
  in
  let* dv = frequency [ (1, constant 0); (6, int_range 1 3) ] in
  let* mask = of_list [ Unmasked; Causal; Drawn ] in
  let* sinks = of_list [ None; Some 2. ] in
  let+ seed = float_range 0. 10. in
  { dtype; batch; hk; groups; n; m; d; dv; mask; sinks; scale; amp = 1.; seed }

(* Inputs no draw reaches, or not on every run: float16 scores past its largest
   finite 65504 (raw q kᵀ from 120000 to 280000, scaled by 1 / sqrt 8), which
   compute in the float32 accumulator; bfloat16 at the same scores; sinks to
   ±200, far above and below the scores; keys of no feature, whose scores are
   all 0. *)
let core_examples =
  let large =
    {
      dtype = 2;
      batch = 1;
      hk = 1;
      groups = 1;
      n = 1;
      m = 3;
      d = 8;
      dv = 2;
      mask = Unmasked;
      sinks = None;
      scale = None;
      amp = 256.;
      seed = 0.;
    }
  in
  [
    large;
    { large with dtype = 1 };
    {
      dtype = 0;
      batch = 2;
      hk = 2;
      groups = 2;
      n = 3;
      m = 5;
      d = 4;
      dv = 3;
      mask = Drawn;
      sinks = Some 200.;
      scale = None;
      amp = 1.;
      seed = 0.;
    };
    {
      dtype = 0;
      batch = 2;
      hk = 2;
      groups = 2;
      n = 3;
      m = 4;
      d = 0;
      dv = 2;
      mask = Drawn;
      sinks = Some 2.;
      scale = Some 2.5;
      amp = 1.;
      seed = 1.;
    };
  ]

let drawn c offset shape =
  Nx.create Nx.float32 shape
    (Array.init (numel shape) (fun i ->
         Float.sin ((Float.of_int i *. 2.3) +. c.seed +. offset)))

(* Which keys a lane's row sees under [c]'s mask, as an array of the mask's
   shape [batch; 1; n; m] and as a function of lane, row and key. *)
let shown c = Nx.less_s (drawn c 5. [| c.batch; 1; c.n; c.m |]) 0.4

let sees c =
  match c.mask with
  | Unmasked -> fun _ _ _ -> true
  | Causal -> fun _ i j -> j <= i + (c.m - c.n)
  | Drawn ->
      let a = Nx.to_array (shown c) in
      fun bi i j -> a.((((bi * c.n) + i) * c.m) + j)

let u32 = 0x1p-24
let gamma n u = Float.of_int n *. u /. (1. -. (Float.of_int n *. u))

(* nx_cpu.mli's float32 [ε] for a row of [t] keys in [n = ⌈t / 64⌉] blocks:
   [e^λu (11n + 11 + ln t) u' + (e^λu - 1) ln t + u' |lse|], [u' = u / (1 - u)]
   and [λ = 3], its log's error in float32. *)
let cpu_epsilon t lse =
  let u' = u32 /. (1. -. u32) and ln = Float.log (Float.of_int t) in
  let grow = Float.exp (3. *. u32) in
  let n = (t + 63) / 64 in
  (grow *. (Float.of_int ((11 * n) + 11) +. ln) *. u')
  +. ((grow -. 1.) *. ln)
  +. (u' *. Float.abs lse)

(* nx.mli's bound for a row of [tk] keys, [a] its greatest [|scale| Σ |q k|]
   over its visible keys, [ε] being nx.cpu's: [lse]'s, [η + ε], and the factor
   of [M] in [o]'s, [e^w - 1 + γ(tk + 1, 2u) e^w]. *)
let row_bound ~dk ~tk ~a ~lse =
  let u = u32 in
  let eta =
    ((((1. +. u) ** 2.) *. (1. +. gamma (dk + 1) (2. *. u))) -. 1.) *. a
  in
  let ln = Float.log (Float.of_int (max 1 tk)) in
  let eps = cpu_epsilon (max 1 tk) lse in
  let w =
    ((1. +. u) *. ((2. *. eta) +. eps)) +. (((2. *. a) +. ln +. 5.) *. u)
  in
  (eta +. eps, Float.expm1 w +. (gamma (tk + 1) (2. *. u) *. Float.exp w))

(* One row's visible keys as [(j, score, |scale| Σ |q k|)], from the arrays [q]
   and [k] of a row and of its keys, [d] wide. *)
let scores ~scale ~d ~q ~k ~qi ~kj keys =
  List.map
    (fun j ->
      let s = ref 0. and a = ref 0. in
      for e = 0 to d - 1 do
        let x = q.((qi * d) + e) *. k.(((kj + j) * d) + e) in
        s := !s +. x;
        a := !a +. Float.abs x
      done;
      (j, scale *. !s, Float.abs scale *. !a))
    keys

(* [c]'s queries, keys, values and sinks rounded to [dt], and its mask. *)
let inputs (type b) c (dt : (float, b) Nx.dtype) =
  let heads = c.hk * c.groups in
  let b = c.batch and n = c.n and m = c.m and d = c.d and dv = c.dv in
  let round x = Nx.cast dt x in
  let q = round (Nx.mul_s (drawn c 1. [| b; heads; n; d |]) c.amp) in
  let k = round (Nx.mul_s (drawn c 2. [| b; c.hk; m; d |]) c.amp) in
  let v = round (drawn c 3. [| b; c.hk; m; dv |]) in
  let sinks =
    Option.map (fun s -> round (Nx.mul_s (drawn c 4. [| heads |]) s)) c.sinks
  in
  let mask =
    match c.mask with
    | Unmasked -> None
    | Causal -> Some Nx.causal
    | Drawn -> Some (captured (shown c))
  in
  (q, k, v, sinks, mask)

(* The core's result for [c] within the bound of its definition; under a causal
   mask, a last row decoded alone is within the bound of its prefill. [uq] is
   the dtype's unit roundoff, also [v]'s, for the weights the value product may
   round, and [tiny] half its subnormal spacing: the bound bars underflow, and a
   result of that size rounds to it. Below float16's [2^-14] a weight may lose
   [2^-25 |v|]. *)
let check_case (type b) c (dt : (float, b) Nx.dtype) ~uq ~tiny ~tiny_weight =
  let heads = c.hk * c.groups in
  let b = c.batch and n = c.n and m = c.m and d = c.d and dv = c.dv in
  let q, k, v, sinks, mask = inputs c dt in
  let sees = sees c in
  let scale =
    Option.value c.scale ~default:(1. /. Float.sqrt (Float.of_int d))
  in
  let read t = Nx.to_array (f64 t) in
  let qa = read q and ka = read k and va = read v in
  let sa = Option.map read sinks in
  (* The reference and each element's tolerance, [b; heads; n; dv]. A row that
     sees no key is held to exactly 0. *)
  let expected = Array.make (b * heads * n * dv) 0. in
  let tol = Array.make (b * heads * n * dv) 0. in
  for bi = 0 to b - 1 do
    for h = 0 to heads - 1 do
      let kj = ((bi * c.hk) + (h / c.groups)) * m in
      for i = 0 to n - 1 do
        let qi = (((bi * heads) + h) * n) + i in
        let keys = List.filter (sees bi i) (List.init m Fun.id) in
        let at = scores ~scale ~d ~q:qa ~k:ka ~qi ~kj keys in
        let top =
          List.fold_left
            (fun t (_, s, _) -> Float.max t s)
            Float.neg_infinity at
        in
        let sum =
          List.fold_left (fun t (_, s, _) -> t +. exp (s -. top)) 0. at
        in
        let lse = top +. log sum in
        let a = List.fold_left (fun t (_, _, a) -> Float.max t a) 0. at in
        let lse_tol, factor = row_bound ~dk:d ~tk:m ~a ~lse in
        for e = 0 to dv - 1 do
          let value j = va.(((kj + j) * dv) + e) in
          let o = ref 0. and mabs = ref 0. and lost = ref 0. in
          List.iter
            (fun (j, s, _) ->
              let p = exp (s -. lse) in
              o := !o +. (p *. value j);
              mabs := !mabs +. (p *. Float.abs (value j));
              if p < tiny_weight then
                lost := !lost +. (0x1p-25 *. Float.abs (value j)))
            at;
          let o_tol =
            (uq *. Float.abs !o)
            +. ((1. +. uq) *. factor *. !mabs)
            +. (uq *. !mabs) +. !lost +. tiny
          in
          let x = (qi * dv) + e in
          match (at, sa) with
          | [], _ -> ()
          | _, None ->
              expected.(x) <- !o;
              tol.(x) <- o_tol
          | _, Some sa ->
              (* The definition: one softmax over the keys and the sink, whose
                 column is dropped. *)
              let sink = sa.(h) in
              let top' = Float.max top sink in
              let sum' = (sum *. exp (top -. top')) +. exp (sink -. top') in
              let o' =
                List.fold_left
                  (fun t (j, s, _) -> t +. (exp (s -. top') /. sum' *. value j))
                  0. at
              in
              expected.(x) <- o';
              (* [o · sigmoid (lse - sink)]: [lse]'s error and the subtraction's
                 rounding move the sigmoid by at most themselves relatively, the
                 sigmoid adds 3 ulps, [6u], and the product rounds in float32,
                 then to the dtype. *)
              let kept = 1. /. (1. +. exp (sink -. lse)) in
              let moved = lse_tol +. (u32 *. Float.abs (lse -. sink)) in
              tol.(x) <-
                (kept *. o_tol)
                +. (Float.abs !o *. kept *. (moved +. (6. *. u32)))
                +. ((u32 +. uq) *. Float.abs o')
                +. tiny
        done
      done
    done
  done;
  (* First order: the products of two errors stay below [4 uq] of the sum. *)
  let tol = Array.map (fun t -> t *. (1. +. (4. *. uq))) tol in
  (* The result has the shape of the reference, and each element lies within its
     bound, which no NaN does. *)
  let within ~msg ?(rows = n) ?(row = fun i -> i) got =
    equal ~msg (array int) [| b; heads; rows; dv |] (Nx.shape got);
    Array.iteri
      (fun i y ->
        let r = row i in
        let claim =
          Printf.sprintf "element %d within %g of %g" r tol.(r) expected.(r)
        in
        satisfies ~msg ~claim float_exact
          (fun y -> Float.abs (y -. expected.(r)) <= tol.(r))
          y)
      (read got)
  in
  let attend q =
    Attention.attend ?mask ?scale:c.scale ?sinks q (Nx.Dense { k; v })
  in
  within ~msg:"attend" (attend q);
  let groups = Array.init heads (fun h -> h / c.groups) in
  let repeat t = Nx.take ~axis:1 (int64s [| heads |] groups) t in
  within ~msg:"scaled_dot_product_attention"
    (Attention.scaled_dot_product_attention ?mask ?scale:c.scale
       ?sinks:(Option.map (Nx.reshape [| heads; 1 |]) sinks)
       q (repeat k) (repeat v));
  if c.mask = Causal && n > 0 then
    (* Element [r] of the decoded row is element [bh, n - 1, e] of the prefill,
       whose definition it is held to. *)
    within ~msg:"a last row decoded alone" ~rows:1
      ~row:(fun r -> (((r / dv * n) + n - 1) * dv) + (r mod dv))
      (attend (Nx.slice [ A; A; R (n - 1, n) ] q))

(* At float64 the core is Nx.attention in float64, and its sinks [o · sigmoid
   (lse - σ)] in float64, bit for bit: nx's laws hold that operation to its
   bound. *)
let check_float64 c =
  let q, k, v, sinks, mask = inputs c Nx.float64 in
  let o, lse =
    Nx.attention ?scale:c.scale ?mask ~acc:Nx.float64 ~kv:(Nx.Dense { k; v }) q
  in
  let expected =
    match sinks with
    | None -> o
    | Some s ->
        let kept = Nx.sigmoid (Nx.sub lse (Nx.reshape [| Nx.dim 0 s; 1 |] s)) in
        Nx.mul o (Nx.unsqueeze ~axes:[ -1 ] kept)
  in
  let bits t = Array.map Int64.bits_of_float (Nx.to_array t) in
  equal ~msg:"attend at float64" (array int64) (bits expected)
    (bits (Attention.attend ?mask ?scale:c.scale ?sinks q (Nx.Dense { k; v })))

let law_core c =
  let heads = c.hk * c.groups in
  let sees = sees c in
  let empty_row =
    heads > 0
    && List.exists
         (fun bi ->
           List.exists
             (fun i -> not (List.exists (sees bi i) (List.init c.m Fun.id)))
             (List.init c.n Fun.id))
         (List.init c.batch Fun.id)
  in
  (* A zero-size result compares no value, so every other cover needs one. *)
  let values = c.batch * heads * c.n * c.dv > 0 in
  cover "zero-size" ((not values) || c.m = 0);
  cover "float16" (values && c.dtype = 2);
  cover "bfloat16" (values && c.dtype = 1);
  cover "float64" (values && c.dtype = 3);
  cover "grouped heads" (values && c.groups > 1);
  cover "sinks" (values && c.sinks <> None);
  cover "sinks at half precision"
    (values && c.sinks <> None && (c.dtype = 1 || c.dtype = 2));
  cover "a row that sees no key" (values && empty_row);
  cover "keys of no feature" (values && c.m > 0 && c.d = 0);
  cover "a causal row decoded alone" (values && c.mask = Causal && c.dtype <> 3);
  match c.dtype with
  | 0 -> check_case c Nx.float32 ~uq:u32 ~tiny:0x1p-150 ~tiny_weight:0.
  | 1 -> check_case c Nx.bfloat16 ~uq:0x1p-8 ~tiny:0x1p-134 ~tiny_weight:0.
  | 2 -> check_case c Nx.float16 ~uq:0x1p-11 ~tiny:0x1p-25 ~tiny_weight:0x1p-14
  | _ -> check_float64 c

(* d = 1, so the scale is 1. Scores are [0; ln 4], weights softmax of that:
   [1/5; 4/5], and the output 0.2 * 1 + 0.8 * 6 = 5. And d = 4: raw scores are
   [4 ln 2; 0], scaled by 1/sqrt 4 to [ln 4; 0], so the weights are [4/5; 1/5]
   and the output 0.8 * 1 + 0.2 * 6 = 2; without the scale the weights would be
   [16/17; 1/17]. In float64 the few roundings stay below 1e-14 of 5. *)
let analytic =
  let c = log 2. in
  [
    ("d = 1", ([| 1. |], [| 0.; log 4. |], 5.));
    ("d = 4", ([| 1.; 1.; 1.; 1. |], [| c; c; c; c; 0.; 0.; 0.; 0. |], 2.));
  ]

let test_core_analytic (_, (q, k, expected)) =
  let d = Array.length q in
  let q = Nx.create Nx.float64 [| 1; d |] q in
  let k = Nx.create Nx.float64 [| 2; d |] k in
  let v = Nx.create Nx.float64 [| 2; 1 |] [| 1.; 6. |] in
  equal
    (array (float 1e-14))
    [| expected |]
    (flat (Attention.scaled_dot_product_attention q k v))

let test_core_batch_broadcast () =
  let q = normal 46 Nx.float32 [| 2; 1; 3; 4 |] in
  let k = normal 47 Nx.float32 [| 1; 3; 5; 4 |] in
  let v = normal 48 Nx.float32 [| 1; 3; 5; 2 |] in
  let wide t =
    Nx.broadcast_to (Array.append [| 2; 3 |] (Array.sub (Nx.shape t) 2 2)) t
  in
  equal exactly
    (flat (Attention.scaled_dot_product_attention (wide q) (wide k) (wide v)))
    (flat (Attention.scaled_dot_product_attention q k v))

let test_core_hidden_weight () =
  (* Both keys tie, but the second is hidden: its weight is exactly 0, not
     merely small, so the large hidden value does not leak through. *)
  let q = Nx.create Nx.float32 [| 1; 1 |] [| 0. |] in
  let k = Nx.create Nx.float32 [| 2; 1 |] [| 0.; 0. |] in
  let v = Nx.create Nx.float32 [| 2; 1 |] [| 1.; 100. |] in
  let shown = Nx.create Nx.bool [| 1; 2 |] [| true; false |] in
  equal exactly [| 1.0 |]
    (flat (Attention.scaled_dot_product_attention ~mask:(captured shown) q k v))

(* Row 0 sees no key; row 1 sees keys 0 and 2. *)
let one_row_hidden =
  lazy
    (Nx.create Nx.bool [| 2; 3 |] [| false; false; false; true; false; true |])

let test_core_hidden_gradients sinked =
  let p =
    qkvs_of ~q:[| 2; 2; 2 |] ~k:[| 2; 3; 2 |] ~v:[| 2; 3; 2 |] ~sinks:[| 2; 1 |]
  in
  let mask = captured (Lazy.force one_row_hidden) in
  (* The hidden row's output alone, linearly, so any derivative shows. *)
  let loss { Qkvs.q; k; v; sinks } =
    let sinks = if sinked then Some sinks else None in
    let y = Attention.scaled_dot_product_attention ~mask ?sinks q k v in
    Nx.sum (Nx.slice [ A; I 0 ] y)
  in
  let g = Rune.grad qkvs loss p in
  all_zero ~msg:"q" (flat g.q);
  all_zero ~msg:"k" (flat g.k);
  all_zero ~msg:"v" (flat g.v);
  all_zero ~msg:"sinks" (flat g.sinks)

let sinked_id s = if s then "with sinks" else "without sinks"

type core_grad = {
  shapes : int array * int array * int array * int array;
  hidden : bool;
  sinked : bool;
  core_scale : float option;
}

let core_grads =
  [
    ( "no mask",
      {
        shapes = ([| 3; 2 |], [| 4; 2 |], [| 4; 2 |], [| 1 |]);
        hidden = false;
        sinked = false;
        core_scale = None;
      } );
    ( "a row that sees no key",
      {
        shapes = ([| 2; 2 |], [| 3; 2 |], [| 3; 2 |], [| 1 |]);
        hidden = true;
        sinked = false;
        core_scale = None;
      } );
    ( "sinks and a scale",
      {
        shapes = ([| 3; 4; 2 |], [| 3; 4; 2 |], [| 3; 4; 2 |], [| 3; 1 |]);
        hidden = false;
        sinked = true;
        core_scale = Some 0.9;
      } );
    ( "sinks and a row that sees no key",
      {
        shapes = ([| 2; 2; 2 |], [| 2; 3; 2 |], [| 2; 3; 2 |], [| 2; 1 |]);
        hidden = true;
        sinked = true;
        core_scale = None;
      } );
  ]

let test_core_gradients (_, c) =
  let q, k, v, sinks = c.shapes in
  let p = qkvs_of ~q ~k ~v ~sinks in
  let mask =
    if c.hidden then Some (captured (Lazy.force one_row_hidden)) else None
  in
  let loss { Qkvs.q; k; v; sinks } =
    let sinks = if c.sinked then Some sinks else None in
    let y =
      Attention.scaled_dot_product_attention ?mask ?scale:c.core_scale ?sinks q
        k v
    in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune.check_grads qkvs loss p)

(* Keys past the length are NaN, which reaches no row, eagerly or compiled. *)
let test_core_lengths_0d lead =
  Compiled.here ();
  let shape rows cols = Array.append lead [| rows; cols |] in
  let q = normal 49 Nx.float64 (shape 3 4) in
  let k = poison_past 2 (normal 50 Nx.float64 (shape 5 4)) in
  let v = poison_past 2 (normal 51 Nx.float64 (shape 5 4)) in
  let first2 t =
    Nx.slice (List.map (fun _ -> Nx.A) (Array.to_list lead) @ [ Nx.R (0, 2) ]) t
  in
  let expected =
    flat (Attention.scaled_dot_product_attention q (first2 k) (first2 v))
  in
  let f q =
    Attention.scaled_dot_product_attention ~lengths:(Nx.scalar Nx.int64 2L) q k
      v
  in
  equal ~msg:"eager" exactly expected (flat (f q));
  equal ~msg:"compiled" reordered expected (flat (Rune.jit' f q))

let test_core_lengths () =
  Compiled.here ();
  (* Batch axes [2; 2]: lengths of shape [2], one per index of the first, for
     both of its heads. Lane 0 holds 1 key, the rest NaN, lane 1 all 5. *)
  let q = normal 52 Nx.float64 [| 2; 2; 3; 4 |] in
  let keys seed =
    let t = normal seed Nx.float64 [| 2; 2; 5; 4 |] in
    Nx.concatenate ~axis:0
      [ poison_past 1 (Nx.slice [ R (0, 1) ] t); Nx.slice [ R (1, 2) ] t ]
  in
  let k = keys 53 and v = keys 54 in
  let lane i n =
    let keys t = Nx.slice [ R (i, i + 1); A; R (0, n) ] t in
    flat
      (Attention.scaled_dot_product_attention
         (Nx.slice [ R (i, i + 1) ] q)
         (keys k) (keys v))
  in
  let expected = Array.append (lane 0 1) (lane 1 5) in
  let f q =
    Attention.scaled_dot_product_attention
      ~lengths:(int64s [| 2 |] [| 1; 5 |])
      q k v
  in
  equal ~msg:"eager" exactly expected (flat (f q));
  equal ~msg:"compiled" reordered expected (flat (Rune.jit' f q))

let test_sinks_per_row () =
  let q = normal 55 Nx.float32 [| 2; 4; 3; 6 |] in
  let k = normal 56 Nx.float32 [| 2; 4; 5; 6 |] in
  let v = normal 57 Nx.float32 [| 2; 4; 5; 6 |] in
  let per_row = normal 58 Nx.float32 [| 2; 4; 3 |] in
  let head h =
    let at t = Nx.slice [ A; R (h, h + 1) ] t in
    Attention.scaled_dot_product_attention ~sinks:(at per_row) (at q) (at k)
      (at v)
  in
  equal exactly
    (flat (Nx.concatenate ~axis:1 (List.init 4 head)))
    (flat (Attention.scaled_dot_product_attention ~sinks:per_row q k v))

let test_sinks_compiled () =
  Compiled.here ();
  let q = normal 59 Nx.float64 [| 2; 4; 3; 6 |] in
  let k = normal 60 Nx.float64 [| 2; 4; 5; 6 |] in
  let v = normal 61 Nx.float64 [| 2; 4; 5; 6 |] in
  let sinks = Nx.create Nx.float64 [| 4; 1 |] [| -1.5; 0.25; 2.0; 0.75 |] in
  let shown = Nx.less_s (normal 62 Nx.float32 [| 3; 5 |]) 0.5 in
  let f q =
    Attention.scaled_dot_product_attention ~mask:(captured shown) ~scale:0.7
      ~sinks q k v
  in
  equal reordered (flat (f q)) (flat (Rune.jit' f q))

let core_refusals =
  let t shape = Nx.zeros Nx.float32 shape in
  let sdpa ?mask ?lengths ?sinks q k v =
    ignore (Attention.scaled_dot_product_attention ?mask ?lengths ?sinks q k v)
  in
  [
    ( "q is a float4 format",
      "float4_e2m1fn is not",
      fun () ->
        let x = Nx.zeros Nx.float4_e2m1fn [| 1; 2 |] in
        ignore (Attention.scaled_dot_product_attention x x x) );
    ( "q has fewer than 2 axes",
      "at least 2 axes",
      fun () -> sdpa (t [| 3 |]) (t [| 3; 2 |]) (t [| 3; 2 |]) );
    ( "q and k differ in their last axis",
      "q has 2 features but k has 3",
      fun () -> sdpa (t [| 1; 2 |]) (t [| 4; 3 |]) (t [| 4; 2 |]) );
    ( "k and v differ in their second-to-last axis",
      "k has 2 positions but v has 3",
      fun () -> sdpa (t [| 1; 2 |]) (t [| 2; 2 |]) (t [| 3; 2 |]) );
    ( "the batch axes do not broadcast",
      "do not broadcast",
      fun () -> sdpa (t [| 2; 1; 2 |]) (t [| 3; 4; 2 |]) (t [| 3; 4; 2 |]) );
    ( "d is 0 without scale",
      "no feature need a scale",
      fun () -> sdpa (t [| 1; 0 |]) (t [| 2; 0 |]) (t [| 2; 2 |]) );
    ( "sinks do not broadcast",
      "sinks of shape [3; 1] do not broadcast",
      fun () ->
        sdpa
          ~sinks:(t [| 3; 1 |])
          (t [| 4; 2; 2 |])
          (t [| 4; 2; 2 |])
          (t [| 4; 2; 2 |]) );
    ( "lengths are not of the batch axes without their last",
      "lengths must have shape [2]",
      fun () ->
        sdpa
          ~lengths:(Nx.zeros Nx.int64 [| 3 |])
          (t [| 2; 4; 1; 2 |])
          (t [| 2; 4; 3; 2 |])
          (t [| 2; 4; 3; 2 |]) );
    ( "the mask does not broadcast",
      "does not broadcast",
      fun () ->
        sdpa
          ~mask:(captured (Nx.ones Nx.bool [| 3; 3 |]))
          (t [| 2; 2 |])
          (t [| 2; 2 |])
          (t [| 2; 2 |]) );
  ]

(* walk *)

let visits p =
  List.map
    (Format.asprintf "%a" Nx.Ptree.pp_visit)
    (Nx.Ptree.visits (Nx.Ptree.instantiate (module Attention)) p)

let test_walk bias =
  let p = Attention.make ~bias ~key:(Nx.Rng.key 63) ~embed_dim:4 Nx.float32 in
  let proj name =
    if bias then [ name ^ ".w: a leaf"; name ^ ".b: Some"; name ^ ".b: a leaf" ]
    else [ name ^ ".w: a leaf"; name ^ ".b: None" ]
  in
  equal (list string) (List.concat_map proj [ "q"; "k"; "v"; "out" ]) (visits p)

let bias_id b = if b then "with biases" else "without biases"

type cast_to = Cast : string * (float, 'b) Nx.dtype -> cast_to

let cast_id (Cast (name, _)) = name

let test_cast (Cast (name, dt)) =
  let p = layer 64 ~embed_dim:4 Nx.float32 in
  let expected =
    Nx.Ptree.fold
      (Nx.Ptree.instantiate (module Attention))
      (fun path t acc ->
        (Nx.Ptree.Path.to_string path, name, flat (f64 (Nx.cast dt t))) :: acc)
      p []
  in
  let got =
    Nx.Ptree.fold
      (Nx.Ptree.instantiate (module Attention))
      (fun path t acc ->
        (Nx.Ptree.Path.to_string path, Nx.Dtype.name (Nx.dtype t), flat (f64 t))
        :: acc)
      (Nx.Ptree.cast (module Attention) dt p)
      []
  in
  equal (list (triple string string exactly)) (List.rev expected) (List.rev got)

let refusal_id (id, _, _) = id
let test_refusal (_, sub, f) = raises_match (Exn.invalid_arg ~sub) f

let () =
  exit
    (run "kaun attention"
       [
         group "make"
           [
             cases "q maps embed_dim to q_dim, k and v to kv_dim, out back"
               ~id:dims_id
               [ (None, None); (Some 8, Some 4); (Some 2, None) ]
               test_make_shapes;
             cases
               "its projections are Linear.make's, from four keys Nx.Rng.split \
                makes"
               ~id:projection_args_id
               [ Defaults; Initializers; No_bias ]
               test_make_projections;
             cases "raises on a dimension that is not positive"
               ~id:refused_dims_id
               [ (0, None, None); (6, Some 0, None); (6, None, Some (-1)) ]
               test_make_refuses;
           ];
         group "init"
           [
             test "is make at float32" test_init_is_make;
             test "draws Glorot-uniform weights and zero biases"
               test_init_glorot;
           ];
         group "apply"
           [
             cases "the result has x's shape" ~id:shape_id
               [
                 [| 5; 8 |];
                 [| 2; 5; 8 |];
                 [| 2; 3; 5; 8 |];
                 [| 0; 5; 8 |];
                 [| 2; 0; 8 |];
               ]
               test_apply_shape;
             prop ~count:200
               "is split, rope at positions 0 to seq - 1, attend and merge \
                over the batch axes folded"
               (Gen.with_pp pp_layer_case layer_gen)
               law_pieces;
             test "head h attends over its own head_dim features"
               test_heads_attend_apart;
             test "grouped keys equal repeated keys"
               test_grouped_equals_repeated;
             test "without a mask every token sees every token"
               test_apply_unmasked;
             test "under Nx.causal a token sees no later token"
               test_apply_causal;
             test
               "a query that sees no key yields the output projection of zero"
               test_apply_no_key;
             cases "lengths keep the padding from real tokens" ~id:lengths_id
               [
                 ([| 2 |], [| 2; 4 |], false);
                 ([| 2 |], [| 2; 4 |], true);
                 ([||], [| 2 |], false);
                 ([||], [| 2 |], true);
               ]
               test_lengths_hide_padding;
             test "lengths keep the padding from real tokens, compiled"
               test_lengths_compiled;
             test "Nx.causal aligns the last query with the last real key"
               test_causal_alignment;
             cases "is differentiable" ~id:grad_layer_id
               [ Plain; Causal_mask; Rotated; Grouped ]
               test_apply_gradients;
             cases
               "raises on a float8 or float4 x, a misshapen x, lengths or \
                mask, a head_dim that is not positive or does not divide the \
                widths, or key and value widths that differ"
               ~id:refusal_id apply_refusals test_refusal;
           ];
         group "split"
           [
             cases
               "head h of a token holds features h * head_dim to (h + 1) * \
                head_dim - 1 of its projection"
               ~id:(Printf.sprintf "head_dim %d")
               [ 1; 2; 3; 6 ] test_split;
             cases
               "raises on an x not of three axes or a head_dim that is not \
                positive or does not divide the width"
               ~id:refusal_id split_refusals test_refusal;
           ];
         group "attend"
           [
             cases
               "a lane over a shared row is the lane over a copy of it, \
                eagerly and compiled"
               ~id:shared_id
               [ (4, 1, 5); (6, 2, 5); (6, 3, 1); (4, 2, 0); (0, 2, 5) ]
               test_shared_rows;
             test "is differentiable over shared rows"
               test_shared_rows_gradients;
             prop ~count:200
               "a query over a row of its own is a lane of one query over it"
               (Gen.with_pp pp_own own_gen)
               law_own_rows;
             test "a compiled query over a row of its own is eager's"
               test_own_rows_compiled;
             test "is differentiable over rows of their own"
               test_own_rows_gradients;
             test "lengths bound the keys" test_attend_lengths;
             test "a float32 bias keeps its bits under bfloat16 queries"
               test_bias_keeps_its_bits;
             cases "dropout drops the weights by the mask of their shape"
               ~id:drop_case_id drop_cases test_dropout_values;
             cases "dropout's gradients are those of the dropped weights"
               ~id:drop_case_id drop_cases test_dropout_gradients;
             test "compiled dropout is eager dropout" test_dropout_compiled;
             cases "a rate of 0 is attention"
               ~id:(fun s ->
                 if s then "rows shared by lanes" else "a row per lane")
               [ false; true ] test_dropout_zero;
             test "paged keys drop as dense ones" test_dropout_paged;
             test "dropout rounds to q's dtype twice" test_dropout_rounds_twice;
             test "is differentiable through grouped heads and sinks"
               test_attend_gradients;
             cases
               "raises on a float8 or float4 q, a misshapen q, kv, sinks, \
                lengths, mask, bias or keys, rows that do not divide batch, a \
                dropout outside [0, 1), or shared rows told apart"
               ~id:refusal_id attend_refusals test_refusal;
           ];
         group "merge"
           [
             test "projects the heads, concatenated, through l"
               test_merge_projects;
             cases "undoes split"
               ~id:(Printf.sprintf "head_dim %d")
               [ 1; 2; 4 ] test_merge_undoes_split;
             cases
               "raises on a y not of four axes or heads * d other than the \
                input width"
               ~id:refusal_id merge_refusals test_refusal;
           ];
         group "scaled_dot_product_attention"
           [
             prop ~count:200 ~examples:core_examples
               "is within nx.mli's bound of the definition, as attend is"
               (Gen.with_pp pp_case case_gen)
               law_core;
             cases "is softmax (q kᵀ / sqrt d) v" ~id:fst analytic
               test_core_analytic;
             test "batch axes broadcast" test_core_batch_broadcast;
             test "a hidden key's weight is exactly 0" test_core_hidden_weight;
             cases "a row that sees no key has zero gradients" ~id:sinked_id
               [ false; true ] test_core_hidden_gradients;
             cases "is differentiable" ~id:fst core_grads test_core_gradients;
             cases "0-d lengths bound every row, eagerly and compiled"
               ~id:(fun l -> "batch axes " ^ shape_id l)
               [ [||]; [| 2 |] ] test_core_lengths_0d;
             test
               "lengths of the batch axes but the last bound their rows, \
                eagerly and compiled"
               test_core_lengths;
             test "sinks broadcast to the scores without their last axis"
               test_sinks_per_row;
             test "compiled with sinks is eager" test_sinks_compiled;
             cases
               "raises on a float8 or float4 q, fewer than 2 axes, unpaired \
                shapes, d = 0 without scale, or misshapen sinks, lengths or \
                mask"
               ~id:refusal_id core_refusals test_refusal;
           ];
         group "walk"
           [
             cases "visits q, k, v then out by Linear.walk" ~id:bias_id
               [ true; false ] test_walk;
             cases "Nx.Ptree.cast converts every leaf" ~id:cast_id
               [
                 Cast ("float16", Nx.float16);
                 Cast ("bfloat16", Nx.bfloat16);
                 Cast ("float64", Nx.float64);
               ]
               test_cast;
           ];
       ])
