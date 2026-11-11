(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'd f32 = 'd Nx.float32_t

type ('a, 'd) layer = {
  attn : ('a, 'd) Kaun.Attention.t;
  qk : ('d f32 * 'd f32) option;
  norm1 : ('d f32, 'd) Kaun.Layer_norm.t;
  ffn1 : ('a, 'd) Kaun.Linear.t;
  ffn2 : ('a, 'd) Kaun.Linear.t;
  norm2 : ('d f32, 'd) Kaun.Layer_norm.t;
}

type ('a, 'd) backbone = {
  embed : ('a, 'd) Kaun.Linear.t * ('a, 'd) Kaun.Linear.t;
  layers : ('a, 'd) layer list;
  pool : ('a, 'd) Kaun.Linear.t;
}

type ('a, 'd) reducer = {
  up : ('a, 'd) Kaun.Linear.t;
  norm : ('d f32, 'd) Kaun.Layer_norm.t;
  down : ('a, 'd) Kaun.Linear.t;
}

type ('a, 'd) net =
  | Post_ln of {
      s2 : ('a, 'd) backbone;
      s1 : ('a, 'd) backbone;
      reducer : ('a, 'd) reducer;
    }
  | Pre_ln of {
      s2 : ('a, 'd) backbone;
      s1 : ('a, 'd) backbone;
      modality : 'd f32;
      fusion : ('a, 'd) layer list;
      reducer : ('a, 'd) reducer;
    }

type 'd norm = {
  s2_mean : 'd Nx.float32_t;
  s2_std : 'd Nx.float32_t;
  s1_mean : 'd Nx.float32_t;
  s1_std : 'd Nx.float32_t;
}

type ('a, 'd) t = {
  net : ('a, 'd) net;
  heads : int;
  norm : 'd norm;
  prefixes : int list;
}

(* Structure *)

(* Norms' parameters are fixed float32 arrays: a cast keeps them. *)
let norm c n = Nx.Ptree.Walk.(under Kaun.Layer_norm.walk tensor) c n

let walk_layer c l =
  let open Nx.Ptree.Walk in
  let attn = field c "attn" Kaun.Attention.walk l.attn in
  let pair c (q, k) =
    let q = index c 0 tensor q in
    let k = index c 1 tensor k in
    (q, k)
  in
  let qk = field c "qk" (option pair) l.qk in
  let norm1 = field c "norm1" norm l.norm1 in
  let ffn1 = field c "ffn1" Kaun.Linear.walk l.ffn1 in
  let ffn2 = field c "ffn2" Kaun.Linear.walk l.ffn2 in
  let norm2 = field c "norm2" norm l.norm2 in
  { attn; qk; norm1; ffn1; ffn2; norm2 }

let walk_backbone c b =
  let open Nx.Ptree.Walk in
  let e1, e2 = b.embed in
  let e1 = field c "embed1" Kaun.Linear.walk e1 in
  let e2 = field c "embed2" Kaun.Linear.walk e2 in
  let layers = field c "layers" (list walk_layer) b.layers in
  let pool = field c "pool" Kaun.Linear.walk b.pool in
  { embed = (e1, e2); layers; pool }

let walk_reducer c r =
  let open Nx.Ptree.Walk in
  let up = field c "up" Kaun.Linear.walk r.up in
  let norm = field c "norm" norm r.norm in
  let down = field c "down" Kaun.Linear.walk r.down in
  { up; norm; down }

let walk_net c =
  let open Nx.Ptree.Walk in
  function
  | Post_ln { s2; s1; reducer } ->
      case c "post_ln";
      let s2 = field c "s2" walk_backbone s2 in
      let s1 = field c "s1" walk_backbone s1 in
      let reducer = field c "reducer" walk_reducer reducer in
      Post_ln { s2; s1; reducer }
  | Pre_ln { s2; s1; modality; fusion; reducer } ->
      case c "pre_ln";
      let s2 = field c "s2" walk_backbone s2 in
      let s1 = field c "s1" walk_backbone s1 in
      let modality = field c "modality" tensor modality in
      let fusion = field c "fusion" (list walk_layer) fusion in
      let reducer = field c "reducer" walk_reducer reducer in
      Pre_ln { s2; s1; modality; fusion; reducer }

let walk_norm c n =
  let open Nx.Ptree.Walk in
  let s2_mean = field c "s2_mean" tensor n.s2_mean in
  let s2_std = field c "s2_std" tensor n.s2_std in
  let s1_mean = field c "s1_mean" tensor n.s1_mean in
  let s1_std = field c "s1_std" tensor n.s1_std in
  { s2_mean; s2_std; s1_mean; s1_std }

module Net = struct
  type nonrec ('a, 'd) t = ('a, 'd) net

  let walk = walk_net
end

let walk c m =
  let open Nx.Ptree.Walk in
  let net = field c "net" walk_net m.net in
  let heads = field c "heads" int m.heads in
  let norm = field c "norm" walk_norm m.norm in
  { net; heads; norm; prefixes = m.prefixes }

let reducer_of = function
  | Post_ln { reducer; _ } | Pre_ln { reducer; _ } -> reducer

let dims m = Nx.dim 1 (reducer_of m.net).down.w

(* Statistics *)

(* Standardisation constants of the released checkpoints, bound to them by
   training (tessera_infer_v2 student/model.py:23-36, teacher/model.py:21-35).
   The students hold the ascending and descending orbits apart; the teacher
   pools them. Bands in the order B04 B02 B03 B08 B8A B05 B06 B07 B11 B12, then
   VV and VH. *)
let s2_mean =
  [|
    1633.0042;
    1341.1090;
    1539.5536;
    3054.8269;
    3117.4658;
    2004.1648;
    2694.7275;
    2945.1504;
    2266.6079;
    1657.3094;
  |]

let s2_std =
  [|
    1999.4603;
    2014.7549;
    1929.2201;
    1754.2493;
    1649.9807;
    1936.8988;
    1748.6041;
    1708.6991;
    1207.5250;
    1108.6046;
  |]

let s1a_mean = [| 5909.3921; 3405.0322 |]
let s1a_std = [| 1507.1750; 1531.2615 |]
let s1d_mean = [| 5816.1382; 3277.7576 |]
let s1d_std = [| 1554.6475; 1546.4733 |]
let s1_pooled_mean = [| 5862.7652; 3341.3949 |]
let s1_pooled_std = [| 1531.8051; 1540.2014 |]

let stats at ~s1_mean:(ma, md) ~s1_std:(sa, sd) =
  let v shape a = Nx.place at (Nx.create Nx.float32 shape a) in
  {
    s2_mean = v [| 10 |] s2_mean;
    s2_std = v [| 10 |] s2_std;
    s1_mean = v [| 2; 2 |] (Array.append ma md);
    s1_std = v [| 2; 2 |] (Array.append sa sd);
  }

let student_stats at =
  stats at ~s1_mean:(s1a_mean, s1d_mean) ~s1_std:(s1a_std, s1d_std)

let teacher_stats at =
  stats at
    ~s1_mean:(s1_pooled_mean, s1_pooled_mean)
    ~s1_std:(s1_pooled_std, s1_pooled_std)

(* Applying *)

let f32 = Nx.float32
let eps = 1e-5

(* float32's machine epsilon, [nn.RMSNorm]'s eps for float32 inputs. *)
let qk_eps = Float.ldexp 1. (-23)
let product = Product.apply

(* A linear layer at its parameters' dtype, on and to float32. *)
let linear p x = Nx.cast f32 (product p x)

let layer_norm (p : ('d f32, 'd) Kaun.Layer_norm.t) x =
  Nx.standardize ~eps ~scale:p.gamma ~shift:p.beta x

let relu x = Nx.maximum x (Nx.scalar_like x 0.)

(* Dropout *)

type dropout = { layers : float; reducer : float }
type 'd record = string -> (unit -> 'd Nx.float32_t) -> unit

let v2_dropout = { layers = 0.1; reducer = 0.2 }
let no_dropout = { layers = 0.; reducer = 0. }

(* One call's dropout: its rates, each pixel place's key, and where its masks
   are reported. *)
type 'd drop = {
  rates : dropout;
  keys : 'd Nx.Rng.key;
  mask : string -> (unit -> 'd Nx.bool_t) -> unit;
}

(* A mask's site: the sensor (0 for Sentinel-2, 1 for Sentinel-1, 2 for the
   fusion), the layer and the layer's site, or the reducer alone. *)
let attn_p = 0
let attn_out = 1
let ffn_hidden = 2
let ffn_out = 3
let fusion = 2
let reducer_site = 3
let site keys path = List.fold_left Nx.Rng.fold_in keys path

(* Each token's key, [[r; t; 2]]: the key of its pixel's place [place] ([[r;
   t]]), indexed by its [index] in its slot. *)
let token_keys keys ~place ~index =
  let r = Nx.dim 0 place and t = Nx.dim 1 place in
  let words =
    Nx.take ~axis:0 (Nx.reshape [| r * t |] place) (Nx.Rng.to_tensor keys)
  in
  Nx.Rng.fold_in_tensor
    (Nx.Rng.of_tensor (Nx.reshape [| r; t; 2 |] words))
    index

(* Keeps [x]'s elements where [m] holds, scaled by [1 / (1 − rate)]. *)
let keep ~rate m x =
  Nx.where m (Nx.mul_s x (1. /. (1. -. rate))) (Nx.scalar_like x 0.)

(* A mask over [x], one key of [keys] per element of [x]'s leading axes, shaped
   as they are followed by ones. *)
let draw ~rate keys x =
  let p =
    Nx.broadcast_to (Nx.shape x) (Nx.cast f32 (Nx.scalar_like x (1. -. rate)))
  in
  Nx.Rng.bernoulli ~key:keys p

(* A token site's mask over [x], [[r; t; c]], from token keys [[r; t; 2]]. *)
let token_mask ~rate k x =
  let r = Nx.dim 0 x and t = Nx.dim 1 x in
  draw ~rate
    (Nx.Rng.of_tensor (Nx.reshape [| r; t; 1; 2 |] (Nx.Rng.to_tensor k)))
    x

(* The attention probabilities' mask, [[r; h; t; t]], one key per pair of tokens
   drawing each head: the query's key indexed by the key's [index]. *)
let pair_mask ~rate k ~index p =
  let r = Nx.dim 0 p and t = Nx.dim 2 p in
  let q = Nx.Rng.of_tensor (Nx.reshape [| r; t; 1; 2 |] (Nx.Rng.to_tensor k)) in
  let pairs = Nx.Rng.fold_in_tensor q (Nx.reshape [| r; 1; t |] index) in
  draw ~rate
    (Nx.Rng.of_tensor (Nx.reshape [| r; 1; t; t; 2 |] (Nx.Rng.to_tensor pairs)))
    p

(* The tokens a layer reads: the bounds of each query's keys, [[r; t]], each
   token's pixel place and index in its slot, and where its masks come from. *)
type 'd tokens = {
  start : 'd Nx.int32_t;
  stop : 'd Nx.int32_t;
  bias : 'd f32; (* [[r; t]]: each key's [log count]. *)
  place : 'd Nx.int64_t;
  index : 'd Nx.int64_t;
  drop : ('d drop * int) option; (* With the sensor's site. *)
}

(* Layer [i]'s dropout: its rate, each site's token keys, and where its masks
   are reported. *)
type 'd sites = {
  rate : float;
  keys : int -> 'd Nx.Rng.key;
  report : string -> (unit -> 'd Nx.bool_t) -> unit;
}

let sites tk i =
  match tk.drop with
  | Some (d, sensor) when d.rates.layers > 0. ->
      let keys k =
        token_keys
          (site d.keys [ sensor; i; k ])
          ~place:tk.place ~index:tk.index
      in
      let report n m = d.mask (Printf.sprintf "layer%d_%s" i n) m in
      Some { rate = d.rates.layers; keys; report }
  | _ -> None

let masked ds ~name k x =
  match ds with
  | None -> x
  | Some s ->
      let m = token_mask ~rate:s.rate (s.keys k) x in
      s.report name (fun () -> m);
      keep ~rate:s.rate m x

(* Keys [k] with [start_q ≤ k < stop_q], the bounds [[rows; 1; queries; 1]]. *)
let within start stop =
  let n = Nx.dim 0 start and q = Nx.dim 1 start in
  let start = Nx.reshape [| n; 1; q; 1 |] start
  and stop = Nx.reshape [| n; 1; q; 1 |] stop in
  fun (p : 'd Nx.pair) ->
    let k = Nx.cast Nx.int32 p.k in
    Nx.logical_and (Nx.greater_equal k start) (Nx.less k stop)

let key_bias b =
  let n = Nx.dim 0 b and k = Nx.dim 1 b in
  let b = Nx.reshape [| n; 1; 1; k |] b in
  fun _ s -> Nx.add s b

let scores = Nx.Pattern.v "r h q d, r h k d -> r h q k | d"
let mix = Nx.Pattern.v "r h q k, r h k d -> r h q d | k"

(* Attention with its probabilities dropped, composed: float32 scores and
   softmax over each query's visible keys, the kept probabilities rounded to
   [v]'s dtype for the value product. *)
let dense (tk : 'd tokens) s q k v =
  let r = Nx.dim 0 q and t = Nx.dim 2 q in
  let dt = Nx.dtype v in
  let scale = 1. /. Float.sqrt (Float.of_int (Nx.dim (-1) q)) in
  let x = Nx.mul_s (Nx.contract f32 scores q k) scale in
  let x = Nx.add x (Nx.reshape [| r; 1; 1; t |] tk.bias) in
  let col =
    Nx.reshape [| 1; 1; 1; t |]
      (Nx.arange_at (Nx.beside tk.start) Nx.int32 0 t 1)
  in
  let bound b = Nx.reshape [| r; 1; t; 1 |] b in
  let visible =
    Nx.logical_and
      (Nx.greater_equal col (bound tk.start))
      (Nx.less col (bound tk.stop))
  in
  let p = Nx.softmax ~where:visible x in
  let m = pair_mask ~rate:s.rate (s.keys attn_p) ~index:tk.index p in
  s.report "attn_p" (fun () -> m);
  Nx.contract dt mix (Nx.cast dt (keep ~rate:s.rate m p)) v

let split ~heads (p : ((float, 's, 'd) Nx.t, 'd) Kaun.Linear.t) x =
  let b = Nx.dim 0 x and t = Nx.dim 1 x and d = Nx.dim 1 p.w in
  Nx.swapaxes 1 2 (Nx.reshape [| b; t; heads; d / heads |] (product p x))

let merge (p : ((float, 's, 'd) Nx.t, 'd) Kaun.Linear.t) o =
  let b = Nx.dim 0 o and h = Nx.dim 1 o and t = Nx.dim 2 o and d = Nx.dim 3 o in
  linear p (Nx.reshape [| b; t; h * d |] (Nx.swapaxes 1 2 o))

(* Self-attention of [x], [[r; t; d]], at the parameters' dtype. *)
let attention ~heads l (tk : 'd tokens) ds x =
  let p = l.attn in
  let dt = Nx.dtype p.q.w in
  let x = Nx.cast dt x in
  let q = split ~heads p.q x
  and k = split ~heads p.k x
  and v = split ~heads p.v x in
  let q, k =
    match l.qk with
    | None -> (q, k)
    | Some (gq, gk) ->
        let norm g x =
          Nx.cast dt (Nx.rms_norm ~eps:qk_eps ~scale:g (Nx.cast f32 x))
        in
        (norm gq q, norm gk k)
  in
  let o =
    match ds with
    | Some s -> dense tk s q k v
    | None ->
        let mask = within tk.start tk.stop and score = key_bias tk.bias in
        fst (Nx.attention ~mask ~score ~acc:f32 ~kv:(Nx.Dense { k; v }) q)
  in
  merge p.out o

let ffn l ds ~record x =
  let h = linear l.ffn1 x in
  record "linear1" (fun () -> h);
  let h = masked ds ~name:"ffn_hidden" ffn_hidden (relu h) in
  masked ds ~name:"ffn" ffn_out (linear l.ffn2 h)

let layer ~pre_ln ~heads tk ~record i x l =
  let ds = sites tk i in
  let record name v = record (Printf.sprintf "layer%d_%s" i name) v in
  let attend x = masked ds ~name:"attn" attn_out (attention ~heads l tk ds x) in
  if pre_ln then begin
    let x = Nx.add x (attend (layer_norm l.norm1 x)) in
    Nx.add x (ffn l ds ~record (layer_norm l.norm2 x))
  end
  else begin
    let a = attend x in
    record "attn" (fun () -> a);
    let x = layer_norm l.norm1 (Nx.add x a) in
    record "norm1" (fun () -> x);
    layer_norm l.norm2 (Nx.add x (ffn l ds ~record x))
  end

(* A stack of layers, each computed again in the backward pass with [remat]. *)
let layers ~pre_ln ~heads ~remat tk ~record x ls =
  List.fold_left
    (fun (i, x) l ->
      let f x = layer ~pre_ln ~heads tk ~record i x l in
      let x =
        if remat then Rune.remat Nx.Ptree.(tensor @-> returns tensor) f x
        else f x
      in
      record (Printf.sprintf "layer%d" i) (fun () -> x);
      (i + 1, x))
    (0, x) ls
  |> snd

(* The sinusoid's base and standardisation's guard (tessera_infer_v2
   student/model.py:47-63, infer.py:161). *)
let wavelength = 10000.
let std_guard = 1e-9

(* The sinusoid of [doy], [[rows; width]], over [d] channels. Its frequencies
   [exp (−2k ln wavelength / d)] are constants, computed on the host in float64
   and rounded once to float32: a day up to 366 multiplies a frequency's last
   bit, so the encoding depends on a device only through [sin] and [cos]. *)
let encoding d doy =
  let shape = Nx.shape doy in
  let rate = -.Float.log wavelength /. Float.of_int d in
  let freq k = Float.exp (Float.of_int (2 * k.(0)) *. rate) in
  let div = Nx.place (Nx.beside doy) (Nx.init f32 [| d / 2 |] freq) in
  let angle = Nx.mul (Nx.reshape (Array.append shape [| 1 |]) doy) div in
  let pe = Nx.stack ~axis:(-1) [ Nx.sin angle; Nx.cos angle ] in
  Nx.reshape (Array.append shape [| d |]) pe

(* A token's standardised bands, [(x − mean) / (std + std_guard)], zeros for a
   stand-in. [mean] and [std] broadcast to [bands]. *)
let standard ~source bands mean std =
  let z = Nx.div (Nx.sub bands mean) (Nx.add_s std std_guard) in
  Nx.where (Nx.less_s source 0.) (Nx.scalar_like z 0.) z

(* A token's channels ({!Batch.rows}): its bands, its day, its source. *)
let channel tokens c = Nx.slice [ Nx.A; Nx.A; Nx.R (c, c + 1) ] tokens
let bands_of tokens n = Nx.slice [ Nx.A; Nx.A; Nx.R (0, n) ] tokens
let source_of tokens n = channel tokens (n + 1)

let s2_inputs norm (r : 'd Batch.rows) =
  let n = Pixels.s2_bands in
  standard ~source:(source_of r.tokens n) (bands_of r.tokens n) norm.s2_mean
    norm.s2_std

let s1_inputs norm (r : 'd Batch.rows) =
  let n = Pixels.s1_bands in
  let bands = bands_of r.tokens n and source = source_of r.tokens n in
  let orbit o = Nx.get [ o ] in
  let z o =
    standard ~source bands (orbit o norm.s1_mean) (orbit o norm.s1_std)
  in
  Nx.where (Nx.equal_s source 1.) (z 1) (z 0)

(* One query per slot over the slot's tokens, [x] [[rows; width; d]], each score
   [w · x + b] at the products' dtype plus [log count], the softmax and the
   weighted sum in float32. *)
let pool (r : 'd Batch.rows) logits x =
  let rows = Nx.dim 0 x and width = Nx.dim 1 x and d = Nx.dim 2 x in
  let slots = Nx.dim 1 r.slot_start in
  let k =
    Nx.reshape [| 1; 1; width |]
      (Nx.arange_at (Nx.beside r.slot_start) Nx.int32 0 width 1)
  in
  let bound b = Nx.reshape [| rows; slots; 1 |] b in
  let inside =
    Nx.logical_and
      (Nx.greater_equal k (bound r.slot_start))
      (Nx.less k (bound r.slot_stop))
  in
  let scores =
    Nx.broadcast_to [| rows; slots; width |]
      (Nx.reshape [| rows; 1; width |] (Nx.add logits r.log_count))
  in
  let o = Nx.matmul (Nx.softmax ~where:inside scores) x in
  Nx.take ~axis:0 r.at (Nx.reshape [| rows * slots; d |] o)

(* Each token's index in its slot, [[rows; width]]. *)
let slot_index (r : 'd Batch.rows) =
  let w = Nx.dim 1 r.start in
  let col =
    Nx.reshape [| 1; w |] (Nx.arange_at (Nx.beside r.start) Nx.int64 0 w 1)
  in
  Nx.sub col (Nx.cast Nx.int64 r.start)

(* Inputs *)

(* A sensor's inputs: its standardised bands [[rows; width; bands]] and its days
   [[rows; width]]. *)
type 'd sensor = { bands : 'd f32; days : 'd f32 }

(* A batch's inputs: each sensor's, with the batch's layout. The batch's tokens
   are not read again. *)
type 'd inputs = { batch : 'd Batch.t; s2_in : 'd sensor; s1_in : 'd sensor }

let sensor_inputs z (r : 'd Batch.rows) =
  { bands = z; days = Nx.squeeze ~axes:[ 2 ] (channel r.tokens (Nx.dim 2 z)) }

let inputs m (b : 'd Batch.t) =
  {
    batch = b;
    s2_in = sensor_inputs (s2_inputs m.norm b.s2) b.s2;
    s1_in = sensor_inputs (s1_inputs m.norm b.s1) b.s1;
  }

(* Every array of [b]'s layout, by name, for the layout check. *)
let layout_shapes (b : 'd Batch.t) =
  let rows name (r : 'd Batch.rows) =
    List.map
      (fun (n, s) -> (name ^ "." ^ n, s))
      [
        ("tokens", Nx.shape r.tokens);
        ("log_count", Nx.shape r.log_count);
        ("start", Nx.shape r.start);
        ("stop", Nx.shape r.stop);
        ("slot_start", Nx.shape r.slot_start);
        ("slot_stop", Nx.shape r.slot_stop);
        ("place", Nx.shape r.place);
        ("at", Nx.shape r.at);
      ]
  in
  (("pixel", Nx.shape b.pixel) :: rows "s2" b.s2) @ rows "s1" b.s1

let mixup ~alpha i i' =
  if Nx.ndim alpha <> 0 then
    invalid_arg "Tessera.Model.mixup: alpha is not a scalar";
  List.iter2
    (fun (n, s) (_, s') ->
      if s <> s' then
        invalid_arg
          (Printf.sprintf "Tessera.Model.mixup: inputs of other layouts (%s)" n))
    (layout_shapes i.batch) (layout_shapes i'.batch);
  let beta = Nx.rsub_s 1. alpha in
  let mix x x' = Nx.add (Nx.mul alpha x) (Nx.mul beta x') in
  let sensor a a' =
    { bands = mix a.bands a'.bands; days = mix a.days a'.days }
  in
  { i with s2_in = sensor i.s2_in i'.s2_in; s1_in = sensor i.s1_in i'.s1_in }

let backbone ~pre_ln ~heads ~remat ~record ~drop b (r : 'd Batch.rows)
    { bands = z; days = doy } =
  record "in" (fun () ->
      Nx.concatenate ~axis:2 [ z; Nx.unsqueeze ~axes:[ 2 ] doy ]);
  let e1, e2 = b.embed in
  let h = linear e1 z in
  record "embed_hidden" (fun () -> h);
  let x = linear e2 (relu h) in
  record "embedding" (fun () -> x);
  let pe = encoding (Nx.dim 2 x) doy in
  record "pe" (fun () -> pe);
  let x = Nx.add x pe in
  record "encoder_in" (fun () -> x);
  let tk =
    {
      start = r.start;
      stop = r.stop;
      bias = r.log_count;
      place = r.place;
      index = slot_index r;
      drop;
    }
  in
  let x = layers ~pre_ln ~heads ~remat tk ~record x b.layers in
  let logits = Nx.squeeze ~axes:[ 2 ] (linear b.pool x) in
  record "pool_logits" (fun () -> logits);
  let v = pool r logits x in
  record "pooled" (fun () -> v);
  v

let reduce ~record ~drop r x =
  let x = linear r.up x in
  record "reducer0_linear0" (fun () -> x);
  let x = layer_norm r.norm x in
  record "reducer1_ln1" (fun () -> x);
  let x = relu x in
  record "reducer2_relu" (fun () -> x);
  let x =
    match drop with
    | Some d when d.rates.reducer > 0. ->
        let p = Nx.dim 0 x in
        let k = site d.keys [ reducer_site ] in
        let k =
          Nx.Rng.of_tensor (Nx.reshape [| p; 1; 2 |] (Nx.Rng.to_tensor k))
        in
        let m = draw ~rate:d.rates.reducer k x in
        d.mask "reducer" (fun () -> m);
        keep ~rate:d.rates.reducer m x
    | _ -> x
  in
  let x = linear r.down x in
  record "reducer4_linear4" (fun () -> x);
  let x = Nx.standardize ~eps x in
  record "reducer5_output" (fun () -> x);
  record "embedding" (fun () -> x);
  x

let forward ~record ?drop ~remat m (i : 'd inputs) =
  let heads = m.heads in
  let b = i.batch in
  (* Sensor [k]'s pooled vectors. *)
  let sensor name k pre_ln bb rows x =
    let drop =
      Option.map
        (fun d -> ({ d with mask = (fun n v -> d.mask (name ^ "_" ^ n) v) }, k))
        drop
    in
    backbone ~pre_ln ~heads ~remat
      ~record:(fun n v -> record (name ^ "_" ^ n) v)
      ~drop bb rows x
  in
  let s2_sensor pre_ln bb = sensor "s2" 0 pre_ln bb b.s2 i.s2_in
  and s1_sensor pre_ln bb = sensor "s1" 1 pre_ln bb b.s1 i.s1_in in
  match m.net with
  | Post_ln { s2; s1; reducer } ->
      let v2 = s2_sensor false s2 in
      let v1 = s1_sensor false s1 in
      reduce ~record ~drop reducer (Nx.concatenate ~axis:1 [ v2; v1 ])
  | Pre_ln { s2; s1; modality; fusion = layers_; reducer } ->
      let v2 = s2_sensor true s2 in
      let v1 = s1_sensor true s1 in
      let x = Nx.add (Nx.stack ~axis:1 [ v2; v1 ]) modality in
      let p = Nx.dim 0 x and d = Nx.dim 2 x in
      let at = Nx.beside b.pixel in
      let tk =
        {
          start = Nx.zeros_at at Nx.int32 [| p; 2 |];
          stop = Nx.full_at at Nx.int32 [| p; 2 |] 2l;
          bias = Nx.zeros_at at f32 [| p; 2 |];
          place =
            Nx.broadcast_to [| p; 2 |]
              (Nx.reshape [| p; 1 |] (Nx.arange_at at Nx.int64 0 p 1));
          index =
            Nx.broadcast_to [| p; 2 |]
              (Nx.reshape [| 1; 2 |] (Nx.arange_at at Nx.int64 0 2 1));
          drop =
            Option.map
              (fun d ->
                ({ d with mask = (fun n v -> d.mask ("fusion_" ^ n) v) }, fusion))
              drop;
        }
      in
      let fused n v = record ("fusion_" ^ n) v in
      let x = layers ~pre_ln:true ~heads ~remat tk ~record:fused x layers_ in
      reduce ~record ~drop reducer (Nx.reshape [| p; 2 * d |] x)

let ignore_record _ _ = ()
let apply m b = forward ~record:ignore_record ~remat:false m (inputs m b)

let check_dropout fn (r : dropout) keys (b : 'd Batch.t) =
  let bad x = not (x >= 0. && x < 1.) in
  if bad r.layers || bad r.reducer then
    invalid_arg (Printf.sprintf "Tessera.Model.%s: a rate outside [0, 1)" fn);
  let pixels = Nx.dim 0 b.pixel in
  if Nx.shape (Nx.Rng.to_tensor keys) <> [| pixels; 2 |] then
    invalid_arg
      (Printf.sprintf "Tessera.Model.%s: keys of shape [%d; 2] expected" fn
         pixels)

let train ?record ~dropout ~remat ~keys m i =
  check_dropout "train" dropout keys i.batch;
  if remat && Option.is_some record then
    invalid_arg "Tessera.Model.train: a record with remat";
  let record = Option.value ~default:ignore_record record in
  let drop = { rates = dropout; keys; mask = ignore_record } in
  forward ~record ~drop ~remat m i

let blocks m i =
  let acc = ref [] in
  ignore
    (forward ~record:(fun n v -> acc := (n, v ()) :: !acc) ~remat:false m i);
  List.rev !acc

let masks ~dropout ~keys m b =
  check_dropout "masks" dropout keys b;
  let acc = ref [] in
  let drop =
    { rates = dropout; keys; mask = (fun n v -> acc := (n, v ()) :: !acc) }
  in
  ignore (forward ~record:ignore_record ~drop ~remat:false m (inputs m b));
  List.rev !acc

(* Loading *)

module Pt = Nx_io.Pytorch

(* A checkpoint's tensors, and the names read so far. *)
type reader = {
  file : string;
  archive : Nx.host Nx_io.Archive.t;
  used : (string, unit) Hashtbl.t;
}

let fail file fmt =
  Printf.ksprintf (fun m -> failwith ("Tessera.Model: " ^ file ^ ": " ^ m)) fmt

let get r shape name =
  Hashtbl.replace r.used name ();
  try Nx_io.Archive.float ~shape f32 name r.archive
  with Failure m -> fail r.file "%s" m

(* PyTorch stores a linear layer's weight as [[outputs; inputs]]. *)
let linear_of r ?(bias = true) name ~inputs ~outputs =
  let w =
    Nx.matrix_transpose (get r [| outputs; inputs |] (name ^ ".weight"))
  in
  let b = if bias then Some (get r [| outputs |] (name ^ ".bias")) else None in
  { Kaun.Linear.w; b }

let layer_norm_of r name d =
  {
    Kaun.Layer_norm.gamma = get r [| d |] (name ^ ".weight");
    beta = get r [| d |] (name ^ ".bias");
  }

(* [nn.TransformerEncoderLayer]: q, k and v packed in [in_proj]. *)
let packed_layer r name ~d ~ffn =
  let w = get r [| 3 * d; d |] (name ^ ".self_attn.in_proj_weight") in
  let b = get r [| 3 * d |] (name ^ ".self_attn.in_proj_bias") in
  let part i =
    {
      Kaun.Linear.w =
        Nx.matrix_transpose (Nx.slice [ Nx.R (i * d, (i + 1) * d); Nx.A ] w);
      b = Some (Nx.slice [ Nx.R (i * d, (i + 1) * d) ] b);
    }
  in
  let out = linear_of r (name ^ ".self_attn.out_proj") ~inputs:d ~outputs:d in
  {
    attn = { Kaun.Attention.q = part 0; k = part 1; v = part 2; out };
    qk = None;
    norm1 = layer_norm_of r (name ^ ".norm1") d;
    ffn1 = linear_of r (name ^ ".linear1") ~inputs:d ~outputs:ffn;
    ffn2 = linear_of r (name ^ ".linear2") ~inputs:ffn ~outputs:d;
    norm2 = layer_norm_of r (name ^ ".norm2") d;
  }

(* [QKNormEncoderLayer]: q, k and v apart, RMS scales per head. *)
let qk_layer r name ~d ~heads ~ffn =
  let lin n = linear_of r (name ^ "." ^ n) ~inputs:d ~outputs:d in
  let attn =
    {
      Kaun.Attention.q = lin "q_proj";
      k = lin "k_proj";
      v = lin "v_proj";
      out = lin "out_proj";
    }
  in
  let scale n = get r [| d / heads |] (name ^ "." ^ n ^ ".weight") in
  {
    attn;
    qk = Some (scale "q_norm", scale "k_norm");
    norm1 = layer_norm_of r (name ^ ".norm1") d;
    ffn1 = linear_of r (name ^ ".linear1") ~inputs:d ~outputs:ffn;
    ffn2 = linear_of r (name ^ ".linear2") ~inputs:ffn ~outputs:d;
    norm2 = layer_norm_of r (name ^ ".norm2") d;
  }

let backbone_of r name ~bands ~d ~layers layer =
  let embed i inputs =
    linear_of r (Printf.sprintf "%s.embedding.%d" name i) ~inputs ~outputs:d
  in
  {
    embed = (embed 0 bands, embed 2 d);
    layers = List.init layers (layer r);
    pool = linear_of r (name ^ ".attn_pool.query") ~inputs:d ~outputs:1;
  }

let reducer_of_file r ~d ~dims =
  {
    up = linear_of r "dim_reducer.0" ~inputs:(2 * d) ~outputs:(4 * d);
    norm = layer_norm_of r "dim_reducer.1" (4 * d);
    down = linear_of r "dim_reducer.4" ~inputs:(4 * d) ~outputs:dims;
  }

let rec plain = function Pt.Shared (_, v) -> plain v | v -> v

let entry key = function
  | Pt.Dict kvs -> Option.map plain (List.assoc_opt (Pt.String key) kvs)
  | _ -> None

(* A field of [args] or [arch], the reference's default where absent. *)
let int_arg file dict name default =
  match Option.bind dict (entry name) with
  | None -> default
  | Some (Pt.Int n) -> n
  | Some _ -> fail file "%s: an int expected" name

let bool_arg file dict name default =
  match Option.bind dict (entry name) with
  | None -> default
  | Some (Pt.Bool b) -> b
  | Some (Pt.Int n) -> n <> 0
  | Some _ -> fail file "%s: a bool expected" name

(* [matryoshka_dims], which the students store as the string
   ["16,32,64,128"]. *)
let prefixes_arg file dict =
  let ints l =
    List.map (function Pt.Int n -> n | _ -> fail file "matryoshka_dims") l
  in
  match Option.bind dict (entry "matryoshka_dims") with
  | None -> [ 16; 32; 64; 128 ]
  | Some (Pt.String s) ->
      List.map
        (fun x ->
          match int_of_string_opt (String.trim x) with
          | Some n -> n
          | None -> fail file "matryoshka_dims: %S" s)
        (String.split_on_char ',' s)
  | Some (Pt.List l | Pt.Tuple l) -> ints (List.map plain l)
  | Some _ -> fail file "matryoshka_dims: a string expected"

let reader file state =
  { file; archive = Pt.tensors state; used = Hashtbl.create 256 }

let check_used r =
  List.iter
    (fun n -> if not (Hashtbl.mem r.used n) then fail r.file "%s: unused" n)
    (Nx_io.Archive.names r.archive)

(* tessera_infer_v2 student/model.py, [load_model]: [nn.TransformerEncoder]'s
   layers, or [QKNormEncoderLayer]s in a [ModuleList] with [enable_qk_norm]. *)
let student file state args =
  let r = reader file state in
  let arg name default = int_arg file args name default in
  let d = 4 * arg "latent_dim" 64 and heads = arg "nhead" 4 in
  let layers = arg "num_layers" 4 and ffn = arg "dim_feedforward" 1024 in
  let dims = arg "repr_dim" 128 in
  let qk = bool_arg file args "enable_qk_norm" false in
  let backbone name ~bands =
    let layer r i =
      if qk then
        qk_layer r
          (Printf.sprintf "%s.transformer_encoder.%d" name i)
          ~d ~heads ~ffn
      else
        packed_layer r
          (Printf.sprintf "%s.transformer_encoder.layers.%d" name i)
          ~d ~ffn
    in
    backbone_of r name ~bands ~d ~layers layer
  in
  let s2 = backbone "s2_backbone" ~bands:Pixels.s2_bands in
  let s1 = backbone "s1_backbone" ~bands:Pixels.s1_bands in
  let reducer = reducer_of_file r ~d ~dims in
  check_used r;
  {
    net = Post_ln { s2; s1; reducer };
    heads;
    norm = student_stats Nx.Host.on;
    prefixes = prefixes_arg file args;
  }

(* tessera_infer_v2 teacher/model.py, [load_model]: two fusion layers of 4 heads
   and feed-forward width [2d]. *)
let fusion_layers = 2
let fusion_heads = 4

let teacher_of file state arch =
  let r = reader file state in
  let arg name default = int_arg file arch name default in
  let d = 4 * arg "latent_dim" 1024 and heads = arg "nhead" 4 in
  let layers = arg "num_layers" 4 and ffn = arg "dim_feedforward" 16384 in
  let dims = arg "repr_dim" 1024 in
  if heads <> fusion_heads then fail file "arch.nhead: %d, 4 expected" heads;
  if not (bool_arg file arch "final_layernorm" true) then
    fail file "arch.final_layernorm: false is not supported";
  let backbone name ~bands =
    let layer r i =
      qk_layer r
        (Printf.sprintf "%s.transformer_encoder.layers.%d" name i)
        ~d ~heads ~ffn
    in
    backbone_of r name ~bands ~d ~layers layer
  in
  let s2 = backbone "s2_backbone" ~bands:Pixels.s2_bands in
  let s1 = backbone "s1_backbone" ~bands:Pixels.s1_bands in
  let modality = get r [| 2; d |] "fusion_modality_embed" in
  let fusion =
    List.init fusion_layers (fun i ->
        qk_layer r
          (Printf.sprintf "fusion_transformer.layers.%d" i)
          ~d ~heads:fusion_heads ~ffn:(2 * d))
  in
  let reducer = reducer_of_file r ~d ~dims in
  check_used r;
  {
    net = Pre_ln { s2; s1; modality; fusion; reducer };
    heads;
    norm = teacher_stats Nx.Host.on;
    prefixes = [];
  }

let of_pytorch ~name v =
  let v = plain v in
  match (entry "model" v, entry "encoder_state_dict" v) with
  | Some state, _ -> student name state (entry "args" v)
  | None, Some state -> teacher_of name state (entry "arch" v)
  | None, None -> fail name "neither a student's nor the teacher's checkpoint"

let load file = of_pytorch ~name:file (Nx_io.load_pytorch file)

(* Making *)

type arch = {
  pre_ln : bool;
  width : int;
  heads : int;
  layers : int;
  ffn : int;
  dims : int;
}

let student_arch ~latent ~layers ~ffn =
  { pre_ln = false; width = 4 * latent; heads = 4; layers; ffn; dims = 128 }

let nano = student_arch ~latent:36 ~layers:2 ~ffn:384
let small = student_arch ~latent:64 ~layers:4 ~ffn:1024
let medium = student_arch ~latent:110 ~layers:4 ~ffn:1792
let large = student_arch ~latent:160 ~layers:4 ~ffn:2560

let teacher =
  {
    pre_ln = true;
    width = 4096;
    heads = 4;
    layers = 4;
    ffn = 16384;
    dims = 1024;
  }

let init ~key a =
  if a.width <= 0 || a.heads <= 0 || a.layers <= 0 || a.ffn <= 0 || a.dims <= 0
  then
    invalid_arg
      "Tessera.Model.init: a field of the architecture is not positive";
  if a.width mod (2 * a.heads) <> 0 then
    invalid_arg "Tessera.Model.init: width is not a multiple of 2 * heads";
  let at = Nx.Rng.placement key in
  let n = ref 0 in
  let key () =
    incr n;
    Nx.Rng.fold_in key !n
  in
  let linear inputs outputs =
    Kaun.Linear.make ~key:(key ()) ~inputs ~outputs f32
  in
  let d = a.width in
  let layer ~heads ~ffn _ =
    let qk =
      if a.pre_ln then
        let ones () = Nx.ones_at at f32 [| d / heads |] in
        Some (ones (), ones ())
      else None
    in
    {
      attn = Kaun.Attention.make ~key:(key ()) ~embed_dim:d f32;
      qk;
      norm1 = Kaun.Layer_norm.init ~dim:d at;
      ffn1 = linear d ffn;
      ffn2 = linear ffn d;
      norm2 = Kaun.Layer_norm.init ~dim:d at;
    }
  in
  let backbone bands =
    {
      embed = (linear bands d, linear d d);
      layers = List.init a.layers (layer ~heads:a.heads ~ffn:a.ffn);
      pool = linear d 1;
    }
  in
  let s2 = backbone Pixels.s2_bands and s1 = backbone Pixels.s1_bands in
  let reducer =
    {
      up = linear (2 * d) (4 * d);
      norm = Kaun.Layer_norm.init ~dim:(4 * d) at;
      down = linear (4 * d) a.dims;
    }
  in
  let net, prefixes =
    if a.pre_ln then
      let modality = Nx.Rng.normal_at at ~key:(key ()) f32 [| 2; d |] in
      let fusion =
        List.init fusion_layers (layer ~heads:a.heads ~ffn:(2 * d))
      in
      (Pre_ln { s2; s1; modality; fusion; reducer }, [])
    else
      ( Post_ln { s2; s1; reducer },
        List.filter (fun k -> k <= a.dims) [ 16; 32; 64; 128 ] )
  in
  { net; heads = a.heads; norm = student_stats at; prefixes }
