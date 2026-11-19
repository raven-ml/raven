(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'd f32 = 'd Nx.float32_t

type ('a, 'd) layer = {
  attn : ('a, 'd) Kaun.Attention.t;
  qk : ('d Nx.float32_t * 'd Nx.float32_t) option;
  norm1 : ('d Nx.float32_t, 'd) Kaun.Layer_norm.t;
  ffn1 : ('a, 'd) Kaun.Linear.t;
  ffn2 : ('a, 'd) Kaun.Linear.t;
  norm2 : ('d Nx.float32_t, 'd) Kaun.Layer_norm.t;
}
[@@deriving ptree]

type ('a, 'd) backbone = {
  embed : ('a, 'd) Kaun.Linear.t * ('a, 'd) Kaun.Linear.t;
  layers : ('a, 'd) layer list;
  pool : ('a, 'd) Kaun.Linear.t;
}
[@@deriving ptree]

type ('a, 'd) reducer = {
  up : ('a, 'd) Kaun.Linear.t;
  norm : ('d Nx.float32_t, 'd) Kaun.Layer_norm.t;
  down : ('a, 'd) Kaun.Linear.t;
}
[@@deriving ptree]

type ('a, 'd) net =
  | Post_ln of {
      s2 : ('a, 'd) backbone;
      s1 : ('a, 'd) backbone;
      reducer : ('a, 'd) reducer;
    }
  | Pre_ln of {
      s2 : ('a, 'd) backbone;
      s1 : ('a, 'd) backbone;
      modality : 'd Nx.float32_t;
      fusion : ('a, 'd) layer list;
      reducer : ('a, 'd) reducer;
    }
[@@deriving ptree]

type 'd norm = {
  s2_mean : 'd Nx.float32_t;
  s2_std : 'd Nx.float32_t;
  s1_mean : 'd Nx.float32_t;
  s1_std : 'd Nx.float32_t;
}
[@@deriving ptree]

type ('a, 'd) t = {
  net : ('a, 'd) net;
  heads : int; [@ptree.int]
  norm : 'd norm;
}
[@@deriving ptree]

module Net = struct
  type nonrec ('a, 'd) t = ('a, 'd) net

  let walk = walk_net
end

let reducer_of = function
  | Post_ln { reducer; _ } | Pre_ln { reducer; _ } -> reducer

let dims m = Nx.dim 0 (reducer_of m.net).down.w

(* The students' prefixes, which upstream's command states (tessera_infer_v2
   infer_v2.py:130). *)
let prefixes m =
  match m.net with
  | Pre_ln _ -> []
  | Post_ln _ -> List.filter (fun k -> k <= dims m) [ 16; 32; 64; 128 ]

(* Statistics *)

(* Standardisation constants of the released checkpoints, bound to them by
   training (tessera_infer_v2 student/model.py:23-36, teacher/model.py:21-35).
   The students hold the ascending and descending orbits apart; the teacher
   pools them. Bands in the order B04 B02 B03 B08 B8A B05 B06 B07 B11 B12, then
   VV and VH. *)
let s2_mean =
  [| 1633.0042; 1341.1090; 1539.5536; 3054.8269; 3117.4658; 2004.1648;
     2694.7275; 2945.1504; 2266.6079; 1657.3094 |]
[@@ocamlformat "disable"]

let s2_std =
  [| 1999.4603; 2014.7549; 1929.2201; 1754.2493; 1649.9807; 1936.8988;
     1748.6041; 1708.6991; 1207.5250; 1108.6046 |]
[@@ocamlformat "disable"]

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

(* [x · w + b] at [w]'s dtype, [x] cast to it first as autocast does (a gap of
   kaun's: mixed precision). *)
let product (p : (_, _) Kaun.Linear.t) x =
  Kaun.Linear.apply p (Nx.cast (Nx.dtype p.w) x)

let linear p x = Nx.cast f32 (product p x)

let layer_norm (p : ('d f32, 'd) Kaun.Layer_norm.t) x =
  Nx.standardize ~eps ~scale:p.gamma ?shift:p.beta x

(* Dropout *)

type dropout = { layers : float; reducer : float }

let v2_dropout = { layers = 0.1; reducer = 0.2 }
let no_dropout = { layers = 0.; reducer = 0. }

(* One call's dropout: its rates, each pixel place's key, and where its masks
   are reported. Kaun.Dropout takes one key a call, and no attention
   probabilities (a gap of kaun's: keyed dropout), so the masks are drawn
   here. *)
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
    (Nx.Rng.reshape [| r; t; 1 |] k)
    x

(* The attention probabilities' mask over [x], [[r; h; t; t]], one key per pair
   of tokens drawing each head: the query's key indexed by the key's [index]. *)
let pair_mask ~rate k ~index x =
  let r = Nx.dim 0 x and t = Nx.dim 2 x in
  let q = Nx.Rng.reshape [| r; t; 1 |] k in
  let pairs = Nx.Rng.fold_in_tensor q (Nx.reshape [| r; 1; t |] index) in
  draw ~rate
    (Nx.Rng.reshape [| r; 1; t; t |] pairs)
    x

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

(* Attention with its probabilities dropped, as two calls sharing the scores:
   for the keep mask [z], [Σ_j z_j p_j v_j] is [o_kept] times the kept keys'
   share of the row's weight, [σ (lse_kept − lse_lost)], from the call over
   the kept keys and the call over the lost ones, read for its [lse] alone
   and so given values of no column. The share carries the
   [lse]s' errors scaled by the lost share, about [rate], where [exp
   (lse_kept − lse)] would carry them whole; {!Nx.sigmoid} never overflows,
   where [1 / (1 + exp (lse_lost − lse_kept))] would lose its derivative to
   [0 · ∞] once the lost keys outweigh the kept ones by [e^89]. [o_kept]
   rounds to [q]'s dtype before the product, which rounds again.

   A row that keeps no key, whether it sees none (a padding token) or drops
   every one it sees, gives 0. The kept call gives it [o_kept = 0] and
   [lse_kept = −∞], and its share is 0 with its argument taken at 0: where
   the row sees no key, [lse_lost] is [−∞] too, and [−∞ − (−∞)] would make
   the argument and its cotangent NaN. *)
let dropped (tk : 'd tokens) s q k v =
  let r = Nx.dim 0 q and h = Nx.dim 1 q and t = Nx.dim 2 q in
  let x = Nx.broadcast_to [| r; h; t; t |] (Nx.scalar_like tk.bias 0.) in
  let m = pair_mask ~rate:s.rate (s.keys attn_p) ~index:tk.index x in
  s.report "attn_p" (fun () -> m);
  let within = within tk.start tk.stop and score = key_bias tk.bias in
  let kept p = Nx.logical_and (within p) m in
  (* At each pair: [logical_not m] would be an array of its own. *)
  let lost p = Nx.logical_xor (within p) (kept p) in
  let o, lse_kept =
    Nx.attention ~mask:kept ~score ~acc:f32 ~kv:(Nx.Dense { k; v }) q
  in
  let none = Nx.slice [ Nx.A; Nx.A; Nx.A; Nx.R (0, 0) ] v in
  let _, lse_lost =
    Nx.attention ~mask:lost ~score ~acc:f32 ~kv:(Nx.Dense { k; v = none }) q
  in
  let live = Nx.not_equal_s lse_kept Float.neg_infinity in
  let share =
    Nx.sigmoid
      (Nx.where live (Nx.sub lse_kept lse_lost) (Nx.scalar_like lse_kept 0.))
  in
  let c = Nx.reshape [| r; h; t; 1 |] (keep ~rate:s.rate live share) in
  Nx.cast (Nx.dtype v) (Nx.mul (Nx.cast f32 o) c)

let split ~heads (p : ((float, 's, 'd) Nx.t, 'd) Kaun.Linear.t) x =
  let b = Nx.dim 0 x and t = Nx.dim 1 x and d = Nx.dim 0 p.w in
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
    | Some s -> dropped tk s q k v
    | None ->
        let mask = within tk.start tk.stop and score = key_bias tk.bias in
        fst (Nx.attention ~mask ~score ~acc:f32 ~kv:(Nx.Dense { k; v }) q)
  in
  merge p.out o

let ffn l ds x =
  let h = linear l.ffn1 x in
  let h = masked ds ~name:"ffn_hidden" ffn_hidden (Kaun.Fn.relu h) in
  masked ds ~name:"ffn" ffn_out (linear l.ffn2 h)

let layer ~pre_ln ~heads tk i x l =
  let ds = sites tk i in
  let attend x = masked ds ~name:"attn" attn_out (attention ~heads l tk ds x) in
  if pre_ln then begin
    let x = Nx.add x (attend (layer_norm l.norm1 x)) in
    Nx.add x (ffn l ds (layer_norm l.norm2 x))
  end
  else begin
    let x = layer_norm l.norm1 (Nx.add x (attend x)) in
    layer_norm l.norm2 (Nx.add x (ffn l ds x))
  end

let layers ~pre_ln ~heads tk x ls =
  List.fold_left
    (fun (i, x) l -> (i + 1, layer ~pre_ln ~heads tk i x l))
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

(* TESSERA's encoding of a Sentinel-2 band: below 0, raised by [fold]. *)
let fold = 1000.

let s2_inputs norm (r : 'd Batch.rows) =
  let n = Pixels.s2_bands in
  let x = bands_of r.tokens n in
  let x = Nx.where (Nx.less_s x 0.) (Nx.add_s x fold) x in
  standard ~source:(source_of r.tokens n) x norm.s2_mean norm.s2_std

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

let backbone ~pre_ln ~heads ~drop b (r : 'd Batch.rows)
    { bands = z; days = doy } =
  let e1, e2 = b.embed in
  let x = linear e2 (Kaun.Fn.relu (linear e1 z)) in
  let x = Nx.add x (encoding (Nx.dim 2 x) doy) in
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
  let x = layers ~pre_ln ~heads tk x b.layers in
  pool r (Nx.squeeze ~axes:[ 2 ] (linear b.pool x)) x

let reduce ~drop (r : _ reducer) x =
  let x = Kaun.Fn.relu (layer_norm r.norm (linear r.up x)) in
  let x =
    match drop with
    | Some d when d.rates.reducer > 0. ->
        let p = Nx.dim 0 x in
        let k = site d.keys [ reducer_site ] in
        let k =
          Nx.Rng.reshape [| p; 1 |] k
        in
        let m = draw ~rate:d.rates.reducer k x in
        d.mask "reducer" (fun () -> m);
        keep ~rate:d.rates.reducer m x
    | _ -> x
  in
  Nx.standardize ~eps (linear r.down x)

let forward ?drop m (i : 'd inputs) =
  let heads = m.heads in
  let b = i.batch in
  (* Sensor [k]'s pooled vectors. *)
  let sensor name k pre_ln bb rows x =
    let drop =
      Option.map
        (fun d -> ({ d with mask = (fun n v -> d.mask (name ^ "_" ^ n) v) }, k))
        drop
    in
    backbone ~pre_ln ~heads ~drop bb rows x
  in
  let s2_sensor pre_ln bb = sensor "s2" 0 pre_ln bb b.s2 i.s2_in
  and s1_sensor pre_ln bb = sensor "s1" 1 pre_ln bb b.s1 i.s1_in in
  match m.net with
  | Post_ln { s2; s1; reducer } ->
      let v2 = s2_sensor false s2 in
      let v1 = s1_sensor false s1 in
      reduce ~drop reducer (Nx.concatenate ~axis:1 [ v2; v1 ])
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
      let x = layers ~pre_ln:true ~heads tk x layers_ in
      reduce ~drop reducer (Nx.reshape [| p; 2 * d |] x)

let apply m b = forward m (inputs m b)

let check_dropout fn (r : dropout) keys (b : 'd Batch.t) =
  let bad x = not (x >= 0. && x < 1.) in
  if bad r.layers || bad r.reducer then
    invalid_arg (Printf.sprintf "Tessera.Model.%s: a rate outside [0, 1)" fn);
  let pixels = Nx.dim 0 b.pixel in
  if Nx.shape (Nx.Rng.to_tensor keys) <> [| pixels; 2 |] then
    invalid_arg
      (Printf.sprintf "Tessera.Model.%s: keys of shape [%d; 2] expected" fn
         pixels)

let train ~dropout ~keys m i =
  check_dropout "train" dropout keys i.batch;
  forward ~drop:{ rates = dropout; keys; mask = (fun _ _ -> ()) } m i

let masks ~dropout ~keys m b =
  check_dropout "masks" dropout keys b;
  let acc = ref [] in
  let drop =
    { rates = dropout; keys; mask = (fun n v -> acc := (n, v ()) :: !acc) }
  in
  ignore (forward ~drop m (inputs m b));
  List.rev !acc

(* Loading *)

module Pt = Nx_io.Pytorch

let fail file fmt =
  Printf.ksprintf (fun m -> failwith ("Tessera.Model: " ^ file ^ ": " ^ m)) fmt

(* [tensors file state f] is [f] of [state]'s tensors. It refuses a checkpoint
   holding a tensor [f] does not read, and its failures name [file]. *)
let tensors file state f =
  let a = Pt.tensors state in
  try Nx_io.Archive.import f a with Failure m -> fail file "%s" m

let get a shape name = Nx_io.Archive.float ~shape f32 name a

(* PyTorch stores a linear layer's weight as [[outputs; inputs]],
   [Kaun.Linear]'s layout, in C order, as [init]'s weights lie. *)
let linear_of a ?(bias = true) name ~inputs ~outputs =
  let w = get a [| outputs; inputs |] (name ^ ".weight") in
  let b = if bias then Some (get a [| outputs |] (name ^ ".bias")) else None in
  { Kaun.Linear.w; b }

let layer_norm_of a name d =
  {
    Kaun.Layer_norm.gamma = get a [| d |] (name ^ ".weight");
    beta = Some (get a [| d |] (name ^ ".bias"));
  }

(* [nn.TransformerEncoderLayer]: q, k and v packed in [in_proj]. *)
let packed_layer a name ~d ~ffn =
  let w = get a [| 3 * d; d |] (name ^ ".self_attn.in_proj_weight") in
  let b = get a [| 3 * d |] (name ^ ".self_attn.in_proj_bias") in
  let part i =
    {
      Kaun.Linear.w = Nx.copy (Nx.slice [ Nx.R (i * d, (i + 1) * d); Nx.A ] w);
      b = Some (Nx.copy (Nx.slice [ Nx.R (i * d, (i + 1) * d) ] b));
    }
  in
  let out = linear_of a (name ^ ".self_attn.out_proj") ~inputs:d ~outputs:d in
  {
    attn = { Kaun.Attention.q = part 0; k = part 1; v = part 2; out };
    qk = None;
    norm1 = layer_norm_of a (name ^ ".norm1") d;
    ffn1 = linear_of a (name ^ ".linear1") ~inputs:d ~outputs:ffn;
    ffn2 = linear_of a (name ^ ".linear2") ~inputs:ffn ~outputs:d;
    norm2 = layer_norm_of a (name ^ ".norm2") d;
  }

(* [QKNormEncoderLayer]: q, k and v apart, RMS scales per head. *)
let qk_layer a name ~d ~heads ~ffn =
  let lin n = linear_of a (name ^ "." ^ n) ~inputs:d ~outputs:d in
  let attn =
    {
      Kaun.Attention.q = lin "q_proj";
      k = lin "k_proj";
      v = lin "v_proj";
      out = lin "out_proj";
    }
  in
  let scale n = get a [| d / heads |] (name ^ "." ^ n ^ ".weight") in
  {
    attn;
    qk = Some (scale "q_norm", scale "k_norm");
    norm1 = layer_norm_of a (name ^ ".norm1") d;
    ffn1 = linear_of a (name ^ ".linear1") ~inputs:d ~outputs:ffn;
    ffn2 = linear_of a (name ^ ".linear2") ~inputs:ffn ~outputs:d;
    norm2 = layer_norm_of a (name ^ ".norm2") d;
  }

let backbone_of a name ~bands ~d ~layers layer =
  let embed i inputs =
    linear_of a (Printf.sprintf "%s.embedding.%d" name i) ~inputs ~outputs:d
  in
  {
    embed = (embed 0 bands, embed 2 d);
    layers = List.init layers (layer a);
    pool = linear_of a (name ^ ".attn_pool.query") ~inputs:d ~outputs:1;
  }

let reducer_of_file a ~d ~dims =
  {
    up = linear_of a "dim_reducer.0" ~inputs:(2 * d) ~outputs:(4 * d);
    norm = layer_norm_of a "dim_reducer.1" (4 * d);
    down = linear_of a "dim_reducer.4" ~inputs:(4 * d) ~outputs:dims;
  }

let find file path v = try Pt.find path v with Failure m -> fail file "%s" m

(* A field of the checkpoint's [args] or [arch], the reference's default where
   absent. *)
let int_arg file v dict name default =
  match find file [ dict; name ] v with
  | None -> default
  | Some (Pt.Int n) -> n
  | Some _ -> fail file "%s: an int expected" name

let bool_arg file v dict name default =
  match find file [ dict; name ] v with
  | None -> default
  | Some (Pt.Bool b) -> b
  | Some (Pt.Int n) -> n <> 0
  | Some _ -> fail file "%s: a bool expected" name

(* tessera_infer_v2 student/model.py, [load_model]: [nn.TransformerEncoder]'s
   layers. No released student sets [enable_qk_norm]. *)
let student file state v =
  let arg name default = int_arg file v "args" name default in
  let d = 4 * arg "latent_dim" 64 and heads = arg "nhead" 4 in
  let layers = arg "num_layers" 4 and ffn = arg "dim_feedforward" 1024 in
  let dims = arg "repr_dim" 128 in
  if bool_arg file v "args" "enable_qk_norm" false then
    fail file "args.enable_qk_norm: true is not supported";
  let backbone a name ~bands =
    let layer a i =
      packed_layer a
        (Printf.sprintf "%s.transformer_encoder.layers.%d" name i)
        ~d ~ffn
    in
    backbone_of a name ~bands ~d ~layers layer
  in
  let net a =
    let s2 = backbone a "s2_backbone" ~bands:Pixels.s2_bands in
    let s1 = backbone a "s1_backbone" ~bands:Pixels.s1_bands in
    Post_ln { s2; s1; reducer = reducer_of_file a ~d ~dims }
  in
  let net = tensors file state net in
  { net; heads; norm = student_stats Nx.Host.on }

(* tessera_infer_v2 teacher/model.py, [load_model]: two fusion layers of 4 heads
   and feed-forward width [2d]. *)
let fusion_layers = 2
let fusion_heads = 4

let teacher_of file state v =
  let arg name default = int_arg file v "arch" name default in
  let d = 4 * arg "latent_dim" 1024 and heads = arg "nhead" 4 in
  let layers = arg "num_layers" 4 and ffn = arg "dim_feedforward" 16384 in
  let dims = arg "repr_dim" 1024 in
  if heads <> fusion_heads then fail file "arch.nhead: %d, 4 expected" heads;
  if not (bool_arg file v "arch" "final_layernorm" true) then
    fail file "arch.final_layernorm: false is not supported";
  let backbone a name ~bands =
    let layer a i =
      qk_layer a
        (Printf.sprintf "%s.transformer_encoder.layers.%d" name i)
        ~d ~heads ~ffn
    in
    backbone_of a name ~bands ~d ~layers layer
  in
  let net a =
    let s2 = backbone a "s2_backbone" ~bands:Pixels.s2_bands in
    let s1 = backbone a "s1_backbone" ~bands:Pixels.s1_bands in
    let modality = get a [| 2; d |] "fusion_modality_embed" in
    let fusion =
      List.init fusion_layers (fun i ->
          qk_layer a
            (Printf.sprintf "fusion_transformer.layers.%d" i)
            ~d ~heads:fusion_heads ~ffn:(2 * d))
    in
    Pre_ln { s2; s1; modality; fusion; reducer = reducer_of_file a ~d ~dims }
  in
  let net = tensors file state net in
  { net; heads; norm = teacher_stats Nx.Host.on }

let of_pytorch ~name v =
  match (find name [ "model" ] v, find name [ "encoder_state_dict" ] v) with
  | Some state, _ -> student name state v
  | None, Some state -> teacher_of name state v
  | None, None -> fail name "neither a student's nor the teacher's checkpoint"

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
  let net =
    if a.pre_ln then
      let modality = Nx.Rng.normal_at at ~key:(key ()) f32 [| 2; d |] in
      let fusion =
        List.init fusion_layers (layer ~heads:a.heads ~ffn:(2 * d))
      in
      Pre_ln { s2; s1; modality; fusion; reducer }
    else Post_ln { s2; s1; reducer }
  in
  { net; heads = a.heads; norm = student_stats at }

(* Files *)

let arch_of m =
  let pre_ln, (s2 : _ backbone), r =
    match m.net with
    | Post_ln n -> (false, n.s2, n.reducer)
    | Pre_ln n -> (true, n.s2, n.reducer)
  in
  {
    pre_ln;
    width = Nx.dim 0 (snd s2.embed).w;
    heads = m.heads;
    layers = List.length s2.layers;
    ffn = Nx.dim 0 (List.hd s2.layers).ffn1.w;
    dims = Nx.dim 0 r.down.w;
  }

(* The model as a structure: this module's own {!Nx.Ptree.S}. *)
module Self = struct
  type nonrec ('a, 'd) t = ('a, 'd) t

  let walk = walk
end

let save file m =
  let a = arch_of m in
  let metadata =
    ("pre_ln", string_of_bool a.pre_ln)
    :: List.map
         (fun (k, n) -> (k, string_of_int n))
         [
           ("width", a.width);
           ("heads", a.heads);
           ("layers", a.layers);
           ("ffn", a.ffn);
           ("dims", a.dims);
         ]
  in
  Nx_io.save_safetensors ~metadata file
    (Nx_io.Archive.of_value
       (Nx.Ptree.instantiate (module Self))
       (Nx.Ptree.place (module Self) Nx.Host.on m))

let load file =
  if not (Filename.check_suffix file ".safetensors") then
    of_pytorch ~name:file (Nx_io.load_pytorch file)
  else
    let metadata = Nx_io.load_safetensors_metadata file in
    let field k read =
      match Option.bind (List.assoc_opt k metadata) read with
      | Some v -> v
      | None -> fail file "metadata %s: missing or malformed" k
    in
    let int k = field k int_of_string_opt in
    let arch =
      {
        pre_ln = field "pre_ln" bool_of_string_opt;
        width = int "width";
        heads = int "heads";
        layers = int "layers";
        ffn = int "ffn";
        dims = int "dims";
      }
    in
    Nx_io.Archive.to_value
      (Nx.Ptree.instantiate (module Self))
      ~like:(init ~key:(Nx.Rng.key 0) arch)
      (Nx_io.load_safetensors file)
