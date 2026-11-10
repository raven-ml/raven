(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('a, 'd) layer = {
  attn : ('a, 'd) Kaun.Attention.t;
  qk : ('a * 'a) option;
  norm1 : ('a, 'd) Kaun.Layer_norm.t;
  ffn1 : ('a, 'd) Kaun.Linear.t;
  ffn2 : ('a, 'd) Kaun.Linear.t;
  norm2 : ('a, 'd) Kaun.Layer_norm.t;
}

type ('a, 'd) backbone = {
  embed : ('a, 'd) Kaun.Linear.t * ('a, 'd) Kaun.Linear.t;
  layers : ('a, 'd) layer list;
  pool : ('a, 'd) Kaun.Linear.t;
}

type ('a, 'd) reducer = {
  up : ('a, 'd) Kaun.Linear.t;
  norm : ('a, 'd) Kaun.Layer_norm.t;
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
      modality : 'a;
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

let walk_layer c l =
  let open Nx.Ptree.Walk in
  let attn = field c "attn" Kaun.Attention.walk l.attn in
  let pair c (q, k) =
    let q = index c 0 leaf q in
    let k = index c 1 leaf k in
    (q, k)
  in
  let qk = field c "qk" (option pair) l.qk in
  let norm1 = field c "norm1" Kaun.Layer_norm.walk l.norm1 in
  let ffn1 = field c "ffn1" Kaun.Linear.walk l.ffn1 in
  let ffn2 = field c "ffn2" Kaun.Linear.walk l.ffn2 in
  let norm2 = field c "norm2" Kaun.Layer_norm.walk l.norm2 in
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
  let norm = field c "norm" Kaun.Layer_norm.walk r.norm in
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
      let modality = field c "modality" leaf modality in
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

(* A linear layer at its parameters' dtype, on and to float32. *)
let linear (p : ((float, 's, 'd) Nx.t, 'd) Kaun.Linear.t) x =
  Nx.cast f32 (Kaun.Linear.apply p (Nx.cast (Nx.dtype p.w) x))

let layer_norm (p : ((float, 's, 'd) Nx.t, 'd) Kaun.Layer_norm.t) x =
  Nx.standardize ~eps ~scale:(Nx.cast f32 p.gamma) ~shift:(Nx.cast f32 p.beta) x

let relu x = Nx.maximum x (Nx.scalar_like x 0.)

let ffn l ~record x =
  let h = linear l.ffn1 x in
  record "linear1" (fun () -> h);
  linear l.ffn2 (relu h)

(* Self-attention of [x], [[batch; seq; d]], at the parameters' dtype. *)
let attention ~heads l ?mask ?score x =
  let p = l.attn in
  let dt = Nx.dtype p.q.w in
  let head_dim = Nx.dim (-1) x / heads in
  let split (w : ((float, 's, 'd) Nx.t, 'd) Kaun.Linear.t) =
    Kaun.Attention.split ~head_dim w (Nx.cast dt x)
  in
  let q = split p.q and k = split p.k and v = split p.v in
  let q, k =
    match l.qk with
    | None -> (q, k)
    | Some (gq, gk) ->
        let norm g x =
          Nx.cast dt
            (Nx.rms_norm ~eps:qk_eps ~scale:(Nx.cast f32 g) (Nx.cast f32 x))
        in
        (norm gq q, norm gk k)
  in
  let o, _ = Nx.attention ?mask ?score ~acc:f32 ~kv:(Nx.Dense { k; v }) q in
  Nx.cast f32 (Kaun.Attention.merge p.out o)

let layer ~pre_ln ~heads ?mask ?score ~record x l =
  let attend x = attention ~heads l ?mask ?score x in
  if pre_ln then begin
    let x = Nx.add x (attend (layer_norm l.norm1 x)) in
    Nx.add x (ffn l ~record (layer_norm l.norm2 x))
  end
  else begin
    let a = attend x in
    record "attn" (fun () -> a);
    let x = layer_norm l.norm1 (Nx.add x a) in
    record "norm1" (fun () -> x);
    layer_norm l.norm2 (Nx.add x (ffn l ~record x))
  end

(* The sinusoid's base and standardisation's guard (tessera_infer_v2
   student/model.py:47-63, infer.py:161). *)
let wavelength = 10000.
let std_guard = 1e-9

(* The sinusoid of [doy], [[rows; width]], over [d] channels. *)
let encoding d doy =
  let shape = Nx.shape doy in
  let i = Nx.arange_at (Nx.beside doy) f32 0 d 2 in
  let div = Nx.exp (Nx.mul_s i (-.(Float.log wavelength /. Float.of_int d))) in
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

(* One query per slot over the slot's tokens, [x] [[rows; width; d]], each score
   [w · x + b + log count]. *)
let pool (p : ((float, 's, 'd) Nx.t, 'd) Kaun.Linear.t) (r : 'd Batch.rows) x =
  let rows = Nx.dim 0 x and width = Nx.dim 1 x and d = Nx.dim 2 x in
  let slots = Nx.dim 1 r.slot_start in
  let w = Nx.reshape [| 1; 1; 1; d |] (Nx.cast f32 p.w) in
  let q = Nx.broadcast_to [| rows; 1; slots; d |] w in
  let kv = Nx.reshape [| rows; 1; width; d |] x in
  let logits =
    match p.b with
    | None -> r.log_count
    | Some b -> Nx.add r.log_count (Nx.reshape [| 1; 1 |] (Nx.cast f32 b))
  in
  let o, _ =
    Nx.attention ~scale:1.
      ~mask:(within r.slot_start r.slot_stop)
      ~score:(key_bias logits) ~acc:f32
      ~kv:(Nx.Dense { k = kv; v = kv })
      q
  in
  Nx.take ~axis:0 r.at (Nx.reshape [| rows * slots; d |] o)

let backbone ~pre_ln ~heads ~record b (r : 'd Batch.rows) z =
  let doy = Nx.squeeze ~axes:[ 2 ] (channel r.tokens (Nx.dim 2 z)) in
  record "in" (fun () ->
      Nx.concatenate ~axis:2 [ z; channel r.tokens (Nx.dim 2 z) ]);
  let e1, e2 = b.embed in
  let x = linear e2 (relu (linear e1 z)) in
  record "embedding" (fun () -> x);
  let pe = encoding (Nx.dim 2 x) doy in
  record "pe" (fun () -> pe);
  let x = Nx.add x pe in
  record "encoder_in" (fun () -> x);
  let mask = within r.start r.stop and score = key_bias r.log_count in
  let x =
    List.fold_left
      (fun (i, x) l ->
        let record name v =
          record
            (Printf.sprintf "layer%d%s" i
               (if name = "" then "" else "_" ^ name))
            v
        in
        let x = layer ~pre_ln ~heads ~mask ~score ~record x l in
        record "" (fun () -> x);
        (i + 1, x))
      (0, x) b.layers
    |> snd
  in
  record "pool_logits" (fun () -> Nx.squeeze ~axes:[ 2 ] (linear b.pool x));
  let v = pool b.pool r x in
  record "pooled" (fun () -> v);
  v

let reduce ~record r x =
  let x = linear r.up x in
  record "reducer0_linear0" (fun () -> x);
  let x = layer_norm r.norm x in
  record "reducer1_ln1" (fun () -> x);
  let x = relu x in
  record "reducer2_relu" (fun () -> x);
  let x = linear r.down x in
  record "reducer4_linear4" (fun () -> x);
  let x = Nx.standardize ~eps x in
  record "reducer5_output" (fun () -> x);
  record "embedding" (fun () -> x);
  x

let forward ~record m (b : 'd Batch.t) =
  let heads = m.heads in
  let sensor name pre_ln bb rows z =
    backbone ~pre_ln ~heads
      ~record:(fun n v -> record (name ^ "_" ^ n) v)
      bb rows z
  in
  let z2 = s2_inputs m.norm b.s2 and z1 = s1_inputs m.norm b.s1 in
  match m.net with
  | Post_ln { s2; s1; reducer } ->
      let v2 = sensor "s2" false s2 b.s2 z2 in
      let v1 = sensor "s1" false s1 b.s1 z1 in
      reduce ~record reducer (Nx.concatenate ~axis:1 [ v2; v1 ])
  | Pre_ln { s2; s1; modality; fusion; reducer } ->
      let v2 = sensor "s2" true s2 b.s2 z2 in
      let v1 = sensor "s1" true s1 b.s1 z1 in
      let x = Nx.add (Nx.stack ~axis:1 [ v2; v1 ]) (Nx.cast f32 modality) in
      let skip _ _ = () in
      let x =
        List.fold_left (layer ~pre_ln:true ~heads ~record:skip) x fusion
      in
      let p = Nx.dim 0 x and d = Nx.dim 2 x in
      reduce ~record reducer (Nx.reshape [| p; 2 * d |] x)

let apply m b = forward ~record:(fun _ _ -> ()) m b

let blocks m b =
  let acc = ref [] in
  ignore (forward ~record:(fun n v -> acc := (n, v ()) :: !acc) m b);
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
