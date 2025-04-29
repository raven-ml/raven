type ('layout, 'dev) tensor = (float, 'layout, [ `cpu ]) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype

module Rng = struct
  type t = int

  let create ?seed () =
    match seed with Some s -> s | None -> Random.int 1_000_000

  let normal key ~dtype ~shape =
    let tensor = Rune.randn dtype ~seed:key shape in
    tensor

  let uniform key ~dtype ~shape =
    let tensor = Rune.rand dtype ~seed:key shape in
    tensor
end

module Activation = struct
  type ('layout, 'dev) t = ('layout, 'dev) tensor -> ('layout, 'dev) tensor

  let identity x = x
  let relu = Rune.relu
  let tanh = Rune.tanh
  let sigmoid = Rune.sigmoid
  let elu alpha = Rune.elu ~alpha
  let leaky_relu negative_slope = Rune.leaky_relu ~negative_slope
  let softplus _beta = Rune.softplus (* TODO: Rune’s softplus lacks beta *)
end

module Linear = struct
  type ('layout, 'dev) params = {
    w : ('layout, 'dev) tensor;
    b : ('layout, 'dev) tensor option;
  }

  let glorot_uniform ~rng ~dtype ~din ~dout =
    let limit = sqrt (6.0 /. float_of_int (din + dout)) in
    let u01 = Rng.uniform rng ~dtype ~shape:[| din; dout |] in
    let scale = Rune.scalar dtype (2.0 *. limit)
    and shift = Rune.scalar dtype limit in
    Rune.(sub (mul u01 scale) shift)

  let init ~rng ?(use_bias = true) ~dtype ~device in_features out_features =
    let w_cpu =
      glorot_uniform ~rng ~dtype ~din:in_features ~dout:out_features
    in
    let w = Rune.move device w_cpu in
    let b =
      if use_bias then
        Some (Rune.zeros dtype [| out_features |] |> Rune.move device)
      else None
    in
    { w; b }

  let forward { w; b } x =
    let z = Rune.matmul x w in
    match b with Some b -> Rune.add z b | None -> z

  let update ~lr { w; b } { w = dw; b = db_opt } =
    let dev = Rune.device w in
    let dtype = Rune.dtype w in
    let lr_t = Rune.scalar dtype lr |> Rune.move dev in
    let w' = Rune.sub w (Rune.mul lr_t dw) in
    let b' =
      match (b, db_opt) with
      | Some b, Some db -> Some (Rune.sub b (Rune.mul lr_t db))
      | _ -> b
    in
    { w = w'; b = b' }

  let params { w; b } = match b with Some b -> [ w; b ] | None -> [ w ]
end

module Optimizer = struct
  module Sgd = struct
    type config = { lr : float }

    let create ~lr = { lr }
    let init _params = ()

    let update (type layout dev) (config : config)
        (grads : (layout, dev) tensor list) : unit * (layout, dev) tensor list =
      let updates =
        List.map
          (fun (g : (layout, dev) tensor) ->
            let dtype = Rune.dtype g in
            let neg_lr = Rune.scalar dtype (-.config.lr) in
            Rune.mul g neg_lr)
          grads
      in
      ((), updates)
  end

  type 'op t = Sgd : Sgd.config -> [ `sgd ] t
  type 'op state = Sgd_state : unit -> 'op state

  let sgd lr = Sgd (Sgd.create ~lr)

  let init (type op) (optimizer : op t) (params : ('layout, 'dev) tensor list) :
      op state =
    match optimizer with
    | Sgd _config ->
        Sgd.init params;
        Sgd_state ()

  let update (type op) (optimizer : op t) (state : op state) grads :
      op state * ('layout, 'dev) tensor list =
    match (optimizer, state) with
    | Sgd config, Sgd_state () ->
        let (), updates = Sgd.update config grads in
        (Sgd_state (), updates)

  let apply_updates params updates =
    List.map2 (fun p u -> Rune.add p u) params updates
end

module Loss = struct
  type ('layout, 'dev) t = ('layout, 'dev) tensor -> ('layout, 'dev) tensor

  let sigmoid_binary_cross_entropy logits labels =
    let dtype = Rune.dtype logits in
    let one = Rune.scalar dtype 1.0 in
    let log_sig = Rune.log_sigmoid logits in
    let log_sig_neg = Rune.log_sigmoid (Rune.neg logits) in
    let term1 = Rune.mul labels log_sig in
    let term2 = Rune.mul (Rune.sub one labels) log_sig_neg in
    let loss_per_example = Rune.neg (Rune.add term1 term2) in
    Rune.mean loss_per_example
end
