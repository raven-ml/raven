type ('layout, 'dev) tensor = (float, 'layout, [ `cpu ]) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype

(* Parameter tree to represent model parameters hierarchically *)
type ('layout, 'dev) ptree =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) ptree list
  | Record of (string * ('layout, 'dev) ptree) list

type ('model, 'layout, 'dev) lens = {
  to_ptree : 'model -> ('layout, 'dev) ptree;
  of_ptree : ('layout, 'dev) ptree -> 'model;
}

let split_at n lst =
  let rec aux i acc = function
    | [] -> (List.rev acc, [])
    | h :: t as l ->
        if i = 0 then (List.rev acc, l) else aux (i - 1) (h :: acc) t
  in
  aux n [] lst

(* Updated flatten_ptree to handle parameter trees and silence pattern match
   warning *)
let rec flatten_ptree = function
  | Tensor t ->
      ( [ t ],
        function
        | [ t' ] -> Tensor t'
        | _ -> failwith "Invalid number of tensors" )
  | List l ->
      let pairs = List.map flatten_ptree l in
      let tensors = List.concat (List.map fst pairs) in
      let rebuild =
       fun tensors ->
        let rec aux tensors acc pairs =
          match pairs with
          | [] -> List.rev acc
          | (tensors_pt, rebuild_pt) :: pairs' ->
              let n = List.length tensors_pt in
              let tensors_for_pt, tensors_rest = split_at n tensors in
              let pt' = rebuild_pt tensors_for_pt in
              aux tensors_rest (pt' :: acc) pairs'
        in
        List (aux tensors [] pairs)
      in
      (tensors, rebuild)
  | Record r ->
      let pairs = List.map (fun (k, pt) -> (k, flatten_ptree pt)) r in
      let tensors =
        List.concat (List.map (fun (_, (tensors_pt, _)) -> tensors_pt) pairs)
      in
      let rebuild =
       fun tensors ->
        let rec aux tensors acc pairs =
          match pairs with
          | [] -> List.rev acc
          | (k, (tensors_pt, rebuild_pt)) :: pairs' ->
              let n = List.length tensors_pt in
              let tensors_for_pt, tensors_rest = split_at n tensors in
              let pt' = rebuild_pt tensors_for_pt in
              aux tensors_rest ((k, pt') :: acc) pairs'
        in
        Record (aux tensors [] pairs)
      in
      (tensors, rebuild)

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

module Initializer = struct
  type ('layout, 'dev) t =
    Rng.t -> int array -> 'layout dtype -> ('layout, 'dev) tensor

  let constant value = fun _rng shape dtype -> Rune.full dtype shape value

  let glorot_uniform ~in_axis ~out_axis =
   fun rng shape dtype ->
    let din = shape.(in_axis) in
    let dout = shape.(out_axis) in
    let fan_in = din in
    let fan_out = dout in
    let limit = sqrt (6.0 /. float_of_int (fan_in + fan_out)) in
    let u01 = Rng.uniform rng ~dtype ~shape in
    let scale = Rune.scalar dtype (2.0 *. limit) in
    let shift = Rune.scalar dtype limit in
    Rune.(sub (mul u01 scale) shift)
end

module Linear = struct
  type ('layout, 'dev) t = {
    w : ('layout, 'dev) tensor;
    b : ('layout, 'dev) tensor option;
  }

  let init ~rng ?(use_bias = true) ~dtype ~device in_features out_features =
    let glorot_uniform = Initializer.glorot_uniform ~in_axis:0 ~out_axis:1 in
    let w_cpu = glorot_uniform rng [| in_features; out_features |] dtype in
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

  let params { w; b } =
    match b with
    | Some b -> Record [ ("w", Tensor w); ("b", Tensor b) ]
    | None -> Record [ ("w", Tensor w) ]

  let of_ptree = function
    | Record [ ("w", Tensor w); ("b", Tensor b) ] -> { w; b = Some b }
    | Record [ ("w", Tensor w) ] -> { w; b = None }
    | _ -> failwith "Invalid param_tree for Linear"

  let lens = { to_ptree = params; of_ptree }
end

module Optimizer = struct
  module Sgd = struct
    type cfg = { lr : float }

    let make ~lr = { lr }

    let updates { lr } (grads : ('l, 'd) tensor list) =
      List.map
        (fun g ->
          let dtype = Rune.dtype g in
          let scale = Rune.scalar dtype (-.lr) in
          Rune.mul g scale)
        grads
  end

  type _ spec = Sgd : Sgd.cfg -> [ `sgd ] spec

  let sgd ~lr : [ `sgd ] spec = Sgd (Sgd.make ~lr)

  type (_, _, _, _) t =
    | Sgd : {
        cfg : Sgd.cfg;
        lens : ('m, 'l, 'd) lens;
        model : 'm ref;
      }
        -> ([ `sgd ], 'm, 'l, 'd) t

  let init ~(lens : ('m, 'l, 'd) lens) (model : 'm) (type op) (spec : op spec) :
      (op, 'm, 'l, 'd) t =
    match spec with Sgd cfg -> Sgd { cfg; lens; model = ref model }

  let update (type op m l d) (opt : (op, m, l, d) t) (grads : (l, d) ptree) :
      unit =
    match opt with
    | Sgd { cfg; lens; model } ->
        let params_pt = lens.to_ptree !model in
        let p_list, rebuild = flatten_ptree params_pt in
        let g_list, _ = flatten_ptree grads in
        let updates = Sgd.updates cfg g_list in
        let new_params = List.map2 Rune.add_inplace p_list updates in
        model := lens.of_ptree (rebuild new_params)
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

let value_and_grad ~(lens : ('model, 'l, 'd) lens)
    (f : 'model -> ('v_l, 'v_d, [ `cpu ]) Rune.t) (model : 'model) :
    ('v_l, 'v_d, [ `cpu ]) Rune.t * ('l, 'd) ptree =
  let ptree = lens.to_ptree model in
  let tensors, rebuild = flatten_ptree ptree in
  let f_on_list ts = f (lens.of_ptree (rebuild ts)) in
  let value, grads_list = Rune.value_and_grads f_on_list tensors in
  let grad_ptree = rebuild grads_list in
  (value, grad_ptree)
