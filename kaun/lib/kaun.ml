type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype
type 'dev device = 'dev Rune.device

(* Parameter tree - alias for Ptree.t *)
type ('layout, 'dev) params = ('layout, 'dev) Ptree.t =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) params list
  | Record of ('layout, 'dev) params Ptree.Record.t

type module_ = Module.t = {
  init :
    'layout 'dev.
    rngs:Rune.Rng.key -> ('layout, 'dev) tensor -> ('layout, 'dev) Ptree.t;
  apply :
    'layout 'dev.
    ('layout, 'dev) Ptree.t ->
    training:bool ->
    ?rngs:Rune.Rng.key ->
    ('layout, 'dev) tensor ->
    ('layout, 'dev) tensor;
}

let init m ~rngs x =
  let result = m.init ~rngs x in
  result

let apply m params ~training ?rngs x = m.apply params ~training ?rngs x

(* Helper functions for ptree manipulation *)
let split_at n lst =
  let rec aux i acc = function
    | [] -> (List.rev acc, [])
    | h :: t as l ->
        if i = 0 then (List.rev acc, l) else aux (i - 1) (h :: acc) t
  in
  aux n [] lst

let rec flatten_ptree : type layout dev.
    (layout, dev) params ->
    (layout, dev) tensor list
    * ((layout, dev) tensor list -> (layout, dev) params) = function
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
      (* CRITICAL FIX: Sort record fields to ensure consistent ordering *)
      let sorted_r =
        List.sort
          (fun (k1, _) (k2, _) -> String.compare k1 k2)
          (Ptree.Record.bindings r)
      in
      let pairs = List.map (fun (k, pt) -> (k, flatten_ptree pt)) sorted_r in
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
        Record (Ptree.Record.of_list (aux tensors [] pairs))
      in
      (tensors, rebuild)

let value_and_grad f params =
  let tensors, rebuild = flatten_ptree params in
  let f_on_list ts =
    let params' = rebuild ts in
    f params'
  in
  let value, grads_list = Rune.value_and_grads f_on_list tensors in
  let grad_ptree = rebuild grads_list in
  (value, grad_ptree)

let grad f params =
  let _, grads = value_and_grad f params in
  grads

  module Module = Module
module Metrics = Metrics
module Loss = Loss
module Initializer = Initializers
module Layer = Layer
module Checkpoint = Kaun_checkpoint
module Ptree = Ptree
module Optimizer = Kaun_optim
module Activations = Activations
module Transformers = Kaun_transformers
module Dataset = Kaun_dataset
