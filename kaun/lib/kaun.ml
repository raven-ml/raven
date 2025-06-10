(* Core types *)
type ('layout, 'dev) tensor = (float, 'layout) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype
type 'dev device = 'dev Rune.device

(* Parameter tree to represent model parameters hierarchically *)
type ('layout, 'dev) ptree =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) ptree list
  | Record of (string * ('layout, 'dev) ptree) list

(* Params type is now parameterized by layout and device *)
type ('layout, 'dev) params = ('layout, 'dev) ptree

(* Rngs module *)
module Rngs = struct
  type t = int

  let create ~seed () = seed

  let split t =
    let new_seed1 = (t * 1664525) + 1013904223 in
    let new_seed2 = (new_seed1 * 1664525) + 1013904223 in
    (new_seed1, new_seed2)
end

(* Model type - stores the structure and initialization logic *)
type model =
  | Model : {
      init :
        'layout 'dev.
        rngs:Rngs.t -> ('layout, 'dev) tensor -> ('layout, 'dev) params;
      apply :
        'layout 'dev.
        ('layout, 'dev) params ->
        training:bool ->
        ('layout, 'dev) tensor ->
        ('layout, 'dev) tensor;
    }
      -> model

let init (Model m) ~rngs x = m.init ~rngs x
let apply (Model m) params ~training x = m.apply params ~training x

(* Helper functions for ptree manipulation *)
let split_at n lst =
  let rec aux i acc = function
    | [] -> (List.rev acc, [])
    | h :: t as l ->
        if i = 0 then (List.rev acc, l) else aux (i - 1) (h :: acc) t
  in
  aux n [] lst

let rec flatten_ptree : type layout dev.
    (layout, dev) ptree ->
    (layout, dev) tensor list
    * ((layout, dev) tensor list -> (layout, dev) ptree) = function
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

(* Metrics module *)
module Metrics = struct
  type metric_type = Avg | Sum | Accuracy
  type metric = { name : string; metric_type : metric_type }

  type t = {
    metrics : metric list;
    mutable values : (string * float) list;
    mutable counts : (string * int) list;
  }

  let avg name = { name; metric_type = Avg }
  let sum name = { name; metric_type = Sum }
  let accuracy name = { name; metric_type = Accuracy }

  let create metrics =
    let values = List.map (fun m -> (m.name, 0.0)) metrics in
    let counts = List.map (fun m -> (m.name, 0)) metrics in
    { metrics; values; counts }

  let update t ?loss ?logits ?labels () =
    (* Update loss metric if provided *)
    (match loss with
    | Some loss_tensor ->
        let loss_val = Rune.unsafe_get [] loss_tensor in
        t.values <-
          List.map
            (fun (n, v) -> if n = "loss" then (n, v +. loss_val) else (n, v))
            t.values;
        t.counts <-
          List.map
            (fun (n, c) -> if n = "loss" then (n, c + 1) else (n, c))
            t.counts
    | None -> ());

    (* Update accuracy metric if both logits and labels are provided *)
    match (logits, labels) with
    | Some logits_tensor, Some labels_tensor ->
        let predictions = Rune.argmax logits_tensor ~axis:(-1) in
        (* Labels should be class indices *)
        let labels_int = Rune.cast (Rune.dtype predictions) labels_tensor in
        let correct = Rune.equal predictions labels_int in
        let accuracy_val =
          Rune.unsafe_get []
            (Rune.mean (Rune.cast (Rune.dtype logits_tensor) correct))
        in
        t.values <-
          List.map
            (fun (n, v) ->
              if n = "accuracy" then (n, v +. accuracy_val) else (n, v))
            t.values;
        t.counts <-
          List.map
            (fun (n, c) -> if n = "accuracy" then (n, c + 1) else (n, c))
            t.counts
    | _ -> ()

  let compute t =
    List.map2
      (fun metric (name, total) ->
        let count = List.assoc name t.counts in
        match metric.metric_type with
        | Avg -> (name, if count = 0 then 0.0 else total /. float_of_int count)
        | Sum -> (name, total)
        | Accuracy ->
            (name, if count = 0 then 0.0 else total /. float_of_int count))
      t.metrics t.values

  let get t name =
    let total = List.assoc name t.values in
    let count = List.assoc name t.counts in
    let metric = List.find (fun m -> m.name = name) t.metrics in
    match metric.metric_type with
    | Avg | Accuracy -> if count = 0 then 0.0 else total /. float_of_int count
    | Sum -> total

  let reset t =
    t.values <- List.map (fun (n, _) -> (n, 0.0)) t.values;
    t.counts <- List.map (fun (n, _) -> (n, 0)) t.counts
end

(* Dataset module based on Seq *)
module Dataset = struct
  type 'a t = 'a Seq.t

  let of_xy (x, y) =
    let n_samples = (Rune.shape x).(0) in
    let indices = Seq.ints 0 |> Seq.take n_samples in
    Seq.map
      (fun i ->
        let x_i = Rune.get [ i ] x in
        let y_i = Rune.get [ i ] y in
        (x_i, y_i))
      indices

  let map f ds = Seq.map f ds

  (* Batch function for tensor pairs *)
  let batch_xy batch_size ds =
    let rec batch_seq seq =
      if Seq.is_empty seq then Seq.empty
      else
        let batch = Seq.take batch_size seq in
        let rest = Seq.drop batch_size seq in
        let batch_list = List.of_seq batch in
        if batch_list = [] then Seq.empty
        else
          match batch_list with
          | [] -> batch_seq rest
          | samples ->
              let xs = List.map fst samples in
              let ys = List.map snd samples in
              let x_batch = Rune.stack xs ~axis:0 in
              let y_batch = Rune.stack ys ~axis:0 in
              Seq.cons (x_batch, y_batch) (batch_seq rest)
    in
    batch_seq ds

  (* Generic batch function - for now just returns the dataset unchanged *)
  (* TODO: Implement proper polymorphic batching *)
  let batch _batch_size ds = ds

  let shuffle ?seed ds =
    let rng =
      match seed with
      | Some s -> Random.State.make [| s |]
      | None -> Random.State.make_self_init ()
    in
    let array = Array.of_seq ds in
    let n = Array.length array in
    (* Fisher-Yates shuffle *)
    for i = n - 1 downto 1 do
      let j = Random.State.int rng (i + 1) in
      let tmp = array.(i) in
      array.(i) <- array.(j);
      array.(j) <- tmp
    done;
    Array.to_seq array

  let iter f ds = Seq.iter f ds
  let length ds = Seq.length ds
end

(* Loss module *)
module Loss = struct
  let softmax_cross_entropy logits labels =
    (* Assumes labels are one-hot encoded *)
    let max_logits = Rune.max logits ~axes:[| -1 |] ~keepdims:true in
    let exp_logits = Rune.exp (Rune.sub logits max_logits) in
    let sum_exp = Rune.sum exp_logits ~axes:[| -1 |] ~keepdims:true in
    let log_softmax =
      Rune.sub logits (Rune.add max_logits (Rune.log sum_exp))
    in
    let loss =
      Rune.neg (Rune.sum (Rune.mul labels log_softmax) ~axes:[| -1 |])
    in
    Rune.mean loss

  let softmax_cross_entropy_with_indices logits indices =
    (* Convert indices to one-hot encoding *)
    let indices_int = Rune.cast Rune.int32 indices in
    let num_classes = (Rune.shape logits).(1) in
    let one_hot = Rune.one_hot ~num_classes indices_int in
    let one_hot_float = Rune.cast (Rune.dtype logits) one_hot in
    softmax_cross_entropy logits one_hot_float

  let binary_cross_entropy logits labels =
    let dtype = Rune.dtype logits in
    let dev = Rune.device logits in
    let one = Rune.scalar dev dtype 1.0 in
    let log_sig = Rune.log_sigmoid logits in
    let log_sig_neg = Rune.log_sigmoid (Rune.neg logits) in
    let term1 = Rune.mul labels log_sig in
    let term2 = Rune.mul (Rune.sub one labels) log_sig_neg in
    let loss_per_example = Rune.neg (Rune.add term1 term2) in
    Rune.mean loss_per_example

  let mse predictions targets =
    let diff = Rune.sub predictions targets in
    let squared = Rune.mul diff diff in
    Rune.mean squared

  let mae predictions targets =
    let diff = Rune.sub predictions targets in
    let abs_diff = Rune.abs diff in
    Rune.mean abs_diff
end

(* Initializer module *)
module Initializer = struct
  type t =
    | Constant of float
    | GlorotUniform of { in_axis : int; out_axis : int }
    | Normal of { mean : float; std : float }

  let constant value = Constant value
  let glorot_uniform ~in_axis ~out_axis = GlorotUniform { in_axis; out_axis }
  let normal ~mean ~std = Normal { mean; std }

  (* Helper function to apply an initializer - not exposed in interface *)
  let apply init rng shape dev dtype =
    match init with
    | Constant value -> Rune.full dev dtype shape value
    | GlorotUniform { in_axis; out_axis } ->
        let fan_in = shape.(in_axis) in
        let fan_out = shape.(out_axis) in
        let limit = sqrt (6.0 /. float_of_int (fan_in + fan_out)) in
        let u01 = Rune.rand dev dtype ~seed:rng shape in
        let scale = Rune.scalar dev dtype (2.0 *. limit) in
        let shift = Rune.scalar dev dtype limit in
        Rune.(sub (mul u01 scale) shift)
    | Normal { mean; std } ->
        let z = Rune.randn dev dtype ~seed:rng shape in
        let z = Rune.mul z (Rune.scalar dev dtype std) in
        Rune.add z (Rune.scalar dev dtype mean)
end

(* Layer module *)
module Layer = struct
  let conv2d ~in_channels ~out_channels ?(kernel_size = (3, 3)) ~rngs () =
    let rng1, _rng2 = Rngs.split rngs in
    let kh, kw = kernel_size in
    Model
      {
        init =
          (fun (type l d) ~rngs:_ (x : (l, d) tensor) ->
            let dev = Rune.device x in
            let dtype = Rune.dtype x in
            let fan_in = in_channels * kh * kw in
            let fan_out = out_channels * kh * kw in
            let limit = sqrt (6.0 /. float_of_int (fan_in + fan_out)) in
            let weight_shape = [| out_channels; in_channels; kh; kw |] in
            let w = Rune.rand dev dtype ~seed:rng1 weight_shape in
            let w =
              Rune.sub
                (Rune.mul w (Rune.scalar dev dtype (2.0 *. limit)))
                (Rune.scalar dev dtype limit)
            in
            let b = Rune.zeros dev dtype [| out_channels |] in
            Record [ ("weight", Tensor w); ("bias", Tensor b) ]);
        apply =
          (fun (type l d)
            (params : (l, d) params)
            ~training:_
            (x : (l, d) tensor)
          ->
            match params with
            | Record fields ->
                (* Handle fields in any order *)
                let w =
                  match List.assoc_opt "weight" fields with
                  | Some (Tensor t) -> t
                  | _ -> failwith "conv2d: missing or invalid weight parameter"
                in
                let b =
                  match List.assoc_opt "bias" fields with
                  | Some (Tensor t) -> t
                  | _ -> failwith "conv2d: missing or invalid bias parameter"
                in
                let conv =
                  Rune.convolve2d x w ~stride:(1, 1) ~padding_mode:`Same
                in
                let b_reshaped = Rune.reshape [| 1; out_channels; 1; 1 |] b in
                Rune.add conv b_reshaped
            | _ -> failwith "conv2d: invalid params structure");
      }

  let linear ~in_features ~out_features ?weight_init ?bias_init ~rngs () =
    let rng1, rng2 = Rngs.split rngs in
    let weight_init =
      match weight_init with
      | Some init -> init
      | None -> Initializer.glorot_uniform ~in_axis:0 ~out_axis:1
    in
    let bias_init =
      match bias_init with
      | Some init -> init
      | None -> Initializer.constant 0.0
    in
    Model
      {
        init =
          (fun (type layout dev)
            ~rngs:_
            (x : (layout, dev) tensor)
            :
            (layout, dev) params
          ->
            let dev = Rune.device x in
            let dtype = Rune.dtype x in
            let w =
              Initializer.apply weight_init rng1
                [| in_features; out_features |]
                dev dtype
            in
            let b =
              Initializer.apply bias_init rng2 [| out_features |] dev dtype
            in
            Record [ ("weight", Tensor w); ("bias", Tensor b) ]);
        apply =
          (fun (type l d)
            (params : (l, d) params)
            ~training:_
            (x : (l, d) tensor)
          ->
            match params with
            | Record fields ->
                (* Handle fields in any order *)
                let w =
                  match List.assoc_opt "weight" fields with
                  | Some (Tensor t) -> t
                  | _ -> failwith "linear: missing or invalid weight parameter"
                in
                let b =
                  match List.assoc_opt "bias" fields with
                  | Some (Tensor t) -> t
                  | _ -> failwith "linear: missing or invalid bias parameter"
                in
                let z = Rune.matmul x w in
                Rune.add z b
            | _ -> failwith "linear: invalid params structure");
      }

  let dropout ~rate ~rngs () =
    Model
      {
        init = (fun ~rngs:_ _x -> List []);
        apply =
          (fun _params ~training x ->
            if training && rate > 0.0 then
              (* Generate dropout mask *)
              let dev = Rune.device x in
              let dtype = Rune.dtype x in
              let shape = Rune.shape x in
              let seed = Rngs.split rngs |> fst in
              let mask = Rune.rand dev dtype ~seed shape in
              let keep_prob = 1.0 -. rate in
              let threshold = Rune.scalar dev dtype keep_prob in
              let binary_mask = Rune.less mask threshold in
              let binary_mask_float = Rune.cast dtype binary_mask in
              let scale = Rune.scalar dev dtype (1.0 /. keep_prob) in
              (* Apply mask and scale *)
              Rune.mul x (Rune.mul binary_mask_float scale)
            else x);
      }

  let batch_norm ~num_features ~rngs () =
    let _rng1, _rng2 = Rngs.split rngs in
    Model
      {
        init =
          (fun ~rngs:_ x ->
            let dev = Rune.device x in
            let dtype = Rune.dtype x in
            (* Initialize scale (gamma) to 1 and bias (beta) to 0 *)
            let scale = Rune.ones dev dtype [| num_features |] in
            let bias = Rune.zeros dev dtype [| num_features |] in
            (* We don't track running stats in this simple implementation *)
            Record [ ("scale", Tensor scale); ("bias", Tensor bias) ]);
        apply =
          (fun params ~training:_ x ->
            match params with
            | Record fields ->
                (* Handle fields in any order *)
                let scale =
                  match List.assoc_opt "scale" fields with
                  | Some (Tensor t) -> t
                  | _ ->
                      failwith "batch_norm: missing or invalid scale parameter"
                in
                let bias =
                  match List.assoc_opt "bias" fields with
                  | Some (Tensor t) -> t
                  | _ ->
                      failwith "batch_norm: missing or invalid bias parameter"
                in
                (* Compute batch statistics *)
                let axes =
                  match Array.length (Rune.shape x) with
                  | 2 -> [| 0 |] (* (batch, features) *)
                  | 4 -> [| 0; 2; 3 |] (* (batch, channels, height, width) *)
                  | _ -> [| 0 |]
                  (* Default to first axis *)
                in
                let mean = Rune.mean x ~axes ~keepdims:true in
                let variance = Rune.var x ~axes ~keepdims:true in
                let eps = 1e-5 in
                let dtype = Rune.dtype x in
                let dev = Rune.device x in
                let epsilon = Rune.scalar dev dtype eps in
                (* Normalize *)
                let x_normalized =
                  Rune.div (Rune.sub x mean)
                    (Rune.sqrt (Rune.add variance epsilon))
                in
                (* Scale and shift *)
                let scale_shape =
                  match Array.length (Rune.shape x) with
                  | 2 -> [| 1; num_features |]
                  | 4 -> [| 1; num_features; 1; 1 |]
                  | _ -> [| 1; num_features |]
                in
                let scale_reshaped = Rune.reshape scale_shape scale in
                let bias_reshaped = Rune.reshape scale_shape bias in
                Rune.add (Rune.mul x_normalized scale_reshaped) bias_reshaped
            | _ -> failwith "batch_norm: invalid params structure");
      }

  let max_pool2d ~kernel_size ?stride () =
    let stride = match stride with Some s -> s | None -> kernel_size in
    Model
      {
        init = (fun ~rngs:_ _x -> List []);
        apply =
          (fun _params ~training:_ x ->
            let pooled, _ = Rune.max_pool2d x ~kernel_size ~stride in
            pooled);
      }

  let avg_pool2d ~kernel_size ?stride () =
    let stride = match stride with Some s -> s | None -> kernel_size in
    Model
      {
        init = (fun ~rngs:_ _x -> List []);
        apply =
          (fun _params ~training:_ x -> Rune.avg_pool2d x ~kernel_size ~stride);
      }

  let flatten () =
    Model
      {
        init = (fun ~rngs:_ _x -> List []);
        apply =
          (fun _params ~training:_ x ->
            let shape = Rune.shape x in
            let batch_size = shape.(0) in
            let flat_size =
              Array.fold_left ( * ) 1
                (Array.sub shape 1 (Array.length shape - 1))
            in
            Rune.reshape [| batch_size; flat_size |] x);
      }

  let relu () =
    Model
      {
        init = (fun ~rngs:_ _x -> List []);
        apply = (fun _params ~training:_ x -> Rune.relu x);
      }

  let sigmoid () =
    Model
      {
        init = (fun ~rngs:_ _x -> List []);
        apply = (fun _params ~training:_ x -> Rune.sigmoid x);
      }

  let tanh () =
    Model
      {
        init = (fun ~rngs:_ _x -> List []);
        apply = (fun _params ~training:_ x -> Rune.tanh x);
      }

  let sequential models =
    Model
      {
        init =
          (fun ~rngs x ->
            (* Initialize each layer sequentially, threading the output shape
               through *)
            let rec init_layers models x acc =
              match models with
              | [] -> List (List.rev acc)
              | Model m :: rest ->
                  let params = m.init ~rngs x in
                  (* Apply this layer to get output shape for next layer *)
                  let x' = m.apply params ~training:false x in
                  init_layers rest x' (params :: acc)
            in
            init_layers models x []);
        apply =
          (fun params ~training x ->
            match params with
            | List param_list ->
                (* Apply each layer in sequence *)
                let rec apply_layers models params x =
                  match (models, params) with
                  | [], [] -> x
                  | Model m :: ms, p :: ps ->
                      let x' = m.apply p ~training x in
                      apply_layers ms ps x'
                  | _ -> failwith "sequential: mismatched models and params"
                in
                apply_layers models param_list x
            | _ -> failwith "sequential: invalid params structure");
      }
end

(* Optimizer module *)
module Optimizer = struct
  type transform =
    | SGD of { lr : float; momentum : float option }
    | Adam of { lr : float; beta1 : float; beta2 : float; eps : float }
    | AdamW of {
        lr : float;
        beta1 : float;
        beta2 : float;
        eps : float;
        weight_decay : float;
      }

  type ('layout, 'dev) state = {
    m_tensors : ('layout, 'dev) tensor list;
    v_tensors : ('layout, 'dev) tensor list;
  }

  type ('layout, 'dev) t = {
    transform : transform;
    mutable state : ('layout, 'dev) state option;
    mutable step : int;
  }

  let sgd ~lr ?momentum () = SGD { lr; momentum }

  let adam ~lr ?(beta1 = 0.9) ?(beta2 = 0.999) ?(eps = 1e-8) () =
    Adam { lr; beta1; beta2; eps }

  let adamw ~lr ?(beta1 = 0.9) ?(beta2 = 0.999) ?(eps = 1e-8)
      ?(weight_decay = 0.01) () =
    AdamW { lr; beta1; beta2; eps; weight_decay }

  let create transform = { transform; state = None; step = 0 }

  let rec apply_updates_inplace : type a b. (a, b) ptree -> (a, b) ptree -> unit
      =
   fun params updates ->
    match (params, updates) with
    | Tensor t, Tensor u -> ignore (Rune.isub t u)
    | List ps, List us -> List.iter2 apply_updates_inplace ps us
    | Record ps, Record us ->
        let sorted_ps =
          List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2) ps
        in
        let sorted_us =
          List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2) us
        in
        List.iter2
          (fun (k1, p) (k2, u) ->
            assert (k1 = k2);
            apply_updates_inplace p u)
          sorted_ps sorted_us
    | _ -> failwith "Mismatched parameter structure"

  let update opt params grads =
    opt.step <- opt.step + 1;
    let params_tensors, rebuild_params = flatten_ptree params in
    let grads_tensors, _ = flatten_ptree grads in

    match opt.transform with
    | SGD { lr; momentum } ->
        (* Handle momentum if specified *)
        let updates =
          match momentum with
          | None ->
              (* Simple SGD without momentum *)
              List.map
                (fun g ->
                  let dev = Rune.device g in
                  let dt = Rune.dtype g in
                  Rune.mul g (Rune.scalar dev dt lr))
                grads_tensors
          | Some momentum_val ->
              (* SGD with momentum *)
              let state =
                match opt.state with
                | None ->
                    (* Initialize velocity tensors *)
                    let v_tensors = List.map Rune.zeros_like params_tensors in
                    let s = { m_tensors = v_tensors; v_tensors = [] } in
                    opt.state <- Some s;
                    s
                | Some s -> s
              in

              (* Update velocities and compute updates *)
              let new_velocities, updates =
                List.fold_left2
                  (fun (v_acc, u_acc) g v_old ->
                    let dev = Rune.device g in
                    let dt = Rune.dtype g in
                    (* v = momentum * v_old + lr * grad *)
                    let v_new =
                      Rune.(
                        add
                          (mul (scalar dev dt momentum_val) v_old)
                          (mul (scalar dev dt lr) g))
                    in
                    (v_new :: v_acc, v_new :: u_acc))
                  ([], []) grads_tensors state.m_tensors
              in

              (* Update state with new velocities *)
              opt.state <-
                Some { m_tensors = List.rev new_velocities; v_tensors = [] };
              List.rev updates
        in
        apply_updates_inplace params (rebuild_params updates)
    | Adam { lr; beta1; beta2; eps } ->
        (* Initialize state if needed *)
        let state =
          match opt.state with
          | None ->
              let m_tensors = List.map Rune.zeros_like params_tensors in
              let v_tensors = List.map Rune.zeros_like params_tensors in
              let s = { m_tensors; v_tensors } in
              opt.state <- Some s;
              s
          | Some s -> s
        in

        let t_f = float_of_int opt.step in
        let bc1 = 1. -. (beta1 ** t_f) in
        let bc2 = 1. -. (beta2 ** t_f) in

        let new_m_tensors, new_v_tensors, updates =
          List.fold_left2
            (fun (m_acc, v_acc, u_acc) (p, g) (m_old, v_old) ->
              let dev = Rune.device p in
              let dt = Rune.dtype p in

              (* Update biased first moment estimate *)
              let m_new =
                Rune.(
                  add
                    (mul (scalar dev dt beta1) m_old)
                    (mul (scalar dev dt (1. -. beta1)) g))
              in

              (* Update biased second moment estimate *)
              let v_new =
                Rune.(
                  add
                    (mul (scalar dev dt beta2) v_old)
                    (mul (scalar dev dt (1. -. beta2)) (mul g g)))
              in

              (* Bias correction *)
              let m_hat = Rune.div m_new (Rune.scalar dev dt bc1) in
              let v_hat = Rune.div v_new (Rune.scalar dev dt bc2) in

              (* Compute update *)
              let update =
                Rune.(
                  mul (scalar dev dt lr)
                    (div m_hat (add (sqrt v_hat) (scalar dev dt eps))))
              in

              (m_new :: m_acc, v_new :: v_acc, update :: u_acc))
            ([], [], [])
            (List.combine params_tensors grads_tensors)
            (List.combine state.m_tensors state.v_tensors)
        in

        opt.state <-
          Some
            {
              m_tensors = List.rev new_m_tensors;
              v_tensors = List.rev new_v_tensors;
            };
        apply_updates_inplace params (rebuild_params (List.rev updates))
    | AdamW { lr; beta1; beta2; eps; weight_decay } ->
        (* Initialize state if needed *)
        let state =
          match opt.state with
          | None ->
              let m_tensors = List.map Rune.zeros_like params_tensors in
              let v_tensors = List.map Rune.zeros_like params_tensors in
              let s = { m_tensors; v_tensors } in
              opt.state <- Some s;
              s
          | Some s -> s
        in

        let t_f = float_of_int opt.step in
        let bc1 = 1. -. (beta1 ** t_f) in
        let bc2 = 1. -. (beta2 ** t_f) in

        let new_m_tensors, new_v_tensors, updates =
          List.fold_left2
            (fun (m_acc, v_acc, u_acc) (p, g) (m_old, v_old) ->
              let dev = Rune.device p in
              let dt = Rune.dtype p in

              (* Update biased first moment estimate *)
              let m_new =
                Rune.(
                  add
                    (mul (scalar dev dt beta1) m_old)
                    (mul (scalar dev dt (1. -. beta1)) g))
              in

              (* Update biased second moment estimate *)
              let v_new =
                Rune.(
                  add
                    (mul (scalar dev dt beta2) v_old)
                    (mul (scalar dev dt (1. -. beta2)) (mul g g)))
              in

              (* Bias correction *)
              let m_hat = Rune.div m_new (Rune.scalar dev dt bc1) in
              let v_hat = Rune.div v_new (Rune.scalar dev dt bc2) in

              (* Compute update with weight decay applied directly to
                 parameters *)
              let adam_update =
                Rune.(
                  mul (scalar dev dt lr)
                    (div m_hat (add (sqrt v_hat) (scalar dev dt eps))))
              in

              (* Add weight decay term: lr * weight_decay * param *)
              let decay_update =
                Rune.mul (Rune.scalar dev dt (lr *. weight_decay)) p
              in

              (* Total update = adam_update + decay_update *)
              let total_update = Rune.add adam_update decay_update in

              (m_new :: m_acc, v_new :: v_acc, total_update :: u_acc))
            ([], [], [])
            (List.combine params_tensors grads_tensors)
            (List.combine state.m_tensors state.v_tensors)
        in

        opt.state <-
          Some
            {
              m_tensors = List.rev new_m_tensors;
              v_tensors = List.rev new_v_tensors;
            };
        apply_updates_inplace params (rebuild_params (List.rev updates))
end
