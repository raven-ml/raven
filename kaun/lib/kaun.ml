(* Core types *)
type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype
type 'dev device = 'dev Rune.device

(* Parameter tree to represent model parameters hierarchically *)
type ('layout, 'dev) params =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) params list
  | Record of (string * ('layout, 'dev) params) list

(* Rngs module *)
module Rngs = struct
  type t = int

  let create ~seed () =
    abs seed land 0x7FFFFFFF (* Ensure positive 31-bit int *)

  let split t =
    (* MurmurHash-inspired integer hash for better distribution *)
    let hash x =
      (* Use Int32 operations to handle large constants on 32-bit systems *)
      let open Int32 in
      let x = of_int x in
      let x = logxor x (shift_right_logical x 16) in
      let x = mul x 0x85ebca6bl in
      let x = logxor x (shift_right_logical x 13) in
      let x = mul x 0xc2b2ae35l in
      let x = logxor x (shift_right_logical x 16) in
      to_int (logand x 0x7FFFFFFFl)
      (* Ensure positive 31-bit int *)
    in
    let new_seed1 = hash ((t * 2) + 1) in
    let new_seed2 = hash ((t * 2) + 2) in
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
        ?rngs:Rngs.t ->
        ('layout, 'dev) tensor ->
        ('layout, 'dev) tensor;
    }
      -> model

let init (Model m) ~rngs x =
  let result = m.init ~rngs x in
  result

let apply (Model m) params ~training ?rngs x = m.apply params ~training ?rngs x

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
        List.sort (fun (k1, _) (k2, _) -> String.compare k1 k2) r
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

  let of_xy (x, y) = Seq.return (x, y)
  let map f ds = Seq.map f ds

  (* Batch function for tensor pairs *)
  let batch_xy batch_size ds =
    Seq.flat_map
      (fun (x, y) ->
        (* Check if this is a full tensor dataset *)
        if Array.length (Rune.shape x) > 1 then
          (* This is a full dataset - create batches efficiently *)
          let n_samples = (Rune.shape x).(0) in
          let rec create_batches start =
            if start >= n_samples then Seq.empty
            else
              let end_idx = min (start + batch_size) n_samples in
              let x_batch = Rune.slice [ R [ start; end_idx ] ] x in
              let y_batch = Rune.slice [ R [ start; end_idx ] ] y in
              (* Ensure batches are contiguous for operations like flatten *)
              let x_batch =
                if Rune.is_c_contiguous x_batch then x_batch
                else
                  let result = Rune.contiguous x_batch in
                  result
              in
              let y_batch =
                if Rune.is_c_contiguous y_batch then y_batch
                else Rune.contiguous y_batch
              in
              Seq.cons (x_batch, y_batch) (create_batches end_idx)
          in
          create_batches 0
        else Seq.return (x, y))
      ds

  (* Generic batch function - for now just returns the dataset unchanged *)
  (* TODO: Implement proper polymorphic batching *)
  let batch _batch_size ds = ds

  let shuffle ?seed ds =
    (* For now, pass through without shuffling but maintain the structure *)
    (* TODO: Implement efficient shuffling without gather *)
    let _ = seed in
    ds

  let iter f ds = Seq.iter f ds
  let length ds = Seq.length ds

  let take n ds =
    let rec take_aux n acc seq =
      if n <= 0 then List.rev acc
      else
        match seq () with
        | Seq.Nil -> List.rev acc
        | Seq.Cons (x, rest) -> take_aux (n - 1) (x :: acc) rest
    in
    take_aux n [] ds
end

(* Loss module *)
module Loss = struct
  let softmax_cross_entropy logits labels =
    Rune.debug_with_context "softmax_cross_entropy" (fun () ->
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
        Rune.mean loss)

  let softmax_cross_entropy_with_indices logits indices =
    (* Convert indices to one-hot encoding *)
    let indices_int = Rune.cast Rune.int32 indices in
    let num_classes = (Rune.shape logits).(1) in
    let one_hot = Rune.one_hot ~num_classes indices_int in
    let one_hot_float = Rune.cast (Rune.dtype logits) one_hot in
    softmax_cross_entropy logits one_hot_float

  let binary_cross_entropy logits labels =
    Rune.debug_with_context "binary_cross_entropy" (fun () ->
        let dtype = Rune.dtype logits in
        let dev = Rune.device logits in
        let one = Rune.scalar dev dtype 1.0 in
        let log_sig = Rune.log_sigmoid logits in
        let log_sig_neg = Rune.log_sigmoid (Rune.neg logits) in
        let term1 = Rune.mul labels log_sig in
        let term2 = Rune.mul (Rune.sub one labels) log_sig_neg in
        let loss_per_example = Rune.neg (Rune.add term1 term2) in
        Rune.mean loss_per_example)

  let sigmoid_binary_cross_entropy logits labels =
    Rune.debug_with_context "sigmoid_binary_cross_entropy" (fun () ->
        let dtype = Rune.dtype logits in
        let dev = Rune.device logits in
        let one = Rune.scalar dev dtype 1.0 in
        let log_sig = Rune.log_sigmoid logits in
        let log_sig_neg = Rune.log_sigmoid (Rune.neg logits) in
        let term1 = Rune.mul labels log_sig in
        let term2 = Rune.mul (Rune.sub one labels) log_sig_neg in
        Rune.neg (Rune.add term1 term2))

  let mse predictions targets =
    Rune.debug_with_context "mse" (fun () ->
        let diff = Rune.sub predictions targets in
        let squared = Rune.mul diff diff in
        Rune.mean squared)

  let mae predictions targets =
    Rune.debug_with_context "mae" (fun () ->
        let diff = Rune.sub predictions targets in
        let abs_diff = Rune.abs diff in
        Rune.mean abs_diff)
end

(* Initializer module - wrapper around the Initializers module for backward
   compatibility *)
module Initializer = struct
  (* We build a spec instead of passing a function, as the function would be
     polymorphic on dev and layout, and we want keep the model spec device and
     layout agnostic *)
  type t =
    | Constant of float
    | Zeros
    | Ones
    | Uniform of { scale : float }
    | Normal of { mean : float; std : float }
    | TruncatedNormal of { stddev : float; lower : float; upper : float }
    | VarianceScaling of {
        scale : float;
        mode : [ `Fan_in | `Fan_out | `Fan_avg ];
        distribution : [ `Normal | `Truncated_normal | `Uniform ];
        in_axis : int;
        out_axis : int;
      }
    | GlorotUniform of { in_axis : int; out_axis : int }
    | GlorotNormal of { in_axis : int; out_axis : int }
    | HeUniform of { in_axis : int; out_axis : int }
    | HeNormal of { in_axis : int; out_axis : int }
    | LecunUniform of { in_axis : int; out_axis : int }
    | LecunNormal of { in_axis : int; out_axis : int }
    | Orthogonal of { scale : float; column_axis : int }
    | DeltaOrthogonal of { scale : float; column_axis : int }
    | UniformRange of { low : float; high : float }
    | NormalRange of { mean : float; stddev : float }

  let constant value = Constant value
  let zeros () = Zeros
  let ones () = Ones
  let uniform ?(scale = 0.01) () = Uniform { scale }
  let normal ~mean ~std = Normal { mean; std }

  let truncated_normal ?(stddev = 0.01) ?(lower = -2.0) ?(upper = 2.0) () =
    TruncatedNormal { stddev; lower; upper }

  let variance_scaling ~scale ~mode ~distribution ~in_axis ~out_axis () =
    VarianceScaling { scale; mode; distribution; in_axis; out_axis }

  let glorot_uniform ?(in_axis = -2) ?(out_axis = -1) () =
    GlorotUniform { in_axis; out_axis }

  let glorot_normal ?(in_axis = -2) ?(out_axis = -1) () =
    GlorotNormal { in_axis; out_axis }

  let xavier_uniform = glorot_uniform
  let xavier_normal = glorot_normal

  let he_uniform ?(in_axis = -2) ?(out_axis = -1) () =
    HeUniform { in_axis; out_axis }

  let he_normal ?(in_axis = -2) ?(out_axis = -1) () =
    HeNormal { in_axis; out_axis }

  let kaiming_uniform = he_uniform
  let kaiming_normal = he_normal

  let lecun_uniform ?(in_axis = -2) ?(out_axis = -1) () =
    LecunUniform { in_axis; out_axis }

  let lecun_normal ?(in_axis = -2) ?(out_axis = -1) () =
    LecunNormal { in_axis; out_axis }

  let orthogonal ?(scale = 1.0) ?(column_axis = -1) () =
    Orthogonal { scale; column_axis }

  let delta_orthogonal ?(scale = 1.0) ?(column_axis = -1) () =
    DeltaOrthogonal { scale; column_axis }

  let uniform_range ~low ~high () = UniformRange { low; high }
  let normal_range ~mean ~stddev () = NormalRange { mean; stddev }

  (* Apply function that works within locally abstract types *)
  let apply (type layout dev) init rng shape (dev : dev Rune.device)
      (dtype : (float, layout) Rune.dtype) =
    match init with
    | Constant value -> Rune.full dev dtype shape value
    | Zeros -> Initializers.zeros () rng shape dev dtype
    | Ones -> Initializers.ones () rng shape dev dtype
    | Uniform { scale } -> Initializers.uniform ~scale () rng shape dev dtype
    | Normal { mean; std } ->
        Initializers.normal_range ~mean ~stddev:std () rng shape dev dtype
    | TruncatedNormal { stddev; lower; upper } ->
        Initializers.truncated_normal_init ~stddev ~lower ~upper () rng shape
          dev dtype
    | VarianceScaling { scale; mode; distribution; in_axis; out_axis } ->
        Initializers.variance_scaling ~scale ~mode ~distribution ~in_axis
          ~out_axis () rng shape dev dtype
    | GlorotUniform { in_axis; out_axis } ->
        (* Handle edge cases to avoid exceptions in polymorphic context *)
        let rank = Array.length shape in
        if rank = 0 then
          (* Scalar - use constant 0 *)
          Rune.zeros dev dtype shape
        else if rank = 1 then
          (* 1D tensor - use uniform with appropriate scale *)
          let n = float_of_int shape.(0) in
          let scale = sqrt (3.0 /. n) in
          Initializers.uniform ~scale () rng shape dev dtype
        else
          (* 2D+ tensor - safe to use glorot *)
          Initializers.glorot_uniform ~in_axis ~out_axis () rng shape dev dtype
    | GlorotNormal { in_axis; out_axis } ->
        (* Handle edge cases to avoid exceptions in polymorphic context *)
        let rank = Array.length shape in
        if rank = 0 then
          (* Scalar - use constant 0 *)
          Rune.zeros dev dtype shape
        else if rank = 1 then
          (* 1D tensor - use normal with appropriate scale *)
          let n = float_of_int shape.(0) in
          let stddev = sqrt (1.0 /. n) in
          Initializers.normal ~stddev () rng shape dev dtype
        else
          (* 2D+ tensor - safe to use glorot *)
          Initializers.glorot_normal ~in_axis ~out_axis () rng shape dev dtype
    | HeUniform { in_axis; out_axis } ->
        Initializers.he_uniform ~in_axis ~out_axis () rng shape dev dtype
    | HeNormal { in_axis; out_axis } ->
        Initializers.he_normal ~in_axis ~out_axis () rng shape dev dtype
    | LecunUniform { in_axis; out_axis } ->
        Initializers.lecun_uniform ~in_axis ~out_axis () rng shape dev dtype
    | LecunNormal { in_axis; out_axis } ->
        Initializers.lecun_normal ~in_axis ~out_axis () rng shape dev dtype
    | Orthogonal { scale; column_axis } ->
        Initializers.orthogonal ~scale ~column_axis () rng shape dev dtype
    | DeltaOrthogonal { scale; column_axis } ->
        Initializers.delta_orthogonal ~scale ~column_axis () rng shape dev dtype
    | UniformRange { low; high } ->
        Initializers.uniform_range ~low ~high () rng shape dev dtype
    | NormalRange { mean; stddev } ->
        Initializers.normal_range ~mean ~stddev () rng shape dev dtype
end

(* Layer module *)
module Layer = struct
  let conv2d ~in_channels ~out_channels ?(kernel_size = (3, 3)) () =
    let kh, kw = kernel_size in
    Model
      {
        init =
          (fun (type l d) ~rngs (x : (l, d) tensor) ->
            Rune.debug_with_context
              (Printf.sprintf "conv2d_%dx%d_%dx%d_init" in_channels out_channels
                 kh kw) (fun () ->
                let rng1, _rng2 = Rngs.split rngs in
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
                Record [ ("weight", Tensor w); ("bias", Tensor b) ]));
        apply =
          (fun (type l d)
            (params : (l, d) params)
            ~training:_
            ?rngs:_
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
                Rune.debug_with_context
                  (Printf.sprintf "conv2d_%dx%d_%dx%d" in_channels out_channels
                     kh kw) (fun () ->
                    let conv =
                      Rune.convolve2d x w ~stride:(1, 1) ~padding_mode:`Same
                    in
                    let b_reshaped =
                      Rune.reshape [| 1; out_channels; 1; 1 |] b
                    in
                    Rune.add conv b_reshaped)
            | _ -> failwith "conv2d: invalid params structure");
      }

  let linear ~in_features ~out_features ?weight_init ?bias_init () =
    (* No ~rngs parameter at all *)
    let weight_init =
      match weight_init with
      | Some init -> init
      | None -> Initializer.glorot_uniform ~in_axis:0 ~out_axis:1 ()
    in
    let bias_init =
      match bias_init with
      | Some init -> init
      | None -> Initializer.constant 0.0
    in
    Model
      {
        init =
          (fun ~rngs x ->
            Rune.debug_with_context
              (Printf.sprintf "linear_%dx%d_init" in_features out_features)
              (fun () ->
                (* Use the rngs passed during initialization *)
                let rng1, rng2 = Rngs.split rngs in
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
                Record [ ("weight", Tensor w); ("bias", Tensor b) ]));
        apply =
          (fun (type l d)
            (params : (l, d) params)
            ~training:_
            ?rngs:_
            (x : (l, d) tensor)
          ->
            Rune.debug_with_context
              (Printf.sprintf "linear_%dx%d" in_features out_features)
              (fun () ->
                match params with
                | Record fields ->
                    let w =
                      match List.assoc_opt "weight" fields with
                      | Some (Tensor t) -> t
                      | _ ->
                          failwith "linear: missing or invalid weight parameter"
                    in
                    let b =
                      match List.assoc_opt "bias" fields with
                      | Some (Tensor t) -> t
                      | _ ->
                          failwith "linear: missing or invalid bias parameter"
                    in
                    let z = Rune.matmul x w in
                    Rune.add z b
                | _ -> failwith "linear: invalid params structure"));
      }

  let dropout ~rate () =
    Model
      {
        init = (fun ~rngs:_ _x -> List []);
        (* No params needed *)
        apply =
          (fun _params ~training ?rngs x ->
            if training && rate > 0.0 then
              match rngs with
              | Some rng ->
                  (* Generate dropout mask *)
                  let seed = fst (Rngs.split rng) in
                  let dev = Rune.device x in
                  let dtype = Rune.dtype x in
                  let shape = Rune.shape x in
                  let mask = Rune.rand dev dtype ~seed shape in
                  let keep_prob = 1.0 -. rate in
                  let threshold = Rune.scalar dev dtype keep_prob in
                  let binary_mask = Rune.less mask threshold in
                  let binary_mask_float = Rune.cast dtype binary_mask in
                  let scale = Rune.scalar dev dtype (1.0 /. keep_prob) in
                  (* Apply mask and scale *)
                  Rune.mul x (Rune.mul binary_mask_float scale)
              | None -> failwith "dropout requires RNG during training"
            else x);
      }

  let batch_norm ~num_features () =
    Model
      {
        init =
          (fun ~rngs x ->
            Rune.debug_with_context
              (Printf.sprintf "batch_norm_%d_init" num_features) (fun () ->
                let _rng1, _rng2 = Rngs.split rngs in
                let dev = Rune.device x in
                let dtype = Rune.dtype x in
                (* Initialize scale (gamma) to 1 and bias (beta) to 0 *)
                let scale = Rune.ones dev dtype [| num_features |] in
                let bias = Rune.zeros dev dtype [| num_features |] in
                (* We don't track running stats in this simple implementation *)
                Record [ ("scale", Tensor scale); ("bias", Tensor bias) ]));
        apply =
          (fun params ~training:_ ?rngs:_ x ->
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
                Rune.debug_with_context
                  (Printf.sprintf "batch_norm_%d_apply" num_features) (fun () ->
                    (* Compute mean and variance *)
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
                    Rune.add
                      (Rune.mul x_normalized scale_reshaped)
                      bias_reshaped)
            | _ -> failwith "batch_norm: invalid params structure");
      }

  let max_pool2d ~kernel_size ?stride () =
    let stride = match stride with Some s -> s | None -> kernel_size in
    Model
      {
        init = (fun ~rngs:_ _x -> List []);
        apply =
          (fun _params ~training:_ ?rngs:_ x ->
            let pooled, _ = Rune.max_pool2d x ~kernel_size ~stride in
            pooled);
      }

  let avg_pool2d ~kernel_size ?stride () =
    let stride = match stride with Some s -> s | None -> kernel_size in
    Model
      {
        init = (fun ~rngs:_ _x -> List []);
        apply =
          (fun _params ~training:_ ?rngs:_ x ->
            Rune.avg_pool2d x ~kernel_size ~stride);
      }

  let flatten () =
    Model
      {
        init = (fun ~rngs:_ _x -> List []);
        apply =
          (fun _params ~training:_ ?rngs:_ x ->
            let shape = Rune.shape x in
            let batch_size = shape.(0) in
            let flat_size =
              Array.fold_left ( * ) 1
                (Array.sub shape 1 (Array.length shape - 1))
            in
            (* Ensure tensor is contiguous before reshaping *)
            let x = if Rune.is_c_contiguous x then x else Rune.contiguous x in
            Rune.reshape [| batch_size; flat_size |] x);
      }

  let relu () =
    Model
      {
        init = (fun ~rngs:_ _x -> List []);
        apply = (fun _params ~training:_ ?rngs:_ x -> Rune.relu x);
      }

  let sigmoid () =
    Model
      {
        init = (fun ~rngs:_ _x -> List []);
        apply = (fun _params ~training:_ ?rngs:_ x -> Rune.sigmoid x);
      }

  let tanh () =
    Model
      {
        init = (fun ~rngs:_ _x -> List []);
        apply = (fun _params ~training:_ ?rngs:_ x -> Rune.tanh x);
      }

  let sequential models =
    Model
      {
        init =
          (fun ~rngs x ->
            let rec init_layers models x acc rngs_current layer_idx =
              match models with
              | [] -> List (List.rev acc)
              | Model m :: rest ->
                  let rngs_layer, rngs_rest = Rngs.split rngs_current in
                  let params = m.init ~rngs:rngs_layer x in
                  let x' = m.apply params ~training:false x in
                  init_layers rest x' (params :: acc) rngs_rest (layer_idx + 1)
            in
            init_layers models x [] rngs 1);
        apply =
          (fun params ~training ?rngs:_ x ->
            match params with
            | List param_list ->
                let rec apply_layers models params x layer_idx =
                  match (models, params) with
                  | [], [] -> x
                  | Model m :: ms, p :: ps ->
                      let x' = m.apply p ~training x in
                      apply_layers ms ps x' (layer_idx + 1)
                  | _ -> failwith "sequential: mismatched models and params"
                in
                apply_layers models param_list x 1
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

  let rec apply_updates_inplace : type a b.
      (a, b) params -> (a, b) params -> unit =
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
            (* This assertion is now much safer *)
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
                    (* v = momentum * v_old + grad *)
                    let v_new =
                      Rune.(add (mul (scalar dev dt momentum_val) v_old) g)
                    in
                    (* update = lr * v *)
                    let update = Rune.mul (Rune.scalar dev dt lr) v_new in
                    (v_new :: v_acc, update :: u_acc))
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

        let t = opt.step in
        let bc1 = 1. -. (beta1 ** float_of_int t) in
        let bc2 = 1. -. (beta2 ** float_of_int t) in

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

        let t = opt.step in
        let bc1 = 1. -. (beta1 ** float_of_int t) in
        let bc2 = 1. -. (beta2 ** float_of_int t) in

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
