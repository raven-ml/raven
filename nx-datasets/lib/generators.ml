let pi = 4. *. atan 1.

module Rng = struct
  let init_state seed =
    match seed with
    | None -> Random.State.make_self_init ()
    | Some s -> Random.State.make [| s |]

  let normal state ?(mean = 0.) ?(std = 1.) () =
    let u1 = Random.State.float state 1. in
    let u2 = Random.State.float state 1. in
    let z0 = sqrt (-2. *. log u1) *. cos (2. *. pi *. u2) in
    mean +. (std *. z0)

  let standard_normal state shape =
    let total = Array.fold_left ( * ) 1 shape in
    let data = Array.init total (fun _ -> normal state ()) in
    Nx.reshape shape (Nx.create Float32 [| total |] data)

  let uniform state ?(low = 0.) ?(high = 1.) shape =
    let total = Array.fold_left ( * ) 1 shape in
    let range = high -. low in
    let data =
      Array.init total (fun _ -> low +. Random.State.float state range)
    in
    Nx.reshape shape (Nx.create Float32 [| total |] data)

  let randint state ?(low = 0) high shape =
    let total = Array.fold_left ( * ) 1 shape in
    let data =
      Array.init total (fun _ -> low + Random.State.int state (high - low))
    in
    Nx.reshape shape (Nx.create Int32 [| total |] (Array.map Int32.of_int data))

  let shuffle state arr =
    let n = Array.length arr in
    for i = n - 1 downto 1 do
      let j = Random.State.int state (i + 1) in
      let tmp = arr.(i) in
      arr.(i) <- arr.(j);
      arr.(j) <- tmp
    done

  let take_indices tensor ~axis indices_array =
    (* Simple implementation that rebuilds array from scratch *)
    let arr = Nx.to_array tensor in
    let shape = Nx.shape tensor in
    let n_dims = Array.length shape in
    let n_indices = Array.length indices_array in

    (* Create result shape *)
    let result_shape = Array.copy shape in
    result_shape.(axis) <- n_indices;

    (* Calculate strides for original and result *)
    let orig_strides = Array.make n_dims 1 in
    let result_strides = Array.make n_dims 1 in
    for i = n_dims - 2 downto 0 do
      orig_strides.(i) <- orig_strides.(i + 1) * shape.(i + 1);
      result_strides.(i) <- result_strides.(i + 1) * result_shape.(i + 1)
    done;

    (* Calculate result size *)
    let result_size = Array.fold_left ( * ) 1 result_shape in
    let result_arr = Array.make result_size arr.(0) in

    (* Copy elements *)
    let rec copy_recursive idx_in idx_out dim =
      if dim = n_dims then (
        (* Convert multi-dimensional indices to flat index *)
        let flat_in = ref 0 in
        let flat_out = ref 0 in
        for d = 0 to n_dims - 1 do
          flat_in := !flat_in + (idx_in.(d) * orig_strides.(d));
          flat_out := !flat_out + (idx_out.(d) * result_strides.(d))
        done;
        result_arr.(!flat_out) <- arr.(!flat_in))
      else if dim = axis then
        (* Iterate through selected indices *)
        for i = 0 to n_indices - 1 do
          idx_in.(dim) <- indices_array.(i);
          idx_out.(dim) <- i;
          copy_recursive idx_in idx_out (dim + 1)
        done
      else
        (* Iterate through all indices in this dimension *)
        for i = 0 to shape.(dim) - 1 do
          idx_in.(dim) <- i;
          idx_out.(dim) <- i;
          copy_recursive idx_in idx_out (dim + 1)
        done
    in

    if n_dims > 0 then
      copy_recursive (Array.make n_dims 0) (Array.make n_dims 0) 0;

    Nx.create (Nx.dtype tensor) result_shape result_arr
end

let make_blobs ?(n_samples = 100) ?(n_features = 2) ?(centers = `N 3)
    ?(cluster_std = 1.0) ?(center_box = (-10.0, 10.0)) ?(shuffle = true)
    ?random_state () =
  let state = Rng.init_state random_state in

  let n_centers, center_coords =
    match centers with
    | `N n ->
        let low, high = center_box in
        (n, Rng.uniform state ~low ~high [| n; n_features |])
    | `Array arr ->
        let shape = Nx.shape arr in
        (shape.(0), arr)
  in

  let samples_per_center = n_samples / n_centers in
  let extra_samples = n_samples mod n_centers in

  let x_list = ref [] in
  let y_list = ref [] in

  for i = 0 to n_centers - 1 do
    let n =
      if i < extra_samples then samples_per_center + 1 else samples_per_center
    in
    let center = Nx.slice [ I i; R [] ] center_coords in
    let center = Nx.reshape [| n_features |] center in

    let samples = Rng.standard_normal state [| n; n_features |] in
    let samples = Nx.mul_s samples cluster_std in
    let samples = Nx.add samples (Nx.broadcast_to [| n; n_features |] center) in

    x_list := samples :: !x_list;

    let labels = Nx.full Int32 [| n |] (Int32.of_int i) in
    y_list := labels :: !y_list
  done;

  let x = Nx.concatenate ~axis:0 (List.rev !x_list) in
  let y = Nx.concatenate ~axis:0 (List.rev !y_list) in

  if shuffle then (
    let indices = Array.init n_samples (fun i -> i) in
    Rng.shuffle state indices;
    let x = Rng.take_indices x ~axis:0 indices in
    let y = Rng.take_indices y ~axis:0 indices in
    (x, y))
  else (x, y)

let make_classification ?(n_samples = 100) ?(n_features = 20)
    ?(n_informative = 2) ?(n_redundant = 2) ?(n_repeated = 0) ?(n_classes = 2)
    ?(n_clusters_per_class = 2) ?weights ?(flip_y = 0.01) ?(class_sep = 1.0)
    ?(hypercube = true) ?(shift = 0.0) ?(scale = 1.0) ?(shuffle = true)
    ?random_state () =
  let state = Rng.init_state random_state in

  if n_informative + n_redundant + n_repeated > n_features then
    failwith
      "make_classification: n_informative + n_redundant + n_repeated must be \
       <= n_features";

  let n_useless = n_features - n_informative - n_redundant - n_repeated in
  let n_clusters = n_classes * n_clusters_per_class in

  (* Generate hypercube vertices as cluster centers *)
  let centroids =
    if hypercube then (
      let vertices = Nx.zeros Float32 [| n_clusters; n_informative |] in
      let vertices_arr = Nx.to_array vertices in
      for i = 0 to n_clusters - 1 do
        for j = 0 to n_informative - 1 do
          let bit = (i lsr j) land 1 in
          vertices_arr.((i * n_informative) + j) <-
            (Float.of_int bit *. 2.) -. 1.
        done
      done;
      Nx.create Float32 [| n_clusters; n_informative |] vertices_arr)
    else Rng.standard_normal state [| n_clusters; n_informative |]
  in

  let centroids = Nx.mul_s centroids class_sep in

  (* Assign clusters to classes *)
  let y = Nx.zeros Int32 [| n_samples |] in
  let y_arr = Nx.to_array y |> Array.map Int32.to_int in

  let weights =
    match weights with
    | None -> Array.make n_classes (1.0 /. float n_classes)
    | Some w -> Array.of_list w
  in

  let samples_per_class =
    Array.map (fun w -> int_of_float (w *. float n_samples)) weights
  in

  (* Generate samples *)
  let x_informative = Nx.zeros Float32 [| n_samples; n_informative |] in
  let start_idx = ref 0 in

  for c = 0 to n_classes - 1 do
    let n_samples_c = samples_per_class.(c) in
    let cluster_ids =
      Array.init n_clusters_per_class (fun i -> (c * n_clusters_per_class) + i)
    in

    for i = 0 to n_samples_c - 1 do
      let idx = !start_idx + i in
      if idx < n_samples then (
        y_arr.(idx) <- c;
        let cluster = cluster_ids.(i mod n_clusters_per_class) in
        let centroid = Nx.slice [ I cluster; R [] ] centroids in
        let noise = Rng.standard_normal state [| 1; n_informative |] in
        let point = Nx.add centroid noise in
        for j = 0 to n_informative - 1 do
          let v = Nx.get_item [ 0; j ] point in
          Nx.set_item [ idx; j ] v x_informative
        done)
    done;
    start_idx := !start_idx + n_samples_c
  done;

  let y = Nx.create Int32 [| n_samples |] (Array.map Int32.of_int y_arr) in

  (* Add redundant features *)
  let x =
    if n_redundant > 0 then
      let b = Rng.standard_normal state [| n_informative; n_redundant |] in
      let redundant = Nx.matmul x_informative b in
      Nx.concatenate ~axis:1 [ x_informative; redundant ]
    else x_informative
  in

  (* Add repeated features *)
  let x =
    if n_repeated > 0 then
      let indices =
        Rng.randint state (n_informative + n_redundant) [| n_repeated |]
      in
      let indices_arr = Array.map Int32.to_int (Nx.to_array indices) in
      let repeated = Rng.take_indices x ~axis:1 indices_arr in
      Nx.concatenate ~axis:1 [ x; repeated ]
    else x
  in

  (* Add useless features *)
  let x =
    if n_useless > 0 then
      let useless = Rng.standard_normal state [| n_samples; n_useless |] in
      Nx.concatenate ~axis:1 [ x; useless ]
    else x
  in

  (* Apply transformations *)
  let x = if shift <> 0.0 then Nx.add_s x shift else x in
  let x = if scale <> 1.0 then Nx.mul_s x scale else x in

  (* Flip labels *)
  if flip_y > 0.0 then (
    let n_flip = int_of_float (flip_y *. float n_samples) in
    let flip_indices = Array.init n_samples (fun i -> i) in
    Rng.shuffle state flip_indices;
    let y_arr = Nx.to_array y |> Array.map Int32.to_int in
    for i = 0 to n_flip - 1 do
      let idx = flip_indices.(i) in
      y_arr.(idx) <- Random.State.int state n_classes
    done;
    let y = Nx.create Int32 [| n_samples |] (Array.map Int32.of_int y_arr) in

    if shuffle then (
      let indices = Array.init n_samples (fun i -> i) in
      Rng.shuffle state indices;
      let indices_tensor =
        Nx.create Int32 [| n_samples |] (Array.map Int32.of_int indices)
      in
      let x =
        Rng.take_indices x ~axis:0
          (Array.map Int32.to_int (Nx.to_array indices_tensor))
      in
      let y =
        Rng.take_indices y ~axis:0
          (Array.map Int32.to_int (Nx.to_array indices_tensor))
      in
      (x, y))
    else (x, y))
  else if shuffle then (
    let indices = Array.init n_samples (fun i -> i) in
    Rng.shuffle state indices;
    let indices_tensor =
      Nx.create Int32 [| n_samples |] (Array.map Int32.of_int indices)
    in
    let x =
      Rng.take_indices x ~axis:0
        (Array.map Int32.to_int (Nx.to_array indices_tensor))
    in
    let y =
      Rng.take_indices y ~axis:0
        (Array.map Int32.to_int (Nx.to_array indices_tensor))
    in
    (x, y))
  else (x, y)

let make_gaussian_quantiles ?mean ?(cov = 1.0) ?(n_samples = 100)
    ?(n_features = 2) ?(n_classes = 3) ?(shuffle = true) ?random_state () =
  let state = Rng.init_state random_state in

  let mean = match mean with None -> Array.make n_features 0. | Some m -> m in

  let x = Rng.standard_normal state [| n_samples; n_features |] in
  let x = Nx.mul_s x (sqrt cov) in
  let mean_tensor = Nx.create Float32 [| n_features |] mean in
  let x = Nx.add x (Nx.broadcast_to [| n_samples; n_features |] mean_tensor) in

  (* Compute distances from origin *)
  let distances_arr = Array.make n_samples 0. in
  let x_arr = Nx.to_array x in
  for i = 0 to n_samples - 1 do
    let sum = ref 0. in
    for j = 0 to n_features - 1 do
      let v = x_arr.((i * n_features) + j) in
      sum := !sum +. (v *. v)
    done;
    distances_arr.(i) <- sqrt !sum
  done;

  (* Sort indices by distance *)
  let indices = Array.init n_samples (fun i -> i) in
  Array.sort (fun i j -> compare distances_arr.(i) distances_arr.(j)) indices;

  (* Assign labels based on quantiles *)
  let y_arr = Array.make n_samples 0 in
  let samples_per_class = n_samples / n_classes in

  for i = 0 to n_samples - 1 do
    let class_idx = min (i / samples_per_class) (n_classes - 1) in
    y_arr.(indices.(i)) <- class_idx
  done;

  let y = Nx.create Int32 [| n_samples |] (Array.map Int32.of_int y_arr) in

  if shuffle then (
    let shuffle_indices = Array.init n_samples (fun i -> i) in
    Rng.shuffle state shuffle_indices;
    let x = Rng.take_indices x ~axis:0 shuffle_indices in
    let y = Rng.take_indices y ~axis:0 shuffle_indices in
    (x, y))
  else (x, y)

let make_hastie_10_2 ?(n_samples = 12000) ?random_state () =
  let state = Rng.init_state random_state in

  (* Generate 10 random features *)
  let x = Rng.standard_normal state [| n_samples; 10 |] in

  (* Compute y according to Hastie et al. 2009 formula *)
  let y_arr = Array.make n_samples 0 in
  let x_arr = Nx.to_array x in

  for i = 0 to n_samples - 1 do
    let sum = ref 0. in
    for j = 0 to 9 do
      sum := !sum +. (x_arr.((i * 10) + j) ** 2.)
    done;
    y_arr.(i) <- (if !sum > 9.34 then 1 else 0)
  done;

  let y = Nx.create Int32 [| n_samples |] (Array.map Int32.of_int y_arr) in
  (x, y)

let make_circles ?(n_samples = 100) ?(shuffle = true) ?(noise = 0.0)
    ?random_state ?(factor = 0.8) () =
  let state = Rng.init_state random_state in

  if factor <= 0. || factor >= 1. then
    failwith "make_circles: factor must be between 0 and 1";

  let n_samples_out = n_samples / 2 in
  let n_samples_in = n_samples - n_samples_out in

  (* Generate outer circle *)
  let linspace start stop n =
    Array.init n (fun i ->
        start +. (float i /. float (n - 1) *. (stop -. start)))
  in

  let theta_out = linspace 0. (2. *. pi) n_samples_out in
  let x_out = Array.map cos theta_out in
  let y_out = Array.map sin theta_out in

  (* Generate inner circle *)
  let theta_in = linspace 0. (2. *. pi) n_samples_in in
  let x_in = Array.map (fun t -> factor *. cos t) theta_in in
  let y_in = Array.map (fun t -> factor *. sin t) theta_in in

  (* Combine circles *)
  let x_data = Array.append x_out x_in in
  let y_data = Array.append y_out y_in in

  let x = Nx.zeros Float32 [| n_samples; 2 |] in
  for i = 0 to n_samples - 1 do
    Nx.set_item [ i; 0 ] x_data.(i) x;
    Nx.set_item [ i; 1 ] y_data.(i) x
  done;

  (* Add noise *)
  let x =
    if noise > 0.0 then
      Nx.add x (Nx.mul_s (Rng.standard_normal state [| n_samples; 2 |]) noise)
    else x
  in

  (* Create labels *)
  let y_arr =
    Array.init n_samples (fun i -> if i < n_samples_out then 0 else 1)
  in
  let y = Nx.create Int32 [| n_samples |] (Array.map Int32.of_int y_arr) in

  if shuffle then (
    let indices = Array.init n_samples (fun i -> i) in
    Rng.shuffle state indices;
    let x = Rng.take_indices x ~axis:0 indices in
    let y = Rng.take_indices y ~axis:0 indices in
    (x, y))
  else (x, y)

let make_moons ?(n_samples = 100) ?(shuffle = true) ?(noise = 0.0) ?random_state
    () =
  let state = Rng.init_state random_state in

  let n_samples_out = n_samples / 2 in
  let n_samples_in = n_samples - n_samples_out in

  let linspace start stop n =
    Array.init n (fun i ->
        start +. (float i /. float (n - 1) *. (stop -. start)))
  in

  (* Generate first moon *)
  let theta_out = linspace 0. pi n_samples_out in
  let x_out = Array.map cos theta_out in
  let y_out = Array.map sin theta_out in

  (* Generate second moon *)
  let theta_in = linspace 0. pi n_samples_in in
  let x_in = Array.map (fun t -> 1. -. cos t) theta_in in
  let y_in = Array.map (fun t -> 1. -. sin t -. 0.5) theta_in in

  (* Combine moons *)
  let x_data = Array.append x_out x_in in
  let y_data = Array.append y_out y_in in

  let x = Nx.zeros Float32 [| n_samples; 2 |] in
  for i = 0 to n_samples - 1 do
    Nx.set_item [ i; 0 ] x_data.(i) x;
    Nx.set_item [ i; 1 ] y_data.(i) x
  done;

  (* Add noise *)
  let x =
    if noise > 0.0 then
      Nx.add x (Nx.mul_s (Rng.standard_normal state [| n_samples; 2 |]) noise)
    else x
  in

  (* Create labels *)
  let y_arr =
    Array.init n_samples (fun i -> if i < n_samples_out then 0 else 1)
  in
  let y = Nx.create Int32 [| n_samples |] (Array.map Int32.of_int y_arr) in

  if shuffle then (
    let indices = Array.init n_samples (fun i -> i) in
    Rng.shuffle state indices;
    let x = Rng.take_indices x ~axis:0 indices in
    let y = Rng.take_indices y ~axis:0 indices in
    (x, y))
  else (x, y)

let make_multilabel_classification ?(n_samples = 100) ?(n_features = 20)
    ?(n_classes = 5) ?(n_labels = 2) ?(length = 50) ?(allow_unlabeled = true)
    ?(sparse = false) ?(return_indicator = false)
    ?(return_distributions = false) ?random_state () =
  let _ = return_distributions in
  let state = Rng.init_state random_state in

  (* Generate random class probabilities *)
  let p_c = Rng.uniform state [| n_classes |] in
  let p_c_sum = Nx.sum p_c in
  let p_c = Nx.div p_c p_c_sum in

  (* Generate random feature probabilities for each class *)
  let p_w_c = Rng.uniform state [| n_features; n_classes |] in
  (* Normalize each column *)
  let p_w_c =
    let sums = Nx.sum p_w_c ~axes:[| 0 |] in
    Nx.div p_w_c (Nx.broadcast_to [| n_features; n_classes |] sums)
  in

  let x = Nx.zeros Float32 [| n_samples; n_features |] in
  let y_float = Nx.zeros Float32 [| n_samples; n_classes |] in
  let y_int = Nx.zeros Int32 [| n_samples; n_labels |] in

  let p_c_arr = Nx.to_array p_c in
  let p_w_c_arr = Nx.to_array p_w_c in

  for i = 0 to n_samples - 1 do
    (* Sample number of labels *)
    (* Simple Poisson approximation using normal distribution *)
    let lambda = float n_labels in
    let n_doc = int_of_float (lambda +. (sqrt lambda *. Rng.normal state ())) in
    let n_doc = max 0 n_doc in
    let n_doc = if allow_unlabeled then n_doc else max 1 n_doc in
    let n_doc = min n_doc n_classes in

    (* Sample labels *)
    let labels = Array.make n_doc 0 in
    let cum_prob = ref 0. in
    let label_idx = ref 0 in

    for _ = 0 to n_doc - 1 do
      let r = Random.State.float state 1. in
      cum_prob := 0.;
      for c = 0 to n_classes - 1 do
        cum_prob := !cum_prob +. p_c_arr.(c);
        if r < !cum_prob && !label_idx < n_doc then (
          labels.(!label_idx) <- c;
          incr label_idx)
      done
    done;

    if return_indicator then
      (* Set binary indicators *)
      for j = 0 to n_doc - 1 do
        Nx.set_item [ i; labels.(j) ] 1. y_float
      done
    else
      (* Store first n_labels labels *)
      for j = 0 to min (n_labels - 1) (n_doc - 1) do
        Nx.set_item [ i; j ] (Int32.of_int labels.(j)) y_int
      done;

    (* Generate document *)
    for _ = 0 to length - 1 do
      if n_doc > 0 then
        let label = labels.(Random.State.int state n_doc) in
        let r = Random.State.float state 1. in
        let mut_cum_prob = ref 0. in

        for f = 0 to n_features - 1 do
          mut_cum_prob := !mut_cum_prob +. p_w_c_arr.((f * n_classes) + label);
          if r < !mut_cum_prob then
            let current = Nx.get_item [ i; f ] x in
            Nx.set_item [ i; f ] (current +. 1.) x
        done
    done
  done;

  if sparse then
    failwith "make_multilabel_classification: sparse output not yet implemented"
  else if return_indicator then (x, `Float y_float)
  else (x, `Int y_int)

let make_regression ?(n_samples = 100) ?(n_features = 100) ?(n_informative = 10)
    ?(n_targets = 1) ?(bias = 0.0) ?(effective_rank = None)
    ?(tail_strength = 0.5) ?(noise = 0.0) ?(shuffle = true) ?(coef = false)
    ?random_state () =
  let _ = tail_strength in
  let state = Rng.init_state random_state in

  let n_informative = min n_features n_informative in

  (* Generate X *)
  let x = Rng.standard_normal state [| n_samples; n_features |] in

  (* Generate ground truth model *)
  let ground_truth = Rng.standard_normal state [| n_features; n_targets |] in
  let ground_truth = Nx.slice [ R [ 0; n_informative ]; R [] ] ground_truth in

  (* Pad with zeros for non-informative features *)
  let coef_tensor = Nx.zeros Float32 [| n_features; n_targets |] in
  for i = 0 to n_informative - 1 do
    for j = 0 to n_targets - 1 do
      let v = Nx.get_item [ i; j ] ground_truth in
      Nx.set_item [ i; j ] v coef_tensor
    done
  done;

  (* Apply effective rank *)
  let x_final =
    match effective_rank with
    | None -> x
    | Some rank ->
        (* Simple low-rank approximation *)
        let u = Rng.standard_normal state [| n_samples; rank |] in
        let v = Rng.standard_normal state [| rank; n_features |] in
        Nx.matmul u v
  in

  (* Generate y *)
  let y = Nx.matmul x_final coef_tensor in
  let y = Nx.add_s y bias in

  (* Add noise *)
  let y =
    if noise > 0.0 then
      Nx.add y
        (Nx.mul_s (Rng.standard_normal state [| n_samples; n_targets |]) noise)
    else y
  in

  if shuffle then (
    let indices = Array.init n_samples (fun i -> i) in
    Rng.shuffle state indices;
    let x_final = Rng.take_indices x_final ~axis:0 indices in
    let y = Rng.take_indices y ~axis:0 indices in
    if coef then (x_final, y, Some coef_tensor) else (x_final, y, None))
  else if coef then (x_final, y, Some coef_tensor)
  else (x_final, y, None)

let make_sparse_uncorrelated ?(n_samples = 100) ?(n_features = 10) ?random_state
    () =
  let state = Rng.init_state random_state in

  (* Generate random features *)
  let x = Rng.standard_normal state [| n_samples; n_features |] in

  (* Only first 4 features are used *)
  let y = Nx.zeros Float32 [| n_samples |] in
  let x_arr = Nx.to_array x in
  let y_arr = Nx.to_array y in

  for i = 0 to n_samples - 1 do
    let idx = i * n_features in
    if n_features >= 4 then
      y_arr.(i) <-
        x_arr.(idx)
        +. (2. *. x_arr.(idx + 1))
        -. (2. *. x_arr.(idx + 2))
        -. (1.5 *. x_arr.(idx + 3))
  done;

  let y = Nx.create Float32 [| n_samples |] y_arr in
  (x, y)

let make_friedman1 ?(n_samples = 100) ?(n_features = 10) ?(noise = 0.0)
    ?random_state () =
  let state = Rng.init_state random_state in

  if n_features < 5 then
    failwith "make_friedman1: n_features must be at least 5";

  (* Generate uniform random features *)
  let x = Rng.uniform state [| n_samples; n_features |] in

  (* Compute y according to Friedman #1 formula *)
  let y = Nx.zeros Float32 [| n_samples |] in
  let x_arr = Nx.to_array x in
  let y_arr = Nx.to_array y in

  for i = 0 to n_samples - 1 do
    let idx = i * n_features in
    let x0 = x_arr.(idx) in
    let x1 = x_arr.(idx + 1) in
    let x2 = x_arr.(idx + 2) in
    let x3 = x_arr.(idx + 3) in
    let x4 = x_arr.(idx + 4) in

    y_arr.(i) <-
      (10. *. sin (pi *. x0 *. x1))
      +. (20. *. ((x2 -. 0.5) ** 2.))
      +. (10. *. x3) +. (5. *. x4)
  done;

  let y = Nx.create Float32 [| n_samples |] y_arr in

  (* Add noise *)
  let y =
    if noise > 0.0 then
      Nx.add y (Nx.mul_s (Rng.standard_normal state [| n_samples |]) noise)
    else y
  in

  (x, y)

let make_friedman2 ?(n_samples = 100) ?(noise = 0.0) ?random_state () =
  let state = Rng.init_state random_state in

  (* Generate features with different ranges *)
  let x = Nx.zeros Float32 [| n_samples; 4 |] in

  let x0 = Rng.uniform state ~low:0. ~high:100. [| n_samples |] in
  let x1 = Rng.uniform state ~low:40. ~high:560. [| n_samples |] in
  let x2 = Rng.uniform state ~low:0. ~high:1. [| n_samples |] in
  let x3 = Rng.uniform state ~low:1. ~high:11. [| n_samples |] in

  for i = 0 to n_samples - 1 do
    Nx.set_item [ i; 0 ] (Nx.get_item [ i ] x0) x;
    Nx.set_item [ i; 1 ] (Nx.get_item [ i ] x1) x;
    Nx.set_item [ i; 2 ] (Nx.get_item [ i ] x2) x;
    Nx.set_item [ i; 3 ] (Nx.get_item [ i ] x3) x
  done;

  (* Compute y *)
  let y = Nx.zeros Float32 [| n_samples |] in
  let x_arr = Nx.to_array x in
  let y_arr = Nx.to_array y in

  for i = 0 to n_samples - 1 do
    let x0 = x_arr.(i * 4) in
    let x1 = x_arr.((i * 4) + 1) in
    let x2 = x_arr.((i * 4) + 2) in
    let x3 = x_arr.((i * 4) + 3) in

    let term1 = x0 ** 2. in
    let term2 = ((x1 *. x2) -. (1. /. (x1 *. x3))) ** 2. in
    y_arr.(i) <- sqrt (term1 +. term2)
  done;

  let y = Nx.create Float32 [| n_samples |] y_arr in

  (* Add noise *)
  let y =
    if noise > 0.0 then
      Nx.add y (Nx.mul_s (Rng.standard_normal state [| n_samples |]) noise)
    else y
  in

  (x, y)

let make_friedman3 ?(n_samples = 100) ?(noise = 0.0) ?random_state () =
  let state = Rng.init_state random_state in

  (* Generate features with different ranges *)
  let x = Nx.zeros Float32 [| n_samples; 4 |] in

  let x0 = Rng.uniform state ~low:0. ~high:100. [| n_samples |] in
  let x1 = Rng.uniform state ~low:40. ~high:560. [| n_samples |] in
  let x2 = Rng.uniform state ~low:0. ~high:1. [| n_samples |] in
  let x3 = Rng.uniform state ~low:1. ~high:11. [| n_samples |] in

  for i = 0 to n_samples - 1 do
    Nx.set_item [ i; 0 ] (Nx.get_item [ i ] x0) x;
    Nx.set_item [ i; 1 ] (Nx.get_item [ i ] x1) x;
    Nx.set_item [ i; 2 ] (Nx.get_item [ i ] x2) x;
    Nx.set_item [ i; 3 ] (Nx.get_item [ i ] x3) x
  done;

  (* Compute y *)
  let y = Nx.zeros Float32 [| n_samples |] in
  let x_arr = Nx.to_array x in
  let y_arr = Nx.to_array y in

  for i = 0 to n_samples - 1 do
    let x0 = x_arr.(i * 4) in
    let x1 = x_arr.((i * 4) + 1) in
    let x2 = x_arr.((i * 4) + 2) in
    let x3 = x_arr.((i * 4) + 3) in

    let numerator = (x1 *. x2) -. (1. /. (x1 *. x3)) in
    y_arr.(i) <- atan (numerator /. x0)
  done;

  let y = Nx.create Float32 [| n_samples |] y_arr in

  (* Add noise *)
  let y =
    if noise > 0.0 then
      Nx.add y (Nx.mul_s (Rng.standard_normal state [| n_samples |]) noise)
    else y
  in

  (x, y)

let make_s_curve ?(n_samples = 100) ?(noise = 0.0) ?random_state () =
  let state = Rng.init_state random_state in

  let t = Rng.uniform state ~low:0. ~high:(3. *. pi) [| n_samples |] in
  let t_arr = Nx.to_array t in

  let x = Nx.zeros Float32 [| n_samples; 3 |] in

  for i = 0 to n_samples - 1 do
    let t_val = t_arr.(i) in
    Nx.set_item [ i; 0 ] (sin t_val) x;
    Nx.set_item [ i; 1 ] (2. *. Random.State.float state 1.) x;
    let cos_val = cos t_val in
    let sign = if cos_val >= 0. then 1. else -1. in
    Nx.set_item [ i; 2 ] (sign *. cos_val) x
  done;

  (* Add noise *)
  let x =
    if noise > 0.0 then
      Nx.add x (Nx.mul_s (Rng.standard_normal state [| n_samples; 3 |]) noise)
    else x
  in

  (x, t)

let make_swiss_roll ?(n_samples = 100) ?(noise = 0.0) ?random_state
    ?(hole = false) () =
  let state = Rng.init_state random_state in

  let t = Rng.uniform state ~low:0. ~high:(1.5 *. pi) [| n_samples |] in
  let height = Rng.uniform state ~low:0. ~high:21. [| n_samples |] in

  let t_arr = Nx.to_array t in
  let height_arr = Nx.to_array height in

  let x = Nx.zeros Float32 [| n_samples; 3 |] in

  let valid_count = ref 0 in
  let indices = ref [] in

  for i = 0 to n_samples - 1 do
    let t_val = t_arr.(i) +. (1.5 *. pi) in

    (* Check for hole *)
    let in_hole = hole && t_val > 10. && t_val < 14. in

    if not in_hole then (
      Nx.set_item [ !valid_count; 0 ] (t_val *. cos t_val) x;
      Nx.set_item [ !valid_count; 1 ] height_arr.(i) x;
      Nx.set_item [ !valid_count; 2 ] (t_val *. sin t_val) x;
      indices := i :: !indices;
      incr valid_count)
  done;

  let valid_samples = !valid_count in
  let x = Nx.slice [ R [ 0; valid_samples ]; R [] ] x in

  (* Get corresponding t values *)
  let t_valid = Nx.zeros Float32 [| valid_samples |] in
  List.iteri
    (fun idx i -> Nx.set_item [ valid_samples - 1 - idx ] t_arr.(i) t_valid)
    !indices;

  (* Add noise *)
  let x =
    if noise > 0.0 then
      Nx.add x
        (Nx.mul_s (Rng.standard_normal state [| valid_samples; 3 |]) noise)
    else x
  in

  (x, t_valid)

let make_low_rank_matrix ?(n_samples = 100) ?(n_features = 100)
    ?(effective_rank = 10) ?(tail_strength = 0.5) ?random_state () =
  let state = Rng.init_state random_state in

  (* Generate singular values *)
  let singular_values =
    Array.init n_features (fun i ->
        let decay = exp (-.float (i - effective_rank) *. tail_strength) in
        if i < effective_rank then 1. else decay)
  in

  (* Generate random orthogonal matrices *)
  let u = Rng.standard_normal state [| n_samples; n_samples |] in
  let v = Rng.standard_normal state [| n_features; n_features |] in

  (* Create diagonal matrix of singular values *)
  let s = Nx.zeros Float32 [| n_samples; n_features |] in
  let min_dim = min n_samples n_features in
  for i = 0 to min_dim - 1 do
    Nx.set_item [ i; i ] singular_values.(i) s
  done;

  (* X = U * S * V^T *)
  let x = Nx.matmul u s in
  let x = Nx.matmul x (Nx.transpose v) in

  x

let make_sparse_coded_signal ~n_samples ~n_components ~n_features
    ~n_nonzero_coefs ?random_state () =
  let state = Rng.init_state random_state in

  (* Generate dictionary *)
  let d = Rng.standard_normal state [| n_features; n_components |] in

  (* Normalize dictionary columns *)
  let d =
    let d_squared = Nx.mul d d in
    let norms_squared = Nx.sum d_squared ~axes:[| 0 |] in
    let norms = Nx.sqrt norms_squared in
    Nx.div d (Nx.broadcast_to [| n_features; n_components |] norms)
  in

  (* Generate sparse codes *)
  let x = Nx.zeros Float32 [| n_components; n_samples |] in

  for i = 0 to n_samples - 1 do
    (* Select random components *)
    let indices = Array.init n_components (fun j -> j) in
    Rng.shuffle state indices;

    (* Set non-zero coefficients *)
    for j = 0 to n_nonzero_coefs - 1 do
      let coef = Rng.normal state () in
      Nx.set_item [ indices.(j); i ] coef x
    done
  done;

  (* Generate signal Y = D * X *)
  let y = Nx.matmul d x in

  (y, d, x)

let make_spd_matrix ?(n_dim = 30) ?random_state () =
  let state = Rng.init_state random_state in

  (* Generate random matrix *)
  let a = Rng.standard_normal state [| n_dim; n_dim |] in

  (* Make it symmetric positive definite: A^T * A *)
  let spd = Nx.matmul (Nx.transpose a) a in

  (* Add small diagonal to ensure positive definiteness *)
  let eye = Nx.eye Float32 n_dim in
  Nx.add spd (Nx.mul_s eye 0.01)

let make_sparse_spd_matrix ?(n_dim = 30) ?(alpha = 0.95) ?(norm_diag = false)
    ?(smallest_coef = 0.1) ?(largest_coef = 0.9) ?random_state () =
  let _ = norm_diag in
  let state = Rng.init_state random_state in

  let a = Nx.zeros Float32 [| n_dim; n_dim |] in

  (* Fill upper triangle with sparse coefficients *)
  for i = 0 to n_dim - 1 do
    for j = i to n_dim - 1 do
      if Random.State.float state 1. > alpha then (
        let coef =
          smallest_coef
          +. Random.State.float state (largest_coef -. smallest_coef)
        in
        let coef = if Random.State.bool state then coef else -.coef in
        Nx.set_item [ i; j ] coef a;
        if i <> j then Nx.set_item [ j; i ] coef a)
    done
  done;

  (* Make positive definite *)
  let spd = Nx.matmul (Nx.transpose a) a in

  (* Return without diagonal normalization for now *)
  spd

let make_biclusters ?(shape = (100, 100)) ?(n_clusters = 5) ?(noise = 0.0)
    ?(minval = 10) ?(maxval = 100) ?(shuffle = true) ?random_state () =
  let state = Rng.init_state random_state in
  let n_rows, n_cols = shape in

  (* Initialize data matrix *)
  let x = Nx.zeros Float32 [| n_rows; n_cols |] in

  (* Generate bicluster assignments *)
  let rows_per_cluster = n_rows / n_clusters in
  let cols_per_cluster = n_cols / n_clusters in

  let row_labels = Nx.zeros Int32 [| n_rows |] in
  let col_labels = Nx.zeros Int32 [| n_cols |] in

  (* Assign cluster labels *)
  for i = 0 to n_rows - 1 do
    Nx.set_item [ i ] (Int32.of_int (i / rows_per_cluster)) row_labels
  done;

  for j = 0 to n_cols - 1 do
    Nx.set_item [ j ] (Int32.of_int (j / cols_per_cluster)) col_labels
  done;

  (* Fill biclusters with values *)
  for c = 0 to n_clusters - 1 do
    let value = float (minval + Random.State.int state (maxval - minval)) in

    for i = 0 to n_rows - 1 do
      for j = 0 to n_cols - 1 do
        if
          Nx.get_item [ i ] row_labels = Int32.of_int c
          && Nx.get_item [ j ] col_labels = Int32.of_int c
        then Nx.set_item [ i; j ] value x
      done
    done
  done;

  (* Add noise *)
  let x =
    if noise > 0.0 then
      Nx.add x (Nx.mul_s (Rng.standard_normal state [| n_rows; n_cols |]) noise)
    else x
  in

  if shuffle then (
    (* Shuffle rows *)
    let row_indices = Array.init n_rows (fun i -> i) in
    Rng.shuffle state row_indices;
    let x = Rng.take_indices x ~axis:0 row_indices in
    let row_labels = Rng.take_indices row_labels ~axis:0 row_indices in

    (* Shuffle columns *)
    let col_indices = Array.init n_cols (fun i -> i) in
    Rng.shuffle state col_indices;
    let x = Rng.take_indices x ~axis:1 col_indices in
    let col_labels = Rng.take_indices col_labels ~axis:0 col_indices in

    (x, row_labels, col_labels))
  else (x, row_labels, col_labels)

let make_checkerboard ?(shape = (100, 100)) ?(n_clusters = (8, 8))
    ?(noise = 0.0) ?(minval = 10) ?(maxval = 100) ?(shuffle = true)
    ?random_state () =
  let state = Rng.init_state random_state in
  let n_rows, n_cols = shape in
  let n_clusters_row, n_clusters_col = n_clusters in

  (* Initialize data matrix *)
  let x = Nx.zeros Float32 [| n_rows; n_cols |] in

  (* Generate cluster assignments *)
  let rows_per_cluster = n_rows / n_clusters_row in
  let cols_per_cluster = n_cols / n_clusters_col in

  let row_labels = Nx.zeros Int32 [| n_rows |] in
  let col_labels = Nx.zeros Int32 [| n_cols |] in

  (* Assign cluster labels *)
  for i = 0 to n_rows - 1 do
    Nx.set_item [ i ] (Int32.of_int (i / rows_per_cluster)) row_labels
  done;

  for j = 0 to n_cols - 1 do
    Nx.set_item [ j ] (Int32.of_int (j / cols_per_cluster)) col_labels
  done;

  (* Fill checkerboard pattern *)
  for i = 0 to n_rows - 1 do
    for j = 0 to n_cols - 1 do
      let row_cluster = Int32.to_int (Nx.get_item [ i ] row_labels) in
      let col_cluster = Int32.to_int (Nx.get_item [ j ] col_labels) in

      (* Checkerboard pattern: alternate high/low values *)
      let is_high = (row_cluster + col_cluster) mod 2 = 0 in
      let value = float (if is_high then maxval else minval) in
      Nx.set_item [ i; j ] value x
    done
  done;

  (* Add noise *)
  let x =
    if noise > 0.0 then
      Nx.add x (Nx.mul_s (Rng.standard_normal state [| n_rows; n_cols |]) noise)
    else x
  in

  if shuffle then (
    (* Shuffle rows *)
    let row_indices = Array.init n_rows (fun i -> i) in
    Rng.shuffle state row_indices;
    let x = Rng.take_indices x ~axis:0 row_indices in
    let row_labels = Rng.take_indices row_labels ~axis:0 row_indices in

    (* Shuffle columns *)
    let col_indices = Array.init n_cols (fun i -> i) in
    Rng.shuffle state col_indices;
    let x = Rng.take_indices x ~axis:1 col_indices in
    let col_labels = Rng.take_indices col_labels ~axis:0 col_indices in

    (x, row_labels, col_labels))
  else (x, row_labels, col_labels)
