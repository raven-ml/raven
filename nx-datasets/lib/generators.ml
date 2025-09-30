open Bigarray_ext

let pi = 4. *. atan 1.

module IntSet = Set.Make (Int)

module Rng = struct
  let init_state seed =
    match seed with
    | None -> Random.State.make_self_init ()
    | Some s -> Random.State.make [| s |]

  let normal state ?(mean = 0.) ?(std = 1.) () =
    (* Box-Muller transform for a single normal variate *)
    let u1 = Random.State.float state 1. in
    let u2 = Random.State.float state 1. in
    let z0 = sqrt (-2. *. log u1) *. cos (2. *. pi *. u2) in
    mean +. (std *. z0)

  let standard_normal state shape =
    let total = Array.fold_left ( * ) 1 shape in
    let data = Array1.create Float32 C_layout total in
    let i = ref 0 in
    while !i < total do
      let u1 = Random.State.float state 1. in
      let u2 = Random.State.float state 1. in
      let r = sqrt (-2. *. log u1) in
      let theta = 2. *. pi *. u2 in
      let z0 = r *. cos theta in
      let z1 = r *. sin theta in
      data.{!i} <- z0;
      incr i;
      if !i < total then (
        data.{!i} <- z1;
        incr i)
    done;
    Nx.reshape shape (Nx.of_bigarray_ext (genarray_of_array1 data))

  let uniform state ?(low = 0.) ?(high = 1.) shape =
    let total = Array.fold_left ( * ) 1 shape in
    let range = high -. low in
    let data = Array1.create Float32 C_layout total in
    for i = 0 to total - 1 do
      data.{i} <- low +. Random.State.float state range
    done;
    Nx.reshape shape (Nx.of_bigarray_ext (genarray_of_array1 data))

  let randint state ?(low = 0) high shape =
    let total = Array.fold_left ( * ) 1 shape in
    let range = high - low in
    let data = Array1.create Int32 C_layout total in
    for i = 0 to total - 1 do
      data.{i} <- Int32.of_int (low + Random.State.int state range)
    done;
    Nx.reshape shape (Nx.of_bigarray_ext (genarray_of_array1 data))

  let shuffle state arr =
    let n = Array.length arr in
    for i = n - 1 downto 1 do
      let j = Random.State.int state (i + 1) in
      let tmp = arr.(i) in
      arr.(i) <- arr.(j);
      arr.(j) <- tmp
    done

  let take_indices tensor ~axis indices_array =
    (* Use take to gather along the specified axis *)
    let indices_tensor =
      Nx.create Nx.int32
        [| Array.length indices_array |]
        (Array.map Int32.of_int indices_array)
    in
    Nx.take ~axis indices_tensor tensor
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

  let samples_per_center =
    Array.init n_centers (fun i ->
        let base = n_samples / n_centers in
        let extra = if i < n_samples mod n_centers then 1 else 0 in
        base + extra)
  in

  let x_list =
    List.init n_centers (fun i ->
        let n = samples_per_center.(i) in
        if n > 0 then
          let center = Nx.slice [ Nx.I i; Nx.A ] center_coords in
          let noise = Rng.standard_normal state [| n; n_features |] in
          Some (Nx.add (Nx.mul_s noise cluster_std) center)
        else None)
    |> List.filter_map Fun.id
  in

  let y_list =
    List.init n_centers (fun i ->
        let n = samples_per_center.(i) in
        if n > 0 then Some (Nx.full Int32 [| n |] (Int32.of_int i)) else None)
    |> List.filter_map Fun.id
  in

  let x = Nx.concatenate ~axis:0 x_list in
  let y = Nx.concatenate ~axis:0 y_list in

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

  let centroids =
    if hypercube then
      Nx.init Float32 [| n_clusters; n_informative |] (fun idx ->
          let i, j = (idx.(0), idx.(1)) in
          let bit = (i lsr j) land 1 in
          (Float.of_int bit *. 2.) -. 1.)
    else Rng.standard_normal state [| n_clusters; n_informative |]
  in
  let centroids = Nx.mul_s centroids class_sep in

  let weights =
    match weights with
    | None -> Array.make n_classes (1.0 /. float n_classes)
    | Some w -> Array.of_list w
  in

  let y_arr = Array.make n_samples 0 in
  let cluster_indices_arr = Array.make n_samples 0 in
  let current_pos = ref 0 in
  for c = 0 to n_classes - 1 do
    let n_samples_c = int_of_float (weights.(c) *. float n_samples) in
    let cluster_ids =
      Array.init n_clusters_per_class (fun i -> (c * n_clusters_per_class) + i)
    in
    for i = 0 to n_samples_c - 1 do
      let sample_idx = !current_pos + i in
      if sample_idx < n_samples then (
        y_arr.(sample_idx) <- c;
        cluster_indices_arr.(sample_idx) <-
          cluster_ids.(Random.State.int state n_clusters_per_class))
    done;
    current_pos := !current_pos + n_samples_c
  done;

  let sample_centroids =
    Rng.take_indices centroids ~axis:0 cluster_indices_arr
  in
  let noise = Rng.standard_normal state [| n_samples; n_informative |] in
  let x_informative = Nx.add sample_centroids noise in

  let x =
    if n_redundant > 0 then
      let b = Rng.standard_normal state [| n_informative; n_redundant |] in
      Nx.concatenate ~axis:1 [ x_informative; Nx.matmul x_informative b ]
    else x_informative
  in

  let x =
    if n_repeated > 0 then
      let indices =
        Array.init n_repeated (fun _ ->
            Random.State.int state (n_informative + n_redundant))
      in
      Nx.concatenate ~axis:1 [ x; Rng.take_indices x ~axis:1 indices ]
    else x
  in

  let x =
    if n_useless > 0 then
      let useless = Rng.standard_normal state [| n_samples; n_useless |] in
      Nx.concatenate ~axis:1 [ x; useless ]
    else x
  in

  let x = if shift <> 0.0 then Nx.add_s x shift else x in
  let x = if scale <> 1.0 then Nx.mul_s x scale else x in

  if flip_y > 0.0 then (
    let n_flip = int_of_float (flip_y *. float n_samples) in
    let flip_indices = Array.init n_samples (fun i -> i) in
    Rng.shuffle state flip_indices;
    for i = 0 to n_flip - 1 do
      let idx = flip_indices.(i) in
      y_arr.(idx) <- Random.State.int state n_classes
    done);

  let y = Nx.create Int32 [| n_samples |] (Array.map Int32.of_int y_arr) in

  if shuffle then (
    let indices = Array.init n_samples (fun i -> i) in
    Rng.shuffle state indices;
    let x = Rng.take_indices x ~axis:0 indices in
    let y = Rng.take_indices y ~axis:0 indices in
    (x, y))
  else (x, y)

let make_gaussian_quantiles ?mean ?(cov = 1.0) ?(n_samples = 100)
    ?(n_features = 2) ?(n_classes = 3) ?(shuffle = true) ?random_state () =
  let state = Rng.init_state random_state in

  let mean_tensor =
    match mean with
    | None -> Nx.zeros Float32 [| n_features |]
    | Some m -> Nx.create Float32 [| n_features |] m
  in

  let x = Rng.standard_normal state [| n_samples; n_features |] in
  let x = Nx.mul_s x (sqrt cov) in
  let x = Nx.add x (Nx.broadcast_to [| n_samples; n_features |] mean_tensor) in

  let distances_sq = Nx.sum ~axes:[| 1 |] (Nx.square x) in
  let distances_arr = Nx.to_array (Nx.sqrt distances_sq) in

  let indices = Array.init n_samples (fun i -> i) in
  Array.sort (fun i j -> compare distances_arr.(i) distances_arr.(j)) indices;

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
  let x = Rng.standard_normal state [| n_samples; 10 |] in
  let chi2 = Nx.sum ~axes:[| 1 |] (Nx.square x) in
  let y = Nx.cast Int32 (Nx.cmpgt chi2 (Nx.scalar Float32 9.34)) in
  (x, y)

let make_circles ?(n_samples = 100) ?(shuffle = true) ?(noise = 0.0)
    ?random_state ?(factor = 0.8) () =
  let state = Rng.init_state random_state in

  if factor <= 0. || factor >= 1. then
    failwith "make_circles: factor must be between 0 and 1";

  let n_samples_out = n_samples / 2 in
  let n_samples_in = n_samples - n_samples_out in

  let linspace start stop n =
    if n = 0 then [||]
    else if n = 1 then [| start |]
    else
      Array.init n (fun i ->
          start +. (float i /. float (n - 1) *. (stop -. start)))
  in

  let theta_out = linspace 0. (2. *. pi) n_samples_out in
  let theta_in = linspace 0. (2. *. pi) n_samples_in in
  let x_flat = Array.make (n_samples * 2) 0. in
  for i = 0 to n_samples_out - 1 do
    x_flat.(i * 2) <- cos theta_out.(i);
    x_flat.((i * 2) + 1) <- sin theta_out.(i)
  done;
  for i = 0 to n_samples_in - 1 do
    let idx = n_samples_out + i in
    x_flat.(idx * 2) <- factor *. cos theta_in.(i);
    x_flat.((idx * 2) + 1) <- factor *. sin theta_in.(i)
  done;
  let x = Nx.create Float32 [| n_samples; 2 |] x_flat in

  let x =
    if noise > 0.0 then
      Nx.add x (Nx.mul_s (Rng.standard_normal state [| n_samples; 2 |]) noise)
    else x
  in

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

  let outer_theta = Nx.linspace Float32 0. pi n_samples_out in
  let inner_theta = Nx.linspace Float32 0. pi n_samples_in in

  let x_outer = Nx.cos outer_theta in
  let y_outer = Nx.sin outer_theta in
  let x_inner = Nx.sub (Nx.scalar Float32 1.) (Nx.cos inner_theta) in
  let y_inner = Nx.sub_s (Nx.add_s (Nx.sin inner_theta) 1.) 0.5 in

  let x =
    Nx.concatenate ~axis:0
      [
        Nx.stack ~axis:1 [ x_outer; y_outer ];
        Nx.stack ~axis:1 [ x_inner; y_inner ];
      ]
  in

  let x =
    if noise > 0.0 then
      Nx.add x (Nx.mul_s (Rng.standard_normal state [| n_samples; 2 |]) noise)
    else x
  in

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
  if sparse then
    failwith "make_multilabel_classification: sparse output not yet implemented";

  let state = Rng.init_state random_state in

  let p_c = Rng.uniform state [| n_classes |] in
  let p_c = Nx.div p_c (Nx.sum p_c) in

  let p_w_c = Rng.uniform state [| n_features; n_classes |] in
  let p_w_c = Nx.div p_w_c (Nx.sum p_w_c ~axes:[| 0 |]) in

  let y_indicator = Array.make_matrix n_samples n_classes false in
  let doc_topics = Array.make n_samples [] in

  let p_c_arr = Nx.to_array p_c in
  let cdf = Array.mapi (fun i p -> (i, p)) p_c_arr in
  let cdf =
    Array.sort (fun (_, p1) (_, p2) -> compare p2 p1) cdf;
    let sum = ref 0. in
    Array.map
      (fun (i, p) ->
        sum := !sum +. p;
        (i, !sum))
      cdf
  in

  for i = 0 to n_samples - 1 do
    let n_doc_labels =
      let lambda = float n_labels in
      let n = int_of_float (lambda +. (sqrt lambda *. Rng.normal state ())) in
      let n = max 0 n in
      let n = if allow_unlabeled then n else max 1 n in
      min n n_classes
    in
    let labels_set = ref IntSet.empty in
    while IntSet.cardinal !labels_set < n_doc_labels do
      let r = Random.State.float state 1. in
      let topic =
        let rec find_topic i =
          if i >= Array.length cdf then n_classes - 1
          else
            let topic, p = cdf.(i) in
            if r < p then topic else find_topic (i + 1)
        in
        find_topic 0
      in
      labels_set := IntSet.add topic !labels_set
    done;
    let labels = IntSet.elements !labels_set in
    doc_topics.(i) <- labels;
    List.iter (fun lbl -> y_indicator.(i).(lbl) <- true) labels
  done;

  let x_data = Array1.create Float32 C_layout (n_samples * n_features) in
  let p_w_c_arr = Nx.to_array p_w_c in
  for i = 0 to n_samples - 1 do
    let topics = doc_topics.(i) in
    if topics <> [] then
      let n_topics = List.length topics in
      for _ = 0 to length - 1 do
        let topic = List.nth topics (Random.State.int state n_topics) in
        let r = Random.State.float state 1. in
        let cum_prob = ref 0. in
        try
          for f = 0 to n_features - 1 do
            cum_prob := !cum_prob +. p_w_c_arr.((f * n_classes) + topic);
            if r < !cum_prob then (
              let idx = (i * n_features) + f in
              x_data.{idx} <- x_data.{idx} +. 1.;
              raise Exit)
          done
        with Exit -> ()
      done
  done;
  let x = Nx.of_bigarray_ext (genarray_of_array1 x_data) in
  let x = Nx.reshape [| n_samples; n_features |] x in

  let y =
    if return_indicator then (
      let data = Array.make (n_samples * n_classes) 0. in
      for r = 0 to n_samples - 1 do
        for c = 0 to n_classes - 1 do
          if y_indicator.(r).(c) then data.((r * n_classes) + c) <- 1.
        done
      done;
      `Float (Nx.create Float32 [| n_samples; n_classes |] data))
    else
      let data = Array.make (n_samples * n_labels) 0l in
      for r = 0 to n_samples - 1 do
        List.iteri
          (fun c_idx topic ->
            if c_idx < n_labels then
              data.((r * n_labels) + c_idx) <- Int32.of_int topic)
          doc_topics.(r)
      done;
      `Int (Nx.create Int32 [| n_samples; n_labels |] data)
  in
  (x, y)

let make_regression ?(n_samples = 100) ?(n_features = 100) ?(n_informative = 10)
    ?(n_targets = 1) ?(bias = 0.0) ?(effective_rank = None)
    ?(tail_strength = 0.5) ?(noise = 0.0) ?(shuffle = true) ?(coef = false)
    ?random_state () =
  let _ = tail_strength in
  let state = Rng.init_state random_state in
  let n_informative = min n_features n_informative in

  let x =
    match effective_rank with
    | None -> Rng.standard_normal state [| n_samples; n_features |]
    | Some rank ->
        let u = Rng.standard_normal state [| n_samples; rank |] in
        let v = Rng.standard_normal state [| rank; n_features |] in
        Nx.matmul u v
  in

  let ground_truth =
    Rng.uniform state ~low:(-100.) ~high:100. [| n_informative; n_targets |]
  in

  let coef_tensor =
    if n_informative = n_features then ground_truth
    else
      let zeros =
        Nx.zeros Float32 [| n_features - n_informative; n_targets |]
      in
      Nx.concatenate ~axis:0 [ ground_truth; zeros ]
  in

  let y = Nx.matmul x coef_tensor in
  let y = Nx.add_s y bias in

  let y =
    if noise > 0.0 then
      let noise_tensor = Rng.standard_normal state [| n_samples; n_targets |] in
      Nx.add y (Nx.mul_s noise_tensor noise)
    else y
  in

  let y = if n_targets = 1 then Nx.reshape [| n_samples |] y else y in

  let x_final, y_final, coef_final =
    if shuffle then (
      let indices = Array.init n_samples (fun i -> i) in
      Rng.shuffle state indices;
      ( Rng.take_indices x ~axis:0 indices,
        Rng.take_indices y ~axis:0 indices,
        Some coef_tensor ))
    else (x, y, Some coef_tensor)
  in

  if coef then (x_final, y_final, coef_final) else (x_final, y_final, None)

let make_sparse_uncorrelated ?(n_samples = 100) ?(n_features = 10) ?random_state
    () =
  let state = Rng.init_state random_state in
  let x = Rng.standard_normal state [| n_samples; n_features |] in

  let y =
    if n_features < 4 then Nx.zeros Float32 [| n_samples |]
    else
      let relevant_x = Nx.slice [ Nx.A; Nx.Rs (0, 4, 1) ] x in
      let coeffs = Nx.create Float32 [| 4 |] [| 1.; 2.; -2.; -1.5 |] in
      Nx.reshape [| n_samples |]
        (Nx.matmul relevant_x (Nx.reshape [| 4; 1 |] coeffs))
  in
  (x, y)

let make_friedman1 ?(n_samples = 100) ?(n_features = 10) ?(noise = 0.0)
    ?random_state () =
  let state = Rng.init_state random_state in
  if n_features < 5 then
    failwith "make_friedman1: n_features must be at least 5";

  let x = Rng.uniform state [| n_samples; n_features |] in
  let x_slice = Nx.slice [ Nx.A; Nx.Rs (0, 5, 1) ] x in
  let x0 = Nx.slice [ Nx.A; Nx.I 0 ] x_slice in
  let x1 = Nx.slice [ Nx.A; Nx.I 1 ] x_slice in
  let x2 = Nx.slice [ Nx.A; Nx.I 2 ] x_slice in
  let x3 = Nx.slice [ Nx.A; Nx.I 3 ] x_slice in
  let x4 = Nx.slice [ Nx.A; Nx.I 4 ] x_slice in

  let term1 = Nx.mul_s (Nx.sin (Nx.mul_s (Nx.mul x0 x1) pi)) 10. in
  let term2 = Nx.mul_s (Nx.square (Nx.sub_s x2 0.5)) 20. in
  let term3 = Nx.mul_s x3 10. in
  let term4 = Nx.mul_s x4 5. in
  let y = Nx.add (Nx.add term1 term2) (Nx.add term3 term4) in

  let y =
    if noise > 0.0 then
      Nx.add y (Nx.mul_s (Rng.standard_normal state [| n_samples |]) noise)
    else y
  in
  (x, y)

let make_friedman2 ?(n_samples = 100) ?(noise = 0.0) ?random_state () =
  let state = Rng.init_state random_state in
  let x0 = Rng.uniform state ~low:0. ~high:100. [| n_samples; 1 |] in
  let x1 = Rng.uniform state ~low:40. ~high:560. [| n_samples; 1 |] in
  let x2 = Rng.uniform state ~low:0. ~high:1. [| n_samples; 1 |] in
  let x3 = Rng.uniform state ~low:1. ~high:11. [| n_samples; 1 |] in
  let x = Nx.concatenate ~axis:1 [ x0; x1; x2; x3 ] in

  let term1 = Nx.square x0 in
  let term2 =
    Nx.square
      (Nx.sub (Nx.mul x1 x2) (Nx.div (Nx.scalar Float32 1.) (Nx.mul x1 x3)))
  in
  let y = Nx.reshape [| n_samples |] (Nx.sqrt (Nx.add term1 term2)) in

  let y =
    if noise > 0.0 then
      Nx.add y (Nx.mul_s (Rng.standard_normal state [| n_samples |]) noise)
    else y
  in
  (x, y)

let make_friedman3 ?(n_samples = 100) ?(noise = 0.0) ?random_state () =
  let state = Rng.init_state random_state in
  let x0 = Rng.uniform state ~low:0. ~high:100. [| n_samples; 1 |] in
  let x1 = Rng.uniform state ~low:40. ~high:560. [| n_samples; 1 |] in
  let x2 = Rng.uniform state ~low:0. ~high:1. [| n_samples; 1 |] in
  let x3 = Rng.uniform state ~low:1. ~high:11. [| n_samples; 1 |] in
  let x = Nx.concatenate ~axis:1 [ x0; x1; x2; x3 ] in

  let numerator =
    Nx.sub (Nx.mul x1 x2) (Nx.div (Nx.scalar Float32 1.) (Nx.mul x1 x3))
  in
  let y = Nx.reshape [| n_samples |] (Nx.atan (Nx.div numerator x0)) in

  let y =
    if noise > 0.0 then
      Nx.add y (Nx.mul_s (Rng.standard_normal state [| n_samples |]) noise)
    else y
  in
  (x, y)

let make_s_curve ?(n_samples = 100) ?(noise = 0.0) ?random_state () =
  let state = Rng.init_state random_state in
  let t = Rng.uniform state ~low:(-.pi) ~high:pi [| n_samples; 1 |] in
  let x_coord = Nx.sin t in
  let y_coord = Rng.uniform state ~low:(-2.) ~high:2. [| n_samples; 1 |] in
  let z_coord = Nx.mul (Nx.sign (Nx.cos t)) (Nx.cos t) in
  let x = Nx.concatenate ~axis:1 [ x_coord; y_coord; z_coord ] in

  let x =
    if noise > 0.0 then
      Nx.add x (Nx.mul_s (Rng.standard_normal state [| n_samples; 3 |]) noise)
    else x
  in
  (x, Nx.reshape [| n_samples |] t)

let make_swiss_roll ?(n_samples = 100) ?(noise = 0.0) ?random_state
    ?(hole = false) () =
  let state = Rng.init_state random_state in
  let n_samples_pre =
    if hole then int_of_float (float n_samples *. 1.25) else n_samples
  in

  let t_pre =
    Rng.uniform state ~low:(1.5 *. pi) ~high:(4.5 *. pi) [| n_samples_pre; 1 |]
  in

  let t, height =
    if hole then (
      let t_flat = Nx.reshape [| n_samples_pre |] t_pre in
      let mask_lower = Nx.cmpgt t_flat (Nx.scalar Float32 10.5) in
      let mask_upper = Nx.cmplt t_flat (Nx.scalar Float32 14.0) in
      let hole_mask = Nx.logical_and mask_lower mask_upper in
      let keep_mask = Nx.logical_not hole_mask in

      (* Find indices where keep_mask is true *)
      let keep_array = Nx.to_array (Nx.cast UInt8 keep_mask) in
      let indices = ref [] in
      Array.iteri (fun i v -> if v > 0 then indices := i :: !indices) keep_array;
      let indices_array = Array.of_list (List.rev !indices) in
      let final_indices =
        Array.sub indices_array 0 (min n_samples (Array.length indices_array))
      in
      let t = Rng.take_indices t_pre ~axis:0 final_indices in
      let height = Rng.uniform state ~low:0. ~high:21. [| Nx.dim 0 t; 1 |] in
      (t, height))
    else
      let height = Rng.uniform state ~low:0. ~high:21. [| n_samples; 1 |] in
      (Nx.slice [ Nx.Rs (0, n_samples, 1); Nx.I 0 ] t_pre, height)
  in
  let final_n_samples = Nx.dim 0 t in

  let x_coord = Nx.mul t (Nx.cos t) in
  let z_coord = Nx.mul t (Nx.sin t) in
  let x_coord = Nx.reshape [| final_n_samples; 1 |] x_coord in
  let z_coord = Nx.reshape [| final_n_samples; 1 |] z_coord in
  let x = Nx.concatenate ~axis:1 [ x_coord; height; z_coord ] in
  let x =
    if noise > 0.0 then
      Nx.add x
        (Nx.mul_s (Rng.standard_normal state [| final_n_samples; 3 |]) noise)
    else x
  in

  (x, Nx.reshape [| final_n_samples |] t)

let make_low_rank_matrix ?(n_samples = 100) ?(n_features = 100)
    ?(effective_rank = 10) ?(tail_strength = 0.5) ?random_state () =
  let state = Rng.init_state random_state in
  (* Create low-rank matrix as product of two smaller matrices *)
  let a = Rng.standard_normal state [| n_samples; effective_rank |] in
  let b = Rng.standard_normal state [| effective_rank; n_features |] in
  let low_rank_part = Nx.matmul a b in

  (* Add some small noise controlled by tail_strength *)
  let noise =
    Nx.mul_s
      (Rng.standard_normal state [| n_samples; n_features |])
      (tail_strength *. 0.01)
  in
  Nx.add low_rank_part noise

let make_sparse_coded_signal ~n_samples ~n_components ~n_features
    ~n_nonzero_coefs ?random_state () =
  let state = Rng.init_state random_state in

  let d = Rng.standard_normal state [| n_features; n_components |] in
  let d = Nx.div d (Nx.sqrt (Nx.sum ~axes:[| 0 |] (Nx.square d))) in

  let x = Nx.init Float32 [| n_components; n_samples |] (fun _ -> 0.) in
  let indices = Array.init n_components (fun j -> j) in
  for i = 0 to n_samples - 1 do
    Rng.shuffle state indices;
    for j = 0 to n_nonzero_coefs - 1 do
      let coef = Rng.normal state () in
      Nx.set_item [ indices.(j); i ] coef x
    done
  done;

  let y = Nx.matmul d x in
  (y, d, x)

let make_spd_matrix ?(n_dim = 30) ?random_state () =
  let state = Rng.init_state random_state in
  let a = Rng.standard_normal state [| n_dim; n_dim |] in
  let spd = Nx.matmul (Nx.transpose a) a in
  Nx.add spd (Nx.mul_s (Nx.eye Float32 n_dim) 0.01)

let make_sparse_spd_matrix ?(n_dim = 30) ?(alpha = 0.95) ?(norm_diag = false)
    ?(smallest_coef = 0.1) ?(largest_coef = 0.9) ?random_state () =
  let _ = norm_diag in
  (* norm_diag not implemented for simplicity *)
  let state = Rng.init_state random_state in

  let a =
    Nx.init Float32 [| n_dim; n_dim |] (fun idx ->
        let i, j = (idx.(0), idx.(1)) in
        if i > j then 0. (* Work on upper triangle only *)
        else if Random.State.float state 1. > alpha then
          let coef =
            smallest_coef
            +. Random.State.float state (largest_coef -. smallest_coef)
          in
          if Random.State.bool state then coef else -.coef
        else 0.)
  in
  (* Make symmetric by averaging with transpose *)
  let a_sym = Nx.mul_s (Nx.add a (Nx.transpose ~axes:[| 1; 0 |] a)) 0.5 in
  (* Make positive definite by A^T * A *)
  let spd = Nx.matmul (Nx.transpose ~axes:[| 1; 0 |] a_sym) a_sym in
  (* Add diagonal to ensure positive definiteness *)
  Nx.add spd (Nx.mul_s (Nx.eye Float32 n_dim) (smallest_coef *. float n_dim))

let make_biclusters ?(shape = (100, 100)) ?(n_clusters = 5) ?(noise = 0.0)
    ?(minval = 10) ?(maxval = 100) ?(shuffle = true) ?random_state () =
  let state = Rng.init_state random_state in
  let n_rows, n_cols = shape in

  let rows_per_cluster = n_rows / n_clusters in
  let cols_per_cluster = n_cols / n_clusters in

  let row_labels =
    Nx.init Int32 [| n_rows |] (fun idx ->
        Int32.of_int (min (idx.(0) / rows_per_cluster) (n_clusters - 1)))
  in
  let col_labels =
    Nx.init Int32 [| n_cols |] (fun idx ->
        Int32.of_int (min (idx.(0) / cols_per_cluster) (n_clusters - 1)))
  in

  let x = Nx.zeros Float32 [| n_rows; n_cols |] in
  for c = 0 to n_clusters - 1 do
    let value = float (minval + Random.State.int state (maxval - minval)) in
    let row_mask = Nx.cmpeq row_labels (Nx.scalar Int32 (Int32.of_int c)) in
    let col_mask = Nx.cmpeq col_labels (Nx.scalar Int32 (Int32.of_int c)) in
    let bicluster_mask =
      Nx.logical_and
        (Nx.broadcast_to [| n_rows; n_cols |]
           (Nx.reshape [| n_rows; 1 |] row_mask))
        (Nx.broadcast_to [| n_rows; n_cols |] col_mask)
    in
    let to_add = Nx.mul_s (Nx.cast Float32 bicluster_mask) value in
    ignore (Nx.iadd x to_add)
  done;

  let x =
    if noise > 0.0 then
      Nx.add x (Nx.mul_s (Rng.standard_normal state [| n_rows; n_cols |]) noise)
    else x
  in

  if shuffle then (
    let row_indices = Array.init n_rows (fun i -> i) in
    Rng.shuffle state row_indices;
    let x = Rng.take_indices x ~axis:0 row_indices in
    let row_labels = Rng.take_indices row_labels ~axis:0 row_indices in

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

  let rows_per_cluster = n_rows / n_clusters_row in
  let cols_per_cluster = n_cols / n_clusters_col in

  let row_labels =
    Nx.init Int32 [| n_rows |] (fun idx ->
        Int32.of_int (min (idx.(0) / rows_per_cluster) (n_clusters_row - 1)))
  in
  let col_labels =
    Nx.init Int32 [| n_cols |] (fun idx ->
        Int32.of_int (min (idx.(0) / cols_per_cluster) (n_clusters_col - 1)))
  in
  let cluster_sum =
    Nx.add
      (Nx.broadcast_to [| n_rows; n_cols |]
         (Nx.reshape [| n_rows; 1 |] row_labels))
      (Nx.broadcast_to [| n_rows; n_cols |] col_labels)
  in
  let is_high_mask =
    Nx.cmpeq (Nx.mod_s cluster_sum (Int32.of_int 2)) (Nx.scalar Int32 0l)
  in
  let x =
    Nx.where is_high_mask
      (Nx.full Float32 [||] (float maxval))
      (Nx.full Float32 [||] (float minval))
  in

  let x =
    if noise > 0.0 then
      Nx.add x (Nx.mul_s (Rng.standard_normal state [| n_rows; n_cols |]) noise)
    else x
  in

  if shuffle then (
    let row_indices = Array.init n_rows (fun i -> i) in
    Rng.shuffle state row_indices;
    let x = Rng.take_indices x ~axis:0 row_indices in
    let row_labels = Rng.take_indices row_labels ~axis:0 row_indices in

    let col_indices = Array.init n_cols (fun i -> i) in
    Rng.shuffle state col_indices;
    let x = Rng.take_indices x ~axis:1 col_indices in
    let col_labels = Rng.take_indices col_labels ~axis:0 col_indices in
    (x, row_labels, col_labels))
  else (x, row_labels, col_labels)
