(** DBSCAN density-based clustering.

    Generate two dense clusters with scattered noise, find clusters using
    neighbourhood density, and report cluster sizes and noise count. *)

let () =
  let eps = 1.5 in
  let min_samples = 5 in

  (* Two tight blobs plus uniform noise *)
  let data =
    Nx.(Infix.(
      let c1 = (Rng.normal float64 [| 80; 2 |] *$ 0.6) +$ 3.0 in
      let c2 = (Rng.normal float64 [| 80; 2 |] *$ 0.6) -$ 3.0 in
      let noise = (Rng.uniform float64 [| 20; 2 |] *$ 14.0) -$ 7.0 in
      concatenate ~axis:0 [ c1; c2; noise ]))
  in
  let n = Nx.dim 0 data in
  Printf.printf "Data: %d points (eps=%.1f, min_samples=%d)\n\n" n eps
    min_samples;

  (* Pairwise Euclidean distance matrix [n, n] *)
  let diff =
    Nx.sub (Nx.unsqueeze ~axes:[ 1 ] data) (Nx.unsqueeze ~axes:[ 0 ] data)
  in
  let dist = Nx.sqrt (Nx.sum ~axes:[ 2 ] (Nx.square diff)) in

  (* Neighbour adjacency and core-point mask *)
  let neighbours = Nx.less_equal_s dist eps in
  let counts = Nx.sum ~axes:[ 1 ] (Nx.cast Nx.float64 neighbours) in
  let core = Nx.greater_equal_s counts (Float.of_int min_samples) in

  (* BFS cluster expansion *)
  let labels = Array.make n (-1) in
  let cluster_id = ref 0 in
  for i = 0 to n - 1 do
    if labels.(i) = -1 && Nx.item [ i ] core then begin
      let c = !cluster_id in
      incr cluster_id;
      labels.(i) <- c;
      let q = Queue.create () in
      Queue.push i q;
      while not (Queue.is_empty q) do
        let p = Queue.pop q in
        for j = 0 to n - 1 do
          if labels.(j) = -1 && Nx.item [ p; j ] neighbours then begin
            labels.(j) <- c;
            if Nx.item [ j ] core then Queue.push j q
          end
        done
      done
    end
  done;

  let n_clusters = !cluster_id in
  let n_noise =
    Array.fold_left (fun acc l -> if l = -1 then acc + 1 else acc) 0 labels
  in
  Printf.printf "Clusters found: %d\n" n_clusters;
  Printf.printf "Noise points:   %d\n\n" n_noise;
  for c = 0 to n_clusters - 1 do
    let count =
      Array.fold_left (fun acc l -> if l = c then acc + 1 else acc) 0 labels
    in
    Printf.printf "  Cluster %d: %d points\n" c count
  done
