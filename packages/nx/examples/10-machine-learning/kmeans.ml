(** K-means clustering with kmeans++ initialisation.

    Generate synthetic blobs, cluster them with Lloyd's algorithm, and report
    centroid positions and inertia. *)

(* Pairwise squared L2 distances: [n, d] x [k, d] -> [n, k] *)
let sq_distances a b =
  let diff = Nx.sub (Nx.unsqueeze ~axes:[ 1 ] a) (Nx.unsqueeze ~axes:[ 0 ] b) in
  Nx.sum ~axes:[ 2 ] (Nx.square diff)

(* Isotropic Gaussian blobs around given centres, rows shuffled. *)
let make_blobs ~samples_per_cluster centers =
  let d = Nx.dim 1 centers in
  let blobs =
    List.init (Nx.dim 0 centers) (fun c ->
        Nx.add
          (Nx.Rng.normal Nx.float64 [| samples_per_cluster; d |])
          (Nx.get [ c ] centers))
  in
  let data = Nx.concatenate ~axis:0 blobs in
  Nx.take ~axis:0 (Nx.Rng.permutation (Nx.dim 0 data)) data

(* Kmeans++ initialisation: pick k centres from data. *)
let kmeanspp data k =
  let n = Nx.dim 0 data in
  let d = Nx.dim 1 data in
  let idx = Int32.to_int (Nx.item [] (Nx.Rng.randint ~high:n [||])) in
  let centroids = ref (Nx.reshape [| 1; d |] (Nx.get [ idx ] data)) in
  for _ = 1 to k - 1 do
    let min_d = Nx.min ~axes:[ 1 ] (sq_distances data !centroids) in
    let logits = Nx.log (Nx.clamp ~min:1e-30 min_d) in
    let chosen = Int64.to_int (Nx.item [] (Nx.Rng.categorical logits)) in
    centroids :=
      Nx.concatenate ~axis:0
        [ !centroids; Nx.reshape [| 1; d |] (Nx.get [ chosen ] data) ]
  done;
  !centroids

(* The number of points with label [c]. *)
let cluster_size labels c =
  Int64.to_float (Nx.item [] (Nx.count (Nx.equal_s labels (Int64.of_int c))))

let () =
  let true_centers =
    Nx.create Nx.float64 [| 3; 2 |] [| 0.0; 0.0; 7.0; 7.0; -5.0; 10.0 |]
  in
  let data = make_blobs ~samples_per_cluster:100 true_centers in
  let n = Nx.dim 0 data in
  let d = Nx.dim 1 data in
  let k = 3 in
  Printf.printf "Data: %d points, %d features, %d clusters\n\n" n d k;

  let centroids = ref (kmeanspp data k) in
  let labels = ref (Nx.zeros Nx.int64 [| n |]) in
  let max_iter = 100 in
  let tol = 1e-6 in
  let converged = ref false in
  let iter = ref 0 in
  while !iter < max_iter && not !converged do
    labels := Nx.argmin ~axis:1 (sq_distances data !centroids);

    let old = !centroids in
    for c = 0 to k - 1 do
      let mask =
        Nx.cast Nx.float64 (Nx.equal_s !labels (Int64.of_int c))
      in
      let count = cluster_size !labels c in
      if count > 0.0 then begin
        let total =
          Nx.sum ~axes:[ 0 ] (Nx.mul data (Nx.unsqueeze ~axes:[ 1 ] mask))
        in
        centroids := Nx.set [ Nx.I c ] (Nx.div_s total count) !centroids
      end
    done;

    let shift = Nx.item [] (Nx.max (Nx.abs (Nx.sub !centroids old))) in
    converged := shift < tol;
    incr iter
  done;

  Printf.printf "Converged after %d iterations\n\n" !iter;
  Printf.printf "Centroids:\n%s\n" (Nx.to_string !centroids);

  for c = 0 to k - 1 do
    Printf.printf "  Cluster %d: %.0f points\n" c (cluster_size !labels c)
  done;

  let inertia =
    Nx.item [] (Nx.sum (Nx.min ~axes:[ 1 ] (sq_distances data !centroids)))
  in
  Printf.printf "\nInertia: %.2f\n" inertia
