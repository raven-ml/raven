(** K-means clustering with kmeans++ initialisation.

    Generate synthetic blobs, cluster them with Lloyd's algorithm, and report
    centroid positions and inertia. *)

open Nx

(* Pairwise squared L2 distances: [n, d] x [k, d] -> [n, k] *)
let sq_distances a b =
  sum ~axes:[ 2 ] (square (sub (expand_dims [ 1 ] a) (expand_dims [ 0 ] b)))

(* Isotropic Gaussian blobs around given centres. *)
let make_blobs ~samples_per_cluster centers =
  let d = (shape centers).(1) in
  let blobs =
    List.init
      (shape centers).(0)
      (fun c ->
        add (randn Float64 [| samples_per_cluster; d |]) (get [ c ] centers))
  in
  shuffle (concatenate ~axis:0 blobs)

(* Kmeans++ initialisation: pick k centres from data. *)
let kmeanspp data k =
  let n = (shape data).(0) in
  let d = (shape data).(1) in
  let centroids = zeros Float64 [| k; d |] in
  let idx = Int32.to_int (item [] (randint Int32 ~high:n [||] 0)) in
  set [ 0 ] centroids (get [ idx ] data);
  for c = 1 to k - 1 do
    let current = slice [ R (0, c); A ] centroids in
    let min_d = min ~axes:[ 1 ] (sq_distances data current) in
    let chosen =
      Int32.to_int (item [] (categorical (log (clamp ~min:1e-30 min_d))))
    in
    set [ c ] centroids (get [ chosen ] data)
  done;
  centroids

let () =
  let true_centers =
    create Float64 [| 3; 2 |] [| 0.0; 0.0; 7.0; 7.0; -5.0; 10.0 |]
  in
  let data = make_blobs ~samples_per_cluster:100 true_centers in
  let n = (shape data).(0) in
  let d = (shape data).(1) in
  let k = 3 in
  Printf.printf "Data: %d points, %d features, %d clusters\n\n" n d k;

  let centroids = kmeanspp data k in
  let labels = ref (zeros Int32 [| n |]) in
  let max_iter = 100 in
  let tol = 1e-6 in
  let converged = ref false in
  let iter = ref 0 in
  while !iter < max_iter && not !converged do
    labels := argmin ~axis:1 (sq_distances data centroids);

    let old = copy centroids in
    for c = 0 to k - 1 do
      let mask = cast Float64 (equal !labels (scalar Int32 (Int32.of_int c))) in
      let count = item [] (sum mask) in
      if count > 0.0 then begin
        let total = sum ~axes:[ 0 ] (mul data (expand_dims [ 1 ] mask)) in
        set [ c ] centroids (div_s total count)
      end
    done;

    let shift = item [] (max (abs (sub centroids old))) in
    converged := shift < tol;
    incr iter
  done;

  Printf.printf "Converged after %d iterations\n\n" !iter;
  Printf.printf "Centroids:\n%s\n" (data_to_string centroids);

  for c = 0 to k - 1 do
    let count =
      item []
        (sum (cast Float64 (equal !labels (scalar Int32 (Int32.of_int c)))))
    in
    Printf.printf "  Cluster %d: %.0f points\n" c count
  done;

  let inertia = item [] (sum (min ~axes:[ 1 ] (sq_distances data centroids))) in
  Printf.printf "\nInertia: %.2f\n" inertia
