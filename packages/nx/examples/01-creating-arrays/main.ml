(** Build arrays from scratch: constants, ranges, grids, and custom data.

    This example walks through the most common ways to create arrays. By the end
    you'll know how to pick a dtype, fill arrays with constants, generate
    ranges, and build grids and triangular matrices. Each array is made on the
    host. *)

let () =
  (* Constant-filled arrays: zeros, ones, and an arbitrary fill value. *)
  let z = Nx.zeros Nx.float32 [| 2; 3 |] in
  Printf.printf "zeros (2×3):\n%s\n\n" (Nx.to_string z);

  let o = Nx.ones Nx.float64 [| 3 |] in
  Printf.printf "ones (3):\n%s\n\n" (Nx.to_string o);

  let pi = Nx.full Nx.float64 [| 2; 2 |] Float.pi in
  Printf.printf "full π (2×2):\n%s\n\n" (Nx.to_string pi);

  (* Ranges: integer steps and evenly-spaced floats. *)
  let ints = Nx.arange Nx.int32 0 10 1 in
  Printf.printf "arange 0..9:\n%s\n\n" (Nx.to_string ints);

  let spaced = Nx.linspace Nx.float64 0.0 1.0 5 in
  Printf.printf "linspace 0..1 (5 points):\n%s\n\n" (Nx.to_string spaced);

  let decades = Nx.logspace Nx.float64 1.0 4.0 4 in
  Printf.printf "logspace 10¹..10⁴:\n%s\n\n" (Nx.to_string decades);

  (* From raw data: pack an OCaml array into a 2×3 matrix. *)
  let data =
    Nx.create Nx.float64 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  Printf.printf "create from data (2×3):\n%s\n\n" (Nx.to_string data);

  (* Build a multiplication table with [init]. *)
  let mul_table =
    Nx.init Nx.int32 [| 5; 5 |] (fun idx ->
        Int32.of_int ((idx.(0) + 1) * (idx.(1) + 1)))
  in
  Printf.printf "5×5 multiplication table:\n%s\n\n" (Nx.to_string mul_table);

  (* Identity and eye: diagonal matrices. *)
  let id = Nx.eye Nx.float64 3 in
  Printf.printf "identity 3×3:\n%s\n\n" (Nx.to_string id);

  let e = Nx.eye ~k:1 Nx.float64 3 in
  Printf.printf "eye (k=1, superdiagonal):\n%s\n\n" (Nx.to_string e);

  (* Coordinate grids with meshgrid. With [`Xy], [xs] runs along columns and
     [ys] along rows, as a plane's x and y do. *)
  let xs = Nx.arange_f Nx.float64 0.0 3.0 1.0 in
  let ys = Nx.arange_f Nx.float64 0.0 2.0 1.0 in
  List.iter2
    (fun name grid ->
      Printf.printf "meshgrid %s:\n%s\n" name (Nx.to_string grid))
    [ "X"; "Y" ]
    (Nx.meshgrid ~indexing:`Xy [ xs; ys ]);
  print_newline ();

  (* Triangular matrices: tril and triu. *)
  let m = Nx.ones Nx.float64 [| 4; 4 |] in
  Printf.printf "tril (lower triangle):\n%s\n" (Nx.to_string (Nx.tril m));
  Printf.printf "triu (upper triangle):\n%s\n" (Nx.to_string (Nx.triu m))
