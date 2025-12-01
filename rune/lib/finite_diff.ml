open Nx_core
open Nx_rune
module T = Tensor

type method_ = [ `Central | `Forward | `Backward ]

let default_eps = 1e-4 (* Better for float32 precision *)

let finite_diff (type a b c d) ?(eps = default_eps) ?(method_ = `Central)
    (f : (a, b) T.t -> (c, d) T.t) (x : (a, b) T.t) : (a, b) T.t =
  let x_shape = T.shape x in
  let x_numel = Array.fold_left ( * ) 1 x_shape in

  if x_numel = 0 then T.zeros (dtype x) x_shape
  else
    (* Create epsilon scalar with proper type *)
    let eps_scalar =
      let dt = dtype x in
      T.full dt [||] (Dtype.of_float dt eps)
    in

    (* For simple scalar case *)
    if x_numel = 1 then
      match method_ with
      | `Central ->
          let x_plus = T.add x eps_scalar in
          let x_minus = T.sub x eps_scalar in
          let f_plus = f x_plus in
          let f_minus = f x_minus in
          (* Cast result back to input type *)
          let result = T.sub f_plus f_minus in
          let two_eps = T.add eps_scalar eps_scalar in
          (* We need to cast the result to match input type *)
          T.cast (dtype x) (T.div result (T.cast (dtype result) two_eps))
      | `Forward ->
          let x_plus = T.add x eps_scalar in
          let f_plus = f x_plus in
          let f_x = f x in
          let result = T.sub f_plus f_x in
          T.cast (dtype x) (T.div result (T.cast (dtype result) eps_scalar))
      | `Backward ->
          let x_minus = T.sub x eps_scalar in
          let f_x = f x in
          let f_minus = f x_minus in
          let result = T.sub f_x f_minus in
          T.cast (dtype x) (T.div result (T.cast (dtype result) eps_scalar))
    else
      (* For vector/matrix case - need to compute gradient elementwise *)
      let grad = T.zeros (dtype x) x_shape in
      let x_flat = T.reshape [| x_numel |] x in
      let grad_flat = T.reshape [| x_numel |] grad in

      for i = 0 to x_numel - 1 do
        let x_copy_plus = T.copy x_flat in
        let x_copy_minus = T.copy x_flat in

        let current_val = T.get [ i ] x_flat in

        match method_ with
        | `Central ->
            T.set [ i ] x_copy_plus (T.add current_val eps_scalar);
            T.set [ i ] x_copy_minus (T.sub current_val eps_scalar);

            let x_plus = T.reshape x_shape x_copy_plus in
            let x_minus = T.reshape x_shape x_copy_minus in

            let f_plus = f x_plus in
            let f_minus = f x_minus in

            if T.shape f_plus <> [||] then
              failwith "finite_diff: function must return scalar";

            let two_eps = T.add eps_scalar eps_scalar in
            let result = T.sub f_plus f_minus in
            let grad_i =
              T.cast (dtype x) (T.div result (T.cast (dtype result) two_eps))
            in
            T.set [ i ] grad_flat grad_i
        | `Forward ->
            T.set [ i ] x_copy_plus (T.add current_val eps_scalar);

            let x_plus = T.reshape x_shape x_copy_plus in
            let x_orig = T.reshape x_shape x_flat in

            let f_plus = f x_plus in
            let f_x = f x_orig in

            if T.shape f_plus <> [||] then
              failwith "finite_diff: function must return scalar";

            let result = T.sub f_plus f_x in
            let grad_i =
              T.cast (dtype x) (T.div result (T.cast (dtype result) eps_scalar))
            in
            T.set [ i ] grad_flat grad_i
        | `Backward ->
            T.set [ i ] x_copy_minus (T.sub current_val eps_scalar);

            let x_minus = T.reshape x_shape x_copy_minus in
            let x_orig = T.reshape x_shape x_flat in

            let f_x = f x_orig in
            let f_minus = f x_minus in

            if T.shape f_x <> [||] then
              failwith "finite_diff: function must return scalar";

            let result = T.sub f_x f_minus in
            let grad_i =
              T.cast (dtype x) (T.div result (T.cast (dtype result) eps_scalar))
            in
            T.set [ i ] grad_flat grad_i
      done;

      T.reshape x_shape grad_flat

let finite_diff_jacobian (type a b c d) ?(eps = default_eps)
    ?(method_ = `Central) (f : (a, b) T.t -> (c, d) T.t) (x : (a, b) T.t) :
    (c, d) T.t =
  let x_shape = T.shape x in
  let x_numel = Array.fold_left ( * ) 1 x_shape in

  let f_x = f x in
  let output_shape = T.shape f_x in
  let output_numel = Array.fold_left ( * ) 1 output_shape in

  let jacobian = T.zeros (dtype f_x) [| output_numel; x_numel |] in

  if x_numel = 0 || output_numel = 0 then jacobian
  else
    let x_flat = T.reshape [| x_numel |] x in

    (* Create epsilon scalar with proper type *)
    let eps_scalar =
      let dt = dtype x in
      T.full dt [||] (Dtype.of_float dt eps)
    in

    for i = 0 to x_numel - 1 do
      let x_copy_plus = T.copy x_flat in
      let x_copy_minus = T.copy x_flat in

      let current_val = T.get [ i ] x_flat in

      match method_ with
      | `Central ->
          T.set [ i ] x_copy_plus (T.add current_val eps_scalar);
          T.set [ i ] x_copy_minus (T.sub current_val eps_scalar);

          let x_plus = T.reshape x_shape x_copy_plus in
          let x_minus = T.reshape x_shape x_copy_minus in

          let f_plus = T.reshape [| output_numel |] (f x_plus) in
          let f_minus = T.reshape [| output_numel |] (f x_minus) in

          let two_eps_out =
            let dt = dtype f_x in
            T.full dt [||] (Dtype.of_float dt (2.0 *. eps))
          in
          let grad_col = T.div (T.sub f_plus f_minus) two_eps_out in

          for j = 0 to output_numel - 1 do
            T.set [ j; i ] jacobian (T.get [ j ] grad_col)
          done
      | `Forward ->
          T.set [ i ] x_copy_plus (T.add current_val eps_scalar);

          let x_plus = T.reshape x_shape x_copy_plus in

          let f_plus = T.reshape [| output_numel |] (f x_plus) in
          let f_x_flat = T.reshape [| output_numel |] f_x in

          let eps_out =
            let dt = dtype f_x in
            T.full dt [||] (Dtype.of_float dt eps)
          in
          let grad_col = T.div (T.sub f_plus f_x_flat) eps_out in

          for j = 0 to output_numel - 1 do
            T.set [ j; i ] jacobian (T.get [ j ] grad_col)
          done
      | `Backward ->
          T.set [ i ] x_copy_minus (T.sub current_val eps_scalar);

          let x_minus = T.reshape x_shape x_copy_minus in

          let f_x_flat = T.reshape [| output_numel |] f_x in
          let f_minus = T.reshape [| output_numel |] (f x_minus) in

          let eps_out =
            let dt = dtype f_x in
            T.full dt [||] (Dtype.of_float dt eps)
          in
          let grad_col = T.div (T.sub f_x_flat f_minus) eps_out in

          for j = 0 to output_numel - 1 do
            T.set [ j; i ] jacobian (T.get [ j ] grad_col)
          done
    done;

    if output_shape = [||] then T.reshape x_shape (T.get [ 0 ] jacobian)
    else jacobian
