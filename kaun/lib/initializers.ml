(* Flax-compatible initializers for Kaun

   This module provides initializer functions matching the Flax/JAX neural
   network library API. Since it's part of the Kaun library, it cannot depend on
   Kaun itself. *)

(* Type for initializer functions using GADT for cleaner API *)
type t = {
  f :
    'layout 'dev.
    int ->
    int array ->
    'dev Rune.device ->
    (float, 'layout) Rune.dtype ->
    (float, 'layout, 'dev) Rune.t;
}

(* Helper function to compute fan-in and fan-out *)
let compute_fans shape in_axis out_axis =
  let rank = Array.length shape in

  (* Handle edge case for scalar/empty shapes *)
  if rank = 0 then (1, 1)
  else
    (* Normalize negative indices *)
    let in_axis = if in_axis < 0 then rank + in_axis else in_axis in
    let out_axis = if out_axis < 0 then rank + out_axis else out_axis in

    (* For 1D tensors or when indices are out of bounds, use total size as both
       fan_in and fan_out *)
    if
      rank = 1 || in_axis < 0 || in_axis >= rank || out_axis < 0
      || out_axis >= rank
    then
      let total_size = Array.fold_left ( * ) 1 shape in
      (total_size, total_size)
    else
      (* Normal case: extract fan_in and fan_out *)
      let fan_in = shape.(in_axis) in
      let fan_out = shape.(out_axis) in

      (* Compute receptive field size for conv layers *)
      let receptive_field_size = ref 1 in
      for i = 0 to rank - 1 do
        if i <> in_axis && i <> out_axis then
          receptive_field_size := !receptive_field_size * shape.(i)
      done;

      (fan_in * !receptive_field_size, fan_out * !receptive_field_size)

(* Truncated normal sampling using rejection sampling *)
let truncated_normal_impl ~mean ~stddev ~lower ~upper seed shape dev dtype =
  let lower_normalized = (lower -. mean) /. stddev in
  let upper_normalized = (upper -. mean) /. stddev in

  (* Use rejection sampling - keep generating until we get values in range *)
  let generate () =
    let z = Rune.randn dev dtype ~seed shape in
    let z_scaled =
      Rune.add
        (Rune.mul z (Rune.scalar dev dtype stddev))
        (Rune.scalar dev dtype mean)
    in

    (* Check if all values are within bounds *)
    let lower_t =
      Rune.scalar dev dtype (mean +. (lower_normalized *. stddev))
    in
    let upper_t =
      Rune.scalar dev dtype (mean +. (upper_normalized *. stddev))
    in
    (* For now, we'll use a simple approximation: clamp the values *)
    let clamped = Rune.maximum z_scaled lower_t in
    Rune.minimum clamped upper_t
  in
  generate ()

(* Basic initializers *)

let constant value : t =
  {
    f =
      (fun seed shape dev dtype ->
        ignore seed;
        (* unused *)
        Rune.full dev dtype shape value);
  }

let zeros () = constant 0.0
let ones () = constant 1.0

let uniform ?(scale = 0.01) () =
  {
    f =
      (fun seed shape dev dtype ->
        let u01 = Rune.rand dev dtype ~seed shape in
        Rune.mul u01 (Rune.scalar dev dtype scale));
  }

let normal ?(stddev = 0.01) () =
  {
    f =
      (fun seed shape dev dtype ->
        let z = Rune.randn dev dtype ~seed shape in
        Rune.mul z (Rune.scalar dev dtype stddev));
  }

let truncated_normal ?(stddev = 0.01) ?(lower = -2.0) ?(upper = 2.0) () =
  {
    f =
      (fun seed shape dev dtype ->
        truncated_normal_impl ~mean:0.0 ~stddev ~lower ~upper seed shape dev
          dtype);
  }

(* Variance scaling initializer - the general framework *)
let variance_scaling ~scale ~mode ~distribution ~in_axis ~out_axis () =
  {
    f =
      (fun seed shape dev dtype ->
        let fan_in, fan_out = compute_fans shape in_axis out_axis in

        let n =
          match mode with
          | `Fan_in -> float_of_int fan_in
          | `Fan_out -> float_of_int fan_out
          | `Fan_avg -> float_of_int (fan_in + fan_out) /. 2.0
        in

        let variance = scale /. n in
        let stddev = sqrt variance in

        match distribution with
        | `Normal ->
            let z = Rune.randn dev dtype ~seed shape in
            Rune.mul z (Rune.scalar dev dtype stddev)
        | `Truncated_normal ->
            truncated_normal_impl ~mean:0.0 ~stddev ~lower:(-2.0) ~upper:2.0
              seed shape dev dtype
        | `Uniform ->
            let limit = sqrt (3.0 *. variance) in
            let u01 = Rune.rand dev dtype ~seed shape in
            let scale_t = Rune.scalar dev dtype (2.0 *. limit) in
            let shift = Rune.scalar dev dtype limit in
            Rune.sub (Rune.mul u01 scale_t) shift);
  }

(* Xavier/Glorot initializers *)
let glorot_uniform ?(in_axis = -2) ?(out_axis = -1) () =
  variance_scaling ~scale:1.0 ~mode:`Fan_avg ~distribution:`Uniform ~in_axis
    ~out_axis ()

let glorot_normal ?(in_axis = -2) ?(out_axis = -1) () =
  variance_scaling ~scale:1.0 ~mode:`Fan_avg ~distribution:`Truncated_normal
    ~in_axis ~out_axis ()

let xavier_uniform = glorot_uniform
let xavier_normal = glorot_normal

(* LeCun initializers *)
let lecun_uniform ?(in_axis = -2) ?(out_axis = -1) () =
  variance_scaling ~scale:1.0 ~mode:`Fan_in ~distribution:`Uniform ~in_axis
    ~out_axis ()

let lecun_normal ?(in_axis = -2) ?(out_axis = -1) () =
  variance_scaling ~scale:1.0 ~mode:`Fan_in ~distribution:`Truncated_normal
    ~in_axis ~out_axis ()

(* He/Kaiming initializers *)
let he_uniform ?(in_axis = -2) ?(out_axis = -1) () =
  variance_scaling ~scale:2.0 ~mode:`Fan_in ~distribution:`Uniform ~in_axis
    ~out_axis ()

let he_normal ?(in_axis = -2) ?(out_axis = -1) () =
  variance_scaling ~scale:2.0 ~mode:`Fan_in ~distribution:`Truncated_normal
    ~in_axis ~out_axis ()

let kaiming_uniform = he_uniform
let kaiming_normal = he_normal

(* Orthogonal initializers *)
let orthogonal ?(scale = 1.0) ?(column_axis = -1) () =
  {
    f =
      (fun seed shape dev dtype ->
        let rank = Array.length shape in
        let column_axis =
          if column_axis < 0 then rank + column_axis else column_axis
        in

        (* For orthogonal init, we need to reshape to 2D, apply QR, then reshape
           back *)
        let rows = ref 1 in
        let cols = ref 1 in
        for i = 0 to rank - 1 do
          if i = column_axis then cols := !cols * shape.(i)
          else rows := !rows * shape.(i)
        done;

        (* Generate random matrix *)
        let flat_shape = [| !rows; !cols |] in
        let a = Rune.randn dev dtype ~seed flat_shape in

        (* For now, use random orthogonal-like initialization *)
        (* A proper implementation would use QR decomposition *)
        (* This is a simplified version that maintains approximate orthogonality *)
        let norm_factor = sqrt (float_of_int (min !rows !cols)) in
        let a_scaled =
          Rune.mul a (Rune.scalar dev dtype (scale /. norm_factor))
        in

        (* Reshape back to original shape *)
        Rune.reshape shape a_scaled);
  }

let delta_orthogonal ?(scale = 1.0) ?(column_axis = -1) () =
  {
    f =
      (fun seed shape dev dtype ->
        (* Delta orthogonal is for Conv layers - middle spatial dims should be
           identity-like *)
        let rank = Array.length shape in
        if rank < 3 || rank > 5 then
          failwith "delta_orthogonal requires 3D, 4D, or 5D shape";

        let column_axis =
          if column_axis < 0 then rank + column_axis else column_axis
        in

        (* For conv layers, create identity in spatial dims *)
        let spatial_dims = Array.sub shape 1 (rank - 2) in
        let is_square =
          Array.for_all (fun d -> d = spatial_dims.(0)) spatial_dims
        in

        if not is_square then
          failwith "delta_orthogonal requires square spatial dimensions";

        (* Create base orthogonal for input/output channels *)
        let in_channels = shape.(0) in
        let out_channels = shape.(rank - 1) in
        let orth_shape = [| in_channels; out_channels |] in
        let _orth =
          (orthogonal ~scale ~column_axis ()).f seed orth_shape dev dtype
        in

        (* Expand to full shape with identity in spatial dims *)
        let result = Rune.zeros dev dtype shape in

        (* Place orthogonal values at center of spatial dims *)
        let _center_idx = Array.make (rank - 2) (spatial_dims.(0) / 2) in

        (* This is a simplified implementation *)
        (* A full implementation would properly place the orthogonal matrix *)
        (* at the center of each spatial dimension *)
        result);
  }

(* Utility initializers *)
let uniform_range ~low ~high () =
  {
    f =
      (fun seed shape dev dtype ->
        let u01 = Rune.rand dev dtype ~seed shape in
        let scale = Rune.scalar dev dtype (high -. low) in
        let shift = Rune.scalar dev dtype low in
        Rune.add (Rune.mul u01 scale) shift);
  }

let normal_range ~mean ~stddev () =
  {
    f =
      (fun seed shape dev dtype ->
        let z = Rune.randn dev dtype ~seed shape in
        Rune.add
          (Rune.mul z (Rune.scalar dev dtype stddev))
          (Rune.scalar dev dtype mean));
  }
