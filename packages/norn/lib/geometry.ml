(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('u, 'f) t = {
  mean : 'u;
  scale : 'u;
  directions : 'u; (* each tensor with a leading axis of [k] *)
  variances : (float, 'f) Nx.t; (* [k] *)
}

type ('u, 'f) gaussian = ('u, 'f) t

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let shape_string s =
  String.concat "; " (Array.to_list (Array.map string_of_int s))

(* Leafwise linear algebra *)

(* [rows k x] is [x], whose leading axis has [k] rows, as a matrix of [k] rows;
   [k] may be [0]. *)
let rows k x =
  let s = Nx.shape x in
  Nx.reshape
    [| k; Array.fold_left ( * ) 1 (Array.sub s 1 (Array.length s - 1)) |]
    x

(* [coefficients u dirs z] is [U z], each direction's inner product with [z], at
   the dtype of [like], shape [[k]]. *)
let coefficients (type f) u (like : (float, f) Nx.t) dirs z : (float, f) Nx.t =
  let k = (Nx.shape like).(0) in
  let products =
    Nx.Ptree.map2 u (fun _ d x -> Nx.mul d (Nx.unsqueeze ~axes:[ 0 ] x)) dirs z
  in
  Nx.Ptree.fold u
    (fun _ p acc ->
      if not (Nx_dtype.is Float (Nx.dtype p)) then acc
      else Nx.add acc (Nx.cast (Nx.dtype like) (Nx.sum ~axes:[ 1 ] (rows k p))))
    products (Nx.zeros_like like)

(* [combine u dirs c z] is [z + Σ_j c_j u_j]. *)
let combine u dirs c z =
  let k = (Nx.shape c).(0) in
  Nx.Ptree.map2 u
    (fun _ d x ->
      if not (Nx_dtype.is Float (Nx.dtype x)) then x
      else
        let c = Nx.reshape [| 1; k |] (Nx.cast (Nx.dtype x) c) in
        let flat = Nx.matmul c (rows k d) in
        Nx.add x (Nx.reshape (Nx.shape x) flat))
    dirs z

(* [stretch g u e z] is [z + U ((v^e - 1) ⊙ Uᵀ z)]: [A] at [e = 1/2], its
   inverse at [-1/2]. *)
let stretch g u e z =
  let c = coefficients u g.variances g.directions z in
  let w = Nx.sub_s (Nx.pow g.variances (Nx.scalar_like g.variances e)) 1. in
  combine u g.directions (Nx.mul w c) z

let direction u g z =
  Nx.Ptree.map2 u (fun _ s x -> Nx.mul s x) g.scale (stretch g u 0.5 z)

let color u g z =
  Nx.Ptree.map2 u (fun _ m x -> Nx.add m x) g.mean (direction u g z)

let whiten u g x =
  let d = Nx.Ptree.map2 u (fun _ x m -> Nx.sub x m) x g.mean in
  stretch g u (-0.5) (Nx.Ptree.map2 u (fun _ d s -> Nx.div d s) d g.scale)

let log_det (type f) (g : (_, f) t) u : (float, f) Nx.t =
  let dt = Nx.dtype g.variances in
  let logs =
    Nx.Ptree.fold u
      (fun _ s acc ->
        if not (Nx_dtype.is Float (Nx.dtype s)) then acc
        else Nx.add acc (Nx.cast dt (Nx.sum (Nx.log s))))
      g.scale (Nx.zeros dt [||])
  in
  Nx.add logs (Nx.mul_s (Nx.sum (Nx.log g.variances)) 0.5)

let elements u x =
  Nx.Ptree.fold u
    (fun _ t n -> if Nx_dtype.is Float (Nx.dtype t) then n + Nx.numel t else n)
    x 0

(* Constructors *)

let no_directions u like =
  Nx.Ptree.map u
    (fun _ x -> Nx.zeros (Nx.dtype x) (Array.append [| 0 |] (Nx.shape x)))
    like

let diagonal u dtype ~mean ~scale =
  {
    mean;
    scale;
    directions = no_directions u mean;
    variances = Nx.zeros dtype [| 0 |];
  }

(* [matrix u like dirs] is the directions [dirs] as the columns of a matrix [[d;
   k]] at [like]'s dtype, and [split] its inverse. *)
let matrix (type f) u (like : (float, f) Nx.t) dirs : (float, f) Nx.t =
  let k = (Nx.shape like).(0) in
  let columns =
    Nx.Ptree.fold u
      (fun _ d acc ->
        if not (Nx_dtype.is Float (Nx.dtype d)) then acc
        else Nx.cast (Nx.dtype like) (rows k d) :: acc)
      dirs []
  in
  match columns with
  | [] -> Nx.zeros (Nx.dtype like) [| 0; k |]
  | cs -> Nx.transpose (Nx.concatenate ~axis:1 (List.rev cs))

let split u like m =
  let k = (Nx.shape m).(1) in
  let offset = ref 0 in
  Nx.Ptree.map u
    (fun _ x ->
      if not (Nx_dtype.is Float (Nx.dtype x)) then
        Nx.zeros (Nx.dtype x) (Array.append [| k |] (Nx.shape x))
      else
        let n = Nx.numel x in
        let rows = Nx.shrink [| (!offset, !offset + n); (0, k) |] m in
        offset := !offset + n;
        Nx.reshape
          (Array.append [| k |] (Nx.shape x))
          (Nx.cast (Nx.dtype x) (Nx.transpose rows)))
    like

let check_positive context name v =
  Nx.check Nx.Ptree.tensor
    (Nx.logical_and (Nx.greater v (Nx.zeros_like v)) (Nx.isfinite v))
    v
    (fun i x ->
      Invalid_argument
        (Printf.sprintf "%s: %s at [%s] is %s, not in (0, inf)" context name
           (shape_string i) (Nx.to_string x)))

let low_rank u ~mean ~scale ~directions ~variances =
  let context = "Norn.Gaussian.low_rank" in
  let k = Nx.Ptree.fold u (fun _ d k -> (Nx.shape d).(0) :: k) directions [] in
  (match (Nx.shape variances, k) with
  | [| n |], ks when List.for_all (( = ) n) ks -> ()
  | s, _ ->
      invalid_argf "%s: variances of shape [%s] do not match the directions"
        context (shape_string s));
  check_positive context "variances" variances;
  let q, _ = Nx.qr (matrix u variances directions) in
  { mean; scale; directions = split u mean q; variances }

let of_precision (type f) u (dtype : (float, f) Nx.dtype) ?low_rank ~mean p :
    (_, f) t =
  let context = "Norn.Gaussian.of_precision" in
  Nx.Ptree.fold u
    (fun path x () ->
      if Nx_dtype.is Float (Nx.dtype x) then
        Nx.check Nx.Ptree.tensor
          (Nx.logical_and (Nx.greater x (Nx.zeros_like x)) (Nx.isfinite x))
          x
          (fun i v ->
            Invalid_argument
              (Printf.sprintf "%s: %s at [%s] is %s, not in (0, inf)" context
                 (Nx.Ptree.Path.to_string path)
                 (shape_string i) (Nx.to_string v))))
    p ();
  let scale = Nx.Ptree.map u (fun _ x -> Nx.rsqrt x) p in
  match low_rank with
  | None -> diagonal u dtype ~mean ~scale
  | Some (w, l) ->
      Nx.check Nx.Ptree.tensor
        (Nx.logical_and (Nx.greater_equal l (Nx.zeros_like l)) (Nx.isfinite l))
        l
        (fun i v ->
          Invalid_argument
            (Printf.sprintf
               "%s: the low-rank weights at [%s] are %s, not in [0, inf)"
               context (shape_string i) (Nx.to_string v)));
      (* With [ŵ = S w], the precision is [S⁻¹ (I + Σ l_j ŵ_j ŵ_jᵀ) S⁻¹]: from
         [B = Ŵ diag (sqrt l) = Q R], the inner matrix is [I + Q (R Rᵀ) Qᵀ],
         whose eigenvectors [Q V] carry the covariance's variances [1 / (1 +
         μ)], [μ] the eigenvalues of [R Rᵀ]. *)
      let scaled =
        Nx.Ptree.map2 u
          (fun _ w s -> Nx.mul w (Nx.unsqueeze ~axes:[ 0 ] s))
          w scale
      in
      let b =
        Nx.mul (matrix u l scaled) (Nx.unsqueeze ~axes:[ 0 ] (Nx.sqrt l))
      in
      let q, r = Nx.qr b in
      let mu, v = Nx.eigh (Nx.matmul r (Nx.transpose r)) in
      let mu = Nx.cast dtype mu in
      let variances = Nx.recip (Nx.add_s mu 1.) in
      { mean; scale; directions = split u mean (Nx.matmul q v); variances }

(* Eliminators *)

let batched u f x = Rune.vmap Nx.Ptree.(u @-> returns tensor) f x

let log_density u g x =
  let dt = Nx.dtype g.variances in
  let d = float_of_int (elements u g.mean) in
  let normaliser =
    Nx.add_s (log_det g u) (0.5 *. d *. Float.log (2. *. Float.pi))
  in
  batched u
    (fun x ->
      let z = whiten u g x in
      let sq =
        Nx.Ptree.fold u
          (fun _ t acc ->
            if Nx_dtype.is Float (Nx.dtype t) then
              Nx.add acc (Nx.cast dt (Nx.sum (Nx.square t)))
            else acc)
          z (Nx.zeros dt [||])
      in
      Nx.sub (Nx.mul_s sq (-0.5)) normaliser)
    x

let sample u k ~n g =
  if n < 0 then invalid_argf "Norn.Gaussian.sample: n = %d is negative" n;
  let j = ref (-1) in
  let z =
    Nx.Ptree.map u
      (fun _ m ->
        incr j;
        Noise.normal_like m (Nx.Rng.fold_in k !j)
          (Array.append [| n |] (Nx.shape m)))
      g.mean
  in
  Rune.vmap Nx.Ptree.(u @-> returns u) (color u g) z

let mean _ g = g.mean
let rank g = (Nx.shape g.variances).(Nx.ndim g.variances - 1)

let variance u g =
  (* [S² (1 + Σ_j (v_j - 1) u_j²)], element by element. *)
  let w = Nx.sub_s g.variances 1. in
  let k = (Nx.shape w).(0) in
  Nx.Ptree.map2 u
    (fun _ s d ->
      if not (Nx_dtype.is Float (Nx.dtype s)) then s
      else
        let w =
          Nx.reshape
            (Array.append [| k |] (Array.make (Nx.ndim s) 1))
            (Nx.cast (Nx.dtype s) w)
        in
        let along = Nx.sum ~axes:[ 0 ] (Nx.mul w (Nx.square d)) in
        Nx.mul (Nx.square s) (Nx.add along (Nx.ones_like along)))
    g.scale g.directions

let ptree (type u f) (u : u Nx.Ptree.t) : (u, f) t Nx.Ptree.t =
  let module S = struct
    type _ t = (u, f) gaussian

    let walk c g =
      let open Nx.Ptree.Walk in
      let mean = field c "mean" (structure u) g.mean in
      let scale = field c "scale" (structure u) g.scale in
      let directions = field c "directions" (structure u) g.directions in
      let variances = field c "variances" tensor g.variances in
      { mean; scale; directions; variances }
  end in
  Nx.Ptree.nest (module S) Nx.Ptree.unit

let pp u ppf g =
  Format.fprintf ppf "gaussian(dimension %d, rank %d)" (elements u g.mean)
    (Nx.shape g.variances).(0)
