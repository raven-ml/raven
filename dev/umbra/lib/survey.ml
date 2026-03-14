(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let f64 = Nx.float64
let c_km_s = 299792.458
let h0_ref = 100.0
let steradian_to_arcmin2 = 11818102.86004228
let c1_rho_crit = 0.0134

(* Redshift distributions *)

type nz = { eval : Nx.float64_t -> Nx.float64_t; zmax : float }

let simps_float f a b n =
  let h = (b -. a) /. Float.of_int n in
  let sum = ref (f a +. f b) in
  for i = 1 to n - 1 do
    let x = a +. (Float.of_int i *. h) in
    let w = if i mod 2 = 1 then 4.0 else 2.0 in
    sum := !sum +. (w *. f x)
  done;
  !sum *. h /. 3.0

let smail ?(zmax = 10.0) ~a ~b ~z0 () =
  let raw z_f = (z_f ** a) *. Float.exp (-.((z_f /. z0) ** b)) in
  let norm = simps_float raw 0.0 zmax 256 in
  let eval z =
    let z_f = Nx.item [] z in
    Nx.scalar f64 (raw z_f /. norm)
  in
  { eval; zmax }

let tabulated ~z ~pz () =
  let n = (Nx.shape z).(0) in
  let zmax = Nx.item [ n - 1 ] z in
  let norm = ref 0.0 in
  for i = 0 to n - 2 do
    let dz = Nx.item [ i + 1 ] z -. Nx.item [ i ] z in
    norm := !norm +. (0.5 *. (Nx.item [ i ] pz +. Nx.item [ i + 1 ] pz) *. dz)
  done;
  let eval zq =
    let zq_f = Nx.item [] zq in
    if zq_f <= Nx.item [ 0 ] z || zq_f >= zmax then Nx.scalar f64 0.0
    else begin
      let idx = ref 0 in
      for i = 0 to n - 2 do
        if Nx.item [ i ] z <= zq_f then idx := i
      done;
      let i = !idx in
      let z0 = Nx.item [ i ] z and z1 = Nx.item [ i + 1 ] z in
      let p0 = Nx.item [ i ] pz and p1 = Nx.item [ i + 1 ] pz in
      let frac = (zq_f -. z0) /. (z1 -. z0) in
      Nx.scalar f64 ((p0 +. (frac *. (p1 -. p0))) /. !norm)
    end
  in
  { eval; zmax }

let custom_nz ?(zmax = 10.0) eval = { eval; zmax }
let eval_nz nz z = nz.eval z
let nz_zmax nz = nz.zmax

(* Galaxy bias *)

type bias = Cosmo.params -> Nx.float64_t -> Nx.float64_t

let constant_bias b _p _z = Nx.scalar f64 b

let inverse_growth_bias b0 p z =
  let d = Cosmo.growth_factor ~p z in
  Nx.div (Nx.scalar f64 b0) d

(* Power spectrum backends *)

type power = Cosmo.params -> Nx.float64_t -> Nx.float64_t -> Nx.float64_t

let linear p k z = Cosmo.linear_power ~p k z
let nonlinear p k z = Cosmo.nonlinear_power ~p k z

let baryonic_feedback ?(a_bary = 0.0) ?(log10_k_star = 1.0) ?(sigma = 0.55)
    base_power =
 fun p k z ->
  let pk = base_power p k z in
  if a_bary = 0.0 then pk
  else
    let inv_sigma2 = -1.0 /. (sigma *. sigma) in
    let log10_k = Nx.div_s (Nx.log k) (Float.log 10.0) in
    let delta = Nx.sub_s log10_k log10_k_star in
    let gauss = Nx.exp (Nx.mul_s (Nx.mul delta delta) inv_sigma2) in
    Nx.sub pk (Nx.mul_s (Nx.mul gauss pk) a_bary)

(* Tracers *)

type tracer_kind =
  | Weak_lensing of { ia_bias : bias option; sigma_e : float; m_bias : float }
  | Number_counts of { bias : bias }
  | Custom of {
      kernel :
        p:Cosmo.params -> z:Nx.float64_t -> chi:Nx.float64_t -> Nx.float64_t;
    }

type tracer = {
  nz : nz option;
  n_gal : float;
  noise : float;
  kind : tracer_kind;
  zmax : float;
}

let weak_lensing ?ia_bias ?(sigma_e = 0.26) ?(m_bias = 0.0) ?(n_gal = 1.0) nz =
  let noise = sigma_e *. sigma_e /. (n_gal *. steradian_to_arcmin2) in
  {
    nz = Some nz;
    n_gal;
    noise;
    kind = Weak_lensing { ia_bias; sigma_e; m_bias };
    zmax = nz.zmax;
  }

let number_counts ~bias ?(n_gal = 1.0) nz =
  let noise = 1.0 /. (n_gal *. steradian_to_arcmin2) in
  { nz = Some nz; n_gal; noise; kind = Number_counts { bias }; zmax = nz.zmax }

let tracer ?(noise = 0.0) ?(zmax = 3.0) kernel =
  { nz = None; n_gal = 0.0; noise; kind = Custom { kernel }; zmax }

(* Cls result type *)

type cls = {
  ell : Nx.float64_t;
  tracers : tracer array;
  spectra : Nx.float64_t;
}

(* Cl index ordering: upper triangle *)

let pair_index nt i j =
  let a, b = if i <= j then (i, j) else (j, i) in
  (a * ((2 * nt) - a - 1) / 2) + b

let cl_pairs nt =
  let pairs = ref [] in
  for i = 0 to nt - 1 do
    for j = i to nt - 1 do
      pairs := (i, j) :: !pairs
    done
  done;
  List.rev !pairs

(* Evaluate n(z) for one bin on the z grid. Returns tensor [n_z]. Uses Nx.stack
   so gradients flow through custom_nz eval functions. *)
let eval_nz_grid nz z_arr n_z =
  Nx.stack (List.init n_z (fun j -> nz.eval (Nx.scalar f64 z_arr.(j))))

(* Reverse cumulative trapezoidal sum of tensor [n] with spacing dz. result[j] =
   ∫_{x_j}^{x_{n-1}} f(x) dx via trapezoidal rule. *)
let rev_cumtrapz f_vec n dz =
  let left = Nx.slice [ R (0, n - 1) ] f_vec in
  let right = Nx.slice [ R (1, n) ] f_vec in
  let mid = Nx.mul_s (Nx.add left right) (0.5 *. dz) in
  let partial = Nx.flip (Nx.cumsum ~axis:0 (Nx.flip mid)) in
  Nx.concatenate [ partial; Nx.zeros f64 [| 1 |] ]

(* Angular power spectra *)

let angular_cl ?(p = Cosmo.planck18) ?(power = nonlinear) ~ell tracers =
  let tracers_arr = Array.of_list tracers in
  let nt = Array.length tracers_arr in
  let pairs = cl_pairs nt in
  let pairs_arr = Array.of_list pairs in
  let zmax =
    Array.fold_left (fun acc t -> Float.max acc t.zmax) 0.0 tracers_arr
  in
  let n_z = 100 in
  let dz = zmax /. Float.of_int (n_z - 1) in
  let z_arr = Array.init n_z (fun i -> Float.of_int i *. dz) in
  z_arr.(0) <- 1e-6;
  let z_vec = Nx.create f64 [| n_z |] z_arr in

  (* Simpson weights: tensor [n_z] *)
  let sw =
    Array.init n_z (fun i ->
        if i = 0 || i = n_z - 1 then 1.0 else if i mod 2 = 1 then 4.0 else 2.0)
  in
  let simpson_w = Nx.mul_s (Nx.create f64 [| n_z |] sw) (dz /. 3.0) in

  (* Precompute z-dependent quantities as tensors — differentiable through p.
     comoving_distance and growth_factor use GL quadrature internally and cannot
     accept vector z, so we loop over scalar z values. *)
  let h_t = Nx.div (Cosmo.h0 p) (Nx.scalar f64 h0_ref) in
  let chi_vec =
    Nx.stack
      (List.init n_z (fun j ->
           let z_t = Nx.scalar f64 z_arr.(j) in
           Nx.mul (Unit.Length.in_mpc (Cosmo.comoving_distance ~p z_t)) h_t))
  in
  let chi_safe = Nx.clamp ~min:1e-10 chi_vec in
  let h_vec = Cosmo.hubble ~p z_vec in
  let dchi_dz_vec = Nx.div (Nx.mul_s h_t c_km_s) h_vec in
  let growth_vec =
    Nx.stack
      (List.init n_z (fun j -> Cosmo.growth_factor ~p (Nx.scalar f64 z_arr.(j))))
  in
  let omega_m_t = Cosmo.omega_m p in

  (* n(z) values per tracer: tensors [n_z], differentiable through custom_nz *)
  let nz_arrs = Array.make nt (Nx.zeros f64 [| n_z |]) in
  Array.iteri
    (fun idx t ->
      match t.nz with
      | Some nz -> nz_arrs.(idx) <- eval_nz_grid nz z_arr n_z
      | None -> ())
    tracers_arr;

  (* Kernel base vectors per tracer: tensor [n_z], without ell_factor for WL *)
  let kernel_bases = Array.make nt (Nx.zeros f64 [| n_z |]) in
  let kernel_has_ell_factor = Array.make nt false in
  Array.iteri
    (fun idx t ->
      match t.kind with
      | Weak_lensing { ia_bias; sigma_e = _; m_bias } ->
          let nz_tensor = nz_arrs.(idx) in
          (* A(z_j) = ∫_{z_j}^{zmax} n(z') dz' *)
          let a_vec = rev_cumtrapz nz_tensor n_z dz in
          (* B(z_j) = ∫_{z_j}^{zmax} n(z')/χ(z') dz' — tensor, through chi *)
          let nz_over_chi = Nx.div nz_tensor chi_safe in
          let b_vec = rev_cumtrapz nz_over_chi n_z dz in
          (* g = A - chi * B *)
          let g_vec = Nx.sub a_vec (Nx.mul chi_vec b_vec) in
          (* WL kernel base: (3 H0² Ωm / 2c) × (1+z) × χ × g *)
          let prefactor =
            Nx.mul_s omega_m_t (3.0 *. h0_ref *. h0_ref /. (2.0 *. c_km_s))
          in
          let one_plus_z = Nx.add_s z_vec 1.0 in
          let k_base =
            Nx.mul prefactor (Nx.mul one_plus_z (Nx.mul chi_vec g_vec))
          in
          (* Add NLA intrinsic alignment if present *)
          let k_base =
            match ia_bias with
            | None -> k_base
            | Some ia_b ->
                let ia_tensor =
                  Nx.stack
                    (List.init n_z (fun j -> ia_b p (Nx.scalar f64 z_arr.(j))))
                in
                (* K_IA = -(C₁ ρ_crit Ωm / D(z)) × n(z) × b_IA(z) × H(z) *)
                let ia_kernel =
                  Nx.mul
                    (Nx.mul_s omega_m_t (-.c1_rho_crit))
                    (Nx.mul
                       (Nx.div nz_tensor growth_vec)
                       (Nx.mul ia_tensor h_vec))
                in
                Nx.add k_base ia_kernel
          in
          (* Shear multiplicative bias: W_obs = (1+m) W_true *)
          let k_base =
            if m_bias = 0.0 then k_base else Nx.mul_s k_base (1.0 +. m_bias)
          in
          kernel_bases.(idx) <- k_base;
          kernel_has_ell_factor.(idx) <- true
      | Number_counts { bias } ->
          let nz_tensor = nz_arrs.(idx) in
          let bias_tensor =
            Nx.stack (List.init n_z (fun j -> bias p (Nx.scalar f64 z_arr.(j))))
          in
          (* NC kernel: n(z) × b(z) × H(z) — no ell factor *)
          kernel_bases.(idx) <- Nx.mul nz_tensor (Nx.mul bias_tensor h_vec);
          kernel_has_ell_factor.(idx) <- false
      | Custom { kernel } ->
          (* Custom kernel: user provides the full W(z) *)
          kernel_bases.(idx) <-
            Nx.stack
              (List.init n_z (fun j ->
                   let z_t = Nx.scalar f64 z_arr.(j) in
                   let chi_t = Nx.get [ j ] chi_safe in
                   kernel ~p ~z:z_t ~chi:chi_t));
          kernel_has_ell_factor.(idx) <- false)
    tracers_arr;

  (* Common integration weight: dchi/dz / chi² / c² × simpson *)
  let integ_weight =
    Nx.mul simpson_w
      (Nx.div_s
         (Nx.div dchi_dz_vec (Nx.mul chi_safe chi_safe))
         (c_km_s *. c_km_s))
  in

  (* Power spectrum grid [n_z, n_ell]: loop over z (scalar), vectorized over k.
     Both linear_power and nonlinear_power accept vector k but scalar z. *)
  let pk_grid =
    Nx.stack
      (List.init n_z (fun z_idx ->
           let z_t = Nx.scalar f64 z_arr.(z_idx) in
           let chi_z = Nx.get [ z_idx ] chi_safe in
           let k_vec = Nx.div (Nx.add_s ell 0.5) chi_z in
           power p k_vec z_t))
  in

  (* ell_factor vector [n_ell]: sqrt((ℓ-1)ℓ(ℓ+1)(ℓ+2)) / (ℓ+0.5)² *)
  let ell_factor_vec =
    let l = ell in
    let num =
      Nx.mul
        (Nx.mul (Nx.sub_s l 1.0) l)
        (Nx.mul (Nx.add_s l 1.0) (Nx.add_s l 2.0))
    in
    let den = Nx.mul (Nx.add_s l 0.5) (Nx.add_s l 0.5) in
    Nx.div (Nx.sqrt (Nx.abs num)) den
  in

  (* Limber integration: functional, no in-place mutation. integ_weight is
     [n_z], pk_grid is [n_z, n_ell]. For each pair (i,j): C_ℓ = Σ_z K_i(z)
     K_j(z) P(k,z) w(z) kernel_bases are [n_z], broadcast with pk_grid [n_z,
     n_ell]. *)
  let w_pk = Nx.mul (Nx.reshape [| n_z; 1 |] integ_weight) pk_grid in
  let spectra =
    Nx.stack
      (List.map
         (fun (i, j) ->
           let ki = Nx.reshape [| n_z; 1 |] kernel_bases.(i) in
           let kj = Nx.reshape [| n_z; 1 |] kernel_bases.(j) in
           let integrand = Nx.mul (Nx.mul ki kj) w_pk in
           let cl_row = Nx.sum ~axes:[ 0 ] integrand in
           let ell_power =
             (if kernel_has_ell_factor.(i) then 1 else 0)
             + if kernel_has_ell_factor.(j) then 1 else 0
           in
           if ell_power = 0 then cl_row
           else if ell_power = 1 then Nx.mul ell_factor_vec cl_row
           else Nx.mul (Nx.mul ell_factor_vec ell_factor_vec) cl_row)
         (Array.to_list pairs_arr))
  in
  { ell; tracers = tracers_arr; spectra }

(* Cls submodule *)

module Cls = struct
  let get cls ~i ~j =
    let n = Array.length cls.tracers in
    if i < 0 || i >= n || j < 0 || j >= n then
      invalid_arg "Survey.Cls.get: index out of range";
    Nx.slice [ I (pair_index n i j) ] cls.spectra

  let ell cls = cls.ell
  let n_tracers cls = Array.length cls.tracers
  let to_tensor cls = cls.spectra

  let noise cls =
    let n_ell = (Nx.shape cls.ell).(0) in
    let nt = Array.length cls.tracers in
    let pairs = cl_pairs nt in
    let n_cls = List.length pairs in
    let result = Nx.zeros f64 [| n_cls; n_ell |] in
    let pair_idx = ref 0 in
    List.iter
      (fun (i, j) ->
        if i = j then begin
          let noise_val = cls.tracers.(i).noise in
          for l = 0 to n_ell - 1 do
            Nx.set_item [ !pair_idx; l ] noise_val result
          done
        end;
        incr pair_idx)
      pairs;
    result

  let gaussian_covariance ?(f_sky = 0.25) cls =
    let ell = cls.ell in
    let n_ell = (Nx.shape ell).(0) in
    let nt = Array.length cls.tracers in
    let pairs = cl_pairs nt in
    let n_cls = List.length pairs in
    let n = n_cls * n_ell in
    let cov = Nx.zeros f64 [| n; n |] in
    let cl_noise = noise cls in
    let cl_obs = Nx.add cls.spectra cl_noise in
    let pairs_arr = Array.of_list pairs in
    let find_pair a b = pair_index nt a b in
    (* Δℓ via finite differences *)
    let dell =
      Array.init n_ell (fun l ->
          if l = 0 then Nx.item [ 1 ] ell -. Nx.item [ 0 ] ell
          else if l = n_ell - 1 then Nx.item [ l ] ell -. Nx.item [ l - 1 ] ell
          else 0.5 *. (Nx.item [ l + 1 ] ell -. Nx.item [ l - 1 ] ell))
    in
    for p1 = 0 to n_cls - 1 do
      let i, j = pairs_arr.(p1) in
      for p2 = p1 to n_cls - 1 do
        let m, nn = pairs_arr.(p2) in
        let im = find_pair i m and jn = find_pair j nn in
        let in_ = find_pair i nn and jm = find_pair j m in
        for l = 0 to n_ell - 1 do
          let ell_l = Nx.item [ l ] ell in
          let norm = ((2.0 *. ell_l) +. 1.0) *. dell.(l) *. f_sky in
          let c_im = Nx.get [ im; l ] cl_obs in
          let c_jn = Nx.get [ jn; l ] cl_obs in
          let c_in = Nx.get [ in_; l ] cl_obs in
          let c_jm = Nx.get [ jm; l ] cl_obs in
          let val_ =
            Nx.div_s (Nx.add (Nx.mul c_im c_jn) (Nx.mul c_in c_jm)) norm
          in
          let row = (p1 * n_ell) + l in
          let col = (p2 * n_ell) + l in
          Nx.set [ row; col ] cov val_;
          if p1 <> p2 then Nx.set [ col; row ] cov val_
        done
      done
    done;
    cov
end
