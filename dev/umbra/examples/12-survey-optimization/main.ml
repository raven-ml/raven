(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Differentiable survey optimization via autodiff gradients through the Fisher
   information matrix.

   Traditional survey optimization uses grid search over discrete Fisher
   forecasts. Umbra's fully differentiable cosmology pipeline enables
   gradient-based continuous optimization: compute Fisher(survey_params) and
   minimize sigma(S8) with respect to survey parameters using exact autodiff
   gradients from Rune.

   Part 1: Area/depth tradeoff -- optimize f_sky with fixed n(z) shape. Part 2:
   Joint area + bin edge optimization -- optimize f_sky and tomographic bin
   edges simultaneously, with gradients flowing through the lensing kernel
   computation via differentiable n(z) windowing. *)

open Nx
open Umbra

let f64 = Nx.float64
let sigma_e = 0.26
let steradian_to_arcmin2 = 11818102.86004228
let c_km_s = 299792.458
let h0_ref = 100.0

(* Fiducial cosmology *)
let p_fid = Cosmo.planck18
let omega_m_fid = Nx.item [] (Cosmo.omega_m p_fid)
let sigma8_fid = Nx.item [] (Cosmo.sigma8 p_fid)

(* S8 = sigma8 * sqrt(omega_m / 0.3) -- derivatives at fiducial *)
let ds8_dom = sigma8_fid /. (2.0 *. Float.sqrt (0.3 *. omega_m_fid))
let ds8_ds8 = Float.sqrt (omega_m_fid /. 0.3)

(* ell weights: (2*ell+1) * dell / 2 *)
let ell_weights ell =
  let n_ell = (Nx.shape ell).(0) in
  let dell =
    Array.init n_ell (fun l ->
        if l = 0 then Nx.item [ 1 ] ell -. Nx.item [ 0 ] ell
        else if l = n_ell - 1 then Nx.item [ l ] ell -. Nx.item [ l - 1 ] ell
        else 0.5 *. (Nx.item [ l + 1 ] ell -. Nx.item [ l - 1 ] ell))
  in
  Nx.create f64 [| n_ell |]
    (Array.init n_ell (fun l ->
         ((2.0 *. Nx.item [ l ] ell) +. 1.0) *. dell.(l) /. 2.0))

(* Compute dCl/d(theta) via central finite differences *)
let finite_diff_cl ~ell ~tracers ~param_name ~set_param ~fid_val ~eps =
  let p_plus = set_param (scalar f64 (fid_val +. eps)) p_fid in
  let p_minus = set_param (scalar f64 (fid_val -. eps)) p_fid in
  let cl_p =
    Survey.Cls.to_tensor
      (Survey.angular_cl ~p:p_plus ~power:Survey.linear ~ell tracers)
  in
  let cl_m =
    Survey.Cls.to_tensor
      (Survey.angular_cl ~p:p_minus ~power:Survey.linear ~ell tracers)
  in
  let dcl = Nx.div_s (Nx.sub cl_p cl_m) (2.0 *. eps) in
  Printf.printf "  dCl/d(%-8s): max=%.3e\n" param_name (Nx.item [] (Nx.max dcl));
  dcl

(* 2x2 analytical Fisher inverse -> sigma(S8) -- all differentiable *)
let sigma_s8_from_fisher f11 f12 f22 =
  let det = Nx.sub (Nx.mul f11 f22) (Nx.mul f12 f12) in
  let a = scalar f64 ds8_dom and b = scalar f64 ds8_ds8 in
  let sigma_sq =
    Nx.div
      (Nx.add
         (Nx.sub
            (Nx.mul f22 (Nx.mul a a))
            (Nx.mul_s (Nx.mul f12 (Nx.mul a b)) 2.0))
         (Nx.mul f11 (Nx.mul b b)))
      det
  in
  Nx.sqrt sigma_sq

(* ===================================================================== *)
(* Part 1: Area/depth tradeoff (single bin) *)
(* ===================================================================== *)

let part1 () =
  Printf.printf "--- Part 1: Area/Depth Tradeoff (1 bin) ---\n\n";
  let budget = 10.0 in
  let ell = Nx.logspace f64 1.0 3.0 30 in
  let w_ell = ell_weights ell in

  let nz = Survey.smail ~a:2.0 ~b:1.5 ~z0:0.3 () in
  let wl = Survey.weak_lensing nz in

  Printf.printf "Precomputing signal derivatives...\n";
  let cl_fid =
    Survey.Cls.to_tensor
      (Survey.angular_cl ~p:p_fid ~power:Survey.linear ~ell [ wl ])
  in
  let cl_fid_flat = Nx.flatten cl_fid in
  let eps = 1e-4 in
  let dcl_dom =
    Nx.flatten
      (finite_diff_cl ~ell ~tracers:[ wl ] ~param_name:"omega_m"
         ~set_param:(fun v p -> Cosmo.set_t ~omega_m:v p)
         ~fid_val:omega_m_fid ~eps)
  in
  let dcl_ds8 =
    Nx.flatten
      (finite_diff_cl ~ell ~tracers:[ wl ] ~param_name:"sigma8"
         ~set_param:(fun v p -> Cosmo.set_t ~sigma8:v p)
         ~fid_val:sigma8_fid ~eps)
  in
  Printf.printf "\n";

  let objective log_f_sky =
    let f_sky = Nx.sigmoid log_f_sky in
    let n_gal = Nx.div (scalar f64 budget) f_sky in
    let noise =
      Nx.div
        (scalar f64 (sigma_e *. sigma_e))
        (Nx.mul_s n_gal steradian_to_arcmin2)
    in
    let cl_obs = Nx.add cl_fid_flat noise in
    let cl_obs_sq = Nx.mul cl_obs cl_obs in
    let weighted_dom =
      Nx.div (Nx.mul w_ell (Nx.mul dcl_dom dcl_dom)) cl_obs_sq
    in
    let weighted_ds8 =
      Nx.div (Nx.mul w_ell (Nx.mul dcl_ds8 dcl_ds8)) cl_obs_sq
    in
    let weighted_x = Nx.div (Nx.mul w_ell (Nx.mul dcl_dom dcl_ds8)) cl_obs_sq in
    let f11 = Nx.mul f_sky (Nx.sum weighted_dom) in
    let f12 = Nx.mul f_sky (Nx.sum weighted_x) in
    let f22 = Nx.mul f_sky (Nx.sum weighted_ds8) in
    sigma_s8_from_fisher f11 f12 f22
  in

  (* Gradient check *)
  let log_f_sky_init = scalar f64 0.0 in
  let v0, g0 = Rune.value_and_grad objective log_f_sky_init in
  let fd_eps = 1e-5 in
  let vp = item [] (objective (scalar f64 fd_eps)) in
  let vm = item [] (objective (scalar f64 (-.fd_eps))) in
  let fd = (vp -. vm) /. (2.0 *. fd_eps) in
  Printf.printf "Gradient check: AD=%.6e  FD=%.6e  rel=%.2e\n\n" (item [] g0) fd
    (Float.abs (item [] g0 -. fd) /. Float.abs fd);

  let f_sky_0 = 1.0 /. (1.0 +. Float.exp (-0.0)) in
  Printf.printf "Initial: f_sky=%.3f  n_gal=%.1f  sigma(S8)=%.6f\n" f_sky_0
    (budget /. f_sky_0) (item [] v0);

  let algo = Vega.adam (Vega.Schedule.constant 0.01) in
  let log_f_sky = ref log_f_sky_init in
  let state = ref (Vega.init algo !log_f_sky) in
  let best_sigma = ref (item [] v0) in
  let best_f_sky = ref f_sky_0 in
  Printf.printf "\n%5s  %8s  %8s  %10s\n" "step" "f_sky" "n_gal" "sigma(S8)";
  Printf.printf "%5s  %8s  %8s  %10s\n" "-----" "--------" "--------"
    "----------";
  let steps = 300 in
  for i = 0 to steps - 1 do
    let sigma_val, grad = Rune.value_and_grad objective !log_f_sky in
    let p, s = Vega.step !state ~grad ~param:!log_f_sky in
    log_f_sky := p;
    state := s;
    let f_sky_cur = 1.0 /. (1.0 +. Float.exp (-.item [] !log_f_sky)) in
    let sigma_cur = item [] sigma_val in
    if sigma_cur < !best_sigma then begin
      best_sigma := sigma_cur;
      best_f_sky := f_sky_cur
    end;
    if i mod 50 = 0 || i = steps - 1 then
      Printf.printf "%5d  %8.4f  %8.1f  %10.6f\n" i f_sky_cur
        (budget /. f_sky_cur) sigma_cur
  done;
  Printf.printf "\nOptimal: f_sky=%.4f  n_gal=%.1f gal/arcmin2\n" !best_f_sky
    (budget /. !best_f_sky);
  Printf.printf "Improvement: sigma(S8) reduced by %.1f%% vs initial\n\n"
    ((1.0 -. (!best_sigma /. item [] v0)) *. 100.0)

(* ===================================================================== *)
(* Part 2: Joint area + bin edge optimization (3 bins) *)
(* ===================================================================== *)

(* Precomputed cosmological grids -- expensive, done once per cosmology. *)
type cosmo_grid = {
  n_z : int;
  dz : float;
  z_arr : float array;
  z_vec : Nx.float64_t;
  chi_safe : Nx.float64_t;
  omega_m_t : Nx.float64_t;
  integ_weight : Nx.float64_t;
  w_pk : Nx.float64_t;
  ell_factor_sq : Nx.float64_t;
}

let precompute_grid ~p ~ell =
  let zmax = 3.0 in
  let n_z = 50 in
  let dz = zmax /. Float.of_int (n_z - 1) in
  let z_arr = Array.init n_z (fun i -> Float.of_int i *. dz) in
  z_arr.(0) <- 1e-6;
  let z_vec = Nx.create f64 [| n_z |] z_arr in
  let sw =
    Array.init n_z (fun i ->
        if i = 0 || i = n_z - 1 then 1.0 else if i mod 2 = 1 then 4.0 else 2.0)
  in
  let simpson_w = Nx.mul_s (Nx.create f64 [| n_z |] sw) (dz /. 3.0) in
  let h_t = Nx.item [] (Nx.div (Cosmo.h0 p) (Nx.scalar f64 h0_ref)) in
  let chi_vec =
    Nx.create f64 [| n_z |]
      (Array.init n_z (fun j ->
           let z_t = Nx.scalar f64 z_arr.(j) in
           let chi =
             Nx.item [] (Unit.Length.in_mpc (Cosmo.comoving_distance ~p z_t))
           in
           chi *. h_t))
  in
  let chi_safe = Nx.clamp ~min:1e-10 chi_vec in
  let h_vec_f =
    Array.init n_z (fun j ->
        Nx.item [] (Cosmo.hubble ~p (Nx.scalar f64 z_arr.(j))))
  in
  let dchi_dz_vec =
    Nx.create f64 [| n_z |]
      (Array.init n_z (fun j -> h_t *. c_km_s /. h_vec_f.(j)))
  in
  let omega_m_t = Nx.scalar f64 (Nx.item [] (Cosmo.omega_m p)) in
  let integ_weight =
    Nx.create f64 [| n_z |]
      (Array.init n_z (fun j ->
           let sw_j = Nx.item [ j ] simpson_w in
           let dchi_j = Nx.item [ j ] dchi_dz_vec in
           let chi_j = Nx.item [ j ] chi_safe in
           sw_j *. dchi_j /. (chi_j *. chi_j) /. (c_km_s *. c_km_s)))
  in
  let pk_grid =
    Nx.stack
      (List.init n_z (fun j ->
           let z_t = Nx.scalar f64 z_arr.(j) in
           let chi_j = Nx.item [ j ] chi_safe in
           let k_vec = Nx.div_s (Nx.add_s ell 0.5) chi_j in
           Cosmo.linear_power ~p k_vec z_t))
  in
  let w_pk =
    Nx.mul
      (Nx.reshape [| n_z; 1 |]
         (Nx.create f64 [| n_z |]
            (Array.init n_z (fun j -> Nx.item [ j ] integ_weight))))
      pk_grid
  in
  let l = ell in
  let num =
    Nx.mul
      (Nx.mul (Nx.sub_s l 1.0) l)
      (Nx.mul (Nx.add_s l 1.0) (Nx.add_s l 2.0))
  in
  let den = Nx.mul (Nx.add_s l 0.5) (Nx.add_s l 0.5) in
  let ell_factor = Nx.div (Nx.sqrt (Nx.abs num)) den in
  let ell_factor_sq = Nx.mul ell_factor ell_factor in
  {
    n_z;
    dz;
    z_arr;
    z_vec;
    chi_safe;
    omega_m_t;
    integ_weight;
    w_pk;
    ell_factor_sq;
  }

(* Reverse cumulative trapezoidal sum *)
let rev_cumtrapz f_vec n dz =
  let left = Nx.slice [ R (0, n - 1) ] f_vec in
  let right = Nx.slice [ R (1, n) ] f_vec in
  let mid = Nx.mul_s (Nx.add left right) (0.5 *. dz) in
  let partial = Nx.flip (Nx.cumsum ~axis:0 (Nx.flip mid)) in
  Nx.concatenate [ partial; Nx.zeros f64 [| 1 |] ]

(* Fast WL-only angular Cl from precomputed cosmo grid + pre-evaluated n(z)
   tensors. nz_tensors are [n_z] tensors, one per bin, evaluated on the z grid.
   Differentiable through the n(z) values. *)
let fast_wl_cl grid nz_tensors =
  let n_z = grid.n_z and dz = grid.dz in
  let n_bins = Array.length nz_tensors in
  (* Build WL kernels *)
  let prefactor =
    Nx.mul_s grid.omega_m_t (3.0 *. h0_ref *. h0_ref /. (2.0 *. c_km_s))
  in
  let one_plus_z = Nx.add_s grid.z_vec 1.0 in
  let kernels =
    Array.init n_bins (fun b ->
        let nz_t = nz_tensors.(b) in
        let a_vec = rev_cumtrapz nz_t n_z dz in
        let nz_over_chi = Nx.div nz_t grid.chi_safe in
        let b_vec = rev_cumtrapz nz_over_chi n_z dz in
        let g_vec = Nx.sub a_vec (Nx.mul grid.chi_safe b_vec) in
        Nx.mul prefactor (Nx.mul one_plus_z (Nx.mul grid.chi_safe g_vec)))
  in
  (* Limber integration for all pairs *)
  let pairs = ref [] in
  for i = 0 to n_bins - 1 do
    for j = i to n_bins - 1 do
      pairs := (i, j) :: !pairs
    done
  done;
  let pairs = List.rev !pairs in
  Nx.stack
    (List.map
       (fun (i, j) ->
         let ki = Nx.reshape [| n_z; 1 |] kernels.(i) in
         let kj = Nx.reshape [| n_z; 1 |] kernels.(j) in
         let integrand = Nx.mul (Nx.mul ki kj) grid.w_pk in
         Nx.mul grid.ell_factor_sq (Nx.sum ~axes:[ 0 ] integrand))
       pairs)

(* Parent n(z): Smail distribution, evaluated as float *)
let parent_nz =
  let a = 2.0 and b = 1.5 and z0 = 0.3 in
  let raw z_f = (z_f ** a) *. Float.exp (-.((z_f /. z0) ** b)) in
  let norm =
    let n = 256 in
    let h = 3.0 /. Float.of_int n in
    let s = ref (raw 1e-6 +. raw 3.0) in
    for i = 1 to n - 1 do
      let x = Float.of_int i *. h in
      let w = if i mod 2 = 1 then 4.0 else 2.0 in
      s := !s +. (w *. raw x)
    done;
    !s *. h /. 3.0
  in
  fun z_f -> raw z_f /. norm

(* Build a differentiable bin n(z) with smooth sigmoid edges *)
let make_bin_eval z_lo z_hi delta z =
  let parent_val = parent_nz (Nx.item [] z) in
  if parent_val < 1e-30 then scalar f64 0.0
  else
    let lo_gate = Nx.sigmoid (Nx.div_s (Nx.sub z z_lo) delta) in
    let hi_gate = Nx.sigmoid (Nx.div_s (Nx.sub z_hi z) delta) in
    Nx.mul_s (Nx.mul lo_gate hi_gate) parent_val

let part2 () =
  Printf.printf "--- Part 2: Joint Area + Bin Edges (3 bins) ---\n\n";
  let budget = 10.0 in
  let ell = Nx.logspace f64 1.0 3.0 20 in
  let w_ell = ell_weights ell in
  let eps = 1e-4 in
  let delta = 0.03 in

  Printf.printf "Precomputing cosmo grids (fiducial + 4 perturbations)...\n";
  let grid_fid = precompute_grid ~p:p_fid ~ell in
  let grid_p_om =
    precompute_grid
      ~p:(Cosmo.set_t ~omega_m:(scalar f64 (omega_m_fid +. eps)) p_fid)
      ~ell
  in
  let grid_m_om =
    precompute_grid
      ~p:(Cosmo.set_t ~omega_m:(scalar f64 (omega_m_fid -. eps)) p_fid)
      ~ell
  in
  let grid_p_s8 =
    precompute_grid
      ~p:(Cosmo.set_t ~sigma8:(scalar f64 (sigma8_fid +. eps)) p_fid)
      ~ell
  in
  let grid_m_s8 =
    precompute_grid
      ~p:(Cosmo.set_t ~sigma8:(scalar f64 (sigma8_fid -. eps)) p_fid)
      ~ell
  in
  Printf.printf "Done.\n\n";

  let n_z = grid_fid.n_z in
  let z_arr = grid_fid.z_arr in
  let dz = grid_fid.dz in

  let objective params =
    let log_f_sky = Nx.get [ 0 ] params in
    let z1 = Nx.get [ 1 ] params in
    let z2 = Nx.get [ 2 ] params in
    let f_sky = Nx.sigmoid log_f_sky in
    let n_gal = Nx.div (scalar f64 budget) f_sky in

    (* Differentiable n(z) bin functions *)
    let nz_funs =
      [|
        make_bin_eval (scalar f64 0.0) z1 delta;
        make_bin_eval z1 z2 delta;
        make_bin_eval z2 (scalar f64 3.0) delta;
      |]
    in

    (* Evaluate n(z) on z grid -- differentiable through bin edges *)
    let nz_tensors =
      Array.init 3 (fun b ->
          Nx.stack
            (List.init n_z (fun j -> nz_funs.(b) (Nx.scalar f64 z_arr.(j)))))
    in

    (* Galaxy fraction per bin: integral of window_i(z) n(z) dz. Parent n(z) is
       normalized so this gives the fraction of total galaxies in each bin.
       Differentiable through bin edges -- narrow bins get fewer galaxies. *)
    let gal_fracs =
      Array.init 3 (fun b ->
          let nz_t = nz_tensors.(b) in
          let left = Nx.slice [ R (0, n_z - 2) ] nz_t in
          let right = Nx.slice [ R (1, n_z - 1) ] nz_t in
          Nx.mul_s (Nx.sum (Nx.add left right)) (0.5 *. dz))
    in

    (* Per-bin noise: sigma_e^2 / (n_gal_bin * ster) where n_gal_bin = n_gal *
       f_i. Bins with fewer galaxies have higher shot noise. *)
    let noise_per_bin =
      Array.init 3 (fun b ->
          Nx.div
            (scalar f64 (sigma_e *. sigma_e))
            (Nx.mul_s (Nx.mul n_gal gal_fracs.(b)) steradian_to_arcmin2))
    in

    (* Fast Cl from precomputed grids -- only n(z) -> kernel is traced *)
    let cl_fid = fast_wl_cl grid_fid nz_tensors in
    let cl_p_om = fast_wl_cl grid_p_om nz_tensors in
    let cl_m_om = fast_wl_cl grid_m_om nz_tensors in
    let cl_p_s8 = fast_wl_cl grid_p_s8 nz_tensors in
    let cl_m_s8 = fast_wl_cl grid_m_s8 nz_tensors in
    let dcl_dom = Nx.div_s (Nx.sub cl_p_om cl_m_om) (2.0 *. eps) in
    let dcl_ds8 = Nx.div_s (Nx.sub cl_p_s8 cl_m_s8) (2.0 *. eps) in

    (* Full Fisher via Tr[C^-1 dC/dtheta_i C^-1 dC/dtheta_j] with analytical 3x3
       inverse. Vectorized over ell: each matrix element is a [n_ell] tensor. *)
    let n_bins = 3 in

    (* Pair index: (i,j) -> spectrum row in cl arrays. Ordering: (0,0)=0,
       (0,1)=1, (0,2)=2, (1,1)=3, (1,2)=4, (2,2)=5 *)
    let pidx i j =
      let a, b = if i <= j then (i, j) else (j, i) in
      (a * ((2 * n_bins) - a - 1) / 2) + b
    in

    (* Build 3x3 C(ell) = Cl + N, stored as flat [9] of [n_ell] tensors *)
    let c =
      Array.init 9 (fun idx ->
          let i = idx / 3 and j = idx mod 3 in
          let cl_ij = Nx.slice [ I (pidx i j) ] cl_fid in
          if i = j then Nx.add cl_ij noise_per_bin.(i) else cl_ij)
    in

    (* 3x3 inverse via cofactors / determinant *)
    let det =
      Nx.add
        (Nx.sub
           (Nx.mul c.(0) (Nx.sub (Nx.mul c.(4) c.(8)) (Nx.mul c.(5) c.(7))))
           (Nx.mul c.(1) (Nx.sub (Nx.mul c.(3) c.(8)) (Nx.mul c.(5) c.(6)))))
        (Nx.mul c.(2) (Nx.sub (Nx.mul c.(3) c.(7)) (Nx.mul c.(4) c.(6))))
    in
    let ci = Array.make 9 (scalar f64 0.0) in
    ci.(0) <- Nx.div (Nx.sub (Nx.mul c.(4) c.(8)) (Nx.mul c.(5) c.(7))) det;
    ci.(1) <- Nx.div (Nx.sub (Nx.mul c.(2) c.(7)) (Nx.mul c.(1) c.(8))) det;
    ci.(2) <- Nx.div (Nx.sub (Nx.mul c.(1) c.(5)) (Nx.mul c.(2) c.(4))) det;
    ci.(3) <- Nx.div (Nx.sub (Nx.mul c.(5) c.(6)) (Nx.mul c.(3) c.(8))) det;
    ci.(4) <- Nx.div (Nx.sub (Nx.mul c.(0) c.(8)) (Nx.mul c.(2) c.(6))) det;
    ci.(5) <- Nx.div (Nx.sub (Nx.mul c.(2) c.(3)) (Nx.mul c.(0) c.(5))) det;
    ci.(6) <- Nx.div (Nx.sub (Nx.mul c.(3) c.(7)) (Nx.mul c.(4) c.(6))) det;
    ci.(7) <- Nx.div (Nx.sub (Nx.mul c.(1) c.(6)) (Nx.mul c.(0) c.(7))) det;
    ci.(8) <- Nx.div (Nx.sub (Nx.mul c.(0) c.(4)) (Nx.mul c.(1) c.(3))) det;

    (* Build dC/dtheta matrices: symmetric, no noise term *)
    let dc_om =
      Array.init 9 (fun idx ->
          Nx.slice [ I (pidx (idx / 3) (idx mod 3)) ] dcl_dom)
    in
    let dc_s8 =
      Array.init 9 (fun idx ->
          Nx.slice [ I (pidx (idx / 3) (idx mod 3)) ] dcl_ds8)
    in

    (* 3x3 matmul: (AB)_ij = sum_k A_ik B_kj, vectorized over ell *)
    let mm a b =
      Array.init 9 (fun idx ->
          let i = idx / 3 and j = idx mod 3 in
          Nx.add
            (Nx.add (Nx.mul a.(i * 3) b.(j)) (Nx.mul a.((i * 3) + 1) b.(3 + j)))
            (Nx.mul a.((i * 3) + 2) b.(6 + j)))
    in

    (* Tr[AB] = sum_ij A_ij B_ji, returns [n_ell] tensor *)
    let tr a b =
      let t = ref (Nx.mul a.(0) b.(0)) in
      for i = 0 to 2 do
        for j = 0 to 2 do
          if i > 0 || j > 0 then
            t := Nx.add !t (Nx.mul a.((i * 3) + j) b.((j * 3) + i))
        done
      done;
      !t
    in

    (* D1 = C^-1 dC/d(Omega_m), D2 = C^-1 dC/d(sigma8) *)
    let d1 = mm ci dc_om in
    let d2 = mm ci dc_s8 in

    (* F_ij = f_sky * sum_ell w_ell * Tr[D_i D_j] *)
    let f11 = Nx.mul f_sky (Nx.sum (Nx.mul w_ell (tr d1 d1))) in
    let f12 = Nx.mul f_sky (Nx.sum (Nx.mul w_ell (tr d1 d2))) in
    let f22 = Nx.mul f_sky (Nx.sum (Nx.mul w_ell (tr d2 d2))) in
    sigma_s8_from_fisher f11 f12 f22
  in

  let params = Nx.create f64 [| 3 |] [| -1.1; 0.5; 1.0 |] in
  Printf.printf "Computing initial sigma(S8)...\n";
  let v0 = item [] (objective params) in
  let f_sky_0 = 1.0 /. (1.0 +. Float.exp 1.1) in
  Printf.printf
    "Initial: f_sky=%.3f  bins=[0, 0.50, 1.00, 3.0]  sigma(S8)=%.6f\n\n" f_sky_0
    v0;

  let algo = Vega.adam (Vega.Schedule.constant 0.03) in
  let params = ref params in
  let state = ref (Vega.init algo !params) in
  let best_sigma = ref v0 in
  let best_params = ref !params in
  Printf.printf "%5s  %8s  %8s  %8s  %10s\n" "step" "f_sky" "z1" "z2"
    "sigma(S8)";
  Printf.printf "%5s  %8s  %8s  %8s  %10s\n" "-----" "--------" "--------"
    "--------" "----------";
  let steps = 500 in
  for i = 0 to steps - 1 do
    let sigma_val, grad = Rune.value_and_grad objective !params in
    let p, s = Vega.step !state ~grad ~param:!params in
    let z1 = Float.max 0.1 (Float.min 2.8 (item [ 1 ] p)) in
    let z2 = Float.max (z1 +. 0.1) (Float.min 2.9 (item [ 2 ] p)) in
    params := Nx.create f64 [| 3 |] [| item [ 0 ] p; z1; z2 |];
    state := s;
    let sigma_cur = item [] sigma_val in
    if sigma_cur < !best_sigma then begin
      best_sigma := sigma_cur;
      best_params := !params
    end;
    if i mod 50 = 0 || i = steps - 1 then begin
      let f_sky = 1.0 /. (1.0 +. Float.exp (-.item [ 0 ] !params)) in
      Printf.printf "%5d  %8.4f  %8.3f  %8.3f  %10.6f\n" i f_sky
        (item [ 1 ] !params) (item [ 2 ] !params) sigma_cur
    end
  done;
  let f_sky_opt = 1.0 /. (1.0 +. Float.exp (-.item [ 0 ] !best_params)) in
  Printf.printf
    "\nGrad optimal: f_sky=%.4f  bins=[0, %.2f, %.2f, 3.0]  sigma(S8)=%.6f\n"
    f_sky_opt (item [ 1 ] !best_params) (item [ 2 ] !best_params) !best_sigma;

  (* Grid search validation *)
  let grid_best_sigma = ref infinity in
  let grid_best_fs = ref 0.0 in
  let grid_best_z1 = ref 0.0 in
  let grid_best_z2 = ref 0.0 in
  let n_fs = 12 and n_z1 = 15 and n_z2 = 15 in
  let n_grid_evals = ref 0 in
  Printf.printf "\nGrid search (%d*%d*%d)...\n%!" n_fs n_z1 n_z2;
  for fi = 0 to n_fs - 1 do
    let fs = 0.1 +. (Float.of_int fi *. 0.88 /. Float.of_int (n_fs - 1)) in
    let log_fs = Float.log (fs /. (1.0 -. fs)) in
    for z1i = 0 to n_z1 - 1 do
      let z1_v = 0.2 +. (Float.of_int z1i *. 2.4 /. Float.of_int (n_z1 - 1)) in
      for z2i = 0 to n_z2 - 1 do
        let z2_v =
          z1_v +. 0.15
          +. (Float.of_int z2i *. (2.7 -. z1_v) /. Float.of_int (n_z2 - 1))
        in
        if z2_v > z1_v +. 0.1 && z2_v < 2.9 then begin
          incr n_grid_evals;
          let p = Nx.create f64 [| 3 |] [| log_fs; z1_v; z2_v |] in
          let s = item [] (objective p) in
          if s < !grid_best_sigma then begin
            grid_best_sigma := s;
            grid_best_fs := fs;
            grid_best_z1 := z1_v;
            grid_best_z2 := z2_v
          end
        end
      done
    done
  done;
  Printf.printf
    "Grid optimal: f_sky=%.4f  bins=[0, %.2f, %.2f, 3.0]  sigma(S8)=%.6f  (%d \
     evals)\n"
    !grid_best_fs !grid_best_z1 !grid_best_z2 !grid_best_sigma !n_grid_evals;
  Printf.printf "\nComparison:\n";
  Printf.printf "  Gradient:  sigma(S8)=%.6f  (%d evals)\n" !best_sigma steps;
  Printf.printf "  Grid:      sigma(S8)=%.6f  (%d evals)\n" !grid_best_sigma
    !n_grid_evals;
  let rel = (1.0 -. (!best_sigma /. !grid_best_sigma)) *. 100.0 in
  if rel >= 0.0 then
    Printf.printf "  Gradient %.1f%% better with %.0f* fewer evaluations\n" rel
      (Float.of_int !n_grid_evals /. Float.of_int steps)
  else
    Printf.printf
      "  Gradient within %.1f%% of grid with %.0f* fewer evaluations\n"
      (Float.abs rel)
      (Float.of_int !n_grid_evals /. Float.of_int steps)

let () =
  Printf.printf "=== Differentiable Survey Optimization ===\n";
  Printf.printf "Stage IV Weak Lensing Survey\n\n";
  part1 ();
  part2 ()
