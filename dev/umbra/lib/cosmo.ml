(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Cosmological distance calculations for ΛCDM, wCDM, and w0waCDM universes.

   w0waCDM subsumes all models: - flat ΛCDM: omega_k = 0, w0 = -1, wa = 0 -
   non-flat ΛCDM: w0 = -1, wa = 0 - wCDM: wa = 0 - w0waCDM: general case

   All computations use Nx tensor ops, making them natively differentiable
   through Rune's autodiff. GL quadrature is vectorized as tensor operations. *)

let f64 = Nx.float64
let c_km_s = Nx.scalar f64 299792.458
let _mpc_m = 3.085_677_581_491_367_3e22

type params = {
  h0 : Nx.float64_t;
  omega_m : Nx.float64_t;
  omega_l : Nx.float64_t;
  omega_k : Nx.float64_t;
  w0 : Nx.float64_t;
  wa : Nx.float64_t;
  omega_b : Nx.float64_t option;
  n_s : Nx.float64_t option;
  sigma8 : Nx.float64_t option;
}

let err_missing name =
  invalid_arg
    ("Cosmo: " ^ name ^ " not set (use Cosmo.set or a preset like planck18)")

(* --- Constructors --- *)

let flat_lcdm ~h0 ~omega_m =
  if h0 <= 0.0 then invalid_arg "Cosmo.flat_lcdm: h0 must be positive";
  if omega_m < 0.0 then
    invalid_arg "Cosmo.flat_lcdm: omega_m must be non-negative";
  {
    h0 = Nx.scalar f64 h0;
    omega_m = Nx.scalar f64 omega_m;
    omega_l = Nx.scalar f64 (1.0 -. omega_m);
    omega_k = Nx.scalar f64 0.0;
    w0 = Nx.scalar f64 (-1.0);
    wa = Nx.scalar f64 0.0;
    omega_b = None;
    n_s = None;
    sigma8 = None;
  }

let lcdm ~h0 ~omega_m ~omega_l =
  if h0 <= 0.0 then invalid_arg "Cosmo.lcdm: h0 must be positive";
  {
    h0 = Nx.scalar f64 h0;
    omega_m = Nx.scalar f64 omega_m;
    omega_l = Nx.scalar f64 omega_l;
    omega_k = Nx.scalar f64 (1.0 -. omega_m -. omega_l);
    w0 = Nx.scalar f64 (-1.0);
    wa = Nx.scalar f64 0.0;
    omega_b = None;
    n_s = None;
    sigma8 = None;
  }

let wcdm ~h0 ~omega_m ?omega_l ~w0 () =
  if h0 <= 0.0 then invalid_arg "Cosmo.wcdm: h0 must be positive";
  let omega_l = match omega_l with Some v -> v | None -> 1.0 -. omega_m in
  {
    h0 = Nx.scalar f64 h0;
    omega_m = Nx.scalar f64 omega_m;
    omega_l = Nx.scalar f64 omega_l;
    omega_k = Nx.scalar f64 (1.0 -. omega_m -. omega_l);
    w0 = Nx.scalar f64 w0;
    wa = Nx.scalar f64 0.0;
    omega_b = None;
    n_s = None;
    sigma8 = None;
  }

let w0wacdm ~h0 ~omega_m ?omega_l ~w0 ~wa () =
  if h0 <= 0.0 then invalid_arg "Cosmo.w0wacdm: h0 must be positive";
  let omega_l = match omega_l with Some v -> v | None -> 1.0 -. omega_m in
  {
    h0 = Nx.scalar f64 h0;
    omega_m = Nx.scalar f64 omega_m;
    omega_l = Nx.scalar f64 omega_l;
    omega_k = Nx.scalar f64 (1.0 -. omega_m -. omega_l);
    w0 = Nx.scalar f64 w0;
    wa = Nx.scalar f64 wa;
    omega_b = None;
    n_s = None;
    sigma8 = None;
  }

(* Tensor constructors for differentiable construction *)

let create_flat_lcdm ~h0 ~omega_m =
  {
    h0;
    omega_m;
    omega_l = Nx.sub (Nx.scalar f64 1.0) omega_m;
    omega_k = Nx.scalar f64 0.0;
    w0 = Nx.scalar f64 (-1.0);
    wa = Nx.scalar f64 0.0;
    omega_b = None;
    n_s = None;
    sigma8 = None;
  }

let create_lcdm ~h0 ~omega_m ~omega_l =
  {
    h0;
    omega_m;
    omega_l;
    omega_k = Nx.sub (Nx.scalar f64 1.0) (Nx.add omega_m omega_l);
    w0 = Nx.scalar f64 (-1.0);
    wa = Nx.scalar f64 0.0;
    omega_b = None;
    n_s = None;
    sigma8 = None;
  }

let create_wcdm ~h0 ~omega_m ?omega_l ~w0 () =
  let omega_l =
    match omega_l with
    | Some v -> v
    | None -> Nx.sub (Nx.scalar f64 1.0) omega_m
  in
  {
    h0;
    omega_m;
    omega_l;
    omega_k = Nx.sub (Nx.scalar f64 1.0) (Nx.add omega_m omega_l);
    w0;
    wa = Nx.scalar f64 0.0;
    omega_b = None;
    n_s = None;
    sigma8 = None;
  }

let create_w0wacdm ~h0 ~omega_m ?omega_l ~w0 ~wa () =
  let omega_l =
    match omega_l with
    | Some v -> v
    | None -> Nx.sub (Nx.scalar f64 1.0) omega_m
  in
  {
    h0;
    omega_m;
    omega_l;
    omega_k = Nx.sub (Nx.scalar f64 1.0) (Nx.add omega_m omega_l);
    w0;
    wa;
    omega_b = None;
    n_s = None;
    sigma8 = None;
  }

(* Accessors *)

let h0 p = p.h0
let omega_m p = p.omega_m
let omega_l p = p.omega_l
let omega_k p = p.omega_k
let w0 p = p.w0
let wa p = p.wa

let omega_b p =
  match p.omega_b with Some v -> v | None -> err_missing "omega_b"

let n_s p = match p.n_s with Some v -> v | None -> err_missing "n_s"
let sigma8 p = match p.sigma8 with Some v -> v | None -> err_missing "sigma8"

let set ?omega_b ?n_s ?sigma8 p =
  let omega_b =
    match omega_b with Some v -> Some (Nx.scalar f64 v) | None -> p.omega_b
  in
  let n_s = match n_s with Some v -> Some (Nx.scalar f64 v) | None -> p.n_s in
  let sigma8 =
    match sigma8 with Some v -> Some (Nx.scalar f64 v) | None -> p.sigma8
  in
  { p with omega_b; n_s; sigma8 }

let set_t ?h0 ?omega_m ?omega_l ?omega_b ?n_s ?sigma8 p =
  let h0 = match h0 with Some v -> v | None -> p.h0 in
  let omega_m = match omega_m with Some v -> v | None -> p.omega_m in
  let omega_l = match omega_l with Some v -> v | None -> p.omega_l in
  let omega_k = Nx.sub (Nx.scalar f64 1.0) (Nx.add omega_m omega_l) in
  let omega_b = match omega_b with Some v -> Some v | None -> p.omega_b in
  let n_s = match n_s with Some v -> Some v | None -> p.n_s in
  let sigma8 = match sigma8 with Some v -> Some v | None -> p.sigma8 in
  { p with h0; omega_m; omega_l; omega_k; omega_b; n_s; sigma8 }

(* Presets *)

let default = flat_lcdm ~h0:70.0 ~omega_m:0.3

let planck18 =
  flat_lcdm ~h0:67.66 ~omega_m:0.3111
  |> set ~omega_b:0.0490 ~n_s:0.9665 ~sigma8:0.8102

let planck15 =
  flat_lcdm ~h0:67.74 ~omega_m:0.3075
  |> set ~omega_b:0.0486 ~n_s:0.9667 ~sigma8:0.8159

let wmap9 =
  flat_lcdm ~h0:69.32 ~omega_m:0.2865
  |> set ~omega_b:0.0463 ~n_s:0.9608 ~sigma8:0.820

(* --- E(z) computation ---

   E(z) = H(z)/H0 = sqrt(Ω_m(1+z)³ + Ω_k(1+z)² + Ω_de(z))

   where Ω_de(z) = Ω_Λ * (1+z)^(3(1+w0+wa)) * exp(-3*wa*z/(1+z))

   For ΛCDM (w0=-1, wa=0): Ω_de(z) = Ω_Λ (constant) For wCDM (wa=0): Ω_de(z) =
   Ω_Λ * (1+z)^(3(1+w0)) *)

let e_of p z =
  let one_plus_z = Nx.add_s z 1.0 in
  let cubed = Nx.mul one_plus_z (Nx.mul one_plus_z one_plus_z) in
  let matter = Nx.mul p.omega_m cubed in
  let curvature = Nx.mul p.omega_k (Nx.mul one_plus_z one_plus_z) in
  (* Dark energy: Ω_Λ * (1+z)^(3(1+w0+wa)) * exp(-3*wa*z/(1+z)) *)
  let w_eff = Nx.add_s (Nx.add p.w0 p.wa) 1.0 in
  let de_power = Nx.pow one_plus_z (Nx.mul_s w_eff 3.0) in
  let wa_arg = Nx.mul (Nx.mul_s p.wa (-3.0)) (Nx.div z one_plus_z) in
  let de = Nx.mul p.omega_l (Nx.mul de_power (Nx.exp wa_arg)) in
  Nx.sqrt (Nx.add matter (Nx.add curvature de))

(* 16-point Gauss-Legendre nodes and weights on [-1, 1] as Nx tensors *)
let gl_nodes =
  Nx.create f64 [| 16 |]
    [|
      -0.9894009349916499;
      -0.9445750230732326;
      -0.8656312023878318;
      -0.7554044083550030;
      -0.6178762444026438;
      -0.4580167776572274;
      -0.2816035507792589;
      -0.0950125098376374;
      0.0950125098376374;
      0.2816035507792589;
      0.4580167776572274;
      0.6178762444026438;
      0.7554044083550030;
      0.8656312023878318;
      0.9445750230732326;
      0.9894009349916499;
    |]

let gl_weights =
  Nx.create f64 [| 16 |]
    [|
      0.0271524594117541;
      0.0622535239386479;
      0.0951585116824928;
      0.1246289712555339;
      0.1495959888165767;
      0.1691565193950025;
      0.1826034150449236;
      0.1894506104550685;
      0.1894506104550685;
      0.1826034150449236;
      0.1691565193950025;
      0.1495959888165767;
      0.1246289712555339;
      0.0951585116824928;
      0.0622535239386479;
      0.0271524594117541;
    |]

(* GL quadrature in scale-factor space.

   All cosmological integrals ∫₀ᶻ g(z') dz' are evaluated via the substitution a
   = 1/(1+z), which maps [0, z] → [1/(1+z), 1]. This bounded range is
   well-resolved by 16-point GL even at z = 1089 (CMB). Direct quadrature over
   [0, z] in redshift space under-resolves the integrand at large z. *)
let gl_quad_a p z f =
  let a_lo = Nx.recip (Nx.add_s z 1.0) in
  let one = Nx.scalar f64 1.0 in
  let half = Nx.div_s (Nx.sub one a_lo) 2.0 in
  let mid = Nx.div_s (Nx.add one a_lo) 2.0 in
  let a = Nx.add (Nx.mul half gl_nodes) mid in
  let e_z = e_of p (Nx.sub_s (Nx.recip a) 1.0) in
  Nx.mul half (Nx.sum (Nx.mul (f a e_z) gl_weights))

(* ∫₀ᶻ dz'/E(z') = ∫_{a_lo}^1 da/(a² E(a)) *)
let integrate_inv_ez p z =
  gl_quad_a p z (fun a e -> Nx.recip (Nx.mul (Nx.mul a a) e))

(* ∫₀ᶻ dz'/((1+z') E(z')) = ∫_{a_lo}^1 da/(a E(a)) *)
let integrate_inv_z1_ez p z = gl_quad_a p z (fun a e -> Nx.recip (Nx.mul a e))

(* --- Derived quantities --- *)

let hubble ?(p = default) z = Nx.mul p.h0 (e_of p z)

let critical_density ?(p = default) z =
  let h_z = hubble ~p z in
  let h_si = Nx.div_s (Nx.mul_s h_z 1e3) _mpc_m in
  Nx.div_s (Nx.mul_s (Nx.mul h_si h_si) 3.0) (8.0 *. Float.pi *. 6.674_30e-11)

(* --- Distances ---

   Line-of-sight comoving distance: χ = d_H ∫₀ᶻ dz'/E(z')

   Transverse comoving distance (curvature-corrected): - Ω_k > 0 (open): d_M =
   d_H/√Ω_k · sinh(√Ω_k · χ/d_H) - Ω_k = 0 (flat): d_M = χ - Ω_k < 0 (closed):
   d_M = d_H/√|Ω_k| · sin(√|Ω_k| · χ/d_H) *)

let comoving_distance_mpc p z =
  let d_h = Nx.div c_km_s p.h0 in
  Nx.mul d_h (integrate_inv_ez p z)

let transverse_comoving_mpc p z =
  let d_h = Nx.div c_km_s p.h0 in
  let chi = Nx.mul d_h (integrate_inv_ez p z) in
  let ok_f = Nx.item [] p.omega_k in
  if Float.abs ok_f < 1e-10 then chi (* flat *)
  else
    let sqrt_ok = Nx.sqrt (Nx.abs p.omega_k) in
    let arg = Nx.div (Nx.mul sqrt_ok chi) d_h in
    if ok_f > 0.0 then Nx.div (Nx.mul d_h (Nx.sinh arg)) sqrt_ok
    else Nx.div (Nx.mul d_h (Nx.sin arg)) sqrt_ok

let comoving_distance ?(p = default) z =
  Unit.Length.of_tensor (Nx.mul_s (comoving_distance_mpc p z) _mpc_m)

let luminosity_distance ?(p = default) z =
  let dm_mpc = transverse_comoving_mpc p z in
  Unit.Length.of_tensor (Nx.mul_s (Nx.mul (Nx.add_s z 1.0) dm_mpc) _mpc_m)

let angular_diameter_distance ?(p = default) z =
  let dm_mpc = transverse_comoving_mpc p z in
  Unit.Length.of_tensor (Nx.mul_s (Nx.div dm_mpc (Nx.add_s z 1.0)) _mpc_m)

let distance_modulus ?(p = default) z =
  let dl_mpc = Nx.mul (Nx.add_s z 1.0) (transverse_comoving_mpc p z) in
  (* mu = 5 * log10(dL_Mpc) + 25 = 5/ln10 * ln(dL_Mpc) + 25 *)
  let five_over_ln10 = 5.0 /. Float.log 10.0 in
  Nx.add_s (Nx.mul_s (Nx.log dl_mpc) five_over_ln10) 25.0

(* --- Angular scale --- *)

let angular_size ?(p = default) ~z phys =
  let da = angular_diameter_distance ~p z in
  Unit.Angle.of_tensor
    (Nx.div (Unit.Length.to_tensor phys) (Unit.Length.to_tensor da))

let physical_size ?(p = default) ~z ang =
  let da = angular_diameter_distance ~p z in
  Unit.Length.of_tensor
    (Nx.mul (Unit.Angle.to_tensor ang) (Unit.Length.to_tensor da))

(* --- Cosmic times --- *)

(* 1/H0 in seconds: (km/s/Mpc)^{-1} = Mpc/km · s *)
let _hubble_time_s p = Nx.mul_s (Nx.recip p.h0) 3.0856776e19

let lookback_time ?(p = default) z =
  Unit.Time.of_tensor (Nx.mul (_hubble_time_s p) (integrate_inv_z1_ez p z))

let age ?(p = default) z =
  (* age(z) = t_H ∫₀^{1/(1+z)} da/(a E(a)). We reuse gl_quad_a with an upper
     limit at z_max=1000 (≈ a_lo → 0) for the total integral, then subtract the
     lookback from 0 to z. *)
  let t_h_s = _hubble_time_s p in
  let total = integrate_inv_z1_ez p (Nx.scalar f64 1000.0) in
  let lb = integrate_inv_z1_ez p z in
  Unit.Time.of_tensor (Nx.mul t_h_s (Nx.sub total lb))

(* --- z_at_value: inverse lookup via Brent's method ---

   Given a monotonic cosmological function f and a target value, find the
   redshift z such that f(z) ≈ target. Not differentiable. *)

let z_at_value ?(p = default) ?(zmin = 1e-8) ?(zmax = 1000.0) ?(xtol = 1e-8) f
    target =
  let target_v = Nx.item [] target in
  let eval z = Nx.item [] (f ~p (Nx.scalar f64 z)) -. target_v in
  (* Brent's method *)
  let a = ref zmin and b = ref zmax in
  let fa = ref (eval !a) and fb = ref (eval !b) in
  if !fa *. !fb > 0.0 then
    invalid_arg "Cosmo.z_at_value: target outside [f(zmin), f(zmax)]";
  if Float.abs !fa < Float.abs !fb then begin
    let tmp = !a in
    a := !b;
    b := tmp;
    let tmp = !fa in
    fa := !fb;
    fb := tmp
  end;
  let c = ref !a and fc = ref !fa in
  let d = ref (!b -. !a) in
  let mflag = ref true in
  let max_iter = 100 in
  let i = ref 0 in
  while Float.abs !fb > xtol && !i < max_iter do
    let s =
      if Float.abs (!fa -. !fc) > 1e-30 && Float.abs (!fb -. !fc) > 1e-30 then
        (* Inverse quadratic interpolation *)
        let s1 = !a *. !fb *. !fc /. ((!fa -. !fb) *. (!fa -. !fc)) in
        let s2 = !b *. !fa *. !fc /. ((!fb -. !fa) *. (!fb -. !fc)) in
        let s3 = !c *. !fa *. !fb /. ((!fc -. !fa) *. (!fc -. !fb)) in
        s1 +. s2 +. s3
      else
        (* Secant method *)
        !b -. (!fb *. (!b -. !a) /. (!fb -. !fa))
    in
    let cond1 =
      let lo = ((3.0 *. !a) +. !b) /. 4.0 in
      not (if lo < !b then lo <= s && s <= !b else !b <= s && s <= lo)
    in
    let cond2 = !mflag && Float.abs (s -. !b) >= Float.abs (!b -. !c) /. 2.0 in
    let cond3 =
      (not !mflag) && Float.abs (s -. !b) >= Float.abs (!c -. !d) /. 2.0
    in
    let cond4 = !mflag && Float.abs (!b -. !c) < xtol in
    let cond5 = (not !mflag) && Float.abs (!c -. !d) < xtol in
    let s =
      if cond1 || cond2 || cond3 || cond4 || cond5 then begin
        mflag := true;
        (!a +. !b) /. 2.0
      end
      else begin
        mflag := false;
        s
      end
    in
    let fs = eval s in
    d := !c;
    c := !b;
    fc := !fb;
    if !fa *. fs < 0.0 then begin
      b := s;
      fb := fs
    end
    else begin
      a := s;
      fa := fs
    end;
    if Float.abs !fa < Float.abs !fb then begin
      let tmp = !a in
      a := !b;
      b := tmp;
      let tmp = !fa in
      fa := !fb;
      fb := tmp
    end;
    incr i
  done;
  Nx.scalar f64 !b

(* Growth factor and growth rate *)

(* E(a) from scale factor: a = 1/(1+z), so z = 1/a - 1 *)
let e_at_a p a = e_of p (Nx.sub_s (Nx.recip a) 1.0)

(* GL quadrature of f(a') from 0 to a. Transforms [-1,1] to [0,a]. *)
let gl_integrate_a p a f =
  let half = Nx.div_s a 2.0 in
  let a_prime = Nx.add (Nx.mul half gl_nodes) half in
  let e_a = e_at_a p a_prime in
  Nx.mul half (Nx.sum (Nx.mul (f a_prime e_a) gl_weights))

(* Growth integral: J(a) = ∫₀ᵃ da' / (a'³ E³(a')) Integrand at a'→0:
   ~a'^(3/2)/Ω_m^(3/2) → 0, so well-behaved. *)
let growth_integral p a =
  gl_integrate_a p a (fun a_prime e_a ->
      let a3 = Nx.mul a_prime (Nx.mul a_prime a_prime) in
      let e3 = Nx.mul e_a (Nx.mul e_a e_a) in
      Nx.recip (Nx.mul a3 e3))

(* Unnormalized growth factor: D(a) ∝ E(a) × J(a) *)
let growth_unnorm p a = Nx.mul (e_at_a p a) (growth_integral p a)

let growth_factor ?(p = default) z =
  let a = Nx.recip (Nx.add_s z 1.0) in
  let d_a = growth_unnorm p a in
  let d_1 = growth_unnorm p (Nx.scalar f64 1.0) in
  Nx.div d_a d_1

(* Growth rate: f(a) = dlnD/dlna D(a) = E(a) J(a) / const, so f = dlnE/dlna +
   (dJ/dlna) / J = dlnE/dlna + 1 / (a² E³(a) J(a))

   dlnE/dlna = a/(2E²) dE²/da dE²/da = -3Ωm a⁻⁴ - 2Ωk a⁻³ + ΩΛ exp(f_de)
   (-3(1+w0+wa)/a + 3wa) *)
let growth_rate ?(p = default) z =
  let a = Nx.recip (Nx.add_s z 1.0) in
  let e_a = e_at_a p a in
  let e2 = Nx.mul e_a e_a in
  let j_a = growth_integral p a in
  (* dE²/da *)
  let a2 = Nx.mul a a in
  let a3 = Nx.mul a2 a in
  let a4 = Nx.mul a3 a in
  let dm = Nx.mul_s (Nx.div p.omega_m a4) (-3.0) in
  let dk = Nx.mul_s (Nx.div p.omega_k a3) (-2.0) in
  (* Dark energy contribution: need f_de(a) and f_de'(a) *)
  let f_de =
    Nx.add
      (Nx.mul (Nx.mul_s (Nx.add_s (Nx.add p.w0 p.wa) 1.0) (-3.0)) (Nx.log a))
      (Nx.mul p.wa (Nx.mul_s (Nx.sub_s a 1.0) 3.0))
  in
  let f_de_prime =
    Nx.add
      (Nx.div (Nx.mul_s (Nx.add_s (Nx.add p.w0 p.wa) 1.0) (-3.0)) a)
      (Nx.mul_s p.wa 3.0)
  in
  let dde = Nx.mul (Nx.mul p.omega_l (Nx.exp f_de)) f_de_prime in
  let de2_da = Nx.add dm (Nx.add dk dde) in
  (* dlnE/dlna = a/(2E²) × dE²/da *)
  let dln_e = Nx.div (Nx.mul a de2_da) (Nx.mul_s e2 2.0) in
  (* 1/(a² E³ J) *)
  let e3 = Nx.mul e_a e2 in
  let term2 = Nx.recip (Nx.mul a2 (Nx.mul e3 j_a)) in
  Nx.add dln_e term2

(* Eisenstein-Hu transfer function (1998) *)

let t_cmb = 2.7255

(* Eisenstein & Hu (1998) transfer function with baryon oscillations. Scalar
   cosmological quantities are computed in float arithmetic (the transfer
   function is a fitting formula. The wavenumber k may be a tensor of arbitrary
   shape; the result has the same shape. Differentiable through cosmological
   parameters via Rune. *)
let eisenstein_hu p k =
  let s = Nx.scalar f64 in
  let om = p.omega_m in
  let ob = omega_b p in
  let h = Nx.div_s p.h0 100.0 in
  let h2 = Nx.mul h h in
  let w_m = Nx.mul om h2 in
  let w_b = Nx.mul ob h2 in
  let fb = Nx.div ob om in
  let fc = Nx.sub (s 1.0) fb in
  let t27sq = (t_cmb /. 2.7) ** 2.0 in
  let t27_4 = t27sq *. t27sq in
  (* Eq. 2,3: equality epoch *)
  let z_eq = Nx.div_s (Nx.mul_s w_m 2.50e4) t27_4 in
  let k_eq = Nx.div (Nx.div_s (Nx.mul_s w_m 7.46e-2) t27sq) h in
  (* Eq. 4: drag epoch *)
  let b1 =
    Nx.mul
      (Nx.pow w_m (s (-0.419)))
      (Nx.add_s (Nx.mul_s (Nx.pow w_m (s 0.674)) 0.607) 1.0)
    |> fun x -> Nx.mul_s x 0.313
  in
  let b2 = Nx.mul_s (Nx.pow w_m (s 0.223)) 0.238 in
  let z_d =
    Nx.mul
      (Nx.div
         (Nx.mul_s (Nx.pow w_m (s 0.251)) 1291.0)
         (Nx.add_s (Nx.mul_s (Nx.pow w_m (s 0.828)) 0.659) 1.0))
      (Nx.add_s (Nx.mul b1 (Nx.pow w_b b2)) 1.0)
  in
  (* Eq. 5: baryon/photon momentum ratios *)
  let r_d = Nx.mul (Nx.div_s (Nx.mul_s w_b 31.5) t27_4) (Nx.div (s 1e3) z_d) in
  let r_eq =
    Nx.mul (Nx.div_s (Nx.mul_s w_b 31.5) t27_4) (Nx.div (s 1e3) z_eq)
  in
  (* Eq. 6: sound horizon *)
  let sh_d =
    Nx.mul
      (Nx.mul
         (Nx.div (s 2.0) (Nx.mul_s k_eq 3.0))
         (Nx.sqrt (Nx.div (s 6.0) r_eq)))
      (Nx.log
         (Nx.div
            (Nx.add (Nx.sqrt (Nx.add_s r_d 1.0)) (Nx.sqrt (Nx.add r_eq r_d)))
            (Nx.add_s (Nx.sqrt r_eq) 1.0)))
  in
  (* Eq. 7: Silk damping *)
  let k_silk =
    Nx.div
      (Nx.mul
         (Nx.mul (Nx.mul_s (Nx.pow w_b (s 0.52)) 1.6) (Nx.pow w_m (s 0.73)))
         (Nx.add_s (Nx.pow (Nx.mul_s w_m 10.4) (s (-0.95))) 1.0))
      h
  in
  (* CDM transfer function (Eqs. 11, 12, 17, 18) *)
  let a1 =
    Nx.mul
      (Nx.pow (Nx.mul_s w_m 46.9) (s 0.670))
      (Nx.add_s (Nx.pow (Nx.mul_s w_m 32.1) (s (-0.532))) 1.0)
  in
  let a2 =
    Nx.mul
      (Nx.pow (Nx.mul_s w_m 12.0) (s 0.424))
      (Nx.add_s (Nx.pow (Nx.mul_s w_m 45.0) (s (-0.582))) 1.0)
  in
  let alpha_c =
    Nx.mul
      (Nx.pow a1 (Nx.neg fb))
      (Nx.pow a2 (Nx.neg (Nx.mul fb (Nx.mul fb fb))))
  in
  let b1c =
    Nx.div (s 0.944) (Nx.add_s (Nx.pow (Nx.mul_s w_m 458.0) (s (-0.708))) 1.0)
  in
  let b2c = Nx.pow (Nx.mul_s w_m 0.395) (s (-0.0266)) in
  let beta_c =
    Nx.recip (Nx.add_s (Nx.mul b1c (Nx.sub (Nx.pow fc b2c) (s 1.0))) 1.0)
  in
  (* T_tilde: Eq. 10, 19. Operates on k tensor. alpha, beta are scalar
     tensors. *)
  let t_tilde k1 alpha beta =
    let q = Nx.div k1 (Nx.mul_s k_eq 13.41) in
    let l = Nx.log (Nx.add_s (Nx.mul q (Nx.mul_s beta 1.8)) (Float.exp 1.0)) in
    let c =
      Nx.add
        (Nx.div (s 386.0) (Nx.add_s (Nx.mul_s (Nx.pow q (s 1.08)) 69.9) 1.0))
        (Nx.div (s 14.2) alpha)
    in
    Nx.div l (Nx.add l (Nx.mul c (Nx.mul q q)))
  in
  let ksh = Nx.mul k sh_d in
  (* Eq. 17, 18 *)
  let f_ =
    let x = Nx.div_s ksh 5.4 in
    let x2 = Nx.mul x x in
    Nx.recip (Nx.add_s (Nx.mul x2 x2) 1.0)
  in
  let tc =
    Nx.add
      (Nx.mul f_ (t_tilde k (s 1.0) beta_c))
      (Nx.mul (Nx.sub (s 1.0) f_) (t_tilde k alpha_c beta_c))
  in
  (* Baryon transfer function (Eqs. 14, 19, 21) *)
  let y = Nx.div (Nx.add_s z_eq 1.0) (Nx.add_s z_d 1.0) in
  let x_ = Nx.sqrt (Nx.add_s y 1.0) in
  let g_eh =
    Nx.mul y
      (Nx.add (Nx.mul_s x_ (-6.0))
         (Nx.mul
            (Nx.add_s (Nx.mul_s y 3.0) 2.0)
            (Nx.log (Nx.div (Nx.add_s x_ 1.0) (Nx.sub_s x_ 1.0)))))
  in
  let alpha_b =
    Nx.mul_s
      (Nx.mul (Nx.mul k_eq sh_d)
         (Nx.mul (Nx.pow (Nx.add_s r_d 1.0) (s (-0.75))) g_eh))
      2.07
  in
  let beta_node = Nx.mul_s (Nx.pow w_m (s 0.435)) 8.41 in
  let beta_b =
    Nx.add (Nx.add_s fb 0.5)
      (Nx.mul
         (Nx.sub_s (Nx.mul_s fb 2.0) 3.0)
         (Nx.neg
            (Nx.sqrt
               (Nx.add_s (Nx.mul (Nx.mul_s w_m 17.2) (Nx.mul_s w_m 17.2)) 1.0))))
  in
  (* Eq. 22: tilde_s per-k *)
  let tilde_s =
    let bns = Nx.div beta_node ksh in
    let bns3 = Nx.mul bns (Nx.mul bns bns) in
    Nx.div sh_d (Nx.pow (Nx.add_s bns3 1.0) (s (1.0 /. 3.0)))
  in
  let tb =
    let term1 =
      Nx.div
        (t_tilde k (s 1.0) (s 1.0))
        (Nx.add_s
           (let x = Nx.div_s ksh 5.2 in
            Nx.mul x x)
           1.0)
    in
    let bbks = Nx.div beta_b ksh in
    let bbks3 = Nx.mul bbks (Nx.mul bbks bbks) in
    let term2 =
      Nx.mul
        (Nx.div alpha_b (Nx.add_s bbks3 1.0))
        (Nx.exp (Nx.neg (Nx.pow (Nx.div k k_silk) (s 1.4))))
    in
    let sinc_arg = Nx.mul k tilde_s in
    Nx.mul (Nx.add term1 term2) (Nx.div (Nx.sin sinc_arg) sinc_arg)
  in
  (* Total: fb * Tb + fc * Tc *)
  Nx.add (Nx.mul tb fb) (Nx.mul tc fc)

(* Matter power spectrum *)

(* Simpson's rule integration on a uniform grid of n+1 points from a to b. n
   must be even. f is evaluated at each grid point, returns [n+1] tensor. *)
let simps_integrate f a b n =
  let h = (b -. a) /. Float.of_int n in
  let xs =
    Nx.create f64
      [| n + 1 |]
      (Array.init (n + 1) (fun i -> a +. (Float.of_int i *. h)))
  in
  let ys = f xs in
  (* Simpson weights: 1, 4, 2, 4, 2, ..., 4, 1 *)
  let w =
    Array.init (n + 1) (fun i ->
        if i = 0 || i = n then 1.0 else if i mod 2 = 1 then 4.0 else 2.0)
  in
  let weights = Nx.create f64 [| n + 1 |] w in
  Nx.mul_s (Nx.sum (Nx.mul ys weights)) (h /. 3.0)

(* σ²(R) = 1/(2π²) ∫ k³ P_unnorm(k) W²(kR) d(ln k) where P_unnorm = k^n_s ×
   T²(k) and W is the top-hat window. Integration in ln(k) space: the integrand
   is k³ P W² (the dk/k from d(ln k) cancels one power of k, giving k² P W² dk
   equivalent). *)
let sigma_sq p r =
  let ns = n_s p in
  simps_integrate
    (fun lnk ->
      let k = Nx.exp lnk in
      let x = Nx.mul_s k r in
      (* Top-hat window: W(x) = 3(sin x - x cos x)/x³ *)
      let x2 = Nx.mul x x in
      let x3 = Nx.mul x2 x in
      let w =
        Nx.div (Nx.mul_s (Nx.sub (Nx.sin x) (Nx.mul x (Nx.cos x))) 3.0) x3
      in
      let t = eisenstein_hu p k in
      let pk = Nx.mul (Nx.pow k ns) (Nx.mul t t) in
      let k3 = Nx.mul k (Nx.mul k k) in
      Nx.mul k3 (Nx.mul (Nx.mul w w) pk))
    (Float.log 1e-4) (Float.log 1e4) 512
  |> fun integral -> Nx.div_s integral (2.0 *. Float.pi *. Float.pi)

let linear_power ?(p = default) k z =
  let s8 = sigma8 p in
  let g = growth_factor ~p z in
  let t = eisenstein_hu p k in
  let ns = n_s p in
  let pk_unnorm = Nx.mul (Nx.pow k ns) (Nx.mul t t) in
  (* Normalization: A = σ8² / σ²_unnorm(R=8) *)
  let s2 = sigma_sq p 8.0 in
  let norm = Nx.div (Nx.mul s8 s8) s2 in
  Nx.mul norm (Nx.mul pk_unnorm (Nx.mul g g))

(* Halofit (Takahashi et al. 2012) *)

(* Ω_m(a) = Ω_m a⁻³ / E²(a) *)
let omega_m_a p a =
  let e2 =
    let e = e_at_a p a in
    Nx.mul e e
  in
  let a3 = Nx.mul a (Nx.mul a a) in
  Nx.div (Nx.div p.omega_m a3) e2

(* Ω_de(a) = Ω_Λ exp(f_de(a)) / E²(a) *)
let omega_de_a p a =
  let e2 =
    let e = e_at_a p a in
    Nx.mul e e
  in
  let f_de =
    Nx.add
      (Nx.mul (Nx.mul_s (Nx.add_s (Nx.add p.w0 p.wa) 1.0) (-3.0)) (Nx.log a))
      (Nx.mul p.wa (Nx.mul_s (Nx.sub_s a 1.0) 3.0))
  in
  Nx.div (Nx.mul p.omega_l (Nx.exp f_de)) e2

(* w(a) = w0 + wa(1-a) *)
let w_of p a = Nx.add p.w0 (Nx.mul p.wa (Nx.sub (Nx.scalar f64 1.0) a))

(* σ²(R, z) using linear P(k) at z=0, scaled by D²(z)/D²(0)=D²(z). For Halofit
   we need σ(R) at various R to find k_nl, plus derivatives. *)
let sigma_sq_at_z p r z =
  let g = growth_factor ~p z in
  Nx.mul (sigma_sq p r) (Nx.mul g g)

(* Find k_nl where σ(1/k_nl, z) = 1, plus n_eff and C at the nonlinear scale. We
   compute σ²(R) on a grid, interpolate to find R_nl, then compute spectral
   index and curvature from Gaussian-filtered integrals. *)
let halofit_params p z =
  let g = growth_factor ~p z in
  let g2 = Nx.mul g g in
  let ns = n_s p in
  let s8 = sigma8 p in
  let s2_8 = sigma_sq p 8.0 in
  let pknorm = Nx.div (Nx.mul s8 s8) s2_8 in
  let n_r = 256 in
  let logr =
    Nx.create f64 [| n_r |]
      (Array.init n_r (fun i ->
           Float.log 1e-4
           +. Float.of_int i
              *. (Float.log 1e1 -. Float.log 1e-4)
              /. Float.of_int (n_r - 1)))
  in
  (* Compute σ²(R) for each R using Gaussian filter exp(-(kR)²) *)
  let n_k = 512 in
  let lnk_min = Float.log 1e-4 in
  let lnk_max = Float.log 1e4 in
  let dlnk = (lnk_max -. lnk_min) /. Float.of_int (n_k - 1) in
  let lnk =
    Nx.create f64 [| n_k |]
      (Array.init n_k (fun i -> lnk_min +. (Float.of_int i *. dlnk)))
  in
  let k = Nx.exp lnk in
  let t = eisenstein_hu p k in
  let pk_base = Nx.mul pknorm (Nx.mul (Nx.pow k ns) (Nx.mul t t)) in
  let pk_at_z = Nx.mul pk_base g2 in
  (* k³ P(k) / (2π²) *)
  let k3pk =
    Nx.div
      (Nx.mul (Nx.mul k (Nx.mul k k)) pk_at_z)
      (Nx.scalar f64 (2.0 *. Float.pi *. Float.pi))
  in
  (* Trapezoidal weights [n_k] *)
  let trap_w =
    Nx.create f64 [| n_k |]
      (Array.init n_k (fun j -> if j = 0 || j = n_k - 1 then 0.5 else 1.0))
  in
  (* Float-level σ²(R) grid for root-finding *)
  let sigma2_arr = Array.make n_r 0.0 in
  for i = 0 to n_r - 1 do
    let r = Float.exp (Nx.item [ i ] logr) in
    let kr = Nx.mul_s k r in
    let y2 = Nx.mul kr kr in
    let gauss = Nx.exp (Nx.neg y2) in
    let integrand = Nx.mul k3pk gauss in
    sigma2_arr.(i) <-
      Nx.item [] (Nx.mul_s (Nx.sum (Nx.mul trap_w integrand)) dlnk)
  done;
  (* Find R_nl where σ² = 1 by linear interpolation in log space *)
  let r_nl = ref (Float.exp (Nx.item [ 0 ] logr)) in
  (let found = ref false in
   for i = 0 to n_r - 2 do
     if (not !found) && sigma2_arr.(i) >= 1.0 && sigma2_arr.(i + 1) <= 1.0 then begin
       let ls0 = Float.log sigma2_arr.(i) in
       let ls1 = Float.log sigma2_arr.(i + 1) in
       let lr0 = Nx.item [ i ] logr in
       let lr1 = Nx.item [ i + 1 ] logr in
       let frac = (0.0 -. ls0) /. (ls1 -. ls0) in
       r_nl := Float.exp (lr0 +. (frac *. (lr1 -. lr0)));
       found := true
     end
   done);
  let r_nl_f = !r_nl in
  (* Differentiable Newton refinement for R_nl. Compute σ² at the float root,
     then one Newton step: R' = R + R*(σ²-1)/dn where dσ²/dR = -dn/R.
     Numerically R' ≈ R, but the gradient dR'/dp = -(∂σ²/∂p)/(∂σ²/∂R) is exact
     via the implicit function theorem. *)
  let kr0 = Nx.mul_s k r_nl_f in
  let y2_0 = Nx.mul kr0 kr0 in
  let gauss0 = Nx.exp (Nx.neg y2_0) in
  let integrand0 = Nx.mul k3pk gauss0 in
  let trap_sum f = Nx.mul_s (Nx.sum (Nx.mul trap_w f)) dlnk in
  let s2_0 = trap_sum integrand0 in
  let dn_0 = trap_sum (Nx.mul_s (Nx.mul integrand0 y2_0) 2.0) in
  let r_nl_t =
    Nx.add_s (Nx.mul_s (Nx.div (Nx.sub_s s2_0 1.0) dn_0) r_nl_f) r_nl_f
  in
  let k_nl = Nx.recip r_nl_t in
  (* Recompute n_eff and C at the tensor R_nl for full differentiability. *)
  let kr = Nx.mul k r_nl_t in
  let y2 = Nx.mul kr kr in
  let gauss = Nx.exp (Nx.neg y2) in
  let integrand = Nx.mul k3pk gauss in
  let s2 = trap_sum integrand in
  let dn = trap_sum (Nx.mul_s (Nx.mul integrand y2) 2.0) in
  let dc =
    trap_sum (Nx.mul (Nx.mul_s integrand 4.0) (Nx.sub y2 (Nx.mul y2 y2)))
  in
  let n_eff = Nx.sub_s dn 3.0 in
  let c_curv = Nx.add (Nx.mul dn dn) (Nx.div dc s2) in
  (k_nl, n_eff, c_curv)

let nonlinear_power ?(p = default) k z =
  let s = Nx.scalar f64 in
  let pk_lin = linear_power ~p k z in
  let k_nl, n, c = halofit_params p z in
  let n2 = Nx.mul n n in
  let n3 = Nx.mul n2 n in
  let n4 = Nx.mul n3 n in
  let a = Nx.recip (Nx.add_s z 1.0) in
  let om_m = omega_m_a p a in
  let om_de = omega_de_a p a in
  let w = w_of p a in
  let odew1 = Nx.mul om_de (Nx.add_s w 1.0) in
  (* Takahashi et al. 2012 coefficients — all tensor *)
  let a_n =
    Nx.pow (s 10.0)
      (Nx.add
         (Nx.add
            (Nx.add
               (Nx.add
                  (Nx.add
                     (Nx.add_s (Nx.mul_s n 2.8553) 1.5222)
                     (Nx.mul_s n2 2.3706))
                  (Nx.mul_s n3 0.9903))
               (Nx.mul_s n4 0.2250))
            (Nx.mul_s c (-0.6038)))
         (Nx.mul_s odew1 0.1749))
  in
  let b_n =
    Nx.pow (s 10.0)
      (Nx.add
         (Nx.add
            (Nx.add
               (Nx.add_s (Nx.mul_s n 0.5864) (-0.5642))
               (Nx.mul_s n2 0.5716))
            (Nx.mul_s c (-1.5474)))
         (Nx.mul_s odew1 0.2279))
  in
  let c_n =
    Nx.pow (s 10.0)
      (Nx.add
         (Nx.add (Nx.add_s (Nx.mul_s n 2.0404) 0.3698) (Nx.mul_s n2 0.8161))
         (Nx.mul_s c 0.5869))
  in
  let gamma_n =
    Nx.add (Nx.add_s (Nx.mul_s n (-0.0843)) 0.1971) (Nx.mul_s c 0.8460)
  in
  let alpha_n =
    Nx.abs
      (Nx.add
         (Nx.add (Nx.add_s (Nx.mul_s n 1.3373) 6.0835) (Nx.mul_s n2 (-0.1959)))
         (Nx.mul_s c (-5.5274)))
  in
  let beta_n =
    Nx.add
      (Nx.add
         (Nx.add
            (Nx.add
               (Nx.add_s (Nx.mul_s n (-0.7354)) 2.0379)
               (Nx.mul_s n2 0.3157))
            (Nx.mul_s n3 1.2490))
         (Nx.mul_s n4 0.3980))
      (Nx.mul_s c (-0.1682))
  in
  let nu_n = Nx.pow (s 10.0) (Nx.add_s (Nx.mul_s n 3.6902) 5.2105) in
  let f1 = Nx.pow om_m (s (-0.0307)) in
  let f2 = Nx.pow om_m (s (-0.0585)) in
  let f3 = Nx.pow om_m (s 0.0743) in
  let y = Nx.div k k_nl in
  (* Δ²_L = k³ P_lin / (2π²) *)
  let d2l =
    Nx.div
      (Nx.mul (Nx.mul k (Nx.mul k k)) pk_lin)
      (s (2.0 *. Float.pi *. Float.pi))
  in
  (* f(y) = y/4 + y²/8 *)
  let fy = Nx.add (Nx.div_s y 4.0) (Nx.div_s (Nx.mul y y) 8.0) in
  (* Quasi-linear term: Δ²_Q *)
  let d2q =
    Nx.mul d2l
      (Nx.mul
         (Nx.div
            (Nx.pow (Nx.add_s d2l 1.0) beta_n)
            (Nx.add_s (Nx.mul d2l alpha_n) 1.0))
         (Nx.exp (Nx.neg fy)))
  in
  (* Halo term: Δ²_H *)
  let three_f1 = Nx.mul_s f1 3.0 in
  let d2h_prime =
    Nx.div
      (Nx.mul a_n (Nx.pow y three_f1))
      (Nx.add_s
         (Nx.add
            (Nx.mul b_n (Nx.pow y f2))
            (Nx.pow (Nx.mul (Nx.mul c_n f3) y) (Nx.sub_s gamma_n 3.0)))
         1.0)
  in
  let d2h = Nx.div d2h_prime (Nx.add_s (Nx.div nu_n (Nx.mul y y)) 1.0) in
  let d2nl = Nx.add d2q d2h in
  Nx.div (Nx.mul_s d2nl (2.0 *. Float.pi *. Float.pi)) (Nx.mul k (Nx.mul k k))

(* BAO distance measures *)

let dh ?(p = default) z =
  Unit.Length.of_tensor (Nx.mul_s (Nx.div c_km_s (hubble ~p z)) _mpc_m)

let dm ?(p = default) z =
  Unit.Length.of_tensor (Nx.mul_s (transverse_comoving_mpc p z) _mpc_m)

let dv ?(p = default) z =
  let dh_mpc = Nx.div c_km_s (hubble ~p z) in
  let dm_mpc = transverse_comoving_mpc p z in
  let cube = Nx.mul z (Nx.mul dh_mpc (Nx.mul dm_mpc dm_mpc)) in
  Unit.Length.of_tensor (Nx.mul_s (Nx.pow_s cube (1.0 /. 3.0)) _mpc_m)

let sound_horizon ?(p = default) () =
  let ob = omega_b p in
  let h = Nx.div_s p.h0 100.0 in
  let h2 = Nx.mul h h in
  let w_m = Nx.mul p.omega_m h2 in
  let w_b = Nx.mul ob h2 in
  (* Eisenstein & Hu (1998) Eq. 2–6: sound horizon at drag epoch in Mpc/h *)
  let t27sq = (t_cmb /. 2.7) ** 2.0 in
  let t27_4 = t27sq *. t27sq in
  let z_eq = Nx.div_s (Nx.mul_s w_m 2.50e4) t27_4 in
  let k_eq = Nx.div (Nx.div_s (Nx.mul_s w_m 7.46e-2) t27sq) h in
  let b1_z =
    Nx.mul
      (Nx.pow w_m (Nx.scalar f64 (-0.419)))
      (Nx.add_s (Nx.mul_s (Nx.pow w_m (Nx.scalar f64 0.674)) 0.607) 1.0)
    |> fun x -> Nx.mul_s x 0.313
  in
  let b2_z = Nx.mul_s (Nx.pow w_m (Nx.scalar f64 0.223)) 0.238 in
  let z_d =
    Nx.mul
      (Nx.div
         (Nx.mul_s (Nx.pow w_m (Nx.scalar f64 0.251)) 1291.0)
         (Nx.add_s (Nx.mul_s (Nx.pow w_m (Nx.scalar f64 0.828)) 0.659) 1.0))
      (Nx.add_s (Nx.mul b1_z (Nx.pow w_b b2_z)) 1.0)
  in
  let r_d =
    Nx.mul (Nx.div_s (Nx.mul_s w_b 31.5) t27_4) (Nx.div (Nx.scalar f64 1e3) z_d)
  in
  let r_eq =
    Nx.mul
      (Nx.div_s (Nx.mul_s w_b 31.5) t27_4)
      (Nx.div (Nx.scalar f64 1e3) z_eq)
  in
  (* Eq. 6 from Eisenstein & Hu: sound horizon in Mpc/h *)
  let sh_d =
    Nx.mul
      (Nx.mul
         (Nx.div (Nx.scalar f64 2.0) (Nx.mul_s k_eq 3.0))
         (Nx.sqrt (Nx.div (Nx.scalar f64 6.0) r_eq)))
      (Nx.log
         (Nx.div
            (Nx.add (Nx.sqrt (Nx.add_s r_d 1.0)) (Nx.sqrt (Nx.add r_eq r_d)))
            (Nx.add_s (Nx.sqrt r_eq) 1.0)))
  in
  (* sh_d is in Mpc/h, convert to Mpc then to metres *)
  let rs_mpc = Nx.div sh_d h in
  Unit.Length.of_tensor (Nx.mul_s rs_mpc _mpc_m)
