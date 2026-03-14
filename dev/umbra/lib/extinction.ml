(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let f64 = Nx.float64

(* Extinction law: wavelength in metres → A_λ/A_V *)
type law = wavelength:Nx.float64_t -> Nx.float64_t

(* Horner evaluation: c0 + y*(c1 + y*(c2 + ...)) *)
let horner y coeffs =
  let n = Array.length coeffs in
  let acc = ref (Nx.scalar f64 coeffs.(n - 1)) in
  for i = n - 2 downto 0 do
    acc := Nx.add_s (Nx.mul y !acc) coeffs.(i)
  done;
  !acc

(* Shared CCM89/O'Donnell94 implementation parameterized by R_V. Only the
   optical a/b polynomial coefficients differ between the two laws; IR and UV
   regions are identical. Uses Nx.where for differentiable piecewise selection.
   Valid for 0.125–3.5 μm (x = 0.3–8.0 μm⁻¹). *)
let ccm89_impl a_opt b_opt ~rv ~wavelength =
  (* Convert wavelength (m) to inverse microns *)
  let x = Nx.div (Nx.scalar f64 1e-6) wavelength in
  (* Infrared: 0.3 ≤ x < 1.1 *)
  let a_ir = Nx.mul_s (Nx.pow_s x 1.61) 0.574 in
  let b_ir = Nx.mul_s (Nx.pow_s x 1.61) (-0.527) in
  (* Optical/NIR: 1.1 ≤ x ≤ 3.3, polynomial in (x - 1.82) *)
  let y = Nx.sub_s x 1.82 in
  let a_o = horner y a_opt in
  let b_o = horner y b_opt in
  (* UV: 3.3 < x ≤ 8.0 *)
  let fa =
    Nx.where (Nx.greater_equal_s x 5.9)
      (Nx.add
         (Nx.mul_s (Nx.square (Nx.sub_s x 5.9)) (-0.04473))
         (Nx.mul_s (Nx.pow_s (Nx.sub_s x 5.9) 3.0) (-0.009779)))
      (Nx.scalar f64 0.0)
  in
  let fb =
    Nx.where (Nx.greater_equal_s x 5.9)
      (Nx.add
         (Nx.mul_s (Nx.square (Nx.sub_s x 5.9)) 0.2130)
         (Nx.mul_s (Nx.pow_s (Nx.sub_s x 5.9) 3.0) 0.1207))
      (Nx.scalar f64 0.0)
  in
  (* a(x) = 1.752 - 0.316*x - 0.104/((x-4.67)² + 0.341) + F_a *)
  let a_uv_base = Nx.add_s (Nx.mul_s x (-0.316)) 1.752 in
  let bump_a =
    Nx.div (Nx.scalar f64 (-0.104))
      (Nx.add (Nx.square (Nx.sub_s x 4.67)) (Nx.scalar f64 0.341))
  in
  let a_uv = Nx.add (Nx.add a_uv_base bump_a) fa in
  (* b(x) = -3.090 + 1.825*x + 1.206/((x-4.62)² + 0.263) + F_b *)
  let b_uv_base = Nx.add_s (Nx.mul_s x 1.825) (-3.090) in
  let bump_b =
    Nx.div (Nx.scalar f64 1.206)
      (Nx.add (Nx.square (Nx.sub_s x 4.62)) (Nx.scalar f64 0.263))
  in
  let b_uv = Nx.add (Nx.add b_uv_base bump_b) fb in
  (* Piecewise selection using Nx.where *)
  let ir_mask = Nx.less_s x 1.1 in
  let uv_mask = Nx.greater_s x 3.3 in
  let a = Nx.where ir_mask a_ir (Nx.where uv_mask a_uv a_o) in
  let b = Nx.where ir_mask b_ir (Nx.where uv_mask b_uv b_o) in
  (* A_λ/A_V = a(x) + b(x)/R_V *)
  Nx.add a (Nx.div b rv)

(* CCM89: Cardelli, Clayton & Mathis 1989, ApJ 345, 245 — optical
   coefficients *)
let ccm89_a =
  [| 1.0; 0.17699; -0.50447; -0.02427; 0.72085; 0.01979; -0.77530; 0.32999 |]

let ccm89_b =
  [| 0.0; 1.41338; 2.28305; 1.07233; -5.38434; -0.62251; 5.30260; -2.09002 |]

let ccm89 ~rv = fun ~wavelength -> ccm89_impl ccm89_a ccm89_b ~rv ~wavelength

(* O'Donnell 1994, ApJ 422, 158 — revised optical coefficients *)
let od94_a = [| 1.0; 0.104; -0.609; 0.701; -1.221; 0.700; -0.048; -0.091 |]
let od94_b = [| 0.0; 1.952; 2.908; -3.989; 7.985; -5.002; -0.478; 1.149 |]
let odonnell94 ~rv = fun ~wavelength -> ccm89_impl od94_a od94_b ~rv ~wavelength

(* Calzetti 2000: Calzetti et al. 2000, ApJ 533, 682. Starburst attenuation law.
   Fixed R_V = 4.05. Valid 0.12–2.2 μm. *)
let calzetti00 =
 fun ~wavelength ->
  let lam_um = Nx.mul_s wavelength 1e6 in
  let rv = 4.05 in
  (* Blue: 0.12–0.63 μm k'(λ) = 2.659 * (-2.156 + 1.509/λ - 0.198/λ² + 0.011/λ³)
     + R_V *)
  let k_blue =
    Nx.add_s
      (Nx.mul_s
         (Nx.add_s
            (Nx.add
               (Nx.mul_s (Nx.recip lam_um) 1.509)
               (Nx.add
                  (Nx.mul_s (Nx.pow_s lam_um (-2.0)) (-0.198))
                  (Nx.mul_s (Nx.pow_s lam_um (-3.0)) 0.011)))
            (-2.156))
         2.659)
      rv
  in
  (* Red: 0.63–2.2 μm k'(λ) = 2.659 * (-1.857 + 1.040/λ) + R_V *)
  let k_red =
    Nx.add_s
      (Nx.mul_s (Nx.add_s (Nx.mul_s (Nx.recip lam_um) 1.040) (-1.857)) 2.659)
      rv
  in
  let blue_mask = Nx.less_s lam_um 0.63 in
  let k = Nx.where blue_mask k_blue k_red in
  (* A_λ/A_V = k'(λ) / R_V *)
  Nx.div_s k rv

(* Fitzpatrick 1999: Fitzpatrick 1999, PASP 111, 63. R_V-dependent extinction
   using cubic spline for optical/NIR and Fitzpatrick & Massa parameterization
   for UV. Valid 0.1–3.5 μm. *)

(* FM UV parameters (fixed) *)
let f99_x0_sq = 4.596 *. 4.596
let f99_gamma_sq = 0.99 *. 0.99
let f99_c3 = 3.23
let f99_c4 = 0.41
let f99_c5 = 5.9

(* Spline anchor x-values (inverse microns) *)
let f99_xk =
  [|
    0.;
    1e4 /. 26500.;
    1e4 /. 12200.;
    1e4 /. 6000.;
    1e4 /. 5470.;
    1e4 /. 4670.;
    1e4 /. 4110.;
    1e4 /. 2700.;
    1e4 /. 2600.;
  |]

let f99_hk = Array.init 8 (fun i -> f99_xk.(i + 1) -. f99_xk.(i))

(* Drude profile at a fixed x-value *)
let f99_drude x =
  let x2 = x *. x in
  let y = x2 -. f99_x0_sq in
  x2 /. ((y *. y) +. (x2 *. f99_gamma_sq))

(* Precompute spline basis matrix M (7×9): maps 9 anchor y-values to 7 interior
   second derivatives. Natural boundary conditions: m[0] = m[8] = 0.

   The tridiagonal system Am = Dy is solved offline; M = A⁻¹D is stored. At
   runtime m[j] = Σ M[j][i] y[i] — a weighted sum of Nx scalars, fully
   differentiable through Rune. *)
let f99_basis =
  let n = 7 in
  let h = f99_hk in
  (* Right-hand side matrix D (7×9) *)
  let d_mat =
    Array.init n (fun j ->
        Array.init 9 (fun i ->
            if i = j then 6.0 /. h.(j)
            else if i = j + 1 then ~-.((6.0 /. h.(j + 1)) +. (6.0 /. h.(j)))
            else if i = j + 2 then 6.0 /. h.(j + 1)
            else 0.0))
  in
  (* Tridiagonal A: diag, sub, sup *)
  let diag = Array.init n (fun j -> 2.0 *. (h.(j) +. h.(j + 1))) in
  let sub j = h.(j) in
  let sup j = h.(j + 1) in
  (* Solve A X_col = D_col for each of 9 columns via Thomas algorithm *)
  let m = Array.init n (fun _ -> Array.make 9 0.0) in
  for col = 0 to 8 do
    let b = Array.init n (fun j -> d_mat.(j).(col)) in
    let c = Array.make n 0.0 in
    let d = Array.make n 0.0 in
    c.(0) <- sup 0 /. diag.(0);
    d.(0) <- b.(0) /. diag.(0);
    for i = 1 to n - 1 do
      let w = diag.(i) -. (sub i *. c.(i - 1)) in
      c.(i) <- (if i < n - 1 then sup i /. w else 0.0);
      d.(i) <- (b.(i) -. (sub i *. d.(i - 1))) /. w
    done;
    m.(n - 1).(col) <- d.(n - 1);
    for i = n - 2 downto 0 do
      m.(i).(col) <- d.(i) -. (c.(i) *. m.(i + 1).(col))
    done
  done;
  m

(* Evaluate a cubic spline piece on [xk, xk1] at tensor x. mk and mk1 are second
   derivatives (Nx scalars); yk, yk1 are y-values. *)
let f99_eval_piece hk xk yk yk1 mk mk1 x =
  let a = yk in
  let c = Nx.mul_s mk 0.5 in
  let d = Nx.div_s (Nx.sub mk1 mk) (6.0 *. hk) in
  let b =
    Nx.sub
      (Nx.div_s (Nx.sub yk1 yk) hk)
      (Nx.mul_s (Nx.add (Nx.mul_s mk 2.0) mk1) (hk /. 6.0))
  in
  let t = Nx.sub_s x xk in
  Nx.add a (Nx.mul t (Nx.add b (Nx.mul t (Nx.add c (Nx.mul t d)))))

let fitzpatrick99 ~rv =
  let rv2 = Nx.mul rv rv in
  let rv3 = Nx.mul rv2 rv in
  let rv4 = Nx.mul rv2 rv2 in
  (* FM UV c1, c2 — computed once, used for anchor y-values and the closure *)
  let c2_uv = Nx.add_s (Nx.mul_s (Nx.recip rv) 4.717) (-0.824) in
  let c1_uv = Nx.sub (Nx.scalar f64 2.030) (Nx.mul_s c2_uv 3.007) in
  let uv_anchor xk =
    Nx.add c1_uv (Nx.add_s (Nx.mul_s c2_uv xk) (f99_c3 *. f99_drude xk))
  in
  (* 9 anchor E(λ-V)/E(B-V) values *)
  let y =
    [|
      Nx.neg rv;
      Nx.sub (Nx.mul_s rv (0.26469 /. 3.1)) rv;
      Nx.sub (Nx.mul_s rv (0.82925 /. 3.1)) rv;
      Nx.sub
        (Nx.add
           (Nx.add_s (Nx.mul_s rv 1.00270) (-0.422809))
           (Nx.mul_s rv2 2.13572e-04))
        rv;
      Nx.sub
        (Nx.add
           (Nx.add_s (Nx.mul_s rv 1.00216) (-5.13540e-02))
           (Nx.mul_s rv2 (-7.35778e-05)))
        rv;
      Nx.sub
        (Nx.add
           (Nx.add_s (Nx.mul_s rv 1.00184) 0.700127)
           (Nx.mul_s rv2 (-3.32598e-05)))
        rv;
      Nx.sub
        (Nx.add
           (Nx.add
              (Nx.add
                 (Nx.add_s (Nx.mul_s rv 1.01707) 1.19456)
                 (Nx.mul_s rv2 (-5.46959e-03)))
              (Nx.mul_s rv3 7.97809e-04))
           (Nx.mul_s rv4 (-4.45636e-05)))
        rv;
      (* UV anchors from FM parameterization *)
      uv_anchor f99_xk.(7);
      uv_anchor f99_xk.(8);
    |]
  in
  (* Second derivatives m[0..8]: m[0] = m[8] = 0, m[1..7] from basis matrix *)
  let zero = Nx.scalar f64 0.0 in
  let m2 = Array.make 9 zero in
  for j = 0 to 6 do
    let acc = ref zero in
    for i = 0 to 8 do
      acc := Nx.add !acc (Nx.mul_s y.(i) f99_basis.(j).(i))
    done;
    m2.(j + 1) <- !acc
  done;
  (* Precompute spline piece coefficients for intervals 0..6 *)
  let h = f99_hk in
  let pieces =
    Array.init 7 (fun k ->
        let hk = h.(k) in
        let yk = y.(k) in
        let yk1 = y.(k + 1) in
        let mk = m2.(k) in
        let mk1 = m2.(k + 1) in
        (hk, f99_xk.(k), yk, yk1, mk, mk1))
  in
  fun ~wavelength ->
    (* Convert wavelength (m) to inverse microns *)
    let x = Nx.div (Nx.scalar f64 1e-6) wavelength in
    (* Evaluate spline for each interval *)
    let eval k =
      let hk, xk, yk, yk1, mk, mk1 = pieces.(k) in
      f99_eval_piece hk xk yk yk1 mk mk1 x
    in
    let s0 = eval 0 in
    let s1 = eval 1 in
    let s2 = eval 2 in
    let s3 = eval 3 in
    let s4 = eval 4 in
    let s5 = eval 5 in
    let s6 = eval 6 in
    let opt_nir =
      Nx.where
        (Nx.less_s x f99_xk.(1))
        s0
        (Nx.where
           (Nx.less_s x f99_xk.(2))
           s1
           (Nx.where
              (Nx.less_s x f99_xk.(3))
              s2
              (Nx.where
                 (Nx.less_s x f99_xk.(4))
                 s3
                 (Nx.where
                    (Nx.less_s x f99_xk.(5))
                    s4
                    (Nx.where (Nx.less_s x f99_xk.(6)) s5 s6)))))
    in
    (* UV: FM parameterization for x ≥ 1e4/2700 *)
    let x2 = Nx.square x in
    let y_bump = Nx.sub x2 (Nx.scalar f64 f99_x0_sq) in
    let drude =
      Nx.div x2 (Nx.add (Nx.mul y_bump y_bump) (Nx.mul_s x2 f99_gamma_sq))
    in
    let fuv =
      Nx.where
        (Nx.greater_equal_s x f99_c5)
        (let dx = Nx.sub_s x f99_c5 in
         let dx2 = Nx.square dx in
         Nx.add (Nx.mul_s dx2 0.5392) (Nx.mul_s (Nx.mul dx2 dx) 0.05644))
        (Nx.scalar f64 0.0)
    in
    let k_uv =
      Nx.add c1_uv
        (Nx.add (Nx.mul c2_uv x)
           (Nx.add (Nx.mul_s drude f99_c3) (Nx.mul_s fuv f99_c4)))
    in
    (* Select optical/NIR vs UV *)
    let e_over_ebv = Nx.where (Nx.less_s x f99_xk.(7)) opt_nir k_uv in
    (* A(λ)/A(V) = E(λ-V)/E(B-V) / R_V + 1 *)
    Nx.add_s (Nx.div e_over_ebv rv) 1.0

let curve law ~wavelength = law ~wavelength:(Unit.Length.to_tensor wavelength)
let ln10_over_2_5 = Float.log 10.0 *. 0.4

let scale_flux sign law ~av spectrum =
  let wave_m = Unit.Length.to_tensor (Spectrum.wavelength spectrum) in
  let a_lambda = Nx.mul (law ~wavelength:wave_m) av in
  let factor = Nx.exp (Nx.mul_s a_lambda (sign *. ln10_over_2_5)) in
  Spectrum.scale factor spectrum

let apply law ~av spectrum = scale_flux (-1.0) law ~av spectrum
let unredden law ~av spectrum = scale_flux 1.0 law ~av spectrum
