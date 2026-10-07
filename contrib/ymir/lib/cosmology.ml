(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The background of a homogeneous expanding universe.

   Every result reads the scaled expansion rate

     a^4 E^2(a) = Omega_cb a + Omega_k a^2 + Omega_de a^4 exp(-3(1+w0+wa) ln a
                  - 3 wa (1-a)) + Omega_r(a)

   at the scale factor a, whose terms stay bounded from today to the big bang.
   A distance or a time is one integral of it, a Gauss-Legendre sum over the
   nodes of [Cosmology_tables], in a variable where the integrand is smooth. A
   node axis stands in front of the lanes, so every leaf meets its own lane.

   Each function runs on safe inputs (z > -1, H0 > 0, a^4 E^2 > 0 at every node)
   and [where] puts NaN in the lanes where an input was not, so that every
   partial derivative is finite and an unselected lane's cotangent stays
   zero. *)

open Ymir_units
module T = Cosmology_tables

let strf = Printf.sprintf

type 'p t = {
  codata : Codata.t;
  h0 : 'p Quantity.t;
  omega_cb : 'p;
  omega_k : 'p;
  w0 : 'p;
  wa : 'p;
  t_cmb : 'p Quantity.t;
  n_eff : 'p;
  m_nu : 'p Quantity.t;
}

let walk c x =
  let open Nx.Ptree.Walk in
  let year c codata = ignore (int c (Codata.year codata)) in
  field c "codata" year x.codata;
  let h0 = field c "h0" Quantity.walk x.h0 in
  let omega_cb = field c "omega_cb" leaf x.omega_cb in
  let omega_k = field c "omega_k" leaf x.omega_k in
  let w0 = field c "w0" leaf x.w0 in
  let wa = field c "wa" leaf x.wa in
  let t_cmb = field c "t_cmb" Quantity.walk x.t_cmb in
  let n_eff = field c "n_eff" leaf x.n_eff in
  let m_nu = field c "m_nu" Quantity.walk x.m_nu in
  { codata = x.codata; h0; omega_cb; omega_k; w0; wa; t_cmb; n_eff; m_nu }

let pp ppf c =
  Format.fprintf ppf
    "@[<hv 1>{codata = %d;@ h0 = %a;@ omega_cb = %a;@ omega_k = %a;@ w0 = %a;@ \
     wa = %a;@ t_cmb = %a;@ n_eff = %a;@ m_nu = %a}@]"
    (Codata.year c.codata) Quantity.pp c.h0 Nx.pp c.omega_cb Nx.pp c.omega_k
    Nx.pp c.w0 Nx.pp c.wa Quantity.pp c.t_cmb Nx.pp c.n_eff Quantity.pp c.m_nu

(* Static checks *)

let payload q = Quantity.value (Quantity.unit q) q

let shape_text s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

(* [broadcast a b] is the shape [a] and [b] broadcast to, from the right. *)
let broadcast a b =
  let na = Array.length a and nb = Array.length b in
  let n = max na nb in
  let dim s ns i = if i < n - ns then 1 else s.(i - (n - ns)) in
  let out = Array.make n 0 in
  let rec go i =
    if i = n then Some out
    else
      let x = dim a na i and y = dim b nb i in
      if x = y || y = 1 then (
        out.(i) <- x;
        go (i + 1))
      else if x = 1 then (
        out.(i) <- y;
        go (i + 1))
      else None
  in
  go 0

(* [symbols u] is the part of [u] made of symbols: what is left of a quotient
   that does not convert, as in ["their quotient keeps m"]. *)
let symbols u =
  List.fold_left
    (fun acc (t, num, den) ->
      match t with
      | Unit.Symbol { name; scope } ->
          let s =
            match scope with
            | None -> Unit.symbol name
            | Some scope -> Unit.scoped ~scope name
          in
          Unit.(acc * (root den s ** num))
      | Prime _ | Pi -> acc)
    Unit.one (Unit.terms u)

let check_unit fn field q target =
  let u = Quantity.unit q in
  if not (Unit.convertible u target) then
    invalid_arg
      (strf
         "%s: %s is in %s, which does not convert to %s: their quotient keeps \
          %s"
         fn field (Unit.to_string u) (Unit.to_string target)
         (Unit.to_string (symbols Unit.(u / target))))

let rate = Unit.(one / second)

let check_dtype (type b) fn (d : (float, b) Nx.dtype) =
  match d with
  | Float32 | Float64 -> ()
  | d ->
      invalid_arg
        (strf
           "%s: %s has no rule; the cosmology computes in float32 and float64"
           fn (Nx_dtype.to_string d))

(* [lanes fn c] checks [c] and is the broadcast shape of its leaves, [m_nu]'s
   without its last axis. *)
let lanes (type b) fn (c : (float, b) Nx.t t) =
  let h = payload c.h0 in
  check_dtype fn (Nx.dtype h);
  check_unit fn "h0" c.h0 rate;
  check_unit fn "t_cmb" c.t_cmb Unit.kelvin;
  check_unit fn "m_nu" c.m_nu Unit.joule;
  let m = Nx.shape (payload c.m_nu) in
  let k = Array.length m in
  if k = 0 then
    invalid_arg
      (strf
         "%s: m_nu is a scalar; its last axis lists the massive species ([1] \
          for one, [0] for none)"
         fn);
  let leaves =
    [
      ("h0", Nx.shape h);
      ("omega_cb", Nx.shape c.omega_cb);
      ("omega_k", Nx.shape c.omega_k);
      ("w0", Nx.shape c.w0);
      ("wa", Nx.shape c.wa);
      ("t_cmb", Nx.shape (payload c.t_cmb));
      ("n_eff", Nx.shape c.n_eff);
      ("m_nu", Array.sub m 0 (k - 1));
    ]
  in
  List.fold_left
    (fun acc (name, s) ->
      match broadcast acc s with
      | Some s -> s
      | None ->
          invalid_arg
            (strf
               "%s: the cosmology's leaves do not broadcast: %s is %s where \
                the fields before it broadcast to %s"
               fn name (shape_text s) (shape_text acc)))
    [||] leaves

(* [broadcast_with fn lanes name x] is [lanes] broadcast with the shape of the
   argument [name], [x]. *)
let broadcast_with fn lanes name x =
  let s = Nx.shape x in
  match broadcast lanes s with
  | Some s -> s
  | None ->
      let count = Array.fold_left ( * ) 1 lanes in
      let leaves = Array.append lanes (Array.make (Array.length s) 1) in
      invalid_arg
        (strf
           "%s: the cosmology's lanes %s and %s %s do not broadcast; for every \
            redshift in each of %d lanes, give leaves of shape %s or map with \
            Rune.vmap"
           fn (shape_text lanes) name (shape_text s) count (shape_text leaves))

(* The expansion rate *)

(* The radiation's constants: each massless species adds [c_ur] Omega_g, with
   the neutrinos at (4/11)^(1/3) of the photons' temperature, and each massive
   one [c_nu] Omega_g F(y a), at T_ncdm = 0.71611 of it. A massive species of
   zero mass counts as c_nu / c_ur massless ones, which N_ur subtracts, so

     Omega_r(a) = Omega_g (1 + c_ur N_eff + c_nu sum_i (F(y_i a) - 1)). *)
let t_ncdm = Unit.decimal "0.71611"
let c_ur = Unit.(int 7 / int 8 * (root 3 (int 4 / int 11) ** 4))
let c_nu = Unit.(int 7 / int 8 * (t_ncdm ** 4))

let horner cs t =
  let acc = ref (Nx.full_like t cs.(0)) in
  for i = 1 to Array.length cs - 1 do
    acc := Nx.add_s (Nx.mul !acc t) cs.(i)
  done;
  !acc

(* [chebyshev cs t] is sum_j cs.(j) T_j(t), by Clenshaw's recurrence. *)
let chebyshev cs t =
  let n = Array.length cs in
  let t2 = Nx.mul_s t 2. in
  let b1 = ref (Nx.full_like t cs.(n - 1)) and b2 = ref (Nx.zeros_like t) in
  for j = n - 2 downto 1 do
    let b = Nx.add_s (Nx.sub (Nx.mul t2 !b1) !b2) cs.(j) in
    b2 := !b1;
    b1 := b
  done;
  Nx.add_s (Nx.sub (Nx.mul t !b1) !b2) cs.(0)

(* [neutrino tb y] is F(y) = (120 / 7 pi^4) int_0^inf x^2 sqrt(x^2 + y^2) /
   (e^x + 1) dx for y >= 0, a massive species' energy over its massless value.
   Below [f_series_below] it is the series 1 + y^2 (P(y) + c y^2 log y), whose
   log reads [max y f_log_floor] so that F(0) = 1 and F'(0) = 0 exactly; from
   [f_asymptotic_from] the expansion y Q(1/y^2); between, a Chebyshev series in
   log y. *)
let neutrino (tb : T.t) y =
  let small = Nx.less_equal_s y tb.f_series_below in
  let large = Nx.greater_equal_s y tb.f_asymptotic_from in
  let ys = Nx.minimum_s y tb.f_series_below in
  let ys2 = Nx.mul ys ys in
  let log_ys = Nx.log (Nx.maximum_s ys tb.f_log_floor) in
  let series =
    Nx.add_s
      (Nx.mul ys2
         (Nx.add (horner tb.f_series ys)
            (Nx.mul_s (Nx.mul ys2 log_ys) tb.f_series_log)))
      1.
  in
  let ym =
    Nx.minimum_s (Nx.maximum_s y tb.f_series_below) tb.f_asymptotic_from
  in
  let t = Nx.mul_s (Nx.sub_s (Nx.log ym) tb.f_cheb_mid) tb.f_cheb_scale in
  let middle = chebyshev tb.f_cheb t in
  let yl = Nx.maximum_s y tb.f_asymptotic_from in
  let asymptotic =
    Nx.mul yl (horner tb.f_asymptotic (Nx.recip (Nx.mul yl yl)))
  in
  Nx.where small series (Nx.where large asymptotic middle)

(* [safe_h h] is H0's payload [h] at 1 where it is not positive, and where it
   was: every result is a multiple of 1/H0. *)
let safe_h h =
  let ok = Nx.logical_not (Nx.less_equal_s h 0.) in
  (Nx.where ok h (Nx.ones_like h), ok)

(* A y past [y_cap] is in F's linear regime, where its term is below every
   other's: the cap keeps y finite where T_CMB is subnormal. *)
let y_cap = 0x1p60

(* The radiation's part of a^4 E^2, for one call: Omega_g, c_ur N_eff, c_nu,
   and y = m c^2 / (k_B T_ncdm T_CMB) on the lanes and a last axis of the
   massive species, if there are any. *)
type 'b radiation = {
  tables : T.t;
  omega_g : (float, 'b) Nx.t;
  ur : (float, 'b) Nx.t;
  c_nu : float;
  y : (float, 'b) Nx.t option;
}

(* [neutrinos r a] is the neutrinos' term of a^4 E^2 at [a]. *)
let neutrinos r a =
  match r.y with
  | None -> Nx.mul r.omega_g r.ur
  | Some y ->
      let ya = Nx.mul y (Nx.unsqueeze ~axes:[ Nx.ndim a ] a) in
      let f = Nx.sub_s (neutrino r.tables ya) 1. in
      let massive = Nx.sum ~axes:[ Nx.ndim ya - 1 ] f in
      Nx.mul r.omega_g (Nx.add r.ur (Nx.mul_s massive r.c_nu))

(* A model is a cosmology read for one call: its payloads in their own units,
   with H0 at 1 where it is not positive, and the constants of a^4 E^2. *)
type 'b model = {
  tables : T.t;
  ndim : int; (* the lanes' rank *)
  h : (float, 'b) Nx.t;
  h_ok : (bool, Nx.bool_elt) Nx.t;
  omega_cb : (float, 'b) Nx.t;
  omega_k : (float, 'b) Nx.t;
  omega_de : (float, 'b) Nx.t;
  w0 : (float, 'b) Nx.t;
  wa : (float, 'b) Nx.t;
  radiation : 'b radiation;
}

let model (type b) (c : (float, b) Nx.t t) ~ndim : b model =
  let h = payload c.h0 in
  let d = Nx.dtype h in
  let tables = match d with Float64 -> T.float64 | _ -> T.float32 in
  let h, h_ok = safe_h h in
  let g = Constant.quantity d (Codata.newtonian_gravitation c.codata) in
  (* Omega_g = 32 pi G sigma T^4 / (3 c^3 H0^2): the record's quantities and
     one rounded factor. *)
  let omega_g =
    Quantity.(div (mul g (pow 4 c.t_cmb)) (pow 2 (v (unit c.h0) h)))
    |> Quantity.times
         Unit.(int 32 * pi * stefan_boltzmann / (int 3 * (speed_of_light ** 3)))
    |> Quantity.value Unit.one
  in
  let y =
    let m = payload c.m_nu in
    if (Nx.shape m).(Nx.ndim m - 1) = 0 then None
    else
      (* T_CMB = 0 has no radiation: Omega_g is 0, and y reads T_CMB = 1. *)
      let t = payload c.t_cmb in
      let t = Nx.where (Nx.equal_s t 0.) (Nx.ones_like t) t in
      let t = Nx.unsqueeze ~axes:[ Nx.ndim t ] t in
      let y =
        Quantity.(div c.m_nu (v (unit c.t_cmb) t))
        |> Quantity.per Unit.(boltzmann * t_ncdm)
        |> Quantity.value Unit.one
      in
      Some (Nx.minimum_s (Nx.abs y) y_cap)
  in
  let one u = Unit.ratio d u Unit.one in
  let radiation =
    { tables; omega_g; ur = Nx.mul_s c.n_eff (one c_ur); c_nu = one c_nu; y }
  in
  let today = Nx.add omega_g (neutrinos radiation (Nx.ones_like omega_g)) in
  let omega_de = Nx.sub (Nx.sub (Nx.rsub_s 1. c.omega_cb) c.omega_k) today in
  {
    tables;
    ndim;
    h;
    h_ok;
    omega_cb = c.omega_cb;
    omega_k = c.omega_k;
    omega_de;
    w0 = c.w0;
    wa = c.wa;
    radiation;
  }

(* The components' terms of a^4 E^2 at [a], given [ln a] and [1 - a], which
   each caller computes where it does not cancel. *)
type 'b terms = {
  cold : (float, 'b) Nx.t;
  curvature : (float, 'b) Nx.t;
  dark : (float, 'b) Nx.t;
  photons : (float, 'b) Nx.t;
  neutrinos : (float, 'b) Nx.t;
}

let terms m ~a ~ln_a ~one_minus_a =
  let a2 = Nx.mul a a in
  let w = Nx.mul_s (Nx.add_s (Nx.add m.w0 m.wa) 1.) (-3.) in
  let de =
    Nx.exp (Nx.sub (Nx.mul w ln_a) (Nx.mul_s (Nx.mul m.wa one_minus_a) 3.))
  in
  {
    cold = Nx.mul m.omega_cb a;
    curvature = Nx.mul m.omega_k a2;
    dark = Nx.mul (Nx.mul m.omega_de (Nx.mul a2 a2)) de;
    photons = m.radiation.omega_g;
    neutrinos = neutrinos m.radiation a;
  }

let total t =
  Nx.add
    (Nx.add (Nx.add t.cold t.curvature) t.dark)
    (Nx.add t.photons t.neutrinos)

let a4e2 m ~a ~ln_a ~one_minus_a = total (terms m ~a ~ln_a ~one_minus_a)

(* The domain *)

let nan_where_not ok x = Nx.where ok x (Nx.full_like x Float.nan)

(* [safe_z z] is [z] at 0 where z <= -1, and where it was above. NaN passes, so
   a NaN redshift gives NaN with a NaN derivative. *)
let safe_z z =
  let ok = Nx.logical_not (Nx.less_equal_s z (-1.)) in
  (Nx.where ok z (Nx.zeros_like z), ok)

(* [safe_e2 e2] is [e2] at 1 where it is not positive, and where it was. *)
let safe_e2 e2 =
  let ok = Nx.logical_not (Nx.less_equal_s e2 0.) in
  (Nx.where ok e2 (Nx.ones_like e2), ok)

(* The rules *)

(* [rule m xs] is the table [xs] as a tensor of the node axis, in front of the
   lanes. *)
let rule m xs =
  let n = Array.length xs in
  Nx.create (Nx.dtype m.h) [| n |] xs
  |> Nx.reshape (Array.append [| n |] (Array.make m.ndim 1))

(* The integrals over the line of sight: chi / D_H and H0 t_L. *)
type radial = Distance | Lookback

(* [radial m kind z] is int_0^v_z 2 u^p / sqrt(u^8 E^2(u^2)) dv with u = 1 - v
   and v_z = 1 - (1 + z)^(-1/2), p = 1 for [Distance] and 3 for [Lookback].
   Its second component holds where a^4 E^2 > 0 at every node. *)
let radial m kind z =
  let t = rule m m.tables.distance_nodes in
  let w = rule m m.tables.distance_weights in
  let r = Nx.sqrt (Nx.add_s z 1.) in
  let vz = Nx.div z (Nx.mul r (Nx.add_s r 1.)) in
  let v = Nx.mul vz t in
  let u = Nx.rsub_s 1. v in
  let a = Nx.mul u u in
  let ln_a = Nx.mul_s (Nx.log1p (Nx.neg v)) 2. in
  let one_minus_a = Nx.mul v (Nx.rsub_s 2. v) in
  let e2, ok = safe_e2 (a4e2 m ~a ~ln_a ~one_minus_a) in
  let num = match kind with Distance -> u | Lookback -> Nx.mul u a in
  let f = Nx.div (Nx.mul_s num 2.) (Nx.sqrt e2) in
  (Nx.mul vz (Nx.sum ~axes:[ 0 ] (Nx.mul w f)), Nx.all ~axes:[ 0 ] ok)

(* [since_big_bang m z] is H0 t = int_0^s_z 4 s^7 / sqrt(s^16 E^2(s^4)) ds with
   s_z = (1 + z)^(-1/4), and where a^4 E^2 > 0 at every node. *)
let since_big_bang m z =
  let t = rule m m.tables.age_nodes in
  let w = rule m m.tables.age_weights in
  let sz = Nx.rsqrt (Nx.sqrt (Nx.add_s z 1.)) in
  let s = Nx.mul sz t in
  let s2 = Nx.mul s s in
  let a = Nx.mul s2 s2 in
  let ln_a = Nx.mul_s (Nx.log s) 4. in
  let one_minus_a = Nx.rsub_s 1. a in
  let e2, ok = safe_e2 (a4e2 m ~a ~ln_a ~one_minus_a) in
  let f = Nx.div (Nx.mul_s (Nx.mul (Nx.mul s2 s) a) 4.) (Nx.sqrt e2) in
  (Nx.mul sz (Nx.sum ~axes:[ 0 ] (Nx.mul w f)), Nx.all ~axes:[ 0 ] ok)

(* Curvature *)

(* [sinc tb x] is S(x) = sum_n x^n / (2n+1)!: sinh (sqrt x) / sqrt x above 1,
   sin (sqrt -x) / sqrt -x below -1, its Taylor polynomial between. Each region
   reads its input clamped into it, so its partials are finite everywhere. *)
let sinc (tb : T.t) x =
  let mid = horner tb.sinc_taylor (Nx.minimum_s (Nx.maximum_s x (-1.)) 1.) in
  let rp = Nx.sqrt (Nx.maximum_s x 1.) in
  let rn = Nx.sqrt (Nx.neg (Nx.minimum_s x (-1.))) in
  Nx.where (Nx.greater_s x 1.)
    (Nx.div (Nx.sinh rp) rp)
    (Nx.where (Nx.less_s x (-1.)) (Nx.div (Nx.sin rn) rn) mid)

(* [volume tb x] is W(x) = (S(4x) - 1) / (2x) = sum_(n>=1) 4^n x^(n-1) / (2
   (2n+1)!), W(0) = 1/3, in [sinc]'s three regions. *)
let volume (tb : T.t) x =
  let mid = horner tb.volume_taylor (Nx.minimum_s (Nx.maximum_s x (-1.)) 1.) in
  let xp = Nx.maximum_s x 1. and xn = Nx.minimum_s x (-1.) in
  let rp = Nx.mul_s (Nx.sqrt xp) 2.
  and rn = Nx.mul_s (Nx.sqrt (Nx.neg xn)) 2. in
  let w s r x = Nx.div (Nx.sub_s (Nx.div s r) 1.) (Nx.mul_s x 2.) in
  Nx.where (Nx.greater_s x 1.)
    (w (Nx.sinh rp) rp xp)
    (Nx.where (Nx.less_s x (-1.)) (w (Nx.sin rn) rn xn) mid)

(* Calls *)

(* [call fn c z] checks [c] and [z] and is the lanes' shape and [c]'s model. *)
let call fn c z =
  let lanes = broadcast_with fn (lanes fn c) "z" z in
  (lanes, model c ~ndim:(Array.length lanes))

let result lanes ok x = Nx.broadcast_to lanes (nan_where_not ok x)
let all = List.fold_left Nx.logical_and

(* [at m z] is the terms of a^4 E^2 at [z], a safe redshift. *)
let at m z =
  let zp1 = Nx.add_s z 1. in
  terms m ~a:(Nx.recip zp1)
    ~ln_a:(Nx.neg (Nx.log1p z))
    ~one_minus_a:(Nx.div z zp1)

let distance_unit c = Unit.(speed_of_light / Quantity.unit c.h0)

(* Expansion *)

(* [expansion fn c z] is the lanes, the model, E(z) and where it is defined. *)
let expansion fn c z =
  let lanes, m = call fn c z in
  let z, z_ok = safe_z z in
  let e2, e_ok = safe_e2 (total (at m z)) in
  let zp1 = Nx.add_s z 1. in
  let e = Nx.mul (Nx.sqrt e2) (Nx.mul zp1 zp1) in
  (lanes, m, e, all z_ok [ m.h_ok; e_ok ])

let hubble c z =
  let lanes, m, e, ok = expansion "Cosmology.hubble" c z in
  Quantity.v (Quantity.unit c.h0) (result lanes ok (Nx.mul m.h e))

type component = Cold_matter | Photons | Neutrinos | Dark_energy | Curvature

let density_parameter c i z =
  let lanes, m = call "Cosmology.density_parameter" c z in
  let z, z_ok = safe_z z in
  let t = at m z in
  let e2, e_ok = safe_e2 (total t) in
  let term =
    match i with
    | Cold_matter -> t.cold
    | Photons -> t.photons
    | Neutrinos -> t.neutrinos
    | Dark_energy -> t.dark
    | Curvature -> t.curvature
  in
  result lanes (all z_ok [ m.h_ok; e_ok ]) (Nx.div term e2)

let critical_density c z =
  let fn = "Cosmology.critical_density" in
  let lanes, m, e, ok = expansion fn c z in
  let g =
    Constant.quantity (Nx.dtype m.h) (Codata.newtonian_gravitation c.codata)
  in
  let h = Quantity.v (Quantity.unit c.h0) (result lanes ok (Nx.mul m.h e)) in
  Quantity.(div (pow 2 h) g |> times Unit.(int 3 / (int 8 * pi)))

(* Distances *)

(* [comoving fn c z] is the lanes, the model, chi / D_H at [z] and where it is
   defined, with [z] safe. *)
let comoving fn c z =
  let lanes, m = call fn c z in
  let z, z_ok = safe_z z in
  let chi, nodes_ok = radial m Distance z in
  (lanes, m, z, chi, all z_ok [ m.h_ok; nodes_ok ])

let transverse_hat m chi =
  Nx.mul chi (sinc m.tables (Nx.mul m.omega_k (Nx.mul chi chi)))

let comoving_distance c z =
  let lanes, m, _, chi, ok = comoving "Cosmology.comoving_distance" c z in
  Quantity.v (distance_unit c) (result lanes ok (Nx.div chi m.h))

let transverse (type b) (c : (float, b) Nx.t t) (d : (float, b) Nx.t Quantity.t)
    =
  let fn = "Cosmology.transverse" in
  check_unit fn "d" d Unit.metre;
  let x = payload d in
  let lanes = broadcast_with fn (lanes fn c) "d" x in
  let h = payload c.h0 in
  let tables = match Nx.dtype h with Float64 -> T.float64 | _ -> T.float32 in
  let h, h_ok = safe_h h in
  let d_hat =
    Quantity.(mul d (v (unit c.h0) h))
    |> Quantity.per Unit.speed_of_light
    |> Quantity.value Unit.one
  in
  let s = sinc tables (Nx.mul c.omega_k (Nx.mul d_hat d_hat)) in
  Quantity.v (Quantity.unit d) (result lanes h_ok (Nx.mul x s))

let angular_diameter_distance c z =
  let fn = "Cosmology.angular_diameter_distance" in
  let lanes, m, z, chi, ok = comoving fn c z in
  let d = Nx.div (Nx.div (transverse_hat m chi) m.h) (Nx.add_s z 1.) in
  Quantity.v (distance_unit c) (result lanes ok d)

let luminosity fn c ?observed z =
  let lanes, m, z, chi, ok = comoving fn c z in
  let observed, lanes =
    match observed with
    | None -> (z, lanes)
    | Some o -> (o, broadcast_with fn lanes "observed" o)
  in
  let d = Nx.mul (Nx.div (transverse_hat m chi) m.h) (Nx.add_s observed 1.) in
  Quantity.v (distance_unit c) (result lanes ok d)

let luminosity_distance c ?observed z =
  luminosity "Cosmology.luminosity_distance" c ?observed z

(* 5 / ln 10: a magnitude is 5 log10 of a distance ratio. *)
let magnitude = 5. /. Float.log 10.

let distance_modulus c ?observed z =
  let d = luminosity "Cosmology.distance_modulus" c ?observed z in
  let r =
    Unit.ratio
      (Nx.dtype (payload d))
      (Quantity.unit d)
      Unit.(int 10 * Units.parsec)
  in
  Nx.mul_s (Nx.log (Nx.mul_s (payload d) r)) magnitude

(* Volumes *)

let volume_unit c = Unit.((distance_unit c ** 3) / steradian)

let comoving_volume c z =
  let lanes, m, _, chi, ok = comoving "Cosmology.comoving_volume" c z in
  let chi3 = Nx.mul (Nx.mul chi chi) chi in
  let w = volume m.tables (Nx.mul m.omega_k (Nx.mul chi chi)) in
  let h3 = Nx.mul (Nx.mul m.h m.h) m.h in
  Quantity.v (volume_unit c) (result lanes ok (Nx.div (Nx.mul chi3 w) h3))

let comoving_volume_element c z =
  let fn = "Cosmology.comoving_volume_element" in
  let lanes, m, z, chi, ok = comoving fn c z in
  let e2, e_ok = safe_e2 (total (at m z)) in
  let zp1 = Nx.add_s z 1. in
  let e = Nx.mul (Nx.sqrt e2) (Nx.mul zp1 zp1) in
  let dm = transverse_hat m chi in
  let h3 = Nx.mul (Nx.mul m.h m.h) m.h in
  let v = Nx.div (Nx.mul dm dm) (Nx.mul e h3) in
  Quantity.v (volume_unit c) (result lanes (Nx.logical_and ok e_ok) v)

(* Times *)

let time_unit c = Unit.(one / Quantity.unit c.h0)

let lookback_time c z =
  let lanes, m = call "Cosmology.lookback_time" c z in
  let z, z_ok = safe_z z in
  let t, nodes_ok = radial m Lookback z in
  let ok = all z_ok [ m.h_ok; nodes_ok ] in
  Quantity.v (time_unit c) (result lanes ok (Nx.div t m.h))

let age c z =
  let lanes, m = call "Cosmology.age" c z in
  let z, z_ok = safe_z z in
  let t, nodes_ok = since_big_bang m z in
  let ok = all z_ok [ m.h_ok; nodes_ok ] in
  Quantity.v (time_unit c) (result lanes ok (Nx.div t m.h))

(* Realisations *)

(* [decimal_sum ds] is the exact sum of the positive decimals [ds], in
   [Unit.decimal]'s text: ["0.02242"; "0.11933"] is ["14175e-5"]. *)
let decimal_sum ds =
  let parse s =
    match String.index_opt s '.' with
    | None -> (int_of_string s, 0)
    | Some i ->
        let frac = String.sub s (i + 1) (String.length s - i - 1) in
        (int_of_string (String.sub s 0 i ^ frac), String.length frac)
  in
  let terms = List.map parse ds in
  let scale = List.fold_left (fun acc (_, e) -> max acc e) 0 terms in
  let rec shift m e = if e = 0 then m else shift (m * 10) (e - 1) in
  let sum =
    List.fold_left (fun acc (m, e) -> acc + shift m (scale - e)) 0 terms
  in
  strf "%de-%d" sum scale

(* A paper's flat LambdaCDM fit, as its decimals: H0 in km s^-1 Mpc^-1, the
   physical densities whose sum is omega_cb h^2, T_CMB in K, N_eff and the
   massive species' rest energies in eV. *)
type paper = {
  h0 : string;
  omega_cb_h2 : string list;
  t_cmb : string;
  n_eff : string;
  m_nu : string list;
}

let realise (type b) fn ~codata (d : (float, b) Nx.dtype) p : (float, b) Nx.t t
    =
  check_dtype fn d;
  let number u = Unit.ratio d u Unit.one in
  let scalar s = Nx.scalar d (number (Unit.decimal s)) in
  let h = Unit.(decimal p.h0 / int 100) in
  let omega_cb = Unit.(decimal (decimal_sum p.omega_cb_h2) / (h ** 2)) in
  let m_nu = List.map (fun m -> number (Unit.decimal m)) p.m_nu in
  {
    codata;
    h0 = Quantity.v Unit.(kilo metre / second / mega Units.parsec) (scalar p.h0);
    omega_cb = Nx.scalar d (number omega_cb);
    omega_k = Nx.scalar d 0.;
    w0 = Nx.scalar d (-1.);
    wa = Nx.scalar d 0.;
    t_cmb = Quantity.v Unit.kelvin (scalar p.t_cmb);
    n_eff = scalar p.n_eff;
    m_nu =
      Quantity.v Unit.electronvolt
        (Nx.create d [| List.length m_nu |] (Array.of_list m_nu));
  }

let planck ~h0 ~omega_cb_h2 =
  { h0; omega_cb_h2; t_cmb = "2.7255"; n_eff = "3.046"; m_nu = [ "0.06" ] }

let wmap ~h0 ~omega_cb_h2 =
  { h0; omega_cb_h2; t_cmb = "2.725"; n_eff = "3.04"; m_nu = [] }

let planck2018 ~codata d =
  realise "Cosmology.planck2018" ~codata d
    (planck ~h0:"67.66" ~omega_cb_h2:[ "0.02242"; "0.11933" ])

let planck2015 ~codata d =
  realise "Cosmology.planck2015" ~codata d
    (planck ~h0:"67.74" ~omega_cb_h2:[ "0.02230"; "0.1188" ])

let planck2013 ~codata d =
  realise "Cosmology.planck2013" ~codata d
    (planck ~h0:"67.77" ~omega_cb_h2:[ "0.022161"; "0.11889" ])

let wmap9 ~codata d =
  realise "Cosmology.wmap9" ~codata d
    (wmap ~h0:"69.32" ~omega_cb_h2:[ "0.02223"; "0.1153" ])

let wmap7 ~codata d =
  realise "Cosmology.wmap7" ~codata d
    (wmap ~h0:"70.4" ~omega_cb_h2:[ "0.02253"; "0.1122" ])

let wmap5 ~codata d =
  realise "Cosmology.wmap5" ~codata d
    (wmap ~h0:"70.2" ~omega_cb_h2:[ "0.02262"; "0.1138" ])

let wmap3 ~codata d =
  realise "Cosmology.wmap3" ~codata d
    (wmap ~h0:"70.1" ~omega_cb_h2:[ "0.1349" ])

let wmap1 ~codata d =
  realise "Cosmology.wmap1" ~codata d (wmap ~h0:"72" ~omega_cb_h2:[ "0.133" ])
