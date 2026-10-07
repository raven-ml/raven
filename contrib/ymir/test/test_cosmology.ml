(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The cosmology against its references and its laws.

   Values: every function against mpmath's integrals of the same model
   (golden/cosmology/mpmath.golden) to 2^-46 in float64 and 2^-20 in float32,
   eagerly and compiled; H, distances and times against CLASS (class.golden)
   to 1e-6 and astropy (astropy.golden) to 1e-7, astropy's Planck18 to 5e-5;
   each realisation's Omega_m, Omega_Lambda and age against its paper's
   printed digits. The goldens are written by gen/references.py.

   Laws: the density parameters sum to 1; flat is exact (D_M = D_C bit for
   bit); the curvature derivative at Omega_k = 0 is d^3 / (6 D_H^2); jvp and
   grad agree with central differences in every parameter and in z; a species
   of zero mass counts as massless; broadcast lanes equal vmap's rows; the
   domain's NaN lanes have zero derivatives and leave the others alone. *)

open Windtrap
open Ymir

let mpc = Unit.mega Units.parsec
let gyr = Unit.giga Units.julian_year
let km_s_mpc = Unit.(kilo metre / second / mpc)
let mpc3_sr = Unit.((mpc ** 3) / steradian)
let kg_m3 = Unit.(kilogram / (metre ** 3))

(* The critical density in GeV c^-2 m^-3: in kg m^-3 its conversion factor
   from km s^-1 Mpc^-1 is subnormal in float32. *)
let gev_m3 = Unit.(giga electronvolt / (speed_of_light ** 2) / (metre ** 3))
let kg_per_gev = Unit.ratio Nx.float64 kg_m3 gev_m3
let f64 x = Nx.scalar Nx.float64 x
let item x = Nx.item [] x

(* Models *)

type model = {
  codata : Codata.t;
  h0 : float;
  omega_cb : float;
  omega_k : float;
  w0 : float;
  wa : float;
  t_cmb : float;
  n_eff : float;
  m_nu : float list;
}

let planck_like =
  {
    codata = Codata.v2022;
    h0 = 67.66;
    omega_cb = 0.3096;
    omega_k = 0.;
    w0 = -1.;
    wa = 0.;
    t_cmb = 2.7255;
    n_eff = 3.046;
    m_nu = [ 0.06 ];
  }

let cosmology (type b) (d : (float, b) Nx.dtype) m : (float, b) Nx.t Cosmology.t
    =
  let s x = Nx.scalar d x in
  {
    codata = m.codata;
    h0 = Quantity.v km_s_mpc (s m.h0);
    omega_cb = s m.omega_cb;
    omega_k = s m.omega_k;
    w0 = s m.w0;
    wa = s m.wa;
    t_cmb = Quantity.v Unit.kelvin (s m.t_cmb);
    n_eff = s m.n_eff;
    m_nu =
      Quantity.v Unit.electronvolt
        (Nx.create d [| List.length m.m_nu |] (Array.of_list m.m_nu));
  }

let cosmology_ptree () = Nx.Ptree.instantiate (module Cosmology)

(* Functions by name, as the goldens write them, each in the goldens' unit. *)

let component = function
  | "cold_matter" -> Cosmology.Cold_matter
  | "photons" -> Photons
  | "neutrinos" -> Neutrinos
  | "dark_energy" -> Dark_energy
  | "curvature" -> Curvature
  | s -> failwith ("unknown component " ^ s)

let evaluate (type b) name (c : (float, b) Nx.t Cosmology.t) ?observed
    (z : (float, b) Nx.t) : (float, b) Nx.t =
  match String.split_on_char ':' name with
  | [ "density_parameter"; i ] -> Cosmology.density_parameter c (component i) z
  | _ -> (
      match name with
      | "hubble" -> Cosmology.hubble c z |> Quantity.value km_s_mpc
      | "critical_density" ->
          Cosmology.critical_density c z |> Quantity.value gev_m3
      | "comoving_distance" ->
          Cosmology.comoving_distance c z |> Quantity.value mpc
      | "transverse" ->
          Cosmology.(transverse c (comoving_distance c z)) |> Quantity.value mpc
      | "angular_diameter_distance" ->
          Cosmology.angular_diameter_distance c z |> Quantity.value mpc
      | "luminosity_distance" ->
          Cosmology.luminosity_distance c ?observed z |> Quantity.value mpc
      | "distance_modulus" -> Cosmology.distance_modulus c ?observed z
      | "comoving_volume" ->
          Cosmology.comoving_volume c z |> Quantity.value mpc3_sr
      | "comoving_volume_element" ->
          Cosmology.comoving_volume_element c z |> Quantity.value mpc3_sr
      | "lookback_time" -> Cosmology.lookback_time c z |> Quantity.value gyr
      | "age" -> Cosmology.age c z |> Quantity.value gyr
      | s -> failwith ("unknown function " ^ s))

(* Goldens *)

type row = {
  model : string;
  fn : string;
  z : float;
  observed : float option;
  value : float;
}

let model_of_fields fields =
  let get k = List.assoc k fields in
  let num k = float_of_string (get k) in
  {
    codata = (if get "codata" = "2018" then Codata.v2018 else Codata.v2022);
    h0 = num "h0";
    omega_cb = num "omega_cb";
    omega_k = num "omega_k";
    w0 = num "w0";
    wa = num "wa";
    t_cmb = num "t_cmb";
    n_eff = num "n_eff";
    m_nu =
      (match get "m_nu" with
      | "-" -> []
      | s -> List.map float_of_string (String.split_on_char ',' s));
  }

let read_golden file =
  let ic = open_in (Filename.concat "golden/cosmology" file) in
  let rec loop models rows =
    match input_line ic with
    | exception End_of_file ->
        close_in ic;
        (List.rev models, List.rev rows)
    | line when String.length line = 0 || line.[0] = '#' -> loop models rows
    | line -> (
        match String.split_on_char ' ' line with
        | "model" :: name :: fields ->
            let field f =
              match String.index_opt f '=' with
              | Some i ->
                  ( String.sub f 0 i,
                    String.sub f (i + 1) (String.length f - i - 1) )
              | None -> failwith ("bad field " ^ f)
            in
            loop
              ((name, model_of_fields (List.map field fields)) :: models)
              rows
        | [ model; fn; z; value ] ->
            let z, observed =
              match String.split_on_char '/' z with
              | [ z ] -> (float_of_string z, None)
              | [ z; o ] -> (float_of_string z, Some (float_of_string o))
              | _ -> failwith ("bad redshift " ^ z)
            in
            let value = float_of_string value in
            (* The goldens hold the critical density in kg m^-3. *)
            let value =
              if fn = "critical_density" then value *. kg_per_gev else value
            in
            let row = { model; fn; z; observed; value } in
            loop models (row :: rows)
        | _ -> failwith ("bad line " ^ line))
  in
  loop [] []

let row_name r =
  let z =
    match r.observed with
    | None -> Printf.sprintf "%g" r.z
    | Some o -> Printf.sprintf "%g/%g" r.z o
  in
  Printf.sprintf "%s %s z=%s" r.model r.fn z

(* [check_rows ~rel d models rows] evaluates every row at dtype [d], with the
   model's and the redshift's values rounded to [d]: eagerly, or [~compiled]
   by one compiled function per function name, of the record and z. *)
let check_rows (type b) ?(compiled = false) ~rel (d : (float, b) Nx.dtype)
    models rows =
  let programs = Hashtbl.create 16 in
  let compile fn =
    match Hashtbl.find_opt programs fn with
    | Some f -> f
    | None ->
        let f =
          Rune.jit
            Nx.Ptree.(cosmology_ptree () @-> tensor @-> returns tensor)
            (fun c z -> evaluate fn c z)
        in
        Hashtbl.add programs fn f;
        f
  in
  List.iter
    (fun r ->
      let c = cosmology d (List.assoc r.model models) in
      let z = Nx.scalar d r.z in
      let got =
        match r.observed with
        | Some o -> evaluate r.fn c ~observed:(Nx.scalar d o) z
        | None -> if compiled then compile r.fn c z else evaluate r.fn c z
      in
      (* A density parameter is a fraction of the budget, within the bound of
         it: a component a thousand times below the others has its exp's
         rounding of the dark energy's growth, relative to itself. *)
      let abs = if String.starts_with ~prefix:"density" r.fn then rel else 0. in
      equal ~msg:(row_name r) (float_rel ~rel ~abs) r.value
        (Nx.item [] (Nx.cast Nx.float64 got)))
    rows

let mpmath =
  let models, rows = read_golden "mpmath.golden" in
  let f32 r = String.ends_with ~suffix:"/f32" r.model in
  let rows64 = List.filter (fun r -> not (f32 r)) rows in
  let rows32 = List.filter f32 rows in
  group "mpmath"
    [
      test "every function in float64, to 2^-46" (fun () ->
          check_rows ~rel:0x1p-46 Nx.float64 models rows64);
      test "every function in float32, to 2^-20" (fun () ->
          check_rows ~rel:0x1p-20 Nx.float32 models rows32);
      test "every function compiled, in float64, to 2^-46" (fun () ->
          check_rows ~compiled:true ~rel:0x1p-46 Nx.float64 models rows64);
    ]

let class_ =
  let models, rows = read_golden "class.golden" in
  test "CLASS: H, distances and times to 1e-6" (fun () ->
      check_rows ~rel:1e-6 Nx.float64 models rows)

let astropy =
  let models, rows = read_golden "astropy.golden" in
  let planck r = r.model = "planck2018" in
  group "astropy"
    [
      test "models without massive neutrinos, to 1e-7" (fun () ->
          check_rows ~rel:1e-7 Nx.float64 models
            (List.filter (fun r -> not (planck r)) rows));
      test "Planck18 against planck2018, to 5e-5" (fun () ->
          (* astropy's Om0 subtracts its neutrino fit from the paper's Omega_m,
             which puts its distances 1.5e-5 from the transcription at z = 1
             and 3.5e-5 at z = 1000, and its ages up to 4.6e-5. *)
          let c = Cosmology.planck2018 ~codata:Codata.v2018 Nx.float64 in
          List.iter
            (fun r ->
              if planck r then
                equal ~msg:(row_name r)
                  (float_rel ~rel:5e-5 ~abs:0.)
                  r.value
                  (item (evaluate r.fn c (f64 r.z))))
            rows);
    ]

(* The papers: each realisation's Omega_m, Omega_Lambda and age today against
   the paper's printed value, within one unit of its last digit. A paper's
   posterior mean of a derived quantity is not that quantity at the mean
   parameters the realisation transcribes, so three values, marked with
   their printed 68% half-width, hold only to it: Planck 2015's age, WMAP
   nine-year's Omega_Lambda and WMAP three-year's Omega_m. *)

type printed = Omega_m | Omega_lambda | Age

let papers =
  let digit s =
    match String.index_opt s '.' with
    | None -> 1.
    | Some i -> 10. ** -.Float.of_int (String.length s - i - 1)
  in
  let last s = (s, digit s) and sigma s w = (s, w) in
  let realisations =
    [
      ( "planck2018",
        Cosmology.planck2018,
        [
          (Omega_m, last "0.3111");
          (Omega_lambda, last "0.6889");
          (Age, last "13.787");
        ] );
      ( "planck2015",
        Cosmology.planck2015,
        [
          (Omega_m, last "0.3089");
          (Omega_lambda, last "0.6911");
          (Age, sigma "13.799" 0.021);
        ] );
      ( "planck2013",
        Cosmology.planck2013,
        [ (Omega_lambda, last "0.6914"); (Age, last "13.7965") ] );
      ( "wmap9",
        Cosmology.wmap9,
        [ (Omega_lambda, sigma "0.7135" 0.0096); (Age, last "13.772") ] );
      ( "wmap7",
        Cosmology.wmap7,
        [ (Omega_lambda, last "0.728"); (Age, last "13.76") ] );
      ( "wmap5",
        Cosmology.wmap5,
        [ (Omega_lambda, last "0.723"); (Age, last "13.72") ] );
      ("wmap3", Cosmology.wmap3, [ (Omega_m, sigma "0.276" 0.026) ]);
    ]
  in
  cases
    ~name:(fun (n, _, _) -> n)
    "papers" realisations
    (fun (_, realise, values) ->
      let c = realise ~codata:Codata.v2022 Nx.float64 in
      let z = f64 0. in
      let omega i = item (Cosmology.density_parameter c i z) in
      List.iter
        (fun (what, (s, tol)) ->
          let msg, got =
            match what with
            | Omega_m -> ("Omega_m", omega Cold_matter +. omega Neutrinos)
            | Omega_lambda -> ("Omega_Lambda", omega Dark_energy)
            | Age -> ("age", item (Quantity.value gyr (Cosmology.age c z)))
          in
          equal ~msg (float tol) (float_of_string s) got)
        values)

(* Laws *)

let models =
  let open Gen in
  let+ omega_cb = float_range 0.1 1.
  and+ omega_k = float_range (-0.3) 0.3
  and+ w0 = float_range (-1.5) (-0.5)
  and+ wa = float_range (-1.) 0.5
  and+ h0 = float_range 40. 100.
  and+ t_cmb = one_of [ constant 0.; float_range 1. 3. ]
  and+ n_eff = float_range 0. 5.
  and+ m_nu = list ~size:(int_range 0 3) (float_range 0. 1.) in
  { planck_like with omega_cb; omega_k; w0; wa; h0; t_cmb; n_eff; m_nu }

let redshifts = Gen.(one_of [ float_range 0. 3.; float_range 3. 1100. ])

let laws =
  group "laws"
    [
      prop "the density parameters sum to 1"
        Gen.(pair models redshifts)
        (fun (m, z) ->
          let c = cosmology Nx.float64 m in
          let sum =
            List.fold_left
              (fun acc i ->
                acc +. item (Cosmology.density_parameter c i (f64 z)))
              0.
              Cosmology.
                [ Cold_matter; Photons; Neutrinos; Dark_energy; Curvature ]
          in
          equal (float 1e-14) 1. sum);
      prop "flat: D_M is D_C bit for bit"
        Gen.(pair models redshifts)
        (fun (m, z) ->
          let c = cosmology Nx.float64 { m with omega_k = 0. } in
          let chi = Cosmology.comoving_distance c (f64 z) in
          let dm = Cosmology.transverse c chi in
          equal float_exact
            (item (Quantity.value mpc chi))
            (item (Quantity.value mpc dm)));
      prop "a species of zero mass counts as massless"
        Gen.(pair models redshifts)
        (fun (m, z) ->
          let massless = cosmology Nx.float64 { m with m_nu = [] } in
          let zero = cosmology Nx.float64 { m with m_nu = [ 0. ] } in
          let f c =
            item (Quantity.value mpc (Cosmology.comoving_distance c (f64 z)))
          in
          equal (float_rel ~rel:1e-14 ~abs:0.) (f massless) (f zero));
      test "hubble at z = 0 is h0" (fun () ->
          let c = cosmology Nx.float64 planck_like in
          equal
            (float_rel ~rel:0x1p-52 ~abs:0.)
            67.66
            (item (Quantity.value km_s_mpc (Cosmology.hubble c (f64 0.)))));
    ]

(* Derivatives *)

(* theta = (h0, omega_cb, omega_k, w0, wa, t_cmb, n_eff, m, z), bare, against
   one function's value. *)
let names =
  [| "h0"; "omega_cb"; "omega_k"; "w0"; "wa"; "t_cmb"; "n_eff"; "m"; "z" |]

let of_theta theta : (float, Nx.float64_elt) Nx.t Cosmology.t * _ =
  let at i = Nx.get [ i ] theta in
  ( {
      codata = Codata.v2022;
      h0 = Quantity.v km_s_mpc (at 0);
      omega_cb = at 1;
      omega_k = at 2;
      w0 = at 3;
      wa = at 4;
      t_cmb = Quantity.v Unit.kelvin (at 5);
      n_eff = at 6;
      m_nu = Quantity.v Unit.electronvolt (Nx.reshape [| 1 |] (at 7));
    },
    at 8 )

let functions =
  [
    "hubble";
    "density_parameter:neutrinos";
    "critical_density";
    "comoving_distance";
    "angular_diameter_distance";
    "distance_modulus";
    "comoving_volume";
    "comoving_volume_element";
    "lookback_time";
    "age";
  ]

let theta_of m z =
  let mass = match m.m_nu with [] -> 0. | x :: _ -> x in
  Nx.create Nx.float64 [| 9 |]
    [| m.h0; m.omega_cb; m.omega_k; m.w0; m.wa; m.t_cmb; m.n_eff; mass; z |]

let value fn theta =
  let c, z = of_theta theta in
  evaluate fn c z

(* [central fn theta i] is the derivative of [fn] along [theta]'s [i]-th
   coordinate from central differences at steps e and e/2, e = 1e-3 of the
   coordinate, or of 0.05 nearer 0, combined by Richardson's extrapolation:
   its error is O(e^4) relative to the coordinate's scale, and the
   evaluation's own error, about 2^-46 of the value, counts once per e. *)
let central fn theta i =
  let x = Nx.to_array theta in
  let at s =
    let y = Array.copy x in
    y.(i) <- y.(i) +. s;
    item (value fn (Nx.create Nx.float64 [| 9 |] y))
  in
  let d e = (at e -. at (-.e)) /. (2. *. e) in
  let e = 1e-3 *. Float.max 0.05 (Float.abs x.(i)) in
  ((4. *. d (e /. 2.)) -. d e) /. 3.

let derivatives =
  let point = Gen.(pair models (float_range 0.05 20.)) in
  let point =
    Gen.map
      (fun (m, z) ->
        let m = { m with t_cmb = (if m.t_cmb = 0. then 2. else m.t_cmb) } in
        theta_of
          { m with m_nu = (match m.m_nu with [] -> [ 0.1 ] | l -> l) }
          z)
      point
  in
  (* A point where the universe has no past is the domain's, with its own
     tests. *)
  let defined theta =
    List.for_all (fun fn -> Float.is_finite (item (value fn theta))) functions
  in
  let close fn theta i got =
    let fd = central fn theta i in
    let scale = Float.abs (item (value fn theta)) in
    equal
      ~msg:(Printf.sprintf "%s d/d%s" fn names.(i))
      (float_rel ~rel:1e-7 ~abs:(Float.max (1e-9 *. scale) 1e-300))
      fd got
  in
  group "derivatives"
    [
      prop ~count:20 "grad agrees with central differences" point (fun theta ->
          assume (defined theta);
          List.iter
            (fun fn ->
              let g = Rune.grad' (fun t -> value fn t) theta in
              Array.iteri (fun i gi -> close fn theta i gi) (Nx.to_array g))
            functions);
      prop ~count:20 "jvp agrees with central differences" point (fun theta ->
          assume (defined theta);
          List.iter
            (fun fn ->
              for i = 0 to 8 do
                let e =
                  Nx.init Nx.float64 [| 9 |] (fun j ->
                      if j.(0) = i then 1. else 0.)
                in
                let _, dy = Rune.jvp' (fun t -> value fn t) theta e in
                close fn theta i (item dy)
              done)
            functions);
      test "a compiled gradient equals the eager one" (fun () ->
          let theta = theta_of planck_like 1.5 in
          let g = Rune.grad' (value "distance_modulus") in
          equal
            (array (float_rel ~rel:1e-12 ~abs:1e-300))
            (Nx.to_array (g theta))
            (Nx.to_array (Rune.jit' g theta)));
      test "the curvature term at Omega_k = 0 is d^3 / (6 D_H^2)" (fun () ->
          let c = cosmology Nx.float64 planck_like in
          let d = Quantity.v mpc (f64 3000.) in
          let f ok =
            Quantity.value mpc (Cosmology.transverse { c with omega_k = ok } d)
          in
          let d_h = 299792.458 /. 67.66 in
          let expected = (3000. ** 3.) /. (6. *. d_h *. d_h) in
          let tolerance = float_rel ~rel:1e-13 ~abs:0. in
          equal ~msg:"grad" tolerance expected (item (Rune.grad' f (f64 0.)));
          equal ~msg:"jvp" tolerance expected
            (item (snd (Rune.jvp' f (f64 0.) (f64 1.)))));
    ]

(* The structure *)

let structure =
  group "structure"
    [
      test "walk reports the release, then each field" (fun () ->
          let c = cosmology Nx.float64 planck_like in
          let visits =
            List.map
              (Format.asprintf "%a" Nx.Ptree.pp_visit)
              (Nx.Ptree.visits (cosmology_ptree ()) c)
          in
          equal (list string)
            [
              "codata: int 2022";
              "h0: case \"1/96939420213600000000 pi s^-1\"";
              "h0: a leaf";
              "omega_cb: a leaf";
              "omega_k: a leaf";
              "w0: a leaf";
              "wa: a leaf";
              "t_cmb: case \"K\"";
              "t_cmb: a leaf";
              "n_eff: a leaf";
              "m_nu: case \"1602176634e-28 kg m^2 s^-2\"";
              "m_nu: a leaf";
            ]
            visits);
      test "a compiled function of the record compiles its leaves" (fun () ->
          let f =
            Rune.jit
              Nx.Ptree.(cosmology_ptree () @-> tensor @-> returns tensor)
              (fun c z -> Quantity.value mpc (Cosmology.comoving_distance c z))
          in
          let z = f64 1. in
          let at m =
            let c = cosmology Nx.float64 m in
            ( item (f c z),
              item (Quantity.value mpc (Cosmology.comoving_distance c z)) )
          in
          List.iter
            (fun m ->
              let compiled, eager = at m in
              equal (float_rel ~rel:0x1p-50 ~abs:0.) eager compiled)
            [ planck_like; { planck_like with omega_k = 0.2; w0 = -0.8 } ]);
    ]

(* Batching *)

let batching =
  test "leaves of shape [k; 1] give vmap's rows" (fun () ->
      let c = cosmology Nx.float64 planck_like in
      let omega_k = Nx.create Nx.float64 [| 3 |] [| -0.2; 0.; 0.3 |] in
      let z = Nx.create Nx.float64 [| 4 |] [| 0.1; 1.; 10.; 1000. |] in
      let broadcast =
        Cosmology.comoving_distance
          { c with omega_k = Nx.reshape [| 3; 1 |] omega_k }
          z
        |> Quantity.value mpc
      in
      let mapped =
        Rune.vmap'
          (fun ok ->
            Cosmology.comoving_distance { c with omega_k = ok } z
            |> Quantity.value mpc)
          omega_k
      in
      equal (array float_exact) (Nx.to_array mapped) (Nx.to_array broadcast))

(* The domain *)

let domain =
  let c = cosmology Nx.float64 planck_like in
  (* Omega_m = 0.1, Omega_Lambda = 2: a^4 E^2 < 0 near a = 0.2, no big bang. *)
  let bounce =
    cosmology Nx.float64
      { planck_like with omega_cb = 0.1; omega_k = -1.1; t_cmb = 0. }
  in
  let dist c z = Cosmology.comoving_distance c z |> Quantity.value mpc in
  let at f z = item (f (f64 z)) in
  (* [lane_derivative f c z] is d f(c, z) / d omega_k by reverse mode. *)
  let lane_derivative f c z =
    let _, pullback =
      Rune.vjp'
        (fun ok -> f { c with Cosmology.omega_k = ok } (f64 z))
        c.omega_k
    in
    item (pullback (f64 1.))
  in
  let nan = float_exact in
  group "domain"
    [
      test "z <= -1 is NaN with a zero derivative; other lanes untouched"
        (fun () ->
          let lanes z = Nx.to_array (dist c (Nx.create Nx.float64 [| 4 |] z)) in
          let valid = lanes [| 1.; 2.; 3.; 0.5 |] in
          let expected = [| valid.(0); Float.nan; Float.nan; valid.(3) |] in
          equal (array nan) expected (lanes [| 1.; -1.; -2.; 0.5 |]);
          equal nan 0. (lane_derivative dist c (-2.)));
      test "a universe with no big bang" (fun () ->
          let age c z = Quantity.value gyr (Cosmology.age c z) in
          let hubble c z = Quantity.value km_s_mpc (Cosmology.hubble c z) in
          greater float_exact ~than:0. (at (dist bounce) 0.1);
          equal nan Float.nan (at (dist bounce) 10.);
          equal nan 0. (lane_derivative dist bounce 10.);
          equal nan Float.nan (at (age bounce) 0.);
          equal nan Float.nan (at (hubble bounce) 4.);
          equal nan 0. (lane_derivative hubble bounce 4.));
      test "H0 <= 0 is NaN" (fun () ->
          let c = { c with h0 = Quantity.v km_s_mpc (f64 (-70.)) } in
          let transverse c z =
            Quantity.value mpc (Cosmology.transverse c (Quantity.v mpc z))
          in
          equal nan Float.nan (at (dist c) 1.);
          equal nan Float.nan (at (transverse c) 1.));
      test "a NaN parameter has a NaN derivative" (fun () ->
          let f ok = dist { c with omega_k = ok } (f64 1.) in
          equal nan Float.nan (item (Rune.grad' f (f64 Float.nan))));
      test "distance_modulus at z = 0 is -infinity" (fun () ->
          equal float_exact Float.neg_infinity
            (item (Cosmology.distance_modulus c (f64 0.))));
    ]

(* Errors *)

let errors =
  let c = cosmology Nx.float64 planck_like in
  group "errors"
    [
      test "lanes that do not broadcast" (fun () ->
          let c = { c with omega_cb = Nx.full Nx.float64 [| 4 |] 0.3 } in
          raises
            (Invalid_argument
               "Cosmology.distance_modulus: the cosmology's lanes [4] and z \
                [1590] do not broadcast; for every redshift in each of 4 \
                lanes, give leaves of shape [4; 1] or map with Rune.vmap")
            (fun () ->
              Cosmology.distance_modulus c (Nx.zeros Nx.float64 [| 1590 |])));
      test "leaves that do not broadcast" (fun () ->
          let c =
            {
              c with
              omega_cb = Nx.full Nx.float64 [| 4 |] 0.3;
              omega_k = Nx.zeros Nx.float64 [| 3 |];
            }
          in
          raises
            (Invalid_argument
               "Cosmology.hubble: the cosmology's leaves do not broadcast: \
                omega_k is [3] where the fields before it broadcast to [4]")
            (fun () -> Cosmology.hubble c (f64 0.)));
      test "h0 not a rate" (fun () ->
          let c =
            { c with h0 = Quantity.v Unit.(kilo metre / second) (f64 70.) }
          in
          raises
            (Invalid_argument
               "Cosmology.age: h0 is in 1e3 m s^-1, which does not convert to \
                s^-1: their quotient keeps m") (fun () ->
              Cosmology.age c (f64 0.)));
      test "a scalar m_nu" (fun () ->
          let c = { c with m_nu = Quantity.v Unit.electronvolt (f64 0.06) } in
          raises
            (Invalid_argument
               "Cosmology.comoving_distance: m_nu is a scalar; its last axis \
                lists the massive species ([1] for one, [0] for none)")
            (fun () -> Cosmology.comoving_distance c (f64 1.)));
      test "float16" (fun () ->
          let c = cosmology Nx.float16 planck_like in
          raises
            (Invalid_argument
               "Cosmology.hubble: float16 has no rule; the cosmology computes \
                in float32 and float64") (fun () ->
              Cosmology.hubble c (Nx.scalar Nx.float16 0.)));
    ]

let () =
  exit
    (run "Cosmology"
       [
         mpmath;
         class_;
         astropy;
         papers;
         structure;
         laws;
         derivatives;
         batching;
         domain;
         errors;
       ])
