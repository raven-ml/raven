(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The 1999 supernova cosmology result redone on Pantheon+.

     pantheon.exe DIR

   DIR holds Pantheon+SH0ES.dat and Pantheon+SH0ES_STAT+SYS.cov from the
   Pantheon+ data release; run.sh fetches them by digest and runs this. The
   1590 supernovae with z_HD > 0.01 are fitted with their STAT+SYS covariance,
   z_HD placing each and z_HEL the redshift observed, for Omega_m and
   Omega_Lambda with curvature free, and for Omega_m in a flat universe.

   Each answer is checked against two references. The same likelihood's
   maximum, from scipy's least squares over quadrature distances
   (reference.py), to 1e-5. And Brout et al. 2022 (ApJ 938, 110), Section 6:
   its values are posterior means from nested sampling where these are the
   likelihood's maximum, so each must lie within the published standard
   deviation, and each standard deviation, here from (J^T J)^-1, within 10% of
   the published one. The run exits 1 if a check fails. *)

open Pantheon_fit

let f64 = Nx.float64

(* Reading the release *)

let words line = List.filter (( <> ) "") (String.split_on_char ' ' line)

let read_lines path =
  In_channel.with_open_text path (fun ic -> In_channel.input_lines ic)

(* [column header name] is the position of [name] among [header]'s words. *)
let column header name =
  let rec find i = function
    | [] -> failwith ("Pantheon+SH0ES.dat has no column " ^ name)
    | w :: _ when w = name -> i
    | _ :: rest -> find (i + 1) rest
  in
  find 0 header

let z_cut = 0.01

(* [read dir] is the data of the supernovae above [z_cut]. *)
let read dir =
  match read_lines (Filename.concat dir "Pantheon+SH0ES.dat") with
  | [] -> failwith "Pantheon+SH0ES.dat is empty"
  | header :: rows ->
      let header = words header in
      let z_hd = column header "zHD" and z_hel = column header "zHEL" in
      let m_b = column header "m_b_corr" in
      let rows =
        Array.of_list (List.map (fun r -> Array.of_list (words r)) rows)
      in
      let at r i = float_of_string r.(i) in
      let kept =
        List.filter
          (fun i -> at rows.(i) z_hd > z_cut)
          (List.init (Array.length rows) Fun.id)
        |> Array.of_list
      in
      let n = Array.length kept in
      let vector col =
        Nx.init f64 [| n |] (fun i -> at rows.(kept.(i.(0))) col)
      in
      let cov =
        match
          read_lines (Filename.concat dir "Pantheon+SH0ES_STAT+SYS.cov")
        with
        | [] -> failwith "Pantheon+SH0ES_STAT+SYS.cov is empty"
        | size :: values ->
            let size = int_of_string (String.trim size) in
            if size <> Array.length rows then
              failwith "the covariance's size is not the table's";
            Array.of_list
              (List.map (fun v -> float_of_string (String.trim v)) values)
      in
      let total = Array.length rows in
      let c =
        Nx.init f64 [| n; n |] (fun ij ->
            cov.((kept.(ij.(0)) * total) + kept.(ij.(1))))
      in
      {
        Fit.z_hd = vector z_hd;
        z_hel = vector z_hel;
        m_b = vector m_b;
        chol = Nx.cholesky c;
      }

(* The references *)

type value = { name : string; index : int }

(* Brout et al. 2022, Section 6 and Table 3: the Pantheon+ supernovae alone,
   with statistical and systematic uncertainties. *)
let published = function
  | Fit.Curved -> [ (0.306, 0.057); (0.625, 0.084) ]
  | Flat -> [ (0.334, 0.018) ]

(* The likelihood's maximum and (J^T J)^-1's standard deviations, by
   reference.py with scipy 1.15.1. *)
let maximum = function
  | Fit.Curved -> [ (0.29539444, 0.05423431); (0.6129948, 0.08098029) ]
  | Flat -> [ (0.33157577, 0.01812654) ]

let values = function
  | Fit.Curved ->
      [ { name = "Omega_m"; index = 0 }; { name = "Omega_Lambda"; index = 1 } ]
  | Flat -> [ { name = "Omega_m"; index = 0 } ]

let start = function
  | Fit.Curved -> Nx.create f64 [| 3 |] [| 0.3; 0.7; -19.3 |]
  | Flat -> Nx.create f64 [| 2 |] [| 0.3; -19.3 |]

(* The fit *)

let failures = ref 0

let check what ok =
  if not ok then (
    incr failures;
    Printf.printf "  FAIL %s\n" what)

let fit data (label, model) =
  let theta = Jera.Solution.get (Fit.fit data model (start model)) in
  let cov = Fit.covariance data model theta in
  let n = (Nx.shape data.Fit.m_b).(0) in
  let p = (Nx.shape theta).(0) in
  Printf.printf "%s: chi2 %.2f for %d degrees of freedom\n" label
    (Fit.chi2 data model theta)
    (n - p);
  List.iter2
    (fun v ((pub, pub_sigma), (ref_value, ref_sigma)) ->
      let x = Nx.item [ v.index ] theta in
      let sigma = Float.sqrt (Nx.item [ v.index; v.index ] cov) in
      Printf.printf
        "  %-12s %.6f +- %.6f   maximum %.6f +- %.6f   Brout 2022 %.3f +- %.3f\n"
        v.name x sigma ref_value ref_sigma pub pub_sigma;
      check
        (v.name ^ " is not the likelihood's maximum")
        (Float.abs (x -. ref_value) <= 1e-5);
      check
        (v.name ^ " is outside Brout 2022's standard deviation")
        (Float.abs (x -. pub) <= pub_sigma);
      check
        (v.name ^ "'s deviation is not within 10% of Brout 2022's")
        (Float.abs (sigma -. pub_sigma) <= 0.1 *. pub_sigma))
    (values model)
    (List.combine (published model) (maximum model))

let () =
  match Sys.argv with
  | [| _; dir |] ->
      let data = read dir in
      Printf.printf "Pantheon+: %d supernovae with z_HD > %g, STAT+SYS\n"
        (Nx.shape data.m_b).(0)
        z_cut;
      List.iter (fit data)
        [ ("LambdaCDM", Fit.Curved); ("flat LambdaCDM", Flat) ];
      exit (if !failures = 0 then 0 else 1)
  | _ ->
      prerr_endline "usage: pantheon.exe DIR";
      exit 2
