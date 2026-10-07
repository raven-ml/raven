(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The 1999 supernova cosmology result on Pantheon+: Omega_m and Omega_Lambda
   fit freely, closed models allowed, against the distances of the data release
   with their full STAT+SYS covariance, by Levenberg-Marquardt, with the fit's
   covariance propagated from the data through the derivative of the answer.

   Two likelihoods, each curvature free and flat. The calibrated one is that of
   every row of Brout et al. 2022's Table 3: the supernovae with z_HD > 0.01 and
   the Cepheid calibrators, whose distance modulus is CEPH_DIST, against
   m_b_corr - M with H0 and M free. The supernova-only one drops the calibrators
   and fits an offset that absorbs M and H0, which supernovae alone do not
   separate; it has no published counterpart and is reported.

   It prints a table and exits 1 when a calibrated fit misses a goal post. Its
   parameters must lie within 1e-4 of their error of scipy's fit of the same
   likelihood (support/pantheon_reference.ml), its Fisher errors within 1e-4
   relative and its chi^2 within 1e-9. Its values must lie within a quarter of
   the paper's error of the paper's, and its errors within 10% of the paper's:
   the paper prints posterior means and 68% limits of a sampled posterior, which
   differ from a maximum and its propagated error by a fraction of the error. *)

open Ymir
module Sn = Ymir_test.Supernova
module Reference = Ymir_test.Pantheon_reference

let failed = ref false

(* Data *)

let fetch dir name url blake2b =
  let path = Filename.concat dir name in
  if not (Sys.file_exists path) then begin
    let part = path ^ ".part" in
    Printf.eprintf "downloading %s to %s\n%!" url path;
    let curl =
      Printf.sprintf "mkdir -p %s && curl -fsSL -o %s %s" (Filename.quote dir)
        (Filename.quote part) (Filename.quote url)
    in
    if Sys.command curl <> 0 then failwith ("could not download " ^ url);
    Sys.rename part path
  end;
  let digest = Digest.BLAKE256.(to_hex (file path)) in
  if digest <> blake2b then
    failwith
      (Printf.sprintf "%s: BLAKE2b-256 %s, expected %s" path digest blake2b);
  path

let words line = String.split_on_char ' ' line |> List.filter (( <> ) "")

type row = {
  z_hd : float;
  z_hel : float;
  m_b : float;
  ceph : float;
  calibrator : bool;
}

let rows path =
  let lines = In_channel.with_open_text path In_channel.input_lines in
  let header = Array.of_list (words (List.hd lines)) in
  let column name =
    match Array.find_index (( = ) name) header with
    | Some i -> i
    | None -> failwith (path ^ ": no column " ^ name)
  in
  let z_hd = column "zHD" and z_hel = column "zHEL" in
  let m_b = column "m_b_corr" and ceph = column "CEPH_DIST" in
  let calibrator = column "IS_CALIBRATOR" in
  List.tl lines
  |> List.map (fun line ->
      let w = Array.of_list (words line) in
      let f i = float_of_string w.(i) in
      {
        z_hd = f z_hd;
        z_hel = f z_hel;
        m_b = f m_b;
        ceph = f ceph;
        calibrator = w.(calibrator) = "1";
      })
  |> Array.of_list

(* The first line is the size n, then the n^2 entries in C order. *)
let covariance path =
  let lines = In_channel.with_open_text path In_channel.input_lines in
  let n = int_of_string (String.trim (List.hd lines)) in
  let c = Array.of_list (List.map float_of_string (List.tl lines)) in
  if Array.length c <> n * n then failwith (path ^ ": not n^2 entries");
  (n, c)

(* The selected rows and their covariance's Cholesky factor. *)
type sample = {
  z_hd : Sn.vector;
  z_hel : Sn.vector;
  m_b : Sn.vector;
  ceph : Sn.vector;
  calibrator : (bool, Nx.bool_elt) Nx.t;
  chol : Sn.vector;
}

let sample (rows : row array) (n, c) keep =
  let index =
    Array.of_list
      (List.filter
         (fun i -> keep rows.(i))
         (List.init (Array.length rows) Fun.id))
  in
  let k = Array.length index in
  let column f = Nx.init Nx.float64 [| k |] (fun i -> f rows.(index.(i.(0)))) in
  let cov =
    Nx.init Nx.float64 [| k; k |] (fun ij ->
        c.((index.(ij.(0)) * n) + index.(ij.(1))))
  in
  {
    z_hd = column (fun r -> r.z_hd);
    z_hel = column (fun r -> r.z_hel);
    m_b = column (fun r -> r.m_b);
    ceph = column (fun r -> r.ceph);
    calibrator =
      Nx.init Nx.bool [| k |] (fun i -> rows.(index.(i.(0))).calibrator);
    chol = Nx.cholesky cov;
  }

(* Models *)

let f64 = Sn.f64
let vector xs = Nx.create Nx.float64 [| Array.length xs |] xs

(* theta = (Omega_m, Omega_Lambda, offset) or, flat, (Omega_m, offset). *)
let sn ~flat s theta =
  let at i = Nx.get [ i ] theta in
  let omega_m = at 0 in
  let omega_l = if flat then Nx.sub (f64 1.) omega_m else at 1 in
  let offset = at (if flat then 1 else 2) in
  let c = Sn.lcdm ~omega_m ~omega_l () in
  Nx.add (Cosmology.distance_modulus c ~observed:s.z_hel s.z_hd) offset

(* theta = (Omega_m, Omega_Lambda, H0, M) or, flat, (Omega_m, H0, M), H0 in km
   s^-1 Mpc^-1. *)
let shoes ~flat s theta =
  let at i = Nx.get [ i ] theta in
  let omega_m = at 0 in
  let omega_l = if flat then Nx.sub (f64 1.) omega_m else at 1 in
  let h0 = at (if flat then 1 else 2) and m = at (if flat then 2 else 3) in
  let c = Sn.lcdm ~h0 ~omega_m ~omega_l () in
  let mu = Cosmology.distance_modulus c ~observed:s.z_hel s.z_hd in
  Nx.add (Nx.where s.calibrator s.ceph mu) m

(* What a fit is held to: a published counterpart, which gates it, or none, and
   it is reported. A counterpart lists the value and error of the fit's first
   parameters. *)
type counterpart = Reported | Published of (float * float) array

(* Brout et al. 2022, ApJ 938, 110, Table 3, Pantheon+ & SH0ES: means and 68%
   half-widths of Omega_m, Omega_Lambda and H0. *)
let brout_lcdm = Published [| (0.306, 0.057); (0.625, 0.084); (73.4, 1.1) |]
let brout_flat = Published [| (0.334, 0.018); (73.6, 1.1) |]

(* Report *)

let run ~title ~names ~model ~start ~(reference : Reference.fit) ~against s =
  let check what ok =
    match against with
    | Published _ when not ok ->
        failed := true;
        Printf.printf "  FAIL %s\n" what
    | Published _ | Reported -> ()
  in
  let published = match against with Published p -> p | Reported -> [||] in
  let t = Unix.gettimeofday () in
  let fit = Sn.fit ~chol:s.chol ~model:(model s) s.m_b (vector start) in
  let seconds = Unix.gettimeofday () -. t in
  Printf.printf "\n%s\n  N = %d, chi2 = %.4f, %d evaluations, %.1f s\n" title
    (Nx.dim 0 s.m_b) fit.chi2 fit.evaluations seconds;
  Printf.printf "  %-12s %10s %9s %9s   %10s %9s   %16s\n" "" "ymir" "sigma"
    "Fisher" "scipy" "sigma" "Brout 2022";
  Array.iteri
    (fun i name ->
      let theta = Nx.item [ i ] fit.theta in
      let sigma = Sn.sigma fit.cov i and fisher = Sn.sigma fit.fisher i in
      let paper =
        if i < Array.length published then Some published.(i) else None
      in
      Printf.printf "  %-12s %10.5f %9.5f %9.5f   %10.5f %9.5f   %16s\n" name
        theta sigma fisher reference.theta.(i) reference.sigma.(i)
        (match paper with
        | Some (v, e) -> Printf.sprintf "%g ± %g" v e
        | None -> "");
      check
        (name ^ " differs from scipy's")
        (Float.abs (theta -. reference.theta.(i)) <= 1e-4 *. reference.sigma.(i));
      check
        (name ^ "'s Fisher error differs from scipy's")
        (Float.abs (fisher -. reference.sigma.(i))
        <= 1e-4 *. reference.sigma.(i));
      Option.iter
        (fun (v, e) ->
          Printf.printf
            "  %-12s %+.2f of the paper's error, error %.2f of the paper's\n" ""
            ((theta -. v) /. e)
            (sigma /. e);
          check
            (name ^ " differs from the paper's")
            (Float.abs (theta -. v) <= 0.25 *. e);
          check
            (name ^ "'s error differs from the paper's")
            (Float.abs ((sigma /. e) -. 1.) <= 0.1))
        paper)
    names;
  check "chi2 differs from scipy's"
    (Float.abs (fit.chi2 -. reference.chi2) <= 1e-9 *. reference.chi2)

let () =
  let dir =
    match Sys.argv with
    | [| _ |] -> Filename.concat (Sys.getenv "HOME") ".cache/ymir"
    | [| _; dir |] -> dir
    | _ ->
        prerr_endline "usage: pantheon.exe [DIR]";
        exit 2
  in
  let rows =
    rows
      (fetch dir Reference.distances Reference.distances_url
         Reference.distances_blake2b)
  in
  let cov =
    covariance
      (fetch dir Reference.covariance Reference.covariance_url
         Reference.covariance_blake2b)
  in
  let hubble_flow (r : row) = r.z_hd > 0.01 in
  let sn_rows = sample rows cov hubble_flow in
  let shoes_rows = sample rows cov (fun r -> hubble_flow r || r.calibrator) in
  run ~title:"Pantheon+ & SH0ES calibrators, LCDM"
    ~names:[| "Omega_m"; "Omega_L"; "H0"; "M" |]
    ~model:(shoes ~flat:false) ~start:[| 0.5; 0.5; 70.; -19. |]
    ~reference:Reference.shoes ~against:brout_lcdm shoes_rows;
  run ~title:"Pantheon+ & SH0ES calibrators, flat LCDM"
    ~names:[| "Omega_m"; "H0"; "M" |] ~model:(shoes ~flat:true)
    ~start:[| 0.5; 70.; -19. |] ~reference:Reference.shoes_flat
    ~against:brout_flat shoes_rows;
  run ~title:"Pantheon+ alone, LCDM, offset (reported)"
    ~names:[| "Omega_m"; "Omega_L"; "offset" |]
    ~model:(sn ~flat:false) ~start:[| 0.5; 0.5; -19. |] ~reference:Reference.sn
    ~against:Reported sn_rows;
  run ~title:"Pantheon+ alone, flat LCDM, offset (reported)"
    ~names:[| "Omega_m"; "offset" |] ~model:(sn ~flat:true)
    ~start:[| 0.5; -19. |] ~reference:Reference.sn_flat ~against:Reported
    sn_rows;
  if !failed then exit 1
