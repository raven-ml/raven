(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* scipy.signal.periodogram's scalings, from golden/spectra.npz *)
let golden = lazy (Nx_io.load_npz "golden/spectra.npz")

let entry key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack Nx.float64 p
  | None -> failf "golden/spectra.npz has no %s" key

(* Squared magnitudes of the windowed signal's [n]-point transform. *)
let power n x win =
  let z = Nx.rfft Nx.complex128 ~s:[ n ] (Nx.mul x win) in
  Nx.add (Nx.square (Nx.real Nx.float64 z)) (Nx.square (Nx.imag Nx.float64 z))

(* Within 1e-13 of the largest bin: the transforms' rounding is relative to the
   signal's energy. *)
let near want got =
  let want = Nx.to_array want in
  let top = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. want in
  equal (array (float (1e-13 *. top))) want (Nx.to_array got)

let periodogram =
  cases "periodogram scalings" ~id:string_of_int (List.init 8 Fun.id) (fun i ->
      let key s = Printf.sprintf "%d.%s" i s in
      let x = entry (key "x") and win = entry (key "win") in
      let params = Nx.to_array (entry (key "params")) in
      let rate = params.(0) and n = Float.to_int params.(1) in
      let p = power n x win in
      near (entry (key "density")) (Alda.psd ~n ~rate ~win p);
      near (entry (key "spectrum")) (Alda.power_spectrum ~n ~win p))

(* A sinusoid of amplitude [a] at a bin's frequency reads [a² / 2] there. *)
let sinusoid =
  test "a sinusoid's power is half its squared amplitude" (fun () ->
      let n = 64 and k = 5 and a = 3. in
      let x =
        Nx.init Nx.float64 [| 2; n |] (fun i ->
            a
            *. Float.cos
                 (2. *. Float.pi *. Float.of_int (k * i.(1)) /. Float.of_int n))
      in
      let win = Alda.Window.hann Nx.float64 Alda.Window.Periodic n in
      let s = Alda.power_spectrum ~win (power n x win) in
      equal (float 1e-12) (a *. a /. 2.) (Nx.item [ 1; k ] s))

let errors =
  let win = Nx.ones Nx.float64 [| 8 |] in
  group "misuse"
    [
      test "bins of another transform" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Alda.psd") (fun () ->
              Alda.psd ~rate:1. ~win (Nx.zeros Nx.float64 [| 4 |])));
      test "transform shorter than the window" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Alda.power_spectrum") (fun () ->
              Alda.power_spectrum ~n:4 ~win (Nx.zeros Nx.float64 [| 3 |])));
      test "rate of zero" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Alda.psd") (fun () ->
              Alda.psd ~rate:0. ~win (Nx.zeros Nx.float64 [| 5 |])));
    ]

let () = exit (run "alda.spectra" [ periodogram; sinusoid; errors ])
