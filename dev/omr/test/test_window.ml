(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module W = Omr.Window

(* scipy.signal.windows at float64, keyed "<kind>.<sampling>.<n>" *)
let golden = lazy (Nx_io.load_npz "golden/windows.npz")

let reference key n =
  Nx.to_array
    (Nx_io.Archive.tensor ~shape:[| n |] Nx.float64 key (Lazy.force golden))

let sizes = [ 0; 1; 2; 3; 4; 5; 16; 17; 400; 401 ]

type maker = {
  make :
    's. (float, 's) Nx.dtype -> W.sampling -> int -> (float, 's, Nx.host) Nx.t;
}

(* Name, absolute tolerance at float64 against scipy, window. *)
let windows =
  [
    ("hann", 1e-15, { make = W.hann });
    ("hamming", 1e-15, { make = W.hamming });
    ("blackman", 1e-15, { make = W.blackman });
    ( "cosine",
      1e-15,
      { make = (fun dt s n -> W.cosine_sum dt s [| 0.3; 0.4; 0.2; 0.1 |] n) } );
    ("kaiser0", 1e-15, { make = (fun dt s n -> W.kaiser dt s ~beta:0. n) });
    ("kaiser8.6", 1e-14, { make = (fun dt s n -> W.kaiser dt s ~beta:8.6 n) });
    ("kaiser14", 1e-14, { make = (fun dt s n -> W.kaiser dt s ~beta:14. n) });
    ( "gaussian1.5",
      1e-15,
      { make = (fun dt s n -> W.gaussian dt s ~sigma:1.5 n) } );
    ("gaussian7", 1e-15, { make = (fun dt s n -> W.gaussian dt s ~sigma:7. n) });
  ]

let samplings = [ ("periodic", W.Periodic); ("symmetric", W.Symmetric) ]

let rows =
  List.concat_map
    (fun (name, tol, mk) ->
      List.concat_map
        (fun (sname, s) ->
          List.map (fun n -> (name, tol, mk, sname, s, n)) sizes)
        samplings)
    windows

let id (name, _, _, sname, _, n) = Printf.sprintf "%s.%s.%d" name sname n

(* Each point within [tol] of scipy's at float64. *)
let float64_cases =
  cases "float64 against scipy" ~id rows (fun ((_, tol, mk, _, s, n) as row) ->
      let got = Nx.to_array (mk.make Nx.float64 s n) in
      equal (array (float tol)) (reference (id row) n) got)

(* One rounding of a float64 value within a few ulps of scipy's is within one
   float32 ulp of scipy's rounded to float32. *)
let float32_cases =
  cases "float32 rounds once" ~id rows (fun ((_, _, mk, _, s, n) as row) ->
      let want = reference (id row) n in
      let got = Nx.to_array (mk.make Nx.float32 s n) in
      equal int n (Array.length got);
      Array.iteri
        (fun i w ->
          let w32 = Int32.float_of_bits (Int32.bits_of_float w) in
          let a = Float.abs w32 in
          let ulp =
            Int32.float_of_bits (Int32.add (Int32.bits_of_float a) 1l) -. a
          in
          equal ~msg:(Printf.sprintf "point %d" i) (float ulp) w32 got.(i))
        want)

(* Point [i] of a symmetric window has the bits of point [n - 1 - i], and of a
   periodic one those of point [n - i]. *)
let symmetry =
  cases "mirror points share bits" ~id rows (fun (_, _, mk, _, s, n) ->
      let a = Nx.to_array (mk.make Nx.float64 s n) in
      let mirror i =
        match s with W.Symmetric -> n - 1 - i | W.Periodic -> (n - i) mod n
      in
      Array.iteri (fun i v -> equal float_exact a.(mirror i) v) a)

let errors =
  group "misuse"
    [
      test "negative points" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Window.hann") (fun () ->
              W.hann Nx.float64 W.Periodic (-1)));
      test "no coefficient" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Window.cosine_sum") (fun () ->
              W.cosine_sum Nx.float64 W.Periodic [||] 4));
      test "negative beta" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Window.kaiser") (fun () ->
              W.kaiser Nx.float64 W.Periodic ~beta:(-1.) 4));
      test "infinite beta" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Window.kaiser") (fun () ->
              W.kaiser Nx.float64 W.Periodic ~beta:Float.infinity 4));
      test "zero sigma" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Window.gaussian") (fun () ->
              W.gaussian Nx.float64 W.Periodic ~sigma:0. 4));
    ]

let () =
  exit (run "omr.window" [ float64_cases; float32_cases; symmetry; errors ])
