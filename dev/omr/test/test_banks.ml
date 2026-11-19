(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module M = Omr.Mel

(* librosa's banks at float64 and Whisper's mel_filters.npz, from
   golden/banks.npz *)
let golden = lazy (Nx_io.load_npz "golden/banks.npz")

let entry dt key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/banks.npz has no %s" key

let floats a = Nx.to_array (Nx.cast Nx.float64 a)
let top a = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. a

(* [ulps32 ~floor want got] asserts each of [got] within one float32 ulp of
   [want] rounded to float32, or within [floor]. *)
let ulps32 ~floor want got =
  equal int (Array.length want) (Array.length got);
  Array.iteri
    (fun i w ->
      let w32 = Int32.float_of_bits (Int32.bits_of_float w) in
      let a = Float.abs w32 in
      let ulp =
        Int32.float_of_bits (Int32.add (Int32.bits_of_float a) 1l) -. a
      in
      equal
        ~msg:(Printf.sprintf "element %d" i)
        (float (Float.max ulp floor))
        w32 got.(i))
    want

let bank dt i =
  let p =
    Nx.to_array (entry Nx.float64 (Printf.sprintf "librosa.%d.params" i))
  in
  let scale = if p.(0) = 1. then M.Htk else M.Slaney in
  let norm = if p.(1) = 1. then Omr.Unit_area else Omr.Unit_peak in
  let n = Float.to_int p.(3) in
  let bands =
    Nx.dim 1 (entry Nx.float64 (Printf.sprintf "librosa.%d.bank" i))
  in
  M.bank dt scale norm ~rate:p.(2) ~n ~low:p.(4) ~high:p.(5) bands

(* A weight's error is its edges' relative error times its band's frequency over
   its width, up to 4e-14 of the bank's largest weight on these banks. *)
let tol want = 4e-14 *. top want

let librosa64 =
  cases "librosa's banks at float64" ~id:string_of_int (List.init 8 Fun.id)
    (fun i ->
      let want = entry Nx.float64 (Printf.sprintf "librosa.%d.bank" i) in
      let got = bank Nx.float64 i in
      equal (array int) (Nx.shape want) (Nx.shape got);
      equal (array (float (tol (floats want)))) (floats want) (floats got))

let librosa32 =
  cases "librosa's banks rounded once" ~id:string_of_int (List.init 8 Fun.id)
    (fun i ->
      let want =
        floats (entry Nx.float64 (Printf.sprintf "librosa.%d.bank" i))
      in
      ulps32 ~floor:(tol want) want (floats (bank Nx.float32 i)))

let whisper =
  cases "Whisper's banks" ~id:string_of_int [ 80; 128 ] (fun b ->
      let want = entry Nx.float32 (Printf.sprintf "whisper.%d" b) in
      let got =
        M.bank Nx.float32 M.Slaney Omr.Unit_area ~rate:16_000. ~n:400 b
      in
      equal (array int) (Nx.shape want) (Nx.shape got);
      ulps32 ~floor:0. (floats want) (floats got))

(* Triangles *)

let peaks =
  test "a unit-peak triangle rises and falls linearly" (fun () ->
      let at = Nx.create Nx.float64 [| 7 |] [| 0.; 1.; 2.; 3.; 4.; 5.; 6. |] in
      let e = Nx.create Nx.float64 [| 4 |] [| 1.; 3.; 4.; 6. |] in
      let t = Omr.triangles Omr.Unit_peak ~at e in
      equal (array int) [| 7; 2 |] (Nx.shape t);
      equal (array float_exact)
        [| 0.; 0.; 0.; 0.; 0.5; 0.; 1.; 0.; 0.; 1.; 0.; 0.5; 0.; 0. |]
        (Nx.to_array t))

let areas =
  test "a unit-area triangle has area one" (fun () ->
      let k = 4001 in
      let at = Nx.linspace Nx.float64 0. 10. k in
      let e = Nx.create Nx.float64 [| 5 |] [| 1.; 2.5; 4.; 7.; 8.5 |] in
      let t = Omr.triangles Omr.Unit_area ~at e in
      let dx = 10. /. Float.of_int (k - 1) in
      let area = Nx.to_array (Nx.mul_s (Nx.sum ~axes:[ 0 ] t) dx) in
      equal (array (float 1e-6)) [| 1.; 1.; 1. |] area)

(* Rune.grad through the triangles, in their edges and points, against central
   differences. *)
type tri = { at : float array; edges : float array; weights : float array }

let tri =
  let open Gen in
  let* k = int_range 1 12 and+ b = int_range 1 6 in
  let* at = array ~size:(constant k) (float_range 0. 100.) in
  let* gaps = array ~size:(constant (b + 2)) (float_range 1. 20.) in
  let+ weights = array ~size:(constant (k * b)) (float_range (-1.) 1.) in
  let edges = Array.make (b + 2) 0. in
  Array.iteri
    (fun i g -> edges.(i) <- (if i = 0 then g else edges.(i - 1) +. g))
    gaps;
  { at; edges; weights }

let gradients =
  prop "gradients are central differences" ~count:200 tri (fun t ->
      let k = Array.length t.at and b = Array.length t.edges - 2 in
      (* Away from the kinks, where a point meets an edge. *)
      assume
        (Array.for_all
           (fun p -> Array.for_all (fun e -> Float.abs (p -. e) > 1e-3) t.edges)
           t.at);
      let w = Nx.create Nx.float64 [| k; b |] t.weights in
      let loss norm (at, e) = Nx.sum (Nx.mul (Omr.triangles norm ~at e) w) in
      List.iter
        (fun norm ->
          let at = Nx.create Nx.float64 [| k |] t.at in
          let e = Nx.create Nx.float64 [| b + 2 |] t.edges in
          let ga, ge =
            Rune.grad Nx.Ptree.(pair tensor tensor) (loss norm) (at, e)
          in
          let central f x i =
            let h = 1e-6 in
            let moved d =
              Array.mapi (fun j v -> if j = i then v +. d else v) x
            in
            (f (moved h) -. f (moved (-.h))) /. (2. *. h)
          in
          let value at e =
            Nx.item []
              (loss norm
                 ( Nx.create Nx.float64 [| k |] at,
                   Nx.create Nx.float64 [| b + 2 |] e ))
          in
          let tol = float_rel ~rel:1e-5 ~abs:1e-7 in
          equal (array tol)
            (Array.init k (central (fun a -> value a t.edges) t.at))
            (Nx.to_array ga);
          equal (array tol)
            (Array.init (b + 2) (central (fun e -> value t.at e) t.edges))
            (Nx.to_array ge))
        [ Omr.Unit_peak; Omr.Unit_area ])

(* Scales *)

(* The scales' derivatives are central differences, 0 Hz and 0 mel among the
   points, away from Slaney's knee at 1000 Hz, 15 mel. *)
let scale_gradients =
  let point =
    Gen.frequency [ (1, Gen.constant 0.); (4, Gen.float_range 0. 20_000.) ]
  in
  prop "the scales' gradients are central differences" ~count:200 point
    (fun f ->
      cover ~share:0.1 "0 Hz" (f = 0.);
      assume (Float.abs (f -. 1000.) > 1.);
      let m = f /. 300. in
      assume (Float.abs (m -. 15.) > 0.01);
      List.iter
        (fun (sc, conv, x) ->
          let value x =
            Nx.item [ 0 ] (conv sc (Nx.create Nx.float64 [| 1 |] [| x |]))
          in
          let h = 1e-6 *. Float.max 1. x in
          let central = (value (x +. h) -. value (x -. h)) /. (2. *. h) in
          let g =
            Rune.grad Nx.Ptree.tensor
              (fun x -> Nx.sum (conv sc x))
              (Nx.create Nx.float64 [| 1 |] [| x |])
          in
          equal (float_rel ~rel:1e-6 ~abs:1e-9) central (Nx.item [ 0 ] g))
        [
          (M.Slaney, M.of_hz, f);
          (M.Htk, M.of_hz, f);
          (M.Slaney, M.to_hz, m);
          (M.Htk, M.to_hz, m);
        ])

let round_trip =
  prop "to_hz inverts of_hz" (Gen.float_range 0. 30_000.) (fun f ->
      List.iter
        (fun sc ->
          let x = Nx.create Nx.float64 [| 1 |] [| f |] in
          let back = Nx.item [ 0 ] (M.to_hz sc (M.of_hz sc x)) in
          equal (float_rel ~rel:1e-13 ~abs:1e-12) f back)
        [ M.Slaney; M.Htk ])

let scales =
  cases "scales' values"
    ~id:(fun (sc, f, _) ->
      Printf.sprintf "%s %g"
        (match sc with M.Slaney -> "Slaney" | M.Htk -> "Htk")
        f)
    [
      (M.Slaney, 0., 0.);
      (M.Slaney, 200., 3.);
      (M.Slaney, 1000., 15.);
      (M.Slaney, 6400., 42.);
      (M.Htk, 0., 0.);
      (M.Htk, 700., 1127. *. Float.log 2.);
    ]
    (fun (sc, f, m) ->
      let x = Nx.create Nx.float64 [| 1 |] [| f |] in
      equal (float_rel ~rel:1e-15 ~abs:1e-15) m (Nx.item [ 0 ] (M.of_hz sc x)))

let edges =
  test "edges include both ends and space evenly" (fun () ->
      let e = M.edges Nx.float64 M.Htk ~low:100. ~high:7000. 12 in
      let a = Nx.to_array e in
      equal float_exact 100. a.(0);
      equal float_exact 7000. a.(11);
      let m = Nx.to_array (M.of_hz M.Htk e) in
      let step = (m.(11) -. m.(0)) /. 11. in
      Array.iteri
        (fun i v -> if i > 0 then equal (float 1e-9) step (v -. m.(i - 1)))
        m)

let errors =
  group "misuse"
    [
      test "a band of no bin" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Mel.bank") (fun () ->
              M.bank Nx.float32 M.Slaney Omr.Unit_area ~rate:16_000. ~n:64 128));
      test "high past half the rate" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Mel.bank") (fun () ->
              M.bank Nx.float32 M.Htk Omr.Unit_peak ~rate:16_000. ~n:400
                ~high:9000. 40));
      test "too few edges" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.triangles") (fun () ->
              Omr.triangles Omr.Unit_peak
                ~at:(Nx.zeros Nx.float64 [| 3 |])
                (Nx.zeros Nx.float64 [| 2 |])));
      test "one edge" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.Mel.edges") (fun () ->
              M.edges Nx.float64 M.Htk ~low:0. ~high:10. 1));
    ]

let () =
  exit
    (run "omr.banks"
       [
         librosa64;
         librosa32;
         whisper;
         peaks;
         areas;
         gradients;
         scale_gradients;
         round_trip;
         scales;
         edges;
         errors;
       ])
