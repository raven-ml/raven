(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The benchmark figures of the Guide, on small deterministic data. *)

open Hugin

let f64 a = Nx.create Nx.float64 [| Array.length a |] a

(* Confusion matrix. *)

let confusion () =
  let classes = [| "plane"; "car"; "bird"; "cat" |] in
  let m =
    Nx.create Nx.int32 [| 4; 4 |]
      [| 50l; 3l; 1l; 0l; 4l; 41l; 6l; 2l; 2l; 7l; 38l; 9l; 0l; 1l; 12l; 44l |]
  in
  let counts = Nx.cast Nx.float64 m in
  let recall =
    num ~title:(Text.v "recall")
      Nx.(div counts (sum ~axes:[ 1 ] ~keepdims:true counts))
  in
  let predicted = dim ~title:(Text.v "predicted") ~labels:classes 1
  and truth = dim ~title:(Text.v "true") ~labels:classes 0 in
  layer
    [
      rect ~x:predicted ~y:truth ~fill:recall ();
      text ~x:predicted ~y:truth ~text:(num m)
        ~fill:(map_range Color.contrast recall)
        ();
    ]
  |> coord (Coord.cartesian ~aspect:1. ())

(* Attention grid: a : [2; 2; T; T]. *)

let attention () =
  let tokens = [| "a"; "b"; "c"; "d" |] in
  let t = Array.length tokens in
  let a =
    Nx.init Nx.float32 [| 2; 2; t; t |] (fun i ->
        let q = i.(2) and k = i.(3) in
        if k > q then 0.
        else
          float (1 + ((i.(0) + (2 * i.(1)) + q + k) mod 4))
          /. float (4 * (q + 1)))
  in
  rect
    ~fy:(dim ~title:(Text.v "layer") 0)
    ~fx:(dim ~title:(Text.v "head") 1)
    ~y:(dim ~labels:tokens 2) ~x:(dim ~labels:tokens 3)
    ~fill:(num ~title:(Text.v "attention") a)
    ()

(* Embedding scatter: e : [n; 2], points on a spiral coloured by digit. *)

let embedding n =
  let digits = Array.init 10 string_of_int in
  let e =
    Nx.init Nx.float32 [| n; 2 |] (fun i ->
        let k = float i.(0) in
        let r = 1. +. (k /. float n *. 9.) and a = k *. 2.39996 in
        if i.(1) = 0 then r *. Float.cos a else r *. Float.sin a)
  in
  let labels =
    Nx.init Nx.int32 [| n |] (fun i -> Int32.of_int (i.(0) mod 10))
  in
  dot
    ~x:(num Nx.(slice [ A; I 0 ] e))
    ~y:(num Nx.(slice [ A; I 1 ] e))
    ~fill:(cat ~labels:digits labels)
    ~opacity:(const 0.3) ()

(* Image grid: batch : [6; 8; 8; 3] in [0, 1]. *)

let images () =
  let classes = [| "dark"; "light" |] in
  let batch =
    Nx.init Nx.float32 [| 6; 8; 8; 3 |] (fun i ->
        let b = i.(0) and y = i.(1) and x = i.(2) and c = i.(3) in
        let v = float ((x + y + b + c) mod 8) /. 7. in
        if b mod 2 = 0 then 0.2 *. v else 0.5 +. (0.5 *. v))
  in
  let truth = f64 [| 0.; 1.; 0.; 1.; 0.; 1. |] |> Nx.cast Nx.int32
  and pred = f64 [| 0.; 1.; 1.; 1.; 0.; 0. |] |> Nx.cast Nx.int32 in
  let name c = classes.(Int32.to_int c) in
  let captions =
    Array.map2
      (fun t p -> name t ^ " → " ^ name p)
      (Nx.to_array truth) (Nx.to_array pred)
  in
  layer
    [
      image ~fx:(dim ~scale:(Scale.band ~wrap:3 ()) ~labels:captions 0) batch;
      rect
        ~fx:(dim ~valid:(Nx.not_equal truth pred) 0)
        ~stroke:(const Color.red) ();
    ]

(* Loss landscape: loss : [nb; na], path : [n; 2]. *)

let landscape () =
  let alphas = Nx.linspace Nx.float64 (-2.) 2. 21 in
  let betas = Nx.linspace Nx.float64 (-1.) 1. 11 in
  let a = Nx.reshape [| 1; 21 |] alphas and b = Nx.reshape [| 11; 1 |] betas in
  let loss = Nx.add_s (Nx.add (Nx.square a) (Nx.mul_s (Nx.square b) 4.)) 0.1 in
  let path =
    Nx.create Nx.float64 [| 5; 2 |]
      [| -1.5; 0.8; -0.8; 0.4; -0.3; 0.1; -0.1; 0.02; 0.; 0. |]
  in
  let px = num Nx.(slice [ A; I 0 ] path)
  and py = num Nx.(slice [ A; I 1 ] path) in
  layer
    [
      contour ~x:(num alphas)
        ~y:(num Nx.(slice [ A; N ] betas))
        ~fill:(num ~scale:(Scale.log ()) ~title:(Text.v "loss") loss)
        ();
      line ~x:px ~y:py ();
      dot ~x:px ~y:py ~size:(const 9.) ();
    ]

(* Training dashboard, the run's history given as tensors. *)

let dashboard () =
  let steps = 200 in
  let step = Nx.linspace Nx.float64 0. (float (steps - 1)) steps in
  let loss =
    Nx.init Nx.float64 [| steps |] (fun i ->
        let s = float i.(0) in
        (Float.exp (-.s /. 60.) +. 0.05) *. (1. +. (0.2 *. Float.sin (s *. 0.7))))
  in
  let vstep = f64 [| 50.; 100.; 150.; 200. |] in
  let acc = f64 [| 0.62; 0.78; 0.83; 0.81 |] in
  let epochs = f64 [| 0.; 50.; 100.; 150.; 200. |] in
  let at t = Nx.take ~indices:(Nx.reshape [| 1 |] (Nx.argmax acc)) t in
  let losses =
    layer
      [
        rule ~x:(num epochs) ~opacity:(const 0.15) ();
        line
          ~x:(num ~title:(Text.v "step") step)
          ~y:(num ~scale:(Scale.log ()) ~title:(Text.v "loss") loss)
          ~opacity:(const 0.3) ();
        line ~x:(num step) ~y:(num (Nx.ewma ~alpha:0.1 loss)) ();
      ]
  and accuracy =
    layer
      [
        dot ~x:(num vstep) ~y:(num ~title:(Text.v "val. accuracy") acc) ();
        text
          ~x:(num (at vstep))
          ~y:(num (at acc))
          ~text:(num (at acc))
          ~dy:6. ();
      ]
  in
  grid [ [ losses ]; [ accuracy ] ] |> share [ ("x", `Shared) ]

(* Two-column paper figure, in the bundled faces at 8 points. *)

let paper () =
  let panel s f = title ~align:`Left (Text.bold (Text.v s)) f in
  let t = Nx.linspace Nx.float64 0. 6. 40 in
  let k = Nx.reshape [| 2; 1 |] (f64 [| 1.; 2. |]) in
  let a =
    line
      ~x:(num ~title:(Text.v "time (s)") t)
      ~y:(num ~title:(Text.v "amplitude") (Nx.sin (Nx.mul k t)))
      ~stroke:(dim ~title:(Text.v "harmonic") ~labels:[| "1st"; "2nd" |] 0)
      ()
  in
  let b =
    dot
      ~x:
        (num ~scale:(Scale.log ()) ~title:(Text.v "parameters")
           (f64 [| 10.; 100.; 1000.; 10000. |]))
      ~y:(num ~title:(Text.v "error") (f64 [| 0.9; 0.5; 0.3; 0.2 |]))
      ()
  in
  let methods = [| "ours"; "base" |] in
  let c =
    rect
      ~x:(strings ~title:(Text.v "method") methods)
      ~y:(num ~title:(Text.v "score") (f64 [| 0.82; 0.71 |]))
      ()
  in
  let d =
    rect ~x:(dim 1) ~y:(dim 0)
      ~fill:
        (num ~title:(Text.v "response")
           (Nx.init Nx.float64 [| 4; 5 |] (fun i -> float (i.(0) * i.(1)))))
      ()
  in
  grid [ [ panel "(a)" a; panel "(b)" b ]; [ panel "(c)" c; panel "(d)" d ] ]

(* Bands: two runs' mean accuracy over steps, each over the band of one standard
   deviation about it. *)

let bands () =
  let t = 24 in
  let at k i = float i.(1) +. (7. *. float k) in
  let mean =
    Nx.init Nx.float64 [| 2; t |] (fun i ->
        1. -. exp (-.at 0 i /. (6. +. (6. *. float i.(0)))))
  and std =
    Nx.init Nx.float64 [| 2; t |] (fun i ->
        0.05 +. (0.04 *. Float.abs (sin (at 1 i /. 3.))))
  in
  let run = dim ~title:(Text.v "run") ~labels:[| "a"; "b" |] 0 in
  let step = index ~title:(Text.v "step") (-1) in
  layer
    [
      area ~x:step
        ~y:(num ~title:(Text.v "accuracy") (Nx.add mean std))
        ~y2:(num (Nx.sub mean std))
        ~fill:run ~opacity:(const 0.25) ();
      line ~x:step ~y:(num mean) ~stroke:run ();
    ]

let paper_theme = Theme.v ~size:8. ()
let paper_size = Size.figure (Size.mm 180.) (Size.mm 110.)

(* The goldens: a name, a figure, a theme and a size. *)

let goldens =
  let default = Size.figure 360. 240. in
  [
    ("confusion", confusion, Theme.default, default);
    ("attention", attention, Theme.default, default);
    ("embedding", (fun () -> embedding 400), Theme.default, default);
    ("embedding-dense", (fun () -> embedding 21_000), Theme.default, default);
    ("images", images, Theme.default, default);
    ("landscape", landscape, Theme.default, default);
    ("dashboard", dashboard, Theme.default, default);
    ("paper", paper, paper_theme, paper_size);
    ("bands", bands, Theme.default, default);
  ]

(* Goldens are drawn at one device pixel per point. *)
let density = 1.
let drawing (_, f, theme, size) = render ~theme ~density size (f ())
