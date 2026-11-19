(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let floats a = Nx.to_array (Nx.cast Nx.float64 a)

let bits a =
  Array.map
    (fun v -> if Float.is_nan v then 0x7ff8L else Int64.bits_of_float v)
    (floats a)

let equal_bits want got =
  equal (array int) (Nx.shape want) (Nx.shape got);
  equal (array int64) (bits want) (bits got)

let name : Omr.edge -> string = function
  | `Zero -> "Zero"
  | `Replicate -> "Replicate"
  | `Symmetric_whole -> "Symmetric_whole"
  | `Symmetric_half -> "Symmetric_half"
  | `Antisymmetric_whole -> "Antisymmetric_whole"

type case = {
  edge : Omr.edge;
  ends : (int * int) option;
  m : int;
  n : int;
  channels : bool;
  seed : int;
}

let pp_case ppf c =
  Format.fprintf ppf "{edge %s; ends %s; m %d; n %d; %s; seed %d}" (name c.edge)
    (match c.ends with
    | None -> "default"
    | Some (lo, hi) -> Printf.sprintf "(%d, %d)" lo hi)
    c.m c.n
    (if c.channels then "[n; 2] along 0" else "[n]")
    c.seed

let case =
  let open Gen in
  with_pp pp_case
  @@
  let* edge =
    of_list
      [
        `Zero;
        `Replicate;
        `Symmetric_whole;
        `Symmetric_half;
        `Antisymmetric_whole;
      ]
  in
  let* m = frequency [ (1, constant 1); (4, int_range 1 12) ] in
  let small = frequency [ (1, constant 0); (3, int_range 0 15) ] in
  let* ends =
    frequency
      [ (1, constant None); (3, map (fun p -> Some p) (pair small small)) ]
  in
  let* n = frequency [ (1, int_range 1 4); (4, int_range 1 120) ] in
  let* channels = bool in
  let+ seed = int_range 0 1_000_000 in
  { edge; ends; m; n; channels; seed }

let random seed shape =
  let r = Random.State.make [| seed |] in
  Nx.init Nx.float64 shape (fun _ -> Random.State.float r 2. -. 1.)

let signal c = random c.seed (if c.channels then [| c.n; 2 |] else [| c.n |])
let taps c = random (c.seed + 1) [| c.m |]
let axis c = if c.channels then 0 else -1

(* [direct c h x] is the formula of fir's doc on one signal, the extension read
   by Nx.extend. *)
let direct c h x =
  let lo, hi = Option.value c.ends ~default:(c.m - 1, 0) in
  let e = floats (Nx.extend (c.edge :> Nx.extension) (lo, hi) x) in
  let h = floats h in
  let count = Int.max 0 (c.n + lo + hi - c.m + 1) in
  Array.init count (fun j ->
      let s = ref 0. in
      for k = 0 to c.m - 1 do
        s := !s +. (h.(k) *. e.(j + c.m - 1 - k))
      done;
      !s)

let column c x k = if c.channels then Nx.slice [ Nx.A; Nx.I k ] x else x

let formula =
  prop "fir is its formula" ~count:300 case (fun c ->
      cover ~share:0.1 "default ends" (c.ends = None);
      cover ~share:0.05 "a signal shorter than an end"
        (match c.ends with
        | Some (lo, hi) -> c.n <= Int.max lo hi
        | None -> false);
      let h = taps c and x = signal c in
      let y = Omr.fir ~axis:(axis c) ~edge:c.edge ?ends:c.ends h x in
      for k = 0 to if c.channels then 1 else 0 do
        let want = direct c h (column c x k) in
        equal (array (float 1e-12)) want (floats (column c y k))
      done)

(* Chunk lengths summing to [n], empty ones among them. *)
let chunks seed n =
  let r = Random.State.make [| seed; 2 |] in
  let rec go left =
    if left = 0 then [ 0 ]
    else
      let k = Int.min left (Random.State.int r 9) in
      k :: go (left - k)
  in
  go n

let piece c x a b =
  if c.channels then Nx.slice [ Nx.R (a, b); Nx.A ] x
  else Nx.slice [ Nx.R (a, b) ] x

let stream_law =
  prop "every split gives fir's bits" ~count:300 case (fun c ->
      let h = taps c and x = signal c in
      let fir_feed = Omr.Fir.feed ~axis:(axis c) ~edge:c.edge ?ends:c.ends h in
      let s, outs, _ =
        List.fold_left
          (fun (s, outs, at) k ->
            let s, y = fir_feed s (piece c x at (at + k)) in
            (s, y :: outs, at + k))
          (Omr.Fir.start, [], 0) (chunks c.seed c.n)
      in
      cover ~share:0.3 "outputs from feeds"
        (List.exists (fun y -> Nx.dim (axis c) y > 0) outs);
      let last = Omr.Fir.finish ~axis:(axis c) ~edge:c.edge ?ends:c.ends h s in
      equal_bits
        (Omr.fir ~axis:(axis c) ~edge:c.edge ?ends:c.ends h x)
        (Nx.concatenate ~axis:(axis c) (List.rev (last :: outs))))

let complex =
  test "complex taps and signals" (fun () ->
      let r = Random.State.make [| 11 |] in
      let v _ =
        Complex.
          {
            re = Random.State.float r 2. -. 1.;
            im = Random.State.float r 2. -. 1.;
          }
      in
      let h = Nx.init Nx.complex128 [| 5 |] v
      and x = Nx.init Nx.complex128 [| 40 |] v in
      let y = Nx.to_array (Omr.fir h x) in
      let h = Nx.to_array h and x = Nx.to_array x in
      let want =
        Array.init 40 (fun j ->
            let s = ref Complex.zero in
            for k = 0 to 4 do
              if j - k >= 0 then
                s := Complex.add !s (Complex.mul h.(k) x.(j - k))
            done;
            !s)
      in
      equal
        (array (float 1e-14))
        (Array.map (fun z -> z.Complex.re) want)
        (Array.map (fun z -> z.Complex.re) y);
      equal
        (array (float 1e-14))
        (Array.map (fun z -> z.Complex.im) want)
        (Array.map (fun z -> z.Complex.im) y))

(* [<fir h x, w>] is bilinear: its gradients are central differences. *)
let gradient =
  test "fir is differentiable in its taps and signal" (fun () ->
      let h = random 1 [| 7 |]
      and x = random 2 [| 50 |]
      and w = random 3 [| 50 |] in
      let f (h, x) =
        Nx.sum (Nx.mul w (Omr.fir ~edge:`Symmetric_half ~ends:(3, 3) h x))
      in
      let gh, gx = Rune.grad Nx.Ptree.(pair tensor tensor) f (h, x) in
      let fd p k =
        let a = Nx.to_array p in
        let at d =
          let b = Array.copy a in
          b.(k) <- a.(k) +. d;
          Nx.create Nx.float64 (Nx.shape p) b
        in
        (Nx.item [] (f (if p == h then (at 1e-6, x) else (h, at 1e-6)))
        -. Nx.item [] (f (if p == h then (at (-1e-6), x) else (h, at (-1e-6))))
        )
        /. 2e-6
      in
      equal (array (float 1e-8)) (Array.init 7 (fd h)) (Nx.to_array gh);
      equal (array (float 1e-8)) (Array.init 50 (fd x)) (Nx.to_array gx))

(* scipy.signal.sosfiltfilt, from golden/zero_phase.npz *)
let golden = lazy (Nx_io.load_npz "golden/zero_phase.npz")

let entry key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack Nx.float64 p
  | None -> failf "golden/zero_phase.npz has no %s" key

let peak a = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. a

(* The two evaluations round differently: within 1e-12 of the peak. *)
let near want got = equal (array (float (1e-12 *. peak want))) want got

let even =
  cases "zero_phase is sosfiltfilt at even orders" ~id:string_of_int [ 0; 1; 2 ]
    (fun i ->
      let key s = Printf.sprintf "%d.%s" i s in
      near
        (floats (entry (key "y")))
        (floats (Omr.zero_phase (entry (key "sos")) (entry (key "x")))))

(* An odd order: SciPy's pad is 6 S, 3 shorter. With it passed the results
   agree. Without it they differ near the ends by a share of the peak, and the
   difference decays into the signal as the filter's transient does: by the
   middle third it is below a hundredth of the ends'. *)
let odd =
  cases "zero_phase at odd orders" ~id:string_of_int [ 3; 4; 5 ] (fun i ->
      let key s = Printf.sprintf "%d.%s" i s in
      let c = entry (key "sos") and x = entry (key "x") in
      let p = 6 * Nx.dim 0 c in
      near
        (floats (entry (key "y6")))
        (floats (Omr.zero_phase ~ends:(p, p) c x));
      let want = floats (entry (key "y"))
      and got = floats (Omr.zero_phase c x) in
      let n = Array.length want in
      let gap a b =
        let m = ref 0. in
        for j = a to b - 1 do
          m := Float.max !m (Float.abs (want.(j) -. got.(j)))
        done;
        !m
      in
      let ends = Float.max (gap 0 100) (gap (n - 100) n) in
      less float_exact ~than:ends (1e-3 *. peak want);
      less float_exact ~than:(1e-2 *. ends) (gap (n / 3) (2 * n / 3)))

let errors =
  group "misuse"
    [
      test "taps of two axes" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.fir") (fun () ->
              Omr.fir
                (Nx.ones Nx.float64 [| 2; 2 |])
                (Nx.ones Nx.float64 [| 5 |])));
      test "no taps" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.fir") (fun () ->
              Omr.fir (Nx.ones Nx.float64 [| 0 |]) (Nx.ones Nx.float64 [| 5 |])));
      test "a negative end" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Omr.zero_phase") (fun () ->
              Omr.zero_phase ~ends:(-1, 0)
                (Nx.create Nx.float64 [| 1; 6 |] [| 1.; 0.; 0.; 1.; 0.; 0. |])
                (Nx.ones Nx.float64 [| 5 |])));
    ]

let () =
  exit
    (run "omr.fir"
       [ formula; stream_law; complex; gradient; even; odd; errors ])
