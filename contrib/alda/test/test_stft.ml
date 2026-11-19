(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module F = Alda.Frames

(* torch.istft's inverses and numpy's least-squares ones, from
   golden/istft.npz *)
let golden = lazy (Nx_io.load_npz "golden/istft.npz")

let entry dt key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/istft.npz has no %s" key

let edge_of = function
  | 0 -> `Zero
  | 1 -> `Replicate
  | 2 -> `Symmetric_whole
  | 3 -> `Symmetric_half
  | e -> failf "edge %d" e

let floats a = Nx.to_array (Nx.cast Nx.float64 a)

(* Each element's bits, every NaN as one. *)
let bits a =
  Array.map
    (fun v -> if Float.is_nan v then 0x7ff8L else Int64.bits_of_float v)
    (floats a)

let equal_bits want got =
  equal (array int) (Nx.shape want) (Nx.shape got);
  equal (array int64) (bits want) (bits got)

let inverse_case tol name =
  let grid =
    Array.map Int64.to_int (Nx.to_array (entry Nx.int64 (name ^ ".grid")))
  in
  let g =
    F.v
      ~edge:(edge_of grid.(0))
      ~ends:(grid.(1), grid.(2))
      ~window:grid.(3) ~hop:grid.(4)
  in
  let z = entry Nx.complex128 (name ^ ".z")
  and win = entry Nx.float64 (name ^ ".win") in
  let got = Alda.istft Nx.float64 g ~win grid.(5) z in
  let want = entry Nx.float64 (name ^ ".x") in
  equal (array int) (Nx.shape want) (Nx.shape got);
  equal (array (float tol)) (floats want) (floats got)

let torch =
  cases "istft is torch.istft under Zero" ~id:Fun.id
    [ "torch.plain"; "torch.centred"; "torch.whisper" ]
    (inverse_case 1e-15)

let least_squares =
  cases "istft is the least-squares inverse" ~id:Fun.id
    (List.init 24 (fun i -> Printf.sprintf "ls.%d" i))
    (inverse_case 1e-13)

(* Grids, signals and splits *)

type case = {
  edge : Alda.edge;
  lo : int;
  hi : int;
  window : int;
  hop : int;
  n : int;
  split : [ `Ones | `By of int | `Random ];
  empties : bool;
  channels : bool;
  seed : int;
}

let name : Alda.edge -> string = function
  | `Zero -> "Zero"
  | `Replicate -> "Replicate"
  | `Symmetric_whole -> "Symmetric_whole"
  | `Symmetric_half -> "Symmetric_half"
  | `Antisymmetric_whole -> "Antisymmetric_whole"

let pp_case ppf c =
  Format.fprintf ppf
    "{edge %s; ends (%d, %d); window %d; hop %d; n %d; split %s%s; %s; seed %d}"
    (name c.edge) c.lo c.hi c.window c.hop c.n
    (match c.split with
    | `Ones -> "ones"
    | `By k -> string_of_int k
    | `Random -> "random")
    (if c.empties then " with empty chunks" else "")
    (if c.channels then "[n; 2] along 0" else "[n]")
    c.seed

let case =
  let open Gen in
  with_pp pp_case
  @@
  let small = frequency [ (1, constant 0); (3, int_range 0 24) ] in
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
  let* lo = small and+ hi = small in
  let* window = int_range 1 24 and+ hop = int_range 1 30 in
  let* n = frequency [ (1, int_range 0 6); (4, int_range 0 200) ] in
  let* split = of_list [ `Ones; `By 2; `By 5; `Random ] in
  let* empties = bool and+ channels = bool in
  let+ seed = int_range 0 1_000_000 in
  { edge; lo; hi; window; hop; n; split; empties; channels; seed }

let grid c = F.v ~edge:c.edge ~ends:(c.lo, c.hi) ~window:c.window ~hop:c.hop

(* The grid of [c] if the inverse takes it. *)
let inverting c =
  match c.edge with
  | #Alda.invertible as edge ->
      Some (F.v ~edge ~ends:(c.lo, c.hi) ~window:c.window ~hop:c.hop)
  | `Antisymmetric_whole -> None

(* [extra g n] is what a signal of [n] samples adds to the shortest of its
   frames. *)
let extra (g : _ F.t) n =
  let lo, hi = g.F.ends in
  let d = n + lo + hi - g.F.window in
  if d < 0 then n else d mod g.F.hop

(* Chunk lengths summing to [total], at least one, with empty chunks. *)
let chunks c total =
  let r = Random.State.make [| c.seed; 2 |] in
  let rec go left acc =
    if left = 0 then List.rev acc
    else
      let k =
        match c.split with
        | `Ones -> 1
        | `By k -> Int.min k left
        | `Random -> 1 + Random.State.int r (Int.min left 9)
      in
      let acc =
        if c.empties && Random.State.int r 3 = 0 then 0 :: acc else acc
      in
      go (left - k) (k :: acc)
  in
  match go total [] with
  | [] -> [ 0 ]
  | l -> if c.empties then (0 :: l) @ [ 0 ] else l

let random_window c =
  let r = Random.State.make [| c.seed; 3 |] in
  Nx.init Nx.float64 [| c.window |] (fun _ -> 0.1 +. Random.State.float r 0.9)

let random_spectra c m =
  let r = Random.State.make [| c.seed; 4 |] in
  let v _ =
    Complex.
      { re = Random.State.float r 2. -. 1.; im = Random.State.float r 2. -. 1. }
  in
  let b = (c.window / 2) + 1 in
  if c.channels then Nx.init Nx.complex128 [| 2; m; b |] v
  else Nx.init Nx.complex128 [| m; b |] v

let piece z a b =
  Nx.slice
    (List.init (Nx.ndim z) (fun i ->
         if i = Nx.ndim z - 2 then Nx.R (a, b) else Nx.A))
    z

let axis c = if c.channels then 0 else -1

let covers c =
  let short = c.n < Int.max c.lo c.hi + 1 in
  cover ~share:0.05 "signal shorter than an end"
    (short && F.count (grid c) c.n > 0);
  cover ~share:0.05 "hop past window" (c.hop > c.window);
  cover ~share:0.2 "many frames" (F.count (grid c) c.n > 4)

(* For the laws of overlap-add, which every edge has. *)
let covers_all c =
  covers c;
  cover ~share:0.1 "Antisymmetric_whole" (c.edge = `Antisymmetric_whole)

(* Feeds the spectra in chunks to [feed]: the last state and the samples out, in
   order. *)
let fed c z feed =
  let s, outs, _ =
    List.fold_left
      (fun (s, outs, at) k ->
        let s, x = feed s (piece z at (at + k)) in
        (s, x :: outs, at + k))
      (`Start, [], 0)
      (chunks c (Nx.dim (-2) z))
  in
  cover ~share:0.2 "samples out of a feed"
    (List.exists (fun x -> Nx.dim (axis c) x > 0) outs);
  (s, List.rev outs)

(* The joined samples of the feeds and [finish]. *)
let stream c z feed finish =
  let s, outs = fed c z feed in
  Nx.concatenate ~axis:(axis c) (outs @ [ finish s ])

let istft_stream =
  prop "every split inverts as the whole call" ~count:300 case (fun c ->
      covers c;
      let g = match inverting c with Some g -> g | None -> reject () in
      let win = random_window c in
      let z = random_spectra c (F.count g c.n) in
      let whole = Alda.istft ~axis:(axis c) Nx.float64 g ~win c.n z in
      let feed s z =
        let s = match s with `Start -> Alda.Istft.start | `S s -> s in
        let s, x = Alda.Istft.feed ~axis:(axis c) Nx.float64 g ~win s z in
        (`S s, x)
      in
      let finish = function
        | `S s -> Alda.Istft.finish ~axis:(axis c) g ~extra:(extra g c.n) s
        | `Start -> fail "no chunk"
      in
      equal_bits whole (stream c z feed finish))

let overlap_add_stream =
  prop "every split overlap-adds as the whole call" ~count:300 case (fun c ->
      covers_all c;
      let g = grid c in
      let r = Random.State.make [| c.seed; 5 |] in
      let lead = if c.channels then [| 2 |] else [||] in
      let f =
        Nx.init Nx.float64
          (Array.append lead [| F.count g c.n; c.window |])
          (fun _ -> Random.State.float r 2. -. 1.)
      in
      let whole = F.overlap_add ~axis:(axis c) g c.n f in
      let feed s f =
        let s = match s with `Start -> F.Overlap_add.start | `S s -> s in
        let s, x = F.Overlap_add.feed ~axis:(axis c) g s f in
        (`S s, x)
      in
      let finish = function
        | `S s -> F.Overlap_add.finish ~axis:(axis c) g ~extra:(extra g c.n) s
        | `Start -> fail "no chunk"
      in
      equal_bits whole (stream c f feed finish))

(* After [m] frames the signal's length is open but for [extra]: each [extra] in
   its range ends the signal of its length, and no other is taken. *)
let extras =
  prop "each extra ends the signal of its length" ~count:200 case (fun c ->
      covers_all c;
      let g = grid c in
      let m = F.count g c.n in
      let range = if m = 0 then c.window - c.lo - c.hi else c.hop in
      let r = Random.State.make [| c.seed; 8 |] in
      let f =
        Nx.init Nx.float64 [| m; c.window |] (fun _ ->
            Random.State.float r 2. -. 1.)
      in
      let feed s f =
        let s = match s with `Start -> F.Overlap_add.start | `S s -> s in
        let s, x = F.Overlap_add.feed g s f in
        (`S s, x)
      in
      let s, outs = fed { c with channels = false } f feed in
      let s = match s with `S s -> s | `Start -> fail "no chunk" in
      let finish extra = F.Overlap_add.finish g ~extra s in
      for extra = 0 to range - 1 do
        let n =
          if m = 0 then extra
          else ((m - 1) * c.hop) + c.window - c.lo - c.hi + extra
        in
        if n < 0 then
          raises_match (Exn.invalid_arg ~sub:"Alda.Frames.Overlap_add.finish")
            (fun () -> finish extra)
        else
          equal_bits (F.overlap_add g n f)
            (Nx.concatenate ~axis:(-1) (outs @ [ finish extra ]))
      done;
      List.iter
        (fun extra ->
          raises_match (Exn.invalid_arg ~sub:"Alda.Frames.Overlap_add.finish")
            (fun () -> finish extra))
        [ -1; Int.max 0 range ])

(* Sample [1] of 3 is read at three positions under [`Symmetric_whole] with ends
   (2, 2), which hold 1e16, -1e16 and 1: in order of position the sum is 1, in
   reverse 0. Under [`Antisymmetric_whole] with ends (1, 0), position [-1] reads
   [2 x(0) - x(1)]. *)
let fold_order =
  group "overlap-add's fold"
    [
      test "adds in order of position" (fun () ->
          let g = F.v ~edge:`Symmetric_whole ~ends:(2, 2) ~window:1 ~hop:1 in
          let f =
            Nx.create Nx.float64 [| 7; 1 |]
              [| 0.; 1e16; 0.; -1e16; 0.; 1.; 0. |]
          in
          equal (array float_exact) [| 0.; 1.; 0. |]
            (Nx.to_array (F.overlap_add g 3 f)));
      test "weighs by each position's coefficients" (fun () ->
          let g =
            F.v ~edge:`Antisymmetric_whole ~ends:(1, 0) ~window:1 ~hop:1
          in
          let f = Nx.create Nx.float64 [| 4; 1 |] [| 3.; 5.; 7.; 11. |] in
          equal (array float_exact) [| 11.; 4.; 11. |]
            (Nx.to_array (F.overlap_add g 3 f)));
    ]

(* [<frames g x, f> = <x, overlap_add g n f>]. *)
let adjoint =
  prop "overlap-add is framing's adjoint" ~count:200 case (fun c ->
      covers_all c;
      let g = grid c in
      let r = Random.State.make [| c.seed; 6 |] in
      let v _ = Random.State.float r 2. -. 1. in
      let x = Nx.init Nx.float64 [| c.n |] v in
      let f = Nx.init Nx.float64 [| F.count g c.n; c.window |] v in
      match F.frames g x with
      | exception Invalid_argument _ -> assume false
      | fx ->
          let dot a b = Nx.item [] (Nx.sum (Nx.mul a b)) in
          let l = dot fx f and rhs = dot x (F.overlap_add g c.n f) in
          equal (float_rel ~rel:1e-12 ~abs:1e-12) l rhs)

(* The inverse of a signal's transform is the signal where the envelope is
   positive, and [0.] where it is zero. *)
let round_trip =
  prop "istft inverts stft" ~count:200 case (fun c ->
      covers c;
      let g = match inverting c with Some g -> g | None -> reject () in
      let win = random_window c in
      let r = Random.State.make [| c.seed; 7 |] in
      let x =
        Nx.init Nx.float64 [| c.n |] (fun _ -> Random.State.float r 2. -. 1.)
      in
      match Alda.stft Nx.complex128 g ~win x with
      | exception Invalid_argument _ -> assume false
      | z ->
          let y = floats (Alda.istft Nx.float64 g ~win c.n z) in
          let env =
            floats
              (F.overlap_add g c.n
                 (Nx.broadcast_to [| F.count g c.n; c.window |] (Nx.square win)))
          in
          let want =
            Array.mapi (fun i v -> if env.(i) = 0. then 0. else v) (floats x)
          in
          equal (array (float 1e-12)) want y)

(* Once past the start, [k] frames give [k hop] samples and a state of the same
   structure and shapes. *)
let steady =
  test "frames past the start leave the state's shapes" (fun () ->
      let g = F.v ~edge:`Symmetric_whole ~ends:(200, 40) ~window:400 ~hop:160 in
      let st = Nx.Ptree.instantiate (module F.Overlap_add.State) in
      let shapes s =
        let leaves, sk = Nx.Ptree.flatten st s in
        (sk, List.map (fun (Nx.P a) -> Nx.shape a) leaves)
      in
      let f = Nx.ones Nx.float32 [| 2; 1; 400 |] in
      let s = ref F.Overlap_add.start and seen = ref [] in
      for _ = 1 to 12 do
        let s', x = F.Overlap_add.feed g !s f in
        s := s';
        seen := (Nx.dim (-1) x, shapes s') :: !seen
      done;
      match !seen with
      | (k, (sk, sh)) :: rest ->
          equal int 160 k;
          List.iteri
            (fun i (k', (sk', sh')) ->
              if i < 6 then begin
                equal int 160 k';
                equal (option string) None
                  (Nx.Ptree.Skeleton.diff ~this:"last" sk ~that:"earlier" sk');
                equal (list (array int)) sh sh'
              end)
            rest
      | [] -> fail "no feed")

let errors =
  group "misuse"
    [
      test "spectra of another count" (fun () ->
          let g = F.v ~edge:`Zero ~ends:(0, 0) ~window:4 ~hop:2 in
          raises_match (Exn.invalid_arg ~sub:"Alda.istft") (fun () ->
              Alda.istft Nx.float64 g
                ~win:(Nx.ones Nx.float64 [| 4 |])
                8
                (Nx.zeros Nx.complex128 [| 2; 3 |])));
      test "window of another length" (fun () ->
          let g = F.v ~edge:`Zero ~ends:(0, 0) ~window:4 ~hop:2 in
          raises_match (Exn.invalid_arg ~sub:"Alda.stft") (fun () ->
              Alda.stft Nx.complex128 g
                ~win:(Nx.ones Nx.float64 [| 3 |])
                (Nx.zeros Nx.float64 [| 8 |])));
    ]

let () =
  exit
    (run "alda.stft"
       [
         torch;
         least_squares;
         istft_stream;
         overlap_add_stream;
         extras;
         fold_order;
         adjoint;
         round_trip;
         steady;
         errors;
       ])
