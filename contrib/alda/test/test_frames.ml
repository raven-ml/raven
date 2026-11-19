(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module F = Alda.Frames

(* torch.stft's framing and Kaldi's, from golden/frames.npz
   (contrib/alda/test/gen/frames.py) *)
let golden = lazy (Nx_io.load_npz "golden/frames.npz")

let entry dt key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/frames.npz has no %s" key

(* Each element's bits, every NaN as one: a NaN's sign and payload are left to
   the kernel library. *)
let bits a =
  Array.map
    (fun v -> if Float.is_nan v then 0x7ff8L else Int64.bits_of_float v)
    (Nx.to_array (Nx.cast Nx.float64 a))

let equal_bits ?msg want got =
  equal ?msg (array int) (Nx.shape want) (Nx.shape got);
  equal ?msg (array int64) (bits want) (bits got)

(* Whisper frames 400 samples every 160 of the signal reflected by 200 at its
   start, as torch.stft centred does, without its last frame. *)
let whisper : [ `Symmetric_whole ] F.t =
  F.v ~edge:`Symmetric_whole ~ends:(200, 40) ~window:400 ~hop:160

let whisper_counts =
  test "Whisper's count is torch.stft's less one" (fun () ->
      let c = Nx.to_array (entry Nx.int64 "whisper.counts") in
      for i = 0 to (Array.length c / 2) - 1 do
        let n = Int64.to_int c.(2 * i) and t = Int64.to_int c.((2 * i) + 1) in
        equal
          ~msg:(Printf.sprintf "%d samples" n)
          int (t - 1) (F.count whisper n)
      done)

let whisper_lengths =
  List.init 20 (fun i -> 201 + i)
  @ [ 399; 400; 401; 559; 560; 561; 1000; 1601; 4097; 6000 ]

let whisper_frames =
  cases "Whisper's frames are torch.stft's" ~id:string_of_int whisper_lengths
    (fun n ->
      let x = Nx.slice [ R (0, n) ] (entry Nx.float32 "whisper.signal") in
      let padded = entry Nx.float32 (Printf.sprintf "whisper.padded.%d" n) in
      let torch = Nx.sliding_window ~window:400 ~step:160 padded in
      let torch = Nx.slice [ R (0, Nx.dim 0 torch - 1) ] torch in
      equal_bits torch (F.frames whisper x))

(* Kaldi's grids: snip_edges true frames within the signal; false centres frame
   [m] on [m hop + hop / 2], reading the signal half-sample symmetric past both
   ends. *)
let kaldi snip w h =
  if snip then F.v ~edge:`Zero ~ends:(0, 0) ~window:w ~hop:h
  else
    let lo = (w / 2) - (h / 2) in
    F.v ~edge:`Symmetric_half ~ends:(lo, w - h + (h / 2) - lo) ~window:w ~hop:h

let kaldi_rows =
  List.concat_map
    (fun snip ->
      List.map
        (fun n -> (snip, 25, 10, n))
        (List.init 60 (fun i -> i + 1) @ [ 101; 499 ])
      @ List.map (fun n -> (snip, 400, 160, n)) [ 1; 7; 199; 400; 401; 499 ])
    [ true; false ]

let kaldi_id (snip, w, h, n) =
  Printf.sprintf "kaldi.%s.%d.%d.%d" (if snip then "snip" else "nosnip") w h n

let kaldi_frames =
  cases "Kaldi's frames" ~id:kaldi_id kaldi_rows
    (fun ((snip, w, h, n) as row) ->
      let x = Nx.slice [ R (0, n) ] (entry Nx.float64 "kaldi.signal") in
      equal_bits (entry Nx.float64 (kaldi_id row)) (F.frames (kaldi snip w h) x))

(* Streams *)

type layout = Mono | Lead | Channels

type case = {
  edge : Alda.edge;
  lo : int;
  hi : int;
  window : int;
  hop : int;
  n : int;
  split : [ `Ones | `By of int | `Random ];
  empties : bool;
  layout : layout;
  seed : int;
}

let pp_case ppf c =
  Format.fprintf ppf
    "{edge %s; ends (%d, %d); window %d; hop %d; n %d; split %s%s; layout %s; \
     seed %d}"
    (match c.edge with
    | `Zero -> "Zero"
    | `Replicate -> "Replicate"
    | `Symmetric_whole -> "Symmetric_whole"
    | `Symmetric_half -> "Symmetric_half"
    | `Antisymmetric_whole -> "Antisymmetric_whole")
    c.lo c.hi c.window c.hop c.n
    (match c.split with
    | `Ones -> "ones"
    | `By k -> string_of_int k
    | `Random -> "random")
    (if c.empties then " with empty chunks" else "")
    (match c.layout with
    | Mono -> "[n]"
    | Lead -> "[2; n]"
    | Channels -> "[n; 3]")
    c.seed

let name : Alda.edge -> string = function
  | `Zero -> "Zero"
  | `Replicate -> "Replicate"
  | `Symmetric_whole -> "Symmetric_whole"
  | `Symmetric_half -> "Symmetric_half"
  | `Antisymmetric_whole -> "Antisymmetric_whole"

let edges : Alda.edge list =
  [ `Zero; `Replicate; `Symmetric_whole; `Symmetric_half; `Antisymmetric_whole ]

let case =
  let open Gen in
  with_pp pp_case
  @@
  let small = frequency [ (1, constant 0); (3, int_range 0 50) ] in
  let* edge = of_list edges in
  let* lo = small and+ hi = small in
  let* window = int_range 1 40 and+ hop = int_range 1 50 in
  let* n = frequency [ (1, int_range 0 6); (4, int_range 0 400) ] in
  let* split = of_list [ `Ones; `By 160; `By 441; `By 7; `Random ] in
  let* empties = bool and+ layout = of_list [ Mono; Lead; Channels ] in
  let+ seed = int_range 0 1_000_000 in
  { edge; lo; hi; window; hop; n; split; empties; layout; seed }

(* Chunk lengths summing to [n], at least one, with empty chunks among them. *)
let chunks c =
  let r = Random.State.make [| c.seed |] in
  let rec go left acc =
    if left = 0 then List.rev acc
    else
      let k =
        match c.split with
        | `Ones -> 1
        | `By k -> Int.min k left
        | `Random -> 1 + Random.State.int r (Int.min left 120)
      in
      let acc =
        if c.empties && Random.State.int r 3 = 0 then 0 :: acc else acc
      in
      go (left - k) (k :: acc)
  in
  match go c.n [] with
  | [] -> [ 0 ]
  | l -> if c.empties then (0 :: l) @ [ 0 ] else l

let signal c =
  let r = Random.State.make [| c.seed; 1 |] in
  let value _ =
    match Random.State.int r 20 with
    | 0 -> -0.
    | 1 -> Float.nan
    | _ -> Random.State.float r 2. -. 1.
  in
  match c.layout with
  | Mono -> (-1, Nx.init Nx.float64 [| c.n |] value)
  | Lead -> (-1, Nx.init Nx.float64 [| 2; c.n |] value)
  | Channels -> (0, Nx.init Nx.float64 [| c.n; 3 |] value)

let piece axis x a b =
  let r = Nx.ndim x in
  let axis = if axis < 0 then axis + r else axis in
  Nx.slice (List.init r (fun i -> if i = axis then Nx.R (a, b) else Nx.A)) x

let outcome f =
  match f () with v -> Ok v | exception Invalid_argument _ -> Error ()

let stream_law =
  prop "every split frames as the whole call" ~count:400 case (fun c ->
      let g = F.v ~edge:c.edge ~ends:(c.lo, c.hi) ~window:c.window ~hop:c.hop in
      let axis, x = signal c in
      let fed = ref 0 in
      let whole = outcome (fun () -> F.frames ~axis g x) in
      let streamed =
        outcome (fun () ->
            let s, outs, _ =
              List.fold_left
                (fun (s, outs, at) k ->
                  let s, o = F.feed ~axis g s (piece axis x at (at + k)) in
                  fed := !fed + Nx.dim (-2) o;
                  (s, o :: outs, at + k))
                (F.start, [], 0) (chunks c)
            in
            Nx.concatenate ~axis:(-2) (List.rev (F.finish g s :: outs)))
      in
      let short = c.n < Int.max c.lo c.hi + 1 in
      cover ~share:0.05 "signal shorter than an end" (short && F.count g c.n > 0);
      cover ~share:0.05 "hop past window" (c.hop > c.window);
      cover ~share:0.2 "frames out of a feed" (!fed > 0);
      cover ~share:0.1 "a feed's frame reads before the start"
        (!fed > 0 && c.lo > 0);
      match (whole, streamed) with
      | Ok w, Ok s -> equal_bits w s
      | Error (), Error () -> ()
      | Ok _, Error () -> fail "the stream raised where the whole call did not"
      | Error (), Ok _ -> fail "the whole call raised where the stream did not")

(* Once the frames that read before the start are out, a chunk of [k hop]
   samples gives [k] frames and a state of the same structure and shapes. *)
let steady =
  test "chunks of whole hops leave the state's shapes" (fun () ->
      let st = Nx.Ptree.instantiate (module F.State) in
      let shapes s =
        let leaves, sk = Nx.Ptree.flatten st s in
        (sk, List.map (fun (Nx.P a) -> Nx.shape a) leaves)
      in
      let x =
        Nx.init Nx.float32 [| 2; 160 * 12 |] (fun i -> Float.of_int i.(1))
      in
      let chunk j = Nx.slice [ A; R (160 * j, 160 * (j + 1)) ] x in
      let s = ref F.start and seen = ref [] in
      for j = 0 to 11 do
        let s', f = F.feed whisper !s (chunk j) in
        s := s';
        seen := (Nx.dim (-2) f, shapes s') :: !seen
      done;
      match !seen with
      | (k, (sk, sh)) :: rest ->
          equal int 1 k;
          List.iteri
            (fun i (k', (sk', sh')) ->
              if i < 6 then begin
                equal int 1 k';
                equal (option string) None
                  (Nx.Ptree.Skeleton.diff ~this:"last" sk ~that:"earlier" sk');
                equal (list (array int)) sh sh'
              end)
            rest
      | [] -> fail "no feed")

(* The first frame of a grid of window 4, ends (3, 0) reads sample 0 and the
   samples its edge reads before the start: none, [x(0)], [x(0)] to [x(2)],
   [x(0)] to [x(3)] and [x(0)] to [x(3)]. It comes out of the feed that
   completes them, and not before. *)
let first_frame =
  cases "the first frame comes out once its samples are known"
    ~id:(fun (e, _) -> name e)
    [
      (`Zero, 1);
      (`Replicate, 1);
      (`Symmetric_half, 3);
      (`Symmetric_whole, 4);
      (`Antisymmetric_whole, 4);
    ]
    (fun (edge, known) ->
      let g = F.v ~edge ~ends:(3, 0) ~window:4 ~hop:5 in
      let x = Nx.init Nx.float64 [| 8 |] (fun i -> Float.of_int (i.(0) + 1)) in
      let frames k =
        let _, f = F.feed g F.start (Nx.slice [ R (0, k) ] x) in
        Nx.dim 0 f
      in
      equal ~msg:"one sample short" int 0 (frames (known - 1));
      equal ~msg:"the samples it reads" int 1 (frames known);
      let _, f = F.feed g F.start (Nx.slice [ R (0, known) ] x) in
      equal_bits (Nx.slice [ R (0, 1) ] (F.frames g x)) f)

(* Layout *)

let layout =
  test "framing along axis 0 of [n; c] gives [c; frames; window]" (fun () ->
      let g = F.v ~edge:`Zero ~ends:(2, 3) ~window:4 ~hop:2 in
      let x =
        Nx.init Nx.float64 [| 9; 2 |] (fun i ->
            Float.of_int ((10 * i.(0)) + i.(1)))
      in
      let f = F.frames ~axis:0 g x in
      equal (array int) [| 2; F.count g 9; 4 |] (Nx.shape f);
      let c1 = Nx.get [ 1 ] (Nx.transpose x) in
      equal_bits (F.frames g c1) (Nx.get [ 1 ] f))

let count_cases =
  cases "count"
    ~id:(fun (w, h, lo, hi, n, _) ->
      Printf.sprintf "%d %d (%d,%d) %d" w h lo hi n)
    [
      (400, 160, 0, 0, 399, 0);
      (400, 160, 0, 0, 400, 1);
      (400, 160, 0, 0, 559, 1);
      (400, 160, 0, 0, 560, 2);
      (400, 160, 200, 200, 0, 1);
      (400, 160, 200, 40, 480_000, 3000);
      (3, 5, 0, 0, 2, 0);
      (1, 1, 0, 0, 0, 0);
    ]
    (fun (w, h, lo, hi, n, want) ->
      let g = F.v ~edge:`Zero ~ends:(lo, hi) ~window:w ~hop:h in
      equal int want (F.count g n))

(* Instants *)

type centred = { sampling : Alda.Window.sampling; window : int; hop : int }

let pp_centred ppf c =
  Format.fprintf ppf "{%s; window %d; hop %d}"
    (match c.sampling with Periodic -> "periodic" | Symmetric -> "symmetric")
    c.window c.hop

let centred =
  let open Gen in
  with_pp pp_centred
  @@
  let* sampling = of_list [ Alda.Window.Periodic; Alda.Window.Symmetric ] in
  let+ window = frequency [ (1, constant 1); (5, int_range 2 40) ]
  and+ hop = int_range 1 30 in
  { sampling; window; hop }

let parity (c : centred) =
  match (c.sampling, c.window mod 2) with
  | _ when c.window = 1 -> "one point"
  | Periodic, 0 -> "periodic, even window"
  | Periodic, _ -> "periodic, odd window"
  | Symmetric, 0 -> "symmetric, even window"
  | Symmetric, _ -> "symmetric, odd window"

(* The frames of the ramp [x(i) = i] under [`Antisymmetric_whole] hold the
   positions their points lie at, the extension reading [2 x(0) - x(i) = -i]
   before the start and [n - 1 + i] past the end; point [i] of frame [m] is [m
   hop - lo + i]. A window's points [i] and [P - i] share bits, so the positions
   of each such pair sum to twice the instant the Frames section states for [lo
   = window / 2]. A window of one point is centred on it. *)
let instants =
  prop "a frame's instant is its window's centre" ~count:400 centred (fun c ->
      List.iter
        (fun l -> cover ~share:0.1 l (parity c = l))
        [
          "one point";
          "periodic, even window";
          "periodic, odd window";
          "symmetric, even window";
          "symmetric, odd window";
        ];
      let w = c.window and n = (3 * c.hop) + c.window in
      let lo = w / 2 in
      let g =
        F.v ~edge:`Antisymmetric_whole ~ends:(lo, lo) ~window:w ~hop:c.hop
      in
      let ramp = Nx.init Nx.float64 [| n |] (fun i -> Float.of_int i.(0)) in
      let pos = Nx.to_array (F.frames g ramp) in
      let win = Nx.to_array (Alda.Window.hann Nx.float64 c.sampling w) in
      let p = match c.sampling with Periodic -> w | Symmetric -> w - 1 in
      let shift =
        match (c.sampling, w mod 2) with
        | _ when w = 1 -> 0.
        | Periodic, 1 -> 0.5
        | Symmetric, 0 -> -0.5
        | _ -> 0.
      in
      for m = 0 to F.count g n - 1 do
        let instant = Float.of_int (m * c.hop) +. shift in
        for i = 0 to w - 1 do
          equal
            ~msg:(Printf.sprintf "frame %d, point %d" m i)
            float_exact
            (Float.of_int ((m * c.hop) - lo + i))
            pos.((m * w) + i)
        done;
        if w = 1 then
          equal ~msg:(Printf.sprintf "frame %d" m) float_exact instant pos.(m)
        else
          for i = Int.max 0 (p - w + 1) to Int.min (w - 1) p do
            let msg = Printf.sprintf "frame %d, points %d and %d" m i (p - i) in
            equal ~msg float_exact win.(i) win.(p - i);
            equal ~msg float_exact (2. *. instant)
              (pos.((m * w) + i) +. pos.((m * w) + p - i))
          done
      done)

(* A grid of ends [(window / 2, window / 2)] has [n / hop + 1] frames for an
   even [window], one fewer with [hi = window / 2 - hop], and ends [(0, 0)] keep
   the frames within the signal. *)
let centred_counts =
  prop "centred grids' counts" centred (fun c ->
      let w = c.window and h = c.hop in
      cover ~share:0.1 "even window, hop at most half of it"
        (w mod 2 = 0 && h <= w / 2);
      let count lo hi n =
        F.count (F.v ~edge:`Zero ~ends:(lo, hi) ~window:w ~hop:h) n
      in
      for n = 0 to (3 * h) + w do
        let msg = Printf.sprintf "%d samples" n in
        if w mod 2 = 0 then begin
          equal ~msg int ((n / h) + 1) (count (w / 2) (w / 2) n);
          if h <= w / 2 then
            equal ~msg int (n / h) (count (w / 2) ((w / 2) - h) n)
        end;
        let within = ref 0 in
        while (!within * h) + w <= n do
          incr within
        done;
        equal ~msg int !within (count 0 0 n)
      done)

let errors =
  group "misuse"
    [
      test "window below 1" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Alda.Frames.v") (fun () ->
              F.v ~edge:`Zero ~ends:(0, 0) ~window:0 ~hop:1));
      test "negative end" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Alda.Frames.v") (fun () ->
              F.v ~edge:`Zero ~ends:(-1, 0) ~window:4 ~hop:1));
      test "no sample read past an end" (fun () ->
          let g = F.v ~edge:`Symmetric_whole ~ends:(1, 0) ~window:2 ~hop:1 in
          raises_match (Exn.invalid_arg ~sub:"Alda.Frames.frames") (fun () ->
              F.frames g (Nx.zeros Nx.float32 [| 0 |])));
      test "axis outside the signal" (fun () ->
          let g = F.v ~edge:`Zero ~ends:(0, 0) ~window:2 ~hop:1 in
          raises_match (Exn.invalid_arg ~sub:"Alda.Frames.frames") (fun () ->
              F.frames ~axis:1 g (Nx.zeros Nx.float32 [| 4 |])));
      test "chunks of other shapes" (fun () ->
          let g = F.v ~edge:`Zero ~ends:(0, 0) ~window:2 ~hop:1 in
          let s, _ = F.feed g F.start (Nx.zeros Nx.float32 [| 2; 3 |]) in
          raises_match (Exn.invalid_arg ~sub:"Alda.Frames.feed") (fun () ->
              F.feed g s (Nx.zeros Nx.float32 [| 3; 3 |])));
      test "finish before any chunk" (fun () ->
          let g = F.v ~edge:`Zero ~ends:(0, 0) ~window:2 ~hop:1 in
          raises_match (Exn.invalid_arg ~sub:"Alda.Frames.finish") (fun () ->
              (F.finish g F.start : (float, Nx.float32_elt, Nx.host) Nx.t)));
    ]

let () =
  exit
    (run "alda.frames"
       [
         whisper_counts;
         whisper_frames;
         kaldi_frames;
         stream_law;
         steady;
         first_frame;
         layout;
         count_cases;
         instants;
         centred_counts;
         errors;
       ])
