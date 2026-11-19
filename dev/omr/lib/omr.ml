(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* Windows *)

module Window = struct
  type sampling = Periodic | Symmetric

  let period s n = match s with Periodic -> n | Symmetric -> n - 1

  (* [points ~by dt n f] is [f n p], the float64 window of [n] points of period
     [p], rounded once into [dt]. One point is [1.] under both samplings. *)
  let points ~by dt s n f =
    if n < 0 then invalid_argf "%s: %d points" by n;
    if n = 1 then Nx.ones dt [| 1 |] else Nx.cast dt (f n (period s n))

  (* [of_ints n g] is the float64 host array of [g i] for [i < n]; [g] is
     integer arithmetic, so the point it gives is exact. *)
  let of_ints n g =
    Nx.create Nx.float64 [| n |] (Array.init n (fun i -> Float.of_int (g i)))

  let cosines ~by dt s a n =
    if Array.length a = 0 then invalid_argf "%s: no coefficient" by;
    points ~by dt s n @@ fun n p ->
    (* Term [k] at point [i] is [cos (2π m / p)] for [m = ki mod p] folded into
       [\[0, p/2\]], so points [i] and [p - i] read one angle. *)
    let term k =
      let m =
        of_ints n (fun i ->
            let r = k * i mod p in
            Int.min r (p - r))
      in
      let c = Nx.cos (Nx.mul_s m (2. *. Float.pi /. Float.of_int p)) in
      Nx.mul_s c (if k land 1 = 0 then a.(k) else -.a.(k))
    in
    let acc = ref (term 0) in
    for k = 1 to Array.length a - 1 do
      acc := Nx.add !acc (term k)
    done;
    !acc

  let cosine_sum dt s a n = cosines ~by:"Omr.Window.cosine_sum" dt s a n
  let hann dt s n = cosines ~by:"Omr.Window.hann" dt s [| 0.5; 0.5 |] n
  let hamming dt s n = cosines ~by:"Omr.Window.hamming" dt s [| 0.54; 0.46 |] n

  let blackman dt s n =
    cosines ~by:"Omr.Window.blackman" dt s [| 0.42; 0.5; 0.08 |] n

  (* [centred n p] is [(2i - p) / p] at each point [i], [2u - 1] for [u = i /
     p]: one rounding, and the negation of point [p - i]'s. *)
  let centred n p = Nx.div_s (of_ints n (fun i -> (2 * i) - p)) (Float.of_int p)

  let kaiser dt s ~beta n =
    let by = "Omr.Window.kaiser" in
    if not (Float.is_finite beta && beta >= 0.) then
      invalid_argf "%s: beta %g" by beta;
    points ~by dt s n @@ fun n p ->
    let t = centred n p in
    let a = Nx.mul_s (Nx.sqrt (Nx.rsub_s 1. (Nx.square t))) beta in
    (* [I₀ a / I₀ β = (i0e a / i0e β) e^(a - β)], finite as [a ≤ β]. *)
    let i0e_beta = Nx.i0e (Nx.scalar Nx.float64 beta) in
    Nx.mul (Nx.div (Nx.i0e a) i0e_beta) (Nx.exp (Nx.sub_s a beta))

  let gaussian dt s ~sigma n =
    let by = "Omr.Window.gaussian" in
    if not (Float.is_finite sigma && sigma > 0.) then
      invalid_argf "%s: sigma %g" by sigma;
    points ~by dt s n @@ fun n p ->
    (* [(i - p/2) / σ = (2i - p) / 2σ], [2σ] exact. *)
    let z = Nx.div_s (of_ints n (fun i -> (2 * i) - p)) (2. *. sigma) in
    Nx.exp (Nx.mul_s (Nx.square z) (-0.5))
end

(* Axes *)

(* [time ~by axis x] is [x] with its time axis [axis] moved last, a view. *)
let time ~by axis x =
  let r = Nx.ndim x in
  let a = if axis < 0 then axis + r else axis in
  if a < 0 || a >= r then
    invalid_argf "%s: no axis %d in a %d-d signal" by axis r;
  if a = r - 1 then x else Nx.moveaxis a (-1) x

(* [along x a b] is positions [a] to [b - 1] of [x]'s last axis, a view. *)
let along x a b =
  Nx.slice (List.init (Nx.ndim x - 1) (fun _ -> Nx.A) @ [ Nx.R (a, b) ]) x

(* [untime ~by axis x] is [x] with its last axis moved to [axis] of the result,
   a view. *)
let untime ~by axis x =
  let r = Nx.ndim x in
  let a = if axis < 0 then axis + r else axis in
  if a < 0 || a >= r then
    invalid_argf "%s: no axis %d in a %d-d result" by axis r;
  if a = r - 1 then x else Nx.moveaxis (-1) a x

(* [fit x len] is [x] cut or extended with zeros to [len] along its last
   axis. *)
let fit x len =
  let d = Nx.dim (-1) x in
  if d >= len then along x 0 len
  else
    Nx.pad
      (Array.init (Nx.ndim x) (fun i ->
           if i = Nx.ndim x - 1 then (0, len - d) else (0, 0)))
      0. x

let lead x k = Array.sub (Nx.shape x) 0 (Nx.ndim x - k)

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

(* Frames *)

type edge =
  [ `Zero
  | `Replicate
  | `Symmetric_whole
  | `Symmetric_half
  | `Antisymmetric_whole ]

(* [need e w] is the number of samples an end extended by [w] under [e] reads,
   [0] for none: a signal of at least that many reads the same extension
   whatever lies past them, with no fold. *)
let need (e : edge) w =
  if w = 0 then 0
  else
    match e with
    | `Zero -> 0
    | `Replicate -> 1
    | `Symmetric_half -> w
    | `Symmetric_whole | `Antisymmetric_whole -> w + 1

module Frames = struct
  type 'e t = { edge : 'e; ends : int * int; window : int; hop : int }

  let v ~edge ~ends ~window ~hop =
    let lo, hi = ends in
    if window < 1 || hop < 1 || lo < 0 || hi < 0 then
      invalid_argf "Omr.Frames.v: window %d, hop %d, ends (%d, %d)" window hop
        lo hi;
    { edge; ends; window; hop }

  (* [rule g] is [g]'s edge, among every edge. *)
  let rule (g : [< edge ] t) = (g.edge :> edge)

  let frame_count g n =
    let lo, hi = g.ends in
    let d = n + lo + hi - g.window in
    if d < 0 then 0 else (d / g.hop) + 1

  let count g n =
    if n < 0 then invalid_argf "Omr.Frames.count: %d samples" n;
    frame_count g n

  let extend g (lo, hi) x = Nx.extend (rule g :> Nx.extension) (lo, hi) x

  (* [span g x at k] is the samples of [x] that the [k] frames whose first
     starts at [at] read, along [x]'s last axis. *)
  let span g x at k =
    let len = if k = 0 then 0 else ((k - 1) * g.hop) + g.window in
    along x at (at + len)

  (* A {e cutter} makes what a stream gives from the [k] frames of a span: [cut
     g span k] is them, [drop first out] is [out] from frame [first]. *)
  type ('a, 'b) cutter = {
    cut : 'e. 'e t -> 'a -> int -> 'b;
    drop : int -> 'b -> 'b;
  }

  let framer =
    {
      cut = (fun g x _ -> Nx.sliding_window ~window:g.window ~step:g.hop x);
      drop =
        (fun first f ->
          Nx.slice
            (List.init (Nx.ndim f - 2) (fun _ -> Nx.A)
            @ [ Nx.R (first, Nx.dim (-2) f) ])
            f);
    }

  let cut c g x at k = c.cut g (span g x at k) k

  let whole_with ~by c g x =
    let lo, hi = g.ends in
    if Nx.dim (-1) x = 0 && rule g <> `Zero && lo + hi > 0 then
      invalid_argf "%s: no sample to read past the ends of %a" by pp_shape
        (Nx.shape x);
    let e = extend g (lo, hi) x in
    cut c g e 0 (frame_count g (Nx.dim (-1) x))

  let frames ?(axis = -1) g x =
    let by = "Omr.Frames.frames" in
    whole_with ~by framer g (time ~by axis x)

  (* A state holds the samples [x(s)], …, [x(r - 1)] that frames from the next
     on read, and the position of the next frame's start relative to [x(s)].
     [Head] holds the signal from its first sample, [s = 0]: until the frames
     that read before the start are out, and while the signal is shorter than
     the end's extension reads, which may then fold. *)
  module State = struct
    type ('a, _) t =
      | Start
      | Head of { samples : 'a; offset : int }
      | Running of { samples : 'a; offset : int }

    let walk c s =
      let open Nx.Ptree.Walk in
      match s with
      | Start ->
          case c "start";
          Start
      | Head { samples; offset } ->
          case c "head";
          let samples = field c "samples" leaf samples in
          let offset = field c "offset" int offset in
          Head { samples; offset }
      | Running { samples; offset } ->
          case c "running";
          let samples = field c "samples" leaf samples in
          let offset = field c "offset" int offset in
          Running { samples; offset }
  end

  let start = State.Start

  let join ~by kept x =
    if lead kept 1 <> lead x 1 then
      invalid_argf "%s: a chunk of %a after chunks of %a" by pp_shape
        (Nx.shape x) pp_shape (Nx.shape kept);
    Nx.concatenate ~axis:(-1) [ kept; x ]

  let feed_with ~by c g s x =
    let lo, hi = g.ends in
    let head, kept, o =
      match s with
      | State.Start -> (true, x, -lo)
      | Head { samples; offset } -> (true, join ~by samples x, offset)
      | Running { samples; offset } -> (false, join ~by samples x, offset)
    in
    let len = Nx.dim (-1) kept in
    (* Frames whose samples are all known, the first only once the start's
       extension is known if it reads before the start. *)
    let k =
      if (o < 0 && len < need (rule g) lo) || len - g.window - o < 0 then 0
      else ((len - g.window - o) / g.hop) + 1
    in
    let out =
      if o < 0 && k > 0 then cut c g (extend g (lo, 0) kept) (o + lo) k
      else cut c g kept (Int.max o 0) k
    in
    (* Keep from the next frame's start, and the last samples the end's
       extension reads; everything while frames before the start wait. *)
    let next = o + (k * g.hop) in
    let from = Int.max 0 (Int.min next (len - need (rule g) hi)) in
    let samples = along kept from len and offset = next - from in
    let s =
      if head && from = 0 then State.Head { samples; offset }
      else State.Running { samples; offset }
    in
    (s, out)

  let feed ?(axis = -1) g s x =
    let by = "Omr.Frames.feed" in
    feed_with ~by framer g s (time ~by axis x)

  let finish_with ~by c g s =
    let lo, hi = g.ends in
    match s with
    | State.Start -> invalid_argf "%s: no chunk was fed" by
    | Head { samples; offset }
      when Nx.dim (-1) samples < Int.max (need (rule g) lo) (need (rule g) hi)
      ->
        (* The whole signal, whose extension may fold: the whole call's frames
           from the next. *)
        c.drop ((offset + lo) / g.hop) (whole_with ~by c g samples)
    | Head { samples; offset } | Running { samples; offset } ->
        let before = if offset < 0 then lo else 0 in
        let e = extend g (before, hi) samples in
        let at = offset + before in
        let room = Nx.dim (-1) e - g.window - at in
        cut c g e at (if room < 0 then 0 else (room / g.hop) + 1)

  let finish g s = finish_with ~by:"Omr.Frames.finish" framer g s

  (* Overlap-add *)

  let check_frames ~by g f =
    if Nx.ndim f < 2 || Nx.dim (-1) f <> g.window then
      invalid_argf "%s: frames %a are not [...; frames; %d]" by pp_shape
        (Nx.shape f) g.window

  (* [ola g init at f] adds the frames [f], [[...; k; window]], onto the sums
     [init] along its last axis, frame [j] from position [at + j hop], and
     extends [init] with zeros as far as they reach. Position [q] in hop-sized
     block [b] holds segment [i] of frame [b - i], for [i < c = ⌈window / hop⌉]:
     adding the segments for [i = c - 1] down to [0] adds each position's frames
     in their order, frame by frame, in [c] passes. *)
  let ola g init at f =
    let w = g.window and h = g.hop in
    let c = (w + h - 1) / h and k = Nx.dim (-2) f and lead = lead f 2 in
    let len = Int.max (Nx.dim (-1) init) (at + ((k + c - 1) * h)) in
    let f = fit f (c * h) in
    let f = Nx.reshape (Array.append lead [| k; c; h |]) f in
    let acc = ref (fit init len) in
    for i = c - 1 downto 0 do
      let t =
        Nx.slice
          (List.init (Array.length lead + 1) (fun _ -> Nx.A) @ [ Nx.I i ])
          f
      in
      let t = Nx.reshape (Array.append lead [| k * h |]) t in
      let before = at + (i * h) in
      let t =
        Nx.pad
          (Array.init (Nx.ndim t) (fun a ->
               if a = Nx.ndim t - 1 then (before, len - before - (k * h))
               else (0, 0)))
          0. t
      in
      acc := Nx.add !acc t
    done;
    !acc

  (* [reads e n p] is the samples that position [p] of a signal of [n > 0]
     samples reads under [e], each with its coefficient: the rule's equation,
     read again at each position it reads outside the signal, as {!Nx.extend}
     does. Under [`Antisymmetric_whole] position [-i] reads [2 x(0) - x(i)]. *)
  let rec reads (e : edge) n p =
    if 0 <= p && p < n then [ (p, 1) ]
    else if n = 1 then if e = `Zero then [] else [ (0, 1) ]
    else
      match e with
      | `Zero -> []
      | `Replicate -> [ ((if p < 0 then 0 else n - 1), 1) ]
      | `Symmetric_whole -> reads e n (if p < 0 then -p else (2 * (n - 1)) - p)
      | `Symmetric_half -> reads e n (if p < 0 then -p - 1 else (2 * n) - 1 - p)
      | `Antisymmetric_whole ->
          let at, mirror =
            if p < 0 then (0, -p) else (n - 1, (2 * (n - 1)) - p)
          in
          (at, 2) :: List.map (fun (i, c) -> (i, -c)) (reads e n mirror)

  (* [merged terms] is [terms] with one coefficient per sample, in order of
     sample. *)
  let merged terms =
    let add acc (i, c) =
      match acc with
      | (j, d) :: rest when i = j -> (j, c + d) :: rest
      | _ -> (i, c) :: acc
    in
    List.rev (List.fold_left add [] (List.sort compare terms))

  (* [fold g (lo, hi) n o] is the adjoint of extending [n] samples by [(lo, hi)]
     under [g]'s edge, of [o], [[...; lo + n + hi]]: sample [i] is the sum, from
     [+0], of [c] times [o] at every position that reads [x(i)] with coefficient
     [c], in order of position. Round [r] adds the [r]-th term of every sample
     that has one. *)
  let fold g (lo, hi) n o =
    if rule g = `Zero || n = 0 || lo + hi = 0 then along o lo (lo + n)
    else begin
      let terms = Array.make n [] in
      for q = lo + n + hi - 1 downto 0 do
        List.iter
          (fun (i, c) -> terms.(i) <- (q, c) :: terms.(i))
          (merged (reads (rule g) n (q - lo)))
      done;
      let at = Nx.beside o in
      let ints a =
        Nx.place at
          (Nx.create Nx.int64 [| Array.length a |] (Array.map Int64.of_int a))
      in
      (* [term p c] is [c] times [o] at the positions [p]. *)
      let term p c =
        let t = Nx.take ~axis:(-1) (ints p) o in
        if Array.for_all (( = ) 1) c then t
        else
          let c = Array.map Float.of_int c in
          let c = Nx.create Nx.float64 [| Array.length c |] c in
          Nx.mul t (Nx.place at (Nx.cast (Nx.dtype o) c))
      in
      let firsts = Array.map List.hd terms in
      let acc =
        ref (Nx.add_s (term (Array.map fst firsts) (Array.map snd firsts)) 0.)
      in
      let rest = ref (Array.map List.tl terms) in
      while Array.exists (fun l -> l <> []) !rest do
        let next = ref [] in
        for i = n - 1 downto 0 do
          match !rest.(i) with
          | (q, c) :: tl ->
              next := (i, q, c) :: !next;
              !rest.(i) <- tl
          | [] -> ()
        done;
        let next = Array.of_list !next in
        let p = ints (Array.map (fun (i, _, _) -> i) next) in
        let sum =
          Nx.add
            (Nx.take ~axis:(-1) p !acc)
            (term
               (Array.map (fun (_, q, _) -> q) next)
               (Array.map (fun (_, _, c) -> c) next))
        in
        acc := Nx.scatter ~axis:(-1) (Nx.broadcast_to (Nx.shape sum) p) sum !acc
      done;
      !acc
    end

  (* [added ~by g n f] is the overlap-add of [f] to [n] samples, last. *)
  let added ~by g n f =
    if n < 0 then invalid_argf "%s: %d samples" by n;
    check_frames ~by g f;
    if Nx.dim (-2) f <> frame_count g n then
      invalid_argf "%s: %d frames for %d samples, which have %d" by
        (Nx.dim (-2) f) n (frame_count g n);
    let lo, hi = g.ends in
    let none =
      Nx.zeros_at (Nx.beside f) (Nx.dtype f) (Array.append (lead f 2) [| 0 |])
    in
    fold g (lo, hi) n (fit (ola g none 0 f) (lo + n + hi))

  let overlap_add ?(axis = -1) g n f =
    let by = "Omr.Frames.overlap_add" in
    untime ~by axis (added ~by g n f)

  module Overlap_add = struct
    (* [Head] holds the sums of every position from the extended signal's start,
       while the start's extension may still read samples to come. [Running]
       holds the sums from the first sample not yet out: [held g] final ones,
       then the [partial g] that later frames still reach. *)
    module State = struct
      type ('a, _) t =
        | Start
        | Head of { sums : 'a; frames : int }
        | Running of { sums : 'a }

      let walk c s =
        let open Nx.Ptree.Walk in
        match s with
        | Start ->
            case c "start";
            Start
        | Head { sums; frames } ->
            case c "head";
            let sums = field c "sums" leaf sums in
            let frames = field c "frames" int frames in
            Head { sums; frames }
        | Running { sums } ->
            case c "running";
            let sums = field c "sums" leaf sums in
            Running { sums }
    end

    let start = State.Start
    let partial g = Int.max 0 (g.window - g.hop)

    (* Past the next frame's start, a stream holds back the samples a position
       past the end may still read: a signal of [m] frames has at least [(m - 1)
       hop + window - lo - hi] samples. *)
    let held g =
      let _, hi = g.ends in
      Int.max 0 (g.hop - g.window + hi + need (rule g) hi)

    let feed_last ~by g s f =
      check_frames ~by g f;
      let lo, hi = g.ends and h = g.hop in
      let k = Nx.dim (-2) f in
      let same sums =
        if lead sums 1 <> lead f 2 then
          invalid_argf "%s: frames %a after sums %a" by pp_shape (Nx.shape f)
            pp_shape (Nx.shape sums)
      in
      match s with
      | State.Running { sums } ->
          same sums;
          let acc = ola g sums (held g) f in
          let rest = held g + partial g in
          ( State.Running { sums = along acc (k * h) ((k * h) + rest) },
            along acc 0 (k * h) )
      | Start | Head _ ->
          let m0, init =
            match s with
            | State.Head { sums; frames } ->
                same sums;
                (frames, sums)
            | _ ->
                ( 0,
                  Nx.zeros_at (Nx.beside f) (Nx.dtype f)
                    (Array.append (lead f 2) [| 0 |]) )
          in
          let m = m0 + k in
          let acc = fit (ola g init (m0 * h) f) ((m * h) + partial g) in
          (* The samples out once the start's extension is read: below both the
             first position a later frame reaches and the first an end may
             read. *)
          let e =
            Int.min
              ((m * h) - lo)
              (((m - 1) * h) + g.window - lo - hi - need (rule g) hi)
          in
          if m > 0 && e >= Int.max 0 (need (rule g) lo) then
            ( State.Running { sums = along acc (lo + e) ((m * h) + partial g) },
              fold g (lo, 0) e (along acc 0 (lo + e)) )
          else (State.Head { sums = acc; frames = m }, along acc 0 0)

    (* A signal of [m ≥ 1] frames has [(m - 1) hop + window - lo - hi + extra]
       samples for [0 ≤ extra < hop], and one of none has [extra] samples for [0
       ≤ extra < window - lo - hi]. *)
    let finish_last ~by g ~extra s =
      let lo, hi = g.ends and h = g.hop and w = g.window in
      let check range =
        if extra < 0 || extra >= range then
          invalid_argf "%s: extra %d is outside [0, %d)" by extra range
      in
      match s with
      | State.Start ->
          invalid_argf "%s: no chunk was fed; feed an empty chunk first" by
      | Head { sums; frames = 0 } ->
          check (w - lo - hi);
          fold g (lo, hi) extra (fit sums (lo + extra + hi))
      | Head { sums; frames = m } ->
          check h;
          let n = ((m - 1) * h) + w - lo - hi + extra in
          if n < 0 then
            invalid_argf "%s: %d frames and extra %d make %d samples" by m extra
              n;
          fold g (lo, hi) n (fit sums (lo + n + hi))
      | Running { sums } ->
          check h;
          (* The samples from the first not out: [n - e] for [e] as [feed] gives
             it after [m] frames. *)
          let rest = w - h - hi + held g + extra in
          fold g (0, hi) rest (fit sums (rest + hi))

    let feed ?(axis = -1) g s f =
      let by = "Omr.Frames.Overlap_add.feed" in
      let s, x = feed_last ~by g s f in
      (s, untime ~by axis x)

    let finish ?(axis = -1) g ~extra s =
      let by = "Omr.Frames.Overlap_add.finish" in
      untime ~by axis (finish_last ~by g ~extra s)
  end
end

(* Short-time transforms *)

type invertible = [ `Zero | `Replicate | `Symmetric_whole | `Symmetric_half ]

let check_win ~by (g : _ Frames.t) win =
  if Nx.shape win <> [| g.window |] then
    invalid_argf "%s: win %a is not [%d]" by pp_shape (Nx.shape win) g.window

let check_bins ~by (g : _ Frames.t) z =
  if Nx.ndim z < 2 || Nx.dim (-1) z <> (g.window / 2) + 1 then
    invalid_argf "%s: spectra %a are not [...; frames; %d]" by pp_shape
      (Nx.shape z)
      ((g.window / 2) + 1)

let stft ?axis dt g ~win x =
  check_win ~by:"Omr.stft" g win;
  Nx.rfft dt (Nx.mul (Frames.frames ?axis g x) win)

(* [windowed dt g win z] is each spectrum's frame, times [win]. *)
let windowed dt (g : _ Frames.t) win z =
  Nx.mul (Nx.irfft dt ~s:[ g.window ] z) win

(* [envelope g win k] is [k] frames of [win²]. *)
let envelope (g : _ Frames.t) win k =
  Nx.broadcast_to [| k; g.window |] (Nx.square win)

(* [divide x env] is [x / env], [0.] where [env] is zero. *)
let divide x env =
  Nx.where (Nx.equal_s env 0.) (Nx.zeros_like env) (Nx.div x env)

let istft ?(axis = -1) dt (g : [< invertible ] Frames.t) ~win n z =
  let by = "Omr.istft" in
  check_win ~by g win;
  check_bins ~by g z;
  let x = Frames.added ~by g n (windowed dt g win z) in
  let env = Frames.added ~by g n (envelope g win (Nx.dim (-2) z)) in
  untime ~by axis (divide x env)

module Istft = struct
  module Ola = Frames.Overlap_add

  module State = struct
    type ('a, 'd) t = { num : ('a, 'd) Ola.State.t; env : ('a, 'd) Ola.State.t }

    let walk c { num; env } =
      let open Nx.Ptree.Walk in
      let num = field c "num" Ola.State.walk num in
      let env = field c "env" Ola.State.walk env in
      { num; env }
  end

  let start = { State.num = Ola.start; env = Ola.start }

  let feed ?(axis = -1) dt (g : [< invertible ] Frames.t) ~win (s : _ State.t) z
      =
    let by = "Omr.Istft.feed" in
    check_win ~by g win;
    check_bins ~by g z;
    let num, x = Ola.feed_last ~by g s.num (windowed dt g win z) in
    let env, e = Ola.feed_last ~by g s.env (envelope g win (Nx.dim (-2) z)) in
    ({ State.num; env }, untime ~by axis (divide x e))

  let finish ?(axis = -1) (g : [< invertible ] Frames.t) ~extra (s : _ State.t)
      =
    let by = "Omr.Istft.finish" in
    let x = Ola.finish_last ~by g ~extra s.num in
    untime ~by axis (divide x (Ola.finish_last ~by g ~extra s.env))
end

(* Spectral scaling *)

let check_rate ~by rate =
  if not (Float.is_finite rate && rate > 0.) then
    invalid_argf "%s: rate %g" by rate

(* [one_sided ~by ?n ~win p] is [c_k p_k]: bins [1] to [n / 2], [n / 2] excluded
   for an even [n], doubled. *)
let one_sided ~by ?n ~win p =
  if Nx.ndim win <> 1 || Nx.dim 0 win = 0 then
    invalid_argf "%s: win %a is not a window" by pp_shape (Nx.shape win);
  let n = Option.value n ~default:(Nx.dim 0 win) in
  if n < Nx.dim 0 win then
    invalid_argf "%s: %d points for a window of %d" by n (Nx.dim 0 win);
  let bins = (n / 2) + 1 in
  if Nx.ndim p = 0 || Nx.dim (-1) p <> bins then
    invalid_argf "%s: p %a has not %d bins" by pp_shape (Nx.shape p) bins;
  let last = if n mod 2 = 0 then bins - 1 else bins in
  Nx.concatenate ~axis:(-1)
    [ along p 0 1; Nx.mul_s (along p 1 last) 2.; along p last bins ]

let psd ?n ~rate ~win p =
  let by = "Omr.psd" in
  check_rate ~by rate;
  let p = one_sided ~by ?n ~win p in
  Nx.div p (Nx.mul_s (Nx.sum (Nx.square win)) rate)

let power_spectrum ?n ~win p =
  let p = one_sided ~by:"Omr.power_spectrum" ?n ~win p in
  Nx.div p (Nx.square (Nx.sum win))

(* Filterbanks *)

type norm = Unit_peak | Unit_area

let triangles nm ~at e =
  let by = "Omr.triangles" in
  if Nx.ndim at <> 1 || Nx.ndim e <> 1 || Nx.dim 0 e < 3 then
    invalid_argf "%s: points %a and edges %a" by pp_shape (Nx.shape at) pp_shape
      (Nx.shape e);
  let b = Nx.dim 0 e - 2 in
  let p = Nx.reshape [| Nx.dim 0 at; 1 |] at in
  let lower = Nx.slice [ R (0, b) ] e
  and centre = Nx.slice [ R (1, b + 1) ] e
  and upper = Nx.slice [ R (2, b + 2) ] e in
  let rise = Nx.div (Nx.sub p lower) (Nx.sub centre lower) in
  let fall = Nx.div (Nx.sub upper p) (Nx.sub upper centre) in
  let w = Nx.maximum_s (Nx.minimum rise fall) 0. in
  match nm with
  | Unit_peak -> w
  | Unit_area -> Nx.mul w (Nx.rdiv_s 2. (Nx.sub upper lower))

module Mel = struct
  type scale = Slaney | Htk

  (* ln 6.4, correctly rounded; Slaney's scale gains a mel every [ln 6.4 / 27]
     of log frequency from 1000 Hz, at 15 mel. *)
  let ln_6_4 = 1.8562979903656263
  let slaney_step = ln_6_4 /. 27.

  let of_hz sc f =
    match sc with
    | Htk -> Nx.mul_s (Nx.log1p (Nx.div_s f 700.)) 1127.
    | Slaney ->
        let linear = Nx.div_s (Nx.mul_s f 3.) 200. in
        let log =
          (* At [f] from 1000 Hz, read so that the branch [where] drops has a
             finite derivative. *)
          let f = Nx.maximum_s f 1000. in
          Nx.add_s (Nx.div_s (Nx.log (Nx.div_s f 1000.)) slaney_step) 15.
        in
        Nx.where (Nx.less_s f 1000.) linear log

  let to_hz sc m =
    match sc with
    | Htk -> Nx.mul_s (Nx.expm1 (Nx.div_s m 1127.)) 700.
    | Slaney ->
        let linear = Nx.div_s (Nx.mul_s m 200.) 3. in
        let log =
          Nx.mul_s (Nx.exp (Nx.mul_s (Nx.sub_s m 15.) slaney_step)) 1000.
        in
        Nx.where (Nx.less_s m 15.) linear log

  (* [hz sc ~low ~high k] is the float64 edges, [low] and [high] exact. *)
  let hz ~by sc ~low ~high k =
    if k < 2 || not (0. <= low && low < high && Float.is_finite high) then
      invalid_argf "%s: %d edges from %g to %g" by k low high;
    let ends = of_hz sc (Nx.create Nx.float64 [| 2 |] [| low; high |]) in
    let lo = Nx.item [ 0 ] ends and hi = Nx.item [ 1 ] ends in
    let step = (hi -. lo) /. Float.of_int (k - 1) in
    let m = Nx.add_s (Nx.mul_s (Nx.arange Nx.float64 0 k 1) step) lo in
    let f = to_hz sc m in
    Nx.concatenate ~axis:0
      [
        Nx.create Nx.float64 [| 1 |] [| low |];
        Nx.slice [ R (1, k - 1) ] f;
        Nx.create Nx.float64 [| 1 |] [| high |];
      ]

  let edges dt sc ~low ~high k =
    Nx.cast dt (hz ~by:"Omr.Mel.edges" sc ~low ~high k)

  let bank dt sc nm ~rate ~n ?(low = 0.) ?high b =
    let by = "Omr.Mel.bank" in
    check_rate ~by rate;
    let high = Option.value high ~default:(rate /. 2.) in
    if n < 2 || b < 1 then invalid_argf "%s: %d points, %d bands" by n b;
    if not (0. <= low && low < high && high <= rate /. 2.) then
      invalid_argf "%s: bands from %g to %g at rate %g" by low high rate;
    let e = hz ~by sc ~low ~high (b + 2) in
    (* [k rate / n], one rounding. *)
    let at =
      Nx.div_s
        (Nx.mul_s (Nx.arange Nx.float64 0 ((n / 2) + 1) 1) rate)
        (Float.of_int n)
    in
    let w = triangles nm ~at e in
    let peaks = Nx.to_array (Nx.max ~axes:[ 0 ] w) in
    Array.iteri
      (fun j p ->
        if not (p > 0.) then invalid_argf "%s: band %d weighs no bin" by j)
      peaks;
    Nx.cast dt w
end

(* IIR filters *)

(* A recursion's blocks: 4096 samples, nx.cpu's Sum-scan chunk. *)
let block = 4096

(* [complex_of dt k] is [k.f] at the complex dtype whose parts have [dt]'s
   precision, the formats narrower than float64 at complex64. *)
type 'r complex_k = { f : 'c. (Complex.t, 'c) Nx.dtype -> 'r }

let complex_of (type s) (dt : (float, s) Nx.dtype) k =
  match dt with Nx.Dtype.Float64 -> k.f Nx.complex128 | _ -> k.f Nx.complex64

let check_cascade ~by c =
  if Nx.ndim c <> 2 || Nx.dim 1 c <> 6 || Nx.dim 0 c < 1 then
    invalid_argf "%s: cascade %a is not [sections; 6]" by pp_shape (Nx.shape c)

(* [sample x k] is position [k] of [x]'s last axis, without that axis. *)
let sample x k =
  Nx.slice (List.init (Nx.ndim x - 1) (fun _ -> Nx.A) @ [ Nx.I k ]) x

let column x = Nx.unsqueeze ~axes:[ -1 ] x

(* [first_order r v] is [wₙ = r wₙ₋₁ + vₙ] from rest along [v]'s last axis: the
   scan of the affine maps [w ↦ r w + vₙ]. *)
let first_order r v =
  let a = Nx.broadcast_to [| Nx.dim (-1) v |] r in
  let combine (a1, b1) (a2, b2) = (Nx.mul a2 a1, Nx.add (Nx.mul a2 b1) b2) in
  snd
    (Nx.associative_scan ~axis:(-1)
       Nx.Ptree.(pair tensor tensor)
       combine (a, v))

(* [poles a1 a2 v] is [yₙ = vₙ - a1 yₙ₋₁ - a2 yₙ₋₂] from rest along [v]'s last
   axis, as [(1 - p z⁻¹)⁻¹] then [(1 - q z⁻¹)⁻¹] over complex numbers: [p] the
   root of [z² + a1 z + a2] of the larger modulus, from [a1² - 4 a2] rounded
   once, and [q = a2 / p], [0] where [p] is. *)
let poles a1 a2 v =
  let dt = Nx.dtype v in
  complex_of dt
    {
      f =
        (fun cdt ->
          let c x = Nx.cast cdt x in
          let d = Nx.fma a1 a1 (Nx.mul_s a2 (-4.)) in
          let s =
            Nx.where (Nx.greater_equal_s a1 0.) (Nx.scalar_like a1 1.)
              (Nx.scalar_like a1 (-1.))
          in
          let p =
            Nx.mul_s
              (Nx.sub (Nx.neg (c a1)) (Nx.mul (c s) (Nx.sqrt (c d))))
              Complex.{ re = 0.5; im = 0. }
          in
          let zero = Nx.equal_s p Complex.zero in
          let q =
            Nx.where zero (Nx.zeros_like p)
              (Nx.div (c a2) (Nx.where zero (Nx.ones_like p) p))
          in
          Nx.real dt (first_order q (first_order p (c v))));
    }

(* [section_block a1 a2 (y1, y2) v] is one block of the recursion [yₙ = vₙ - a1
   yₙ₋₁ - a2 yₙ₋₂] after the outputs [y1] and [y2], [[lead]] each, and its last
   two outputs. The state enters through [v]'s first two samples, as the
   recursion defines them there. A block of fewer than three samples is the
   recursion itself, a longer one {!poles} from rest. *)
let section_block a1 a2 (y1, y2) v =
  let l = Nx.dim (-1) v in
  let v0 =
    Nx.add (sample v 0) (Nx.sub (Nx.mul (Nx.neg a1) y1) (Nx.mul a2 y2))
  in
  if l = 1 then ((v0, y1), column v0)
  else
    let v1 = Nx.add (sample v 1) (Nx.mul (Nx.neg a2) y1) in
    if l = 2 then
      let o1 = Nx.add v1 (Nx.mul (Nx.neg a1) v0) in
      ((o1, v0), Nx.concatenate ~axis:(-1) [ column v0; column o1 ])
    else
      let v = Nx.concatenate ~axis:(-1) [ column v0; column v1; along v 2 l ] in
      let y = poles a1 a2 v in
      ((sample y (l - 1), sample y (l - 2)), y)

(* [blocked a1 a2 y1 y2 v] is the recursion over [v], [[lead; n]], after the
   outputs [y1] and [y2], in blocks from [v]'s start folded in order, the whole
   ones by a scan; [a1] and [a2] are 0-d. *)
let blocked a1 a2 y1 y2 v =
  let n = Nx.dim (-1) v and lead = lead v 1 in
  let k = n / block in
  let s, whole =
    if k = 0 then ((y1, y2), [])
    else
      let blocks =
        Nx.moveaxis (-2) 0
          (Nx.reshape
             (Array.append lead [| k; block |])
             (along v 0 (k * block)))
      in
      let s, ys =
        Rune.scan
          Nx.Ptree.(pair tensor tensor)
          Nx.Ptree.tensor Nx.Ptree.tensor (section_block a1 a2) ~init:(y1, y2)
          blocks
      in
      let ys = Nx.moveaxis 0 (-2) ys in
      (s, [ Nx.reshape (Array.append lead [| k * block |]) ys ])
  in
  let last =
    if n = k * block then []
    else [ snd (section_block a1 a2 s (along v (k * block) n)) ]
  in
  match whole @ last with [] -> v | ys -> Nx.concatenate ~axis:(-1) ys

(* [delayed s y] is [y] delayed by the samples [s], the nearest last. *)
let delayed s y =
  along
    (Nx.concatenate ~axis:(-1) (List.map column s @ [ y ]))
    0 (Nx.dim (-1) y)

(* [linear a1 a2 (y1, y2, v)] is the recursion from the state [(y1, y2)] over
   [v], linear in both. Its transpose filters the cotangent [w] backward in time
   from rest, [g], which is [v]'s cotangent; the state entered as [v₀ += -a1 y₁
   - a2 y₂] and [v₁ += -a2 y₁], so its cotangents are [-a1 g₀ - a2 g₁] and [-a2
   g₀]. *)
let linear a1 a2 args =
  let transpose w =
    let n = Nx.dim (-1) w in
    let z = Nx.zeros_at (Nx.beside w) (Nx.dtype w) (lead w 1) in
    let rev = Nx.flip ~axes:[ -1 ] in
    let g = if n = 0 then w else rev (blocked a1 a2 z z (rev w)) in
    let at k = if n > k then sample g k else z in
    ( Nx.sub (Nx.mul (Nx.neg a1) (at 0)) (Nx.mul a2 (at 1)),
      Nx.mul (Nx.neg a2) (at 0),
      g )
  in
  Rune.custom_transpose
    Nx.Ptree.(triple tensor tensor tensor)
    Nx.Ptree.tensor
    (fun (y1, y2, v) -> blocked a1 a2 y1 y2 v)
    ~transpose args

(* [allpole a1 a2 y1 y2 v] is {!blocked}'s recursion with the derivative of the
   transfer function, [d(v / A) = (dv - dA y) / A], whose linear part is
   {!linear}. The first derivative differentiates no pole; a second one
   differentiates the [y] the tangent map reads, which {!blocked} computes
   through the poles. *)
let allpole a1 a2 y1 y2 v =
  Rune.custom_jvp
    Nx.Ptree.(pair (pair tensor tensor) (triple tensor tensor tensor))
    Nx.Ptree.tensor
    (fun ((a1, a2), (y1, y2, v)) ->
      let y = blocked a1 a2 y1 y2 v in
      let tangent ((da1, da2), (dy1, dy2, dv)) =
        let du =
          Nx.sub
            (Nx.sub dv (Nx.mul da1 (delayed [ y1 ] y)))
            (Nx.mul da2 (delayed [ y2; y1 ] y))
        in
        linear a1 a2 (dy1, dy2, du)
      in
      (y, tangent))
    ((a1, a2), (y1, y2, v))

module Sos = struct
  module State = struct
    type ('a, _) t = Rest | At of { inputs : 'a; outputs : 'a }

    let walk c s =
      let open Nx.Ptree.Walk in
      match s with
      | Rest ->
          case c "rest";
          Rest
      | At { inputs; outputs } ->
          case c "at";
          let inputs = field c "inputs" leaf inputs in
          let outputs = field c "outputs" leaf outputs in
          At { inputs; outputs }
  end

  let start = State.Rest

  (* [rows c] is [c]'s rows divided by their [a0], rounded once. *)
  let rows c = Nx.div c (Nx.slice [ Nx.A; Nx.R (3, 4) ] c)
  let coef r s j = Nx.slice [ Nx.I s; Nx.I j ] r

  (* [prepend h x] is [x] after the last two samples [h], [[lead; 2]] most
     recent first. *)
  let prepend h x =
    Nx.concatenate ~axis:(-1) [ column (sample h 1); column (sample h 0); x ]

  (* [latest x] is [x]'s last two samples, most recent first, [x] holding at
     least two. *)
  let latest x =
    let n = Nx.dim (-1) x in
    Nx.stack ~axis:(-1) [ sample x (n - 1); sample x (n - 2) ]

  let feed_last ~by c s x =
    check_cascade ~by c;
    let lead = lead x 1 and n = Nx.dim (-1) x and k = Nx.dim 0 c in
    let zeros shape = Nx.zeros_at (Nx.beside x) (Nx.dtype x) shape in
    let inputs, outputs =
      match s with
      | State.Rest ->
          ( zeros (Array.append lead [| 2 |]),
            zeros (Array.append lead [| k; 2 |]) )
      | At { inputs; outputs } ->
          if
            Nx.shape inputs <> Array.append lead [| 2 |]
            || Nx.shape outputs <> Array.append lead [| k; 2 |]
          then
            invalid_argf
              "%s: a state of %a and %a for a chunk of %a and %d sections" by
              pp_shape (Nx.shape inputs) pp_shape (Nx.shape outputs) pp_shape
              (Nx.shape x) k;
          (inputs, outputs)
    in
    let r = rows c in
    let past s =
      Nx.slice
        (List.init (Array.length lead) (fun _ -> Nx.A) @ [ Nx.I s; Nx.A ])
        outputs
    in
    let u = ref x and hist = ref inputs and states = ref [] in
    for s = 0 to k - 1 do
      let e = prepend !hist !u in
      (* [b0 xₙ + b1 xₙ₋₁ + b2 xₙ₋₂], added in that order. *)
      let v =
        Nx.add
          (Nx.add
             (Nx.mul (coef r s 0) (along e 2 (n + 2)))
             (Nx.mul (coef r s 1) (along e 1 (n + 1))))
          (Nx.mul (coef r s 2) (along e 0 n))
      in
      let ys = past s in
      let y = allpole (coef r s 4) (coef r s 5) (sample ys 0) (sample ys 1) v in
      states := latest (prepend ys y) :: !states;
      hist := past s;
      u := y
    done;
    let inputs = latest (prepend inputs x) in
    let outputs = Nx.stack ~axis:(-2) (List.rev !states) in
    (State.At { inputs; outputs }, !u)

  let feed ?(axis = -1) c s x =
    let by = "Omr.Sos.feed" in
    let s, y = feed_last ~by c s (time ~by axis x) in
    (s, untime ~by axis y)

  let steady c u =
    check_cascade ~by:"Omr.Sos.steady" c;
    let r = rows c and k = Nx.dim 0 c in
    let gain s =
      let ( + ) = Nx.add in
      Nx.div
        (coef r s 0 + coef r s 1 + coef r s 2)
        (coef r s 3 + coef r s 4 + coef r s 5)
    in
    let g = ref (gain 0) and outs = ref [] in
    for s = 0 to k - 1 do
      if s > 0 then g := Nx.mul !g (gain s);
      let o = Nx.mul u !g in
      outs := Nx.stack ~axis:(-1) [ o; o ] :: !outs
    done;
    State.At
      {
        inputs = Nx.stack ~axis:(-1) [ u; u ];
        outputs = Nx.stack ~axis:(-2) (List.rev !outs);
      }

  let stable c =
    check_cascade ~by:"Omr.Sos.stable" c;
    let r = rows c in
    let col j = Nx.slice [ Nx.A; Nx.I j ] r in
    let a1 = col 4 and a2 = col 5 in
    Nx.logical_and
      (Nx.less_s (Nx.abs a2) 1.)
      (Nx.less (Nx.abs a1) (Nx.add_s a2 1.))
end

let sos ?(axis = -1) c x =
  let by = "Omr.sos" in
  untime ~by axis (snd (Sos.feed_last ~by c Sos.start (time ~by axis x)))

(* Designs *)

module Biquad = struct
  (* [rows cols] is the six coefficient arrays [cols], broadcast together and
     stacked along a last axis. *)
  let rows cols = Nx.stack ~axis:(-1) (Nx.broadcast_arrays cols)

  (* [angle ~rate freq] is [cos w], [sin w] for [w = 2π freq / rate]. *)
  let angle ~rate freq =
    let w = Nx.mul_s freq (2. *. Float.pi /. rate) in
    (Nx.cos w, Nx.sin w)

  let alpha s q = Nx.div s (Nx.mul_s q 2.)
  let ( + ) = Nx.add
  let ( - ) = Nx.sub
  let ( * ) = Nx.mul

  (* The cookbook's denominator, [1 + α], [-2 cos w], [1 - α]. *)
  let poles c a =
    let one = Nx.ones_like a in
    [ one + a; Nx.mul_s c (-2.); one - a ]

  let lowpass ~rate ~freq ~q =
    let c, s = angle ~rate freq in
    let a = alpha s q in
    let one = Nx.ones_like c in
    let h = Nx.mul_s (one - c) 0.5 in
    rows ([ h; one - c; h ] @ poles c a)

  let highpass ~rate ~freq ~q =
    let c, s = angle ~rate freq in
    let a = alpha s q in
    let one = Nx.ones_like c in
    let h = Nx.mul_s (one + c) 0.5 in
    rows ([ h; Nx.neg (one + c); h ] @ poles c a)

  let bandpass ~rate ~freq ~q =
    let c, s = angle ~rate freq in
    let a = alpha s q in
    rows ([ a; Nx.zeros_like a; Nx.neg a ] @ poles c a)

  let notch ~rate ~freq ~q =
    let c, s = angle ~rate freq in
    let a = alpha s q in
    let one = Nx.ones_like c in
    rows ([ one; Nx.mul_s c (-2.); one ] @ poles c a)

  let allpass ~rate ~freq ~q =
    let c, s = angle ~rate freq in
    let a = alpha s q in
    let one = Nx.ones_like a in
    rows ([ one - a; Nx.mul_s c (-2.); one + a ] @ poles c a)

  (* [amplitude gain] is [A = 10^(gain / 40)]. *)
  let amplitude gain = Nx.exp (Nx.mul_s gain (Float.log 10. /. 40.))

  let peaking ~rate ~freq ~q ~gain =
    let c, s = angle ~rate freq in
    let a = alpha s q and amp = amplitude gain in
    let one = Nx.ones_like a in
    let m = Nx.mul_s c (-2.) in
    rows
      [
        one + (a * amp);
        m;
        one - (a * amp);
        one + Nx.div a amp;
        m;
        one - Nx.div a amp;
      ]

  (* A shelf's terms: [A + 1], [A - 1], [2 √A α]. *)
  let shelf ~rate ~freq ~q ~gain =
    let c, s = angle ~rate freq in
    let amp = amplitude gain in
    let one = Nx.ones_like amp in
    (c, amp, amp + one, amp - one, Nx.mul_s (Nx.sqrt amp * alpha s q) 2.)

  let low_shelf ~rate ~freq ~q ~gain =
    let c, amp, p, m, r = shelf ~rate ~freq ~q ~gain in
    rows
      [
        amp * (p - (m * c) + r);
        Nx.mul_s (amp * (m - (p * c))) 2.;
        amp * (p - (m * c) - r);
        p + (m * c) + r;
        Nx.mul_s (m + (p * c)) (-2.);
        p + (m * c) - r;
      ]

  let high_shelf ~rate ~freq ~q ~gain =
    let c, amp, p, m, r = shelf ~rate ~freq ~q ~gain in
    rows
      [
        amp * (p + (m * c) + r);
        Nx.mul_s (amp * (m + (p * c))) (-2.);
        amp * (p + (m * c) - r);
        p - (m * c) + r;
        Nx.mul_s (m - (p * c)) 2.;
        p - (m * c) - r;
      ]
end

(* Host float64 arithmetic by nx operations, so that a design has the same bits
   on every machine. *)
module H = struct
  let f1 op x = Nx.item [] (op (Nx.scalar Nx.float64 x))

  let f2 op x y =
    Nx.item [] (op (Nx.scalar Nx.float64 x) (Nx.scalar Nx.float64 y))

  let sin = f1 Nx.sin
  let cos = f1 Nx.cos
  let tan = f1 Nx.tan
  let sinh = f1 Nx.sinh
  let cosh = f1 Nx.cosh
  let asinh = f1 Nx.asinh
  let expm1 = f1 Nx.expm1
  let exp = f1 Nx.exp
  let log = f1 Nx.log
  let hypot = f2 Nx.hypot
  let atan2 = f2 Nx.atan2
  let ( ** ) = f2 Nx.pow
  let pi = Float.pi

  (* [pow10m1 x] is [10^x - 1]. *)
  let pow10m1 x = expm1 (x *. 2.302585092994046)

  (* Complex numbers *)

  let c re im = Complex.{ re; im }
  let re r = c r 0.
  let ( +: ) = Complex.add
  let ( -: ) = Complex.sub
  let ( *: ) = Complex.mul
  let ( /: ) = Complex.div
  let abs z = hypot z.Complex.re z.im

  let sqrt (z : Complex.t) =
    if z.re = 0. && z.im = 0. then c 0. z.im
    else
      let m = abs z in
      let a = Float.sqrt ((m +. Float.abs z.re) /. 2.) in
      if z.re >= 0. then c a (z.im /. (2. *. a))
      else c (Float.abs z.im /. (2. *. a)) (Float.copy_sign a z.im)

  let log (z : Complex.t) = c (log (abs z)) (atan2 z.im z.re)

  (* [cos], [sin] and [asin] of a complex number. *)
  let ccos (z : Complex.t) =
    c (cos z.re *. cosh z.im) (-.(sin z.re *. sinh z.im))

  let csin (z : Complex.t) = c (sin z.re *. cosh z.im) (cos z.re *. sinh z.im)

  let casin z =
    (* -i log (i z + √(1 - z²)) *)
    let i = c 0. 1. in
    let l = log ((i *: z) +: sqrt (re 1. -: (z *: z))) in
    c l.im (-.l.re)

  (* Scaled numbers *)

  (* [{ m; e }] is [m 2ᵉ], [m]'s larger part in [\[½, 1)] or [m] zero or not
     finite: a gain that multiplies hundreds of factors, as a design of high
     order or narrow band does, leaves float64's range though each section's
     share stays in it. *)
  type scaled = { m : Complex.t; e : int }

  let scale (z : Complex.t) e =
    let a = Float.max (Float.abs z.re) (Float.abs z.im) in
    if a = 0. || not (Float.is_finite a) then { m = z; e }
    else
      let _, x = Float.frexp a in
      { m = c (Float.ldexp z.re (-x)) (Float.ldexp z.im (-x)); e = e + x }

  let scaled z = scale z 0
  let ( *& ) a b = scale (a.m *: b.m) (a.e + b.e)
  let ( /& ) a b = scale (a.m /: b.m) (a.e - b.e)
  let sprod l = List.fold_left (fun a z -> a *& scaled z) (scaled (re 1.)) l
  let spow x n = sprod (List.init n (fun _ -> re x))

  (* [root k n] is the real [k]'s [n]-th root of its magnitude, [|m 2ᵉ|^(1/n) =
     2^q (|m| 2ʳ)^(1/n)] for [e = q n + r], [0 ≤ r < n], so that the exponent's
     rounding stays below a unit of the last place. *)
  let root k n =
    let m = Float.abs k.m.re in
    if m = 0. || not (Float.is_finite m) then m
    else
      let q = if k.e >= 0 then k.e / n else -((-k.e + n - 1) / n) in
      let r = k.e - (q * n) in
      Float.ldexp
        (exp
           ((f1 Nx.log m +. (Float.of_int r *. f1 Nx.log 2.)) /. Float.of_int n))
        q
end

module Iir = struct
  type 'e band =
    | Lowpass of 'e
    | Highpass of 'e
    | Bandpass of 'e * 'e
    | Bandstop of 'e * 'e

  type transition = { pass : float; stop : float }

  type family =
    | Butterworth
    | Chebyshev1 of { ripple : float }
    | Chebyshev2 of { reject : float }
    | Elliptic of { ripple : float; reject : float }
    | Bessel

  open H

  let ln = f1 Nx.log

  (* An analog filter [k Π (s - zᵢ) / Π (s - pⱼ)], its zeros and poles as lists
     of complex numbers, [k] scaled. *)
  type zpk = { z : Complex.t list; p : Complex.t list; k : scaled }

  (* [around n] is [-n + 1], [-n + 3], …, [n - 1]. *)
  let around n = List.init n (fun i -> Float.of_int ((2 * i) - n + 1))

  let butterworth n =
    let p =
      List.map
        (fun m ->
          let t = pi *. m /. Float.of_int (2 * n) in
          c (-.cos t) (-.sin t))
        (around n)
    in
    { z = []; p; k = scaled (re 1.) }

  let chebyshev1 n ripple =
    let e2 = pow10m1 (0.1 *. ripple) in
    let mu = asinh (1. /. Float.sqrt e2) /. Float.of_int n in
    (* -sinh (μ + iθ) *)
    let p =
      List.map
        (fun m ->
          let t = pi *. m /. Float.of_int (2 * n) in
          c (-.(sinh mu *. cos t)) (-.(cosh mu *. sin t)))
        (around n)
    in
    let k = sprod (List.map Complex.neg p) in
    let k =
      if n mod 2 = 0 then k /& scaled (re (Float.sqrt (1. +. e2))) else k
    in
    { z = []; p; k }

  let chebyshev2 n reject =
    let de = 1. /. Float.sqrt (pow10m1 (0.1 *. reject)) in
    let mu = asinh (1. /. de) /. Float.of_int n in
    let ms = List.filter (fun m -> m <> 0.) (around n) in
    let z =
      List.map (fun m -> c 0. (1. /. sin (m *. pi /. Float.of_int (2 * n)))) ms
    in
    let p =
      List.map
        (fun m ->
          let t = pi *. m /. Float.of_int (2 * n) in
          re (-1.) /: c (sinh mu *. cos t) (cosh mu *. sin t))
        (around n)
    in
    let k = sprod (List.map Complex.neg p) /& sprod (List.map Complex.neg z) in
    { z; p; k }

  (* Elliptic functions by Landen's transformations (Orfanidis, "Lecture Notes
     on Elliptic Filter Design", 2006, §4), each modulus [k] carried with its
     complement [k'] so that neither is computed from the other near [1]. *)

  (* Landen's descending moduli fall quadratically: twelve steps reach [0] from
     any modulus below [1] in float64. *)
  let landen_steps = 12

  (* [landen (k, k')] is the descending moduli [k₁, k₂, …] down to [0] or for
     [landen_steps], from the complement [k'] alone. *)
  let landen (_, k') =
    let rec go k' acc steps =
      let kn = (1. -. k') /. (1. +. k')
      and kn' = 2. *. Float.sqrt k' /. (1. +. k') in
      let acc = kn :: acc in
      if kn = 0. || steps = 1 then List.rev acc else go kn' acc (steps - 1)
    in
    go k' [] landen_steps

  (* [cde u m] is [cd(u K, k)], or [sn(u K, k)] with [~sn:true], for complex
     [u], eq. 55. *)
  let cde ?(sn = false) u m =
    let w0 = (if sn then csin else ccos) (u *: re (pi /. 2.)) in
    List.fold_left
      (fun w kn -> re (1. +. kn) *: w /: (re 1. +: (re kn *: w *: w)))
      w0
      (List.rev (landen m))

  (* [asne w (k, k')] is [u] with [sn(u K, k) = w], eq. 56. *)
  let asne w (k, k') =
    let w, _ =
      List.fold_left
        (fun (w, kp) kn ->
          let r = sqrt (re 1. -: (re (kp *. kp) *: w *: w)) in
          (re 2. *: w /: (re (1. +. kn) *: (re 1. +: r)), kn))
        (w, k)
        (landen (k, k'))
    in
    casin w *: re (2. /. pi)

  (* [degree n (k1, k1')] is the modulus pair [(k, k')] that the degree equation
     gives for order [n] and the selectivity pair, eq. 47. *)
  let degree n (k1, k1') =
    let kp = ref (k1' ** Float.of_int n) in
    for i = 1 to n / 2 do
      let u = Float.of_int ((2 * i) - 1) /. Float.of_int n in
      let s = (cde ~sn:true (re u) (k1', k1)).re in
      kp := !kp *. (s *. s *. s *. s)
    done;
    (Float.sqrt (1. -. (!kp *. !kp)), !kp)

  let elliptic n ripple reject =
    let e2 = pow10m1 (0.1 *. ripple) in
    let ep = Float.sqrt e2 and es = Float.sqrt (pow10m1 (0.1 *. reject)) in
    let k1 = ep /. es in
    let k1' = Float.sqrt (1. -. (k1 *. k1)) in
    let ((k, _) as m) = degree n (k1, k1') in
    let us =
      List.init (n / 2) (fun i -> Float.of_int ((2 * i) + 1) /. Float.of_int n)
    in
    let za = List.map (fun u -> c 0. (1. /. (k *. (cde (re u) m).re))) us in
    let v0 =
      (asne (c 0. (1. /. ep)) (k1, k1') *: c 0. (-1.) /: re (Float.of_int n)).re
    in
    let pa = List.map (fun u -> c 0. 1. *: cde (c u (-.v0)) m) us in
    let z = za @ List.map Complex.conj za in
    let p = pa @ List.map Complex.conj pa in
    let p =
      if n mod 2 = 1 then p @ [ c 0. 1. *: cde ~sn:true (c 0. v0) m ] else p
    in
    let h0 = if n mod 2 = 0 then 1. /. Float.sqrt (1. +. e2) else 1. in
    let k =
      scaled (re h0)
      *& sprod (List.map Complex.neg p)
      /& sprod (List.map Complex.neg z)
    in
    { z; p; k }

  (* Bessel's prototype

     Its poles are the reciprocals of the zeros of the Bessel polynomial [y_n],
     scaled to put the -3 dB point at [1]. The zeros solve, for [k = 1 … n],
     [Σ_{j≠k} 1 / (z_k - z_j) + (z_k + 1) / z_k² = 0]: at a zero of [y_n], whose
     differential equation is [z² y'' + (2z + 2) y' - n(n+1) y = 0], the ratio
     [y'' / y'] is [2 Σ_{j≠k} 1 / (z_k - z_j)]. Newton's method on this system
     (Pasquini, "Accurate computation of the zeros of the generalized Bessel
     polynomials", Numer. Math. 86, 2000) evaluates no polynomial, whose values
     near its zeros lose every digit past order 25 however they are computed
     (Dunster, Gil, Ruiz-Antolín and Segura, ETNA 65, 2026, §1 and §4). It needs
     good starts: Campos and Calderón's closed forms ("Approximate closed-form
     formulas for the zeros of the Bessel polynomials", 2011, as SciPy's
     [_campos_zeros] gives them), which it converges from for every order from 3
     to 1000 tested; order 2's start lies outside the basin, so orders 1 and 2
     are closed forms. Two identities of [y_n]'s coefficients check the result,
     [Σ z_k = -1] and [Σ 1 / z_k = -n(n+1)/2]. *)

  (* [campos n] is Campos and Calderón's starts for the zeros of [y_n], [n ≥
     2]. *)
  let campos n =
    let poly x cs = List.fold_right (fun c acc -> c +. (x *. acc)) cs 0. in
    let nf = Float.of_int n in
    let s = poly nf [ 0.; 0.; 2.; 0.; -3.; 1. ] in
    let b3 = poly nf [ 16.; -8. ] /. s
    and b2 = poly nf [ -24.; -12.; 12. ] /. s
    and b1 = poly nf [ 8.; 24.; -12.; -2. ] /. s
    and b0 = poly nf [ 0.; -6.; 0.; 5.; -1. ] /. s in
    let r = poly nf [ 0.; 0.; 2.; 1. ] in
    let a1 = poly nf [ -6.; -6. ] /. r and a2 = 6. /. r in
    Array.init n (fun i ->
        let k = Float.of_int (i + 1) in
        c (poly k [ 0.; a1; a2 ]) (poly k [ b0; b1; b2; b3 ]))

  (* [newton z] is the zeros' system solved by Newton's method from [z], stopped
     once a step stops shrinking, as it does at rounding. *)
  let newton z =
    let n = Array.length z in
    let step z =
      let f = Array.make n (re 0.) and jac = Array.make (n * n) (re 0.) in
      for k = 0 to n - 1 do
        let d = ref (re 0.) and fk = ref (re 0.) in
        for j = 0 to n - 1 do
          if j <> k then begin
            let r = re 1. /: (z.(k) -: z.(j)) in
            let r2 = r *: r in
            fk := !fk +: r;
            jac.((k * n) + j) <- r2;
            d := !d -: r2
          end
        done;
        let iz = re 1. /: z.(k) in
        let iz2 = iz *: iz in
        f.(k) <- !fk +: iz +: iz2;
        jac.((k * n) + k) <- !d -: iz2 -: (re 2. *: iz2 *: iz)
      done;
      let m a s = Nx.create Nx.complex128 s a in
      Nx.to_array (Nx.solve (m jac [| n; n |]) (m f [| n; 1 |]))
    in
    let norm s = Array.fold_left (fun a x -> Float.max a (abs x)) 0. s in
    let rec go z last =
      let s = step z in
      let size = norm s in
      if not (size < last) then z
      else go (Array.mapi (fun i x -> x -: s.(i)) z) size
    in
    go z Float.infinity

  let bessel_zeros ~by n =
    if n = 1 then [| re (-1.) |]
    else if n = 2 then
      let r3 = Float.sqrt 3. /. 6. in
      [| c (-0.5) r3; c (-0.5) (-.r3) |]
    else
      let z = newton (campos n) in
      (* Each identity's sum carries the rounding of [n] additions and of zeros
         good to a few units of the last place: [n u] of the sum of magnitudes,
         four times over. *)
      let u = epsilon_float /. 2. in
      let sum f = Array.fold_left (fun a x -> a +: f x) (re 0.) z in
      let mag f = Array.fold_left (fun a x -> a +. abs (f x)) 0. z in
      let tol f = 4. *. Float.of_int n *. u *. mag f in
      let inv x = re 1. /: x in
      let nf = Float.of_int n in
      if
        not
          (abs (sum Fun.id +: re 1.) <= tol Fun.id
          && abs (sum inv +: re (nf *. (nf +. 1.) /. 2.)) <= tol inv)
      then failwith (Printf.sprintf "%s: Bessel order %d did not converge" by n);
      z

  (* [bessel ~by n] is the prototype: poles [1 / z_k], unit delay at zero
     frequency, scaled by [ω₃], the frequency where its gain is [1 / √2]. [ω₃]
     solves [Σ ln |p_k| - Σ ln |iω - p_k| = -ln 2 / 2], a sum that cannot
     overflow, by Newton's method from [√((2n - 1) ln 2)], to which [ω₃] tends
     as the order grows, stopped once a step stops shrinking. *)
  let bessel ~by n =
    let p = Array.map (fun z -> re 1. /: z) (bessel_zeros ~by n) in
    let half = ln 2. /. 2. in
    let f w =
      Array.fold_left
        (fun (v, d) pk ->
          let q = c (-.pk.Complex.re) (w -. pk.im) in
          (v +. ln (abs pk) -. ln (abs q), d -. (c 0. 1. /: q).re))
        (half, 0.) p
    in
    let rec go w last =
      let v, d = f w in
      let s = v /. d in
      if not (Float.abs s < last) then w else go (w -. s) (Float.abs s)
    in
    let start = Float.sqrt (((2. *. Float.of_int n) -. 1.) *. ln 2.) in
    let w3 = go start Float.infinity in
    let p = Array.to_list (Array.map (fun x -> x /: re w3) p) in
    { z = []; p; k = sprod (List.map Complex.neg p) }

  (* Band transformations of a prototype normalised at [1] rad/s. *)

  let relative f = List.length f.p - List.length f.z
  let zeros n v = List.init n (fun _ -> v)
  let neg_prod l = sprod (List.map Complex.neg l)

  let to_lowpass wo f =
    let s x = x *: re wo in
    { z = List.map s f.z; p = List.map s f.p; k = f.k *& spow wo (relative f) }

  let to_highpass wo f =
    let inv x = re wo /: x in
    {
      z = List.map inv f.z @ zeros (relative f) (re 0.);
      p = List.map inv f.p;
      k = f.k *& neg_prod f.z /& neg_prod f.p;
    }

  (* [split wo l] maps each [x] of [l] to [x ± √(x² - wo²)], every [+] first. *)
  let split wo l =
    let r x = sqrt ((x *: x) -: re (wo *. wo)) in
    List.map (fun x -> x +: r x) l @ List.map (fun x -> x -: r x) l

  let to_bandpass wo bw f =
    let h x = x *: re (bw /. 2.) in
    {
      z = split wo (List.map h f.z) @ zeros (relative f) (re 0.);
      p = split wo (List.map h f.p);
      k = f.k *& spow bw (relative f);
    }

  let to_bandstop wo bw f =
    let h x = re (bw /. 2.) /: x in
    let d = relative f in
    {
      z = split wo (List.map h f.z) @ zeros d (c 0. wo) @ zeros d (c 0. (-.wo));
      p = split wo (List.map h f.p);
      k = f.k *& neg_prod f.z /& neg_prod f.p;
    }

  (* [bilinear fs2 f] is [f] under [s = fs2 (z - 1) / (z + 1)], its zeros at
     infinity moved to [-1]. *)
  let bilinear fs2 f =
    let m x = (re fs2 +: x) /: (re fs2 -: x) in
    let d x = re fs2 -: x in
    {
      z = List.map m f.z @ zeros (relative f) (re (-1.));
      p = List.map m f.p;
      k = f.k *& sprod (List.map d f.z) /& sprod (List.map d f.p);
    }

  (* Sections *)

  let is_real (x : Complex.t) = x.im = 0.

  (* [conjugates l] is the conjugate pairs of [l], one each with its positive
     imaginary part, then its real values, as scipy's [_cplxreal] gives them,
     with a relative tolerance of 100 ulps. *)
  let conjugates ~by l =
    let tol = 100. *. epsilon_float in
    let l =
      List.stable_sort
        (fun (a : Complex.t) (b : Complex.t) ->
          match Float.compare a.re b.re with
          | 0 -> Float.compare (Float.abs a.im) (Float.abs b.im)
          | o -> o)
        l
    in
    let real, cplx =
      List.partition (fun x -> Float.abs x.Complex.im <= tol *. abs x) l
    in
    let pos = List.filter (fun x -> x.Complex.im > 0.) cplx in
    let neg = List.filter (fun x -> x.Complex.im < 0.) cplx in
    if List.length pos <> List.length neg then
      invalid_argf "%s: a complex value lacks its conjugate" by;
    let pos = Array.of_list pos and neg = Array.of_list neg in
    (* Within runs of one real part, sort each half by its imaginary part. *)
    let n = Array.length pos in
    let i = ref 0 in
    while !i < n do
      let j = ref (!i + 1) in
      while
        !j < n && pos.(!j).re -. pos.(!j - 1).re <= tol *. abs pos.(!j - 1)
      do
        incr j
      done;
      let by_im a =
        let s = Array.sub a !i (!j - !i) in
        Array.stable_sort
          (fun (x : Complex.t) y ->
            Float.compare (Float.abs x.im) (Float.abs y.im))
          s;
        Array.blit s 0 a !i (!j - !i)
      in
      by_im pos;
      by_im neg;
      i := !j
    done;
    Array.iteri
      (fun i p ->
        if abs (p -: Complex.conj neg.(i)) > tol *. abs neg.(i) then
          invalid_argf "%s: a complex value lacks its conjugate" by)
      pos;
    Array.to_list
      (Array.mapi (fun i p -> (p +: Complex.conj neg.(i)) /: re 2.) pos)
    @ List.map (fun x -> re x.Complex.re) real

  (* [poly l] is [Π (1 - r z⁻¹)] over the roots [l], one or two, as coefficients
     from [z⁰], real parts. *)
  let poly l =
    let mul a r =
      let n = Array.length a in
      Array.init (n + 1) (fun i ->
          let x = if i < n then a.(i) else re 0. in
          if i = 0 then x else x +: (a.(i - 1) *: Complex.neg r))
    in
    Array.map (fun x -> x.Complex.re) (List.fold_left mul [| re 1. |] l)

  (* [single z p] is a section of the zeros [z] and poles [p], at most two each,
     its polynomials aligned to the right as scipy's [_single_zpksos]. *)
  let single z p =
    let b = poly z and a = poly p in
    let row = Array.make 6 0. in
    Array.blit b 0 row (3 - Array.length b) (Array.length b);
    Array.blit a 0 row (6 - Array.length a) (Array.length a);
    row

  (* [nearest l x which] is the index in [l] of the value nearest [x], among the
     real ones, the complex ones or any. *)
  let nearest l x which =
    let best = ref (-1) and d = ref Float.infinity in
    Array.iteri
      (fun i v ->
        let ok =
          match which with
          | `Any -> true
          | `Real -> is_real v
          | `Complex -> not (is_real v)
        in
        let e = abs (v -: x) in
        if ok && e < !d then (
          best := i;
          d := e))
      l;
    !best

  let remove l i =
    Array.append (Array.sub l 0 i)
      (Array.sub l (i + 1) (Array.length l - i - 1))

  (* [worst l] is the index of the pole nearest the unit circle, the first of
     equals. *)
  let worst l =
    let best = ref 0 in
    Array.iteri
      (fun i v ->
        if Float.abs (1. -. abs v) < Float.abs (1. -. abs l.(!best)) then
          best := i)
      l;
    !best

  (* [sections ~by f] is scipy's [zpk2sos] with its [nearest] pairing: the poles
     nearest the unit circle go last, each with its nearest zeros. Each of the
     [S] sections takes the [S]-th root of the gain's magnitude, the first its
     sign too, so that no row's numerator underflows where the gain would. *)
  let sections ~by f =
    let z = ref (Array.of_list f.z) and p = ref (Array.of_list f.p) in
    let pad a n = Array.append a (Array.make (Int.max 0 n) (re 0.)) in
    (* Both lists padded to one length, at least one value, so that the empty
       cascade is one section of the gain. *)
    let n = Int.max 1 (Int.max (Array.length !z) (Array.length !p)) in
    p := pad !p (n - Array.length !p);
    z := pad !z (n - Array.length !z);
    let count = (n + 1) / 2 in
    if Array.length !p mod 2 = 1 then (
      p := pad !p 1;
      z := pad !z 1);
    z := Array.of_list (conjugates ~by (Array.to_list !z));
    p := Array.of_list (conjugates ~by (Array.to_list !p));
    let take r i =
      let v = !r.(i) in
      r := remove !r i;
      v
    in
    let reals a =
      Array.fold_left (fun n v -> if is_real v then n + 1 else n) 0 a
    in
    let rows = Array.make count [||] in
    for si = count - 1 downto 0 do
      let p1 = take p (worst !p) in
      rows.(si) <-
        (if is_real p1 && reals !p = 0 then
           let z1 = take z (nearest !z p1 `Real) in
           single [ z1; re 0. ] [ p1; re 0. ]
         else if
           Array.length !p + 1 = Array.length !z
           && (not (is_real p1))
           && reals !p = 1
           && reals !z = 1
         then
           let z1 = take z (nearest !z p1 `Complex) in
           single [ z1; Complex.conj z1 ] [ p1; Complex.conj p1 ]
         else
           let p2 =
             if is_real p1 then begin
               let idx =
                 List.filter
                   (fun i -> is_real !p.(i))
                   (List.init (Array.length !p) Fun.id)
               in
               let sub = Array.of_list (List.map (fun i -> !p.(i)) idx) in
               take p (List.nth idx (worst sub))
             end
             else Complex.conj p1
           in
           if Array.length !z = 0 then single [] [ p1; p2 ]
           else
             let z1 = take z (nearest !z p1 `Any) in
             if not (is_real z1) then single [ z1; Complex.conj z1 ] [ p1; p2 ]
             else if Array.length !z > 0 then
               let z2 = take z (nearest !z p1 `Real) in
               single [ z1; z2 ] [ p1; p2 ]
             else single [ z1 ] [ p1; p2 ])
    done;
    let share = root f.k count in
    let sign = Float.copy_sign 1. f.k.m.re in
    Array.iteri
      (fun i row ->
        let g = if i = 0 then sign *. share else share in
        for j = 0 to 2 do
          row.(j) <- row.(j) *. g
        done)
      rows;
    rows

  let cascade dt rows =
    Nx.cast dt
      (Nx.create Nx.float64
         [| Array.length rows; 6 |]
         (Array.concat (Array.to_list rows)))

  let design dt ~rate ~order family band =
    let by = "Omr.Iir.design" in
    check_rate ~by rate;
    if order < 1 then invalid_argf "%s: order %d" by order;
    let edge f =
      if not (0. < f && f < rate /. 2.) then
        invalid_argf "%s: edge %g outside (0, %g)" by f (rate /. 2.)
    in
    (match band with
    | Lowpass f | Highpass f -> edge f
    | Bandpass (f, g) | Bandstop (f, g) ->
        edge f;
        edge g;
        if not (f < g) then invalid_argf "%s: band edges %g and %g" by f g);
    let positive name v =
      if not (v > 0. && Float.is_finite v) then
        invalid_argf "%s: %s %g" by name v
    in
    let proto =
      match family with
      | Butterworth -> butterworth order
      | Chebyshev1 { ripple } ->
          positive "ripple" ripple;
          chebyshev1 order ripple
      | Chebyshev2 { reject } ->
          positive "reject" reject;
          chebyshev2 order reject
      | Elliptic { ripple; reject } ->
          positive "ripple" ripple;
          positive "reject" reject;
          elliptic order ripple reject
      | Bessel -> bessel ~by order
    in
    (* At a rate of 2, band edges [f / (rate / 2)] prewarped to [4 tan (π w /
       2)]. *)
    let warp f = 4. *. tan (pi *. (f /. (rate /. 2.)) /. 2.) in
    let f =
      match band with
      | Lowpass f -> to_lowpass (warp f) proto
      | Highpass f -> to_highpass (warp f) proto
      | Bandpass (a, b) ->
          let a = warp a and b = warp b in
          to_bandpass (Float.sqrt (a *. b)) (b -. a) proto
      | Bandstop (a, b) ->
          let a = warp a and b = warp b in
          to_bandstop (Float.sqrt (a *. b)) (b -. a) proto
    in
    cascade dt (sections ~by (bilinear 4. f))

  (* Orders *)

  (* [agm a b] is the arithmetic-geometric mean of [a ≥ b > 0], its iteration
     stopped once the gap stops shrinking, as it does at rounding. *)
  let agm a b =
    let rec go a b =
      let a' = (a +. b) /. 2. and b' = Float.sqrt (a *. b) in
      if a' -. b' >= a -. b then a' else go a' b'
    in
    go a b

  (* [kk (k, k')] is the complete elliptic integral [K(k) = π / (2 M(1, k'))],
     from the complement so that [k] near [1] keeps its digits. [K'(k)] is [kk
     (k', k)]. *)
  let kk (_, k') = pi /. (2. *. agm 1. k')

  (* [pair k] is [(k, √(1 - k²))] for [0 < k < 1]. *)
  let pair k = (k, Float.sqrt ((1. -. k) *. (1. +. k)))

  (* [acosh_inv (k, k')] is [acosh (1 / k) = ln ((1 + k') / k)]. *)
  let acosh_inv (k, k') = ln ((1. +. k') /. k)

  (* The least order is the degree equation's, rounded up (Orfanidis, eqs. 9 and
     34): with the edges prewarped, [k] the ratio of the pass edge's frequency
     to the stop edge's in the low pass equivalent, [ε] and [εs] of [ripple] and
     [reject] as [√(10^(dB/10) - 1)] and [k₁ = ε / εs], it is [ln k₁ / ln k] for
     Butterworth, [acosh (1/k₁) / acosh (1/k)] for either Chebyshev, and [K(k)
     K'(k₁) / (K'(k) K(k₁))] for elliptic. A band pass centres its transform on
     the pass edges' geometric mean (eq. 118), a band stop on the stop edges'
     (eq. 121), each the centre that gives the least order; a band stop's pass
     band then reaches past its looser pass edge. The edges returned are where
     each family places them so that the pass edges sit at [-ripple] dB: the
     pass edges for Chebyshev I and elliptic, the -3 dB points for Butterworth,
     and for Chebyshev II the stop band's start moved toward the pass band. The
     transform keeps the prototype's gain at each prewarped frequency, so this
     holds in exact arithmetic and no certificate is computed. *)
  let meet ~rate ~ripple ~reject kind spec =
    let by = "Omr.Iir.meet" in
    check_rate ~by rate;
    let positive name v =
      if not (v > 0. && Float.is_finite v) then
        invalid_argf "%s: %s %g" by name v
    in
    positive "ripple" ripple;
    positive "reject" reject;
    if not (reject > ripple) then
      invalid_argf "%s: reject %g not above ripple %g" by reject ripple;
    let edge f =
      if not (0. < f && f < rate /. 2.) then
        invalid_argf "%s: edge %g outside (0, %g)" by f (rate /. 2.)
    in
    let ordered l =
      List.iter edge l;
      let rec go = function
        | a :: (b :: _ as rest) -> a < b && go rest
        | _ -> true
      in
      if not (go l) then
        invalid_argf "%s: edges %s out of order" by
          (String.concat ", " (List.map (Printf.sprintf "%g") l))
    in
    (match spec with
    | Lowpass t -> ordered [ t.pass; t.stop ]
    | Highpass t -> ordered [ t.stop; t.pass ]
    | Bandpass (l, h) -> ordered [ l.stop; l.pass; h.pass; h.stop ]
    | Bandstop (l, h) -> ordered [ l.pass; l.stop; h.stop; h.pass ]);
    (* Edges prewarped to [Ω = tan (π f / rate)], and back. *)
    let warp f = tan (pi *. f /. rate) in
    let unwarp w = rate /. pi *. atan2 w 1. in
    (* The low pass equivalent's selectivity [k], the pass edge's frequency over
       the stop edge's, with its complement, and the band's centre [w0²] and
       width [b] (eqs. 118 and 121). *)
    let k, w02, b =
      match spec with
      | Lowpass { pass; stop } ->
          let p = warp pass and s = warp stop in
          ((p /. s, Float.sqrt ((s -. p) *. (s +. p)) /. s), 0., 0.)
      | Highpass { pass; stop } ->
          let p = warp pass and s = warp stop in
          ((s /. p, Float.sqrt ((p -. s) *. (p +. s)) /. p), 0., 0.)
      | Bandpass (l, h) ->
          let p1 = warp l.pass and p2 = warp h.pass in
          let w02 = p1 *. p2 and b = p2 -. p1 in
          let lp s = Float.abs ((s *. s) -. w02) /. (b *. s) in
          let s = Float.min (lp (warp l.stop)) (lp (warp h.stop)) in
          (pair (1. /. s), w02, b)
      | Bandstop (l, h) ->
          let s1 = warp l.stop and s2 = warp h.stop in
          let w02 = s1 *. s2 in
          let lp p = p /. Float.abs (w02 -. (p *. p)) in
          let v = Float.max (lp (warp l.pass)) (lp (warp h.pass)) in
          (pair ((s2 -. s1) *. v), w02, 1. /. v)
    in
    (* The ratio [k₁ = ε / εs]; [pow10m1] keeps a small ripple's digits. *)
    let e = Float.sqrt (pow10m1 (ripple /. 10.)) in
    let k1 = pair (e /. Float.sqrt (pow10m1 (reject /. 10.))) in
    let n =
      match kind with
      | `Butterworth -> ln (1. /. fst k1) /. ln (1. /. fst k)
      | `Chebyshev1 | `Chebyshev2 -> acosh_inv k1 /. acosh_inv k
      | `Elliptic ->
          let swap (a, b) = (b, a) in
          kk k *. kk (swap k1) /. (kk (swap k) *. kk k1)
    in
    let order = Int.max 1 (Float.to_int (Float.ceil n)) in
    let family =
      match kind with
      | `Butterworth -> Butterworth
      | `Chebyshev1 -> Chebyshev1 { ripple }
      | `Chebyshev2 -> Chebyshev2 { reject }
      | `Elliptic -> Elliptic { ripple; reject }
    in
    (* The family's edge in the low pass equivalent, its pass edge at [1]: the
       -3 dB point of a Butterworth whose gain there is [-ripple] dB, the stop
       band's start of a Chebyshev II whose gain there is. *)
    let on =
      match kind with
      | `Butterworth -> exp (-.ln e /. Float.of_int order)
      | `Chebyshev1 | `Elliptic -> 1.
      | `Chebyshev2 -> cosh (acosh_inv k1 /. Float.of_int order)
    in
    let band =
      match spec with
      | Lowpass { pass; _ } -> Lowpass (unwarp (warp pass *. on))
      | Highpass { pass; _ } -> Highpass (unwarp (warp pass /. on))
      | Bandpass _ ->
          let b = b *. on in
          let hi = (b +. Float.sqrt ((b *. b) +. (4. *. w02))) /. 2. in
          Bandpass (unwarp (w02 /. hi), unwarp hi)
      | Bandstop _ ->
          let b = b /. on in
          let lo = 2. *. w02 /. (b +. Float.sqrt ((b *. b) +. (4. *. w02))) in
          Bandstop (unwarp lo, unwarp (w02 /. lo))
    in
    (order, family, band)

  let digital dt ~zeros ~poles ~gain =
    let by = "Omr.Iir.digital" in
    cascade dt (sections ~by { z = zeros; p = poles; k = scaled (re gain) })

  let analog dt ~rate ~zeros ~poles ~gain =
    let by = "Omr.Iir.analog" in
    check_rate ~by rate;
    if List.length zeros > List.length poles then
      invalid_argf "%s: %d zeros for %d poles" by (List.length zeros)
        (List.length poles);
    if poles = [] then invalid_argf "%s: no pole" by;
    List.iter
      (fun (p : Complex.t) ->
        if not (p.re < 0.) then
          invalid_argf "%s: a pole of real part %g" by p.re)
      poles;
    cascade dt
      (sections ~by
         (bilinear (2. *. rate) { z = zeros; p = poles; k = scaled (re gain) }))
end

(* FIR filters *)

let check_taps ~by h =
  if Nx.ndim h <> 1 || Nx.dim 0 h = 0 then
    invalid_argf "%s: taps %a are not [m] with m > 0" by pp_shape (Nx.shape h)

(* [fir_grid ~by ?edge ?ends h] is the frame grid of [h]'s windows, hop 1. *)
let fir_grid ~by ?(edge : edge = `Zero) ?ends h =
  check_taps ~by h;
  let m = Nx.dim 0 h in
  let ((lo, hi) as ends) = Option.value ends ~default:(m - 1, 0) in
  if lo < 0 || hi < 0 then invalid_argf "%s: ends (%d, %d)" by lo hi;
  Frames.v ~edge ~ends ~window:m ~hop:1

(* [convolver h] reduces each window of a span against [h] reversed: one
   [Nx.conv] of one channel, whose output is one reduction of its window. *)
let convolver h =
  let w = Nx.reshape [| 1; Nx.dim 0 h; 1 |] (Nx.flip h) in
  let cut (_ : _ Frames.t) x k =
    let lead = lead x 1 in
    let rows = Array.fold_left ( * ) 1 lead in
    let x = Nx.reshape [| rows; Nx.dim (-1) x; 1 |] x in
    let y =
      if k = 0 then Nx.zeros_at (Nx.beside x) (Nx.dtype x) [| rows; 0; 1 |]
      else Nx.conv x w
    in
    Nx.reshape (Array.append lead [| k |]) y
  in
  { Frames.cut; drop = (fun first y -> along y first (Nx.dim (-1) y)) }

let fir ?(axis = -1) ?edge ?ends h x =
  let by = "Omr.fir" in
  let g = fir_grid ~by ?edge ?ends h in
  untime ~by axis (Frames.whole_with ~by (convolver h) g (time ~by axis x))

module Fir = struct
  module State = Frames.State

  let start = Frames.start

  let feed ?(axis = -1) ?edge ?ends h s x =
    let by = "Omr.Fir.feed" in
    let g = fir_grid ~by ?edge ?ends h in
    let s, y = Frames.feed_with ~by (convolver h) g s (time ~by axis x) in
    (s, untime ~by axis y)

  let finish ?(axis = -1) ?edge ?ends h s =
    let by = "Omr.Fir.finish" in
    let g = fir_grid ~by ?edge ?ends h in
    untime ~by axis (Frames.finish_with ~by (convolver h) g s)
end

(* Zero phase *)

let zero_phase ?(axis = -1) ?(edge = `Antisymmetric_whole) ?ends c x =
  let by = "Omr.zero_phase" in
  check_cascade ~by c;
  let x = time ~by axis x in
  let n = Nx.dim (-1) x and p = 3 * ((2 * Nx.dim 0 c) + 1) in
  let ((lo, hi) as ends) = Option.value ends ~default:(p, p) in
  if lo < 0 || hi < 0 then invalid_argf "%s: ends (%d, %d)" by lo hi;
  if n = 0 && edge <> `Zero && lo + hi > 0 then
    invalid_argf "%s: no sample to read past the ends of %a" by pp_shape
      (Nx.shape x);
  let e = Nx.extend edge ends x in
  if Nx.dim (-1) e = 0 then untime ~by axis x
  else
    let rev = Nx.flip ~axes:[ -1 ] in
    let pass e = snd (Sos.feed_last ~by c (Sos.steady c (sample e 0)) e) in
    untime ~by axis (along (rev (pass (rev (pass e)))) lo (lo + n))

(* FFT convolution *)

(* [transform_size m] is the least power of two [N >= m] that minimises [N log₂
   N / (N - m + 1)], the transform work per output of a block. *)
let transform_size m =
  let cost n l = Float.of_int (n * l) /. Float.of_int (n - m + 1) in
  let rec first n l = if n >= m then (n, l) else first (2 * n) (l + 1) in
  let n0, l0 = first 1 0 in
  let rec go n l best bl =
    if l > l0 + 24 then best
    else
      let best, bl = if cost n l < cost best bl then (n, l) else (best, bl) in
      go (2 * n) (l + 1) best bl
  in
  go n0 l0 n0 l0

let norm_axes ~by r axes =
  let axes = List.map (fun a -> if a < 0 then a + r else a) axes in
  List.iter
    (fun a -> if a < 0 || a >= r then invalid_argf "%s: no axis in %d-d" by r)
    axes;
  if List.length (List.sort_uniq compare axes) <> List.length axes then
    invalid_argf "%s: axes repeat" by;
  axes

(* [blocks ~axes x k] is the convolution of [x], extended already, with [k]
   along [axes]: overlap-save in blocks of [N - m + 1] outputs from output 0,
   [N] from [m] alone along each axis, the last block's window filled with
   zeros. *)
let blocks (type v s d) ~by ~axes (x : (v, s, d) Nx.t) (k : (v, s, d) Nx.t) :
    (v, s, d) Nx.t =
  let r = Nx.ndim x in
  let sizes =
    List.map
      (fun a ->
        let m = Nx.dim a k and n = Nx.dim a x in
        let big = transform_size m in
        let out = Int.max 0 (n - m + 1) in
        let step = big - m + 1 in
        (a, m, big, step, out, (out + step - 1) / step))
      axes
  in
  let shape =
    Array.init r (fun i ->
        match List.find_opt (fun (a, _, _, _, _, _) -> a = i) sizes with
        | Some (_, _, _, _, out, _) -> out
        | None -> Nx.dim i x)
  in
  if Array.exists (( = ) 0) shape then
    Nx.zeros_at (Nx.beside x) (Nx.dtype x) shape
  else begin
    (* Pad to whole blocks, then cut each axis into windows of [N] every [N - m
       + 1], the windows' axes appended in [axes]' order. *)
    let pads =
      Array.init r (fun i ->
          match List.find_opt (fun (a, _, _, _, _, _) -> a = i) sizes with
          | Some (_, _, big, step, _, b) ->
              (0, ((b - 1) * step) + big - Nx.dim i x)
          | None -> (0, 0))
    in
    let w =
      List.fold_left
        (fun w (a, _, big, step, _, _) ->
          Nx.sliding_window ~axis:a ~window:big ~step w)
        (Nx.pad pads (Nx.Dtype.zero (Nx.dtype x)) x)
        sizes
    in
    let spectral = List.init (List.length axes) (fun i -> r + i) in
    let bigs = List.map (fun (_, _, big, _, _, _) -> big) sizes in
    (* The kernel's spectra, its axes [axes] moved last and [1] in their
       place. *)
    let kernel_like spec =
      let others =
        List.filter (fun i -> not (List.mem i axes)) (List.init r Fun.id)
      in
      let t = Nx.transpose ~axes:(others @ axes) spec in
      let lead =
        Array.init r (fun i -> if List.mem i axes then 1 else Nx.dim i spec)
      in
      Nx.reshape
        (Array.append lead
           (Array.of_list (List.map (fun a -> Nx.dim a spec) axes)))
        t
    in
    let y : (v, s, d) Nx.t =
      match Nx.Dtype.kind (Nx.dtype x) with
      | Nx.Dtype.Float ->
          let dt = Nx.dtype x in
          complex_of dt
            {
              f =
                (fun cdt ->
                  let ks = kernel_like (Nx.rfft cdt ~axes ~s:bigs k) in
                  let xs = Nx.rfft cdt ~axes:spectral w in
                  Nx.irfft dt ~axes:spectral ~s:bigs (Nx.mul xs ks));
            }
      | Nx.Dtype.Complex ->
          let ks = kernel_like (Nx.fft ~axes ~s:bigs k) in
          Nx.ifft ~axes:spectral (Nx.mul (Nx.fft ~axes:spectral w) ks)
      | _ -> invalid_argf "%s: a dtype other than float or complex" by
    in
    (* Keep each window's last [N - m + 1] points, then join each axis' blocks
       and cut them to its outputs. *)
    let y =
      Nx.slice
        (List.init r (fun _ -> Nx.A)
        @ List.map (fun (_, m, big, _, _, _) -> Nx.R (m - 1, big)) sizes)
        y
    in
    let perm =
      List.concat
        (List.init r (fun i ->
             match List.find_index (fun (a, _, _, _, _, _) -> a = i) sizes with
             | Some j -> [ i; r + j ]
             | None -> [ i ]))
    in
    let joined =
      Array.init r (fun i ->
          match List.find_opt (fun (a, _, _, _, _, _) -> a = i) sizes with
          | Some (_, _, _, step, _, b) -> b * step
          | None -> Nx.dim i y)
    in
    let y = Nx.reshape joined (Nx.transpose ~axes:perm y) in
    Nx.slice (Array.to_list (Array.map (fun n -> Nx.R (0, n)) shape)) y
  end

(* [check_kernel ~by ~axes x k] raises unless [k] broadcasts as a kernel of [x]
   along [axes]. *)
let check_kernel ~by ~axes x k =
  if Nx.ndim k <> Nx.ndim x then
    invalid_argf "%s: kernel %a for a signal of %a" by pp_shape (Nx.shape k)
      pp_shape (Nx.shape x);
  Array.iteri
    (fun i d ->
      if List.mem i axes then (
        if d < 1 then invalid_argf "%s: kernel %a" by pp_shape (Nx.shape k))
      else if d <> 1 && d <> Nx.dim i x then
        invalid_argf "%s: kernel %a for a signal of %a" by pp_shape (Nx.shape k)
          pp_shape (Nx.shape x))
    (Nx.shape k)

let check_ends ~by axes ends =
  if List.length ends <> List.length axes then
    invalid_argf "%s: %d pairs of ends for %d axes" by (List.length ends)
      (List.length axes);
  List.iter
    (fun (lo, hi) ->
      if lo < 0 || hi < 0 then invalid_argf "%s: ends (%d, %d)" by lo hi)
    ends

let check_dtype (type v s) ~by (dt : (v, s) Nx.dtype) =
  match Nx.Dtype.kind dt with
  | Nx.Dtype.Float -> ()
  | Nx.Dtype.Complex -> ()
  | _ -> invalid_argf "%s: dtype %s" by (Nx.Dtype.name dt)

(* [zero_ends x pads] is [x] extended with zeros by [pads], per axis. *)
let zero_ends x pads = Nx.pad pads (Nx.Dtype.zero (Nx.dtype x)) x

let fft_convolve ~axes ~ends x k =
  let by = "Omr.fft_convolve" in
  check_dtype ~by (Nx.dtype x);
  let axes = norm_axes ~by (Nx.ndim x) axes in
  check_ends ~by axes ends;
  check_kernel ~by ~axes x k;
  let pads =
    Array.init (Nx.ndim x) (fun i ->
        match List.find_index (( = ) i) axes with
        | Some j -> List.nth ends j
        | None -> (0, 0))
  in
  blocks ~by ~axes (zero_ends x pads) k

module Fft_convolve = struct
  module State = struct
    type ('a, _) t = Start | Running of { samples : 'a }

    let walk c s =
      let open Nx.Ptree.Walk in
      match s with
      | Start ->
          case c "start";
          Start
      | Running { samples } ->
          case c "running";
          Running { samples = field c "samples" leaf samples }
  end

  let start = State.Start

  (* [setup ~by ~axis ~axes ~ends x k] is the streamed axis, the axes and the
     extension of a chunk along the other axes. *)
  let setup ~by ~axis ~axes ~ends x k =
    check_dtype ~by (Nx.dtype x);
    let r = Nx.ndim x in
    let axes = norm_axes ~by r axes in
    check_ends ~by axes ends;
    let a = if axis < 0 then axis + r else axis in
    let j =
      match List.find_index (( = ) a) axes with
      | Some j -> j
      | None -> invalid_argf "%s: axis %d is not among the axes" by axis
    in
    check_kernel ~by ~axes x k;
    let pads =
      Array.init r (fun i ->
          if i = a then (0, 0)
          else
            match List.find_index (( = ) i) axes with
            | Some j -> List.nth ends j
            | None -> (0, 0))
    in
    (a, axes, List.nth ends j, pads)

  let span a x lo hi =
    Nx.slice
      (List.init (Nx.ndim x) (fun i -> if i = a then Nx.R (lo, hi) else Nx.A))
      x

  let feed ~axis ~axes ~ends k s x =
    let by = "Omr.Fft_convolve.feed" in
    let a, axes, (lo, _), pads = setup ~by ~axis ~axes ~ends x k in
    let x = zero_ends x pads in
    let kept =
      match s with
      | State.Start ->
          zero_ends x
            (Array.init (Nx.ndim x) (fun i -> if i = a then (lo, 0) else (0, 0)))
      | Running { samples } ->
          let shape s =
            Array.mapi (fun i d -> if i = a then 0 else d) (Nx.shape s)
          in
          if shape samples <> shape x then
            invalid_argf "%s: a chunk of %a after %a" by pp_shape (Nx.shape x)
              pp_shape (Nx.shape samples);
          Nx.concatenate ~axis:a [ samples; x ]
    in
    let m = Nx.dim a k in
    let big = transform_size m in
    let step = big - m + 1 and len = Nx.dim a kept in
    let b = if len < big then 0 else ((len - big) / step) + 1 in
    let out =
      blocks ~by ~axes
        (span a kept 0 (if b = 0 then 0 else ((b - 1) * step) + big))
        k
    in
    (State.Running { samples = span a kept (b * step) len }, out)

  let finish ~axis ~axes ~ends k s =
    let by = "Omr.Fft_convolve.finish" in
    match s with
    | State.Start ->
        invalid_argf "%s: no chunk was fed; feed an empty chunk first" by
    | Running { samples } ->
        let a, axes, (_, hi), _ = setup ~by ~axis ~axes ~ends samples k in
        let e =
          zero_ends samples
            (Array.init (Nx.ndim samples) (fun i ->
                 if i = a then (0, hi) else (0, 0)))
        in
        blocks ~by ~axes e k
end

(* Kernels *)

module Kernel = struct
  type window = Kaiser of float | Hann | Lanczos

  type t =
    | Box
    | Triangle
    | Cubic of float
    | Sinc of { cutoff : float; half_width : float; window : window }

  let half_width = function
    | Box -> 0.5
    | Triangle -> 1.
    | Cubic _ -> 2.
    | Sinc { half_width; _ } -> half_width

  (* [sinc x] is [sin (π x) / (π x)], [1] at [0]. *)
  let sinc x =
    let zero = Nx.equal_s x 0. in
    let px = Nx.mul_s (Nx.where zero (Nx.ones_like x) x) Float.pi in
    Nx.where zero (Nx.ones_like x) (Nx.div (Nx.sin px) px)

  (* [series beta] is Kaiser's window as a polynomial in [q = 1 - r²]: [I₀(2 √x)
     = Σ xᵏ / (k!)²] for [x = β² q / 4], so its coefficients are [cₖ = (β² / 4)ᵏ
     / ((k!)² I₀(β))], which sum to the window at [q = 1], [1]. The terms grow
     to the largest, [k = ⌊β / 2⌋], then shrink; the series ends before the
     first past it under [u], the unit roundoff of float64. Each is computed at
     float64 relative to the largest, so none overflows, then divided by their
     sum. A [β] that is not finite gives a NaN window. *)
  let series beta =
    if not (Float.is_finite beta) then [| Float.nan |]
    else
      let x = beta *. beta /. 4. in
      let peak = Float.to_int (Float.abs beta /. 2.) in
      let u = epsilon_float /. 2. in
      let below = Array.make (peak + 1) 1. in
      for k = peak downto 1 do
        below.(k - 1) <- below.(k) *. Float.of_int (k * k) /. x
      done;
      (* Past the peak, terms down to [u²] of the sum, enough to fix it. *)
      let rec above k t sum acc =
        let t = t *. x /. Float.of_int ((k + 1) * (k + 1)) in
        if t < u *. u *. sum then (sum, List.rev acc)
        else above (k + 1) t (sum +. t) (t :: acc)
      in
      let sum, after = above peak 1. (Array.fold_left ( +. ) 0. below) [] in
      let rec keep = function
        | t :: rest when t /. sum >= u -> (t /. sum) :: keep rest
        | _ -> []
      in
      Array.append
        (Array.map (fun t -> t /. sum) below)
        (Array.of_list (keep after))

  (* [kaiser beta r] is Kaiser's window at [r = τ / h] by Horner's rule in [q],
     in [r]'s dtype. A polynomial, its derivatives are regular at every order.
     [q] is held at [0] past [|r| = 1], whose values [values] drops, so that a
     far position neither overflows the powers nor sends their [0 · ∞] into a
     derivative. *)
  let kaiser beta r =
    let c = series beta in
    let q = Nx.rsub_s 1. (Nx.square r) in
    let q = Nx.where (Nx.greater_equal_s q 0.) q (Nx.zeros_like q) in
    let w = ref (Nx.full_like q c.(Array.length c - 1)) in
    for k = Array.length c - 2 downto 0 do
      w := Nx.add_s (Nx.mul !w q) c.(k)
    done;
    !w

  (* [values ~closed k t] is [k] at the positions [t], in [t]'s dtype, zero
     outside [[-h, h)], or outside [[-h, h]] when [closed]. *)
  let values ?(closed = false) k t =
    let h = half_width k in
    let inside =
      Nx.logical_and
        (Nx.greater_equal_s t (-.h))
        (if closed then Nx.less_equal_s t h else Nx.less_s t h)
    in
    let a = Nx.abs t in
    let v =
      match k with
      | Box -> Nx.ones_like t
      | Triangle -> Nx.rsub_s 1. a
      | Cubic c ->
          (* Keys: [(c + 2) a³ - (c + 3) a² + 1] to [1], then [c a³ - 5 c a² + 8
             c a - 4 c]. *)
          let a2 = Nx.square a in
          let a3 = Nx.mul a2 a in
          let near =
            Nx.add_s (Nx.sub (Nx.mul_s a3 (c +. 2.)) (Nx.mul_s a2 (c +. 3.))) 1.
          in
          let far =
            Nx.sub_s
              (Nx.add
                 (Nx.sub (Nx.mul_s a3 c) (Nx.mul_s a2 (5. *. c)))
                 (Nx.mul_s a (8. *. c)))
              (4. *. c)
          in
          Nx.where (Nx.less_equal_s a 1.) near far
      | Sinc { cutoff; half_width = h; window } ->
          let s = Nx.mul_s (sinc (Nx.mul_s t (2. *. cutoff))) (2. *. cutoff) in
          let w =
            match window with
            | Hann -> Nx.square (Nx.cos (Nx.mul_s t (Float.pi /. (2. *. h))))
            | Lanczos -> sinc (Nx.div_s t h)
            | Kaiser beta -> kaiser beta (Nx.div_s t h)
          in
          Nx.mul s w
    in
    Nx.where inside v (Nx.zeros_like t)

  let eval k t = values k t

  (* [integers k] is the integers in [[-h, h)], and [h] too for a [Sinc] of an
     integer [h]. *)
  let integers k =
    let h = half_width k in
    let lo = Float.to_int (Float.ceil (-.h)) in
    let hi = Float.to_int (Float.ceil h) - 1 in
    let hi =
      match k with Sinc _ when Float.is_integer h -> hi + 1 | _ -> hi
    in
    Array.init (Int.max 0 (hi - lo + 1)) (fun i -> Float.of_int (lo + i))

  let taps64 k =
    let v =
      values ~closed:true k
        (Nx.create Nx.float64 [| Array.length (integers k) |] (integers k))
    in
    Nx.div v (Nx.sum v)

  let taps dt k = Nx.cast dt (taps64 k)

  (* Certifying a gain

     Taps [t] at distances [j] from their centre, whole for an odd length and
     half-integers for an even one, have at [x] cycles per sample the amplitude
     [A(x) = Σ t_j cos (2π j x)] when symmetric and [A(x) = -Σ t_j sin (2π j x)]
     when antisymmetric. Either way [|A'| ≤ 2π Σ |t_j j|] and the third
     derivative is bounded by [M₃ = 8π³ Σ |t_j| |j|³]. A band's error is [e = |A
     - D|] for a target [D] linear in [x], so [e]'s first derivative is at most
     [|A' - D'|], its second [|A''|], and Taylor's theorem bounds [e] within [s]
     of a point [p] by [e(p) + |A'(p) - D'| s + |A''(p)| s² / 2 + M₃ s³ / 6].
     The bound clears the interval when it is at most [τ(p) - |τ'| s], the
     band's tolerance [τ], linear, at its least over the interval. *)

  (* [centred t] is each tap's distance from the centre [(m - 1) / 2]. *)
  let centred t =
    Nx.sub_s
      (Nx.arange Nx.float64 0 (Nx.dim 0 t) 1)
      (Float.of_int (Nx.dim 0 t - 1) /. 2.)

  (* [moment t m] is [t_j jᵐ]. *)
  let moment t m =
    let j = centred t in
    let rec go m acc = if m = 0 then acc else go (m - 1) (Nx.mul acc j) in
    go m t

  let two_pi = 2. *. Float.pi
  let rec pow_int x m = if m = 0 then 1. else x *. pow_int x (m - 1)

  (* [direct anti t x] is [A], [A'] and [A''] at [x], one tap at a time. *)
  let direct anti t x =
    let ph = Nx.mul_s (centred t) (two_pi *. x) in
    let c = Nx.cos ph and s = Nx.sin ph in
    let sum a = Nx.item [] (Nx.sum a) in
    if anti then
      ( -.sum (Nx.mul t s),
        -.two_pi *. sum (Nx.mul (moment t 1) c),
        two_pi *. two_pi *. sum (Nx.mul (moment t 2) s) )
    else
      ( sum (Nx.mul t c),
        -.two_pi *. sum (Nx.mul (moment t 1) s),
        -.(two_pi *. two_pi) *. sum (Nx.mul (moment t 2) c) )

  (* [on_grid anti t n] is [A], [A'] and [A''] at [i / n] for [i ≤ n / 2], from
     transforms of [t jᵐ] of [n] points turned by [e^(2πi c i / n)], [c = (m -
     1) / 2] the centre: [Σ t_j jᵐ e^(-2πi j x)], whose real part is the cosine
     sum and whose imaginary part the negated sine sum. *)
  let on_grid anti t n =
    let len = Nx.dim 0 t in
    let turn =
      let r =
        Nx.init Nx.float64
          [| (n / 2) + 1 |]
          (fun i ->
            Float.of_int ((len - 1) * i.(0) mod (2 * n))
            *. Float.pi /. Float.of_int n)
      in
      Nx.complex Nx.complex128 ~re:(Nx.cos r) ~im:(Nx.sin r)
    in
    let r m = Nx.mul (Nx.rfft Nx.complex128 ~s:[ n ] (moment t m)) turn in
    let arr x = Nx.to_array x in
    let re m s = arr (Nx.mul_s (Nx.real Nx.float64 (r m)) s)
    and im m s = arr (Nx.mul_s (Nx.imag Nx.float64 (r m)) s) in
    if anti then (im 0 1., re 1 (-.two_pi), im 2 (-.(two_pi *. two_pi)))
    else (re 0 1., im 1 two_pi, re 2 (-.(two_pi *. two_pi)))

  let norm1 a = Nx.item [] (Nx.sum (Nx.abs a))
  let norm2 a = Nx.item [] (Nx.sqrt (Nx.sum (Nx.square a)))

  (* A grid of at least [oversampling] points a tap, and at most [subdivisions]
     halvings of an interval the Taylor bound does not clear. *)
  let oversampling = 64
  let subdivisions = 40

  (* [allowance t] is the rounding allowance of [A] on [n] points, below. *)
  let allowance t n =
    let len = Nx.dim 0 t in
    let u = epsilon_float /. 2. in
    let e_n = (16. *. Float.sqrt 2. *. Float.log2 (Float.of_int n)) +. 16. in
    fun m ->
      let tm = moment t m in
      u
      *. ((e_n *. Float.sqrt (Float.of_int n) *. norm2 tm)
         +. (Float.of_int (len + 4) *. norm1 tm))
      *. pow_int two_pi m

  (* [points t] is the grid's size for [t]: the least power of two of at least
     [oversampling] points a tap. *)
  let points t =
    let n = ref 1 in
    while !n < oversampling * Nx.dim 0 t do
      n := 2 * !n
    done;
    !n

  (* [within ~anti t bands] is whether the amplitude of [t] stays within each
     band's tolerance of its target, [(lo, hi, (d₀, d₁), (τ₀, τ₁))] in cycles
     per sample with the target and tolerance linear from [lo] to [hi]:
     certified on every interval between the band's edges and the grid's points
     by the Taylor bound, halving the intervals it does not clear. Every value,
     on the grid or summed directly, carries one rounding allowance, the sum of
     nx.cpu's transform bound [e(n) u √n ‖t jᵐ‖₂], [e(n) ≤ 16 √2 log₂ n + 16],
     and the direct sums' [(taps + 4) u Σ |t_j| |j|ᵐ], times [(2π)ᵐ]. The second
     leaves out the rounding of the phase [2π j x], about [2π u Σ |t_j| |j|ᵐ⁺¹],
     and takes nx's [cos] and [sin] to within a few units of the last place,
     which nx does not state. The allowance stays sound because the transform
     term, added to every value, is the larger below about 10⁷ taps. *)
  let within ~anti t bands =
    let n = points t in
    let eta = allowance t n in
    let eta0 = eta 0 and eta1 = eta 1 and eta2 = eta 2 in
    let m3 = pow_int two_pi 3 *. norm1 (moment t 3) in
    let ga, ga1, ga2 = on_grid anti t n in
    let step = 1. /. Float.of_int n in
    let band (lo, hi, (d0, d1), (t0, t1)) =
      let slope y0 y1 = if hi = lo then 0. else (y1 -. y0) /. (hi -. lo) in
      let target x = d0 +. (slope d0 d1 *. (x -. lo)) in
      let tol x = t0 +. (slope t0 t1 *. (x -. lo)) in
      let d' = slope d0 d1 and t' = Float.abs (slope t0 t1) in
      let err x a = Float.abs (a -. target x) +. eta0 in
      (* A point: its error and its derivatives' magnitudes with allowances. *)
      let point x (a, a1, a2) =
        (err x a, Float.abs (a1 -. d') +. eta1, Float.abs a2 +. eta2)
      in
      let at x = point x (direct anti t x) in
      let bound (e, g1, g2) s =
        e +. (g1 *. s) +. (g2 *. s *. s /. 2.) +. (m3 *. s *. s *. s /. 6.)
      in
      let rec clear depth (xa, pa) (xb, pb) =
        let s = (xb -. xa) /. 2. in
        if bound pa s <= tol xa -. (t' *. s) && bound pb s <= tol xb -. (t' *. s)
        then true
        else if depth = subdivisions then false
        else
          let xm = xa +. s in
          let pm = at xm in
          let e, _, _ = pm in
          e <= tol xm
          && clear (depth + 1) (xa, pa) (xm, pm)
          && clear (depth + 1) (xm, pm) (xb, pb)
      in
      let first = Float.to_int (Float.floor (lo /. step)) + 1 in
      let last = Float.to_int (Float.ceil (hi /. step)) - 1 in
      let grid i =
        let x = Float.of_int i *. step in
        (x, point x (ga.(i), ga1.(i), ga2.(i)))
      in
      let ok (x, (e, _, _)) = e <= tol x in
      let lo_p = (lo, at lo) and hi_p = (hi, at hi) in
      if not (ok lo_p && ok hi_p) then false
      else
        let rec go prev i =
          if i > last then clear 0 prev hi_p
          else
            let p = grid i in
            ok p && clear 0 prev p && go p (i + 1)
        in
        go lo_p first
    in
    List.for_all band bands

  (* Kaiser's formulas: the window's β for [reject] dB, and the half-width for a
     transition of [dw] radians a sample. *)
  let kaiser_beta a =
    if a > 50. then 0.1102 *. (a -. 8.7)
    else if a >= 21. then
      (0.5842 *. H.( ** ) (a -. 21.) 0.4) +. (0.07886 *. (a -. 21.))
    else 0.

  let lowpass ~rate ~pass ~stop ~reject =
    let by = "Omr.Kernel.lowpass" in
    check_rate ~by rate;
    if not (0. < pass && pass < stop && stop <= rate /. 2.) then
      invalid_argf "%s: pass %g, stop %g at rate %g" by pass stop rate;
    if not (reject > 0. && Float.is_finite reject) then
      invalid_argf "%s: reject %g" by reject;
    let delta = H.( ** ) 10. (-.reject /. 20.) in
    let beta = kaiser_beta reject in
    let cutoff = (pass +. stop) /. (2. *. rate) in
    let dw = 2. *. Float.pi *. (stop -. pass) /. rate in
    let first =
      Int.max 1 (Float.to_int (Float.ceil ((reject -. 7.95) /. (4.57 *. dw))))
    in
    let kernel h =
      Sinc { cutoff; half_width = Float.of_int h; window = Kaiser beta }
    in
    let meets h =
      within ~anti:false
        (taps64 (kernel h))
        [
          (0., pass /. rate, (1., 1.), (delta, delta));
          (stop /. rate, 0.5, (0., 0.), (delta, delta));
        ]
    in
    let last = (4 * first) + 64 in
    let rec go h =
      if h > last then
        invalid_argf "%s: no half-width up to %d meets %g dB" by last reject
      else if meets h then kernel h
      else go (h + 1)
    in
    go first
end

(* Tap designs *)

module Taps = struct
  type symmetry = Symmetric | Antisymmetric

  type band = {
    lo : float;
    hi : float;
    gain : float * float;
    tol : float * float;
  }

  (* The basis

     The amplitude of [m] taps of symmetry [s] is [Σₖ cₖ φ(fₖ θ)] for [θ = ω /
     u], [u] the [unit]: [φ] is [cos] for symmetric taps and [sin] for
     antisymmetric ones, and the [fₖ] are integers, the distances [fₖ / u] from
     the centre of the tap pairs. An even [m] has half-integer distances, so its
     unit halves the frequency. Coefficient [cₖ] is twice tap pair [k]'s value,
     its tap after the centre negated when antisymmetric:

     - symmetric, odd [m = 2M + 1]: [f = 0 … M], [u = 1]; - symmetric, even [m]:
     [f = 1, 3, …, m - 1], [u = 2]; - antisymmetric, odd: [f = 1 … M], [u = 1];
     - antisymmetric, even: [f = 1, 3, …, m - 1], [u = 2].

     For a symmetric odd [m] the centre tap is [c₀] itself. *)
  type basis = { s : symmetry; m : int; unit : float; f : int array }

  let basis s m =
    if m mod 2 = 1 then
      let h = m / 2 in
      match s with
      | Symmetric -> { s; m; unit = 1.; f = Array.init (h + 1) Fun.id }
      | Antisymmetric -> { s; m; unit = 1.; f = Array.init h (fun k -> k + 1) }
    else { s; m; unit = 2.; f = Array.init (m / 2) (fun k -> (2 * k) + 1) }

  (* [columns bs w] is the [[points; coefficients]] matrix of [φ(fₖ ω / u)] at
     the frequencies [w], [[points]], float64. *)
  let columns bs w =
    let f =
      Nx.create Nx.float64 [| Array.length bs.f |] (Array.map Float.of_int bs.f)
    in
    let t = Nx.outer (Nx.div_s w bs.unit) f in
    match bs.s with Symmetric -> Nx.cos t | Antisymmetric -> Nx.sin t

  (* [taps dt bs c] is the taps whose amplitude has the coefficients [c]. *)
  let taps dt bs c =
    let c = Nx.to_array c and h = Array.make bs.m 0. in
    let odd = bs.m mod 2 = 1 in
    Array.iteri
      (fun k fk ->
        (* Tap pair at distance [fk / unit] from the centre [(m - 1) / 2]. *)
        let before, after =
          if odd then ((bs.m / 2) - fk, (bs.m / 2) + fk)
          else ((bs.m / 2) - ((fk + 1) / 2), (bs.m / 2) - 1 + ((fk + 1) / 2))
        in
        if fk = 0 then h.(before) <- c.(k)
        else (
          h.(before) <- c.(k) /. 2.;
          h.(after) <-
            (match bs.s with
            | Symmetric -> c.(k) /. 2.
            | Antisymmetric -> -.c.(k) /. 2.)))
      bs.f;
    Nx.cast dt (Nx.create Nx.float64 [| bs.m |] h)

  (* A band in radians a sample, its target and tolerance as functions of the
     frequency, and its place in the list. *)
  type rband = {
    a : float;
    b : float;
    target : float -> float;
    tolerance : float -> float;
    index : int;
  }

  let linear a b (ya, yb) =
    if b = a then fun _ -> ya
    else fun w -> ya +. ((yb -. ya) *. (w -. a) /. (b -. a))

  (* [setup ~by ~rate s bands m] checks the section's rules and gives the basis
     and the bands in radians a sample. *)
  let setup ~by ~rate s bands m =
    check_rate ~by rate;
    if m < 1 then invalid_argf "%s: length %d" by m;
    if bands = [] then invalid_argf "%s: no band" by;
    let half = rate /. 2. in
    let finite x = Float.is_finite x in
    let positive x = x > 0. && finite x in
    let forced_low = s = Antisymmetric in
    let forced_half =
      match s with Symmetric -> m land 1 = 0 | Antisymmetric -> m land 1 = 1
    in
    let _ =
      List.fold_left
        (fun prev bd ->
          let { lo; hi; gain = g0, g1; tol = t0, t1 } = bd in
          if
            not
              (0. <= lo && lo <= hi && hi <= half && finite g0 && finite g1
             && positive t0 && positive t1 && prev < lo)
          then
            invalid_argf "%s: band [%g, %g], gain (%g, %g), tol (%g, %g)" by lo
              hi g0 g1 t0 t1;
          if forced_low && lo = 0. && Float.abs g0 > t0 then
            invalid_argf "%s: gain %g at 0, where the taps' gain is 0" by g0;
          if forced_half && hi = half && Float.abs g1 > t1 then
            invalid_argf "%s: gain %g at %g, where the taps' gain is 0" by g1
              half;
          hi)
        Float.neg_infinity bands
    in
    let w f = if f = half then Float.pi else 2. *. Float.pi *. f /. rate in
    ( basis s m,
      List.mapi
        (fun index { lo; hi; gain; tol } ->
          let a = w lo and b = w hi in
          { a; b; target = linear a b gain; tolerance = linear a b tol; index })
        bands )

  (* Quadrature on a band

     [rule n h] is [n + 1] angles in [(-h, h)] and positive weights that
     integrate every trigonometric polynomial of degree [n] over [\[-h, h\]]
     exactly, for [0 < h ≤ π] (Da Fies and Vianello, "Trigonometric Gaussian
     quadrature on subintervals of the period", ETNA 39, 2012, Proposition 2.1):
     the angles [2 asin (σ ξⱼ)], [σ = sin (h / 2)], and the weights of the Gauss
     rule [(ξⱼ, λⱼ)] for the weight [2σ / √(1 - σ² x²)] on [\[-1, 1\]].

     That Gauss rule comes from the weight's recurrence by Lanczos' process on
     Fejér's first rule of [N] points, which integrates the products of
     polynomials of degree [n + 1] against the weight to rounding: the weight is
     analytic inside the Bernstein ellipse of parameter [ρ = 1/σ + √(1/σ² - 1)],
     where its poles [±1/σ] lie, so Fejér's rule reaches float64's unit roundoff
     [u] about [ln (1/u) / ln ρ] points past the degree; twice that is taken.
     The rule's nodes are the eigenvalues of the recurrence's tridiagonal
     matrix, found by bisection on its Sturm counts, and its weights [λⱼ = μ₀ /
     Σₖ pₖ(ξⱼ)²] over the orthonormal polynomials [pₖ], [μ₀ = 2h] the weight's
     integral. *)

  (* [fejer n] is Fejér's first rule of [n] points on [\[-1, 1\]]: the nodes
     [cos ψᵢ], [ψᵢ = (2i + 1) π / (2n)], and the weights [(2/n) (1 - 2 Σₖ cos
     (2k ψᵢ) / (4k² - 1))] over [1 ≤ k ≤ n / 2], that sum the inverse transform
     of [e^(iπk/n) / (4k² - 1)] at [i]. *)
  let fejer n =
    let k = Nx.arange_f Nx.float64 0. (Float.of_int n) 1. in
    let half = Nx.less_equal_s k (Float.of_int (n / 2)) in
    let a =
      Nx.where
        (Nx.logical_and half (Nx.greater_s k 0.))
        (Nx.recip (Nx.sub_s (Nx.mul_s (Nx.square k) 4.) 1.))
        (Nx.zeros_like k)
    in
    let ph = Nx.mul_s k (Float.pi /. Float.of_int n) in
    let b =
      Nx.complex Nx.complex128
        ~re:(Nx.mul a (Nx.cos ph))
        ~im:(Nx.mul a (Nx.sin ph))
    in
    let s = Nx.real Nx.float64 (Nx.ifft ~norm:Nx.Forward b) in
    let w = Nx.mul_s (Nx.rsub_s 1. (Nx.mul_s s 2.)) (2. /. Float.of_int n) in
    let psi =
      Nx.mul_s (Nx.add_s (Nx.mul_s k 2.) 1.) (Float.pi /. Float.of_int (2 * n))
    in
    (Nx.cos psi, w)

  (* [jacobi x p n] is the off-diagonal [b₁ … bₙ] of the Jacobi matrix of the
     discrete measure of the points [x] and positive masses [p], symmetric about
     [0] so that its diagonal is zero: Lanczos' process from [√p]. *)
  let jacobi x p n =
    let len = Array.length x in
    let mass = Array.fold_left ( +. ) 0. p in
    let q = Array.map (fun pi -> Float.sqrt (pi /. mass)) p in
    let prev = Array.make len 0. and z = Array.make len 0. in
    let b = Array.make (n + 1) 0. in
    for k = 0 to n - 1 do
      let norm = ref 0. in
      for i = 0 to len - 1 do
        z.(i) <- (x.(i) *. q.(i)) -. (b.(k) *. prev.(i));
        norm := !norm +. (z.(i) *. z.(i))
      done;
      b.(k + 1) <- Float.sqrt !norm;
      for i = 0 to len - 1 do
        prev.(i) <- q.(i);
        q.(i) <- z.(i) /. b.(k + 1)
      done
    done;
    Array.sub b 1 n

  (* [below b x] is the number of eigenvalues below [x] of the tridiagonal
     matrix of zero diagonal and off-diagonal [b], by the signs of its LDLᵀ
     pivots (Sturm's count). A zero pivot is taken as the smallest negative
     float's neighbour, as LAPACK's [dstebz] does. *)
  let below b x =
    let count = ref 0 and d = ref (-.x) in
    if !d < 0. then incr count;
    for i = 0 to Array.length b - 1 do
      let prev = if !d = 0. then -.Float.min_float else !d in
      d := -.x -. (b.(i) *. b.(i) /. prev);
      if !d < 0. then incr count
    done;
    !count

  (* [eigen b j] is the [j]-th smallest eigenvalue, in [(-1, 1)], bisected until
     the interval holds no float between its ends. *)
  let eigen b j =
    let rec go lo hi =
      let mid = (lo +. hi) /. 2. in
      if mid <= lo || mid >= hi then mid
      else if below b mid > j then go lo mid
      else go mid hi
    in
    go (-1.) 1.

  let rule n h =
    let s = H.sin (h /. 2.) in
    let rho = (1. /. s) +. Float.sqrt ((1. /. (s *. s)) -. 1.) in
    let u = epsilon_float /. 2. in
    let ln = H.f1 Nx.log in
    let extra = Float.to_int (Float.ceil (2. *. ln (1. /. u) /. ln rho)) in
    let x, v = fejer ((2 * n) + 3 + extra) in
    (* The weight [2σ / √(1 - σ² x²)] at Fejér's nodes. *)
    let wx =
      Nx.mul_s
        (Nx.rsqrt (Nx.rsub_s 1. (Nx.mul_s (Nx.square x) (s *. s))))
        (2. *. s)
    in
    let p = Nx.to_array (Nx.mul v wx) and x = Nx.to_array x in
    let b = jacobi x p n in
    let mu = 2. *. h in
    (* The nodes are symmetric about [0]: the upper half and the middle are
       computed, the lower half mirrored, so that the angles are too. *)
    let xi = Array.make (n + 1) 0. in
    for j = (n + 1) / 2 to n do
      xi.(j) <- (if 2 * j = n then 0. else eigen b j);
      xi.(n - j) <- -.xi.(j)
    done;
    let weight x =
      (* [Σₖ pₖ(x)²] over the orthonormal polynomials, [p₀ = 1]. *)
      let acc = ref 1. and p0 = ref 0. and p1 = ref 1. in
      for k = 0 to n - 1 do
        let back = if k = 0 then 0. else b.(k - 1) *. !p0 in
        let p2 = ((x *. !p1) -. back) /. b.(k) in
        p0 := !p1;
        p1 := p2;
        acc := !acc +. (p2 *. p2)
      done;
      mu /. !acc
    in
    let xi' = Nx.create Nx.float64 [| n + 1 |] xi in
    let theta = Nx.mul_s (Nx.asin (Nx.mul_s xi' s)) 2. in
    (theta, Nx.create Nx.float64 [| n + 1 |] (Array.map weight xi))

  (* Least squares *)

  (* [squares bs rb] is the coefficients minimising the sampled error: on each
     band the rule's angles in [θ = ω / unit], each sample scaled by the square
     root of its weight over the tolerance, solved by QR. Products of two terms
     have degree [2 f_max] in [θ], so a rule of that degree integrates them
     exactly. *)
  let squares bs rb =
    let fmax = bs.f.(Array.length bs.f - 1) in
    let band r =
      let theta, w = rule (2 * fmax) ((r.b -. r.a) /. (2. *. bs.unit)) in
      let om = Nx.add_s (Nx.mul_s theta bs.unit) ((r.a +. r.b) /. 2.) in
      let at f =
        Nx.create Nx.float64 [| Nx.dim 0 om |] (Array.map f (Nx.to_array om))
      in
      let scale = Nx.div (Nx.sqrt (Nx.mul_s w bs.unit)) (at r.tolerance) in
      ( Nx.mul (columns bs om) (Nx.reshape [| -1; 1 |] scale),
        Nx.mul (at r.target) scale )
    in
    let rows = List.map band rb in
    let a = Nx.concatenate ~axis:0 (List.map fst rows)
    and y = Nx.concatenate ~axis:0 (List.map snd rows) in
    let q, r = Nx.qr a in
    Nx.solve_triangular ~triangle:Upper r (Nx.matmul (Nx.transpose q) y)

  (* The error is sampled at each band's angles of the trigonometric Gauss rule,
     scaled by the square roots of their weights over the tolerance, and solved
     by QR, whose conditioning is the square root of the normal equations'. The
     angles integrate the products of the amplitude's terms, and a constant
     target times a term, exactly. *)
  let least_squares dt ~rate s bands m =
    let by = "Omr.Taps.least_squares" in
    let bs, rb = setup ~by ~rate s bands m in
    List.iter
      (fun { lo; hi; _ } ->
        if lo = hi then invalid_argf "%s: a band of one frequency, %g" by lo)
      bands;
    if Array.length bs.f = 0 then Nx.zeros dt [| m |]
    else taps dt bs (squares bs rb)

  (* Equiripple

     The amplitude factors as [A(ω) = Q(ω) P(cos ω)], [P] a polynomial of degree
     [L - 1] for the [L] coefficients and [Q] one of [1], [cos (ω/2)], [sin ω]
     and [sin (ω/2)] for the four symmetries and parities. The error in units of
     the tolerance [τ] is [(A - D) / τ = (Q / τ) (P - D / Q)], a weighted
     polynomial approximation on the bands' frequencies where [Q ≠ 0]. Remez'
     exchange levels it on a reference of [L + 1] frequencies: [P] interpolates
     [D / Q - (-1)ⁱ δ τ / Q] there, in Lagrange's barycentric form (Filip, §3),
     and the reference moves to the error's extrema, found on a grid and refined
     to the nearest float. *)

  (* [factor bs w] is [Q] at [w]. *)
  let factor bs w =
    match (bs.s, bs.m mod 2) with
    | Symmetric, 1 -> 1.
    | Symmetric, _ -> H.cos (w /. 2.)
    | Antisymmetric, 1 -> H.sin w
    | Antisymmetric, _ -> H.sin (w /. 2.)

  (* A frequency of a band, with what the exchange reads there: [x = cos w], the
     target [d = D / Q] and the weight [v = Q / τ]. *)
  type point = { w : float; x : float; d : float; v : float; band : int }

  let point bs (r : rband) w =
    let q = factor bs w in
    {
      w;
      x = H.cos w;
      d = r.target w /. q;
      v = q /. r.tolerance w;
      band = r.index;
    }

  (* Parks and McClellan's program samples the bands at 16 points a coefficient
     over their total width. *)
  let density = 16

  (* [forced bs w] is whether the symmetry and length force [Q], and so the
     amplitude, to zero at [w]: zero frequency for antisymmetric taps, [π] for
     even symmetric and odd antisymmetric ones. [setup] gives these two
     frequencies exactly. *)
  let forced bs w =
    match (bs.s, bs.m land 1) with
    | Antisymmetric, 1 -> w = 0. || w = Float.pi
    | Antisymmetric, _ -> w = 0.
    | Symmetric, 1 -> false
    | Symmetric, _ -> w = Float.pi

  (* [grid bs rb] is the bands' grid, the edges included, without the
     frequencies where [Q] is forced to zero, whose error the bands' rules
     settle. *)
  let grid bs rb =
    let l = Array.length bs.f in
    let width = List.fold_left (fun a r -> a +. (r.b -. r.a)) 0. rb in
    let one (r : rband) =
      let k =
        if r.b = r.a then 0
        else
          Int.max 1
            (Float.to_int
               (Float.ceil
                  (Float.of_int (density * l) *. (r.b -. r.a) /. width)))
      in
      List.init (k + 1) (fun i ->
          if i = k then r.b
          else r.a +. ((r.b -. r.a) *. Float.of_int i /. Float.of_int k))
      |> List.filter (fun w -> not (forced bs w))
      |> List.map (point bs r)
    in
    Array.of_list (List.concat_map one rb)

  (* [bary x] is the barycentric weights [1 / Πⱼ (xᵢ - xⱼ)] of the points [x] up
     to a common factor, each from the sum of the logarithms of [2 |xᵢ - xⱼ|]
     less the largest such sum, so that none overflows. *)
  let bary x =
    let n = Array.length x in
    let v = Nx.create Nx.float64 [| n |] x in
    let d = Nx.sub (Nx.reshape [| n; 1 |] v) (Nx.reshape [| 1; n |] v) in
    let ad = Nx.add (Nx.mul_s (Nx.abs d) 2.) (Nx.eye Nx.float64 n) in
    let l = Nx.neg (Nx.sum ~axes:[ 1 ] (Nx.log ad)) in
    let e = Nx.to_array (Nx.exp (Nx.sub_s l (Nx.item [] (Nx.max l)))) in
    let neg =
      Nx.to_array (Nx.sum ~axes:[ 1 ] (Nx.cast Nx.int32 (Nx.less_s d 0.)))
    in
    Array.mapi (fun i e -> if Int32.to_int neg.(i) land 1 = 0 then e else -.e) e

  (* [level r] is [δ] on the reference [r] and the interpolant [P]. *)
  let level r =
    let l = Array.length r - 1 in
    let x = Array.map (fun p -> p.x) r in
    let g = bary x in
    let sign i = if i land 1 = 0 then 1. else -1. in
    let num = ref 0. and den = ref 0. in
    Array.iteri
      (fun i p ->
        num := !num +. (g.(i) *. p.d);
        den := !den +. (g.(i) *. sign i /. p.v))
      r;
    let delta = !num /. !den in
    (* The values at all [L + 1] points lie on [P], so the second barycentric
       formula over all of them is [P]; leaving one out, as an end, loses digits
       there (Filip, §5.2). *)
    let y = Array.mapi (fun i q -> q.d -. (sign i *. delta /. q.v)) r in
    let p t =
      let rec go i num den =
        if i > l then num /. den
        else
          let dt = t -. x.(i) in
          if dt = 0. then y.(i)
          else go (i + 1) (num +. (g.(i) *. y.(i) /. dt)) (den +. (g.(i) /. dt))
      in
      go 0 0. 0.
    in
    (delta, p)

  let error p q = q.v *. (p q.x -. q.d)

  (* [refine bs rb p a c b] is the extremum of the error between the points
     [a] and [b] of one band around [c], by golden sections until the
     bracket holds no float between its ends. *)
  (* [golden f a b] is where [f] is largest between [a] and [b], by golden
     sections until the bracket holds no float between its ends. *)
  let golden f a b =
    let phi = (Float.sqrt 5. -. 1.) /. 2. in
    let rec go a b =
      let x1 = b -. (phi *. (b -. a)) and x2 = a +. (phi *. (b -. a)) in
      if not (a < x1 && x1 < x2 && x2 < b) then (a +. b) /. 2.
      else if f x1 >= f x2 then go a x2
      else go x1 b
    in
    go a b

  let refine bs rb p a c b =
    let r = List.find (fun (r : rband) -> r.index = c.band) rb in
    let s = if error p c > 0. then 1. else -1. in
    let w = golden (fun w -> s *. error p (point bs r w)) a.w b.w in
    let q = point bs r w in
    if s *. error p q > s *. error p c then q else c

  (* [peaks g e] is the grid indices of the error [e]'s local extrema on each
     band of the grid [g]. *)
  let peaks g e =
    let n = Array.length e in
    let peak i =
      let ok j =
        j < 0 || j >= n || g.(j).band <> g.(i).band
        || if e.(i) > 0. then e.(i) >= e.(j) else e.(i) <= e.(j)
      in
      e.(i) <> 0. && ok (i - 1) && ok (i + 1)
    in
    List.filter peak (List.init n Fun.id)

  (* [alternating c l] is the next reference from the candidates [c], points
     with their errors in increasing frequency: consecutive ones of one sign
     merged into the largest, then the [l + 1] consecutive ones that end at the
     largest, or start the list if it lies before them (Pachón and Trefethen,
     §2.2, the second Remez algorithm); [None] if fewer than [l + 1] remain. *)
  let alternating c l =
    let merged =
      List.fold_left
        (fun acc ((_, v) as x) ->
          match acc with
          | (_, u) :: rest when u > 0. = (v > 0.) ->
              if Float.abs v > Float.abs u then x :: rest else acc
          | _ -> x :: acc)
        [] c
      |> List.rev |> Array.of_list
    in
    let n = Array.length merged in
    if n < l + 1 then None
    else
      let k = ref 0 in
      Array.iteri
        (fun i (_, v) ->
          if Float.abs v > Float.abs (snd merged.(!k)) then k := i)
        merged;
      let start = Int.min (Int.max 0 (!k - l)) (n - l - 1) in
      Some (Array.map fst (Array.sub merged start (l + 1)))

  (* [single r delta q eq] is the reference [r] with the point [q] of largest
     error [eq] exchanged in, the signs still alternating: the first Remez
     algorithm's step (Pachón and Trefethen, "Barycentric-Remez algorithms for
     best polynomial approximation in the chebfun system", BIT 49, 2009, §2.2).
     Point [i]'s error has the sign of [(-1)ⁱ δ], taken as [(-1)ⁱ] when [δ] is
     zero. Inside the reference [q] replaces the neighbour whose sign it has;
     outside, the end it lies beyond if their signs agree, else the other end,
     the rest shifting. The levelled error of the new reference averages its
     points' errors with positive weights, so it exceeds [|δ|] whenever [|eq|]
     does. *)
  let single r delta q eq =
    let l = Array.length r - 1 in
    let sign i =
      let s = if i land 1 = 0 then 1. else -1. in
      if delta < 0. then -.s else s
    in
    let same i = sign i > 0. = (eq > 0.) in
    let j = Array.fold_left (fun n p -> if p.w < q.w then n + 1 else n) 0 r in
    let put i = Array.mapi (fun k p -> if k = i then q else p) r in
    if Array.exists (fun p -> p.w = q.w) r then None
    else if j = 0 then
      Some (if same 0 then put 0 else Array.append [| q |] (Array.sub r 0 l))
    else if j = l + 1 then
      Some (if same l then put l else Array.append (Array.sub r 1 l) [| q |])
    else Some (put (if same (j - 1) then j - 1 else j))

  (* [exchange bs rb g r] runs the exchange from the reference [r] until an
     exchange does not raise [|δ|]: first on the grid [g], then with each
     extremum refined between its grid neighbours. Each exchange moves the
     reference to the error's extrema (Remez' second algorithm), or where they
     are fewer than the reference's points, as when the reference levels no
     error, moves its point of largest error in ([single]). It is the last
     reference and its interpolant. *)
  let exchange bs rb g r =
    let l = Array.length r - 1 in
    let pick ~fine p i =
      if not fine then g.(i)
      else
        let side j =
          if j >= 0 && j < Array.length g && g.(j).band = g.(i).band then g.(j)
          else g.(i)
        in
        refine bs rb p (side (i - 1)) g.(i) (side (i + 1))
    in
    let next ~fine r (delta, p) =
      let e = Array.map (error p) g in
      (* The extrema at least as large as [|δ|], refined when [fine] before they
         are compared, and the reference itself, whose errors are [±δ]. *)
      let found =
        List.filter_map
          (fun i ->
            let q = pick ~fine p i in
            let v = error p q in
            if Float.abs v >= Float.abs delta then Some (q, v) else None)
          (peaks g e)
      and kept =
        List.filter
          (fun (_, v) -> v <> 0.)
          (Array.to_list (Array.map (fun q -> (q, error p q)) r))
      in
      let c =
        List.stable_sort
          (fun (a, _) (b, _) -> Float.compare a.w b.w)
          (found @ kept)
      in
      match alternating c l with
      | Some r' -> Some r'
      | None ->
          let k = ref 0 in
          Array.iteri
            (fun i v -> if Float.abs v > Float.abs e.(!k) then k := i)
            e;
          if not (Float.abs e.(!k) > Float.abs delta) then None
          else
            let q = pick ~fine p !k in
            single r delta q (error p q)
    in
    let rec go ~fine r (delta, p) =
      match next ~fine r (delta, p) with
      | None -> if fine then (r, p) else go ~fine:true r (delta, p)
      | Some r' ->
          let ((delta', _) as lv) = level r' in
          if Float.abs delta' > Float.abs delta then go ~fine r' lv
          else if fine then (r, p)
          else go ~fine:true r (delta, p)
    in
    go ~fine:false r (level r)

  (* [uniform g l] is [l + 1] points of the grid [g] evenly spread by index. *)
  let uniform g l =
    let n = Array.length g in
    Array.init (l + 1) (fun i -> g.(i * (n - 1) / l))

  (* [completed g ws l] is the frequencies [ws] of a smaller design's final
     reference moved to the grid's nearest points and completed to [l + 1] by a
     midpoint in each of its widest gaps, so that each band keeps its share of
     the reference (Filip, §4.3); evenly spread if that cannot be done. *)
  let completed g ws l =
    let n = Array.length g in
    let nearest w =
      let rec go lo hi =
        if hi - lo <= 1 then
          if Float.abs (g.(lo).w -. w) <= Float.abs (g.(hi).w -. w) then lo
          else hi
        else
          let mid = (lo + hi) / 2 in
          if g.(mid).w <= w then go mid hi else go lo mid
      in
      go 0 (n - 1)
    in
    (* One midpoint in each gap, the widest first, as many as are missing; again
       while some remain. *)
    let rec gaps = function
      | a :: (b :: _ as rest) ->
          if b - a >= 2 then (a, b) :: gaps rest else gaps rest
      | _ -> []
    in
    let rec fill s =
      let missing = l + 1 - List.length s in
      if missing <= 0 then Some s
      else
        let wide =
          List.stable_sort
            (fun (a, b) (c, d) -> compare (d - c) (b - a))
            (gaps s)
        in
        if wide = [] then None
        else
          let mids =
            List.filteri (fun i _ -> i < missing) wide
            |> List.map (fun (a, b) -> (a + b) / 2)
          in
          fill (List.sort_uniq compare (mids @ s))
    in
    let s = List.sort_uniq compare (Array.to_list (Array.map nearest ws)) in
    match fill s with
    | Some s -> Array.of_list (List.map (fun i -> g.(i)) s)
    | None -> uniform g l

  (* [reference bs rb] is the final reference of [bs]'s design and its
     interpolant, starting from the reference of the design of half as many
     coefficients, recursively, or evenly spread at two or fewer. *)
  let rec reference bs rb =
    let l = Array.length bs.f in
    let g = grid bs rb in
    if Array.length g < l + 1 then None
    else
      let start =
        if l <= 2 then uniform g l
        else
          let l' = ((l - 1) / 2) + 1 in
          let m' =
            match (bs.s, bs.m land 1) with
            | Symmetric, 1 -> (2 * l') - 1
            | Antisymmetric, 1 -> (2 * l') + 1
            | _ -> 2 * l'
          in
          match reference (basis bs.s m') rb with
          | Some (r', _) -> completed g (Array.map (fun p -> p.w) r') l
          | None -> uniform g l
      in
      Some (exchange bs rb g start)

  (* Filip's convergence threshold for the gap between the largest error and the
     levelled one (TOMS 2016, §7.1). *)
  let convergence = 0.01

  (* [remez ~by bs rb] is the coefficients of the final interpolant and, at each
     point of the final reference, the magnitude of their error and the
     tolerance's reciprocal. It raises [Failure] unless that reference's
     frequencies increase and its errors alternate in sign, and the taps'
     largest error is within [convergence] of the least on it. *)
  let remez ~by bs rb =
    let l = Array.length bs.f in
    match reference bs rb with
    | None -> failwith (Printf.sprintf "%s: fewer frequencies than values" by)
    | Some (r, p) ->
        (* The coefficients from the amplitude at [ω_j = π (2j + 1) / (2L)],
           where the basis is a discrete cosine or sine transform's and so well
           conditioned. *)
        let wj =
          Nx.init Nx.float64 [| l |] (fun j ->
              Float.pi *. Float.of_int ((2 * j.(0)) + 1) /. Float.of_int (2 * l))
        in
        let a =
          Nx.create Nx.float64 [| l |]
            (Array.map (fun w -> factor bs w *. p (H.cos w)) (Nx.to_array wj))
        in
        let c = Nx.solve (columns bs wj) a in
        (* The error of these coefficients on the reference, [(A - D) / τ = v (A
           / Q - d)]. *)
        let wr =
          Nx.create Nx.float64 [| l + 1 |] (Array.map (fun q -> q.w) r)
        in
        let amp = Nx.to_array (Nx.matmul (columns bs wr) c) in
        let e =
          Array.mapi (fun i q -> q.v *. ((amp.(i) /. factor bs q.w) -. q.d)) r
        in
        for i = 0 to l - 1 do
          if not (e.(i) *. e.(i + 1) < 0. && r.(i).w < r.(i + 1).w) then
            failwith
              (Printf.sprintf
                 "%s: the error does not alternate on the final reference" by)
        done;
        (* The taps' largest error, searched apart from the reference: on the
           grid, each local extremum refined by golden sections. Through an
           ill-conditioned reference the taps recovered from [P] can stray far
           from it between the reference's points; Filip's convergence test
           (§7.1, from Parks and Burrus) bounds the gap, [(E - |δ|) / E ≤ 0.01],
           where the least error on the reference stands for [|δ|]. *)
        let g = grid bs rb in
        let errs ws qs =
          let a =
            Nx.to_array
              (Nx.matmul
                 (columns bs (Nx.create Nx.float64 [| Array.length ws |] ws))
                 c)
          in
          Array.mapi (fun i q -> q.v *. ((a.(i) /. factor bs q.w) -. q.d)) qs
        in
        let chunk = 1024 in
        let eg =
          Array.concat
            (List.init
               ((Array.length g + chunk - 1) / chunk)
               (fun k ->
                 let qs =
                   Array.sub g (k * chunk)
                     (Int.min chunk (Array.length g - (k * chunk)))
                 in
                 errs (Array.map (fun q -> q.w) qs) qs))
        in
        let at (q : point) w =
          let r = List.find (fun (r : rband) -> r.index = q.band) rb in
          let q' = point bs r w in
          (errs [| w |] [| q' |]).(0)
        in
        let largest =
          List.fold_left
            (fun m i ->
              let side j =
                if j >= 0 && j < Array.length g && g.(j).band = g.(i).band then
                  g.(j)
                else g.(i)
              in
              let w =
                golden
                  (fun w -> Float.abs (at g.(i) w))
                  (side (i - 1)).w (side (i + 1)).w
              in
              Float.max m
                (Float.max (Float.abs eg.(i)) (Float.abs (at g.(i) w))))
            0. (peaks g eg)
        in
        let lower =
          Array.fold_left
            (fun m x -> Float.min m (Float.abs x))
            Float.infinity e
        in
        if largest -. lower > convergence *. largest then
          failwith
            (Printf.sprintf
               "%s: the largest error %g is more than %g of it above the \
                levelled error %g"
               by largest convergence lower);
        (c, Array.mapi (fun i q -> (Float.abs e.(i), q.v /. factor bs q.w)) r)

  let equiripple dt ~rate s bands m =
    let by = "Omr.Taps.equiripple" in
    let bs, rb = setup ~by ~rate s bands m in
    if Array.length bs.f = 0 then Nx.zeros dt [| m |]
    else taps dt bs (fst (remez ~by bs rb))

  (* Minimum phase *)

  let minimum_phase dt ~length g =
    let by = "Omr.Taps.minimum_phase" in
    if Nx.ndim g <> 1 || Nx.dim 0 g < 2 then
      invalid_argf "%s: gain %a is not [k] with k > 1" by pp_shape (Nx.shape g);
    let g = Nx.cast Nx.float64 g in
    let n = 2 * (Nx.dim 0 g - 1) in
    if length < 1 || length > n then
      invalid_argf "%s: length %d outside [1, %d]" by length n;
    let v = Nx.to_array g in
    if Array.exists (fun x -> not (Float.is_finite x && x >= 0.)) v then
      invalid_argf "%s: a gain is negative or not finite" by;
    let top = Array.fold_left Float.max 0. v in
    if top = 0. then invalid_argf "%s: every gain is 0" by;
    (* The real cepstrum of the gain floored at [u] times its largest, its
       negative quefrencies folded onto the positive: [c₀], [2 cₖ] for [0 < k <
       n / 2], [c_(n/2)], zero after. *)
    let floor = Nx.maximum_s g (epsilon_float /. 2. *. top) in
    let c =
      Nx.irfft Nx.float64 ~s:[ n ]
        (Nx.complex Nx.complex128 ~re:(Nx.log floor) ~im:(Nx.zeros_like floor))
    in
    let fold =
      Nx.init Nx.float64 [| n |] (fun i ->
          let k = i.(0) in
          if k = 0 || k = n / 2 then 1. else if k < n / 2 then 2. else 0.)
    in
    let s = Nx.rfft Nx.complex128 (Nx.mul c fold) in
    (* [exp] of the transform, [e^re (cos im + i sin im)], inverted. *)
    let m = Nx.exp (Nx.real Nx.float64 s) and ph = Nx.imag Nx.float64 s in
    let h =
      Nx.irfft Nx.float64 ~s:[ n ]
        (Nx.complex Nx.complex128
           ~re:(Nx.mul m (Nx.cos ph))
           ~im:(Nx.mul m (Nx.sin ph)))
    in
    Nx.cast dt (Nx.slice [ Nx.R (0, length) ] h)

  (* Savitzky-Golay *)

  (* [legendre x ~order ~deriv] is the Legendre polynomials' [deriv]-th
     derivatives [P_k⁽ᵈ⁾(x)] for [k ≤ order], by [P_k = ((2k - 1) x P_(k-1) - (k
     - 1) P_(k-2)) / k] and [P_k⁽ᵈ⁾ = P_(k-2)⁽ᵈ⁾ + (2k - 1) P_(k-1)⁽ᵈ⁻¹⁾]. *)
  let legendre x ~order ~deriv =
    let t = Array.make_matrix (deriv + 1) (order + 1) 0. in
    for d = 0 to deriv do
      for k = 0 to order do
        t.(d).(k) <-
          (if d = 0 then
             if k = 0 then 1.
             else if k = 1 then x
             else
               ((Float.of_int ((2 * k) - 1) *. x *. t.(0).(k - 1))
               -. (Float.of_int (k - 1) *. t.(0).(k - 2)))
               /. Float.of_int k
           else if k < d then 0.
           else
             (if k >= 2 then t.(d).(k - 2) else 0.)
             +. (Float.of_int ((2 * k) - 1) *. t.(d - 1).(k - 1)))
      done
    done;
    t.(deriv)

  (* The fit is computed from the QR factors of the Legendre polynomials of the
     positions scaled to [[-1, 1]], whose conditioning, unlike scaled
     monomials', does not grow with the order. *)
  let savitzky_golay dt ~window ~order ~deriv ~at =
    let by = "Omr.Taps.savitzky_golay" in
    if order < 0 || deriv < 0 || window <= order || not (Float.is_finite at)
    then
      invalid_argf "%s: window %d, order %d, deriv %d, at %g" by window order
        deriv at;
    (* Positions scaled to [\[-1, 1\]], [x = (2i - (w - 1)) / (w - 1)]. *)
    let half = if window = 1 then 1. else Float.of_int (window - 1) /. 2. in
    let scaled i = (i -. ((Float.of_int window -. 1.) /. 2.)) /. half in
    let v =
      Nx.create Nx.float64
        [| window; order + 1 |]
        (Array.concat
           (List.init window (fun i ->
                legendre (scaled (Float.of_int i)) ~order ~deriv:0)))
    in
    let q, r = Nx.qr v in
    (* The fit's [d]-th derivative at [at] is [gᵀ a] for the Legendre series'
       coefficients [a = R⁻¹ Qᵀ y], [g] the basis' derivatives there times [(1 /
       half)ᵈ]; its row is [Q R⁻ᵀ g]. *)
    let scale = ref 1. in
    for _ = 1 to deriv do
      scale := !scale /. half
    done;
    let g =
      Array.map (fun p -> p *. !scale) (legendre (scaled at) ~order ~deriv)
    in
    let z =
      Nx.solve_triangular ~triangle:Upper ~transpose:Transpose r
        (Nx.create Nx.float64 [| order + 1 |] g)
    in
    Nx.cast dt (Nx.flip (Nx.matmul q z))

  (* Least length *)

  let meet ~rate ~odd s bands =
    let by = "Omr.Taps.meet" in
    let first =
      match (s, odd) with Antisymmetric, true -> 3 | _, true -> 1 | _ -> 2
    in
    ignore (setup ~by ~rate s bands first);
    (* Where the symmetry and parity force the gain to 0, the error is the
       target's magnitude, which the bound's rounding allowance pushes past a
       tolerance it equals. *)
    let forced_low = s = Antisymmetric
    and forced_half = s = Antisymmetric = odd in
    List.iter
      (fun { lo; hi; gain = g0, g1; tol = t0, t1 } ->
        let at f g t = f && Float.abs g >= t in
        if
          at (forced_low && lo = 0.) g0 t0
          || at (forced_half && hi = rate /. 2.) g1 t1
        then
          invalid_argf
            "%s: a band's target is its tolerance away from 0 where the taps' \
             gain is forced to 0, so no length can be certified"
            by)
      bands;
    let cycles =
      List.map
        (fun { lo; hi; gain; tol } -> (lo /. rate, hi /. rate, gain, tol))
        bands
    in
    let least =
      List.fold_left
        (fun a { tol = t0, t1; _ } -> Float.min a (Float.min t0 t1))
        Float.infinity bands
    in
    (* A length's status: its design certified to meet the bands, proved too
       short by its alternating error on the final reference, or neither. *)
    let memo = Hashtbl.create 16 in
    let status m =
      match Hashtbl.find_opt memo m with
      | Some v -> v
      | None ->
          let bs, rb = setup ~by ~rate s bands m in
          let c, errors =
            remez ~by:(Printf.sprintf "%s, length %d" by m) bs rb
          in
          let t = taps Nx.float64 bs c in
          let eta = Kernel.allowance t (Kernel.points t) 0 in
          if eta >= least then
            invalid_argf "%s: no length past %d can be certified" by m;
          (* De la Vallée Poussin's bound holds for the exact errors: each
             computed one, less the amplitude's rounding allowance in units of
             the tolerance, must exceed 1. *)
          let short =
            Array.for_all (fun (e, inv) -> e -. (eta *. inv) > 1.) errors
          in
          let v =
            if Kernel.within ~anti:(s = Antisymmetric) t cycles then `Meets
            else if short then `Short
            else `Unknown
          in
          Hashtbl.add memo m v;
          v
    in
    (* Lengths of the parity are searched by doubling, then by bisection between
       the last that did not meet the bands and the first that did; the answer
       is then proved least at [m - 2]. *)
    let next m = (2 * m) + if odd then 1 else 0 in
    let rec double lo m =
      if status m = `Meets then (lo, m) else double (Some m) (next m)
    in
    let rec bisect lo hi =
      if hi - lo <= 2 then hi
      else
        let mid = (lo + hi) / 2 in
        let mid = if mid land 1 = 1 = odd then mid else mid + 1 in
        if status mid = `Meets then bisect lo mid else bisect mid hi
    in
    let m =
      match double None first with
      | lo, hi -> ( match lo with None -> hi | Some lo -> bisect lo hi)
    in
    if m = first || status (m - 2) = `Short then m
    else
      failwith
        (Printf.sprintf
           "%s: length %d is neither certified nor proved too short" by (m - 2))
end

(* Resampling *)

module Resample = struct
  type t = {
    up : int;
    down : int;
    edge : [ edge | `Renormalise ];
    centres : int array * int;
        (* [c_j] for [j < up], as numerators over a common denominator *)
    first : int array; (* each phase's first input, output [j]'s from [0] *)
    raw : float array array; (* each phase's weights, [k (τ) / s] *)
    used : float array array; (* the weights resampling applies *)
  }

  let rec gcd a b = if b = 0 then a else gcd b (a mod b)

  (* [fdiv a b] is [⌊a / b⌋] for [b > 0]. *)
  let fdiv a b = if a >= 0 then a / b else -((-a + b - 1) / b)

  let v ~grid ~edge ~kernel ~by:(l, m) =
    if l < 1 || m < 1 then invalid_argf "Omr.Resample.v: by (%d, %d)" l m;
    let g = gcd l m in
    let l = l / g and m = m / g in
    let shrink = m > l in
    let s = if shrink then Float.of_int m /. Float.of_int l else 1. in
    let sh = s *. Kernel.half_width kernel in
    (* [c_j = num j / den]: [j m / l] on points, [((2j + 1) m - l) / 2l] on
       cells. *)
    let den = match grid with `Points -> l | `Cells -> 2 * l in
    let num j =
      match grid with `Points -> j * m | `Cells -> (((2 * j) + 1) * m) - l
    in
    (* [τ = (c_j - i) / s = (num - i den) / dd], [dd] an integer. *)
    let dd = if shrink then den * m / l else den in
    let phase j =
      let c = Float.of_int (num j) /. Float.of_int den in
      let lo = Float.to_int (Float.floor (c -. sh)) + 1 in
      let hi = Float.to_int (Float.floor (c +. sh)) in
      let tau =
        Array.init
          (hi - lo + 1)
          (fun k -> Float.of_int (num j - ((lo + k) * den)) /. Float.of_int dd)
      in
      let v =
        Kernel.values kernel (Nx.create Nx.float64 [| Array.length tau |] tau)
      in
      (lo, Nx.to_array (Nx.div_s v s))
    in
    let phases = Array.init l phase in
    let raw = Array.map snd phases in
    let normalised w =
      let t = Array.fold_left ( +. ) 0. w in
      if t = 0. then w else Array.map (fun v -> v /. t) w
    in
    let used =
      match edge with `Renormalise -> Array.map normalised raw | _ -> raw
    in
    {
      up = l;
      down = m;
      edge;
      centres = (Array.init l num, den);
      first = Array.map fst phases;
      raw;
      used;
    }

  let count r n = ((n * r.up) + r.down - 1) / r.down

  let weights r =
    let nums, den = r.centres in
    Array.init r.up (fun j ->
        (r.first.(j) - fdiv nums.(j) den, Array.copy r.used.(j)))

  (* Output [u]'s first and last input. *)
  let first r u = r.first.(u mod r.up) + (u / r.up * r.down)
  let last r u = first r u + Array.length r.raw.(u mod r.up) - 1

  (* [completed r p] is the number of outputs whose last input is below [p]. *)
  let completed r p =
    let total = ref 0 in
    for j = 0 to r.up - 1 do
      let b = r.first.(j) + Array.length r.raw.(j) - 1 in
      total := !total + Int.max 0 (fdiv (p - 1 - b) r.down + 1)
    done;
    !total

  (* [reads r w] is the samples an end extended by [w] reads. *)
  let reads r w = match r.edge with `Renormalise -> 0 | #edge as e -> need e w

  (* [extended r (lo, hi) x] is [x] extended under [r]'s edge, with zeros under
     [`Renormalise]. *)
  let extended r ends x =
    match r.edge with
    | `Renormalise -> Nx.extend `Zero ends x
    | #edge as e -> Nx.extend (e :> Nx.extension) ends x

  (* How far past a signal's end its outputs read, at most: the last output
     starts within [down] samples past the end. *)
  let reach r =
    Array.fold_left (fun m w -> Int.max m (Array.length w)) 0 r.raw + r.down + 1

  (* [row r e w] is the weights [w], [[1; k; 1]], in [e]'s dtype beside it. *)
  let row e w =
    Nx.place (Nx.beside e)
      (Nx.cast (Nx.dtype e) (Nx.create Nx.float64 [| 1; Array.length w; 1 |] w))

  (* [outputs r ~head ~n e base u0 u1] is outputs [u0] to [u1 - 1] of the
     samples [e], [[rows; len]], whose first lies at position [base]. Positions
     below [0] lie before the signal where [head], positions from [n] past its
     end where [n] is given. Under [`Renormalise] the outputs reading those are
     computed alone, from their weights inside the signal summed to [1]; the
     others are one strided reduction a phase. *)
  let outputs r ~head ~n e base u0 u1 =
    let rows = Nx.dim 0 e in
    let renormalise = r.edge = `Renormalise in
    let before u = renormalise && head && first r u < 0 in
    let after u =
      renormalise && match n with Some n -> last r u >= n | None -> false
    in
    let ia = ref u0 in
    while !ia < u1 && before !ia do
      incr ia
    done;
    let ib = ref u1 in
    while !ib > !ia && after (!ib - 1) do
      decr ib
    done;
    let alone u =
      let a = first r u and w = r.raw.(u mod r.up) in
      let lo = if head then Int.max a 0 else a in
      let hi =
        match n with
        | Some n -> Int.min (a + Array.length w) n
        | None -> a + Array.length w
      in
      if hi <= lo then Nx.zeros_at (Nx.beside e) (Nx.dtype e) [| rows; 1 |]
      else
        let w = Array.sub w (lo - a) (hi - lo) in
        let t = Array.fold_left ( +. ) 0. w in
        let w = if t = 0. then w else Array.map (fun v -> v /. t) w in
        let win = along e (lo - base) (hi - base) in
        Nx.sum ~axes:[ -1 ] ~keepdims:true
          (Nx.mul win (Nx.reshape [| 1; Array.length w |] (row e w)))
    in
    let strided u0 u1 =
      let cnt = u1 - u0 in
      let lp = Int.min r.up cnt and k0 = (cnt + r.up - 1) / r.up in
      let phase rho =
        let u = u0 + rho in
        let w = r.used.(u mod r.up) in
        let k = (cnt - rho + r.up - 1) / r.up in
        let start = first r u - base in
        let len = ((k - 1) * r.down) + Array.length w in
        let win = Nx.reshape [| rows; len; 1 |] (along e start (start + len)) in
        let y =
          Nx.reshape [| rows; k |] (Nx.conv ~stride:[| r.down |] win (row e w))
        in
        if k = k0 then y
        else Nx.pad [| (0, 0); (0, k0 - k) |] (Nx.Dtype.zero (Nx.dtype e)) y
      in
      let y = Nx.stack ~axis:(-1) (List.init lp phase) in
      along (Nx.reshape [| rows; k0 * lp |] y) 0 cnt
    in
    let parts =
      List.init (!ia - u0) (fun i -> alone (u0 + i))
      @ (if !ib > !ia then [ strided !ia !ib ] else [])
      @ List.init (u1 - !ib) (fun i -> alone (!ib + i))
    in
    match parts with
    | [] -> Nx.zeros_at (Nx.beside e) (Nx.dtype e) [| rows; 0 |]
    | ps -> Nx.concatenate ~axis:(-1) ps

  (* A state holds the samples from position [from] that later outputs read.
     [Head] holds the signal from its first sample, [from = 0], and the next
     output [next], while outputs that read before the start wait or the signal
     is shorter than the end's extension reads. [Running] counts positions and
     outputs from the start of the next output's period: [from = offset], the
     next output [phase]. *)
  module State = struct
    type ('a, _) t =
      | Start
      | Head of { samples : 'a; next : int }
      | Running of { samples : 'a; phase : int; offset : int }

    let walk c s =
      let open Nx.Ptree.Walk in
      match s with
      | Start ->
          case c "start";
          Start
      | Head { samples; next } ->
          case c "head";
          let samples = field c "samples" leaf samples in
          let next = field c "next" int next in
          Head { samples; next }
      | Running { samples; phase; offset } ->
          case c "running";
          let samples = field c "samples" leaf samples in
          let phase = field c "phase" int phase in
          let offset = field c "offset" int offset in
          Running { samples; phase; offset }
  end

  let start = State.Start

  (* [flat x] is [x], [[lead; n]], as [[rows; n]], and its lead. *)
  let flat x =
    let lead = lead x 1 in
    (Nx.reshape [| Array.fold_left ( * ) 1 lead; Nx.dim (-1) x |] x, lead)

  let unflat lead y = Nx.reshape (Array.append lead [| Nx.dim (-1) y |]) y

  let feed_last ~by r s x =
    let kept, from, next, head =
      match s with
      | State.Start -> (x, 0, 0, true)
      | Head { samples; next } -> (Frames.join ~by samples x, 0, next, true)
      | Running { samples; phase; offset } ->
          (Frames.join ~by samples x, offset, phase, false)
    in
    let known = from + Nx.dim (-1) kept in
    let total = completed r known in
    (* At the start, wait for the samples an output's extension reads. *)
    let stop =
      if not head then total
      else
        let rec go u =
          if u >= total || first r u >= 0 then total
          else if reads r (-first r u) > known then u
          else go (u + 1)
        in
        go next
    in
    let flat_kept, lead = flat kept in
    let y =
      if stop <= next then
        Nx.zeros_at (Nx.beside kept) (Nx.dtype kept) [| Nx.dim 0 flat_kept; 0 |]
      else
        let lo = if head then Int.max 0 (-first r next) else 0 in
        let e = if lo > 0 then extended r (lo, 0) flat_kept else flat_kept in
        outputs r ~head ~n:None e (from - lo) next stop
    in
    let keep = Int.min (first r stop) (known - reads r (reach r)) in
    let at = Int.max from (Int.min keep known) in
    let samples = along kept (at - from) (known - from) in
    let s =
      if head && at = 0 then State.Head { samples; next = stop }
      else
        State.Running
          {
            samples;
            phase = stop mod r.up;
            offset = at - (stop / r.up * r.down);
          }
    in
    (s, unflat lead y)

  let finish_last ~by r s =
    let kept, from, next, head =
      match s with
      | State.Start ->
          invalid_argf "%s: no chunk was fed; feed an empty chunk first" by
      | Head { samples; next } -> (samples, 0, next, true)
      | Running { samples; phase; offset } -> (samples, offset, phase, false)
    in
    let n = from + Nx.dim (-1) kept in
    let total = count r n in
    let flat_kept, lead = flat kept in
    let y =
      if total <= next then
        Nx.zeros_at (Nx.beside kept) (Nx.dtype kept) [| Nx.dim 0 flat_kept; 0 |]
      else begin
        if Nx.dim (-1) kept = 0 && r.edge <> `Zero && r.edge <> `Renormalise
        then invalid_argf "%s: no sample to read past the ends" by;
        let lo = if head then Int.max 0 (-first r next) else 0 in
        let hi = Int.max 0 (last r (total - 1) - (n - 1)) in
        outputs r ~head ~n:(Some n)
          (extended r (lo, hi) flat_kept)
          (from - lo) next total
      end
    in
    unflat lead y

  let feed ?(axis = -1) r s x =
    let by = "Omr.Resample.feed" in
    let s, y = feed_last ~by r s (time ~by axis x) in
    (s, untime ~by axis y)

  let finish ?(axis = -1) r s =
    let by = "Omr.Resample.finish" in
    untime ~by axis (finish_last ~by r s)
end

let resample ?(axis = -1) r x =
  let by = "Omr.resample" in
  let x = time ~by axis x in
  let s, y = Resample.feed_last ~by r Resample.start x in
  untime ~by axis
    (Nx.concatenate ~axis:(-1) [ y; Resample.finish_last ~by r s ])
