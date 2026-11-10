(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Bigarray

type counts = { s2 : Nx.host Nx.int32_t; s1 : Nx.host Nx.int32_t }

(* Counts *)

(* The reference's bins are the multiples of 8 up to 256 (tessera_infer_v2
   student/infer.py:40), and it fills a pixel's [n] observations to its bin with
   [_pad_pattern] (:60-72). *)
let bin_step = 8
let max_bin = 256
let bin n = min max_bin ((n + bin_step - 1) / bin_step * bin_step)

(* Adds to [c.{base + i}] the counts of [_pad_pattern n (bin n)]. *)
let fill (c : (int32, int32_elt) Flat.t) base n =
  let add i = c.{base + i} <- Int32.succ c.{base + i} in
  let b = bin n in
  if n >= b then begin
    (* numpy's [linspace (0, n − 1, b)] at int64: [j · step] in float64,
       floored, and the last point [n − 1] itself. *)
    let step = Float.of_int (n - 1) /. Float.of_int (b - 1) in
    for j = 0 to b - 2 do
      add (Float.to_int (Float.of_int j *. step))
    done;
    add (n - 1)
  end
  else begin
    for i = 0 to n - 1 do
      add i
    done;
    let r = b - n in
    if r <= n then begin
      (* [np.array_split (arange n) r]: the first [n mod r] runs are one
         longer. *)
      let q = n / r and long = n mod r in
      for k = 0 to r - 1 do
        let start = (k * q) + min k long in
        let size = if k < long then q + 1 else q in
        add (start + (size / 2))
      done
    end
    else
      for i = 0 to r - 1 do
        add (i mod n)
      done
  end

let sensor_counts (type v s) (r : (v, s, Nx.host) Nx_ragged.t) =
  let len = Nx.dim 0 (Nx_ragged.values r) in
  let c = Flat.create int32 len 0l in
  Flat.read (Nx_ragged.offsets r) (fun o ->
      for p = 0 to Nx_ragged.length r - 1 do
        let base = Int64.to_int o.{p} in
        let n = Int64.to_int o.{p + 1} - base in
        if n > 0 then fill c base n
      done);
  Flat.wrap Nx.int32 [| len |] c

let v2_counts (px : Nx.host Pixels.t) =
  { s2 = sensor_counts px.s2; s1 = sensor_counts px.s1 }

(* Duplicate filling *)

(* A pixel's tokens in passes: pass [k] holds, in order, its observations of
   count at least [k]. Under v2's counts this is the reference's fill order,
   every observation once and then the fill ([_pad_pattern]'s [concatenate
   ([arange n], fill)]). *)
let expand_sensor (type s) kind dt (r : (int, s, Nx.host) Nx_ragged.t)
    (c : Nx.host Nx.int32_t) =
  let n = Nx_ragged.length r in
  let cells = Nx.dim 1 (Nx_ragged.values r) in
  let o' = Flat.create int64 (n + 1) 0L in
  let v' =
    Flat.read (Nx_ragged.offsets r) @@ fun o ->
    Flat.read c @@ fun c ->
    for p = 0 to n - 1 do
      let t = ref o'.{p} in
      for j = Int64.to_int o.{p} to Int64.to_int o.{p + 1} - 1 do
        t := Int64.add !t (Int64.of_int32 c.{j})
      done;
      o'.{p + 1} <- !t
    done;
    let v' = Flat.create kind (Int64.to_int o'.{n} * cells) 0 in
    Flat.read (Nx_ragged.values r) @@ fun v ->
    let at = ref 0 in
    for p = 0 to n - 1 do
      let lo = Int64.to_int o.{p} and hi = Int64.to_int o.{p + 1} in
      let most = ref 0l in
      for j = lo to hi - 1 do
        if c.{j} > !most then most := c.{j}
      done;
      for k = 1 to Int32.to_int !most do
        for j = lo to hi - 1 do
          if Int32.to_int c.{j} >= k then begin
            for ch = 0 to cells - 1 do
              v'.{(!at * cells) + ch} <- v.{(j * cells) + ch}
            done;
            incr at
          end
        done
      done
    done;
    v'
  in
  let total = Int64.to_int o'.{n} in
  ( Nx_ragged.v
      ~offsets:(Flat.wrap Nx.int64 [| n + 1 |] o')
      (Flat.wrap dt [| total; cells |] v'),
    Nx.ones Nx.int32 [| total |] )

let expand (px : Nx.host Pixels.t) (c : counts) =
  if
    Nx.shape c.s2 <> [| Nx.dim 0 (Nx_ragged.values px.s2) |]
    || Nx.shape c.s1 <> [| Nx.dim 0 (Nx_ragged.values px.s1) |]
  then invalid_arg "Tessera.Batch.expand: one count per observation expected";
  let s2, c2 = expand_sensor int16_unsigned Nx.uint16 px.s2 c.s2 in
  let s1, c1 = expand_sensor int16_signed Nx.int16 px.s1 c.s1 in
  (({ s2; s1 } : Nx.host Pixels.t), ({ s2 = c2; s1 = c1 } : counts))

(* Batches *)

type shape = {
  pixels : int;
  s2_rows : int;
  s1_rows : int;
  slots : int;
  width : int;
}

type 'd rows = {
  tokens : 'd Nx.float32_t;
  log_count : 'd Nx.float32_t;
  start : 'd Nx.int32_t;
  stop : 'd Nx.int32_t;
  slot_start : 'd Nx.int32_t;
  slot_stop : 'd Nx.int32_t;
  place : 'd Nx.int64_t;
  at : 'd Nx.int64_t;
}

type 'd t = { s2 : 'd rows; s1 : 'd rows; pixel : 'd Nx.int64_t }

let invalid fmt =
  Printf.ksprintf (fun m -> invalid_arg ("Tessera.Batch.pack: " ^ m)) fmt

(* One sensor's rows as the host fills them: [fill.(r)] tokens and [used.(r)]
   slots of row [r] are taken. *)
type builder = {
  rows : int;
  cells : int;
  bands : int;
  toks : (float, float32_elt) Flat.t;
  logc : (float, float32_elt) Flat.t;
  starts : (int32, int32_elt) Flat.t;
  stops : (int32, int32_elt) Flat.t;
  slot_starts : (int32, int32_elt) Flat.t;
  slot_stops : (int32, int32_elt) Flat.t;
  token_places : (int64, int64_elt) Flat.t;
  places : (int64, int64_elt) Flat.t;
  fill : int array;
  used : int array;
  mutable first_open : int; (* rows below are full *)
}

(* Channels of a token: the bands, the day, the source. *)
let channels b = b.bands + 2

(* An observation's cells are its bands, its day and, for Sentinel-1, its orbit,
   which is the token's source: a token's channels are the first [cells] of
   [channels] and, for Sentinel-2, a source of [0]. *)
let builder shape ~rows ~bands =
  let c = bands + 2 in
  let cells = if bands = Pixels.s1_bands then c else bands + 1 in
  let w = shape.width in
  {
    rows;
    cells;
    bands;
    toks = Flat.create float32 (rows * w * c) 0.;
    logc = Flat.create float32 (rows * w) 0.;
    starts = Flat.create int32 (rows * w) 0l;
    stops = Flat.create int32 (rows * w) 0l;
    slot_starts = Flat.create int32 (rows * shape.slots) 0l;
    slot_stops = Flat.create int32 (rows * shape.slots) 0l;
    token_places = Flat.create int64 (rows * w) 0L;
    places = Flat.create int64 shape.pixels 0L;
    fill = Array.make rows 0;
    used = Array.make rows 0;
    first_open = 0;
  }

(* The first row with room for [t] tokens and a slot, if any. *)
let row shape b t =
  while
    b.first_open < b.rows
    && (b.fill.(b.first_open) >= shape.width
       || b.used.(b.first_open) >= shape.slots)
  do
    b.first_open <- b.first_open + 1
  done;
  let rec find r =
    if r >= b.rows then -1
    else if b.fill.(r) + t <= shape.width && b.used.(r) < shape.slots then r
    else find (r + 1)
  in
  find b.first_open

(* Writes the tokens of observations [lo, hi) of nonzero count, [t] of them, or
   a stand-in if [t = 0], in row [r] as pixel place [place]'s slot. *)
let put (type s) shape b r ~place ~(values : (int, s) Flat.t)
    ~(counts : (int32, int32_elt) Flat.t) lo hi t =
  let w = shape.width and c = channels b in
  let first = b.fill.(r) in
  let n = max t 1 in
  let stop = Int32.of_int (first + n) and start = Int32.of_int first in
  for k = 0 to n - 1 do
    let i = (r * w) + first + k in
    b.starts.{i} <- start;
    b.stops.{i} <- stop;
    b.token_places.{i} <- Int64.of_int place
  done;
  if t = 0 then b.toks.{(((r * w) + first) * c) + c - 1} <- -1.
  else begin
    let k = ref 0 in
    for j = lo to hi - 1 do
      let m = counts.{j} in
      if m > 0l then begin
        let i = (r * w) + first + !k in
        let src = j * b.cells and dst = i * c in
        for ch = 0 to b.cells - 1 do
          b.toks.{dst + ch} <- Float.of_int values.{src + ch}
        done;
        b.logc.{i} <- Float.log (Int32.to_float m);
        incr k
      end
    done
  end;
  let s = (r * shape.slots) + b.used.(r) in
  b.slot_starts.{s} <- start;
  b.slot_stops.{s} <- stop;
  b.places.{place} <- Int64.of_int s;
  b.fill.(r) <- first + n;
  b.used.(r) <- b.used.(r) + 1

let rows_of shape b =
  let w = shape.width in
  let int32 n x = Flat.wrap Nx.int32 [| b.rows; n |] x in
  {
    tokens = Flat.wrap Nx.float32 [| b.rows; w; channels b |] b.toks;
    log_count = Flat.wrap Nx.float32 [| b.rows; w |] b.logc;
    start = int32 w b.starts;
    stop = int32 w b.stops;
    slot_start = int32 shape.slots b.slot_starts;
    slot_stop = int32 shape.slots b.slot_stops;
    place = Flat.wrap Nx.int64 [| b.rows; w |] b.token_places;
    at = Flat.wrap Nx.int64 [| shape.pixels |] b.places;
  }

(* The tokens of row [p] of a sensor: its observations of nonzero count. *)
let tokens (o : (int64, int64_elt) Flat.t) (c : (int32, int32_elt) Flat.t) p =
  let t = ref 0 in
  for j = Int64.to_int o.{p} to Int64.to_int o.{p + 1} - 1 do
    if c.{j} > 0l then incr t
  done;
  !t

let check_counts name (type v s) (r : (v, s, Nx.host) Nx_ragged.t)
    (c : Nx.host Nx.int32_t) =
  if Nx.shape c <> [| Nx.dim 0 (Nx_ragged.values r) |] then
    invalid "counts.%s: one count per observation expected" name

let pack shape (px : Nx.host Pixels.t) (counts : counts) ~from =
  if
    shape.pixels <= 0 || shape.s2_rows <= 0 || shape.s1_rows <= 0
    || shape.slots <= 0 || shape.width <= 0
  then invalid "a field of the shape is not positive";
  if Nx_ragged.length px.s1 <> Nx_ragged.length px.s2 then
    invalid "%d Sentinel-2 rows and %d Sentinel-1 rows" (Nx_ragged.length px.s2)
      (Nx_ragged.length px.s1);
  check_counts "s2" px.s2 counts.s2;
  check_counts "s1" px.s1 counts.s1;
  let b2 = builder shape ~rows:shape.s2_rows ~bands:Pixels.s2_bands in
  let b1 = builder shape ~rows:shape.s1_rows ~bands:Pixels.s1_bands in
  let pixel = Flat.create int64 shape.pixels (-1L) in
  let n = Pixels.length px in
  let next =
    Flat.read (Nx_ragged.offsets px.s2) @@ fun o2 ->
    Flat.read (Nx_ragged.offsets px.s1) @@ fun o1 ->
    Flat.read (Nx_ragged.values px.s2) @@ fun v2 ->
    Flat.read (Nx_ragged.values px.s1) @@ fun v1 ->
    Flat.read counts.s2 @@ fun c2 ->
    Flat.read counts.s1 @@ fun c1 ->
    let rec go place p =
      if place >= shape.pixels || p >= n then p
      else
        let t2 = tokens o2 c2 p and t1 = tokens o1 c1 p in
        if t2 = 0 && t1 = 0 then go place (p + 1)
        else begin
          if t2 > shape.width || t1 > shape.width then
            invalid "pixel %d has %d tokens, more than the width %d" p
              (max t2 t1) shape.width;
          let r2 = row shape b2 (max t2 1) and r1 = row shape b1 (max t1 1) in
          if r2 < 0 || r1 < 0 then p
          else begin
            let span (o : (int64, int64_elt) Flat.t) =
              (Int64.to_int o.{p}, Int64.to_int o.{p + 1})
            in
            let lo, hi = span o2 in
            put shape b2 r2 ~place ~values:v2 ~counts:c2 lo hi t2;
            let lo, hi = span o1 in
            put shape b1 r1 ~place ~values:v1 ~counts:c1 lo hi t1;
            pixel.{place} <- Int64.of_int p;
            go (place + 1) (p + 1)
          end
        end
    in
    go 0 from
  in
  let s2 = rows_of shape b2 and s1 = rows_of shape b1 in
  ({ s2; s1; pixel = Flat.wrap Nx.int64 [| shape.pixels |] pixel }, next)

(* Structure *)

let walk_rows c r =
  let open Nx.Ptree.Walk in
  let tokens = field c "tokens" tensor r.tokens in
  let log_count = field c "log_count" tensor r.log_count in
  let start = field c "start" tensor r.start in
  let stop = field c "stop" tensor r.stop in
  let slot_start = field c "slot_start" tensor r.slot_start in
  let slot_stop = field c "slot_stop" tensor r.slot_stop in
  let place = field c "place" tensor r.place in
  let at = field c "at" tensor r.at in
  { tokens; log_count; start; stop; slot_start; slot_stop; place; at }

let walk c b =
  let open Nx.Ptree.Walk in
  let s2 = field c "s2" walk_rows b.s2 in
  let s1 = field c "s1" walk_rows b.s1 in
  let pixel = field c "pixel" tensor b.pixel in
  { s2; s1; pixel }

module Structure = struct
  type nonrec ('a, 'd) t = 'd t

  let walk = walk
end

let place at b = Nx.Ptree.place (module Structure) at b

(* Each batch's arrays by path, then the first's arrays replaced by the stack of
   every batch's at their path. *)
let stack = function
  | [] -> invalid_arg "Tessera.Batch.stack: no batch"
  | first :: _ as bs ->
      let s = Nx.Ptree.v walk in
      let arrays b =
        Nx.Ptree.fold s
          (fun p a acc -> (Nx.Ptree.Path.to_string p, Nx.P a) :: acc)
          b []
      in
      let all = List.map arrays bs in
      Nx.Ptree.map s
        (fun p a ->
          let name = Nx.Ptree.Path.to_string p in
          let row t =
            let x = Nx.unpack (Nx.dtype a) (List.assoc name t) in
            if Nx.shape x <> Nx.shape a then
              invalid_arg
                ("Tessera.Batch.stack: " ^ name ^ ": batches of two shapes");
            x
          in
          Nx.stack (List.map row all))
        first

(* Shapes *)

(* perf_500's typical pixel has 40 Sentinel-2 and 64 Sentinel-1 tokens. *)
let typical_s2 = 40
let typical_s1 = 64

let typical ~pixels ~slots =
  let width = max_bin in
  let rows tokens = max 1 (((pixels * tokens) + width - 1) / width) in
  { pixels; s2_rows = rows typical_s2; s1_rows = rows typical_s1; slots; width }
