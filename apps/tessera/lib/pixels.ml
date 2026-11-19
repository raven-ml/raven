(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'd t = {
  s2 : (int, Nx.int16_signed_elt, 'd) Nx_ragged.t;
  s1 : (int, Nx.int16_signed_elt, 'd) Nx_ragged.t;
}

let length px = Nx_ragged.length px.s2
let s2_bands = 10
let s1_bands = 2

(* Gathering *)

(* An array of [n] elements of kind [k], each [v]. *)
let filled k n v =
  let b = Bigarray.Array1.create k Bigarray.c_layout n in
  Bigarray.Array1.fill b v;
  b

(* Offsets of rows whose lengths are [count]. *)
let offsets count =
  let n = Bigarray.Array1.dim count in
  let o = filled Bigarray.int64 (n + 1) 0L in
  for p = 0 to n - 1 do
    let next = Int64.(add o.{p} (of_int count.{p})) in
    o.{p + 1} <- next
  done;
  o

let ragged cells count values =
  let n = Bigarray.Array1.dim count in
  let o = offsets count in
  let total = Int64.to_int o.{n} in
  let offsets = Nx.Host.wrap [| n + 1 |] o in
  Nx_ragged.v ~offsets (Nx.Host.wrap [| total; cells |] values)

(* Sentinel-2: a pass to count each pixel's masked frames, then one over the
   frames in order, each valid one appended to its pixel's row. *)
let s2 ~t ~n (bands : Nx.host Nx.int16_t) (mask : Nx.host Nx.uint8_t)
    (days : Nx.host Nx.uint16_t) =
  let count = filled Bigarray.int n 0 in
  Nx.Host.read mask (fun m ->
      for i = 0 to (t * n) - 1 do
        if m.{i} <> 0 then
          let p = i mod n in
          count.{p} <- count.{p} + 1
      done);
  let o = offsets count in
  let cells = s2_bands + 1 in
  let total = Int64.to_int o.{n} in
  let v = filled Bigarray.int16_signed (total * cells) 0 in
  let next = Array.init n (fun p -> Int64.to_int o.{p}) in
  Nx.Host.read bands (fun b ->
      Nx.Host.read mask (fun m ->
          Nx.Host.read days (fun d ->
              for f = 0 to t - 1 do
                for p = 0 to n - 1 do
                  let i = (f * n) + p in
                  if m.{i} <> 0 then begin
                    let at = next.(p) * cells in
                    for c = 0 to s2_bands - 1 do
                      v.{at + c} <- b.{(i * s2_bands) + c}
                    done;
                    v.{at + s2_bands} <- d.{f};
                    next.(p) <- next.(p) + 1
                  end
                done
              done)));
  ragged cells count v

(* Sentinel-1: the same over the ascending frames, then the descending ones. *)
let s1 ~n (orbits : (Nx.host Nx.int16_t * Nx.host Nx.int16_t * int) list) =
  let count = filled Bigarray.int n 0 in
  let valid b i =
    b.{2 * i} <> 0 || b.{(2 * i) + 1} <> 0
  in
  List.iter
    (fun (x, _, _) ->
      Nx.Host.read x (fun b ->
          for i = 0 to (Nx.dim 0 x * n) - 1 do
            if valid b i then
              let p = i mod n in
              count.{p} <- count.{p} + 1
          done))
    orbits;
  let o = offsets count in
  let cells = s1_bands + 2 in
  let total = Int64.to_int o.{n} in
  let v = filled Bigarray.int16_signed (total * cells) 0 in
  let next = Array.init n (fun p -> Int64.to_int o.{p}) in
  List.iter
    (fun (x, days, orbit) ->
      Nx.Host.read x (fun b ->
          Nx.Host.read days (fun d ->
              for f = 0 to Nx.dim 0 x - 1 do
                for p = 0 to n - 1 do
                  let i = (f * n) + p in
                  if valid b i then begin
                    let at = next.(p) * cells in
                    v.{at} <- b.{2 * i};
                    v.{at + 1} <- b.{(2 * i) + 1};
                    v.{at + 2} <- d.{f};
                    v.{at + 3} <- orbit;
                    next.(p) <- next.(p) + 1
                  end
                done
              done)))
    orbits;
  ragged cells count v

let of_tile (tile : Tile.t) =
  let t = Nx.dim 0 tile.bands and h, w = Tile.shape tile in
  let n = h * w in
  let orbits =
    List.filter
      (fun (x, _, _) -> Nx.dim 0 x > 0)
      [
        (tile.ascending, tile.ascending_doys, 0);
        (tile.descending, tile.descending_doys, 1);
      ]
  in
  ({ s2 = s2 ~t ~n tile.bands tile.masks tile.doys; s1 = s1 ~n orbits }, (h, w))

(* Corpora *)

let sub px ~pos ~len =
  {
    s2 = Nx_ragged.sub px.s2 ~offset:pos ~length:len;
    s1 = Nx_ragged.sub px.s1 ~offset:pos ~length:len;
  }

let take ~indices px =
  { s2 = Nx_ragged.take ~indices px.s2; s1 = Nx_ragged.take ~indices px.s1 }

let file dir sensor = Filename.concat dir (sensor ^ ".npy")
let offsets_file dir sensor = Filename.concat dir (sensor ^ "_offsets.npy")

(* A sensor's rows from offset [0]: its values from the first offset to the
   last, and the offsets less the first. *)
let rebase (type s) (r : (int, s, Nx.host) Nx_ragged.t) =
  let o = Nx_ragged.offsets r in
  let n = Nx_ragged.length r in
  let lo = Nx.item [ 0 ] o and hi = Nx.item [ n ] o in
  ( Nx.slice [ Nx.R (Int64.to_int lo, Int64.to_int hi) ] (Nx_ragged.values r),
    Nx.sub_s o lo )

let save dir px =
  if not (Sys.file_exists dir) then Sys.mkdir dir 0o755;
  let one name r =
    let v, o = rebase r in
    Nx_io.save_npy (file dir name) v;
    Nx_io.save_npy (offsets_file dir name) o
  in
  one "s2" px.s2;
  one "s1" px.s1

let load dir =
  let get name dt cells =
    let fail file m = failwith ("Tessera.Pixels.load: " ^ file ^ ": " ^ m) in
    let read file dt =
      let x =
        try Nx_io.load_npy file
        with Failure m -> failwith ("Tessera.Pixels.load: " ^ m)
      in
      match Nx.unpack dt x with
      | x -> x
      | exception Invalid_argument m -> fail file m
    in
    let v = read (file dir name) dt in
    let o = read (offsets_file dir name) Nx.int64 in
    if Nx.ndim v <> 2 || Nx.dim 1 v <> cells then
      fail (file dir name) (Printf.sprintf "cells of %d values expected" cells);
    Nx_ragged.v ~offsets:o v
  in
  let s2 = get "s2" Nx.int16 (s2_bands + 1) in
  let s1 = get "s1" Nx.int16 (s1_bands + 2) in
  if Nx_ragged.length s2 <> Nx_ragged.length s1 then
    failwith ("Tessera.Pixels.load: " ^ dir ^ ": sensors of different lengths");
  { s2; s1 }

(* Rows [row] of the sensor [name] of the corpus [dir], of [cells] values a
   cell: each row's two offsets as a window of the offsets file, then its
   observations as a window of the values file. A window outside the offsets
   is an index outside the corpus; offsets that decrease, or reach past the
   values, are the offsets file's own fault. *)
let read_rows (type s) dir name (dt : (int, s) Nx.dtype) cells row =
  let n = Array.length row in
  let offsets = offsets_file dir name and values = file dir name in
  let corrupt file m = failwith ("Tessera.Pixels.read: " ^ file ^ ": " ^ m) in
  let o =
    match Nx_io.load_npy_rows offsets ~row ~rows:(Array.make n 2) with
    | o -> o
    | exception Invalid_argument _ ->
        invalid_arg "Tessera.Pixels.read: an index outside the corpus"
  in
  let o =
    match Nx.unpack Nx.int64 o with
    | o -> Nx.to_array o
    | exception Invalid_argument m -> corrupt offsets m
  in
  let first = Array.init n (fun j -> Int64.to_int o.(2 * j)) in
  let lengths =
    Array.init n (fun j -> Int64.to_int o.((2 * j) + 1) - first.(j))
  in
  if Array.exists (fun l -> l < 0) lengths then
    corrupt offsets "offsets that decrease";
  let v =
    match Nx_io.load_npy_rows values ~row:first ~rows:lengths with
    | v -> v
    | exception Invalid_argument _ ->
        corrupt offsets "offsets outside the values' rows"
  in
  let v =
    match Nx.unpack dt v with
    | v -> v
    | exception Invalid_argument m -> corrupt values m
  in
  if Nx.ndim v <> 2 || Nx.dim 1 v <> cells then
    corrupt values (Printf.sprintf "cells of %d values expected" cells);
  Nx_ragged.of_lengths
    (Nx.create Nx.int64 [| n |] (Array.map Int64.of_int lengths))
    v

let read dir indices =
  if Nx.ndim indices <> 1 then
    invalid_arg "Tessera.Pixels.read: indices of one axis expected";
  let row = Array.map Int64.to_int (Nx.to_array indices) in
  {
    s2 = read_rows dir "s2" Nx.int16 (s2_bands + 1) row;
    s1 = read_rows dir "s1" Nx.int16 (s1_bands + 2) row;
  }

let stream ~key ~n ~first len =
  if n <= 0 then invalid_arg "Tessera.Pixels.stream: a corpus of no pixel";
  if first < 0 || len < 0 then
    invalid_arg "Tessera.Pixels.stream: a negative position or length";
  let pos = Nx.arange Nx.int64 first (first + len) 1 in
  let n' = Int64.of_int n in
  let e = Nx.div_s pos n' and i = Nx.mod_s pos n' in
  let shuffled = Nx.Rng.permute ~key:(Nx.Rng.fold_in_tensor key e) ~n i in
  Nx.where (Nx.equal_s e 0L) i shuffled

let window dir corpus ~first indices =
  if first + Nx.numel indices <= length corpus then take ~indices corpus
  else read dir indices

(* Shuffling *)

(* One sensor of a [t] and its values' dtype. *)
type 's t_sensor = {
  get : Nx.host t -> (int, 's, Nx.host) Nx_ragged.t;
  dtype : (int, 's) Nx.dtype;
}

(* Bytes of a Sentinel-2 and a Sentinel-1 observation. *)
let s2_bytes = 2 * (s2_bands + 1)
let s1_bytes = 2 * (s1_bands + 2)

(* A host copy of [x]'s elements. *)
let copy x =
  Nx.Host.read x (fun b ->
      let c =
        Bigarray.Array1.create (Bigarray.Array1.kind b) Bigarray.c_layout
          (Bigarray.Array1.dim b)
      in
      Bigarray.Array1.blit b c;
      c)

let lengths r = copy (Nx_ragged.lengths r)

let shuffle ~key ~bytes dir parts =
  if bytes <= 0 then invalid_arg "Tessera.Pixels.shuffle: bytes is not positive";
  let parts = Array.of_list (List.map load parts) in
  let l2 = Array.map (fun p -> lengths p.s2) parts in
  let l1 = Array.map (fun p -> lengths p.s1) parts in
  let n = Array.fold_left (fun a p -> a + length p) 0 parts in
  (* Pixel [j] of the concatenation goes to position [dest j]; position [q]
     holds the part's pixel [src_part.(q)], [src_row.(q)]. *)
  let dest =
    if n = 0 then Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout 0
    else copy (Nx.Rng.permute ~key ~n (Nx.arange Nx.int64 0 n 1))
  in
  let src_part = Array.make n 0 and src_row = Array.make n 0 in
  let size = Array.make n 0 in
  let j = ref 0 in
  Array.iteri
    (fun k p ->
      for i = 0 to length p - 1 do
        let q = Int64.to_int dest.{!j} in
        src_part.(q) <- k;
        src_row.(q) <- i;
        size.(q) <-
          (Int64.to_int l2.(k).{i} * s2_bytes)
          + (Int64.to_int l1.(k).{i} * s1_bytes);
        incr j
      done)
    parts;
  (* Buckets: runs of positions of at most [bytes], or one pixel. *)
  let buckets =
    let rec go acc start pos total =
      if pos = n then
        List.rev (if pos > start then (start, pos) :: acc else acc)
      else if pos > start && total + size.(pos) > bytes then
        go ((start, pos) :: acc) pos pos 0
      else go acc start (pos + 1) (total + size.(pos))
    in
    go [] 0 0 0
  in
  (* A bucket's rows of one sensor in position order: each part's rows taken in
     that order, then the parts' rows interleaved. *)
  let block (type s) (sensor : s t_sensor) (lo, hi) : (int, s, Nx.host) Nx.t =
    let picks = Array.map (fun _ -> ref []) parts in
    let order = Array.make (hi - lo) 0 in
    for q = hi - 1 downto lo do
      picks.(src_part.(q)) := src_row.(q) :: !(picks.(src_part.(q)))
    done;
    let base = Array.make (Array.length parts) 0 in
    let total = ref 0 in
    Array.iteri
      (fun k l ->
        base.(k) <- !total;
        total := !total + List.length !l)
      picks;
    let seen = Array.make (Array.length parts) 0 in
    for q = lo to hi - 1 do
      let k = src_part.(q) in
      order.(q - lo) <- base.(k) + seen.(k);
      seen.(k) <- seen.(k) + 1
    done;
    let taken =
      Array.to_list parts
      |> List.mapi (fun k p ->
          let indices =
            Nx.create Nx.int64
              [| List.length !(picks.(k)) |]
              (Array.of_list (List.map Int64.of_int !(picks.(k))))
          in
          Nx_ragged.take ~indices (sensor.get p))
    in
    let all = Nx_ragged.concat taken in
    let indices =
      Nx.create Nx.int64 [| hi - lo |] (Array.map Int64.of_int order)
    in
    Nx_ragged.values (Nx_ragged.take ~indices all)
  in
  if not (Sys.file_exists dir) then Sys.mkdir dir 0o755;
  let write (type s) name (sensor : s t_sensor) cells l =
    let o = filled Bigarray.int64 (n + 1) 0L in
    for q = 0 to n - 1 do
      o.{q + 1} <- Int64.add o.{q} l.(src_part.(q)).{src_row.(q)}
    done;
    Nx_io.save_npy (offsets_file dir name) (Nx.Host.wrap [| n + 1 |] o);
    Nx_io.save_npy_concat (file dir name) sensor.dtype [| cells |]
      (Seq.map (block sensor) (List.to_seq buckets))
  in
  write "s2" { get = (fun p -> p.s2); dtype = Nx.int16 } (s2_bands + 1) l2;
  write "s1" { get = (fun p -> p.s1); dtype = Nx.int16 } (s1_bands + 2) l1;
  n

(* Draws *)

(* [length] observations of each row of [r], each uniform over the row from the
   row's key in [keys] ([[n; 2]]). *)
let draw_rows (type s) keys ~length (r : (int, s, Nx.host) Nx_ragged.t) =
  let n = Nx_ragged.length r in
  let counts = Nx.reshape [| n; 1 |] (Nx_ragged.lengths r) in
  let keys = Nx.Rng.reshape [| n; 1 |] keys in
  let index =
    Nx.Rng.randint ~key:keys (Nx.broadcast_to [| n; length |] counts)
  in
  let first =
    Nx.reshape [| n; 1 |] (Nx.slice [ Nx.R (0, n) ] (Nx_ragged.offsets r))
  in
  let at = Nx.reshape [| n * length |] (Nx.add first index) in
  Nx_ragged.of_lengths
    (Nx.full Nx.int64 [| n |] (Int64.of_int length))
    (Nx.take ~axis:0 at (Nx_ragged.values r))

let draw ~keys ~length:l px =
  let n = length px in
  if l < 1 then invalid_arg "Tessera.Pixels.draw: length is not positive";
  if Nx.shape (Nx.Rng.to_tensor keys) <> [| n; 2 |] then
    invalid_arg
      (Printf.sprintf "Tessera.Pixels.draw: keys of shape [%d; 2] expected" n);
  let empty r = n > 0 && Nx.item [] (Nx.min (Nx_ragged.lengths r)) = 0L in
  if empty px.s2 || empty px.s1 then
    invalid_arg "Tessera.Pixels.draw: a pixel has no observation of a sensor";
  {
    s2 = draw_rows (Nx.Rng.fold_in keys 0) ~length:l px.s2;
    s1 = draw_rows (Nx.Rng.fold_in keys 1) ~length:l px.s1;
  }
