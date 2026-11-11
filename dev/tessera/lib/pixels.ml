(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'd t = {
  s2 : (int, Nx.int16_unsigned_elt, 'd) Nx_ragged.t;
  s1 : (int, Nx.int16_signed_elt, 'd) Nx_ragged.t;
}

let length px = Nx_ragged.length px.s2
let s2_bands = 10
let s1_bands = 2

let invalid file fmt =
  Printf.ksprintf
    (fun m -> invalid_arg ("Tessera.Pixels.of_tile: " ^ file ^ ": " ^ m))
    fmt

(* Files *)

let load dir name dt =
  let file = Filename.concat dir name in
  if not (Sys.file_exists file) then None
  else
    match Nx.unpack dt (Nx_io.load_npy file) with
    | x -> Some (file, x)
    | exception Invalid_argument m -> invalid file "%s" m

let check (file, x) shape =
  if Nx.shape x <> shape then
    invalid file "shape [%s], [%s] expected"
      (String.concat "; "
         (Array.to_list (Array.map string_of_int (Nx.shape x))))
      (String.concat "; " (Array.to_list (Array.map string_of_int shape)));
  x

(* An orbit's frames and their days, [None] where the tile has none. *)
let orbit dir ~h ~w name =
  match load dir (name ^ ".npy") Nx.int16 with
  | None -> None
  | Some (file, x) ->
      let t = Nx.dim 0 x in
      let x = check (file, x) [| t; h; w; s1_bands |] in
      let days =
        match load dir (name ^ "_doy.npy") Nx.int16 with
        | None -> invalid file "no %s_doy.npy" name
        | Some f -> check f [| t |]
      in
      if t = 0 then None else Some (x, days)

(* Gathering *)

(* Offsets of rows whose lengths are [count]. *)
let offsets count =
  let n = Bigarray.Array1.dim count in
  let o = Flat.create Bigarray.int64 (n + 1) 0L in
  for p = 0 to n - 1 do
    let next = Int64.(add o.{p} (of_int count.{p})) in
    o.{p + 1} <- next
  done;
  o

let ragged dt cells count values =
  let n = Bigarray.Array1.dim count in
  let o = offsets count in
  let total = Int64.to_int o.{n} in
  let offsets = Flat.wrap Nx.int64 [| n + 1 |] o in
  Nx_ragged.v ~offsets (Flat.wrap dt [| total; cells |] values)

(* Sentinel-2: a pass to count each pixel's masked frames, then one over the
   frames in order, each valid one appended to its pixel's row. *)
let s2 ~t ~n (bands : (int, Nx.int16_unsigned_elt, Nx.host) Nx.t)
    (mask : Nx.host Nx.uint8_t) (days : Nx.host Nx.uint16_t) =
  let count = Flat.create Bigarray.int n 0 in
  Flat.read mask (fun m ->
      for i = 0 to (t * n) - 1 do
        if m.{i} <> 0 then
          let p = i mod n in
          count.{p} <- count.{p} + 1
      done);
  let o = offsets count in
  let cells = s2_bands + 1 in
  let total = Int64.to_int o.{n} in
  let v = Flat.create Bigarray.int16_unsigned (total * cells) 0 in
  let next = Array.init n (fun p -> Int64.to_int o.{p}) in
  Flat.read bands (fun b ->
      Flat.read mask (fun m ->
          Flat.read days (fun d ->
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
  ragged Nx.uint16 cells count v

(* Sentinel-1: the same over the ascending frames, then the descending ones. *)
let s1 ~n (orbits : (Nx.host Nx.int16_t * Nx.host Nx.int16_t * int) list) =
  let count = Flat.create Bigarray.int n 0 in
  let valid (b : (int, Nx.int16_signed_elt) Flat.t) i =
    b.{2 * i} <> 0 || b.{(2 * i) + 1} <> 0
  in
  List.iter
    (fun (x, _, _) ->
      Flat.read x (fun b ->
          for i = 0 to (Nx.dim 0 x * n) - 1 do
            if valid b i then
              let p = i mod n in
              count.{p} <- count.{p} + 1
          done))
    orbits;
  let o = offsets count in
  let cells = s1_bands + 2 in
  let total = Int64.to_int o.{n} in
  let v = Flat.create Bigarray.int16_signed (total * cells) 0 in
  let next = Array.init n (fun p -> Int64.to_int o.{p}) in
  List.iter
    (fun (x, days, orbit) ->
      Flat.read x (fun b ->
          Flat.read days (fun d ->
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
  ragged Nx.int16 cells count v

let of_tile dir =
  let bands =
    match load dir "bands.npy" Nx.uint16 with
    | None -> invalid (Filename.concat dir "bands.npy") "missing"
    | Some (file, x) ->
        if Nx.ndim x <> 4 || Nx.dim 3 x <> s2_bands then
          invalid file "shape [t; h; w; %d] expected" s2_bands;
        x
  in
  let t = Nx.dim 0 bands and h = Nx.dim 1 bands and w = Nx.dim 2 bands in
  let n = h * w in
  let mask =
    match load dir "masks.npy" Nx.uint8 with
    | None -> Nx.zeros Nx.uint8 [| t; h; w |]
    | Some f -> check f [| t; h; w |]
  in
  let days =
    match load dir "doys.npy" Nx.uint16 with
    | None -> Nx.zeros Nx.uint16 [| t |]
    | Some f -> check f [| t |]
  in
  let orbits =
    List.filter_map
      (fun (name, o) ->
        Option.map (fun (x, d) -> (x, d, o)) (orbit dir ~h ~w name))
      [ ("sar_ascending", 0); ("sar_descending", 1) ]
  in
  ({ s2 = s2 ~t ~n bands mask days; s1 = s1 ~n orbits }, (h, w))

(* Corpora *)

let sub px ~pos ~len =
  {
    s2 = Nx_ragged.sub px.s2 ~offset:pos ~length:len;
    s1 = Nx_ragged.sub px.s1 ~offset:pos ~length:len;
  }

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
    let fail file m = invalid_arg ("Tessera.Pixels.load: " ^ file ^ ": " ^ m) in
    let read file dt =
      match Nx.unpack dt (Nx_io.load_npy file) with
      | x -> x
      | exception (Invalid_argument m | Failure m) -> fail file m
    in
    let v = read (file dir name) dt in
    let o = read (offsets_file dir name) Nx.int64 in
    if Nx.ndim v <> 2 || Nx.dim 1 v <> cells then
      fail (file dir name) (Printf.sprintf "cells of %d values expected" cells);
    Nx_ragged.v ~offsets:o v
  in
  let s2 = get "s2" Nx.uint16 (s2_bands + 1) in
  let s1 = get "s1" Nx.int16 (s1_bands + 2) in
  if Nx_ragged.length s2 <> Nx_ragged.length s1 then
    invalid_arg
      ("Tessera.Pixels.load: " ^ dir ^ ": sensors of different lengths");
  { s2; s1 }

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
  Flat.read x (fun b ->
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
  let write (type s) name (sensor : s t_sensor) cells
      (l : (int64, Bigarray.int64_elt) Flat.t array) =
    let o = Flat.create Bigarray.int64 (n + 1) 0L in
    for q = 0 to n - 1 do
      o.{q + 1} <- Int64.add o.{q} l.(src_part.(q)).{src_row.(q)}
    done;
    Nx_io.save_npy (offsets_file dir name) (Flat.wrap Nx.int64 [| n + 1 |] o);
    Nx_io.save_npy_concat (file dir name) sensor.dtype [| cells |]
      (Seq.map (block sensor) (List.to_seq buckets))
  in
  write "s2" { get = (fun p -> p.s2); dtype = Nx.uint16 } (s2_bands + 1) l2;
  write "s1" { get = (fun p -> p.s1); dtype = Nx.int16 } (s1_bands + 2) l1;
  n

(* Draws *)

(* [length] observations of each row of [r], each uniform over the row from the
   row's key in [keys] ([[n; 2]]). *)
let draw_rows (type s) keys ~length (r : (int, s, Nx.host) Nx_ragged.t) =
  let n = Nx_ragged.length r in
  let counts = Nx.reshape [| n; 1 |] (Nx_ragged.lengths r) in
  let keys =
    Nx.Rng.of_tensor (Nx.reshape [| n; 1; 2 |] (Nx.Rng.to_tensor keys))
  in
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
