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
