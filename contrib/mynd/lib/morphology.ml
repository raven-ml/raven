(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Image

let check_footprint fn fp =
  if Nx.ndim fp <> 2 then
    invalid_argf "%s: a footprint has axes [kh; kw], not %d" fn (Nx.ndim fp)

let ordered fn x =
  if Nx.Dtype.is Nx.Dtype.Complex (Nx.dtype x) then
    invalid_argf "%s: complex values are no image's" fn

(* [extreme ~least fn fp x] folds [Nx.minimum] ([least]) or [Nx.maximum] over
   the selected taps, each a view of [x] padded by the fold's identity, which a
   tap outside the image reads, so that it is left out. Erosion's tap [(p, q)]
   reads offset [(p - kh / 2, q - kw / 2)], dilation's the reflected one; both
   stay within a pad of [kh / 2] rows and [kw / 2] columns each side. Every
   result is one of the values the fold compares, so the order of the taps
   changes no bits but a NaN's.

   On floats a custom derivative shares the tangent equally among the taps
   inside equal to the result, as [Nx.min]'s does, where a fold of [Nx.minimum]
   would halve it at each tie and weigh the taps by their order. No tap equals a
   NaN result, and a pixel with no tap inside has none, so the count guard gives
   them a zero tangent. *)
let extreme (type v r d) ~least fn fp (x : (v, r, d) Nx.t) : (v, r, d) Nx.t =
  check_footprint fn fp;
  image fn x;
  ordered fn x;
  let dt = Nx.dtype x in
  let id = if least then Nx.Dtype.max_value dt else Nx.Dtype.min_value dt in
  let kh = Nx.dim 0 fp and kw = Nx.dim 1 fp in
  let h = Nx.dim (-3) x and w = Nx.dim (-2) x in
  let selected = Nx.to_array fp in
  let offsets =
    List.concat
      (List.init kh (fun p ->
           List.filter_map
             (fun q ->
               if not selected.((p * kw) + q) then None
               else if least then Some (p - (kh / 2), q - (kw / 2))
               else Some ((kh / 2) - p, (kw / 2) - q))
             (List.init kw Fun.id)))
  in
  let r = Nx.ndim x in
  let pads =
    Array.init r (fun k ->
        if k = r - 3 then (kh / 2, kh / 2)
        else if k = r - 2 then (kw / 2, kw / 2)
        else (0, 0))
  in
  let read p (di, dj) =
    cut ~axis:(-2) ((kw / 2) + dj) w (cut ~axis:(-3) ((kh / 2) + di) h p)
  in
  let pick = if least then Nx.minimum else Nx.maximum in
  let fold x =
    match offsets with
    | [] -> Nx.full_like x id
    | o :: os ->
        let p = Nx.pad pads id x in
        List.fold_left (fun acc o -> pick acc (read p o)) (read p o) os
  in
  match Nx.Dtype.kind dt with
  | Nx.Dtype.Float ->
      Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor
        (fun x ->
          let y = fold x in
          (* [shares acc dx] sums and counts at [acc], float32 for the narrow
             formats, which stop counting at 16 (float8_e4m3fn), 256 (bfloat16)
             or 2048 (float16), and rounds the share once. *)
          let shares (type a) (acc : (float, a) Nx.dtype) dx =
            let dx = Nx.cast acc dx in
            let p = Nx.pad pads id x and dp = Nx.pad pads 0. dx in
            let inside =
              Nx.pad pads false (Nx.logical_not (Nx.zeros_like (Nx.isnan x)))
            in
            let zero = Nx.zeros_like dx in
            let sum, count =
              List.fold_left
                (fun (sum, count) o ->
                  let tie =
                    Nx.logical_and (read inside o) (Nx.equal (read p o) y)
                  in
                  ( Nx.add sum (Nx.where tie (read dp o) zero),
                    Nx.add count (Nx.cast acc tie) ))
                (zero, zero) offsets
            in
            Nx.cast dt
              (Nx.where (Nx.greater_s count 0.)
                 (Nx.div sum (Nx.maximum_s count 1.))
                 zero)
          in
          let tangent dx =
            if narrow x then shares Nx.float32 dx else shares dt dx
          in
          (y, tangent))
        x
  | _ -> fold x

let erode fp x = extreme ~least:true "Mynd.Filter.erode" fp x
let dilate fp x = extreme ~least:false "Mynd.Filter.dilate" fp x
