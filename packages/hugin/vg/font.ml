(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  tt : Truetype.t;
  outlines : (int, Path.t) Hashtbl.t;
      (** Glyph outlines in font units, y up, built on first use. *)
}

type error = Unsupported of string | Malformed of string

let pp_error fmt = function
  | Unsupported s -> Format.fprintf fmt "unsupported font: %s" s
  | Malformed s -> Format.fprintf fmt "malformed font: %s" s

let of_string s =
  match Truetype.of_string s with
  | tt -> Ok { tt; outlines = Hashtbl.create 256 }
  | exception Truetype.Unsupported m -> Error (Unsupported m)
  | exception Truetype.Malformed m -> Error (Malformed m)

let bundled s =
  match of_string s with
  | Ok f -> f
  | Error e -> Format.kasprintf failwith "bundled font: %a" pp_error e

let regular = bundled Font_data.inter_regular
let bold = bundled Font_data.inter_bold
let family f = f.tt.family
let weight f = f.tt.weight
let bytes f = f.tt.data
let scale f ~size = size /. float f.tt.units_per_em
let ascent f ~size = float f.tt.ascender *. scale f ~size
let descent f ~size = -.float f.tt.descender *. scale f ~size

(* Layout in font units: [fold_glyphs f s init k] folds [k acc gid pen_x] over
   the glyphs of [s], with kerning applied to the pen. *)
let fold_glyphs f s init k =
  let tt = f.tt in
  let n = String.length s in
  let rec go i prev pen acc =
    if i >= n then (acc, pen)
    else begin
      let d = String.get_utf_8_uchar s i in
      let g =
        Truetype.glyph_of_uchar tt (Uchar.to_int (Uchar.utf_decode_uchar d))
      in
      let pen = if prev < 0 then pen else pen + Truetype.kerning tt prev g in
      let acc = k acc g pen in
      go (i + Uchar.utf_decode_length d) g (pen + Truetype.advance tt g) acc
    end
  in
  go 0 (-1) 0 init

let advance f ~size s =
  let _, pen = fold_glyphs f s () (fun () _ _ -> ()) in
  float pen *. scale f ~size

let bounds f ~size s =
  let box, _ =
    fold_glyphs f s None (fun acc g pen ->
        match Truetype.bounds f.tt g with
        | None -> acc
        | Some (x0, y0, x1, y1) -> (
            let x0 = pen + x0 and x1 = pen + x1 in
            match acc with
            | None -> Some (x0, y0, x1, y1)
            | Some (a, b, c, d) -> Some (min a x0, min b y0, max c x1, max d y1)
            ))
  in
  match box with
  | None -> None
  | Some (x0, y0, x1, y1) ->
      let k = scale f ~size in
      Some
        (Box.v
           (float x0 *. k)
           (-.float y1 *. k)
           (float x1 *. k)
           (-.float y0 *. k))

type glyph = { id : int; x : float; advance : float }

let glyphs f ~size s =
  let k = scale f ~size in
  let acc, _ =
    fold_glyphs f s [] (fun acc g pen ->
        {
          id = g;
          x = float pen *. k;
          advance = float (Truetype.advance f.tt g) *. k;
        }
        :: acc)
  in
  List.rev acc

(* Quadratic control points convert exactly to cubic ones at two thirds of the
   way from each end point. *)
let outline_units f g =
  match Hashtbl.find_opt f.outlines g with
  | Some p -> p
  | None ->
      let p = ref Path.empty in
      let cx = ref 0. and cy = ref 0. in
      Truetype.outline f.tt g
        ~move:(fun x y ->
          p := Path.move_to x y !p;
          cx := x;
          cy := y)
        ~line:(fun x y ->
          p := Path.line_to x y !p;
          cx := x;
          cy := y)
        ~quad:(fun qx qy x y ->
          let c1x = !cx +. (2. /. 3. *. (qx -. !cx))
          and c1y = !cy +. (2. /. 3. *. (qy -. !cy)) in
          let c2x = x +. (2. /. 3. *. (qx -. x))
          and c2y = y +. (2. /. 3. *. (qy -. y)) in
          p := Path.curve_to c1x c1y c2x c2y x y !p;
          cx := x;
          cy := y)
        ~close:(fun () -> p := Path.close !p);
      Hashtbl.replace f.outlines g !p;
      !p

let glyph_advance f ~size g = float (Truetype.advance f.tt g) *. scale f ~size

let glyph_path f ~size g =
  let k = scale f ~size in
  Path.transform (Affine.scale k (-.k)) (outline_units f g)
