(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type rule = [ `Nonzero | `Evenodd ]

type t =
  | Empty
  | Fill of { rule : rule; color : Color.t; path : Path.t }
  | Stroke of { stroke : Stroke.t; color : Color.t; path : Path.t }
  | Text of {
      font : Font.t;
      size : float;
      color : Color.t;
      x : float;
      y : float;
      text : string;
    }
  | Image of { x : float; y : float; w : float; h : float; data : Nx.uint8_t }
  | Group of t list
  | Clip of { path : Path.t; picture : t }
  | Transform of { m : Affine.t; picture : t }
  | Stamp of { picture : t; xs : float array; ys : float array }

let empty = Empty

let fill ?(rule = `Nonzero) color path =
  if Path.is_empty path then Empty else Fill { rule; color; path }

let stroke stroke color path =
  if Path.is_empty path then Empty else Stroke { stroke; color; path }

let text font ~size color ~x ~y text =
  if text = "" then Empty else Text { font; size; color; x; y; text }

let image ~x ~y ~w ~h data =
  (match Nx.shape data with
  | [| _; _ |] | [| _; _; 1 | 3 | 4 |] -> ()
  | _ ->
      invalid_arg
        "Picture.image: expected shape [|rows; cols|] or [|rows; cols; c|] \
         with c in 1, 3, 4");
  Image { x; y; w; h; data }

let group = function [] -> Empty | [ p ] -> p | ps -> Group ps

let clip path picture =
  match picture with Empty -> Empty | _ -> Clip { path; picture }

let transform m picture =
  match picture with Empty -> Empty | _ -> Transform { m; picture }

let stamp picture xs ys =
  if Array.length xs <> Array.length ys then
    invalid_arg "Picture.stamp: xs and ys differ in length";
  match picture with
  | Empty -> Empty
  | _ -> if Array.length xs = 0 then Empty else Stamp { picture; xs; ys }

let union a b =
  match (a, b) with
  | None, x | x, None -> x
  | Some a, Some b -> Some (Box.union a b)

let linear_scale (m : Affine.t) =
  Float.sqrt (Float.abs ((m.xx *. m.yy) -. (m.xy *. m.yx)))

let rec bounds_under m = function
  | Empty -> None
  | Fill { path; _ } -> Path.bounds (Path.transform m path)
  | Stroke { stroke; path; _ } -> (
      match Path.bounds (Path.transform m path) with
      | None -> None
      | Some b ->
          let pen =
            stroke.width /. 2. *. linear_scale m
            *. if stroke.join = `Miter then stroke.miter_limit else 1.
          in
          Some (Box.v (b.x0 -. pen) (b.y0 -. pen) (b.x1 +. pen) (b.y1 +. pen)))
  | Text { font; size; x; y; text; _ } ->
      Option.map
        (fun b -> Box.transform Affine.(m * translate x y) b)
        (Font.bounds font ~size text)
  | Image { x; y; w; h; _ } ->
      Some (Box.transform m (Box.v x y (x +. w) (y +. h)))
  | Group ps ->
      List.fold_left (fun acc p -> union acc (bounds_under m p)) None ps
  | Clip { picture; _ } -> bounds_under m picture
  | Transform { m = m'; picture } -> bounds_under Affine.(m * m') picture
  | Stamp { picture; xs; ys } -> (
      match bounds_under { m with x0 = 0.; y0 = 0. } picture with
      | None -> None
      | Some b ->
          let acc = ref None in
          Array.iteri
            (fun i x ->
              let dx, dy = Affine.apply m x ys.(i) in
              if Float.is_finite dx && Float.is_finite dy then
                acc :=
                  union !acc
                    (Some
                       (Box.v (b.x0 +. dx) (b.y0 +. dy) (b.x1 +. dx) (b.y1 +. dy))))
            xs;
          !acc)

let bounds p = bounds_under Affine.id p
