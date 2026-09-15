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
