(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Scene IR — resolved primitives in device pixels *)

type primitive =
  | Path of {
      points : (float * float) array;
      close : bool;
      fill : Color.t option;
      stroke : Color.t option;
      line_width : float;
      dash : float list;
    }
  | Markers of {
      points : (float * float) array;
      shape : Spec.marker;
      size : float;
      sizes : float array option;
      fill : Color.t option;
      fills : Color.t array option;
      stroke : Color.t option;
    }
  | Text of {
      x : float;
      y : float;
      content : string;
      font : Theme.font;
      color : Color.t;
      anchor : [ `Start | `Middle | `End ];
      baseline : [ `Top | `Middle | `Bottom ];
      angle : float;
    }
  | Image of { x : float; y : float; w : float; h : float; data : Nx.uint8_t }
  | Clip of {
      x : float;
      y : float;
      w : float;
      h : float;
      children : primitive list;
    }
  | Group of primitive list

type t = { width : float; height : float; primitives : primitive list }

let rec fold_primitive f acc = function
  | Group children | Clip { children; _ } ->
      List.fold_left (fold_primitive f) acc children
  | p -> f acc p

let fold f scene acc = List.fold_left (fold_primitive f) acc scene.primitives
