(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scene intermediate representation.

    {b Internal module.} Resolved drawing primitives in device-pixel
    coordinates. All data-space concepts are gone; backends fold over these
    primitives to produce output. *)

(** {1:types Types} *)

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
  | Group of primitive list  (** The type for drawing primitives. *)

type t = { width : float; height : float; primitives : primitive list }
(** The type for resolved scenes. *)

(** {1:traversal Traversal} *)

val fold : ('a -> primitive -> 'a) -> t -> 'a -> 'a
(** [fold f scene acc] folds [f] over every leaf primitive in [scene],
    descending into {!Clip} and {!Group} nodes. *)
