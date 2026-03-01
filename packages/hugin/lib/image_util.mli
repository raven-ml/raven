(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Image encoding utilities.

    {b Internal module.} Shared base64 encoding and Nx-to-Cairo-surface
    conversion used by both rendering backends. *)

(** {1:base64 Base64} *)

val base64_encode : string -> string
(** [base64_encode s] is the base64 encoding of [s]. *)

(** {1:surface Cairo surface conversion} *)

val nx_to_cairo_surface : Nx.uint8_t -> Ucairo.surface
(** [nx_to_cairo_surface data] is a Cairo ARGB32 image surface from [data].
    [data] has shape [[|h; w; 3|]] (RGB) or [[|h; w; 4|]] (RGBA). *)

val nx_to_png_base64 : Nx.uint8_t -> string
(** [nx_to_png_base64 data] is the base64-encoded PNG of [data]. *)
