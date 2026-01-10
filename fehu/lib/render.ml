(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Pixel = struct
  type format = [ `RGB8 | `RGBA8 | `GRAY8 | `RGBf | `RGBAf ]
end

type image = {
  width : int;
  height : int;
  pixel_format : Pixel.format;
  data_u8 :
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
    option;
  data_f32 :
    (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t option;
}

type text = string
type svg = string
type frame = Image of image | Text of text | Svg of svg | None
type t = frame

let image_u8 ~width ~height ~pixel_format ~data () =
  { width; height; pixel_format; data_u8 = Some data; data_f32 = None }

let image_f32 ~width ~height ~pixel_format ~data () =
  { width; height; pixel_format; data_u8 = None; data_f32 = Some data }
