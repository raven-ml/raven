(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

include Values

type extension = Image.extension

module Colour = Colour

module Sample = struct
  include Sample
  include Resize
end

module Transform = Transform

module Filter = struct
  include Filter
  include Gradient
  include Morphology
end

module Box = Box
module Mask = Mask
module Rle = Rle
