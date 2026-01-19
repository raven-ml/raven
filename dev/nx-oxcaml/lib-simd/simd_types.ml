(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

module Blend2 : sig
  type t = private int

  val make : int -> int -> t
end = struct
  type t = int

  let make a b = a lor (b lsl 1)
end

module Blend4 : sig
  type t = private int

  val make : int -> int -> int -> int -> t
end = struct
  type t = int

  let make a b c d = a lor (b lsl 1) lor (c lsl 2) lor (d lsl 3)
end

module Blend8 : sig
  type t = private int

  val make : int -> int -> int -> int -> int -> int -> int -> int -> t
end = struct
  type t = int

  let make a b c d e f g h =
    a
    lor (b lsl 1)
    lor (c lsl 2)
    lor (d lsl 3)
    lor (e lsl 4)
    lor (f lsl 5)
    lor (g lsl 6)
    lor (h lsl 7)
end

module Shuffle2 : sig
  type t = private int

  val make : int -> int -> t
end = struct
  type t = int

  let make a b = a lor (b lsl 1)
end

module Shuffle2x2 : sig
  type t = private int

  val make : int -> int -> int -> int -> t
end = struct
  type t = int

  let make a b c d = a lor (b lsl 1) lor (c lsl 2) lor (d lsl 3)
end

module Shuffle4 : sig
  type t = private int

  val make : int -> int -> int -> int -> t
end = struct
  type t = int

  let make a b c d = a lor (b lsl 2) lor (c lsl 4) lor (d lsl 6)
end

module Permute2 = Shuffle2
module Permute2x2 = Shuffle2x2
module Permute4 = Shuffle4

module Float = struct
  module Comparison : sig
    type t = private int

    val equal : t
    val less : t
    val less_or_equal : t
    val unordered : t
    val not_equal : t
    val not_less : t
    val not_less_or_equal : t
    val ordered : t
  end = struct
    type t = int

    let equal = 0x0
    let less = 0x1
    let less_or_equal = 0x2
    let unordered = 0x3
    let not_equal = 0x4
    let not_less = 0x5
    let not_less_or_equal = 0x6
    let ordered = 0x7
  end

  module Rounding : sig
    type t = private int

    val nearest : t
    val negative_infinity : t
    val positive_infinity : t
    val zero : t
    val current : t
  end = struct
    type t = int

    let nearest = 0x8
    let negative_infinity = 0x9
    let positive_infinity = 0xA
    let zero = 0xB
    let current = 0xC
  end
end
