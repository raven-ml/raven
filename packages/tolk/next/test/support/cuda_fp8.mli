(** CUDA sources that keep float8 infinities special (DIVERGENCES D16).

    Where tinygrad's CUDA source converts a value to a float8 type [T] with the
    saturating constructor [(T)(value)], tolk.next's calls
    [tg_fp8<T>(value, byte)], a helper the kernel declares only when it
    converts: it converts as the constructor does, then writes the bits of the
    infinity's image, [byte] and its sign, if the value was infinite. An
    infinite float8 constant is [tg_bitcast<T>((unsigned char)byte)], the bits
    of its image. *)

open Tolk_next

val helper : string
(** [helper] is the start of the line that declares [tg_fp8]. *)

val infinity_byte : Dtype.t -> float -> string
(** [infinity_byte dt inf] is the byte, as [0xHH], of the image of the infinity
    [inf] in the float8 type [dt]: itself in e5m2 and a NaN of its sign in e4m3.
*)

val tinygrad_of_d16 : string -> string
(** [tinygrad_of_d16 src] is the CUDA source [src] with each call of [tg_fp8]
    and each infinite constant written back as tinygrad writes it, and without
    the declaration of [tg_fp8].

    Raises [Invalid_argument] on a constant whose byte is the image of no
    infinity. *)
