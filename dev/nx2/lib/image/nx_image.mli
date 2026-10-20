(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Images to and from PNG and JPEG files.

    An image is a [uint8] array of shape [[|height; width|]] for one channel, or
    [[|height; width; c|]] for [c] channels: one for grey, three for red, green
    and blue, four for those and alpha. Its first row is the top of the picture.
    A loaded image is a host value ({!Nx.Host}). A save takes an image of any
    set and reads it on the host ({!Nx.Repr.read}), which leaves its memory
    unshared. Beyond the image's own memory it holds a copy where the host does
    not address the image, a C-order copy where the image is not in C order, up
    to twice the image's bytes, and the encoded file.

    PNG files are those of the
    {{:https://www.w3.org/TR/png-3/}PNG Specification, Third Edition}: every
    colour type and bit depth, palettes, transparency and interlacing. JPEG
    files are those of {{:https://www.itu.int/rec/T-REC-T.81}ITU-T T.81} with
    8-bit samples and Huffman coding, sequential or progressive: grey, YCbCr,
    RGB, CMYK and YCCK, with any sampling and restart intervals.

    A save writes a new file beside its path, orders the file's bytes before
    every later change to the file system, then renames it over the path and
    syncs the directory. After a crash, the path holds what it held before or
    the whole new file, and a save that fails leaves it as it was and nothing
    beside it.

    A file that cannot be read or written, or whose contents are no PNG or JPEG
    stream, raises [Failure] whose message starts with the file's path. An image
    a format cannot hold raises [Invalid_argument] naming the function. *)

val load : ?grayscale:bool -> string -> Nx.host Nx.uint8_t
(** [load path] is the image of the PNG or JPEG file [path], recognised by its
    contents. It has shape [[|height; width; 3|]], or [[|height; width|]] with
    [~grayscale:true] (defaults to [false]). An alpha channel is dropped, a grey
    image loads as three equal channels, and [~grayscale:true] reads a colour
    image as its luma [0.299 R + 0.587 G + 0.114 B] (ITU-R BT.601), within one
    level. A 16-bit sample keeps its high byte, and a sample of 1, 2 or 4 bits
    is scaled to 8 bits, rounded to nearest. The image lies in host memory of
    its own, which its donation may write ({!Nx.donate}).

    Raises [Failure] if the file is no PNG or JPEG stream, or a malformed or
    truncated one, or if a checksum does not match. *)

val save : ?overwrite:bool -> string -> 'd Nx.uint8_t -> unit
(** [save path x] writes the image [x] to [path], as PNG for the extension
    [.png] and as JPEG for [.jpg] and [.jpeg], in any case. PNG holds 1, 3 or 4
    channels at 8 bits; JPEG holds 1 or 3 channels, written baseline at quality
    90 (the standard tables of T.81's Annex K scaled to 20%), colour with its
    chroma subsampled 4:2:0. [overwrite] defaults to [true]. With
    [~overwrite:false] the new file is linked to the path, which fails if the
    path names something; on a file system without hard links, such as FAT, the
    path is checked and then the file renamed.

    Raises [Invalid_argument] if the extension is neither, or if [x] has no
    pixel or a shape its format does not hold. *)

val encode_png : ?dpi:float -> ?srgb:bool -> 'd Nx.uint8_t -> string
(** [encode_png x] is the contents of a PNG file of the image [x], without
    touching the file system:
    - [dpi], the image's physical resolution in pixels per inch, is written as a
      [pHYs] chunk of [Float.round (dpi /. 0.0254)] pixels per metre on both
      axes, so that viewers and printers show the image at its physical size.
      Without it the file states no physical size.
    - [srgb] is whether to write an [sRGB] chunk with the perceptual rendering
      intent, which states that the samples are sRGB. Defaults to [false].

    Without [dpi] and [srgb] it is the file {!save} writes for [x].

    Raises [Invalid_argument] if [x] has no pixel or not 1, 3 or 4 channels, or
    if [Float.round (dpi /. 0.0254)] is not in \[[1];[2{^31} - 1]\], the range
    of a PNG integer: [dpi] is below about [0.0127], above about [5.45e7], or
    not a number. *)
