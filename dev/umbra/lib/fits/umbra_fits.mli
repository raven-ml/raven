(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** FITS file I/O.

    Reads and writes {{:https://fits.gsfc.nasa.gov/fits_standard.html}FITS}
    files. Binary tables are loaded into {!Talon.t} dataframes and images into
    {!Nx.t} tensors. All data is converted from FITS big-endian on read and
    written as big-endian on write. *)

(** {1:inspect Inspection} *)

(** The type for FITS header data unit kinds. *)
type hdu_type =
  | Primary  (** Primary HDU. *)
  | Image  (** Image extension. *)
  | Bintable  (** Binary table extension. *)
  | Ascii_table  (** ASCII table extension. *)

type hdu_info = {
  index : int;  (** Zero-based HDU index. *)
  hdu_type : hdu_type;  (** Kind of HDU. *)
  dimensions : int array;  (** NAXIS values. *)
  num_rows : int option;  (** Row count for table HDUs. *)
  num_cols : int option;  (** Column count for table HDUs. *)
}
(** The type for HDU summary information. *)

type header_card = {
  key : string;  (** Keyword name (up to 8 characters). *)
  value : string;  (** Parsed value string. *)
  comment : string;  (** Inline comment, if any. *)
}
(** The type for FITS header cards. *)

val info : string -> hdu_info list
(** [info path] is the summary information for every HDU in the FITS file at
    [path].

    Raises [Failure] if [path] cannot be read or is not a valid FITS file. *)

val header : ?hdu:int -> string -> header_card list
(** [header path] is the header cards for HDU [hdu] in the FITS file at [path],
    including COMMENT and HISTORY cards.

    [hdu] defaults to [0] (primary HDU).

    Raises [Failure] if [hdu] is out of range. *)

(** {1:reading Reading} *)

val read_table : ?hdu:int -> string -> Talon.t
(** [read_table path] reads a BINTABLE extension into a dataframe.

    [hdu] defaults to [1] (first extension).

    Supported TFORM types: [E] (float32), [D] (float64), [J] (int32), [K]
    (int64), [I] (int16), [B] (uint8), [L] (logical), [A] (string). Vector
    columns (repeat > 1) are not supported except for strings. TSCAL and TZERO
    are applied when present.

    Raises [Failure] if the HDU is not a BINTABLE, [hdu] is out of range, or a
    column has an unsupported TFORM type. *)

val read_image : ?hdu:int -> string -> Nx_io.packed
(** [read_image path] reads an image HDU into a packed {!Nx.t} tensor.

    [hdu] defaults to [0] (primary HDU).

    Supported BITPIX values: [8], [16], [32], [64], [-32], [-64].

    When BSCALE or BZERO header cards are present with non-trivial values
    (BSCALE != 1.0 or BZERO != 0.0), the physical values [BZERO + BSCALE * raw]
    are computed and the result is returned as float64 regardless of the
    original BITPIX. When neither card is present or both have default values,
    the raw data type is preserved.

    Raises [Failure] if the HDU is not an image, [hdu] is out of range, or
    BITPIX is unsupported. *)

(** {1:writing Writing} *)

val write_table : ?overwrite:bool -> string -> Talon.t -> unit
(** [write_table path df] writes [df] as a single BINTABLE extension preceded by
    an empty primary HDU.

    [overwrite] defaults to [true].

    Raises [Failure] if [overwrite] is [false] and [path] already exists. *)

val write_image : ?overwrite:bool -> string -> ('a, 'b) Nx.t -> unit
(** [write_image path tensor] writes [tensor] as a primary image HDU.

    [overwrite] defaults to [true].

    Supported dtypes: uint8, int16, int32, int64, float32, float64.

    Raises [Failure] if [overwrite] is [false] and [path] already exists, or if
    the dtype is unsupported. *)
