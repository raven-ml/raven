(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The Multimodal Universe's survey files, read as AION-1's inputs.

    The Multimodal Universe (MMU, arXiv:2412.02527) publishes each survey as
    HDF5 files, one per HEALPix pixel, each dataset a column with one row per
    object. A {e source} is one survey's datasets in one group of one file: the
    root of MMU's own files, or a group, such as ["/desi"], of a file that holds
    several surveys' rows, as AION-1's tutorial file does. Its objects are the
    rows of its id dataset: [source_id] for Gaia, [object_id] for the others.

    A modality's data are the datasets its survey stores, as this table maps
    them:
    {v
    legacysurvey  Image                 image_array, bands from image_band
                  Flux_g .. Flux_w4     FLUX_G .. FLUX_Z, FLUX_W1 .. FLUX_W4
                  Shape_r, _e1, _e2     SHAPE_R, SHAPE_E1, SHAPE_E2
                  Ebv                   EBV
    hsc           Image_hsc             image_array, bands from image_band
                  A_g .. A_y            a_g .. a_y
                  Mag_g .. Mag_y        g_cmodel_mag .. y_cmodel_mag
                  Shape11, 22, 12       i_sdssshape_shape11, 22, 12
    desi, sdss    Spectrum _            spectrum_flux, _ivar, _mask, _lambda
                  Z                     Z
    gaia          Xp_bp, Xp_rp          coeff, columns 0 to 54 and 55 to 109
                  Flux_g_gaia ..        phot_g_mean_flux, phot_bp_mean_flux,
                                        phot_rp_mean_flux
                  Parallax, Ra, Dec     parallax, ra, dec
    v}
    A source gives the modalities whose datasets it holds. Units are MMU's,
    which are those each codec takes ({!Image.unit} for pixels).

    Every value is cast to float32 once, as it is read: MMU's dataset builders
    declare every float column float32, which is how AION-1's training read
    them, though the files hold some as float64 (Gaia's fluxes, parallax, [ra]
    and [dec], DESI's [Z]). Bands are named in lower case, as ["des-g"],
    {!Image.name}'s in upper case. MMU pads each row of a spectrum file to its
    longest row with pixels of wavelength [-1]; a row's length is the number of
    pixels before them ({!Spectrum.pad}).

    Two sources' objects are the same object where they are each other's
    nearest neighbour within {!radius}, as AION-1's training set was
    cross-matched. Positions are compared as given: Gaia's, at epoch J2016.0,
    are not moved. *)

(** {1:surveys Surveys} *)

(** The type for MMU's surveys whose modalities AION-1 reads. *)
type survey =
  | Legacy_survey  (** The Legacy Surveys' DR10 south. *)
  | Hsc  (** The Hyper Suprime-Cam Subaru Strategic Program's PDR3. *)
  | Desi  (** DESI's early data release. *)
  | Sdss  (** The Sloan Digital Sky Survey's spectra. *)
  | Gaia  (** Gaia's DR3. *)

val name : survey -> string
(** [name s] is MMU's name of [s], its directory: ["legacysurvey"], ["hsc"],
    ["desi"], ["sdss"] or ["gaia"]. *)

(** {1:sources Sources} *)

type t
(** The type for sources. A source keeps its file's datasets, which read the
    file again for each {!read}: the file must not change in place while the
    source is reachable. *)

val load : ?group:string -> survey -> string -> t
(** [load ~group s path] is the source of [s] in the group [group] (default
    ["/"]) of the HDF5 file [path], as {!Nx_io.load_hdf5} reads it.

    Raises [Failure] as {!Nx_io.load_hdf5} does, or naming [path] and [group]
    if the group holds no id dataset, [ra] or [dec]. *)

val length : t -> int
(** [length t] is [t]'s number of objects. *)

val modalities : t -> Modality.t list
(** [modalities t] is the modalities whose datasets [t] holds, in
    {!Modality.all}'s order. *)

val read : t -> int array -> Modality.t -> Nx.host Codecs.data
(** [read t rows m] is [m]'s data of the objects [rows] of [t], in that order:
    a spectrum with each row's length.

    Raises [Invalid_argument] if [m] is not in {!modalities}[ t] or a row is
    outside [\[0, length t)]. Raises [Failure] naming the file, the dataset and
    the row for a dataset of another number of rows, for an image whose rows'
    bands differ or name no band of [m] ({!Image.name}), and for a spectrum row
    whose wavelengths of [-1] are not a suffix or are all of its pixels; and as
    {!Nx_io.Hdf5.rows} and {!Nx_io.Hdf5.strings} do. *)

(** {1:matching Cross-matching} *)

val radius : (float, Nx.float64_elt, Nx.host) Nx.t Ymir.Quantity.t
(** [radius] is 1″, AION-1's matching radius (arXiv:2510.17960, §3.6), which is
    also the default of MMU's cross-matching. *)

val pair : t -> t -> int array * int array
(** [pair a b] is [(i, j)], the rows [i.(k)] of [a] and [j.(k)] of [b] that are
    each other's nearest neighbour within {!radius}, the lowest row winning a
    tie, in increasing [i]. A row whose [ra] or [dec] is NaN pairs with none.

    Raises [Invalid_argument] as {!Ymir.Direction.lonlat} does for an infinite
    angle. *)

