(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The narrow upstream fixtures of [upstream/], read from the directory a test
    runs in, and the inputs they hold. The test dune file names their
    generators. *)

(** {1:files Files} *)

val load : string -> Nx.host Nx_io.Archive.t
(** [load name] is [upstream/name.safetensors], read once per process. *)

val get :
  Nx.host Nx_io.Archive.t ->
  ('a, 'b) Nx.dtype ->
  string ->
  ('a, 'b, Nx.host) Nx.t
(** [get a dt name] is [a]'s entry [name] of dtype [dt].

    Raises [Failure] naming [name] if [a] lacks it. *)

val ensemble : Nx.host Nx_io.Archive.t -> int
(** [ensemble a] is the number of runs of [a]'s rounding ensemble, its entry
    ["n"]. *)

val take : int list -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [take rows x] is the rows [rows] of [x], along its first axis. *)

(** {1:inputs Inputs} *)

val spectra :
  Nx.host Nx_io.Archive.t ->
  (Nx.host Nx.float32_t, Nx.host) Aion.Spectrum.spectrum
(** [spectra a] is the spectra of [a]'s entries [input.flux], [input.ivar],
    [input.mask] (nonzero where bad) and [input.lambda], as spectrum_golden
    holds them. *)

val transformer : unit -> (Nx.host Nx.float32_t, Nx.host) Aion.Transformer.t
(** [transformer ()] is the narrow AION of [transformer.safetensors]: width 16,
    2 heads, 2 encoder and 2 decoder blocks, over tok_xp_bp, tok_flux_g, tok_z
    and tok_xp_rp. *)

val inputs :
  ?rows:int list ->
  ?step:string ->
  Aion.Modality.t list ->
  Nx.host Aion.Tokens.t list
(** [inputs ~rows ~step ms] is transformer_golden's inputs of the modalities
    [ms] for its objects [rows] (default all four), whose tokens are partly
    absent. With [step], one of roar_golden's steps ["roar.0"] or ["roar.1"],
    tok_xp_rp follows, its tokens present where the step has revealed them. *)
