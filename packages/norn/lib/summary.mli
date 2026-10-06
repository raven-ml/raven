(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Summaries of draws, with their findings.

    A summary has a row per element of the draws, such as [theta[3]], and a
    column per statistic: the mean, the standard deviation, the 5%, 50% and 95%
    quantiles, the Monte Carlo standard error of the mean, the bulk and tail
    effective sample sizes and R-hat ({!Norn.Diag}). Its findings are the
    problems those numbers and the transitions' statistics show, as data: there
    are no warnings.

    {[
    Format.printf "%a@." Norn.Summary.pp (Norn.Summary.v schools ~stats post)
    ]} *)

type t
(** The type for summaries. *)

(** The type for findings. A finding about an element names it by the path of
    its tensor and its index in the tensor. *)
type finding =
  | Rhat_high of { path : Nx.Ptree.Path.t; index : int array; rhat : float }
      (** R-hat above [1.01]: split R-hat without superchains, nested R-hat with
          them. *)
  | Ess_low of {
      path : Nx.Ptree.Path.t;
      index : int array;
      bulk : float;
      tail : float;
    }
      (** A bulk or tail effective sample size below [100] per chain, or below
          [400] in total with superchains. *)
  | Not_finite of { path : Nx.Ptree.Path.t; index : int array; count : int }
      (** [count] of the element's draws are NaN or infinite: it has no
          diagnostic. *)
  | Constant of { path : Nx.Ptree.Path.t; index : int array }
      (** Every draw of the element is equal: it has no R-hat nor effective
          size. *)
  | Divergent of { count : int; total : int; regions : region list }
      (** [count] of the [total] transitions diverged. *)
  | Saturated of { count : int; total : int }
      (** [count] of the [total] transitions reached their maximum length before
          they turned. *)
  | Ebfmi_low of { chain : int; ebfmi : float }
      (** A chain's E-BFMI is below [0.3]. *)

and region = { path : Nx.Ptree.Path.t; index : int array; shift : float }
(** The type for where divergences gather: an element whose draws from
    transitions that diverged have a mean [shift] standard deviations from its
    mean, by more than one ({!Norn.Diag.divergent_shift}). *)

val v :
  'u Nx.Ptree.t ->
  ?stats:'f Stats.t Draws.t ->
  ?superchains:int ->
  'u Draws.t ->
  t
(** [v u ?stats ?superchains d] summarises the draws [d] of [u]. [stats], the
    transitions' statistics, adds the findings on divergences, depth and E-BFMI.
    [superchains] takes the chains as that many consecutive groups for
    {!Norn.Diag.nested_rhat}, which then fills the [rhat] column.

    The statistics are computed at float64, whatever the draws' dtype.

    Raises [Invalid_argument] if the chains have fewer than 4 draws, if
    [superchains] does not divide the chains, and as {!Norn.Diag} does for a
    tensor that is not of a float dtype. *)

val concat : t list -> t
(** [concat ss] is the rows and findings of [ss], in order: derived quantities
    summarised beside parameters. *)

val findings : t -> finding list
(** [findings s] is [s]'s findings, the elements' in row order, then the
    transitions'. *)

val labels : t -> string array
(** [labels s] is [s]'s row labels: an element's path, then its index in
    brackets for a tensor that is not a scalar, as [theta[3]] or [b[1,0]]. At
    the root path, a tensor is labelled by its index alone, and a scalar
    [value]. *)

val columns : t -> (string * Nx.float64_t) list
(** [columns s] is [s]'s columns, each a vector with one element per row:
    [mean], [sd], [q5], [median], [q95], [mcse], [ess_bulk], [ess_tail] and
    [rhat]. *)

val pp_finding : Format.formatter -> finding -> unit
(** [pp_finding ppf f] formats [f] as a sentence. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf s] formats [s] as a table, then its findings, one per line. *)
