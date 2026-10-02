(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scales.

    A scale denotes a {e normalisation}, a function from a domain of data values
    to the reals that maps the domain onto \[[0];[1]\], with an inverse where
    one exists. {!Ticks} chooses the ticks of a scale.

    There are three kinds of scales ({!type-kind}): quantitative scales over
    floats ({!linear}, {!log}, {!symlog}, {!pow}, {!custom}), temporal scales
    over instants ({!time}) and categorical scales over category names
    ({!band}). The first two are {e continuous}.

    A scale value is a {e specification}. Each of its properties is either set,
    by an argument of its constructor or by the functions of
    {{!section-specs}specifications}, or unset, and the scale means the
    normalisation its properties give once each unset one takes its default.
    Specifications combine with {!merge} and {!imply}, and {!fit} sets the
    domain from observed data.

    {1:normalisation Normalisation}

    {2:continuous Continuous scales}

    A continuous scale has a domain \[[a];[b]\] with [a <= b] and a strictly
    monotone {e transform} [T], which its constructor fixes. It normalises [x]
    to

    {[
    N x = (T x - T a) / (T b - T a)
    ]}

    which is exactly [0.] at [a], [1.] at [b], and beyond \[[0];[1]\] for [x]
    outside the domain: scales extrapolate. The formula is evaluated without
    overflow, so this holds on \[[-.max_float];[max_float]\] too. A domain is
    {e constant} if its ends' transforms are the same float, [T a = T b]: always
    if [a = b], and for distinct ends too close for the transform to separate,
    such as \[[1e308];[1.0000000000000002e308]\] on a log scale. A constant
    domain normalises every value that is not missing to [0.5]. A reversed scale
    normalises [x] to [1 - N x], and a clamped scale clamps the result into
    \[[0];[1]\], after reversing. The transforms are:
    - {!linear}: [T x = x];
    - {!log} in base [b]: [T x = log x / log b], defined for [x > 0];
    - {!symlog} with constant [c]: [T x = sign x * log1p (|x| / c)];
    - {!pow} with exponent [e]: [T x = sign x * |x / m| ** e], where [m] is the
      greater magnitude of the domain's ends, or [1.] if both are [0.];
    - {!custom}: the caller's [forward] function.

    A temporal scale normalises instants by the same formula with [T] the
    identity, the differences [t - a] and [b - a] taken exactly in nanoseconds
    and each rounded once to a float. Its domain is constant iff [a = b].

    {2:categorical Categorical scales}

    A band scale over [n] categories with padding [p] divides \[[0];[1]\] into
    [n] steps of width [1 / (n + p)], in domain order, with [p / (2 (n + p))]
    left over at each end, and normalises the category [c_i] at index [i] of its
    domain, from [0], to the centre of its step

    {[
    N c_i = (i + ((1 + p) / 2)) / (n + p)
    ]}

    or to [1 - N c_i] if the scale is reversed. A band is its step narrowed by
    [p]: its width is [(1 - p) / (n + p)] ({!bandwidth}).

    {2:missing Missing values}

    A value is {e missing} for a scale, and normalises to [nan], if it is [nan]
    or infinite, not positive on a log scale, a value at which a custom
    transform is not finite, or a string that names no category of a band scale.
*)

open Hugin_next_gg

(** {1:types Types} *)

type 'd t
(** The type for scales over domain values of type ['d]. *)

(** The type for kinds of scales. The kind of a scale is fixed by the type of
    its domain values. *)
type _ kind =
  | Quantitative : float kind  (** Quantities. *)
  | Temporal : Time.t kind  (** Instants. *)
  | Categorical : string kind  (** Category names. *)

(** The type for the categories of band scales. Wherever a scale takes or
    returns a category, a labelled category is named by its label and an indexed
    one by its integer as [string_of_int] writes it. Arrays are copied where
    they enter or leave a scale. *)
type categories =
  | Labels of string array
      (** Labelled categories, in domain order, each identified by its label.
          Labels are distinct. *)
  | Indices of (int * string) array
      (** Indexed categories, in increasing order of their distinct integers,
          each identified by its integer and shown by its text. Texts may
          repeat. *)

(** The type for domains, one case per kind, so [let Floats (a, b) = domain s]
    is exhaustive. *)
type _ domain =
  | Floats : float * float -> float domain  (** A quantitative domain. *)
  | Instants : Time.t * Time.t -> Time.t domain  (** A temporal domain. *)
  | Categories : categories -> string domain  (** A categorical domain. *)

(** {1:constructors Constructors}

    A constructor fixes the kind of its scale and, for a quantitative scale, its
    transform. Each of its optional arguments is a property: given, it is set;
    omitted, it is unset and counts as the default stated here. The properties
    are:
    - [name], the name of the scale. Unset, the scale specifies the default
      scale of the role that reads it.
    - [domain], the domain, final as given: {!fit} neither widens nor rounds a
      set domain. A continuous domain [(a, b)] must have finite ends with
      [a <= b]. Unset, the domain is the default its constructor states until
      {!fit} sets it.
    - [nice], whether {!fit} rounds a fitted domain outward ({!section-nice}).
      Unset counts as [true].
    - [zero], whether {!fit} widens a fitted domain to include [0.]
      ({!section-nice}). Unset counts as [false].
    - [clamp], whether normalised values are clamped into \[[0];[1]\]. Unset
      counts as [false].
    - [reverse], whether normalised values run from [1] to [0]. Unset counts as
      [false].
    - [scheme], the colours that colour roles paint with. A continuous scale
      colours a value with {!Scheme.color} of its normalised value, and a band
      scale over [n] categories colours category [i] with colour [i] of
      [Scheme.colors n], whether or not it is reversed: reversing a band scale
      moves its categories and keeps their colours. Unset, a role takes its
      theme's scheme on a continuous scale and its theme's palette on a band
      scale.
    - [areas], the symbol areas in square points that the size role draws at the
      normalised values [0] and [1]. Both must be finite and not negative.
      Unset, the role takes its theme's areas.
    - [symbols], the symbols that the symbol role draws the categories of a band
      scale with: category [i] with [symbols.(i mod k)] for [k] symbols. The
      array must not be empty, and is copied in and out. Unset, the role takes
      its theme's set for the way the mark paints its symbols.
    - [unknown], the colour that colour roles paint missing values with. Unset,
      missing values draw no paint.

    Constructors raise [Invalid_argument] if a property breaks the constraint
    stated for it. *)

val linear :
  ?name:string ->
  ?domain:float * float ->
  ?nice:bool ->
  ?zero:bool ->
  ?clamp:bool ->
  ?reverse:bool ->
  ?scheme:Scheme.t ->
  ?areas:float * float ->
  ?unknown:Color.t ->
  unit ->
  float t
(** [linear ()] is a linear scale. Its default domain is \[[0];[1]\]. *)

val log :
  ?base:float ->
  ?name:string ->
  ?domain:float * float ->
  ?nice:bool ->
  ?clamp:bool ->
  ?reverse:bool ->
  ?scheme:Scheme.t ->
  ?areas:float * float ->
  ?unknown:Color.t ->
  unit ->
  float t
(** [log ~base ()] is a logarithmic scale in base [base], which defaults to
    [10.] and must be finite and greater than [1.]. Its domain must be positive,
    and its default domain is \[[1];[base]\]. It takes no [zero], since [0.] is
    missing on a log scale.

    The rules for ticks, tick labels and minor ticks ({!Ticks}) that name an
    {e integer base} mean a base that is an integer from [2] to [16]; a greater
    integer base, whose powers hold more multiples than an axis can show,
    follows the rules of other bases. *)

val symlog :
  ?constant:float ->
  ?name:string ->
  ?domain:float * float ->
  ?nice:bool ->
  ?zero:bool ->
  ?clamp:bool ->
  ?reverse:bool ->
  ?scheme:Scheme.t ->
  ?areas:float * float ->
  ?unknown:Color.t ->
  unit ->
  float t
(** [symlog ~constant ()] is a symmetric logarithmic scale with constant
    [constant], which defaults to [1.] and must be finite and positive. Its
    default domain is \[[0];[1]\]. *)

val pow :
  exponent:float ->
  ?name:string ->
  ?domain:float * float ->
  ?nice:bool ->
  ?zero:bool ->
  ?clamp:bool ->
  ?reverse:bool ->
  ?scheme:Scheme.t ->
  ?areas:float * float ->
  ?unknown:Color.t ->
  unit ->
  float t
(** [pow ~exponent ()] is a power scale with exponent [exponent], which must be
    finite and positive. Its default domain is \[[0];[1]\]. *)

val custom :
  transform:string ->
  forward:(float -> float) ->
  inverse:(float -> float) ->
  ?name:string ->
  ?domain:float * float ->
  ?nice:bool ->
  ?zero:bool ->
  ?clamp:bool ->
  ?reverse:bool ->
  ?scheme:Scheme.t ->
  ?areas:float * float ->
  ?unknown:Color.t ->
  unit ->
  float t
(** [custom ~transform ~forward ~inverse ()] is a quantitative scale with
    transform [forward], whose inverse is [inverse], and which [transform] names
    in printed forms. Its default domain is \[[0];[1]\], its nice domains are
    those of a linear scale over the same domain, and its tick candidates the
    decimal steps of one ({!Ticks}). [forward] must be strictly monotone where
    it is finite, [inverse] must be its inverse there, and neither may read
    mutable state; results are unspecified otherwise. A set domain must have
    ends that are not missing, where [forward] is finite. An unfitted custom
    scale whose default ends are missing, such as a logit, normalises every
    value to [nan]. *)

val time :
  ?name:string ->
  ?domain:Time.t * Time.t ->
  ?nice:bool ->
  ?clamp:bool ->
  ?reverse:bool ->
  ?tz_offset_s:Time.tz_offset_s ->
  ?scheme:Scheme.t ->
  ?unknown:Color.t ->
  unit ->
  Time.t t
(** [time ()] is a temporal scale whose ticks and nice domains follow the
    calendar at the offset [tz_offset_s] (see {!Time.type-tz_offset_s}); unset
    counts as [0]. Its default domain is the first day of 1970 UTC, from
    {!Time.epoch} to one day later. *)

val band :
  ?name:string ->
  ?domain:categories ->
  ?padding:float ->
  ?reverse:bool ->
  ?wrap:int ->
  ?scheme:Scheme.t ->
  ?symbols:Symbol.t array ->
  ?unknown:Color.t ->
  unit ->
  string t
(** [band ()] is a categorical scale ({!section-categorical}) with:
    - [padding], the fraction of each step that is not band, in \[[0];[1]\];
      [1.] makes bands points. Unset counts as [0.].
    - [wrap], at least [1], the number of panels per row when a facet channel
      reads the scale. Unset, panels do not wrap.

    Its default domain is [Labels [||]], without categories. A domain's labels
    must be distinct and its integers strictly increasing. *)

(** {1:properties Properties} *)

val kind : 'd t -> 'd kind
(** [kind s] is the kind of [s]. *)

val equal_kind : 'a kind -> 'b kind -> ('a, 'b) Type.eq option
(** [equal_kind k k'] is [Some Type.Equal] iff [k] and [k'] are the same kind.
*)

val name : 'd t -> string option
(** [name s] is the name of [s], if set. *)

val domain : 'd t -> 'd domain
(** [domain s] is the domain of [s]: the one set, or the default. *)

val bandwidth : string t -> float
(** [bandwidth s] is the width of the bands of [s] in normalised units,
    [(1 - p) / (n + p)] for [n] categories and padding [p], and [0.] if [s] has
    no category. *)

val length : 'd t -> float
(** [length s] is the length of the domain of [s] in the units its normalisation
    divides: [|T b - T a|] for a continuous domain \[[a];[b]\] and the transform
    [T] ({!section-continuous}), in seconds on a temporal scale, and [n + p] on
    a band scale of [n] categories and padding [p], the steps that \[[0];[1]\]
    holds ({!section-categorical}), or [0.] without categories. It is [infinity]
    if the difference overflows and [nan] if an end of the domain is missing. *)

val wrap : string t -> int option
(** [wrap s] is the number of facet panels per row of [s], if set. *)

val scheme : 'd t -> Scheme.t option
(** [scheme s] is the colour scheme of [s], if set. *)

val areas : float t -> (float * float) option
(** [areas s] is the symbol areas of [s], if set. *)

val symbols : string t -> Symbol.t array option
(** [symbols s] is the symbols of [s], if set, as a fresh array. *)

val unknown : 'd t -> Color.t option
(** [unknown s] is the colour of the missing values of [s], if set. *)

(** The type for the transforms of quantitative scales ({!section-continuous}).
    A custom transform is known by its name. *)
type transform =
  | Linear  (** {!linear}. *)
  | Log of float  (** {!log} in the given base. *)
  | Symlog of float  (** {!symlog} with the given constant. *)
  | Pow of float  (** {!pow} with the given exponent. *)
  | Custom of string  (** {!custom} with the given name. *)

val transform : float t -> transform
(** [transform s] is the transform of [s]. *)

val tz_offset_s : Time.t t -> Time.tz_offset_s
(** [tz_offset_s s] is the offset of the calendar of [s], [0] if unset. *)

(** {1:mapping Normalising and inverting} *)

val normalize : 'd t -> 'd -> float
(** [normalize s x] is the normalised value of [x] on [s]
    ({!section-normalisation}), [nan] iff [x] or an end of the domain of [s] is
    {{!section-missing}missing} for [s]; only the default domain of a {!custom}
    scale can have a missing end. [normalize s] may be applied once and reused:
    the partial application computes what depends on [s] alone. *)

val normalize_index : string t -> int -> float
(** [normalize_index s k] is [normalize s c] for the category [c] at the index
    [k] of the domain of [s], and [nan] if [k] is not an index of the domain.
    Like [normalize s], [normalize_index s] may be applied once and reused. *)

val invert : 'd t -> float -> 'd option
(** [invert s u] is the value that [s] normalises to [u]:
    - on a continuous scale, [Some x] for the [x] with [normalize s x = u] up to
      rounding, [u] first clamped into \[[0];[1]\] if [s] clamps; on a
      {{!section-continuous}constant} domain \[[a];[b]\], [Some a] for every
      finite [u]. It is [None] if [u] is not finite, if [x] is
      {{!section-missing}missing} for [s] or, for an instant, if [x] is not
      representable. Instants are rounded to the nearest nanosecond.
    - on a band scale, [Some c] for the category [c] whose step contains [u],
      the earlier in domain order on the boundary of two steps, and [None] if no
      step contains [u]: in the outer padding, outside \[[0];[1]\], or without
      categories. *)

(** {1:time_intervals Time intervals}

    Temporal ticks and nice domains fall on the boundaries
    ({!Time.type-interval}) of one of these intervals, finest first:
    - [1], [2] and [5] times [10^k] nanoseconds for [k] from [0] to [8];
    - [1], [5], [15] and [30] seconds, and as many minutes;
    - [1], [3], [6] and [12] hours;
    - [1] and [2] days, and as many weeks;
    - [1], [3] and [6] months;
    - [1], [2], [5], [10], [20], [50] and [100] years, then every greater [1],
      [2] or [5] times a power of ten years.

    To compare durations, a month counts 30 days and a year 365 days; the
    boundaries themselves are those of the calendar. *)

(** {1:specs Specifications and fitting}

    {2:nice Nice domains}

    [zero] and [nice] change fitted domains only. [zero] widens a domain to
    include [0.] if [0.] is not {{!section-missing}missing} for the scale: never
    on a log scale. [nice] then rounds the ends of a domain that is not
    {{!section-continuous}constant} outward:
    - on linear, pow and custom scales, to multiples of the {e step} [m × 10^k]
      nearest by ratio to [t], a tenth of the domain's length, repeated until
      that step stops changing, at most ten times. With [t] written [f × 10^k],
      [1 <= f < 10], [m] is [10] if [f >= √50], [5] if [f >= √10], [2] if
      [f >= √2] and [1] otherwise. For the domain \[[a];[b]\], [t] is
      [(b -. a) /. 10.], or twice [(b /. 2. -. a /. 2.) /. 10.] if [b -. a]
      overflows. A multiple is the float nearest [i × m × 10^k] for its integer
      index [i], so [3 × 0.1] is [0.3];
    - on log scales, to integer powers of the base;
    - on symlog scales of constant [c], an end farther than [c] from zero to a
      power of ten or its negation, or to [c] or [-c] if that power is within
      [c] of zero, and an end within [c] of zero to [0.], [c] or [-c];
    - on temporal scales, to boundaries of the
      {{!section-time_intervals}time interval} whose duration is nearest by
      ratio to a tenth of the domain's length, the coarser of two equally near,
      repeated until that interval stops changing, at most ten times.

    On every kind, an end whose rounded value is not finite, is missing for the
    scale or, for an instant, is not representable stays as it is. *)

(** The type for the properties of scales. Each case stands for the constructor
    argument {!pp_property} names, and [Transform] for the constructor itself
    with its [base], [constant], [exponent] or functions. *)
type property =
  | Name
  | Transform
  | Domain
  | Nice
  | Zero
  | Clamp
  | Reverse
  | Padding
  | Wrap
  | Tz_offset_s
  | Scheme
  | Areas
  | Symbols
  | Unknown

val sets : property -> 'd t -> bool
(** [sets p s] is [true] iff [s] sets [p]. Every constructor sets [Transform].
*)

val merge : 'd t -> 'd t -> ('d t, property) result
(** [merge s s'] is [Ok m] if no property is set to different values in [s] and
    [s'], where [m] sets each property that [s] or [s'] sets to its value there,
    and [Error p] otherwise, for the first such property [p] in the order of
    {!type-property}. Values compare as {!equal} compares them, so two custom
    transforms differ unless their functions are physically equal, and labelled
    and indexed categories always differ. [merge] is commutative and associative
    on its successes, and [merge s s] is [Ok s]. *)

val imply : 'd t -> 'd t -> 'd t
(** [imply i s] is [s] with each property that [s] leaves unset taken from [i],
    except the name, which [imply] keeps from [s] or keeps absent. Since every
    constructor sets the transform, the transform is that of [s]. A domain that
    [i] sets is taken only if it satisfies the constraints the constructor of
    [s] states on domains. *)

val missing : float t -> ('a, 'b) Nx.t -> Nx.bool_t
(** [missing s x] is [true] where an element of [x] is
    {{!section-missing}missing} for [s], and has the shape of [x]. Elements are
    converted to floats first, so an integer beyond 2{^ 53} is rounded, and only
    the transform of [s] is read. It is computed where [x] lives, except on a
    custom scale, whose transform is an OCaml function: [x] is then read to the
    host.

    Raises [Invalid_argument] if [x] has a complex or boolean dtype. *)

val fit : 'd domain option -> 'd t -> 'd t
(** [fit observed s] is [s] with [nice] and [zero] unset and its domain set to:
    - the domain of [s], if set;
    - otherwise, for [Some (Floats (lo, hi))], \[[lo];[hi]\], the hull of the
      observed values, widened by [zero] and rounded by [nice]
      ({!section-nice});
    - for [Some (Instants (lo, hi))], the hull \[[lo];[hi]\] of the observed
      instants, rounded by [nice];
    - for [Some (Categories c)], [c];
    - for [None], when nothing was observed, the default domain of [s].

    Once the domain is set [nice] and [zero] have no effect, so the result
    normalises as [s] does with that domain, and fitting it again changes
    nothing: [fit o (fit o' s)] is [fit o' s].

    Raises [Invalid_argument] if the domain of [s] is unset and [observed]
    breaks the constraints the constructor of [s] states on domains. *)

val with_domain : 'd domain -> 'd t -> 'd t
(** [with_domain d s] is [s] with its domain set to [d].

    Raises [Invalid_argument] if [d] breaks the constraints the constructor of
    [s] states on domains. *)

val pp_property : Format.formatter -> property -> unit
(** [pp_property ppf p] formats [p] in lowercase, as the argument that sets it
    is named, such as [domain] or [tz_offset_s]. *)

(** {1:comparing Comparing and formatting} *)

val equal : 'd t -> 'd t -> bool
(** [equal s s'] is [true] iff [s] and [s'] set the same properties to equal
    values: floats by [Float.equal], colours by {!Color.equal}, schemes by
    {!Scheme.equal}, symbols element by element by {!Symbol.equal}, and the
    functions of custom transforms physically. A property unset differs from the
    same property set to its default. *)

val pp : Format.formatter -> 'd t -> unit
(** [pp ppf s] formats the properties [s] sets, for debugging and tests, a
    custom transform by its name. *)
