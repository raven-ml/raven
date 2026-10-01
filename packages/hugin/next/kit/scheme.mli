(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Colour schemes.

    A scheme is the colours a scale's colour roles paint with. It has two
    readings. Its {e continuous} reading gives the colour of a normalised value
    ({!color}), for quantitative and temporal scales; its {e discrete} reading
    gives [n] colours for [n] classes ({!colors}), for a band scale over [n]
    categories. Every scheme has both, so a band scale can colour ordered
    categories with a sequential scheme, and a quantitative scale can bin its
    values into the classes of a {!palette}.

    Schemes are {{!section-sequential}sequential}, for quantities from low to
    high; {{!section-diverging}diverging}, for quantities on either side of a
    midpoint at the normalised value [0.5]; {{!section-cyclic}cyclic}, for
    angles and phases; or {{!section-qualitative}qualitative}, for categories
    without order. {!simulate} gives a colour as a viewer with a colour vision
    deficiency sees it, so that tests can check that a scheme's colours stay
    apart.

    {1:continuous Continuous reading}

    The continuous reading of a scheme is its {e table} ({!table}): [N >= 1]
    colours that divide \[[0];[1]\] into [N] bins of equal width. A normalised
    value [u] takes the colour [T.(i)] of its bin, where [i] is the floor of the
    float product [float N *. u] clamped to \[[0];[N - 1]\]. Values below [0],
    [neg_infinity] included, take the first colour; values from [1] up,
    [infinity] included, the last; and [nan] takes the colour given to {!color}
    for unknown values. A figure that gathers colours from the table by this
    index, as a heatmap drawn as one image or a colour bar does, paints exactly
    the colours {!color} gives.

    A scheme published as a table, such as {!viridis}, has that table. A {!ramp}
    samples its interpolant at [256] evenly spaced values, ends included. A
    {!palette} of [k] colours is its own table, of [k] bins.

    {1:discrete Discrete reading}

    The discrete reading [colors n s] gives the colours of classes [0] to
    [n - 1] by the first of these rules that applies:
    + The {!reverse} of a Brewer scheme or of a {!ramp} gives the colours of the
      scheme it reverses, in reverse order.
    + A {!palette} of [k] colours, as every qualitative scheme is, and its
      {!reverse} give the colours of their table in order and start again from
      the first after the last: class [i] has colour [i mod k] of the table. A
      class thus keeps its colour whatever [n] is.
    + A Brewer sequential or diverging scheme, for [n] up to the classes of its
      largest designed table, gives the table its authors designed for [n]
      classes. They designed none below three: for one class a sequential scheme
      gives the middle colour of its 3-class table and for two classes the last
      two, the darker; a diverging scheme gives the middle colour for one class
      and the first and last for two.
    + A {!ramp} through the [m] colours [cs], and a Brewer scheme past its
      largest designed table as the ramp through that table, gives class [i] the
      colour [cs.(j)] if [r = 0] and otherwise
      [Color.mix (float r /. float (n - 1)) cs.(j) cs.(j + 1)], where [j] and
      [r] are the quotient and the remainder of the integer division of
      [i × (m - 1)] by [n - 1]. For one class it gives class [1] of three, the
      middle of the ramp. Thus [colors m (ramp cs)] is [cs].
    + A cyclic scheme and its {!reverse} give their continuous reading at
      [i / n], since [0] and [1] are one angle.
    + Any other scheme and its {!reverse} give their continuous reading at
      [i / (n - 1)], ends included, or at [0.5] for one class.

    Past its [k] colours a palette repeats them, so categories share colours. A
    figure that warns about it compares the colours {!colors} gives.

    References:
    - Cynthia A. Brewer, Mark Harrower and The Pennsylvania State University.
      {{:https://colorbrewer2.org}ColorBrewer}. Apache License 2.0.
    - Stéfan van der Walt, Nathaniel Smith and Eric Firing.
      {{:https://bids.github.io/colormap/}A better default colormap for
       Matplotlib}. 2015. CC0.
    - Jamie R. Nuñez, Christopher R. Anderton and Ryan S. Renslow.
      {e Optimizing colormaps with consideration for color vision deficiency to
         enable accurate interpretation of scientific data}. PLOS ONE 13(7),
      2018.
    - Anton Mikhailov.
      {e Turbo, an improved rainbow colormap for visualization}. Google, 2019.
      Apache License 2.0.
    - Masataka Okabe and Kei Ito. {e Color Universal Design}. 2008.
    - Gustavo M. Machado, Manuel M. Oliveira and Leandro A. F. Fernandes.
      {e A physiologically-based model for simulation of color vision
         deficiency}. IEEE Transactions on Visualization and Computer Graphics
      15(6), 2009. *)

open Hugin_next_gg

(** {1:making Making schemes} *)

type t
(** The type for colour schemes. *)

val ramp : Color.t array -> t
(** [ramp cs] is the sequential scheme that runs through the [m] colours [cs] in
    order, at evenly spaced normalised values from [0] to [1], interpolating in
    Oklab with {!Color.mix} between consecutive ones. Its table is
    [colors 256 (ramp cs)] ({!section-discrete}), so a ramp through [256]
    colours has them as its table. [cs] is copied.

    Raises [Invalid_argument] if [cs] is empty. *)

val palette : Color.t array -> t
(** [palette cs] is the qualitative scheme of the colours [cs], in order. Its
    table is [cs], so its continuous reading bins \[[0];[1]\] into
    [Array.length cs] classes: [palette (colors 5 blues)] colours a quantitative
    scale in five classes. [cs] is copied.

    Raises [Invalid_argument] if [cs] is empty. *)

val reverse : t -> t
(** [reverse s] is [s] with its table in reverse order, so that low values take
    the colours [s] gives to high ones: [reverse rdbu] colours high values red.
    Its discrete reading is that of [s] in reverse order for a Brewer scheme or
    a ramp, and is otherwise read from the reversed table ({!section-discrete}),
    so that a reversed palette keeps each class's colour as classes are added
    and a reversed cyclic scheme keeps the colour of angle [0].
    [reverse (reverse s)] is [s]. *)

(** {1:readings Readings} *)

val color : ?unknown:Color.t -> t -> float -> Color.t
(** [color ~unknown s u] is the colour of the normalised value [u] in the
    {{!section-continuous}continuous reading} of [s], and [unknown] if [u] is
    [nan]. [unknown] defaults to {!Color.transparent}, which paints nothing; a
    figure passes [?unknown:(Scale.unknown sc)] for the scale [sc] it colours
    with. [color ~unknown s] may be applied once and reused. *)

val table : t -> Color.t array
(** [table s] is the table of [s] ({!section-continuous}), from the colour of
    [0] to the colour of [1], as a fresh array. *)

val colors : int -> t -> Color.t array
(** [colors n s] is the [n] colours of the {{!section-discrete}discrete reading}
    of [s], in class order, as a fresh array. [colors 0 s] is empty.

    Raises [Invalid_argument] if [n < 0]. *)

(** {1:sequential Sequential schemes}

    The first five are perceptually uniform: lightness rises steadily from [0]
    to [1], so that equal steps of data look like equal steps of colour, and
    they read in greyscale. Their tables are their published tables of [256]
    colours. *)

val viridis : t
(** [viridis] runs from dark purple through blue and green to yellow (van der
    Walt, Smith and Firing). *)

val magma : t
(** [magma] runs from black through purple and orange to light yellow. *)

val inferno : t
(** [inferno] runs from black through purple and orange to pale yellow, with
    more contrast at the high end than {!magma}. *)

val plasma : t
(** [plasma] runs from dark blue through magenta to yellow. *)

val cividis : t
(** [cividis] runs from dark blue through grey to yellow, and looks nearly the
    same to viewers with a red or green deficiency (Nuñez, Anderton and
    Renslow). *)

val turbo : t
(** [turbo] is a rainbow from dark blue through cyan, green, yellow and red to
    dark red (Mikhailov), with its published table of [256] colours. Its
    lightness rises then falls, so it shows bands that are not in the data and
    misreads in greyscale; it suits finding fine detail more than reading
    values. Hugin uses it only when asked. *)

(** {2:brewer_sequential Brewer sequential schemes}

    ColorBrewer's sequential schemes, light to dark, with designed tables of 3
    to 9 classes. Their continuous reading is the {!ramp} through their 9-class
    table. *)

val blues : t
(** [blues] is Brewer's Blues. *)

val greens : t
(** [greens] is Brewer's Greens. *)

val greys : t
(** [greys] is Brewer's Greys. *)

val oranges : t
(** [oranges] is Brewer's Oranges. *)

val purples : t
(** [purples] is Brewer's Purples. *)

val reds : t
(** [reds] is Brewer's Reds. *)

val bugn : t
(** [bugn] is Brewer's BuGn, blue to green. *)

val bupu : t
(** [bupu] is Brewer's BuPu, blue to purple. *)

val gnbu : t
(** [gnbu] is Brewer's GnBu, green to blue. *)

val orrd : t
(** [orrd] is Brewer's OrRd, orange to red. *)

val pubu : t
(** [pubu] is Brewer's PuBu, purple to blue. *)

val pubugn : t
(** [pubugn] is Brewer's PuBuGn, purple through blue to green. *)

val purd : t
(** [purd] is Brewer's PuRd, purple to red. *)

val rdpu : t
(** [rdpu] is Brewer's RdPu, red to purple. *)

val ylgn : t
(** [ylgn] is Brewer's YlGn, yellow to green. *)

val ylgnbu : t
(** [ylgnbu] is Brewer's YlGnBu, yellow through green to blue. *)

val ylorbr : t
(** [ylorbr] is Brewer's YlOrBr, yellow through orange to brown. *)

val ylorrd : t
(** [ylorrd] is Brewer's YlOrRd, yellow through orange to red. *)

(** {1:diverging Diverging schemes}

    ColorBrewer's diverging schemes, with designed tables of 3 to 11 classes
    whose middle class, for an odd number of classes, is a light neutral. Their
    continuous reading is the {!ramp} through their 11-class table, which is
    light at [0.5]. A figure puts a data value at the midpoint through its
    scale's domain, such as a domain symmetric about [0.]. *)

val brbg : t
(** [brbg] is Brewer's BrBG, brown to blue-green. *)

val piyg : t
(** [piyg] is Brewer's PiYG, pink to yellow-green. *)

val prgn : t
(** [prgn] is Brewer's PRGn, purple to green. *)

val puor : t
(** [puor] is Brewer's PuOr which, in Brewer's order, runs from orange to
    purple. *)

val rdbu : t
(** [rdbu] is Brewer's RdBu, red to blue. *)

val rdgy : t
(** [rdgy] is Brewer's RdGy, red to grey. *)

val rdylbu : t
(** [rdylbu] is Brewer's RdYlBu, red through yellow to blue. *)

val rdylgn : t
(** [rdylgn] is Brewer's RdYlGn, red through yellow to green. Viewers with a red
    or green deficiency confuse its ends. *)

val spectral : t
(** [spectral] is Brewer's Spectral, red through yellow to blue. *)

(** {1:cyclic Cyclic schemes} *)

val twilight : t
(** [twilight] is the perceptually uniform cyclic scheme of Matplotlib 3.0:
    light at [0] and [1], dark at [0.5], blue in between on one side and red on
    the other. Its table is the published table of [510] colours. *)

(** {1:qualitative Qualitative schemes}

    Palettes for categories. Each is the {!palette} of the colours listed. *)

val okabe_ito : t
(** [okabe_ito] is the eight colours of Okabe and Ito, which stay apart for
    viewers with any common colour vision deficiency: [#000000] black, [#e69f00]
    orange, [#56b4e9] sky blue, [#009e73] bluish green, [#f0e442] yellow,
    [#0072b2] blue, [#d55e00] vermilion and [#cc79a7] reddish purple. The yellow
    is faint on white paper. *)

val tableau10 : t
(** [tableau10] is the ten colours of Tableau 10 by Maureen Stone: [#4e79a7],
    [#f28e2c], [#e15759], [#76b7b2], [#59a14f], [#edc949], [#af7aa1], [#ff9da7],
    [#9c755f] and [#bab0ab]. *)

val accent : t
(** [accent] is Brewer's Accent, eight colours. *)

val dark2 : t
(** [dark2] is Brewer's Dark2, eight colours. *)

val paired : t
(** [paired] is Brewer's Paired, twelve colours in light and dark pairs. *)

val pastel1 : t
(** [pastel1] is Brewer's Pastel1, nine colours. *)

val pastel2 : t
(** [pastel2] is Brewer's Pastel2, eight colours. *)

val set1 : t
(** [set1] is Brewer's Set1, nine colours. *)

val set2 : t
(** [set2] is Brewer's Set2, eight colours. *)

val set3 : t
(** [set3] is Brewer's Set3, twelve colours. *)

(** {1:cvd Colour vision deficiency}

    A viewer with a colour vision deficiency lacks one kind of cone, or has one
    whose response is shifted. {!simulate} gives the colour such a viewer
    perceives, so that a test can check that a scheme's colours stay apart for
    them: that the Oklab distances ({!Color.to_oklab}) between the colours of
    [colors 8 okabe_ito], each mapped by [simulate Deutan], stay above a bound.
*)

(** The type for colour vision deficiencies, by the cone they affect. *)
type deficiency =
  | Protan
      (** Long-wavelength (red) cones: protanomaly, and protanopia at full
          severity. *)
  | Deutan
      (** Medium-wavelength (green) cones: deuteranomaly, and deuteranopia at
          full severity. The most common. *)
  | Tritan
      (** Short-wavelength (blue) cones: tritanomaly, and tritanopia at full
          severity. The rarest. *)

val simulate : ?severity:float -> deficiency -> Color.t -> Color.t
(** [simulate ~severity d c] is [c] as a viewer with deficiency [d] of severity
    [severity] perceives it, by the model of Machado, Oliveira and Fernandes:
    the linear-light sRGB components of [c] times the model's matrix for [d] at
    [severity], each clamped to \[[0];[1]\], then encoded. The authors tabulate
    the matrices at the multiples of [0.1]; between two of them the matrix is
    interpolated linearly. [severity] defaults to [1.], a dichromat, and
    [simulate ~severity:0.] is the identity up to rounding. The alpha of [c] is
    kept.

    Raises [Invalid_argument] if [severity] is not in \[[0];[1]\], [nan]
    included. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal s s'] is [true] iff [s] and [s'] are the same named scheme or are
    made by the same constructor, {!ramp} or {!palette}, from colours that are
    pairwise {!Color.equal}, and both or neither are reversed. Equality is
    structural, as in {{!Hugin_next_gg.section-conventions}[hugin.next.gg]}: a
    ramp through the table of {!viridis} paints as {!viridis} does and is not
    equal to it. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf s] formats [s] for debugging: a named scheme by its name, such as
    [viridis], a made one by its constructor and colours, such as
    [ramp(#000000 #ffffff)], and a reversed one as [reverse(rdbu)]. *)
