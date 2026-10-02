(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Statistics.

    A statistic summarises data into tensors that a figure draws. It computes
    when it is called, with nx operations on the device of its data, and reads
    no value back to the host. *)

(** {1:histograms Histograms} *)

type bins = {
  x : Nx.float64_t;  (** The left edge of each bin. *)
  x2 : Nx.float64_t;  (** The right edge of each bin. *)
  count : Nx.float64_t;  (** The number of values in each bin. *)
  density : Nx.float64_t;
      (** Each count divided by its histogram's total count and by its bin's
          width. *)
}
(** The type for histograms. [x] and [x2] have the shape [[|n|]] for [n] bins,
    and the bin [j] runs from [x.{j}] to [x2.{j}], with [x2.{j} = x.{j+1}].
    [count] and [density] have the shape of the data with its last axis replaced
    by [n]: one histogram per index of the data's other axes, over the bins they
    share.

    A figure draws the bin [j] as the rectangle from [x] to [x2] along x and
    from [0.] to [count] or [density] along y. *)

val histogram : ?bins:int -> ('a, 'b) Nx.t -> bins
(** [histogram ~bins v] counts the values along the last axis of [v] into [bins]
    bins of equal width, one histogram per index of [v]'s other axes. Values are
    converted to [float64] first, as {!Nx.cast} does.

    - The bins span the finite values of all of [v], from the least [lo] to the
      greatest [hi], which are the left edge of the first bin and the right edge
      of the last. If there is no finite value, [lo] is [0.] and [hi] is [1.].
    - So that every bin has a positive width, a span narrower than
      [w = 4. *. float bins *. m *. epsilon_float], with [m] the greater of
      [abs_float lo] and [abs_float hi], widens by the same amount on either
      side to the width [w], shifted to stay within the finite floats. If
      [lo = hi], [w] is at least [1.].
    - A bin holds the values from its left edge included to its right edge
      excluded; the last bin holds its right edge too. Every finite value is in
      one bin, and NaN and infinities are in none.
    - The densities of a histogram, times the widths of their bins, sum to [1.],
      unless a bin is wider than [max_float]: its width is then infinite and its
      density [0.]. A histogram with no finite value has the density [nan] in
      every bin, which a figure drops as
      {{!Hugin_next_kit.section-conventions}missing}.

    [bins] defaults to Sturges' rule from the length [n] of the last axis:
    [1 + ceil (log2 n)], and [1] for [n <= 1].

    Raises [Invalid_argument] if [bins < 1], if [v] is a scalar, or if [v] is
    complex or boolean. *)
