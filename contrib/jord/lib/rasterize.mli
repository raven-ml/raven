(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The cells of a grid that features cover, and how much of each.

    A {e rule} decides whether a feature {e covers} a cell of a grid. A rule's
    result is the {e relation} of every pair (feature, cell) it covers, as two
    int64 tensors of one length, the features' indices and the cells' flat
    indices [row × cols + col], sorted by feature, then by cell. Every pair's
    cell lies in the grid.

    The features' vertices are taken to pixel points by {!Grid.pixel}, rounded
    once there, and each rule is decided on those pixel points in float64
    arithmetic. Edges are straight in the grid's system, which must be the
    features' ({!Polygons.to_crs} moves them there).

    {b Compositions.} A raster of owners holds, for each cell, the last feature
    in order that covers it, or -1. {!Nx.scatter} keeps the last of a target's
    updates, and the relation lists features in order:

    {[
      let feature, cell = Rasterize.touched grid fields in
      let none = Nx.full Nx.int64 [| grid.rows * grid.cols |] (-1L) in
      let owner =
        Nx.reshape [| grid.rows; grid.cols |]
          (Nx.scatter ~axis:0 cell feature none)
    ]}

    Where features overlap, the owner depends on their order. A cell that one
    feature covers is that feature's in every order, and a cell that several
    cover is the last of them in the order given. The number of features
    covering each cell, which shows where the order decides, is

    {[
    let count =
      Nx.scatter ~combine:Add ~axis:0 cell (Nx.ones_like feature)
        (Nx.zeros Nx.int64 [| grid.rows * grid.cols |])
    ]}

    A label raster is [Nx.take labels owner], which reads 0 where [owner] is -1.

    A zonal mean of an image [x] whose cells are [valid], both flattened to
    [[rows · cols]], weighs each cell by {!areas}' share:

    {[
      let feature, cell, area = Rasterize.areas grid parcels in
      let n = Polygons.count parcels in
      let mean x valid =
        let ok = Nx.take ~axis:0 cell valid in
        let zero = Nx.zeros_like area in
        let w = Nx.where ok area zero in
        let wx = Nx.where ok (Nx.mul area (Nx.take ~axis:0 cell x)) zero in
        let sum v =
          Nx.scatter ~combine:Add ~axis:0 feature v
            (Nx.zeros Nx.float64 [| n |])
        in
        Nx.div (sum wx) (sum w)
    ]}

    A cell that is not valid is selected away by [Nx.where], since a NaN
    times 0 is NaN; a feature with no valid cell has the mean 0 / 0, NaN.
    The relation is made once per grid and features, and the sums run per
    image. *)

(** {1:rules Rules}

    Each rule raises [Invalid_argument] if the features' system is not the
    grid's. *)

val centres : Grid.t -> Polygons.t -> Nx.host Nx.int64_t * Nx.host Nx.int64_t
(** [centres g p] is the relation of the rule by which a feature covers a
    cell when the cell's centre lies in the feature's region. A centre on the
    region's boundary is decided as if moved by an infinitesimal step toward
    the previous column and by a step infinitesimal against that one toward the
    next row. Each edge's crossing of a row of centres is computed from its
    endpoints in order of row, so an edge that two features share (the same two
    endpoints) crosses each row at the same number in both, and features that
    tile the plane edge to edge cover each centre exactly once. *)

val touched : Grid.t -> Polygons.t -> Nx.host Nx.int64_t * Nx.host Nx.int64_t
(** [touched g p] is the relation of the rule by which a feature covers a
    cell when {!centres} gives it the cell, or when one of its edges meets
    the cell's square shrunk by τ = 1e-6 pixel on each side ({!section-tau}),
    the open square (i + τ, i + 1 - τ) × (j + τ, j + 1 - τ) of cell [(i, j)].
    It contains {!centres}'s relation.

    For a region that is the closure of its interior, as a valid polygon's is
    (OGC Simple Features 1.2.1, §6.1.11.1, assertion d), these are the cells
    whose square the region overlaps with positive area, except overlaps no
    thicker than τ. A region that only shares a border or a corner with a cell
    does not cover it. *)

(** {2:tau The margin τ}

    τ exceeds the error of a float64 round trip of a vertex through another
    system on grids of a centimetre and coarser, so a polygon traced along
    cell borders covers only its own cells after one. Coarser errors, such
    as those of coordinates rounded to six decimals of a degree, exceed
    it. *)

(** {1:shares Shares}

    {!areas} gives the share of each cell that a feature covers, within a
    bound of the exact share ({!section-share_bound}). *)

val areas :
  Grid.t ->
  Polygons.t ->
  Nx.host Nx.int64_t * Nx.host Nx.int64_t * Nx.host Nx.float64_t
(** [areas g p] is {!touched}'s relation and, for each of its pairs, the share
    of the cell's square that the feature's region covers, in \[0, 1\], the
    weights of an area-weighted mean over each feature. A cell that no edge of
    the feature meets has exactly 1. The share is that of the feature's
    region, rings that cross included.

    Since the relation is {!touched}'s, a feature that only shares a border
    or a corner with a cell has no pair for it, an overlap thinner than τ is
    left out, and the areas of a feature within the grid sum to its area in
    pixels to within τ times its perimeter in pixels and
    {{!section-share_bound}the bound on a share}.

    Raises [Invalid_argument] if the features' system is not the grid's. *)

(** {2:share_bound The bound on a share}

    To first order each share {!areas} gives is within

    {[
      (24M + K + 75H) u W + 10 u X
    ]}

    of the exact share of the region of the feature's pixel points, where u =
    2{^-53}; W is the sum of the widths the feature's edges span within the
    cell's column; K the number of the {e runs} summed for the cell, the
    spans of abscissae over which one edge bounds the region from one side;
    M ≥ 1 the largest magnitude of the coordinates, from the cell's corner,
    of the edges that meet the cell's row; H ≥ 1 the rows from the top of the
    cell holding the feature's least vertex row down to its greatest; and X
    the number of crossings of two of its edges within the column. *)
