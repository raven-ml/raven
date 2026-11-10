(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The cells of a grid that features cover.

    A {e rule} decides whether a feature {e covers} a cell of a grid. A rule's
    result is the {e relation} of every pair (feature, cell) it covers, as two
    int64 tensors of one length: the features' indices, and the cells' flat
    indices [row × cols + col], sorted by feature, then by cell. A cell outside
    the grid is in no pair.

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

    Each rule raises [Invalid_argument] if the features' system is not the
    grid's. *)

val centres : Grid.t -> Polygons.t -> Nx.host Nx.int64_t * Nx.host Nx.int64_t
(** [centres g p] is the relation of the rule: a feature covers a cell when the
    cell's centre lies in the feature's region. A centre on the region's
    boundary is decided as if moved by an infinitesimal step toward the previous
    column and by a step infinitesimal against that one toward the next row.
    Each edge's crossing of a row of centres is computed from its endpoints in
    order of row, so an edge that two features share (the same two endpoints)
    crosses each row at the same number in both, and features that tile the
    plane edge to edge cover each centre exactly once. *)

val touched : Grid.t -> Polygons.t -> Nx.host Nx.int64_t * Nx.host Nx.int64_t
(** [touched g p] is the relation of the rule: a feature covers a cell when
    {!centres} gives it the cell, or when one of its edges meets the cell's
    square shrunk by τ = 1e-6 pixel on each side, the open square (i + τ, i + 1
    \- τ) × (j + τ, j + 1 - τ) of cell [(i, j)]. It contains {!centres}'s
    relation.

    For a region that is the closure of its interior, as a valid polygon's is
    (OGC Simple Features 1.2.1, §6.1.11.1, assertion d), these are the cells
    whose square the region overlaps with positive area, except overlaps no
    thicker than τ. A region that only shares a border or a corner with a cell
    does not cover it.

    τ lies above the error a vertex takes from a float64 round trip through
    another system, a few ulps of a coordinate in metres (an ulp of 10⁷ m is
    under 2e-9 m), on grids of a centimetre and coarser, so a polygon traced
    along cell borders still covers only its own cells after that trip, which a
    GeoJSON file's polygons take through CRS84. It lies far below any geometry,
    10 µm on a 10 m grid. It does not cover coarser errors, such as those of
    coordinates rounded to six decimals of a degree. *)
