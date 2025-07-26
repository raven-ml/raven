(** Vectorizing map (vmap) for Rune tensors.

    vmap creates a function which maps a computation over additional axes. It is
    useful for expressing batched computations without explicit loops. *)

open Nx_rune

(** Type to represent mapping specification for a single axis *)
type axis_spec =
  | Map of int  (** Map over this axis index *)
  | NoMap  (** Don't map this axis *)

(** Type to represent container mapping specifications *)
type 'a in_axes_spec = Single of axis_spec | Container of 'a

(** Type to represent output axes specification *)
type 'a out_axes_spec = OutSingle of int option | OutContainer of 'a

val vmap :
  ?in_axes:'a in_axes_spec ->
  ?out_axes:'b out_axes_spec ->
  ?axis_name:string ->
  ?axis_size:int ->
  (('c, 'd) t -> ('e, 'f) t) ->
  ('c, 'd) t ->
  ('e, 'f) t
(** [vmap ?in_axes ?out_axes ?axis_name ?axis_size f] creates a vectorized
    version of function [f].

    @param in_axes
      Specifies which input array axes to map over. Default: Single (Map 0) -
      maps over the first axis.
    @param out_axes
      Specifies where the mapped axis should appear in output. Default:
      OutSingle (Some 0) - mapped axis at position 0. Use None to not include
      mapped axis in output.
    @param axis_name
      Optional name for the mapped axis (for collective operations).
    @param axis_size
      Optional size of the mapped axis. Required when in_axes is NoMap.
    @param f The function to be mapped.

    Examples:
    {[
      (* Simple batched matrix multiplication *)
      let batched_matmul = vmap (fun x -> matmul x w)

      (* Map over second axis *)
      let f = vmap ~in_axes:(Single (Map 1)) fn

      (* Don't include batch dimension in output *)
      let g = vmap ~out_axes:(OutSingle None) sum
    ]} *)

val vmaps :
  ?in_axes:axis_spec list ->
  ?out_axes:'b out_axes_spec ->
  ?axis_name:string ->
  ?axis_size:int ->
  (('c, 'd) t list -> ('e, 'f) t) ->
  ('c, 'd) t list ->
  ('e, 'f) t
(** [vmaps ?in_axes ?out_axes ?axis_name ?axis_size f] creates a vectorized
    version of function [f] that takes multiple tensor arguments.

    @param in_axes
      List of axis specifications for each input tensor. Default: all Map 0.
    @param out_axes
      Specifies where the mapped axis should appear in output. Default:
      OutSingle (Some 0) - mapped axis at position 0.
    @param axis_name
      Optional name for the mapped axis (for collective operations).
    @param axis_size
      Optional size of the mapped axis. Required when any in_axes is NoMap.
    @param f The function to be mapped.

    Examples:
    {[
      (* Batched function with two inputs *)
      let batched_add = vmaps (fun [ x; y ] -> add x y)

      (* Map over different axes for different inputs *)
      let f = vmaps ~in_axes:[ Map 0; Map 1 ] fn
    ]} *)
