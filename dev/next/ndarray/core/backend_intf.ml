module type S = sig
  type ('a, 'b) buffer

  type ('a, 'b) t = {
    dtype : ('a, 'b) Dtype.dtype;
    buffer : ('a, 'b) buffer;
    view : View.view;
  }

  type context

  val create_context : unit -> context

  (* These are the equivalent of tinygrad's UOp Do not add anything here that's
     not a UOp in tinygrad. *)

  val const :
    context ->
    int array ->
    ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
    ('a, 'b) t

  val buffer : context -> ('a, 'b) Dtype.dtype -> int -> ('a, 'b) t
  val reshape : context -> ('a, 'b) t -> int array -> ('a, 'b) t
  val expand : context -> ('a, 'b) t -> int array -> ('a, 'b) t
  val add : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
end
