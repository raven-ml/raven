module Make (B : Backend_intf.S) = struct
  type ('a, 'b) t = ('a, 'b) B.t = {
    dtype : ('a, 'b) Dtype.dtype;
    buffer : ('a, 'b) B.buffer;
    view : View.view;
  }

  let create_context () = B.create_context ()

  (* *)
  let shape t = View.shape t.view
  let size t = View.size t.view
  let dtype t = t.dtype

  (* *)

  (* Step‑by‑step broadcast of a single tensor to [target_shape]. *)
  let broadcast_to (ctx : B.context) (x : ('a, 'b) t) (target_shape : int array)
      : ('a, 'b) t =
    (* Sanity – cannot down‑rank. *)
    let rank_src = Array.length x.view.shape
    and rank_dst = Array.length target_shape in
    if rank_dst < rank_src then
      invalid_arg "broadcast_to: cannot broadcast to lower rank";

    (* Left‑pad with 1‑dims so ranks match. *)
    let reshaped =
      if rank_dst = rank_src then x
      else
        let pad = Array.make (rank_dst - rank_src) 1 in
        let new_shape = Array.append pad x.view.shape in
        let new_view = View.reshape x.view new_shape in
        let reshaped = { x with view = new_view } in
        B.reshape ctx reshaped new_shape
    in

    (* Insert stride‑0 dims where the source has size 1. *)
    if Array.for_all2 ( = ) reshaped.view.shape target_shape then reshaped
    else
      let new_view = View.expand reshaped.view target_shape in
      let reshaped = { reshaped with view = new_view } in
      B.expand ctx reshaped target_shape

  (* Broadcast two tensors to a common shape – tinygrad._broadcasted *)
  let broadcasted (ctx : B.context) (a : ('a, 'b) t) (b : ('a, 'b) t) :
      ('a, 'b) t * ('a, 'b) t =
    let out_shape = View.broadcast_shapes a.view.shape b.view.shape in
    (broadcast_to ctx a out_shape, broadcast_to ctx b out_shape)

  let reshape (ctx : B.context) (x : ('a, 'b) t) (shape : int array) :
      ('a, 'b) t =
    let new_view = View.reshape x.view shape in
    let reshaped = { x with view = new_view } in
    B.reshape ctx reshaped shape

  let add ctx (a : ('a, 'b) B.t) (b : ('a, 'b) B.t) : ('a, 'b) B.t =
    let a', b' = broadcasted ctx a b in
    let out_size = size a' in
    let out = B.buffer ctx (dtype a) out_size in
    let out' = reshape ctx out (shape a') in
    B.add ctx a' b' out';
    out
end
