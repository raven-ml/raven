open Internal

let eval_handler : ('a, 'a) Effect.Deep.handler =
  let open Effect.Deep in
  let handle_ap2 op x y k =
    let data = op x y in
    continue k (create_internal data)
  in

  let effc : type a. a Effect.t -> ((a, _) continuation -> _) option = function
    | Const v -> Some (fun k -> continue k (create_internal v))
    | Neg x ->
        Some (fun k -> continue k (create_internal (Dispatch.neg x.data)))
    | Exp x ->
        Some (fun k -> continue k (create_internal (Dispatch.exp x.data)))
    | Log x ->
        Some (fun k -> continue k (create_internal (Dispatch.log x.data)))
    | Abs x ->
        Some (fun k -> continue k (create_internal (Dispatch.abs x.data)))
    | Sin x ->
        Some (fun k -> continue k (create_internal (Dispatch.sin x.data)))
    | Cos x ->
        Some (fun k -> continue k (create_internal (Dispatch.cos x.data)))
    | Add (x, y) -> Some (handle_ap2 Dispatch.add x.data y.data)
    | Sub (x, y) -> Some (handle_ap2 Dispatch.sub x.data y.data)
    | Mul (x, y) -> Some (handle_ap2 Dispatch.mul x.data y.data)
    | Div (x, y) -> Some (handle_ap2 Dispatch.div x.data y.data)
    | Maximum (x, y) -> Some (handle_ap2 Dispatch.maximum x.data y.data)
    | Minimum (x, y) -> Some (handle_ap2 Dispatch.minimum x.data y.data)
    | Matmul (x, y) -> Some (handle_ap2 Dispatch.matmul x.data y.data)
    | Transpose x ->
        Some (fun k -> continue k (create_internal (Dispatch.transpose x.data)))
    | Sum (x, axes, keepdims) ->
        Some
          (fun k ->
            continue k (create_internal (Dispatch.sum ?axes ?keepdims x.data)))
    | Mean (x, axes, keepdims) ->
        Some
          (fun k ->
            continue k (create_internal (Dispatch.mean ?axes ?keepdims x.data)))
    | Max (x, axes, keepdims) ->
        Some
          (fun k ->
            continue k (create_internal (Dispatch.max ?axes ?keepdims x.data)))
    | Min (x, axes, keepdims) ->
        Some
          (fun k ->
            continue k (create_internal (Dispatch.min ?axes ?keepdims x.data)))
    | Reshape (x, shape) ->
        Some
          (fun k ->
            continue k (create_internal (Dispatch.reshape shape x.data)))
    | Slice (x, starts, stops, steps) ->
        Some
          (fun k ->
            continue k
              (create_internal (Dispatch.slice ?steps starts stops x.data)))
    | Cast (dtype, x) ->
        Some
          (fun k -> continue k (create_internal (Dispatch.astype dtype x.data)))
    | Move (device, x) ->
        Some
          (fun k -> continue k (create_internal (Dispatch.move device x.data)))
    | _ -> None
  in
  { retc = (fun x -> x); exnc = raise; effc }

let eval (type a b) (f : a -> b) (x : a) : b =
  Effect.Deep.match_with f x eval_handler
