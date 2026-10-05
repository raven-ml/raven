let open_device =
  Some
    (fun () ->
      match Nx_metal_device.get 0 with Ok d -> d | Error e -> failwith e)
