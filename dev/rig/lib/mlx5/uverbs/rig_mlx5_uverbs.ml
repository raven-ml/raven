(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let ( let* ) = Result.bind
let mlx5_driver = "mlx5_core"
let is_mlx5 root name = Path.driver_name root name = Some mlx5_driver
let names ?(root = "/") () = List.filter (is_mlx5 root) (Path.devices root)

let open_ ?(root = "/") name =
  let* () =
    if Path.exists root name then Ok ()
    else Error (strf "%s: no RDMA device of this name" name)
  in
  let* () =
    if is_mlx5 root name then Ok ()
    else Error (strf "%s: the %s driver does not hold it" name mlx5_driver)
  in
  let* path = Path.open_ ~driver:Defs.rdma_driver_mlx5 ~root name in
  Rig_mlx5.make path
