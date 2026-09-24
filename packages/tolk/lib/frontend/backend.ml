(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop

(* One process-wide default device backs every realization. This mirrors
   tinygrad, whose [Tensor.realize] resolves the device internally rather than
   taking it as an argument. Backend openers are installed in the shared
   device registry so scheduled graphs can name any device instance; the
   default is chosen the same way tinygrad selects [Device.DEFAULT]: the
   first [DEV] target picks a backend, otherwise backends are
   scanned in priority order and the first one that opens wins, falling back
   to CPU. *)
let all_backends : (string * (string -> Tolk.Device.t)) list =
  (match Device_metal.opener with
  | Some create -> [ ("METAL", create) ]
  | None -> [])
  @ [
      ("AMD", Tolk_amd.create);
      ("NV", Tolk_nv.create);
      ("CUDA", Tolk_cuda.create);
      ("CPU", fun name -> Tolk_cpu.create name);
    ]

let () =
  List.iter
    (fun (prefix, create) -> Tolk.Device.register prefix create)
    all_backends

let default_device =
  lazy
    (Tolk.Helpers.select_first_inited ~message:"no usable devices"
       (List.map (fun (name, _) () -> Tolk.Device.get name) all_backends))

let device () =
  match Tolk.Helpers.Context_var.get Tolk.Helpers.dev with
  | target :: _ when target.Tolk_uop.Target.device <> "" ->
      Tolk.Device.get target.device
  | _ -> Lazy.force default_device
let device_name () = Tolk.Device.name (device ())

