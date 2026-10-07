(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* Sizes and addresses *)

let kib = 1024
let mib = 1 lsl 20
let gib = 1 lsl 30
let round_up n a = (n + a - 1) / a * a
let pp_hex ppf x = Format.fprintf ppf "0x%x" x

let hex =
  Testable.with_compare Int.compare (Testable.make ~pp:pp_hex ~equal:Int.equal)

let on_linux = Sys.file_exists "/sys/bus/pci/devices"

external now_ns : unit -> int = "device_pci_test_now_ns"

(* Process memory and far machines *)

external memory : int -> int = "device_pci_test_memory"
external far : int -> int -> int = "device_pci_test_far"
external break : int -> unit = "device_pci_test_far_break"
external hold : int -> unit = "device_pci_test_far_hold"
external waiting : int -> bool = "device_pci_test_far_waiting"
external let_go : int -> unit = "device_pci_test_far_let_go"
external log : int -> (bool * int * int) list = "device_pci_test_far_log"
