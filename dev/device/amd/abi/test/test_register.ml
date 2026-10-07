(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GC registers by name, at their PM4 addresses: the offset in the GC's headers
   from its segment's base in vega20_ip_offset.h (GFX9) or
   sienna_cichlid_ip_offset.h (GFX10 on). Fields as the headers' masks lay them
   out. *)

open Windtrap
open Device_amd_abi

let gpu gc =
  {
    Gpu.target = gc;
    gc;
    sdma = (6, 0, 0);
    xccs = 1;
    shader_engines = 4;
    compute_units = 32;
    scratch_slots = 32;
  }

let version (a, b, c) = Printf.sprintf "%d.%d.%d" a b c

let registers =
  group "registers"
    [
      cases
        ~name:(fun (v, _) -> version v)
        "a GC takes the registers of the latest version of its major before it"
        [
          ((9, 4, 4), (9, 4, 3));
          ((11, 0, 2), (11, 0, 0));
          ((11, 0, 3), (11, 0, 3));
          ((12, 0, 1), (12, 0, 0));
        ]
        (fun (v, family) ->
          equal (list string)
            (List.map
               (fun (r : Register.t) -> r.name)
               (Register.registers (gpu family)))
            (List.map
               (fun (r : Register.t) -> r.name)
               (Register.registers (gpu v))));
      cases ~name:version "a GC with no version of its major has none"
        [ (10, 3, 0); (9, 4, 2) ]
        (fun v -> equal int 0 (List.length (Register.registers (gpu v))));
    ]

let address =
  cases
    ~name:(fun ((v, n), _) -> Printf.sprintf "%s of %s" n (version v))
    "a register's address"
    [
      (((11, 0, 0), "regCOMPUTE_PGM_LO"), 0x2e0c);
      (((12, 0, 0), "regCOMPUTE_PGM_LO"), 0x2e0c);
      (((9, 4, 3), "regCOMPUTE_PGM_LO"), 0x2e0c);
      (((11, 0, 0), "regCOMPUTE_PGM_RSRC3"), 0x2e28);
      (((9, 4, 3), "regCOMPUTE_PGM_RSRC3"), 0x2e2d);
      (((9, 4, 3), "regGRBM_GFX_INDEX"), 0xc200);
      (((11, 5, 0), "regGRBM_GFX_INDEX"), 0xc200);
    ]
    (fun ((v, name), addr) ->
      let g = gpu v in
      equal int addr (Register.address g (require_some (Register.find g name))))

let encode =
  let r =
    lazy (require_some (Register.find (gpu (11, 0, 0)) "regGRBM_GFX_INDEX"))
  in
  group "encode"
    [
      test "a field's value is cut to its width" (fun () ->
          equal int
            ((0xff lsl 16) lor (1 lsl 31))
            (Register.encode (Lazy.force r)
               [ ("se_index", 0x1ff); ("se_broadcast_writes", 1) ]));
      test "a field the register lacks is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Register.encode") (fun () ->
              Register.encode (Lazy.force r) [ ("nope", 1) ]));
    ]

let () = exit (run "device_amd_abi.register" [ registers; address; encode ])
