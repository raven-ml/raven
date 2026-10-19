(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GC registers by name, at their PM4 addresses: the offset in the GC's headers
   from its segment's base in vega20_ip_offset.h (GFX9) or
   sienna_cichlid_ip_offset.h (GFX10 on). Fields as the headers' masks lay them
   out. *)

open Windtrap
open Rig_amd_abi
module S = Rig_amd_abi_support

let strf = Printf.sprintf
let timeout = S.timeout
let gpu = S.gpu
let version = S.version

let names g = List.map (fun (r : Register.t) -> r.name) (Register.registers g)

let registers =
  group ~timeout "registers"
    [
      cases
        ~name:(fun (v, _) -> version v)
        "a GC takes the registers of the latest version of its major before it"
        [
          ((9, 4, 4), (9, 4, 3));
          ((11, 0, 2), (11, 0, 0));
          ((11, 0, 3), (11, 0, 3));
          ((11, 0, 4), (11, 0, 3));
          ((11, 5, 3), (11, 5, 0));
          ((12, 0, 1), (12, 0, 0));
        ]
        (fun (v, family) ->
          equal (list string) (names (gpu family)) (names (gpu v)));
      cases ~name:version "a GC with no version of its major before it has none"
        [ (10, 3, 0); (9, 4, 2); (8, 0, 0); (13, 0, 0) ]
        (fun v -> equal (list string) [] (names (gpu v)));
      cases
        ~name:(fun n -> strf "%S" n)
        "a name of no register is none"
        [ ""; "GRBM_GFX_INDEX"; "reggrbm_gfx_index"; "regGRBM_GFX_INDEX " ]
        (fun name -> is_none (Register.find (gpu (11, 0, 0)) name));
    ]

let no_base v segment =
  let r = { Register.name = "regNONE"; offset = 0; segment; fields = [] } in
  raises_match (Exn.invalid_arg ~substring:"Register.address") (fun () ->
      Register.address (gpu v) r)

let address =
  group ~timeout "address"
    [
      cases
        ~name:(fun ((v, n), _) -> strf "%s of %s" n (version v))
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
          equal int addr
            (Register.address g (require_some (Register.find g name))));
      cases
        ~name:(fun (v, s) -> strf "segment %d of %s" s (version v))
        "a segment with no base is refused"
        [ ((9, 4, 3), 2); ((11, 0, 0), 4); ((12, 0, 0), 100) ]
        (fun (v, segment) -> no_base v segment);
      test "a negative segment is refused" (fun () -> no_base (9, 4, 3) (-1));
    ]

(* Values at the edges of a field and of the integers. *)
let values = [ 0; 1; -1; 0xff; 0x1_0000_0000; 0xffff_ffff; max_int; min_int ]

let single (r : Register.t) f v =
  let lo, hi = List.assoc f r.fields in
  (v land ((1 lsl (hi - lo + 1)) - 1)) lsl lo

(* GRBM_GFX_INDEX's fields start at bit 0, end at bit 31 and include 1-bit
   ones; COMPUTE_PGM_LO's one field is the whole word. *)
let encoded = [ "regGRBM_GFX_INDEX"; "regCOMPUTE_PGM_LO" ]
let gfx11 name = require_some (Register.find (gpu (11, 0, 0)) name)

let encode =
  let r = lazy (gfx11 "regGRBM_GFX_INDEX") in
  group ~timeout "encode"
    [
      cases ~name:Fun.id "a field's value is cut to its width, other bits zero"
        encoded (fun name ->
          let r = gfx11 name in
          List.iter
            (fun (f, _) ->
              List.iter
                (fun n ->
                  equal ~msg:(strf "%s = %d" f n) int (single r f n)
                    (Register.encode r [ (f, n) ]))
                values)
            r.fields);
      cases ~name:Fun.id "fields together are each field's bits" encoded
        (fun name ->
          let r = gfx11 name in
          let fs =
            List.mapi (fun i (f, _) -> (f, i * 0x9e3779b9 lxor -1)) r.fields
          in
          equal int
            (List.fold_left (fun w (f, n) -> w lor single r f n) 0 fs)
            (Register.encode r fs));
      test "a GRBM_GFX_INDEX word" (fun () ->
          equal int
            ((0xff lsl 16) lor (1 lsl 31))
            (Register.encode (Lazy.force r)
               [ ("se_index", 0x1ff); ("se_broadcast_writes", 1) ]));
      test "no field is word zero" (fun () ->
          equal int 0 (Register.encode (Lazy.force r) []));
      test "a field the register lacks is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Register.encode") (fun () ->
              Register.encode (Lazy.force r) [ ("nope", 1) ]));
    ]

let () = exit (run "rig_amd_abi.register" [ registers; address; encode ])
