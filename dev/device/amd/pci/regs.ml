(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs
module Register = Device_amd_abi.Register
module Window = Device_pci.Window

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

(* Layouts *)

type layout = {
  d : Discovery.t;
  gpu : Device_amd_abi.Gpu.t;
  regs : (string, Register.t * int) Hashtbl.t; (* by name: register, block *)
  guarded : (int * int) list;
}

let dotted (a, b, c) = strf "%d.%d.%d" a b c

(* The table of [prefix] for a block of version [v]: the latest at or before [v]
   of its major. *)
let family prefix ((major, _, _) as v) =
  List.fold_left
    (fun best (p, ((m, _, _) as v'), regs) ->
      if p <> prefix || m <> major || compare v' v > 0 then best
      else
        match best with
        | Some (v'', _) when compare v'' v' >= 0 -> best
        | _ -> Some (v', regs))
    None D.registers
  |> Option.map snd

(* MP1's message registers are named in MP 11.0's table, at MP1's bases. *)
let mp1_messages = (11, 0, 0)

(* GC 9.4.3 runs the code objects of gfx942. *)
let target = function 9, 4, 3 -> (9, 4, 2) | gc -> gc

let layout d =
  let block b =
    match Discovery.version d b with
    | Some v -> Ok v
    | None -> Error (strf "the GPU has no %s block" (Discovery.name b))
  in
  let* gc = block D.gc_hwid in
  let* sdma = block D.sdma0_hwid in
  let* () = Result.map ignore (block D.mp1_hwid) in
  let unbooted b v =
    Error
      (strf "%s %s is a version this library does not boot" (Discovery.name b)
         (dotted v))
  in
  let table prefix b =
    let* v = block b in
    match family prefix v with
    | Some regs -> Ok (b, regs)
    | None -> unbooted b v
  in
  let gpu =
    let g = d.gc in
    {
      Device_amd_abi.Gpu.target = target gc;
      gc;
      sdma;
      xccs = max 1 (List.length (Discovery.live d D.gc_hwid));
      shader_engines = g.engines;
      compute_units = g.engines * g.arrays * g.units;
      scratch_slots = g.scratch_slots;
    }
  in
  let gc_regs = Register.registers gpu in
  let* () = if gc_regs = [] then unbooted D.gc_hwid gc else Ok () in
  let* mp = table "mp" D.mp0_hwid in
  let* hdp = table "hdp" D.hdp_hwid in
  let* mmhub = table "mmhub" D.mmhub_hwid in
  let* osssys = table "osssys" D.osssys_hwid in
  let* nbio = table (if gc < (12, 0, 0) then "nbio" else "nbif") D.nbif_hwid in
  (* SDMA 4's engines have registers of their own; later ones are GC's. *)
  let* sdma_regs =
    match sdma with
    | 4, _, _ -> Result.map (fun t -> [ t ]) (table "sdma" D.sdma0_hwid)
    | _ -> Ok []
  in
  let mp1 = (D.mp1_hwid, Option.get (family "mp" mp1_messages)) in
  let regs = Hashtbl.create 1024 in
  (* A later table's register of a name replaces an earlier's: MP1's view of the
     message registers is MP0's last. *)
  List.iter
    (fun (b, rs) ->
      List.iter (fun (r : Register.t) -> Hashtbl.replace regs r.name (r, b)) rs)
    ([ mp; hdp; (D.gc_hwid, gc_regs); mmhub; osssys; nbio ]
    @ sdma_regs @ [ mp1 ]);
  (* A virtual function reaches through the RLC the GC registers from each
     segment's base to the last one programmed in it. *)
  let last =
    List.fold_left
      (fun acc (r : Register.t) ->
        let top = Option.value ~default:0 (List.assoc_opt r.segment acc) in
        (r.segment, max top r.offset) :: List.remove_assoc r.segment acc)
      [] gc_regs
  in
  let guarded =
    List.concat_map
      (fun (_, segs) ->
        List.filter_map
          (fun (s, top) ->
            if s < Array.length segs then Some (segs.(s), segs.(s) + top)
            else None)
          last)
      (Discovery.live d D.gc_hwid)
    |> List.sort compare
  in
  Ok { d; gpu; regs; guarded }

let gpu l = l.gpu
let discovery l = l.d
let version l b = Option.get (Discovery.version l.d b)
let has l name = Hashtbl.mem l.regs name
let guarded l = l.guarded

let find l name =
  match Hashtbl.find_opt l.regs name with
  | Some r -> r
  | None ->
      invalid_argf "Device_amd_pci.open_: GC %s has no register %s"
        (dotted l.gpu.gc) name

let register l name = fst (find l name)

let address ?(inst = 0) l name =
  let r, b = find l name in
  let insts = Option.value ~default:[] (List.assoc_opt b l.d.bases) in
  match List.assoc_opt inst insts with
  | Some segs when r.segment < Array.length segs -> segs.(r.segment) + r.offset
  | _ ->
      invalid_argf "Device_amd_pci.open_: %s has no instance %d with segment %d"
        name inst r.segment

(* Access *)

type t = {
  fn : Device_pci.Function.t;
  mmio : Window.t;
  layout : layout;
  vf : bool;
}

exception Stuck of string

let make fn mmio layout ~vf = { fn; mmio; layout; vf }
let fn r = r.fn
let layout_of r = r.layout
let vf r = r.vf
let machine r = Device_pci.Function.machine r.fn
let default_ms = 10_000

let wait ?(ms = default_ms) r what f =
  if not (Device_pci.Machine.wait (machine r) ~ms f) then
    match Device_pci.Function.failed r.fn with
    | Some why -> raise (Stuck (strf "%s: %s" what why))
    | None -> raise (Stuck (strf "%s did not answer in %d ms" what ms))

let pause r ms =
  ignore (Device_pci.Machine.wait (machine r) ~ms (fun () -> false))

let words r = Window.length r.mmio / 4
let lo32 v = v land 0xffff_ffff
let hi32 v = (v lsr 32) land 0xffff_ffff

let in_guarded r a =
  r.vf && List.exists (fun (lo, hi) -> lo <= a && a <= hi) r.layout.guarded

(* The registers of the RLC's gateway, and of the RSMU's window past the end of
   the register BAR. *)
let rsmu_index = "regBIF_BX_PF0_RSMU_INDEX"
let rsmu_data = "regBIF_BX_PF0_RSMU_DATA"

(* A gateway command's flag for a read, above the register's address. *)
let rlcg_read = 1 lsl 28

(* The gateway clears the low 20 bits of its second scratch register once it has
   performed the access. *)
let rlcg_pending = 0xfffff

let rec get_at r ~direct a =
  if (not direct) && in_guarded r a then rlcg r a 0 ~read:true
  else if a >= words r then begin
    set_at r ~direct:true (address r.layout rsmu_index) (a * 4);
    get_at r ~direct:true (address r.layout rsmu_data)
  end
  else Window.get32 r.mmio (a * 4)

and set_at r ~direct a v =
  if (not direct) && in_guarded r a then ignore (rlcg r a v ~read:false)
  else if a >= words r then begin
    set_at r ~direct:true (address r.layout rsmu_index) (a * 4);
    set_at r ~direct:true (address r.layout rsmu_data) v
  end
  else Window.set32 r.mmio (a * 4) v

(* A virtual function writes a guarded register's address and value into scratch
   registers and has the RLC perform the access; GRBM's selection registers it
   sets through two scratch registers of their own. *)
and rlcg r a v ~read =
  let at name = address r.layout name in
  let direct name v = set_at r ~direct:true (at name) v in
  if a = at "regGRBM_GFX_CNTL" then (
    direct "regSCRATCH_REG2" v;
    v)
  else if a = at "regGRBM_GFX_INDEX" then (
    direct "regSCRATCH_REG3" v;
    v)
  else begin
    let cmd = ((a lor if read then rlcg_read else 0) lsl 32) lor v in
    direct "regSCRATCH_REG0" (lo32 cmd);
    direct "regSCRATCH_REG1" (hi32 cmd);
    direct "regRLC_SPARE_INT" 1;
    wait r (strf "the RLC gateway on register 0x%x" a) (fun () ->
        get_at r ~direct:true (at "regSCRATCH_REG1") land rlcg_pending = 0);
    get_at r ~direct:true (at "regSCRATCH_REG0")
  end

let get r a = get_at r ~direct:false a
let set r a v = set_at r ~direct:false a v
let read ?inst r name = get r (address ?inst r.layout name)

let write ?inst ?(value = 0) r name fs =
  set r
    (address ?inst r.layout name)
    (value lor Register.encode (register r.layout name) fs)

let mask (reg : Register.t) names =
  List.fold_left
    (fun acc f ->
      match List.assoc_opt f reg.fields with
      | Some (lo, hi) -> acc lor (((1 lsl (hi - lo + 1)) - 1) lsl lo)
      | None ->
          invalid_argf "Device_amd_pci.open_: %s has no field %s" reg.name f)
    0 names

let update ?inst r name fs =
  let old = read ?inst r name in
  let keep = old land lnot (mask (register r.layout name) (List.map fst fs)) in
  write ?inst ~value:keep r name fs

let fields ?inst r name =
  let v = read ?inst r name in
  List.map
    (fun (f, (lo, hi)) -> (f, (v lsr lo) land ((1 lsl (hi - lo + 1)) - 1)))
    (register r.layout name).fields

let field ?inst r name f =
  match List.assoc_opt f (fields ?inst r name) with
  | Some v -> v
  | None -> invalid_argf "Device_amd_pci.open_: %s has no field %s" name f

let write64 ?inst r base ~lo ~hi x =
  write ?inst ~value:(lo32 x) r (base ^ lo) [];
  write ?inst ~value:(hi32 x) r (base ^ hi) []

(* The indirect window names a die's PCIe register by its byte address, the die
   in bits 32-33 and a flag in bit 34. *)
let set_pcie ?(aid = 0) r a v =
  let a =
    a * 4 lor if aid > 0 then ((aid land 3) lsl 32) lor (1 lsl 34) else 0
  in
  write ~value:(lo32 a) r "regBIF_BX0_PCIE_INDEX2" [];
  if hi32 a > 0 then
    write ~value:(hi32 a land 0xff) r "regBIF_BX0_PCIE_INDEX2_HI" [];
  write ~value:v r "regBIF_BX0_PCIE_DATA2" [];
  if hi32 a > 0 then write ~value:0 r "regBIF_BX0_PCIE_INDEX2_HI" []
