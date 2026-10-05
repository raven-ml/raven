(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's kernels on AMD GPUs, from the code objects the library carries.

   One path runs every kernel: an operation names its module's key, its operands
   and its result; the operands' views are coalesced; the module's contiguous
   form [c] runs when every view is C-contiguous after merging, and its strided
   form [s] otherwise; one launch on the device's compute queue. The code
   objects of a key are loaded on a device at its first use, and kept while the
   device is. A GPU that no carried target covers refuses every kernel, as do
   the operations, dtypes and layouts the kernels do not serve. *)

module View = Nx_array.View
module Program = Nx_device.Program
module Code_object = Nx_amd_code_object

let name = "nx.amd"
let refuse fmt = Printf.ksprintf (fun r -> raise (Nx_backend.Refused r)) fmt

(* The kernel ABI (kernels/src/common.h): a workgroup's threads, and the most
   axes and source operands of a strided form. *)
let threads = 256
let max_rank = 32
let max_operands = 4

(* Workgroups: enough for each thread to take an element, at most [waves] per
   compute unit, past which each thread walks several. *)
let waves = 8

(* Shipped targets *)

(* The carried target whose code objects GPU [gpu] runs, by their generic
   processor's members. A GPU of several dies takes AQL packets, which launches
   do not encode. *)
let target ~gpu ~aql =
  let targets =
    List.map
      (fun (t, co) ->
        match Code_object.of_string co with
        | Ok co -> (t, co)
        | Error e -> failwith (Printf.sprintf "nx.amd: %s: %s" t e))
      (Lazy.force Archive.targets)
  in
  if aql then Error (gpu ^ " takes AQL packets, on its several dies")
  else
    match List.find_opt (fun (_, co) -> Code_object.runs_on co gpu) targets with
    | Some (t, _) -> Ok t
    | None ->
        Error
          (Printf.sprintf "nx.amd ships kernels for %s; this GPU is %s"
             (String.concat ", " (List.map fst targets))
             gpu)

(* Devices *)

(* What a device needs to run kernels: its carried target, or why it has none,
   its properties, and the programs of each key loaded on it. *)
type device = {
  target : (string, string) result;
  props : Nx_amd_device.props;
  programs : (string, Program.t * Program.t) Hashtbl.t;
}

let devices : (Nx_device.t * device) list ref = ref []
let lock = Mutex.create ()

let device d =
  Mutex.protect lock @@ fun () ->
  match List.find_opt (fun (d', _) -> Nx_device.equal d d') !devices with
  | Some (_, s) -> s
  | None ->
      let a = Option.get (Nx_amd_device.of_device d) in
      let s =
        {
          target = target ~gpu:(Nx_device.arch d) ~aql:(Nx_amd_device.aql a);
          props = Nx_amd_device.props a;
          programs = Hashtbl.create 16;
        }
      in
      devices := (d, s) :: !devices;
      s

(* The programs [c] and [s] of [key] on [d]. Two domains that load one key at
   once both load it, and find the same load. *)
let programs d s target key =
  match Mutex.protect lock (fun () -> Hashtbl.find_opt s.programs key) with
  | Some p -> p
  | None ->
      let path = target ^ "/" ^ key in
      let binary =
        match Archive.find path with
        | Some b -> b
        | None -> failwith ("nx.amd carries no code object " ^ path)
      in
      let load name =
        match Program.load d ~binary ~name with
        | Ok p -> p
        | Error why -> failwith why
      in
      let p = (load "c", load "s") in
      Mutex.protect lock (fun () -> Hashtbl.replace s.programs key p);
      p

(* Running a kernel *)

type operand = Operand : ('a, 'b) Nx_array.t -> operand

let address (Operand a) = Nativeint.to_int (Nx_device.Buffer.address a.buffer)
let itemsize (Operand a) = Nx_dtype.itemsize a.dtype
let view (Operand a) = a.view

(* Runs [key]'s module writing [dst] from [srcs]. *)
let run key ~dst srcs =
  let (Operand d) = dst in
  let dev = Nx_device.Buffer.device d.buffer in
  let s = device dev in
  let target = match s.target with Ok t -> t | Error e -> refuse "%s" e in
  let ops = dst :: srcs in
  let views = View.coalesce (List.map view ops) in
  let n = View.numel d.view and rank = View.ndim (List.hd views) in
  if rank > max_rank then
    refuse "operands of %d axes once merged; kernels take %d" rank max_rank;
  if List.length srcs > max_operands then
    refuse "%d operands; kernels take %d" (List.length srcs) max_operands;
  if n > 0 then begin
    let c, strided = programs dev s target key in
    let units = s.props.compute_units * s.props.xccs in
    let groups = Int.min ((n + threads - 1) / threads) (waves * units) in
    let b = Buffer.create 2048 in
    let i64 x = Buffer.add_int64_le b (Int64.of_int x) in
    let at o v = address o + (View.offset v * itemsize o) in
    let unit_stride v = View.ndim v = 0 || View.strides v = [| 1 |] in
    let program =
      if List.for_all unit_stride views then begin
        List.iter2 (fun o v -> i64 (at o v)) ops views;
        i64 n;
        i64 groups;
        c
      end
      else begin
        let dv = List.hd views and svs = Array.of_list (List.tl views) in
        i64 (at dst dv);
        List.iter (fun o -> i64 (address o)) srcs;
        i64 n;
        i64 rank;
        i64 groups;
        let shape = View.shape dv in
        for i = 0 to max_rank - 1 do
          i64 (if i < rank then shape.(i) else 0)
        done;
        for k = 0 to max_operands - 1 do
          i64 (if k < Array.length svs then View.offset svs.(k) else 0)
        done;
        for k = 0 to max_operands - 1 do
          let strides =
            if k < Array.length svs then View.strides svs.(k) else [||]
          in
          for i = 0 to max_rank - 1 do
            i64 (if i < Array.length strides then strides.(i) else 0)
          done
        done;
        strided
      end
    in
    Nx_amd_device.launch
      ~touches:(List.map (fun (Operand a) -> a.buffer) ops)
      [
        {
          program;
          groups = (groups, 1, 1);
          threads = (threads, 1, 1);
          args = Buffer.contents b;
        };
      ]
  end

(* The name of a dtype the kernels serve, as module keys spell it. *)
let served (type a b) (dt : (a, b) Nx_dtype.t) =
  match dt with
  | Complex64 | Complex128 -> refuse "no complex dtypes"
  | Int4 -> refuse "no int4"
  | UInt4 -> refuse "no uint4"
  | Bit -> refuse "no bit"
  | _ -> Nx_dtype.to_string dt

(* Kernels *)

module Kernels : Nx_backend.S = struct
  let name = name
  let runs_on d = Option.is_some (Nx_amd_device.of_device d)
  let owns = runs_on
  let no what = refuse "no %s" what

  let contiguous (type a b) (x : (a, b) Nx_array.t) ~(dst : (a, b) Nx_array.t) =
    ignore (served x.dtype);
    run
      (Printf.sprintf "contiguous.%d" (Nx_dtype.itemsize x.dtype))
      ~dst:(Operand dst) [ Operand x ]

  let cast (type a b c d) (x : (a, b) Nx_array.t) ~(dst : (c, d) Nx_array.t) =
    let s = served x.dtype and d = served dst.dtype in
    if s = d then
      run
        (Printf.sprintf "contiguous.%d" (Nx_dtype.itemsize x.dtype))
        ~dst:(Operand dst) [ Operand x ]
    else run (Printf.sprintf "cast.%s.%s" s d) ~dst:(Operand dst) [ Operand x ]

  let unary _ _ ~dst:_ = no "unary kernels"
  let binary _ _ _ ~dst:_ = no "binary kernels"
  let compare _ _ _ ~dst:_ = no "comparisons"
  let fma _ _ _ ~dst:_ = no "fma"
  let where _ _ _ ~dst:_ = no "where"
  let threefry _ _ ~dst:_ = no "threefry"
  let reduce _ ~axes:_ _ ~dst:_ = no "reductions"
  let scan _ ~axis:_ _ ~dst:_ = no "scans"
  let arg_reduce _ ~axis:_ _ ~dst:_ = no "arg_reduce"
  let sort ~descending:_ ~axis:_ _ ~dst:_ = no "sort"
  let argsort ~descending:_ ~axis:_ _ ~dst:_ = no "argsort"
  let group _ ~dst:_ = no "group"
  let pad _ _ _ ~dst:_ = no "pad"
  let cat ~axis:_ _ ~dst:_ = no "cat"
  let gather ~axis:_ _ _ ~dst:_ = no "gather"

  let scatter ~mode:_ ~unique:_ ~axis:_ ~indices:_ ~updates:_ _ ~dst:_ =
    no "scatter"

  let update _ ~starts:_ _ ~dst:_ = no "update"

  let unfold ~kernel_size:_ ~stride:_ ~dilation:_ ~padding:_ _ ~dst:_ =
    no "unfold"

  let fold ~output_size:_ ~kernel_size:_ ~stride:_ ~dilation:_ ~padding:_ _
      ~dst:_ =
    no "fold"

  let matmul _ _ ~dst:_ = no "matmul"
  let fft ~inverse:_ ~axes:_ _ ~dst:_ = no "fft"
  let rfft ~axes:_ _ ~dst:_ = no "rfft"
  let irfft ~axes:_ ~s:_ _ ~dst:_ = no "irfft"
  let cholesky ~upper:_ _ ~dst:_ = no "cholesky"
  let qr ~reduced:_ _ ~q:_ ~r:_ = no "qr"
  let lu _ ~lu:_ ~pivots:_ ~perm:_ = no "lu"
  let svd _ ~u:_ ~s:_ ~vt:_ = no "svd"
  let eig _ ~values:_ ~vectors:_ = no "eig"
  let eigh _ ~values:_ ~vectors:_ = no "eigh"

  let solve_triangular ~upper:_ ~transpose:_ ~unit_diag:_ _ _ ~dst:_ =
    no "solve_triangular"
end

let backend = Nx_backend.v (module Kernels)
