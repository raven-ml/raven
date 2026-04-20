(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Environment *)

let debug = Helpers.getenv "DEBUG" 0

(* Runners *)

module Runner = struct
  type t = {
    display_name : string;
    device : Device.t;
    estimates : Program_spec.Estimates.t;
    mutable first_run : bool;
    call :
      Device.Buffer.t list -> (string * int) list ->
      wait:bool -> timeout:int option -> float option;
  }

  let make ~display_name ~device ?(estimates = Program_spec.Estimates.zero) call =
    { display_name; device; estimates; first_run = true; call }

  let dev t = t.device
  let display_name t = t.display_name
  let estimates t = t.estimates

  let call t bufs var_vals ~wait ~timeout =
    t.call bufs var_vals ~wait ~timeout

  let exec t rawbufs ?(var_vals = []) () =
    t.call rawbufs var_vals ~wait:false ~timeout:None
end

(* Local size optimization *)

let max_workgroup = 1024

let optimize_local_size ~device (prg : Device.prog) global_size
    (rawbufs : Device.Buffer.t list) =
  (* Avoid clobbering output if it also appears as input. *)
  let bufs = match rawbufs with
    | out :: rest when
        List.exists (fun b ->
          Device.Buffer.base_id b = Device.Buffer.base_id out) rest ->
        let test_out =
          Device.create_buffer ~size:(Device.Buffer.size out)
            ~dtype:(Device.Buffer.dtype out) device
        in
        Device.Buffer.ensure_allocated test_out;
        test_out :: rest
    | _ -> rawbufs
  in
  let buf_addrs = Array.of_list (List.map Device.Buffer.addr bufs) in
  let ndims = Array.length global_size in
  let powers = [| 1; 2; 4; 8; 16; 32; 64; 128; 256; max_workgroup |] in
  (* For each dimension, valid local sizes are {sz} ∪ powers that fit. *)
  let local_dims = Array.init ndims (fun i ->
    let sz = global_size.(i) in
    List.filter (fun x -> x <= sz)
      (List.sort_uniq Int.compare (sz :: Array.to_list powers)))
  in
  (* Enumerate all combinations with product ≤ max_workgroup. *)
  let local_sizes = ref [] in
  let rec enumerate acc dim =
    if dim >= ndims then begin
      let ls = Array.of_list (List.rev acc) in
      if Array.fold_left ( * ) 1 ls <= max_workgroup then
        local_sizes := ls :: !local_sizes
    end else
      List.iter (fun x -> enumerate (x :: acc) (dim + 1)) local_dims.(dim)
  in
  enumerate [] 0;
  (* Try each size twice, in random order. *)
  let all = Array.of_list (!local_sizes @ !local_sizes) in
  let n = Array.length all in
  for i = n - 1 downto 1 do
    let j = Random.int (i + 1) in
    let tmp = all.(i) in all.(i) <- all.(j); all.(j) <- tmp
  done;
  let best_time = ref infinity in
  let best_local = ref (Array.make ndims 1) in
  for k = 0 to n - 1 do
    let local_size = all.(k) in
    let global = Array.init ndims (fun i -> global_size.(i) / local_size.(i)) in
    let tm =
      try
        match prg.call buf_addrs ~global ~local:(Some local_size)
                ~vals:[||] ~wait:true ~timeout:None with
        | Some t -> t
        | None -> infinity
      with _ -> infinity
    in
    if tm < !best_time then begin
      best_time := tm;
      best_local := local_size
    end
  done;
  if Float.is_infinite !best_time then
    invalid_arg "all optimize_local_size exec failed";
  !best_local

(* Compiled runner *)

module Compiled_runner = struct
  type t = {
    runner : Runner.t;
    p : Program_spec.t;
    prg : Device.prog;
  }

  let runtimevars_of_spec p =
    List.filter_map (fun (i, (v : Program_spec.var)) ->
      if v.name = "core_id" then Some (v.name, i) else None)
      (List.mapi (fun i v -> (i, v)) (Program_spec.vars p))

  let create ~device ?prg (p : Program_spec.t) =
    if debug >= 3 && Program_spec.applied_opts p <> [] then
      Printf.eprintf "%s\n%!"
        (String.concat ", "
           (List.map Tolk_ir.Kernel.Opt.to_string
              (Program_spec.applied_opts p)));
    if debug >= 4 then
      Printf.eprintf "%s\n%!" (Program_spec.src p);
    let p, lib = match Program_spec.lib p with
      | Some lib -> p, lib
      | None ->
          let comp = match Renderer.compiler (Device.renderer device) with
            | Some c -> c
            | None -> invalid_arg "no compiler for device"
          in
          let lib = Compiler.compile_cached comp (Program_spec.src p) in
          Program_spec.with_lib lib p, lib
    in
    let prg = match prg with
      | Some h -> h
      | None ->
          Device.runtime device (Program_spec.name p) lib
            ~runtimevars:(runtimevars_of_spec p)
    in
    let vars = Program_spec.vars p in
    let call bufs var_vals ~wait ~timeout =
      let global, local = Program_spec.launch_dims p var_vals in
      let vals = Array.of_list (List.map (fun (v : Program_spec.var) ->
        match List.assoc_opt v.name var_vals with
        | Some n -> Int64.of_int n | None -> 0L) vars)
      in
      let buf_addrs = Array.of_list (List.map Device.Buffer.addr bufs) in
      prg.call buf_addrs ~global ~local ~vals ~wait ~timeout
    in
    let runner =
      Runner.make ~display_name:(Program_spec.name p)
        ~device ~estimates:(Program_spec.estimates p) call
    in
    { runner; p; prg }

  let p t = t.p
  let runner t = t.runner

  let call t bufs var_vals ~wait ~timeout =
    t.runner.call bufs var_vals ~wait ~timeout
end

(* View op *)

let view_op ~device (buf : Device.Buffer.t) =
  let display_name =
    strf "view %8d @ %-10d"
      (Device.Buffer.nbytes buf) (Device.Buffer.offset buf)
  in
  let call rawbufs _var_vals ~wait:_ ~timeout:_ =
    (match rawbufs with
     | [ dst; src ] ->
         if Device.Buffer.base_id dst <> Device.Buffer.base_id src then
           invalid_arg "view: dst must share base with src"
     | _ -> invalid_arg "view: expected exactly two buffers");
    None
  in
  Runner.make ~display_name ~device call

(* Buffer copy *)

let buffer_copy ~device ~total_sz ~dest_device ~src_device =
  let sz =
    if total_sz >= 1_000_000
    then strf "%7.2fM" (Float.of_int total_sz /. 1e6)
    else strf "%8d" total_sz
  in
  let dest_short = String.sub dest_device 0 (min 7 (String.length dest_device)) in
  let src_short = String.sub src_device 0 (min 7 (String.length src_device)) in
  let display_name = strf "copy %s, %7s <- %-7s" sz dest_short src_short in
  let call rawbufs _var_vals ~wait ~timeout:_ =
    match rawbufs with
    | [ dest; src ] ->
        if Device.Buffer.size dest <> Device.Buffer.size src
           || not (Tolk_ir.Dtype.equal
                     (Device.Buffer.dtype dest) (Device.Buffer.dtype src))
        then invalid_arg "buffer copy: size or dtype mismatch";
        let st = Unix.gettimeofday () in
        let tmp = Bytes.create (Device.Buffer.nbytes src) in
        Device.Buffer.copyout src tmp;
        Device.Buffer.copyin dest tmp;
        if wait then begin
          Device.synchronize device;
          Some (Unix.gettimeofday () -. st)
        end else None
    | _ -> invalid_arg "buffer copy: expected exactly two buffers"
  in
  let estimates = Program_spec.Estimates.{
    ops = Int 0; lds = Int total_sz; mem = Int total_sz } in
  Runner.make ~display_name ~device ~estimates call

(* XXX: BufferXfer — device-to-device transfer via allocator._transfer.
   Implement when multi-device support lands. *)

(* XXX: EncDec — hardware encode/decode (HEVC).  Out of scope. *)

(* Method cache *)

let method_cache : (string, Compiled_runner.t) Hashtbl.t = Hashtbl.create 64

let cache_key ~device ~ast_key ~base =
  let compiler_name = match Renderer.compiler (Device.renderer device) with
    | Some c -> Compiler.name c | None -> "" in
  strf "%s:%s:%s:%b" (Device.name device) compiler_name ast_key base

let get_runner ~device ~get_program (ast : Tolk_ir.Kernel.t) =
  let ast_key =
    Digest.to_hex (Digest.string (Marshal.to_string ast [])) in
  let ckey = cache_key ~device ~ast_key ~base:false in
  match Hashtbl.find_opt method_cache ckey with
  | Some car -> car
  | None ->
      let bkey = cache_key ~device ~ast_key ~base:true in
      match Hashtbl.find_opt method_cache bkey with
      | Some bcar ->
          let car = Compiled_runner.create ~device (Compiled_runner.p bcar) in
          Hashtbl.replace method_cache ckey car;
          car
      | None ->
          let p = get_program ast in
          let car = Compiled_runner.create ~device p in
          Hashtbl.replace method_cache ckey car;
          Hashtbl.replace method_cache bkey car;
          car

(* Resolve a scheduled Call node to a runner by dispatching on its
   callee: inline kernel ASTs are compiled, buffer views and copies
   are mapped to their respective runners. *)
let lower_ast ~device ~get_program (ast : Tolk_ir.Tensor.t)
    (bufs : Device.Buffer.t list) : Runner.t =
  let module T = Tolk_ir.Tensor in
  match T.view ast with
  | Call { callee = Ast kernel; _ } ->
      Compiled_runner.runner (get_runner ~device ~get_program kernel)
  | Call { callee = Ref ref_node; _ } -> begin
      match T.view ref_node with
      | Buffer_view _ ->
          view_op ~device (List.hd bufs)
      | Copy _ ->
          let dest = List.hd bufs and src = List.nth bufs 1 in
          buffer_copy ~device
            ~total_sz:(Device.Buffer.nbytes dest)
            ~dest_device:(Device.Buffer.device dest)
            ~src_device:(Device.Buffer.device src)
      | v ->
          invalid_arg (Format.asprintf
            "lower_ast: unsupported callee %a" T.pp_view v)
    end
  | v ->
      invalid_arg (Format.asprintf
        "lower_ast: expected Call, got %a" T.pp_view v)

(* Exec item *)

module Exec_item = struct
  type t = {
    ast : Tolk_ir.Tensor.t;
    bufs : Device.Buffer.t option list;
    var_vals : (string * int) list;
    mutable prg : Runner.t option;
  }

  let make ~ast ~bufs ?(var_vals = []) ?prg () =
    { ast; bufs; var_vals; prg }

  let ast t = t.ast
  let bufs t = t.bufs
  let var_vals t = t.var_vals

  let lower ~device ~get_program t =
    if Option.is_some t.prg then t
    else begin
      let bufs = List.filter_map Fun.id t.bufs in
      t.prg <- Some (lower_ast ~device ~get_program t.ast bufs);
      t
    end

  let run t ?(var_vals = []) ?(wait = false) ?(do_update_stats = true) () =
    let prg = match t.prg with
      | Some p -> p
      | None -> invalid_arg "exec item not lowered"
    in
    let merged = t.var_vals @ var_vals in
    let bufs = List.filter_map (fun b ->
      match b with
      | Some buf -> Device.Buffer.ensure_allocated buf; Some buf
      | None -> None)
      t.bufs
    in
    let et = prg.call bufs merged
        ~wait:(wait || debug >= 2) ~timeout:None in
    if do_update_stats then
      prg.first_run <- false;
    et
end

(* Run schedule *)

let run_schedule ~device ~get_program schedule
    ?(var_vals = []) ?(do_update_stats = true) () =
  List.iter (fun ei ->
    let ei = Exec_item.lower ~device ~get_program ei in
    ignore (Exec_item.run ei ~var_vals ~do_update_stats ()))
    schedule
