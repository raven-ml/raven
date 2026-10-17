open Tolk

let program r uops =
  let sink = List.find (fun u -> Ops.op u = Op.Sink) (List.rev uops) in
  let sink =
    match Ops.arg sink with
    | Ops.Kernel _ -> sink
    | _ -> Ops.replace sink ~arg:(Ops.Kernel (Ops.kernel_info ()))
  in
  let info = Call.program_info_of_sink ~target:r.Renderer.target sink in
  Codegen.compile
    (Ops.v Op.Program
       ~src:[ sink; Ops.v Op.Linear ~src:uops ]
       ~arg:(Program info))
    r

(* Elements as their bytes, little-endian *)

let unsigned dt =
  match Dtype.itemsize dt with
  | 1 -> Dtype.Uint8
  | 2 -> Dtype.Uint16
  | 4 -> Dtype.Uint32
  | 8 -> Dtype.Uint64
  | n -> invalid_arg (Printf.sprintf "no host element of %d bytes" n)

let bits dt (v : Dtype.value) =
  match (dt, v) with
  | Dtype.Bool, `Bool b -> Bigint.of_int (Bool.to_int b)
  | _ -> (
      match Dtype.bitcast dt (unsigned dt) v with
      | `Int z -> z
      | _ -> invalid_arg "an element's bits are no integer")

let of_bits dt z : Dtype.value =
  match dt with
  | Dtype.Bool -> `Bool (not (Bigint.equal z Bigint.zero))
  | _ -> Dtype.bitcast (unsigned dt) dt (`Int z)

let encode dt values =
  let size = Dtype.itemsize dt in
  let bytes =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout
      (max 1 (Array.length values * size))
  in
  Array.iteri
    (fun i v ->
      let z = bits dt v in
      for k = 0 to size - 1 do
        let byte =
          Bigint.to_int
            (Bigint.logand (Bigint.shift_right z (8 * k)) (Bigint.of_int 0xff))
        in
        bytes.{(i * size) + k} <- Char.chr byte
      done)
    values;
  bytes

let decode dt n bytes =
  let size = Dtype.itemsize dt in
  Array.init n (fun i ->
      let z = ref Bigint.zero in
      for k = size - 1 downto 0 do
        z :=
          Bigint.logor (Bigint.shift_left !z 8)
            (Bigint.of_int (Char.code bytes.{(i * size) + k}))
      done;
      of_bits dt !z)

(* Buffers of values *)

let filled b bytes =
  let n = Bigarray.Array1.dim bytes in
  Rig.Buffer.blit_from_string
    (String.init n (Bigarray.Array1.get bytes))
    0 b 0 n;
  b

let buffer d dt values =
  let bytes = encode dt values in
  filled (Rig.Buffer.create d (Bigarray.Array1.dim bytes)) bytes

let values dt b =
  let n = Rig.Buffer.length b in
  let bytes = Bytes.create n in
  Rig.Buffer.blit_to_bytes b 0 bytes 0 n;
  let a =
    Bigarray.Array1.init Bigarray.char Bigarray.c_layout n (Bytes.get bytes)
  in
  decode dt (n / Dtype.itemsize dt) a

(* A buffer of [n] bytes of the host that starts on 64 bytes. The kernels these
   tests run, recorded from tinygrad, access vectors up to that width, wider
   than the 16 bytes a host buffer is known to start on. *)
let aligned_host n =
  let b = Rig.Buffer.create Rig.host (n + 63) in
  let rec first j =
    if Rig.Buffer.aligned b ~at:j 64 then j else first (j + 1)
  in
  Rig.Buffer.view b ~first:(first 0) ~length:n

(* A buffer of the host holding [values] as the parameter [p]. *)
let storage (p : Ops.param_arg) values =
  let n = Option.get p.size in
  if Array.length values <> n then
    invalid_arg
      (Printf.sprintf "slot %d holds %d elements, not %d" p.slot n
         (Array.length values));
  let bytes = encode p.dtype values in
  filled (aligned_host (Bigarray.Array1.dim bytes)) bytes

let contents (p : Ops.param_arg) b =
  Array.sub (values p.dtype b) 0 (Option.get p.size)

(* The parameters of [prg] by slot: each buffer, in the order its linear order
   declares them, and each positional variable. *)
let buffer_params prg =
  List.filter_map
    (fun u ->
      match Ops.arg u with
      | Ops.Param p
        when Op.equal (Ops.op u) Param && p.addrspace <> Some Dtype.Alu ->
          Some p
      | _ -> None)
    (Ops.src (Ops.nth prg 1))

let on_host ?vars prg buffers =
  let info =
    match Ops.arg prg with
    | Ops.Program info -> info
    | _ -> invalid_arg "not a compiled program"
  in
  let params = buffer_params prg in
  let initial (p : Ops.param_arg) =
    match List.assoc_opt p.slot buffers with
    | Some values -> values
    | None ->
        Array.make (Option.get p.size)
          (Dtype.truncate p.dtype (`Int Bigint.zero))
  in
  let memory = List.map (fun p -> (p, storage p (initial p))) params in
  (* The call passes each buffer parameter as the run's slot of its number, and
     each positional variable as itself. *)
  let positional =
    List.filter
      (fun v -> match Ops.arg v with Ops.Param p -> p.slot >= 0 | _ -> false)
      info.vars
  in
  let slots = List.length info.globals + List.length positional in
  let arg slot =
    match List.find_opt (fun (p : Ops.param_arg) -> p.slot = slot) params with
    | Some p ->
        Call.param
          ~shape:[ Int (Option.get p.size) ]
          ~device:(Single "CPU") slot p.dtype
    | None ->
        List.find
          (fun v ->
            match Ops.arg v with Ops.Param p -> p.slot = slot | _ -> false)
          positional
  in
  let call = Ops.call prg (List.init slots arg) in
  let s = Tolk_engine.link_call ~devices:(fun _ -> Rig.host) call in
  let buffers = Array.make (List.length info.globals) [] in
  List.iter
    (fun ((p : Ops.param_arg), b) ->
      if p.slot < Array.length buffers then buffers.(p.slot) <- [ b ])
    memory;
  Tolk_engine.run ?vars s buffers;
  List.map
    (fun ((p : Ops.param_arg), b) -> (p.slot, contents p b))
    memory

(* Test devices *)

let opened =
  lazy
    (("CPU", Rig.host)
    :: List.map
         (fun n -> (n, Result.get_ok (Rig.memory_device n)))
         [ "CPU:1"; "CPU:2"; "CPU:3" ])

let devices () = Lazy.force opened
