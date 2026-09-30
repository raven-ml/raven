open Tolk_next

let program r uops =
  let sink = List.find (fun u -> Ops.op u = Op.Sink) (List.rev uops) in
  let sink =
    match Ops.arg sink with
    | Ops.Kernel _ -> sink
    | _ -> Ops.replace sink ~arg:(Ops.Kernel (Ops.kernel_info ()))
  in
  Codegen.to_program
    (Ops.v Op.Program ~src:[ sink; Ops.v Op.Linear ~src:uops ])
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
  | Dtype.Bool, `Bool b -> Z.of_int (Bool.to_int b)
  | _ -> (
      match Dtype.bitcast dt (unsigned dt) v with
      | `Int z -> z
      | _ -> invalid_arg "an element's bits are no integer")

let of_bits dt z : Dtype.value =
  match dt with
  | Dtype.Bool -> `Bool (not (Z.equal z Z.zero))
  | _ -> Dtype.bitcast (unsigned dt) dt (`Int z)

let storage (p : Ops.param_arg) values =
  let n = Option.get p.size and size = Dtype.itemsize p.dtype in
  if Array.length values <> n then
    invalid_arg
      (Printf.sprintf "slot %d holds %d elements, not %d" p.slot n
         (Array.length values));
  let bytes =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (max 1 (n * size))
  in
  Array.iteri
    (fun i v ->
      let z = bits p.dtype v in
      for k = 0 to size - 1 do
        let byte =
          Z.to_int (Z.logand (Z.shift_right z (8 * k)) (Z.of_int 0xff))
        in
        bytes.{(i * size) + k} <- Char.chr byte
      done)
    values;
  bytes

let contents (p : Ops.param_arg) bytes =
  let size = Dtype.itemsize p.dtype in
  Array.init (Option.get p.size) (fun i ->
      let z = ref Z.zero in
      for k = size - 1 downto 0 do
        z :=
          Z.logor (Z.shift_left !z 8)
            (Z.of_int (Char.code bytes.{(i * size) + k}))
      done;
      of_bits p.dtype !z)

(* The buffer parameters of [prg], in the order its linear order declares
   them. *)
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
  let globals =
    match Ops.arg prg with
    | Ops.Program info -> info.globals
    | _ -> invalid_arg "not a compiled program"
  in
  let params = buffer_params prg in
  let initial (p : Ops.param_arg) =
    match List.assoc_opt p.slot buffers with
    | Some values -> values
    | None ->
        Array.make (Option.get p.size) (Dtype.truncate p.dtype (`Int Z.zero))
  in
  let memory = List.map (fun p -> (p, storage p (initial p))) params in
  let by_slot slot =
    let _, bytes =
      List.find (fun ((p : Ops.param_arg), _) -> p.slot = slot) memory
    in
    Nx_device.Buffer.of_bigarray bytes
  in
  let p = Tolk_next_engine.Program.load Nx_device.host prg in
  Tolk_next_engine.Program.run ?vars p (List.map by_slot globals);
  List.map
    (fun ((p : Ops.param_arg), bytes) -> (p.slot, contents p bytes))
    memory
