open Tolk

let program r uops =
  let sink = List.find (fun u -> Ops.op u = Op.Sink) (List.rev uops) in
  let sink =
    match Ops.arg sink with
    | Ops.Kernel _ -> sink
    | _ -> Ops.replace sink ~arg:(Ops.Kernel (Ops.kernel_info ()))
  in
  let info = Ops.program_info_of_sink ~target:r.Renderer.target sink in
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

let storage (p : Ops.param_arg) values =
  let n = Option.get p.size in
  if Array.length values <> n then
    invalid_arg
      (Printf.sprintf "slot %d holds %d elements, not %d" p.slot n
         (Array.length values));
  encode p.dtype values

let contents (p : Ops.param_arg) bytes =
  decode p.dtype (Option.get p.size) bytes

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
        Array.make (Option.get p.size)
          (Dtype.truncate p.dtype (`Int Bigint.zero))
  in
  let memory = List.map (fun p -> (p, storage p (initial p))) params in
  let by_slot slot =
    let _, bytes =
      List.find (fun ((p : Ops.param_arg), _) -> p.slot = slot) memory
    in
    Nx_device.Buffer.of_bigarray bytes
  in
  let p = Tolk_engine.Program.load Nx_device.host prg in
  Tolk_engine.Program.run ?vars p (List.map by_slot globals);
  List.map
    (fun ((p : Ops.param_arg), bytes) -> (p.slot, contents p bytes))
    memory

(* Buffers of values *)

let buffer d dt values =
  let src = Nx_device.Buffer.of_bigarray (encode dt values) in
  let dst =
    Nx_device.Buffer.create d Nx_dtype.Scalar.UInt8
      (Nx_device.Buffer.nbytes src)
  in
  Nx_device.Buffer.copy ~src ~dst;
  dst

let values dt b =
  let n = Nx_device.Buffer.nbytes b in
  let bytes = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Nx_device.Buffer.copy ~src:b ~dst:(Nx_device.Buffer.of_bigarray bytes);
  decode dt (n / Dtype.itemsize dt) bytes

(* Test devices *)

let test_device ?(mapping = Nx_device.Driver.Identity) name =
  Nx_device.Driver.device ~name ~arch:"test" ~budget:max_int
    (Host_visible
       { memory = Nx_device.Driver.host_memory; mapping = Some mapping })

(* Maps host memory where it is, a page at least, as a GPU's driver does. *)
let pages =
  Nx_device.Driver.Pages
    {
      map = (fun a n -> Ok (Nx_device.Driver.Region.v ~host:a a n));
      unmap = ignore;
    }

let opened =
  lazy
    (("CPU", Nx_device.host)
     :: List.map (fun n -> (n, test_device n)) [ "CPU:1"; "CPU:2"; "CPU:3" ]
    @ [ ("CPU:4", test_device ~mapping:pages "CPU:4") ])

let devices () = Lazy.force opened
