open Tolk_next

let target =
  let machine =
    match Host_machine.architecture with "amd64" -> "x86_64" | m -> m
  in
  {
    Helpers.Target.device = "CPU";
    renderer = "CLANG";
    arch = machine ^ ",native";
    interface = "";
    indices = "";
  }

let params uops =
  List.filter_map
    (fun u ->
      match Ops.arg u with
      | Ops.Param p when Op.equal (Ops.op u) Param -> Some p
      | _ -> None)
    uops

let function_name uops =
  List.find_map
    (fun u ->
      match Ops.arg u with
      | Ops.Kernel k -> Some (Ops.function_name k)
      | _ -> None)
    uops
  |> Option.value ~default:"test"

let is_buffer (p : Ops.param_arg) = p.addrspace <> Some Dtype.Alu

let entry name ps =
  let pass (buffers, values, args) p =
    if is_buffer p then
      (buffers + 1, values, Printf.sprintf "b[%d]" buffers :: args)
    else (buffers, values + 1, Printf.sprintf "v[%d]" values :: args)
  in
  let _, _, args = List.fold_left pass (0, 0, []) ps in
  Printf.sprintf "\nvoid tolk_entry(void **b, const long long *v) { %s(%s); }\n"
    name
    (String.concat ", " (List.rev args))

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
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (n * size)
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

let value vars (p : Ops.param_arg) =
  match (Option.bind p.name (fun n -> List.assoc_opt n vars), p.bound) with
  | Some v, _ -> v
  | None, Some (`Int z) -> Z.to_int z
  | None, _ ->
      invalid_arg
        (Printf.sprintf "variable %s is unbound"
           (Option.value p.name ~default:(string_of_int p.slot)))

type t = { params : Ops.param_arg list; program : Nx_device.Program.t }

let load (r : Renderer.t) uops =
  let params = params uops in
  let source = r.render uops ^ entry (function_name uops) params in
  let binary = Renderer.Compiler.compile r.compiler source in
  match Nx_device.Program.load Nx_device.host ~binary ~name:"tolk_entry" with
  | Ok program -> { params; program }
  | Error why -> invalid_arg why

let run ?(vars = []) k buffers =
  let buffer_params = List.filter is_buffer k.params in
  let initial (p : Ops.param_arg) =
    match List.assoc_opt p.slot buffers with
    | Some values -> values
    | None ->
        Array.make (Option.get p.size) (Dtype.truncate p.dtype (`Int Z.zero))
  in
  let memory = List.map (fun p -> storage p (initial p)) buffer_params in
  let values =
    List.filter (fun p -> not (is_buffer p)) k.params |> List.map (value vars)
  in
  Nx_device.Program.call k.program
    (Array.of_list (List.map Nx_device.Buffer.of_bigarray memory))
    (Array.of_list values);
  List.map2
    (fun (p : Ops.param_arg) bytes -> (p.slot, contents p bytes))
    buffer_params memory
