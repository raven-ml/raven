let strf = Printf.sprintf
let fail fmt = Printf.ksprintf failwith fmt

(* Values *)

(* A value of the format: what an argument or a tag is written as. *)
type value =
  | None_
  | Bool of bool
  | Int of Bigint.t
  | Float of float
  | Invalid
  | Str of string
  | Bytes of string
  | Name of string * string  (** A prefix and a name: [dtypes] and [half]. *)
  | Node of Ops.t
  | Tuple of value list
  | Record of string * (string * value) list

let rec equal_value v0 v1 =
  match (v0, v1) with
  | Float f0, Float f1 ->
      Int64.equal (Int64.bits_of_float f0) (Int64.bits_of_float f1)
  | Node u0, Node u1 -> Ops.equal u0 u1
  | Tuple l0, Tuple l1 -> List.equal equal_value l0 l1
  | Record (n0, f0), Record (n1, f1) ->
      String.equal n0 n1
      && List.equal
           (fun (k0, v0) (k1, v1) -> String.equal k0 k1 && equal_value v0 v1)
           f0 f1
  | Int z0, Int z1 -> Bigint.equal z0 z1
  | None_, None_ | Invalid, Invalid -> true
  | Bool b0, Bool b1 -> Bool.equal b0 b1
  | Str s0, Str s1 | Bytes s0, Bytes s1 -> String.equal s0 s1
  | Name (p0, n0), Name (p1, n1) -> String.equal p0 p1 && String.equal n0 n1
  | _ -> false

let kind = function
  | None_ -> "None"
  | Bool _ -> "a boolean"
  | Int _ -> "an integer"
  | Float _ -> "a float"
  | Invalid -> "Invalid"
  | Str _ -> "a string"
  | Bytes _ -> "bytes"
  | Name (p, n) -> p ^ "." ^ n
  | Node _ -> "a node"
  | Tuple _ -> "a tuple"
  | Record (n, _) -> "a " ^ n

let expected what v = fail "expected %s, found %s" what (kind v)

(* Printing *)

let quote s =
  let b = Buffer.create (String.length s + 2) in
  Buffer.add_char b '"';
  let escape = function
    | ('"' | '\\') as c ->
        Buffer.add_char b '\\';
        Buffer.add_char b c
    | '\n' -> Buffer.add_string b "\\n"
    | '\t' -> Buffer.add_string b "\\t"
    | ' ' .. '~' as c -> Buffer.add_char b c
    | c -> Buffer.add_string b (strf "\\x%02x" (Char.code c))
  in
  String.iter escape s;
  Buffer.add_char b '"';
  Buffer.contents b

let rec print index = function
  | None_ -> "None"
  | Bool b -> if b then "True" else "False"
  | Int z -> Bigint.to_string z
  | Float f
    when Float.is_nan f
         && not (Int64.equal (Int64.bits_of_float f) 0x7FF8_0000_0000_0000L) ->
      strf "nan(0x%016Lx)" (Int64.bits_of_float f)
  | Float f -> Format.asprintf "%a" Dtype.pp_const (`Float f)
  | Invalid -> "Invalid"
  | Str s -> quote s
  | Bytes s -> "b" ^ quote s
  | Name (p, n) -> p ^ "." ^ n
  | Node u -> "%" ^ string_of_int (index u)
  | Tuple [ v ] -> "(" ^ print index v ^ ",)"
  | Tuple l -> "(" ^ String.concat ", " (List.map (print index) l) ^ ")"
  | Record (n, fields) ->
      let field (k, v) = k ^ "=" ^ print index v in
      n ^ "(" ^ String.concat ", " (List.map field fields) ^ ")"

(* Parsing *)

type cursor = { text : string; mutable pos : int }

let peek c = if c.pos < String.length c.text then Some c.text.[c.pos] else None
let advance c = c.pos <- c.pos + 1
let rest c = String.sub c.text c.pos (String.length c.text - c.pos)

let expect c s =
  if String.starts_with ~prefix:s (rest c) then c.pos <- c.pos + String.length s
  else fail "expected %S at column %d" s (c.pos + 1)

let take_while c ok =
  let start = c.pos in
  while match peek c with Some ch -> ok ch | None -> false do
    advance c
  done;
  String.sub c.text start (c.pos - start)

let is_ident = function
  | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' -> true
  | _ -> false

let string c =
  expect c "\"";
  let b = Buffer.create 16 in
  let rec loop () =
    match peek c with
    | None -> fail "unterminated string"
    | Some '"' -> advance c
    | Some '\\' ->
        advance c;
        (match peek c with
        | Some (('"' | '\\') as ch) -> Buffer.add_char b ch
        | Some 'n' -> Buffer.add_char b '\n'
        | Some 't' -> Buffer.add_char b '\t'
        | Some 'x' when c.pos + 2 < String.length c.text -> (
            let hex = String.sub c.text (c.pos + 1) 2 in
            match int_of_string_opt ("0x" ^ hex) with
            | Some n ->
                Buffer.add_char b (Char.chr n);
                c.pos <- c.pos + 2
            | None -> fail "bad escape \\x%s" hex)
        | _ -> fail "bad escape at column %d" (c.pos + 1));
        advance c;
        loop ()
    | Some ch ->
        Buffer.add_char b ch;
        advance c;
        loop ()
  in
  loop ();
  Buffer.contents b

let number c =
  let sign =
    if peek c = Some '-' then (
      advance c;
      "-")
    else ""
  in
  let numeric = function
    | '0' .. '9' | '.' | 'e' | 'E' | '+' | '-' -> true
    | _ -> false
  in
  let digits =
    match take_while c numeric with "" -> take_while c is_ident | d -> d
  in
  let text = sign ^ digits in
  if
    String.exists
      (function '.' | 'e' | 'E' | 'i' | 'n' -> true | _ -> false)
      digits
  then
    match float_of_string_opt text with
    | Some f -> Float f
    | None -> fail "bad number %s" text
  else
    match Bigint.of_string text with
    | z -> Int z
    | exception Invalid_argument _ -> fail "bad number %s" text

(* [items c item] reads the items of a parenthesised list, after its opening
   parenthesis, up to and including its closing one. *)
let rec items c item =
  if peek c = Some ')' then (
    advance c;
    [])
  else
    let x = item c in
    match peek c with
    | Some ')' ->
        advance c;
        [ x ]
    | Some ',' ->
        advance c;
        if peek c = Some ' ' then advance c;
        x :: items c item
    | _ -> fail "expected ',' or ')' at column %d" (c.pos + 1)

let rec parse nodes c =
  match peek c with
  | Some '"' -> Str (string c)
  | Some '%' ->
      advance c;
      Node (nodes (int_of_string (take_while c is_ident)))
  | Some '(' ->
      advance c;
      Tuple (items c (parse nodes))
  | Some ('-' | '0' .. '9') -> number c
  | Some 'b' when String.starts_with ~prefix:"b\"" (rest c) ->
      advance c;
      Bytes (string c)
  | Some ('a' .. 'z' | 'A' .. 'Z') -> (
      let id = take_while c is_ident in
      match (id, peek c) with
      | "None", _ -> None_
      | "True", _ -> Bool true
      | "False", _ -> Bool false
      | "Invalid", _ -> Invalid
      | "nan", Some '(' ->
          advance c;
          let bits = take_while c is_ident in
          expect c ")";
          Float (Int64.float_of_bits (Int64.of_string bits))
      | ("inf" | "nan"), _ -> Float (float_of_string id)
      | _, Some '.' ->
          advance c;
          Name (id, take_while c is_ident)
      | _, Some '(' ->
          advance c;
          let field c =
            let k = take_while c is_ident in
            expect c "=";
            (k, parse nodes c)
          in
          Record (id, items c field)
      | _ -> fail "unknown name %s" id)
  | _ -> fail "expected a value at column %d" (c.pos + 1)

(* Codecs *)

(* A codec writes an OCaml value as a value of the format and reads it back. *)
type 'a codec = { write : 'a -> value; read : value -> 'a }

let int =
  let read = function
    | Int z when Bigint.fits_int z -> Bigint.to_int z
    | v -> expected "an integer" v
  in
  { write = (fun n -> Int (Bigint.of_int n)); read }

let bool =
  {
    write = (fun b -> Bool b);
    read = (function Bool b -> b | v -> expected "a boolean" v);
  }

let str =
  {
    write = (fun s -> Str s);
    read = (function Str s -> s | v -> expected "a string" v);
  }

let bytes =
  {
    write = (fun s -> Bytes s);
    read = (function Bytes s -> s | v -> expected "bytes" v);
  }

let node =
  {
    write = (fun u -> Node u);
    read = (function Node u -> u | v -> expected "a node" v);
  }

let list c =
  let read = function
    | Tuple l -> List.map c.read l
    | v -> expected "a tuple" v
  in
  { write = (fun l -> Tuple (List.map c.write l)); read }

let pair c0 c1 =
  let read = function
    | Tuple [ a; b ] -> (c0.read a, c1.read b)
    | v -> expected "a pair" v
  in
  { write = (fun (a, b) -> Tuple [ c0.write a; c1.write b ]); read }

let triple c0 c1 c2 =
  let read = function
    | Tuple [ a; b; c ] -> (c0.read a, c1.read b, c2.read c)
    | v -> expected "a triple" v
  in
  {
    write = (fun (a, b, c) -> Tuple [ c0.write a; c1.write b; c2.write c ]);
    read;
  }

let option c =
  let read = function None_ -> None | v -> Some (c.read v) in
  { write = (function None -> None_ | Some x -> c.write x); read }

(* A name written after its prefix, such as [dtypes.half], as [pp] prints it,
   and read back with [of_string]. *)
let named prefix pp of_string =
  let write x =
    let s = Format.asprintf "%a" pp x in
    Name
      ( prefix,
        String.sub s
          (String.length prefix + 1)
          (String.length s - String.length prefix - 1) )
  in
  let read = function
    | Name (p, n) when String.equal p prefix -> (
        match of_string n with Ok x -> x | Error e -> failwith e)
    | v -> expected (prefix ^ ".<NAME>") v
  in
  { write; read }

let dtype = named "dtypes" Dtype.pp Dtype.of_string
let op = named "Ops" Op.pp Op.of_string
let axis_type = named "AxisType" Ops.Axis_type.pp Ops.Axis_type.of_string

let addr_space =
  named "AddrSpace" Dtype.pp_addr_space Dtype.addr_space_of_string

let dvalue : Dtype.value codec =
  let read = function
    | Bool b -> `Bool b
    | Int z -> `Int z
    | Float f -> `Float f
    | v -> expected "a constant" v
  in
  {
    write =
      (function `Bool b -> Bool b | `Int z -> Int z | `Float f -> Float f);
    read;
  }

let const : Dtype.const codec =
  let read = function
    | Invalid -> `Invalid
    | v -> (dvalue.read v :> Dtype.const)
  in
  {
    write =
      (function `Invalid -> Invalid | #Dtype.value as v -> dvalue.write v);
    read;
  }

let device =
  let read = function
    | Str s -> Ops.Single s
    | Tuple l -> Multi (List.map str.read l)
    | v -> expected "a device" v
  in
  {
    write =
      (function
      | Ops.Single s -> Str s
      | Multi l -> Tuple (List.map str.write l));
    read;
  }

let sint =
  let read = function Node u -> Ops.Sym u | v -> Ops.Int (int.read v) in
  { write = (function Ops.Int n -> int.write n | Sym u -> Node u); read }

(* Records *)

(* A record is described once, field by field in declaration order, and the
   description both writes and reads it. A field with a default is left out when
   it holds it. *)
type ('r, 'dec) fields = {
  name : string;
  names : string list;
  read_fields : (string -> value option) -> 'dec;
  write_fields : ('r -> (string * value) option) list;
}

let record name dec =
  { name; names = []; read_fields = (fun _ -> dec); write_fields = [] }

let field k ?default c get r =
  let read lookup =
    let dec = r.read_fields lookup in
    match (lookup k, default) with
    | Some v, _ -> dec (c.read v)
    | None, Some d -> dec d
    | None, None -> fail "%s needs its field %s" r.name k
  in
  let write x =
    let v = c.write (get x) in
    match default with
    | Some d when equal_value v (c.write d) -> None
    | _ -> Some (k, v)
  in
  {
    r with
    names = k :: r.names;
    read_fields = read;
    write_fields = write :: r.write_fields;
  }

let finish r =
  let write x =
    Record (r.name, List.filter_map (fun w -> w x) (List.rev r.write_fields))
  in
  let read = function
    | Record (n, fs) when String.equal n r.name ->
        let check (k, _) =
          if not (List.mem k r.names) then fail "%s has no field %s" r.name k
        in
        List.iter check fs;
        r.read_fields (fun k -> List.assoc_opt k fs)
    | v -> expected ("a " ^ r.name) v
  in
  { write; read }

(* Arguments *)

let param_arg : Ops.param_arg codec =
  let make slot dtype size vmin_vmax multiple_of name addrspace device volatile
      bind_on_realize bound phase align : Ops.param_arg =
    {
      slot;
      dtype;
      size;
      vmin_vmax;
      multiple_of;
      name;
      addrspace;
      device;
      volatile;
      bind_on_realize;
      bound;
      phase;
      align;
    }
  in
  let get (f : Ops.param_arg -> _) = f in
  record "ParamArg" make
  |> field "slot" int (get (fun p -> p.slot))
  |> field "dtype" dtype (get (fun p -> p.dtype))
  |> field "size" ~default:None (option int) (get (fun p -> p.size))
  |> field "vmin_vmax" ~default:None
       (option (pair dvalue dvalue))
       (get (fun p -> p.vmin_vmax))
  |> field "multiple_of" ~default:None (option int)
       (get (fun p -> p.multiple_of))
  |> field "name" ~default:None (option str) (get (fun p -> p.name))
  |> field "addrspace" ~default:(Some Dtype.Global) (option addr_space)
       (get (fun p -> p.addrspace))
  |> field "device" ~default:None (option device) (get (fun p -> p.device))
  |> field "volatile" ~default:false bool (get (fun p -> p.volatile))
  |> field "bind_on_realize" ~default:false bool
       (get (fun p -> p.bind_on_realize))
  |> field "val" ~default:None (option dvalue) (get (fun p -> p.bound))
  |> field "phase" ~default:0 int (get (fun p -> p.phase))
  |> field "align" ~default:16 int (get (fun p -> p.align))
  |> finish

(* An optimisation is written with its operation, its axis and an argument whose
   shape depends on the operation. A split's target is written as the axis type
   of the axis it makes. *)
let opt : Opt.t codec =
  let ints = list int in
  let target =
    let write : Opt.target -> Ops.Axis_type.t = function
      | Upcast -> Upcast
      | Unroll -> Unroll
      | Local -> Local
    in
    let read : Ops.Axis_type.t -> Opt.target = function
      | Upcast -> Upcast
      | Unroll -> Unroll
      | Local -> Local
      | t ->
          fail "a split cannot make an axis of type %s"
            (Format.asprintf "%a" Ops.Axis_type.pp t)
    in
    {
      write = (fun t -> axis_type.write (write t));
      read = (fun v -> read (axis_type.read v));
    }
  in
  let write (o : Opt.t) =
    let name, axis, arg =
      match o with
      | Tc t -> ("TC", t.axis, ints.write [ t.tc_select; t.tc_opt; t.use_tc ])
      | Split s ->
          let top = if s.top then [ Bool true ] else [] in
          ( "SPLIT",
            s.axis,
            Tuple (int.write s.amount :: target.write s.target :: top) )
      | Padto p -> ("PADTO", p.axis, int.write p.amount)
      | Swap s -> ("SWAP", s.axis, int.write s.with_axis)
    in
    Record
      ( "Opt",
        [
          ("op", Name ("OptOps", name)); ("axis", int.write axis); ("arg", arg);
        ] )
  in
  let read v : Opt.t =
    match v with
    | Record
        ("Opt", [ ("op", Name ("OptOps", name)); ("axis", axis); ("arg", arg) ])
      -> (
        let axis = int.read axis in
        match (name, arg) with
        | "TC", Tuple [ s; o; u ] ->
            Tc
              {
                axis;
                tc_select = int.read s;
                tc_opt = int.read o;
                use_tc = int.read u;
              }
        | "SPLIT", Tuple [ amount; t ] ->
            Split
              {
                axis;
                amount = int.read amount;
                target = target.read t;
                top = false;
              }
        | "SPLIT", Tuple [ amount; t; top ] ->
            Split
              {
                axis;
                amount = int.read amount;
                target = target.read t;
                top = bool.read top;
              }
        | "PADTO", amount -> Padto { axis; amount = int.read amount }
        | "SWAP", other -> Swap { axis; with_axis = int.read other }
        | _ -> expected "an optimisation" v)
    | v -> expected "an Opt" v
  in
  { write; read }

let estimates : Ops.estimates codec =
  let make ops lds mem : Ops.estimates = { ops; lds; mem } in
  let get (f : Ops.estimates -> _) = f in
  record "Estimates" make
  |> field "ops" ~default:(Ops.Int 0) sint (get (fun e -> e.ops))
  |> field "lds" ~default:(Ops.Int 0) sint (get (fun e -> e.lds))
  |> field "mem" ~default:(Ops.Int 0) sint (get (fun e -> e.mem))
  |> finish

let kernel_info : Ops.kernel_info codec =
  let make name applied_opts opts_to_apply estimates beam split :
      Ops.kernel_info =
    let split =
      Option.map
        (fun (iterations, lo, hi) : Ops.split -> { iterations; lo; hi })
        split
    in
    { name; applied_opts; opts_to_apply; estimates; beam; split }
  in
  let get (f : Ops.kernel_info -> _) = f in
  record "KernelInfo" make
  |> field "name" ~default:"test" str (get (fun k -> k.name))
  |> field "applied_opts" ~default:[] (list opt) (get (fun k -> k.applied_opts))
  |> field "opts_to_apply" ~default:None
       (option (list opt))
       (get (fun k -> k.opts_to_apply))
  |> field "estimates" ~default:None (option estimates)
       (get (fun k -> k.estimates))
  |> field "beam" ~default:0 int (get (fun k -> k.beam))
  |> field "split" ~default:None
       (option (triple sint int int))
       (get (fun k ->
            Option.map
              (fun (s : Ops.split) -> (s.iterations, s.lo, s.hi))
              k.split))
  |> finish

let target : Helpers.Target.t codec =
  let make device renderer arch interface indices : Helpers.Target.t =
    { device; renderer; arch; interface; indices }
  in
  let get (f : Helpers.Target.t -> _) = f in
  record "Target" make
  |> field "device" ~default:"" str (get (fun t -> t.device))
  |> field "renderer" ~default:"" str (get (fun t -> t.renderer))
  |> field "arch" ~default:"" str (get (fun t -> t.arch))
  |> field "interface" ~default:"" str (get (fun t -> t.interface))
  |> field "indices" ~default:"" str (get (fun t -> t.indices))
  |> finish

let program_info : Ops.program_info codec =
  let make global_size local_size vars globals outs ins target :
      Ops.program_info =
    { global_size; local_size; vars; globals; outs; ins; target }
  in
  let get (f : Ops.program_info -> _) = f in
  let size = [ Ops.Int 1; Int 1; Int 1 ]
  and no_target = target.read (Record ("Target", [])) in
  record "ProgramInfo" make
  |> field "global_size" ~default:size (list sint)
       (get (fun p -> p.global_size))
  |> field "local_size" ~default:size (list sint) (get (fun p -> p.local_size))
  |> field "vars" ~default:[] (list node) (get (fun p -> p.vars))
  |> field "globals" ~default:[] (list int) (get (fun p -> p.globals))
  |> field "outs" ~default:[] (list int) (get (fun p -> p.outs))
  |> field "ins" ~default:[] (list int) (get (fun p -> p.ins))
  |> field "target" ~default:no_target target (get (fun p -> p.target))
  |> finish

let bufferize_opts : Ops.bufferize_opts codec =
  let make device addrspace removable : Ops.bufferize_opts =
    { device; addrspace; removable }
  in
  let get (f : Ops.bufferize_opts -> _) = f in
  record "BufferizeOpts" make
  |> field "device" (option device) (get (fun b -> b.device))
  |> field "addrspace" ~default:Dtype.Global addr_space
       (get (fun b -> b.addrspace))
  |> field "removable" ~default:true bool (get (fun b -> b.removable))
  |> finish

(* A kernel that a command-queue call enqueues is written as a tuple: [(devices,
   name, estimates, stamps, profile key, input slots, (outs, ins))]. *)
let hcq_kernel : Ops.hcq_kernel codec =
  let strs = list str and ints = list int and key = option bytes in
  let write (k : Ops.hcq_kernel) =
    Tuple
      [
        strs.write k.devices;
        str.write k.name;
        estimates.write k.estimates;
        ints.write k.stamps;
        key.write k.profile_key;
        ints.write k.input_slots;
        (pair ints ints).write (k.outs, k.ins);
      ]
  in
  let read : value -> Ops.hcq_kernel = function
    | Tuple [ devices; name; e; stamps; profile_key; input_slots; outs_ins ] ->
        let outs, ins = (pair ints ints).read outs_ins in
        {
          devices = strs.read devices;
          name = str.read name;
          estimates = estimates.read e;
          stamps = ints.read stamps;
          profile_key = key.read profile_key;
          input_slots = ints.read input_slots;
          outs;
          ins;
        }
    | v -> expected "a queue kernel" v
  in
  { write; read }

(* tinygrad's queue data has no [writes]: it is read as [written_bufs], the
   storage tinygrad says the call writes. It has no [copies] either: they are
   read as none. *)
let hcq_info : Ops.hcq_info codec =
  let make device kernels estimates nargs table inputs slots written_bufs :
      Ops.hcq_info =
    {
      device;
      kernels;
      estimates;
      nargs;
      table;
      inputs;
      slots;
      written_bufs;
      writes = written_bufs;
      copies = [];
    }
  in
  let get (f : Ops.hcq_info -> _) = f in
  let no_cost = estimates.read (Record ("Estimates", [])) in
  record "HCQInfo" make
  |> field "device" (list str) (get (fun h -> h.device))
  |> field "kernels" ~default:[] (list hcq_kernel) (get (fun h -> h.kernels))
  |> field "estimates" ~default:no_cost estimates (get (fun h -> h.estimates))
  |> field "nargs" ~default:0 int (get (fun h -> h.nargs))
  |> field "table" ~default:(-1) int (get (fun h -> h.table))
  |> field "inputs" ~default:[]
       (list (triple node int str))
       (get (fun h -> h.inputs))
  |> field "slots" ~default:[] (list (pair str int)) (get (fun h -> h.slots))
  |> field "written_bufs" ~default:[] (list node)
       (get (fun h -> h.written_bufs))
  |> finish

(* A call that precompiles its backward pass differentiates, which tolk does
   not, so the field is written and read only as its default. *)
let call_info : Ops.call_info codec =
  let make name precompile () aux dtype : Ops.call_info =
    { name; precompile; aux; dtype }
  in
  let get (f : Ops.call_info -> _) = f in
  let no_backward =
    let read = function
      | Bool false -> ()
      | _ -> failwith "a call that precompiles its backward pass differentiates"
    in
    { write = (fun () -> Bool false); read }
  in
  record "CallInfo" make
  |> field "name" ~default:None (option str) (get (fun c -> c.name))
  |> field "precompile" ~default:false bool (get (fun c -> c.precompile))
  |> field "precompile_backward" ~default:() no_backward (fun _ -> ())
  |> field "aux" ~default:None (option hcq_info) (get (fun c -> c.aux))
  |> field "dtype" ~default:Dtype.Void dtype (get (fun c -> c.dtype))
  |> finish

let wmma : Ops.wmma codec =
  let axes = list (pair (list int) int) in
  let dims = triple int int int and upcast = option (triple axes axes axes) in
  let write (w : Ops.wmma) =
    Tuple
      [
        dims.write w.dims;
        dtype.write w.dtype_in;
        int.write w.threads;
        upcast.write w.upcast_axes;
      ]
  in
  let read : value -> Ops.wmma = function
    | Tuple [ d; t; n; u ] ->
        {
          dims = dims.read d;
          dtype_in = dtype.read t;
          threads = int.read n;
          upcast_axes = upcast.read u;
        }
    | v -> expected "a tensor core argument" v
  in
  { write; read }

let arg_value : Ops.arg -> value option = function
  | No_arg -> None
  | Const c -> Some (const.write c)
  | Dtype d -> Some (dtype.write d)
  | Param p -> Some (param_arg.write p)
  | Range r ->
      Some
        (Tuple (List.map int.write r.axis_id @ [ axis_type.write r.axis_type ]))
  | Reduce r -> Some ((pair op int).write (r.op, r.num_axes))
  | Allreduce a -> Some ((pair op device).write (a.op, a.device))
  | Device d -> Some (device.write d)
  | Shard n -> Some (int.write n)
  | Axes l -> Some ((list int).write l)
  | Flips l -> Some ((list bool).write l)
  | String s -> Some (str.write s)
  | Bytes s -> Some (bytes.write s)
  | Queue q -> Some ((pair (list str) str).write (q.devices, q.queue))
  | Region { name; align = 128 } -> Some (str.write name)
  | Region r -> Some ((pair str int).write (r.name, r.align))
  | Code c -> Some ((pair str dtype).write (c.code, c.dtype))
  | Bufferize b -> Some (bufferize_opts.write b)
  | Kernel k -> Some (kernel_info.write k)
  | Program p -> Some (program_info.write p)
  | Call c -> Some (call_info.write c)
  | Wmma w -> Some (wmma.write w)

let arg_of (o : Op.t) v : Ops.arg =
  match o with
  | Const -> Const (const.read v)
  | Cast | Bitcast -> Dtype (dtype.read v)
  | Param | Buffer | Alloc -> Param (param_arg.read v)
  | Range -> (
      match v with
      | Tuple l -> (
          match List.rev l with
          | t :: ids ->
              Range
                {
                  axis_id = List.rev_map int.read ids;
                  axis_type = axis_type.read t;
                }
          | [] -> expected "a range" v)
      | v -> expected "a range" v)
  | Reduce ->
      let op, num_axes = (pair op int).read v in
      Reduce { op; num_axes }
  | Allreduce ->
      let op, device = (pair op device).read v in
      Allreduce { op; device }
  | Copy | Getaddr -> Device (device.read v)
  | Mselect -> Shard (int.read v)
  | Permute | Unshard -> Axes ((list int).read v)
  | Flip -> Flips ((list bool).read v)
  | Special | Custom_function | Source | Load | Store -> String (str.read v)
  | Linear -> (
      match v with
      | Str name -> Region { name; align = 128 }
      | Tuple [ Str name; align ] -> Region { name; align = int.read align }
      | v ->
          let devices, queue = (pair (list str) str).read v in
          Queue { devices; queue })
  | Binary -> Bytes (bytes.read v)
  | Custom | Customi | Ins ->
      let code, dtype = (pair str dtype).read v in
      Code { code; dtype }
  | Stage -> Bufferize (bufferize_opts.read v)
  | Sink -> Kernel (kernel_info.read v)
  | Program -> Program (program_info.read v)
  | Call -> Call (call_info.read v)
  | Wmma -> Wmma (wmma.read v)
  | o -> fail "%s takes no argument" (Format.asprintf "%a" Op.pp o)

let rec tag_value : Ops.Tag.t -> value = function
  | Bool b -> Bool b
  | Int n -> int.write n
  | String s -> Str s
  | Bytes s -> Bytes s
  | Dtype d -> dtype.write d
  | Tuple l -> Tuple (List.map tag_value l)

let rec tag_of : value -> Ops.Tag.t = function
  | Bool b -> Bool b
  | Int _ as v -> Int (int.read v)
  | Str s -> String s
  | Bytes s -> Bytes s
  | Name ("dtypes", _) as v -> Dtype (dtype.read v)
  | Tuple l -> Tuple (List.map tag_of l)
  | v -> expected "a tag" v

(* Graphs *)

(* The nodes an argument holds, in the order it is written. *)
let arg_nodes arg =
  let rec nodes acc = function
    | Node u -> u :: acc
    | Tuple l -> List.fold_left nodes acc l
    | Record (_, fields) ->
        List.fold_left (fun acc (_, v) -> nodes acc v) acc fields
    | _ -> acc
  in
  match arg_value arg with None -> [] | Some v -> List.rev (nodes [] v)

(* Sources come before the nodes of the argument, each list in order, so that a
   graph whose arguments hold no node is in {!Ops.toposort} order. *)
let toposort sink =
  let index = Ops.Tbl.create 256
  and order = ref []
  and stack = Stack.create () in
  Stack.push (sink, false) stack;
  while not (Stack.is_empty stack) do
    let u, visited = Stack.pop stack in
    if not (Ops.Tbl.mem index u) then
      if visited then (
        Ops.Tbl.add index u (Ops.Tbl.length index);
        order := u :: !order)
      else (
        Stack.push (u, true) stack;
        let children = Ops.src u @ arg_nodes (Ops.arg u) in
        List.iter (fun v -> Stack.push (v, false) stack) (List.rev children))
  done;
  (index, List.rev !order)

let to_string sink =
  let index, order = toposort sink in
  let index = Ops.Tbl.find index in
  let b = Buffer.create 4096 in
  let line u =
    let srcs = List.map (fun v -> string_of_int (index v)) (Ops.src u) in
    Buffer.add_string b
      (strf "%d %s %s [%s]" (index u)
         (print index (op.write (Ops.op u)))
         (print index (dtype.write (Ops.dtype u)))
         (String.concat ", " srcs));
    Option.iter
      (fun v -> Buffer.add_string b (" " ^ print index v))
      (arg_value (Ops.arg u));
    Option.iter
      (fun t -> Buffer.add_string b (" tag=" ^ print index (tag_value t)))
      (Ops.tag u);
    Buffer.add_char b '\n'
  in
  List.iter line order;
  Buffer.contents b

let node_line nodes i text =
  let c = { text; pos = 0 } in
  let word codec =
    let w = take_while c (fun ch -> ch <> ' ') in
    expect c " ";
    codec.read (parse nodes { text = w; pos = 0 })
  in
  let index = word int in
  if index <> i then fail "index %d is not the line's position %d" index i;
  let o = word op in
  let written = word dtype in
  expect c "[";
  let src =
    match take_while c (fun ch -> ch <> ']') with
    | "" -> []
    | s ->
        List.map
          (fun i -> nodes (int_of_string (String.trim i)))
          (String.split_on_char ',' s)
  in
  expect c "]";
  let arg =
    if
      c.pos < String.length text
      && not (String.starts_with ~prefix:" tag=" (rest c))
    then (
      expect c " ";
      arg_of o (parse nodes c))
    else No_arg
  in
  let tag =
    if c.pos < String.length text then (
      expect c " tag=";
      Some (tag_of (parse nodes c)))
    else None
  in
  if c.pos <> String.length text then
    fail "unexpected text at column %d" (c.pos + 1);
  let u = Ops.v ~src ~arg ?tag o in
  let text dt = Format.asprintf "%a" Dtype.pp dt in
  if not (Dtype.equal (Ops.dtype u) written) then
    fail "the node's data type is %s, not the written %s"
      (text (Ops.dtype u))
      (text written);
  u

let of_string text =
  let lines = String.split_on_char '\n' text in
  let lines =
    match List.rev lines with "" :: rest -> List.rev rest | _ -> lines
  in
  if lines = [] then failwith "the graph has no node";
  let nodes = Array.make (List.length lines) None in
  let node_at i j =
    if j >= 0 && j < i then Option.get nodes.(j)
    else fail "node %d is not an earlier line" j
  in
  List.iteri
    (fun i text ->
      match node_line (node_at i) i text with
      | u -> nodes.(i) <- Some u
      | exception (Failure e | Invalid_argument e) -> fail "node %d: %s" i e)
    lines;
  (* The graph's storage slots are taken: a slot {!Ops.unique_num} hands out
     later must not name one of them, or new storage would be the graph's. *)
  let taken =
    Array.fold_left
      (fun m u ->
        match Ops.arg (Option.get u) with Param p -> max m p.slot | _ -> m)
      (-1) nodes
  in
  while Ops.unique_num () <= taken do
    ()
  done;
  Option.get nodes.(Array.length nodes - 1)

(* Disk *)

let cached ~table ~key ~valid make =
  let read text =
    match of_string text with
    | g when valid g -> Some g
    | _ | (exception (Failure _ | Invalid_argument _)) -> None
  in
  let kept =
    match Helpers.Diskcache.get ~table key with
    | Some text -> read text
    | None | (exception Failure _) -> None
  in
  match kept with
  | Some g -> (g, true)
  | None ->
      let g = make () in
      Helpers.Diskcache.put ~table key (to_string g);
      (g, false)
