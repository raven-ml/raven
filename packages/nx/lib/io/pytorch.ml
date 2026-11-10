(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  The files torch.save writes, read as PyTorch's torch/serialization.py and
  torch/_utils.py read them (a datapoint: PyTorch documents the container, not
  its pickle's vocabulary).
  ---------------------------------------------------------------------------*)

(*= A load runs in passes over the pickle's nodes, each a loop with its own
    stack, so none recurses:

    decode       Pickle.decode     bytes -> nodes, calls as data
    interpret    children first    each node's meaning and key
    exclusivity  parents first     who may reach a mutable node
    exposure     parents first     the nodes the value holds, and how often
    first paths  depth first       where a shared value appears in full
    storages     once each         mapped or read into memory
    build        children first    the value returned
*)

module A = Nx_array
module B = Rig.Buffer
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Pickle

type value =
  | Null
  | Bool of bool
  | Int of int
  | Long of string
  | Float of float
  | Complex of Complex.t
  | String of string
  | Bytes of string
  | Tuple of value list
  | List of value list
  | Set of value list
  | Dict of (value * value) list
  | Tensor of Nx.host Nx.packed
  | Global of string
  | Object of string
  | Shared of int * value
  | Ref of int

(* Dtypes *)

(* How one of PyTorch's dtypes loads: at nx's dtype, or as its elements' [bytes]
   at uint8, in components of [part] bytes, which a swap of byte order
   reverses. *)
type dtype = Nx of D.any | Raw of { bytes : int; part : int }

let raw bytes part = Raw { bytes; part }

(* PyTorch's dtypes by the names [torch.dtype] pickles as, with their aliases,
   and each one's size from torch 2.15's [dtype.itemsize]. *)
let dtypes =
  D.
    [
      ("float64", Nx (Any Float64));
      ("double", Nx (Any Float64));
      ("float32", Nx (Any Float32));
      ("float", Nx (Any Float32));
      ("float16", Nx (Any Float16));
      ("half", Nx (Any Float16));
      ("bfloat16", Nx (Any Bfloat16));
      ("float8_e4m3fn", Nx (Any Float8_e4m3fn));
      ("float8_e5m2", Nx (Any Float8_e5m2));
      ("int64", Nx (Any Int64));
      ("long", Nx (Any Int64));
      ("int32", Nx (Any Int32));
      ("int", Nx (Any Int32));
      ("int16", Nx (Any Int16));
      ("short", Nx (Any Int16));
      ("int8", Nx (Any Int8));
      ("uint64", Nx (Any Uint64));
      ("uint32", Nx (Any Uint32));
      ("uint16", Nx (Any Uint16));
      ("uint8", Nx (Any Uint8));
      ("bool", Nx (Any Bool));
      ("complex128", Nx (Any Complex128));
      ("cdouble", Nx (Any Complex128));
      ("complex64", Nx (Any Complex64));
      ("cfloat", Nx (Any Complex64));
      ("complex32", raw 4 2);
      ("chalf", raw 4 2);
      ("bcomplex32", raw 4 2);
      ("float8_e4m3fnuz", raw 1 1);
      ("float8_e5m2fnuz", raw 1 1);
      ("float8_e8m0fnu", raw 1 1);
      ("float4_e2m1fn_x2", raw 1 1);
      ("bits1x8", raw 1 1);
      ("bits2x4", raw 1 1);
      ("bits4x2", raw 1 1);
      ("bits8", raw 1 1);
      ("bits16", raw 2 2);
      ("qint8", raw 1 1);
      ("quint8", raw 1 1);
      ("qint32", raw 4 4);
      ("quint4x2", raw 1 1);
      ("quint2x4", raw 1 1);
    ]
  @ List.concat_map
      (fun k ->
        [
          (Printf.sprintf "int%d" k, raw 1 1);
          (Printf.sprintf "uint%d" k, raw 1 1);
        ])
      [ 1; 2; 3; 4; 5; 6; 7 ]

(* The typed storage classes and their dtypes ([_dtype_to_storage_type_map]). *)
let storage_classes =
  [
    ("DoubleStorage", "float64");
    ("FloatStorage", "float32");
    ("HalfStorage", "float16");
    ("BFloat16Storage", "bfloat16");
    ("LongStorage", "int64");
    ("IntStorage", "int32");
    ("ShortStorage", "int16");
    ("CharStorage", "int8");
    ("ByteStorage", "uint8");
    ("BoolStorage", "bool");
    ("ComplexDoubleStorage", "complex128");
    ("ComplexFloatStorage", "complex64");
    ("QInt8Storage", "qint8");
    ("QInt32Storage", "qint32");
    ("QUInt8Storage", "quint8");
    ("QUInt4x2Storage", "quint4x2");
    ("QUInt2x4Storage", "quint2x4");
  ]

(* Untyped storages hold bytes: torch reads them at uint8. *)
let untyped = [ "torch.UntypedStorage"; "torch.storage.UntypedStorage" ]
let width = function Nx (D.Any dt) -> D.bits dt / 8 | Raw r -> r.bytes
let part = function Nx (D.Any dt) -> Storage.alignment dt | Raw r -> r.part
let alignment = function Nx (D.Any dt) -> Storage.alignment dt | Raw _ -> 1

(* Errors *)

(* [fail ctx at fmt] fails naming the pickle [ctx] and its byte [at]. *)
let fail ctx at fmt = Fail.failf ("%s: byte %d: " ^^ fmt) ctx at

(* Storages *)

(* A storage, by its key: the class and the count its first persistent id
   states, the views over it the value returned holds, the widest alignment and
   the components they read it at, and its memory once loaded. *)
type storage = {
  key : string;
  class_ : string;
  numel : int;
  mutable uses : int;
  mutable align : int;
  mutable parts : int list;
  mutable token : int;
  mutable memory : memory option;
}

and memory = Mapped of B.t | Own of B.t

(* A persistent id's storage: bytes [first] to [first + length] of a key's
   record, as a legacy id's view of another names them, at a class's dtype. *)
type sref = { storage : storage; first : int; length : int; dtype : string }

let dtype_of ctx at name =
  match List.assoc_opt name dtypes with
  | Some d -> d
  | None -> fail ctx at "the dtype torch.%s" name

(* What a node means. *)
type interp =
  | Data  (** A tuple, list, dict, set, frozenset or bytearray, as it is. *)
  | Storage_of of sref
  | View of view
  | Same of int  (** [_rebuild_parameter] of the tensor of a node. *)
  | Odict
  | Counter of P.value option
  | Pyset of P.value option
  | Complex_of of Complex.t
  | Bytes_of of string
  | Bytearray_of of string
  | Size of P.value
  | Opaque of string

and view = {
  sref : sref;
  dt : dtype;
  offset : int;  (** In elements, from the record's first byte. *)
  shape : int array;
  strides : int array;
}

(* Values of the pickle *)
(* Values of the pickle *)

let children (n : P.node) f =
  match n.kind with
  | Tuple a | Frozenset a -> Array.iter f a
  | Bytearray _ -> ()
  | List d | Set d -> Dynarray.iter f d
  | Dict d ->
      Dynarray.iter
        (fun (k, v) ->
          f k;
          f v)
        d
  | Persistent v -> f v
  | Call c ->
      f c.fn;
      f c.args;
      Option.iter f c.kwargs;
      Option.iter f c.state;
      Dynarray.iter f c.items;
      Dynarray.iter
        (fun (k, v) ->
          f k;
          f v)
        c.entries

(* [order root count] is the nodes reachable from [root], parents before
   children: the reverse of a post-order, found with an explicit stack. *)
let order root count =
  let seen = Bytes.make count '\000' in
  let post = Dynarray.create () in
  let stack = Stack.create () in
  (match root with P.Node n -> Stack.push (`Enter n) stack | _ -> ());
  while not (Stack.is_empty stack) do
    match Stack.pop stack with
    | `Exit n -> Dynarray.add_last post n
    | `Enter (n : P.node) ->
        if Bytes.get seen n.id = '\000' then begin
          Bytes.set seen n.id '\001';
          Stack.push (`Exit n) stack;
          children n (fun c ->
              match c with
              | P.Node c when Bytes.get seen c.id = '\000' ->
                  Stack.push (`Enter c) stack
              | _ -> ())
        end
  done;
  let n = Dynarray.length post in
  Array.init n (fun i -> Dynarray.get post (n - 1 - i))

(* Interpretation *)

type loader = {
  ctx : string;
  legacy : bool;
  decoded : P.decoded;
  storages : (string, storage) Hashtbl.t;
  views : (string, sref) Hashtbl.t;  (** A legacy id's views, by their key. *)
  interp : interp array;
}

let interp_of l = function P.Node n -> Some l.interp.(n.id) | _ -> None
let tuple = function P.Node { kind = Tuple a; _ } -> Some a | _ -> None

let ints l at v =
  let a =
    match (tuple v, interp_of l v) with
    | Some a, Some Data -> a
    | _, Some (Size s) -> Option.get (tuple s)
    | _ -> fail l.ctx at "a shape or strides that are no tuple"
  in
  Array.map
    (function
      | P.Int i when i >= 0 -> i
      | _ -> fail l.ctx at "a shape or strides that are not natural numbers")
    a

(* The entries of a dict or an [OrderedDict], keys and values in turn. *)
let entries l v =
  let flat d =
    Array.concat (List.map (fun (k, v) -> [| k; v |]) (Dynarray.to_list d))
  in
  match (v, interp_of l v) with
  | P.Node { kind = Dict d; _ }, _ -> Some (flat d)
  | P.Node { kind = Call c; _ }, Some Odict -> Some (flat c.entries)
  | _ -> None

(* An empty dict: what torch writes for a tensor's hooks and metadata. *)
let empty l v =
  match (v, interp_of l v) with
  | P.Null, _ -> true
  | P.Node { kind = Call c; _ }, Some Odict ->
      Dynarray.length c.entries = 0 && Option.is_none c.state
  | _ -> ( match entries l v with Some [||] -> true | _ -> false)

let storage_class ctx at name =
  if List.mem name untyped then "uint8"
  else
    let class_ =
      match String.rindex_opt name '.' with
      | Some i when List.mem (String.sub name 0 i) [ "torch"; "torch.cuda" ] ->
          String.sub name (i + 1) (String.length name - i - 1)
      | _ -> fail ctx at "the storage class %s" name
    in
    match List.assoc_opt class_ storage_classes with
    | Some d -> d
    | None -> fail ctx at "the storage class %s" name

(* [storage l at ~class_ ~key ~numel] is the storage [key], whose ids must all
   state one class and count unless it is empty, as torch's own save allows. The
   count's bytes are checked to fit an [int] first. *)
let storage l at ~class_ ~key ~numel =
  let w = width (dtype_of l.ctx at class_) in
  if Hashtbl.mem l.views key then
    fail l.ctx at "storage %S, which a view's key also names" key;
  match Hashtbl.find_opt l.storages key with
  | None ->
      let s =
        {
          key;
          class_;
          numel;
          uses = 0;
          align = 1;
          parts = [];
          token = min_int;
          memory = None;
        }
      in
      Hashtbl.add l.storages key s;
      s
  | Some s ->
      let w' = width (dtype_of l.ctx at s.class_) in
      if
        (s.class_ <> class_ || s.numel <> numel)
        && (s.numel * w' > 0 || numel * w > 0)
      then fail l.ctx at "storage %S named with two dtypes or lengths" key;
      s

(* A persistent id: [("storage", class, key, location, numel)], and in the
   legacy format after it [None] or a view [(key, offset, size)] of the storage,
   in elements. torch reads a view key once and keeps what it first read under
   it ([_legacy_load]), so a view key named again with another storage, offset,
   size or dtype, or that names a storage, is refused: the two readings would
   differ. *)
let persistent l at pid =
  let fields = match tuple pid with Some a -> a | None -> [||] in
  let class_, key, numel, view =
    match fields with
    | [|
     P.String ("storage", _);
     P.Global (c, _);
     P.String (key, _);
     P.String _;
     P.Int n;
    |]
      when (not l.legacy) && n >= 0 ->
        (c, key, n, P.Null)
    | [|
     P.String ("storage", _);
     P.Global (c, _);
     P.String (key, _);
     P.String _;
     P.Int n;
     v;
    |]
      when l.legacy && n >= 0 ->
        (c, key, n, v)
    | _ -> fail l.ctx at "a persistent id that names no storage"
  in
  let class_ = storage_class l.ctx at class_ in
  let w = width (dtype_of l.ctx at class_) in
  if numel > max_int / w then
    fail l.ctx at "a storage of more bytes than an int counts";
  let s = storage l at ~class_ ~key ~numel in
  match view with
  | P.Null -> { storage = s; first = 0; length = numel * w; dtype = class_ }
  | v -> (
      match tuple v with
      | Some [| P.String (view, _); P.Int offset; P.Int size |]
        when offset >= 0 && size >= 0 && offset <= numel
             && size <= numel - offset -> (
          let r =
            {
              storage = s;
              first = offset * w;
              length = size * w;
              dtype = class_;
            }
          in
          if Hashtbl.mem l.storages view then
            fail l.ctx at "view %S, which a storage's key also names" view;
          match Hashtbl.find_opt l.views view with
          | None ->
              Hashtbl.add l.views view r;
              r
          | Some r' ->
              if
                r'.storage != s || r'.first <> r.first || r'.length <> r.length
                || r'.dtype <> r.dtype
              then
                fail l.ctx at
                  "view %S named with another storage, offset, size or dtype"
                  view;
              r')
      | _ -> fail l.ctx at "a storage view that does not lie in its storage")

(* [layout v] is [v]'s layout over its storage's record. A dtype nx lacks reads
   its elements' bytes, so positions, strides and shape gain a factor or an axis
   of its width, each checked to fit an [int].

   Raises [Invalid_argument] if they do not, or as [Layout.v] does. *)
let layout (v : view) =
  let shape, strides, offset =
    match v.dt with
    | Nx _ | Raw { bytes = 1; _ } -> (v.shape, v.strides, v.offset)
    | Raw { bytes = k; _ } ->
        let scale x = if x > max_int / k then invalid_arg "scale" else x * k in
        ( Array.append v.shape [| k |],
          Array.append (Array.map scale v.strides) [| 1 |],
          scale v.offset )
  in
  L.v ~offset ~strides:(Iarray.of_array strides) (Iarray.of_array shape)

(* [rebuild l at sref ~dt offset size stride] is the view a rebuild call makes,
   checked to lie in its storage. *)
let rebuild l at sref ~dt offset shape strides =
  let w = width dt in
  let past () = fail l.ctx at "a view that reaches past its storage" in
  if sref.first mod w <> 0 then
    fail l.ctx at "a view at no multiple of its elements' size";
  if Array.length shape <> Array.length strides then
    fail l.ctx at "a shape and strides of different lengths";
  (try Storage.check_shape shape with Failure m -> fail l.ctx at "%s" m);
  if offset > max_int - (sref.first / w) then past ();
  let v = { sref; dt; offset = (sref.first / w) + offset; shape; strides } in
  let l' = try layout v with Invalid_argument _ -> past () in
  (* The layout counts positions in elements, of a byte for a dtype nx lacks. *)
  let size = match dt with Nx _ -> w | Raw _ -> 1 in
  let _, hi = L.span l' in
  if hi > (sref.first + sref.length) / size then past ();
  v

let latin1 ctx at s =
  let b = Buffer.create (String.length s) in
  let rec go i =
    if i < String.length s then begin
      let d = String.get_utf_8_uchar s i in
      let u = Uchar.to_int (Uchar.utf_decode_uchar d) in
      if (not (Uchar.utf_decode_is_valid d)) || u > 0xff then
        fail ctx at "a string latin-1 does not encode";
      Buffer.add_char b (Char.chr u);
      go (i + Uchar.utf_decode_length d)
    end
  in
  go 0;
  Buffer.contents b

let latin1_names = [ "latin1"; "latin-1" ]

(* The rebuild calls' arguments: storage, offset, size and stride; then for v2
   [requires_grad] and hooks, for v3 a dtype too, and optional metadata. *)
let tensor_call l at name args =
  let storage_of v =
    match (v, interp_of l v) with
    | P.Node _, Some (Storage_of s) -> s
    | _ -> fail l.ctx at "%s of a value that is no storage" name
  in
  let offset = function
    | P.Int i when i >= 0 -> i
    | _ -> fail l.ctx at "a storage offset that is not a natural number"
  in
  let flag = function
    | P.Bool _ -> ()
    | _ -> fail l.ctx at "a requires_grad that is no bool"
  in
  let hooks v =
    if not (empty l v) then fail l.ctx at "a tensor saved with backward hooks"
  in
  let metadata v =
    if not (empty l v) then
      fail l.ctx at "a tensor's metadata, such as its conjugate or negative bit"
  in
  let view s off size stride ~dt =
    View (rebuild l at s ~dt (offset off) (ints l at size) (ints l at stride))
  in
  let class_dt s = dtype_of l.ctx at s.dtype in
  match (name, args) with
  | "_rebuild_tensor", [| s; off; size; stride |] ->
      let s = storage_of s in
      view s off size stride ~dt:(class_dt s)
  | "_rebuild_tensor_v2", [| s; off; size; stride; rg; h |] ->
      flag rg;
      hooks h;
      let s = storage_of s in
      view s off size stride ~dt:(class_dt s)
  | "_rebuild_tensor_v2", [| s; off; size; stride; rg; h; m |] ->
      flag rg;
      hooks h;
      metadata m;
      let s = storage_of s in
      view s off size stride ~dt:(class_dt s)
  | ( "_rebuild_tensor_v3",
      ( [| s; off; size; stride; rg; h; dt |]
      | [| s; off; size; stride; rg; h; dt; _ |] ) ) ->
      flag rg;
      hooks h;
      if Array.length args = 8 then metadata args.(7);
      let dt =
        match dt with
        | P.Global (g, _) when String.starts_with ~prefix:"torch." g ->
            dtype_of l.ctx at (String.sub g 6 (String.length g - 6))
        | _ -> fail l.ctx at "a dtype that is no global"
      in
      view (storage_of s) off size stride ~dt
  | "_rebuild_parameter", [| data; rg; h |] -> (
      flag rg;
      hooks h;
      match (data, interp_of l data) with
      | P.Node d, Some (View _ | Same _) -> Same d.id
      | _ -> fail l.ctx at "a Parameter of a value that is no tensor")
  | _ -> fail l.ctx at "torch._utils.%s with arguments it does not take" name

let rebuilds =
  [
    "torch._utils._rebuild_tensor";
    "torch._utils._rebuild_tensor_v2";
    "torch._utils._rebuild_tensor_v3";
    "torch._utils._rebuild_parameter";
  ]

(* [call l at c] is what the call [c] means: a value of the vocabulary, refused
   if it has an argument, a state, items or entries its global does not take, or
   an opaque value. *)
let call l at (c : P.call) =
  let args = Option.get (tuple c.args) in
  let bare () =
    Option.is_none c.state
    && Dynarray.length c.items = 0
    && Dynarray.length c.entries = 0
    && Option.is_none c.kwargs
  in
  let wrong g = fail l.ctx at "%s with what it does not take" g in
  (* Bytes, as a pickle holds them or as protocol 2's [_codecs.encode] spells
     them. *)
  let bytes = function
    | P.Bytes (b, _) -> Some b
    | v -> ( match interp_of l v with Some (Bytes_of b) -> Some b | _ -> None)
  in
  let name_of = function
    | P.Global (g, _) -> g
    | P.Node _ as v -> (
        match interp_of l v with
        | Some (Opaque name) -> name
        | _ -> fail l.ctx at "a call of a value that is not callable")
    | _ -> fail l.ctx at "a call of a value that is not callable"
  in
  if c.new_ then Opaque (name_of c.fn)
  else
    match c.fn with
    | P.Global (("collections.OrderedDict" as g), _) ->
        let state_ok =
          match c.state with None -> true | Some v -> entries l v <> None
        in
        if
          Array.length args > 0
          || Dynarray.length c.items > 0
          || (not state_ok) || Option.is_some c.kwargs
        then wrong g;
        Odict
    | P.Global (("collections.Counter" as g), _) -> (
        if not (bare ()) then wrong g;
        match args with
        | [||] -> Counter None
        | [| d |] when entries l d <> None -> Counter (Some d)
        | _ -> wrong g)
    | P.Global ((("builtins.set" | "builtins.frozenset") as g), _) -> (
        if not (bare ()) then wrong g;
        match args with
        | [||] -> Pyset None
        | [| P.Node { kind = List _ | Tuple _; _ } as v |] -> Pyset (Some v)
        | _ -> wrong g)
    | P.Global (("builtins.complex" as g), _) -> (
        let number = function
          | P.Int i -> Float.of_int i
          | P.Float f -> f
          | _ -> wrong g
        in
        if not (bare ()) then wrong g;
        match args with
        | [| re; im |] -> Complex_of { re = number re; im = number im }
        | _ -> wrong g)
    | P.Global (("_codecs.encode" as g), _) -> (
        if not (bare ()) then wrong g;
        match args with
        | [| P.String (s, _); P.String (e, _) |] when List.mem e latin1_names ->
            Bytes_of (latin1 l.ctx at s)
        | _ -> wrong g)
    | P.Global (("builtins.bytes" as g), _) -> (
        if not (bare ()) then wrong g;
        match args with
        | [||] -> Bytes_of ""
        | [| b |] when bytes b <> None -> Bytes_of (Option.get (bytes b))
        | _ -> wrong g)
    | P.Global (("builtins.bytearray" as g), _) -> (
        if not (bare ()) then wrong g;
        match args with
        | [||] -> Bytearray_of ""
        | [| b |] when bytes b <> None -> Bytearray_of (Option.get (bytes b))
        | [| P.String (s, _); P.String (e, _) |] when List.mem e latin1_names ->
            Bytearray_of (latin1 l.ctx at s)
        | _ -> wrong g)
    | P.Global (("torch.Size" as g), _) -> (
        if not (bare ()) then wrong g;
        match args with
        | [| v |] when tuple v <> None ->
            ignore (ints l at v);
            Size v
        | _ -> wrong g)
    | P.Global (g, _) when List.mem g rebuilds ->
        let name = String.sub g 13 (String.length g - 13) in
        if Option.is_some c.state then
          fail l.ctx at "a tensor that BUILD gives a state";
        if not (bare ()) then wrong g;
        tensor_call l at name args
    | fn -> Opaque (name_of fn)

let interpret l (n : P.node) =
  match n.kind with
  | Tuple _ | Frozenset _ | Bytearray _ | List _ | Dict _ | Set _ -> Data
  | Persistent pid -> Storage_of (persistent l n.at pid)
  | Call c -> call l n.at c

(* Exclusivity *)

(* Whether a node's value is one a call could change: a dict, list, set,
   bytearray, object, tensor or storage. *)
let mutable_ (n : P.node) = function
  | Data -> ( match n.kind with Tuple _ | Frozenset _ -> false | _ -> true)
  | Complex_of _ | Bytes_of _ | Size _ -> false
  | Storage_of _ | View _ | Same _ | Odict | Counter _ | Pyset _
  | Bytearray_of _ | Opaque _ ->
      true

(* Every node reachable from what a call outside the vocabulary, or a state, is
   given must be reachable only through it, where it reaches a mutable value. A
   node's token names the innermost such call it lies under: the tokens of its
   parents' edges must agree. Parents come first, so each node's token is final
   when its children's are set. *)
let exclusive l (order : P.node array) count =
  let live = Bytes.make count '\000' in
  for i = Array.length order - 1 downto 0 do
    let n = order.(i) in
    let alive = ref (mutable_ n l.interp.(n.id)) in
    children n (function
      | P.Node c when Bytes.get live c.id = '\001' -> alive := true
      | _ -> ());
    if !alive then Bytes.set live n.id '\001'
  done;
  let unset = min_int and none = -1 in
  let token = Array.make count unset in
  let by_id = Array.make count None in
  Array.iter (fun (n : P.node) -> by_id.(n.id) <- Some n) order;
  let name t =
    match by_id.(t) with
    | Some n -> (
        match l.interp.(n.id) with Opaque name -> name | _ -> "a state")
    | None -> "a call"
  in
  let given at t t' =
    let t = if t >= 0 then t else t' in
    fail l.ctx at "a value given to %s is reachable elsewhere" (name t)
  in
  if Array.length order > 0 then token.(order.(0).id) <- none;
  Array.iter
    (fun (n : P.node) ->
      let own = token.(n.id) in
      let edge child =
        match (l.interp.(n.id), n.kind) with
        | Opaque _, _ -> n.id
        | Odict, Call { state = Some (P.Node s); _ } when s == child -> n.id
        | _ -> own
      in
      children n (fun c ->
          match c with
          | P.Node c when Bytes.get live c.id = '\001' ->
              let t = edge c in
              if token.(c.id) = unset then token.(c.id) <- t
              else if token.(c.id) <> t then given c.at t token.(c.id)
          | _ -> ());
      match l.interp.(n.id) with
      | Storage_of { storage = s; _ } ->
          if s.token = unset then s.token <- own
          else if s.token <> own then given n.at own s.token
      | _ -> ())
    order

(* Keys *)

(* A key as Python compares it, as ints: numbers by value, so [True], [1], [1.0]
   and [1+0j] are one key, and strings, bytes, large integers and globals by
   atom. A tuple is its elements' key numbers, so each key is built once, in
   constant time per element. *)
module Key = Hashtbl.MakeSeeded (struct
  type t = int array

  let equal a b = Array.length a = Array.length b && Array.for_all2 ( = ) a b

  (* Each element is mixed under the seed, which a table made with [~random]
     draws: a file that chooses its keys cannot choose their hashes, as it could
     a linear fold's. *)
  let seeded_hash seed a =
    Array.fold_left (fun h x -> Hashtbl.seeded_hash seed (h, x)) seed a
end)

(* Key numbers that are not keys: a value Python's class compares, and NaN,
   which equals nothing. *)
let unhashable = -1
let nan_key = -2
let bits f = Int64.bits_of_float (f +. 0.)
let lo f = Int64.to_int (Int64.logand (bits f) 0xffffffffL)
let hi f = Int64.to_int (Int64.shift_right_logical (bits f) 32)

(* [exact f] is the two's complement, little-endian, of the integral float [f]:
   its 53-bit significand shifted by its exponent. *)
let exact f =
  let m, e = Float.frexp (Float.abs f) in
  let m = Int64.of_float (Float.ldexp m 53) and e = e - 53 in
  let b = Bytes.make (((53 + e) / 8) + 2) '\000' in
  for i = 0 to 52 do
    if Int64.logand (Int64.shift_right_logical m i) 1L = 1L then
      let k = i + e in
      Bytes.set b (k / 8)
        (Char.chr (Char.code (Bytes.get b (k / 8)) lor (1 lsl (k mod 8))))
  done;
  if f < 0. then begin
    let carry = ref 1 in
    for i = 0 to Bytes.length b - 1 do
      let v = (lnot (Char.code (Bytes.get b i)) land 0xff) + !carry in
      Bytes.set b i (Char.chr (v land 0xff));
      carry := v lsr 8
    done
  end;
  Bytes.to_string b

type keys = { table : int Key.t; atom : char -> string -> int }

let intern ks k =
  match Key.find_opt ks.table k with
  | Some i -> i
  | None ->
      let i = Key.length ks.table in
      Key.add ks.table k i;
      i

let number ks f =
  if Float.is_nan f then nan_key
  else if Float.is_integer f then
    if f >= -0x1p62 && f < 0x1p62 then intern ks [| 1; Float.to_int f |]
    else
      match P.integer (exact f) with
      | Ok i -> intern ks [| 1; i |]
      | Error b -> intern ks [| 4; ks.atom 'L' b |]
  else intern ks [| 2; lo f; hi f |]

let scalar_key ks = function
  | P.Null -> intern ks [| 0 |]
  | P.Bool b -> intern ks [| 1; Bool.to_int b |]
  | P.Int i -> intern ks [| 1; i |]
  | P.Float f -> number ks f
  | P.Long (_, a) | P.String (_, a) | P.Bytes (_, a) | P.Global (_, a) ->
      intern ks [| 4; a |]
  | P.Node _ -> unhashable

(* [keys l order] is each node's key number, children first. *)
let keys l (order : P.node array) count =
  let ks = { table = Key.create ~random:true 64; atom = P.atom l.decoded } in
  let kid = Array.make count unhashable in
  let of_value = function P.Node n -> kid.(n.id) | v -> scalar_key ks v in
  for i = Array.length order - 1 downto 0 do
    let n = order.(i) in
    kid.(n.id) <-
      (match (l.interp.(n.id), n.kind) with
      | Data, Tuple a ->
          let ids = Array.map of_value a in
          if Array.exists (fun k -> k = nan_key) ids then nan_key
          else if Array.exists (fun k -> k < 0) ids then unhashable
          else intern ks (Array.append [| 5 |] ids)
      | Size v, _ -> of_value v
      | Complex_of { re; im }, _ ->
          if Float.is_nan re || Float.is_nan im then nan_key
          else if im = 0. then number ks re
          else intern ks [| 3; lo re; hi re; lo im; hi im |]
      | Bytes_of s, _ -> intern ks [| 4; ks.atom 'B' s |]
      | _ -> unhashable)
  done;
  of_value

(* Exposure: the value's children of a node, in the order the value holds them,
   keys and values in turn for a dict. *)
let kids l (n : P.node) =
  let of_list = function
    | P.Node { kind = Tuple a; _ } -> a
    | P.Node { kind = List d; _ } -> Dynarray.to_array d
    | _ -> [||]
  in
  match (l.interp.(n.id), n.kind) with
  | Data, (Tuple a | Frozenset a) -> a
  | Data, (List d | Set d) -> Dynarray.to_array d
  | Data, Dict _ | Odict, _ -> Option.get (entries l (P.Node n))
  | Counter (Some d), _ -> Option.get (entries l d)
  | Pyset (Some v), _ | Size v, _ -> of_list v
  | _ -> [||]

(* Whether a node's value is a container: a tuple, list, set or dict. *)
let container l (n : P.node) =
  match (l.interp.(n.id), n.kind) with
  | Data, Bytearray _ -> false
  | (Data | Odict | Counter _ | Pyset _ | Size _), _ -> true
  | _ -> false

(* [expose l order count] is, for each node the value holds, its children, and
   the number of its parents' edges. A storage counts the views and values over
   it, and the alignment and components they read it at. *)
let expose l (order : P.node array) count =
  let held = Array.make count None in
  let parents = Array.make count 0 in
  if Array.length order > 0 then held.(order.(0).id) <- Some (kids l order.(0));
  let use s dt =
    s.uses <- s.uses + 1;
    s.align <- Int.max s.align (alignment dt);
    if not (List.mem (part dt) s.parts) then s.parts <- part dt :: s.parts
  in
  let hold (n : P.node) =
    if Option.is_none held.(n.id) then held.(n.id) <- Some (kids l n)
  in
  Array.iter
    (fun (n : P.node) ->
      match held.(n.id) with
      | None -> ()
      | Some ks -> (
          Array.iter
            (function
              | P.Node c ->
                  parents.(c.id) <- parents.(c.id) + 1;
                  hold c
              | _ -> ())
            ks;
          match l.interp.(n.id) with
          | Storage_of r -> use r.storage (dtype_of l.ctx n.at r.dtype)
          | View v -> use v.sref.storage v.dt
          | Same d -> held.(d) <- Some [||]
          | _ -> ()))
    order;
  (held, parents)

(* [first_paths l root held parents count] numbers each container that two edges
   reach, in the order a depth-first walk of the value meets it, and gives the
   edge, a parent and a child's index, where the walk first meets it: there it
   is [Shared], and elsewhere a [Ref]. *)
let first_paths l root held parents count =
  let shared = Array.make count (-1) in
  let first = Array.make count (-1, -1) in
  let seen = Bytes.make count '\000' in
  let next = ref 0 in
  let stack = Stack.create () in
  (match root with
  | P.Node n ->
      Bytes.set seen n.id '\001';
      Stack.push (n, ref 0) stack
  | _ -> ());
  while not (Stack.is_empty stack) do
    let n, i = Stack.top stack in
    let ks = Option.value ~default:[||] held.(n.id) in
    if !i >= Array.length ks then ignore (Stack.pop stack)
    else begin
      let k = !i in
      incr i;
      match ks.(k) with
      | P.Node c when Bytes.get seen c.id = '\000' ->
          Bytes.set seen c.id '\001';
          if parents.(c.id) >= 2 && container l c then begin
            shared.(c.id) <- !next;
            incr next;
            first.(c.id) <- (n.id, k)
          end;
          Stack.push (c, ref 0) stack
      | _ -> ()
    end
  done;
  (shared, first)

(* Storages' memory *)

(* [own ~part src] is a copy of [src] in new host memory, with the bytes of each
   [part]-byte component reversed when [part] exceeds one: the elements in the
   other byte order. *)
let own ~part src =
  let n = B.length src in
  let dst = B.create Rig.host n in
  if part <= 1 then B.copy ~src ~dst
  else
    Storage.read src (fun s ->
        B.write dst Bigarray.int8_unsigned (fun d ->
            for i = 0 to (n / part) - 1 do
              for j = 0 to part - 1 do
                Bigarray.Array1.unsafe_set d
                  ((i * part) + j)
                  (Bigarray.Array1.unsafe_get s ((i * part) + part - 1 - j))
              done
            done));
  dst

(* [load_storage l ~swap s data] gives [s] its memory from its record [data]:
   mapped where the record is stored, in the host's byte order and aligned for
   every view of it, and read once into memory of its own otherwise. *)
let load_storage l ~swap s (data : Zip.data) =
  let class_ = dtype_of l.ctx 0 s.class_ in
  let b = match data with Zip.Stored b | Zip.Inflated b -> b in
  if B.length b <> s.numel * width class_ then
    Fail.failf "storage %S: %d bytes, %d for its persistent id" s.key
      (B.length b)
      (s.numel * width class_);
  let memory =
    match data with
    | Zip.Stored b when (not swap) && B.offset b mod s.align = 0 -> Mapped b
    | Zip.Stored b | Zip.Inflated b ->
        if not swap then Own (own ~part:1 b)
        else if List.for_all (( = ) (part class_)) s.parts then
          Own (own ~part:(part class_) b)
        else
          Fail.failf
            "storage %S of another byte order than the host's, read at a dtype \
             whose bytes the file does not order"
            s.key
  in
  s.memory <- Some memory

let tensor (v : view) =
  let s = v.sref.storage in
  let layout = layout v in
  let (D.Any dt) = match v.dt with Nx d -> d | Raw _ -> D.Any D.Uint8 in
  match Option.get s.memory with
  | Mapped b -> Nx.P (Storage.value b dt layout)
  | Own b ->
      let a = A.v dt layout b in
      (* One view owns the memory, which its donation may write; several share
         it, and theirs write new memory. *)
      if s.uses = 1 then Nx.P (Nx.Repr.unsafe_of_array Nx.Host.v a)
      else Nx.P (Nx.Repr.of_array Nx.Host.v a)

(* A storage held as a value: a vector of its elements. *)
let storage_value ctx at (r : sref) =
  let dt = dtype_of ctx at r.dtype in
  let n = r.length / width dt in
  tensor
    {
      sref = r;
      dt;
      offset = r.first / width dt;
      shape = [| n |];
      strides = [| 1 |];
    }

(* Building *)

let scalar = function
  | P.Null -> Null
  | P.Bool b -> Bool b
  | P.Int i -> Int i
  | P.Long (s, _) -> Long s
  | P.Float f -> Float f
  | P.String (s, _) -> String s
  | P.Bytes (s, _) -> Bytes s
  | P.Global (g, _) -> Global g
  | P.Node _ -> assert false

let distinct l at key items =
  let seen = Hashtbl.create ~random:true (Array.length items) in
  Array.iter
    (fun k ->
      match key k with
      | k when k = nan_key -> fail l.ctx at "a NaN key, which no key equals"
      | k when k < 0 ->
          fail l.ctx at
            "a key or set element whose equality Python's class decides"
      | k ->
          if Hashtbl.mem seen k then
            fail l.ctx at "a key or set element given twice";
          Hashtbl.add seen k ())
    items

(* [values l order count held shared first key] is each held node's value,
   children first: a container two edges reach is [Shared] at the edge where a
   depth-first walk first meets it and a [Ref] at the others. *)
let values l (order : P.node array) count held shared first key =
  let values = Array.make count Null in
  for i = Array.length order - 1 downto 0 do
    let n = order.(i) in
    match held.(n.id) with
    | None -> ()
    | Some ks ->
        let edge j = function
          | P.Node c when shared.(c.id) >= 0 ->
              if first.(c.id) = (n.id, j) then
                Shared (shared.(c.id), values.(c.id))
              else Ref shared.(c.id)
          | P.Node c -> values.(c.id)
          | v -> scalar v
        in
        let all () = Array.to_list (Array.mapi edge ks) in
        let pairs () =
          List.init
            (Array.length ks / 2)
            (fun j ->
              (edge (2 * j) ks.(2 * j), edge ((2 * j) + 1) ks.((2 * j) + 1)))
        in
        let dict () =
          distinct l n.at key
            (Array.init (Array.length ks / 2) (fun j -> ks.(2 * j)));
          Dict (pairs ())
        in
        let set () =
          distinct l n.at key ks;
          Set (all ())
        in
        values.(n.id) <-
          (match (l.interp.(n.id), n.kind) with
          | Data, Tuple _ -> Tuple (all ())
          | Data, (Frozenset _ | Set _) -> set ()
          | Data, Bytearray s -> Bytes s
          | Data, List _ -> List (all ())
          | Data, Dict _ -> dict ()
          | Data, (Persistent _ | Call _) -> assert false
          | Storage_of r, _ -> Tensor (storage_value l.ctx n.at r)
          | View view, _ -> Tensor (tensor view)
          | Same d, _ -> values.(d)
          | (Odict | Counter _), _ -> dict ()
          | Pyset _, _ -> set ()
          | Size _, _ -> Tuple (all ())
          | Complex_of c, _ -> Complex c
          | (Bytes_of s | Bytearray_of s), _ -> Bytes s
          | Opaque name, _ -> Object name)
  done;
  values

(* [value l ~records ~swap d] is the value of the decoded pickle [d], whose
   storages [records] gives. *)
let value l ~records ~swap (d : P.decoded) =
  match d.value with
  | P.Node root ->
      let order = order d.value d.nodes in
      for i = Array.length order - 1 downto 0 do
        let n = order.(i) in
        l.interp.(n.id) <- interpret l n
      done;
      exclusive l order d.nodes;
      let key = keys l order d.nodes in
      let held, parents = expose l order d.nodes in
      let shared, first = first_paths l d.value held parents d.nodes in
      Hashtbl.iter
        (fun _ s -> if s.uses > 0 then load_storage l ~swap s (records s.key))
        l.storages;
      (values l order d.nodes held shared first key).(root.id)
  | s -> scalar s

let loader ~ctx ~legacy (d : P.decoded) =
  {
    ctx;
    legacy;
    decoded = d;
    storages = Hashtbl.create ~random:true 64;
    views = Hashtbl.create ~random:true 8;
    interp = Array.make d.nodes Data;
  }

(* Containers *)

let zip_signature = "PK\003\004"

(* The legacy format's magic number, 0x1950A86A20F9469CFC6C, as LONG1's bytes,
   and its version ([torch/serialization.py]). *)
let magic = "\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19"
let legacy_version = 1001

(* A tar archive's [ustar] mark, at byte 257 of its first header. *)
let tar_mark = "ustar"
let tar_mark_at = 257

let decode ~ctx b ~pos =
  try Storage.read b (fun data -> P.decode data ~pos)
  with Failure m -> Fail.failf "%s: %s" ctx m

let load_zip file =
  let zip = Zip.open_in file in
  let prefix =
    match Zip.entries zip with
    | [] -> failwith "a zip archive with no entry"
    | e :: _ -> (
        match String.index_opt (Zip.name e) '/' with
        | Some i -> String.sub (Zip.name e) 0 (i + 1)
        | None ->
            failwith "a zip archive whose first entry lies in no directory")
  in
  let find name = Zip.find zip (prefix ^ name) in
  if Option.is_some (find "constants.pkl") then
    failwith "a TorchScript archive, which torch.jit.load reads";
  let read name =
    match find name with
    | Some e -> Zip.read zip e
    | None -> Fail.failf "no %s entry" (prefix ^ name)
  in
  let bytes = function Zip.Stored b | Zip.Inflated b -> b in
  let swap =
    match find "byteorder" with
    | None -> Sys.big_endian
    | Some e -> (
        let b = bytes (Zip.read zip e) in
        match Storage.string b ~pos:0 (B.length b) with
        | "little" -> Sys.big_endian
        | "big" -> not Sys.big_endian
        | s -> Fail.failf "the byte order %S" s)
  in
  let pickle = bytes (read "data.pkl") in
  let d = decode ~ctx:"data.pkl" pickle ~pos:0 in
  if d.stop <> B.length pickle then
    Fail.failf "data.pkl: byte %d: bytes after its STOP" d.stop;
  let l = loader ~ctx:"data.pkl" ~legacy:false d in
  value l ~records:(fun key -> read ("data/" ^ key)) ~swap d

let load_legacy file =
  let pages = Storage.pages file in
  let next pos what =
    let d = decode ~ctx:(Printf.sprintf "the %s's pickle" what) pages ~pos in
    (d, d.stop)
  in
  let m, pos = next 0 "magic number" in
  (match m.value with
  | P.Long (b, _) when b = magic -> ()
  | _ -> failwith "neither a zip archive nor PyTorch's legacy format");
  let v, pos = next pos "version" in
  if v.value <> P.Int legacy_version then
    failwith "a legacy format version other than 1001";
  let _, pos = next pos "system information" in
  let d, pos = next pos "value" in
  let keys, pos = next pos "storage keys" in
  let keys =
    match keys.value with
    | P.Node { kind = List ks; _ } ->
        List.map
          (function
            | P.String (k, _) -> k
            | _ -> failwith "a storage key that is no string")
          (Dynarray.to_list ks)
    | _ -> failwith "storage keys that are no list"
  in
  let l = loader ~ctx:"the value's pickle" ~legacy:true d in
  (* The storages follow in the keys' order, each its element count as a
     little-endian int64 then its elements, little-endian. Their lengths are
     their persistent ids', known once the value's pickle is read. *)
  let records =
    lazy
      (let records = Hashtbl.create ~random:true 64 in
       let stop =
         List.fold_left
           (fun pos key ->
             let s =
               match Hashtbl.find_opt l.storages key with
               | Some s -> s
               | None -> Fail.failf "storage %S, which the value names not" key
             in
             if Hashtbl.mem records key then
               Fail.failf "storage %S listed twice" key;
             let count = String.get_int64_le (Storage.string pages ~pos 8) 0 in
             if count <> Int64.of_int s.numel then
               Fail.failf "storage %S: %Ld elements, %d for its persistent id"
                 key count s.numel;
             let length = s.numel * width (dtype_of l.ctx 0 s.class_) in
             if length > B.length pages - pos - 8 then
               Fail.failf "storage %S: the file ends inside it" key;
             Hashtbl.add records key (B.view pages ~first:(pos + 8) ~length);
             pos + 8 + length)
           pos keys
       in
       if stop <> B.length pages then
         Fail.failf "byte %d: bytes after the last storage" stop;
       records)
  in
  let record key =
    match Hashtbl.find_opt (Lazy.force records) key with
    | Some b -> Zip.Stored b
    | None -> Fail.failf "storage %S, which the storage keys do not list" key
  in
  value l ~records:record ~swap:Sys.big_endian d

let load file =
  let length = B.length file in
  let head n = if length >= n then Storage.string file ~pos:0 n else "" in
  if head 4 = zip_signature then load_zip file
  else if
    length >= tar_mark_at + String.length tar_mark
    && Storage.string file ~pos:tar_mark_at (String.length tar_mark) = tar_mark
  then failwith "the tar archive of PyTorch before 0.1.10"
  else load_legacy file

(* Naming tensors *)

(* [tensors v] walks [v] with an explicit stack, depth first, as [build] laid
   out [Shared] and [Ref]. A path is its components in reverse, or [None] under
   a key that names nothing or in a set. Each [Shared] records, once walked,
   whether it holds a tensor, which a later [Ref] to it may not. *)
let tensors v =
  let name path = String.concat "." (List.rev path) in
  let fail path fmt =
    Fail.failf
      ("Nx_io.Pytorch.tensors: %s: " ^^ fmt)
      (match path with Some [] | None -> "the value" | Some p -> name p)
  in
  let stack = Stack.create () in
  Stack.push (`Walk (Some [], v)) stack;
  let archive = ref Archive.empty in
  let found = ref 0 in
  let holds = Hashtbl.create 16 in
  let under path c = Option.map (List.cons c) path in
  while not (Stack.is_empty stack) do
    match Stack.pop stack with
    | `Close (id, before) -> Hashtbl.replace holds id (!found > before)
    | `Walk (path, v) -> (
        match v with
        | Tensor x -> (
            incr found;
            match path with
            | None ->
                fail path "a tensor under a key that names nothing, or in a set"
            | Some [] -> fail path "a tensor with the empty name"
            | Some p ->
                let n = name p in
                if Archive.find n !archive <> None then
                  fail path "two tensors have this name";
                archive := Archive.add n x !archive)
        | Object o ->
            fail path "%s, an object, whose call may have been given a tensor" o
        | Shared (id, x) ->
            Stack.push (`Close (id, !found)) stack;
            Stack.push (`Walk (path, x)) stack
        | Ref id -> (
            match Hashtbl.find_opt holds id with
            | Some false -> ()
            | Some true ->
                fail path "a reference to shared value %d, which holds a tensor"
                  id
            | None ->
                fail path "a reference to shared value %d, outside the value" id
            )
        | Dict kvs ->
            List.iter
              (fun (k, x) ->
                let c =
                  match k with
                  | String s -> under path s
                  | Int i -> under path (string_of_int i)
                  | _ -> None
                in
                Stack.push (`Walk (c, x)) stack;
                Stack.push (`Walk (None, k)) stack)
              (List.rev kvs)
        | List xs | Tuple xs ->
            let n = List.length xs in
            List.iteri
              (fun i x ->
                let c = under path (string_of_int (n - 1 - i)) in
                Stack.push (`Walk (c, x)) stack)
              (List.rev xs)
        | Set xs ->
            List.iter
              (fun x -> Stack.push (`Walk (None, x)) stack)
              (List.rev xs)
        | Null | Bool _ | Int _ | Long _ | Float _ | Complex _ | String _
        | Bytes _ | Global _ ->
            ())
  done;
  !archive
