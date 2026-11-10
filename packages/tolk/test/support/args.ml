(* Nodes of every kind of argument, for the laws of their text and their key.
   Each field is drawn from few values, among them values a repr prints alike:
   NaNs of different bits, a stage a custom kernel's argument makes beside one
   the user materialises, targets whose fields split one text two ways, and
   nodes in arguments that differ only there. *)

open Windtrap
open Tolk

let few l = Gen.of_list l
let lit k = Ops.v ~arg:(Const (`Int (Bigint.of_int k))) Const
let cast dt u = Ops.v ~src:[ u ] ~arg:(Dtype dt) Cast
let x = cast Float32 (lit 1)
let half = cast Float16 (lit 1)

let n =
  Ops.v
    ~arg:
      (Param
         (Ops.param_arg ~slot:(-1)
            ~vmin_vmax:(`Int Bigint.zero, `Int (Bigint.of_int 9))
            ~name:"n" ~addrspace:Alu Weak_int))
    Param

let buffer =
  Ops.v
    ~src:[ Ops.size_src (Some 16) ]
    ~arg:(Param (Ops.param_arg ~slot:0 ~device:"CPU" Float32))
    Buffer

let quiet = Int64.float_of_bits 0x7ff8000000000000L
let signalling = Int64.float_of_bits 0x7ff8000000000001L

(* A scalar parameter bounded by [v], which prints as [nan] for both NaNs. *)
let bounded v =
  Ops.v ~src:[ Ops.size_src None ]
    ~arg:
      (Param
         (Ops.param_arg ~vmin_vmax:(`Float 0., `Float v) ~bound:(`Float v)
            ~addrspace:Alu ~slot:1 Float32))
    Param

let built f = match f () with r -> r | exception Invalid_argument _ -> reject ()

(* Fields *)

let a_dtype =
  Gen.of_list ~pp:Dtype.pp Dtype.[ Bool; Uint8; Float16; Float32 ]

let a_float =
  Gen.frequency
    [
      (3, few [ 0.; -0.; 0.1; Float.infinity; quiet; signalling ]);
      (1, Gen.any_float);
    ]

let a_value : Dtype.value Gen.t =
  Gen.one_of
    [
      Gen.map (fun b -> `Bool b) Gen.bool;
      Gen.map
        (fun z -> `Int z)
        (few Bigint.[ zero; minus_one; shift_left one 70 ]);
      Gen.map (fun x -> `Float x) a_float;
    ]

let a_const : Dtype.const Gen.t =
  Gen.frequency
    [
      (6, Gen.map (fun v -> (v :> Dtype.const)) a_value);
      (1, Gen.constant `Invalid);
    ]

let a_string = few [ ""; "a"; "b\"\n" ]
let an_axis = Gen.of_list Ops.Axis_type.[ Global; Local; Warp; Weak; Loop; Upcast ]
let ids = Gen.list ~size:(Gen.int_range 1 2) (few [ -1; 0; 3 ])
let a_node_in_arg = few [ n; bounded quiet; bounded signalling ]

let an_sint =
  Gen.one_of
    [
      Gen.map (fun k -> Ops.Int k) (few [ 1; 4 ]);
      Gen.map (fun u -> Ops.Sym u) a_node_in_arg;
    ]

let an_opt : Opt.t Gen.t =
  let open Gen in
  let* axis = few [ 0; 1 ] in
  one_of
    [
      (let+ tc_select, tc_opt, use_tc =
         triple (few [ -1; 0 ]) (few [ 0; 2 ]) (few [ 1; 2 ])
       in
       Opt.Tc { axis; tc_select; tc_opt; use_tc });
      (let+ amount, target, top =
         triple (few [ 0; 4 ]) (of_list Opt.[ Upcast; Local ]) bool
       in
       Opt.Split { axis; amount; target; top });
      map (fun amount -> Opt.Padto { axis; amount }) (few [ 4; 32 ]);
      map (fun with_axis -> Opt.Swap { axis; with_axis }) (few [ 0; 2 ]);
    ]

let opts = Gen.list ~size:(Gen.int_range 0 2) an_opt
let sizes = Gen.list ~size:(Gen.int_range 1 3) an_sint
let slots = Gen.list ~size:(Gen.int_range 0 2) (few [ 0; 1 ])

let estimates =
  Gen.option
    (Gen.map
       (fun (ops, lds, mem) : Ops.estimates -> { ops; lds; mem })
       (Gen.triple an_sint an_sint an_sint))

let split =
  Gen.option
    (Gen.map
       (fun (iterations, lo) : Ops.split -> { iterations; lo; hi = lo + 1 })
       (Gen.pair an_sint (few [ 0; 3 ])))

let marks = Gen.list ~size:(Gen.int_range 0 2) (Gen.pair (few [ 0; 1 ]) (few [ 0; 2 ]))

let wmma_axes =
  Gen.list ~size:(Gen.int_range 0 2) (Gen.pair (Gen.pair an_axis ids) (few [ 2; 4 ]))

(* Kinds *)

(* A kind of argument: a draw of what its node holds, a redraw of each of its
   fields by name, and the node. *)
type 'r kind = {
  draw : 'r Gen.t;
  fields : (string * ('r -> 'r Gen.t)) list;
  node : 'r -> Ops.t;
}

type any = Kind : 'r kind -> any

let field name g set = (name, fun r -> Gen.map (set r) g)

let param_arg =
  let open Gen in
  let* slot, dtype, (vmin_vmax, name) =
    triple (few [ -1; 0; 2 ]) a_dtype
      (pair (option (pair a_value a_value)) (option a_string))
  in
  let* addrspace, device, (volatile, bind_on_realize) =
    triple
      (of_list Dtype.[ Global; Local; Reg; Alu ])
      (option (few [ "CPU"; "DISK:x" ]))
      (pair bool bool)
  in
  let+ bound, phase, align =
    triple (option a_value) (few [ 0; 8 ]) (few [ 16; 8; 1 ])
  in
  built (fun () ->
      Ops.param_arg ?vmin_vmax ?name ~addrspace ?device ~volatile
        ~bind_on_realize ?bound ~phase ~align ~slot dtype)

let param =
  let f name g set =
    field ("param." ^ name) g (fun (o, (p : Ops.param_arg), s) v ->
        (o, set p v, s))
  in
  Kind
    {
      draw =
        Gen.triple
          (Gen.of_list [ Op.Param; Op.Buffer; Op.Alloc ])
          param_arg
          (Gen.option (few [ 1; 16 ]));
      fields =
        [
          f "slot" (few [ -1; 0; 2 ]) (fun p slot -> { p with slot });
          f "dtype" a_dtype (fun p dtype -> { p with dtype });
          f "vmin_vmax"
            Gen.(option (pair a_value a_value))
            (fun p vmin_vmax -> { p with vmin_vmax });
          f "name" (Gen.option a_string) (fun p name -> { p with name });
          f "addrspace"
            (Gen.of_list Dtype.[ Global; Local; Reg; Alu ])
            (fun p addrspace -> { p with addrspace });
          f "device"
            (Gen.option (few [ "CPU"; "DISK:x" ]))
            (fun p device -> { p with device });
          f "volatile" Gen.bool (fun p volatile -> { p with volatile });
          f "bind_on_realize" Gen.bool (fun p bind_on_realize ->
              { p with bind_on_realize });
          f "bound" (Gen.option a_value) (fun p bound -> { p with bound });
          f "phase" (few [ 0; 8 ]) (fun p phase -> { p with phase });
          f "align" (few [ 16; 8; 1 ]) (fun p align -> { p with align });
        ];
      node =
        (fun (o, p, s) ->
          built (fun () -> Ops.v ~src:[ Ops.size_src s ] ~arg:(Param p) o));
    }

let kernel =
  let f name g set =
    field ("kernel." ^ name) g (fun (k : Ops.kernel_info) v -> set k v)
  in
  Kind
    {
      draw =
        Gen.(
          let* name, applied_opts, opts_to_apply =
            triple (few [ "test"; "r_4" ]) opts (option opts)
          in
          let* estimates, beam = pair estimates (few [ 0; 2 ]) in
          let+ optimized, kernel, split = triple bool bool split in
          Ops.kernel_info ~name ~applied_opts ?opts_to_apply ?estimates ~beam
            ~optimized ~kernel ?split ());
      fields =
        [
          f "name" (few [ "test"; "r_4" ]) (fun k name -> { k with name });
          f "applied_opts" opts (fun k applied_opts -> { k with applied_opts });
          f "opts_to_apply" (Gen.option opts) (fun k opts_to_apply ->
              { k with opts_to_apply });
          f "estimates" estimates (fun k estimates -> { k with estimates });
          f "beam" (few [ 0; 2 ]) (fun k beam -> { k with beam });
          f "optimized" Gen.bool (fun k optimized -> { k with optimized });
          f "kernel" Gen.bool (fun k kernel -> { k with kernel });
          f "split" split (fun k split -> { k with split });
        ];
      node = (fun k -> Ops.sink ~kernel:k [ x ]);
    }

let target_fields =
  [
    ("device", few [ "CPU"; "CUDA" ], fun (t : Helpers.Target.t) device ->
      { t with device });
    ("renderer", few [ "CLANG"; "" ], fun t renderer -> { t with renderer });
    ("arch", few [ ""; "sm_89" ], fun t arch -> { t with arch });
    ("interface", few [ ""; "a"; "a:b" ], fun t interface -> { t with interface });
    ("indices", few [ ""; "b" ], fun t indices -> { t with indices });
  ]

let program =
  let f name g set =
    field ("program." ^ name) g (fun (p : Ops.program_info) v -> set p v)
  in
  let no_target : Helpers.Target.t =
    { device = ""; renderer = ""; arch = ""; interface = ""; indices = "" }
  in
  let target =
    List.fold_left
      (fun t (_, g, set) ->
        Gen.(
          let* t = t in
          map (set t) g))
      (Gen.constant no_target) target_fields
  in
  Kind
    {
      draw =
        Gen.(
          let* global_size, local_size, params =
            triple sizes sizes (list ~size:(int_range 0 1) a_node_in_arg)
          in
          let* globals, outs, ins = triple slots slots slots in
          let+ target = target in
          ({ global_size; local_size; params; globals; outs; ins; target }
            : Ops.program_info));
      fields =
        [
          f "global_size" sizes (fun p global_size -> { p with global_size });
          f "local_size" sizes (fun p local_size -> { p with local_size });
          f "params"
            (Gen.list ~size:(Gen.int_range 0 1) a_node_in_arg)
            (fun p params -> { p with params });
          f "globals" slots (fun p globals -> { p with globals });
          f "outs" slots (fun p outs -> { p with outs });
          f "ins" slots (fun p ins -> { p with ins });
        ]
        @ List.map
            (fun (name, g, set) ->
              f ("target." ^ name) g (fun p v ->
                  { p with target = set p.target v }))
            target_fields;
      node = (fun p -> Ops.v ~arg:(Program p) Program);
    }

let wmma =
  let f name g set = field ("wmma." ^ name) g (fun (w : Ops.wmma) v -> set w v) in
  let dims = few [ (8, 16, 16); (16, 8, 16) ] and threads = few [ 32; 8 ] in
  let upcast = Gen.option (Gen.triple wmma_axes wmma_axes wmma_axes) in
  Kind
    {
      draw =
        Gen.(
          let+ (dims, dtype_in, threads), upcast_axes =
            pair (triple dims a_dtype threads) upcast
          in
          ({ dims; dtype_in; threads; upcast_axes } : Ops.wmma));
      fields =
        [
          f "dims" dims (fun w dims -> { w with dims });
          f "dtype_in" a_dtype (fun w dtype_in -> { w with dtype_in });
          f "threads" threads (fun w threads -> { w with threads });
          f "upcast_axes" upcast (fun w upcast_axes -> { w with upcast_axes });
        ];
      node =
        (fun w -> built (fun () -> Ops.v ~src:[ half; half; x ] ~arg:(Wmma w) Wmma));
    }

let call =
  let f name g set =
    field ("call." ^ name) g (fun (c : Ops.call_info) v -> set c v)
  in
  Kind
    {
      draw =
        Gen.(
          let+ (name, precompile, inline), write =
            pair (triple (option a_string) bool bool) marks
          in
          ({ name; precompile; inline; write } : Ops.call_info));
      fields =
        [
          f "name" (Gen.option a_string) (fun c name -> { c with name });
          f "precompile" Gen.bool (fun c precompile -> { c with precompile });
          f "inline" Gen.bool (fun c inline -> { c with inline });
          f "write" marks (fun c write -> { c with write });
        ];
      node =
        (fun c -> Ops.v ~src:[ Ops.sink [ x ]; buffer ] ~arg:(Call c) Call);
    }

let stage =
  let f name g set =
    field ("stage." ^ name) g (fun (b : Ops.bufferize_opts) v -> set b v)
  in
  let device = Gen.option (Gen.constant "CPU") in
  let addrspace = Gen.of_list Dtype.[ Global; Local ] in
  let keep = Gen.of_list Ops.[ Removable; Broadcast; Whole; Argument ] in
  Kind
    {
      draw =
        Gen.(
          let+ device, addrspace, keep = triple device addrspace keep in
          ({ device; addrspace; keep } : Ops.bufferize_opts));
      fields =
        [
          f "device" device (fun b device -> { b with device });
          f "addrspace" addrspace (fun b addrspace -> { b with addrspace });
          f "keep" keep (fun b keep -> { b with keep });
        ];
      node = (fun b -> Ops.v ~src:[ buffer ] ~arg:(Bufferize b) Stage);
    }

(* A kind whose node holds two fields. *)
let two prefix (n0, g0) (n1, g1) node =
  Kind
    {
      draw = Gen.pair g0 g1;
      fields =
        [
          field (prefix ^ "." ^ n0) g0 (fun (_, b) a -> (a, b));
          field (prefix ^ "." ^ n1) g1 (fun (a, _) b -> (a, b));
        ];
      node;
    }

(* A kind whose node holds one field. *)
let one name g node =
  Kind { draw = g; fields = [ field name g (fun _ v -> v) ]; node }

let kinds =
  [
    one "const" a_const (fun c -> Ops.v ~arg:(Const c) Const);
    one "cast" a_dtype (fun dt -> cast dt (lit 1));
    param;
    two "range" ("axis_type", an_axis) ("axis_id", ids)
      (fun (axis_type, axis_id) ->
        Ops.v ~src:[ lit 16 ] ~arg:(Range { axis_type; axis_id }) Range);
    two "reduce"
      ("op", Gen.of_list Op.[ Add; Mul; Max ])
      ("num_axes", few [ 0; 1 ])
      (fun (op, num_axes) ->
        Ops.v ~src:[ x ] ~arg:(Reduce { op; num_axes }) Reduce);
    one "copy" a_string (fun d -> Ops.v ~src:[ x ] ~arg:(Device d) Copy);
    one "special" a_string (fun s ->
        Ops.v ~src:[ lit 4 ] ~arg:(String s) Special);
    one "source" a_string (fun s -> Ops.v ~arg:(String s) Source);
    one "linear" a_string (fun s -> Ops.v ~src:[ x ] ~arg:(String s) Linear);
    one "binary" (few [ ""; "\x7fELF\x00" ]) (fun s ->
        Ops.v ~arg:(Bytes s) Binary);
    two "ins" ("code", a_string) ("dtype", a_dtype) (fun (code, dtype) ->
        Ops.v ~arg:(Code { code; dtype }) Ins);
    stage;
    kernel;
    program;
    Kind
      {
        draw = Gen.triple a_string a_dtype Gen.bool;
        fields =
          [
            field "custom.name" a_string (fun (_, d, f) s -> (s, d, f));
            field "custom.dtype" a_dtype (fun (s, _, f) d -> (s, d, f));
            field "custom.fmt" Gen.bool (fun (s, d, _) f -> (s, d, f));
          ];
        node =
          (fun (name, dtype, fmt) ->
            Ops.call (Ops.custom_function ~dtype ~fmt name []) [ x ]);
      };
    call;
    wmma;
    Kind
      {
        draw = Gen.constant ();
        fields = [];
        node =
          (fun () ->
            Ops.v ~src:[ buffer; Ops.store buffer x ] ~arg:Disjoint After);
      };
  ]

let node =
  Gen.with_pp Ops.pp
    (Gen.bind (Gen.of_list kinds) (fun (Kind k) -> Gen.map k.node k.draw))

type field = Field : 'r kind * string * ('r -> 'r Gen.t) -> field

let fields =
  List.concat_map
    (fun (Kind k) -> List.map (fun (name, g) -> Field (k, name, g)) k.fields)
    kinds

let field_names = List.map (fun (Field (_, name, _)) -> name) fields

let neighbours =
  let pp ppf (name, u, v) =
    Format.fprintf ppf "%s:@.%a@.%a" name Ops.pp u Ops.pp v
  in
  Gen.with_pp pp
    (Gen.bind (Gen.of_list fields) (fun (Field (k, name, redraw)) ->
         Gen.bind k.draw (fun r ->
             Gen.map (fun r' -> (name, k.node r, k.node r')) (redraw r))))

(* Two nodes of one kind, drawn apart. *)
let pairs =
  let pp ppf (u, v) = Format.fprintf ppf "%a@.%a" Ops.pp u Ops.pp v in
  Gen.with_pp pp
    (Gen.bind (Gen.of_list kinds) (fun (Kind k) ->
         Gen.map (fun (r, r') -> (k.node r, k.node r')) (Gen.pair k.draw k.draw)))
