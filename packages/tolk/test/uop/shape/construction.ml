(* Construction: constants, stacks, movement, aranges, storage, variables,
   divisibility, calls and programs. *)

open Windtrap
open Tolk
open Common

let cpu = "CPU"
let param ?device shape slot dt = Shape.param ?device ~shape:(ints shape) slot dt

(* Constants and stacks *)

let constants =
  group "constants"
    [
      test "consts takes the committed type of its literals" (fun () ->
          equal dtype Int32 (Ops.dtype (Shape.consts [ i 1; i 2 ]));
          equal dtype Int64
            (Ops.dtype
               (Shape.consts [ i 1; `Int (Bigint.shift_left Bigint.one 40) ]));
          equal dtype Float32 (Ops.dtype (Shape.consts [ i 1; f 2.5 ]));
          equal dtype Bool (Ops.dtype (Shape.consts [ `Bool true; `Bool false ])));
      test "const_like expands to the node's shape, in its type" (fun () ->
          let p = param [ 2; 3 ] 0 Float16 in
          let c = Shape.const_like p (i 1) in
          equal dtype Float16 (Ops.dtype c);
          equal shape (ints [ 2; 3 ]) (Shape.shape c);
          equal uop (Ops.int ~dtype:Float16 1) (Ops.base c));
    ]

let kernel_nodes =
  group "kernel nodes"
    [
      test "stack rejects nothing to stack and differing shapes" (fun () ->
          rejects (fun () -> Shape.stack []);
          rejects (fun () ->
              Shape.stack [ param [ 2 ] 0 Float32; param [ 3 ] 1 Float32 ]));
      test "contract rejects a range that is not upcast" (fun () ->
          let r = Ops.range ~axis_type:Weak (Int 2) [ 0 ] in
          rejects (fun () -> Shape.contract Ops.O.(r + int 1) [ r ]));
    ]

(* Movement *)

(* Up to three axes of up to four elements, some of them named in either
   order. *)
let gen_flip =
  Gen.(
    let* dims = list ~size:(int_range 1 3) (int_range 0 4) in
    let+ flags = list ~size:(constant (List.length dims)) bool
    and+ backwards = bool in
    let axes =
      List.filter_map Fun.id
        (List.mapi (fun a f -> if f then Some a else None) flags)
    in
    (dims, if backwards then List.rev axes else axes))

(* The row-major position in [dims] of the element that output position [i]
   holds once [axes] are reversed. *)
let flipped dims axes i =
  let rank = List.length dims in
  let rec coords a i acc =
    if a < 0 then acc
    else
      let d = List.nth dims a in
      let j = if List.mem a axes then d - 1 - (i mod d) else i mod d in
      coords (a - 1) (i / d) (j :: acc)
  in
  List.fold_left2 (fun acc j d -> (acc * d) + j) 0 (coords (rank - 1) i []) dims

(* Aranges of any length *)

(* The values of [u]'s elements [lo] to [lo + k - 1]: what the one kernel that
   stores them into storage of [k] elements writes. It reads no other element,
   so an arange of billions of elements is read without its memory; a schedule
   that stores more fails before it is run. *)
let window u lo k =
  let out = Ops.new_buffer ~slot:0 cpu k (Ops.dtype u) in
  let view = Shape.shrink u [ Some (Int lo, Int (lo + k)) ] in
  let sink = Ops.sink [ Ops.after out [ Ops.store out view ] ] in
  let linear, _ = Schedule.create_linear_with_vars ~cores:Cores.host sink in
  equal ~msg:"kernels" int 1 (List.length (Ops.src linear));
  List.iter
    (fun b ->
      if Ops.op b = Op.Buffer then
        at_most ~msg:"stored elements" int ~than:k (Shape.max_numel b))
    (Ops.toposort ~calls:Skip linear);
  List.map (fun (_, _, v) -> v) (Kernel_graphs.linear_writes linear)

(* Short lengths, lengths about the powers of two where an element's index or
   value leaves 32 bits, and past them up to the longest. *)
let long_lengths =
  [
    3;
    1000;
    (1 lsl 30) - 1;
    1 lsl 30;
    (1 lsl 30) + 1;
    1518500249;
    1518500250;
    (1 lsl 31) - 1;
    1 lsl 31;
    (1 lsl 31) + 1;
    (1 lsl 32) + 3;
    (1 lsl 46) + 5;
    (1 lsl 60) + 7;
    max_int - (1 lsl 30);
  ]

(* The longest arange that is a running sum. *)
let pooled_run = 1518500249

(* The integer [v] in [dt]: rounded once to a float type, whose double holds
   it exactly below [2^53]. *)
let element dt v : Dtype.value =
  match dt with
  | Dtype.Float32 ->
      `Float (Int32.float_of_bits (Int32.bits_of_float (Float.of_int v)))
  | Float64 | Float16 | Bfloat16 -> `Float (Float.of_int v)
  | _ -> `Int (Bigint.of_int v)

(* An arange of [n] elements from [start] by [step], the type it is asked for
   or [None], and the first of three elements read. An integer type, when
   given, holds the elements, and a float type is given where it holds them
   exactly or the arange is laid out in rows, its elements below [2^53]; the
   weak integer holds them all. Past [2^46] elements, a step of [1] keeps the
   arange within an int. *)
let gen_long_arange =
  let open Gen in
  let pp_int = Format.pp_print_int in
  let* n = of_list ~pp:pp_int long_lengths in
  let* step =
    of_list ~pp:pp_int (if n > 1 lsl 46 then [ 1; -1 ] else [ 1; -1; 3; -7 ])
  in
  let span = (n - 1) * abs step in
  let* start =
    of_list ~pp:pp_int
      (if step > 0 then [ 0; -5; -(span / 2); min_int ]
       else [ span; span + 9; max_int ])
  in
  let last = start + ((n - 1) * step) in
  let lo' = Stdlib.min start last and hi' = Stdlib.max start last in
  let below m = -m <= lo' && hi' <= m in
  let holds dt =
    match dt with
    | Dtype.Float32 ->
        below (1 lsl 24) || (n > pooled_run && below ((1 lsl 53) - 1))
    | Float64 -> below (1 lsl 53)
    | Float16 -> below (1 lsl 11)
    | Bfloat16 -> below (1 lsl 8)
    | Weak_int -> true
    | _ ->
        let lo, hi = Dtypes.int_bounds dt in
        Bigint.(leq lo (of_int lo') && leq (of_int hi') hi)
  in
  let pp_type ppf = function
    | None -> Format.pp_print_string ppf "default"
    | Some dt -> Dtype.pp ppf dt
  in
  let* dtype =
    of_list ~pp:pp_type
      (None
      :: List.filter_map
           (fun dt -> if holds dt then Some (Some dt) else None)
           Dtype.
             [
               Int8;
               Uint8;
               Int16;
               Uint16;
               Int32;
               Uint32;
               Int64;
               Uint64;
               Weak_int;
               Float16;
               Bfloat16;
               Float32;
               Float64;
             ])
  in
  let+ first =
    of_list ~pp:pp_int
      (List.map (min (n - 3))
         [ 0; n - 3; (1 lsl 31) - 2; (1 lsl 32) - 2; n / 2 ])
  in
  (n, start, step, dtype, first)

let long_aranges =
  group "aranges"
    [
      prop "an arange's element i is start + i * step at any length" ~count:200
        gen_long_arange (fun (n, start, step, dtype, first) ->
          cover "a length past 2^30.5" (n > pooled_run);
          cover "a length past 2^31" (n > 1 lsl 31);
          cover "a length past 2^32" (n > 1 lsl 32);
          cover "a descending arange" (step < 0);
          cover "a start at an int's bounds"
            (start = min_int || start = max_int);
          cover "an unsigned type"
            (Option.fold ~none:false ~some:Dtype.is_unsigned dtype);
          cover "a float type past 2^30.5"
            (n > pooled_run
            && Option.fold ~none:false ~some:Dtype.is_float dtype);
          cover "elements across 2^31"
            (first < 1 lsl 31 && first + 3 > 1 lsl 31);
          cover "the weak integer" (dtype = Some Dtype.Weak_int);
          let u = Shape.arange ~start ~step ?dtype (start + (n * step)) in
          equal shape (ints [ n ]) (Shape.shape u);
          (* A weak arange is stored at the type that holds an int. *)
          let u = if dtype = Some Weak_int then Ops.cast u Int64 else u in
          equal (list value)
            (List.init 3 (fun j ->
                 element (Ops.dtype u) (start + ((first + j) * step))))
            (window u first 3));
    ]

let movement =
  let p = param [ 2; 3; 4 ] 0 Float32 in
  group "movement"
    [
      test "reshape infers a -1 and rejects two" (fun () ->
          equal shape
            (ints [ 4; 6 ])
            (Shape.shape (Shape.reshape p (ints [ 4; -1 ])));
          rejects (fun () -> Shape.reshape p (ints [ -1; -1 ])));
      test "reshape rejects a different number of elements" (fun () ->
          rejects (fun () -> Shape.reshape p (ints [ 5; 5 ])));
      test "cumalu past 512 elements sums each prefix" (fun () ->
          let sums = Shape.cumalu (Shape.arange ~dtype:Int32 600) 0 Op.Add in
          equal (list int)
            (List.init 600 (fun i -> i * (i + 1) / 2))
            (List.map
               (function `Int z -> Bigint.to_int z | _ -> -1)
               (Array.to_list (Tensors.eval sums))));
      test "arange rejects a step of 0 and a dtype that cannot hold it"
        (fun () ->
          rejects (fun () -> Shape.arange ~step:0 4);
          rejects (fun () -> Shape.arange ~dtype:Int8 200));
      test "arange rejects more than max_int - 2^30 elements" (fun () ->
          rejects (fun () -> Shape.arange (max_int - (1 lsl 30) + 1));
          rejects (fun () -> Shape.arange ~start:(-(1 lsl 61)) (1 lsl 61)));
      test "arange counts its elements past an int's range" (fun () ->
          let step = 1 lsl 60 in
          let u = Shape.arange ~start:min_int ~step max_int in
          let at k =
            Bigint.(add (of_int min_int) (mul (of_int k) (of_int step)))
          in
          equal (list Dtypes.const)
            (List.init 8 (fun k -> `Int (at k)))
            (Array.to_list (Tensors.eval u));
          let u = Shape.arange ~dtype:Int8 ~start:(-100) ~step:max_int (-50) in
          equal (list Dtypes.const) [ i (-100) ]
            (Array.to_list (Tensors.eval u)));
      test "arange's type holds its last element, past [stop - step]"
        (fun () ->
          rejects (fun () -> Shape.arange ~dtype:Int8 ~step:2 129);
          let u = Shape.arange ~step:2 ((1 lsl 31) + 1) in
          equal (list value)
            [ `Int (Bigint.of_int (1 lsl 31)) ]
            (window u (1 lsl 30) 1));
      test "rop rejects a repeated axis" (fun () ->
          rejects (fun () -> Shape.rop p Op.Add [ 1; 1 ]));
      test "permute rejects an order that is not a permutation" (fun () ->
          rejects (fun () -> Shape.permute p [ 0; 0; 1 ]);
          rejects (fun () -> Shape.permute p [ 0; 1 ]);
          rejects (fun () -> Shape.permute p [ 0; 1; 3 ]));
      test "mop rejects an empty pad or shrink of a node with axes" (fun () ->
          rejects (fun () -> Shape.mop p (Pad []));
          rejects (fun () -> Shape.mop p (Shrink []));
          let s = Ops.int 1 in
          is_true (Shape.mop s (Pad []) == s);
          is_true (Shape.mop s (Shrink []) == s));
      test "mop stores a shape argument as sources and reads it back" (fun () ->
          let r = Shape.mop p (Reshape (ints [ 6; 4 ])) in
          equal uops
            [ p; Ops.v ~src:[ Ops.int 6; Ops.int 4 ] Op.Stack ]
            (Ops.src r);
          equal shape
            (ints [ 6; 4 ])
            (match Shape.marg r with Reshape s -> s | _ -> fail "a reshape");
          equal shape (ints [ 6; 4 ]) (Shape.as_shape (Ops.nth r 1));
          equal shape (ints [ 3 ]) (Shape.as_shape (Ops.int 3)));
      test "as_shape of a node is the node" (fun () ->
          let n = weak_var "n" 1 4 in
          equal shape [ Sym n ] (Shape.as_shape n));
      test "marg reads each movement's argument and rejects other nodes"
        (fun () ->
          (match Shape.marg (Shape.permute p [ 2; 0; 1 ]) with
          | Permute o -> equal (list int) [ 2; 0; 1 ] o
          | _ -> fail "permute");
          (match
             Shape.marg (Shape.shrink p [ Some (Int 1, Int 2); None; None ])
           with
          | Shrink l ->
              equal
                (list (pair sint sint))
                [ (Int 1, Int 1); (Int 0, Int 3); (Int 0, Int 4) ]
                l
          | _ -> fail "shrink");
          (match Shape.marg (Shape.pad p [ Some (Int 1, Int 2); None; None ]) with
          | Pad l ->
              equal
                (list (pair sint sint))
                [ (Int 1, Int 5); (Int 0, Int 3); (Int 0, Int 4) ]
                l
          | _ -> fail "pad");
          rejects (fun () -> Shape.marg (Ops.int 1)));
      test "flip rejects an axis given twice, and is the node for no axis"
        (fun () ->
          rejects (fun () -> Shape.flip p [ 0; 0 ]);
          rejects (fun () -> Shape.flip (Shape.param 1 Float32) [ 0 ]);
          is_true (Shape.flip (Shape.param 1 Float32) [] == Shape.param 1 Float32);
          is_true (Shape.flip p [] == p));
      test "repeat refuses a negative repeat" (fun () ->
          rejects (fun () -> Shape.repeat (param [ 1 ] 1 Float32) [ -1 ]));
      test "a flip of a view of variable size reverses it at each size"
        (fun () ->
          let n = Ops.variable "n" (`Int Bigint.zero) (`Int (Bigint.of_int 4)) in
          let x = param [ 4 ] 1 Int32 in
          let flipped = Shape.flip (Shape.shrink x [ Some (Int 0, Sym n) ]) [ 0 ] in
          let memory : Dtype.value array =
            Array.init 4 (fun j -> `Int (Bigint.of_int j))
          in
          List.iter
            (fun k ->
              let bound =
                Ops.substitute ~calls:Skip ~pass:Fixed_point flipped
                  [ (n, Ops.int k) ]
              in
              equal ~msg:(string_of_int k) (list Dtypes.const)
                (List.init k (fun i -> `Int (Bigint.of_int (k - 1 - i))))
                (Array.to_list (Tensors.eval ~buffers:[ (1, memory) ] bound)))
            [ 0; 1; 2; 3; 4 ]);
      prop "a flip reverses each axis it names and keeps the others"
        gen_flip (fun (dims, axes) ->
          let n = List.fold_left ( * ) 1 dims in
          let flips_long = List.exists (fun a -> List.nth dims a > 1) axes in
          cover "an empty axis beside a flipped one"
            (List.mem 0 dims && flips_long);
          cover "an axis of one element is flipped"
            (List.exists (fun a -> List.nth dims a = 1) axes);
          cover "two axes are flipped" (List.length axes >= 2);
          let x = Shape.reshape (param [ n ] 1 Int32) (ints dims) in
          let memory : Dtype.value array =
            Array.init n (fun j -> `Int (Bigint.of_int j))
          in
          equal (list Dtypes.const)
            (List.init n (fun i ->
                 (memory.(flipped dims axes i) :> Dtype.const)))
            (Array.to_list
               (Tensors.eval ~buffers:[ (1, memory) ] (Shape.flip x axes))));
      test "shrink, pad and pad_to reject a list of another length" (fun () ->
          rejects (fun () -> Shape.shrink p [ None ]);
          rejects (fun () -> Shape.pad p [ None; None ]);
          rejects (fun () -> Shape.pad_to p [ None ]));
      test "pad_to to the same shape is the node, whatever the value" (fun () ->
          is_true (Shape.pad_to ~value:(f 1.) p [ None; None; None ] == p));
      test "squeeze leaves an axis of another size" (fun () ->
          is_true (Shape.squeeze ~axis:1 p == p));
      test "cat rejects shapes that differ off its axis" (fun () ->
          rejects (fun () -> Shape.cat p [ param [ 2; 4; 4 ] 1 Float32 ]));
      test "nbytes is the elements times their size" (fun () ->
          equal int 96 (Shape.nbytes p);
          equal int 2 (Shape.nbytes (Shape.param 1 Float16)));
    ]

(* Storage *)

let devices =
  group "devices"
    [
      test "empty_like on another device takes the value's shape" (fun () ->
          let u =
            Shape.empty_like ~dtype:Int32 ~device:"CPU:2"
              (param ~device:cpu [ 8; 6 ] 1 Float32)
          in
          equal shape (ints [ 8; 6 ]) (Shape.shape u);
          equal (option device) (Some "CPU:2") (Ops.device u);
          is_true (Ops.has_buffer_identity u));
    ]

let storage =
  group "storage"
    [
      test "an alloc given a slot is no alloc made without one" (fun () ->
          (* The number the next fresh alloc would take, given by hand first. *)
          let slot = Ops.unique_num () + 1 in
          let given = Shape.alloc ~slot ~device:cpu (ints [ 8 ]) Float32 in
          let made = Shape.alloc ~device:cpu (ints [ 8 ]) Float32 in
          not_equal int
            (Ops.param_arg_of (Ops.base given)).slot
            (Ops.param_arg_of (Ops.base made)).slot);
      test "a placeholder given a slot is no placeholder made without one"
        (fun () ->
          let check addrspace =
            let slot = Ops.unique_num () + 1 in
            let given = Shape.placeholder ~slot ~addrspace [ 8 ] Float32 in
            let made = Shape.placeholder ~addrspace [ 8 ] Float32 in
            not_equal int (Ops.param_arg_of given).slot
              (Ops.param_arg_of made).slot
          in
          List.iter check [ Dtype.Global; Dtype.Local; Dtype.Reg ]);
      test "an alloc or a placeholder cannot be given the last number"
        (fun () ->
          rejects (fun () -> Shape.alloc ~slot:max_int (ints [ 8 ]) Float32);
          rejects (fun () -> Shape.placeholder ~slot:max_int [ 8 ] Float32));
      test "placeholder rejects a scalar variable's address space" (fun () ->
          rejects (fun () ->
              Shape.placeholder ~slot:0 ~addrspace:Alu [ 4 ] Float32));
      test "storage rejects a weak type" (fun () ->
          rejects (fun () -> Ops.new_buffer cpu 4 Weak_float);
          rejects (fun () -> Shape.empty ~device:cpu (ints [ 4 ]) Weak_int);
          rejects (fun () -> Shape.param 0 Weak_int));
      test "placeholder commits a weak type" (fun () ->
          equal dtype Int32 (Ops.dtype (Shape.placeholder ~slot:0 [ 4 ] Weak_int)));
      test "a clone of a weak value commits its type" (fun () ->
          equal dtype Int32
            (Ops.dtype
               (Shape.clone ~device:cpu (Shape.expand (Ops.int 3) (ints [ 4 ])))));
      test "clone rejects a disk, named in any case" (fun () ->
          let p = param ~device:cpu [ 4 ] 0 Float32 in
          rejects (fun () -> Shape.clone ~device:"DISK:/tmp/f" p);
          rejects (fun () -> Shape.clone ~device:"disk:0" p);
          is_true
            (String.equal "DISKS"
               (Option.get (Ops.device (Shape.clone ~device:"DISKS" p)))));
      test
        "param_like keeps its storage's phase, moved by a shrink, and knows it \
         modulo what a symbolic start moves by" (fun () ->
          let b = Ops.new_buffer ~phase:4 "CPU" 16 Dtype.Float32 in
          let phase u =
            match Ops.arg (Ops.buf_uop (Shape.param_like u 0)) with
            | Param p -> p.phase
            | _ -> fail "a parameter"
          in
          let shrunk k = Shape.shrink b [ Some (Int k, Int (k + 4)) ] in
          equal int 4 (phase b);
          equal int 8 (phase (shrunk 1));
          equal int 0 (phase (shrunk 3));
          equal int 8 (phase (Shape.reshape (shrunk 1) [ Int 2; Int 2 ]));
          let square = Shape.reshape b [ Int 4; Int 4 ] in
          equal int 4 (phase (Shape.shrink square [ Some (Int 1, Int 3); None ]));
          equal int 8
            (phase
               (Shape.shrink square [ Some (Int 1, Int 3); Some (Int 1, Int 3) ]));
          equal int 4
            (phase
               (Shape.shrink
                  (Shape.permute square [ 1; 0 ])
                  [ Some (Int 1, Int 3); None ]));
          (* a start [k * every] with [k] from 0 to [8 / every] *)
          let multiple every hi =
            let k =
              Ops.variable "k" (`Int Bigint.zero) (`Int (Bigint.of_int hi))
            in
            if every = 1 then k else Ops.mul k (Ops.int every)
          in
          let at ?(every = 1) () =
            let k = multiple every (8 / every) in
            Shape.shrink b [ Some (Sym k, Sym (Ops.add k (Ops.int 4))) ]
          in
          let known u =
            match Ops.arg (Ops.buf_uop (Shape.param_like u 0)) with
            | Param p -> (p.align, p.phase)
            | _ -> fail "a parameter"
          in
          equal (pair int int) ~msg:"a start of any element" (4, 0)
            (known (at ()));
          equal (pair int int) ~msg:"a start of every other element" (8, 4)
            (known (at ~every:2 ()));
          equal (pair int int) ~msg:"a start of every fourth element" (16, 4)
            (known (at ~every:4 ()));
          equal (pair int int) ~msg:"a start of every fourth element of a stage"
            (16, 0)
            (known
               (let k = multiple 4 2 in
                Shape.shrink
                  (Ops.contiguous (Ops.add b b))
                  [ Some (Sym k, Sym (Ops.add k (Ops.int 4))) ]));
          equal (pair int int) ~msg:"a start into a reordered view" (4, 0)
            (known
               (let k = multiple 1 2 in
                Shape.shrink
                  (Shape.permute square [ 1; 0 ])
                  [ Some (Sym k, Sym (Ops.add k (Ops.int 2))); None ]));
          equal (pair int int) ~msg:"a start one past every fourth element"
            (16, 8)
            (known
               (let k = multiple 4 2 in
                Shape.shrink b
                  [
                    Some
                      (Sym (Ops.add k (Ops.int 1)), Sym (Ops.add k (Ops.int 5)));
                  ])));
    ]

(* Variables *)

let variables =
  let n = weak_var "n" 1 8 in
  group "variables"
    [
      test "param_like names the parameter it makes, or leaves it unnamed"
        (fun () ->
          let p = param ~device:cpu [ 4 ] 0 Float32 in
          equal string "x" (Ops.expr (Shape.param_like ~name:"x" p 1));
          equal string "y" (Ops.expr (Shape.param_like ~name:"y" n 2));
          rejects (fun () -> Ops.expr (Shape.param_like n 2));
          rejects (fun () -> Ops.expr (Shape.param_like p 1)));
      test "bind binds a value within the range" (fun () ->
          let b = Shape.bind n (i 3) in
          is_true (Shape.is_bound_var b);
          is_true (Shape.is_variable b);
          equal (pair uop value) (n, i 3) (Shape.unbind b);
          ignore (Shape.bind n (i 1));
          ignore (Shape.bind n (i 8)));
      test
        "bind rejects a bound variable and a value out of range" (fun () ->
          rejects (fun () -> Shape.bind (Shape.bind n (i 3)) (i 4));
          rejects (fun () -> Shape.bind n (i 0));
          rejects (fun () -> Shape.bind n (i 9));
          rejects (fun () -> Shape.bind (param [ 4 ] 0 Float32) (i 1)));
      test "unbound strips the value" (fun () ->
          is_true (Shape.unbound (Shape.bind n (i 3)) == n);
          is_true (Shape.unbound n == n));
      test "unbound rejects a node that is not a variable" (fun () ->
          rejects (fun () -> Shape.unbound (param [ 4 ] 0 Float32)));
      test "variables lists a scalar parameter without a range" (fun () ->
          let s = Shape.param ~addrspace:Alu 0 Int32 in
          equal uops [ s ] (Shape.variables Ops.O.(s + int 1)));
      test "unbind rejects a variable that is not bound" (fun () ->
          rejects (fun () -> Shape.unbind n));
      test "unbind_all unbinds each variable and lists its value" (fun () ->
          let m = weak_var "m" 0 4 in
          let e = Ops.O.(Shape.bind n (i 3) + Shape.bind m (i 2)) in
          let u, values = Shape.unbind_all e in
          equal uop Ops.O.(n + m) u;
          equal
            (slist (pair uop value) (fun (a, _) (b, _) -> Ops.compare a b))
            [ (n, i 3); (m, i 2) ]
            values);
      test "variables are sorted by name" (fun () ->
          let m = weak_var "m" 0 4 in
          equal uops [ m; n ] (Shape.variables Ops.O.(n + m));
          equal uops [ m; n ] (Shape.variables Ops.O.(Shape.bind n (i 3) + m)));
      test "variables lists each variable once, sorted by name" (fun () ->
          let a = weak_var "a" 0 10
          and b = weak_var "b" 0 10
          and c = weak_var "c" 0 10 in
          let vars u = Shape.variables u in
          equal (list uop) [] (vars (Ops.int 0));
          equal (list uop) [ a ] (vars Ops.O.(a * int 3));
          equal (list uop) [ a; b; c ] (vars Ops.O.(a + b + c));
          equal (list uop) [ a; b; c ] (vars Ops.O.(a + (b * c)));
          equal (list uop) [ a; b ] (vars Ops.O.((a % int 3) + (b // int 5)));
          equal (list uop) [ a; b; c ] (vars Ops.O.(a + b + c - a));
          equal (list uop) [ a ] (vars Ops.O.(a * a));
          equal (list uop) [ a ] (vars Ops.O.((a // int 4) + (a // int 6))));
    ]

(* Divisibility *)

let divisibility =
  let x = weak_var "x" 10 20 in
  group "divisibility"
    [
      test "divides divides a product through its left factor first" (fun () ->
          let k = weak_var "k" 0 4 in
          equal (option uop)
            (Some Ops.O.(k * int 2 * x))
            (Shape.divides Ops.O.(k * int 4 * x) (Bigint.of_int 2));
          is_none (Shape.divides Ops.O.(x + int 3) (Bigint.of_int 2)));
      test "gcd rejects nothing" (fun () -> rejects (fun () -> Shape.gcd []));
      test "gcd of constants is their gcd" (fun () ->
          equal uop (Ops.int 2) (Shape.gcd [ Ops.int 6; Ops.int 4 ]));
      test "divides divides a known multiple" (fun () ->
          equal (option z)
            (Some (Bigint.of_int 6))
            (Option.map Ops.const_factor
               (Shape.divides (Ops.int 42) (Bigint.of_int 7)));
          is_none (Shape.divides (Ops.int 42) (Bigint.of_int 5));
          equal (option z) (Some Bigint.one)
            (Option.map Ops.const_factor
               (Shape.divides Ops.O.((x * int 6) + int 18) (Bigint.of_int 6)));
          is_none
            (Shape.divides Ops.O.(weak_var "x" 15 45 * int 4) (Bigint.of_int 3));
          equal (option uop) (Some x) (Shape.divides x Bigint.one));
      test "divides divides a float constant only into an integer" (fun () ->
          is_none (Shape.divides (Ops.float 2.5) (Bigint.of_int 2));
          equal (option uop)
            (Some (Ops.float 2.))
            (Shape.divides (Ops.float 4.) (Bigint.of_int 2)));
      test "divides divides a typed constant, a cast of one, as its value"
        (fun () ->
          equal (option uop)
            (Some (Ops.int ~dtype:Int32 4))
            (Shape.divides (Ops.int ~dtype:Int32 8) (Bigint.of_int 2));
          is_none (Shape.divides (Ops.int ~dtype:Int32 8) (Bigint.of_int 3)));
      test "gcd keeps the common factors and the gcd of the coefficients"
        (fun () ->
          let y = weak_var "y" 1 4 and z' = weak_var "z" 1 4 in
          equal uop Ops.O.(int 1 * x) (Shape.gcd Ops.O.[ x * y; x * z' ]);
          equal uop (Ops.int 2) (Shape.gcd Ops.O.[ x * int 6; y * int 4 ]));
      test "divide_exact divides each term, or gives up" (fun () ->
          let y = weak_var "y" 1 4 and z' = weak_var "z" 1 4 in
          equal (option uop) (Some (Ops.int 1)) (Shape.divide_exact x x);
          equal (option uop)
            (Some Ops.O.((int 1 * y) + (int 1 * z')))
            (Shape.divide_exact Ops.O.((x * y) + (x * z')) x);
          equal (option uop)
            (Some (Ops.int 3))
            (Shape.divide_exact (Ops.int 6) (Ops.int 2));
          is_none (Shape.divide_exact Ops.O.(x + int 1) x);
          is_none
            (Shape.divide_exact
               Ops.O.(x * int 6)
               Ops.O.(weak_var "y" 1 4 * int 2));
          is_none
            (Shape.divide_exact
               (Shape.stack Ops.O.[ x * int 2; x * int 4 ])
               (Shape.consts [ i 2; i 4 ])));
      test "divide_exact gives up on what does not divide" (fun () ->
          let a = weak_var "a" 1 8
          and b = weak_var "b" 1 8
          and x = weak_var "x" (-20) 0 in
          let none u d = is_none ~pp:(Testable.pp uop) (Shape.divide_exact u d) in
          none a b;
          none Ops.O.(a + int 2) a;
          none Ops.O.(x * int (-1)) a;
          none Ops.O.(a * int 5) Ops.O.(a * int 10);
          none Ops.O.((a * int 10) - int 1) Ops.O.(a * int 10));
    ]

(* Calls *)

let calls =
  let arg = param ~device:cpu [ 4 ] 0 Float32 in
  group "calls"
    [
      test
        "call_with_outputs rejects output positions that do not ascend within \
         the arguments" (fun () ->
          let v = Ops.O.(arg + float 1.) in
          rejects (fun () ->
              Shape.call_with_outputs ~output_pos:[ 1; 0 ] [ v; v ] [ arg ]);
          rejects (fun () ->
              Shape.call_with_outputs ~output_pos:[ 0; 0 ] [ v; v ] [ arg ]);
          rejects (fun () ->
              Shape.call_with_outputs ~output_pos:[ 5 ] [ v ] [ arg ]);
          rejects (fun () ->
              Shape.call_with_outputs ~output_pos:[ 0; 1 ] [ v ] [ arg ]));
      test "a call compiled apart stages its arguments" (fun () ->
          let v = Ops.O.(arg + float 1.) in
          let out = Shape.call_with_output ~precompile:true v [ v ] in
          equal op Op.Stage (Ops.op (Ops.nth (Ops.nth out 1) 1)));
    ]

(* Programs *)

let alu_param ?(lo = 0) ?(hi = 8) name slot =
  Shape.param ~vmin_vmax:(i lo, i hi) ~name ~addrspace:Alu slot Int32

let slot u = match Ops.arg u with Param p -> p.slot | _ -> -2
let scalar u = Ops.addrspace u = Some Alu

let programs =
  group "programs"
    [
      test "program_info_of_sink reads launch sizes, variables and buffers"
        (fun () ->
          let core_id = alu_param ~hi:3 "core_id" 0
          and n = alu_param ~lo:2 "n" 1 in
          let input = Shape.param ~shape:(ints [ 16 ]) 2 Float32
          and output = Shape.param ~shape:(ints [ 16 ]) 3 Float32 in
          let stored =
            Ops.store (Ops.index output [ n ])
              (Ops.load (Ops.index input [ n ]) [])
          in
          let sink =
            Ops.sink
              [
                stored;
                Ops.special (Int 4) "gidx2";
                Ops.special (Int 8) "lidx1";
                core_id;
              ]
          in
          let info = Shape.program_info_of_sink sink in
          equal (list int) [ 2; 3 ] info.globals;
          equal (list int) [ 3 ] info.outs;
          equal (list int) [ 2 ] info.ins;
          equal (list int) [ 0; 1; 2; 3 ] (List.map slot info.params);
          equal uops [ core_id; n ] (List.filter scalar info.params);
          equal
            (pair (list int) (list int))
            ([ 1; 1; 4 ], [ 1; 8; 1 ])
            (Shape.launch_dims info []);
          is_true
            (info.target
            = Helpers.Target.
                {
                  device = "";
                  renderer = "";
                  arch = "";
                  interface = "";
                  indices = "";
                }));
      test "a load through a cast and a store into a shrink are accesses"
        (fun () ->
          let p0 = Shape.param ~shape:(ints [ 4 ]) 0 Uint8
          and p1 = Shape.param ~shape:(ints [ 4 ]) 1 Uint8 in
          let cast_read =
            Ops.load
              (Ops.v
                 ~src:[ Ops.index p0 [ Ops.int 0 ] ]
                 ~arg:(Dtype Int8) Op.Cast)
              []
          in
          let write =
            Ops.store
              (Shape.shrink p1 [ Some (Int 0, Int 2) ])
              (Shape.expand (Ops.int ~dtype:Uint8 1) (ints [ 2 ]))
          in
          let info = Shape.program_info_of_sink (Ops.sink [ write; cast_read ]) in
          equal (list int) [ 0 ] info.ins;
          equal (list int) [ 1 ] info.outs);
      test "a load through a cast of something other than an index is no access"
        (fun () ->
          let p0 = Shape.param ~shape:(ints [ 4 ]) 0 Uint8
          and p1 = Shape.param ~shape:(ints [ 4 ]) 1 Uint8 in
          let write =
            Ops.store (Ops.index p1 [ Ops.int 0 ]) (Ops.int ~dtype:Uint8 1)
          in
          let info =
            Shape.program_info_of_sink
              (Ops.sink [ write; Ops.load (Ops.cast p0 Int8) [] ])
          in
          equal (list int) [ 1 ] info.outs;
          equal (list int) [] info.ins);
      test "the parameters of a called body are not the program's" (fun () ->
          let p0 = Shape.param ~shape:(ints [ 4 ]) 0 Uint8
          and p5 = Shape.param ~shape:(ints [ 4 ]) 5 Uint8 in
          let one p = Ops.store (Ops.index p [ Ops.int 0 ]) (Ops.int ~dtype:Uint8 1) in
          let body = Ops.sink [ one p5 ] in
          let info =
            Shape.program_info_of_sink
              (Ops.sink [ one p0; Ops.call ~name:"f" body [] ])
          in
          equal (list int) [ 0 ] info.globals;
          equal (list int) [ 0 ] info.outs;
          equal (list int) [] info.ins);
      test "program_info_of_sink reads symbolic launch sizes" (fun () ->
          let core_id = alu_param ~hi:3 "core_id" 0
          and n = alu_param ~lo:2 "n" 1 in
          let input = Shape.param ~shape:(ints [ 16 ]) 2 Float32
          and output = Shape.param ~shape:(ints [ 16 ]) 3 Float32 in
          let stored =
            Ops.store (Ops.index output [ n ])
              (Ops.load (Ops.index input [ n ]) [])
          in
          let sink =
            Ops.sink
              ~kernel:(Ops.kernel_info ~name:"kernel name" ())
              [
                stored;
                Ops.special (Sym Ops.O.(n + int 1)) "gidx2";
                Ops.special (Int 8) "lidx1";
                core_id;
              ]
          in
          let info = Shape.program_info_of_sink sink in
          equal (list int) [ 2; 3 ] info.globals;
          equal (list int) [ 3 ] info.outs;
          equal (list int) [ 2 ] info.ins;
          equal (list int) [ 0; 1; 2; 3 ] (List.map slot info.params);
          equal uops [ core_id; n ] (List.filter scalar info.params);
          equal
            (pair (list int) (list int))
            ([ 1; 1; 7 ], [ 1; 8; 1 ])
            (Shape.launch_dims info [ ("n", 6); ("core_id", 2) ]);
          (* A size reads its variable as a run binds it, at no slot. *)
          equal (list int) [ -1 ]
            (match List.nth info.global_size 2 with
            | Sym u ->
                List.filter_map
                  (fun x -> if Ops.op x = Param then Some (slot x) else None)
                  (Ops.toposort ~calls:Skip u)
            | Int _ -> []);
          is_true
            (info.target
            = Helpers.Target.
                {
                  device = "";
                  renderer = "";
                  arch = "";
                  interface = "";
                  indices = "";
                }));
      test "every buffer reads and writes when no access says otherwise"
        (fun () ->
          let info =
            Shape.program_info_of_sink
              (Ops.sink
                 [
                   Shape.param ~shape:(ints [ 4 ]) 0 Float32;
                   Shape.param ~shape:(ints [ 4 ]) 5 Float32;
                 ])
          in
          equal (list int) [ 0; 5 ] info.globals;
          equal (list int) [ 0; 5 ] info.outs;
          equal (list int) [ 0; 5 ] info.ins;
          equal (list int) [ 1; 1; 1 ] (fst (Shape.launch_dims info [])));
      test "launch sizes divide as their operations say" (fun () ->
          let n = alu_param ~lo:(-10) ~hi:10 "n" 0 in
          let info =
            Shape.program_info_of_sink
              (Ops.sink
                 [
                   Ops.special (Sym Ops.O.(n // int 3)) "gidx0";
                   Ops.special (Sym Ops.O.(n % int 3)) "gidx1";
                 ])
          in
          equal (list int) [ -3; 2; 1 ]
            (fst (Shape.launch_dims info [ ("n", -7) ])));
      test "launch_dims rejects a missing variable" (fun () ->
          let extent = alu_param "extent" 0 in
          let info =
            Shape.program_info_of_sink
              (Ops.sink [ Ops.special (Sym extent) "gidx0" ])
          in
          rejects (fun () -> Shape.launch_dims info []));
      test "program_info_of_sink records its target" (fun () ->
          let target = Result.get_ok (Helpers.Target.of_string "CPU:CLANG") in
          is_true
            ((Shape.program_info_of_sink ~target (Ops.sink [])).target = target));
    ]
