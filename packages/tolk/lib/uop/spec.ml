(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape
open Call

type t = (unit, bool) Pattern_matcher.t

(* Rules *)

let ops l = Op.Set.of_list l

let pat ?dtype ?src ?each ?allow_any_len ?arg ?name o =
  Upat.v ~op:(ops o) ?dtype ?src ?each ?allow_any_len ?arg ?name ()

let accept p = Pattern_matcher.rule p (fun _ -> Some true)
let reject p = Pattern_matcher.rule p (fun _ -> Some false)
let check p name f = Pattern_matcher.rule p (fun m -> Some (f (m name)))
let decide p name f = Pattern_matcher.rule p (fun m -> f (m name))

(* Helpers *)

let validate_index ?(gate = bool true) uidx =
  match src uidx with
  | [ buf; idx ]
    when (not (is_invalid idx)) && Setting.value Setting.check_oob
    ->
      (* Without an SMT solver, the bounds of the index are the proof: an index
         they do not prove fails, as the solver's unknown verdict does. *)
      let sz = max_numel buf in
      Dtype.Value.(of_int 0 <= vmin idx && vmax idx < of_int sz)
      || begin
        Format.eprintf
          "# INDEX NOT PROVEN IN BOUNDS: [%a, %a] is not within 0 - %d, and \
           the bound cannot be proven without a solver@.idx=%s@.mask=%s@."
          Dtype.pp_const (vmin idx) Dtype.pp_const (vmax idx) sz
          (Render.render ~simplify:false idx)
          (Render.render ~simplify:false gate);
        false
      end
  | _ -> true

let valid_device_range device src =
  match (device, src) with
  | Some (Multi devices), [ rng ] ->
      op rng = Op.Range
      && Axis_type.equal (axis_type rng) Axis_type.Device
      && Dtype.Value.to_int (vmax rng) + 1 = List.length devices
  | Some (Multi _), _ -> false
  | _, src -> List.is_empty src

(* The failure message writes the node's own argument as text, with names and
   float values bare, and its sources' arguments as literals. *)
let pp_arg_text ppf = function
  | String s | Device (Single s) -> Format.pp_print_string ppf s
  | Const c -> Dtype.pp_const ppf c
  | a -> pp_arg ppf a

let type_verify ~calls spec ast =
  let lst = toposort ~calls ast in
  List.iteri
    (fun i u ->
      if Pattern_matcher.rewrite spec () u <> Some true then begin
        if Setting.value Setting.debug >= 3 then
          Format.eprintf "%a@." Render.pp_uops lst;
        let pp_src ppf x =
          Format.fprintf ppf "(%a, %a, %a)" Op.pp (op x) Dtype.pp (dtype x)
            pp_arg (arg x)
        in
        invalid_arg
          (Format.asprintf "UOp verification failed at %d on %a %a %d [%a] %a" i
             Op.pp (op u) Dtype.pp (dtype u)
             (List.length (src u))
             (Format.pp_print_list
                ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
                pp_src)
             (src u) pp_arg_text (arg u))
      end)
    lst

(* Specifications *)

let no_arg u = match arg u with No_arg -> true | _ -> false
let matches_dtype x dt = Dtype.equal (dtype x) dt || is_invalid (base x)
let is_weak x = List.mem (dtype x) Dtype.weaks
let all_same eq = function [] -> true | x :: r -> List.for_all (eq x) r
let shape_equal s0 s1 = List.equal Sint.equal s0 s1

let param_of u =
  match arg u with Param p -> p | _ -> invalid_arg "not a parameter"

(* Construction gives CONST, CAST, BITCAST, PARAM, BUFFER, ALLOC, CUSTOM,
   CUSTOMI and INS the argument their type comes from, and a CONST the type of
   its value, so the rules check only the arguments construction leaves free. *)

let mselect_fits x =
  match (device (nth x 0), arg x) with
  | Some (Multi devices), Shard i -> i < List.length devices
  | _ -> false

let mstack_fits x =
  List.for_all
    (fun s -> match device s with Some (Single _) -> true | _ -> false)
    (src x)
  || (all_same ( == ) (src x) && Option.is_none (device (nth x 0)))

let memory = Upat.or_casted (pat [ Op.Index; Op.Shrink ] ~name:"uidx")

(* Each argument of storage starts where its parameter in the body takes it to:
   what is known of the argument's start implies what the parameter assumes,
   which its vector accesses were merged for. *)
let args_fit c =
  let args = Array.of_list (src_without_body c) in
  List.for_all
    (fun u ->
      match (op u, arg u) with
      | Op.Param, Param p
        when p.addrspace = Some Dtype.Global
             && p.slot >= 0
             && p.slot < Array.length args ->
          let align, phase = storage_phase args.(p.slot) in
          align >= p.align && phase mod p.align = p.phase
      | _ -> true)
    (toposort ~calls:Skip (body c))

let shared : t =
  Pattern_matcher.fold
    (fun () -> [
      accept (pat [ Op.Sink ] ~dtype:[ Dtype.Void ]);
      accept (pat [ Op.Noop ]);
      accept (pat [ Op.Const ] ~src:[]);
      accept (pat [ Op.Stack ] ~dtype:[ Dtype.Void ] ~src:[]);
      check (pat [ Op.Stack ] ~src:[ Upat.wild ] ~allow_any_len:true ~name:"s")
        "s" (fun s ->
          all_same shape_equal (List.map shape (src s))
          && List.for_all
               (fun x -> matches_dtype x (dtype s) || is_weak x)
               (src s));
      check
        (pat [ Op.Where ] ~name:"w"
           ~src:[ Upat.v ~dtype:[ Dtype.Bool ] (); Upat.wild; Upat.wild ])
        "w"
        (fun w ->
          List.for_all
            (fun s -> matches_dtype s (dtype w) || is_weak s)
            (List.tl (src w)));
      Pattern_matcher.rule
        (Upat.v ~op:Op.Set.comparison ~dtype:[ Dtype.Bool ]
           ~src:[ Upat.var "x"; Upat.var "y" ]
           ())
        (fun m ->
          let x = m "x" and y = m "y" in
          Some
            (matches_dtype x (dtype y)
            || matches_dtype y (dtype x)
            || is_weak x || is_weak y));
      decide
        (pat [ Op.And; Op.Or; Op.Xor; Op.Shl; Op.Shr ] ~name:"x")
        "x"
        (fun x ->
          if List.exists (fun s -> Dtype.is_float (dtype s)) (src x) then
            Some false
          else None);
      Pattern_matcher.rule
        (pat [ Op.Shl; Op.Shr ] ~src:[ Upat.var "x"; Upat.var "c" ] ~name:"a")
        (fun m ->
          let a = m "a" and x = m "x" and c = m "c" in
          Some
            (matches_dtype c (dtype a)
            || List.mem (dtype c) Dtype.[ Uint32; Weak_int ]
            || is_invalid (base x)));
      decide
        (pat [ Op.Cdiv; Op.Cmod; Op.Floordiv; Op.Floormod ] ~name:"x")
        "x"
        (fun x ->
          if
            Dtype.is_int (dtype x)
            || List.exists (fun s -> is_invalid (base s)) (src x)
          then None
          else Some false);
      check (Upat.v ~op:Op.Set.alu ~name:"x" ()) "x" (fun x ->
          List.for_all (fun y -> matches_dtype y (dtype x) || is_weak y) (src x));
      accept (pat [ Op.Bitcast; Op.Cast ] ~src:[ Upat.wild ]);
      check
        (pat [ Op.Range ] ~src:[ Upat.wild ] ~allow_any_len:true ~name:"rng")
        "rng" (fun rng ->
          match arg rng with
          | Range { axis_id; _ } -> axis_id <> []
          | _ -> false);
      decide (pat [ Op.Index ] ~name:"x") "x" (fun x ->
          match src x with
          | _ :: idxs
            when List.for_all
                   (fun y -> Dtype.is_int (dtype y) || is_invalid (base y))
                   idxs ->
              Some true
          | _ -> None);
      check
        (pat [ Op.End ]
           ~src:[ Upat.v ~dtype:[ Dtype.Void ] () ]
           ~allow_any_len:true ~name:"x")
        "x"
        (fun x ->
          no_arg x
          && List.for_all
               (fun u -> op u = Op.Range && Dtype.is_int (dtype u))
               (List.tl (src x)));
      check
        (pat [ Op.Backedge ] ~dtype:[ Dtype.Void ] ~name:"x"
           ~src:
             [
               Upat.wild;
               pat [ Op.Range ] ~dtype:[ Dtype.Void ];
               Upat.v ~dtype:[ Dtype.Bool ] ();
             ])
        "x"
        (fun x ->
          let cond = nth x 2 in
          no_arg x && shape cond = [] && not (is_invalid (base cond)));
      (* Around calls, a loop's range bounds its trips, a range of one trip
         being its value [0], and its condition is storage of one element, or,
         in a host batch's program, a value read from one, after the loop that
         walks its calls. *)
      check
        (pat [ Op.Backedge ] ~dtype:[ Dtype.Void ] ~name:"x"
           ~src:
             [
               pat [ Op.Call; Op.Linear; Op.End ];
               pat [ Op.Range; Op.Const ];
               Upat.v ~dtype:[ Dtype.Bool ] ();
             ])
        "x"
        (fun x ->
          let r = nth x 1 in
          no_arg x
          && Dtype.is_int (dtype r)
          && (if op r = Op.Const then arg r = Const (`Int Bigint.zero)
              else Axis_type.equal (axis_type r) Axis_type.Loop)
          && (addrspace (nth x 2) = Some Dtype.Alu
             || max_numel (buf_uop (nth x 2)) = 1));
      accept (pat [ Op.Param ] ~src:[]);
      check (pat [ Op.Buffer ] ~src:[] ~name:"x") "x" (fun x ->
          List.mem (addrspace x) [ Some Dtype.Reg; Some Dtype.Local ]);
      check (pat [ Op.Binary ] ~dtype:[ Dtype.Uint8 ] ~src:[] ~name:"x") "x"
        (fun x -> match arg x with Bytes _ -> true | _ -> false);
      accept
        (pat [ Op.Group ] ~dtype:[ Dtype.Void ]
           ~each:(Upat.v ~dtype:[ Dtype.Void ] ()));
      accept
        (pat [ Op.After ] ~allow_any_len:true
           ~src:
             [
               Upat.v
                 ~op:
                   (Op.Set.union Op.Set.movement
                      (ops
                         Op.
                           [
                             Param;
                             Buffer;
                             Alloc;
                             Stage;
                             Index;
                             After;
                             Unshard;
                             Bitcast;
                             Ins;
                           ]))
                 ();
             ]);
      accept (pat [ Op.Customi; Op.Custom ]);
      check (pat [ Op.Custom_function ] ~allow_any_len:true ~name:"x") "x"
        (fun x -> match arg x with String _ -> true | _ -> false);
      check
        (pat [ Op.Call ]
           ~src:[ Upat.v ~op:opaque_call_bodies () ]
           ~allow_any_len:true ~name:"x")
        "x"
        (fun x -> match arg x with Call _ -> args_fit x | _ -> false);
      accept (pat [ Op.Barrier ] ~dtype:[ Dtype.Void ]);
      accept (pat [ Op.Ins ]);
      check (Upat.load memory []) "uidx" validate_index;
      Pattern_matcher.rule
        (Upat.load ~name:"load" memory
           [ Upat.var "alt"; Upat.var ~dtype:[ Dtype.Bool ] "gate" ])
        (fun m ->
          Some
            (matches_dtype (m "alt") (dtype (m "load"))
            && validate_index ~gate:(m "gate") (m "uidx")));
      check (Upat.store memory [ Upat.wild ]) "uidx" validate_index;
      Pattern_matcher.rule
        (Upat.store memory [ Upat.wild; Upat.var ~dtype:[ Dtype.Bool ] "gate" ])
        (fun m -> Some (validate_index ~gate:(m "gate") (m "uidx")));
      decide
        (pat [ Op.Store ] ~dtype:[ Dtype.Void ]
           ~src:[ Upat.var "x"; Upat.wild ])
        "x"
        (fun x ->
          let b = storage_base x in
          if List.mem (op b) Op.[ Buffer; Alloc; Param; Stage ] then Some true
          else if op b = Op.Index then None
          else Some false);
      check
        (pat [ Op.Wmma ] ~src:[ Upat.wild; Upat.wild; Upat.wild ] ~name:"x")
        "x"
        (fun x -> match arg x with Wmma _ -> true | _ -> false);
    ])

let is_device = Option.is_some

let tensor : t =
  Pattern_matcher.append
    (Pattern_matcher.fold
       (fun () -> [
         check
           (pat
              [ Op.Sin; Op.Log2; Op.Exp2; Op.Sqrt; Op.Reciprocal ]
              ~src:[ Upat.wild ] ~name:"u")
           "u"
           (fun u -> Dtype.is_float (dtype u) || is_invalid (base (nth u 0)));
         decide (pat [ Op.Buffer ] ~name:"buf") "buf" (fun buf ->
             let p = param_of buf in
             if addrspace buf = Some Dtype.Global then
               Some
                 (Option.is_some p.size && is_device p.device
                 && valid_device_range p.device (src buf))
             else None);
         check (pat [ Op.Alloc ] ~name:"buf") "buf" (fun buf ->
             let p = param_of buf in
             List.mem (addrspace buf)
               Dtype.[ Some Global; Some Local; Some Reg ]
             && (Option.is_none p.device
                || (addrspace buf = Some Dtype.Global && is_device p.device))
             && valid_device_range p.device (src buf));
         decide (pat [ Op.Param ] ~src:[] ~name:"buf") "buf" (fun buf ->
             if is_variable buf then Some (Option.is_none (param_of buf).device)
             else None);
         check (pat [ Op.Custom_function ] ~name:"x") "x" (fun x ->
             match arg x with String _ -> true | _ -> false);
         check
           (pat [ Op.Special ]
              ~src:[ Upat.v ~dtype:[ Dtype.Weak_int ] () ]
              ~name:"s")
           "s"
           (fun s -> match arg s with String _ -> true | _ -> false);
         accept (pat [ Op.Reshape; Op.Expand ] ~src:[ Upat.wild; Upat.wild ]);
         check
           (pat [ Op.Pad; Op.Shrink ]
              ~src:[ Upat.wild; Upat.wild; Upat.wild ]
              ~name:"x")
           "x"
           (fun x -> shape_equal (shape (nth x 1)) (shape (nth x 2)));
         check
           (pat [ Op.Permute; Op.Flip ] ~name:"mv" ~src:[ Upat.wild ])
           "mv"
           (fun mv ->
             match (op mv, arg mv) with
             | Op.Permute, Axes _ | Op.Flip, Flips _ -> true
             | _ -> false);
         check
           (pat [ Op.Reduce ] ~src:[ Upat.wild ] ~allow_any_len:true ~name:"x")
           "x" (fun x ->
             match arg x with
             | Reduce { op; _ } ->
                 Op.Set.mem op Op.Set.reduce
                 && List.for_all
                      (fun y -> List.mem (dtype y) Dtype.[ Weak_int; Int32 ])
                      (List.tl (src x))
             | _ -> false);
         check
           (pat [ Op.Copy ] ~name:"copy" ~src:[ Upat.wild ] ~allow_any_len:true)
           "copy" (fun copy ->
             match arg copy with
             | Device d ->
                 (not (is_disk_device d))
                 && valid_device_range (Some d) (List.tl (src copy))
             | _ -> false);
         check (pat [ Op.Allreduce ] ~name:"red" ~src:[ Upat.wild ]) "red"
           (fun red ->
             match arg red with
             | Allreduce { op; _ } -> Op.Set.mem op Op.Set.reduce || op = Op.Or
             | _ -> false);
         check (pat [ Op.Unshard ] ~name:"multi") "multi" (fun multi ->
             match arg multi with
             | Axes axes ->
                 List.length (src multi) = 1 + List.length axes
                 && List.for_all is_weak (List.tl (src multi))
             | _ -> false);
         check (pat [ Op.Mselect ] ~name:"x") "x" mselect_fits;
         check (pat [ Op.Mstack ] ~name:"x") "x" mstack_fits;
         accept
           (pat
              [ Op.Detach; Op.Contiguous_backward ]
              ~src:[ Upat.wild ] ~arg:No_arg);
         accept (pat [ Op.Stage ] ~src:[ Upat.wild ] ~allow_any_len:true);
         accept (pat [ Op.Linear ] ~dtype:[ Dtype.Void ]);
         accept (pat [ Op.Source ] ~dtype:[ Dtype.Void ] ~src:[]);
         accept
           (pat [ Op.Program ] ~dtype:[ Dtype.Void ] ~src:[ pat [ Op.Sink ] ]);
         accept
           (pat [ Op.Program ] ~dtype:[ Dtype.Void ]
              ~src:[ pat [ Op.Sink ]; pat [ Op.Linear ] ]);
         accept
           (pat [ Op.Program ] ~dtype:[ Dtype.Void ]
              ~src:[ pat [ Op.Sink ]; pat [ Op.Linear ]; pat [ Op.Source ] ]);
         accept
           (pat [ Op.Program ] ~dtype:[ Dtype.Void ]
              ~src:
                [
                  pat [ Op.Sink ];
                  pat [ Op.Linear ];
                  pat [ Op.Source ];
                  pat [ Op.Binary ];
                ]);
       ]))
    shared

(* The rules of programs past their elementwise operations on values. *)
let program_rules vectors : t =
  Pattern_matcher.append
    (Pattern_matcher.fold
       (fun () -> [
         (* Every elementwise operation on values is on scalars: renderers whose
            vectors are structs without arithmetic cannot write one on a vector
           . A renderer that computes on vectors takes an operation on one
            axis, each source of it a vector of its lanes or a scalar every
            lane reads. A bitcast of memory views it, and a node without a
            shape is judged by the other rules. *)
         decide (Upat.v ~op:Op.Set.elementwise ~name:"x" ()) "x" (fun x ->
             let lanes s =
               match (shape_opt s, shape_opt x) with
               | Some [], _ -> true
               | Some s, Some xs -> shape_equal s xs
               | _ -> false
             in
             match (addrspace x, shape_opt x) with
             | Some Dtype.Alu, Some [ _ ]
               when vectors
                    && List.for_all
                         (fun s -> is_invalid (base s) || lanes s)
                         (src x) ->
                 None
             | Some Dtype.Alu, Some (_ :: _) -> Some false
             | _ | (exception Invalid_argument _) -> None);
         (* A lane of a vector value is read at a constant lane, for the same
            renderers: their lanes are members, named in the source. *)
         decide (pat [ Op.Index ] ~name:"x") "x" (fun x ->
             let constant i =
               op i = Op.Cast
               && match src i with [ c ] -> op c = Op.Const | _ -> false
             in
             match src x with
             | v :: lanes when addrspace v = Some Dtype.Alu ->
                 if List.for_all constant lanes then None else Some false
             | _ -> None);
         decide (Upat.v ~op:Op.Set.all ~name:"x" ()) "x" (fun x ->
             if
               op x <> Op.Cast && List.exists (fun s -> op s = Op.Const) (src x)
             then Some false
             else None);
         reject
           (Upat.v
              ~op:(Op.Set.diff Op.Set.all (ops [ Op.Const ]))
              ~dtype:Dtype.weaks ());
         accept
           (pat [ Op.Shrink ]
              ~src:
                [
                  Upat.or_bitcasted
                    (pat [ Op.Param; Op.Buffer; Op.Alloc; Op.After ]);
                  Upat.wild;
                  Upat.or_casted (pat [ Op.Const ]);
                ]);
         reject (Upat.v ~op:Op.Set.movement ());
         check
           (pat [ Op.Buffer; Op.Alloc ] ~name:"x")
           "x"
           (fun x -> List.mem (addrspace x) Dtype.[ Some Reg; Some Local ]);
         reject (Upat.const `Invalid);
         accept
           (pat [ Op.If ] ~dtype:[ Dtype.Void ]
              ~src:
                [
                  Upat.v ~dtype:[ Dtype.Bool ] ();
                  pat [ Op.Cast; Op.Index; Op.Shrink ];
                ]);
         accept (pat [ Op.Endif ] ~dtype:[ Dtype.Void ] ~src:[ pat [ Op.If ] ]);
         check
           (pat [ Op.Special ]
              ~src:[ Upat.v ~dtype:[ Dtype.Int32 ] () ]
              ~name:"s")
           "s"
           (fun s -> match arg s with String _ -> true | _ -> false);
       ]))
    shared

let program = program_rules false
let vector_program = program_rules true

let hcq : t =
  Pattern_matcher.append
    (Pattern_matcher.fold
       (fun () -> [
         check
           (pat [ Op.Getaddr ] ~dtype:[ Dtype.Uint64 ] ~name:"x"
              ~src:
                [
                  Upat.or_after
                    (pat
                       Op.
                         [
                           Buffer;
                           Alloc;
                           Param;
                           Shrink;
                           Bitcast;
                           Mstack;
                           Mselect;
                           Linear;
                         ]);
                ])
           "x"
           (fun x -> match arg x with Device _ -> true | _ -> false);
         accept
           (pat [ Op.Program ] ~dtype:[ Dtype.Void ]
              ~src:[ Upat.or_after (pat [ Op.Buffer; Op.Param ]) ]);
       ]))
    shared

let full : t =
  Pattern_matcher.concat
    [
      Pattern_matcher.fold
        (fun () -> [
          check
            (pat [ Op.End ]
               ~src:[ Upat.v ~dtype:[ Dtype.Void ] (); Upat.wild ]
               ~allow_any_len:true ~name:"x")
            "x"
            (fun x ->
              no_arg x
              && List.for_all
                   (fun u -> Dtype.is_int (dtype u))
                   (List.tl (src x)));
          accept (pat [ Op.After ] ~src:[ Upat.wild ] ~allow_any_len:true);
          accept (pat [ Op.Load; Op.Store ]);
        ]);
      tensor;
      program;
      hcq;
    ]

let loop r = Axis_type.equal (axis_type r) Axis_type.Loop

(* A bound of a view that moves with loops: constants, loop ranges, and weak
   integer sums and products of them. *)
let rec loop_bound u =
  match op u with
  | Op.Const -> true
  | Op.Range -> loop u
  | Op.Add | Op.Mul ->
      Dtype.equal (dtype u) Dtype.Weak_int && List.for_all loop_bound (src u)
  | _ -> false

let kernel_graph : t =
  Pattern_matcher.fold
    (fun () -> [
      accept (pat [ Op.Sink ] ~dtype:[ Dtype.Void ]);
      accept (pat [ Op.Const ] ~src:[]);
      accept (pat [ Op.Cast ] ~src:[ pat [ Op.Const ] ~src:[] ]);
      decide (pat [ Op.Stack ] ~name:"s") "s" (fun s ->
          if List.for_all (fun x -> op x = Op.Param || loop_bound x) (src s)
          then Some true
          else None);
      accept (pat [ Op.Param ] ~src:[]);
      check (pat [ Op.Buffer ] ~name:"x") "x" (fun x ->
          valid_device_range (param_of x).device (src x)
          && List.mem (addrspace x) Dtype.[ Some Global; Some Local; Some Reg ]);
      check (pat [ Op.Alloc ] ~name:"x") "x" (fun x ->
          addrspace x = Some Dtype.Global
          && valid_device_range (param_of x).device (src x));
      accept (pat [ Op.Bitcast ]);
      check (pat [ Op.Mstack ] ~name:"x") "x" mstack_fits;
      check (pat [ Op.Mselect ] ~name:"x") "x" mselect_fits;
      (* An open device range is bound per device at launch; a loop range runs
         the calls an end closes it around. *)
      check (pat [ Op.Range ] ~name:"r") "r" (fun r ->
          Axis_type.equal (axis_type r) Axis_type.Device || loop r);
      check (pat [ Op.End ] ~name:"e") "e" (fun e ->
          op (nth e 0) = Op.Call
          && List.for_all (fun r -> op r = Op.Range && loop r) (List.tl (src e)));
      check
        (pat [ Op.Backedge ] ~name:"e"
           ~src:[ pat [ Op.Call ]; Upat.wild; Upat.wild ])
        "e"
        (fun e ->
          let r = nth e 1 in
          (op r = Op.Range && loop r) || op r = Op.Const);
      (* A call in a loop reads a view of storage that moves with the loop's
         ranges, its bounds weak integer arithmetic on them. *)
      check (pat [ Op.Shrink ] ~allow_any_len:true ~name:"v") "v" (fun v ->
          let rs = Nodes.to_list (ranges v) in
          rs <> [] && List.for_all loop rs);
      check
        (pat [ Op.Add; Op.Mul ] ~dtype:[ Dtype.Weak_int ] ~name:"x")
        "x"
        (fun x -> List.for_all loop_bound (src x));
      check
        (pat [ Op.Call ]
           ~src:[ Upat.v ~op:opaque_call_bodies () ]
           ~allow_any_len:true ~name:"x")
        "x" args_fit;
      accept
        (pat [ Op.After ] ~allow_any_len:true
           ~src:
             [
               Upat.v
                 ~op:
                   (Op.Set.union Op.Set.movement
                      (ops
                         Op.
                           [
                             Param;
                             After;
                             Buffer;
                             Alloc;
                             Mstack;
                             Mselect;
                             Bitcast;
                             Reshape;
                           ]))
                 ();
             ]);
    ])
