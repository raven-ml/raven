(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Forward-mode differentiation as an interpreter of Nx operations.

   Tangents propagate eagerly: every operation computes its primal by
   evaluating the operation in the enclosing interpretation, and, if any input
   has a tangent in the store, computes and stores the output tangent
   immediately. There is no tape and no second pass. A tensor absent from the
   store is a constant with zero tangent.

   Tangent arithmetic runs in the enclosing interpretation too, so composing
   with grad (forward-over-reverse and reverse-over-forward) and nesting jvp
   both work.

   Every operation is matched explicitly, and the compiler checks it, with the
   same deliberate categories as the reverse engine: zero-derivative operations
   stay inactive; operations with no rule yet raise when an input is active. *)

open Nx.Op
open Prim
module T = Nx

let err_no_rule op =
  invalid_arg
    (Printf.sprintf
       "Rune: the tangent of %s is not implemented; detach its input if \
        differentiation should not flow through it"
       op)

let err_quant () =
  invalid_arg
    "Rune: a part of a quantised weight is differentiated; capture the weight, \
     or build it from Rune.detached tensors"

(* A scan's leaves followed by the tangents of its active ones:
   [zip actives leaves ts] pairs each active leaf with its tangent. *)
let rec zip actives leaves ts =
  match (actives, leaves, ts) with
  | [], [], [] -> []
  | true :: actives, l :: leaves, t :: ts -> (l, t) :: zip actives leaves ts
  | false :: actives, _ :: leaves, ts -> zip actives leaves ts
  | _ -> assert false

(* [install tangents f] is [f ()] with its operations' tangents in
   [tangents]: an interpreter of the operations, and a handler of the effects
   its rules answer, installed outside the interpreter. *)
let rec install : type a. Tensor_map.t -> (unit -> a) -> a =
 fun tangents f ->
  let open Effect.Deep in
  let paused = ref 0 in
  let tangent x = Tensor_map.find tangents x in
  let active x = Option.is_some (tangent x) in
  let tan_or_zeros x =
    match tangent x with Some dx -> dx | None -> T.zeros_like x
  in
  (* Materialize stored tangents: rule outputs can be lazy views (broadcasts,
     transposes), and later rules may reshape them. *)
  let set_tangent out v = Tensor_map.set tangents out (T.contiguous v) in

  (* [lift1 out x dfun] stores [dfun dx] as the tangent of [out] when [x] has
     tangent [dx]. *)
  let lift1 (type a b c d) (out : (a, b) T.t) (x : (c, d) T.t)
      (dfun : (c, d) T.t -> (a, b) T.t) =
    Tensor_map.fresh out x;
    (match tangent x with None -> () | Some dx -> set_tangent out (dfun dx));
    out
  in

  (* [lift2 out a b fa fb] stores the sum of [fa da] and [fb db] as the tangent
     of [out], with a term only for an input that has a tangent. An input
     without one adds no term: its coefficient can be infinite or NaN where the
     input is constant, as the exponent's [log a] of a power at [a = 0], and
     must not meet a zero. *)
  let lift2 (type a b) (out : (a, b) T.t) (a_in : (a, b) T.t)
      (b_in : (a, b) T.t) (fa : (a, b) T.t -> (a, b) T.t)
      (fb : (a, b) T.t -> (a, b) T.t) =
    Tensor_map.fresh out a_in;
    Tensor_map.fresh out b_in;
    (match (tangent a_in, tangent b_in) with
    | None, None -> ()
    | Some da, None -> set_tangent out (fa da)
    | None, Some db -> set_tangent out (fb db)
    | Some da, Some db -> set_tangent out (T.add (fa da) (fb db)));
    out
  in

  let no_rule (type c) (op : string) (inputs_active : bool) (out : unit -> c) =
    if inputs_active then err_no_rule op else out ()
  in

  (* Each operation: its primal is the operation itself, evaluated in the
     enclosing interpretation, and its tangent is computed at once when an
     operand is active. A linear operation's tangent is the operation on the
     tangent. *)
  let run : type c. c Nx.Op.t -> c =
   fun op ->
    match[@warning "@4@8"] op with
    | Unary (k, x) -> (
        match k with
        (* On complex dtypes [sign z = z / |z|] is not piecewise constant. *)
        | Sign when Nx_dtype.is_complex (T.dtype x) ->
            let out = eval op in
            lift1 out x (Derivs.sign_push x out)
        | Sign | Trunc | Ceil | Floor | Round -> eval op
        | Neg -> lift1 (eval op) x T.neg
        | Sin -> lift1 (eval op) x (fun dx -> T.mul dx (T.cos x))
        | Cos -> lift1 (eval op) x (fun dx -> T.mul dx (T.neg (T.sin x)))
        | Tan -> lift1 (eval op) x (fun dx -> T.mul dx (Derivs.tan' x))
        | Asin -> lift1 (eval op) x (fun dx -> T.mul dx (Derivs.asin' x))
        | Acos ->
            lift1 (eval op) x (fun dx -> T.mul dx (T.neg (Derivs.asin' x)))
        | Atan -> lift1 (eval op) x (fun dx -> T.mul dx (Derivs.atan' x))
        | Sinh -> lift1 (eval op) x (fun dx -> T.mul dx (T.cosh x))
        | Cosh -> lift1 (eval op) x (fun dx -> T.mul dx (T.sinh x))
        | Tanh ->
            let out = eval op in
            lift1 out x (fun dx -> T.mul dx (Derivs.tanh' out))
        | Exp ->
            let out = eval op in
            lift1 out x (fun dx -> T.mul dx out)
        | Log -> lift1 (eval op) x (fun dx -> T.mul dx (T.recip x))
        | Sqrt ->
            let out = eval op in
            lift1 out x (fun dx -> T.mul dx (Derivs.sqrt' out))
        | Recip -> lift1 (eval op) x (fun dx -> T.mul dx (Derivs.recip' x))
        (* On complex dtypes [abs] is the modulus: real-valued, and not
           holomorphic. Its pushforward conjugates the direction — going
           through [sign z] itself would flip the sign of the imaginary
           contribution — and keeps the real part of the product, since a
           real-valued output cannot move in the imaginary direction. Both are
           the identity on real dtypes. *)
        | Abs ->
            lift1 (eval op) x (fun dx ->
                Derivs.real_part (T.mul dx (T.conjugate (T.sign x))))
        | Erf -> lift1 (eval op) x (fun dx -> T.mul dx (Derivs.erf' x)))
    | Binary (k, a, b) -> (
        match k with
        | Idiv | And | Or | Xor -> eval op
        | Add -> lift2 (eval op) a b Fun.id Fun.id
        | Sub -> lift2 (eval op) a b Fun.id T.neg
        | Mul ->
            lift2 (eval op) a b (fun da -> T.mul da b) (fun db -> T.mul a db)
        | Fdiv ->
            lift2 (eval op) a b
              (fun da -> T.div da b)
              (fun db -> T.neg (T.mul (T.div a (T.mul b b)) db))
        | Pow ->
            let out = eval op in
            lift2 out a b
              (fun da -> T.mul da (Derivs.pow_wrt_base a b))
              (fun db -> T.mul db (Derivs.pow_wrt_exp a out))
        | Maximum ->
            let out = eval op in
            let mask = lazy (T.cast (T.dtype out) (T.greater a b)) in
            lift2 out a b
              (fun da -> T.mul da (Lazy.force mask))
              (fun db ->
                let mask = Lazy.force mask in
                T.mul db (T.rsub_s (Derivs.one_like mask) mask))
        | Minimum ->
            let out = eval op in
            let mask = lazy (T.cast (T.dtype out) (T.less a b)) in
            lift2 out a b
              (fun da -> T.mul da (Lazy.force mask))
              (fun db ->
                let mask = Lazy.force mask in
                T.mul db (T.rsub_s (Derivs.one_like mask) mask))
        | Atan2 ->
            let denom = lazy (T.add (T.mul a a) (T.mul b b)) in
            lift2 (eval op) a b
              (fun da -> T.mul da (T.div b (Lazy.force denom)))
              (fun db -> T.neg (T.mul db (T.div a (Lazy.force denom))))
        | Mod -> no_rule "mod" (active a || active b) (fun () -> eval op))
    | Compare _ | Arg_reduce _ | Argsort _ | Threefry _ | Read _ -> eval op
    | Convert (Bitcast, _, _) -> eval op
    | Convert (Cast, dtype, x) -> lift1 (eval op) x (fun dx -> T.cast dtype dx)
    | Where (condition, if_true, if_false) ->
        let out = eval op in
        if active if_true || active if_false then begin
          let mask = T.cast (T.dtype out) condition in
          let dt_ = tan_or_zeros if_true and df = tan_or_zeros if_false in
          set_tangent out
            (T.add (T.mul dt_ mask)
               (T.mul df (T.rsub_s (Derivs.one_like mask) mask)))
        end;
        out
    (* Placement and movement are linear: the tangent moves with its primal. *)
    | Place (p, x) -> lift1 (eval op) x (place p)
    | Move (x, m) -> lift1 (eval op) x (fun dx -> move dx m)
    | Contiguous x -> lift1 (eval op) x Fun.id
    (* The fill value is a constant: the tangent pads with zero. *)
    | Pad (padding, _, x) ->
        lift1 (eval op) x (fun dx -> pad padding (Nx_dtype.zero (T.dtype x)) dx)
    | Cat (axis, xs) ->
        let out = eval op in
        if List.exists active xs then
          set_tangent out (cat ~axis (List.map tan_or_zeros xs));
        out
    (* Reductions *)
    | Reduce (Sum, axes, x) ->
        lift1 (eval op) x (fun dx -> T.sum dx ~axes:(Array.to_list axes))
    | Reduce ((Max | Min), axes, x) ->
        let out = eval op in
        let axes = Array.to_list axes in
        lift1 out x (fun dx ->
            T.sum ~axes (T.mul dx (Derivs.extrema' ~axes x out)))
    | Reduce (Prod, axes, x) ->
        let out = eval op in
        let axes = Array.to_list axes in
        lift1 out x (fun dx ->
            T.sum ~axes (T.mul dx (Derivs.prod' ~axes x out)))
    (* Sorting: a sort is a gather at the argsort indices. *)
    | Sort { descending; axis; x } ->
        lift1 (eval op) x (fun dx ->
            let indices = argsort ~descending ~axis x in
            gather ~axis indices dx)
    (* Scans *)
    | Scan (k, axis, x) ->
        let out = eval op in
        lift1 out x (fun dx ->
            match k with
            | Sum -> scan Sum ~axis dx
            | Prod ->
                (* d cumprod_k = cumprod_k * sum_{i<=k} dx_i / x_i; requires
                   nonzero inputs, like the reverse rule. *)
                let ratio = T.div dx x in
                T.mul out (scan Sum ~axis ratio)
            | Max | Min ->
                (* The tangent flows from positions where the running extremum
                   strictly improves. *)
                let shape = T.shape out in
                let ndim = Array.length shape in
                let axis_norm = if axis < 0 then axis + ndim else axis in
                let dt = T.dtype x in
                let boundary =
                  match k with
                  | Max -> Nx_dtype.min_value dt
                  | Sum | Prod | Min -> Nx_dtype.max_value dt
                in
                let pad_left =
                  Array.mapi
                    (fun i _ -> if i = axis_norm then (1, 0) else (0, 0))
                    shape
                in
                let padded = T.pad pad_left boundary out in
                let slice_specs = Array.map (fun dim -> T.R (0, dim)) shape in
                let shifted = T.slice (Array.to_list slice_specs) padded in
                let active_mask =
                  match k with
                  | Max -> T.greater out shifted
                  | Sum | Prod | Min -> T.less out shifted
                in
                (* Positions where the extremum does not improve keep a zero
                   tangent rather than carrying the previous extremum's
                   tangent; this matches the reverse rule (they are transposes
                   of each other). *)
                T.mul dx (T.cast dt active_mask))
    (* Gather / scatter *)
    | Gather (axis, indices, data) ->
        lift1 (eval op) data (fun dx -> gather ~axis indices dx)
    | Scatter { mode; unique; axis; indices; updates; into } ->
        let out = eval op in
        if active into || active updates then begin
          let d_template =
            match mode with
            | `Add -> tan_or_zeros into
            | `Set ->
                let mask =
                  scatter ~mode:`Set ~unique ~axis ~indices
                    ~updates:(T.zeros_like updates) (T.ones_like into)
                in
                T.mul (tan_or_zeros into) mask
          in
          let d_updates =
            scatter ~mode ~unique ~axis ~indices ~updates:(tan_or_zeros updates)
              (T.zeros_like into)
          in
          set_tangent out (T.add d_template d_updates)
        end;
        out
    (* The window write is linear in [x] and [v] together. *)
    | Update (x, starts, v) ->
        let out = eval op in
        if active x || active v then
          set_tangent out (update (tan_or_zeros x) ~starts (tan_or_zeros v));
        out
    (* Windowing: unfold and fold are linear. *)
    | Unfold { kernel_size; stride; dilation; padding; x } ->
        lift1 (eval op) x (fun dx ->
            unfold dx ~kernel_size ~stride ~dilation ~padding)
    | Fold { output_size; kernel_size; stride; dilation; padding; x } ->
        lift1 (eval op) x (fun dx ->
            fold dx ~output_size ~kernel_size ~stride ~dilation ~padding)
    (* Matrix multiplication *)
    | Matmul (a, b) ->
        let out = eval op in
        (match (tangent a, tangent b) with
        | None, None -> ()
        | Some da, None -> set_tangent out (matmul da b)
        | None, Some db -> set_tangent out (matmul a db)
        | Some da, Some db ->
            set_tangent out (T.add (matmul da b) (matmul a db)));
        out
    (* FFT: linear operations apply to the tangent. *)
    | Fft { inverse; axes; x } ->
        lift1 (eval op) x (fun dx -> fft ~inverse ~axes dx)
    | Rfft { dtype; axes; x } ->
        lift1 (eval op) x (fun dx -> rfft dtype ~axes dx)
    | Irfft { dtype; axes; s; x } ->
        lift1 (eval op) x (fun dx -> irfft ?s dtype ~axes dx)
    (* Linear algebra *)
    (* The factor reads the Hermitian matrix H that A's strict lower triangle
       and the real part of its diagonal name, H = L Lᴴ, so dL = L
       Φ(L^-1 dH L^-H), with Φ the lower triangle less half the diagonal.
       Under [upper] the factor is U = Lᴴ. *)
    | Cholesky { upper; x } ->
        let out = eval op in
        lift1 out x (fun da ->
            let l = if upper then Derivs.adjoint out else out in
            let dh =
              let low = T.tril ~k:(-1) da in
              T.add
                (T.add low (Derivs.adjoint low))
                (Derivs.diag_matrix (Derivs.real_part (T.diagonal da)))
            in
            let left =
              solve_triangular ~upper:false ~transpose:false ~unit_diag:false l
            in
            let m = Derivs.adjoint (left (Derivs.adjoint (left dh))) in
            let phi =
              let diag_m = T.diagonal m in
              let two = Nx_dtype.of_float (T.dtype diag_m) 2.0 in
              T.sub (T.tril m) (Derivs.diag_matrix (T.div_s diag_m two))
            in
            let dl = T.matmul l phi in
            if upper then Derivs.adjoint dl else dl)
    | Solve_triangular { upper; transpose; unit_diag; a; b } ->
        let out = eval op in
        if active a || active b then begin
          (* A_op X = B, so A_op dX = dB - dA_op X, with dA restricted to the
             triangle the solve reads. A_op is Aᴴ under [transpose]: the
             conjugate transpose on complex. *)
          let db = tan_or_zeros b in
          let rhs =
            match tangent a with
            | None -> db
            | Some da ->
                let da_used =
                  let tri = if upper then T.triu da else T.tril da in
                  if unit_diag then
                    T.sub tri (Derivs.diag_matrix (T.diagonal tri))
                  else tri
                in
                let da_op =
                  if transpose then T.conjugate (T.matrix_transpose da_used)
                  else da_used
                in
                let out_2d, was_1d =
                  if T.ndim out = T.ndim a - 1 then
                    (T.unsqueeze ~axes:[ -1 ] out, true)
                  else (out, false)
                in
                let prod = T.matmul da_op out_2d in
                let prod =
                  if was_1d then T.reshape (T.shape b) prod else prod
                in
                T.sub db prod
          in
          set_tangent out (solve_triangular ~upper ~transpose ~unit_diag a rhs)
        end;
        out
    | Qr { x; _ } -> if active x then err_no_rule "qr" else eval op
    | Lu x ->
        let ((packed, _, perm) as out) = eval op in
        (match tangent x with
        | None -> ()
        | Some da ->
            if T.dim (-1) x <> T.dim (-2) x then
              err_no_rule "lu of a rectangular matrix";
            (* P A = L U. X = L^-1 Nx.P dA U^-1 is L^-1 dL, strictly lower, plus
               dU U^-1, upper: dL = L tril_-1(X) and dU = triu(X) U, packed as
               the factors are. *)
            let pda =
              T.take_along_axis ~axis:(-2)
                ~indices:
                  (T.broadcast_to (T.shape da) (T.unsqueeze ~axes:[ -1 ] perm))
                da
            in
            let y =
              solve_triangular ~upper:false ~transpose:false ~unit_diag:true
                packed pda
            in
            let x =
              T.matrix_transpose
                (solve_triangular ~upper:false ~transpose:false
                   ~unit_diag:false (T.matrix_transpose packed)
                   (T.matrix_transpose y))
            in
            let l =
              T.add (T.tril ~k:(-1) packed)
                (T.eye (T.dtype packed) (T.dim (-1) packed))
            in
            set_tangent packed
              (T.add
                 (T.matmul l (T.tril ~k:(-1) x))
                 (T.matmul (T.triu x) (T.triu packed))));
        out
    | Svd { x; _ } -> no_rule "svd" (active x) (fun () -> eval op)
    | Eig { vectors; x } ->
        no_rule (if vectors then "eig" else "eigvals") (active x) (fun () ->
            eval op)
    | Eigh { vectors; x } ->
        no_rule (if vectors then "eigh" else "eigvalsh") (active x) (fun () ->
            eval op)
  in

  let rule : type c. c Effect.t -> (unit -> c) option =
   fun eff ->
    match eff with
    | Pause.E_pause -> Some (Pause.hold paused)
    | _ when !paused > 0 -> None
    (* Scan. When a stager lies beyond, the scan passes on as the scan of its
       jvp: the carry and the rows gain the tangents of their active leaves,
       the outputs those of theirs, and the step runs the body under a nested
       instance of this handler. A carry whose tangent is zero at [init] gains
       one only once a step makes it active: the step then aborts its run with
       [Grow] and the scan passes on again, carrying it. Otherwise, the eager
       fold runs under a nested instance of this handler, and every step's
       operations acquire their tangents. *)
    | Scan.E_scan_probe -> Some Scan.probe
    | Scan.E_scan req ->
        Some
          (fun () ->
            let fold () = install tangents (fun () -> Scan.eager req) in
            Scan.pass_on ~fold @@ fun () ->
              let exception Grow of bool list in
              let flags = List.map (fun (Nx.P l) -> active l) in
              let tangents_of actives leaves =
                List.concat
                  (List.map2
                     (fun a (Nx.P l) ->
                       if a then [ Nx.P (tan_or_zeros l) ] else [])
                     actives leaves)
              in
              let seed actives leaves ts =
                List.iter
                  (fun (Nx.P l, d) ->
                    Tensor_map.set tangents l (T.unpack (T.dtype l) d))
                  (zip actives leaves ts)
              in
              let nc = List.length req.req_carry
              and nx = List.length req.req_xs in
              let rows = flags req.req_xs in
              let rec attempt carried =
                let outputs = ref [] in
                let run c x =
                  let c, dc = Scan.split nc c and x, dx = Scan.split nx x in
                  seed carried c dc;
                  seed rows x dx;
                  let c', y =
                    install tangents (fun () -> req.req_step.run c x)
                  in
                  let next = flags c' in
                  if List.exists2 (fun a a' -> a' && not a) carried next then
                    raise (Grow (List.map2 ( || ) carried next));
                  outputs := flags y;
                  (c' @ tangents_of carried c', y @ tangents_of !outputs y)
                in
                match
                  Effect.perform
                    (Scan.E_scan
                       {
                         req with
                         req_carry =
                           req.req_carry @ tangents_of carried req.req_carry;
                         req_xs = req.req_xs @ tangents_of rows req.req_xs;
                         req_step = { run };
                       })
                with
                | res ->
                    let set actives (leaves, ts) =
                      List.iter
                        (fun (Nx.P l, d) ->
                          set_tangent l (T.unpack (T.dtype l) d))
                        (zip actives leaves ts);
                      leaves
                    in
                    let r_carry = set carried (Scan.split nc res.r_carry) in
                    let r_ys =
                      set !outputs
                        (Scan.split (List.length !outputs) res.r_ys)
                    in
                    { Scan.r_carry; r_ys }
                (* The aborted run's slot tensors are never reached again. *)
                | exception Grow carried -> attempt carried
              in
              attempt (flags req.req_carry))
    (* A gather is linear. *)
    | Axis.E_lanes { axis; t_in } ->
        Some (fun () -> lift1 (Axis.lanes axis t_in) t_in (Axis.lanes axis))
    (* Custom rules. *)
    | Custom.E_custom_jvp
        (Custom.Jvp_call { params_s; result_s; params; f; jvp }) ->
        Some
          (fun () ->
            if
              not
                (Nx.Ptree.fold params_s
                   (fun _ leaf any -> any || active leaf)
                   params false)
            then f params
            else begin
              let dparams =
                Nx.Ptree.map params_s (fun _ leaf -> tan_or_zeros leaf) params
              in
              let y, dy = jvp params dparams in
              (* A result that is one of the parameters is aliased, so the
                 parameter keeps its own tangent. *)
              let y = Structure.aliases result_s y in
              let set path yl dyl =
                if T.shape yl <> T.shape dyl then
                  invalid_arg
                    (Printf.sprintf
                       "Rune.custom_jvp: %s: tangent shape [%s] does not \
                        match result shape [%s]"
                       (Structure.describe path)
                       (Structure.shape_string (T.shape dyl))
                       (Structure.shape_string (T.shape yl)));
                set_tangent yl dyl;
                yl
              in
              ignore
                (Structure.map2 "Rune.custom_jvp" result_s ~this:"the result"
                   ~that:"jvp's tangents" set y dy);
              y
            end)
    (* A custom_vjp has no forward rule. One whose result holds no tensor
       has nothing to differentiate, and its function runs. *)
    | Custom.E_custom_vjp
        (Custom.Vjp_call { params_s; result_s; params; fwd; _ }) ->
        Some
          (fun () ->
            let y = fst (fwd params) in
            if
              Structure.holds_tensor result_s y
              && Nx.Ptree.fold params_s
                   (fun _ leaf any -> any || active leaf)
                   params false
            then
              invalid_arg
                "Rune: a custom_vjp function is not forward-differentiable; \
                 define a custom_jvp rule instead"
            else y)
    (* Gradient checkpointing. The call passes on as the remat of [f]'s jvp: a
       function of the call's arguments that gives each argument it receives
       the tangent of the call's argument at its position, runs [f] under this
       handler and returns [f]'s results and the tangents of its active ones,
       so that an enclosing transformation sees the tangents as results of the
       remat. The tangents of the arguments, like those of the tensors [f]
       captures, are tensors the function closes over. *)
    | Remat.E_remat (Remat.Call { params_s; result_s; params; f; residuals })
      ->
        Some
          (fun () ->
            let dparams =
              List.map
                (fun (Nx.P p) -> Option.map (fun d -> Nx.P d) (tangent p))
                (fst (Nx.Ptree.flatten params_s params))
            in
            let active_out = ref [] in
            let f' params =
              List.iter2
                (fun (Nx.P p) d ->
                  Option.iter
                    (fun d ->
                      Tensor_map.set tangents p (T.unpack (T.dtype p) d))
                    d)
                (fst (Nx.Ptree.flatten params_s params))
                dparams;
              let y = install tangents (fun () -> f params) in
              let ys = fst (Nx.Ptree.flatten result_s y) in
              active_out := List.map (fun (Nx.P l) -> active l) ys;
              ( y,
                List.filter_map
                  (fun (Nx.P l) -> Option.map (fun d -> Nx.P d) (tangent l))
                  ys )
            in
            let y, dy =
              Remat.run
                (Remat.Call
                   {
                     params_s;
                     result_s = Nx.Ptree.pair result_s Structure.packed_list;
                     params;
                     f = f';
                     residuals;
                   })
            in
            ignore
              (List.fold_left2
                 (fun dy (Nx.P l) is_active ->
                   match (is_active, dy) with
                   | true, d :: dy ->
                       set_tangent l (T.unpack (T.dtype l) d);
                       dy
                   | true, [] -> assert false
                   | false, dy -> dy)
                 dy
                 (fst (Nx.Ptree.flatten result_s y))
                 !active_out);
            y)
    (* The barrier is the identity, and the tangents pass through it with
       their values: an output's tangent is its value's tangent after the
       barrier. A tangent read around the barrier would let the
       recomputation's tangents share the forward pass's. *)
    | Remat.E_barrier { values; after } ->
        let tangents =
          List.filter_map
            (fun (Nx.P v) -> Option.map (fun d -> Nx.P d) (tangent v))
            values
        in
        if tangents = [] then None
        else
          Some
            (fun () ->
              let out = Remat.barrier ~after (values @ tangents) in
              let rec split vs out =
                match (vs, out) with
                | [], tangents -> ([], tangents)
                | _ :: vs, o :: out ->
                    let os, tangents = split vs out in
                    (o :: os, tangents)
                | _ :: _, [] -> assert false
              in
              let out, tangents = split values out in
              ignore
                (List.fold_left2
                   (fun tangents (Nx.P v) o ->
                     match (tangent v, tangents) with
                     | Some _, d :: tangents ->
                         set_tangent
                           (T.unpack (T.dtype v) o)
                           (T.unpack (T.dtype v) d);
                         tangents
                     | Some _, [] -> assert false
                     | None, tangents -> tangents)
                   tangents values out);
              out)
    (* Quantised products. A weight is never differentiated; the tangent of a
       product is the product of the tangent of [x]. *)
    | Nx_quant.Effect.E_quant
        { w = Nx_quant.Mxfp4 { codes; scales } as w; op } ->
        Some
          (fun () ->
            if active codes || active scales then err_quant ();
            let y = Nx_quant.Effect.perform w op in
            (match op with
            | Apply { ids; x; transpose } -> (
                match tangent x with
                | None -> ()
                | Some dx ->
                    set_tangent y
                      (Nx_quant.Effect.perform w
                         (Apply { ids; x = dx; transpose })))
            | Dequant _ -> ());
            y)
    | _ -> None
  in
  let effc : type c. c Effect.t -> ((c, a) continuation -> a) option =
   fun eff -> Option.map Answer.deliver (rule eff)
  in
  (* While paused, every operation passes on as it is. *)
  let run op = if !paused > 0 then eval op else run op in
  match_with
    (fun () -> intercept { run } f)
    ()
    { retc = Fun.id; exnc = raise; effc }
