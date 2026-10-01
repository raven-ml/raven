(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reverse-mode differentiation as an interpreter of Nx operations.

   Forward pass: every operation computes its primal by evaluating the
   operation in the enclosing interpretation (so nested grads compose), and, if
   any input is tracked on the tape, marks its output tracked and records a pull
   thunk. Operations whose inputs are all untracked are constants with respect
   to the differentiated inputs and are recorded nowhere.

   Backward pass: [Tape.backward] runs the pull thunks in reverse. A pull thunk
   reads its output cotangent from the tape and accumulates input contributions.
   Pull thunks execute ordinary Nx operations, so an enclosing grad
   differentiates them: higher-order derivatives work.

   Every operation is matched explicitly, and the compiler checks it: a new one
   breaks this build. Operations without a gradient fall into two deliberate
   categories: zero derivative (comparisons, bitwise and integer ops, bitcasts,
   rounding, argmax/argmin/argsort, RNG, reads), whose outputs stay untracked,
   which yields the correct zero gradient; and no rule implemented (svd, eig,
   eigh, lanes, mod), which raise when an input is tracked instead of silently
   producing a zero gradient — detach the input if differentiation should not
   flow through it. *)

open Nx.Op
open Prim
module T = Nx

(* Reduce a cotangent to the shape of a broadcast source. *)
let unbroadcast (type a b) (g : (a, b) T.t) (src_shape : int array) : (a, b) T.t
    =
  let dst_shape = T.shape g in
  if src_shape = dst_shape then g
  else
    let src_rank = Array.length src_shape in
    let dst_rank = Array.length dst_shape in
    let axes = ref [] in
    for i = 0 to dst_rank - src_rank - 1 do
      axes := i :: !axes
    done;
    for i = 0 to src_rank - 1 do
      if src_shape.(i) = 1 && dst_shape.(i + (dst_rank - src_rank)) > 1 then
        axes := (i + (dst_rank - src_rank)) :: !axes
    done;
    match !axes with
    | [] -> g
    | ax ->
        let summed = T.sum g ~axes:ax ~keepdims:true in
        if T.shape summed <> src_shape then T.reshape src_shape summed
        else summed

(* Single-axis resizes: [pad_axis ax (lo, hi) v x] pads [x] along [ax] with [v];
   [shrink_axis ax (lo, hi) x] keeps indices [lo, hi) along [ax]. Other axes are
   untouched. *)
let pad_axis ax (lo, hi) v x =
  let cfg = Array.make (T.ndim x) (0, 0) in
  cfg.(ax) <- (lo, hi);
  T.pad cfg v x

let shrink_axis ax (lo, hi) x =
  let lim = Array.map (fun d -> (0, d)) (T.shape x) in
  lim.(ax) <- (lo, hi);
  T.shrink lim x

let err_no_rule op =
  invalid_arg
    (Printf.sprintf
       "Rune: the gradient of %s is not implemented; detach its input if \
        differentiation should not flow through it"
       op)

let err_quant () =
  invalid_arg
    "Rune: a part of a quantised weight is differentiated; capture the weight, \
     or build it from Rune.detached tensors"

(* Handler *)

(* [install tape f] is [f ()] with its operations differentiated on [tape]:
   an interpreter of the operations, and a handler of the effects its rules
   answer, installed outside the interpreter. *)
let rec install : type a. Tape.t -> (unit -> a) -> a =
 fun tape f ->
  let open Effect.Deep in
  let tracked x = Tape.tracked tape x in
  let track x = Tape.track tape x in
  let paused = ref 0 in

  (* [pull1 out x f] records: cotangent of [x] += [f] applied to the cotangent
     of [out]. Skips recording when [x] is untracked. *)
  let pull1 (type a b c d) (out : (a, b) T.t) (x : (c, d) T.t)
      (f : (a, b) T.t -> (c, d) T.t) =
    Tensor_map.fresh out x;
    if tracked x then begin
      track out;
      Tape.record tape (fun () ->
          match Tape.find tape out with
          | None -> ()
          | Some g -> Tape.accumulate tape x (f g))
    end;
    out
  in

  (* [pull2 out a b fa fb] is [pull1] for binary arithmetic: contributions are
     reduced back to each input's shape to undo broadcasting. *)
  let pull2 (type a b) (out : (a, b) T.t) (a_in : (a, b) T.t)
      (b_in : (a, b) T.t) (fa : (a, b) T.t -> (a, b) T.t) (fb : (a, b) T.t -> (a, b) T.t) =
    Tensor_map.fresh out a_in;
    Tensor_map.fresh out b_in;
    let ta = tracked a_in and tb = tracked b_in in
    if ta || tb then begin
      track out;
      Tape.record tape (fun () ->
          match Tape.find tape out with
          | None -> ()
          | Some g ->
              if ta then
                Tape.accumulate tape a_in (unbroadcast (fa g) (T.shape a_in));
              if tb then
                Tape.accumulate tape b_in (unbroadcast (fb g) (T.shape b_in)))
    end;
    out
  in

  let no_rule (type c) (op : string) (inputs_tracked : bool) (out : unit -> c) =
    if inputs_tracked then err_no_rule op else out ()
  in

  (* A function this handler runs in its own context, as a custom call's, runs
     past every handler between the call and this one: on a rerun tape its
     additions to a total are dropped here. *)
  let dropped f x = Total.dropping (fun () -> f x) in
  let own f x = if Tape.rerun tape then dropped f x else f x in

  (* Each operation: its primal is the operation itself, evaluated in the
     enclosing interpretation (so nested grads compose), and its pullback is
     recorded when an operand is tracked. Operations without a gradient fall
     into two deliberate categories: zero derivative (comparisons, bitwise and
     integer ops, bitcasts, rounding, argmax/argmin/argsort, RNG, reads), whose
     outputs stay untracked; and no rule implemented (svd, eig, eigh, mod),
     which raise when an operand is tracked. *)
  let run : type c. c Nx.Op.t -> c =
   fun op ->
    match[@warning "@4@8"] op with
    | Unary (k, x) -> (
        match k with
        (* On complex dtypes [sign z = z / |z|] is not piecewise constant. *)
        | Sign when Nx_dtype.is_complex (T.dtype x) ->
            let out = eval op in
            pull1 out x (Derivs.sign_pull x out)
        | Sign | Trunc | Ceil | Floor | Round -> eval op
        | Neg -> pull1 (eval op) x T.neg
        | Sin -> pull1 (eval op) x (fun g -> T.mul g (T.cos x))
        | Cos -> pull1 (eval op) x (fun g -> T.mul g (T.neg (T.sin x)))
        | Tan -> pull1 (eval op) x (fun g -> T.mul g (Derivs.tan' x))
        | Asin -> pull1 (eval op) x (fun g -> T.mul g (Derivs.asin' x))
        | Acos -> pull1 (eval op) x (fun g -> T.mul g (T.neg (Derivs.asin' x)))
        | Atan -> pull1 (eval op) x (fun g -> T.mul g (Derivs.atan' x))
        | Sinh -> pull1 (eval op) x (fun g -> T.mul g (T.cosh x))
        | Cosh -> pull1 (eval op) x (fun g -> T.mul g (T.sinh x))
        | Tanh ->
            let out = eval op in
            pull1 out x (fun g -> T.mul g (Derivs.tanh' out))
        | Exp ->
            let out = eval op in
            pull1 out x (fun g -> T.mul g out)
        | Log -> pull1 (eval op) x (fun g -> T.mul g (T.recip x))
        | Sqrt ->
            let out = eval op in
            pull1 out x (fun g -> T.mul g (Derivs.sqrt' out))
        | Recip -> pull1 (eval op) x (fun g -> T.mul g (Derivs.recip' x))
        (* On complex dtypes [abs] is the modulus: real-valued, and not
           holomorphic. Its pullback conjugates the direction — going through
           [sign z] itself would flip the sign of the imaginary contribution —
           and keeps only the real part of the cotangent, since the imaginary
           component of a real-valued output moves nothing. Both are the
           identity on real dtypes. *)
        | Abs ->
            pull1 (eval op) x (fun g ->
                T.mul (Derivs.real_part g) (T.conjugate (T.sign x)))
        | Erf -> pull1 (eval op) x (fun g -> T.mul g (Derivs.erf' x)))
    | Binary (k, a, b) -> (
        match k with
        | Idiv | And | Or | Xor -> eval op
        | Add -> pull2 (eval op) a b Fun.id Fun.id
        | Sub -> pull2 (eval op) a b Fun.id T.neg
        | Mul -> pull2 (eval op) a b (fun g -> T.mul g b) (fun g -> T.mul g a)
        | Fdiv ->
            pull2 (eval op) a b
              (fun g -> T.div g b)
              (fun g -> T.mul (T.neg g) (T.div a (T.mul b b)))
        | Pow ->
            let out = eval op in
            pull2 out a b
              (fun g -> T.mul g (Derivs.pow_wrt_base a b))
              (fun g -> T.mul g (Derivs.pow_wrt_exp a out))
        | Maximum ->
            let mask g = T.cast (T.dtype g) (T.greater a b) in
            pull2 (eval op) a b
              (fun g -> T.mul g (mask g))
              (fun g ->
                let m = mask g in
                T.mul g (T.rsub_s (Derivs.one_like m) m))
        | Minimum ->
            let mask g = T.cast (T.dtype g) (T.less a b) in
            pull2 (eval op) a b
              (fun g -> T.mul g (mask g))
              (fun g ->
                let m = mask g in
                T.mul g (T.rsub_s (Derivs.one_like m) m))
        | Atan2 ->
            let denom () = T.add (T.mul a a) (T.mul b b) in
            pull2 (eval op) a b
              (fun g -> T.mul g (T.div b (denom ())))
              (fun g -> T.mul g (T.neg (T.div a (denom ()))))
        | Mod -> no_rule "mod" (tracked a || tracked b) (fun () -> eval op))
    | Compare _ | Arg_reduce _ | Argsort _ | Threefry _ | Read _ -> eval op
    | Convert (Bitcast, _, _) -> eval op
    | Convert (Cast, _, x) ->
        pull1 (eval op) x (fun g -> T.cast (T.dtype x) g)
    | Where (condition, if_true, if_false) ->
        let out = eval op in
        let tt = tracked if_true and tf = tracked if_false in
        if tt || tf then begin
          track out;
          Tape.record tape (fun () ->
              match Tape.find tape out with
              | None -> ()
              | Some g ->
                  let mask = T.cast (T.dtype g) condition in
                  if tt then
                    Tape.accumulate tape if_true
                      (unbroadcast (T.mul g mask) (T.shape if_true));
                  if tf then
                    Tape.accumulate tape if_false
                      (unbroadcast
                         (T.mul g (T.rsub_s (Derivs.one_like mask) mask))
                         (T.shape if_false)))
        end;
        out
    (* Placement is linear: a cotangent moves back to its primal's placement,
       taken now, while the primal is in reach of the handlers above. *)
    | Place (_, x) ->
        let back = T.placement x in
        pull1 (eval op) x (place back)
    (* Movement: linear ops whose pull is the transpose movement. *)
    | Move (x, Reshape _) ->
        (* A cotangent can be a lazy view (a transpose, a broadcast), which a
           reshape may not take as it is. *)
        pull1 (eval op) x (fun g -> T.reshape (T.shape x) (T.contiguous g))
    | Move (x, Permute axes) ->
        let inv = Array.make (Array.length axes) 0 in
        Array.iteri (fun i d -> inv.(d) <- i) axes;
        pull1 (eval op) x (fun g -> T.transpose g ~axes:(Array.to_list inv))
    | Move (x, Expand _) ->
        pull1 (eval op) x (fun g -> unbroadcast g (T.shape x))
    | Move (x, Shrink limits) ->
        let out = eval op in
        let pads =
          Array.mapi
            (fun i (start, _) ->
              let total = (T.shape x).(i) in
              let len = (T.shape out).(i) in
              (start, total - start - len))
            limits
        in
        pull1 out x (fun g -> pad pads (Nx_dtype.zero (T.dtype x)) g)
    | Move (x, Flip dims) -> pull1 (eval op) x (fun g -> flip g dims)
    | Move (x, Window { axis; size = window; step }) ->
        (* The transpose is overlap-add: input position [w*step + j] accumulates
           the cotangent of window [w] at offset [j]. That is what [fold]
           computes — it sums the taps landing on each output position in one
           pass, and stores a zero where no window reached (the dropped tail,
           or the gaps left by [step > window]). Its operand layout is
           [(leading…, window, count)], so permute the windowed axis into the
           trailing spatial slot, fold, and permute the result back. *)
        pull1 (eval op) x (fun g ->
            let in_shape = T.shape x in
            let r = Array.length in_shape in
            (* [g] is [(d0…d_axis-1, count, d_axis+1…d_r-1, window)]. *)
            let to_fold =
              List.init (r + 1) (fun i ->
                  if i < axis then i
                  else if i <= r - 2 then i + 1
                  else if i = r - 1 then r
                  else axis)
            in
            let folded =
              fold
                (T.transpose g ~axes:to_fold)
                ~output_size:[| in_shape.(axis) |]
                ~kernel_size:[| window |] ~stride:[| step |] ~dilation:[| 1 |]
                ~padding:[| (0, 0) |]
            in
            (* [folded] carries [axis] last; move it back into place. *)
            let from_fold =
              List.init r (fun j ->
                  if j < axis then j else if j = axis then r - 1 else j - 1)
            in
            T.transpose folded ~axes:from_fold)
    | Pad (padding, _, x) ->
        let limits =
          Array.mapi (fun i (pre, _) -> (pre, pre + (T.shape x).(i))) padding
        in
        pull1 (eval op) x (fun g -> T.shrink limits g)
    | Cat (axis, xs) ->
        let out = eval op in
        if List.exists tracked xs then begin
          track out;
          Tape.record tape (fun () ->
              match Tape.find tape out with
              | None -> ()
              | Some g ->
                  let off = ref 0 in
                  List.iter
                    (fun x ->
                      let len = (T.shape x).(axis) in
                      let lo = !off in
                      off := !off + len;
                      if tracked x then
                        Tape.accumulate tape x
                          (shrink_axis axis (lo, lo + len) g))
                    xs)
        end;
        out
    | Contiguous x -> pull1 (eval op) x Fun.id
    (* Reductions *)
    | Reduce (Sum, axes, x) ->
        let shape_in = T.shape x in
        pull1 (eval op) x (fun g ->
            let kept =
              T.shape (T.sum x ~axes:(Array.to_list axes) ~keepdims:true)
            in
            T.broadcast_to shape_in (T.reshape kept g))
    | Reduce ((Max | Min), axes, x) ->
        let out = eval op in
        let axes = Array.to_list axes in
        pull1 out x (fun g ->
            T.mul
              (Derivs.reduction_kept ~axes x g)
              (Derivs.extrema' ~axes x out))
    | Reduce (Prod, axes, x) ->
        let out = eval op in
        let axes = Array.to_list axes in
        pull1 out x (fun g ->
            T.mul (Derivs.reduction_kept ~axes x g) (Derivs.prod' ~axes x out))
    (* Sorting: a sort is a gather at the argsort indices. *)
    | Sort { descending; axis; x } ->
        pull1 (eval op) x (fun g ->
            let indices = argsort ~descending ~axis x in
            scatter ~mode:`Add ~unique:false ~axis ~indices ~updates:g
              (T.zeros_like x))
    (* Scans *)
    | Scan (k, axis, x) ->
        let out = eval op in
        let shape_in = T.shape x in
        let axis_norm =
          let rank = Array.length shape_in in
          if axis < 0 then axis + rank else axis
        in
        pull1 out x (fun g ->
            match k with
            | Sum ->
                let flipped = T.flip g ~axes:[ axis_norm ] in
                let scanned = T.cumsum ~axis:axis_norm flipped in
                T.flip scanned ~axes:[ axis_norm ]
            | Prod ->
                (* The transpose of the forward rule: [dx_i = y_(i-1) r_i] with
                   [r_i = g_i + x_(i+1) r_(i+1)], a linear scan from the
                   end. *)
                let dt = T.dtype x and axes = [ axis_norm ] in
                let before =
                  Derivs.shifted ~axis:axis_norm 1 (Nx_dtype.one dt) out
                in
                let after =
                  Derivs.shifted ~axis:axis_norm 1 (Nx_dtype.zero dt)
                    (T.flip x ~axes)
                in
                let r =
                  Derivs.linear_scan ~axis:axis_norm after (T.flip g ~axes)
                in
                T.mul before (T.flip r ~axes)
            | Max | Min ->
                (* The cotangent goes to the element each running extremum
                   takes. *)
                scatter ~mode:`Add ~unique:false ~axis:axis_norm
                  ~indices:(Derivs.running_arg ~axis:axis_norm out)
                  ~updates:g (T.zeros_like x))
    (* Gather / scatter *)
    | Gather (axis, indices, data) ->
        pull1 (eval op) data (fun g ->
            scatter ~mode:`Add ~unique:false ~axis ~indices ~updates:g
              (T.zeros_like data))
    | Scatter { mode; unique; axis; indices; updates; into } ->
        let out = eval op in
        let tt = tracked into and tu = tracked updates in
        if tt || tu then begin
          track out;
          Tape.record tape (fun () ->
              match Tape.find tape out with
              | None -> ()
              | Some g ->
                  if tu then begin
                    let gu = gather ~axis indices g in
                    (* Under [`Set] an update shadowed by a later one at the
                       same position reaches no output. Scattering each
                       update's rank along [axis] the same way leaves the
                       winner's rank at every position. *)
                    let gu =
                      match mode with
                      | `Set when not unique ->
                          let shp = T.shape indices in
                          let along = Array.make (Array.length shp) 1 in
                          along.(axis) <- shp.(axis);
                          let rank =
                            T.broadcast_to shp
                              (T.reshape along
                                 (T.arange T.int64 0 shp.(axis) 1))
                          in
                          let winner =
                            scatter ~mode:`Set ~unique:false ~axis ~indices
                              ~updates:rank
                              (T.zeros T.int64 (T.shape into))
                          in
                          T.where
                            (T.equal (gather ~axis indices winner) rank)
                            gu (T.zeros_like gu)
                      | `Set | `Add -> gu
                    in
                    Tape.accumulate tape updates gu
                  end;
                  if tt then begin
                    (* Under [`Set] the written positions shadow the template;
                       under [`Add] the template passes through everywhere. *)
                    let gt =
                      match mode with
                      | `Add -> g
                      | `Set ->
                          let mask =
                            scatter ~mode:`Set ~unique ~axis ~indices
                              ~updates:(T.zeros_like updates)
                              (T.ones_like into)
                          in
                          T.mul g mask
                    in
                    Tape.accumulate tape into gt
                  end)
        end;
        out
    | Update (x, starts, v) ->
        let out = eval op in
        let tt = tracked x and tv = tracked v in
        if tt || tv then begin
          track out;
          Tape.record tape (fun () ->
              match Tape.find tape out with
              | None -> ()
              | Some g ->
                  (* The window shadows [x]; [v] receives the window of the
                     cotangent, read axis by axis with a gather so a traced
                     [starts] stays traced. *)
                  if tt then
                    Tape.accumulate tape x (update g ~starts (T.zeros_like v));
                  if tv then begin
                    let vshape = T.shape v in
                    let rank = Array.length vshape in
                    let win = ref g in
                    for ax = 0 to rank - 1 do
                      let len = vshape.(ax) in
                      let start = T.reshape [||] (T.slice [ I ax ] starts) in
                      let idx = T.add (T.arange T.int64 0 len 1) start in
                      let shp = Array.copy (T.shape !win) in
                      shp.(ax) <- len;
                      let rs = Array.make rank 1 in
                      rs.(ax) <- len;
                      let idx = T.broadcast_to shp (T.reshape rs idx) in
                      win := gather ~axis:ax idx !win
                    done;
                    Tape.accumulate tape v !win
                  end)
        end;
        out
    (* Windowing: unfold and fold are duals. *)
    | Unfold { kernel_size; stride; dilation; padding; x } ->
        let input_shape = T.shape x in
        let num_spatial = Array.length kernel_size in
        let output_size =
          Array.sub input_shape
            (Array.length input_shape - num_spatial)
            num_spatial
        in
        pull1 (eval op) x (fun g ->
            fold g ~output_size ~kernel_size ~stride ~dilation ~padding)
    | Fold { kernel_size; stride; dilation; padding; x; _ } ->
        pull1 (eval op) x (fun g ->
            unfold g ~kernel_size ~stride ~dilation ~padding)
    | Matmul (a, b) ->
        let out = eval op in
        let ta = tracked a and tb = tracked b in
        if ta || tb then begin
          track out;
          Tape.record tape (fun () ->
              match Tape.find tape out with
              | None -> ()
              | Some g ->
                  let a_shape = T.shape a and b_shape = T.shape b in
                  let g_shape = T.shape g in
                  let a_ndim = Array.length a_shape in
                  let b_ndim = Array.length b_shape in
                  let g_ndim = Array.length g_shape in
                  let transpose_last2 x =
                    let nd = Array.length (T.shape x) in
                    if nd < 2 then x
                    else
                      let axes =
                        List.init nd (fun i ->
                            if i = nd - 2 then -1
                            else if i = nd - 1 then -2
                            else i)
                      in
                      T.transpose ~axes x
                  in
                  if ta then begin
                    let grad_a =
                      if a_ndim = 2 && b_ndim >= 3 then
                        let g_bt = T.matmul g (transpose_last2 b) in
                        let batch_dims = List.init (g_ndim - 2) Fun.id in
                        if batch_dims = [] then g_bt
                        else T.sum g_bt ~axes:batch_dims ~keepdims:false
                      else if a_ndim >= 3 && b_ndim >= 3 then
                        (* [a]'s batch axes of extent one broadcast against
                           [b]'s: their cotangent is the sum. *)
                        unbroadcast (T.matmul g (transpose_last2 b)) a_shape
                      else T.matmul g (T.transpose b)
                    in
                    Tape.accumulate tape a grad_a
                  end;
                  if tb then begin
                    let grad_b =
                      if b_ndim = 2 && a_ndim >= 3 then
                        let at_g = T.matmul (transpose_last2 a) g in
                        let batch_dims = List.init (g_ndim - 2) Fun.id in
                        if batch_dims = [] then at_g
                        else T.sum at_g ~axes:batch_dims ~keepdims:false
                      else if a_ndim = 2 && b_ndim >= 3 then
                        let a_t = T.transpose a in
                        let batch_shape = Array.sub g_shape 0 (g_ndim - 2) in
                        let a_t_shape = T.shape a_t in
                        let target_shape =
                          Array.concat [ batch_shape; a_t_shape ]
                        in
                        let a_t_expanded =
                          T.broadcast_to target_shape
                            (T.reshape
                               (Array.concat [ [| 1 |]; a_t_shape ])
                               a_t)
                        in
                        T.matmul a_t_expanded g
                      else if a_ndim >= 3 && b_ndim >= 3 then
                        unbroadcast (T.matmul (transpose_last2 a) g) b_shape
                      else T.matmul (T.transpose a) g
                    in
                    Tape.accumulate tape b grad_b
                  end)
        end;
        out
    (* FFT: the transform matrix is symmetric, so each transform is its own
       transpose and pulls back through itself. Going through the inverse
       instead would reverse the frequency index. *)
    | Fft { inverse; axes; x } ->
        pull1 (eval op) x (fun g -> fft ~inverse ~axes g)
    (* rfft factors as real-embed, fft over every transformed axis, then a slice
       to the first n/2 + 1 bins of the last one. Each factor pulls back through
       its own transpose: the slice through a zero-pad, the fft through itself,
       and the embed through the real part — the same pair the cast rules use
       between float and complex. No Hermitian mirror appears anywhere: the
       forward never built one, so the transpose never reads one. *)
    | Rfft { axes; x; _ } ->
        pull1 (eval op) x (fun g ->
            let last = axes.(Array.length axes - 1) in
            let n = (T.shape x).(last) in
            let m = (T.shape g).(last) in
            let g =
              if n > m then pad_axis last (0, n - m) Complex.zero g else g
            in
            T.real (T.dtype x) (fft ~inverse:false ~axes g))
    (* irfft factors as Hermitian-extend along the last transformed axis, ifft
       over every transformed axis, then the real part. The transpose embeds the
       real cotangent, runs the same inverse transform, and folds the extension
       back: bin k of the input fed slot k directly and slot n - k through a
       conjugate. The cotangent is real, so its transform is Hermitian along the
       last axis and the conjugated mirror equals the head itself: the fold is a
       plain doubling of bins 1 .. n - m (bin 0, and the Nyquist bin when n is
       even, fed one slot only). A per-bin real scale along the last axis
       commutes with the transform over the leading axes, so one inverse over
       all the axes serves, mirroring the rfft pull's single forward transform.
       The frontend resizes the spectrum to exactly n/2 + 1 bins before the
       operation, so no other adjustment remains. *)
    | Irfft { axes; x; _ } ->
        pull1 (eval op) x (fun g ->
            let last = axes.(Array.length axes - 1) in
            let n = (T.shape g).(last) in
            let m = (n / 2) + 1 in
            let gz = fft ~inverse:true ~axes (T.cast (T.dtype x) g) in
            let head = shrink_axis last (0, m) gz in
            if n - m >= 1 then
              T.add head
                (pad_axis last
                   (1, m - 1 - (n - m))
                   Complex.zero
                   (shrink_axis last (1, n - m + 1) head))
            else head)
    (* Linear algebra *)
    (* The factor reads the Hermitian matrix that A's strict lower triangle and
       the real part of its diagonal name, H = L Lᴴ. The rule works on L̄ =
       conj g, the cotangent under the pairing Re tr(L̄ᴴ dL), where a product
       transposes to its conjugate transpose: with Φ the lower triangle less
       half the diagonal and S = L^-H Φ(Lᴴ L̄) L^-1, H's cotangent is the
       Hermitian part of S, and A's lower triangle collects it twice below the
       diagonal and once on it, in the real part. Under [upper] the factor is
       U = Lᴴ, and L̄ = gᵀ. *)
    | Cholesky { upper; x } ->
        let out = eval op in
        pull1 out x (fun g ->
            let l, lbar =
              if upper then (Derivs.adjoint out, T.matrix_transpose g)
              else (out, T.conjugate g)
            in
            let c = T.matmul (Derivs.adjoint l) lbar in
            let phi =
              let diag_c = T.diagonal c in
              let two = Nx_dtype.of_float (T.dtype diag_c) 2.0 in
              T.sub (T.tril c) (Derivs.diag_matrix (T.div_s diag_c two))
            in
            (* L^-H Φ L^-1, the right-hand inverse as the adjoint of a left
               one. *)
            let left =
              solve_triangular ~upper:false ~transpose:true ~unit_diag:false l
            in
            let s = Derivs.adjoint (left (Derivs.adjoint (left phi))) in
            T.conjugate
              (T.sub
                 (T.tril (T.add s (Derivs.adjoint s)))
                 (Derivs.diag_matrix (Derivs.real_part (T.diagonal s)))))
    | Solve_triangular { upper; transpose; unit_diag; a; b } ->
        let out = eval op in
        let ta = tracked a and tb = tracked b in
        if ta || tb then begin
          track out;
          Tape.record tape (fun () ->
              match Tape.find tape out with
              | None -> ()
              | Some g ->
                  (* The solve is X = op(A)^-1 B, where op(A) is A, or Aᴴ under
                     [transpose]. Its transpose in B is op(A)^-T: the conjugate
                     of the other solve, op(A)^-H, applied to the conjugate
                     cotangent. *)
                  let grad_b =
                    T.conjugate
                      (solve_triangular ~upper ~transpose:(not transpose)
                         ~unit_diag a (T.conjugate g))
                  in
                  if tb then Tape.accumulate tape b grad_b;
                  if ta then begin
                    let out_2d, grad_b_2d =
                      if T.ndim g = T.ndim a - 1 then
                        ( T.unsqueeze ~axes:[ -1 ] out,
                          T.unsqueeze ~axes:[ -1 ] grad_b )
                      else (out, grad_b)
                    in
                    (* dX = -op(A)^-1 op(dA) X, and op(dA) = dAᴴ conjugates
                       dA. *)
                    let grad_a_full =
                      if transpose then
                        T.neg
                          (T.conjugate
                             (T.matmul out_2d (T.matrix_transpose grad_b_2d)))
                      else
                        T.neg (T.matmul grad_b_2d (T.matrix_transpose out_2d))
                    in
                    let grad_a =
                      let tri =
                        if upper then T.triu grad_a_full else T.tril grad_a_full
                      in
                      if unit_diag then
                        T.sub tri (Derivs.diag_matrix (T.diagonal tri))
                      else tri
                    in
                    Tape.accumulate tape a grad_a
                  end)
        end;
        out
    (* A = Q R with Qᴴ Q = I and R upper triangular with a real diagonal. The
       rule works on the cotangents Q̄ = conj gq and R̄ = conj gr under the
       pairing Re tr(X̄ᴴ dX), where a product transposes to its conjugate
       transpose: with M = R R̄ᴴ - Q̄ᴴ Q, and copyltu M its strict lower
       triangle mirrored as a Hermitian matrix around the real part of its
       diagonal, Ā = (Q̄ + Q copyltu M) R^-H. The imaginary part of M's
       diagonal is the phase R's real diagonal fixes, and moves nothing. *)
    | Qr { x; _ } ->
        let ((q, r) as out) = eval op in
        if tracked x then begin
          track q;
          track r;
          Tape.record tape (fun () ->
              let found_q = Tape.find tape q and found_r = Tape.find tape r in
              if Option.is_some found_q || Option.is_some found_r then begin
                let qbar =
                  Option.fold found_q ~none:(T.zeros_like q) ~some:T.conjugate
                in
                let rbar =
                  Option.fold found_r ~none:(T.zeros_like r) ~some:(fun g ->
                      T.conjugate (T.triu g))
                in
                let m =
                  T.sub
                    (T.matmul r (Derivs.adjoint rbar))
                    (T.matmul (Derivs.adjoint qbar) q)
                in
                let lower_strict = T.tril ~k:(-1) m in
                let copyltu =
                  T.add
                    (T.add lower_strict (Derivs.adjoint lower_strict))
                    (Derivs.diag_matrix (Derivs.real_part (T.diagonal m)))
                in
                (* conj Ā = conj (rhs R^-H) = (R^-1 rhsᴴ)ᵀ. *)
                let rhs = T.add qbar (T.matmul q copyltu) in
                Tape.accumulate tape x
                  (T.matrix_transpose
                     (solve_triangular ~upper:true ~transpose:false
                        ~unit_diag:false r (Derivs.adjoint rhs)))
              end)
        end;
        out
    (* P A = L U. The pivots and the permutation are integers, so only the
       packed factors carry a cotangent: with M = tril_-1(Lᵀ L̄) + triu(Ū Uᵀ),
       the cotangent of P A is L^-T M U^-T, and P's rows go back to where they
       came from. *)
    | Lu x ->
        let ((packed, _, perm) as out) = eval op in
        if tracked x then begin
          if T.dim (-1) x <> T.dim (-2) x then
            err_no_rule "lu of a rectangular matrix";
          track packed;
          Tape.record tape (fun () ->
              match Tape.find tape packed with
              | None -> ()
              | Some g ->
                  let l =
                    T.add (T.tril ~k:(-1) packed)
                      (T.eye (T.dtype packed) (T.dim (-1) packed))
                  in
                  let u = T.triu packed in
                  let m =
                    T.add
                      (T.tril ~k:(-1)
                         (T.matmul (T.matrix_transpose l) (T.tril ~k:(-1) g)))
                      (T.triu (T.matmul (T.triu g) (T.matrix_transpose u)))
                  in
                  (* Lᵀ is the unit upper triangle of the packed transpose, and
                     (Y U^-T)ᵀ = U^-1 Yᵀ. *)
                  let y =
                    solve_triangular ~upper:true ~transpose:false
                      ~unit_diag:true (T.matrix_transpose packed) m
                  in
                  let gpa =
                    T.matrix_transpose
                      (solve_triangular ~upper:true ~transpose:false
                         ~unit_diag:false packed (T.matrix_transpose y))
                  in
                  let unperm =
                    T.broadcast_to (T.shape gpa)
                      (T.unsqueeze ~axes:[ -1 ] (T.argsort ~axis:(-1) perm))
                  in
                  Tape.accumulate tape x
                    (T.take_along_axis ~axis:(-2) ~indices:unperm gpa))
        end;
        out
    | Svd { x; _ } -> no_rule "svd" (tracked x) (fun () -> eval op)
    | Eig { vectors; x } ->
        no_rule (if vectors then "eig" else "eigvals") (tracked x) (fun () ->
            eval op)
    | Eigh { vectors; x } ->
        no_rule (if vectors then "eigh" else "eigvalsh") (tracked x) (fun () ->
            eval op)
  in

  let body : type c. c Effect.t -> (unit -> c) option =
   fun eff ->
    match eff with
    | Pause.E_pause -> Some (Pause.hold paused)
    | _ when !paused > 0 -> None
    (* Staged scan. When a stager lies beyond, the scan passes on and the
       tape records its transpose, a scan too (see [staged_scan]). Otherwise
       the eager fold runs under a nested instance of this handler, taping
       every step. *)
    | Scan.E_scan_probe -> Some Scan.probe
    | Scan.E_scan req ->
        Some
          (fun () ->
            let fold () = install tape (fun () -> Scan.eager req) in
            Scan.pass_on ~fold (fun () -> staged_scan tape req))
    | Axis.E_lanes { axis; t_in } ->
        Some
          (fun () ->
            no_rule "Rune.lanes" (tracked t_in) (fun () ->
                Axis.lanes axis t_in))
    (* Custom rules. The forward function runs in the enclosing context: this
       handler replaces its internals with the user's rule, while enclosing
       transformations see the forward computation itself. *)
    | Custom.E_custom_vjp
        (Custom.Vjp_call { params_s; result_s; params; fwd; bwd }) ->
        Some
          (fun () ->
            let any =
              Nx.Ptree.fold params_s
                (fun _ leaf any -> any || tracked leaf)
                params false
            in
            let y, res = own fwd params in
            (* A result that is one of the parameters is aliased, so its
               cotangent is the result's alone. *)
            let y = if any then Structure.aliases result_s y else y in
            if any then begin
              Nx.Ptree.fold result_s (fun _ leaf () -> track leaf) y ();
              Tape.record tape (fun () ->
                  let seeded = ref false in
                  (* [bwd] takes and returns gradients, the conjugates of the
                     tape's cotangents (see [Rune.vjp]). *)
                  let cts =
                    Nx.Ptree.map result_s
                      (fun _ leaf ->
                        match Tape.find tape leaf with
                        | Some ct ->
                            seeded := true;
                            T.conjugate ct
                        | None -> T.zeros_like leaf)
                      y
                  in
                  if !seeded then
                    ignore
                      (Structure.map2 "Rune.custom_vjp" params_s
                         ~this:"the parameters" ~that:"bwd's gradients"
                         (fun _ leaf g ->
                           if tracked leaf then
                             Tape.accumulate tape leaf (T.conjugate g);
                           leaf)
                         params (bwd res cts)))
            end;
            y)
    (* A custom_jvp has no reverse rule. One whose result holds no tensor
       has nothing to differentiate, and its function runs. *)
    | Custom.E_custom_jvp
        (Custom.Jvp_call { params_s; result_s; params; f; _ }) ->
        Some
          (fun () ->
            let y = own f params in
            if
              Structure.holds_tensor result_s y
              && Nx.Ptree.fold params_s
                   (fun _ leaf any -> any || tracked leaf)
                   params false
            then
              invalid_arg
                "Rune: a custom_jvp function is not reverse-differentiable; \
                 define a custom_vjp rule instead"
            else y)
    (* Gradient checkpointing. The call passes on with [f] run under this
       handler over a scratch tape linked to this one, which tells whether the
       result depends on a tracked tensor: an argument, or one [f] captures.
       The scratch tape is then dropped. When the result does depend on one,
       the tape keeps the arguments and, once the result's cotangents exist,
       differentiates a second run of [f] (see [recompute]). *)
    | Remat.E_remat (Remat.Call { params_s; result_s; params; f; _ }) ->
        Some
          (fun () ->
            let depends = ref false in
            let f' params =
              let scratch = Tape.create ~parent:tape () in
              let y = install scratch (fun () -> f params) in
              depends :=
                Nx.Ptree.fold result_s
                  (fun _ leaf d -> d || Tape.tracked scratch leaf)
                  y false;
              y
            in
            let y =
              Remat.run
                (Remat.Call
                   { params_s; result_s; params; f = f'; residuals = true })
            in
            if not !depends then y
            else begin
              (* A result that is one of the parameters is aliased, so its
                 cotangent is the result's alone. *)
              let y = Structure.aliases result_s y in
              Nx.Ptree.fold result_s (fun _ leaf () -> track leaf) y ();
              Tape.record tape (fun () ->
                  if
                    Nx.Ptree.fold result_s
                      (fun _ leaf seeded ->
                        seeded || Option.is_some (Tape.find tape leaf))
                      y false
                  then
                    recompute tape params_s result_s f params
                      (Nx.Ptree.map result_s
                         (fun _ leaf -> Tape.cotangent tape leaf)
                         y));
              y
            end)
    (* The barrier is the identity: an output's cotangent is its value's. *)
    | Remat.E_barrier { values; after } ->
        if not (List.exists (fun (Nx.P v) -> tracked v) values) then None
        else
          Some
            (fun () ->
              let out = Remat.barrier ~after values in
              List.iter2
                (fun (Nx.P v) o ->
                  let o = Nx.unpack (T.dtype v) o in
                  if tracked v && o != v then begin
                    track o;
                    Tape.record tape (fun () ->
                        match Tape.find tape o with
                        | None -> ()
                        | Some g -> Tape.accumulate tape v g)
                  end)
                values out;
              out)
    (* Quantised products. A weight is never differentiated; the cotangent of
       [x] is the transposed product with the same ids, summed over the axes
       along which [x] was broadcast. The tape holds the weight and the ids,
       nothing decoded. *)
    | Nx_quant.Effect.E_quant
        { w = Nx_quant.Mxfp4 { codes; scales } as w; op } ->
        Some
          (fun () ->
            if tracked codes || tracked scales then err_quant ();
            let y = Nx_quant.Effect.perform w op in
            (match op with
            | Apply { ids; x; transpose } when tracked x ->
                track y;
                Tape.record tape (fun () ->
                    match Tape.find tape y with
                    | None -> ()
                    | Some g ->
                        let x_shape = T.shape x in
                        let vector = Array.length x_shape = 1 in
                        let g =
                          if vector then T.unsqueeze ~axes:[ T.ndim g - 1 ] g
                          else g
                        in
                        let op =
                          Nx_quant.Effect.Apply
                            { ids; x = g; transpose = not transpose }
                        in
                        let dx = Nx_quant.Effect.perform w op in
                        let dx =
                          if vector then
                            T.reshape x_shape
                              (unbroadcast dx (Array.append [| 1 |] x_shape))
                          else unbroadcast dx x_shape
                        in
                        Tape.accumulate tape x dx)
            | Apply _ | Dequant _ -> ());
            y)
    | _ -> None
  in
  (* A rerun tape differentiates code the forward pass already ran, so it drops
     the code's additions, which are never taped. While paused it keeps its
     claim on the code another handler would run past it: scans, remats and
     custom calls run untaped, with their additions dropped. *)
  let rule : type c. c Effect.t -> (unit -> c) option =
   fun eff ->
    if not (Tape.rerun tape) then body eff
    else
      match eff with
      | Total.E_add _ -> Some (fun () -> ())
      | Pause.E_pause -> body eff
      | _ when !paused = 0 -> body eff
      | Scan.E_scan_probe -> Some Scan.probe
      | Scan.E_scan req ->
          Some
            (fun () ->
              Scan.pass_on
                ~fold:(fun () -> raise Scan.Not_staged)
                (fun () ->
                  let run c x =
                    Total.dropping (fun () -> req.req_step.run c x)
                  in
                  Effect.perform (Scan.E_scan { req with req_step = { run } })))
      | Remat.E_remat (Remat.Call { params_s; result_s; params; f; residuals })
        ->
          Some
            (fun () ->
              Remat.run
                (Remat.Call
                   { params_s; result_s; params; f = dropped f; residuals }))
      | Custom.E_custom_vjp (Custom.Vjp_call { params; fwd; _ }) ->
          Some (fun () -> fst (dropped fwd params))
      | Custom.E_custom_jvp (Custom.Jvp_call { params; f; _ }) ->
          Some (fun () -> dropped f params)
      | _ -> None
  in
  let effc : type c. c Effect.t -> ((c, a) continuation -> a) option =
   fun eff -> Option.map Answer.deliver (rule eff)
  in
  (* While paused, every operation passes on as it is. *)
  let run op = if !paused > 0 then eval op else run op in
  match_with
    (fun () -> intercept { run; claims = (fun _ -> true) } f)
    ()
    { retc = Fun.id; exnc = raise; effc }

(* A staged scan passes on with its step run under this handler over a scratch
   tape linked to [tape], whose leaves are then the tensors the step captures
   that [tape] tracks, and with the carry entering each step among its outputs.
   The tape records the scan's transpose: a scan the other way over the step's
   pullback, which recomputes each step from the carry that entered it. Its
   carry threads the carry's cotangent through the steps and sums each
   capture's, and its outputs are the cotangents of the tracked rows, stacked
   like them. It is an ordinary scan, which the handlers around the backward
   pass transform or stage like any other. *)
and staged_scan tape (req : Scan.scan_req) : Scan.scan_res =
  let Scan.{ req_carry; req_xs; req_step; req_reverse } = req in
  let captures = ref [] in
  let run c x =
    let scratch = Tape.create ~parent:tape () in
    let c', y = install scratch (fun () -> req_step.run c x) in
    captures := Tape.captures scratch;
    (c', y @ c)
  in
  let res = Effect.perform (Scan.E_scan { req with req_step = { run } }) in
  let nc = List.length req_carry and nx = List.length req_xs in
  let ys, stacks = Scan.split (List.length res.r_ys - nc) res.r_ys in
  let captures = !captures in
  let tracked = List.map (fun (Nx.P x) -> Tape.tracked tape x) req_xs in
  let seed t =
    List.iter2 (fun (Nx.P v) d -> Tape.accumulate t v (Nx.unpack (T.dtype v) d))
  in
  let step carry row =
    let dc, acc = Scan.split nc carry in
    let c, row = Scan.split nc row in
    let x, dy = Scan.split nx row in
    let t = Tape.create ~parent:tape ~rerun:true () in
    List.iter (fun (Nx.P l) -> Tape.track t l) (c @ captures);
    List.iter2 (fun r (Nx.P l) -> if r then Tape.track t l) tracked x;
    let c', y = install t (fun () -> req_step.run c x) in
    if not (List.is_empty (Tape.captures t)) then
      invalid_arg
        "Rune.scan: the body reads a differentiated tensor in the backward \
         pass that it did not read in the forward pass";
    seed t c' dc;
    seed t y dy;
    Tape.backward t;
    let ct (Nx.P l) = Nx.P (Tape.cotangent t l) in
    ( List.map ct c
      @ List.map2
          (fun (Nx.P g) a ->
            Nx.P (T.add (Nx.unpack (T.dtype g) a) (Tape.cotangent t g)))
          captures acc,
      List.concat_map (fun (r, l) -> if r then [ ct l ] else [])
        (List.combine tracked x) )
  in
  let transpose () =
    let ct (Nx.P l) = Nx.P (Tape.cotangent tape l) in
    (* The loop's carry lives where the primals do. *)
    let at (Nx.P l) v = Nx.P (T.place (T.placement l) v) in
    let bwd =
      Scan.run
        {
          req_carry =
            List.map
              (fun (Nx.P l as p) -> at p (Tape.cotangent tape l))
              res.r_carry
            @ List.map (fun (Nx.P g as p) -> at p (T.zeros_like g)) captures;
          req_xs = stacks @ req_xs @ List.map ct ys;
          req_step = { run = step };
          req_reverse = not req_reverse;
        }
    in
    let dc, dgs = Scan.split nc bwd.r_carry in
    let accumulate (Nx.P x) d =
      if Tape.tracked tape x then
        Tape.accumulate tape x (Nx.unpack (T.dtype x) d)
    in
    List.iter2 accumulate req_carry dc;
    List.iter2 accumulate
      (List.filter_map
         (fun (r, x) -> if r then Some x else None)
         (List.combine tracked req_xs))
      bwd.r_ys;
    List.iter2 accumulate captures dgs
  in
  let track = List.iter (fun (Nx.P l) -> Tape.track tape l) in
  track res.r_carry;
  track ys;
  Tape.record tape (fun () ->
      if
        List.exists
          (fun (Nx.P l) -> Option.is_some (Tape.find tape l))
          (res.r_carry @ ys)
      then transpose ());
  { res with r_ys = ys }

(* Accumulate into [tape] the pullback of [cts] through a second run of [f] at
   [params]. The run reads [params] through a barrier after [cts], under a tape
   linked to [tape], so a tensor [f] captures that [tape] tracks becomes a leaf
   of the run and its cotangent goes back to [tape] too. *)
and recompute : type p q.
    Tape.t -> p Nx.Ptree.t -> q Nx.Ptree.t -> (p -> q) -> p -> q -> unit =
 fun tape params_s result_s f params cts ->
  let after = fst (Nx.Ptree.flatten result_s cts) in
  let params' =
    Structure.aliases params_s
      (Nx.Ptree.rebuild params_s ~like:params
         (Remat.barrier ~after (fst (Nx.Ptree.flatten params_s params))))
  in
  let run = Tape.create ~parent:tape ~rerun:true () in
  ignore
    (Structure.map2 "Rune.remat" params_s ~this:"the arguments"
       ~that:"their barriers"
       (fun _ p p' ->
         if Tape.tracked tape p then Tape.track run p';
         p)
       params params');
  let y = install run (fun () -> f params') in
  ignore
    (Structure.map2 "Rune.remat" result_s ~this:"the result"
       ~that:"the cotangents"
       (fun _ yl ct ->
         Tape.accumulate run yl ct;
         yl)
       y cts);
  Tape.backward run;
  ignore
    (Structure.map2 "Rune.remat" params_s ~this:"the arguments"
       ~that:"their barriers"
       (fun _ p p' ->
         if Tape.tracked run p' then
           Tape.accumulate tape p (Tape.cotangent run p');
         p)
       params params');
  List.iter
    (fun (Nx.P c) -> Tape.accumulate tape c (Tape.cotangent run c))
    (Tape.captures run)
