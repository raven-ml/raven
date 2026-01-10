open Nx_rune
open Nx_core
module T = Tensor

(* Type to represent mapping specification for a single axis *)
type axis_spec =
  | Map of int (* Map over this axis index *)
  | NoMap (* Don't map this axis *)

(* Type to represent container mapping specifications *)
type 'a in_axes_spec = Single of axis_spec | Container of 'a

(* Type to represent output axes specification *)
type 'a out_axes_spec = OutSingle of int option | OutContainer of 'a

(* Helper to extract mapped axis from in_axes specification *)
let extract_axis_spec = function
  | Single spec -> spec
  | Container _ -> failwith "vmap: container in_axes not yet supported"

(* Helper to extract output axis from out_axes specification *)
let extract_out_axis_spec = function
  | OutSingle spec -> spec
  | OutContainer _ -> failwith "vmap: container out_axes not yet supported"

(* Utility functions for batch level management *)
let insert_at (arr : 'a array) (pos : int) (x : 'a) : 'a array =
  let n = Array.length arr in
  if pos < 0 || pos > n then
    failwith
      (Printf.sprintf "insert_at: invalid position %d for array of length %d"
         pos n);
  Array.concat [ Array.sub arr 0 pos; [| x |]; Array.sub arr pos (n - pos) ]

(* Like insert_at, but if pos > length, left-pad with pad_value up to pos before
   inserting. *)
let insert_at_pad (arr : int array) ~(pad_value : int) (pos : int) (x : int) :
    int array =
  let n = Array.length arr in
  if pos < 0 then
    failwith
      (Printf.sprintf
         "insert_at_pad: invalid position %d for array of length %d" pos n)
  else if pos <= n then insert_at arr pos x
  else
    let pad_len = pos - n in
    let pad = Array.make pad_len pad_value in
    Array.concat [ arr; pad; [| x |] ]

let remove_at (arr : 'a array) (pos : int) : 'a array =
  let n = Array.length arr in
  if pos < 0 || pos >= n then
    failwith
      (Printf.sprintf "remove_at: invalid position %d for array of length %d"
         pos n);
  Array.concat [ Array.sub arr 0 pos; Array.sub arr (pos + 1) (n - pos - 1) ]

(* Map logical (unbatched) axes -> physical axes given a batch dimension *)
let phys_axis ~bdim (i : int) = if i >= bdim then i + 1 else i

(* Helper to create axis permutation for moving axis from -> to *)
let move_axis_perm ~from ~to_ ndim =
  let perm = Array.init ndim (fun i -> i) in
  if from = to_ then perm
  else if from < to_ then (
    (* Shift elements between from and to_ left *)
    for i = from to to_ - 1 do
      perm.(i) <- i + 1
    done;
    perm.(to_) <- from;
    perm)
  else (
    (* from > to_: shift elements between to_ and from right *)
    for i = to_ + 1 to from do
      perm.(i) <- i - 1
    done;
    perm.(to_) <- from;
    perm)

(* Helper to move an axis to the front or back of a tensor *)
let move_axis (tensor : ('a, 'b) t) ~from_axis ~to_axis : ('a, 'b) t =
  let shape = T.shape tensor in
  let ndim = Array.length shape in
  let from_axis = if from_axis < 0 then ndim + from_axis else from_axis in
  let to_axis = if to_axis < 0 then ndim + to_axis + 1 else to_axis in

  if from_axis = to_axis then tensor
  else
    let axes = Array.init ndim (fun i -> i) in
    (* Remove from_axis from its current position *)
    let temp_axes =
      Array.concat
        [
          Array.sub axes 0 from_axis;
          Array.sub axes (from_axis + 1) (ndim - from_axis - 1);
        ]
    in
    (* Insert at to_axis position *)
    let new_axes =
      if to_axis = 0 then Array.concat [ [| from_axis |]; temp_axes ]
      else if to_axis >= ndim then Array.concat [ temp_axes; [| from_axis |] ]
      else
        Array.concat
          [
            Array.sub temp_axes 0 to_axis;
            [| from_axis |];
            Array.sub temp_axes to_axis (Array.length temp_axes - to_axis);
          ]
    in
    T.transpose tensor ~axes:(Array.to_list new_axes)

(* Helper to add a batch dimension to a tensor at a specific position *)
let _add_batch_dim_at (tensor : ('a, 'b) t) ~batch_pos ~size : ('a, 'b) t =
  let shape = T.shape tensor in
  let new_shape = insert_at shape batch_pos size in
  let expanded = T.expand_dims [ batch_pos ] tensor in
  T.broadcast_to new_shape expanded

(* Custom hashtable module that uses physical equality to distinguish tensors *)
module PhysicalTbl = struct
  type level = int
  type t = (Obj.t * (level, int option) Hashtbl.t) list ref

  let create () : t = ref []

  let ensure_map (tbl : t) key =
    let k = Obj.repr key in
    match List.assoc_opt k !tbl with
    | Some m -> m
    | None ->
        let m = Hashtbl.create 4 in
        tbl := (k, m) :: !tbl;
        m

  let set_bdim (tbl : t) key ~level ~bdim =
    let m = ensure_map tbl key in
    Hashtbl.replace m level bdim

  let get_bdim (tbl : t) key ~level : int option =
    match List.assoc_opt (Obj.repr key) !tbl with
    | None -> None
    | Some m -> ( try Hashtbl.find m level with Not_found -> None)

  let has_level (tbl : t) key level = Option.is_some (get_bdim tbl key ~level)

  (* Get all batch dimensions for a tensor across all levels *)
  let get_all_bdims (tbl : t) key : (level * int) list =
    match List.assoc_opt (Obj.repr key) !tbl with
    | None -> []
    | Some m ->
        Hashtbl.fold
          (fun level bdim_opt acc ->
            match bdim_opt with
            | None -> acc
            | Some bdim -> (level, bdim) :: acc)
          m []
        |> List.sort (fun (l1, _) (l2, _) -> compare l1 l2)

  (* Copy all batch dimensions from one table to another *)
  let copy_to (src : t) (dst : t) =
    List.iter
      (fun (key_repr, src_map) ->
        let dst_map =
          match List.assoc_opt key_repr !dst with
          | Some m -> m
          | None ->
              let m = Hashtbl.create 4 in
              dst := (key_repr, m) :: !dst;
              m
        in
        Hashtbl.iter
          (fun level bdim -> Hashtbl.replace dst_map level bdim)
          src_map)
      !src

  (* Clear all batch dimensions at a specific level *)
  let clear_level (tbl : t) level =
    List.iter (fun (_, map) -> Hashtbl.remove map level) !tbl
end

(* ───── Vmap environment (dynamic scope) ───── *)
type env = {
  level : int;
  shared : PhysicalTbl.t;
  batch_sizes : (int, int) Hashtbl.t;
}

let current_env : env option ref = ref None
let current_batch_level : int ref = ref 0

let with_env (env : env) (f : unit -> 'a) : 'a =
  let prev_env = !current_env in
  current_env := Some env;
  current_batch_level := env.level;
  Fun.protect
    ~finally:(fun () ->
      current_env := prev_env;
      current_batch_level := match prev_env with Some e -> e.level | None -> 0)
    f

let get_env () : env =
  match !current_env with
  | Some e -> e
  | None ->
      {
        level = -1;
        shared = PhysicalTbl.create ();
        batch_sizes = Hashtbl.create 8;
      }

let make_vmap_handler ~env ~axis_size ~batched_tensors out_axis axis_name =
  let open Effect.Deep in
  (* Store axis_name for potential use in collective operations *)
  let _ = axis_name in
  (* Currently unused, but available for future collective ops *)

  (* Suspension flag: let shape-manipulation ops bubble to outer handlers (AD)
     while we manage batch metadata. *)
  let suspended = ref false in
  let with_suspended f =
    suspended := true;
    Fun.protect ~finally:(fun () -> suspended := false) f
  in

  (* Get the batch dimension for a tensor at this level *)
  let get_bdim tensor =
    (* Check the shared batch state *)
    PhysicalTbl.get_bdim env.shared tensor ~level:env.level
  in

  (* Set the batch dimension for a tensor at this level *)
  let set_bdim tensor bdim =
    (* Update both local and shared state *)
    PhysicalTbl.set_bdim batched_tensors tensor ~level:env.level ~bdim;
    PhysicalTbl.set_bdim env.shared tensor ~level:env.level ~bdim
  in

  (* Check if a tensor is batched at THIS level *)
  let _is_batched tensor = Option.is_some (get_bdim tensor) in

  (* Helper to get physical shape (backend view) of a tensor *)
  let phys_shape_of : type a b. (a, b) t -> int array =
   fun t ->
    let view = Nx_rune.view t in
    match Symbolic_shape.eval (View.shape view) with
    | Some arr -> arr
    | None -> failwith "vmap: cannot evaluate physical shape"
  in

  (* Derive present batch prefix length by matching leading physical dims
     against known batch sizes for levels 0..env.level. Robust even if bdim
     metadata is partially missing. Assumes tensors are canonicalized. *)
  let prefix_len_by_batch_sizes t =
    let s = phys_shape_of t in
    let n = Array.length s in
    let pos = ref 0 in
    for lv = 0 to env.level do
      let sz = try Hashtbl.find env.batch_sizes lv with Not_found -> 1 in
      if !pos < n && s.(!pos) = sz then incr pos
    done;
    !pos
  in

  let phys_shrink : type a b. (a, b) t -> (int * int) array -> (a, b) t =
   fun t limits -> Nx_rune.op_shrink t limits
  in

  (* Effectful shape ops under suspension so AD can track duals *)
  let phys_reshape : type a b. (a, b) t -> int array -> (a, b) t =
   fun t new_shape ->
    with_suspended (fun () -> op_reshape t (Symbolic_shape.of_ints new_shape))
  in

  let phys_expand : type a b. (a, b) t -> int array -> (a, b) t =
   fun t new_shape ->
    with_suspended (fun () -> op_expand t (Symbolic_shape.of_ints new_shape))
  in

  let phys_permute : type a b. (a, b) t -> int array -> (a, b) t =
   fun t axes -> with_suspended (fun () -> op_permute t axes)
  in

  (* Debug helpers *)
  let pp_shape (a : int array) : string =
    let items =
      a |> Array.to_list |> List.map string_of_int |> String.concat ";"
    in
    "[" ^ items ^ "]"
  in
  let dprintf fmt = Printf.eprintf ("[vmap:l%d] " ^^ fmt ^^ "\n%!") env.level in

  (* Propagate per-level bdim positions through shape transforms *)
  let copy_bdims_insert ~src ~dst ~insert_pos =
    PhysicalTbl.get_all_bdims env.shared src
    |> List.iter (fun (lv, pos) ->
           let new_pos = if pos >= insert_pos then pos + 1 else pos in
           PhysicalTbl.set_bdim env.shared dst ~level:lv ~bdim:(Some new_pos))
  in

  let copy_bdims_same ~src ~dst =
    PhysicalTbl.get_all_bdims env.shared src
    |> List.iter (fun (lv, pos) ->
           PhysicalTbl.set_bdim env.shared dst ~level:lv ~bdim:(Some pos))
  in

  (* Broadcast a canonicalized tensor (batch dims at front) to a target physical
     shape anchored after the batch prefix. *)
  let broadcast_to_canonical : type a b. (a, b) t -> int array -> (a, b) t =
   fun t target_phys ->
    let s = phys_shape_of t in
    dprintf "btc: s=%s target=%s" (pp_shape s) (pp_shape target_phys);
    (* Derive batch prefix length by matching sizes to known batch sizes *)
    let nbd = prefix_len_by_batch_sizes t in
    let s_len = Array.length s in
    let t_len = Array.length target_phys in
    if nbd > t_len then failwith "vmap: target rank smaller than batch prefix";
    (* Insert singleton logical dims after batch prefix to match target logical
       rank *)
    let s_logical = s_len - nbd in
    let t_logical = t_len - nbd in
    let t' =
      if s_logical < t_logical then (
        let insert_count = t_logical - s_logical in
        let inserted = Array.make (s_len + insert_count) 0 in
        Array.blit s 0 inserted 0 nbd;
        for i = 0 to insert_count - 1 do
          inserted.(nbd + i) <- 1
        done;
        Array.blit s nbd inserted (nbd + insert_count) (s_len - nbd);
        dprintf "btc: insert %d ones after nbd=%d -> %s" insert_count nbd
          (pp_shape inserted);
        let t1 = phys_reshape t inserted in
        copy_bdims_insert ~src:t ~dst:t1 ~insert_pos:nbd;
        t1)
      else t
    in
    (* Now expand any size-1 logical dims to match target logical dims; ensure
       batch prefix matches target prefix *)
    let s2 = phys_shape_of t' in
    (* Validate/normalize batch prefix: expand singletons in prefix if needed *)
    let s2' = Array.copy s2 in
    for i = 0 to nbd - 1 do
      let cur = if i < Array.length s2 then s2.(i) else 1 in
      let tgt = target_phys.(i) in
      if cur = tgt || cur = 1 then s2'.(i) <- tgt else s2'.(i) <- cur
    done;
    (* For logical dims, ensure either equal or 1; set to target *)
    for i = nbd to t_len - 1 do
      let cur = if i < Array.length s2 then s2.(i) else 1 in
      let tgt = target_phys.(i) in
      if cur = tgt || cur = 1 then s2'.(i) <- tgt
      else if tgt = 1 then () (* fine, keep cur *)
      else failwith "vmap: incompatible logical broadcast"
    done;
    if Array.length s2' <> Array.length s2 || Array.exists2 ( <> ) s2' s2 then (
      dprintf "btc: expand from %s to %s" (pp_shape s2) (pp_shape s2');
      let t2 = phys_expand t' s2' in
      copy_bdims_same ~src:t' ~dst:t2;
      t2)
    else t'
  in
  let copy_bdims_permute ~src ~dst ~perm =
    let n = Array.length perm in
    let inv = Array.make n 0 in
    for i = 0 to n - 1 do
      inv.(perm.(i)) <- i
    done;
    PhysicalTbl.get_all_bdims env.shared src
    |> List.iter (fun (lv, pos) ->
           let new_pos = if pos >= 0 && pos < n then inv.(pos) else pos in
           PhysicalTbl.set_bdim env.shared dst ~level:lv ~bdim:(Some new_pos))
  in

  (* Removed helpers no longer needed after robust prefix handling in
     reshape/expand *)
  let align_to p tensor =
    match get_bdim tensor with
    | None ->
        (* If the unmarked tensor already has the batch at position [p] with the
           correct size, just record it. Otherwise, insert a singleton at [p]
           (padding with 1s if needed) and expand to [axis_size]. *)
        let phys = phys_shape_of tensor in
        let n = Array.length phys in
        if p < n && phys.(p) = axis_size then (
          PhysicalTbl.set_bdim env.shared tensor ~level:env.level ~bdim:(Some p);
          tensor)
        else
          let inserted =
            if p <= n then insert_at phys p 1
            else insert_at_pad phys ~pad_value:1 p 1
          in
          let t1 = phys_reshape tensor inserted in
          copy_bdims_insert ~src:tensor ~dst:t1 ~insert_pos:p;
          let target = Array.copy inserted in
          target.(p) <- axis_size;
          let t2 = phys_expand t1 target in
          copy_bdims_same ~src:t1 ~dst:t2;
          PhysicalTbl.set_bdim env.shared t2 ~level:env.level ~bdim:(Some p);
          t2
    | Some q when q = p -> tensor
    | Some q ->
        (* Move batch dimension from q to p *)
        let ndim = Array.length (phys_shape_of tensor) in
        let perm = move_axis_perm ~from:q ~to_:p ndim in
        let t' = phys_permute tensor perm in
        PhysicalTbl.set_bdim env.shared t' ~level:env.level ~bdim:(Some p);
        t'
  in

  (* Ensure [t] has all outer batch dims (levels < env.level) present in [like].
     Missing dims are inserted and broadcast to match [like]'s physical
     shape. *)
  let add_missing_outer_bdims ~like t =
    let like_bdims =
      PhysicalTbl.get_all_bdims env.shared like
      |> List.filter (fun (lv, _) -> lv < env.level)
    in
    if like_bdims = [] then t
    else
      let t_missing =
        like_bdims
        |> List.filter (fun (lv, _) ->
               not (PhysicalTbl.has_level env.shared t lv))
        |> List.sort (fun (_, a) (_, b) -> compare b a)
      in
      if t_missing = [] then t
      else
        let t_ref = ref t in
        List.iter
          (fun (lv, _pos) ->
            (* Insert a singleton dim at the front physically by reshaping to
               [1; ... old_shape] *)
            let phys = phys_shape_of !t_ref in
            let inserted = Array.append [| 1 |] phys in
            let t1 = phys_reshape !t_ref inserted in
            (* Broadcast that new leading dim to the batch size for level
               [lv] *)
            let batch_sz =
              try Hashtbl.find env.batch_sizes lv with Not_found -> 1
            in
            let target = Array.copy inserted in
            target.(0) <- batch_sz;
            let t2 = phys_expand t1 target in
            (* Record that [t2] is now batched at level [lv] at [pos]. Preserve
               current-level bdim if it existed on input. *)
            PhysicalTbl.set_bdim env.shared t2 ~level:lv ~bdim:(Some 0);
            (match get_bdim t with
            | Some cp ->
                PhysicalTbl.set_bdim env.shared t2 ~level:env.level
                  ~bdim:(Some cp)
            | None -> ());
            t_ref := t2)
          t_missing;
        !t_ref
  in

  let unify_outer_bdims a b =
    let a' = add_missing_outer_bdims ~like:b a in
    let b' = add_missing_outer_bdims ~like:a b in
    (a', b')
  in

  (* Note: broadcasting to physical shapes is not needed when canonicalizing
     batch dims and delegating logical broadcasting to the frontend. *)

  (* Move all batch dims 0..env.level to the front in level order for [t]. *)
  let canonicalize_batch_positions t =
    (* Ensure all OUTER levels 0..env.level-1 are present; don't insert current
       level *)
    let t =
      let t_ref = ref t in
      for lv = env.level - 1 downto 0 do
        if lv >= 0 && not (PhysicalTbl.has_level env.shared !t_ref lv) then (
          let phys = phys_shape_of !t_ref in
          let inserted = Array.append [| 1 |] phys in
          let t1 = phys_reshape !t_ref inserted in
          copy_bdims_insert ~src:!t_ref ~dst:t1 ~insert_pos:0;
          let batch_sz =
            try Hashtbl.find env.batch_sizes lv with Not_found -> 1
          in
          let target = Array.copy inserted in
          target.(0) <- batch_sz;
          let t2 = phys_expand t1 target in
          copy_bdims_same ~src:t1 ~dst:t2;
          PhysicalTbl.set_bdim env.shared t2 ~level:lv ~bdim:(Some 0);
          (match PhysicalTbl.get_bdim env.shared !t_ref ~level:env.level with
          | Some cp ->
              PhysicalTbl.set_bdim env.shared t2 ~level:env.level
                ~bdim:(Some cp)
          | None -> ());
          t_ref := t2)
      done;
      !t_ref
    in
    (* Build permutation to move PRESENT batch dims to the front in level
       order *)
    let phys = phys_shape_of t in
    let r = Array.length phys in
    let present_levels =
      let acc = ref [] in
      for lv = 0 to env.level do
        match PhysicalTbl.get_bdim env.shared t ~level:lv with
        | Some p -> acc := !acc @ [ (lv, p) ]
        | None -> ()
      done;
      !acc
    in
    let batch_positions = List.map snd present_levels in
    let is_batch = Array.make r false in
    List.iter
      (fun p -> if p >= 0 && p < r then is_batch.(p) <- true)
      batch_positions;
    let non_batch_positions =
      let acc = ref [] in
      for i = 0 to r - 1 do
        if not is_batch.(i) then acc := !acc @ [ i ]
      done;
      !acc
    in
    let axes = Array.of_list (batch_positions @ non_batch_positions) in
    let t' = phys_permute t axes in
    (* Update bdim mapping: assign present levels to front in order *)
    List.iteri
      (fun i (lv, _pos) ->
        PhysicalTbl.set_bdim env.shared t' ~level:lv ~bdim:(Some i))
      present_levels;
    t'
  in

  {
    retc =
      (fun result ->
        (* Handle output axis specification *)
        match out_axis with
        | None -> (
            (* JAX semantics: out_axes=None means the output is not batched.
               Take the first element along THIS level's batch axis *)
            match get_bdim result with
            | None -> result
            | Some p ->
                dprintf "retc(None): shrink along p=%d shape=%s" p
                  (pp_shape (phys_shape_of result));
                let phys = phys_shape_of result in
                let shrink_spec =
                  Array.mapi (fun i d -> if i = p then (0, 1) else (0, d)) phys
                in
                let r' = phys_shrink result shrink_spec in
                (* Remove current level mapping and shift others after p *)
                PhysicalTbl.set_bdim env.shared r' ~level:env.level ~bdim:None;
                PhysicalTbl.get_all_bdims env.shared result
                |> List.iter (fun (lv, pos) ->
                       if lv <> env.level then
                         let new_pos = if pos > p then pos - 1 else pos in
                         PhysicalTbl.set_bdim env.shared r' ~level:lv
                           ~bdim:(Some new_pos));
                r')
        | Some out_pos -> (
            (* Move batch dimension to specified position *)
            match get_bdim result with
            | None -> result
            | Some p when p = out_pos -> result
            | Some p ->
                dprintf "retc(Some %d): move from p=%d shape=%s" out_pos p
                  (pp_shape (phys_shape_of result));
                let ndim = Array.length (phys_shape_of result) in
                let perm = move_axis_perm ~from:p ~to_:out_pos ndim in
                let r' = phys_permute result perm in
                copy_bdims_permute ~src:result ~dst:r' ~perm;
                r'));
    exnc = raise;
    effc =
      (fun (type c) (eff : c Effect.t) ->
        if !suspended then None
        else
          match eff with
          (* Collective: psum over current batch level *)
          | E_psum { t_in } ->
              Some
                (fun (k : (c, _) continuation) ->
                  match get_bdim t_in with
                  | None ->
                      let result = op_copy t_in in
                      continue k result
                  | Some p ->
                      (* Compute output shape by removing axis p *)
                      let in_shape = phys_shape_of t_in in
                      let out_shape =
                        in_shape |> Array.to_list
                        |> List.filteri (fun i _ -> i <> p)
                        |> Array.of_list
                      in
                      (* Allocate output tensor *)
                      let dt = dtype t_in in
                      let result = T.empty dt out_shape in
                      op_reduce_sum ~out:result ~axes:[| p |] ~keepdims:false
                        t_in;
                      (* Update bdim mappings: current level removed; others
                         after p shift left *)
                      PhysicalTbl.set_bdim env.shared result ~level:env.level
                        ~bdim:None;
                      PhysicalTbl.get_all_bdims env.shared t_in
                      |> List.iter (fun (lv, pos) ->
                             if lv <> env.level then
                               let new_pos = if pos > p then pos - 1 else pos in
                               PhysicalTbl.set_bdim env.shared result ~level:lv
                                 ~bdim:(Some new_pos));
                      continue k result)
          (* CRITICAL: Intercept view to return unbatched view *)
          | E_view tensor ->
              Some
                (fun (k : (c, _) continuation) ->
                  (* Get the actual view from the backend *)
                  let actual_view = Nx_rune.view tensor in

                  (* Collect ALL batch dims from outermost (0) to current
                     level *)
                  let batch_dims_to_remove =
                    let acc = ref [] in
                    for lv = 0 to env.level do
                      match
                        PhysicalTbl.get_bdim env.shared tensor ~level:lv
                      with
                      | Some bdim -> acc := (lv, bdim) :: !acc
                      | None -> ()
                    done;
                    (* Sort by physical position desc so removals are stable *)
                    List.sort (fun (_, a) (_, b) -> compare b a) !acc
                  in

                  if batch_dims_to_remove = [] then continue k actual_view
                  else
                    let shape = View.shape actual_view in
                    (* Remove batch dims from the symbolic shape directly *)
                    let unbatched_shape =
                      let arr = ref shape in
                      List.iter
                        (fun (_, pos) ->
                          if pos >= 0 && pos < Array.length !arr then
                            arr := remove_at !arr pos)
                        batch_dims_to_remove;
                      !arr
                    in
                    (* Preserve strides and offset if available *)
                    let unbatched_view =
                      match View.strides_opt actual_view with
                      | None -> View.create unbatched_shape
                      | Some strides -> (
                          let unbatched_strides =
                            let s = ref strides in
                            List.iter
                              (fun (_, pos) ->
                                if pos >= 0 && pos < Array.length !s then
                                  s := remove_at !s pos)
                              batch_dims_to_remove;
                            !s
                          in
                          match
                            Symbolic_shape.eval_dim
                              (View.offset_dim actual_view)
                          with
                          | Some offset ->
                              View.create unbatched_shape
                                ~strides:unbatched_strides ~offset
                          | None -> View.create unbatched_shape)
                    in
                    continue k unbatched_view)
          (* Creation operations - create unbatched tensors *)
          | E_const_scalar { context; value; dtype } ->
              Some
                (fun k ->
                  let result = op_const_scalar context value dtype in
                  (* Register as unbatched at ALL levels from 0 to current *)
                  for lv = 0 to env.level do
                    PhysicalTbl.set_bdim env.shared result ~level:lv ~bdim:None
                  done;
                  (* Also set in local table *)
                  PhysicalTbl.set_bdim batched_tensors result ~level:env.level
                    ~bdim:None;
                  continue k result)
          | E_from_host { context; array } ->
              Some
                (fun k ->
                  let result = from_host context array in
                  (* Register as unbatched at ALL levels from 0 to current *)
                  for lv = 0 to env.level do
                    PhysicalTbl.set_bdim env.shared result ~level:lv ~bdim:None
                  done;
                  (* Also set in local table *)
                  PhysicalTbl.set_bdim batched_tensors result ~level:env.level
                    ~bdim:None;
                  continue k result)
          (* Binary operations - handle broadcasting *)
          | E_add { out; a; b } ->
              Some
                (fun k ->
                  let a =
                    a
                    |> add_missing_outer_bdims ~like:b
                    |> canonicalize_batch_positions
                  in
                  let b =
                    b
                    |> add_missing_outer_bdims ~like:a
                    |> canonicalize_batch_positions
                  in
                  let ba = get_bdim a and bb = get_bdim b in
                  (* Determine target position: use leftmost batch position if
                     any *)
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  (* Align both operands to position p, then restore canonical
                     batch order *)
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  let a' = canonicalize_batch_positions a' in
                  let b' = canonicalize_batch_positions b' in
                  let sa = phys_shape_of a' and sb = phys_shape_of b' in
                  let a_prefix_len =
                    PhysicalTbl.get_all_bdims env.shared a'
                    |> List.filter (fun (lv, _) -> lv <= env.level)
                    |> List.length
                  in
                  let b_prefix_len =
                    PhysicalTbl.get_all_bdims env.shared b'
                    |> List.filter (fun (lv, _) -> lv <= env.level)
                    |> List.length
                  in
                  let nbd = max a_prefix_len b_prefix_len in
                  let a_log =
                    Array.sub sa a_prefix_len (Array.length sa - a_prefix_len)
                  in
                  let b_log =
                    Array.sub sb b_prefix_len (Array.length sb - b_prefix_len)
                  in
                  let target_log = Shape.broadcast a_log b_log in
                  let target_pref =
                    Array.init nbd (fun lv ->
                        try Hashtbl.find env.batch_sizes lv
                        with Not_found -> 1)
                  in
                  let target_phys = Array.append target_pref target_log in
                  let a'' = broadcast_to_canonical a' target_phys in
                  let b'' = broadcast_to_canonical b' target_phys in
                  op_add ~out a'' b'';
                  (* Set result bdim based on whether any input was batched *)
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          | E_mul { out; a; b } ->
              Some
                (fun k ->
                  let a =
                    a
                    |> add_missing_outer_bdims ~like:b
                    |> canonicalize_batch_positions
                  in
                  let b =
                    b
                    |> add_missing_outer_bdims ~like:a
                    |> canonicalize_batch_positions
                  in
                  let ba = get_bdim a and bb = get_bdim b in
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  let a' = canonicalize_batch_positions a' in
                  let b' = canonicalize_batch_positions b' in
                  let sa = phys_shape_of a' and sb = phys_shape_of b' in
                  let a_prefix_len =
                    PhysicalTbl.get_all_bdims env.shared a'
                    |> List.filter (fun (lv, _) -> lv <= env.level)
                    |> List.length
                  in
                  let b_prefix_len =
                    PhysicalTbl.get_all_bdims env.shared b'
                    |> List.filter (fun (lv, _) -> lv <= env.level)
                    |> List.length
                  in
                  let nbd = max a_prefix_len b_prefix_len in
                  let a_log =
                    Array.sub sa a_prefix_len (Array.length sa - a_prefix_len)
                  in
                  let b_log =
                    Array.sub sb b_prefix_len (Array.length sb - b_prefix_len)
                  in
                  let target_log = Shape.broadcast a_log b_log in
                  let target_pref =
                    Array.init nbd (fun lv ->
                        try Hashtbl.find env.batch_sizes lv
                        with Not_found -> 1)
                  in
                  let target_phys = Array.append target_pref target_log in
                  let a'' = broadcast_to_canonical a' target_phys in
                  let b'' = broadcast_to_canonical b' target_phys in
                  op_mul ~out a'' b'';
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          | E_fdiv { out; a; b } ->
              Some
                (fun k ->
                  let a, b = unify_outer_bdims a b in
                  let ba = get_bdim a and bb = get_bdim b in
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  op_fdiv ~out a' b';
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          | E_idiv { out; a; b } ->
              Some
                (fun k ->
                  let a, b = unify_outer_bdims a b in
                  let ba = get_bdim a and bb = get_bdim b in
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  op_idiv ~out a' b';
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          | E_max { out; a; b } ->
              Some
                (fun k ->
                  let a, b = unify_outer_bdims a b in
                  let ba = get_bdim a and bb = get_bdim b in
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  op_max ~out a' b';
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          | E_mod { out; a; b } ->
              Some
                (fun k ->
                  let a, b = unify_outer_bdims a b in
                  let ba = get_bdim a and bb = get_bdim b in
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  op_mod ~out a' b';
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          | E_pow { out; a; b } ->
              Some
                (fun k ->
                  let a, b = unify_outer_bdims a b in
                  let ba = get_bdim a and bb = get_bdim b in
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  op_pow ~out a' b';
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          | E_xor { out; a; b } ->
              Some
                (fun k ->
                  let a, b = unify_outer_bdims a b in
                  let ba = get_bdim a and bb = get_bdim b in
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  op_xor ~out a' b';
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          | E_or { out; a; b } ->
              Some
                (fun k ->
                  let a, b = unify_outer_bdims a b in
                  let ba = get_bdim a and bb = get_bdim b in
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  op_or ~out a' b';
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          | E_and { out; a; b } ->
              Some
                (fun k ->
                  let a, b = unify_outer_bdims a b in
                  let ba = get_bdim a and bb = get_bdim b in
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  op_and ~out a' b';
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          (* Comparison operations *)
          | E_cmplt { out; a; b } ->
              Some
                (fun k ->
                  let ba = get_bdim a and bb = get_bdim b in
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  op_cmplt ~out a' b';
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          | E_cmpne { out; a; b } ->
              Some
                (fun k ->
                  let ba = get_bdim a and bb = get_bdim b in
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  op_cmpne ~out a' b';
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          | E_cmpeq { out; a; b } ->
              Some
                (fun k ->
                  let ba = get_bdim a and bb = get_bdim b in
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  op_cmpeq ~out a' b';
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          | E_cmple { out; a; b } ->
              Some
                (fun k ->
                  let ba = get_bdim a and bb = get_bdim b in
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  op_cmple ~out a' b';
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          (* Unary operations - preserve batch status *)
          | E_neg { out; t_in } ->
              Some
                (fun k ->
                  op_neg ~out t_in;
                  set_bdim out (get_bdim t_in);
                  continue k ())
          | E_sin { out; t_in } ->
              Some
                (fun k ->
                  op_sin ~out t_in;
                  set_bdim out (get_bdim t_in);
                  continue k ())
          | E_sqrt { out; t_in } ->
              Some
                (fun k ->
                  op_sqrt ~out t_in;
                  set_bdim out (get_bdim t_in);
                  continue k ())
          | E_recip { out; t_in } ->
              Some
                (fun k ->
                  op_recip ~out t_in;
                  set_bdim out (get_bdim t_in);
                  continue k ())
          (* Reduction operations with correct axes adjustment *)
          | E_reduce_sum { out; t_in; axes; keepdims } ->
              Some
                (fun k ->
                  match get_bdim t_in with
                  | None ->
                      op_reduce_sum ~out ~axes ~keepdims t_in;
                      set_bdim out None;
                      continue k ()
                  | Some p ->
                      let adjusted_axes = Array.map (phys_axis ~bdim:p) axes in
                      op_reduce_sum ~out ~axes:adjusted_axes ~keepdims t_in;
                      (* Update bdim based on axes removed *)
                      let new_p =
                        if keepdims then Some p
                        else
                          let num_removed_before_p =
                            Array.fold_left
                              (fun acc a -> if a < p then acc + 1 else acc)
                              0 adjusted_axes
                          in
                          Some (p - num_removed_before_p)
                      in
                      set_bdim out new_p;
                      continue k ())
          | E_reduce_max { out; t_in; axes; keepdims } ->
              Some
                (fun k ->
                  match get_bdim t_in with
                  | None ->
                      op_reduce_max ~out ~axes ~keepdims t_in;
                      set_bdim out None;
                      continue k ()
                  | Some p ->
                      let adjusted_axes = Array.map (phys_axis ~bdim:p) axes in
                      op_reduce_max ~out ~axes:adjusted_axes ~keepdims t_in;
                      let new_p =
                        if keepdims then Some p
                        else
                          let num_removed_before_p =
                            Array.fold_left
                              (fun acc a -> if a < p then acc + 1 else acc)
                              0 adjusted_axes
                          in
                          Some (p - num_removed_before_p)
                      in
                      set_bdim out new_p;
                      continue k ())
          | E_reduce_prod { out; t_in; axes; keepdims } ->
              Some
                (fun k ->
                  match get_bdim t_in with
                  | None ->
                      op_reduce_prod ~out ~axes ~keepdims t_in;
                      set_bdim out None;
                      continue k ()
                  | Some p ->
                      let adjusted_axes = Array.map (phys_axis ~bdim:p) axes in
                      op_reduce_prod ~out ~axes:adjusted_axes ~keepdims t_in;
                      let new_p =
                        if keepdims then Some p
                        else
                          let num_removed_before_p =
                            Array.fold_left
                              (fun acc a -> if a < p then acc + 1 else acc)
                              0 adjusted_axes
                          in
                          Some (p - num_removed_before_p)
                      in
                      set_bdim out new_p;
                      continue k ())
          (* Shape operations - adjust for batch dimension only if batched *)
          | E_reshape { t_in; new_shape } ->
              Some
                (fun k ->
                  (* User shape is logical. Preserve present batch prefix and
                     reshape only the logical tail when element counts match;
                     otherwise leave unchanged and let broadcasting handle
                     it. *)
                  let s_phys = phys_shape_of t_in in
                  let nbd =
                    PhysicalTbl.get_all_bdims env.shared t_in
                    |> List.filter (fun (lv, _) -> lv <= env.level)
                    |> List.length
                  in
                  let tail_len = max 0 (Array.length s_phys - nbd) in
                  let old_tail =
                    if tail_len = 0 then [||] else Array.sub s_phys nbd tail_len
                  in
                  let prod arr = Array.fold_left (fun a b -> a * b) 1 arr in
                  let prod_old = prod old_tail in
                  let target_logical =
                    match Symbolic_shape.eval new_shape with
                    | Some arr -> arr
                    | None ->
                        failwith "vmap reshape requires concrete target shape"
                  in
                  let prod_new = prod target_logical in
                  let prefix =
                    if nbd = 0 then [||] else Array.sub s_phys 0 nbd
                  in
                  let phys_target = Array.append prefix target_logical in
                  let result =
                    if prod_old = prod_new then
                      op_reshape t_in (Symbolic_shape.of_ints phys_target)
                    else t_in
                  in
                  set_bdim result (get_bdim t_in);
                  continue k result)
          | E_expand { t_in; new_target_shape } ->
              Some
                (fun k ->
                  let new_target_arr =
                    match Symbolic_shape.eval new_target_shape with
                    | Some arr -> arr
                    | None ->
                        failwith "vmap expand requires concrete target shape"
                  in
                  (* Logical expand: canonicalize batches, then broadcast
                     current logical dims with the requested new_target_shape.
                     Keep the existing batch prefix untouched. *)
                  let t0 = canonicalize_batch_positions t_in in
                  let s = phys_shape_of t0 in
                  dprintf "E_expand: s=%s new_target=%s" (pp_shape s)
                    (pp_shape new_target_arr);
                  let nbd = prefix_len_by_batch_sizes t0 in
                  let prefix = if nbd = 0 then [||] else Array.sub s 0 nbd in
                  let cur_log =
                    let sl = Array.length s in
                    if sl > nbd then Array.sub s nbd (sl - nbd) else [||]
                  in
                  dprintf "E_expand: nbd=%d prefix=%s cur_log=%s" nbd
                    (pp_shape prefix) (pp_shape cur_log);
                  (* If the requested target already includes the current
                     prefix, strip it *)
                  let logical_target =
                    let lt = Array.length new_target_arr in
                    if lt >= nbd then
                      let starts_with_prefix =
                        let ok = ref true in
                        let i = ref 0 in
                        while !ok && !i < nbd && !i < lt do
                          if new_target_arr.(!i) <> prefix.(!i) then ok := false;
                          incr i
                        done;
                        !ok
                      in
                      if starts_with_prefix then
                        Array.sub new_target_arr nbd (lt - nbd)
                      else new_target_arr
                    else new_target_arr
                  in
                  (* Align ranks by left-padding current logical dims with 1s *)
                  let lt_len = Array.length logical_target in
                  let cl_len = Array.length cur_log in
                  let cur_log_padded =
                    if cl_len >= lt_len then cur_log
                    else Array.append (Array.make (lt_len - cl_len) 1) cur_log
                  in
                  dprintf "E_expand: logical_target=%s cur_log_padded=%s"
                    (pp_shape logical_target) (pp_shape cur_log_padded);
                  (* Only expand if each dim is either equal or 1; otherwise,
                     skip *)
                  let broadcastable =
                    let ok = ref true in
                    for i = 0 to lt_len - 1 do
                      let cur = cur_log_padded.(i) in
                      let tgt = logical_target.(i) in
                      if not (cur = tgt || cur = 1) then ok := false
                    done;
                    !ok
                  in
                  if not broadcastable then (
                    dprintf "E_expand: skip (not broadcastable)";
                    (* Normalize rank by reshaping to prefix @ cur_log_padded so
                       downstream indexing (e.g., shrink/permutation) stays
                       consistent. *)
                    let fallback_phys = Array.append prefix cur_log_padded in
                    let rshape = phys_reshape t0 fallback_phys in
                    copy_bdims_same ~src:t0 ~dst:rshape;
                    set_bdim rshape (get_bdim t_in);
                    continue k rshape)
                  else
                    let target_log =
                      Shape.broadcast cur_log_padded logical_target
                    in
                    let target_phys = Array.append prefix target_log in
                    dprintf "E_expand: target_log=%s target_phys=%s"
                      (pp_shape target_log) (pp_shape target_phys);
                    let result = broadcast_to_canonical t0 target_phys in
                    set_bdim result (get_bdim t_in);
                    continue k result)
          | E_permute { t_in; axes } ->
              Some
                (fun k ->
                  match get_bdim t_in with
                  | None ->
                      let result = op_permute t_in axes in
                      set_bdim result None;
                      continue k result
                  | Some p ->
                      let rank = Array.length (T.shape t_in) in
                      if Array.length axes = rank then (
                        (* Physical permutation: apply as-is and move bdim
                           accordingly *)
                        let result = op_permute t_in axes in
                        (* Find new position of previous p *)
                        let new_p =
                          let idx = ref 0 in
                          while !idx < rank && axes.(!idx) <> p do
                            incr idx
                          done;
                          if !idx >= rank then p else !idx
                        in
                        set_bdim result (Some new_p);
                        continue k result)
                      else
                        (* Logical permutation: build physical permutation
                           keeping p fixed *)
                        let rank_log = rank - 1 in
                        if Array.length axes <> rank_log then
                          failwith "vmap: permute axes length mismatch"
                        else
                          let phys = Array.init rank (fun _ -> -1) in
                          phys.(p) <- p;
                          Array.iteri
                            (fun j old_log ->
                              let old_phys = phys_axis ~bdim:p old_log in
                              let new_phys = phys_axis ~bdim:p j in
                              phys.(new_phys) <- old_phys)
                            axes;
                          let result = op_permute t_in phys in
                          set_bdim result (Some p);
                          continue k result)
          (* Matrix multiplication *)
          | E_matmul { out; a; b } ->
              Some
                (fun k ->
                  let a = canonicalize_batch_positions a in
                  let b = canonicalize_batch_positions b in
                  let ba = get_bdim a and bb = get_bdim b in
                  let p =
                    match (ba, bb) with
                    | Some pa, Some pb -> min pa pb
                    | Some pa, None -> pa
                    | None, Some pb -> pb
                    | None, None -> 0
                  in
                  let a', b' =
                    if ba = None && bb = None then (a, b)
                    else (align_to p a, align_to p b)
                  in
                  op_matmul ~out a' b';
                  set_bdim out
                    (match (ba, bb) with None, None -> None | _ -> Some p);
                  continue k ())
          (* Where operation *)
          | E_where { out; condition; if_true; if_false } ->
              Some
                (fun k ->
                  (* Canonicalize and unify outer batch dims across all three
                     operands *)
                  let condition =
                    condition
                    |> add_missing_outer_bdims ~like:if_true
                    |> add_missing_outer_bdims ~like:if_false
                    |> canonicalize_batch_positions
                  in
                  let if_true =
                    if_true
                    |> add_missing_outer_bdims ~like:condition
                    |> add_missing_outer_bdims ~like:if_false
                    |> canonicalize_batch_positions
                  in
                  let if_false =
                    if_false
                    |> add_missing_outer_bdims ~like:condition
                    |> add_missing_outer_bdims ~like:if_true
                    |> canonicalize_batch_positions
                  in
                  let bc = get_bdim condition in
                  let bt = get_bdim if_true in
                  let bf = get_bdim if_false in

                  (* Determine target position: use leftmost batch position if
                     any *)
                  let p =
                    match (bc, bt, bf) with
                    | Some pc, Some pt, Some pf -> min pc (min pt pf)
                    | Some pc, Some pt, None -> min pc pt
                    | Some pc, None, Some pf -> min pc pf
                    | None, Some pt, Some pf -> min pt pf
                    | Some pc, None, None -> pc
                    | None, Some pt, None -> pt
                    | None, None, Some pf -> pf
                    | None, None, None -> 0
                  in

                  let any_batched =
                    Option.is_some bc || Option.is_some bt || Option.is_some bf
                  in

                  let condition', if_true', if_false' =
                    if any_batched then
                      ( align_to p condition,
                        align_to p if_true,
                        align_to p if_false )
                    else (condition, if_true, if_false)
                  in

                  (* Compute per-operand prefix length and broadcast logical
                     shapes *)
                  let sc = phys_shape_of condition'
                  and st = phys_shape_of if_true'
                  and sf = phys_shape_of if_false' in
                  dprintf "E_where: sc=%s st=%s sf=%s" (pp_shape sc)
                    (pp_shape st) (pp_shape sf);
                  let c_prefix_len = prefix_len_by_batch_sizes condition' in
                  let t_prefix_len = prefix_len_by_batch_sizes if_true' in
                  let f_prefix_len = prefix_len_by_batch_sizes if_false' in
                  let nbd = max c_prefix_len (max t_prefix_len f_prefix_len) in
                  let c_log =
                    Array.sub sc c_prefix_len (Array.length sc - c_prefix_len)
                  in
                  let t_log =
                    Array.sub st t_prefix_len (Array.length st - t_prefix_len)
                  in
                  let f_log =
                    Array.sub sf f_prefix_len (Array.length sf - f_prefix_len)
                  in
                  dprintf "E_where: nbd=%d c_prefix=%d t_prefix=%d f_prefix=%d"
                    nbd c_prefix_len t_prefix_len f_prefix_len;
                  (* Align ranks by left-padding with 1s to max logical rank *)
                  let max_len =
                    max (Array.length c_log)
                      (max (Array.length t_log) (Array.length f_log))
                  in
                  let pad_left v =
                    let lv = Array.length v in
                    if lv >= max_len then v
                    else Array.append (Array.make (max_len - lv) 1) v
                  in
                  let c_log = pad_left c_log in
                  let t_log = pad_left t_log in
                  let f_log = pad_left f_log in
                  let target_log =
                    Shape.broadcast c_log (Shape.broadcast t_log f_log)
                  in
                  let target_pref =
                    Array.init nbd (fun lv ->
                        try Hashtbl.find env.batch_sizes lv
                        with Not_found -> 1)
                  in
                  let target_phys = Array.append target_pref target_log in
                  dprintf "E_where: target_log=%s target_phys=%s"
                    (pp_shape target_log) (pp_shape target_phys);
                  let condition'' =
                    broadcast_to_canonical condition' target_phys
                  in
                  let if_true'' = broadcast_to_canonical if_true' target_phys in
                  let if_false'' =
                    broadcast_to_canonical if_false' target_phys
                  in

                  op_where ~out condition'' if_true'' if_false'';
                  set_bdim out (if any_batched then Some p else None);
                  continue k ())
          (* Cast operation *)
          | E_cast { t_in; target_dtype } ->
              Some
                (fun k ->
                  let result = op_cast t_in target_dtype in
                  set_bdim result (get_bdim t_in);
                  continue k result)
          (* Copy operations *)
          | E_contiguous { t_in } ->
              Some
                (fun k ->
                  let result = op_contiguous t_in in
                  set_bdim result (get_bdim t_in);
                  continue k result)
          | E_copy { t_in } ->
              Some
                (fun k ->
                  let result = op_copy t_in in
                  set_bdim result (get_bdim t_in);
                  continue k result)
          (* Operations that need more complex handling *)
          | E_gather { data; indices; axis } ->
              Some
                (fun k ->
                  let bd = get_bdim data and bi = get_bdim indices in
                  match bd with
                  | None ->
                      let result = op_gather data indices axis in
                      set_bdim result bi;
                      continue k result
                  | Some p ->
                      let adjusted_axis = phys_axis ~bdim:p axis in
                      let indices' =
                        if Option.is_none bi then align_to p indices
                        else indices
                      in
                      let result = op_gather data indices' adjusted_axis in
                      set_bdim result (Some p);
                      continue k result)
          | E_scatter { data_template; indices; updates; axis } ->
              Some
                (fun k ->
                  let bd = get_bdim data_template in
                  let bi = get_bdim indices in
                  let bu = get_bdim updates in
                  match bd with
                  | None ->
                      let result =
                        op_scatter data_template indices updates axis
                      in
                      set_bdim result
                        (match (bi, bu) with None, None -> None | _ -> Some 0);
                      continue k result
                  | Some p ->
                      let adjusted_axis = phys_axis ~bdim:p axis in
                      let indices' =
                        if Option.is_none bi then align_to p indices
                        else indices
                      in
                      let updates' =
                        if Option.is_none bu then align_to p updates
                        else updates
                      in
                      let result =
                        op_scatter data_template indices' updates' adjusted_axis
                      in
                      set_bdim result (Some p);
                      continue k result)
          | E_cat { t_list; axis } ->
              Some
                (fun k ->
                  let bdims = List.map get_bdim t_list in
                  let any_batched = List.exists Option.is_some bdims in
                  if not any_batched then (
                    let result = op_cat t_list axis in
                    set_bdim result None;
                    continue k result)
                  else
                    (* Find leftmost batch position *)
                    let p =
                      List.fold_left
                        (fun acc bd ->
                          match bd with
                          | Some p' -> (
                              match acc with
                              | None -> Some p'
                              | Some a -> Some (min a p'))
                          | None -> acc)
                        None bdims
                      |> Option.get
                    in
                    (* Align all tensors to position p *)
                    let t_list' = List.map (fun t -> align_to p t) t_list in
                    let adjusted_axis = phys_axis ~bdim:p axis in
                    let result = op_cat t_list' adjusted_axis in
                    set_bdim result (Some p);
                    continue k result)
          | E_pad { t_in; padding_config; fill_value } ->
              Some
                (fun k ->
                  match get_bdim t_in with
                  | None ->
                      let result = op_pad t_in padding_config fill_value in
                      set_bdim result None;
                      continue k result
                  | Some p ->
                      (* Insert no padding for batch dimension at p *)
                      let adjusted_padding =
                        let n = Array.length padding_config + 1 in
                        Array.init n (fun i ->
                            if i = p then (0, 0)
                            else
                              let j = if i < p then i else i - 1 in
                              padding_config.(j))
                      in
                      let result = op_pad t_in adjusted_padding fill_value in
                      set_bdim result (Some p);
                      continue k result)
          | E_shrink { t_in; limits } ->
              Some
                (fun k ->
                  match get_bdim t_in with
                  | None ->
                      let result = op_shrink t_in limits in
                      set_bdim result None;
                      continue k result
                  | Some p ->
                      (* Don't shrink batch dimension at p *)
                      let adjusted_limits =
                        let n = Array.length limits + 1 in
                        Array.init n (fun i ->
                            if i = p then (0, axis_size)
                            else
                              let j = if i < p then i else i - 1 in
                              limits.(j))
                      in
                      let result = op_shrink t_in adjusted_limits in
                      set_bdim result (Some p);
                      continue k result)
          | E_flip { t_in; dims_to_flip } ->
              Some
                (fun k ->
                  match get_bdim t_in with
                  | None ->
                      let result = op_flip t_in dims_to_flip in
                      set_bdim result None;
                      continue k result
                  | Some p ->
                      (* Don't flip batch dimension at p *)
                      let adjusted_dims =
                        let n = Array.length dims_to_flip + 1 in
                        Array.init n (fun i ->
                            if i = p then false
                            else
                              let j = if i < p then i else i - 1 in
                              dims_to_flip.(j))
                      in
                      let result = op_flip t_in adjusted_dims in
                      set_bdim result (Some p);
                      continue k result)
          | E_as_strided { t_in; new_shape; new_strides; offset } ->
              Some
                (fun k ->
                  match get_bdim t_in with
                  | None ->
                      let result =
                        op_as_strided t_in
                          (Symbolic_shape.of_ints new_shape)
                          new_strides offset
                      in
                      set_bdim result None;
                      continue k result
                  | Some p ->
                      (* Insert batch dimension at p *)
                      let batched_shape = insert_at new_shape p axis_size in
                      (* Calculate batch stride from trailing dimensions *)
                      let orig_shape = T.shape t_in in
                      let trailing_len = Array.length orig_shape - (p + 1) in
                      let trailing =
                        if trailing_len <= 0 then [||]
                        else Array.sub orig_shape (p + 1) trailing_len
                      in
                      let batch_stride =
                        if Array.length trailing = 0 then 1
                        else Array.fold_left ( * ) 1 trailing
                      in
                      let batched_strides =
                        insert_at new_strides p batch_stride
                      in
                      let result =
                        op_as_strided t_in
                          (Symbolic_shape.of_ints batched_shape)
                          batched_strides offset
                      in
                      set_bdim result (Some p);
                      continue k result)
          (* Let other operations pass through *)
          | _ -> None);
  }

(* ============================================================================
   The Main vmap Function
   ============================================================================ *)

let vmap ?(in_axes = Single (Map 0)) ?(out_axes = OutSingle (Some 0)) ?axis_name
    ?axis_size f =
 fun input ->
  (* Extract axis specifications *)
  let axis_spec = extract_axis_spec in_axes in
  let out_axis_spec = extract_out_axis_spec out_axes in

  (* Establish or extend the vmap environment (partial; finalize after size). *)
  let parent_env = !current_env in
  let shared =
    match parent_env with Some e -> e.shared | None -> PhysicalTbl.create ()
  in
  let level = match parent_env with Some e -> e.level + 1 | None -> 0 in
  let batched_tensors = PhysicalTbl.create () in
  (* Clear any stale mapping at this level for the input before shape queries *)
  PhysicalTbl.set_bdim shared input ~level ~bdim:None;

  (* Determine batch size and set bdim without moving axes *)
  let axis_size =
    match (axis_spec, axis_size) with
    | Map axis_idx, None ->
        (* axis_idx is logical; adjust to physical by adding OUTER prefix
           length. *)
        let shape = T.shape input in
        (* Compute physical axis by accounting for existing outer batch dims
           already present on this input. *)
        let physical_k =
          let outer_bdims =
            List.init level (fun lev ->
                PhysicalTbl.get_bdim shared input ~level:lev)
            |> List.filter_map (fun x -> x)
            |> List.sort compare
          in
          List.fold_left
            (fun k_acc outer_bdim ->
              if outer_bdim <= k_acc then k_acc + 1 else k_acc)
            axis_idx outer_bdims
        in
        if physical_k < 0 || physical_k >= Array.length shape then
          failwith
            (Printf.sprintf "vmap: invalid axis %d (physical %d) for rank %d"
               axis_idx physical_k (Array.length shape));
        shape.(physical_k)
    | NoMap, Some size -> size
    | NoMap, None ->
        failwith "vmap: axis_size must be provided when in_axes is NoMap"
    | Map axis_idx, Some size ->
        (* Verify provided size matches the physical dimension corresponding to
           logical axis. *)
        let shape = T.shape input in
        let physical_k =
          let outer_bdims =
            List.init level (fun lev ->
                PhysicalTbl.get_bdim shared input ~level:lev)
            |> List.filter_map (fun x -> x)
            |> List.sort compare
          in
          List.fold_left
            (fun k_acc outer_bdim ->
              if outer_bdim <= k_acc then k_acc + 1 else k_acc)
            axis_idx outer_bdims
        in
        if physical_k < 0 || physical_k >= Array.length shape then
          failwith
            (Printf.sprintf "vmap: invalid axis %d (physical %d) for rank %d"
               axis_idx physical_k (Array.length shape));
        if shape.(physical_k) <> size then
          failwith
            (Printf.sprintf
               "vmap: axis_size %d doesn't match axis %d (physical %d) size %d"
               size axis_idx physical_k shape.(physical_k));
        size
  in

  (* Finalize env now that axis_size is known *)
  let batch_sizes =
    match parent_env with
    | Some e -> Hashtbl.copy e.batch_sizes
    | None -> Hashtbl.create 8
  in
  Hashtbl.replace batch_sizes level axis_size;
  let env = { level; shared; batch_sizes } in

  (* Mark input bdim, accounting for outer batch dimensions *)
  (match axis_spec with
  | Map k ->
      (* Adjust logical position to physical by adding OUTER prefix length *)
      let physical_k =
        let outer_bdims =
          List.init level (fun lev ->
              PhysicalTbl.get_bdim shared input ~level:lev)
          |> List.filter_map (fun x -> x)
          |> List.sort compare
        in
        List.fold_left
          (fun k_acc outer_bdim ->
            if outer_bdim <= k_acc then k_acc + 1 else k_acc)
          k outer_bdims
      in
      PhysicalTbl.set_bdim batched_tensors input ~level ~bdim:(Some physical_k);
      PhysicalTbl.set_bdim shared input ~level ~bdim:(Some physical_k)
  | NoMap ->
      PhysicalTbl.set_bdim batched_tensors input ~level ~bdim:None;
      PhysicalTbl.set_bdim shared input ~level ~bdim:None);

  (* Create the vmap handler with the level and local table *)
  let vmap_handler =
    make_vmap_handler ~env ~axis_size ~batched_tensors out_axis_spec axis_name
  in

  with_env env (fun () ->
      match Effect.Deep.match_with f input vmap_handler with
      | result ->
          PhysicalTbl.clear_level env.shared level;
          result
      | exception exn ->
          PhysicalTbl.clear_level env.shared level;
          raise exn)

(* vmaps for multiple arguments *)
let vmaps ?(in_axes = []) ?(out_axes = OutSingle (Some 0)) ?axis_name ?axis_size
    f =
 fun inputs ->
  (* Default to Map 0 for all inputs if in_axes is empty *)
  let axis_specs =
    if in_axes = [] then List.map (fun _ -> Map 0) inputs
    else if List.length in_axes <> List.length inputs then
      failwith "vmaps: in_axes must have the same length as inputs or be empty"
    else in_axes
  in

  let out_axis_spec = extract_out_axis_spec out_axes in

  (* Establish or extend the vmap environment (partial; finalize after size). *)
  let parent_env = !current_env in
  let shared =
    match parent_env with Some e -> e.shared | None -> PhysicalTbl.create ()
  in
  let level = match parent_env with Some e -> e.level + 1 | None -> 0 in
  let batched_tensors = PhysicalTbl.create () in
  (* Clear any stale mapping at this level for inputs before shape queries *)
  List.iter
    (fun inp -> PhysicalTbl.set_bdim shared inp ~level ~bdim:None)
    inputs;

  (* Determine batch size from first mapped input *)
  let axis_size =
    match axis_size with
    | Some size -> size
    | None ->
        (* Choose the maximum mapped size across inputs to allow broadcasting
           smaller ones *)
        let rec collect_sizes acc ins sp =
          match (ins, sp) with
          | input :: rest_i, Map axis_idx :: rest_s ->
              let shape = T.shape input in
              let physical_axis =
                let outer_bdims =
                  List.init level (fun lev ->
                      PhysicalTbl.get_bdim shared input ~level:lev)
                  |> List.filter_map (fun x -> x)
                  |> List.sort compare
                in
                List.fold_left
                  (fun k_acc outer_bdim ->
                    if outer_bdim <= k_acc then k_acc + 1 else k_acc)
                  axis_idx outer_bdims
              in
              if physical_axis < 0 || physical_axis >= Array.length shape then
                failwith
                  (Printf.sprintf
                     "vmaps: invalid axis %d (physical %d) for rank %d" axis_idx
                     physical_axis (Array.length shape));
              collect_sizes (max acc shape.(physical_axis)) rest_i rest_s
          | _ :: rest_i, NoMap :: rest_s -> collect_sizes acc rest_i rest_s
          | [], [] -> acc
          | _ -> failwith "vmaps: internal error"
        in
        collect_sizes 1 inputs axis_specs
  in

  (* Finalize env now that axis_size is known *)
  let batch_sizes =
    match parent_env with
    | Some e -> Hashtbl.copy e.batch_sizes
    | None -> Hashtbl.create 8
  in
  Hashtbl.replace batch_sizes level axis_size;
  let env = { level; shared; batch_sizes } in

  (* Mark each input's bdim, accounting for outer batch dimensions *)
  List.iter2
    (fun input axis_spec ->
      match axis_spec with
      | Map axis_idx ->
          (* Check how many batch dimensions from outer levels come before
             axis_idx *)
          let physical_idx =
            let outer_bdims =
              List.init level (fun lev ->
                  PhysicalTbl.get_bdim shared input ~level:lev)
              |> List.filter_map (fun x -> x)
              |> List.sort compare
            in
            List.fold_left
              (fun k_acc outer_bdim ->
                if outer_bdim <= k_acc then k_acc + 1 else k_acc)
              axis_idx outer_bdims
          in
          (* If this input's mapped dimension is size 1 and axis_size > 1,
             broadcast it. *)
          let input_shape = T.shape input in
          let input_axis_size = input_shape.(physical_idx) in
          let input' =
            if input_axis_size = axis_size then input
            else if input_axis_size = 1 then
              (* Build target physical shape by replacing that axis with
                 axis_size *)
              let target =
                Array.mapi
                  (fun i d -> if i = physical_idx then axis_size else d)
                  input_shape
              in
              op_expand input (Symbolic_shape.of_ints target)
            else
              failwith
                (Printf.sprintf
                   "vmaps: cannot broadcast mapped axis of size %d to %d"
                   input_axis_size axis_size)
          in
          PhysicalTbl.set_bdim batched_tensors input' ~level
            ~bdim:(Some physical_idx);
          PhysicalTbl.set_bdim shared input' ~level ~bdim:(Some physical_idx)
      | NoMap ->
          PhysicalTbl.set_bdim batched_tensors input ~level ~bdim:None;
          PhysicalTbl.set_bdim shared input ~level ~bdim:None)
    inputs axis_specs;

  (* Create the vmap handler with the level and local table *)
  let vmap_handler =
    make_vmap_handler ~env ~axis_size ~batched_tensors out_axis_spec axis_name
  in

  with_env env (fun () ->
      match
        Effect.Deep.match_with (fun inputs -> f inputs) inputs vmap_handler
      with
      | result ->
          PhysicalTbl.clear_level env.shared level;
          result
      | exception exn ->
          PhysicalTbl.clear_level env.shared level;
          raise exn)
