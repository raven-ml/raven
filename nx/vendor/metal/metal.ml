open Objc
open Ctypes

module Block = struct
  include Block

  let make' f ~typ =
    let b = Ctypes_memory.make t in
    setf b isa (Objc.get_class "__NSStackBlock__");
    setf b descriptor desc_ptr;
    setf b invoke
      (coerce
         (Foreign.funptr ~runtime_lock:true ~thread_registration:true typ)
         (ptr void) f);
    setf b flags 0;
    allocate t b |> coerce (ptr t) (ptr void)

  (** Create a global block which encapsulates the code for execution at a later
      time. *)
  let make f ~args ~return =
    let typ = Objc_type.method_typ ~args:(Objc_type.id :: args) return in
    make' f ~typ
end

let msg_send_suspended ~self ~cmd ~typ =
  Foreign.foreign ~release_runtime_lock:true "objc_msgSend"
    (id @-> _SEL @-> typ)
    self cmd

let msg_send ~self ~select ~typ =
  let cmd = selector select in
  Objc.Low_level.send_message ~self ~cmd ~typ

let msg_send_suspended ~self ~select ~typ =
  let cmd = selector select in
  msg_send_suspended ~self ~cmd ~typ

type lifetime = Lifetime : 'a -> lifetime

(* While we could use Foreign.funptr to attach lifetimes to ARC objects, we'd
   need to do this for every function pointer type, and it's easier to just use
   a custom type. *)
type payload = { id : Objc.object_t; mutable lifetime : lifetime }

let nil_ptr : Objc.object_t ptr = coerce (ptr void) (ptr Objc.id) null
let nil : Objc.object_t = nil

(* Helper to convert NSString to OCaml string *)
let ocaml_string_from_nsstring nsstring_id =
  if is_nil nsstring_id then ""
  else msg_send ~self:nsstring_id ~select:"UTF8String" ~typ:(returning string)

let from_nsarray nsarray_id =
  if is_nil nsarray_id then [||]
  else
    let count =
      msg_send ~self:nsarray_id ~select:"count" ~typ:(returning size_t)
    in
    let count_int = Unsigned.Size_t.to_int count in
    Array.init count_int (fun i ->
        let obj_id =
          msg_send ~self:nsarray_id ~select:"objectAtIndex:"
            ~typ:(ulong @-> returning Objc.id)
            (Unsigned.ULong.of_int i)
        in
        obj_id (* Or further processing if needed *))

let to_nsarray ?count buffer =
  let nsarray_class = Objc.get_class "NSArray" in
  let count = match count with Some c -> c | None -> List.length buffer in
  if count = 0 then
    (* Create empty array *)
    msg_send ~self:nsarray_class ~select:"array" ~typ:(returning Objc.id)
  else
    (* Create array with contents *)
    let array_with_objects_count =
      msg_send ~self:nsarray_class ~select:"arrayWithObjects:count:"
        ~typ:(ptr Objc.id @-> size_t @-> returning Objc.id)
    in
    (* Note: the buffer is copied, so no need to keep the array alive. *)
    let buffer_ptr = CArray.start (CArray.of_list Objc.id buffer) in
    array_with_objects_count buffer_ptr (Unsigned.Size_t.of_int count)

(* Error Handling Helper *)
let get_error_description nserror =
  if is_nil nserror then "No error"
  else
    let localized_description =
      msg_send ~self:nserror ~select:"localizedDescription"
        ~typ:(returning Objc.id)
    in
    if is_nil localized_description then "Unknown error (no description)"
    else ocaml_string_from_nsstring localized_description

(* Check error pointer immediately after the call *)
let check_error label err_ptr =
  (* Dereference to get the ptr id *)
  assert (not (is_nil err_ptr));
  (* Check if the pointer itself is nil *)
  let error_id = !@err_ptr in
  (* Dereference the non-nil pointer to get the id *)
  if not (is_nil error_id) then
    let desc = get_error_description error_id in
    failwith (Printf.sprintf "%s failed: %s" label desc)

(* === Basic Structures === *)

module Size = struct
  type t = { width : int; height : int; depth : int }
  type mtlsize
  type mtl = mtlsize structure ptr

  let mtlsize_t : mtlsize structure typ = structure "MTLSize"
  let width_field = field mtlsize_t "width" ulong
  let height_field = field mtlsize_t "height" ulong
  let depth_field = field mtlsize_t "depth" ulong
  let () = seal mtlsize_t

  let from_struct (s : mtl) : t =
    let s = !@s in
    {
      width = Unsigned.ULong.to_int (getf s width_field);
      height = Unsigned.ULong.to_int (getf s height_field);
      depth = Unsigned.ULong.to_int (getf s depth_field);
    }

  (* Convert OCaml record to C structure for function calls *)
  let make ~width ~height ~depth : mtl =
    let s = make mtlsize_t in
    setf s width_field (Unsigned.ULong.of_int width);
    setf s height_field (Unsigned.ULong.of_int height);
    setf s depth_field (Unsigned.ULong.of_int depth);
    s.structured

  let to_value (t : t) : mtl =
    make ~width:t.width ~height:t.height ~depth:t.depth
end

module Origin = struct
  type t = { x : int; y : int; z : int }
  type mtlorigin
  type mtl = mtlorigin structure ptr

  let mtlorigin_t : mtlorigin structure typ = structure "MTLOrigin"
  let x_field = field mtlorigin_t "x" ulong
  let y_field = field mtlorigin_t "y" ulong
  let z_field = field mtlorigin_t "z" ulong
  let () = seal mtlorigin_t

  let from_struct (s : mtl) : t =
    let s = !@s in
    {
      x = Unsigned.ULong.to_int (getf s x_field);
      y = Unsigned.ULong.to_int (getf s y_field);
      z = Unsigned.ULong.to_int (getf s z_field);
    }

  (* Convert OCaml record to C structure for function calls *)
  let make ~x ~y ~z : mtl =
    let s = make mtlorigin_t in
    setf s x_field (Unsigned.ULong.of_int x);
    setf s y_field (Unsigned.ULong.of_int y);
    setf s z_field (Unsigned.ULong.of_int z);
    s.structured

  let to_value (t : t) : mtl = make ~x:t.x ~y:t.y ~z:t.z
end

module Region = struct
  type t = { origin : Origin.t; size : Size.t }
  type mtlregion
  type mtl = mtlregion structure ptr

  let mtlregion_t : mtlregion structure typ = structure "MTLRegion"
  let origin_field = field mtlregion_t "origin" Origin.mtlorigin_t
  let size_field = field mtlregion_t "size" Size.mtlsize_t
  let () = seal mtlregion_t

  let from_struct (s : mtl) : t =
    let s = !@s in
    {
      origin = Origin.from_struct (getf s origin_field).structured;
      size = Size.from_struct (getf s size_field).structured;
    }

  (* Convert OCaml record to C structure for function calls *)
  let to_value (t : t) : mtl =
    let s = make mtlregion_t in
    setf s origin_field !@(Origin.to_value t.origin);
    setf s size_field !@(Size.to_value t.size);
    s.structured

  let make ~x ~y ~z ~width ~height ~depth : mtl =
    to_value { origin = { x; y; z }; size = { width; height; depth } }
end

module Range = struct
  type t = { location : int; length : int }
  type nsrange
  type ns = nsrange structure ptr

  let nsrange_t : nsrange structure typ = structure "_NSRange"
  let location_field = field nsrange_t "location" ulong
  let length_field = field nsrange_t "length" ulong
  let () = seal nsrange_t

  let from_struct (s : ns) : t =
    let s = !@s in
    {
      location = Unsigned.ULong.to_int (getf s location_field);
      length = Unsigned.ULong.to_int (getf s length_field);
    }

  let make ~location ~length =
    let ns_range = make nsrange_t in
    setf ns_range location_field (Unsigned.ULong.of_int location);
    setf ns_range length_field (Unsigned.ULong.of_int length);
    ns_range.structured

  let to_value t = make ~location:t.location ~length:t.length
end

let gc ~select obj =
  if is_nil obj then failwith @@ "Failed to create object via " ^ select;
  obj |> gc_autorelease

(* Same as Objc.new_object, but with nil check and autoreleases the object *)
let new_gc ~class_name =
  let obj = alloc_object class_name in
  if is_nil obj then
    failwith @@ "Failed to create object of class " ^ class_name;
  obj |> init |> gc_autorelease

module Device = struct
  type t = Objc.object_t

  let create_system_default () =
    let select = "MTLCreateSystemDefaultDevice" in
    gc ~select (Foreign.foreign select (void @-> returning Objc.id) ())

  let copy_all_devices () =
    let select = "MTLCopyAllDevices" in
    let devices_nsarray =
      gc ~select (Foreign.foreign select (void @-> returning Objc.id) ())
    in
    from_nsarray devices_nsarray

  module GPUFamily = struct
    type t =
      | Apple1
      | Apple2
      | Apple3
      | Apple4
      | Apple5
      | Apple6
      | Apple7
      | Apple8
      | Apple9
      | Mac1
      | Mac2
      | Common1
      | Common2
      | Common3
      | MacCatalyst1
      | MacCatalyst2
      | Metal3

    let to_int = function
      | Apple1 -> 1001
      | Apple2 -> 1002
      | Apple3 -> 1003
      | Apple4 -> 1004
      | Apple5 -> 1005
      | Apple6 -> 1006
      | Apple7 -> 1007
      | Apple8 -> 1008
      | Apple9 -> 1009
      | Mac1 -> 2001
      | Mac2 -> 2002
      | Common1 -> 3001
      | Common2 -> 3002
      | Common3 -> 3003
      | MacCatalyst1 -> 4001
      | MacCatalyst2 -> 4002
      | Metal3 -> 5001

    let from_int = function
      | 1001 -> Apple1
      | 1002 -> Apple2
      | 1003 -> Apple3
      | 1004 -> Apple4
      | 1005 -> Apple5
      | 1006 -> Apple6
      | 1007 -> Apple7
      | 1008 -> Apple8
      | 1009 -> Apple9
      | 2001 -> Mac1
      | 2002 -> Mac2
      | 3001 -> Common1
      | 3002 -> Common2
      | 3003 -> Common3
      | 4001 -> MacCatalyst1
      | 4002 -> MacCatalyst2
      | 5001 -> Metal3
      | _ -> invalid_arg "Unknown MTLGPUFamily value"
  end

  let supports_family (self : t) (family : GPUFamily.t) : bool =
    msg_send ~self ~select:"supportsFamily:"
      ~typ:(long @-> returning bool)
      (Signed.Long.of_int (GPUFamily.to_int family))

  module ArgumentBuffersTier = struct
    type t = Tier1 | Tier2

    let from_ulong (i : Unsigned.ULong.t) =
      if Unsigned.ULong.equal i Unsigned.ULong.zero then Tier1
      else if Unsigned.ULong.equal i Unsigned.ULong.one then Tier2
      else
        invalid_arg
          ("Unknown ArgumentBuffersTier: " ^ Unsigned.ULong.to_string i)

    let to_ulong (t : t) : Unsigned.ulong =
      match t with Tier1 -> Unsigned.ULong.zero | Tier2 -> Unsigned.ULong.one
  end

  (* Provide local modules with manual sexp converters for Unsigned types *)
  type ulong = Unsigned.ULong.t
  type ullong = Unsigned.ULLong.t

  type attributes = {
    name : string;
    registry_id : ullong;
    max_threads_per_threadgroup : Size.t;
    max_buffer_length : ulong;
    max_threadgroup_memory_length : ulong;
    argument_buffers_support : ArgumentBuffersTier.t;
    recommended_max_working_set_size : ullong;
    is_low_power : bool;
    is_removable : bool;
    is_headless : bool;
    has_unified_memory : bool;
    peer_count : ulong;
    peer_group_id : ullong;
    supported_gpu_families : GPUFamily.t list;
  }

  let get_attributes (self : t) : attributes =
    let name =
      ocaml_string_from_nsstring
        (msg_send ~self ~select:"name" ~typ:(returning Objc.id))
    in
    let registry_id =
      msg_send ~self ~select:"registryID" ~typ:(returning ullong)
    in
    let max_threads_per_threadgroup_struct : Size.t =
      (* Use Size.t *)
      (* Use the locally defined mtlsize_t struct *)
      let size_struct : Size.mtlsize Ctypes.structure =
        (* Use Size.mtlsize *)
        Objc.Low_level.send_message_stret ~self
          ~cmd:(selector "maxThreadsPerThreadgroup")
          ~typ:(returning Size.mtlsize_t) (* Return the local struct type *)
          ~return_type:Size.mtlsize_t
        (* Pass struct type for stret size check *)
      in
      Size.from_struct size_struct.structured
    in
    let max_buffer_length =
      msg_send ~self ~select:"maxBufferLength" ~typ:(returning ulong)
    in
    let max_threadgroup_memory_length =
      msg_send ~self ~select:"maxThreadgroupMemoryLength" ~typ:(returning ulong)
    in
    let argument_buffers_support_val =
      msg_send ~self ~select:"argumentBuffersSupport" ~typ:(returning ulong)
      (* Enum usually maps to long long *)
    in
    let argument_buffers_support =
      ArgumentBuffersTier.from_ulong argument_buffers_support_val
    in
    let recommended_max_working_set_size =
      msg_send ~self ~select:"recommendedMaxWorkingSetSize"
        ~typ:(returning ullong)
    in
    let is_low_power =
      msg_send ~self ~select:"isLowPower" ~typ:(returning bool)
    in
    let is_removable =
      msg_send ~self ~select:"isRemovable" ~typ:(returning bool)
    in
    let is_headless =
      msg_send ~self ~select:"isHeadless" ~typ:(returning bool)
    in
    let has_unified_memory =
      msg_send ~self ~select:"hasUnifiedMemory" ~typ:(returning bool)
    in
    let peer_count =
      msg_send ~self ~select:"peerCount" ~typ:(returning ulong)
    in
    let peer_group_id =
      msg_send ~self ~select:"peerGroupID" ~typ:(returning ullong)
    in

    (* Test each GPU family and collect the supported ones *)
    let all_families =
      [
        GPUFamily.Apple1;
        GPUFamily.Apple2;
        GPUFamily.Apple3;
        GPUFamily.Apple4;
        GPUFamily.Apple5;
        GPUFamily.Apple6;
        GPUFamily.Apple7;
        GPUFamily.Apple8;
        GPUFamily.Apple9;
        GPUFamily.Mac1;
        GPUFamily.Mac2;
        GPUFamily.Common1;
        GPUFamily.Common2;
        GPUFamily.Common3;
        GPUFamily.MacCatalyst1;
        GPUFamily.MacCatalyst2;
        GPUFamily.Metal3;
      ]
    in
    let supported_gpu_families =
      List.filter (fun family -> supports_family self family) all_families
    in

    {
      name;
      registry_id;
      max_threads_per_threadgroup = max_threads_per_threadgroup_struct;
      max_buffer_length;
      max_threadgroup_memory_length;
      argument_buffers_support;
      recommended_max_working_set_size;
      is_low_power;
      is_removable;
      is_headless;
      has_unified_memory;
      peer_count;
      peer_group_id;
      supported_gpu_families;
    }
end

(* === Resource Configuration === *)

module ResourceOptions = struct
  type t = Unsigned.ULong.t

  (* Storage Modes (MTLStorageMode) *)
  let storage_mode_shared = Unsigned.ULong.of_int 0
  let storage_mode_managed = Unsigned.ULong.of_int 16 (* 1 << 4 *)
  let storage_mode_private = Unsigned.ULong.of_int 32 (* 2 << 4 *)
  let storage_mode_memoryless = Unsigned.ULong.of_int 48 (* 3 << 4 *)

  (* CPU Cache Modes (MTLCPUCacheMode) *)
  let cpu_cache_mode_default_cache = Unsigned.ULong.of_int 0
  let cpu_cache_mode_write_combined = Unsigned.ULong.of_int 1

  (* Hazard Tracking Modes (MTLHazardTrackingMode) *)
  let hazard_tracking_mode_default = Unsigned.ULong.of_int 0 (* 0 << 8 *)
  let hazard_tracking_mode_untracked = Unsigned.ULong.of_int 256 (* 1 << 8 *)
  let hazard_tracking_mode_tracked = Unsigned.ULong.of_int 512 (* 2 << 8 *)

  (* Combine options using Bitmask *)
  let ( + ) = Unsigned.ULong.logor

  (* Helper function to create options *)
  let make ?(storage_mode = storage_mode_shared)
      ?(cpu_cache_mode = cpu_cache_mode_default_cache)
      ?(hazard_tracking_mode = hazard_tracking_mode_default) () =
    storage_mode + cpu_cache_mode + hazard_tracking_mode
end

module PipelineOption = struct
  type t = Unsigned.ULong.t

  let none = Unsigned.ULong.of_int 0
  let argument_info = Unsigned.ULong.of_int 1 (* 1 << 0 *)
  let buffer_type_info = Unsigned.ULong.of_int 2 (* 1 << 1 *)
  let fail_on_binary_archive_miss = Unsigned.ULong.of_int 4 (* 1 << 2 *)
  let ( + ) = Unsigned.ULong.logor
end

module CompileOptions = struct
  type t = Objc.object_t

  let init () = new_gc ~class_name:"MTLCompileOptions"

  module LanguageVersion = struct
    type t = Unsigned.ULong.t

    let version_1_0 = Unsigned.ULong.of_int 0 (* Deprecated *)
    let version_1_1 = Unsigned.ULong.of_int 65537
    let version_1_2 = Unsigned.ULong.of_int 65538
    let version_2_0 = Unsigned.ULong.of_int 131072
    let version_2_1 = Unsigned.ULong.of_int 131073
    let version_2_2 = Unsigned.ULong.of_int 131074
    let version_2_3 = Unsigned.ULong.of_int 131075
    let version_2_4 = Unsigned.ULong.of_int 131076
    let version_3_0 = Unsigned.ULong.of_int 196608
    let version_3_1 = Unsigned.ULong.of_int 196609
    let version_3_2 = Unsigned.ULong.of_int 196610 (* macOS 15.0, iOS 18.0 *)
  end

  module LibraryType = struct
    type t = Unsigned.ULong.t

    let executable = Unsigned.ULong.of_int 0
    let dynamic = Unsigned.ULong.of_int 1
    let to_ulong (t : t) : Unsigned.ulong = t
  end

  module OptimizationLevel = struct
    type t = Unsigned.ULong.t

    let default = Unsigned.ULong.of_int 0
    let size = Unsigned.ULong.of_int 1
    let to_ulong (t : t) : Unsigned.ulong = t
  end

  module MathMode = struct
    type t = Safe | Relaxed | Fast

    let to_ulong = function
      | Safe -> Unsigned.ULong.of_int 0
      | Relaxed -> Unsigned.ULong.of_int 1
      | Fast -> Unsigned.ULong.of_int 2

    let from_ulong ul =
      match Unsigned.ULong.to_int ul with
      | 0 -> Safe
      | 1 -> Relaxed
      | 2 -> Fast
      | n -> failwith (Printf.sprintf "Unknown MathMode value: %d" n)
  end

  module MathFloatingPointFunctions = struct
    type t = Fast | Precise

    let to_ulong = function
      | Fast -> Unsigned.ULong.of_int 0
      | Precise -> Unsigned.ULong.of_int 1

    let from_ulong ul =
      match Unsigned.ULong.to_int ul with
      | 0 -> Fast
      | 1 -> Precise
      | n ->
          failwith
            (Printf.sprintf "Unknown MathFloatingPointFunctions value: %d" n)
  end

  let set_fast_math_enabled (self : t) (value : bool) : unit =
    msg_send ~self ~select:"setFastMathEnabled:"
      ~typ:(bool @-> returning void)
      value

  let get_fast_math_enabled (self : t) : bool =
    msg_send ~self ~select:"fastMathEnabled" ~typ:(returning bool)

  let set_math_mode (self : t) (mode : MathMode.t) : unit =
    msg_send ~self ~select:"setMathMode:"
      ~typ:(ulong @-> returning void)
      (MathMode.to_ulong mode)

  let get_math_mode (self : t) : MathMode.t =
    let mode_val = msg_send ~self ~select:"mathMode" ~typ:(returning ulong) in
    MathMode.from_ulong mode_val

  let set_math_floating_point_functions (self : t)
      (funcs : MathFloatingPointFunctions.t) : unit =
    msg_send ~self ~select:"setMathFloatingPointFunctions:"
      ~typ:(ulong @-> returning void)
      (MathFloatingPointFunctions.to_ulong funcs)

  let get_math_floating_point_functions (self : t) :
      MathFloatingPointFunctions.t =
    let funcs_val =
      msg_send ~self ~select:"mathFloatingPointFunctions" ~typ:(returning ulong)
    in
    MathFloatingPointFunctions.from_ulong funcs_val

  let set_enable_logging (self : t) (value : bool) : unit =
    msg_send ~self ~select:"setEnableLogging:"
      ~typ:(bool @-> returning void)
      value

  let get_enable_logging (self : t) : bool =
    msg_send ~self ~select:"enableLogging" ~typ:(returning bool)

  let set_max_total_threads_per_threadgroup (self : t) (value : int) : unit =
    msg_send ~self ~select:"setMaxTotalThreadsPerThreadgroup:"
      ~typ:(ulong @-> returning void)
      (Unsigned.ULong.of_int value)

  let get_max_total_threads_per_threadgroup (self : t) : int =
    let val_ulong =
      msg_send ~self ~select:"maxTotalThreadsPerThreadgroup"
        ~typ:(returning ulong)
    in
    Unsigned.ULong.to_int val_ulong

  let set_language_version (self : t) (version : LanguageVersion.t) : unit =
    msg_send ~self ~select:"setLanguageVersion:"
      ~typ:(ulong @-> returning void)
      version

  let get_language_version (self : t) : LanguageVersion.t =
    msg_send ~self ~select:"languageVersion" ~typ:(returning ulong)

  let set_library_type (self : t) (lt : LibraryType.t) : unit =
    msg_send ~self ~select:"setLibraryType:" ~typ:(ulong @-> returning void) lt

  let get_library_type (self : t) : LibraryType.t =
    msg_send ~self ~select:"libraryType" ~typ:(returning ulong)

  let set_install_name (self : t) (name : string) : unit =
    let ns_name = new_string name in
    msg_send ~self ~select:"setInstallName:"
      ~typ:(Objc.id @-> returning void)
      ns_name

  let get_install_name (self : t) : string =
    let ns_name =
      msg_send ~self ~select:"installName" ~typ:(returning Objc.id)
    in
    ocaml_string_from_nsstring ns_name

  let set_optimization_level (self : t) (level : OptimizationLevel.t) : unit =
    msg_send ~self ~select:"setOptimizationLevel:"
      ~typ:(ulong @-> returning void)
      level

  let get_optimization_level (self : t) : OptimizationLevel.t =
    msg_send ~self ~select:"optimizationLevel" ~typ:(returning ulong)
end

(* === Resources === *)

module Resource = struct
  type t = Objc.object_t

  let set_label (self : t) label =
    let ns_label = new_string label in
    msg_send ~self ~select:"setLabel:"
      ~typ:(Objc.id @-> returning void)
      ns_label

  let get_label (self : t) =
    let ns_label = msg_send ~self ~select:"label" ~typ:(returning Objc.id) in
    ocaml_string_from_nsstring ns_label

  let get_device (self : t) : Device.t =
    msg_send ~self ~select:"device" ~typ:(returning Objc.id)

  module PurgeableState = struct
    type t = KeepCurrent | NonVolatile | Volatile | Empty

    let to_int = function
      | KeepCurrent -> 1
      | NonVolatile -> 2
      | Volatile -> 3
      | Empty -> 4

    let from_int = function
      | 1 -> KeepCurrent
      | 2 -> NonVolatile
      | 3 -> Volatile
      | 4 -> Empty
      | _ -> invalid_arg "Invalid MTLPurgeableState value"
  end

  let set_purgeable_state (self : t) state =
    let prev_state_ulong =
      msg_send ~self ~select:"setPurgeableState:"
        ~typ:(ulong @-> returning ulong)
        (Unsigned.ULong.of_int (PurgeableState.to_int state))
    in
    PurgeableState.from_int (Unsigned.ULong.to_int prev_state_ulong)

  module CPUCacheMode = struct
    type t = DefaultCache | WriteCombined

    let from_ulong i =
      match Unsigned.ULong.to_int i with
      | 0 -> DefaultCache
      | 1 -> WriteCombined
      | _ -> invalid_arg "Unknown CPUCacheMode"

    let to_ulong = function
      | DefaultCache -> Unsigned.ULong.zero
      | WriteCombined -> Unsigned.ULong.one
  end

  module StorageMode = struct
    type t = Shared | Managed | Private | Memoryless

    let from_ulong i =
      match Unsigned.ULong.to_int i with
      | 0 -> Shared
      | 1 -> Managed (* macOS only *)
      | 2 -> Private
      | 3 -> Memoryless
      | _ -> invalid_arg "Unknown StorageMode"

    let to_ulong = function
      | Shared -> Unsigned.ULong.zero
      | Managed -> Unsigned.ULong.one
      | Private -> Unsigned.ULong.of_int 2
      | Memoryless -> Unsigned.ULong.of_int 3
  end

  module HazardTrackingMode = struct
    type t = Default | Untracked | Tracked

    let from_ulong i =
      match Unsigned.ULong.to_int i with
      | 0 -> Default
      | 1 -> Untracked
      | 2 -> Tracked
      | _ -> invalid_arg "Unknown HazardTrackingMode"

    let to_ulong = function
      | Default -> Unsigned.ULong.zero
      | Untracked -> Unsigned.ULong.one
      | Tracked -> Unsigned.ULong.of_int 2
  end

  let get_cpu_cache_mode (self : t) : CPUCacheMode.t =
    let mode_val =
      msg_send ~self ~select:"cpuCacheMode" ~typ:(returning ulong)
    in
    CPUCacheMode.from_ulong mode_val

  let get_storage_mode (self : t) : StorageMode.t =
    let mode_val =
      msg_send ~self ~select:"storageMode" ~typ:(returning ulong)
    in
    StorageMode.from_ulong mode_val

  let get_hazard_tracking_mode (self : t) : HazardTrackingMode.t =
    let mode_val =
      msg_send ~self ~select:"hazardTrackingMode" ~typ:(returning ulong)
    in
    HazardTrackingMode.from_ulong mode_val

  let get_resource_options (self : t) : ResourceOptions.t =
    msg_send ~self ~select:"resourceOptions" ~typ:(returning ulong)

  let get_heap (self : t) =
    (* Returns id, needs Heap module *)
    msg_send ~self ~select:"heap" ~typ:(returning Objc.id)

  let get_heap_offset (self : t) : int =
    let offset = msg_send ~self ~select:"heapOffset" ~typ:(returning ulong) in
    Unsigned.ULong.to_int offset

  let get_allocated_size (self : t) : int =
    let size = msg_send ~self ~select:"allocatedSize" ~typ:(returning ulong) in
    Unsigned.ULong.to_int size

  let make_aliasable (self : t) : unit =
    msg_send ~self ~select:"makeAliasable" ~typ:(returning void)

  let is_aliasable (self : t) : bool =
    msg_send ~self ~select:"isAliasable" ~typ:(returning bool)
end

module Buffer = struct
  type t = payload

  (* Buffers are Resources *)

  let super b = b.id

  let on_device (device : Device.t) ~length options : t =
    let select = "newBufferWithLength:options:" in
    let id =
      msg_send ~self:device ~select
        ~typ:(ulong @-> ulong @-> returning Objc.id)
        (Unsigned.ULong.of_int length)
        options
    in
    { id = gc ~select id; lifetime = Lifetime () }

  let on_device_with_bytes (device : Device.t) ~bytes ~length options : t =
    let select = "newBufferWithBytes:length:options:" in
    let id =
      msg_send ~self:device ~select
        ~typ:(ptr void @-> ulong @-> ulong @-> returning Objc.id)
        bytes
        (Unsigned.ULong.of_int length)
        options
    in
    { id = gc ~select id; lifetime = Lifetime () }

  let on_device_with_bytes_no_copy (device : Device.t) ~bytes ~length
      ?deallocator options : t =
    let dealloc_block, callback =
      match deallocator with
      | None -> (null, None)
      | Some d ->
          let callback _block = d () in
          (Block.make ~args:[] ~return:Objc_type.void callback, Some callback)
    in
    let select = "newBufferWithBytesNoCopy:length:options:deallocator:" in
    let id =
      msg_send ~self:device ~select
        ~typ:(ptr void @-> ulong @-> ulong @-> ptr void @-> returning Objc.id)
        bytes
        (Unsigned.ULong.of_int length)
        options dealloc_block
    in
    { id = gc ~select id; lifetime = Lifetime callback }

  let length (self : t) : int =
    let len = msg_send ~self:self.id ~select:"length" ~typ:(returning ulong) in
    Unsigned.ULong.to_int len

  let contents (self : t) : unit ptr =
    msg_send ~self:self.id ~select:"contents" ~typ:(returning (ptr void))

  let did_modify_range (self : t) range =
    let ns_range = Range.to_value range in
    msg_send ~self:self.id ~select:"didModifyRange:"
      ~typ:(Range.nsrange_t @-> returning void)
      !@ns_range

  let add_debug_marker (self : t) ~marker range =
    let ns_marker = new_string marker in
    let ns_range = Range.to_value range in
    msg_send ~self:self.id ~select:"addDebugMarker:range:"
      ~typ:(Objc.id @-> Range.nsrange_t @-> returning void)
      ns_marker !@ns_range

  let remove_all_debug_markers (self : t) =
    msg_send ~self:self.id ~select:"removeAllDebugMarkers" ~typ:(returning void)

  let get_gpu_address (self : t) : Unsigned.ULLong.t =
    msg_send ~self:self.id ~select:"gpuAddress" ~typ:(returning ullong)

  (* Note: newTextureWithDescriptor is omitted as per user request (graphics focus) *)
  (* Note: remoteStorageBuffer/newRemoteBufferViewForDevice omitted for simplicity, can be added if needed *)
end

(* === Libraries and Functions === *)

module FunctionType = struct
  type t = Vertex | Fragment | Kernel | Visible | Intersection | Mesh | Object

  let from_ulong i =
    match Unsigned.ULong.to_int i with
    | 1 -> Vertex
    | 2 -> Fragment
    | 3 -> Kernel
    | 5 -> Visible
    | 6 -> Intersection
    | 7 -> Mesh
    | 8 -> Object
    | _ -> invalid_arg "Unknown FunctionType"

  let to_ulong = function
    | Vertex -> Unsigned.ULong.one
    | Fragment -> Unsigned.ULong.of_int 2
    | Kernel -> Unsigned.ULong.of_int 3
    | Visible -> Unsigned.ULong.of_int 5
    | Intersection -> Unsigned.ULong.of_int 6
    | Mesh -> Unsigned.ULong.of_int 7
    | Object -> Unsigned.ULong.of_int 8
end

module Function = struct
  type t = Objc.object_t

  let set_label (self : t) label = Resource.set_label self label
  let get_label (self : t) = Resource.get_label self
  let get_device (self : t) = Resource.get_device self

  let get_function_type (self : t) : FunctionType.t =
    let ft = msg_send ~self ~select:"functionType" ~typ:(returning ulong) in
    FunctionType.from_ulong ft

  (* Skipping patchType, patchControlPointCount, vertexAttributes,
     stageInputAttributes as they are graphics/tessellation related *)

  let get_name (self : t) : string =
    let ns_name = msg_send ~self ~select:"name" ~typ:(returning Objc.id) in
    ocaml_string_from_nsstring ns_name

  (* Skipping functionConstantsDictionary for brevity, can add if needed *)

  let get_options (self : t) : Unsigned.ULLong.t =
    (* MTLFunctionOptions is NSUInteger *)
    msg_send ~self ~select:"options" ~typ:(returning ullong)

  (* Skipping newArgumentEncoderWithBufferIndex for now, as ArgumentEncoder
     itself is complex *)
end

module Library = struct
  type t = payload

  let set_label (self : t) label = Resource.set_label self.id label
  let get_label (self : t) = Resource.get_label self.id
  let get_device (self : t) = Resource.get_device self.id

  let on_device (device : Device.t) ~source options : t =
    let select = "newLibraryWithSource:options:error:" in
    let ns_source = new_string source in
    let err_ptr = allocate Objc.id nil in
    let lib =
      msg_send ~self:device ~select
        ~typ:(Objc.id @-> Objc.id @-> ptr Objc.id @-> returning Objc.id)
        ns_source options err_ptr
    in
    check_error select err_ptr;
    { id = gc ~select lib; lifetime = Lifetime () }

  let on_device_with_data (device : Device.t) data : t =
    let err_ptr = allocate Objc.id nil in
    let select = "newLibraryWithData:error:" in
    let lib =
      msg_send ~self:device ~select
        ~typ:(ptr void @-> ptr Objc.id @-> returning Objc.id)
          (* dispatch_data_t maps to ptr void? Check C *)
        (coerce (ptr void) (ptr void) data)
        (* Assuming data is already a ptr void representing dispatch_data_t *)
        err_ptr
    in
    check_error select err_ptr;
    { id = gc ~select lib; lifetime = Lifetime data }

  let new_function_with_name (self : t) name : Function.t =
    let select = "newFunctionWithName:" in
    let ns_name = new_string name in
    let func =
      msg_send ~self:self.id ~select
        ~typ:(Objc.id @-> returning Objc.id)
        ns_name
    in
    gc ~select func

  (* Skipping newFunctionWithName:constantValues variants for brevity *)
  (* Skipping newFunctionWithDescriptor variants for brevity *)
  (* Skipping newIntersectionFunctionWithDescriptor variants for brevity *)

  let get_function_names (self : t) : string array =
    let ns_array =
      msg_send ~self:self.id ~select:"functionNames" ~typ:(returning Objc.id)
    in
    let id_array = from_nsarray ns_array in
    Array.map ocaml_string_from_nsstring id_array

  let get_library_type (self : t) : CompileOptions.LibraryType.t =
    let lt_val = msg_send ~self:self.id ~select:"type" ~typ:(returning ulong) in
    lt_val (* It's already the correct type *)

  let get_install_name (self : t) : string option =
    let ns_name =
      msg_send ~self:self.id ~select:"installName" ~typ:(returning Objc.id)
    in
    if is_nil ns_name then None else Some (ocaml_string_from_nsstring ns_name)
end

(* === Compute Pipeline === *)

module ComputePipelineDescriptor = struct
  type t = Objc.object_t

  let create () = new_gc ~class_name:"MTLComputePipelineDescriptor"
  let set_label (self : t) label = Resource.set_label self label
  let get_label (self : t) = Resource.get_label self

  let set_compute_function (self : t) (func : Function.t) =
    msg_send ~self ~select:"setComputeFunction:"
      ~typ:(Objc.id @-> returning void)
      func

  let get_compute_function (self : t) : Function.t =
    msg_send ~self ~select:"computeFunction" ~typ:(returning Objc.id)

  let set_support_indirect_command_buffers (self : t) support =
    msg_send ~self ~select:"setSupportIndirectCommandBuffers:"
      ~typ:(bool @-> returning void)
      support

  let get_support_indirect_command_buffers (self : t) : bool =
    msg_send ~self ~select:"supportIndirectCommandBuffers" ~typ:(returning bool)

  (* Skipping buffers, stageInputDescriptor, dynamic libraries, binary archives,
     linkedFunctions etc for brevity *)
end

module ComputePipelineState = struct
  type t = Objc.object_t

  let on_device_with_function (device : Device.t)
      ?(options = PipelineOption.none) ?(reflection = false) func : t * _ =
    let select =
      "newComputePipelineStateWithFunction:options:reflection:error:"
    in
    let err_ptr = allocate Objc.id nil in
    let maybe_reflection_ptr =
      if reflection then allocate Objc.id nil else nil_ptr
    in
    let pso =
      msg_send ~self:device ~select
        ~typ:
          (Objc.id @-> ulong @-> ptr Objc.id @-> ptr Objc.id
         @-> returning Objc.id)
        func options maybe_reflection_ptr err_ptr
    in
    check_error select err_ptr;
    (gc ~select pso, maybe_reflection_ptr)

  let on_device_with_descriptor (device : Device.t)
      ?(options = PipelineOption.none) ?(reflection = false) desc : t * _ =
    let select =
      "newComputePipelineStateWithDescriptor:options:reflection:error:"
    in
    let err_ptr = allocate Objc.id nil in
    let maybe_reflection_ptr =
      if reflection then allocate Objc.id nil else nil_ptr
    in
    let pso =
      msg_send ~self:device ~select
        ~typ:
          (Objc.id @-> ulong @-> ptr Objc.id @-> ptr Objc.id
         @-> returning Objc.id)
        desc options maybe_reflection_ptr err_ptr
    in
    check_error select err_ptr;
    ignore (Sys.opaque_identity desc);
    (gc ~select pso, maybe_reflection_ptr)

  let get_label (self : t) = Resource.get_label self
  let get_device (self : t) = Resource.get_device self

  let get_max_total_threads_per_threadgroup (self : t) : int =
    let count =
      msg_send ~self ~select:"maxTotalThreadsPerThreadgroup"
        ~typ:(returning ulong)
    in
    Unsigned.ULong.to_int count

  let get_thread_execution_width (self : t) : int =
    let width =
      msg_send ~self ~select:"threadExecutionWidth" ~typ:(returning ulong)
    in
    Unsigned.ULong.to_int width

  let get_static_threadgroup_memory_length (self : t) : int =
    let length =
      msg_send ~self ~select:"staticThreadgroupMemoryLength"
        ~typ:(returning ulong)
    in
    Unsigned.ULong.to_int length

  let get_support_indirect_command_buffers (self : t) : bool =
    msg_send ~self ~select:"supportIndirectCommandBuffers" ~typ:(returning bool)
end

(* === Command Infrastructure === *)

module LogLevel = struct
  type t = Undefined | Debug | Info | Notice | Error | Fault

  let from_long i =
    let i64 = Signed.Long.to_int64 i in
    match Int64.to_int i64 with
    | 0 -> Undefined
    | 1 -> Debug
    | 2 -> Info
    | 3 -> Notice
    | 4 -> Error
    | 5 -> Fault
    | n -> invalid_arg ("Unknown LogLevel: " ^ string_of_int n)

  let to_long : t -> Signed.long = function
    | Undefined -> Signed.Long.zero
    | Debug -> Signed.Long.one
    | Info -> Signed.Long.of_int 2
    | Notice -> Signed.Long.of_int 3
    | Error -> Signed.Long.of_int 4
    | Fault -> Signed.Long.of_int 5
end

module LogStateDescriptor = struct
  type t = Objc.object_t

  let create () = new_gc ~class_name:"MTLLogStateDescriptor"

  let set_level (self : t) (level : LogLevel.t) : unit =
    msg_send ~self ~select:"setLevel:"
      ~typ:(long @-> returning void)
      (LogLevel.to_long level)

  let get_level (self : t) : LogLevel.t =
    let level_val = msg_send ~self ~select:"level" ~typ:(returning long) in
    LogLevel.from_long level_val

  let set_buffer_size (self : t) (size : int) : unit =
    msg_send ~self ~select:"setBufferSize:"
      ~typ:(long @-> returning void)
      (Signed.Long.of_int size)

  let get_buffer_size (self : t) : int =
    let size_val = msg_send ~self ~select:"bufferSize" ~typ:(returning long) in
    Signed.Long.to_int size_val
end

module LogState = struct
  type t = payload

  let on_device_with_descriptor (device : Device.t)
      (descriptor : LogStateDescriptor.t) : t =
    let select = "newLogStateWithDescriptor:error:" in
    let err_ptr = allocate Objc.id nil in
    let id =
      msg_send ~self:device ~select
        ~typ:(Objc.id @-> ptr Objc.id @-> returning Objc.id)
        descriptor err_ptr
    in
    check_error select err_ptr;
    { id = gc ~select id; lifetime = Lifetime descriptor }
  (* Keep descriptor alive *)

  (* Note: Handling the lifetime of the OCaml closure requires careful
     consideration. For now, this binding assumes the block might be
     copied/retained by Metal, or that the LogState object itself keeps the
     necessary context alive. If issues arise, the lifetime management here
     might need refinement. *)
  let add_log_handler (self : t)
      (handler :
        sub_system:string option ->
        category:string option ->
        level:LogLevel.t ->
        message:string ->
        unit) =
    let block_impl _block ns_sub_system ns_category log_level ns_message =
      let sub_system =
        if is_nil ns_sub_system then None
        else Some (ocaml_string_from_nsstring ns_sub_system)
      in
      let category =
        if is_nil ns_category then None
        else Some (ocaml_string_from_nsstring ns_category)
      in
      let level = LogLevel.from_long log_level in
      let message = ocaml_string_from_nsstring ns_message in
      handler ~sub_system ~category ~level ~message
    in
    (* Type: (nullable NSString*, nullable NSString*, MTLLogLevel, NSString* ) -> void *)
    (* MTLLogLevel is NSInteger, which maps to long *)
    let block_ptr =
      Block.make block_impl
        ~args:Objc_type.[ id; id; long; id ]
        ~return:Objc_type.void
    in
    (* Store the closure in the lifetime field to keep it alive *)
    self.lifetime <- Lifetime (self.lifetime, block_impl);
    msg_send ~self:self.id ~select:"addLogHandler:"
      ~typ:(ptr void @-> returning void)
      block_ptr
end

(* Add this module *)
module CommandQueueDescriptor = struct
  type t = payload

  let create () : t =
    {
      id = new_gc ~class_name:"MTLCommandQueueDescriptor";
      lifetime = Lifetime ();
    }

  let set_max_command_buffer_count (self : t) (count : int) : unit =
    msg_send ~self:self.id ~select:"setMaxCommandBufferCount:"
      ~typ:(ulong @-> returning void)
      (Unsigned.ULong.of_int count)

  let get_max_command_buffer_count (self : t) : int =
    let count_val =
      msg_send ~self:self.id ~select:"maxCommandBufferCount"
        ~typ:(returning ulong)
    in
    Unsigned.ULong.to_int count_val

  let set_log_state (self : t) (log_state : LogState.t option) : unit =
    let log_state_id =
      match log_state with
      | None -> nil
      | Some ls ->
          (* NOTE: This is defensive and might leak a bit of memory if the
             descriptor is reused. *)
          self.lifetime <- Lifetime (self.lifetime, ls.lifetime);
          ls.id
    in
    msg_send ~self:self.id ~select:"setLogState:"
      ~typ:(Objc.id @-> returning void)
      log_state_id

  (* Returns LogState.t option because the property is nullable *)
  let get_log_state (self : t) : LogState.t option =
    let log_state_id =
      msg_send ~self:self.id ~select:"logState" ~typ:(returning Objc.id)
    in
    (* NOTE: This is defensive and might leak a bit of memory if the log state
       is reused. *)
    if is_nil log_state_id then None
    else Some { id = log_state_id; lifetime = self.lifetime }
end

(* === Command Infrastructure === *)

module CommandQueue = struct
  type t = payload (* Changed from Objc.object_t to payload *)

  let on_device (device : Device.t) : t =
    let select = "newCommandQueue" in
    let queue_id = msg_send ~self:device ~select ~typ:(returning Objc.id) in
    { id = gc ~select queue_id; lifetime = Lifetime () }
  (* Provide an empty lifetime *)

  let on_device_with_max_buffer_count (device : Device.t) max_count : t =
    let select = "newCommandQueueWithMaxCommandBufferCount:" in
    let queue_id =
      msg_send ~self:device ~select
        ~typ:(ulong @-> returning Objc.id)
        (Unsigned.ULong.of_int max_count)
    in
    { id = gc ~select queue_id; lifetime = Lifetime () }
  (* Provide an empty lifetime *)

  let on_device_with_descriptor (device : Device.t)
      (descriptor : CommandQueueDescriptor.t) : t =
    let select = "newCommandQueueWithDescriptor:" in
    let queue_id =
      msg_send ~self:device ~select
        ~typ:(Objc.id @-> returning Objc.id)
        descriptor.id
    in
    (* The crucial change: The new CommandQueue.t OCaml value now holds onto the
       descriptor's lifetime. This descriptor's lifetime, in turn, holds onto
       the LogState's lifetime, which holds the OCaml closure and C block. *)
    { id = gc ~select queue_id; lifetime = descriptor.lifetime }

  let set_label (self : t) label = Resource.set_label self.id label
  let get_label (self : t) = Resource.get_label self.id
  let get_device (self : t) = Resource.get_device self.id

  (* insertDebugCaptureBoundary is deprecated *)

  (* Access t.id for device *)
end

module CommandBuffer = struct
  type t = payload

  let set_label (self : t) label = Resource.set_label self.id label
  let get_label (self : t) = Resource.get_label self.id
  let get_device (self : t) = Resource.get_device self.id

  let get_command_queue (self : t) : CommandQueue.t =
    (* Return type is still CommandQueue.t, which is now payload *)
    let cq_id =
      msg_send ~self:self.id ~select:"commandQueue" ~typ:(returning Objc.id)
    in
    (* If we need to return a CommandQueue.t (payload), we must also attach its
       lifetime. The command buffer's lifetime should ideally include the
       queue's lifetime if the queue was created with a descriptor that had
       important OCaml lifetimes. A simple way is to take the command buffer's
       own lifetime, assuming it would be transitively kept alive by whatever
       keeps the command buffer alive. However, the command queue is a distinct
       entity. A more robust way would be if the command buffer's lifetime
       itself included a reference to the OCaml queue payload it was created
       from.

       For now, let's assume the caller manages the queue's lifetime separately,
       or the queue's lifetime is empty if created without a descriptor. This
       specific getter might need more thought if queues can be GC'd while
       buffers exist. Given our current problem, focusing on queue creation is
       key. Let's construct a new payload here, assuming the cq_id is what
       matters, and the lifetime should ideally be the one from the original
       queue. This requires CommandBuffer.on_queue to store the queue's
       payload. *)
    { id = cq_id; lifetime = self.lifetime }
  (* Simplification: use self's lifetime, or an empty one if more correct *)

  let get_retained_references (self : t) : bool =
    msg_send ~self:self.id ~select:"retainedReferences" ~typ:(returning bool)

  let on_queue (queue : CommandQueue.t) : t =
    (* queue is now payload *)
    let select = "commandBuffer" in
    let id = msg_send ~self:queue.id ~select ~typ:(returning Objc.id) in
    (* The command buffer should keep the queue's lifetime alive *)
    { id = gc ~select id; lifetime = queue.lifetime }

  let on_queue_with_unretained_references (queue : CommandQueue.t) : t =
    (* queue is now payload *)
    let select = "commandBufferWithUnretainedReferences" in
    let id = msg_send ~self:queue.id ~select ~typ:(returning Objc.id) in
    (* The command buffer should keep the queue's lifetime alive *)
    { id = gc ~select id; lifetime = queue.lifetime }

  (* command_buffer_with_descriptor requires MTLCommandBufferDescriptor,
     skipping for now *)

  let enqueue (self : t) =
    msg_send ~self:self.id ~select:"enqueue" ~typ:(returning void)

  let commit (self : t) =
    msg_send ~self:self.id ~select:"commit" ~typ:(returning void)

  let add_scheduled_handler (self : t) (handler : t -> unit) =
    (* The block takes one argument: the command buffer itself (id) *)
    let block_impl = fun _block _cmd_buf_arg -> handler self in
    self.lifetime <- Lifetime (self.lifetime, block_impl);
    (* args should list the types of arguments *after* the implicit _block *)
    let block_ptr =
      Objc_type.(Block.make block_impl ~args:[ id ] ~return:void)
    in
    msg_send ~self:self.id ~select:"addScheduledHandler:"
      ~typ:(ptr void @-> returning void) (* Pass the block pointer directly *)
      block_ptr

  let add_completed_handler (self : t) (handler : t -> unit) =
    (* The block takes one argument: the command buffer itself (id) *)
    let block_impl = fun _block _cmd_buf_arg -> handler self in
    self.lifetime <- Lifetime (self.lifetime, block_impl);
    (* args should list the types of arguments *after* the implicit _block *)
    let block_ptr =
      Objc_type.(Block.make block_impl ~args:[ id ] ~return:void)
    in
    msg_send ~self:self.id ~select:"addCompletedHandler:"
      ~typ:(ptr void @-> returning void) (* Pass the block pointer directly *)
      block_ptr

  let wait_until_scheduled (self : t) =
    msg_send_suspended ~self:self.id ~select:"waitUntilScheduled"
      ~typ:(returning void)

  let wait_until_completed (self : t) =
    msg_send_suspended ~self:self.id ~select:"waitUntilCompleted"
      ~typ:(returning void)

  module Status = struct
    type t =
      | NotEnqueued
      | Enqueued
      | Committed
      | Scheduled
      | Completed
      | Error

    let from_ulong i =
      match Unsigned.ULong.to_int i with
      | 0 -> NotEnqueued
      | 1 -> Enqueued
      | 2 -> Committed
      | 3 -> Scheduled
      | 4 -> Completed
      | 5 -> Error
      | _ -> invalid_arg "Unknown CommandBufferStatus"

    let to_ulong = function
      | NotEnqueued -> Unsigned.ULong.zero
      | Enqueued -> Unsigned.ULong.one
      | Committed -> Unsigned.ULong.of_int 2
      | Scheduled -> Unsigned.ULong.of_int 3
      | Completed -> Unsigned.ULong.of_int 4
      | Error -> Unsigned.ULong.of_int 5
  end

  let get_status (self : t) : Status.t =
    let status_val =
      msg_send ~self:self.id ~select:"status" ~typ:(returning ulong)
    in
    Status.from_ulong status_val

  let get_error (self : t) : string option =
    (* NSError *)
    let err = msg_send ~self:self.id ~select:"error" ~typ:(returning Objc.id) in
    if is_nil err then None else Some (get_error_description err)

  let get_gpu_start_time (self : t) : float =
    msg_send ~self:self.id ~select:"GPUStartTime" ~typ:(returning double)

  let get_gpu_end_time (self : t) : float =
    msg_send ~self:self.id ~select:"GPUEndTime" ~typ:(returning double)

  let encode_wait_for_event self event value =
    msg_send ~self:self.id ~select:"encodeWaitForEvent:value:"
      ~typ:(Objc.id @-> ullong @-> returning void)
      event value

  let encode_signal_event self event value =
    msg_send ~self:self.id ~select:"encodeSignalEvent:value:"
      ~typ:(Objc.id @-> ullong @-> returning void)
      event value

  let push_debug_group (self : t) group_name =
    let ns_group = new_string group_name in
    msg_send ~self:self.id ~select:"pushDebugGroup:"
      ~typ:(Objc.id @-> returning void)
      ns_group

  let pop_debug_group (self : t) =
    msg_send ~self:self.id ~select:"popDebugGroup" ~typ:(returning void)

  (* Skipping renderCommandEncoder, parallelRenderCommandEncoder,
     resourceStateCommandEncoder, accelerationStructureCommandEncoder *)
end

module CommandEncoder = struct
  type t = Objc.object_t

  let set_label (self : t) label = Resource.set_label self label
  let get_label (self : t) = Resource.get_label self
  let get_device (self : t) = Resource.get_device self

  let end_encoding (self : t) =
    msg_send ~self ~select:"endEncoding" ~typ:(returning void)

  let insert_debug_signpost (self : t) signpost =
    let ns_signpost = new_string signpost in
    msg_send ~self ~select:"insertDebugSignpost:"
      ~typ:(Objc.id @-> returning void)
      ns_signpost

  let push_debug_group (self : t) group_name =
    let ns_group = new_string group_name in
    msg_send ~self ~select:"pushDebugGroup:"
      ~typ:(Objc.id @-> returning void)
      ns_group

  let pop_debug_group (self : t) =
    msg_send ~self ~select:"popDebugGroup" ~typ:(returning void)
end

module ResourceUsage = struct
  type t = Unsigned.ULong.t (* MTLResourceUsage is NSUInteger *)

  let read = Unsigned.ULong.of_int 1 (* 1 << 0 *)
  let write = Unsigned.ULong.of_int 2 (* 1 << 1 *)
  let ( + ) = Unsigned.ULong.logor
end

module ComputeCommandEncoder = struct
  type t = Objc.object_t

  let set_label (self : t) label = CommandEncoder.set_label self label
  let get_label (self : t) = CommandEncoder.get_label self
  let get_device (self : t) = CommandEncoder.get_device self

  let on_buffer (self : CommandBuffer.t) : t =
    (* Returns ComputeCommandEncoder.t *)
    let select = "computeCommandEncoder" in
    let encoder = msg_send ~self:self.id ~select ~typ:(returning Objc.id) in
    gc ~select encoder

  module DispatchType = struct
    type t = Serial | Concurrent

    let to_ulong = function
      | Serial -> Unsigned.ULong.zero
      | Concurrent -> Unsigned.ULong.one
  end

  let on_buffer_with_dispatch_type (self : CommandBuffer.t) dispatch_type : t =
    (* Returns ComputeCommandEncoder.t *)
    let select = "computeCommandEncoderWithDispatchType:" in
    let encoder =
      msg_send ~self:self.id ~select
        ~typ:(ulong @-> returning Objc.id)
        (DispatchType.to_ulong dispatch_type)
    in
    gc ~select encoder

  let end_encoding (self : t) = CommandEncoder.end_encoding self

  let insert_debug_signpost (self : t) signpost =
    CommandEncoder.insert_debug_signpost self signpost

  let push_debug_group (self : t) group_name =
    CommandEncoder.push_debug_group self group_name

  let pop_debug_group (self : t) = CommandEncoder.pop_debug_group self

  let set_compute_pipeline_state (self : t) (pso : ComputePipelineState.t) =
    msg_send ~self ~select:"setComputePipelineState:"
      ~typ:(Objc.id @-> returning void)
      pso

  let set_buffer (self : t) ?(offset = 0) ~index buffer =
    msg_send ~self ~select:"setBuffer:offset:atIndex:"
      ~typ:(Objc.id @-> ulong @-> ulong @-> returning void)
      buffer.id
      (Unsigned.ULong.of_int offset)
      (Unsigned.ULong.of_int index)

  let set_buffers (self : t) ~offsets ~index (buffers : Buffer.t list) =
    let buffer_ids = List.map (fun b -> b.id) buffers in
    let ns_array_buffers = to_nsarray buffer_ids in
    let lifetime, offsets_ptr =
      let offsets_array =
        CArray.(of_list ulong (List.map Unsigned.ULong.of_int offsets))
      in
      (offsets_array, CArray.(start offsets_array))
    in
    let range = Range.make ~location:index ~length:(List.length buffers) in
    msg_send ~self ~select:"setBuffers:offsets:withRange:"
      ~typ:(Objc.id @-> ptr ulong @-> Range.nsrange_t @-> returning void)
      ns_array_buffers offsets_ptr !@range;
    ignore (Sys.opaque_identity lifetime)

  let set_bytes (self : t) ~bytes ~length ~index =
    msg_send ~self ~select:"setBytes:length:atIndex:"
      ~typ:(ptr void @-> ulong @-> ulong @-> returning void)
      bytes
      (Unsigned.ULong.of_int length)
      (Unsigned.ULong.of_int index)

  let set_threadgroup_memory_length (self : t) ~length ~index =
    msg_send ~self ~select:"setThreadgroupMemoryLength:atIndex:"
      ~typ:(ulong @-> ulong @-> returning void)
      (Unsigned.ULong.of_int length)
      (Unsigned.ULong.of_int index)

  let dispatch_threadgroups (self : t) ~threadgroups_per_grid
      ~threads_per_threadgroup =
    let grid_size_val = Size.to_value threadgroups_per_grid in
    let group_size_val = Size.to_value threads_per_threadgroup in
    msg_send ~self ~select:"dispatchThreadgroups:threadsPerThreadgroup:"
      ~typ:(Size.mtlsize_t @-> Size.mtlsize_t @-> returning void)
      !@grid_size_val !@group_size_val

  let use_resource (self : t) resource usage =
    msg_send ~self ~select:"useResource:usage:"
      ~typ:(Objc.id @-> ulong @-> returning void)
      resource usage

  let use_resources (self : t) resources usage =
    let count = List.length resources in
    let ns_array = to_nsarray ~count resources in
    msg_send ~self ~select:"useResources:count:usage:"
      ~typ:(Objc.id @-> ulong @-> ulong @-> returning void)
      ns_array
      (Unsigned.ULong.of_int count)
      usage

  let execute_commands_in_buffer (self : t) buffer range =
    let ns_range = Range.to_value range in
    msg_send ~self ~select:"executeCommandsInBuffer:withRange:"
      ~typ:(Objc.id @-> Range.nsrange_t @-> returning void)
      buffer !@ns_range

  (* Skipping many other methods (textures, samplers, barriers, events, etc.)
     for brevity, add as needed *)
end

module BlitCommandEncoder = struct
  type t = Objc.object_t

  let set_label (self : t) label = CommandEncoder.set_label self label
  let get_label (self : t) = CommandEncoder.get_label self
  let get_device (self : t) = CommandEncoder.get_device self

  let on_buffer (cmdbuf : CommandBuffer.t) : t =
    (* Correct argument type to CommandBuffer.t *)
    (* Returns BlitCommandEncoder.t *)
    let select = "blitCommandEncoder" in
    let encoder = msg_send ~self:cmdbuf.id ~select ~typ:(returning Objc.id) in
    gc ~select encoder

  let end_encoding (self : t) = CommandEncoder.end_encoding self

  let insert_debug_signpost (self : t) signpost =
    CommandEncoder.insert_debug_signpost self signpost

  let push_debug_group (self : t) group_name =
    CommandEncoder.push_debug_group self group_name

  let pop_debug_group (self : t) = CommandEncoder.pop_debug_group self

  let copy_from_buffer (self : t) ~(source_buffer : Buffer.t) ~source_offset
      ~(destination_buffer : Buffer.t) ~destination_offset ~size =
    msg_send ~self
      ~select:"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"
      ~typ:(Objc.id @-> ulong @-> Objc.id @-> ulong @-> ulong @-> returning void)
      source_buffer.id
      (Unsigned.ULong.of_int source_offset)
      destination_buffer.id
      (Unsigned.ULong.of_int destination_offset)
      (Unsigned.ULong.of_int size)

  let fill_buffer (self : t) (buffer : Buffer.t) range ~value =
    if value < 0 || value > 255 then failwith "Invalid value for fill_buffer";
    let ns_range = Range.to_value range in
    msg_send ~self ~select:"fillBuffer:range:value:"
      ~typ:(Objc.id @-> Range.nsrange_t @-> uchar @-> returning void)
      buffer.id !@ns_range
      (Unsigned.UChar.of_int value);
    ignore (Sys.opaque_identity ns_range)

  let synchronize_resource (self : t) resource =
    msg_send ~self ~select:"synchronizeResource:"
      ~typ:(Objc.id @-> returning void)
      resource

  let update_fence (self : t) fence =
    msg_send ~self ~select:"updateFence:"
      ~typ:(Objc.id @-> returning void)
      fence

  let wait_for_fence (self : t) fence =
    msg_send ~self ~select:"waitForFence:"
      ~typ:(Objc.id @-> returning void)
      fence

  (* Skipping texture copies for now, add if needed *)
  (* Skipping generateMipmaps, optimizeContents, indirect command buffer ops *)
end

(* === Synchronization === *)

module Event = struct
  type t = Objc.object_t

  let get_device (self : t) : Device.t = Resource.get_device self
  let set_label (self : t) label = Resource.set_label self label
  let get_label (self : t) = Resource.get_label self
end

module SharedEvent = struct
  type t = payload

  module SharedEventListener = struct
    type t = Objc.object_t

    let init () = new_gc ~class_name:"MTLSharedEventListener"
  end

  module SharedEventHandle = struct
    type t = Objc.object_t

    let get_label self =
      let ns_label = msg_send ~self ~select:"label" ~typ:(returning Objc.id) in
      if is_nil ns_label then None
      else Some (ocaml_string_from_nsstring ns_label)
  end

  let super e = e.id

  let on_device (device : Device.t) : t =
    let select = "newSharedEvent" in
    let id = msg_send ~self:device ~select ~typ:(returning Objc.id) in
    { id = gc ~select id; lifetime = Lifetime () }

  let get_device (self : t) : Device.t =
    Resource.get_device self.id (* SharedEvent inherits from Event *)

  let set_label (self : t) label = Resource.set_label self.id label
  let get_label (self : t) = Resource.get_label self.id

  let set_signaled_value (self : t) value =
    msg_send ~self:self.id ~select:"setSignaledValue:"
      ~typ:(ullong @-> returning void)
      value

  let get_signaled_value (self : t) : Unsigned.ULLong.t =
    msg_send ~self:self.id ~select:"signaledValue" ~typ:(returning ullong)

  let notify_listener (self : t) (listener : SharedEventListener.t) ~value
      (handler : t -> Unsigned.ULLong.t -> unit) =
    let block_impl =
     fun _block _event_arg value_arg -> handler self value_arg
    in
    self.lifetime <- Lifetime (self.lifetime, block_impl);
    let block_ptr =
      Objc_type.(Block.make block_impl ~args:[ id; ullong ] ~return:void)
    in
    msg_send ~self:self.id ~select:"notifyListener:atValue:block:"
      ~typ:(Objc.id @-> ullong @-> ptr void @-> returning void)
      listener value block_ptr

  let new_shared_event_handle (self : t) : SharedEventHandle.t =
    let select = "newSharedEventHandle" in
    let handle = msg_send ~self:self.id ~select ~typ:(returning Objc.id) in
    gc ~select handle

  let wait_until_signaled_value (self : t) ~value ~timeout_ms : bool =
    msg_send ~self:self.id ~select:"waitUntilSignaledValue:timeoutMS:"
      ~typ:(ullong @-> ullong @-> returning bool)
      value timeout_ms
end

module Fence = struct
  type t = Objc.object_t

  let on_device (device : Device.t) : t =
    let select = "newFence" in
    let fence = msg_send ~self:device ~select ~typ:(returning Objc.id) in
    gc ~select fence

  let get_device (self : t) : Device.t = Resource.get_device self
  let set_label (self : t) label = Resource.set_label self label
  let get_label (self : t) = Resource.get_label self
end

(* === Indirect Command Buffers === *)

module IndirectCommandType = struct
  type t = Unsigned.ULong.t (* MTLIndirectCommandType is NSUInteger *)

  let draw = Unsigned.ULong.of_int 1 (* 1 << 0 *)
  let draw_indexed = Unsigned.ULong.of_int 2 (* 1 << 1 *)
  let draw_patches = Unsigned.ULong.of_int 4 (* 1 << 2, macOS only *)
  let draw_indexed_patches = Unsigned.ULong.of_int 8 (* 1 << 3, macOS only *)
  let concurrent_dispatch = Unsigned.ULong.of_int 32 (* 1 << 5 *)
  let concurrent_dispatch_threads = Unsigned.ULong.of_int 64 (* 1 << 6 *)
  let ( + ) = Unsigned.ULong.logor
end

module IndirectCommandBufferDescriptor = struct
  type t = Objc.object_t

  let create () = new_gc ~class_name:"MTLIndirectCommandBufferDescriptor"

  let set_command_types (self : t) types =
    msg_send ~self ~select:"setCommandTypes:"
      ~typ:(ulong @-> returning void)
      types

  let get_command_types (self : t) : IndirectCommandType.t =
    msg_send ~self ~select:"commandTypes" ~typ:(returning ulong)

  let set_inherit_pipeline_state (self : t) inherit_pso =
    msg_send ~self ~select:"setInheritPipelineState:"
      ~typ:(bool @-> returning void)
      inherit_pso

  let get_inherit_pipeline_state (self : t) : bool =
    msg_send ~self ~select:"inheritPipelineState" ~typ:(returning bool)

  let set_inherit_buffers (self : t) inherit_buffers =
    msg_send ~self ~select:"setInheritBuffers:"
      ~typ:(bool @-> returning void)
      inherit_buffers

  let get_inherit_buffers (self : t) : bool =
    msg_send ~self ~select:"inheritBuffers" ~typ:(returning bool)

  let set_max_kernel_buffer_bind_count (self : t) count =
    msg_send ~self ~select:"setMaxKernelBufferBindCount:"
      ~typ:(ulong @-> returning void)
      (Unsigned.ULong.of_int count)

  let get_max_kernel_buffer_bind_count (self : t) : int =
    let count =
      msg_send ~self ~select:"maxKernelBufferBindCount" ~typ:(returning ulong)
    in
    Unsigned.ULong.to_int count

  (* Skipping maxVertexBufferBindCount, maxFragmentBufferBindCount as they are
     graphics related *)
end

module IndirectComputeCommand = struct
  type t = Objc.object_t

  let set_compute_pipeline_state (self : t) (pso : ComputePipelineState.t) =
    msg_send ~self ~select:"setComputePipelineState:"
      ~typ:(Objc.id @-> returning void)
      pso

  let set_kernel_buffer (self : t) ?(offset = 0) ~index buffer =
    msg_send ~self ~select:"setKernelBuffer:offset:atIndex:"
      ~typ:(Objc.id @-> ulong @-> ulong @-> returning void)
      buffer.id
      (Unsigned.ULong.of_int offset)
      (Unsigned.ULong.of_int index)

  let concurrent_dispatch_threadgroups (self : t) ~threadgroups_per_grid
      ~threads_per_threadgroup =
    let grid_size_val = Size.to_value threadgroups_per_grid in
    let group_size_val = Size.to_value threads_per_threadgroup in
    msg_send ~self
      ~select:"concurrentDispatchThreadgroups:threadsPerThreadgroup:"
      ~typ:(Size.mtlsize_t @-> Size.mtlsize_t @-> returning void)
      !@grid_size_val !@group_size_val

  let set_barrier (self : t) =
    msg_send ~self ~select:"setBarrier" ~typ:(returning void)

  (* Skipping setKernelBytes, setThreadgroupMemoryLength, setStageInRegion,
     reset *)
end

module IndirectCommandBuffer = struct
  type t = Objc.object_t

  let on_device_with_descriptor (device : Device.t) descriptor
      ~max_command_count ~options : t =
    let select =
      "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:"
    in
    let icb =
      msg_send ~self:device ~select
        ~typ:(Objc.id @-> ulong @-> ulong @-> returning Objc.id)
        descriptor
        (Unsigned.ULong.of_int max_command_count)
        options
    in
    gc ~select icb

  let get_size (self : t) : int =
    let size = msg_send ~self ~select:"size" ~typ:(returning ulong) in
    Unsigned.ULong.to_int size

  let indirect_compute_command_at_index (self : t) index :
      IndirectComputeCommand.t =
    let select = "indirectComputeCommandAtIndex:" in
    let cmd =
      msg_send ~self ~select
        ~typ:(ulong @-> returning Objc.id)
        (Unsigned.ULong.of_int index)
    in
    gc ~select cmd

  (* Skipping indirect_render_command_at_index *)

  let reset_with_range (self : t) range =
    let ns_range = Range.to_value range in
    msg_send ~self ~select:"resetWithRange:"
      ~typ:(Range.nsrange_t @-> returning void)
      !@ns_range
end

(* === Dynamic Library Placeholder === *)
(* Binding MTLDynamicLibrary requires more setup (compile options, linking) *)
module DynamicLibrary = struct
  type t = Objc.object_t

  (* Add bindings here if needed *)
end
