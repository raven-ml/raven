module Id = struct
  type t = nativeint (* Represents any Objective-C object instance (id). *)

  let to_nativeint (x : t) : nativeint = x
  let of_nativeint (x : nativeint) : t = x
  let null : t = Nativeint.zero (* Represents Objective-C 'nil'. *)
  let is_null (x : t) : bool = x = Nativeint.zero
end

module Sel = struct
  type t = nativeint (* Represents an Objective-C selector (SEL). *)

  let to_nativeint (x : t) : nativeint = x
  let of_nativeint (x : nativeint) : t = x
  let null : t = Nativeint.zero
  let is_null (x : t) : bool = x = Nativeint.zero
end

module Class = struct
  type t = nativeint (* Represents an Objective-C class (Class). *)

  let to_nativeint (x : t) : nativeint = x
  let of_nativeint (x : nativeint) : t = x

  (* In Objective-C, a Class is also an id. These allow conversion. *)
  let to_id (x : t) : Id.t = Id.of_nativeint (to_nativeint x)

  (* Unsafe because it assumes the Id.t truly is a Class pointer. *)
  let from_id_unsafe (x : Id.t) : t = of_nativeint (Id.to_nativeint x)
  let null : t = Nativeint.zero
  let is_null (x : t) : bool = x = Nativeint.zero
end
