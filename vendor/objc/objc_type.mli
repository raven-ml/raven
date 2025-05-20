(** Represents an Objective-C type. *)
type _ t =
  | Id : C.Types.object_t t
  | Class : C.Types.object_t t
  | Sel : C.Types.selector_t t
  | Void : unit t
  | Str : string t
  | Char : char t
  | Bool : bool t
  | Int : int t
  | Short : int t
  | Long : Signed.long t
  | LLong : Signed.llong t
  | ULLong : Unsigned.ullong t
  | Float : float t
  | Double : float t
  | Unknown : unit Ctypes.ptr t
  | Ptr : 'a t -> 'a Ctypes.ptr t
  | Arr : 'a t -> 'a Ctypes.carray t
  | Struc : 'a Ctypes.structure Ctypes.typ -> 'a Ctypes.structure t
  | Union : 'a t -> 'a Ctypes.union t
  | Imp : unit Ctypes.ptr t
  | Enc : C.Types._Enc t
  | Proto : C.Types.protocol_t t
  | Ivar : C.Types.ivar_t t

(** Represents a tuple of Objective-C types. *)
type (_, _) tlist =
  | [] : ('r, 'r) tlist
  | ( :: ) : 'a t * ('b, 'r) tlist -> ('a -> 'b, 'r) tlist

val method_typ : args:('a, 'b) tlist -> 'b t -> 'a Ctypes.fn
(** Returns the ctypes fn typ corresponding to the Objective-C method type. *)

val value_typ : 'a t -> 'a Ctypes.typ
(** Returns the ctypes value typ corresponding to the Objective-C value type. *)

val encode_method : args:('a, 'b) tlist -> 'c t -> string
(** Returns the encoding of a method type in the Objective-C runtime. *)

val encode_value : 'a t -> string
(** Returns the encoding of a value type in the Objective-C runtime. *)

(* Convenience constructors *)

val noargs : ('a, 'a) tlist
(** The method accepts only the implicit [self] and [cmd] arguments. *)

val id : C.Types.object_t t
(** A reference to any Objective-C object regardless of its class. *)

val _Class : C.Types.object_t t
val _SEL : C.Types.selector_t t
val _IMP : unit Ctypes.ptr t
val _Enc : C.Types._Enc t
val _Protocol : C.Types.protocol_t t
val _Ivar : C.Types.ivar_t t
val void : unit t
val string : string t
val char : char t
val bool : bool t
val int : int t
val short : int t
val long : Signed.long t
val llong : Signed.llong t
val ullong : Unsigned.ullong t
val float : float t
val double : float t
val unknown : unit Ctypes.ptr t
val ptr : 'a t -> 'a Ctypes.ptr t
val array : 'a t -> 'a Ctypes.carray t
val struc : 'a Ctypes.structure Ctypes.typ -> 'a Ctypes.structure t
val union : 'a t -> 'a Ctypes.union t
