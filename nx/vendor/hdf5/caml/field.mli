module Unpacked : sig
  type 'a t = private {
    name    : string;
    type_   : 'a Type.Unpacked.t;
    default : 'a option;
  }
end

type t = T : _ Unpacked.t -> t

val create : string -> 'a Type.Unpacked.t -> 'a option -> t
