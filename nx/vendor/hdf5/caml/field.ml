module Unpacked = struct
  type 'a t = {
    name    : string;
    type_   : 'a Type.Unpacked.t;
    default : 'a option;
  }
end

type t = T : _ Unpacked.t -> t

let create name type_ default = T { name; type_; default }
