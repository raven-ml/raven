module Missing = struct
  type 'a t = { x : 'a }
end

type 'a t = { sub : 'a Missing.t } [@@deriving ptree]
