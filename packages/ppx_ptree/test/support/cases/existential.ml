type ('a, 'd) t = Pack : 'b * 'a -> ('a, 'd) t [@@deriving ptree]
