open Angstrom
open Base
open Ast

let is_digit = function
  | '0' .. '9' -> true
  | _ -> false


let is_letter = function
  | 'a' .. 'z' | 'A' .. 'Z' -> true
  | _ -> false


let ws =
  let ws =
    skip_while (function
      | '\x20' | '\x0a' | '\x0d' | '\x09' -> true
      | _ -> false)
  in
  let comment =
    let start = string "#" in
    start *> many (not_char '\n')
  in
  ws *> fix (fun this -> comment *> ws *> this <|> ws)


let ident =
  let is_id c = is_digit c || is_letter c || Char.equal c '_' in
  let* ident = ws *> take_while1 is_id <* ws in
  if is_digit @@ String.get ident 0
  then fail (Printf.sprintf "invalid name %s" ident)
  else return ident


let chainl1 e op =
  let rec go acc = lift2 (fun f x -> f acc x) op e >>= go <|> return acc in
  e >>= fun init -> go init


let lchar c = ws *> char c
let rchar c = char c <* ws

let sep_by_comma p =
  let comma_sep = ws *> char ',' <* ws in
  sep_by comma_sep p


let parens p = rchar '(' *> p <* lchar ')'

let func =
  let* proto =
    let* name = string "def" *> ws *> ident <* ws in
    let* args = parens (sep_by_comma ident) in
    return (Prototype (name, args))
  in
  let* body =
    let expr =
      fix (fun expr ->
        let var_decl =
          (* var a<2> = [[1], [2]] *)
          let* name = string "var " *> ws *> ident <* ws in
          let* shape =
            let integer = take_while1 is_digit >>| fun s -> Int.of_string s in
            option [] (char '<' *> sep_by_comma integer <* char '>') <* ws
          in
          let* init = string "=" *> ws *> expr in
          return (VarDecl (name, Array.of_list shape, init))
        in
        let print =
          (* print([1, 2]) *)
          let* _ = string "print" in
          let* expr = expr in
          return (Print expr)
        in
        let ret =
          (* return [1, 2] | return([1, 2]) | return; *)
          let* expr =
            string "return"
            *> (ws *> peek_char
                >>= (function
                       | Some ';' -> return None
                       | _ -> fail "non void")
                <|> (char ' ' <|> return ' ') *> (expr >>| Option.some))
          in
          return (Return expr)
        in
        let num =
          let* n =
            let* s = take_while1 (fun c -> is_digit c || Char.equal c '.') in
            try return (Float.of_string s) with
            | _ -> failwith ("wrong number: " ^ s)
          in
          return (Num n)
        in
        let literal =
          let* lit_without_shapes =
            let f p = rchar '[' *> sep_by_comma p <* lchar ']' in
            fix (fun self ->
              (let* exprs = f self <|> f num in
               return (Literal ([||], exprs)))
              <|> num)
          in
          let rec init_shapes = function
            | Literal (_, exs) ->
              let exs =
                List.fold_right exs ~init:[] ~f:(fun e acc -> init_shapes e :: acc)
              in
              let nest_shape =
                match List.hd_exn exs with
                | Literal (shape, _) -> shape
                | _ -> [||]
              in
              Literal (Array.append [| List.length exs |] nest_shape, exs)
            | Num n -> Num n
            | _ -> assert false
          in
          return (init_shapes lit_without_shapes)
        in
        let ident_exp =
          let* ident = ident in
          return (Var ident)
        in
        let call =
          let* callee = ident in
          let* args = parens (sep_by_comma expr) in
          return (Call (callee, args))
        in
        let bin_op =
          let op s = char s *> return (fun lhs rhs -> BinOp (s, lhs, rhs)) in
          fix (fun ex ->
            let add = op '+' in
            let mul = op '*' in
            let factor = parens ex <|> choice [ call; literal; ident_exp ] in
            let term = chainl1 factor (ws *> mul <* ws) in
            chainl1 term (ws *> add <* ws))
        in
        choice [ var_decl; print; ret; bin_op ])
      <* ws
    in
    let exprs = many1 (ws *> expr <* lchar ';') in
    lchar '{' *> exprs <* lchar '}' <* ws
  in
  return (Function (proto, body))


let parse str =
  match Angstrom.parse_string ~consume:Consume.All (many1 (ws *> func)) str with
  | Result.Ok res -> res
  | Error msg -> failwith msg
