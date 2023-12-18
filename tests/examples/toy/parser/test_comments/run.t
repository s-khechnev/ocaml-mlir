  $ cat code.toy
  # def func(a){
  #   return a; 
  # }
  # 
  # def main(){
  #   var a = 1;
  #   var b = [1];
  #   var c = func(a);
  #   var d = [1] + b;
  #   var e = [1] + d;
  #   var f<2, 3> = [[1, 2], [3, 4], [5, 6]];
  #   print(f);
  #   return;
  # }
  #
  #
  #
  #
  
  def func(a)   {
    return a;
        }
  
  def 
  #comment
  main(
  #comment
  )     
  #comment  
  {
    var    a       =        1           
    ;
    var b #comment
    = [ #comment
       1         ];
    var c = func(  a   );
    var d = [1] #comment
                + #
                b   ;
    var e = [1] #comment
                + #
                d   ;
    var f<2, #comment
            3> = [[1, 2], [3, 4], [5, 6]];
    print(  #qwe
      f   );
    return    ;
  }
  
  $ dune exec -- toy -emit ast -f code.toy
  [(Ast.Function ((Ast.Prototype ("func", ["a"])),
      [(Ast.Return (Some (Ast.Var "a")))]));
    (Ast.Function ((Ast.Prototype ("main", [])),
       [(Ast.VarDecl ("a", [||], (Ast.Num 1.)));
         (Ast.VarDecl ("b", [||], (Ast.Literal ([|1|], [(Ast.Num 1.)]))));
         (Ast.VarDecl ("c", [||], (Ast.Call ("func", [(Ast.Var "a")]))));
         (Ast.VarDecl ("d", [||],
            (Ast.BinOp (`Add, (Ast.Literal ([|1|], [(Ast.Num 1.)])),
               (Ast.Var "b")))
            ));
         (Ast.VarDecl ("e", [||],
            (Ast.BinOp (`Add, (Ast.Literal ([|1|], [(Ast.Num 1.)])),
               (Ast.Var "d")))
            ));
         (Ast.VarDecl ("f", [|2; 3|],
            (Ast.Literal ([|3; 2|],
               [(Ast.Literal ([|2|], [(Ast.Num 1.); (Ast.Num 2.)]));
                 (Ast.Literal ([|2|], [(Ast.Num 3.); (Ast.Num 4.)]));
                 (Ast.Literal ([|2|], [(Ast.Num 5.); (Ast.Num 6.)]))]
               ))
            ));
         (Ast.Print (Ast.Var "f")); (Ast.Return None)]
       ))
    ]
