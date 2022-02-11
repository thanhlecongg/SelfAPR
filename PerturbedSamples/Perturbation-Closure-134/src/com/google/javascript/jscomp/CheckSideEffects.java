[REPLACE]^static final DiagnosticType USELESS_CODE_ERROR  = null ;^39^40^41^^^39^41^[REPLACE] static final DiagnosticType USELESS_CODE_ERROR = DiagnosticType.warning ( "JSC_USELESS_CODE", "Suspicious code. {0}" ) ;^ [CLASS] CheckSideEffects  
[REPLACE]^private  CheckLevel level;^43^^^^^^^[REPLACE] private final CheckLevel level;^ [CLASS] CheckSideEffects  
[REPLACE]^this.level =  null;^46^^^^^45^47^[REPLACE] this.level = level;^[METHOD] <init> [TYPE] CheckLevel) [PARAMETER] CheckLevel level [CLASS] CheckSideEffects   [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  boolean false  true 
[REPLACE]^if  ( n.getType (  )  == Token.EMPTY ) {^54^55^^^^49^105^[REPLACE] if  ( n.getType (  )  == Token.EMPTY || n.getType (  )  == Token.COMMA )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[ADD]^^54^55^56^57^^49^105^[ADD] if  ( n.getType (  )  == Token.EMPTY || n.getType (  )  == Token.COMMA )  { return; }^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  ( parent !=n ) return;^59^60^^^^49^105^[REPLACE] if  ( parent == null ) return;^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[ADD]^return;int pt = parent.getType (  ) ;^59^60^^62^^49^105^[ADD] if  ( parent == null ) return; int pt = parent.getType (  ) ;^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^int pt = n.getLastChild (  ) ;^62^^^^^49^105^[REPLACE] int pt = parent.getType (  ) ;^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  ( pt  &&  Token.COMMA )  {^63^^^^^49^105^[REPLACE] if  ( pt == Token.COMMA )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^}  if  ( pt != Token.EXPR_RESULT || pt != Token.BLOCK )  {^76^^^^^49^105^[REPLACE] } else if  ( pt != Token.EXPR_RESULT && pt != Token.BLOCK )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  ( pt == Token.FOR ) {^77^78^79^^^49^105^[REPLACE] if  ( pt == Token.FOR && parent.getChildCount (  )  == 4 && ( n == parent.getFirstChild (  )  || n == parent.getFirstChild (  ) .getNext (  ) .getNext (  )  )  )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[ADD]^^77^78^79^^^49^105^[ADD] if  ( pt == Token.FOR && parent.getChildCount (  )  == 4 && ( n == parent.getFirstChild (  )  || n == parent.getFirstChild (  ) .getNext (  ) .getNext (  )  )  )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  ( pt == Token.FOR || parent.getChildCount (  )  == 0 || ( n == parent.getLastChild (  )  || n == parent.getFirstChild (  ) .getNext (  ) .getNext (  )  )  )  {^77^78^79^^^49^105^[REPLACE] if  ( pt == Token.FOR && parent.getChildCount (  )  == 4 && ( n == parent.getFirstChild (  )  || n == parent.getFirstChild (  ) .getNext (  ) .getNext (  )  )  )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  ( n  || parent (  )  )  {^64^^^^^49^105^[REPLACE] if  ( n == parent.getLastChild (  )  )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[ADD]^^64^65^66^67^68^49^105^[ADD] if  ( n == parent.getLastChild (  )  )  { for  ( Node an : parent.getAncestors (  )  )  { int ancestorType = an.getType (  ) ; if  ( ancestorType == Token.COMMA ) continue;^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  ( ancestorType  !=  Token.COMMA ) continue;^67^68^^^^49^105^[REPLACE] if  ( ancestorType == Token.COMMA ) continue;^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  ( pt != Token.EXPR_RESULT || ancestorType != Token.BLOCK ) return;^69^70^71^^^49^105^[REPLACE] if  ( ancestorType != Token.EXPR_RESULT && ancestorType != Token.BLOCK ) return;^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[ADD]^^69^70^71^72^73^49^105^[ADD] if  ( ancestorType != Token.EXPR_RESULT && ancestorType != Token.BLOCK ) return; else break;^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^int pt = parent.getType (  ) ;^65^^^^^49^105^[REPLACE] for  ( Node an : parent.getAncestors (  )  )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^int parentcestorType = an.getType (  ) ;^66^^^^^49^105^[REPLACE] int ancestorType = an.getType (  ) ;^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  ( ancestorType  &&  Token.COMMA ) continue;^67^68^^^^49^105^[REPLACE] if  ( ancestorType == Token.COMMA ) continue;^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  ( ancestorType != Token.EXPR_RESULT || ancestorType != Token.BLOCK ) return;^69^70^71^^^49^105^[REPLACE] if  ( ancestorType != Token.EXPR_RESULT && ancestorType != Token.BLOCK ) return;^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  (parent  &&  parent.getLastChild (  )  )  {^64^^^^^49^105^[REPLACE] if  ( n == parent.getLastChild (  )  )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  ( pt  &&  Token.COMMA ) continue;^67^68^^^^49^105^[REPLACE] if  ( ancestorType == Token.COMMA ) continue;^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^}  else {^76^^^^^49^105^[REPLACE] } else if  ( pt != Token.EXPR_RESULT && pt != Token.BLOCK )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[ADD]^}^76^77^78^79^^49^105^[ADD] else if  ( pt != Token.EXPR_RESULT && pt != Token.BLOCK )  { if  ( pt == Token.FOR && parent.getChildCount (  )  == 4 && ( n == parent.getFirstChild (  )  || n == parent.getFirstChild (  ) .getNext (  ) .getNext (  )  )  )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  (  parent.getChildCount (  )  == 4 || ( n == parent.getFirstChild (  )  || n == parent.getFirstChild (  ) .getNext (  ) .getNext (  )  )  )  {^77^78^79^^^49^105^[REPLACE] if  ( pt == Token.FOR && parent.getChildCount (  )  == 4 && ( n == parent.getFirstChild (  )  || n == parent.getFirstChild (  ) .getNext (  ) .getNext (  )  )  )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  ( NodeUtil.isSimpleOperatorType ( n.getType (  )  )  ) {^86^87^^^^49^105^[REPLACE] if  ( NodeUtil.isSimpleOperatorType ( n.getType (  )  )  || !NodeUtil.mayHaveSideEffects ( n )  )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  ( n.isQualifiedName (  )  && n.getJSDocInfo (  )  == null )  {^88^^^^^86^104^[REPLACE] if  ( n.isQualifiedName (  )  && n.getJSDocInfo (  )  != null )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^}  else {^92^^^^^86^104^[REPLACE] } else if  ( NodeUtil.isExpressionNode ( n )  )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[ADD]^}   return;^92^93^94^95^^86^104^[ADD] else if  ( NodeUtil.isExpressionNode ( n )  )  {  return; }^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  ( n.getType (  )   ||  Token.STRING )  {^98^^^^^86^104^[REPLACE] if  ( n.getType (  )  == Token.STRING )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^msg  = null ;^99^^^^^86^104^[REPLACE] msg = "Is there a missing '+' on the previous line?";^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^msg ;^99^^^^^86^104^[REPLACE] msg = "Is there a missing '+' on the previous line?";^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^int pt = parent.getType (  ) ;^97^^^^^86^104^[REPLACE] String msg = "This code lacks side-effects. Is there a bug?";^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  (  n.getJSDocInfo (  )  == null )  {^88^^^^^49^105^[REPLACE] if  ( n.isQualifiedName (  )  && n.getJSDocInfo (  )  != null )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^} else if  ( NodeUtil.isExpressionNode ( parent )  )  {^92^^^^^49^105^[REPLACE] } else if  ( NodeUtil.isExpressionNode ( n )  )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^}  else {^92^^^^^49^105^[REPLACE] } else if  ( NodeUtil.isExpressionNode ( n )  )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^if  ( n.getType (  )   &&  Token.STRING )  {^98^^^^^49^105^[REPLACE] if  ( n.getType (  )  == Token.STRING )  {^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[ADD]^msg = "Is there a missing '+' on the previous line?";^98^99^100^^^49^105^[ADD] if  ( n.getType (  )  == Token.STRING )  { msg = "Is there a missing '+' on the previous line?"; }^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^msg ;^99^^^^^49^105^[REPLACE] msg = "Is there a missing '+' on the previous line?";^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[REPLACE]^int pt = parent.getType (  ) ;^97^^^^^49^105^[REPLACE] String msg = "This code lacks side-effects. Is there a bug?";^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 
[ADD]^^97^^^^^49^105^[ADD] String msg = "This code lacks side-effects. Is there a bug?";^[METHOD] visit [TYPE] void [PARAMETER] NodeTraversal t Node n Node parent [CLASS] CheckSideEffects   [TYPE]  boolean false  true  [TYPE]  NodeTraversal t  [TYPE]  DiagnosticType USELESS_CODE_ERROR  [TYPE]  CheckLevel level  [TYPE]  Node an  n  parent  [TYPE]  String msg  [TYPE]  int ancestorType  pt 