[buglab_swap_variables]^if  ( nonempty != context && context != empty )  {^276^^^^^273^286^if  ( context != nonempty && context != empty )  {^[CLASS] JsonWriter  [METHOD] close [RETURN_TYPE] JsonWriter   JsonScope empty JsonScope nonempty String closeBracket [VARIABLES] List  stack  boolean  htmlSafe  lenient  String  closeBracket  indent  openBracket  separator  JsonScope  context  empty  nonempty  Writer  out  
[buglab_swap_variables]^if  ( context != empty && context != nonempty )  {^276^^^^^273^286^if  ( context != nonempty && context != empty )  {^[CLASS] JsonWriter  [METHOD] close [RETURN_TYPE] JsonWriter   JsonScope empty JsonScope nonempty String closeBracket [VARIABLES] List  stack  boolean  htmlSafe  lenient  String  closeBracket  indent  openBracket  separator  JsonScope  context  empty  nonempty  Writer  out  
[buglab_swap_variables]^if  ( nonempty == context )  {^281^^^^^273^286^if  ( context == nonempty )  {^[CLASS] JsonWriter  [METHOD] close [RETURN_TYPE] JsonWriter   JsonScope empty JsonScope nonempty String closeBracket [VARIABLES] List  stack  boolean  htmlSafe  lenient  String  closeBracket  indent  openBracket  separator  JsonScope  context  empty  nonempty  Writer  out  
[buglab_swap_variables]^stack.set ( topOfStack.size (  )  - 1, stack ) ;^299^^^^^298^300^stack.set ( stack.size (  )  - 1, topOfStack ) ;^[CLASS] JsonWriter  [METHOD] replaceTop [RETURN_TYPE] void   JsonScope topOfStack [VARIABLES] List  stack  boolean  htmlSafe  lenient  String  closeBracket  indent  openBracket  separator  JsonScope  topOfStack  Writer  out  
[buglab_swap_variables]^stack.set ( stack.size (  )  - 1 ) ;^299^^^^^298^300^stack.set ( stack.size (  )  - 1, topOfStack ) ;^[CLASS] JsonWriter  [METHOD] replaceTop [RETURN_TYPE] void   JsonScope topOfStack [VARIABLES] List  stack  boolean  htmlSafe  lenient  String  closeBracket  indent  openBracket  separator  JsonScope  topOfStack  Writer  out  
[buglab_swap_variables]^for  ( lengthnt i = 0, i = value.length (  ) ; i < length; i++ )  {^426^^^^^411^441^for  ( int i = 0, length = value.length (  ) ; i < length; i++ )  {^[CLASS] JsonWriter  [METHOD] string [RETURN_TYPE] void   String value [VARIABLES] char  c  List  stack  boolean  htmlSafe  lenient  value  String  closeBracket  indent  name  openBracket  separator  string  value  Writer  out  int  i  length  
[buglab_swap_variables]^for  ( int i = 0 = value.length (  ) ; i < length; i++ )  {^426^^^^^411^441^for  ( int i = 0, length = value.length (  ) ; i < length; i++ )  {^[CLASS] JsonWriter  [METHOD] string [RETURN_TYPE] void   String value [VARIABLES] char  c  List  stack  boolean  htmlSafe  lenient  value  String  closeBracket  indent  name  openBracket  separator  string  value  Writer  out  int  i  length  
[buglab_swap_variables]^char c = i.charAt ( value ) ;^427^^^^^412^442^char c = value.charAt ( i ) ;^[CLASS] JsonWriter  [METHOD] string [RETURN_TYPE] void   String value [VARIABLES] char  c  List  stack  boolean  htmlSafe  lenient  value  String  closeBracket  indent  name  openBracket  separator  string  value  Writer  out  int  i  length  
[buglab_swap_variables]^} else if  ( JsonScope.EMPTY_OBJECT != context )  {^505^^^^^501^510^} else if  ( context != JsonScope.EMPTY_OBJECT )  {^[CLASS] JsonWriter  [METHOD] beforeName [RETURN_TYPE] void   [VARIABLES] List  stack  boolean  htmlSafe  lenient  value  String  closeBracket  indent  name  openBracket  separator  string  value  JsonScope  context  Writer  out  