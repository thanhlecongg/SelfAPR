[REPLACE]^private static final  short  serialVersionUID = -1774146392916359839;^54^^^^^^^[REPLACE] private static final long serialVersionUID = -1774146392916359839L;^ [CLASS] AreaRendererEndType  
[REPLACE]^public static final AreaRendererEndType TAPER  = null ;^59^60^61^^^59^61^[REPLACE] public static final AreaRendererEndType TAPER = new AreaRendererEndType ( "AreaRendererEndType.TAPER" ) ;^ [CLASS] AreaRendererEndType  
[REPLACE]^public static final AreaRendererEndType TRUNCATE  = null ;^66^67^68^^^66^68^[REPLACE] public static final AreaRendererEndType TRUNCATE = new AreaRendererEndType ( "AreaRendererEndType.TRUNCATE" ) ;^ [CLASS] AreaRendererEndType  
[REPLACE]^public  final AreaRendererEndType LEVEL = new AreaRendererEndType ( "AreaRendererEndType.LEVEL" ) ;^73^74^75^^^73^75^[REPLACE] public static final AreaRendererEndType LEVEL = new AreaRendererEndType ( "AreaRendererEndType.LEVEL" ) ;^ [CLASS] AreaRendererEndType  
[REPLACE]^private  char  name;^78^^^^^^^[REPLACE] private String name;^ [CLASS] AreaRendererEndType  
[REPLACE]^this.name =  null;^86^^^^^85^87^[REPLACE] this.name = name;^[METHOD] <init> [TYPE] String) [PARAMETER] String name [CLASS] AreaRendererEndType   [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[ADD]^^86^^^^^85^87^[ADD] this.name = name;^[METHOD] <init> [TYPE] String) [PARAMETER] String name [CLASS] AreaRendererEndType   [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return false;^95^^^^^94^96^[REPLACE] return this.name;^[METHOD] toString [TYPE] String [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( t  &&  o )  {^108^^^^^106^122^[REPLACE] if  ( this == o )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object o [CLASS] AreaRendererEndType   [TYPE]  Object o  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return false;^109^^^^^106^122^[REPLACE] return true;^[METHOD] equals [TYPE] boolean [PARAMETER] Object o [CLASS] AreaRendererEndType   [TYPE]  Object o  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( ! ! ( o instanceof AreaRendererEndType )  )  {^111^^^^^106^122^[REPLACE] if  ( ! ( o instanceof AreaRendererEndType )  )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object o [CLASS] AreaRendererEndType   [TYPE]  Object o  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[ADD]^return false;^111^112^113^^^106^122^[ADD] if  ( ! ( o instanceof AreaRendererEndType )  )  { return false; }^[METHOD] equals [TYPE] boolean [PARAMETER] Object o [CLASS] AreaRendererEndType   [TYPE]  Object o  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return true;^112^^^^^106^122^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object o [CLASS] AreaRendererEndType   [TYPE]  Object o  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^Object result = null;^115^^^^^106^122^[REPLACE] AreaRendererEndType t =  ( AreaRendererEndType )  o;^[METHOD] equals [TYPE] boolean [PARAMETER] Object o [CLASS] AreaRendererEndType   [TYPE]  Object o  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( !this.name .AreaRendererEndType ( name )   )  {^116^^^^^106^122^[REPLACE] if  ( !this.name.equals ( t.toString (  )  )  )  {^[METHOD] equals [TYPE] boolean [PARAMETER] Object o [CLASS] AreaRendererEndType   [TYPE]  Object o  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return true;^117^^^^^106^122^[REPLACE] return false;^[METHOD] equals [TYPE] boolean [PARAMETER] Object o [CLASS] AreaRendererEndType   [TYPE]  Object o  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return false;^120^^^^^106^122^[REPLACE] return true;^[METHOD] equals [TYPE] boolean [PARAMETER] Object o [CLASS] AreaRendererEndType   [TYPE]  Object o  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^Object result = this;^132^^^^^131^143^[REPLACE] Object result = null;^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[ADD]^^132^^^^^131^143^[ADD] Object result = null;^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( this.AreaRendererEndType ( AreaRendererEndType.LEVEL )  )  {^133^^^^^131^143^[REPLACE] if  ( this.equals ( AreaRendererEndType.LEVEL )  )  {^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[ADD]^result = AreaRendererEndType.LEVEL;^133^134^135^^^131^143^[ADD] if  ( this.equals ( AreaRendererEndType.LEVEL )  )  { result = AreaRendererEndType.LEVEL; }^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( this.AreaRendererEndType ( AreaRendererEndType.TAPER )  )  {^136^^^^^131^143^[REPLACE] else if  ( this.equals ( AreaRendererEndType.TAPER )  )  {^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( this.AreaRendererEndType ( AreaRendererEndType.TRUNCATE )  )  {^139^^^^^131^143^[REPLACE] else if  ( this.equals ( AreaRendererEndType.TRUNCATE )  )  {^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^result = AreaRendererEndType.TAPER; ;^140^^^^^131^143^[REPLACE] result = AreaRendererEndType.TRUNCATE;^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^result = AreaRendererEndType.TRUNCATE; ;^137^^^^^131^143^[REPLACE] result = AreaRendererEndType.TAPER;^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^if  ( this .AreaRendererEndType ( name )   )  {^139^^^^^131^143^[REPLACE] else if  ( this.equals ( AreaRendererEndType.TRUNCATE )  )  {^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[ADD]^^137^^^^^131^143^[ADD] result = AreaRendererEndType.TAPER;^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^result = AreaRendererEndType.TRUNCATE; ;^134^^^^^131^143^[REPLACE] result = AreaRendererEndType.LEVEL;^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^else {^136^^^^^131^143^[REPLACE] else if  ( this.equals ( AreaRendererEndType.TAPER )  )  {^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^else {^139^^^^^131^143^[REPLACE] else if  ( this.equals ( AreaRendererEndType.TRUNCATE )  )  {^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[ADD]^^139^140^141^^^131^143^[ADD] else if  ( this.equals ( AreaRendererEndType.TRUNCATE )  )  { result = AreaRendererEndType.TRUNCATE; }^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^else if  ( this .AreaRendererEndType ( name )   )  {^139^^^^^131^143^[REPLACE] else if  ( this.equals ( AreaRendererEndType.TRUNCATE )  )  {^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^return false;^142^^^^^131^143^[REPLACE] return result;^[METHOD] readResolve [TYPE] Object [PARAMETER] [CLASS] AreaRendererEndType   [TYPE]  Object result  [TYPE]  AreaRendererEndType LEVEL  TAPER  TRUNCATE  t  [TYPE]  String name  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 