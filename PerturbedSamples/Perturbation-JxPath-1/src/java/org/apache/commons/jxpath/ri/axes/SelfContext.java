[REPLACE]^private boolean startedSet ;^33^^^^^^^[REPLACE] private boolean startedSet = false;^ [CLASS] SelfContext  
[REPLACE]^super.reset (  ) ;^37^^^^^36^39^[REPLACE] super ( parentContext ) ;^[METHOD] <init> [TYPE] NodeTest) [PARAMETER] EvalContext parentContext NodeTest nodeTest [CLASS] SelfContext   [TYPE]  EvalContext parentContext  [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[REPLACE]^this.nodeTest =  null;^38^^^^^36^39^[REPLACE] this.nodeTest = nodeTest;^[METHOD] <init> [TYPE] NodeTest) [PARAMETER] EvalContext parentContext NodeTest nodeTest [CLASS] SelfContext   [TYPE]  EvalContext parentContext  [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[ADD]^^38^^^^^36^39^[ADD] this.nodeTest = nodeTest;^[METHOD] <init> [TYPE] NodeTest) [PARAMETER] EvalContext parentContext NodeTest nodeTest [CLASS] SelfContext   [TYPE]  EvalContext parentContext  [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[REPLACE]^return parentContext .getCurrentNodePointer (  )  ;^42^^^^^41^43^[REPLACE] return parentContext.getSingleNodePointer (  ) ;^[METHOD] getSingleNodePointer [TYPE] Pointer [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[REPLACE]^if  ( position  ||  0 )  {^46^^^^^45^52^[REPLACE] if  ( position == 0 )  {^[METHOD] getCurrentNodePointer [TYPE] NodePointer [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[ADD]^^46^47^48^49^^45^52^[ADD] if  ( position == 0 )  { if  ( !setPosition ( 1 )  )  { return null; }^[METHOD] getCurrentNodePointer [TYPE] NodePointer [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[REPLACE]^if  ( !setPosition ( 2 )  )  {^47^^^^^45^52^[REPLACE] if  ( !setPosition ( 1 )  )  {^[METHOD] getCurrentNodePointer [TYPE] NodePointer [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[ADD]^^47^48^49^^^45^52^[ADD] if  ( !setPosition ( 1 )  )  { return null; }^[METHOD] getCurrentNodePointer [TYPE] NodePointer [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[REPLACE]^return true;^48^^^^^45^52^[REPLACE] return null;^[METHOD] getCurrentNodePointer [TYPE] NodePointer [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[REPLACE]^return false;^48^^^^^45^52^[REPLACE] return null;^[METHOD] getCurrentNodePointer [TYPE] NodePointer [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[REPLACE]^if  ( !setPosition ( 1 / 3 )  )  {^47^^^^^45^52^[REPLACE] if  ( !setPosition ( 1 )  )  {^[METHOD] getCurrentNodePointer [TYPE] NodePointer [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[ADD]^return null;^47^48^49^^^45^52^[ADD] if  ( !setPosition ( 1 )  )  { return null; }^[METHOD] getCurrentNodePointer [TYPE] NodePointer [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[REPLACE]^return this;^48^^^^^45^52^[REPLACE] return null;^[METHOD] getCurrentNodePointer [TYPE] NodePointer [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[REPLACE]^return null;^51^^^^^45^52^[REPLACE] return nodePointer;^[METHOD] getCurrentNodePointer [TYPE] NodePointer [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[REPLACE]^return setPosition ( getCurrentPosition (  )   ;^55^^^^^54^56^[REPLACE] return setPosition ( getCurrentPosition (  )  + 1 ) ;^[METHOD] nextNode [TYPE] boolean [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[REPLACE]^super.getCurrentPosition (  ) ;^59^^^^^58^61^[REPLACE] super.reset (  ) ;^[METHOD] reset [TYPE] void [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[ADD]^^59^60^^^^58^61^[ADD] super.reset (  ) ; startedSet = false;^[METHOD] reset [TYPE] void [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[REPLACE]^startedSet = true;^60^^^^^58^61^[REPLACE] startedSet = false;^[METHOD] reset [TYPE] void [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[ADD]^^60^^^^^58^61^[ADD] startedSet = false;^[METHOD] reset [TYPE] void [PARAMETER] [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest 
[REPLACE]^if  ( position  ==  1 )  {^64^^^^^63^78^[REPLACE] if  ( position != 1 )  {^[METHOD] setPosition [TYPE] boolean [PARAMETER] int position [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest  [TYPE]  int position 
[REPLACE]^return true;^65^^^^^63^78^[REPLACE] return false;^[METHOD] setPosition [TYPE] boolean [PARAMETER] int position [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest  [TYPE]  int position 
[REPLACE]^super.reset (  ) ;^67^^^^^63^78^[REPLACE] super.setPosition ( position ) ;^[METHOD] setPosition [TYPE] boolean [PARAMETER] int position [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest  [TYPE]  int position 
[REPLACE]^if  ( startedSet )  {^68^^^^^63^78^[REPLACE] if  ( !startedSet )  {^[METHOD] setPosition [TYPE] boolean [PARAMETER] int position [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest  [TYPE]  int position 
[ADD]^startedSet = true;nodePointer =  ( NodePointer )  parentContext.getCurrentNodePointer (  ) ;^68^69^70^71^^63^78^[ADD] if  ( !startedSet )  { startedSet = true; nodePointer =  ( NodePointer )  parentContext.getCurrentNodePointer (  ) ; }^[METHOD] setPosition [TYPE] boolean [PARAMETER] int position [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest  [TYPE]  int position 
[REPLACE]^startedSet = false;^69^^^^^63^78^[REPLACE] startedSet = true;^[METHOD] setPosition [TYPE] boolean [PARAMETER] int position [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest  [TYPE]  int position 
[REPLACE]^nodePointer ;^70^^^^^63^78^[REPLACE] nodePointer =  ( NodePointer )  parentContext.getCurrentNodePointer (  ) ;^[METHOD] setPosition [TYPE] boolean [PARAMETER] int position [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest  [TYPE]  int position 
[REPLACE]^nodePointer  =  nodePointer ;^70^^^^^63^78^[REPLACE] nodePointer =  ( NodePointer )  parentContext.getCurrentNodePointer (  ) ;^[METHOD] setPosition [TYPE] boolean [PARAMETER] int position [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest  [TYPE]  int position 
[ADD]^^70^^^^^63^78^[ADD] nodePointer =  ( NodePointer )  parentContext.getCurrentNodePointer (  ) ;^[METHOD] setPosition [TYPE] boolean [PARAMETER] int position [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest  [TYPE]  int position 
[REPLACE]^if  ( nodePointer != null )  {^73^^^^^63^78^[REPLACE] if  ( nodePointer == null )  {^[METHOD] setPosition [TYPE] boolean [PARAMETER] int position [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest  [TYPE]  int position 
[REMOVE]^if  ( ! ( setPosition ( 1 )  )  )  {     return null; }^73^^^^^63^78^[REMOVE] ^[METHOD] setPosition [TYPE] boolean [PARAMETER] int position [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest  [TYPE]  int position 
[REPLACE]^return true;^74^^^^^63^78^[REPLACE] return false;^[METHOD] setPosition [TYPE] boolean [PARAMETER] int position [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest  [TYPE]  int position 
[REPLACE]^return true ;^77^^^^^63^78^[REPLACE] return nodeTest == null || nodePointer.testNode ( nodeTest ) ;^[METHOD] setPosition [TYPE] boolean [PARAMETER] int position [CLASS] SelfContext   [TYPE]  boolean false  startedSet  true  [TYPE]  NodePointer nodePointer  [TYPE]  NodeTest nodeTest  [TYPE]  int position 