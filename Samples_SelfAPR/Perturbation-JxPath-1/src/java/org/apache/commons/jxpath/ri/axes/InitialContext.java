[P3_Replace_Literal]^private boolean startedSet = true;^32^^^^^27^37^private boolean startedSet = false;^[CLASS] InitialContext   [VARIABLES] 
[P8_Replace_Mix]^private boolean startedSet  = null ;^32^^^^^27^37^private boolean startedSet = false;^[CLASS] InitialContext   [VARIABLES] 
[P3_Replace_Literal]^private boolean started = true;^33^^^^^28^38^private boolean started = false;^[CLASS] InitialContext   [VARIABLES] 
[P8_Replace_Mix]^private boolean started ;^33^^^^^28^38^private boolean started = false;^[CLASS] InitialContext   [VARIABLES] 
[P14_Delete_Statement]^^38^^^^^37^45^super ( parentContext ) ;^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P8_Replace_Mix]^nodePointer =  ( NodePointer )  null.getCurrentNodePointer (  ) .clone (  ) ;^39^40^^^^37^45^nodePointer = ( NodePointer )  parentContext.getCurrentNodePointer (  ) .clone (  ) ;^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P8_Replace_Mix]^( NodePointer )  parentContext .getCurrentNodePointer (  )  .clone (  ) ;^40^^^^^37^45^( NodePointer )  parentContext.getCurrentNodePointer (  ) .clone (  ) ;^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P14_Delete_Statement]^^40^^^^^37^45^( NodePointer )  parentContext.getCurrentNodePointer (  ) .clone (  ) ;^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P2_Replace_Operator]^if  ( nodePointer == null )  {^41^^^^^37^45^if  ( nodePointer != null )  {^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P6_Replace_Expression]^if  ( nodePointer.getIndex() == WHOLE_COLLECTION )  {^41^^^^^37^45^if  ( nodePointer != null )  {^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P8_Replace_Mix]^if  ( nodePointer != true )  {^41^^^^^37^45^if  ( nodePointer != null )  {^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P15_Unwrap_Block]^collection = (nodePointer.getIndex()) == (org.apache.commons.jxpath.ri.model.NodePointer.WHOLE_COLLECTION);^41^42^43^44^^37^45^if  ( nodePointer != null )  { collection = ( nodePointer.getIndex (  )  == NodePointer.WHOLE_COLLECTION ) ; }^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P16_Remove_Block]^^41^42^43^44^^37^45^if  ( nodePointer != null )  { collection = ( nodePointer.getIndex (  )  == NodePointer.WHOLE_COLLECTION ) ; }^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P2_Replace_Operator]^collection = ( nodePointer.getIndex (  )  != NodePointer.WHOLE_COLLECTION ) ;^42^43^^^^37^45^collection = ( nodePointer.getIndex (  )  == NodePointer.WHOLE_COLLECTION ) ;^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P7_Replace_Invocation]^collection = ( nodePointer.getLength (  )  == NodePointer.WHOLE_COLLECTION ) ;^42^43^^^^37^45^collection = ( nodePointer.getIndex (  )  == NodePointer.WHOLE_COLLECTION ) ;^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P8_Replace_Mix]^collection  =  collection ;^42^43^^^^37^45^collection = ( nodePointer.getIndex (  )  == NodePointer.WHOLE_COLLECTION ) ;^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P13_Insert_Block]^if  (  ( nodePointer )  != null )  {     collection =  ( nodePointer.getIndex (  )  )  ==  ( WHOLE_COLLECTION ) ; }^42^^^^^37^45^[Delete]^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P7_Replace_Invocation]^( nodePointer.getLength (  )  == NodePointer.WHOLE_COLLECTION ) ;^43^^^^^37^45^( nodePointer.getIndex (  )  == NodePointer.WHOLE_COLLECTION ) ;^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P14_Delete_Statement]^^43^^^^^37^45^( nodePointer.getIndex (  )  == NodePointer.WHOLE_COLLECTION ) ;^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P2_Replace_Operator]^collection = ( nodePointer.getIndex (  )  < NodePointer.WHOLE_COLLECTION ) ;^42^43^^^^37^45^collection = ( nodePointer.getIndex (  )  == NodePointer.WHOLE_COLLECTION ) ;^[CLASS] InitialContext  [METHOD] <init> [RETURN_TYPE] EvalContext)   EvalContext parentContext [VARIABLES] EvalContext  parentContext  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P7_Replace_Invocation]^return nodePointer.getIndex (  ) ;^56^^^^^55^57^return nodePointer.getValue (  ) ;^[CLASS] InitialContext  [METHOD] getValue [RETURN_TYPE] Object   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[P14_Delete_Statement]^^56^^^^^55^57^return nodePointer.getValue (  ) ;^[CLASS] InitialContext  [METHOD] getValue [RETURN_TYPE] Object   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[P2_Replace_Operator]^return setPosition ( position  &&  1 ) ;^60^^^^^59^61^return setPosition ( position + 1 ) ;^[CLASS] InitialContext  [METHOD] nextNode [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[P3_Replace_Literal]^return setPosition ( position  ) ;^60^^^^^59^61^return setPosition ( position + 1 ) ;^[CLASS] InitialContext  [METHOD] nextNode [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[P7_Replace_Invocation]^return InitialContext ( position + 1 ) ;^60^^^^^59^61^return setPosition ( position + 1 ) ;^[CLASS] InitialContext  [METHOD] nextNode [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[P8_Replace_Mix]^return InitialContext ( position  &&  1 ) ;^60^^^^^59^61^return setPosition ( position + 1 ) ;^[CLASS] InitialContext  [METHOD] nextNode [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[P2_Replace_Operator]^return setPosition ( position  ||  1 ) ;^60^^^^^59^61^return setPosition ( position + 1 ) ;^[CLASS] InitialContext  [METHOD] nextNode [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[P3_Replace_Literal]^return setPosition ( position + null ) ;^60^^^^^59^61^return setPosition ( position + 1 ) ;^[CLASS] InitialContext  [METHOD] nextNode [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[P14_Delete_Statement]^^60^^^^^59^61^return setPosition ( position + 1 ) ;^[CLASS] InitialContext  [METHOD] nextNode [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[P8_Replace_Mix]^this.position =  null;^64^^^^^63^75^this.position = position;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P5_Replace_Variable]^if  ( startedSet )  {^65^^^^^63^75^if  ( collection )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P6_Replace_Expression]^if  ( position >= 1 )  {^65^^^^^63^75^if  ( collection )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P6_Replace_Expression]^if  ( position <= nodePointer.getLength (  ) )  {^65^^^^^63^75^if  ( collection )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P6_Replace_Expression]^if  ( position - 1 )  {^65^^^^^63^75^if  ( collection )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P6_Replace_Expression]^if  ( position == 1 )  {^65^^^^^63^75^if  ( collection )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P15_Unwrap_Block]^if ((position >= 1) && (position <= (nodePointer.getLength()))) {    nodePointer.setIndex((position - 1));    return true;}; return false;^65^66^67^68^69^63^75^if  ( collection )  { if  ( position >= 1 && position <= nodePointer.getLength (  )  )  { nodePointer.setIndex ( position - 1 ) ; return true; }^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P16_Remove_Block]^^65^66^67^68^69^63^75^if  ( collection )  { if  ( position >= 1 && position <= nodePointer.getLength (  )  )  { nodePointer.setIndex ( position - 1 ) ; return true; }^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P13_Insert_Block]^if  (  ( position >= 1 )  &&  ( position <=  ( nodePointer.getLength (  )  )  )  )  {     nodePointer.setIndex (  ( position - 1 )  ) ;     return true; }^65^^^^^63^75^[Delete]^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P2_Replace_Operator]^return position != 1;^73^^^^^63^75^return position == 1;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P3_Replace_Literal]^return position == 4;^73^^^^^63^75^return position == 1;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P8_Replace_Mix]^return position  ||  1;^73^^^^^63^75^return position == 1;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P2_Replace_Operator]^if  ( position >= 1 || position <= nodePointer.getLength (  )  )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P2_Replace_Operator]^if  ( position > 1 && position <= nodePointer.getLength (  )  )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P2_Replace_Operator]^if  ( position >= 1 && position < nodePointer.getLength (  )  )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P3_Replace_Literal]^if  ( position >= position && position <= nodePointer.getLength (  )  )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P5_Replace_Variable]^if  ( nodePointer >= 1 && position <= position.getLength (  )  )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P6_Replace_Expression]^if  ( position >= 1 ) {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P6_Replace_Expression]^if  (  position <= nodePointer.getLength (  )  )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P6_Replace_Expression]^if  ( position - 1 )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P7_Replace_Invocation]^if  ( position >= 1 && position <= nodePointer.getIndex (  )  )  {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P8_Replace_Mix]^if  ( position >= 0 ) {^66^^^^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  {^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P15_Unwrap_Block]^nodePointer.setIndex((position - 1)); return true;^66^67^68^69^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  { nodePointer.setIndex ( position - 1 ) ; return true; }^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P16_Remove_Block]^^66^67^68^69^^63^75^if  ( position >= 1 && position <= nodePointer.getLength (  )  )  { nodePointer.setIndex ( position - 1 ) ; return true; }^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P13_Insert_Block]^if  ( collection )  {     if  (  ( position >= 1 )  &&  ( position <=  ( nodePointer.getLength (  )  )  )  )  {         nodePointer.setIndex (  ( position - 1 )  ) ;         return true;     }     return false; }else {     return position == 1; }^66^^^^^63^75^[Delete]^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P3_Replace_Literal]^return false;^68^^^^^63^75^return true;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P2_Replace_Operator]^nodePointer.setIndex ( position  <<  1 ) ;^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P3_Replace_Literal]^nodePointer.setIndex ( position  ) ;^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P7_Replace_Invocation]^nodePointer .getIndex (  )  ;^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P14_Delete_Statement]^^67^68^^^^63^75^nodePointer.setIndex ( position - 1 ) ; return true;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P11_Insert_Donor_Statement]^return setPosition ( position + 1 ) ;nodePointer.setIndex ( position - 1 ) ;^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P3_Replace_Literal]^return true;^70^^^^^63^75^return false;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P2_Replace_Operator]^nodePointer.setIndex ( position  >>  1 ) ;^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P3_Replace_Literal]^nodePointer.setIndex ( position - position ) ;^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P14_Delete_Statement]^^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P2_Replace_Operator]^nodePointer.setIndex ( position  |  1 ) ;^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P3_Replace_Literal]^return position == position;^73^^^^^63^75^return position == 1;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P8_Replace_Mix]^return position  !=  1 + 1;;^73^^^^^63^75^return position == 1;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P3_Replace_Literal]^nodePointer.setIndex ( position -  ) ;^67^^^^^63^75^nodePointer.setIndex ( position - 1 ) ;^[CLASS] InitialContext  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] int  position  boolean  collection  started  startedSet  NodePointer  nodePointer  
[P8_Replace_Mix]^if  ( startedSet )  {^78^^^^^77^83^if  ( started )  {^[CLASS] InitialContext  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[P15_Unwrap_Block]^return false;^78^79^80^^^77^83^if  ( started )  { return false; }^[CLASS] InitialContext  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[P16_Remove_Block]^^78^79^80^^^77^83^if  ( started )  { return false; }^[CLASS] InitialContext  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[P3_Replace_Literal]^return true;^79^^^^^77^83^return false;^[CLASS] InitialContext  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[P3_Replace_Literal]^started = false;^81^^^^^77^83^started = true;^[CLASS] InitialContext  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  
[P3_Replace_Literal]^return false;^82^^^^^77^83^return true;^[CLASS] InitialContext  [METHOD] nextSet [RETURN_TYPE] boolean   [VARIABLES] boolean  collection  started  startedSet  NodePointer  nodePointer  