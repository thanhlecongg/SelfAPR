[buglab_swap_variables]^if  ( testAttr (  name )  )  {^60^^^^^42^66^if  ( testAttr ( attr, name )  )  {^[CLASS] DOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] NamedNodeMap  map  boolean  QName  name  Node  node  List  attributes  NodePointer  parent  String  lname  int  count  i  position  Attr  attr  
[buglab_swap_variables]^if  ( testAttr ( name, attr )  )  {^60^^^^^42^66^if  ( testAttr ( attr, name )  )  {^[CLASS] DOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] NamedNodeMap  map  boolean  QName  name  Node  node  List  attributes  NodePointer  parent  String  lname  int  count  i  position  Attr  attr  
[buglab_swap_variables]^if  ( testAttr ( attr )  )  {^60^^^^^42^66^if  ( testAttr ( attr, name )  )  {^[CLASS] DOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] NamedNodeMap  map  boolean  QName  name  Node  node  List  attributes  NodePointer  parent  String  lname  int  count  i  position  Attr  attr  
[buglab_swap_variables]^for  ( countnt i = 0; i < i; i++ )  {^58^^^^^42^66^for  ( int i = 0; i < count; i++ )  {^[CLASS] DOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] NamedNodeMap  map  boolean  QName  name  Node  node  List  attributes  NodePointer  parent  String  lname  int  count  i  position  Attr  attr  
[buglab_swap_variables]^Attr attr =  ( Attr )  i.item ( map ) ;^59^^^^^42^66^Attr attr =  ( Attr )  map.item ( i ) ;^[CLASS] DOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] NamedNodeMap  map  boolean  QName  name  Node  node  List  attributes  NodePointer  parent  String  lname  int  count  i  position  Attr  attr  
[buglab_swap_variables]^Attr attr = getAttribute (  ( Element )  node ) ;^50^^^^^42^66^Attr attr = getAttribute (  ( Element )  node, name ) ;^[CLASS] DOMAttributeIterator  [METHOD] <init> [RETURN_TYPE] QName)   NodePointer parent QName name [VARIABLES] NamedNodeMap  map  boolean  QName  name  Node  node  List  attributes  NodePointer  parent  String  lname  int  count  i  position  Attr  attr  
[buglab_swap_variables]^if  ( nodeLocalName == null && nodePrefix.equals ( "xmlns" )  )  {^76^^^^^61^91^if  ( nodePrefix == null && nodeLocalName.equals ( "xmlns" )  )  {^[CLASS] DOMAttributeIterator  [METHOD] testAttr [RETURN_TYPE] boolean   Attr attr QName testName [VARIABLES] boolean  QName  name  testName  List  attributes  NodePointer  parent  String  nodeLocalName  nodeNS  nodePrefix  testLocalName  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^if  ( nodeLocalName.equals ( "*" )  || testLocalName.equals ( testLocalName )  )  {^81^^^^^66^96^if  ( testLocalName.equals ( "*" )  || testLocalName.equals ( nodeLocalName )  )  {^[CLASS] DOMAttributeIterator  [METHOD] testAttr [RETURN_TYPE] boolean   Attr attr QName testName [VARIABLES] boolean  QName  name  testName  List  attributes  NodePointer  parent  String  nodeLocalName  nodeNS  nodePrefix  testLocalName  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^if  ( equalStrings ( nodePrefix, testPrefix )  )  {^84^^^^^69^99^if  ( equalStrings ( testPrefix, nodePrefix )  )  {^[CLASS] DOMAttributeIterator  [METHOD] testAttr [RETURN_TYPE] boolean   Attr attr QName testName [VARIABLES] boolean  QName  name  testName  List  attributes  NodePointer  parent  String  nodeLocalName  nodeNS  nodePrefix  testLocalName  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^if  ( equalStrings (  nodePrefix )  )  {^84^^^^^69^99^if  ( equalStrings ( testPrefix, nodePrefix )  )  {^[CLASS] DOMAttributeIterator  [METHOD] testAttr [RETURN_TYPE] boolean   Attr attr QName testName [VARIABLES] boolean  QName  name  testName  List  attributes  NodePointer  parent  String  nodeLocalName  nodeNS  nodePrefix  testLocalName  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^if  ( equalStrings ( testPrefix )  )  {^84^^^^^69^99^if  ( equalStrings ( testPrefix, nodePrefix )  )  {^[CLASS] DOMAttributeIterator  [METHOD] testAttr [RETURN_TYPE] boolean   Attr attr QName testName [VARIABLES] boolean  QName  name  testName  List  attributes  NodePointer  parent  String  nodeLocalName  nodeNS  nodePrefix  testLocalName  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^testNS = testPrefix.getNamespaceURI ( parent ) ;^90^^^^^75^105^testNS = parent.getNamespaceURI ( testPrefix ) ;^[CLASS] DOMAttributeIterator  [METHOD] testAttr [RETURN_TYPE] boolean   Attr attr QName testName [VARIABLES] boolean  QName  name  testName  List  attributes  NodePointer  parent  String  nodeLocalName  nodeNS  nodePrefix  testLocalName  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^nodeNS = nodePrefix.getNamespaceURI ( parent ) ;^95^^^^^80^110^nodeNS = parent.getNamespaceURI ( nodePrefix ) ;^[CLASS] DOMAttributeIterator  [METHOD] testAttr [RETURN_TYPE] boolean   Attr attr QName testName [VARIABLES] boolean  QName  name  testName  List  attributes  NodePointer  parent  String  nodeLocalName  nodeNS  nodePrefix  testLocalName  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^return equalStrings ( nodeNS, testNS ) ;^97^^^^^82^112^return equalStrings ( testNS, nodeNS ) ;^[CLASS] DOMAttributeIterator  [METHOD] testAttr [RETURN_TYPE] boolean   Attr attr QName testName [VARIABLES] boolean  QName  name  testName  List  attributes  NodePointer  parent  String  nodeLocalName  nodeNS  nodePrefix  testLocalName  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^return equalStrings (  nodeNS ) ;^97^^^^^82^112^return equalStrings ( testNS, nodeNS ) ;^[CLASS] DOMAttributeIterator  [METHOD] testAttr [RETURN_TYPE] boolean   Attr attr QName testName [VARIABLES] boolean  QName  name  testName  List  attributes  NodePointer  parent  String  nodeLocalName  nodeNS  nodePrefix  testLocalName  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^return equalStrings ( testNS ) ;^97^^^^^82^112^return equalStrings ( testNS, nodeNS ) ;^[CLASS] DOMAttributeIterator  [METHOD] testAttr [RETURN_TYPE] boolean   Attr attr QName testName [VARIABLES] boolean  QName  name  testName  List  attributes  NodePointer  parent  String  nodeLocalName  nodeNS  nodePrefix  testLocalName  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^if  ( s2 == null && s1 != null )  {^103^^^^^102^110^if  ( s1 == null && s2 != null )  {^[CLASS] DOMAttributeIterator  [METHOD] equalStrings [RETURN_TYPE] boolean   String s1 String s2 [VARIABLES] List  attributes  NodePointer  parent  String  s1  s2  boolean  QName  name  testName  int  count  i  position  
[buglab_swap_variables]^if  ( s2 != null && !s1.equals ( s1 )  )  {^106^^^^^102^110^if  ( s1 != null && !s1.equals ( s2 )  )  {^[CLASS] DOMAttributeIterator  [METHOD] equalStrings [RETURN_TYPE] boolean   String s1 String s2 [VARIABLES] List  attributes  NodePointer  parent  String  s1  s2  boolean  QName  name  testName  int  count  i  position  
[buglab_swap_variables]^testNS = testPrefix.getNamespaceURI ( parent ) ;^117^^^^^112^141^testNS = parent.getNamespaceURI ( testPrefix ) ;^[CLASS] DOMAttributeIterator  [METHOD] getAttribute [RETURN_TYPE] Attr   Element element QName name [VARIABLES] NamedNodeMap  nnm  boolean  QName  name  testName  Element  element  List  attributes  NodePointer  parent  String  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^return name.getAttributeNode ( element.getName (  )  ) ;^139^^^^^120^140^return element.getAttributeNode ( name.getName (  )  ) ;^[CLASS] DOMAttributeIterator  [METHOD] getAttribute [RETURN_TYPE] Attr   Element element QName name [VARIABLES] NamedNodeMap  nnm  boolean  QName  name  testName  Element  element  List  attributes  NodePointer  parent  String  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^if  ( testAttr ( name, attr )  )  {^132^^^^^112^141^if  ( testAttr ( attr, name )  )  {^[CLASS] DOMAttributeIterator  [METHOD] getAttribute [RETURN_TYPE] Attr   Element element QName name [VARIABLES] NamedNodeMap  nnm  boolean  QName  name  testName  Element  element  List  attributes  NodePointer  parent  String  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^if  ( testAttr (  name )  )  {^132^^^^^112^141^if  ( testAttr ( attr, name )  )  {^[CLASS] DOMAttributeIterator  [METHOD] getAttribute [RETURN_TYPE] Attr   Element element QName name [VARIABLES] NamedNodeMap  nnm  boolean  QName  name  testName  Element  element  List  attributes  NodePointer  parent  String  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^if  ( testAttr ( attr )  )  {^132^^^^^112^141^if  ( testAttr ( attr, name )  )  {^[CLASS] DOMAttributeIterator  [METHOD] getAttribute [RETURN_TYPE] Attr   Element element QName name [VARIABLES] NamedNodeMap  nnm  boolean  QName  name  testName  Element  element  List  attributes  NodePointer  parent  String  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^attr =  ( Attr )  i.item ( nnm ) ;^131^^^^^112^141^attr =  ( Attr )  nnm.item ( i ) ;^[CLASS] DOMAttributeIterator  [METHOD] getAttribute [RETURN_TYPE] Attr   Element element QName name [VARIABLES] NamedNodeMap  nnm  boolean  QName  name  testName  Element  element  List  attributes  NodePointer  parent  String  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^Attr attr = testNS.getAttributeNodeNS ( element, name.getName (  )  ) ;^121^^^^^112^141^Attr attr = element.getAttributeNodeNS ( testNS, name.getName (  )  ) ;^[CLASS] DOMAttributeIterator  [METHOD] getAttribute [RETURN_TYPE] Attr   Element element QName name [VARIABLES] NamedNodeMap  nnm  boolean  QName  name  testName  Element  element  List  attributes  NodePointer  parent  String  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^Attr attr = element.getAttributeNodeNS ( name, testNS.getName (  )  ) ;^121^^^^^112^141^Attr attr = element.getAttributeNodeNS ( testNS, name.getName (  )  ) ;^[CLASS] DOMAttributeIterator  [METHOD] getAttribute [RETURN_TYPE] Attr   Element element QName name [VARIABLES] NamedNodeMap  nnm  boolean  QName  name  testName  Element  element  List  attributes  NodePointer  parent  String  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^Attr attr = element.getAttributeNodeNS (  name.getName (  )  ) ;^121^^^^^112^141^Attr attr = element.getAttributeNodeNS ( testNS, name.getName (  )  ) ;^[CLASS] DOMAttributeIterator  [METHOD] getAttribute [RETURN_TYPE] Attr   Element element QName name [VARIABLES] NamedNodeMap  nnm  boolean  QName  name  testName  Element  element  List  attributes  NodePointer  parent  String  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^Attr attr = element.getAttributeNodeNS ( testNS.getName (  )  ) ;^121^^^^^112^141^Attr attr = element.getAttributeNodeNS ( testNS, name.getName (  )  ) ;^[CLASS] DOMAttributeIterator  [METHOD] getAttribute [RETURN_TYPE] Attr   Element element QName name [VARIABLES] NamedNodeMap  nnm  boolean  QName  name  testName  Element  element  List  attributes  NodePointer  parent  String  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^Attr attr = name.getAttributeNodeNS ( testNS, element.getName (  )  ) ;^121^^^^^112^141^Attr attr = element.getAttributeNodeNS ( testNS, name.getName (  )  ) ;^[CLASS] DOMAttributeIterator  [METHOD] getAttribute [RETURN_TYPE] Attr   Element element QName name [VARIABLES] NamedNodeMap  nnm  boolean  QName  name  testName  Element  element  List  attributes  NodePointer  parent  String  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^for  ( nnmnt i = 0; i < i.getLength (  ) ; i++ )  {^130^^^^^112^141^for  ( int i = 0; i < nnm.getLength (  ) ; i++ )  {^[CLASS] DOMAttributeIterator  [METHOD] getAttribute [RETURN_TYPE] Attr   Element element QName name [VARIABLES] NamedNodeMap  nnm  boolean  QName  name  testName  Element  element  List  attributes  NodePointer  parent  String  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^return name.getAttributeNode ( element.getName (  )  ) ;^139^^^^^112^141^return element.getAttributeNode ( name.getName (  )  ) ;^[CLASS] DOMAttributeIterator  [METHOD] getAttribute [RETURN_TYPE] Attr   Element element QName name [VARIABLES] NamedNodeMap  nnm  boolean  QName  name  testName  Element  element  List  attributes  NodePointer  parent  String  testNS  testPrefix  int  count  i  position  Attr  attr  
[buglab_swap_variables]^return new DOMAttributePointer ( parent,  ( Attr )  index.get ( attributes )  ) ;^154^^^^^143^155^return new DOMAttributePointer ( parent,  ( Attr )  attributes.get ( index )  ) ;^[CLASS] DOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  attributes  NodePointer  parent  boolean  QName  name  testName  int  count  i  index  position  
[buglab_swap_variables]^return new DOMAttributePointer ( index,  ( Attr )  attributes.get ( parent )  ) ;^154^^^^^143^155^return new DOMAttributePointer ( parent,  ( Attr )  attributes.get ( index )  ) ;^[CLASS] DOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  attributes  NodePointer  parent  boolean  QName  name  testName  int  count  i  index  position  
[buglab_swap_variables]^return new DOMAttributePointer (   ( Attr )  attributes.get ( index )  ) ;^154^^^^^143^155^return new DOMAttributePointer ( parent,  ( Attr )  attributes.get ( index )  ) ;^[CLASS] DOMAttributeIterator  [METHOD] getNodePointer [RETURN_TYPE] NodePointer   [VARIABLES] List  attributes  NodePointer  parent  boolean  QName  name  testName  int  count  i  index  position  
[buglab_swap_variables]^return attributes >= 1 && position <= position.size (  ) ;^163^^^^^161^164^return position >= 1 && position <= attributes.size (  ) ;^[CLASS] DOMAttributeIterator  [METHOD] setPosition [RETURN_TYPE] boolean   int position [VARIABLES] List  attributes  NodePointer  parent  boolean  QName  name  testName  int  count  i  index  position  