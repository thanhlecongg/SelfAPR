[P8_Replace_Mix]^this.root =  null;^73^^^^^72^75^this.root = root;^[CLASS] CategorySeriesHandler  [METHOD] <init> [RETURN_TYPE] RootHandler)   RootHandler root [VARIABLES] RootHandler  root  Comparable  seriesKey  boolean  DefaultKeyedValues  values  
[P8_Replace_Mix]^this.values  =  this.values ;^74^^^^^72^75^this.values = new DefaultKeyedValues (  ) ;^[CLASS] CategorySeriesHandler  [METHOD] <init> [RETURN_TYPE] RootHandler)   RootHandler root [VARIABLES] RootHandler  root  Comparable  seriesKey  boolean  DefaultKeyedValues  values  
[P5_Replace_Variable]^this.seriesKey = seriesKey;^83^^^^^82^84^this.seriesKey = key;^[CLASS] CategorySeriesHandler  [METHOD] setSeriesKey [RETURN_TYPE] void   Comparable key [VARIABLES] RootHandler  root  Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  
[P5_Replace_Variable]^this.values.addValue ( seriesKey, value ) ;^93^^^^^92^94^this.values.addValue ( key, value ) ;^[CLASS] CategorySeriesHandler  [METHOD] addItem [RETURN_TYPE] void   Comparable key Number value [VARIABLES] RootHandler  root  Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  
[P5_Replace_Variable]^this.values.addValue (  value ) ;^93^^^^^92^94^this.values.addValue ( key, value ) ;^[CLASS] CategorySeriesHandler  [METHOD] addItem [RETURN_TYPE] void   Comparable key Number value [VARIABLES] RootHandler  root  Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  
[P5_Replace_Variable]^this.values.addValue ( key ) ;^93^^^^^92^94^this.values.addValue ( key, value ) ;^[CLASS] CategorySeriesHandler  [METHOD] addItem [RETURN_TYPE] void   Comparable key Number value [VARIABLES] RootHandler  root  Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  
[P5_Replace_Variable]^this.values.addValue ( value, key ) ;^93^^^^^92^94^this.values.addValue ( key, value ) ;^[CLASS] CategorySeriesHandler  [METHOD] addItem [RETURN_TYPE] void   Comparable key Number value [VARIABLES] RootHandler  root  Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  
[P14_Delete_Statement]^^93^^^^^92^94^this.values.addValue ( key, value ) ;^[CLASS] CategorySeriesHandler  [METHOD] addItem [RETURN_TYPE] void   Comparable key Number value [VARIABLES] RootHandler  root  Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  
[P5_Replace_Variable]^if  ( namespaceURI.equals ( SERIES_TAG )  )  {^111^^^^^106^127^if  ( qName.equals ( SERIES_TAG )  )  {^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P15_Unwrap_Block]^setSeriesKey(atts.getValue("name")); org.jfree.data.xml.ItemHandler subhandler = new org.jfree.data.xml.ItemHandler(this.root, this); this.root.pushSubHandler(subhandler);^111^112^113^114^115^106^127^if  ( qName.equals ( SERIES_TAG )  )  { setSeriesKey ( atts.getValue ( "name" )  ) ; ItemHandler subhandler = new ItemHandler ( this.root, this ) ; this.root.pushSubHandler ( subhandler ) ; }^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P16_Remove_Block]^^111^112^113^114^115^106^127^if  ( qName.equals ( SERIES_TAG )  )  { setSeriesKey ( atts.getValue ( "name" )  ) ; ItemHandler subhandler = new ItemHandler ( this.root, this ) ; this.root.pushSubHandler ( subhandler ) ; }^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^else if  ( namespaceURI.equals ( ITEM_TAG )  )  {^116^^^^^106^127^else if  ( qName.equals ( ITEM_TAG )  )  {^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P8_Replace_Mix]^else {^116^^^^^106^127^else if  ( qName.equals ( ITEM_TAG )  )  {^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P15_Unwrap_Block]^org.jfree.data.xml.ItemHandler subhandler = new org.jfree.data.xml.ItemHandler(this.root, this); this.root.pushSubHandler(subhandler); subhandler.startElement(namespaceURI, localName, qName, atts);^116^117^118^119^120^106^127^else if  ( qName.equals ( ITEM_TAG )  )  { ItemHandler subhandler = new ItemHandler ( this.root, this ) ; this.root.pushSubHandler ( subhandler ) ; subhandler.startElement ( namespaceURI, localName, qName, atts ) ; }^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P16_Remove_Block]^^116^117^118^119^120^106^127^else if  ( qName.equals ( ITEM_TAG )  )  { ItemHandler subhandler = new ItemHandler ( this.root, this ) ; this.root.pushSubHandler ( subhandler ) ; subhandler.startElement ( namespaceURI, localName, qName, atts ) ; }^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P3_Replace_Literal]^throw new SAXException ( "ting <Series> " + qName ) ;^123^124^125^^^106^127^throw new SAXException ( "Expecting <Series> or <Item> tag...found " + qName ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^throw new SAXException ( "Expecting <Series> or <Item> tag...found " + namespaceURI ) ;^123^124^125^^^106^127^throw new SAXException ( "Expecting <Series> or <Item> tag...found " + qName ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P3_Replace_Literal]^throw new SAXException ( "eries> or <I" + qName ) ;^123^124^125^^^106^127^throw new SAXException ( "Expecting <Series> or <Item> tag...found " + qName ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^ItemHandler subhandler = new ItemHandler ( root, this ) ;^117^^^^^106^127^ItemHandler subhandler = new ItemHandler ( this.root, this ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P7_Replace_Invocation]^this.root .popSubHandler (  )  ;^118^^^^^106^127^this.root.pushSubHandler ( subhandler ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P14_Delete_Statement]^^118^119^^^^106^127^this.root.pushSubHandler ( subhandler ) ; subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P11_Insert_Donor_Statement]^this.root.popSubHandler (  ) ;this.root.pushSubHandler ( subhandler ) ;^118^^^^^106^127^this.root.pushSubHandler ( subhandler ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^subhandler.startElement ( qName, localName, qName, atts ) ;^119^^^^^106^127^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, qName, qName, atts ) ;^119^^^^^106^127^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^subhandler.startElement (  localName, qName, atts ) ;^119^^^^^106^127^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI,  qName, atts ) ;^119^^^^^106^127^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, localName,  atts ) ;^119^^^^^106^127^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, localName, qName ) ;^119^^^^^106^127^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^subhandler.startElement ( localName, namespaceURI, qName, atts ) ;^119^^^^^106^127^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, qName, localName, atts ) ;^119^^^^^106^127^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^subhandler.startElement ( atts, localName, qName, namespaceURI ) ;^119^^^^^106^127^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P8_Replace_Mix]^subhandler.startElement ( namespaceURI, localName, localName, atts ) ;^119^^^^^106^127^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P14_Delete_Statement]^^119^^^^^106^127^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P3_Replace_Literal]^throw new SAXException ( "cting <xpecting <Series> or <Item> tag...found " + qName ) ;^123^124^125^^^106^127^throw new SAXException ( "Expecting <Series> or <Item> tag...found " + qName ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P3_Replace_Literal]^throw new SAXException ( "Expecting <Series> or <Item> tag...found ng <Se" + qName ) ;^123^124^125^^^106^127^throw new SAXException ( "Expecting <Series> or <Item> tag...found " + qName ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P14_Delete_Statement]^^118^^^^^106^127^this.root.pushSubHandler ( subhandler ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, namespaceURI, qName, atts ) ;^119^^^^^106^127^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^subhandler.startElement ( namespaceURI, localName, namespaceURI, atts ) ;^119^^^^^106^127^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^subhandler.startElement ( qName, localName, namespaceURI, atts ) ;^119^^^^^106^127^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^ItemHandler subhandler = new ItemHandler ( root, this ) ;^113^^^^^106^127^ItemHandler subhandler = new ItemHandler ( this.root, this ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P3_Replace_Literal]^setSeriesKey ( atts.getValue ( "namenam" )  ) ;^112^^^^^106^127^setSeriesKey ( atts.getValue ( "name" )  ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P7_Replace_Invocation]^addItem ( atts.getValue ( "name" )  ) ;^112^^^^^106^127^setSeriesKey ( atts.getValue ( "name" )  ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P14_Delete_Statement]^^112^113^^^^106^127^setSeriesKey ( atts.getValue ( "name" )  ) ; ItemHandler subhandler = new ItemHandler ( this.root, this ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P3_Replace_Literal]^setSeriesKey ( atts.getValue ( "ame" )  ) ;^112^^^^^106^127^setSeriesKey ( atts.getValue ( "name" )  ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P8_Replace_Mix]^setSeriesKey ( atts .getValue ( qName )   ) ;^112^^^^^106^127^setSeriesKey ( atts.getValue ( "name" )  ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P8_Replace_Mix]^this.root .popSubHandler (  )  ;^114^^^^^106^127^this.root.pushSubHandler ( subhandler ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P14_Delete_Statement]^^114^^^^^106^127^this.root.pushSubHandler ( subhandler ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P11_Insert_Donor_Statement]^this.root.popSubHandler (  ) ;this.root.pushSubHandler ( subhandler ) ;^114^^^^^106^127^this.root.pushSubHandler ( subhandler ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P3_Replace_Literal]^throw new SAXException ( "Expecting <Series> or <Item> tag...found Expe" + qName ) ;^123^124^125^^^106^127^throw new SAXException ( "Expecting <Series> or <Item> tag...found " + qName ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P3_Replace_Literal]^throw new SAXException ( "Expecting <Series> or <Item> tag...found ies> or <Item>" + qName ) ;^123^124^125^^^106^127^throw new SAXException ( "Expecting <Series> or <Item> tag...found " + qName ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P3_Replace_Literal]^throw new SAXException ( "xpecting <Sxpecting <Series> or <Item> tag...found " + qName ) ;^123^124^125^^^106^127^throw new SAXException ( "Expecting <Series> or <Item> tag...found " + qName ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P3_Replace_Literal]^throw new SAXException ( "Expe" + qName ) ;^123^124^125^^^106^127^throw new SAXException ( "Expecting <Series> or <Item> tag...found " + qName ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P14_Delete_Statement]^^112^^^^^106^127^setSeriesKey ( atts.getValue ( "name" )  ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P7_Replace_Invocation]^setSeriesKey ( atts .getValue ( namespaceURI )   ) ;^112^^^^^106^127^setSeriesKey ( atts.getValue ( "name" )  ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P8_Replace_Mix]^setSeriesKey ( atts .getValue ( localName )   ) ;^112^^^^^106^127^setSeriesKey ( atts.getValue ( "name" )  ) ;^[CLASS] CategorySeriesHandler  [METHOD] startElement [RETURN_TYPE] void   String namespaceURI String localName String qName Attributes atts [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  ItemHandler  subhandler  Attributes  atts  RootHandler  root  String  localName  namespaceURI  qName  
[P8_Replace_Mix]^while  ( iterator .next (  )   )  {^144^^^^^136^153^while  ( iterator.hasNext (  )  )  {^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P8_Replace_Mix]^Comparable key =  ( Comparable )  iterator .hasNext (  )  ;^145^^^^^136^153^Comparable key =  ( Comparable )  iterator.next (  ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^Number value = values.getValue ( key ) ;^146^^^^^136^153^Number value = this.values.getValue ( key ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^Number value = key.getValue ( this.values ) ;^146^^^^^136^153^Number value = this.values.getValue ( key ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P8_Replace_Mix]^Number value = this.values.getValue ( seriesKey ) ;^146^^^^^136^153^Number value = this.values.getValue ( key ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P14_Delete_Statement]^^145^^^^^136^153^Comparable key =  ( Comparable )  iterator.next (  ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P11_Insert_Donor_Statement]^while  ( iterator.hasNext (  )  )  { Comparable key =  ( Comparable )  iterator.next (  ) ;Comparable key =  ( Comparable )  iterator.next (  ) ;^145^^^^^136^153^Comparable key =  ( Comparable )  iterator.next (  ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P14_Delete_Statement]^^146^^^^^136^153^Number value = this.values.getValue ( key ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^handler.addItem ( this.seriesKey, seriesKey, value ) ;^147^^^^^136^153^handler.addItem ( this.seriesKey, key, value ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^handler.addItem ( seriesKey, key, value ) ;^147^^^^^136^153^handler.addItem ( this.seriesKey, key, value ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^handler.addItem ( this.seriesKey,  value ) ;^147^^^^^136^153^handler.addItem ( this.seriesKey, key, value ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^handler.addItem ( this.seriesKey, key ) ;^147^^^^^136^153^handler.addItem ( this.seriesKey, key, value ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^handler.addItem (  key, value ) ;^147^^^^^136^153^handler.addItem ( this.seriesKey, key, value ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^handler.addItem ( key, this.seriesKey, value ) ;^147^^^^^136^153^handler.addItem ( this.seriesKey, key, value ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^handler.addItem ( this.seriesKey, value, key ) ;^147^^^^^136^153^handler.addItem ( this.seriesKey, key, value ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^handler.addItem ( value, key, this.seriesKey ) ;^147^^^^^136^153^handler.addItem ( this.seriesKey, key, value ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P14_Delete_Statement]^^147^^^^^136^153^handler.addItem ( this.seriesKey, key, value ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P2_Replace_Operator]^if  ( this.root  !=  CategoryDatasetHandler )  {^140^^^^^136^153^if  ( this.root instanceof CategoryDatasetHandler )  {^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^if  ( root instanceof CategoryDatasetHandler )  {^140^^^^^136^153^if  ( this.root instanceof CategoryDatasetHandler )  {^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P5_Replace_Variable]^Iterator iterator = values.getKeys (  ) .iterator (  ) ;^143^^^^^136^153^Iterator iterator = this.values.getKeys (  ) .iterator (  ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P7_Replace_Invocation]^Iterator iterator = this.values.DefaultKeyedValues (  ) .iterator (  ) ;^143^^^^^136^153^Iterator iterator = this.values.getKeys (  ) .iterator (  ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P14_Delete_Statement]^^143^^^^^136^153^Iterator iterator = this.values.getKeys (  ) .iterator (  ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P8_Replace_Mix]^Iterator iterator = values.DefaultKeyedValues (  ) .iterator (  ) ;^143^^^^^136^153^Iterator iterator = this.values.getKeys (  ) .iterator (  ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P14_Delete_Statement]^^144^145^^^^136^153^while  ( iterator.hasNext (  )  )  { Comparable key =  ( Comparable )  iterator.next (  ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P14_Delete_Statement]^^145^146^^^^136^153^Comparable key =  ( Comparable )  iterator.next (  ) ; Number value = this.values.getValue ( key ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P8_Replace_Mix]^this.root .pushSubHandler ( null )  ;^150^^^^^136^153^this.root.popSubHandler (  ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P14_Delete_Statement]^^150^^^^^136^153^this.root.popSubHandler (  ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P11_Insert_Donor_Statement]^this.root.pushSubHandler ( subhandler ) ;this.root.popSubHandler (  ) ;^150^^^^^136^153^this.root.popSubHandler (  ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  
[P7_Replace_Invocation]^Iterator iterator = this.values .getValue ( namespaceURI )  .iterator (  ) ;^143^^^^^136^153^Iterator iterator = this.values.getKeys (  ) .iterator (  ) ;^[CLASS] CategorySeriesHandler  [METHOD] endElement [RETURN_TYPE] void   String namespaceURI String localName String qName [VARIABLES] Comparable  key  seriesKey  boolean  DefaultKeyedValues  values  Number  value  CategoryDatasetHandler  handler  Iterator  iterator  RootHandler  root  String  localName  namespaceURI  qName  