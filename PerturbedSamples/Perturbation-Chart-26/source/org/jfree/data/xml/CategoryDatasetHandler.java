[REPLACE]^this.dataset = true;^63^^^^^62^64^[REPLACE] this.dataset = null;^[METHOD] <init> [TYPE] CategoryDatasetHandler() [PARAMETER] [CLASS] CategoryDatasetHandler   [TYPE]  DefaultCategoryDataset dataset  [TYPE]  boolean false  true 
[ADD]^^63^^^^^62^64^[ADD] this.dataset = null;^[METHOD] <init> [TYPE] CategoryDatasetHandler() [PARAMETER] [CLASS] CategoryDatasetHandler   [TYPE]  DefaultCategoryDataset dataset  [TYPE]  boolean false  true 
[REPLACE]^this.dataset .DefaultCategoryDataset (  )  ;^83^^^^^82^84^[REPLACE] this.dataset.addValue ( value, rowKey, columnKey ) ;^[METHOD] addItem [TYPE] void [PARAMETER] Comparable rowKey Comparable columnKey Number value [CLASS] CategoryDatasetHandler   [TYPE]  Comparable columnKey  rowKey  [TYPE]  boolean false  true  [TYPE]  Number value  [TYPE]  DefaultCategoryDataset dataset 
[REPLACE]^CategorySeriesHandler subhandler = new CategorySeriesHandler ( this ) ;^101^^^^^96^117^[REPLACE] DefaultHandler current = getCurrentHandler (  ) ;^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[ADD]^^101^^^^^96^117^[ADD] DefaultHandler current = getCurrentHandler (  ) ;^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[REPLACE]^if  ( current  ==  this )  {^102^^^^^96^117^[REPLACE] if  ( current != this )  {^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[REPLACE]^else {^105^^^^^96^117^[REPLACE] else if  ( qName.equals ( CATEGORYDATASET_TAG )  )  {^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[ADD]^^105^106^107^^^96^117^[ADD] else if  ( qName.equals ( CATEGORYDATASET_TAG )  )  { this.dataset = new DefaultCategoryDataset (  ) ; }^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[REPLACE]^else {^108^^^^^96^117^[REPLACE] else if  ( qName.equals ( SERIES_TAG )  )  {^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[REPLACE]^return ;^114^^^^^96^117^[REPLACE] throw new SAXException  (" ")  ;^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[REPLACE]^DefaultHandler current = getCurrentHandler (  ) ;^109^^^^^96^117^[REPLACE] CategorySeriesHandler subhandler = new CategorySeriesHandler ( this ) ;^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[ADD]^^109^110^^^^96^117^[ADD] CategorySeriesHandler subhandler = new CategorySeriesHandler ( this ) ; getSubHandlers (  ) .push ( subhandler ) ;^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[REPLACE]^this.dataset  =  this.dataset ;^106^^^^^96^117^[REPLACE] this.dataset = new DefaultCategoryDataset (  ) ;^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[ADD]^^106^^^^^96^117^[ADD] this.dataset = new DefaultCategoryDataset (  ) ;^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[REPLACE]^else if  ( qName.equals ( CATEGORYDATASET_TAG )  )  {^108^^^^^96^117^[REPLACE] else if  ( qName.equals ( SERIES_TAG )  )  {^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[REPLACE]^subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^103^^^^^96^117^[REPLACE] current.startElement ( namespaceURI, localName, qName, atts ) ;^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[ADD]^^103^^^^^96^117^[ADD] current.startElement ( namespaceURI, localName, qName, atts ) ;^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[ADD]^^114^^^^^96^117^[ADD] throw new SAXException  (" ")  ;^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[ADD]^getSubHandlers (  ) .push ( subhandler ) ;^109^110^^^^96^117^[ADD] CategorySeriesHandler subhandler = new CategorySeriesHandler ( this ) ; getSubHandlers (  ) .push ( subhandler ) ;^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[REPLACE]^this.dataset ;^106^^^^^96^117^[REPLACE] this.dataset = new DefaultCategoryDataset (  ) ;^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[ADD]^CategorySeriesHandler subhandler = new CategorySeriesHandler ( this ) ;getSubHandlers (  ) .push ( subhandler ) ;subhandler.startElement ( namespaceURI, localName, qName, atts ) ;^108^109^110^111^112^96^117^[ADD] else if  ( qName.equals ( SERIES_TAG )  )  { CategorySeriesHandler subhandler = new CategorySeriesHandler ( this ) ; getSubHandlers (  ) .push ( subhandler ) ; subhandler.startElement ( namespaceURI, localName, qName, atts ) ; }^[METHOD] startElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName Attributes atts [CLASS] CategoryDatasetHandler   [TYPE]  Attributes atts  [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  CategorySeriesHandler subhandler  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[REPLACE]^CategorySeriesHandler subhandler = new CategorySeriesHandler ( this ) ;^132^^^^^128^137^[REPLACE] DefaultHandler current = getCurrentHandler (  ) ;^[METHOD] endElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName [CLASS] CategoryDatasetHandler   [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[REPLACE]^if  ( current  ==  this )  {^133^^^^^128^137^[REPLACE] if  ( current != this )  {^[METHOD] endElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName [CLASS] CategoryDatasetHandler   [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 
[REPLACE]^current .startElement ( localName , qName , qName , null )  ;^134^^^^^128^137^[REPLACE] current.endElement ( namespaceURI, localName, qName ) ;^[METHOD] endElement [TYPE] void [PARAMETER] String namespaceURI String localName String qName [CLASS] CategoryDatasetHandler   [TYPE]  String localName  namespaceURI  qName  [TYPE]  boolean false  true  [TYPE]  DefaultCategoryDataset dataset  [TYPE]  DefaultHandler current 