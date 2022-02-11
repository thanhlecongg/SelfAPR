[REPLACE]^private static final long serialVersionUID  = null ;^21^^^^^^^[REPLACE] private static final long serialVersionUID = 1L;^ [CLASS] DOMDeserializer NodeDeserializer DocumentDeserializer  
[REPLACE]^private final  DocumentBuilderFactory _parserFactory;^23^^^^^^^[REPLACE] private final static DocumentBuilderFactory _parserFactory;^ [CLASS] DOMDeserializer NodeDeserializer DocumentDeserializer  
[REPLACE]^private static final long serialVersionUID ;^52^^^^^^^[REPLACE] private static final long serialVersionUID = 1L;^ [CLASS] DOMDeserializer NodeDeserializer DocumentDeserializer  
[REPLACE]^private  final long serialVersionUID = 1L;^62^^^^^^^[REPLACE] private static final long serialVersionUID = 1L;^ [CLASS] DOMDeserializer NodeDeserializer DocumentDeserializer  
[REPLACE]^protected DOMDeserializer ( Class<T> null )  { super ( cls ) ; }^30^^^^^^^[REPLACE] protected DOMDeserializer ( Class<T> cls )  { super ( cls ) ; }^[METHOD] <init> [TYPE] Class) [PARAMETER] Class<T> cls [CLASS] DOMDeserializer NodeDeserializer DocumentDeserializer   [TYPE]  Class cls  [TYPE]  DocumentBuilderFactory _parserFactory  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^public NodeDeserializer (  )  { super ( Node.class ) ; } @Override^53^^^^^^^[REPLACE] public NodeDeserializer (  )  { super ( Node.class ) ; }^[METHOD] <init> [TYPE] DOMDeserializer$NodeDeserializer() [PARAMETER] [CLASS] DOMDeserializer NodeDeserializer DocumentDeserializer   [TYPE]  long serialVersionUID  [TYPE]  DocumentBuilderFactory _parserFactory  [TYPE]  boolean false  true 
[REPLACE]^public DocumentDeserializer (  )  { super ( Document.class ) ; } @Override^63^^^^^^^[REPLACE] public DocumentDeserializer (  )  { super ( Document.class ) ; }^[METHOD] <init> [TYPE] DOMDeserializer$DocumentDeserializer() [PARAMETER] [CLASS] DOMDeserializer NodeDeserializer DocumentDeserializer   [TYPE]  long serialVersionUID  [TYPE]  DocumentBuilderFactory _parserFactory  [TYPE]  boolean false  true 
[ADD]^return _parserFactory.newDocumentBuilder (  ) .parse ( new InputSource ( new StringReader ( value )  )  ) ;^37^38^39^40^41^35^42^[ADD] try { return _parserFactory.newDocumentBuilder (  ) .parse ( new InputSource ( new StringReader ( value )  )  ) ; } catch  ( Exception e )  { throw new IllegalArgumentException  (" ")  ; }^[METHOD] parse [TYPE] Document [PARAMETER] String value [CLASS] DOMDeserializer NodeDeserializer DocumentDeserializer   [TYPE]  DocumentBuilderFactory _parserFactory  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  Exception e 
[REPLACE]^return _parserFactory.newDocumentBuilder (  )  .parse ( value )   ) ;^38^^^^^35^42^[REPLACE] return _parserFactory.newDocumentBuilder (  ) .parse ( new InputSource ( new StringReader ( value )  )  ) ;^[METHOD] parse [TYPE] Document [PARAMETER] String value [CLASS] DOMDeserializer NodeDeserializer DocumentDeserializer   [TYPE]  DocumentBuilderFactory _parserFactory  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  Exception e 
[REPLACE]^return ;^40^^^^^35^42^[REPLACE] throw new IllegalArgumentException  (" ")  ;^[METHOD] parse [TYPE] Document [PARAMETER] String value [CLASS] DOMDeserializer NodeDeserializer DocumentDeserializer   [TYPE]  DocumentBuilderFactory _parserFactory  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID  [TYPE]  Exception e 
[REPLACE]^return _parserFactory.newDocumentBuilder (  ) .parse ( new InputSource ( new StringReader ( value )  )  ) ;^56^^^^^55^57^[REPLACE] return parse ( value ) ;^[METHOD] _deserialize [TYPE] Node [PARAMETER] String value DeserializationContext ctxt [CLASS] DOMDeserializer NodeDeserializer DocumentDeserializer   [TYPE]  DeserializationContext ctxt  [TYPE]  DocumentBuilderFactory _parserFactory  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^private static final long serialVersionUID = 1;^52^^^^^^^[REPLACE] private static final long serialVersionUID = 1L;^[METHOD] _deserialize [TYPE] Document [PARAMETER] String value DeserializationContext ctxt [CLASS] NodeDeserializer   [TYPE]  DeserializationContext ctxt  [TYPE]  DocumentBuilderFactory _parserFactory  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^public NodeDeserializer (  )  { super ( Node.class ) ; } @Override^53^^^^^^^[REPLACE] public NodeDeserializer (  )  { super ( Node.class ) ; }^[METHOD] <init> [TYPE] DOMDeserializer$NodeDeserializer() [PARAMETER] [CLASS] NodeDeserializer   [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[REPLACE]^private  final long serialVersionUID = 1L;^62^^^^^^^[REPLACE] private static final long serialVersionUID = 1L;^[METHOD] _deserialize [TYPE] Node [PARAMETER] String value DeserializationContext ctxt [CLASS] DocumentDeserializer   [TYPE]  DeserializationContext ctxt  [TYPE]  String value  [TYPE]  boolean false  true  [TYPE]  long serialVersionUID 
[REPLACE]^public DocumentDeserializer (  )  { super ( Document.class ) ; } @Override^63^^^^^^^[REPLACE] public DocumentDeserializer (  )  { super ( Document.class ) ; }^[METHOD] <init> [TYPE] DOMDeserializer$DocumentDeserializer() [PARAMETER] [CLASS] DocumentDeserializer   [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 
[ADD]^^63^64^65^^^^^[ADD] public DocumentDeserializer (  )  { super ( Document.class ) ; } @Override public Document _deserialize ( String value, DeserializationContext ctxt )  throws IllegalArgumentException {^[METHOD] <init> [TYPE] DOMDeserializer$DocumentDeserializer() [PARAMETER] [CLASS] DocumentDeserializer   [TYPE]  long serialVersionUID  [TYPE]  boolean false  true 