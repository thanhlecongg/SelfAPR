[REPLACE]^protected  InputStream _originalStream;^15^^^^^^^[REPLACE] protected final InputStream _originalStream;^ [CLASS] DataFormatMatcher  
[REPLACE]^private final  long  _bufferedStart;^25^^^^^^^[REPLACE] protected final int _bufferedStart;^ [CLASS] DataFormatMatcher  
[REPLACE]^protected final  long  _bufferedLength;^30^^^^^^^[REPLACE] protected final int _bufferedLength;^ [CLASS] DataFormatMatcher  
[REPLACE]^_originalStream =  null;^46^^^^^42^52^[REPLACE] _originalStream = in;^[METHOD] <init> [TYPE] MatchStrength) [PARAMETER] InputStream in byte[] buffered int bufferedStart int bufferedLength JsonFactory match MatchStrength strength [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^_bufferedStart = bufferedStart; ;^47^^^^^42^52^[REPLACE] _bufferedData = buffered;^[METHOD] <init> [TYPE] MatchStrength) [PARAMETER] InputStream in byte[] buffered int bufferedStart int bufferedLength JsonFactory match MatchStrength strength [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^_bufferedData = buffered; ;^48^^^^^42^52^[REPLACE] _bufferedStart = bufferedStart;^[METHOD] <init> [TYPE] MatchStrength) [PARAMETER] InputStream in byte[] buffered int bufferedStart int bufferedLength JsonFactory match MatchStrength strength [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^_bufferedLength =  bufferedStart;^49^^^^^42^52^[REPLACE] _bufferedLength = bufferedLength;^[METHOD] <init> [TYPE] MatchStrength) [PARAMETER] InputStream in byte[] buffered int bufferedStart int bufferedLength JsonFactory match MatchStrength strength [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^_match =  null;^50^^^^^42^52^[REPLACE] _match = match;^[METHOD] <init> [TYPE] MatchStrength) [PARAMETER] InputStream in byte[] buffered int bufferedStart int bufferedLength JsonFactory match MatchStrength strength [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^_matchStrength =  null;^51^^^^^42^52^[REPLACE] _matchStrength = strength;^[METHOD] <init> [TYPE] MatchStrength) [PARAMETER] InputStream in byte[] buffered int bufferedStart int bufferedLength JsonFactory match MatchStrength strength [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^public boolean hasMatch (  )  { return _match == null; }^64^^^^^^^[REPLACE] public boolean hasMatch (  )  { return _match != null; }^[METHOD] hasMatch [TYPE] boolean [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^return   MatchStrength.INCONCLUSIVE ;^71^^^^^70^72^[REPLACE] return  ( _matchStrength == null )  ? MatchStrength.INCONCLUSIVE : _matchStrength;^[METHOD] getMatchStrength [TYPE] MatchStrength [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^public boolean hasMatch (  )  { return _match != null; }^77^^^^^^^[REPLACE] public JsonFactory getMatch (  )  { return _match; }^[METHOD] getMatch [TYPE] JsonFactory [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^return _match.createParser ( getDataStream (  )  ) ;^87^^^^^86^88^[REPLACE] return _match.getFormatName (  ) ;^[METHOD] getMatchedFormatName [TYPE] String [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^if  ( _match != this  )  {^102^^^^^101^109^[REPLACE] if  ( _match == null )  {^[METHOD] createParserWithMatch [TYPE] JsonParser [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[ADD]^return null;^102^103^104^^^101^109^[ADD] if  ( _match == null )  { return null; }^[METHOD] createParserWithMatch [TYPE] JsonParser [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^return true;^103^^^^^101^109^[REPLACE] return null;^[METHOD] createParserWithMatch [TYPE] JsonParser [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^return false;^103^^^^^101^109^[REPLACE] return null;^[METHOD] createParserWithMatch [TYPE] JsonParser [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^if  ( _originalStream != null )  {^105^^^^^101^109^[REPLACE] if  ( _originalStream == null )  {^[METHOD] createParserWithMatch [TYPE] JsonParser [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[ADD]^^105^106^107^^^101^109^[ADD] if  ( _originalStream == null )  { return _match.createParser ( _bufferedData, _bufferedStart, _bufferedLength ) ; }^[METHOD] createParserWithMatch [TYPE] JsonParser [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^return _match.getFormatName (  ) ;^106^^^^^101^109^[REPLACE] return _match.createParser ( _bufferedData, _bufferedStart, _bufferedLength ) ;^[METHOD] createParserWithMatch [TYPE] JsonParser [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^return _match .createParser ( in )  ;^106^^^^^101^109^[REPLACE] return _match.createParser ( _bufferedData, _bufferedStart, _bufferedLength ) ;^[METHOD] createParserWithMatch [TYPE] JsonParser [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^return  ( _matchStrength == null )  ? MatchStrength.INCONCLUSIVE : _matchStrength;^108^^^^^101^109^[REPLACE] return _match.createParser ( getDataStream (  )  ) ;^[METHOD] createParserWithMatch [TYPE] JsonParser [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^if  ( _originalStream != null )  {^119^^^^^118^123^[REPLACE] if  ( _originalStream == null )  {^[METHOD] getDataStream [TYPE] InputStream [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[ADD]^^119^120^121^^^118^123^[ADD] if  ( _originalStream == null )  { return new ByteArrayInputStream ( _bufferedData, _bufferedStart, _bufferedLength ) ; }^[METHOD] getDataStream [TYPE] InputStream [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^return new MergedStream ( null, _originalStream, _bufferedData, _bufferedStart, _bufferedLength ) ;^120^^^^^118^123^[REPLACE] return new ByteArrayInputStream ( _bufferedData, _bufferedStart, _bufferedLength ) ;^[METHOD] getDataStream [TYPE] InputStream [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 
[REPLACE]^return new MergedStream ( true, _originalStream, _bufferedData, _bufferedStart, _bufferedLength ) ;^122^^^^^118^123^[REPLACE] return new MergedStream ( null, _originalStream, _bufferedData, _bufferedStart, _bufferedLength ) ;^[METHOD] getDataStream [TYPE] InputStream [PARAMETER] [CLASS] DataFormatMatcher   [TYPE]  byte[] _bufferedData  buffered  [TYPE]  JsonFactory _match  match  [TYPE]  MatchStrength _matchStrength  strength  [TYPE]  boolean false  true  [TYPE]  InputStream _originalStream  in  [TYPE]  int _bufferedLength  _bufferedStart  bufferedLength  bufferedStart 