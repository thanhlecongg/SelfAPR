[REPLACE]^private static final Pattern PATTERN =  new DateBuilder (  )  .number ( " ( dd )  ( dd )  ( dd ) . ( ddd ) ," ) .number ( " ( dd )  ( dd.dddd )  ( [NS] ) ," ) .number ( " ( ddd )  ( dd.dddd )  ( [EW] ) ," )^35^36^37^38^^35^47^[REPLACE] private static final Pattern PATTERN = new PatternBuilder (  ) .number ( " ( dd )  ( dd )  ( dd ) . ( ddd ) ," ) .number ( " ( dd )  ( dd.dddd )  ( [NS] ) ," ) .number ( " ( ddd )  ( dd.dddd )  ( [EW] ) ," )^ [CLASS] TrackboxProtocolDecoder  
[ADD]^^59^^^^^57^107^[ADD] String sentence =  ( String )  msg;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^if  ( sentence.startsWith ( "a=connect"  )  || ! ( parser.matches (  )  )  )  )  {^61^^^^^57^107^[REPLACE] if  ( sentence.startsWith ( "a=connect" )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^if  ( getDeviceSession ( channel, remoteAddress, id )  == this )  {^63^^^^^57^107^[REPLACE] if  ( getDeviceSession ( channel, remoteAddress, id )  != null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^return true;^66^^^^^57^107^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^String id = sentence.substring ( sentence.indexOf ( "i=" )   2 ) ;^62^^^^^57^107^[REPLACE] String id = sentence.substring ( sentence.indexOf ( "i=" )  + 2 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^if  ( getDeviceSession ( channel, remoteAddress, id )  == false )  {^63^^^^^57^107^[REPLACE] if  ( getDeviceSession ( channel, remoteAddress, id )  != null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^return this;^66^^^^^57^107^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^if  ( deviceSession != null  && sentence.startsWith ( "a=connect" )  )  {^70^^^^^57^107^[REPLACE] if  ( deviceSession == null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[ADD]^return null;^70^71^72^^^57^107^[ADD] if  ( deviceSession == null )  { return null; }^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^return false;^71^^^^^57^107^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^return true;^71^^^^^57^107^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^if  ( sentence.startsWith ( "a=connect" )  || ( !parser .Parser ( PATTERN , sentence )   )  {^75^^^^^57^107^[REPLACE] if  ( !parser.matches (  )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^return this;^76^^^^^57^107^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[ADD]^^78^^^^^57^107^[ADD] sendResponse ( channel ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position.set ( getProtocolName (  )  ) ;^81^^^^^57^107^[REPLACE] position.setProtocol ( getProtocolName (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^DateBuilder dateBuilder = new DateBuilder (  ) .setTime ( parser.next (  ) , parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^84^85^^^^57^107^[REPLACE] DateBuilder dateBuilder = new DateBuilder (  ) .setTime ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position.setLongitude ( parser.nextCoordinate (  )  )  ;^87^^^^^57^107^[REPLACE] position.setLatitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position.setLatitude ( parser.nextInt (  )  ) ;^87^^^^^57^107^[REPLACE] position.setLatitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REMOVE]^parser.nextInt (  )  ;^87^^^^^57^107^[REMOVE] ^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position.setLatitude ( parser.nextCoordinate (  )  )  ;^88^^^^^57^107^[REPLACE] position.setLongitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position.setLongitude ( parser.nextInt (  )  ) ;^88^^^^^57^107^[REPLACE] position.setLongitude ( parser.nextCoordinate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position.set ( Position.KEY_HDOP, parser.nextInt (  )  ) ;^90^^^^^57^107^[REPLACE] position.set ( Position.KEY_HDOP, parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position .setLatitude (  )  ;^92^^^^^57^107^[REPLACE] position.setAltitude ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position.setAltitude ( parser.nextInt (  )  ) ;^92^^^^^57^107^[REPLACE] position.setAltitude ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[ADD]^^92^^^^^57^107^[ADD] position.setAltitude ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^int fix = parser .next (  )  ;^94^^^^^57^107^[REPLACE] int fix = parser.nextInt (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position.set ( Position.KEY_HDOP, parser.next (  )  )  ;^95^^^^^57^107^[REPLACE] position.set ( Position.KEY_GPS, fix ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position.setValid ( fix  !=  0 ) ;^96^^^^^57^107^[REPLACE] position.setValid ( fix > 0 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position .setTime (  )  ;^98^^^^^57^107^[REPLACE] position.setCourse ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^parser.nextInt (  )  ;^98^^^^^57^107^[REPLACE] position.setCourse ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position.set ( parser.nextDouble (  )  ) ;^99^^^^^57^107^[REPLACE] position.setSpeed ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position.setSpeed ( parser .nextCoordinate (  )   ) ;^99^^^^^57^107^[REPLACE] position.setSpeed ( parser.nextDouble (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^dateBuilder.setDateReverse ( parser.next (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^101^^^^^57^107^[REPLACE] dateBuilder.setDateReverse ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[ADD]^^101^102^^^^57^107^[ADD] dateBuilder.setDateReverse ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ; position.setTime ( dateBuilder.getDate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^dateBuilder.setDateReverse ( parser .next (  )  , parser^101^^^^^57^107^[REPLACE] dateBuilder.setDateReverse ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[ADD]^^101^^^^^57^107^[ADD] dateBuilder.setDateReverse ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position .getDate (  )  ;^102^^^^^57^107^[REPLACE] position.setTime ( dateBuilder.getDate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[ADD]^^102^^^^^57^107^[ADD] position.setTime ( dateBuilder.getDate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position.setTime ( dateBuilder .setTime (  )   ) ;^102^^^^^57^107^[REPLACE] position.setTime ( dateBuilder.getDate (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position.set ( Position.KEY_HDOP, parser.next (  )  )  ;^104^^^^^57^107^[REPLACE] position.set ( Position.KEY_SATELLITES, parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
[REPLACE]^position.set ( Position.KEY_SATELLITES, parser.nextInt (  )  ) ;^104^^^^^57^107^[REPLACE] position.set ( Position.KEY_SATELLITES, parser.next (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] TrackboxProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String id  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int fix  [TYPE]  Parser parser 
