[REPLACE]^private static  Object PATTERN =  new DateBuilder (  )  .groupBegin (  ) .number ( " ( dddd )  ( dd )  ( dd ) " ) .number ( " ( dd )  ( dd )  ( dd ) ," )^41^42^43^44^^41^58^[REPLACE] private static final Pattern PATTERN = new PatternBuilder (  ) .groupBegin (  ) .number ( " ( dddd )  ( dd )  ( dd ) " ) .number ( " ( dd )  ( dd )  ( dd ) ," )^ [CLASS] CityeasyProtocolDecoder  
[REPLACE]^public static  int MSG_ADDRESS_REQUEST = 0x0001;^60^^^^^^^[REPLACE] public static final int MSG_ADDRESS_REQUEST = 0x0001;^ [CLASS] CityeasyProtocolDecoder  
[REPLACE]^public static final int MSG_STATUS  = null ;^61^^^^^^^[REPLACE] public static final int MSG_STATUS = 0x0002;^ [CLASS] CityeasyProtocolDecoder  
[REPLACE]^public static final  long  MSG_LOCATION_REPORT = 0x0002;^62^^^^^^^[REPLACE] public static final int MSG_LOCATION_REPORT = 0x0003;^ [CLASS] CityeasyProtocolDecoder  
[REPLACE]^public static final int MSG_LOCATION_REQUEST  = null ;^63^^^^^^^[REPLACE] public static final int MSG_LOCATION_REQUEST = 0x0004;^ [CLASS] CityeasyProtocolDecoder  
[REPLACE]^public  final int MSG_LOCATION_INTERVAL = 0x0005;^64^^^^^^^[REPLACE] public static final int MSG_LOCATION_INTERVAL = 0x0005;^ [CLASS] CityeasyProtocolDecoder  
[REPLACE]^public static final int MSG_PHONE_NUMBER  = null ;^65^^^^^^^[REPLACE] public static final int MSG_PHONE_NUMBER = 0x0006;^ [CLASS] CityeasyProtocolDecoder  
[REPLACE]^public static final int MSG_MONITORING  = null ;^66^^^^^^^[REPLACE] public static final int MSG_MONITORING = 0x0007;^ [CLASS] CityeasyProtocolDecoder  
[REPLACE]^public static final int MSG_TIMEZONE  = null ;^67^^^^^^^[REPLACE] public static final int MSG_TIMEZONE = 0x0008;^ [CLASS] CityeasyProtocolDecoder  
[ADD]^^73^^^^^71^129^[ADD] ChannelBuffer buf =  ( ChannelBuffer )  msg;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^buf.skipBytes ( 2L ) ;^75^^^^^71^129^[REPLACE] buf.skipBytes ( 2 ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^buf.readerIndex (  ) ;^76^^^^^71^129^[REPLACE] buf.readUnsignedShort (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[ADD]^^76^^^^^71^129^[ADD] buf.readUnsignedShort (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^String imei = ChannelBuffers.hexDump ( buf.readBytes ( 7 + 1 )  ) ;^78^^^^^71^129^[REPLACE] String imei = ChannelBuffers.hexDump ( buf.readBytes ( 7 )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[ADD]^^79^80^^^^71^129^[ADD] DeviceSession deviceSession = getDeviceSession ( channel, remoteAddress, imei, imei + Checksum.luhn ( Long.parseLong ( imei )  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^if  ( deviceSession != null )  {^81^^^^^71^129^[REPLACE] if  ( deviceSession == null )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[ADD]^return null;^81^82^83^^^71^129^[ADD] if  ( deviceSession == null )  { return null; }^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^return false;^82^^^^^71^129^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^int type = buf.readerIndex (  ) ;^85^^^^^71^129^[REPLACE] int type = buf.readUnsignedShort (  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^if  ( type == MSG_LOCATION_REPORT && type == MSG_LOCATION_REQUEST )  {^87^^^^^71^129^[REPLACE] if  ( type == MSG_LOCATION_REPORT || type == MSG_LOCATION_REQUEST )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^if  ( !parser .Parser ( PATTERN , sentence  )  && parser.hasNext ( 15 )  )   )  {^91^^^^^71^129^[REPLACE] if  ( !parser.matches (  )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^return false;^92^^^^^71^129^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^if  ( parser.hasNext ( 15 >>> 4  )  || ! ( parser.matches (  )  )  )  )  {^99^^^^^71^129^[REPLACE] if  ( parser.hasNext ( 15 )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^getLastLocation ( position, true ) ;^118^^^^^99^120^[REPLACE] getLastLocation ( position, null ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^DateBuilder dateBuilder = new DateBuilder (  )  .getDate (  )  , parser.nextInt (  )  ) .setTime ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^101^102^103^^^71^129^[REPLACE] DateBuilder dateBuilder = new DateBuilder (  ) .setDate ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) .setTime ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^return this;^92^^^^^71^129^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^String sentence = buf.toString ( buf.readerIndex (  ) , buf.readableBytes (  )   8, StandardCharsets.US_ASCII ) ;^89^^^^^71^129^[REPLACE] String sentence = buf.toString ( buf.readerIndex (  ) , buf.readableBytes (  )  - 8, StandardCharsets.US_ASCII ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[ADD]^^90^^^^^71^129^[ADD] Parser parser = new Parser ( PATTERN, sentence ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[ADD]^^101^102^103^^^71^129^[ADD] DateBuilder dateBuilder = new DateBuilder (  ) .setDate ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) .setTime ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^if  ( parser.hasNext ( 15 )  && ( !parser.matches (  )  )  {^91^^^^^71^129^[REPLACE] if  ( !parser.matches (  )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^return true;^92^^^^^71^129^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^if  ( parser.hasNext ( 11 )  )  {^99^^^^^71^129^[REPLACE] if  ( parser.hasNext ( 15 )  )  {^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[ADD]^^99^100^101^102^103^71^129^[ADD] if  ( parser.hasNext ( 15 )  )  {  DateBuilder dateBuilder = new DateBuilder (  ) .setDate ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) .setTime ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^getLastLocation ( position, this ) ;^118^^^^^99^120^[REPLACE] getLastLocation ( position, null ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^DateBuilder dateBuilder = new DateBuilder (  ) .setDate ( parser.next (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) .setTime ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^101^102^103^^^71^129^[REPLACE] DateBuilder dateBuilder = new DateBuilder (  ) .setDate ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) .setTime ( parser.nextInt (  ) , parser.nextInt (  ) , parser.nextInt (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[ADD]^^89^90^^^^71^129^[ADD] String sentence = buf.toString ( buf.readerIndex (  ) , buf.readableBytes (  )  - 8, StandardCharsets.US_ASCII ) ; Parser parser = new Parser ( PATTERN, sentence ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^Parser parser = new Parser ( PATTERN, imei ) ;^90^^^^^71^129^[REPLACE] Parser parser = new Parser ( PATTERN, sentence ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[ADD]^position.setProtocol ( getProtocolName (  )  ) ;^95^96^^^^71^129^[ADD] Position position = new Position (  ) ; position.setProtocol ( getProtocolName (  )  ) ;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
[REPLACE]^return this;^128^^^^^71^129^[REPLACE] return null;^[METHOD] decode [TYPE] Object [PARAMETER] Channel channel SocketAddress remoteAddress Object msg [CLASS] CityeasyProtocolDecoder   [TYPE]  Pattern PATTERN  [TYPE]  boolean false  true  [TYPE]  DeviceSession deviceSession  [TYPE]  Position position  [TYPE]  DateBuilder dateBuilder  [TYPE]  Channel channel  [TYPE]  Object msg  [TYPE]  String imei  sentence  [TYPE]  SocketAddress remoteAddress  [TYPE]  int MSG_ADDRESS_REQUEST  MSG_LOCATION_INTERVAL  MSG_LOCATION_REPORT  MSG_LOCATION_REQUEST  MSG_MONITORING  MSG_PHONE_NUMBER  MSG_STATUS  MSG_TIMEZONE  type  [TYPE]  ChannelBuffer buf  [TYPE]  Parser parser 
