[REPLACE]^public static final int MSG_CUT_OIL  = null ;^27^^^^^^^[REPLACE] public static final int MSG_CUT_OIL = 0x39;^ [CLASS] KhdProtocolEncoder  
[REPLACE]^public static final int MSG_RESUME_OIL  = null ;^28^^^^^^^[REPLACE] public static final int MSG_RESUME_OIL = 0x38;^ [CLASS] KhdProtocolEncoder  
[ADD]^^32^^^^^30^46^[ADD] ChannelBuffer buf = ChannelBuffers.dynamicBuffer (  ) ;^[METHOD] encodeCommand [TYPE] ChannelBuffer [PARAMETER] int command [CLASS] KhdProtocolEncoder   [TYPE]  int MSG_CUT_OIL  MSG_RESUME_OIL  command  [TYPE]  ChannelBuffer buf  [TYPE]  boolean false  true 
[REPLACE]^buf.writeByte ( 13 )  ;^34^^^^^30^46^[REPLACE] buf.writeByte ( 0x29 ) ;^[METHOD] encodeCommand [TYPE] ChannelBuffer [PARAMETER] int command [CLASS] KhdProtocolEncoder   [TYPE]  int MSG_CUT_OIL  MSG_RESUME_OIL  command  [TYPE]  ChannelBuffer buf  [TYPE]  boolean false  true 
[REPLACE]^buf.writeByte ( 13 )  ;^35^^^^^30^46^[REPLACE] buf.writeByte ( 0x29 ) ;^[METHOD] encodeCommand [TYPE] ChannelBuffer [PARAMETER] int command [CLASS] KhdProtocolEncoder   [TYPE]  int MSG_CUT_OIL  MSG_RESUME_OIL  command  [TYPE]  ChannelBuffer buf  [TYPE]  boolean false  true 
[REPLACE]^buf.writeInt ( MSG_RESUME_OIL ) ;^37^^^^^30^46^[REPLACE] buf.writeByte ( command ) ;^[METHOD] encodeCommand [TYPE] ChannelBuffer [PARAMETER] int command [CLASS] KhdProtocolEncoder   [TYPE]  int MSG_CUT_OIL  MSG_RESUME_OIL  command  [TYPE]  ChannelBuffer buf  [TYPE]  boolean false  true 
[REPLACE]^buf.writeInt ( 6L ) ;^38^^^^^30^46^[REPLACE] buf.writeShort ( 6 ) ;^[METHOD] encodeCommand [TYPE] ChannelBuffer [PARAMETER] int command [CLASS] KhdProtocolEncoder   [TYPE]  int MSG_CUT_OIL  MSG_RESUME_OIL  command  [TYPE]  ChannelBuffer buf  [TYPE]  boolean false  true 
[REPLACE]^buf.writeInt ( 1 ) ;^40^^^^^30^46^[REPLACE] buf.writeInt ( 0 ) ;^[METHOD] encodeCommand [TYPE] ChannelBuffer [PARAMETER] int command [CLASS] KhdProtocolEncoder   [TYPE]  int MSG_CUT_OIL  MSG_RESUME_OIL  command  [TYPE]  ChannelBuffer buf  [TYPE]  boolean false  true 
[REPLACE]^buf.writeByte ( Checksum.xor ( buf.writeByte (  )  )  ) ;^42^^^^^30^46^[REPLACE] buf.writeByte ( Checksum.xor ( buf.toByteBuffer (  )  )  ) ;^[METHOD] encodeCommand [TYPE] ChannelBuffer [PARAMETER] int command [CLASS] KhdProtocolEncoder   [TYPE]  int MSG_CUT_OIL  MSG_RESUME_OIL  command  [TYPE]  ChannelBuffer buf  [TYPE]  boolean false  true 
[ADD]^^42^^^^^30^46^[ADD] buf.writeByte ( Checksum.xor ( buf.toByteBuffer (  )  )  ) ;^[METHOD] encodeCommand [TYPE] ChannelBuffer [PARAMETER] int command [CLASS] KhdProtocolEncoder   [TYPE]  int MSG_CUT_OIL  MSG_RESUME_OIL  command  [TYPE]  ChannelBuffer buf  [TYPE]  boolean false  true 
[REPLACE]^buf.writeByte ( xor ( buf.toByteBuffer (  )  )  )  ;^42^^^^^30^46^[REPLACE] buf.writeByte ( Checksum.xor ( buf.toByteBuffer (  )  )  ) ;^[METHOD] encodeCommand [TYPE] ChannelBuffer [PARAMETER] int command [CLASS] KhdProtocolEncoder   [TYPE]  int MSG_CUT_OIL  MSG_RESUME_OIL  command  [TYPE]  ChannelBuffer buf  [TYPE]  boolean false  true 
[ADD]^^42^43^^^^30^46^[ADD] buf.writeByte ( Checksum.xor ( buf.toByteBuffer (  )  )  ) ; buf.writeByte ( 0x0D ) ;^[METHOD] encodeCommand [TYPE] ChannelBuffer [PARAMETER] int command [CLASS] KhdProtocolEncoder   [TYPE]  int MSG_CUT_OIL  MSG_RESUME_OIL  command  [TYPE]  ChannelBuffer buf  [TYPE]  boolean false  true 
[REPLACE]^buf.writeByte ( 41 )  ;^43^^^^^30^46^[REPLACE] buf.writeByte ( 0x0D ) ;^[METHOD] encodeCommand [TYPE] ChannelBuffer [PARAMETER] int command [CLASS] KhdProtocolEncoder   [TYPE]  int MSG_CUT_OIL  MSG_RESUME_OIL  command  [TYPE]  ChannelBuffer buf  [TYPE]  boolean false  true 
[REPLACE]^return null  ;^45^^^^^30^46^[REPLACE] return buf;^[METHOD] encodeCommand [TYPE] ChannelBuffer [PARAMETER] int command [CLASS] KhdProtocolEncoder   [TYPE]  int MSG_CUT_OIL  MSG_RESUME_OIL  command  [TYPE]  ChannelBuffer buf  [TYPE]  boolean false  true 
