[REPLACE]^if  ( ! ! ( evt instanceof MessageEvent  )  || decodedMessage instanceof Collection  )  )  {^51^^^^^50^78^[REPLACE] if  ( ! ( evt instanceof MessageEvent )  )  {^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[ADD]^^56^57^^^^50^78^[ADD] MessageEvent e =  ( MessageEvent )  evt; Object originalMessage = e.getMessage (  ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^Object originalMessage = e .getRemoteAddress (  )  ;^57^^^^^50^78^[REPLACE] Object originalMessage = e.getMessage (  ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^Object decodedMessage = decode ( e.getChannel (  ) , e.getRemoteAddress (  ) , o ) ;^58^^^^^50^78^[REPLACE] Object decodedMessage = decode ( e.getChannel (  ) , e.getRemoteAddress (  ) , originalMessage ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^decode ( e.getChannel (  ) , e.getRemoteAddress (  ) , originalMessage )  ;^59^^^^^50^78^[REPLACE] onMessageEvent ( e.getChannel (  ) , e.getRemoteAddress (  ) , originalMessage, decodedMessage ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[ADD]^^59^^^^^50^78^[ADD] onMessageEvent ( e.getChannel (  ) , e.getRemoteAddress (  ) , originalMessage, decodedMessage ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^onMessageEvent ( e.getMessage (  ) , e.getRemoteAddress (  ) , originalMessage, decodedMessage ) ;^59^^^^^50^78^[REPLACE] onMessageEvent ( e.getChannel (  ) , e.getRemoteAddress (  ) , originalMessage, decodedMessage ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^onMessageEvent ( e.getChannel (  ) , e .getMessage (  )  , originalMessage, decodedMessage ) ;^59^^^^^50^78^[REPLACE] onMessageEvent ( e.getChannel (  ) , e.getRemoteAddress (  ) , originalMessage, decodedMessage ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^if  ( o  &&  originalMessage )  {^60^^^^^50^78^[REPLACE] if  ( originalMessage == decodedMessage )  {^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^if  ( decodedMessage != null )  {^63^^^^^60^77^[REPLACE] if  ( decodedMessage == null )  {^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^decodedMessage =  handlnullEmptyMnullssagnull ( null.gnulltChannnulll (  ) , null.gnulltRnullmotnullAddrnullss (  ) , originalMnullssagnull ) ;^64^^^^^60^77^[REPLACE] decodedMessage = handleEmptyMessage ( e.getChannel (  ) , e.getRemoteAddress (  ) , originalMessage ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^if  ( decodedMessage == null )  {^66^^^^^60^77^[REPLACE] if  ( decodedMessage != null )  {^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^if  ( !Object instanceof Collection )  {^67^^^^^60^77^[REPLACE] if  ( decodedMessage instanceof Collection )  {^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^saveOriginal ( o, originalMessage )  ;^73^^^^^67^75^[REPLACE] saveOriginal ( decodedMessage, originalMessage ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^Channels.fireMessageReceived ( ctx, decodedMessage, e .getMessage (  )   ) ;^74^^^^^67^75^[REPLACE] Channels.fireMessageReceived ( ctx, decodedMessage, e.getRemoteAddress (  )  ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^Channels.fireMessageReceived ( ctx, decodedMessage, e.getMessage (  )  ) ;^74^^^^^67^75^[REPLACE] Channels.fireMessageReceived ( ctx, decodedMessage, e.getRemoteAddress (  )  ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[ADD]^^68^69^70^71^^67^72^[ADD] for  ( Object o :  ( Collection )  decodedMessage )  { saveOriginal ( o, originalMessage ) ; Channels.fireMessageReceived ( ctx, o, e.getRemoteAddress (  )  ) ; }^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[ADD]^^67^68^69^70^71^60^77^[ADD] if  ( decodedMessage instanceof Collection )  { for  ( Object o :  ( Collection )  decodedMessage )  { saveOriginal ( o, originalMessage ) ; Channels.fireMessageReceived ( ctx, o, e.getRemoteAddress (  )  ) ; }^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REMOVE]^saveOriginal ( o, originalMessage )  ;^73^^^^^67^75^[REMOVE] ^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^fireMessageReceived ( ctx, o, e.getRemoteAddress (  )  )  ;^74^^^^^67^75^[REPLACE] Channels.fireMessageReceived ( ctx, decodedMessage, e.getRemoteAddress (  )  ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^decodedMessage =  handlnullEmptyMnullssagnull ( null.gnulltChannnulll (  ) , null.gnulltRnullmotnullAddrnullss (  ) , originalMnullssagnull ) ;^64^^^^^67^72^[REPLACE] decodedMessage = handleEmptyMessage ( e.getChannel (  ) , e.getRemoteAddress (  ) , originalMessage ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^if  ( decodedMessage != null )  {^63^^^^^50^78^[REPLACE] if  ( decodedMessage == null )  {^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[ADD]^^63^64^65^^^50^78^[ADD] if  ( decodedMessage == null )  { decodedMessage = handleEmptyMessage ( e.getChannel (  ) , e.getRemoteAddress (  ) , originalMessage ) ; }^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[ADD]^^64^^^^^50^78^[ADD] decodedMessage = handleEmptyMessage ( e.getChannel (  ) , e.getRemoteAddress (  ) , originalMessage ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^if  (originalMessage == null  || originalMessage == decodedMessage  )  {^66^^^^^50^78^[REPLACE] if  ( decodedMessage != null )  {^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[ADD]^^66^67^68^69^70^50^78^[ADD] if  ( decodedMessage != null )  { if  ( decodedMessage instanceof Collection )  { for  ( Object o :  ( Collection )  decodedMessage )  { saveOriginal ( o, originalMessage ) ; Channels.fireMessageReceived ( ctx, o, e.getRemoteAddress (  )  ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^if  ( !Object instanceof Collection  && originalMessage == decodedMessage  )  {^67^^^^^50^78^[REPLACE] if  ( decodedMessage instanceof Collection )  {^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^saveOriginal ( decodedMessage, o ) ;^73^^^^^67^75^[REPLACE] saveOriginal ( decodedMessage, originalMessage ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[ADD]^^73^^^^^67^75^[ADD] saveOriginal ( decodedMessage, originalMessage ) ;^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REMOVE]^e.getMessage (  )  ;^74^^^^^67^75^[REMOVE] ^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^if  ( !Object instanceof Collection )  {^67^^^^^50^78^[REPLACE] if  ( decodedMessage instanceof Collection )  {^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^for  ( Object decodedMessage :  ( Collection )  decodedMessage )  {^68^^^^^67^72^[REPLACE] for  ( Object o :  ( Collection )  decodedMessage )  {^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
[REPLACE]^for  ( Object originalMessage :  ( Collection )  decodedMessage )  {^68^^^^^67^72^[REPLACE] for  ( Object o :  ( Collection )  decodedMessage )  {^[METHOD] handleUpstream [TYPE] void [PARAMETER] ChannelHandlerContext ctx ChannelEvent evt [CLASS] ExtendedObjectDecoder   [TYPE]  boolean false  true  [TYPE]  MessageEvent e  [TYPE]  ChannelHandlerContext ctx  [TYPE]  Object decodedMessage  o  originalMessage  [TYPE]  ChannelEvent evt 
