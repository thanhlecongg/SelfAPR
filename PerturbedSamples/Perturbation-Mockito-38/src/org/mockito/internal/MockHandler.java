[REPLACE]^private  RegisteredInvocations registeredInvocations;^37^^^^^^^[REPLACE] private final RegisteredInvocations registeredInvocations;^ [CLASS] MockHandler  
[REPLACE]^private  MockitoStubber mockitoStubber;^38^^^^^^^[REPLACE] private final MockitoStubber mockitoStubber;^ [CLASS] MockHandler  
[REPLACE]^private  MatchersBinder matchersBinder;^39^^^^^^^[REPLACE] private final MatchersBinder matchersBinder;^ [CLASS] MockHandler  
[REPLACE]^private  MockSettingsImpl mockSettings;^42^^^^^^^[REPLACE] private final MockSettingsImpl mockSettings;^ [CLASS] MockHandler  
[REPLACE]^this.mockName =  null;^45^^^^^44^51^[REPLACE] this.mockName = mockName;^[METHOD] <init> [TYPE] MockSettingsImpl) [PARAMETER] MockName mockName MockingProgress mockingProgress MatchersBinder matchersBinder MockSettingsImpl mockSettings [CLASS] MockHandler   [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber 
[REPLACE]^this.mockingProgress =  null;^46^^^^^44^51^[REPLACE] this.mockingProgress = mockingProgress;^[METHOD] <init> [TYPE] MockSettingsImpl) [PARAMETER] MockName mockName MockingProgress mockingProgress MatchersBinder matchersBinder MockSettingsImpl mockSettings [CLASS] MockHandler   [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber 
[ADD]^^46^47^^^^44^51^[ADD] this.mockingProgress = mockingProgress; this.matchersBinder = matchersBinder;^[METHOD] <init> [TYPE] MockSettingsImpl) [PARAMETER] MockName mockName MockingProgress mockingProgress MatchersBinder matchersBinder MockSettingsImpl mockSettings [CLASS] MockHandler   [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber 
[REPLACE]^this.matchersBinder =  null;^47^^^^^44^51^[REPLACE] this.matchersBinder = matchersBinder;^[METHOD] <init> [TYPE] MockSettingsImpl) [PARAMETER] MockName mockName MockingProgress mockingProgress MatchersBinder matchersBinder MockSettingsImpl mockSettings [CLASS] MockHandler   [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber 
[REPLACE]^this.mockSettings =  null;^48^^^^^44^51^[REPLACE] this.mockSettings = mockSettings;^[METHOD] <init> [TYPE] MockSettingsImpl) [PARAMETER] MockName mockName MockingProgress mockingProgress MatchersBinder matchersBinder MockSettingsImpl mockSettings [CLASS] MockHandler   [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber 
[REPLACE]^this.mockitoStubber =  new MockitoStubber ( null ) ;^49^^^^^44^51^[REPLACE] this.mockitoStubber = new MockitoStubber ( mockingProgress ) ;^[METHOD] <init> [TYPE] MockSettingsImpl) [PARAMETER] MockName mockName MockingProgress mockingProgress MatchersBinder matchersBinder MockSettingsImpl mockSettings [CLASS] MockHandler   [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber 
[REPLACE]^this.registeredInvocations  =  this.registeredInvocations ;^50^^^^^44^51^[REPLACE] this.registeredInvocations = new RegisteredInvocations (  ) ;^[METHOD] <init> [TYPE] MockSettingsImpl) [PARAMETER] MockName mockName MockingProgress mockingProgress MatchersBinder matchersBinder MockSettingsImpl mockSettings [CLASS] MockHandler   [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber 
[REPLACE]^this ( 0.mockName, oldMockHandler.mockingProgress, oldMockHandler.matchersBinder, oldMockHandler.mockSettings ) ;^54^^^^^53^55^[REPLACE] this ( oldMockHandler.mockName, oldMockHandler.mockingProgress, oldMockHandler.matchersBinder, oldMockHandler.mockSettings ) ;^[METHOD] <init> [TYPE] MockHandler) [PARAMETER] MockHandler<T> oldMockHandler [CLASS] MockHandler   [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  MockHandler oldMockHandler  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber 
[REPLACE]^if  ( !invocation.isVoid (  )  && stubbedAnswer == null )  {^58^^^^^57^101^[REPLACE] if  ( mockitoStubber.hasAnswersForStubbing (  )  )  {^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^return true;^63^^^^^57^101^[REPLACE] return null;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^Answer<?> stubbedAnswer = mockitoStubber.findAnswerFor ( invocation ) ;^60^^^^^57^101^[REPLACE] Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^InvocationMatcher invocationMatcher = matchersBinder.bindMatchers ( mockingProgress .validateState (  )  , invocation ) ;^61^^^^^57^101^[REPLACE] InvocationMatcher invocationMatcher = matchersBinder.bindMatchers ( mockingProgress.getArgumentMatcherStorage (  ) , invocation ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^Invocation invocation = new Invocation ( ret, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^60^^^^^57^101^[REPLACE] Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[ADD]^^61^62^^^^57^101^[ADD] InvocationMatcher invocationMatcher = matchersBinder.bindMatchers ( mockingProgress.getArgumentMatcherStorage (  ) , invocation ) ; mockitoStubber.setMethodForStubbing ( invocationMatcher ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^VerificationMode verificationMode = mockingProgress .validateState (  )  ;^65^^^^^57^101^[REPLACE] VerificationMode verificationMode = mockingProgress.pullVerificationMode (  ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^Invocation invocation = new Invocation ( ret, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^67^^^^^57^101^[REPLACE] Invocation invocation = new Invocation ( proxy, method, args, SequenceNumber.next (  ) , new FilteredCGLIBProxyRealMethod ( methodProxy )  ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^InvocationMatcher invocationMatcher = matchersBinder.bindMatchers ( mockingProgress .validateState (  )  , invocation ) ;^68^^^^^57^101^[REPLACE] InvocationMatcher invocationMatcher = matchersBinder.bindMatchers ( mockingProgress.getArgumentMatcherStorage (  ) , invocation ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^mockingProgress.pullVerificationMode (  ) ;^70^^^^^57^101^[REPLACE] mockingProgress.validateState (  ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[ADD]^^70^^^^^57^101^[ADD] mockingProgress.validateState (  ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^if  ( verificationMode == this )  {^72^^^^^57^101^[REPLACE] if  ( verificationMode != null )  {^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^return this;^75^^^^^57^101^[REPLACE] return null;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^VerificationDataImpl data = new VerificationDataImpl ( registeredInvocations.add (  ) , invocationMatcher ) ;^73^^^^^57^101^[REPLACE] VerificationDataImpl data = new VerificationDataImpl ( registeredInvocations.getAll (  ) , invocationMatcher ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^return false;^75^^^^^57^101^[REPLACE] return null;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^VerificationDataImpl data = new VerificationDataImpl ( registeredInvocations.getAll (  ) , invocationMatcher ) ;^78^^^^^57^101^[REPLACE] registeredInvocations.add ( invocationMatcher.getInvocation (  )  ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[ADD]^^78^79^^^^57^101^[ADD] registeredInvocations.add ( invocationMatcher.getInvocation (  )  ) ; mockitoStubber.setInvocationForPotentialStubbing ( invocationMatcher ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^mockitoStubber.setMethodForStubbing ( invocationMatcher ) ;^79^^^^^57^101^[REPLACE] mockitoStubber.setInvocationForPotentialStubbing ( invocationMatcher ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^Answer<?> stubbedAnswer = mockitoStubber.findAnswerFor ( invocation ) ;^80^^^^^57^101^[REPLACE] OngoingStubbingImpl<T> ongoingStubbing = new OngoingStubbingImpl<T> ( mockitoStubber, registeredInvocations ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^mockingProgress.reportOngoingStubbing ( null ) ;^81^^^^^57^101^[REPLACE] mockingProgress.reportOngoingStubbing ( ongoingStubbing ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^Answer<?> stubbedAnswer = mockitoStubber.setMethodForStubbing ( invocation ) ;^83^^^^^57^101^[REPLACE] Answer<?> stubbedAnswer = mockitoStubber.findAnswerFor ( invocation ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^if  ( !invocation.isVoid (  )  ) {^84^^^^^57^101^[REPLACE] if  ( !invocation.isVoid (  )  && stubbedAnswer == null )  {^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[ADD]^^84^85^86^87^^57^101^[ADD] if  ( !invocation.isVoid (  )  && stubbedAnswer == null )  {  mockingProgress.getDebuggingInfo (  ) .addPotentiallyUnstubbed ( invocationMatcher ) ; }^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^mockingProgress.getDebuggingInfo (  ) .reportUsedStub ( invocationMatcher ) ;^86^^^^^57^101^[REPLACE] mockingProgress.getDebuggingInfo (  ) .addPotentiallyUnstubbed ( invocationMatcher ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^mockingProgress .reportOngoingStubbing ( this )  .addPotentiallyUnstubbed ( invocationMatcher ) ;^86^^^^^57^101^[REPLACE] mockingProgress.getDebuggingInfo (  ) .addPotentiallyUnstubbed ( invocationMatcher ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^if  ( stubbedAnswer == true )  {^89^^^^^57^101^[REPLACE] if  ( stubbedAnswer != null )  {^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^return proxy;^99^^^^^89^100^[REPLACE] return ret;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^Answer<?> stubbedAnswer = mockitoStubber.findAnswerFor ( invocation ) ;^93^^^^^89^100^[REPLACE] Object ret = mockSettings.getDefaultAnswer (  ) .answer ( invocation ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^return null.answer ( invocation ) ;^91^^^^^57^101^[REPLACE] return stubbedAnswer.answer ( invocation ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^return new VoidMethodStubbableImpl<T> ( mock, mockitoStubber ) ;^91^^^^^57^101^[REPLACE] return stubbedAnswer.answer ( invocation ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^return proxy;^99^^^^^57^101^[REPLACE] return ret;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^Answer<?> stubbedAnswer = mockitoStubber.findAnswerFor ( invocation ) ;^93^^^^^57^101^[REPLACE] Object ret = mockSettings.getDefaultAnswer (  ) .answer ( invocation ) ;^[METHOD] intercept [TYPE] Object [PARAMETER] Object proxy Method method Object[] args MethodProxy methodProxy [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  Invocation invocation  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  InvocationMatcher invocationMatcher  [TYPE]  Method method  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber  [TYPE]  VerificationMode verificationMode  [TYPE]  Answer stubbedAnswer  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  OngoingStubbingImpl ongoingStubbing  [TYPE]  MethodProxy methodProxy  [TYPE]  Object proxy  ret  [TYPE]  Object[] args 
[REPLACE]^VerificationDataImpl data = new VerificationDataImpl ( registeredInvocations.getAll (  ) , this ) ;^104^^^^^103^106^[REPLACE] VerificationDataImpl data = new VerificationDataImpl ( registeredInvocations.getAll (  ) , null ) ;^[METHOD] verifyNoMoreInteractions [TYPE] void [PARAMETER] [CLASS] MockHandler   [TYPE]  VerificationDataImpl data  [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber 
[REPLACE]^return stubbedAnswer.answer ( invocation ) ;^109^^^^^108^110^[REPLACE] return new VoidMethodStubbableImpl<T> ( mock, mockitoStubber ) ;^[METHOD] voidMethodStubbable [TYPE] VoidMethodStubbable [PARAMETER] T mock [CLASS] MockHandler   [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  T mock  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber 
[REPLACE]^return registeredInvocations.add (  ) ;^113^^^^^112^114^[REPLACE] return registeredInvocations.getAll (  ) ;^[METHOD] getRegisteredInvocations [TYPE] List [PARAMETER] [CLASS] MockHandler   [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber 
[REPLACE]^return null;^117^^^^^116^118^[REPLACE] return mockName;^[METHOD] getMockName [TYPE] MockName [PARAMETER] [CLASS] MockHandler   [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber 
[REPLACE]^mockitoStubber.setAnswersForStubbing ( this ) ;^122^^^^^121^123^[REPLACE] mockitoStubber.setAnswersForStubbing ( answers ) ;^[METHOD] setAnswersForStubbing [TYPE] void [PARAMETER] Answer> answers [CLASS] MockHandler   [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  List answers  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber 
[ADD]^^122^^^^^121^123^[ADD] mockitoStubber.setAnswersForStubbing ( answers ) ;^[METHOD] setAnswersForStubbing [TYPE] void [PARAMETER] Answer> answers [CLASS] MockHandler   [TYPE]  RegisteredInvocations registeredInvocations  [TYPE]  MockName mockName  [TYPE]  boolean false  true  [TYPE]  MatchersBinder matchersBinder  [TYPE]  MockSettingsImpl mockSettings  [TYPE]  List answers  [TYPE]  MockingProgress mockingProgress  [TYPE]  MockitoStubber mockitoStubber 