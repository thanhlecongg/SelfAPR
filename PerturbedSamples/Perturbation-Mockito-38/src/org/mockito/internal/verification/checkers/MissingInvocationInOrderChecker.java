[REPLACE]^private  InvocationsFinder finder;^18^^^^^^^[REPLACE] private final InvocationsFinder finder;^ [CLASS] MissingInvocationInOrderChecker  
[REPLACE]^List<Invocation> chunk = finder.findAllMatchingUnverifiedChunks ( invocations, wanted ) ;^21^^^^^20^22^[REPLACE] this ( new InvocationsFinder (  ) , new Reporter (  )  ) ;^[METHOD] <init> [TYPE] MissingInvocationInOrderChecker() [PARAMETER] [CLASS] MissingInvocationInOrderChecker   [TYPE]  InvocationsFinder finder  [TYPE]  Reporter reporter  [TYPE]  boolean false  true 
[REPLACE]^this.finder =  null;^25^^^^^24^27^[REPLACE] this.finder = finder;^[METHOD] <init> [TYPE] Reporter) [PARAMETER] InvocationsFinder finder Reporter reporter [CLASS] MissingInvocationInOrderChecker   [TYPE]  InvocationsFinder finder  [TYPE]  Reporter reporter  [TYPE]  boolean false  true 
[REPLACE]^this.reporter =  null;^26^^^^^24^27^[REPLACE] this.reporter = reporter;^[METHOD] <init> [TYPE] Reporter) [PARAMETER] InvocationsFinder finder Reporter reporter [CLASS] MissingInvocationInOrderChecker   [TYPE]  InvocationsFinder finder  [TYPE]  Reporter reporter  [TYPE]  boolean false  true 
[ADD]^^26^^^^^24^27^[ADD] this.reporter = reporter;^[METHOD] <init> [TYPE] Reporter) [PARAMETER] InvocationsFinder finder Reporter reporter [CLASS] MissingInvocationInOrderChecker   [TYPE]  InvocationsFinder finder  [TYPE]  Reporter reporter  [TYPE]  boolean false  true 
[REPLACE]^List<Invocation> chunk = finder.findAllMatchingUnverifiedChunks ( 1, wanted ) ;^30^^^^^29^42^[REPLACE] List<Invocation> chunk = finder.findAllMatchingUnverifiedChunks ( invocations, wanted ) ;^[METHOD] check [TYPE] void [PARAMETER] Invocation> invocations InvocationMatcher wanted VerificationMode mode [CLASS] MissingInvocationInOrderChecker   [TYPE]  boolean false  true  [TYPE]  Invocation previousInOrder  [TYPE]  InvocationsFinder finder  [TYPE]  Reporter reporter  [TYPE]  InvocationMatcher wanted  [TYPE]  List chunk  invocations  [TYPE]  VerificationMode mode 
[REPLACE]^if  ( chunk.isEmpty (  )  )  {^32^^^^^29^42^[REPLACE] if  ( !chunk.isEmpty (  )  )  {^[METHOD] check [TYPE] void [PARAMETER] Invocation> invocations InvocationMatcher wanted VerificationMode mode [CLASS] MissingInvocationInOrderChecker   [TYPE]  boolean false  true  [TYPE]  Invocation previousInOrder  [TYPE]  InvocationsFinder finder  [TYPE]  Reporter reporter  [TYPE]  InvocationMatcher wanted  [TYPE]  List chunk  invocations  [TYPE]  VerificationMode mode 
[ADD]^^32^33^34^^^29^42^[ADD] if  ( !chunk.isEmpty (  )  )  { return; }^[METHOD] check [TYPE] void [PARAMETER] Invocation> invocations InvocationMatcher wanted VerificationMode mode [CLASS] MissingInvocationInOrderChecker   [TYPE]  boolean false  true  [TYPE]  Invocation previousInOrder  [TYPE]  InvocationsFinder finder  [TYPE]  Reporter reporter  [TYPE]  InvocationMatcher wanted  [TYPE]  List chunk  invocations  [TYPE]  VerificationMode mode 
[REPLACE]^Invocation previousInOrder = finder.findPreviousVerifiedInOrder ( this ) ;^36^^^^^29^42^[REPLACE] Invocation previousInOrder = finder.findPreviousVerifiedInOrder ( invocations ) ;^[METHOD] check [TYPE] void [PARAMETER] Invocation> invocations InvocationMatcher wanted VerificationMode mode [CLASS] MissingInvocationInOrderChecker   [TYPE]  boolean false  true  [TYPE]  Invocation previousInOrder  [TYPE]  InvocationsFinder finder  [TYPE]  Reporter reporter  [TYPE]  InvocationMatcher wanted  [TYPE]  List chunk  invocations  [TYPE]  VerificationMode mode 
[REPLACE]^if  ( previousInOrder != false )  {^37^^^^^29^42^[REPLACE] if  ( previousInOrder == null )  {^[METHOD] check [TYPE] void [PARAMETER] Invocation> invocations InvocationMatcher wanted VerificationMode mode [CLASS] MissingInvocationInOrderChecker   [TYPE]  boolean false  true  [TYPE]  Invocation previousInOrder  [TYPE]  InvocationsFinder finder  [TYPE]  Reporter reporter  [TYPE]  InvocationMatcher wanted  [TYPE]  List chunk  invocations  [TYPE]  VerificationMode mode 
[REPLACE]^reporter .wantedButNotInvoked ( wanted )  ;^40^^^^^29^42^[REPLACE] reporter.wantedButNotInvokedInOrder ( wanted, previousInOrder ) ;^[METHOD] check [TYPE] void [PARAMETER] Invocation> invocations InvocationMatcher wanted VerificationMode mode [CLASS] MissingInvocationInOrderChecker   [TYPE]  boolean false  true  [TYPE]  Invocation previousInOrder  [TYPE]  InvocationsFinder finder  [TYPE]  Reporter reporter  [TYPE]  InvocationMatcher wanted  [TYPE]  List chunk  invocations  [TYPE]  VerificationMode mode 
[REPLACE]^reporter.wantedButNotInvokedInOrder ( wanted, previousInOrder ) ;^38^^^^^29^42^[REPLACE] reporter.wantedButNotInvoked ( wanted ) ;^[METHOD] check [TYPE] void [PARAMETER] Invocation> invocations InvocationMatcher wanted VerificationMode mode [CLASS] MissingInvocationInOrderChecker   [TYPE]  boolean false  true  [TYPE]  Invocation previousInOrder  [TYPE]  InvocationsFinder finder  [TYPE]  Reporter reporter  [TYPE]  InvocationMatcher wanted  [TYPE]  List chunk  invocations  [TYPE]  VerificationMode mode 