[REPLACE]^private  AbstractCompiler compiler;^29^^^^^^^[REPLACE] private final AbstractCompiler compiler;^ [CLASS] PotentialCheck  
[REPLACE]^this.compiler =  null;^33^^^^^32^35^[REPLACE] this.compiler = compiler;^[METHOD] <init> [TYPE] JSError) [PARAMETER] AbstractCompiler compiler JSError err [CLASS] PotentialCheck   [TYPE]  AbstractCompiler compiler  [TYPE]  JSError err  [TYPE]  boolean false  true 
[REPLACE]^this.err =  null;^34^^^^^32^35^[REPLACE] this.err = err;^[METHOD] <init> [TYPE] JSError) [PARAMETER] AbstractCompiler compiler JSError err [CLASS] PotentialCheck   [TYPE]  AbstractCompiler compiler  [TYPE]  JSError err  [TYPE]  boolean false  true 
[REPLACE]^compiler .report (  )  ;^41^^^^^40^42^[REPLACE] compiler.report ( err ) ;^[METHOD] report [TYPE] void [PARAMETER] [CLASS] PotentialCheck   [TYPE]  AbstractCompiler compiler  [TYPE]  JSError err  [TYPE]  boolean false  true 
[REPLACE]^compiler.report ( err ) ;^49^^^^^47^51^[REPLACE] report (  ) ;^[METHOD] evaluate [TYPE] void [PARAMETER] [CLASS] PotentialCheck   [TYPE]  AbstractCompiler compiler  [TYPE]  JSError err  [TYPE]  boolean false  true 