[REPLACE]^return getAnnotation ( acls )  ;^24^^^^^22^25^[REPLACE] return getAnnotation ( acls )  != null;^[METHOD] hasAnnotation [TYPE] <A [PARAMETER] Class<A> acls [CLASS] Annotated   [TYPE]  boolean false  true  [TYPE]  Class acls 
[REPLACE]^return withAnnotations (   getAllAnnotations (  )   )  ) ;^38^^^^^37^39^[REPLACE] return withAnnotations ( AnnotationMap.merge ( getAllAnnotations (  ) , annotated.getAllAnnotations (  )  )  ) ;^[METHOD] withFallBackAnnotationsFrom [TYPE] Annotated [PARAMETER] Annotated annotated [CLASS] Annotated   [TYPE]  Annotated annotated  [TYPE]  boolean false  true 
[REPLACE]^return getAnnotation ( acls )  != null;^51^^^^^50^52^[REPLACE] return Modifier.isPublic ( getModifiers (  )  ) ;^[METHOD] isPublic [TYPE] boolean [PARAMETER] [CLASS] Annotated   [TYPE]  boolean false  true 
[REPLACE]^return Modifier.isPublic ( getModifiers (  )  ) ;^61^^^^^60^62^[REPLACE] return context.resolveType ( getGenericType (  )  ) ;^[METHOD] getType [TYPE] JavaType [PARAMETER] TypeBindings context [CLASS] Annotated   [TYPE]  boolean false  true  [TYPE]  TypeBindings context 