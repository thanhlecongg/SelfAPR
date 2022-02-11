[REPLACE]^private  int _size;^15^^^^^^^[REPLACE] private final int _size;^ [CLASS] JsonSerializerMap Bucket  
[REPLACE]^public  JsonSerializer<Object> value;^81^^^^^^^[REPLACE] public final JsonSerializer<Object> value;^ [CLASS] JsonSerializerMap Bucket  
[REPLACE]^public  Bucket next;^82^^^^^^^[REPLACE] public final Bucket next;^ [CLASS] JsonSerializerMap Bucket  
[REPLACE]^int size = findSize ( this.size (  )  ) ;^19^^^^^17^29^[REPLACE] int size = findSize ( serializers.size (  )  ) ;^[METHOD] <init> [TYPE] Map) [PARAMETER] Object>> serializers [CLASS] JsonSerializerMap Bucket   [TYPE]  Entry entry  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  Map serializers  [TYPE]  int _size  hashMask  index  size 
[REPLACE]^_size =  index;^20^^^^^17^29^[REPLACE] _size = size;^[METHOD] <init> [TYPE] Map) [PARAMETER] Object>> serializers [CLASS] JsonSerializerMap Bucket   [TYPE]  Entry entry  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  Map serializers  [TYPE]  int _size  hashMask  index  size 
[REPLACE]^int hashMask = (size >= 0 );^21^^^^^17^29^[REPLACE] int hashMask =  ( size-1 ) ;^[METHOD] <init> [TYPE] Map) [PARAMETER] Object>> serializers [CLASS] JsonSerializerMap Bucket   [TYPE]  Entry entry  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  Map serializers  [TYPE]  int _size  hashMask  index  size 
[REPLACE]^Bucket bucket = _buckets[index];^22^^^^^17^29^[REPLACE] Bucket[] buckets = new Bucket[size];^[METHOD] <init> [TYPE] Map) [PARAMETER] Object>> serializers [CLASS] JsonSerializerMap Bucket   [TYPE]  Entry entry  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  Map serializers  [TYPE]  int _size  hashMask  index  size 
[REPLACE]^buckets[index]  =  buckets[index] ;^26^^^^^17^29^[REPLACE] buckets[index] = new Bucket ( buckets[index], key, entry.getValue (  )  ) ;^[METHOD] <init> [TYPE] Map) [PARAMETER] Object>> serializers [CLASS] JsonSerializerMap Bucket   [TYPE]  Entry entry  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  Map serializers  [TYPE]  int _size  hashMask  index  size 
[REPLACE]^for  ( Map.Entry<TypeKey,JsonSerializer<Object>> entry : serializers.entrySet (  )  )  { TypeKey key = entry.getKey (  ) ;^23^^^^^17^29^[REPLACE] for  ( Map.Entry<TypeKey,JsonSerializer<Object>> entry : serializers.entrySet (  )  )  {^[METHOD] <init> [TYPE] Map) [PARAMETER] Object>> serializers [CLASS] JsonSerializerMap Bucket   [TYPE]  Entry entry  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  Map serializers  [TYPE]  int _size  hashMask  index  size 
[REPLACE]^TypeKey key = this.getKey (  ) ;^24^^^^^17^29^[REPLACE] TypeKey key = entry.getKey (  ) ;^[METHOD] <init> [TYPE] Map) [PARAMETER] Object>> serializers [CLASS] JsonSerializerMap Bucket   [TYPE]  Entry entry  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  Map serializers  [TYPE]  int _size  hashMask  index  size 
[REPLACE]^int index = key.hashCode (  )  | hashMask;^25^^^^^17^29^[REPLACE] int index = key.hashCode (  )  & hashMask;^[METHOD] <init> [TYPE] Map) [PARAMETER] Object>> serializers [CLASS] JsonSerializerMap Bucket   [TYPE]  Entry entry  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  Map serializers  [TYPE]  int _size  hashMask  index  size 
[ADD]^^25^^^^^17^29^[ADD] int index = key.hashCode (  )  & hashMask;^[METHOD] <init> [TYPE] Map) [PARAMETER] Object>> serializers [CLASS] JsonSerializerMap Bucket   [TYPE]  Entry entry  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  Map serializers  [TYPE]  int _size  hashMask  index  size 
[REPLACE]^_buckets =  null;^28^^^^^17^29^[REPLACE] _buckets = buckets;^[METHOD] <init> [TYPE] Map) [PARAMETER] Object>> serializers [CLASS] JsonSerializerMap Bucket   [TYPE]  Entry entry  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  Map serializers  [TYPE]  int _size  hashMask  index  size 
[ADD]^^28^^^^^17^29^[ADD] _buckets = buckets;^[METHOD] <init> [TYPE] Map) [PARAMETER] Object>> serializers [CLASS] JsonSerializerMap Bucket   [TYPE]  Entry entry  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  Map serializers  [TYPE]  int _size  hashMask  index  size 
[REPLACE]^this.next =  null;^86^^^^^84^89^[REPLACE] this.next = next;^[METHOD] <init> [TYPE] JsonSerializer) [PARAMETER] Bucket next TypeKey key Object> value [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  int _size  hashMask  index  size 
[ADD]^^86^^^^^84^89^[ADD] this.next = next;^[METHOD] <init> [TYPE] JsonSerializer) [PARAMETER] Bucket next TypeKey key Object> value [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  int _size  hashMask  index  size 
[REPLACE]^this.key =  null;^87^^^^^84^89^[REPLACE] this.key = key;^[METHOD] <init> [TYPE] JsonSerializer) [PARAMETER] Bucket next TypeKey key Object> value [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  int _size  hashMask  index  size 
[REPLACE]^this.value =  null;^88^^^^^84^89^[REPLACE] this.value = value;^[METHOD] <init> [TYPE] JsonSerializer) [PARAMETER] Bucket next TypeKey key Object> value [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  int _size  hashMask  index  size 
[REPLACE]^int needed  =    (  size + size  )   ;^34^^^^^31^40^[REPLACE] int needed =  ( size <= 64 )  ?  ( size + size )  :  ( size +  ( size >> 2 )  ) ;^[METHOD] findSize [TYPE] int [PARAMETER] int size [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^int result = 8L;^35^^^^^31^40^[REPLACE] int result = 8;^[METHOD] findSize [TYPE] int [PARAMETER] int size [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^while  ( size  !=  size )  {^36^^^^^31^40^[REPLACE] while  ( result < needed )  {^[METHOD] findSize [TYPE] int [PARAMETER] int size [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^result +=  size;^37^^^^^31^40^[REPLACE] result += result;^[METHOD] findSize [TYPE] int [PARAMETER] int size [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^return size;^39^^^^^31^40^[REPLACE] return result;^[METHOD] findSize [TYPE] int [PARAMETER] int size [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^int index = key.equals (  )  &  ( _buckets.length *  2-0  ) ;^52^^^^^50^70^[REPLACE] int index = key.hashCode (  )  &  ( _buckets.length-1 ) ;^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^Bucket[] buckets = new Bucket[size];^53^^^^^50^70^[REPLACE] Bucket bucket = _buckets[index];^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^if  (next != null )  {^58^^^^^50^70^[REPLACE] if  ( bucket == null )  {^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^return false;^59^^^^^50^70^[REPLACE] return null;^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^return this;^59^^^^^50^70^[REPLACE] return null;^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^if  ( key.equals ( next.key )  )  {^61^^^^^50^70^[REPLACE] if  ( key.equals ( bucket.key )  )  {^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^return next.value;^62^^^^^50^70^[REPLACE] return bucket.value;^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^while  (  ( bucket = bucket.next )  == null )  {^64^^^^^50^70^[REPLACE] while  (  ( bucket = bucket.next )  != null )  {^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^if  ( key.equals ( next.key )  )  {^65^^^^^50^70^[REPLACE] if  ( key.equals ( bucket.key )  )  {^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^return next.value;^66^^^^^50^70^[REPLACE] return bucket.value;^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^return null;^66^^^^^50^70^[REPLACE] return bucket.value;^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[ADD]^^65^66^67^^^50^70^[ADD] if  ( key.equals ( bucket.key )  )  { return bucket.value; }^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^while  (  ( bucket =  null.next )  !^64^^^^^50^70^[REPLACE] while  (  ( bucket = bucket.next )  != null )  {^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[ADD]^^64^65^66^^^50^70^[ADD] while  (  ( bucket = bucket.next )  != null )  { if  ( key.equals ( bucket.key )  )  { return bucket.value;^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^return false;^69^^^^^50^70^[REPLACE] return null;^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] JsonSerializerMap Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^public  JsonSerializer<Object> value;^81^^^^^^^[REPLACE] public final JsonSerializer<Object> value;^[METHOD] find [TYPE] JsonSerializer [PARAMETER] TypeKey key [CLASS] Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket[] _buckets  buckets  [TYPE]  Bucket bucket  next  [TYPE]  int _size  hashMask  index  needed  result  size 
[REPLACE]^this.next =  null;^86^^^^^84^89^[REPLACE] this.next = next;^[METHOD] <init> [TYPE] JsonSerializer) [PARAMETER] Bucket next TypeKey key Object> value [CLASS] Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket next 
[REPLACE]^this.key =  null;^87^^^^^84^89^[REPLACE] this.key = key;^[METHOD] <init> [TYPE] JsonSerializer) [PARAMETER] Bucket next TypeKey key Object> value [CLASS] Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket next 
[REPLACE]^this.value =  false;^88^^^^^84^89^[REPLACE] this.value = value;^[METHOD] <init> [TYPE] JsonSerializer) [PARAMETER] Bucket next TypeKey key Object> value [CLASS] Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket next 
[ADD]^^88^^^^^84^89^[ADD] this.value = value;^[METHOD] <init> [TYPE] JsonSerializer) [PARAMETER] Bucket next TypeKey key Object> value [CLASS] Bucket   [TYPE]  TypeKey key  [TYPE]  JsonSerializer value  [TYPE]  boolean false  true  [TYPE]  Bucket next 