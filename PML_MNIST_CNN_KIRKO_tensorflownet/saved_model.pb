
єФ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
А
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.15.02v2.15.0-rc1-8-g6887368d6d48У
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
Ё
Adam/v/dense2/biasVarHandleOp*
_output_shapes
: *#

debug_nameAdam/v/dense2/bias/*
dtype0*
shape:
*#
shared_nameAdam/v/dense2/bias
u
&Adam/v/dense2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense2/bias*
_output_shapes
:
*
dtype0
Ё
Adam/m/dense2/biasVarHandleOp*
_output_shapes
: *#

debug_nameAdam/m/dense2/bias/*
dtype0*
shape:
*#
shared_nameAdam/m/dense2/bias
u
&Adam/m/dense2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense2/bias*
_output_shapes
:
*
dtype0
Ќ
Adam/v/dense2/kernelVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense2/kernel/*
dtype0*
shape:	
*%
shared_nameAdam/v/dense2/kernel
~
(Adam/v/dense2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense2/kernel*
_output_shapes
:	
*
dtype0
Ќ
Adam/m/dense2/kernelVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense2/kernel/*
dtype0*
shape:	
*%
shared_nameAdam/m/dense2/kernel
~
(Adam/m/dense2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense2/kernel*
_output_shapes
:	
*
dtype0
Ђ
Adam/v/dense1/biasVarHandleOp*
_output_shapes
: *#

debug_nameAdam/v/dense1/bias/*
dtype0*
shape:*#
shared_nameAdam/v/dense1/bias
v
&Adam/v/dense1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense1/bias*
_output_shapes	
:*
dtype0
Ђ
Adam/m/dense1/biasVarHandleOp*
_output_shapes
: *#

debug_nameAdam/m/dense1/bias/*
dtype0*
shape:*#
shared_nameAdam/m/dense1/bias
v
&Adam/m/dense1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense1/bias*
_output_shapes	
:*
dtype0
­
Adam/v/dense1/kernelVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense1/kernel/*
dtype0*
shape:
 *%
shared_nameAdam/v/dense1/kernel

(Adam/v/dense1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense1/kernel* 
_output_shapes
:
 *
dtype0
­
Adam/m/dense1/kernelVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense1/kernel/*
dtype0*
shape:
 *%
shared_nameAdam/m/dense1/kernel

(Adam/m/dense1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense1/kernel* 
_output_shapes
:
 *
dtype0

Adam/v/conv2/biasVarHandleOp*
_output_shapes
: *"

debug_nameAdam/v/conv2/bias/*
dtype0*
shape: *"
shared_nameAdam/v/conv2/bias
s
%Adam/v/conv2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2/bias*
_output_shapes
: *
dtype0

Adam/m/conv2/biasVarHandleOp*
_output_shapes
: *"

debug_nameAdam/m/conv2/bias/*
dtype0*
shape: *"
shared_nameAdam/m/conv2/bias
s
%Adam/m/conv2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2/bias*
_output_shapes
: *
dtype0
А
Adam/v/conv2/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/conv2/kernel/*
dtype0*
shape: *$
shared_nameAdam/v/conv2/kernel

'Adam/v/conv2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2/kernel*&
_output_shapes
: *
dtype0
А
Adam/m/conv2/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/conv2/kernel/*
dtype0*
shape: *$
shared_nameAdam/m/conv2/kernel

'Adam/m/conv2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2/kernel*&
_output_shapes
: *
dtype0

Adam/v/conv1/biasVarHandleOp*
_output_shapes
: *"

debug_nameAdam/v/conv1/bias/*
dtype0*
shape:*"
shared_nameAdam/v/conv1/bias
s
%Adam/v/conv1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv1/bias*
_output_shapes
:*
dtype0

Adam/m/conv1/biasVarHandleOp*
_output_shapes
: *"

debug_nameAdam/m/conv1/bias/*
dtype0*
shape:*"
shared_nameAdam/m/conv1/bias
s
%Adam/m/conv1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv1/bias*
_output_shapes
:*
dtype0
А
Adam/v/conv1/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/conv1/kernel/*
dtype0*
shape:*$
shared_nameAdam/v/conv1/kernel

'Adam/v/conv1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv1/kernel*&
_output_shapes
:*
dtype0
А
Adam/m/conv1/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/conv1/kernel/*
dtype0*
shape:*$
shared_nameAdam/m/conv1/kernel

'Adam/m/conv1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv1/kernel*&
_output_shapes
:*
dtype0

learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0

	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	

dense2/biasVarHandleOp*
_output_shapes
: *

debug_namedense2/bias/*
dtype0*
shape:
*
shared_namedense2/bias
g
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes
:
*
dtype0

dense2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense2/kernel/*
dtype0*
shape:	
*
shared_namedense2/kernel
p
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel*
_output_shapes
:	
*
dtype0

dense1/biasVarHandleOp*
_output_shapes
: *

debug_namedense1/bias/*
dtype0*
shape:*
shared_namedense1/bias
h
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes	
:*
dtype0

dense1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense1/kernel/*
dtype0*
shape:
 *
shared_namedense1/kernel
q
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel* 
_output_shapes
:
 *
dtype0


conv2/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2/bias/*
dtype0*
shape: *
shared_name
conv2/bias
e
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes
: *
dtype0

conv2/kernelVarHandleOp*
_output_shapes
: *

debug_nameconv2/kernel/*
dtype0*
shape: *
shared_nameconv2/kernel
u
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*&
_output_shapes
: *
dtype0


conv1/biasVarHandleOp*
_output_shapes
: *

debug_nameconv1/bias/*
dtype0*
shape:*
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
:*
dtype0

conv1/kernelVarHandleOp*
_output_shapes
: *

debug_nameconv1/kernel/*
dtype0*
shape:*
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
:*
dtype0

serving_default_conv1_inputPlaceholder*/
_output_shapes
:џџџџџџџџџ*
dtype0*$
shape:џџџџџџџџџ
Б
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1_inputconv1/kernel
conv1/biasconv2/kernel
conv2/biasdense1/kerneldense1/biasdense2/kerneldense2/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_23681

NoOpNoOp
ЊA
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*х@
valueл@Bи@ Bб@

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
Ш
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
Ш
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op*

)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses* 

/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses* 
І
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias*
І
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias*
<
0
1
&2
'3
;4
<5
C6
D7*
<
0
1
&2
'3
;4
<5
C6
D7*
* 
А
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Jtrace_0
Ktrace_1* 

Ltrace_0
Mtrace_1* 
* 

N
_variables
O_iterations
P_learning_rate
Q_index_dict
R
_momentums
S_velocities
T_update_step_xla*

Userving_default* 

0
1*

0
1*
* 

Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

[trace_0* 

\trace_0* 
\V
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

btrace_0* 

ctrace_0* 

&0
'1*

&0
'1*
* 

dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

itrace_0* 

jtrace_0* 
\V
VARIABLE_VALUEconv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

ptrace_0* 

qtrace_0* 
* 
* 
* 

rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses* 

wtrace_0* 

xtrace_0* 

;0
<1*

;0
<1*
* 

ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

~trace_0* 

trace_0* 
]W
VARIABLE_VALUEdense1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

C0
D1*

C0
D1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

trace_0* 

trace_0* 
]W
VARIABLE_VALUEdense2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*

0
1*
* 
* 
* 
* 
* 
* 

O0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
D
0
1
2
3
4
5
6
7*
D
0
1
2
3
4
5
6
7*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

 count
Ё
_fn_kwargs*
^X
VARIABLE_VALUEAdam/m/conv1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv1/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/conv1/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/conv1/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/conv2/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/conv2/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/conv2/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/conv2/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense1/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense1/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense1/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense1/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense2/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense2/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense2/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense2/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
І
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasconv2/kernel
conv2/biasdense1/kerneldense1/biasdense2/kerneldense2/bias	iterationlearning_rateAdam/m/conv1/kernelAdam/v/conv1/kernelAdam/m/conv1/biasAdam/v/conv1/biasAdam/m/conv2/kernelAdam/v/conv2/kernelAdam/m/conv2/biasAdam/v/conv2/biasAdam/m/dense1/kernelAdam/v/dense1/kernelAdam/m/dense1/biasAdam/v/dense1/biasAdam/m/dense2/kernelAdam/v/dense2/kernelAdam/m/dense2/biasAdam/v/dense2/biastotal_1count_1totalcountConst*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_23994
Ё
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasconv2/kernel
conv2/biasdense1/kerneldense1/biasdense2/kerneldense2/bias	iterationlearning_rateAdam/m/conv1/kernelAdam/v/conv1/kernelAdam/m/conv1/biasAdam/v/conv1/biasAdam/m/conv2/kernelAdam/v/conv2/kernelAdam/m/conv2/biasAdam/v/conv2/biasAdam/m/dense1/kernelAdam/v/dense1/kernelAdam/m/dense1/biasAdam/v/dense1/biasAdam/m/dense2/kernelAdam/v/dense2/kernelAdam/m/dense2/biasAdam/v/dense2/biastotal_1count_1totalcount**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_24093ЁА
К 
ы
E__inference_sequential_layer_call_and_return_conditional_losses_23547
conv1_input%
conv1_23484:
conv1_23486:%
conv2_23501: 
conv2_23503:  
dense1_23525:
 
dense1_23527:	
dense2_23541:	

dense2_23543:

identityЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallЂdense1/StatefulPartitionedCallЂdense2/StatefulPartitionedCallю
conv1/StatefulPartitionedCallStatefulPartitionedCallconv1_inputconv1_23484conv1_23486*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_23483й
pool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_pool1_layer_call_and_return_conditional_losses_23455
conv2/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0conv2_23501conv2_23503*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_23500й
pool2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_pool2_layer_call_and_return_conditional_losses_23465Ю
flatten/PartitionedCallPartitionedCallpool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_23512
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_23525dense1_23527*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_23524
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_23541dense2_23543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense2_layer_call_and_return_conditional_losses_23540v
IdentityIdentity'dense2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Є
NoOpNoOp^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall:%!

_user_specified_name23543:%!

_user_specified_name23541:%!

_user_specified_name23527:%!

_user_specified_name23525:%!

_user_specified_name23503:%!

_user_specified_name23501:%!

_user_specified_name23486:%!

_user_specified_name23484:\ X
/
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameconv1_input
д

ѕ
A__inference_dense1_layer_call_and_return_conditional_losses_23524

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

\
@__inference_pool2_layer_call_and_return_conditional_losses_23465

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

\
@__inference_pool2_layer_call_and_return_conditional_losses_23741

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
д

ѕ
A__inference_dense1_layer_call_and_return_conditional_losses_23772

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ђ
A
%__inference_pool1_layer_call_fn_23706

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_pool1_layer_call_and_return_conditional_losses_23455
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

в
*__inference_sequential_layer_call_fn_23616
conv1_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: 
	unknown_3:
 
	unknown_4:	
	unknown_5:	

	unknown_6:

identityЂStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallconv1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_23574o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name23612:%!

_user_specified_name23610:%!

_user_specified_name23608:%!

_user_specified_name23606:%!

_user_specified_name23604:%!

_user_specified_name23602:%!

_user_specified_name23600:%!

_user_specified_name23598:\ X
/
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameconv1_input

\
@__inference_pool1_layer_call_and_return_conditional_losses_23711

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


%__inference_conv1_layer_call_fn_23690

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_23483w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name23686:%!

_user_specified_name23684:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

\
@__inference_pool1_layer_call_and_return_conditional_losses_23455

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ћ2
И
 __inference__wrapped_model_23450
conv1_inputI
/sequential_conv1_conv2d_readvariableop_resource:>
0sequential_conv1_biasadd_readvariableop_resource:I
/sequential_conv2_conv2d_readvariableop_resource: >
0sequential_conv2_biasadd_readvariableop_resource: D
0sequential_dense1_matmul_readvariableop_resource:
 @
1sequential_dense1_biasadd_readvariableop_resource:	C
0sequential_dense2_matmul_readvariableop_resource:	
?
1sequential_dense2_biasadd_readvariableop_resource:

identityЂ'sequential/conv1/BiasAdd/ReadVariableOpЂ&sequential/conv1/Conv2D/ReadVariableOpЂ'sequential/conv2/BiasAdd/ReadVariableOpЂ&sequential/conv2/Conv2D/ReadVariableOpЂ(sequential/dense1/BiasAdd/ReadVariableOpЂ'sequential/dense1/MatMul/ReadVariableOpЂ(sequential/dense2/BiasAdd/ReadVariableOpЂ'sequential/dense2/MatMul/ReadVariableOp
&sequential/conv1/Conv2D/ReadVariableOpReadVariableOp/sequential_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Р
sequential/conv1/Conv2DConv2Dconv1_input.sequential/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

'sequential/conv1/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0А
sequential/conv1/BiasAddBiasAdd sequential/conv1/Conv2D:output:0/sequential/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџz
sequential/conv1/ReluRelu!sequential/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџЕ
sequential/pool1/MaxPoolMaxPool#sequential/conv1/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

&sequential/conv2/Conv2D/ReadVariableOpReadVariableOp/sequential_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ж
sequential/conv2/Conv2DConv2D!sequential/pool1/MaxPool:output:0.sequential/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

'sequential/conv2/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0А
sequential/conv2/BiasAddBiasAdd sequential/conv2/Conv2D:output:0/sequential/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ z
sequential/conv2/ReluRelu!sequential/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ Е
sequential/pool2/MaxPoolMaxPool#sequential/conv2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
sequential/flatten/ReshapeReshape!sequential/pool2/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
'sequential/dense1/MatMul/ReadVariableOpReadVariableOp0sequential_dense1_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0Ћ
sequential/dense1/MatMulMatMul#sequential/flatten/Reshape:output:0/sequential/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
(sequential/dense1/BiasAdd/ReadVariableOpReadVariableOp1sequential_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
sequential/dense1/BiasAddBiasAdd"sequential/dense1/MatMul:product:00sequential/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџu
sequential/dense1/ReluRelu"sequential/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
'sequential/dense2/MatMul/ReadVariableOpReadVariableOp0sequential_dense2_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0Ћ
sequential/dense2/MatMulMatMul$sequential/dense1/Relu:activations:0/sequential/dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

(sequential/dense2/BiasAdd/ReadVariableOpReadVariableOp1sequential_dense2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ќ
sequential/dense2/BiasAddBiasAdd"sequential/dense2/MatMul:product:00sequential/dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
z
sequential/dense2/SoftmaxSoftmax"sequential/dense2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
IdentityIdentity#sequential/dense2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
ђ
NoOpNoOp(^sequential/conv1/BiasAdd/ReadVariableOp'^sequential/conv1/Conv2D/ReadVariableOp(^sequential/conv2/BiasAdd/ReadVariableOp'^sequential/conv2/Conv2D/ReadVariableOp)^sequential/dense1/BiasAdd/ReadVariableOp(^sequential/dense1/MatMul/ReadVariableOp)^sequential/dense2/BiasAdd/ReadVariableOp(^sequential/dense2/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 2R
'sequential/conv1/BiasAdd/ReadVariableOp'sequential/conv1/BiasAdd/ReadVariableOp2P
&sequential/conv1/Conv2D/ReadVariableOp&sequential/conv1/Conv2D/ReadVariableOp2R
'sequential/conv2/BiasAdd/ReadVariableOp'sequential/conv2/BiasAdd/ReadVariableOp2P
&sequential/conv2/Conv2D/ReadVariableOp&sequential/conv2/Conv2D/ReadVariableOp2T
(sequential/dense1/BiasAdd/ReadVariableOp(sequential/dense1/BiasAdd/ReadVariableOp2R
'sequential/dense1/MatMul/ReadVariableOp'sequential/dense1/MatMul/ReadVariableOp2T
(sequential/dense2/BiasAdd/ReadVariableOp(sequential/dense2/BiasAdd/ReadVariableOp2R
'sequential/dense2/MatMul/ReadVariableOp'sequential/dense2/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
/
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameconv1_input


%__inference_conv2_layer_call_fn_23720

inputs!
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_23500w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name23716:%!

_user_specified_name23714:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


!__inference__traced_restore_24093
file_prefix7
assignvariableop_conv1_kernel:+
assignvariableop_1_conv1_bias:9
assignvariableop_2_conv2_kernel: +
assignvariableop_3_conv2_bias: 4
 assignvariableop_4_dense1_kernel:
 -
assignvariableop_5_dense1_bias:	3
 assignvariableop_6_dense2_kernel:	
,
assignvariableop_7_dense2_bias:
&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: A
'assignvariableop_10_adam_m_conv1_kernel:A
'assignvariableop_11_adam_v_conv1_kernel:3
%assignvariableop_12_adam_m_conv1_bias:3
%assignvariableop_13_adam_v_conv1_bias:A
'assignvariableop_14_adam_m_conv2_kernel: A
'assignvariableop_15_adam_v_conv2_kernel: 3
%assignvariableop_16_adam_m_conv2_bias: 3
%assignvariableop_17_adam_v_conv2_bias: <
(assignvariableop_18_adam_m_dense1_kernel:
 <
(assignvariableop_19_adam_v_dense1_kernel:
 5
&assignvariableop_20_adam_m_dense1_bias:	5
&assignvariableop_21_adam_v_dense1_bias:	;
(assignvariableop_22_adam_m_dense2_kernel:	
;
(assignvariableop_23_adam_v_dense2_kernel:	
4
&assignvariableop_24_adam_m_dense2_bias:
4
&assignvariableop_25_adam_v_dense2_bias:
%
assignvariableop_26_total_1: %
assignvariableop_27_count_1: #
assignvariableop_28_total: #
assignvariableop_29_count: 
identity_31ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Л
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*с
valueзBдB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЎ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B К
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense1_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense1_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense2_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:Г
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_10AssignVariableOp'assignvariableop_10_adam_m_conv1_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_11AssignVariableOp'assignvariableop_11_adam_v_conv1_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_12AssignVariableOp%assignvariableop_12_adam_m_conv1_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_13AssignVariableOp%assignvariableop_13_adam_v_conv1_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_m_conv2_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_v_conv2_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_m_conv2_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_17AssignVariableOp%assignvariableop_17_adam_v_conv2_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_m_dense1_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_v_dense1_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_m_dense1_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_21AssignVariableOp&assignvariableop_21_adam_v_dense1_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_m_dense2_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_v_dense2_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_m_dense2_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_25AssignVariableOp&assignvariableop_25_adam_v_dense2_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 у
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: Ќ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_31Identity_31:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%!

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:2.
,
_user_specified_nameAdam/v/dense2/bias:2.
,
_user_specified_nameAdam/m/dense2/bias:40
.
_user_specified_nameAdam/v/dense2/kernel:40
.
_user_specified_nameAdam/m/dense2/kernel:2.
,
_user_specified_nameAdam/v/dense1/bias:2.
,
_user_specified_nameAdam/m/dense1/bias:40
.
_user_specified_nameAdam/v/dense1/kernel:40
.
_user_specified_nameAdam/m/dense1/kernel:1-
+
_user_specified_nameAdam/v/conv2/bias:1-
+
_user_specified_nameAdam/m/conv2/bias:3/
-
_user_specified_nameAdam/v/conv2/kernel:3/
-
_user_specified_nameAdam/m/conv2/kernel:1-
+
_user_specified_nameAdam/v/conv1/bias:1-
+
_user_specified_nameAdam/m/conv1/bias:3/
-
_user_specified_nameAdam/v/conv1/kernel:3/
-
_user_specified_nameAdam/m/conv1/kernel:-
)
'
_user_specified_namelearning_rate:)	%
#
_user_specified_name	iteration:+'
%
_user_specified_namedense2/bias:-)
'
_user_specified_namedense2/kernel:+'
%
_user_specified_namedense1/bias:-)
'
_user_specified_namedense1/kernel:*&
$
_user_specified_name
conv2/bias:,(
&
_user_specified_nameconv2/kernel:*&
$
_user_specified_name
conv1/bias:,(
&
_user_specified_nameconv1/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ћ
C
'__inference_flatten_layer_call_fn_23746

inputs
identityЎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_23512a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ф
^
B__inference_flatten_layer_call_and_return_conditional_losses_23512

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

в
*__inference_sequential_layer_call_fn_23595
conv1_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: 
	unknown_3:
 
	unknown_4:	
	unknown_5:	

	unknown_6:

identityЂStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallconv1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_23547o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name23591:%!

_user_specified_name23589:%!

_user_specified_name23587:%!

_user_specified_name23585:%!

_user_specified_name23583:%!

_user_specified_name23581:%!

_user_specified_name23579:%!

_user_specified_name23577:\ X
/
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameconv1_input
Џ
љ
@__inference_conv1_layer_call_and_return_conditional_losses_23701

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К 
ы
E__inference_sequential_layer_call_and_return_conditional_losses_23574
conv1_input%
conv1_23550:
conv1_23552:%
conv2_23556: 
conv2_23558:  
dense1_23563:
 
dense1_23565:	
dense2_23568:	

dense2_23570:

identityЂconv1/StatefulPartitionedCallЂconv2/StatefulPartitionedCallЂdense1/StatefulPartitionedCallЂdense2/StatefulPartitionedCallю
conv1/StatefulPartitionedCallStatefulPartitionedCallconv1_inputconv1_23550conv1_23552*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_23483й
pool1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_pool1_layer_call_and_return_conditional_losses_23455
conv2/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0conv2_23556conv2_23558*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2_layer_call_and_return_conditional_losses_23500й
pool2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_pool2_layer_call_and_return_conditional_losses_23465Ю
flatten/PartitionedCallPartitionedCallpool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_23512
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_23563dense1_23565*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_23524
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_23568dense2_23570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense2_layer_call_and_return_conditional_losses_23540v
IdentityIdentity'dense2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Є
NoOpNoOp^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall:%!

_user_specified_name23570:%!

_user_specified_name23568:%!

_user_specified_name23565:%!

_user_specified_name23563:%!

_user_specified_name23558:%!

_user_specified_name23556:%!

_user_specified_name23552:%!

_user_specified_name23550:\ X
/
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameconv1_input
Ф
^
B__inference_flatten_layer_call_and_return_conditional_losses_23752

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Џ
љ
@__inference_conv2_layer_call_and_return_conditional_losses_23731

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
э

&__inference_dense1_layer_call_fn_23761

inputs
unknown:
 
	unknown_0:	
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense1_layer_call_and_return_conditional_losses_23524p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name23757:%!

_user_specified_name23755:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ац
№
__inference__traced_save_23994
file_prefix=
#read_disablecopyonread_conv1_kernel:1
#read_1_disablecopyonread_conv1_bias:?
%read_2_disablecopyonread_conv2_kernel: 1
#read_3_disablecopyonread_conv2_bias: :
&read_4_disablecopyonread_dense1_kernel:
 3
$read_5_disablecopyonread_dense1_bias:	9
&read_6_disablecopyonread_dense2_kernel:	
2
$read_7_disablecopyonread_dense2_bias:
,
"read_8_disablecopyonread_iteration:	 0
&read_9_disablecopyonread_learning_rate: G
-read_10_disablecopyonread_adam_m_conv1_kernel:G
-read_11_disablecopyonread_adam_v_conv1_kernel:9
+read_12_disablecopyonread_adam_m_conv1_bias:9
+read_13_disablecopyonread_adam_v_conv1_bias:G
-read_14_disablecopyonread_adam_m_conv2_kernel: G
-read_15_disablecopyonread_adam_v_conv2_kernel: 9
+read_16_disablecopyonread_adam_m_conv2_bias: 9
+read_17_disablecopyonread_adam_v_conv2_bias: B
.read_18_disablecopyonread_adam_m_dense1_kernel:
 B
.read_19_disablecopyonread_adam_v_dense1_kernel:
 ;
,read_20_disablecopyonread_adam_m_dense1_bias:	;
,read_21_disablecopyonread_adam_v_dense1_bias:	A
.read_22_disablecopyonread_adam_m_dense2_kernel:	
A
.read_23_disablecopyonread_adam_v_dense2_kernel:	
:
,read_24_disablecopyonread_adam_m_dense2_bias:
:
,read_25_disablecopyonread_adam_v_dense2_bias:
+
!read_26_disablecopyonread_total_1: +
!read_27_disablecopyonread_count_1: )
read_28_disablecopyonread_total: )
read_29_disablecopyonread_count: 
savev2_const
identity_61ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_conv1_kernel"/device:CPU:0*
_output_shapes
 Ї
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_conv1_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_conv1_bias"/device:CPU:0*
_output_shapes
 
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_conv1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:y
Read_2/DisableCopyOnReadDisableCopyOnRead%read_2_disablecopyonread_conv2_kernel"/device:CPU:0*
_output_shapes
 ­
Read_2/ReadVariableOpReadVariableOp%read_2_disablecopyonread_conv2_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: w
Read_3/DisableCopyOnReadDisableCopyOnRead#read_3_disablecopyonread_conv2_bias"/device:CPU:0*
_output_shapes
 
Read_3/ReadVariableOpReadVariableOp#read_3_disablecopyonread_conv2_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_4/DisableCopyOnReadDisableCopyOnRead&read_4_disablecopyonread_dense1_kernel"/device:CPU:0*
_output_shapes
 Ј
Read_4/ReadVariableOpReadVariableOp&read_4_disablecopyonread_dense1_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
 *
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
 e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
 x
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_dense1_bias"/device:CPU:0*
_output_shapes
 Ё
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_dense1_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:z
Read_6/DisableCopyOnReadDisableCopyOnRead&read_6_disablecopyonread_dense2_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_6/ReadVariableOpReadVariableOp&read_6_disablecopyonread_dense2_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	
*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	
f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	
x
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_dense2_bias"/device:CPU:0*
_output_shapes
  
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_dense2_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_iteration^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_10/DisableCopyOnReadDisableCopyOnRead-read_10_disablecopyonread_adam_m_conv1_kernel"/device:CPU:0*
_output_shapes
 З
Read_10/ReadVariableOpReadVariableOp-read_10_disablecopyonread_adam_m_conv1_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_11/DisableCopyOnReadDisableCopyOnRead-read_11_disablecopyonread_adam_v_conv1_kernel"/device:CPU:0*
_output_shapes
 З
Read_11/ReadVariableOpReadVariableOp-read_11_disablecopyonread_adam_v_conv1_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*&
_output_shapes
:
Read_12/DisableCopyOnReadDisableCopyOnRead+read_12_disablecopyonread_adam_m_conv1_bias"/device:CPU:0*
_output_shapes
 Љ
Read_12/ReadVariableOpReadVariableOp+read_12_disablecopyonread_adam_m_conv1_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_13/DisableCopyOnReadDisableCopyOnRead+read_13_disablecopyonread_adam_v_conv1_bias"/device:CPU:0*
_output_shapes
 Љ
Read_13/ReadVariableOpReadVariableOp+read_13_disablecopyonread_adam_v_conv1_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_14/DisableCopyOnReadDisableCopyOnRead-read_14_disablecopyonread_adam_m_conv2_kernel"/device:CPU:0*
_output_shapes
 З
Read_14/ReadVariableOpReadVariableOp-read_14_disablecopyonread_adam_m_conv2_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_15/DisableCopyOnReadDisableCopyOnRead-read_15_disablecopyonread_adam_v_conv2_kernel"/device:CPU:0*
_output_shapes
 З
Read_15/ReadVariableOpReadVariableOp-read_15_disablecopyonread_adam_v_conv2_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*&
_output_shapes
: 
Read_16/DisableCopyOnReadDisableCopyOnRead+read_16_disablecopyonread_adam_m_conv2_bias"/device:CPU:0*
_output_shapes
 Љ
Read_16/ReadVariableOpReadVariableOp+read_16_disablecopyonread_adam_m_conv2_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_17/DisableCopyOnReadDisableCopyOnRead+read_17_disablecopyonread_adam_v_conv2_bias"/device:CPU:0*
_output_shapes
 Љ
Read_17/ReadVariableOpReadVariableOp+read_17_disablecopyonread_adam_v_conv2_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_18/DisableCopyOnReadDisableCopyOnRead.read_18_disablecopyonread_adam_m_dense1_kernel"/device:CPU:0*
_output_shapes
 В
Read_18/ReadVariableOpReadVariableOp.read_18_disablecopyonread_adam_m_dense1_kernel^Read_18/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
 *
dtype0q
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
 g
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0* 
_output_shapes
:
 
Read_19/DisableCopyOnReadDisableCopyOnRead.read_19_disablecopyonread_adam_v_dense1_kernel"/device:CPU:0*
_output_shapes
 В
Read_19/ReadVariableOpReadVariableOp.read_19_disablecopyonread_adam_v_dense1_kernel^Read_19/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
 *
dtype0q
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
 g
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0* 
_output_shapes
:
 
Read_20/DisableCopyOnReadDisableCopyOnRead,read_20_disablecopyonread_adam_m_dense1_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_20/ReadVariableOpReadVariableOp,read_20_disablecopyonread_adam_m_dense1_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_21/DisableCopyOnReadDisableCopyOnRead,read_21_disablecopyonread_adam_v_dense1_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_21/ReadVariableOpReadVariableOp,read_21_disablecopyonread_adam_v_dense1_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_22/DisableCopyOnReadDisableCopyOnRead.read_22_disablecopyonread_adam_m_dense2_kernel"/device:CPU:0*
_output_shapes
 Б
Read_22/ReadVariableOpReadVariableOp.read_22_disablecopyonread_adam_m_dense2_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	
*
dtype0p
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	
f
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:	

Read_23/DisableCopyOnReadDisableCopyOnRead.read_23_disablecopyonread_adam_v_dense2_kernel"/device:CPU:0*
_output_shapes
 Б
Read_23/ReadVariableOpReadVariableOp.read_23_disablecopyonread_adam_v_dense2_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	
*
dtype0p
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	
f
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:	

Read_24/DisableCopyOnReadDisableCopyOnRead,read_24_disablecopyonread_adam_m_dense2_bias"/device:CPU:0*
_output_shapes
 Њ
Read_24/ReadVariableOpReadVariableOp,read_24_disablecopyonread_adam_m_dense2_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:

Read_25/DisableCopyOnReadDisableCopyOnRead,read_25_disablecopyonread_adam_v_dense2_bias"/device:CPU:0*
_output_shapes
 Њ
Read_25/ReadVariableOpReadVariableOp,read_25_disablecopyonread_adam_v_dense2_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_26/DisableCopyOnReadDisableCopyOnRead!read_26_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_26/ReadVariableOpReadVariableOp!read_26_disablecopyonread_total_1^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_27/DisableCopyOnReadDisableCopyOnRead!read_27_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_27/ReadVariableOpReadVariableOp!read_27_disablecopyonread_count_1^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_28/DisableCopyOnReadDisableCopyOnReadread_28_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_28/ReadVariableOpReadVariableOpread_28_disablecopyonread_total^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_29/DisableCopyOnReadDisableCopyOnReadread_29_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_29/ReadVariableOpReadVariableOpread_29_disablecopyonread_count^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: И
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*с
valueзBдB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЋ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *-
dtypes#
!2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_60Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_61IdentityIdentity_60:output:0^NoOp*
T0*
_output_shapes
: е
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_61Identity_61:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:%!

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:2.
,
_user_specified_nameAdam/v/dense2/bias:2.
,
_user_specified_nameAdam/m/dense2/bias:40
.
_user_specified_nameAdam/v/dense2/kernel:40
.
_user_specified_nameAdam/m/dense2/kernel:2.
,
_user_specified_nameAdam/v/dense1/bias:2.
,
_user_specified_nameAdam/m/dense1/bias:40
.
_user_specified_nameAdam/v/dense1/kernel:40
.
_user_specified_nameAdam/m/dense1/kernel:1-
+
_user_specified_nameAdam/v/conv2/bias:1-
+
_user_specified_nameAdam/m/conv2/bias:3/
-
_user_specified_nameAdam/v/conv2/kernel:3/
-
_user_specified_nameAdam/m/conv2/kernel:1-
+
_user_specified_nameAdam/v/conv1/bias:1-
+
_user_specified_nameAdam/m/conv1/bias:3/
-
_user_specified_nameAdam/v/conv1/kernel:3/
-
_user_specified_nameAdam/m/conv1/kernel:-
)
'
_user_specified_namelearning_rate:)	%
#
_user_specified_name	iteration:+'
%
_user_specified_namedense2/bias:-)
'
_user_specified_namedense2/kernel:+'
%
_user_specified_namedense1/bias:-)
'
_user_specified_namedense1/kernel:*&
$
_user_specified_name
conv2/bias:,(
&
_user_specified_nameconv2/kernel:*&
$
_user_specified_name
conv1/bias:,(
&
_user_specified_nameconv1/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
л
Ы
#__inference_signature_wrapper_23681
conv1_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: 
	unknown_3:
 
	unknown_4:	
	unknown_5:	

	unknown_6:

identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_23450o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name23677:%!

_user_specified_name23675:%!

_user_specified_name23673:%!

_user_specified_name23671:%!

_user_specified_name23669:%!

_user_specified_name23667:%!

_user_specified_name23665:%!

_user_specified_name23663:\ X
/
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameconv1_input
Џ
љ
@__inference_conv2_layer_call_and_return_conditional_losses_23500

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
б

ѓ
A__inference_dense2_layer_call_and_return_conditional_losses_23540

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
б

ѓ
A__inference_dense2_layer_call_and_return_conditional_losses_23792

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Џ
љ
@__inference_conv1_layer_call_and_return_conditional_losses_23483

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
щ

&__inference_dense2_layer_call_fn_23781

inputs
unknown:	

	unknown_0:

identityЂStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense2_layer_call_and_return_conditional_losses_23540o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name23777:%!

_user_specified_name23775:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ
A
%__inference_pool2_layer_call_fn_23736

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_pool2_layer_call_and_return_conditional_losses_23465
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"эL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Й
serving_defaultЅ
K
conv1_input<
serving_default_conv1_input:0џџџџџџџџџ:
dense20
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict:ЅЇ
Љ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
н
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
н
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias"
_tf_keras_layer
Л
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias"
_tf_keras_layer
X
0
1
&2
'3
;4
<5
C6
D7"
trackable_list_wrapper
X
0
1
&2
'3
;4
<5
C6
D7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ч
Jtrace_0
Ktrace_12
*__inference_sequential_layer_call_fn_23595
*__inference_sequential_layer_call_fn_23616Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zJtrace_0zKtrace_1
§
Ltrace_0
Mtrace_12Ц
E__inference_sequential_layer_call_and_return_conditional_losses_23547
E__inference_sequential_layer_call_and_return_conditional_losses_23574Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zLtrace_0zMtrace_1
ЯBЬ
 __inference__wrapped_model_23450conv1_input"
В
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

N
_variables
O_iterations
P_learning_rate
Q_index_dict
R
_momentums
S_velocities
T_update_step_xla"
experimentalOptimizer
,
Userving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
п
[trace_02Т
%__inference_conv1_layer_call_fn_23690
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z[trace_0
њ
\trace_02н
@__inference_conv1_layer_call_and_return_conditional_losses_23701
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z\trace_0
&:$2conv1/kernel
:2
conv1/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
п
btrace_02Т
%__inference_pool1_layer_call_fn_23706
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zbtrace_0
њ
ctrace_02н
@__inference_pool1_layer_call_and_return_conditional_losses_23711
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zctrace_0
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
п
itrace_02Т
%__inference_conv2_layer_call_fn_23720
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zitrace_0
њ
jtrace_02н
@__inference_conv2_layer_call_and_return_conditional_losses_23731
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zjtrace_0
&:$ 2conv2/kernel
: 2
conv2/bias
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
п
ptrace_02Т
%__inference_pool2_layer_call_fn_23736
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zptrace_0
њ
qtrace_02н
@__inference_pool2_layer_call_and_return_conditional_losses_23741
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zqtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
с
wtrace_02Ф
'__inference_flatten_layer_call_fn_23746
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zwtrace_0
ќ
xtrace_02п
B__inference_flatten_layer_call_and_return_conditional_losses_23752
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zxtrace_0
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
р
~trace_02У
&__inference_dense1_layer_call_fn_23761
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z~trace_0
ћ
trace_02о
A__inference_dense1_layer_call_and_return_conditional_losses_23772
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
!:
 2dense1/kernel
:2dense1/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
т
trace_02У
&__inference_dense2_layer_call_fn_23781
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
§
trace_02о
A__inference_dense2_layer_call_and_return_conditional_losses_23792
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 :	
2dense2/kernel
:
2dense2/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
*__inference_sequential_layer_call_fn_23595conv1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
*__inference_sequential_layer_call_fn_23616conv1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_23547conv1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_23574conv1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ў
O0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
Е2ВЏ
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
зBд
#__inference_signature_wrapper_23681conv1_input"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
  

kwonlyargs
jconv1_input
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЯBЬ
%__inference_conv1_layer_call_fn_23690inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъBч
@__inference_conv1_layer_call_and_return_conditional_losses_23701inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЯBЬ
%__inference_pool1_layer_call_fn_23706inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъBч
@__inference_pool1_layer_call_and_return_conditional_losses_23711inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЯBЬ
%__inference_conv2_layer_call_fn_23720inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъBч
@__inference_conv2_layer_call_and_return_conditional_losses_23731inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЯBЬ
%__inference_pool2_layer_call_fn_23736inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъBч
@__inference_pool2_layer_call_and_return_conditional_losses_23741inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
бBЮ
'__inference_flatten_layer_call_fn_23746inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ьBщ
B__inference_flatten_layer_call_and_return_conditional_losses_23752inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
&__inference_dense1_layer_call_fn_23761inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
A__inference_dense1_layer_call_and_return_conditional_losses_23772inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
&__inference_dense2_layer_call_fn_23781inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
A__inference_dense2_layer_call_and_return_conditional_losses_23792inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

 count
Ё
_fn_kwargs"
_tf_keras_metric
+:)2Adam/m/conv1/kernel
+:)2Adam/v/conv1/kernel
:2Adam/m/conv1/bias
:2Adam/v/conv1/bias
+:) 2Adam/m/conv2/kernel
+:) 2Adam/v/conv2/kernel
: 2Adam/m/conv2/bias
: 2Adam/v/conv2/bias
&:$
 2Adam/m/dense1/kernel
&:$
 2Adam/v/dense1/kernel
:2Adam/m/dense1/bias
:2Adam/v/dense1/bias
%:#	
2Adam/m/dense2/kernel
%:#	
2Adam/v/dense2/kernel
:
2Adam/m/dense2/bias
:
2Adam/v/dense2/bias
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
 1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 __inference__wrapped_model_23450y&';<CD<Ђ9
2Ђ/
-*
conv1_inputџџџџџџџџџ
Њ "/Њ,
*
dense2 
dense2џџџџџџџџџ
З
@__inference_conv1_layer_call_and_return_conditional_losses_23701s7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ
 
%__inference_conv1_layer_call_fn_23690h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ")&
unknownџџџџџџџџџЗ
@__inference_conv2_layer_call_and_return_conditional_losses_23731s&'7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ 
 
%__inference_conv2_layer_call_fn_23720h&'7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ")&
unknownџџџџџџџџџ Њ
A__inference_dense1_layer_call_and_return_conditional_losses_23772e;<0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
&__inference_dense1_layer_call_fn_23761Z;<0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ ""
unknownџџџџџџџџџЉ
A__inference_dense2_layer_call_and_return_conditional_losses_23792dCD0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 
&__inference_dense2_layer_call_fn_23781YCD0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ
Ў
B__inference_flatten_layer_call_and_return_conditional_losses_23752h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ 
 
'__inference_flatten_layer_call_fn_23746]7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ ""
unknownџџџџџџџџџ ъ
@__inference_pool1_layer_call_and_return_conditional_losses_23711ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ф
%__inference_pool1_layer_call_fn_23706RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџъ
@__inference_pool2_layer_call_and_return_conditional_losses_23741ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ф
%__inference_pool2_layer_call_fn_23736RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџЧ
E__inference_sequential_layer_call_and_return_conditional_losses_23547~&';<CDDЂA
:Ђ7
-*
conv1_inputџџџџџџџџџ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 Ч
E__inference_sequential_layer_call_and_return_conditional_losses_23574~&';<CDDЂA
:Ђ7
-*
conv1_inputџџџџџџџџџ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ

 Ё
*__inference_sequential_layer_call_fn_23595s&';<CDDЂA
:Ђ7
-*
conv1_inputџџџџџџџџџ
p

 
Њ "!
unknownџџџџџџџџџ
Ё
*__inference_sequential_layer_call_fn_23616s&';<CDDЂA
:Ђ7
-*
conv1_inputџџџџџџџџџ
p 

 
Њ "!
unknownџџџџџџџџџ
А
#__inference_signature_wrapper_23681&';<CDKЂH
Ђ 
AЊ>
<
conv1_input-*
conv1_inputџџџџџџџџџ"/Њ,
*
dense2 
dense2џџџџџџџџџ
