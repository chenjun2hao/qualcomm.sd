usage: qnn-onnx-converter [--out_node OUT_NAMES] [--input_type INPUT_NAME INPUT_TYPE]
                          [--input_dtype INPUT_NAME INPUT_DTYPE] [--input_encoding  ...]]
                          [--input_layout INPUT_NAME INPUT_LAYOUT] [--custom_io CUSTOM_IO]
                          [--preserve_io [PRESERVE_IO ...]]
                          [--dump_qairt_io_config_yaml DUMP_QAIRT_IO_CONFIG_YAML]
                          [--enable_framework_trace] [--dry_run [DRY_RUN]] [-d INPUT_NAME INPUT_DIM]
                          [-n] [-b BATCH] [-s SYMBOL_NAME VALUE]
                          [--dump_custom_io_config_template DUMP_CUSTOM_IO_CONFIG_TEMPLATE]
                          [--quantization_overrides QUANTIZATION_OVERRIDES] [--keep_quant_nodes]
                          [--disable_batchnorm_folding] [--expand_lstm_op_structure]
                          [--keep_disconnected_nodes]
                          [--apply_masked_softmax {compressed,uncompressed}]
                          [--packed_masked_softmax_inputs PACKED_MASKED_SOFTMAX_INPUTS [PACKED_MASKED_SOFTMAX_INPUTS ...]]
                          [--packed_max_seq PACKED_MAX_SEQ] [--input_list INPUT_LIST]
                          [--param_quantizer PARAM_QUANTIZER] [--act_quantizer ACT_QUANTIZER]
                          [--algorithms ALGORITHMS [ALGORITHMS ...]] [--bias_bitwidth BIAS_BITWIDTH]
                          [--bias_bw BIAS_BITWIDTH] [--act_bitwidth ACT_BITWIDTH]
                          [--act_bw ACT_BITWIDTH] [--weights_bitwidth WEIGHTS_BITWIDTH]
                          [--weight_bw WEIGHTS_BITWIDTH] [--ignore_encodings]
                          [--use_per_channel_quantization] [--use_per_row_quantization]
                          [--float_fallback] [--use_native_input_files] [--use_native_dtype]
                          [--use_native_output_files] [--disable_relu_squashing]
                          [--restrict_quantization_steps ENCODING_MIN, ENCODING_MAX]
                          [--pack_4_bit_weights] [--keep_weights_quantized]
                          [--act_quantizer_calibration ACT_QUANTIZER_CALIBRATION]
                          [--param_quantizer_calibration PARAM_QUANTIZER_CALIBRATION]
                          [--act_quantizer_schema ACT_QUANTIZER_SCHEMA]
                          [--param_quantizer_schema PARAM_QUANTIZER_SCHEMA]
                          [--percentile_calibration_value PERCENTILE_CALIBRATION_VALUE]
                          [--dump_qairt_quantizer_command DUMP_QAIRT_QUANTIZER_COMMAND]
                          --input_network INPUT_NETWORK [--debug [DEBUG]] [-o OUTPUT_PATH]
                          [--copyright_file COPYRIGHT_FILE] [--float_bitwidth FLOAT_BITWIDTH]
                          [--float_bw FLOAT_BW] [--float_bias_bitwidth FLOAT_BIAS_BITWIDTH]
                          [--float_bias_bw FLOAT_BIAS_BW] [--overwrite_model_prefix]
                          [--exclude_named_tensors] [--model_version MODEL_VERSION]
                          [--op_package_lib OP_PACKAGE_LIB]
                          [--converter_op_package_lib CONVERTER_OP_PACKAGE_LIB]
                          [-p PACKAGE_NAME | --op_package_config CUSTOM_OP_CONFIG_PATHS [CUSTOM_OP_CONFIG_PATHS ...]]
                          [-h] [--arch_checker] [--validate_models]

Script to convert ONNX model into QNN

required arguments:
  --input_network INPUT_NETWORK, -i INPUT_NETWORK
                        Path to the source framework model.

optional arguments:
  --out_node OUT_NAMES, --out_name OUT_NAMES
                        Name of the graph's output Tensor Names. Multiple output names should be
                        provided separately like:
                            --out_name out_1 --out_name out_2
  --input_type INPUT_NAME INPUT_TYPE, -t INPUT_NAME INPUT_TYPE
                        Type of data expected by each input op/layer. Type for each input is
                        |default| if not specified. For example: "data" image.Note that the quotes
                        should always be included in order to handle special characters, spaces,etc.
                        For multiple inputs specify multiple --input_type on the command line.
                        Eg:
                           --input_type "data1" image --input_type "data2" opaque
                        These options get used by DSP runtime and following descriptions state how
                        input will be handled for each option.
                        Image:
                        Input is float between 0-255 and the input's mean is 0.0f and the input's
                        max is 255.0f. We will cast the float to uint8ts and pass the uint8ts to the
                        DSP.
                        Default:
                        Pass the input as floats to the dsp directly and the DSP will quantize it.
                        Opaque:
                        Assumes input is float because the consumer layer(i.e next layer) requires
                        it as float, therefore it won't be quantized.
                        Choices supported:
                           image
                           default
                           opaque
  --input_dtype INPUT_NAME INPUT_DTYPE
                        The names and datatype of the network input layers specified in the format
                        [input_name datatype], for example:
                            'data' 'float32'
                        Default is float32 if not specified
                        Note that the quotes should always be included in order to handlespecial
                        characters, spaces, etc.
                        For multiple inputs specify multiple --input_dtype on the command line like:
                            --input_dtype 'data1' 'float32' --input_dtype 'data2' 'float32'
  --input_encoding  ...], -e  ...]
                        Usage:     --input_encoding "INPUT_NAME" INPUT_ENCODING_IN
                        [INPUT_ENCODING_OUT]
                        Input encoding of the network inputs. Default is bgr.
                        e.g.
                           --input_encoding "data" rgba
                        Quotes must wrap the input node name to handle special characters,
                        spaces, etc. To specify encodings for multiple inputs, invoke
                        --input_encoding for each one.
                        e.g.
                            --input_encoding "data1" rgba --input_encoding "data2" other
                        Optionally, an output encoding may be specified for an input node by
                        providing a second encoding. The default output encoding is bgr.
                        e.g.
                            --input_encoding "data3" rgba rgb
                        Input encoding types:
                             image color encodings: bgr,rgb, nv21, nv12, ...
                            time_series: for inputs of rnn models;
                            other: not available above or is unknown.
                        Supported encodings:
                           bgr
                           rgb
                           rgba
                           argb32
                           nv21
                           nv12
                           time_series
                           other
  --input_layout INPUT_NAME INPUT_LAYOUT, -l INPUT_NAME INPUT_LAYOUT
                        Layout of each input tensor. If not specified, it will use the default
                        based on the Source Framework, shape of input and input encoding.
                        Accepted values are-
                            NCDHW, NDHWC, NCHW, NHWC, HWIO, OIHW, NFC, NCF, NTF, TNF, NF, NC, F,
                        NONTRIVIAL
                        N = Batch, C = Channels, D = Depth, H = Height, W = Width, F = Feature, T =
                        Time
                        NDHWC/NCDHW used for 5d inputs
                        NHWC/NCHW used for 4d image-like inputs
                        NFC/NCF used for inputs to Conv1D or other 1D ops
                        NTF/TNF used for inputs with time steps like the ones used for LSTM op
                        NF used for 2D inputs, like the inputs to Dense/FullyConnected layers
                        NC used for 2D inputs with 1 for batch and other for Channels (rarely used)
                        F used for 1D inputs, e.g. Bias tensor
                        NONTRIVIAL for everything elseFor multiple inputs specify multiple
                        --input_layout on the command line.
                        Eg:
                            --input_layout "data1" NCHW --input_layout "data2" NCHW
  --custom_io CUSTOM_IO
                        Use this option to specify a yaml file for custom IO.
  --preserve_io [PRESERVE_IO ...]
                        Use this option to preserve IO layout and datatype. The different ways of
                        using this option are as follows:
                            --preserve_io layout <space separated list of names of inputs and
                        outputs of the graph>
                            --preserve_io datatype <space separated list of names of inputs and
                        outputs of the graph>
                        In this case, user should also specify the string - layout or datatype in
                        the command to indicate that converter needs to
                        preserve the layout or datatype. e.g.
                           --preserve_io layout input1 input2 output1
                           --preserve_io datatype input1 input2 output1
                        Optionally, the user may choose to preserve the layout and/or datatype for
                        all the inputs and outputs of the graph.
                        This can be done in the following two ways:
                            --preserve_io layout
                            --preserve_io datatype
                        Additionally, the user may choose to preserve both layout and datatypes for
                        all IO tensors by just passing the option as follows:
                            --preserve_io
                        Note: Only one of the above usages are allowed at a time.
                        Note: --custom_io gets higher precedence than --preserve_io.
  --dump_qairt_io_config_yaml DUMP_QAIRT_IO_CONFIG_YAML
                        Use this option to dump a yaml file which contains the equivalent I/O
                        configurations of QAIRT Converter along with the QAIRT Converter Command and
                        can be passed to QAIRT Converter using the option --io_config.
  --enable_framework_trace
                        Use this option to enable converter to trace the op/tensor change
                        information.
                        Currently framework op trace is supported only for ONNX converter.
  --dry_run [DRY_RUN]   Evaluates the model without actually converting any ops, and returns
                        unsupported ops/attributes as well as unused inputs and/or outputs if any.
                        Leave empty or specify "info" to see dry run as a table, or specify "debug"
                        to show more detailed messages only"
  -d INPUT_NAME INPUT_DIM, --input_dim INPUT_NAME INPUT_DIM
                        The name and dimension of all the input buffers to the network specified in
                        the format [input_name comma-separated-dimensions],
                        for example: 'data' 1,224,224,3.
                        Note that the quotes should always be included in order to handle special
                        characters, spaces, etc.
                        NOTE: This feature works only with Onnx 1.6.0 and above
  -n, --no_simplification
                        Do not attempt to simplify the model automatically. This may prevent some
                        models from properly converting
                        when sequences of unsupported static operations are present.
  -b BATCH, --batch BATCH
                        The batch dimension override. This will take the first dimension of all
                        inputs and treat it as a batch dim, overriding it with the value provided
                        here. For example:
                        --batch 6
                        will result in a shape change from [1,3,224,224] to [6,3,224,224].
                        If there are inputs without batch dim this should not be used and each input
                        should be overridden independently using -d option for input dimension
                        overrides.
  -s SYMBOL_NAME VALUE, --define_symbol SYMBOL_NAME VALUE
                        This option allows overriding specific input dimension symbols. For instance
                        you might see input shapes specified with variables such as :
                        data: [1,3,height,width]
                        To override these simply pass the option as:
                        --define_symbol height 224 --define_symbol width 448
                        which results in dimensions that look like:
                        data: [1,3,224,448]
  --dump_custom_io_config_template DUMP_CUSTOM_IO_CONFIG_TEMPLATE
                        Dumps the yaml template for Custom I/O configuration. This file canbe edited
                        as per the custom requirements and passed using the option --custom_ioUse
                        this option to specify a yaml file to which the custom IO config template is
                        dumped.
  --disable_batchnorm_folding
  --expand_lstm_op_structure
                        Enables optimization that breaks the LSTM op to equivalent math ops
  --keep_disconnected_nodes
                        Disable Optimization that removes Ops not connected to the main graph.
                        This optimization uses output names provided over commandline OR
                        inputs/outputs extracted from the Source model to determine the main graph
  --debug [DEBUG]       Run the converter in debug mode.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path where the converted Output model should be saved.If not specified, the
                        converter model will be written to a file with same name as the input model
  --copyright_file COPYRIGHT_FILE
                        Path to copyright file. If provided, the content of the file will be added
                        to the output model.
  --float_bitwidth FLOAT_BITWIDTH
                        Use the --float_bitwidth option to convert the graph to the specified float
                        bitwidth, either 32 (default) or 16.
  --float_bw FLOAT_BW   Note: --float_bw is deprecated, use --float_bitwidth.
  --float_bias_bitwidth FLOAT_BIAS_BITWIDTH
                        Use the --float_bias_bitwidth option to select the bitwidth to use for float
                        bias tensor
  --float_bias_bw FLOAT_BIAS_BW
                        Note: --float_bias_bw is deprecated, use --float_bias_bitwidth.
  --overwrite_model_prefix
                        If option passed, model generator will use the output path name to use as
                        model prefix to name functions in <qnn_model_name>.cpp. (Useful for running
                        multiple models at once) eg: ModelName_composeGraphs. Default is to use
                        generic "QnnModel_".
  --exclude_named_tensors
                        Remove using source framework tensorNames; instead use a counter for naming
                        tensors. Note: This can potentially help to reduce  the final model library
                        that will be generated(Recommended for deploying model). Default is False.
  --model_version MODEL_VERSION
                        User-defined ASCII string to identify the model, only first 64 bytes will be
                        stored
  -h, --help            show this help message and exit
  --validate_models     Validate the original onnx model against optimized onnx model.
                        Constant inputs with all value 1s will be generated and will be used
                        by both models and their outputs are checked against each other.
                        The {'option_strings': ['--validate_models'], 'dest': 'validate_models',
                        'nargs': 0, 'const': True, 'default': False, 'type': None, 'choices': None,
                        'required': False, 'help': 'Validate the original onnx model against
                        optimized onnx model.\nConstant inputs with all value 1s will be generated
                        and will be used \nby both models and their outputs are checked against each
                        other.\nThe % average error and 90th percentile of output differences will
                        be calculated for this.\nNote: Usage of this flag will incur extra time due
                        to inference of the models.', 'metavar': None, 'container':
                        <argparse._ArgumentGroup object at 0x7f24df221c90>, 'prog': 'qnn-onnx-
                        converter'}verage error and 90th percentile of output differences will be
                        calculated for this.
                        Note: Usage of this flag will incur extra time due to inference of the
                        models.

Custom Op Package Options:
  --op_package_lib OP_PACKAGE_LIB, -opl OP_PACKAGE_LIB
                        Use this argument to pass an op package library for quantization. Must be in
                        the form <op_package_lib_path:interfaceProviderName> and be separated by a
                        comma for multiple package libs
  --converter_op_package_lib CONVERTER_OP_PACKAGE_LIB, -cpl CONVERTER_OP_PACKAGE_LIB
                        Absolute path to converter op package library compiled by the OpPackage
                        generator. Must be separated by a comma for multiple package libraries.
                        Note: Order of converter op package libraries must follow the order of xmls.
                        Ex1: --converter_op_package_lib absolute_path_to/libExample.so
                        Ex2: -cpl absolute_path_to/libExample1.so,absolute_path_to/libExample2.so
  -p PACKAGE_NAME, --package_name PACKAGE_NAME
                        A global package name to be used for each node in the Model.cpp file.
                        Defaults to Qnn header defined package name
  --op_package_config CUSTOM_OP_CONFIG_PATHS [CUSTOM_OP_CONFIG_PATHS ...], -opc CUSTOM_OP_CONFIG_PATHS [CUSTOM_OP_CONFIG_PATHS ...]
                        Path to a Qnn Op Package XML configuration file that contains user defined
                        custom operations.

Quantizer Options:
  --quantization_overrides QUANTIZATION_OVERRIDES
                        Use this option to specify a json file with parameters to use for
                        quantization. These will override any quantization data carried from
                        conversion (eg TF fake quantization) or calculated during the normal
                        quantization process. Format defined as per AIMET specification.
  --keep_quant_nodes    Use this option to keep activation quantization nodes in the graph rather
                        than stripping them.
  --input_list INPUT_LIST
                        Path to a file specifying the input data. This file should be a plain text
                        file, containing one or more absolute file paths per line. Each path is
                        expected to point to a binary file containing one input in the "raw" format,
                        ready to be consumed by the quantizer without any further preprocessing.
                        Multiple files per line separated by spaces indicate multiple inputs to the
                        network. See documentation for more details. Must be specified for
                        quantization. All subsequent quantization options are ignored when this is
                        not provided.
  --param_quantizer PARAM_QUANTIZER
                        Optional parameter to indicate the weight/bias quantizer to use. Must be
                        followed by one of the following options:
                        "tf": Uses the real min/max of the data and specified bitwidth (default).
                        "enhanced": Uses an algorithm useful for quantizing models with long tails
                        present in the weight distribution.
                        "adjusted": Note: "adjusted" mode is deprecated.
                        "symmetric": Ensures min and max have the same absolute values about zero.
                        Data will be stored as int#_t data such that the offset is always 0.Note:
                        Legacy option --param_quantizer will be deprecated, use
                        --param_quantizer_calibration instead
  --act_quantizer ACT_QUANTIZER
                        Optional parameter to indicate the activation quantizer to use. Must be
                        followed by one of the following options:
                        "tf": Uses the real min/max of the data and specified bitwidth (default).
                        "enhanced": Uses an algorithm useful for quantizing models with long tails
                        present in the weight distribution.
                        "adjusted": Note: "adjusted" mode is deprecated.
                        "symmetric": Ensures min and max have the same absolute values about zero.
                        Data will be stored as int#_t data such that the offset is always 0.Note:
                        Legacy option --act_quantizer will be deprecated, use
                        --act_quantizer_calibration instead
  --algorithms ALGORITHMS [ALGORITHMS ...]
                        Use this option to enable new optimization algorithms. Usage is:
                        --algorithms <algo_name1> ... The available optimization algorithms are:
                        "cle" - Cross layer equalization includes a number of methods for equalizing
                        weights and biases across layers in order to rectify imbalances that cause
                        quantization errors.
  --bias_bitwidth BIAS_BITWIDTH
                        Use the --bias_bitwidth option to select the bitwidth to use when quantizing
                        the biases, either 8 (default) or 32.
  --bias_bw BIAS_BITWIDTH
                        Note: --bias_bw is deprecated, use --bias_bitwidth.
  --act_bitwidth ACT_BITWIDTH
                        Use the --act_bitwidth option to select the bitwidth to use when quantizing
                        the activations, either 8 (default) or 16.
  --act_bw ACT_BITWIDTH
                        Note: --act_bw is deprecated, use --act_bitwidth.
  --weights_bitwidth WEIGHTS_BITWIDTH
                        Use the --weights_bitwidth option to select the bitwidth to use when
                        quantizing the weights, either 4 or 8 (default).
  --weight_bw WEIGHTS_BITWIDTH
                        Note: --weight_bw is deprecated, use --weights_bitwidth.
  --ignore_encodings    Use only quantizer generated encodings, ignoring any user or model provided
                        encodings.
                        Note: Cannot use --ignore_encodings with --quantization_overrides
  --use_per_channel_quantization
                        Use this option to enable per-channel quantization for convolution-based op
                        weights.
                        Note: This will replace built-in model QAT encodings when used for a given
                        weight.
  --use_per_row_quantization
                        Use this option to enable rowwise quantization of Matmul and FullyConnected
                        ops.
  --float_fallback      Use this option to enable fallback to floating point (FP) instead of fixed
                        point.
                        This option can be paired with --float_bitwidth to indicate the bitwidth for
                        FP (by default 32).
                        If this option is enabled, then input list must not be provided and
                        --ignore_encodings must not be provided.
                        The external quantization encodings (encoding file/FakeQuant encodings)
                        might be missing quantization parameters for some interim tensors.
                        First it will try to fill the gaps by propagating across math-invariant
                        functions. If the quantization params are still missing,
                        then it will apply fallback to nodes to floating point.
  --use_native_input_files
                        Boolean flag to indicate how to read input files:
                        1. float (default): reads inputs as floats and quantizes if necessary based
                        on quantization parameters in the model.
                        2. native:          reads inputs assuming the data type to be native to the
                        model. For ex., uint8_t.
  --use_native_dtype    Note: This option is deprecated, use --use_native_input_files option in
                        future.
                        Boolean flag to indicate how to read input files:
                        1. float (default): reads inputs as floats and quantizes if necessary based
                        on quantization parameters in the model.
                        2. native:          reads inputs assuming the data type to be native to the
                        model. For ex., uint8_t.
  --use_native_output_files
                        Use this option to indicate the data type of the output files
                        1. float (default): output the file as floats.
                        2. native:          outputs the file that is native to the model. For ex.,
                        uint8_t.
  --disable_relu_squashing
                        Disables squashing of Relu against Convolution based ops for quantized
                        models
  --restrict_quantization_steps ENCODING_MIN, ENCODING_MAX
                        Specifies the number of steps to use for computing quantization encodings
                        such that scale = (max - min) / number of quantization steps.
                        The option should be passed as a space separated pair of hexadecimal string
                        minimum and maximum valuesi.e. --restrict_quantization_steps "MIN MAX".
                         Please note that this is a hexadecimal string literal and not a signed
                        integer, to supply a negative value an explicit minus sign is required.
                        E.g.--restrict_quantization_steps "-0x80 0x7F" indicates an example 8 bit
                        range,
                            --restrict_quantization_steps "-0x8000 0x7F7F" indicates an example 16
                        bit range.
                        This argument is required for 16-bit Matmul operations.
  --pack_4_bit_weights  Store 4-bit quantized weights in packed format in a single byte i.e. two
                        4-bit quantized tensors can be stored in one byte
  --keep_weights_quantized
                        Use this option to keep the weights quantized even when the output of the op
                        is in floating point. Bias will be converted to floating point as per the
                        output of the op. Required to enable wFxp_actFP configurations according to
                        the provided bitwidth for weights and activations
                        Note: These modes are not supported by all runtimes. Please check
                        corresponding Backend OpDef supplement if these are supported
  --act_quantizer_calibration ACT_QUANTIZER_CALIBRATION
                        Specify which quantization calibration method to use for activations
                        supported values: min-max (default), sqnr, entropy, mse, percentile
                        This option can be paired with --act_quantizer_schema to override the
                        quantization
                        schema to use for activations otherwise default schema(asymmetric) will be
                        used
  --param_quantizer_calibration PARAM_QUANTIZER_CALIBRATION
                        Specify which quantization calibration method to use for parameters
                        supported values: min-max (default), sqnr, entropy, mse, percentile
                        This option can be paired with --param_quantizer_schema to override the
                        quantization
                        schema to use for parameters otherwise default schema(asymmetric) will be
                        used
  --act_quantizer_schema ACT_QUANTIZER_SCHEMA
                        Specify which quantization schema to use for activations
                        supported values: asymmetric (default), symmetric, unsignedsymmetric
                        This option cannot be used with legacy quantizer option --act_quantizer
  --param_quantizer_schema PARAM_QUANTIZER_SCHEMA
                        Specify which quantization schema to use for parameters
                        supported values: asymmetric (default), symmetric, unsignedsymmetric
                        This option cannot be used with legacy quantizer option --param_quantizer
  --percentile_calibration_value PERCENTILE_CALIBRATION_VALUE
                        Specify the percentile value to be used with Percentile calibration method
                        The specified float value must lie within 90 and 100, default: 99.99
  --dump_qairt_quantizer_command DUMP_QAIRT_QUANTIZER_COMMAND
                        Use this option to dump a file which contains the equivalent Commandline
                        input for QAIRT Quantizer

Masked Softmax Optimization Options:
  --apply_masked_softmax {compressed,uncompressed}
                        This flag enables the pass that creates a MaskedSoftmax Op and
                        rewrites the graph to include this Op. MaskedSoftmax Op may not
                        be supported by all the QNN backends. Please check the
                        supplemental backend XML for the targeted backend.
                        This argument takes a string parameter input that selects
                        the mode of MaskedSoftmax Op.
                        'compressed' value rewrites the graph with the compressed version of
                        MaskedSoftmax Op.
                        'uncompressed' value rewrites the graph with the uncompressed version of
                        MaskedSoftmax Op.
  --packed_masked_softmax_inputs PACKED_MASKED_SOFTMAX_INPUTS [PACKED_MASKED_SOFTMAX_INPUTS ...]
                        Mention the input ids tensor name which will be packed in the single
                        inference.
                        This is applicable only for Compressed MaskedSoftmax Op.
                        This will create a new input to the graph named 'position_ids'
                        with same shape as the provided input name in this flag.
                        During runtime, this input shall be provided with the token
                        locations for individual sequences so that the same will be
                        internally passed to positional embedding layer.
                        E.g. If 2 sequences of length 20 and 30 are packed together
                        in single batch of 64 tokens then this new input 'position_ids' should have
                        value [0, 1, ..., 19, 0, 1, ..., 29, 0, 0, 0, ..., 0]
                        Usage: --packed_masked_softmax input_ids
                        Packed model will enable the user to pack multiple sequences into
                        single batch of inference.
  --packed_max_seq PACKED_MAX_SEQ
                        Number of sequences packed in the single input ids and
                        single attention mask inputs. Applicable only for
                        Compressed MaskedSoftmax Op.

Architecture Checker Options(Experimental):
  --arch_checker        Pass this option to enable architecture checker tool.
                        This is an experimental option for models that are intended to run on HTP
                        backend.

Note: Only one of: {'package_name', 'op_package_config'} can be specified Note: Only one of:
{'package_name', 'op_package_config'} can be specified