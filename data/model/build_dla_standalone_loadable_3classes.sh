#!/bin/bash
# Build the DLA loadables for the 3-class custom model (v1 precision set):
#   - FP16 loadable from the FP32 trimmed ONNX (yolov5_3clases_fp32_trimmed.onnx,
#     exported with: qat.py export best.pt --size=672 --dynamic --noanchor --noqadd)
#   - INT8 loadable from the QAT model translated by qdq_translator
#     (yolov5_3clases_qat_noqdq.onnx + calib cache), only the 3 head convs
#     forced to FP16 (faster; see *_v2.sh for the higher-accuracy variant)
# Deploying requires the matching app build: make NUM_CLASSES=3
mkdir -p data/loadable
TRTEXEC=/usr/src/tensorrt/bin/trtexec
# TensorRT 10.x refuses the TRT-8600 cache header ("Calibration table does not
# match calibrator algorithm type") and then fails trying to recalibrate with
# no data. Rewrite the header to the locally installed TRT version.
TRT_VERSION=$(${TRTEXEC} --version 2>&1 | grep -oE 'TensorRT v[0-9]+' | head -1 | grep -oE '[0-9]+')
CALIB_CACHE=data/loadable/qat2ptq_3classes.cache
sed "1s/^TRT-[0-9]*/TRT-${TRT_VERSION}/" data/model/yolov5_3clases_qat_precision_config_calib.cache > ${CALIB_CACHE}

# FP16 loadable (input/output fp16:chw16)
${TRTEXEC} --onnx=data/model/yolov5_3clases_fp32_trimmed.onnx --fp16 \
    --saveEngine=data/loadable/yolov5_3classes.fp16.fp16chw16in.fp16chw16out.standalone.bin \
    --inputIOFormats=fp16:chw16 --outputIOFormats=fp16:chw16 --buildDLAStandalone --useDLACore=0

# INT8 loadable, v1 precision set (3 head convs in FP16)
${TRTEXEC} --minShapes=images:1x3x672x672 --maxShapes=images:1x3x672x672 --optShapes=images:1x3x672x672 --shapes=images:1x3x672x672 \
    --onnx=data/model/yolov5_3clases_qat_noqdq.onnx --useDLACore=0 --buildDLAStandalone \
    --saveEngine=data/loadable/yolov5_3classes.int8.int8hwc4in.fp16chw16out.standalone.bin \
    --inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16 --int8 --fp16 \
    --calib=${CALIB_CACHE} --precisionConstraints=obey \
    --layerPrecisions="/model.24/m.0/Conv":fp16,"/model.24/m.1/Conv":fp16,"/model.24/m.2/Conv":fp16
