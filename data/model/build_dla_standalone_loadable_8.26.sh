#!/bin/bash
# Build the DLA INT8 loadable from the 8.26 QAT fine-tuned model
# (yolov5_trimmed_qat_8.26.onnx translated by qdq_translator).
# Derived from build_dla_standalone_loadable.sh — INT8 part only; the FP16
# loadable (yolov5s_trimmed_reshape_tranpose.onnx) is unaffected by retraining.
mkdir -p data/loadable
TRTEXEC=/usr/src/tensorrt/bin/trtexec
# TensorRT 10.x refuses the TRT-8600 cache header ("Calibration table does not
# match calibrator algorithm type") and then fails trying to recalibrate with
# no data. Rewrite the header to the locally installed TRT version.
TRT_VERSION=$(${TRTEXEC} --version 2>&1 | grep -oE 'TensorRT v[0-9]+' | head -1 | grep -oE '[0-9]+')
CALIB_CACHE=data/loadable/qat2ptq_8.26.cache
sed "1s/^TRT-[0-9]*/TRT-${TRT_VERSION}/" data/model/yolov5_trimmed_qat_8.26.cache > ${CALIB_CACHE}
${TRTEXEC} --minShapes=images:1x3x672x672 --maxShapes=images:1x3x672x672 --optShapes=images:1x3x672x672 --shapes=images:1x3x672x672 \
    --onnx=data/model/yolov5_trimmed_qat_8.26_noqdq.onnx --useDLACore=0 --buildDLAStandalone \
    --saveEngine=data/loadable/yolov5_8.26.int8.int8hwc4in.fp16chw16out.standalone.bin \
    --inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16 --int8 --fp16 \
    --calib=${CALIB_CACHE} --precisionConstraints=obey \
    --layerPrecisions="/model.24/m.0/Conv":fp16,"/model.24/m.1/Conv":fp16,"/model.24/m.2/Conv":fp16
